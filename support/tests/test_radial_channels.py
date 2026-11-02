from dataclasses import replace
import math
import unittest
import numpy as np

from mri_magnet.geometry import Design, RingSpec, planar_pocket_clearance
from mri_magnet.manufacturing import audit_manufacturing, make_manufacturing_seed, radial_packing_bounds


PROCESS = {"enabled": True, "layout_mode": "radial_channels",
           "pocket_clearance_per_side_mm": .2, "pocket_web_mm": 1.2,
           "inner_wall_mm": 1.2, "outer_wall_mm": 1.2, "base_thickness_mm": 1.6,
           "sectors_per_layer": 4, "printer_bed_mm": [220,220,250],
           "bed_margin_per_side_mm": 4, "max_magnet_count": 1736,
           "min_axial_module_gap_mm": 0}
ENVELOPE = {"bore_radius_mm": 120., "outer_radius_mm": math.hypot(192,6),
            "half_length_mm": 253., "min_gap_mm": 0., "tolerance_mm": 1e-6,
            "preserve_exact_bore": True}


def ring(slot, radius, *, phase=0., count=28, layer=0, z=20.):
    return RingSpec(f"R{layer+1:02d}-L{slot+1}",layer,slot,radius,z,count,"out",phase)


class RadialChannelTests(unittest.TestCase):
    def test_contact_allowed_without_internal_pocket_allowance_or_web(self):
        design=Design((ring(0,130.),ring(1,142.)))
        audit=audit_manufacturing(design,PROCESS,ENVELOPE)
        self.assertTrue(audit['valid'],audit['violations'])
        self.assertEqual(audit['nominal_geometry']['overlap_pairs'],0)
        self.assertEqual(audit['packing']['required_radial_center_pitch_mm'],12.)
        self.assertTrue(audit['packing']['same_column_contact_allowed'])
        self.assertEqual(audit['slot_clearance']['same_column_pairs_exempted'],28)
        self.assertEqual(audit['packing']['radial_end_clearance_per_side_mm'],.2)

    def test_old_independent_pockets_still_reject_contact(self):
        design=Design((ring(0,130.),ring(1,142.)))
        old=dict(PROCESS,layout_mode='independent_pockets')
        audit=audit_manufacturing(design,old,ENVELOPE)
        self.assertFalse(audit['valid'])
        self.assertFalse(audit['checks']['pocket_web_obb_clearance'])
        self.assertAlmostEqual(audit['packing']['required_radial_center_pitch_mm'],13.6)
        self.assertEqual(audit['slot_clearance']['same_column_pairs_exempted'],0)

    def test_real_3d_overlap_is_never_exempted(self):
        design=Design((ring(0,130.),ring(1,141.999)))
        audit=audit_manufacturing(design,PROCESS,ENVELOPE)
        self.assertFalse(audit['valid'])
        self.assertFalse(audit['checks']['nominal_magnet_geometry'])
        self.assertFalse(audit['checks']['radial_magnet_separation'])
        self.assertGreater(audit['nominal_geometry']['overlap_pairs'],0)

    def test_channel_end_walls_remain_enforced(self):
        packing=radial_packing_bounds(12,PROCESS,ENVELOPE)
        lo,hi=packing['minimum_magnet_center_radius_mm'],packing['maximum_magnet_center_radius_mm']
        exact=Design((ring(0,lo),ring(1,hi)))
        audit=audit_manufacturing(exact,PROCESS,ENVELOPE)
        self.assertTrue(audit['valid'],audit['violations'])
        self.assertFalse(audit_manufacturing(Design((ring(0,lo-.001),)),PROCESS,ENVELOPE)['checks']['pocket_inner_wall'])
        self.assertFalse(audit_manufacturing(Design((ring(0,hi+.001),)),PROCESS,ENVELOPE)['checks']['pocket_outer_wall'])
        # Six 12mm magnets still cannot fit after preserving both channel ends.
        self.assertEqual(packing['maximum_radial_rings'],5)

    def test_nominal_optional_gap_is_preserved_explicitly(self):
        envelope=dict(ENVELOPE,min_gap_mm=.3)
        design=Design((ring(0,130.),ring(1,142.)))
        audit=audit_manufacturing(design,PROCESS,envelope)
        self.assertFalse(audit['checks']['radial_magnet_separation'])
        self.assertEqual(audit['packing']['required_radial_center_pitch_mm'],12.3)
        self.assertFalse(audit['packing']['same_column_contact_allowed'])

    def test_same_array_index_with_different_azimuth_is_not_same_column(self):
        design=Design((ring(0,130.),ring(1,142.,phase=.01)))
        proxy=planar_pocket_clearance(design,13.6,allow_same_column_contact=True)
        self.assertEqual(proxy['same_column_pairs_exempted'],0)
        self.assertFalse(proxy['valid'])
        audit=audit_manufacturing(design,PROCESS,ENVELOPE)
        self.assertFalse(audit['checks']['shared_column_phase'])

    def test_different_counts_match_only_genuinely_identical_periodic_angles(self):
        design=Design((ring(0,130.,count=28),ring(1,142.,count=32)))
        proxy=planar_pocket_clearance(design,13.6,allow_same_column_contact=True)
        # 28 and 32 grids at phase zero coincide at just four radial rays.
        self.assertEqual(proxy['same_column_pairs_exempted'],4)
        self.assertFalse(proxy['valid'])
        self.assertFalse(audit_manufacturing(design,PROCESS,ENVELOPE)['checks']['shared_azimuth_columns'])

    def test_phase_wrap_identifies_same_column(self):
        design=Design((ring(0,130.),ring(1,142.,phase=360.)))
        audit=audit_manufacturing(design,PROCESS,ENVELOPE)
        self.assertTrue(audit['valid'],audit['violations'])
        self.assertEqual(audit['slot_clearance']['same_column_pairs_exempted'],28)

    def test_column_walls_are_checked_without_charging_radial_web(self):
        # At radius 130, 64 columns' narrow pitch leaves less than the existing
        # 1.2mm column wall, even though actual nominal cubes remain separate.
        design=Design((ring(0,130.,count=64),))
        audit=audit_manufacturing(design,PROCESS,ENVELOPE)
        self.assertFalse(audit['checks']['pocket_web_obb_clearance'])
        self.assertEqual(audit['slot_clearance']['same_column_pairs_exempted'],0)

    def test_seed_keeps_nonuniform_source_shape_and_exact_contact(self):
        source=Design(tuple(ring(i,r) for i,r in enumerate([127.4,139.4,170.])))
        seed,report=make_manufacturing_seed(source,PROCESS,ENVELOPE)
        self.assertTrue(report['valid'],report['violations'])
        np.testing.assert_allclose([r.radius_mm for r in seed.rings],[127.4,139.4,170.])
        self.assertEqual(report['removed_source_slot_ids'],[])
        self.assertEqual(report['packing']['layout_mode'],'radial_channels')

    def test_invalid_layout_mode_rejected(self):
        with self.assertRaisesRegex(ValueError,'layout_mode'):
            radial_packing_bounds(12,dict(PROCESS,layout_mode='typo'),ENVELOPE)


if __name__=='__main__':
    unittest.main()

from dataclasses import replace
import math
import unittest
import numpy as np

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.manufacturing import audit_manufacturing, make_manufacturing_seed, max_radial_rings, radial_packing_bounds


CONFIG = {"enabled": True, "pocket_clearance_per_side_mm": .2, "pocket_web_mm": 1.2,
          "inner_wall_mm": 1.2, "outer_wall_mm": 1.2, "base_thickness_mm": 1.6,
          "sectors_per_layer": 4, "printer_bed_mm": [220, 220, 250],
          "bed_margin_per_side_mm": 4, "max_magnet_count": 1736, "min_axial_module_gap_mm": 0}
ENVELOPE = {"bore_radius_mm": 120., "outer_radius_mm": math.hypot(192, 6),
            "half_length_mm": 253., "preserve_exact_bore": True, "min_gap_mm": 0., "tolerance_mm": 1e-6}


def ring(ident, layer, slot, radius, z, count=28, polarity="out"):
    return RingSpec(ident, layer, slot, radius, z, count, polarity, -90 + 180 / count)


class ManufacturingTests(unittest.TestCase):
    def test_necessary_radial_bound_is_five_not_six(self):
        self.assertEqual(max_radial_rings(12, CONFIG, ENVELOPE), 5)
        d = Design(tuple(ring(f"L{i}", 0, i, 126 + 12 * i, 50) for i in range(6)))
        result = audit_manufacturing(d, CONFIG, ENVELOPE)
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["necessary_radial_packing_bound"])
        self.assertFalse(result["checks"]["pocket_web_obb_clearance"])

    def test_seed_retains_source_polarities_and_removes_traceable_slots(self):
        original = Design(tuple(
            ring(f"R{layer}-L{i}", layer, i, 126 + 12 * i, z, polarity="out" if z > 0 else "in")
            for layer, z in enumerate((247, -247)) for i in range(6)))
        seed, report = make_manufacturing_seed(original, CONFIG, ENVELOPE)
        self.assertTrue(report["valid"], report["violations"])
        self.assertEqual(len(seed.rings), 10)
        self.assertEqual(len(report["removed_source_slot_ids"]), 2)
        self.assertAlmostEqual(max(r.z_mm for r in seed.rings), 246.8)
        source = {r.id: r for r in original.rings}
        self.assertTrue(all(r.polarity == source[r.id].polarity for r in seed.rings))
        self.assertGreater(report["manufacturing_audit"]["nominal_geometry"]["bore_diameter_mm"], 240)

    def test_quarter_sector_fits_bed_instead_of_using_diagonal_chord(self):
        d = Design((ring("R1-L1", 0, 0, 150, 20),))
        result = audit_manufacturing(d, CONFIG, ENVELOPE)
        self.assertTrue(result["bed_fit"]["fits"])
        self.assertLess(max(result["bed_fit"]["reserved_bbox_mm"][:2]), 201)
        self.assertAlmostEqual(result["bed_fit"]["part_bbox_mm"][0], ENVELOPE["outer_radius_mm"])
        self.assertAlmostEqual(result["bed_fit"]["part_bbox_mm"][2], 14)

    def test_smaller_bed_and_non_divisible_columns_are_rejected(self):
        d = Design((ring("R1-L1", 0, 0, 150, 20, count=30),))
        result = audit_manufacturing(d, dict(CONFIG, printer_bed_mm=[180, 180, 250]), ENVELOPE)
        self.assertFalse(result["checks"]["columns_divisible_by_sectors"])
        self.assertFalse(result["checks"]["sector_fits_printer_bed"])

    def test_slot_not_nominal_magnet_controls_inner_and_outer_walls(self):
        nominal_ok = Design((ring("inner", 0, 0, 126, 20), ring("outer", 1, 0, 186, -20)))
        result = audit_manufacturing(nominal_ok, CONFIG, ENVELOPE)
        self.assertTrue(result["checks"]["nominal_magnet_geometry"])
        self.assertFalse(result["checks"]["pocket_inner_wall"])
        self.assertFalse(result["checks"]["pocket_outer_wall"])

    def test_baseplate_and_pocket_axial_envelope_are_checked(self):
        d = Design((ring("R1-L1", 0, 0, 150, 247),))
        result = audit_manufacturing(d, CONFIG, ENVELOPE)
        self.assertFalse(result["checks"]["module_axial_envelope"])
        close = Design((ring("a", 0, 0, 150, 5), ring("b", 1, 0, 150, -5)))
        self.assertFalse(audit_manufacturing(close, CONFIG, ENVELOPE)["checks"]["module_axial_separation"])

    def test_columns_may_differ_between_layers_but_are_shared_within_layer(self):
        design = Design((ring("a", 0, 0, 150, 20, count=28),
                         ring("b", 1, 0, 150, -20, count=32)))
        audit = audit_manufacturing(design, CONFIG, ENVELOPE)
        self.assertTrue(audit["valid"], audit["violations"])
        self.assertTrue(audit["checks"]["shared_azimuth_columns"])
        self.assertTrue(audit["checks"]["shared_column_phase"])
        same_layer = replace(design, rings=(design.rings[0], replace(design.rings[1], layer_index=0, radius_mm=175, z_mm=20)))
        self.assertFalse(audit_manufacturing(same_layer, CONFIG, ENVELOPE)["checks"]["shared_azimuth_columns"])

    def test_cached_bed_projection_does_not_share_mutable_audit_data(self):
        design = Design((ring("a", 0, 0, 150, 20),))
        first = audit_manufacturing(design, CONFIG, ENVELOPE)
        first["bed_fit"]["printer_bed_mm"][0] = -1
        first["bed_fit"]["reserved_bbox_mm"][0] = -1
        second = audit_manufacturing(design, CONFIG, ENVELOPE)
        self.assertEqual(second["bed_fit"]["printer_bed_mm"][0], 220)
        self.assertGreater(second["bed_fit"]["reserved_bbox_mm"][0], 200)

    def test_seed_projection_preserves_nonuniform_source_shape_and_preselected_slots(self):
        source = Design(tuple(ring('s'+str(i), 0, i, radius, 20) for i, radius in enumerate([127., 150., 175.])))
        seed, report = make_manufacturing_seed(source, CONFIG, ENVELOPE)
        self.assertTrue(report['valid'], report['violations'])
        np.testing.assert_allclose([r.radius_mm for r in seed.rings], [127.4, 150., 175.])
        self.assertEqual([r.id for r in seed.rings], [r.id for r in source.rings])
        self.assertEqual(report['removed_source_slot_ids'], [])
        self.assertAlmostEqual(report['radial_subset_projection'][0]['projection_squared_displacement_mm2'], .16)

    def test_thin_base_and_large_xy_web_do_not_false_reject_separate_layers(self):
        process = dict(CONFIG, base_thickness_mm=.8, pocket_web_mm=2.)
        design = Design((ring('a', 0, 0, 150, 20), ring('b', 1, 0, 150, 33.2)))
        audit = audit_manufacturing(design, process, ENVELOPE)
        self.assertTrue(audit['valid'], audit['violations'])
        self.assertTrue(audit['checks']['module_axial_separation'])
        self.assertTrue(audit['checks']['pocket_web_obb_clearance'])
        overlap = replace(design, rings=(design.rings[0], replace(design.rings[1], z_mm=33.1)))
        self.assertFalse(audit_manufacturing(overlap, process, ENVELOPE)['checks']['module_axial_separation'])

    def test_center_base_collision_is_repaired_by_minimum_axial_projection(self):
        source = Design((ring('a', 0, 0, 150, 5), ring('b', 1, 0, 150, -5)))
        seed, report = make_manufacturing_seed(source, CONFIG, ENVELOPE)
        self.assertTrue(report['valid'], report['violations'])
        np.testing.assert_allclose([r.z_mm for r in seed.rings], [7.8, -7.8])
        self.assertTrue(report['manufacturing_audit']['checks']['module_axial_separation'])

    def test_axial_projection_moves_only_out_of_bounds_end_layer(self):
        source = Design((ring('outer', 0, 0, 150, 247), ring('inner', 1, 0, 150, 19)))
        seed, report = make_manufacturing_seed(source, CONFIG, ENVELOPE)
        self.assertTrue(report['valid'], report['violations'])
        np.testing.assert_allclose([r.z_mm for r in seed.rings], [246.8, 19.])

    def test_disabled_manufacturing_keeps_nominal_geometry_and_budget(self):
        process = dict(CONFIG, enabled=False)
        source = Design(tuple(ring('s'+str(i), 0, i, 126+12*i, 247) for i in range(6)))
        seed, report = make_manufacturing_seed(source, process, ENVELOPE)
        self.assertTrue(report['valid'], report['violations'])
        self.assertEqual(seed.rings, source.rings)
        self.assertEqual(radial_packing_bounds(12, process, ENVELOPE)['maximum_radial_rings'], 6)
        self.assertFalse(audit_manufacturing(seed, process, ENVELOPE)['checks']['manufacturing_enabled'])
        budget_audit = audit_manufacturing(seed, dict(process, max_magnet_count=100), ENVELOPE)
        self.assertFalse(budget_audit['valid'])
        self.assertFalse(budget_audit['checks']['magnet_budget'])

    def test_candidate_footprint_can_fit_when_fixed_template_does_not(self):
        design = Design((ring('inner', 0, 0, 135, 20),))
        process = dict(CONFIG, printer_bed_mm=[160, 160, 250])
        self.assertFalse(audit_manufacturing(design, process, ENVELOPE)['bed_fit']['fits'])
        adaptive = audit_manufacturing(design, dict(process, bed_fit_mode='candidate_envelope'), ENVELOPE)
        self.assertTrue(adaptive['bed_fit']['fits'])
        self.assertLess(adaptive['bed_fit']['footprint_outer_radius_mm'], ENVELOPE['outer_radius_mm'])


if __name__ == "__main__":
    unittest.main()

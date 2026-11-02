"""Physical and provenance checks for constraint-only topology synthesis."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mri_magnet.de_novo import build_basis, synthesize, _pair_design
from mri_magnet.field import FieldEngine
from mri_magnet.geometry import Design
from mri_magnet.manufacturing import audit_manufacturing
from mri_magnet.sampling import sphere_samples


def configuration():
    return {
        'mode': 'de_novo', 'source_geometry': 'THIS_FILE_MUST_NOT_BE_OPENED.json',
        'target': {'geometry': 'sphere', 'diameter_mm': 20., 'min_field_mT': 15., 'max_ppm': 3000.},
        'envelope': {'protected_bore_diameter_mm': 240., 'max_outer_diameter_mm': 384.187454246,
                     'max_length_mm': 506.},
        'magnet': {'cube_mm': 12., 'remanence_T': 1.43, 'reference_mur': 1.05},
        'manufacturing': {'enabled': True, 'layout_mode': 'radial_channels',
            'pocket_clearance_per_side_mm': .2, 'pocket_web_mm': 1.2,
            'inner_wall_mm': 1.2, 'outer_wall_mm': 1.2, 'base_thickness_mm': 1.6,
            'printer_bed_mm': [220., 220., 250.], 'sectors_per_layer': 4,
            'bed_margin_per_side_mm': 4., 'max_magnet_count': 1736,
            'min_axial_module_gap_mm': 0., 'bed_fit_mode': 'fixed_envelope'},
        'de_novo': {'pair_counts': [5, 6, 7], 'column_counts': [24, 28, 32],
                    'seed_count': 3, 'milp_time_limit_seconds': 1., 'max_attempts': 6},
        'polarization_model': 'remanence', 'chunk_points': 64,
    }


class BasisTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.sample = sphere_samples(20., 8, 12)

    def test_unit_basis_matches_actual_superposition_and_opposite_polarity(self):
        engine = FieldEngine()
        basis = build_basis(self.config, self.sample, 5, 24, engine=engine)
        indices = [0, 10, 24, 42, 64]
        rings = []
        for i in indices:
            rings.extend(_pair_design(basis['material'], 5, 24, basis['basis'][i], False).rings)
        design = Design(tuple(rings), cube_mm=12., remanence_T=1.43, reference_mur=1.05)
        prediction = basis['A_T_per_T'][:, :, indices].sum(axis=2)*1.43
        actual = engine.field(design, self.sample['points_mm'])
        np.testing.assert_allclose(prediction, actual, atol=2e-16, rtol=1e-12)
        np.testing.assert_array_equal(basis['A_T_per_T'][:, :, 0], -basis['A_T_per_T'][:, :, 1])

    def test_cache_fingerprint_changes_with_coordinates_or_material_response(self):
        with tempfile.TemporaryDirectory() as folder:
            first = build_basis(self.config, self.sample, 5, 24, cache_dir=folder)
            again = build_basis(self.config, self.sample, 5, 24, cache_dir=folder)
            self.assertTrue(again['cache_hit'])
            np.testing.assert_array_equal(first['A_T_per_T'], again['A_T_per_T'])
            changed = dict(self.sample, points_mm=np.array(self.sample['points_mm'], copy=True))
            changed['points_mm'][1, 0] += .01
            other = build_basis(self.config, changed, 5, 24, cache_dir=folder)
            self.assertNotEqual(first['fingerprint'], other['fingerprint'])
            config = copy.deepcopy(self.config); config['polarization_model'] = 'isolated_cube_self_demag'
            demag = build_basis(config, self.sample, 5, 24, cache_dir=folder)
            self.assertNotEqual(first['fingerprint'], demag['fingerprint'])

    def test_invalid_axial_packing_rejected_before_field_work(self):
        self.config['envelope']['max_length_mm'] = 20
        with self.assertRaisesRegex(ValueError, 'axial'):
            build_basis(self.config, self.sample, 8, 24)


class SynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        cls.sample = sphere_samples(20., 16, 24)
        with patch('mri_magnet.geometry.load_source', side_effect=AssertionError('source access forbidden')):
            cls.result = synthesize(cls.config, cls.sample, log=lambda message: None)

    def test_no_source_file_and_heterogeneous_cluster(self):
        self.assertFalse(self.result['metadata']['source_layout_read'])
        self.assertEqual(len(self.result['seeds']), 3, self.result['metadata'])
        self.assertEqual(len({s['canonical_physical_sha256'] for s in self.result['seeds']}), 3)
        self.assertGreaterEqual(len({s['pair_count'] for s in self.result['seeds']}), 2)

    def test_every_seed_passes_real_audit_quantity_and_symmetry(self):
        from mri_magnet.de_novo import limits_from_config
        from mri_magnet.sampling import symmetry_profile
        for row in self.result['seeds']:
            design = Design.from_dict(row['design'])
            self.assertLessEqual(design.magnet_count, 1736)
            self.assertTrue(audit_manufacturing(design, self.config['manufacturing'],
                                              limits_from_config(self.config))['valid'])
            self.assertTrue(symmetry_profile(design)['verified'])
            self.assertGreaterEqual(row['metrics']['center_Bz_T'], .015-1e-8)

    def test_real_cuboid_seed_has_useful_small_dsv_homogeneity(self):
        # A 20 mm 3D sphere, not the old 20 mm equatorial disk. This checks
        # actual analytical fields and does not mirror the MILP objective.
        self.assertLess(min(s['metrics']['pp_norm_ppm'] for s in self.result['seeds']), 3000.)

    def test_impossible_inventory_is_diagnostic_and_never_returns_illegal_seed(self):
        config = copy.deepcopy(self.config)
        config['manufacturing']['max_magnet_count'] = 20
        config['de_novo']['max_attempts'] = 1
        result = synthesize(config, self.sample, log=lambda message: None)
        self.assertEqual(result['seeds'], [])
        self.assertEqual(result['metadata']['status'], 'no_feasible_seed_found')
        self.assertFalse(result['metadata']['global_optimum_proven'])


if __name__ == '__main__':
    unittest.main()

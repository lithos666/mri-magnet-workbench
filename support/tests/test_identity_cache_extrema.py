"""Physical equivalence, prepared-field reuse and independent extrema oracles."""
from dataclasses import replace
import unittest
import numpy as np

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.identity import (canonical_phase_deg, canonical_physical_sha256,
                                 raw_design_sha256)
from mri_magnet.field import FieldEngine
from mri_magnet.metrics import field_metrics
from mri_magnet.sampling import sphere_samples, symmetry_profile, fundamental_sphere_samples
from mri_magnet.validation import refine_sphere_extrema


class IdentityCacheTests(unittest.TestCase):
    def setUp(self):
        self.design = Design((RingSpec('one', 0, 0, 50., 10., 24, 'out', -82.5),
                              RingSpec('two', 13, 1, 65., -10., 24, 'in', -82.5)))

    def test_physical_equivalence_excludes_labels_and_wraps_phase(self):
        other = replace(self.design, name='renamed', source_sha256='another-file',
            rings=tuple(replace(r, id='new_'+r.id, layer_index=77-r.layer_index,
                                slot_index=99, phase_deg=r.phase_deg+360.)
                        for r in reversed(self.design.rings)))
        self.assertNotEqual(raw_design_sha256(self.design), raw_design_sha256(other))
        self.assertEqual(canonical_physical_sha256(self.design), canonical_physical_sha256(other))
        self.assertEqual(canonical_phase_deg(-1e-12, 28), 0.)
        self.assertEqual(canonical_phase_deg(360/28-1e-12, 28), 0.)
        for changes in ({'remanence_T': 1.44}, {'reference_mur': 1.06}, {'cube_mm': 12.1}):
            self.assertNotEqual(canonical_physical_sha256(self.design),
                                canonical_physical_sha256(replace(self.design, **changes)))
        self.assertNotEqual(canonical_physical_sha256(self.design),
            canonical_physical_sha256(self.design, 'isolated_cube_self_demag'))

    def test_warm_cache_and_equivalent_design_prepare_once(self):
        engine = FieldEngine(prepared_cache_size=2)
        points = np.array([[0, 0, 0], [3, -1, 6]])
        first = engine.field(self.design, points)
        second = engine.field(self.design, points)
        equivalent = replace(self.design, name='copy', rings=tuple(
            replace(r, id='copy_'+r.id, phase_deg=r.phase_deg+360/r.count)
            for r in self.design.rings))
        third = engine.field(equivalent, points)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)
        self.assertEqual(engine.statistics['preparations'], 1)
        self.assertEqual(engine.statistics['engine_calls'], 3)
        self.assertEqual(engine.statistics['field_points'], 6)
        self.assertEqual(engine.statistics['prepared_cache_hits'], 2)

    def test_cache_size_is_bounded_and_eviction_reprepares(self):
        engine = FieldEngine(prepared_cache_size=1)
        for design in (self.design, replace(self.design, remanence_T=1.5), self.design):
            engine.field(design, [[0, 0, 0]])
        self.assertEqual(engine.statistics['preparations'], 3)
        self.assertEqual(len(engine._prepared), 1)

    def test_common_symmetry_checks_geometry_magnetization_and_counts(self):
        symmetric = replace(self.design, rings=(self.design.rings[0],
            replace(self.design.rings[1], radius_mm=50.)))
        profile = symmetry_profile(symmetric, rotation_order=4, reflection_phase_deg=-90.)
        self.assertTrue(profile['verified'])
        self.assertFalse(symmetry_profile(symmetric, rotation_order=7,
                                         reflection_phase_deg=-90.)['verified'])
        broken = replace(symmetric, rings=(symmetric.rings[0],
            replace(symmetric.rings[1], polarity='out')))
        self.assertFalse(symmetry_profile(broken, rotation_order=4,
                                         reflection_phase_deg=-90.)['verified'])
        sample = fundamental_sphere_samples(180, 2048, 128, 11, rotation_order=4)
        p = sample['points_mm']
        self.assertTrue(np.all(p[:, 2] >= -1e-12))
        self.assertLessEqual(np.linalg.norm(p, axis=1).max(), 90+1e-10)
        # Uniform ball integration of invariant radial/z moments is analytic.
        weights = sample['weights']
        self.assertAlmostEqual(weights@np.sum(p*p, axis=1), 3*90**2/5, delta=5)
        self.assertAlmostEqual(weights@(p[:, 2]**2), 90**2/5, delta=5)
        self.assertAlmostEqual(float(weights.sum()), 1)


class LinearSourceFreeEngine:
    polarization_model = 'analytic_test_oracle'

    def __init__(self, matrix, offset):
        self.matrix = np.asarray(matrix)
        self.offset = np.asarray(offset)
        self.batch_sizes = []
        assert np.allclose(self.matrix, self.matrix.T)
        assert abs(np.trace(self.matrix)) < 1e-12

    def field(self, design, points):
        points = np.asarray(points).reshape(-1, 3)
        self.batch_sizes.append(len(points))
        return (points @ self.matrix.T + self.offset)*1e-3


class ExtremaOracleTests(unittest.TestCase):
    def check(self, engine):
        sample = sphere_samples(2., 128, 96, seed=11)
        B = engine.field(None, sample['points_mm'])
        metrics = field_metrics(B, sample['weights'], B[0])
        result = refine_sphere_extrema(None, engine, sample, metrics, B, starts=3)
        self.assertEqual(result['mean_norm_T'], metrics['mean_norm_T'])
        self.assertEqual(result['rms_norm_ppm'], metrics['rms_norm_ppm'])
        self.assertGreaterEqual(result['max_norm_T'], metrics['max_norm_T']-1e-14)
        self.assertLessEqual(result['min_norm_T'], metrics['min_norm_T']+1e-14)
        self.assertEqual(result['sampled_pp_Bz_ppm'], metrics['pp_Bz_ppm'])
        self.assertGreater(result['extrema_refinement']['field_points'],
                           result['extrema_refinement']['engine_calls'])
        self.assertTrue(any(size == 4 for size in engine.batch_sizes))
        self.assertFalse(result['extrema_refinement']['global_optimum_proven'])
        return result

    def test_offaxis_interior_null_and_separate_component_extrema(self):
        null = np.array([.23, -.17, .11])
        matrix = np.diag([1., 1., -2.])
        result = self.check(LinearSourceFreeEngine(matrix, -matrix@null))
        self.assertLess(result['min_norm_T'], 3e-8)
        np.testing.assert_allclose(result['extrema_refinement']['min_point_mm'], null, atol=3e-5)
        self.assertAlmostEqual(result['min_Bz_T'], -1.78e-3, delta=2e-8)
        self.assertAlmostEqual(result['max_Bz_T'], 2.22e-3, delta=2e-8)
        expected_transverse = (1+np.linalg.norm(null[:2]))*1e-3
        self.assertAlmostEqual(result['max_transverse_T'], expected_transverse, delta=2e-8)

    def test_tangent_chart_leaves_axis_and_crosses_angular_seam(self):
        direction = np.array([.08, -.06, 1.])
        direction /= np.linalg.norm(direction)
        matrix = np.eye(3)-3*np.outer(direction, direction)
        result = self.check(LinearSourceFreeEngine(matrix, .4*direction))
        self.assertAlmostEqual(result['max_norm_T'], 2.4e-3, delta=3e-8)
        np.testing.assert_allclose(result['extrema_refinement']['max_point_mm'], -direction, atol=2e-4)
        zamp = np.linalg.norm(matrix[2])
        self.assertAlmostEqual(result['min_Bz_T'], (.4*direction[2]-zamp)*1e-3, delta=2e-8)
        self.assertAlmostEqual(result['max_Bz_T'], (.4*direction[2]+zamp)*1e-3, delta=2e-8)


if __name__ == '__main__':
    unittest.main()

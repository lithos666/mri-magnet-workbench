"""Independent physical limits and symmetry oracles for optional recoil cells."""
from dataclasses import replace
import unittest

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

from mri_magnet.field import FieldEngine
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.material_response import (CoupledResponseEngine, coupled_response,
                                         preflight_response, _solve_polarization)


def fixture(counts=(6, 6), mur=1.05):
    return Design((RingSpec('upper', 0, 0, 50., 20., counts[0], 'out', 13.),
                   RingSpec('lower', 1, 0, 64., -20., counts[1], 'in', -8.)),
                  cube_mm=10., reference_mur=mur)


class MaterialResponseTests(unittest.TestCase):
    def test_mu_r_one_exactly_preserves_original_field_engine(self):
        design = fixture((4, 8), mur=1.)
        points = np.array([[0., 0., 0.], [3., -2., 5.]])
        response = coupled_response(design, points)
        np.testing.assert_array_equal(response['fields_T'], FieldEngine().field(design, points))
        self.assertEqual(response['metadata']['relative_recoil_residual'], 0.)
        self.assertFalse(response['metadata']['fem_validated'])

    def test_zero_neighbor_limit_is_analytic_isolated_cube_response(self):
        remanence = np.array([1.43, -.2, .7])
        actual, relative, absolute = _solve_polarization(np.zeros((3, 3)), remanence, .05)
        np.testing.assert_allclose(actual, remanence/(1.+.05/3.), atol=1e-15, rtol=0.)
        self.assertLess(relative, 1e-15)
        self.assertLess(absolute, 1e-15)

    def test_rotational_reduction_matches_full_independent_cell_solve(self):
        design = fixture((4, 8))
        reduced = CoupledResponseEngine(design)
        full = CoupledResponseEngine(design, {'symmetry_reduction': False})
        np.testing.assert_allclose(reduced.local_polarization_T, full.local_polarization_T, atol=1e-13, rtol=1e-12)
        points = [[1., 2., 3.], [-7., 1., -2.]]
        np.testing.assert_allclose(reduced.field(design, points), full.field(design, points), atol=1e-14, rtol=1e-12)
        self.assertEqual(reduced.diagnostics['preflight']['rotation_order'], 4)
        self.assertEqual(reduced.diagnostics['preflight']['representative_cells'], 3)
        self.assertGreater(full.diagnostics['preflight']['vector_unknowns'], 9)

    def test_volume_quadrature_reduction_matches_full_cells(self):
        design = fixture((4, 4))
        reduced = CoupledResponseEngine(design, {'quadrature_order': 2})
        full = CoupledResponseEngine(design, {'quadrature_order': 2, 'symmetry_reduction': False})
        np.testing.assert_allclose(reduced.local_polarization_T, full.local_polarization_T, atol=1e-13, rtol=1e-12)
        self.assertEqual(reduced.diagnostics['preflight']['target_points_per_cell'], 8)
        self.assertFalse(reduced.diagnostics['spatial_convergence_proven'])

    def test_global_rotation_covariance_includes_local_vector_response(self):
        design = fixture()
        angle = np.deg2rad(37.)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        rotated = replace(design, rings=tuple(replace(r, phase_deg=r.phase_deg+37.) for r in design.rings))
        points = np.array([[3., -5., 1.], [1., 2., -7.]])
        expected = CoupledResponseEngine(design).field(design, points) @ rotation.T
        actual = CoupledResponseEngine(rotated).field(rotated, points @ rotation.T)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-11)

    def test_mirrored_magnets_preserve_even_axial_odd_transverse_field(self):
        upper = RingSpec('a', 0, 0, 50., 20., 8, 'out', 0.)
        design = Design((upper, replace(upper, id='b', layer_index=1, z_mm=-20., polarity='in')),
                        cube_mm=10., reference_mur=1.05)
        engine = CoupledResponseEngine(design)
        fields = engine.field(design, [[3., 2., 7.], [3., 2., -7.]])
        np.testing.assert_allclose(fields[0, :2], -fields[1, :2], atol=1e-14)
        self.assertAlmostEqual(fields[0, 2], fields[1, 2], places=13)

    def test_mutual_response_is_nonzero_and_recoil_equation_residual_is_small(self):
        design = fixture()
        engine = CoupledResponseEngine(design)
        self.assertGreater(engine.diagnostics['max_polarization_change_from_isolated_T'], 1e-7)
        self.assertLess(engine.diagnostics['relative_recoil_residual'], 1e-12)
        self.assertEqual(engine.diagnostics['self_demagnetization_tensor'], (np.eye(3)/3).tolist())
        self.assertEqual(engine.statistics['engine_calls'], 0)
        engine.field(design, [[0., 0., 0.]])
        self.assertEqual(engine.statistics['field_points'], 1)

    def test_radial_ring_matches_independent_scalar_recoil_equation(self):
        ring = RingSpec('single', 0, 0, 20., 0., 4, 'out', 0.)
        design = Design((ring,), cube_mm=10., remanence_T=1.43, reference_mur=1.05)
        sources = []
        for angle in (90., 180., 270.):
            rotation = Rotation.from_euler('z', angle, degrees=True)
            position = rotation.apply([.020, 0., 0.])
            sources.append(magpy.magnet.Cuboid(position=position, dimension=(.01,)*3,
                orientation=rotation, polarization=(1., 0., 0.)))
        neighbor_mu0_Hr_per_J = magpy.getB(sources, [.020, 0., 0.], sumup=True)[0]
        expected_J = 1.43/(1.+.05/3.-.05*neighbor_mu0_Hr_per_J)
        engine = CoupledResponseEngine(design)
        np.testing.assert_allclose(engine.local_polarization_T[:, 0], expected_J, atol=1e-14)
        np.testing.assert_allclose(engine.local_polarization_T[:, 1:], 0., atol=1e-14)

    def test_solved_vector_field_matches_independent_public_objects(self):
        design = fixture()
        engine = CoupledResponseEngine(design)
        arrays = design.magnet_arrays()
        objects = []
        for center, angle, local_j in zip(arrays['positions_mm'], arrays['radial_angles_deg'], engine.local_polarization_T):
            objects.append(magpy.magnet.Cuboid(position=center*1e-3, dimension=(design.cube_mm*1e-3,)*3,
                orientation=Rotation.from_euler('z', angle, degrees=True), polarization=local_j))
        points = np.array([[0., 0., 0.], [2., -3., 4.], [1., 7., -5.]])
        expected = magpy.getB(objects, points*1e-3, sumup=True)
        np.testing.assert_allclose(engine.field(design, points), expected, atol=1e-14, rtol=1e-12)

    def test_preflight_rejects_cost_before_field_or_matrix_allocation(self):
        design = fixture()
        plan = preflight_response(design, {'max_unknowns': 3})
        self.assertFalse(plan['valid'])
        with self.assertRaisesRegex(ValueError, 'max_unknowns'):
            CoupledResponseEngine(design, {'max_unknowns': 3})
        self.assertFalse(preflight_response(design, {'max_matrix_mb': .00001})['valid'])

    def test_overlap_invalid_material_and_reused_wrong_design_fail_explicitly(self):
        design = fixture()
        invalid = replace(design, rings=(design.rings[0], replace(design.rings[0], id='duplicate')))
        with self.assertRaisesRegex(ValueError, 'non-overlapping'):
            CoupledResponseEngine(invalid)
        with self.assertRaises(ValueError):
            CoupledResponseEngine(replace(design, reference_mur=.9))
        engine = CoupledResponseEngine(design)
        with self.assertRaisesRegex(ValueError, 'different physical design'):
            engine.field(replace(design, cube_mm=11.), [[0., 0., 0.]])
        with self.assertRaises(ValueError):
            engine.field(design, [[np.nan, 0., 0.]])


if __name__ == '__main__':
    unittest.main()

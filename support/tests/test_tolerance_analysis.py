"""Physical correlation and domain contracts for bounded tolerance estimates."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

from mri_magnet.field import FieldEngine
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.metrics import field_metrics
from mri_magnet.sampling import disk_samples, sphere_samples, fundamental_sphere_samples
from mri_magnet.tolerance import analyze_tolerance, _sources, _contributions
from mri_magnet.workbench_config import default_config


def fixture():
    # Two rings per physical layer: both members of a sector must move together.
    rings = []
    for layer, z in ((0, 30.), (1, -30.)):
        for slot, radius in enumerate((40., 52.)):
            rings.append(RingSpec('R%d-L%d' % (layer+1, slot+1), layer, slot,
                                  radius, z, 4, 'out' if z > 0 else 'in', 45.))
    return Design(tuple(rings), cube_mm=6., reference_mur=1.05)


def configuration(model='remanence'):
    config = default_config()
    config['manufacturing']['sectors_per_layer'] = 4
    config['polarization_model'] = model
    config['target'].update(geometry='sphere', diameter_mm=20., min_field_mT=.001, max_ppm=None)
    return config


def object_field(design, points, relative_br=None, shifts=None, model='remanence'):
    """Independent public-object path, including non-ring-uniform Br errors."""
    arrays = design.magnet_arrays()
    relative_br = np.zeros(design.magnet_count) if relative_br is None else relative_br
    shifts = np.zeros_like(arrays['positions_mm']) if shifts is None else shifts
    polarization = design.remanence_T
    if model == 'isolated_cube_self_demag':
        polarization /= 1+(design.reference_mur-1)/3
    objects = []
    for center, angle, north, br_error, shift in zip(arrays['positions_mm'], arrays['radial_angles_deg'],
                                                   arrays['north_vectors'], relative_br, shifts):
        rotation = Rotation.from_euler('z', angle, degrees=True)
        local_j = rotation.inv().apply(north*polarization*(1+br_error))
        objects.append(magpy.magnet.Cuboid(position=(center+shift)*1e-3,
            dimension=(design.cube_mm*1e-3,)*3, orientation=rotation, polarization=local_j))
    return magpy.getB(objects, np.asarray(points)*1e-3, sumup=True)


class FixedDraws:
    def __init__(self, raw):
        self.raw = raw
    def uniform(self, lower, upper, size):
        assert lower == -1 and upper == 1 and tuple(size) == self.raw.shape
        return self.raw.copy()


class ToleranceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.design = fixture()
        self.sample = sphere_samples(20., 16, 16, seed=8)
        self.config = configuration()

    def test_per_block_response_sums_to_existing_fast_engine_for_both_models(self):
        for model in ('remanence', 'isolated_cube_self_demag'):
            positions, cs, sn, local_j = _sources(self.design, model)
            response = _contributions(self.sample['points_mm'], positions, cs, sn, local_j, self.design.cube_mm)
            expected = FieldEngine(polarization_model=model).field(self.design, self.sample['points_mm'])
            self.assertEqual(response.shape, (len(expected), 3, self.design.magnet_count))
            np.testing.assert_allclose(response.sum(axis=2), expected, atol=1e-14, rtol=1e-12)

    def test_zero_errors_are_deterministic_preserve_geometry_and_match_nominal(self):
        original = deepcopy(self.design.to_dict())
        options = {'trials': 7, 'remanence_half_range_fraction': 0., 'sector_translation_half_range_mm': 0., 'seed': 1}
        first = analyze_tolerance(self.design, self.sample, self.config, options)
        second = analyze_tolerance(self.design, self.sample, self.config, dict(options, seed=999))
        expected = field_metrics(FieldEngine().field(self.design, self.sample['points_mm']), self.sample['weights'])
        np.testing.assert_allclose(first['ppm_samples'], expected['pp_norm_ppm'], atol=1e-8, rtol=1e-12)
        np.testing.assert_allclose(first['mean_field_mT_samples'], expected['mean_norm_T']*1e3, atol=1e-12)
        self.assertEqual(first['ppm_samples'], second['ppm_samples'])
        self.assertEqual(self.design.to_dict(), original)
        self.assertTrue(all(row['sensitivity_ppm'] == 0. for row in first['sensitivity_rings']))
        self.assertLess(max(c['max_vector_error_ppm_of_nominal_mean'] for c in first['linearization_checks']), 1e-7)

    def test_independent_br_scatter_matches_exact_nonuniform_block_objects(self):
        n = self.design.magnet_count
        raw = np.zeros((2, n+3*8))
        raw[0, :n] = np.linspace(-.9, .8, n)
        raw[1, :n] = np.tile([-.7, .2, .8, -.3], 4)
        options = {'trials': 2, 'remanence_half_range_fraction': .015, 'sector_translation_half_range_mm': 0.}
        with patch('mri_magnet.tolerance.np.random.default_rng', return_value=FixedDraws(raw)):
            result = analyze_tolerance(self.design, self.sample, self.config, options)
        for trial in range(2):
            fields = object_field(self.design, self.sample['points_mm'], relative_br=.015*raw[trial, :n])
            expected = field_metrics(fields, self.sample['weights'])
            self.assertAlmostEqual(result['mean_field_mT_samples'][trial], expected['mean_norm_T']*1e3, places=11)
            self.assertAlmostEqual(result['ppm_samples'][trial], expected['pp_norm_ppm'], delta=1e-7)
        self.assertLess(max(c['max_vector_error_ppm_of_nominal_mean'] for c in result['linearization_checks']), 1e-7)

    def test_same_layer_sector_translation_is_shared_by_both_radial_rings(self):
        n = self.design.magnet_count
        raw = np.zeros((2, n+3*8))
        raw[0, n:] = np.linspace(-.8, .7, 24)
        raw[1, n:] = np.tile([.6, -.2, -.7], 8)
        half_range = 1e-6
        options = {'trials': 2, 'remanence_half_range_fraction': 0.,
                   'sector_translation_half_range_mm': half_range, 'translation_fd_step_mm': .01}
        with patch('mri_magnet.tolerance.np.random.default_rng', return_value=FixedDraws(raw)):
            result = analyze_tolerance(self.design, self.sample, self.config, options)
        for trial in range(2):
            shifts = []
            # 45/135/225/315 degrees map to sectors 0/1/2/3. The two
            # radial rings of each layer reuse each sector's exact same draw.
            for ring in self.design.rings:
                shifts.extend(raw[trial, n:].reshape(8, 3)[4*ring.layer_index:4*ring.layer_index+4]*half_range)
            fields = object_field(self.design, self.sample['points_mm'], shifts=np.asarray(shifts))
            expected = field_metrics(fields, self.sample['weights'])
            self.assertAlmostEqual(result['mean_field_mT_samples'][trial], expected['mean_norm_T']*1e3, delta=1e-10)
            self.assertAlmostEqual(result['ppm_samples'][trial], expected['pp_norm_ppm'], delta=1e-5)
        self.assertIn('same physical layer/quadrant', result['assumptions']['translation_correlation'])

    def test_full_global_domain_is_required_for_sphere_and_direct_sector_disk(self):
        reduced = [fundamental_sphere_samples(20., 8, 8, 1, rotation_order=4),
                   disk_samples(20., 8, 8, 1, rotation_order=4)]
        for sample in reduced:
            with self.subTest(sample=sample['name']), self.assertRaisesRegex(ValueError, 'global samples'):
                analyze_tolerance(self.design, sample, self.config, {'trials': 2})

    def test_wrapped_phases_keep_same_physical_column_in_same_printed_sector(self):
        original = replace(self.design, rings=tuple(replace(r, phase_deg=0.) for r in self.design.rings))
        wrapped = replace(original, rings=tuple(replace(r, phase_deg=360.) if i == 0 else r
                                               for i, r in enumerate(original.rings)))
        options = {'trials': 3, 'remanence_half_range_fraction': 0.,
                   'sector_translation_half_range_mm': .15, 'seed': 7}
        first = analyze_tolerance(original, self.sample, self.config, options)
        second = analyze_tolerance(wrapped, self.sample, self.config, options)
        np.testing.assert_allclose(first['mean_field_mT_samples'], second['mean_field_mT_samples'], atol=1e-12)
        np.testing.assert_allclose(first['ppm_samples'], second['ppm_samples'], atol=1e-6)

    def test_small_memory_limit_rejects_before_source_expansion_or_field(self):
        with patch('mri_magnet.tolerance._sources', side_effect=AssertionError('source expansion must not happen')):
            with self.assertRaisesRegex(ValueError, 'limit'):
                analyze_tolerance(self.design, self.sample, self.config, {'trials': 2, 'max_memory_mb': .000001})

    def test_yield_is_explicitly_sampled_field_only_without_ppm_requirement(self):
        result = analyze_tolerance(self.design, self.sample, self.config,
            {'trials': 4, 'sector_translation_half_range_mm': 0.})
        self.assertEqual(result['summary']['yield_scope'], 'field_only_no_ppm_limit')
        self.assertFalse(result['assumptions']['dense_extrema_verified'])
        self.assertFalse(result['assumptions']['manufacturing_yield_included'])
        self.assertFalse(result['assumptions']['coupled_response_uncertainty_included'])
        self.assertFalse(result['assumptions']['symmetry_reduction_used'])
        self.assertGreaterEqual(result['summary']['yield_confidence95'][0], 0.)
        self.assertLessEqual(result['summary']['yield_confidence95'][1], 1.)

    def test_actual_ppm_limit_is_applied_and_invalid_errors_are_rejected(self):
        config = deepcopy(self.config)
        config['target']['max_ppm'] = .001
        result = analyze_tolerance(self.design, self.sample, config,
            {'trials': 3, 'remanence_half_range_fraction': 0., 'sector_translation_half_range_mm': 0.})
        self.assertEqual(result['summary']['yield_fraction'], 0.)
        self.assertEqual(result['summary']['yield_scope'], 'field_and_sampled_ppm')
        for changes in ({'remanence_half_range_fraction': -1}, {'sector_translation_half_range_mm': -1},
                        {'translation_fd_step_mm': 0}, {'trials': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analyze_tolerance(self.design, self.sample, self.config, changes)


if __name__ == '__main__':
    unittest.main()

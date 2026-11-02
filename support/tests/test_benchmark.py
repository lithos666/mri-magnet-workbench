"""Toy orchestration contracts, not performance claims about real magnets."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np

from mri_magnet import benchmark
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.objectives import check_performance


class ToyEngine:
    def __init__(self, **kwargs):
        self.statistics = {'engine_calls': 0, 'field_points': 0, 'preparations': 1}
    def field(self, design, points):
        self.statistics['engine_calls'] += 1
        self.statistics['field_points'] += len(points)
        return np.tile([0., 0., .04], (len(points), 1))


class ToyTopologies(list):
    metadata = {'fixture': 'synthetic'}
    def __init__(self, seed):
        super().__init__([seed, replace(seed, name='alternative')])
        self.warm_starts = [replace(seed, name='slot_swap')]


class BenchmarkContractTests(unittest.TestCase):
    def config(self):
        return {'seed': 123, 'max_evaluations': 24, 'quick': {'max_evaluations': 12},
            'source_geometry': 'not-read-by-fixture.json', 'chunk_points': 8,
            'polarization_model': 'remanence', 'density_kg_m3': 7500.,
            'fov_diameter_mm': 20., 'fov_z_mm':0., 'popsize': 3, 'maxiter': 5,
            'requirements': {'min_mean_norm_reference': 'baseline', 'min_mean_norm_ratio': 1.0},
            'manufacturing': {'enabled': True},
            'search': {'topology_limit': 2, 'seed_subset_field_selection': True,
                       'warm_start_runs': ['synthetic-run'], 'optimize_azimuth_count': True,
                       'optimize_radial_ring_count': True, 'optimize_magnetization': False},
            'sampling': {'quick_training_area': 8, 'quick_training_boundary': 8,
                         'quick_validation_area': 16, 'quick_validation_boundary': 16},
            'benchmark': {'refine_extrema': False}}

    def execute_fixture(self, target, config, seeds):
        original = Design((RingSpec('fixture', 0, 0, 130., 20., 4, 'out', 0.),))
        seed = replace(original, name='projected_seed')
        common_warm = replace(seed, name='external_warm')
        seen_samples, search_calls = [], []
        def evaluate(design, engine, sample, density):
            seen_samples.append((sample['name'], id(sample), design.name))
            B = engine.field(design, sample['points_mm'])
            ppm = 900. if design.name.startswith('mixed') else 1000.
            field = config.get('_fixture_training_field', .04) if 'A8_C8_' in sample['name'] else .04
            return {'mean_norm_T': field, 'pp_norm_ppm': ppm, 'mean_Bz_T': field,
                    'pp_Bz_ppm': ppm, 'pointset': sample['name']}, B
        def prepare(source, process, limits, search, field_error):
            self.assertIsNotNone(field_error)
            self.assertEqual(field_error(seed), 0.)
            return seed, {'valid': True, 'seed_field_evaluations': 1}
        def optimize(topologies, process, limits, search, evaluator, checker, output,
                     settings, requirements, warm_starts=()):
            mode = 'mixed' if output.parent.name.startswith('mixed') else 'fixed'
            design = replace(seed, name=mode+'_synthetic_best')
            metrics = evaluator(design)
            search_calls.append({'mode': mode, 'settings': copy.deepcopy(settings),
                'requirements': requirements, 'warm': list(warm_starts), 'topologies': len(topologies),
                'first_design': topologies[0], 'training_metrics': metrics})
            return {'records': [{'candidate_id': mode+'_fixture', 'design': design.to_dict(), 'metrics': metrics}],
                'metadata': {'field_evaluations': settings['max_evaluations'], 'topologies_run': len(topologies),
                    'runs': [{'topology': 1, 'active_dimension': 3, 'metadata': {'rounds': [
                        {'population_size': 9, 'generations': 2, 'state': 'budget_limited',
                         'initialization_complete': True}]}}]}}
        with patch.object(benchmark, 'load_source', return_value=(original, {})), \
             patch.object(benchmark, '_limits', return_value={'bore_radius_mm': 120.}), \
             patch.object(benchmark, '_prepare_seed', side_effect=prepare) as prepared, \
             patch.object(benchmark, '_design_check', return_value={'valid': True, 'violations': []}), \
             patch.object(benchmark, '_load_warm_starts', return_value=([common_warm], [])), \
             patch.object(benchmark, 'FieldEngine', ToyEngine), \
             patch.object(benchmark, 'evaluate_design', side_effect=evaluate), \
             patch.object(benchmark, 'propose_topologies', side_effect=lambda source, initial, *a: ToyTopologies(initial)), \
             patch.object(benchmark, 'run_topology_optimization', side_effect=optimize):
            result = benchmark.run_benchmark(config, target, quick=True, seeds=seeds, budget=12)
            self.assertEqual(prepared.call_count, 1)
        return result, search_calls, seen_samples

    def test_two_modes_and_two_seeds_share_points_start_thresholds_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            result, calls, samples = self.execute_fixture(Path(folder), self.config(), (11, 22))
            self.assertEqual(len(calls), 4)
            self.assertEqual({(c['mode'], c['settings']['seed']) for c in calls},
                             {('fixed', 11), ('mixed', 11), ('fixed', 22), ('mixed', 22)})
            self.assertEqual({c['settings']['max_evaluations'] for c in calls}, {12})
            self.assertEqual({id(c['requirements']) for c in calls}, {id(calls[0]['requirements'])})
            self.assertEqual({id(c['first_design']) for c in calls}, {id(calls[0]['first_design'])})
            self.assertTrue(all(c['warm'][0].name == 'external_warm' for c in calls))
            self.assertTrue(all(len(c['warm']) == (1 if c['mode'] == 'fixed' else 2) for c in calls))
            training = [s for s in samples if 'A8_C8_' in s[0]]
            validation = [s for s in samples if 'A16_C16_' in s[0]]
            self.assertEqual(len({s[1] for s in training}), 1)
            self.assertEqual(len({s[1] for s in validation}), 1)
            self.assertNotEqual(training[0][1], validation[0][1])
            self.assertEqual(len({r['training_sampling_sha256'] for r in result['rows']}), 1)
            self.assertEqual(len({r['seed_design_sha256'] for r in result['rows']}), 1)
            self.assertTrue(all(r['performance_pass'] for r in result['rows']))
            self.assertEqual(result['paired_comparisons'][0]['mixed_ppm_improvement_percent'], 10.)
            for key in ('report_html', 'csv', 'summary_json'):
                self.assertTrue(Path(result[key]).is_file())
            self.assertEqual(len(list((Path(folder)/'详细数据').glob('*_seed_*/benchmark_run.json'))), 4)
            self.assertFalse(result['native_fem_executed'])
            self.assertFalse(result['global_optimum_proven'])

    def test_failed_thresholds_are_diagnostic_and_not_counted_as_improvement(self):
        config = self.config()
        config['requirements']['min_mean_norm_mT'] = 100.
        with tempfile.TemporaryDirectory() as folder:
            result, _, _ = self.execute_fixture(Path(folder), config, (11,))
            self.assertTrue(all(not r['performance_pass'] for r in result['rows']))
            self.assertIsNone(result['paired_comparisons'][0]['mixed_ppm_improvement_percent'])
            self.assertIsNone(result['mode_statistics']['mixed']['feasible_ppm_min'])
            self.assertIn('未达门槛', Path(result['report_html']).read_text(encoding='utf-8'))

    def test_training_reference_gap_does_not_make_the_seed_infeasible(self):
        config = self.config()
        config['_fixture_training_field'] = .039
        config['requirements']['min_mean_norm_reference'] = 'manufacturing_seed'
        with tempfile.TemporaryDirectory() as folder:
            result, calls, _ = self.execute_fixture(Path(folder), config, (11,))
            self.assertTrue(all(check_performance(c['training_metrics'], c['requirements'])['valid'] for c in calls))
            self.assertEqual(result['training_requirements']['min_mean_norm_T'], .039)
            self.assertEqual(result['requirements']['min_mean_norm_T'], .04)
            self.assertTrue(all(r['performance_pass'] for r in result['rows']))
            self.assertEqual(result['requirements']['field_tolerance_T'], 0.)

    def test_reference_and_candidates_share_the_same_extrema_refinement(self):
        config = self.config()
        config['benchmark']['refine_extrema'] = True
        refined = []
        def refine(design, engine, sample, metrics, fields, starts):
            refined.append((design.name, sample['name'], starts))
            return dict(metrics, pp_norm_ppm=metrics['pp_norm_ppm']*1.1)
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark, 'refine_disk_extrema', side_effect=refine):
            result, _, _ = self.execute_fixture(Path(folder), config, (11,))
            self.assertEqual(len(refined), 4)  # original + seed + one candidate per mode
            self.assertEqual({r[2] for r in refined}, {1})
            self.assertEqual(len({r[1] for r in refined}), 1)
            self.assertEqual(result['requirements']['max_pp_norm_ppm'], 1100.)
            self.assertTrue(all(r['performance_pass'] for r in result['rows']))

    def test_duplicate_seed_or_invalid_budget_rejected_before_source_read(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark, 'load_source') as source:
            with self.assertRaises(ValueError):
                benchmark.run_benchmark(self.config(), folder, seeds=(1, 1), budget=12)
            with self.assertRaises(ValueError):
                benchmark.run_benchmark(self.config(), folder, seeds=(1, 2), budget=0)
            source.assert_not_called()


if __name__ == '__main__':
    unittest.main()

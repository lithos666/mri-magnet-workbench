"""Mechanism checks for fair screening, budget depth, migration and workers."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.islands import run_islands
from mri_magnet.manufacturing import audit_manufacturing


def seed(columns=28, pairs=5, radius=60.):
    rings = []
    for i, z in enumerate(np.linspace(120., 20., pairs)):
        rings.append(RingSpec('R%02d-L1' % (i+1), i, 0, radius, float(z), columns, 'out', 0.))
        j = 2*pairs-1-i
        rings.append(RingSpec('R%02d-L1' % (j+1), j, 0, radius, -float(z), columns, 'in', 0.))
    return Design(tuple(sorted(rings, key=lambda r:r.layer_index)), cube_mm=4., name='island_seed')


PROCESS = dict(enabled=True, layout_mode='independent_pockets',
               pocket_clearance_per_side_mm=.2, pocket_web_mm=.8,
               inner_wall_mm=1., outer_wall_mm=1., base_thickness_mm=1.,
               bed_margin_per_side_mm=2., min_axial_module_gap_mm=1.,
               sectors_per_layer=4, printer_bed_mm=[220., 220., 250.],
               max_magnet_count=2000)
LIMITS = dict(bore_radius_mm=20., outer_radius_mm=100., half_length_mm=150.,
              min_gap_mm=0., preserve_exact_bore=False, tolerance_mm=1e-6)
SEARCH = dict(optimize_radius=True, optimize_axial_positions=True, axial_shift_limit_mm=5.)
REQUIREMENTS = dict(min_mean_norm_T=.03, max_pp_norm_ppm=100., field_scale_T=.03, ppm_scale=100.)


@dataclass
class Checker:
    def __call__(self, design):
        return audit_manufacturing(design, PROCESS, LIMITS)


@dataclass
class Evaluator:
    low_field_columns: int = 0
    calls: int = 0

    def __call__(self, design):
        self.calls += 1
        count = design.rings[0].count
        radius_error = sum((ring.radius_mm-58.)**2 for ring in design.rings)/len(design.rings)
        return dict(mean_norm_T=.02 if count == self.low_field_columns else .04,
                    pp_norm_ppm=(.001 if count == self.low_field_columns else 70-count) + radius_error*.0001)


def settings(budget=100, workers=1):
    return dict(max_evaluations=budget, seed=41, popsize=3, maxiter=100,
                optimization_options=dict(population_size=5, initial_local_fraction=.7,
                    initial_sigma_mm=.8, local_refine_evaluations=8, pair_refine_fraction=.5),
                islands=dict(max_workers=workers, finalists=2, initial_budget_fraction=.4))


class IslandTests(unittest.TestCase):
    def run_job(self, topologies=None, config=None, evaluator=None, callback=None):
        topologies = topologies or [seed(24,5), seed(28,6), seed(32,8)]
        with tempfile.TemporaryDirectory() as root:
            result = run_islands(topologies, PROCESS, LIMITS, SEARCH, evaluator or Evaluator(),
                                 Checker(), root, config or settings(), REQUIREMENTS,
                                 event_callback=callback)
            self.assertTrue((Path(root)/'optimization_result.json').exists())
            self.assertTrue((Path(root)/'island_progress.json').exists())
            return result

    def test_successive_halving_promotes_stronger_heterogeneous_islands(self):
        result = self.run_job()
        rounds = result['metadata']['island_rounds']
        self.assertEqual(result['metadata']['island_round_sizes'], [3,2])
        self.assertEqual(rounds[0]['promoted_islands'], [3,2])
        self.assertEqual(rounds[0]['eliminated_islands'], [1])
        self.assertEqual({row['island'] for row in rounds[1]['islands']}, {2,3})
        self.assertTrue(all(row['de_generations'] >= 1 for row in rounds[0]['islands']))
        self.assertTrue(all(row['initialization_complete'] for row in rounds[0]['islands']))
        self.assertFalse(result['metadata']['global_optimum_proven'])

    def test_exact_global_budget_and_monotone_live_counters(self):
        evaluator, events = Evaluator(), []
        result = self.run_job(evaluator=evaluator, callback=events.append)
        self.assertEqual(evaluator.calls, result['metadata']['field_evaluations'])
        self.assertEqual(evaluator.calls, 100)
        self.assertLessEqual(evaluator.calls, result['metadata']['max_evaluations'])
        counters = [event['global_evaluations'] for event in events]
        self.assertTrue(counters)
        self.assertEqual(counters, sorted(counters))
        self.assertLessEqual(max(counters), evaluator.calls)
        for round_info in result['metadata']['island_rounds']:
            self.assertTrue(all(row['field_evaluations'] <= row['allocation'] for row in round_info['islands']))

    def test_infeasible_low_ppm_does_not_eliminate_feasible_islands(self):
        result = self.run_job(evaluator=Evaluator(low_field_columns=32))
        self.assertEqual(result['metadata']['island_rounds'][0]['promoted_islands'], [2,1])
        self.assertEqual(result['metadata']['island_rounds'][0]['eliminated_islands'], [3])
        self.assertTrue(result['diagnostic_best']['feasible'])

    def test_too_small_budget_rejected_before_any_field_evaluation(self):
        evaluator = Evaluator()
        with self.assertRaisesRegex(ValueError, 'need at least 50'):
            self.run_job(config=settings(budget=49), evaluator=evaluator)
        self.assertEqual(evaluator.calls, 0)

    def test_compatible_physical_migration_without_cross_topology_vectors(self):
        result = self.run_job(topologies=[seed(28,5,58.), seed(28,5,63.), seed(32,5,60.)])
        rows = result['metadata']['island_rounds'][-1]['islands']
        same = next(row for row in rows if row['island'] == 1)
        # Both count-28 islands may have the same score order. Select 1 or 2
        # directly with a finalists=3 single round would not test migration;
        # here the count-32 champion and one 28 island survive. Their topologies
        # differ, so no wrong-coordinate migration is allowed.
        self.assertTrue(all(not value.startswith('other_island_3')
                            for value in same['compatible_warm_sources']))
        self.assertIn('own_incumbent', same['compatible_warm_sources'])

    def test_callback_error_is_reported_without_changing_search_budget(self):
        def fail(event):
            raise RuntimeError('preview broken')
        result = self.run_job(callback=fail)
        self.assertEqual(result['metadata']['monitor_errors'], ['preview broken'])
        self.assertEqual(result['metadata']['field_evaluations'], 100)

    def test_same_topology_elites_migrate_as_physical_designs(self):
        result = self.run_job(topologies=[seed(28,5,58.), seed(28,5,63.), seed(24,5,60.)])
        self.assertEqual(set(result['metadata']['island_rounds'][0]['promoted_islands']), {1,2})
        rows = result['metadata']['island_rounds'][-1]['islands']
        self.assertIn('other_island_2', rows[0]['compatible_warm_sources'])
        self.assertIn('other_island_1', rows[1]['compatible_warm_sources'])
        self.assertTrue(result['metadata']['stage_summaries'])
        self.assertTrue(all('global_evaluations' in row and 'topology_index' in row
                            for row in result['metadata']['stage_summaries']))

    def test_parallel_and_serial_find_identical_candidates_under_same_schedule(self):
        serial = self.run_job(config=settings(workers=1))
        events = []
        parallel = self.run_job(config=settings(workers=2), callback=events.append)
        self.assertEqual(parallel['metadata']['execution'], 'process_parallel')
        self.assertEqual(parallel['metadata']['field_evaluations'], serial['metadata']['field_evaluations'])
        self.assertEqual(parallel['diagnostic_best']['canonical_physical_sha256'],
                         serial['diagnostic_best']['canonical_physical_sha256'])
        self.assertEqual([event['global_evaluations'] for event in events],
                         sorted(event['global_evaluations'] for event in events))
        self.assertLessEqual(max(event['global_evaluations'] for event in events), 100)

    def test_parallel_requires_picklable_callables_before_start(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'picklable'):
                run_islands([seed()], PROCESS, LIMITS, SEARCH,
                            lambda d: {'mean_norm_T':.04,'pp_norm_ppm':1.}, Checker(), root,
                            settings(budget=20,workers=2), REQUIREMENTS)


if __name__ == '__main__':
    unittest.main()

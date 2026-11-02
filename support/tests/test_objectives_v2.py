"""Performance requirements use dense references and strict feasibility ordering."""
import math
import unittest

from mri_magnet.objectives import resolve_requirements, check_performance, feasibility_score


def metrics(field, ppm=100000.):
    return {'mean_norm_T': field, 'pp_norm_ppm': ppm}


class PerformanceObjectivesV2Tests(unittest.TestCase):
    def setUp(self):
        self.original_dense = metrics(.037345828, 127099.6)
        self.printing_dense = metrics(.035400592, 129076.2)
        self.requirements = resolve_requirements({}, self.original_dense, self.printing_dense)

    def test_default_requirement_preserves_printable_seed_field_and_ppm(self):
        self.assertEqual(self.requirements['reference'], 'manufacturing_seed')
        self.assertEqual(self.requirements['min_mean_norm_T'], self.printing_dense['mean_norm_T'])
        self.assertEqual(self.requirements['max_pp_norm_ppm'], self.printing_dense['pp_norm_ppm'])
        self.assertEqual(self.requirements['ppm_scale'], self.printing_dense['pp_norm_ppm'])
        self.assertTrue(check_performance(self.printing_dense, self.requirements)['valid'])
        self.assertFalse(check_performance(metrics(.04, self.printing_dense['pp_norm_ppm']+1), self.requirements)['valid'])

    def test_original_reference_and_optional_unbounded_ppm_remain_available(self):
        req = resolve_requirements({'min_mean_norm_reference': 'baseline', 'max_pp_norm_reference': None},
                                   self.original_dense, self.printing_dense)
        self.assertEqual(req['min_mean_norm_T'], self.original_dense['mean_norm_T'])
        self.assertIsNone(req['max_pp_norm_ppm'])

    def test_ppm_reference_ratio_and_explicit_override(self):
        req = resolve_requirements({'max_pp_norm_reference': 'baseline', 'max_pp_norm_ratio': .5},
                                   self.original_dense, self.printing_dense)
        self.assertEqual(req['max_pp_norm_ppm'], self.original_dense['pp_norm_ppm']*.5)
        req = resolve_requirements({'max_pp_norm_reference': 'baseline', 'max_pp_norm_ratio': .5,
                                    'max_pp_norm_ppm': 210.}, self.original_dense, self.printing_dense)
        self.assertEqual(req['max_pp_norm_ppm'], 210.)

    def test_reference_gates_resolve_on_their_own_pointsets_without_tolerance(self):
        training_seed = dict(metrics(.0354001, 129075.), pointset='training')
        dense_seed = dict(self.printing_dense, pointset='validation')
        training_req = resolve_requirements({}, self.original_dense, training_seed)
        dense_req = resolve_requirements({}, self.original_dense, dense_seed)
        self.assertTrue(check_performance(training_seed, training_req)['valid'])
        self.assertTrue(check_performance(dense_seed, dense_req)['valid'])
        self.assertFalse(check_performance(training_seed, dense_req)['valid'])
        self.assertEqual(training_req['field_tolerance_T'], 0.)
        self.assertEqual(training_req['reference_metric_pointsets']['manufacturing_seed'], 'training')
        absolute = {'min_mean_norm_mT': 37., 'max_pp_norm_ppm': 210.}
        for req in (resolve_requirements(absolute, self.original_dense, training_seed),
                    resolve_requirements(absolute, self.original_dense, dense_seed)):
            self.assertEqual(req['min_mean_norm_T'], .037)
            self.assertEqual(req['max_pp_norm_ppm'], 210.)

    def test_uniform_reference_can_require_zero_ppm(self):
        uniform = metrics(.04, 0.)
        req = resolve_requirements({}, uniform, uniform)
        self.assertTrue(check_performance(uniform, req)['valid'])
        self.assertFalse(check_performance(metrics(.04, 1e-9), req)['valid'])

    def test_explicit_millitesla_and_seed_relative_thresholds(self):
        absolute = resolve_requirements({'min_mean_norm_mT': 38., 'min_mean_norm_ratio': .9},
                                       self.original_dense, self.printing_dense)
        self.assertAlmostEqual(absolute['min_mean_norm_T'], .038)
        relative = resolve_requirements({'min_mean_norm_reference': 'manufacturing_seed', 'min_mean_norm_ratio': 1.1},
                                       self.original_dense, self.printing_dense)
        self.assertAlmostEqual(relative['min_mean_norm_T'], 1.1*self.printing_dense['mean_norm_T'])

    def test_any_finite_feasible_ppm_score_beats_an_infeasible_field(self):
        # A very large ppm cannot cross the feasibility boundary as a big-M
        # penalty can. There is intentionally no configured ppm upper bound.
        self.requirements = resolve_requirements({'max_pp_norm_reference': None}, self.original_dense, self.printing_dense)
        feasible = metrics(.038, 1e150)
        infeasible = metrics(self.requirements['min_mean_norm_T']-1e-9, 0.)
        self.assertTrue(check_performance(feasible, self.requirements)['valid'])
        self.assertLess(feasibility_score(feasible, self.requirements), 0.)
        self.assertGreater(feasibility_score(infeasible, self.requirements), 0.)
        self.assertLess(feasibility_score(feasible, self.requirements), feasibility_score(infeasible, self.requirements))

    def test_infeasible_field_gap_order_is_not_perturbed_by_ppm(self):
        self.requirements = resolve_requirements({'min_mean_norm_reference': 'baseline', 'max_pp_norm_reference': None}, self.original_dense, self.printing_dense)
        smaller_gap = metrics(.037, 1e12)
        larger_gap = metrics(.036, 0.)
        self.assertFalse(check_performance(smaller_gap, self.requirements)['valid'])
        self.assertLess(feasibility_score(smaller_gap, self.requirements), feasibility_score(larger_gap, self.requirements))
        self.assertLess(feasibility_score(metrics(.038, 60000), self.requirements),
                        feasibility_score(metrics(.041, 90000), self.requirements))

    def test_optional_ppm_upper_limit_is_an_independent_requirement(self):
        requirements = resolve_requirements({'max_pp_norm_ppm': 80000}, self.original_dense, self.printing_dense)
        passing = check_performance(metrics(.04, 80000), requirements)
        failing = check_performance(metrics(.05, 80001), requirements)
        self.assertTrue(passing['valid'])
        self.assertFalse(failing['valid'])
        self.assertIn('peak-to-peak ppm exceeds required maximum', failing['violations'])
        self.assertGreater(feasibility_score(metrics(.05, 80001), requirements), 0.)

    def test_field_tolerance_is_explicit_and_invalid_inputs_are_rejected(self):
        requirements = resolve_requirements({'min_mean_norm_mT': 40, 'field_tolerance_mT': .1},
                                           self.original_dense, self.printing_dense)
        self.assertTrue(check_performance(metrics(.03995), requirements)['valid'])
        self.assertFalse(check_performance(metrics(.0398), requirements)['valid'])
        for options in ({'min_mean_norm_mT': -1}, {'min_mean_norm_ratio': float('nan')},
                        {'max_pp_norm_ppm': 0}, {'field_tolerance_mT': -1},
                        {'min_mean_norm_reference': 'training_seed'}, {'max_pp_norm_reference': 'training_seed'},
                        {'max_pp_norm_ratio': 0}, {'max_pp_norm_ratio': float('nan')}):
            with self.assertRaises(ValueError):
                resolve_requirements(options, self.original_dense, self.printing_dense)
        self.assertFalse(check_performance(metrics(float('nan')), requirements)['valid'])


if __name__ == '__main__':
    unittest.main()

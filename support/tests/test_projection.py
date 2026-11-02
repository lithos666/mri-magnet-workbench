"""Bounded PAVA checked against known projections and an independent QP."""
import unittest
import numpy as np
from scipy.optimize import minimize

from mri_magnet.projection import project_ordered


class OrderedProjectionTests(unittest.TestCase):
    def test_original_nonuniform_three_ring_layouts_move_only_required_endpoints(self):
        upper = 184.593016004
        np.testing.assert_allclose(project_ordered([127, 150, 175], 127.4, upper, 13.6), [127.4, 150, 175], atol=1e-12)
        np.testing.assert_allclose(project_ordered([127, 162, 186], 127.4, upper, 13.6), [127.4, 162, upper], atol=1e-12)

    def test_projection_matches_independent_constrained_least_squares(self):
        rng = np.random.default_rng(51)
        for n in (2, 3, 5):
            for _ in range(4):
                values = np.sort(rng.uniform(122, 190, n))
                actual = project_ordered(values, 127.4, 184.593016, 13.6)
                initial = np.linspace(127.4, 184.593016, n)
                qp = minimize(lambda x: np.sum((x-values)**2), initial,
                              jac=lambda x: 2*(x-values), method='SLSQP',
                              bounds=[(127.4, 184.593016)] * n,
                              constraints={'type': 'ineq', 'fun': lambda x: np.diff(x)-13.6},
                              options={'ftol': 1e-10, 'maxiter': 200})
                self.assertLessEqual(abs(np.sum((actual-values)**2)-qp.fun), 1e-6)
                self.assertGreaterEqual(actual[0], 127.4-1e-12)
                self.assertLessEqual(actual[-1], 184.593016+1e-12)
                self.assertTrue(np.all(np.diff(actual) >= 13.6-1e-12))

    def test_feasible_vector_is_exactly_preserved_and_zero_slack_is_supported(self):
        values = np.array([130., 148., 178.])
        np.testing.assert_array_equal(project_ordered(values, 127.4, 184.593016, 13.6), values)
        np.testing.assert_allclose(project_ordered([0, 5, 20], 4, 12, 4), [4, 8, 12])

    def test_invalid_and_empty_feasible_regions_are_rejected(self):
        for values, lower, upper, gap in (([], 0, 1, 0), ([1, np.nan], 0, 1, 0),
                                          ([1, 2, 3], 0, 1, 1), ([1], 0, 1, -1)):
            with self.assertRaises(ValueError):
                project_ordered(values, lower, upper, gap)


if __name__ == '__main__':
    unittest.main()

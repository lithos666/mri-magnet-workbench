"""Regression checks for budgets, physical cache provenance and integer priors."""
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from mri_magnet.optimization import run_optimization, initialize_population


@dataclass(frozen=True)
class Toy:
    radius: float=0.
    name: str='first'
    def to_dict(self):return vars(self)


def metrics(design):return {'mean_norm_T':.04+design.radius*.001,'pp_norm_ppm':100.+design.radius**2}


class OptimizerV2Tests(unittest.TestCase):
    def test_integer_random_population_endpoints_are_not_half_weighted(self):
        variables=[dict(name='choice',lower=0,upper=4,initial=0,integer=True)]
        values=initialize_population(variables,np.random.default_rng(91),100000)[:,0].astype(int)
        counts=np.bincount(values,minlength=5)
        self.assertTrue(np.all(np.abs(counts-20000)<700),counts)

    def test_budget_interrupts_initialization_without_shrinking_population(self):
        variables=[dict(name='x'+str(i),lower=-1,upper=1,initial=0,integer=False) for i in range(8)]
        calls=[]
        def evaluate(d):calls.append(d);return metrics(d)
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables,lambda x:Toy(float(np.sum(x))),evaluate,
                lambda d:{'valid':True},Toy(),folder,max_evaluations=7,popsize=3,maxiter=30)
        self.assertEqual(len(calls),7)
        info=result['metadata']['rounds'][0]
        self.assertEqual(info['population_size'],24)
        self.assertFalse(info['initialization_complete'])
        self.assertEqual(info['state'],'initialization_budget_limited')

    def test_shared_cache_keeps_new_candidate_content_and_raw_fingerprint(self):
        cache={};calls=[]
        def evaluate(d):calls.append(d);return metrics(d)
        with tempfile.TemporaryDirectory() as folder:
            first=run_optimization([],lambda x:Toy(name='first'),evaluate,lambda d:{'valid':True},
                                   Toy(name='first'),Path(folder)/'first',numeric_cache=cache)
            second=run_optimization([],lambda x:Toy(name='second'),evaluate,lambda d:{'valid':True},
                                    Toy(name='second'),Path(folder)/'second',numeric_cache=cache)
        self.assertEqual(len(calls),1)
        self.assertEqual(second['baseline']['design']['name'],'second')
        self.assertNotEqual(first['baseline']['raw_design_sha256'],second['baseline']['raw_design_sha256'])
        self.assertEqual(first['baseline']['canonical_physical_sha256'],second['baseline']['canonical_physical_sha256'])
        self.assertEqual(second['metadata']['field_evaluations'],0)

    def test_infeasible_seed_can_search_toward_field_requirement(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization([dict(name='radius',lower=0,upper=10,initial=0,integer=False)],
                lambda x:Toy(float(x[0])),metrics,lambda d:{'valid':True},Toy(),folder,
                max_evaluations=40,popsize=8,maxiter=10,
                requirements={'min_mean_norm_T':.045,'field_scale_T':.04,'ppm_scale':100})
        self.assertFalse(result['baseline']['feasible'])
        self.assertTrue(result['pareto'])
        self.assertTrue(all(r['metrics']['mean_norm_T']>=.045 for r in result['pareto']))


if __name__=='__main__':unittest.main()

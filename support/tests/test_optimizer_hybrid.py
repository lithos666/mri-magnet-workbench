import unittest,tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from mri_magnet.optimization import initialize_population,run_optimization

@dataclass(frozen=True)
class Point:
    coords: tuple
    def to_dict(self):return {'coordinates':list(self.coords)}

def variables(n,initial=.4):
    return [dict(name='x%d'%i,lower=0.,upper=1.,initial=initial,integer=False) for i in range(n)]

def metric(p):return {'mean_norm_T':.04,'pp_norm_ppm':100+100*sum((x-.45)**2 for x in p.coords)}

class HybridRegressionTests(unittest.TestCase):
    def test_normalized_local_neighbours_and_reproducible_global_remainder(self):
        v=variables(5);n=1001;f=.65;count=int(round((n-1)*f))
        first=initialize_population(v,np.random.default_rng(36),n,local_fraction=f,local_sigma=.06)
        second=initialize_population(v,np.random.default_rng(36),n,local_fraction=f,local_sigma=.06)
        np.testing.assert_array_equal(first,second)
        self.assertLess(np.std(first[1:count+1]-.4),.065)
        self.assertGreater(np.std(first[count+1:]),.25)
        self.assertTrue(np.all((first>=0)&(first<=1)))

    def test_local_poll_improves_at_zero_de_generations_with_hard_total_cap(self):
        calls=[]
        def evaluate(d):calls.append(d);return metric(d)
        with tempfile.TemporaryDirectory() as folder:
            r=run_optimization(variables(6),lambda x:Point(tuple(x)),evaluate,lambda d:{'valid':True},
                Point((.4,)*6),folder,seed=6,max_evaluations=40,population_size=5,maxiter=0,
                initial_local_fraction=1,initial_sigma=1e-12,local_refine_evaluations=30,local_step=.05,
                requirements={'min_mean_norm_T':.039,'max_pp_norm_ppm':102.,'field_scale_T':.04,'ppm_scale':100})
        info=r['metadata']['rounds'][0]
        self.assertLessEqual(len(calls),40)
        self.assertEqual(len(calls),r['metadata']['field_evaluations'])
        self.assertGreater(info['local_refinement']['field_evaluations'],0)
        self.assertLessEqual(info['local_refinement']['field_evaluations'],30)
        self.assertGreater(info['local_refinement']['accepted_moves'],0)
        self.assertLess(r['diagnostic_best']['metrics']['pp_norm_ppm'],r['baseline']['metrics']['pp_norm_ppm']-.5)

    def test_explicit_population_not_implicitly_reduced_by_small_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            r=run_optimization(variables(8),lambda x:Point(tuple(x)),metric,lambda d:{'valid':True},Point((.4,)*8),folder,
                max_evaluations=7,population_size=32,initial_local_fraction=.65,local_refine_evaluations=80)
        info=r['metadata']['rounds'][0]
        self.assertEqual(info['population_size'],32)
        self.assertEqual(r['metadata']['field_evaluations'],7)
        self.assertFalse(info['initialization_complete'])
        self.assertEqual(info['local_field_budget_reserved'],0)

    def test_baseline_is_retained_when_all_perturbations_break_ppm_cap(self):
        def field(p):return {'mean_norm_T':.04+sum(p.coords)*.001,'pp_norm_ppm':100+1000*sum(x*x for x in p.coords)}
        with tempfile.TemporaryDirectory() as folder:
            r=run_optimization(variables(4,0),lambda x:Point(tuple(x)),field,lambda d:{'valid':True},Point((0.,)*4),folder,
                seed=5,max_evaluations=35,population_size=8,initial_local_fraction=.65,local_refine_evaluations=10,
                requirements={'min_mean_norm_T':.04,'max_pp_norm_ppm':100.,'field_scale_T':.04,'ppm_scale':100.})
        self.assertEqual(r['diagnostic_best']['candidate_id'],'baseline')
        self.assertTrue(r['baseline']['feasible'])

    def test_repeat_seed_repeats_proposals_and_outcome(self):
        out=[]
        with tempfile.TemporaryDirectory() as folder:
            for i in range(2):
                r=run_optimization(variables(4),lambda x:Point(tuple(x)),metric,lambda d:{'valid':True},Point((.4,)*4),Path(folder)/str(i),
                    seed=19,max_evaluations=35,population_size=8,initial_local_fraction=.65,local_refine_evaluations=10)
                out.append([(row['parameters'],row['metrics']) for row in r['records']])
        self.assertEqual(out[0],out[1])

if __name__=='__main__':unittest.main(verbosity=2)

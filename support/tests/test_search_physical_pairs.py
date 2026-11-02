"""Independent constraint-surface and physical-millimetre search regressions."""
import copy
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
import numpy as np
from parameters import CONFIG
from mri_magnet.geometry import load_source,envelope_limits
from mri_magnet.manufacturing import make_manufacturing_seed,audit_manufacturing
from mri_magnet.parameterization import get_search_space
from mri_magnet.optimization import run_optimization,initialize_population
from mri_magnet.search_moves import reflect_interval,compensated_steps


@dataclass(frozen=True)
class Point:
    x:float=.5
    y:float=.5
    def to_dict(self):return vars(self)


def field(point):
    # At x+y=1 all single-axis ppm improvements violate the field floor;
    # (-h,+h) improves ppm and preserves field exactly.
    return {'mean_norm_T':.04+.001*(point.x+point.y-1.),'pp_norm_ppm':100+point.x+2*point.y}


class PairSearchTests(unittest.TestCase):
    def solve(self,pair_fraction,callback=None,checker=None):
        variables=[dict(name=name,lower=0.,upper=1.,initial=.5,integer=False) for name in ('x','y')]
        calls=[]
        def evaluate(p):
            if checker:self.assertTrue(checker(p)['valid'])
            calls.append(p);return field(p)
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables,lambda x:Point(*x),evaluate,checker or (lambda d:{'valid':True}),
                Point(),folder,seed=6,max_evaluations=65,population_size=5,maxiter=0,
                initial_local_fraction=1.,initial_sigma=1e-14,local_refine_evaluations=60,
                local_step=.05,pair_refine_fraction=pair_fraction,pair_max_trials=12,
                requirements={'min_mean_norm_T':.04,'max_pp_norm_ppm':101.5,'field_scale_T':.04,'ppm_scale':101.5},
                event_callback=callback)
        return result,calls

    def test_pair_step_escapes_single_axis_constraint_surface_stall(self):
        single,_=self.solve(0.)
        paired,calls=self.solve(.4)
        self.assertAlmostEqual(single['diagnostic_best']['metrics']['pp_norm_ppm'],101.5,places=10)
        best=paired['diagnostic_best']
        self.assertLess(best['metrics']['pp_norm_ppm'],101.49)
        self.assertGreaterEqual(best['metrics']['mean_norm_T'],.04)
        self.assertTrue(best['performance_check']['valid'])
        info=paired['metadata']['rounds'][0]['local_refinement']
        self.assertGreater(info['paired_field_evaluations'],0)
        self.assertGreater(info['paired_accepted_moves'],0)
        self.assertLessEqual(info['field_evaluations'],60)
        self.assertLessEqual(len(calls),65)

    def test_stale_secants_are_rejected_after_the_incumbent_changes(self):
        old=[{'origin':[.5,.5],'index':0,'delta':-.05,'delta_field_T':-.00005,'delta_ppm':-.05},
             {'origin':[.5,.5],'index':1,'delta':.05,'delta_field_T':.00005,'delta_ppm':.1}]
        self.assertEqual(list(compensated_steps(np.array([.5,.45]),old,np.zeros(2),np.ones(2),np.zeros(2,bool),12)),[])
        variables=[dict(name=n,lower=0.,upper=1.,initial=.5,integer=False) for n in ('x','y')]
        calls=[]
        def evaluate(p):calls.append(p);return field(p)
        with tempfile.TemporaryDirectory() as folder:
            r=run_optimization(variables,lambda x:Point(*x),evaluate,lambda d:{'valid':True},Point(),folder,
                seed=6,max_evaluations=65,population_size=5,maxiter=0,initial_local_fraction=1.,
                initial_sigma=1e-14,local_refine_evaluations=60,local_step=.05,pair_refine_fraction=.4,
                requirements={'min_mean_norm_T':.03995,'max_pp_norm_ppm':101.5,'field_scale_T':.04,'ppm_scale':101.5})
        info=r['metadata']['rounds'][0]['local_refinement']
        self.assertGreater(info['accepted_moves'],0)
        self.assertGreater(info['paired_stale_probes_discarded'],0)
        self.assertGreater(info['paired_accepted_moves'],0)
        self.assertLessEqual(info['field_evaluations'],60)
        self.assertLessEqual(len(calls),65)
        self.assertGreaterEqual(r['diagnostic_best']['metrics']['mean_norm_T'],.03995)

    def test_stage_counters_are_captured_before_later_stage_costs(self):
        result,_=self.solve(.4)
        stages={s['stage']:s for s in result['metadata']['stage_summaries']}
        self.assertLess(stages['de']['field_evaluations'],stages['local']['field_evaluations'])
        self.assertLess(stages['local']['field_evaluations'],stages['paired']['field_evaluations'])
        self.assertEqual(stages['paired']['field_evaluations'],result['metadata']['field_evaluations'])

    def test_paired_proposals_still_validate_geometry_before_solving(self):
        check=lambda p:{'valid':p.x<=.53,'violations':[] if p.x<=.53 else ['wall']}
        result,calls=self.solve(.4,checker=check)
        self.assertGreater(result['metadata']['infeasible_proposals'],0)
        self.assertTrue(all(p.x<=.53 for p in calls))
        self.assertLessEqual(len(calls),65)

    def test_callback_has_actual_design_stages_and_no_extra_field_cost(self):
        events=[]
        with_callback,calls=self.solve(.4,events.append)
        without_callback,other_calls=self.solve(.4)
        self.assertEqual(len(calls),len(other_calls))
        self.assertEqual(with_callback['diagnostic_best']['metrics'],without_callback['diagnostic_best']['metrics'])
        self.assertTrue(any(e['event']=='baseline' for e in events))
        self.assertTrue(any(e['phase']=='paired' for e in events))
        self.assertTrue(all({'design','metrics','candidate_id','global_evaluations','topology_index'}<=set(e) for e in events))
        self.assertEqual({s['stage'] for s in with_callback['metadata']['stage_summaries']},
            {'initialization','de','local','paired'})
        self.assertTrue(all('search_stage' in row and 'field_evaluations' in row for row in with_callback['records']))

    def test_generation_events_contain_true_cached_population_without_resolving_fields(self):
        events=[];calls=[]
        def evaluate(p):calls.append(p);return field(p)
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization([dict(name=n,lower=0.,upper=1.,initial=.5,integer=False) for n in ('x','y')],
                lambda x:Point(*x),evaluate,lambda p:{'valid':True},Point(),folder,
                population_size=5,maxiter=2,max_evaluations=30,event_callback=events.append)
        generations=[e for e in events if e['event']=='generation']
        self.assertEqual(len(generations),2)
        self.assertTrue(all(len(e['population_metrics'])==5 for e in generations))
        self.assertTrue(all(e['population_unresolved']==0 for e in generations))
        self.assertEqual(len(calls),result['metadata']['field_evaluations'])

    def test_bad_monitor_is_disabled_and_does_not_change_numeric_search(self):
        count=[]
        def broken(event):count.append(event);raise OSError('preview not writable')
        result,calls=self.solve(.4,broken)
        self.assertEqual(len(count),1)
        self.assertEqual(len(result['metadata']['monitor_errors']),1)
        self.assertTrue(result['diagnostic_best']['feasible'])

    def test_reflection_does_not_put_half_the_population_on_hard_boundary(self):
        rng=np.random.default_rng(9)
        values=reflect_interval(rng.normal(0,.1,10000),0,1)
        self.assertEqual(np.count_nonzero(values==0),0)
        self.assertEqual(np.count_nonzero(values==1),0)
        self.assertTrue(np.all((values>0)&(values<1)))


class PhysicalSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        path=Path(os.environ.get('MRI_SOURCE',str(root/'examples'/'seeded'/'placement.json')))
        if not path.is_file():raise unittest.SkipTest('Actual source geometry unavailable')
        source,_=load_source(path)
        cls.process=copy.deepcopy(CONFIG['manufacturing'])
        cls.limits=dict(envelope_limits(source),preserve_exact_bore=False)
        cls.seed,_=make_manufacturing_seed(source,cls.process,cls.limits)
        cls.variables,builder=get_search_space(cls.seed,cls.process,cls.limits,CONFIG['search'])
        cls.builder=staticmethod(builder)

    def test_physical_mm_perturbations_remove_tenfold_layer_scale_jump(self):
        population=self.builder.local_sampler(np.random.default_rng(51),120,.8)
        stats=self.builder.local_sampler.last_statistics
        five=[r['rms_delta_R_mm'] for r in stats['radial_pairs'] if r['ring_count']==5]
        three=[r['rms_delta_R_mm'] for r in stats['radial_pairs'] if r['ring_count']==3]
        self.assertTrue(five and three)
        self.assertLess(np.mean(three)/np.mean(five),3.)
        self.assertLess(max(three),1.2)
        for row in population[:6]:
            design=self.builder(row)
            self.assertTrue(audit_manufacturing(design,self.process,self.limits)['valid'])

    def test_physical_population_is_reproducible_and_keeps_local_global_split(self):
        first=initialize_population(self.variables,np.random.default_rng(77),32,self.builder,.65,.06,'physical_mm',.8)
        second=initialize_population(self.variables,np.random.default_rng(77),32,self.builder,.65,.06,'physical_mm',.8)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(self.builder.local_sampler.last_statistics['sample_count'],20)


if __name__=='__main__':unittest.main()

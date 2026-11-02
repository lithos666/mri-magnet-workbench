"""Mechanism tests for elite continuation, constrained LP steps and safe stops."""
from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from mri_magnet.active_set import active_set_polish
from mri_magnet.metrics import field_metrics
from mri_magnet.optimization import run_optimization
from mri_magnet.islands import run_islands
from test_islands import seed,PROCESS,LIMITS,SEARCH,REQUIREMENTS,Checker,Evaluator,settings


@dataclass(frozen=True)
class Point:
    coords:tuple
    def to_dict(self):return {'coordinates':list(self.coords)}


class Builder:
    def __init__(self):self.centers=[];self.metadata={'original_windows':[0.,1.]}
    def __call__(self,x):return Point(tuple(x))
    def encode_design(self,d):return np.asarray(d.coords)
    def local_sampler(self,rng,size,sigma_mm,center_design=None):
        center=np.array((.25,.25) if center_design is None else center_design.coords)
        self.centers.append(tuple(center));return np.tile(center,(size,1))


class VectorEvaluator:
    def __init__(self):self.calls=0
    def field_values(self,d):
        self.calls+=1;x,y=d.coords;mean=.04+.001*(y-x);amplitude=.001-.002*x
        return np.array([[0.,0.,mean-amplitude],[0.,0.,mean+amplitude]])
    def evaluate_with_field(self,d):
        b=self.field_values(d);return field_metrics(b,[.5,.5]),b
    def __call__(self,d):return self.evaluate_with_field(d)[0]


@dataclass
class StopFileEvaluator:
    stop_file:str
    calls:int=0
    def __call__(self,d):
        self.calls+=1
        if self.calls>=4:Path(self.stop_file).write_text('stop',encoding='utf-8')
        return {'mean_norm_T':.04,'pp_norm_ppm':50.+sum((r.radius_mm-58.)**2 for r in d.rings)*.01}


def variables():return [dict(name='x',lower=0.,upper=1.,initial=.25),dict(name='y',lower=0.,upper=1.,initial=.25)]


class V3SolverTests(unittest.TestCase):
    def test_lp_couples_coordinates_at_hard_field_boundary_and_counts_each_field_once(self):
        evaluate=VectorEvaluator();builder=Builder()
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables(),builder,evaluate,lambda d:{'valid':True},Point((.25,.25)),folder,
                max_evaluations=20,population_size=5,maxiter=0,initial_local_fraction=1.,initial_sigma=1e-12,
                active_set_evaluations=8,active_set_trust_radius=.1,field_values=evaluate.field_values,field_weights=np.array([.5,.5]),
                requirements={'min_mean_norm_T':.04,'max_pp_norm_ppm':30000.,'field_scale_T':.04})
            info=result['metadata']['rounds'][0]['active_set_polish']
        self.assertEqual(evaluate.calls,result['metadata']['field_evaluations'])
        self.assertLessEqual(evaluate.calls,20);self.assertEqual(info['jacobian_columns'],2)
        self.assertEqual(info['accepted_steps'],1)
        self.assertLess(result['diagnostic_best']['metrics']['pp_norm_ppm'],20000.)
        self.assertGreaterEqual(result['diagnostic_best']['metrics']['mean_norm_T'],.04)

    def test_lp_skips_incomplete_jacobian_without_hidden_field_calls(self):
        calls=[]
        _,info=active_set_polish(np.array([.2,.2]),np.zeros(2),np.ones(2),np.zeros(2,bool),
            lambda x:calls.append(x),lambda:3)
        self.assertEqual(calls,[]);self.assertEqual(info['state'],'insufficient_budget_for_full_jacobian')

    def test_lp_nonlinear_predictions_are_rejected_when_actual_score_is_worse(self):
        origin=np.array([.25,.25]);calls=[]
        def query(x):
            calls.append(np.array(x));is_probe=np.linalg.norm(x-origin)<.01
            x0,x1=x;mean=.04+.001*(x1-x0);amplitude=.001-.002*x0
            b=np.array([[0.,0.,mean-amplitude],[0.,0.,mean+amplitude]])
            return {'fields':b,'metrics':field_metrics(b,[.5,.5]),'score':1. if is_probe else 2.}
        x,info=active_set_polish(origin,np.zeros(2),np.ones(2),np.zeros(2,bool),query,lambda:20,
            weights=[.5,.5],requirements={'min_mean_norm_T':.04,'max_pp_norm_ppm':30000.},trust_radius=.1)
        np.testing.assert_array_equal(x,origin);self.assertEqual(info['accepted_steps'],0)

    def test_continuation_uses_elite_row_zero_and_all_local_rows_without_moving_builder_windows(self):
        builder=Builder();evaluator=VectorEvaluator()
        with tempfile.TemporaryDirectory() as folder:
            run_optimization(variables(),builder,evaluator,lambda d:{'valid':True},Point((.25,.25)),folder,
                initial_design=Point((.6,.6)),population_size=6,max_evaluations=8,maxiter=0,
                initial_local_fraction=1.,initial_scale_mode='physical_mm')
            population=json.loads((Path(folder)/'initial_population.json').read_text(encoding='utf-8'))
        np.testing.assert_allclose(population['population'],np.full((6,2),.6))
        self.assertTrue(all(center==(.6,.6) for center in builder.centers))
        self.assertEqual(builder.metadata['original_windows'],[0.,1.])

    def test_graceful_stop_saves_all_completed_candidates_and_no_new_calls(self):
        calls=[]
        def evaluate(d):calls.append(d);return {'mean_norm_T':.04,'pp_norm_ppm':100.+sum(d.coords)}
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables(),Builder(),evaluate,lambda d:{'valid':True},Point((.25,.25)),folder,
                max_evaluations=50,population_size=8,graceful_stop=True,should_stop=lambda:len(calls)>=3)
            saved=json.loads((Path(folder)/'optimization_result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['state'],'stopped_with_result');self.assertEqual(saved['state'],result['state'])
        self.assertEqual(len(calls),3);self.assertEqual(len(result['records']),3)
        self.assertEqual(result['metadata']['field_evaluations'],3)

    def test_stop_before_first_field_is_honest_no_result(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables(),Builder(),lambda d:(_ for _ in ()).throw(AssertionError('field forbidden')),
                lambda d:{'valid':True},Point((.25,.25)),folder,max_evaluations=50,graceful_stop=True,should_stop=lambda:True)
        self.assertEqual(result['state'],'stopped_no_result');self.assertEqual(result['records'],[])
        self.assertEqual(result['metadata']['field_evaluations'],0)

    def test_parallel_workers_return_every_completed_record_on_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            flag=Path(folder)/'stop.flag';config=settings(80,2)
            config['islands'].update(graceful_stop=True,stop_file=str(flag))
            result=run_islands([seed(24,5),seed(32,8)],PROCESS,LIMITS,SEARCH,
                StopFileEvaluator(str(flag)),Checker(),folder,config,REQUIREMENTS)
            saved=json.loads((Path(folder)/'optimization_result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['state'],'stopped_with_result');self.assertEqual(saved['state'],result['state'])
        self.assertGreater(len(result['records']),0);self.assertLessEqual(result['metadata']['field_evaluations'],8)
        self.assertEqual(len(result['records']),result['metadata']['field_evaluations'])
        self.assertTrue(result['metadata']['stopped_with_completed_records'])

    def test_islands_stop_before_baseline_returns_no_result_without_empty_round_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            flag=Path(folder)/'stop.flag';flag.write_text('stop',encoding='utf-8')
            config=settings(80,1);config['islands'].update(graceful_stop=True,stop_file=str(flag))
            evaluator=Evaluator()
            result=run_islands([seed(24,5),seed(32,8)],PROCESS,LIMITS,SEARCH,
                evaluator,Checker(),folder,config,REQUIREMENTS)
        self.assertEqual(result['state'],'stopped_no_result')
        self.assertEqual(result['records'],[]);self.assertEqual(evaluator.calls,0)
        self.assertEqual(result['metadata']['field_evaluations'],0)
        self.assertEqual(result['metadata']['topologies_run'],0)

    def test_island_stop_after_baseline_before_population_saves_baseline_and_actual_topology_count(self):
        with tempfile.TemporaryDirectory() as folder:
            flag=Path(folder)/'stop.flag';config=settings(80,1)
            config['islands'].update(graceful_stop=True,stop_file=str(flag))
            def stop_on_baseline(event):
                if event.get('event')=='baseline':flag.write_text('stop',encoding='utf-8')
            result=run_islands([seed(24,5),seed(32,8)],PROCESS,LIMITS,SEARCH,
                Evaluator(),Checker(),folder,config,REQUIREMENTS,event_callback=stop_on_baseline)
        self.assertEqual(result['state'],'stopped_with_result')
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['metadata']['field_evaluations'],1)
        self.assertEqual(result['metadata']['topologies_run'],1)
        self.assertEqual(result['metadata']['island_rounds'][0]['promoted_islands'],[])

    def test_island_round_two_initial_center_is_round_one_measured_incumbent(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_islands([seed(24,5),seed(28,6),seed(32,8)],PROCESS,LIMITS,SEARCH,
                Evaluator(),Checker(),folder,settings(100),REQUIREMENTS)
            for item in result['metadata']['runs']:
                if item['island_round']!=2:continue
                first=next(r for r in result['metadata']['runs'] if r['topology']==item['topology'] and r['island_round']==1)
                previous=json.loads((Path(folder)/first['relative_result']).read_text(encoding='utf-8'))['diagnostic_best']['design']
                self.assertEqual(item['metadata']['initial_center_design'],previous)
                self.assertTrue(item['metadata']['original_parameter_windows_retained'])


if __name__=='__main__':unittest.main()

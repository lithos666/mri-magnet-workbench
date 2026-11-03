"""Coupled physics must control ranking, LP queries and persistent workflow."""
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from mri_magnet.config_schema import canonicalize_config
from mri_magnet.workbench_config import default_config
from mri_magnet.workbench import Workbench,FieldEvaluator,ManufacturingChecker,envelope_for,requirements_for,solver_settings_for
from mri_magnet.workbench_domains import make_sample
from mri_magnet.geometry import Design,RingSpec
from mri_magnet.physics_workflow import evaluate_candidates
from mri_magnet.islands import run_islands


def design(radius=40.):
    return Design((RingSpec('R01-L1',0,0,radius,25.,8,'out',0.),
                   RingSpec('R02-L1',1,0,radius,-25.,8,'in',0.)),cube_mm=4.)


def config():
    cfg=default_config()
    cfg.update(magnet={'cube_mm':4.,'remanence_T':1.43,'reference_mur':1.05},
        target={'geometry':'sphere','diameter_mm':10.,'z_mm':0.,'min_field_mT':.001,'max_ppm':None},
        envelope={'protected_bore_diameter_mm':20.,'max_outer_diameter_mm':120.,'max_length_mm':100.})
    cfg['manufacturing'].update(enabled=False,max_magnet_count=1000)
    cfg['sampling'].update(training_volume=8,training_surface=12,validation_volume=16,validation_surface=24,refine_starts=1)
    cfg['search'].update(optimize_radius=True,optimize_axial_positions=True,axial_shift_limit_mm=3.)
    cfg['solver'].update(max_evaluations=80,population_size=5,local_refine_evaluations=12,
        active_set_evaluations=12,budget_policy='allow_partial',max_workers=1)
    cfg['evolution']['enabled']=False
    return cfg


class CoupledWorkflowTests(unittest.TestCase):
    def test_new_defaults_coupled_and_old_files_remain_reproducible(self):
        self.assertEqual(default_config()['physics']['field_model'],'coupled')
        self.assertEqual(default_config()['deliverables']['air_radius_mm'],3000.)
        old=canonicalize_config({'schema_version':3,'polarization_model':'isolated_cube_self_demag'})
        self.assertEqual(old['physics']['field_model'],'legacy')
        self.assertEqual(old['deliverables']['air_radius_mm'],650.)
        partial=canonicalize_config({'solver':{'max_evaluations':1200}},base=default_config())
        self.assertEqual(partial['physics']['field_model'],'coupled')

    def test_full_evaluator_and_lp_oracle_use_identical_coupled_fields(self):
        evaluator=FieldEvaluator(config(),make_sample(config(),designs=[design()]))
        metrics,fields=evaluator.evaluate_with_field(design())
        self.assertEqual(metrics['field_model'],'coupled_uniform_cells')
        np.testing.assert_allclose(fields,evaluator.field_values(design()),rtol=0,atol=1e-14)
        self.assertEqual(evaluator.engine.statistics['preparations'],1)
        moved=design(41.)
        evaluator.field_values(moved)
        self.assertEqual(evaluator.engine.statistics['preparations'],2)

    def test_official_seed_ranking_can_reverse_proposal_ranking(self):
        cfg=config();sample=make_sample(cfg)
        rows=[{'id':'a','style':'one','design':design(38.).to_dict(),'metrics':{'mean_norm_T':.04,'pp_norm_ppm':1.}},
              {'id':'b','style':'two','design':design(42.).to_dict(),'metrics':{'mean_norm_T':.04,'pp_norm_ppm':1000.}}]
        class Engine:
            polarization_model='coupled_uniform_cells'
            statistics={}
        def compute(d,engine,points,density):
            return {'mean_norm_T':.04,'pp_norm_ppm':1000. if d.rings[0].radius_mm<40 else 1.},None
        req={'min_mean_norm_T':.03,'max_pp_norm_ppm':None,'ppm_scale':1000.}
        with patch('mri_magnet.physics_workflow.evaluate_design',side_effect=compute):
            chosen,evaluated=evaluate_candidates(rows,cfg,sample,req,Engine(),limit=1)
        self.assertEqual(chosen[0]['id'],'b')
        self.assertEqual(evaluated[0]['proposal_metrics']['pp_norm_ppm'],1.)

    def test_multistage_island_returns_only_coupled_metrics_and_real_lp(self):
        cfg=config();sample=make_sample(cfg,designs=[design()]);evaluator=FieldEvaluator(cfg,sample)
        events=[]
        with tempfile.TemporaryDirectory() as folder:
            result=run_islands([design()],cfg['manufacturing'],envelope_for(cfg),cfg['search'],evaluator,
                ManufacturingChecker(cfg['manufacturing'],envelope_for(cfg)),folder,solver_settings_for(cfg),requirements_for(cfg),events.append)
        self.assertTrue(result['records'])
        self.assertTrue(all(row['metrics']['field_model']=='coupled_uniform_cells' for row in result['records']))
        self.assertTrue(all(row['evaluation_fingerprint']==evaluator.evaluation_fingerprint for row in result['records']))
        self.assertLessEqual(result['metadata']['field_evaluations'],80)
        stages=result['metadata']['runs'][0]['metadata']['rounds'][0]
        self.assertGreater(stages['active_set_polish'].get('jacobian_columns',0),0)
        self.assertTrue(any(e.get('prepared_response') for e in events if e.get('event')=='new_best'))

    def test_shared_numeric_cache_does_not_mix_physics(self):
        from mri_magnet.optimization import run_optimization
        from mri_magnet.parameterization import get_search_space
        cfg=config();d=design();sample=make_sample(cfg,designs=[d]);cache={}
        variables,builder=get_search_space(d,cfg['manufacturing'],envelope_for(cfg),cfg['search'])
        checker=ManufacturingChecker(cfg['manufacturing'],envelope_for(cfg))
        with tempfile.TemporaryDirectory() as folder:
            for index,mode in enumerate(('legacy','coupled')):
                current=deepcopy(cfg);current['physics']['field_model']=mode
                evaluator=FieldEvaluator(current,sample)
                result=run_optimization(variables,builder,evaluator,checker,d,Path(folder)/str(index),
                    max_evaluations=10,population_size=5,maxiter=1,numeric_cache=cache)
                self.assertGreater(result['metadata']['field_evaluations'],0)
                self.assertEqual(result['baseline']['metrics']['field_model'],
                    'coupled_uniform_cells' if mode=='coupled' else cfg['polarization_model'])
        self.assertTrue(any(':' in key for key in cache))

    def test_model_change_invalidates_cached_results_and_fem_status(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(config())
            wb.data.update(best={'design':design().to_dict()},physics_validation={'fem_verified':True},
                physics_calibration={'model_accuracy_passed':True})
            wb.update_config({'physics':{'quadrature_order':3}})
            for key in ('best','physics_validation','physics_calibration'):self.assertNotIn(key,wb.data)

    def test_export_and_fem_settings_together_preserve_completed_optimization(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(config())
            row={'design':design().to_dict(),'metrics':{'mean_norm_T':.04,'pp_norm_ppm':12.},
                'physics_validation':{'fem_verified':True}}
            wb.data.update(best=deepcopy(row),current_best=deepcopy(row),seeds=[{'id':'known'}])
            for step in wb.data['steps']:step['status']='passed'
            wb.update_config({'deliverables':{'air_radius_mm':4500.},'fem_validation':{'air_radius_mm':4500.}})
            self.assertEqual(wb.data['best']['metrics'],row['metrics'])
            self.assertEqual(wb.data['seeds'],[{'id':'known'}])
            self.assertTrue(all(step['status']=='passed' for step in wb.data['steps'][:5]))
            self.assertEqual(wb.data['steps'][5]['status'],'pending')
            self.assertNotIn('physics_validation',wb.data['best'])

if __name__=='__main__':unittest.main()

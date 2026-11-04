"""Search, dense reports and exports share one non-solving ranking contract."""
from copy import deepcopy
from dataclasses import dataclass
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import differential_evolution as scipy_de

from mri_magnet import objectives, optimization, reporting
from mri_magnet.deliverables import selected_candidates, finalize_results, export_candidates
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.parameterization import get_search_space
from mri_magnet.solver_budget import plan_island_budgets
from mri_magnet.workbench import requirements_for, envelope_for, solver_settings_for
from mri_magnet.workbench_archive import _requirements, archive_run
from mri_magnet.workbench_config import default_config


def metric(field=.05, ppm=100., pointset='dense'):
    return {'mean_norm_T':field,'pp_norm_ppm':ppm,'magnet_count':48,
            'mass_magnet_kg':.1,'pointset':pointset,
            'min_norm_T':field*(1-ppm/2e6),'max_norm_T':field*(1+ppm/2e6)}


def design():
    return Design((RingSpec('R01-L1',0,0,140.,30.,24,'out',-82.5),
                   RingSpec('R02-L1',1,0,140.,-30.,24,'in',-82.5)),cube_mm=4.)


def row(ident, training, dense):
    return {'candidate_id':ident,'design':design().to_dict(),'metrics':training,
            'validation_metrics':dense,'feasible':True,'geometry_feasible':True,
            'geometry_check':{'valid':True},'manufacturing_check':{'valid':True},
            'performance_check':{'valid':True}}


def reversal():
    # A dominates on training samples; B dominates after independent validation.
    return [row('A',metric(.06,10.,'training'),metric(.05,900.)),
            row('B',metric(.05,400.,'training'),metric(.055,100.))]


@dataclass(frozen=True)
class Point:
    values: tuple
    def to_dict(self):return {'values':self.values}


class ParetoContractTests(unittest.TestCase):
    def test_historical_imports_are_aliases_of_the_same_implementation(self):
        self.assertIs(optimization.pareto_front,objectives.pareto_front)
        self.assertIs(requirements_for,objectives.requirements_for)
        self.assertIs(_requirements,objectives.requirements_for)

    def test_dense_reversal_is_identical_in_solver_report_and_deliverables(self):
        rows=reversal();req={'min_mean_norm_T':.04,'max_pp_norm_ppm':1000.}
        for select in (objectives.pareto_front,optimization.pareto_front):
            self.assertEqual([r['candidate_id'] for r in select(rows,req)],['B'])
        self.assertEqual(reporting._pareto_flags(rows,req).tolist(),[False,True])
        payload={'candidates':rows,'requirements':req,'manufacturing':{'config':{}}}
        self.assertEqual([r['candidate_id'] for r in selected_candidates(payload)],['B'])
        displayed=reporting._candidate_rows(payload)
        self.assertEqual(displayed[0]['metrics'],rows[0]['validation_metrics'])
        self.assertEqual(displayed[1]['metrics'],rows[1]['validation_metrics'])
        self.assertEqual(rows[0]['metrics']['pointset'],'training')

    def test_validation_then_dense_then_training_precedence(self):
        winner=row('winner',metric(.08,0.),metric(.05,20.))
        winner['dense_metrics']=metric(.02,100000.)
        self.assertIs(objectives.record_metrics(winner),winner['validation_metrics'])
        del winner['validation_metrics']
        self.assertIs(objectives.record_metrics(winner),winner['dense_metrics'])
        del winner['dense_metrics']
        self.assertIs(objectives.record_metrics(winner),winner['metrics'])

    def test_invalid_metrics_and_explicit_failed_reviews_never_enter_front(self):
        valid=row('valid',metric(),metric())
        bad=[]
        for key in ('feasible','geometry_feasible'):
            item=row(key,metric(1.,0.),metric(1.,0.));item[key]=False;bad.append(item)
        for key in ('performance_check','geometry_check','manufacturing_check','manufacturing_audit','audit'):
            item=row(key,metric(1.,0.),metric(1.,0.));item[key]={'valid':False};bad.append(item)
        for field,ppm in ((float('nan'),0.),(1.,float('inf')),(0.,0.),(1.,-1.)):
            item=row(str((field,ppm)),metric(),{'mean_norm_T':field,'pp_norm_ppm':ppm});bad.append(item)
        bad.append(row('missing',metric(),{'mean_norm_T':1.}))
        rows=[valid]+bad
        self.assertEqual(objectives.pareto_front(rows),[valid])
        self.assertEqual(reporting._pareto_flags(rows).tolist(),[True]+[False]*len(bad))
        self.assertEqual(objectives.pareto_roles(bad),{})
        self.assertNotIn('missing',[r['candidate_id'] for r in objectives.select_validation_candidates(rows,limit=30)])

    def test_physical_duplicate_policy_is_shared_but_distinct_equal_ties_remain(self):
        first=row('first',metric(),metric());first['canonical_physical_sha256']='one-physical-design'
        duplicate=deepcopy(first);duplicate['candidate_id']='duplicate'
        separate=deepcopy(first);separate['candidate_id']='separate';separate['canonical_physical_sha256']='second-design'
        self.assertEqual({r['candidate_id'] for r in objectives.pareto_front([first,duplicate,separate])},{'first','separate'})

    def test_physical_alias_prefers_dense_review_even_when_training_appears_first(self):
        training={'candidate_id':'training_alias','canonical_physical_sha256':'same',
                  'metrics':metric(.08,0.),'feasible':True}
        dense=row('dense_alias',metric(.08,0.,'training'),metric(.05,900.))
        dense['canonical_physical_sha256']='same'
        peer=row('peer',metric(.06,1000.),metric(.055,100.))
        for records in ([training,dense,peer],[dense,training,peer]):
            self.assertEqual(objectives.pareto_front(records),[peer])
            self.assertNotIn(training,objectives.select_validation_candidates(records,limit=3))

    def test_failed_dense_review_cannot_fall_back_to_passing_training_alias(self):
        training={'candidate_id':'training_alias','canonical_physical_sha256':'same',
                  'metrics':metric(.08,0.),'feasible':True}
        dense=row('dense_alias',metric(.08,0.,'training'),metric(.039,5.))
        dense['canonical_physical_sha256']='same'
        self.assertEqual(objectives.pareto_front([training,dense],{'min_mean_norm_T':.04}),[])
        dense['validation_metrics']['pp_norm_ppm']=float('nan')
        self.assertEqual(objectives.pareto_front([training,dense]),[])

    def test_archive_cannot_resurrect_training_when_invalid_dense_alias_is_dropped(self):
        cfg=default_config();cfg['target'].update(min_field_mT=40.,max_ppm=None)
        training={'candidate_id':'training_alias','canonical_physical_sha256':'same',
                  'design':design().to_dict(),'metrics':metric(.08,0.),'feasible':True}
        invalid=row('invalid_dense',metric(.08,0.,'training'),metric(.05,float('nan')))
        invalid['canonical_physical_sha256']='same';invalid['feasible']=False
        with tempfile.TemporaryDirectory() as folder:
            with patch('mri_magnet.workbench_archive.render_rz_layout') as render:
                result=archive_run(folder,cfg,[],[training,invalid],{},None,export_comsol=False)
            render.assert_not_called()
            self.assertIsNone(result['best'])
            self.assertEqual(result['pareto_recommendations'],[])
            manifest=json.loads((Path(folder)/'归档清单.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['pareto_front_ids'],[])

    def test_export_cannot_resurrect_lower_scope_when_failed_dense_alias_is_filtered(self):
        earlier=row('earlier_alias',metric(.08,0.),None)
        earlier['dense_metrics']=metric(.08,0.);earlier['canonical_physical_sha256']='same'
        failed=row('final_alias',metric(.08,0.),metric(.039,10.))
        failed.update(canonical_physical_sha256='same',feasible=False,performance_check={'valid':False})
        payload={'candidates':[earlier,failed],'requirements':{'min_mean_norm_T':.04}}
        self.assertEqual(selected_candidates(payload),[])
        with tempfile.TemporaryDirectory() as folder:
            with patch('mri_magnet.deliverables.render_rz_layout') as render, \
                 patch('mri_magnet.deliverables.export_comsol_model') as export:
                result=export_candidates(folder,Path(folder)/'details',payload,
                    {'comsol_root':'not_used','results':{'diagnostic_exports':False}},log=lambda _:None)
            render.assert_not_called();export.assert_not_called()
            self.assertEqual(result['pareto_front_ids'],[])
            self.assertEqual(result['candidates'],[])

    def test_training_failure_does_not_override_a_passed_dense_review(self):
        checked=row('dense_pass',metric(.039,2000.,'training'),metric(.045,100.))
        checked['training_performance_check']={'valid':False}
        self.assertEqual(objectives.pareto_front([checked],{'min_mean_norm_T':.04}),[checked])

    def test_stale_passed_training_gate_cannot_label_a_failed_dense_row_as_passed(self):
        checked=row('failed_dense',metric(.05,10.,'training'),metric(.039,10.))
        checked['validation_status']={'label':'双门槛通过','dense_pass':True}
        payload={'candidates':[checked],'requirements':{'min_mean_norm_T':.04}}
        displayed=reporting._candidate_rows(payload)
        self.assertFalse(displayed[0]['performance_check']['valid'])
        self.assertEqual(displayed[0]['validation_status']['label'],'独立复核未通过')
        self.assertFalse(reporting._pareto_flags(displayed,payload['requirements'])[0])
        self.assertTrue(checked['performance_check']['valid'])

    def test_reexport_manifest_csv_report_and_summary_share_dense_front_and_values(self):
        rows=reversal();payload={'candidates':rows,'baseline':rows[0],
            'manufacturing':{'config':{}},'requirements':{'min_mean_norm_T':.04,'max_pp_norm_ppm':1000.},
            'verified_pareto_ids':['A']}
        def layout(value,path,**kwargs):path.write_bytes(b'layout');return path
        def export(value,target,comsol,settings):
            java=target/'checked.java';java.write_text('class checked {}',encoding='ascii')
            return {'status':'java_ready','compiled':False,'java_file':str(java),'solve_executed':False}
        def report(public,content,details_dir=None):
            self.assertEqual(content['verified_pareto_ids'],['B'])
            self.assertEqual(reporting._pareto_flags(reporting._candidate_rows(content),content['requirements']).tolist(),[False,True])
            return {'report_html':str(public/'report_中文.html')}
        with tempfile.TemporaryDirectory() as folder:
            public=Path(folder);details=public/'详细数据';details.mkdir()
            (details/'run_summary.json').write_text(json.dumps({'verified_pareto_ids':['A']}),encoding='utf-8')
            cfg={'comsol_root':'not_used','deliverables':{'compile':False},'results':{'overview':False}}
            with patch('mri_magnet.deliverables.render_rz_layout',side_effect=layout) as render, \
                 patch('mri_magnet.deliverables.export_comsol_model',side_effect=export), \
                 patch('mri_magnet.reporting.create_report',side_effect=report):
                _,success=finalize_results(public,details,payload,cfg,log=lambda _:None)
            self.assertTrue(success)
            manifest=json.loads((details/'deliverables_manifest.json').read_text(encoding='utf-8'))
            summary=json.loads((details/'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['pareto_front_ids'],['B'])
            self.assertEqual(summary['verified_pareto_ids'],['B'])
            self.assertEqual(manifest['candidates'][0]['metrics'],rows[1]['validation_metrics'])
            self.assertTrue(manifest['candidates'][0]['is_pareto'])
            self.assertIn('55.000 mT / 100.000 ppm',render.call_args.kwargs['title'])
            with (public/'方案对比.csv').open(encoding='utf-8-sig',newline='') as stream:
                table={r['方案编号']:r for r in csv.DictReader(stream)}
            self.assertEqual(table['A']['可行帕累托候选'],'False')
            self.assertEqual(table['B']['可行帕累托候选'],'True')
            self.assertAlmostEqual(float(table['B']['平均场_mT']),55.)
            self.assertAlmostEqual(float(table['B']['峰峰值_ppm']),100.)


class RequirementsContractTests(unittest.TestCase):
    def test_nondefault_ppm_scale_is_the_same_in_training_and_archive_contract(self):
        cfg=default_config();cfg['target'].update(min_field_mT=40.,max_ppm=None)
        for mode in ('ppm','balanced'):
            cfg['scoring'].update(mode=mode,ppm_scale=37.)
            self.assertEqual(requirements_for(cfg),_requirements(cfg))
            self.assertEqual(requirements_for(cfg)['ppm_scale'],37.)
            value=metric(.044,370.)
            self.assertEqual(objectives.feasibility_score(value,requirements_for(cfg)),
                             objectives.feasibility_score(value,_requirements(cfg)))

    def test_custom_balanced_scale_changes_the_recommendation_and_step6_preserves_it(self):
        cfg=default_config();cfg['target'].update(min_field_mT=40.,max_ppm=None)
        cfg['scoring'].update(mode='balanced',ppm_scale=250.)
        rows=[{'candidate_id':name,'design':design().to_dict(),'metrics':metric(field,ppm)}
              for name,field,ppm in (('low',.04,10.),('tradeoff',.044,300.),('high',.08,1000000.))]
        expected=objectives.pareto_roles(rows,requirements_for(cfg))['balanced']['candidate_id']
        self.assertEqual(expected,'low')
        default=deepcopy(cfg);default['scoring']['ppm_scale']=1000.
        self.assertEqual(objectives.pareto_roles(rows,requirements_for(default))['balanced']['candidate_id'],'tradeoff')
        def layout(value,path,**kwargs):path.write_bytes(b'layout');return path
        def pareto(*args):args[-1].write_bytes(b'pareto')
        with tempfile.TemporaryDirectory() as folder:
            with patch('mri_magnet.workbench_archive.render_rz_layout',side_effect=layout), \
                 patch('mri_magnet.workbench_archive._render_pareto',side_effect=pareto):
                result=archive_run(folder,cfg,[],rows,{},None,export_comsol=False)
            self.assertEqual(result['best']['candidate_id'],expected)
            self.assertEqual(result['requirements'],requirements_for(cfg))
            self.assertEqual(next(r for r in result['pareto_recommendations'] if r['role']=='balanced')['candidate_id'],expected)

    def test_old_loaded_objectives_can_import_new_archivers_without_reload(self):
        # Freeze the actual pre-contract module as data: installed packages do
        # not need git history, and no second production Pareto is introduced.
        fixture=Path(__file__).parent/'fixtures'/'objectives_v6_1f51d7c.py.txt'
        code=r'''
from pathlib import Path
import importlib,json,math,sys,types
import mri_magnet
legacy=types.ModuleType('mri_magnet.objectives');legacy.__package__='mri_magnet'
exec(compile(Path(sys.argv[1]).read_text(encoding='utf-8'),'frozen_legacy_objectives','exec'),legacy.__dict__)
sys.modules['mri_magnet.objectives']=legacy;mri_magnet.objectives=legacy
assert not hasattr(legacy,'record_metrics') and not hasattr(legacy,'requirements_for')
old_score=legacy.feasibility_score
config={'target':{'min_field_mT':40.,'max_ppm':None},
        'scoring':{'mode':'balanced','field_weight':.5,'field_gain_scale':.1,'ppm_scale':1000.}}
old_requirements={'min_mean_norm_T':.04,'max_pp_norm_ppm':None,'field_tolerance_T':0.,
        'field_scale_T':.04,'ppm_scale':1000.,'scoring':dict(config['scoring'])}
measurements=[{'mean_norm_T':field,'pp_norm_ppm':ppm} for field in (.039,.04,.044,.08) for ppm in (0.,10.,300.,1e200)]
before=[old_score(m,old_requirements) for m in measurements]
archive=importlib.import_module('mri_magnet.workbench_archive')
deliverables=importlib.import_module('mri_magnet.deliverables')
reporting=importlib.import_module('mri_magnet.reporting')
assert sys.modules['mri_magnet.objectives'] is legacy
assert legacy.feasibility_score is old_score
assert [old_score(m,archive._requirements(config)) for m in measurements]==before
good={'candidate_id':'good','metrics':{'mean_norm_T':.045,'pp_norm_ppm':100.},'feasible':True}
bad={'candidate_id':'bad','metrics':{'mean_norm_T':.08,'pp_norm_ppm':0.},'geometry_feasible':False}
assert reporting._pareto_flags([good,bad],old_requirements).tolist()==[True,False]
assert archive._requirements.__module__=='mri_magnet.design_contract'
print(json.dumps({'old_module_identity_preserved':True,'old_score_identity_preserved':True,'score_values_bitwise_equal':True}))
'''
        result=subprocess.run([sys.executable,'-c',code,str(fixture)],capture_output=True,text=True,
            encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'),timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['old_score_identity_preserved'])


class PopulationContractTests(unittest.TestCase):
    def test_workbench_budget_and_scipy_init_have_one_explicit_population_authority(self):
        cfg=default_config();cfg['solver'].update(population_size=7,max_evaluations=24,
            maxiter=0,local_refine_evaluations=0,active_set_evaluations=0)
        settings=solver_settings_for(cfg)
        self.assertNotIn('popsize',settings)
        model=design();limits=envelope_for(cfg)
        variables,builder=get_search_space(model,cfg['manufacturing'],limits,cfg['search'])
        plan=plan_island_budgets([model],cfg['manufacturing'],limits,cfg['search'],settings,requirements_for(cfg))
        self.assertTrue(plan['valid']);self.assertEqual(plan['population_size'],7)
        with tempfile.TemporaryDirectory() as folder, patch('mri_magnet.optimization.differential_evolution',wraps=scipy_de) as de:
            result=optimization.run_optimization(variables,builder,lambda d:metric(),lambda d:{'valid':True},
                model,folder,max_evaluations=24,population_size=7,popsize=0,maxiter=0)
        self.assertEqual(len(de.call_args.kwargs['init']),plan['population_size'])
        self.assertNotIn('popsize',de.call_args.kwargs)
        info=result['metadata']['rounds'][0]
        self.assertEqual(info['population_size'],7)
        self.assertIsNone(info['legacy_population_multiplier'])

    def test_legacy_cli_multiplier_remains_a_fallback_without_budget_shrinkage(self):
        variables=[{'name':'x%d'%i,'lower':0.,'upper':1.,'initial':.4,'integer':False} for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            result=optimization.run_optimization(variables,lambda x:Point(tuple(x)),lambda d:metric(),
                lambda d:{'valid':True},Point((.4,)*4),folder,max_evaluations=6,popsize=4,maxiter=0)
            initial=json.loads((Path(folder)/'initial_population.json').read_text(encoding='utf-8'))
        info=result['metadata']['rounds'][0]
        self.assertEqual(info['population_size'],16)
        self.assertEqual(info['legacy_population_multiplier'],4)
        self.assertEqual(len(initial['population']),16)
        self.assertEqual(result['metadata']['field_evaluations'],6)
        self.assertFalse(info['initialization_complete'])


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from concurrent.futures import Future
from mri_magnet.geometry import Design,RingSpec
from mri_magnet.serialization import save_json
from mri_magnet.workbench import Workbench,preflight
from mri_magnet.workbench_config import default_config
from mri_magnet.config_schema import canonicalize_config
from mri_magnet.workbench_server import create_server


def case():
    return canonicalize_config(json.loads((Path(__file__).resolve().parents[2]/'工程案例_100mmDSV.json').read_text(encoding='utf-8')))


def design72():
    rings=[]
    for k in range(8):
        for j in range(8):
            z=223.-29*k;r=116.+12*j
            rings.extend((RingSpec('p%d_%d'%(k,j),k,j,r,z,28,'out',-90+180/28),
                          RingSpec('m%d_%d'%(k,j),15-k,j,r,-z,28,'in',-90+180/28)))
    return Design(tuple(rings))


def add_seed(wb):
    wb.data['seeds']=[{'id':'d72','style':'actual72','design':design72().to_dict()}]
    wb.data['selected_seed_ids']=['d72']


class BudgetAndViewBackendTests(unittest.TestCase):
    def test_real_strict_solve_runs_full_jacobian_instead_of_budget_skip(self):
        from test_v3_workbench import configure,install_seed
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);configure(wb)
            wb.update_config({'solver':{'budget_policy':'strict','max_evaluations':200,'maxiter':1,
                'local_refine_evaluations':40,'active_set_evaluations':60}})
            install_seed(wb)
            plan=wb.solver_preflight();self.assertTrue(plan['valid'],plan['violations'])
            result=wb.run_step(5)
            self.assertEqual(result['state'],'complete')
            saved=json.loads((wb._run_root()/'详细数据'/'优化结果.json').read_text(encoding='utf-8'))
            lp=saved['metadata']['runs'][0]['metadata']['rounds'][0]['active_set_polish']
            self.assertNotEqual(lp['state'],'insufficient_budget_for_full_jacobian')
            self.assertGreaterEqual(lp['linearization_passes'],1)
            self.assertTrue(all(p['jacobian_columns']==p['dimensions'] for p in lp['passes']))
            self.assertLessEqual(saved['metadata']['field_evaluations'],200)
            island=wb.state()['island_statuses'][0]
            self.assertEqual(island['status'],'completed')
            self.assertEqual(island['refinement']['lp_passes'],lp['linearization_passes'])

    def test_strict_workbench_blocks_before_field_and_before_run_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(case());add_seed(wb)
            wb.update_config({'solver':{'local_refine_evaluations':64,'active_set_evaluations':70}})
            with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('physics must not run')) as field, patch('mri_magnet.islands.run_islands') as solver:
                with self.assertRaisesRegex(ValueError,'预算预检'):wb._step5()
                field.assert_not_called();solver.assert_not_called()
            self.assertFalse((Path(folder)/'runs').exists())
            self.assertFalse(wb.data['solver_preflight']['valid'])
            fixed=wb.apply_budget_suggestion()
            self.assertTrue(fixed['valid'],fixed['violations'])
            self.assertEqual(fixed['field_evaluations'],0)

    def test_even_columns_preflight_respects_seam_switch(self):
        cfg=case();cfg['de_novo']['column_counts']=[26,30,34]
        self.assertTrue(preflight(cfg)['valid'])
        cfg['manufacturing']['allow_split_pocket_seams']=False
        self.assertFalse(preflight(cfg)['valid'])
        cfg['manufacturing']['allow_split_pocket_seams']=True;cfg['de_novo']['column_counts']=[31]
        self.assertFalse(preflight(cfg)['valid'])

    def test_old_timeline_and_all_candidate_layouts_are_exposed_without_archive_rewrite(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);root=Path(folder);run=root/'runs'/'old';run.mkdir(parents=True)
            (run/'frame_0000.png').write_bytes(b'known recorded frame')
            page=run/'演化时间轴.html';page.write_text('const frames='+json.dumps([{'image':'frame_0000.png','evaluations':9}])+';show();',encoding='utf-8')
            (run/'layout.png').write_bytes(b'known layout')
            save_json(run/'归档清单.json',{'candidates':[{'candidate_id':'c1','archive_label':'均衡推荐','layout_file':'layout.png'}]})
            original=page.read_bytes();wb.data.update(run_dir='runs/old',evolution={'enabled':True,'frames_count':1},artifacts={'manifest':'runs/old/归档清单.json'})
            wb.data['steps'][4]['status']='passed'
            state=wb.state()
            self.assertEqual(state['evolution']['frames'][0]['image'],'runs/old/frame_0000.png')
            self.assertEqual(state['evolution']['frames'][0]['evaluations'],9)
            self.assertEqual(state['artifacts']['candidate_layouts'][0]['path'],'runs/old/layout.png')
            self.assertEqual(page.read_bytes(),original)
            self.assertNotIn('frames',wb.data['evolution'])


class PresetAndBudgetHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.wb=Workbench(self.tmp.name)
        self.server=create_server(self.tmp.name,workbench=self.wb,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:%d'%self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def post(self,path,body):
        with urlopen(Request(self.base+path,json.dumps(body).encode(),{'Content-Type':'application/json'})) as response:return json.load(response)

    def test_presets_load_real_case_without_starting_solver_or_deleting_results(self):
        marker=Path(self.tmp.name)/'earlier_result.class';marker.write_bytes(b'retained')
        with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('no fields')):
            result=self.post('/api/preset',{'id':'engineering_100mm'})
            self.assertEqual(result['state']['config']['target']['diameter_mm'],100.)
            self.assertEqual(result['state']['config']['de_novo']['pair_counts'],[8])
            self.assertFalse(result['state']['busy'])
            result=self.post('/api/preset',{'id':'default_20mm'})
            self.assertEqual(result['state']['config']['target']['diameter_mm'],20.)
        self.assertEqual(marker.read_bytes(),b'retained')
        with self.assertRaises(HTTPError) as error:self.post('/api/preset',{'id':'../../arbitrary'})
        self.assertEqual(error.exception.code,400)

    def test_running_job_rejects_presets(self):
        pending=Future();self.server.current_future=pending;before=self.wb.state()['config']
        try:
            with self.assertRaises(HTTPError) as error:self.post('/api/preset',{'id':'engineering_100mm'})
            self.assertEqual(error.exception.code,409);self.assertEqual(self.wb.config,before)
        finally:pending.set_result(None)

    def test_budget_endpoints_fix_exact_dimensions_without_field_calls(self):
        self.wb.update_config(case());add_seed(self.wb)
        self.wb.update_config({'solver':{'local_refine_evaluations':64,'active_set_evaluations':70}})
        with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('no fields')):
            report=self.post('/api/solver-preflight',{})['result']
            self.assertFalse(report['valid']);self.assertEqual(report['per_island'][0]['D'],72)
            self.assertTrue(report['per_island'][0]['lp_will_skip'])
            fixed=self.post('/api/apply-budget-suggestion',{})['result']
            self.assertTrue(fixed['valid'],fixed['violations'])

if __name__=='__main__':unittest.main()

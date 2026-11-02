"""Independent tests for readable packaging and preserved validation failures."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mri_magnet.geometry import Design,RingSpec
from mri_magnet.deliverables import export_candidates,selected_candidates
from mri_magnet.run_overview import write_run_overview,best_record
from mri_magnet.reporting import create_report

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGNcAAAAASUVORK5CYII=')


class ReadableOutputsTests(unittest.TestCase):
    def record(self,ident,radius=130.,mean=.04,ppm=10.,feasible=True):
        design=Design((RingSpec('source',0,0,radius,20.,8,'out',0.),),name=ident)
        metrics={'mean_norm_T':mean,'pp_norm_ppm':ppm,'magnet_count':8,'mass_magnet_kg':.1}
        return {'candidate_id':ident,'design':design.to_dict(),'metrics':metrics,'validation_metrics':dict(metrics),
            'geometry_feasible':True,'geometry_check':{'valid':True},'manufacturing_check':{'valid':True},
            'performance_check':{'valid':feasible},'feasible':feasible}

    def payload(self):
        seed=self.record('manufacturing_seed')
        baseline=self.record('baseline',mean=.045,ppm=15.,feasible=False)
        baseline['geometry_feasible']=False;baseline['manufacturing_check']={'valid':False}
        return {'baseline':baseline,'scan_reference':seed,'manufacturing':{'seed_record':seed},
            'candidates':[baseline,seed],'model_info':{'fov_geometry':'disk','fov_diameter_mm':20.,'fov_z_mm':0.},
            'requirements':{'min_mean_norm_T':.04,'max_pp_norm_ppm':10.,'field_scale_T':.04,'ppm_scale':10.}}

    @staticmethod
    def render(design,target,**kwargs):target.write_bytes(PNG);return target

    @staticmethod
    def export(design,target,root,settings):
        java=target/'MRI.class.java';java.write_text('class MRI {}',encoding='utf-8')
        return {'compiled':False,'error':None,'java_file':str(java),'class_file':None,'status':'java_generated',
                'magnet_count':design.magnet_count,'ring_count':len(design.rings),'solve_executed':False}

    def test_failed_dense_candidate_kept_diagnostic_without_becoming_pareto(self):
        payload=self.payload();bad=self.record('bad',radius=131.,mean=.041,ppm=10.1,feasible=False)
        bad['training_performance_check']={'valid':True}
        bad['validation_status']={'code':'training_pass_dense_fail','label':'训练达标／复核未通过（待审查）'}
        payload['candidates'].append(bad)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch('mri_magnet.deliverables.render_rz_layout',side_effect=self.render) as render,patch('mri_magnet.deliverables.export_comsol_model',side_effect=self.export):
                result=export_candidates(root,root/'详细数据',payload,{'comsol_root':'unused','deliverables':{'compile':False}},log=lambda _:None)
            self.assertEqual({r['id'] for r in result['candidates']},{'manufacturing_seed','bad'})
            diagnostic=next(r for r in result['candidates'] if r['id']=='bad')
            self.assertTrue(diagnostic['diagnostic_only'])
            self.assertIn('复核未通过_诊断',diagnostic['layout_file'])
            self.assertIsNotNone(render.call_args.kwargs['reference_design'])
            self.assertEqual([r['candidate_id'] for r in selected_candidates(payload)],['manufacturing_seed'])

    def test_rounded_metric_names_do_not_collide_and_ids_are_preserved(self):
        payload=self.payload();a=self.record('opaque_a',radius=132.,mean=.040001,ppm=9.0001)
        b=self.record('opaque_b',radius=133.,mean=.040002,ppm=9.0002)
        payload['candidates']=[a,b]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch('mri_magnet.deliverables.render_rz_layout',side_effect=self.render),patch('mri_magnet.deliverables.export_comsol_model',side_effect=self.export):
                result=export_candidates(root,root/'详细数据',payload,{'comsol_root':'unused','deliverables':{'compile':False}},log=lambda _:None)
            files=[r['layout_file'] for r in result['candidates']]
            self.assertEqual(len(set(files)),2)
            self.assertTrue(all('mT_' in path and 'ppm_8块' in path for path in files))
            self.assertEqual({r['id'] for r in result['candidates']},{'opaque_a','opaque_b'})

    def test_overview_uses_actual_slots_and_missing_stages_are_not_invented(self):
        payload=self.payload();better=self.record('improved',radius=131.25,mean=.041,ppm=5.)
        payload['candidates'].append(better)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=write_run_overview(root,payload)
            text=path.read_text(encoding='utf-8')
            self.assertIn('阶段1 混合初始化最优 | — | —',text)
            self.assertIn('L1: 131.25 (Δ+1.250)',text)
            changes=json.loads((root/'详细数据/物理尺寸改动.json').read_text(encoding='utf-8'))
            self.assertEqual(changes['rings'][0]['delta_r_mm'],1.25)
            self.assertEqual(best_record(payload)['candidate_id'],'improved')

    def test_stage_evaluation_counts_retain_trailing_zeros_and_topology_offset(self):
        payload=self.payload()
        payload['optimization_metadata']={'runs':[
            {'topology':1,'metadata':{'field_evaluations':360,'stage_summaries':[]}},
            {'topology':2,'metadata':{'field_evaluations':120,'stage_summaries':[{'stage':'de','field_evaluations':120,'metrics':payload['scan_reference']['metrics']}]}}]}
        with tempfile.TemporaryDirectory() as folder:
            text=write_run_overview(Path(folder),payload).read_text(encoding='utf-8')
            self.assertIn('| 480 | 训练点集 |',text)
            self.assertIn('| 0 | 训练点集 |',text)

    def test_report_leads_with_selected_layout_and_folds_internal_parameters(self):
        payload=self.payload();better=self.record('improved',radius=131.25,mean=.041,ppm=5.)
        better['parameters']={'radial_slack_pair_1_gap_0':.123}
        payload['candidates'].append(better)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout=root/'layout.png';layout.write_bytes(PNG)
            gif=root/'movie.gif';gif.write_bytes(base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'))
            timeline=root/'timeline.html';timeline.write_text('<html>timeline</html>',encoding='utf-8')
            payload['deliverables']={'candidates':[{'id':'improved','candidate_id':'improved','display_name':'最优','layout_file':'layout.png'}]}
            payload['evolution']={'enabled':True,'gif_file':'movie.gif','timeline_file':'timeline.html'}
            report=create_report(root,payload,details_dir=root/'详细数据')
            page=Path(report['report_html']).read_text(encoding='utf-8')
            self.assertLess(page.index('核心结果 · 磁铁布局与起点位移'),page.index('原始物理对照指标'))
            self.assertIn('image/gif;base64',page)
            self.assertIn('<summary>参数与溯源</summary>',page)
            self.assertIn('41<small> mT',page)
            self.assertIn('候选方案核心对比',page)


if __name__=='__main__':unittest.main()

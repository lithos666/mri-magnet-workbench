from pathlib import Path
import html,json,re,tempfile,unittest
from copy import deepcopy
from mri_magnet.verification_archive import append_fem_evidence
from mri_magnet.identity import canonical_physical_sha256
from mri_magnet.field_models import model_fingerprint
from mri_magnet.serialization import save_json
from mri_magnet.workbench_archive import archive_run
from test_v6_coupled_workflow import config,design


class VerificationArchiveTests(unittest.TestCase):
    def test_matching_fem_updates_all_workbench_roles_and_safe_folder_lookup(self):
        from mri_magnet.workbench import Workbench
        cfg=config();d=design()
        report={'design_sha256':canonical_physical_sha256(d),'model_fingerprint':model_fingerprint(cfg),
            'domain':cfg['target'],'fem_verified':True,'model_accuracy_passed':True}
        with tempfile.TemporaryDirectory() as temporary:
            wb=Workbench(temporary);wb.update_config(cfg)
            folder=Path(temporary)/'roles'/'balanced';folder.mkdir(parents=True)
            save_json(folder/'设计.json',d.to_dict())
            wb.data.update(best={'design':d.to_dict()},current_best={'design':d.to_dict()},
                pareto_recommendations=[{'folder':'roles/balanced'},{'design':design(41.).to_dict()}],
                candidates=[{'design':d.to_dict()}])
            wb._bind_fem_to_candidates(report)
            for row in (wb.data['best'],wb.data['current_best'],wb.data['pareto_recommendations'][0],wb.data['candidates'][0]):
                self.assertTrue(row['comsol_fem_verified'])
                self.assertTrue(row['physics_status']['model_accuracy_passed'])
            self.assertNotIn('physics_validation',wb.data['pareto_recommendations'][1])
            self.assertIn('已收敛',wb.data['steps'][5]['message'])
            self.assertIn('Python模型精度通过',wb.data['steps'][5]['message'])
            wrong=deepcopy(report);wrong['domain']=dict(cfg['target'],diameter_mm=20.)
            wb._bind_fem_to_candidates(wrong)
            self.assertFalse(wb.data['best']['comsol_fem_verified'])

    def test_later_fem_updates_matching_json_and_links_without_resetting_optimizer(self):
        cfg=config();d=design()
        report={'job_id':'test','status':'passed','fem_verified':True,'model_accuracy_passed':True,
            'design_sha256':canonical_physical_sha256(d),'model_fingerprint':model_fingerprint(cfg),
            'domain':cfg['target'],'physics':cfg['physics'],'design':d.to_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'COMSOL验证'/'one';folder.mkdir(parents=True)
            save_json(root/'配置快照.json',cfg);save_json(folder/'fem_report.json',report)
            row=root/'候选方案'/'one';row.mkdir(parents=True)
            save_json(row/'设计.json',d.to_dict());save_json(row/'复核指标.json',{'dense':{'pp_norm_ppm':12.}})
            candidate={'design':d.to_dict(),'candidate_id':'one','layout_file':'one.png'}
            save_json(root/'归档清单.json',{'candidates':[candidate],'best':candidate,
                'pareto_recommendations':[{'candidate_id':'one'}]})
            (root/'方案对比.csv').write_text('布局图,Python模型精度,FEM复核\none.png,待标定,待验证\nother.png,待标定,待验证\n',encoding='utf-8-sig')
            (root/'本次运行速览.md').write_text('Original optimization summary',encoding='utf-8')
            result=append_fem_evidence(root,report,folder/'fem_report.json')
            first=(root/'本次运行速览.md').read_text(encoding='utf-8')
            append_fem_evidence(root,report,folder/'fem_report.json')
            self.assertEqual(first,(root/'本次运行速览.md').read_text(encoding='utf-8'))
            self.assertIn('Original optimization summary',first)
            checked=json.loads((row/'复核指标.json').read_text(encoding='utf-8'))
            self.assertTrue(checked['fem_verified']);self.assertTrue(checked['physics_status']['model_accuracy_passed'])
            self.assertEqual(checked['dense']['pp_norm_ppm'],12.)
            self.assertTrue((root/result['fem_summary']).is_file())
            manifest=json.loads((root/'归档清单.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['fem_verified']);self.assertTrue(manifest['model_accuracy_passed'])
            self.assertTrue(manifest['best']['physics_status']['fem_verified'])
            self.assertTrue(manifest['pareto_recommendations'][0]['physics_status']['fem_verified'])
            comparison=(root/'方案对比.csv').read_text(encoding='utf-8-sig')
            self.assertIn('one.png,通过,已验证',comparison)
            self.assertIn('other.png,待标定,待验证',comparison)

    def test_mismatched_design_does_not_change_candidate_certificate(self):
        cfg=config();d=design()
        report={'job_id':'other','status':'passed','fem_verified':True,'model_accuracy_passed':True,
            'design_sha256':canonical_physical_sha256(design(41.)),'domain':cfg['target'],'physics':cfg['physics']}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'fem';folder.mkdir();save_json(folder/'fem_report.json',report)
            row=root/'候选方案'/'one';row.mkdir(parents=True);save_json(row/'设计.json',d.to_dict())
            save_json(row/'复核指标.json',{'fem_verified':False})
            append_fem_evidence(root,report,folder/'fem_report.json')
            self.assertFalse(json.loads((row/'复核指标.json').read_text())['fem_verified'])

    def _real_archive(self,root):
        cfg=config();cfg['scoring']['mode']='ppm'
        def row(name,radius,ppm):
            mean=.04
            return {'candidate_id':name,'design':design(radius).to_dict(),
                    'metrics':{'mean_norm_T':mean,'pp_norm_ppm':ppm,
                        'min_norm_T':mean*(1-ppm/2e6),'max_norm_T':mean*(1+ppm/2e6),
                        'pointset':'dense_archive_fixture','n_points':32}}
        save_json(root/'配置快照.json',cfg)
        result=archive_run(root,cfg,[],[row('matching',40.,10.),row('untouched',42.,50.)],
                           {'metadata':{'field_evaluations':72}},None,export_comsol=False)
        self.assertEqual(result['best']['candidate_id'],'matching')
        self.assertEqual(result['artifacts']['report'],'report_中文.html')
        self.assertTrue((root/result['artifacts']['report']).is_file())
        self.assertFalse((root/'报告.html').exists())
        return cfg,result

    def _bound_report(self,root,cfg,candidate,job_id='bound-real-archive'):
        report={'job_id':job_id,'status':'passed','fem_verified':True,'model_accuracy_passed':True,
                'mesh_converged':True,'boundary_converged':True,
                'design':candidate['design'],'design_sha256':canonical_physical_sha256(candidate['design']),
                'model_fingerprint':model_fingerprint(cfg),'physics':cfg['physics'],'domain':cfg['target'],
                'comparison':{'fem_metrics':{'mean_norm_T':.040001,'pp_norm_ppm':11.,'n_points':32},
                              'mean_relative_error':.000025,'ppm_absolute_error':1.,'ppm_error_limit':10.}}
        folder=root/'COMSOL验证'/job_id;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'fem_report.json';save_json(path,report)
        return report,path

    def _cards(self,text):
        return {html.unescape(re.search(r'<a href="([^"]+)"',card)[1]):card
                for card in re.findall(r'<article>.*?</article>',text,re.S)}

    def test_real_archive_html_and_best_conclusion_update_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,archive=self._real_archive(root)
            page=root/archive['artifacts']['report'];original=page.read_text(encoding='utf-8')
            other=next(r for r in archive['candidates'] if r['candidate_id']=='untouched')
            other_metrics=root/other['folder']/'复核指标.json'
            untouched=other_metrics.read_bytes()
            inputs=root/'详细数据'/'archive_inputs.json';input_bytes=inputs.read_bytes()
            report,path=self._bound_report(root,cfg,archive['best'])
            append_fem_evidence(root,report,path)
            current=page.read_text(encoding='utf-8');cards=self._cards(current)
            self.assertIn('；模型精度通过；FEM 已收敛',cards[archive['best']['layout_file']])
            self.assertEqual(cards[other['layout_file']],self._cards(original)[other['layout_file']])
            header=current.split('</header>',1)[0]
            self.assertIn('最佳方案Python模型精度通过；FEM 已收敛验证',header)
            self.assertIn('COMSOL验证/bound-real-archive/COMSOL验证结论.md',header)
            self.assertNotIn('FEM 待验证',header)
            self.assertFalse((root/'报告.html').exists())
            self.assertEqual(untouched,other_metrics.read_bytes())
            self.assertEqual(input_bytes,inputs.read_bytes())
            overview=(root/'本次运行速览.md').read_bytes()
            append_fem_evidence(root,report,path)
            self.assertEqual(current,page.read_text(encoding='utf-8'))
            self.assertEqual(overview,(root/'本次运行速览.md').read_bytes())
            self.assertEqual(current.count('<!-- v6-fem:bound-real-archive -->'),1)

    def test_manifest_report_takes_precedence_with_page_relative_evidence_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,archive=self._real_archive(root)
            canonical=root/archive['artifacts']['report'];original=canonical.read_bytes()
            preferred=root/'pages'/'custom.html';preferred.parent.mkdir()
            preferred.write_bytes(original)
            archive['artifacts']['report']='pages/custom.html';save_json(root/'归档清单.json',archive)
            report,path=self._bound_report(root,cfg,archive['best'])
            append_fem_evidence(root,report,path)
            self.assertEqual(canonical.read_bytes(),original)
            text=preferred.read_text(encoding='utf-8')
            self.assertIn('最佳方案Python模型精度通过；FEM 已收敛验证',text)
            self.assertIn('href="../COMSOL验证/bound-real-archive/COMSOL验证结论.md"',text)

    def test_unsafe_manifest_link_falls_back_without_writing_outside_run(self):
        for value in ('../outside.html','ABSOLUTE','配置快照.json',None):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);root=base/'run';cfg,archive=self._real_archive(root)
                canonical=root/archive['artifacts']['report']
                outside=base/'outside.html';outside.write_bytes(canonical.read_bytes())
                original=outside.read_bytes()
                archive['artifacts']['report']=str(outside) if value=='ABSOLUTE' else value
                save_json(root/'归档清单.json',archive)
                report,path=self._bound_report(root,cfg,archive['best'])
                append_fem_evidence(root,report,path)
                self.assertEqual(outside.read_bytes(),original)
                self.assertIn('最佳方案Python模型精度通过',canonical.read_text(encoding='utf-8'))
                self.assertEqual(json.loads((root/'配置快照.json').read_text(encoding='utf-8')),cfg)

    def test_legacy_report_name_is_compatible_when_canonical_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,archive=self._real_archive(root)
            legacy=root/'报告.html';(root/archive['artifacts']['report']).rename(legacy)
            archive['artifacts'].pop('report');save_json(root/'归档清单.json',archive)
            report,path=self._bound_report(root,cfg,archive['best'])
            append_fem_evidence(root,report,path)
            self.assertIn('最佳方案Python模型精度通过',legacy.read_text(encoding='utf-8'))

    def test_failed_or_old_certificate_replaces_previously_passed_html_status(self):
        for case in ('failed','old-model','missing-binding'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);cfg,archive=self._real_archive(root)
                page=root/archive['artifacts']['report']
                report,path=self._bound_report(root,cfg,archive['best'])
                append_fem_evidence(root,report,path)
                old_cards=self._cards(page.read_text(encoding='utf-8'))
                changed=deepcopy(report)
                if case=='failed':
                    changed.update(status='solver_failed',fem_verified=False,model_accuracy_passed=False,comparison={})
                elif case=='old-model':changed['model_fingerprint']='old-model-fingerprint'
                else:changed.pop('model_fingerprint')
                save_json(path,changed);append_fem_evidence(root,changed,path)
                text=page.read_text(encoding='utf-8');cards=self._cards(text)
                self.assertIn('；模型精度未通过；FEM 未通过或待复核',cards[archive['best']['layout_file']])
                self.assertIn('最佳方案Python模型精度未通过；FEM 未通过或待验证',text.split('</header>',1)[0])
                if case!='failed':self.assertIn('与当前工程的模型或评价域绑定不匹配',text.split('</header>',1)[0])
                other=next(r for r in archive['candidates'] if r['candidate_id']=='untouched')
                self.assertEqual(cards[other['layout_file']],old_cards[other['layout_file']])
                self.assertEqual(text.count('<!-- v6-fem:bound-real-archive -->'),1)
                self.assertIn('| COMSOL FEM | — | — | 未计算或未通过验证 |',
                              (root/'本次运行速览.md').read_text(encoding='utf-8'))
                append_fem_evidence(root,changed,path)
                self.assertEqual(text,page.read_text(encoding='utf-8'))

    def test_non_best_fem_does_not_transfer_certificate_to_best_card(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,archive=self._real_archive(root)
            page=root/archive['artifacts']['report'];original=self._cards(page.read_text(encoding='utf-8'))
            other=next(r for r in archive['candidates'] if r['candidate_id']=='untouched')
            report,path=self._bound_report(root,cfg,other)
            append_fem_evidence(root,report,path)
            text=page.read_text(encoding='utf-8');cards=self._cards(text)
            self.assertEqual(cards[archive['best']['layout_file']],original[archive['best']['layout_file']])
            self.assertIn('；模型精度通过；FEM 已收敛',cards[other['layout_file']])
            manifest=json.loads((root/'归档清单.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['fem_verified']);self.assertFalse(manifest['model_accuracy_passed'])
            self.assertNotIn('最佳方案Python模型精度通过',text.split('</header>',1)[0])

if __name__=='__main__':unittest.main()

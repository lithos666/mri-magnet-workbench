from pathlib import Path
import json,tempfile,unittest
from copy import deepcopy
from mri_magnet.verification_archive import append_fem_evidence
from mri_magnet.identity import canonical_physical_sha256
from mri_magnet.field_models import model_fingerprint
from mri_magnet.serialization import save_json
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
            'domain':cfg['target'],'physics':cfg['physics']}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'COMSOL验证'/'one';folder.mkdir(parents=True)
            save_json(root/'配置快照.json',cfg);save_json(folder/'fem_report.json',report)
            row=root/'候选方案'/'one';row.mkdir(parents=True)
            save_json(row/'设计.json',d.to_dict());save_json(row/'复核指标.json',{'dense':{'pp_norm_ppm':12.}})
            save_json(root/'归档清单.json',{'candidates':[{'design':d.to_dict()}]})
            (root/'本次运行速览.md').write_text('Original optimization summary',encoding='utf-8')
            (root/'报告.html').write_text('<header>Original</header>',encoding='utf-8')
            result=append_fem_evidence(root,report,folder/'fem_report.json')
            first=(root/'本次运行速览.md').read_text(encoding='utf-8')
            append_fem_evidence(root,report,folder/'fem_report.json')
            self.assertEqual(first,(root/'本次运行速览.md').read_text(encoding='utf-8'))
            self.assertIn('Original optimization summary',first)
            checked=json.loads((row/'复核指标.json').read_text(encoding='utf-8'))
            self.assertTrue(checked['fem_verified']);self.assertTrue(checked['physics_status']['model_accuracy_passed'])
            self.assertEqual(checked['dense']['pp_norm_ppm'],12.)
            self.assertTrue((root/result['fem_summary']).is_file())

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

if __name__=='__main__':unittest.main()

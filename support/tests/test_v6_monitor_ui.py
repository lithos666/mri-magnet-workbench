"""V6 shared physics, transient response reuse and separate validation flags."""
from pathlib import Path
from unittest.mock import patch
import json,re,shutil,subprocess,tempfile,unittest
import numpy as np

from mri_magnet.geometry import Design,RingSpec
from mri_magnet.evolution import EvolutionMonitor
from mri_magnet.field_models import make_field_engine,model_fingerprint
from mri_magnet.identity import canonical_physical_sha256
from mri_magnet.workbench_archive import physics_status,metric_comparison_rows
from mri_magnet.workbench_config import default_config


def fixture():
    return Design((RingSpec('a',0,0,30.,12.,8,'out',0.),
                   RingSpec('b',1,0,30.,-12.,8,'in',0.)),cube_mm=6.)


class CoupledMonitorTests(unittest.TestCase):
    def test_prepared_response_is_reused_without_matrix_build_or_persistent_payload(self):
        config=default_config();config['physics'].update(quadrature_order=1)
        config['target'].update(geometry='sphere',diameter_mm=10.)
        config['evolution'].update(enabled=True,grid_points=9)
        design=fixture();engine=make_field_engine(config)
        expected=engine.field(design,np.array([[0.,0.,0.],[1.,2.,3.]]))
        payload=engine.export_prepared(design)
        metrics={'mean_norm_T':.01,'pp_norm_ppm':10.,'model_fingerprint':model_fingerprint(config)}
        with tempfile.TemporaryDirectory() as folder:
            monitor=EvolutionMonitor(folder,design,metrics,{},config)
            captured=[]
            def preview(event):
                captured.append(monitor.engine.field(design,np.array([[0.,0.,0.],[1.,2.,3.]])))
            with patch.object(monitor,'_render',side_effect=preview), \
                 patch('mri_magnet.material_response._interaction_matrix',side_effect=AssertionError('must reuse prepared response')):
                monitor({'event':'new_best','design':design.to_dict(),'metrics':metrics,'prepared_response':payload})
            self.assertIsNone(monitor.error)
            self.assertEqual(len(captured),1)
            np.testing.assert_allclose(captured[0],expected,rtol=1e-12,atol=1e-14)
            self.assertEqual(monitor.engine.statistics['matrix_builds'],0)
            self.assertEqual(monitor.engine.statistics['imported_preparations'],1)
            history=monitor.history_path.read_text(encoding='utf-8')
            self.assertNotIn('prepared_response',history)
            result=monitor.finish()
            self.assertEqual(result['model_fingerprint'],model_fingerprint(config))
            self.assertTrue(result['preview_cost_excluded_from_search_budget'])

    def test_metrics_from_different_physics_stop_preview_before_field_call(self):
        config=default_config();design=fixture()
        with tempfile.TemporaryDirectory() as folder:
            monitor=EvolutionMonitor(folder,design,{'mean_norm_T':.01,'pp_norm_ppm':1.},{},config)
            with patch.object(monitor,'_render') as draw:
                monitor({'event':'new_best','design':design.to_dict(),
                    'metrics':{'mean_norm_T':.01,'pp_norm_ppm':1.,'model_fingerprint':'wrong-model'}})
            draw.assert_not_called()
            self.assertIsNotNone(monitor.error)


class IndependentVerificationFlagsTests(unittest.TestCase):
    def test_export_is_not_fem_or_model_accuracy_proof(self):
        config=default_config();design=fixture().to_dict()
        row={'design':design,'model_export':{'compiled':True},'metrics':{'mean_norm_T':.05,'pp_norm_ppm':80.}}
        status=physics_status(row,config)
        self.assertFalse(status['fem_verified']);self.assertFalse(status['model_accuracy_passed'])
        row['physics_validation']={'fem_verified':True,'model_accuracy_passed':False,
            'fem_metrics':{'mean_norm_T':.049,'pp_norm_ppm':200.},
            'design_sha256':canonical_physical_sha256(design),'model_fingerprint':model_fingerprint(config),
            'domain':config['target']}
        status=physics_status(row,config)
        self.assertTrue(status['fem_verified']);self.assertFalse(status['model_accuracy_passed'])
        self.assertTrue(status['verification_binding_valid'])
        self.assertEqual(metric_comparison_rows(row,config)[-1][1]['pp_norm_ppm'],200.)

    def test_stale_or_missing_certificate_bindings_never_claim_fem_or_model_accuracy(self):
        config=default_config();design=fixture().to_dict()
        certificate={'fem_verified':True,'model_accuracy_passed':True,
            'design_sha256':canonical_physical_sha256(design),'model_fingerprint':model_fingerprint(config),
            'domain':dict(config['target']),'fem_metrics':{'mean_norm_T':.04,'pp_norm_ppm':100.}}
        source={'design':design,'physics_validation':certificate}
        self.assertTrue(physics_status(source,config)['model_accuracy_passed'])
        for key in ('design_sha256','model_fingerprint','domain'):
            with self.subTest(missing=key):
                changed=dict(certificate);changed.pop(key)
                status=physics_status(dict(source,physics_validation=changed),config)
                self.assertFalse(status['fem_verified']);self.assertFalse(status['model_accuracy_passed'])
                self.assertEqual(status['verification'],changed)
        for key,value in (('design_sha256','different-design'),('model_fingerprint','different-model'),
                          ('domain',dict(config['target'],diameter_mm=100.))):
            with self.subTest(mismatch=key):
                changed=dict(certificate,**{key:value})
                status=physics_status(dict(source,physics_validation=changed),config)
                self.assertFalse(status['fem_verified']);self.assertFalse(status['model_accuracy_passed'])
                self.assertIsNone(metric_comparison_rows(dict(source,physics_validation=changed),config)[-1][1])


@unittest.skipUnless(shutil.which('node'),'Node needed for frontend execution')
class CoupledUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[2]
        cls.html=(cls.root/'mri_magnet/workbench.html').read_text(encoding='utf-8')
        cls.helper=cls.root/'mri_magnet/workbench_analysis.js'

    def js(self,body):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'check.js';p.write_text('const a=require('+json.dumps(str(self.helper))+');\n'+body,encoding='utf-8')
            result=subprocess.run(['node',str(p)],check=True,capture_output=True,text=True,encoding='utf-8')
        return json.loads(result.stdout)

    def test_follow_run_analysis_does_not_silently_force_fixed_source(self):
        data=self.js("const design={rings:[{radius_mm:30,z_mm:12,count:8,phase_deg:0,polarity:'out'}]}; console.log(JSON.stringify(a.requestPayload({design,kind:'candidate'},{tolerance:false}))); ")
        self.assertEqual(data['analysis_options'],{'tolerance':False})

    def test_precision_changes_clear_previous_analysis_but_cache_capacity_does_not(self):
        data=self.js("""
const design={rings:[{radius_mm:30,z_mm:12,count:8,phase_deg:0,polarity:'out'}]};
const config={target:{geometry:'sphere',diameter_mm:10},physics:{field_model:'coupled',quadrature_order:2,cells_per_axis:1},polarization_model:'remanence'};
const report={design,domain:config.target,physics:config.physics,polarization_model:'remanence'};
console.log(JSON.stringify({same:!!a.matchingAnalysis(report,design,config),
 changed:a.matchingAnalysis(report,design,{...config,physics:{...config.physics,cells_per_axis:2}}),
 capacity:!!a.matchingAnalysis(report,design,{...config,physics:{...config.physics,prepared_cache_size:8}})}));
""")
        self.assertTrue(data['same']);self.assertIsNone(data['changed']);self.assertTrue(data['capacity'])

    def test_numeric_precision_selectors_produce_numbers(self):
        parse=re.search(r'^function parseFieldValue\(.*$',self.html,re.M).group(0)
        data=self.js(parse+"\nconsole.log(JSON.stringify(parseFieldValue({value:'2',type:'select-one',dataset:{number:'true'}}))); ")
        self.assertIsInstance(data,int);self.assertEqual(data,2)

    def test_runtime_summary_separates_ledger_cost_and_accuracy(self):
        data=self.js("""
const state={config:{physics:{field_model:'coupled',quadrature_order:2,cells_per_axis:1}},
solver_preflight:{physics:{max_unknowns:348,estimated_seconds:7200,estimated_parallel_working_bytes:128e6}},
job:{physics_stage:{stage:'coupling_matrix',progress_percent:50},engine_statistics:{self_consistent_solves:9,preparation_seconds:3,field_seconds:2,prepared_cache_hits:4}},
physics_validation:{fem_verified:true,model_accuracy_passed:false}};
console.log(JSON.stringify(a.physicsSummary(state)));
""")
        self.assertIn('348',data['cost']);self.assertIn('2.00',data['cost'])
        self.assertIn('50%',data['stage']);self.assertIn('9',data['ledger'])
        self.assertIn('FEM',data['accuracy']);self.assertIn('待标定',data['accuracy'])
        labels=self.js("console.log(JSON.stringify([a.candidateVerificationText({physics_status:{verification_binding_valid:true,fem_verified:true,model_accuracy_passed:false}}),a.candidateVerificationText({physics_status:{fem_verified:true,model_accuracy_passed:true}})]));")
        self.assertIn('已收敛',labels[0]);self.assertIn('精度未通过',labels[0])
        self.assertIn('待验证',labels[1])

    def test_physics_task_requests_require_explicit_click_and_real_design(self):
        for control in ('calibrate-physics','verify-fem'):
            self.assertIn('id="'+control+'"',self.html)
        self.assertIn("api('/api/'+action,{design:source.design})",self.html)
        poll=re.search(r'async function poll\(\).*',self.html).group(0)
        self.assertNotIn('verify-fem',poll);self.assertNotIn('calibrate',poll)
        self.assertIn("['illustrative','reference','unavailable'].includes(source.kind)",self.html)

    def test_fem_status_cannot_follow_another_geometry_or_precision(self):
        data=self.js("""
const design={rings:[{radius_mm:30,z_mm:12,count:8,phase_deg:0,polarity:'out'}]};
const physics={field_model:'coupled',quadrature_order:2,cells_per_axis:1};
const state={config:{physics},physics_validation:{design,physics,fem_verified:true,model_accuracy_passed:true}};
const same=a.physicsSummary(state,design).accuracy;
const moved=structuredClone(design);moved.rings[0].radius_mm+=.1;
const changed=a.physicsSummary(state,moved).accuracy;
state.config={physics:{...physics,quadrature_order:3}};
console.log(JSON.stringify({same,changed,precision:a.physicsSummary(state,design).accuracy}));
""")
        self.assertIn('FEM',data['same']);self.assertIn('已验证',data['same'])
        self.assertNotIn('已验证',data['changed']);self.assertIn('其他方案',data['changed'])
        self.assertNotIn('已验证',data['precision'])

    def test_fem_status_cannot_follow_another_evaluation_domain(self):
        data=self.js("""
const design={rings:[{radius_mm:30,z_mm:12,count:8,phase_deg:0,polarity:'out'}]};
const physics={field_model:'coupled',quadrature_order:2,cells_per_axis:1};
const target={geometry:'sphere',diameter_mm:10,z_mm:0};
const state={config:{physics,target},physics_validation:{design,physics,domain:target,fem_verified:true,model_accuracy_passed:true}};
const same=a.physicsSummary(state,design).accuracy;
state.config={physics,target:{...target,geometry:'disk',z_mm:3}};
console.log(JSON.stringify({same,changed:a.physicsSummary(state,design).accuracy}));
""")
        self.assertIn('已验证',data['same']);self.assertNotIn('已验证',data['changed'])

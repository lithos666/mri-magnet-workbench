"""Real loopback orchestration, with no COMSOL or physical solver launched."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import http.client,json,tempfile,threading,time,unittest
import numpy as np

from mri_magnet.workbench_server import create_server
from mri_magnet.workbench import Workbench,CancelledError
from mri_magnet.geometry import Design,RingSpec


class VerificationBackend:
    def __init__(self):
        self.entered=threading.Event();self.release=threading.Event()
        self.cancelled=False;self.fail=False;self.calls=[]

    def state(self):
        return {'config':{},'steps':[{'number':i,'status':'pending'} for i in range(1,7)]}

    def verify_fem(self,design=None,settings=None):
        self.calls.append(('verify-fem',design,settings));return self._wait()

    def calibrate(self,design=None,fem_report=None,apply=False):
        self.calls.append(('calibrate',design,fem_report,apply));return self._wait()

    def analyze(self,**kwargs):
        self.calls.append(('analysis',kwargs));return self._wait()

    def _wait(self):
        self.entered.set()
        if not self.release.wait(5):raise RuntimeError('test backend timeout')
        if self.cancelled:raise CancelledError('cooperative verification cancellation')
        if self.fail:raise ValueError('explicit test FEM failure')
        return {'design_sha256':'test-geometry','fem_verified':False,'model_accuracy_passed':False}

    def cancel(self):
        self.cancelled=True;self.release.set();return {'requested':True}


class VerificationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.backend=VerificationBackend()
        self.server=create_server(self.temp.name,port=0,workbench=self.backend)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.port=self.server.server_address[1]

    def tearDown(self):
        self.backend.release.set()
        if self.server.analysis_future:
            try:self.server.analysis_future.result(timeout=5)
            except BaseException:pass
        self.server.shutdown();self.server.server_close();self.thread.join(5);self.temp.cleanup()

    def request(self,path='/api/state',body=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=2)
        conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),
                     headers={'Content-Type':'application/json'})
        response=conn.getresponse();status=response.status;data=json.loads(response.read());conn.close()
        return status,data

    def test_fem_is_nonblocking_and_excludes_mutation_solve_analysis_and_export(self):
        status,data=self.request('/api/verify-fem',{'design':{'rings':[]}})
        self.assertEqual(status,202);self.assertEqual(data['analysis_job']['kind'],'verify-fem')
        self.assertTrue(self.backend.entered.wait(2))
        status,state=self.request();self.assertEqual(status,200)
        self.assertEqual(state['analysis_job']['status'],'running')
        for route,body in (('/api/config',{'config':{}}),('/api/step/5',{}),('/api/analysis',{}),
                           ('/api/calibrate',{}),('/api/export-current',{})):
            self.assertEqual(self.request(route,body)[0],409,route)
        self.assertEqual(len(self.backend.calls),1)
        self.backend.release.set();self.server.analysis_future.result(timeout=5)
        self.assertEqual(self.request()[1]['analysis_job']['status'],'completed')

    def test_cancelled_baseexception_updates_terminal_status_and_releases_buttons(self):
        self.assertEqual(self.request('/api/verify-fem',{})[0],202)
        self.assertTrue(self.backend.entered.wait(2))
        self.assertEqual(self.request('/api/cancel',{})[0],200)
        self.server.analysis_future.result(timeout=5)
        state=self.request()[1]
        self.assertEqual(state['analysis_job']['status'],'cancelled')
        self.assertIn('finished_at',state['analysis_job'])
        self.assertFalse(state['busy'])
        # Completion of a cancelled task must not leave the backend exclusive.
        self.backend.cancelled=False;self.backend.entered.clear()
        self.assertEqual(self.request('/api/calibrate',{'apply':False})[0],202)
        self.server.analysis_future.result(timeout=5)
        self.assertEqual(self.request()[1]['analysis_job']['status'],'completed')

    def test_calibration_options_are_forwarded_and_invalid_apply_is_rejected(self):
        self.assertEqual(self.request('/api/calibrate',{'apply':'yes'})[0],400)
        self.assertEqual(self.request('/api/verify-fem',{'unknown':True})[0],400)
        self.backend.release.set()
        design={'rings':[]}
        status,_=self.request('/api/calibrate',{'design':design,'apply':False,'fem_report':{'fem_verified':False}})
        self.assertEqual(status,202);self.server.analysis_future.result(timeout=5)
        self.assertEqual(self.backend.calls[0],('calibrate',design,{'fem_verified':False},False))

    def test_failure_is_visible_and_never_sets_verification_passed(self):
        self.backend.fail=True;self.backend.release.set()
        self.assertEqual(self.request('/api/verify-fem',{})[0],202)
        self.server.analysis_future.result(timeout=5)
        state=self.request()[1]
        self.assertEqual(state['analysis_job']['status'],'failed')
        self.assertIn('FEM failure',state['analysis_job']['error'])
        self.assertNotIn('physics_validation',state)


class ConstantEngine:
    def field(self,design,points):
        out=np.zeros((len(points),3));out[:,2]=.04;return out


class StaleVerificationStateTests(unittest.TestCase):
    def test_configuration_change_retains_report_but_cannot_publish_current_fem_pass(self):
        design=Design((RingSpec('upper',0,0,135.,35.,24,'out',0.),
                       RingSpec('lower',1,0,135.,-35.,24,'in',0.)))
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder)
            before=deepcopy(wb.config['target'])
            def changing_job(*args,**kwargs):
                wb.update_config({'target':{'diameter_mm':30.}})
                return {'fem_verified':True,'model_accuracy_passed':True,'status':'passed'}
            with patch('mri_magnet.field_models.make_field_engine',return_value=ConstantEngine()), \
                 patch('mri_magnet.comsol_validation.prepare_fem_validation',return_value={}), \
                 patch('mri_magnet.comsol_validation.run_fem_validation',side_effect=changing_job):
                with self.assertRaisesRegex(ValueError,'配置已改变'):
                    wb.verify_fem(design.to_dict())
            self.assertNotIn('physics_validation',wb.data)
            reports=list(Path(folder).rglob('fem_report.json'))
            self.assertEqual(len(reports),1)
            report=json.loads(reports[0].read_text(encoding='utf-8'))
            self.assertEqual(report['domain'],before)

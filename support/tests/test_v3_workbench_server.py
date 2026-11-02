"""Safe stop and independent snapshot-export HTTP contracts."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from mri_magnet.workbench_server import create_server


class SnapshotBackend:
    def __init__(self):
        self.entered=threading.Event();self.release=threading.Event()
        self.export_entered=threading.Event();self.export_release=threading.Event()
        self.status='pending';self.export_options=None
    def state(self):
        return {'config':{},'steps':[{'number':5,'status':self.status}],
                'job':{'status':self.status,'best_mean_mT':35.25,'best_ppm':.8,
                       'max_evaluations':480,'evaluations':43,'progress_percent':43/480*100,
                       'topology_name':'宽展型','log_tail':['43 次求场，发现新的最优解']}}
    def run_step(self,number,**body):
        self.status='running';self.entered.set();self.release.wait(5)
        self.status='stopped_with_result'
        return {'state':'stopped_with_result','message':'最优解已保存'}
    def export_current(self,**options):
        self.export_options=options;self.export_entered.set();self.export_release.wait(5)
        return {'artifacts':{'layout':'snapshots/独立布局.png'},'source':'incumbent_copy'}
    def cancel(self):
        self.release.set();return {'requested':True}


class SnapshotServerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.wb=SnapshotBackend();self.server=create_server(self.root,port=0,workbench=self.wb)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.port=self.server.server_address[1]
    def tearDown(self):
        self.wb.release.set();self.wb.export_release.set()
        for future in (self.server.current_future,self.server.export_future):
            if future:future.result(timeout=5)
        self.server.shutdown();self.server.server_close();self.thread.join(5);self.temp.cleanup()
    def request(self,path='/api/state',body=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body))
        response=connection.getresponse();status=response.status;value=json.loads(response.read());connection.close()
        return status,value
    def test_live_export_runs_independently_and_does_not_block_stop_or_mutate_solver_job(self):
        status,_=self.request('/api/step/5',{});self.assertEqual(status,202);self.assertTrue(self.wb.entered.wait(2))
        status,_=self.request('/api/export-current',{});self.assertEqual(status,202);self.assertTrue(self.wb.export_entered.wait(2))
        status,state=self.request();self.assertTrue(state['busy']);self.assertEqual(state['job']['step'],5)
        self.assertEqual(state['export_job']['status'],'running');self.assertEqual(self.wb.export_options,{'compile':False})
        self.assertEqual(state['job']['best_mean_mT'],35.25);self.assertEqual(state['job']['log_tail'],['43 次求场，发现新的最优解'])
        self.assertEqual(self.request('/api/config',{'config':{}})[0],409)
        self.assertEqual(self.request('/api/export-current',{})[0],409)
        self.assertEqual(self.request('/api/cancel',{})[0],200)
        self.server.current_future.result(3)
        status,state=self.request();self.assertFalse(state['busy'])
        self.assertEqual(state['job']['status'],'stopped_with_result')
        self.assertEqual(state['steps'][0]['status'],'stopped_with_result')
        self.assertEqual(state['export_job']['status'],'running')
        self.wb.export_release.set();self.server.export_future.result(3)
        self.assertEqual(self.request()[1]['export_job']['result']['source'],'incumbent_copy')
    def test_persisted_stop_status_is_not_overwritten_by_idle_http_server(self):
        self.wb.status='stopped_with_result'
        status,state=self.request();self.assertEqual(status,200)
        self.assertEqual(state['job']['status'],'stopped_with_result')
        self.assertEqual(state['steps'][0]['status'],'stopped_with_result')
    def test_export_capability_and_options_fail_explicitly(self):
        self.assertEqual(self.request('/api/export-current',{'compile':'yes'})[0],400)
        self.wb.export_current=None
        self.assertEqual(self.request('/api/export-current',{})[0],501)

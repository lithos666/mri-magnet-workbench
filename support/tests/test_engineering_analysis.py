"""Independent regression of domain semantics and analysis/acceptance separation."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

from mri_magnet.geometry import Design,RingSpec
from mri_magnet.workbench_config import default_config,validate_spec
from mri_magnet.config_schema import canonicalize_config
from mri_magnet.field_analysis import field_slices,analysis_sample
from mri_magnet.workbench import Workbench
from mri_magnet.workbench_server import create_server


def fixture():
    return Design((RingSpec('upper',0,0,30.,12.,8,'up',0.),
                   RingSpec('lower',1,0,30.,-12.,8,'up',0.)),cube_mm=6.,reference_mur=1.05)


def configuration():
    return canonicalize_config({'magnet':{'cube_mm':6.},'manufacturing':{'enabled':False},
        'envelope':{'protected_bore_diameter_mm':20.,'max_outer_diameter_mm':100.,'max_length_mm':100.},
        'target':{'geometry':'sphere','diameter_mm':10.,'min_field_mT':.001},
        'analysis':{'harmonic_max_degree':2,'slice_grid_size':7,'integration_volume':8,'integration_surface':16}})


class LinearField:
    polarization_model='test_linear'
    def field(self,design,points):
        points=np.asarray(points,float)
        result=np.zeros_like(points);result[:,2]=.04+points@np.array([1e-6,2e-6,3e-6])
        return result


class SliceContractTests(unittest.TestCase):
    def test_three_planes_are_engineering_coordinates_with_shared_signed_scale(self):
        config=configuration();config['analysis']['slice_color_limit_ppm']=500.
        slices=field_slices(fixture(),LinearField(),config,.04)
        self.assertEqual(slices['color_limit_ppm'],500.)
        for plane in slices['planes']:
            points=np.asarray(plane['points_mm']);valid=np.asarray(plane['valid_mask'])
            fixed=({'XY':2,'XZ':1,'YZ':0})[plane['name']]
            self.assertTrue(np.all(points[:,fixed]==0))
            expected=(points[valid]@np.array([1e-6,2e-6,3e-6]))/.04*1e6
            self.assertTrue(np.allclose(np.array(plane['values_ppm'],object)[valid].astype(float),expected))
            self.assertTrue(all(value is None for value,flag in zip(plane['values_ppm'],valid) if not flag))

    def test_planar_fov_does_not_invent_xz_yz_or_change_z(self):
        config=configuration();config['target'].update(geometry='disk',z_mm=3.)
        planes=field_slices(fixture(),LinearField(),config,.04)['planes']
        self.assertEqual([row['name'] for row in planes],['XY'])
        self.assertTrue(np.all(np.asarray(planes[0]['points_mm'])[:,2]==3.))

    def test_global_analysis_samples_never_use_fundamental_domain(self):
        sample=analysis_sample(configuration())
        points=np.asarray(sample['points_mm'])
        self.assertTrue(np.any(points[:,2]<0))
        self.assertTrue(np.any(points[:,0]<0))
        self.assertAlmostEqual(float(np.sum(sample['weights'])),1.)

    def test_invalid_zero_reference_cannot_create_misleading_ppm(self):
        with self.assertRaises(ValueError):field_slices(fixture(),LinearField(),configuration(),0.)

    def test_invalid_analysis_and_harmonic_controls_are_rejected(self):
        for change in ({'analysis':{'slice_grid_size':10000}},
                       {'analysis':{'tolerance':{'remanence_half_range_fraction':1.}}},
                       {'de_novo':{'harmonic_order_weights':{'8':-1.}}}):
            with self.assertRaises(ValueError):canonicalize_config(change)


class AnalysisPersistenceTests(unittest.TestCase):
    def test_analysis_limits_reject_before_any_field_call(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(configuration())
            with patch('mri_magnet.field_analysis.FieldEngine.field') as solve:
                wb.update_config({'analysis':{'max_source_point_pairs':1}})
                with self.assertRaises(ValueError):wb.analyze(design=fixture().to_dict())
                solve.assert_not_called()
            wb.update_config({'analysis':{'max_source_point_pairs':75000000,'mutual_demag':{'max_unknowns':1}}})
            with patch('mri_magnet.field_analysis.FieldEngine.field') as solve:
                with self.assertRaises(ValueError):wb.analyze(design=fixture().to_dict(),analysis_options={'mutual_demag':True})
                solve.assert_not_called()

    def test_optional_coupled_response_receives_physics_settings_without_ui_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(configuration())
            report=wb.analyze(design=fixture().to_dict(),analysis_options={'mutual_demag':True,'tolerance':False})
            self.assertEqual(report['field_model_used'],'coupled_uniform_cells')
            self.assertEqual(report['polarization_model'],wb.config['polarization_model'])
            self.assertEqual(report['material_response']['status'],'completed')
            self.assertNotIn('best',wb.data)

    def test_real_diagnostics_persist_without_replacing_dense_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(configuration())
            wb.data['best']={'design':fixture().to_dict(),'metrics':{'mean_norm_T':.123,'pp_norm_ppm':7.}}
            before=deepcopy(wb.data['best'])
            report=wb.analyze(analysis_options={'tolerance':False,'mutual_demag':False})
            self.assertEqual(wb.data['best'],before)
            self.assertFalse(report['fem_verified'])
            self.assertFalse(report['dense_acceptance_replaced'])
            self.assertTrue(report['harmonics']['spherical']['available'])
            self.assertLess(report['slices']['reference_Bz_T'],0.)
            self.assertEqual(report['harmonics']['spherical']['reference_Bz_T'],report['slices']['reference_Bz_T'])
            self.assertEqual(len(report['slices']['planes']),3)
            self.assertTrue((Path(folder)/report['artifacts']['overview']).is_file())
            self.assertEqual(Workbench(folder).state()['analysis']['design_sha256'],report['design_sha256'])
            wb.update_config({'analysis':{'slice_grid_size':9}})
            self.assertNotIn('analysis',wb.data)
            self.assertEqual(wb.data['best'],before)

    def test_empty_or_mismatching_material_is_rejected_before_field_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config(configuration())
            with patch('mri_magnet.field_analysis.analyze_design') as compute:
                with self.assertRaises(ValueError):wb.analyze(design={'rings':[]})
                bad=fixture().to_dict();bad['remanence_T']=1.
                with self.assertRaises(ValueError):wb.analyze(design=bad)
                compute.assert_not_called()


class AsyncAnalysisHTTPTests(unittest.TestCase):
    def test_second_server_cannot_silently_share_an_existing_workbench_port(self):
        class Backend:
            def state(self):return {}
        with tempfile.TemporaryDirectory() as folder:
            server=create_server(folder,port=0,workbench=Backend())
            try:
                with self.assertRaises(OSError):create_server(folder,port=server.server_port,workbench=Backend())
            finally:server.server_close()

    def test_analysis_runs_in_background_is_exclusive_and_does_not_mislabel_solver_job(self):
        class Backend:
            def __init__(self):self.entered=threading.Event();self.release=threading.Event();self.saved=None
            def state(self):return {'config':{},'steps':[],'job':{},'analysis':self.saved}
            def analyze(self,**options):
                self.entered.set();self.release.wait(3);self.saved={'design_sha256':'test'};return self.saved
        with tempfile.TemporaryDirectory() as folder:
            backend=Backend();server=create_server(folder,port=0,workbench=backend)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def request(path,body=None):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                conn.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),{'Content-Type':'application/json'})
                response=conn.getresponse();value=json.loads(response.read());conn.close();return response.status,value
            try:
                status,_=request('/api/analysis',{'analysis_options':{'tolerance':False}})
                self.assertEqual(status,202);self.assertTrue(backend.entered.wait(1))
                status,state=request('/api/state');self.assertEqual(status,200)
                self.assertFalse(state['busy']);self.assertEqual(state['analysis_job']['status'],'running')
                self.assertEqual(request('/api/step/1',{})[0],409)
                self.assertEqual(request('/api/analysis',{})[0],409)
                backend.release.set();server.analysis_future.result(3)
                self.assertEqual(request('/api/state')[1]['analysis_job']['status'],'completed')
            finally:
                backend.release.set();server.shutdown();server.server_close();thread.join(3)

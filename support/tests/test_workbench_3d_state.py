"""Offline asset and exact, read-only design state for the interactive viewport."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.serialization import save_json
from mri_magnet.workbench import Workbench
from mri_magnet.workbench_server import create_server


def design(radius, polarity='out', count=28):
    return json.loads(json.dumps(Design((RingSpec('R01-L1',0,0,radius,20.,count,polarity,-83.5714285714),), name='view fixture').to_dict()))


class InteractiveDesignStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.wb=Workbench(self.root);self.run=self.root/'runs'/'completed';self.run.mkdir(parents=True)
        self.manifest=self.run/'归档清单.json'
        self.wb.data.update(run_dir='runs/completed',artifacts={'manifest':'runs/completed/归档清单.json'})

    def tearDown(self):self.temp.cleanup()

    def layout(self,name):
        directory=self.run/name;directory.mkdir(exist_ok=True)
        path=directory/'磁铁布局_RZ.png';path.write_bytes(b'original archived PNG')
        return path.relative_to(self.run).as_posix()

    def test_each_role_has_its_own_exact_design_and_view_never_persists_hydration(self):
        designs=[design(130.),design(140.,'down',30),design(150.,'in',32)]
        layouts=[self.layout('candidate%d'%i) for i in range(3)]
        rows=[{'candidate_id':'c%d'%i,'archive_label':'candidate%d'%i,'layout_file':layout,'design':data}
              for i,(layout,data) in enumerate(zip(layouts,designs))]
        save_json(self.manifest,{'candidates':rows})
        self.wb.data['pareto_recommendations']=[{'role':role,'candidate_id':'c%d'%i,
            'layout_file':'runs/completed/'+layouts[i]} for i,role in enumerate(('balanced','high_field','lowest_ppm'))]
        self.wb.data['best']={'candidate_id':'c0','metrics':{'pp_norm_ppm':200.}}
        self.wb.data['seeds']=[{'id':'seed','design':design(160.)}]
        self.wb.data['current_best']={'design':design(170.),'metrics':{'pp_norm_ppm':100.}}
        self.wb._save()
        saved=self.wb.state_path.read_bytes();archive=self.manifest.read_bytes();original=deepcopy(self.wb.data)
        with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('view must never evaluate physics')):
            state=self.wb.state()
        for i in range(3):
            self.assertEqual(state['artifacts']['candidate_layouts'][i]['design'],designs[i])
            self.assertEqual(state['pareto_recommendations'][i]['design'],designs[i])
        self.assertEqual(state['best']['design'],designs[0])
        self.assertEqual(state['current_best'],original['current_best'])
        self.assertEqual(state['seeds'],original['seeds'])
        state['pareto_recommendations'][0]['design']['rings'][0]['radius_mm']=999.
        self.assertEqual(self.wb.state()['best']['design'],designs[0])
        self.assertEqual(self.wb.data,original)
        self.assertEqual(self.wb.state_path.read_bytes(),saved)
        self.assertEqual(self.manifest.read_bytes(),archive)

    def test_legacy_design_json_is_loaded_and_its_cache_tracks_file_changes(self):
        layout=self.layout('legacy')
        path=self.run/'legacy'/'设计.json';save_json(path,design(130.))
        save_json(self.manifest,{'candidates':[{'candidate_id':'legacy','layout_file':layout}]})
        first=self.wb.state()['artifacts']['candidate_layouts'][0]
        self.assertEqual(first['design']['rings'][0]['radius_mm'],130.)
        self.assertEqual(first['design_file'],'runs/completed/legacy/设计.json')
        save_json(path,design(140.))
        second=self.wb.state()['artifacts']['candidate_layouts'][0]
        self.assertEqual(second['design']['rings'][0]['radius_mm'],140.)

    def test_explicit_design_file_and_manifest_replacements_invalidate_cache(self):
        layout=self.layout('explicit');path=self.run/'specific.json';save_json(path,design(125.))
        save_json(self.manifest,{'candidates':[{'candidate_id':'first','layout_file':layout,'design_file':'specific.json'}]})
        first=self.wb.state()['artifacts']['candidate_layouts'][0]
        self.assertEqual(first['design'],design(125.))
        save_json(path,design(135.));self.assertEqual(self.wb.state()['artifacts']['candidate_layouts'][0]['design'],design(135.))
        save_json(self.manifest,{'candidates':[{'candidate_id':'second','layout_file':layout,'design':design(145.)}]})
        second=self.wb.state()['artifacts']['candidate_layouts'][0]
        self.assertEqual(second['candidate_id'],'second');self.assertEqual(second['design'],design(145.))

    def test_same_size_same_metadata_cloud_drive_update_is_not_stale(self):
        path=self.run/'specific.json';save_json(path,design(125.))
        known=path.stat()
        self.assertEqual(self.wb._view_json(path),design(125.))
        save_json(path,design(135.))
        with patch('pathlib.Path.stat',return_value=known):
            self.assertEqual(self.wb._view_json(path),design(135.))

    def test_candidate_id_and_layout_conflict_never_substitute_another_design(self):
        layouts=[self.layout('a'),self.layout('b')]
        save_json(self.manifest,{'candidates':[{'candidate_id':'a','layout_file':layouts[0],'design':design(130.)},
            {'candidate_id':'b','layout_file':layouts[1],'design':design(140.)}]})
        self.wb.data['pareto_recommendations']=[{'candidate_id':'a','layout_file':'runs/completed/'+layouts[1]}]
        self.assertNotIn('design',self.wb.state()['pareto_recommendations'][0])

    def test_foreign_files_and_malformed_json_never_enter_view_state(self):
        with tempfile.TemporaryDirectory() as outside:
            foreign=Path(outside)/'design.json';save_json(foreign,design(999.))
            layout=self.layout('safe')
            save_json(self.manifest,{'candidates':[
                {'candidate_id':'safe','layout_file':layout,'design_file':str(foreign)},
                {'candidate_id':'escape','layout_file':str(foreign),'design':design(800.)}]})
            rows=self.wb.state()['artifacts']['candidate_layouts']
            self.assertEqual(len(rows),1);self.assertNotIn('design',rows[0]);self.assertNotIn('design_file',rows[0])
            self.assertIsNone(self.wb._project_file('invalid\x00.json'))
            self.manifest.write_text('invalid JSON',encoding='utf-8')
            self.assertNotIn('candidate_layouts',self.wb.state()['artifacts'])

    def test_archive_only_recommendations_are_mapped_to_project_relative_paths(self):
        layout=self.layout('archive_only')
        save_json(self.manifest,{'candidates':[{'candidate_id':'c','layout_file':layout,'design':design(130.)}],
            'pareto_recommendations':[{'role':'balanced','candidate_id':'c','layout_file':layout}]})
        row=self.wb.state()['pareto_recommendations'][0]
        self.assertEqual(row['layout_file'],'runs/completed/'+layout)
        self.assertEqual(row['design'],design(130.))

    def test_live_best_is_a_coherent_snapshot_and_not_replaced_by_archived_best(self):
        self.wb.data['current_best']={'design':design(127.),'metrics':{'marker':127.}}
        self.wb.data['best']={'design':design(150.),'candidate_id':'old'}
        def updates():
            for value in range(120,150):
                with self.wb._lock:
                    self.wb.data['current_best']={'design':design(float(value)),'metrics':{'marker':float(value)}}
        worker=threading.Thread(target=updates);worker.start()
        for unused in range(20):
            state=self.wb.state();record=state['current_best']
            self.assertEqual(record['design']['rings'][0]['radius_mm'],record['metrics']['marker'])
        worker.join()
        self.assertEqual(self.wb.state()['current_best']['design'],design(149.))
        self.assertEqual(self.wb.state()['best']['design'],design(150.))


class OfflineViewportAssetTests(unittest.TestCase):
    def test_fixed_local_asset_has_correct_mime_and_arbitrary_assets_are_rejected(self):
        class Backend:
            def state(self):return {'current_best':{'design':design(130.)},'steps':[]}
        with tempfile.TemporaryDirectory() as root:
            server=create_server(root,workbench=Backend(),port=0)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            def request(route):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=5)
                conn.request('GET',route);response=conn.getresponse()
                result=(response.status,response.getheader('Content-Type'),response.read())
                conn.close();return result
            try:
                fixture=b'window.MRIMagnet3D={Viewer: function(){}};'
                with patch('mri_magnet.workbench_server.Path.is_file',return_value=True), patch('mri_magnet.workbench_server.Path.read_bytes',return_value=fixture):
                    status,mime,data=request('/assets/workbench_3d.js')
                self.assertEqual(status,200);self.assertEqual(mime,'application/javascript; charset=utf-8');self.assertEqual(data,fixture)
                for route in ['/assets/workbench.py','/assets/../workbench.py','/assets/%2e%2e/workbench.py','/assets/missing.js']:
                    self.assertEqual(request(route)[0],404)
                with patch('mri_magnet.workbench_server.Path.is_file',return_value=False):
                    self.assertEqual(request('/assets/workbench_3d.js')[0],404)
                self.assertEqual(json.loads(request('/api/state')[2])['current_best']['design'],design(130.))
            finally:
                server.shutdown();server.server_close();worker.join()


if __name__=='__main__':unittest.main()

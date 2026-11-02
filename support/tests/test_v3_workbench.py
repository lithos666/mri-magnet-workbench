"""Real field-backed workbench stop, snapshot, topology and drawing contracts."""
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mri_magnet.field import FieldEngine
from mri_magnet.geometry import Design
from mri_magnet.layout import layout_manifest
from mri_magnet.manufacturing import audit_manufacturing
from mri_magnet.metrics import evaluate_design
from mri_magnet.workbench import Workbench,FieldEvaluator,envelope_for,requirements_for
from mri_magnet.workbench_domains import make_sample
from mri_magnet.workbench_archive import physical_change_rows
from test_islands import seed,PROCESS


def configure(wb,mode='de_novo'):
    wb.update_config({'mode':mode,'source_geometry':'synthetic_layout.json',
        'target':{'geometry':'sphere','diameter_mm':10.,'z_mm':0.,'min_field_mT':.001,'max_ppm':1e6},
        'envelope':{'protected_bore_diameter_mm':40.,'max_outer_diameter_mm':200.,'max_length_mm':300.},
        'magnet':{'cube_mm':4.,'remanence_T':1.43,'reference_mur':1.05},
        'manufacturing':deepcopy(PROCESS),'sampling':{'training_volume':8,'training_surface':12,
            'validation_volume':16,'validation_surface':24,'refine_starts':1},
        'search':{'azimuth_count_candidates':[24,28,32],'min_radial_rings_per_layer':1,
            'max_radial_rings_per_layer':2,'topology_limit':4,'stock_aware_column_neighbors':True},
        'solver':{'max_evaluations':80,'population_size':5,'maxiter':20,'local_refine_evaluations':8,
            'active_set_evaluations':0,'max_workers':1,'budget_policy':'allow_partial'},
        'evolution':{'enabled':False},'deliverables':{'compile':False},'max_validation_candidates':1})


def install_seed(wb):
    design=seed(24,5);sample=make_sample(wb.config,designs=[design])
    metrics,_=evaluate_design(design,FieldEngine(),sample)
    audit=audit_manufacturing(design,wb.config['manufacturing'],envelope_for(wb.config))
    if not audit['valid']:raise AssertionError(audit)
    wb.data['seeds']=[{'id':'synthetic_01','style':'紧凑测试拓扑','design':design.to_dict(),
        'metrics':metrics,'training_metrics':metrics,'audit':audit}]
    wb.data['selected_seed_ids']=['synthetic_01']
    for row in wb.data['steps'][:4]:row.update(status='passed',message='fixture checked')
    wb._save()
    return design


class V3WorkbenchIntegrationTests(unittest.TestCase):
    def test_real_stop_preserves_result_unlocks_dense_step_and_snapshot_does_not_change_solver(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);configure(wb);install_seed(wb)
            blocked=threading.Event();release=threading.Event();output=[];errors=[];calls=[]
            original=FieldEvaluator.evaluate_with_field
            def controlled(evaluator,design):
                calls.append(design)
                if len(calls)==3:
                    blocked.set()
                    if not release.wait(10):raise RuntimeError('test release timeout')
                return original(evaluator,design)
            def solve():
                try:output.append(wb.run_step(5))
                except BaseException as error:errors.append(error)
            with patch.object(FieldEvaluator,'evaluate_with_field',controlled):
                thread=threading.Thread(target=solve);thread.start()
                try:
                    self.assertTrue(blocked.wait(10))
                    state=wb.state();job=state['job']
                    for key in ('best_mean_mT','best_ppm','max_evaluations','progress_percent','topology_name'):
                        self.assertIn(key,job)
                    self.assertGreater(job['best_mean_mT'],0)
                    self.assertEqual(state['steps'][4]['status'],'running')
                    snapshot=wb.export_current(compile=False)
                    self.assertFalse(snapshot['dense_verified']);self.assertFalse(snapshot['fem_verified'])
                    self.assertTrue((Path(folder)/snapshot['artifacts']['java']).is_file())
                    after=wb.state();self.assertEqual(after['steps'][4]['status'],'running')
                    self.assertEqual(after['current_best'],state['current_best'])
                    wb.cancel()
                finally:
                    release.set();thread.join(10)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(output[0]['state'],'stopped_with_result')
            self.assertEqual(wb.data['steps'][4]['status'],'stopped_with_result')
            saved=json.loads((wb._run_root()/'详细数据'/'优化结果.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['state'],'stopped_with_result');self.assertEqual(len(saved['records']),3)
            verified=wb.run_step(6,export_comsol=True,compile=False)
            self.assertEqual(wb.data['steps'][5]['status'],'passed')
            self.assertTrue((Path(folder)/verified['artifacts']['java']).is_file())
            manifest=json.loads((wb._run_root()/'归档清单.json').read_text(encoding='utf-8'))
            candidate=manifest['candidates'][0]
            dims=wb._run_root()/candidate['folder']/'尺寸改动_毫米.json'
            rows=json.loads(dims.read_text(encoding='utf-8'))['rows']
            for row in rows:
                if row['中心R_mm'] is not None:self.assertAlmostEqual(row['中心R_mm']-row['底边R_mm'],2.)
            self.assertEqual(saved['metadata']['field_evaluations'],3)

    def test_seeded_topology_switches_create_real_legal_count_variants(self):
        source=seed(24,5)
        extra=tuple(replace(r,id=r.id+'_outer',slot_index=1,radius_mm=r.radius_mm+15.) for r in source.rings)
        source=replace(source,rings=source.rings+extra)
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(Path(folder)/'open');configure(wb,'seeded')
            wb.update_config({'search':{'optimize_azimuth_count':True,'optimize_radial_ring_count':True}})
            for row in wb.data['steps'][:2]:row['status']='passed'
            with patch('mri_magnet.workbench.load_source',return_value=(source,{})):
                result=wb.run_step(3)
            self.assertGreater(len(result['seeds']),1)
            designs=[Design.from_dict(s['design']) for s in result['seeds']]
            self.assertTrue(any(d.rings[0].count!=24 for d in designs))
            self.assertTrue(any(len(d.rings)<len(source.rings) for d in designs))
            self.assertTrue(all(audit_manufacturing(d,wb.config['manufacturing'],envelope_for(wb.config))['valid'] for d in designs))
            fixed=Workbench(Path(folder)/'fixed');configure(fixed,'seeded')
            fixed.update_config({'search':{'optimize_azimuth_count':False,'optimize_radial_ring_count':False,'optimize_magnetization':False}})
            for row in fixed.data['steps'][:2]:row['status']='passed'
            with patch('mri_magnet.workbench.load_source',return_value=(source,{})):
                one=fixed.run_step(3)
            self.assertEqual(len(one['seeds']),1)

    def test_dimension_rows_match_diagram_bottom_labels_for_non_12mm_magnets(self):
        original=seed(24,5);moved=replace(original,rings=tuple(replace(r,radius_mm=r.radius_mm+1.) for r in original.rings))
        rows=physical_change_rows(moved,original);manifest=layout_manifest(moved)
        image_rows={(r['layer_index'],r['slot_index']):r for r in manifest['rings']}
        for row in rows:
            layer=int(row['层'][1:])-1;slot=row['径向槽']-1;diagram=image_rows[(layer,slot)]
            self.assertEqual(row['底边R_mm'],diagram['display_radius_mm'])
            self.assertEqual(row['中心R_mm'],diagram['center_r_mm'])


if __name__=='__main__':unittest.main()

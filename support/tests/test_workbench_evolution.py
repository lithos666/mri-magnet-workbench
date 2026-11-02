"""The 3D monitor must sample the stated DSV and compare the matching island."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.evolution import EvolutionMonitor
from mri_magnet.workbench import Workbench
from mri_magnet.workbench_config import default_config


class PreviewEngine:
    queries=[]
    def __init__(self,*args,**kwargs):self.statistics={'engine_calls':0,'field_points':0}
    def field(self,design,points):
        p=np.asarray(points);self.queries.append(p.copy());self.statistics['engine_calls']+=1
        return np.c_[np.zeros((len(p),2)),.04+1e-9*(p[:,0]**2-2*p[:,2]**2)]


def topology(pairs):
    rings=[]
    for i in range(pairs):
        z=20.+20*(pairs-1-i)
        rings.extend([RingSpec('positive%d'%i,i,0,140.,z,24,'out',-82.5),
                      RingSpec('negative%d'%i,2*pairs-1-i,0,140.,-z,24,'in',-82.5)])
    return Design(tuple(rings))


class WorkbenchEvolutionTests(unittest.TestCase):
    def test_sphere_slice_and_axial_lines_stay_inside_ball_and_follow_matching_island(self):
        cfg=default_config();cfg['evolution'].update(grid_points=9,max_frames=4,min_refresh_seconds=0)
        m={'mean_norm_T':.04,'pp_norm_ppm':100.};a,b=topology(5),topology(6)
        PreviewEngine.queries=[]
        with tempfile.TemporaryDirectory() as folder,patch('mri_magnet.evolution.FieldEngine',PreviewEngine):
            monitor=EvolutionMonitor(folder,a,m,{},cfg,reference_designs=[(a,m),(b,m)])
            monitor({'event':'baseline','design':a.to_dict(),'metrics':m,'topology_index':1})
            monitor({'event':'final','design':b.to_dict(),'metrics':dict(m,pp_norm_ppm=90),'topology_index':2})
            self.assertIs(monitor.seed,b)
            result=monitor.finish()
            self.assertIsNone(result['error']);self.assertEqual(result['region']['geometry'],'sphere')
            self.assertTrue((Path(folder)/result['gif_file']).is_file())
            self.assertEqual(len(result['frames']),result['frames_count'])
            self.assertEqual(result['frames_path_base'],'run')
            self.assertGreater(len(result['frames']),1)
            self.assertTrue(all((Path(folder)/frame['image']).is_file() for frame in result['frames']))
            self.assertIn('球形 DSV',(Path(folder)/result['timeline_file']).read_text(encoding='utf-8'))
        for points in PreviewEngine.queries:
            self.assertTrue(np.all(points[:,1]==0))
            self.assertTrue(np.all(np.linalg.norm(points,axis=1)<=10+1e-9))
            self.assertTrue(np.any(points[:,2]!=0))

    def test_result_rename_rebases_all_step_and_seed_previews(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.data['run_dir']='runs/old'
            wb.data['steps'][1]['preview']='runs/old/工艺.png'
            wb.data['seeds']=[{'preview':'runs/old/起点.png'}]
            wb.data['artifacts']={'layout':'runs/old/layout.png'}
            wb._relocate_references('runs/old','runs/new')
            self.assertEqual(wb.data['run_dir'],'runs/new')
            self.assertEqual(wb.data['steps'][1]['preview'],'runs/new/工艺.png')
            self.assertEqual(wb.data['seeds'][0]['preview'],'runs/new/起点.png')
            self.assertEqual(wb.data['artifacts']['layout'],'runs/new/layout.png')

    def test_budget_edit_forks_run_preserving_the_completed_archive_and_input_previews(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);old=wb._run_root()
            frozen=old/'详细数据'/'优化结果.json';frozen.write_text('{"frozen":true}',encoding='utf-8')
            preview=old/'采样点分布.png';preview.write_bytes(b'original preview')
            wb.data['steps'][3].update(status='passed',preview=wb._relative(preview))
            wb.update_config({'solver':{'max_evaluations':wb.config['solver']['max_evaluations']+100}})
            new=wb._run_root()
            self.assertNotEqual(old,new)
            self.assertEqual(frozen.read_text(encoding='utf-8'),'{"frozen":true}')
            self.assertEqual(preview.read_bytes(),b'original preview')
            self.assertEqual((new/'采样点分布.png').read_bytes(),b'original preview')
            self.assertFalse((new/'详细数据'/'优化结果.json').exists())
            self.assertEqual(wb.data['steps'][3]['status'],'passed')
            self.assertTrue((new/'详细数据'/'代码快照SHA256.json').is_file())

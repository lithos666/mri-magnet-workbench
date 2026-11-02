"""Real constraint-only Step 1--4 contract, including 35 mT 3D DSV fields."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mri_magnet.geometry import Design
from mri_magnet.workbench import Workbench, envelope_for
from mri_magnet.manufacturing import audit_manufacturing


class ConstraintOnlyWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.workbench = Workbench(cls.root)
        cls.workbench.update_config({
            'mode':'de_novo', 'source_geometry':'DOES_NOT_EXIST/placement-05.json',
            'target':{'geometry':'sphere','diameter_mm':20.,'z_mm':0.,'min_field_mT':35.,'max_ppm':3000.},
            'de_novo':{'pair_counts':[5,6,7], 'column_counts':[24,28,32], 'seed_count':3,
                       'milp_time_limit_seconds':3.,'max_attempts':6},
            'sampling':{'preset':'quick','training_volume':16,'training_surface':24,
                        'validation_volume':32,'validation_surface':48},
            'evolution':{'enabled':False},'deliverables':{'compile':False},
        })
        with patch('mri_magnet.workbench.load_source', side_effect=AssertionError('source forbidden')), \
             patch('mri_magnet.geometry.load_source', side_effect=AssertionError('source forbidden')):
            cls.results = [cls.workbench.run_step(i) for i in range(1,5)]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_steps_complete_and_state_previews_are_persisted(self):
        reloaded = Workbench(self.root).state()
        self.assertTrue(all(s['status']=='passed' for s in reloaded['steps'][:4]))
        for number in (2,3,4):
            self.assertTrue((self.root/reloaded['steps'][number-1]['preview']).is_file())
        self.assertEqual(reloaded['config']['target']['geometry'],'sphere')
        self.assertEqual(reloaded['config']['target']['diameter_mm'],20.)

    def test_no_source_cluster_is_heterogeneous_and_physically_legal(self):
        state=self.workbench.state(); seeds=state['seeds']
        self.assertGreaterEqual(len(seeds),3)
        self.assertFalse(state['synthesis_metadata']['source_layout_read'])
        self.assertGreaterEqual(len({s['pair_count'] for s in seeds}),2)
        limits=envelope_for(state['config'])
        for seed in seeds:
            design=Design.from_dict(seed['design'])
            self.assertTrue(audit_manufacturing(design,state['config']['manufacturing'],limits)['valid'])
            self.assertLessEqual(design.magnet_count,state['config']['manufacturing']['max_magnet_count'])

    def test_real_35mt_sphere_initial_fields_have_useful_uniformity(self):
        seeds=self.workbench.state()['seeds']
        for seed in seeds:
            self.assertGreaterEqual(seed['metrics']['center_Bz_T'],.035-1e-8)
            self.assertEqual(seed['metrics']['evaluation_geometry'],'sphere')
        self.assertLess(min(s['metrics']['pp_norm_ppm'] for s in seeds),100.)
        self.assertGreaterEqual(sum(s['metrics']['pp_norm_ppm']<3000. for s in seeds),2)

    def test_common_training_and_independent_global_validation_pointsets(self):
        state=self.workbench.state()
        self.assertTrue(state['sampling']['symmetry']['verified'])
        self.assertTrue(state['sampling']['symmetry']['used'])
        selected={s['id']:s for s in state['seeds']}
        pointsets={selected[i]['training_metrics']['pointset'] for i in state['selected_seed_ids']}
        self.assertEqual(len(pointsets),1)
        self.assertNotEqual(state['sampling']['training_count'],state['sampling']['validation_count'])


if __name__=='__main__':
    unittest.main()

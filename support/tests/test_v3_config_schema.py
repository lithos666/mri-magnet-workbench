"""UI/CLI share explicit domains; historical units migrate without guessing."""
from pathlib import Path
import tempfile,json,unittest,runpy
from types import SimpleNamespace
from mri_magnet.config_schema import canonicalize_config,legacy_view
from mri_magnet.workbench_config import default_config
from mri_magnet.workbench import Workbench
from mri_magnet.cli import settings


class ConfigSchemaTests(unittest.TestCase):
    def test_flat_domain_migration_and_conflicts(self):
        self.assertEqual(canonicalize_config({'fov_diameter_mm':22.,'fov_z_mm':1.})['target'],
            dict(geometry='disk',diameter_mm=22.,z_mm=1.,min_field_mT=30.,max_ppm=1000.))
        self.assertEqual(canonicalize_config({'dsv_diameter_mm':28.})['target']['geometry'],'sphere')
        with self.assertRaises(ValueError):canonicalize_config({'dsv_diameter_mm':20.,'fov_diameter_mm':20.})
        with self.assertRaises(ValueError):canonicalize_config({'target':{'geometry':'sphere'},'fov_diameter_mm':20.})

    def test_derived_legacy_view_roundtrips_one_authority(self):
        cfg=default_config();cfg['target'].update(geometry='sphere',diameter_mm=36.,min_field_mT=42.)
        cfg['manufacturing']['max_magnet_count']=2000
        back=canonicalize_config(legacy_view(cfg))
        self.assertEqual(back['target'],cfg['target']);self.assertEqual(back['manufacturing'],cfg['manufacturing'])

    def test_default_cli_and_workbench_use_identical_target_and_process(self):
        root=Path(__file__).resolve().parents[2]
        args=SimpleNamespace(parameters=str(root/'parameters.py'),config=None,source=None,mph=None,budget=None,quick=False)
        configured=settings(args)
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder)
            self.assertEqual(configured['canonical_config']['target'],wb.config['target'])
            self.assertEqual(configured['canonical_config']['manufacturing'],wb.config['manufacturing'])
            self.assertTrue(configured['_use_workbench_workflow'])

    def test_workbench_accepts_legacy_sphere_key_and_validates_it(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config({'dsv_diameter_mm':30.})
            self.assertEqual(wb.config['mode'],'de_novo')
            self.assertEqual(wb.config['target']['geometry'],'sphere')
            self.assertEqual(wb.config['target']['diameter_mm'],30.)
            self.assertFalse(wb.preflight()['field_evaluations'])

    def test_invalid_saved_form_can_be_reopened_and_corrected(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config({'solver':{'population_size':2}})
            reopened=Workbench(folder)
            self.assertFalse(reopened.preflight()['valid'])
            reopened.update_config({'solver':{'population_size':12}})
            self.assertTrue(reopened.preflight()['valid'])

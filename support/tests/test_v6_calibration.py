"""A FEM calibration certificate must bind geometry, material and sample data."""
from copy import deepcopy
from pathlib import Path
import hashlib,tempfile,unittest
from unittest.mock import patch
import numpy as np
from mri_magnet.physics_calibration import calibrate_design
from mri_magnet.comsol_validation import sample_fingerprint
from mri_magnet.comsol_bridge import write_field_csv
from mri_magnet.identity import canonical_physical_sha256
from mri_magnet.field_models import make_field_engine
from test_v6_coupled_workflow import config,design
from mri_magnet.workbench_domains import make_sample


class CalibrationBindingTests(unittest.TestCase):
    def reference(self,folder,cfg,d,sample):
        values=make_field_engine(cfg).field(d,sample['points_mm'])
        path=Path(folder)/'reference.csv';write_field_csv(path,sample['points_mm'],values)
        return {'fem_verified':True,'design_sha256':canonical_physical_sha256(d),
            'sample_fingerprint':sample_fingerprint(sample),'reference_field_csv':str(path),
            'reference_field_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_another_design_cannot_supply_its_fem_certificate(self):
        cfg=config();d=design();sample=make_sample(cfg)
        with tempfile.TemporaryDirectory() as folder:
            report=self.reference(folder,cfg,d,sample)
            with patch('mri_magnet.field_models.make_field_engine',side_effect=AssertionError('must reject before solve')):
                with self.assertRaisesRegex(ValueError,'different physical design'):
                    calibrate_design(design(41.),cfg,sample,Path(folder)/'calibrate',report)

    def test_changed_weights_and_modified_csv_are_rejected(self):
        cfg=config();d=design();sample=make_sample(cfg)
        with tempfile.TemporaryDirectory() as folder:
            report=self.reference(folder,cfg,d,sample)
            changed=deepcopy(sample);changed['weights']=np.roll(changed['weights'],1)
            with self.assertRaisesRegex(ValueError,'different observation'):
                calibrate_design(d,cfg,changed,Path(folder)/'a',report)
            Path(report['reference_field_csv']).write_text('tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                calibrate_design(d,cfg,sample,Path(folder)/'b',report)

    def test_matching_reference_selects_only_fem_accepted_profile(self):
        cfg=config();d=design();sample=make_sample(cfg)
        with tempfile.TemporaryDirectory() as folder:
            report=self.reference(folder,cfg,d,sample)
            result=calibrate_design(d,cfg,sample,Path(folder)/'calibrate',report)
        self.assertTrue(result['model_accuracy_passed'])
        self.assertEqual(result['selected_physics']['quadrature_order'],2)
        self.assertEqual(result['selected_physics']['cells_per_axis'],1)

    def test_failed_fem_never_certifies_python_even_when_all_profiles_compute(self):
        cfg=config();d=design();sample=make_sample(cfg)
        with tempfile.TemporaryDirectory() as folder:
            result=calibrate_design(d,cfg,sample,folder,{'status':'failed','fem_verified':False})
        self.assertFalse(result['model_accuracy_passed'])
        self.assertIsNone(result['selected_physics'])

if __name__=='__main__':unittest.main()

"""Readable export checks without running or mocking physical field values."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.workbench_archive import archive_run


def config():
    return {'target': {'geometry': 'sphere', 'diameter_mm': 20., 'min_field_mT': 10., 'max_ppm': 1000.},
            'envelope': {'protected_bore_diameter_mm': 240., 'max_outer_diameter_mm': 384.187454246,
                         'max_length_mm': 506.},
            'magnet': {'cube_mm': 12., 'remanence_T': 1.43, 'reference_mur': 1.05},
            'manufacturing': {'enabled': True, 'layout_mode': 'radial_channels',
                'pocket_clearance_per_side_mm': .2, 'pocket_web_mm': 1.2, 'inner_wall_mm': 1.2,
                'outer_wall_mm': 1.2, 'base_thickness_mm': 1.6, 'printer_bed_mm': [220.,220.,250.],
                'sectors_per_layer': 4, 'bed_margin_per_side_mm': 4., 'max_magnet_count': 1736,
                'min_axial_module_gap_mm': 0., 'bed_fit_mode': 'fixed_envelope'},
            'deliverables': {'enabled': True, 'compile': False}, 'comsol_root': 'no_compiler_needed'}


def design(count, radius, z):
    return Design((RingSpec('R01-L1', 0, 0, radius, z, count, 'out', -90+180/count),
                   RingSpec('R02-L1', 1, 0, radius, -z, count, 'in', -90+180/count)))


def metric(ppm=100., field=.02, pointset='dense_D20'):
    return {'mean_norm_T': field, 'pp_norm_ppm': ppm, 'pointset': pointset,
            'min_norm_T': field*(1-ppm/2e6), 'max_norm_T': field*(1+ppm/2e6)}


class ArchiveTests(unittest.TestCase):
    def test_reference_is_matching_topology_and_artifacts_survive_folder_rename(self):
        seed0, seed1 = design(24, 140., 20.), design(28, 160., 40.)
        current = design(28, 161., 41.)
        seeds = [{'id':'first', 'design':seed0.to_dict(), 'metrics':metric(500.)},
                 {'id':'second', 'design':seed1.to_dict(), 'metrics':metric(400.), 'dense_metrics':metric(600.)}]
        row = {'candidate_id':'opaque_id', 'design':current.to_dict(), 'metrics':metric(), 'topology_index':2,
               'training_metrics':metric(90., pointset='training_D20')}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            with patch('mri_magnet.field.FieldEngine.field', side_effect=AssertionError('no new fields')):
                result=archive_run(root, config(), seeds, [row], {'metadata':{}}, {}, compile_class=False)
            self.assertEqual(result['status'], 'verified_target_met')
            best=result['best']; self.assertEqual(best['reference_seed_id'], 'second')
            with (root/best['changes_file']).open(encoding='utf-8-sig',newline='') as stream:
                changes=list(csv.DictReader(stream))
            self.assertEqual(float(changes[0]['ΔR_mm']), 1.)
            self.assertEqual(float(changes[0]['ΔZ_mm']), 1.)
            self.assertEqual(best['model_export']['status'], 'java_generated_not_compiled')
            self.assertIsNone(best['model_export']['class_file'])
            self.assertTrue((root/best['model_export']['java_file']).is_file())
            overview=(root/'本次运行速览.md').read_text(encoding='utf-8')
            self.assertIn('| 对应拓扑起点 | 20 | 600 |', overview)
            self.assertIn('未执行或未记录', overview)
            new=Path(folder)/'renamed_20mT_100ppm'; root.rename(new)
            for path in result['artifacts'].values():
                self.assertTrue((new/path).is_file(),path)
            self.assertTrue((new/best['layout_file']).is_file())
            self.assertTrue((new/best['model_export']['java_file']).is_file())

    def test_unmet_threshold_candidate_is_retained_and_labelled_diagnostic(self):
        candidate=design(24, 140., 20.)
        row={'candidate_id':'not_target_met','design':candidate.to_dict(),'metrics':metric(2000.),'topology_index':1}
        with tempfile.TemporaryDirectory() as folder:
            result=archive_run(folder,config(),[],[row],{},None,export_comsol=False)
            self.assertEqual(len(result['candidates']),1)
            self.assertEqual(result['status'],'diagnostic_only')
            self.assertIn('诊断未达标',result['best']['folder'])
            self.assertFalse(result['best']['feasible'])
            self.assertIn('未取得双门槛通过方案',(Path(folder)/'本次运行速览.md').read_text(encoding='utf-8'))

    def test_empty_dense_results_still_produce_honest_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            result=archive_run(folder,config(),[],[],{},None,export_comsol=False)
            self.assertEqual(result['status'],'no_candidates')
            self.assertIsNone(result['best'])
            self.assertIn('没有可归档',(Path(folder)/'本次运行速览.md').read_text(encoding='utf-8'))


if __name__=='__main__':
    unittest.main()

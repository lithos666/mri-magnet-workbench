"""Planar FOV integration contracts: every active coordinate keeps its fixed z."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parameters import CONFIG
from mri_magnet import benchmark, cli, workflow
from mri_magnet.objectives import check_performance


class PlanarToyEngine:
    queries = []

    def __init__(self, *args, **kwargs):
        self.polarization_model = kwargs.get('polarization_model', args[1] if len(args)>1 else 'remanence')
        self.statistics = {'engine_calls': 0, 'field_points': 0, 'preparations': 1,
                           'prepared_cache_hits': 0, 'field_seconds': 0., 'preparation_seconds': 0.}

    def field(self, design, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not np.all(points[:, 2] == 0.):
            raise AssertionError('A planar FOV calculation queried an out-of-plane point')
        self.queries.append(points.copy())
        self.statistics['engine_calls'] += 1
        self.statistics['field_points'] += len(points)
        bz = .04 + 1e-6 * (points[:, 0]**2 + 2*points[:, 1]**2)
        return np.column_stack((np.zeros(len(points)), np.zeros(len(points)), bz))


class PlanarWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.source = Path(os.environ.get('MRI_SOURCE', str(root/'examples'/'seeded'/'placement.json')))
        if not cls.source.is_file():
            raise unittest.SkipTest('Set MRI_SOURCE to exercise actual-source planar integration')

    def setUp(self):
        PlanarToyEngine.queries = []

    def configuration(self, directory):
        config = copy.deepcopy(CONFIG)
        config.pop('dsv_diameter_mm', None)
        config.pop('cylinder_diameter_mm', None)
        config.pop('cylinder_length_mm', None)
        config.update(fov_diameter_mm=20., fov_z_mm=0., source_geometry=str(self.source),
                      comsol_model=str(directory/'absent.mph'), parameters_file=str(directory/'parameters.py'))
        config['deliverables']['enabled'] = False
        config['search'].update(warm_start_runs=[], topology_limit=2, seed_subset_field_selection=True)
        config['sampling'].update(training_domain='global', training_area=8, training_boundary=12,
            validation_area=16, validation_boundary=24, quick_training_area=8, quick_training_boundary=12,
            quick_validation_area=16, quick_validation_boundary=24, refine_starts=1)
        config['scan'].update(quick_axis_line_points=7, quick_plane_points=7)
        config['benchmark'] = {'refine_extrema': False}
        return config

    def test_baseline_area_reference_and_plots_never_leave_z_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); config = self.configuration(directory); out = directory/'baseline'
            with patch.object(cli, 'settings', return_value=config), \
                 patch.object(workflow, 'FieldEngine', PlanarToyEngine), \
                 patch.object(workflow, 'refine_disk_extrema', side_effect=lambda d,e,s,m,b,starts=3: m), \
                 patch('mri_magnet.reporting.create_report', return_value={'report_html': str(out/'report_中文.html')}), \
                 patch.object(cli, 'log'):
                self.assertEqual(cli.main(['baseline', '--quick', '--out', str(out)]), 0)
            payload = json.loads((out/'详细数据'/'report_payload.json').read_text(encoding='utf-8'))
            training = json.loads((out/'详细数据'/'training_sampling.json').read_text(encoding='utf-8'))
            validation = json.loads((out/'详细数据'/'validation_sampling.json').read_text(encoding='utf-8'))
            self.assertEqual(training['geometry'], 'disk')
            self.assertEqual(validation['geometry'], 'disk')
            self.assertEqual(training['measure'], 'area')
            self.assertEqual(validation['z_mm'], 0.)
            self.assertEqual(payload['model_info']['fov_geometry'], 'disk')
            self.assertEqual(payload['model_info']['fov_diameter_mm'], 20.)
            self.assertEqual(payload['model_info']['fov_z_mm'], 0.)
            self.assertNotIn('baseline_cylinder_metrics', payload['model_info'])
            self.assertNotIn('dsv_diameter_mm', payload['model_info'])
            self.assertEqual(payload['requirements']['min_mean_norm_T'], payload['scan_reference']['metrics']['mean_norm_T'])
            self.assertEqual(payload['requirements']['max_pp_norm_ppm'], payload['scan_reference']['metrics']['pp_norm_ppm'])
            self.assertEqual({row['axis'] for row in payload['plot_data']['axis_lines']}, {'x', 'y'})
            self.assertEqual([row['plane'] for row in payload['plot_data']['planes']], ['xy'])
            self.assertGreater(len(PlanarToyEngine.queries), 5)
            self.assertTrue(all(np.all(points[:, 2] == 0.) for points in PlanarToyEngine.queries))

    def test_relative_gates_keep_seed_feasible_in_training_and_validation_in_both_manufacturing_modes(self):
        for enabled in (True, False):
            with self.subTest(manufacturing_enabled=enabled), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder); config = self.configuration(directory); out = directory/'optimize'
                config['manufacturing']['enabled'] = enabled
                observed = []
                def optimize(topologies, process, limits, search, evaluator, checker, output,
                             settings, requirements, warm_starts=(), **kwargs):
                    design = topologies[0]; metrics = evaluator(design)
                    self.assertTrue(check_performance(metrics, requirements)['valid'])
                    observed.append(requirements)
                    return {'records': [], 'metadata': {'field_evaluations': 1}}
                with patch.object(cli, 'settings', return_value=config), \
                     patch.object(workflow, 'FieldEngine', PlanarToyEngine), \
                     patch.object(workflow, 'refine_disk_extrema', side_effect=lambda d,e,s,m,b,starts=3: m), \
                     patch.object(workflow, 'run_topology_optimization', side_effect=optimize), \
                     patch('mri_magnet.reporting.create_report', return_value={'report_html': str(out/'report_中文.html')}), \
                     patch.object(cli, 'log'):
                    self.assertEqual(cli.main(['optimize', '--quick', '--out', str(out)]), 0)
                payload = json.loads((out/'详细数据'/'report_payload.json').read_text(encoding='utf-8'))
                self.assertEqual(observed[0], payload['training_requirements'])
                self.assertTrue(check_performance(payload['scan_reference']['metrics'], payload['requirements'])['valid'])
                self.assertEqual(payload['requirements']['field_tolerance_T'], 0.)
                self.assertNotEqual(payload['requirements']['min_mean_norm_T'], observed[0]['min_mean_norm_T'])
                self.assertNotEqual(payload['requirements']['reference_metric_pointsets'], observed[0]['reference_metric_pointsets'])

    def test_comsol_point_export_requests_the_disk_only(self):
        captured = []
        def prepare(mph, points, output, comsol_root):
            captured.append(np.asarray(points).copy())
            return {'fixture': True}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); config = self.configuration(directory)
            with patch.object(cli, 'settings', return_value=config), \
                 patch.object(cli, 'prepare_existing_solution_export', side_effect=prepare), \
                 patch.object(cli, 'run_existing_solution_export', return_value={'status': 'success'}), \
                 patch.object(cli, 'log'):
                self.assertEqual(cli.main(['comsol-export', '--out', str(directory/'export')]), 0)
        self.assertEqual(len(captured), 1)
        self.assertTrue(np.all(captured[0][:, 2] == 0.))
        self.assertLessEqual(np.max(np.linalg.norm(captured[0][:, :2], axis=1)), 10.+1e-10)
        self.assertTrue(np.any(np.linalg.norm(captured[0][:, :2], axis=1) < 9.))

    def test_reference_csv_outside_plane_or_radius_is_rejected_before_field(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);engine=PlanarToyEngine()
            for index,point in enumerate(((0.,0.,1.),(11.,0.,0.))):
                with self.subTest(point=point):
                    reference=directory/('outside_%d.csv'%index)
                    cli.write_field_csv(reference,np.array([point]),np.array([[0.,0.,.04]]))
                    with patch.object(engine,'field',wraps=engine.field) as called:
                        with self.assertRaisesRegex(ValueError,'FOV以外'):
                            cli.compare_reference(None,engine,reference,directory,fov_diameter_mm=20.,z_mm=0.)
                        called.assert_not_called()
            points=np.array([[0.,0.,0.],[10.,0.,0.]])
            fields=np.array([[0.,0.,.04],[0.,0.,.0401]])
            reference=directory/'valid.csv'
            cli.write_field_csv(reference,points,fields)
            result=cli.compare_reference(None,engine,reference,directory,fov_diameter_mm=20.,z_mm=0.)
            self.assertEqual(result['matched_point_count'],2)
            self.assertLess(result['maximum_vector_error_T'],1e-15)

    def test_benchmark_modes_share_planar_area_pointsets(self):
        seen = []
        def optimize(topologies, process, limits, search, evaluator, checker, output,
                     settings, requirements, warm_starts=(), **kwargs):
            design = topologies[0]
            metrics = evaluator(design)
            seen.append((str(output), settings['seed'], settings['max_evaluations']))
            return {'records': [{'candidate_id': 'fixture', 'design': design.to_dict(), 'metrics': metrics}],
                    'metadata': {'field_evaluations': settings['max_evaluations'], 'topologies_run': len(topologies), 'runs': []}}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); config = self.configuration(directory)
            with patch.object(benchmark, 'FieldEngine', PlanarToyEngine), \
                 patch.object(benchmark, 'run_topology_optimization', side_effect=optimize):
                result = benchmark.run_benchmark(config, directory/'benchmark', quick=True, seeds=(101,102), budget=5)
            sample = json.loads((directory/'benchmark'/'详细数据'/'benchmark_sampling.json').read_text(encoding='utf-8'))
            for region in ('training', 'validation'):
                self.assertEqual(sample[region]['geometry'], 'disk')
                self.assertEqual(sample[region]['measure'], 'area')
                points = np.asarray(sample[region]['points_mm'])
                self.assertTrue(np.all(points[:, 2] == 0.))
                self.assertLessEqual(np.max(np.linalg.norm(points[:, :2], axis=1)), 10.+1e-10)
            self.assertEqual(len(seen), 4)
            self.assertEqual({r[2] for r in seen}, {5})
            self.assertEqual(len({row['training_sampling_sha256'] for row in result['rows']}), 1)
            self.assertEqual(len({row['validation_sampling_sha256'] for row in result['rows']}), 1)
            self.assertTrue(all(np.all(points[:, 2] == 0.) for points in PlanarToyEngine.queries))


if __name__ == '__main__':
    unittest.main()

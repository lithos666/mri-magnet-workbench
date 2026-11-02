"""Source-pose and pipeline regressions without a magnetic-field solve."""

import itertools
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mri_magnet.geometry import Design, RingSpec, envelope_limits, load_source, validate_design


class SourceGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("MRI_SOURCE")
        if override:
            cls.source_path = Path(override)
            if not cls.source_path.is_file():
                raise FileNotFoundError("MRI_SOURCE does not identify a source JSON: " + override)
        else:
            project_root = Path(__file__).resolve().parents[2]
            cls.source_path = project_root / "examples" / "seeded" / "placement.json"
            if not cls.source_path.is_file():
                raise unittest.SkipTest("Source placement JSON is absent; set MRI_SOURCE to run source-pose checks.")
        cls.baseline, cls.source = load_source(cls.source_path)

    def test_initial_parameter_vector_preserves_actual_source_design(self):
        from mri_magnet.parameterization import get_search_space
        from parameters import CONFIG
        limits=envelope_limits(self.baseline)
        process=dict(CONFIG['manufacturing'],enabled=False)
        search=dict(CONFIG['search'],max_radial_rings_per_layer=6)
        variables,builder=get_search_space(self.baseline,process,limits,search)
        reconstructed=builder([v['initial'] for v in variables])
        self.assertEqual(reconstructed.rings,self.baseline.rings)
        self.assertEqual(reconstructed.magnet_count,self.source['metadata']['totalMagnetCount'])
        self.assertTrue(validate_design(reconstructed,limits)['valid'])

    def test_all_cube_vertices_match_explicit_source_geometry(self):
        arrays = self.baseline.magnet_arrays()
        positions = arrays["positions_mm"]
        theta = np.deg2rad(arrays["radial_angles_deg"])
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        local = signs * self.baseline.cube_mm / 2.0
        vertices = np.empty((len(positions), 8, 3))
        vertices[:, :, 0] = positions[:, None, 0] + np.cos(theta)[:, None] * local[None, :, 0] - np.sin(theta)[:, None] * local[None, :, 1]
        vertices[:, :, 1] = positions[:, None, 1] + np.sin(theta)[:, None] * local[None, :, 0] + np.cos(theta)[:, None] * local[None, :, 1]
        vertices[:, :, 2] = positions[:, None, 2] + local[None, :, 2]
        explicit = {magnet["id"]: magnet for magnet in self.source["magnets"]}
        reference = np.array([explicit[identifier]["verticesXYZMM"] for identifier in arrays["ids"]])
        # Use a two-sided point-set distance, independent of corner ordering.
        distances = np.linalg.norm(vertices[:, :, None, :] - reference[:, None, :, :], axis=3)
        maximum_error = max(float(distances.min(axis=2).max()), float(distances.min(axis=1).max()))
        self.assertLess(maximum_error, 1e-6)


class DenseParetoPipelineTests(unittest.TestCase):
    def test_dense_metrics_override_coarse_feasibility_in_unified_workflow(self):
        import copy
        from parameters import CONFIG
        from mri_magnet import cli, workflow
        from mri_magnet.identity import canonical_physical_sha256
        source = Design(tuple(RingSpec('fixture-%02d'%i,i,0,130.0,z,28,'out' if i<7 else 'in',0.0)
                    for i,z in enumerate([240,205,170,135,100,65,30,-30,-65,-100,-135,-170,-205,-240])))
        candidate=replace(source,rings=tuple(replace(r,radius_mm=131.) for r in source.rings),name='candidate')
        def metrics(field,ppm):
            return dict(mean_norm_T=field,pp_norm_ppm=ppm,rms_norm_ppm=ppm/3,mean_Bz_T=field,
                        pp_Bz_ppm=ppm,center_Bz_T=field,max_transverse_T=0.,magnet_count=392,mass_magnet_kg=5.)
        def evaluate(design,engine,sample,density=7500.):
            m=metrics(.9,110.) if design.name=='candidate' else metrics(1.,100.)
            return m,np.tile([0.,0.,m['mean_norm_T']],(len(sample['points_mm']),1))
        proposed=dict(candidate_id='proposed',metrics=metrics(1.1,90.),design=candidate.to_dict(),
                      parameters={},feasible=True,canonical_physical_sha256=canonical_physical_sha256(candidate))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);config=copy.deepcopy(CONFIG)
            config.update(source_geometry=str(out/'source.json'),comsol_model=str(out/'absent.mph'),
                          parameters_file=str(out/'parameters.py'))
            config['deliverables']['enabled']=False
            config['search'].update(min_radial_rings_per_layer=1,max_radial_rings_per_layer=1,
                optimize_radius=False,optimize_axial_positions=False,optimize_azimuth_count=False,
                optimize_radial_ring_count=False,seed_subset_field_selection=False,topology_limit=1,warm_start_runs=[])
            with patch.object(cli,'settings',return_value=config), \
                 patch.object(workflow,'load_source',return_value=(source,{})), \
                 patch.object(workflow,'FieldEngine') as engine_type, \
                 patch.object(workflow,'evaluate_design',side_effect=evaluate), \
                 patch.object(workflow,'refine_disk_extrema',side_effect=lambda d,e,s,m,b,starts=3:m), \
                 patch.object(workflow,'run_topology_optimization',return_value={'records':[proposed],'metadata':{}}), \
                 patch.object(cli,'plot_data',return_value={}), \
                 patch('mri_magnet.reporting.create_report',return_value={'report_html':str(out/'report.html')}), \
                 patch.object(cli,'log'):
                engine_type.return_value.statistics={'engine_calls':0}
                self.assertEqual(cli.main(['optimize','--quick','--out',str(out)]),0)
                engine_type.return_value.field.assert_not_called()
            payload=json.loads((out/'详细数据'/'report_payload.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['verified_pareto_ids'],['baseline'])
            row=next(r for r in payload['candidates'] if r['candidate_id']=='proposed')
            self.assertFalse(row['performance_check']['valid'])
            self.assertFalse(row['feasible'])
            self.assertEqual(row['training_metrics']['mean_norm_T'],1.1)


if __name__ == "__main__":
    unittest.main()

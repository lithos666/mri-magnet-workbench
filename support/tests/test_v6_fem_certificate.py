"""Engineering thresholds and resumed FEM evidence cannot silently weaken."""
from copy import deepcopy
from pathlib import Path
import hashlib,tempfile,unittest
import numpy as np
from mri_magnet.comsol_validation import DEFAULTS,ENGINEERING_LIMITS,_settings,_cached_task,_task_fingerprint,prepare_fem_validation,classify_fem_failure
from mri_magnet.geometry import Design,RingSpec
from mri_magnet.comsol_bridge import write_field_csv


class FemCertificateTests(unittest.TestCase):
    def design(self):
        return Design((RingSpec('R01-L1',0,0,32.,22.,8,'out',22.5),
                       RingSpec('R02-L1',1,0,32.,-22.,8,'in',22.5)),cube_mm=6.)

    def test_product_default_auto_resolves_compatible_sector_and_cubic_cg_amg(self):
        sample={'points_mm':[[0.,0.,0.],[2.,0.,0.]],'weights':[1.,0.]}
        with tempfile.TemporaryDirectory() as folder:
            job=prepare_fem_validation(self.design(),sample,folder,'missing')
            self.assertEqual(job['requested_symmetry_mode'],'auto')
            self.assertEqual(job['settings']['symmetry_mode'],'rotational_sector')
            self.assertEqual(job['settings']['element_order'],3)
            self.assertEqual(job['settings']['air_radius_mm'],3000.)
            source=Path(job['tasks'][0]['java_file']).read_text(encoding='ascii')
            self.assertIn('"order_magneticscalarpotential",3',source)
            self.assertIn('"linsolver","cg"',source)
            self.assertIn('"prefun","amg"',source)

    def test_auto_falls_back_to_full_for_incompatible_phase(self):
        from dataclasses import replace
        d=self.design();d=replace(d,rings=(d.rings[0],replace(d.rings[1],phase_deg=23.)))
        sample={'points_mm':[[0.,0.,0.]],'weights':[1.]}
        with tempfile.TemporaryDirectory() as folder:
            job=prepare_fem_validation(d,sample,folder,'missing')
            self.assertEqual(job['settings']['symmetry_mode'],'full')
            self.assertIn('phase',job['symmetry_resolution_reason'])
            self.assertFalse(job['symmetry']['enabled'])

    def test_final_union_absolute_repair_is_fixed_in_mm_before_geometry_build(self):
        sample={'points_mm':[[0.,0.,0.]],'weights':[1.]}
        for symmetry in ('full','auto'):
            with self.subTest(symmetry=symmetry), tempfile.TemporaryDirectory() as folder:
                job=prepare_fem_validation(self.design(),sample,folder,'missing',
                    settings={'symmetry_mode':symmetry})
                for task in job['tasks']:
                    source=Path(task['java_file']).read_text(encoding='ascii')
                    units=source.index('g.lengthUnit("mm")')
                    mode=source.index('g.feature("fin").set("repairtoltype","absolute")')
                    tolerance=source.index('g.feature("fin").set("absrepairtol",1e-6)')
                    build=source.index('g.run();')
                    self.assertLess(units,mode)
                    self.assertLess(mode,tolerance)
                    self.assertLess(tolerance,build)

    def test_memory_exhaustion_is_distinct_from_convergence_failure(self):
        self.assertEqual(classify_fem_failure('LU 因式分解时内存不足。'),'resource_memory_exhausted')
        self.assertEqual(classify_fem_failure('Java heap space'),'resource_memory_exhausted')
        self.assertEqual(classify_fem_failure('返回的解不收敛'),'solver_or_runtime_error')

    def test_every_engineering_gate_rejects_looser_limits(self):
        for key in ('mean_relative_tolerance','ppm_absolute_tolerance','ppm_relative_tolerance',
                    'mesh_mean_relative_tolerance','convergence_ppm_fraction','solver_relative_tolerance'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError,'V6 engineering acceptance'):
                    _settings({key:ENGINEERING_LIMITS[key]*1.01})

    def test_stricter_engineering_limits_remain_supported(self):
        tighter={key:DEFAULTS[key]*.5 for key in ('mean_relative_tolerance','ppm_absolute_tolerance',
            'ppm_relative_tolerance','mesh_mean_relative_tolerance','convergence_ppm_fraction','solver_relative_tolerance')}
        result=_settings(tighter)
        self.assertTrue(all(result[key]==value for key,value in tighter.items()))

    def fixture(self,folder):
        points=np.array([[0.,0.,0.],[1.,0.,0.]])
        values=np.array([[0.,0.,.04],[0.,0.,.040001]])
        path=Path(folder)/'field.csv';write_field_csv(path,points,values)
        task={'name':'mesh_01','output_csv':str(path),'java_sha256':'source-a',
            'mesh_size':7,'air_radius_mm':650.,'magnet_hmax_mm':6.,'target_hmax_mm':8.,'target_center_z_mm':0.}
        job={'design_sha256':'design','sample_fingerprint':'sample','settings':dict(DEFAULTS),
             'symmetry':{'enabled':False},'rotation_angles_rad':None}
        item={'name':'mesh_01','status':'solved_sampled','task_fingerprint':_task_fingerprint(task,job),
              'field_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return points,values,task,job,{'tasks':[item]}

    def test_matching_completed_field_is_reused_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            points,values,task,job,prior=self.fixture(folder)
            item,actual=_cached_task(task,prior,points,job)
            self.assertTrue(item['reused_completed_task'])
            np.testing.assert_array_equal(actual,values)

    def test_source_mesh_order_and_mapping_changes_cannot_reuse_old_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            points,_,task,job,prior=self.fixture(folder)
            for key,value in (('java_sha256','source-b'),('magnet_hmax_mm',3.),('air_radius_mm',975.)):
                changed=dict(task,**{key:value})
                with self.subTest(key=key):self.assertIsNone(_cached_task(changed,prior,points,job))
            changed=deepcopy(job);changed['settings']['element_order']=2
            self.assertIsNone(_cached_task(task,prior,points,changed))
            changed=deepcopy(job);changed['rotation_angles_rad']=[0.,1.]
            self.assertIsNone(_cached_task(task,prior,points,changed))

    def test_legacy_unbound_report_is_not_a_reusable_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            points,_,task,job,prior=self.fixture(folder)
            prior['tasks'][0].pop('task_fingerprint')
            self.assertIsNone(_cached_task(task,prior,points,job))

    def test_stricter_acceptance_does_not_discard_same_numerical_solution(self):
        with tempfile.TemporaryDirectory() as folder:
            points,_,task,job,prior=self.fixture(folder)
            changed=deepcopy(job);changed['settings']['mean_relative_tolerance']=.001
            self.assertIsNotNone(_cached_task(task,prior,points,changed))

    def test_modified_csv_with_matching_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            points,_,task,job,prior=self.fixture(folder)
            Path(task['output_csv']).write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'CSV changed'):_cached_task(task,prior,points,job)

if __name__=='__main__':unittest.main()

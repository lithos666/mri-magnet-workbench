import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from mri_magnet.comsol_validation import (prepare_fem_validation, run_fem_validation,
    compare_fem_fields, parse_fem_rows, sample_fingerprint, rotational_sector,
    restore_sector_fields, _convergence, _settings)
from mri_magnet.geometry import Design,RingSpec


class FemVerificationTests(unittest.TestCase):
    def fixture(self):
        design=Design((RingSpec('R01-L1',0,0,32.,22.,8,'out',22.5),
                       RingSpec('R02-L1',1,0,32.,-22.,8,'in',22.5)),cube_mm=6.)
        return design,{'points_mm':[[0,0,0],[2,0,0],[0,0,2]],'weights':[.5,.5,0.]}

    def test_self_contained_geometry_original_material_and_refinement(self):
        d,s=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(d,s,tmp,'missing',{'air_radius_mm':150,'symmetry_mode':'full'})
            self.assertEqual(len(job['tasks']),4)
            for task in job['tasks']:
                self.assertEqual(task['java_sha256'],hashlib.sha256(Path(task['java_file']).read_bytes()).hexdigest())
            text=Path(job['tasks'][0]['java_file']).read_text(encoding='ascii')
            self.assertIn('model.sol("sol1").runAll()',text)
            self.assertIn('model.sol("sol1").feature("s1").set("stol",',text)
            self.assertIn('set("io_Br","1.4299999999999999[T]")',text)
            self.assertIn('set("io_mur","1.05")',text)
            self.assertNotIn('loadCopy',text)
            self.assertIn('p[2][2]=2;',text)
            self.assertIn('geom1_dsv_dom',text)
            self.assertEqual(job['tasks'][-1]['air_radius_mm'],225)
            self.assertLess(job['tasks'][2]['magnet_hmax_mm'],job['tasks'][0]['magnet_hmax_mm'])

    def test_ppm_gate_can_fail_even_when_mean_passes(self):
        _,s=self.fixture();f=np.array([[0,0,.04],[0,0,.040004],[0,0,.04]])
        p=np.array([[0,0,.04],[0,0,.040020],[0,0,.04]])
        r=compare_fem_fields(p,f,s)
        self.assertLess(r['mean_relative_error'],.005)
        self.assertFalse(r['accuracy_passed'])

    def test_volume_weights_not_surface_points_define_mean(self):
        _,s=self.fixture();f=np.array([[0,0,.04],[0,0,.040004],[0,0,.2]])
        r=compare_fem_fields(f,f,s)
        self.assertAlmostEqual(r['fem_metrics']['mean_norm_T'],.040002)
        self.assertTrue(r['accuracy_passed'])

    def test_parser_checks_all_points_and_completion_and_job_id(self):
        text='MRI_FEM_ROW,a,0,0,0,.04\nMRI_FEM_ROW,a,1,0,0,.05\nMRI_FEM_COMPLETE,a\n'
        np.testing.assert_allclose(parse_fem_rows(text+'MRI_FEM_ROW,a,0,0,0,.04\n','a',2),[[0,0,.04],[0,0,.05]])
        np.testing.assert_allclose(parse_fem_rows(text+'[INFO] MRI_FEM_ROW,a,0,0,0,.04 [com.comsol.util]\n','a',2),[[0,0,.04],[0,0,.05]])
        with self.assertRaises(RuntimeError):parse_fem_rows(text,'b',2)
        with self.assertRaises(RuntimeError):parse_fem_rows(text,'a',3)
        with self.assertRaises(ValueError):parse_fem_rows(text+'MRI_FEM_ROW,a,0,0,0,.03','a',2)

    def test_three_meshes_and_larger_domain_required(self):
        with self.assertRaises(ValueError):_settings({'mesh_levels':[7,6]})
        with self.assertRaises(ValueError):_settings({'mesh_levels':[7,6,6]})
        with self.assertRaises(ValueError):_settings({'air_expansion_factor':1.})

    def test_mesh_convergence_uses_tighter_ppm_threshold(self):
        _,s=self.fixture();f=np.array([[0,0,.04],[0,0,.04],[0,0,.04]])
        g=f.copy();g[1,2]+=.0000004
        self.assertFalse(_convergence(f,g,s,_settings())['passed'])

    def test_missing_solver_never_means_fem_verified(self):
        d,s=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(d,s,tmp,'missing')
            r=run_fem_validation(job)
            self.assertEqual(r['status'],'failed');self.assertFalse(r['fem_verified'])
            self.assertFalse(r['model_accuracy_passed'])
            self.assertTrue((Path(tmp)/'fem_report.json').exists())

    def test_changed_java_is_rejected_before_launch(self):
        d,s=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(d,s,tmp,'missing')
            Path(job['tasks'][0]['java_file']).write_text('changed',encoding='ascii')
            with patch('mri_magnet.comsol_validation._execute') as execute:
                r=run_fem_validation(job)
            execute.assert_not_called();self.assertIn('changed',r['error'])

    def test_target_sphere_cannot_intersect_magnets(self):
        d,s=self.fixture();s['points_mm']=[[0,0,0],[40,0,0],[0,40,0]]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare_fem_validation(d,s,tmp,'missing')

    def test_sample_identity_includes_coordinates_and_weights_and_survives_json(self):
        _,sample=self.fixture()
        signature=sample_fingerprint(sample)
        self.assertEqual(signature,sample_fingerprint(json.loads(json.dumps(sample))))
        changed=json.loads(json.dumps(sample));changed['points_mm'][1][0]+=.001
        self.assertNotEqual(signature,sample_fingerprint(changed))
        changed=json.loads(json.dumps(sample));changed['weights']=[.6,.4,0]
        self.assertNotEqual(signature,sample_fingerprint(changed))

    def test_harmonic_comparison_is_signed_bz_not_magnitude_ppm(self):
        from mri_magnet.harmonics import sphere_quadrature
        points,weights,_=sphere_quadrature(50.,lmax=8,n_theta=10,n_phi=25)
        p2=.5*(3*(points[:,2]/50.)**2-1)
        f=np.zeros_like(points);f[:,2]=.04*(1+100e-6*p2)
        p=np.zeros_like(points);p[:,2]=.04*(1+120e-6*p2)
        sample={'points_mm':points,'weights':weights,'diameter_mm':100.,'n_surface':len(points)}
        report=compare_fem_fields(p,f,sample)
        spectrum=report['harmonic_comparison']
        self.assertTrue(spectrum['available'])
        self.assertAlmostEqual(spectrum['zonal_difference_ppm']['2'],20.,places=6)
        self.assertEqual(spectrum['fem']['field_component'],'Bz')
        sample['geometry']='disk'
        self.assertFalse(compare_fem_fields(p,f,sample)['harmonic_comparison']['available'])

    def test_sector_mapping_and_vector_rotation_match_full_analytical_array(self):
        from mri_magnet.field import FieldEngine
        from mri_magnet.sampling import sphere_samples
        design,_=self.fixture();points=sphere_samples(10.,32,64,seed=22)['points_mm']
        metadata,mapped,angles=rotational_sector(design,points)
        self.assertEqual(metadata['sector_magnet_count'],2)
        engine=FieldEngine()
        np.testing.assert_allclose(restore_sector_fields(engine.field(design,mapped),angles),
                                   engine.field(design,points),rtol=1e-10,atol=1e-13)

    def test_sector_rejects_incompatible_grids_and_cuboids_cut_by_planes(self):
        from dataclasses import replace
        design,sample=self.fixture()
        for ring in (replace(design.rings[1],phase_deg=23.),replace(design.rings[1],count=10)):
            with self.assertRaises(ValueError):rotational_sector(replace(design,rings=(design.rings[0],ring)),sample['points_mm'])
        tight=replace(design,rings=tuple(replace(r,count=36,radius_mm=20.) for r in design.rings))
        with self.assertRaises(ValueError):rotational_sector(tight,sample['points_mm'])

    def test_sector_java_has_one_intact_cuboid_per_ring_and_clipped_air(self):
        from dataclasses import replace
        d,s=self.fixture();d=replace(d,rings=tuple(replace(r,phase_deg=r.phase_deg-360) for r in d.rings))
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(d,s,tmp,'missing',{'air_radius_mm':150,'symmetry_mode':'rotational_sector'})
            source=Path(job['tasks'][0]['java_file']).read_text(encoding='ascii')
            self.assertNotIn('g.create(rotation, "Rotate")',source)
            self.assertIn('g.create("airClip","Intersection")',source)
            self.assertIn('geom1_dsvClip_dom',source)
            self.assertIn('selectedCount != 2 || uniqueDomains.size() != 2',source)
            self.assertEqual(job['symmetry']['full_magnet_count'],16)
            self.assertEqual(job['symmetry']['sector_magnet_count'],2)

    def test_cancellation_is_persisted_and_propagated(self):
        class Stop(BaseException):pass
        def stop():raise Stop('test stop')
        design,sample=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(design,sample,tmp,'missing')
            with self.assertRaises(Stop):run_fem_validation(job,cancel_check=stop)
            report=json.loads((Path(tmp)/'fem_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'cancelled')
            self.assertFalse(report['fem_verified'])

    def test_changed_sampling_identity_is_rejected_before_launch(self):
        design,sample=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(design,sample,tmp,'missing')
            job['sample']['weights']=[.9,.1,0]
            with patch('mri_magnet.comsol_validation._execute') as execute:
                report=run_fem_validation(job)
            execute.assert_not_called()
            self.assertIn('identity changed',report['error'])

    def test_changed_sector_mapping_is_rejected_before_launch(self):
        design,sample=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(design,sample,tmp,'missing',{'symmetry_mode':'rotational_sector'})
            job['rotation_angles_rad'][1]+=.01
            with patch('mri_magnet.comsol_validation._execute') as execute:
                report=run_fem_validation(job)
            execute.assert_not_called()
            self.assertIn('mapping changed',report['error'])

    def test_offset_planar_target_uses_its_own_mesh_sphere_center(self):
        from mri_magnet.sampling import disk_samples
        d,_=self.fixture();sample=disk_samples(10.,8,16,z_mm=20.)
        with tempfile.TemporaryDirectory() as tmp:
            job=prepare_fem_validation(d,sample,tmp,'missing',{'symmetry_mode':'rotational_sector'})
            source=Path(job['tasks'][0]['java_file']).read_text(encoding='ascii')
            self.assertAlmostEqual(job['target_mesh_radius_mm'],5.1)
            self.assertIn('g.feature("dsv").set("pos",new double[] {0,0,20});',source)
            self.assertIn('g.feature("dsvWedge").set("quickz",14.',source)


if __name__=='__main__':unittest.main()

"""Physical oracles and cache isolation for optimization's coupled evaluator."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import magpylib as magpy
import numpy as np

from mri_magnet.field import FieldEngine
from mri_magnet.field_models import (make_field_engine, model_fingerprint,
    evaluation_fingerprint, response_options, CoupledFieldEngine)
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.material_response import (CoupledResponseEngine, _orbit_arrays,
    _interaction_matrix, _target_quadrature, preflight_response)
from mri_magnet.metrics import field_metrics
from mri_magnet.cuboid_tensor import external_cube_tensor


def design():
    return Design((RingSpec('a',0,0,26.,13.,4,'out',11.),
                   RingSpec('b',1,0,31.,-13.,4,'in',-7.)),cube_mm=8.,reference_mur=1.05)


def config(**options):
    settings=dict(field_model='coupled',quadrature_order=2,cells_per_axis=1,prepared_cache_size=2)
    settings.update(options)
    return {'physics':settings,
            'polarization_model':'isolated_cube_self_demag','chunk_points':16,
            'target':{'geometry':'sphere','diameter_mm':10.},'density_kg_m3':7500.}


POINTS=np.array([[0.,0.,0.],[1.,2.,3.],[-3.,2.,-1.]])


class V6FieldModelTests(unittest.TestCase):
    def test_missing_physics_is_reproducible_legacy(self):
        cfg={'polarization_model':'isolated_cube_self_demag'}
        engine=make_field_engine(cfg)
        self.assertIsInstance(engine,FieldEngine)
        np.testing.assert_array_equal(engine.field(design(),POINTS),
            FieldEngine(polarization_model='isolated_cube_self_demag').field(design(),POINTS))

    def test_coupled_factory_uses_raw_remanence_not_prior_isolated_correction(self):
        cfg=config();d=design();engine=make_field_engine(cfg)
        expected=CoupledResponseEngine(d,response_options(cfg)).field(d,POINTS)
        np.testing.assert_array_equal(engine.field(d,POINTS),expected)
        self.assertIsInstance(engine,CoupledFieldEngine)

    def test_prepared_cache_reuses_field_points_but_geometry_and_material_rebuild(self):
        engine=make_field_engine(config());d=design()
        engine.field(d,POINTS);engine.field(d,POINTS[:1])
        self.assertEqual(engine.statistics['preparations'],1)
        changed=replace(d,rings=(replace(d.rings[0],radius_mm=27.),d.rings[1]))
        engine.field(changed,POINTS)
        engine.field(replace(d,reference_mur=1.04),POINTS)
        self.assertEqual(engine.statistics['preparations'],3)
        self.assertEqual(len(engine._prepared),2)
        self.assertEqual(engine.statistics['peak_prepared_entries'],2)
        engine.field(d,POINTS)
        self.assertEqual(engine.statistics['preparations'],4)

    def test_uniform_remanence_scale_preserves_ppm_in_coupled_model(self):
        d=design();scaled=replace(d,remanence_T=d.remanence_T*.91)
        original=make_field_engine(config()).field(d,POINTS)
        new=make_field_engine(config()).field(scaled,POINTS)
        np.testing.assert_allclose(new,original*.91,atol=1e-16,rtol=1e-12)
        self.assertAlmostEqual(field_metrics(original)['pp_norm_ppm'],field_metrics(new)['pp_norm_ppm'],places=7)

    def test_mu_one_returns_fixed_source_even_with_subdivision(self):
        d=replace(design(),reference_mur=1.)
        engine=make_field_engine(config(cells_per_axis=2))
        np.testing.assert_array_equal(engine.field(d,POINTS),FieldEngine().field(d,POINTS))
        self.assertEqual(engine.statistics['self_consistent_solves'],0)

    def test_subdivided_rotational_solve_matches_full_cell_system(self):
        d=design();options={'quadrature_order':2,'cells_per_axis':2}
        reduced=CoupledResponseEngine(d,options)
        full=CoupledResponseEngine(d,dict(options,symmetry_reduction=False))
        np.testing.assert_allclose(reduced.local_polarization_T,full.local_polarization_T,atol=1e-13,rtol=1e-12)
        np.testing.assert_allclose(reduced.field(d,POINTS),full.field(d,POINTS),atol=1e-14,rtol=1e-12)
        self.assertEqual(reduced.diagnostics['source_cells'],64)
        self.assertEqual(reduced.local_polarization_T.shape,(64,3))

    def test_subcells_geometry_tiles_physical_magnet_without_changing_volume(self):
        d=design();arrays=_orbit_arrays(d,4,2)
        original=d.magnet_arrays();positions=arrays['positions_m'].reshape(-1,8,3)
        np.testing.assert_allclose(positions.mean(axis=1),original['positions_mm']*1e-3,atol=1e-16)
        rotations=arrays['rotations'].reshape(-1,8,3,3)[:,0]
        relative=np.einsum('nsi,nij->nsj',positions-original['positions_mm'][:,None,:]*1e-3,rotations)
        np.testing.assert_allclose(np.abs(relative),d.cube_mm*1e-3/4,atol=1e-16)
        self.assertEqual(len(arrays['representatives']),16)

    def test_two_column_tensor_assembly_matches_three_independent_unit_sources(self):
        d=design();arrays=_orbit_arrays(d,4)
        cube=d.cube_mm*1e-3
        actual=_interaction_matrix(arrays,cube,2)
        offsets,weights=_target_quadrature(cube,2)
        positions,rotations=arrays['positions_m'],arrays['rotations']
        expected=np.zeros((len(arrays['representatives']),3,len(arrays['representatives']),3))
        for ti,physical in enumerate(arrays['representatives']):
            obs=positions[physical]+offsets@rotations[physical].T
            active=np.arange(len(positions))!=physical
            local=np.einsum('qni,nij->qnj',obs[:,None]-positions[None,active],rotations[active])
            for component in range(3):
                unit=np.eye(3)[component]
                fields=magpy.getB('Cuboid',observers=local.reshape(-1,3),
                    dimension=(cube,)*3,polarization=unit).reshape(len(offsets),-1,3)
                global_b=np.einsum('nij,qnj->qni',rotations[active],fields)
                target_local=np.einsum('q,qni->ni',weights,global_b)@rotations[physical]
                for source,vector in zip(arrays['orbit'][active],target_local):
                    expected[ti,:,source,component]+=vector
        np.testing.assert_allclose(actual,expected.reshape(actual.shape),atol=2e-16,rtol=1e-10)

    def test_subdivision_is_included_in_resource_preflight(self):
        d=design();one=preflight_response(d,{'cells_per_axis':1});eight=preflight_response(d,{'cells_per_axis':2})
        self.assertEqual(eight['vector_unknowns'],8*one['vector_unknowns'])
        self.assertEqual(eight['matrix_bytes'],64*one['matrix_bytes'])
        self.assertEqual(eight['source_cells'],8*d.magnet_count)
        self.assertFalse(preflight_response(d,{'cells_per_axis':2,'max_unknowns':12})['valid'])

    def test_imported_preparation_reuses_response_without_building_matrix(self):
        d=design();engine=make_field_engine(config());expected=engine.field(d,POINTS)
        payload=engine.export_prepared(d);other=make_field_engine(config())
        with patch('mri_magnet.material_response._interaction_matrix',side_effect=AssertionError('rebuilt')):
            other.import_prepared(d,payload)
            np.testing.assert_array_equal(other.field(d,POINTS),expected)
        self.assertEqual(other.statistics['matrix_builds'],0)
        self.assertEqual(other.statistics['imported_preparations'],1)

    def test_prepared_payload_rejects_different_model_or_design(self):
        d=design();engine=make_field_engine(config());engine.field(d,POINTS)
        payload=engine.export_prepared(d)
        with self.assertRaises(ValueError):make_field_engine(config(quadrature_order=3)).import_prepared(d,payload)
        with self.assertRaises(ValueError):make_field_engine(config()).import_prepared(replace(d,cube_mm=7.),payload)

    def test_prepared_import_respects_equivalent_ring_order_and_periodic_phases(self):
        d=replace(design(),rings=(replace(design().rings[0],count=4),replace(design().rings[1],count=8)))
        alternate=replace(d,rings=tuple(replace(r,phase_deg=r.phase_deg+360./r.count)
                                        for r in reversed(d.rings)))
        engine=make_field_engine(config());expected=engine.field(d,POINTS)
        imported=make_field_engine(config());imported.import_prepared(alternate,engine.export_prepared(d))
        np.testing.assert_allclose(imported.field(alternate,POINTS),expected,atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(imported.field(alternate,POINTS),
            make_field_engine(config()).field(alternate,POINTS),atol=1e-14,rtol=1e-12)

    def test_prepared_import_rejects_invalid_residual_metadata(self):
        d=design();engine=make_field_engine(config());engine.field(d,POINTS)
        payload=engine.export_prepared(d);payload['response']['diagnostics']['relative_recoil_residual']=float('nan')
        with self.assertRaises(FloatingPointError):make_field_engine(config()).import_prepared(d,payload)

    def test_model_and_evaluation_fingerprints_isolate_precision_points_and_weights(self):
        cfg=config();sample={'points_mm':POINTS,'weights':np.array([.2,.3,.5])}
        self.assertNotEqual(model_fingerprint(cfg),model_fingerprint(config(quadrature_order=3)))
        self.assertNotEqual(model_fingerprint(cfg),model_fingerprint(config(cells_per_axis=2)))
        baseline=evaluation_fingerprint(cfg,sample)
        for changed in ({'points_mm':POINTS[::-1],'weights':sample['weights']},
                        {'points_mm':POINTS,'weights':[.3,.2,.5]}):
            self.assertNotEqual(baseline,evaluation_fingerprint(cfg,changed))
        equivalent=deepcopy(cfg);equivalent['physics']['prepared_cache_size']=7
        self.assertEqual(model_fingerprint(cfg),model_fingerprint(equivalent))

    def test_cancel_during_preparation_propagates_and_does_not_leave_cache(self):
        class Stop(BaseException):pass
        calls=[]
        def cancel():
            calls.append(1)
            if len(calls)>=2:raise Stop()
        engine=make_field_engine(config(),cancel_check=cancel)
        with self.assertRaises(Stop):engine.field(design(),POINTS)
        self.assertEqual(len(engine._prepared),0)
        self.assertEqual(engine.statistics['preparation_failures'],1)

    def test_status_reports_assembly_and_solve_stages(self):
        statuses=[];make_field_engine(config(),status_callback=statuses.append).field(design(),POINTS)
        stages={s['stage'] for s in statuses}
        self.assertIn('coupling_matrix',stages);self.assertIn('coupling_linear_solve',stages)

    def test_failed_preparation_never_returns_previous_design_response(self):
        engine=make_field_engine(config());d=design();engine.field(d,POINTS)
        invalid=replace(d,rings=(d.rings[0],replace(d.rings[0],id='overlap')))
        with self.assertRaises(ValueError):engine.field(invalid,POINTS)
        self.assertEqual(engine.statistics['preparation_failures'],1)
        self.assertEqual(engine.statistics['preparations'],1)


class ExactCuboidTensorTests(unittest.TestCase):
    @staticmethod
    def public_tensor(points,edge):
        return np.stack([magpy.getB('Cuboid',observers=points,
            dimension=(edge,)*3,polarization=unit) for unit in np.eye(3)],axis=2)

    def test_random_external_tensor_matches_all_public_unit_polarizations(self):
        rng=np.random.default_rng(631)
        for edge in (.001,.012,1.,10.):
            points=rng.uniform(-12.,12.,(500,3))*edge
            points=points[np.any(np.abs(points)>edge*.51,axis=1)]
            actual=external_cube_tensor(points,edge)
            expected=self.public_tensor(points,edge)
            np.testing.assert_allclose(actual,expected,atol=5e-15,rtol=5e-11)

    def test_near_faces_edges_axes_and_all_octants_match_public_fields(self):
        edge=.012;half=edge/2;points=[]
        for epsilon in (1e-12,1e-9,1e-6,.01,1.):
            for signs in np.array(np.meshgrid([-1.,1.],[-1.,1.],[-1.,1.])).T.reshape(-1,3):
                points.extend([signs*np.array([half*(1+epsilon),0.,0.]),
                    signs*np.array([half*(1+epsilon),half,0.]),
                    signs*np.array([half*(1+epsilon),half,half]),
                    signs*np.array([half*(1+epsilon),half*.999,half*.999])])
        points=np.asarray(points)
        actual=external_cube_tensor(points,edge)
        safe=np.all(np.isfinite(actual),axis=(1,2))
        self.assertGreater(np.count_nonzero(safe),100)
        np.testing.assert_allclose(actual[safe],self.public_tensor(points[safe],edge),atol=5e-15,rtol=5e-10)
        # At representationally singular edge extensions the primitive must
        # request the public API fallback, never invent a finite approximation.
        self.assertTrue(np.all(np.isnan(actual[~safe])))

    def test_arbitrary_polarization_and_rotation_covariance_match_public_objects(self):
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(937)
        for _ in range(16):
            edge=.012;rotation=Rotation.random(random_state=rng)
            local=rng.uniform(-5.,5.,(50,3))*edge
            local=local[np.any(np.abs(local)>.51*edge,axis=1)]
            center=rng.uniform(-.2,.2,3);polarization=rng.normal(size=3)
            expected=magpy.magnet.Cuboid(dimension=(edge,)*3,polarization=polarization,
                position=center,orientation=rotation).getB(center+rotation.apply(local))
            actual=rotation.apply(np.einsum('nij,j->ni',external_cube_tensor(local,edge),polarization))
            np.testing.assert_allclose(actual,expected,atol=5e-15,rtol=5e-10)

    def test_surface_edge_and_interior_rows_require_fallback(self):
        edge=.012;h=edge/2
        points=np.array([[0.,0.,0.],[h,0.,0.],[h,h,0.],[h,h,h],
                         [np.nextafter(h,np.inf),0.,0.]])
        self.assertTrue(np.all(np.isnan(external_cube_tensor(points,edge))))

    def test_full_coupled_mixed_counts_and_subcells_match_debug_backend(self):
        d=replace(design(),rings=(design().rings[0],replace(design().rings[1],count=8)))
        for cells,order in ((1,1),(1,3),(2,2)):
            settings={'cells_per_axis':cells,'quadrature_order':order}
            actual=CoupledResponseEngine(d,settings)
            expected=CoupledResponseEngine(d,dict(settings,interaction_backend='public_magpylib'))
            np.testing.assert_allclose(actual.local_polarization_T,expected.local_polarization_T,atol=1e-13,rtol=1e-12)
            np.testing.assert_allclose(actual.field(d,POINTS),expected.field(d,POINTS),atol=1e-14,rtol=1e-12)

    def test_nonfinite_tensor_rows_fall_back_to_public_finite_cuboid_fields(self):
        d=design();cfg=config();cfg['physics']['interaction_backend']='public_magpylib'
        expected=make_field_engine(cfg).field(d,POINTS)
        def unsafe(points,edge):return np.full((len(points),3,3),np.nan)
        with patch('mri_magnet.cuboid_tensor.external_cube_tensor',side_effect=unsafe):
            actual=make_field_engine(config());field=actual.field(d,POINTS)
        np.testing.assert_allclose(field,expected,atol=1e-14,rtol=1e-12)
        self.assertGreater(actual.diagnostics['tensor_fallback_observer_rows'],0)

    def test_nonfinite_public_fallback_is_rejected_explicitly(self):
        def unsafe(points,edge):return np.full((len(points),3,3),np.nan)
        def invalid_field(source,observers,dimension,polarization):return np.full((len(observers),3),np.nan)
        with patch('mri_magnet.cuboid_tensor.external_cube_tensor',side_effect=unsafe), \
             patch('mri_magnet.material_response.magpy.getB',side_effect=invalid_field):
            with self.assertRaisesRegex(FloatingPointError,'public Magpylib fallback'):
                make_field_engine(config()).field(design(),POINTS)

    def test_vector_field_on_random_observers_matches_public_sampling_path(self):
        rng=np.random.default_rng(1219);d=design()
        points=rng.uniform(-60.,60.,(173,3))
        actual=make_field_engine(config());expected=make_field_engine(config(interaction_backend='public_magpylib'))
        np.testing.assert_allclose(actual.field(d,points),expected.field(d,points),atol=2e-14,rtol=2e-11)
        self.assertEqual(actual.statistics['field_points'],len(points))

    def test_source_interior_faces_edges_and_near_boundaries_use_exact_public_fallback(self):
        d=design();arrays=d.magnet_arrays();points=[]
        theta=np.deg2rad(arrays['radial_angles_deg'][0]);cs,sn=np.cos(theta),np.sin(theta)
        rotation=np.array([[cs,-sn,0.],[sn,cs,0.],[0.,0.,1.]])
        h=d.cube_mm/2
        offsets=[[0.,0.,0.],[.1,.2,.3],[h,0.,0.],[h,h,0.],[h,h,h]]
        for epsilon in (1e-12,1e-8,1e-4):offsets.extend([[h*(1+epsilon),0.,0.],[h*(1+epsilon),h,h]])
        points=arrays['positions_mm'][0]+np.asarray(offsets)@rotation.T
        actual=make_field_engine(config());expected=make_field_engine(config(interaction_backend='public_magpylib'))
        np.testing.assert_allclose(actual.field(d,points),expected.field(d,points),atol=2e-13,rtol=2e-11)
        self.assertGreater(actual.statistics['field_tensor_fallback_rows'],0)

    def test_subdivided_mixed_rotation_sampling_matches_public_all_components(self):
        rng=np.random.default_rng(1261)
        d=replace(design(),rings=(replace(design().rings[0],phase_deg=36.5),
                                  replace(design().rings[1],count=8,phase_deg=-18.)))
        points=rng.uniform(-40.,40.,(83,3))
        actual=make_field_engine(config(cells_per_axis=2,chunk_points=7))
        expected=make_field_engine(config(cells_per_axis=2,interaction_backend='public_magpylib',chunk_points=7))
        np.testing.assert_allclose(actual.field(d,points),expected.field(d,points),atol=2e-14,rtol=2e-11)

    def test_sampling_unsafe_rows_fall_back_without_rebuilding_prepared_matrix(self):
        d=design();actual=make_field_engine(config());actual.prepare(d)
        expected=make_field_engine(config(interaction_backend='public_magpylib')).field(d,POINTS)
        def unsafe(points,edge):return np.full((len(points),3,3),np.nan)
        with patch('mri_magnet.cuboid_tensor.external_cube_tensor',side_effect=unsafe):
            field=actual.field(d,POINTS)
        np.testing.assert_allclose(field,expected,atol=1e-14,rtol=1e-12)
        self.assertEqual(actual.statistics['matrix_builds'],1)
        self.assertEqual(actual.statistics['field_tensor_fallback_rows'],len(POINTS)*d.magnet_count)


if __name__=='__main__':unittest.main()

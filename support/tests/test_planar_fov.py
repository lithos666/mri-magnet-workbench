"""Planar area quadrature and independent source-free extrema oracles."""
import unittest
import numpy as np

from mri_magnet.metrics import field_metrics
from mri_magnet.sampling import disk_samples
from mri_magnet.validation import refine_disk_extrema


class DiskSamplingTests(unittest.TestCase):
    def test_samples_are_planar_with_uniform_area_weights_and_boundary_excluded(self):
        sample=disk_samples(20., 4096, 64, seed=22, z_mm=2.5)
        points, weights=sample['points_mm'],sample['weights']
        self.assertEqual(sample['geometry'],'disk')
        self.assertEqual(sample['measure'],'area')
        np.testing.assert_array_equal(points[:,2],np.full(len(points),2.5))
        self.assertEqual(np.count_nonzero(weights),4096)
        self.assertEqual(weights[0],0.)
        self.assertTrue(np.all(weights[1+4096:]==0))
        self.assertAlmostEqual(weights.sum(),1.)
        radius2=np.sum(points[:,:2]**2,axis=1)
        self.assertLessEqual(radius2.max(),100.+1e-10)
        self.assertAlmostEqual(weights@radius2,50.,delta=.02)
        self.assertAlmostEqual(weights@(points[:,0]**2),25.,delta=.04)
        self.assertAlmostEqual(weights@(points[:,1]**2),25.,delta=.04)
        self.assertEqual(sample['name'],'disk_D20_z2.5_A4096_C64_seed22')

    def test_fundamental_sector_preserves_area_radial_statistics(self):
        sample=disk_samples(20.,2048,64,17,z_mm=0,rotation_order=4,reflection_phase_deg=-90.)
        p=sample['points_mm'];angles=np.arctan2(p[1:,1],p[1:,0])
        self.assertTrue(np.all(angles>=-np.pi/2-1e-12))
        self.assertTrue(np.all(angles<=-np.pi/4+1e-12))
        self.assertAlmostEqual(sample['weights']@np.sum(p[:,:2]**2,axis=1),50.,delta=.03)
        self.assertEqual(sample['symmetry_profile']['domain_area_fraction'],1/8)
        self.assertTrue(sample['symmetry_profile']['requires_verified_design'])

    def test_invalid_planar_sampling_inputs_are_rejected(self):
        for kwargs in ({'diameter_mm':0},{'z_mm':np.nan},{'n_area':1.2},{'n_boundary':0},{'rotation_order':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):disk_samples(**kwargs)


class SourceFreePolynomial:
    """Curl-free/divergence-free cubic-potential field; planar max is interior."""
    def __init__(self,centre=(.23,-.17),z0=0.):
        self.centre=np.asarray(centre);self.z0=z0;self.points=[]
    def field(self,design,points):
        p=np.asarray(points,float).reshape(-1,3);self.points.append(p.copy())
        x,y=(p[:,:2]-self.centre).T;z=p[:,2]-self.z0
        return np.column_stack((-.6*z*x,-.6*z*y,1+.6*z*z-.3*(x*x+y*y)))*1e-3


class SourceFreeLinear:
    def __init__(self,centre=(.23,-.17),z0=0.):
        self.centre=np.asarray(centre);self.z0=z0;self.points=[]
    def field(self,design,points):
        p=np.asarray(points,float).reshape(-1,3);self.points.append(p.copy())
        x,y=(p[:,:2]-self.centre).T;z=p[:,2]-self.z0
        return np.column_stack((x,y,.2-2*z))*1e-3


class DiskExtremaTests(unittest.TestCase):
    def refine(self,engine,z=0.):
        sample=disk_samples(2.,128,96,11,z_mm=z)
        B=engine.field(None,sample['points_mm'])
        metrics=field_metrics(B,sample['weights'],B[0])
        result=refine_disk_extrema(None,engine,sample,metrics,B,starts=3)
        self.assertEqual(result['mean_norm_T'],metrics['mean_norm_T'])
        self.assertEqual(result['rms_norm_ppm'],metrics['rms_norm_ppm'])
        self.assertLessEqual(result['min_norm_T'],metrics['min_norm_T']+1e-14)
        self.assertGreaterEqual(result['max_norm_T'],metrics['max_norm_T']-1e-14)
        self.assertEqual(result['metric_sources']['mean_norm_T'],'sampled_disk_area_quadrature')
        self.assertFalse(result['extrema_refinement']['global_optimum_proven'])
        for p in engine.points:np.testing.assert_array_equal(p[:,2],np.full(len(p),z))
        self.assertTrue(any(len(p)==3 for p in engine.points))
        return result

    def test_planar_magnitude_and_bz_maximum_can_be_inside_disk(self):
        centre=np.array([.23,-.17]);engine=SourceFreePolynomial(centre,z0=2.5)
        result=self.refine(engine,z=2.5)
        self.assertAlmostEqual(result['max_norm_T'],1e-3,delta=3e-9)
        self.assertAlmostEqual(result['max_Bz_T'],1e-3,delta=3e-9)
        np.testing.assert_allclose(result['extrema_refinement']['max_point_mm'],[*centre,2.5],atol=3e-5)
        expected_min=(1-.3*(1+np.linalg.norm(centre))**2)*1e-3
        self.assertAlmostEqual(result['min_norm_T'],expected_min,delta=3e-9)
        self.assertAlmostEqual(result['max_transverse_T'],0.,delta=1e-14)

    def test_interior_offaxis_minimum_and_independent_transverse_boundary_max(self):
        centre=np.array([.23,-.17]);engine=SourceFreeLinear(centre)
        result=self.refine(engine)
        self.assertAlmostEqual(result['min_norm_T'],.2e-3,delta=3e-9)
        np.testing.assert_allclose(result['extrema_refinement']['min_point_mm'],[*centre,0.],atol=3e-5)
        expected_transverse=(1+np.linalg.norm(centre))*1e-3
        self.assertAlmostEqual(result['max_transverse_T'],expected_transverse,delta=3e-9)
        self.assertAlmostEqual(result['max_norm_T'],np.hypot(expected_transverse,.2e-3),delta=3e-9)
        self.assertAlmostEqual(result['min_Bz_T'],.2e-3,delta=1e-14)
        self.assertAlmostEqual(result['max_Bz_T'],.2e-3,delta=1e-14)

    def test_refinement_rejects_points_off_requested_plane(self):
        sample=disk_samples(2.,16,16,1)
        sample['points_mm'][1,2]=.1
        B=np.tile([0.,0.,1e-3],(len(sample['points_mm']),1))
        metrics=field_metrics(B,sample['weights'])
        with self.assertRaisesRegex(ValueError,'fixed-z'):
            refine_disk_extrema(None,SourceFreeLinear(),sample,metrics,B)


if __name__=='__main__':unittest.main()

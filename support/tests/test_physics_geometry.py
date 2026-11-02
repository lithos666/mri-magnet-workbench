import math
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from mri_magnet.geometry import Design,RingSpec,load_source,envelope_limits,validate_design
from mri_magnet.field import FieldEngine
from mri_magnet.sampling import sphere_samples
from mri_magnet.metrics import field_metrics

class PhysicsGeometryTests(unittest.TestCase):
    def setUp(self):
        self.engine=FieldEngine(chunk_points=3)
        self.design=Design((RingSpec('R01-L1',0,0,50.,10.,4,'out',13.),))

    def test_rotated_vectorized_field_against_public_object_interface(self):
        points=np.array([[0,0,0],[12,-8,4],[-10,5,25]])
        for mode in ('remanence','isolated_cube_self_demag'):
            engine=FieldEngine(chunk_points=3,polarization_model=mode)
            np.testing.assert_allclose(engine.field(self.design,points),engine.object_reference(self.design,points),atol=1e-14,rtol=1e-10)

    def test_independent_axial_solid_angle_formula(self):
        # Isolate a single cuboid by subtracting neither fits nor a second implementation.
        import magpylib as m
        J=1.43;a=b=c=.006;z=.05
        angle=lambda q:math.atan(a*b/(q*math.sqrt(q*q+a*a+b*b)))
        analytic=J/math.pi*(angle(z-c)-angle(z+c))
        actual=m.getB('Cuboid',observers=(0,0,z),dimension=(.012,.012,.012),polarization=(0,0,J))[2]
        self.assertAlmostEqual(actual,analytic,places=13)

    def test_field_linearity_and_length_unit_invariance(self):
        pts=np.array([[0,0,0],[4,7,12]])
        b=self.engine.field(self.design,pts)
        np.testing.assert_allclose(self.engine.field(replace(self.design,remanence_T=2.86),pts),b*2,rtol=1e-12,atol=1e-14)
        scaled=replace(self.design,cube_mm=24,rings=tuple(replace(r,radius_mm=r.radius_mm*2,z_mm=r.z_mm*2) for r in self.design.rings))
        np.testing.assert_allclose(self.engine.field(scaled,pts*2),b,rtol=1e-12,atol=1e-14)

    def test_duplicate_cubes_overlap_and_touching_is_allowed(self):
        limits=envelope_limits(self.design)
        self.assertTrue(validate_design(self.design,limits)['valid'])
        duplicate=replace(self.design,rings=self.design.rings*2)
        self.assertFalse(validate_design(duplicate,limits)['valid'])
        adjacent=replace(self.design,rings=self.design.rings+(replace(self.design.rings[0],id='R01-L2',radius_mm=62),))
        self.assertTrue(validate_design(adjacent,envelope_limits(adjacent))['valid'])

    def test_volume_mean_excludes_boundary_samples(self):
        B=np.array([[0,0,1],[0,0,2],[0,0,3],[0,0,10]])
        result=field_metrics(B,[0,.5,.5,0],center_B_T=B[0])
        self.assertAlmostEqual(result['mean_norm_T'],2.5)
        self.assertAlmostEqual(result['pp_norm_ppm'],9/2.5*1e6)

    def test_sphere_uniform_volume_and_boundary(self):
        s=sphere_samples(180,1024,128,seed=1)
        radii=np.linalg.norm(s['points_mm'],axis=1)
        self.assertLessEqual(radii.max(),90+1e-10)
        self.assertAlmostEqual(float(s['weights']@(radii**2)),3*90**2/5,delta=15)

    def test_invalid_weights_and_zero_field(self):
        with self.assertRaises(ValueError):field_metrics(np.zeros((3,3)))
        with self.assertRaises(ValueError):field_metrics(np.ones((3,3)),[1,-1,1])

if __name__=='__main__':unittest.main()

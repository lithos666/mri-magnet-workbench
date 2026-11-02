"""Observable contracts of relative-mean synthesis and alternate radial slots."""
import copy
import tempfile
import unittest
import numpy as np
from mri_magnet.de_novo import build_basis, _solve_basis, _layout_lattice, STYLES
from mri_magnet.sampling import sphere_samples
from mri_magnet.workbench_config import default_config


def miniature_basis(axial, transverse=None):
    axial=np.asarray(axial,float)
    npoints=len(axial[0]);a=np.zeros((npoints,3,4))
    a[:,2,0]=axial[0];a[:,2,1]=-axial[0]
    a[:,2,2]=axial[1];a[:,2,3]=-axial[1]
    if transverse is not None:
        a[:,0,0]=transverse[0];a[:,0,1]=-np.asarray(transverse[0])
        a[:,0,2]=transverse[1];a[:,0,3]=-np.asarray(transverse[1])
    weights=np.r_[0.,np.full(npoints-1,1/(npoints-1))]
    return {'A_T_per_T':a,'center_A_T_per_T':a[0],
        'mean_Az_T_per_T':weights@a[:,2,:],
        'quadrature_weights':weights,'axis_coefficients_T_per_T':np.zeros((4,4)),
        'material':{'cube_mm':12.,'remanence_T':1.,'reference_mur':1.05},
        'basis':[{'pair_index':0,'radial_index':0,'radius_mm':140.,'z_mm':40.,'polarity':p} for p in ('out','in','down','up')],
        'pair_count':1,'column_count':24,'style':'wide','radii_mm':np.array([140.]),'fingerprint':'test-field',
        'packing':{'required_radial_center_pitch_mm':12.,'minimum_magnet_center_radius_mm':127.4,'maximum_magnet_center_radius_mm':184.59}}


def miniature_config():
    cfg=default_config();cfg['target'].update(min_field_mT=1000.)
    cfg['manufacturing']['max_magnet_count']=48
    cfg['de_novo'].update(milp_time_limit_seconds=2.,sparsity_weight=0.,field_margin_ratio=0.)
    return cfg


class RelativeInverseTests(unittest.TestCase):
    def test_relative_normalization_can_choose_higher_absolute_error_but_lower_relative_error(self):
        basis=miniature_basis([[1.2,1.08,1.32],[2.4,2.25,2.55]])
        design,report=_solve_basis(basis,miniature_config())
        self.assertIsNotNone(design,report)
        self.assertEqual(design.rings[0].polarity,'down')
        self.assertAlmostEqual(report['actual_field_denominator_T'],2.4,places=8)
        self.assertAlmostEqual(report['minimax_vector_error_fraction'],.15/2.4,places=8)

    def test_mean_constraint_uses_quadrature_and_not_centre_sample(self):
        basis=miniature_basis([[2.,.9,.9],[.8,1.5,1.5]])
        design,report=_solve_basis(basis,miniature_config())
        self.assertEqual(design.rings[0].polarity,'down')
        self.assertLess(report['center_Bz_surrogate_T'],1.)
        self.assertGreaterEqual(report['mean_Bz_surrogate_T'],1.)
        self.assertAlmostEqual(report['actual_field_denominator_T'],1.5,places=8)

    def test_lower_transverse_weight_prioritizes_axial_homogeneity_without_changing_exact_field(self):
        basis=miniature_basis([[1.2,1.2,1.2],[1.2,1.16,1.24]],[[.2,.2,.2],[0,0,0]])
        cfg=miniature_config();cfg['de_novo']['transverse_weight']=.05
        design,_=_solve_basis(basis,cfg)
        self.assertEqual(design.rings[0].polarity,'out')
        cfg['de_novo']['transverse_weight']=1.
        other,_=_solve_basis(basis,cfg)
        self.assertEqual(other.rings[0].polarity,'down')

    def test_zero_mean_headroom_or_invalid_weight_is_not_silently_relaxed(self):
        cfg=miniature_config();cfg['de_novo']['transverse_weight']=-.1
        with self.assertRaises(ValueError):_solve_basis(miniature_basis([[1.2]*3,[1.4]*3]),cfg)
        cfg=miniature_config();cfg['de_novo']['field_margin_ratio']=.1
        design,report=_solve_basis(miniature_basis([[1.01]*3,[1.02]*3]),cfg)
        self.assertIsNone(design)


class RadialBasisTests(unittest.TestCase):
    def test_all_five_styles_have_different_radial_basis(self):
        cfg=default_config();radii=[_layout_lattice(cfg,5,24,s)[3] for s in STYLES]
        self.assertEqual(len({tuple(np.round(r,8)) for r in radii}),5)
        for values in radii:self.assertTrue(np.all(np.diff(values)>=12.-1e-8))

    def test_six_candidates_are_compatible_alternatives_not_six_simultaneous_rings(self):
        cfg=default_config();cfg['de_novo'].update(radial_candidate_count=6,milp_time_limit_seconds=3.)
        cfg['target']['min_field_mT']=15.
        sample=sphere_samples(20.,8,12)
        basis=build_basis(cfg,sample,5,24)
        self.assertEqual(len(basis['radii_mm']),6)
        self.assertEqual(basis['packing']['maximum_radial_rings'],5)
        design,report=_solve_basis(basis,cfg)
        self.assertIsNotNone(design,report)
        self.assertGreater(report['radial_conflict_constraints'],0)
        for layer in {r.layer_index for r in design.rings}:
            radii=sorted(r.radius_mm for r in design.rings if r.layer_index==layer)
            self.assertLessEqual(len(radii),5)
            self.assertTrue(np.all(np.diff(radii)>=12.-1e-6))

    def test_quadrature_weights_are_normalized_and_part_of_basis_fingerprint(self):
        cfg=default_config();sample=sphere_samples(20.,8,12)
        with tempfile.TemporaryDirectory() as folder:
            first=build_basis(cfg,sample,5,24,cache_dir=folder)
            changed=copy.deepcopy(sample);changed['weights'][1:9]=np.arange(1,9,dtype=float)
            other=build_basis(cfg,changed,5,24,cache_dir=folder)
            self.assertNotEqual(first['fingerprint'],other['fingerprint'])
            self.assertAlmostEqual(float(other['quadrature_weights'].sum()),1.)
            self.assertTrue(np.all(other['quadrature_weights'][np.asarray(sample['weights'])==0]==0))
            np.testing.assert_allclose(other['mean_Az_T_per_T'],other['quadrature_weights']@other['A_T_per_T'][:,2,:])


if __name__=='__main__':unittest.main()

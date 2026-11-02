"""Split seams relax sector identity, never physical magnet/slot constraints."""
from dataclasses import replace
import copy
import unittest
import numpy as np

from mri_magnet.geometry import Design,RingSpec
from mri_magnet.manufacturing import audit_manufacturing,column_partition_check,_configuration
from mri_magnet.de_novo import build_basis,synthesize
from mri_magnet.sampling import sphere_samples
from mri_magnet.topology import propose_topologies
from mri_magnet.workflow import _prepare_seed


PROCESS={'enabled':True,'layout_mode':'radial_channels','allow_split_pocket_seams':True,
    'pocket_clearance_per_side_mm':.2,'pocket_web_mm':1.6,'inner_wall_mm':2.4,'outer_wall_mm':2.4,
    'base_thickness_mm':5.,'sectors_per_layer':4,'printer_bed_mm':[480.,480.,255.],
    'machine_bed_mm':[255.,255.,255.],'bed_margin_per_side_mm':5.,'max_magnet_count':4608,
    'min_axial_module_gap_mm':2.,'min_central_magnet_gap_mm':10.,'radial_inner_clearance_mm':10.}
LIMITS={'bore_radius_mm':100.,'outer_radius_mm':220.,'half_length_mm':250.,
    'min_gap_mm':0.,'tolerance_mm':1e-6,'preserve_exact_bore':False}


def design(count=26,layers=16):
    rings=[]
    for pair in range(layers//2):
        z=20.+30*(layers//2-1-pair)
        for positive,layer in ((True,pair),(False,layers-1-pair)):
            rings.append(RingSpec('R%02d-L1'%(layer+1),layer,0,150.,z if positive else -z,
                count,'out' if positive else 'in',-90+180/count))
    return Design(tuple(sorted(rings,key=lambda r:r.layer_index)))


class SplitSeamManufacturingTests(unittest.TestCase):
    def test_even_non_four_multiple_counts_pass_real_16_layer_audit(self):
        for count in (26,30,34):
            with self.subTest(count=count):
                report=audit_manufacturing(design(count),PROCESS,LIMITS)
                self.assertTrue(report['valid'],report['violations'])
                self.assertTrue(report['checks']['nominal_magnet_geometry'])
                self.assertTrue(report['checks']['pocket_web_obb_clearance'])
                self.assertTrue(report['checks']['sector_fits_printer_bed'])
                self.assertTrue(report['checks']['split_seam_even_columns'])
                self.assertFalse(report['column_partition'][0]['divisible'])
                partition=report['sector_partition']
                self.assertFalse(partition['all_layer_patterns_rotationally_identical'])
                self.assertEqual(partition['split_pocket_count'],32)
                self.assertEqual({row['seam_index'] for row in partition['pockets_split_by_seams']},{0,2})

    def test_disabled_or_missing_flag_retains_identical_sector_check(self):
        for process in (dict(PROCESS,allow_split_pocket_seams=False),{k:v for k,v in PROCESS.items() if k!='allow_split_pocket_seams'}):
            result=audit_manufacturing(design(),process,LIMITS)
            self.assertFalse(result['valid'])
            self.assertFalse(result['checks']['columns_divisible_by_sectors'])
            self.assertFalse(result['column_partition'][0]['valid'])

    def test_odd_and_noninteger_counts_are_not_silently_admitted(self):
        for value in (25,27,31,3,26.5,True,float('nan')):
            self.assertFalse(column_partition_check(value,PROCESS)['valid'])
        self.assertTrue(column_partition_check(27,dict(PROCESS,allow_split_pocket_seams=False,sectors_per_layer=3))['valid'])
        self.assertTrue(column_partition_check(27,dict(PROCESS,enabled=False))['valid'])
        with self.assertRaises(ValueError):_configuration(dict(PROCESS,allow_split_pocket_seams='true'))

    def test_overlap_stock_and_real_bed_stay_enforced(self):
        d=design()
        for process,check in ((dict(PROCESS,max_magnet_count=400),'magnet_budget'),
                              (dict(PROCESS,machine_bed_mm=[100.,100.,255.]),'sector_fits_printer_bed')):
            result=audit_manufacturing(d,process,LIMITS)
            self.assertFalse(result['valid']);self.assertFalse(result['checks'][check])
            self.assertTrue(result['checks']['split_seam_even_columns'])
        overlap=replace(d,rings=d.rings+(replace(d.rings[0],id='overlap',slot_index=1,radius_mm=151.),))
        result=audit_manufacturing(overlap,PROCESS,LIMITS)
        self.assertFalse(result['valid']);self.assertFalse(result['checks']['nominal_magnet_geometry'])
        self.assertFalse(result['checks']['radial_magnet_separation'])

    def test_phase_changes_update_actual_seam_intersections(self):
        aligned=design(28,layers=2)
        normal=audit_manufacturing(aligned,PROCESS,LIMITS)
        self.assertEqual(normal['sector_partition']['split_pocket_count'],0)
        shifted=replace(aligned,rings=tuple(replace(r,phase_deg=0.) for r in aligned.rings))
        report=audit_manufacturing(shifted,PROCESS,LIMITS)
        self.assertTrue(report['valid'],report['violations'])
        self.assertEqual(report['sector_partition']['split_pocket_count'],8)


class SplitSeamSynthesisTests(unittest.TestCase):
    def config(self):
        return {'mode':'de_novo','source_geometry':'__MUST_NOT_BE_READ__.json',
            'target':{'geometry':'sphere','diameter_mm':20.,'min_field_mT':1.,'max_ppm':None},
            'envelope':{'protected_bore_diameter_mm':200.,'max_outer_diameter_mm':440.,'max_length_mm':500.},
            'magnet':{'cube_mm':12.,'remanence_T':1.43,'reference_mur':1.05},
            'manufacturing':copy.deepcopy(PROCESS),'search':{'max_radial_rings_per_layer':2},
            'de_novo':{'pair_counts':[5],'column_counts':[26],'seed_count':1,
                'radial_candidate_count':2,'max_attempts':1,'milp_time_limit_seconds':2.},
            'polarization_model':'remanence','chunk_points':48}

    def test_real_source_free_26_column_basis_and_integer_seed(self):
        cfg=self.config();sample=sphere_samples(20.,8,12)
        basis=build_basis(cfg,sample,5,26)
        self.assertEqual(basis['column_count'],26)
        self.assertTrue(np.all(np.isfinite(basis['A_T_per_T'])))
        result=synthesize(cfg,sample,log=lambda *args:None)
        self.assertEqual(len(result['seeds']),1,result['metadata']['attempts'])
        seed=result['seeds'][0]
        self.assertEqual({r['count'] for r in seed['design']['rings']},{26})
        self.assertTrue(seed['audit']['valid'])
        self.assertFalse(result['metadata']['source_layout_read'])
        self.assertGreaterEqual(seed['metrics']['mean_norm_T'],.001)
        cfg['manufacturing']['allow_split_pocket_seams']=False
        with self.assertRaises(ValueError):build_basis(cfg,sample,5,26)

    def test_topology_and_seed_projection_keep_26_30_34_without_rounding(self):
        source=design(24,layers=2)
        search={'min_radial_rings_per_layer':1,'max_radial_rings_per_layer':1,
            'azimuth_count_candidates':[24,26,30,34],'azimuth_count_mode':'shared',
            'optimize_radial_ring_count':False,'optimize_azimuth_count':True,
            'optimize_magnetization':False,'stock_aware_column_neighbors':True,'seed_subset_max_candidates':8}
        proposals=propose_topologies(source,source,PROCESS,LIMITS,search,np.random.default_rng(10),4)
        self.assertEqual({d.rings[0].count for d in proposals},{24,26,30,34})
        self.assertTrue(all(audit_manufacturing(d,PROCESS,LIMITS)['valid'] for d in proposals))
        defaults=dict(search);defaults.pop('azimuth_count_candidates')
        neighbours=propose_topologies(source,source,PROCESS,LIMITS,defaults,np.random.default_rng(10),2)
        self.assertEqual(neighbours[1].rings[0].count,26)
        chosen,report=_prepare_seed(source,PROCESS,LIMITS,dict(search,azimuth_count_candidates=[26]))
        self.assertTrue(report['valid']);self.assertEqual({r.count for r in chosen.rings},{26})
        with self.assertRaises(ValueError):
            propose_topologies(source,source,dict(PROCESS,allow_split_pocket_seams=False),LIMITS,search,np.random.default_rng(10),4)


if __name__=='__main__':unittest.main()

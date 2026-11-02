"""Direction grouping preserves discrete cuboid polarizations and source traceability."""
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np

from mri_magnet.comsol_export import generate_java_model
from mri_magnet.geometry import Design, RingSpec


class DirectionGroupedExportTests(unittest.TestCase):
    def fixture(self):
        # Same radial polarity deliberately spans three different angular grids.
        return Design((
            RingSpec('out28_a',0,0,128.,247.,28,'out',276.428571),
            RingSpec('out32',1,0,154.,208.,32,'out',275.625),
            RingSpec('out24',2,0,173.,169.,24,'out',277.5),
            RingSpec('out28_b',3,0,160.,130.,28,'out',276.428571),
            RingSpec('in32',4,0,136.,91.,32,'in',275.625),
            RingSpec('down24',5,0,140.,52.,24,'down',277.5),
            RingSpec('up28',6,0,148.,13.,28,'up',276.428571),
        ),name='direction grouping fixture')

    def test_each_source_ring_maps_to_one_of_four_direction_groups(self):
        design=self.fixture();_,source,manifest=generate_java_model(design)
        groups=manifest['direction_groups'];by_polarity={g['polarity']:g for g in groups}
        self.assertEqual(set(by_polarity),{'out','in','down','up'})
        self.assertEqual(len(groups),4)
        self.assertEqual(sum(g['magnet_count'] for g in groups),design.magnet_count)
        all_indices=[]
        for polarity,group in by_polarity.items():
            indices=[i for i,r in enumerate(design.rings) if r.polarity==polarity]
            self.assertEqual(group['ring_indices'],indices)
            self.assertEqual(group['ring_ids'],[design.rings[i].id for i in indices])
            self.assertEqual(group['magnet_count'],sum(design.rings[i].count for i in indices))
            self.assertEqual(group['domain_selection'],'mag_'+polarity)
            self.assertEqual(group['physics_feature'],'pm_'+polarity if len(group['azimuth_grids'])==1 else None)
            self.assertEqual(group['physics_features'],[g['physics_feature'] for g in group['azimuth_grids']])
            all_indices.extend(indices)
        self.assertEqual(sorted(all_indices),list(range(len(design.rings))))
        for i,ring in enumerate(manifest['rings']):
            self.assertEqual(ring['domain_selection'],'mag_'+design.rings[i].polarity)
            self.assertEqual(ring['physics_feature'],manifest['grid_groups'][ring['grid_index']]['physics_feature'])
            self.assertEqual(ring['geometry_domain_selections'],['geom1_b%d_dom'%i,'geom1_r%d_dom'%i])
        self.assertNotIn('addMagnetPhysics(',source)

    def test_mixed_azimuth_grids_are_retained_inside_one_radial_group(self):
        design=self.fixture();_,source,manifest=generate_java_model(design)
        group=next(g for g in manifest['direction_groups'] if g['polarity']=='out')
        grids=group['azimuth_grids']
        self.assertEqual(len(grids),3)
        members=[i for grid in grids for i in grid['ring_indices']]
        self.assertEqual(sorted(members),[0,1,2,3])
        self.assertTrue(any(set(grid['ring_indices'])=={0,3} for grid in grids))
        for grid in grids:
            for i in grid['ring_indices']:
                ring=design.rings[i]
                self.assertEqual(grid['count'],ring.count)
                # Equivalent phase representatives are allowed.
                period=360/ring.count
                self.assertAlmostEqual((grid['phase_deg']-ring.phase_deg+period/2)%period-period/2,0.,places=9)
        self.assertNotIn('dom==',source)
        self.assertIn('variable(variables).selection().named(selection)',source)
        self.assertEqual(manifest['magnetic_physics_feature_count'],6)
        self.assertIn('round(',source)
        self.assertIn('atan2(y,x)',source)
        self.assertNotIn('x/sqrt(x*x+y*y)',source)

    def test_angular_quantization_stays_constant_at_each_cuboid_corner(self):
        design=self.fixture();_,_,manifest=generate_java_model(design)
        radial=[g for g in manifest['direction_groups'] if g['polarity'] in ('out','in')]
        for group in radial:
            for grid in group['azimuth_grids']:
                phase=np.deg2rad(grid['phase_deg']);step=2*np.pi/grid['count']
                for i in grid['ring_indices']:
                    ring=design.rings[i]
                    for column in range(ring.count):
                        theta=np.deg2rad(ring.phase_deg+column*360/ring.count)
                        radial_axis=np.array([np.cos(theta),np.sin(theta)])
                        tangential_axis=np.array([-np.sin(theta),np.cos(theta)])
                        center=ring.radius_mm*radial_axis
                        for dr,dt in ((-6.,-6.),(-6.,6.),(6.,-6.),(6.,6.),(0.,0.)):
                            xy=center+dr*radial_axis+dt*tangential_axis
                            quantized=phase+step*np.round((np.arctan2(xy[1],xy[0])-phase)/step)
                            expected=radial_axis*(1 if group['polarity']=='out' else -1)
                            actual=np.array([np.cos(quantized),np.sin(quantized)])*(1 if group['polarity']=='out' else -1)
                            np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=0)

    def test_unused_directions_do_not_create_empty_component_groups(self):
        design=self.fixture();design=replace(design,rings=design.rings[:4])
        _,_,manifest=generate_java_model(design)
        self.assertEqual([g['polarity'] for g in manifest['direction_groups']],['out'])
        self.assertEqual({r['domain_selection'] for r in manifest['rings']},{'mag_out'})
        self.assertEqual({r['physics_feature'] for r in manifest['rings']},{'pm_out_g01','pm_out_g02','pm_out_g03'})

    def test_single_grid_keeps_legacy_selection_and_physics_tags(self):
        design=self.fixture();design=replace(design,rings=(design.rings[0],design.rings[3]))
        _,source,manifest=generate_java_model(design)
        self.assertEqual(len(manifest['grid_groups']),1)
        self.assertEqual(manifest['grid_groups'][0]['domain_selection'],'mag_out')
        self.assertEqual(manifest['grid_groups'][0]['physics_feature'],'pm_out')
        self.assertEqual({r['physics_feature'] for r in manifest['rings']},{'pm_out'})
        self.assertNotIn('dom==',source)

    def test_equivalent_periodic_phases_share_grid_without_erasing_distinct_grids(self):
        ring=self.fixture().rings[0]
        design=Design((ring,replace(ring,id='period_alias',layer_index=1,z_mm=200.,phase_deg=ring.phase_deg+360/ring.count)))
        _,_,manifest=generate_java_model(design)
        self.assertEqual(len(manifest['grid_groups']),1)
        self.assertEqual(manifest['grid_groups'][0]['ring_indices'],[0,1])
        changed=replace(design,rings=(ring,replace(design.rings[1],phase_deg=ring.phase_deg+1.)))
        _,_,other=generate_java_model(changed)
        self.assertEqual(len(other['grid_groups']),2)

    def test_six_decimal_phase_grouping_preserves_precise_representative(self):
        ring=replace(self.fixture().rings[0],phase_deg=7.12345611)
        other=replace(ring,id='same_phase_bin',layer_index=1,z_mm=200.,phase_deg=7.12345612)
        _,_,manifest=generate_java_model(Design((ring,other)))
        self.assertEqual(len(manifest['grid_groups']),1)
        self.assertEqual(manifest['grid_groups'][0]['ring_indices'],[0,1])
        self.assertAlmostEqual(manifest['grid_groups'][0]['canonical_phase_deg'],7.12345611,places=9)

    def test_new_export_schema_changes_model_identity_not_physical_identity(self):
        design=self.fixture();name,_,manifest=generate_java_model(design)
        old_hash=hashlib.sha256(json.dumps({'physical_sha256':manifest['design_sha256'],
            'physics':manifest['physics_settings'],'export_schema':'self-contained-v3-direction-groups'},
            sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
        self.assertNotEqual(manifest['model_sha256'],old_hash)
        renamed_name,_,renamed=generate_java_model(replace(design,name='display-only rename'))
        self.assertEqual(renamed_name,name)
        self.assertEqual(renamed['design_sha256'],manifest['design_sha256'])
        self.assertEqual(renamed['model_sha256'],manifest['model_sha256'])


if __name__=='__main__':unittest.main()

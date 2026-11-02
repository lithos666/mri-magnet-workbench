"""Engineering-case packing contracts, independent of magnetic objectives."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from mri_magnet.de_novo import _layout_lattice, _pair_design, limits_from_config
from mri_magnet.geometry import Design
from mri_magnet.manufacturing import (audit_manufacturing, axial_packing_bounds,
    make_manufacturing_seed, radial_packing_bounds)
from mri_magnet.parameterization import get_search_space
from mri_magnet.workbench_config import default_config
from mri_magnet.topology import propose_topologies
from mri_magnet.workflow import _prepare_seed


def engineering_config():
    cfg = default_config()
    cfg['envelope'].update(protected_bore_diameter_mm=200., max_outer_diameter_mm=440., max_length_mm=500.)
    cfg['manufacturing'].update(layout_mode='radial_channels', pocket_clearance_per_side_mm=.2,
        pocket_web_mm=1.6, inner_wall_mm=2.4, outer_wall_mm=2.4, base_thickness_mm=5.,
        radial_inner_clearance_mm=10., min_central_magnet_gap_mm=10.,
        min_axial_module_gap_mm=2., printer_bed_mm=[480.,480.,255.],
        machine_bed_mm=[255.,255.,255.], sectors_per_layer=4,
        bed_margin_per_side_mm=5., max_magnet_count=4608)
    cfg['search']['max_radial_rings_per_layer'] = None
    return cfg


def geometry(cfg, pairs=8, all_radii=False):
    material, limits, packing, radii, axial = _layout_lattice(cfg, pairs, 24, 'wide')
    rings = []
    for k,z in enumerate(axial):
        for j,radius in enumerate(radii if all_radii else radii[:1]):
            rings.extend(_pair_design(material, pairs, 24, {'pair_index': k,
                'radial_index': j, 'radius_mm': radius, 'z_mm': z, 'polarity':'out'}, unit=False).rings)
    return Design(tuple(rings), **material)


class EngineeringPackingTests(unittest.TestCase):
    def test_eight_rings_come_from_real_face_and_corner_bounds(self):
        cfg=engineering_config(); limits=limits_from_config(cfg)
        packing=radial_packing_bounds(12.,cfg['manufacturing'],limits)
        self.assertEqual(packing['maximum_radial_rings'],8)
        self.assertAlmostEqual(packing['minimum_magnet_center_radius_mm'],116.)
        self.assertAlmostEqual(packing['minimum_magnet_inner_face_radius_mm'],110.)
        self.assertAlmostEqual(packing['required_radial_center_pitch_mm'],12.)
        design=geometry(cfg,all_radii=True)
        audit=audit_manufacturing(design,cfg['manufacturing'],limits)
        self.assertTrue(audit['valid'],audit['violations'])
        self.assertEqual(len(design.rings),16*8)
        self.assertTrue(audit['bed_fit']['assembly_fits'])
        self.assertTrue(audit['bed_fit']['sector_fits_machine_bed'])

    def test_explicit_cap_limits_lattice_without_changing_physical_capacity(self):
        cfg=engineering_config()
        self.assertEqual(len(_layout_lattice(cfg,8,24,'wide')[3]),8)
        cfg['search']['max_radial_rings_per_layer']=3
        _,_,packing,radii,_=_layout_lattice(cfg,8,24,'wide')
        self.assertEqual(packing['maximum_radial_rings'],8)
        self.assertEqual(len(radii),3)

    def test_central_gap_also_respects_two_baseplates(self):
        cfg=engineering_config(); bounds=axial_packing_bounds(12.,cfg['manufacturing'],limits_from_config(cfg))
        self.assertAlmostEqual(bounds['minimum_abs_magnet_center_z_mm'],12.2)
        self.assertAlmostEqual(bounds['minimum_central_magnet_face_gap_mm'],12.4)
        design=geometry(cfg)
        limits=limits_from_config(cfg)
        variables,build=get_search_space(design,cfg['manufacturing'],limits,cfg['search'])
        for vector in (np.zeros(len(variables)),np.ones(len(variables))):
            audit=audit_manufacturing(build(vector),cfg['manufacturing'],limits)
            self.assertTrue(audit['valid'],audit['violations'])
            self.assertGreaterEqual(audit['central_magnet_face_gap_mm'],10.-1e-6)

    def test_seed_projection_respects_large_central_gap(self):
        cfg=engineering_config(); original=geometry(cfg)
        cfg['manufacturing']['min_central_magnet_gap_mm']=60.
        projected,report=make_manufacturing_seed(original,cfg['manufacturing'],limits_from_config(cfg))
        self.assertTrue(report['valid'],report['violations'])
        self.assertGreaterEqual(min(abs(r.z_mm) for r in projected.rings),36.-1e-6)
        mirrors={(r.layer_index,r.slot_index):r for r in projected.rings}
        for r in projected.rings:
            self.assertAlmostEqual(r.z_mm,-mirrors[(15-r.layer_index,r.slot_index)].z_mm)

    def test_assembled_envelope_does_not_hide_small_physical_bed(self):
        cfg=engineering_config(); cfg['manufacturing']['machine_bed_mm']=[190.,190.,255.]
        audit=audit_manufacturing(geometry(cfg),cfg['manufacturing'],limits_from_config(cfg))
        self.assertTrue(audit['bed_fit']['assembly_fits'])
        self.assertFalse(audit['bed_fit']['sector_fits_machine_bed'])
        self.assertFalse(audit['valid'])

    def test_magnet_face_clearance_is_independent_of_inner_wall(self):
        cfg=engineering_config(); design=geometry(cfg)
        bad=replace(design,rings=tuple(replace(r,radius_mm=115.9) for r in design.rings))
        audit=audit_manufacturing(bad,cfg['manufacturing'],limits_from_config(cfg))
        self.assertFalse(audit['checks']['magnet_inner_face_clearance'])
        self.assertFalse(audit['valid'])

    def test_seeded_sixteen_layers_and_topology_search_accept_dynamic_capacity(self):
        cfg=engineering_config(); source=geometry(cfg,all_radii=True)
        cfg['search'].update(optimize_radial_ring_count=True,optimize_azimuth_count=False,
            seed_subset_max_candidates=1,min_radial_rings_per_layer=1)
        limits=limits_from_config(cfg)
        seed,report=_prepare_seed(source,cfg['manufacturing'],limits,cfg['search'])
        self.assertTrue(report['valid'],report['violations'])
        self.assertEqual(len(seed.rings),128)
        proposals=propose_topologies(source,seed,cfg['manufacturing'],limits,cfg['search'],
            np.random.default_rng(291),3)
        self.assertGreaterEqual(len(proposals),2)
        for design in proposals:
            self.assertTrue(audit_manufacturing(design,cfg['manufacturing'],limits)['valid'])
            counts=[sum(r.layer_index==layer for r in design.rings) for layer in range(16)]
            self.assertTrue(all(1<=count<=8 for count in counts))


if __name__=='__main__': unittest.main()

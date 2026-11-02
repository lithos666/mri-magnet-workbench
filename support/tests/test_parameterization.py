"""Physical reachability, ordered feasibility and unbiased topology regressions."""
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import unittest
import numpy as np

from parameters import CONFIG
from mri_magnet.geometry import envelope_limits, load_source
from mri_magnet.manufacturing import audit_manufacturing, make_manufacturing_seed
from mri_magnet.parameterization import (get_search_space, encode_slacks, decode_slacks,
    radius_slacks, topology_identity)
from mri_magnet.topology import propose_topologies, sample_ring_counts


class CentralizedParametersTests(unittest.TestCase):
    def test_config_is_json_serializable_and_keeps_numerical_settings_together(self):
        restored=json.loads(json.dumps(CONFIG))
        self.assertEqual(restored["fov_diameter_mm"],20)
        self.assertIn("quick_validation_boundary",restored["sampling"])
        self.assertIn("axial_shift_limit_mm",restored["search"])


class SlackCoordinatesTests(unittest.TestCase):
    def test_boundary_slacks_round_trip_without_softmax_exclusion(self):
        for gaps in ([0,2,0,4],[6,0,0,0],[0,0,0,6],[0,0,0,0]):
            actual=decode_slacks(encode_slacks(gaps),sum(gaps))
            np.testing.assert_allclose(actual,gaps,atol=1e-14,rtol=0)

    def test_invalid_or_negative_slacks_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):encode_slacks([1,-1,2])
        with self.assertRaises(ValueError):decode_slacks([1.01],2)

    def test_inclusive_integer_random_part_does_not_half_weight_endpoints(self):
        counts=sample_ring_counts(np.random.default_rng(184),2,5,60000)
        histogram=np.bincount(counts,minlength=6)[2:6]
        self.assertTrue(np.all(np.abs(histogram-15000)<500),histogram)


class PrintableParameterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        source_path=Path(os.environ.get("MRI_SOURCE",str(root/"examples"/"seeded"/"placement.json")))
        if not source_path.is_file():raise unittest.SkipTest("Set MRI_SOURCE to the actual source geometry")
        cls.source,_=load_source(source_path)
        cls.envelope=dict(envelope_limits(cls.source),preserve_exact_bore=False)
        cls.process=copy.deepcopy(CONFIG["manufacturing"])
        cls.search=copy.deepcopy(CONFIG["search"])
        cls.seed,cls.seed_report=make_manufacturing_seed(cls.source,cls.process,cls.envelope)

    def space(self,seed=None,**updates):
        return get_search_space(seed or self.seed,self.process,self.envelope,dict(self.search,**updates))

    def assert_printable(self,design):
        audit=audit_manufacturing(design,self.process,self.envelope)
        self.assertTrue(audit["valid"],audit["violations"])

    def test_fixed_topology_has_independent_continuous_radii_and_literal_warm_start(self):
        variables,builder=self.space()
        initial=np.array([v["initial"] for v in variables])
        self.assertEqual(builder(initial).rings,self.seed.rings)
        self.assertEqual(len(variables),36)
        self.assertTrue(all(not v["integer"] for v in variables))
        self.assertFalse(any("subset" in v["name"] or "count_choice" in v["name"] for v in variables))
        self.assert_printable(builder(initial))

    def test_nonuniform_radii_with_both_ends_retreating_are_representable(self):
        _,builder=self.space()
        radii=[130.0,148.0,178.0]
        maps={r.slot_index:value for r,value in zip(sorted((r for r in self.seed.rings if r.layer_index==4),key=lambda r:r.radius_mm),radii)}
        target=replace(self.seed,rings=tuple(replace(r,radius_mm=maps[r.slot_index]) if r.layer_index in (4,9) else r for r in self.seed.rings))
        actual=builder(builder.encode_design(target))
        np.testing.assert_allclose([r.radius_mm for r in actual.rings if r.layer_index==4],radii,atol=1e-11,rtol=0)
        self.assert_printable(actual)

    def test_full_box_corners_and_dirichlet_population_preserve_geometry(self):
        variables,builder=self.space()
        samples=builder.initial_sampler(np.random.default_rng(91),12,include_initial=True)
        for vector in [np.zeros(len(variables)),np.ones(len(variables)),np.arange(len(variables))%2,*samples]:
            self.assert_printable(builder(vector))

    def test_radial_initialisation_is_uniform_slack_not_uniform_stick_weights(self):
        variables,builder=self.space()
        pop=builder.initial_sampler(np.random.default_rng(162),1200)
        indices=[i for i,v in enumerate(variables) if v["name"].startswith("radial_slack_pair_5_")]
        packing=builder.metadata['radial_packing']
        total=packing['maximum_magnet_center_radius_mm']-packing['minimum_magnet_center_radius_mm']-2*packing['required_radial_center_pitch_mm']
        gaps=np.array([decode_slacks(row[indices],total) for row in pop])
        np.testing.assert_allclose(gaps.mean(axis=0),np.full(4,total/4),atol=total*.03,rtol=0)

    def test_neighbour_layers_can_cooperatively_shift_twenty_mm(self):
        _,builder=self.space(axial_shift_limit_mm=25)
        for sign in (-1,1):
            rings=[]
            for ring in self.seed.rings:
                if ring.layer_index in (2,3):ring=replace(ring,z_mm=ring.z_mm+sign*20)
                elif ring.layer_index in (10,11):ring=replace(ring,z_mm=ring.z_mm-sign*20)
                rings.append(ring)
            target=replace(self.seed,rings=tuple(rings))
            actual=builder(builder.encode_design(target))
            np.testing.assert_allclose([r.z_mm for r in actual.rings],[r.z_mm for r in target.rings],atol=1e-10,rtol=0)
            self.assert_printable(actual)

    def test_out_of_window_or_colliding_positions_cannot_be_encoded(self):
        _,builder=self.space(axial_shift_limit_mm=25)
        target=replace(self.seed,rings=tuple(replace(r,z_mm=r.z_mm+30) if r.layer_index==2 else r for r in self.seed.rings))
        with self.assertRaises(ValueError):builder.encode_design(target)

    def test_same_physical_topology_can_use_other_provenance_labels(self):
        _,builder=self.space()
        relabelled=replace(self.seed,rings=tuple(replace(r,id="old_"+r.id,slot_index=r.slot_index+100) for r in self.seed.rings),source_sha256="historical")
        self.assertEqual(topology_identity(self.seed),topology_identity(relabelled))
        actual=builder(builder.encode_design(relabelled))
        np.testing.assert_allclose([r.radius_mm for r in actual.rings],[r.radius_mm for r in self.seed.rings],atol=1e-10,rtol=0)

    def test_switches_can_freeze_all_continuous_geometry(self):
        variables,builder=self.space(optimize_radius=False,optimize_axial_positions=False)
        self.assertEqual(variables,[])
        self.assertEqual(builder([]).rings,self.seed.rings)

    def test_zero_radial_slack_drops_inactive_dimensions(self):
        pack=self.space()[1].metadata["radial_packing"]
        rmin=pack["minimum_magnet_center_radius_mm"];pitch=pack["required_radial_center_pitch_mm"]
        half=pack["pocket_edge_mm"]/2
        outer=self.process["outer_wall_mm"]+math.hypot(rmin+pitch+half,half)
        env=dict(self.envelope,outer_radius_mm=outer)
        rings=[]
        for layer in range(14):
            group=sorted((r for r in self.seed.rings if r.layer_index==layer),key=lambda r:r.radius_mm)[:2]
            rings.extend(replace(r,radius_mm=rmin+index*pitch) for index,r in enumerate(group))
        compact=replace(self.seed,rings=tuple(rings))
        variables,builder=get_search_space(compact,self.process,env,self.search)
        self.assertFalse(any(v["name"].startswith("radial") for v in variables))
        self.assertEqual(len(variables),7)

    def test_seed_and_restored_source_slot_enter_outer_proposals(self):
        # This test exercises the random prior; deterministic stock-aware
        # coverage has separate tests and must not consume this two-slot quota.
        search=dict(self.search,stock_aware_column_neighbors=False,optimize_radial_ring_count=True,optimize_azimuth_count=True)
        proposals=propose_topologies(self.source,self.seed,self.process,self.envelope,search,np.random.default_rng(3),2)
        self.assertIs(proposals[0],self.seed)
        self.assertEqual(len(proposals),2)
        self.assertEqual(len(proposals.warm_starts[0].rings),58)
        self.assertIn(2,[r.slot_index for r in proposals.warm_starts[0].rings if r.layer_index==1])
        self.assertNotEqual(topology_identity(proposals[0]),topology_identity(proposals[1]))
        self.assertEqual(proposals.metadata["warm_swap_count"],1)
        self.assertGreater(proposals.metadata["random_draws"],0)

    def test_random_topologies_obey_stock_mirrors_and_source_pool(self):
        proposals=propose_topologies(self.source,self.seed,self.process,self.envelope,dict(self.search,azimuth_count_mode="per_pair",optimize_radial_ring_count=True,optimize_azimuth_count=True),np.random.default_rng(32),12)
        self.assertEqual(len(proposals),12)
        for design in proposals:
            self.assertLessEqual(design.magnet_count,self.process["max_magnet_count"])
            self.assert_printable(design)
            rings={(r.layer_index,r.slot_index):r for r in design.rings}
            for ring in design.rings:
                mirror=rings[(13-ring.layer_index,ring.slot_index)]
                self.assertEqual(ring.radius_mm,mirror.radius_mm)
                self.assertEqual(ring.z_mm,-mirror.z_mm)
                self.assertEqual(ring.count,mirror.count)
        dims={len(self.space(seed=design)[0]) for design in proposals}
        self.assertGreater(len(dims),1)
        self.assertGreater(proposals.metadata["random_draws"],0)

    def test_fixed_discrete_switches_return_only_seed(self):
        search=dict(self.search,optimize_radial_ring_count=False,optimize_azimuth_count=False,optimize_magnetization=False)
        proposals=propose_topologies(self.source,self.seed,self.process,self.envelope,search,np.random.default_rng(2),10)
        self.assertEqual(proposals,[self.seed])


if __name__=="__main__":unittest.main()

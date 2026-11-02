"""Deterministic quantity coverage without altering the historical random prior."""
from dataclasses import replace
import copy
import math
import unittest
import numpy as np
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.manufacturing import make_manufacturing_seed, audit_manufacturing
from mri_magnet.topology import propose_topologies


PROCESS={"enabled":True,"layout_mode":"radial_channels",
    "pocket_clearance_per_side_mm":.2,"pocket_web_mm":1.2,"inner_wall_mm":1.2,
    "outer_wall_mm":1.2,"base_thickness_mm":1.6,"sectors_per_layer":4,
    "printer_bed_mm":[220,220,250],"bed_margin_per_side_mm":4,
    "max_magnet_count":1736,"min_axial_module_gap_mm":0}
LIMITS={"bore_radius_mm":120.,"outer_radius_mm":math.hypot(192,6),
    "half_length_mm":253.,"min_gap_mm":0.,"tolerance_mm":1e-6,"preserve_exact_bore":False}
SEARCH={"min_radial_rings_per_layer":2,"max_radial_rings_per_layer":6,
    "azimuth_count_candidates":[24,28,32,36,40],"azimuth_count_mode":"shared",
    "optimize_radial_ring_count":True,"optimize_azimuth_count":True,
    "optimize_magnetization":False,"stock_aware_column_neighbors":True}


class StockAwareTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rings=[]
        for pair,(n,z) in enumerate(zip([5,6,5,6,3,3,3],[247,209,171,133,95,57,19])):
            radii=np.linspace(128 if n==5 else 126 if n==6 else 127,186,n)
            for positive,layer in ((True,pair),(False,13-pair)):
                for slot,radius in enumerate(radii):
                    rings.append(RingSpec(f"R{layer+1:02d}-L{slot+1}",layer,slot,float(radius),
                        float(z if positive else -z),28,"out" if positive else "in",-90+180/28))
        cls.source=Design(tuple(sorted(rings,key=lambda r:(r.layer_index,r.slot_index))))
        cls.seed,report=make_manufacturing_seed(cls.source,PROCESS,LIMITS)
        assert report['valid'],report['violations']
        assert cls.seed.magnet_count==1624

    def proposals(self,search=None,process=None,limit=2,random_seed=20261028):
        return propose_topologies(self.source,self.seed,process or PROCESS,LIMITS,
            search or SEARCH,np.random.default_rng(random_seed),limit)

    def test_quick_quota_guarantees_valid_increased_quantity_neighbour(self):
        proposals=self.proposals()
        self.assertEqual([(len(d.rings),d.magnet_count) for d in proposals],[(58,1624),(54,1728)])
        neighbour=proposals[1]
        self.assertEqual({r.count for r in neighbour.rings},{32})
        self.assertEqual({r.layer_index for r in neighbour.rings},set(range(14)))
        self.assertEqual([sum(r.layer_index==i for r in neighbour.rings) for i in range(7)],
                         [4,4,5,5,3,3,3])
        audit=audit_manufacturing(neighbour,PROCESS,LIMITS)
        self.assertTrue(audit['valid'],audit['violations'])
        self.assertLessEqual(neighbour.magnet_count,PROCESS['max_magnet_count'])
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],1)
        attempt=proposals.metadata['stock_neighbor_attempts'][0]
        self.assertEqual(attempt['deleted_mirrored_ring_pairs'],2)
        self.assertEqual(attempt['status'],'accepted')
        self.assertEqual(proposals.metadata['random_draws'],0)
        self.assertTrue(all(not row for row in proposals.metadata['random_prior_ring_count_histograms']))

    def test_disabled_flag_preserves_original_rng_result_and_warm_swap(self):
        disabled=self.proposals(dict(SEARCH,stock_aware_column_neighbors=False))
        absent_search=dict(SEARCH);del absent_search['stock_aware_column_neighbors']
        absent=self.proposals(absent_search)
        self.assertEqual([d.to_dict() for d in disabled],[d.to_dict() for d in absent])
        self.assertEqual(disabled[1].magnet_count,1008)
        self.assertEqual({r.count for r in disabled[1].rings},{28})
        self.assertEqual([sum(r.layer_index==i for r in disabled[1].rings) for i in range(7)],
                         [2,2,4,2,2,3,3])
        self.assertEqual(disabled.metadata['random_draws'],1)
        self.assertEqual(disabled.metadata['stock_neighbor_accepted'],0)
        enabled=self.proposals()
        self.assertEqual([d.to_dict() for d in enabled.warm_starts],
                         [d.to_dict() for d in disabled.warm_starts])

    def test_fixed_columns_disables_stock_neighbour(self):
        proposals=self.proposals(dict(SEARCH,optimize_azimuth_count=False))
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],0)
        self.assertTrue(all({r.count for r in d.rings}=={28} for d in proposals))

    def test_fixed_ring_counts_cannot_be_deleted_to_fund_shared_columns(self):
        proposals=self.proposals(dict(SEARCH,optimize_radial_ring_count=False))
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],0)
        self.assertTrue(all(len(d.rings)==58 for d in proposals))
        self.assertTrue(all('ring counts are fixed' in a['status']
            for a in proposals.metadata['stock_neighbor_attempts']))

    def test_per_pair_column_mode_changes_one_pair_without_forcing_shared_count(self):
        proposals=self.proposals(dict(SEARCH,azimuth_count_mode='per_pair',optimize_radial_ring_count=False))
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],1)
        neighbour=proposals[1]
        self.assertEqual(len(neighbour.rings),58)
        self.assertEqual(neighbour.magnet_count,1664)
        self.assertEqual([{r.count for r in neighbour.rings if r.layer_index==pair}
            for pair in range(7)],[{32},{28},{28},{28},{28},{28},{28}])
        self.assertTrue(audit_manufacturing(neighbour,PROCESS,LIMITS)['valid'])

    def test_insufficient_stock_does_not_raise_inventory_or_claim_increase(self):
        process=dict(PROCESS,max_magnet_count=self.seed.magnet_count)
        proposals=self.proposals(process=process)
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],0)
        self.assertTrue(all(d.magnet_count<=self.seed.magnet_count for d in proposals))

    def test_limit_one_preserves_seed_only(self):
        proposals=self.proposals(limit=1)
        self.assertEqual(list(proposals),[self.seed])
        self.assertEqual(proposals.metadata['stock_neighbor_attempts'],[])

    def test_no_higher_allowed_columns_cannot_invent_32(self):
        proposals=self.proposals(dict(SEARCH,azimuth_count_candidates=[24,28]))
        self.assertEqual(proposals.metadata['stock_neighbor_accepted'],0)
        self.assertTrue(all({r.count for r in d.rings}<={24,28} for d in proposals))


if __name__=='__main__':unittest.main()

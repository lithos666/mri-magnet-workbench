"""Keep topology champions in the bounded independent dense-validation set."""
import copy
import unittest
from mri_magnet.workflow import _select_validation_candidates
from mri_magnet.geometry import Design, RingSpec


REQUIREMENTS={'min_mean_norm_T':.03,'max_pp_norm_ppm':100.,'ppm_scale':100.}

def row(name,ppm,topology=None,field=.04,key=None):
    result={'candidate_id':name,'canonical_physical_sha256':key or name,
            'metrics':{'mean_norm_T':field,'pp_norm_ppm':ppm}}
    if topology is not None:result['topology_index']=topology
    return result


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.records=[row('a1',.1,1),row('a2',.2,1),row('a3',.3,1),row('a4',.4,1),
                      row('more_magnets',.8,2,field=.041)]

    def select(self,records=None,maximum=4,excluded=()):
        return _select_validation_candidates(self.records if records is None else records,
            maximum,REQUIREMENTS,excluded)

    def test_second_topology_not_crowded_out_by_four_similar_primary_candidates(self):
        before=copy.deepcopy(self.records)
        selected,metadata=self.select()
        self.assertEqual([r['candidate_id'] for r in selected],['a1','more_magnets','a2','a3'])
        self.assertEqual(metadata['covered_topology_indices'],[1,2])
        self.assertEqual(metadata['selected_count'],4)
        self.assertTrue(metadata['all_available_topologies_covered'])
        self.assertFalse(metadata['thresholds_relaxed'])
        self.assertEqual(self.records,before)

    def test_single_slot_is_always_global_best_and_uncovered_topology_is_explicit(self):
        selected,metadata=self.select(maximum=1)
        self.assertEqual([r['candidate_id'] for r in selected],['a1'])
        self.assertEqual(metadata['uncovered_topology_indices'],[2])
        self.assertFalse(metadata['all_available_topologies_covered'])

    def test_champion_priority_uses_same_feasibility_score_and_respects_cap(self):
        records=self.records+[row('third',.6,3),row('weak_field',.01,4,field=.02)]
        selected,metadata=self.select(records,maximum=2)
        self.assertEqual([r['candidate_id'] for r in selected],['a1','third'])
        self.assertEqual(metadata['uncovered_topology_indices'],[2,4])
        self.assertEqual(len(selected),2)

    def test_excludes_references_and_duplicate_physical_designs_before_coverage(self):
        records=[row('reference',.01,2,key='seed'),row('a1',.1,1,key='physical_a'),
                 row('duplicate_label',.15,2,key='physical_a'),row('b',.8,2),row('c',.4,1)]
        selected,metadata=self.select(records,maximum=3,excluded={'seed'})
        self.assertEqual([r['candidate_id'] for r in selected],['a1','b','c'])
        self.assertEqual(metadata['duplicate_physical_records'],1)
        self.assertEqual(metadata['excluded_reference_records'],1)
        self.assertEqual(metadata['covered_topology_indices'],[1,2])

    def test_zero_cap_runs_no_dense_candidates(self):
        selected,metadata=self.select(maximum=0)
        self.assertEqual(selected,[])
        self.assertEqual(metadata['selected_count'],0)
        self.assertEqual(metadata['covered_topology_indices'],[])
        self.assertEqual(metadata['uncovered_topology_indices'],[1,2])

    def test_legacy_records_without_topology_keep_global_ranking(self):
        records=[row('b',2.),row('a',1.),row('c',3.)]
        selected,metadata=self.select(records,maximum=2)
        self.assertEqual([r['candidate_id'] for r in selected],['a','b'])
        self.assertEqual(metadata['available_topology_indices'],[])

    def test_missing_canonical_hash_falls_back_to_physical_design(self):
        design=Design((RingSpec('r',0,0,130.,20.,28,'out',0.),)).to_dict()
        records=[{'candidate_id':'first','metrics':{'mean_norm_T':.04,'pp_norm_ppm':1.},'design':design},
                 {'candidate_id':'same','metrics':{'mean_norm_T':.04,'pp_norm_ppm':2.},'design':design}]
        selected,metadata=self.select(records)
        self.assertEqual([r['candidate_id'] for r in selected],['first'])
        self.assertEqual(metadata['duplicate_physical_records'],1)

    def test_empty_records_and_invalid_caps(self):
        selected,metadata=self.select([],maximum=4)
        self.assertEqual(selected,[])
        self.assertEqual(metadata['selected_count'],0)
        for invalid in (-1,True,1.5):
            with self.subTest(cap=invalid),self.assertRaises(ValueError):self.select(maximum=invalid)


if __name__=='__main__':unittest.main()

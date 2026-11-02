"""Real-source constraint integration, with no magnetic-field solve."""
import copy
import os
from pathlib import Path
import runpy
import unittest
from mri_magnet.geometry import load_source
from mri_magnet.workflow import _limits,_prepare_seed

class ManufacturingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.config=runpy.run_path(str(root/'parameters.py'))['CONFIG']
        path=Path(os.environ.get('MRI_SOURCE',str(root/'examples'/'seeded'/'placement.json')))
        if not path.is_file():raise unittest.SkipTest('Actual source JSON not available')
        cls.source,_=load_source(path)
        cls.envelope=_limits(cls.source,cls.config)

    def test_material_budget_constructs_a_feasible_seed_instead_of_false_no_solution(self):
        for budget in (1736,1400,1000):
            with self.subTest(budget=budget):
                process=dict(self.config['manufacturing'],max_magnet_count=budget)
                # This case explicitly permits count changes; UI defaults no
                # longer decide the mathematical scope of the fixture.
                search=dict(self.config['search'],optimize_radial_ring_count=True,optimize_azimuth_count=True)
                seed,report=_prepare_seed(self.source,process,self.envelope,search)
                self.assertTrue(report['valid'],report['violations'])
                self.assertLessEqual(seed.magnet_count,budget)
                self.assertEqual(len({r.layer_index for r in seed.rings}),14)

    def test_impossible_source_subset_range_cannot_be_mislabeled_valid(self):
        search=dict(self.config['search'],min_radial_rings_per_layer=4,max_radial_rings_per_layer=5)
        with self.assertRaisesRegex(ValueError,'源槽数量不足'):
            _prepare_seed(self.source,self.config['manufacturing'],self.envelope,search)
        search['min_radial_rings_per_layer']=6
        with self.assertRaises(ValueError):_prepare_seed(self.source,self.config['manufacturing'],self.envelope,search)

if __name__=='__main__':unittest.main()

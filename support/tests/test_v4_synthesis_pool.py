"""Quality screening must retain late, physically legal topology incumbents."""
import unittest
from unittest.mock import patch
import numpy as np

from mri_magnet.de_novo import synthesize, STYLES
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.workbench_config import default_config


class SynthesisPoolTests(unittest.TestCase):
    def run_pool(self, pool_size=None, attempts=20):
        cfg=default_config()
        cfg['target'].update(min_field_mT=40., max_ppm=None)
        cfg['scoring']={'mode':'balanced','field_weight':.5,'field_gain_scale':.1,'ppm_scale':1000.}
        cfg['de_novo'].update(pair_counts=[8], column_counts=[24,28,32,36], seed_count=5,
            max_attempts=attempts)
        if pool_size is not None:
            cfg['de_novo']['candidate_pool_size']=pool_size
        examined=[]
        def basis(config,sample,pairs,columns,style,*args):
            examined.append((pairs,columns,style))
            return {'index':len(examined),'pair_count':pairs,'column_count':columns,'style':style,
                'mean_Az_T_per_T':np.array([.2]), 'material':cfg['magnet']}
        def solve(b,config):
            index=b['index'];count=b['column_count'];pairs=b['pair_count']
            radius=145.+index*.1
            rings=[]
            for layer in range(pairs):
                z=200.-24*layer
                for current,zz,pol in ((layer,z,'out'),(2*pairs-1-layer,-z,'in')):
                    rings.append(RingSpec('R%d-L1'%current,current,0,radius,zz,count,pol,-90+180/count))
            return Design(tuple(rings),name='pool_%d'%index,**cfg['magnet']), {'integer_incumbent':True}
        def evaluate(design,*args):
            index=int(design.name.split('_')[-1])
            # The final candidate strictly dominates the entire early prefix.
            metrics={'mean_norm_T':.06 if index==20 else .045+index*.00001,
                'pp_norm_ppm':5. if index==20 else 1000.-index*10.}
            return metrics,np.zeros((1,3))
        with patch('mri_magnet.de_novo.build_basis',side_effect=basis), \
             patch('mri_magnet.de_novo._solve_basis',side_effect=solve), \
             patch('mri_magnet.de_novo.evaluate_design',side_effect=evaluate):
            result=synthesize(cfg,{},log=lambda *unused:None)
        return result,examined

    def test_late_dominating_candidate_survives_style_quality_screen(self):
        result,examined=self.run_pool(20)
        self.assertEqual(len(result['seeds']),5)
        self.assertEqual(result['seeds'][0]['id'],'synthesis_20')
        self.assertEqual({row['style'] for row in result['seeds']},set(STYLES))
        self.assertTrue(all(row['audit']['valid'] for row in result['candidate_pool']))
        self.assertEqual(len(result['candidate_pool']),20)
        self.assertEqual(result['metadata']['seed_field_evaluations'],20)
        self.assertEqual(result['metadata']['generated_candidate_pool_size'],20)
        self.assertEqual(len(result['metadata']['candidate_pool_metrics']),20)
        self.assertFalse(result['metadata']['source_layout_read'])

    def test_complete_case_pool_covers_every_column_style_combination(self):
        result,examined=self.run_pool(20)
        expected={(8,col,style) for col in (24,28,32,36) for style in STYLES}
        self.assertEqual(set(examined),expected)
        self.assertEqual(len(examined),20)
        self.assertTrue(result['metadata']['coverage']['full_cartesian_search_performed'])
        self.assertFalse(result['metadata']['global_optimum_proven'])

    def test_unspecified_pool_preserves_first_legal_cluster(self):
        result,examined=self.run_pool()
        self.assertEqual(len(examined),5)
        self.assertEqual([row['id'] for row in result['seeds']],
            ['synthesis_%02d'%i for i in range(1,6)])
        self.assertEqual(result['metadata']['seed_field_evaluations'],5)

    def test_attempt_limit_preserves_honest_partial_pool(self):
        result,examined=self.run_pool(20,attempts=7)
        self.assertEqual(len(examined),7)
        self.assertEqual(len(result['seeds']),5)
        self.assertEqual(result['metadata']['candidate_pool_status'],'partial')
        self.assertEqual(result['metadata']['seed_field_evaluations'],7)
        self.assertEqual(len(result['candidate_pool']),7)
        self.assertFalse(result['metadata']['coverage']['full_cartesian_search_performed'])

    def test_pool_cannot_be_smaller_than_output_cluster(self):
        with self.assertRaisesRegex(ValueError,'candidate_pool_size'):
            self.run_pool(4)


if __name__=='__main__':unittest.main()

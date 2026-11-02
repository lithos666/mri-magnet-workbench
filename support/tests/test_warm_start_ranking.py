"""Cached warm-priority rules do not reject valid geometry or solve fields."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
from mri_magnet.geometry import Design,RingSpec
from mri_magnet.workflow import _load_warm_starts

def design(name):return Design((RingSpec(name,0,0,130.,20.,28,'out',0.),),name=name)
def metrics(ppm,field=.04,diameter=20.,z=0.,polarization='remanence'):
    return dict(mean_norm_T=field,pp_norm_ppm=ppm,evaluation_geometry='disk',
        evaluation_z_mm=z,evaluation_diameter_mm=diameter,polarization_model=polarization)
def row(name,data=None):
    value={'candidate_id':name,'design':design(name).to_dict()}
    if data is not None:value['metrics']=data
    return value

class WarmRankingTests(unittest.TestCase):
    def execute(self,payloads,requirements=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config={'parameters_file':str(root/'parameters.py'),'fov_diameter_mm':20.,'fov_z_mm':0.,
                'polarization_model':'remanence','search':{'warm_start_runs':list(payloads)}}
            for name,rows in payloads.items():
                path=root/name/'详细数据';path.mkdir(parents=True)
                (path/'report_payload.json').write_text(json.dumps({'candidates':rows}),encoding='utf-8')
            checker=lambda d:{'valid':d.name!='invalid_baseline'}
            with patch('mri_magnet.workflow.FieldEngine',side_effect=AssertionError('ranking must not solve fields')):
                return _load_warm_starts(config,checker,requirements)

    def test_best_cached_candidate_precedes_old_seed_across_runs(self):
        loaded,audit=self.execute({'one':[row('invalid_baseline',metrics(.01)),row('old_seed',metrics(47.))],
                                  'two':[row('best',metrics(.209)),row('second',metrics(.216))]})
        self.assertEqual([d.name for d in loaded],['best','second','old_seed'])
        self.assertEqual([a['candidate_id'] for a in audit],['invalid_baseline','old_seed','best','second'])
        self.assertEqual([a['source_index'] for a in audit],list(range(4)))
        self.assertEqual([a['load_rank'] for a in audit],[None,3,1,2])
        self.assertTrue(all(a['cached_rank_only'] for a in audit))
        self.assertTrue(all(a['ranking_extra_field_evaluations']==0 for a in audit))

    def test_requirements_rank_feasible_candidate_ahead_of_weaker_flat_one(self):
        req={'min_mean_norm_T':.039,'max_pp_norm_ppm':100.,'field_scale_T':.04,'ppm_scale':100.}
        loaded,audit=self.execute({'run':[row('weak',metrics(.01,field=.02)),row('feasible',metrics(1.))]},req)
        self.assertEqual([d.name for d in loaded],['feasible','weak'])
        self.assertEqual(audit[0]['cached_rank_rule'],'current_requirements')

    def test_missing_mismatched_and_invalid_caches_remain_in_source_order_at_tail(self):
        entries=[row('missing'),row('wrong_diameter',metrics(.001,diameter=40.)),
            row('matching',metrics(5.)),row('wrong_z',metrics(.001,z=1.)),
            row('wrong_material',metrics(.001,polarization='isolated_cube_self_demag')),
            row('invalid_cache',metrics(float('nan')))]
        loaded,audit=self.execute({'one':entries})
        self.assertEqual([d.name for d in loaded],['matching','missing','wrong_diameter','wrong_z','wrong_material','invalid_cache'])
        self.assertEqual([a['metric_scope_matches'] for a in audit],[False,False,True,False,False,True])
        self.assertEqual([a['cached_rank_only'] for a in audit],[False,False,True,False,False,False])

    def test_training_cache_preferred_and_matching_validation_cache_can_fallback(self):
        a=row('a',metrics(100.));a['training_metrics']=metrics(1.)
        b=row('b',metrics(.1,diameter=40.));b['training_metrics']=metrics(.01,z=1.);b['validation_metrics']=metrics(2.)
        loaded,audit=self.execute({'one':[b,a]})
        self.assertEqual([d.name for d in loaded],['a','b'])
        self.assertEqual([item['cached_metric_source'] for item in audit],['validation_metrics','training_metrics'])

if __name__=='__main__':unittest.main()

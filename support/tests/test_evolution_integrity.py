"""Independent contracts for previews, final acceptance, curves and run relocation."""
from pathlib import Path
import tempfile,unittest,json
from unittest.mock import patch
import numpy as np
from mri_magnet.geometry import Design,RingSpec
from mri_magnet.evolution import EvolutionMonitor
from mri_magnet.validation_status import validation_status
from mri_magnet.run_naming import finish_run_name
from mri_magnet.benchmark_curves import optimization_curves,compare_curves


class PreviewField:
    queries=[]
    def __init__(self,*args,**kwargs):self.statistics={'engine_calls':0,'field_points':0}
    def field(self,design,points):
        points=np.asarray(points);self.queries.append(points.copy())
        self.statistics['engine_calls']+=1;self.statistics['field_points']+=len(points)
        return np.c_[np.zeros((len(points),2)),.04+1e-9*np.sum(points[:,:2]**2,axis=1)]


class EvolutionIntegrityTests(unittest.TestCase):
    def test_preview_stays_on_disk_and_writes_animation_and_timeline(self):
        seed=Design((RingSpec('a',0,0,130.,20.,4,'out',0.),))
        metrics={'mean_norm_T':.04,'pp_norm_ppm':20.}
        config={'fov_diameter_mm':20.,'fov_z_mm':0.,'chunk_points':96,'polarization_model':'remanence',
            'evolution':{'enabled':True,'grid_points':9,'min_refresh_seconds':0,'max_frames':4}}
        PreviewField.queries=[]
        with tempfile.TemporaryDirectory() as folder,patch('mri_magnet.evolution.FieldEngine',PreviewField):
            monitor=EvolutionMonitor(folder,seed,metrics,{'min_mean_norm_T':.039,'max_pp_norm_ppm':20.},config)
            monitor({'event':'baseline','phase':'seed','design':seed.to_dict(),'metrics':metrics,'field_evaluations':0})
            monitor({'event':'final','phase':'validation','design':seed.to_dict(),'metrics':dict(metrics,pp_norm_ppm=10.),'field_evaluations':8})
            result=monitor.finish()
            self.assertIsNone(result['error'])
            self.assertEqual(result['frames_count'],2)
            self.assertEqual(result['preview_engine_statistics']['engine_calls'],1)
            self.assertGreaterEqual(result['preview_field_cache_hits'],2)
            self.assertTrue((Path(folder)/result['gif_file']).is_file())
            self.assertTrue((Path(folder)/result['timeline_file']).is_file())
            self.assertIn('快照',(Path(folder)/result['timeline_file']).read_text(encoding='utf-8'))
            self.assertTrue(result['preview_cost_excluded_from_search_budget'])
        self.assertTrue(PreviewField.queries)
        for points in PreviewField.queries:
            self.assertTrue(np.all(points[:,2]==0))
            self.assertTrue(np.all(np.sum(points[:,:2]**2,axis=1)<=100+1e-8))

    def test_dense_rebound_remains_unqualified_and_does_not_relax_threshold(self):
        req={'min_mean_norm_T':.035,'max_pp_norm_ppm':47.}
        status=validation_status({'mean_norm_T':.036,'pp_norm_ppm':46.},
            {'mean_norm_T':.036,'pp_norm_ppm':47.00001},req,req)
        self.assertEqual(status['code'],'training_pass_dense_fail')
        self.assertFalse(status['dense_pass']);self.assertFalse(status['thresholds_relaxed'])
        self.assertEqual(req['max_pp_norm_ppm'],47.)

    def test_rename_rebases_artifact_paths_and_preserves_executed_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'results/20261029_230000_123456';details=run/'详细数据';details.mkdir(parents=True)
            model=run/'model.class';model.write_bytes(bytes.fromhex('cafebabe'))
            payload={'requirements':{},'candidates':[{'candidate_id':'best','feasible':True,
                'validation_metrics':{'mean_norm_T':.036,'pp_norm_ppm':.8}}]}
            (details/'report_payload.json').write_text(json.dumps(payload),encoding='utf-8')
            command=['compile',str(model)]
            (details/'export.json').write_text(json.dumps({'class_file':str(model),'compile_command':command}),encoding='utf-8')
            config={'parameters_file':str(root/'parameters.py'),'results':{'readable_names':True}}
            named=finish_run_name(run,config)
            self.assertIn('_最优36.00mT_0.800ppm',named.name)
            data=json.loads((named/'详细数据/export.json').read_text(encoding='utf-8'))
            self.assertTrue(Path(data['class_file']).is_file())
            self.assertEqual(data['executed_compile_command'],command)
            self.assertTrue((named/'详细数据/run_relocation.json').is_file())

    def test_explicit_output_name_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'user_folder';path.mkdir()
            self.assertEqual(finish_run_name(path,{'results':{'readable_names':True}},explicit_output=True),path)

    def test_matched_generation_axis_excludes_later_polish(self):
        def result(last):
            records=[{'candidate_id':'g%d'%i,'metrics':{'mean_norm_T':.04,'pp_norm_ppm':100/i},'global_evaluations':i*10} for i in range(1,last+1)]
            records.append({'candidate_id':'polish','metrics':{'mean_norm_T':.04,'pp_norm_ppm':.1},'global_evaluations':99})
            return {'records':records,'metadata':{'runs':[{'metadata':{'iteration_logs':[
                {'event':'generation','generation':i,'best_candidate_id':'g%d'%i,'field_evaluations_total':i*10} for i in range(1,last+1)]}}]}}
        fixed=optimization_curves(result(3),{});mixed=optimization_curves(result(2),{})
        self.assertEqual(fixed['total_evaluations'][-1]['metrics']['pp_norm_ppm'],.1)
        self.assertEqual(fixed['primary_de_generations'][-1]['metrics']['pp_norm_ppm'],100/3)
        with tempfile.TemporaryDirectory() as folder:
            comparisons,image=compare_curves([{'mode':'fixed','optimizer_seed':1,'curves':fixed},
                {'mode':'mixed','optimizer_seed':1,'curves':mixed}],folder)
            self.assertEqual(comparisons[0]['common_primary_generations'],[1,2])
            self.assertEqual(comparisons[0]['common_actual_evaluation_limit'],99)
            self.assertEqual(comparisons[0]['matched_actual_evaluation_rows']['fixed']['metrics']['pp_norm_ppm'],.1)
            self.assertTrue((Path(folder)/image).is_file())

if __name__=='__main__':unittest.main()

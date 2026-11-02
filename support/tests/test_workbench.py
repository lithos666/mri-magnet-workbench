"""State-machine, no-source, cancellation and variable-layer regression tests."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.manufacturing import audit_manufacturing
from mri_magnet.parameterization import get_search_space
from mri_magnet.workbench import (Workbench, preflight, requirements_for,
                                  envelope_for, FieldEvaluator, CancelledError, ManufacturingChecker)
from mri_magnet.workbench_config import default_config
from mri_magnet.workbench_domains import make_sample


def symmetric_seed(pairs):
    """An independently specified test fixture; never loads placement JSON."""
    rings = []
    for index, z in enumerate(np.linspace(220., 20., pairs)):
        for slot, radius in enumerate((130., 155., 180.)):
            partner = 2*pairs-1-index
            rings.extend((
                RingSpec('R%02d-L%d'%(index+1,slot+1), index, slot, radius, float(z), 24, 'out', -90.+180./24),
                RingSpec('R%02d-L%d'%(partner+1,slot+1), partner, slot, radius, -float(z), 24, 'in', -90.+180./24)))
    return Design(tuple(sorted(rings,key=lambda r:(r.layer_index,r.slot_index))))


class CancelWorkerEvaluator:
    def __call__(self, design):
        raise CancelledError('cooperative worker cancel')


class WorkbenchStateTests(unittest.TestCase):
    def test_de_novo_steps1_and_preflight_do_not_open_source_or_evaluate_field(self):
        with tempfile.TemporaryDirectory() as folder:
            wb = Workbench(folder)
            wb.update_config({'mode':'de_novo','source_geometry':'Z:/absent/placement-05.json'})
            with patch('mri_magnet.workbench.load_source', side_effect=AssertionError('source accessed')), \
                 patch('mri_magnet.field.FieldEngine.field', side_effect=AssertionError('field accessed')):
                result = wb.run_step(1)
                check = wb.preflight()
            self.assertIn('球形 DSV', result['message'])
            self.assertTrue(check['valid'], check['violations'])
            self.assertEqual(check['field_evaluations'], 0)
            self.assertGreaterEqual(check['max_axial_pairs'], 8)
            self.assertGreater(check['max_radial_rings'], 0)

    def test_step_dependencies_reject_out_of_order_work(self):
        with tempfile.TemporaryDirectory() as folder:
            wb = Workbench(folder)
            for step in (2,3,4,5,6):
                with self.assertRaisesRegex(ValueError, '前置步骤'):
                    wb.run_step(step)
            self.assertTrue(all(step['status']=='pending' for step in wb.state()['steps']))
            for invalid in (0,7):
                with self.assertRaisesRegex(ValueError, '1～6'):
                    wb.run_step(invalid)

    def test_config_changes_invalidate_all_completed_steps_and_old_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            wb = Workbench(folder)
            wb.run_step(1)
            for step in wb.data['steps']:
                step.update(status='passed',preview='old_preview.png')
            wb.data.update(seeds=[{'id':'old'}],selected_seed_ids=['old'])
            wb.data['artifacts']={'layout':'old_layout.png'}
            wb.update_config({'target':{'diameter_mm':22.}})
            state = wb.state()
            self.assertTrue(all(step['status']=='pending' and 'preview' not in step for step in state['steps']))
            self.assertEqual(state['seeds'], [])
            self.assertEqual(state['selected_seed_ids'], [])
            self.assertEqual(state['artifacts'], {})
            self.assertIsNone(state['run_dir'])
            self.assertEqual(state['config']['target']['min_field_mT'],30.)

    def test_identical_config_update_keeps_step_status(self):
        with tempfile.TemporaryDirectory() as folder:
            wb = Workbench(folder);wb.run_step(1)
            before=wb.state()
            wb.update_config({'target':{'diameter_mm':20.}})
            self.assertEqual(wb.state(),before)

    def test_persistent_state_reload_and_snapshot_are_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            wb = Workbench(folder);wb.run_step(1)
            snapshot=wb.state();snapshot['config']['target']['min_field_mT']=999.
            self.assertEqual(wb.config['target']['min_field_mT'],30.)
            loaded=Workbench(folder)
            self.assertEqual(loaded.data['steps'][0]['status'],'passed')
            self.assertEqual(loaded.config,wb.config)
            self.assertTrue((Path(folder)/loaded.data['run_dir']/'配置快照.json').exists())

    def test_select_seeds_requires_one_to_three_known_unique_ids_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder)
            wb.data['seeds']=[{'id':str(i)} for i in range(4)]
            for step in wb.data['steps']:
                step['status']='passed'
            for invalid in ([],['missing'],['0','1','2','3']):
                with self.assertRaisesRegex(ValueError,'1～3'):
                    wb.select_seeds(invalid)
            wb.select_seeds(['2','0','2'])
            self.assertEqual(wb.state()['selected_seed_ids'],['2','0'])
            self.assertEqual([s['id'] for s in wb._selected()],['2','0'])
            self.assertTrue(all(s['status']=='passed' for s in wb.data['steps'][:3]))
            self.assertTrue(all(s['status']=='pending' for s in wb.data['steps'][3:]))

    def test_errors_mark_conflicting_step_and_prevent_downstream_run(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder)
            wb.update_config({'target':{'diameter_mm':300.}})
            wb.run_step(1)
            with self.assertRaisesRegex(ValueError,'主评价域直径'):
                wb.run_step(2)
            self.assertEqual(wb.data['steps'][1]['status'],'error')
            self.assertIn('主评价域直径',wb.data['error'])
            with self.assertRaisesRegex(ValueError,'前置步骤'):
                wb.run_step(3)

    def test_seeded_mode_requires_a_source_but_de_novo_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.update_config({'mode':'seeded','source_geometry':''})
            with self.assertRaisesRegex(ValueError,'源排布路径'):
                wb.run_step(1)
            self.assertEqual(wb.data['steps'][0]['status'],'error')
            wb.update_config({'mode':'de_novo'})
            self.assertIn('指标已确认',wb.run_step(1)['message'])

    def test_cooperative_cancel_restores_pending_state_and_keeps_saved_files(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder)
            for step in wb.data['steps'][:4]:
                step['status']='passed'
            snapshot=Path(folder)/'snapshot.json';snapshot.write_text('{}',encoding='utf-8')
            def cancelled_step(**unused):
                wb.cancel();wb._check_cancel()
            with patch.object(wb,'_step5',side_effect=cancelled_step):
                result=wb.run_step(5)
            self.assertTrue(result['cancelled'])
            self.assertEqual(wb.data['steps'][4]['status'],'pending')
            self.assertTrue(snapshot.exists())
            # Starting another explicit step acknowledges/clears the old flag.
            wb.run_step(1)
            self.assertFalse(wb.cancel_file.exists())


class WorkbenchPhysicsContractTests(unittest.TestCase):
    def test_conflicting_process_and_quantity_bounds_are_explicit(self):
        cases=[({'envelope':{'max_outer_diameter_mm':245.}},'磁槽'),
               ({'manufacturing':{'max_magnet_count':1}},'数量预算'),
               ({'de_novo':{'pair_counts':[99]}},'轴向可容纳上限'),
               ({'de_novo':{'column_counts':[31]}},'偶数'),
               ({'manufacturing':{'printer_bed_mm':[10.,10.,10.]}},'打印床')]
        from mri_magnet.workbench_config import merge_config
        for changes, message in cases:
            with self.subTest(changes=changes):
                result=preflight(merge_config(default_config(),changes))
                self.assertFalse(result['valid'])
                self.assertIn(message,'；'.join(result['violations']))
                self.assertEqual(result['field_evaluations'],0)

    def test_requirements_are_absolute_and_do_not_reuse_old_baseline(self):
        config=default_config();config['target'].update(min_field_mT=42.,max_ppm=210.)
        req=requirements_for(config)
        self.assertEqual(req['min_mean_norm_T'],.042)
        self.assertEqual(req['max_pp_norm_ppm'],210.)
        self.assertEqual(req['field_tolerance_T'],0.)

    def test_5_and_8_pair_builders_preserve_mirror_and_manufacturing_constraints(self):
        config=default_config();process=config['manufacturing'];limits=envelope_for(config)
        for pairs in (5,8):
            with self.subTest(pairs=pairs):
                initial=symmetric_seed(pairs)
                self.assertTrue(audit_manufacturing(initial,process,limits)['valid'])
                variables,builder=get_search_space(initial,process,limits,config['search'])
                rng=np.random.default_rng(19+pairs)
                candidates=builder.local_sampler(rng,8,.8)
                for vector in candidates:
                    design=builder(vector)
                    self.assertEqual(len({r.layer_index for r in design.rings}),2*pairs)
                    self.assertTrue(audit_manufacturing(design,process,limits)['valid'])
                    np.testing.assert_allclose(builder.encode_design(design),vector,atol=1e-10)

    def test_sphere_disk_and_dense_sampling_keep_distinct_domain_contracts(self):
        config=default_config();config['sampling'].update(training_volume=8,training_surface=16,
            validation_volume=16,validation_surface=24)
        design=symmetric_seed(5)
        train=make_sample(config,designs=[design]);dense=make_sample(config,dense=True)
        self.assertTrue(train['workbench_symmetry']['used'])
        self.assertFalse(dense['workbench_symmetry']['used'])
        self.assertNotEqual(train['name'],dense['name'])
        self.assertTrue(np.any(np.asarray(dense['points_mm'])[:,2]!=0))
        config['target']['geometry']='disk'
        disk=make_sample(config,designs=[design])
        self.assertTrue(np.all(np.asarray(disk['points_mm'])[:,2]==0.))
        self.assertEqual(disk['geometry'],'disk')

    def test_field_evaluator_cancels_before_import_or_field_and_pickles_without_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            flag=Path(folder)/'cancel.flag';flag.write_text('cancel',encoding='utf-8')
            evaluator=FieldEvaluator({}, {}, str(flag))
            with self.assertRaises(CancelledError):
                evaluator(symmetric_seed(5))
            self.assertIsNone(evaluator.engine)
            evaluator.engine=object()
            restored=pickle.loads(pickle.dumps(evaluator))
            self.assertIsNone(restored.engine)

    def test_process_worker_cancellation_propagates_through_islands(self):
        from mri_magnet.islands import run_islands
        config=default_config();limits=envelope_for(config)
        settings={'max_evaluations':40,'maxiter':20,'optimization_options':{'population_size':5},
                  'islands':{'max_workers':2}}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(CancelledError):
                run_islands([symmetric_seed(5),symmetric_seed(8)],config['manufacturing'],limits,
                            config['search'],CancelWorkerEvaluator(),
                            ManufacturingChecker(config['manufacturing'],limits),folder,
                            settings,requirements_for(config))
            self.assertFalse((Path(folder)/'optimization_result.json').exists())


if __name__=='__main__':
    unittest.main()

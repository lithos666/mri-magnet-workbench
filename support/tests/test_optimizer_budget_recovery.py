"""Regression checks for completed local polling and auditable warm neighborhoods."""
from dataclasses import dataclass,replace
from pathlib import Path
import json,tempfile,unittest
import numpy as np
from parameters import CONFIG
from mri_magnet.optimization import run_optimization
from mri_magnet.geometry import load_source,envelope_limits
from mri_magnet.manufacturing import make_manufacturing_seed,audit_manufacturing
from mri_magnet.parameterization import get_search_space

@dataclass(frozen=True)
class Point:
    coords:tuple
    def to_dict(self):return {'coordinates':list(self.coords)}

def build_point(x):return Point(tuple(x))
build_point.encode_design=lambda p:np.asarray(p.coords,dtype=float)

def variables(n,initial):
    return [dict(name=f'x{i}',lower=0.,upper=1.,initial=initial,integer=False) for i in range(n)]

class RecoveryAndWarmTests(unittest.TestCase):
    def test_unused_pair_budget_finishes_all_axes_then_reduces_step(self):
        calls=[]
        def evaluate(p):
            calls.append(p)
            return {'mean_norm_T':.04,'pp_norm_ppm':1.+sum((x-.501)**2 for x in p.coords)}
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables(36,.5),build_point,evaluate,lambda p:{'valid':True},
                Point((.5,)*36),folder,seed=36,max_evaluations=85,population_size=5,maxiter=0,
                initial_local_fraction=1.,initial_sigma=1e-12,local_refine_evaluations=80,
                local_step=.03,pair_refine_fraction=.4)
        info=result['metadata']['rounds'][0]['local_refinement']
        self.assertEqual(info['paired_trials'],0)
        self.assertEqual(info['coordinate_initial_field_evaluations'],48)
        self.assertEqual(info['coordinate_recovery_field_evaluations'],32)
        self.assertEqual(info['coordinate_axes_visited'],list(range(36)))
        self.assertGreaterEqual(info['coordinate_sweeps_completed'],1)
        self.assertGreaterEqual(info['coordinate_step_reductions'],1)
        self.assertEqual(info['coordinate_final_steps'],[.015]*36)
        self.assertEqual(info['field_evaluations'],80)
        self.assertEqual(info['field_evaluations'],info['coordinate_field_evaluations']+info['paired_field_evaluations'])
        self.assertEqual(len(calls),result['metadata']['field_evaluations'])
        self.assertLessEqual(len(calls),85)
        # The first completed sweep visits each direction exactly once; a
        # restart from the leading axes after the paired stage would fail.
        local=[row for row in result['records'] if row['search_stage']=='local']
        axes=[int(np.argmax(np.abs(np.array([row['parameters'][f'x{i}'] for i in range(36)])-.5))) for row in local[:72]]
        self.assertEqual(np.bincount(axes,minlength=36).tolist(),[2]*36)
        final=result['metadata']['stage_summaries'][-1]
        self.assertEqual(final['stage'],'local_recovery')
        self.assertEqual(final['field_evaluations'],result['metadata']['field_evaluations'])

    def test_warm_neighborhood_is_centered_on_logged_input_without_extra_solves(self):
        calls=[]
        def evaluate(p):
            calls.append(p)
            return {'mean_norm_T':.04,'pp_norm_ppm':1.+sum((x-.8)**2 for x in p.coords)}
        with tempfile.TemporaryDirectory() as folder:
            result=run_optimization(variables(36,.4),build_point,evaluate,lambda p:{'valid':True},
                Point((.4,)*36),folder,seed=36,max_evaluations=32,population_size=32,maxiter=0,
                initial_local_fraction=.65,initial_sigma=.01,local_refine_evaluations=0,
                warm_starts=[Point((.8,)*36),Point((.6,)*36)],warm_local_fraction=.5)
            saved=json.loads((Path(folder)/'initial_population.json').read_text(encoding='utf-8'))
        population=np.array(saved['population'])
        warm=next(row for row in saved['local_center_audit'] if row['source']=='first_encodable_input_warm')
        seed=next(row for row in saved['local_center_audit'] if row['source']=='manufacturing_seed')
        self.assertEqual(len(warm['sample_rows']),10)
        self.assertEqual(len(seed['sample_rows']),10)
        self.assertEqual(warm['selection'],'input_order_not_score_ranked')
        self.assertEqual(warm['parameters'],{f'x{i}':.8 for i in range(36)})
        self.assertLess(float(np.sqrt(np.mean((population[warm['sample_rows']]-.8)**2))),.013)
        self.assertLess(float(np.sqrt(np.mean((population[seed['sample_rows']]-.4)**2))),.013)
        np.testing.assert_array_equal(population[warm['single_point_row']],np.full(36,.8))
        self.assertEqual(len(calls),result['metadata']['field_evaluations'])
        self.assertLessEqual(len(calls),32)
        self.assertEqual(result['diagnostic_best']['metrics']['pp_norm_ppm'],1.)

    def test_all_local_population_still_reserves_warm_center_and_neighbors(self):
        with tempfile.TemporaryDirectory() as folder:
            run_optimization(variables(2,.4),build_point,
                lambda p:{'mean_norm_T':.04,'pp_norm_ppm':1.+sum((x-.8)**2 for x in p.coords)},
                lambda p:{'valid':True},Point((.4,)*2),folder,seed=9,max_evaluations=8,
                population_size=8,maxiter=0,initial_local_fraction=1.,initial_sigma=.001,
                local_refine_evaluations=0,warm_starts=[Point((.8,)*2)],warm_local_fraction=.5)
            saved=json.loads((Path(folder)/'initial_population.json').read_text(encoding='utf-8'))
        warm=next(row for row in saved['local_center_audit'] if row['source']=='first_encodable_input_warm')
        self.assertGreater(len(warm['sample_rows']),0)
        self.assertNotIn(warm['single_point_row'],warm['sample_rows'])
        np.testing.assert_array_equal(saved['population'][warm['single_point_row']],[.8,.8])

    def test_bad_warm_fraction_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'warm_local_fraction'):
                run_optimization(variables(2,.5),build_point,lambda p:{'mean_norm_T':.04,'pp_norm_ppm':1.},
                    lambda p:{'valid':True},Point((.5,)*2),folder,warm_local_fraction=1.01)

class WarmPhysicalSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        path=(root/CONFIG['source_geometry']).resolve()
        if not path.is_file():raise unittest.SkipTest('Actual source unavailable')
        source,_=load_source(path)
        cls.process=dict(CONFIG['manufacturing'])
        cls.limits=dict(envelope_limits(source),preserve_exact_bore=False)
        cls.seed,_=make_manufacturing_seed(source,cls.process,cls.limits)
        cls.variables,builder=get_search_space(cls.seed,cls.process,cls.limits,CONFIG['search'])
        cls.builder=staticmethod(builder)

    def test_provided_center_reconstructs_and_retains_original_windows(self):
        center=self.builder(self.builder.initial_sampler(np.random.default_rng(91),1)[0])
        self.assertGreater(float(np.max(np.abs(center.magnet_arrays()['positions_mm']-self.seed.magnet_arrays()['positions_mm']))),1.)
        population=self.builder.local_sampler(np.random.default_rng(11),6,1e-10,center_design=center)
        for vector in population:
            design=self.builder(vector)
            np.testing.assert_allclose(design.magnet_arrays()['positions_mm'],center.magnet_arrays()['positions_mm'],rtol=0,atol=1e-8)
            self.assertTrue(audit_manufacturing(design,self.process,self.limits)['valid'])
        stats=self.builder.local_sampler.last_statistics
        self.assertEqual(stats['center_kind'],'provided_feasible_design')
        self.assertTrue(stats['axial_windows_remain_seed_windows'])
        np.testing.assert_allclose(stats['center_parameters'],self.builder.encode_design(center),rtol=0,atol=1e-12)

    def test_center_cannot_redefine_seed_displacement_windows(self):
        rings=tuple(replace(r,z_mm=r.z_mm+(25 if r.z_mm>0 else -25)) for r in self.seed.rings)
        displaced=replace(self.seed,rings=rings)
        with self.assertRaisesRegex(ValueError,'Axial position violates'):
            self.builder.local_sampler(np.random.default_rng(4),1,.8,center_design=displaced)

if __name__=='__main__':unittest.main()

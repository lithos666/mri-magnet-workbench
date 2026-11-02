"""Actual 16-layer dimensions must be affordable before any field work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mri_magnet.de_novo import _layout_lattice, _pair_design, limits_from_config
from mri_magnet.geometry import Design
from mri_magnet.islands import run_islands
from mri_magnet.solver_budget import (plan_island_budgets, effective_refinement_budgets,
                                      required_local_budget)
from mri_magnet.workbench_config import default_config


def physical_case(counts=None):
    cfg=default_config()
    cfg['envelope'].update(protected_bore_diameter_mm=200.,max_outer_diameter_mm=440.,max_length_mm=500.)
    cfg['manufacturing'].update(layout_mode='radial_channels',pocket_clearance_per_side_mm=.2,
        pocket_web_mm=1.6,inner_wall_mm=2.4,outer_wall_mm=2.4,base_thickness_mm=5.,
        radial_inner_clearance_mm=10.,min_central_magnet_gap_mm=10.,min_axial_module_gap_mm=2.,
        printer_bed_mm=[480.,480.,255.],machine_bed_mm=[255.,255.,255.],
        bed_margin_per_side_mm=5.,max_magnet_count=4608,sectors_per_layer=4)
    cfg['search'].update(max_radial_rings_per_layer=None,optimize_radius=True,
                         optimize_axial_positions=True,axial_shift_limit_mm=25.)
    material,limits,_,radii,zs=_layout_lattice(cfg,8,24,'wide')
    counts=counts or [8]*8
    rings=[]
    for k,(z,count) in enumerate(zip(zs,counts)):
        for j,radius in enumerate(radii[:count]):
            rings.extend(_pair_design(material,8,24,dict(pair_index=k,radial_index=j,
                radius_mm=float(radius),z_mm=float(z),polarity='out'),unit=False).rings)
    return cfg,Design(tuple(rings),name='16_layer_test',**material),limits


def settings(total=2400,population=32,local=240,lp=240,policy='strict'):
    return {'max_evaluations':total,'maxiter':100,'seed':12,
        'optimization_options':{'population_size':population,'local_refine_evaluations':local,
            'active_set_evaluations':lp,'pair_refine_fraction':.4},
        'islands':{'halving_factor':2,'finalists':2,'initial_budget_fraction':.3,
                   'min_round_evaluations':2*population,'budget_policy':policy}}


REQ={'min_mean_norm_T':.04,'scoring':{'mode':'balanced'}}


class SolverBudgetTests(unittest.TestCase):
    def plan(self,counts=None,config=None,n=1):
        cfg,design,limits=physical_case(counts)
        report=plan_island_budgets([design]*n,cfg['manufacturing'],limits,cfg['search'],config or settings(),REQ)
        return report

    def test_real_maximum_topology_counts_72_continuous_variables_without_fields(self):
        with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('must not solve fields')):
            report=self.plan()
        self.assertTrue(report['valid'],report['violations'])
        self.assertEqual(report['field_evaluations'],0)
        row=report['per_island'][0]
        self.assertEqual(row['dimensions'],72)
        self.assertEqual(row['coordinate_required_evaluations'],144)
        self.assertEqual(row['coordinate_initial_evaluations'],144)
        self.assertEqual(row['lp_first_pass_minimum'],74)
        self.assertEqual(row['lp_required_evaluations'],228)
        self.assertEqual(row['lp_affordable_passes'],3)

    def test_old_70_lp_limit_distinguishes_68_and_69_dimensions(self):
        low=self.plan([8,8,8,8,7,7,7,7],settings(local=0,lp=70,policy='allow_partial'))
        high=self.plan([8,8,8,8,8,7,7,7],settings(local=0,lp=70,policy='allow_partial'))
        self.assertEqual(low['per_island'][0]['D'],68)
        self.assertFalse(low['per_island'][0]['lp_will_skip'])
        self.assertEqual(high['per_island'][0]['D'],69)
        self.assertTrue(high['per_island'][0]['lp_will_skip'])
        self.assertEqual(high['per_island'][0]['lp_first_pass_minimum'],71)

    def test_strict_rejects_incomplete_scans_and_suggests_sufficient_combined_budget(self):
        original=settings(total=1200,population=24,local=64,lp=70)
        report=self.plan(config=original,n=3)
        self.assertFalse(report['valid'])
        self.assertTrue(any('跳过LP' in text for text in report['violations']))
        self.assertTrue(any('双向扫描' in text for text in report['violations']))
        suggestion=report['suggested_solver']
        updated=deepcopy(original);updated['max_evaluations']=suggestion['max_evaluations']
        updated['optimization_options'].update({k:v for k,v in suggestion.items() if k!='max_evaluations'})
        fixed=self.plan(config=updated,n=3)
        self.assertTrue(fixed['valid'],fixed['violations'])
        self.assertEqual(original['max_evaluations'],1200,'precheck cannot alter user cap')

    def test_island_allocation_priority_exposes_lp_clipping_even_with_large_requested_lp(self):
        report=self.plan(config=settings(total=1200,population=24,local=240,lp=300),n=3)
        self.assertEqual(report['round_sizes'],[3,2])
        self.assertEqual(report['rounds'][0]['allocations'],[120,120,120])
        self.assertEqual(report['rounds'][1]['allocations'],[420,420])
        row=report['per_island'][0]
        self.assertEqual(row['effective_lp_evaluations'],132)
        self.assertFalse(report['valid'])
        self.assertTrue(any('压缩' in warning for warning in report['warnings']))

    def test_allow_partial_is_explicit_warning_and_zero_reserves_are_deliberate_off(self):
        report=self.plan(config=settings(local=64,lp=70,policy='allow_partial'))
        self.assertTrue(report['valid']);self.assertEqual(report['status'],'warning')
        self.assertTrue(report['violations'])
        disabled=self.plan(config=settings(local=0,lp=0))
        self.assertTrue(disabled['valid'])
        self.assertEqual(disabled['per_island'][0]['local_state'],'disabled')
        self.assertEqual(disabled['per_island'][0]['lp_state'],'disabled')

    def test_180_does_not_claim_three_lp_passes_and_pair_reserve_is_not_double_counted(self):
        report=self.plan(config=settings(local=180,lp=180,policy='allow_partial'))
        row=report['per_island'][0]
        self.assertEqual(row['coordinate_initial_evaluations'],108)
        self.assertEqual(row['paired_reserved_evaluations'],72)
        self.assertEqual(row['lp_affordable_passes'],2)
        self.assertEqual(row['coordinate_axis_coverage'],54)

    def test_100_percent_pair_reserve_has_actionable_fix(self):
        config=settings();config['optimization_options']['pair_refine_fraction']=1.
        report=self.plan(config=config)
        self.assertFalse(report['valid'])
        self.assertEqual(report['suggested_solver']['pair_refine_fraction'],.4)
        self.assertIsNone(required_local_budget(72,1.))

    def test_allow_partial_cannot_waive_hard_scheduler_minimum(self):
        report=self.plan(config=settings(total=1,local=0,lp=0,policy='allow_partial'),n=3)
        self.assertFalse(report['valid'])
        self.assertEqual(report['status'],'conflict')
        self.assertGreater(report['minimum_scheduled_budget'],report['max_evaluations'])

    def test_strict_guard_runs_before_field_checker_and_output_directory(self):
        cfg,design,limits=physical_case()
        evaluator=Mock();checker=Mock()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'must_not_exist'
            with self.assertRaisesRegex(ValueError,'求解预算预检未通过'):
                run_islands([design],cfg['manufacturing'],limits,cfg['search'],evaluator,checker,output,
                            settings(local=64,lp=70),REQ)
            self.assertFalse(output.exists())
        evaluator.assert_not_called();checker.assert_not_called()

    def test_scheduler_and_precheck_share_effective_budget_arithmetic(self):
        self.assertEqual(effective_refinement_budgets(420,24,240,300,True),(240,132))
        self.assertEqual(effective_refinement_budgets(120,24,240,300,False),(0,0))
        report=self.plan(config=settings(total=2400),n=3)
        self.assertTrue(report['valid'])
        self.assertEqual(report['per_island'][0]['final_allocation_min'],840)
        self.assertGreaterEqual(report['per_island'][0]['de_generations_nominal'],1)

    def test_lifecycle_events_are_live_and_keep_global_ledger_monotone(self):
        from test_islands import seed, PROCESS, LIMITS, SEARCH, Evaluator, Checker, settings as legacy_settings
        events=[]
        with tempfile.TemporaryDirectory() as folder:
            result=run_islands([seed(24,5),seed(28,6),seed(32,8)],PROCESS,LIMITS,SEARCH,
                Evaluator(),Checker(),folder,legacy_settings(),{'min_mean_norm_T':.03},event_callback=events.append)
        statuses=[event for event in events if event['event']=='island_status']
        for expected in ('queued','running','completed','promoted','eliminated'):
            self.assertIn(expected,{event['island_status'] for event in statuses})
        counts=[event['global_evaluations'] for event in events]
        self.assertEqual(counts,sorted(counts))
        self.assertLessEqual(max(counts),result['metadata']['field_evaluations'])


if __name__=='__main__':unittest.main()

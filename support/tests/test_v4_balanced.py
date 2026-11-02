"""Balanced scoring hard gates and dense Pareto deliverables."""
import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mri_magnet.geometry import Design, RingSpec
from mri_magnet.objectives import (balanced_components, check_performance,
    feasibility_score, pareto_front, pareto_roles, scoring_options,
    select_validation_candidates)
from mri_magnet.workbench_archive import archive_run
from mri_magnet.workbench_config import default_config


def metrics(field_mT, ppm):
    return {'mean_norm_T':field_mT*1e-3,'pp_norm_ppm':ppm,'pointset':'dense_D100',
            'min_norm_T':field_mT*1e-3*(1-ppm/2e6),
            'max_norm_T':field_mT*1e-3*(1+ppm/2e6)}


def requirements():
    return {'min_mean_norm_T':.04,'max_pp_norm_ppm':None,'field_scale_T':.04,
            'scoring':{'mode':'balanced','field_weight':.5,'field_gain_scale':.1,'ppm_scale':1000.}}


def rows():
    return [{'candidate_id':name,'metrics':metrics(field,ppm)} for name,field,ppm in
            [('low',40.,10.),('balanced',44.,300.),('high',80.,1000000.),
             ('dominated',42.,500.),('below_gate',39.,0.)]]


class BalancedScoringTests(unittest.TestCase):
    def test_hard_field_gate_cannot_be_bought_by_zero_ppm(self):
        req=requirements()
        self.assertFalse(check_performance(metrics(39.999999,0.),req)['valid'])
        self.assertLess(feasibility_score(metrics(40.,1e200),req),0.)
        self.assertGreater(feasibility_score(metrics(39.999999,0.),req),0.)

    def test_balanced_prefers_a_real_tradeoff_over_both_extremes(self):
        req=requirements();candidates=rows()
        role=pareto_roles(candidates,req)
        self.assertEqual(role['balanced']['candidate_id'],'balanced')
        self.assertEqual(role['lowest_ppm']['candidate_id'],'low')
        self.assertEqual(role['high_field']['candidate_id'],'high')
        self.assertLess(feasibility_score(metrics(40.,10.),req),
                        feasibility_score(metrics(40.04,1000.),req))

    def test_score_is_monotone_in_each_objective(self):
        req=requirements()
        self.assertLess(feasibility_score(metrics(45.,300.),req),feasibility_score(metrics(44.,300.),req))
        self.assertLess(feasibility_score(metrics(44.,200.),req),feasibility_score(metrics(44.,300.),req))
        for field in (40.,44.,1e9):
            self.assertTrue(-1. <= feasibility_score(metrics(field,1e100),req) < 0.)

    def test_legacy_ppm_mode_keeps_original_ranking(self):
        req=requirements();req.pop('scoring');req['ppm_scale']=1000.
        self.assertEqual(feasibility_score(metrics(40.,300.),req),-1/(1+.3))
        self.assertLess(feasibility_score(metrics(40.,10.),req),feasibility_score(metrics(80.,300.),req))

    def test_formula_has_explicit_dimensionless_scales(self):
        value=balanced_components(metrics(44.,1000.),requirements())
        self.assertAlmostEqual(value['field_gain_fraction'],.1)
        self.assertAlmostEqual(value['field_reward'],math.log(2.))
        self.assertAlmostEqual(value['ppm_penalty'],math.log(2.))
        self.assertAlmostEqual(value['loss'],0.)

    def test_front_and_top_k_preserve_distinct_extremes(self):
        req=requirements();candidates=rows()
        self.assertEqual({r['candidate_id'] for r in pareto_front(candidates,req)},
                         {'low','balanced','high'})
        selected=select_validation_candidates(candidates,req,limit=3)
        self.assertEqual([r['candidate_id'] for r in selected],['balanced','low','high'])
        self.assertEqual(pareto_roles([candidates[-1]],req),{})
        self.assertEqual(select_validation_candidates(candidates,req,limit=0),[])

    def test_explicit_manufacturing_failure_never_gets_a_role(self):
        candidates=rows();candidates[1]['manufacturing_check']={'valid':False}
        self.assertNotIn('balanced',{r['candidate_id'] for r in pareto_front(candidates,requirements())})

    def test_equal_metric_ties_and_equal_field_domination_are_exact(self):
        candidates=[{'candidate_id':str(i),'metrics':metrics(field,ppm)}
                    for i,(field,ppm) in enumerate([(44.,300.),(44.,300.),(44.,400.),(43.,300.)])]
        self.assertEqual({r['candidate_id'] for r in pareto_front(candidates,requirements())},{'0','1'})

    def test_invalid_scales_are_rejected(self):
        for options in ({'mode':'unknown'},{'field_weight':-1.},{'field_weight':float('nan')},
                        {'field_gain_scale':0.},{'ppm_scale':float('inf')}):
            with self.assertRaises(ValueError):scoring_options(options)


class BalancedArchiveTests(unittest.TestCase):
    def test_dense_roles_dual_radii_and_front_are_archived_without_new_field_solves(self):
        config=default_config();config['target'].update(diameter_mm=100.,min_field_mT=40.,max_ppm=None)
        config['scoring']=requirements()['scoring'];config['deliverables']['compile']=False
        design=Design((RingSpec('R01-L1',0,0,140.,30.,24,'out',-82.5),
                       RingSpec('R02-L1',1,0,140.,-30.,24,'in',-82.5)),cube_mm=4.)
        candidates=rows()[:4]
        for row in candidates:row['design']=design.to_dict()
        with tempfile.TemporaryDirectory() as folder:
            with patch('mri_magnet.field.FieldEngine.field',side_effect=AssertionError('archive must not solve')):
                result=archive_run(folder,config,[],candidates,{},None,export_comsol=False)
            self.assertEqual(result['best']['candidate_id'],'balanced')
            self.assertEqual(len(result['pareto_recommendations']),3)
            self.assertEqual(set(result['pareto_front_ids']),{'low','balanced','high'})
            self.assertTrue((Path(folder)/result['artifacts']['pareto']).is_file())
            self.assertEqual(result['best']['archive_label'],'均衡推荐')
            overview=(Path(folder)/'本次运行速览.md').read_text(encoding='utf-8')
            self.assertIn('峰峰值不设硬上限',overview)
            self.assertIn('帕累托候选对比',overview)
            self.assertIn('场强增益尺度=10%',overview)
            with (Path(folder)/result['best']['changes_file']).open(encoding='utf-8-sig',newline='') as stream:
                physical=list(csv.DictReader(stream))
            self.assertEqual(float(physical[0]['中心R_mm'])-float(physical[0]['底边R_mm']),2.)
            md=(Path(folder)/result['best']['folder']/'尺寸改动_毫米.md').read_text(encoding='utf-8')
            self.assertIn('内侧底面 R_底 (mm)',md);self.assertIn('中心 R_中心 (mm)',md)

    def test_no_feasible_design_means_no_recommendation_labels(self):
        config=default_config();config['target'].update(min_field_mT=40.,max_ppm=None)
        config['scoring']=requirements()['scoring']
        design=Design((RingSpec('R01-L1',0,0,140.,30.,24,'out',-82.5),
                       RingSpec('R02-L1',1,0,140.,-30.,24,'in',-82.5)))
        row={'candidate_id':'diagnostic','design':design.to_dict(),'metrics':metrics(39.,1.)}
        with tempfile.TemporaryDirectory() as folder:
            result=archive_run(folder,config,[],[row],{},None,export_comsol=False)
            self.assertEqual(result['pareto_recommendations'],[])
            self.assertEqual(result['pareto_front_ids'],[])
            self.assertEqual(result['best']['archive_label'],'诊断未达标')


if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mri_magnet.config_schema import canonicalize_config, legacy_view
from mri_magnet.workbench import Workbench, preflight, requirements_for
from mri_magnet.workbench_config import default_config


class EngineeringConfigTests(unittest.TestCase):
    def test_default_bed_height_and_dynamic_rings_are_shared(self):
        cfg=default_config(); old=legacy_view(cfg)
        self.assertEqual(cfg['manufacturing']['printer_bed_mm'],[480.,480.,255.])
        self.assertEqual(cfg['manufacturing']['machine_bed_mm'],[255.,255.,255.])
        self.assertEqual(old['manufacturing']['bed_margin_per_side_mm'],5.)
        self.assertEqual(old['envelope']['max_half_height_mm'],250.)
        self.assertIsNone(cfg['search']['max_radial_rings_per_layer'])

    def test_document_aliases_normalize_without_second_authority(self):
        cfg=canonicalize_config({'MaterialsAndManufacturing':{'printer_bed_mm':{
            'bed_x_mm':480.,'bed_y_mm':480.,'bed_z_mm':255.,'margin_mm':5.}},
            'Constraints':{'envelope':{'max_total_height_mm':500.}}})
        self.assertNotIn('Constraints',cfg)
        self.assertNotIn('MaterialsAndManufacturing',cfg)
        self.assertEqual(cfg['envelope']['max_length_mm'],500.)
        self.assertEqual(cfg['manufacturing']['printer_bed_mm'],[480.,480.,255.])
        with self.assertRaises(ValueError):canonicalize_config({'envelope':{'max_total_height_mm':500.,'max_half_height_mm':200.}})

    def test_real_case_preflight_is_source_free_and_eight_ring_capacity(self):
        path=Path(__file__).resolve().parents[2]/'工程案例_100mmDSV.json'
        cfg=canonicalize_config(json.loads(path.read_text(encoding='utf-8')))
        check=preflight(cfg)
        self.assertTrue(check['valid'],check['violations'])
        self.assertEqual(check['field_evaluations'],0)
        self.assertEqual(check['max_radial_rings'],8)
        self.assertEqual(check['packing']['minimum_magnet_center_radius_mm'],116.)
        self.assertGreaterEqual(check['axial_packing']['minimum_central_magnet_face_gap_mm'],10.)
        self.assertGreaterEqual(check['max_axial_pairs'],8)
        self.assertTrue(check['bed_fit']['sector_fits_machine_bed'])
        self.assertTrue(check['bed_fit']['assembly_fits'])
        req=requirements_for(cfg)
        self.assertEqual(req['min_mean_norm_T'],.04)
        self.assertIsNone(req['max_pp_norm_ppm'])
        self.assertEqual(req['scoring']['mode'],'balanced')

    def test_upstream_edit_clears_preflight_and_pareto_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            wb=Workbench(folder);wb.preflight()
            wb.data['pareto_recommendations']=[{'role':'balanced'}]
            wb.update_config({'envelope':{'max_length_mm':510.}})
            self.assertNotIn('preflight',wb.data)
            self.assertNotIn('pareto_recommendations',wb.data)

    def test_fixed_eight_pairs_still_examines_every_requested_column_count(self):
        from mri_magnet.de_novo import synthesize
        cfg=default_config();cfg['de_novo'].update(pair_counts=[8],column_counts=[24,28,32,36],max_attempts=4)
        examined=[]
        def probe(config,sample,pairs,columns,*args):
            examined.append((pairs,columns))
            raise ValueError('catalog-only probe, no physics')
        with patch('mri_magnet.de_novo.build_basis',side_effect=probe):
            result=synthesize(cfg,{},log=lambda *args:None)
        self.assertEqual(examined,[(8,24),(8,28),(8,32),(8,36)])
        self.assertEqual(result['metadata']['total_engine_statistics_delta']['engine_calls'],0)

if __name__=='__main__':unittest.main()

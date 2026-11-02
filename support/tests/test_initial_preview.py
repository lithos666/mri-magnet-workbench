"""Initial viewport geometry is constrained, deterministic and display-only."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from mri_magnet.config_schema import canonicalize_config
from mri_magnet.geometry import Design,validate_design
from mri_magnet.manufacturing import audit_manufacturing
from mri_magnet.workbench_config import default_config
from mri_magnet.workbench import envelope_for


@unittest.skipUnless(shutil.which('node'),'Node is needed for the actual preview generator')
class InitialPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project=Path(__file__).resolve().parents[2]
        cls.module=cls.project/'mri_magnet/workbench_preview.js'

    def js(self,config,extra=''):
        source='const api=require('+json.dumps(str(self.module))+');const config='+json.dumps(config)+';'
        body=extra or 'console.log(JSON.stringify(api.makeDesign(config)));'
        result=subprocess.run(['node','-e',source+body],check=True,capture_output=True,text=True,encoding='utf-8')
        return json.loads(result.stdout)

    def engineering(self):
        return canonicalize_config(json.loads((self.project/'工程案例_100mmDSV.json').read_text(encoding='utf-8')))

    def model(self,data):
        return Design.from_dict({key:value for key,value in data.items() if key!='preview_only'})

    def test_engineering_preview_fits_exact_cubes_and_printed_module_clearances(self):
        config=self.engineering();data=self.js(config);self.assertTrue(data['preview_only'])
        model=self.model(data);limits=envelope_for(config)
        self.assertTrue(validate_design(model,limits)['valid'])
        self.assertTrue(audit_manufacturing(model,config['manufacturing'],limits)['valid'])
        self.assertEqual(len({r.layer_index for r in model.rings}),16)
        self.assertLessEqual(model.magnet_count,config['manufacturing']['max_magnet_count'])

    def test_default_and_individual_pocket_geometry_pass_actual_obb_audits(self):
        for mode in ['radial_channels','independent_pockets']:
            config=default_config();config['manufacturing']['layout_mode']=mode
            with self.subTest(mode=mode):
                model=self.model(self.js(config));limits=envelope_for(config)
                self.assertTrue(validate_design(model,limits)['valid'])
                self.assertTrue(audit_manufacturing(model,config['manufacturing'],limits)['valid'])

    def test_sister_pairs_keep_polarity_and_actual_center_gap(self):
        config=self.engineering();model=self.model(self.js(config));positive=[r for r in model.rings if r.z_mm>0]
        for ring in positive:
            opposite=[r for r in model.rings if r.slot_index==ring.slot_index and r.z_mm==-ring.z_mm]
            self.assertEqual(len(opposite),1)
            self.assertEqual(opposite[0].radius_mm,ring.radius_mm)
            self.assertEqual(opposite[0].count,ring.count)
            self.assertEqual(opposite[0].polarity,{'out':'in','in':'out','down':'down','up':'up'}[ring.polarity])
        gap=2*min(r.z_mm for r in positive)-model.cube_mm
        self.assertGreaterEqual(gap,config['manufacturing']['min_central_magnet_gap_mm'])

    def test_strict_quantity_limit_and_ring_limit_are_respected(self):
        config=self.engineering();config['manufacturing']['max_magnet_count']=180
        config['search']['max_radial_rings_per_layer']=1
        data=self.js(config);model=self.model(data)
        self.assertLessEqual(model.magnet_count,180)
        self.assertTrue(all(sum(r.layer_index==layer for r in model.rings)==1 for layer in {r.layer_index for r in model.rings}))
        self.assertTrue(validate_design(model,envelope_for(config))['valid'])

    def test_no_source_layout_is_read_and_config_is_never_modified(self):
        config=self.engineering();config['source_geometry']='Z:/does-not-exist/placement-05.json'
        result=self.js(config,"""
const before=JSON.stringify(config),a=api.makeDesign(config),b=api.makeDesign(config);
console.log(JSON.stringify({same:JSON.stringify(a)===JSON.stringify(b),unchanged:before===JSON.stringify(config),
  preview:a.preview_only,metrics:a.metrics,seeds:config.seeds}));
""")
        self.assertTrue(result['same']);self.assertTrue(result['unchanged']);self.assertTrue(result['preview'])
        self.assertNotIn('metrics',result);self.assertNotIn('seeds',result)

    def test_conflicting_envelopes_and_too_small_budget_do_not_fake_a_design(self):
        variants=[]
        config=self.engineering();config['envelope']['max_outer_diameter_mm']=220;variants.append(config)
        config=self.engineering();config['envelope']['max_length_mm']=20;variants.append(config)
        config=self.engineering();config['manufacturing']['max_magnet_count']=7;variants.append(config)
        config=self.engineering();config['magnet']['cube_mm']=-12;variants.append(config)
        config=self.engineering();config['de_novo']['column_counts']=[25];variants.append(config)
        for config in variants:
            with self.subTest(config=config):self.assertIsNone(self.js(config))

    def test_shorter_envelope_reduces_layer_count_without_overlapping(self):
        config=self.engineering();config['envelope']['max_length_mm']=150
        model=self.model(self.js(config))
        self.assertLess(len({r.layer_index for r in model.rings}),16)
        self.assertTrue(validate_design(model,envelope_for(config))['valid'])


if __name__=='__main__':unittest.main()

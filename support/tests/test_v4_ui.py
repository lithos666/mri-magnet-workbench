"""Exercise real frontend form conversion, including automatic/null settings."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed for actual frontend conversion checks')
class WorkbenchEngineeringFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        page = Path(__file__).resolve().parents[2] / 'mri_magnet' / 'workbench.html'
        cls.html = page.read_text(encoding='utf-8')
        cls.functions = '\n'.join(re.search(r'^function '+name+r'\(.*$', cls.html, re.M).group(0)
                                  for name in ('get', 'set', 'fillForm', 'parseFieldValue', 'readForm'))

    def run_frontend(self, code):
        result = subprocess.run(['node', '-e', self.functions+'\n'+code],
                                check=True, capture_output=True, text=True, encoding='utf-8')
        return json.loads(result.stdout)

    def test_automatic_ring_limit_and_optional_ppm_roundtrip_as_null(self):
        result = self.run_frontend("""
const inputs=[
 {type:'number',value:'',dataset:{path:'search.max_radial_rings_per_layer',empty:'null'}},
 {type:'number',value:'',dataset:{path:'target.max_ppm',empty:'null'}},
 {type:'number',value:'10',min:'0.01',max:'',dataset:{path:'scoring.field_gain_scale',factor:'0.01'}}
];
let config={search:{max_radial_rings_per_layer:8},target:{max_ppm:1000},scoring:{field_gain_scale:.2}};
const $$=()=>inputs;
console.log(JSON.stringify(readForm()));
""")
        self.assertIsNone(result['search']['max_radial_rings_per_layer'])
        self.assertIsNone(result['target']['max_ppm'])
        self.assertAlmostEqual(result['scoring']['field_gain_scale'], .1)

    def test_saved_values_render_automatic_blank_and_percent(self):
        result = self.run_frontend("""
const inputs=[
 {type:'number',value:'8',dataset:{path:'search.max_radial_rings_per_layer',empty:'null'}},
 {type:'number',value:'0',dataset:{path:'target.max_ppm',empty:'null'}},
 {type:'number',value:'0',dataset:{path:'scoring.field_gain_scale',factor:'0.01'}},
 {type:'text',value:'',dataset:{path:'manufacturing.machine_bed_mm',type:'array'}}
];
let config={search:{max_radial_rings_per_layer:null},target:{max_ppm:null},scoring:{field_gain_scale:.1},manufacturing:{machine_bed_mm:[255,255,255]}};
const $$=()=>inputs,modeVisibility=()=>{};let editingJson=true;
fillForm();console.log(JSON.stringify(inputs.map(input=>input.value)));
""")
        self.assertEqual(result, ['', '', 10, '255, 255, 255'])

    def test_weight_upper_bound_and_empty_required_printer_array_rejected(self):
        result = self.run_frontend("""
const field={type:'number',value:'1.1',min:'0',max:'1',dataset:{},closest:()=>({childNodes:[{textContent:'场强权重'}]})};
const printer={type:'text',value:'',required:true,dataset:{type:'array'}};
const results=[field,printer].map(input=>{try{parseFieldValue(input);return false}catch(error){return true}});
console.log(JSON.stringify(results));
""")
        self.assertEqual(result, [True, True])

    def test_invalidated_results_clear_old_links_and_preflight_metrics(self):
        functions = '\n'.join(re.search(r'^function '+name+r'\(.*$', self.html, re.M).group(0)
                              for name in ('fileURL', 'artifactPaths', 'renderArtifacts', 'renderPreflight'))
        result = self.run_frontend(functions+"""
const elements={};const $=id=>elements[id]??(elements[id]={innerHTML:'<a href="old.class">旧结果</a>',textContent:'8 环',classList:{toggle:()=>{}}});
let lastArtifactSignature='old',frames=[];
renderArtifacts({});renderPreflight(undefined);
console.log(JSON.stringify({files:$('#artifact-list').innerHTML,layouts:$('#result-layouts').innerHTML,preflight:$('#preflight-metrics').innerHTML,details:$('#preflight-details').textContent}));
""")
        self.assertNotIn('old.class', result['files'])
        self.assertNotIn('old.class', result['layouts'])
        self.assertIn('尚未生成交付物', result['files'])
        self.assertEqual(result['details'], '等待校验')
        self.assertEqual(result['preflight'].count('class="value">—'), 6)


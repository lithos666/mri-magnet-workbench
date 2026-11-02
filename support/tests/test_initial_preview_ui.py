"""Illustration is view-only and never substitutes a selected real design."""
import json
from pathlib import Path
import unittest
from mri_magnet.workbench_config import default_config
import test_v5_3d_ui as existing_ui


class InitialPreviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing_ui.InteractiveMagnetUITests.setUpClass()

    def run_display(self,extra):
        helper=existing_ui.InteractiveMagnetUITests()
        names=['fileURL','prettyNumber','escapeHTML','viewPaths','physicsDraftChanged',
               'activeMagnetSource','candidateLayouts','artifactPaths','seedImage','renderCanvas']
        module=Path(__file__).resolve().parents[2]/'mri_magnet/workbench_preview.js'
        config=default_config()
        body="globalThis.MRIInitialPreview=require("+json.dumps(str(module))+');\n'
        body+='config='+json.dumps(config)+';let timelinePinned=false,frames=[],analysisView="convergence",captured=null;\n'
        body+='let next={config:structuredClone(config),seeds:[],selected_seed_ids:[],artifacts:{}};\n'
        body+='function updateMagnetViewport(source,cfg,active){captured={source,cfg,active};}\n'
        body+=extra
        return helper.js(names,body)

    def test_empty_startup_gets_model_without_inserting_solver_seed_or_metrics(self):
        result=self.run_display('''
const original=JSON.stringify(next);renderCanvas(next);
console.log(JSON.stringify({captured,unchanged:JSON.stringify(next)===original,metrics:$('#canvas-metrics').innerHTML}));
''')
        self.assertTrue(result['unchanged'])
        self.assertTrue(result['captured']['source']['design']['preview_only'])
        self.assertGreater(len(result['captured']['source']['design']['rings']),0)
        self.assertIn('初步磁体示意',result['captured']['source']['label'])
        self.assertNotIn('密集复核</td>',result['metrics'])

    def test_real_design_bypasses_the_illustration_generator(self):
        result=self.run_display('''
const real={rings:[{id:'real',radius_mm:150,z_mm:20,count:28,polarity:'out',phase_deg:0}],cube_mm:12};
next.current_best={design:real};globalThis.MRIInitialPreview={makeDesign(){throw Error('must not generate');}};
renderCanvas(next);console.log(JSON.stringify(captured.source));
''')
        self.assertEqual(result['kind'],'live')
        self.assertEqual(result['design']['rings'][0]['id'],'real')
        self.assertNotIn('preview_only',result['design'])

    def test_missing_explicit_candidate_does_not_show_an_unrelated_illustration(self):
        result=self.run_display('''
previewSelection={path:'runs/missing.png',label:'missing',kind:'candidate'};
renderCanvas(next);console.log(JSON.stringify(captured.source));
''')
        self.assertEqual(result['kind'],'unavailable')
        self.assertIsNone(result['design'])

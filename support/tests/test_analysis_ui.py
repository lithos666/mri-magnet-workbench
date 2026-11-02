"""Execute offline analysis helpers and source wiring without a browser/GPU."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed for actual frontend checks')
class EngineeringAnalysisUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.helper = cls.root / 'mri_magnet/workbench_analysis.js'
        cls.html = (cls.root / 'mri_magnet/workbench.html').read_text(encoding='utf-8')

    def js(self, body):
        bootstrap = 'const a=require('+json.dumps(str(self.helper))+');\n'
        with tempfile.TemporaryDirectory(prefix='mri-analysis-ui-') as folder:
            script = Path(folder) / 'check.js'
            script.write_text(bootstrap + body, encoding='utf-8')
            result = subprocess.run(['node', str(script)], check=True, capture_output=True, text=True, encoding='utf-8')
        return json.loads(result.stdout)

    def test_signed_colormap_is_shared_clamped_and_rejects_invalid_values(self):
        data = self.js("console.log(JSON.stringify([a.ppmColor(-2,1),a.ppmColor(0,1),a.ppmColor(2,1),a.ppmColor(NaN,1),a.ppmColor(1,0)]));")
        self.assertEqual(data, [[40,104,179],[247,249,251],[190,62,66],None,None])

    def test_physical_identity_ignores_labels_and_full_ring_phase_periods(self):
        data = self.js("""
const d={cube_mm:12,remanence_T:1.43,reference_mur:1.05,rings:[{id:'a',radius_mm:120,z_mm:20,count:28,phase_deg:-90,polarity:'out'}]};
const e=JSON.parse(JSON.stringify(d));e.rings[0].id='renamed';e.rings[0].phase_deg+=360/28;
const same=a.physicalSignature(d)===a.physicalSignature(e);e.rings[0].radius_mm+=.001;
console.log(JSON.stringify({same,moved:a.physicalSignature(d)!==a.physicalSignature(e),preview:a.physicalSignature({...d,preview_only:true})}));
""")
        self.assertTrue(data['same'])
        self.assertTrue(data['moved'])
        self.assertEqual(data['preview'], '')

    def test_stale_design_domain_and_material_never_match(self):
        data = self.js("""
const d={cube_mm:12,rings:[{radius_mm:120,z_mm:20,count:28,phase_deg:0,polarity:'out'}]};
const config={target:{geometry:'sphere',diameter_mm:100,z_mm:0},polarization_model:'remanence'};
const report={design:d,domain:{...config.target},polarization_model:'remanence'};
const valid=Boolean(a.matchingAnalysis(report,d,config));
const moved=structuredClone(d);moved.rings[0].z_mm+=1;
console.log(JSON.stringify({valid,moved:a.matchingAnalysis(report,moved,config),domain:a.matchingAnalysis(report,d,{...config,target:{...config.target,diameter_mm:20}}),material:a.matchingAnalysis(report,d,{...config,polarization_model:'isolated_cube_self_demag'}),preview:a.matchingAnalysis(report,{...d,preview_only:true},config)}));
""")
        self.assertTrue(data.pop('valid'))
        self.assertTrue(all(value is None for value in data.values()))

    def test_slice_cells_preserve_engineering_axes_masks_and_fov_plane(self):
        data = self.js("""
const plane=(name,points)=>({name,grid_size:2,points_mm:points,values_ppm:[-1,0,1,2],valid_mask:[true,true,true,true]});
const xy=plane('XY',[[0,0,7],[1,0,7],[0,1,7],[1,1,7]]),xz=plane('XZ',[[0,0,0],[1,0,0],[0,0,1],[1,0,1]]),yz=plane('YZ',[[0,0,0],[0,1,0],[0,0,1],[0,1,1]]);
let report={domain:{geometry:'sphere'},slices:{color_limit_ppm:2,planes:[xy,xz,yz]}};
const all=a.sliceCells(report),disk=a.sliceCells({...report,domain:{geometry:'disk'}});
xy.valid_mask[0]=false;const masked=a.sliceCells(report);
console.log(JSON.stringify({all,disk,masked:masked.map(c=>c.plane)}));
""")
        self.assertEqual([cell['plane'] for cell in data['all']], ['XY','XZ','YZ'])
        self.assertEqual(data['all'][0]['points'], [[0,0,7],[1,0,7],[1,1,7],[0,1,7]])
        self.assertEqual(data['all'][1]['points'][2], [1,0,1])
        self.assertEqual([cell['plane'] for cell in data['disk']], ['XY'])
        self.assertEqual(data['masked'], ['XZ','YZ'])

    def test_cutaway_discards_front_cells_including_cells_touching_clip_plane(self):
        data = self.js("""
const report={domain:{geometry:'sphere'},slices:{color_limit_ppm:1,planes:[{name:'XY',grid_size:2,points_mm:[[0,-1,0],[1,-1,0],[0,0,0],[1,0,0]],values_ppm:[0,0,0,0],valid_mask:[true,true,true,true]}]}};
console.log(JSON.stringify({normal:a.sliceCells(report).length,clipped:a.sliceCells(report,{cutaway:true}).length}));
""")
        self.assertEqual(data, {'normal':1,'clipped':0})

    def test_request_payload_is_explicit_and_never_accepts_initial_preview(self):
        data = self.js("""
const d={rings:[{radius_mm:120,z_mm:20,count:28,phase_deg:0,polarity:'out'}]},source={design:d,label:'real',kind:'candidate'};
let rejected=0;for(const kind of ['illustrative','reference','unavailable'])try{a.requestPayload({...source,kind});}catch{rejected++;}
console.log(JSON.stringify({payload:a.requestPayload(source,{mutual_demag:true,tolerance:false}),rejected}));
""")
        self.assertEqual(data['rejected'], 3)
        self.assertEqual(data['payload']['analysis_options'], {'mutual_demag':True,'tolerance':False})
        self.assertEqual(data['payload']['source_label'], 'real')

    def test_harmonic_bar_labels_escape_content_and_keep_signed_units(self):
        data = self.js("console.log(JSON.stringify(a.harmonicBars({spherical:{zonal:[{degree:2,label:'Z2<bad>',coefficient_ppm:-3.25},{degree:4,label:'Z4',coefficient_ppm:1}]}}))); ")
        self.assertIn('Z2&lt;bad&gt;', data)
        self.assertIn('-3.2500 ppm', data)
        self.assertNotIn('Z2<bad>', data)

    def test_sensitivity_outline_matches_ring_identity_without_changing_poles(self):
        data = self.js("""
const d={rings:[{id:'R1',radius_mm:120,z_mm:20,polarity:'out'},{id:'R2',radius_mm:130,z_mm:30,polarity:'in'}]};
const before=JSON.stringify(d);const report={tolerance:{sensitivity_rings:[{ring_id:'R2',sensitivity_ppm:2},{ring_id:'R1',sensitivity_ppm:1},{ring_id:'unknown',sensitivity_ppm:1}]}};
console.log(JSON.stringify({rings:a.sensitiveRings(report,d),unchanged:JSON.stringify(d)===before}));
""")
        self.assertTrue(data['unchanged'])
        self.assertEqual([row['ring_id'] for row in data['rings']], ['R1','R2'])
        self.assertEqual(data['rings'][1]['color'], [194,76,83])

    def test_tolerance_yield_scope_and_uncoupled_model_are_explicit(self):
        data = self.js("""
const tolerance={available:true,polarization_model:'isolated_cube_self_demag',nominal:{pp_norm_ppm:143.67},
summary:{p50_ppm:956,p90_ppm:1371,yield_fraction:1,yield_scope:'field_only_no_ppm_limit',yield_confidence95:[.98,1]},
scope_note:'fixed-source tolerance; coupled nominal is separate <do not mix>',
assumptions:{remanence_half_range_fraction:.015,sector_translation_half_range_mm:.15},
timings:{precompute_seconds:3.25,sampling_seconds:.12}};
const field=a.toleranceSummaryHTML(tolerance);tolerance.summary.yield_scope='field_and_sampled_ppm';
console.log(JSON.stringify({field,both:a.toleranceSummaryHTML(tolerance)}));
""")
        self.assertIn('仅场强通过率（未设ppm门槛） 100%', data['field'])
        self.assertNotIn('场强及采样PPM通过率', data['field'])
        self.assertIn('场强及采样PPM通过率 100%', data['both'])
        self.assertIn('isolated_cube_self_demag', data['field'])
        self.assertIn('143.67 ppm', data['field'])
        self.assertIn('coupled nominal is separate &lt;do not mix&gt;', data['field'])
        self.assertIn('±1.5%', data['field'])
        self.assertIn('±0.15 mm', data['field'])
        self.assertIn('不是标准差 σ', data['field'])
        self.assertIn('95% Wilson', data['field'])

    def test_analysis_is_a_separate_tab_and_http_is_only_in_explicit_handler(self):
        self.assertIn('data-view="physics"', self.html)
        self.assertIn('id="field-colorbar"', self.html)
        self.assertIn('/assets/workbench_analysis.js', self.html)
        self.assertIn("api('/api/analysis',payload)", self.html)
        poll = re.search(r"async function poll\(\).*", self.html).group(0)
        self.assertNotIn('/api/analysis', poll)
        self.assertIn("if(state.busy||state.export_job?.status==='running'||state.analysis_job?.status==='running')", self.html)

    def test_cli_analysis_input_is_visible_without_fabricating_optimization_results(self):
        names = ['fileURL','artifactPaths','seedImage','candidateLayouts','activeMagnetSource']
        functions = '\n'.join(re.search(r'^function '+name+r'\(.*$', self.html, re.M).group(0) for name in names)
        data = self.js(functions + """
let previewSelection=null,currentStep=1,liveFollowRequested=false;
const design={cube_mm:12,rings:[{id:'ring',radius_mm:135,z_mm:35,count:24,phase_deg:0,polarity:'out'}]};
const config={target:{geometry:'sphere',diameter_mm:20,z_mm:0},polarization_model:'remanence'};
const next={config,analysis:{design,domain:{...config.target},polarization_model:'remanence'}};
const source=activeMagnetSource(next),hasFakeBest=Object.hasOwn(next,'best')||Object.hasOwn(next,'current_best');
const seed=activeMagnetSource({...next,seeds:[{id:'real-seed',design:{...design,name:'seed'}}]});
const best=activeMagnetSource({...next,best:{design:{...design,name:'verified'}}});
const stale=activeMagnetSource({...next,config:{...config,target:{...config.target,diameter_mm:100}}});
const preview=activeMagnetSource({...next,analysis:{...next.analysis,design:{...design,preview_only:true}}});
console.log(JSON.stringify({source,hasFakeBest,seedKind:seed.kind,bestKind:best.kind,staleKind:stale.kind,previewKind:preview.kind,emptyKind:activeMagnetSource({}).kind}));
""")
        self.assertEqual(data['source']['kind'], 'analysis')
        self.assertEqual(data['source']['label'], '工程分析输入 · 非优化结果')
        self.assertEqual(data['source']['design']['rings'][0]['radius_mm'], 135)
        self.assertFalse(data['hasFakeBest'])
        self.assertEqual(data['seedKind'], 'seed')
        self.assertEqual(data['bestKind'], 'verified')
        self.assertEqual(data['staleKind'], 'reference')
        self.assertEqual(data['previewKind'], 'reference')
        self.assertEqual(data['emptyKind'], 'reference')


if __name__ == '__main__':
    unittest.main()

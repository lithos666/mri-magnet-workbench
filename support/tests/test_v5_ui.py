"""Execute the actual browser helpers against minimal DOM behaviour."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'),'Node is required for real frontend helper checks')
class V5WorkbenchUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page=Path(__file__).resolve().parents[2]/'mri_magnet'/'workbench.html'
        cls.html=cls.page.read_text(encoding='utf-8')

    def js(self,names,body):
        functions='\n'.join(re.search(r'^function '+name+r'\(.*$',self.html,re.M).group(0) for name in names)
        stub="""
const elements={};const $=key=>elements[key]??(elements[key]={innerHTML:'',textContent:'',hidden:false,value:'0',max:'0',dataset:{},src:'',disabled:false,className:'',classList:{values:new Set(),toggle(k,v){v?this.values.add(k):this.values.delete(k)}},removeAttribute(k){delete this[k]}});
const $$=()=>[];
"""
        result=subprocess.run(['node','-e',functions+'\n'+stub+body],check=True,capture_output=True,text=True,encoding='utf-8')
        return json.loads(result.stdout)

    def test_state_evolution_frames_show_and_user_frame_survives_poll(self):
        result=self.js(['fileURL','prettyNumber','artifactPaths','timelineFrames','renderTimeline','renderFrame'],"""
let frames=[],timelineRun='',timelinePinned=false,lastTimelineSignature='',state=null,currentView='geometry';const stageNames={de:'DE 演化'};
const next={run_dir:'runs/one',evolution:{frames:[{image:'runs/one/frame_1.png',stage:'de',evaluations:20,mean_mT:42,ppm:200},{image:'runs/one/frame_2.png',stage:'de',evaluations:40,mean_mT:43,ppm:100}]},artifacts:{evolution_gif:'runs/one/evo.gif'}};
renderTimeline(next);const visible=!$('#timeline-area').classList.values.has('hidden');
timelinePinned=true;$('#timeline-slider').value='0';renderTimeline(next);
next.evolution.frames.push({image:'runs/one/frame_3.png',stage:'de',evaluations:60,mean_mT:44,ppm:50});renderTimeline(next);
console.log(JSON.stringify({visible,max:$('#timeline-slider').max,selected:$('#timeline-slider').value,src:$('#timeline-image').src,label:$('#timeline-label').textContent}));
""")
        self.assertTrue(result['visible'])
        self.assertEqual(result['max'],2)
        self.assertEqual(str(result['selected']),'0')
        self.assertIn('frame_1.png',result['src'])
        self.assertIn('求场 20',result['label'])

    def test_frame_selection_resets_only_for_new_run(self):
        result=self.js(['fileURL','prettyNumber','artifactPaths','timelineFrames','renderTimeline','renderFrame'],"""
let frames=[],timelineRun='',timelinePinned=false,lastTimelineSignature='',state=null,currentView='geometry';const stageNames={};
renderTimeline({run_dir:'one',evolution:{frames:['one/a.png','one/b.png']}});
timelinePinned=true;$('#timeline-slider').value='0';renderTimeline({run_dir:'two',evolution:{frames:['two/a.png','two/b.png','two/c.png']}});
console.log(JSON.stringify({pinned:timelinePinned,value:$('#timeline-slider').value,src:$('#timeline-image').src}));
""")
        self.assertFalse(result['pinned']);self.assertEqual(result['value'],2)
        self.assertIn('two/c.png',result['src'])

    def test_legacy_artifact_frames_are_still_supported(self):
        result=self.js(['fileURL','artifactPaths','timelineFrames'],"""
console.log(JSON.stringify(timelineFrames({artifacts:{frames:[{image:'runs/old/f1.png'}]}})));
""")
        self.assertEqual(result[0]['path'],'runs/old/f1.png')

    def test_old_flat_image_artifacts_are_recovered_without_frames_metadata(self):
        result=self.js(['fileURL','artifactPaths','timelineFrames','candidateLayouts'],"""
console.log(JSON.stringify({frames:timelineFrames({artifacts:{one:'runs/a/演化过程/frame_0001.png'}}),layouts:candidateLayouts({artifacts:{layout:'runs/a/磁铁布局_RZ.png'}})}));
""")
        self.assertEqual(len(result['frames']),1)
        self.assertEqual(len(result['layouts']),1)

    def test_budget_panel_displays_actual_dimensions_and_skip_warning(self):
        result=self.js(['get','escapeHTML','prettyNumber','budgetValue','renderSolverBudget'],"""
const styleNames={};renderSolverBudget({valid:false,policy:'strict',violations:['岛1：72维LP至少74次，当前70次'],warnings:[],per_island:[{island:1,D:72,coordinate_nominal_sweeps:.4,lp_affordable_passes:0,lp_target_passes:3,de_generations_nominal:5,lp_will_skip:true}]});
console.log(JSON.stringify({status:$('#solver-budget-status').textContent,table:$('#solver-budget-table').innerHTML,notes:$('#solver-budget-notes').innerHTML}));
""")
        self.assertIn('预算冲突',result['status'])
        self.assertIn('72',result['table']);self.assertIn('0.4 轮',result['table'])
        self.assertIn('至少74次',result['notes'])

    def test_nominal_lp_start_is_not_confused_with_backtracking_reserve(self):
        result=self.js(['get','escapeHTML','prettyNumber','budgetValue','renderSolverBudget'],"""
const styleNames={};renderSolverBudget({valid:false,violations:[],warnings:[],per_island:[{island:1,D:68,coordinate_nominal_sweeps:1,lp_nominal_passes_without_backtracking:1,lp_affordable_passes:0,lp_target_passes:3,de_generations_nominal:2,lp_will_skip:false}]});
console.log(JSON.stringify($('#solver-budget-table').innerHTML));
""")
        self.assertIn('1 / 3',result)
        self.assertIn('含回溯余量 0',result)
        self.assertNotIn('将跳过',result)

    def test_range_max_is_updated_before_setting_new_latest_frame(self):
        result=self.js(['fileURL','prettyNumber','artifactPaths','timelineFrames','renderTimeline','renderFrame'],"""
let frames=[],timelineRun='',timelinePinned=false,lastTimelineSignature='',state=null,currentView='geometry';const stageNames={};
const slider=$('#timeline-slider');let current=0,maximum=0;Object.defineProperties(slider,{max:{get:()=>maximum,set:v=>maximum=Number(v)},value:{get:()=>current,set:v=>current=Math.max(0,Math.min(maximum,Number(v)))}});
renderTimeline({run_dir:'one',evolution:{frames:['one/a.png','one/b.png','one/c.png']}});
console.log(JSON.stringify({index:slider.value,max:slider.max,src:$('#timeline-image').src}));
""")
        self.assertEqual(result['index'],2)
        self.assertEqual(result['max'],2)
        self.assertIn('one/c.png',result['src'])

    def test_candidate_and_seed_clicks_preview_without_checkbox_side_effect(self):
        result=self.js(['handleLayoutPreview'],"""
const picked=[];function selectLayoutPreview(...args){picked.push(args)};
const card={dataset:{previewLayout:'runs/a.png',previewLabel:'最低PPM',previewKind:'candidate',previewId:'a'},hasAttribute:()=>true};
let prevented=0;
handleLayoutPreview({type:'click',target:{closest:selector=>selector==='[data-preview-layout]'?card:card},preventDefault(){prevented++}});
const input={hasAttribute:()=>false};handleLayoutPreview({type:'click',target:{closest:selector=>selector==='[data-preview-layout]'?card:input},preventDefault(){prevented++}});
console.log(JSON.stringify({picked,prevented}));
""")
        self.assertEqual(result['prevented'],1);self.assertEqual(len(result['picked']),1)
        self.assertEqual(result['picked'][0],['runs/a.png','最低PPM','a','candidate'])

    def test_selected_layout_and_pareto_are_used_by_main_canvas(self):
        result=self.js(['fileURL','viewPaths'],"""
let currentStep=6,previewSelection={path:'runs/low.png'},timelinePinned=false,frames=[];
const paths=viewPaths({artifacts:{layout:'runs/main.png',pareto:'runs/pareto.png'}});
console.log(JSON.stringify(paths));
""")
        self.assertEqual(result['geometry'],'runs/low.png')
        self.assertEqual(result['pareto'],'runs/pareto.png')

    def test_candidate_grid_includes_all_backend_layouts_and_deduplicates_roles(self):
        result=self.js(['fileURL','artifactPaths','candidateLayouts'],"""
console.log(JSON.stringify(candidateLayouts({candidate_layouts:[{path:'runs/a/磁铁布局_RZ.png',label:'A'},{path:'runs/b/磁铁布局_RZ.png',label:'B'}],pareto_recommendations:[{layout_file:'runs/a/磁铁布局_RZ.png',label:'均衡'}],artifacts:{layout:'runs/a/磁铁布局_RZ.png'}})));
""")
        self.assertEqual(len(result),2)
        self.assertEqual({r['path'] for r in result},{'runs/a/磁铁布局_RZ.png','runs/b/磁铁布局_RZ.png'})

    def test_pipeline_uses_real_island_statuses_and_configured_disabled_stages(self):
        result=self.js(['escapeHTML','prettyNumber','renderIslands'],"""
let islandRun='',recordedStages=new Set();const styleNames={wide:'宽展型'},stageNames={de:'DE 演化',running:'运行中',promoted:'已晋级',eliminated:'已淘汰'};
renderIslands({run_dir:'runs/a',busy:true,job:{step:5,stage:'de'},config:{solver:{local_refine_evaluations:0,active_set_evaluations:240}},island_statuses:[{island:1,name:'wide',status:'running',stage:'de',round:2,evaluations:200,best_mean_mT:45,best_ppm:123,refinement:{lp_passes:3,lp_accepted_steps:3,coordinate_axes_visited:30}},{island:2,name:'wide',status:'eliminated',round:1,evaluations:100}]});
console.log(JSON.stringify({pipeline:$('#phase-pipeline').innerHTML,islands:$('#island-statuses').innerHTML}));
""")
        self.assertIn('phase-chip current',result['pipeline'])
        self.assertIn('已关闭',result['pipeline'])
        self.assertIn('已淘汰',result['islands']);self.assertIn('123 ppm',result['islands'])
        self.assertIn('实际 LP 3 轮 / 接受 3 步',result['islands'])

    def test_visible_form_has_unique_v4_controls_and_presets(self):
        # Alongside actual JS conversion tests, check real controls are reachable.
        for path in ('de_novo.candidate_pool_size','solver.population_size','solver.local_refine_evaluations','solver.active_set_evaluations','solver.budget_policy','manufacturing.allow_split_pocket_seams'):
            self.assertEqual(self.html.count('data-path="'+path+'"'),1)
        self.assertIn('data-preset="engineering_100mm"',self.html)
        self.assertIn('data-preset="default_20mm"',self.html)
        self.assertIn('data-view="pareto"',self.html)

if __name__=='__main__':unittest.main()

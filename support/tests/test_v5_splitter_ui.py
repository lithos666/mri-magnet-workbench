"""Execute pane sizing and renderer readiness behaviour from the real page."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed for frontend behaviour checks')
class WorkbenchSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html=(Path(__file__).resolve().parents[2]/'mri_magnet/workbench.html').read_text(encoding='utf-8')

    def js(self,names,body):
        functions='\n'.join(re.search(r'^function '+name+r'\(.*$',self.html,re.M).group(0) for name in names)
        stub="""
const elements={},styles={},attributes={},stored=new Map(),frames=[],classes=new Set();let width=1000,resizes=0;
const splitter={setAttribute(k,v){attributes[k]=v},releasePointerCapture(){},focus(){}};
elements['#workspace-grid']={getBoundingClientRect:()=>({left:100,width}),style:{setProperty(k,v){styles[k]=v}}};
elements['#workspace-splitter']=splitter;
const $=k=>elements[k]??(elements[k]={textContent:'',value:'0'}),$$=()=>[];
const localStorage={getItem:k=>stored.has(k)?stored.get(k):null,setItem(k,v){stored.set(k,v)}};
const document={body:{classList:{contains:k=>classes.has(k),add:k=>classes.add(k),remove:k=>classes.delete(k)}}};
const window={innerWidth:1600,magnetViewer:{resize(){resizes++}}};
const requestAnimationFrame=f=>{frames.push(f);return frames.length};
const WORKSPACE_SPLIT_KEY='mri-workbench-v5-panel-share',WORKSPACE_SPLIT_DEFAULT=.35;
let workspacePreferredShare=.35,workspaceSplitDrag=null,workspaceSplitFrame=0;
"""
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'split.js';script.write_text(stub+functions+'\n'+body,encoding='utf-8')
            result=subprocess.run(['node',str(script)],check=True,capture_output=True,text=True,encoding='utf-8')
        return json.loads(result.stdout)

    @property
    def sizing(self):
        return ['workspaceSplitLimits','clampWorkspaceSplit','storeWorkspaceSplit','applyWorkspaceSplit']

    def test_dragging_commits_preference_and_keeps_both_panels_usable(self):
        result=self.js(self.sizing+['setWorkspaceSplitAt'],"""
const left=setWorkspaceSplitAt(-500,true)*workspaceSplitLimits(width).available;
const right=(1-setWorkspaceSplitAt(5000,true))*workspaceSplitLimits(width).available;
frames.splice(0).forEach(f=>f());
console.log(JSON.stringify({left,right,share:workspacePreferredShare,stored:Number(stored.get(WORKSPACE_SPLIT_KEY)),attributes,resizes}));
""")
        self.assertAlmostEqual(result['left'],270)
        self.assertAlmostEqual(result['right'],320)
        self.assertEqual(result['share'],result['stored'])
        self.assertEqual(result['resizes'],1)
        self.assertIn('画布',result['attributes']['aria-valuetext'])

    def test_resizing_does_not_replace_the_users_saved_proportion(self):
        result=self.js(self.sizing,"""
width=1700;applyWorkspaceSplit(.26,true);
width=700;const constrained=applyWorkspaceSplit()*workspaceSplitLimits(width).available,cssShare=Number(styles['--parameter-share']);
width=2000;const restored=applyWorkspaceSplit();
console.log(JSON.stringify({constrained,restored,cssShare,preferred:workspacePreferredShare,saved:Number(stored.get(WORKSPACE_SPLIT_KEY))}));
""")
        self.assertAlmostEqual(result['constrained'],270)
        self.assertAlmostEqual(result['restored'],.26,places=5)
        self.assertEqual(result['cssShare'],.26)
        self.assertEqual(result['preferred'],.26)
        self.assertEqual(result['saved'],.26)

    def test_keyboard_arrows_and_home_use_same_size_and_persistence_rules(self):
        result=self.js(self.sizing+['workspaceSplitEnabled','resetWorkspaceSplit','moveWorkspaceSplitByKey'],"""
const start=applyWorkspaceSplit()*workspaceSplitLimits(width).available;
moveWorkspaceSplitByKey('ArrowRight');const regular=Number(styles['--parameter-share'])*workspaceSplitLimits(width).available;
moveWorkspaceSplitByKey('ArrowLeft',true);const shifted=Number(styles['--parameter-share'])*workspaceSplitLimits(width).available;
moveWorkspaceSplitByKey('Home');const reset=workspacePreferredShare;
window.innerWidth=760;const narrowIgnored=moveWorkspaceSplitByKey('ArrowRight');
console.log(JSON.stringify({start,regular,shifted,reset,narrowIgnored,stored:Number(stored.get(WORKSPACE_SPLIT_KEY))}));
""")
        self.assertAlmostEqual(result['regular']-result['start'],12,places=3)
        self.assertAlmostEqual(result['regular']-result['shifted'],32,places=3)
        self.assertEqual(result['reset'],.35)
        self.assertEqual(result['stored'],.35)
        self.assertFalse(result['narrowIgnored'])

    def test_cancelled_pointer_drag_rolls_back_without_saving_partial_state(self):
        result=self.js(self.sizing+['setWorkspaceSplitAt','finishWorkspaceSplit'],"""
applyWorkspaceSplit(.35,true);workspaceSplitDrag={pointerId:7,initialShare:.35};classes.add('split-resizing');
setWorkspaceSplitAt(700);finishWorkspaceSplit({pointerId:7,clientX:700},true);
console.log(JSON.stringify({share:workspacePreferredShare,saved:Number(stored.get(WORKSPACE_SPLIT_KEY)),left:Number(styles['--parameter-share'])*workspaceSplitLimits(width).available,drag:workspaceSplitDrag,resizing:classes.has('split-resizing')}));
""")
        self.assertEqual(result['share'],.35)
        self.assertEqual(result['saved'],.35)
        self.assertAlmostEqual(result['left'],(1000-14)*.35,places=3)
        self.assertIsNone(result['drag'])
        self.assertFalse(result['resizing'])

    def test_invalid_or_blocked_storage_never_creates_nan_geometry(self):
        result=self.js(['readWorkspaceSplit'],"""
const values=[null,'NaN','Infinity','0','1','{}','.42'],answers=[];
for(const value of values){if(value===null)stored.delete(WORKSPACE_SPLIT_KEY);else stored.set(WORKSPACE_SPLIT_KEY,value);answers.push(readWorkspaceSplit());}
localStorage.getItem=()=>{throw Error('blocked')};answers.push(readWorkspaceSplit());
console.log(JSON.stringify(answers));
""")
        self.assertEqual(result,[.35,.35,.35,.35,.35,.35,.42,.35])

    def test_expanded_mode_disables_split_keyboard_without_losing_preference(self):
        result=self.js(self.sizing+['workspaceSplitEnabled','resetWorkspaceSplit','moveWorkspaceSplitByKey'],"""
applyWorkspaceSplit(.29,true);classes.add('canvas-expanded');const moved=moveWorkspaceSplitByKey('ArrowRight');
classes.delete('canvas-expanded');applyWorkspaceSplit();
console.log(JSON.stringify({moved,preferred:workspacePreferredShare,saved:Number(stored.get(WORKSPACE_SPLIT_KEY))}));
""")
        self.assertFalse(result['moved'])
        self.assertEqual(result['preferred'],.29)
        self.assertEqual(result['saved'],.29)

    def renderer(self,body):
        return self.js(['updateMagnetViewport'],"""
window.magnetViewer=null;let creations=0,nativeCreations=0,lastModelSignature='',rendererInitializationFailed=false,studioRendererReady=false;
let previewSelection=null;const inspectMagnet=()=>{},source={label:'candidate',design:{rings:[{radius_mm:140}]}},viewConfig={target:{diameter_mm:100}};
class Native {constructor(){nativeCreations++;}setOptions(v){this.options=v;}setActive(v){this.active=v;}update(v){this.model=v;}resize(){this.resizes=(this.resizes||0)+1;}}
window.MRIMagnet3D={Viewer:Native};let StudioViewerConstructor=class extends Native{constructor(){super();creations++;}};
"""+body)

    def test_renderer_waits_for_local_module_before_acquiring_any_canvas_context(self):
        result=self.renderer("""
updateMagnetViewport(source,viewConfig,true);const pending={creations,nativeCreations,viewer:!!window.magnetViewer,status:$('#model-render-status').textContent};
studioRendererReady=true;updateMagnetViewport(source,viewConfig,true);
console.log(JSON.stringify({pending,creations,nativeCreations,model:window.magnetViewer.model,active:window.magnetViewer.active}));
""")
        self.assertEqual(result['pending']['creations'],0)
        self.assertEqual(result['pending']['nativeCreations'],0)
        self.assertFalse(result['pending']['viewer'])
        self.assertIn('准备',result['pending']['status'])
        self.assertEqual(result['creations'],1)
        self.assertEqual(result['model']['design']['rings'][0]['radius_mm'],140)
        self.assertTrue(result['active'])

    def test_unavailable_studio_retains_local_compatible_renderer(self):
        result=self.renderer("""
studioRendererReady=true;StudioViewerConstructor=class{constructor(){throw Error('GPU unavailable')}};
updateMagnetViewport(source,viewConfig,true);
console.log(JSON.stringify({nativeCreations,failed:rendererInitializationFailed,model:window.magnetViewer.model}));
""")
        self.assertEqual(result['nativeCreations'],1)
        self.assertFalse(result['failed'])
        self.assertEqual(result['model']['config']['target']['diameter_mm'],100)


if __name__=='__main__':unittest.main()

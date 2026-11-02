"""Buffer lifetime regressions without relying on a graphics driver."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed for actual viewport methods')
class ViewportResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = Path(__file__).resolve().parents[2] / 'mri_magnet'

    def js(self, body):
        preamble = '''
import {StudioViewer} from %s;
await import(%s);
const NativeViewer=globalThis.MRIMagnet3D.Viewer;
globalThis.window={devicePixelRatio:1};globalThis.devicePixelRatio=1;
const callbacks=new Map(),events=[];let sequence=0,tick=0;
globalThis.requestAnimationFrame=fn=>{const id=++sequence;callbacks.set(id,fn);return id;};
globalThis.cancelAnimationFrame=id=>callbacks.delete(id);
function flush(){const pending=[...callbacks];callbacks.clear();tick++;for(const [id,fn]of pending)fn(tick*16);}
function surface(name,rect,w=640,h=480){
 let width=w,height=h;const value={style:{},offsetLeft:0,offsetTop:0,getBoundingClientRect:()=>({...rect})};
 Object.defineProperties(value,{
  width:{get:()=>width,set:n=>{width=n;events.push([name+'.width',n,tick]);if(name==='canvas')value.frameColor='black';}},
  height:{get:()=>height,set:n=>{height=n;events.push([name+'.height',n,tick]);if(name==='canvas')value.frameColor='black';}}
 });value.frameColor='white';return value;
}
function fixture(native=false){
 const rect={width:640,height:480};const v=Object.create((native?NativeViewer:StudioViewer).prototype);
 v.canvas=surface('canvas',rect);v.overlay=surface('overlay',rect);v.drawCanvas=null;
 v.active=true;v.disposed=false;v._fallback=null;v._frame=null;v._transition=null;
 v._pendingSize=null;v._appliedSize={width:640,height:480,dpr:1};v.width=640;v.height=480;v.dpr=1;
 v.bufferResizeCount=0;v._hasFit=false;v.frameCount=0;v.geometryDirty=false;if(native)v.gl=null;
 v.renderer={
  setPixelRatio(){throw Error('Pixel ratio setter would allocate an extra buffer');},
  setSize(){throw Error('Size setter would allocate an extra buffer');},
  setDrawingBufferSize(w,h,d){events.push(['allocate',w,h,d,tick]);v.canvas.width=Math.floor(w*d);v.canvas.height=Math.floor(h*d);},
  render(){events.push(['draw',tick]);v.canvas.frameColor='white';}
 };
 v.camera={aspect:640/480,updateProjectionMatrix(){events.push(['projection',tick]);}};
 v.controls={update(){return false;}};v.scene={};
 v._drawOverlay=()=>events.push(['overlay.draw',tick]);
 v._draw2D=()=>{events.push(['native.draw',tick]);v.canvas.frameColor='white';};
 v._fit=()=>events.push(['fit',tick]);
 return {v,rect};
}
''' % (json.dumps((self.package / 'workbench_studio3d.js').as_uri()),
       json.dumps((self.package / 'workbench_3d.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', preamble + body],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_resize_request_keeps_the_previous_image_until_the_next_draw(self):
        result = self.js('''
const {v,rect}=fixture();rect.width=900;v.resize();
console.log(JSON.stringify({events,color:v.canvas.frameColor,width:v.canvas.width,pending:callbacks.size}));
''')
        self.assertEqual(result['events'], [])
        self.assertEqual(result['color'], 'white')
        self.assertEqual(result['width'], 640)
        self.assertEqual(result['pending'], 1)

    def test_multiple_drag_sizes_apply_only_the_latest_and_draw_in_same_frame(self):
        result = self.js('''
const {v,rect}=fixture();for(const width of [900,1000,1100,950]){rect.width=width;v.resize();}
const before=events.slice();flush();
console.log(JSON.stringify({before,events,color:v.canvas.frameColor,width:v.canvas.width,count:v.bufferResizeCount,frames:v.frameCount}));
''')
        self.assertEqual(result['before'], [])
        allocations = [event for event in result['events'] if event[0] == 'allocate']
        self.assertEqual(allocations, [['allocate', 950, 480, 1, 1]])
        draw = next(i for i, event in enumerate(result['events']) if event[0] == 'draw')
        self.assertLess(result['events'].index(allocations[0]), draw)
        self.assertEqual(result['events'][draw], ['draw', 1])
        self.assertEqual(result['color'], 'white')
        self.assertEqual(result['width'], 950)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['frames'], 1)

    def test_polling_an_unchanged_size_does_not_clear_any_canvas(self):
        result = self.js('''
const {v}=fixture();for(let i=0;i<8;i++){v.setActive(true);v.resize();flush();}
console.log(JSON.stringify({writes:events.filter(e=>e[0]==='allocate'||e[0].endsWith('.width')||e[0].endsWith('.height')),color:v.canvas.frameColor}));
''')
        self.assertEqual(result['writes'], [])
        self.assertEqual(result['color'], 'white')

    def test_pixel_ratio_change_allocates_once_and_redraws(self):
        result = self.js('''
const {v}=fixture();window.devicePixelRatio=1.5;v.resize();v.resize();const before=events.slice();flush();
console.log(JSON.stringify({before,events,w:v.canvas.width,h:v.canvas.height,dpr:v.dpr,count:v.bufferResizeCount,color:v.canvas.frameColor}));
''')
        self.assertEqual(result['before'], [])
        self.assertEqual([e for e in result['events'] if e[0] == 'allocate'],
                         [['allocate', 640, 480, 1.5, 1]])
        self.assertEqual((result['w'], result['h'], result['dpr']), (960, 720, 1.5))
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['color'], 'white')

    def test_fractional_css_changes_keep_equal_pixel_buffers_and_overlay(self):
        result = self.js('''
const {v,rect}=fixture();rect.width=640.2;v.resize();flush();
console.log(JSON.stringify({events,width:v.width,aspect:v.camera.aspect,color:v.canvas.frameColor}));
''')
        self.assertFalse(any(e[0] == 'allocate' or e[0].endswith(('.width', '.height'))
                             for e in result['events']))
        self.assertEqual(result['width'], 640.2)
        self.assertAlmostEqual(result['aspect'], 640.2 / 480)
        self.assertEqual(result['color'], 'white')

    def test_reopening_hidden_view_draws_again_without_reallocating(self):
        result = self.js('''
const {v}=fixture();v.invalidate();v.setActive(false);const hidden=callbacks.size;
v.setActive(true);const shown=callbacks.size;flush();
console.log(JSON.stringify({hidden,shown,events,frames:v.frameCount}));
''')
        self.assertEqual(result['hidden'], 0)
        self.assertEqual(result['shown'], 1)
        self.assertTrue(any(e[0] == 'draw' for e in result['events']))
        self.assertFalse(any(e[0] == 'allocate' for e in result['events']))
        self.assertEqual(result['frames'], 1)

    def test_size_queued_while_hidden_is_applied_when_visible(self):
        result = self.js('''
const {v,rect}=fixture();v.setActive(false);rect.width=840;v.resize();const hidden={pending:callbacks.size,color:v.canvas.frameColor};
v.setActive(true);flush();console.log(JSON.stringify({hidden,events,width:v.canvas.width,color:v.canvas.frameColor}));
''')
        self.assertEqual(result['hidden'], {'pending': 0, 'color': 'white'})
        self.assertEqual(result['width'], 840)
        self.assertEqual(len([e for e in result['events'] if e[0] == 'allocate']), 1)
        self.assertEqual(result['color'], 'white')

    def test_native_compatible_canvas_coalesces_resize_and_draw(self):
        result = self.js('''
const {v,rect}=fixture(true);for(const width of [720,850,960]){rect.width=width;v.resize();}
const before=events.slice();flush();
console.log(JSON.stringify({before,events,width:v.canvas.width,color:v.canvas.frameColor,frames:v.frameCount}));
''')
        self.assertEqual(result['before'], [])
        self.assertEqual([e for e in result['events'] if e[0] == 'canvas.width'],
                         [['canvas.width', 960, 1]])
        self.assertIn(['native.draw', 1], result['events'])
        self.assertEqual(result['width'], 960)
        self.assertEqual(result['color'], 'white')
        self.assertEqual(result['frames'], 1)


if __name__ == '__main__':
    unittest.main()

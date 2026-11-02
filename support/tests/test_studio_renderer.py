"""Product-studio display geometry and camera fit preserve engineering data."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for the local studio geometry checks')
class StudioRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package=Path(__file__).resolve().parents[2]/'mri_magnet'

    def js(self,body):
        package=self.package
        preamble='''
import * as THREE from %s;
import {StudioViewer,engineeringToWorld,displayCenter,makePoleMaterial,poleHalfSpec} from %s;
await import(%s);
const native=globalThis.MRIMagnet3D;
globalThis.window={MRIMagnet3D:native};
const ring=(extra={})=>({id:'R01-L1',layer_index:0,slot_index:0,radius_mm:116,z_mm:80,count:4,polarity:'out',phase_deg:0,...extra});
function fixture(rings=[ring(),ring({id:'R02-L1',z_mm:-80,polarity:'in'})]){
 const v=Object.create(StudioViewer.prototype);v.magnets=native.buildMagnets({cube_mm:12,rings});v.displayMagnets=v.magnets;
 v.config={envelope:{protected_bore_diameter_mm:200,max_outer_diameter_mm:440,max_length_mm:500}};
 v.options={cutaway:false,explode:0};v._homeDistance=1000;
 v.camera=new THREE.PerspectiveCamera(37,1.8,.1,50000);v.camera.up.set(0,0,1);v.camera.position.set(800,-1100,650);
 v.controls={target:new THREE.Vector3(),update(){v.camera.lookAt(this.target);v.camera.updateMatrixWorld();}};
 return v;
}
'''%(json.dumps((package/'vendor/three.module.js').as_uri()),
      json.dumps((package/'workbench_studio3d.js').as_uri()),
      json.dumps((package/'workbench_3d.js').as_uri()))
        result=subprocess.run(['node','--input-type=module','-e',preamble+'\n'+body],
            check=True,capture_output=True,text=True,encoding='utf-8')
        return json.loads(result.stdout)

    def test_horizontal_presentation_is_a_rigid_rotation_of_engineering_axes(self):
        result=self.js("console.log(JSON.stringify([[10,20,30],[1,0,0],[0,1,0],[0,0,1]].map(engineeringToWorld)));")
        expected=[[10,30,-20],[1,0,0],[0,0,-1],[0,1,0]]
        for actual,target in zip(result,expected):
            for a,b in zip(actual,target):self.assertAlmostEqual(a,b,places=10)

    def test_ns_materials_match_the_mri_ring_reference(self):
        result=self.js("console.log(JSON.stringify(['north','south'].map(p=>{const m=makePoleMaterial(p);return {color:m.color.getHexString(),metalness:m.metalness,roughness:m.roughness}})));")
        self.assertEqual([row['color'] for row in result],['e03d34','0071e3'])
        for row in result:
            self.assertEqual(row['metalness'],.12)
            self.assertEqual(row['roughness'],.4)

    def test_red_half_points_north_and_two_halves_fill_each_exact_cube(self):
        result=self.js("""
const rows=[];for(const polarity of ['out','in','down','up']){
 const magnets=native.buildMagnets({cube_mm:12,rings:[ring({polarity,count:7,phase_deg:13.5})]});
 for(const m of magnets){const before=JSON.stringify(m),n=poleHalfSpec(m,'north'),s=poleHalfSpec(m,'south');
 const delta=n.center.map((x,i)=>x-s.center[i]);
 rows.push({dot:delta.reduce((v,x,i)=>v+x*m.north_vector[i],0),mid:n.center.map((x,i)=>(x+s.center[i])/2),expected:m.position_mm,
  volume:2*n.scale.reduce((v,x)=>v*x,1),scale:n.scale,unchanged:before===JSON.stringify(m)});
 }
}console.log(JSON.stringify(rows));
""")
        self.assertEqual(len(result),28)
        for row in result:
            self.assertAlmostEqual(row['dot'],6)
            for a,b in zip(row['mid'],row['expected']):self.assertAlmostEqual(a,b)
            self.assertEqual(row['volume'],12**3)
            self.assertEqual(sorted(row['scale']),[6,12,12])
            self.assertTrue(row['unchanged'])

    def test_ns_split_tracks_interpolated_radial_orientation(self):
        result=self.js("""
const m=native.buildMagnets({cube_mm:12,rings:[ring()]})[0];m.theta=Math.PI/2;
const n=poleHalfSpec(m,'north');console.log(JSON.stringify(n.center.map((x,i)=>x-m.position_mm[i])));
""")
        self.assertAlmostEqual(result[0],0)
        self.assertAlmostEqual(result[1],3)
        self.assertAlmostEqual(result[2],0)

    def test_explosion_is_display_only_and_does_not_rewrite_positions_or_radius(self):
        result=self.js('''
const magnet=native.buildMagnets({cube_mm:12,rings:[ring()]})[0],original=JSON.stringify(magnet);
const positions=[0,1].map(e=>displayCenter(magnet,e));
console.log(JSON.stringify({positions,unchanged:JSON.stringify(magnet)===original,source:magnet.position_mm,bottom:magnet.bottom_radius_mm}));
''')
        self.assertTrue(result['unchanged']);self.assertEqual(result['source'],[116,0,80])
        self.assertEqual(result['positions'][0],[116,0,80])
        self.assertAlmostEqual(result['positions'][1][0],116*1.32)
        self.assertEqual(result['positions'][1][2],140)
        self.assertEqual(result['bottom'],110)

    def test_cutaway_uses_original_engineering_y_after_horizontal_presentation(self):
        result=self.js('''
const v=fixture();v.options.cutaway=true;
console.log(JSON.stringify({all:v.magnets.length,visible:v._visibleMagnets().map(m=>m.position_mm)}));
''')
        self.assertEqual(result['all'],8);self.assertEqual(len(result['visible']),6)
        self.assertTrue(all(point[1]>=-1e-8 for point in result['visible']))

    def test_model_fit_contains_every_exact_cube_corner_at_multiple_aspects(self):
        result=self.js('''
const checks=[];for(const aspect of [.65,1,1.8,3.2])for(const view of ['iso','top','side']){
 const v=fixture();v.camera.aspect=aspect;v.camera.updateProjectionMatrix();v._fit(view);
 let max=0;for(const m of v.magnets)for(const p of native.cubeVertices(m)){const q=new THREE.Vector3(...engineeringToWorld(p)).project(v.camera);max=Math.max(max,Math.abs(q.x),Math.abs(q.y));}
 checks.push({aspect,view,max});
}console.log(JSON.stringify(checks));
''')
        self.assertEqual(len(result),12)
        for check in result:self.assertLess(check['max'],.96,check)

    def test_resize_refit_preserves_user_zoom_and_pan(self):
        result=self.js('''
const v=fixture();v._fit('iso');
v.camera.position.sub(v.controls.target).multiplyScalar(.5).add(v.controls.target);v.controls.target.add(new THREE.Vector3(10,-20,5));v.camera.position.add(new THREE.Vector3(10,-20,5));v.controls.update();
const before={zoom:v._homeDistance/v.camera.position.distanceTo(v.controls.target),target:v.controls.target.toArray()};
v.camera.aspect=3;v.camera.updateProjectionMatrix();v._fit(null,true);
console.log(JSON.stringify({before,after:{zoom:v._homeDistance/v.camera.position.distanceTo(v.controls.target),target:v.controls.target.toArray()}}));
''')
        self.assertAlmostEqual(result['before']['zoom'],2)
        self.assertAlmostEqual(result['after']['zoom'],2)
        self.assertEqual(result['before']['target'],result['after']['target'])

    def test_empty_project_uses_finite_constraint_bounds_and_valid_camera_fit(self):
        result=self.js('''
const v=fixture([]),box=v._worldBounds();v._fit('iso');
console.log(JSON.stringify({min:box.min.toArray(),max:box.max.toArray(),distance:v.camera.position.length(),near:v.camera.near,far:v.camera.far}));
''')
        for actual,expected in zip(result['min'],[-220,-250,-220]):self.assertAlmostEqual(actual,expected,places=10)
        for actual,expected in zip(result['max'],[220,250,220]):self.assertAlmostEqual(actual,expected,places=10)
        self.assertGreater(result['distance'],0);self.assertGreater(result['near'],0)
        self.assertGreater(result['far'],result['distance'])

    def test_first_synthesized_model_refits_actual_bounds_after_reference_preview(self):
        result=self.js('''
const v=fixture([]);v._fit('iso');v._hasFit=true;v.label='reference';v.reducedMotion=true;
for(const method of ['_rebuildInstances','_rebuildReferences','_updateMatrices','_notify','_status','invalidate'])v[method]=()=>{};
const calls=[],fit=v._fit.bind(v);v._fit=(view,preserve)=>{calls.push({view,preserve});fit(view,preserve)};
v.update({design:{cube_mm:12,rings:[ring()]},config:v.config,label:'new seed'});
console.log(JSON.stringify({calls,magnetCount:v.magnets.length,center:engineeringToWorld(v.magnets[0].position_mm)}));
''')
        self.assertEqual(result['magnetCount'],4)
        self.assertEqual(result['calls'],[{'view':None,'preserve':True}])


if __name__=='__main__':unittest.main()

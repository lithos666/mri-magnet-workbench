"""Geometry/picking contract tests for the actual offline JavaScript renderer."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for native viewer geometry checks')
class WorkbenchNative3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Path(__file__).resolve().parents[2] / 'mri_magnet/workbench_3d.js'

    def js(self, body):
        setup = 'const api=require(' + json.dumps(str(self.renderer)) + ');\n'
        setup += "const ring=(extra={})=>({id:'R01-L1',layer_index:0,slot_index:0,radius_mm:150,z_mm:27,count:4,polarity:'out',phase_deg:0,...extra});\n"
        result = subprocess.run(['node', '-e', setup + body], capture_output=True,
                                text=True, encoding='utf-8', check=True)
        return json.loads(result.stdout)

    def assertVectorAlmostEqual(self, actual, expected, places=8):
        self.assertEqual(len(actual), len(expected))
        for value, ref in zip(actual, expected):
            self.assertAlmostEqual(value, ref, places=places)

    def test_positions_bottom_radius_and_exact_four_palette_match_engineering_design(self):
        result = self.js("""
const magnets=api.buildMagnets({cube_mm:12,rings:[ring({phase_deg:30})]});
console.log(JSON.stringify({first:magnets[0],second:magnets[1],colors:api.colors}));
""")
        self.assertVectorAlmostEqual(result['first']['position_mm'], [129.9038105676658, 75, 27])
        self.assertVectorAlmostEqual(result['second']['position_mm'], [-75, 129.9038105676658, 27])
        self.assertEqual(result['first']['bottom_radius_mm'], 144)
        self.assertEqual(result['first']['magnet_id'], 'R01-L1-A01')
        self.assertEqual(result['colors'], {'out':'#AB2D35','in':'#1955B0','down':'#A56E27','up':'#22877A'})

    def test_four_north_vectors_obey_radial_and_axial_polarity(self):
        result = self.js("""
console.log(JSON.stringify(['out','in','down','up'].map(p=>api.buildMagnets({rings:[ring({polarity:p,phase_deg:90})]})[0].north_vector)));
""")
        for actual, expected in zip(result, [[0,1,0],[0,-1,0],[0,0,-1],[0,0,1]]):
            self.assertVectorAlmostEqual(actual, expected)

    def test_rotated_cube_corners_keep_edge_lengths_and_local_radial_faces(self):
        result = self.js("""
const magnet=api.buildMagnets({cube_mm:12,rings:[ring({phase_deg:45})]})[0];
const vertices=api.cubeVertices(magnet);
console.log(JSON.stringify({vertices,edge:Math.hypot(...vertices[1].map((x,i)=>x-vertices[0][i])),center:magnet.position_mm}));
""")
        self.assertAlmostEqual(result['edge'], 12)
        center = [sum(v[i] for v in result['vertices'])/8 for i in range(3)]
        self.assertVectorAlmostEqual(center, result['center'])
        self.assertAlmostEqual(result['vertices'][1][0] - result['vertices'][0][0], 12 / 2**.5)
        self.assertAlmostEqual(result['vertices'][1][1] - result['vertices'][0][1], 12 / 2**.5)

    def test_ray_uses_rotated_obb_not_circumscribed_bounding_sphere(self):
        result = self.js("""
const magnet=api.buildMagnets({cube_mm:12,rings:[ring({phase_deg:45})]})[0];
const center=magnet.position_mm;
console.log(JSON.stringify({hit:api.rayCubeDistance([center[0],center[1],100],[0,0,-1],magnet),
  miss:api.rayCubeDistance([center[0]+8,center[1]+8,100],[0,0,-1],magnet)}));
""")
        self.assertAlmostEqual(result['hit'], 67)
        self.assertIsNone(result['miss'])

    def test_pick_returns_frontmost_cube_and_cutaway_excludes_hidden_half(self):
        result = self.js("""
const magnets=api.buildMagnets({rings:[ring({phase_deg:90}),ring({id:'R02-L1',phase_deg:90,z_mm:47})]});
const near=api.pickMagnet(magnets,[0,150,100],[0,0,-1]);
const visible=api.pickMagnet(magnets,[0,-300,27],[0,1,0],{cutaway:true});
const uncut=api.pickMagnet(magnets,[0,-300,27],[0,1,0]);
console.log(JSON.stringify({near:near?.ring_id,visible:visible?.position_mm,uncut:uncut?.position_mm}));
""")
        self.assertEqual(result['near'], 'R02-L1')
        self.assertGreaterEqual(result['visible'][1], 0)
        self.assertLess(result['uncut'][1], 0)

    def test_explode_is_display_only_and_pick_uses_displayed_coordinates(self):
        result = self.js("""
const magnet=api.buildMagnets({rings:[ring()]})[0],before=JSON.stringify(magnet);
const p=api.displayCenter(magnet,1),hit=api.pickMagnet([magnet],[p[0],p[1],100],[0,0,-1],{explode:1});
console.log(JSON.stringify({p,original:magnet.position_mm,unchanged:before===JSON.stringify(magnet),hit:hit?.magnet_id,bottom:hit?.bottom_radius_mm}));
""")
        self.assertVectorAlmostEqual(result['p'], [198, 0, 47.25])
        self.assertVectorAlmostEqual(result['original'], [150, 0, 27])
        self.assertTrue(result['unchanged'])
        self.assertEqual(result['hit'], 'R01-L1-A01')
        self.assertEqual(result['bottom'], 144)

    def test_empty_reference_and_invalid_geometry_never_allocate_unbounded_mesh(self):
        result = self.js("""
const errors=[];
for(const rings of [[ring({count:0})],[ring({polarity:'random'})],[ring({count:50001})],[ring({radius_mm:NaN})]]){
 try{api.buildMagnets({rings});errors.push(false);}catch(_){errors.push(true);}
}
for(const cube_mm of [Infinity,0,-12]){try{api.buildMagnets({rings:[ring()],cube_mm});errors.push(false);}catch(_){errors.push(true);}}
console.log(JSON.stringify({empty:api.buildMagnets(null),errors}));
""")
        self.assertEqual(result['empty'], [])
        self.assertEqual(result['errors'], [True] * 7)

    def test_camera_projection_basis_stays_orthonormal_at_polar_top_and_side_views(self):
        result = self.js("""
const bases=[api.cameraBasis(-Math.PI/2,Math.PI/2-1e-5),api.cameraBasis(-Math.PI/2,0),api.cameraBasis(-1.1,.52)];
console.log(JSON.stringify(bases.map(b=>{const vectors=[b.right,b.up,b.towardEye];return vectors.flatMap(a=>vectors.map(c=>a.reduce((s,x,i)=>s+x*c[i],0)));})));
""")
        for matrix in result:
            self.assertVectorAlmostEqual(matrix, [1,0,0,0,1,0,0,0,1])

    def test_javascript_block_centers_and_vectors_match_python_physics_geometry(self):
        from mri_magnet.geometry import Design, RingSpec
        design = Design(tuple(RingSpec('R%02d-L1' % (i+1), i, 0, 116.2+i*23.7,
                                      (-1)**i*(13.8+i*25.1), count, polarity, phase)
                              for i, (count, polarity, phase) in enumerate(
                                  [(26, 'out', -83.076923), (28, 'in', -83.571429),
                                   (30, 'down', 277.123456), (32, 'up', 81.25)])), cube_mm=12)
        actual = self.js('const magnets=api.buildMagnets(' + json.dumps(design.to_dict()) + ');'
                         'console.log(JSON.stringify(magnets.map(m=>({p:m.position_mm,n:m.north_vector,id:m.magnet_id}))));')
        expected = design.magnet_arrays()
        self.assertEqual(len(actual), design.magnet_count)
        for index, block in enumerate(actual):
            self.assertEqual(block['id'], expected['ids'][index])
            self.assertVectorAlmostEqual(block['p'], expected['positions_mm'][index], places=10)
            self.assertVectorAlmostEqual(block['n'], expected['north_vectors'][index], places=12)


if __name__ == '__main__':
    unittest.main()

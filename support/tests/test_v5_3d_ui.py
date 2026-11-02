"""Exercise real selection and inspection helpers, independent of a GPU."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is needed for actual frontend checks')
class InteractiveMagnetUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (Path(__file__).resolve().parents[2] / 'mri_magnet/workbench.html').read_text(encoding='utf-8')

    def js(self, names, body):
        functions = '\n'.join(re.search(r'^function '+name+r'\(.*$', self.html, re.M).group(0) for name in names)
        stub = """
const elements={};
const $=key=>elements[key]??(elements[key]={innerHTML:'',textContent:'',hidden:false,href:'',dataset:{},classList:{values:new Set(),toggle(key,value){value?this.values.add(key):this.values.delete(key)}},setAttribute(){}});
const $$=()=>[];
const document={querySelector:$};
let previewSelection=null,currentStep=5,liveFollowRequested=false,currentView='model3d',dirty=false,config={};
"""
        with tempfile.TemporaryDirectory(prefix='mri-3d-ui-') as root:
            script = Path(root) / 'check.js'
            script.write_text(stub + functions + '\n' + body, encoding='utf-8')
            result = subprocess.run(['node', str(script)], check=True, capture_output=True, text=True, encoding='utf-8')
        return json.loads(result.stdout)

    def source(self, body):
        return self.js(['fileURL', 'artifactPaths', 'seedImage', 'candidateLayouts', 'activeMagnetSource'], body)

    def test_explicit_candidate_beats_live_and_resolves_fresh_state(self):
        result = self.source("""
previewSelection={kind:'candidate',path:'runs/low.png',id:'low',label:'最低 PPM'};
const live={rings:[{radius_mm:190}]},low={rings:[{radius_mm:120}]};
const next={current_best:{design:live},artifacts:{candidate_layouts:[{path:'runs/low.png',candidate_id:'low',design:low}]}};
const before=activeMagnetSource(next);next.artifacts.candidate_layouts[0].design={rings:[{radius_mm:121}]};
console.log(JSON.stringify({before,after:activeMagnetSource(next)}));
""")
        self.assertEqual(result['before']['design']['rings'][0]['radius_mm'], 120)
        self.assertEqual(result['after']['design']['rings'][0]['radius_mm'], 121)
        self.assertEqual(result['after']['label'], '最低 PPM')

    def test_explicit_seed_beats_live_then_follow_restores_live(self):
        result = self.source("""
previewSelection={kind:'seed',path:'seed/two.png',id:'two',label:'起点二'};
const next={current_best:{design:{rings:[{radius_mm:150}]}},seeds:[{id:'two',preview:'seed/two.png',design:{rings:[{radius_mm:130}]}}]};
const picked=activeMagnetSource(next);previewSelection=null;const live=activeMagnetSource(next);
console.log(JSON.stringify({picked,live}));
""")
        self.assertEqual(result['picked']['design']['rings'][0]['radius_mm'], 130)
        self.assertEqual(result['live']['design']['rings'][0]['radius_mm'], 150)
        self.assertEqual(result['live']['kind'], 'live')

    def test_idle_step6_default_matches_verified_2d_layout(self):
        result = self.source("""
currentStep=6;const next={busy:false,current_best:{design:{rings:[{radius_mm:150}]}},best:{design:{rings:[{radius_mm:125}]}},artifacts:{layout:'verified.png'}};
const dense=activeMagnetSource(next);liveFollowRequested=true;const follow=activeMagnetSource(next);next.busy=true;liveFollowRequested=false;const running=activeMagnetSource(next);
console.log(JSON.stringify({dense,follow,running}));
""")
        self.assertEqual(result['dense']['design']['rings'][0]['radius_mm'], 125)
        self.assertEqual(result['dense']['kind'], 'verified')
        self.assertEqual(result['follow']['design']['rings'][0]['radius_mm'], 150)
        self.assertEqual(result['running']['kind'], 'live')

    def test_missing_selected_geometry_never_substitutes_other_candidate(self):
        result = self.source("""
previewSelection={kind:'candidate',path:'runs/a.png',id:'a',label:'候选 A'};
const next={current_best:{design:{rings:[{radius_mm:199}]}},candidate_layouts:[{path:'runs/a.png',candidate_id:'a'}]};
const missing=activeMagnetSource(next);next.candidate_layouts=[{path:'runs/a.png',candidate_id:'b',design:{rings:[{radius_mm:111}]}}];
console.log(JSON.stringify({missing,conflict:activeMagnetSource(next)}));
""")
        for key in ['missing', 'conflict']:
            self.assertIsNone(result[key]['design'])
            self.assertEqual(result[key]['kind'], 'unavailable')

    def test_conflicting_role_cannot_relabel_another_candidates_geometry(self):
        result = self.source("""
previewSelection={kind:'candidate',path:'runs/b.png',id:'a',label:'错误路径角色'};
const next={artifacts:{candidate_layouts:[{path:'runs/b.png',candidate_id:'b',design:{rings:[{radius_mm:155}]}}]},pareto_recommendations:[{layout_file:'runs/b.png',candidate_id:'a'}]};
console.log(JSON.stringify({layouts:candidateLayouts(next),source:activeMagnetSource(next)}));
""")
        self.assertEqual(result['layouts'][0]['id'], 'b')
        self.assertIsNone(result['source']['design'])
        self.assertEqual(result['source']['kind'], 'unavailable')

    def test_selected_seed_fallback_and_empty_envelope_reference(self):
        result = self.source("""
const seeds=[{id:'a',design:{rings:[{radius_mm:110}]}},{id:'b',design:{rings:[{radius_mm:130}]}}];
console.log(JSON.stringify({selected:activeMagnetSource({seeds,selected_seed_ids:['b']}),first:activeMagnetSource({seeds}),empty:activeMagnetSource({})}));
""")
        self.assertEqual(result['selected']['id'], 'b')
        self.assertEqual(result['first']['id'], 'a')
        self.assertEqual(result['empty']['kind'], 'reference')
        self.assertIsNone(result['empty']['design'])

    def test_layout_deduplication_preserves_geometry_and_id(self):
        result = self.source("""
console.log(JSON.stringify(candidateLayouts({artifacts:{candidate_layouts:[{path:'runs/a.png',candidate_id:'actual',design:{cube_mm:12,rings:[{radius_mm:140}]}}],layout:'runs/a.png'},pareto_recommendations:[{layout_file:'runs/a.png',label:'均衡'}]})));
""")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'actual')
        self.assertEqual(result[0]['design']['rings'][0]['radius_mm'], 140)

    def test_candidate_preview_highlight_survives_poll_rebuild(self):
        result = self.js(['fileURL', 'escapeHTML', 'prettyNumber', 'artifactPaths', 'candidateLayouts', 'renderCandidateLayouts', 'renderRecommendations'], """
previewSelection={kind:'candidate',path:'runs/low.png',id:'low'};
const next={candidate_layouts:[{path:'runs/low.png',candidate_id:'low',label:'最低 PPM'}],pareto_recommendations:[{layout_file:'runs/low.png',candidate_id:'low',role:'lowest_ppm',metrics:{mean_norm_T:.045,pp_norm_ppm:90}}]};
renderCandidateLayouts(next);renderRecommendations(next);renderCandidateLayouts(next);renderRecommendations(next);
console.log(JSON.stringify({layouts:$('#result-layouts').innerHTML,roles:$('#pareto-recommendations').innerHTML}));
""")
        self.assertIn('preview-selected', result['layouts'])
        self.assertIn('preview-selected', result['roles'])

    def test_card_preview_keeps_3d_active_and_switches_other_views_to_2d(self):
        result = self.js(['fileURL', 'selectLayoutPreview'], """
let state=null;selectLayoutPreview('a.png','A','a');const first=currentView;currentView='metrics';selectLayoutPreview('b.png','B','b');
console.log(JSON.stringify({first,second:currentView,selected:previewSelection}));
""")
        self.assertEqual(result['first'], 'model3d')
        self.assertEqual(result['second'], 'geometry')
        self.assertEqual(result['selected']['path'], 'b.png')

    def test_inspector_uses_actual_positions_and_bottom_surface_and_escapes_text(self):
        result = self.js(['prettyNumber', 'escapeHTML', 'inspectMagnet'], """
inspectMagnet({ring_id:'R01-<L1>',magnet_id:'block 3',position_mm:[120,0,19],z_mm:19,radius_mm:120,bottom_radius_mm:114,cube_mm:12,theta_deg:0,polarity:'up',count:28},{pinned:true});
const full=$('#magnet-inspector').innerHTML,corner=$('#magnet-hover-info').innerHTML,mode=$('#inspector-mode').textContent;
inspectMagnet(null);console.log(JSON.stringify({full,corner,mode,cleared:$('#magnet-hover-info').hidden}));
""")
        self.assertIn('114', result['full'])
        self.assertIn('120, 0, 19', result['full'])
        self.assertIn('R01-&lt;L1&gt;', result['full'])
        self.assertNotIn('R01-<L1>', result['full'])
        self.assertIn('N 轴向 +Z', result['full'])
        self.assertEqual(result['mode'], '已点击锁定')
        self.assertTrue(result['cleared'])

    def test_physics_draft_change_is_distinguished_from_solver_budget_edit(self):
        result = self.js(['physicsDraftChanged'], """
dirty=true;config={target:{diameter_mm:100},solver:{max_evaluations:999}};const next={config:{target:{diameter_mm:100},solver:{max_evaluations:20}}};
const solverOnly=physicsDraftChanged(next);config.target.diameter_mm=200;const target=physicsDraftChanged(next);dirty=false;
console.log(JSON.stringify({solverOnly,target,saved:physicsDraftChanged(next)}));
""")
        self.assertFalse(result['solverOnly'])
        self.assertTrue(result['target'])
        self.assertFalse(result['saved'])

    def test_draft_target_does_not_claim_old_dense_metrics_as_new_domain(self):
        names = ['fileURL', 'prettyNumber', 'escapeHTML', 'viewPaths', 'physicsDraftChanged', 'activeMagnetSource', 'candidateLayouts', 'artifactPaths', 'seedImage', 'renderCanvas']
        result = self.js(names, """
dirty=true;currentView='metrics';currentStep=6;config={target:{geometry:'sphere',diameter_mm:200,min_field_mT:40}};
let timelinePinned=false,frames=[],analysisView='convergence';function updateMagnetViewport(){};
renderCanvas({config:{target:{geometry:'sphere',diameter_mm:100,min_field_mT:40}},best:{metrics:{mean_norm_T:.048,pp_norm_ppm:143},comsol_fem_verified:true}});
console.log(JSON.stringify({table:$('#canvas-metrics').innerHTML,caption:$('#canvas-source').textContent}));
""")
        self.assertIn('Φ200', result['caption'])
        self.assertIn('参数草稿 · 尚未重新计算', result['table'])
        self.assertNotIn('143', result['table'])
        self.assertNotIn('已完成', result['table'])
        self.assertNotIn('密集复核</td>', result['table'])


if __name__ == '__main__':
    unittest.main()

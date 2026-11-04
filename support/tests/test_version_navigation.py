"""Version navigation opens isolated workbenches without touching solver state."""
from pathlib import Path
import json,re,shutil,subprocess,tempfile,unittest
from mri_magnet.version_navigation import inject_version_navigation

ROOT=Path(__file__).resolve().parents[2]
PAGE=ROOT/'mri_magnet'/'workbench.html'


class VersionNavigationTests(unittest.TestCase):
    def test_v5_overlay_preserves_original_backend_page_and_uses_single_component(self):
        original='<html><head><title>V5</title></head><body><aside><b>V5 original</b></aside><script>var v5Physics=true;</script></body></html>'
        page=inject_version_navigation(original,PAGE,'v5')
        self.assertIn('data-version="v5"',page)
        self.assertIn('V5 · 固定源',page)
        self.assertIn('<script>var v5Physics=true;</script>',page)
        self.assertEqual(page.count('id="version-navigation"'),1)
        self.assertEqual(inject_version_navigation(page,PAGE,'v5'),page)
        restored=re.sub(r'<!-- MRI_VERSION_(STYLE|MARKUP|SCRIPT)_START -->.*?<!-- MRI_VERSION_\1_END -->','',page,flags=re.S)
        self.assertEqual(restored,original)

    def test_unknown_version_or_broken_page_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):inject_version_navigation('<html/>',PAGE,'v7')
        with self.assertRaisesRegex(ValueError,'sidebar'):inject_version_navigation('<html><head></head><body></body></html>',PAGE,'v5')

    @unittest.skipUnless(shutil.which('node'),'Node required for real navigation script')
    def test_shared_script_resolves_live_v6_port_and_rejects_foreign_cookie_targets(self):
        html=PAGE.read_text(encoding='utf-8')
        script=re.search(r'<script id="version-navigation-script">(.*?)</script>',html,re.S).group(1)
        js=script+'''\nconst api=globalThis.MRIVersionNavigation;
console.log(JSON.stringify([
 api.targets('v6','http://127.0.0.1:8766',''),
 api.targets('v5','http://127.0.0.1:8764','mri_v6_active_port=8766'),
 api.targets('v5','http://localhost:8764','mri_v6_active_port=https://evil.example'),
 api.targets('v5','http://127.0.0.1:8764','mri_v6_active_port=99999')
]));'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nav.js';path.write_text(js,encoding='utf-8')
            result=subprocess.run(['node',str(path)],check=True,capture_output=True,text=True)
        rows=json.loads(result.stdout)
        self.assertEqual(rows[0]['v6'],'http://127.0.0.1:8766/')
        self.assertEqual(rows[1]['v6'],'http://127.0.0.1:8766/')
        self.assertEqual(rows[2]['v6'],'http://localhost:8765/')
        self.assertEqual(rows[3]['v6'],'http://127.0.0.1:8765/')
        self.assertTrue(all(row['v5'].endswith(':8764/') for row in rows))

    def test_busy_solver_does_not_disable_version_navigation_or_change_its_config(self):
        page=PAGE.read_text(encoding='utf-8')
        markup=re.search(r'<!-- MRI_VERSION_MARKUP_START -->(.*?)<!-- MRI_VERSION_MARKUP_END -->',page,re.S).group(1)
        self.assertEqual(markup.count('target="_blank"'),2)
        self.assertEqual(markup.count('rel="noopener"'),2)
        self.assertNotIn('data-preset',markup);self.assertNotIn('data-path',markup)
        script=re.search(r'<script id="version-navigation-script">(.*?)</script>',page,re.S).group(1)
        self.assertNotIn('fetch(',script);self.assertNotIn('/api/config',script);self.assertNotIn('/api/cancel',script)
        self.assertIn('MRIVersionNavigation?.render(next)',page)


if __name__=='__main__':unittest.main()

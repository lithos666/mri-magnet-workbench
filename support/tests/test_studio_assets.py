"""Serve the photographic studio entirely from explicit local resources."""
import hashlib
import http.client
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.parse import urljoin
from unittest.mock import patch

from mri_magnet.workbench_server import create_server, VIEWPORT_ASSETS


class StudioAssetsTests(unittest.TestCase):
    def setUp(self):
        class Backend:
            def state(self): return {'steps': []}
        self.temp = tempfile.TemporaryDirectory()
        self.server = create_server(self.temp.name, port=0, workbench=Backend())
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.package = Path(__file__).resolve().parents[2] / 'mri_magnet'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)
        self.temp.cleanup()

    def request(self, route):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=5)
        connection.request('GET', route)
        response = connection.getresponse()
        result = response.status, response.getheader('Content-Type'), response.read()
        connection.close()
        return result

    def test_local_studio_dependency_bytes_and_mime_are_served_exactly(self):
        # The interactive module is authored independently; its route is tested
        # by browser integration too. Everything it imports is a real local file.
        for route, relative in VIEWPORT_ASSETS.items():
            if relative == 'workbench_studio3d.js':
                continue
            with self.subTest(route=route):
                status, mime, content = self.request(route)
                self.assertEqual(status, 200)
                expected = 'application/javascript' if relative.endswith('.js') else 'text/plain'
                self.assertEqual(mime, expected + '; charset=utf-8')
                self.assertEqual(content, (self.package / relative).read_bytes())
        with patch('mri_magnet.workbench_server.Path.is_file', return_value=True), \
                patch('mri_magnet.workbench_server.Path.read_bytes', return_value=b'export class StudioViewer {}'):
            status, mime, content = self.request('/assets/workbench_studio3d.js')
        self.assertEqual(status, 200)
        self.assertEqual(mime, 'application/javascript; charset=utf-8')
        self.assertEqual(content, b'export class StudioViewer {}')

    def test_nested_arbitrary_files_and_traversal_are_never_public_assets(self):
        for route in ('/assets/', '/assets/vendor/', '/assets/workbench.py',
                      '/assets/vendor/../workbench_server.py',
                      '/assets/vendor/%2e%2e/workbench_server.py',
                      '/assets/vendor/%2F..%2Fworkbench.py',
                      '/assets/vendor/three.module.js/extra',
                      '/assets/vendor/missing.js'):
            with self.subTest(route=route):
                self.assertEqual(self.request(route)[0], 404)

    def test_missing_packaged_asset_has_explicit_404(self):
        with patch('mri_magnet.workbench_server.Path.is_file', return_value=False):
            for route in VIEWPORT_ASSETS:
                self.assertEqual(self.request(route)[0], 404)

    def test_every_module_import_resolves_to_a_whitelisted_offline_module(self):
        imports = []
        for route, relative in VIEWPORT_ASSETS.items():
            file = self.package / relative
            if not relative.endswith('.js') or not file.is_file():
                continue
            text = file.read_text(encoding='utf-8')
            # Only module declarations at the start of a line; documentation
            # snippets prefixed with '*' are not executable imports.
            for specifier in re.findall(r'''(?m)^[ \t]*(?:import|export)\b[^;]*?\bfrom\s*['"]([^'"]+)['"]\s*;''', text):
                with self.subTest(module=relative, dependency=specifier):
                    self.assertTrue(specifier.startswith('./') or specifier.startswith('../'))
                    resolved = urljoin(route, specifier)
                    self.assertIn(resolved, VIEWPORT_ASSETS)
                    self.assertTrue((self.package / VIEWPORT_ASSETS[resolved]).is_file())
                    imports.append((relative, specifier))
        self.assertIn(('vendor/three.module.js', './three.core.js'), imports)
        self.assertIn(('vendor/OrbitControls.js', './three.module.js'), imports)
        self.assertIn(('workbench_studio_lighting.js', './vendor/three.module.js'), imports)

    def test_vendored_library_provenance_and_mri_reference_lighting(self):
        # Fixed source digests protect the reused, locally licensed engine.
        expected = {
            'three.module.js': 'c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a',
            'three.core.js': 'eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20',
            'THREE-LICENSE.txt': 'bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((self.package / 'vendor' / name).read_bytes()).hexdigest(), digest)
        lighting=(self.package/'workbench_studio_lighting.js').read_text(encoding='utf-8')
        self.assertIn('Mri-ring-explorer/components/mri-scene.tsx',lighting)
        self.assertIn("north:'#e03d34',south:'#0071e3'",lighting)
        self.assertIn('exposure:1.1,environmentIntensity:.35,fov:37,damping:.08',lighting)
        self.assertIn("from './vendor/RoomEnvironment.js'",lighting)
        orbit = (self.package / 'vendor/OrbitControls.js').read_bytes()
        restored_orbit = orbit.replace(b"from './three.module.js'", b"from 'three'")
        self.assertEqual(hashlib.sha256(restored_orbit).hexdigest(), 'b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c')
        self.assertIn(b'MIT License', (self.package / 'vendor/THREE-LICENSE.txt').read_bytes())


if __name__ == '__main__':
    unittest.main()

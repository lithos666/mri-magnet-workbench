"""Candidate exports remain self-contained and failures stay reviewable."""
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from mri_magnet.comsol_export import export_comsol_model, generate_java_model
from mri_magnet.geometry import Design, RingSpec


class ComsolExportTests(unittest.TestCase):
    def fixture(self):
        return Design((
            RingSpec("R01-L1", 0, 0, 128.375, 247.0, 28, "out", 276.428571),
            RingSpec("R02-L3", 1, 2, 154.4, 208.25, 32, "in", 275.625),
            RingSpec("R13-L1", 12, 0, 130.0, -208.25, 24, "down", 277.5),
            RingSpec("R14-L4", 13, 3, 180.625, -246.8, 28, "up", 276.428571),
        ), name="含中文的候选", source_sha256="source-reference")

    def test_embedded_ring_table_reconstructs_actual_candidate_centers_and_directions(self):
        design = self.fixture()
        _, source, manifest = generate_java_model(design)
        table = re.search(r"RINGS = new double\[\]\[\] \{(.*?)\n  \};", source, re.S).group(1)
        rows = [json.loads("[" + row + "]") for row in re.findall(r"\{([^{}]+)\}", table)]
        actual = design.magnet_arrays()
        centers, directions = [], []
        for radius, z, count, phase, polarity in rows:
            for column in range(int(count)):
                theta = np.deg2rad(phase + column * 360 / count)
                radial = np.array([np.cos(theta), np.sin(theta), 0.0])
                centers.append([radius * radial[0], radius * radial[1], z])
                directions.append((radial, -radial, np.array([0., 0., -1.]), np.array([0., 0., 1.]))[int(polarity)])
        np.testing.assert_allclose(centers, actual["positions_mm"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(directions, actual["north_vectors"], rtol=0, atol=1e-14)
        self.assertEqual(manifest["magnet_count"], len(centers))
        self.assertEqual(manifest["ring_count"], 4)
        self.assertEqual(manifest["direction_counts"], {"out": 28, "in": 32, "down": 24, "up": 28})
        self.assertEqual([row["slot_index"] for row in manifest["rings"]], [0, 2, 0, 3])

    def test_hash_stable_for_renaming_and_sensitive_to_geometry_and_material(self):
        design = self.fixture()
        class_name, _, original = generate_java_model(design)
        renamed_class, _, renamed = generate_java_model(replace(design, name="display name only"))
        self.assertEqual(original["design_sha256"], renamed["design_sha256"])
        self.assertEqual(class_name, renamed_class)
        changes = [
            replace(design, remanence_T=1.4),
            replace(design, reference_mur=1.08),
            replace(design, rings=(replace(design.rings[0], z_mm=248),) + design.rings[1:]),
            replace(design, rings=(replace(design.rings[0], polarity="in"),) + design.rings[1:]),
        ]
        for changed in changes:
            with self.subTest(design=changed):
                changed_class, _, manifest = generate_java_model(changed)
                self.assertNotEqual(manifest["design_sha256"], original["design_sha256"])
                self.assertNotEqual(changed_class, class_name)
        other_class, _, other = generate_java_model(design, {"air_radius_mm": 700})
        self.assertEqual(original["design_sha256"], other["design_sha256"])
        self.assertNotEqual(class_name, other_class)

    def test_model_is_self_contained_without_silent_fem_execution(self):
        class_name, source, manifest = generate_java_model(self.fixture())
        source.encode("ascii")
        self.assertRegex(class_name, r"^MRI_Design_[0-9a-f]{12}$")
        self.assertNotIn("ModelUtil.load", source)
        self.assertNotIn(".mph", source)
        self.assertNotIn(".save(", source)
        self.assertNotRegex(source, r'\.study\([^)]*\)\.run\(')
        self.assertNotRegex(source, r'\.mesh\([^)]*\)\.run\(')
        self.assertIn('"MagnetostaticsNoCurrents"', source)
        self.assertIn('"RemanentFluxDensity"', source)
        exported_br = re.search(r'\.set\("io_Br","([^\[]+)\[T\]"\)', source).group(1)
        self.assertEqual(float(exported_br), self.fixture().remanence_T)
        self.assertFalse(manifest["mesh_built"])
        self.assertFalse(manifest["solve_executed"])
        self.assertFalse(manifest["native_model_constructed"])

    def test_missing_compiler_preserves_java_and_explicit_failure_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = export_comsol_model(self.fixture(), root / "candidate", root / "no_comsol", {"logs_dir": root / "logs"})
            self.assertEqual(result["status"], "compile_failed")
            self.assertFalse(result["compiled"])
            self.assertIsNone(result["class_file"])
            self.assertIn("compiler not found", result["error"])
            self.assertTrue(Path(result["java_file"]).is_file())
            saved = json.loads(Path(result["manifest_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "compile_failed")
            self.assertEqual(saved["design_sha256"], result["design_sha256"])
            self.assertEqual(Path(result["manifest_file"]).parent, root / "logs")

    def test_java_only_export_does_not_report_stale_class_as_success(self):
        design = self.fixture()
        name, _, _ = generate_java_model(design)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stale = root / (name + ".class")
            stale.write_bytes(b"old invalid class file")
            result = export_comsol_model(design, root, root / "no_comsol", {"compile": False})
            self.assertEqual(result["status"], "java_generated_not_compiled")
            self.assertFalse(result["compiled"])
            self.assertIsNone(result["class_file"])
            self.assertFalse(result["solve_executed"])

    def test_invalid_settings_or_overlapping_geometry_fail_before_export(self):
        design = self.fixture()
        settings = [
            {"air_radius_mm": 100}, {"air_radius_mm": float("nan")},
            {"air_mu_r": 0}, {"timeout_seconds": -1},
            {"mesh_size": 0}, {"mesh_size": 5.5},
            {"native_processors": 0}, {"class_prefix": "bad-name"},
        ]
        for options in settings:
            with self.subTest(settings=options):
                with self.assertRaises(ValueError):
                    generate_java_model(design, options)
        with self.assertRaises(ValueError):
            generate_java_model(replace(design, rings=(design.rings[0], replace(design.rings[0], id="duplicate"))))
        with self.assertRaises(ValueError):
            generate_java_model(replace(design, reference_mur=-1))


if __name__ == "__main__":
    unittest.main()

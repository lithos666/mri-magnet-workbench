"""Run packaging keeps infeasible references out and preserves failed exports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mri_magnet.deliverables import export_candidates, export_saved_run, finalize_results, selected_candidates
from mri_magnet.geometry import Design, RingSpec


class DeliverablesTests(unittest.TestCase):
    def design(self):
        return Design((RingSpec("R01-L1", 0, 0, 130., 20., 8, "out", 12.),), name="checked")

    def record(self, ident, mean=0.04, ppm=1000., **updates):
        metrics = {"mean_norm_T": mean, "pp_norm_ppm": ppm, "magnet_count": 8, "mass_magnet_kg": 0.1}
        row = {"candidate_id": ident, "feasible": True, "metrics": metrics,
               "validation_metrics": dict(metrics), "design": self.design().to_dict(),
               "geometry_check": {"valid": True}, "manufacturing_check": {"valid": True}}
        row.update(updates)
        return row

    def payload(self):
        baseline = self.record("baseline", mean=0.05, ppm=100., feasible=False,
                               manufacturing_check={"valid": False})
        seed = self.record("manufacturing_seed", mean=0.04, ppm=1000.)
        return {"baseline": baseline, "manufacturing": {"config": {}}, "candidates": [baseline, seed]}

    def test_candidate_selection_excludes_unchecked_or_failed_records(self):
        payload = self.payload()
        payload["candidates"].extend([
            self.record("unchecked", mean=0.1, ppm=1., validation_metrics=None),
            self.record("overlapping", mean=0.1, ppm=1., geometry_check={"valid": False}),
            self.record("fails_printing", mean=0.1, ppm=1., manufacturing_check={"valid": False}),
            self.record("no_printing_check", mean=0.1, ppm=1., manufacturing_check={}),
        ])
        self.assertEqual([r["candidate_id"] for r in selected_candidates(payload)], ["manufacturing_seed"])

    def test_selection_keeps_independent_pareto_choices_and_seed_fallback(self):
        payload = self.payload()
        payload["candidates"].extend([
            self.record("strong", mean=0.05, ppm=2000.),
            self.record("uniform", mean=0.03, ppm=100.),
            self.record("dominated", mean=0.03, ppm=2000.),
        ])
        self.assertEqual([r["candidate_id"] for r in selected_candidates(payload)],
                         ["uniform", "manufacturing_seed", "strong"])
        payload["candidates"] = payload["candidates"][:2]
        self.assertEqual([r["candidate_id"] for r in selected_candidates(payload)], ["manufacturing_seed"])

    def _render(self, design, target, **kwargs):
        target.write_bytes(b"layout output")
        return target

    def _export(self, design, target, comsol_root, settings):
        java = target / "MRI_checked.java"
        compiled = target / "MRI_checked.class"
        java.write_text("class MRI_checked {}", encoding="ascii")
        compiled.write_bytes(bytes.fromhex("CAFEBABE") + b"compiled model")
        logs = Path(settings["logs_dir"])
        manifest = logs / "MRI_checked_export.json"
        manifest.write_text("{}", encoding="utf-8")
        log = logs / "MRI_checked_compile.log"
        log.write_text("compiled", encoding="utf-8")
        return {"status": "compiled_class_ready_not_fem_verified", "compiled": True,
                "java_file": str(java), "class_file": str(compiled),
                "manifest_file": str(manifest), "compile_log": str(log),
                "magnet_count": design.magnet_count, "ring_count": len(design.rings),
                "model_built": False, "solve_executed": False}

    def test_layout_and_class_use_the_same_actual_design_and_relative_files(self):
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            details = public / "详细数据"
            with patch("mri_magnet.deliverables.render_rz_layout", side_effect=self._render) as render, \
                 patch("mri_magnet.deliverables.export_comsol_model", side_effect=self._export) as export:
                result = export_candidates(public, details, self.payload(), {"comsol_root": "COMSOL"}, log=lambda _: None)
            self.assertTrue(result["success"])
            self.assertIs(render.call_args.args[0], export.call_args.args[0])
            self.assertEqual(render.call_args.args[0].to_dict(), self.design().to_dict())
            item = result["candidates"][0]
            self.assertTrue(item["is_starting_reference"])
            self.assertFalse(item["solve_executed"])
            for key in ("layout_file", "java_file", "class_file", "manifest_file", "compile_log"):
                self.assertFalse(Path(item[key]).is_absolute())
                self.assertTrue((public / item[key]).is_file())
            saved = json.loads((details / "deliverables_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["success"])

    def test_compile_failure_keeps_layout_and_marks_failure_explicitly(self):
        def failed(design, target, comsol_root, settings):
            return {"status": "compile_failed", "compiled": False, "error": "compiler unavailable"}
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            with patch("mri_magnet.deliverables.render_rz_layout", side_effect=self._render), \
                 patch("mri_magnet.deliverables.export_comsol_model", side_effect=failed):
                result = export_candidates(public, public / "详细数据", self.payload(),
                                           {"comsol_root": "missing"}, log=lambda _: None)
            self.assertFalse(result["success"])
            item = result["candidates"][0]
            self.assertEqual(item["status"], "compile_failed")
            self.assertIn("compiler unavailable", item["error"])
            self.assertTrue((public / item["layout_file"]).is_file())
            self.assertIsNone(item["class_file"])

    def test_finalize_still_saves_comparison_payload_and_report_after_failure(self):
        failed = {"enabled": True, "success": False, "candidates": [], "error": "compile failed"}
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder) / "results" / "run01"
            details = public / "详细数据"
            payload = deepcopy(self.payload())
            report = {"report_html": str(public / "report_中文.html")}
            with patch("mri_magnet.deliverables.export_candidates", return_value=failed), \
                 patch("mri_magnet.reporting.create_report", return_value=report) as build_report:
                actual, success = finalize_results(public, details, payload, {"comsol_root": "missing"}, log=lambda _: None)
            self.assertEqual(actual, report)
            self.assertFalse(success)
            self.assertTrue((public / "方案对比.csv").is_file())
            saved = json.loads((details / "report_payload.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["deliverables"]["success"])
            build_report.assert_called_once_with(public.resolve(), payload, details_dir=details.resolve())

    def test_saved_run_failure_returns_nonzero_without_field_recomputation(self):
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            details = public / "详细数据"
            details.mkdir()
            (details / "report_payload.json").write_text(json.dumps(self.payload()), encoding="utf-8")
            with patch("mri_magnet.deliverables.finalize_results", return_value=({"report_html": "report"}, False)):
                self.assertEqual(export_saved_run(public, {"comsol_root": "missing"}, log=lambda _: None), 2)


if __name__ == "__main__":
    unittest.main()

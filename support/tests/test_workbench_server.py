"""Exercise local HTTP orchestration without invoking magnetic calculations."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from mri_magnet.workbench_server import create_server


class FakeWorkbench:
    def __init__(self):
        self.config = {"mode": "de_novo"}
        self.selected = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.fail = False
        self.cancelled = False

    def state(self):
        return {"config": deepcopy(self.config), "steps": [
            {"number": i, "status": "pending", "name": str(i), "message": ""}
            for i in range(1, 7)], "selected_seed_ids": list(self.selected)}

    def update_config(self, config):
        self.config = deepcopy(config)
        return {"saved": True}

    def preflight(self):
        return {"valid": True, "maximum_radial_rings": 5}

    def select_seeds(self, ids):
        if not 1 <= len(ids) <= 3:
            raise ValueError("请选择 1–3 套起点")
        self.selected = list(ids)
        return self.selected

    def run_step(self, number, **body):
        self.entered.set()
        self.release.wait(5)
        try:
            if self.fail:
                raise ValueError("测试中的工艺冲突")
        finally:
            self.finished.set()

    def cancel(self):
        self.cancelled = True
        self.release.set()
        return {"cancel_requested": True}


class WorkbenchServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wb = FakeWorkbench()
        self.server = create_server(self.root, port=0, workbench=self.wb)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.wb.release.set()
        if self.server.current_future:
            try:
                self.server.current_future.result(timeout=5)
            except Exception:
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def request(self, path="/api/state", body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        method = "POST" if body is not None or raw is not None else "GET"
        payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
        conn.request(method, path, body=payload, headers=headers or {})
        response = conn.getresponse()
        status, data = response.status, response.read()
        conn.close()
        return status, data

    def test_page_and_state_have_no_solver_dependency(self):
        status, data = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("MRI 磁体设计".encode(), data)
        self.assertIn(b"/api/step/", data)
        status, data = self.request()
        value = json.loads(data)
        self.assertEqual(value["config"]["mode"], "de_novo")
        self.assertFalse(value["busy"])
        self.assertEqual(len(value["steps"]), 6)

    def test_config_preflight_and_seed_selection(self):
        status, _ = self.request("/api/config", {"config": {"mode": "seeded"}})
        self.assertEqual(status, 200)
        self.assertEqual(self.wb.config["mode"], "seeded")
        status, data = self.request("/api/preflight", {})
        self.assertTrue(json.loads(data)["result"]["valid"])
        status, _ = self.request("/api/select", {"ids": ["seed_1"]})
        self.assertEqual(status, 200)
        self.assertEqual(self.wb.selected, ["seed_1"])
        status, _ = self.request("/api/select", {"ids": []})
        self.assertEqual(status, 400)

    def test_background_job_is_nonblocking_and_exclusive(self):
        status, _ = self.request("/api/step/5", {})
        self.assertEqual(status, 202)
        self.assertTrue(self.wb.entered.wait(2))
        status, data = self.request()
        self.assertTrue(json.loads(data)["busy"])
        for path, body in [("/api/step/6", {}), ("/api/config", {"config": {}}),
                           ("/api/select", {"ids": ["seed_1"]}), ("/api/preflight", {})]:
            status, _ = self.request(path, body)
            self.assertEqual(status, 409)
        status, _ = self.request("/api/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue(self.wb.cancelled)
        self.server.current_future.result(timeout=5)
        status, data = self.request()
        self.assertFalse(json.loads(data)["busy"])

    def test_background_error_becomes_visible_error_state(self):
        self.wb.fail = True
        self.wb.release.set()
        status, _ = self.request("/api/step/3", {})
        self.assertEqual(status, 202)
        self.server.current_future.result(timeout=5)
        status, data = self.request()
        value = json.loads(data)
        self.assertEqual(value["job"]["status"], "failed")
        self.assertIn("工艺冲突", value["error"])
        self.assertEqual(value["steps"][2]["status"], "error")
        status, _ = self.request("/api/config", {"config": {"mode": "de_novo"}})
        self.assertEqual(status, 200)
        status, data = self.request()
        self.assertEqual(json.loads(data)["job"]["status"], "idle")

    def test_unicode_artifacts_are_served_and_paths_stay_in_project(self):
        (self.root / "磁铁布局.png").write_bytes(b"image-content")
        from urllib.parse import quote
        status, data = self.request("/files/" + quote("磁铁布局.png"))
        self.assertEqual(status, 200)
        self.assertEqual(data, b"image-content")
        for path in ["/files/../outside.txt", "/files/%2e%2e/outside.txt", "/files/C:%5Csecret.txt"]:
            status, _ = self.request(path)
            self.assertIn(status, (400, 403))
        status, _ = self.request("/files/missing.png")
        self.assertEqual(status, 404)

    def test_foreign_origin_and_host_are_rejected(self):
        status, _ = self.request("/api/config", {"config": {}},
                                 headers={"Origin": "https://external.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("/api/state", headers={"Host": "external.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("/api/config", {"config": {"ok": True}},
                                 headers={"Origin": "http://127.0.0.1:%d" % self.port})
        self.assertEqual(status, 200)

    def test_invalid_requests_report_json_errors(self):
        for path, raw in [("/api/config", "[]"), ("/api/config", "{"),
                           ("/api/config", '{"config": 1}'), ("/api/step/0", "{}"),
                           ("/api/step/not-a-number", "{}")]:
            status, data = self.request(path, raw=raw)
            self.assertEqual(status, 400)
            self.assertIn("error", json.loads(data))


class RealBackendServerTests(WorkbenchServerTests):
    """Real persistent step/config integration; no field evaluation is needed."""

    # Inherit request/teardown helpers, not the fake-backend behavior tests.
    test_page_and_state_have_no_solver_dependency = None
    test_config_preflight_and_seed_selection = None
    test_background_job_is_nonblocking_and_exclusive = None
    test_background_error_becomes_visible_error_state = None
    test_unicode_artifacts_are_served_and_paths_stay_in_project = None
    test_foreign_origin_and_host_are_rejected = None
    test_invalid_requests_report_json_errors = None

    def setUp(self):
        from mri_magnet.workbench import Workbench
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wb = Workbench(self.root)
        self.server = create_server(self.root, port=0, workbench=self.wb)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        if self.server.current_future:
            self.server.current_future.result(timeout=10)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def run_step(self, number):
        status, data = self.request("/api/step/%d" % number, {})
        self.assertEqual(status, 202, data)
        self.server.current_future.result(timeout=15)
        status, data = self.request()
        return json.loads(data)

    def test_real_steps_one_two_preflight_preview_and_persistence(self):
        with patch("mri_magnet.field.FieldEngine.field", side_effect=AssertionError("preflight cannot evaluate fields")):
            first = self.run_step(1)
            self.assertEqual(first["steps"][0]["status"], "passed")
            second = self.run_step(2)
            self.assertEqual(second["steps"][1]["status"], "passed")
            self.assertGreater(second["preflight"]["max_radial_rings"], 0)
            self.assertGreater(second["preflight"]["max_axial_pairs"], 0)
            preview = second["steps"][1]["preview"]
            from urllib.parse import quote
            status, data = self.request("/files/" + quote(preview))
            self.assertEqual(status, 200)
            self.assertTrue(data.startswith(b"\x89PNG"))
            saved = json.loads((self.root / "工作台状态.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["steps"][1]["status"], "passed")

    def test_real_solver_config_edit_keeps_existing_seeds_and_upstream_checks(self):
        self.wb.data["seeds"] = [{"id": "island_a"}]
        self.wb.data["selected_seed_ids"] = ["island_a"]
        for step in self.wb.data["steps"]:
            step.update(status="passed", message="previously checked")
        old_budget = self.wb.config["solver"]["max_evaluations"]
        status, data = self.request("/api/config", {"config": {"solver": {"max_evaluations": old_budget + 1}}})
        self.assertEqual(status, 200, data)
        state = json.loads(data)["state"]
        self.assertTrue(all(step["status"] == "passed" for step in state["steps"][:4]))
        self.assertEqual(state["selected_seed_ids"], ["island_a"])
        self.assertEqual(state["steps"][4]["status"], "pending")

    def test_real_target_edit_invalidates_topologies_and_upstream_checks(self):
        self.wb.data["seeds"] = [{"id": "island_a"}]
        self.wb.data["selected_seed_ids"] = ["island_a"]
        for step in self.wb.data["steps"]:
            step["status"] = "passed"
        status, data = self.request("/api/config", {"config": {"target": {"diameter_mm": 22}}})
        self.assertEqual(status, 200, data)
        state = json.loads(data)["state"]
        self.assertEqual(state["selected_seed_ids"], [])
        self.assertEqual(state["seeds"], [])
        self.assertEqual(state["steps"][0]["status"], "pending")

    def test_backend_progress_is_preserved_by_http_snapshot(self):
        self.wb.data["job"] = {"evaluations": 43, "stage": "paired", "message": "精修中"}
        status, data = self.request()
        job = json.loads(data)["job"]
        self.assertEqual(job["evaluations"], 43)
        self.assertEqual(job["stage"], "paired")


if __name__ == "__main__":
    unittest.main()

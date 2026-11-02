"""Loopback-only, dependency-free web front end for the six-step workbench.

The HTTP layer owns one background job. Physics and persisted workflow state
belong to Workbench, which is imported lazily so this server can be tested
without loading a field solver.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import socket
import threading
from urllib.parse import unquote, urlsplit


# Exact packaged resource names: no directory browsing or arbitrary module paths.
VIEWPORT_ASSETS = {
    '/assets/workbench_analysis.js': 'workbench_analysis.js',
    '/assets/workbench_3d.js': 'workbench_3d.js',
    '/assets/workbench_preview.js': 'workbench_preview.js',
    '/assets/workbench_studio3d.js': 'workbench_studio3d.js',
    '/assets/workbench_studio_lighting.js': 'workbench_studio_lighting.js',
    '/assets/vendor/three.module.js': 'vendor/three.module.js',
    '/assets/vendor/three.core.js': 'vendor/three.core.js',
    '/assets/vendor/OrbitControls.js': 'vendor/OrbitControls.js',
    '/assets/vendor/RoomEnvironment.js': 'vendor/RoomEnvironment.js',
    '/assets/vendor/THREE-LICENSE.txt': 'vendor/THREE-LICENSE.txt',
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Cannot serialize " + type(value).__name__)


class WorkbenchHTTPServer(ThreadingHTTPServer):
    """A locally bound server; use create_server for deterministic teardown."""

    daemon_threads = True
    # On Windows SO_REUSEADDR can let two workbenches claim one port and
    # leave clients talking to the older backend. Require exclusive binding.
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name=='nt' and hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

    def __init__(self, address, workbench, project_dir):
        if address[0] != "127.0.0.1":
            raise ValueError("The workbench must bind to 127.0.0.1")
        self.workbench = workbench
        self.project_dir = Path(project_dir).resolve()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-workbench")
        self.export_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-checkpoint-export")
        self.analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-engineering-analysis")
        self.job_lock = threading.RLock()
        self.current_future = None
        self.export_future = None
        self.analysis_future = None
        self.analysis_job = {"status": "idle"}
        self.export_job = {"status": "idle"}
        self.job = {"status": "idle"}
        self._job_counter = 0
        super().__init__(address, WorkbenchHandler)

    def snapshot(self):
        getter = getattr(self.workbench, "state", None)
        if getter is None:
            getter = getattr(self.workbench, "get_state", {})
        state = getter() if callable(getter) else getter
        # Copy through JSON rather than exposing a mutable backend dictionary.
        result = json.loads(json.dumps(state or {}, default=_json_default, allow_nan=False))
        with self.job_lock:
            job = deepcopy(self.job)
            busy = self.current_future is not None and not self.current_future.done()
            export_job = deepcopy(self.export_job)
            analysis_job = deepcopy(self.analysis_job)
        result["busy"] = busy
        backend_job=result.get("job") or {}
        result["job"] = dict(backend_job, **job) if job.get('status')!='idle' else dict(job,**backend_job)
        if not busy and job.get('status')=='idle':
            solver_step=next((s for s in result.get('steps',[]) if s.get('number')==5),{})
            if solver_step.get('status') in ('passed','stopped_with_result'):
                result['job']['status']='completed' if solver_step['status']=='passed' else 'stopped_with_result'
        result["export_job"] = export_job
        result["analysis_job"] = analysis_job
        if job.get("status") == "failed":
            result["error"] = job.get("error")
        for step in result.get("steps", []):
            if step.get("number") == job.get("step"):
                if busy:
                    step.update(status="running", message="正在运行")
                elif job.get("status") == "failed":
                    step.update(status="error", message=job.get("error", "运行失败"))
        return result

    def start_export(self, body):
        exporter=getattr(self.workbench,'export_current',None)
        if not callable(exporter):
            raise NotImplementedError('此后端尚不支持导出当前最优解')
        if not isinstance(body.get('compile',False),bool):
            raise ValueError('compile 必须为 true 或 false')
        with self.job_lock:
            if self.export_future is not None and not self.export_future.done():
                raise RuntimeError('当前快照正在导出，请等待完成。')
            self.export_job={'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
            options=dict(body);options.setdefault('compile',False)
            self.export_future=self.export_executor.submit(self._execute_export,exporter,options)
            return deepcopy(self.export_job)

    def start_analysis(self, body):
        analyzer=getattr(self.workbench,'analyze',None)
        if not callable(analyzer):raise NotImplementedError('此后端尚不支持工程分析')
        if set(body)-{'design','source_label','analysis_options'}:raise ValueError('未知工程分析请求字段')
        if body.get('design') is not None and not isinstance(body['design'],dict):raise ValueError('design必须为设计对象')
        if not isinstance(body.get('source_label',''),str):raise ValueError('source_label必须为文本')
        options=body.get('analysis_options') or {}
        if not isinstance(options,dict) or set(options)-{'mutual_demag','tolerance'} or any(not isinstance(v,bool) for v in options.values()):
            raise ValueError('分析选项仅允许mutual_demag/tolerance布尔开关')
        with self.job_lock:
            if any(future is not None and not future.done() for future in (self.current_future,self.export_future,self.analysis_future)):
                raise RuntimeError('已有求解、导出或工程分析正在运行')
            self.analysis_job={'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
            self.analysis_future=self.analysis_executor.submit(self._execute_analysis,analyzer,body)
            return deepcopy(self.analysis_job)

    def _execute_analysis(self, analyzer, body):
        try:result=analyzer(**body)
        except Exception as error:
            with self.job_lock:self.analysis_job.update(status='failed',error=str(error))
        else:
            with self.job_lock:self.analysis_job.update(status='completed',design_sha256=result.get('design_sha256'))
        finally:
            with self.job_lock:self.analysis_job['finished_at']=datetime.now(timezone.utc).isoformat()

    def _execute_export(self,exporter,options):
        try:
            result=exporter(**options)
        except Exception as error:
            with self.job_lock:self.export_job.update(status='failed',error=str(error))
        else:
            with self.job_lock:self.export_job.update(status='completed',result=result)
        finally:
            with self.job_lock:self.export_job['finished_at']=datetime.now(timezone.utc).isoformat()

    def start_step(self, number, body):
        with self.job_lock:
            if self.analysis_future is not None and not self.analysis_future.done():raise RuntimeError('工程分析运行期间不能启动求解')
            if self.current_future is not None and not self.current_future.done():
                raise RuntimeError("已有工序正在运行，请等待完成或先取消。")
            self._job_counter += 1
            job_id = self._job_counter
            self.job = {"id": job_id, "step": number, "status": "running",
                        "started_at": datetime.now(timezone.utc).isoformat()}
            self.current_future = self.executor.submit(self._execute, number, body, job_id)
            return deepcopy(self.job)

    def _execute(self, number, body, job_id):
        try:
            result=self.workbench.run_step(number, **body)
        except Exception as error:
            with self.job_lock:
                self.job.update(status="failed", error=str(error))
        else:
            with self.job_lock:
                terminal=(result.get('state') or result.get('status')) if isinstance(result,dict) else None
                if terminal!='stopped_with_result':
                    state=self.workbench.state() if callable(getattr(self.workbench,'state',None)) else {}
                    terminal=next((step.get('status') for step in state.get('steps',[]) if step.get('number')==number),terminal)
                self.job.update(status='stopped_with_result' if terminal=='stopped_with_result' else "cancelled" if isinstance(result,dict) and result.get('cancelled') else "completed")
        finally:
            with self.job_lock:
                self.job["finished_at"] = datetime.now(timezone.utc).isoformat()

    def server_close(self):
        super().server_close()
        # Do not terminate an executing solver thread. Cooperative cancellation
        # is delegated to the backend's cancel method when the user requests it.
        self.executor.shutdown(wait=False)
        self.export_executor.shutdown(wait=False)
        self.analysis_executor.shutdown(wait=False)


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "MRIWorkbench/1.0"

    def log_message(self, format, *args):
        # Polling every second should not flood the solver console.
        return

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def _json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, default=_json_default,
                          allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _local_request(self):
        """Reject foreign hosts/origins even though the listener is loopback."""
        expected_port = self.server.server_address[1]
        valid_hosts = {"127.0.0.1", "localhost"}
        host = urlsplit("//" + self.headers.get("Host", "")).hostname
        if host not in valid_hosts:
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                return (parsed.scheme == "http" and parsed.hostname in valid_hosts
                        and (parsed.port or 80) == expected_port)
            except ValueError:
                return False
        return True

    def _body(self):
        try:
            count = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("无效的请求长度")
        if count < 0 or count > 1024 * 1024:
            raise ValueError("配置请求不得超过 1 MB")
        if not count:
            return {}
        data = json.loads(self.rfile.read(count).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求内容必须是 JSON 对象")
        return data

    def _file(self, relative):
        relative = unquote(relative)
        if not relative or "\\" in relative or "\x00" in relative:
            self._json({"error": "无效的文件路径"}, 400)
            return
        path = (self.server.project_dir / relative).resolve()
        try:
            path.relative_to(self.server.project_dir)
        except ValueError:
            self._json({"error": "文件必须位于当前工作台工程目录内"}, 403)
            return
        if not path.is_file():
            self._json({"error": "文件不存在"}, 404)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/"):
            content_type += "; charset=utf-8"
        self._headers(200, content_type, len(data))
        self.wfile.write(data)

    def do_GET(self):
        if not self._local_request():
            self._json({"error": "只允许本机访问"}, 403)
            return
        route = urlsplit(self.path).path
        try:
            if route == "/api/state":
                self._json(self.server.snapshot())
            elif route in ("/", "/index.html"):
                data = Path(__file__).with_name("workbench.html").read_bytes()
                self._headers(200, "text/html; charset=utf-8", len(data))
                self.wfile.write(data)
            elif route in VIEWPORT_ASSETS:
                # Every renderer dependency is local and explicitly allowlisted.
                asset = Path(__file__).parent / VIEWPORT_ASSETS[route]
                if not asset.is_file():
                    self._json({'error': '本地三维渲染资源不存在'}, 404)
                    return
                data = asset.read_bytes()
                mime = 'application/javascript' if asset.suffix == '.js' else 'text/plain'
                self._headers(200, mime + '; charset=utf-8', len(data))
                self.wfile.write(data)
            elif route == '/api/presets':
                self._json({'presets':[{'id':'engineering_100mm','label':'100mm DSV · 16层工程案例'},
                                       {'id':'default_20mm','label':'默认20mm球体'}]})
            elif route.startswith("/files/"):
                self._file(route[len("/files/"):])
            else:
                self._json({"error": "地址不存在"}, 404)
        except Exception as error:
            self._json({"error": str(error)}, 500)

    def do_POST(self):
        if not self._local_request():
            self._json({"error": "只允许同源本机请求"}, 403)
            return
        route = urlsplit(self.path).path
        try:
            body = self._body()
            with self.server.job_lock:
                busy = (self.server.current_future is not None
                        and not self.server.current_future.done())
                analysis_busy=self.server.analysis_future is not None and not self.server.analysis_future.done()
                if analysis_busy and route!='/api/cancel':
                    self._json({'error':'工程分析正在运行；请等待完成后再修改或启动其他任务'},409)
                    return
                if busy and route not in ("/api/cancel","/api/export-current"):
                    self._json({"error": "求解运行期间不能修改工程；请等待或取消。"}, 409)
                    return
                wb = self.server.workbench
                if route == "/api/config":
                    config = body.get("config")
                    if not isinstance(config, dict):
                        raise ValueError("config 必须是配置对象")
                    result = wb.update_config(config)
                    self.server.job = {"status": "idle"}
                elif route == "/api/preflight":
                    result = wb.preflight()
                elif route == '/api/solver-preflight':
                    result = wb.solver_preflight()
                elif route == '/api/apply-budget-suggestion':
                    result = wb.apply_budget_suggestion()
                    self.server.job = {'status':'idle'}
                elif route == '/api/preset':
                    result = wb.load_preset(body.get('id'))
                    self.server.job = {'status':'idle'}
                elif route == "/api/select":
                    ids = body.get("ids")
                    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                        raise ValueError("ids 必须是种子编号列表")
                    result = wb.select_seeds(ids)
                elif route == "/api/cancel":
                    result = wb.cancel()
                elif route == '/api/export-current':
                    export_job=self.server.start_export(body)
                    self._json({'export_job':export_job},202)
                    return
                elif route == '/api/analysis':
                    analysis_job=self.server.start_analysis(body)
                    self._json({'analysis_job':analysis_job},202)
                    return
                elif route.startswith("/api/step/"):
                    number = int(route[len("/api/step/"):])
                    if not 1 <= number <= 6:
                        raise ValueError("工序编号必须在 1 到 6 之间")
                    job = self.server.start_step(number, body)
                    self._json({"job": job}, 202)
                    return
                else:
                    self._json({"error": "地址不存在"}, 404)
                    return
            self._json({"result": result, "state": self.server.snapshot()})
        except NotImplementedError as error:
            self._json({'error':str(error)},501)
        except (ValueError, TypeError, KeyError) as error:
            self._json({"error": str(error)}, 400)
        except RuntimeError as error:
            self._json({"error": str(error)}, 409)
        except Exception as error:
            self._json({"error": str(error)}, 500)


def create_server(project_dir, config_path=None, port=8765, workbench=None):
    """Create (but do not run) a loopback server, optionally injecting a backend."""
    if workbench is None:
        from .workbench import Workbench
        workbench = Workbench(project_dir, config_path=config_path)
    return WorkbenchHTTPServer(("127.0.0.1", int(port)), workbench, project_dir)


def serve(project_dir, config_path=None, port=8765):
    server = create_server(project_dir, config_path=config_path, port=port)
    print("MRI 仿真工作台：http://127.0.0.1:%d" % server.server_address[1], flush=True)
    print("仅本机访问。按 Ctrl+C 关闭工作台服务。", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        server.workbench.cancel()
    finally:
        server.server_close()


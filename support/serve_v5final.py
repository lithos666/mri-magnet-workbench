"""Run the installed, fixed-source V5 in its own interpreter and project."""
from pathlib import Path
import argparse,importlib.util,json,sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--installation',required=True,type=Path)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--port',default=8764,type=int)
    args=parser.parse_args()
    canonical=Path(__file__).resolve().parents[1]
    installation=args.installation.resolve();app=installation/'app'
    if not (app/'mri_magnet'/'workbench_server.py').is_file():raise ValueError('Install V5final before opening V5')
    sys.path.insert(0,str(app))
    # Import the original V5 modules after selecting its immutable source tree.
    from mri_magnet.workbench import Workbench
    from mri_magnet.workbench_server import WorkbenchHandler,create_server
    import mri_magnet.workbench_server as server_module
    if not Path(server_module.__file__).resolve().is_relative_to(app):
        raise RuntimeError('V5 must use its own backend')
    spec=importlib.util.spec_from_file_location('mri_version_navigation',canonical/'mri_magnet'/'version_navigation.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    original_get=WorkbenchHandler.do_GET
    from urllib.parse import urlsplit
    def page_with_navigation(handler):
        if urlsplit(handler.path).path in ('/','/index.html') and handler._local_request():
            page=(app/'mri_magnet'/'workbench.html').read_text(encoding='utf-8')
            data=helper.inject_version_navigation(page,canonical/'mri_magnet'/'workbench.html','v5').encode('utf-8')
            handler._headers(200,'text/html; charset=utf-8',len(data));handler.wfile.write(data)
        else:original_get(handler)
    WorkbenchHandler.do_GET=page_with_navigation
    project=(args.project or app/'results'/'V5_版本切换工作台').resolve()
    config=app/'V5无退磁_100mmDSV.json'
    workbench=Workbench(project,config if not (project/'工作台状态.json').exists() else None)
    server=create_server(project,port=args.port,workbench=workbench)
    print(json.dumps({'version':'V5final','port':args.port,'project':str(project),
        'backend_source':str(Path(server_module.__file__).resolve()),
        'Br_T':workbench.config['magnet']['remanence_T'],
        'polarization_model':workbench.config.get('polarization_model')},ensure_ascii=False),flush=True)
    try:server.serve_forever(poll_interval=.5)
    finally:server.server_close()


if __name__=='__main__':main()

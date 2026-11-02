"""CLI operations on the same persistent project used by the local web workbench."""
import argparse
import json
from pathlib import Path
from .workbench import Workbench
from .workbench_config import default_config
from .serialization import jsonable


def main(argv=None):
    parser=argparse.ArgumentParser(description='MRI 六步仿真工作台（旧 check/all/optimize 命令保持兼容）')
    parser.add_argument('action',choices=['serve','init','status','preflight','solver-preflight','apply-budget-suggestion','preset','step','select','all'])
    parser.add_argument('number',nargs='?',type=int,help='step 的步骤号1～6')
    parser.add_argument('--project',default='results/工作台工程',help='独立工程目录，保存配置、状态和每次运行')
    parser.add_argument('--config',help='工作台 JSON 配置；不使用旧 CONFIG 的平面域隐式值')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--preset',choices=['engineering_100mm','default_20mm'])
    parser.add_argument('--auto',action='store_true',help='step 时自动补跑缺失的前置工序')
    parser.add_argument('--no-compile',action='store_true',help='只导出 Java；明确记录 CLASS 未编译')
    parser.add_argument('--no-comsol',action='store_true',help='仅归档 Python 结果')
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    project=Path(args.project);project=project if project.is_absolute() else root/project
    config=Path(args.config).resolve() if args.config else None
    if args.action=='serve':
        from .workbench_server import serve
        return serve(project,config_path=config,port=args.port) or 0
    wb=Workbench(project,config_path=config)
    if args.action=='init':
        from .serialization import save_json
        save_json(project/'工作台配置.json',wb.config)
        result={'project':str(project),'config':str(project/'工作台配置.json'),'next':'workbench step 1 --project ...'}
    elif args.action=='status':result=wb.state()
    elif args.action=='preflight':result=wb.preflight()
    elif args.action=='solver-preflight':result=wb.solver_preflight()
    elif args.action=='apply-budget-suggestion':result=wb.apply_budget_suggestion()
    elif args.action=='preset':
        if not args.preset:parser.error('preset 需要 --preset 工程预设名称')
        result=wb.load_preset(args.preset)
    elif args.action=='select':
        if not args.ids:parser.error('select 需要 --ids 种子ID ...')
        result=wb.select_seeds(args.ids)
    else:
        if args.action=='step' and args.number is None:parser.error('step 需要步骤号')
        last=6 if args.action=='all' else args.number
        numbers=range(1,last+1) if args.auto or args.action=='all' else [last]
        result={}
        for number in numbers:
            if len(numbers)>1 and wb.data['steps'][number-1]['status'] in ('passed','stopped_with_result'):continue
            options={'export_comsol':not args.no_comsol,'compile':False if args.no_compile else None} if number==6 else {}
            result=wb.run_step(number,**options)
            print('Step %d: %s'%(number,result.get('message','')),flush=True)
            if result.get('cancelled') or result.get('state')=='stopped_with_result':break
    print(json.dumps(jsonable(result),ensure_ascii=False,allow_nan=False,indent=2))
    return 0

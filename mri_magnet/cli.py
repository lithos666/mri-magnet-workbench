"""Single configurable command entry; manufacturing toggles checks, never algorithms."""
import argparse
import copy
import csv
import json
import runpy
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from .serialization import jsonable, save_json
from .sampling import line_points, plane_points, disk_samples
from .comsol_bridge import read_field_csv, write_field_csv, compare_field_tables, prepare_existing_solution_export, run_existing_solution_export

ROOT=Path(__file__).resolve().parents[1]

def log(message): print(message,flush=True)

def settings(args):
    from .config_schema import canonicalize_config, legacy_view
    from .legacy_defaults import legacy_runtime_options
    path=Path(getattr(args,'parameters','parameters.py'))
    if not path.is_absolute():path=ROOT/path
    namespace=runpy.run_path(str(path))
    raw=copy.deepcopy(namespace.get('WORKBENCH_CONFIG',namespace.get('CONFIG',{})))
    legacy_input='WORKBENCH_CONFIG' not in namespace and 'target' not in raw
    canonical=canonicalize_config(raw)
    runtime=copy.deepcopy(namespace.get('CONFIG',legacy_runtime_options()))
    if args.config:
        override_path=Path(args.config)
        if not override_path.is_absolute():override_path=ROOT/override_path
        override=json.loads(override_path.read_text(encoding='utf-8-sig'))
        canonical=canonicalize_config(override,base=canonical)
        legacy_input='target' not in override and 'fov_diameter_mm' in override
        if legacy_input and 'mode' not in override:canonical['mode']='seeded'
        from .workbench_config import merge_config
        runtime=merge_config(runtime,override)
    if args.source:canonical.update(source_geometry=str(Path(args.source).resolve()),mode='seeded')
    if args.mph:canonical['comsol_model']=str(Path(args.mph).resolve())
    for name in ('source_geometry','comsol_model'):
        value=canonical.get(name,'')
        if value:
            target=Path(value);canonical[name]=str(target if target.is_absolute() else (ROOT/target).resolve())
    if args.quick:
        canonical['sampling']['preset']='quick'
        canonical['solver']['max_evaluations']=runtime.get('quick',{}).get('max_evaluations',canonical['solver']['max_evaluations'])
    if args.budget is not None:canonical['solver']['max_evaluations']=args.budget
    canonical=canonicalize_config(canonical)
    config=legacy_view(canonical,runtime)
    config['_use_workbench_workflow']=not legacy_input
    config['parameters_file']=str(path.resolve())
    config['source_geometry']=canonical.get('source_geometry','')
    config['comsol_model']=canonical.get('comsol_model',runtime.get('comsol_model',''))
    if args.quick and 'topology_limit' in config.get('quick',{}):
        config['search']['topology_limit']=config['quick']['topology_limit']
    return config


def run_canonical_command(args,config,public):
    """Normal UI and CLI share the persistent workbench and one domain schema."""
    from .workbench import Workbench,requirements_for
    from .identity import canonical_physical_sha256
    cfg=copy.deepcopy(config['canonical_config'])
    if getattr(args,'fixed_topology',False):
        cfg['search'].update(optimize_azimuth_count=False,optimize_radial_ring_count=False,optimize_magnetization=False)
    wb=Workbench(public);wb.update_config(cfg)
    last=2 if args.command=='check' else 4 if args.command=='baseline' else 6
    for number in range(1,last+1):
        result=wb.run_step(number);log('Step %d: %s'%(number,result.get('message','')))
        if result.get('state')=='stopped_with_result' or result.get('cancelled'):return 0
    if args.command=='baseline':
        records=[]
        for index,seed in enumerate(wb._selected(),1):
            records.append({'candidate_id':seed['id'],'topology_index':index,'design':seed['design'],
                'metrics':seed.get('training_metrics',seed['metrics']), 'search_stage':'synthesis',
                'canonical_physical_sha256':canonical_physical_sha256(seed['design'])})
        save_json(wb._run_root()/'详细数据'/'优化结果.json',{'state':'baseline_only','records':records,
            'metadata':{'algorithm':'baseline_only_no_optimization','field_evaluations':0,'max_evaluations':0,'stage_summaries':[]}})
        save_json(wb._run_root()/'详细数据'/'演化索引.json',{'enabled':False,'reason':'baseline_only_no_optimization'})
        wb._status(5,'passed','仅基线评价，未执行优化')
        wb.run_step(6)
    log('工作台结果：'+str(wb._run_root()))
    return 0

def export_design(path,design):
    save_json(path.with_suffix('.json'),design.to_dict())
    arrays=design.magnet_arrays()
    with path.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['Magnet_ID','x_mm','y_mm','z_mm','radial_angle_deg','N_x','N_y','N_z','cube_mm','polarization_T'])
        for ident,p,a,n in zip(arrays['ids'],arrays['positions_mm'],arrays['radial_angles_deg'],arrays['north_vectors']):
            w.writerow([ident,*p,a,*n,design.cube_mm,design.remanence_T])

def plot_data(design,engine,diameter_mm,quick,plot_settings=None,z_mm=0.0):
    result={'axis_lines':[],'planes':[],'geometry':'disk','z_mm':float(z_mm)}
    plot_settings=plot_settings or {}
    axis_count=plot_settings.get('quick_axis_line_points' if quick else 'axis_line_points',121 if quick else 201)
    plane_count=plot_settings.get('quick_plane_points' if quick else 'plane_points',31 if quick else 51)
    for axis in 'xy':
        coordinate,points=line_points(axis,diameter_mm/2,n=axis_count)
        points[:,2]=float(z_mm)
        result['axis_lines'].append({'axis':axis,'coordinate_mm':coordinate,'B_T':engine.field(design,points)})
    for plane in ('xy',):
        u,v,points,mask=plane_points(plane,diameter_mm/2,n=plane_count)
        points[:,:,2]=float(z_mm)
        field=np.full(points.shape,np.nan);field[mask]=engine.field(design,points[mask])
        result['planes'].append({'plane':plane,'u_mm':u,'v_mm':v,'B_T':field,'mask':mask})
    return result

def compare_reference(baseline,engine,path,out,coord_unit=None,field_unit=None,fov_diameter_mm=None,z_mm=0.0):
    reference=read_field_csv(path,coordinate_unit=coord_unit,field_unit=field_unit)
    if fov_diameter_mm is not None:
        points=np.asarray(reference['points_mm'],dtype=float)
        if (not np.allclose(points[:,2],z_mm,rtol=0,atol=1e-7)
                or np.any(np.linalg.norm(points[:,:2],axis=1)>fov_diameter_mm/2+1e-7)):
            raise ValueError('参考CSV含当前平面圆盘FOV以外的坐标；请按同一z平面和直径重新导出')
    predicted={'points_mm':reference['points_mm'],'fields_T':engine.field(baseline,reference['points_mm'])}
    result=compare_field_tables(predicted,reference,coordinate_tolerance_mm=1e-6)
    write_field_csv(out/'python_at_comsol_points.csv',predicted['points_mm'],predicted['fields_T'])
    save_json(out/'comsol_comparison.json',result)
    return result

def compare_cached_line(baseline,engine,audit,out):
    """Use only hash-matched saved 1D extrema; this is never 3D FEM calibration."""
    table_path=ROOT/'support'/'comsol'/'validation'/'existing_line_tables.json'
    if not table_path.exists():return None
    cached=json.loads(table_path.read_text(encoding='utf-8'))
    if cached.get('source_sha256')!=audit.get('source_sha256'):return None
    coordinate,points=line_points('x',15.,301)
    B=engine.field(baseline,points)
    py_min=float(B[:,2].min()*1e3);py_max=float(B[:,2].max()*1e3)
    result={'scope':'历史x=-15至15mm,y=z=0截线极值；范围超出当前20mm圆盘，不参与新FOV指标',
        'source_mph_sha256':cached['source_sha256'],'python_min_mT':py_min,'python_max_mT':py_max,
        'comsol_saved_min_mT':cached['line_Bz_min_mT'],'comsol_saved_max_mT':cached['line_Bz_max_mT'],
        'minimum_bias_percent':(py_min/cached['line_Bz_min_mT']-1)*100,
        'maximum_bias_percent':(py_max/cached['line_Bz_max_mT']-1)*100,
        'comsol_mean_available':False,'point_export_status':'此前尝试被COMSOL文件系统访问首选项拒绝；未更改安全设置'}
    save_json(out/'cached_comsol_line_comparison.json',result)
    return result


def main(argv=None):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='MRI：有序独立几何、拓扑分层与严格场强门槛优化')
    parser.add_argument('command',choices=['check','baseline','scan','optimize','all','benchmark','compare','comsol-export','export'])
    parser.add_argument('--parameters',default='parameters.py')
    parser.add_argument('--config');parser.add_argument('--source');parser.add_argument('--mph')
    parser.add_argument('--out');parser.add_argument('--quick',action='store_true');parser.add_argument('--budget',type=int)
    parser.add_argument('--run',help='已有结果目录，仅补导出')
    parser.add_argument('--seeds',type=int,nargs='+',help='benchmark使用的优化随机种子；采样点集保持固定')
    parser.add_argument('--fixed-topology',action='store_true')
    parser.add_argument('--reference');parser.add_argument('--coordinate-unit');parser.add_argument('--field-unit')
    args=parser.parse_args(argv);config=settings(args)
    if args.command=='export':
        if not args.run:parser.error('export需要 --run results/本次目录')
        from .deliverables import export_saved_run
        path=Path(args.run)
        return export_saved_run(path if path.is_absolute() else ROOT/path,config,log)
    public=Path(args.out).resolve() if args.out else ROOT/'results'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    if config.get('_use_workbench_workflow'):
        if args.command=='check' and config['canonical_config']['mode']=='seeded':
            # Seeded geometry checking also projects/audits the actual source,
            # without computing any field; de novo has no source to project.
            from .workflow import run_manufacturing_workflow
            details=public/'详细数据';details.mkdir(parents=True,exist_ok=True)
            return run_manufacturing_workflow(args,config,details,sys.modules[__name__],report_root=public)
        if args.command in ('check','baseline','optimize','all'):return run_canonical_command(args,config,public)
        if args.command in ('scan','benchmark','compare','comsol-export'):
            cfg=config['canonical_config']
            if cfg['target']['geometry']!='disk' or cfg['mode']!='seeded':
                raise ValueError('历史扫描/对照接口需要显式 seeded + disk 配置；球形DSV和正向生成请使用六步工作台，避免静默切换评价域')
    details=public/'详细数据';details.mkdir(parents=True,exist_ok=True)
    if args.command=='benchmark':
        from .benchmark import run_benchmark
        result=run_benchmark(config,public,quick=args.quick,seeds=args.seeds,budget=args.budget)
        log('固定拓扑／开放拓扑对照报告：'+result['report_html']);return 0
    if args.command in ('check','baseline','scan','optimize','all'):
        from .workflow import run_manufacturing_workflow
        status=run_manufacturing_workflow(args,config,details,sys.modules[__name__],report_root=public)
        if args.command in ('optimize','all'):
            from .run_naming import finish_run_name
            actual=finish_run_name(public,config,explicit_output=args.out is not None)
            if actual!=public:log('指标命名后的结果目录：'+str(actual))
        return status
    from .geometry import load_source
    from .field import FieldEngine
    baseline,_=load_source(config['source_geometry'])
    engine=FieldEngine(config['chunk_points'],config['polarization_model'])
    if args.command=='compare':
        if not args.reference:parser.error('compare需要 --reference')
        result=compare_reference(baseline,engine,args.reference,details,args.coordinate_unit,args.field_unit,
                                 fov_diameter_mm=config['fov_diameter_mm'],z_mm=config['fov_z_mm'])
        log(json.dumps(jsonable(result),ensure_ascii=False));return 0
    sample=disk_samples(config['fov_diameter_mm'],32,48,config['seed']+71,z_mm=config['fov_z_mm'])
    job=prepare_existing_solution_export(config['comsol_model'],sample['points_mm'],details/'comsol',config['comsol_root'])
    result=run_existing_solution_export(job,timeout=240)
    save_json(details/'comsol_export_result.json',result);log(str(result))
    return 0 if result.get('status')=='success' else 2

if __name__=='__main__':raise SystemExit(main())

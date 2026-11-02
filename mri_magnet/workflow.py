"""One configurable workflow for geometry, epsilon search and independent validation."""
import copy
import itertools
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
import magpylib

from .geometry import Design, load_source, envelope_limits, validate_design
from .manufacturing import audit_manufacturing, make_manufacturing_seed, radial_packing_bounds, column_partition_check
from .parameterization import get_search_space, packing_for_design
from .field import FieldEngine
from .metrics import evaluate_design
from .sampling import disk_samples
from .validation import refine_disk_extrema
from .optimization import run_topology_optimization, pareto_front, _design_key
from .objectives import resolve_requirements, check_performance, feasibility_score
from .validation_status import validation_status
from .identity import raw_design_sha256, canonical_physical_sha256
from .serialization import save_json
from .comsol_bridge import inspect_mph, write_field_csv


def _limits(source, config):
    result = envelope_limits(source); envelope = config['envelope']
    result.update(bore_radius_mm=envelope['protected_bore_diameter_mm']/2,
        outer_radius_mm=envelope['max_outer_diameter_mm']/2, half_length_mm=envelope['max_length_mm']/2,
        preserve_exact_bore=False, min_gap_mm=0.)
    if result['outer_radius_mm'] <= result['bore_radius_mm'] or result['half_length_mm'] <= 0:
        raise ValueError('外形边界必须大于保护通道，轴向长度必须为正')
    return result


def _design_check(design, process, limits):
    geometry = validate_design(design, limits)
    manufacture = audit_manufacturing(design, process, limits)
    return {'valid': geometry['valid'] and manufacture['valid'],
        'violations': geometry['violations']+manufacture['violations'],
        'geometry': geometry, 'manufacturing': manufacture}


def _prepare_seed(source, process, limits, search, field_error=None):
    """Enumerate retained source subsets, project nearest geometry, optionally compare vector fields."""
    layer_count=len({r.layer_index for r in source.rings})
    groups = {layer: sorted([r for r in source.rings if r.layer_index == layer], key=lambda r:r.radius_mm)
              for layer in range(layer_count)}
    if layer_count<2 or layer_count%2 or any(not rings for rings in groups.values()):
        raise ValueError('起点需要连续且完整的偶数镜像层源排布')
    pair_count=layer_count//2
    raw_max,raw_min=search['max_radial_rings_per_layer'],search['min_radial_rings_per_layer']
    capacity=packing_for_design(source,process,limits)['maximum_radial_rings']
    if raw_max is None:raw_max=capacity
    if any(isinstance(x,bool) or int(x)!=x for x in (raw_max,raw_min)):
        raise ValueError('径向环数必须是整数')
    maximum = int(raw_max); minimum = int(raw_min)
    if minimum < 1 or maximum < minimum: raise ValueError('径向环数范围无效')
    maximum = min(maximum, capacity)
    counts = sorted(set(search['azimuth_count_candidates']), key=lambda n:(abs(n-source.rings[0].count),n))
    if not search['optimize_azimuth_count']: counts = [source.rings[0].count]
    caps = range(maximum, minimum-1, -1) if search['optimize_radial_ring_count'] else [maximum]
    attempts, last = [], None
    for cap in caps:
        catalogs = []
        for pair in range(pair_count):
            n = min(cap, len(groups[pair]))
            if n < minimum:
                raise ValueError('源槽数量不足以满足最小径向环数；不代表其他新拓扑无解')
            catalogs.append(list(itertools.combinations(groups[pair], n)))
        for columns in counts:
            if not column_partition_check(columns,process)['valid']: continue
            proposals = []
            candidate_limit = int(search.get('seed_subset_max_candidates',64))
            if candidate_limit<1 or candidate_limit!=search.get('seed_subset_max_candidates',64):
                raise ValueError('seed_subset_max_candidates必须是正整数')
            catalog_size=math.prod(len(catalog) for catalog in catalogs)
            for combination in itertools.islice(itertools.product(*catalogs), candidate_limit):
                rings = []
                for pair, selected in enumerate(combination):
                    other = {r.slot_index:r for r in groups[layer_count-1-pair]}
                    for ring in selected:
                        for item in (ring, other[ring.slot_index]):
                            phase = item.phase_deg if columns == item.count else item.phase_deg-180/item.count+180/columns
                            rings.append(replace(item,count=int(columns),phase_deg=phase))
                trial_source = replace(source,rings=tuple(sorted(rings,key=lambda r:(r.layer_index,r.slot_index))))
                seed, report = make_manufacturing_seed(trial_source,process,limits)
                last = (seed,report)
                item = {'radial_cap':cap,'columns':columns,'magnet_count':seed.magnet_count,
                    'retained_source_slot_ids':[r.id for r in seed.rings], 'valid':report['valid']}
                if report['valid']:
                    source_by_id = {r.id:r for r in source.rings}
                    displacement = sum((r.radius_mm-source_by_id[r.id].radius_mm)**2+
                                       (r.z_mm-source_by_id[r.id].z_mm)**2 for r in seed.rings)
                    error = float(field_error(seed)) if field_error else displacement
                    item.update(selection_error=error, squared_geometry_displacement_mm2=displacement)
                    proposals.append((error,seed,report))
                attempts.append(item)
            if proposals:
                _, seed, report = min(proposals,key=lambda item:item[0])
                report.update(source_magnet_count=source.magnet_count,source_ring_count=len(source.rings),
                    removed_source_slot_ids=[r.id for r in source.rings if r.id not in {s.id for s in seed.rings}],
                    seed_construction_attempts=attempts, seed_field_evaluations=sum(r['valid'] for r in attempts) if field_error else 0,
                    seed_catalog_size=catalog_size,seed_catalog_examined=min(catalog_size,candidate_limit),
                    seed_catalog_truncated=catalog_size>candidate_limit,
                    seed_selection='fixed-point vector field MSE' if field_error else 'minimum squared source geometry displacement',
                    column_count_changed_from_source=columns!=source.rings[0].count)
                return seed, report
    if last:
        last[1]['seed_construction_attempts']=attempts
        return last
    raise ValueError('没有构造出满足环数、数量与扇区要求的起点')


def _exploration(variables,builder,evaluator,checker,settings,out,util):
    initial=np.array([v['initial'] for v in variables]); continuous=[i for i,v in enumerate(variables) if not v['integer']]
    indices=[]
    for prefix in ('radial_slack_pair_5','axial_slack_pair_','radial_slack_pair_2'):
        index=next((i for i in continuous if variables[i]['name'].startswith(prefix)),None)
        if index is not None and index not in indices: indices.append(index)
    if not indices: indices=continuous[:3]
    scans=[]
    for index in indices:
        var=variables[index]; rows=[]
        values=np.unique(np.r_[np.linspace(var['lower'],var['upper'],settings['points_per_curve']),var['initial']])
        for value in values:
            vector=initial.copy();vector[index]=value;design=builder(vector);check=checker(design)
            rows.append({'value':float(value),'valid':check['valid'],'metrics':evaluator(design) if check['valid'] else {},
                         'geometry_check':check,'magnet_count':design.magnet_count})
        scans.append({'parameter':var['name'],'parameter_label':var['name']+'（可行余量坐标）','rows':rows})
        util.log('完成可行坐标扫描：'+var['name'])
    surface=None
    if len(continuous)>=2:
        ix,iy=(indices+[i for i in continuous if i not in indices])[:2]
        xs=np.unique(np.r_[np.linspace(0,1,settings['response_points_per_axis']),initial[ix]])
        ys=np.unique(np.r_[np.linspace(0,1,settings['response_points_per_axis']),initial[iy]])
        mean=np.full((len(ys),len(xs)),np.nan);ppm=mean.copy();rejections=[]
        for row,y in enumerate(ys):
            for col,x in enumerate(xs):
                vector=initial.copy();vector[ix]=x;vector[iy]=y;design=builder(vector);check=checker(design)
                if check['valid']:
                    m=evaluator(design);mean[row,col]=m['mean_norm_T'];ppm[row,col]=m['pp_norm_ppm']
                else: rejections.append({'x':float(x),'y':float(y),'violations':check['violations']})
        surface={'x_name':variables[ix]['name'],'y_name':variables[iy]['name'],'x_values':xs,'y_values':ys,
                 'mean_norm_T':mean,'pp_norm_ppm':ppm,'rejections':rejections}
        save_json(out/'response_surface.json',surface)
    base=evaluator(builder(initial)); sensitivity=[]
    for index in continuous:
        var=variables[index];span=var['upper']-var['lower'];step=span*settings['sensitivity_relative_step'];values=[]
        for sign in (-1,1):
            vector=initial.copy();vector[index]=np.clip(initial[index]+sign*step,var['lower'],var['upper'])
            design=builder(vector)
            if checker(design)['valid']: values.append((vector[index],evaluator(design)))
        if len(values)==2 and values[1][0]>values[0][0]:
            delta=values[1][0]-values[0][0]
            db=(values[1][1]['mean_norm_T']-values[0][1]['mean_norm_T'])/delta
            dp=(values[1][1]['pp_norm_ppm']-values[0][1]['pp_norm_ppm'])/delta
            sensitivity.append({'parameter':var['name'],'step_low':values[0][0],'step_high':values[1][0],
                'd_mean_mT_per_parameter':db*1e3,'d_ppm_per_parameter':dp,
                'normalized_strength_sensitivity':db*span/base['mean_norm_T'],
                'normalized_ppm_sensitivity':dp*span/base['pp_norm_ppm']})
    save_json(out/'scans.json',scans);save_json(out/'sensitivity.json',{'baseline':base,'derivatives':sensitivity})
    return scans,surface,sensitivity


def _select_validation_candidates(records, max_candidates, requirements, excluded_physical_keys=()):
    """Select a bounded, physically unique dense-validation set without field calls.

    Preserve the global best eligible design first. Explicit topology champions
    follow in the same strict feasibility-score order, then fill globally. Old
    records without topology labels retain their global ranking behavior.
    Coverage is limited by max_candidates and is recorded, never promised.
    """
    if (isinstance(max_candidates,bool) or int(max_candidates)!=max_candidates
            or max_candidates<0):
        raise ValueError('max_candidates must be a non-negative integer')
    maximum=int(max_candidates);excluded=set(excluded_physical_keys)
    ranked=sorted(records,key=lambda row:feasibility_score(row['metrics'],requirements))
    eligible=[];seen=set();excluded_count=0;duplicate_count=0
    for row in ranked:
        key=row.get('canonical_physical_sha256')
        if not key:
            design=row['design']
            key=_design_key(design if isinstance(design,Design) else Design.from_dict(design))
        if key in excluded:
            excluded_count+=1;continue
        if key in seen:
            duplicate_count+=1;continue
        seen.add(key);eligible.append((row,key))
    champions={}
    for row,key in eligible:
        topology=row.get('topology_index')
        if topology is not None and topology not in champions:
            champions[topology]=(row,key)
    chosen=[];chosen_keys=set();reasons=[]
    def take(entry,reason):
        row,key=entry
        if len(chosen)>=maximum or key in chosen_keys:return
        chosen.append(row);chosen_keys.add(key)
        reasons.append({'candidate_id':row.get('candidate_id'),
            'topology_index':row.get('topology_index'),'canonical_physical_sha256':key,'reason':reason})
    if eligible:take(eligible[0],'global_best')
    # Python dictionaries preserve insertion order: champions are ordered by
    # their best strict feasibility score, with stable input order for ties.
    for entry in champions.values():take(entry,'topology_champion')
    for entry in eligible:take(entry,'global_rank_fill')
    covered=[]
    for row in chosen:
        topology=row.get('topology_index')
        if topology is not None and topology not in covered:covered.append(topology)
    available=list(champions)
    uncovered=[topology for topology in available if topology not in covered]
    metadata={'strategy':'global_best_then_topology_champions_then_global_fill',
        'champion_priority':'same feasibility_score as optimizer; stable input order breaks ties',
        'max_validation_candidates':maximum,'eligible_unique_candidates':len(eligible),
        'excluded_reference_records':excluded_count,'duplicate_physical_records':duplicate_count,
        'selected_ids':[row.get('candidate_id') for row in chosen],
        'selected_count':len(chosen),'selected_records':reasons,
        'available_topology_indices':available,'covered_topology_indices':covered,
        'uncovered_topology_indices':uncovered,'all_available_topologies_covered':not uncovered,
        'topology_champions':[{'topology_index':topology,'candidate_id':entry[0].get('candidate_id'),
            'selected':entry[1] in chosen_keys} for topology,entry in champions.items()],
        'coverage_scope':'explicit topology_index among eligible unique records; constrained by candidate cap; legacy unlabelled records use global ranking',
        'thresholds_relaxed':False,'extra_search_evaluations':0}
    return chosen,metadata


def _load_warm_starts(config, checker, requirements=None):
    """Prioritize cached same-scope metrics without any ranking field solve.

    Geometry-valid designs with missing/mismatched caches remain available in
    original input order after ranked entries. Cached scores determine loading
    priority only: every inserted design is evaluated again by the optimizer.
    Audit stays in source order; load_rank is one-based across all source runs.
    """
    pending=[];audit=[];source_index=0
    root=Path(config['parameters_file']).resolve().parent
    diameter=float(config.get('fov_diameter_mm',math.nan))
    z=float(config.get('fov_z_mm',0.))
    polarization=config.get('polarization_model','remanence')

    def scope_matches(metrics):
        if not isinstance(metrics,dict):return False
        try:
            return (metrics.get('evaluation_geometry')=='disk'
                and metrics.get('polarization_model')==polarization
                and math.isclose(float(metrics.get('evaluation_diameter_mm',math.nan)),diameter,rel_tol=0.,abs_tol=1e-8)
                and math.isclose(float(metrics.get('evaluation_z_mm',math.nan)),z,rel_tol=0.,abs_tol=1e-8))
        except (ValueError,TypeError):return False

    for raw in config['search'].get('warm_start_runs',[]):
        path=Path(raw);path=path if path.is_absolute() else root/path
        payload_file=path/'详细数据'/'report_payload.json'
        if not payload_file.is_file():
            audit.append({'run':str(raw),'status':'not_found','load_rank':None,
                          'cached_rank_only':False,'metric_scope_matches':False})
            continue
        payload=json.loads(payload_file.read_text(encoding='utf-8'))
        for row in payload.get('candidates',[]):
            design=Design.from_dict(row['design']);check=checker(design)
            cached=None;metric_source=None
            for key in ('training_metrics','validation_metrics','metrics'):
                if scope_matches(row.get(key)):
                    cached=row[key];metric_source=key;break
            score=None
            if cached is not None:
                try:
                    field=float(cached['mean_norm_T']);ppm=float(cached['pp_norm_ppm'])
                    if math.isfinite(field) and field>0 and math.isfinite(ppm) and ppm>=0:
                        proposed=feasibility_score(cached,requirements) if requirements is not None else ppm
                        if math.isfinite(proposed):score=proposed
                except (KeyError,ValueError,TypeError):pass
            item={'run':str(raw),'candidate_id':row['candidate_id'],'geometry_valid':check['valid'],
                'source_index':source_index,'load_rank':None,'cached_rank_only':score is not None,
                'metric_scope_matches':cached is not None,'cached_metric_source':metric_source,
                'cached_metric_pointset':None if cached is None else cached.get('pointset'),
                'cached_rank_score':score,'cached_rank_rule':'current_requirements' if requirements is not None else 'cached_ppm_only',
                'ranking_extra_field_evaluations':0}
            audit.append(item)
            if check['valid']:pending.append(((0,score,source_index) if score is not None else (1,0.,source_index),design,item))
            source_index+=1
    result=[]
    for rank,(_,design,item) in enumerate(sorted(pending,key=lambda entry:entry[0]),1):
        item['load_rank']=rank;result.append(design)
    return result,audit


def run_manufacturing_workflow(args,config,out,util,report_root=None):
    start=time.perf_counter();out=Path(out);report_root=Path(report_root) if report_root else out
    original,_=load_source(config['source_geometry']);limits=_limits(original,config)
    if not 0<config['fov_diameter_mm']<=2*limits['bore_radius_mm']: raise ValueError('平面FOV直径不得超过保护通道')
    if not math.isfinite(float(config['fov_z_mm'])): raise ValueError('FOV平面位置必须有限')
    process=copy.deepcopy(config['manufacturing']);search=copy.deepcopy(config['search'])
    if isinstance(config['max_evaluations'],bool) or int(config['max_evaluations'])!=config['max_evaluations'] or config['max_evaluations']<1:
        raise ValueError('max_evaluations必须为正整数')
    if (isinstance(config['max_validation_candidates'],bool) or int(config['max_validation_candidates'])!=config['max_validation_candidates']
            or config['max_validation_candidates']<0):raise ValueError('max_validation_candidates必须是非负整数')
    if args.fixed_topology:
        for key in ('optimize_azimuth_count','optimize_radial_ring_count','optimize_magnetization'):search[key]=False
        search['topology_limit']=1
    overrides=config.get('source_overrides',{});source=original
    if overrides.get('enabled',False):
        source=replace(original,cube_mm=overrides['cube_mm'],remanence_T=overrides['remanence_T'],reference_mur=overrides['reference_mur'],name='material_override')
    engine=FieldEngine(config['chunk_points'],config['polarization_model'],prepared_cache_size=config.get('prepared_cache_size',8))
    sampling=config['sampling'];prefix='quick_' if args.quick else ''
    training=disk_samples(config['fov_diameter_mm'],sampling[prefix+'training_area'],sampling[prefix+'training_boundary'],config['seed'],z_mm=config['fov_z_mm'])
    field_error=None
    if args.command!='check' and search.get('seed_subset_field_selection',True):
        reference_B=engine.field(source,training['points_mm'])
        def field_error(design):
            delta=engine.field(design,training['points_mm'])-reference_B
            return float(training['weights']@np.sum(delta*delta,axis=1))
    seed,seed_report=_prepare_seed(source,process,limits,search,field_error)
    seed_report['field_comparison_pointset']=training['name'] if field_error else None
    seed_report['field_comparison_units']='weighted vector MSE [T^2]' if field_error else None
    seed_report['engine_statistics_after_seed_selection']=dict(engine.statistics)
    seed_audit=seed_report['manufacturing_audit'];original_check=_design_check(original,process,limits)
    save_json(out/'manufacturing_audit.json',{'configuration':process,'envelope':limits,'baseline':original_check['manufacturing'],
                                            'seed':seed_audit,'seed_generation':seed_report})
    util.export_design(out/'original_baseline_design',original);util.export_design(out/'manufacturing_seed_design',seed)
    snapshot={'configuration':config,'geometry_limits':limits,'python':sys.version,'numpy':np.__version__,
        'scipy':scipy.__version__,'magpylib':magpylib.__version__,'command':vars(args),
        'original_source_sha256':original.source_sha256,'revision':'planar-disk-fov-v3'}
    save_json(out/'run_config.json',snapshot)
    if Path(config.get('parameters_file','')).is_file():shutil.copyfile(config['parameters_file'],out/'parameters_snapshot.py')
    util.log(f"原始{original.magnet_count}块；投影起点{len(seed.rings)}环/{seed.magnet_count}块；几何{'通过' if seed_report['valid'] else '未通过'}")
    if not seed_report['valid']:return 2
    variables,builder=get_search_space(seed,process,limits,search)
    checker=lambda d:_design_check(d,process,limits)
    if not checker(builder([v['initial'] for v in variables]))['valid']:raise ValueError('初始参数向量重建后未通过几何检查')
    save_json(out/'search_space.json',{'variables':variables,'construction_metadata':builder.metadata,'search_switches':search})
    if args.command=='check':
        target=config.get('canonical_config',{}).get('target',{'geometry':'disk','diameter_mm':config['fov_diameter_mm'],'z_mm':config['fov_z_mm']})
        save_json(out/'constraint_summary.json',{'baseline_valid':original_check['valid'],'seed_valid':True,
            'seed_magnets':seed.magnet_count,'radial_packing':seed_audit['packing'],'bed_fit':seed_audit['bed_fit'],'variables':variables,'fov_geometry':target['geometry'],'fov_diameter_mm':target['diameter_mm'],'fov_z_mm':target.get('z_mm',0.),'evaluation_domain':target})
        label=('原点球形DSV' if target['geometry']=='sphere' else 'z=%gmm 平面圆盘'%target.get('z_mm',0.))
        (report_root/'约束检查.txt').write_text(f"几何投影起点通过：{seed.magnet_count}块，{len(variables)}个连续变量。\n主评价为{label}，直径{target['diameter_mm']:g}mm；本命令未求场。\n",encoding='utf-8')
        util.log('约束检查完成，未求场。');return 0
    from .topology import propose_topologies
    topologies=propose_topologies(source,seed,process,limits,search,np.random.default_rng(config['seed']),int(search.get('topology_limit',1)))
    # A common verified symmetry domain applies to every architecture compared.
    domain={'requested':sampling.get('training_domain','full_disk'),'used':'full_disk',
            'geometry':'disk','measure':'area','z_mm':config['fov_z_mm'],'diameter_mm':config['fov_diameter_mm']}
    if sampling.get('training_domain')=='common_symmetry':
        from .sampling import symmetry_profile
        order=math.gcd(*[r.count for d in topologies for r in d.rings])
        checks=[symmetry_profile(d,rotation_order=order,reflection_phase_deg=-90.) for d in topologies]
        domain['profiles']=checks
        if all(c['verified'] for c in checks):
            training=disk_samples(config['fov_diameter_mm'],sampling[prefix+'training_area'],
                sampling[prefix+'training_boundary'],config['seed'],z_mm=config['fov_z_mm'],rotation_order=order,reflection_phase_deg=-90.)
            domain.update(used='common_symmetry',rotation_order=order)
    save_json(out/'training_sampling.json',{k:v for k,v in training.items() if k not in ('points_mm','weights')})
    save_json(out/'training_domain.json',domain)
    validation=disk_samples(config['fov_diameter_mm'],sampling[prefix+'validation_area'],sampling[prefix+'validation_boundary'],config['seed']+101,z_mm=config['fov_z_mm'])
    def evaluate(design):return evaluate_design(design,engine,training,config['density_kg_m3'])[0]
    def dense(design):
        metrics,B=evaluate_design(design,engine,validation,config['density_kg_m3'])
        return refine_disk_extrema(design,engine,validation,metrics,B,starts=sampling['refine_starts']),B
    original_metrics,original_B=dense(original);seed_metrics,seed_B=dense(seed)
    requirements=resolve_requirements(config.get('requirements'),original_metrics,seed_metrics)
    original_training_metrics,seed_training_metrics=evaluate(original),evaluate(seed)
    training_requirements=resolve_requirements(config.get('requirements'),original_training_metrics,seed_training_metrics)
    save_json(out/'requirements_resolved.json',requirements)
    save_json(out/'training_requirements_resolved.json',training_requirements)
    save_json(out/'requirements_scope.json',{'validation':requirements,'training':training_requirements,
        'rule':'Relative gates use their own pointset reference; explicit gates are identical; final acceptance uses dense validation.'})
    def record(ident,design,metrics,parameters=None):
        check=checker(design);performance=check_performance(metrics,requirements)
        return {'candidate_id':ident,'design':design.to_dict(),'metrics':metrics,'validation_metrics':metrics,
            'geometry_check':check['geometry'],'manufacturing_check':check['manufacturing'],'geometry_feasible':check['valid'],
            'performance_check':performance,'feasible':check['valid'] and performance['valid'],
            'parameters':parameters or {},'canonical_physical_sha256':canonical_physical_sha256(design),
            'raw_design_sha256':raw_design_sha256(design)}
    baseline_record=record('baseline',original,original_metrics);seed_record=record('manufacturing_seed',seed,seed_metrics)
    baseline_record['reference_only']=not baseline_record['geometry_feasible']
    baseline_record['training_metrics']=original_training_metrics
    seed_record['training_metrics']=seed_training_metrics
    for row in (baseline_record,seed_record):
        row['validation_status']=validation_status(row['training_metrics'],row['metrics'],training_requirements,requirements)
    util.log(f"原始{original_metrics['mean_norm_T']*1e3:.4f}mT/{original_metrics['pp_norm_ppm']:.1f}ppm；起点{seed_metrics['mean_norm_T']*1e3:.4f}mT/{seed_metrics['pp_norm_ppm']:.1f}ppm")
    util.log(f"复核门槛：场强≥{requirements['min_mean_norm_T']*1e3:.6f}mT（{requirements['reference']}）；ppm≤{requirements['max_pp_norm_ppm']}（{requirements['max_pp_norm_reference']}）；连续变量{len(variables)}")
    util.log(f"训练门槛：场强≥{training_requirements['min_mean_norm_T']*1e3:.6f}mT；ppm≤{training_requirements['max_pp_norm_ppm']}；相对门槛按各点集参考值解析，无隐式放宽")
    for ident,B in [('baseline',original_B),('manufacturing_seed',seed_B)]:write_field_csv(out/(ident+'_fov_samples.csv'),validation['points_mm'],B)
    save_json(out/'baseline_metrics.json',original_metrics);save_json(out/'manufacturing_seed_metrics.json',seed_metrics)
    save_json(out/'validation_sampling.json',{k:v for k,v in validation.items() if k not in ('points_mm','weights')})
    model_info={'status':'平面圆盘FOV面积评价；ε约束快速解析预测','comsol_status':'同坐标FEM平面复核待完成',
        'fov_geometry':'disk','fov_diameter_mm':config['fov_diameter_mm'],'fov_z_mm':config['fov_z_mm'],
        'integration_measure':'area','limitations':[('打印为同列连续径向槽和扇区包络几何初筛；同列可贴紧，端部和列间壁保留；结构、接头及磁力尚未验证。' if process.get('layout_mode','independent_pockets')=='radial_channels' else '打印为独立槽和扇区包络几何初筛；结构、接头及磁力尚未验证。'),
        '快速模型未完整求解材料回复和相互退磁。','平面圆盘内密集采样与局部极值搜索不证明全局最优。',
        '所有评价与极值复核固定在指定z平面的圆盘内。','未设置最高ppm时，场强达标不等于均匀性满足成像要求。']}
    if Path(config['comsol_model']).is_file():
        audit=inspect_mph(config['comsol_model']);save_json(out/'source_mph_audit.json',audit)
        model_info['historical_comsol_line_scope']='原模型已有30mm截线未用于本次圆盘FOV的面积平均场和均匀性指标。'
    payload={'baseline':baseline_record,'scan_reference':seed_record,'baseline_training_metrics':seed_training_metrics,'original_baseline_training_metrics':original_training_metrics,
        'magnet_arrays':original.magnet_arrays(),'plot_data':util.plot_data(original,engine,config['fov_diameter_mm'],args.quick,config['scan'],z_mm=config['fov_z_mm']),
        'requirements':requirements,'training_requirements':training_requirements,'model_info':model_info,'manufacturing':{'config':process,'baseline_audit':original_check['manufacturing'],
        'seed_audit':seed_audit,'seed_record':seed_record,'seed_generation':seed_report,'envelope':limits},
        'candidates':[baseline_record,seed_record],'scans':[]}
    if args.command in ('scan','all'):
        payload['scans'],payload['response_surface'],payload['sensitivity']=_exploration(variables,builder,evaluate,checker,config['scan'],out,util)
    monitor=None
    if args.command in ('optimize','all'):
        from .evolution import EvolutionMonitor
        monitor=EvolutionMonitor(report_root,seed,seed_metrics,training_requirements,config)
        monitor({'event':'baseline','phase':'seed','stage_name':'PAVA打印起点','global_evaluations':0,
                 'design':seed.to_dict(),'metrics':seed_training_metrics,'candidate_id':'manufacturing_seed'})
        warm,warm_audit=_load_warm_starts(config,checker,training_requirements)
        warm += list(getattr(topologies,'warm_starts',[]))
        save_json(out/'warm_start_audit.json',warm_audit)
        save_json(out/'topology_proposals.json',getattr(topologies,'metadata',{}))
        util.log(f"开始外层{len(topologies)}种拓扑／内层独立几何搜索，场评估上限{config['max_evaluations']}")
        options={'event_callback':monitor} if monitor.enabled else {}
        optimized=run_topology_optimization(topologies,process,limits,search,evaluate,checker,out/'optimization',config,training_requirements,warm,**options)
        payload['optimization_metadata']=optimized['metadata']
        chosen,selection=_select_validation_candidates(optimized['records'],
            config['max_validation_candidates'],training_requirements,
            excluded_physical_keys={_design_key(seed),_design_key(original)})
        save_json(out/'validation_selection.json',selection)
        payload['validation_selection']=selection
        for row in chosen:
            design=Design.from_dict(row['design']);metrics,B=dense(design)
            item=record(row['candidate_id'],design,metrics,row['parameters']);item['training_metrics']=row['metrics']
            if 'topology_index' in row:item['topology_index']=row['topology_index']
            item['validation_status']=validation_status(row['metrics'],metrics,training_requirements,requirements)
            monitor({'event':'validation','phase':'validation','stage_name':'密集复核 '+row['candidate_id'],
                'global_evaluations':optimized['metadata'].get('field_evaluations',0),
                'candidate_id':row['candidate_id'],'design':design.to_dict(),'metrics':metrics,
                'validation_status':item['validation_status']})
            payload['candidates'].append(item);util.export_design(out/row['candidate_id'],design)
            write_field_csv(out/(row['candidate_id']+'_fov_samples.csv'),validation['points_mm'],B)
            util.log(f"独立复核 {row['candidate_id']}: {metrics['mean_norm_T']*1e3:.4f}mT / {metrics['pp_norm_ppm']:.1f}ppm；性能门槛{'通过' if item['feasible'] else '未通过'}")
    payload['verified_pareto_ids']=[r['candidate_id'] for r in pareto_front(payload['candidates'])]
    payload['performance_summary']={'requirements_met':bool(payload['verified_pareto_ids']),
        'verified_pareto_ids':payload['verified_pareto_ids'],'diagnostic_only':not bool(payload['verified_pareto_ids'])}
    payload['validation_summary']={'all_selected_training_candidates_retained':True,
        'candidate_count':len(payload['candidates']),
        'training_pass_dense_fail_ids':[r['candidate_id'] for r in payload['candidates']
            if r.get('validation_status',{}).get('code')=='training_pass_dense_fail'],
        'thresholds_relaxed':False,
        'candidate_selection':payload.get('validation_selection',{})}
    if monitor is not None:
        rows=[r for r in payload['candidates'] if r.get('feasible')]
        if not rows:rows=[r for r in payload['candidates'] if r.get('geometry_feasible')]
        if rows:
            best=min(rows,key=lambda r:feasibility_score(r['metrics'],requirements))
            monitor({'event':'final','phase':'dense_final','stage_name':'最终密集复核方案',
                'global_evaluations':payload.get('optimization_metadata',{}).get('field_evaluations',0),
                'candidate_id':best['candidate_id'],'design':best['design'],'metrics':best['metrics'],
                'validation_status':best.get('validation_status')})
        payload['evolution']=monitor.finish()
    save_json(out/'validated_candidates.json',payload['candidates'])
    if args.reference:model_info['comsol_comparison']=util.compare_reference(original,engine,args.reference,out,args.coordinate_unit,args.field_unit,fov_diameter_mm=config['fov_diameter_mm'],z_mm=config['fov_z_mm'])
    model_info['elapsed_seconds']=time.perf_counter()-start;model_info['field_engine_statistics']=engine.statistics
    from .deliverables import finalize_results
    report,export_ok=finalize_results(report_root,out,payload,config,util.log)
    save_json(out/'run_summary.json',{'baseline':original_metrics,'manufacturing_seed':seed_metrics,'candidates':payload['candidates'],
        'verified_pareto_ids':payload['verified_pareto_ids'],'requirements':requirements,'training_requirements':training_requirements,'performance_summary':payload['performance_summary'],
        'optimization_metadata':payload.get('optimization_metadata',{}),'field_engine_statistics':engine.statistics,
        'manufacturing_config':process,'search_switches':search,'elapsed_seconds':model_info['elapsed_seconds'],
        'deliverables':payload['deliverables'],'evolution':payload.get('evolution',{}),'validation_summary':payload.get('validation_summary',{}),'report':report})
    util.log('报告：'+report['report_html'])
    if not export_ok:return 2
    return 0 if payload['performance_summary']['requirements_met'] else 3

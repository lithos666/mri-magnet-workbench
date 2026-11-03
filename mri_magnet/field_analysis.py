"""Demand-driven engineering diagnostics; never replace optimizer/FEM metrics."""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
from .field import FieldEngine
from .metrics import field_metrics
from .sampling import sphere_samples, disk_samples
from .identity import canonical_physical_sha256
from .serialization import save_json


def analysis_sample(config):
    options=config['analysis'];target=config['target']
    args=(target['diameter_mm'],int(options['integration_volume']),int(options['integration_surface']),int(config['seed'])+7181)
    if target['geometry']=='disk':return disk_samples(*args,z_mm=target.get('z_mm',0.))
    return sphere_samples(*args)


def field_slices(design,engine,config,reference_bz_T):
    options=config['analysis'];target=config['target'];grid=int(options['slice_grid_size'])
    radius=float(target['diameter_mm'])/2
    u,v=np.meshgrid(np.linspace(-radius,radius,grid),np.linspace(-radius,radius,grid))
    mask=(u*u+v*v<=radius*radius+1e-10).ravel()
    if not math.isfinite(reference_bz_T) or abs(reference_bz_T)<1e-12:raise ValueError('平均Bz接近零，无法定义相对Bz切片')
    planes=[]
    for name,axis_u,axis_v in (('XY',0,1),('XZ',0,2),('YZ',1,2)):
        if target['geometry']=='disk' and name!='XY':continue
        points=np.zeros((grid*grid,3));points[:,axis_u]=u.ravel();points[:,axis_v]=v.ravel()
        offset=float(target.get('z_mm',0.)) if target['geometry']=='disk' else 0.
        if name=='XY':points[:,2]=offset
        values=np.zeros(len(points))
        values[mask]=(engine.field(design,points[mask])[:,2]-reference_bz_T)/abs(reference_bz_T)*1e6
        planes.append({'name':name,'axis_u':axis_u,'axis_v':axis_v,'offset_mm':offset,
            'points_mm':points.tolist(),'values_ppm':[float(value) if valid else None for value,valid in zip(values,mask)],
            'valid_mask':mask.tolist(),'grid_size':grid,'extent_mm':radius})
    maximum=max(abs(value) for plane in planes for value in plane['values_ppm'] if value is not None)
    requested=options.get('slice_color_limit_ppm')
    limit=max(maximum,1e-6) if requested is None else float(requested)
    return {'planes':planes,'quantity':'Bz','reference_Bz_T':float(reference_bz_T),
        'reference':'same-domain weighted mean Bz; ppm=(Bz-meanBz)/abs(meanBz)*1e6',
        'color_limit_ppm':limit,'color_scale':'fixed across the three planes; configure slice_color_limit_ppm to compare candidates',
        'source_model':engine.polarization_model,'sampled_only':True}


def analyze_design(design,config,output_dir,options=None,source_label=''):
    """Explicit standalone analysis; independent from Step6 acceptance records."""
    from .harmonics import analyze_harmonics
    options=dict(options or {});started=time.perf_counter();settings=config['analysis']
    for key in options:
        if key not in ('mutual_demag','tolerance'):raise ValueError('未知分析选项: '+key)
        if not isinstance(options[key],bool):raise ValueError(key+'必须为布尔值')
    sample=analysis_sample(config)
    run_coupled=config.get('physics',{}).get('field_model')=='coupled'
    selected_mutual=options.get('mutual_demag',run_coupled or settings['mutual_demag']['enabled'])
    selected_tolerance=options.get('tolerance',settings['tolerance']['enabled'])
    maximum_columns=max(r.count for r in design.rings)
    n_phi=max(65,4*maximum_columns+1)
    harmonic_points=max(16,int(settings['harmonic_max_degree'])+2)*n_phi if config['target']['geometry']=='sphere' else 327
    plane_count=3 if config['target']['geometry']=='sphere' else 1
    estimated_points=(len(sample['points_mm'])+plane_count*int(settings['slice_grid_size'])**2+harmonic_points+max(97,n_phi))
    if selected_tolerance:estimated_points+=7*len(sample['points_mm'])+96
    estimated_pairs=estimated_points*design.magnet_count
    if estimated_pairs>settings['max_source_point_pairs']:
        raise ValueError('工程分析预计约%d次源点贡献，超过配置上限%d；请降低分析采样/网格或显式调高预算'%(estimated_pairs,settings['max_source_point_pairs']))
    response_options={key:value for key,value in settings['mutual_demag'].items() if key!='enabled'}
    if run_coupled:
        from .field_models import response_options as physics_options
        response_options=physics_options(config)
    response_plan=None;tolerance_plan=None
    if selected_mutual:
        from .material_response import preflight_response
        response_plan=preflight_response(design,response_options)
        if not response_plan['valid']:raise ValueError('；'.join(response_plan['violations']))
    if selected_tolerance:
        from .tolerance import preflight_tolerance
        tolerance_plan=preflight_tolerance(design,len(sample['points_mm']),config,settings['tolerance'])
        if not tolerance_plan['valid']:raise ValueError('公差预计算工作数组约%.1f MB，超过%.1f MB上限'%(tolerance_plan['estimated_peak_memory_mb'],tolerance_plan['limit_mb']))
    fast=FieldEngine(min(int(config['chunk_points']),48),config['polarization_model'])
    baseline=fast.field(design,sample['points_mm'])
    fast_metrics=field_metrics(baseline,sample['weights'])
    engine=fast;response={'status':'not_run','enabled':False,'fem_verified':False}
    if selected_mutual:
        from .material_response import CoupledResponseEngine
        engine=CoupledResponseEngine(design,response_options)
        response={'status':'completed','enabled':True,'preflight':response_plan,
            'diagnostics':engine.diagnostics,'ring_polarization':engine.ring_summary,'fem_verified':False}
    fields=baseline if engine is fast else engine.field(design,sample['points_mm'])
    metrics=field_metrics(fields,sample['weights'])
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    report={'schema':'engineering-analysis-v1','design':design.to_dict(),
        'design_sha256':canonical_physical_sha256(design),'source_label':str(source_label)[:240],
        'polarization_model':config['polarization_model'],'field_model_used':engine.polarization_model,
        'physics':deepcopy(config.get('physics',{})),
        'model_scope':'run_model' if selected_mutual==run_coupled else 'explicit_comparison_model',
        'domain':deepcopy(config['target']),'pointset':sample['name'],'point_count':len(fields),
        'fast_model_metrics':fast_metrics,'analysis_model_metrics':metrics,
        'preflight':{'estimated_source_point_pairs':estimated_pairs,'source_point_pair_limit':settings['max_source_point_pairs'],
                     'material_response':response_plan,'tolerance':tolerance_plan,'effective_fast_chunk_points':min(int(config['chunk_points']),48)},
        'material_response':response,'slices':field_slices(design,engine,config,metrics['mean_Bz_T']),
        'harmonics':analyze_harmonics(design,engine,config,lmax=int(settings['harmonic_max_degree']),
            reference_T=metrics['mean_Bz_T'],output_dir=folder),
        'tolerance':{'available':False,'reason':'未选择公差分析'},
        'fem_verified':False,'dense_acceptance_replaced':False,
        'scope':'按需工程诊断；不覆盖原优化指标，不构成全局极值、FEM或结构安全证明'}
    if selected_tolerance:
        from .tolerance import analyze_tolerance
        report['tolerance']=analyze_tolerance(design,sample,config,settings['tolerance'])
        if engine is not fast:
            report['tolerance']['scope_note']='公差按固定源快速模型独立估计；没有传播互退磁耦合的不确定性'
    report['elapsed_seconds']=time.perf_counter()-started
    save_json(folder/'工程分析.json',report)
    tolerance=report['tolerance'];lines=['# 工程分析速览','',report['scope'],'',
        '评价域：%s，直径%g mm；分析求场模型：%s。'%(config['target']['geometry'],config['target']['diameter_mm'],engine.polarization_model),'',
        '| 模型 | 采样均值 / mT | 采样峰峰值 / ppm |','|---|---:|---:|',
        '| 原快速模型 | %.6f | %.6f |'%(fast_metrics['mean_norm_T']*1000,fast_metrics['pp_norm_ppm']),
        '| 本次分析模型 | %.6f | %.6f |'%(metrics['mean_norm_T']*1000,metrics['pp_norm_ppm']),'',
        '球谐分解针对 Bz；三维切片也显示相对平均 Bz 的偏差。均值与峰峰值表按 |B| 计算，二者不会混标。','']
    if tolerance.get('available'):
        summary=tolerance['summary'];lines += ['## 公差估计','',
            'P50：%.3f ppm；P90：%.3f ppm；样本通过率：%.1f%%（门槛范围：%s）。'%(summary['p50_ppm'],summary['p90_ppm'],100*summary['yield_fraction'],summary['yield_scope']),
            '分布：独立磁块剩磁均匀半幅±%.3f%%；同层同扇区共用XYZ平移误差，均匀半幅±%.3f mm。'%(settings['tolerance']['remanence_half_range_fraction']*100,settings['tolerance']['sector_translation_half_range_mm']),
            '这是第一阶位移代理与全域有限采样的统计估计；不含制造碰撞良率或密集极值认证。','']
    lines += ['互退磁采用均匀磁块近似，积分阶数与空间细分仍需收敛检查。方程残差小不等于已经与COMSOL达到相同ppm精度。',
              '底板挠度与DXF/STL未在此模块中生成：需要材料、打印层向、紧固、支承及连接结构参数。']
    (folder/'工程分析速览.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

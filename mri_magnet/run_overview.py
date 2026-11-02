"""Readable summaries from saved designs and measured metrics; never solves fields."""
from __future__ import annotations
import math
from pathlib import Path
from .geometry import Design
from .objectives import feasibility_score
from .serialization import save_json


def reference_record(payload):
    return (payload.get('manufacturing',{}).get('seed_record') or payload.get('scan_reference') or {})


def reference_design(payload):
    record=reference_record(payload)
    try:return Design.from_dict(record['design']) if record.get('design') else None
    except (KeyError,ValueError,TypeError):return None


def metric_values(record):
    return record.get('validation_metrics') or record.get('metrics') or {}


def geometry_valid(row,payload):
    valid=bool(row.get('geometry_feasible',row.get('geometry_check',{}).get('valid',True)))
    if payload.get('manufacturing'):valid &= bool(row.get('manufacturing_check',{}).get('valid',False))
    return valid


def best_record(payload):
    rows=[r for r in payload.get('candidates',[]) if r.get('design') and r.get('validation_metrics') and geometry_valid(r,payload)]
    valid=[r for r in rows if r.get('feasible',False) and r.get('performance_check',{}).get('valid',True)]
    if valid:return min(valid,key=lambda r:feasibility_score(metric_values(r),payload.get('requirements')))
    if rows:return min(rows,key=lambda r:feasibility_score(metric_values(r),payload.get('requirements')))
    return reference_record(payload) or payload.get('baseline',{})


def validation_label(row):
    label=row.get('validation_status')
    if isinstance(label,dict):label=label.get('label') or label.get('status')
    if label:return str(label)
    training=row.get('training_performance_check') or {}
    dense=row.get('performance_check') or {}
    if training.get('valid') and dense.get('valid') is False:return '训练达标 / 密集复核未通过'
    if dense:return '双门槛通过' if dense.get('valid') else '复核未通过'
    return '未记录性能门槛'


def _fmt(value,digits=3):
    try:
        number=float(value)
        if not math.isfinite(number):return '—'
    except (ValueError,TypeError):return '—'
    return f'{number:.0f}' if digits==0 else f'{number:.{digits}f}'.rstrip('0').rstrip('.')


def _signed(value):
    try:return f'{float(value):+.3f}'
    except (ValueError,TypeError):return '—'


def _ring_count(row):return len(row.get('design',{}).get('rings',[]))


def _magnet_count(row):
    value=metric_values(row).get('magnet_count')
    return value if value is not None else sum(int(r.get('count',0)) for r in row.get('design',{}).get('rings',[]))


def stage_records(payload):
    """Use only actual optimizer summaries. Older runs show missing phases."""
    meta=payload.get('optimization_metadata') or {}
    result=list(meta.get('stage_summaries') or [])
    offset=0
    for run in meta.get('runs') or []:
        for row in run.get('metadata',{}).get('stage_summaries') or []:
            item=dict(row,topology_index=run.get('topology'))
            if 'global_evaluations' not in item and item.get('field_evaluations') is not None:
                item['global_evaluations']=offset+item['field_evaluations']
            result.append(item)
        offset+=run.get('metadata',{}).get('field_evaluations',0)
    return result


def _stage_best(payload,names):
    rows=[r for r in stage_records(payload) if str(r.get('stage',r.get('stage_name',''))) in names and r.get('metrics')]
    if not rows:return None
    return min(rows,key=lambda r:feasibility_score(r['metrics'],payload.get('training_requirements') or payload.get('requirements')))


def comparison_manifest(payload,chosen):
    reference=reference_design(payload)
    if reference is None or not chosen.get('design'):return None
    from .layout import layout_comparison_manifest
    return layout_comparison_manifest(Design.from_dict(chosen['design']),reference)


def write_run_overview(public_dir,payload):
    public=Path(public_dir); chosen=best_record(payload); seed=reference_record(payload)
    seed_m=metric_values(seed);seed_field=seed_m.get('mean_norm_T');seed_ppm=seed_m.get('pp_norm_ppm')
    info=payload.get('model_info') or {}
    lines=['# 本次运行速览','',
        f"评价域：z={_fmt(info.get('fov_z_mm',0))} mm，直径{_fmt(info.get('fov_diameter_mm'))} mm圆形FOV。平均场按面积积分。" if info.get('fov_geometry')=='disk' else '评价域与旧结果记录一致；详见本次报告。',
        '',f"当前选中：{chosen.get('candidate_id','未选择')}；{validation_label(chosen)}。",'',
        '## 三基准核心对比','',
        '| 方案 | 磁环 / 磁块 | 平均场 mT | ppm | 相对打印起点 ΔB mT | 相对起点 ppm 降幅 | 验收状态 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for label,row in [('原始裸磁体',payload.get('baseline') or {}),('PAVA可打印起点',seed),('本轮选中方案',chosen)]:
        m=metric_values(row);field=m.get('mean_norm_T');ppm=m.get('pp_norm_ppm')
        delta=(field-seed_field)*1000 if field is not None and seed_field is not None else None
        change=(1-ppm/seed_ppm)*100 if ppm is not None and seed_ppm else None
        status=validation_label(row)
        if not geometry_valid(row,payload):status+='；几何/制造未通过'
        lines.append(f"| {label} | {_ring_count(row)} / {_magnet_count(row)} | {_fmt(field*1000 if field is not None else None,6)} | {_fmt(ppm,6)} | {_signed(delta)} | {_fmt(change,3)}% | {status} |")
    lines += ['', '环数来自实际设计记录，不按历史注释推算。ppm改善只在同一评价区域、同一指标定义下解释。', '',
        '## 各阶段优化贡献','',
        '| 阶段 | 平均场 mT | ppm | 累计搜索求场 | 数据口径 |', '|---|---:|---:|---:|---|']
    initial_metrics=payload.get('baseline_training_metrics') or seed_m
    stage_items=[('阶段0 PAVA起点',{'metrics':initial_metrics,'field_evaluations':0}),
        ('阶段1 混合初始化最优',_stage_best(payload,{'initialization','initialization_end'})),
        ('阶段2 差分进化最优',_stage_best(payload,{'de','de_end'})),
        ('阶段3 坐标与成对精修最优',_stage_best(payload,{'local','local_end','paired','paired_end','local_refinement','local_recovery'})),
        ('阶段4 密集复核最终值',{'metrics':metric_values(chosen),'field_evaluations':None})]
    for label,row in stage_items:
        m=(row or {}).get('metrics') or {};field=m.get('mean_norm_T')
        scope='密集复核' if label.startswith('阶段4') else ('训练点集' if row else '旧记录未保存该阶段')
        lines.append(f"| {label} | {_fmt(field*1000 if field is not None else None,6)} | {_fmt(m.get('pp_norm_ppm'),6)} | {_fmt((row or {}).get('global_evaluations',(row or {}).get('field_evaluations')),0)} | {scope} |")
    lines += ['', '阶段值来自实际阶段记录；— 表示未记录，不推测该阶段改善。训练到密集复核的指标变化包含采样与极值细化差异。', '',
        '## 物理尺寸改动 ΔR / ΔZ','',
        '| 镜像层对 | 各层保留环数 | Z与相对起点ΔZ / mm | 正半轴各槽 R与ΔR / mm |', '|---|---|---|---|']
    comparison=comparison_manifest(payload,chosen)
    if comparison:
        by_layer={int(r['layer_index']):r for r in comparison.get('layers',[])}
        for pair in range(7):
            a,b=pair,13-pair
            if a not in by_layer and b not in by_layer:continue
            layers=[by_layer[i] for i in (a,b) if i in by_layer]
            counts=' / '.join(str(x.get('current_ring_count',0)) for x in layers)
            z='；'.join(f"R{int(x['layer_index'])+1:02d}: {_fmt(x.get('current_z_mm'))} (Δ{_signed(x.get('delta_z_mm'))})" for x in layers)
            radial=[]
            for r in [r for r in comparison.get('rings',[]) if int(r['layer_index'])==a]:
                slot=int(r['slot_index'])+1;status=r['status']
                radial.append(f"L{slot}: 已删" if status=='removed' else f"L{slot}: {_fmt(r.get('current_r_mm'))} ("+('新增' if status=='added' else 'Δ'+_signed(r.get('delta_r_mm')))+')')
            lines.append(f"| R{a+1:02d}/R{b+1:02d} | {counts} | {z} | {'；'.join(radial) or '该层无记录'} |")
        save_json(public/'详细数据'/'物理尺寸改动.json',comparison)
    else:lines.append('| — | — | 未提供可匹配的起点设计 | — |')
    lines += ['', '磁环按层号与源槽号匹配；新增和删减单独标出。负半轴的独立记录保存在“详细数据/物理尺寸改动.json”，不以排序位置冒认同一磁环。', '',
        '## 文件入口','', '- [完整HTML报告](report_中文.html)', '- [候选方案对比](方案对比.csv)', '- [演化时间轴]('+str(payload['evolution']['timeline_file']).replace('\\','/')+')' if payload.get('evolution',{}).get('timeline_file') else '- 本次未生成演化回放。', '',
        '布局图与CLASS位于“候选方案”；诊断方案显式标为复核未通过。CLASS编译成功不表示FEM已求解。']
    path=public/'本次运行速览.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path

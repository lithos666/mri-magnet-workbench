"""Package a run around its human-readable report and actual candidate models."""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

from .geometry import Design
from .layout import render_rz_layout, layout_manifest
from .comsol_export import export_comsol_model
from .serialization import save_json
from .objectives import feasibility_score, pareto_front
from .design_contract import (check_performance, record_metrics, record_is_feasible, valid_record_metrics,
                              eligible_records, preferred_records)
from .run_overview import reference_design, write_run_overview, validation_label


def _save(path, value):
    save_json(path, value)


def selected_candidates(payload):
    """Use only independently checked records; never export an infeasible reference."""
    rows = list(payload.get('candidates') or [])
    baseline = payload.get('baseline', {})
    if baseline and not any(r['candidate_id'] == baseline.get('candidate_id') for r in rows):
        rows.append(baseline)
    rows = preferred_records(rows)
    feasible = [r for r in rows if r.get('feasible', False)
                and (r.get('validation_metrics') or r.get('dense_metrics')) and r.get('design')
                and (r.get('geometry_check') or {}).get('valid', True)
                and (r.get('manufacturing_check') or {}).get('valid', True)]
    if payload.get('manufacturing'):
        feasible = [r for r in feasible if (r.get('manufacturing_check') or {}).get('valid', False)]
    requirements = payload.get('requirements')
    return sorted(pareto_front(eligible_records(feasible,requirements),requirements),
                  key=lambda r: (record_metrics(r)['pp_norm_ppm'], r['candidate_id']))


def export_candidates(public_dir, details_dir, payload, config, log=print):
    public_dir, details_dir = Path(public_dir).resolve(), Path(details_dir).resolve()
    options = dict(config.get('deliverables', {}))
    enabled = options.pop('enabled', True)
    qualified = selected_candidates(payload)
    result = {'enabled': enabled, 'success': True, 'candidates': [],
              'pareto_front_ids': [row['candidate_id'] for row in qualified],
              'selection': '独立复核后可行方案的帕累托前沿', 'solve_executed': False}
    if payload.get('model_info',{}).get('fov_geometry')=='disk':
        info=payload['model_info']
        result['evaluation_region']={'geometry':'disk','z_mm':info['fov_z_mm'],
            'diameter_mm':info['fov_diameter_mm'],'integration_measure':'area'}
    if not enabled:
        return result
    result_options = dict(config.get('results', {}))
    diagnostic = [r for r in preferred_records(payload.get('candidates',[])) if (r.get('validation_metrics') or r.get('dense_metrics'))
                  and r.get('geometry_feasible',r.get('geometry_check',{}).get('valid',False))
                  and r.get('manufacturing_check',{}).get('valid',True)
                  and valid_record_metrics(r)
                  and not record_is_feasible(r,payload.get('requirements'))]
    diagnostic.sort(key=lambda r:feasibility_score(record_metrics(r),payload.get('requirements')))
    diagnostic_limit = max(0,int(result_options.get('max_diagnostic_candidates',result_options.get('max_diagnostic_exports',4))))
    diagnostic = diagnostic[:diagnostic_limit] if result_options.get('diagnostic_exports',True) else []
    candidates = list(qualified)
    included = {r['candidate_id'] for r in candidates}
    candidates += [r for r in diagnostic if r['candidate_id'] not in included]
    if not candidates:
        result.update(success=False, error='没有通过几何检查的候选，未生成建模文件。')
        return result
    qualified_ids = {r['candidate_id'] for r in qualified}
    if not qualified:
        result.update(diagnostic_only=True, selection='未达性能门槛；仅供审查的诊断方案')
    reference = reference_design(payload)
    readable = bool(result_options.get('readable_names',True))
    best_field = max((record_metrics(r)['mean_norm_T'] for r in qualified),default=0.)
    for index, record in enumerate(candidates, 1):
        ident = record['candidate_id']
        safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', ident)
        is_diagnostic = ident not in qualified_ids
        metrics = record_metrics(record)
        role = ('复核未通过_诊断' if is_diagnostic else '打印起点' if ident in ('baseline','manufacturing_seed')
                else '最优' if index==1 else '高场候选' if metrics['mean_norm_T']>=best_field else '候选')
        count = int(metrics.get('magnet_count',sum(int(r.get('count',0)) for r in record.get('design',{}).get('rings',[]))))
        folder = ('%02d_%s_%.2fmT_%.3fppm_%d块' % (index,role,metrics['mean_norm_T']*1000,metrics['pp_norm_ppm'],count)
                  if readable else '%02d_%s' % (index,safe))
        target = public_dir / '候选方案' / folder
        logs = details_dir / '导出日志' / safe
        target.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        item = {'id': ident, 'candidate_id': ident, 'compiled': False,
                'diagnostic_only':is_diagnostic, 'display_name':role,
                'validation_status':'独立复核未通过（诊断方案）' if is_diagnostic else validation_label(record),
                'is_starting_reference': ident in ('baseline', 'manufacturing_seed'),
                'metrics': metrics, 'is_pareto': not is_diagnostic}
        def relative(path):
            return str(Path(path).resolve().relative_to(public_dir)) if path else None
        try:
            design = Design.from_dict(record['design'])
            suffix = ('（复核未通过，仅供诊断）' if is_diagnostic else '') + ('（打印起点）' if ident == 'manufacturing_seed' else '')
            layout = render_rz_layout(design, target / '磁铁布局_RZ.png',
                title='MRI · %s · %.3f mT / %.3f ppm%s · 磁铁放置尺寸' % (role,metrics['mean_norm_T']*1000,metrics['pp_norm_ppm'],suffix),
                manufacturing_audit=record.get('manufacturing_check'), reference_design=reference,
                displacement_threshold_mm=float(result_options.get('displacement_threshold_mm',.5)))
            item['layout_file'] = relative(layout)
            _save(logs / 'layout_manifest.json', layout_manifest(design))
            if reference is not None:
                from .layout import layout_comparison_manifest
                _save(logs / 'layout_comparison_manifest.json',layout_comparison_manifest(design,reference))
            _save(logs / 'design.json', design.to_dict())
            log('生成布局与 COMSOL 模型：' + ident)
            exported = export_comsol_model(design, target, config['comsol_root'],
                                           {**options, 'logs_dir': str(logs)})
            item.update({k: exported.get(k) for k in ('status', 'compiled', 'model_built', 'error',
                'design_sha256', 'class_name', 'magnet_count', 'ring_count', 'solve_executed')})
            for key in ('java_file', 'class_file', 'manifest_file', 'compile_log'):
                item[key] = relative(exported.get(key))
            # A compiled file and a native construction check are distinct requirements.
            item['success'] = not exported.get('error') and (not options.get('compile', True) or exported['compiled'])
            if not item['success']:
                log('导出未完成 ' + ident + '：' + str(item.get('error')))
        except Exception as error:
            item.update(status='export_failed', error=str(error), success=False)
            log('导出未完成 ' + ident + '：' + str(error))
        result['candidates'].append(item)
    result['success'] = all(r.get('success', False) for r in result['candidates'])
    _save(details_dir / 'deliverables_manifest.json', result)
    return result


def write_comparison(public_dir, payload):
    exported = {r['id']: r for r in payload.get('deliverables', {}).get('candidates', [])}
    pareto_ids = {r['candidate_id'] for r in selected_candidates(payload)}
    path = Path(public_dir) / '方案对比.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        mean_label='FOV面积平均场_mT' if payload.get('model_info',{}).get('fov_geometry')=='disk' else '平均场_mT'
        writer.writerow(['方案编号', '几何约束通过', '性能门槛通过','可行帕累托候选', mean_label, '峰峰值_ppm', '磁铁数量',
                         '磁材重量_kg', '布局图', 'CLASS文件', '导出状态', '训练门槛通过', '复核状态'])
        for row in payload.get('candidates', []):
            metrics = record_metrics(row)
            performance = row.get('performance_check',{}).get('valid','旧记录未定义门槛')
            if payload.get('requirements') is not None:
                performance = valid_record_metrics(row) and check_performance(metrics,payload['requirements'])['valid']
            item = exported.get(row['candidate_id'], {})
            writer.writerow([row['candidate_id'], row.get('geometry_feasible', row.get('feasible',False)),
                performance, row['candidate_id'] in pareto_ids,
                metrics.get('mean_norm_T', 0) * 1000, metrics.get('pp_norm_ppm'), metrics.get('magnet_count'),
                metrics.get('mass_magnet_kg'), item.get('layout_file', ''), item.get('class_file', ''), item.get('status', '未选为导出候选'), row.get('training_performance_check',{}).get('valid','未记录'), validation_label(row)])
    return path


def finalize_results(public_dir, details_dir, payload, config, log=print):
    from .reporting import create_report
    public_dir, details_dir = Path(public_dir).resolve(), Path(details_dir).resolve()
    public_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    # Re-exporting a historical payload must not preserve an old training-front
    # label when independent validation reverses the candidate ordering.
    payload['verified_pareto_ids'] = [row['candidate_id'] for row in selected_candidates(payload)]
    payload.setdefault('performance_summary',{}).update(
        requirements_met=bool(payload['verified_pareto_ids']),
        verified_pareto_ids=payload['verified_pareto_ids'],
        diagnostic_only=not bool(payload['verified_pareto_ids']))
    payload['deliverables'] = export_candidates(public_dir, details_dir, payload, config, log)
    _save(details_dir / 'export_settings.json', {'comsol_root': config['comsol_root'],
                                               'deliverables': config.get('deliverables', {})})
    _save(details_dir / 'report_payload.json', payload)
    write_comparison(public_dir, payload)
    if config.get('results',{}).get('overview',True):
        payload['overview_file'] = str(write_run_overview(public_dir,payload).relative_to(public_dir))
        _save(details_dir / 'report_payload.json', payload)
    report = create_report(public_dir, payload, details_dir=details_dir)
    summary_path = details_dir / 'run_summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary.update(report=report, deliverables=payload['deliverables'],
                       verified_pareto_ids=payload['verified_pareto_ids'],
                       performance_summary=payload['performance_summary'])
        _save(summary_path, summary)
    if public_dir.parent.name == 'results':
        from urllib.parse import quote
        link = quote(public_dir.name) + '/' + quote('report_中文.html')
        (public_dir.parent / '最新结果.html').write_text(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最新 MRI 结果</title>'
            '<meta http-equiv="refresh" content="0;url=' + html.escape(link, quote=True) + '">'
            '<a href="' + html.escape(link, quote=True) + '">打开最新结果：' + html.escape(public_dir.name) + '</a></html>', encoding='utf-8')
    return report, payload['deliverables']['success']


def export_saved_run(run_dir, config, log=print):
    root = Path(run_dir).resolve()
    details = root / '详细数据'
    source = details / 'report_payload.json'
    if not source.is_file():
        raise FileNotFoundError('请指向包含“详细数据/report_payload.json”的运行目录：' + str(root))
    payload = json.loads(source.read_text(encoding='utf-8'))
    report, success = finalize_results(root, details, payload, config, log)
    log('补导出完成，报告：' + report['report_html'])
    return 0 if success else 2

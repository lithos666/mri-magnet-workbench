"""Auditable fixed-versus-mixed topology experiments at equal search budgets.

One projected start, fixed planar disk training/validation points and resolved
requirements are shared by every run. Random seeds change only proposal/DE
randomness. Seed-selection and dense-verification costs are recorded separately
from the per-run search cap. This module does not run COMSOL or claim improvement.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import html
from pathlib import Path
import time

import numpy as np

from .field import FieldEngine
from .geometry import Design, load_source
from .identity import raw_design_sha256, canonical_physical_sha256
from .metrics import evaluate_design
from .objectives import resolve_requirements, check_performance, feasibility_score
from .optimization import run_topology_optimization
from .sampling import disk_samples
from .serialization import save_json
from .benchmark_curves import optimization_curves, compare_curves
from .topology import propose_topologies
from .validation import refine_disk_extrema
from .workflow import _prepare_seed, _limits, _design_check, _load_warm_starts


def _sampling_fingerprint(sample):
    digest = hashlib.sha256()
    for key in ('points_mm', 'weights'):
        digest.update(np.ascontiguousarray(sample[key], dtype='<f8').tobytes())
    return digest.hexdigest()


def _run_rounds(metadata):
    rows = []
    for run in metadata.get('runs', []):
        local = run.get('metadata', {})
        for info in local.get('rounds', []):
            rows.append({'topology': run.get('topology'), 'active_dimension': run.get('active_dimension'),
                         'population_size': info.get('population_size'),
                         'generations': info.get('generations'), 'state': info.get('state'),
                         'initialization_complete': info.get('initialization_complete')})
    return rows


def _mean(values):
    return float(np.mean(values)) if values else None


def run_benchmark(config, output_dir, quick=True, seeds=None, budget=None):
    """Return fixed/mixed experiment rows and write CSV, JSON and offline HTML.

    All search runs receive the same *maximum* number of uncached field
    evaluations. Actual counts can be smaller due to convergence or no free
    variables and are reported. Validation selects one training-best design per
    run; it does not tune on the validation set. ``benchmark.refine_extrema``
    defaults to True, with one local start per independent extrema objective.
    """
    started = time.perf_counter()
    config = copy.deepcopy(config)
    base_seed = int(config['seed'])
    seeds = tuple((base_seed, base_seed+1, base_seed+2) if seeds is None else seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or int(s) != s or s < 0 for s in seeds)):
        raise ValueError('Benchmark seeds must be distinct non-negative integers')
    seeds = tuple(int(s) for s in seeds)
    if budget is None:
        budget = (config.get('quick', {}).get('max_evaluations', config['max_evaluations'])
                  if quick else config['max_evaluations'])
    if isinstance(budget, bool) or int(budget) != budget or budget < 1:
        raise ValueError('Benchmark budget must be a positive integer')
    budget = int(budget)
    public = Path(output_dir).resolve()
    details = public/'详细数据'
    details.mkdir(parents=True, exist_ok=True)
    save_json(details/'benchmark_config.json', {'configuration': config, 'quick': bool(quick),
                                                'seeds': seeds, 'search_budget_per_run': budget})
    original, _ = load_source(config['source_geometry'])
    limits = _limits(original, config)
    if not 0 < config['fov_diameter_mm'] <= 2*limits['bore_radius_mm']:
        raise ValueError('Planar FOV must fit the protected passage')
    process, search = copy.deepcopy(config['manufacturing']), copy.deepcopy(config['search'])
    if quick:
        search['topology_limit'] = config.get('quick', {}).get('topology_limit', search.get('topology_limit', 1))
    source = original
    overrides = config.get('source_overrides', {})
    if overrides.get('enabled', False):
        source = replace(original, cube_mm=overrides['cube_mm'], remanence_T=overrides['remanence_T'],
                         reference_mur=overrides['reference_mur'], name='material_override')
    sampling = config['sampling']
    prefix = 'quick_' if quick else ''
    training = disk_samples(config['fov_diameter_mm'], sampling[prefix+'training_area'],
                              sampling[prefix+'training_boundary'], base_seed, z_mm=config['fov_z_mm'])
    validation = disk_samples(config['fov_diameter_mm'], sampling[prefix+'validation_area'],
                                sampling[prefix+'validation_boundary'], base_seed+101, z_mm=config['fov_z_mm'])
    training_hash, validation_hash = _sampling_fingerprint(training), _sampling_fingerprint(validation)
    save_json(details/'benchmark_sampling.json', {
        'training': training, 'validation': validation,
        'training_sha256': training_hash, 'validation_sha256': validation_hash,
        'domain': 'full planar disk, fixed across all modes and optimizer seeds'})
    engine_options = {'chunk_points': config['chunk_points'],
                      'polarization_model': config['polarization_model'],
                      'prepared_cache_size': config.get('prepared_cache_size', 8)}
    prepare_engine = FieldEngine(**engine_options)
    selection_started = time.perf_counter()
    field_error = None
    if search.get('seed_subset_field_selection', True):
        reference_B = prepare_engine.field(source, training['points_mm'])
        def field_error(design):
            delta = prepare_engine.field(design, training['points_mm'])-reference_B
            return float(training['weights']@np.sum(delta*delta, axis=1))
    seed, seed_report = _prepare_seed(source, process, limits, search, field_error)
    selection = {'report': seed_report, 'elapsed_seconds': time.perf_counter()-selection_started,
                 'field_engine_statistics': dict(prepare_engine.statistics)}
    save_json(details/'benchmark_seed_selection.json', selection)
    if not seed_report['valid']:
        raise ValueError('Benchmark could not construct a feasible starting layout; see seed selection audit')
    checker = lambda design: _design_check(design, process, limits)
    refine_reference = bool(config.get('benchmark', {}).get('refine_extrema', True))
    def validation_metrics(design, engine):
        metrics, fields = evaluate_design(design, engine, validation, config['density_kg_m3'])
        if refine_reference:
            metrics = refine_disk_extrema(design, engine, validation, metrics, fields, starts=1)
        return metrics
    original_metrics = validation_metrics(original, prepare_engine)
    seed_metrics = validation_metrics(seed, prepare_engine)
    requirements = resolve_requirements(config.get('requirements'), original_metrics, seed_metrics)
    original_training_metrics = evaluate_design(original, prepare_engine, training, config['density_kg_m3'])[0]
    seed_training_metrics = evaluate_design(seed, prepare_engine, training, config['density_kg_m3'])[0]
    training_requirements = resolve_requirements(config.get('requirements'), original_training_metrics, seed_training_metrics)
    warm, warm_audit = (_load_warm_starts(config, checker, training_requirements)
                        if search.get('warm_start_runs') else ([], []))
    save_json(details/'benchmark_reference.json', {'original_metrics': original_metrics,
        'seed_metrics': seed_metrics, 'requirements': requirements,
        'original_training_metrics': original_training_metrics, 'seed_training_metrics': seed_training_metrics,
        'training_requirements': training_requirements,
        'threshold_reference_scope': 'independent full-disk area mean; reference extrema refined identically to candidates when enabled',
        'training_threshold_scope': 'relative limits use same training-pointset reference, explicit limits unchanged',
        'seed_design': seed.to_dict(), 'raw_seed_sha256': raw_design_sha256(seed),
        'canonical_seed_sha256': canonical_physical_sha256(seed), 'warm_start_audit': warm_audit,
        'preparation_engine_statistics': dict(prepare_engine.statistics)})
    rows = []
    for random_seed in seeds:
        for mode in ('fixed', 'mixed'):
            run_started = time.perf_counter()
            folder = details/('%s_seed_%d' % (mode, random_seed))
            folder.mkdir(parents=True, exist_ok=True)
            local_search = copy.deepcopy(search)
            if mode == 'fixed':
                for flag in ('optimize_azimuth_count', 'optimize_radial_ring_count', 'optimize_magnetization'):
                    local_search[flag] = False
                topologies = [seed]
            else:
                topologies = propose_topologies(source, seed, process, limits, local_search,
                    np.random.default_rng(random_seed), int(local_search.get('topology_limit', 1)))
            local_warm = list(warm)+list(getattr(topologies, 'warm_starts', []))
            engine = FieldEngine(**engine_options)
            def evaluate(design):
                return evaluate_design(design, engine, training, config['density_kg_m3'])[0]
            settings = dict(config, seed=random_seed, max_evaluations=budget)
            search_started = time.perf_counter()
            result = run_topology_optimization(topologies, process, limits, local_search,
                evaluate, checker, folder/'optimization', settings, training_requirements, warm_starts=local_warm)
            search_seconds = time.perf_counter()-search_started
            search_statistics = dict(engine.statistics)
            metadata = result['metadata']
            actual = int(metadata['field_evaluations'])
            if actual > budget:
                raise RuntimeError('Optimizer exceeded the benchmark field-evaluation cap')
            if not result.get('records'):
                raise RuntimeError('Benchmark optimizer returned no valid field records')
            best = min(result['records'], key=lambda row: feasibility_score(row['metrics'], training_requirements))
            design = Design.from_dict(best['design'])
            geometry = checker(design)
            validation_started = time.perf_counter()
            dense_metrics, dense_B = evaluate_design(design, engine, validation, config['density_kg_m3'])
            refined = bool(config.get('benchmark', {}).get('refine_extrema', True))
            if refined:
                dense_metrics = refine_disk_extrema(design, engine, validation, dense_metrics, dense_B, starts=1)
            validation_seconds = time.perf_counter()-validation_started
            performance = check_performance(dense_metrics, requirements)
            valid = bool(geometry['valid'] and performance['valid'])
            rounds = _run_rounds(metadata)
            row = {'mode': mode, 'optimizer_seed': random_seed, 'search_budget': budget,
                'field_evaluations': actual, 'training_sampling_sha256': training_hash,
                'validation_sampling_sha256': validation_hash, 'seed_design_sha256': raw_design_sha256(seed),
                'requirements': requirements, 'training_requirements': training_requirements, 'selected_candidate_id': best['candidate_id'],
                'training_metrics': best['metrics'], 'validation_metrics': dense_metrics,
                'mean_norm_mT': dense_metrics['mean_norm_T']*1e3, 'pp_norm_ppm': dense_metrics['pp_norm_ppm'],
                'magnet_count': design.magnet_count, 'ring_count': len(design.rings),
                'geometry_valid': bool(geometry['valid']), 'performance_pass': valid,
                'performance_check': performance, 'validation_extrema_refined': refined,
                'search_seconds': search_seconds, 'validation_seconds': validation_seconds,
                'run_seconds': time.perf_counter()-run_started,
                'search_engine_statistics': search_statistics, 'total_engine_statistics': dict(engine.statistics),
                'topologies_proposed': len(topologies), 'topologies_run': metadata.get('topologies_run'),
                'population_generations': rounds, 'external_warm_starts': len(warm),
                'topology_warm_starts': len(local_warm)-len(warm),
                'optimization_result': str((folder/'optimization'/'optimization_result.json').relative_to(public)),
                'native_fem_executed': False}
            save_json(folder/'benchmark_run.json', {'row': row, 'selected_record': best,
                'validated_design': design.to_dict(), 'geometry_check': geometry,
                'optimization_metadata': metadata, 'topology_proposal_metadata': getattr(topologies, 'metadata', {})})
            row['curves']=optimization_curves(result,training_requirements)
            initial_file=folder/'optimization/topology_01/initial_population.json'
            if initial_file.is_file():
                import json
                initial=json.loads(initial_file.read_text(encoding='utf-8'))['population']
                row['primary_initial_population_sha256']=hashlib.sha256(np.asarray(initial,dtype='<f8').tobytes()).hexdigest()
            else:row['primary_initial_population_sha256']=None
            rows.append(row)

    depth_comparisons,curve_image=compare_curves(rows,public)
    statistics = {}
    for mode in ('fixed', 'mixed'):
        selected = [r for r in rows if r['mode'] == mode]
        feasible_ppm = [r['pp_norm_ppm'] for r in selected if r['performance_pass']]
        statistics[mode] = {'runs': len(selected), 'requirements_passed': len(feasible_ppm),
            'feasible_ppm_min': min(feasible_ppm) if feasible_ppm else None,
            'feasible_ppm_max': max(feasible_ppm) if feasible_ppm else None,
            'feasible_ppm_mean': _mean(feasible_ppm),
            'search_seconds_mean': _mean([r['search_seconds'] for r in selected])}
    pairs = []
    for random_seed in seeds:
        fixed = next(r for r in rows if r['mode'] == 'fixed' and r['optimizer_seed'] == random_seed)
        mixed = next(r for r in rows if r['mode'] == 'mixed' and r['optimizer_seed'] == random_seed)
        both = fixed['performance_pass'] and mixed['performance_pass']
        improvement = ((fixed['pp_norm_ppm']-mixed['pp_norm_ppm'])/fixed['pp_norm_ppm']*100
                       if both and fixed['pp_norm_ppm'] > 0 else None)
        pairs.append({'optimizer_seed': random_seed, 'both_passed_requirements': bool(both),
                      'mixed_ppm_improvement_percent': improvement})
    csv_path, html_path, json_path = public/'优化对照.csv', public/'benchmark_中文.html', public/'benchmark_summary.json'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['模式', '优化随机种子', '搜索评估上限', '实际搜索评估', '复核平均场_mT',
                         '复核峰峰值_ppm', '磁铁数', '径向环总数', '门槛通过', '搜索秒', '复核秒', '停止状态'])
        for row in rows:
            writer.writerow([row['mode'], row['optimizer_seed'], budget, row['field_evaluations'],
                row['mean_norm_mT'], row['pp_norm_ppm'], row['magnet_count'], row['ring_count'],
                row['performance_pass'], row['search_seconds'], row['validation_seconds'],
                '; '.join(str(r['state']) for r in row['population_generations'])])
    documents = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MRI 优化对照</title>',
        '<style>body{font-family:Microsoft YaHei,sans-serif;margin:36px;line-height:1.7;color:#20324a}'
        'table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #cdd5df;text-align:left}'
        'th{background:#edf3f9}.note{padding:16px;background:#fff6df}a{color:#175a96}</style>',
        '<h1>固定拓扑与混合拓扑 · 同预算对照</h1>',
        '<p>固定同一平面圆盘 FOV、全圆盘训练点集、独立密集复核点集、打印起点和性能门槛。随机种子仅改变优化提案。'
        '每次搜索上限 '+str(budget)+' 次场评估，实际使用量列出。起点选择、参考计算与最终复核另计。</p>',
        '<p>场强门槛：%.6f mT。验收参考使用独立密集面积平均；相对训练门槛使用训练点集参考；每次只复核训练评分最好的一个方案。</p>'
            % (requirements['min_mean_norm_T']*1e3),
        '<p class="note">本对照为快速解析模型实验。未执行 FEM，不保证混合拓扑改善，也不保证全局最优。'
        '未达门槛的结果仅作诊断；均匀性改善百分比只在两种模式都达标时计算。预算不足完成首代会明确标记。</p>',
        '<p><a href="优化对照.csv">方案对照 CSV</a> · <a href="benchmark_summary.json">完整对照 JSON</a></p>',
        '<table><tr><th>模式 / 随机种子</th><th>实际评估 / 上限</th><th>平均场 mT</th>'
        '<th>峰峰值 ppm</th><th>磁铁 / 环</th><th>门槛</th><th>搜索秒</th><th>停止状态</th></tr>']
    for row in rows:
        status = '; '.join(str(r['state']) for r in row['population_generations'])
        documents.append('<tr><td>%s / %d</td><td>%d / %d</td><td>%.6f</td><td>%.1f</td>'
            '<td>%d / %d</td><td>%s</td><td>%.2f</td><td>%s</td></tr>' % (
                html.escape(row['mode']), row['optimizer_seed'], row['field_evaluations'], budget,
                row['mean_norm_mT'], row['pp_norm_ppm'], row['magnet_count'], row['ring_count'],
                '通过' if row['performance_pass'] else '未达门槛', row['search_seconds'], html.escape(status)))
    if curve_image:
        documents.append('<p>左图按总求场预算比较；右图按主拓扑完成的DE代数比较（只在共同代数范围内评估）。后者不含局部精修，初始化/热启动可能不同，不代表已消除全部实验差异。</p><img style="max-width:100%" src="'+curve_image+'"><p><a href="双口径收敛曲线.json">两条曲线的原始数值</a></p>')
    documents.append('</table><p>种群大小、实际代数、求场点数和每轮原始结果见“详细数据”。</p></html>')
    html_path.write_text(''.join(documents), encoding='utf-8')
    result = {'schema': 'mri-optimization-benchmark-v2', 'report_html': str(html_path),
        'csv': str(csv_path), 'summary_json': str(json_path), 'rows': rows, 'mode_statistics': statistics,
        'paired_comparisons': pairs, 'primary_depth_comparisons':depth_comparisons,'curve_image':curve_image, 'requirements': requirements, 'training_requirements': training_requirements, 'optimizer_seeds': seeds,
        'search_budget_per_run': budget, 'training_sampling_sha256': training_hash,
        'validation_sampling_sha256': validation_hash, 'training_domain': 'full_planar_disk', 'fov_geometry':'disk', 'fov_diameter_mm':config['fov_diameter_mm'],'fov_z_mm':config['fov_z_mm'],
        'seed_selection_cost': selection, 'reference_original_metrics': original_metrics,
        'reference_seed_metrics': seed_metrics, 'elapsed_seconds': time.perf_counter()-started,
        'global_optimum_proven': False, 'native_fem_executed': False,
        'limitations': ['One training-best candidate per run is independently validated.',
            'Actual evaluations may be lower than the equal cap; seed/reference/validation costs are separate.',
            'Small budgets can interrupt initialization and do not establish algorithm convergence.',
            'Full disk training points are used even if ordinary optimization enables a verified angular sector.']}
    save_json(json_path, result)
    return result

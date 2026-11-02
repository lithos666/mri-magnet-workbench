"""Successive-halving islands over audited, physically distinct topologies.

The existing fixed-topology solver remains the only continuous optimizer.
Rounds restart DE around saved incumbents (they do not pretend to resume a
SciPy population). Original parameter windows stay fixed across all rounds.
Optional process workers have independent field engines and numerical caches.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import queue
import time

import numpy as np

from .geometry import Design
from .objectives import feasibility_score
from .optimization import run_optimization, pareto_front
from .parameterization import get_search_space
from .serialization import save_json, jsonable
from .solver_budget import (round_sizes as _round_sizes, split_budget as _split_budget,
                           allocate_round_pool, effective_refinement_budgets, plan_island_budgets)


_SOLVER_OPTIONS = {
    'population_size', 'initial_local_fraction', 'initial_sigma',
    'initial_scale_mode', 'initial_sigma_mm', 'local_refine_evaluations',
    'local_step', 'pair_refine_fraction', 'pair_max_trials', 'warm_local_fraction',
    'active_set_evaluations','active_set_trust_radius','active_set_fd_step',
}


def _integer(value, name, minimum=1):
    try:
        valid = not isinstance(value, bool) and math.isfinite(float(value)) and int(value) == value and value >= minimum
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _as_design(value):
    return Design.from_dict(value) if isinstance(value, dict) else value


def _run_job(job):
    """Picklable worker entry; construct closures only inside the worker."""
    topology = _as_design(job['topology'])
    variables, builder = get_search_space(topology, job['process'], job['limits'], job['search'])
    compatible = []
    accepted_sources = []
    for source, data in job['warm_starts']:
        design = _as_design(data)
        try:
            encoded = np.asarray(builder.encode_design(design), float)
            bounds = [(v['lower'], v['upper']) for v in variables]
            if len(encoded) != len(bounds) or any(
                    x < lo - 1e-10 or x > hi + 1e-10 for x, (lo, hi) in zip(encoded, bounds)):
                continue
            if not job['checker'](design).get('valid', False):
                continue
        except (ValueError, AttributeError, ArithmeticError):
            continue
        compatible.append(design)
        accepted_sources.append(source)
    event_queue = job.get('event_queue')
    callback = job.get('direct_callback')
    if event_queue is not None:
        def callback(event):
            event_queue.put((job['island'], event))
    if callback:
        callback({'event':'island_status','island_status':'running','stage_name':'initialization',
                  'field_evaluations':0,'allocation':job['allocation']})
    options = deepcopy(job['options'])
    # Screening spends its allocation on DE. Final rounds reserve local work
    # only after initialization plus at least one complete generation can fit.
    requested_local = int(options.get('local_refine_evaluations', 0))
    requested_lp=int(options.get('active_set_evaluations',0))
    options['local_refine_evaluations'],options['active_set_evaluations']=effective_refinement_budgets(
        job['allocation'],job['population_size'],requested_local,requested_lp,job['final_round'])
    # Retained elites drive a physical local neighborhood after the first round;
    # identical-topology migrants pass the same encoder and all manufacturing
    # checks. Heterogeneous vectors are never copied by coordinate index.
    initial_design=compatible[0] if compatible else None
    if initial_design is not None:
        # All local rows are centered on the saved elite; other warm starts
        # remain explicit points, never a second competing local center.
        options['warm_local_fraction']=0.
    oracle=getattr(job['evaluator'],'field_values',None)
    weights=getattr(job['evaluator'],'sample',{}).get('weights')
    def should_stop():
        return bool(job.get('stop_file') and Path(job['stop_file']).exists())
    cache = dict(job['cache'])
    result = run_optimization(
        variables, builder, job['evaluator'], job['checker'], topology,
        job['output_dir'], seed=job['seed'], max_evaluations=job['allocation'],
        popsize=job['popsize'], maxiter=job['maxiter'],
        requirements=job['requirements'], numeric_cache=cache,
        warm_starts=compatible, candidate_prefix=job['prefix'],
        event_callback=callback, topology_index=job['island'],
        global_evaluation_offset=0,initial_design=initial_design,
        field_values=oracle,field_weights=weights,
        graceful_stop=job.get('graceful_stop',False),
        should_stop=should_stop if job.get('graceful_stop') else None, **options)
    result['metadata']['island_round'] = {
        'requested_local_refine_evaluations': requested_local,
        'effective_local_refine_evaluations': options['local_refine_evaluations'],
        'requested_active_set_evaluations':requested_lp,'effective_active_set_evaluations':options['active_set_evaluations'],
        'initial_center_source':accepted_sources[0] if compatible else 'original_topology',
        'compatible_warm_sources': accepted_sources,
        'continuation': 'restart_with_saved_incumbents_in_original_parameter_windows',
    }
    if callback:
        callback({'event':'island_status','island_status':'stopped' if str(result.get('state','')).startswith('stopped') else 'completed',
                  'stage_name':'completed','field_evaluations':result['metadata']['field_evaluations'],
                  'allocation':job['allocation']})
    return {'result': result, 'cache': cache}


def run_islands(topologies, process, limits, search, evaluator, checker, output_dir,
                settings, requirements, event_callback=None):
    """Optimize audited seeds, eliminate weaker basins, and deepen finalists.

    ``settings['islands']`` accepts ``max_workers`` (default 1),
    ``halving_factor`` (2), ``finalists`` (2),
    ``initial_budget_fraction`` (.4), ``min_round_evaluations`` (0), and
    ``migration_count`` (2). Default population size is 16 unless explicitly
    supplied in ``optimization_options``. A budget too small to initialize
    and evolve every scheduled island round is rejected before any field call.

    Process mode requires picklable top-level evaluator/checker callables.
    Events are delivered by the parent only; their global counter is the sum
    of observed completed worker evaluations, monotone even with concurrency.
    Dense validation, synthesis, and monitor field work are separate budgets.
    """
    started = time.perf_counter()
    topologies = [_as_design(value) for value in topologies]
    if not topologies:
        raise ValueError('At least one audited topology seed is required')
    total = _integer(settings['max_evaluations'], 'max_evaluations')
    config = dict(settings.get('islands', {}))
    allowed = {'max_workers', 'halving_factor', 'finalists', 'initial_budget_fraction',
               'min_round_evaluations', 'migration_count','graceful_stop','stop_file','budget_policy'}
    if set(config) - allowed:
        raise ValueError('Unknown islands settings: ' + str(sorted(set(config) - allowed)))
    workers = _integer(config.get('max_workers', 1), 'max_workers')
    factor = _integer(config.get('halving_factor', 2), 'halving_factor', 2)
    finalists = min(len(topologies), _integer(config.get('finalists', 2), 'finalists'))
    migration_count = _integer(config.get('migration_count', 2), 'migration_count', 0)
    fraction = float(config.get('initial_budget_fraction', .4))
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('initial_budget_fraction must be in (0,1)')
    options = dict(settings.get('optimization_options', {}))
    options.pop('primary_topology_fraction', None)  # legacy outer allocation only
    if set(options) - _SOLVER_OPTIONS:
        raise ValueError('Unknown optimization_options: ' + str(sorted(set(options) - _SOLVER_OPTIONS)))
    options.setdefault('population_size', 16)
    population = _integer(options['population_size'], 'population_size', 5)
    options.setdefault('initial_local_fraction', .7)
    options.setdefault('initial_scale_mode', 'physical_mm')
    options.setdefault('initial_sigma_mm', .8)
    options.setdefault('local_refine_evaluations', 80)
    options.setdefault('pair_refine_fraction', .4)
    minimum = max(2 * population,
                  _integer(config.get('min_round_evaluations', 0), 'min_round_evaluations', 0))
    sizes = _round_sizes(len(topologies), finalists, factor)
    minimum_budget = minimum * sum(sizes)
    if total < minimum_budget:
        raise ValueError(
            'Island budget too small: need at least %d evaluations for rounds %s '
            'with population_size=%d. Increase max_evaluations, reduce selected seeds '
            'or explicitly reduce population_size (minimum 5).' %
            (minimum_budget, sizes, population))
    budget_plan=None
    if 'budget_policy' in config:
        budget_plan=plan_island_budgets(topologies,process,limits,search,settings,requirements)
        if not budget_plan['valid']:
            raise ValueError('求解预算预检未通过：'+'；'.join(budget_plan['violations']))
        if (config['budget_policy']=='strict' and options.get('active_set_evaluations',0)>0
                and budget_plan['max_dimensions']>0 and not callable(getattr(evaluator,'field_values',None))):
            raise ValueError('求解预算预检未通过：请求灵敏度LP精修，但求场器未提供完整矢量场接口')
    # Fail before starting any worker or field solve for invalid seeds/callables.
    for topology in topologies:
        report = checker(topology)
        if not report.get('valid', False):
            raise ValueError('An island seed violates geometry/manufacturing constraints: ' + str(report))
        get_search_space(topology, process, limits, search)
    if workers > 1:
        try:
            pickle.dumps((evaluator, checker))
        except Exception as error:
            raise ValueError('Process islands need picklable top-level evaluator/checker callables') from error
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    used = 0
    active = list(range(len(topologies)))
    caches = {i: {} for i in active}
    elites = {i: [] for i in active}
    all_records, rounds, runs, stage_summaries = [], [], [], []
    monitor_errors = []
    callback_enabled = event_callback is not None
    first_baseline = None
    stopped=False
    manager = multiprocessing.Manager() if workers > 1 and event_callback else None
    try:
        for round_index, count in enumerate(sizes):
            final_round = round_index == len(sizes) - 1
            remaining = total - used
            pool = allocate_round_pool(remaining,count,sizes[round_index+1:],minimum,fraction,final_round)
            allocations = _split_budget(pool, count)
            round_start = used
            observed = {i: 0 for i in active}
            jobs = []
            event_queue = manager.Queue() if manager else None

            def forward(island, event):
                nonlocal callback_enabled
                observed[island - 1] = max(observed[island - 1], int(event['field_evaluations']))
                if not callback_enabled:
                    return
                value = dict(event, topology_index=island, island_index=island,
                             island_round=round_index + 1,
                             global_evaluations=round_start + sum(observed.values()),
                             global_counter_scope='observed_completed_field_evaluations')
                try:
                    event_callback(value)
                except Exception as error:
                    callback_enabled = False
                    monitor_errors.append(str(error))

            for position, island in enumerate(active):
                forward(island+1,{'event':'island_status','island_status':'queued','stage_name':'initialization',
                    'field_evaluations':0,'allocation':allocations[position]})
                warm = [('own_incumbent', row['design']) for row in elites[island][:migration_count or 1]]
                other = sorted((row for key in active if key != island for row in elites[key][:1]),
                               key=lambda row: feasibility_score(row['metrics'], requirements))
                warm.extend(('other_island_' + str(row['topology_index']), row['design'])
                            for row in other[:migration_count])
                folder = output / ('island_%02d' % (island + 1)) / ('round_%02d' % (round_index + 1))
                jobs.append({
                    'island': island + 1, 'topology': topologies[island].to_dict(),
                    'process': process, 'limits': limits, 'search': search,
                    'evaluator': evaluator, 'checker': checker, 'output_dir': str(folder),
                    'options': options, 'population_size': population,
                    'allocation': allocations[position], 'final_round': final_round,
                    'requirements': requirements, 'cache': caches[island], 'warm_starts': warm,
                    'seed': int(settings.get('seed', 20261101)) + 1009 * round_index + island,
                    'popsize': settings.get('popsize', 3), 'maxiter': settings.get('maxiter', 100),
                    'prefix': 'i%02d_r%02d_' % (island + 1, round_index + 1),
                    'event_queue': event_queue,
                    'graceful_stop':bool(config.get('graceful_stop',False)), 'stop_file':config.get('stop_file'),
                })
            completed = {}
            if workers == 1:
                for job in jobs:
                    if event_callback:
                        job['direct_callback'] = lambda event, i=job['island']: forward(i, event)
                    value = _run_job(job)
                    completed[job['island'] - 1] = value
                    observed[job['island'] - 1] = value['result']['metadata']['field_evaluations']
            else:
                # The parent owns the callback/Matplotlib; workers own fields.
                with ProcessPoolExecutor(max_workers=min(workers, count)) as executor:
                    pending = {executor.submit(_run_job, job): job['island'] - 1 for job in jobs}
                    while pending:
                        if event_queue is not None:
                            while True:
                                try:
                                    island, event = event_queue.get_nowait()
                                    forward(island, event)
                                except queue.Empty:
                                    break
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            island = pending.pop(future)
                            completed[island] = future.result()
                            observed[island] = completed[island]['result']['metadata']['field_evaluations']
                    if event_queue is not None:
                        while True:
                            try:
                                island, event = event_queue.get_nowait()
                                forward(island, event)
                            except queue.Empty:
                                break
            rows_this_round = []
            # Records use a deterministic budget ledger order; wall-clock
            # concurrency is explicitly reserved for the live event counters.
            ledger = round_start
            for job in jobs:
                island = job['island'] - 1
                item = completed[island]
                result = item['result']
                if result.get('state','').startswith('stopped_'):stopped=True
                cost = int(result['metadata']['field_evaluations'])
                if cost > job['allocation']:
                    raise RuntimeError('Inner solver exceeded its island allocation')
                caches[island] = item['cache']
                for row in result['records']:
                    row['topology_index'] = island + 1
                    row['island_round'] = round_index + 1
                    row['global_evaluations'] = ledger + row['field_evaluations']
                    all_records.append(row)
                for summary in result['metadata'].get('stage_summaries', []):
                    summary['topology_index'] = island + 1
                    summary['island_round'] = round_index + 1
                    summary['global_evaluations'] = ledger + summary['field_evaluations']
                    stage_summaries.append(summary)
                if first_baseline is None:
                    first_baseline = result['baseline']
                combined = elites[island] + result['records']
                distinct = {}
                for row in combined:
                    distinct.setdefault(row['canonical_physical_sha256'], row)
                elites[island] = sorted(distinct.values(),
                    key=lambda row: feasibility_score(row['metrics'], requirements))[:max(2, migration_count)]
                if not elites[island]:
                    ledger+=cost;used+=cost
                    runs.append({'topology':island+1,'island_round':round_index+1,'metadata':result['metadata']})
                    continue
                best = elites[island][0]
                inner_round = (result['metadata'].get('rounds') or [{}])[0]
                local = inner_round.get('local_refinement', {})
                rows_this_round.append({
                    'island': island + 1, 'allocation': job['allocation'], 'field_evaluations': cost,
                    'population_size': population, 'active_dimension': result['metadata']['active_dimension'],
                    'de_generations': inner_round.get('generations', 0),
                    'initialization_complete': inner_round.get('initialization_complete', False),
                    'local_field_evaluations': local.get('field_evaluations', 0),
                    'paired_field_evaluations': local.get('paired_field_evaluations', 0),
                    'paired_accepted_moves': local.get('paired_accepted_moves', 0),
                    'best_candidate_id': best['candidate_id'], 'best_metrics': best['metrics'],
                    'score': feasibility_score(best['metrics'], requirements),
                    'compatible_warm_sources': result['metadata']['island_round']['compatible_warm_sources'],
                })
                runs.append({'topology': island + 1, 'island_round': round_index + 1,
                             'ring_count': len(topologies[island].rings),
                             'magnet_count': topologies[island].magnet_count,
                             'relative_result': str(Path(job['output_dir']).relative_to(output) / 'optimization_result.json'),
                             'metadata': result['metadata']})
                ledger += cost
                used += cost
            ranked = sorted((i for i in active if elites[i]), key=lambda i: (
                feasibility_score(elites[i][0]['metrics'], requirements), i))
            next_active = ranked if final_round else ranked[:sizes[round_index + 1]]
            for island in active:
                status=('stopped' if stopped else 'completed' if final_round else
                        'promoted' if island in next_active else 'eliminated')
                forward(island+1,{'event':'island_status','island_status':status,'stage_name':'completed',
                    'field_evaluations':observed[island],'allocation':allocations[active.index(island)]})
            rounds.append({'round': round_index + 1,
                           'stage': 'finalists_de_and_compensated_polish' if final_round else 'successive_halving_screen',
                           'islands': rows_this_round,
                           'promoted_islands': [] if stopped else [i + 1 for i in next_active],
                           'eliminated_islands': [i + 1 for i in ranked if i not in next_active],
                           'field_evaluations_total': used})
            active = next_active
            save_json(output / 'island_progress.json', {'rounds': rounds, 'field_evaluations': used,
                                                        'max_evaluations': total})
            if stopped:break
    finally:
        if manager is not None:
            manager.shutdown()
    unique = {}
    for row in all_records:
        unique.setdefault(row['canonical_physical_sha256'], row)
    records = list(unique.values())
    metadata = {
        'algorithm': 'multi-basin successive-halving islands / fixed-topology DE / flux-compensated polish',
        'max_evaluations': total, 'field_evaluations': used, 'remaining_evaluations': total - used,
        'minimum_scheduled_budget': minimum_budget, 'population_size': population,
        'island_round_sizes': sizes, 'island_rounds': rounds, 'runs': runs,
        'stage_summaries': stage_summaries,
        'topologies_proposed': len(topologies),
        'topologies_run':len({r['topology'] for r in runs if r['metadata'].get('field_evaluations',0)>0}),
        'topology_jobs_started':len(runs),
        'finalist_islands': [i + 1 for i in active],
        'execution': 'process_parallel' if workers > 1 else 'sequential_memory_conservative',
        'max_workers': workers, 'optimization_options': options,
        'continuation': 'DE restart with audited incumbents; original parameter windows retained',
        'migration': 'physical designs accepted only by target topology encoder and manufacturing audit',
        'record_global_order': 'deterministic island ledger; live counters track observed completed work',
        'field_budget_excludes_seed_selection_and_dense_validation': True,
        'field_budget_excludes_monitor_preview': True,
        'global_optimum_proven': False, 'independent_dense_validation_required': True,
        'monitor_errors': monitor_errors, 'elapsed_seconds': time.perf_counter() - started,
        'budget_preflight':budget_plan,
        'graceful_stop_enabled':bool(config.get('graceful_stop',False)),
        'stopped_with_completed_records':bool(stopped and records),
        'files': {'result': 'optimization_result.json', 'progress': 'island_progress.json'},
    }
    result = {'state': ('stopped_with_result' if records else 'stopped_no_result') if stopped else 'budget_exhausted' if used == total else 'scheduled_rounds_finished',
              'baseline': first_baseline, 'records': records, 'pareto': pareto_front(records),
              'diagnostic_best': min(records, key=lambda row: feasibility_score(row['metrics'], requirements))
              if records else None, 'metadata': metadata}
    save_json(output / 'optimization_result.json', result)
    with (output / 'optimization_records.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + '\n')
    return result

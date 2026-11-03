"""Budgeted epsilon-constrained DE inside explicitly selected topologies."""
import hashlib
from copy import deepcopy
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from .objectives import check_performance, feasibility_score
from .serialization import jsonable, save_json
from .identity import canonical_physical_sha256, raw_design_sha256
from .search_moves import reflect_interval, compensated_steps
from .metrics import field_metrics


class _BudgetReached(Exception):
    pass


class _GracefulStop(BaseException):
    pass


def _design_dict(design):
    return jsonable(design.to_dict() if hasattr(design, 'to_dict') else dict(design))


def _design_key(design):
    value = _design_dict(design) if not isinstance(design, dict) else design
    if 'rings' in value: return canonical_physical_sha256(value)
    physical = {k: v for k, v in value.items() if k != 'name'}
    return hashlib.sha256(json.dumps(physical, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pareto_front(records):
    valid, seen = [], set()
    for row in records:
        if not row.get('feasible', False) or not row.get('performance_check', {}).get('valid', True): continue
        try:
            field, ppm = (float(row['metrics'][k]) for k in ('mean_norm_T', 'pp_norm_ppm'))
        except (TypeError, ValueError, KeyError): continue
        if not math.isfinite(field) or not math.isfinite(ppm) or field < 0 or ppm < 0: continue
        key = row.get('canonical_physical_sha256')
        if key and key in seen: continue
        if key: seen.add(key)
        valid.append(row)
    result = []
    for row in valid:
        field, ppm = row['metrics']['mean_norm_T'], row['metrics']['pp_norm_ppm']
        if not any(o['metrics']['mean_norm_T'] >= field and o['metrics']['pp_norm_ppm'] <= ppm
                   and (o['metrics']['mean_norm_T'] > field or o['metrics']['pp_norm_ppm'] < ppm) for o in valid):
            result.append(row)
    return sorted(result, key=lambda r: (r['metrics']['pp_norm_ppm'], -r['metrics']['mean_norm_T'], r['candidate_id']))


def _prepare_variables(variables):
    result = []
    for original in variables:
        item = dict(original)
        lo, hi, initial = (float(item[k]) for k in ('lower', 'upper', 'initial'))
        if not all(math.isfinite(v) for v in (lo, hi, initial)) or not lo <= initial <= hi:
            raise ValueError('Variable initial value must be inside finite ordered bounds')
        if item.get('integer', False) and (not initial.is_integer() or math.ceil(lo) > math.floor(hi)):
            raise ValueError('Integer variable has no legal initial value')
        item.update(lower=lo, upper=hi, initial=initial, integer=bool(item.get('integer', False)))
        result.append(item)
    if len({r['name'] for r in result}) != len(result): raise ValueError('Variable names must be unique')
    return result


def initialize_population(variables, rng, size, builder=None, local_fraction=0., local_sigma=.06,
                          scale_mode='normalized',sigma_mm=.8,center_design=None):
    """Mix feasible global draws with reflected normalized or physical-mm neighbours.

    Integer dimensions keep the inclusive global prior. Only continuous active
    dimensions receive local perturbations. Physical mode delegates to the
    builder so radial and axial motions use millimetres before inverse encoding. The exact start is inserted by the
    caller, and all proposals still pass the full geometry audit before solving.
    """
    if not math.isfinite(local_fraction) or not 0 <= local_fraction <= 1:
        raise ValueError('initial_local_fraction must be in [0,1]')
    if not math.isfinite(local_sigma) or local_sigma <= 0:
        raise ValueError('initial_sigma must be positive and finite')
    if hasattr(builder, 'initial_sampler'):
        population = np.asarray(builder.initial_sampler(rng, size, include_initial=False), float)
    else:
        population = np.empty((size, len(variables)))
        for i, v in enumerate(variables):
            population[:, i] = (rng.integers(math.ceil(v['lower']), math.floor(v['upper'])+1, size=size)
                                if v.get('integer', False) else rng.uniform(v['lower'], v['upper'], size=size))
    if population.shape != (size, len(variables)) or not np.all(np.isfinite(population)):
        raise ValueError('Invalid initial population supplied by builder')
    count = min(max(0, size-1), int(round(max(0, size-1)*local_fraction)))
    continuous = [i for i,v in enumerate(variables) if not v.get('integer',False) and v['upper']>v['lower']]
    if scale_mode not in ('normalized','physical_mm'):raise ValueError('Invalid initial_scale_mode')
    if not math.isfinite(sigma_mm) or sigma_mm<=0:raise ValueError('initial_sigma_mm must be positive')
    if count and scale_mode=='physical_mm':
        if not hasattr(builder,'local_sampler'):raise ValueError('physical_mm initialization needs a physical builder.local_sampler')
        local=np.asarray(builder.local_sampler(rng,count,sigma_mm,**({'center_design':center_design} if center_design is not None else {})),float)
        if local.shape!=(count,len(variables)) or not np.all(np.isfinite(local)):
            raise ValueError('Invalid physical local population')
        population[1:1+count]=local
    else:
        for i in continuous:
            v = variables[i]
            population[1:1+count,i] = reflect_interval(v['initial'] + rng.normal(0,local_sigma,size=count)
                * (v['upper']-v['lower']), v['lower'],v['upper'])
    return population


def _run_optimization_impl(variables, build_design, evaluate_design, validate_design, baseline_design,
                     output_dir, seed=20261028, max_evaluations=300, weights=None,
                     popsize=3, maxiter=30, requirements=None, numeric_cache=None,
                     warm_starts=(), candidate_prefix='', population_size=None,
                     initial_local_fraction=0., initial_sigma=.06,
                     local_refine_evaluations=0, local_step=.03,
                     initial_scale_mode='normalized',initial_sigma_mm=.8,
                     pair_refine_fraction=0.,pair_max_trials=12,event_callback=None,
                     topology_index=1,global_evaluation_offset=0,warm_local_fraction=0.,
                     initial_design=None,field_values=None,field_weights=None,
                     active_set_evaluations=0,active_set_trust_radius=.03,active_set_fd_step=.002,
                     should_stop=None,_runtime_state=None):
    """One topology, strict epsilon ordering, optional compact DE plus local poll.

    A small budget can interrupt initialization, recorded explicitly. ``weights``
    is a compatibility input and cannot change the epsilon objective. Shared
    numerical cache entries contain metrics, never another candidate's metadata.
    """
    started = time.perf_counter(); variables = _prepare_variables(variables)
    for name, value, lower in (('max_evaluations', max_evaluations, 1), ('popsize', popsize, 1), ('maxiter', maxiter, 0)):
        if isinstance(value, bool) or int(value) != value or value < lower:
            raise ValueError(name+' must be a valid integer')
    max_evaluations, popsize, maxiter = int(max_evaluations), int(popsize), int(maxiter)
    if isinstance(active_set_evaluations,bool) or int(active_set_evaluations)!=active_set_evaluations or active_set_evaluations<0:
        raise ValueError('active_set_evaluations must be a nonnegative integer')
    if (isinstance(local_refine_evaluations,bool) or int(local_refine_evaluations)!=local_refine_evaluations
            or local_refine_evaluations<0): raise ValueError('local_refine_evaluations must be non-negative integer')
    if not math.isfinite(local_step) or not 0 < local_step <= 1: raise ValueError('local_step must be in (0,1]')
    if population_size is not None and (isinstance(population_size,bool) or int(population_size)!=population_size):
        raise ValueError('population_size must be an integer or None')
    # Explicit population choice never changes with the budget. Reserve local
    # evaluations only if the chosen population can first be initialized.
    if initial_scale_mode not in ('normalized','physical_mm'):raise ValueError('Invalid initial_scale_mode')
    if not math.isfinite(initial_sigma_mm) or initial_sigma_mm<=0:raise ValueError('initial_sigma_mm must be positive')
    if not math.isfinite(pair_refine_fraction) or not 0<=pair_refine_fraction<=1:
        raise ValueError('pair_refine_fraction must be in [0,1]')
    if isinstance(pair_max_trials,bool) or int(pair_max_trials)!=pair_max_trials or pair_max_trials<1:
        raise ValueError('pair_max_trials must be a positive integer')
    if not math.isfinite(warm_local_fraction) or not 0<=warm_local_fraction<=1:
        raise ValueError('warm_local_fraction must be in [0,1]')
    active_budget = max_evaluations
    phase = 'initialization'
    phase_calls = {'initialization':0,'de':0,'local':0,'paired':0}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    low = np.array([r['lower'] for r in variables]); high = np.array([r['upper'] for r in variables])
    integer = np.array([r['integer'] for r in variables], dtype=bool)
    original_initial = np.array([r['initial'] for r in variables])
    original_baseline_design=baseline_design
    if initial_design is not None:
        initial=np.asarray(build_design.encode_design(initial_design),float)
        if initial.shape!=original_initial.shape or np.any(initial<low-1e-10) or np.any(initial>high+1e-10):raise ValueError('Continuation initial design cannot be encoded in original windows')
        baseline_design=initial_design
        for variable,value in zip(variables,initial):variable['initial']=float(value)
    else:initial=original_initial.copy()
    dimensions = int(np.count_nonzero(low < high))
    cache = {} if numeric_cache is None else numeric_cache
    evaluation_namespace=getattr(evaluate_design,'evaluation_fingerprint','')
    if callable(evaluation_namespace):evaluation_namespace=evaluation_namespace()
    counters = dict(field_evaluations=0, proposal_calls=0, cache_hits=0, infeasible_proposals=0,
                    build_failures=0, evaluation_failures=0)
    records, logs, local_by_physical, vector_cache, field_cache = [], [], {}, {}, {}
    stage_summaries=[];monitor_errors=[];callback_active=event_callback is not None
    best_recorded_score=math.inf;initialization_recorded=False
    baseline_check = validate_design(baseline_design)
    if not baseline_check.get('valid', False):
        raise ValueError('The baseline design violates geometry constraints: '+str(baseline_check))
    baseline_matches = _design_key(build_design(initial.copy())) == _design_key(baseline_design)
    requested_population = max(5, popsize*dimensions) if population_size is None else int(population_size)
    if requested_population < 5: raise ValueError('DE population must contain at least five members')
    round_info = {'round': 1, 'state': 'not_started', 'generations': 0,
                  'population_size': requested_population, 'nominal_population_size': requested_population,
                  'field_evaluation_allocation': max_evaluations}
    runtime={} if _runtime_state is None else _runtime_state
    runtime.update(records=records,counters=counters,round_info=round_info,stage_summaries=stage_summaries,
        logs=logs,variables=variables,requirements=requirements or {},max_evaluations=max_evaluations,
        seed=int(seed),started=started,baseline=None,initial_center_design=_design_dict(baseline_design),
        original_topology_design=_design_dict(original_baseline_design))

    def check_stop():
        if should_stop is not None and should_stop():raise _GracefulStop()

    def pars(vector):
        return {v['name']: int(round(x)) if v['integer'] else float(x) for v, x in zip(variables, vector)}

    with (output/'optimization_records.jsonl').open('w', encoding='utf-8') as record_stream, \
         (output/'optimization_rejections.jsonl').open('w', encoding='utf-8') as rejected_stream, \
         (output/'optimization_iterations.jsonl').open('w', encoding='utf-8') as log_stream, \
         (output/'optimization_events.jsonl').open('w', encoding='utf-8') as event_stream:
        def write(stream, value):
            stream.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()

        def log(value):
            logs.append(value); write(log_stream, value)

        def emit(event,row=None,stage=None,**extra):
            nonlocal callback_active
            if row is None and records:row=min(records,key=lambda r:feasibility_score(r['metrics'],requirements))
            if row is None:return
            value={'event':event,'phase':stage or phase,'stage_name':stage or phase,
                'field_evaluations':counters['field_evaluations'],
                'global_evaluations':global_evaluation_offset+counters['field_evaluations'],'topology_index':topology_index,
                'generation':round_info['generations'],'candidate_id':row['candidate_id'],
                'design':row['design'],'metrics':row['metrics'],'feasible':row['feasible'],**extra}
            write(event_stream,value)
            if callback_active:
                try:event_callback(jsonable(value))
                except Exception as error:
                    callback_active=False
                    detail={'event':'monitor_disabled','error':str(error),'field_evaluations':counters['field_evaluations']}
                    monitor_errors.append(detail);log(detail)

        def summarize(stage):
            nonlocal initialization_recorded
            if not records:return
            if stage=='initialization':
                if initialization_recorded:return
                initialization_recorded=True
            row=min(records,key=lambda r:feasibility_score(r['metrics'],requirements))
            stage_summaries.append({'stage':stage,'field_evaluations':counters['field_evaluations'],
                'generation':round_info['generations'],'topology_index':topology_index,
                'global_evaluations':global_evaluation_offset+counters['field_evaluations'],'candidate_id':row['candidate_id'],
                'metrics':row['metrics'],'design':row['design'],'score':feasibility_score(row['metrics'],requirements)})
            emit('stage_end',row,stage)

        def rejection(vector, reason, notes):
            write(rejected_stream, {'parameters': pars(vector), 'reason': reason, 'violations': notes})

        def completed_evaluation():
            nonlocal callback_active
            # Legacy callbacks receive design-bearing events only. The V6
            # workbench explicitly opts into lightweight progress messages.
            if not callback_active or not getattr(evaluate_design,'emit_evaluation_progress',False):return
            try:
                event_callback({'event':'evaluation_progress','phase':phase,'stage_name':phase,
                    'field_evaluations':counters['field_evaluations'],
                    'global_evaluations':global_evaluation_offset+counters['field_evaluations'],
                    'topology_index':topology_index})
            except Exception as error:
                callback_active=False;monitor_errors.append(str(error))

        def visit(design, vector, baseline=False):
            nonlocal best_recorded_score
            check_stop()
            valid = baseline_check if baseline else validate_design(design)
            if not valid.get('valid', False):
                counters['infeasible_proposals'] += 1
                rejection(vector, 'geometry_constraint', valid.get('violations', [])); return None
            physical = _design_key(design)
            numeric_key=(str(evaluation_namespace)+':'+physical) if evaluation_namespace else physical
            if physical in local_by_physical:
                counters['cache_hits'] += 1; return local_by_physical[physical]
            if numeric_key in cache:
                counters['cache_hits'] += 1; metrics = dict(cache[numeric_key])
            else:
                if counters['field_evaluations'] >= active_budget: raise _BudgetReached()
                counters['field_evaluations'] += 1
                try:
                    if field_values is not None and hasattr(evaluate_design,'evaluate_with_field'):
                        metrics,B=evaluate_design.evaluate_with_field(design)
                        metrics=dict(metrics);field_cache[physical]=np.asarray(B,float)
                    elif field_values is not None:
                        B=np.asarray(field_values(design),float);metrics=field_metrics(B,field_weights)
                        field_cache[physical]=B
                        if hasattr(design,'magnet_count'):metrics['magnet_count']=design.magnet_count
                    else:metrics = dict(evaluate_design(design))
                    if not math.isfinite(check_performance(metrics, requirements)['normalized_violation']):
                        raise ValueError('Invalid field metrics')
                except Exception as error:
                    counters['evaluation_failures'] += 1
                    completed_evaluation()
                    rejection(vector, 'evaluation_error', [str(error)]); return None
                cache[numeric_key] = dict(metrics)
                completed_evaluation()
            performance = check_performance(metrics, requirements)
            ident = candidate_prefix+('baseline' if baseline else 'candidate_%06d' % len(records))
            row = {'candidate_id': ident, 'parameters': pars(vector), 'design': _design_dict(design),
                   'metrics': metrics, 'evaluation_fingerprint':evaluation_namespace,
                   'geometry_feasible': True, 'performance_check': performance,
                   'feasible': performance['valid'], 'canonical_physical_sha256': physical,
                   'raw_design_sha256': raw_design_sha256(_design_dict(design)),
                   'search_stage':phase,'field_evaluations':counters['field_evaluations'],
                   'global_evaluations':global_evaluation_offset+counters['field_evaluations'],'topology_index':topology_index}
            records.append(row); local_by_physical[physical] = row; write(record_stream, row)
            score=feasibility_score(metrics,requirements)
            if baseline:emit('baseline',row)
            if score<best_recorded_score:
                best_recorded_score=score
                if not baseline:emit('new_best',row)
            return row

        baseline = visit(baseline_design, initial, baseline=True)
        runtime['baseline']=baseline
        if baseline is None: raise ValueError('The baseline must return finite positive field metrics')
        if baseline_matches: vector_cache[tuple(initial)] = baseline

        def objective(raw):
            nonlocal phase
            check_stop()
            if phase=='initialization' and phase_calls['initialization']>=requested_population:
                summarize('initialization');phase='de'
            counters['proposal_calls'] += 1
            phase_calls[phase] += 1
            vector = np.clip(np.asarray(raw, float), low, high)
            vector[integer] = np.rint(vector[integer]); key = tuple(vector)
            if key in vector_cache:
                counters['cache_hits'] += 1; row = vector_cache[key]
            else:
                try: design = build_design(vector.copy())
                except (ValueError, ArithmeticError) as error:
                    counters['build_failures'] += 1
                    rejection(vector, 'build_error', [str(error)]); vector_cache[key] = None; return math.inf
                row = visit(design, vector); vector_cache[key] = row
            return math.inf if row is None else feasibility_score(row['metrics'], requirements)

        def callback(intermediate_result):
            nonlocal phase
            check_stop()
            convergence=intermediate_result.convergence
            phase = 'de'
            round_info['generations'] += 1
            best = min(records, key=lambda r: feasibility_score(r['metrics'], requirements))
            members=[];missing=0
            for raw in intermediate_result.population:
                vector=np.clip(np.asarray(raw,float),low,high);vector[integer]=np.rint(vector[integer])
                row=vector_cache.get(tuple(vector))
                if row is None:missing+=1;continue
                members.append({'candidate_id':row['candidate_id'],'mean_norm_T':row['metrics']['mean_norm_T'],
                    'pp_norm_ppm':row['metrics']['pp_norm_ppm'],'feasible':row['feasible']})
            emit('generation',best,stage='de',population_metrics=members,
                population_scope='actual_scipy_de_population_cached_metrics',population_unresolved=missing)
            log({'event': 'generation', 'generation': round_info['generations'],
                 'field_evaluations_total': counters['field_evaluations'], 'best_candidate_id': best['candidate_id'],
                 'best_score': feasibility_score(best['metrics'], requirements), 'best_metrics':best['metrics'], 'convergence': float(convergence)})
            return False

        log({'event': 'round_start', **round_info})
        if not dimensions: round_info['state'] = 'no_free_variables'
        elif counters['field_evaluations'] >= max_evaluations: round_info['state'] = 'budget_exhausted'
        else:
            population = initialize_population(variables, rng, requested_population, build_design,
                                               initial_local_fraction,initial_sigma,initial_scale_mode,initial_sigma_mm,
                                               center_design=initial_design)
            seed_sampling_statistics=deepcopy(getattr(getattr(build_design,'local_sampler',None),'last_statistics',None))
            population[0] = initial; warm_count = 0; warm_rows = [0]; encoded_warms=[]
            local_rows = int(round((requested_population-1)*initial_local_fraction))
            warm_limit = max(0,(requested_population-1-local_rows)//2) if initial_local_fraction else max(1,requested_population//2-1)
            if warm_local_fraction and local_rows:warm_limit=max(1,warm_limit)
            for warm in warm_starts:
                try:
                    encoded = np.asarray(build_design.encode_design(warm), float)
                    if np.all(encoded >= low-1e-10) and np.all(encoded <= high+1e-10):
                        if warm_count >= warm_limit:break
                        if any(np.array_equal(encoded,population[row]) for row in warm_rows):continue
                        position = requested_population-1-warm_count if initial_local_fraction else 1+warm_count
                        population[position] = encoded; warm_rows.append(position); warm_count += 1
                        encoded_warms.append((encoded.copy(),position))
                except (ValueError, AttributeError): continue
            # The supplied ordering is explicit: no unbudgeted field solve is
            # performed to rank warm centers. Use the first encodable warm.
            # In an all-local population, reserve the tail for warm center
            # single points rather than silently overwriting their neighbors.
            local_rows=min(local_rows,requested_population-1-warm_count)
            warm_local_rows=min(local_rows,int(round(local_rows*warm_local_fraction))) if encoded_warms else 0
            local_center_audit=[];warm_sampling_statistics=None
            if warm_local_rows:
                center_vector,center_row=encoded_warms[0]
                if initial_scale_mode=='physical_mm':
                    center_design=build_design(center_vector)
                    local=np.asarray(build_design.local_sampler(rng,warm_local_rows,initial_sigma_mm,
                                                               center_design=center_design),float)
                    warm_sampling_statistics=deepcopy(getattr(build_design.local_sampler,'last_statistics',None))
                else:
                    local=population[1:1+warm_local_rows].copy()
                    for i,v in enumerate(variables):
                        if not integer[i] and high[i]>low[i]:
                            local[:,i]=reflect_interval(center_vector[i]+rng.normal(0,initial_sigma,size=warm_local_rows)
                                                        *(high[i]-low[i]),low[i],high[i])
                if local.shape!=(warm_local_rows,len(variables)) or not np.all(np.isfinite(local)):
                    raise ValueError('Invalid warm-centered local population')
                population[1:1+warm_local_rows]=local
                local_center_audit.append({'source':'first_encodable_input_warm','selection':'input_order_not_score_ranked',
                    'parameters':pars(center_vector),'sample_rows':list(range(1,1+warm_local_rows)),
                    'single_point_row':center_row,'physical_sampling_statistics':warm_sampling_statistics})
            if local_rows>warm_local_rows:
                local_center_audit.append({'source':'continuation_elite' if initial_design is not None else 'manufacturing_seed','parameters':pars(initial),
                    'sample_rows':list(range(1+warm_local_rows,1+local_rows)),
                    'physical_sampling_statistics':seed_sampling_statistics})
            reserve = min(int(local_refine_evaluations), max(0,max_evaluations-requested_population))
            lp_reserve = min(int(active_set_evaluations),max(0,max_evaluations-requested_population-reserve)) if field_values is not None else 0
            active_budget = max_evaluations-reserve-lp_reserve
            round_info.update(warm_starts_inserted=warm_count, population_not_shrunk_for_budget=True,
                population_policy='explicit' if population_size is not None else 'popsize_times_dimension',
                initial_local_fraction=initial_local_fraction,initial_sigma=initial_sigma,
                initial_scale_mode=initial_scale_mode,initial_sigma_mm=initial_sigma_mm,
                warm_local_fraction=warm_local_fraction,warm_local_rows=warm_local_rows,
                local_center_audit=local_center_audit,de_field_budget=active_budget,local_field_budget_reserved=reserve,
                active_set_field_budget_reserved=lp_reserve,
                initial_center='continuation_elite' if initial_design is not None else 'original_topology',
                original_parameter_windows_retained=True)
            save_json(output/'initial_population.json', {'variables': variables, 'population': population,
                'warm_start_rows': warm_rows, 'random_sampling': 'local Gaussian continuous neighbours + feasible global sampler / inclusive integers',
                'initial_local_fraction': initial_local_fraction,'initial_sigma': initial_sigma,
                'initial_scale_mode':initial_scale_mode,'initial_sigma_mm':initial_sigma_mm,
                'physical_sampling_statistics':seed_sampling_statistics,'warm_local_fraction':warm_local_fraction,
                'warm_local_rows':warm_local_rows,'local_center_audit':local_center_audit})
            try:
                result = differential_evolution(objective, bounds=list(zip(low, high)), integrality=integer,
                    init=population, maxiter=maxiter, popsize=popsize, strategy='best1bin',
                    seed=int(rng.integers(0, 2**31-1)), tol=0., atol=0., polish=False,
                    updating='immediate', workers=1, callback=callback)
                round_info.update(state='converged' if result.success else 'iteration_limit', message=str(result.message))
            except _BudgetReached:
                round_info.update(state='budget_limited', message='Stopped before field-evaluation allocation was exceeded')
        summarize('initialization');summarize('de')
        # Retain the best recorded incumbent, including the manufacturing start.
        # Coordinate polling supplies local directions even when an explicitly
        # compact DE population is smaller than the continuous dimension.
        de_evaluations = counters['field_evaluations']
        de_proposals = counters['proposal_calls']
        local_info = {'enabled':bool(local_refine_evaluations and dimensions),'field_evaluations':0,
                      'proposal_calls':0,'accepted_moves':0,'sweeps':0,'initial_step':local_step}
        if local_refine_evaluations and dimensions and de_evaluations < max_evaluations:
            local_limit=min(max_evaluations,de_evaluations+int(local_refine_evaluations))
            pair_reserve=min(int(round((local_limit-de_evaluations)*pair_refine_fraction)),max(0,local_limit-de_evaluations-2))
            active_budget=local_limit-pair_reserve
            phase='local';probes=[]
            incumbent=min(records,key=lambda r:feasibility_score(r['metrics'],requirements))
            x=np.array([incumbent['parameters'][v['name']] for v in variables],float)
            score=feasibility_score(incumbent['metrics'],requirements)
            current_metrics=dict(incumbent['metrics'])
            local_info.update(starting_candidate_id=incumbent['candidate_id'],starting_score=score,
                pair_budget_reserved=pair_reserve,paired_trials=0,paired_accepted_moves=0)
            steps=(high-low)*local_step;steps[integer]=np.maximum(1,np.rint(steps[integer]))
            proposals_before=counters['proposal_calls']
            # Retain a paused coordinate queue across the paired stage. A
            # reserved-but-unused pair budget must never strand unvisited axes.
            poll={'order':None,'cursor':0,'sign':0,'axis_active':False,'improved':False,'completed':0}
            visited_axes=set()
            local_info.update(coordinate_sweeps_completed=0,coordinate_step_reductions=0,
                              recovery_anchor_resets=0,coordinate_recovery_field_evaluations=0)
            construction=getattr(build_design,'metadata',{})
            spans=construction.get('radial_free_spans_mm',[])
            local_info['step_audit']={'coordinate_type':'normalized_constructive_slack',
                'initial_normalized_step':float(local_step),'radial_free_spans_mm':spans,
                'leading_radial_step_upper_bounds_mm':[float(v*local_step) for v in spans],
                'scope':'geometric step bounds, not an equal-mm step promise; no field solves for audit'}

            def coordinate_poll():
                nonlocal x,score,current_metrics
                while True:
                    if poll['order'] is None:
                        if counters['field_evaluations']>=active_budget or poll['completed']>=8:return
                        poll.update(order=rng.permutation(np.flatnonzero(low<high)).tolist(),cursor=0,
                                    sign=0,axis_active=False,improved=False)
                        local_info['sweeps']+=1
                    if poll['cursor']>=len(poll['order']):
                        poll['completed']+=1;local_info['coordinate_sweeps_completed']=poll['completed']
                        if not poll['improved']:
                            steps[~integer]*=.5;local_info['coordinate_step_reductions']+=1
                        poll['order']=None
                        continue
                    index=poll['order'][poll['cursor']]
                    if poll['axis_active'] and poll['sign']>=2:
                        if poll['best_score']<score:
                            x,score,current_metrics=poll['best_x'],poll['best_score'],dict(poll['best_metrics'])
                            poll['improved']=True;local_info['accepted_moves']+=1
                        poll.update(axis_active=False,cursor=poll['cursor']+1,sign=0)
                        continue
                    if counters['field_evaluations']>=active_budget:return
                    if not poll['axis_active']:
                        poll.update(axis_active=True,sign=0,origin=x.copy(),origin_metrics=dict(current_metrics),
                                    best_x=x.copy(),best_score=score,best_metrics=dict(current_metrics))
                    sign=1. if poll['sign']==0 else -1.
                    origin=poll['origin'];trial=origin.copy()
                    trial[index]=np.clip(origin[index]+sign*steps[index],low[index],high[index])
                    if integer[index]:trial[index]=np.rint(trial[index])
                    if np.array_equal(trial,origin):poll['sign']+=1;continue
                    trial_score=objective(trial);row=vector_cache.get(tuple(trial))
                    visited_axes.add(int(index));poll['sign']+=1
                    if row is not None and not integer[index]:
                        probes.append({'origin':origin.tolist(),'index':int(index),'delta':float(trial[index]-origin[index]),
                            'delta_field_T':row['metrics']['mean_norm_T']-poll['origin_metrics']['mean_norm_T'],
                            'delta_ppm':row['metrics']['pp_norm_ppm']-poll['origin_metrics']['pp_norm_ppm']})
                    if trial_score<poll['best_score']:
                        poll.update(best_x=trial,best_score=trial_score,best_metrics=row['metrics'])

            def reanchor_poll():
                nonlocal x,score,current_metrics
                incumbent=min(records,key=lambda r:feasibility_score(r['metrics'],requirements))
                new_x=np.array([incumbent['parameters'][v['name']] for v in variables],float)
                if poll['axis_active'] and not np.array_equal(poll['origin'],new_x):
                    # Remaining axis order is retained. Old single-axis probes
                    # cannot be reused at a moved center, so restart this axis.
                    poll.update(axis_active=False,sign=0)
                    local_info['recovery_anchor_resets']+=1
                x,score,current_metrics=new_x,feasibility_score(incumbent['metrics'],requirements),dict(incumbent['metrics'])
                return incumbent

            try:coordinate_poll()
            except _BudgetReached:pass
            coordinate_end=counters['field_evaluations'];summarize('local')
            # Re-anchor on the actual best saved evaluation, including the last
            # trial if the field budget interrupted its second opposite probe.
            incumbent=reanchor_poll()
            local_info['coordinate_queue_at_pair']={'remaining_axes':[] if poll['order'] is None else poll['order'][poll['cursor']:],
                'next_sign':poll['sign'],'sweeps_completed':poll['completed']}
            if pair_reserve and probes and counters['field_evaluations']<local_limit:
                phase='paired';active_budget=local_limit
                local_info['paired_anchor_parameters']=pars(x)
                anchored=[p for p in probes if np.array_equal(np.asarray(p['origin'],float),x)]
                local_info['paired_stale_probes_discarded']=len(probes)-len(anchored)
                local_info['paired_anchor_reprobes']=0
                try:
                    # Old secants rank coordinates only. They are never used as
                    # delta-B at a different incumbent. If no complete pair is
                    # available at x, solve a few new single-axis probes here.
                    if not list(compensated_steps(x,anchored,low,high,integer,1)):
                        targets=sorted((p for p in probes if p['delta_field_T']<0 and p['delta_ppm']<0),
                            key=lambda p:p['delta_ppm']/max(-p['delta_field_T'],1e-30))
                        gains=sorted((p for p in probes if p['delta_field_T']>0),
                            key=lambda p:p['delta_ppm']/max(p['delta_field_T'],1e-30))
                        directions=[];seen=set()
                        for pool in (targets,gains):
                            added=0
                            for p in pool:
                                direction=(p['index'],1 if p['delta']>0 else -1)
                                if direction in seen or integer[p['index']]:continue
                                directions.append(p);seen.add(direction);added+=1
                                if added>=2:break
                        for old_probe in directions:
                            index=old_probe['index'];trial=x.copy()
                            trial[index]=np.clip(x[index]+old_probe['delta'],low[index],high[index])
                            if np.array_equal(trial,x):continue
                            objective(trial);row=vector_cache.get(tuple(trial))
                            local_info['paired_anchor_reprobes']+=1
                            if row is not None:
                                anchored.append({'origin':x.copy().tolist(),'index':index,'delta':float(trial[index]-x[index]),
                                    'delta_field_T':row['metrics']['mean_norm_T']-incumbent['metrics']['mean_norm_T'],
                                    'delta_ppm':row['metrics']['pp_norm_ppm']-incumbent['metrics']['pp_norm_ppm']})
                    for trial,pair_info in compensated_steps(x,anchored,low,high,integer,int(pair_max_trials)):
                        local_info['paired_trials']+=1
                        trial_score=objective(trial);row=vector_cache.get(tuple(trial))
                        # One secant correction is allowed for nonlinear field
                        # response. It is a new solved design under the same cap.
                        if row is not None and requirements and row['metrics']['mean_norm_T']<requirements.get('min_mean_norm_T',0):
                            j=pair_info['compensator_index'];slope=pair_info['measured_compensator_slope_T']
                            if abs(slope)>1e-20:
                                deficit=requirements['min_mean_norm_T']-row['metrics']['mean_norm_T']
                                corrected=trial.copy();corrected[j]+=(deficit*1.01+1e-15)/slope
                                if np.all(corrected>=low) and np.all(corrected<=high):
                                    corrected_score=objective(corrected)
                                    if corrected_score<trial_score:trial,trial_score=corrected,corrected_score
                        if trial_score<score:
                            score=trial_score;local_info['paired_accepted_moves']+=1
                        if counters['field_evaluations']>=local_limit:break
                except _BudgetReached:pass
                summarize('paired')
            paired_end=counters['field_evaluations']
            local_info['unused_pair_budget_reallocated']=max(0,local_limit-paired_end) if pair_reserve else 0
            if pair_reserve and counters['field_evaluations']<local_limit:
                reanchor_poll();phase='local';active_budget=local_limit
                try:coordinate_poll()
                except _BudgetReached:pass
                if counters['field_evaluations']>paired_end:summarize('local_recovery')
            score=min(feasibility_score(r['metrics'],requirements) for r in records)
            recovery_cost=counters['field_evaluations']-paired_end
            local_info.update(field_evaluations=counters['field_evaluations']-de_evaluations,
                coordinate_field_evaluations=coordinate_end-de_evaluations+recovery_cost,
                coordinate_initial_field_evaluations=coordinate_end-de_evaluations,
                coordinate_recovery_field_evaluations=recovery_cost,
                paired_field_evaluations=paired_end-coordinate_end,coordinate_axes_visited=sorted(visited_axes),
                coordinate_final_steps=steps.tolist(),
                proposal_calls=counters['proposal_calls']-proposals_before,ending_score=score)
            log({'event':'local_refinement',**local_info})
        round_info['local_refinement'] = local_info
        round_info['de_field_evaluations'] = de_evaluations
        round_info['field_evaluations'] = counters['field_evaluations']
        round_info['initialization_complete'] = de_proposals >= requested_population if dimensions else True
        if not round_info['initialization_complete'] and dimensions: round_info['state'] = 'initialization_budget_limited'
        log({'event': 'round_end', **round_info})
        lp_info={'enabled':bool(active_set_evaluations and field_values is not None),'state':'no_field_oracle' if field_values is None else 'not_requested','field_evaluations':0}
        if active_set_evaluations and field_values is not None and dimensions:
            from .active_set import active_set_polish
            lp_start=counters['field_evaluations'];lp_limit=min(max_evaluations,lp_start+int(active_set_evaluations))
            active_budget=lp_limit;phase='active_set';phase_calls.setdefault(phase,0)
            incumbent=min(records,key=lambda r:feasibility_score(r['metrics'],requirements))
            x=np.array([incumbent['parameters'][v['name']] for v in variables],float)
            def query_fields(vector):
                check_stop();objective(vector)
                row=vector_cache.get(tuple(vector))
                if row is None:return None
                physical=row['canonical_physical_sha256']
                if physical not in field_cache:
                    if counters['field_evaluations']>=active_budget:raise _BudgetReached()
                    check_stop();counters['field_evaluations']+=1
                    values=np.asarray(field_values(build_design(vector)),float)
                    if values.ndim!=2 or values.shape[1]!=3 or not np.all(np.isfinite(values)):raise ValueError('LP field oracle must return finite Nx3 T')
                    field_cache[physical]=values
                return {'fields':field_cache[physical],'metrics':row['metrics'],'score':feasibility_score(row['metrics'],requirements)}
            try:
                passes=[]
                # Recompute the Jacobian at each accepted physical design.
                # The caller's LP reserve bounds all passes, not each pass.
                pass_limit=3 if (requirements or {}).get('scoring',{}).get('mode')=='balanced' else 1
                for _ in range(pass_limit):
                    if lp_limit-counters['field_evaluations']<int(np.sum((high>low)&~integer))+2:break
                    x,current=active_set_polish(x,low,high,integer,query_fields,
                        lambda:lp_limit-counters['field_evaluations'],weights=field_weights,requirements=requirements,
                        trust_radius=active_set_trust_radius,fd_step=active_set_fd_step)
                    passes.append(current)
                    if not current.get('accepted_steps'):break
                lp_info=dict(passes[-1]) if passes else {'enabled':True,'state':'insufficient_budget_for_full_jacobian'}
                lp_info.update(linearization_passes=len(passes),passes=passes,
                    accepted_steps=sum(p.get('accepted_steps',0) for p in passes),
                    relinearization='new complete Jacobian after each independently accepted step')
            except _BudgetReached:lp_info={'enabled':True,'state':'budget_exhausted','all_physics_queries_charged_by_caller':True}
            lp_info['field_evaluations']=counters['field_evaluations']-lp_start
            summarize('active_set');log({'event':'active_set_polish',**lp_info})
        round_info['active_set_polish']=lp_info
        round_info['field_evaluations']=counters['field_evaluations']
    state = ('budget_exhausted' if counters['field_evaluations'] >= max_evaluations else
             'no_free_variables' if dimensions == 0 else round_info['state'])
    metadata = {'algorithm': 'fixed-topology mixed-initialization DE + coordinate refinement; strict epsilon ordering', 'state': state,
        'requirements': requirements or {}, 'variables': variables, 'active_dimension': dimensions,
        'seed': int(seed), 'max_evaluations': max_evaluations, **counters,
        'remaining_evaluations': max_evaluations-counters['field_evaluations'],
        'feasible_records': sum(r['feasible'] for r in records), 'evaluated_records': len(records),
        'baseline_matches_initial_vector': baseline_matches, 'rounds': [round_info], 'phase_proposal_calls':phase_calls,
        'original_topology_design':_design_dict(original_baseline_design),
        'initial_center_design':_design_dict(baseline_design),'original_parameter_windows_retained':True,
        'legacy_weights_ignored': list(weights) if weights is not None else None,
        'global_optimum_proven': False, 'independent_dense_validation_required': True,
        'stage_summaries':stage_summaries,'monitor_errors':monitor_errors,
        'iteration_logs': logs, 'elapsed_seconds': time.perf_counter()-started,
        'files': {'records': 'optimization_records.jsonl', 'rejections': 'optimization_rejections.jsonl',
                  'iterations': 'optimization_iterations.jsonl','events':'optimization_events.jsonl', 'result': 'optimization_result.json'}}
    result = {'state': state, 'records': records, 'baseline': baseline, 'pareto': pareto_front(records),
              'diagnostic_best': min(records, key=lambda r: feasibility_score(r['metrics'], requirements)), 'metadata': metadata}
    save_json(output/'optimization_result.json', result)
    return result


def run_optimization(*args,graceful_stop=False,**kwargs):
    """Run the numerical solver, optionally saving every completed record on stop.

    Legacy callers keep exception propagation. The workbench explicitly enables
    graceful stopping; interrupted or failed field evaluations never become
    invented candidate metrics.
    """
    runtime={}
    if not graceful_stop:return _run_optimization_impl(*args,**kwargs)
    try:
        return _run_optimization_impl(*args,**kwargs,_runtime_state=runtime)
    except BaseException as error:
        if not isinstance(error,(_GracefulStop,KeyboardInterrupt)) and type(error).__name__!='CancelledError':raise
        output=Path(args[5] if len(args)>5 else kwargs['output_dir'])
        output.mkdir(parents=True,exist_ok=True)
        records=runtime.get('records',[]);counters=runtime.get('counters',{})
        state='stopped_with_result' if records else 'stopped_no_result'
        maximum=runtime.get('max_evaluations',kwargs.get('max_evaluations',300))
        metadata={'algorithm':'fixed-topology DE / local / active-set LP; gracefully interrupted',
            'state':state,'max_evaluations':maximum,**counters,
            'field_evaluations':counters.get('field_evaluations',0),
            'remaining_evaluations':maximum-counters.get('field_evaluations',0),
            'active_dimension':sum(v['upper']>v['lower'] for v in runtime.get('variables',[])),
            'variables':runtime.get('variables',[]),'requirements':runtime.get('requirements',{}),
            'rounds':[runtime.get('round_info',{})],'stage_summaries':runtime.get('stage_summaries',[]),
            'iteration_logs':runtime.get('logs',[]),'initial_center_design':runtime.get('initial_center_design'),
            'original_topology_design':runtime.get('original_topology_design'),
            'all_completed_records_saved':True,'graceful_stop':True,'global_optimum_proven':False,
            'independent_dense_validation_required':True,'elapsed_seconds':time.perf_counter()-runtime.get('started',time.perf_counter())}
        result={'state':state,'records':records,'baseline':runtime.get('baseline') or (records[0] if records else None),
            'pareto':pareto_front(records),'diagnostic_best':min(records,key=lambda r:feasibility_score(r['metrics'],runtime.get('requirements'))) if records else None,
            'metadata':metadata}
        save_json(output/'optimization_result.json',result)
        return result


def run_topology_optimization(topologies, process, limits, search, evaluator, checker, output_dir,
                              settings, requirements, warm_starts=(),event_callback=None):
    """Discrete architectures are separate runs with their own continuous coordinates."""
    from .parameterization import get_search_space
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    raw_total=settings['max_evaluations']
    if isinstance(raw_total,bool) or int(raw_total)!=raw_total or raw_total<1:
        raise ValueError('max_evaluations must be a positive integer')
    if not topologies:raise ValueError('At least one topology is required')
    options = dict(settings.get('optimization_options',{}))
    primary_share = float(options.pop('primary_topology_fraction',.5))
    if not math.isfinite(primary_share) or not 0<primary_share<=1:
        raise ValueError('primary_topology_fraction must be in (0,1]')
    allowed = {'population_size','initial_local_fraction','initial_sigma','initial_scale_mode','initial_sigma_mm',
               'local_refine_evaluations','local_step','pair_refine_fraction','pair_max_trials','warm_local_fraction'}
    if set(options)-allowed:raise ValueError('Unknown optimization_options: '+str(sorted(set(options)-allowed)))
    total = int(raw_total); used = 0; cache = {}; runs = []; records = []
    first_baseline = None
    callback_state={'enabled':event_callback is not None,'errors':[]}
    for index, topology in enumerate(topologies):
        remaining = total-used
        if remaining <= 0: break
        allocation = (max(1,int(total*primary_share)) if index==0 and len(topologies)>1
                      else max(1,remaining//(len(topologies)-index)))
        allocation = min(remaining,allocation)
        variables, builder = get_search_space(topology, process, limits, search)
        folder = output/('topology_%02d' % (index+1))
        offset=used
        def forward_event(event):
            if not callback_state['enabled']:return
            value=dict(event,topology_index=index+1,global_evaluations=offset+event['field_evaluations'])
            try:event_callback(value)
            except Exception as error:
                callback_state['enabled']=False;callback_state['errors'].append(str(error))
                raise
        run = run_optimization(variables, builder, evaluator, checker, topology, folder,
            seed=int(settings['seed'])+index, max_evaluations=allocation,
            popsize=settings['popsize'], maxiter=settings['maxiter'], requirements=requirements,
            numeric_cache=cache, warm_starts=warm_starts, candidate_prefix='t%02d_' % (index+1),
            event_callback=forward_event if event_callback is not None else None,
            topology_index=index+1,global_evaluation_offset=offset, **options)
        for row in run['records']:
            row['topology_index']=index+1;row['global_evaluations']=offset+row['field_evaluations']
        if first_baseline is None: first_baseline = run['baseline']
        used += run['metadata']['field_evaluations']; records.extend(run['records'])
        runs.append({'topology': index+1, 'ring_count': len(topology.rings), 'magnet_count': topology.magnet_count,
                     'active_dimension': len(variables), 'relative_result': str(folder.relative_to(output)/'optimization_result.json'),
                     'metadata': run['metadata']})
    unique = {}
    for row in records: unique.setdefault(row['canonical_physical_sha256'], row)
    records = list(unique.values())
    metadata = {'algorithm': 'outer balanced topology proposals / inner continuous epsilon-DE',
        'max_evaluations': total, 'field_evaluations': used, 'remaining_evaluations': total-used,
        'runs': runs, 'topologies_proposed': len(topologies), 'primary_topology_fraction':primary_share, 'optimization_options':options, 'topologies_run': len(runs),
        'monitor_errors':callback_state['errors'],
        'field_budget_excludes_seed_selection_and_dense_validation': True, 'files': {'result': 'optimization_result.json'},
        'global_optimum_proven': False, 'performance_feasible_records': sum(r['feasible'] for r in records)}
    result = {'baseline': first_baseline, 'records': records, 'pareto': pareto_front(records), 'metadata': metadata,
        'diagnostic_best': min(records, key=lambda r: feasibility_score(r['metrics'], requirements)) if records else None}
    save_json(output/'optimization_result.json', result)
    with (output/'optimization_records.jsonl').open('w', encoding='utf-8') as stream:
        for row in records: stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False)+'\n')
    return result

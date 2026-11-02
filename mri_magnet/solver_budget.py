"""No-field, dimension-aware budgets shared by the UI and island scheduler.

The report reserves a nominal complete coordinate sweep and full LP Jacobians.
It cannot promise convergence, successful nonlinear steps, or bound arbitrary
field failures. Cached earlier rounds can refund evaluations; the precheck uses
the conservative full-spend schedule, before the promoted island IDs are known.
"""
from __future__ import annotations

import math
from .geometry import Design
from .parameterization import get_search_space


def round_sizes(count, finalists, factor):
    sizes = [count]
    while sizes[-1] > finalists:
        sizes.append(max(finalists, int(math.ceil(sizes[-1]/factor))))
    return sizes


def split_budget(total, count):
    quotient, remainder = divmod(total, count)
    return [quotient+(i < remainder) for i in range(count)]


def allocate_round_pool(remaining, count, future_counts, minimum, fraction, final_round):
    """Exactly the pool arithmetic used at runtime, including future reserves."""
    future_reserve = minimum*sum(future_counts)
    return (remaining if final_round else
            min(remaining-future_reserve,
                max(minimum*count, int(math.floor(remaining*fraction)))))


def effective_refinement_budgets(allocation, population, requested_local, requested_lp, final_round):
    """Local and LP share the island cap, after init plus one DE generation."""
    if not final_round:
        return 0, 0
    local = min(requested_local, max(0, allocation-2*population))
    lp = min(requested_lp, max(0, allocation-2*population-local))
    return local, lp


def _integer(value, label, minimum=0):
    try:
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < minimum:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError(label+' must be an integer >= '+str(minimum))
    return int(value)


def _coordinate_initial_budget(local, pair_fraction):
    reserved = min(int(round(local*pair_fraction)), max(0, local-2))
    return local-reserved, reserved


def required_local_budget(dimensions, pair_fraction):
    """Initial coordinate portion must fit both directions for every free axis.

    Use the same integer rounding as the runtime reserve. Paired work may refund
    unused budget later; that uncertain refund cannot justify an upfront pass.
    """
    if dimensions <= 0:
        return 0
    if pair_fraction >= 1:
        return None
    result = max(2, int(math.ceil(2*dimensions/(1-pair_fraction))))
    while _coordinate_initial_budget(result, pair_fraction)[0] < 2*dimensions:
        result += 1
    while result > 2 and _coordinate_initial_budget(result-1, pair_fraction)[0] >= 2*dimensions:
        result -= 1
    return result


def _schedule(total, sizes, minimum, fraction):
    remaining = total
    rows = []
    for index, count in enumerate(sizes):
        final = index == len(sizes)-1
        pool = allocate_round_pool(remaining, count, sizes[index+1:], minimum, fraction, final)
        allocations = split_budget(pool, count)
        rows.append({'round':index+1, 'island_count':count, 'final_round':final,
                     'pool':pool, 'allocations':allocations,
                     'minimum_allocation':min(allocations), 'maximum_allocation':max(allocations),
                     'refinements_enabled':final})
        remaining -= pool
    return rows


def _minimum_global(sizes, minimum, fraction, per_finalist):
    """Smallest total supporting the final reserve under full screen spending."""
    lower = minimum*sum(sizes)
    def enough(total):
        schedule = _schedule(total, sizes, minimum, fraction)
        return schedule[-1]['minimum_allocation'] >= per_finalist
    upper = max(lower, per_finalist*sizes[-1])
    while not enough(upper):
        upper *= 2
    while lower < upper:
        middle = (lower+upper)//2
        if enough(middle): upper = middle
        else: lower = middle+1
    return lower


def plan_island_budgets(topologies, process, limits, search, settings, requirements=None):
    """Plan every possible finalist before computing any field or opening output.

    ``settings`` is the same object accepted by ``run_islands``. ``strict``
    rejects incomplete nominal scans/Jacobian passes. ``allow_partial`` keeps
    the budget cap but makes the limitations explicit. Omitting a policy is a
    legacy direct-solver mode; the workbench explicitly defaults to strict.
    """
    topologies = [Design.from_dict(d) if isinstance(d, dict) else d for d in topologies]
    if not topologies:
        return {'valid':False, 'status':'pending', 'policy':'strict', 'violations':[],
                'warnings':['请先生成并选择拓扑起点，才能按实际维数检查预算'],
                'per_island':[], 'rounds':[], 'field_evaluations':0, 'suggested_solver':{}}
    total = _integer(settings['max_evaluations'], 'max_evaluations', 1)
    options = settings.get('optimization_options', {})
    islands = settings.get('islands', {})
    policy = islands.get('budget_policy', 'legacy')
    if policy not in ('strict', 'allow_partial', 'legacy'):
        raise ValueError('budget_policy must be strict or allow_partial')
    population = _integer(options.get('population_size',16), 'population_size',5)
    local = _integer(options.get('local_refine_evaluations',80), 'local_refine_evaluations')
    lp = _integer(options.get('active_set_evaluations',0), 'active_set_evaluations')
    pair_fraction = float(options.get('pair_refine_fraction',.4))
    if not math.isfinite(pair_fraction) or not 0 <= pair_fraction <= 1:
        raise ValueError('pair_refine_fraction must be in [0,1]')
    factor = _integer(islands.get('halving_factor',2), 'halving_factor',2)
    finalists = min(len(topologies),_integer(islands.get('finalists',2),'finalists',1))
    minimum = max(2*population,_integer(islands.get('min_round_evaluations',0),'min_round_evaluations'))
    fraction = float(islands.get('initial_budget_fraction',.4))
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('initial_budget_fraction must be in (0,1)')
    maxiter = _integer(settings.get('maxiter',100),'maxiter')
    sizes = round_sizes(len(topologies),finalists,factor)
    minimum_schedule = minimum*sum(sizes)
    rounds = _schedule(max(total,minimum_schedule),sizes,minimum,fraction)
    final = rounds[-1]
    allocated_min, allocated_max = final['minimum_allocation'],final['maximum_allocation']
    effective_local, effective_lp = effective_refinement_budgets(allocated_min,population,local,lp,True)
    pass_target = 3 if (requirements or {}).get('scoring',{}).get('mode') == 'balanced' else 1
    errors = []
    warnings = []
    if total < minimum_schedule:
        errors.append('总预算%d低于多岛初始化和一代演化的最低调度预算%d'%(total,minimum_schedule))
    rows = []
    local_requirements, lp_requirements = [], []
    for index, design in enumerate(topologies):
        # Without a screening round the order is already known. Later rounds
        # are score-ranked, so use the worst slot for every possible finalist.
        row_min,row_max = ((final['allocations'][index],final['allocations'][index])
                           if len(sizes)==1 else (allocated_min,allocated_max))
        effective_local,effective_lp=effective_refinement_budgets(row_min,population,local,lp,True)
        variables,_ = get_search_space(design,process,limits,search)
        free = [v for v in variables if not v.get('integer',False) and float(v['upper']) > float(v['lower'])]
        dimension = len(free)
        need_local = required_local_budget(dimension,pair_fraction) if local and dimension else 0
        first_lp = dimension+2 if dimension and lp else 0
        per_pass_with_backtrack = dimension+4 if dimension and lp else 0
        need_lp = pass_target*per_pass_with_backtrack if lp and dimension else 0
        local_requirements.append(need_local)
        lp_requirements.append(need_lp)
        coordinate, pair_reserved = _coordinate_initial_budget(effective_local,pair_fraction)
        affordable_lp = (min(pass_target,effective_lp//per_pass_with_backtrack)
                         if per_pass_with_backtrack else 0)
        nominal_lp = (min(pass_target,effective_lp//first_lp) if first_lp else 0)
        reasons = []
        if local and dimension:
            if need_local is None:
                reasons.append('成对预算占比为100%，无法预留完整双向坐标扫描')
            elif effective_local < need_local:
                reasons.append('坐标/成对预算%d次，初始坐标额度%d次，小于完整双向扫描%d次（局部预算至少%d次）'%(
                    effective_local,coordinate,2*dimension,need_local))
        if lp and dimension:
            if effective_lp < first_lp:
                reasons.append('LP有效预算%d次低于完整雅可比启动下限%d次，将跳过LP'%(effective_lp,first_lp))
            elif effective_lp < need_lp:
                reasons.append('LP有效预算%d次不足%d轮完整线性化及每轮2次回溯余量，需至少%d次'%(
                    effective_lp,pass_target,need_lp))
        clipped = []
        if effective_local < local: clipped.append('局部预算被岛配额从%d压缩到%d'%(local,effective_local))
        if effective_lp < lp: clipped.append('LP预算被岛配额从%d压缩到%d'%(lp,effective_lp))
        for message in reasons:
            errors.append('岛%d（D=%d）：%s'%(index+1,dimension,message))
        for message in clipped:
            warnings.append('岛%d：%s'%(index+1,message))
        if not local and dimension: warnings.append('岛%d：坐标/成对精修已明确关闭'%(index+1))
        if not lp and dimension: warnings.append('岛%d：灵敏度LP精修已明确关闭'%(index+1))
        de_budget = row_min-effective_local-effective_lp
        rows.append({'island':index+1, 'id':'island_%02d'%(index+1), 'dimensions':dimension, 'D':dimension,
            'all_variables':len(variables), 'name':getattr(design,'name','拓扑岛%d'%(index+1)),
            'conditional':'若该岛晋级末轮；晋级前不运行精修',
            'final_allocation_min':row_min,'final_allocation_max':row_max,
            'first_round_allocation':rounds[0]['allocations'][index],
            'requested_local_evaluations':local,'effective_local_evaluations':effective_local,
            'requested_lp_evaluations':lp,'effective_lp_evaluations':effective_lp,
            'coordinate_required_evaluations':2*dimension,
            'coordinate_initial_evaluations':coordinate,'paired_reserved_evaluations':pair_reserved,
            'coordinate_axis_coverage':min(dimension,coordinate//2),
            'coordinate_nominal_sweeps':coordinate//(2*dimension) if dimension and local else 0,
            'required_local_evaluations':need_local,
            'lp_first_pass_minimum':first_lp,'lp_required_evaluations':need_lp,
            'lp_target_passes':pass_target if lp and dimension else 0,
            'lp_affordable_passes':affordable_lp,'lp_nominal_passes_without_backtracking':nominal_lp,
            'lp_will_skip':bool(lp and dimension and effective_lp < first_lp),
            'de_budget_min':de_budget,
            'de_generations_nominal':min(maxiter,max(0,de_budget//population-1)) if dimension else 0,
            'truncated':bool(reasons or clipped),'reasons':reasons,'warnings':clipped,
            'local_state':'not_needed' if not dimension else 'disabled' if not local else 'partial' if need_local is None or effective_local < need_local else 'nominal_complete',
            'lp_state':'not_needed' if not dimension else 'disabled' if not lp else 'skipped' if effective_lp < first_lp else 'partial' if effective_lp < need_lp else 'nominal_complete'})
    feasible_local = [v for v in local_requirements if v is not None]
    suggested_local = max([local]+feasible_local) if local else 0
    suggested_lp = max([lp]+lp_requirements) if lp else 0
    required_per_finalist = 2*population+suggested_local+suggested_lp
    suggested_total = max(total,_minimum_global(sizes,minimum,fraction,required_per_finalist))
    suggestion = {'max_evaluations':suggested_total,'population_size':population,
                  'local_refine_evaluations':suggested_local,'active_set_evaluations':suggested_lp}
    if any(v is None for v in local_requirements):
        suggestion['pair_refine_fraction']=.4
        suggested_local=max(local,max(required_local_budget(row['dimensions'],.4) for row in rows))
        suggestion['local_refine_evaluations']=suggested_local
        suggestion['max_evaluations']=max(total,_minimum_global(sizes,minimum,fraction,2*population+suggested_local+suggested_lp))
    # A partial-refinement opt-in cannot waive the scheduler's hard minimum.
    valid = total>=minimum_schedule and (not errors or policy in ('allow_partial','legacy'))
    return {'valid':valid,'status':'conflict' if not valid else 'warning' if errors or warnings else 'passed',
        'policy':policy,'violations':errors,'warnings':warnings+([] if policy=='strict' else errors),
        'per_island':rows,'rounds':rounds,'round_sizes':sizes,
        'max_evaluations':total,'population_size':population,'minimum_scheduled_budget':minimum_schedule,
        'finalist_count':finalists,'max_dimensions':max(row['dimensions'] for row in rows),
        'suggested_solver':suggestion,'field_evaluations':0,
        'assumptions':{'schedule':'conservative full spending in all earlier rounds; any selected island may be promoted',
            'scheduled_total_for_preview':max(total,minimum_schedule),
            'promotion_order_unknown':True,'cache_refunds_can_increase_final_allocations':True,
            'allocation_max_is_rounding_range_not_upper_limit_after_cache_refunds':True,
            'lp_backtracking_reserve_per_pass':2,'coordinate_scope':'one nominal +/- trial per free coordinate before paired work',
            'successful_lp_steps_or_convergence_guaranteed':False,
            'arbitrary_field_failures_and_geometry_retries_may_exhaust_budget':True}}

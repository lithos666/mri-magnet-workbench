"""Hard performance gates, balanced ranking and physical Pareto selection."""
import math
from .design_contract import (check_performance, scoring_options, requirements_for,
                              record_metrics, valid_record_metrics, record_is_feasible,
                              eligible_records, preferred_records)


def resolve_requirements(options, baseline_metrics, seed_metrics):
    """Resolve absolute limits or relative limits on one supplied point set.

    Resolve training and dense-validation limits separately using their own
    reference metrics. Relative requirements then mean improvement over the
    *same* reference design, without silently adding an integration tolerance.
    Explicit mT/ppm limits remain identical on both point sets.
    """
    options = dict(options or {})
    references = {'baseline': baseline_metrics, 'manufacturing_seed': seed_metrics}
    reference_name = options.get('min_mean_norm_reference', 'manufacturing_seed')
    if reference_name not in references:
        raise ValueError('min_mean_norm_reference must be baseline or manufacturing_seed')
    reference = references[reference_name]
    explicit = options.get('min_mean_norm_mT')
    ratio = float(options.get('min_mean_norm_ratio', 1.0))
    minimum = float(explicit)*1e-3 if explicit is not None else ratio*float(reference['mean_norm_T'])
    maximum_reference = options.get('max_pp_norm_reference', 'manufacturing_seed')
    if maximum_reference is not None and maximum_reference not in references:
        raise ValueError('max_pp_norm_reference must be baseline, manufacturing_seed or None')
    maximum_ratio = float(options.get('max_pp_norm_ratio', 1.0))
    explicit_maximum = options.get('max_pp_norm_ppm')
    maximum_metrics = references[maximum_reference] if maximum_reference else None
    maximum = (float(explicit_maximum) if explicit_maximum is not None else
               maximum_ratio*float(maximum_metrics['pp_norm_ppm']) if maximum_metrics else None)
    tolerance = float(options.get('field_tolerance_mT', 0.0))*1e-3
    if not math.isfinite(minimum) or minimum < 0 or not math.isfinite(ratio) or ratio < 0:
        raise ValueError('Field requirement must be finite and non-negative')
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0:
        raise ValueError('max_pp_norm_ratio must be finite and positive')
    if maximum is not None and (not math.isfinite(maximum) or maximum < 0 or
                               (explicit_maximum is not None and maximum == 0)):
        raise ValueError('max_pp_norm_ppm must be None or positive; reference ppm may be zero')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('field_tolerance_mT must be finite and non-negative')
    return {'min_mean_norm_T': minimum, 'max_pp_norm_ppm': maximum, 'field_tolerance_T': tolerance,
            'reference': reference_name, 'reference_mean_norm_T': reference['mean_norm_T'], 'ratio': ratio,
            'explicit_minimum_mT': explicit, 'field_scale_T': max(float(reference['mean_norm_T']), 1e-12),
            'max_pp_norm_reference': maximum_reference,
            'reference_pp_norm_ppm': None if maximum_metrics is None else maximum_metrics['pp_norm_ppm'],
            'max_pp_norm_ratio': maximum_ratio, 'explicit_maximum_ppm': explicit_maximum,
            'ppm_scale': max(float(reference['pp_norm_ppm']), 1.),
            'reference_metric_pointsets': {name: metrics.get('pointset') for name,metrics in references.items()},
            'scoring': scoring_options(options.get('scoring')),
            'rule': 'strict epsilon feasibility; rank feasible set using declared scoring'}


def feasibility_score(metrics, requirements=None):
    """A lower score is better; feasible values are always strictly negative.

    Historical ``ppm`` mode is unchanged. Balanced mode compares dimensionless
    logarithmic changes: its field scale is a *fractional* gain above B_min,
    and its ppm scale is an explicit absolute ppm reference. The logarithms
    give diminishing returns to both objectives; a vanishing field gain does
    not justify an arbitrarily large loss of homogeneity.
    """
    check = check_performance(metrics, requirements)
    if not check['valid']: return check['normalized_violation']
    options = scoring_options((requirements or {}).get('scoring'))
    if options['mode'] == 'balanced':
        loss = balanced_components(metrics, requirements)['loss']
        # Stable bounded logistic: preserves feasibility-first ordering even
        # when a user's chosen scales are many orders of magnitude apart.
        return -1./(1.+math.exp(max(-700., min(700., loss))))
    scale = max(float((requirements or {}).get('ppm_scale', 1.)), 1.)
    return -1./(1.+float(metrics['pp_norm_ppm'])/scale)


def balanced_components(metrics, requirements=None):
    """Readable, dimensionless components of the balanced objective."""
    req = requirements or {}
    options = scoring_options(req.get('scoring'))
    field = float(metrics['mean_norm_T'])
    minimum = max(float(req.get('min_mean_norm_T', req.get('field_scale_T', field))), 1e-12)
    gain = max(0., field/minimum-1.)
    field_reward = math.log1p(gain/options['field_gain_scale'])
    ppm_penalty = math.log1p(float(metrics['pp_norm_ppm'])/options['ppm_scale'])
    weight = options['field_weight']
    return {'field_gain_fraction': gain, 'field_reward': field_reward,
            'ppm_penalty': ppm_penalty, 'loss': (1.-weight)*ppm_penalty-weight*field_reward,
            'settings': options}


# Historical private imports remain usable without a second selection policy.
_record_metrics = record_metrics
_record_is_feasible = record_is_feasible
_valid_record_metrics = valid_record_metrics


def pareto_front(records, requirements=None):
    """Non-dominated, hard-gate feasible B↑ / ppm↓ records.

    No convex-front assumption is made. Returned records are the originals,
    so callers retain exact candidate identities and dense-evaluation traces.
    """
    feasible = eligible_records(records, requirements)
    ordered = sorted(feasible,key=lambda row: (-float(_record_metrics(row)['mean_norm_T']),
                                               float(_record_metrics(row)['pp_norm_ppm'])))
    result, best_ppm, best_field = [], float('inf'), None
    for row in ordered:
        m = _record_metrics(row)
        field, ppm = float(m['mean_norm_T']), float(m['pp_norm_ppm'])
        # With field descending and ppm ascending, only a new lowest ppm can
        # improve the front. Equal metric ties remain non-dominated: neither
        # record is strictly better in either objective. O(n log n).
        if ppm < best_ppm:
            result.append(row);best_ppm,best_field=ppm,field
        elif ppm == best_ppm and field == best_field:
            result.append(row)
    return sorted(result, key=lambda row: (float(_record_metrics(row)['mean_norm_T']),
                                          float(_record_metrics(row)['pp_norm_ppm'])))


def pareto_roles(records, requirements=None):
    """Three honest engineering roles; empty if no feasible record exists."""
    front = pareto_front(records, requirements)
    if not front:
        return {}
    return {
        'balanced': min(front, key=lambda row: feasibility_score(_record_metrics(row), requirements)),
        'high_field': max(front, key=lambda row: (float(_record_metrics(row)['mean_norm_T']),
                                                -float(_record_metrics(row)['pp_norm_ppm']))),
        'lowest_ppm': min(front, key=lambda row: (float(_record_metrics(row)['pp_norm_ppm']),
                                                -float(_record_metrics(row)['mean_norm_T']))),
    }


def select_validation_candidates(records, requirements=None, limit=5):
    """Preserve a recommendation and both Pareto extremes before filling Top-K.

    Candidates without a passing hard gate remain diagnostic fallbacks and
    are not mislabeled as a Pareto recommendation. No field solves occur.
    """
    if limit < 1:
        return []
    rows = [row for row in preferred_records(records) if valid_record_metrics(row)]
    roles = pareto_roles(rows, requirements)
    selected = []
    for role in ('balanced', 'lowest_ppm', 'high_field'):
        row = roles.get(role)
        if row is not None and not any(row is item for item in selected):
            selected.append(row)
    front = pareto_front(rows, requirements)
    ordered = sorted(front, key=lambda row: feasibility_score(_record_metrics(row), requirements))
    ordered += sorted(rows, key=lambda row: (not record_is_feasible(row, requirements),
                                            feasibility_score(_record_metrics(row), requirements)))
    for row in ordered:
        if not any(row is item for item in selected):
            selected.append(row)
    return selected[:int(limit)]

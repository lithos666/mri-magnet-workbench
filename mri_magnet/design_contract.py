"""Lightweight, shared field-metric, feasibility and workbench requirements contract."""
import math
from collections.abc import Mapping


def check_performance(metrics, requirements=None):
    req = requirements or {}
    field, ppm = float(metrics['mean_norm_T']), float(metrics['pp_norm_ppm'])
    if not math.isfinite(field) or field <= 0 or not math.isfinite(ppm) or ppm < 0:
        return {'valid': False, 'normalized_violation': float('inf'), 'violations': ['invalid field metrics']}
    minimum = float(req.get('min_mean_norm_T', 0.))
    tolerance = float(req.get('field_tolerance_T', 0.))
    maximum = req.get('max_pp_norm_ppm')
    deficit = max(0., minimum-tolerance-field)
    excess = max(0., ppm-maximum) if maximum is not None else 0.
    violation = deficit/max(float(req.get('field_scale_T', minimum or field)), 1e-12)
    if maximum is not None: violation += excess/max(float(maximum), 1.)
    notes = []
    if deficit > 0: notes.append('mean field below required minimum')
    if excess > 0: notes.append('peak-to-peak ppm exceeds required maximum')
    return {'valid': not notes, 'normalized_violation': violation, 'violations': notes,
            'min_mean_norm_T': minimum, 'actual_mean_norm_T': field, 'max_pp_norm_ppm': maximum,
            'actual_pp_norm_ppm': ppm, 'field_tolerance_T': tolerance}


def scoring_options(options=None):
    """Validate the scoring contract shared by solver, archive and UI."""
    raw = dict(options or {})
    result = {'mode': raw.get('mode', 'ppm'),
              'field_weight': float(raw.get('field_weight', .5)),
              'field_gain_scale': float(raw.get('field_gain_scale', .10)),
              'ppm_scale': float(raw.get('ppm_scale', 1000.))}
    if result['mode'] not in ('ppm', 'balanced'):
        raise ValueError('scoring.mode must be ppm or balanced')
    weight = result['field_weight']
    if not math.isfinite(weight) or not 0. <= weight <= 1.:
        raise ValueError('scoring.field_weight must be between 0 and 1')
    for key in ('field_gain_scale', 'ppm_scale'):
        if not math.isfinite(result[key]) or result[key] <= 0.:
            raise ValueError('scoring.'+key+' must be finite and positive')
    return result


def requirements_for(config):
    """One absolute workbench contract for search and dense-result ranking."""
    target = config['target']
    maximum = target.get('max_ppm')
    scoring = scoring_options(config.get('scoring'))
    minimum = float(target['min_field_mT'])*1e-3
    return {'min_mean_norm_T': minimum,
            'max_pp_norm_ppm': None if maximum is None else float(maximum),
            'field_tolerance_T': 0., 'field_scale_T': max(minimum, 1e-12),
            'ppm_scale': max(float(maximum if maximum is not None else scoring['ppm_scale']), 1.),
            'scoring': scoring,
            'rule': ('absolute hard limits, then balanced field/ppm scoring'
                     if scoring['mode'] == 'balanced' else
                     'absolute user limits; strict feasibility then minimize ppm')}


def record_metrics(row):
    """Use independently validated metrics first; training remains a fallback."""
    return row.get('validation_metrics') or row.get('dense_metrics') or row.get('metrics') or {}


def valid_record_metrics(row):
    if not isinstance(row, Mapping):
        return False
    try:
        return check_performance(record_metrics(row))['valid']
    except (TypeError, ValueError, KeyError, OverflowError):
        return False


def record_is_feasible(row, requirements=None):
    """Reject explicit failed reviews and invalid fields at every entry point.

    Search records need not have engineering review fields yet. When a caller
    requires manufacturing review, a positive audit is mandatory. Explicit
    final flags remain authoritative; training_performance_check is separate.
    """
    if not valid_record_metrics(row):
        return False
    if any(key in row and not row[key] for key in ('feasible', 'geometry_feasible')):
        return False
    for key in ('geometry_check', 'manufacturing_check', 'manufacturing_audit',
                'audit', 'performance_check'):
        check = row.get(key) or {}
        if not isinstance(check, Mapping):
            return False
        if not check.get('valid', True):
            return False
    if row.get('_manufacturing_review_required'):
        audits = (row.get('manufacturing_check'), row.get('manufacturing_audit'), row.get('audit'))
        if not any(check and check.get('valid') is True for check in audits):
            return False
    return check_performance(record_metrics(row), requirements)['valid']


def _metric_scope(row):
    return 2 if row.get('validation_metrics') else 1 if row.get('dense_metrics') else 0


def preferred_records(records):
    """One physical design uses its most independent available review.

    Resolve duplicate aliases before checking validity. A failed dense review
    must not be replaced by an earlier, attractive training-only prediction.
    Equal-scope aliases retain their original order and exact record identity.
    """
    result, positions = [], {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        physical = row.get('canonical_physical_sha256')
        if not physical:
            result.append(row)
        elif physical not in positions:
            positions[physical] = len(result)
            result.append(row)
        elif _metric_scope(row) > _metric_scope(result[positions[physical]]):
            result[positions[physical]] = row
    return result


def eligible_records(records, requirements=None):
    """The same reviewed, finite input set for new or already-loaded selectors."""
    return [row for row in preferred_records(records) if record_is_feasible(row, requirements)]

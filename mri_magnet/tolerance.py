"""Bounded first-order tolerance analysis, with independent block Br scatter.

Source amplitudes are exactly linear for fixed uniform magnetization. Printed
sector translations use central spatial differences and are only first order.
This module never labels a sampled surrogate as dense/FEM-certified yield.
"""
from __future__ import annotations
import math
import time
import numpy as np
import magpylib as magpy


def _sources(design, polarization_model):
    if polarization_model not in ('remanence', 'isolated_cube_self_demag'):
        raise ValueError('Tolerance surrogate requires a fixed-source polarization model')
    arrays = design.magnet_arrays()
    angles = np.deg2rad(arrays['radial_angles_deg'])
    cs, sn = np.cos(angles), np.sin(angles)
    north = arrays['north_vectors']
    br = design.remanence_T
    if polarization_model == 'isolated_cube_self_demag':
        br /= 1+(design.reference_mur-1)/3
    local = br*np.c_[cs*north[:, 0]+sn*north[:, 1],
                     -sn*north[:, 0]+cs*north[:, 1], north[:, 2]]
    return arrays['positions_mm'], cs, sn, local


def _contributions(points_mm, positions_mm, cs, sn, local_j, cube_mm, chunk=32):
    points = np.asarray(points_mm, float)
    result = np.empty((len(points), 3, len(positions_mm)))
    for start in range(0, len(points), chunk):
        delta = (points[start:start+chunk, None, :]-positions_mm[None, :, :])*1e-3
        local = np.empty_like(delta)
        local[:, :, 0] = delta[:, :, 0]*cs+delta[:, :, 1]*sn
        local[:, :, 1] = -delta[:, :, 0]*sn+delta[:, :, 1]*cs
        local[:, :, 2] = delta[:, :, 2]
        values = magpy.getB('Cuboid', observers=local.reshape(-1, 3),
            dimension=(cube_mm*1e-3,)*3,
            polarization=np.tile(local_j, (len(delta), 1))).reshape(len(delta), -1, 3)
        result[start:start+len(delta), 0, :] = values[:, :, 0]*cs-values[:, :, 1]*sn
        result[start:start+len(delta), 1, :] = values[:, :, 0]*sn+values[:, :, 1]*cs
        result[start:start+len(delta), 2, :] = values[:, :, 2]
    return result


def _positive_integer(value, name, maximum):
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or not 1 <= value <= maximum:
        raise ValueError(name+' must be an integer in [1, %d]' % maximum)
    return int(value)


def _wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def _memory_estimate(point_count, magnet_count, trials, dimension):
    """Conservative explicit-array estimate, including chunk and MC overlap.

    Four response-array equivalents reserve the contribution/derivative and
    sensitivity workspaces; four draw equivalents include matrix-product,
    addition and magnitude arrays. The cuboid evaluator also uses temporary
    vector/corner arrays inside each bounded 32-observation source chunk.
    This is an allocation budget estimate, not total interpreter RSS.
    """
    chunk_points = min(32, point_count)
    doubles = (4*point_count*3*dimension + 4*trials*point_count*3
               + 2*trials*dimension + 24*chunk_points*3*magnet_count
               + 32*magnet_count)
    return int(8*doubles)


def preflight_tolerance(design, point_count, config, options=None):
    """Conservative memory bound without expanding sources or computing fields."""
    options=dict(options or {})
    trials=_positive_integer(options.get('trials',200),'trials',2000)
    sectors=_positive_integer(config['manufacturing']['sectors_per_layer'],'sectors',64)
    groups=min(design.magnet_count,len({r.layer_index for r in design.rings})*sectors)
    estimated=_memory_estimate(int(point_count),design.magnet_count,trials,design.magnet_count+3*groups)
    limit=float(options.get('max_memory_mb',192))
    if not math.isfinite(limit) or limit<=0:raise ValueError('max_memory_mb must be finite and positive')
    return {'valid':estimated<=limit*1e6,'estimated_peak_memory_mb':estimated/1e6,'limit_mb':limit,
            'point_count':int(point_count),'trials':trials,'maximum_translation_groups':groups}


def analyze_tolerance(design, sample, config, options=None):
    """200-trial default: block Br errors + shared layer/sector translations.

    Bounds are *uniform half widths*, not Gaussian standard deviations. The
    same physical source errors are used at every observation point. Full
    global samples must be supplied because tolerances break exact symmetry.
    """
    options = dict(options or {})
    trials = _positive_integer(options.get('trials', 200), 'trials', 2000)
    sectors = _positive_integer(config['manufacturing']['sectors_per_layer'], 'sectors', 64)
    half_br = float(options.get('remanence_half_range_fraction', .015))
    half_position = float(options.get('sector_translation_half_range_mm', .15))
    fd = float(options.get('translation_fd_step_mm', .05))
    if not all(math.isfinite(v) for v in (half_br, half_position, fd)) or not 0 <= half_br < 1 or half_position < 0 or fd <= 0:
        raise ValueError('Invalid Br/translation tolerance or finite-difference step')
    points = np.asarray(sample['points_mm'], float)
    weights = np.asarray(sample['weights'], float)
    if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),) or not np.isfinite(points).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid global tolerance observation sample')
    profile = sample.get('symmetry_profile') or {}
    if (sample.get('rotation_order') or sample.get('workbench_symmetry', {}).get('used')
            or profile.get('requires_verified_design')
            or profile.get('domain_area_fraction', 1.) < 1.
            or profile.get('domain_volume_fraction', 1.) < 1.
            or 'fundamental' in sample.get('name', '')):
        raise ValueError('Tolerance breaks exact symmetry: use global samples')
    weights = weights/weights.sum()
    n = design.magnet_count
    limit = float(options.get('max_memory_mb', 192))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('max_memory_mb must be finite and positive')
    # Check an upper bound before expanding any magnet arrays. Each layer
    # has at most ``sectors`` occupied translation groups.
    group_upper = min(n, len({r.layer_index for r in design.rings})*sectors)
    early_estimate = _memory_estimate(len(points), n, trials, n+3*group_upper)
    if early_estimate > limit*1e6:
        raise ValueError('Tolerance response requires approximately %.1f MB; limit %.1f MB' % (early_estimate/1e6, limit))
    ring_indices = np.concatenate([np.full(r.count, i, int) for i, r in enumerate(design.rings)])
    positions, cs, sn, local_j = _sources(design, config['polarization_model'])
    layer_indices = np.concatenate([np.full(r.count, r.layer_index, int) for r in design.rings])
    azimuth = np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2*np.pi)
    sector_coordinate = azimuth*sectors/(2*np.pi)
    # Equivalent phases such as 0 and 360 degrees can land on opposite sides
    # of a seam because sin(2*pi) is a tiny negative float. Assign nominal
    # seam centres consistently before grouping their shared translations.
    nearest_seam = np.rint(sector_coordinate)
    sector_coordinate = np.where(np.abs(sector_coordinate-nearest_seam) < 1e-10,
                                 nearest_seam, sector_coordinate)
    sector_indices = np.floor(sector_coordinate).astype(int) % sectors
    group_keys = sorted(set(zip(layer_indices.tolist(), sector_indices.tolist())))
    dimension = n+3*len(group_keys)
    estimated_bytes = _memory_estimate(len(points), n, trials, dimension)
    if estimated_bytes > limit*1e6:
        raise ValueError('Tolerance response requires approximately %.1f MB; limit %.1f MB' % (estimated_bytes/1e6, limit))
    start = time.perf_counter()
    response = np.zeros((len(points), 3, dimension))
    response[:, :, :n] = _contributions(points, positions, cs, sn, local_j, design.cube_mm)
    baseline = response[:, :, :n].sum(axis=2)
    if half_position:
        for group_index, key in enumerate(group_keys):
            indices = np.flatnonzero((layer_indices == key[0]) & (sector_indices == key[1]))
            for axis in range(3):
                shift = np.zeros(3); shift[axis] = fd
                plus = _contributions(points, positions[indices]+shift, cs[indices], sn[indices], local_j[indices], design.cube_mm).sum(axis=2)
                minus = _contributions(points, positions[indices]-shift, cs[indices], sn[indices], local_j[indices], design.cube_mm).sum(axis=2)
                response[:, :, n+3*group_index+axis] = (plus-minus)/(2*fd)
    precompute_seconds = time.perf_counter()-start
    rng = np.random.default_rng(int(options.get('seed', config.get('seed', 0)+709)))
    perturbations = rng.uniform(-1., 1., (trials, dimension))
    perturbations[:, :n] *= half_br
    perturbations[:, n:] *= half_position
    started_mc = time.perf_counter()
    draws = (response.reshape(len(points)*3, dimension) @ perturbations.T).T.reshape(trials, len(points), 3)+baseline
    magnitudes = np.linalg.norm(draws, axis=2)
    means = magnitudes @ weights
    if np.any(means <= 1e-12) or not np.isfinite(means).all():
        raise ValueError('Perturbed mean field too small/non-finite for relative ppm')
    ppm = np.ptp(magnitudes, axis=1)/means*1e6
    minimum = float(config['target']['min_field_mT'])*1e-3
    maximum = config['target'].get('max_ppm')
    accepted = means >= minimum
    if maximum is not None:
        accepted &= ppm <= float(maximum)
    sampling_seconds = time.perf_counter()-started_mc
    nominal_norm = np.linalg.norm(baseline, axis=1)
    nominal_mean = float(weights@nominal_norm)
    # First-order variance contribution of independent block Br errors to
    # centered |B|; no coherent whole-ring Br fluctuation is substituted.
    direction = baseline/np.maximum(nominal_norm[:, None], 1e-15)
    norm_response = np.einsum('pc,pcn->pn', direction, response[:, :, :n])
    centered_response = norm_response-weights@norm_response
    sensitivity = []
    for index, ring in enumerate(design.rings):
        variance = float(weights @ np.sum(centered_response[:, ring_indices == index]**2, axis=1))*half_br**2/3
        sensitivity.append({'ring_id': ring.id, 'sensitivity_ppm': math.sqrt(max(0., variance))/nominal_mean*1e6,
                            'scope': 'independent_Br_only'})
    sensitivity.sort(key=lambda row: -row['sensitivity_ppm'])
    # Check two held-out realizations against exact displaced cuboids at a
    # bounded subset of observations. This measures local linearization error;
    # it is not a dense extrema certificate.
    holdout_count = min(2, trials)
    check_indices = np.unique(np.linspace(0, len(points)-1, min(48, len(points))).astype(int))
    checks = []
    for trial in range(holdout_count):
        displaced = positions.copy()
        for group_index, key in enumerate(group_keys):
            mask = (layer_indices == key[0]) & (sector_indices == key[1])
            displaced[mask] += perturbations[trial, n+3*group_index:n+3*group_index+3]
        exact = _contributions(points[check_indices], displaced, cs, sn,
            local_j*(1+perturbations[trial, :n, None]), design.cube_mm).sum(axis=2)
        error = np.linalg.norm(exact-draws[trial, check_indices], axis=1)
        checks.append({'trial': trial, 'max_vector_error_ppm_of_nominal_mean': float(error.max()/nominal_mean*1e6)})
    return {'available': True, 'method': 'fixed-source Br superposition + first-order rigid sector translation surrogate',
        'samples': trials, 'trials': trials, 'pointset': sample.get('name'), 'point_count': len(points),
        'polarization_model': config['polarization_model'],
        'summary': {'p50_ppm': float(np.percentile(ppm, 50)), 'p90_ppm': float(np.percentile(ppm, 90)),
            'yield_fraction': float(accepted.mean()), 'yield_confidence95': _wilson(int(accepted.sum()), trials),
            'yield_scope': 'field_and_sampled_ppm' if maximum is not None else 'field_only_no_ppm_limit',
            'failed_samples': int((~accepted).sum())},
        'nominal': {'mean_field_mT': nominal_mean*1e3, 'pp_norm_ppm': float(np.ptp(nominal_norm)/nominal_mean*1e6)},
        'ppm_samples': ppm.tolist(), 'mean_field_mT_samples': (means*1e3).tolist(),
        'sensitivity_rings': sensitivity,
        'assumptions': {'distribution': 'independent bounded uniform; supplied values are half-ranges, not sigma',
            'remanence_half_range_fraction': half_br, 'sector_translation_half_range_mm': half_position,
            'translation_correlation': 'all magnet centers in the same physical layer/quadrant share one XYZ translation',
            'sector_assignment': 'magnet center azimuth; split-pocket contact is not modeled',
            'angular_size_mur_errors_included': False, 'coupled_response_uncertainty_included': False,
            'manufacturing_yield_included': False, 'dense_extrema_verified': False,
            'symmetry_reduction_used': False, 'linearization': 'translations first order; Br exact only at fixed positions'},
        'linearization_checks': checks,
        'timings': {'precompute_seconds': precompute_seconds, 'sampling_seconds': sampling_seconds,
                    'total_seconds': time.perf_counter()-start},
        'estimated_peak_memory_mb': estimated_bytes/1e6}

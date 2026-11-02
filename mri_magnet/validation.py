"""Nonsingular local extrema checks in a source-free homogeneous-air DSV.

Maxima of |B| and transverse magnitude, and both Bz extrema, are sought on
the sphere. The |B| minimum is searched throughout the ball. Finite local
multistart searches do not prove global extrema.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize


def _tangent_basis(normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _spaced_indices(points, values, count, maximize=False):
    chosen = []
    for index in np.argsort(-values if maximize else values):
        if all(np.linalg.norm(points[index]-points[j]) > 0.3 for j in chosen):
            chosen.append(int(index))
        if len(chosen) >= count:
            break
    return chosen


def refine_sphere_extrema(design, engine, sample, metrics, B_T, starts=3):
    """Refine each extrema objective, preserving volume average and RMS.

    starts gives separated starts per objective. Charts are recentered at
    their boundary up to twice. Objective/jacobian share finite-difference
    batches; field calls and number of field points are recorded separately.
    """
    started = time.perf_counter()
    if int(starts) != starts or starts < 1:
        raise ValueError('starts must be a positive integer')
    starts = int(starts)
    radius = float(sample['diameter_mm']) / 2
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError('Invalid DSV radius')
    points = np.asarray(sample['points_mm'], dtype=float)
    B = np.asarray(B_T, dtype=float)
    if points.shape != B.shape or points.shape[1:] != (3,) or not np.all(np.isfinite(B)):
        raise ValueError('Expected matching finite Nx3 samples and fields')
    unit_points = points / radius
    norms = np.linalg.norm(B, axis=1)
    zvalues = B[:, 2]
    transverse = np.linalg.norm(B[:, :2], axis=1)
    surface_mask = np.isclose(np.linalg.norm(unit_points, axis=1), 1, rtol=0, atol=1e-7)
    if not np.any(surface_mask):
        raise ValueError('Extrema refinement requires sphere-boundary samples')
    surface_points = unit_points[surface_mask]
    surface_B = B[surface_mask]
    counters = {'engine_calls': 0, 'field_points': 0, 'objective_batches': 0}
    statuses = []

    def field(positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        counters['engine_calls'] += 1
        counters['field_points'] += len(positions)
        return np.asarray(engine.field(design, positions), dtype=float)

    def objective_values(fields, kind):
        scaled = fields * 1e3
        if kind == 'norm2':
            return np.einsum('ij,ij->i', scaled, scaled)
        if kind == 'z':
            return scaled[:, 2]
        return np.einsum('ij,ij->i', scaled[:, :2], scaled[:, :2])

    def batched_objective(transform, kind, sign):
        previous = {}
        def evaluate(x):
            x = np.asarray(x, dtype=float)
            key = tuple(x)
            if previous.get('key') == key:
                return previous['result']
            step = 1e-5
            stencil = np.vstack((x, x + np.eye(len(x))*step))
            values = sign * objective_values(field(transform(stencil)), kind)
            result = float(values[0]), (values[1:]-values[0])/step
            previous.update(key=key, result=result)
            counters['objective_batches'] += 1
            return result
        return evaluate

    def surface_extreme(kind, maximize):
        initial_values = objective_values(surface_B, kind)
        all_values = objective_values(B, kind)
        index = int(np.argmax(all_values) if maximize else np.argmin(all_values))
        best_value, best_point = float(all_values[index]), points[index].copy()
        sign = -1.0 if maximize else 1.0
        for chosen in _spaced_indices(surface_points, initial_values, starts, maximize):
            normal = surface_points[chosen].copy()
            for recenter in range(3):
                first, second = _tangent_basis(normal)
                def transform(uv, n=normal.copy(), e1=first.copy(), e2=second.copy()):
                    directions = n + uv[:, 0, None]*e1 + uv[:, 1, None]*e2
                    return radius*directions/np.linalg.norm(directions, axis=1)[:, None]
                result = minimize(batched_objective(transform, kind, sign), np.zeros(2),
                                  method='L-BFGS-B', jac=True, bounds=[(-0.6, 0.6)]*2,
                                  options={'maxiter': 35, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 15})
                if not np.all(np.isfinite(result.x)):
                    break
                candidate = transform(np.asarray(result.x)[None, :])[0]
                value = float(objective_values(field(candidate[None, :]), kind)[0])
                if sign*value < sign*best_value:
                    best_value, best_point = value, candidate
                statuses.append({'objective': ('max_' if maximize else 'min_')+kind,
                                 'success': bool(result.success), 'status': int(result.status),
                                 'iterations': int(result.nit), 'chart_recenter': recenter})
                if np.max(np.abs(result.x)) < 0.6-1e-5:
                    break
                normal = candidate/radius
        return best_value, best_point

    max_norm_squared, max_norm_point = surface_extreme('norm2', True)
    min_z_mT, min_z_point = surface_extreme('z', False)
    max_z_mT, max_z_point = surface_extreme('z', True)
    max_transverse_squared, max_transverse_point = surface_extreme('transverse2', True)
    min_index = int(np.argmin(norms))
    min_squared = float(norms[min_index]**2*1e6)
    min_point = points[min_index].copy()
    chosen_points = [unit_points[i] for i in _spaced_indices(unit_points, norms, starts)]
    if not any(np.linalg.norm(p) < 1e-12 for p in chosen_points):
        chosen_points.append(np.zeros(3))
    for initial in chosen_points:
        result = minimize(batched_objective(lambda x: radius*x, 'norm2', 1.0), initial,
                          method='SLSQP', jac=True, bounds=[(-1., 1.)]*3,
                          constraints=[{'type': 'ineq', 'fun': lambda x: 1.0-float(x@x),
                                        'jac': lambda x: -2*np.asarray(x)}],
                          options={'maxiter': 60, 'ftol': 1e-12})
        statuses.append({'objective': 'min_norm2_ball', 'success': bool(result.success),
                         'status': int(result.status), 'iterations': int(result.nit)})
        if not np.all(np.isfinite(result.x)) or np.linalg.norm(result.x) > 1+1e-7:
            continue
        candidate_unit = np.asarray(result.x)
        if np.linalg.norm(candidate_unit) > 1:
            candidate_unit = candidate_unit/np.linalg.norm(candidate_unit)
        candidate = radius*candidate_unit
        value = float(objective_values(field(candidate[None, :]), 'norm2')[0])
        if value < min_squared:
            min_squared, min_point = value, candidate

    minimum = float(np.sqrt(max(0., min_squared))/1e3)
    maximum = float(np.sqrt(max(0., max_norm_squared))/1e3)
    min_z, max_z = min_z_mT/1e3, max_z_mT/1e3
    maximum_transverse = float(np.sqrt(max(0., max_transverse_squared))/1e3)
    refined = dict(metrics)
    refined.update({
        'sampled_pp_norm_ppm': metrics['pp_norm_ppm'],
        'sampled_min_norm_T': float(norms.min()), 'sampled_max_norm_T': float(norms.max()),
        'sampled_pp_Bz_ppm': metrics.get('pp_Bz_ppm'),
        'sampled_min_Bz_T': float(zvalues.min()), 'sampled_max_Bz_T': float(zvalues.max()),
        'sampled_max_transverse_T': float(transverse.max()),
        'min_norm_T': minimum, 'max_norm_T': maximum, 'delta_norm_T': maximum-minimum,
        'pp_norm_ppm': (maximum-minimum)/metrics['mean_norm_T']*1e6,
        'min_Bz_T': min_z, 'max_Bz_T': max_z,
        'pp_Bz_ppm': ((max_z-min_z)/abs(metrics['mean_Bz_T'])*1e6
                      if abs(metrics.get('mean_Bz_T', 0)) > 1e-12 else None),
        'max_transverse_T': maximum_transverse,
        'metric_sources': {'mean_norm_T': 'sampled_volume_quadrature',
                           'rms_norm_ppm': 'sampled_volume_quadrature',
                           'pp_norm_ppm': 'dense_samples_plus_surface_max_and_ball_min',
                           'pp_Bz_ppm': 'dense_samples_plus_separate_surface_extrema',
                           'max_transverse_T': 'dense_samples_plus_separate_surface_max'},
        'extrema_refinement': {
            'method': 'unit-normal tangent charts; Cartesian ball SLSQP; batched finite differences',
            'assumption': 'source-free homogeneous air inside DSV', 'global_optimum_proven': False,
            'field_evaluations': counters['engine_calls'], **counters,
            'elapsed_seconds': time.perf_counter()-started,
            'min_norm_T': minimum, 'max_norm_T': maximum,
            'min_point_mm': min_point.tolist(), 'max_point_mm': max_norm_point.tolist(),
            'min_Bz_point_mm': min_z_point.tolist(), 'max_Bz_point_mm': max_z_point.tolist(),
            'max_transverse_point_mm': max_transverse_point.tolist(), 'local_runs': statuses}})
    return refined


def refine_disk_extrema(design, engine, sample, metrics, B_T, starts=3):
    """Independently refine every objective over a fixed-z closed planar disk.

    The maximum principle for a 3-D ball does not make planar disk extrema
    boundary-only. Both maxima and minima therefore use constrained two-
    dimensional interior searches and independent periodic boundary searches.
    Local multistart refinement is not a global-extrema certificate.
    """
    started = time.perf_counter()
    if isinstance(starts, bool) or int(starts) != starts or starts < 1:
        raise ValueError('starts must be a positive integer')
    starts = int(starts)
    radius, z_mm = float(sample['diameter_mm'])/2, float(sample.get('z_mm', 0.))
    points, B = np.asarray(sample['points_mm'], float), np.asarray(B_T, float)
    if (not np.isfinite(radius) or radius <= 0 or not np.isfinite(z_mm)
            or points.shape != B.shape or points.shape[1:] != (3,)
            or not len(points) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(B))):
        raise ValueError('Expected finite matching Nx3 disk points and fields')
    if (not np.allclose(points[:, 2], z_mm, rtol=0, atol=1e-10)
            or np.linalg.norm(points[:, :2], axis=1).max() > radius+1e-8):
        raise ValueError('Disk sample points must lie in the fixed-z FOV')
    unit_points = points[:, :2]/radius
    norms, transverse = np.linalg.norm(B, axis=1), np.linalg.norm(B[:, :2], axis=1)
    counters = {'engine_calls': 0, 'field_points': 0, 'objective_batches': 0}
    statuses = []

    def field(positions):
        positions = np.asarray(positions, float).reshape(-1, 3)
        if not np.all(positions[:, 2] == z_mm): raise AssertionError('Planar refinement changed z')
        counters['engine_calls'] += 1; counters['field_points'] += len(positions)
        values = np.asarray(engine.field(design, positions), float)
        if values.shape != positions.shape or not np.all(np.isfinite(values)):
            raise ValueError('Field engine returned invalid disk fields')
        return values

    def values(fields, kind):
        scaled = fields*1e3
        if kind == 'norm2': return np.einsum('ij,ij->i', scaled, scaled)
        if kind == 'z': return scaled[:, 2]
        return np.einsum('ij,ij->i', scaled[:, :2], scaled[:, :2])

    def lift(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack((radius*xy, np.full(len(xy), z_mm)))

    def objective(transform, kind, sign):
        previous = {}
        def evaluate(x):
            x = np.asarray(x, float); key = tuple(x)
            if previous.get('key') == key: return previous['result']
            step = 1e-5
            stencil = np.vstack((x, x+np.eye(len(x))*step))
            f = sign*values(field(transform(stencil)), kind)
            result = float(f[0]), (f[1:]-f[0])/step
            counters['objective_batches'] += 1
            previous.update(key=key, result=result)
            return result
        return evaluate

    boundary_mask = np.isclose(np.linalg.norm(unit_points, axis=1), 1, rtol=0, atol=1e-7)
    boundary_indices = np.flatnonzero(boundary_mask)

    def extreme(kind, maximize):
        initial_values = values(B, kind)
        index = int(np.argmax(initial_values) if maximize else np.argmin(initial_values))
        best_value, best_point = float(initial_values[index]), points[index].copy()
        sign = -1. if maximize else 1.
        chosen = [unit_points[i].copy() for i in _spaced_indices(unit_points, initial_values, starts, maximize)]
        if not any(np.linalg.norm(p) < 1e-12 for p in chosen): chosen.append(np.zeros(2))
        # Explicit directional interior starts avoid a stationary sampled centre
        # becoming the sole local oracle for a nontrivial planar polynomial.
        if len(chosen) < starts+1:
            for p in (np.array([.5, 0.]), np.array([0., .5])):
                if all(np.linalg.norm(p-q) > .2 for q in chosen): chosen.append(p)
                if len(chosen) >= starts+1: break
        for initial in chosen:
            result = minimize(objective(lift, kind, sign), initial, method='SLSQP', jac=True,
                bounds=[(-1., 1.)]*2,
                constraints=[{'type': 'ineq', 'fun': lambda x: 1.-float(x@x), 'jac': lambda x: -2*np.asarray(x)}],
                options={'maxiter': 60, 'ftol': 1e-12})
            statuses.append({'objective': ('max_' if maximize else 'min_')+kind,
                'domain': 'disk_interior_and_boundary', 'success': bool(result.success),
                'status': int(result.status), 'iterations': int(result.nit)})
            if not np.all(np.isfinite(result.x)) or np.linalg.norm(result.x) > 1+1e-7: continue
            xy = np.asarray(result.x)
            if np.linalg.norm(xy) > 1: xy = xy/np.linalg.norm(xy)
            candidate = lift(xy)[0]
            value = float(values(field(candidate[None, :]), kind)[0])
            if sign*value < sign*best_value: best_value, best_point = value, candidate
        # Angle is unbounded and trigonometric transform periodic: no artificial
        # phi=0 / 2pi cut, and every search remains exactly at the configured z.
        if len(boundary_indices):
            local = _spaced_indices(unit_points[boundary_indices], initial_values[boundary_indices], starts, maximize)
            def circle(theta):
                theta = np.asarray(theta).reshape(-1)
                return np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), z_mm)))
            for item in local:
                point = unit_points[boundary_indices[item]]
                angle = np.arctan2(point[1], point[0])
                result = minimize(objective(circle, kind, sign), [angle], method='BFGS', jac=True,
                    options={'maxiter': 40, 'gtol': 1e-8})
                statuses.append({'objective': ('max_' if maximize else 'min_')+kind,
                    'domain': 'periodic_circle_boundary', 'success': bool(result.success),
                    'status': int(result.status), 'iterations': int(result.nit)})
                if not np.all(np.isfinite(result.x)): continue
                candidate = circle(result.x)[0]
                value = float(values(field(candidate[None, :]), kind)[0])
                if sign*value < sign*best_value: best_value, best_point = value, candidate
        return best_value, best_point

    minimum2, min_point = extreme('norm2', False)
    maximum2, max_point = extreme('norm2', True)
    min_z_mT, min_z_point = extreme('z', False)
    max_z_mT, max_z_point = extreme('z', True)
    maximum_transverse2, max_transverse_point = extreme('transverse2', True)
    minimum, maximum = np.sqrt(max(0., minimum2))/1e3, np.sqrt(max(0., maximum2))/1e3
    min_z, max_z = min_z_mT/1e3, max_z_mT/1e3
    result = dict(metrics)
    result.update({'sampled_pp_norm_ppm': metrics['pp_norm_ppm'],
        'sampled_min_norm_T': float(norms.min()), 'sampled_max_norm_T': float(norms.max()),
        'sampled_pp_Bz_ppm': metrics.get('pp_Bz_ppm'),
        'sampled_min_Bz_T': float(B[:, 2].min()), 'sampled_max_Bz_T': float(B[:, 2].max()),
        'sampled_max_transverse_T': float(transverse.max()),
        'min_norm_T': float(minimum), 'max_norm_T': float(maximum), 'delta_norm_T': float(maximum-minimum),
        'pp_norm_ppm': float((maximum-minimum)/metrics['mean_norm_T']*1e6),
        'min_Bz_T': float(min_z), 'max_Bz_T': float(max_z),
        'pp_Bz_ppm': float((max_z-min_z)/abs(metrics['mean_Bz_T'])*1e6) if abs(metrics.get('mean_Bz_T', 0)) > 1e-12 else None,
        'max_transverse_T': float(np.sqrt(max(0., maximum_transverse2))/1e3),
        'metric_sources': {'mean_norm_T': 'sampled_disk_area_quadrature',
            'rms_norm_ppm': 'sampled_disk_area_quadrature',
            'pp_norm_ppm': 'disk_samples_plus_2d_interior_and_boundary_extrema',
            'pp_Bz_ppm': 'disk_samples_plus_separate_2d_extrema',
            'max_transverse_T': 'disk_samples_plus_separate_2d_extrema'},
        'extrema_refinement': {'method': 'Cartesian disk SLSQP; periodic circle BFGS; batched finite differences',
            'geometry': 'disk', 'measure': 'area', 'z_mm': z_mm, 'diameter_mm': radius*2,
            'assumption': 'all evaluated field points remain on fixed-z planar FOV; no boundary maximum theorem used',
            'global_optimum_proven': False, 'field_evaluations': counters['engine_calls'], **counters,
            'elapsed_seconds': time.perf_counter()-started, 'min_norm_T': float(minimum), 'max_norm_T': float(maximum),
            'min_point_mm': min_point.tolist(), 'max_point_mm': max_point.tolist(),
            'min_Bz_point_mm': min_z_point.tolist(), 'max_Bz_point_mm': max_z_point.tolist(),
            'max_transverse_point_mm': max_transverse_point.tolist(), 'local_runs': statuses}})
    return result


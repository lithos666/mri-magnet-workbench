"""Optional, bounded linear recoil response of uniform cuboid magnetic cells.

This is a separate analysis engine, never an independent-ring MILP basis. Each
magnet is one uniformly polarized cell. Off-diagonal interactions are exact
cuboid fields, collocated at the target centre (or target-volume Gauss points).
The cube self term is ``mu0 H_self = -J/3``. Spatially varying magnetization
inside one magnet, nonlinear material laws and structural response are absent.
Consequently a small linear-system residual is NOT a FEM accuracy certificate.

An exact common rotational subgroup reduces repeated magnets to vector-valued
orbit representatives. No mirror or axial-polarization approximation is used.
"""
from __future__ import annotations

from functools import reduce
import math
import time

import magpylib as magpy
import numpy as np
from scipy.linalg import solve

from .field import FieldEngine
from .geometry import envelope_limits, validate_design
from .identity import canonical_physical_sha256


DEFAULTS = {
    'max_unknowns': 1200, 'max_magnets': 10000, 'max_matrix_mb': 128.,
    'quadrature_order': 1, 'symmetry_reduction': True, 'chunk_points': 48,
    'residual_tolerance': 1e-10,
}


def _options(options):
    result = dict(DEFAULTS)
    if options:
        unknown = set(options) - set(result)
        if unknown:
            raise ValueError('Unknown material-response options: ' + ', '.join(sorted(unknown)))
        result.update(options)
    for key in ('max_unknowns', 'max_magnets', 'quadrature_order', 'chunk_points'):
        value = result[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < 1:
            raise ValueError(key + ' must be a positive integer')
        result[key] = int(value)
    if result['quadrature_order'] not in (1, 2, 3):
        raise ValueError('quadrature_order must be 1, 2 or 3')
    if not isinstance(result['symmetry_reduction'], bool):
        raise ValueError('symmetry_reduction must be boolean')
    for key in ('max_matrix_mb', 'residual_tolerance'):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key + ' must be finite and positive')
        result[key] = value
    return result


def preflight_response(design, options=None):
    """Cheap source/orbit/memory bounds before allocating any interaction matrix."""
    settings = _options(options)
    if not design.rings:
        raise ValueError('Material response needs at least one magnetic ring')
    for ring in design.rings:
        if (isinstance(ring.count, bool) or int(ring.count) != ring.count or ring.count < 3
                or ring.polarity not in ('in', 'out', 'up', 'down')
                or not np.all(np.isfinite([ring.radius_mm, ring.z_mm, ring.phase_deg]))
                or ring.radius_mm <= 0):
            raise ValueError('Invalid ring in material-response design')
    if not np.all(np.isfinite([design.cube_mm, design.remanence_T, design.reference_mur])):
        raise ValueError('Magnet material and dimensions must be finite')
    if design.cube_mm <= 0 or design.remanence_T <= 0 or not 1 <= design.reference_mur <= 2:
        raise ValueError('Linear recoil analysis requires cube>0, Br>0 and 1<=mu_r<=2')
    order = reduce(math.gcd, (int(r.count) for r in design.rings)) if settings['symmetry_reduction'] else 1
    representatives = sum(int(r.count)//order for r in design.rings)
    unknowns = representatives*3
    # Matrix, solve work array and temporary assembly/safety overhead. This is
    # an explicit bound estimate, not a promise of total interpreter RSS.
    matrix_bytes = unknowns*unknowns*8
    # Include observer-chunk arrays too; a large custom chunk must not bypass
    # the matrix preflight even when rotational reduction makes D tiny.
    field_chunk_bytes=max(1,design.magnet_count)*settings['chunk_points']*3*8*12
    estimated_bytes = matrix_bytes*4 + max(1, design.magnet_count)*settings['quadrature_order']**3*3*8*12 + field_chunk_bytes
    violations = []
    if design.magnet_count > settings['max_magnets']:
        violations.append('magnet count exceeds max_magnets')
    if unknowns > settings['max_unknowns']:
        violations.append('rotationally reduced vector dimension exceeds max_unknowns')
    if estimated_bytes > settings['max_matrix_mb']*1e6:
        violations.append('estimated material-response working memory exceeds max_matrix_mb')
    return {'valid': not violations, 'violations': violations,
            'magnet_count': int(design.magnet_count), 'ring_count': len(design.rings),
            'rotation_order': order, 'representative_cells': representatives,
            'vector_unknowns': unknowns, 'matrix_bytes': matrix_bytes,
            'estimated_working_bytes': estimated_bytes,
            'quadrature_order': settings['quadrature_order'],
            'target_points_per_cell': settings['quadrature_order']**3,
            'source_target_evaluations': int(representatives*design.magnet_count*settings['quadrature_order']**3*3),
            'settings': settings, 'scope': 'bounded uniform-cuboid linear material response'}


def _orbit_arrays(design, order):
    arrays = design.magnet_arrays()
    angles = np.deg2rad(arrays['radial_angles_deg'])
    cs, sn = np.cos(angles), np.sin(angles)
    rotations = np.zeros((len(angles), 3, 3))
    rotations[:, 0, 0] = cs
    rotations[:, 0, 1] = -sn
    rotations[:, 1, 0] = sn
    rotations[:, 1, 1] = cs
    rotations[:, 2, 2] = 1.
    orbit, representatives, ring_index = [], [], []
    offset = 0
    next_orbit = 0
    for index, ring in enumerate(design.rings):
        distinct = int(ring.count)//order
        representatives.extend(offset + np.arange(distinct))
        orbit.extend(next_orbit + np.arange(ring.count) % distinct)
        ring_index.extend([index]*ring.count)
        next_orbit += distinct
        offset += ring.count
    local_north = np.einsum('nji,nj->ni', rotations, arrays['north_vectors'])
    return {'positions_m': arrays['positions_mm']*1e-3, 'rotations': rotations,
            'orbit': np.asarray(orbit, dtype=int), 'representatives': np.asarray(representatives, dtype=int),
            'ring_index': np.asarray(ring_index, dtype=int),
            'local_remanence_T': local_north*design.remanence_T}


def _target_quadrature(cube_m, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    grid = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    weight_grid = np.meshgrid(weights, weights, weights, indexing='ij')
    offsets = np.stack(grid, axis=-1).reshape(-1, 3)*cube_m/2
    values = np.prod(np.stack(weight_grid, axis=-1), axis=-1).ravel()/8
    return offsets, values


def _interaction_matrix(arrays, cube_m, order):
    positions, rotations = arrays['positions_m'], arrays['rotations']
    orbit, representatives = arrays['orbit'], arrays['representatives']
    count = len(representatives)
    offsets, weights = _target_quadrature(cube_m, order)
    interaction = np.zeros((count, 3, count, 3))
    for target_index, physical_index in enumerate(representatives):
        target_points = positions[physical_index] + offsets @ rotations[physical_index].T
        relative_global = target_points[:, None, :] - positions[None, :, :]
        relative_local = np.einsum('qni,nij->qnj', relative_global, rotations)
        active = np.arange(len(positions)) != physical_index
        for component in range(3):
            polarization = np.zeros((len(target_points)*int(active.sum()), 3))
            polarization[:, component] = 1.
            field_local = magpy.getB('Cuboid',
                observers=relative_local[:, active].reshape(-1, 3),
                dimension=(cube_m,)*3, polarization=polarization).reshape(len(target_points), -1, 3)
            field_global = np.einsum('nij,qnj->qni', rotations[active], field_local)
            averaged_global = np.einsum('q,qni->ni', weights, field_global)
            averaged_target_local = averaged_global @ rotations[physical_index]
            # Several physical magnets in one rotational orbit share one
            # unknown LOCAL vector, with each correctly rotated at its source.
            np.add.at(interaction[target_index, :, :, component].T,
                      orbit[active], averaged_target_local)
    return interaction.reshape(count*3, count*3)


def _solve_polarization(interaction, remanence, susceptibility):
    """Solve J=Jr+chi(-J/3 + D J), with T-valued J and mu0 H."""
    matrix = np.eye(len(remanence))*(1.+susceptibility/3.) - susceptibility*interaction
    polarization = solve(matrix, remanence, assume_a='gen', check_finite=True)
    residual = matrix @ polarization - remanence
    absolute = float(np.max(np.abs(residual)))
    relative = absolute/max(float(np.max(np.abs(remanence))), np.finfo(float).tiny)
    return polarization, relative, absolute


class CoupledResponseEngine:
    """Prepared optional material-response evaluator, compatible with field()."""
    polarization_model = 'coupled_uniform_cells'

    def __init__(self, design, options=None):
        started = time.perf_counter()
        plan = preflight_response(design, options)
        if not plan['valid']:
            raise ValueError('; '.join(plan['violations']))
        geometry = validate_design(design, envelope_limits(design))
        if not geometry['valid']:
            raise ValueError('Material response requires non-overlapping magnetic cells: '
                             + '; '.join(geometry['violations']))
        self.settings = plan['settings']
        self.design_sha256 = canonical_physical_sha256(design, self.polarization_model)
        self.design = design
        self.statistics = {'engine_calls': 0, 'field_points': 0, 'field_seconds': 0., 'preparations': 1}
        arrays = _orbit_arrays(design, plan['rotation_order'])
        self._arrays = arrays
        chi = float(design.reference_mur)-1.
        representatives = arrays['representatives']
        remanence = arrays['local_remanence_T'][representatives].ravel()
        self._uncoupled = FieldEngine(self.settings['chunk_points'], 'remanence') if chi == 0 else None
        if chi == 0:
            polarization, relative, absolute = remanence.copy(), 0., 0.
            contraction_bound = 0.
        else:
            interaction = _interaction_matrix(arrays, design.cube_mm*1e-3, self.settings['quadrature_order'])
            contraction_bound = float(chi*np.linalg.norm(interaction, ord=np.inf)/(1.+chi/3.))
            polarization, relative, absolute = _solve_polarization(interaction, remanence, chi)
        if not np.all(np.isfinite(polarization)) or relative > self.settings['residual_tolerance']:
            raise FloatingPointError('Material-response recoil equation did not meet residual tolerance')
        self.local_polarization_T = polarization.reshape(-1, 3)[arrays['orbit']]
        isolated = arrays['local_remanence_T']/(1.+chi/3.)
        magnitude = np.linalg.norm(self.local_polarization_T, axis=1)
        self.ring_summary = []
        for index, ring in enumerate(design.rings):
            members = arrays['ring_index'] == index
            local = self.local_polarization_T[members]
            self.ring_summary.append({'ring_id': ring.id, 'layer_index': ring.layer_index,
                'radius_mm': ring.radius_mm, 'z_mm': ring.z_mm, 'count': ring.count,
                'mean_local_polarization_T': local.mean(axis=0).tolist(),
                'min_polarization_norm_T': float(magnitude[members].min()),
                'max_polarization_norm_T': float(magnitude[members].max()),
                'max_change_from_isolated_T': float(np.linalg.norm(local-isolated[members], axis=1).max())})
        self.diagnostics = {'model': self.polarization_model,
            'design_sha256': self.design_sha256, 'recoil_permeability': design.reference_mur,
            'susceptibility': chi, 'preflight': plan,
            'self_demagnetization_tensor': (np.eye(3)/3.).tolist(),
            'self_demagnetization_factor': 1./3.,
            'equation': '(I + chi/3 - chi D) J = Jr; D uses exact cuboid fields in local rotating frames',
            'relative_recoil_residual': relative, 'absolute_recoil_residual_T': absolute,
            'linear_response_contraction_bound_inf': contraction_bound,
            'contraction_sufficient_condition_satisfied': contraction_bound < 1.,
            'max_polarization_change_from_remanence_T': float(np.linalg.norm(
                self.local_polarization_T-arrays['local_remanence_T'], axis=1).max()),
            'max_polarization_change_from_isolated_T': float(np.linalg.norm(
                self.local_polarization_T-isolated, axis=1).max()),
            'preparation_seconds': time.perf_counter()-started,
            'spatial_model': 'one uniform vector polarization per full-size cuboid',
            'target_integration': 'cell-centre collocation' if self.settings['quadrature_order'] == 1
                else 'tensor Gauss target-volume averaging; exact finite cuboid sources',
            'fem_validated': False, 'spatial_convergence_proven': False,
            'limitations': ['uniform polarization cannot resolve within-magnet material response',
                'linear isotropic recoil law only; no coercivity or nonlinear B-H curve',
                'solver residual controls algebra only, not FEM agreement or ppm accuracy',
                'coupled fields cannot be added from independently solved topology basis rings']}

    def field(self, design, points_mm):
        if canonical_physical_sha256(design, self.polarization_model) != self.design_sha256:
            raise ValueError('Prepared material response belongs to a different physical design')
        started = time.perf_counter()
        points = np.asarray(points_mm, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('Material-response observers must be finite Nx3 points in mm')
        if self._uncoupled is not None:
            result = self._uncoupled.field(design, points)
        else:
            arrays = self._arrays
            rotations, positions = arrays['rotations'], arrays['positions_m']
            result = np.empty((len(points), 3))
            for start in range(0, len(points), self.settings['chunk_points']):
                part = points[start:start+self.settings['chunk_points']]*1e-3
                relative = part[:, None, :] - positions[None, :, :]
                local = np.einsum('qni,nij->qnj', relative, rotations)
                fields = magpy.getB('Cuboid', observers=local.reshape(-1, 3),
                    dimension=(design.cube_mm*1e-3,)*3,
                    polarization=np.tile(self.local_polarization_T, (len(part), 1))).reshape(len(part), -1, 3)
                result[start:start+len(part)] = np.einsum('nij,qnj->qi', rotations, fields)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('Non-finite coupled magnetic field')
        self.statistics['engine_calls'] += 1
        self.statistics['field_points'] += len(points)
        self.statistics['field_seconds'] += time.perf_counter()-started
        return result


def coupled_response(design, points_mm, options=None):
    engine = CoupledResponseEngine(design, options)
    return {'fields_T': engine.field(design, points_mm),
            'metadata': engine.diagnostics, 'ring_polarization': engine.ring_summary}

"""Optional, bounded linear recoil response of uniform cuboid magnetic cells.

This prepared engine is also used by the V6 optimization factory. It is never
an independent-ring MILP basis. A magnet has one or eight polarized cells.
Off-diagonal interactions are exact
cuboid fields, collocated at the target centre (or target-volume Gauss points).
The cube self term is ``mu0 H_self = -J/3``. Unresolved variation within each
cell, nonlinear material laws and structural response are absent.
Consequently a small linear-system residual is NOT a FEM accuracy certificate.

An exact common rotational subgroup reduces repeated magnets to vector-valued
orbit representatives. No mirror or axial-polarization approximation is used.
"""
from __future__ import annotations

from functools import reduce
import math
import json
import time

import magpylib as magpy
import numpy as np
from scipy.linalg import solve

from .field import FieldEngine
from .geometry import envelope_limits, validate_design
from .identity import canonical_physical_sha256, canonical_phase_deg


DEFAULTS = {
    'max_unknowns': 1200, 'max_magnets': 10000, 'max_matrix_mb': 128.,
    'quadrature_order': 1, 'symmetry_reduction': True, 'chunk_points': 48,
    'residual_tolerance': 1e-10, 'cells_per_axis': 1, 'interaction_backend':'analytic_tensor',
}


def _options(options):
    result = dict(DEFAULTS)
    if options:
        unknown = set(options) - set(result)
        if unknown:
            raise ValueError('Unknown material-response options: ' + ', '.join(sorted(unknown)))
        result.update(options)
    for key in ('max_unknowns', 'max_magnets', 'quadrature_order', 'chunk_points', 'cells_per_axis'):
        value = result[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < 1:
            raise ValueError(key + ' must be a positive integer')
        result[key] = int(value)
    if result['quadrature_order'] not in (1, 2, 3):
        raise ValueError('quadrature_order must be 1, 2 or 3')
    if result['cells_per_axis'] not in (1, 2):
        raise ValueError('cells_per_axis must be 1 or 2')
    if not isinstance(result['symmetry_reduction'], bool):
        raise ValueError('symmetry_reduction must be boolean')
    if result['interaction_backend'] not in ('analytic_tensor','public_magpylib'):
        raise ValueError('interaction_backend must be analytic_tensor or public_magpylib')
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
    cells_per_magnet = settings['cells_per_axis']**3
    source_cells = design.magnet_count*cells_per_magnet
    representatives = sum(int(r.count)//order for r in design.rings)*cells_per_magnet
    unknowns = representatives*3
    # Matrix, solve work array and temporary assembly/safety overhead. This is
    # an explicit bound estimate, not a promise of total interpreter RSS.
    matrix_bytes = unknowns*unknowns*8
    # Include observer-chunk arrays too; a large custom chunk must not bypass
    # the matrix preflight even when rotational reduction makes D tiny.
    # The tensor path holds geometric scalar terms and nine tensor components
    # per source-observer row. Include its peak temporary footprint, rather
    # than counting only the final vector field arrays.
    temporary_factor=20 if settings['interaction_backend']=='analytic_tensor' else 12
    field_chunk_bytes=max(1,source_cells)*settings['chunk_points']*3*8*temporary_factor
    estimated_bytes = matrix_bytes*4 + max(1, source_cells)*settings['quadrature_order']**3*3*8*16 + field_chunk_bytes
    violations = []
    if design.magnet_count > settings['max_magnets']:
        violations.append('magnet count exceeds max_magnets')
    if unknowns > settings['max_unknowns']:
        violations.append('rotationally reduced vector dimension exceeds max_unknowns')
    if estimated_bytes > settings['max_matrix_mb']*1e6:
        violations.append('estimated material-response working memory exceeds max_matrix_mb')
    return {'valid': not violations, 'violations': violations,
            'magnet_count': int(design.magnet_count), 'ring_count': len(design.rings),
            'source_cells': int(source_cells), 'cells_per_magnet': cells_per_magnet,
            'cells_per_axis': settings['cells_per_axis'],
            'rotation_order': order, 'representative_cells': representatives,
            'vector_unknowns': unknowns, 'matrix_bytes': matrix_bytes,
            'estimated_working_bytes': estimated_bytes,
            'quadrature_order': settings['quadrature_order'],
            'target_points_per_cell': settings['quadrature_order']**3,
            'source_target_evaluations': int(representatives*source_cells*settings['quadrature_order']**3*2),
            'settings': settings, 'scope': 'bounded uniform-cuboid linear material response'}


def _orbit_arrays(design, order, cells_per_axis=1):
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
    result = {'positions_m': arrays['positions_mm']*1e-3, 'rotations': rotations,
            'orbit': np.asarray(orbit, dtype=int), 'representatives': np.asarray(representatives, dtype=int),
            'ring_index': np.asarray(ring_index, dtype=int),
            'local_remanence_T': local_north*design.remanence_T}
    count = cells_per_axis**3
    if count > 1:
        axis = (np.arange(cells_per_axis)+.5)/cells_per_axis-.5
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)*design.cube_mm*1e-3
        result['positions_m'] = (result['positions_m'][:, None, :]
            + np.einsum('nij,sj->nsi', rotations, offsets)).reshape(-1, 3)
        result['rotations'] = np.repeat(rotations, count, axis=0)
        result['local_remanence_T'] = np.repeat(result['local_remanence_T'], count, axis=0)
        result['ring_index'] = np.repeat(result['ring_index'], count)
        result['orbit'] = (result['orbit'][:, None]*count+np.arange(count)).ravel()
        result['representatives'] = (result['representatives'][:, None]*count+np.arange(count)).ravel()
    return result


def _target_quadrature(cube_m, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    grid = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    weight_grid = np.meshgrid(weights, weights, weights, indexing='ij')
    offsets = np.stack(grid, axis=-1).reshape(-1, 3)*cube_m/2
    values = np.prod(np.stack(weight_grid, axis=-1), axis=-1).ravel()/8
    return offsets, values


def _representative_keys(design, arrays, rotation_order, cells_per_axis):
    """Physical orbit labels for IPC across equivalent ring/column orderings.

    Geometry SHA deliberately ignores user ring order and periodic phase.
    Representative vectors cannot therefore be transported by array index.
    Local vectors are equal around one rotational orbit; its angle modulo
    the common subgroup period and its local subcell label identify it.
    """
    period=360./rotation_order
    result=[]
    for index in arrays['representatives']:
        ring=design.rings[int(arrays['ring_index'][index])]
        rotation=arrays['rotations'][index]
        angle=math.degrees(math.atan2(rotation[1,0],rotation[0,0]))%period
        angle=round(angle,9)
        if angle<=.5e-9 or period-angle<=.5e-9:angle=0.
        key=[round(ring.z_mm,9),round(ring.radius_mm,9),ring.count,
             canonical_phase_deg(ring.phase_deg,ring.count),ring.polarity,
             angle,int(index)%(cells_per_axis**3)]
        result.append(json.dumps(key,separators=(',',':')))
    return result


def _interaction_matrix(arrays, cube_m, order, cancel_check=None, status_callback=None,
                        backend='analytic_tensor', assembly_info=None):
    positions, rotations = arrays['positions_m'], arrays['rotations']
    orbit, representatives = arrays['orbit'], arrays['representatives']
    count = len(representatives)
    offsets, weights = _target_quadrature(cube_m, order)
    interaction = np.zeros((count, 3, count, 3))
    fallback_rows=0
    for target_index, physical_index in enumerate(representatives):
        if cancel_check is not None: cancel_check()
        if status_callback is not None and (target_index % max(1, count//20) == 0):
            status_callback({'stage':'coupling_matrix', 'completed':target_index, 'total':count,
                             'progress_percent':100.*target_index/count})
        target_points = positions[physical_index] + offsets @ rotations[physical_index].T
        active = np.arange(len(positions)) != physical_index
        relative_global = target_points[:, None, :] - positions[None, active, :]
        relative_local = np.einsum('qni,nij->qnj', relative_global, rotations[active])
        # Outside every other non-overlapping source, its field kernel is a
        # symmetric, trace-free Hessian. Two public Magpylib unit-source calls
        # therefore determine all nine tensor entries; this saves one full
        # cuboid evaluation without dropping any interaction or component.
        observers=relative_local.reshape(-1,3)
        if backend=='analytic_tensor':
            from .cuboid_tensor import external_cube_tensor
            tensor=external_cube_tensor(observers,cube_m)
            invalid=~np.all(np.isfinite(tensor),axis=(1,2))
            if np.any(invalid):
                fallback_rows+=int(invalid.sum())
                # Public API handles limiting edge/surface cases. This fallback
                # changes no material physics and omits no source interaction.
                for component in range(3):
                    unit=np.eye(3)[component]
                    tensor[invalid,:,component]=magpy.getB('Cuboid',observers=observers[invalid],
                        dimension=(cube_m,)*3,polarization=unit)
            if not np.all(np.isfinite(tensor)):
                raise FloatingPointError('Non-finite cuboid interaction after public Magpylib fallback')
            kernel=np.einsum('q,qnij->nij',weights,tensor.reshape(len(target_points),-1,3,3))
        elif backend=='public_magpylib':
            columns=[]
            for component in range(2):
                polarization=np.zeros(3);polarization[component]=1.
                values=magpy.getB('Cuboid',observers=observers,
                    dimension=(cube_m,)*3,polarization=polarization).reshape(len(target_points),-1,3)
                columns.append(np.einsum('q,qni->ni',weights,values))
            kernel=np.empty((int(active.sum()),3,3))
            kernel[:,:,0],kernel[:,:,1]=columns
            kernel[:,0,1]=kernel[:,1,0]=(columns[0][:,1]+columns[1][:,0])/2
            kernel[:,0,2]=columns[0][:,2]
            kernel[:,1,2]=columns[1][:,2]
            kernel[:,2,2]=-columns[0][:,0]-columns[1][:,1]
        else:raise ValueError('Unknown interaction tensor backend')
        if not np.all(np.isfinite(kernel)):
            raise FloatingPointError('Non-finite public cuboid interaction kernel')
        source_to_target = np.einsum('ab,nbc->nac', rotations[physical_index].T, rotations[active])
        target_kernel = np.einsum('nab,nbc->nac', source_to_target, kernel)
        for row in range(3):
            for column in range(3):
                interaction[target_index, row, :, column] = np.bincount(
                    orbit[active], weights=target_kernel[:, row, column], minlength=count)
    if status_callback is not None:
        status_callback({'stage':'coupling_matrix', 'completed':count, 'total':count, 'progress_percent':100.})
    if assembly_info is not None:assembly_info.update(backend=backend,fallback_observer_rows=fallback_rows)
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

    def __init__(self, design, options=None, cancel_check=None, status_callback=None, prepared_response=None):
        started = time.perf_counter()
        plan = preflight_response(design, options)
        if not plan['valid']:
            raise ValueError('; '.join(plan['violations']))
        geometry = validate_design(design, envelope_limits(design))
        if not geometry['valid']:
            raise ValueError('Material response requires non-overlapping magnetic cells: '
                             + '; '.join(geometry['violations']))
        self.settings = plan['settings']
        self.cancel_check=cancel_check
        self.design_sha256 = canonical_physical_sha256(design, self.polarization_model)
        self.design = design
        self.statistics = {'engine_calls': 0, 'field_points': 0, 'field_seconds': 0., 'preparations': 1,
                           'field_tensor_fallback_rows':0}
        arrays = _orbit_arrays(design, plan['rotation_order'], self.settings['cells_per_axis'])
        self._arrays = arrays
        self._representative_keys=_representative_keys(design,arrays,plan['rotation_order'],self.settings['cells_per_axis'])
        self.cell_edge_m = design.cube_mm*1e-3/self.settings['cells_per_axis']
        chi = float(design.reference_mur)-1.
        representatives = arrays['representatives']
        remanence = arrays['local_remanence_T'][representatives].ravel()
        self._uncoupled = FieldEngine(self.settings['chunk_points'], 'remanence') if chi == 0 else None
        if prepared_response is not None:
            expected = (len(representatives), 3)
            polarization = np.asarray(prepared_response['representative_polarization_T'], dtype=float)
            if polarization.shape != expected or not np.all(np.isfinite(polarization)):
                raise ValueError('Invalid prepared representative polarization')
            if (prepared_response.get('design_sha256') != self.design_sha256
                    or prepared_response.get('settings') != self.settings):
                raise ValueError('Prepared response identity/settings mismatch')
            source_keys=prepared_response.get('representative_keys')
            if (not isinstance(source_keys,list) or len(source_keys)!=len(self._representative_keys)
                    or len(set(source_keys))!=len(source_keys)
                    or set(source_keys)!=set(self._representative_keys)):
                raise ValueError('Prepared response representative identity mismatch')
            by_key={key:vector for key,vector in zip(source_keys,polarization)}
            polarization=np.asarray([by_key[key] for key in self._representative_keys])
            polarization = polarization.ravel()
            prior = prepared_response['diagnostics']
            relative, absolute = prior['relative_recoil_residual'], prior['absolute_recoil_residual_T']
            contraction_bound = prior['linear_response_contraction_bound_inf']
        elif chi == 0:
            polarization, relative, absolute = remanence.copy(), 0., 0.
            contraction_bound = 0.
        else:
            assembly_info={}
            interaction = _interaction_matrix(arrays, self.cell_edge_m, self.settings['quadrature_order'], cancel_check, status_callback,
                backend=self.settings['interaction_backend'],assembly_info=assembly_info)
            contraction_bound = float(chi*np.linalg.norm(interaction, ord=np.inf)/(1.+chi/3.))
            if cancel_check is not None: cancel_check()
            if status_callback is not None: status_callback({'stage':'coupling_linear_solve', 'message':'正在求解自洽磁化；停止将在本次线性求解结束后生效'})
            polarization, relative, absolute = _solve_polarization(interaction, remanence, chi)
            if cancel_check is not None: cancel_check()
        if (not np.all(np.isfinite(polarization)) or not np.all(np.isfinite([relative,absolute,contraction_bound]))
                or relative < 0 or absolute < 0 or relative > self.settings['residual_tolerance']):
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
            'material_remanence_T': design.remanence_T,
            'material_law': 'linear isotropic recoil J = Br*n + mu0*(mu_r-1)*H',
            'measured_JH_curve_used': False, 'irreversible_demagnetization_modeled': False,
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
            'spatial_model': ('one uniform vector polarization per full-size cuboid' if self.settings['cells_per_axis']==1
                              else 'eight independently polarized cuboid cells per physical magnet'),
            'cells_per_axis': self.settings['cells_per_axis'], 'source_cells': plan['source_cells'],
            'response_imported': prepared_response is not None,
            'interaction_backend':self.settings['interaction_backend'],
            'tensor_fallback_observer_rows':assembly_info.get('fallback_observer_rows',0) if chi and prepared_response is None else 0,
            'field_tensor_fallback_observer_rows':0,
            'target_integration': 'cell-centre collocation' if self.settings['quadrature_order'] == 1
                else 'tensor Gauss target-volume averaging; exact finite cuboid sources',
            'fem_validated': False, 'spatial_convergence_proven': False,
            'limitations': ['piecewise-uniform polarization requires spatial convergence checks',
                'linear isotropic recoil law only; no coercivity or nonlinear B-H curve',
                'solver residual controls algebra only, not FEM agreement or ppm accuracy',
                'coupled fields cannot be added from independently solved topology basis rings']}

    def export_prepared(self):
        """Internal worker-to-monitor data, never a substitute for validation."""
        return {'design_sha256': self.design_sha256, 'settings': dict(self.settings),
                'representative_keys':list(self._representative_keys),
                'representative_polarization_T': self.local_polarization_T[self._arrays['representatives']].tolist(),
                'diagnostics': dict(self.diagnostics)}

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
                if self.cancel_check is not None:self.cancel_check()
                part = points[start:start+self.settings['chunk_points']]*1e-3
                relative = part[:, None, :] - positions[None, :, :]
                local = np.einsum('qni,nij->qnj', relative, rotations)
                if self.settings['interaction_backend']=='analytic_tensor':
                    from .cuboid_tensor import external_cube_tensor
                    tensor=external_cube_tensor(local.reshape(-1,3),self.cell_edge_m).reshape(len(part),-1,3,3)
                    unsafe=~np.all(np.isfinite(tensor),axis=(2,3))
                    fields=np.einsum('qnij,nj->qni',tensor,self.local_polarization_T)
                    if np.any(unsafe):
                        local_j=np.broadcast_to(self.local_polarization_T,local.shape)
                        fields[unsafe]=magpy.getB('Cuboid',observers=local[unsafe],
                            dimension=(self.cell_edge_m,)*3,polarization=local_j[unsafe])
                        self.statistics['field_tensor_fallback_rows']+=int(unsafe.sum())
                        self.diagnostics['field_tensor_fallback_observer_rows']=self.statistics['field_tensor_fallback_rows']
                    if not np.all(np.isfinite(fields)):
                        raise FloatingPointError('Non-finite cuboid field after public Magpylib fallback')
                else:
                    fields = magpy.getB('Cuboid', observers=local.reshape(-1, 3),
                        dimension=(self.cell_edge_m,)*3,
                        polarization=np.tile(self.local_polarization_T, (len(part), 1))).reshape(len(part), -1, 3)
                result[start:start+len(part)] = np.einsum('nij,qnj->qi', rotations, fields)
                # Release the previous chunk before evaluating the next kernel;
                # otherwise Python assignment keeps two tensor chunks alive.
                del relative,local,fields
                if self.settings['interaction_backend']=='analytic_tensor':del tensor,unsafe
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

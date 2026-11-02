"""Reproducible 3D quadrature, boundary checks and independent validation points."""
from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from collections import Counter
import math
from functools import reduce
from .identity import canonical_physical_data


def symmetry_profile(design, rotation_order=None, reflection_phase_deg=None):
    """Conservative exact-model symmetries of field magnitude, not arbitrary B.

    Rotation is common gcd of ring counts. A meridian may be a column plane
    or a seam plane; geometry and axial-vector magnetization determine the
    z reflection. Manufacturing perturbations invalidate this ideal symmetry.
    """
    rings = canonical_physical_data(design)['rings']
    if not rings:
        raise ValueError('Cannot determine symmetry of empty design')
    natural_order = reduce(math.gcd, (r['count'] for r in rings))
    order = natural_order if rotation_order is None else int(rotation_order)
    if order < 1 or (rotation_order is not None and order != rotation_order):
        raise ValueError('rotation_order must be a positive integer')
    rotation_valid = all(r['count'] % order == 0 for r in rings)
    first = rings[0]
    candidate_planes = ((first['phase_deg'], first['phase_deg']-180/first['count'])
                        if reflection_phase_deg is None else (float(reflection_phase_deg),))
    def mirror_plane(plane):
        for ring in rings:
            period = 360/ring['count']
            error = (2*(ring['phase_deg']-plane)+period/2) % period-period/2
            if abs(error) > 5e-8:
                return False
        return True
    meridian = next((plane for plane in candidate_planes if mirror_plane(plane)), None)
    def signature(ring):
        return (round(ring['radius_mm'], 8), round(ring['z_mm'], 8), ring['count'],
                round(ring['phase_deg'], 8), ring['polarity'])
    mapping = {'out': 'in', 'in': 'out', 'down': 'down', 'up': 'up'}
    actual = Counter(signature(r) for r in rings)
    reflected = Counter(signature(dict(r, z_mm=-r['z_mm'], polarity=mapping[r['polarity']]))
                        for r in rings)
    mirror_z = actual == reflected
    fraction = 1/order/(2 if meridian is not None else 1)/(2 if mirror_z else 1)
    return {'rotation_order': order, 'natural_rotation_order': natural_order,
            'rotation_valid': rotation_valid, 'verified': bool(rotation_valid and mirror_z and meridian is not None),
            'mirror_z': mirror_z,
            'meridian_reflection': meridian is not None,
            'meridian_plane_deg': float(meridian or 0), 'domain_volume_fraction': fraction,
            'valid_for': ['field_magnitude', 'Bz_volume_mean', 'transverse_magnitude'],
            'assumption': 'ideal uniform rings with listed geometry and magnetization; no tolerance perturbations',
            'speedup_guaranteed': False}


def fundamental_sphere_samples(diameter_mm=180., n_volume=128, n_surface=128,
                               seed=20261028, rotation_order=1, reflection_phase_deg=-90.0):
    """Uniform quadrature in the verified ideal-model symmetry domain.

    The mean is normalized over this equal-volume domain. Returned vector
    components are fields at the listed points; they are not full-domain
    extrema of Bx or By. Other diagnostics must use their explicit symmetries.
    """
    if diameter_mm <= 0 or n_volume < 1 or n_surface < 1:
        raise ValueError('Invalid sphere sampling')
    if int(rotation_order) != rotation_order or rotation_order < 1:
        raise ValueError('rotation_order must be a positive integer')
    profile = {'rotation_order': int(rotation_order), 'mirror_z': True,
               'meridian_reflection': True, 'meridian_plane_deg': float(reflection_phase_deg),
               'domain_volume_fraction': 1/(4*int(rotation_order)),
               'requires_verified_design': True, 'speedup_guaranteed': False}
    start = math.radians(profile['meridian_plane_deg'])
    angle = (math.pi if profile['meridian_reflection'] else 2*math.pi)/profile['rotation_order']
    def directions(unit):
        cosine = unit[:, 0] if profile['mirror_z'] else 1-2*unit[:, 0]
        phi = start+angle*unit[:, 1]
        radial = np.sqrt(np.maximum(0, 1-cosine*cosine))
        return np.column_stack((radial*np.cos(phi), radial*np.sin(phi), cosine))
    volume_unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(n_volume))))[:n_volume]
    volume = directions(volume_unit[:, 1:])*(diameter_mm/2*volume_unit[:, 0]**(1/3))[:, None]
    surface_unit = qmc.Sobol(2, scramble=True, seed=seed+37).random_base2(
        int(np.ceil(np.log2(n_surface))))[:n_surface]
    surface = directions(surface_unit)*diameter_mm/2
    # Explicit boundaries and pole remain, without biasing volume averages.
    anchor_unit = np.array([[0, 0], [0, 1], [0, .5], [1, 0], [1, 1], [.5, .5]])
    anchors = directions(anchor_unit)*diameter_mm/2
    points = np.vstack((np.zeros((1, 3)), volume, surface, anchors))
    weights = np.r_[0., np.full(n_volume, 1/n_volume), np.zeros(n_surface+len(anchors))]
    return {'points_mm': points, 'weights': weights, 'diameter_mm': diameter_mm,
            'n_volume': n_volume, 'n_surface': n_surface, 'seed': seed,
            'symmetry_profile': profile, 'scope': 'ideal-model symmetry fundamental domain',
            'name': f'sphere_fundamental_D{diameter_mm:g}_V{n_volume}_S{n_surface}_seed{seed}'}

def sphere_samples(diameter_mm=180.,n_volume=128,n_surface=96,seed=20261028):
    if diameter_mm<=0 or n_volume<1 or n_surface<1:raise ValueError('Invalid sphere sampling')
    exponent=int(np.ceil(np.log2(n_volume)))
    unit=qmc.Sobol(3,scramble=True,seed=seed).random_base2(exponent)[:n_volume]
    radius=diameter_mm/2*unit[:,0]**(1/3)
    z=1-2*unit[:,1];phi=2*np.pi*unit[:,2]
    volume=np.column_stack((radius*np.sqrt(1-z*z)*np.cos(phi),radius*np.sqrt(1-z*z)*np.sin(phi),radius*z))
    i=np.arange(n_surface);z=1-2*(i+.5)/n_surface;phi=i*np.pi*(3-np.sqrt(5))+.371
    surface=diameter_mm/2*np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
    axes=np.vstack((np.eye(3),-np.eye(3)))*diameter_mm/2
    points=np.vstack((np.zeros((1,3)),volume,surface,axes))
    # Boundary samples contribute to extrema, never bias the volume mean/RMS.
    weights=np.r_[0.,np.full(n_volume,1/n_volume),np.zeros(n_surface+6)]
    return {"points_mm":points,"weights":weights,"name":f"sphere_D{diameter_mm:g}_V{n_volume}_S{n_surface}_seed{seed}",
            "diameter_mm":diameter_mm,"n_volume":n_volume,"n_surface":n_surface,"seed":seed}


def disk_samples(diameter_mm=20., n_area=128, n_boundary=128, seed=20261028,
                 z_mm=0., rotation_order=None, reflection_phase_deg=-90.):
    """Area quadrature and boundary anchors on one fixed-z circular FOV.

    Radius is R*sqrt(u), so only area points have nonzero mean/RMS weights.
    Supplying rotation_order requests a meridian half-sector of angle pi/N;
    the caller must first verify that common symmetry for every candidate.
    This reduces the integration domain, never the physical FOV definition.
    """
    diameter_mm, z_mm = float(diameter_mm), float(z_mm)
    if not np.isfinite(diameter_mm) or diameter_mm <= 0 or not np.isfinite(z_mm):
        raise ValueError('Disk diameter must be positive and z finite')
    for name, count in (('n_area', n_area), ('n_boundary', n_boundary)):
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError(name+' must be a positive integer')
    n_area, n_boundary = int(n_area), int(n_boundary)
    if rotation_order is None:
        start, width = 0., 2*np.pi
        boundary_angles = start+width*np.arange(n_boundary)/n_boundary
        anchor_angles = np.arange(4)*np.pi/2
        profile = None
    else:
        if isinstance(rotation_order, bool) or int(rotation_order) != rotation_order or rotation_order < 1:
            raise ValueError('rotation_order must be a positive integer')
        if not np.isfinite(reflection_phase_deg): raise ValueError('Reflection phase must be finite')
        start = math.radians(float(reflection_phase_deg))
        width = np.pi/int(rotation_order)
        boundary_angles = np.linspace(start, start+width, n_boundary)
        anchor_angles = np.linspace(start, start+width, 4)
        profile = {'rotation_order': int(rotation_order), 'meridian_reflection': True,
            'meridian_plane_deg': float(reflection_phase_deg), 'domain_area_fraction': 1/(2*int(rotation_order)),
            'requires_verified_design': True, 'speedup_guaranteed': False}
    radius = diameter_mm/2
    unit = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n_area))))[:n_area]
    rho, angle = radius*np.sqrt(unit[:, 0]), start+width*unit[:, 1]
    area = np.column_stack((rho*np.cos(angle), rho*np.sin(angle), np.full(n_area, z_mm)))
    boundary = np.column_stack((radius*np.cos(boundary_angles), radius*np.sin(boundary_angles), np.full(n_boundary, z_mm)))
    anchors = np.column_stack((radius*np.cos(anchor_angles), radius*np.sin(anchor_angles), np.full(4, z_mm)))
    points = np.vstack(([0., 0., z_mm], area, boundary, anchors))
    return {'points_mm': points, 'weights': np.r_[0., np.full(n_area, 1/n_area), np.zeros(n_boundary+4)],
        'geometry': 'disk', 'measure': 'area', 'diameter_mm': diameter_mm, 'z_mm': z_mm,
        'n_area': n_area, 'n_boundary': n_boundary, 'seed': seed,
        'symmetry_profile': profile, 'scope': 'fixed-z planar circular FOV; area statistics',
        'name': f'disk_D{diameter_mm:g}_z{z_mm:g}_A{n_area}_C{n_boundary}_seed{seed}'}

def cylinder_samples(diameter_mm=140.,length_mm=100.,n_volume=512,seed=20261029):
    unit=qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n_volume))))[:n_volume]
    r=diameter_mm/2*np.sqrt(unit[:,0]);phi=2*np.pi*unit[:,1]
    points=np.column_stack((r*np.cos(phi),r*np.sin(phi),length_mm*(unit[:,2]-.5)))
    angles=np.linspace(0,2*np.pi,48,endpoint=False);z=np.linspace(-length_mm/2,length_mm/2,9)
    zz,aa=np.meshgrid(z,angles)
    side=np.column_stack((diameter_mm/2*np.cos(aa.ravel()),diameter_mm/2*np.sin(aa.ravel()),zz.ravel()))
    # Include both full end faces as well as the cylindrical wall in extrema.
    rr,aa=np.meshgrid(np.linspace(0,diameter_mm/2,9),angles)
    caps=np.vstack([np.column_stack((rr.ravel()*np.cos(aa.ravel()),rr.ravel()*np.sin(aa.ravel()),np.full(rr.size,sgn*length_mm/2))) for sgn in (-1,1)])
    allpoints=np.vstack((np.zeros((1,3)),points,side,caps))
    return {"points_mm":allpoints,"weights":np.r_[0.,np.full(n_volume,1/n_volume),np.zeros(len(side)+len(caps))],
            "name":f"cylinder_D{diameter_mm:g}_L{length_mm:g}","diameter_mm":diameter_mm,"length_mm":length_mm}

def line_points(axis,half_length_mm=90.,n=151):
    coordinate=np.linspace(-half_length_mm,half_length_mm,n);points=np.zeros((n,3));points[:,"xyz".index(axis)]=coordinate
    return coordinate,points

def plane_points(plane,half_size_mm=90.,n=41):
    axis=np.linspace(-half_size_mm,half_size_mm,n);u,v=np.meshgrid(axis,axis)
    points=np.zeros((n,n,3));points[:,:,"xyz".index(plane[0])]=u;points[:,:,"xyz".index(plane[1])]=v
    mask=u*u+v*v<=half_size_mm**2+1e-9
    return axis,axis,points,mask

"""Field-component harmonic diagnostics, separate from |B| homogeneity.

In a source-free ball Bz is harmonic. We project Bz on a spherical surface,
not the non-harmonic magnitude |B| and not a one-dimensional axis fit.
Real tesseral surface functions have unit RMS on the sphere, include the
Condon--Shortley phase, and extend inward with (r/R)**l. A separate sampled
azimuthal Fourier diagnostic resolves the array's first two tooth periods.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, lpmv


def sphere_quadrature(radius_mm, lmax=8, n_phi=None, n_theta=None):
    """Full sphere product quadrature; weights sum to one (surface average)."""
    radius = float(radius_mm)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Harmonic reference radius must be positive and finite')
    if isinstance(lmax, bool) or int(lmax) != lmax or not 0 <= lmax <= 16:
        raise ValueError('lmax must be an integer between 0 and 16')
    n_phi = max(2*int(lmax)+3, int(n_phi or 65))
    n_theta = max(int(lmax)+2, int(n_theta or 16))
    u, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.arange(n_phi)*2*np.pi/n_phi
    uu, pp = np.meshgrid(u, phi, indexing='ij')
    rr = np.sqrt(np.maximum(0., 1-uu*uu))
    points = radius*np.c_[ (rr*np.cos(pp)).ravel(),
                          (rr*np.sin(pp)).ravel(), uu.ravel() ]
    weights = np.repeat(w/(2*n_phi), n_phi)
    return points, weights, {'n_theta': n_theta, 'n_phi': n_phi}


def _real_surface_basis(points_mm, radius_mm, lmax):
    points = np.asarray(points_mm, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    u = np.clip(points[:, 2]/radii, -1., 1.)
    phi = np.arctan2(points[:, 1], points[:, 0])
    terms, columns = [], []
    for degree in range(lmax+1):
        for order in range(degree+1):
            normal = math.sqrt((2*degree+1)*math.exp(
                float(gammaln(degree-order+1)-gammaln(degree+order+1))))
            radial = (radii/radius_mm)**degree
            angular = normal*lpmv(order, degree, u)*radial
            if order == 0:
                terms.append((degree, order, 'zonal'))
                columns.append(angular)
            else:
                for component, trig in (('cos', np.cos), ('sin', np.sin)):
                    terms.append((degree, order, component))
                    columns.append(math.sqrt(2.)*angular*trig(order*phi))
    return np.column_stack(columns), terms


def fit_spherical_bz(points_mm, fields_T, radius_mm, weights, lmax=8,
                     reference_T=None):
    """Weighted spherical projection with explicit normalization and residual.

    Coefficient ppm is the RMS amplitude of each unit-RMS real basis term.
    For zonal terms pole_ppm is the coefficient of P_l(cos theta), the usual
    Z_l boundary-at-pole convention. No total |B| ppm is inferred from it.
    """
    if isinstance(lmax, bool) or int(lmax) != lmax or not 0 <= lmax <= 16:
        raise ValueError('lmax must be an integer between 0 and 16')
    lmax = int(lmax)
    radius = float(radius_mm)
    points = np.asarray(points_mm, dtype=float)
    fields = np.asarray(fields_T, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or fields.shape != points.shape
            or len(points) < (lmax+1)**2 or w.shape != (len(points),)
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(fields))
            or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid spherical field samples or quadrature weights')
    if not np.allclose(np.linalg.norm(points, axis=1), radius, atol=1e-7, rtol=1e-9):
        raise ValueError('Spherical decomposition requires points on its reference sphere')
    w = w/w.sum()
    mean_bz = float(w@fields[:, 2])
    reference = mean_bz if reference_T is None else float(reference_T)
    if not math.isfinite(reference) or abs(reference) < 1e-12:
        raise ValueError('Reference Bz is too close to zero for harmonic ppm')
    basis, terms = _real_surface_basis(points, radius, lmax)
    weighted = basis*np.sqrt(w[:, None])
    # A direct weighted fit remains valid for well-resolved non-product point
    # sets too. Singular point sets must not masquerade as a spectrum.
    target = fields[:, 2]*np.sqrt(w)
    coefficient, _, rank, singular = np.linalg.lstsq(weighted, target, rcond=None)
    if rank != len(terms):
        raise ValueError('Spherical samples do not resolve the requested harmonic spectrum')
    residual = fields[:, 2]-basis@coefficient
    rows = []
    for value, (degree, order, component) in zip(coefficient, terms):
        row = {'degree': degree, 'order': order, 'component': component,
               'coefficient_T': float(value),
               'coefficient_ppm': float(value/abs(reference)*1e6)}
        if order == 0:
            row.update(label='B0' if degree == 0 else 'Z%d'%degree,
                       pole_ppm=float(value*math.sqrt(2*degree+1)/abs(reference)*1e6))
        else:
            row['label'] = 'l%d_m%d_%s'%(degree, order, component)
        rows.append(row)
    zonal = {str(row['degree']): row['pole_ppm'] for row in rows
             if row['order'] == 0 and row['degree'] > 0}
    by_degree = [{'degree': degree, 'rms_ppm': float(math.sqrt(sum(
        row['coefficient_ppm']**2 for row in rows if row['degree'] == degree)))}
        for degree in range(1, lmax+1)]
    return {'available': True, 'field_component': 'Bz', 'reference_radius_mm': radius,
            'reference_Bz_T': reference, 'surface_mean_Bz_T': mean_bz,
            'lmax': lmax, 'sample_count': len(points), 'fit_rank': int(rank),
            'condition_number': float(singular[0]/singular[-1]),
            'normalization': 'real tesseral functions RMS=1 on sphere; Condon-Shortley phase; interior factor (r/R)^l',
            'zonal_convention': 'Z_l is the coefficient of P_l(cos(theta)) at r=R, in ppm of abs(reference_Bz)',
            'coefficients': rows, 'zonal_ppm': zonal,
            'zonal': [{'degree': degree, 'label': 'Z%d'%degree,
                       'coefficient_ppm': zonal[str(degree)]}
                      for degree in (2, 4, 6, 8) if str(degree) in zonal],
            'degree_rms_ppm': by_degree,
            'residual_rms_ppm': float(np.sqrt(w@(residual*residual))/abs(reference)*1e6),
            'sampled_residual_peak_to_peak_ppm': float(np.ptp(residual)/abs(reference)*1e6),
            'scope': 'truncated Bz component decomposition; finite quadrature and residual do not certify global |B| extrema'}


def azimuthal_fourier(phi_rad, bz_T, reference_T=None, expected_orders=()):
    """Fourier amplitudes for uniformly spaced samples on one circle.

    amplitude_ppm is the peak sinusoidal amplitude; a pure term has twice
    this peak-to-peak range. It is not a spherical-harmonic coefficient.
    """
    phi = np.asarray(phi_rad, dtype=float)
    values = np.asarray(bz_T, dtype=float)
    if (phi.ndim != 1 or values.shape != phi.shape or len(phi) < 8
            or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(values))
            or not np.allclose(np.diff(phi), 2*np.pi/len(phi), atol=1e-12, rtol=0)):
        raise ValueError('Fourier samples must be finite and uniformly cover one circle without duplicate endpoint')
    mean = float(values.mean())
    reference = mean if reference_T is None else float(reference_T)
    if not math.isfinite(reference) or abs(reference) < 1e-12:
        raise ValueError('Reference Bz is too close to zero for Fourier ppm')
    modes = np.fft.rfft(values-mean)/len(values)
    modes *= np.exp(-1j*np.arange(len(modes))*phi[0])
    rows = []
    for order in range(1, len(modes)):
        factor = 1. if len(values)%2 == 0 and order == len(values)//2 else 2.
        rows.append({'order': order, 'cosine_ppm': float(factor*modes[order].real/abs(reference)*1e6),
                     'sine_ppm': float(-factor*modes[order].imag/abs(reference)*1e6),
                     'amplitude_ppm': float(factor*abs(modes[order])/abs(reference)*1e6),
                     'expected_array_order': order in expected_orders})
    return {'available': True, 'field_component': 'Bz', 'reference_Bz_T': reference,
            'sample_count': len(phi), 'maximum_resolved_order': len(values)//2,
            'normalization': 'peak sine/cosine amplitude / abs(reference_Bz) * 1e6; pure-term peak-to-peak is twice amplitude',
            'modes': rows, 'array_orders': [row for row in rows if row['expected_array_order']],
            'terms': [row for row in rows if row['expected_array_order']],
            'leading_modes': sorted(rows, key=lambda row: row['amplitude_ppm'], reverse=True)[:8],
            'scope': 'sampled equatorial-circle azimuthal Fourier spectrum, not full 3D spherical harmonics; higher unresolvable orders may alias'}


def analyze_harmonics(design, engine, config, *, lmax=8, reference_T=None,
                      output_dir=None):
    """Analyze the active design; disk domains never claim a 3D decomposition."""
    target = config.get('target', config)
    geometry = target.get('geometry', 'sphere')
    radius = float(target['diameter_mm'])/2
    z = float(target.get('z_mm', 0.))
    if geometry not in ('sphere', 'disk') or not math.isfinite(radius) or radius <= 0 or not math.isfinite(z):
        raise ValueError('Harmonic analysis needs a positive sphere or disk target')
    counts = sorted({int(r.count) for r in design.rings})
    if not counts or counts[-1] > 512:
        raise ValueError('Harmonic diagnostic column counts must be in 1..512')
    if geometry == 'sphere' and z != 0.:
        raise ValueError('Spherical harmonics currently use the origin-centered DSV')
    if geometry == 'sphere':
        nearest_source = min(math.hypot(max(0., r.radius_mm-design.cube_mm/2),
            max(0., abs(r.z_mm)-design.cube_mm/2)) for r in design.rings)
        if radius >= nearest_source-1e-8:
            raise ValueError('Spherical harmonic reference ball must contain no magnetic sources')
    expected_orders = sorted(set(counts+[2*c for c in counts]))
    n_circle = max(97, 4*max(counts)+1)
    phi = np.arange(n_circle)*2*np.pi/n_circle
    circle = np.c_[radius*np.cos(phi), radius*np.sin(phi), np.full(n_circle, z)]
    if geometry == 'sphere':
        points, weights, resolution = sphere_quadrature(radius, lmax,
            # Resolve both N and 2N in the full-sphere data too; otherwise,
            # e.g. 36 columns sampled at 73 angles aliases m=72 into m=1.
            n_phi=max(65, 4*max(counts)+1))
        field = np.asarray(engine.field(design, points), dtype=float)
        spherical = fit_spherical_bz(points, field, radius, weights, lmax, reference_T)
        spherical['quadrature'] = resolution
        reference = spherical['reference_Bz_T']
    else:
        spherical = {'available': False,
                     'reason': '平面 FOV 不是球形 DSV；仅凭圆盘场不能给出三维球谐分解。'}
        # Use the same area-weighted disk convention as the active workbench.
        from .sampling import disk_samples
        sample = disk_samples(2*radius, 128, 192, int(config.get('seed', 0)), z_mm=z)
        field = np.asarray(engine.field(design, sample['points_mm']), dtype=float)
        reference = float(np.asarray(sample['weights'])@field[:, 2]) if reference_T is None else float(reference_T)
    circle_field = np.asarray(engine.field(design, circle), dtype=float)
    fourier = azimuthal_fourier(phi, circle_field[:, 2], reference, expected_orders)
    fourier.update(radius_mm=radius, z_mm=z, array_column_counts=counts)
    result = {'schema': 'bz-harmonic-diagnostics-v1', 'geometry': geometry,
              'diameter_mm': 2*radius, 'polarization_model': engine.polarization_model,
              'spherical': spherical, 'azimuthal': fourier,
              'not_a_homogeneity_metric': '本报告分析 Bz 分量来源，不替代 |B| 峰峰值密集复核。'}
    if output_dir is not None:
        from .serialization import save_json
        folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
        save_json(folder/'harmonics.json', result)
        result['plot'] = str(plot_harmonics(result, folder/'harmonics.png'))
    return result


def plot_harmonics(report, path):
    """Standalone diagnostic artifact with explicitly separate coefficient units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    spherical = report['spherical']
    if spherical['available']:
        orders = (2, 4, 6, 8)
        axes[0].bar(['Z%d'%n for n in orders], [spherical['zonal_ppm'].get(str(n), 0.) for n in orders], color='#3173ae')
        axes[0].set_ylabel('Bz zonal pole coefficient / ppm')
        axes[0].set_title('Sphere r=%.1f mm; residual RMS %.3g ppm'%(
            spherical['reference_radius_mm'], spherical['residual_rms_ppm']))
    else:
        axes[0].text(.5, .5, 'Planar FOV\n3D spherical spectrum unavailable', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_axis_off()
    rows = report['azimuthal']['array_orders']
    axes[1].bar([str(row['order']) for row in rows], [row['amplitude_ppm'] for row in rows], color='#258879')
    axes[1].set_xlabel('Azimuthal order m (array N and 2N)')
    axes[1].set_ylabel('Circle Bz peak amplitude / ppm')
    axes[1].set_title('Separate azimuthal tooth diagnostic')
    for axis in axes:
        axis.axhline(0, color='#8090a0', lw=.7)
        axis.grid(axis='y', alpha=.2)
    figure.savefig(path, dpi=150); plt.close(figure)
    return path

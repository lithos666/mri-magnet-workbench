"""Constraint-only, deterministic ring synthesis using analytical unit fields.

No source-layout reader is used here. Binary variables mean whole mirrored
ring pairs, never fractional magnets. The MILP is a vector-field surrogate;
all returned designs are re-evaluated by the exact cuboid engine and audited.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .field import FieldEngine
from .geometry import Design, RingSpec
from .identity import canonical_physical_sha256
from .manufacturing import audit_manufacturing, radial_packing_bounds, axial_packing_bounds, column_partition_check
from .metrics import evaluate_design
from .objectives import check_performance, feasibility_score
from .projection import project_ordered
from .serialization import save_json


MIRROR_POLARITY = {"out": "in", "in": "out", "down": "down", "up": "up"}
STYLES = ("compact", "wide", "counterphase", "end_compensated", "hybrid")
_BASIS_VERSION = "mirrored-cuboid-unit-Br-v3-axis-through-Z8"


def _axis_features(values, t):
    """Even axis-polynomial coefficients, fitted in a conditioned basis.

    This is a 1D synthesis surrogate, not the full 3D spherical spectrum.
    Legendre fitting improves conditioning; returned coefficients multiply
    powers 1,t^2,t^4,t^6,t^8 with t=z/(target_diameter/2).
    """
    degrees = (0, 2, 4, 6, 8)
    matrix = np.polynomial.legendre.legvander(t, 8)[:, degrees]
    converted = np.zeros((5, 5))
    for j, degree in enumerate(degrees):
        unit = np.zeros(degree+1); unit[degree] = 1.
        monomial = np.polynomial.legendre.leg2poly(unit)
        converted[:len(monomial[::2]), j] = monomial[::2]
    fit = converted @ np.linalg.pinv(matrix)
    return fit @ values, float(np.linalg.cond(matrix))


def _axis_order_weights(config, available_orders):
    options = config.get('de_novo', {})
    default_order = 8 if config.get('target', {}).get('geometry', 'sphere') == 'sphere' else 0
    maximum = options.get('harmonic_max_order', default_order)
    if isinstance(maximum, bool) or maximum not in (0, 2, 4, 6, 8):
        raise ValueError('harmonic_max_order must be one of 0,2,4,6,8')
    configured = options.get('harmonic_order_weights', {})
    if not isinstance(configured, dict) or any(str(key) not in ('2', '4', '6', '8') for key in configured):
        raise ValueError('harmonic_order_weights must map orders 2,4,6,8 to nonnegative weights')
    default = {2: 1., 4: 1., 6: .25, 8: .1}
    result = {}
    for degree in (2, 4, 6, 8):
        weight = float(configured.get(str(degree), configured.get(degree, default[degree])))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('harmonic_order_weights must be finite and nonnegative')
        if degree <= maximum and degree in available_orders and weight > 0:
            result[degree] = weight
    return result


def _radial_lattice(packing, style, requested=None):
    """Style-specific legal base packing plus mutually exclusive alternatives.

    Six *candidate* radii need not fit simultaneously. Extra radii are legal
    alternatives to a base radius; their overlap graph is constrained by the
    inverse model rather than by silently shifting a selected configuration.
    """
    maximum = int(packing['maximum_radial_rings'])
    count = maximum if requested is None else _positive_integer(requested, 'radial_candidate_count')
    if count > max(64, 4*maximum):
        raise ValueError('radial_candidate_count is too large for bounded synthesis')
    base_count = min(count, maximum)
    lower = float(packing['minimum_magnet_center_radius_mm'])
    upper = float(packing['maximum_magnet_center_radius_mm'])
    pitch = float(packing['required_radial_center_pitch_mm'])
    slack = max(0., upper-lower-(base_count-1)*pitch)
    u = np.linspace(0., 1., base_count)
    profiles = {
        'compact': .04+.28*u**1.2,
        'wide': u,
        'counterphase': .03+.94*u**2,
        'end_compensated': .68+.28*u**.8,
        'hybrid': .02+.96*(.5-.5*np.cos(np.pi*u)),
    }
    base = lower+np.arange(base_count)*pitch+slack*profiles[style]
    base = project_ordered(base, lower, upper, pitch)
    result = list(base)
    # Retain a maximum-cardinality legal base while adding deterministic
    # alternatives. Reject impossible requests for a zero-free-span packing.
    for _ in range(count-base_count):
        candidates=[]
        for i,value in enumerate(base):
            lo=lower if i==0 else base[i-1]+pitch
            hi=upper if i==base_count-1 else base[i+1]-pitch
            for endpoint in (lo,hi):
                trial=(value+endpoint)/2
                distance=min(abs(trial-r) for r in result)
                candidates.append((distance,trial))
        distance,trial=max(candidates)
        if distance < 1e-7:
            raise ValueError('No distinct alternate radial candidate fits the current envelope')
        result.append(float(trial))
    return np.asarray(sorted(result))


def limits_from_config(config):
    """Explicit envelope conversion with no dependency on a prior geometry."""
    source = config["envelope"]
    if "bore_radius_mm" in source:
        result = dict(source)
    else:
        result = {
            "bore_radius_mm": float(source["protected_bore_diameter_mm"]) / 2,
            "outer_radius_mm": float(source["max_outer_diameter_mm"]) / 2,
            "half_length_mm": float(source["max_length_mm"]) / 2,
        }
    result.setdefault("min_gap_mm", 0.)
    result.setdefault("tolerance_mm", 1e-6)
    result["preserve_exact_bore"] = False
    if (not all(math.isfinite(float(result[k])) for k in
                ("bore_radius_mm", "outer_radius_mm", "half_length_mm")) or
            result["bore_radius_mm"] <= 0 or
            result["outer_radius_mm"] <= result["bore_radius_mm"] or
            result["half_length_mm"] <= 0):
        raise ValueError("Invalid de novo bore/outer/axial envelope")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _ring_count_limits(config, physical_maximum):
    search = config.get('search', {})
    minimum = _positive_integer(search.get('min_radial_rings_per_layer', 1), 'min_radial_rings_per_layer')
    requested_maximum = search.get('max_radial_rings_per_layer')
    maximum = int(physical_maximum) if requested_maximum is None else min(
        int(physical_maximum), _positive_integer(requested_maximum, 'max_radial_rings_per_layer'))
    if minimum > maximum:
        raise ValueError('Requested minimum radial rings exceeds physical/configured capacity')
    return minimum, maximum


def _magnet(config):
    data = dict(config["magnet"])
    for name, default in (("cube_mm", 12.), ("remanence_T", 1.43), ("reference_mur", 1.05)):
        data[name] = float(data.get(name, default))
        if not math.isfinite(data[name]) or data[name] <= 0:
            raise ValueError(name + " must be positive and finite")
    return data


def _layout_lattice(config, pair_count, column_count, style):
    pair_count = _positive_integer(pair_count, "pair_count")
    column_count = _positive_integer(column_count, "column_count")
    if column_count < 3:
        raise ValueError("A ring needs at least three columns")
    if style not in STYLES:
        raise ValueError("Unknown synthesis style: " + style)
    process = config["manufacturing"]
    partition=column_partition_check(column_count,process)
    if not partition['valid']:
        raise ValueError('; '.join(partition['violations']))
    material = _magnet(config)
    limits = limits_from_config(config)
    packing = radial_packing_bounds(material["cube_mm"], process, limits)
    if packing["maximum_radial_rings"] < 1:
        raise ValueError("No radial ring fits the manufacturing envelope")
    printing = process.get("enabled", True)
    pocket = packing["pocket_edge_mm"]
    base = float(process["base_thickness_mm"]) if printing else 0.
    gap = float(process["min_axial_module_gap_mm"]) if printing else float(limits["min_gap_mm"])
    axial = axial_packing_bounds(material['cube_mm'], process, limits)
    z_low = axial['minimum_abs_magnet_center_z_mm']
    z_high = axial['maximum_abs_magnet_center_z_mm']
    separation = axial['minimum_same_side_center_pitch_mm']
    if z_low + (pair_count - 1) * separation > z_high + 1e-9:
        raise ValueError("Requested mirrored layer pairs do not fit the axial envelope")
    minimum_span = (pair_count - 1) * separation
    # Styles are geometrically different attraction-basin proposals. All use
    # the same legal pitch; no random radius or polarity initialization occurs.
    span = z_high - z_low
    fractions = {"compact": .63, "wide": 1., "counterphase": .84,
                 "end_compensated": .96, "hybrid": .75}
    endpoint = z_low + max(minimum_span, span * fractions[style])
    unit = np.linspace(0., 1., pair_count)
    exponent = {"compact": 1.05, "wide": .85, "counterphase": 1.3,
                "end_compensated": .7, "hybrid": 1.15}[style]
    raw = z_low + (endpoint-z_low) * unit ** exponent
    z_values = project_ordered(raw, z_low, z_high, separation)
    _, maximum = _ring_count_limits(config, packing['maximum_radial_rings'])
    radii = _radial_lattice(dict(packing, maximum_radial_rings=maximum), style,
        config.get('de_novo',{}).get('radial_candidate_count'))
    return material, limits, packing, radii, z_values


def _pair_design(material, pair_count, column_count, descriptor, unit=True):
    k, j = descriptor["pair_index"], descriptor["radial_index"]
    # Positive layers are indexed outer-to-inner, followed by their mirrors.
    positive_layer = pair_count - 1 - k
    negative_layer = 2 * pair_count - 1 - positive_layer
    phase = -90. + 180. / column_count
    p = descriptor["polarity"]
    rings = (
        RingSpec("R%02d-L%d" % (positive_layer+1, j+1), positive_layer, j,
                 descriptor["radius_mm"], descriptor["z_mm"], column_count, p, phase),
        RingSpec("R%02d-L%d" % (negative_layer+1, j+1), negative_layer, j,
                 descriptor["radius_mm"], -descriptor["z_mm"], column_count, MIRROR_POLARITY[p], phase),
    )
    return Design(rings, cube_mm=material["cube_mm"],
                  remanence_T=1. if unit else material["remanence_T"],
                  reference_mur=material["reference_mur"], name="de_novo_basis_pair")


def build_basis(config, sample, pair_count, column_count, style="wide", engine=None, cache_dir=None):
    """Return vector contribution matrix per 1 T Br, with exact cache identity.

    ``A_T_per_T`` has shape (training points, xyz, basis ring pairs). Axis
    features are independent of training quadrature and carry no mean weight.
    Opposite polarities are exact negatives, so only two physical fields per
    grid cell are evaluated. Mirrored geometry, phase, material response model
    and every observer coordinate are included in the cache fingerprint.
    """
    pair_count = _positive_integer(pair_count, "pair_count")
    column_count = _positive_integer(column_count, "column_count")
    material, limits, packing, radii, z_values = _layout_lattice(
        config, pair_count, column_count, style)
    points = np.asarray(sample["points_mm"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid synthesis sample points")
    weights=np.asarray(sample.get('weights'),dtype=float)
    if weights.shape!=(len(points),) or not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Synthesis requires nonnegative quadrature weights with positive sum')
    weights=weights/weights.sum()
    diameter = float(config["target"]["diameter_mm"])
    z_center = float(config["target"].get("z_mm", 0.))
    if not math.isfinite(diameter) or diameter <= 0 or not math.isfinite(z_center):
        raise ValueError("Synthesis domain diameter must be positive and center finite")
    # Twenty-one symmetric Lobatto nodes resolve through eighth order without
    # relying on the old nine-point, four-term fit. Endpoints stay explicit.
    axis_t = np.cos(np.linspace(np.pi, 0., 21))
    axis_z = axis_t*diameter/2
    feature_points = np.c_[np.zeros(len(axis_t)), np.zeros(len(axis_t)), axis_z + z_center]
    observers = np.vstack((points, feature_points, [0., 0., z_center]))
    descriptors = []
    for k, z in enumerate(z_values):
        for j, radius in enumerate(radii):
            for p in ("out", "in", "down", "up"):
                descriptors.append({"pair_index": k, "radial_index": j,
                                    "radius_mm": float(radius), "z_mm": float(z), "polarity": p})
    mode = config.get("polarization_model", "remanence")
    if engine is not None and getattr(engine, "polarization_model", mode) != mode:
        raise ValueError("Basis engine polarization model differs from configured physics")
    identity = {"version": _BASIS_VERSION, "cube_mm": material["cube_mm"],
                "reference_mur": material["reference_mur"], "polarization_model": mode,
                "column_count": column_count, "phase_deg": -90.+180./column_count,
                "descriptors": descriptors, "observers_mm": observers.tolist(),
                "quadrature_weights":weights.tolist()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir)/("basis_"+digest+".npz") if cache_dir else None
    cached = False
    values = None
    if cache_path and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) == digest:
                candidate = archive["field"]
                if candidate.shape == (len(observers), 3, len(descriptors)) and np.all(np.isfinite(candidate)):
                    values = candidate
                    cached = True
    if values is None:
        engine = engine or FieldEngine(config.get("chunk_points", 96), mode)
        values = np.empty((len(observers), 3, len(descriptors)))
        for start in range(0, len(descriptors), 4):
            for offset in (0, 2):
                value = engine.field(_pair_design(material, pair_count, column_count,
                                                  descriptors[start+offset]), observers)
                values[:, :, start+offset] = value
                values[:, :, start+offset+1] = -value
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, fingerprint=np.array(digest), field=values)
            temporary.replace(cache_path)
    # Axis features and the full spherical diagnostic have distinct purposes.
    coeff, axis_condition = _axis_features(values[len(points):len(points)+len(axis_t), 2, :], axis_t)
    return {"A_T_per_T": values[:len(points)], "axis_A_T_per_T": values[len(points):len(points)+len(axis_t)],
            "center_A_T_per_T": values[-1], "mean_Az_T_per_T":weights@values[:len(points),2,:],
            "quadrature_weights":weights,"pointset":sample.get('name'),
            "axis_coefficients_T_per_T": coeff, 'axis_coefficient_orders': [0, 2, 4, 6, 8],
            'axis_fit_condition_number': axis_condition,
            "axis_feature_points_mm": feature_points, "basis": descriptors,
            "fingerprint": digest, "cache_hit": cached, "unit_remanence_T": 1.,
            "pair_count": pair_count, "column_count": column_count, "style": style,
            "packing": packing, "limits": limits, "radii_mm": radii,
            "positive_z_mm": z_values, "material": material}


def _solve_basis(basis, config):
    """Exact binary linear-fractional Chebyshev surrogate.

    Charnes-Cooper introduces tau=1/b, y=tau*x. Four linear envelopes per
    binary x make the product exact; no t*b bilinear term is mislabelled MILP.
    b is the *quadrature mean Bz* normalized by Bmin, not the sphere centre
    sample and not the nonlinear mean |B|. Positive mean Bz is a sufficient
    lower bound on mean |B|; final performance is always computed separately.
    """
    options = config.get("de_novo", {})
    target = config["target"]
    minimum = float(target["min_field_mT"]) * 1e-3
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("De novo synthesis needs a finite positive B_min")
    remanence = basis["material"]["remanence_T"]
    scale = remanence / minimum
    mean = np.asarray(basis['mean_Az_T_per_T'])*scale
    count = len(mean)
    transverse_weight=float(options.get('transverse_weight',.05))
    margin=float(options.get('field_margin_ratio',.005))
    if not math.isfinite(transverse_weight) or not 0<=transverse_weight<=1:
        raise ValueError('transverse_weight must be in [0,1]')
    if not math.isfinite(margin) or not 0<=margin<1:
        raise ValueError('field_margin_ratio must be in [0,1)')
    sparse = float(options.get("sparsity_weight", 1e-9))
    if not math.isfinite(sparse) or sparse < 0:
        raise ValueError("sparsity_weight must be finite and nonnegative")
    maximum = _positive_integer(config['manufacturing']['max_magnet_count'], 'max_magnet_count')
    maximum_pairs = maximum // (2*basis['column_count'])
    upper_b = float(np.sort(np.maximum(mean,0.))[-maximum_pairs:].sum()) if maximum_pairs else 0.
    lower_b=1.+margin
    if upper_b<lower_b:
        return None,{'solver_status':2,'solver_message':'Weighted mean Bz necessary bound is below the requested headroom',
                     'necessary_mean_Bz_upper_bound_T':upper_b*minimum,'minimum_mean_Bz_T':lower_b*minimum}
    tau_lower,tau_upper=1./upper_b,1./lower_b
    # Binary occupancies x, continuous y=x*tau, tau, and relative error t.
    size=2*count+2;tau_index=2*count;error_index=tau_index+1
    objective=np.zeros(size);objective[error_index]=1.;objective[:count]=sparse
    rows, lower, upper = [], [], []
    def constraint(coefficients, lo=-np.inf, hi=np.inf):
        rows.append(coefficients); lower.append(lo); upper.append(hi)
    # At most one polarity per lattice cell; every selected layer pair occupied.
    for i in range(0, count, 4):
        row = np.zeros(size); row[i:i+4] = 1
        constraint(row, hi=1.)
    minimum_rings, maximum_rings = _ring_count_limits(config,
        basis['packing'].get('maximum_radial_rings', len(basis['radii_mm'])))
    for k in range(basis["pair_count"]):
        row = np.zeros(size)
        row[:count] = [d["pair_index"] == k for d in basis["basis"]]
        constraint(row, lo=float(minimum_rings), hi=float(maximum_rings))
    # Candidate radii can overlap. Selecting conflicting cells is forbidden;
    # extra candidate slots never bypass the actual physical packing bound.
    conflict_count=0
    pitch=basis['packing']['required_radial_center_pitch_mm']
    radii=basis['radii_mm'];nrad=len(radii)
    for k in range(basis['pair_count']):
        for i in range(nrad):
            for j in range(i+1,nrad):
                if radii[j]-radii[i]>=pitch-1e-8:continue
                row=np.zeros(size)
                row[(k*nrad+i)*4:(k*nrad+i+1)*4]=1.
                row[(k*nrad+j)*4:(k*nrad+j+1)*4]=1.
                constraint(row,hi=1.);conflict_count+=1
    row = np.zeros(size);row[:count]=1.
    constraint(row, hi=float(maximum_pairs))
    row=np.zeros(size);row[count:2*count]=mean
    constraint(row,lo=1.,hi=1.)
    for i in range(count):
        row=np.zeros(size);row[count+i]=1.;row[i]=-tau_lower
        constraint(row,lo=0.)
        row=np.zeros(size);row[count+i]=1.;row[i]=-tau_upper
        constraint(row,hi=0.)
        row=np.zeros(size);row[count+i]=1.;row[tau_index]=-1.;row[i]=-tau_upper
        constraint(row,lo=-tau_upper)
        row=np.zeros(size);row[count+i]=1.;row[tau_index]=-1.;row[i]=-tau_lower
        constraint(row,hi=-tau_lower)
    # Axial deviation is first order in |B|. Transverse components contribute
    # quadratically near a strong axial field and get a configurable lower
    # surrogate weight, while all components remain in exact final evaluation.
    vector_rows = basis["A_T_per_T"].reshape(-1, count)*scale
    for index, contribution in enumerate(vector_rows):
        axial=index%3==2
        weight=1. if axial else transverse_weight
        if weight==0:continue
        row=np.zeros(size);row[count:2*count]=contribution*weight;row[error_index]=-1.
        constraint(row,hi=1. if axial else 0.)
        opposite=-row;opposite[error_index]=-1.
        constraint(opposite,hi=-1. if axial else 0.)
    harmonic_weight = float(options.get("harmonic_weight", 2.))
    if not math.isfinite(harmonic_weight) or harmonic_weight < 0:
        raise ValueError("harmonic_weight must be finite and nonnegative")
    axis_coefficients = np.asarray(basis['axis_coefficients_T_per_T'])
    axis_orders = basis.get('axis_coefficient_orders', list(range(0, 2*len(axis_coefficients), 2)))
    order_weights = _axis_order_weights(config, axis_orders)
    for degree, coefficient in zip(axis_orders, axis_coefficients):
        if degree not in order_weights:
            continue
        row=np.zeros(size);row[count:2*count]=coefficient*scale*harmonic_weight*order_weights[degree];row[error_index]=-1.
        constraint(row, hi=0.)
        opposite = -row; opposite[error_index] = -1.
        constraint(opposite, hi=0.)
    # Template tie-breaking is physical and deliberately weak relative to
    # field matching. It does not assign random polarities.
    for index, item in enumerate(basis["basis"]):
        z, radius = item["z_mm"], item["radius_mm"]
        preferred = "down" if z < radius*.55 else "out"
        if basis["style"] == "counterphase" and item["radial_index"] == len(basis["radii_mm"])-1:
            preferred = "in" if z < radius*.55 else "up"
        objective[index] += sparse*.1*(item["polarity"] != preferred)
    bounds = Bounds(np.r_[np.zeros(2*count),tau_lower,0.],
                    np.r_[np.ones(count),np.full(count,tau_upper),tau_upper,np.inf])
    seconds = float(options.get("milp_time_limit_seconds", 2.))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("milp_time_limit_seconds must be positive")
    started=time.perf_counter()
    solved = milp(objective, integrality=np.r_[np.ones(count),np.zeros(count+2)], bounds=bounds,
                  constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": seconds, "mip_rel_gap": .02, "presolve": True})
    diagnostic = {"solver_status": int(solved.status), "solver_message": str(solved.message),
                  "surrogate_optimal": bool(solved.success), "binary_variables": count,
                  "maximum_pair_count_from_inventory": maximum_pairs,
                  'relative_normalization':'actual quadrature mean Bz via exact binary Charnes-Cooper products',
                  'mean_constraint':'positive weighted mean Bz; sufficient lower bound, not exact mean |B|',
                  'transverse_weight':transverse_weight,'field_margin_ratio':margin,
                  'radial_conflict_constraints':conflict_count,
                  'radial_rings_per_layer_range':[minimum_rings, maximum_rings],
                  'elapsed_solver_seconds':time.perf_counter()-started,'configured_solver_seconds':seconds,
                  "harmonic_feature": "1D normalized-axis even Bz polynomial; not full spherical harmonic decomposition",
                  'harmonic_orders': list(order_weights), 'harmonic_order_weights': order_weights,
                  'axis_fit_condition_number': basis.get('axis_fit_condition_number'),
                  "homogeneity_constraint": "weighted axial/transverse relative minimax surrogate; exact |B| ppm independently evaluated"}
    if solved.x is None:
        return None, diagnostic
    x = np.asarray(solved.x[:count])
    # Reject fractional or constraint-violating time-limit output explicitly.
    full = np.asarray(solved.x)
    product = np.asarray(rows) @ full
    if (np.max(np.abs(x-np.rint(x))) > 1e-5 or
            np.any(product < np.asarray(lower)-1e-5) or np.any(product > np.asarray(upper)+1e-5)):
        diagnostic["rejected"] = "MILP returned no independently verified integer incumbent"
        return None, diagnostic
    selected = np.flatnonzero(x > .5)
    rings = []
    for index in selected:
        rings.extend(_pair_design(basis["material"], basis["pair_count"], basis["column_count"],
                                  basis["basis"][index], unit=False).rings)
    # Explicit PAVA projection on every layer, with the very same bounds used
    # by downstream continuous parameterization and manufacturing audit.
    projected = []
    for layer in sorted({r.layer_index for r in rings}):
        group = sorted((r for r in rings if r.layer_index == layer), key=lambda r:r.radius_mm)
        packing = basis["packing"]
        radii = project_ordered([r.radius_mm for r in group],
                                packing["minimum_magnet_center_radius_mm"],
                                packing["maximum_magnet_center_radius_mm"],
                                packing["required_radial_center_pitch_mm"])
        if np.max(np.abs(radii-np.asarray([r.radius_mm for r in group])))>1e-6:
            diagnostic['rejected']='Selected lattice violates exact packing; refusing surrogate-invalidating PAVA displacement'
            return None,diagnostic
        projected.extend(replace(r, radius_mm=float(radius)) for r, radius in zip(group, radii))
    design = Design(tuple(projected), cube_mm=basis["material"]["cube_mm"],
                    remanence_T=remanence, reference_mur=basis["material"]["reference_mur"],
                    name="de_novo_"+basis["style"])
    diagnostic.update(selected_ring_pairs=len(selected), center_Bz_surrogate_T=float(basis['center_A_T_per_T'][2]@np.rint(x)*remanence),
                      mean_Bz_surrogate_T=float(mean@np.rint(x)*minimum),
                      actual_field_denominator_T=float(minimum/solved.x[tau_index]),
                      inverse_normalized_mean=float(solved.x[tau_index]),
                      minimax_vector_error_fraction=float(solved.x[-1]),
                      basis_fingerprint=basis["fingerprint"], pava_projection=True)
    return design, diagnostic


def synthesize(config, sample, engine=None, output_dir=None, log=print):
    """Generate 3--5 heterogeneous, manufacturing-legal constraint-only seeds.

    Small custom ``seed_count`` values are accepted for tests and exploratory
    runs. An empty/short cluster is reported honestly; arbitrary targets are
    not guaranteed feasible. Evaluation counts here are separate from DE.
    """
    started = time.perf_counter()
    _magnet(config)
    limits = limits_from_config(config)
    options = config.get("de_novo", {})
    requested = _positive_integer(options.get("seed_count", 4), "seed_count")
    if requested > 5:
        raise ValueError("seed_count must be at most five")
    pool_option=options.get('candidate_pool_size')
    pool_target = requested if pool_option is None else _positive_integer(pool_option, 'candidate_pool_size')
    if pool_target < requested:
        raise ValueError('candidate_pool_size must be at least seed_count')
    pair_counts = [_positive_integer(v, "pair_counts") for v in options.get("pair_counts", [5, 6, 7, 8])]
    default_columns = list(range(24,37,2)) if config['manufacturing'].get('allow_split_pocket_seams',False) else [24,28,32,36]
    columns = [_positive_integer(v, "column_counts") for v in options.get("column_counts", default_columns)]
    if not pair_counts or not columns:
        raise ValueError("Nonempty pair_counts and column_counts are required")
    if len(set(pair_counts)) != len(pair_counts) or len(set(columns)) != len(columns):
        raise ValueError("Duplicate pair/column alternatives")
    engine = engine or FieldEngine(config.get("chunk_points", 96), config.get("polarization_model", "remanence"))
    start_statistics = dict(engine.statistics)
    basis_cost = {"engine_calls": 0, "field_points": 0, "field_seconds": 0.}
    seed_cost = {"engine_calls": 0, "field_points": 0, "field_seconds": 0.}
    target = config["target"]
    maximum_ppm = target.get('max_ppm')
    requirements = {"min_mean_norm_T": float(target["min_field_mT"])*1e-3,
                    "max_pp_norm_ppm": None if maximum_ppm is None else float(maximum_ppm),
                    'scoring': dict(config.get('scoring', {})),
                    "field_tolerance_T": 0., "field_scale_T": max(float(target["min_field_mT"])*1e-3, 1e-12)}
    if (not math.isfinite(requirements["min_mean_norm_T"]) or requirements["min_mean_norm_T"] <= 0 or
            (requirements["max_pp_norm_ppm"] is not None and
             (not math.isfinite(requirements["max_pp_norm_ppm"]) or requirements["max_pp_norm_ppm"] <= 0))):
        raise ValueError("Positive finite target field and optional positive ppm ceiling are required")
    out = Path(output_dir).resolve() if output_dir else None
    cache = out/"basis_cache" if out else None
    max_attempts = _positive_integer(options.get("max_attempts", max(8, pool_target*2)), "max_attempts")
    candidate_pool, attempts, seen = [], [], set()
    # Cycling independently through layer counts, column counts, and styles
    # yields distinct physical architectures before any repeat is attempted.
    scenarios = []
    for cycle in range(max(max_attempts, max(len(pair_counts), len(columns), requested)*2)):
        pairs = pair_counts[cycle % len(pair_counts)]
        # Advancing by two when only one layer count is requested would
        # permanently omit every second column count (e.g. 28 and 36).
        col = columns[(cycle//len(pair_counts) + cycle%len(pair_counts)) % len(columns)]
        style = STYLES[cycle % len(STYLES)]
        key = (pairs, col, style)
        if key not in scenarios:
            scenarios.append(key)
    # Stratification can revisit a subset when cardinalities have common
    # factors. Append every missing architecture/style combination explicitly,
    # preserving the existing early-scenario order for saved workflows.
    cartesian = [(pairs,col,style) for pairs in pair_counts for col in columns for style in STYLES]
    for key in cartesian:
        if key not in scenarios:
            scenarios.append(key)
    for pairs, col, style in scenarios[:max_attempts]:
        item = {"pair_count": pairs, "column_count": col, "style": style}
        log("De Novo: %d layer pairs, %d columns, %s" % (pairs, col, style))
        try:
            before = dict(engine.statistics)
            basis = build_basis(config, sample, pairs, col, style, engine, cache)
            for name in basis_cost:
                basis_cost[name] += engine.statistics[name]-before[name]
            # A generous necessary weighted-mean Bz upper bound ignores
            # conflicts. It matches the actual inverse-model requirement.
            best = np.maximum(basis["mean_Az_T_per_T"], 0.) * basis["material"]["remanence_T"]
            capacity = config["manufacturing"]["max_magnet_count"] // (2*col)
            upper_field = float(np.sort(best)[-int(capacity):].sum()) if capacity else 0.
            item["necessary_weighted_mean_Bz_upper_bound_T"] = upper_field
            if capacity < pairs or upper_field < requirements["min_mean_norm_T"]-1e-12:
                item.update(status="discrete_target_infeasible", reason="inventory/weighted-mean Bz necessary bound")
                attempts.append(item)
                continue
            design, inverse = _solve_basis(basis, config)
            item["inverse"] = inverse
            if design is None:
                item["status"] = "no_integer_incumbent"
                attempts.append(item)
                continue
            audit = audit_manufacturing(design, config["manufacturing"], limits)
            if not audit["valid"]:
                item.update(status="manufacturing_rejected", violations=audit["violations"])
                attempts.append(item)
                continue
            physical = canonical_physical_sha256(design, engine.polarization_model)
            if physical in seen:
                item["status"] = "duplicate_physical_layout"
                attempts.append(item)
                continue
            before = dict(engine.statistics)
            metrics, _ = evaluate_design(design, engine, sample, config.get("density_kg_m3", 7500.))
            for name in seed_cost:
                seed_cost[name] += engine.statistics[name]-before[name]
            performance = check_performance(metrics, requirements)
            row = {"id": "synthesis_%02d" % (len(candidate_pool)+1), "style": style,
                   "pair_count": pairs, "column_count": col, "design": design.to_dict(),
                   "metrics": metrics, "audit": audit, "performance": performance,
                   "canonical_physical_sha256": physical, "inverse": inverse,
                   "geometry_valid": True, "performance_valid": performance["valid"],
                   "origin": "constraints_only_no_source_layout", "dense_verified": False}
            row['selection_score'] = float(feasibility_score(metrics, requirements))
            candidate_pool.append(row); seen.add(physical)
            item.update(status="accepted", candidate_id=row['id'], metrics=metrics, performance_valid=performance["valid"])
            attempts.append(item)
            if len(candidate_pool) >= pool_target:
                break
        except ValueError as error:
            item.update(status="invalid_configuration_or_lattice", reason=str(error))
            attempts.append(item)
    if pool_target == requested:
        # Existing profiles retain their exact first-legal cluster/order unless
        # the user explicitly requests a larger quality-screening pool.
        seeds = candidate_pool[:requested]
        selection_rule = 'first legal cluster (compatibility mode)'
    else:
        ordered = sorted(candidate_pool, key=lambda row: (row['selection_score'], row['id']))
        seeds, represented = [], set()
        for row in ordered:
            if row['style'] not in represented:
                seeds.append(row); represented.add(row['style'])
                if len(seeds) == requested:
                    break
        for row in ordered:
            if len(seeds) >= requested:
                break
            if not any(row is chosen for chosen in seeds):
                seeds.append(row)
        selection_rule = 'same declared feasibility/engineering score; best per style, then ranked fill'
    selected_ids = [row['id'] for row in seeds]
    for item in attempts:
        if item.get('candidate_id'):
            item['selected_for_deep_optimization'] = item['candidate_id'] in selected_ids
    examined = {(row['pair_count'], row['column_count'], row['style']) for row in attempts}
    result = {"seeds": seeds, 'candidate_pool': candidate_pool, "metadata": {
        "algorithm": "style-specific mirrored lattice + exact binary linear-fractional relative inverse + Z2/Z4 suppression",
        "source_layout_read": False, "source_layout_required": False,
        "global_optimum_proven": False, "basis_surrogate_is_exact_final_metrics": False,
        "basis_linear_superposition_exact_for_fixed_geometry": True,
        "basin_membership_proven": False,
        "coverage": {"requested_pair_counts": pair_counts, "requested_column_counts": columns,
                     "styles": list(STYLES), "examined_scenarios": len(attempts),
                     "available_architecture_style_scenarios": len(cartesian),
                     "full_cartesian_search_performed": set(cartesian) <= examined,
                     "selection": "deterministic stratified scenarios with missing Cartesian combinations appended; not exhaustive physical topology enumeration"},
        'requested_candidate_pool_size': pool_target, 'generated_candidate_pool_size': len(candidate_pool),
        'candidate_pool_status': 'complete' if len(candidate_pool) >= pool_target else 'partial',
        'seed_selection_rule': selection_rule, 'selected_seed_ids': selected_ids,
        'candidate_pool_metrics': [{'id':row['id'], 'style':row['style'], 'pair_count':row['pair_count'],
            'column_count':row['column_count'], 'metrics':row['metrics'], 'selection_score':row['selection_score'],
            'performance_valid':row['performance_valid'], 'selected':row['id'] in selected_ids}
            for row in candidate_pool],
        "requested_seeds": requested, "generated_seeds": len(seeds), "attempts": attempts,
        "status": "complete" if len(seeds) == requested else "partial_cluster" if seeds else "no_feasible_seed_found",
        "meaning_of_failure": "No legal integer incumbent found within the chosen lattices/time limits; not a proof of general physical infeasibility",
        "all_returned_seeds_manufacturing_valid": all(s["audit"]["valid"] for s in seeds),
        "performance_valid_seeds": sum(s["performance_valid"] for s in seeds),
        "seed_field_evaluations": len(candidate_pool), "basis_engine_statistics": basis_cost,
        "seed_evaluation_engine_statistics": seed_cost,
        "total_engine_statistics_delta": {name: engine.statistics[name]-start_statistics[name]
                                          for name in start_statistics},
        "requirements": requirements, "elapsed_seconds": time.perf_counter()-started,
        "independent_dense_validation_required": True}}
    if out:
        save_json(out/"synthesis_result.json", result)
    return result

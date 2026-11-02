"""Geometric checks for printed magnet-pocket or shared radial-channel modules.

All adjustable process values are supplied in ``manufacturing_config`` (the
application keeps them in parameters.py).  Passing these checks is NOT a
structural, magnetic-force, thermal, or manufacturing-process qualification.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
import math
from typing import Mapping

import numpy as np

from .geometry import Design, validate_design, planar_pocket_clearance
from .projection import project_ordered


_LENGTH_KEYS = (
    "pocket_clearance_per_side_mm", "pocket_web_mm", "inner_wall_mm",
    "outer_wall_mm", "base_thickness_mm", "bed_margin_per_side_mm",
    "min_axial_module_gap_mm",
)


def _configuration(config: Mapping) -> dict:
    required = {*_LENGTH_KEYS, "sectors_per_layer", "printer_bed_mm", "max_magnet_count"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Missing centralized manufacturing parameters: " + ", ".join(missing))
    result = dict(config)
    # Missing flags retain the saved, identical-sector manufacturing contract.
    result.setdefault('allow_split_pocket_seams', False)
    if not isinstance(result['allow_split_pocket_seams'], bool):
        raise ValueError('allow_split_pocket_seams must be a boolean')
    for key in ('radial_inner_clearance_mm', 'min_central_magnet_gap_mm'):
        result[key] = float(result.get(key, 0.))
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Invalid manufacturing length: ' + key)
    # Old saved configurations retain their original individual-pocket model.
    # New radial channels share one uninterrupted slot along each radial column.
    result.setdefault("layout_mode", "independent_pockets")
    if result["layout_mode"] not in ("independent_pockets", "radial_channels"):
        raise ValueError("layout_mode must be independent_pockets or radial_channels")
    for key in _LENGTH_KEYS:
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("Invalid manufacturing length: " + key)
    for key in ("pocket_web_mm", "inner_wall_mm", "outer_wall_mm", "base_thickness_mm"):
        if result.get("enabled", True) and result[key] <= 0:
            raise ValueError("A printed wall/base must have positive thickness: " + key)
    for key in ("sectors_per_layer", "max_magnet_count"):
        value = float(result[key])
        if not value.is_integer() or value <= 0:
            raise ValueError("A positive integer is required: " + key)
        result[key] = int(value)
    if result["sectors_per_layer"] < 2:
        raise ValueError("At least two sectors per layer are required")
    raw_bed = result['printer_bed_mm']
    if isinstance(raw_bed, Mapping):
        raw_bed = [raw_bed['bed_x_mm'], raw_bed['bed_y_mm'], raw_bed['bed_z_mm']]
    bed = [float(value) for value in raw_bed]
    if len(bed) != 3 or not all(math.isfinite(value) and value > 0 for value in bed):
        raise ValueError("printer_bed_mm requires three positive finite dimensions")
    result["printer_bed_mm"] = bed
    if result.get('machine_bed_mm') is not None:
        machine = [float(value) for value in result['machine_bed_mm']]
        if len(machine) != 3 or not all(math.isfinite(value) and value > 0 for value in machine):
            raise ValueError('machine_bed_mm requires three positive finite dimensions')
        result['machine_bed_mm'] = machine
    result.setdefault("bed_fit_mode", "fixed_envelope")
    if result["bed_fit_mode"] not in ("fixed_envelope", "candidate_envelope"):
        raise ValueError("bed_fit_mode must be fixed_envelope or candidate_envelope")
    return result


def column_partition_check(count, process: Mapping) -> dict:
    """Validate the declared seam policy without changing magnet geometry.

    Split seams allow even whole-ring counts; the assembled magnet slot can be
    made from two half-slots. This does not imply four identical quarter parts.
    """
    allowed = process.get('allow_split_pocket_seams', False)
    if not isinstance(allowed, bool):
        raise ValueError('allow_split_pocket_seams must be a boolean')
    sectors = process.get('sectors_per_layer', 4)
    if isinstance(sectors, bool) or not math.isfinite(float(sectors)) or int(sectors) != sectors or sectors < 2:
        raise ValueError('sectors_per_layer must be an integer >=2')
    sectors = int(sectors)
    valid_number = (not isinstance(count, bool) and isinstance(count, (int, float, np.integer, np.floating))
                    and math.isfinite(float(count)) and int(count) == count and count >= 3)
    number = int(count) if valid_number else None
    divisible = bool(valid_number and number % sectors == 0)
    printing = bool(process.get('enabled', True))
    valid = bool(valid_number)
    if printing and valid:
        valid = (number >= 4 and number % 2 == 0) if allowed else divisible
    violations = []
    if not valid_number:
        violations.append('column count must be an integer >=3')
    elif printing and allowed and (number < 4 or number % 2):
        violations.append('split pocket seams require an even column count >=4')
    elif printing and not allowed and not divisible:
        violations.append('azimuth columns must divide into equal identical printed sectors when split pocket seams are disabled')
    return {'valid': valid, 'count': number, 'sectors_per_layer': sectors,
            'allow_split_pocket_seams': allowed, 'divisible': divisible,
            'identical_sector_pattern': divisible, 'violations': violations,
            'rule': ('bare magnets; no printed-sector divisibility constraint' if not printing else
                     'even full-ring count; seam-crossing pockets allowed, quarter parts may differ'
                     if allowed else 'count divisible by sectors; identical rotated sector pattern')}


def _seam_partition_metadata(design: Design, pocket: float, process: Mapping, tolerance: float) -> dict:
    """Report actual pocket/seam intersections at the undeformed assembled layout."""
    sectors = int(process['sectors_per_layer'])
    angles = np.arange(sectors, dtype=float)*2*np.pi/sectors
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    normals = np.column_stack((-np.sin(angles), np.cos(angles)))
    local = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])*pocket/2
    split = []
    for ring in design.rings:
        theta = np.deg2rad(ring.phase_deg + np.arange(ring.count)*360/ring.count)
        radial = np.column_stack((np.cos(theta),np.sin(theta)))
        tangent = np.column_stack((-np.sin(theta),np.cos(theta)))
        vertices = (ring.radius_mm*radial[:,None,:] +
                    local[None,:,0,None]*radial[:,None,:] + local[None,:,1,None]*tangent[:,None,:])
        signed = np.einsum('nvi,si->nvs',vertices,normals)
        forward = np.einsum('nvi,si->nvs',vertices,rays)
        crossing = ((signed.min(axis=1) < -tolerance) & (signed.max(axis=1) > tolerance) &
                    (forward.max(axis=1) > tolerance))
        for column,seam in np.argwhere(crossing):
            split.append({'ring_id':ring.id,'layer_index':ring.layer_index,
                          'column_index':int(column),'seam_index':int(seam),
                          'seam_angle_deg':float(seam*360/sectors)})
    return {'allow_split_pocket_seams':bool(process.get('allow_split_pocket_seams',False)),
            'all_layer_patterns_rotationally_identical':all(r.count%sectors==0 for r in design.rings),
            'pockets_split_by_seams':split, 'split_pocket_count':len({(r['ring_id'],r['column_index']) for r in split}),
            'seam_intersection_count':len(split), 'seam_angle_origin_deg':0.,
            'scope':'equal-angle assembled seam rays; actual pocket XY footprints; no joint-tab CAD or retention structure generated'}


def _packing_bounds(cube_mm: float, config: Mapping, envelope: Mapping) -> dict:
    process = _configuration(config)
    cube = float(cube_mm)
    if not math.isfinite(cube) or cube <= 0:
        raise ValueError("Magnet edge must be positive and finite")
    printing = bool(process.get("enabled", True))
    pocket = cube + (2 * process["pocket_clearance_per_side_mm"] if printing else 0.0)
    half = pocket / 2
    bore = float(envelope["bore_radius_mm"])
    outer = float(envelope["outer_radius_mm"])
    # Clearance is measured from the protected bore wall to the MAGNET face.
    # Pocket clearance and the printed inner wall remain separate constraints.
    radial_start = max(bore + (process["inner_wall_mm"] if printing else 0.0) + half,
                       bore + process['radial_inner_clearance_mm'] + cube / 2)
    outer_inside_wall = outer - (process["outer_wall_mm"] if printing else 0.0)
    radial_end = (math.sqrt(outer_inside_wall ** 2 - half ** 2) - half
                  if outer_inside_wall > half else -math.inf)
    shared_channels = printing and process["layout_mode"] == "radial_channels"
    nominal_gap = float(envelope.get("min_gap_mm", 0.0))
    if not math.isfinite(nominal_gap) or nominal_gap < 0:
        raise ValueError("min_gap_mm must be a non-negative finite magnet clearance")
    # The end allowance belongs to the whole channel. It must never be charged
    # between touching magnets, nor must a radial web be inserted there.
    pitch = (cube + nominal_gap if shared_channels or not printing
             else pocket + process["pocket_web_mm"])
    tolerance = float(envelope.get("tolerance_mm", 1e-6))
    maximum = max(0, math.floor((radial_end - radial_start + tolerance) / pitch) + 1) if math.isfinite(radial_end) else 0
    return {
        "magnet_edge_mm": cube, "pocket_edge_mm": pocket,
        "required_radial_center_pitch_mm": pitch,
        "minimum_magnet_center_radius_mm": radial_start,
        "maximum_magnet_center_radius_mm": radial_end,
        "maximum_radial_rings": maximum,
        "protected_assembly_bore_radius_mm": bore,
        "outer_assembly_radius_mm": outer,
        "necessary_bound_only": True, "manufacturing_enabled": printing,
        "layout_mode": process["layout_mode"],
        "same_column_contact_allowed": (shared_channels or not printing) and nominal_gap == 0,
        "radial_spacing_policy": ("nominal magnet edge plus explicit min_gap_mm; no internal channel web or slot allowance"
                                  if shared_channels else "independent pocket edge plus radial web" if printing
                                  else "nominal magnet edge plus explicit min_gap_mm"),
        "radial_end_clearance_per_side_mm": process["pocket_clearance_per_side_mm"] if printing else 0.0,
        "circumferential_web_mm": process["pocket_web_mm"] if printing else 0.0,
        "minimum_magnet_inner_face_radius_mm": radial_start-cube/2,
        "radial_inner_clearance_mm": process['radial_inner_clearance_mm'],
    }


def axial_packing_bounds(cube_mm: float, config: Mapping, envelope: Mapping) -> dict:
    """Shared exact axial bounds for lattice, PAVA seed and continuous search.

    The central gap is a magnet-face gap. Two central baseplates and the
    inter-module clearance can require a larger magnet-face gap than requested.
    """
    process = _configuration(config)
    printing = bool(process.get('enabled', True))
    cube = float(cube_mm)
    pocket = cube + (2*process['pocket_clearance_per_side_mm'] if printing else 0.)
    base = process['base_thickness_mm'] if printing else 0.
    gap = process['min_axial_module_gap_mm'] if printing else float(envelope.get('min_gap_mm', 0.))
    minimum = max(pocket/2+base+gap/2, (cube+process['min_central_magnet_gap_mm'])/2)
    maximum = float(envelope['half_length_mm'])-pocket/2
    separation = pocket+base+gap
    return {'minimum_abs_magnet_center_z_mm': minimum,
            'maximum_abs_magnet_center_z_mm': maximum,
            'minimum_same_side_center_pitch_mm': separation,
            'minimum_central_magnet_face_gap_mm': 2*minimum-cube,
            'requested_central_magnet_face_gap_mm': process['min_central_magnet_gap_mm'],
            'maximum_layer_pairs': max(0, math.floor((maximum-minimum+1e-9)/separation)+1)}


def max_radial_rings(cube_mm: float, manufacturing_config: Mapping, envelope: Mapping) -> int:
    """Necessary radial packing bound; actual pockets still require OBB checks."""
    return _packing_bounds(cube_mm, manufacturing_config, envelope)["maximum_radial_rings"]


def radial_packing_bounds(cube_mm: float, manufacturing_config: Mapping, envelope: Mapping) -> dict:
    """Public shared bounds for parameterization; still only a necessary bound."""
    return _packing_bounds(cube_mm, manufacturing_config, envelope)


def _sector_bbox(inner: float, outer: float, angle: float, rotation: float) -> tuple[float, float]:
    """Exact axis projection of an annular sector, including arc extrema."""
    start, end = rotation, rotation + angle
    angles = [start, end]
    for quadrant in range(-8, 13):
        value = quadrant * math.pi / 2
        if start <= value <= end:
            angles.append(value)
    points = [(radius * math.cos(theta), radius * math.sin(theta))
              for radius in (inner, outer) for theta in angles]
    xs, ys = zip(*points)
    return max(xs) - min(xs), max(ys) - min(ys)


def _bed_fit(config: Mapping, envelope: Mapping, module_thickness: float) -> dict:
    # Footprints are constant across a search with the same assembly envelope
    # and process values. Never expose a cached mutable audit dictionary.
    configured_bed = list(config['printer_bed_mm'])
    physical_bed = config.get('machine_bed_mm') or configured_bed
    result = deepcopy(_bed_fit_cached(
        float(envelope["bore_radius_mm"]), float(envelope["outer_radius_mm"]),
        int(config["sectors_per_layer"]), tuple(physical_bed),
        float(config["bed_margin_per_side_mm"]), float(module_thickness)))
    if config.get('machine_bed_mm') is not None:
        margin = float(config['bed_margin_per_side_mm'])
        assembled = [2*float(envelope['outer_radius_mm'])+2*margin,
                     2*float(envelope['outer_radius_mm'])+2*margin, float(module_thickness)]
        assembly_fit = all(v <= b+1e-9 for v,b in zip(assembled, configured_bed))
        result.update(sector_fits_machine_bed=result['fits'],
                      machine_bed_mm=list(physical_bed), assembly_envelope_mm=configured_bed,
                      assembly_reserved_bbox_mm=assembled, assembly_fits=assembly_fit,
                      fits=bool(result['fits'] and assembly_fit),
                      assembly_scope='one assembled support layer; complete magnet stack is checked against axial envelope separately')
    return result


@lru_cache(maxsize=128)
def _bed_fit_cached(inner: float, outer: float, sectors: int, bed_dimensions: tuple,
                    margin: float, module_thickness: float) -> dict:
    sector_angle = 2 * math.pi / sectors
    bed = list(bed_dimensions)
    options = []
    # Each projected bounding box is exact; the search over rotations is sampled.
    # 0/90 degrees are included, so a quarter annulus is correctly R-by-R.
    for degrees in np.linspace(0, 180, 361):
        width, depth = _sector_bbox(inner, outer, sector_angle, math.radians(float(degrees)))
        reserved = [width + 2 * margin, depth + 2 * margin, module_thickness]
        utilization = max(value / available for value, available in zip(reserved, bed))
        options.append((utilization, float(degrees), width, depth, reserved))
    utilization, angle, width, depth, reserved = min(options)
    return {
        "fits": utilization <= 1 + 1e-9, "rotation_deg": angle,
        "part_bbox_mm": [width, depth, module_thickness],
        "reserved_bbox_mm": reserved, "printer_bed_mm": bed,
        "bed_margin_per_side_mm": margin, "maximum_bed_utilization": utilization,
        "area_utilization_percent": 100.*reserved[0]*reserved[1]/(bed[0]*bed[1]),
        "allowance_scope": "XY bed/joinery space reservation only; no actual joint tabs or connectors have been designed",
        "sector_angle_deg": math.degrees(sector_angle),
        "projection_method": "exact annular-sector boundary extrema, sampled in-plane rotations",
        "scope": "flat sector geometric fit; no joint detail, support, warpage or strength qualification",
    }


def _layer_modules(design: Design, pocket: float, base: float, tolerance: float) -> tuple[list, list]:
    groups = {}
    for ring in design.rings:
        groups.setdefault(ring.layer_index, []).append(ring)
    modules, errors = [], []
    for layer, rings in sorted(groups.items()):
        z = rings[0].z_mm
        if any(abs(r.z_mm - z) > tolerance for r in rings):
            errors.append(f"layer {layer} has inconsistent axial ring centers")
        # Base faces the central plane, leaving the open pocket on the outer side.
        # Its support center is offset from the magnet center by base/2.
        lower = z - pocket / 2 - (base if z >= 0 else 0)
        upper = z + pocket / 2 + (base if z < 0 else 0)
        modules.append({
            "layer_index": layer, "magnet_center_z_mm": z,
            "module_center_z_mm": (lower + upper) / 2,
            "module_z_min_mm": lower, "module_z_max_mm": upper,
            "module_thickness_mm": pocket + base,
            "base_direction": "-Z" if z >= 0 else "+Z",
            "pocket_opening_direction": "+Z" if z >= 0 else "-Z",
            "radial_ring_count": len(rings), "source_slot_ids": [r.id for r in rings],
            "magnet_count": sum(r.count for r in rings),
            "magnets_per_ring": sorted({r.count for r in rings}),
            "azimuth_columns": rings[0].count if len({r.count for r in rings}) == 1 else None,
        })
    return modules, errors


def audit_manufacturing(design: Design, manufacturing_config: Mapping, envelope: Mapping) -> dict:
    """Check pocket/wall packing, module envelopes, sectors, bed and magnet budget.

    ``envelope.bore_radius_mm`` protects the assembly passage.  It is not
    applied again as an exact naked-magnet bore condition: pocket clearances
    and the inner printed wall necessarily move magnets farther outward.
    """
    process = _configuration(manufacturing_config)
    if not process.get("enabled", True):
        nominal = validate_design(design, dict(envelope, preserve_exact_bore=False))
        budget_valid = design.magnet_count <= process["max_magnet_count"]
        violations = list(nominal["violations"])
        if not budget_valid:
            violations.append(f"magnet budget exceeded: {design.magnet_count} > {process['max_magnet_count']}")
        modules, module_errors = _layer_modules(design, design.cube_mm, 0.0, float(envelope.get("tolerance_mm", 1e-6)))
        violations.extend(module_errors)
        return {"valid": not violations, "violations": violations,
                "checks": {"manufacturing_enabled": False, "nominal_magnet_geometry": nominal["valid"], "magnet_budget": budget_valid},
                "packing": _packing_bounds(design.cube_mm, process, envelope), "magnet_count": design.magnet_count,
                "magnet_count_budget": process["max_magnet_count"], "layer_modules": modules,
                "sectors_per_layer": process["sectors_per_layer"], "total_printed_sectors": 0,
                "bed_fit": {"fits": True, "checked": False, "scope": "manufacturing checks disabled"},
                "process_assumptions": dict(process, base_orientation="none"), "nominal_geometry": nominal,
                "scope": "manufacturing checks explicitly disabled; nominal geometry and magnet budget remain enforced; no printability claim"}
    packing = _packing_bounds(design.cube_mm, process, envelope)
    tolerance = float(envelope.get("tolerance_mm", 1e-6))
    pocket, half = packing["pocket_edge_mm"], packing["pocket_edge_mm"] / 2
    violations, checks = [], {}
    # Nominal magnets use only the minimum passage and outer envelope; they do
    # not need to touch the passage boundary when a support wall is introduced.
    nominal_limits = dict(envelope, preserve_exact_bore=False,
                          min_gap_mm=(float(envelope.get("min_gap_mm", 0.0))
                                      if process["layout_mode"] == "radial_channels" else 0.0))
    nominal = validate_design(design, nominal_limits)
    checks["nominal_magnet_geometry"] = nominal["valid"]
    if not nominal["valid"]:
        violations.extend("nominal geometry: " + item for item in nominal["violations"])
    checks["magnet_budget"] = design.magnet_count <= process["max_magnet_count"]
    if not checks["magnet_budget"]:
        violations.append(f"magnet budget exceeded: {design.magnet_count} > {process['max_magnet_count']}")
    counts = {r.count for r in design.rings}
    layer_groups = {}
    for ring in design.rings:
        layer_groups.setdefault(ring.layer_index, []).append(ring)
    checks["unique_slot_ids"] = len({r.id for r in design.rings}) == len(design.rings)
    if not checks["unique_slot_ids"]:
        violations.append("source slot identifiers must be unique")
    checks["shared_azimuth_columns"] = bool(layer_groups) and all(
        len({r.count for r in rings}) == 1 for rings in layer_groups.values())
    partitions = [column_partition_check(count,process) for count in sorted(counts)]
    checks['column_partition_policy'] = bool(counts) and all(row['valid'] for row in partitions)
    checks['split_seam_even_columns' if process['allow_split_pocket_seams'] else 'columns_divisible_by_sectors'] = checks['column_partition_policy']
    if not checks["shared_azimuth_columns"]:
        violations.append("all radial rings within each axial layer must share one azimuth column count")
    if not checks['column_partition_policy']:
        violations.extend(note for row in partitions for note in row['violations'])
    phases_valid = True
    for rings in layer_groups.values():
        local_counts = {r.count for r in rings}
        if len(local_counts) != 1 or next(iter(local_counts)) <= 0:
            phases_valid = False
            continue
        pitch = 360 / next(iter(local_counts))
        phase = rings[0].phase_deg
        if not all(abs((r.phase_deg - phase + pitch / 2) % pitch - pitch / 2) <= 1e-7 for r in rings):
            phases_valid = False
    checks["shared_column_phase"] = phases_valid
    if not phases_valid:
        violations.append("radial rings within an axial layer do not share the same sector column phase")
    inner_bad, outer_bad = [], []
    for ring in design.rings:
        if ring.radius_mm - half < envelope["bore_radius_mm"] + process["inner_wall_mm"] - tolerance:
            inner_bad.append(ring.id)
        corner = math.hypot(ring.radius_mm + half, half)
        if corner > envelope["outer_radius_mm"] - process["outer_wall_mm"] + tolerance:
            outer_bad.append(ring.id)
    clearance_bad = [r.id for r in design.rings if
                     r.radius_mm-design.cube_mm/2 < envelope['bore_radius_mm']+process['radial_inner_clearance_mm']-tolerance]
    checks['magnet_inner_face_clearance'] = not clearance_bad
    if clearance_bad:
        violations.append('magnet inner face violates protected-bore clearance: ' + ', '.join(clearance_bad))
    checks["pocket_inner_wall"] = not inner_bad
    checks["pocket_outer_wall"] = not outer_bad
    if inner_bad:
        violations.append("pocket inner wall invades protected passage: " + ", ".join(inner_bad))
    if outer_bad:
        violations.append("pocket outer wall exceeds assembly envelope: " + ", ".join(outer_bad))
    module_list, module_errors = _layer_modules(design, pocket, process["base_thickness_mm"], tolerance)
    violations.extend(module_errors)
    radial_bad = [m["layer_index"] for m in module_list if m["radial_ring_count"] > packing["maximum_radial_rings"]]
    checks["necessary_radial_packing_bound"] = not radial_bad
    if radial_bad:
        violations.append("radial ring count exceeds necessary pocket/wall packing bound in layers: " + str(radial_bad))
    axial_bad = [m["layer_index"] for m in module_list
                 if m["module_z_min_mm"] < -envelope["half_length_mm"] - tolerance
                 or m["module_z_max_mm"] > envelope["half_length_mm"] + tolerance]
    checks["module_axial_envelope"] = not axial_bad
    if axial_bad:
        violations.append("printed layer module exceeds axial envelope in layers: " + str(axial_bad))
    ordered = sorted(module_list, key=lambda m: m["module_z_min_mm"])
    axial_overlaps = [(a["layer_index"], b["layer_index"]) for a, b in zip(ordered, ordered[1:])
                      if a["module_z_max_mm"] + process["min_axial_module_gap_mm"] > b["module_z_min_mm"] + tolerance]
    checks["module_axial_separation"] = not axial_overlaps and not module_errors
    if axial_overlaps:
        violations.append("printed layer modules overlap axially: " + str(axial_overlaps))
    positive = [m['magnet_center_z_mm'] for m in module_list if m['magnet_center_z_mm'] >= 0]
    negative = [m['magnet_center_z_mm'] for m in module_list if m['magnet_center_z_mm'] < 0]
    center_face_gap = min(positive)-max(negative)-design.cube_mm if positive and negative else None
    checks['central_magnet_face_gap'] = (center_face_gap is None or
        center_face_gap >= process['min_central_magnet_gap_mm']-tolerance)
    if not checks['central_magnet_face_gap']:
        violations.append('central magnet-face gap below required minimum')
    shared_channels = process["layout_mode"] == "radial_channels"
    if shared_channels:
        radial_gap = packing["required_radial_center_pitch_mm"]
        bad_radial_pairs = [(a.id, b.id) for rings in layer_groups.values()
                            for a, b in zip(sorted(rings, key=lambda r: r.radius_mm),
                                            sorted(rings, key=lambda r: r.radius_mm)[1:])
                            if b.radius_mm - a.radius_mm < radial_gap - tolerance]
        checks["radial_magnet_separation"] = not bad_radial_pairs
        if bad_radial_pairs:
            violations.append("same-column magnets violate nominal edge/min_gap spacing: " + str(bad_radial_pairs))
    # XY webs remain between different circumferential columns. In a shared
    # radial channel, magnets on the exact same radial ray have no intervening
    # printed wall. Real magnet overlap is checked above using full 3D cubes.
    # Cross-layer spacing uses complete axial modules, never an inflated Z.
    planar_edge = pocket + process["pocket_web_mm"]
    clearance = planar_pocket_clearance(design, planar_edge, tolerance,
                                        allow_same_column_contact=shared_channels)
    checks["pocket_web_obb_clearance"] = clearance["valid"]
    if not clearance["valid"]:
        violations.append(f"pocket/web XY OBB: overlapping pairs: {clearance['overlap_pairs']}")
    footprint = dict(envelope)
    if process["bed_fit_mode"] == "candidate_envelope":
        footprint["bore_radius_mm"] = max(0.0, min(r.radius_mm-half-process["inner_wall_mm"] for r in design.rings))
        footprint["outer_radius_mm"] = max(math.hypot(r.radius_mm+half, half)+process["outer_wall_mm"] for r in design.rings)
    bed_fit = _bed_fit(process, footprint, pocket + process["base_thickness_mm"])
    bed_fit.update(footprint_mode=process["bed_fit_mode"],
                   footprint_inner_radius_mm=footprint["bore_radius_mm"], footprint_outer_radius_mm=footprint["outer_radius_mm"])
    checks["sector_fits_printer_bed"] = bed_fit["fits"]
    if not bed_fit["fits"]:
        violations.append("printed sector plus XY allowance does not fit the configured printer bed")
    return {
        "valid": not violations, "violations": violations, "checks": checks,
        "packing": packing, "magnet_count": design.magnet_count,
        "magnet_count_budget": process["max_magnet_count"], "layer_modules": module_list,
        "sectors_per_layer": process["sectors_per_layer"],
        "column_sharing_scope": "within_each_axial_layer; different axial layers may have different column counts",
        "minimum_required_module_gap_mm": process["min_axial_module_gap_mm"],
        "central_magnet_face_gap_mm": center_face_gap,
        "minimum_central_magnet_face_gap_mm": process['min_central_magnet_gap_mm'],
        "total_printed_sectors": len(module_list) * process["sectors_per_layer"],
        "column_partition": partitions,
        "sector_partition": _seam_partition_metadata(design,pocket,process,tolerance),
        "bed_fit": bed_fit,
        "slot_clearance": {"inflated_obb_edge_mm": planar_edge,
                           "overlapping_proxy_pairs": clearance.get("overlap_pairs"),
                           "same_column_pairs_exempted": clearance.get("same_column_pairs_exempted", 0),
                           "same_column_contact_allowed": packing["same_column_contact_allowed"],
                           "same_column_web_exempt": shared_channels,
                           "method": ("different-column same-layer XY OBB checks; aligned radial magnets share one channel; full nominal 3D collision remains enforced"
                                      if shared_channels else "same-layer XY separating-axis test, local XY web/2 inflation; cross-layer Z handled by module envelopes")},
        "process_assumptions": dict(process, base_orientation="toward_midplane"),
        "nominal_geometry": nominal,
        "scope": "geometric process feasibility screen using parametric pocket OBB and annular-sector envelopes; no actual STL/CAD was generated or validated; structural strength, magnet retention, joints, forces, heat and print tolerances not qualified",
    }


def make_manufacturing_seed(source_design: Design, manufacturing_config: Mapping,
                            envelope: Mapping) -> tuple[Design, dict]:
    """Project source geometry to the nearest ordered printable radii/axial centers.

    No magnetic objective is used here. Callers can enumerate source-slot
    subsets and compare their fields separately. If a supplied layer contains
    more slots than packing permits, choose the subset with the smallest radial
    projection displacement, retaining an explicit provenance record.
    """
    process = _configuration(manufacturing_config)
    packing = _packing_bounds(source_design.cube_mm, process, envelope)
    maximum = packing["maximum_radial_rings"]
    if maximum < 1:
        raise ValueError("No radial pocket fits the assembly envelope")
    groups = {}
    for ring in source_design.rings:
        groups.setdefault(ring.layer_index, []).append(ring)
    printing = bool(process.get("enabled", True))
    pocket = packing["pocket_edge_mm"]
    base = process["base_thickness_mm"] if printing else 0.0
    gap = process["min_axial_module_gap_mm"] if printing else float(envelope.get("min_gap_mm", 0.0))
    radial_lower = packing["minimum_magnet_center_radius_mm"]
    radial_upper = packing["maximum_magnet_center_radius_mm"]
    pitch = packing["required_radial_center_pitch_mm"]
    tolerance = float(envelope.get("tolerance_mm", 1e-6))
    axial = []
    for layer, group in groups.items():
        z = group[0].z_mm
        if any(abs(r.z_mm-z) > tolerance for r in group):
            raise ValueError("Source layer has inconsistent axial ring centers: " + str(layer))
        sign = 1 if z >= 0 else -1
        # Every module has equal thickness; translating to its center removes
        # the orientation-dependent base offset. For a mirrored source this
        # convex projection preserves mirror symmetry automatically.
        axial.append((z-sign*base/2, layer, sign))
    axial.sort()
    module_half = (pocket+base)/2
    nearest = project_ordered([center for center, _, _ in axial],
                              -float(envelope["half_length_mm"])+module_half,
                              float(envelope["half_length_mm"])-module_half,
                              pocket+base+gap)
    projected_z = {layer: float(center+sign*base/2)
                   for (_, layer, sign), center in zip(axial, nearest)}
    if process['min_central_magnet_gap_mm'] > 0:
        bounds = axial_packing_bounds(source_design.cube_mm, process, envelope)
        for sign in (1, -1):
            side = sorted((abs(group[0].z_mm), layer) for layer,group in groups.items()
                          if (1 if group[0].z_mm >= 0 else -1) == sign)
            if not side:
                continue
            positions = project_ordered([z for z,_ in side],
                bounds['minimum_abs_magnet_center_z_mm'], bounds['maximum_abs_magnet_center_z_mm'],
                bounds['minimum_same_side_center_pitch_mm'])
            for (_,layer),z in zip(side, positions):
                projected_z[layer] = sign*float(z)
    for layer, group in groups.items():
        if abs(projected_z[layer]-group[0].z_mm) <= 1e-12:
            projected_z[layer] = group[0].z_mm
    rings, retained, removed, mapping = [], [], [], []
    selection = []
    for layer, original in sorted(groups.items()):
        original = sorted(original, key=lambda r: r.radius_mm)
        count = min(len(original), maximum)
        options = []
        for subset in combinations(range(len(original)), count):
            source_radii = np.array([original[i].radius_mm for i in subset])
            candidate_radii = project_ordered(source_radii, radial_lower, radial_upper, pitch)
            displacement = float(np.sum((candidate_radii-source_radii)**2))
            options.append((displacement, tuple(subset), candidate_radii))
        displacement, indices, radii = min(options, key=lambda item: (item[0], item[1]))
        chosen = set(indices)
        selection.append({"layer_index": layer, "supplied_source_slots": len(original),
                          "retained_source_indices": list(indices), "projection_squared_displacement_mm2": displacement})
        for position, (index, radius) in enumerate(zip(indices, radii)):
            source = original[index]
            z = projected_z[layer]
            rings.append(replace(source, radius_mm=float(radius), z_mm=z))
            retained.append(source.id)
            mapping.append({"source_slot_id": source.id, "seed_slot_id": source.id,
                            "source_slot_index": source.slot_index, "seed_radial_position_index": position,
                            "source_radius_mm": source.radius_mm, "seed_radius_mm": radius,
                            "source_z_mm": source.z_mm, "seed_z_mm": z,
                            "inherited_polarity": source.polarity, "count": source.count})
        removed.extend(ring.id for index, ring in enumerate(original) if index not in chosen)
    seed = replace(source_design, rings=tuple(rings), name="manufacturing_seed_not_optimized")
    audit = audit_manufacturing(seed, process, envelope)
    report = {
        "valid": audit["valid"], "violations": audit["violations"],
        "kind": "geometric_manufacturing_seed_not_field_optimized",
        "source_magnet_count": source_design.magnet_count, "seed_magnet_count": seed.magnet_count,
        "source_ring_count": len(source_design.rings), "seed_ring_count": len(seed.rings),
        "retained_source_slot_ids": retained, "removed_source_slot_ids": removed,
        "source_to_seed_slots": mapping,
        "radial_projection": "bounded equal-weight isotonic least-squares projection with minimum pitch",
        "axial_projection": "bounded ordered least-squares projection of complete module centers, including bases",
        "radial_subset_projection": selection,
        "axial_displacements_mm": {str(layer): projected_z[layer]-group[0].z_mm for layer, group in sorted(groups.items())},
        "polarities_inherited_without_optimization": True, "packing": packing, "manufacturing_audit": audit,
        "note": "Original slot IDs and slot_index are retained for provenance; radial_position_index is the new contiguous packing position.",
    }
    return seed, report

"""Read the supplied IO-ring geometry and construct constrained design variants."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree

POLARITIES = ("out", "in", "down", "up")
_LABELS = {"径向朝外": "out", "径向朝内": "in", "轴向−Z": "down", "轴向-Z": "down", "轴向朝下": "down", "轴向+Z": "up"}

@dataclass(frozen=True)
class RingSpec:
    id: str
    layer_index: int
    slot_index: int
    radius_mm: float
    z_mm: float
    count: int
    polarity: str
    phase_deg: float

@dataclass(frozen=True)
class Design:
    rings: Tuple[RingSpec, ...]
    cube_mm: float = 12.0
    remanence_T: float = 1.43
    reference_mur: float = 1.05
    name: str = "baseline"
    source_sha256: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(rings=tuple(RingSpec(**x) for x in data["rings"]), **{k:v for k,v in data.items() if k != "rings"})

    @property
    def magnet_count(self):
        return sum(r.count for r in self.rings)

    def magnet_arrays(self):
        positions, directions, angles, ids = [], [], [], []
        for ring in self.rings:
            theta = np.deg2rad(ring.phase_deg + np.arange(ring.count)*360.0/ring.count)
            cs, sn = np.cos(theta), np.sin(theta)
            positions.append(np.column_stack((ring.radius_mm*cs, ring.radius_mm*sn, np.full(ring.count,ring.z_mm))))
            if ring.polarity in ("out", "in"):
                sign = 1.0 if ring.polarity == "out" else -1.0
                directions.append(np.column_stack((sign*cs,sign*sn,np.zeros(ring.count))))
            elif ring.polarity in ("down", "up"):
                directions.append(np.tile([0,0,-1.0 if ring.polarity == "down" else 1.0],(ring.count,1)))
            else:
                raise ValueError("Unknown polarity: " + ring.polarity)
            angles.extend(np.rad2deg(theta))
            ids.extend(f"{ring.id}-A{i+1:02d}" for i in range(ring.count))
        return {"positions_mm":np.concatenate(positions), "north_vectors":np.concatenate(directions),
                "radial_angles_deg":np.asarray(angles), "ids":ids, "cube_mm":self.cube_mm}

def load_source(path):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    m = data["metadata"]
    rings = []
    for i, layer in enumerate(data["rings"]):
        for j, slot in enumerate(layer["slots"]):
            rings.append(RingSpec(f"{layer['id']}-L{slot['radialLayer']}",i,j,float(slot["centerRMM"]),
                                  float(layer["centerZMM"]),int(m["azimuthColumnsPerRing"]),
                                  _LABELS[slot["direction"]],float(m["matlabFirstColumnAzimuthDegrees"])))
    design = Design(tuple(rings),float(m["cubeEdgeMM"]),float(m["remanenceT"]),1.05,
                    "baseline",hashlib.sha256(raw).hexdigest())
    # Validate original orientations against explicit source vectors, rather than Euler-name guesses.
    arrays = design.magnet_arrays()
    supplied = {x["id"]:x for x in data["magnets"]}
    if len(supplied) != len(data["magnets"]) or len(supplied) != design.magnet_count:
        raise ValueError("Source magnet IDs/count do not agree with ring table")
    for ident, pos, direction in zip(arrays["ids"],arrays["positions_mm"],arrays["north_vectors"]):
        item = supplied[ident]
        if not np.allclose(pos,item["centerXYZMM"],atol=1e-6,rtol=0):
            raise ValueError("Coordinate mismatch: " + ident)
        if not np.allclose(direction,item["northDirectionUnitXYZ"],atol=1e-7,rtol=0):
            raise ValueError("Magnetization mismatch: " + ident)
    return design, data

def envelope_limits(design):
    half = design.cube_mm/2
    return {"bore_radius_mm":min(r.radius_mm-half for r in design.rings),
            "outer_radius_mm":max(math.hypot(r.radius_mm+half,half) for r in design.rings),
            "half_length_mm":max(abs(r.z_mm)+half for r in design.rings),
            "min_gap_mm":0.0,"tolerance_mm":1e-6,"preserve_exact_bore":True}

def validate_design(design, limits):
    """Exact XY separating-axis tests for yawed cubes, with axis-aligned Z.

    Face contact is permitted at min_gap=0, as it occurs in the supplied baseline.
    Positive min_gap conservatively inflates both cubes by that clearance.
    """
    violations = []
    tol = float(limits.get("tolerance_mm",1e-6))
    half = design.cube_mm/2
    if not design.rings or design.cube_mm<=0 or not np.isfinite(design.remanence_T) or design.remanence_T<=0:
        return {"valid":False,"violations":["invalid dimensions/remanence or no rings"]}
    for r in design.rings:
        if r.count < 3 or int(r.count)!=r.count or r.polarity not in POLARITIES or not np.all(np.isfinite([r.radius_mm,r.z_mm,r.phase_deg])):
            violations.append("invalid ring: "+r.id)
    if violations:
        return {"valid":False,"violations":violations}
    bore = min(r.radius_mm-half for r in design.rings)
    outer = max(math.hypot(r.radius_mm+half,half) for r in design.rings)
    axial = max(abs(r.z_mm)+half for r in design.rings)
    if bore < limits["bore_radius_mm"]-tol:violations.append("bore invaded")
    if limits.get("preserve_exact_bore",False) and abs(bore-limits["bore_radius_mm"])>tol:violations.append("fixed bore changed")
    if outer > limits["outer_radius_mm"]+tol:violations.append("outer envelope exceeded")
    if axial > limits["half_length_mm"]+tol:violations.append("axial envelope exceeded")
    arr = design.magnet_arrays()
    pos = arr["positions_mm"]
    angle = np.deg2rad(arr["radial_angles_deg"])
    axes = np.stack((np.column_stack((np.cos(angle),np.sin(angle))),np.column_stack((-np.sin(angle),np.cos(angle)))),axis=1)
    test_half = half + float(limits.get("min_gap_mm",0))/2
    pairs = cKDTree(pos).query_pairs(2*test_half*math.sqrt(3)+tol,output_type="ndarray")
    overlap_count = 0
    if len(pairs):
        left,right = pairs[:,0],pairs[:,1]
        candidate = np.abs(pos[left,2]-pos[right,2]) < 2*test_half-tol
        left,right=left[candidate],right[candidate]
        if len(left):
            delta = pos[right,:2]-pos[left,:2]
            tests = np.concatenate((axes[left],axes[right]),axis=1)
            projected_distance = np.abs(np.einsum('nki,ni->nk',tests,delta))
            extent_left = test_half*np.sum(np.abs(np.einsum('nki,nji->nkj',tests,axes[left])),axis=2)
            extent_right = test_half*np.sum(np.abs(np.einsum('nki,nji->nkj',tests,axes[right])),axis=2)
            overlap = np.all(projected_distance < extent_left+extent_right-tol,axis=1)
            overlap_count=int(np.sum(overlap))
    if overlap_count:violations.append(f"overlapping cube pairs: {overlap_count}")
    return {"valid":not violations,"violations":violations,"magnet_count":design.magnet_count,
            "bore_diameter_mm":2*bore,"outer_diameter_mm":2*outer,"length_mm":2*axial,
            "overlap_pairs":overlap_count,"minimum_required_gap_mm":limits.get("min_gap_mm",0)}


def planar_pocket_clearance(design, xy_edge_mm, tolerance_mm=1e-6, *, allow_same_column_contact=False):
    """Same-layer XY separating-axis check, without any axial web inflation.

    Each pocket is conservatively inflated along its two local planar axes.
    Different axial layers are excluded: their complete module envelopes,
    including bases, are checked by manufacturing._layer_modules instead.
    When ``allow_same_column_contact`` is true, exactly aligned radial magnets
    share a continuous channel and are excluded only from this printed-web
    proxy test. Callers must still check their true full-size 3D geometry.
    Column membership uses actual periodic azimuths, never matching array
    indices: rings with different column counts/phases cannot be conflated.
    """
    edge, tol = float(xy_edge_mm), float(tolerance_mm)
    if not np.isfinite(edge) or edge <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("Invalid planar pocket edge/tolerance")
    arrays = design.magnet_arrays()
    positions = arrays["positions_mm"][:, :2]
    angle = np.deg2rad(arrays["radial_angles_deg"])
    axes = np.stack((np.column_stack((np.cos(angle), np.sin(angle))),
                     np.column_stack((-np.sin(angle), np.cos(angle)))), axis=1)
    layers = np.concatenate([np.full(r.count, r.layer_index, dtype=int) for r in design.rings])
    half, overlap_count, exempt_count = edge / 2, 0, 0
    for layer in np.unique(layers):
        indices = np.flatnonzero(layers == layer)
        local_pairs = cKDTree(positions[indices]).query_pairs(edge * math.sqrt(2) + tol, output_type="ndarray")
        if not len(local_pairs):
            continue
        left, right = indices[local_pairs[:, 0]], indices[local_pairs[:, 1]]
        if allow_same_column_contact:
            angular_difference = np.arctan2(np.sin(angle[right] - angle[left]),
                                           np.cos(angle[right] - angle[left]))
            # Equal angles modulo 2pi identify the same radial ray. Opposite
            # directions are different columns even though their axes align.
            same_column = np.abs(angular_difference) <= 1e-10
            exempt_count += int(np.count_nonzero(same_column))
            left, right = left[~same_column], right[~same_column]
            if not len(left):
                continue
        delta = positions[right] - positions[left]
        tests = np.concatenate((axes[left], axes[right]), axis=1)
        distance = np.abs(np.einsum('nki,ni->nk', tests, delta))
        extent_left = half * np.sum(np.abs(np.einsum('nki,nji->nkj', tests, axes[left])), axis=2)
        extent_right = half * np.sum(np.abs(np.einsum('nki,nji->nkj', tests, axes[right])), axis=2)
        overlap_count += int(np.count_nonzero(np.all(distance < extent_left + extent_right - tol, axis=1)))
    return {"valid": overlap_count == 0, "overlap_pairs": overlap_count,
            "xy_inflated_edge_mm": edge, "same_column_pairs_exempted": exempt_count,
            "same_column_contact_allowed": bool(allow_same_column_contact),
            "scope": "same axial layer only; XY OBB inflation, no Z inflation; shared-channel exemptions do not waive nominal magnet collision"}


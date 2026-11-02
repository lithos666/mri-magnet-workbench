"""Separate source-content provenance from tolerance-defined physical identity.

Positions and lengths are quantized to 1e-9 mm; phase to 1e-9 degree after
reduction by the ring period. Material inputs retain their finite float values.
Labels and source-file hashes do not affect physical equivalence. This is an
explicit numerical tolerance contract, not an exact geometric-isomorphism test.
"""
from __future__ import annotations

import hashlib
import json
import math

from .serialization import jsonable

CANONICAL_SCHEMA_VERSION = "mri-physical-v2-mm9-deg9"


def _number(value, digits=None):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Physical fingerprints require finite numbers")
    value = round(value, digits) if digits is not None else value
    return 0.0 if value == 0 else value


def canonical_phase_deg(phase_deg, count):
    if int(count) != count or count < 1:
        raise ValueError("Ring count must be a positive integer")
    period = 360.0 / int(count)
    value = _number(float(phase_deg) % period, 9)
    # Rounding a tiny negative phase can land on the upper periodic boundary.
    if value <= 0.5e-9 or period - value <= 0.5e-9:
        return 0.0
    return value


def canonical_physical_data(design, polarization_model="remanence"):
    data = design.to_dict() if hasattr(design, "to_dict") else design
    rings = []
    for ring in data["rings"]:
        count = int(ring["count"])
        if count != ring["count"] or count < 1:
            raise ValueError("Invalid ring count")
        rings.append({"z_mm": _number(ring["z_mm"], 9),
                      "radius_mm": _number(ring["radius_mm"], 9),
                      "count": count,
                      "phase_deg": canonical_phase_deg(ring["phase_deg"], count),
                      "polarity": str(ring["polarity"])})
    rings.sort(key=lambda r: (r["z_mm"], r["radius_mm"], r["count"], r["phase_deg"], r["polarity"]))
    return {"schema": CANONICAL_SCHEMA_VERSION, "units": {"length": "mm", "polarization": "T"},
            "cube_mm": _number(data["cube_mm"], 9),
            "remanence_T": _number(data["remanence_T"]),
            "reference_mur": _number(data["reference_mur"]),
            "polarization_model": str(polarization_model), "rings": rings}


def _digest(value):
    encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_design_sha256(design):
    """Hash the full serialized design, distinct from original file-byte SHA."""
    return _digest(design.to_dict() if hasattr(design, "to_dict") else design)


def canonical_physical_sha256(design, polarization_model="remanence"):
    return _digest(canonical_physical_data(design, polarization_model))

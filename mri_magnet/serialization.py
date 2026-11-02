"""Dependency-light, deterministic JSON serialization shared by all modules."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import numpy as np


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                 allow_nan=False), encoding="utf-8")


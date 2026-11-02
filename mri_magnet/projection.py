"""Exact Euclidean projections onto bounded, ordered spacing constraints."""
from __future__ import annotations

import math
import numpy as np


def isotonic_projection(values):
    """Equal-weight nondecreasing least-squares projection, using O(n) PAVA."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Expected a nonempty finite one-dimensional vector")
    sums, sizes = [], []
    for value in data:
        sums.append(float(value)); sizes.append(1)
        while len(sums) > 1 and sums[-2] / sizes[-2] > sums[-1] / sizes[-1]:
            sums[-2] += sums[-1]
            sizes[-2] += sizes[-1]
            sums.pop(); sizes.pop()
    return np.concatenate([np.full(size, total / size) for total, size in zip(sums, sizes)])


def project_ordered(values, lower, upper, min_gap=0.0):
    """Nearest vector with lower <= x[0], diff(x) >= gap, x[-1] <= upper.

    Subtracting the cumulative minimum gap gives an isotonic regression with
    one common box.  PAVA followed by clipping to that common box is the exact
    Euclidean projection; it does not replace the input by an evenly spaced grid.
    Input order defines the physical order and is deliberately not sorted here.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("Expected a nonempty finite one-dimensional vector")
    lower, upper, gap = float(lower), float(upper), float(min_gap)
    if not all(math.isfinite(v) for v in (lower, upper, gap)) or gap < 0:
        raise ValueError("Projection bounds/gap must be finite and gap nonnegative")
    transformed_upper = upper - (len(data) - 1) * gap
    if transformed_upper < lower - 1e-12:
        raise ValueError("The requested ordered spacing does not fit the bounds")
    transformed_upper = max(lower, transformed_upper)
    offset = np.arange(len(data), dtype=float) * gap
    fitted = isotonic_projection(data - offset)
    return np.clip(fitted, lower, transformed_upper) + offset

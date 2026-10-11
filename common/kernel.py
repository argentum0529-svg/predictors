"""Look-alike kernel: distance to today's state, Gaussian weights, bandwidth choice.

Kept free of any predictor's output format so other predictors can reuse it.
"""

from __future__ import annotations

import math

from common.stats import effective_sample_size, kish_size

H_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
TARGET_NEFF = 20.0
THIN_DISTANCE = 3.0


def distances(z_rows: list[tuple[float, ...]], z_today: tuple[float, ...]) -> list[float]:
    """Euclidean distance of each row's z-vector to ``z_today``."""
    return [math.sqrt(sum((a - b) ** 2 for a, b in zip(z, z_today))) for z in z_rows]


def gaussian_weights(dists: list[float], h: float) -> list[float]:
    return [math.exp(-d * d / (2.0 * h * h)) for d in dists]


def select_bandwidth(
    taus: list[float],
    dists: list[float],
    grid: tuple[float, ...] = H_GRID,
    target_neff: float = TARGET_NEFF,
) -> dict:
    """Smallest ``h`` in ``grid`` whose weights reach ``target_neff``.

    ``taus`` and ``dists`` describe the eligible rows, in the same order.
    Falls back to the largest ``h`` with ``reached = False`` when none does.
    ``empty`` is true when every weight is zero even at that bandwidth.
    """
    if not dists:
        return {
            "h": max(grid),
            "k": [],
            "reached": False,
            "empty": True,
            "min_dist": None,
            "kish_ratio": None,
            "neff": 0.0,
        }
    chosen = max(grid)
    reached = False
    weights = gaussian_weights(dists, chosen)
    for h in sorted(grid):
        candidate = gaussian_weights(dists, h)
        if effective_sample_size(candidate, taus) >= target_neff:
            chosen, weights, reached = h, candidate, True
            break
    else:
        weights = gaussian_weights(dists, chosen)
    sum_w = sum(weights)
    positive = sum(1 for w in weights if w > 0)
    kish_ratio = None
    if sum_w > 0 and positive:
        kish_ratio = kish_size(weights) / positive
    return {
        "h": chosen,
        "k": weights,
        "reached": reached,
        "empty": sum_w <= 0,
        "min_dist": min(dists),
        "kish_ratio": kish_ratio,
        "neff": effective_sample_size(weights, taus),
    }

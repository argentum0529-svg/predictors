"""Option 2 time-weighted stats, effective sample size, and a block-bootstrap range."""

from __future__ import annotations

import hashlib
import math
import random

BOOTSTRAP_DRAWS = 500


def seed_for(ticker: str, as_of: str) -> int:
    digest = hashlib.sha256(f"{ticker}|{as_of}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def kish_size(weights: list[float]) -> float:
    """``(sum w)^2 / sum w^2``. Weights are rescaled first so tiny ones do not underflow."""
    peak = max(weights, default=0.0)
    if peak <= 0:
        return 0.0
    scaled = [w / peak for w in weights]
    return sum(scaled) ** 2 / sum(w * w for w in scaled)


def effective_sample_size(weights: list[float], taus: list[float]) -> float:
    """``kish / over``: uneven weights and overlapping trades both shrink the evidence."""
    sum_w = sum(weights)
    if sum_w <= 0:
        return 0.0
    over = max(1.0, sum(w * t for w, t in zip(weights, taus)) / sum_w)
    return kish_size(weights) / over


def compute(rows: list[dict], *, ticker: str, as_of: str, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Stats over rows that carry ``G``, ``tau``, ``censored``, and ``eligible``.

    Rows that are not eligible are left out. Each remaining row has weight
    ``row["w"]`` (1 when the key is missing). ``lo`` and ``hi`` are the 5th and
    95th percentiles of a block bootstrap of ``r_tw``; zero-weight rows stay in
    the resampled pool so blocks keep their calendar shape.
    """
    pool = [row for row in rows if row.get("eligible")]
    pool.sort(key=lambda row: row.get("d0", 0))
    weights = [float(row.get("w", 1.0)) for row in pool]
    empty = {
        "r_tw": None,
        "r_mean": None,
        "sd": None,
        "score": None,
        "p_pos": None,
        "days": None,
        "p_1d": None,
        "n": 0,
        "neff": 0.0,
        "kish": 0.0,
        "lo": None,
        "hi": None,
    }
    sum_w = sum(weights)
    if not pool or sum_w <= 0:
        return empty

    sum_g = sum(w * row["G"] for w, row in zip(weights, pool))
    sum_tau = sum(w * row["tau"] for w, row in zip(weights, pool))
    if sum_tau <= 0:
        return empty
    r_tw = sum_g / sum_tau
    r_mean = sum(w * row["G"] / row["tau"] for w, row in zip(weights, pool)) / sum_w
    var = sum(
        w * row["tau"] * (row["G"] / row["tau"] - r_tw) ** 2 for w, row in zip(weights, pool)
    ) / sum_tau
    sd = math.sqrt(max(0.0, var))
    score = None if sd == 0 else r_tw / sd
    p_pos = sum(w for w, row in zip(weights, pool) if row["G"] > 0) / sum_w
    days = sum_tau / sum_w
    p_1d = sum(w for w, row in zip(weights, pool) if row["tau"] <= 1 and not row["censored"]) / sum_w
    kish = kish_size(weights)
    over = max(1.0, days)
    neff = kish / over
    lo, hi = _bootstrap(pool, weights, r_draws=draws, seed=seed_for(ticker, as_of), over=over)
    return {
        "r_tw": r_tw,
        "r_mean": r_mean,
        "sd": sd,
        "score": score,
        "p_pos": p_pos,
        "days": days,
        "p_1d": p_1d,
        "n": sum(1 for w in weights if w > 0),
        "neff": neff,
        "kish": kish,
        "lo": lo,
        "hi": hi,
    }


def _bootstrap(
    rows: list[dict], weights: list[float], *, r_draws: int, seed: int, over: float
) -> tuple[float | None, float | None]:
    n = len(rows)
    length = max(1, math.ceil(over))
    rng = random.Random(seed)
    n_blocks = max(1, math.ceil(n / length))
    last_start = max(0, n - length)
    gains = [w * row["G"] for w, row in zip(weights, rows)]
    spans = [w * row["tau"] for w, row in zip(weights, rows)]
    samples: list[float] = []
    for _ in range(r_draws):
        sum_g = 0.0
        sum_tau = 0.0
        for _block in range(n_blocks):
            start = rng.randrange(0, last_start + 1)
            sum_g += sum(gains[start : start + length])
            sum_tau += sum(spans[start : start + length])
        if sum_tau <= 0:
            continue
        samples.append(sum_g / sum_tau)
    if not samples:
        return None, None
    return _percentile(samples, 0.05), _percentile(samples, 0.95)

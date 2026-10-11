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


def compute(rows: list[dict], *, ticker: str, as_of: str, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Stats over rows that carry ``G``, ``tau``, ``censored``, and ``eligible``.

    Weight is 1 for an eligible row and 0 otherwise. ``lo`` and ``hi`` are the
    5th and 95th percentiles of a block bootstrap of ``r_tw``.
    """
    weighted = [row for row in rows if row.get("eligible")]
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
        "lo": None,
        "hi": None,
    }
    if not weighted:
        return empty

    sum_w = float(len(weighted))
    sum_g = sum(row["G"] for row in weighted)
    sum_tau = sum(row["tau"] for row in weighted)
    if sum_tau <= 0:
        return empty
    r_tw = sum_g / sum_tau
    r_mean = sum(row["G"] / row["tau"] for row in weighted) / sum_w
    sum_v = sum_tau  # v_i = w_i * tau_i and w_i = 1
    var = sum(row["tau"] * (row["G"] / row["tau"] - r_tw) ** 2 for row in weighted) / sum_v
    sd = math.sqrt(max(0.0, var))
    score = None if sd == 0 else r_tw / sd
    p_pos = sum(1 for row in weighted if row["G"] > 0) / sum_w
    days = sum_tau / sum_w
    p_1d = sum(1 for row in weighted if row["tau"] <= 1 and not row["censored"]) / sum_w
    # Equal weights: kish = n. Overlap ≈ average holding time.
    over = max(1.0, days)
    neff = sum_w / over
    lo, hi = _bootstrap(weighted, r_draws=draws, seed=seed_for(ticker, as_of), over=over)
    return {
        "r_tw": r_tw,
        "r_mean": r_mean,
        "sd": sd,
        "score": score,
        "p_pos": p_pos,
        "days": days,
        "p_1d": p_1d,
        "n": int(sum_w),
        "neff": neff,
        "lo": lo,
        "hi": hi,
    }


def _bootstrap(rows: list[dict], *, r_draws: int, seed: int, over: float) -> tuple[float | None, float | None]:
    n = len(rows)
    length = max(1, math.ceil(over))
    rng = random.Random(seed)
    n_blocks = max(1, math.ceil(n / length))
    last_start = max(0, n - length)
    samples: list[float] = []
    for _ in range(r_draws):
        chosen: list[dict] = []
        for _block in range(n_blocks):
            start = rng.randrange(0, last_start + 1)
            chosen.extend(rows[start : start + length])
        sum_tau = sum(row["tau"] for row in chosen)
        if sum_tau <= 0:
            continue
        samples.append(sum(row["G"] for row in chosen) / sum_tau)
    if not samples:
        return None, None
    return _percentile(samples, 0.05), _percentile(samples, 0.95)

"""State of a ticker at the close of each session, and its robust z-scores.

Every number for session ``s`` uses sessions ``0..s`` only, so cutting the
series at ``s`` never changes the state at ``s``.
"""

from __future__ import annotations

import math
from statistics import median

from common.clock import Session

WINDOW_DD = 60
WINDOW_BOUNCE = 20
WINDOW_SLOPE = 20
WINDOW_RV = 20

FEATURES = ("dd_60", "bounce_20", "slope_20", "rv_20")
MAD_SCALE = 1.4826
MIN_LOG_RV = 1e-12


def realised_variance(session: Session) -> float:
    """Sum of squared minute log returns. The first minute starts from its own open."""
    previous = session.bars[0][0]
    total = 0.0
    for _open, _high, _low, close in session.bars:
        total += math.log(close / previous) ** 2
        previous = close
    return total


def _slope(closes: list[float]) -> float:
    ys = [math.log(value) for value in closes]
    n = len(ys)
    x_mean = (n - 1) / 2
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den


def raw_state(sessions: list[Session]) -> list[dict | None]:
    """One dict per session, ``None`` until ``WINDOW_DD`` sessions exist."""
    closes = [session.close for session in sessions]
    highs = [session.high for session in sessions]
    lows = [session.low for session in sessions]
    variances = [realised_variance(session) for session in sessions]
    out: list[dict | None] = []
    for s in range(len(sessions)):
        if s + 1 < max(WINDOW_DD, WINDOW_BOUNCE, WINDOW_SLOPE, WINDOW_RV):
            out.append(None)
            continue
        peak = max(highs[s + 1 - WINDOW_DD : s + 1])
        trough = min(lows[s + 1 - WINDOW_BOUNCE : s + 1])
        rv_window = variances[s + 1 - WINDOW_RV : s + 1]
        out.append(
            {
                "dd_60": 1.0 - closes[s] / peak,
                "bounce_20": closes[s] / trough - 1.0,
                "slope_20": _slope(closes[s + 1 - WINDOW_SLOPE : s + 1]),
                "rv_20": math.sqrt(sum(rv_window) / len(rv_window)),
            }
        )
    return out


def _feature_value(state: dict, name: str) -> float:
    value = state[name]
    if name == "rv_20":
        return math.log(max(value, MIN_LOG_RV))
    return value


def standardise(
    raw: list[dict | None], upto: int | None = None
) -> tuple[list[tuple[float, ...] | None], tuple[float, ...] | None, list[str]]:
    """Robust z-scores for sessions ``0..upto`` (default: the last one).

    Median and MAD come from those sessions only. A feature with MAD 0 is
    dropped (z = 0 everywhere) and named in the third return value.
    Returns ``(z_by_session, z_today, dropped_features)``.
    """
    last = len(raw) - 1 if upto is None else upto
    if last < 0:
        return [], None, []
    window = raw[: last + 1]
    present = [state for state in window if state is not None]
    scales: dict[str, tuple[float, float]] = {}
    dropped: list[str] = []
    for name in FEATURES:
        values = [_feature_value(state, name) for state in present]
        if not values:
            dropped.append(name)
            continue
        centre = median(values)
        mad = median(abs(value - centre) for value in values)
        if mad == 0:
            dropped.append(name)
            continue
        scales[name] = (centre, MAD_SCALE * mad)
    z_rows: list[tuple[float, ...] | None] = []
    for state in window:
        if state is None:
            z_rows.append(None)
            continue
        row = []
        for name in FEATURES:
            if name not in scales:
                row.append(0.0)
                continue
            centre, scale = scales[name]
            row.append((_feature_value(state, name) - centre) / scale)
        z_rows.append(tuple(row))
    return z_rows, z_rows[last], dropped

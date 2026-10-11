"""StopSimulator: the prod monitor's two nets, on minute bars."""

from __future__ import annotations

import math

from common.clock import MINUTES_PER_DAY, Session

# Mirrored from util/shell/fund/defense/scripts/prod_monitor.py
TICK = 0.05
GTT_DROP = 0.10
STOPLOSS_DROP = 0.05
MIN_TRIGGER_GAP = 0.005
CHECKPOINT_MIN = 5
SLIPPAGE = 0.001


def tick(price: float) -> float:
    return round(round(float(price) / TICK) * TICK, 2)


def trig(base: float, drop: float, ltp: float) -> float:
    return min(tick(base * (1 - drop)), tick(ltp * (1 - MIN_TRIGGER_GAP)))


def simulate(sessions: list[Session], d0: int, last: int) -> dict:
    """Replay an entry at the close of session ``d0``, censored at ``last`` if still open.

    ``last`` is the index of the latest session (T). The walk starts at ``d0 + 1``.
    """
    p0 = sessions[d0].close
    gtt = trig(p0, GTT_DROP, p0)
    for d in range(d0 + 1, last + 1):
        sl: float | None = None
        for m, (open_, _high, low, _close) in enumerate(sessions[d].bars):
            if m % CHECKPOINT_MIN == 0:
                ltp = open_
                gtt = max(gtt, trig(ltp, GTT_DROP, ltp))
                if sl is None:
                    sl = trig(p0, STOPLOSS_DROP, ltp)
                else:
                    sl = max(sl, trig(ltp, STOPLOSS_DROP, ltp))
            net = gtt if sl is None else max(gtt, sl)
            if open_ <= net:
                exit_px = open_ * (1 - SLIPPAGE)
                by = "sl" if sl is not None and sl >= gtt else "gtt"
            elif low <= net:
                exit_px = net * (1 - SLIPPAGE)
                by = "sl" if sl is not None and sl >= gtt else "gtt"
            else:
                continue
            tau = (d - d0 - 1) + (m + 1) / MINUTES_PER_DAY
            return {
                "d0": d0,
                "P0": p0,
                "G": math.log(exit_px / p0),
                "tau": tau,
                "censored": False,
                "by": by,
                "exit_d": d,
            }
    close_t = sessions[last].close
    return {
        "d0": d0,
        "P0": p0,
        "G": math.log(close_t / p0),
        "tau": float(last - d0),
        "censored": True,
        "by": None,
        "exit_d": None,
    }

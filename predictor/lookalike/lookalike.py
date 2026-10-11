"""Look-alike days: the per-day return of buying this ticker on days that looked like today."""

from __future__ import annotations

import json
from datetime import date, timedelta

from brokers.broker import BaseBroker
from common.cache import cached_entries, load_predictor_result, save_predictor_result
from common.clock import sessions_from_candles
from common.kernel import H_GRID, TARGET_NEFF, THIN_DISTANCE, distances, select_bandwidth
from common.state import FEATURES, WINDOW_BOUNCE, WINDOW_DD, WINDOW_RV, WINDOW_SLOPE, raw_state, standardise
from common.stats import BOOTSTRAP_DRAWS, compute
from predictor.baseline.baseline import HISTORY_YEARS, MIN_HISTORY_DAYS
from predictor.predictor import BasePredictor

NAME = "lookalike"
TOP_DATES = 5
THIN_NEFF = 10

# Anything here changes the answer, so it is part of the cached result's key.
PARAMS_KEY = json.dumps(
    {
        "grid": list(H_GRID),
        "target_neff": TARGET_NEFF,
        "thin_distance": THIN_DISTANCE,
        "features": list(FEATURES),
        "windows": [WINDOW_DD, WINDOW_BOUNCE, WINDOW_SLOPE, WINDOW_RV],
        "draws": BOOTSTRAP_DRAWS,
        "top_dates": TOP_DATES,
    },
    sort_keys=True,
)

DESCRIPTION = """\
Look-alike is the per-day return of buying this ticker on past days that looked like today.

Every past day is described by four numbers: how far the price is below its 60-day high, how far it has bounced off its 20-day low, its 20-day trend, and its 20-day volatility. Each is put on the ticker's own scale, and a past day gets a weight from how close its four numbers are to today's. The number is r_tw = weighted total log growth / weighted total days held, using the same replay of the GTT and daily stoploss as the baseline.

Read it next to the baseline. If the two are close, today looks ordinary and look-alike adds nothing. If it is clearly above or below, the history says this kind of day did better or worse than a random day. The range lo..hi keeps the chosen bandwidth fixed, so it is slightly narrower than it should be. status is thin when there are few look-alikes or today is unlike anything in the history, and the number then sits close to the baseline.
"""


def _null_result(instrument: str, status: str, **extras) -> dict:
    result = {
        "predictor": NAME,
        "instrument": instrument,
        "status": status,
        "r_tw": None,
        "r_mean": None,
        "sd": None,
        "score": None,
        "p_pos": None,
        "days": None,
        "p_1d": None,
        "n": 0,
        "neff": None,
        "lo": None,
        "hi": None,
        "share_gtt": None,
        "censored_share": None,
        "n_days": None,
        "h": None,
        "top_dates": [],
        "min_dist": None,
        "kish_ratio": None,
        "dropped_features": [],
        "state_today": None,
    }
    result.update(extras)
    return result


def build_result(
    instrument: str,
    rows: list[dict],
    z_rows: list[tuple[float, ...] | None],
    z_today: tuple[float, ...],
    *,
    ticker: str,
    as_of: str,
    days: list[str],
    state_today: dict | None,
    dropped: list[str],
    grid: tuple[float, ...] = H_GRID,
    target_neff: float = TARGET_NEFF,
) -> dict:
    """Weight the entry rows by similarity to ``z_today`` and summarise them.

    ``z_rows`` and ``days`` are indexed by session, like each row's ``d0``.
    """
    common = {"dropped_features": list(dropped), "state_today": state_today}
    pool = [
        row
        for row in rows
        if row["eligible"] and 0 <= row["d0"] < len(z_rows) and z_rows[row["d0"]] is not None
    ]
    if not pool:
        return _null_result(instrument, "empty", **common)
    dists = distances([z_rows[row["d0"]] for row in pool], z_today)
    chosen = select_bandwidth([row["tau"] for row in pool], dists, grid, target_neff)
    if chosen["empty"]:
        return _null_result(instrument, "empty", h=chosen["h"], min_dist=chosen["min_dist"], **common)

    weighted = [dict(row, w=k) for row, k in zip(pool, chosen["k"])]
    stats = compute(weighted, ticker=f"{ticker}|{NAME}", as_of=as_of)
    if stats["n"] == 0:
        return _null_result(instrument, "empty", h=chosen["h"], min_dist=chosen["min_dist"], **common)

    if stats["neff"] < THIN_NEFF or not chosen["reached"] or chosen["min_dist"] > THIN_DISTANCE:
        status = "thin"
    else:
        status = "ok"

    total_w = sum(row["w"] for row in weighted)
    finished = [row for row in weighted if not row["censored"]]
    share_gtt = None
    finished_w = sum(row["w"] for row in finished)
    if finished_w > 0:
        share_gtt = sum(row["w"] for row in finished if row["by"] == "gtt") / finished_w
    censored_share = sum(row["w"] for row in weighted if row["censored"]) / total_w
    n_days = len({row["d0"] for row in weighted if row["w"] > 0})

    ranked = sorted((row for row in weighted if row["w"] > 0), key=lambda row: (row["w"], row["d0"]), reverse=True)
    top_dates = [{"date": days[row["d0"]], "k": row["w"], "r": row["r"]} for row in ranked[:TOP_DATES]]

    return {
        "predictor": NAME,
        "instrument": instrument,
        "status": status,
        "r_tw": stats["r_tw"],
        "r_mean": stats["r_mean"],
        "sd": stats["sd"],
        "score": stats["score"],
        "p_pos": stats["p_pos"],
        "days": stats["days"],
        "p_1d": stats["p_1d"],
        "n": stats["n"],
        "neff": stats["neff"],
        "lo": stats["lo"],
        "hi": stats["hi"],
        "share_gtt": share_gtt,
        "censored_share": censored_share,
        "n_days": n_days,
        "h": chosen["h"],
        "top_dates": top_dates,
        "min_dist": chosen["min_dist"],
        "kish_ratio": chosen["kish_ratio"],
        **common,
    }


class LookalikePredictor(BasePredictor):
    name = NAME

    def description(self) -> str:
        return DESCRIPTION

    def predict(self, instrument: str) -> dict:
        text = instrument.strip().upper()
        exchange, _, ticker = text.partition(":")
        if not exchange or not ticker:
            raise ValueError(f"instrument must be EXCHANGE:TICKER, got {instrument!r}")
        broker = BaseBroker.default_broker(exchange)
        end = date.today()
        start = end - timedelta(days=365 * HISTORY_YEARS + 20)
        candles = broker.fetch_minute(ticker, start, end)
        sessions = sessions_from_candles(candles)
        if not sessions:
            return _null_result(text, "insufficient_history")
        last_day = sessions[-1].day.isoformat()
        cached = load_predictor_result(self.name, text, last_day, PARAMS_KEY)
        if cached is not None:
            return cached
        if len(sessions) < MIN_HISTORY_DAYS:
            result = _null_result(text, "insufficient_history")
        else:
            raw = raw_state(sessions)
            z_rows, z_today, dropped = standardise(raw)
            if z_today is None:
                result = _null_result(text, "insufficient_history")
            else:
                result = build_result(
                    text,
                    cached_entries(text, sessions),
                    z_rows,
                    z_today,
                    ticker=ticker,
                    as_of=last_day,
                    days=[session.day.isoformat() for session in sessions],
                    state_today=raw[-1],
                    dropped=dropped,
                )
        save_predictor_result(self.name, text, last_day, result, PARAMS_KEY)
        return result


BasePredictor.register(NAME, LookalikePredictor)

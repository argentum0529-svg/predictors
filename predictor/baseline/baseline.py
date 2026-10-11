"""Baseline: the per-day return of buying this ticker on any day."""

from __future__ import annotations

from datetime import date, timedelta

from brokers.broker import BaseBroker
from common.cache import cached_entries, load_predictor_result, save_predictor_result
from common.clock import sessions_from_candles
from common.stats import compute
from predictor.predictor import BasePredictor

MIN_HISTORY_DAYS = 250
HISTORY_YEARS = 3

DESCRIPTION = """\
Baseline is the per-day return of buying this ticker at the close and letting the monitor sell it, averaged over every past day in its own history.

The number is r_tw = total log growth / total days held. A trade that lasted an afternoon cannot outvote a trade that held the money for weeks, and its loss still counts in the total. It ignores what the stock looks like today, so it is nearly the same at a peak and at a bottom.

It is the yardstick. Another predictor is only saying something about today when its number differs from this one. For a stock that moves at random, the nets do not change this per-day return: it equals the stock's own drift.
"""


def _null_result(instrument: str, status: str) -> dict:
    return {
        "predictor": "baseline",
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
    }


class BaselinePredictor(BasePredictor):
    name = "baseline"

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
        cached = load_predictor_result(self.name, text, last_day)
        if cached is not None:
            return cached
        if len(sessions) < MIN_HISTORY_DAYS:
            result = _null_result(text, "insufficient_history")
        else:
            result = self._from_rows(text, cached_entries(text, sessions), ticker, last_day)
        save_predictor_result(self.name, text, last_day, result)
        return result

    def _from_rows(self, instrument: str, rows: list[dict], ticker: str, as_of: str) -> dict:
        stats = compute(rows, ticker=ticker, as_of=as_of)
        eligible = [row for row in rows if row["eligible"]]
        finished = [row for row in eligible if not row["censored"]]
        if not eligible:
            status = "empty"
        elif stats["neff"] < 10:
            status = "thin"
        else:
            status = "ok"
        share_gtt = None
        if finished:
            share_gtt = sum(1 for row in finished if row["by"] == "gtt") / len(finished)
        censored_share = None
        n_days = None
        if eligible:
            censored_share = sum(1 for row in eligible if row["censored"]) / len(eligible)
            n_days = len({row["d0"] for row in eligible})
        return {
            "predictor": self.name,
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
        }


BasePredictor.register("baseline", BaselinePredictor)

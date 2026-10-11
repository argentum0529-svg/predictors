"""Entry-table cache: same-day hit, and the next session only replays open trades."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from common.cache import cached_entries, load_predictor_result, save_predictor_result
from common.clock import Session
from common.entries import build_entries


def _day(day: date, price: float = 100.0) -> Session:
    return Session(day=day, bars=[(price, price, price, price)] * 375)


def _span(n: int) -> list[Session]:
    start = date(2024, 1, 1)
    return [_day(start + timedelta(days=i)) for i in range(n)]


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("PREDICTORS_CACHE")
        os.environ["PREDICTORS_CACHE"] = self._dir.name

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("PREDICTORS_CACHE", None)
        else:
            os.environ["PREDICTORS_CACHE"] = self._prev
        self._dir.cleanup()

    def test_same_day_does_not_replay(self) -> None:
        sessions = _span(66)
        first = cached_entries("NSE:HAL", sessions)
        with patch("common.cache.build_entries", side_effect=AssertionError("replayed")):
            second = cached_entries("NSE:HAL", sessions)
        self.assertEqual([row["d0"] for row in first], [row["d0"] for row in second])
        self.assertTrue((Path(self._dir.name) / "entries").is_dir())

    def test_next_day_keeps_finished_trades(self) -> None:
        sessions = _span(66)
        longer = sessions + [_day(sessions[-1].day + timedelta(days=1))]
        finished = {
            "d0": 60,
            "P0": 100.0,
            "G": 0.1,
            "tau": 2.0,
            "censored": False,
            "by": "gtt",
            "exit_d": 62,
            "r": 0.05,
            "eligible": True,
        }
        open_row = {
            "d0": 64,
            "P0": 100.0,
            "G": 0.0,
            "tau": 1.0,
            "censored": True,
            "by": None,
            "exit_d": None,
            "r": 0.0,
            "eligible": False,
        }
        with patch("common.cache._read", return_value={
            "last_day": sessions[-1].day.isoformat(),
            "n_sessions": len(sessions),
            "rows": [finished, open_row],
        }):
            seen: dict[str, list[int]] = {}

            def fake_build(sessions_arg, d0s=None):
                seen["d0s"] = list(d0s)
                return build_entries(sessions_arg, d0s)

            with patch("common.cache.build_entries", side_effect=fake_build):
                rows = cached_entries("NSE:HAL", longer)
        self.assertEqual(seen["d0s"], [64, 65])
        kept = next(row for row in rows if row["d0"] == 60)
        self.assertEqual(kept["G"], 0.1)
        self.assertEqual(kept["exit_d"], 62)
        self.assertIn(65, [row["d0"] for row in rows])

    def test_predictor_result_roundtrip(self) -> None:
        save_predictor_result("baseline", "NSE:HAL", "2026-10-10", {"predictor": "baseline", "r_tw": 0.01})
        loaded = load_predictor_result("baseline", "NSE:HAL", "2026-10-10")
        self.assertEqual(loaded["r_tw"], 0.01)
        self.assertIsNone(load_predictor_result("baseline", "NSE:HAL", "2026-10-11"))


if __name__ == "__main__":
    unittest.main()

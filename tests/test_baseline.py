"""Baseline structure: simulator, stats, registry, and the description command."""

from __future__ import annotations

import json
import math
import subprocess
import unittest
from datetime import date, timedelta
from pathlib import Path

from common.clock import Session
from common.entries import build_entries
from common.simulator import simulate
from common.stats import compute
from predictor import BasePredictor
from predictor.baseline.baseline import BaselinePredictor

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "predictor"


def _day(day: date, price: float = 100.0, minutes: dict | None = None) -> Session:
    bars = []
    for m in range(375):
        if minutes and m in minutes:
            bars.append(minutes[m])
        else:
            bars.append((price, price, price, price))
    return Session(day=day, bars=bars)


def _span(n: int, price: float = 100.0) -> list[Session]:
    start = date(2024, 1, 1)
    return [_day(start + timedelta(days=i), price) for i in range(n)]


class StatsTests(unittest.TestCase):
    def test_three_trade_option2(self) -> None:
        rows = [
            {"G": -0.0513, "tau": 0.3, "censored": False, "eligible": True, "d0": 0},
            {"G": 0.1823, "tau": 60.0, "censored": False, "eligible": True, "d0": 1},
            {"G": 0.0953, "tau": 30.0, "censored": False, "eligible": True, "d0": 2},
        ]
        stats = compute(rows, ticker="HAL", as_of="2026-10-10", draws=20)
        self.assertAlmostEqual(stats["r_tw"], 0.2263 / 90.3)
        r_mean = (-0.0513 / 0.3 + 0.1823 / 60 + 0.0953 / 30) / 3
        self.assertAlmostEqual(stats["r_mean"], r_mean)
        self.assertLess(stats["r_mean"], -0.05)
        again = compute(rows, ticker="HAL", as_of="2026-10-10", draws=20)
        self.assertEqual(stats["lo"], again["lo"])
        self.assertEqual(stats["hi"], again["hi"])


class SimulatorTests(unittest.TestCase):
    def test_gtt_after_a_rise(self) -> None:
        start = date(2024, 1, 1)
        sessions = [
            _day(start, 100),
            _day(start + timedelta(days=1), 110),
            _day(start + timedelta(days=2), 120),
            _day(start + timedelta(days=3), 130),
            _day(start + timedelta(days=4), 130, minutes={0: (130, 130, 100, 130)}),
        ]
        trade = simulate(sessions, 0, 4)
        exit_px = 100 * math.exp(trade["G"])
        self.assertFalse(trade["censored"])
        self.assertEqual(trade["by"], "gtt")
        self.assertEqual(trade["exit_d"], 4)
        self.assertAlmostEqual(exit_px, 117 * (1 - 0.001), places=5)

    def test_intraday_stop(self) -> None:
        start = date(2024, 1, 1)
        # Checkpoint at minute 5 sees 110; a later minute drops 5.2% from that.
        drop = 110 * (1 - 0.052)
        sessions = [
            _day(start, 100),
            _day(
                start + timedelta(days=1),
                100,
                minutes={
                    5: (110, 110, 110, 110),
                    6: (110, 110, drop, drop),
                },
            ),
        ]
        trade = simulate(sessions, 0, 1)
        self.assertFalse(trade["censored"])
        self.assertEqual(trade["by"], "sl")
        self.assertLess(trade["tau"], 1)

    def test_gap_open_exits_at_the_open(self) -> None:
        start = date(2024, 1, 1)
        sessions = [
            _day(start, 100),
            _day(start + timedelta(days=1), 100, minutes={0: (80, 80, 80, 80)}),
        ]
        trade = simulate(sessions, 0, 1)
        exit_px = 100 * math.exp(trade["G"])
        self.assertAlmostEqual(exit_px, 80 * (1 - 0.001), places=5)
        self.assertNotAlmostEqual(exit_px, 90, places=2)

    def test_expired_intraday_stop_does_not_carry(self) -> None:
        start = date(2024, 1, 1)
        sessions = [
            _day(start, 100),
            _day(start + timedelta(days=1), 130),
            # 120 is below yesterday's trailed 5% stop (~123.5) and above the GTT (117).
            _day(start + timedelta(days=2), 130, minutes={0: (130, 130, 120, 125)}),
        ]
        trade = simulate(sessions, 0, 2)
        self.assertTrue(trade["censored"])

    def test_flat_path_is_censored(self) -> None:
        sessions = _span(4, 100)
        trade = simulate(sessions, 0, 3)
        self.assertTrue(trade["censored"])
        self.assertIsNone(trade["exit_d"])
        self.assertAlmostEqual(trade["G"], 0.0)
        self.assertEqual(trade["tau"], 3)


class EntryTests(unittest.TestCase):
    def test_recent_censored_rows_are_ineligible(self) -> None:
        sessions = _span(66, 100)
        rows = build_entries(sessions)
        eligible = [row for row in rows if row["eligible"]]
        self.assertEqual(len(eligible), 1)
        self.assertGreaterEqual(eligible[0]["tau"], 5)


class CommandTests(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertIsInstance(BasePredictor.get("baseline"), BaselinePredictor)

    def test_description_command(self) -> None:
        result = subprocess.run([str(SCRIPT), "baseline", "description"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("yardstick", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_unknown_predictor_is_json(self) -> None:
        result = subprocess.run([str(SCRIPT), "nope", "description"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        body = json.loads(result.stdout)
        self.assertIn("error", body)

    def test_usage(self) -> None:
        result = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("description", result.stderr)
        self.assertIn("predictor", result.stderr)


if __name__ == "__main__":
    unittest.main()

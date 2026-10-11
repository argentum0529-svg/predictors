"""Look-alike predictor: state, weighted stats, kernel, bandwidth, statuses, CLI."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

from brokers.broker import BaseBroker
from common.cache import config_hash, load_predictor_result, save_predictor_result
from common.clock import Session
from common.kernel import H_GRID, distances, select_bandwidth
from common.state import FEATURES, raw_state, realised_variance, standardise
from common.stats import compute, effective_sample_size
from predictor import BasePredictor
from predictor.lookalike.lookalike import LookalikePredictor, build_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "predictor"
START = date(2023, 1, 2)


def _flat(day: date, price: float) -> Session:
    return Session(day=day, bars=[(price, price, price, price)] * 375)


def _ramp(n: int, growth: float = 1.01) -> list[Session]:
    return [_flat(START + timedelta(days=i), 100.0 * growth**i) for i in range(n)]


def _walk(n: int, seed: int = 1) -> list[Session]:
    """Random sessions with overnight gaps, so the state varies from day to day."""
    rng = random.Random(seed)
    price = 100.0
    sessions = []
    for i in range(n):
        price *= 1 + rng.gauss(0, 0.01)
        bars = []
        for _ in range(375):
            open_ = price
            close = open_ * (1 + rng.gauss(0, 0.0005))
            high = max(open_, close) * (1 + abs(rng.gauss(0, 0.0002)))
            low = min(open_, close) * (1 - abs(rng.gauss(0, 0.0002)))
            bars.append((open_, high, low, close))
            price = close
        sessions.append(Session(day=START + timedelta(days=i), bars=bars))
    return sessions


def _candles(sessions: list[Session]) -> list[list]:
    out = []
    for session in sessions:
        base = datetime(session.day.year, session.day.month, session.day.day, 9, 15)
        for m, (open_, high, low, close) in enumerate(session.bars):
            out.append([base + timedelta(minutes=m), open_, high, low, close, 0])
    return out


def _row(d0: int, g: float, tau: float, *, by: str = "gtt", censored: bool = False) -> dict:
    return {
        "d0": d0,
        "G": g,
        "tau": tau,
        "r": g / tau,
        "censored": censored,
        "by": None if censored else by,
        "eligible": True,
    }


class StateTests(unittest.TestCase):
    def test_truncating_does_not_change_state(self) -> None:
        sessions = _walk(90)
        full = raw_state(sessions)
        for d in (59, 70, 85):
            self.assertEqual(raw_state(sessions[: d + 1])[d], full[d])

    def test_no_state_before_warmup(self) -> None:
        raw = raw_state(_walk(61))
        self.assertTrue(all(state is None for state in raw[:59]))
        self.assertIsNotNone(raw[59])

    def test_hand_computed_ramp(self) -> None:
        raw = raw_state(_ramp(80))
        state = raw[70]
        self.assertAlmostEqual(state["dd_60"], 0.0, places=12)
        self.assertAlmostEqual(state["bounce_20"], 1.01**19 - 1, places=9)
        self.assertAlmostEqual(state["slope_20"], math.log(1.01), places=9)
        # Prices are flat inside each day, so the overnight jumps add no variance.
        self.assertAlmostEqual(state["rv_20"], 0.0, places=12)

    def test_realised_variance_excludes_the_overnight_gap(self) -> None:
        session = _flat(START, 150.0)
        self.assertEqual(realised_variance(session), 0.0)
        bars = [(100.0, 100.0, 100.0, 100.0)] * 375
        bars[0] = (100.0, 101.0, 100.0, 101.0)
        expected = math.log(1.01) ** 2 + math.log(100 / 101) ** 2
        self.assertAlmostEqual(realised_variance(Session(day=START, bars=bars)), expected)

    def test_constant_features_are_dropped(self) -> None:
        raw = raw_state(_ramp(80))
        z_rows, z_today, dropped = standardise(raw)
        # The slope of a pure ramp differs only by float noise, so it may or may not be dropped.
        for name in ("dd_60", "bounce_20", "rv_20"):
            self.assertIn(name, dropped)
        self.assertEqual([z_today[0], z_today[1], z_today[3]], [0.0, 0.0, 0.0])
        self.assertIsNone(z_rows[0])

    def test_z_scores_use_data_up_to_today_only(self) -> None:
        sessions = _walk(90)
        raw = raw_state(sessions)
        _, z_before, dropped = standardise(raw, upto=80)
        self.assertEqual(dropped, [])
        spike = _flat(START + timedelta(days=90), sessions[-1].close * 3)
        raw_after = raw_state(sessions + [spike])
        _, z_after, _ = standardise(raw_after, upto=80)
        self.assertEqual(z_before, z_after)

    def test_median_is_the_centre(self) -> None:
        raw = raw_state(_walk(90))
        z_rows, _, _ = standardise(raw)
        present = [z for z in z_rows if z is not None]
        for i in range(len(FEATURES)):
            ordered = sorted(z[i] for z in present)
            self.assertAlmostEqual(ordered[len(ordered) // 2], 0.0, delta=0.05)


class WeightedStatsTests(unittest.TestCase):
    def test_weight_two_equals_duplicate_row(self) -> None:
        a = _row(0, 0.05, 3.0)
        b = _row(1, -0.02, 1.0)
        doubled = compute([dict(a, w=2.0), dict(b, w=1.0)], ticker="X", as_of="d", draws=5)
        duplicated = compute([a, dict(a, d0=2), b], ticker="X", as_of="d", draws=5)
        for key in ("r_tw", "r_mean", "sd", "p_pos", "days"):
            self.assertAlmostEqual(doubled[key], duplicated[key])

    def test_effective_sample_size(self) -> None:
        self.assertAlmostEqual(effective_sample_size([1.0] * 40, [2.0] * 40), 20.0)
        self.assertEqual(effective_sample_size([0.0, 0.0], [1.0, 1.0]), 0.0)
        # Uneven weights are worth fewer trades than equal ones.
        self.assertLess(effective_sample_size([1.0, 0.01, 0.01], [1.0] * 3), 3.0)

    def test_missing_weight_means_one(self) -> None:
        rows = [_row(i, 0.01, 1.0) for i in range(10)]
        plain = compute(rows, ticker="X", as_of="d", draws=5)
        explicit = compute([dict(row, w=1.0) for row in rows], ticker="X", as_of="d", draws=5)
        self.assertEqual(plain["r_tw"], explicit["r_tw"])
        self.assertEqual(plain["neff"], explicit["neff"])


class KernelTests(unittest.TestCase):
    def test_distances(self) -> None:
        self.assertEqual(distances([(3.0, 4.0, 0.0, 0.0)], (0.0, 0.0, 0.0, 0.0)), [5.0])

    def test_picks_smallest_bandwidth_with_enough_evidence(self) -> None:
        dists = [i * 0.05 for i in range(300)]
        taus = [1.0] * 300
        chosen = select_bandwidth(taus, dists)
        expected = None
        for h in sorted(H_GRID):
            weights = [math.exp(-d * d / (2 * h * h)) for d in dists]
            if effective_sample_size(weights, taus) >= 20:
                expected = h
                break
        self.assertIsNotNone(expected)
        self.assertEqual(chosen["h"], expected)
        self.assertTrue(chosen["reached"])
        self.assertGreater(expected, min(H_GRID))

    def test_falls_back_to_largest_bandwidth(self) -> None:
        chosen = select_bandwidth([1.0] * 10, [0.0] * 10)
        self.assertEqual(chosen["h"], max(H_GRID))
        self.assertFalse(chosen["reached"])
        self.assertFalse(chosen["empty"])

    def test_underflow_is_empty(self) -> None:
        chosen = select_bandwidth([1.0] * 5, [1000.0] * 5)
        self.assertTrue(chosen["empty"])


def _build(rows, z_rows, z_today, **kwargs):
    kwargs.setdefault("days", [f"d{i}" for i in range(len(z_rows))])
    return build_result(
        "NSE:X",
        rows,
        z_rows,
        z_today,
        ticker="X",
        as_of="2024-01-01",
        state_today=None,
        dropped=[],
        **kwargs,
    )


class BuildResultTests(unittest.TestCase):
    def test_huge_bandwidth_is_the_baseline(self) -> None:
        rng = random.Random(7)
        rows = [_row(i, rng.gauss(0.01, 0.03), rng.uniform(1, 10)) for i in range(300)]
        z_rows = [tuple(rng.gauss(0, 1) for _ in range(4)) for _ in rows]
        result = _build(rows, z_rows, (0.0, 0.0, 0.0, 0.0), grid=(1e6,))
        baseline = compute(rows, ticker="X", as_of="2024-01-01", draws=5)
        self.assertAlmostEqual(result["r_tw"], baseline["r_tw"], places=9)
        self.assertAlmostEqual(result["neff"], baseline["neff"], places=6)

    def test_exact_match_with_tiny_bandwidth(self) -> None:
        rows = [_row(i, 0.01 * (i + 1), 2.0 + i) for i in range(10)]
        z_rows = [(float(i), 0.0, 0.0, 0.0) for i in range(10)]
        result = _build(rows, z_rows, z_rows[4], grid=(0.01,), target_neff=0.01)
        self.assertAlmostEqual(result["r_tw"], rows[4]["G"] / rows[4]["tau"], places=12)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["h"], 0.01)

    def test_smooth_function_is_recovered(self) -> None:
        n = 600
        xs = [-2 + 4 * i / (n - 1) for i in range(n)]
        rows = [_row(i, 0.002 * x, 1.0) for i, x in enumerate(xs)]
        z_rows = [(x, 0.0, 0.0, 0.0) for x in xs]
        result = _build(rows, z_rows, (0.5, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(result["r_tw"], 0.002 * 0.5, delta=1e-4)
        self.assertEqual(result["status"], "ok")

    def test_far_from_history_is_thin(self) -> None:
        rows = [_row(i, 0.01, 1.0) for i in range(100)]
        z_rows = [(10.0, 0.0, 0.0, 0.0)] * 100
        result = _build(rows, z_rows, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["status"], "thin")
        self.assertGreater(result["min_dist"], 3)

    def test_underflow_is_empty(self) -> None:
        rows = [_row(i, 0.01, 1.0) for i in range(20)]
        z_rows = [(1000.0, 0.0, 0.0, 0.0)] * 20
        result = _build(rows, z_rows, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["r_tw"])

    def test_no_eligible_rows_is_empty(self) -> None:
        rows = [dict(_row(0, 0.01, 1.0), eligible=False)]
        result = _build(rows, [(0.0, 0.0, 0.0, 0.0)], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["status"], "empty")

    def test_few_rows_is_thin(self) -> None:
        rows = [_row(i, 0.01, 1.0) for i in range(5)]
        z_rows = [(0.0, 0.0, 0.0, 0.0)] * 5
        result = _build(rows, z_rows, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["status"], "thin")
        self.assertEqual(result["h"], max(H_GRID))

    def test_deterministic(self) -> None:
        rng = random.Random(3)
        rows = [_row(i, rng.gauss(0.01, 0.03), rng.uniform(1, 10)) for i in range(200)]
        z_rows = [tuple(rng.gauss(0, 1) for _ in range(4)) for _ in rows]
        first = _build(rows, z_rows, (0.1, 0.2, 0.3, 0.4))
        second = _build(rows, z_rows, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))

    def test_top_dates(self) -> None:
        rows = [_row(1, 0.02, 2.0), _row(2, -0.01, 4.0), _row(3, 0.03, 3.0), _row(4, 0.0, 1.0)]
        z_rows = [None, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.0, 0.0)]
        days = ["a", "b", "c", "d", "e"]
        result = _build(rows, z_rows, (0.0, 0.0, 0.0, 0.0), days=days, grid=(1.0,), target_neff=0.001)
        top = result["top_dates"]
        # The two exact matches tie at k = 1 and the newer day comes first.
        self.assertEqual([entry["date"] for entry in top[:3]], ["c", "b", "d"])
        self.assertAlmostEqual(top[0]["k"], 1.0)
        self.assertAlmostEqual(top[2]["k"], math.exp(-0.5))
        self.assertAlmostEqual(top[0]["r"], -0.01 / 4.0)


class PredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"PREDICTORS_CACHE": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _predict(self, sessions: list[Session]) -> dict:
        broker = mock.Mock()
        broker.fetch_minute.return_value = _candles(sessions)
        with mock.patch.object(BaseBroker, "default_broker", return_value=broker):
            return LookalikePredictor().predict("nse:abc")

    def test_registry(self) -> None:
        self.assertIsInstance(BasePredictor.get("lookalike"), LookalikePredictor)

    def test_no_candles_is_insufficient(self) -> None:
        broker = mock.Mock()
        broker.fetch_minute.return_value = []
        with mock.patch.object(BaseBroker, "default_broker", return_value=broker):
            result = LookalikePredictor().predict("NSE:ABC")
        self.assertEqual(result["status"], "insufficient_history")
        self.assertIsNone(result["r_tw"])

    def test_short_history_is_insufficient(self) -> None:
        result = self._predict(_walk(100))
        self.assertEqual(result["status"], "insufficient_history")
        self.assertEqual(result["instrument"], "NSE:ABC")

    def test_bad_instrument(self) -> None:
        with self.assertRaises(ValueError):
            LookalikePredictor().predict("ABC")

    def test_full_run_is_json_and_cached(self) -> None:
        sessions = _walk(270, seed=5)
        result = self._predict(sessions)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["predictor"], "lookalike")
        self.assertIn(result["status"], {"ok", "thin"})
        self.assertIsNotNone(result["r_tw"])
        self.assertLessEqual(result["lo"], result["hi"])
        self.assertLessEqual(len(result["top_dates"]), 5)
        self.assertIsNotNone(result["state_today"])
        last_day = sessions[-1].day.isoformat()
        from predictor.lookalike.lookalike import PARAMS_KEY

        self.assertEqual(load_predictor_result("lookalike", "NSE:ABC", last_day, PARAMS_KEY), result)

    def test_result_cache_is_keyed_by_parameters(self) -> None:
        save_predictor_result("lookalike", "NSE:ABC", "2024-01-01", {"a": 1}, "params-a")
        self.assertEqual(load_predictor_result("lookalike", "NSE:ABC", "2024-01-01", "params-a"), {"a": 1})
        self.assertIsNone(load_predictor_result("lookalike", "NSE:ABC", "2024-01-01", "params-b"))
        self.assertIsNone(load_predictor_result("lookalike", "NSE:ABC", "2024-01-01"))

    def test_entry_cache_key_is_unchanged_without_extra(self) -> None:
        self.assertEqual(config_hash(), config_hash(""))
        self.assertNotEqual(config_hash(), config_hash("x"))


class CommandTests(unittest.TestCase):
    def test_description_command(self) -> None:
        result = subprocess.run([str(SCRIPT), "lookalike", "description"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("look-alike", result.stdout.lower())
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()

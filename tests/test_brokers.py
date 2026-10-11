"""Registry and local-curl vs ssh-curl."""

from __future__ import annotations

import socket
import subprocess
import unittest
from datetime import date
from unittest.mock import patch

import brokers
from brokers.base import BrokerClient, BrokerError
from brokers.broker import BaseBroker
from brokers.zerodha import ZerodhaBroker


class RegistryTests(unittest.TestCase):
    def test_default_broker_is_zerodha(self) -> None:
        broker = BaseBroker.default_broker("nse")
        self.assertIsInstance(broker, ZerodhaBroker)
        self.assertEqual(broker.exchanges(), ["nse"])

    def test_base_exchanges_empty(self) -> None:
        self.assertEqual(BaseBroker().exchanges(), [])

    def test_unknown_exchange(self) -> None:
        with self.assertRaises(KeyError):
            BaseBroker.default_broker("nyse")

    def test_package_exports_registry(self) -> None:
        self.assertIsInstance(brokers.BaseBroker.default_broker("NSE"), brokers.ZerodhaBroker)


class CurlTests(unittest.TestCase):
    def test_matching_hostname_curls_locally(self) -> None:
        client = BrokerClient("argentum", 8785, timeout=120)
        captured: dict[str, list[str]] = {}

        def fake_run(argv, capture_output, text):
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout='{"ok": true}', stderr="")

        with patch.object(socket, "gethostname", return_value="Argentum.local"), patch.object(
            subprocess, "run", side_effect=fake_run
        ):
            body = client.get("/brokers/zerodha/health")
        self.assertEqual(body, {"ok": True})
        self.assertEqual(captured["argv"][0], "curl")
        self.assertNotIn("ssh", captured["argv"])
        self.assertIn("http://127.0.0.1:8785/brokers/zerodha/health", captured["argv"])

    def test_other_hostname_sshs_then_curls(self) -> None:
        client = BrokerClient("argentum", 8785)
        captured: dict[str, list[str]] = {}

        def fake_run(argv, capture_output, text):
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

        with patch.object(socket, "gethostname", return_value="laptop"), patch.object(
            subprocess, "run", side_effect=fake_run
        ):
            client.get("/brokers/zerodha/minute-candles", {"instrument": "NSE:HAL", "from": "20261001", "to": "20261010"})
        argv = captured["argv"]
        self.assertEqual(argv[:4], ["ssh", "-o", "BatchMode=yes", "argentum"])
        remote = argv[4]
        self.assertIn("curl", remote)
        self.assertIn("http://127.0.0.1:8785/brokers/zerodha/minute-candles?", remote)
        self.assertIn("instrument=NSE%3AHAL", remote)

    def test_curl_failure_raises(self) -> None:
        client = BrokerClient("argentum", 8785)

        def fake_run(argv, capture_output, text):
            return subprocess.CompletedProcess(argv, 7, stdout="", stderr="connection refused")

        with patch.object(socket, "gethostname", return_value="argentum"), patch.object(
            subprocess, "run", side_effect=fake_run
        ):
            with self.assertRaises(BrokerError) as raised:
                client.get("/health")
        self.assertIn("connection refused", str(raised.exception))


class FetchMinuteTests(unittest.TestCase):
    def test_fetch_minute_query(self) -> None:
        broker = ZerodhaBroker()
        seen: dict[str, object] = {}

        def fake_get(path, params=None):
            seen["path"] = path
            seen["params"] = params
            return {"candles": [["2026-10-01T09:15:00+05:30", 1, 2, 0.5, 1.5, 10]]}

        with patch.object(socket, "gethostname", return_value="argentum"):
            client = broker.client
        client.get = fake_get  # type: ignore[method-assign]
        with patch.object(ZerodhaBroker, "client", property(lambda self: client)):
            candles = broker.fetch_minute("hal", date(2026, 10, 1), date(2026, 10, 10))
        self.assertEqual(seen["path"], "/brokers/zerodha/minute-candles")
        self.assertEqual(
            seen["params"],
            {"instrument": "NSE:HAL", "from": "20261001", "to": "20261010"},
        )
        self.assertEqual(len(candles), 1)

    def test_fetch_minute_keeps_identifier(self) -> None:
        broker = ZerodhaBroker()
        seen: dict[str, object] = {}

        def fake_get(path, params=None):
            seen["params"] = params
            return {"candles": []}

        client = broker.client
        client.get = fake_get  # type: ignore[method-assign]
        with patch.object(ZerodhaBroker, "client", property(lambda self: client)):
            broker.fetch_minute("NSE:BEL", date(2026, 1, 1), date(2026, 1, 2))
        self.assertEqual(seen["params"]["instrument"], "NSE:BEL")  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()

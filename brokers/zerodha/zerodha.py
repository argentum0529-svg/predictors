"""Zerodha as the default broker for NSE. Data comes from the brokers service."""

from __future__ import annotations

from datetime import date

from brokers.base import BrokerClient
from brokers.broker import BaseBroker


class ZerodhaBroker(BaseBroker):
    """Curl the brokers service. No Kite token is stored here."""

    HOST = "argentum"
    PORT = 8785

    def exchanges(self) -> list[str]:
        return ["nse"]

    @property
    def client(self) -> BrokerClient:
        return BrokerClient(self.HOST, self.PORT)

    def fetch_minute(self, ticker: str, start: date, end: date) -> list:
        """One-minute candles ``[[ts, open, high, low, close, volume], ...]``.

        ``ticker`` is an NSE symbol (``HAL``) or an identifier (``NSE:HAL``).
        """
        symbol = ticker.strip().upper()
        instrument = symbol if ":" in symbol else f"NSE:{symbol}"
        body = self.client.get(
            "/brokers/zerodha/minute-candles",
            {
                "instrument": instrument,
                "from": start.strftime("%Y%m%d"),
                "to": end.strftime("%Y%m%d"),
            },
        )
        candles = body.get("candles") or []
        if not isinstance(candles, list):
            raise TypeError("minute-candles response field 'candles' was not a list")
        return candles


BaseBroker.register("nse", ZerodhaBroker)

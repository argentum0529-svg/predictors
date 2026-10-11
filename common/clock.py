"""Sessions and daily bars from one-minute candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

MINUTES_PER_DAY = 375
SESSION_OPEN_MINUTES = 9 * 60 + 15  # 09:15


@dataclass
class Session:
    day: date
    bars: list[tuple[float, float, float, float]]  # open, high, low, close; length 375

    @property
    def open(self) -> float:
        return self.bars[0][0]

    @property
    def close(self) -> float:
        return self.bars[-1][3]

    @property
    def high(self) -> float:
        return max(bar[1] for bar in self.bars)

    @property
    def low(self) -> float:
        return min(bar[2] for bar in self.bars)


def _parsed_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Kite sometimes emits +0530 without a colon.
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _sane(open_: float, high: float, low: float, close: float) -> bool:
    if min(open_, high, low, close) <= 0:
        return False
    return low <= min(open_, close) and max(open_, close) <= high


def sessions_from_candles(candles: list) -> list[Session]:
    """Valid NSE sessions only: exactly 375 minute bars from 09:15 through 15:29."""
    by_day: dict[date, list[tuple[datetime, float, float, float, float]]] = {}
    for candle in candles:
        if not candle or len(candle) < 5:
            continue
        when = _parsed_time(candle[0])
        if when is None:
            continue
        try:
            open_, high, low, close = (float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]))
        except (TypeError, ValueError):
            continue
        if not _sane(open_, high, low, close):
            continue
        by_day.setdefault(when.date(), []).append((when, open_, high, low, close))

    sessions: list[Session] = []
    for day in sorted(by_day):
        rows = sorted(by_day[day], key=lambda item: item[0])
        # Keep the last print if a timestamp repeats.
        unique: dict[datetime, tuple[float, float, float, float]] = {}
        for when, open_, high, low, close in rows:
            unique[when] = (open_, high, low, close)
        ordered = [unique[when] for when in sorted(unique)]
        if len(ordered) != MINUTES_PER_DAY:
            continue
        first = min(when for when in unique)
        minute_of_day = first.hour * 60 + first.minute
        if minute_of_day != SESSION_OPEN_MINUTES:
            continue
        sessions.append(Session(day=day, bars=ordered))
    return sessions

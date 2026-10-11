"""Disk cache for the shared entry table and for one predictor's JSON result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from common.clock import Session
from common.entries import MIN_TAU_CENSORED, WARMUP_SESSIONS, build_entries
from common.simulator import CHECKPOINT_MIN, GTT_DROP, MIN_TRIGGER_GAP, SLIPPAGE, STOPLOSS_DROP, TICK


# Shared cache on a host where the install tree is not writable (``/static``).
SHARED_CACHE = Path("/var/cache/predictors")


def cache_root() -> Path:
    """Where entry tables and predictor JSON are stored.

    ``PREDICTORS_CACHE`` wins. Otherwise the project ``.cache`` directory is
    used when this user can write the project. A read-only install, such as
    ``/static/predictors``, uses ``/var/cache/predictors``.
    """
    override = os.environ.get("PREDICTORS_CACHE", "").strip()
    if override:
        return Path(override)
    project = Path(__file__).resolve().parents[1]
    if os.access(project, os.W_OK):
        return project / ".cache"
    return SHARED_CACHE


def config_hash(extra: str = "") -> str:
    """Hash of the simulator constants; ``extra`` adds a predictor's own parameters.

    Without ``extra`` the value is unchanged, so entry-table and baseline caches stay valid.
    """
    payload: dict = {
        "GTT_DROP": GTT_DROP,
        "STOPLOSS_DROP": STOPLOSS_DROP,
        "SLIPPAGE": SLIPPAGE,
        "TICK": TICK,
        "CHECKPOINT_MIN": CHECKPOINT_MIN,
        "MIN_TRIGGER_GAP": MIN_TRIGGER_GAP,
        "WARMUP_SESSIONS": WARMUP_SESSIONS,
        "MIN_TAU_CENSORED": MIN_TAU_CENSORED,
    }
    if extra:
        payload["extra"] = extra
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _instrument_key(instrument: str) -> str:
    return instrument.strip().upper().replace(":", "_")


def _entry_path(instrument: str) -> Path:
    return cache_root() / "entries" / _instrument_key(instrument) / f"{config_hash()}.json"


def _predictor_path(name: str, instrument: str) -> Path:
    return cache_root() / "predictors" / name / f"{_instrument_key(instrument)}.json"


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _write(path: Path, payload: dict) -> None:
    root = cache_root().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    # umask 027 would leave files 640 and directories 750, so the next account
    # could not read a cache written by the first. The cache root itself is
    # left alone: the shared directory is 1777 and must stay writable.
    path.chmod(0o644)
    current = path.parent.resolve()
    while current != root and root in current.parents:
        try:
            current.chmod(0o755)
        except OSError:
            break
        current = current.parent


def load_predictor_result(name: str, instrument: str, last_day: str, extra: str = "") -> dict | None:
    """The stored ``predict`` dict when it was built for this session and these constants."""
    data = _read(_predictor_path(name, instrument))
    if data is None:
        return None
    if data.get("last_day") != last_day or data.get("config_hash") != config_hash(extra):
        return None
    result = data.get("result")
    return result if isinstance(result, dict) else None


def save_predictor_result(name: str, instrument: str, last_day: str, result: dict, extra: str = "") -> None:
    _write(
        _predictor_path(name, instrument),
        {"last_day": last_day, "config_hash": config_hash(extra), "result": result},
    )


def cached_entries(instrument: str, sessions: list[Session]) -> list[dict]:
    """Entry rows for ``sessions``, replaying only what the cache does not already have."""
    if len(sessions) < 2:
        return []
    last_day = sessions[-1].day.isoformat()
    path = _entry_path(instrument)
    cached = _read(path)
    if (
        cached
        and cached.get("last_day") == last_day
        and cached.get("n_sessions") == len(sessions)
        and isinstance(cached.get("rows"), list)
    ):
        return cached["rows"]

    if _is_next_session(cached, sessions):
        rows = _extend(cached["rows"], sessions)
    else:
        rows = build_entries(sessions)
    _write(path, {"last_day": last_day, "n_sessions": len(sessions), "rows": rows})
    return rows


def _is_next_session(cached: dict | None, sessions: list[Session]) -> bool:
    if not cached or not isinstance(cached.get("rows"), list):
        return False
    n_cached = cached.get("n_sessions")
    if not isinstance(n_cached, int) or n_cached + 1 != len(sessions):
        return False
    return cached.get("last_day") == sessions[-2].day.isoformat()


def _extend(old_rows: list[dict], sessions: list[Session]) -> list[dict]:
    """Keep finished trades. Replay trades that were still open, plus the new entry day."""
    kept = [row for row in old_rows if row.get("exit_d") is not None]
    open_d0s = [int(row["d0"]) for row in old_rows if row.get("exit_d") is None]
    new_d0 = len(sessions) - 2  # the previous last session is now an entry day
    d0s = sorted(set(open_d0s + [new_d0]))
    fresh = build_entries(sessions, d0s)
    rows = kept + fresh
    rows.sort(key=lambda row: row["d0"])
    return rows

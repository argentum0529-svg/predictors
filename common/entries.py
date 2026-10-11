"""Entry table: one pretend purchase per past close, walked forward by the simulator."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

from common.clock import Session
from common.simulator import simulate

WARMUP_SESSIONS = 60
MIN_TAU_CENSORED = 5.0

# Set in each worker so the session list is pickled once, not once per day.
_WORKER_SESSIONS: list[Session] = []


def _annotate(row: dict) -> dict:
    row["r"] = row["G"] / row["tau"]
    row["eligible"] = not (row["censored"] and row["tau"] < MIN_TAU_CENSORED)
    return row


def _init_worker(sessions: list[Session]) -> None:
    global _WORKER_SESSIONS
    _WORKER_SESSIONS = sessions


def _run_chunk(d0s: list[int], last: int) -> list[dict]:
    return [_annotate(simulate(_WORKER_SESSIONS, d0, last)) for d0 in d0s]


def _chunks(d0s: list[int], workers: int) -> list[list[int]]:
    if workers <= 1 or len(d0s) <= 1:
        return [d0s]
    workers = min(workers, len(d0s))
    size = math.ceil(len(d0s) / workers)
    return [d0s[i : i + size] for i in range(0, len(d0s), size)]


def build_entries(sessions: list[Session], d0s: list[int] | None = None) -> list[dict]:
    """Rows for entry sessions ``[warmup, T - 1]``, or for ``d0s`` when given.

    The full range is split across one process per core. A single chunk
    (one core, or a handful of days) stays in this process.
    """
    if len(sessions) < 2:
        return []
    last = len(sessions) - 1
    if d0s is None:
        if len(sessions) < WARMUP_SESSIONS + 2:
            return []
        d0s = list(range(WARMUP_SESSIONS, last))
    if not d0s:
        return []
    chunks = _chunks(d0s, os.cpu_count() or 1)
    if len(chunks) <= 1:
        return [_annotate(simulate(sessions, d0, last)) for d0 in d0s]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_worker,
        initargs=(sessions,),
    ) as pool:
        parts = pool.map(_run_chunk, chunks, [last] * len(chunks))
    rows: list[dict] = []
    for part in parts:
        rows.extend(part)
    return rows

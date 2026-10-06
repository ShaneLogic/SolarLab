"""Attempt-scoped replay of committed RunStore events; no event broker.

IDs are the store's global SQLite sequences, scoped by store and attempt.
One stored terminal event contains the result and artifact references. There
are no synthetic result/error/done siblings that could share a replay cursor.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from solarlab.io.run_store import RunStore

MAX_SEQUENCE = 2**63 - 1
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


def attempt_record(store: RunStore, run_id: str, attempt_id: str) -> dict[str, Any]:
    """Read an existing attempt, including its unaltered result/terminal data."""
    for record in store.attempts(run_id):
        if record["attempt_id"] == attempt_id:
            return record
    raise KeyError(attempt_id)


def replay_cursor(
    store: RunStore, run_id: str, attempt_id: str, after: int, last_event_id: str | None,
) -> tuple[int, bool]:
    """A supplied Last-Event-ID overrides ``after`` and must belong here.

    Zero starts a replay. Acknowledging this attempt's terminal produces an
    empty HTTP 204 response, which also stops EventSource auto-reconnection.
    A future, foreign, or malformed cursor is an error, not an empty success.
    """
    cursor = after
    if last_event_id is not None:
        if not (last_event_id.isascii() and last_event_id.isdecimal()
                and len(last_event_id) <= 19):
            raise ValueError("Last-Event-ID must be a bounded nonnegative integer")
        cursor = int(last_event_id)
    if not 0 <= cursor <= MAX_SEQUENCE:
        raise ValueError("event cursor is outside the SQLite sequence domain")
    if cursor == 0:
        return 0, False
    records = store.events(after=cursor - 1, limit=1, run_id=run_id)
    if (not records or records[0]["sequence"] != cursor
            or records[0]["attempt_id"] != attempt_id):
        raise ValueError("event cursor does not identify an event of this attempt")
    return cursor, records[0]["kind"] == "terminal"


async def event_stream(
    request: Request, store: RunStore, run_id: str, attempt_id: str, *,
    after: int, page_size: int, poll_interval: float, keepalive_interval: float,
) -> AsyncIterator[str]:
    """Poll bounded pages without consuming another subscriber's events.

    A retry needs its own attempt URL. Disconnect/shutdown cancels only this
    iterator. If the store fails after HTTP headers, the connection fails;
    no business failure or successful terminal is fabricated in its place.
    """
    clock = asyncio.get_running_loop().time
    cursor = after
    next_keepalive = clock() + keepalive_interval
    while not await request.is_disconnected():
        records = await run_in_threadpool(store.events, after=cursor, limit=page_size, run_id=run_id)
        if not records:
            attempt = await run_in_threadpool(attempt_record, store, run_id, attempt_id)
            if attempt["state"] in TERMINAL_STATES:
                # The first page can precede the terminal commit. Reading again
                # AFTER the terminal state prevents closing across that window.
                records = await run_in_threadpool(
                    store.events, after=cursor, limit=page_size, run_id=run_id,
                )
                if not records:
                    return
        for record in records:
            cursor = record["sequence"]
            if record["attempt_id"] != attempt_id:
                continue
            data = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            yield f"id: {cursor}\nevent: {record['kind']}\ndata: {data}\n\n"
            next_keepalive = clock() + keepalive_interval
            if record["kind"] == "terminal":
                return
        if records:
            continue
        if clock() >= next_keepalive:
            yield ": keepalive\n\n"
            next_keepalive = clock() + keepalive_interval
        await asyncio.sleep(min(poll_interval, max(0.0, next_keepalive - clock())))

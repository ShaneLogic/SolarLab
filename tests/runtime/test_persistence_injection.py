"""Real event consumers accept a structural port without constructing RunStore."""

import asyncio
import inspect
import json
from typing import Any, get_type_hints

import pytest
from starlette.requests import Request

from solarlab.contracts.persistence import PersistenceReader
from solarlab_server import events


class RecordedEvents:
    """Small read-only test adapter; neither inherits nor wraps RunStore."""

    store_id = "injected"

    def __init__(self):
        self.reads = []
        self.records = [
            {"sequence": 1, "run_id": "run", "attempt_id": "attempt", "kind": "progress", "payload": {"n": 1}},
            {"sequence": 2, "run_id": "run", "attempt_id": "attempt", "kind": "terminal", "payload": {"state": "failed"}},
        ]

    def get_run(self, run_id: str) -> dict[str, Any]:
        assert run_id == "run"
        return {"run_id": run_id, "state": "failed", "current_attempt_id": "attempt"}

    def attempts(self, run_id: str) -> list[dict[str, Any]]:
        assert run_id == "run"
        return [{"attempt_id": "attempt", "state": "failed", "result": {"reason": "original_failure"}}]

    def events(self, *, after: int = 0, limit: int = 100, run_id: str | None = None) -> list[dict[str, Any]]:
        self.reads.append((after, limit, run_id))
        return [row for row in self.records if row["sequence"] > after
                and (run_id is None or row["run_id"] == run_id)][:limit]


@pytest.mark.parametrize("consumer", [events.attempt_record, events.replay_cursor, events.event_stream])
def test_consumers_declare_the_low_level_port(consumer):
    # Protocol imports are static-only to preserve the solver-free read path.
    assert get_type_hints(consumer, localns={"PersistenceReader": PersistenceReader})["store"] is PersistenceReader
    assert "from solarlab.io.run_store import" not in inspect.getsource(events)


def test_injected_reader_preserves_failure_and_cursor_validation():
    port: PersistenceReader = RecordedEvents()
    assert isinstance(port, PersistenceReader)
    assert events.attempt_record(port, "run", "attempt")["result"] == {"reason": "original_failure"}
    assert events.replay_cursor(port, "run", "attempt", 0, "1") == (1, False)
    assert events.replay_cursor(port, "run", "attempt", 0, "2") == (2, True)
    with pytest.raises(KeyError):
        events.attempt_record(port, "run", "foreign")
    with pytest.raises(ValueError, match="does not identify"):
        events.replay_cursor(port, "run", "foreign", 0, "2")


def test_event_stream_uses_only_injected_port_and_retains_wire_bytes():
    adapter = RecordedEvents()
    port: PersistenceReader = adapter

    async def receive():
        return {"type": "http.request"}

    async def collect():
        request = Request({"type": "http"}, receive=receive)
        return [frame async for frame in events.event_stream(
            request, port, "run", "attempt", after=0, page_size=1,
            poll_interval=0.01, keepalive_interval=1.0,
        )]

    frames = asyncio.run(collect())
    assert frames == [
        f"id: {row['sequence']}\nevent: {row['kind']}\ndata: "
        + json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n\n"
        for row in adapter.records
    ]
    assert adapter.reads == [(0, 1, "run"), (1, 1, "run")]

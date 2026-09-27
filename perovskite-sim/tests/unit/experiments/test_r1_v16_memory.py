"""Bounded native memory diagnostics never substitute for scientific evidence."""
from dataclasses import dataclass
import json
import os
import weakref

import numpy as np
import pytest

from scripts import r1_v16_memory as memory


@pytest.fixture
def fixed_rss(monkeypatch):
    monkeypatch.setattr(memory, "_current_rss_reader", lambda: (lambda: 123456, "test_current_RSS"))
    monkeypatch.setattr(memory, "_high_water_bytes", lambda: 456789)


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_balanced_phase_records_are_durable_and_keep_distinct_rss_measures(tmp_path, fixed_rss):
    path = tmp_path / "native.jsonl"
    observer = memory.NativeMemoryObserver(path)
    rows = [{"field": [1.0, 2.0]}]
    result = {"accepted_steps": rows}
    observer("conversion", "begin", {"raw_result": result, "result": result, "rows": rows})
    assert len(read_records(path)) == 1
    observer("conversion", "end", {"result": result, "persisted": None})
    observer.close()
    records = read_records(path)
    assert [r["sequence"] for r in records] == [0, 1]
    assert records[0]["same_nonnull_root_identity"] == [["raw_result", "result"]]
    assert records[0]["current_rss_bytes"] == 123456
    assert records[0]["cumulative_high_water_rss_bytes"] == 456789
    assert records[0]["roots"]["rows"]["length"] == 1
    assert records[1]["roots"]["persisted"]["present"] is False
    summary = observer.summary()
    assert summary["passed"] and summary["closed"] and summary["event_count"] == 2
    assert summary["artifact_bytes"] == path.stat().st_size
    assert summary["collector_elapsed_s"] >= 0
    assert summary["roots_retained"] is False and summary["forced_gc"] is False


def test_nested_error_boundary_closes_only_the_matching_phase(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    observer("run", "begin", {})
    observer("write", "begin", {})
    observer("write", "error", {})
    observer("run", "end", {})
    observer.close()
    assert observer.summary()["passed"]


def test_owning_arrays_views_and_shared_references_are_not_double_counted():
    owner = np.arange(9, dtype=np.float64)
    view = owner[::2]
    root = {"owner": owner, "view": view, "alias": owner, "scalar": np.int16(4)}
    result = memory.summarize_root(root)["bounded_sample"]
    assert result["type_counts"]["ndarray"] == 2
    assert result["duplicate_edges"] == 1
    assert result["sampled_owning_ndarray_nbytes"] == owner.nbytes
    assert result["sampled_view_logical_nbytes"] == view.nbytes
    assert result["sampled_numpy_scalar_nbytes"] == 2
    assert result["dtype_counts"] == {"float64": 2}
    assert memory.summarize_root(view)["bounded_sample"]["sampled_owning_ndarray_nbytes"] == 0


def test_object_arrays_do_not_traverse_payload_or_report_payload_ownership():
    secret = np.ones(200, dtype=np.float64)
    value = np.empty(1, dtype=object)
    value[0] = secret
    sample = memory.summarize_root(value)["bounded_sample"]
    assert sample["type_counts"]["ndarray"] == 1
    assert sample["sampled_owning_ndarray_nbytes"] == value.nbytes


def test_live_roots_are_not_retained_and_data_is_never_evaluated(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    array = np.array([float("nan"), float("inf"), -0.0], dtype=np.float64)
    before = array.tobytes()
    reference = weakref.ref(array)
    observer("native", "begin", {"raw_result": array})
    observer("native", "end", {"raw_result": array})
    assert array.tobytes() == before
    del array
    assert reference() is None
    observer.close()
    raw = (tmp_path / "native.jsonl").read_text()
    assert '"NaN"' not in raw and '"Infinity"' not in raw


def test_cycle_and_large_graph_obey_fixed_identity_and_child_budgets():
    cycle = []
    cycle.append(cycle)
    summary = memory.summarize_root(cycle)["bounded_sample"]
    assert summary["unique_nodes"] == 1 and summary["duplicate_edges"] == 1
    large = [{str(index): [float(index + j) for j in range(100)] for index in range(100)}
             for _ in range(100)]
    bounded = memory.summarize_root(large)
    assert bounded["length"] == 100
    assert len(bounded["direct_children_sample"]) == memory.LIMITS["max_children_per_node"]
    assert bounded["bounded_sample"]["unique_nodes"] <= memory.LIMITS["max_nodes_per_root"]
    assert bounded["bounded_sample"]["node_limit_reached"]
    assert bounded["bounded_sample"]["unvisited_children"] > 0


def test_depth_limit_does_not_touch_deeper_array():
    nested = np.ones(10)
    for _ in range(memory.LIMITS["max_depth"] + 1):
        nested = [nested]
    sample = memory.summarize_root(nested)["bounded_sample"]
    assert sample["type_counts"]["ndarray"] == 0
    assert sample["depth_limited_nodes"] == 1


def test_sequence_sampling_includes_endpoints_without_copying_all_rows():
    summary = memory.summarize_root(list(range(100000)))
    keys = [item["key"] for item in summary["direct_children_sample"]]
    assert keys[0] == "0" and keys[-1] == "99999"
    assert len(keys) == memory.LIMITS["max_children_per_node"]


def test_priority_accepted_steps_is_seen_after_large_metadata_prefix():
    result = {str(index): index for index in range(100)}
    result["accepted_steps"] = [{"state": np.ones(10)}]
    summary = memory.summarize_root(result)
    assert summary["direct_children_sample"][0]["key"] == "accepted_steps"
    assert summary["direct_children_sample"][0]["length"] == 1
    assert summary["bounded_sample"]["type_counts"]["ndarray"] == 1


def test_dataclass_storage_is_inspected_without_to_dict_or_properties():
    @dataclass(frozen=True, slots=True)
    class Prepared:
        canonical_json: str

        def to_dict(self):
            raise AssertionError("must not deserialize preparation")

    @dataclass
    class PropertyRecord:
        payload: object

        @property
        def payload(self):
            raise AssertionError("must not call properties")

    prepared = memory.summarize_root(Prepared('{"value":[1,2,3]}'))
    assert prepared["direct_child_count"] == 1
    assert prepared["direct_children_sample"][0]["length"] == 17
    opaque = memory.summarize_root(object.__new__(PropertyRecord))
    assert opaque["direct_children_sample"] == []


def test_custom_dict_property_is_not_called():
    @dataclass(slots=True)
    class Prepared:
        canonical_json: str

        @property
        def __dict__(self):
            raise AssertionError("must not call __dict__ property")

    summary = memory.summarize_root(Prepared("stored"))
    assert summary["direct_children_sample"][0]["length"] == 6


@pytest.mark.parametrize("phase,event,roots", [
    ("", "begin", {}), ("x" * 161, "begin", {}), ("run", "snapshot", {}),
    ("run", "end", {}), ("run", "error", {}), ("run", "begin", {"unrecognized": []}),
    ("run", "begin", []),
])
def test_invalid_contract_is_sticky_failed_and_not_a_silent_missing_sample(tmp_path, fixed_rss, phase, event, roots):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    with pytest.raises(memory.MemoryObservationError):
        observer(phase, event, roots)
    assert not observer.summary()["passed"]
    assert observer.summary()["event_count"] == 0
    with pytest.raises(memory.MemoryObservationError, match="already failed"):
        observer("retry", "begin", {})
    observer.close()


def test_out_of_order_boundary_and_unclosed_close_fail(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    observer("outer", "begin", {})
    with pytest.raises(memory.MemoryObservationError, match="unbalanced"):
        observer("different", "end", {})
    with pytest.raises(memory.MemoryObservationError, match="unclosed"):
        observer.close()
    assert observer.summary()["closed"] and not observer.summary()["passed"]


@pytest.mark.parametrize("operation", ["write", "flush", "fsync"])
def test_write_failures_propagate_and_do_not_count_a_successful_event(tmp_path, monkeypatch, fixed_rss, operation):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    stream = observer._stream

    def fail(*args):
        raise OSError("controlled " + operation + " failure")

    if operation == "fsync":
        monkeypatch.setattr(memory.os, "fsync", fail)
    else:
        class BrokenStream:
            write = staticmethod(fail) if operation == "write" else stream.write
            flush = staticmethod(fail) if operation == "flush" else stream.flush
            fileno = stream.fileno
            close = stream.close
        observer._stream = BrokenStream()
    with pytest.raises(memory.MemoryObservationError, match="controlled"):
        observer("run", "begin", {})
    assert observer.summary()["event_count"] == 0 and not observer.summary()["passed"]
    monkeypatch.undo()
    observer._stream = stream
    observer.close()


def test_invalid_rss_fails_explicitly(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    observer._rss = lambda: 0
    with pytest.raises(memory.MemoryObservationError, match="invalid native RSS"):
        observer("run", "begin", {})
    observer.close()


def test_keyboard_interrupt_is_not_converted_into_an_ordinary_error(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    def interrupt():
        raise KeyboardInterrupt()
    observer._rss = interrupt
    with pytest.raises(KeyboardInterrupt):
        observer("run", "begin", {})
    observer.close()
    assert not observer.summary()["passed"]
    assert observer.summary()["failure"]["type"] == "KeyboardInterrupt"


def test_close_failure_is_sticky_and_explicit(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    observer("run", "begin", {})
    observer("run", "end", {})
    stream = observer._stream
    class BrokenClose:
        flush = stream.flush
        fileno = stream.fileno
        def close(self):
            stream.close()
            raise OSError("controlled close failure")
    observer._stream = BrokenClose()
    with pytest.raises(memory.MemoryObservationError, match="close failure"):
        observer.close()
    assert observer.summary()["closed"] and not observer.summary()["passed"]


def test_zero_events_do_not_qualify_a_requested_observation(tmp_path, fixed_rss):
    observer = memory.NativeMemoryObserver(tmp_path / "native.jsonl")
    observer.close()
    assert not observer.summary()["passed"]


def test_existing_artifact_is_not_overwritten(tmp_path, fixed_rss):
    path = tmp_path / "native.jsonl"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        memory.NativeMemoryObserver(path)
    assert path.read_text() == "original"


def test_native_current_rss_reader_returns_current_resident_bytes():
    reader, source = memory._current_rss_reader()
    value = reader()
    assert type(value) is int and value > 0
    assert "resident" in source
    assert memory._high_water_bytes() > 0
    assert os.getpid() > 0

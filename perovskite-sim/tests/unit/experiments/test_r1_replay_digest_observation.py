"""Replay hash phase boundaries and runner dispatch without physical solves."""

import inspect
import json

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from scripts import run_r1_v9_prototype as runner
from tests.integration.test_r1_v9_backend import pair_preparation
from tests.unit.experiments.test_r1_v16_runner_observation import (
    ROOT_NAMES, SCIENCE_FILES, fixture_api, scalar_report,
)


PHASES = ("replay_saved_rows_digest", "replay_recomputed_rows_digest")


@pytest.mark.parametrize("phase", PHASES)
def test_replay_digest_streams_same_value_and_observes_original_roots(monkeypatch, phase):
    rows = [{"coordinate": np.asarray([1.25, -0.0, 2.0 ** -80]), "label": "界面"}]
    result = {"accepted_steps": rows}
    prepared = object()
    expected = common.digest(rows)
    events, hashes = [], []

    def calculate(value):
        hashes.append(value)
        assert events == [(phase, "begin")]
        return common.streaming_digest(value)

    def observe(actual_phase, event, roots):
        assert roots["rows"] is rows
        assert roots["result"] is result
        assert roots["prepared"] is prepared
        events.append((actual_phase, event))

    def forbidden(*args, **kwargs):
        raise AssertionError("replay helper used whole-document digest")

    monkeypatch.setattr(physics, "streaming_digest", calculate)
    monkeypatch.setattr(physics, "digest", forbidden)
    actual = physics._replay_rows_digest(rows, phase, observe,
                                         prepared=prepared, result=result, rows=rows)
    assert actual == expected
    assert len(hashes) == 1 and hashes[0] is rows
    assert events == [(phase, "begin"), (phase, "end")]


def test_replay_digest_allows_omitted_or_explicit_none_observer():
    rows = [{"state": (np.float64(-0.0), 5e-324), "name": "中文"}]
    expected = common.digest(rows)
    assert physics._replay_rows_digest(rows, PHASES[0]) == expected
    assert physics._replay_rows_digest(rows, PHASES[1], phase_observer=None, rows=rows) == expected


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("observer_also_fails", [False, True])
def test_digest_failure_emits_error_and_preserves_original_exception(
    monkeypatch, error_type, observer_also_fails,
):
    rows = []
    original_error = error_type("original hash error")
    events = []

    def fail_hash(value):
        assert value is rows
        raise original_error

    def observe(phase, event, roots):
        assert roots["rows"] is rows
        events.append((phase, event))
        if event == "error" and observer_also_fails:
            raise RuntimeError("secondary observer error")

    monkeypatch.setattr(physics, "streaming_digest", fail_hash)
    with pytest.raises(error_type) as caught:
        physics._replay_rows_digest(rows, PHASES[0], observe, rows=rows)
    assert caught.value is original_error
    assert events == [(PHASES[0], "begin"), (PHASES[0], "error")]


def test_public_replay_phase_observer_is_optional_and_keyword_only():
    parameter = inspect.signature(physics.verify_r1_step_physics).parameters["phase_observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _run_fixture(directory, *, observer=None, observed_callback=None):
    directory.mkdir()
    case, state, api = fixture_api()
    if observed_callback is not None:
        api["replay_observed"] = observed_callback
    report = runner.run_trajectory(directory, case, "baseline", api, lambda: None,
                                   memory_observer=observer)
    return report, state


def test_runner_forwards_internal_memory_callback_to_observed_replay(tmp_path):
    reference, _ = _run_fixture(tmp_path / "reference")
    directory = tmp_path / "observed"
    directory.mkdir()
    case, state, api = fixture_api()
    original_replay = api["replay"]
    events, forwarding = [], []

    def observe(phase, event, roots):
        assert set(roots) == ROOT_NAMES
        if phase in PHASES:
            assert roots["rows"] is roots["result"]["accepted_steps"]
            assert roots["prepared"] is not None
            assert roots["raw_result"] is None
        events.append((phase, event))

    def observed_replay(prepared, result, incomplete, observer, *, phase_observer):
        # The runner's adapter contributes current roots and timing; passing
        # the external observer directly would lose both contracts.
        assert phase_observer is not observe
        forwarding.append(phase_observer)
        rows = result["accepted_steps"]
        for phase in PHASES:
            assert physics._replay_rows_digest(rows, phase, phase_observer,
                                               rows=rows) == common.digest(rows)
        return original_replay(prepared, result, incomplete, observer)

    def forbidden(*args, **kwargs):
        raise AssertionError("observed replay dispatched to fallback")

    api["replay_observed"] = observed_replay
    api["replay"] = forbidden
    report = runner.run_trajectory(directory, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    assert len(forwarding) == 1
    assert state["calls"].count("replay") == 1
    assert scalar_report(report) == scalar_report(reference)
    assert report["memory_observation"]["passed"]
    assert report["memory_observation"]["call_count"] == len(events)
    assert [(phase, event) for phase, event in events if phase in PHASES] == [
        (phase, event) for phase in PHASES for event in ("begin", "end")]
    for name in SCIENCE_FILES:
        assert (directory / name).read_bytes() == (tmp_path / "reference" / name).read_bytes(), name


def test_runner_without_memory_observer_never_uses_observed_replay(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled observation invoked observed replay")

    reference, _ = _run_fixture(tmp_path / "reference")
    report, state = _run_fixture(tmp_path / "disabled", observed_callback=forbidden)
    assert scalar_report(report) == scalar_report(reference)
    assert "memory_observation" not in report
    assert state["calls"].count("replay") == 1
    for name in SCIENCE_FILES:
        assert (tmp_path / "disabled" / name).read_bytes() == (tmp_path / "reference" / name).read_bytes(), name


def test_runner_with_memory_observer_keeps_legacy_replay_fallback(tmp_path):
    events = []
    report, state = _run_fixture(tmp_path / "fallback",
        observer=lambda phase, event, roots: events.append((phase, event)))
    assert report["four_predicates_passed"]
    assert report["memory_observation"]["passed"]
    assert state["calls"].count("replay") == 1
    assert ("replay", "begin") in events and ("replay", "end") in events
    assert all(phase not in PHASES for phase, _ in events)


@pytest.mark.slow
def test_actual_pair_replay_digest_phases_use_valid_native_collector_roots(pair_preparation, tmp_path):
    """Exercise both production call sites, including the collector root schema."""
    from threadpoolctl import threadpool_limits
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from scripts.r1_v16_memory import NativeMemoryObserver

    stack, binding, prepared = pair_preparation
    path = tmp_path / "NativeMemoryPhasesV1.jsonl"
    with threadpool_limits(1):
        record = protocol.run_r1_step(stack, 16, binding, prepared, control="D",
                                      times_s=[0.0, 1e-9], backend="pair")
        collector = NativeMemoryObserver(path)
        try:
            verified = physics.verify_r1_step_physics(
                stack, 16, binding, prepared, record,
                backend="pair", phase_observer=collector,
            )
        finally:
            collector.close()

    assert record["certificate"]["certified"] and verified["certified"]
    receipt = verified.replay_receipt.to_dict()
    expected = common.digest(record["accepted_steps"])
    assert receipt["saved_rows_digest"] == receipt["actual_recomputed_rows_digest"] == expected
    assert receipt["checked_row_count"] == len(record["accepted_steps"]) == 10
    measurement = collector.summary()
    assert measurement["passed"] and measurement["closed"]
    assert measurement["event_count"] == 4
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(item["phase"], item["event"]) for item in records] == [
        (phase, event) for phase in PHASES for event in ("begin", "end")]
    for item in records:
        assert set(item["roots"]) == {"prepared", "result", "rows", "replay"}
        assert item["roots"]["rows"]["length"] == 10
        assert item["roots"]["replay"]["length"] == 10

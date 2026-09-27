"""Coarse observation must describe real lifetimes without changing evidence."""
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from scripts import run_r1_v9_prototype as runner
from scripts import run_r1_v12_cases as case_runner


ROOT_NAMES = {"prepared", "raw_result", "result", "rows", "persisted", "replay"}
SUCCESS_PHASES = [
    "prepare", "prepared_write", "integrate", "result_conversion", "result_write",
    "numeric_write", "raw_release", "numeric_readback", "replay", "replay_write",
    "rows_readback", "final_validation", "after_main",
]
SCIENCE_FILES = (
    "PreparedV1.json", "ResultV1.json", "AcceptedStepsV1.jsonl", "StateArraysV1.npz",
    "ReplayRowsV1.jsonl", "PhysicsReplayV1.json", "ReplayLedgerV1.json",
)


class RawResult(dict):
    """A weak-referenceable owner for the original native array tree."""


class ReplayResult(dict):
    pass


def json_data(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]
    return value


def fixture_api(*, fail=None):
    """Exercise the runner only: no nonlinear solve or physical validation."""
    state = {"calls": [], "raw_reference": None, "array_reference": None}
    prepared = SimpleNamespace(to_dict=lambda: {"seed": "fixed", "zero": -0.0})
    case = {"time_substeps": [1], "times_s": [0.0, 1.0]}

    def prepare():
        state["calls"].append("prepare")
        if fail == "prepare":
            raise RuntimeError("bounded preparation failure")
        return prepared

    def run(actual_prepared, observer):
        assert actual_prepared is prepared
        state["calls"].append("integrate")
        words = np.array([1.0, -0.0, 2.0**-80], dtype=np.float64)
        rows = [{"substeps": 1, "time_s": t, "phase": "0+" if t == 0 else "step",
                 "state": {"native_words": words.copy()}} for t in case["times_s"]]
        raw = RawResult(accepted_steps=rows, certificate={
            "certified": fail != "integrate", "limits": deepcopy(runner.LIMITS),
            "metrics": {name: 0.0 for name in runner.LIMITS}})
        state["raw_reference"] = weakref.ref(raw)
        state["array_reference"] = weakref.ref(rows[0]["state"]["native_words"])
        for row in rows:
            observer(row)
        if fail == "integrate":
            exc = RuntimeError("bounded integration failure")
            exc.result = raw
            raise exc
        return raw

    def persist(path, raw):
        state["calls"].append("numeric_write")
        assert raw is state["raw_reference"]() and isinstance(raw, RawResult)
        assert raw["accepted_steps"][0]["state"]["native_words"].dtype == np.dtype("float64")
        if fail == "numeric_write":
            raise OSError("bounded numeric persistence failure")
        with path.open("wb") as stream:
            np.savez_compressed(stream, state_words=np.stack([
                row["state"]["native_words"] for row in raw["accepted_steps"]]))

    def verify(path, result):
        state["calls"].append("numeric_readback")
        assert type(result) is dict
        # Failed integration can retain its traceback until the caught error
        # scope exits; the completed path must not keep either native owner.
        if fail is None:
            assert state["raw_reference"]() is None
            assert state["array_reference"]() is None
        with np.load(path, allow_pickle=False) as archive:
            expected = np.array([row["state"]["native_words"] for row in result["accepted_steps"]])
            assert archive.files == ["state_words"]
            assert archive["state_words"].tobytes() == expected.tobytes()

    def replay(actual_prepared, result, incomplete, observer):
        state["calls"].append("replay")
        assert actual_prepared is prepared
        assert incomplete is (fail == "integrate")
        if fail == "replay":
            raise RuntimeError("bounded independent replay failure")
        for index, _ in enumerate(result["accepted_steps"]):
            observer({"row_index": index, "verified": True})
        replay_result = ReplayResult(certified=fail != "integrate")
        replay_result.replay_receipt = SimpleNamespace(to_dict=lambda: {
            "observed_rows": len(result["accepted_steps"]), "input_words_verified": True})
        return replay_result

    def witness(actual_prepared, result, rows, failure):
        state["calls"].append("failure_witness")
        assert actual_prepared is prepared
        assert rows == result["accepted_steps"]
        return {"rows": len(rows), "failure_type": failure["type"]}

    def after_main(actual_prepared, result, replay_result):
        state["calls"].append("after_main")
        assert actual_prepared is prepared and result["accepted_steps"]
        if fail == "after_main":
            raise RuntimeError("bounded saved-state analysis failure")
        return {"passed": fail not in ("integrate", "replay"), "rows": len(result["accepted_steps"])}

    return case, state, {
        "prepare": prepare, "run": run, "json_data": json_data,
        "persist_numeric": persist, "verify_numeric": verify, "replay": replay,
        "failure_witness": witness, "after_main": after_main,
    }


def execute(directory, *, observer=None, fail=None):
    directory.mkdir()
    case, state, api = fixture_api(fail=fail)
    report = runner.run_trajectory(directory, case, "baseline", api, lambda: None,
                                   memory_observer=observer)
    return report, state


def scalar_report(report):
    """Original predicates are compared; measured timing/RSS are not fixtures."""
    return {key: report[key] for key in (
        "execution_status", "physical_execution_started", "numeric_sidecar_exact",
        "replay_completed", "extent", "certificate_check", "original_metric_failures",
        "precision_records", "four_predicates", "four_predicates_passed",
        "replay_observed_rows", "failure", "independent_analysis",
    )}


def test_public_observer_is_optional_and_keyword_only():
    parameter = inspect.signature(runner.run_trajectory).parameters["memory_observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_observation_preserves_every_deterministic_scientific_artifact(tmp_path):
    events = []
    disabled, disabled_state = execute(tmp_path / "disabled")
    enabled, enabled_state = execute(tmp_path / "enabled", observer=lambda phase, event, roots:
                                      events.append((phase, event)))
    assert disabled_state["calls"] == enabled_state["calls"]
    assert scalar_report(disabled) == scalar_report(enabled)
    assert enabled["four_predicates_passed"]
    for name in SCIENCE_FILES:
        assert (tmp_path / "disabled" / name).read_bytes() == (tmp_path / "enabled" / name).read_bytes(), name
    assert enabled["memory_observation"]["passed"]
    assert enabled["memory_observation"]["call_count"] == len(events)
    assert enabled["memory_observation"]["error"] is None


def test_phase_boundaries_describe_actual_native_owner_release(tmp_path):
    directory = tmp_path / "observed"
    directory.mkdir()
    case, state, api = fixture_api()
    boundaries, roots_at_boundary = [], {}

    def observe(phase, event, roots):
        assert set(roots) == ROOT_NAMES
        assert event in {"begin", "end", "error"}
        boundaries.append((phase, event))
        # Only scalar metadata and weak references leave this callback.
        roots_at_boundary[(phase, event)] = {
            "prepared": roots["prepared"] is not None,
            "raw": roots["raw_result"] is not None,
            "result_type": type(roots["result"]).__name__,
            "rows": len(roots["rows"]),
            "persisted": None if roots["persisted"] is None else len(roots["persisted"]),
            "replay": roots["replay"] is not None,
        }
        if phase == "numeric_write":
            assert roots["raw_result"] is state["raw_reference"]()
            assert isinstance(roots["raw_result"], RawResult)
        if phase == "raw_release" and event == "end":
            assert roots["raw_result"] is None
            assert state["raw_reference"]() is None
            assert state["array_reference"]() is None
        if phase == "final_validation" and event == "begin":
            assert roots["persisted"] == roots["rows"] == roots["result"]["accepted_steps"]
            assert roots["persisted"] is not roots["rows"]
            assert roots["persisted"] is not roots["result"]["accepted_steps"]

    report = runner.run_trajectory(directory, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    assert report["memory_observation"]["passed"], report["memory_observation"]
    assert boundaries == [(phase, event) for phase in SUCCESS_PHASES for event in ("begin", "end")]
    assert not roots_at_boundary[("prepare", "begin")]["prepared"]
    assert roots_at_boundary[("prepare", "end")]["prepared"]
    assert roots_at_boundary[("integrate", "end")]["rows"] == 2
    assert roots_at_boundary[("result_conversion", "end")]["result_type"] == "dict"
    assert roots_at_boundary[("result_conversion", "end")]["raw"]
    assert roots_at_boundary[("raw_release", "begin")]["raw"]
    assert not roots_at_boundary[("raw_release", "end")]["raw"]
    assert roots_at_boundary[("rows_readback", "end")]["persisted"] == 2
    assert roots_at_boundary[("after_main", "begin")]["replay"]
    assert state["raw_reference"]() is None and state["array_reference"]() is None
    # The metadata is safe to persist and holds no object IDs or native roots.
    json.dumps({phase + "/" + event: values
                for (phase, event), values in roots_at_boundary.items()})


@pytest.mark.parametrize("failed_phase", ["prepare", "integrate", "numeric_write", "replay", "after_main"])
def test_failed_operation_has_error_boundary_and_keeps_available_evidence(tmp_path, failed_phase):
    boundaries = []
    report, _ = execute(tmp_path / failed_phase, fail=failed_phase,
                         observer=lambda phase, event, roots: boundaries.append((phase, event)))
    assert (failed_phase, "begin") in boundaries
    assert (failed_phase, "error") in boundaries
    assert (failed_phase, "end") not in boundaries
    assert ("rows_readback", "end") in boundaries
    assert ("final_validation", "end") in boundaries
    assert report["memory_observation"]["passed"]
    assert report["extent"]["accepted_rows"] == (0 if failed_phase == "prepare" else 2)
    if failed_phase != "prepare":
        assert (tmp_path / failed_phase / "ResultV1.json").is_file()
        assert len((tmp_path / failed_phase / "AcceptedStepsV1.jsonl").read_text().splitlines()) == 2
    if failed_phase in {"prepare", "integrate", "numeric_write"}:
        assert report["execution_status"] == "failed"
        assert not report["four_predicates_passed"]
        assert (tmp_path / failed_phase / "FailureV1.json").is_file()
    elif failed_phase == "replay":
        assert report["replay_error"]["message"] == "bounded independent replay failure"
        assert not report["four_predicates_passed"]
    else:
        assert not report["independent_analysis"]["passed"]


def test_observer_error_is_reported_once_and_does_not_destroy_science(tmp_path):
    calls = []

    def broken_observer(phase, event, roots):
        calls.append((phase, event))
        if phase == "integrate" and event == "begin":
            raise ValueError("bounded observer failure")

    reference, _ = execute(tmp_path / "reference")
    observed, _ = execute(tmp_path / "observer_failure", observer=broken_observer)
    assert calls[-1] == ("integrate", "begin")
    assert scalar_report(reference) == scalar_report(observed)
    assert observed["four_predicates_passed"]
    assert not observed["memory_observation"]["passed"]
    assert observed["memory_observation"]["error"]["type"] == "ValueError"
    assert observed["memory_observation"]["error"]["message"] == "bounded observer failure"
    assert observed["memory_observation"]["call_count"] == len(calls)
    for name in SCIENCE_FILES:
        assert (tmp_path / "reference" / name).read_bytes() == (tmp_path / "observer_failure" / name).read_bytes()
    assert not (tmp_path / "observer_failure" / "FailureV1.json").exists()


def test_observer_overhead_is_explicit_and_original_main_cost_includes_it(tmp_path, monkeypatch):
    clock = {"time": 100.0}
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["time"])
    events = []

    def observe(phase, event, roots):
        events.append((phase, event))
        clock["time"] += 0.25

    report, _ = execute(tmp_path / "timed", observer=observe)
    measurement = report["memory_observation"]
    assert measurement["passed"]
    assert measurement["call_count"] == len(events)
    assert measurement["elapsed_s"] == len(events) * 0.25
    main_calls = sum(phase != "after_main" for phase, _ in events)
    assert measurement["main_elapsed_s"] == main_calls * 0.25
    assert report["cost"]["elapsed_s"] >= measurement["main_elapsed_s"]
    assert report["independent_analysis_elapsed_s"] >= 0.5
    assert report["phase_timing"]["exclusive_sum_s"] <= report["cost"]["elapsed_s"]


def test_disabled_diagnostics_do_not_require_collector_evidence():
    assert case_runner.memory_observation_passed({}, enabled=False)
    assert case_runner.memory_observation_passed({
        "memory_observation": {"passed": False}}, enabled=False)
    assert not case_runner.memory_observation_passed({}, enabled=True)


def valid_memory_summary():
    return {"memory_observation": {
        "enabled": True, "passed": True, "call_count": 26, "error": None},
        "memory_collector": {"passed": True, "closed": True, "event_count": 26,
                             "open_phases": []}}


def test_requested_diagnostics_need_both_complete_matching_receipts():
    summary = valid_memory_summary()
    original = deepcopy(summary)
    assert case_runner.memory_observation_passed(summary, enabled=True)
    assert summary == original
    for missing in ("memory_observation", "memory_collector"):
        partial = deepcopy(summary)
        del partial[missing]
        assert not case_runner.memory_observation_passed(partial, enabled=True)


@pytest.mark.parametrize("section,key,value", [
    ("memory_observation", "enabled", False),
    ("memory_observation", "passed", False),
    ("memory_observation", "call_count", 0),
    ("memory_observation", "call_count", True),
    ("memory_observation", "call_count", 26.0),
    ("memory_observation", "error", {"type": "RuntimeError", "message": "observer failed"}),
    ("memory_collector", "passed", False),
    ("memory_collector", "closed", False),
    ("memory_collector", "event_count", 25),
    ("memory_collector", "event_count", 26.0),
    ("memory_collector", "open_phases", ["integrate"]),
])
def test_incomplete_or_failed_diagnostics_cannot_pass_integrity(section, key, value):
    summary = valid_memory_summary()
    summary[section][key] = value
    assert not case_runner.memory_observation_passed(summary, enabled=True)


def test_observer_keyboard_interrupt_keeps_existing_interruption_semantics(tmp_path):
    events = []

    def observe(phase, event, roots):
        events.append((phase, event))
        if phase == "integrate" and event == "begin":
            raise KeyboardInterrupt("bounded user interruption")

    report, state = execute(tmp_path / "interrupted", observer=observe)
    assert report["execution_status"] == "interrupted"
    assert not report["four_predicates_passed"]
    assert report["extent"]["accepted_rows"] == 0
    assert state["calls"] == ["prepare"]
    assert report["failure"] == {"type": "KeyboardInterrupt", "message": "bounded user interruption"}
    assert (tmp_path / "interrupted" / "PreparedV1.json").is_file()
    assert (tmp_path / "interrupted" / "FailureV1.json").is_file()
    assert ("final_validation", "end") in events


def cost_summary(mode):
    return {"schema": "R1V12CaseRunV1", "case_id": "D_N256_F0p01_T1", "mode": mode,
            "source_commit": "a" * 40, "source_content_sha256": "b" * 64,
            "request_sha256": "c" * 64, "precision_budget_sha256": "d" * 64,
            "runtime_identity": {"numpy": "2.1.3", "scipy": "1.15.3"},
            "baseline_usable": mode == "baseline", "extent": {"complete": True},
            "cost": {"elapsed_s": 1.0, "peak_rss_bytes": 100, "bytes_per_row": 10.0}}


@pytest.mark.parametrize("pair_profiled,baseline_profiled", [(True, False), (False, True)])
def test_profiled_and_unprofiled_costs_never_share_a_denominator(pair_profiled, baseline_profiled):
    current, baseline = cost_summary("compensated"), cost_summary("baseline")
    current["memory_profile_enabled"] = pair_profiled
    baseline["memory_profile_enabled"] = baseline_profiled
    with pytest.raises(ValueError, match="same-source same-request"):
        case_runner.relative_cost(current, baseline)


@pytest.mark.parametrize("which", ["current", "baseline"])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", []])
def test_cost_mode_requires_a_boolean_for_both_receipts(which, invalid):
    current, baseline = cost_summary("compensated"), cost_summary("baseline")
    {"current": current, "baseline": baseline}[which]["memory_profile_enabled"] = invalid
    with pytest.raises(ValueError, match="explicit boolean"):
        case_runner.relative_cost(current, baseline)


def test_legacy_mode_absence_means_only_unprofiled():
    assert case_runner.memory_profile_mode({}) is False
    current, baseline = cost_summary("compensated"), cost_summary("baseline")
    assert case_runner.relative_cost(current, baseline)["qualified"]
    current["memory_profile_enabled"] = False
    assert case_runner.relative_cost(current, baseline)["qualified"]
    current.pop("memory_profile_enabled")
    baseline["memory_profile_enabled"] = False
    assert case_runner.relative_cost(current, baseline)["qualified"]
    baseline["memory_profile_enabled"] = True
    with pytest.raises(ValueError, match="same-source same-request"):
        case_runner.relative_cost(current, baseline)
    current["memory_profile_enabled"] = True
    assert case_runner.relative_cost(current, baseline)["qualified"]


@pytest.mark.parametrize("pair_profiled,baseline_profiled", [(True, False), (False, True)])
def test_profile_mode_mismatch_is_rejected_before_source_or_science(tmp_path, monkeypatch,
                                                                 pair_profiled, baseline_profiled):
    from scripts import analyze_r1_v8_prototype as analyzer

    args = SimpleNamespace(output=tmp_path / "case", request_file=tmp_path / "Request.json",
        request_sha256="c" * 64, budget_file=tmp_path / "Budget.json", budget_sha256="d" * 64,
        mode="compensated", baseline_summary=tmp_path / "Baseline.json", baseline_sha256="e" * 64,
        expected_commit="a" * 40, source_sha256="b" * 64, memory_profile=pair_profiled)
    baseline = {**cost_summary("baseline"), "memory_profile_enabled": baseline_profiled}
    request = {"case_id": baseline["case_id"], "precision_budget_sha256": args.budget_sha256}
    budget = {"schema": "R1V12PrecisionBudgetV1", "engineering": {
        "relative_limits": case_runner.RELATIVE_LIMITS, "absolute_limits": None}}
    monkeypatch.setattr(case_runner, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=True, no_site=True)))
    for key in case_runner.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(case_runner, "load_request", lambda *unused: request)
    monkeypatch.setattr(analyzer, "validate_budget", lambda *unused: None)
    monkeypatch.setattr(case_runner, "external_json", lambda path, digest:
                        budget if path == args.budget_file else baseline)
    source_calls = []
    monkeypatch.setattr(case_runner, "source_snapshot", lambda *unused: source_calls.append(True))
    with pytest.raises(ValueError, match="baseline identity"):
        case_runner.run(args)
    assert not source_calls
    assert not args.output.exists()

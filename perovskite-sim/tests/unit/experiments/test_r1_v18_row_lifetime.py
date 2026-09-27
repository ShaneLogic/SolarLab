"""Synthetic runner lifetime checks; these do not qualify any physical case."""
from copy import deepcopy
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from scripts import run_r1_v9_prototype as runner


class InputRow(dict):
    pass


class ObservedRow(dict):
    """A weak-referenceable row owned only by the runner's observer copy."""


class PersistedRows(list):
    """A weak-referenceable owner for the independent JSONL readback."""


def fixture_api(monkeypatch, directory, *, fail=None, alias=False, after_main=True):
    """Run the actual file/validation path with two bounded synthetic rows."""
    case = {"time_substeps": [1], "times_s": [0.0, 1.0]}
    interrupted = fail in {"integrate", "interrupt"}
    raw_rows = [InputRow(substeps=1, time_s=time, phase="0+" if time == 0 else "step",
                         state={"words": [1.0, -0.0, 2.0**-80]})
                for time in case["times_s"][:1 if interrupted else 2]]
    raw_result = {"accepted_steps": raw_rows, "certificate": {
        "certified": not interrupted, "limits": deepcopy(runner.LIMITS),
        "metrics": {name: 0.0 for name in runner.LIMITS}}}
    prepared_data = {"seed": "fixed", "words": [1.0, -0.0, 2.0**-80]}
    prepared = SimpleNamespace(to_dict=lambda: prepared_data)
    state = {"calls": [], "observed_references": [], "persisted_reference": None,
             "raw_result": raw_result, "raw_snapshot": deepcopy(raw_result),
             "prepared": prepared, "prepared_snapshot": deepcopy(prepared_data),
             "after_main_entered": False}
    original_readback = runner.read_persisted_rows

    def readback(path):
        rows = PersistedRows(original_readback(path))
        state["persisted_reference"] = weakref.ref(rows)
        return rows

    monkeypatch.setattr(runner, "read_persisted_rows", readback)

    def json_data(value):
        if alias:
            return value
        converted = json.loads(json.dumps(value))
        if isinstance(value, InputRow):
            converted = ObservedRow(converted)
            state["observed_references"].append(weakref.ref(converted))
        return converted

    def run(actual_prepared, observer):
        assert actual_prepared is prepared
        state["calls"].append("integrate")
        for row in raw_rows:
            observer(row)
        if interrupted:
            exception = (KeyboardInterrupt("bounded interrupted prefix") if fail == "interrupt"
                         else RuntimeError("bounded failed prefix"))
            exception.result = raw_result
            raise exception
        return raw_result

    def persist(path, result):
        state["calls"].append("numeric_write")
        assert result is raw_result
        if fail == "numeric_write":
            raise OSError("bounded NPZ write failure")
        with path.open("wb") as stream:
            np.savez_compressed(stream, words=np.array([
                row["state"]["words"] for row in result["accepted_steps"]], dtype=np.float64))

    def verify(path, result):
        state["calls"].append("numeric_readback")
        if fail == "numeric_readback":
            raise ValueError("bounded NPZ readback failure")
        with np.load(path, allow_pickle=False) as archive:
            expected = np.array([row["state"]["words"] for row in result["accepted_steps"]])
            assert archive.files == ["words"]
            assert archive["words"].dtype == expected.dtype
            assert archive["words"].tobytes() == expected.tobytes()

    def replay(actual_prepared, result, incomplete, observer):
        assert actual_prepared is prepared and incomplete is interrupted
        state["calls"].append("replay")
        for index, _ in enumerate(result["accepted_steps"]):
            observer({"row_index": index, "verified": True})
        state["replay"] = {"certified": not incomplete}
        return state["replay"]

    def witness(actual_prepared, result, rows, failure):
        state["calls"].append("failure_witness")
        assert actual_prepared is prepared
        assert rows == result["accepted_steps"] == raw_rows
        assert state["persisted_reference"] is None
        assert all(reference() is not None for reference in state["observed_references"])
        return {"accepted_rows": len(rows), "failure_type": failure["type"]}

    def analyze(actual_prepared, result, actual_replay):
        state["calls"].append("after_main")
        state["after_main_entered"] = True
        assert actual_prepared is prepared
        assert actual_prepared.to_dict() == state["prepared_snapshot"]
        assert result == raw_result == state["raw_snapshot"]
        assert actual_replay is state.get("replay")
        if alias:
            assert result is raw_result
            assert result["accepted_steps"] is raw_rows
        else:
            assert result is not raw_result
            assert result["accepted_steps"] is not raw_rows
            assert all(reference() is None for reference in state["observed_references"])
            assert state["persisted_reference"]() is None
        state["analysis_result"] = result
        return {"passed": not interrupted and actual_replay is not None,
                "accepted_rows": len(result["accepted_steps"])}

    api = {"prepare": lambda: prepared, "run": run, "json_data": json_data,
           "persist_numeric": persist, "verify_numeric": verify,
           "replay": replay, "failure_witness": witness}
    if after_main:
        api["after_main"] = analyze
    return case, state, api


def assert_preserved_files(directory, state):
    assert json.loads((directory / "PreparedV1.json").read_text()) == state["prepared_snapshot"]
    assert json.loads((directory / "ResultV1.json").read_text()) == state["raw_snapshot"]
    rows = [json.loads(line) for line in (directory / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    assert rows == state["raw_snapshot"]["accepted_steps"]
    assert state["raw_result"] == state["raw_snapshot"]
    return rows


@pytest.mark.parametrize("after_main", [True, False])
def test_completed_comparison_copies_die_before_analysis_or_return(tmp_path, monkeypatch, after_main):
    case, state, api = fixture_api(monkeypatch, tmp_path, after_main=after_main)
    events = []

    def observe(phase, event, roots):
        events.append((phase, event))
        if phase == "final_validation":
            assert roots["rows"] == roots["persisted"] == roots["result"]["accepted_steps"]
            assert roots["rows"] is not roots["persisted"]
            assert roots["persisted"] is state["persisted_reference"]()
            assert all(reference() is not None for reference in state["observed_references"])
        if phase == "rows_release" and event == "begin":
            assert roots["rows"] and roots["persisted"]
        if (phase == "rows_release" and event == "end") or phase == "after_main":
            assert roots["rows"] is None and roots["persisted"] is None
            assert all(reference() is None for reference in state["observed_references"])
            assert state["persisted_reference"]() is None
            assert roots["prepared"] is state["prepared"]
            assert roots["result"] == state["raw_snapshot"]
            assert roots["replay"] is state["replay"]

    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    assert report["memory_observation"]["passed"], report["memory_observation"]
    assert report["four_predicates_passed"]
    assert report["cost"]["bytes_per_row"] == report["cost"]["accepted_row_payload_bytes"] / 2
    assert len(state["observed_references"]) == 2
    assert all(reference() is None for reference in state["observed_references"])
    assert state["persisted_reference"]() is None
    assert events.index(("final_validation", "end")) < events.index(("rows_release", "begin"))
    assert events[events.index(("rows_release", "begin")) + 1] == ("rows_release", "end")
    assert state["after_main_entered"] is after_main
    if after_main:
        assert events.index(("rows_release", "end")) < events.index(("after_main", "begin"))
        assert report["independent_analysis"]["passed"]
    assert len(assert_preserved_files(tmp_path, state)) == 2


def test_release_rebinds_shared_lists_without_clearing_input_or_result(tmp_path, monkeypatch):
    case, state, api = fixture_api(monkeypatch, tmp_path, alias=True)
    retained = {}

    def observe(phase, event, roots):
        if (phase, event) == ("final_validation", "end"):
            # Deliberately simulate another owner. Release must not mutate it.
            retained.update(rows=roots["rows"], persisted=roots["persisted"])
        if (phase, event) == ("after_main", "begin"):
            assert roots["rows"] is None and roots["persisted"] is None
            assert len(retained["rows"]) == len(retained["persisted"]) == 2
            assert retained["rows"][0] is roots["result"]["accepted_steps"][0]
            assert retained["rows"] == retained["persisted"] == state["raw_snapshot"]["accepted_steps"]

    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    assert report["memory_observation"]["passed"], report["memory_observation"]
    assert report["four_predicates_passed"] and report["independent_analysis"]["passed"]
    assert retained["rows"] == retained["persisted"] == state["raw_snapshot"]["accepted_steps"]
    assert state["analysis_result"] is state["raw_result"]
    assert_preserved_files(tmp_path, state)


@pytest.mark.parametrize("fail,status,failure_type", [
    ("integrate", "failed", "RuntimeError"),
    ("interrupt", "interrupted", "KeyboardInterrupt"),
])
def test_failed_and_interrupted_prefixes_are_checked_before_release(tmp_path, monkeypatch,
                                                                  fail, status, failure_type):
    case, state, api = fixture_api(monkeypatch, tmp_path, fail=fail)
    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None)
    assert report["execution_status"] == status
    assert report["failure"]["type"] == failure_type
    assert report["extent"]["accepted_rows"] == 1 and not report["extent"]["complete"]
    assert report["four_predicates"]["observer_matches_result"]
    assert not report["four_predicates_passed"]
    assert report["numeric_sidecar_exact"] and report["replay_completed"]
    assert report["replay_observed_rows"] == 1
    assert report["independent_analysis"] == {"passed": False, "accepted_rows": 1}
    assert "failure_witness" in state["calls"]
    assert state["calls"].index("failure_witness") < state["calls"].index("after_main")
    assert json.loads((tmp_path / "FailureWitnessV1.json").read_text()) == {
        "accepted_rows": 1, "failure_type": failure_type}
    assert json.loads((tmp_path / "FailureV1.json").read_text())["type"] == failure_type
    assert len(assert_preserved_files(tmp_path, state)) == 1


@pytest.mark.parametrize("fail,failure_type,npz_exists", [
    ("numeric_write", "OSError", False),
    ("numeric_readback", "ValueError", True),
])
def test_npz_failure_keeps_json_and_complete_observer_prefix(tmp_path, monkeypatch,
                                                          fail, failure_type, npz_exists):
    case, state, api = fixture_api(monkeypatch, tmp_path, fail=fail)
    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None)
    assert report["execution_status"] == "failed"
    assert report["failure"]["type"] == failure_type
    assert not report.get("numeric_sidecar_exact", False)
    assert not report.get("replay_completed", False)
    assert report["extent"]["complete"] and report["extent"]["accepted_rows"] == 2
    assert report["four_predicates"]["observer_matches_result"]
    assert not report["four_predicates_passed"]
    assert report["replay_observed_rows"] == 0
    assert "replay" not in state["calls"] and "failure_witness" not in state["calls"]
    assert report["independent_analysis"] == {"passed": False, "accepted_rows": 2}
    assert (tmp_path / "StateArraysV1.npz").exists() is npz_exists
    assert len(assert_preserved_files(tmp_path, state)) == 2


def test_release_duration_and_observer_overhead_stay_outside_main_cost(tmp_path, monkeypatch):
    case, state, api = fixture_api(monkeypatch, tmp_path)
    clock = {"time": 100.0}
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["time"])
    events = []

    def observe(phase, event, roots):
        events.append((phase, event))
        if phase == "rows_release":
            clock["time"] += 7.0
        elif phase == "after_main":
            clock["time"] += 3.0
        else:
            clock["time"] += 0.25

    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    main_events = sum(phase not in {"rows_release", "after_main"} for phase, _ in events)
    assert report["memory_observation"]["passed"]
    assert report["row_release_elapsed_s"] == 14.0
    assert report["row_release_cost_scope"] == "outside_original_main_timer_included_in_process_wall_time"
    assert report["cost"]["elapsed_s"] == main_events * 0.25
    assert report["memory_observation"]["main_elapsed_s"] == main_events * 0.25
    assert report["memory_observation"]["elapsed_s"] == main_events * 0.25 + 20.0
    assert report["independent_analysis_elapsed_s"] == 6.0
    assert report["phase_timing"]["elapsed_s"] == report["cost"]["elapsed_s"]
    assert report["four_predicates_passed"] and state["after_main_entered"]


def test_release_observer_failure_preserves_cleanup_and_scientific_artifacts(tmp_path, monkeypatch):
    case, state, api = fixture_api(monkeypatch, tmp_path)

    def observe(phase, event, roots):
        if (phase, event) == ("rows_release", "begin"):
            raise RuntimeError("bounded release observer failure")

    report = runner.run_trajectory(tmp_path, case, "baseline", api, lambda: None,
                                   memory_observer=observe)
    assert not report["memory_observation"]["passed"]
    assert report["memory_observation"]["error"]["message"] == "bounded release observer failure"
    assert report["four_predicates_passed"] and report["independent_analysis"]["passed"]
    assert report["failure"] is None
    assert all(reference() is None for reference in state["observed_references"])
    assert state["persisted_reference"]() is None
    assert_preserved_files(tmp_path, state)

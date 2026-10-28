"""Saved-observer orchestration with manufactured authority and data only.

The science plan validator and numerical observer are stand-ins. File pins,
launcher receipt checks, process/clock guards and the history writer are real.
No fixture here grants runtime authority or enters a native producer.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest


# The new entry can be reviewed independently while legacy imports resolve to
# the separately frozen execution tree selected by the test environment.
ENTRY = Path(__file__).resolve().parents[2] / "scripts/benchmarks/g2_saved_observe.py"
SPEC = importlib.util.spec_from_file_location("_saved_observe_under_test", ENTRY)
saved = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(saved)


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True)+"\n")
    return pin(path)


def closure(pid, code):
    return {"schema": "solarlab.owned-tool-process-closure.v1", "pgid": pid,
            "after_tool_exit_code": code, "process_group_absent": True,
            "all_named_owned_processes_and_groups_absent": True, "error": None,
            "remaining_group_members": [], "remaining_owned_processes": [],
            "checks": [{"pid": pid, "pid_absent": True, "same_id_process_group_absent": True}]}


@pytest.fixture
def bound(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    source = tmp_path / "producer"
    source.mkdir()
    (source / "Observation").mkdir()
    (source / "Observation/old-partial").write_bytes(b"immutable failed observation")
    target = tmp_path / "recovery"
    roots = {ENTRY, Path(saved.route.__file__), Path(saved.launcher.__file__)}
    roots.update(Path(saved.route.__file__).with_name(n) for n in saved.route.EVALUATOR_FILES)
    sources = {str(p.resolve()): pin(p)["sha256"] for p in roots}
    gate = write(tmp_path / "Gate.json", {"fixture_only": True})
    request = {"numeric_packet": {"definition": {}},
               "voltage_lift_map": {"mapped_input_profile": "fixture"}}
    request_pin = write(tmp_path / "Request.json", request)
    (source / "NativeRequest.json").write_bytes(Path(request_pin["path"]).read_bytes())
    supervision = {"poll_s": .01, "term_grace_s": .1, "kill_wait_s": .1,
                   "receipt_reserve_s": .1, "rss_bytes": saved.GIB,
                   "artifact_bytes": 8*1024**2, "closure_path": str(tmp_path / "ProducerReceipt.json"),
                   "cancel_path": str(tmp_path / "old.cancel"),
                   "pipeline_started_unix_ns": time.time_ns()-2*10**9, "pipeline_wall_s": 90}
    execution = {"directory": str(source), "channel": 0, "command": ["manufactured", "run"],
                 "wall_s": 90, "supervision": supervision, "request": request_pin,
                 "effective_input_digest": saved.route._digest(request),
                 "max_record_bytes": 1024**2, "max_records": 5000,
                 "integer_io_policy": None,
                 "native_admission": {"operation_id": "original_producer", "fixture_only": True},
                 "execution_provenance": {"original_producer": "unchanged"}}
    case = {"id": "B_N8_Coarse_Probe", "mesh_intervals": 8, "execution": execution}
    plan = {"cases": [case], "definitions": {"gate_spec": gate},
            "source_freeze": {"future_execution_commit": saved.LEGACY_COMMIT,
                              "source_sha256": sources, "frozen_gate_spec": gate},
            "resources": {"rss_bytes_per_case": 4*saved.GIB,
                          "combined_persisted_case_bytes": 4*saved.GIB,
                          "reader_logical_byte_cap": 32*saved.GIB},
            "artifacts": {"channel_directory": str(tmp_path / "channels")}}
    producer_pin = write(tmp_path / "ProducerPlan.json", plan)
    observed = deepcopy(plan)
    prior = observed["cases"][0]["execution"]
    prior["command"] = ["manufactured", "observe"]
    prior["supervision"]["closure_path"] = str(tmp_path / "FailedObserveReceipt.json")
    observed_pin = write(tmp_path / "ObservePlan.json", observed)
    receipt = {"schema": "solarlab.g2-route.launcher-result.v1", "case_id": case["id"],
               "plan_sha256": producer_pin["sha256"], "channel": 0,
               "channel_protocol": saved.launcher.CHANNEL_PROTOCOL,
               "command": execution["command"], "cwd": str(saved.launcher.REPOSITORY),
               "process_outcome": "completed", "termination_reason": "completed", "exit_code": 0,
               "exit_signal": None, "cancellation_signal": None, "error": None, "signals_sent": [],
               "pid": 900001, "pgid": 900001, "session_id": 900001, "observed_pids": [900001],
               "closure": closure(900001, 0), "limits": vars(saved.launcher.Limits.from_execution(execution)),
               "wall_s": 1, "receipt_initial_write_s": .001, "peak_group_rss_bytes": 0,
               "artifact_bytes_before_receipt": 0, "receipt_bytes": 0, "artifact_bytes": 0,
               "peak_artifact_bytes": 0, "last_samples": [], "rss_peak_sample": None,
               "artifact_peak_sample": None}
    path = Path(supervision["closure_path"])
    for _ in range(6):
        ref = write(path, receipt)
        if receipt["receipt_bytes"] == ref["bytes"]:
            break
        receipt.update(receipt_bytes=ref["bytes"], artifact_bytes=ref["bytes"], peak_artifact_bytes=ref["bytes"])
    assert receipt["receipt_bytes"] == path.stat().st_size
    failed = deepcopy(receipt)
    failed.update(plan_sha256=observed_pin["sha256"], command=prior["command"],
                  process_outcome="wall_limit", termination_reason="wall_limit", exit_code=-15,
                  exit_signal=15, pid=900002, pgid=900002, session_id=900002,
                  observed_pids=[900002], closure=closure(900002, -15))
    failed_pin = write(Path(prior["supervision"]["closure_path"]), failed)
    result_pin = write(source / "NativeResult.json", {
        "complete_run_window": True, "request_sha256": execution["effective_input_digest"],
        "history": {"closed": True, "container_complete": True, "error": None}})
    review_pin = write(tmp_path / "SourceReview.json", {
        "accepted": True, "source_sha256": sources, "legacy_execution_commit": saved.LEGACY_COMMIT})
    recovery = deepcopy(execution)
    recovery.update(directory=str(target), cwd=str(saved.launcher.REPOSITORY))
    recovery["supervision"].update(pipeline_started_unix_ns=time.time_ns(),
        closure_path=str(tmp_path / "RecoveryClosure.json"), cancel_path=str(tmp_path / "recovery.cancel"))
    manifest = {"schema": saved.SCHEMA, "operation_id": "separate_recovery", "case_id": case["id"],
        "authority": {"role": "Root", "authorized": True, "scope": "saved-observation-only",
                      "coordinator_message": "msg_manufactured_unit_fixture"},
        "source_directory": str(source), "output_directory": str(target),
        "producer_plan": producer_pin, "producer_result": result_pin, "producer_closure": ref,
        "failed_observe_plan": observed_pin, "failed_observe_closure": failed_pin,
        "source_review": review_pin, "source_sha256": sources, "execution": recovery,
        "read_limits": {"max_record_bytes": 1024**2, "max_records": 5000, "max_logical_bytes": 32*1024**2}}
    manifest_path = tmp_path / "Recovery.json"
    def persist():
        write(manifest_path, manifest)
        return manifest_path
    validations = []
    def validate_plan(path, **kwargs):
        validations.append(kwargs)
        return {"plan": json.loads(Path(path).read_bytes()), "plan_sha256": pin(Path(path))["sha256"]}
    def absent(pid, number):
        assert number == 0
        raise ProcessLookupError(pid)
    monkeypatch.setattr(saved.route, "validate", validate_plan)
    monkeypatch.setattr(saved.os, "kill", absent)
    monkeypatch.setattr(saved.os, "killpg", absent)
    return SimpleNamespace(manifest=manifest, persist=persist, source=source, target=target,
                           plan=plan, validations=validations, tmp=tmp_path)


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_validate_only_preserves_input_and_checks_evaluation(bound, monkeypatch, capsys):
    before = snapshot(bound.source)
    monkeypatch.setattr(saved, "_observe", lambda *args: pytest.fail("observer entered"))
    assert saved.main(["--manifest", str(bound.persist()), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated_not_executed"
    assert bound.validations[-1]["purpose"] == "evaluate"
    assert not bound.target.exists() and snapshot(bound.source) == before


@pytest.mark.parametrize("location", ["same", "nested", "parent", "existing", "alias", "legacy_tree"])
def test_output_overlap_or_existing_path_refused(bound, location):
    locations = {"same": bound.source, "nested": bound.source / "new", "parent": bound.source.parent,
                 "existing": bound.tmp / "occupied", "alias": bound.tmp / "alias",
                 "legacy_tree": saved.launcher.REPOSITORY / "forbidden-recovery"}
    if location == "existing":
        locations[location].mkdir()
    if location == "alias":
        locations[location].symlink_to(bound.source, target_is_directory=True)
    bound.manifest["output_directory"] = str(locations[location])
    with pytest.raises(ValueError):
        saved.validate(bound.persist())


@pytest.mark.parametrize("field", ["producer_plan", "producer_result", "producer_closure",
                                   "failed_observe_plan", "failed_observe_closure", "source_review"])
def test_wrong_upstream_pin_refused_before_output(bound, field):
    bound.manifest[field]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="reference_changed"):
        saved.validate(bound.persist())
    assert not bound.target.exists()
    if field == "producer_plan":
        assert not bound.validations


@pytest.mark.parametrize("change", ["authority", "source", "read_limit", "extra_limit", "old_clock",
                                    "control_inside_source", "missing", "bad_root", "bad_container"])
def test_authority_limits_and_malformed_metadata_refused(bound, change):
    m = bound.manifest
    if change == "authority":
        m["authority"]["authorized"] = 1
    elif change == "source":
        m["source_sha256"][str(ENTRY)] = "0"*64
    elif change == "read_limit":
        m["read_limits"]["max_record_bytes"] = 1024**2+1
    elif change == "extra_limit":
        m["read_limits"]["charge_observer"] = None
    elif change == "old_clock":
        m["execution"]["supervision"]["pipeline_started_unix_ns"] = bound.plan["cases"][0]["execution"]["supervision"]["pipeline_started_unix_ns"]
    elif change == "control_inside_source":
        m["execution"]["supervision"]["cancel_path"] = str(bound.source / "cancel")
    elif change == "missing":
        del m["producer_plan"]
    elif change == "bad_root":
        bound.manifest = []
        path = bound.tmp / "BadRoot.json"
        write(path, [])
        with pytest.raises(ValueError):
            saved.validate(path)
        return
    else:
        m["execution"] = []
    with pytest.raises(ValueError):
        saved.validate(bound.persist())
    assert not bound.target.exists()


@pytest.mark.parametrize("problem", ["live_pid", "live_group", "residue", "bad_check", "command"])
def test_failed_observer_must_be_bound_and_actually_closed(bound, monkeypatch, problem):
    if problem in ("live_pid", "live_group"):
        monkeypatch.setattr(saved.os, "kill" if problem == "live_pid" else "killpg", lambda *args: None)
    else:
        key = "failed_observe_closure"
        ref = bound.manifest[key]
        value = json.loads(Path(ref["path"]).read_bytes())
        if problem == "residue":
            value["closure"]["remaining_group_members"] = [1234]
        elif problem == "bad_check":
            value["closure"]["checks"][0]["pid_absent"] = False
        else:
            value["command"] = ["foreign"]
        bound.manifest[key] = write(Path(ref["path"]), value)
    with pytest.raises(ValueError):
        saved.validate(bound.persist())


def install_observer_stubs(monkeypatch, callback):
    calls = {}
    def module(name, **values):
        value = ModuleType("scripts.benchmarks."+name)
        value.__dict__.update(values)
        monkeypatch.setitem(sys.modules, value.__name__, value)
    def observer(mapping, request, append, checkpoint, *, t1_context):
        calls.update(t1_context=t1_context, append=append)
        return "original-observer-instance"
    def reader(source, **kwargs):
        calls.update(source=source, reader=kwargs)
        kwargs["sample_sink"]("tag", "sample", "certificate")
        callback(calls)
        return {"scientific_status": "unknown", "scientific_pass": False,
                "device": {"status": "unknown_reference_error"}}
    module("coupled_device_prototype", SlabDefinition=lambda **kw: kw,
           AffineCoupledSlab=lambda *args: "model", AffineVoltageMap=lambda *args, **kw: "mapping",
           run_voltage_lift_native_pilot=lambda *args, **kw: pytest.fail("producer entered"))
    module("g2_compare", case_metadata=lambda *args: {}, normalize_ida_sample=lambda *args: {"fixture": True})
    module("native_observation", SealedChargeObserver=observer)
    module("native_readback", verify_history=reader)
    return calls


def permit_run(monkeypatch):
    for key in saved.launcher.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(saved, "_blas_one", lambda: None)
    monkeypatch.setattr(saved.launcher, "inherited_channel", lambda *args: nullcontext(7))


def test_original_admission_and_t1_identity_forwarded_unchanged(bound, monkeypatch):
    from scripts.benchmarks.native_history import HistoryReader
    permit_run(monkeypatch)
    calls = install_observer_stubs(monkeypatch, lambda calls: None)
    before = snapshot(bound.source)
    result = saved.run(bound.persist())
    execution = bound.plan["cases"][0]["execution"]
    assert calls["source"] == bound.source
    assert calls["reader"]["execution_admission"] == execution["native_admission"]
    assert calls["reader"]["execution_provenance"] == execution["execution_provenance"]
    assert calls["t1_context"]["run_id"] == "original_producer"
    assert calls["reader"]["charge_observer"] == "original-observer-instance"
    assert result["device"]["status"] == "unknown_reference_error" and result["scientific_pass"] is False
    assert result["plan_sha256"] == bound.manifest["producer_plan"]["sha256"]
    receipt = json.loads((bound.target / "RecoveryReceipt.json").read_bytes())
    assert receipt["operation_id"] == "separate_recovery" and receipt["producer_runs"] == 0
    with HistoryReader(bound.target / "Charge.jsonl.gz", encoding="gzip", max_record_bytes=1024**2,
                       max_logical_bytes=4*1024**2) as reader:
        rows = [json.loads(line) for line in reader]
    assert rows[0]["evidence"]["run_id"] == "original_producer"
    assert snapshot(bound.source) == before


@pytest.mark.parametrize("failure", ["exception", "cancel", "signal"])
def test_failure_keeps_old_data_and_new_first_failure(bound, monkeypatch, failure):
    permit_run(monkeypatch)
    cancel = Path(bound.manifest["execution"]["supervision"]["cancel_path"])
    def fail(calls):
        if failure == "exception":
            raise RuntimeError("first reader failure")
        if failure == "cancel":
            cancel.touch()
        else:
            signal.raise_signal(signal.SIGTERM)
    install_observer_stubs(monkeypatch, fail)
    before = snapshot(bound.source)
    with pytest.raises((ValueError, RuntimeError)):
        saved.run(bound.persist())
    assert (bound.target / "FirstFailure.json").is_file()
    assert not (bound.target / "RunResult.json").exists()
    assert not (bound.target / "RecoveryReceipt.json").exists()
    assert snapshot(bound.source) == before


@pytest.mark.parametrize("guard", ["expired", "future", "rss", "missing_channel"])
def test_runtime_guard_stops_before_observer(bound, monkeypatch, guard):
    for key in saved.launcher.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(saved, "_blas_one", lambda: None)
    monkeypatch.setattr(saved, "_observe", lambda *args: pytest.fail("observer entered"))
    if guard in ("expired", "future"):
        bound.manifest["execution"]["supervision"]["pipeline_started_unix_ns"] = time.time_ns() + (-1000 if guard == "expired" else 1000)*10**9
    elif guard == "rss":
        monkeypatch.setattr(saved.resource, "getrusage", lambda *args: SimpleNamespace(ru_maxrss=10**15))
    else:
        monkeypatch.delenv(saved.launcher.CHANNEL_FD_ENV, raising=False)
    with pytest.raises(ValueError):
        saved.run(bound.persist())
    assert not bound.target.exists()


def test_inherited_lock_remains_owned_after_consumer_exception(bound, monkeypatch):
    import fcntl
    permit_run(monkeypatch)
    # Use the unchanged actual inherited FD context, without a subprocess.
    spec = importlib.util.spec_from_file_location("_saved_lock_test", saved.launcher.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(saved.launcher, "inherited_channel", module.inherited_channel)
    slots = Path(bound.plan["artifacts"]["channel_directory"])
    slots.mkdir()
    slot = slots / "0.lock"
    with slot.open("w") as owner, slot.open("r") as competitor:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv(module.CHANNEL_FD_ENV, str(owner.fileno()))
        def fail(*args):
            raise RuntimeError("synthetic observer refusal")
        monkeypatch.setattr(saved, "_observe", fail)
        with pytest.raises(RuntimeError, match="synthetic observer refusal"):
            saved.run(bound.persist())
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)

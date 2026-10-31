"""BDF publication and solver-free complete/failed history contracts."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from solarlab.experiments.bdf_results import (
    _canonical, _decode, _digest, _encode, _input_record, _validate_history, _work, read_bdf_packet, read_bdf_result,
)
from solarlab.experiments.transient_results import (
    publish_transient_result, read_transient_result, run_transient_for_publication,
)
from solarlab.io.artifacts import ArtifactIntegrityError, bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab.reproducibility import job_provenance as provenance

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/integration/fixtures/bdf_readiness"


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    from examples.research_1d.steady import load_device
    from examples.research_1d.transient import controls
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.solvers.nonlinear.problem import GateOutcome, SolveBudget
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
    from solarlab.solvers.time.bdf import BDFOptions
    from solarlab.solvers.time.bdf_1d import BDFDrive
    from solarlab.experiments import bdf as bdf_backend
    from support.equation_provenance import counted
    from support.provenance import checked_provenance_owner

    root = tmp_path_factory.mktemp("bdf-publication")
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_options = SteadyOptions()
    initial = solve_steady_state(compiled, voltage_V=0., generation_scale=1., options=initial_options)
    protocol = TransientProtocol([0, 1e-7, 2e-7], [0, 0, 0], [1, 1.0001, 1.0001])
    drive = BDFDrive(protocol.times_s, np.array([[0., 1.], [0., 1.], [0., 1.0001]]),
                     np.array([[0., 1.], [0., 1.0001], [0., 1.0001]]))
    bdf = BDFOptions(initial_step_s=1e-7, max_order=1, adaptive=False, max_steps=100, wall_s=20.)
    captures = {}
    with RunStore(root / "store") as store:
        for name in ("completed", "cap", "newton", "restart-rejected", "restart-at-end", "restart-at-start"):
            options = controls()
            requested = bdf
            active_drive = drive
            active_protocol = protocol
            if name == "cap":
                options = replace(options, max_steps=1)
            if name == "newton":
                options = replace(options, newton_budget=SolveBudget(1, 10, 1, 4))
                active_protocol = TransientProtocol([0, 1e-7, 2e-7], [0, 0, 0], [1, 1, 1.5])
                active_drive = None
            if name in {"restart-at-start", "restart-at-end"}:
                active_protocol = TransientProtocol([0, 2e-7], [0, 0], [1, 1.0001])
                left = [[0., 1.], [0., 1.]] if name == "restart-at-end" else [[0., 1.], [0., 1.0001]]
                right = [[0., 1.], [0., 1.0001]] if name == "restart-at-end" else [[0., 1.0001], [0., 1.0001]]
                active_drive = BDFDrive(active_protocol.times_s, np.array(left), np.array(right))
            with checked_provenance_owner(), counted() as observed, pytest.MonkeyPatch.context() as patch:
                if name.startswith("restart-"):
                    native_events = bdf_backend._events

                    def rejecting_events(*args):
                        events = native_events(*args)

                        def reject(old, new, u):
                            right, assessment = events[0].restart(old, new, u)
                            # A tiny synthetic refused event exercises the real
                            # failure path; production controls are untouched.
                            return right, replace(assessment, checks=(*assessment.checks,
                                GateOutcome("test_restart_refusal", False, 1., 0.)))

                        return (replace(events[0], restart=reject), *events[1:])

                    patch.setattr(bdf_backend, "_events", rejecting_events)
                capture = run_transient_for_publication(compiled, initial, active_protocol,
                    options=options, initial_options=initial_options, method="bdf", bdf_options=requested,
                    bdf_drive=active_drive)
            assert observed.original > 0 and observed.assembled == 0
            assert capture.result.completed == (name == "completed")
            row = store.enqueue(name, opaque_identity=capture.identity, input_metadata={"fixture": name})
            token = store.claim(name, expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="test")
            document = publish_transient_result(store, token, capture.result, nodes_m=compiled.nodes_m, capture=capture)
            with pytest.raises(ValueError, match="committed terminal"):
                read_bdf_result(store, name)
            store.complete(token, state="succeeded" if name == "completed" else "failed",
                           result=document, artifacts=document["artifact_ids"])
            captures[name] = capture
        from examples.research_1d.ionic_residual import prepare_single_ion_device
        from examples.research_1d.ionic_steady import controls as ionic_steady_controls
        from examples.research_1d.ionic_transient import controls as ionic_controls
        from examples.research_1d.transient import json_value
        from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
        from solarlab.experiments.bdf_results import run_bdf_for_publication
        from solarlab.experiments.ionic_transient import load_ionic_steady_initial
        from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state

        device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
        admission = ionic_steady_controls(device.ion_node_count)
        point = solve_ionic_steady_state(device, voltage_V=.002, generation_scale=0., options=admission)
        assert point.accepted
        saved = {"summary": {"accepted": True, "compiled_identity": device.identity,
                            "voltage_V": .002, "generation_scale": 0.}, "result": point}
        raw = (json.dumps(json_value(saved), allow_nan=False) + "\n").encode()
        initial_path = root / "ionic_initial.json"
        initial_path.write_bytes(raw)
        ion_initial = load_ionic_steady_initial(device, initial_path, hashlib.sha256(raw).hexdigest(), controls=admission)
        times = np.linspace(0., 1., 5)
        for name in ("ionic-completed", "ionic-cap"):
            physical = ionic_controls()
            if name == "ionic-cap":
                physical = replace(physical, electronic=replace(physical.electronic, max_steps=1))
            with checked_provenance_owner(), counted() as observed:
                capture = run_bdf_for_publication(device, ion_initial,
                    TransientProtocol(times, .002 + .002 * times, np.zeros(5)), options=physical,
                    bdf_options=BDFOptions(.25, adaptive=False, max_order=1))
            assert observed.original > 0 and observed.assembled == 0
            assert capture.result.completed == (name == "ionic-completed")
            row = store.enqueue(name, opaque_identity=capture.identity, input_metadata={"fixture": name})
            token = store.claim(name, expected_attempt_id=row["current_attempt_id"], expected_generation=0, owner_id="test")
            document = publish_transient_result(store, token, capture.result, nodes_m=device.nodes_m, capture=capture)
            store.complete(token, state="succeeded" if capture.result.completed else "failed",
                           result=document, artifacts=document["artifact_ids"])
            captures[name] = capture
    return root, captures


@pytest.mark.parametrize("name", ["completed", "cap", "newton", "restart-rejected", "restart-at-end",
                                  "restart-at-start", "ionic-completed", "ionic-cap"])
def test_bdf_capture_keeps_the_full_original_only_wiring_binding(published, name):
    from solarlab.equations.implementation import BINDING_SCHEMA

    root, captures = published
    with RunStore(root / "store", read_only=True, immutable=True) as store:
        loaded = read_bdf_result(store, name)
    implementation = loaded["record"]["request"]["equation_implementation"]
    operator = "BDFStep:" + ("Ionic1DProblem" if name.startswith("ionic-") else "DC1DProblem")
    expected = {"operator": operator, "assembler_identity": None, "analysis_identity": provenance.digest({
        "schema": BINDING_SCHEMA, "implementation": "legacy", "analysis": "transient",
        "operator": operator, "assemblers": {}})}
    assert implementation["kind"] == "original" and implementation["unified_assembler"] is False
    assert implementation["provenance_binding"] == expected
    description = json.loads(captures[name].capture_json)["description"]
    assert description["equation_implementation"] == implementation
    admitted = provenance.read_provenance(description["job_provenance"]).export()["assembler"]
    assert admitted == {"operator": operator, "identity": None, "unified_assembler": False,
                        "analysis_identity": expected["analysis_identity"]}


@pytest.mark.parametrize("operator", ["BDFStep:DC1DProblem", "BDFStep:Ionic1DProblem"])
def test_bdf_capability_distinguishes_implicit_default_from_forced_assembled(monkeypatch, operator):
    from solarlab.equations.implementation import (
        ASSEMBLED, ENVIRONMENT, LEGACY, equation_implementation, provenance_binding, selected_implementation,
    )

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    assert selected_implementation() == ASSEMBLED
    assert selected_implementation(operator=operator) == LEGACY
    binding = provenance_binding(operator, object(), assembled=False)
    assert binding["operator"] == operator and binding["assembler_identity"] is None
    with pytest.raises(ValueError, match="BDF currently supports original equations only"):
        provenance_binding(operator, object(), assembled=True)
    with equation_implementation(LEGACY):
        assert selected_implementation(operator=operator) == LEGACY
    with equation_implementation(ASSEMBLED), pytest.raises(ValueError, match="original equations only"):
        provenance_binding(operator, object(), assembled=False)
    assert selected_implementation() == ASSEMBLED
    monkeypatch.setenv(ENVIRONMENT, LEGACY)
    assert selected_implementation(operator=operator) == LEGACY
    monkeypatch.setenv(ENVIRONMENT, ASSEMBLED)
    with pytest.raises(ValueError, match="original equations only"):
        selected_implementation(operator=operator)


@pytest.mark.parametrize("ionic", [False, True])
@pytest.mark.parametrize("forced", ["environment", "scope"])
def test_explicit_assembled_bdf_publication_refuses_before_admission(monkeypatch, ionic, forced):
    from examples.research_1d.ionic_residual import prepare_single_ion_device
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from solarlab.equations.implementation import ASSEMBLED, ENVIRONMENT, equation_implementation
    from solarlab.experiments import bdf_results
    from solarlab.solvers.time.bdf import BDFOptions

    compiled = (compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
                if ionic else compile_device_1d(load_device(), intervals_per_layer=12))
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    if forced == "environment":
        monkeypatch.setenv(ENVIRONMENT, ASSEMBLED)
    monkeypatch.setattr(bdf_results, "job_provenance", lambda *a, **k: pytest.fail("refusal must precede admission"))
    with equation_implementation(ASSEMBLED) if forced == "scope" else nullcontext():
        with pytest.raises(ValueError, match="BDF currently supports original equations only"):
            bdf_results.run_bdf_for_publication(compiled, None, None, options=None,
                initial_options=object(), bdf_options=BDFOptions(1e-7))
        if ionic:
            from solarlab.experiments.ionic_transient import solve_ionic_transient
            with pytest.raises(ValueError, match="BDF currently supports original equations only"):
                solve_ionic_transient(compiled, None, None, options=None, method="bdf", bdf_options=BDFOptions(1e-7))
        else:
            from solarlab.experiments.transient import solve_transient
            with pytest.raises(ValueError, match="BDF currently supports original equations only"):
                run_transient_for_publication(compiled, None, None, options=None, initial_options=object(),
                                              method="bdf", bdf_options=BDFOptions(1e-7))
            with pytest.raises(ValueError, match="BDF currently supports original equations only"):
                solve_transient(compiled, None, None, options=None, initial_options=object(),
                                method="bdf", bdf_options=BDFOptions(1e-7))


@pytest.mark.parametrize("name", ["completed", "cap", "newton"])
def test_completed_and_failed_bdf_round_trip(published, name):
    root, captures = published
    capture = captures[name]
    with RunStore(root / "store", read_only=True, immutable=True) as store:
        loaded = read_transient_result(store, name)
    assert _digest(loaded["history"]) == _digest(capture.result)
    request = loaded["record"]["request"]
    effective = loaded["record"]["effective_bdf_options"]
    assert request["method"] == "bdf"
    assert request["bdf_options"]["max_steps"] == 100
    assert effective["max_steps"] == (1 if name == "cap" else 32)
    assert loaded["record"]["remaining_wall_budget_s"] == capture.result.controls["wall_s"]
    assert loaded["record"]["remaining_wall_budget_s"] < request["bdf_options"]["wall_s"]
    assert loaded["summary"]["qualification"] is False
    assert loaded["summary"]["label"] == "experimental_unqualified"
    assert loaded["summary"]["last_accepted_time_s"] == capture.result.accepted_times_s[-1]
    assert loaded["summary"]["requested_end_s"] == 2e-7
    if name != "completed":
        assert loaded["requested_end_state"] is None
        assert loaded["summary"]["requested_end_reached"] is False
        assert loaded["summary"]["last_accepted_time_s"] < loaded["summary"]["requested_end_s"]
        assert loaded["history"]["samples"] == ()
    if name == "completed":
        assert loaded["history"]["restarts"]
        restart = loaded["history"]["restarts"][0]
        assert restart["time_s"] == 1e-7
        assert not np.array_equal(request["drive"]["left_values"], request["drive"]["right_values"])
    if name == "newton":
        assert any(not step["accepted"] for step in loaded["history"]["attempts"])


def test_solver_free_fresh_process_reads_every_history(published):
    root, _ = published
    code = '''
import hashlib, importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[2])
class NoSolvers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.experiments.transient',
            'solarlab.experiments.ionic_transient')):
            raise AssertionError('reader imported a solver: ' + fullname)
sys.meta_path.insert(0, NoSolvers())
from solarlab.io.run_store import RunStore
from solarlab.experiments.transient_results import read_transient_result
from solarlab.experiments.bdf_results import read_bdf_result
assert hashlib.sha256(pathlib.Path(read_bdf_result.__code__.co_filename).read_bytes()).hexdigest() == sys.argv[3]
with RunStore(sys.argv[1], read_only=True, immutable=True) as store:
    for name in ('completed','cap','newton','restart-rejected','restart-at-end','restart-at-start','ionic-completed','ionic-cap'):
        read_transient_result(store, name)
assert not any(n == 'solarlab.solvers' or n.startswith('solarlab.solvers.') for n in sys.modules)
'''
    reader = Path(read_bdf_result.__code__.co_filename).resolve()
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root / "store"),
                             str(reader.parents[2]), hashlib.sha256(reader.read_bytes()).hexdigest()],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["ionic-completed", "ionic-cap"])
def test_ionic_history_retains_its_original_initial_evidence(published, name):
    root, captures = published
    with RunStore(root / "store", read_only=True, immutable=True) as store:
        loaded = read_bdf_result(store, name)
    assert _digest(loaded["history"]) == _digest(captures[name].result)
    record = loaded["record"]
    assert record["request"]["initial_source_sha256"] == loaded["history"]["controls"]["initial_source_sha256"]
    assert record["request"]["equation_implementation"]["problem"] == "solarlab.equations.ionic_1d.Ionic1DProblem"
    assert loaded["summary"]["completed"] == (name == "ionic-completed")
    if name == "ionic-cap":
        assert loaded["requested_end_state"] is None
        assert loaded["summary"]["last_accepted_time_s"] == .25


@pytest.mark.parametrize("name", ["completed", "cap", "newton", "ionic-completed", "ionic-cap"])
def test_source_drift_is_information_and_strict_mode_refuses(published, tmp_path, monkeypatch, name):
    root, _ = published
    source = tmp_path / "later_source.py"
    source.write_bytes(b"changed implementation")
    with RunStore(root / "store", read_only=True, immutable=True) as store:
        before = read_bdf_result(store, name)
        producer = provenance.read_provenance(before["manifest"]["provenance"]["description"]["job_provenance"])
        if producer.export()["source_state"] == "clean":
            same = read_bdf_result(store, name, current_identity=producer, strict_same_source=True)
            assert same["provenance_comparison"]["status"] == "same"
        else:
            with pytest.raises(ValueError, match="strict_same_source"):
                read_bdf_result(store, name, current_identity=producer, strict_same_source=True)
        current = provenance.capture_job_provenance({}, dirty_policy="record", operator=producer.export()["assembler"]["operator"],
            source_files=(("solarlab/later_source.py", source.read_bytes()),), source_attested=True,
            git_commit=producer.export()["git_commit"], environment_lock=b"changed lock")
        source.write_bytes(b"another edit after current admission")
        monkeypatch.setattr(provenance, "capture_job_provenance", lambda *a, **k: pytest.fail("reader recaptured source"))
        after = read_bdf_result(store, name, current_identity=current)
        assert after["provenance_comparison"]["status"] == "different"
        assert _digest(after["record"]) == _digest(before["record"])
        assert after["manifest"]["provenance"] == before["manifest"]["provenance"]
        with pytest.raises(ValueError, match="strict_same_source"):
            read_bdf_result(store, name, current_identity=current, strict_same_source=True)


@pytest.mark.parametrize("name", ["completed", "cap", "newton"])
def test_history_bytes_tampering_refuses(published, tmp_path, name):
    root, _ = published
    shutil.copytree(root / "store", tmp_path / "store")
    with RunStore(tmp_path / "store", read_only=True, immutable=True) as store:
        loaded = read_bdf_result(store, name)
        record = store.artifact(loaded["manifest"]["history_artifact_id"])
        path = store.root / record["relative_path"]
        original = path.read_bytes()
        path.write_bytes(original[:-1] + b" ")
        with pytest.raises(ArtifactIntegrityError):
            read_bdf_result(store, name)


def test_mutated_capture_and_mixed_methods_refuse_before_publication(published, tmp_path):
    _, captures = published
    capture = captures["completed"]
    with RunStore(tmp_path / "store") as store:
        row = store.enqueue("bad", opaque_identity=capture.identity, input_metadata={})
        token = store.claim("bad", expected_attempt_id=row["current_attempt_id"], expected_generation=0, owner_id="test")
        bad_result = replace(capture.result, controls={**capture.result.controls, "method": "backward_euler"})
        with pytest.raises(ValueError, match="differs from result"):
            publish_transient_result(store, token, bad_result, nodes_m=capture.nodes_m, capture=capture)
        metadata = json.loads(capture.capture_json)
        metadata["description"]["input_record"]["request_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="input binding"):
            publish_transient_result(store, token, capture.result, nodes_m=capture.nodes_m,
                                     capture=replace(capture, capture_json=json.dumps(metadata)))
        assert store.events(run_id="bad")[-1]["kind"] == "running"


def test_sample_failure_at_end_does_not_claim_completed_request(published):
    _, captures = published
    history = _decode(_encode(captures["completed"].result))
    history["completed"] = False
    history["termination"] = "sample_projection_failed"
    history["samples"] = history["samples"][:1]
    history["sample_failure"] = {"time_s": history["requested_output_times_s"][1],
        "accepted_sample_prefix": 1, "newton": None, "error": "synthetic projection failure"}
    summary = _validate_history(history)
    assert summary["last_accepted_time_s"] == summary["requested_end_s"]
    assert summary["requested_end_reached"] is False
    assert summary["completed"] is False


@pytest.mark.parametrize("reader", [read_bdf_result, read_transient_result])
@pytest.mark.parametrize("name", ["restart-rejected", "restart-at-end", "restart-at-start"])
def test_native_rejected_restart_keeps_failed_progress_without_endpoint(published, reader, name):
    root, captures = published
    result = captures[name].result
    assert not result.completed and not result.restarts[-1].accepted
    with RunStore(root / "store", read_only=True, immutable=True) as store:
        loaded = reader(store, name)
    assert _digest(loaded["history"]) == _digest(result)
    assert loaded["run_state"] == "failed"
    expected = "evaluation_error" if name == "restart-at-start" else "restart_physical_gate_rejected"
    assert loaded["summary"]["termination"] == expected
    assert loaded["summary"]["last_accepted_time_s"] == result.restarts[-1].time_s
    assert (loaded["summary"]["last_accepted_time_s"] == loaded["summary"]["requested_end_s"]) is (name == "restart-at-end")
    assert loaded["summary"]["completed"] is False
    assert loaded["summary"]["requested_end_reached"] is False
    assert loaded["requested_end_state"] is None
    assert not loaded["history"]["samples"]


def test_rejected_restart_refuses_before_any_publication(published, tmp_path):
    _, captures = published
    capture = captures["completed"]
    body = _decode(json.loads(capture.history_bytes))
    body["history"]["restarts"][0]["accepted"] = False
    raw = _canonical(_encode(body))
    metadata = json.loads(capture.capture_json)
    metadata["history_sha256"] = hashlib.sha256(raw).hexdigest()
    changed = replace(capture, history_bytes=raw, capture_json=_canonical(metadata).decode())
    with RunStore(tmp_path / "store") as store:
        row = store.enqueue("bad", opaque_identity=changed.identity, input_metadata={"synthetic_fixture": True})
        token = store.claim("bad", expected_attempt_id=row["current_attempt_id"], expected_generation=0, owner_id="test")
        with pytest.raises(ValueError, match="restart"):
            publish_transient_result(store, token, changed.result, nodes_m=changed.nodes_m, capture=changed)
        assert store.events(run_id="bad")[-1]["kind"] == "running"
        assert not list((store.root / "artifacts").rglob("*.json"))


def reseal_test_record(published, target, mutation):
    """Create a new adversarial/synthetic store; never rewrite a historical run."""
    root, _ = published
    failed = mutation in {"false-completion", "nonfinite-failed-candidate", "interrupted-newton"}
    with RunStore(root / "store", read_only=True, immutable=True) as source:
        original = read_bdf_result(source, "newton" if failed else "completed")
    body = original["record"]
    capture = original["manifest"]["provenance"]
    description = capture["description"]
    if mutation in {"requested-method", "effective-method", "history-method"}:
        if mutation == "requested-method":
            body["request"]["method"] = "backward_euler"
        elif mutation == "effective-method":
            body["effective_method"] = "backward_euler"
        else:
            body["history"]["controls"]["method"] = "backward_euler"
    if mutation == "missing-end":
        body["history"]["accepted_times_s"] = body["history"]["accepted_times_s"][:-1]
        body["history"]["accepted_coordinates"] = body["history"]["accepted_coordinates"][:-1]
    if mutation == "rejected-restart":
        restart = body["history"]["restarts"][0]
        assert restart["accepted"] is True
        assert restart["left_u"].tobytes() == restart["right_u"].tobytes()
        # Performance CND3-R1: change this one Boolean and reseal, retaining
        # every numerical word, admission identity and completion claim.
        restart["accepted"] = False
    if mutation == "rejected-restart-receipt":
        restart = body["history"]["restarts"][0]
        restart["accepted"] = False
        restart["assessment"]["checks"][0]["passed"] = False
    if mutation in {"restart-left-storage", "restart-right-storage", "restart-sample-storage"}:
        history = body["history"]
        if mutation == "restart-sample-storage":
            record = next(s for s in history["samples"] if s["time_s"] == history["restarts"][0]["time_s"])
            key = "storage_value"
        else:
            record = history["restarts"][0]
            key = "left_storage" if mutation == "restart-left-storage" else "right_storage"
        record[key] = record[key].copy()
        record[key][0] = 1.
    if mutation in {"missing-newton-work", "extra-newton-work", "evaluation-work", "assessment-work", "storage-work"}:
        history = body["history"]
        if mutation in {"missing-newton-work", "extra-newton-work"}:
            history["nonlinear_calls"] += -1 if mutation == "missing-newton-work" else 1
        else:
            key = {"evaluation-work": "residual", "assessment-work": "assessment", "storage-work": "storage"}[mutation]
            history["evaluation_counts"][key] = 0
        body["summary"]["work"] = _work(history)
    if mutation == "missing-completed-sample":
        body["history"]["samples"] = body["history"]["samples"][:-1]
        body["summary"]["sample_count"] -= 1
        body["summary"]["work"] = _work(body["history"])
    if mutation == "missing-restart":
        body["history"]["restarts"] = ()
        body["summary"]["work"] = _work(body["history"])
    if mutation == "missing-sample-failure":
        body["history"]["completed"] = body["summary"]["completed"] = False
        body["summary"]["requested_end_reached"] = False
        body["history"]["termination"] = body["summary"]["termination"] = "sample_projection_failed"
    if mutation == "false-completion":
        body["summary"]["completed"] = True
        body["summary"]["requested_end_reached"] = True
    if mutation in {"admitted-input", "attempt-null-manifest", "foreign-attempt"}:
        record = provenance.read_provenance(description["job_provenance"]).export()
        record["input_sha256"] = "0" * 64
        if mutation != "admitted-input":
            record["attempt_ref"] = {"store_id": "a" * 32, "run_id": "foreign",
                "attempt_id": "b" * 32, "owner_id": "foreign", "generation": 0}
            record["execution_manifest_sha256"] = None if mutation == "attempt-null-manifest" else "c" * 64
        description["job_provenance"] = provenance.JobProvenance.from_record(record).envelope()
        capture["identity"]["H_execution"] = hashlib.sha256(_canonical(description)).hexdigest()
    if mutation == "producer-tamper":
        description["job_provenance"]["record"]["source_tree_sha256"] = "0" * 64
    if mutation == "nonfinite-failed-candidate":
        trial = body["history"]["attempts"][-1]
        assert not trial["accepted"]
        trial["candidate_coordinates"] = trial["candidate_coordinates"].copy()
        trial["candidate_coordinates"][0] = np.frombuffer(bytes.fromhex("420000000000f87f"), dtype="<f8")[0]
        trial["newton"]["scaled_residual"] = float("-inf")
    if mutation in {"sample-failure", "interrupted-sample"}:
        history = body["history"]
        history["completed"] = False
        history["termination"] = "sample_projection_failed"
        history["samples"] = history["samples"][:1]
        history["sample_failure"] = {"time_s": history["requested_output_times_s"][1],
            "accepted_sample_prefix": 1, "newton": None, "error": "synthetic post-integration sample failure"}
        if mutation == "interrupted-sample":
            history["nonlinear_calls"] += 1
            for key in ("sample_projection", "sample_residual", "residual"):
                history["evaluation_counts"][key] += 1
        body["summary"] = _validate_history(history)
    if mutation == "interrupted-newton":
        history = body["history"]
        history["termination"] = history["attempts"][-1]["termination"] = "evaluation_error"
        history["attempts"][-1]["newton"] = None
        history["attempts"][-1]["error"] = history["error"] = "synthetic exception before Newton returned"
        body["summary"] = _validate_history(history)
    if mutation.startswith("equation-") or mutation == "pre-wiring-binding":
        implementation = body["request"]["equation_implementation"]
        record = provenance.read_provenance(description["job_provenance"]).export()
        if mutation == "equation-analysis":
            record["assembler"]["analysis_identity"] = "a" * 64
        elif mutation == "equation-assembler":
            record["assembler"].update(identity="a" * 64, unified_assembler=True)
        elif mutation == "equation-operator":
            record["assembler"]["operator"] = "BDFStep:Ionic1DProblem"
        else:
            implementation.pop("provenance_binding")
            if mutation == "pre-wiring-binding":
                record["assembler"]["analysis_identity"] = provenance.digest({"operator": record["assembler"]["operator"]})
        description["equation_implementation"] = implementation
        inputs = _input_record(body["request"])
        description["input_record"] = inputs
        record["input_sha256"] = provenance.digest(inputs)
        description["job_provenance"] = provenance.JobProvenance.from_record(record).envelope()
        capture["identity"].update(H_input=hashlib.sha256(_canonical(inputs)).hexdigest(),
                                  H_execution=hashlib.sha256(_canonical(description)).hexdigest())
    encoded = _encode(body)
    if mutation == "array-hash":
        encoded["history"]["initial_u"]["$array"]["sha256"] = "0" * 64
    raw = _canonical(encoded)
    capture["history_sha256"] = hashlib.sha256(raw).hexdigest()
    with RunStore(target) as store:
        row = store.enqueue("fixture", opaque_identity=capture["identity"], input_metadata={"synthetic_fixture": mutation})
        token = store.claim("fixture", expected_attempt_id=row["current_attempt_id"], expected_generation=0, owner_id="test")
        owner = asdict(token)
        artifact = store.publish(token, "bdf_history", bytes_artifact(raw, metadata={"schema": body["schema"], "role": "bdf_history"}))
        # publication_binding validates the envelope, so the deliberately damaged
        # envelope is attached after sealing with the original admission.
        envelope = description["job_provenance"]
        if mutation == "producer-tamper":
            _, captures = published
            envelope = json.loads(captures["completed"].capture_json)["description"]["job_provenance"]
        binding = provenance.publication_binding(envelope, owner=owner,
            request_identity=capture["identity"], result_sha256=artifact["sha256"])
        if mutation == "binding":
            binding["record"]["owner"]["attempt_id"] = "foreign"
        manifest = {"schema": body["schema"], "summary": body["summary"], "owner": owner,
            "identity": capture["identity"], "history_artifact_id": artifact["artifact_id"],
            "history_sha256": artifact["sha256"], "provenance": capture, "publication_binding": binding}
        manifest_raw = _canonical(manifest)
        stored = store.publish(token, "bdf_manifest", bytes_artifact(manifest_raw,
            metadata={"schema": body["schema"], "role": "bdf_manifest"}))
        document = {"schema": body["schema"], "summary": body["summary"], "owner": owner,
            "manifest_artifact_id": stored["artifact_id"], "manifest_sha256": stored["sha256"],
            "artifact_ids": [artifact["artifact_id"], stored["artifact_id"]]}
        if mutation == "manifest-hash":
            document["manifest_sha256"] = "0" * 64
        state = "succeeded" if body["summary"]["completed"] and mutation != "terminal-state" else "failed"
        store.complete(token, state=state, result=document, artifacts=document["artifact_ids"])
        if mutation == "terminal-reference":
            terminal = store.attempts("fixture")[-1]["terminal"]
            terminal["artifacts"][0]["manifest_sha256"] = "0" * 64
            store._db.execute("UPDATE attempts SET terminal_json=? WHERE attempt_id=?", (_canonical(terminal).decode(), token.attempt_id))


@pytest.mark.parametrize("mutation", ["requested-method", "effective-method", "history-method", "missing-end",
    "false-completion", "array-hash", "admitted-input", "attempt-null-manifest", "foreign-attempt", "producer-tamper",
    "binding", "manifest-hash", "terminal-reference", "terminal-state"])
def test_resealed_producer_or_content_tampering_refuses(published, tmp_path, mutation):
    root = tmp_path / "store"
    reseal_test_record(published, root, mutation)
    with RunStore(root, read_only=True, immutable=True) as store, pytest.raises(ValueError):
        read_bdf_result(store, "fixture")


@pytest.mark.parametrize("reader", [read_bdf_result, read_transient_result])
@pytest.mark.parametrize("mutation", ["equation-analysis", "equation-assembler", "equation-operator",
                                     "equation-binding-missing"])
def test_resealed_equation_binding_tampering_refuses_both_readers(published, tmp_path, reader, mutation):
    root = tmp_path / "store"
    reseal_test_record(published, root, mutation)
    with RunStore(root, read_only=True, immutable=True) as store, pytest.raises(ValueError, match="equation binding"):
        reader(store, "fixture")


@pytest.mark.parametrize("reader", [read_bdf_result, read_transient_result])
def test_pre_wiring_operator_only_publication_still_reads(published, tmp_path, reader):
    root = tmp_path / "store"
    reseal_test_record(published, root, "pre-wiring-binding")
    with RunStore(root, read_only=True, immutable=True) as store:
        loaded = reader(store, "fixture")
    assert "provenance_binding" not in loaded["record"]["request"]["equation_implementation"]
    assert loaded["summary"]["completed"] is True
    assert loaded["requested_end_state"] is not None
    assert _digest(loaded["history"]) == _digest(published[1]["completed"].result)


@pytest.mark.parametrize("reader", [read_bdf_result, read_transient_result])
def test_reviewer_rejected_restart_witness_refuses_both_readers(published, tmp_path, reader):
    root = tmp_path / "store"
    reseal_test_record(published, root, "rejected-restart")
    with RunStore(root, read_only=True, immutable=True) as store, pytest.raises(ValueError, match="restart"):
        reader(store, "fixture")


@pytest.mark.parametrize("mutation", ["rejected-restart-receipt", "restart-left-storage", "restart-right-storage",
    "restart-sample-storage", "missing-newton-work", "extra-newton-work", "evaluation-work", "assessment-work",
    "storage-work", "missing-completed-sample", "missing-restart", "missing-sample-failure"])
@pytest.mark.parametrize("reader", [read_bdf_result, read_transient_result])
def test_resealed_history_relationships_refuse_even_with_matching_summaries(published, tmp_path, mutation, reader):
    root = tmp_path / "store"
    reseal_test_record(published, root, mutation)
    with RunStore(root, read_only=True, immutable=True) as store, pytest.raises(ValueError):
        reader(store, "fixture")


@pytest.mark.parametrize("version", ["readinessv1", "readinessv2"])
@pytest.mark.platform_bound
def test_reviewer_rejected_restart_witness_refuses_historical_packets(tmp_path, version):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    row = next(row for row in manifest["fixtures"] if version in row["path"] and row["completed"])
    path = ROOT / row["path"]
    original = path.read_bytes()
    packet = json.loads(gzip.decompress(original))
    restart = packet["result"]["restarts"]["$tuple"][0]
    assert restart["accepted"] is True
    decoded = _decode(restart)
    assert decoded["left_u"].tobytes() == decoded["right_u"].tobytes()
    restart["accepted"] = False
    raw = _canonical(packet)
    target = tmp_path / "rejected_restart.json"
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="restart"):
        read_bdf_packet(target, hashlib.sha256(raw).hexdigest())
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["nonfinite-failed-candidate", "sample-failure", "interrupted-newton", "interrupted-sample"])
def test_synthetic_failure_data_is_readable_without_promoting_an_endpoint(published, tmp_path, kind):
    root = tmp_path / "store"
    reseal_test_record(published, root, kind)
    with RunStore(root, read_only=True, immutable=True) as store:
        loaded = read_bdf_result(store, "fixture")
    assert loaded["run_state"] == "failed"
    assert loaded["summary"]["requested_end_reached"] is False
    assert loaded["requested_end_state"] is None
    if kind == "nonfinite-failed-candidate":
        trial = loaded["history"]["attempts"][-1]
        assert trial["candidate_coordinates"][:1].tobytes().hex() == "420000000000f87f"
        assert trial["newton"]["scaled_residual"] == float("-inf")
        assert not trial["accepted"]
        assert np.all(np.isfinite(loaded["last_accepted_state"]))
    elif kind == "interrupted-newton":
        assert loaded["history"]["attempts"][-1]["newton"] is None
        assert loaded["summary"]["work"]["nonlinear_calls"] > loaded["summary"]["work"]["main_newton_records"]
    else:
        assert loaded["summary"]["last_accepted_time_s"] == loaded["summary"]["requested_end_s"]
        assert loaded["history"]["sample_failure"]["accepted_sample_prefix"] == 1


@pytest.mark.platform_bound
def test_saved_readiness_packets_keep_original_words_and_identity():
    from solarlab.experiments.transient_results import _decode as historical_decode

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for row in manifest["fixtures"]:
        path = ROOT / row["path"]
        loaded = read_bdf_packet(path, row["sha256"])
        original = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(original).hexdigest() == row["original_sha256"]
        assert loaded["original_packet_sha256"] == row["original_sha256"]
        assert _digest(loaded["packet"]) == _digest(historical_decode(json.loads(original)))
        assert loaded["summary"]["completed"] is row["completed"]
        assert loaded["summary"]["termination"] == row["termination"]
        assert loaded["runtime_identity"] is None
        if not row["completed"]:
            assert loaded["requested_end_state"] is None
            assert loaded["summary"]["last_accepted_time_s"] < loaded["summary"]["requested_end_s"]
            if row["termination"] == "evaluation_error":
                assert loaded["summary"]["work"]["auxiliary_newton_failures"] > 0
        if row.get("accepted_restart_side") == "right":
            history = loaded["history"]
            restart = history["restarts"][0]
            index = np.flatnonzero(history["accepted_times_s"] == restart["time_s"])[0]
            assert _digest(history["accepted_coordinates"][index]) == _digest(restart["right_u"])
            assert _digest(restart["left_u"]) != _digest(restart["right_u"])
            trial = next(step for step in history["attempts"] if step["accepted"] and step["end_s"] == restart["time_s"])
            assert _digest(trial["candidate_coordinates"]) == _digest(restart["left_u"])
        with pytest.raises(ValueError, match="historical source/runtime"):
            read_bdf_packet(path, row["sha256"], strict_same_source=True)


@pytest.mark.parametrize("mutation", ["method", "completed", "clock", "restart-left", "restart-right"])
@pytest.mark.platform_bound
def test_historical_packet_structural_tampering_refuses(tmp_path, mutation):
    fixtures = json.loads((FIXTURES / "manifest.json").read_text())["fixtures"]
    row = fixtures[-1] if mutation.startswith("restart-") else fixtures[1]
    packet = json.loads(gzip.decompress((ROOT / row["path"]).read_bytes()))
    if mutation == "method":
        packet["method"] = "backward_euler"
    elif mutation == "completed":
        packet["completed"] = True
    elif mutation == "clock":
        packet["result"]["final_time_s"] = 0.0
    else:
        packet = _decode(packet)
        restart = packet["result"]["restarts"][0]
        key = "left_u" if mutation == "restart-left" else "right_u"
        restart[key] = restart[key].copy()
        restart[key][0] += .001
        packet = _encode(packet)
    raw = _canonical(packet)
    path = tmp_path / "tampered.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_bdf_packet(path, hashlib.sha256(raw).hexdigest())


@pytest.mark.platform_bound
def test_solver_free_process_reads_historical_packets():
    code = '''
import hashlib, importlib.abc, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoNumericalExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.device.compile_1d', 'solarlab.device.compile_ionic_1d',
            'solarlab.experiments.transient', 'solarlab.experiments.ionic_transient')):
            raise AssertionError('packet reader imported numerical execution: ' + fullname)
sys.meta_path.insert(0, NoNumericalExecution())
from solarlab.experiments.bdf_results import read_bdf_packet
assert hashlib.sha256(pathlib.Path(read_bdf_packet.__code__.co_filename).read_bytes()).hexdigest() == sys.argv[3]
root = pathlib.Path(sys.argv[2])
manifest = json.loads((root / 'tests/integration/fixtures/bdf_readiness/manifest.json').read_bytes())
for row in manifest['fixtures']:
    result = read_bdf_packet(root / row['path'], row['sha256'])
    assert result['summary']['completed'] is row['completed']
assert not any(name == 'solarlab.solvers' or name.startswith('solarlab.solvers.') for name in sys.modules)
'''
    reader = Path(read_bdf_packet.__code__.co_filename).resolve()
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(reader.parents[2]), str(ROOT),
                             hashlib.sha256(reader.read_bytes()).hexdigest()], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr

"""Real isolated workers, original short driver calls, and refusal boundaries."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from solarlab.config.identity import ExecutionEngine, execution_manifest
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.worker import ManifestBinding, ResourceLimits, WorkerRequest
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.legacy_engine import legacy_worker_binding, prepare_legacy_job
from legacy_fixtures import ROOT, make_job


def pool(store, path):
    return ProcessSupervisor(store, path, (legacy_worker_binding(),), max_active=1, max_pending=1,
                             shared_rss_bytes=4 * 1024**3, shared_output_bytes=128 * 1024**2)


def limits():
    return ResourceLimits(90., 1024**3, 64 * 1024**2, None, None)


def assert_closed(receipt):
    value = receipt.data
    assert value["process_exit_observed"] and value["process_joined"] and value["terminal_committed"]
    with pytest.raises(ChildProcessError):
        os.waitpid(value["pid"], os.WNOHANG)
    return value


@pytest.mark.parametrize("driver", ("scaps_transient", "ion_hysteresis", "scaps_steady_state"))
def test_real_worker_matches_original_driver_numeric_payload(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("SOLARLAB_DOS_BAND", "1")  # child explicitly clears this inherited value
    before = dict(os.environ)
    job, invocation = make_job(driver)
    # Existing W08 observer calls the original driver directly, independently
    # of the facade, capturing exact scalar/array words and the full return.
    source = tmp_path / "original.yaml"
    source.write_bytes(invocation.input_source.content)
    request = tmp_path / "direct-request.json"
    raw = invocation.export()
    request.write_text(json.dumps({"driver": driver, "input_path": str(source),
        "input_sha256": invocation.input_source.sha256, "environment": raw["environment"],
        "arguments": raw["arguments"], "argument_form": "driver_keywords"}))
    direct = tmp_path / "direct.json"
    environment = dict(os.environ)
    for key, value in job.execution.environment:
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tests/refactor/legacy_driver_observer.py"),
                             "--request", str(request), "--output", str(direct)],
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    original = json.loads(direct.read_bytes())
    assert original["complete"] and original["capture_errors"] == []
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        receipt = assert_closed(supervisor.wait(job.submit(supervisor, driver, limits()), 95))
        assert receipt["store_state"] == "succeeded", receipt
        records = [store.artifact(key) for key in receipt["terminal_artifacts"]]
        payload = next(item for item in records if item["name"] == "legacy_result")
        actual = store.read_artifact(payload["artifact_id"])
        assert json.loads(actual) == original["result"]
        manifest = next(item for item in records if item["name"] == "execution_manifest")
        assert store.read_artifact(manifest["artifact_id"]) == job.execution.manifest.canonical_json
        assert receipt["worker_result"]["manifest_sha256"] == job.execution.expected_sha256
        assert receipt["worker_result"]["scientific_qualification"] is False
        assert receipt["resources"]["calls"] is None  # no invented internal step counters
        events = [row for row in store.events(run_id=driver) if row["kind"] == "terminal"]
        assert len(events) == 1
    assert dict(os.environ) == before
    evidence = os.environ.get("SOLARLAB_LEGACY_WORKER_EVIDENCE")
    if evidence:
        with Path(evidence).open("a") as stream:
            stream.write(json.dumps({"driver": driver, "bitwise_equal": True,
                "direct_pid": original["pid"], "direct_seconds": original["elapsed_seconds"],
                "payload_sha256": hashlib.sha256(actual).hexdigest(), "worker": receipt}, allow_nan=False) + "\n")


@pytest.mark.parametrize("change", ("missing_manifest", "wrong_runtime", "wrong_engine", "original_driver_error", "installed_source", "installed_resource"))
def test_real_worker_refuses_or_fails_without_engine_fallback(tmp_path, change):
    job, invocation = make_job("ion_hysteresis" if change == "wrong_engine" else "scaps_transient",
                               arguments={"n_points": 1} if change == "original_driver_error" else None,
                               changed_resource=change == "installed_resource",
                               runtime_change={"python_version": "0.0"} if change == "wrong_runtime" else None)
    execution = None if change == "missing_manifest" else job.execution
    request = job.request
    if change == "installed_source":
        binding = job.execution
        files = binding.execution.implementation.files
        changed = replace(binding.execution, implementation=replace(binding.execution.implementation,
            files=(replace(files[0], content=files[0].content + b"\n# other source\n"), *files[1:])))
        manifest = execution_manifest(binding.inputs, changed, engine=binding.engine)
        altered = prepare_legacy_job(invocation, binding.inputs, changed, engine=binding.engine,
                                     manifest=manifest, expected_sha256=manifest.sha256)
        execution, request = altered.execution, altered.request
    if change == "wrong_engine":
        engine = ExecutionEngine("solarlab")
        inputs = replace(job.execution.inputs, effective_behavior=None)
        manifest = execution_manifest(inputs, job.execution.execution, engine=engine)
        identity = manifest.export()["document"]["execution_identity"]
        request = WorkerRequest(request.request, request.registry, request.schema,
            SourceDocument("other-engine", json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()),
            identity["payload"]["H_input"], identity["payload"]["H_physics"])
        execution = ManifestBinding(manifest, inputs, job.execution.execution, engine,
                                    manifest.sha256, job.execution.environment)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        value = assert_closed(supervisor.wait(supervisor.submit(change, "legacy", request, limits(), execution=execution), 95))
        assert value["store_state"] == "failed", value
        assert value["worker_result"]["reason"] == "worker_exception", value
        expected = {"missing_manifest": "requires a complete execution manifest", "wrong_runtime": "runtime environment",
                    "wrong_engine": "wrong execution engine", "original_driver_error": "n_points must be >= 2",
                    "installed_source": "installed legacy implementation content differs",
                    "installed_resource": "installed legacy resource content differs"}
        assert expected[change] in value["worker_result"]["detail"], value
        assert not value["terminal_artifacts"]
        terminal = [event for event in store.events(run_id=change) if event["kind"] == "terminal"]
        assert len(terminal) == 1 and terminal[0]["payload"]["state"] == "failed"


def test_preparation_refuses_wrong_engine_changed_roles_and_manifest_pin():
    job, invocation = make_job()
    binding = job.execution
    for changes, message in (({"engine": ExecutionEngine("solarlab")}, "explicit legacy"),
                             ({"expected_sha256": "0" * 64}, "frozen")):
        args = dict(engine=binding.engine, manifest=binding.manifest, expected_sha256=binding.expected_sha256)
        args.update(changes)
        with pytest.raises(ValueError, match=message):
            prepare_legacy_job(invocation, binding.inputs, binding.execution, **args)
    changed = replace(binding.execution, grid=SourceDocument("wrong-grid", b"{}"))
    with pytest.raises(ValueError, match="roles differ"):
        prepare_legacy_job(invocation, binding.inputs, changed, engine=binding.engine,
                           manifest=binding.manifest, expected_sha256=binding.expected_sha256)


def test_isolated_import_does_not_load_an_engine_or_open_a_store(tmp_path):
    code = "import sys; import solarlab_server.legacy_engine; assert not set(sys.modules) & {'perovskite_sim', 'scipy', 'sksundae', 'flint'}"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path,
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []

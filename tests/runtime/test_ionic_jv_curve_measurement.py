"""Source-bound diagnostic admission and HTTP refusal owned by backend CI."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from scripts.ci import ionic_jv_timing as timing


@pytest.fixture(scope="module")
def installed():
    from examples.research_1d.ionic_jv_service import example_submission, installed_catalog
    from examples.research_1d.process_ionic_transient import source_build
    from solarlab_server.job_service import JobSubmission

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=timing.ROOT, text=True).strip()
    catalog = installed_catalog(source_build(commit), commit)
    submission = example_submission()
    spec, execution = catalog.prepare(JobSubmission.model_validate(submission))
    return catalog, submission, spec, execution


@pytest.fixture(scope="module")
def diagnostic(installed):
    return timing.prepare_curve_measurement(installed[2].request, installed[3])


def test_only_curve_and_diagnostic_identity_change(installed, diagnostic):
    from solarlab.experiments.jv import ionic_worker as core

    _, submission, spec, execution = installed
    expected = json.loads(spec.request.request.content)
    expected["schema"] = timing._diagnostic_documents()[0].id
    expected["controls"]["wall_s"] = 64.
    assert json.loads(diagnostic.request.request.content) == expected
    assert submission["input"]["controls"]["wall_s"] == 25.
    assert expected["controls"]["steady"]["wall_s"] == 30.
    assert expected["initial"]["wall_s"] == 25.
    assert diagnostic.request.H_input == spec.request.H_input
    assert diagnostic.request.H_physics == spec.request.H_physics
    assert diagnostic.execution.expected_sha256 != execution.expected_sha256
    diagnostic.execution.validate_request(diagnostic.request)
    preparation = json.loads(diagnostic.execution.execution.preparation.content)
    source = preparation["diagnostic_measurement"]["source"]
    raw = (timing.ROOT / source["id"]).read_bytes()
    assert bytes.fromhex(source["content_hex"]) == raw
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert preparation["diagnostic_measurement"]["recommendation"] is False
    assert core.run_ionic_jv_worker.__globals__["_validate"] is core._validate
    assert core.prepare_ionic_jv_job.__globals__["_roles"] is core._roles
    assert core.run_ionic_jv_worker.__globals__["REQUEST_SCHEMA"] is core.REQUEST_SCHEMA
    assert core.REQUEST_SCHEMA != diagnostic.request.schema
    assert core.REGISTRY != diagnostic.request.registry


def test_http_and_registered_admission_refuse_the_measurement_recipe(tmp_path, monkeypatch, installed, diagnostic):
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.jv.ionic_worker import ionic_jv_worker_binding
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.ionic_jv_preparation import execute_registered_ionic_jv
    from solarlab_server.job_service import JobService, JobSubmission

    catalog, submission, _, _ = installed
    changed = deepcopy(submission)
    changed["input"]["controls"]["wall_s"] = 64.
    with pytest.raises(ValueError, match="unchanged full controls"):
        catalog.prepare(JobSubmission.model_validate(changed))
    for binding in (catalog.worker_binding("solarlab"), ionic_jv_worker_binding()):
        with pytest.raises(ValueError, match="registry/schema"):
            binding.validate_request(diagnostic.request)
    with pytest.raises(ValueError, match="installed route"):
        execute_registered_ionic_jv(diagnostic.request, SimpleNamespace(check_cancel=lambda: None))
    with RunStore(tmp_path / "store") as store, ProcessSupervisor(
        store, tmp_path / "work", catalog.worker_bindings(), max_active=1, max_pending=1,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
    ) as owner:
        monkeypatch.setattr(owner, "submit", lambda *args, **kwargs: pytest.fail("refusal must precede enqueue"))
        with TestClient(create_app(store.root, job_service=JobService(catalog, owner))) as client:
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            response = client.post("/jobs", json=changed, headers=headers)
            assert response.status_code == 422 and "unchanged full controls" in response.text
            assert client.get("/experiments").json()["bindings"][0]["preparation_context"]["fixed_numerical_policy"][
                "controls"]["wall_s"] == 25.
        assert store.events() == []
    assert catalog.prepare(JobSubmission.model_validate(submission))[0].limits.wall_seconds == 128.


@pytest.mark.parametrize("change", ["curve", "point", "initial", "iterations", "tolerance", "grid", "voltages", "device"])
def test_diagnostic_admits_no_other_recipe_change(installed, change):
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab.experiments.jv.ionic_worker import IonicJVProtocol
    from solarlab_server.ionic_jv_preparation import _initial, _options

    prepared = installed[3].inputs.prepared
    body = json.loads(installed[2].request.request.content)
    protocol = IonicJVProtocol(**body["protocol"])
    intervals, initial, options = (12, 12, 12), _initial(), replace(_options(), wall_s=64.)
    if change == "curve":
        options = replace(options, wall_s=65.)
    elif change == "point":
        options = replace(options, steady=replace(options.steady, wall_s=31.))
    elif change == "initial":
        initial = replace(initial, wall_s=26.)
    elif change == "iterations":
        options = replace(options, steady=replace(options.steady, budget=replace(options.steady.budget, max_iterations=61)))
    elif change == "tolerance":
        options = replace(options, steady=replace(options.steady, inventory_relative_tolerance=1e-3))
    elif change == "grid":
        intervals = (13, 12, 12)
    elif change == "voltages":
        protocol = replace(protocol, voltage_V=(0., .026, *protocol.voltage_V[2:]))
    else:
        prepared = prepare_confined_shared_site_ion_device(positive_diffusivity_m2_s=2e-16)
    with pytest.raises(ValueError):
        timing._diagnostic_validate(prepared, protocol, intervals, "absorber", ("blocking", "blocking"), options, initial)


def test_core_admission_still_rejects_64_after_diagnostic_preparation(installed, diagnostic):
    from solarlab.experiments.jv import ionic_worker as core
    from solarlab_server.ionic_jv_preparation import _initial, _options

    body = json.loads(diagnostic.request.request.content)
    with pytest.raises(ValueError, match="curve wall ceiling >25s"):
        core._validate(installed[3].inputs.prepared, core.IonicJVProtocol(**body["protocol"]),
            (12, 12, 12), "absorber", ("blocking", "blocking"), replace(_options(), wall_s=64.), _initial())


def test_changed_diagnostic_source_refuses_before_worker_execution(monkeypatch, diagnostic):
    from solarlab.materials.source import SourceDocument

    source = timing._diagnostic_source()
    monkeypatch.setattr(timing, "_diagnostic_source", lambda: SourceDocument(source.id, source.content + b"\n# changed\n"))
    with pytest.raises(ValueError, match="registry/schema"):
        timing.run_curve_measurement(diagnostic.request, None)


def test_production_sources_do_not_import_diagnostic_override():
    for path in (timing.ROOT / "src").rglob("*.py"):
        assert "ionic_jv_timing" not in path.read_text(), path
    code = ("import sys; from examples.research_1d.ionic_jv_service import installed_catalog; "
            "import solarlab_server.app; assert 'scripts.ci.ionic_jv_timing' not in sys.modules")
    subprocess.run([timing.sys.executable, "-c", code], cwd=timing.ROOT, check=True)


def history():
    return {"controls": {"steady": {"wall_s": 30.}}, "dark_steady": {
        "wall_s": .1, "accepted": True, "termination": "converged", "error": None},
        "illumination": {"wall_s": .2, "accepted": True, "termination": "converged", "error": None},
        "curve": {"wall_s": 64.001, "attempts": {"$tuple": [{
            "index": 50, "voltage_V": 1.25, "accepted": False, "termination": "wall_budget_exhausted",
            "solver_called": True, "wall_s": .501, "error": "_WaveWallLimit",
            "solution": {"wall_s": .5005, "termination": "wall_budget_exhausted", "error": "_WallLimit",
                         "solver_controls": {"wall_s": .5}}}]}}}


@pytest.mark.parametrize("curve_measurement", [False, True])
def test_failed_child_receipt_is_read_before_success_checks(tmp_path, monkeypatch, installed, curve_measurement):
    import multiprocessing

    class Child:
        exitcode = 0

        def __init__(self, args):
            self.directory = Path(args[3])

        def start(self):
            raw = json.dumps(history()).encode()
            (self.directory / "diagnostics.bin").write_bytes(raw)
            terminal = {"state": "failed", "child_peak_rss_bytes": 100, "artifacts": [
                {"name": "ionic_jv_diagnostics", "file": "diagnostics.bin",
                 "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}],
                "result": {"reason": "wall_budget_exhausted", "ionic_jv_result": {
                    "outcome": "partial", "accepted_count": 50, "requested_count": 51}}}
            (self.directory / "terminal.json").write_text(json.dumps(terminal))

        def join(self):
            pass

        def is_alive(self):
            return False

    context = SimpleNamespace(Event=lambda: None, Lock=lambda: None, RawArray=lambda *args: None,
                              Process=lambda **kwargs: Child(kwargs["args"]))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    result = timing.measure(tmp_path, curve_measurement=curve_measurement)
    assert result["status"] == "failed" and result["reason"] == "wall_budget_exhausted"
    assert result["artifact_integrity_verified"] is True
    assert result["numerical_seconds"] == {"dark_steady": .1, "illumination": .2, "curve": 64.001}
    assert len(result["point_timings"]) == 1 and result["censoring"]["numerical_wall"] is True

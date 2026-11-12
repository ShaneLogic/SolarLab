"""Process and HTTP transient workers carry their parent's time-method request (CND2).

Preparation records the parent's request among the execution manifest's
child settings: ``NEWCORE_TIME_METHOD`` is unset for the default and
``backward_euler`` for the explicit opt-out. It is part of the execution
identity; the input and physics identities are unchanged. The spawned child
installs the setting before any engine import, refuses a request that
differs, and records its selection in the saved worker result. No worker
scope is allowlisted, because a worker's initial state is a cold recipe. A
``bdf`` request is therefore refused while the job is prepared, through the
process and HTTP routes alike. A worker never integrates with BDF, and BDF is
never run as backward Euler. The saved arrays are the same words under the
default and the explicit opt-out.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from solarlab.equations import time_method
from solarlab.equations.time_method import BACKWARD_EULER, BDF, ENVIRONMENT
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.worker import ResourceLimits
from solarlab.io.run_store import RunStore

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()


@pytest.fixture(autouse=True)
def no_process_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENVIRONMENT, raising=False)


def transient_job(commit: str) -> Any:
    from examples.research_1d.process_transient import example_job
    from examples.research_1d.service import committed_implementation

    return example_job(committed_implementation(commit), current_repository_head=commit)


def ionic_transient_job(commit: str) -> Any:
    from examples.research_1d.process_ionic_transient import example_job, source_build

    return example_job(source_build(commit), current_repository_head=commit)


JOBS = {"transient": transient_job, "ionic_transient": ionic_transient_job}


def declared(job: Any) -> str | None:
    """The request a prepared job's manifest binds for its child."""
    settings = dict(job.execution.environment)
    assert dict(job.execution.execution.environment.settings)[ENVIRONMENT] == json.dumps(settings[ENVIRONMENT])
    return settings[ENVIRONMENT]


@pytest.mark.parametrize("family", JOBS)
def test_preparation_records_the_parents_request_and_refuses_bdf(family: str, commit: str,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = JOBS[family]
    default = prepare(commit)
    with time_method.time_method(BACKWARD_EULER):
        scoped = prepare(commit)
    monkeypatch.setenv(ENVIRONMENT, BACKWARD_EULER)
    process = prepare(commit)
    monkeypatch.delenv(ENVIRONMENT)
    assert (declared(default), declared(scoped), declared(process)) == (None, BACKWARD_EULER, BACKWARD_EULER)
    assert default.request.H_input == scoped.request.H_input and default.request.H_physics == scoped.request.H_physics
    assert default.execution.manifest.execution_sha256 != scoped.execution.manifest.execution_sha256
    assert scoped.execution.manifest.execution_sha256 == process.execution.manifest.execution_sha256
    with time_method.time_method(BDF), pytest.raises(ValueError, match="not selectable for process or HTTP worker"):
        prepare(commit)
    monkeypatch.setenv(ENVIRONMENT, BDF)
    with pytest.raises(ValueError, match="not selectable for process or HTTP worker"):
        prepare(commit)
    monkeypatch.setenv(ENVIRONMENT, "Bdf")
    with pytest.raises(ValueError, match="unknown time method"):
        prepare(commit)
    # Flipped, the default still forwards no request and the worker still integrates with backward Euler.
    monkeypatch.delenv(ENVIRONMENT)
    monkeypatch.setattr(time_method, "ALLOWLISTED_DEFAULT", (BDF, "CommanderBdfDefaultFreezeV2"))
    assert declared(prepare(commit)) is None


def test_the_http_transient_route_refuses_a_bdf_request_before_enqueueing(tmp_path: Path, commit: str,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from examples.research_1d.service import committed_implementation
    from examples.research_1d.transient_service import example_submission, installed_catalog
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService

    catalog = installed_catalog(committed_implementation(commit), commit)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog=catalog) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(catalog, supervisor))) as client:
            advertised = client.get("/experiments")
            headers = {name: advertised.headers[name] for name in ("x-solarlab-registry", "x-solarlab-job-schema")}
            monkeypatch.setenv(ENVIRONMENT, BDF)
            refused = client.post("/jobs", json=example_submission("http-bdf"), headers=headers)
            assert refused.status_code == 422, refused.text
            assert "not selectable for process or HTTP worker" in refused.text
            assert store.events() == []


def pool(store: RunStore, work: Path, *, catalog: Any = None) -> ProcessSupervisor:
    from solarlab.experiments.ionic_transient_worker import ionic_transient_worker_binding
    from solarlab.experiments.transient_worker import transient_worker_binding

    bindings = (transient_worker_binding(), ionic_transient_worker_binding()) if catalog is None else catalog.worker_bindings()
    return ProcessSupervisor(store, work, bindings, max_active=1, max_pending=4,
                             shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
                             poll_seconds=0.02, soft_grace_seconds=0.2, terminate_grace_seconds=0.2)


def test_spawned_workers_run_and_record_their_prepared_request(commit: str, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from solarlab.experiments.ionic_transient_worker import read_ionic_transient_job
    from solarlab.experiments.transient_worker import read_transient_job

    jobs = {}
    for family, prepare in JOBS.items():
        jobs[family + "-default"] = prepare(commit)
        with time_method.time_method(BACKWARD_EULER):
            jobs[family + "-opt-out"] = prepare(commit)
    limits = ResourceLimits(30.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    readers = {"transient": read_transient_job, "ionic_transient": read_ionic_transient_job}
    saved: dict[str, dict[str, bytes]] = {}
    # The submitting context requests BDF: the manifest, not the submitter, decides.
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        with time_method.time_method(BDF):
            handles = {name: job.submit(supervisor, name, limits) for name, job in jobs.items()}
        for name, handle in handles.items():
            receipt = supervisor.wait(handle, 90).data
            assert receipt["store_state"] == "succeeded", receipt
            attempt, = store.attempts(name)
            result = attempt["result"]["worker_result"]
            family, _, mode = name.partition("-")
            requested = None if mode == "default" else BACKWARD_EULER
            assert result["time_method"] == {
                "schema": "solarlab.time-method-selection.v1", "family": "electronic_1d" if family == "transient"
                else "ionic_1d", "requested": requested, "source": "default" if requested is None else "environment",
                "effective": BACKWARD_EULER, "allowlisted_default": list(time_method.ALLOWLISTED_DEFAULT)}, name
            loaded = readers[family](store, name)
            assert loaded["worker_result"]["time_method"] == result["time_method"]
            records = [store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]]
            saved[name] = {record["name"]: store.read_artifact(record["artifact_id"])
                           for record in records if record["format"] == "npy"}
    for family in JOBS:
        assert saved[family + "-default"] and saved[family + "-default"] == saved[family + "-opt-out"]

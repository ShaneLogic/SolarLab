"""Process workers run the equation selection of the parent that prepared them (Science PW-C2).

Preparation resolves the selection (the innermost ``equation_implementation``
scope, else ``NEWCORE_EQUATIONS``, else assembled) and records
``NEWCORE_EQUATIONS=<selection>`` among the execution manifest's child
settings, so it is part of the execution identity. The spawned child installs
those settings before any engine import, refuses a selection that differs,
and admits its job under the Q6 binding of that implementation, which the
saved result records. A queued job keeps its manifest's selection. A bare
scope is local to its execution context: a job prepared in a new thread
follows that thread's own selection unless the thread runs a copied context.
The saved array payloads of both selections are byte-identical; full result
and reader parity is pinned in process by tests/unit/assembly, and the
saved documents of the process examples by the examples harness.
"""

from __future__ import annotations

from collections.abc import Callable
import contextvars
import io
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Any

import pytest

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.equations.implementation import ASSEMBLED, ENVIRONMENT, LEGACY, equation_implementation
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.worker import ResourceLimits
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.job_provenance import read_provenance

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()


@pytest.fixture(scope="module")
def steady_source(commit: str) -> BuildSource:
    content = subprocess.check_output(["git", "archive", commit, "--", "src/solarlab"], cwd=ROOT, timeout=10)
    sources = []
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for item in archive.getmembers():
            stream = archive.extractfile(item) if item.isfile() and item.name.endswith(".py") and item.size else None
            if stream is not None:
                sources.append(SourceDocument(item.name.removeprefix("src/"), stream.read()))
    return BuildSource(commit, tuple(sources))


def steady_job(source: BuildSource) -> Any:
    from examples.research_1d.steady import load_device
    from solarlab.experiments.jv.worker import SteadyJVProtocol, prepare_steady_jv_job

    return prepare_steady_jv_job(load_device(), SteadyJVProtocol((0.0, 0.05), 12, 1.0, None), implementation=source,
                                 current_repository_head=source.build_source_commit, engine=ExecutionEngine("solarlab"))


def transient_job(commit: str) -> Any:
    from examples.research_1d.process_transient import example_job
    from examples.research_1d.service import committed_implementation

    # Shared pytest collection also loads ionic modules: bind the whole committed installation.
    return example_job(committed_implementation(commit), current_repository_head=commit)


def ionic_transient_job(commit: str) -> Any:
    from examples.research_1d.process_ionic_transient import example_job, source_build

    return example_job(source_build(commit), current_repository_head=commit)


def ionic_jv_job(commit: str) -> Any:
    from examples.research_1d.ionic_steady import controls
    from examples.research_1d.process_ionic_jv import source_build
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.experiments.jv.ionic_worker import IonicJVInitial, IonicJVProtocol, prepare_ionic_jv_job

    source = source_build(commit)
    return prepare_ionic_jv_job(prepare_confined_shared_site_ion_device(), IonicJVProtocol((0.0, 0.025), 1.0, None),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"),
        options=IonicSteadyJVOptions(controls(13), max_points=51, wall_s=25.0),
        initial=IonicJVInitial("dark_direct_density_seed_v1", 0.002, 25.0), implementation=source,
        current_repository_head=commit, engine=ExecutionEngine("solarlab"))


def bindings() -> tuple[Any, ...]:
    from solarlab.experiments.ionic_transient_worker import ionic_transient_worker_binding
    from solarlab.experiments.jv.ionic_worker import ionic_jv_worker_binding
    from solarlab.experiments.jv.worker import steady_jv_worker_binding
    from solarlab.experiments.transient_worker import transient_worker_binding

    return (steady_jv_worker_binding(), transient_worker_binding(), ionic_transient_worker_binding(),
            ionic_jv_worker_binding())


def pool(store: RunStore, work: Path) -> ProcessSupervisor:
    return ProcessSupervisor(store, work, bindings(), max_active=1, max_pending=3,
                             shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
                             poll_seconds=0.02, soft_grace_seconds=0.2, terminate_grace_seconds=0.2)


def declared(job: Any) -> str:
    """The selection a prepared job's manifest binds for its child."""
    settings = dict(job.execution.environment)
    assert dict(job.execution.execution.environment.settings)[ENVIRONMENT] == '"' + settings[ENVIRONMENT] + '"'
    return settings[ENVIRONMENT]


def run(jobs: dict[str, Any], tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Submit every job to one single-slot pool (so all but the first queue); each saved worker result.

    Each result also carries the bytes of its saved array payloads.
    """
    limits = ResourceLimits(30.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    results = {}
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        handles = {name: job.submit(supervisor, name, limits) for name, job in jobs.items()}
        for name, handle in handles.items():
            receipt = supervisor.wait(handle, 90).data
            assert receipt["store_state"] == "succeeded", receipt
            attempt, = store.attempts(name)
            results[name] = attempt["result"]["worker_result"]
            manifest = next(store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]
                            if store.artifact(ref["artifact_id"])["name"] == "execution_manifest")
            assert manifest["metadata"]["job_provenance_sha256"] == read_provenance(results[name]["job_provenance"]).sha256
            records = [store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]]
            results[name]["saved_words"] = {record["name"]: store.read_artifact(record["artifact_id"])
                                            for record in records if record["format"] == "npy"}
    return results


def binding(result: dict[str, Any]) -> dict[str, Any]:
    return read_provenance(result["job_provenance"]).export()["assembler"]


def test_preparation_records_the_resolved_selection_and_fails_closed(
    steady_source: BuildSource, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarlab.equations.implementation import verify_selection

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    default = steady_job(steady_source)
    with equation_implementation(LEGACY):
        scoped = steady_job(steady_source)
    monkeypatch.setenv(ENVIRONMENT, LEGACY)
    process = steady_job(steady_source)
    monkeypatch.delenv(ENVIRONMENT)
    assert (declared(default), declared(scoped), declared(process)) == (ASSEMBLED, LEGACY, LEGACY)
    # The selection is part of the execution identity, not of the input or physics identity.
    assert default.request.H_input == scoped.request.H_input and default.request.H_physics == scoped.request.H_physics
    assert default.execution.manifest.execution_sha256 != scoped.execution.manifest.execution_sha256
    assert scoped.execution.manifest.execution_sha256 == process.execution.manifest.execution_sha256

    def prepared_in_thread(target: Callable[[], Any]) -> str:
        found: list[Any] = []
        thread = threading.Thread(target=lambda: found.append(target()))
        thread.start()
        thread.join()
        return declared(found[0])

    with equation_implementation(LEGACY):
        assert prepared_in_thread(lambda: steady_job(steady_source)) == ASSEMBLED
        context = contextvars.copy_context()
        assert prepared_in_thread(lambda: context.run(steady_job, steady_source)) == LEGACY
    monkeypatch.setenv(ENVIRONMENT, "Legacy")
    with pytest.raises(ValueError, match="unknown equation implementation"):
        steady_job(steady_source)
    monkeypatch.delenv(ENVIRONMENT)
    verify_selection(default.execution.environment)
    with equation_implementation(LEGACY), pytest.raises(ValueError, match="differs from its execution manifest"):
        verify_selection(default.execution.environment)
    with pytest.raises(ValueError, match="differs from its execution manifest"):
        verify_selection(scoped.execution.environment)


def test_spawned_and_queued_steady_workers_run_their_prepared_selection(
    steady_source: BuildSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from support.equation_provenance import expected

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    with equation_implementation(LEGACY):
        scoped = steady_job(steady_source)
    jobs = {"assembled-default": steady_job(steady_source), "legacy-scope": scoped}
    monkeypatch.setenv(ENVIRONMENT, LEGACY)
    jobs["legacy-process"] = steady_job(steady_source)
    monkeypatch.delenv(ENVIRONMENT)
    with equation_implementation(ASSEMBLED):  # the submitting context does not select; the manifest does
        results = run(jobs, tmp_path)
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    for name, result in results.items():
        implementation = declared(jobs[name])
        assert binding(result) == expected("DC1DProblem.steady_jv", device, implementation), name
    assert results["legacy-scope"]["steady_result"] == results["assembled-default"]["steady_result"]
    assert results["legacy-scope"]["saved_words"] == results["assembled-default"]["saved_words"]
    assert binding(results["legacy-scope"]) == binding(results["legacy-process"])


@pytest.mark.parametrize("family", ("transient", "ionic_transient", "ionic_jv"))
def test_every_1d_worker_family_records_its_parent_selection(family: str, commit: str, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from support.equation_provenance import expected

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    prepare = {"transient": transient_job, "ionic_transient": ionic_transient_job, "ionic_jv": ionic_jv_job}[family]
    jobs = {}
    for implementation in (ASSEMBLED, LEGACY):
        with equation_implementation(implementation):
            jobs[implementation] = prepare(commit)
        assert declared(jobs[implementation]) == implementation
    results = run(jobs, tmp_path)
    device, operator, closure = device_of(family)
    for implementation, result in results.items():
        assert binding(result) == expected(operator, device, implementation, closure), implementation
    assert results[ASSEMBLED]["saved_words"] and results[ASSEMBLED]["saved_words"] == results[LEGACY]["saved_words"]


def device_of(family: str) -> tuple[Any, str, int | None]:
    """The compiled device, original operator and closure node of each example worker job."""
    from examples.research_1d.ionic_steady import controls
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d

    if family == "transient":
        return compile_device_1d(load_device(), intervals_per_layer=12), "BackwardEuler1DStep", None
    device = compile_ionic_device_1d(prepare_confined_shared_site_ion_device(), intervals_per_layer=(12, 12, 12),
                                     ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    operator = "IonicBackwardEuler1DStep" if family == "ionic_transient" else "Ionic1DProblem.steady_jv"
    return device, operator, controls(13).inventory_row_node


def test_the_unswitched_2d_steady_worker_states_the_original_equations(commit: str, tmp_path: Path) -> None:
    """The 2D process example, its 1D column seed included, runs and records the original equations."""
    import json
    import os
    import sys

    from examples.research_1d.steady import load_device
    from examples.research_2d.dc_assembly import spatial_input
    from solarlab.device.compile_2d import compile_device_2d
    from support.equation_provenance import expected

    archive = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), commit, "--", "src/solarlab"],
                   cwd=ROOT, check=True, timeout=30)
    device = compile_device_2d(load_device(), spatial_input())
    readbacks = {}
    for implementation in (ASSEMBLED, LEGACY):
        env = {key: value for key, value in os.environ.items() if key != ENVIRONMENT}
        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")})
        if implementation == LEGACY:
            env[ENVIRONMENT] = LEGACY
        receipt = tmp_path / (implementation + ".json")
        result = subprocess.run([sys.executable, "-B", "-m", "examples.research_2d.process_steady", "run",
            "--store", str(tmp_path / implementation / "Store"), "--work", str(tmp_path / implementation / "Work"),
            "--run-id", "steady2d-" + implementation, "--build-source-commit", commit, "--source-archive",
            str(archive), "--current-repository-head", commit, "--targets", "dark", "--receipt-json", str(receipt)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr + result.stdout[-2000:]
        saved = json.loads(receipt.read_text())
        assert binding(saved["receipt"]["worker_result"]) == expected("DC2DProblem.steady", device, LEGACY)
        readbacks[implementation] = saved["readback"]["array_fingerprints"]
    assert readbacks[ASSEMBLED] and readbacks[ASSEMBLED] == readbacks[LEGACY]

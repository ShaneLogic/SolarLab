"""Production process workers lock equation selection to assembled (P09-02).

Preparation resolves the selection and records ``NEWCORE_EQUATIONS="assembled"``
in the execution manifest's child settings. Environment opt-out via
``NEWCORE_EQUATIONS=legacy`` is refused before preparation, and spawned workers
refuse any non-assembled manifest binding via ``verify_selection``. Explicit
in-process ``with equation_implementation(LEGACY):`` remains available only for
unit assembly parity tests in ``tests/unit/assembly/``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.equations.implementation import (
    ASSEMBLED,
    ENVIRONMENT,
    LEGACY,
    equation_implementation,
    verify_selection,
)
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

    return prepare_steady_jv_job(
        load_device(),
        SteadyJVProtocol((0.0, 0.05), 12, 1.0, None),
        implementation=source,
        current_repository_head=source.build_source_commit,
        engine=ExecutionEngine("solarlab"),
    )


def transient_job(commit: str) -> Any:
    from examples.research_1d.process_transient import example_job
    from examples.research_1d.service import committed_implementation

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
    return prepare_ionic_jv_job(
        prepare_confined_shared_site_ion_device(),
        IonicJVProtocol((0.0, 0.025), 1.0, None),
        intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
        options=IonicSteadyJVOptions(controls(13), max_points=51, wall_s=25.0),
        initial=IonicJVInitial("dark_direct_density_seed_v1", 0.002, 25.0),
        implementation=source,
        current_repository_head=commit,
        engine=ExecutionEngine("solarlab"),
    )


def bindings() -> tuple[Any, ...]:
    from solarlab.experiments.ionic_transient_worker import ionic_transient_worker_binding
    from solarlab.experiments.jv.ionic_worker import ionic_jv_worker_binding
    from solarlab.experiments.jv.worker import steady_jv_worker_binding
    from solarlab.experiments.transient_worker import transient_worker_binding

    return (
        steady_jv_worker_binding(),
        transient_worker_binding(),
        ionic_transient_worker_binding(),
        ionic_jv_worker_binding(),
    )


def pool(store: RunStore, work: Path) -> ProcessSupervisor:
    return ProcessSupervisor(
        store,
        work,
        bindings(),
        max_active=1,
        max_pending=3,
        shared_rss_bytes=3 * 1024**3,
        shared_output_bytes=384 * 1024**2,
        poll_seconds=0.02,
        soft_grace_seconds=0.2,
        terminate_grace_seconds=0.2,
    )


def declared(job: Any) -> str:
    settings = dict(job.execution.environment)
    assert dict(job.execution.execution.environment.settings)[ENVIRONMENT] == '"' + settings[ENVIRONMENT] + '"'
    return settings[ENVIRONMENT]


def binding(result: dict[str, Any]) -> dict[str, Any]:
    return read_provenance(result["job_provenance"]).export()["assembler"]


def test_preparation_and_verify_selection_lock_production_workers_to_assembled(
    steady_source: BuildSource, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    default = steady_job(steady_source)
    assert declared(default) == ASSEMBLED
    verify_selection(default.execution.environment)

    # Environment NEWCORE_EQUATIONS=legacy is retired and refused at worker preparation.
    monkeypatch.setenv(ENVIRONMENT, LEGACY)
    with pytest.raises(ValueError, match="NEWCORE_EQUATIONS=legacy is retired"):
        steady_job(steady_source)
    monkeypatch.setenv(ENVIRONMENT, "Legacy")
    with pytest.raises(ValueError, match="unknown equation implementation"):
        steady_job(steady_source)
    monkeypatch.delenv(ENVIRONMENT)

    # Even if a manifest was constructed inside an explicit unit-test LEGACY scope,
    # verify_selection refuses legacy execution in any worker process.
    with equation_implementation(LEGACY):
        scoped_legacy = steady_job(steady_source)
        with pytest.raises(ValueError, match="legacy worker equation selection is retired"):
            verify_selection(scoped_legacy.execution.environment)
        with pytest.raises(ValueError, match="differs from its execution manifest"):
            verify_selection(default.execution.environment)
    with pytest.raises(ValueError, match="differs from its execution manifest"):
        verify_selection(scoped_legacy.execution.environment)


def test_spawned_steady_worker_executes_assembled_and_refuses_legacy_manifest(
    steady_source: BuildSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from support.equation_provenance import expected

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    assembled_job = steady_job(steady_source)
    with equation_implementation(LEGACY):
        legacy_scoped_job = steady_job(steady_source)

    limits = ResourceLimits(30.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        ok_handle = assembled_job.submit(supervisor, "assembled-default", limits)
        bad_handle = legacy_scoped_job.submit(supervisor, "legacy-refused", limits)

        ok_receipt = supervisor.wait(ok_handle, 90).data
        assert ok_receipt["store_state"] == "succeeded", ok_receipt
        attempt, = store.attempts("assembled-default")
        worker_res = attempt["result"]["worker_result"]
        device = compile_device_1d(load_device(), intervals_per_layer=12)
        assert binding(worker_res) == expected("DC1DProblem.steady_jv", device, ASSEMBLED)

        bad_receipt = supervisor.wait(bad_handle, 90).data
        assert bad_receipt["store_state"] == "failed"
        assert "legacy worker equation selection is retired" in bad_receipt["worker_result"]["detail"]


@pytest.mark.parametrize("family", ("transient", "ionic_transient", "ionic_jv"))
def test_every_1d_worker_family_executes_assembled_and_records_provenance(
    family: str, commit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.research_1d.ionic_steady import controls
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from support.equation_provenance import expected

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    prepare = {"transient": transient_job, "ionic_transient": ionic_transient_job, "ionic_jv": ionic_jv_job}[family]
    job = prepare(commit)
    assert declared(job) == ASSEMBLED

    limits = ResourceLimits(30.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        handle = job.submit(supervisor, family, limits)
        receipt = supervisor.wait(handle, 90).data
        assert receipt["store_state"] == "succeeded", receipt
        attempt, = store.attempts(family)
        result = attempt["result"]["worker_result"]

    if family == "transient":
        device: Any = compile_device_1d(load_device(), intervals_per_layer=12)
        operator, closure = "BackwardEuler1DStep", None
    else:
        device = compile_ionic_device_1d(
            prepare_confined_shared_site_ion_device(),
            intervals_per_layer=(12, 12, 12),
            ion_domain_layer="absorber",
            ion_domain_endpoints=("blocking", "blocking"),
        )
        operator = "IonicBackwardEuler1DStep" if family == "ionic_transient" else "Ionic1DProblem.steady_jv"
        closure = controls(13).inventory_row_node
    assert binding(result) == expected(operator, device, ASSEMBLED, closure)


def test_the_2d_steady_worker_executes_assembled_and_refuses_legacy_env(
    commit: str, tmp_path: Path,
) -> None:
    from examples.research_1d.steady import load_device
    from examples.research_2d.dc_assembly import spatial_input
    from solarlab.device.compile_2d import compile_device_2d
    from support.equation_provenance import expected

    archive = tmp_path / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), commit, "--", "src/solarlab"],
        cwd=ROOT,
        check=True,
        timeout=30,
    )
    device = compile_device_2d(load_device(), spatial_input())
    base_env = {key: value for key, value in os.environ.items() if key != ENVIRONMENT}
    base_env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        }
    )

    receipt = tmp_path / "assembled.json"
    ok_res = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "examples.research_2d.process_steady",
            "run",
            "--store",
            str(tmp_path / "assembled" / "Store"),
            "--work",
            str(tmp_path / "assembled" / "Work"),
            "--run-id",
            "steady2d-assembled",
            "--build-source-commit",
            commit,
            "--source-archive",
            str(archive),
            "--current-repository-head",
            commit,
            "--targets",
            "dark",
            "--receipt-json",
            str(receipt),
        ],
        cwd=ROOT,
        env=base_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ok_res.returncode == 0, ok_res.stderr + ok_res.stdout[-2000:]
    saved = json.loads(receipt.read_text())
    worker = saved["receipt"]["worker_result"]
    assert binding(worker) == expected("DC2DProblem.steady", device, ASSEMBLED)
    assert [ENVIRONMENT, json.dumps(ASSEMBLED)] in [list(item) for item in worker["capture"]["runtime"]["settings"]]

    legacy_env = {**base_env, ENVIRONMENT: LEGACY}
    bad_res = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "examples.research_2d.process_steady",
            "run",
            "--store",
            str(tmp_path / "legacy" / "Store"),
            "--work",
            str(tmp_path / "legacy" / "Work"),
            "--run-id",
            "steady2d-legacy",
            "--build-source-commit",
            commit,
            "--source-archive",
            str(archive),
            "--current-repository-head",
            commit,
            "--targets",
            "dark",
            "--receipt-json",
            str(tmp_path / "legacy.json"),
        ],
        cwd=ROOT,
        env=legacy_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert bad_res.returncode != 0
    assert "NEWCORE_EQUATIONS=legacy is retired" in (bad_res.stderr + bad_res.stdout)

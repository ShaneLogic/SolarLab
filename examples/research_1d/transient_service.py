"""Serve explicitly installed steady and ion-free transient jobs on localhost.

GET /experiments supplies the device/schema bindings and submission headers.
Use example_submission() for the small completed or limited-Newton protocol.
POST /jobs returns existing event/result URLs; a numerical failure retains its
prefix with state failed, and cancellation never invents a result history.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.worker import ResourceLimits
    from solarlab.materials.source import SourceDocument
    from solarlab_server.job_service import ExecutionCatalog

ROOT = Path(__file__).resolve().parents[2]


def resource_limits() -> ResourceLimits:
    from solarlab.experiments.worker import ResourceLimits
    return ResourceLimits(60., 1536 * 1024**2, 96 * 1024**2, None, None)


def example_submission(run_id: str = "http-transient", *, failure: bool = False) -> dict[str, Any]:
    times = [0., 1e-7, 2e-7, 3e-7] if failure else [0., 1e-7, 2e-7]
    generation = [1., 1., 1.5, 1.] if failure else [1., 1.0025, 1.01]
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id,
        "experiment_id": "transient_1d", "version": 1, "engine": "solarlab",
        "input": {"device_id": "research_1d", "times_s": times, "voltage_V": [0.] * len(times),
            "generation_scale": generation, "intervals_per_layer": 12,
            "controls": {"max_steps": 32, "wall_s": 30., "newton_budget": {
                "max_iterations": 1 if failure else 50, "max_residual_evaluations": 10 if failure else 1000,
                "max_jacobian_evaluations": 1 if failure else 50, "max_backtracks": 4 if failure else 24}}},
        "limits": {"wall_seconds": 60., "rss_bytes": 1536 * 1024**2, "output_bytes": 96 * 1024**2,
                   "accepted_steps": None, "calls": None}}


def installed_catalog(implementation: BuildSource, current_repository_head: str) -> ExecutionCatalog:
    """Co-install the two concrete adapters on one catalog and one supervisor."""
    from solarlab.experiments.registry import ExperimentRegistry
    from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog
    from solarlab_server.transient_preparation import TransientPreparationContext, transient_execution_catalog
    if __package__:
        from .steady import load_device
    else:
        from steady import load_device

    devices = (("research_1d", load_device()),)
    transient = transient_execution_catalog(TransientPreparationContext(devices, implementation, current_repository_head),
                                            ceiling=resource_limits())
    steady = steady_execution_catalog(SteadyPreparationContext(devices, implementation, current_repository_head),
                                     ceiling=resource_limits())
    registry = ExperimentRegistry((*steady.registry.definitions, *transient.registry.definitions))
    sources: dict[str, SourceDocument] = {}
    for source in (*steady.sources, *transient.sources):
        if source.id in sources and source != sources[source.id]:
            raise ValueError("co-installed source bindings disagree")
        sources[source.id] = source
    return replace(steady, registry=registry, sources=tuple(sources.values()),
        registrations=(*steady.registrations, *transient.registrations), expected_registry_sha256=registry.content_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--build-source-commit", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    # Spawn reconstructs this module before worker isolation; imports stay here.
    import uvicorn
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService
    if __package__:
        from .process_transient import source_build
    else:
        from process_transient import source_build

    # The isolated service does not claim unrelated ionic/AC working sources.
    # Shared-suite tests instead pass the full committed BuildSource explicitly.
    implementation = source_build(args.build_source_commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    catalog = installed_catalog(implementation, head)
    store_path, work = args.store.expanduser().resolve(), args.work.expanduser().resolve()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_path) as store, ProcessSupervisor(store, work, catalog.worker_bindings(),
            max_active=1, max_pending=2, shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        app = create_app(store.root, job_service=JobService(catalog, supervisor))
        print(json.dumps({"base_url": f"http://127.0.0.1:{args.port}",
            "build_source_commit": implementation.build_source_commit, "current_repository_head": head,
            "execution_catalog_sha256": catalog.document.sha256, "job_schema_sha256": catalog.schema.sha256,
            "example_submission": example_submission(), "scientific_qualification": False}), flush=True)
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, loop="asyncio", http="h11")
        except KeyboardInterrupt:
            # Uvicorn re-raises SIGINT after its normal ASGI shutdown. Let the
            # enclosing owner/store contexts finish before returning normally.
            pass
    print(json.dumps({"service_closed": True, "read_handle_closed": app.state.run_store is None,
                      "process_owner_closed": supervisor._closed}), flush=True)


if __name__ == "__main__":
    main()

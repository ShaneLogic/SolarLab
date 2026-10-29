"""Serve installed no-grain full2D steady jobs on the shared HTTP service.

GET /experiments gives the fixed geometry, complete controls and submit headers.
POST /jobs starts an actual ProcessSupervisor child. The existing result,
artifact and event routes also read original historical process attempts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.worker import ResourceLimits
    from solarlab_server.job_service import ExecutionCatalog


def resource_limits() -> ResourceLimits:
    from solarlab.experiments.worker import ResourceLimits
    return ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None)


def example_submission(run_id: str = "http-steady2d-completed", *, insufficient_budget: bool = False) -> dict[str, Any]:
    from solarlab_server.steady_2d_preparation import _policy
    policy = _policy()
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id,
        "experiment_id": "steady_2d", "version": 1, "engine": "solarlab",
        "input": {"device_id": "research_2d_uniform", "voltage_V": .002, "generation_scales": [0., 1.],
            "spatial": policy["spatial"], "initial": policy["initial"],
            "controls": policy["insufficient_budget_controls" if insufficient_budget else "controls"]},
        "limits": asdict(resource_limits())}


def installed_catalog(implementation: BuildSource, current_repository_head: str) -> ExecutionCatalog:
    from examples.research_1d.steady import load_device
    from solarlab_server.steady_2d_preparation import Steady2DPreparationContext, steady_2d_execution_catalog
    context = Steady2DPreparationContext((("research_2d_uniform", load_device()),), implementation, current_repository_head)
    return steady_2d_execution_catalog(context, ceiling=resource_limits())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--build-source-commit", required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--current-repository-head", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between1 and65535")
    from examples.research_1d.process_ionic_transient import source_build
    from solarlab_server.job_service import JobSubmission
    implementation = source_build(args.build_source_commit, args.source_archive)
    catalog = installed_catalog(implementation, args.current_repository_head)
    startup = {"base_url": f"http://127.0.0.1:{args.port}", "build_source_commit": implementation.build_source_commit,
        "current_repository_head": args.current_repository_head, "execution_catalog_sha256": catalog.document.sha256,
        "job_schema_sha256": catalog.schema.sha256, "example_submission": example_submission(), "scientific_qualification": False}
    if args.prepare_only:
        spec, binding = catalog.prepare(JobSubmission.model_validate(example_submission()))
        print(json.dumps({**startup, "prepared_only": True, "new_processes": 0, "new_solves": 0,
            "worker_id": spec.worker_id, "request_sha256": spec.request.request.sha256,
            "dimension": binding.inputs.selection.context.dimension, "manifest": binding.summary()}, indent=2), flush=True)
        return
    import uvicorn
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService
    root, work = args.store.expanduser().resolve(), args.work.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(root) as store, ProcessSupervisor(store, work, catalog.worker_bindings(), max_active=1, max_pending=2,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        app = create_app(store.root, job_service=JobService(catalog, supervisor))
        print(json.dumps({**startup, "store_id": store.store_id}), flush=True)
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, loop="asyncio", http="h11")
        except KeyboardInterrupt:
            pass
    print(json.dumps({"service_closed": True, "read_handle_closed": app.state.run_store is None,
                      "process_owner_closed": supervisor._closed}), flush=True)


if __name__ == "__main__":
    main()

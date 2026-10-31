"""Serve the explicit verified illuminated ionic J-V protocol on existing HTTP.

GET /experiments supplies registered devices, full controls and submit headers.
POST /jobs starts one actual existing process; results/artifacts/events expose
the same committed store, including historical failed process prefixes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.worker import ResourceLimits
    from solarlab_server.job_service import ExecutionCatalog


def resource_limits() -> ResourceLimits:
    from solarlab.experiments.worker import ResourceLimits
    from solarlab.experiments.jv.ionic_worker import REGISTERED_WALL_SECONDS
    return ResourceLimits(REGISTERED_WALL_SECONDS,
                          1536 * 1024**2, 96 * 1024**2, None, None)


def example_submission(run_id: str = "http-ionic-jv-completed") -> dict[str, Any]:
    from solarlab_server.ionic_jv_preparation import _policy
    policy = _policy()
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id,
        "experiment_id": "ionic_steady_jv", "version": 1, "engine": "solarlab",
        "input": {"device_id": "research_1d_shared", **policy["protocol"],
            "intervals_per_layer": policy["intervals_per_layer"], "controls": policy["controls"], "initial": policy["initial"]},
        "limits": asdict(resource_limits())}


def installed_catalog(implementation: BuildSource, current_repository_head: str) -> ExecutionCatalog:
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab_server.ionic_jv_preparation import IonicJVPreparationContext, ionic_jv_execution_catalog
    context = IonicJVPreparationContext((("research_1d_shared", prepare_confined_shared_site_ion_device()),),
                                       "absorber", implementation, current_repository_head)
    return ionic_jv_execution_catalog(context, ceiling=replace(resource_limits(), wall_seconds=30.),
                                     wall_profile_sha256=context.registered_wall_profile())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--build-source-commit", required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--current-repository-head", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between1 and65535")
    import uvicorn
    from examples.research_1d.process_ionic_transient import source_build
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService
    implementation = source_build(args.build_source_commit, args.source_archive)
    catalog = installed_catalog(implementation, args.current_repository_head)
    root, work = args.store.expanduser().resolve(), args.work.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(root) as store, ProcessSupervisor(store, work, catalog.worker_bindings(), max_active=1, max_pending=2,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        app = create_app(store.root, job_service=JobService(catalog, supervisor))
        print(json.dumps({"base_url": f"http://127.0.0.1:{args.port}", "store_id": store.store_id,
            "build_source_commit": implementation.build_source_commit, "current_repository_head": args.current_repository_head,
            "execution_catalog_sha256": catalog.document.sha256, "job_schema_sha256": catalog.schema.sha256,
            "example_submission": example_submission(), "scientific_qualification": False}), flush=True)
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, loop="asyncio", http="h11")
        except KeyboardInterrupt:
            pass
    print(json.dumps({"service_closed": True, "read_handle_closed": app.state.run_store is None,
                      "process_owner_closed": supervisor._closed}), flush=True)


if __name__ == "__main__":
    main()

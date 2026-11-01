"""Serve explicitly installed dark positive-D ionic transient research jobs.

GET /experiments supplies the real devices, schema and required headers.
POST /jobs accepts explicit samples and work budgets. Completed and failed
prefixes share the existing results/artifact/SSE routes; readback never solves.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.worker import ResourceLimits
    from solarlab_server.job_service import ExecutionCatalog

ROOT = Path(__file__).resolve().parents[2]


def resource_limits() -> ResourceLimits:
    from solarlab.experiments.worker import ResourceLimits
    return ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None)


def example_submission(run_id: str = "http-ionic-transient", *, failure: bool = False) -> dict[str, Any]:
    from examples.research_1d.ionic_steady import controls as steady_controls
    from examples.research_1d.ionic_transient import controls as time_controls
    initial = asdict(steady_controls(13))
    electronic = asdict(time_controls())["electronic"]
    if failure:
        electronic["newton_budget"] = {"max_iterations": 1, "max_residual_evaluations": 10,
                                      "max_jacobian_evaluations": 1, "max_backtracks": 4}
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id,
        "experiment_id": "ionic_transient_1d", "version": 1, "engine": "solarlab",
        "input": {"device_id": "research_1d_shared", "times_s": [0., .25, .5, .75, 1.],
            "voltage_V": [.002, .002 if failure else .0025, .003, .0035, .004],
            "generation_scale": [0.] * 5, "intervals_per_layer": [12, 12, 12],
            "controls": {"initial_steady": {name: initial[name] for name in ("budget", "inventory_row_node", "wall_s")},
                "transient": {name: electronic[name] for name in ("newton_budget", "max_steps", "wall_s")}}},
        "limits": asdict(resource_limits())}


def installed_catalog(implementation: BuildSource, current_repository_head: str) -> ExecutionCatalog:
    from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab_server.ionic_transient_preparation import IonicTransientPreparationContext, ionic_transient_execution_catalog
    devices = (("research_1d_positive", prepare_confined_ion_device(diffusion_only=True)),
               ("research_1d_dual", prepare_confined_dual_ion_device(diffusion_only=True)),
               ("research_1d_shared", prepare_confined_shared_site_ion_device()))
    return ionic_transient_execution_catalog(IonicTransientPreparationContext(
        devices, "absorber", implementation, current_repository_head), ceiling=resource_limits())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--build-source-commit", required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--current-repository-head")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    # Spawn reconstructs this module before isolation, so engine imports stay
    # inside the application entry point and ordinary helper calls.
    import uvicorn
    from examples.research_1d.process_ionic_transient import source_build
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.io.run_store import RunStore
    from solarlab_server.app import create_app
    from solarlab_server.job_service import JobService
    implementation = source_build(args.build_source_commit, args.source_archive)
    head = args.current_repository_head or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    catalog = installed_catalog(implementation, head)
    store_path, work = args.store.expanduser().resolve(), args.work.expanduser().resolve()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_path) as store, ProcessSupervisor(store, work, catalog.worker_bindings(),
            max_active=1, max_pending=2, shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        app = create_app(store.root, job_service=JobService(catalog, supervisor))
        print(json.dumps({"base_url": f"http://127.0.0.1:{args.port}", "build_source_commit": implementation.build_source_commit,
            "current_repository_head": head, "execution_catalog_sha256": catalog.document.sha256,
            "job_schema_sha256": catalog.schema.sha256, "example_submission": example_submission(),
            "scientific_qualification": False}), flush=True)
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, loop="asyncio", http="h11")
        except KeyboardInterrupt:
            pass  # Uvicorn re-raises SIGINT after normal ASGI shutdown.
    print(json.dumps({"service_closed": True, "read_handle_closed": app.state.run_store is None,
                      "process_owner_closed": supervisor._closed}), flush=True)


if __name__ == "__main__":
    main()

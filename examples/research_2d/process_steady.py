"""Run/read one explicit new-core 2D process with the existing supervisor/store.

Run uses two cold auxiliary seeds and two full 2D Newton solves for dark/G1.
--insufficient-budget changes only the 2D Newton count budget. Read never
imports a solver or starts a process; failed prefixes intentionally exit 2.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any


def build_job(args: argparse.Namespace) -> Any:
    from examples.research_1d.process_ionic_transient import source_build
    from examples.research_1d.steady import load_device
    from examples.research_2d.dc_assembly import spatial_input
    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.two_dimensional.steady_worker import Steady2DProtocol, prepare_steady_2d_job
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions

    controls = SteadyOptions()
    if args.insufficient_budget:
        controls = replace(controls, max_iterations=1, max_residual_evaluations=1, max_jacobian_evaluations=1)
    scales = {"dark": (0.,), "illuminated": (1.,), "both": (0., 1.)}[args.targets]
    return prepare_steady_2d_job(load_device(), spatial_input(), Steady2DProtocol(.002, scales),
        options=Steady2DOptions(controls, 30.), implementation=source_build(args.build_source_commit, args.source_archive),
        current_repository_head=args.current_repository_head, engine=ExecutionEngine("solarlab"))


def summary(loaded: dict[str, Any]) -> dict[str, Any]:
    return {"state": loaded["run_state"], **loaded["document"],
            "array_fingerprints": loaded["array_fingerprints"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--build-source-commit", required=True)
    run.add_argument("--source-archive", type=Path, required=True)
    run.add_argument("--current-repository-head", required=True)
    run.add_argument("--targets", choices=("dark", "illuminated", "both"), default="both")
    run.add_argument("--insufficient-budget", action="store_true")
    run.add_argument("--prepare-only", action="store_true", help="bind the request without a process, compile or solve")
    run.add_argument("--receipt-json", type=Path)
    read = commands.add_parser("read")
    read.add_argument("--store", type=Path, required=True)
    read.add_argument("--run-id", required=True)
    read.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    from solarlab.experiments.two_dimensional.steady_results import read_steady_2d_job
    from solarlab.io.run_store import RunStore
    if args.command == "read":
        with RunStore(args.store.resolve(), read_only=True) as store:
            loaded = read_steady_2d_job(store, args.run_id, attempt_id=args.attempt_id)
            result = summary(loaded)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0 if result["state"] == "succeeded" else 2

    job = build_job(args)
    if args.prepare_only:
        print(json.dumps({"prepared_only": True, "worker_id": "steady_2d", "new_processes": 0,
            "new_solves": 0, "request": json.loads(job.request.request.content),
            "request_sha256": job.request.request.sha256, "registry_sha256": job.request.registry.sha256,
            "schema_sha256": job.request.schema.sha256, "manifest": job.execution.summary()}, indent=2), flush=True)
        return 0
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.two_dimensional.steady_worker import RESULT_SCHEMA, steady_2d_worker_binding
    from solarlab.experiments.worker import ResourceLimits
    limits = ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None)
    args.store.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.work.resolve().parent.mkdir(parents=True, exist_ok=True)
    with RunStore(args.store.resolve()) as store, ProcessSupervisor(store, args.work.resolve(),
            (steady_2d_worker_binding(),), max_active=1, max_pending=1,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as owner:
        try:
            store.get_run(args.run_id)
        except KeyError:
            pass
        else:
            raise ValueError("run already exists; use read with its exact attempt, or an explicit new run ID")
        handle = job.submit(owner, args.run_id, limits)
        receipt = owner.wait(handle, 40).data
        result = {"state": receipt["store_state"], "run_id": handle.run_id, "attempt_id": handle.attempt_id,
            "store_id": store.store_id, "receipt": receipt}
        if receipt.get("worker_result", {}).get("schema") == RESULT_SCHEMA:
            result["readback"] = summary(read_steady_2d_job(store, handle.run_id, attempt_id=handle.attempt_id))
    result["process_owner_closed"] = owner._closed
    if args.receipt_json:
        args.receipt_json.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt_json.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result["state"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

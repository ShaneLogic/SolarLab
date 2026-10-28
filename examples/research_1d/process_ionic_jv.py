"""Run or read an illuminated shared-site ionic J-V process job.

The default is the accepted37/13-node fixture and51 samples from0 to1.25V.
Initialization explicitly performs cold dark, direct_density_seed_v1, then
the coupled illuminated solve. An exact saved initial may instead be supplied
with its SHA; existing same-device/control admission remains mandatory.
Generation1 is not an irradiance. Efficiency and qualification stay unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from examples.research_1d.process_ionic_transient import source_build

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.jv.ionic_worker import IonicJVJob


def example_job(implementation: BuildSource, *, failure: bool = False, initial_bytes: bytes | None = None,
                initial_sha256: str | None = None, current_repository_head: str | None = None) -> IonicJVJob:
    from examples.research_1d.ionic_steady import controls
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.experiments.jv.ionic_worker import IonicJVInitial, IonicJVProtocol, prepare_ionic_jv_job

    initial = IonicJVInitial("dark_direct_density_seed_v1" if initial_bytes is None else "saved_accepted",
        .002, 25., initial_bytes, initial_sha256)
    return prepare_ionic_jv_job(prepare_confined_shared_site_ion_device(),
        IonicJVProtocol(tuple(tick / 40. for tick in range(51)), 1., None), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"),
        options=IonicSteadyJVOptions(controls(13), max_points=51, wall_s=1. if failure else 25.), initial=initial,
        implementation=implementation, current_repository_head=current_repository_head or implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"))


def summarize(loaded: dict[str, Any]) -> dict[str, Any]:
    worker = loaded["worker_result"]
    return {"state": loaded["run_state"], "result": loaded["document"],
        "requested_voltage_V": loaded["arrays"]["requested_voltage_V"].tolist(),
        "accepted_voltage_V": loaded["arrays"]["voltage_V"].tolist(),
        "output_current_A_m2": loaded["arrays"]["output_current_A_m2"].tolist(),
        "identities": {key: worker[key] for key in ("H_input", "H_physics", "H_execution", "manifest_sha256")},
        "arrays": {name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "unit": loaded["document"]["logical_array_units"][name]} for name, a in loaded["arrays"].items()},
        "initial_artifact_sha256": hashlib.sha256(loaded["initial_bytes"]).hexdigest(),
        "capture": worker["capture"], "scientific_qualification": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--build-source-commit", required=True)
    run.add_argument("--source-archive", type=Path)
    run.add_argument("--current-repository-head")
    run.add_argument("--run-id", default="ionic-jv-completed")
    run.add_argument("--failure-case", action="store_true", help="one-second curve budget, unchanged steady controls")
    run.add_argument("--initial-result", type=Path)
    run.add_argument("--initial-sha256")
    for name in ("read", "export-initial"):
        command = commands.add_parser(name)
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--attempt-id")
        if name == "export-initial":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from solarlab.experiments.jv.ionic_worker import ionic_jv_worker_binding, read_ionic_jv_job
    from solarlab.io.run_store import RunStore
    if args.command in {"read", "export-initial"}:
        with RunStore(args.store.resolve(), read_only=True) as store:
            loaded = read_ionic_jv_job(store, args.run_id, attempt_id=args.attempt_id)
        if args.command == "export-initial":
            if loaded["diagnostics"]["curve"] is None:
                raise ValueError("no admitted illuminated initial exists for this attempt")
            with args.output.open("xb") as stream:
                stream.write(loaded["initial_bytes"])
            print(json.dumps({"path": str(args.output.resolve()), "sha256": hashlib.sha256(loaded["initial_bytes"]).hexdigest(),
                "run_id": args.run_id, "attempt_id": loaded["document"]["attempt_id"], "new_solver_calls": 0}, indent=2))
            return 0
        print(json.dumps(summarize(loaded), indent=2, allow_nan=False))
        return 0 if loaded["run_state"] == "succeeded" else 2
    if (args.initial_result is None) != (args.initial_sha256 is None):
        parser.error("saved initial requires both --initial-result and --initial-sha256")
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.worker import ResourceLimits
    implementation = source_build(args.build_source_commit, args.source_archive)
    job = example_job(implementation, failure=args.failure_case,
        initial_bytes=None if args.initial_result is None else args.initial_result.read_bytes(),
        initial_sha256=args.initial_sha256, current_repository_head=args.current_repository_head)
    with RunStore(args.store.resolve()) as store, ProcessSupervisor(store, args.work.resolve(),
            (ionic_jv_worker_binding(),), max_active=1, max_pending=1,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        handle = job.submit(supervisor, args.run_id, ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None))
        receipt = supervisor.wait(handle, 40).data
        attempt = next(a for a in store.attempts(handle.run_id) if a["attempt_id"] == handle.attempt_id)
        has_result = (attempt["result"] or {}).get("worker_result", {}).get("ionic_jv_result") is not None
        loaded = read_ionic_jv_job(store, handle.run_id, attempt_id=handle.attempt_id) if has_result else None
        result = {"store": str(store.root), "receipt": receipt, "curve": None if loaded is None else summarize(loaded),
            "terminal_count": len([e for e in store.events(run_id=handle.run_id, limit=1000) if e["kind"] == "terminal"])}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if receipt["store_state"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

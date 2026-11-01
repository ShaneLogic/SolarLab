"""Run two cold lifetime points through the existing bounded process runtime.

Run with an explicit committed source revision and disjoint local store/work
directories; use ``read --store ... --declaration ...`` for solver-free readback.
The fixed example uses tau_n = 1 and 2 us, 37 nodes and 0, .05, .1 V. It does
not bracket Voc, validate a model, or implement ions, AC or a timed protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource

ROOT = Path(__file__).resolve().parents[2]
DC_SOURCE_EXCLUSIONS = (
    "solarlab/experiments/impedance.py", "solarlab/experiments/transient.py",
    "solarlab/solvers/linear/", "solarlab/solvers/time/",
)


def committed_dc_implementation(commit: str) -> BuildSource:
    """Reuse exact Git object bytes, with AC/time explicitly outside this scope.

    The steady worker independently refuses a loaded module without a binding.
    Active adapter bytes are bound by that worker, never relabelled as a commit.
    """
    from solarlab.config.identity import BuildSource
    if __package__:
        from .service import committed_implementation
    else:
        from service import committed_implementation
    source = committed_implementation(commit)
    return BuildSource(source.build_source_commit, tuple(item for item in source.files
        if not any(item.id.startswith(prefix) for prefix in DC_SOURCE_EXCLUSIONS)))


def summarize(points: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Compact output keeps every attempt state and the actual current arrays."""
    output = []
    for point in points:
        value = {name: point[name] for name in ("point_id", "run_id", "state", "reason", "attempt_id")}
        value.update(parameters=point["coordinates"]["normalized_coordinates"],
            units=point["coordinates"]["units"],
            attempts=[{name: a[name] for name in ("attempt_id", "generation", "state", "reason")}
                      for a in point["attempts"]])
        loaded = point["result"]
        if loaded is not None:
            value.update(voltage_V=loaded["arrays"]["voltage_V"].tolist(),
                output_current_A_m2=loaded["arrays"]["output_current_A_m2"].tolist(),
                compiled_identity=loaded["document"]["compiled_identity"],
                metrics=loaded["document"]["metrics"],
                manifest_sha256=loaded["worker_result"]["manifest_sha256"],
                identities={name: loaded["worker_result"][name] for name in ("H_input", "H_physics", "H_execution")},
                arrays={name: {"shape": list(array.shape), "dtype": array.dtype.str,
                              "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                              "artifact_id": loaded["document"]["arrays"][name]}
                        for name, array in loaded["arrays"].items()})
        output.append(value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the two-point, single-worker lifetime example")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--build-source-commit", required=True)
    run.add_argument("--study-id", default="process-lifetime")
    read = commands.add_parser("read", help="verify and read committed arrays without a solver")
    read.add_argument("--store", type=Path, required=True)
    read.add_argument("--declaration", type=Path, required=True)
    arguments = parser.parse_args()

    # Spawn replays __main__: keep all scientific imports inside this function.
    from solarlab.experiments.execution import ProcessSupervisor, ScanSpec
    from solarlab.io.run_store import RunStore
    from solarlab.sweeps.steady_worker import read_steady_sweep_jobs

    store_path = arguments.store.expanduser().resolve()
    if arguments.command == "read":
        spec = ScanSpec.from_bytes(arguments.declaration.read_bytes())
        with RunStore(store_path, read_only=True) as store:
            points = read_steady_sweep_jobs(store, spec)
        print(json.dumps({"scan_id": spec.scan_id, "points": summarize(points)}, indent=2, allow_nan=False))
        return 0 if all(p["state"] == "succeeded" for p in points) else 1

    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.jv.worker import SteadyJVProtocol, steady_jv_worker_binding
    from solarlab.experiments.worker import ResourceLimits
    from solarlab.sweeps.steady_worker import prepare_steady_sweep_jobs
    if __package__:
        from .jv_scan import make_sweep
    else:
        from jv_scan import make_sweep

    implementation = committed_dc_implementation(arguments.build_source_commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    sweep = make_sweep(Path(__file__).with_name("device.json"), study_id=arguments.study_id,
                       parameter="tau_n", values=["1 us", "2 us"])
    limits = ResourceLimits(60.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    jobs = prepare_steady_sweep_jobs(sweep, SteadyJVProtocol((0., .05, .1), 12, 1., None),
        limits=limits, implementation=implementation, current_repository_head=head,
        engine=ExecutionEngine("solarlab"))
    work = arguments.work.expanduser().resolve()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_path) as store, ProcessSupervisor(store, work, (steady_jv_worker_binding(),),
            max_active=1, max_pending=1, shared_rss_bytes=3 * 1024**3,
            shared_output_bytes=384 * 1024**2) as supervisor:
        scan = jobs.submit(supervisor)
        status = scan.wait(150)
        points = read_steady_sweep_jobs(store, jobs.spec)
        terminal_counts = {point.run_id: len([event for event in store.events(run_id=point.run_id, limit=1000)
                                            if event["kind"] == "terminal"])
                           for point in jobs.spec.points}
        print(json.dumps({"scan_id": jobs.spec.scan_id, "state": status["state"],
            "complete": status["complete"], "store": str(store_path), "declaration": str(scan.declaration_path),
            "build_source_commit": implementation.build_source_commit, "current_repository_head": head,
            "dc_source_exclusions": DC_SOURCE_EXCLUSIONS,
            "runtime": asdict(jobs.executions[0].execution.environment),
            "max_active": supervisor.max_active, "max_in_flight": scan.max_in_flight,
            "terminal_counts": terminal_counts, "points": summarize(points),
            "scientific_qualification": False}, indent=2, allow_nan=False))
    return 0 if status["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

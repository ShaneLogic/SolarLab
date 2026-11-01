"""Run/read a cold 37-node transient through the existing ProcessSupervisor.

The default case has two fixed-bias light-ramp steps. --failure-case uses an
explicit one-iteration Newton budget and retains its first failed attempt.
Neither case grants a scientific certificate. The read command runs no solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.transient_worker import TransientJob

ROOT = Path(__file__).resolve().parents[2]
SOURCE_EXCLUSIONS = ("solarlab/device/compile_ionic_1d.py", "solarlab/equations/ionic_1d.py",
    "solarlab/solvers/nonlinear/ionic_1d.py", "solarlab/experiments/impedance.py", "solarlab/solvers/linear/")


def source_build(commit: str, archive: Path | None = None) -> BuildSource:
    """Use exact Git archive bytes; active adapters are recorded separately."""
    from solarlab.config.identity import BuildSource
    from solarlab.materials.source import SourceDocument

    content = (archive.read_bytes() if archive is not None else
               subprocess.check_output(["git", "archive", commit, "--", "src/solarlab"], cwd=ROOT, timeout=10))
    sources = []
    with tarfile.open(fileobj=io.BytesIO(content)) as source:
        if source.pax_headers.get("comment") != commit:
            raise ValueError("archive does not declare the exact supplied immutable commit")
        for item in source.getmembers():
            name = item.name.removeprefix("src/")
            if (item.isfile() and item.size and name.startswith("solarlab/") and name.endswith(".py")
                    and not any(name.startswith(prefix) for prefix in SOURCE_EXCLUSIONS)):
                stream = source.extractfile(item)
                assert stream is not None
                sources.append(SourceDocument(name, stream.read()))
    return BuildSource(commit, tuple(sources))


def example_job(implementation: BuildSource, *, failure: bool = False,
                current_repository_head: str | None = None) -> TransientJob:
    """Use the supplied experiment and ordinary final solver gates unchanged."""
    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.transient import TransientOptions, TransientProtocol
    from solarlab.experiments.transient_worker import prepare_transient_job
    from solarlab.solvers.nonlinear.problem import SolveBudget
    from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    if __package__:
        from .steady import load_device
    else:
        from steady import load_device

    times = [0., 1e-7, 2e-7, 3e-7] if failure else [0., 1e-7, 2e-7]
    generation = [1., 1., 1.5, 1.] if failure else [1., 1.0025, 1.01]
    options = TransientOptions(1e-11, 1e-8, 1e-11, .1, 2.,
        SolveBudget(1, 10, 1, 4) if failure else SolveBudget(50, 1000, 50, 24),
        32, 30., "previous", POISSON_CONTACT_BALANCE_V1)
    head = current_repository_head if current_repository_head is not None else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    return prepare_transient_job(load_device(), TransientProtocol(times, [0.] * len(times), generation),
        intervals_per_layer=12, options=options, initial_options=SteadyOptions(), implementation=implementation,
        current_repository_head=head, engine=ExecutionEngine("solarlab"))


def summarize(loaded: dict[str, Any]) -> dict[str, Any]:
    result = loaded["worker_result"]
    return {"state": loaded["run_state"], "result": loaded["document"],
        "accepted_times_s": loaded["arrays"]["accepted_times_s"].tolist(),
        "terminal_total_current_density_A_m2": loaded["arrays"]["terminal_total_current_density_A_m2"].tolist(),
        "attempts": [{"start_s": a["start_s"], "end_s": a["end_s"], "accepted": a["accepted"],
                      "termination": a["termination"], "error": a["error"]} for a in loaded["diagnostics"]["attempts"]],
        "identities": {key: result[key] for key in ("H_input", "H_physics", "H_execution", "manifest_sha256")},
        "arrays": {name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                          "unit": result["payloads"][name]["metadata"]["unit"]} for name, a in loaded["arrays"].items()},
        "build_source_commit": loaded["manifest"]["document"]["execution_identity"]["payload"]["implementation"]["build_source_commit"],
        "cold_runtime": result["cold_start"]["runtime"], "scientific_qualification": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--build-source-commit", required=True)
    run.add_argument("--source-archive", type=Path)
    run.add_argument("--current-repository-head", help="explicit maintenance HEAD; otherwise observe the checkout")
    run.add_argument("--run-id", default="transient-completed")
    run.add_argument("--failure-case", action="store_true")
    read = commands.add_parser("read")
    read.add_argument("--store", type=Path, required=True)
    read.add_argument("--run-id", required=True)
    read.add_argument("--attempt-id")
    args = parser.parse_args()
    # Spawn replays __main__: numerical imports must remain inside functions.
    from solarlab.experiments.transient_worker import read_transient_job, transient_worker_binding
    from solarlab.io.run_store import RunStore

    store_path = args.store.expanduser().resolve()
    if args.command == "read":
        with RunStore(store_path, read_only=True) as store:
            loaded = read_transient_job(store, args.run_id, attempt_id=args.attempt_id)
        print(json.dumps(summarize(loaded), indent=2, allow_nan=False))
        return 0 if loaded["run_state"] == "succeeded" else 2

    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.worker import ResourceLimits

    implementation = source_build(args.build_source_commit, args.source_archive)
    job = example_job(implementation, failure=args.failure_case, current_repository_head=args.current_repository_head)
    limits = ResourceLimits(60., 1536 * 1024**2, 96 * 1024**2, None, None)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(store_path) as store, ProcessSupervisor(store, args.work.expanduser().resolve(),
            (transient_worker_binding(),), max_active=1, max_pending=1,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        handle = job.submit(supervisor, args.run_id, limits)
        # A wait timeout is not permission to submit a second run; context exit
        # closes the original owned process through the shared runtime.
        receipt = supervisor.wait(handle, 90).data
        history = (summarize(read_transient_job(store, handle.run_id, attempt_id=handle.attempt_id))
                   if receipt["terminal_artifacts"] else None)
        output = {"store": str(store_path), "receipt": receipt, "history": history,
            "limits": asdict(limits), "source_exclusions": SOURCE_EXCLUSIONS,
            "terminal_count": len([e for e in store.events(run_id=handle.run_id, limit=1000) if e["kind"] == "terminal"])}
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if receipt["store_state"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

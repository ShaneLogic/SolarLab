"""Run/read a cold blocking ionic history through the shared process owner.

The default shared-site case is37 electrical/13 ion nodes and four2-to4mV
steps. A limited-Newton case keeps an unchanged first step and its real first
failed driven step. No state is reused between separately submitted jobs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.experiments.ionic_transient_worker import IonicTransientJob

ROOT = Path(__file__).resolve().parents[2]


def source_build(commit: str, archive: Path | None = None) -> BuildSource:
    from solarlab.config.identity import BuildSource
    from solarlab.materials.source import SourceDocument
    data = archive.read_bytes() if archive is not None else subprocess.check_output(
        ["git", "archive", commit, "--", "src/solarlab"], cwd=ROOT, timeout=10)
    sources = []
    with tarfile.open(fileobj=io.BytesIO(data)) as snapshot:
        if snapshot.pax_headers.get("comment") != commit:
            raise ValueError("source archive must identify the exact supplied commit")
        for item in snapshot.getmembers():
            if item.isfile() and item.size and item.name.startswith("src/solarlab/") and item.name.endswith(".py"):
                stream = snapshot.extractfile(item)
                assert stream is not None
                sources.append(SourceDocument(item.name.removeprefix("src/"), stream.read()))
    return BuildSource(commit, tuple(sources))


def example_job(implementation: BuildSource, *, failure: bool = False, species: str = "shared",
                current_repository_head: str | None = None) -> IonicTransientJob:
    import numpy as np
    from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
    from examples.research_1d.ionic_steady import controls as steady_controls
    from examples.research_1d.ionic_transient import controls as time_controls
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.ionic_transient_worker import prepare_ionic_transient_job
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.solvers.nonlinear.problem import SolveBudget

    prepared = {"positive": lambda: prepare_confined_ion_device(diffusion_only=True),
                "dual": lambda: prepare_confined_dual_ion_device(diffusion_only=True),
                "shared": prepare_confined_shared_site_ion_device}[species]()
    options = time_controls()
    if failure:
        options = replace(options, electronic=replace(options.electronic, newton_budget=SolveBudget(1, 10, 1, 4)))
    times = np.linspace(0., 1., 5)
    volts = np.asarray([.002, .002, .003, .0035, .004]) if failure else .002 + .002 * times
    head = current_repository_head or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    return prepare_ionic_transient_job(prepared, TransientProtocol(times, volts, np.zeros(5)),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"),
        options=options, initial_options=steady_controls(13), implementation=implementation,
        current_repository_head=head, engine=ExecutionEngine("solarlab"))


def summarize(loaded: dict[str, Any]) -> dict[str, Any]:
    worker = loaded["worker_result"]
    history = loaded["diagnostics"]["history"]
    return {"state": loaded["run_state"], "result": loaded["document"],
        "accepted_times_s": loaded["arrays"]["accepted_times_s"].tolist(),
        "requested_times_s": loaded["arrays"]["requested_times_s"].tolist(),
        "attempts": [] if history is None else [{key: a[key] for key in
            ("start_s", "end_s", "accepted", "termination", "error")} for a in history["attempts"]],
        "identities": {key: worker[key] for key in ("H_input", "H_physics", "H_execution", "manifest_sha256")},
        "arrays": {name: {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "unit": loaded["document"]["logical_array_units"][name]}
            for name, a in loaded["arrays"].items()}, "capture": worker["capture"], "scientific_qualification": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--build-source-commit", required=True)
    run.add_argument("--source-archive", type=Path)
    run.add_argument("--current-repository-head")
    run.add_argument("--run-id", default="ionic-transient-completed")
    run.add_argument("--species", choices=("positive", "dual", "shared"), default="shared")
    run.add_argument("--failure-case", action="store_true")
    read = commands.add_parser("read")
    read.add_argument("--store", type=Path, required=True)
    read.add_argument("--run-id", required=True)
    read.add_argument("--attempt-id")
    args = parser.parse_args()
    from solarlab.experiments.ionic_transient_worker import ionic_transient_worker_binding, read_ionic_transient_job
    from solarlab.io.run_store import RunStore
    if args.command == "read":
        with RunStore(args.store.resolve(), read_only=True) as store:
            loaded = read_ionic_transient_job(store, args.run_id, attempt_id=args.attempt_id)
        print(json.dumps(summarize(loaded), indent=2, allow_nan=False))
        return 0 if loaded["run_state"] == "succeeded" else 2
    from solarlab.experiments.execution import ProcessSupervisor
    from solarlab.experiments.worker import ResourceLimits
    implementation = source_build(args.build_source_commit, args.source_archive)
    job = example_job(implementation, failure=args.failure_case, species=args.species,
                      current_repository_head=args.current_repository_head)
    with RunStore(args.store.resolve()) as store, ProcessSupervisor(store, args.work.resolve(),
            (ionic_transient_worker_binding(),), max_active=1, max_pending=1,
            shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2) as supervisor:
        handle = job.submit(supervisor, args.run_id, ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None))
        receipt = supervisor.wait(handle, 40).data
        history = read_ionic_transient_job(store, handle.run_id, attempt_id=handle.attempt_id) if receipt["terminal_artifacts"] else None
        result = {"store": str(store.root), "receipt": receipt, "history": None if history is None else summarize(history),
            "terminal_count": len([e for e in store.events(run_id=handle.run_id, limit=1000) if e["kind"] == "terminal"])}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if receipt["store_state"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

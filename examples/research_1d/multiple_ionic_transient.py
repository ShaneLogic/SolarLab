"""One four-step dark history from the saved two-domain 2 mV steady state."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls, protocol
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.multiple_ionic_steady import prepare_mixed_immobile_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.experiments.multiple_ionic_transient import (
    MultipleIonicTransientResult, load_multiple_ionic_steady_initial, read_multiple_ionic_transient,
    solve_multiple_ionic_transient, write_multiple_ionic_transient,
)
from solarlab.experiments.transient_results import _encode


def run_case(initial_path: Path, initial_sha256: str, output: Path, *, mixed_immobile: bool = False) -> MultipleIonicTransientResult:
    if output.exists():
        raise FileExistsError("preserve the original attempt; no automatic retry")
    device = compile_multiple_ionic_device_1d(prepare_mixed_immobile_device() if mixed_immobile else prepare_multiple_ionic_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    initial = load_multiple_ionic_steady_initial(device, initial_path, initial_sha256,
                                               controls=steady_controls(device.domains[0].ion_node_count))
    result = solve_multiple_ionic_transient(device, initial, protocol(4), options=controls())
    digest = write_multiple_ionic_transient(result, output)
    read_multiple_ionic_transient(output, digest)
    initial_state = result.states[0]
    changes: dict[str, Any] = {}
    for domain, before in zip(device.domains, initial_state.domain_component_states):
        name = domain.domain_layer_id
        for species in domain.ion_species_names:
            baseline = getattr(before, species + "_ion_density_m3")
            positions = [state.domain_component_states[device.domain_ids.index(name)] for state in result.states]
            values = [getattr(state, species + "_ion_density_m3") for state in positions]
            measured = [attempt.accounting["domains"][name]["species"][species]
                        for attempt in result.attempts if attempt.accepted and attempt.accounting is not None]
            changes[name + "." + species] = {
                "maximum_relative_density_delta": max(float(np.max(abs(value / baseline - 1.0))) for value in values),
                "cell_density_delta_m3": values[-1] - baseline,
                "physical_inventory_increment_m2": np.sum([record["physical_inventory_increment_m2"] for record in measured], axis=0)
                    if measured else np.zeros(domain.ion_node_count),
                "integrated_balance_l1_bound_m2": sum((attempt.end_s - attempt.start_s)
                    * attempt.accounting["domains"][name]["species"][species]["balance_l1_m2_s"]
                    for attempt in result.attempts if attempt.accepted and attempt.accounting is not None),
            }
    names = ("solarlab.solvers.time.multiple_ionic_backward_euler", "solarlab.experiments.multiple_ionic_transient",
             "solarlab.solvers.time.ionic_backward_euler", "solarlab.solvers.time.backward_euler",
             "solarlab.equations.multiple_ionic_1d", "solarlab.solvers.nonlinear.multiple_ionic_1d")
    sources = {}
    for name in names:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise ValueError("the actual numerical source must be bound")
        sources[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    summary = {
        "completed": result.completed, "accepted_steps": len(result.states) - 1,
        "requested_times_s": result.protocol.times_s, "accepted_times_s": result.accepted_times_s,
        "initial_source_sha256": initial_sha256, "result_sha256": digest, "source_sha256": sources,
        "physical_deltas": changes, "iterations": [None if a.newton is None else a.newton.iteration_count for a in result.attempts],
        "evaluation_counts": result.evaluation_counts, "wall_s": result.wall_s,
        "failed": [{"end_s": a.end_s, "termination": a.termination, "error": a.error}
                   for a in result.failed_attempts], "qualification": False,
    }
    output.with_name(output.stem + ".summary.json").write_text(json.dumps(_encode(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_encode(summary), indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--mixed-immobile", action="store_true")
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        document = read_multiple_ionic_transient(args.result, args.sha256)
        result = document["result"]
        failed = [attempt for attempt in result["attempts"] if not attempt["accepted"]]
        first = None if not failed else {
            key: failed[0].get(key) for key in ("start_s", "end_s", "termination", "error")}
        print(json.dumps(_encode({"completed": document["completed"],
            "accepted_times_s": result["accepted_times_s"], "attempt_count": len(result["attempts"]),
            "first_failed_step": first}), indent=2))
        return 0 if document["completed"] else 2
    return 0 if run_case(args.initial_result, args.initial_sha256, args.output,
                         mixed_immobile=args.mixed_immobile).completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

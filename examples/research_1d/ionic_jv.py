"""Equilibrated ionic steady J-V; no scan rate, time advance or hysteresis.

    python -m examples.research_1d.ionic_jv solve --initial-policy compiled_cold \
        --voltages 0.001 0.002 0.003 --generation-scale 0 --output /tmp/ionic-jv.json

An existing accepted same-device input can instead be supplied using
--initial-policy saved_accepted --initial-result PATH --initial-sha256 SHA.
The read subcommand needs only --result and --sha256 and never solves.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device, prepare_confined_dual_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_transient import IonicTransientInitial, load_ionic_steady_initial
from solarlab.experiments.jv.ionic_steady import (
    IonicSteadyJVOptions, IonicSteadyJVResult, read_ionic_steady_jv, run_ionic_steady_jv, write_ionic_steady_jv,
)


def compile_case(kind: str = "shared") -> CompiledIonicDevice1D:
    factories: dict[str, Callable[[], PreparedDevice]] = {"shared": prepare_confined_shared_site_ion_device,
                 "independent": prepare_confined_dual_ion_device, "positive": prepare_confined_ion_device}
    if kind not in factories:
        raise ValueError("kind must be shared, independent or positive")
    return compile_ionic_device_1d(factories[kind](), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))


def summary(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    points = []
    for attempt in result["attempts"]:
        solution = attempt["solution"]
        one = {"voltage_V": attempt["voltage_V"], "accepted": attempt["accepted"],
            "termination": attempt["termination"], "error": attempt["error"],
            "solver_called": attempt["solver_called"]}
        if solution is not None:
            one["checks"] = solution["checks"]
            one["evaluation_counts"] = solution["evaluation_counts"]
            one["newton_iterations"] = None if solution["newton"] is None else len(solution["newton"]["iterations"])
            if solution["state"] is not None and attempt["initial_guess"] is not None:
                one["ion_relative_change_from_guess"] = {species: float(np.max(abs(
                    solution["state"][species + "_ion_density_m3"]
                    / attempt["initial_guess"][species + "_ion_density_m3"] - 1.0))) for species in result["ion_species"]}
        points.append(one)
    return {"completed": record["completed"], "experiment_kind": result["experiment_kind"],
        "requested_voltage_V": result["requested_voltage_V"].tolist(), "points": points,
        "accepted_voltage_V": result["voltage_V"].tolist(),
        "output_current_A_m2": result["output_current_A_m2"].tolist(),
        "metrics": result["metrics"], "generation_scale": result["generation_scale"],
        "initial_policy": result["initial_provenance"]["policy"], "compiled_identity": result["compiled_identity"],
        "solver_calls": result["solver_calls"], "evaluation_counts": result["evaluation_counts"],
        "wall_s": result["wall_s"], "physical_certificate_status": result["physical_certificate_status"]}


def run_curve(
    device: CompiledIonicDevice1D, output: Path, *, voltages: tuple[float, ...] = (0.001, 0.002, 0.003),
    generation_scale: float = 0.0, initial: IonicTransientInitial | None = None,
) -> IonicSteadyJVResult:
    if output.exists():
        raise FileExistsError("preserve existing attempts; choose an unused result path")
    result = run_ionic_steady_jv(device, voltages, generation_scale=generation_scale,
        options=IonicSteadyJVOptions(steady_controls(device.ion_node_count)), initial=initial,
        initial_policy="compiled_cold" if initial is None else "saved_accepted")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_ionic_steady_jv(result, output)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        **summary(read_ionic_steady_jv(output, digest))}, indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--kind", choices=("shared", "independent", "positive"), default="shared")
    solve.add_argument("--voltages", type=float, nargs="+", required=True)
    solve.add_argument("--generation-scale", type=float, required=True)
    solve.add_argument("--initial-policy", choices=("compiled_cold", "saved_accepted"), required=True)
    solve.add_argument("--initial-result", type=Path)
    solve.add_argument("--initial-sha256")
    solve.add_argument("--output", type=Path, required=True)
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        print(json.dumps(summary(read_ionic_steady_jv(args.result, args.sha256)), indent=2, allow_nan=False))
        return 0
    if (args.initial_policy == "saved_accepted") != bool(args.initial_result and args.initial_sha256):
        parser.error("saved_accepted requires both initial-result and initial-sha256; compiled_cold requires neither")
    if args.initial_policy == "compiled_cold" and (args.initial_result or args.initial_sha256):
        parser.error("compiled_cold cannot accept an initial artifact")
    device = compile_case(args.kind)
    initial = (None if args.initial_policy == "compiled_cold" else load_ionic_steady_initial(
        device, args.initial_result, args.initial_sha256, controls=steady_controls(device.ion_node_count)))
    result = run_curve(device, args.output, voltages=tuple(args.voltages), generation_scale=args.generation_scale,
                       initial=initial)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

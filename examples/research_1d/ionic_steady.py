"""One real 13-node jointly coupled steady solve at 2 mV, with fixed ion inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions


def controls(node_count: int) -> IonicSteadyOptions:
    return IonicSteadyOptions(SteadyOptions(), 1e-12, 1e6, 1e6, 1.0, node_count - 1,
                               SolveBudget(60, 600, 60, 24), 30.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing attempts; choose a new output path")
    prepared = prepare_single_ion_device(diffusion_only=True)
    device = compile_ionic_device_1d(prepared, intervals_per_layer=12)
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                     options=controls(device.node_count))
    summary = {"accepted": result.accepted, "termination": result.termination,
               "node_count": device.node_count, "compiled_identity": device.identity,
               "voltage_V": 0.002, "generation_scale": 0.0, "wall_s": result.wall_s,
               "qualification": False, "error": result.error,
               "newton_iterations": None if result.newton is None else result.newton.iteration_count,
               "checks": result.checks}
    if result.state is not None:
        summary["ion_relative_redistribution"] = float(np.max(abs(
            result.state.positive_ion_density_m3 / device.positive_ion_reference_m3 - 1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value({"summary": summary, "result": result}),
                                     indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_value(summary), indent=2, allow_nan=False))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

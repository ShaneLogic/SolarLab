"""One cold coupled steady solve at 2 mV with fixed physical ion inventory.

The default is the original 13-node slab. --multilayer keeps all three
electrical layers (37 nodes) and confines 13 ion nodes to the absorber.
"""

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
    """The count is the number of ion-domain nodes, independent of electrical N."""
    return IonicSteadyOptions(SteadyOptions(), 1e-12, 1e6, 1e6, 1.0, node_count - 1,
                               SolveBudget(60, 600, 60, 24), 30.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multilayer", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing attempts; choose a new output path")
    if args.multilayer:
        from examples.research_1d.ionic_multilayer import prepare_confined_ion_device

        prepared = prepare_confined_ion_device(diffusion_only=True)
        device = compile_ionic_device_1d(prepared, intervals_per_layer=(12, 12, 12),
                                         ion_domain_layer="absorber",
                                         ion_domain_endpoints=("blocking", "blocking"))
    else:
        prepared = prepare_single_ion_device(diffusion_only=True)
        device = compile_ionic_device_1d(prepared, intervals_per_layer=12)
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                     options=controls(device.ion_node_count))
    summary = {"accepted": result.accepted, "termination": result.termination,
               "node_count": device.node_count, "compiled_identity": device.identity,
               "ion_node_count": device.ion_node_count, "state_size": device.state_size,
               "ion_node_indices": device.ion_node_indices, "ion_domain_bounds_m": device.domain_bounds_m,
               "initialization": "compiler cold state; no prior solve",
               "voltage_V": 0.002, "generation_scale": 0.0, "wall_s": result.wall_s,
               "qualification": False, "error": result.error,
               "newton_iterations": None if result.newton is None else result.newton.iteration_count,
               "evaluation_counts": result.evaluation_counts,
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

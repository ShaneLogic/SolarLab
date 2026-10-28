"""A small driven-voltage time history on the real 37-node new-core device.

At fixed light g=1, prescribe V(t)=0.1+0.001*(t/400 ns)^2. Each sample is a
backward-Euler endpoint; this is neither steady JV nor qualified hysteresis.
Use one BLAS thread and --refine for the bounded 4/8/16-step comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time import inventory_m2


def voltage_protocol(steps: int) -> TransientProtocol:
    times = np.linspace(0.0, 4e-7, steps + 1)
    return TransientProtocol(times, 0.1 + 0.001 * (times / times[-1]) ** 2, np.ones(steps + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing result; choose a new output path")
    started = time.monotonic()
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_controls = SteadyOptions()
    initial = solve_steady_state(device, voltage_V=0.1, generation_scale=1.0, options=initial_controls)
    results = []
    for steps in ((4, 8, 16) if args.refine else (4,)):
        result = solve_transient(device, initial, voltage_protocol(steps), options=controls(),
                                 initial_options=initial_controls)
        results.append(result)
        if not result.completed:
            break
    summary = {
        "experiment_kind": "voltage_driven_transient", "node_count": device.node_count,
        "compiled_identity": device.identity, "physical_certificate_status": "not_evaluated",
        "voltage_protocol": "V(t)=0.1+0.001*(t/4e-7 s)^2; BE right-endpoint samples",
        "generation_scale": 1.0, "completed": [r.completed for r in results],
        "accepted_steps": [len(r.states) - 1 for r in results],
        "compute_wall_s": time.monotonic() - started,
        "maxwell_spread_A_m2": [max((a.accounting["maxwell_spread_A_m2"] for a in r.attempts
                                     if a.accounting is not None), default=None) for r in results],
        "failures": [[{"termination": a.termination, "error": a.error,
                       "newton_message": None if a.newton is None else a.newton.message}
                      for a in r.failed_attempts] for r in results],
    }
    if len(results) == 3 and all(r.completed for r in results):
        inventories = [inventory_m2(device, r.states[-1]) for r in results]
        currents = [r.attempts[-1].accounting["terminal_total_current_density_A_m2"] for r in results]
        errors = [float(np.max(abs(a - b))) for a, b in zip(inventories[:-1], inventories[1:])]
        summary["successive_final_inventory_differences_m2"] = errors
        summary["inventory_difference_ratio"] = errors[0] / errors[1] if errors[1] else None
        summary["successive_final_terminal_current_differences_A_m2"] = [
            float(np.max(abs(a - b))) for a, b in zip(currents[:-1], currents[1:])]
        summary["final_terminal_current_density_A_m2"] = currents
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value({"summary": summary, "results": results}),
                                     indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_value(summary), indent=2, allow_nan=False))
    return 0 if all(r.completed for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

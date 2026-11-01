"""A 37-node, fixed-voltage light ramp using conservative physical inventories.

Run with BLAS thread controls set to 1 and optionally --refine for 4/8/16
explicit steps over the same interval. This example calls the real new core;
it performs no G2/native/legacy solve and grants no physical certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.experiments.transient import TransientOptions, TransientProtocol, solve_transient
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time import inventory_m2


def controls() -> TransientOptions:
    return TransientOptions(1e-11, 1e-8, 1e-11, 0.1, 2.0,
                            SolveBudget(50, 1000, 50, 24), 32, 30.0, "previous", POISSON_CONTACT_BALANCE_V1)


def json_value(value):
    """Preserve nonfinite failure values explicitly in otherwise strict JSON."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("the output already exists; preserve the original attempt")
    started = time.monotonic()
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_controls = SteadyOptions()
    # This is an explicit accepted illuminated DC input, outside the time
    # stepper; its full solve/preparation evidence is retained in each result.
    initial = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0, options=initial_controls)
    results = []
    for steps in ((4, 8, 16) if args.refine else (4,)):
        times = np.linspace(0, 4e-7, steps + 1)
        # Explicit right-endpoint samples of g(t)=1+0.01*(t/400 ns)^2.
        # These are not inferred event times or hidden solver substeps.
        protocol = TransientProtocol(times, np.zeros(steps + 1), 1.0 + 0.01 * (times / times[-1]) ** 2)
        result = solve_transient(device, initial, protocol, options=controls(), initial_options=initial_controls)
        results.append(result)
        if not result.completed:
            break
    summary = {"node_count": device.node_count, "compiled_identity": device.identity,
               "completed": [result.completed for result in results],
               "accepted_steps": [len(result.states) - 1 for result in results],
               "generation_protocol": "g(t)=1+0.01*(t/4e-7 s)^2; BE right-endpoint samples",
               "wall_s": time.monotonic() - started, "physical_certificate_status": "not_evaluated"}
    if len(results) == 3 and all(result.completed for result in results):
        final = [inventory_m2(device, result.states[-1]) for result in results]
        errors = [np.max(np.abs(a - b)) for a, b in zip(final[:-1], final[1:])]
        summary["successive_final_cell_inventory_differences_m2"] = errors
        summary["difference_ratio"] = float(errors[0] / errors[1]) if errors[1] != 0 else None
    summary["maxwell_spread_A_m2"] = [max((a.accounting["maxwell_spread_A_m2"]
        for a in result.attempts if a.accounting is not None), default=None) for result in results]
    summary["failures"] = [[{"termination": a.termination, "error": a.error,
                              "newton_message": None if a.newton is None else a.newton.message}
                             for a in result.failed_attempts] for result in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value({"summary": summary, "results": results}),
                                     indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_value(summary), indent=2, allow_nan=False))
    return 0 if all(result.completed for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

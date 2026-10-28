"""A pinned 37/13-node ionic steady input driven from 2 to 4 mV over one second.

--refine runs independent 4/8/16 requested grids once each, stopping after the
first partial history. No initial steady solve is repeated. Raw accounting
and first failures remain in the output; physical status is not_evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.experiments.ionic_transient import (
    IonicTransientOptions, load_ionic_steady_initial, solve_ionic_transient,
)
from solarlab.experiments.transient import TransientOptions, TransientProtocol
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1


def controls() -> IonicTransientOptions:
    electronic = TransientOptions(1e-11, 1e-8, 1e-11, 0.1, 2.0,
        SolveBudget(50, 500, 50, 24), 16, 30.0, "previous", POISSON_CONTACT_BALANCE_V1)
    return IonicTransientOptions(electronic, 1e6, 1.0)


def protocol(steps: int) -> TransientProtocol:
    times = np.linspace(0.0, 1.0, steps + 1)
    return TransientProtocol(times, 0.002 + 0.002 * times, np.zeros(steps + 1))


def history_summary(device, result) -> dict[str, object]:
    widths = device.ion_geometry.cell_measures_m
    initial = result.states[0].positive_ion_density_m3
    initial_total = math.fsum((widths * initial).tolist())
    inventories = [math.fsum((widths * state.positive_ion_density_m3).tolist()) for state in result.states]
    measured = [attempt for attempt in result.attempts if attempt.accounting is not None]
    residual_bound = math.fsum((a.end_s - a.start_s) * math.fsum(abs(a.accounting["ion_balance_m2_s"]).tolist())
                               for a in measured if a.accepted)
    readback_bound = math.fsum((a.end_s - a.start_s)
        * math.fsum(abs(a.accounting["ion_inventory_readback_error_m2_s"]).tolist())
        for a in measured if a.accepted)
    return {
        "requested_steps": result.protocol.times_s.size - 1,
        "accepted_steps": len(result.states) - 1,
        "completed": result.completed,
        "accepted_times_s": result.accepted_times_s,
        "max_ion_relative_redistribution": max(float(np.max(abs(state.positive_ion_density_m3 / initial - 1)))
                                               for state in result.states),
        "ion_inventory_m2": inventories,
        "max_inventory_drift_m2": max(abs(value - initial_total) for value in inventories),
        "integrated_ion_residual_l1_m2": residual_bound,
        "integrated_inventory_readback_l1_m2": readback_bound,
        "max_ion_flux_m2_s": max((float(np.max(abs(a.accounting["ion_domain_fluxes_m2_s"])))
                                   for a in measured), default=None),
        "maxwell_spread_A_m2": max((a.accounting["maxwell_spread_A_m2"] for a in measured), default=None),
        "maxwell_identity_error_A_m2": max((float(np.max(abs(a.accounting["maxwell_identity_error_A_m2"])))
                                             for a in measured), default=None),
        "readback_maxwell_identity_error_A_m2": max((float(np.max(abs(a.accounting["readback_maxwell_identity_error_A_m2"])))
                                                      for a in measured), default=None),
        "max_ion_BE_l1_m2_s": max((math.fsum(abs(a.accounting["ion_balance_m2_s"]).tolist())
                                    for a in measured), default=None),
        "newton_iterations": [None if a.newton is None else a.newton.iteration_count for a in result.attempts],
        "evaluation_counts": result.evaluation_counts,
        "wall_s": result.wall_s,
        "failures": [{"start_s": a.start_s, "end_s": a.end_s, "termination": a.termination,
                      "error": a.error, "newton_message": None if a.newton is None else a.newton.message}
                     for a in result.failed_attempts],
        "physical_certificate_status": result.physical_certificate_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-result", type=Path, required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    if args.output.exists() or temporary.exists():
        raise FileExistsError("preserve the existing history; choose a new output path")
    started = time.monotonic()
    device = compile_ionic_device_1d(prepare_confined_ion_device(diffusion_only=True),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))
    initial = load_ionic_steady_initial(device, args.initial_result, args.initial_sha256,
                                        controls=steady_controls(device.ion_node_count))
    results, summaries = [], []
    requested = (4, 8, 16) if args.refine else (4,)
    summary = {"compiled_identity": device.identity, "electrical_nodes": device.node_count,
               "ion_nodes": device.ion_node_count, "state_size": device.state_size,
               "initial_source_path": initial.source_path, "initial_source_sha256": initial.source_sha256,
               "voltage_protocol": "V(t)=0.002+0.002*t V, 0<=t<=1 s; explicit BE right endpoints",
               "generation_protocol": "g(t)=0", "requested_step_counts": requested,
               "histories": summaries, "physical_certificate_status": "not_evaluated"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for steps in requested:
        result = solve_ionic_transient(device, initial, protocol(steps), options=controls())
        results.append(result)
        summaries.append(history_summary(device, result))
        if len(results) == 3 and all(item.completed for item in results):
            final = [device.ion_geometry.cell_measures_m * item.states[-1].positive_ion_density_m3 for item in results]
            differences = [float(np.max(abs(a - b))) for a, b in zip(final[:-1], final[1:])]
            signal = float(np.max(abs(final[-1] - device.ion_geometry.cell_measures_m * initial.state.positive_ion_density_m3)))
            summary["successive_final_ion_cell_inventory_differences_m2"] = differences
            summary["difference_ratio"] = differences[0] / differences[1] if differences[1] else None
            summary["final_ion_cell_inventory_motion_m2"] = signal
            summary["difference_to_motion_ratios"] = [value / signal for value in differences] if signal else None
        summary["wall_s"] = time.monotonic() - started
        try:
            temporary.write_text(json.dumps(json_value({"summary": summary, "results": results}),
                                              indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps(json_value(summaries[-1]), indent=2, allow_nan=False), flush=True)
        if not result.completed:
            break
    return 0 if len(results) == len(requested) and all(item.completed for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

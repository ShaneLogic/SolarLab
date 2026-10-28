"""One cold 37/13 dual-ion steady solve and a four-step dark voltage ramp.

Both populations have independent finite sites and declared blocking-domain
inventories. The driven result is a dynamic history; no hysteresis or physical
qualification is claimed. Existing evidence is never overwritten or retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls as time_controls, protocol
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_transient import (
    IonicTransientResult, load_ionic_steady_initial, solve_ionic_transient,
)
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyResult, solve_ionic_steady_state


def compile_case() -> CompiledIonicDevice1D:
    return compile_ionic_device_1d(prepare_confined_dual_ion_device(diffusion_only=True),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))


def write_record(path: Path, value: object) -> str:
    """A complete write with exclusive creation preserves any first failure."""
    raw = (json.dumps(json_value(value), indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as output:
        output.write(raw)
    return hashlib.sha256(raw).hexdigest()


def history_summary(device: CompiledIonicDevice1D, result: IonicTransientResult) -> dict[str, object]:
    measured = [attempt for attempt in result.attempts if attempt.accounting is not None]
    species_info = {}
    for species in device.ion_species_names:
        density_name = f"{species}_ion_density_m3"
        initial = getattr(result.states[0], density_name)
        inventories = [state.domain_inventories_m2[species] for state in result.states]
        accepted = [a for a in measured if a.accepted]
        species_info[species] = {
            "max_relative_redistribution": max(float(np.max(abs(getattr(state, density_name) / initial - 1)))
                                               for state in result.states),
            "inventories_m2": inventories,
            "max_inventory_drift_m2": max(abs(value - inventories[0]) for value in inventories),
            "integrated_BE_residual_l1_m2": math.fsum((a.end_s - a.start_s)
                * math.fsum(abs(a.accounting["ion_species"][species]["ion_balance_m2_s"]).tolist()) for a in accepted),
            "integrated_inventory_readback_l1_m2": math.fsum((a.end_s - a.start_s)
                * math.fsum(abs(a.accounting["ion_species"][species]["ion_inventory_readback_error_m2_s"]).tolist())
                for a in accepted),
            "max_BE_residual_l1_m2_s": max((math.fsum(abs(a.accounting["ion_species"][species]["ion_balance_m2_s"]).tolist())
                                           for a in measured), default=None),
            "max_motion_flux_m2_s": max((float(np.max(abs(a.accounting["ion_species"][species]["ion_domain_fluxes_m2_s"])))
                                         for a in measured), default=None),
        }
    return {
        "completed": result.completed, "accepted_steps": len(result.states) - 1,
        "accepted_times_s": result.accepted_times_s, "ion_species": species_info,
        "maxwell_spread_A_m2": max((a.accounting["maxwell_spread_A_m2"] for a in measured), default=None),
        "maxwell_identity_error_A_m2": max((float(np.max(abs(a.accounting["maxwell_identity_error_A_m2"])))
                                            for a in measured), default=None),
        "newton_iterations": [None if a.newton is None else a.newton.iteration_count for a in result.attempts],
        "evaluation_counts": result.evaluation_counts, "wall_s": result.wall_s,
        "first_failure": None if not result.failed_attempts else {
            "start_s": result.failed_attempts[0].start_s, "end_s": result.failed_attempts[0].end_s,
            "termination": result.failed_attempts[0].termination, "error": result.failed_attempts[0].error,
        },
        "physical_certificate_status": result.physical_certificate_status,
    }


def run_wave(output_directory: Path) -> tuple[CompiledIonicDevice1D, IonicSteadyResult, IonicTransientResult | None]:
    """Run each requested operation once and persist the steady result before time."""
    output_directory.mkdir(parents=True, exist_ok=True)
    steady_path, history_path = output_directory / "Steady.json", output_directory / "History.json"
    if steady_path.exists() or history_path.exists():
        raise FileExistsError("preserve existing attempts; choose an unused output directory")
    device = compile_case()
    options = steady_controls(device.ion_node_count)
    steady = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options)
    summary = {
        "accepted": steady.accepted, "termination": steady.termination,
        "compiled_identity": device.identity, "electrical_nodes": device.node_count,
        "ion_nodes_per_species": device.ion_node_count, "ion_species": device.ion_species_names,
        "state_size": device.state_size, "voltage_V": 0.002, "generation_scale": 0.0,
        "initialization": "compiler cold state; no prior solve", "checks": steady.checks,
        "newton_iterations": None if steady.newton is None else steady.newton.iteration_count,
        "evaluation_counts": steady.evaluation_counts, "wall_s": steady.wall_s,
        "error": steady.error, "qualification": False,
    }
    if steady.state is not None:
        summary["ion_relative_redistribution"] = {species: float(np.max(abs(
            getattr(steady.state, f"{species}_ion_density_m3") / getattr(device, f"{species}_ion_reference_m3") - 1)))
            for species in device.ion_species_names}
    digest = write_record(steady_path, {"summary": summary, "result": steady})
    print(json.dumps(json_value(summary), indent=2, allow_nan=False), flush=True)
    if not steady.accepted:
        return device, steady, None
    initial = load_ionic_steady_initial(device, steady_path, digest, controls=options)
    history = solve_ionic_transient(device, initial, protocol(4), options=time_controls())
    observed = history_summary(device, history)
    write_record(history_path, {"summary": observed, "result": history})
    print(json.dumps(json_value(observed), indent=2, allow_nan=False), flush=True)
    return device, steady, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    _, steady, history = run_wave(args.output_directory)
    return 0 if steady.accepted and history is not None and history.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""One 0.8-occupancy shared-site cold solve and four dark voltage-ramp steps.

The common-capacity populations are solved together on a blocking absorber.
Requested endpoints, failures, inventories and raw Maxwell accounting are
retained. This bounded research example makes no physical qualification claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.research_1d.dual_ionic import history_summary, write_record
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls as time_controls, protocol
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.ionic_transient import (
    IonicTransientResult, load_ionic_steady_initial, solve_ionic_transient,
)
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyResult, solve_ionic_steady_state


def compile_case() -> CompiledIonicDevice1D:
    """Core's declared P0=1e24, N0=6e23, common capacity=2e24 fixture."""
    return compile_ionic_device_1d(prepare_confined_shared_site_ion_device(),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))


def occupancy_summary(device: CompiledIonicDevice1D, states: tuple[IonicState1D, ...]) -> dict[str, object]:
    occupied = [(state.positive_ion_density_m3 + state.negative_ion_density_m3)
                 / device.positive_ion_site_limit_m3 for state in states]
    return {
        "scope": "reported initial/candidate steady states or accepted time states",
        "maximum_summed_site_fraction": max(float(np.max(value)) for value in occupied),
        "minimum_vacancy_fraction": min(float(np.min(1.0 - value)) for value in occupied),
        "maximum_summed_site_fraction_by_state": [float(np.max(value)) for value in occupied],
        "minimum_vacancy_fraction_by_state": [float(np.min(1.0 - value)) for value in occupied],
    }


def run_wave(output_directory: Path) -> tuple[CompiledIonicDevice1D, IonicSteadyResult, IonicTransientResult | None]:
    """One bounded cold solve, then one pinned four-step history only on acceptance."""
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
        "ion_site_contract": device.ion_site_contract, "steric_model": device.steric_model,
        "positive_reference_m3": device.positive_ion_reference_m3,
        "negative_reference_m3": device.negative_ion_reference_m3,
        "common_site_capacity_m3": device.positive_ion_site_limit_m3,
        "state_size": device.state_size, "voltage_V": 0.002, "generation_scale": 0.0,
        "initialization": "compiler cold state; no prior solve", "checks": steady.checks,
        "newton_iterations": None if steady.newton is None else steady.newton.iteration_count,
        "evaluation_counts": steady.evaluation_counts, "wall_s": steady.wall_s,
        "error": steady.error, "qualification": False,
    }
    states = (steady.initial_state,) if steady.state is None else (steady.initial_state, steady.state)
    summary["shared_site"] = occupancy_summary(device, states)
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
    observed["shared_site"] = occupancy_summary(device, history.states)
    write_record(history_path, {"summary": observed, "result": history})
    print(json.dumps(json_value(observed), indent=2, allow_nan=False), flush=True)
    return device, steady, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    _, steady, history = run_wave(arguments.output_directory)
    return 0 if steady.accepted and history is not None and history.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

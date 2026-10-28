"""Explicit compiled-reference fixtures for the one-domain immobile-ion slice."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, cast

import numpy as np

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice


def prepare_immobile_ion_device(kind: str = "shared_mixed") -> PreparedDevice:
    """Declared P0/P0_neg are physical targets; no warm-state argument is accepted."""
    if kind in {"shared_mixed", "shared_full"}:
        return prepare_confined_shared_site_ion_device(
            positive_diffusivity_m2_s=1e-16 if kind == "shared_mixed" else 0.0,
            negative_diffusivity_m2_s=0.0,
        )
    if kind == "single":
        prepared = prepare_single_ion_device(diffusion_only=True)
        data = prepared.to_input().editing_data()
        data["layers"][0]["parameters"]["D_ion"] = 0.0
        return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    raise ValueError("immobile fixture kind must be shared_mixed, shared_full or single")


def _complex_json(value: object) -> dict[str, float]:
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError("unsupported immobile witness JSON value")


def run_wave(kind: str, output_directory: Path) -> dict[str, object]:
    """One bounded steady, four native time steps and two AC points; no retries."""
    from examples.research_1d.dual_ionic import write_record
    from examples.research_1d.ionic_steady import controls as steady_controls
    from examples.research_1d.ionic_transient import controls as time_controls, protocol
    from examples.research_1d.transient import json_value
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
    from solarlab.equations.ionic_closure import declared_immobile_profiles, immobile_profile_binding
    from solarlab.equations.ionic_small_signal import build_ionic_small_signal
    from solarlab.experiments.ionic_transient import load_ionic_steady_initial, solve_ionic_transient
    from solarlab.solvers.linear import FrequencyOptions, solve_frequency_systems
    from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state

    started = time.monotonic()
    deadline = started + 30.0

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0.0:
            raise TimeoutError("immobile witness total30s budget exhausted")
        return value

    output_directory.mkdir(parents=True, exist_ok=True)
    if any((output_directory / name).exists() for name in ("Steady.json", "History.json", "AC.json")):
        raise FileExistsError("preserve original immobile attempts; use an unused output directory")
    prepared = prepare_immobile_ion_device(kind)
    if kind == "single":
        device = compile_ionic_device_1d(prepared, intervals_per_layer=4)
    else:
        device = compile_ionic_device_1d(prepared, intervals_per_layer=(4, 4, 4),
                                         ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    options = replace(steady_controls(device.ion_node_count), wall_s=min(30.0, remaining()))
    steady = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options)
    summary: dict[str, object] = {"kind": kind, "accepted": steady.accepted, "termination": steady.termination,
        "compiled_identity": device.identity, "voltage_V": 0.002, "generation_scale": 0.0,
        "shape": [device.node_count, device.ion_node_count, device.state_size],
        "profile_binding": immobile_profile_binding(device), "evaluation_counts": steady.evaluation_counts,
        "wall_s": steady.wall_s, "qualification": False}
    path = output_directory / "Steady.json"
    digest = write_record(path, {"summary": summary, "result": steady})
    if not steady.accepted or steady.state is None:
        return {**summary, "completed": False, "first_failure": "steady"}
    remaining()
    initial = load_ionic_steady_initial(device, path, digest, controls=options)
    time_options = time_controls()
    time_options = replace(time_options, electronic=replace(time_options.electronic, wall_s=min(30.0, remaining())))
    history = solve_ionic_transient(device, initial, protocol(4), options=time_options)
    profiles = declared_immobile_profiles(device)
    profile_error = {species: max(float(np.max(abs(getattr(state, species + "_ion_density_m3") - target) / target))
                                  for state in history.states) for species, target in profiles.items()}
    time_summary = {"completed": history.completed, "accepted_steps": len(history.states) - 1,
                    "immobile_profile_relative_error": profile_error,
                    "evaluation_counts": history.evaluation_counts, "wall_s": history.wall_s}
    write_record(output_directory / "History.json", {"summary": time_summary, "result": history})
    if not history.completed:
        return {**summary, "completed": False, "first_failure": "time", "time": time_summary}
    remaining()
    operator = build_ionic_small_signal(device, steady.state)
    voltage = 2e-6 + 1e-6j
    attempts = []
    for frequency in (0.0, 1.0):
        remaining()
        system = operator.system(frequency, voltage_phasor_V=voltage, ion_mode="mobile")
        solved = solve_frequency_systems(system.jacobian, system.storage_jacobian, system.rhs, [frequency],
                                         options=FrequencyOptions(), fixed_coordinates=system.fixed_coordinates, rhs_rate=system.rhs_rate)[0]
        observed = cast(dict[str, Any], operator.observe(solved.solution, frequency, voltage_phasor_V=voltage, ion_mode="mobile"))
        local = {species: float(np.max(abs(observed["ion_density_perturbation_m3"][species]) / target)) / abs(voltage)
                 for species, target in profiles.items()}
        global_inventory = {species: abs(observed["ion_inventory_perturbation_m2"][species]) /
                            (steady.state.domain_inventories_m2[species] * abs(voltage)) for species in device.ion_species_names}
        accepted = (observed["original_componentwise_backward_error"] <= 1e-11
                    and observed["maxwell_spread_A_m2_V"] <= 1e-8 + 1e-10 * observed["current_scale_A_m2_V"]
                    and max(local.values(), default=0.0) <= options.inventory_relative_tolerance
                    and max(global_inventory.values(), default=0.0) <= options.inventory_relative_tolerance)
        attempts.append({"frequency_Hz": frequency, "accepted": accepted, "local_profile_relative_per_V": local,
                         "inventory_relative_per_V": global_inventory, "solution": solved, "observation": observed})
        if not accepted:
            break
    ac = {"completed": len(attempts) == 2 and all(attempt["accepted"] for attempt in attempts), "attempts": attempts}
    raw = (json.dumps(json_value(ac), default=_complex_json, indent=2, allow_nan=False) + "\n").encode()
    with (output_directory / "AC.json").open("xb") as output:
        output.write(raw)
    remaining()
    return {**summary, "completed": ac["completed"], "first_failure": None if ac["completed"] else "AC",
            "time": time_summary, "total_wall_s": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("shared_mixed", "shared_full", "single"), default="shared_mixed")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    summary = run_wave(args.kind, args.output_directory)
    print(json.dumps(summary, indent=2, default=_complex_json, allow_nan=False))
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

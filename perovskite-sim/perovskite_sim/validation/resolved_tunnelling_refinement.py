"""Versioned numerical evidence for one conservative intraband electron path."""

from dataclasses import asdict, replace
import math
from pathlib import Path

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import solve_quasi_fermi_steady_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.tunneling_channel_device import (
    TUNNELLING_CHANNEL_DEVICE_VERSION,
    physical_quasi_fermi_levels_eV,
)
from perovskite_sim.solver.mol import build_material_arrays
from .dae_refinement import _finite_option, _integer_option, _protocol_metadata
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement
from .tunnelling_channel_refinement import _quadrature_orders, _state_sha256, _with_order


def resolved_tunnelling_protocol(lane: LaneDefinition) -> dict:
    options = lane.options
    return {
        "schema_version": "solarlab-resolved-intraband-refinement-v2",
        "lane_id": lane.lane_id, "executor_version": lane.executor_version,
        "device_implementation": TUNNELLING_CHANNEL_DEVICE_VERSION,
        "channel": "intraband_electron", "barrier": "one isolated material spike",
        "grid_values": list(lane.grid_values), "grid_alpha": _finite_option(options, "grid_alpha", 5.0),
        "tolerance_factors": list(lane.tolerance_factors),
        "bias_V": _finite_option(options, "bias_V", 0.02),
        "illuminated": False, "temperature_K": 300.0,
        "energy_quadrature_orders": list(_quadrature_orders(options)),
        "energy_rule": "open_gauss_legendre", "action_rule": "piecewise_linear_exact_turning_points",
        "particle_transfer": "turning_point_nodal_weights",
        "profile_points_per_layer": _integer_option(options, "profile_points_per_layer", 17, minimum=3),
        "spectrum_energy_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
        "supply_prefactor_m2_s_eV": 1e24,
    }


def _relative_ladder_change(values):
    if any(not np.isfinite(value) or value == 0.0 for value in values):
        raise ValueError("the resolved reference requires a finite nonzero signal on every energy rung")
    return max(abs(right - left) / abs(right) for left, right in zip(values[:-1], values[1:]))


def run_resolved_intraband_refinement(
    lane: LaneDefinition, point: MatrixPoint, project_root: Path,
) -> CellMeasurement:
    """Measure one spatial/tolerance cell, retaining all three energy rungs."""
    options = lane.options
    if options.get("config_loader", "standard") != "standard":
        raise ValueError("resolved tunnelling requires config_loader='standard'")
    protocol = resolved_tunnelling_protocol(lane)
    orders = _quadrature_orders(options)
    bias = float(protocol["bias_V"])
    if bias <= 0.0:
        raise ValueError("the dark conductance reference requires positive bias_V")
    stack = load_device_from_yaml(project_root / lane.config_path)
    if stack.T != 300.0:
        raise ValueError("the v2 material-barrier reference is specified at 300 K")
    if (stack.tunnelling_channels is None or stack.tunnelling_channels.enabled_channels != ("intraband",)
            or stack.tunnelling_channels.intraband.carrier != "electron"):
        raise ValueError("the v2 reference requires exactly one intraband electron channel")
    layers = electrical_layers(stack)
    if len(layers) != 3:
        raise ValueError("the v2 reference requires two reservoirs and one intervening material barrier")
    grid = multilayer_grid([Layer(layer.thickness, point.grid) for layer in layers],
                           alpha=float(protocol["grid_alpha"]))
    material = build_material_arrays(grid, stack)
    controls = {
        "newton_residual_tolerance": _finite_option(options, "base_newton_residual_tolerance", 1e-8) * point.tolerance_factor,
        "poisson_tolerance_V": _finite_option(options, "base_poisson_tolerance_V", 1e-10) * point.tolerance_factor,
        "finite_difference_step": _finite_option(options, "base_finite_difference_step", 1e-5) * math.sqrt(point.tolerance_factor),
        "continuity_tolerance_A_m2": _finite_option(options, "continuity_tolerance_A_m2", 1e-4),
        "current_spread_tolerance_A_m2": _finite_option(options, "current_spread_tolerance_A_m2", 1e-4),
        "max_newton_iterations": _integer_option(options, "max_newton_iterations", 30),
        "require_contact_certificate": True,
    }

    def solve(device, voltage):
        return solve_quasi_fermi_steady_state(grid, device, V_app=voltage, illuminated=False, **controls)

    disabled = solve(_with_order(stack, orders[-1], enabled=False), bias)
    bare = solve(replace(stack, tunnelling_channels=None), bias)
    results = [solve(_with_order(stack, order), bias) for order in orders]
    finest = results[-1]
    equilibrium = solve(_with_order(stack, orders[-1]), 0.0)
    reverse = solve(_with_order(stack, orders[-1]), -bias)
    diagnostics = finest.tunnelling_channel_diagnostics
    if diagnostics is None or diagnostics.intraband_path is None:
        raise ValueError("resolved path diagnostics are missing")
    path = diagnostics.intraband_path
    if path.low_action_supply_fraction is None:
        raise ValueError("empty reservoir supply cannot certify the WKB approximation range")
    paths = [result.tunnelling_channel_diagnostics.intraband_path for result in results]
    if any(value is None for value in paths):
        raise ValueError("every energy rung must report the same resolved path")
    fluxes = [value.flux.net_flux_m2_s for value in paths]
    effects = [result.current_A_m2 - disabled.current_A_m2 for result in results]
    if disabled.current_A_m2 == 0.0:
        raise ValueError("zero reference terminal current cannot define the channel effect")
    transfer = np.diff(np.r_[0.0, diagnostics.electron_face_current_A_m2, 0.0]) / Q
    transfer_scale = float(np.sum(np.abs(transfer)))
    if transfer_scale == 0.0:
        raise ValueError("the resolved channel must transfer a nonzero number of particles")
    electron_level, hole_level = physical_quasi_fermi_levels_eV(
        potential_V=finest.phi, affinity_eV=material.chi_phys, band_gap_eV=material.Eg_phys,
        electron_density_m3=finest.y[:grid.size], hole_density_m3=finest.y[grid.size:2 * grid.size],
        conduction_dos_m3=material.N_C_physical, valence_dos_m3=material.N_V_physical,
        thermal_voltage_V=material.V_T_device,
    )
    positions = []
    offset = 0.0
    profile_count = int(protocol["profile_points_per_layer"])
    for layer in layers:
        positions.extend(offset + (np.arange(profile_count) + 0.5) * layer.thickness / profile_count)
        offset += layer.thickness
    positions = np.asarray(positions)
    fractions = np.asarray(protocol["spectrum_energy_fractions"])
    energies = path.flux.energies_eV
    # Recover the actual Gauss interval from its known rule, not from its
    # interior first/last abscissae, which change with quadrature order.
    gauss_nodes, _ = np.polynomial.legendre.leggauss(orders[-1])
    energy_width = 2 * (energies[-1] - energies[0]) / (gauss_nodes[-1] - gauss_nodes[0])
    energy_base = energies[0] - energy_width * (gauss_nodes[0] + 1) / 2
    normalized_energy = (energies - energy_base) / energy_width
    observed = {
        "dark_terminal_current_A_m2": finest.current_A_m2,
        "disabled_terminal_current_A_m2": disabled.current_A_m2,
        "terminal_tunnelling_effect_A_m2": effects[-1],
        "intraband_event_flux_m2_s": fluxes[-1],
        "maximum_local_tunnelling_current_A_m2": float(np.max(np.abs(diagnostics.electron_face_current_A_m2))),
        "barrier_base_eV": float(energy_base), "barrier_peak_eV": float(energy_base + energy_width),
        "action_at_energy_fractions": np.interp(fractions, normalized_energy, path.actions),
        "potential_profile_V": np.interp(positions, grid, finest.phi),
        "electron_log_density_profile": np.interp(positions, grid, np.log(finest.y[:grid.size])),
        "physical_electron_fermi_profile_eV": np.interp(positions, grid, electron_level),
    }
    all_states = [*results, disabled, bare, equilibrium, reverse]
    current_scale = float(np.max(np.abs(diagnostics.electron_face_current_A_m2)))
    quality = {
        "all_states_certified": float(all(result.certified for result in all_states)),
        "contact_thermodynamics_consistent": float(all(result.contact_thermodynamic_status == "certified" for result in all_states)),
        "energy_orders_completed": float(len(orders)),
        "max_energy_event_flux_relative_change": _relative_ladder_change(fluxes),
        "max_energy_terminal_effect_relative_change": _relative_ladder_change(effects),
        "terminal_effect_fraction": abs(effects[-1] / disabled.current_A_m2),
        "equilibrium_terminal_current_A_m2": abs(equilibrium.current_A_m2),
        "equilibrium_tunnelling_current_A_m2": float(np.max(np.abs(equilibrium.tunnelling_channel_diagnostics.electron_face_current_A_m2))),
        "particle_balance_relative_error": abs(float(np.sum(transfer))) / transfer_scale,
        "path_current_relative_error": float(np.max(np.abs(diagnostics.electron_face_current_A_m2 + Q * path.particle_face_flux_m2_s))) / current_scale,
        "external_tunnelling_face_current_A_m2": float(np.max(np.abs(diagnostics.electron_face_current_A_m2[[0, -1]]))),
        "dissipative_transfer": float(np.dot(electron_level, transfer) <= 0.0),
        "dark_electrical_power_consumption": float(finest.current_A_m2 * bias < 0.0 and reverse.current_A_m2 * (-bias) < 0.0),
        "injected_face_count": float(np.count_nonzero(diagnostics.electron_face_current_A_m2)),
        "disabled_bit_identical": float(disabled.tunnelling_channel_diagnostics is None
                                         and np.array_equal(disabled.y, bare.y)
                                         and np.array_equal(disabled.phi, bare.phi)
                                         and disabled.current_A_m2 == bare.current_A_m2),
        "maximum_reservoir_occupation": float(max(np.max(path.flux.left_occupation), np.max(path.flux.right_occupation))),
        "minimum_reservoir_occupation": float(min(np.min(path.flux.left_occupation), np.min(path.flux.right_occupation))),
        "low_action_supply_fraction": path.low_action_supply_fraction,
        "maximum_electron_density_over_dos": float(np.max(finest.y[:grid.size] / material.N_C_physical)),
        "maximum_carrier_continuity_bound_A_m2": max(max(result.electron_continuity_bound_A_m2, result.hole_continuity_bound_A_m2) for result in all_states),
        "maximum_current_spread_A_m2": max(result.face_current_spread_A_m2 for result in all_states),
        "maximum_poisson_residual_C_m2": max(result.poisson_residual_C_m2 for result in all_states),
        "residual_over_solver_limit": max(result.max_normalized_cell_residual / result.numerical_residual_limit for result in all_states),
    }
    units = {name: "1" for name in [*observed, *quality]}
    for name in units:
        if name.endswith("A_m2"):
            units[name] = "A m-2"
        elif name.endswith("m2_s"):
            units[name] = "m-2 s-1"
        elif name.endswith("C_m2"):
            units[name] = "C m-2"
        elif name.endswith("eV"):
            units[name] = "eV"
        elif name.endswith("_V"):
            units[name] = "V"
    return CellMeasurement.from_mapping({
        "observables": observed, "quality": quality, "units": units,
        "metadata": {
            **_protocol_metadata(protocol),
            "actual": {
                "solve_controls": controls, "grid_nodes": grid.size,
                "grid_intervals_per_layer": point.grid,
                "barrier": asdict(path.region),
                "channel_document_sha256": diagnostics.identity_sha256,
                "energy_ladder_event_flux_m2_s": fluxes,
                "energy_ladder_terminal_effect_A_m2": effects,
                "state_sha256": {"biased": _state_sha256(finest.y, finest.phi),
                                  "equilibrium": _state_sha256(equilibrium.y, equilibrium.phi)},
                "raw": {
                    "positions_m": grid.tolist(), "density_state_m3": finest.y.tolist(),
                    "potential_V": finest.phi.tolist(),
                    "physical_electron_fermi_eV": electron_level.tolist(),
                    "physical_hole_fermi_eV": hole_level.tolist(),
                    "total_current_A_m2": finest.total_face_current_A_m2.tolist(),
                    "tunnelling_current_A_m2": diagnostics.electron_face_current_A_m2.tolist(),
                    "particle_transfer_m2_s": transfer.tolist(),
                    "energy_eV": energies.tolist(), "action": path.actions.tolist(),
                    "transmission": path.flux.transmission.tolist(),
                    "left_occupation": path.flux.left_occupation.tolist(),
                    "right_occupation": path.flux.right_occupation.tolist(),
                    "quadrature_weights_eV": path.flux.quadrature_weights_eV.tolist(),
                    "turning_points_m": path.turning_points_m.tolist(),
                    "profile_positions_m": positions.tolist(),
                },
            },
        },
    })

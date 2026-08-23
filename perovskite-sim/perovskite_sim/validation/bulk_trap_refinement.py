"""Registered P4.3 charged, energy-distributed bulk-trap refinement lane."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.bulk_traps import evaluate_bulk_trap_state
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.solver.bulk_trap_equilibrium import (
    BulkTrapPNEquilibriumResult,
    solve_bulk_trap_pn_equilibrium,
)
from perovskite_sim.solver.mol import (
    BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    BulkTrapChargeCapabilityError,
    build_material_arrays,
)

from .dae_refinement import (
    _finite_option,
    _integer_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def _quadrature_orders(options: dict[str, Any]) -> tuple[int, ...]:
    raw = options.get("energy_quadrature_orders", [16, 32, 64])
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("energy_quadrature_orders must be a list of at least two")
    orders: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("energy_quadrature_orders must contain integers >= 2")
        orders.append(value)
    if any(right != 2 * left for left, right in zip(orders, orders[1:])):
        raise ValueError("energy_quadrature_orders must be consecutive doublings")
    return tuple(orders)


def _execution_protocol(
    lane: LaneDefinition,
    *,
    quadrature_orders: tuple[int, ...],
    base_poisson_tolerance: float,
    max_newton_iterations: int,
    max_potential_step_V: float,
    max_line_search_backtracks: int,
    probe_electron_density_m3: float,
    probe_hole_density_m3: float,
    left_layer_name: str,
    right_layer_name: str,
) -> dict[str, object]:
    return {
        "constitutive_closure": {
            "carrier_statistics": "maxwell_boltzmann",
            "charge_transition": "explicit_donor_or_acceptor_neutral_reference",
            "dopant_ionization": "fully_ionized",
            "energy_distribution": "truncated_gaussian_total_density",
            "occupancy": "steady_capture_emission_balance",
            "recombination": "shared_energy_quadrature_srh",
            "trap_charge": "absolute_not_equilibrium_referenced",
        },
        "energy_quadrature": {
            "coordinate": "truncated_normal_probability",
            "orders": list(quadrature_orders),
            "rule": "gauss_legendre",
            "recombination_probe": {
                "electron_density_m3": probe_electron_density_m3,
                "hole_density_m3": probe_hole_density_m3,
            },
        },
        "matrix": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "operating_point": {
            "applied_voltage_V": 0.0,
            "illumination": "dark",
            "state": "thermal_equilibrium",
        },
        "schema_version": "bulk-trap-equilibrium-refinement-protocol-v1",
        "solver": {
            "base_poisson_tolerance": base_poisson_tolerance,
            "effective_tolerance_formula": (
                "base_poisson_tolerance * matrix.tolerance_factor"
            ),
            "max_line_search_backtracks": max_line_search_backtracks,
            "max_newton_iterations": max_newton_iterations,
            "max_potential_step_V": max_potential_step_V,
            "poisson_coordinate": "electrostatic_potential_V",
            "trap_charge_tangent": "analytic",
        },
        "source": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "left_layer_name": left_layer_name,
            "right_layer_name": right_layer_name,
        },
        "topology": {
            "band_gap_narrowing": "excluded",
            "bulk_trap_profile_from_tau": "excluded",
            "contacts": "trap_aware_semiconductor_work_function",
            "field_dependent_mobility": "excluded",
            "interfaces": "homojunction_zero_recombination",
            "mobile_ions": "excluded",
            "production_mol": "fail_closed",
            "spatial_doping_profiles": "excluded",
        },
    }


def _relative_change(current: np.ndarray | float, previous: np.ndarray | float) -> float:
    current_array = np.asarray(current, dtype=float)
    previous_array = np.asarray(previous, dtype=float)
    scale = max(float(np.max(np.abs(current_array))), np.finfo(float).tiny)
    return float(np.max(np.abs(current_array - previous_array)) / scale)


def _probe_recombination(
    result: BulkTrapPNEquilibriumResult,
    material,
    *,
    electron_density_m3: float,
    hole_density_m3: float,
    quadrature_order: int,
) -> float:
    state = evaluate_bulk_trap_state(
        electron_density_m3,
        hole_density_m3,
        material.bulk_trap_distribution,
        band_gap_eV=float(material.Eg_phys[0]),
        effective_conduction_dos_m3=float(material.N_C_physical[0]),
        effective_valence_dos_m3=float(material.N_V_physical[0]),
        temperature_K=float(material.T_device),
        quadrature_order=quadrature_order,
    )
    if result.quadrature_order != quadrature_order:
        raise AssertionError("quadrature result/probe order mismatch")
    return float(state.recombination_rate_m3_s)


def run_bulk_trap_equilibrium_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one registered charged Gaussian bulk-trap equilibrium cell."""
    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("bulk-trap refinement requires config_loader='standard'")
    quadrature_orders = _quadrature_orders(options)
    base_poisson_tolerance = _finite_option(
        options,
        "base_poisson_tolerance",
        1.0e-8,
    )
    max_newton_iterations = _integer_option(
        options,
        "max_newton_iterations",
        100,
    )
    max_potential_step_V = _finite_option(
        options,
        "max_potential_step_V",
        0.1,
    )
    max_line_search_backtracks = _integer_option(
        options,
        "max_line_search_backtracks",
        20,
        minimum=0,
    )
    grid_alpha = _finite_option(options, "grid_alpha", 3.0)
    probe_n = _finite_option(options, "probe_electron_density_m3", 2.0e20)
    probe_p = _finite_option(options, "probe_hole_density_m3", 2.0e20)
    left_layer_name = _string_option(options, "left_layer_name", "p_trap")
    right_layer_name = _string_option(options, "right_layer_name", "n_trap")

    stack = load_device_from_yaml(project_root / lane.config_path)
    layers = electrical_layers(stack)
    if len(layers) != 2 or tuple(layer.name for layer in layers) != (
        left_layer_name,
        right_layer_name,
    ):
        raise ValueError("bulk-trap source layer identity mismatch")
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, point.grid) for layer in layers),
        alpha=grid_alpha,
    )
    material = build_material_arrays(
        grid,
        stack,
        bulk_trap_charge_closure=(
            BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM
        ),
    )
    contact = require_contact_thermodynamic_certificate(stack, material)
    try:
        build_material_arrays(grid, stack)
    except BulkTrapChargeCapabilityError:
        default_path_rejected = True
    else:
        default_path_rejected = False
    effective_tolerance = base_poisson_tolerance * point.tolerance_factor
    results = tuple(
        solve_bulk_trap_pn_equilibrium(
            grid,
            stack,
            quadrature_order=order,
            poisson_tolerance=effective_tolerance,
            max_newton_iterations=max_newton_iterations,
            max_potential_step_V=max_potential_step_V,
            max_line_search_backtracks=max_line_search_backtracks,
        )
        for order in quadrature_orders
    )
    final = results[-1]
    charge_changes = tuple(
        _relative_change(current.trap_charge_density_C_m3, previous.trap_charge_density_C_m3)
        for previous, current in zip(results, results[1:])
    )
    occupancy_changes = tuple(
        _relative_change(current.trap_occupancy, previous.trap_occupancy)
        for previous, current in zip(results, results[1:])
    )
    probe_rates = tuple(
        _probe_recombination(
            result,
            material,
            electron_density_m3=probe_n,
            hole_density_m3=probe_p,
            quadrature_order=order,
        )
        for result, order in zip(results, quadrature_orders)
    )
    recombination_changes = tuple(
        _relative_change(current, previous)
        for previous, current in zip(probe_rates, probe_rates[1:])
    )
    distribution = material.bulk_trap_distribution
    topology_verified = bool(
        distribution is not None
        and distribution.distribution == "gaussian"
        and material.bulk_trap_charge_closure == "research_equilibrium"
        and material.carrier_statistics == "maxwell_boltzmann"
        and material.dopant_ionization_model == "fully_ionized"
        and material.band_gap_narrowing_model == "off"
        and material.N_iface_state == 0
        and not material.has_selective_contacts
        and not material.has_dual_ions
        and not material.has_field_mobility
        and not material.has_trap_profile
        and np.all(material.D_ion_node == 0.0)
        and np.all(material.P_ion0 == 0.0)
    )
    protocol = _execution_protocol(
        lane,
        quadrature_orders=quadrature_orders,
        base_poisson_tolerance=base_poisson_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_potential_step_V=max_potential_step_V,
        max_line_search_backtracks=max_line_search_backtracks,
        probe_electron_density_m3=probe_n,
        probe_hole_density_m3=probe_p,
        left_layer_name=left_layer_name,
        right_layer_name=right_layer_name,
    )
    built_in = abs(
        final.left_contact.work_function_eV
        - final.right_contact.work_function_eV
    )
    average_field = built_in / float(grid[-1] - grid[0])
    total_trap_sheet_number = (
        float(distribution.total_density_m3) * float(grid[-1] - grid[0])
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "left_contact_trap_occupancy": float(
                    final.left_contact.bulk_trap_state.occupancy
                ),
                "normalized_integrated_bulk_trap_charge": (
                    final.integrated_bulk_trap_charge_C_m2
                    / (Q * total_trap_sheet_number)
                ),
                "peak_field_to_average_built_in_ratio": (
                    final.peak_electric_field_V_m / average_field
                ),
                "right_contact_trap_occupancy": float(
                    final.right_contact.bulk_trap_state.occupancy
                ),
            },
            "quality": {
                "bulk_trap_topology_verified": float(topology_verified),
                "contact_thermodynamics_certified": float(contact.certified),
                "default_production_path_rejected": float(default_path_rejected),
                "energy_quadrature_orders_completed": len(results),
                "max_energy_charge_relative_change": max(charge_changes),
                "max_energy_occupancy_relative_change": max(occupancy_changes),
                "max_energy_recombination_relative_change": max(
                    recombination_changes
                ),
                "max_gauss_law_relative_error": max(
                    result.gauss_law_relative_error for result in results
                ),
                "max_mass_action_relative_error": max(
                    result.maximum_mass_action_relative_error
                    for result in results
                ),
                "max_normalized_poisson_residual": max(
                    result.maximum_normalized_poisson_residual
                    for result in results
                ),
                "max_relative_face_current": max(
                    result.maximum_relative_face_current for result in results
                ),
                "maximum_newton_iterations": max(
                    result.newton_iterations for result in results
                ),
                "minimum_absolute_integrated_bulk_trap_charge_C_m2": min(
                    abs(result.integrated_bulk_trap_charge_C_m2)
                    for result in results
                ),
                "trap_occupancy_bounded": float(
                    all(
                        result.minimum_trap_occupancy >= 0.0
                        and result.maximum_trap_occupancy <= 1.0
                        for result in results
                    )
                ),
                "terminal_densities_positive": float(
                    all(
                        np.all(result.electron_density_m3 > 0.0)
                        and np.all(result.hole_density_m3 > 0.0)
                        for result in results
                    )
                ),
            },
            "units": {
                "bulk_trap_topology_verified": "1",
                "contact_thermodynamics_certified": "1",
                "default_production_path_rejected": "1",
                "energy_quadrature_orders_completed": "1",
                "left_contact_trap_occupancy": "1",
                "max_energy_charge_relative_change": "1",
                "max_energy_occupancy_relative_change": "1",
                "max_energy_recombination_relative_change": "1",
                "max_gauss_law_relative_error": "1",
                "max_mass_action_relative_error": "1",
                "max_normalized_poisson_residual": "1",
                "max_relative_face_current": "1",
                "maximum_newton_iterations": "1",
                "minimum_absolute_integrated_bulk_trap_charge_C_m2": "C/m2",
                "normalized_integrated_bulk_trap_charge": "1",
                "peak_field_to_average_built_in_ratio": "1",
                "right_contact_trap_occupancy": "1",
                "trap_occupancy_bounded": "1",
                "terminal_densities_positive": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "effective_poisson_tolerance": effective_tolerance,
                    "energy_quadrature_orders": list(quadrature_orders),
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": int(grid.size),
                    "newton_iterations": [
                        result.newton_iterations for result in results
                    ],
                    "probe_recombination_rates_m3_s": list(probe_rates),
                    "state_sha256": hashlib.sha256(
                        np.ascontiguousarray(final.state).tobytes()
                    ).hexdigest(),
                },
            },
        }
    )


__all__ = ["run_bulk_trap_equilibrium_refinement"]

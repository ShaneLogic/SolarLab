"""Registered temperature refinement for discrete dopant ionization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.statistics import DISCRETE_LEVEL, FERMI_DIRAC
from perovskite_sim.solver.degenerate_equilibrium import (
    solve_degenerate_pn_equilibrium,
)
from perovskite_sim.solver.mol import (
    DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF,
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


def _temperature_values(options: dict[str, object]) -> tuple[float, ...]:
    raw = options.get("temperature_values_K")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("temperature_values_K must contain at least three values")
    values = tuple(float(value) for value in raw)
    if (
        not all(math.isfinite(value) and value > 0.0 for value in values)
        or any(right <= left for left, right in zip(values, values[1:]))
    ):
        raise ValueError(
            "temperature_values_K must be finite, positive, and increasing"
        )
    return values


def _execution_protocol(
    lane: LaneDefinition,
    *,
    temperatures_K: tuple[float, ...],
    base_poisson_tolerance: float,
    max_newton_iterations: int,
    max_potential_step_V: float,
    max_line_search_backtracks: int,
    left_layer_name: str,
    right_layer_name: str,
) -> dict[str, object]:
    return {
        "carrier_statistics": {
            "density_law": "normalized_complete_fermi_dirac_half",
            "flux": "diffusion_enhanced_generalized_scharfetter_gummel",
        },
        "dopant_ionization": {
            "acceptor_charge": (
                "N_A/[1+g_A*exp(eta_p+(E_A-E_V)/V_T)]"
            ),
            "donor_charge": (
                "N_D/[1+g_D*exp(eta_n+(E_C-E_D)/V_T)]"
            ),
            "model": DISCRETE_LEVEL,
            "poisson_derivative": "analytic_local_occupation_tangent",
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
            "temperature_values_K": list(temperatures_K),
        },
        "schema_version": (
            "incomplete-ionization-temperature-refinement-protocol-v1"
        ),
        "solver": {
            "base_poisson_tolerance": base_poisson_tolerance,
            "effective_tolerance_formula": (
                "base_poisson_tolerance * matrix.tolerance_factor"
            ),
            "max_line_search_backtracks": max_line_search_backtracks,
            "max_newton_iterations": max_newton_iterations,
            "max_potential_step_V": max_potential_step_V,
            "poisson_coordinate": "electrostatic_potential_V",
        },
        "source": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "left_layer_name": left_layer_name,
            "right_layer_name": right_layer_name,
        },
        "topology": {
            "band_gap_narrowing": "excluded",
            "bulk_recombination": "disabled",
            "contacts": "ohmic_semiconductor_work_function",
            "dopant_model_scope": "noninteracting_discrete_levels",
            "interfaces": "homojunction_zero_recombination",
            "mobile_ions": "excluded",
            "production_experiments": "excluded",
        },
    }


def run_incomplete_ionization_temperature_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one grid/tolerance cell across the frozen temperature scan."""
    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError(
            "incomplete-ionization refinement requires config_loader='standard'"
        )
    temperatures = _temperature_values(options)
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
    left_layer_name = _string_option(options, "left_layer_name", "p_discrete")
    right_layer_name = _string_option(options, "right_layer_name", "n_discrete")

    base_stack = load_device_from_yaml(project_root / lane.config_path)
    layers = electrical_layers(base_stack)
    if len(layers) != 2 or tuple(layer.name for layer in layers) != (
        left_layer_name,
        right_layer_name,
    ):
        raise ValueError("incomplete-ionization source layer identity mismatch")
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, point.grid) for layer in layers),
        alpha=grid_alpha,
    )
    effective_tolerance = base_poisson_tolerance * point.tolerance_factor

    results = []
    contacts = []
    topology_checks = []
    state_hash = hashlib.sha256()
    for temperature in temperatures:
        stack = replace(base_stack, T=temperature)
        material = build_material_arrays(
            grid,
            stack,
            carrier_statistics_transport=(
                DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
            ),
        )
        contacts.append(require_contact_thermodynamic_certificate(stack, material))
        topology_checks.append(
            material.carrier_statistics == FERMI_DIRAC
            and material.dopant_ionization_model == DISCRETE_LEVEL
            and material.degenerate_recombination_model == "off"
            and material.donor_binding_energy_eV is not None
            and material.acceptor_binding_energy_eV is not None
            and not material.interface_faces
            and material.N_iface_state == 0
            and not material.has_selective_contacts
            and not material.has_dual_ions
            and not material.has_field_mobility
            and np.all(material.D_ion_node == 0.0)
            and np.all(material.P_ion0 == 0.0)
        )
        result = solve_degenerate_pn_equilibrium(
            grid,
            stack,
            poisson_tolerance=effective_tolerance,
            max_newton_iterations=max_newton_iterations,
            max_potential_step_V=max_potential_step_V,
            max_line_search_backtracks=max_line_search_backtracks,
        )
        results.append(result)
        state_hash.update(np.ascontiguousarray(result.state).tobytes())

    donor_fractions = np.asarray(
        [result.right_contact.neutrality.donor_ionized_fraction for result in results]
    )
    acceptor_fractions = np.asarray(
        [
            result.left_contact.neutrality.acceptor_ionized_fraction
            for result in results
        ]
    )
    built_in = np.asarray(
        [
            abs(
                result.left_contact.work_function_eV
                - result.right_contact.work_function_eV
            )
            for result in results
        ]
    )
    total_thickness = float(base_stack.total_thickness)
    normalized_peak_field = np.asarray(
        [
            result.peak_electric_field_V_m
            / max(voltage / total_thickness, np.finfo(float).tiny)
            for result, voltage in zip(results, built_in)
        ]
    )
    normalized_charge_width = np.asarray(
        [result.depletion_width_m / total_thickness for result in results]
    )
    charge_balance = np.asarray(
        [result.charge_balance_relative_error for result in results]
    )
    monotone = bool(
        np.all(np.diff(donor_fractions) > 0.0)
        and np.all(np.diff(acceptor_fractions) > 0.0)
    )
    bounded = bool(
        np.all((donor_fractions > 0.0) & (donor_fractions < 1.0))
        and np.all((acceptor_fractions > 0.0) & (acceptor_fractions < 1.0))
        and all(
            np.all(result.ionized_donor_density_m3 >= 0.0)
            and np.all(result.ionized_acceptor_density_m3 >= 0.0)
            for result in results
        )
    )
    minimum_freeze_out_change = min(
        donor_fractions[-1] - donor_fractions[0],
        acceptor_fractions[-1] - acceptor_fractions[0],
    )
    protocol = _execution_protocol(
        lane,
        temperatures_K=temperatures,
        base_poisson_tolerance=base_poisson_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_potential_step_V=max_potential_step_V,
        max_line_search_backtracks=max_line_search_backtracks,
        left_layer_name=left_layer_name,
        right_layer_name=right_layer_name,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "acceptor_ionized_fraction_temperature_curve": (
                    acceptor_fractions.tolist()
                ),
                "donor_ionized_fraction_temperature_curve": (
                    donor_fractions.tolist()
                ),
                "normalized_peak_field_temperature_curve": (
                    normalized_peak_field.tolist()
                ),
                "normalized_integrated_charge_width_temperature_curve": (
                    normalized_charge_width.tolist()
                ),
                "space_charge_balance_temperature_curve": (
                    charge_balance.tolist()
                ),
            },
            "quality": {
                "contact_thermodynamics_certified": float(
                    all(contact.certified for contact in contacts)
                ),
                "freeze_out_monotone_with_temperature": float(monotone),
                "incomplete_ionization_topology_verified": float(
                    all(topology_checks)
                ),
                "ionized_fractions_bounded": float(bounded),
                "max_normalized_carrier_rate": max(
                    result.maximum_normalized_carrier_rate for result in results
                ),
                "max_normalized_poisson_residual": max(
                    result.maximum_normalized_poisson_residual for result in results
                ),
                "max_relative_face_current": max(
                    result.maximum_relative_face_current for result in results
                ),
                "max_space_charge_balance_relative_error": float(
                    np.max(charge_balance)
                ),
                "maximum_newton_iterations": max(
                    result.newton_iterations for result in results
                ),
                "minimum_freeze_out_fraction_change": float(
                    minimum_freeze_out_change
                ),
                "temperature_points_completed": len(results),
                "terminal_densities_positive": float(
                    all(
                        np.all(result.electron_density_m3 > 0.0)
                        and np.all(result.hole_density_m3 > 0.0)
                        for result in results
                    )
                ),
            },
            "units": {
                "acceptor_ionized_fraction_temperature_curve": "1",
                "contact_thermodynamics_certified": "1",
                "donor_ionized_fraction_temperature_curve": "1",
                "freeze_out_monotone_with_temperature": "1",
                "incomplete_ionization_topology_verified": "1",
                "ionized_fractions_bounded": "1",
                "max_normalized_carrier_rate": "1",
                "max_normalized_poisson_residual": "1",
                "max_relative_face_current": "1",
                "max_space_charge_balance_relative_error": "1",
                "maximum_newton_iterations": "1",
                "minimum_freeze_out_fraction_change": "1",
                "normalized_peak_field_temperature_curve": "1",
                "normalized_integrated_charge_width_temperature_curve": "1",
                "space_charge_balance_temperature_curve": "1",
                "temperature_points_completed": "1",
                "terminal_densities_positive": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "effective_poisson_tolerance": effective_tolerance,
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": int(grid.size),
                    "state_sha256": state_hash.hexdigest(),
                    "temperature_values_K": list(temperatures),
                },
            },
        }
    )


__all__ = ["run_incomplete_ionization_temperature_refinement"]

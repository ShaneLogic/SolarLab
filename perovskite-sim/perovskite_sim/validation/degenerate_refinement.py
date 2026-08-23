"""Content-addressed refinement adapter for restricted FD p-n equilibrium."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.statistics import FERMI_DIRAC
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


def _execution_protocol(
    lane: LaneDefinition,
    *,
    base_poisson_tolerance: float,
    max_newton_iterations: int,
    max_potential_step_V: float,
    max_line_search_backtracks: int,
    left_layer_name: str,
    right_layer_name: str,
) -> dict[str, object]:
    return {
        "analytic_reference": {
            "model": "abrupt_fully_depleted_homojunction",
            "observables": ["depletion_width", "peak_electric_field"],
            "role": "quality_oracle_not_external_validation",
        },
        "carrier_statistics": {
            "density_law": "normalized_complete_fermi_dirac_half",
            "diffusion_enhancement": "logarithmic_secant_generalized_einstein",
            "flux": "diffusion_enhanced_generalized_scharfetter_gummel",
            "ionization": "fully_ionized",
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
        "schema_version": "degenerate-pn-equilibrium-refinement-protocol-v1",
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
            "field_dependent_mobility": "excluded",
            "interfaces": "homojunction_zero_recombination",
            "mobile_ions": "excluded",
            "spatial_doping_profiles": "excluded",
        },
    }


def run_degenerate_pn_equilibrium_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one registered high-doping FD p+/n+ equilibrium cell."""
    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("degenerate PN refinement requires config_loader='standard'")
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
    left_layer_name = _string_option(options, "left_layer_name", "p_plus")
    right_layer_name = _string_option(options, "right_layer_name", "n_plus")

    stack = load_device_from_yaml(project_root / lane.config_path)
    layers = electrical_layers(stack)
    if len(layers) != 2 or tuple(layer.name for layer in layers) != (
        left_layer_name,
        right_layer_name,
    ):
        raise ValueError("degenerate PN source layer identity mismatch")
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, point.grid) for layer in layers),
        alpha=grid_alpha,
    )
    material = build_material_arrays(
        grid,
        stack,
        carrier_statistics_transport=(
            DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
        ),
    )
    contact = require_contact_thermodynamic_certificate(stack, material)
    effective_tolerance = base_poisson_tolerance * point.tolerance_factor
    result = solve_degenerate_pn_equilibrium(
        grid,
        stack,
        poisson_tolerance=effective_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_potential_step_V=max_potential_step_V,
        max_line_search_backtracks=max_line_search_backtracks,
    )
    topology_verified = bool(
        material.carrier_statistics == FERMI_DIRAC
        and material.degenerate_recombination_model == "off"
        and not material.interface_faces
        and len(material.interface_nodes) == 1
        and material.N_iface_state == 0
        and not material.has_selective_contacts
        and not material.has_dual_ions
        and not material.has_field_mobility
        and np.all(material.D_ion_node == 0.0)
        and np.all(material.P_ion0 == 0.0)
    )
    protocol = _execution_protocol(
        lane,
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
                "depletion_width_to_analytic_ratio": (
                    result.depletion_width_m / result.analytic_depletion_width_m
                ),
                "peak_field_to_analytic_ratio": (
                    result.peak_electric_field_V_m
                    / result.analytic_peak_electric_field_V_m
                ),
                "space_charge_balance_relative_error": (
                    result.charge_balance_relative_error
                ),
            },
            "quality": {
                "contact_thermodynamics_certified": float(contact.certified),
                "fd_recombination_off_topology_verified": float(
                    topology_verified
                ),
                "max_depletion_approximation_relative_error": (
                    result.depletion_width_relative_error
                ),
                "max_normalized_carrier_rate": (
                    result.maximum_normalized_carrier_rate
                ),
                "max_normalized_poisson_residual": (
                    result.maximum_normalized_poisson_residual
                ),
                "max_peak_field_approximation_relative_error": (
                    result.peak_field_relative_error
                ),
                "max_relative_face_current": (
                    result.maximum_relative_face_current
                ),
                "max_space_charge_balance_relative_error": (
                    result.charge_balance_relative_error
                ),
                "newton_iterations": result.newton_iterations,
                "terminal_densities_positive": float(
                    np.all(result.electron_density_m3 > 0.0)
                    and np.all(result.hole_density_m3 > 0.0)
                ),
            },
            "units": {
                "contact_thermodynamics_certified": "1",
                "depletion_width_to_analytic_ratio": "1",
                "fd_recombination_off_topology_verified": "1",
                "max_depletion_approximation_relative_error": "1",
                "max_normalized_carrier_rate": "1",
                "max_normalized_poisson_residual": "1",
                "max_peak_field_approximation_relative_error": "1",
                "max_relative_face_current": "1",
                "max_space_charge_balance_relative_error": "1",
                "newton_iterations": "1",
                "peak_field_to_analytic_ratio": "1",
                "space_charge_balance_relative_error": "1",
                "terminal_densities_positive": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "effective_poisson_tolerance": effective_tolerance,
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": int(grid.size),
                    "newton_iterations": result.newton_iterations,
                    "state_sha256": hashlib.sha256(
                        np.ascontiguousarray(result.state).tobytes()
                    ).hexdigest(),
                },
            },
        }
    )


__all__ = ["run_degenerate_pn_equilibrium_refinement"]

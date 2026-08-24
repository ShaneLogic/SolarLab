"""Content-addressed refinement adapter for the combined ion/interface DAE."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.interface_plane import (
    FERMI_RICHARDSON,
    solve_interface_states_live_qss,
)
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae_ion_interface_integrator import (
    run_ion_interface_backward_euler_reference,
)
from perovskite_sim.solver.dae_ion_interface_jacobian import (
    build_ion_interface_structured_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_ion_interface_states import (
    build_single_ion_algebraic_interface_consistent_initial_condition,
    build_single_ion_algebraic_interface_dae,
)
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    poisson_right_boundary,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy

from .dae_interface_refinement import (
    _max_relative_interface_difference,
    _terminal_carriers,
)
from .dae_ion_refinement import _max_relative_ion_difference
from .dae_refinement import (
    _finite_option,
    _integer_option,
    _max_log_carrier_difference,
    _protocol_metadata,
    _string_option,
    _time_step_count,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def _execution_protocol(
    lane: LaneDefinition,
    *,
    applied_voltage_V: float,
    final_time_s: float,
    carrier_reference_time_s: float,
    ion_reference_time_s: float,
    positive_ion_diffusion_m2_s: float,
    positive_ion_reference_m3: float,
    positive_ion_site_limit_m3: float,
    interface_velocity_m_s: float,
    cross_transmission: float,
    base_time_steps: int,
    reference_grid_intervals_per_layer: int,
    residual_tolerance: float,
    max_newton_iterations: int,
    max_line_search_backtracks: int,
    max_log_density_update: float,
    max_ion_logit_update: float,
    max_interface_logit_update: float,
    finite_difference_relative_step: float,
    mol_rtol: float,
    mol_atol_m3: float,
    mol_max_step_divisor: int,
) -> dict[str, object]:
    return {
        "backward_euler": {
            "base_time_steps": base_time_steps,
            "finite_difference_relative_step": finite_difference_relative_step,
            "jacobian_modes": ["dense_central", "structured_analytic"],
            "max_interface_logit_update": max_interface_logit_update,
            "max_ion_logit_update": max_ion_logit_update,
            "max_line_search_backtracks": max_line_search_backtracks,
            "max_log_density_update": max_log_density_update,
            "max_newton_iterations": max_newton_iterations,
            "reference_grid_intervals_per_layer": (reference_grid_intervals_per_layer),
            "residual_tolerance": residual_tolerance,
            "step_count_formula": (
                "base_time_steps * (intervals_per_layer / "
                "reference_grid_intervals_per_layer)^2 / tolerance_factor"
            ),
            "time_coordinate": "physical_carrier_and_positive_ion_density",
        },
        "interface_transport": {
            "bulk_supply": "live_adjacent_endpoint_fermi_projection",
            "clamp_contract": "fail_closed_if_any_clamp_is_active",
            "cross_exchange": "reciprocal_fermi_richardson",
            "cross_transmission": cross_transmission,
            "interface_velocity_m_s": interface_velocity_m_s,
            "state_balance": "bulk_te_plus_cross_te_plus_shared_occupancy_srh",
            "state_coordinates": "four_dos_bounded_shifted_logits",
        },
        "matrix": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "mol_reference": {
            "atol_m3": mol_atol_m3,
            "interface_states": "locally_eliminated_qss",
            "max_step_divisor": mol_max_step_divisor,
            "method": "Radau",
            "numerical_diagnostics": "research_strict_zero_floors",
            "rtol": mol_rtol,
            "state_coordinates": "density",
        },
        "operating_point": {
            "applied_voltage_V": applied_voltage_V,
            "final_time_s": final_time_s,
            "illuminated": False,
            "photon_flux_m2_s": 0.0,
        },
        "schema_version": "single-ion-algebraic-interface-dae-refinement-protocol-v1",
        "source_slice": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "transformations": [
                "remove_InterfaceDefect",
                "disable_interface_charge_closure",
                "set_Phi_zero",
                "inject_synthetic_single_positive_ion_in_both_layers",
                "uniform_grid_per_electrical_layer",
            ],
        },
        "topology": {
            "carrier_coordinates": ["log_n", "log_p"],
            "carrier_reference_time_s": carrier_reference_time_s,
            "contacts": "ohmic_dirichlet",
            "cross_node_carrier_sampling": "excluded",
            "dynamic_interface_states": "excluded",
            "interface_charge": "off",
            "interface_count": 1,
            "interface_defect": "excluded",
            "interface_sampling": "replaced_face_endpoints",
            "interface_states": "four_algebraic_fermi_richardson_states",
            "ion_boundary": "blocking_zero_flux",
            "ion_coordinate": "shifted_logit_positive_site_occupancy",
            "ion_reference_time_s": ion_reference_time_s,
            "mobile_ions": "single_positive_synthetic",
            "positive_ion_diffusion_m2_s": positive_ion_diffusion_m2_s,
            "positive_ion_reference_m3": positive_ion_reference_m3,
            "positive_ion_site_limit_m3": positive_ion_site_limit_m3,
            "poisson_potential": "algebraic_with_P_minus_P0",
            "steric_law": "device_configured_diffusion_only",
            "two_sided_trace_geometry": "excluded",
        },
    }


def run_single_ion_algebraic_interface_dae_transient(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one combined DAE grid-by-time-step refinement cell."""
    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("the combined DAE lane requires config_loader='standard'")
    applied_voltage_V = _finite_option(
        options,
        "applied_voltage_V",
        0.01,
        positive=False,
    )
    final_time_s = _finite_option(options, "final_time_s", 1.0e-2)
    carrier_reference_time_s = _finite_option(
        options,
        "carrier_reference_time_s",
        1.0e-7,
    )
    ion_reference_time_s = _finite_option(options, "ion_reference_time_s", 1.0)
    ion_diffusion = _finite_option(
        options,
        "positive_ion_diffusion_m2_s",
        1.0e-16,
    )
    ion_reference = _finite_option(
        options,
        "positive_ion_reference_m3",
        1.0e22,
    )
    ion_site_limit = _finite_option(
        options,
        "positive_ion_site_limit_m3",
        1.0e24,
    )
    if ion_reference >= ion_site_limit:
        raise ValueError(
            "positive_ion_reference_m3 must be below positive_ion_site_limit_m3"
        )
    interface_velocity_m_s = _finite_option(
        options,
        "interface_velocity_m_s",
        1.0e5,
    )
    cross_transmission = _finite_option(
        options,
        "cross_transmission",
        1.0,
    )
    if cross_transmission > 1.0:
        raise ValueError("lane option 'cross_transmission' must be <= 1")
    base_time_steps = _integer_option(options, "base_time_steps", 2)
    reference_grid_intervals = _integer_option(
        options,
        "reference_grid_intervals_per_layer",
        4,
    )
    residual_tolerance = _finite_option(
        options,
        "newton_residual_tolerance",
        1.0e-8,
    )
    max_newton_iterations = _integer_option(
        options,
        "max_newton_iterations",
        24,
    )
    max_line_search_backtracks = _integer_option(
        options,
        "max_line_search_backtracks",
        12,
        minimum=0,
    )
    max_log_density_update = _finite_option(
        options,
        "max_log_density_update",
        2.0,
    )
    max_ion_logit_update = _finite_option(
        options,
        "max_ion_logit_update",
        2.0,
    )
    max_interface_logit_update = _finite_option(
        options,
        "max_interface_logit_update",
        2.0,
    )
    finite_difference_relative_step = _finite_option(
        options,
        "finite_difference_relative_step",
        1.0e-6,
    )
    mol_rtol = _finite_option(options, "mol_rtol", 1.0e-10)
    mol_atol_m3 = _finite_option(options, "mol_atol_m3", 1.0e2)
    mol_max_step_divisor = _integer_option(
        options,
        "mol_max_step_divisor",
        400,
    )
    time_steps = _time_step_count(
        base_time_steps,
        point.tolerance_factor,
        grid_intervals=point.grid,
        reference_grid_intervals=reference_grid_intervals,
    )

    source = load_device_from_yaml(project_root / lane.config_path)
    layers = tuple(
        replace(
            layer,
            params=replace(
                layer.params,
                D_ion=ion_diffusion,
                P0=ion_reference,
                P_lim=ion_site_limit,
            ),
        )
        for layer in source.layers
        if layer.params is not None
    )
    if len(layers) != 2:
        raise ValueError(
            "combined DAE source must contain exactly two electrical layers"
        )
    stack = replace(
        source,
        layers=layers,
        interface_defects=(),
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
        Phi=0.0,
        S_n_left=None,
        S_p_left=None,
        S_n_right=None,
        S_p_right=None,
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid(
        [Layer(layer.thickness, point.grid) for layer in stack.layers],
        alpha=1.0,
    )
    model = build_single_ion_algebraic_interface_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=applied_voltage_V,
        illuminated=False,
        carrier_reference_time_s=carrier_reference_time_s,
        ion_reference_time_s=ion_reference_time_s,
        interface_velocity_m_s=interface_velocity_m_s,
        cross_transmission=cross_transmission,
        interface_residual_tolerance=residual_tolerance,
    )
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(
        model,
        residual_tolerance=residual_tolerance,
    )
    if not initial.certified:
        raise RuntimeError("combined DAE consistent initial condition is not certified")

    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=False,
        V_app=applied_voltage_V,
        rtol=mol_rtol,
        atol=mol_atol_m3,
        max_step=final_time_s / mol_max_step_divisor,
        mat=model.material,
        numerical_diagnostics=NumericalDiagnosticsPolicy.research_strict(
            terminal_density_floor_m3=0.0,
            bulk_srh_denominator_floor_s_m3=0.0,
        ),
    )
    if not bool(mol.success) or np.asarray(mol.y).shape != (3 * grid.size, 2):
        raise RuntimeError(
            f"high-accuracy combined MoL reference failed: {mol.message}"
        )
    mol_terminal = StateVec.unpack(mol.y[:, -1], grid.size)
    mol_rho = _charge_density(
        mol_terminal.p,
        mol_terminal.n,
        mol_terminal.P,
        model.material.P_ion0,
        model.material.N_A,
        model.material.N_D,
    )
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        mol_rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, applied_voltage_V),
    )
    mol_interface = solve_interface_states_live_qss(
        model.material,
        stack,
        mol_terminal.n,
        mol_terminal.p,
        mol_potential,
        V_app=applied_voltage_V,
        v_th_eff=model.material.iface_state_v_th,
        cross_transmission=model.material.iface_qss_cross_transmission,
        interface_transport_model=model.material.iface_qss_transport_model,
        residual_tolerance=residual_tolerance,
        fail_on_residual=True,
    )

    time = np.linspace(0.0, final_time_s, time_steps + 1)
    solver_options = {
        "initial": initial,
        "residual_tolerance": residual_tolerance,
        "max_newton_iterations": max_newton_iterations,
        "max_line_search_backtracks": max_line_search_backtracks,
        "max_log_density_update": max_log_density_update,
        "max_ion_logit_update": max_ion_logit_update,
        "max_interface_logit_update": max_interface_logit_update,
        "finite_difference_relative_step": finite_difference_relative_step,
    }
    dense = run_ion_interface_backward_euler_reference(
        model,
        time,
        jacobian_mode="dense_central",
        **solver_options,
    )
    structured = run_ion_interface_backward_euler_reference(
        model,
        time,
        jacobian_mode="structured_analytic",
        **solver_options,
    )
    dense_terminal = StateVec.unpack(dense.physical_states[-1], grid.size)
    structured_terminal = StateVec.unpack(structured.physical_states[-1], grid.size)
    structured_n, structured_p = _terminal_carriers(
        structured.physical_states[-1],
        grid.size,
    )
    dense_n, dense_p = _terminal_carriers(dense.physical_states[-1], grid.size)
    structured_interface = structured.interface_states_m3[-1]
    dense_interface = dense.interface_states_m3[-1]

    terminal_log_error = max(
        float(np.max(np.abs(np.log(structured_n / mol_terminal.n)))),
        float(np.max(np.abs(np.log(structured_p / mol_terminal.p)))),
    )
    terminal_ion_error = float(
        np.max(np.abs(structured_terminal.P - mol_terminal.P) / mol_terminal.P)
    )
    terminal_interface_error = float(
        np.max(
            np.abs(structured_interface - mol_interface.state_m3)
            / mol_interface.state_m3
        )
    )
    terminal_potential_error = float(
        np.max(np.abs(structured.potentials_V[-1] - mol_potential))
    )
    structured_dense_log_difference = _max_log_carrier_difference(
        structured.physical_states,
        dense.physical_states,
        grid.size,
    )
    structured_dense_ion_difference = _max_relative_ion_difference(
        structured.physical_states,
        dense.physical_states,
        grid.size,
    )
    structured_dense_interface_difference = _max_relative_interface_difference(
        structured.interface_states_m3,
        dense.interface_states_m3,
    )
    structured_dense_potential_difference = float(
        np.max(np.abs(structured.potentials_V - dense.potentials_V))
    )

    initial_state = StateVec.unpack(initial.physical_state, grid.size)
    initial_inventory = dual_cell_integral(grid, initial_state.P)
    mol_inventory = dual_cell_integral(grid, mol_terminal.P)
    inventory_rate_scale = initial_inventory / final_time_s
    maximum_ion_balance_relative = (
        max(
            dense.max_positive_ion_balance_defect_m2_s,
            structured.max_positive_ion_balance_defect_m2_s,
        )
        / inventory_rate_scale
    )
    maximum_ion_rhs_inventory_relative = (
        max(
            dense.max_positive_ion_rhs_inventory_rate_m2_s,
            structured.max_positive_ion_rhs_inventory_rate_m2_s,
        )
        / inventory_rate_scale
    )
    maximum_inventory_drift = max(
        dense.max_relative_positive_ion_inventory_drift,
        structured.max_relative_positive_ion_inventory_drift,
        abs(mol_inventory - initial_inventory) / initial_inventory,
    )
    ion_motion = float(
        np.max(np.abs(structured_terminal.P - initial_state.P) / initial_state.P)
    )
    interface_motion = float(
        np.max(
            np.abs(structured_interface - initial.interface_state_m3)
            / initial.interface_state_m3
        )
    )

    step_dt = final_time_s / time_steps
    tangent_evidence = tuple(
        build_ion_interface_structured_backward_euler_jacobian(
            model,
            coordinate,
            step_dt,
        )
        for coordinate in structured.coordinates
    )
    initial_tangent = tangent_evidence[0]
    minimum_projection_margin = min(
        item.local_interface.minimum_projection_occupation_margin
        for item in tangent_evidence
    )
    minimum_cross_margin = min(
        item.local_interface.minimum_cross_occupation_margin
        for item in tangent_evidence
    )
    minimum_srh_margin = min(
        item.local_interface.minimum_srh_occupancy_margin for item in tangent_evidence
    )
    minimum_interface_density = min(
        item.local_interface.minimum_interface_density_margin_m3
        for item in tangent_evidence
    )
    minimum_interface_dos = min(
        item.local_interface.minimum_interface_dos_margin_m3
        for item in tangent_evidence
    )
    minimum_ion_occupation = min(
        item.minimum_positive_ion_occupation_margin for item in tangent_evidence
    )

    max_carrier_residual = max(
        dense.max_normalized_carrier_residual,
        structured.max_normalized_carrier_residual,
    )
    max_ion_residual = max(
        dense.max_normalized_positive_ion_residual,
        structured.max_normalized_positive_ion_residual,
    )
    max_interface_residual = max(
        dense.max_normalized_interface_residual,
        structured.max_normalized_interface_residual,
    )
    max_algebraic_residual = max(
        dense.max_normalized_algebraic_residual,
        structured.max_normalized_algebraic_residual,
    )
    max_electron_balance = max(
        dense.max_electron_balance_defect_A_m2,
        structured.max_electron_balance_defect_A_m2,
    )
    max_hole_balance = max(
        dense.max_hole_balance_defect_A_m2,
        structured.max_hole_balance_defect_A_m2,
    )
    max_interface_balance = Q * max(
        dense.max_interface_state_balance_m2_s,
        structured.max_interface_state_balance_m2_s,
    )
    maximum_condition = max(
        (report.max_scaled_jacobian_condition for report in structured.step_reports),
        default=0.0,
    )
    if dense.total_residual_evaluations <= 0:
        raise RuntimeError("dense combined DAE reference reported no residual work")
    structured_work_fraction = (
        structured.total_residual_evaluations / dense.total_residual_evaluations
    )
    topology_verified = bool(
        len(stack.layers) == 2
        and len(stack.interfaces) == 1
        and not stack.interface_defects
        and stack.interface_charge_closure == "off"
        and len(model.material.interface_nodes) == 1
        and model.material.N_iface_state == 0
        and model.material.iface_qss_exclusive_transport
        and model.material.iface_state_physical_offsets
        and not model.material.iface_state_partition
        and model.material.iface_qss_transport_model == FERMI_RICHARDSON
        and tuple(model.material.interface_eval_node_n)
        == tuple(model.material.interface_nodes)
        and tuple(model.material.interface_eval_node_p)
        == tuple(model.material.interface_nodes)
        and not model.material.has_selective_contacts
        and not model.material.has_dual_ions
        and np.all(model.material.D_ion_node > 0.0)
        and np.all(model.material.P_ion0 > 0.0)
        and np.all(model.material.P_ion0 < model.material.P_lim_node)
        and not model.material.has_field_mobility
        and not model.material.has_radiative_reabsorption
    )
    site_occupancy_admissible = bool(
        np.all(dense.physical_states[:, 2 * grid.size : 3 * grid.size] > 0.0)
        and np.all(
            dense.physical_states[:, 2 * grid.size : 3 * grid.size]
            < model.material.P_lim_node
        )
        and np.all(structured.physical_states[:, 2 * grid.size : 3 * grid.size] > 0.0)
        and np.all(
            structured.physical_states[:, 2 * grid.size : 3 * grid.size]
            < model.material.P_lim_node
        )
        and np.all(mol_terminal.P > 0.0)
        and np.all(mol_terminal.P < model.material.P_lim_node)
    )
    interface_states_bounded = bool(
        np.all(dense.interface_states_m3 > 0.0)
        and np.all(
            dense.interface_states_m3
            < model.layout.interface_capacity_m3[np.newaxis, :]
        )
        and np.all(structured.interface_states_m3 > 0.0)
        and np.all(
            structured.interface_states_m3
            < model.layout.interface_capacity_m3[np.newaxis, :]
        )
    )
    numerical_health = mol.numerical_diagnostics
    protocol = _execution_protocol(
        lane,
        applied_voltage_V=applied_voltage_V,
        final_time_s=final_time_s,
        carrier_reference_time_s=carrier_reference_time_s,
        ion_reference_time_s=ion_reference_time_s,
        positive_ion_diffusion_m2_s=ion_diffusion,
        positive_ion_reference_m3=ion_reference,
        positive_ion_site_limit_m3=ion_site_limit,
        interface_velocity_m_s=interface_velocity_m_s,
        cross_transmission=cross_transmission,
        base_time_steps=base_time_steps,
        reference_grid_intervals_per_layer=reference_grid_intervals,
        residual_tolerance=residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_line_search_backtracks=max_line_search_backtracks,
        max_log_density_update=max_log_density_update,
        max_ion_logit_update=max_ion_logit_update,
        max_interface_logit_update=max_interface_logit_update,
        finite_difference_relative_step=finite_difference_relative_step,
        mol_rtol=mol_rtol,
        mol_atol_m3=mol_atol_m3,
        mol_max_step_divisor=mol_max_step_divisor,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "structured_dense_terminal_interface_state_relative_difference": (
                    float(
                        np.max(
                            np.abs(structured_interface - dense_interface)
                            / dense_interface
                        )
                    )
                ),
                "structured_dense_terminal_log_density_difference": max(
                    float(np.max(np.abs(np.log(structured_n / dense_n)))),
                    float(np.max(np.abs(np.log(structured_p / dense_p)))),
                ),
                "structured_dense_terminal_positive_ion_relative_difference": float(
                    np.max(
                        np.abs(structured_terminal.P - dense_terminal.P)
                        / dense_terminal.P
                    )
                ),
                "structured_dense_terminal_potential_difference_V": float(
                    np.max(np.abs(structured.potentials_V[-1] - dense.potentials_V[-1]))
                ),
                "terminal_interface_state_relative_error": terminal_interface_error,
                "terminal_interface_occupation": (
                    structured_interface / model.layout.interface_capacity_m3
                ),
                "terminal_log_density_error": terminal_log_error,
                "terminal_positive_ion_relative_error": terminal_ion_error,
                "terminal_potential_error_V": terminal_potential_error,
            },
            "quality": {
                "bulk_srh_denominator_positive": float(
                    initial_tangent.minimum_bulk_srh_denominator_s_m3 > 0.0
                ),
                "clamp_inactive_slice_verified": float(
                    minimum_projection_margin > 0.0
                    and minimum_cross_margin > 0.0
                    and minimum_srh_margin > 0.0
                    and minimum_interface_density > 0.0
                    and minimum_interface_dos > 0.0
                    and minimum_ion_occupation > 0.0
                ),
                "combined_ion_interface_topology_verified": float(topology_verified),
                "consistent_initial_condition_certified": float(initial.certified),
                "dense_reference_success": float(dense.success),
                "max_electron_balance_defect_A_m2": max_electron_balance,
                "max_hole_balance_defect_A_m2": max_hole_balance,
                "max_interface_state_balance_A_m2": max_interface_balance,
                "max_normalized_algebraic_residual": max_algebraic_residual,
                "max_normalized_carrier_residual": max_carrier_residual,
                "max_normalized_interface_residual": max_interface_residual,
                "max_normalized_positive_ion_residual": max_ion_residual,
                "max_positive_ion_balance_relative": maximum_ion_balance_relative,
                "max_positive_ion_inventory_relative_drift": maximum_inventory_drift,
                "max_positive_ion_rhs_inventory_relative": (
                    maximum_ion_rhs_inventory_relative
                ),
                "max_structured_dense_interface_state_relative_difference": (
                    structured_dense_interface_difference
                ),
                "max_structured_dense_log_density_difference": (
                    structured_dense_log_difference
                ),
                "max_structured_dense_positive_ion_relative_difference": (
                    structured_dense_ion_difference
                ),
                "max_structured_dense_potential_difference_V": (
                    structured_dense_potential_difference
                ),
                "max_terminal_interface_state_relative_error": (
                    terminal_interface_error
                ),
                "max_terminal_log_density_error": terminal_log_error,
                "max_terminal_positive_ion_relative_error": terminal_ion_error,
                "max_terminal_potential_error_V": terminal_potential_error,
                "minimum_interface_state_relative_motion": interface_motion,
                "minimum_positive_ion_relative_motion": ion_motion,
                "mol_numerical_health_passed": float(
                    numerical_health.would_pass_strict
                    and numerical_health.nonfinite_rhs_evaluations == 0
                    and numerical_health.nonfinite_trial_evaluations == 0
                ),
                "mol_reference_success": float(mol.success),
                "site_occupancy_admissible": float(site_occupancy_admissible),
                "structured_analytic_success": float(structured.success),
                "structured_csr_nonzeros_per_node": (
                    initial_tangent.nonzero_count / grid.size
                ),
                "structured_jacobian_condition_finite": float(
                    math.isfinite(maximum_condition)
                ),
                "structured_rhs_work_fraction": structured_work_fraction,
                "terminal_densities_positive": float(
                    np.all(structured_n > 0.0)
                    and np.all(structured_p > 0.0)
                    and np.all(dense_n > 0.0)
                    and np.all(dense_p > 0.0)
                    and np.all(mol_terminal.n > 0.0)
                    and np.all(mol_terminal.p > 0.0)
                ),
                "terminal_interface_states_bounded": float(interface_states_bounded),
            },
            "units": {
                "bulk_srh_denominator_positive": "1",
                "clamp_inactive_slice_verified": "1",
                "combined_ion_interface_topology_verified": "1",
                "consistent_initial_condition_certified": "1",
                "dense_reference_success": "1",
                "max_electron_balance_defect_A_m2": "A m-2",
                "max_hole_balance_defect_A_m2": "A m-2",
                "max_interface_state_balance_A_m2": "A m-2",
                "max_normalized_algebraic_residual": "1",
                "max_normalized_carrier_residual": "1",
                "max_normalized_interface_residual": "1",
                "max_normalized_positive_ion_residual": "1",
                "max_positive_ion_balance_relative": "1",
                "max_positive_ion_inventory_relative_drift": "1",
                "max_positive_ion_rhs_inventory_relative": "1",
                "max_structured_dense_interface_state_relative_difference": "1",
                "max_structured_dense_log_density_difference": "1",
                "max_structured_dense_positive_ion_relative_difference": "1",
                "max_structured_dense_potential_difference_V": "V",
                "max_terminal_interface_state_relative_error": "1",
                "max_terminal_log_density_error": "1",
                "max_terminal_positive_ion_relative_error": "1",
                "max_terminal_potential_error_V": "V",
                "minimum_interface_state_relative_motion": "1",
                "minimum_positive_ion_relative_motion": "1",
                "mol_numerical_health_passed": "1",
                "mol_reference_success": "1",
                "site_occupancy_admissible": "1",
                "structured_analytic_success": "1",
                "structured_csr_nonzeros_per_node": "1",
                "structured_dense_terminal_interface_state_relative_difference": "1",
                "structured_dense_terminal_log_density_difference": "1",
                "structured_dense_terminal_positive_ion_relative_difference": "1",
                "structured_dense_terminal_potential_difference_V": "V",
                "structured_jacobian_condition_finite": "1",
                "structured_rhs_work_fraction": "1",
                "terminal_densities_positive": "1",
                "terminal_interface_state_relative_error": "1",
                "terminal_interface_occupation": "1",
                "terminal_interface_states_bounded": "1",
                "terminal_log_density_error": "1",
                "terminal_positive_ion_relative_error": "1",
                "terminal_potential_error_V": "V",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "dense_residual_evaluations": dense.total_residual_evaluations,
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": grid.size,
                    "initial_positive_ion_inventory_m2": initial_inventory,
                    "initial_state_sha256": initial.state_sha256,
                    "maximum_structured_jacobian_condition": maximum_condition,
                    "minimum_cross_occupation_margin": minimum_cross_margin,
                    "minimum_interface_density_margin_m3": (minimum_interface_density),
                    "minimum_interface_dos_margin_m3": minimum_interface_dos,
                    "minimum_positive_ion_occupation_margin": (minimum_ion_occupation),
                    "minimum_projection_occupation_margin": (minimum_projection_margin),
                    "minimum_srh_occupancy_margin": minimum_srh_margin,
                    "mol_interface_normalized_residual": (
                        mol_interface.normalized_residual
                    ),
                    "mol_nfev": int(mol.nfev),
                    "structured_residual_evaluations": (
                        structured.total_residual_evaluations
                    ),
                    "time_steps": time_steps,
                    "trajectory_sha256": {
                        "dense": dense.trajectory_sha256,
                        "structured": structured.trajectory_sha256,
                    },
                },
            },
        }
    )


__all__ = ["run_single_ion_algebraic_interface_dae_transient"]

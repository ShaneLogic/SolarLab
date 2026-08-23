from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
)
from perovskite_sim.experiments.ion_aware_structured_jacobian import (
    build_ion_aware_structured_jacobian_protocol,
    run_ion_aware_structured_jacobian_comparison,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import InterfaceDefect


def _solve_comparison(
    stack,
    *,
    grid_points=12,
    frequencies=None,
    state_step=1.0e-5,
    voltage_step=1.0e-5,
):
    x = build_electrical_grid(stack, grid_points)
    dc_state = solve_ion_aware_dc(
        x,
        stack,
        build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
        ),
    )
    impedance_protocol = build_ion_aware_impedance_protocol(
        dc_state,
        (
            np.array([1.0e-3, 1.0, 1.0e3])
            if frequencies is None
            else np.asarray(frequencies, dtype=float)
        ),
        state_step=state_step,
        voltage_step=voltage_step,
    )
    structured_protocol = build_ion_aware_structured_jacobian_protocol(
        impedance_protocol
    )
    return run_ion_aware_structured_jacobian_comparison(
        x,
        stack,
        impedance_protocol,
        structured_protocol,
        dc_state=dc_state,
    )


def test_single_ion_structured_operator_matches_full_poisson_reference():
    result = _solve_comparison(
        load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    )
    certificate = result.certificate

    assert certificate.numerically_certified
    assert certificate.max_poisson_backward_error < 1.0e-12
    assert certificate.rate_jacobian.max_relative_error < 5.0e-6
    assert certificate.conduction_jacobian.max_relative_error < 8.0e-5
    assert certificate.displacement_jacobian.max_relative_error < 5.0e-6
    assert certificate.analytic_transport_conduction_jacobian.passed
    assert certificate.analytic_bulk_reaction_rate_jacobian.passed
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert certificate.analytic_contact_rate_jacobian.passed
    assert all(
        item.jacobian.passed
        for item in certificate.analytic_transport_components
    )
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5
    assert certificate.max_structured_face_spread < (
        result.reference.protocol.max_relative_face_spread
    )
    assert result.structured.mass_matrix.shape == (
        result.reference.coordinate_layout.size,
        result.reference.coordinate_layout.size,
    )


def test_dual_ion_structured_operator_covers_both_charge_signs():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layers = []
    for layer in stack.layers:
        params = layer.params
        if layer.role == "absorber":
            params = replace(
                params,
                D_ion_neg=params.D_ion,
                P0_neg=params.P0,
                P_lim_neg=params.P_lim,
            )
        layers.append(replace(layer, params=params))
    result = _solve_comparison(replace(stack, layers=tuple(layers)))
    certificate = result.certificate
    components = {item.name: item for item in certificate.current_components}

    assert certificate.numerically_certified
    assert "positive_ion" in components
    assert "negative_ion" in components
    assert components["positive_ion"].jacobian.passed
    assert components["negative_ion"].jacobian.passed
    assert all(
        item.jacobian.passed
        for item in certificate.analytic_transport_components
    )
    assert certificate.analytic_bulk_reaction_rate_jacobian.passed
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert certificate.analytic_contact_rate_jacobian.passed
    assert result.reference.coordinate_layout.negative_ion_state_indices
    assert result.structured.current_components[-1].name == "negative_ion"
    assert certificate.rate_jacobian.max_relative_error < 5.0e-6
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6


def test_n61_adaptive_stencils_resolve_strong_columns_and_bound_weak_ones():
    result = _solve_comparison(
        load_device_from_yaml("configs/ionmonger_benchmark.yaml"),
        grid_points=60,
        frequencies=np.array([1.0e-4, 1.0, 1.0e6]),
    )
    certificate = result.certificate
    steps = result.poisson_sensitivity.state_steps

    assert certificate.numerically_certified
    assert np.min(steps) == result.protocol.minimum_state_step
    assert np.max(steps) == result.protocol.maximum_state_step
    assert np.min(steps) < np.median(steps) < np.max(steps)
    assert certificate.rate_jacobian.max_relative_error < 1.0e-6
    assert certificate.conduction_jacobian.passed
    assert not certificate.conduction_jacobian.failed_columns
    assert certificate.conduction_jacobian.max_group_normalized_error < 1.0e-6
    assert certificate.analytic_transport_conduction_jacobian.passed
    assert certificate.analytic_bulk_reaction_rate_jacobian.passed
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert certificate.analytic_contact_rate_jacobian.passed
    assert (
        certificate.analytic_transport_conduction_jacobian
        .max_group_normalized_error
        < 1.0e-6
    )
    assert certificate.displacement_jacobian.passed
    assert not certificate.displacement_jacobian.failed_columns
    assert certificate.displacement_jacobian.max_group_normalized_error < 1.0e-6
    assert certificate.conduction_jacobian.bounded_weak_columns
    assert all(
        item.jacobian.bounded_weak_columns
        for item in certificate.current_components
    )
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6


def test_n91_weak_cross_couplings_pass_the_group_normalized_error_gate():
    result = _solve_comparison(
        load_device_from_yaml("configs/ionmonger_benchmark.yaml"),
        grid_points=90,
        frequencies=np.array([1.0e-4, 1.0, 1.0e6]),
    )
    certificate = result.certificate
    components = {item.name: item for item in certificate.current_components}

    assert certificate.numerically_certified
    assert certificate.max_poisson_backward_error < 2.0e-12
    assert not certificate.conduction_jacobian.failed_columns
    assert certificate.conduction_jacobian.absolute_bounded_columns
    assert not components["hole"].jacobian.failed_columns
    assert components["hole"].jacobian.absolute_bounded_columns
    assert certificate.conduction_jacobian.max_group_normalized_error < 1.0e-6
    assert components["hole"].jacobian.max_group_normalized_error < 1.0e-6
    assert certificate.analytic_transport_conduction_jacobian.passed
    assert certificate.analytic_bulk_reaction_rate_jacobian.passed
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert certificate.analytic_contact_rate_jacobian.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6


def test_selective_contact_structured_operator_replaces_all_boundary_rate_blocks():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    stack = replace(
        base,
        S_n_left=1.0e-3,
        S_p_left=1.0e3,
        S_n_right=1.0e3,
        S_p_right=1.0e-3,
    )
    result = _solve_comparison(stack)
    certificate = result.certificate
    contact = result.analytic_contact

    assert certificate.numerically_certified
    assert contact.active_channels == (
        "electron_left",
        "electron_right",
        "hole_left",
        "hole_right",
    )
    assert certificate.analytic_contact_rate_jacobian.passed
    assert certificate.analytic_contact_rate_voltage_derivative.passed
    assert not certificate.analytic_contact_rate_jacobian.failed_columns
    assert (
        certificate.analytic_contact_rate_jacobian.max_group_normalized_error
        < 1.0e-6
    )
    assert certificate.rate_jacobian.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6


def test_field_mobility_structured_operator_closes_ct_and_pf_chain_rule():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layers = []
    for layer in base.layers:
        params = layer.params
        if layer.role == "HTL":
            params = replace(params, pf_gamma_p=3.0e-4)
        elif layer.role == "absorber":
            params = replace(
                params,
                v_sat_n=1.0e5,
                v_sat_p=1.0e5,
                ct_beta_n=2.0,
                ct_beta_p=2.0,
            )
        layers.append(replace(layer, params=params))
    result = _solve_comparison(
        replace(base, layers=tuple(layers)),
        grid_points=5,
        state_step=4.0e-7,
    )
    certificate = result.certificate
    mobility = result.analytic_transport.field_mobility

    assert certificate.numerically_certified
    assert mobility.active
    assert np.any(mobility.electron_field_derivative_m3_V2_s != 0.0)
    assert np.any(mobility.hole_field_derivative_m3_V2_s != 0.0)
    assert certificate.analytic_electron_field_mobility_derivative.passed
    assert certificate.analytic_hole_field_mobility_derivative.passed
    assert certificate.analytic_transport_conduction_jacobian.passed
    assert (
        result.poisson_sensitivity.max_nonsmooth_state_field_stencil_fraction
        <= result.protocol.max_nonsmooth_field_stencil_fraction
    )
    assert (
        result.poisson_sensitivity.max_nonsmooth_voltage_field_stencil_fraction
        <= result.protocol.max_nonsmooth_field_stencil_fraction
    )
    assert certificate.rate_jacobian.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5


def test_cross_node_interface_srh_closes_the_clamp_inactive_tangent():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    defects = list(base.interface_defects)
    defects.extend([None] * (len(base.interfaces) - len(defects)))
    defects[-1] = InterfaceDefect(
        E_t_eV=0.8,
        calibration_factor=1.0e-10,
    )
    result = _solve_comparison(
        replace(base, interface_defects=tuple(defects))
    )
    certificate = result.certificate
    reaction = result.analytic_interface_reaction
    interface_index = len(reaction.interface_nodes) - 1

    assert certificate.numerically_certified
    assert reaction.cross_node_interface_indices == (interface_index,)
    assert (
        reaction.electron_evaluation_nodes[interface_index]
        == reaction.interface_nodes[interface_index] + 1
    )
    assert (
        reaction.hole_evaluation_nodes[interface_index]
        == reaction.interface_nodes[interface_index] - 1
    )
    assert reaction.minimum_cross_node_clamp_margin_m2_s > 0.0
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert not (
        certificate.analytic_interface_reaction_rate_jacobian.failed_columns
    )
    assert (
        certificate.analytic_interface_reaction_rate_jacobian
        .max_group_normalized_error
        < 1.0e-6
    )
    assert certificate.rate_jacobian.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5


def test_projected_cross_node_interface_srh_closes_the_smooth_tangent():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    defects = list(base.interface_defects)
    defects.extend([None] * (len(base.interfaces) - len(defects)))
    defects[-1] = InterfaceDefect(
        E_t_eV=0.8,
        calibration_factor=1.0e-10,
    )
    result = _solve_comparison(
        replace(
            base,
            interface_defects=tuple(defects),
            interface_plane_projection=True,
        )
    )
    certificate = result.certificate
    reaction = result.analytic_interface_reaction
    interface_index = len(reaction.interface_nodes) - 1

    assert certificate.numerically_certified
    assert reaction.cross_node_interface_indices == (interface_index,)
    assert reaction.projected_interface_indices == (interface_index,)
    assert reaction.minimum_cross_node_clamp_margin_m2_s > 0.0
    assert reaction.minimum_projection_exponent_cap_margin > 0.0
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert (
        certificate.analytic_interface_reaction_rate_voltage_derivative.passed
    )
    assert not (
        certificate.analytic_interface_reaction_rate_jacobian.failed_columns
    )
    assert (
        certificate.analytic_interface_reaction_rate_jacobian
        .max_group_normalized_error
        < 1.0e-6
    )
    assert certificate.rate_jacobian.passed
    assert certificate.rate_voltage_derivative.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5


def test_shared_occupancy_interface_srh_closes_the_positive_density_tangent():
    base = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    defects = list(base.interface_defects)
    defects.extend([None] * (len(base.interfaces) - len(defects)))
    defects[-1] = InterfaceDefect(
        E_t_eV=0.8,
        calibration_factor=1.0e-10,
    )
    result = _solve_comparison(
        replace(
            base,
            interface_defects=tuple(defects),
            interface_shared_occupancy=True,
        )
    )
    certificate = result.certificate
    reaction = result.analytic_interface_reaction
    interface_index = len(reaction.interface_nodes) - 1

    assert certificate.numerically_certified
    assert reaction.cross_node_interface_indices == (interface_index,)
    assert reaction.shared_occupancy_interface_indices == (interface_index,)
    assert not reaction.projected_interface_indices
    assert reaction.minimum_cross_node_clamp_margin_m2_s > 0.0
    assert reaction.minimum_shared_density_floor_margin_m3 > 0.0
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert (
        certificate.analytic_interface_reaction_rate_voltage_derivative.passed
    )
    assert not (
        certificate.analytic_interface_reaction_rate_jacobian.failed_columns
    )
    assert (
        certificate.analytic_interface_reaction_rate_jacobian
        .max_group_normalized_error
        < 1.0e-6
    )
    assert certificate.rate_jacobian.passed
    assert certificate.rate_voltage_derivative.passed
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5

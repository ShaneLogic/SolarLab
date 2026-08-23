from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.interface_plane import solve_interface_states_live_qss
from perovskite_sim.solver.dae import DAECapabilityError
from perovskite_sim.solver.dae_ion_interface_states import (
    build_single_ion_algebraic_interface_consistent_initial_condition,
    build_single_ion_algebraic_interface_dae,
    finite_difference_derivative_jacobian,
    finite_difference_state_jacobian,
    project_single_ion_algebraic_interface_state,
)
from perovskite_sim.solver.dae_interface_states import (
    build_algebraic_interface_state_dae,
    prepare_algebraic_interface_material,
)
from perovskite_sim.solver.dae_ions import build_single_positive_ion_dae
from perovskite_sim.solver.mol import StateVec, assemble_rhs, build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium


def _stack(
    *,
    mobile_ions: bool = True,
    dual_ions: bool = False,
    interface_defect: InterfaceDefect | None = None,
    selective_contact: bool = False,
    field_mobility: bool = False,
) -> DeviceStack:
    left = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=1.0e-16 if mobile_ions else 0.0,
        P_lim=1.0e24,
        P0=1.0e22 if mobile_ions else 0.0,
        D_ion_neg=2.0e-17 if dual_ions else 0.0,
        P0_neg=5.0e21 if dual_ions else 0.0,
        P_lim_neg=1.0e24,
        ni=1.0e12,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e12,
        p1=1.0e12,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
        v_sat_n=1.0e5 if field_mobility else 0.0,
    )
    right = replace(left, chi=4.1)
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, left, role="absorber"),
            LayerSpec("right", 1.0e-7, right, role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        interface_defects=(interface_defect,) if interface_defect else (),
        V_bi=0.0,
        Phi=0.0,
        mode="full",
        S_n_left=1.0e3 if selective_contact else None,
    )


def _grid(stack: DeviceStack, intervals: int = 4) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        alpha=1.0,
    )


def _problem(*, intervals: int = 4, V_app_V: float = 0.01):
    stack = _stack()
    grid = _grid(stack, intervals)
    reference = solve_equilibrium(grid, stack)
    model = build_single_ion_algebraic_interface_dae(
        grid,
        stack,
        reference,
        V_app_V=V_app_V,
        carrier_reference_time_s=1.0e-7,
        ion_reference_time_s=1.0,
    )
    return grid, stack, reference, model


def test_layout_classifies_carrier_ion_interface_and_poisson_rows():
    grid, _stack_value, _reference, model = _problem()
    layout = model.layout

    assert layout.interface_count == 1
    assert layout.interface_state_count == 4
    assert layout.size == 4 * grid.size + 4
    assert np.count_nonzero(layout.differential_mask) == 3 * grid.size - 4
    assert np.count_nonzero(layout.algebraic_mask) == grid.size + 8
    assert np.all(layout.differential_mask[layout.positive_ion_slice])
    assert not np.any(layout.differential_mask[layout.interface_slice])
    assert not np.any(layout.differential_mask[layout.potential_slice])
    assert not layout.differential_mask.flags.writeable


def test_reference_coordinate_recovers_all_physical_state_blocks():
    grid, _stack_value, reference, model = _problem()
    n, p, positive_ion, interface_state, _phi = model.physical_fields(
        np.zeros(model.layout.size)
    )
    packed = StateVec.unpack(reference, grid.size)

    np.testing.assert_allclose(n, packed.n, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p, packed.p, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(positive_ion, packed.P, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(
        interface_state,
        model.layout.interface_reference_m3,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert np.all(positive_ion > 0.0)
    assert np.all(positive_ion < model.layout.positive_ion_site_limit_m3)
    assert np.all(interface_state > 0.0)
    assert np.all(interface_state < model.layout.interface_capacity_m3)


def test_consistent_initial_condition_certifies_every_row_and_inventory():
    _grid_value, _stack_value, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    report = initial.report

    assert initial.certified
    assert report.max_normalized_carrier_residual < 1.0e-12
    assert report.max_normalized_positive_ion_residual < 1.0e-12
    assert report.max_normalized_interface_residual < 1.0e-9
    assert report.max_normalized_differential_residual < 1.0e-12
    assert report.max_normalized_algebraic_residual < 1.0e-9
    assert abs(report.positive_ion_inventory_residual_m2_s) < 1.0e-20
    assert abs(report.positive_ion_rhs_inventory_rate_m2_s) < 1.0e-20
    assert np.max(np.abs(report.interface_bulk_flux_m2_s)) > 1.0e10
    assert len(initial.state_sha256) == 64
    for value in (
        initial.coordinate,
        initial.derivative,
        initial.physical_state,
        initial.interface_state_m3,
        initial.potential_V,
        report.normalized_residual,
    ):
        assert not value.flags.writeable


def test_interface_and_poisson_rows_are_algebraic_but_ion_rows_are_not():
    _grid_value, _stack_value, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    changed = np.array(initial.derivative, copy=True)
    changed[model.layout.interface_slice] = np.array([3.0, -2.0, 5.0, -7.0])
    changed[model.layout.potential_slice] = np.linspace(
        -11.0,
        13.0,
        model.layout.node_count,
    )
    baseline = model.residual(initial.coordinate, initial.derivative)
    algebraic_changed = model.residual(initial.coordinate, changed)
    np.testing.assert_array_equal(algebraic_changed, baseline)

    changed[model.layout.positive_ion_slice.start] += 1.0
    ion_changed = model.residual(initial.coordinate, changed)
    assert not np.array_equal(
        ion_changed[model.layout.positive_ion_slice],
        baseline[model.layout.positive_ion_slice],
    )


def test_exact_derivative_jacobian_matches_independent_central_difference():
    _grid_value, _stack_value, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    analytic = model.derivative_jacobian(initial.coordinate)
    reference = finite_difference_derivative_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(analytic, reference, rtol=2.0e-10, atol=1.0e-12)
    assert np.all(analytic[model.layout.interface_slice] == 0.0)
    assert np.all(analytic[model.layout.potential_slice] == 0.0)


def test_boundary_and_ion_aware_poisson_rows_match_state_stencil():
    grid, _stack_value, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    analytic = model.boundary_poisson_state_jacobian(initial.coordinate)
    reference = finite_difference_state_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
        relative_step=2.0e-6,
    )
    rows = np.concatenate(
        (
            np.array([0, grid.size - 1, grid.size, 2 * grid.size - 1]),
            np.arange(
                model.layout.potential_slice.start,
                model.layout.potential_slice.stop,
            ),
        )
    )

    np.testing.assert_allclose(
        analytic[rows],
        reference[rows],
        rtol=2.0e-6,
        atol=2.0e-9,
    )
    poisson_interior = np.arange(
        model.layout.potential_slice.start + 1,
        model.layout.potential_slice.stop - 1,
    )
    assert np.any(analytic[poisson_interior, model.layout.positive_ion_slice] != 0.0)


def test_explicit_interface_response_matches_nested_qss_with_mobile_ions():
    grid, _stack_value, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    n, p, positive_ion, interface_state, phi = model.physical_fields(initial.coordinate)
    explicit = model.interface_response(n, p, interface_state, phi)
    nested = solve_interface_states_live_qss(
        model.material,
        model.stack,
        n,
        p,
        phi,
        V_app=model.V_app_V,
        v_th_eff=model.material.iface_state_v_th,
        cross_transmission=model.material.iface_qss_cross_transmission,
        interface_transport_model=model.material.iface_qss_transport_model,
        residual_tolerance=model.interface_residual_tolerance,
    )
    np.testing.assert_allclose(explicit.state_m3, nested.state_m3, rtol=1.0e-13)

    packed = StateVec.pack(n, p, positive_ion)
    explicit_rhs = StateVec.unpack(
        assemble_rhs(
            0.0,
            packed,
            grid,
            model.stack,
            model.material,
            illuminated=False,
            V_app=model.V_app_V,
            phi_frozen=phi,
            interface_qss_result=explicit,
        ),
        grid.size,
    )
    nested_rhs = StateVec.unpack(
        assemble_rhs(
            0.0,
            packed,
            grid,
            model.stack,
            model.material,
            illuminated=False,
            V_app=model.V_app_V,
            phi_frozen=phi,
        ),
        grid.size,
    )
    np.testing.assert_allclose(
        explicit_rhs.n / model.layout.electron_rate_scale_m3_s,
        nested_rhs.n / model.layout.electron_rate_scale_m3_s,
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        explicit_rhs.p / model.layout.hole_rate_scale_m3_s,
        nested_rhs.p / model.layout.hole_rate_scale_m3_s,
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_array_equal(explicit_rhs.P, nested_rhs.P)


def test_projected_ion_perturbation_closes_poisson_interface_and_inventory():
    grid, _stack_value, _reference, model = _problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = np.linspace(
        -0.05,
        0.05,
        grid.size,
    )
    projected = project_single_ion_algebraic_interface_state(model, coordinate)
    derivative = model.compatible_derivative(projected)
    report = model.residual_report(projected, derivative)

    assert report.max_normalized_differential_residual < 1.0e-12
    assert report.max_normalized_interface_residual < 1.0e-9
    assert (
        np.max(np.abs(report.poisson_residual_C_m2 / model.layout.poisson_scale_C_m2))
        < 2.0e-14
    )
    assert abs(report.positive_ion_rhs_inventory_rate_m2_s) < 1.0e-12
    _n, _p, positive_ion, _interface, _phi = model.physical_fields(projected)
    assert dual_cell_integral(grid, positive_ion) > 0.0


def test_logit_mappings_fail_closed_at_capacity_surfaces():
    _grid_value, _stack_value, _reference, model = _problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice.start] = 1.0e3
    with pytest.raises(ValueError, match="positive-ion logit coordinate saturated"):
        model.physical_fields(coordinate)

    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.interface_slice.start] = -1.0e3
    with pytest.raises(ValueError, match="interface-state logit coordinate saturated"):
        model.physical_fields(coordinate)
    with pytest.raises(ValueError, match="strictly inside"):
        model.interface_coordinates_from_state(
            np.zeros(model.layout.interface_state_count)
        )


@pytest.mark.parametrize(
    ("stack", "message"),
    [
        (_stack(mobile_ions=False), "positive mobile ion"),
        (_stack(dual_ions=True), "dual mobile ions"),
        (_stack(selective_contact=True), "selective contacts"),
        (_stack(field_mobility=True), "field-dependent mobility"),
        (
            _stack(interface_defect=InterfaceDefect(E_t_eV=0.5)),
            "InterfaceDefect",
        ),
    ],
)
def test_capability_rejects_unimplemented_combined_physics(stack, message):
    grid = _grid(stack)
    material = build_material_arrays(grid, stack)
    if material.has_dual_ions:
        reference = StateVec.pack(
            np.full(grid.size, max(material.n_L, 1.0)),
            np.full(grid.size, max(material.p_L, 1.0)),
            np.maximum(material.P_ion0, 1.0),
            np.maximum(material.P_ion0_neg, 1.0),
        )
    else:
        reference = StateVec.pack(
            np.full(grid.size, max(material.n_L, 1.0)),
            np.full(grid.size, max(material.p_L, 1.0)),
            np.maximum(material.P_ion0, 1.0),
        )
    with pytest.raises(DAECapabilityError, match=message):
        build_single_ion_algebraic_interface_dae(
            grid,
            stack,
            reference,
            material=material,
        )


def test_capability_rejects_cross_node_dynamic_and_charged_interface_modes():
    grid, stack, reference, _model = _problem()
    material = prepare_algebraic_interface_material(grid, stack)
    variants = (
        (
            replace(
                material,
                interface_eval_node_n=(material.interface_nodes[0] + 1,),
            ),
            "cross-node",
        ),
        (replace(material, N_iface_state=1), "dynamic interface"),
        (replace(material, iface_qss_two_sided_trace=True), "two-sided trace"),
        (replace(material, iface_state_charge=1.0), "charge closure"),
    )
    for changed, message in variants:
        with pytest.raises(DAECapabilityError, match=message):
            build_single_ion_algebraic_interface_dae(
                grid,
                stack,
                reference,
                material=changed,
            )


def test_existing_isolated_topologies_still_reject_the_combined_stack():
    grid, stack, reference, _model = _problem()
    with pytest.raises(DAECapabilityError, match="mobile ions"):
        build_algebraic_interface_state_dae(grid, stack, reference)
    with pytest.raises(DAECapabilityError, match="one electrical layer"):
        build_single_positive_ion_dae(grid, stack, reference)

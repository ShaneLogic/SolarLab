from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.interface_plane import (
    compute_interface_srh_occupancy_on_state,
    solve_interface_states_live_qss,
)
from perovskite_sim.solver.dae import DAECapabilityError
from perovskite_sim.solver.dae_interface_states import (
    build_algebraic_interface_consistent_initial_condition,
    build_algebraic_interface_state_dae,
    compatible_derivative,
    finite_difference_derivative_jacobian,
    finite_difference_state_jacobian,
    prepare_algebraic_interface_material,
    project_algebraic_interface_state,
)
from perovskite_sim.solver.mol import StateVec, assemble_rhs, build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium


def _stack(
    *,
    interface_defect: InterfaceDefect | None = None,
    mobile_ions: bool = False,
    selective_contact: bool = False,
) -> DeviceStack:
    left = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=1.0e-16 if mobile_ions else 0.0,
        P_lim=1.0e24,
        P0=1.0e22 if mobile_ions else 0.0,
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


def _model(*, intervals: int = 4, V_app_V: float = 0.01):
    stack = _stack()
    grid = _grid(stack, intervals)
    return build_algebraic_interface_state_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=V_app_V,
        carrier_reference_time_s=1.0e-7,
    )


def test_layout_marks_only_interior_carriers_differential():
    model = _model()
    layout = model.layout

    assert layout.interface_count == 1
    assert layout.interface_state_count == 4
    assert layout.size == 3 * layout.node_count + 4
    assert np.count_nonzero(layout.differential_mask) == 2 * (
        layout.node_count - 2
    )
    assert np.count_nonzero(layout.algebraic_mask) == layout.node_count + 8
    assert not np.any(layout.differential_mask[layout.interface_slice])
    assert not np.any(layout.differential_mask[layout.potential_slice])


def test_consistent_initial_condition_certifies_all_row_classes():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)

    assert initial.certified
    assert initial.report.max_normalized_carrier_residual < 1.0e-12
    assert initial.report.max_normalized_interface_residual < 1.0e-9
    assert initial.report.max_normalized_algebraic_residual < 1.0e-9
    assert initial.report.max_normalized_residual < 1.0e-9
    assert np.max(np.abs(initial.report.interface_bulk_flux_m2_s)) > 1.0e10
    assert np.all(initial.interface_state_m3 > 0.0)
    assert np.all(
        initial.interface_state_m3 < model.layout.interface_capacity_m3
    )
    assert len(initial.state_sha256) == 64
    for value in (
        initial.coordinate,
        initial.derivative,
        initial.physical_state,
        initial.interface_state_m3,
        initial.potential_V,
        initial.report.normalized_residual,
    ):
        assert not value.flags.writeable


def test_interface_rows_are_algebraic_and_ignore_coordinate_rate():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    changed = np.array(initial.derivative, copy=True)
    changed[model.layout.interface_slice] = np.array([3.0, -2.0, 5.0, -7.0])
    changed[model.layout.potential_slice] = np.linspace(
        -11.0,
        13.0,
        model.layout.node_count,
    )

    baseline = model.residual(initial.coordinate, initial.derivative)
    perturbed = model.residual(initial.coordinate, changed)

    np.testing.assert_array_equal(perturbed, baseline)
    tangent = model.derivative_jacobian(initial.coordinate)
    assert np.all(tangent[model.layout.interface_slice] == 0.0)
    assert np.all(tangent[model.layout.potential_slice] == 0.0)
    assert np.all(tangent[:, model.layout.interface_slice] == 0.0)
    assert np.all(tangent[:, model.layout.potential_slice] == 0.0)


def test_exact_derivative_jacobian_matches_independent_central_difference():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    analytic = model.derivative_jacobian(initial.coordinate)
    reference = finite_difference_derivative_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(analytic, reference, rtol=2.0e-10, atol=1.0e-12)


def test_boundary_and_poisson_rows_match_state_finite_difference():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    analytic = model.boundary_poisson_state_jacobian(initial.coordinate)
    reference = finite_difference_state_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
        relative_step=2.0e-6,
    )
    rows = np.concatenate(
        (
            np.array(
                [
                    0,
                    model.layout.node_count - 1,
                    model.layout.node_count,
                    2 * model.layout.node_count - 1,
                ]
            ),
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


def test_explicit_interface_state_matches_nested_qss_carrier_response():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    n, p, state, phi = model.physical_fields(initial.coordinate)
    explicit = model.interface_response(n, p, state, phi)
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
    bulk_scale = model.layout.interface_flux_scale_m2_s
    assert np.max(np.abs(explicit.bulk_flux_m2_s - nested.bulk_flux_m2_s) / bulk_scale) < (
        1.0e-9
    )

    packed = StateVec.pack(n, p, np.zeros(model.layout.node_count))
    explicit_rhs = assemble_rhs(
        0.0,
        packed,
        model.grid_m,
        model.stack,
        model.material,
        illuminated=False,
        V_app=model.V_app_V,
        phi_frozen=phi,
        interface_qss_result=explicit,
    )
    nested_rhs = assemble_rhs(
        0.0,
        packed,
        model.grid_m,
        model.stack,
        model.material,
        illuminated=False,
        V_app=model.V_app_V,
        phi_frozen=phi,
    )
    explicit_state = StateVec.unpack(explicit_rhs, model.layout.node_count)
    nested_state = StateVec.unpack(nested_rhs, model.layout.node_count)
    electron_error = np.max(
        np.abs(explicit_state.n - nested_state.n)
        / model.layout.electron_rate_scale_m3_s
    )
    hole_error = np.max(
        np.abs(explicit_state.p - nested_state.p)
        / model.layout.hole_rate_scale_m3_s
    )
    assert max(electron_error, hole_error) < 1.0e-9


def test_interface_state_balance_uses_same_bulk_cross_and_srh_fluxes():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    n, p, state, phi = model.physical_fields(initial.coordinate)
    response = model.interface_response(n, p, state, phi)
    srh = compute_interface_srh_occupancy_on_state(
        state,
        model.stack,
        model.material,
    )

    np.testing.assert_allclose(
        response.state_flux_m2_s,
        response.bulk_flux_m2_s + response.cross_flux_m2_s + srh,
        rtol=0.0,
        atol=0.0,
    )
    assert srh[0] + srh[2] == pytest.approx(srh[1] + srh[3], rel=1.0e-14)


def test_interface_state_perturbation_moves_only_algebraic_manifold():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    perturbed = np.array(initial.coordinate, copy=True)
    perturbed[model.layout.interface_slice.start] += 0.1
    derivative = compatible_derivative(model, perturbed)
    report = model.residual_report(perturbed, derivative)

    assert report.max_normalized_carrier_residual < 1.0e-12
    assert report.max_normalized_interface_residual > 1.0e-5
    assert report.max_normalized_algebraic_residual > 1.0e-5


def test_projection_restores_interface_and_poisson_algebraic_rows():
    model = _model()
    coordinate = np.zeros(model.layout.size)
    coordinate[2] = 0.02
    coordinate[model.layout.node_count + 2] = -0.03
    projected = project_algebraic_interface_state(model, coordinate)
    report = model.residual_report(
        projected,
        compatible_derivative(model, projected),
    )

    assert report.max_normalized_interface_residual < 1.0e-9
    assert np.max(
        np.abs(
            report.poisson_residual_C_m2
            / model.layout.poisson_scale_C_m2
        )
    ) < 2.0e-15


def test_interface_coordinate_mapping_enforces_strict_dos_bounds():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    coordinate = np.array(initial.coordinate, copy=True)
    coordinate[model.layout.interface_slice.start] = 1.0e3

    with pytest.raises(ValueError, match="saturated"):
        model.physical_fields(coordinate)
    with pytest.raises(ValueError, match="strictly inside"):
        model.interface_coordinates_from_state(
            np.zeros(model.layout.interface_state_count)
        )
    with pytest.raises(ValueError, match="strictly inside"):
        model.interface_coordinates_from_state(
            np.array(model.layout.interface_capacity_m3, copy=True)
        )


def test_capability_rejects_interface_defect_and_cross_node_sampling():
    defect_stack = _stack(
        interface_defect=InterfaceDefect(E_t_eV=0.5, N_t_cm2=1.0e10)
    )
    grid = _grid(defect_stack)
    with pytest.raises(DAECapabilityError, match="InterfaceDefect"):
        build_algebraic_interface_state_dae(
            grid,
            defect_stack,
            solve_equilibrium(grid, defect_stack),
        )

    stack = _stack()
    grid = _grid(stack)
    material = prepare_algebraic_interface_material(grid, stack)
    cross_node = replace(
        material,
        interface_eval_node_n=(material.interface_nodes[0] + 1,),
    )
    with pytest.raises(DAECapabilityError, match="cross-node"):
        build_algebraic_interface_state_dae(
            grid,
            stack,
            solve_equilibrium(grid, stack),
            material=cross_node,
        )


@pytest.mark.parametrize(
    ("stack", "message"),
    [
        (_stack(mobile_ions=True), "mobile ions"),
        (_stack(selective_contact=True), "selective contacts"),
    ],
)
def test_capability_rejects_unsupported_bulk_or_contact_topology(stack, message):
    grid = _grid(stack)
    with pytest.raises(DAECapabilityError, match=message):
        build_algebraic_interface_state_dae(
            grid,
            stack,
            solve_equilibrium(grid, stack),
        )


def test_material_preparation_rejects_dynamic_and_two_sided_state_topologies():
    stack = _stack()
    grid = _grid(stack)
    material = build_material_arrays(grid, stack)

    with pytest.raises(DAECapabilityError, match="dynamic interface"):
        prepare_algebraic_interface_material(
            grid,
            stack,
            material=replace(material, N_iface_state=1),
        )
    with pytest.raises(DAECapabilityError, match="two-sided trace"):
        prepare_algebraic_interface_material(
            grid,
            stack,
            material=replace(material, iface_qss_two_sided_trace=True),
        )

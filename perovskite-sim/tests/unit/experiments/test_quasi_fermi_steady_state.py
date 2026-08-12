"""Focused contracts for the opt-in quasi-Fermi steady-state solver."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim.experiments.quasi_fermi_steady_state as qf_module

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    _QuasiFermiSystem,
    _density_from_log,
    _prepare_two_sided_material,
    _regrid_edge_drops,
    build_two_sided_trace_grid,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import (
    DEDUPLICATED_QSS,
    TWO_SIDED_TRACE,
)
from perovskite_sim.solver.mol import build_material_arrays


def _uniform_stack(*, mobile_ions: bool = False) -> DeviceStack:
    params = MaterialParams(
        eps_r=11.7,
        mu_n=0.1,
        mu_p=0.05,
        D_ion=1.0e-16 if mobile_ions else 0.0,
        P_lim=1.0e24,
        P0=1.0e22 if mobile_ions else 0.0,
        ni=1.0e16,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e16,
        p1=1.0e16,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.05,
        Eg=1.12,
    )
    return DeviceStack(
        layers=(LayerSpec("si", 1.0e-6, params, role="absorber"),),
        V_bi=0.0,
        Phi=0.0,
        interfaces=(),
        mode="legacy",
    )


def _two_layer_interface_stack() -> DeviceStack:
    left = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=0.0,
        P_lim=1.0e24,
        P0=0.0,
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
        interfaces=((0.0, 0.0),),
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )


def test_uniform_dark_equilibrium_is_certified():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    result = solve_quasi_fermi_steady_state(
        x,
        stack,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certified
    assert result.max_normalized_cell_residual < 1.0e-10
    assert result.electron_continuity_bound_A_m2 < 1.0e-10
    assert result.hole_continuity_bound_A_m2 < 1.0e-10
    assert result.face_current_spread_A_m2 < 1.0e-10
    assert result.poisson_residual < 1.0e-10
    assert np.all(np.isfinite(result.y))
    assert np.all(np.isfinite(result.phi))
    assert result.electron_quasi_fermi_reference_V is not None
    assert result.hole_quasi_fermi_reference_V is not None
    assert result.electron_quasi_fermi_increment_V is not None
    assert result.hole_quasi_fermi_increment_V is not None
    assert result.qf_coordinate_system == "nodal_increment"
    assert result.electron_quasi_fermi_edge_drop_V is None
    assert result.hole_quasi_fermi_edge_drop_V is None
    np.testing.assert_allclose(
        result.electron_quasi_fermi_potential_V,
        result.electron_quasi_fermi_reference_V
        + result.electron_quasi_fermi_increment_V,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.hole_quasi_fermi_potential_V,
        result.hole_quasi_fermi_reference_V
        + result.hole_quasi_fermi_increment_V,
        rtol=0.0,
        atol=0.0,
    )

    node_count = len(x)
    mat = build_material_arrays(x, stack)
    log_n = (
        result.electron_quasi_fermi_potential_V + result.phi
        + mat.chi
    ) / mat.V_T_device
    log_p = (
        result.hole_quasi_fermi_potential_V
        - result.phi
        - mat.chi
        - mat.Eg
    ) / mat.V_T_device
    assert np.log(result.y[:node_count]) == pytest.approx(log_n, abs=1.0e-12)
    assert np.log(result.y[node_count : 2 * node_count]) == pytest.approx(
        log_p,
        abs=1.0e-12,
    )


def test_stable_difference_matches_direct_subtraction_away_from_cancellation():
    delta = np.array([-0.7, -0.1, 0.1, 0.7])
    a = np.array([3.0, 4.0, 5.0, 6.0])
    b = a * np.exp(-delta)
    actual = _QuasiFermiSystem._stable_difference(a, b, delta)
    assert actual == pytest.approx(a - b, rel=2.0e-15, abs=1.0e-15)


def test_log_density_is_never_silently_clipped():
    assert _density_from_log(np.array([-99.0, 0.0, 99.0]), context="test") == (
        pytest.approx(np.exp(np.array([-99.0, 0.0, 99.0])))
    )
    with pytest.raises(QuasiFermiSteadyStateError, match="audited exponential"):
        _density_from_log(np.array([101.0]), context="test")


def test_mobile_ions_are_rejected_before_newton():
    stack = _uniform_stack(mobile_ions=True)
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    with pytest.raises(QuasiFermiSteadyStateError, match="mobile ions"):
        solve_quasi_fermi_steady_state(x, stack, V_app=0.0)


def test_thermionic_interface_flux_is_rejected_directly():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    mat = replace(build_material_arrays(x, stack), interface_faces=(0,))
    with pytest.raises(QuasiFermiSteadyStateError, match="thermionic interface"):
        solve_quasi_fermi_steady_state(x, stack, V_app=0.0, mat=mat)


def test_certified_state_warm_starts_a_voltage_sweep():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    voltages = np.array([0.0, 0.01, 0.02])
    sweep = solve_quasi_fermi_jv_sweep(x, stack, voltages)

    assert sweep.certified
    assert sweep.metrics_certified
    assert sweep.voltages_V == pytest.approx(voltages)
    assert sweep.currents_A_m2[0] == pytest.approx(0.0, abs=1.0e-10)
    assert np.all(np.diff(sweep.currents_A_m2) < 0.0)
    assert len(sweep.points) == len(voltages)
    assert all(point.certified for point in sweep.points)
    assert sweep.points[0].illumination_steps == pytest.approx(
        (0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6,
         1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
    )
    assert sweep.points[1].illumination_steps == (1.0,)

    direct = solve_quasi_fermi_steady_state(x, stack, V_app=0.01)
    assert sweep.currents_A_m2[1] == pytest.approx(
        direct.current_A_m2,
        abs=1.0e-10,
    )


def test_certified_short_circuit_seed_is_recertified_directly_at_one_sun():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    short_circuit = solve_quasi_fermi_steady_state(
        x,
        stack,
        V_app=0.0,
    )
    sweep = solve_quasi_fermi_jv_sweep(
        x,
        stack,
        np.array([0.0, 0.01]),
        initial_short_circuit_state=short_circuit,
    )

    assert sweep.points[0].certified
    assert sweep.points[0].illumination_steps == (1.0,)
    assert sweep.points[0].current_A_m2 == pytest.approx(
        short_circuit.current_A_m2,
        abs=1.0e-10,
    )


def test_voltage_bisection_is_explicit_and_retains_only_requested_points(
    monkeypatch,
):
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 4)])
    calls = []

    def fake_solve(grid, device, *, V_app, initial_state=None, **kwargs):
        calls.append((V_app, None if initial_state is None else initial_state.V_app))
        if initial_state is not None and V_app - initial_state.V_app > 0.051:
            raise QuasiFermiSteadyStateError("outside voltage continuation basin")
        return SimpleNamespace(
            V_app=float(V_app),
            current_A_m2=1.0 - 4.0 * float(V_app),
            certified=True,
        )

    monkeypatch.setattr(qf_module, "solve_quasi_fermi_steady_state", fake_solve)
    requested = np.array([0.0, 0.1, 0.2, 0.3])

    sweep = qf_module.solve_quasi_fermi_jv_sweep(
        x,
        stack,
        requested,
        minimum_voltage_step_V=0.025,
    )

    assert sweep.certified
    assert sweep.voltages_V == pytest.approx(requested)
    assert [point.V_app for point in sweep.points] == pytest.approx(requested)
    assert sweep.continuation_bridge_count == 3
    assert sweep.minimum_voltage_step_V == pytest.approx(0.025)
    for expected in ((0.05, 0.0), (0.15, 0.1), (0.25, 0.2)):
        assert any(
            actual == pytest.approx(expected, abs=1.0e-15)
            for actual in calls
        )


def test_voltage_bisection_remains_default_off(monkeypatch):
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 4)])

    def fake_solve(grid, device, *, V_app, initial_state=None, **kwargs):
        if initial_state is not None and V_app - initial_state.V_app > 0.051:
            raise QuasiFermiSteadyStateError("outside voltage continuation basin")
        return SimpleNamespace(
            V_app=float(V_app),
            current_A_m2=1.0 - 4.0 * float(V_app),
            certified=True,
        )

    monkeypatch.setattr(qf_module, "solve_quasi_fermi_steady_state", fake_solve)

    with pytest.raises(QuasiFermiSteadyStateError, match=r"\[0, 0.1\] V"):
        qf_module.solve_quasi_fermi_jv_sweep(
            x,
            stack,
            np.array([0.0, 0.1]),
        )


def test_interface_voltage_sweep_records_nodal_predictor_fallback(monkeypatch):
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 4)])

    def fake_solve(
        grid,
        device,
        *,
        V_app,
        initial_state=None,
        force_nodal_coordinate_predictor=False,
        **kwargs,
    ):
        if (
            initial_state is not None
            and not force_nodal_coordinate_predictor
        ):
            raise QuasiFermiSteadyStateError("direct edge seed missed basin")
        return SimpleNamespace(
            V_app=float(V_app),
            current_A_m2=1.0 - 20.0 * float(V_app),
            certified=True,
        )

    monkeypatch.setattr(qf_module, "solve_quasi_fermi_steady_state", fake_solve)

    sweep = qf_module.solve_quasi_fermi_jv_sweep(
        x,
        stack,
        np.array([0.0, 0.1]),
        interface_boundary=True,
    )

    assert sweep.certified
    assert sweep.continuation_bridge_count == 0
    assert sweep.nodal_predictor_fallback_attempts == 1
    assert sweep.nodal_predictor_fallback_failures == 0


def test_jv_sweep_requires_zero_voltage_for_jsc_extraction():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    with pytest.raises(ValueError, match="start at 0 V"):
        solve_quasi_fermi_jv_sweep(x, stack, np.array([0.01, 0.02]))


def test_uncertified_warm_start_is_rejected():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    result = solve_quasi_fermi_steady_state(
        x,
        stack,
        V_app=0.0,
        illuminated=False,
    )
    with pytest.raises(ValueError, match="physical certificate"):
        solve_quasi_fermi_steady_state(
            x,
            stack,
            V_app=0.01,
            illuminated=False,
            initial_state=replace(result, certified=False),
        )


def test_partial_split_qf_warm_start_is_rejected():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 12)])
    result = solve_quasi_fermi_steady_state(
        x,
        stack,
        V_app=0.0,
        illuminated=False,
    )
    incomplete = replace(result, electron_quasi_fermi_increment_V=None)
    with pytest.raises(ValueError, match="all QF reference/increment arrays"):
        solve_quasi_fermi_steady_state(
            x,
            stack,
            V_app=0.01,
            illuminated=False,
            initial_state=incomplete,
        )


def test_certified_qf_state_can_be_regridded_and_recertified():
    stack = _uniform_stack()
    coarse = multilayer_grid([Layer(stack.layers[0].thickness, 8)])
    fine = multilayer_grid([Layer(stack.layers[0].thickness, 16)])
    coarse_state = solve_quasi_fermi_steady_state(
        coarse,
        stack,
        V_app=0.0,
        illuminated=False,
    )

    fine_state = solve_quasi_fermi_steady_state(
        fine,
        stack,
        V_app=0.0,
        illuminated=False,
        initial_state=coarse_state,
        initial_state_grid=coarse,
    )

    assert fine_state.certified
    assert fine_state.initial_state_regrids == 1
    assert fine_state.max_normalized_cell_residual < 1.0e-10
    assert fine_state.current_A_m2 == pytest.approx(0.0, abs=1.0e-10)


def test_initial_state_grid_requires_a_certified_state():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 8)])
    with pytest.raises(ValueError, match="initial_state_grid requires"):
        solve_quasi_fermi_steady_state(
            x,
            stack,
            V_app=0.0,
            illuminated=False,
            initial_state_grid=x,
        )


def test_edge_coordinates_preserve_sub_ulp_face_drop_and_contact_pins():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 6)])
    system = _QuasiFermiSystem(
        x,
        stack,
        build_material_arrays(x, stack),
        0.0,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    per_carrier = len(x) - 2
    coordinates = np.zeros(2 * per_carrier)
    coordinates[0] = 80.0
    coordinates[1] = 1.0e-16

    z_n, z_p, edge_n, edge_p = system.edge_coordinates_to_increments(
        coordinates
    )

    assert z_n[[0, -1]] == pytest.approx((0.0, 0.0), abs=0.0)
    assert z_p[[0, -1]] == pytest.approx((0.0, 0.0), abs=0.0)
    assert edge_n[1] == 1.0e-16
    assert np.diff(z_n)[1] == 0.0
    assert np.sum(edge_n) == pytest.approx(0.0, abs=1.0e-14)
    assert np.sum(edge_p) == 0.0


def test_exact_edge_drop_regrid_preserves_gradient_and_total_drop():
    source = np.array([0.0, 0.25, 1.0])
    target = np.array([0.0, 0.125, 0.25, 0.5, 1.0])
    drops = np.array([1.0e-16, 2.0])

    mapped = _regrid_edge_drops(source, target, drops)

    assert mapped == pytest.approx(
        (0.5e-16, 0.5e-16, 2.0 / 3.0, 4.0 / 3.0),
        rel=2.0e-15,
        abs=1.0e-30,
    )
    assert np.sum(mapped) == np.sum(drops)


def test_edge_and_nodal_coordinates_agree_when_drops_are_resolvable():
    stack = _uniform_stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 6)])
    system = _QuasiFermiSystem(
        x,
        stack,
        build_material_arrays(x, stack),
        0.0,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    per_carrier = len(x) - 2
    coordinates = np.r_[
        np.linspace(1.0e-4, 5.0e-4, per_carrier),
        np.linspace(-2.0e-4, -6.0e-4, per_carrier),
    ]
    z_n, z_p, _edge_n, _edge_p = (
        system.edge_coordinates_to_increments(coordinates)
    )

    edge = system.evaluate_edge_coordinates(
        coordinates,
        0.0,
        physical_residual=True,
    )
    nodal = system.evaluate(np.r_[z_n, z_p], 0.0)

    np.testing.assert_allclose(edge.residual, nodal.residual, rtol=2.0e-9)
    np.testing.assert_allclose(edge.current_n, nodal.current_n, rtol=2.0e-9)
    np.testing.assert_allclose(edge.current_p, nodal.current_p, rtol=2.0e-9)


def test_interface_exact_edge_drops_warm_start_and_recertify():
    stack = _two_layer_interface_stack()
    x = multilayer_grid(
        [Layer(1.0e-7, 4), Layer(1.0e-7, 4)],
        alpha=(2.0, 2.0),
    )
    result = solve_quasi_fermi_steady_state(
        x,
        stack,
        illuminated=False,
        interface_boundary=True,
    )

    assert result.electron_quasi_fermi_edge_drop_V is not None
    assert result.hole_quasi_fermi_edge_drop_V is not None
    assert result.qf_coordinate_system == "edge_drop"
    assert not result.edge_coordinate_predictor_used
    assert result.electron_quasi_fermi_edge_drop_V.shape == (len(x) - 1,)
    assert result.hole_quasi_fermi_edge_drop_V.shape == (len(x) - 1,)

    repeated = solve_quasi_fermi_steady_state(
        x,
        stack,
        illuminated=False,
        interface_boundary=True,
        initial_state=result,
    )

    assert repeated.certified
    assert repeated.newton_iterations == 0
    np.testing.assert_allclose(
        repeated.electron_quasi_fermi_edge_drop_V,
        result.electron_quasi_fermi_edge_drop_V,
        rtol=0.0,
        atol=2.0e-17,
    )
    np.testing.assert_allclose(
        repeated.hole_quasi_fermi_edge_drop_V,
        result.hole_quasi_fermi_edge_drop_V,
        rtol=0.0,
        atol=2.0e-17,
    )


def test_interface_cross_grid_seed_uses_nodal_predictor_then_edge_certificate():
    stack = _two_layer_interface_stack()
    coarse_grid = multilayer_grid(
        [Layer(1.0e-7, 3), Layer(1.0e-7, 3)],
        alpha=(2.0, 2.0),
    )
    fine_grid = multilayer_grid(
        [Layer(1.0e-7, 5), Layer(1.0e-7, 5)],
        alpha=(2.0, 2.0),
    )
    coarse = solve_quasi_fermi_steady_state(
        coarse_grid,
        stack,
        illuminated=False,
        interface_boundary=True,
    )

    fine = solve_quasi_fermi_steady_state(
        fine_grid,
        stack,
        illuminated=True,
        interface_boundary=True,
        initial_state=coarse,
        initial_state_grid=coarse_grid,
        illumination_steps=(1.0,),
    )

    assert fine.certified
    assert fine.initial_state_regrids == 1
    assert fine.edge_coordinate_predictor_used
    assert fine.qf_coordinate_system == "edge_drop"


def test_interface_warm_start_rejects_partial_exact_edge_state():
    stack = _two_layer_interface_stack()
    x = multilayer_grid(
        [Layer(1.0e-7, 4), Layer(1.0e-7, 4)],
        alpha=(2.0, 2.0),
    )
    result = solve_quasi_fermi_steady_state(
        x,
        stack,
        illuminated=False,
        interface_boundary=True,
    )

    with pytest.raises(ValueError, match="both electron and hole"):
        solve_quasi_fermi_steady_state(
            x,
            stack,
            illuminated=False,
            interface_boundary=True,
            initial_state=replace(
                result,
                hole_quasi_fermi_edge_drop_V=None,
            ),
        )


def test_two_sided_trace_requires_reduced_grid_and_fd_transport():
    stack = _two_layer_interface_stack()
    shared_grid = multilayer_grid(
        [Layer(1.0e-7, 4), Layer(1.0e-7, 4)],
        alpha=(2.0, 2.0),
    )

    with pytest.raises(ValueError, match="currently requires"):
        solve_quasi_fermi_steady_state(
            shared_grid,
            stack,
            illuminated=False,
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
        )
    with pytest.raises(
        QuasiFermiSteadyStateError,
        match="grid without shared interface nodes",
    ):
        solve_quasi_fermi_steady_state(
            shared_grid,
            stack,
            illuminated=False,
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
            interface_transport_model=FERMI_DIRAC_RICHARDSON,
        )


def test_two_sided_trace_grid_has_exact_interface_control_volumes():
    stack = _two_layer_interface_stack()
    shared_grid = multilayer_grid(
        [Layer(1.0e-7, 4), Layer(1.0e-7, 4)],
        alpha=(2.0, 2.0),
    )
    grid = build_two_sided_trace_grid(shared_grid, stack)
    material = build_material_arrays(grid, stack)
    prepared = _prepare_two_sided_material(grid, stack, material)

    assert grid.size == shared_grid.size - 1
    assert not np.any(grid == pytest.approx(1.0e-7))
    assert prepared.iface_qss_two_sided_trace
    assert prepared.interface_faces == ()
    assert prepared.iface_qss_left_nodes == (3,)
    assert prepared.iface_qss_right_nodes == (4,)
    assert prepared.iface_qss_interface_faces == (3,)
    assert np.sum(prepared.dx_cell) == pytest.approx(
        np.sum(material.dx_cell),
        rel=0.0,
        abs=1.0e-22,
    )

    face = prepared.iface_qss_interface_faces[0]
    left = prepared.iface_qss_left_nodes[0]
    right = prepared.iface_qss_right_nodes[0]
    h_left = prepared.iface_qss_left_distances_m[0]
    h_right = prepared.iface_qss_right_distances_m[0]
    expected = EPS_0 / (
        h_left / prepared.eps_r[left]
        + h_right / prepared.eps_r[right]
    )
    assert prepared.poisson_factor is not None
    assert prepared.poisson_factor.C[face] == pytest.approx(expected)


def test_two_sided_trace_full_device_dark_state_is_certified():
    stack = _two_layer_interface_stack()
    shared_grid = multilayer_grid(
        [Layer(1.0e-7, 4), Layer(1.0e-7, 4)],
        alpha=(2.0, 2.0),
    )
    grid = build_two_sided_trace_grid(shared_grid, stack)

    result = solve_quasi_fermi_steady_state(
        grid,
        stack,
        illuminated=False,
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
    )

    assert result.certified
    assert result.interface_topology == TWO_SIDED_TRACE
    assert result.interface_faces == (3,)
    assert result.interface_local_residual < 1.0e-10
    assert result.electron_continuity_bound_A_m2 < 1.0e-4
    assert result.hole_continuity_bound_A_m2 < 1.0e-4
    assert result.face_current_spread_A_m2 < 1.0e-4
    assert result.poisson_residual < 1.0e-8

    with pytest.raises(ValueError, match="same interface topology"):
        solve_quasi_fermi_steady_state(
            grid,
            stack,
            illuminated=False,
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
            interface_transport_model=FERMI_DIRAC_RICHARDSON,
            initial_state=replace(
                result,
                interface_topology=DEDUPLICATED_QSS,
            ),
        )


@pytest.mark.slow
def test_csi_small_grid_has_a_physical_short_circuit_solution():
    stack = load_device_from_yaml(Path("configs/cSi_homojunction.yaml"))
    stack = replace(stack, V_bi=abs(stack.compute_V_bi()))
    electrical = tuple(layer for layer in stack.layers if layer.role != "substrate")
    x = multilayer_grid(
        [
            Layer(electrical[0].thickness, 8),
            Layer(electrical[1].thickness, 32),
        ],
        alpha=(2.0, 3.0),
    )
    result = solve_quasi_fermi_steady_state(x, stack, V_app=0.0)

    assert result.certified
    assert result.current_A_m2 == pytest.approx(357.73, rel=2.0e-3)
    assert 0.0 < result.current_A_m2 < 1.602176634e-19 * stack.Phi
    assert result.electron_continuity_bound_A_m2 < 1.0e-4
    assert result.hole_continuity_bound_A_m2 < 1.0e-4
    assert result.face_current_spread_A_m2 < 1.0e-4
    assert np.ptp(result.total_face_current_A_m2) < 1.0e-4

    mat = build_material_arrays(x, stack)
    actual = np.diff(result.total_face_current_A_m2)
    expected = -mat.junction_polarity * 1.602176634e-19 * mat.dx_cell[1:-1] * (
        result.electron_rate_per_s[1:-1] - result.hole_rate_per_s[1:-1]
    )
    assert actual == pytest.approx(expected, abs=1.0e-10)

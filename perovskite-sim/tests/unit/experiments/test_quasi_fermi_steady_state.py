"""Focused contracts for the opt-in quasi-Fermi steady-state solver."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    _QuasiFermiSystem,
    _density_from_log,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
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

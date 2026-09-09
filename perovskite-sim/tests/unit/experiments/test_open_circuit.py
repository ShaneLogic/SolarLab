"""Independent dielectric and charge-continuity checks for floating voltage."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments.jv_sweep import compute_current_components
from perovskite_sim.experiments.open_circuit import (
    OpenCircuitError,
    OpenCircuitSystem,
    integrate_open_circuit,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import StateVec, build_material_arrays


@pytest.fixture
def dielectric():
    base = load_device_from_yaml("tests/fixtures/configs/nip_MAPbI3.yaml")
    layer = base.layers[1]
    params = replace(
        layer.params, N_A=0.0, N_D=0.0, D_ion=0.0, D_ion_neg=0.0,
        P0=0.0, P0_neg=0.0, alpha=0.0,
    )
    layer = replace(layer, params=params)
    stack = replace(base, layers=(layer,), interfaces=(), interface_defects=(), V_bi=0.0)
    x = np.linspace(0.0, layer.thickness, 9)
    mat = build_material_arrays(x, stack)
    mat = replace(mat, D_n_face=np.zeros(x.size - 1), D_p_face=np.zeros(x.size - 1))
    state = StateVec.pack(np.full(x.size, params.ni), np.full(x.size, params.ni), mat.P_ion0)
    return OpenCircuitSystem(x, stack, mat), state


def test_dielectric_voltage_history_produces_geometric_displacement(dielectric):
    system, state = dielectric
    voltage_before, voltage_after, dt = 0.1, 0.101, 2e-6
    expected = -EPS_0 * system.mat.eps_r[0] / np.ptp(system.x) * 0.001 / dt
    missing_history = compute_current_components(
        system.x, state, system.stack, voltage_after,
        y_prev=state, dt=dt, mat=system.mat,
    )
    correct_history = compute_current_components(
        system.x, state, system.stack, voltage_after,
        y_prev=state, V_app_prev=voltage_before, dt=dt, mat=system.mat,
    )
    np.testing.assert_array_equal(missing_history.J_disp, 0.0)
    np.testing.assert_allclose(correct_history.J_disp, expected, rtol=1e-12)
    np.testing.assert_allclose(
        system.displacement_change(state, voltage_before, state, voltage_after) / dt,
        expected, rtol=1e-12,
    )


@pytest.mark.parametrize("polarity", [-1.0, 1.0])
def test_geometric_capacitance_is_positive_for_both_contact_orientations(dielectric, polarity):
    system, _ = dielectric
    mat = replace(system.mat, junction_polarity=polarity)
    model = OpenCircuitSystem(system.x, system.stack, mat)
    expected = EPS_0 * mat.eps_r[0] / np.ptp(system.x)
    np.testing.assert_allclose(model.displacement_voltage_derivative, -expected, rtol=1e-13)


@pytest.mark.parametrize("rtol", [1e-4, 1e-5, 1e-6])
def test_known_capacitance_and_conductance_decay(dielectric, monkeypatch, rtol):
    system, state = dielectric
    tau = 5e-6
    conductance = system.capacitance_A / tau
    equilibrium_voltage = 1.0
    amplitude = 1e-3
    monkeypatch.setattr(
        system, "conduction_current",
        lambda _y, v: np.full(system.x.size - 1, -conductance * (v - equilibrium_voltage)),
    )
    times = np.linspace(0.0, 4 * tau, 41)
    result = integrate_open_circuit(
        system, state, equilibrium_voltage + amplitude, times,
        rtol=rtol, voltage_atol=1e-11, max_step=tau / 20,
    )
    expected = equilibrium_voltage + amplitude * np.exp(-times / tau)
    np.testing.assert_allclose(result.V, expected, rtol=0.0, atol=1e-8)
    assert np.max(result.max_face_current_A_m2) < 1e-12
    assert result.charge_voltage_error_V[-1] < 1e-9
    assert np.all(result.valid)


def test_output_sampling_does_not_change_voltage_evolution(dielectric, monkeypatch):
    system, state = dielectric
    tau = 5e-6
    monkeypatch.setattr(
        system, "conduction_current",
        lambda _y, v: np.full(system.x.size - 1, -system.capacitance_A / tau * v),
    )
    coarse = integrate_open_circuit(system, state, 0.001, np.linspace(0.0, 4 * tau, 11))
    fine = integrate_open_circuit(system, state, 0.001, np.linspace(0.0, 4 * tau, 41))
    np.testing.assert_array_equal(coarse.V, fine.V[::4])
    assert coarse.nfev == fine.nfev


def test_failed_integrator_never_reuses_the_initial_state(dielectric, monkeypatch):
    system, state = dielectric
    monkeypatch.setattr(
        "perovskite_sim.experiments.open_circuit.solve_ivp",
        lambda *_args, **_kwargs: SimpleNamespace(success=False, message="injected failure"),
    )
    with pytest.raises(OpenCircuitError, match="injected failure"):
        integrate_open_circuit(system, state, 0.0, np.array([0.0, 1e-6]))


def test_wrong_displacement_in_the_integral_is_detected(dielectric, monkeypatch):
    system, state = dielectric
    monkeypatch.setattr(
        system, "conduction_current", lambda _y, _v: np.ones(system.x.size - 1),
    )
    monkeypatch.setattr(
        system, "displacement_change", lambda *_args: np.zeros(system.x.size - 1),
    )
    with pytest.raises(OpenCircuitError, match="integrated Maxwell current"):
        integrate_open_circuit(system, state, 0.0, np.array([0.0, 1e-6]))


@pytest.mark.parametrize("dual", [False, True])
def test_real_density_rhs_has_uniform_instantaneous_maxwell_current(dielectric, dual):
    base, _ = dielectric
    layer = base.stack.layers[0]
    params = replace(
        layer.params, mu_n=1e-4, mu_p=2e-4, D_ion=1e-14,
        P0=1e20, P0_neg=(0.7e20 if dual else 0.0),
        D_ion_neg=(2e-14 if dual else 0.0),
    )
    stack = replace(base.stack, layers=(replace(layer, params=params),))
    mat = build_material_arrays(base.x, stack)
    model = OpenCircuitSystem(base.x, stack, mat)
    perturbation = np.sin(np.pi * base.x / base.x[-1])
    state = StateVec.pack(
        params.ni * (1 + 0.2 * perturbation),
        params.ni * (1 - 0.1 * perturbation),
        mat.P_ion0 * (1 + 0.1 * perturbation),
        mat.P_ion0_neg * (1 - 0.1 * perturbation) if dual else None,
    )
    dy, dv, total = model.rates(0.0, state, 0.03)
    conduction = model.conduction_current(state, 0.03)
    assert np.max(np.abs(conduction)) > 1e-7
    assert np.max(np.abs(dy)) > 0.0
    assert np.isfinite(dv)
    np.testing.assert_allclose(total, 0.0, atol=1e-11)

"""Frozen snapshots preserve both ionic species and require electronic settling."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import degradation
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    compute_ionic_current_components,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import StateVec, build_material_arrays, run_transient
from perovskite_sim.solver.newton import solve_equilibrium


def _problem(species="dual", substrate=False):
    stack = load_device_from_yaml("tests/fixtures/configs/tpv_physical_reference.yaml")
    layers = list(stack.layers)
    positive = species in ("positive", "dual")
    negative = species in ("negative", "dual")
    params = replace(
        layers[1].params, P0=1e20 if positive else 0.0,
        P0_neg=0.7e20 if negative else 0.0,
        D_ion=2e-14 if positive else 0.0,
        D_ion_neg=4e-14 if negative else 0.0, P_lim_neg=1e26,
    )
    layers[1] = replace(layers[1], params=params)
    if substrate:
        layers.insert(0, replace(layers[0], name="glass", role="substrate",
                                 thickness=1e-3, params=None))
    return replace(stack, layers=tuple(layers))


def _nonuniform_state(x, stack, mat):
    state = solve_equilibrium(x, stack)
    blocks = StateVec.unpack(state, x.size)
    wave = np.sin(2 * np.pi * x / x[-1])
    for density, initial, sign in ((blocks.P, mat.P_ion0, 1.0),
                                    (blocks.P_neg, mat.P_ion0_neg, -1.0)):
        if density is None or initial is None:
            continue
        active = initial > 0.0
        if np.any(active):
            centered = wave[active] - np.average(wave[active], weights=mat.dx_cell[active])
            density[active] = initial[active] * (1.0 + sign * 0.2 * centered)
    return state


@pytest.mark.parametrize("species", ["none", "positive", "negative", "dual"])
@pytest.mark.parametrize("substrate", [False, True])
def test_freezing_preserves_materials_background_and_capacities(species, substrate):
    original = _problem(species, substrate)
    frozen = degradation._freeze_ions(original)
    for before, after in zip(original.layers, frozen.layers):
        if before.params is None:
            assert after is before
        else:
            assert after.params == replace(before.params, D_ion=0.0, D_ion_neg=0.0)
    assert original == _problem(species, substrate)


@pytest.mark.parametrize("species", ["none", "positive", "negative", "dual"])
def test_nonuniform_ions_have_zero_frozen_flux_and_never_move(species):
    stack = _problem(species)
    x = build_electrical_grid(stack, 18)
    mat = build_material_arrays(x, stack)
    state = _nonuniform_state(x, stack, mat)
    original_current = compute_ionic_current_components(x, state, stack, 0.1, mat=mat)
    if species in ("positive", "dual"):
        assert np.max(np.abs(original_current.J_positive)) > 1e-10
    if species in ("negative", "dual"):
        assert np.max(np.abs(original_current.J_negative)) > 1e-10
    frozen = degradation._freeze_ions(stack)
    frozen_mat = build_material_arrays(x, frozen)
    assert frozen_mat.has_dual_ions == mat.has_dual_ions
    np.testing.assert_array_equal(frozen_mat.P_ion0, mat.P_ion0)
    if mat.has_dual_ions:
        np.testing.assert_array_equal(frozen_mat.P_ion0_neg, mat.P_ion0_neg)
    sol = run_transient(
        x, state, (0.0, 1e-7), np.linspace(0.0, 1e-7, 5), frozen,
        V_app=0.1, mat=frozen_mat, rtol=1e-6, atol=1e-8, max_step=1e-8,
    )
    assert sol.success, sol.message
    assert np.all(np.isfinite(sol.y))
    for accepted in sol.y.T:
        np.testing.assert_array_equal(accepted[2 * x.size:], state[2 * x.size:])
        current = compute_ionic_current_components(x, accepted, frozen, 0.1, mat=frozen_mat)
        np.testing.assert_array_equal(current.J_positive, 0.0)
        if current.J_negative is not None:
            np.testing.assert_array_equal(current.J_negative, 0.0)


def _mock_snapshot_solves(monkeypatch, state):
    monkeypatch.setattr(degradation, "run_transient", lambda *_a, **_k: SimpleNamespace(
        success=True, y=state[:, None],
    ))
    monkeypatch.setattr(degradation, "solve_steady_state", lambda *_a, **_k: SimpleNamespace(
        y=state.copy(), converged=True, residual=0.0, continuity_current_bound=0.0,
    ), raising=False)


def test_snapshot_observer_uses_the_frozen_material(monkeypatch):
    stack = _problem("dual")
    x = build_electrical_grid(stack, 12)
    mat = build_material_arrays(x, stack)
    state = _nonuniform_state(x, stack, mat)
    _mock_snapshot_solves(monkeypatch, state)
    seen = []

    def current(_x, _state, measured_stack, voltage, *, mat):
        assert all(layer.params.D_ion == layer.params.D_ion_neg == 0.0
                   for layer in electrical_layers(measured_stack))
        np.testing.assert_array_equal(mat.D_ion_face, 0.0)
        np.testing.assert_array_equal(mat.D_ion_neg_face, 0.0)
        seen.append(voltage)
        return 1.0 - voltage

    def components(*args, **kwargs):
        value = current(*args, **kwargs)
        return SimpleNamespace(J_total=np.full(x.size - 1, value), J_ion=np.zeros(x.size - 1))

    monkeypatch.setattr(degradation, "_compute_current", current, raising=False)
    monkeypatch.setattr(degradation, "compute_current_components", components, raising=False)
    degradation._measure_snapshot_metrics(
        x, state, stack, np.array([0.0, 0.6, 1.2]), 1e-4, 1e-6, 1e-8,
    )
    assert seen


def test_successful_transient_is_not_a_steady_state_certificate(monkeypatch):
    stack = _problem("none")
    x = build_electrical_grid(stack, 12)
    state = solve_equilibrium(x, stack)
    _mock_snapshot_solves(monkeypatch, state)

    def not_stationary(*_args, **_kwargs):
        raise RuntimeError("carriers are still evolving")

    monkeypatch.setattr(degradation, "solve_steady_state", not_stationary)
    with pytest.raises(RuntimeError, match="still evolving"):
        degradation._measure_snapshot_metrics(
            x, state, stack, np.array([0.0, 0.6, 1.2]), 1e-12, 1e-6, 1e-8,
        )


@pytest.mark.parametrize("substrate", [False, True])
def test_original_diffusivity_does_not_change_the_same_frozen_snapshot(substrate):
    stack = _problem("dual", substrate)
    x = build_electrical_grid(stack, 18)
    material = build_material_arrays(x, stack)
    state = _nonuniform_state(x, stack, material)
    original_state = state.copy()
    faster = replace(stack, layers=tuple(
        layer if layer.params is None else replace(layer, params=replace(
            layer.params, D_ion=100 * layer.params.D_ion,
            D_ion_neg=30 * layer.params.D_ion_neg,
        )) for layer in stack.layers
    ))
    results = [degradation.measure_frozen_ion_snapshot(
        x, state, device, np.array([0.0, 0.4, 0.8, 1.2]), 1e-4, 1e-6, 1e-8,
    ) for device in (stack, faster)]
    np.testing.assert_array_equal(state, original_state)
    np.testing.assert_array_equal(results[0].J, results[1].J)
    np.testing.assert_array_equal(results[0].states, results[1].states)
    for result in results:
        assert np.max(result.carrier_current_bound_A_m2) <= 0.05
        assert np.max(result.current_spread_A_m2) <= 0.05
        for accepted in result.states:
            np.testing.assert_array_equal(accepted[2 * x.size:], state[2 * x.size:])


def test_empty_negative_species_keeps_the_inherited_state_layout():
    stack = _problem("none")
    layers = list(stack.layers)
    layers[1] = replace(layers[1], params=replace(layers[1].params, D_ion_neg=1e-14))
    stack = replace(stack, layers=tuple(layers))
    x = build_electrical_grid(stack, 12)
    state = solve_equilibrium(x, stack)
    assert state.size == 4 * x.size
    result = degradation.measure_frozen_ion_snapshot(
        x, state, stack, np.array([0.0, 0.8, 1.2]), 1e-4, 1e-6, 1e-8,
    )
    assert result.states.shape == (3, state.size)
    np.testing.assert_array_equal(result.states[:, 3 * x.size:], 0.0)


@pytest.mark.parametrize("current_bound", [np.nan, np.inf, 0.1])
def test_invalid_electronic_certificate_cannot_produce_metrics(monkeypatch, current_bound):
    stack = _problem("none")
    x = build_electrical_grid(stack, 12)
    state = solve_equilibrium(x, stack)
    _mock_snapshot_solves(monkeypatch, state)
    monkeypatch.setattr(degradation, "solve_steady_state", lambda *_a, **_k: SimpleNamespace(
        y=state.copy(), converged=True, residual=0.0, continuity_current_bound=current_bound,
    ))
    with pytest.raises(RuntimeError, match="steady-state certificate"):
        degradation.measure_frozen_ion_snapshot(
            x, state, stack, np.array([0.0, 0.8, 1.2]), 1e-4, 1e-6, 1e-8,
        )


def test_two_snapshots_reach_the_requested_end_and_preserve_both_histories():
    stack = _problem("dual", substrate=True)
    result = degradation.run_degradation(
        stack, t_end=1e-4, n_snapshots=2, V_bias=0.7, N_grid=18,
        dt_max=1e-4, metric_n_points=9, metric_V_max=1.2,
        metric_settle_time=1e-4, rtol=1e-6, atol=1e-8,
    )
    np.testing.assert_array_equal(result.t, [0.0, 1e-4])
    assert result.ion_profiles_neg.shape == result.ion_profiles.shape
    assert np.any(result.ion_profiles_neg[-1] != result.ion_profiles_neg[0])
    assert np.max(result.snapshot_carrier_current_bound_A_m2) <= 0.05
    assert np.max(result.snapshot_current_spread_A_m2) <= 0.05
    assert np.all(np.isfinite(result.PCE))

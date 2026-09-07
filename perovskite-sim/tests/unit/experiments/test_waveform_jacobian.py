from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep as jv, waveform_jv as wf
from perovskite_sim.experiments.waveform_jacobian import (
    WaveformJacobianCapabilityError, build_waveform_density_jacobian,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.solver import mol


def setup_case(population=1e25):
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    layers = list(stack.layers)
    layers[1] = replace(layers[1], params=replace(layers[1].params, P0=population))
    stack = replace(stack, layers=tuple(layers), mode="full")
    x = jv.build_electrical_grid(stack, 30)
    mat = jv.build_material_arrays(x, stack)
    state = jv.solve_equilibrium(x, stack)
    return stack, x, mat, state


@pytest.mark.parametrize("population", [0, 5e24, 1e25, 2e25])
@pytest.mark.parametrize("bias", [-1.0, 0.4])
def test_species_tangents_match_full_rhs_and_preserve_ion_inventory(population, bias):
    stack, x, mat, state = setup_case(population)
    before = asdict(stack)
    count = len(x)
    jacobian = build_waveform_density_jacobian(x, stack, mat, bias)(0, state)
    assert jacobian.shape == (state.size, state.size)
    pinned = [0, count-1, count, 2*count-1]
    np.testing.assert_array_equal(jacobian[pinned], 0)
    np.testing.assert_array_equal(jacobian[:, pinned], 0)
    widths = dual_cell_widths(x)
    cancellation_scale = max(float(np.max(widths @ abs(jacobian[2*count:]))), 1.0)
    assert np.max(abs(widths @ jacobian[2*count:]))/cancellation_scale < 1e-13
    random = np.random.default_rng(1606)
    for block in range(3):
        direction = np.zeros_like(state)
        selected = slice(block*count, (block+1)*count)
        floor = 1e24 if block == 2 else 1
        direction[selected] = random.normal(size=count)*np.maximum(abs(state[selected]), floor)
        expected = jacobian @ direction
        step = 1e-6
        observed = (mol.assemble_rhs(0, state+step*direction, x, stack, mat, False, bias)
                    - mol.assemble_rhs(0, state-step*direction, x, stack, mat, False, bias))/(2*step)
        for output_block in range(3):
            region = slice(output_block*count, (output_block+1)*count)
            error = np.linalg.norm(observed[region]-expected[region])
            scale = max(np.linalg.norm(expected[region]), 1.0)
            assert error/scale < 5e-5
    assert asdict(stack) == before


def test_time_dependent_bias_is_used_without_changing_state():
    stack, x, mat, state = setup_case()
    before = state.copy()
    dynamic = build_waveform_density_jacobian(x, stack, mat, lambda time: time)
    fixed = build_waveform_density_jacobian(x, stack, mat, 0.4)
    np.testing.assert_array_equal(dynamic(0.4, state), fixed(0, state))
    assert not np.array_equal(dynamic(0, state), dynamic(0.4, state))
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("change, message", [
    ({"has_dual_ions": True}, "dual ions"),
    ({"has_field_mobility": True}, "field-dependent mobility"),
    ({"has_radiative_reabsorption": True}, "radiative reabsorption"),
    ({"degenerate_recombination_model": "off"}, "recombination closure"),
])
def test_unsupported_closures_are_refused_without_substitution(change, message):
    stack, x, mat, _ = setup_case()
    with pytest.raises(WaveformJacobianCapabilityError, match=message):
        build_waveform_density_jacobian(x, stack, replace(mat, **change), 0)


def test_high_occupancy_stays_on_the_original_numerical_path():
    stack, x, mat, _ = setup_case()
    with pytest.raises(WaveformJacobianCapabilityError, match="steric clipping"):
        build_waveform_density_jacobian(x, stack, replace(mat, P_lim_node=mat.P_lim_node*1e-5), 0)


@pytest.mark.parametrize("speeds", [
    {"S_n_left": 0.0, "S_p_right": 0.0},
    {"S_n_left": 200.0, "S_p_left": 1000.0, "S_n_right": 300.0, "S_p_right": 400.0},
])
def test_selective_contact_columns_match_existing_boundary_equations(speeds):
    stack, x, _, state = setup_case()
    stack = replace(stack, **speeds)
    mat = jv.build_material_arrays(x, stack)
    count = len(x)
    # Resolve reaction changes against the large boundary transport residual.
    # This is a positive test operating point, not a production density floor.
    state = state.copy()
    state[[0, count-1, count, 2*count-1]] = np.maximum(
        state[[0, count-1, count, 2*count-1]], 1e18)
    jacobian = build_waveform_density_jacobian(x, stack, mat, -1)(0, state)
    for index in (0, count-1, count, 2*count-1):
        direction = np.zeros_like(state)
        direction[index] = max(abs(state[index]), 1)
        step = 1e-5
        observed = (mol.assemble_rhs(0, state+step*direction, x, stack, mat, False, -1)
                    - mol.assemble_rhs(0, state-step*direction, x, stack, mat, False, -1))/(2*step)
        expected = jacobian@direction
        assert np.linalg.norm(observed-expected)/max(np.linalg.norm(expected), 1) < 1e-5


def test_boundary_reaction_derivative_at_tiny_minority_density():
    stack, x, _, state = setup_case()
    stack = replace(stack, S_n_left=200.0, S_p_left=1000.0)
    mat = jv.build_material_arrays(x, stack)
    jacobian = build_waveform_density_jacobian(x, stack, mat, -1)(0, state)
    count = len(x)
    n, p, _, _ = jv._state_fields(x, state, stack, -1, mat)
    step = 1e-4*n[0]
    plus, minus = n.copy(), n.copy()
    plus[0] += step
    minus[0] -= step
    args = (mat.ni_sq, mat.tau_n, mat.tau_p, mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p)
    derivative = (total_recombination(plus, p, *args)[0]-total_recombination(minus, p, *args)[0])/(2*step)
    assert jacobian[count, 0] == pytest.approx(-derivative, rel=1e-8)


def test_finite_contact_velocities_add_exact_relaxation_diagonal():
    stack, x, _, state = setup_case()
    names = ("S_n_left", "S_n_right", "S_p_left", "S_p_right")
    speeds = (200.0, 300.0, 400.0, 500.0)
    blocked = replace(stack, **dict.fromkeys(names, 0.0))
    finite = replace(stack, **dict(zip(names, speeds)))
    matrices = [build_waveform_density_jacobian(x, device, jv.build_material_arrays(x, device), 0.4)(0, state)
                for device in (blocked, finite)]
    difference = matrices[1]-matrices[0]
    expected = np.zeros_like(difference)
    count = len(x)
    for index, speed, width in zip((0, count-1, count, 2*count-1), speeds,
                                   (x[1]-x[0], x[-1]-x[-2], x[1]-x[0], x[-1]-x[-2])):
        expected[index, index] = -speed/width
    np.testing.assert_allclose(difference, expected, rtol=1e-10, atol=1e-3)


def test_dual_ion_waveform_preserves_physics_and_reports_finite_difference(monkeypatch):
    stack, _, _, _ = setup_case()
    layers = list(stack.layers)
    layers[1] = replace(layers[1], params=replace(layers[1].params, P0_neg=1e24, D_ion_neg=1e-18))
    stack = replace(stack, layers=tuple(layers))
    before = asdict(stack)

    def fixed(x, y, *args, **kwargs):
        assert "jacobian" not in kwargs
        return y.copy()

    def ramp(**kwargs):
        assert "jacobian" not in kwargs
        return SimpleNamespace(success=True, y=kwargs["y0"][:, None])

    monkeypatch.setattr(jv, "_integrate_step", fixed)
    monkeypatch.setattr(jv, "run_transient", ramp)
    result = wf.run_waveform_jv(stack, wf.JVWaveform(), N_grid=15, n_points=3, V_max=1)
    assert result.jacobian_evaluator == "finite_difference"
    assert "dual ions" in result.jacobian_fallback_reason
    assert asdict(stack) == before
    assert len(result.inventory_relative_drift) == 2


def test_low_level_jacobian_is_optional_and_forwarded_to_radau(monkeypatch):
    stack, x, mat, state = setup_case()
    captured = []

    def solver(fun, span, y0, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(success=True, y=y0[:, None], t=np.array([span[1]]), nfev=0)

    monkeypatch.setattr(mol, "solve_ivp", solver)
    jacobian = build_waveform_density_jacobian(x, stack, mat, 0)
    for supplied in (None, jacobian):
        mol.run_transient(x, state, (0, 1), np.array([1]), stack, mat=mat, jacobian=supplied)
    assert "jac" not in captured[0]
    assert captured[1]["jac"] is jacobian
    assert all(call["method"] == "Radau" for call in captured)
    with pytest.raises(ValueError, match="density coordinates"):
        mol.run_transient(x, state, (0, 1), np.array([1]), stack, mat=mat,
                          jacobian=jacobian, state_coordinates="research_log_density")
    with pytest.raises(TypeError, match="callable"):
        mol.run_transient(x, state, (0, 1), np.array([1]), stack, mat=mat, jacobian=1)


@pytest.mark.parametrize("legs", [1, 2])
def test_fixed_hold_subdivision_keeps_the_supplied_jacobian(monkeypatch, legs):
    stack, x, mat, state = setup_case()
    jacobian = build_waveform_density_jacobian(x, stack, mat, 0)
    calls = []

    def solver(x, y, span, samples, stack, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=len(calls) != 1, y=y[:, None], message="forced first failure")

    monkeypatch.setattr(jv, "run_transient", solver)
    result = jv._integrate_step(x, state, stack, mat, 0, 0, 1, 1e-4, 100,
                               n_legs=legs, jacobian=jacobian)
    assert len(calls) >= 3
    assert all(call["jacobian"] is jacobian for call in calls)
    np.testing.assert_array_equal(result, state)

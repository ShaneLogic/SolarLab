"""Actual accepted DC states, voltage tangents and Maxwell-current closure."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.impedance import _moving_voltage_partial, solve_impedance, voltage_reference_direction
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    controls = SteadyOptions()
    point = solve_steady_state(compiled, voltage_V=1.0, options=controls)
    result = solve_impedance(compiled, point, [0.0, 1.0, 1000.0], steady_options=controls)
    return compiled, point, controls, result


def test_actual_sparse_response_has_finite_fields_signed_currents_and_source_identity(case):
    compiled, point, _, result = case
    assert result.compiled_identity == compiled.identity
    assert len(result.source_identity) == len(result.working_state_identity) == 64
    assert result.operating_voltage_V == point.voltage_V and result.generation_scale == point.generation_scale
    assert result.voltage_phasor_V == 1.0
    assert result.potential_perturbation_V.shape == (3, compiled.node_count)
    assert np.all(result.admittance_density_S_m2.real > 0.0)
    assert result.admittance_density_S_m2[0].imag == 0.0
    assert np.all(result.admittance_density_S_m2[1:].imag > 0.0)
    np.testing.assert_allclose(result.impedance_area_ohm_m2 * result.admittance_density_S_m2, 1.0, rtol=3e-15)
    assert max(item.componentwise_backward_error for item in result.linear_solutions) < 1e-11
    assert np.max(result.maxwell_spread_A_m2_V) < 1e-8
    all_faces = np.c_[result.terminal_current_density_A_m2[:, :1], result.total_current_density_A_m2,
                      result.terminal_current_density_A_m2[:, 1:]]
    np.testing.assert_allclose(all_faces, np.broadcast_to(all_faces[:, :1], all_faces.shape), rtol=0.0, atol=1e-8)
    assert not result.electron_density_perturbation_m3.flags.writeable


def test_moving_reference_voltage_partial_has_an_independent_dc_derivative(case):
    compiled, point, _, _ = case
    u, voltage = point.state.u, point.voltage_V
    problem = DC1DProblem(compiled, voltage, point.generation_scale)
    analytical = _moving_voltage_partial(problem, u)
    # A sixth-order centered estimate resolves saturated recombination's
    # small derivative without subtracting almost identical plateau values
    # at an unnecessarily tiny voltage step.
    step = 1e-3
    estimates = []
    for width in (step, 2 * step, 4 * step):
        plus = DC1DProblem(compiled, voltage + width, point.generation_scale).residual(u)
        minus = DC1DProblem(compiled, voltage - width, point.generation_scale).residual(u)
        estimates.append((plus - minus) / (2 * width))
    fourth = (4 * estimates[0] - estimates[1]) / 3
    fourth_wide = (4 * estimates[1] - estimates[2]) / 3
    numerical = (16 * fourth - fourth_wide) / 15
    # As for core F_y checks, cancellation-prone rows use their absolute
    # directional conditioning rather than an unresolvable near-zero value.
    scale = np.maximum(abs(problem.jacobian(u)) @ abs(voltage_reference_direction(compiled)) + abs(analytical), 1.0)
    assert np.max(abs(numerical - analytical) / scale) < 2e-8


def test_storage_voltage_drive_is_included_exactly_once(case):
    compiled, point, _, result = case
    problem = DC1DProblem(compiled, point.voltage_V, point.generation_scale)
    J, K = problem.jacobian(point.state.u), problem.storage_jacobian(point.state.u)
    c = voltage_reference_direction(compiled)
    voltage_partial = _moving_voltage_partial(problem, point.state.u)
    omega = 2 * np.pi * result.frequency_Hz[-1]
    matrix = J + 1j * omega * K
    rhs = -voltage_partial + 1j * omega * (K @ c)
    y = result.moving_coordinate_perturbation[-1]
    residual = matrix @ y - rhs
    scale = abs(matrix) @ abs(y) + abs(rhs)
    assert np.max(np.divide(abs(residual), scale, out=np.zeros_like(scale), where=scale != 0.0)) < 1e-11
    assert np.max(abs(K @ c)) > 0.0
    assert np.max(abs(matrix @ y + voltage_partial)) > 1e10
    np.testing.assert_array_equal(result.fixed_coordinate_perturbation, result.moving_coordinate_perturbation - c)


def test_zero_frequency_agrees_with_independent_small_bias_dc_differences(case):
    compiled, point, controls, result = case
    step = 5e-5
    plus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V + step,
                              generation_scale=point.generation_scale, options=controls)
    minus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V - step,
                               generation_scale=point.generation_scale, options=controls)
    derivative = compiled.voltage_polarity * (plus.contact_current_density_A_m2 - minus.contact_current_density_A_m2) / (2 * step)
    np.testing.assert_allclose(result.admittance_density_S_m2[0], derivative, rtol=2e-6, atol=1e-5)
    for actual, name, density in (
        (result.electron_density_perturbation_m3[0], "electron_density_m3", point.state.electron_density_m3),
        (result.hole_density_perturbation_m3[0], "hole_density_m3", point.state.hole_density_m3),
    ):
        expected = (getattr(plus.state, name) - getattr(minus.state, name)) / (2 * step)
        normalized_error = abs(actual - expected) / density
        assert np.max(normalized_error) < 1e-4


def test_linear_normalization_is_not_a_finite_amplitude_drive(case):
    compiled, point, controls, result = case
    voltage = 2e-6 * (1.0 + 0.5j)
    scaled = solve_impedance(compiled, point, result.frequency_Hz, voltage_phasor_V=voltage, steady_options=controls)
    np.testing.assert_allclose(scaled.admittance_density_S_m2, result.admittance_density_S_m2, rtol=2e-10)
    np.testing.assert_allclose(scaled.potential_perturbation_V / voltage, result.potential_perturbation_V, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("contact_kind", ["selective"])
def test_blocking_contacts_remain_explicit_capability_boundaries(contact_kind):
    prepared = load_device()
    document = prepared.to_input().editing_data()
    if contact_kind == "robin":
        document["settings"]["flat_band_contacts"] = True
    else:
        document["contacts"] = [
            dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
            dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0),
        ]
    prepared = resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    compiled = compile_device_1d(prepared, intervals_per_layer=12)
    point = solve_steady_state(compiled, voltage_V=1.0)
    with pytest.raises(ValueError, match="unsupported blocking"):
        solve_impedance(compiled, point, [0.0, 1e6])


def test_missing_failed_foreign_or_rebound_dc_states_are_rejected(case):
    compiled, point, controls, _ = case
    with pytest.raises(ValueError, match="accepted"):
        solve_impedance(compiled, None, [1.0])
    with pytest.raises(ValueError, match="same compiled device|identity"):
        solve_impedance(compile_device_1d(load_device(), intervals_per_layer=24), point, [1.0])
    failed = copy(point)
    object.__setattr__(failed, "newton", replace(point.newton, accepted=False))
    with pytest.raises(ValueError, match="accepted"):
        solve_impedance(compiled, failed, [1.0])
    with pytest.raises(ValueError, match="supplied DC"):
        solve_impedance(compiled, point, [1.0], steady_options=replace(controls, continuity_tolerance_A_m2=2e-8))
    rebound = copy(point)
    object.__setattr__(rebound, "newton", replace(point.newton, scaled_residual=point.newton.scaled_residual + 0.5))
    with pytest.raises(ValueError, match="supplied DC"):
        solve_impedance(compiled, rebound, [1.0])


@pytest.mark.parametrize("voltage", [0.0, np.nan, np.inf, complex(1.0, np.inf)])
def test_invalid_voltage_normalizations_are_rejected(case, voltage):
    compiled, point, _, _ = case
    with pytest.raises(ValueError, match="finite and nonzero"):
        solve_impedance(compiled, point, [1.0], voltage_phasor_V=voltage)

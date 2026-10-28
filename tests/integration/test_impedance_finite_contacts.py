"""Robin AC and endpoint storage under the accepted linear/current criteria."""

from __future__ import annotations

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.impedance import _moving_voltage_partial, solve_impedance, voltage_reference_direction
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def robin():
    prepared = load_device()
    document = prepared.to_input().editing_data()
    document["settings"]["flat_band_contacts"] = True
    prepared = resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    compiled = compile_device_1d(prepared, intervals_per_layer=12)
    point = solve_steady_state(compiled, voltage_V=1.0)
    result = solve_impedance(compiled, point, [0.0, 1.0, 1000.0, 1e6])
    return compiled, point, result


def test_robin_actual_frequencies_close_full_original_rows_and_all_maxwell_faces(robin):
    compiled, point, result = robin
    assert all(channel.kind == "robin" for channel in compiled.channels)
    assert all(gate.passed for gate in point.physical_gates)
    problem = DC1DProblem(compiled, point.voltage_V, point.generation_scale)
    J, K = problem.jacobian(point.state.u), problem.storage_jacobian(point.state.u)
    c = voltage_reference_direction(compiled)
    voltage_partial = _moving_voltage_partial(problem, point.state.u)
    all_faces = np.c_[result.terminal_current_density_A_m2[:, :1], result.total_current_density_A_m2,
                      result.terminal_current_density_A_m2[:, 1:]]
    for index, solution in enumerate(result.linear_solutions):
        omega = 2 * np.pi * solution.frequency_Hz
        rhs = -voltage_partial + 1j * omega * (K @ c)
        residual = (J + 1j * omega * K) @ solution.solution - rhs
        np.testing.assert_array_equal(residual, solution.residual)
        assert solution.componentwise_backward_error <= 1e-11
        spread = float(np.max(abs(all_faces[index] - all_faces[index, 0])))
        assert spread <= 1e-8 + 1e-10 * float(np.max(abs(all_faces[index])))
    assert np.all(result.admittance_density_S_m2.real > 0.0)
    assert np.all(result.admittance_density_S_m2[1:].imag > 0.0)


def test_robin_zero_frequency_matches_both_terminal_dc_differences(robin):
    compiled, point, result = robin
    step = 5e-5
    plus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V + step)
    minus = solve_steady_state(compiled, initial_state=point.state, voltage_V=point.voltage_V - step)
    expected = compiled.voltage_polarity * (
        plus.contact_current_density_A_m2 - minus.contact_current_density_A_m2
    ) / (2 * step)
    actual = compiled.voltage_polarity * result.terminal_current_density_A_m2[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
    assert all(gate.passed for state in (plus, minus) for gate in state.physical_gates)


def test_robin_terminal_storage_is_required_at_original_high_frequency_trigger(robin):
    compiled, _, result = robin
    index = -1
    omega = 2 * np.pi * result.frequency_Hz[index]
    charge = 1.602176634e-19 * (
        result.hole_density_perturbation_m3[index] - result.electron_density_perturbation_m3[index]
    )
    widths = compiled.geometry.cell_measures_m
    displacement = -compiled.poisson.face_capacitance_F_m2 * np.diff(result.potential_perturbation_V[index])
    expected = 1j * omega * np.array([
        displacement[0] - widths[0] * charge[0], displacement[-1] + widths[-1] * charge[-1],
    ])
    np.testing.assert_allclose(result.terminal_displacement_current_density_A_m2[index], expected, rtol=3e-15)
    omitted = result.terminal_conduction_current_density_A_m2[index] + 1j * omega * displacement[[0, -1]]
    assert np.max(abs(omitted - result.terminal_current_density_A_m2[index])) > 1e-8


def test_robin_complex_normalization_is_still_a_linear_derivative(robin):
    compiled, point, result = robin
    voltage = 2e-6 * (1 + 0.5j)
    scaled = solve_impedance(compiled, point, result.frequency_Hz, voltage_phasor_V=voltage)
    np.testing.assert_allclose(scaled.admittance_density_S_m2, result.admittance_density_S_m2, rtol=2e-10)


def test_one_accepted_fixed_reservoir_baseline_is_preserved():
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    point = solve_steady_state(compiled, voltage_V=1.0)
    result = solve_impedance(compiled, point, [0.0, 1.0, 1000.0])
    expected = np.array([
        21.0997362166 + 0j, 21.0997362166 + 0.00472437883213j,
        21.0997369861 + 4.72437883213j,
    ])
    np.testing.assert_allclose(result.admittance_density_S_m2, expected, rtol=2e-10)
    assert np.max(result.maxwell_spread_A_m2_V) < 1e-8

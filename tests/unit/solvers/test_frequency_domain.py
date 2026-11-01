"""Small analytic complex systems, full residuals and bounded refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from solarlab.solvers.linear import FrequencyOptions, FrequencySolveError, solve_frequency_systems


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def test_parallel_resistor_capacitor_has_positive_admittance_parts():
    frequencies = np.array([0.0, 1.0, 1000.0])
    solutions = solve_frequency_systems(csc_matrix([[2.0]]), csc_matrix([[0.25]]), [1.0], frequencies)
    for frequency, result in zip(frequencies, solutions):
        admittance = 1.0 / result.solution[0]
        np.testing.assert_allclose(admittance, 2.0 + 1j * 2 * math.pi * frequency * 0.25, rtol=3e-15)
        assert admittance.real > 0.0 and admittance.imag >= 0.0
        assert result.componentwise_backward_error < 1e-14
        assert np.iscomplexobj(result.solution) and not result.solution.flags.writeable


def test_voltage_rate_forcing_is_retained():
    solutions = solve_frequency_systems(
        csc_matrix([[2.0]]), csc_matrix([[0.25]]), [2.0], [0.0, 1000.0], rhs_rate=[0.25],
    )
    for result in solutions:
        np.testing.assert_allclose(result.solution, 1.0, rtol=3e-15)


def test_exact_algebraic_elimination_checks_all_original_rows():
    J = csc_matrix([[1.0, 0.0], [-1.0, 2.0]])
    K = csc_matrix([[0.0, 0.0], [0.0, 0.25]])
    for result in solve_frequency_systems(J, K, [0.0, 1.0], [0.0, 1000.0], fixed_coordinates={0: 0.0}):
        assert result.solution[0] == 0.0
        residual = (J + 1j * 2 * math.pi * result.frequency_Hz * K) @ result.solution - [0.0, 1.0]
        np.testing.assert_array_equal(result.residual, residual)
        assert result.componentwise_backward_error < 1e-14


def test_complex_rhs_and_large_row_scales_are_preserved():
    J = csc_matrix([[1e-20, 0.0], [0.0, 1e20]])
    b = np.array([1e-20 * (1 + 2j), 1e20 * (3 - 4j)])
    result = solve_frequency_systems(J, csc_matrix((2, 2)), b, [0.0])[0]
    np.testing.assert_allclose(result.solution, [1 + 2j, 3 - 4j], rtol=3e-15)


def test_zero_frequency_singularity_is_explicit_without_regularization():
    with pytest.raises(FrequencySolveError, match="singular"):
        solve_frequency_systems(csc_matrix([[0.0]]), csc_matrix([[1.0]]), [1.0], [0.0])
    result = solve_frequency_systems(csc_matrix([[0.0]]), csc_matrix([[1.0]]), [1.0], [1.0])[0]
    np.testing.assert_allclose(result.solution[0], 1.0 / (2j * math.pi), rtol=3e-15)


@pytest.mark.parametrize("frequencies", [[], [-1.0], [math.nan], [math.inf], [1.0, 1.0], [1 + 2j], [1e308]])
def test_invalid_frequencies_and_angular_overflow_are_rejected(frequencies):
    with pytest.raises(ValueError):
        solve_frequency_systems(csc_matrix([[1.0]]), csc_matrix([[1.0]]), [1.0], frequencies)


def test_frequency_count_is_bounded():
    with pytest.raises(ValueError, match="bounded"):
        solve_frequency_systems(
            csc_matrix([[1.0]]), csc_matrix([[1.0]]), [1.0], [0.0, 1.0],
            options=FrequencyOptions(max_frequencies=1),
        )


@pytest.mark.parametrize("settings", [
    dict(max_frequencies=0), dict(relative_tolerance=0.0),
    dict(relative_tolerance=math.nan), dict(max_refinements=-1),
])
def test_numerical_controls_are_explicit_and_valid(settings):
    with pytest.raises(ValueError):
        FrequencyOptions(**settings)


def test_dynamic_or_coupled_rows_cannot_be_dropped_as_constraints():
    with pytest.raises(ValueError, match="algebraic"):
        solve_frequency_systems(
            csc_matrix(np.eye(2)), csc_matrix(np.eye(2)), [0.0, 1.0], [1.0], fixed_coordinates={0: 0.0},
        )
    with pytest.raises(ValueError, match="no free variables"):
        solve_frequency_systems(
            csc_matrix([[1.0, 1.0], [0.0, 1.0]]), csc_matrix((2, 2)),
            [0.0, 1.0], [0.0], fixed_coordinates={0: 0.0},
        )


def test_wrong_prescribed_values_fail_full_equations():
    with pytest.raises(FrequencySolveError):
        solve_frequency_systems(
            csc_matrix([[1.0]]), csc_matrix([[0.0]]), [1.0], [0.0], fixed_coordinates={0: 0.0},
        )

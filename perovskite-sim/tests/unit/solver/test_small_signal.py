from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.solver.small_signal import (
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


def test_frequency_domain_solver_recovers_series_rc_impedance():
    resistance = 10.0
    capacitance = 1.0e-6
    frequencies = np.array([1.0e2, 1.0e4, 1.0e6])
    V_dc = 0.4

    def evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        capacitor_voltage = float(state[0])
        current = (voltage - capacitor_voltage) / resistance
        return SmallSignalEvaluation(
            storage=np.array([capacitor_voltage]),
            rate=np.array([current / capacitance]),
            conduction_current_faces=np.array([current, current]),
            displacement_charge_faces=np.zeros(2),
        )

    result = solve_frequency_domain(
        evaluate,
        np.array([V_dc]),
        V_dc,
        frequencies,
        face_weights=np.array([0.25, 0.75]),
    )

    expected = resistance + 1.0 / (1j * 2.0 * np.pi * frequencies * capacitance)
    np.testing.assert_allclose(result.impedance, expected, rtol=2.0e-9)
    np.testing.assert_allclose(result.admittance, 1.0 / expected, rtol=2.0e-9)
    expected_storage = 1.0 / (
        1.0 + 1j * 2.0 * np.pi * frequencies * resistance * capacitance
    )
    np.testing.assert_allclose(
        result.state_response[:, 0], expected_storage, rtol=2.0e-9
    )
    np.testing.assert_allclose(
        result.storage_response[:, 0], expected_storage, rtol=2.0e-9
    )
    assert np.all(result.impedance.imag < 0.0)
    np.testing.assert_allclose(result.max_relative_face_spread, 0.0, atol=1.0e-15)
    assert np.all((0.0 < result.reciprocal_condition))
    assert np.all(result.reciprocal_condition <= 1.0 + 1.0e-12)
    assert np.max(result.backward_error) < 1.0e-12


def test_frequency_domain_solver_includes_direct_displacement_response():
    capacitance = 3.0e-4
    frequencies = np.array([1.0e3, 1.0e5])

    def evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        return SmallSignalEvaluation(
            storage=np.array([state[0]]),
            rate=np.array([-state[0]]),
            conduction_current_faces=np.zeros(1),
            displacement_charge_faces=np.array([capacitance * voltage]),
        )

    result = solve_frequency_domain(
        evaluate,
        np.array([0.0]),
        0.0,
        frequencies,
    )

    expected_admittance = 1j * 2.0 * np.pi * frequencies * capacitance
    np.testing.assert_allclose(result.admittance, expected_admittance, rtol=1.0e-12)
    np.testing.assert_allclose(result.storage_response, 0.0, atol=1.0e-15)
    assert np.all(result.impedance.imag < 0.0)
    assert np.all(result.reciprocal_condition > 0.0)
    assert np.max(result.backward_error) < 1.0e-12


def test_frequency_domain_solver_rejects_singular_dynamic_operator():
    def evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        return SmallSignalEvaluation(
            storage=np.ones(1),
            rate=np.zeros(1),
            conduction_current_faces=np.array([voltage]),
            displacement_charge_faces=np.zeros(1),
        )

    with pytest.raises(SmallSignalLinearizationError, match="singular"):
        solve_frequency_domain(
            evaluate,
            np.array([0.0]),
            0.0,
            np.array([1.0e3]),
        )


def test_frequency_domain_solver_exposes_reference_operators_and_components():
    resistance = 4.0
    capacitance = 2.0e-3
    frequency = np.array([10.0])

    def evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        current = (voltage - state[0]) / resistance
        electron = 0.25 * current
        hole = 0.75 * current
        return SmallSignalEvaluation(
            storage=np.array([capacitance * state[0]]),
            rate=np.array([current]),
            conduction_current_faces=np.array([current, current]),
            displacement_charge_faces=np.zeros(2),
            current_components=(
                SmallSignalCurrentComponent(
                    "electron", np.array([electron, electron])
                ),
                SmallSignalCurrentComponent("hole", np.array([hole, hole])),
            ),
        )

    result = solve_frequency_domain(
        evaluate,
        np.array([0.0]),
        0.0,
        frequency,
    )

    np.testing.assert_allclose(result.mass_matrix, [[capacitance]], rtol=1.0e-10)
    np.testing.assert_allclose(
        result.rate_jacobian, [[-1.0 / resistance]], rtol=1.0e-10
    )
    np.testing.assert_allclose(
        result.rate_voltage_derivative, [1.0 / resistance], rtol=1.0e-10
    )
    np.testing.assert_allclose(
        result.conduction_admittance_faces
        + result.displacement_admittance_faces,
        result.admittance_faces,
    )
    components = {item.name: item for item in result.current_components}
    np.testing.assert_allclose(
        components["electron"].admittance_faces
        + components["hole"].admittance_faces,
        result.conduction_admittance_faces,
    )
    np.testing.assert_allclose(
        components["electron"].current_jacobian
        + components["hole"].current_jacobian,
        result.conduction_current_jacobian,
    )
    np.testing.assert_allclose(
        components["electron"].voltage_derivative
        + components["hole"].voltage_derivative,
        result.conduction_current_voltage_derivative,
    )


def test_frequency_domain_solver_rejects_incomplete_current_decomposition():
    def evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        del voltage
        return SmallSignalEvaluation(
            storage=np.array([state[0]]),
            rate=np.array([-state[0]]),
            conduction_current_faces=np.ones(1),
            displacement_charge_faces=np.zeros(1),
            current_components=(
                SmallSignalCurrentComponent("electron", np.array([0.5])),
            ),
        )

    with pytest.raises(SmallSignalLinearizationError, match="must sum"):
        solve_frequency_domain(
            evaluate,
            np.array([1.0]),
            0.0,
            np.array([1.0]),
        )

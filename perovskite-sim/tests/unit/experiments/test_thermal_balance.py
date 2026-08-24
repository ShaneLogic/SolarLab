from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.experiments.thermal_balance import (
    LumpedThermalProtocol,
    ThermalBalanceError,
    ThermalEnergySourceError,
    ThermalIntegrationProtocol,
    ThermalPowerLedger,
    ThermalSteadyStateError,
    ThermalTransientError,
    run_lumped_thermal_transient,
    solve_lumped_thermal_steady_state,
    thermal_power_ledger,
)


def _thermal_protocol(**overrides) -> LumpedThermalProtocol:
    values = {
        "absorbed_optical_power_W_m2": 800.0,
        "ambient_temperature_K": 300.0,
        "areal_heat_capacity_J_m2_K": 2000.0,
        "heat_transfer_coefficient_W_m2_K": 20.0,
        "emissivity": 0.0,
        "maximum_temperature_K": 500.0,
    }
    values.update(overrides)
    return LumpedThermalProtocol(**values)


def _integration_protocol(**overrides) -> ThermalIntegrationProtocol:
    values = {
        "duration_s": 200.0,
        "initial_temperature_K": 300.0,
        "sample_count": 41,
        "relative_tolerance": 1.0e-10,
        "absolute_temperature_tolerance_K": 1.0e-11,
        "absolute_energy_tolerance_J_m2": 1.0e-9,
        "max_step_divisor": 400,
        "energy_balance_tolerance_J_m2": 1.0e-7,
    }
    values.update(overrides)
    return ThermalIntegrationProtocol(**values)


def test_protocols_round_trip_canonical_hash_and_reject_unknown_keys():
    thermal = _thermal_protocol(emissivity=0.85)
    integration = _integration_protocol()

    assert LumpedThermalProtocol.from_dict(thermal.to_dict()) == thermal
    assert ThermalIntegrationProtocol.from_dict(integration.to_dict()) == integration
    assert LumpedThermalProtocol.from_dict(thermal.to_dict()).sha256 == thermal.sha256
    assert (
        ThermalIntegrationProtocol.from_dict(integration.to_dict()).sha256
        == integration.sha256
    )

    payload = thermal.to_dict()
    payload["claim"] = "measured"
    with pytest.raises(ThermalBalanceError, match="extra=.*claim"):
        LumpedThermalProtocol.from_dict(payload)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("absorbed_optical_power_W_m2", -1.0, "non-negative"),
        ("ambient_temperature_K", 0.0, "positive"),
        ("areal_heat_capacity_J_m2_K", 0.0, "positive"),
        ("heat_transfer_coefficient_W_m2_K", -1.0, "non-negative"),
        ("emissivity", 1.01, r"\[0, 1\]"),
        ("maximum_temperature_K", 299.0, "must exceed"),
    ],
)
def test_thermal_protocol_rejects_unphysical_fields(field, value, message):
    with pytest.raises(ValueError, match=message):
        _thermal_protocol(**{field: value})


def test_power_ledger_uses_one_explicit_control_volume():
    protocol = _thermal_protocol(emissivity=0.8)
    ledger = thermal_power_ledger(
        protocol,
        temperature_K=320.0,
        terminal_electrical_export_W_m2=200.0,
    )

    assert ledger.absorbed_optical_power_W_m2 == 800.0
    assert ledger.terminal_electrical_export_W_m2 == 200.0
    assert ledger.linear_heat_rejection_W_m2 == 400.0
    assert ledger.radiative_heat_rejection_W_m2 > 0.0
    assert ledger.net_heating_W_m2 == (
        ledger.absorbed_optical_power_W_m2
        + ledger.constant_internal_heat_W_m2
        - ledger.terminal_electrical_export_W_m2
        - ledger.linear_heat_rejection_W_m2
        - ledger.radiative_heat_rejection_W_m2
    )

    with pytest.raises(ValueError, match="violates"):
        dataclasses.replace(ledger, net_heating_W_m2=ledger.net_heating_W_m2 + 1.0)


def test_steady_zero_heat_is_exactly_ambient():
    protocol = _thermal_protocol(absorbed_optical_power_W_m2=200.0)
    result = solve_lumped_thermal_steady_state(
        protocol,
        terminal_electrical_export_W_m2=200.0,
    )

    assert result.certified
    assert result.temperature_K == 300.0
    assert result.temperature_rise_K == 0.0
    assert result.ledger.net_heating_W_m2 == 0.0


def test_steady_linear_heat_rejection_matches_analytic_solution():
    result = solve_lumped_thermal_steady_state(
        _thermal_protocol(),
        terminal_electrical_export_W_m2=200.0,
    )

    assert result.certified
    assert result.temperature_K == pytest.approx(330.0, abs=1.0e-10)
    assert result.ledger.linear_heat_rejection_W_m2 == pytest.approx(600.0)
    assert abs(result.ledger.net_heating_W_m2) <= 1.0e-8


def test_radiation_reduces_temperature_and_closes_first_law():
    linear = solve_lumped_thermal_steady_state(
        _thermal_protocol(),
        terminal_electrical_export_W_m2=200.0,
    )
    radiative = solve_lumped_thermal_steady_state(
        _thermal_protocol(emissivity=0.85),
        terminal_electrical_export_W_m2=200.0,
    )

    assert 300.0 < radiative.temperature_K < linear.temperature_K
    assert radiative.ledger.radiative_heat_rejection_W_m2 > 0.0
    assert abs(radiative.ledger.net_heating_W_m2) <= 1.0e-8


def test_steady_solver_fails_closed_on_impossible_source_or_envelope():
    protocol = _thermal_protocol()
    with pytest.raises(ThermalEnergySourceError, match="exceeds"):
        solve_lumped_thermal_steady_state(
            protocol,
            terminal_electrical_export_W_m2=801.0,
        )
    with pytest.raises(ThermalEnergySourceError, match="exceeds"):
        solve_lumped_thermal_steady_state(
            _thermal_protocol(
                absorbed_optical_power_W_m2=100.0,
                constant_internal_heat_W_m2=200.0,
            ),
            terminal_electrical_export_W_m2=101.0,
        )
    with pytest.raises(ThermalSteadyStateError, match="no declared rejection"):
        solve_lumped_thermal_steady_state(
            _thermal_protocol(
                heat_transfer_coefficient_W_m2_K=0.0,
                emissivity=0.0,
            ),
            terminal_electrical_export_W_m2=200.0,
        )
    with pytest.raises(ThermalSteadyStateError, match="below maximum"):
        solve_lumped_thermal_steady_state(
            _thermal_protocol(
                heat_transfer_coefficient_W_m2_K=1.0,
                maximum_temperature_K=350.0,
            ),
            terminal_electrical_export_W_m2=200.0,
        )


def test_linear_transient_matches_closed_form_and_energy_ledger():
    thermal = _thermal_protocol()
    integration = _integration_protocol()
    result = run_lumped_thermal_transient(
        thermal,
        integration,
        terminal_electrical_export_W_m2=200.0,
    )

    time_constant_s = (
        thermal.areal_heat_capacity_J_m2_K / thermal.heat_transfer_coefficient_W_m2_K
    )
    expected = 330.0 - 30.0 * np.exp(-result.time_s / time_constant_s)
    np.testing.assert_allclose(result.temperature_K, expected, atol=1.0e-9)
    assert result.certified
    assert result.max_abs_energy_balance_residual_J_m2 <= 1.0e-7
    assert result.temperature_K.flags.writeable is False
    assert result.energy_balance_residual_J_m2.flags.writeable is False
    with pytest.raises(ValueError, match="residual trace"):
        dataclasses.replace(
            result,
            energy_balance_residual_J_m2=(
                result.energy_balance_residual_J_m2
                + np.linspace(0.0, 1.0, result.time_s.size)
            ),
            max_abs_energy_balance_residual_J_m2=1.0,
            certified=False,
        )


def test_transient_zero_heat_at_ambient_is_exact_and_hashes_bind_inputs():
    thermal = _thermal_protocol(absorbed_optical_power_W_m2=200.0)
    integration = _integration_protocol(duration_s=10.0)
    first = run_lumped_thermal_transient(
        thermal,
        integration,
        terminal_electrical_export_W_m2=200.0,
    )
    second = run_lumped_thermal_transient(
        thermal,
        dataclasses.replace(integration, duration_s=20.0),
        terminal_electrical_export_W_m2=200.0,
    )

    assert np.array_equal(first.temperature_K, np.full(41, 300.0))
    assert np.array_equal(first.net_heating_W_m2, np.zeros(41))
    assert first.mapping_sha256 != second.mapping_sha256
    assert first.thermal_protocol_sha256 == thermal.sha256
    assert first.integration_protocol_sha256 == integration.sha256


def test_radiative_transient_approaches_registered_steady_state():
    thermal = _thermal_protocol(
        areal_heat_capacity_J_m2_K=100.0,
        emissivity=0.85,
    )
    steady = solve_lumped_thermal_steady_state(
        thermal,
        terminal_electrical_export_W_m2=200.0,
    )
    transient = run_lumped_thermal_transient(
        thermal,
        _integration_protocol(duration_s=100.0),
        terminal_electrical_export_W_m2=200.0,
    )

    assert transient.temperature_K[-1] == pytest.approx(
        steady.temperature_K,
        abs=1.0e-6,
    )
    assert transient.max_abs_energy_balance_residual_J_m2 <= 1.0e-7


def test_transient_rejects_initial_or_evolved_temperature_above_envelope():
    thermal = _thermal_protocol(maximum_temperature_K=320.0)
    with pytest.raises(ThermalTransientError, match="initial temperature"):
        run_lumped_thermal_transient(
            thermal,
            _integration_protocol(initial_temperature_K=321.0),
            terminal_electrical_export_W_m2=200.0,
        )
    with pytest.raises(ThermalTransientError, match="temperature envelope"):
        run_lumped_thermal_transient(
            thermal,
            _integration_protocol(duration_s=200.0),
            terminal_electrical_export_W_m2=200.0,
        )


def test_public_ledger_rejects_negative_sources():
    with pytest.raises(ValueError, match="non-negative"):
        ThermalPowerLedger(
            temperature_K=300.0,
            absorbed_optical_power_W_m2=-1.0,
            constant_internal_heat_W_m2=0.0,
            terminal_electrical_export_W_m2=0.0,
            linear_heat_rejection_W_m2=0.0,
            radiative_heat_rejection_W_m2=0.0,
            net_heating_W_m2=-1.0,
        )

    with pytest.raises(ThermalEnergySourceError, match="exceeds"):
        ThermalPowerLedger(
            temperature_K=300.0,
            absorbed_optical_power_W_m2=100.0,
            constant_internal_heat_W_m2=200.0,
            terminal_electrical_export_W_m2=101.0,
            linear_heat_rejection_W_m2=0.0,
            radiative_heat_rejection_W_m2=0.0,
            net_heating_W_m2=199.0,
        )

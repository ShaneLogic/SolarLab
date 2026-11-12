"""Unit tests for P04-13 external circuits/thermal balance/tandem matching and P05-04 TPV decay."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.experiments.external_circuit import (
    EXTERNAL_AC_UNIT_SCHEMA,
    EXTERNAL_DC_UNIT_SCHEMA,
    ExternalACCircuitProtocol,
    ExternalCircuitError,
    ExternalCircuitProtocol,
    ExternalCircuitTopologyError,
    apply_external_circuit,
    apply_external_circuit_steady,
    compose_external_ac_impedance,
    map_external_circuit_branch,
)
from solarlab.experiments.tandem_jv import series_match_jv
from solarlab.experiments.thermal_balance import (
    STEFAN_BOLTZMANN_W_M2_K4,
    THERMAL_UNIT_SCHEMA,
    LumpedThermalProtocol,
    ThermalBalanceError,
    ThermalEnergySourceError,
    ThermalIntegrationProtocol,
    ThermalSteadyStateError,
    run_lumped_thermal_transient,
    solve_lumped_thermal_steady_state,
    thermal_power_ledger,
)
from solarlab.experiments.tpv import fit_tpv_decay


def test_external_dc_mapping_conventions_losses_and_folding_guard() -> None:
    # Ideal linear photovoltaic source: J_dev(V) = 200 - 200 * V  (J_sc = 200 A/m^2, V_oc = 1.0 V)
    v_junc = np.linspace(0.0, 1.1, 12)
    j_dev = 200.0 - 200.0 * v_junc

    prot = ExternalCircuitProtocol(
        series_resistance_ohm_m2=5e-4,
        shunt_resistance_ohm_m2=0.05,
    )
    branch = map_external_circuit_branch(
        v_junc, j_dev, prot, direction="forward", incident_power_W_m2=1000.0
    )

    # Shunt sits across V_junction; series drop uses J_terminal:
    expected_shunt = v_junc / 0.05
    expected_j_term = j_dev - expected_shunt
    expected_v_term = v_junc - expected_j_term * 5e-4

    np.testing.assert_allclose(branch.J_shunt, expected_shunt, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(branch.J_terminal, expected_j_term, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(branch.V_terminal, expected_v_term, rtol=0.0, atol=1e-14)

    # Power conservation between junction output and terminal export + parasitic Joule heat:
    # V_junc * J_dev == V_term * J_term + J_term^2 * R_s + V_junc^2 / R_sh
    p_junction = v_junc * j_dev
    p_accounted = (
        branch.power_terminal_W_m2 + branch.series_loss_W_m2 + branch.shunt_loss_W_m2
    )
    np.testing.assert_allclose(p_accounted, p_junction, rtol=1e-14, atol=1e-12)

    # Analytic V_oc with R_sh = 0.05 ohm*m^2: 200 - 200*V - 20*V = 0 => V_oc = 200 / 220 = 10/11 V
    assert branch.metrics.open_circuit_voltage_V == pytest.approx(
        10.0 / 11.0, rel=1e-12
    )
    assert not branch.V_terminal.flags.writeable

    # Dual-branch wrapper & steady tuple wrapper
    dual = apply_external_circuit(
        (v_junc, j_dev, v_junc[::-1], j_dev[::-1]),
        prot,
        incident_power_W_m2=1000.0,
    )
    assert dual.hysteresis_index == pytest.approx(0.0, abs=1e-14)
    steady_res = apply_external_circuit_steady(
        (v_junc, j_dev), prot, incident_power_W_m2=1000.0
    )
    assert steady_res.branch.metrics.open_circuit_voltage_V == pytest.approx(
        10.0 / 11.0, rel=1e-12
    )

    # Folding guard: non-monotonic J_dev where R_s * dJ/dV > 1 folds V_terminal
    v_fold = np.array([0.0, 0.5, 1.0])
    j_fold = np.array([0.0, 200.0, 100.0])  # dJ/dV = +400 A/(m^2 V) on [0, 0.5]
    with pytest.raises(ExternalCircuitTopologyError):
        map_external_circuit_branch(
            v_fold,
            j_fold,
            ExternalCircuitProtocol(series_resistance_ohm_m2=0.01),
            direction="forward",
        )

    with pytest.raises(ExternalCircuitError):
        ExternalCircuitProtocol(series_resistance_ohm_m2=-1e-4)
    with pytest.raises(ExternalCircuitError):
        ExternalCircuitProtocol(shunt_resistance_ohm_m2=0.0)


def test_external_ac_composition_and_unit_schema_separation() -> None:
    # Unit schemas keep electrical F/m^2 strictly distinct from thermal J/(m^2*K)
    assert EXTERNAL_DC_UNIT_SCHEMA["series_resistance_ohm_m2"] == "ohm*m^2"
    assert EXTERNAL_AC_UNIT_SCHEMA["external_capacitance_F_m2"] == "F/m^2"
    assert THERMAL_UNIT_SCHEMA["areal_heat_capacity_J_m2_K"] == "J/(m^2*K)"
    assert "areal_heat_capacity_J_m2_K" not in EXTERNAL_AC_UNIT_SCHEMA
    assert "external_capacitance_F_m2" not in THERMAL_UNIT_SCHEMA

    freqs = np.array([0.0, 1e2, 1e5, 1e9], dtype=float)
    omega = 2.0 * np.pi * freqs
    # Synthetic device: parallel G_dev = 25 S/m^2 and C_dev = 1e-4 F/m^2
    y_dev = 25.0 + 1j * omega * 1e-4

    # 1. Zero-parasitic composition is the exact identity
    zero_res = compose_external_ac_impedance(
        (freqs, y_dev), ExternalACCircuitProtocol()
    )
    np.testing.assert_allclose(
        zero_res.terminal_admittance_S_m2, y_dev, rtol=0.0, atol=1e-15
    )

    # 2. Full R_s + R_sh + C_ext composition matches exact network impedance
    rs = 2e-4
    rsh = 0.1
    cext = 5e-5
    ac_prot = ExternalACCircuitProtocol(
        series_resistance_ohm_m2=rs,
        shunt_resistance_ohm_m2=rsh,
        external_capacitance_F_m2=cext,
    )
    ac_res = compose_external_ac_impedance((freqs, y_dev), ac_prot)

    expected_y_junc = y_dev + (1.0 / rsh) + 1j * omega * cext
    expected_z_term = rs + 1.0 / expected_y_junc
    expected_y_term = 1.0 / expected_z_term

    np.testing.assert_allclose(
        ac_res.junction_admittance_S_m2, expected_y_junc, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        ac_res.terminal_admittance_S_m2, expected_y_term, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        ac_res.impedance_area_ohm_m2, expected_z_term, rtol=1e-14, atol=1e-14
    )
    # High-frequency limit approaches R_s
    assert ac_res.impedance_area_ohm_m2[-1].real == pytest.approx(rs, rel=1e-4)


def test_lumped_thermal_ledger_steady_state_and_transient_energy_closure() -> None:
    prot = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=900.0,
        ambient_temperature_K=300.0,
        areal_heat_capacity_J_m2_K=500.0,
        heat_transfer_coefficient_W_m2_K=15.0,
        emissivity=0.85,
        maximum_temperature_K=420.0,
    )

    # Exporting more power than absorbed without external drive is refused
    with pytest.raises(ThermalEnergySourceError):
        thermal_power_ledger(310.0, prot, exported_electrical_power_W_m2=950.0)

    # Steady-state temperature root with temperature-dependent exported power P_out(T)
    def p_export(temp_K: float) -> float:
        return max(0.0, 200.0 * (1.0 - 0.004 * (temp_K - 300.0)))

    steady = solve_lumped_thermal_steady_state(
        prot, exported_electrical_power_fn=p_export
    )
    assert steady.converged
    assert 300.0 < steady.temperature_K < 420.0
    # Verify Stefan-Boltzmann + Newton cooling balance directly
    dt = steady.temperature_K - 300.0
    expected_cool = 15.0 * dt + 0.85 * STEFAN_BOLTZMANN_W_M2_K4 * (
        steady.temperature_K**4 - 300.0**4
    )
    assert expected_cool == pytest.approx(
        900.0 - p_export(steady.temperature_K), abs=1e-8
    )

    # Unbracketed maximum_temperature_K fails closed
    too_tight = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=900.0,
        ambient_temperature_K=300.0,
        heat_transfer_coefficient_W_m2_K=15.0,
        emissivity=0.85,
        maximum_temperature_K=305.0,
    )
    with pytest.raises(ThermalSteadyStateError):
        solve_lumped_thermal_steady_state(too_tight)

    with pytest.raises(ThermalBalanceError):
        LumpedThermalProtocol(
            absorbed_optical_power_W_m2=500.0,
            heat_transfer_coefficient_W_m2_K=0.0,
            emissivity=0.0,
        )

    # Transient integration relaxes toward the exact steady root and closes energy
    t_eval = np.linspace(0.0, 250.0, 51)
    tr = run_lumped_thermal_transient(
        prot,
        t_eval,
        initial_temperature_K=300.0,
        exported_electrical_power_fn=lambda _t, temp_K: p_export(temp_K),
        integration=ThermalIntegrationProtocol(method="DOP853"),
    )
    assert tr.certified
    assert tr.max_abs_energy_closure_residual_J_m2 < 1e-6
    assert tr.temperature_K[-1] == pytest.approx(steady.temperature_K, rel=1e-4)


def test_series_match_jv_analytic_open_circuit_and_non_overlap_guard() -> None:
    # Linear sub-cells:
    # Top:    V_top(J) = 1.2 - 0.002 * J   (J_sc = 600? No: at V=0 => J=600; restrict J in [-50, 150])
    # Bottom: V_bot(J) = 0.8 - 0.001 * J
    v_top = np.linspace(-0.1, 1.3, 141)
    j_top = (1.2 - v_top) / 0.002  # J from 650 down to -50
    v_bot = np.linspace(-0.1, 0.9, 101)
    j_bot = (0.8 - v_bot) / 0.001  # J from 900 down to -100

    v_tan, j_tan = series_match_jv(
        j_top, v_top, j_bot, v_bot, V_junction=0.05, n_points=81
    )
    assert np.all(np.diff(v_tan) > 0.0)
    # J = 0 is explicitly injected => V_oc = 1.2 + 0.8 + 0.05 = 2.05 V
    assert np.any(np.isclose(j_tan, 0.0, atol=1e-15))
    v_at_zero = float(np.interp(0.0, j_tan[::-1], v_tan[::-1]))
    assert v_at_zero == pytest.approx(2.05, abs=1e-12)

    # Non-overlapping current ranges raise ValueError
    with pytest.raises(ValueError, match="do not overlap"):
        series_match_jv(
            np.array([100.0, 200.0]),
            np.array([0.0, 1.0]),
            np.array([-50.0, 50.0]),
            np.array([0.0, 1.0]),
        )


def test_fit_tpv_decay_single_exponential_and_all_refusal_modes() -> None:
    tau_true = 1.25e-5
    amp_true = 4.5e-3
    t_grid = np.linspace(0.0, 6.0 * tau_true, 60)
    y_clean = amp_true * np.exp(-t_grid / tau_true)

    good = fit_tpv_decay(t_grid, y_clean)
    assert good.status == "single_exponential"
    assert good.non_identifiable_reason is None
    assert good.tau_s == pytest.approx(tau_true, rel=1e-6)
    assert good.amplitude_V == pytest.approx(amp_true, rel=1e-6)
    assert good.r_squared is not None and good.r_squared > 0.999999

    # 1. Zero / negative amplitude -> no_signal
    no_sig = fit_tpv_decay(t_grid, np.zeros_like(t_grid))
    assert no_sig.tau_s is None
    assert no_sig.status == "no_signal"
    assert no_sig.non_identifiable_reason is not None

    # 2. Sign change -> sign_change
    y_flip = y_clean.copy()
    y_flip[20:] = -0.2 * amp_true
    flipped = fit_tpv_decay(t_grid, y_flip)
    assert flipped.tau_s is None
    assert flipped.status == "sign_change"
    assert flipped.non_identifiable_reason is not None

    # 3. Non-monotonic bounce -> non_monotonic
    y_bump = y_clean.copy()
    y_bump[15:25] += 0.15 * amp_true
    bumped = fit_tpv_decay(t_grid, y_bump)
    assert bumped.tau_s is None
    assert bumped.status == "non_monotonic"
    assert bumped.non_identifiable_reason is not None

    # 4. Truncated window (< 1 e-fold) -> insufficient_decay_window
    t_short = np.linspace(0.0, 0.4 * tau_true, 20)
    short_fit = fit_tpv_decay(t_short, amp_true * np.exp(-t_short / tau_true))
    assert short_fit.tau_s is None
    assert short_fit.status == "insufficient_decay_window"
    assert short_fit.non_identifiable_reason is not None
    assert not math.isnan(short_fit.amplitude_V)

    # 5. Bi-exponential mixture -> not_single_exponential
    t_bi = np.geomspace(1e-8, 2e-4, 80) - 1e-8
    y_bi = 0.5 * amp_true * np.exp(-t_bi / 1e-6) + 0.5 * amp_true * np.exp(
        -t_bi / 5e-5
    )
    bi_fit = fit_tpv_decay(t_bi, y_bi)
    assert bi_fit.tau_s is None
    assert bi_fit.status == "not_single_exponential"
    assert bi_fit.non_identifiable_reason is not None

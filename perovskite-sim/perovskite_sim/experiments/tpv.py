"""Charge-conserving transient photovoltage with a matched unpulsed control.

The terminal voltage evolves continuously with the density equations under
zero external Maxwell current. Finite-time preparation does not imply ionic
equilibrium; subtracting an identically prepared control isolates the pulse
response. A decay time is reported only when one exponential is identifiable.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.tpv import TPVDecayFit, TPVNumericalSettings, TPVResult
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    build_material_arrays,
    MaterialArrays,
)
from perovskite_sim.experiments.jv_sweep import (
    _integrate_step,
    build_electrical_grid,
)
from perovskite_sim.experiments.open_circuit import (
    OpenCircuitError,
    OpenCircuitSystem,
    integrate_open_circuit,
)
from perovskite_sim.experiments.protocol import (
    DCSettleCriterion,
    ExperimentProtocol,
    IlluminationStep,
    ProtocolMode,
    SamplingProtocol,
    ScanProtocol,
    VocSearchProtocol,
    resolve_experiment_protocol,
)

ProgressCallback = Callable[[str, int, int, str], None]


def _tpv_sampling_times(
    t_pulse: float,
    t_decay: float,
    n_points: int,
) -> np.ndarray:
    if n_points < 10:
        raise ValueError(f"n_points must be >= 10, got {n_points}")
    t_pulse_pts = min(n_points - 2, max(int(n_points * t_pulse / t_decay), 10))
    t_decay_pts = n_points - t_pulse_pts
    if t_decay_pts < 0:
        raise ValueError(
            "n_points is too small for the requested pulse/decay sampling"
        )
    return np.concatenate([
        np.linspace(0, t_pulse, t_pulse_pts, endpoint=False),
        np.linspace(t_pulse, t_decay, t_decay_pts + 1),
    ])


def build_tpv_experiment_protocol(
    stack: DeviceStack,
    *,
    delta_G_frac: float = 0.05,
    t_pulse: float = 1e-6,
    t_decay: float = 50e-6,
    n_points: int = 200,
    voc_search: VocSearchProtocol | None = None,
    implicit_legacy_protocol: bool = False,
) -> ExperimentProtocol:
    """Describe TPV preconditioning, light pulse, decay, and samples."""

    if not np.isfinite(delta_G_frac) or not 0.0 < delta_G_frac < 1.0:
        raise ValueError(f"delta_G_frac must be finite and in (0, 1), got {delta_G_frac}")
    if not np.isfinite(t_pulse) or t_pulse <= 0.0:
        raise ValueError(f"t_pulse must be finite and positive, got {t_pulse}")
    if not np.isfinite(t_decay) or t_decay <= t_pulse:
        raise ValueError(f"t_decay must be finite and exceed t_pulse, got {t_decay}")
    times = _tpv_sampling_times(t_pulse, t_decay, n_points)
    settle = 1.0e-3
    search = voc_search or VocSearchProtocol()
    return ExperimentProtocol(
        experiment="tpv",
        initial_state_source="finite_time_illuminated_preconditioned",
        pre_bias_V=0.0,
        soak_duration_s=settle,
        dwell_duration_s=None,
        illumination_history=(
            IlluminationStep(
                phase="short_circuit_preconditioning",
                condition="baseline",
                duration_s=settle,
                intensity_suns=1.0,
                source_reference="stack_baseline_generation",
            ),
            IlluminationStep(
                phase="adaptive_open_circuit_search",
                condition="baseline",
                intensity_suns=1.0,
                source_reference="stack_baseline_generation",
            ),
            IlluminationStep(
                phase="photovoltage_pulse",
                condition="pulse",
                duration_s=float(t_pulse),
                intensity_suns=1.0,
                relative_generation_change=float(delta_G_frac),
                source_reference="stack_baseline_generation",
            ),
            IlluminationStep(
                phase="open_circuit_decay",
                condition="baseline",
                duration_s=float(t_decay - t_pulse),
                intensity_suns=1.0,
                source_reference="stack_baseline_generation",
            ),
        ),
        temperature_K=float(stack.T),
        scan=ScanProtocol(
            axis="time_s",
            direction="forward_time",
            start=0.0,
            stop=float(t_decay),
        ),
        ac_excitation=None,
        dc_settle=DCSettleCriterion(kind="finite_time", duration_s=settle),
        sampling=SamplingProtocol(
            axis="time_s",
            mode="piecewise_linear",
            values=tuple(times),
        ),
        voc_search=search,
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def _find_voc(
    x: np.ndarray,
    y_ss: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_guess: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    search: VocSearchProtocol | None = None,
) -> tuple[float, np.ndarray]:
    """Bracket zero instantaneous current after a reproducible preparation.

    The coarse scan establishes a lower-bracket state. Every root candidate
    starts from that same state and receives the same two declared dwell
    intervals. The returned state is the state evaluated at the accepted root;
    no additional unchecked settling changes it afterwards.
    """
    protocol = search or VocSearchProtocol()
    if protocol.fallback != "error" or protocol.warm_start != "fixed_lower_bracket_state":
        raise ValueError("open-circuit search requires error fallback and fixed_lower_bracket_state")
    system = OpenCircuitSystem(x, stack, mat)
    V_lo = protocol.coarse_start_V
    V_hi = max(V_guess, protocol.minimum_guess_V) * protocol.coarse_upper_guess_factor
    if not np.isfinite(V_hi) or V_hi <= V_lo:
        raise OpenCircuitError("invalid open-circuit voltage bracket")

    def fixed_bias_current(t, state, voltage):
        system.validate_state(state, voltage, t)
        _, voltage_rate, _ = system.rates(t, state, voltage)
        return system.capacitance_A * voltage_rate

    y = y_ss.copy()
    previous = None
    bracket = None
    for k, voltage in enumerate(np.linspace(V_lo, V_hi, protocol.coarse_points)):
        end = (k + 1) * protocol.coarse_dwell_s
        y = _integrate_step(
            x, y, stack, mat, float(voltage), k * protocol.coarse_dwell_s,
            end, rtol, atol,
        )
        current = fixed_bias_current(end, y, float(voltage))
        if previous is not None and previous[1] * current <= 0.0:
            bracket = (previous[0], float(voltage))
            y_base = previous[2]
            break
        previous = (float(voltage), current, y.copy())
    if bracket is None:
        raise OpenCircuitError(f"open-circuit current does not cross zero in [{V_lo:g}, {V_hi:g}] V")

    candidates: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(voltage):
        voltage = float(voltage)
        if voltage not in candidates:
            mid = protocol.bisection_dwell_s
            end = mid + protocol.final_settle_s
            candidate = _integrate_step(x, y_base.copy(), stack, mat, voltage,
                                        0.0, mid, rtol, atol)
            candidate = _integrate_step(x, candidate, stack, mat, voltage,
                                        mid, end, rtol, atol)
            current = fixed_bias_current(end, candidate, voltage)
            candidates[voltage] = (current, candidate.copy())
        return candidates[voltage][0]

    if evaluate(bracket[0]) * evaluate(bracket[1]) > 0.0:
        raise OpenCircuitError("open-circuit bracket does not survive the declared final preparation")
    try:
        voltage = float(brentq(
            evaluate, *bracket, xtol=min(protocol.bisection_tolerance_V, 1e-10),
            rtol=4 * np.finfo(float).eps, maxiter=protocol.bisection_max_steps,
        ))
    except (ValueError, RuntimeError) as exc:
        raise OpenCircuitError(f"open-circuit voltage root failed: {exc}") from exc
    current = evaluate(voltage)
    if not np.isfinite(current) or abs(current) > 1e-4:
        raise OpenCircuitError(f"initial open-circuit residual exceeds 1e-4 A/m2: {current:.9g}")
    return voltage, candidates[voltage][1]


def fit_tpv_decay(
    t: np.ndarray, delta_voltage: np.ndarray, *, noise_floor_V: float = 1e-8,
) -> TPVDecayFit:
    """Fit only a resolved, monotone, approximately single-exponential decay.

    Inputs start at pulse turn-off and use the matched unpulsed voltage as
    reference. A window shorter than one observed e-fold is not identifiable.
    """
    t = np.asarray(t, dtype=float)
    delta = np.asarray(delta_voltage, dtype=float)
    if (t.ndim != 1 or delta.shape != t.shape or not np.all(np.isfinite(t))
            or not np.all(np.isfinite(delta)) or np.any(np.diff(t) <= 0.0)):
        return TPVDecayFit(None, 0.0, "invalid_data", 0)
    if not np.isfinite(noise_floor_V) or noise_floor_V <= 0.0:
        raise ValueError("noise_floor_V must be finite and positive")
    if t.size < 5:
        return TPVDecayFit(None, float(delta[0]) if t.size else 0.0, "insufficient_data", t.size)
    peak = float(np.max(np.abs(delta)))
    amplitude = float(delta[0])
    if peak <= noise_floor_V:
        return TPVDecayFit(None, 0.0, "no_signal", t.size)
    sign = float(np.sign(amplitude))
    signal = sign * delta
    if sign == 0.0 or np.any(signal < -noise_floor_V):
        return TPVDecayFit(None, amplitude, "sign_change", t.size)
    if np.any(np.diff(signal) > 3.0 * noise_floor_V):
        return TPVDecayFit(None, amplitude, "non_monotonic", t.size)
    mask = signal > max(0.05 * peak, noise_floor_V)
    count = int(np.count_nonzero(mask))
    if count < 5:
        return TPVDecayFit(None, amplitude, "insufficient_data", count)
    if signal[mask][-1] > signal[mask][0] / np.e:
        return TPVDecayFit(None, amplitude, "insufficient_decay_window", count)
    relative_time = t[mask] - t[0]
    span = float(relative_time[-1])
    log_signal = np.log(signal[mask])
    slope, intercept = np.polyfit(relative_time / span, log_signal, 1)
    prediction = intercept + slope * relative_time / span
    variance = float(np.sum((log_signal - np.mean(log_signal)) ** 2))
    r_squared = 1.0 - float(np.sum((log_signal - prediction) ** 2)) / variance
    error = float(np.max(np.abs(np.exp(prediction) - signal[mask])) / peak)
    if slope >= 0.0 or r_squared < 0.995 or error > 0.02:
        return TPVDecayFit(None, amplitude, "not_single_exponential", count, r_squared, error)
    return TPVDecayFit(float(-span / slope), float(sign * np.exp(intercept)),
                       "accepted", count, r_squared, error)


def _fit_decay_tau(
    t: np.ndarray, V: np.ndarray, V_oc: float,
) -> tuple[float | None, float]:
    """Compatibility wrapper for a decay measured relative to a fixed baseline."""
    fit = fit_tpv_decay(t, np.asarray(V) - V_oc)
    return fit.tau_s, fit.amplitude_V


def run_tpv(
    stack: DeviceStack,
    N_grid: int = 80,
    delta_G_frac: float = 0.05,
    t_pulse: float = 1e-6,
    t_decay: float = 50e-6,
    n_points: int = 200,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
    experiment_protocol: ExperimentProtocol | None = None,
    protocol_mode: ProtocolMode = "compatibility",
    voc_search: VocSearchProtocol | None = None,
    max_step: float | None = None,
    voltage_atol: float = 1e-9,
    max_voltage_error_V: float = 1e-6,
    max_current_error_A_m2: float = 0.05,
) -> TPVResult:
    """Run a transient photovoltage experiment.

    Parameters
    ----------
    stack : DeviceStack
        Device configuration.
    N_grid : int
        Nominal total grid intervals, using the shared electrical grid builder.
    delta_G_frac : float
        Fractional generation perturbation (e.g. 0.05 = 5% pulse).
    t_pulse : float
        Duration of the light pulse [s].
    t_decay : float
        Total observation window after pulse onset [s].
    n_points : int
        Number of time points in the output.
    rtol, atol : float
        ODE solver tolerances.
    progress : callable, optional
        Progress callback: fn(stage, current, total, message).

    Returns
    -------
    TPVResult
        Pulsed and unpulsed voltages, current/charge evidence and an optional
        single-exponential decay time. Invalid trajectories raise.
    """
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if delta_G_frac <= 0 or delta_G_frac >= 1:
        raise ValueError(f"delta_G_frac must be in (0, 1), got {delta_G_frac}")
    if t_pulse <= 0:
        raise ValueError(f"t_pulse must be positive, got {t_pulse}")
    if t_decay <= t_pulse:
        raise ValueError(f"t_decay must exceed t_pulse, got {t_decay}")
    resolved_protocol = resolve_experiment_protocol(
        experiment_protocol,
        build_tpv_experiment_protocol(
            stack,
            delta_G_frac=delta_G_frac,
            t_pulse=t_pulse,
            t_decay=t_decay,
            n_points=n_points,
            voc_search=voc_search,
            implicit_legacy_protocol=True,
        ),
        mode=protocol_mode,
    )

    if max_step is None:
        max_step = min(t_pulse / 5.0, (t_decay - t_pulse) / 20.0)
    for name, value in (("max_step", max_step), ("rtol", rtol), ("atol", atol),
                        ("voltage_atol", voltage_atol),
                        ("max_voltage_error_V", max_voltage_error_V),
                        ("max_current_error_A_m2", max_current_error_A_m2)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    x = build_electrical_grid(stack, N_grid)
    mat_ss = build_material_arrays(x, stack)
    if mat_ss.G_optical is None:
        from perovskite_sim.physics.generation import beer_lambert_generation
        G_baseline = beer_lambert_generation(x, mat_ss.alpha, stack.Phi)
        mat_ss = replace(mat_ss, G_optical=G_baseline)
    mat_pulse = replace(
        mat_ss, G_optical=mat_ss.G_optical * (1.0 + delta_G_frac),
    )
    y_illum = solve_illuminated_ss(
        x, stack, V_app=0.0, rtol=rtol, atol=atol, mat=mat_ss,
        t_settle=resolved_protocol.soak_duration_s,
    )
    search = resolved_protocol.voc_search
    assert search is not None
    V_oc, y_oc = _find_voc(
        x,
        y_illum,
        stack,
        mat_ss,
        V_guess=abs(stack.operating_built_in_potential()),
        rtol=rtol,
        atol=atol,
        search=search,
    )

    if progress is not None:
        progress("tpv", 0, 4, f"V_oc={V_oc:.6f} V")
    t_arr = _tpv_sampling_times(t_pulse, t_decay, n_points)
    traces = []
    for run_index, pulse_material in enumerate((mat_pulse, mat_ss)):
        y, voltage = y_oc.copy(), V_oc
        phases = []
        for phase_index, (material, times) in enumerate((
            (pulse_material, t_arr[t_arr <= t_pulse]),
            (mat_ss, t_arr[t_arr >= t_pulse]),
        )):
            phase = integrate_open_circuit(
                OpenCircuitSystem(x, stack, material), y, voltage, times,
                rtol=rtol, atol=atol, voltage_atol=voltage_atol,
                max_step=max_step, max_voltage_error_V=max_voltage_error_V,
                max_current_error_A_m2=max_current_error_A_m2,
            )
            phases.append(phase)
            y, voltage = phase.y[:, -1], float(phase.V[-1])
            if progress is not None:
                progress("tpv", 2 * run_index + phase_index + 1, 4,
                         f"t={times[-1]:.3e} s, V={voltage:.6f} V")
        joined = {
            field: np.concatenate((getattr(phases[0], field)[:-1], getattr(phases[1], field)))
            for field in ("V", "J", "valid", "max_face_current_A_m2",
                          "interval_current_residual_A_m2", "charge_voltage_error_V")
        }
        joined["charge_voltage_error_V"][len(phases[0].t) - 1:] += phases[0].charge_voltage_error_V[-1]
        traces.append(joined)
    pulsed, reference = traces
    delta_voltage = pulsed["V"] - reference["V"]
    voltage_error = pulsed["charge_voltage_error_V"] + reference["charge_voltage_error_V"]
    signal_amplitude = float(np.max(np.abs(delta_voltage)))
    error_limit = max_voltage_error_V
    if signal_amplitude > max(1e-8, 10 * voltage_atol):
        error_limit = min(error_limit, 0.01 * signal_amplitude)
    if voltage_error[-1] > error_limit:
        raise OpenCircuitError(
            f"paired TPV charge error {voltage_error[-1]:.6g} V exceeds "
            f"the signal-dependent limit {error_limit:.6g} V"
        )
    decay_mask = t_arr >= t_pulse
    fit = fit_tpv_decay(
        t_arr[decay_mask] - t_pulse, delta_voltage[decay_mask],
        noise_floor_V=max(1e-8, 10 * voltage_atol, 5 * float(voltage_error[-1])),
    )
    reference_history = list(resolved_protocol.illumination_history)
    reference_history[2] = replace(
        reference_history[2], phase="unpulsed_reference", condition="baseline",
        relative_generation_change=None,
    )
    return TPVResult(
        t=t_arr, V=pulsed["V"], J=pulsed["J"], V_oc=V_oc,
        tau=fit.tau_s, delta_V0=fit.amplitude_V, protocol=resolved_protocol,
        fit=fit, V_reference=reference["V"], delta_V=delta_voltage,
        valid=pulsed["valid"] & reference["valid"],
        max_face_current_A_m2=np.maximum(pulsed["max_face_current_A_m2"], reference["max_face_current_A_m2"]),
        interval_current_residual_A_m2=np.maximum(pulsed["interval_current_residual_A_m2"], reference["interval_current_residual_A_m2"]),
        charge_voltage_error_V=voltage_error,
        reference_protocol=replace(resolved_protocol, illumination_history=tuple(reference_history)),
        numerical_settings=TPVNumericalSettings(
            rtol, atol, voltage_atol, max_step, max_voltage_error_V,
            max_current_error_A_m2,
        ),
    )

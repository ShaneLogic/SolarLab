"""Transient photovoltage (TPV) experiment with paired unpulsed control baseline.

Contract (P05-04 clauses 1 & 5):
1. Preconditioning is explicit and reproducible:
   - Ion-free ``CompiledDevice1D`` solves the exact 1-sun open-circuit steady
     state ``(state_oc, V_oc)`` via bracketed Brent root finding from a fixed
     lower bracket ``0.0 V`` (never an order-dependent floating guess).
   - Mobile-ion ``CompiledIonicDevice1D`` executes a finite-time illuminated
     open-circuit preconditioning soak of duration ``soak_duration_s`` via
     ``integrate_open_circuit`` and explicitly records
     ``initial_state_source = 'finite_time_illuminated_preconditioned'`` with
     ``equilibrium_claimed = False``.
2. Both the pulsed trajectory (``1 + delta_G_frac`` on ``(0, t_pulse]``, ``1``
   on ``(t_pulse, t_pulse + t_decay]``) and the matched unpulsed control
   trajectory (``1`` throughout ``(0, t_pulse + t_decay]``) start from the
   exact same ``(state_oc, V_oc)`` on the exact same time grid using the
   charge-conserving floating-terminal Backward Euler integrator
   (``I_carrier + I_ion + dD/dt = 0``).
3. ``delta_V = V_pulsed - V_control`` isolates the perturbation response from
   any baseline drift.
4. ``fit_tpv_decay`` fits ``tau_s`` ONLY when single-exponential identifiability
   holds; non-monotonic, sign-changing, truncated, or multi-exponential decays
   return ``tau_s = None`` and an explicit ``non_identifiable_reason``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CompiledDevice1D, State1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.electrothermal import has_declared_ions
from solarlab.experiments.open_circuit import (
    OpenCircuitOptions,
    OpenCircuitTrajectory,
    integrate_open_circuit,
)
from solarlab.solvers.nonlinear.ionic_1d import (
    IonicSteadyOptions,
    solve_ionic_steady_state,
)
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state

__all__ = [
    "TPVDecayFit",
    "TPVError",
    "TPVProtocol",
    "TPVResult",
    "fit_tpv_decay",
    "run_tpv",
    "solve_illuminated_voc_state",
]


class TPVError(RuntimeError):
    """Invalid TPV protocol or failed open-circuit preconditioning/integration."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TPVDecayFit:
    """Single-exponential identifiability outcome for post-pulse ``delta_V(t)``."""

    tau_s: float | None
    amplitude_V: float
    status: str
    non_identifiable_reason: str | None
    points: int
    r_squared: float | None
    normalized_max_error: float | None


@dataclass(frozen=True)
class TPVProtocol:
    """Explicit TPV perturbation, sampling, and preconditioning controls."""

    delta_G_frac: float = 0.05
    t_pulse_s: float = 1e-6
    t_decay_s: float = 5e-5
    pulse_points: int = 10
    decay_points: int = 80
    decay_grid: str = "geometric"
    soak_duration_s: float = 1e-3
    soak_steps: int = 20
    voc_upper_bound_V: float = 1.6
    intervals_per_layer: int = 24
    open_circuit_options: OpenCircuitOptions = OpenCircuitOptions()

    def __post_init__(self) -> None:
        for name in (
            "delta_G_frac",
            "t_pulse_s",
            "t_decay_s",
            "voc_upper_bound_V",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(raw))
        if (
            not _is_real_number(self.soak_duration_s)
            or not math.isfinite(float(self.soak_duration_s))
            or float(self.soak_duration_s) < 0.0
        ):
            raise ValueError("soak_duration_s must be finite and >= 0")
        object.__setattr__(self, "soak_duration_s", float(self.soak_duration_s))

        if type(self.pulse_points) is not int or self.pulse_points < 2:
            raise ValueError("pulse_points must be an integer >= 2")
        if type(self.decay_points) is not int or self.decay_points < 5:
            raise ValueError("decay_points must be an integer >= 5")
        if type(self.soak_steps) is not int or self.soak_steps < 2:
            raise ValueError("soak_steps must be an integer >= 2")
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if self.decay_grid not in {"linear", "geometric"}:
            raise ValueError("decay_grid must be 'linear' or 'geometric'")
        if not isinstance(self.open_circuit_options, OpenCircuitOptions):
            raise TypeError("open_circuit_options must be an OpenCircuitOptions")


@dataclass(frozen=True)
class TPVResult:
    """Paired pulsed/control open-circuit TPV trajectory and decay fit."""

    t: np.ndarray
    V: np.ndarray
    V_control: np.ndarray
    delta_V: np.ndarray
    V_oc: float
    delta_V_0: float
    tau: float | None
    decay_fit: TPVDecayFit
    pulsed_trajectory: OpenCircuitTrajectory
    control_trajectory: OpenCircuitTrajectory
    preconditioning_trajectory: OpenCircuitTrajectory | None
    initial_state_source: str
    equilibrium_claimed: bool
    max_baseline_drift_V: float
    max_current_error_A_m2: float
    charge_voltage_error_V: float
    compiled_identity: str
    result_sha256: str


def fit_tpv_decay(
    t_decay_rel_s: object,
    delta_V_decay: object,
    *,
    min_amplitude_V: float = 1e-9,
    tail_floor_fraction: float = 1e-3,
    min_points: int = 5,
    min_efolds: float = 1.0,
    monotonicity_atol_fraction: float = 1e-4,
    min_r_squared: float = 0.995,
    max_normalized_error: float = 0.02,
) -> TPVDecayFit:
    """Fit ``delta_V(t) = A * exp(-t / tau)`` or refuse with an explicit reason."""
    t_raw = np.asarray(t_decay_rel_s, dtype=float)
    y_raw = np.asarray(delta_V_decay, dtype=float)
    if (
        t_raw.ndim != 1
        or y_raw.shape != t_raw.shape
        or t_raw.size < 2
        or not np.all(np.isfinite(t_raw))
        or not np.all(np.isfinite(y_raw))
        or np.any(np.diff(t_raw) < 0.0)
    ):
        raise ValueError("t_decay_rel_s and delta_V_decay must be matched 1-D finite arrays")

    t_rel = t_raw - float(t_raw[0])
    amp0 = float(y_raw[0])
    if not math.isfinite(amp0) or amp0 <= min_amplitude_V:
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="no_signal",
            non_identifiable_reason=(
                f"initial post-pulse perturbation {amp0:.3e} V <= min_amplitude_V ({min_amplitude_V:.3e} V)"
            ),
            points=0,
            r_squared=None,
            normalized_max_error=None,
        )

    floor_v = tail_floor_fraction * amp0
    if np.any(y_raw < -floor_v):
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="sign_change",
            non_identifiable_reason=(
                "differential photovoltage changed sign below noise floor during decay"
            ),
            points=int(y_raw.size),
            r_squared=None,
            normalized_max_error=None,
        )

    below = np.flatnonzero(y_raw <= floor_v)
    end_idx = int(below[0]) if below.size > 0 else int(y_raw.size)
    t_win = t_rel[:end_idx]
    y_win = y_raw[:end_idx]

    if y_win.size < min_points:
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="insufficient_points",
            non_identifiable_reason=(
                f"only {y_win.size} decay sample(s) above tail floor (< min_points={min_points})"
            ),
            points=int(y_win.size),
            r_squared=None,
            normalized_max_error=None,
        )

    diffs = np.diff(y_win)
    if np.any(diffs > monotonicity_atol_fraction * amp0):
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="non_monotonic",
            non_identifiable_reason=(
                "post-pulse differential photovoltage is non-monotonic (re-rising tail)"
            ),
            points=int(y_win.size),
            r_squared=None,
            normalized_max_error=None,
        )

    efolds_observed = math.log(amp0 / float(y_win[-1]))
    if efolds_observed < min_efolds - 1e-6:
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="insufficient_decay_window",
            non_identifiable_reason=(
                f"decay window spans {efolds_observed:.3f} e-folds (< required {min_efolds:.3f})"
            ),
            points=int(y_win.size),
            r_squared=None,
            normalized_max_error=None,
        )

    # Initial guess from log-linear regression
    log_y = np.log(y_win)
    slope, intercept = np.polyfit(t_win, log_y, 1)
    tau_guess = max(1e-15, -1.0 / float(slope)) if slope < 0.0 else float(t_win[-1])
    amp_guess = max(min_amplitude_V, math.exp(float(intercept)))

    def residuals(params: np.ndarray) -> np.ndarray:
        a_p, tau_p = float(params[0]), float(params[1])
        return a_p * np.exp(-t_win / tau_p) - y_win

    lsq = least_squares(
        residuals,
        x0=np.array([amp_guess, tau_guess], dtype=float),
        bounds=(np.array([0.0, 1e-18]), np.array([10.0 * amp0 + 1.0, 1e6 * max(1.0, float(t_win[-1]))])),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    amp_fit = float(lsq.x[0])
    tau_fit = float(lsq.x[1])
    y_pred = amp_fit * np.exp(-t_win / tau_fit)
    ss_res = float(np.sum((y_win - y_pred) ** 2))
    ss_tot = float(np.sum((y_win - np.mean(y_win)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    norm_err = float(np.max(np.abs(y_win - y_pred)) / amp0)

    if not lsq.success or r2 < min_r_squared or norm_err > max_normalized_error:
        return TPVDecayFit(
            tau_s=None,
            amplitude_V=amp0,
            status="not_single_exponential",
            non_identifiable_reason=(
                f"decay deviates from single exponential (R^2={r2:.5f} vs >= {min_r_squared}, "
                f"normalized_max_error={norm_err:.4f} vs <= {max_normalized_error})"
            ),
            points=int(y_win.size),
            r_squared=r2,
            normalized_max_error=norm_err,
        )

    return TPVDecayFit(
        tau_s=tau_fit,
        amplitude_V=amp_fit,
        status="single_exponential",
        non_identifiable_reason=None,
        points=int(y_win.size),
        r_squared=r2,
        normalized_max_error=norm_err,
    )


def solve_illuminated_voc_state(
    compiled: CompiledDevice1D,
    *,
    generation_scale: float = 1.0,
    upper_bound_V: float = 1.6,
    current_tol_A_m2: float = 1e-9,
    steady_options: SteadyOptions | None = None,
) -> tuple[State1D, float]:
    """Find the exact ion-free steady ``(state_oc, V_oc)`` from fixed lower bracket ``0.0 V``."""
    if not isinstance(compiled, CompiledDevice1D):
        raise TypeError("compiled must be a CompiledDevice1D")
    opts = SteadyOptions() if steady_options is None else steady_options
    polarity = float(compiled.voltage_polarity)

    sol_0 = solve_steady_state(
        compiled,
        initial_state=None,
        voltage_V=0.0,
        generation_scale=generation_scale,
        options=opts,
    )
    if not sol_0.converged:
        raise TPVError("0 V illuminated steady state failed to converge")
    j_0 = float(-polarity * sol_0.current_density_A_m2[0])
    if j_0 <= 0.0:
        raise TPVError(
            f"Device produces nonpositive short-circuit photocurrent ({j_0:.4g} A/m^2); V_oc is not in (0, {upper_bound_V}] V"
        )

    # March deterministically from 0 V to bracket V_oc reliably without basin jumps
    coarse_v = np.linspace(0.0, upper_bound_V, 17)
    prev_state = sol_0.state
    v_lo = 0.0
    state_lo = sol_0.state
    j_lo = j_0
    v_hi: float | None = None
    for v_cand in coarse_v[1:]:
        sol_c = solve_steady_state(
            compiled,
            initial_state=prev_state,
            voltage_V=float(v_cand),
            generation_scale=generation_scale,
            options=opts,
        )
        if not sol_c.converged:
            raise TPVError(f"Steady state failed while bracketing V_oc at V={v_cand:.4f} V")
        j_c = float(-polarity * sol_c.current_density_A_m2[0])
        if j_c <= 0.0:
            v_hi = float(v_cand)
            break
        v_lo = float(v_cand)
        state_lo = sol_c.state
        j_lo = j_c
        prev_state = sol_c.state

    if v_hi is None:
        raise TPVError(
            f"Open-circuit voltage exceeds upper_bound_V={upper_bound_V:.4f} V"
        )
    _ = j_lo

    best: dict[str, Any] = {"state": state_lo, "v": v_lo}

    def current_at(v_trial: float) -> float:
        sol_t = solve_steady_state(
            compiled,
            initial_state=state_lo,
            voltage_V=float(v_trial),
            generation_scale=generation_scale,
            options=opts,
        )
        if not sol_t.converged:
            raise TPVError(f"Steady solve failed during V_oc Brent search at {v_trial:.6f} V")
        j_val = float(-polarity * sol_t.current_density_A_m2[0])
        best["state"] = sol_t.state
        best["v"] = float(v_trial)
        return j_val

    v_root = float(brentq(current_at, v_lo, v_hi, xtol=1e-13, rtol=1e-13, maxiter=80))
    sol_final = solve_steady_state(
        compiled,
        initial_state=state_lo,
        voltage_V=v_root,
        generation_scale=generation_scale,
        options=opts,
    )
    if not sol_final.converged:
        raise TPVError("Final V_oc steady state failed certification")
    j_final = abs(float(-polarity * sol_final.current_density_A_m2[0]))
    if j_final > max(current_tol_A_m2, opts.continuity_tolerance_A_m2 * 10.0):
        raise TPVError(
            f"Residual conduction current at V_oc ({j_final:.3e} A/m^2) exceeds tolerance"
        )
    return sol_final.state, v_root


def _precondition_ionic_open_circuit(
    compiled: CompiledIonicDevice1D,
    protocol: TPVProtocol,
) -> tuple[IonicState1D, float, OpenCircuitTrajectory]:
    """Finite-time illuminated open-circuit preconditioning from dark 0 V."""
    if protocol.soak_duration_s <= 0.0:
        raise TPVError(
            "Mobile-ion TPV requires positive soak_duration_s for explicit finite-time preconditioning"
        )
    steady_opts = IonicSteadyOptions(
        electronic=SteadyOptions(
            poisson_tolerance=protocol.open_circuit_options.poisson_tolerance,
            continuity_tolerance_A_m2=protocol.open_circuit_options.continuity_tolerance_A_m2,
            contact_log_tolerance=protocol.open_circuit_options.contact_log_tolerance,
            convergence_policy=HISTORICAL_MAX_ROW_V1,
        ),
        inventory_relative_tolerance=1e-11,
        ion_balance_tolerance_m2_s=protocol.open_circuit_options.ion_balance_tolerance_m2_s,
        ion_flux_tolerance_m2_s=protocol.open_circuit_options.ion_balance_tolerance_m2_s,
        ion_logit_step=protocol.open_circuit_options.ion_logit_step,
        inventory_row_node=compiled.ion_node_count - 1,
        budget=SolveBudget(120, 960, 122, 12),
        wall_s=30.0,
    )
    dark0 = solve_ionic_steady_state(
        compiled,
        voltage_V=0.0,
        generation_scale=0.0,
        options=steady_opts,
        initial_state=None,
    )
    if not dark0.accepted or dark0.state is None:
        raise TPVError(
            f"Dark 0 V initial state failed before TPV preconditioning: {dark0.termination}"
        )

    # Smooth illumination turn-on followed by geometric open-circuit soak to
    # soak_duration_s (keeping dt >= 2e-5 s so finite-site ionic storage rates
    # stay free of floating-point subtraction noise while resolving carrier and
    # ionic redistribution).
    if protocol.soak_duration_s <= 1e-4:
        soak_t = np.linspace(0.0, protocol.soak_duration_s, max(5, protocol.soak_steps + 1))
        soak_g = np.ones_like(soak_t)
        soak_g[0] = 0.0
        if soak_g.size >= 4:
            soak_g[1] = 1e-3
            soak_g[2] = 1e-1
    else:
        n_tail = max(3, protocol.soak_steps - 2)
        t_ramp1 = min(2e-5, 0.1 * protocol.soak_duration_s)
        t_ramp2 = min(1e-4, 0.3 * protocol.soak_duration_s)
        soak_t = np.r_[
            0.0,
            t_ramp1,
            t_ramp2,
            np.geomspace(t_ramp2 * 2.0, protocol.soak_duration_s, n_tail),
        ]
        soak_g = np.ones_like(soak_t)
        soak_g[0] = 0.0
        soak_g[1] = 1e-3
        soak_g[2] = 1e-1
    pre_traj = integrate_open_circuit(
        compiled,
        dark0.state,
        soak_t,
        soak_g,
        initial_voltage_V=0.0,
        options=protocol.open_circuit_options,
    )
    final_state = pre_traj.states[-1]
    assert isinstance(final_state, IonicState1D)
    return final_state, float(pre_traj.terminal_voltage_V[-1]), pre_traj


def _build_tpv_schedule(
    protocol: TPVProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build matched ``(times_s, gen_pulsed, gen_control, pulse_end_index)``."""
    t_pulse_grid = np.linspace(0.0, protocol.t_pulse_s, protocol.pulse_points + 1)
    if protocol.decay_grid == "geometric":
        dt0 = max(1e-11, 0.05 * (protocol.t_pulse_s / protocol.pulse_points))
        decay_offsets = np.geomspace(
            dt0, protocol.t_decay_s, protocol.decay_points
        )
    else:
        decay_offsets = np.linspace(
            protocol.t_decay_s / protocol.decay_points,
            protocol.t_decay_s,
            protocol.decay_points,
        )
    t_decay_grid = protocol.t_pulse_s + decay_offsets
    times_s = np.r_[t_pulse_grid, t_decay_grid]
    pulse_end_idx = int(protocol.pulse_points)

    gen_control = np.ones_like(times_s)
    gen_pulsed = np.ones_like(times_s)
    gen_pulsed[1 : pulse_end_idx + 1] = 1.0 + protocol.delta_G_frac
    return times_s, gen_pulsed, gen_control, pulse_end_idx


def run_tpv(
    target: PreparedDevice | CompiledDevice1D | CompiledIonicDevice1D,
    protocol: TPVProtocol | None = None,
    *,
    ion_domain_layer: str | None = None,
    ion_domain_endpoints: tuple[str, str] | None = None,
) -> TPVResult:
    """Execute paired pulsed + control open-circuit TPV and extract ``TPVDecayFit``."""
    prot = TPVProtocol() if protocol is None else protocol
    if not isinstance(prot, TPVProtocol):
        raise TypeError("protocol must be a TPVProtocol")

    if isinstance(target, PreparedDevice):
        has_ions, inferred_layer = has_declared_ions(target)
        if not has_ions:
            compiled: CompiledDevice1D | CompiledIonicDevice1D = compile_device_1d(
                target, intervals_per_layer=prot.intervals_per_layer
            )
        else:
            elec_layers = tuple(
                lyr for lyr in target.layers if lyr.role != "substrate"
            )
            chosen_layer = (
                ion_domain_layer if ion_domain_layer is not None else inferred_layer
            )
            chosen_end = (
                ion_domain_endpoints
                if ion_domain_endpoints is not None
                else (("blocking", "blocking") if len(elec_layers) > 1 else None)
            )
            compiled = compile_ionic_device_1d(
                target,
                intervals_per_layer=prot.intervals_per_layer,
                ion_domain_layer=chosen_layer,
                ion_domain_endpoints=chosen_end,
            )
    elif isinstance(target, (CompiledDevice1D, CompiledIonicDevice1D)):
        compiled = target
    else:
        raise TypeError(
            "target must be PreparedDevice, CompiledDevice1D, or CompiledIonicDevice1D"
        )

    pre_traj: OpenCircuitTrajectory | None
    if isinstance(compiled, CompiledDevice1D):
        state_oc: State1D | IonicState1D
        state_oc, v_oc = solve_illuminated_voc_state(
            compiled,
            generation_scale=1.0,
            upper_bound_V=prot.voc_upper_bound_V,
            current_tol_A_m2=prot.open_circuit_options.max_current_error_A_m2,
        )
        init_source = "steady_illuminated_open_circuit"
        eq_claimed = True
        pre_traj = None
    else:
        state_oc, v_oc, pre_traj = _precondition_ionic_open_circuit(compiled, prot)
        init_source = "finite_time_illuminated_preconditioned"
        eq_claimed = False

    times_s, gen_pulsed, gen_control, pulse_end_idx = _build_tpv_schedule(prot)

    pulsed_traj = integrate_open_circuit(
        compiled,
        state_oc,
        times_s,
        gen_pulsed,
        initial_voltage_V=v_oc,
        options=prot.open_circuit_options,
    )
    control_traj = integrate_open_circuit(
        compiled,
        state_oc,
        times_s,
        gen_control,
        initial_voltage_V=v_oc,
        options=prot.open_circuit_options,
    )

    v_pulsed = pulsed_traj.terminal_voltage_V
    v_ctrl = control_traj.terminal_voltage_V
    delta_v = np.array(v_pulsed - v_ctrl, dtype=float, copy=True)
    delta_v.setflags(write=False)

    delta_v_0 = float(delta_v[pulse_end_idx])
    t_decay_slice = times_s[pulse_end_idx:] - float(times_s[pulse_end_idx])
    dv_decay_slice = delta_v[pulse_end_idx:]
    fit = fit_tpv_decay(t_decay_slice, dv_decay_slice)

    max_drift = float(np.max(np.abs(v_ctrl - v_oc)))
    max_i_err = float(
        max(
            np.max(pulsed_traj.max_face_total_current_A_m2),
            np.max(control_traj.max_face_total_current_A_m2),
        )
    )
    v_charge_err = float(
        max(
            pulsed_traj.charge_voltage_error_V,
            control_traj.charge_voltage_error_V,
        )
    )

    sha = _canonical_sha256(
        {
            "compiled_identity": compiled.identity,
            "initial_state_source": init_source,
            "equilibrium_claimed": eq_claimed,
            "V_oc": float(v_oc).hex(),
            "delta_V_0": delta_v_0.hex(),
            "decay_fit": asdict(fit),
            "max_current_error_A_m2": max_i_err.hex(),
            "charge_voltage_error_V": v_charge_err.hex(),
        }
    )
    return TPVResult(
        t=times_s,
        V=v_pulsed,
        V_control=v_ctrl,
        delta_V=delta_v,
        V_oc=float(v_oc),
        delta_V_0=delta_v_0,
        tau=fit.tau_s,
        decay_fit=fit,
        pulsed_trajectory=pulsed_traj,
        control_trajectory=control_traj,
        preconditioning_trajectory=pre_traj,
        initial_state_source=init_source,
        equilibrium_claimed=eq_claimed,
        max_baseline_drift_V=max_drift,
        max_current_error_A_m2=max_i_err,
        charge_voltage_error_V=v_charge_err,
        compiled_identity=compiled.identity,
        result_sha256=sha,
    )

"""Transient photovoltage (TPV) experiment.

Physics: the device is equilibrated at open circuit under steady-state
illumination (generation rate G_ss). A small perturbation pulse
(delta_G * G_ss) is applied for t_pulse seconds. The terminal is kept
at open circuit throughout — the voltage rises during the pulse and
decays back to V_oc as the excess carriers recombine. The decay
timescale tau encodes the dominant recombination lifetime.

Implementation: because the drift-diffusion solver operates at fixed
V_app (Dirichlet Poisson BCs), we approximate open circuit by finding
V_oc from a quick J-V scan and holding V_app = V_oc. The terminal
current J(t) is near zero (true OC), and we track V_oc + delta_V(t) by
reading the quasi-Fermi-level splitting from the spatial profiles. In
practice, since J ≈ 0 at V_oc, the voltage perturbation is small enough
that the fixed-V_app approximation is accurate to first order.

Alternative approach used here: we run successive short transient
intervals and adjust V_app at each step to keep J ≈ 0 (iterative OC
tracking). This gives a true V(t) transient.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.tpv import TPVResult
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    run_transient,
    build_material_arrays,
    MaterialArrays,
)
from perovskite_sim.experiments.jv_sweep import (
    _compute_current,
    _integrate_step,
    compute_metrics,
    _grid_node_count,
)

ProgressCallback = Callable[[str, int, int, str], None]


def _find_voc(
    x: np.ndarray,
    y_ss: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_guess: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Find V_oc by bisection: the voltage where J(V) = 0 under illumination.

    Starts from a coarse bracket [0, V_guess*1.5] and narrows to ~1 mV.
    Returns (V_oc, y_at_voc).
    """
    # Coarse scan to bracket V_oc
    n_coarse = 20
    V_lo, V_hi = 0.0, V_guess * 1.5
    V_scan = np.linspace(V_lo, V_hi, n_coarse)
    dt_coarse = 1e-4  # short settle per voltage step
    y = y_ss.copy()
    J_scan = np.zeros(n_coarse)
    for k, V_k in enumerate(V_scan):
        y_prev = y.copy()
        y = _integrate_step(x, y, stack, mat, V_k, k * dt_coarse,
                            (k + 1) * dt_coarse, rtol, atol)
        J_scan[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev,
                                      dt=dt_coarse, mat=mat)

    # Find bracket where J crosses zero (positive to negative)
    signs = np.sign(J_scan)
    crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    if len(crossings) == 0:
        # Fallback: use the V where |J| is smallest
        idx_min = int(np.argmin(np.abs(J_scan)))
        return float(V_scan[idx_min]), y

    idx = int(crossings[0])
    V_a, V_b = float(V_scan[idx]), float(V_scan[idx + 1])

    # Bisection refinement to ~1 mV
    y_a = y_ss.copy()
    for _iter in range(15):
        V_mid = 0.5 * (V_a + V_b)
        y_prev = y_a.copy()
        t_base = _iter * dt_coarse
        y_mid = _integrate_step(x, y_a, stack, mat, V_mid, t_base,
                                t_base + dt_coarse, rtol, atol)
        J_mid = _compute_current(x, y_mid, stack, V_mid, y_prev=y_prev,
                                  dt=dt_coarse, mat=mat)
        if J_mid > 0:
            V_a = V_mid
            y_a = y_mid
        else:
            V_b = V_mid
        if abs(V_b - V_a) < 1e-3:
            break

    V_oc = 0.5 * (V_a + V_b)
    # Settle at V_oc
    y_oc = _integrate_step(x, y_a, stack, mat, V_oc, 0.0, 1e-4, rtol, atol)
    return V_oc, y_oc


def _adjust_V_for_OC(
    x: np.ndarray,
    y_start: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_guess: float,
    t_lo: float,
    t_hi: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    J_tol: float = 0.05,
    max_iter: int = 5,
    dV_perturb: float = 1e-4,
) -> tuple[float, np.ndarray, float]:
    """Find (V, y_end, J_end) such that J_terminal(y_end, V) ≈ 0 after
    integrating from y_start at (t_lo, V) to t_hi.

    Newton-Raphson with finite-difference dJ/dV, warm-started from V_guess.
    This is the circuit-mode core: instead of post-hoc mapping J(t) to V(t)
    via a linearised R_oc slope (which assumes dV ∝ dJ with a single-point
    slope — incorrect for non-unity ideality and large dV), we impose the
    exact open-circuit algebraic constraint J = 0 at every reported time
    and let V_app float to satisfy it.

    Typical cost: 1–3 transient integrations per step (1 when the solution
    was already at OC, 3 when Newton needs one correction pass).
    """
    def _integrate(V: float) -> tuple[float, np.ndarray]:
        sol = run_transient(
            x, y_start, (t_lo, t_hi), np.array([t_hi]),
            stack, illuminated=True, V_app=V,
            rtol=rtol, atol=atol,
            max_step=max((t_hi - t_lo) / 5.0, 1e-12),
            mat=mat,
        )
        y_end = sol.y[:, -1] if sol.success else y_start.copy()
        dt = t_hi - t_lo
        J = _compute_current(x, y_end, stack, V, y_prev=y_start, dt=dt, mat=mat)
        return J, y_end

    V = V_guess
    J, y_end = _integrate(V)
    for _iter in range(max_iter):
        if abs(J) < J_tol:
            break
        J_plus, _ = _integrate(V + dV_perturb)
        dJdV = (J_plus - J) / dV_perturb
        if abs(dJdV) < 1e-12:
            break  # degenerate slope — bail out rather than divide by zero
        dV = -J / dJdV
        # Damp to 20 mV per iteration. TPV voltages never swing more than
        # ~10 mV between output times, so a larger step is always overshoot.
        if abs(dV) > 0.02:
            dV = 0.02 * np.sign(dV)
        V = V + dV
        J, y_end = _integrate(V)
    return V, y_end, J


def _fit_decay_tau(t: np.ndarray, V: np.ndarray, V_oc: float) -> tuple[float, float]:
    """Fit mono-exponential decay to V(t) - V_oc after the pulse.

    Returns (tau, delta_V0) where V(t) ≈ V_oc + delta_V0 * exp(-t/tau).
    """
    dV = V - V_oc
    # Find the peak (end of pulse / start of decay)
    i_peak = int(np.argmax(np.abs(dV)))
    delta_V0 = float(dV[i_peak])
    if abs(delta_V0) < 1e-8:
        return 1e-6, 0.0  # no perturbation detected

    # Fit on the decay portion
    t_decay = t[i_peak:] - t[i_peak]
    dV_decay = dV[i_peak:]

    # Use log-linear fit on |dV| where it's above noise floor
    mask = np.abs(dV_decay) > 0.05 * abs(delta_V0)
    if np.sum(mask) < 3:
        # Not enough points for a fit — estimate from 1/e crossing
        target = abs(delta_V0) * np.exp(-1)
        crossings = np.where(np.abs(dV_decay) <= target)[0]
        if len(crossings) > 0:
            tau = float(t_decay[crossings[0]])
        else:
            tau = float(t_decay[-1])
        return max(tau, 1e-9), delta_V0

    log_dV = np.log(np.abs(dV_decay[mask]))
    t_fit = t_decay[mask]
    # Linear fit: log|dV| = log|delta_V0| - t/tau
    coeffs = np.polyfit(t_fit, log_dV, 1)
    slope = coeffs[0]
    tau = -1.0 / slope if slope < 0 else float(t_decay[-1])
    return max(tau, 1e-9), delta_V0


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
) -> TPVResult:
    """Run a transient photovoltage experiment.

    Parameters
    ----------
    stack : DeviceStack
        Device configuration.
    N_grid : int
        Grid points per layer (passed to multilayer_grid).
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
        Time-resolved voltage and current, fitted decay time.
    """
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if delta_G_frac <= 0 or delta_G_frac >= 1:
        raise ValueError(f"delta_G_frac must be in (0, 1), got {delta_G_frac}")
    if t_pulse <= 0:
        raise ValueError(f"t_pulse must be positive, got {t_pulse}")
    if t_decay <= t_pulse:
        raise ValueError(f"t_decay must exceed t_pulse, got {t_decay}")

    import dataclasses

    # Build grid (electrical layers only)
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)

    # Build material cache — baseline illumination
    mat_ss = build_material_arrays(x, stack)

    # Ensure G_optical is explicit so we can scale it for the pulse.
    # For TMM devices G_optical is already set; for Beer-Lambert it's None
    # and computed inline in assemble_rhs. Materialise it here.
    if mat_ss.G_optical is None:
        from perovskite_sim.physics.generation import beer_lambert_generation
        G_baseline = beer_lambert_generation(x, mat_ss.alpha, stack.Phi)
        mat_ss = dataclasses.replace(mat_ss, G_optical=G_baseline)

    # Build perturbed material cache — boosted generation for the pulse
    mat_pulse = dataclasses.replace(
        mat_ss, G_optical=mat_ss.G_optical * (1.0 + delta_G_frac),
    )

    # Step 1: Equilibrate at V_oc under steady illumination
    y_illum = solve_illuminated_ss(x, stack, V_app=0.0, rtol=rtol, atol=atol)
    V_oc, y_oc = _find_voc(x, y_illum, stack, mat_ss, V_guess=stack.compute_V_bi(),
                            rtol=rtol, atol=atol)

    if progress is not None:
        progress("tpv", 0, n_points, f"V_oc={V_oc:.4f} V")

    # Step 2: Adaptive time grid — denser during pulse, sparser during decay
    t_pulse_pts = max(int(n_points * t_pulse / t_decay), 10)
    t_decay_pts = n_points - t_pulse_pts
    t_arr = np.concatenate([
        np.linspace(0, t_pulse, t_pulse_pts, endpoint=False),
        np.linspace(t_pulse, t_decay, t_decay_pts + 1),
    ])

    # Step 3: Circuit-mode transient. At every reported time we enforce
    # the open-circuit constraint J_terminal(y, V) = 0 exactly via a
    # Newton root-find on V_app — replacing the prior linearised post-hoc
    # mapping V(t) = V_oc + delta_V_ideal · J(t)/J_plateau which assumed
    # dV ∝ dJ with a single-point slope and broke for non-unity ideality.
    V_arr = np.zeros(len(t_arr))
    J_arr = np.zeros(len(t_arr))
    V_arr[0] = V_oc
    J_arr[0] = 0.0  # steady-state OC
    y = y_oc.copy()
    V_prev = V_oc

    for k in range(1, len(t_arr)):
        t_lo = t_arr[k - 1]
        t_hi = t_arr[k]

        if t_lo < t_pulse < t_hi:
            # Straddle the pulse boundary: advance the first sub-interval
            # (pulse phase) at V = V_prev under mat_pulse, then apply the
            # Newton correction to the second sub-interval (decay phase)
            # under mat_ss. The straddle is at most one step per sweep
            # (fallback to V_prev here is a sub-mV error smeared across
            # one output sample).
            if t_pulse - t_lo > 0:
                sol1 = run_transient(
                    x, y, (t_lo, t_pulse), np.array([t_pulse]),
                    stack, illuminated=True, V_app=V_prev,
                    rtol=rtol, atol=atol,
                    max_step=max((t_pulse - t_lo) / 5.0, 1e-12),
                    mat=mat_pulse,
                )
                y = sol1.y[:, -1] if sol1.success else y
            if t_hi - t_pulse > 0:
                V_k, y, J_k = _adjust_V_for_OC(
                    x, y, stack, mat_ss, V_prev, t_pulse, t_hi, rtol, atol,
                )
            else:
                V_k = V_prev
                J_k = _compute_current(x, y, stack, V_prev, mat=mat_pulse)
        elif t_hi <= t_pulse:
            # Entirely inside the pulse phase — V climbs toward V_oc(pulse)
            V_k, y, J_k = _adjust_V_for_OC(
                x, y, stack, mat_pulse, V_prev, t_lo, t_hi, rtol, atol,
            )
        else:
            # Entirely after the pulse — V relaxes back to V_oc
            V_k, y, J_k = _adjust_V_for_OC(
                x, y, stack, mat_ss, V_prev, t_lo, t_hi, rtol, atol,
            )

        V_arr[k] = V_k
        J_arr[k] = J_k
        V_prev = V_k

        if progress is not None:
            progress("tpv", k, len(t_arr) - 1,
                     f"t={t_hi:.3e} s, V={V_k:.4f} V")

    # Fit decay
    tau, delta_V0 = _fit_decay_tau(t_arr, V_arr, V_oc)

    return TPVResult(
        t=t_arr,
        V=V_arr,
        J=J_arr,
        V_oc=V_oc,
        tau=tau,
        delta_V0=delta_V0,
    )

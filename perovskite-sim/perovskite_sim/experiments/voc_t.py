"""V_oc(T) — open-circuit voltage vs temperature sweep.

Physics: The open-circuit voltage of a solar cell follows roughly

    V_oc(T) ≈ (E_A / q) − (kT / q) · ln(J_00 / J_sc)

where ``E_A`` is the activation energy of the dominant recombination
pathway (≈ Eg for purely radiative / SRH-in-the-bulk recombination;
< Eg when interface recombination dominates) and ``J_00`` is a
temperature-independent prefactor. Plotting V_oc against T and
extrapolating the straight portion back to T = 0 K gives the
activation energy directly as the intercept.

The experiment loops over a user-chosen temperature grid, builds a new
frozen DeviceStack at each T via ``dataclasses.replace(stack, T=T_k)``
so the immutable-data contract is preserved, runs a short illuminated
J-V sweep, and reads V_oc from ``compute_metrics``. The forward sweep
is used because it is carrier-dominated and ionic hysteresis does not
change the open-circuit crossing.

Implementation notes:
- Each J-V call is a fresh problem: there is no steady-state hand-off
  between successive temperatures because the ionic equilibrium
  depends on T.
- The temperature-scaling physics hook (MaterialParams.V_T recompute,
  B_rad(T), Varshni Eg(T)) is delivered by ``build_material_arrays``
  via ``SimulationMode.use_temperature_scaling``; this experiment
  just exposes T at the stack level and lets the solver pick it up.
  In LEGACY mode T is effectively frozen and the V_oc(T) curve shows
  only the weak (kT/q)·ln(...) dependence — this is expected.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Callable

import numpy as np

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.voc_t import VocTResult
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, compute_metrics

ProgressCallback = Callable[[str, int, int, str], None]


def _linear_fit(T: np.ndarray, V: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of V = slope · T + intercept.

    Returns (slope, intercept, R_squared). R_squared = 1 − SS_res/SS_tot;
    clamped to 0 when the denominator vanishes (all V_oc identical).
    """
    T = np.asarray(T, dtype=float)
    V = np.asarray(V, dtype=float)
    if len(T) < 2:
        return 0.0, float(V[0]) if len(V) else 0.0, 0.0

    coeffs = np.polyfit(T, V, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    V_pred = slope * T + intercept
    ss_res = float(np.sum((V - V_pred) ** 2))
    ss_tot = float(np.sum((V - np.mean(V)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return slope, intercept, max(0.0, r2)


def run_voc_t(
    stack: DeviceStack,
    T_min: float = 250.0,
    T_max: float = 350.0,
    n_points: int = 6,
    N_grid: int = 60,
    jv_n_points: int = 30,
    v_rate: float = 1.0,
    V_max: float | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> VocTResult:
    """Sweep temperature and record V_oc(T), then linear-fit to extract E_A.

    Parameters
    ----------
    stack : DeviceStack
        Device configuration. ``stack.T`` is the reference temperature
        and is overridden per sweep point.
    T_min, T_max : float
        Temperature sweep endpoints [K].
    n_points : int
        Number of temperature points (inclusive).
    N_grid : int
        Grid points per electrical layer for each J-V run.
    jv_n_points : int
        Voltage samples inside each J-V sweep.
    v_rate : float
        J-V scan rate [V/s] — quasi-static by default.
    V_max : float | None
        Upper voltage. If None, ``run_jv_sweep`` picks a sensible
        default from the band-offset-aware built-in potential.
    rtol, atol : float
        ODE solver tolerances.
    progress : callable, optional
        Progress callback: ``fn(stage, current, total, message)``.

    Returns
    -------
    VocTResult
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if T_min <= 0:
        raise ValueError(f"T_min must be positive, got {T_min}")
    if T_max <= T_min:
        raise ValueError(f"T_max must exceed T_min, got T_min={T_min}, T_max={T_max}")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")

    T_arr = np.linspace(T_min, T_max, n_points)
    # NaN, not 0.0, at any temperature whose sweep does not bracket V_oc --
    # see the voc_bracketed handling below.
    V_oc_arr = np.full(n_points, np.nan)
    J_sc_arr = np.zeros(n_points)
    bracketed = np.zeros(n_points, dtype=bool)

    for k, T_k in enumerate(T_arr):
        stack_k = dataclasses.replace(stack, T=float(T_k))
        if progress is not None:
            progress("voc_t", k, n_points, f"T={T_k:.1f} K")
        jv = run_jv_sweep(
            stack_k,
            N_grid=N_grid,
            n_points=jv_n_points,
            v_rate=v_rate,
            V_max=V_max,
            rtol=rtol,
            atol=atol,
            illuminated=True,
        )
        metrics = compute_metrics(jv.V_fwd, jv.J_fwd)
        # compute_metrics returns V_oc / FF / PCE as ZEROED SENTINELS when the
        # sweep contains no zero-current crossing (voc_bracketed False); only
        # J_sc stays physically meaningful. Reading V_oc unconditionally puts
        # a 0 V point into the Arrhenius fit, which does not merely shift the
        # answer -- it inverts it. Measured on a 6-point 250-350 K sweep with
        # a true dV_oc/dT = -2.20 mV/K: one unbracketed cold-end point takes
        # the fitted slope to +6.44 mV/K, i.e. V_oc apparently RISING with
        # temperature, which is impossible for a non-degenerate semiconductor,
        # and E_A (this experiment's only scientific output) changes sign.
        bracketed[k] = bool(metrics.voc_bracketed)
        V_oc_arr[k] = float(metrics.V_oc) if bracketed[k] else np.nan
        J_sc_arr[k] = metrics.J_sc
        if progress is not None:
            shown = f"{metrics.V_oc:.4f} V" if bracketed[k] else "not bracketed"
            progress("voc_t", k + 1, n_points, f"T={T_k:.1f} K, V_oc={shown}")

    n_ok = int(np.count_nonzero(bracketed))
    if n_ok < 2:
        failed = ", ".join(f"{t:.1f}" for t in T_arr[~bracketed])
        raise ValueError(
            f"V_oc(T) fit needs at least 2 temperatures with a bracketed "
            f"V_oc, got {n_ok} of {n_points}. Unbracketed at T = [{failed}] K. "
            "Raise V_max (or v_max_max_attempts) so the sweep reaches the "
            "open-circuit crossing at every temperature."
        )
    if n_ok < n_points:
        failed = ", ".join(f"{t:.1f}" for t in T_arr[~bracketed])
        warnings.warn(
            f"V_oc(T): {n_points - n_ok} of {n_points} temperatures did not "
            f"bracket V_oc (T = [{failed}] K) and are EXCLUDED from the fit; "
            "their V_oc is reported as NaN. The activation energy is fitted "
            f"on the remaining {n_ok} points -- check V_max before using it.",
            RuntimeWarning,
            stacklevel=2,
        )

    slope, intercept_0K, r2 = _linear_fit(T_arr[bracketed], V_oc_arr[bracketed])

    return VocTResult(
        T_arr=T_arr,
        V_oc_arr=V_oc_arr,
        J_sc_arr=J_sc_arr,
        slope=slope,
        intercept_0K=intercept_0K,
        E_A_eV=intercept_0K,
        R_squared=r2,
        voc_bracketed_arr=bracketed,
    )

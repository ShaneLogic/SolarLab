"""Dark J-V characterisation with ideality factor and saturation current extraction.

Runs a dark forward sweep (G=0 everywhere) and fits the diode region to
J(V) = J_0 · (exp(V/(n·V_T)) - 1), extracting the ideality factor n and
the saturation current J_0.

The fit window is auto-selected as the contiguous voltage range where
log|J(V)| is most linear — operationally, the window with the smallest
mean |d²(log|J|)/dV²|. Below the diode knee log|J| rises super-linearly
(noise + equilibrium-dominated), and above the knee it rolls off as
series-resistance and high-level-injection kick in. The straight-line
window in between gives the physical ideality factor.

Sign convention
---------------
`run_jv_sweep(illuminated=False)` returns J > 0 below the diode knee
(round-off near zero) and J < 0 under forward injection (current flows
into the device, opposite the solar-convention photocurrent direction).
All fits are performed on |J|.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from perovskite_sim.constants import V_T
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class DarkJVResult:
    """Dark J-V curve plus diode-fit ideality and saturation current.

    Attributes
    ----------
    V, J : np.ndarray
        Forward sweep samples (0 → V_max), V in volts, J in A/m². Sign
        convention matches run_jv_sweep: J < 0 under forward bias
        (injection into the device).
    n_ideality : float
        Diode ideality from log|J| vs V slope in the auto-selected
        straight-line window. n = 1 / (V_T · slope).
    J_0 : float
        Saturation current density [A/m²], exp(intercept) of the
        log|J|-vs-V fit. The dark injection current extrapolated to V = 0.
    V_fit_lo, V_fit_hi : float
        Voltage window [V] over which the diode fit was performed.
    """
    V: np.ndarray
    J: np.ndarray
    n_ideality: float
    J_0: float
    V_fit_lo: float
    V_fit_hi: float


def _select_diode_window(
    V: np.ndarray,
    J: np.ndarray,
    min_points: int = 6,
    J_floor: float = 1e-2,
) -> tuple[int, int]:
    """Pick the contiguous window where log|J(V)| is most linear.

    Algorithm
    ---------
    1. Mask to V > 0.01 V, J < 0 (injection direction — current leaves
       via the contact the opposite way from photocurrent), and
       |J| > J_floor (drops the sub-turn-on leakage/noise region where J
       is dominated by the `-1` term in J_0·(exp(V/nV_T)-1) and is NOT
       exponential in V).
    2. Compute d²(log|J|)/dV² by applying ``np.gradient`` twice on the
       valid subset.
    3. Find the ``min_points``-wide window with the smallest mean |d²|.
    4. Grow that window outward while each neighbor's |d²| stays below
       10× the minimum mean — a generous relative threshold that lets
       the selector include the whole exponential regime even when float
       noise dominates the interior.

    Returns ``(i_lo, i_hi)`` as inclusive indices into the *input*
    arrays. If the valid subset is too small for a fit, falls back to
    the upper half of the sweep.
    """
    absJ = np.abs(np.asarray(J, dtype=float))
    # Require injection direction: J < 0 in the run_jv_sweep convention.
    # The pre-turn-on regime has J ≈ 0 with possible sign flips from
    # round-off; those are physically meaningless and must be excluded.
    valid = (V > 0.01) & (J < 0.0) & (absJ > J_floor)
    idx = np.where(valid)[0]
    if len(idx) < min_points:
        # Fallback: upper half of the sweep. Not reliable — expect a
        # poor ideality number — but at least doesn't crash.
        n = len(V)
        return max(n // 2, 0), n - 1

    V_valid = V[idx]
    logJ = np.log(absJ[idx])
    dlog = np.gradient(logJ, V_valid)
    d2log = np.gradient(dlog, V_valid)

    W = min_points
    best_i = 0
    best_score = float(np.inf)
    for i in range(0, len(idx) - W + 1):
        score = float(np.mean(np.abs(d2log[i : i + W])))
        if score < best_score:
            best_score = score
            best_i = i
    # 10× the minimum-mean score is a pragmatic threshold: a true
    # exponential has |d²| ~ float-noise-floor (say 1e-14), and the
    # diode region has |d²| still orders of magnitude below the
    # curvature near the turn-on knee (~10² - 10³). A 10× factor cleanly
    # separates "straight enough" from "kinked".
    threshold = max(10.0 * best_score, 1e-10)
    lo = best_i
    hi = best_i + W - 1
    while lo > 0 and abs(d2log[lo - 1]) <= threshold:
        lo -= 1
    while hi < len(idx) - 1 and abs(d2log[hi + 1]) <= threshold:
        hi += 1
    return int(idx[lo]), int(idx[hi])


def _fit_ideality_and_J0(
    V: np.ndarray, J: np.ndarray
) -> tuple[float, float, float, float]:
    """Linear fit log|J| = log(J_0) + V / (n · V_T) over the diode window.

    Returns ``(n_ideality, J_0, V_fit_lo, V_fit_hi)``.
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    i_lo, i_hi = _select_diode_window(V, J)
    V_fit = V[i_lo : i_hi + 1]
    logJ_fit = np.log(np.abs(J[i_lo : i_hi + 1]))
    # numpy.polyfit returns [slope, intercept] for degree 1.
    slope, intercept = np.polyfit(V_fit, logJ_fit, 1)
    n = 1.0 / (V_T * slope) if slope > 0 else float("inf")
    J_0 = float(np.exp(intercept))
    return float(n), J_0, float(V_fit[0]), float(V_fit[-1])


def run_dark_jv(
    stack: DeviceStack,
    V_max: float = 1.2,
    n_points: int = 60,
    N_grid: int = 60,
    v_rate: float = 1.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> DarkJVResult:
    """Run a dark forward J-V sweep and extract diode ideality + J_0.

    Parameters
    ----------
    stack : DeviceStack
    V_max : float, default 1.2
        Upper voltage of the forward sweep. Chosen to reach the diode
        exponential region (~0.5 V turn-on for MAPbI3) while staying below
        the series-resistance roll-off that would bias the ideality fit.
    n_points : int, default 60
        Voltage samples between 0 and V_max. Higher gives a cleaner
        second-derivative signal for window selection.
    N_grid : int, default 60
        Total drift-diffusion grid nodes across all electrical layers.
    v_rate : float, default 1.0
        Quasi-static sweep rate [V/s]. 1 V/s is slow enough that the ion
        distribution re-equilibrates for perovskite presets; for pure
        electronic stacks (CIGS, c-Si) this is irrelevant.
    rtol, atol : float
        scipy solver tolerances forwarded to ``run_jv_sweep``.
    progress : ProgressCallback | None
        Forwarded to ``run_jv_sweep``.

    Returns
    -------
    DarkJVResult
    """
    jv = run_jv_sweep(
        stack,
        N_grid=N_grid,
        n_points=n_points,
        v_rate=v_rate,
        V_max=V_max,
        rtol=rtol,
        atol=atol,
        illuminated=False,
        progress=progress,
    )
    # Forward sweep only: the reverse scan in the dark just retraces the
    # same injection curve (no ion hysteresis signal when G = 0).
    V = np.asarray(jv.V_fwd, dtype=float)
    J = np.asarray(jv.J_fwd, dtype=float)
    n, J_0, V_lo, V_hi = _fit_ideality_and_J0(V, J)
    return DarkJVResult(
        V=V, J=J, n_ideality=n, J_0=J_0, V_fit_lo=V_lo, V_fit_hi=V_hi,
    )

"""C-V sweep and Mott-Schottky depletion-capacitance analysis.

Physics
-------
In the dark at reverse/zero bias a p-n junction is a parallel-plate
capacitor whose "plate separation" is the depletion width. For a
one-sided junction (N_heavy >> N_light) the capacitance per area is

    C(V) = sqrt(q · ε_s · ε_0 · N_eff / (2 (V_bi − V)))

where N_eff is the lighter-doped side's net ionised density. Plotting
``1/C²`` against V gives a straight line with slope
``−2 / (q·ε_s·ε_0·N_eff)`` and V-axis intercept at V_bi. Fitting that
line is the standard way experimentalists extract both the built-in
voltage and the doping density from a C-V measurement — the so-called
Mott-Schottky plot.

This experiment is a thin wrapper over ``run_impedance``: at each DC
bias a single AC excitation at ``frequency`` is driven and the
capacitance is read off as ``C = Im(Y)/ω`` where ``Y = 1/Z``. The
Mott-Schottky linear fit is then applied post-hoc, with the fit window
auto-selected as the widest contiguous sub-range whose residual from
the linear model is below an adaptive tolerance — rejecting the
low-bias regime where the device is fully depleted or freeze-out
effects curve 1/C² away from linear.

Measurement conditions
----------------------
- Dark (``illuminated=False`` forced) — photogenerated carriers would
  swamp the depletion capacitance at zero/reverse bias.
- Single AC frequency, default 1e5 Hz (100 kHz). High enough that the
  ionic displacement current in perovskite stacks does not alias into
  the measurement; low enough that the full bulk depletion layer has
  time to respond. Pick a higher frequency for perovskites with very
  mobile ions.
- Bias sweep typically from mild reverse (V < 0) up to roughly half of
  V_bi. Beyond that the exponential injection current starts to
  dominate the admittance and Mott-Schottky linearity breaks.

Sign convention
---------------
Capacitance is reported as a positive real number [F/m²]. The lock-in
returns ``Z = δV / δI`` with Im(Y) > 0 for a capacitor; we take the
magnitude so that downstream fits never have to branch on sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.experiments.impedance import run_impedance

ProgressCallback = Callable[[str, int, int, str], None]

EPS_0 = 8.8541878128e-12  # vacuum permittivity [F/m]


@dataclass(frozen=True)
class MottSchottkyResult:
    """C-V sweep with Mott-Schottky linear fit.

    Attributes
    ----------
    V : np.ndarray
        DC bias samples [V], sorted ascending.
    C : np.ndarray
        Junction capacitance per area at each bias [F/m²]. Derived from
        ``C = Im(Y)/ω`` with ``Y = 1/Z`` at the specified frequency.
    one_over_C2 : np.ndarray
        1/C² values [m⁴/F²] — the Mott-Schottky ordinate.
    V_bi_fit : float
        Built-in voltage extracted from the V-axis intercept of the
        linear fit of ``1/C² = a·V + b`` over the fit window:
        ``V_bi_fit = -b/a``. NaN if the fit failed (e.g. fewer than
        3 linear points).
    N_eff_fit : float
        Ionised dopant density on the lighter-doped side of the
        junction [m⁻³], from ``N = -2/(q·ε_s·ε_0·a)``. NaN on fit
        failure.
    V_fit_lo, V_fit_hi : float
        Bias range over which the linear fit was performed.
    frequency : float
        AC excitation frequency [Hz] used to probe the capacitance.
    eps_r_used : float
        Effective dielectric constant used to convert the slope to
        N_eff. Taken from the absorber-role layer; if no such layer
        exists, the stack's lightest-doped electrical layer is used.
    """
    V: np.ndarray
    C: np.ndarray
    one_over_C2: np.ndarray
    V_bi_fit: float
    N_eff_fit: float
    V_fit_lo: float
    V_fit_hi: float
    frequency: float
    eps_r_used: float


def _resolve_eps_r(stack: DeviceStack) -> float:
    """Pick the ε_r that controls the depletion capacitance.

    For a one-sided junction the depletion extends almost entirely into
    the lightly-doped side, so its ε_r is the physically relevant one.
    Preference order:

    1. A layer whose role is ``"absorber"`` — perovskite / CIGS / c-Si
       base cells all mark the intrinsic / lightly-doped side this way.
    2. Otherwise the electrical layer with the smallest
       ``max(N_A, N_D, ni)`` — i.e. the lightest-doped side.

    Falling back to a hard-coded default would silently miscompute
    N_eff for exotic stacks, so we instead raise if no layer carries
    MaterialParams.
    """
    for layer in stack.layers:
        if layer.role == "absorber" and layer.params is not None:
            return float(layer.params.eps_r)
    # Fallback: lightest-doped electrical layer.
    best = None
    best_doping = float("inf")
    for layer in stack.layers:
        if layer.role == "substrate" or layer.params is None:
            continue
        doping = max(layer.params.N_A, layer.params.N_D, layer.params.ni)
        if doping < best_doping:
            best_doping = doping
            best = layer
    if best is None:
        raise ValueError(
            "Cannot resolve ε_r for Mott-Schottky: no electrical layer with "
            "MaterialParams found in the stack."
        )
    return float(best.params.eps_r)


def _select_ms_window(
    V: np.ndarray,
    y: np.ndarray,
    min_points: int = 4,
    max_resid_ratio: float = 0.1,
) -> tuple[int, int]:
    """Pick the widest contiguous window that is linear in (V, y=1/C²).

    Strategy: try every contiguous window of size >= ``min_points`` (in
    increasing length order), fit a line, accept if the RMS residual is
    below ``max_resid_ratio`` times the window's span in y. Return the
    longest window that qualifies. Falls back to the full input if no
    window passes — callers get the best available linear regression
    and should inspect the fit residual themselves.

    The adaptive tolerance handles both the low-bias tail (1/C² curves
    up when the junction is fully depleted) and the high-bias tail
    (1/C² curves down as injection kicks in). The straight middle band
    is the physical Mott-Schottky regime.
    """
    n = len(V)
    if n < min_points:
        return 0, n - 1

    best_lo, best_hi = 0, n - 1
    best_len = 0
    for lo in range(0, n - min_points + 1):
        for hi in range(lo + min_points - 1, n):
            V_sub = V[lo : hi + 1]
            y_sub = y[lo : hi + 1]
            # Linear fit
            slope, intercept = np.polyfit(V_sub, y_sub, 1)
            resid = y_sub - (slope * V_sub + intercept)
            y_span = float(y_sub.max() - y_sub.min())
            if y_span <= 0.0:
                continue
            rms = float(np.sqrt(np.mean(resid * resid)))
            if rms <= max_resid_ratio * y_span and (hi - lo + 1) > best_len:
                best_len = hi - lo + 1
                best_lo, best_hi = lo, hi
    return best_lo, best_hi


def _fit_mott_schottky(
    V: np.ndarray, C: np.ndarray, eps_r: float
) -> tuple[float, float, float, float]:
    """Linear fit of 1/C² vs V → (V_bi_fit, N_eff_fit, V_lo, V_hi).

    The V-intercept of the line gives V_bi; the slope gives N_eff via
    the Mott-Schottky relation. NaNs are returned if the fit is
    degenerate (flat line, or fewer than 3 points after window
    selection).
    """
    y = 1.0 / (C * C)
    lo, hi = _select_ms_window(V, y)
    V_fit = V[lo : hi + 1]
    y_fit = y[lo : hi + 1]
    if V_fit.size < 3:
        return float("nan"), float("nan"), float(V[lo]), float(V[hi])
    slope, intercept = np.polyfit(V_fit, y_fit, 1)
    # For 1/C² = a·V + b: in Mott-Schottky a < 0, V_bi = -b/a.
    if abs(slope) < 1e-30:
        return float("nan"), float("nan"), float(V_fit[0]), float(V_fit[-1])
    V_bi_fit = -intercept / slope
    # N_eff = -2 / (q · ε_s · ε_0 · slope). Slope is negative, so N_eff > 0.
    N_eff_fit = -2.0 / (Q * eps_r * EPS_0 * slope)
    return (
        float(V_bi_fit),
        float(N_eff_fit),
        float(V_fit[0]),
        float(V_fit[-1]),
    )


def run_mott_schottky(
    stack: DeviceStack,
    V_range: Sequence[float] | np.ndarray = None,
    frequency: float = 1.0e5,
    delta_V: float = 0.01,
    N_grid: int = 40,
    n_cycles: int = 5,
    n_extract: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> MottSchottkyResult:
    """Dark C-V sweep + Mott-Schottky fit.

    Parameters
    ----------
    stack : DeviceStack
    V_range : sequence of float, optional
        DC bias sample points [V]. Default: ``np.linspace(-0.3, 0.4, 8)``
        — mild reverse to moderate forward, which on a typical silicon
        or perovskite junction covers the depletion regime where
        Mott-Schottky is linear, without running into injection.
    frequency : float, default 1e5 Hz
        Single AC excitation frequency. See module docstring.
    delta_V : float, default 0.01 V
        AC amplitude. Small-signal regime: keep below k_B·T/q (~26 mV)
        so the lock-in's linear assumption holds.
    N_grid, n_cycles, n_extract, rtol, atol :
        Forwarded to ``run_impedance`` at every bias.
    progress : ProgressCallback | None
        Called as ``progress("mott_schottky", k, total, msg)`` after
        each bias completes.

    Returns
    -------
    MottSchottkyResult
    """
    if V_range is None:
        V_range = np.linspace(-0.3, 0.4, 8)
    V_arr = np.array(sorted(set(float(v) for v in V_range)), dtype=float)
    if V_arr.size < 3:
        raise ValueError(
            f"Need at least 3 distinct V points for a Mott-Schottky fit, "
            f"got {V_arr.size}."
        )
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    eps_r = _resolve_eps_r(stack)
    omega = 2.0 * np.pi * frequency
    freqs = np.array([frequency], dtype=float)

    C_arr = np.zeros_like(V_arr)
    for k, V_dc in enumerate(V_arr):
        r = run_impedance(
            stack, freqs,
            V_dc=float(V_dc), delta_V=delta_V,
            N_grid=N_grid, n_cycles=n_cycles, n_extract=n_extract,
            rtol=rtol, atol=atol, illuminated=False,
        )
        Z = complex(r.Z[0])
        # C = Im(Y) / ω with Y = 1/Z. Take |Im(Y)| so a noisy negative
        # imaginary part (always indicative of a fit glitch rather than
        # physics) does not poison the 1/C² plot.
        Y = 1.0 / Z
        C_arr[k] = abs(Y.imag) / omega
        if progress is not None:
            progress(
                "mott_schottky", k + 1, len(V_arr),
                f"V={V_dc:.3f} V, C={C_arr[k]:.3e} F/m²",
            )

    V_bi_fit, N_eff_fit, V_lo, V_hi = _fit_mott_schottky(V_arr, C_arr, eps_r)

    return MottSchottkyResult(
        V=V_arr,
        C=C_arr,
        one_over_C2=1.0 / (C_arr * C_arr),
        V_bi_fit=V_bi_fit,
        N_eff_fit=N_eff_fit,
        V_fit_lo=V_lo,
        V_fit_hi=V_hi,
        frequency=float(frequency),
        eps_r_used=float(eps_r),
    )

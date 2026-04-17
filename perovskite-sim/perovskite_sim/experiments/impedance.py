from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import run_transient, build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.experiments.jv_sweep import _compute_current, _total_current_faces


@dataclass(frozen=True)
class ImpedanceResult:
    frequencies: np.ndarray
    Z: np.ndarray           # complex impedance [Ω m²]


def extract_impedance(
    frequencies: np.ndarray,
    delta_V: float = 0.01,
    t_settle: float = 1e-3,
    n_cycles: int = 5,
    dummy_mode: bool = False,
) -> np.ndarray:
    """
    Returns complex impedance array Z [Ω m²] for each frequency.
    dummy_mode=True returns synthetic RC response for testing.
    """
    if dummy_mode:
        # RC circuit: Z = R + 1/(jωC)
        R = 10.0; C = 1e-6
        omega = 2 * np.pi * frequencies
        return R + 1.0 / (1j * omega * C)

    raise NotImplementedError("Full IS requires a DeviceStack argument.")


def _linear_detrend(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove a linear trend from *y* sampled at times *t*.

    Retained as a utility for external callers and diagnostics. The primary
    lock-in path (``_lockin_extract``) no longer uses this helper, because
    detrending *before* projecting onto sin/cos leaks a fraction of the AC
    component into the subtracted slope whenever the sampling window is not
    a strict integer number of periods. Instead ``_lockin_extract`` does a
    joint least-squares fit to ``[1, t, sin(ωt), cos(ωt)]``, which absorbs
    any linear drift into the first two basis vectors while leaving the
    sin/cos amplitudes untouched.
    """
    coeffs = np.polyfit(t, y, 1)
    return y - np.polyval(coeffs, t)


def _lockin_extract(
    I_t: np.ndarray,
    t: np.ndarray,
    freq: float,
    delta_V: float,
) -> complex:
    """Extract complex impedance Z = V̂ / Î from a passive-convention current.

    Implements the lock-in step used inside ``run_impedance`` via a single
    least-squares projection onto the basis ``[1, t, sin(ωt), cos(ωt)]``:

    1. Build the design matrix from the four basis vectors evaluated at the
       sample times.
    2. Solve ``A·c = I_t`` in the least-squares sense. The sin/cos coefficients
       ``c[2], c[3]`` are the in-phase and quadrature amplitudes of the AC
       response; any DC offset or linear drift is absorbed into ``c[0], c[1]``
       and cannot contaminate them.
    3. Form the phasor Î = I_in + j·I_quad under the imag-part convention
       (V(t) = δV·sin(ωt) ⇒ V̂ = δV), and return Z = δV / Î.

    Why joint LS instead of detrend → project
    -----------------------------------------
    The previous implementation detrended *y* first and then projected onto
    sin/cos. When the sampling window was not an exact integer number of
    periods (which is the case in ``run_impedance``: the 80-sample midpoint
    grid for ``n_extract=2, pts_per_cycle=40`` spans 1.975·T, not 2·T), the
    linear-fit slope picks up a non-zero projection onto the sinusoid itself,
    and ~15–20% of the AC amplitude gets subtracted out — biasing |Z| high
    by the same factor. A joint fit to the combined basis is exact for any
    finite window and any sampling pattern, and trivially reduces to the
    textbook lock-in in the orthogonal limit (integer-period window).

    This helper is called by ``run_impedance`` and drives the analytic
    Randles-circuit regression test
    (``tests/unit/experiments/test_impedance_randles.py``).

    Parameters
    ----------
    I_t     : (M,) current samples in passive convention [A m⁻²].
    t       : (M,) sample times [s]; need not be uniform or an integer number
              of periods — the joint fit is unbiased on any window.
    freq    : excitation frequency [Hz].
    delta_V : voltage-excitation amplitude [V].

    Returns
    -------
    Z       : complex impedance [Ω m²].
    """
    y = np.asarray(I_t, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi * freq
    # Design matrix: [1, t, sin(ωt), cos(ωt)]. Scale t to O(1) so the
    # least-squares system is well-conditioned regardless of the absolute
    # time units (which span many decades across the freq sweep).
    t_scale = t_arr[-1] - t_arr[0]
    if t_scale <= 0.0:
        t_scale = 1.0
    A = np.column_stack([
        np.ones_like(t_arr),
        (t_arr - t_arr[0]) / t_scale,
        np.sin(omega * t_arr),
        np.cos(omega * t_arr),
    ])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    I_in = float(coeffs[2])
    I_quad = float(coeffs[3])
    delta_I = I_in + 1j * I_quad
    if abs(delta_I) == 0.0:
        return complex(np.inf, 0.0)
    return delta_V / delta_I


def run_impedance(
    stack: DeviceStack,
    frequencies: np.ndarray,
    V_dc: float = 0.9,
    delta_V: float = 0.01,
    N_grid: int = 60,
    n_cycles: int = 5,
    n_extract: int = 2,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    illuminated: bool = True,
    progress: ProgressCallback | None = None,
) -> ImpedanceResult:
    """Run small-signal impedance at each frequency.

    Parameters
    ----------
    n_cycles : int
        Total AC cycles to simulate (includes settling + extraction).
    n_extract : int
        Number of *final* cycles used for lock-in extraction. The preceding
        ``n_cycles - n_extract`` cycles serve as ionic-settling warm-up.
        Using >1 cycle for extraction reduces noise via averaging.
    illuminated : bool, default True
        If True (default), the DC steady-state and every AC cycle run
        under AM1.5G illumination — appropriate for operating-point
        impedance spectroscopy of solar cells. If False, both legs run
        in the dark (G = 0 everywhere) — required for Mott-Schottky C-V
        analysis, where photogenerated carriers would mask the depletion
        capacitance.
    """
    if len(frequencies) == 0:
        raise ValueError("frequencies must be non-empty")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and positive")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if delta_V <= 0:
        raise ValueError(f"delta_V must be positive, got {delta_V}")
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    n_extract = min(max(n_extract, 1), n_cycles)

    # Grid construction uses electrical layers only; substrate is optical-only.
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    dx_faces = np.diff(x)
    L_total = float(x[-1] - x[0])
    # Build the material cache once — reused across every frequency and
    # every RHS call inside each frequency's transient.
    mat = build_material_arrays(x, stack)
    # Pre-condition: DC steady state at V_dc. Illuminated path uses the
    # dark→light solver; dark path starts from equilibrium and (if
    # V_dc ≠ 0) drives to V_dc via a short dark transient so the AC
    # cycles begin at the correct operating point.
    if illuminated:
        y_dc = solve_illuminated_ss(x, stack, V_app=V_dc, rtol=rtol, atol=atol)
    else:
        y_eq = solve_equilibrium(x, stack)
        if abs(V_dc) < 1e-12:
            y_dc = y_eq
        else:
            t_settle = 1e-3
            sol_dc = run_transient(
                x, y_eq, (0.0, t_settle), np.array([t_settle]),
                stack, illuminated=False, V_app=V_dc,
                rtol=rtol, atol=atol, mat=mat,
            )
            y_dc = sol_dc.y[:, -1] if sol_dc.success else y_eq

    Z_arr = np.zeros(len(frequencies), dtype=complex)
    pts_per_cycle = 40  # integer pts per cycle ⇒ last cycle spans an exact period
    for k, f in enumerate(frequencies):
        T_period = 1.0 / f
        dt = T_period / pts_per_cycle
        n_intervals = n_cycles * pts_per_cycle
        t_eval = np.arange(n_intervals + 1, dtype=float) * dt

        def V_ac(t):
            return V_dc + delta_V * np.sin(2 * np.pi * f * t)

        y = y_dc.copy()
        J_t = np.zeros(n_intervals, dtype=float)
        V_prev = V_dc
        for i in range(n_intervals):
            t_lo, t_hi = t_eval[i], t_eval[i + 1]
            V_i = V_ac(0.5 * (t_lo + t_hi))
            y_prev = y.copy()
            # Cap Radau's internal step so it cannot skip over the AC
            # excitation within a half-period — same rationale as jv_sweep
            # near V_bi, where an under-estimated error lets the solver
            # take a giant step and miss the small-signal response.
            sol = run_transient(x, y, (t_lo, t_hi), np.array([t_hi]),
                                stack, illuminated=illuminated, V_app=V_i,
                                rtol=rtol, atol=atol,
                                max_step=(t_hi - t_lo) / 5.0,
                                mat=mat)
            if not sol.success:
                raise RuntimeError(f"impedance transient failed at f={f:.3e} Hz, step {i}")
            y = sol.y[:, -1]
            # Spatial average of the full-device total (conduction + displacement)
            # current. By 1D current continuity the AC component is essentially
            # space-uniform, and averaging removes the boundary-face quantization
            # noise that otherwise dominates the capacitive signal at face 0.
            J_face = _total_current_faces(
                x, y, stack, V_i, y_prev=y_prev, dt=t_hi - t_lo, mat=mat,
                V_app_prev=V_prev,
            )
            J_t[i] = float(np.sum(J_face * dx_faces) / L_total)
            V_prev = V_i

        # Lock-in over the last n_extract cycles.  Using multiple cycles
        # averages out residual transient noise; the lock-in helper applies
        # linear detrend internally so ionic drift that the settling cycles
        # didn't fully absorb is removed before projection.
        n_extract_pts = n_extract * pts_per_cycle
        J_ext = J_t[-n_extract_pts:]
        t_ext = 0.5 * (t_eval[-n_extract_pts - 1:-1] + t_eval[-n_extract_pts:])
        # Passive-convention current is -J (so that a positive δV drives a
        # positive Î through a resistive device). The lock-in helper assumes
        # passive convention so the sign flip happens here.
        Z_arr[k] = _lockin_extract(-J_ext, t_ext, f, delta_V)

        if progress is not None:
            progress("impedance", k + 1, len(frequencies), f"f={f:.3e} Hz")

    return ImpedanceResult(frequencies=frequencies, Z=Z_arr)

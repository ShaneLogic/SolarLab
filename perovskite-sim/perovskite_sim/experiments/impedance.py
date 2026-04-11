from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import run_transient
from perovskite_sim.experiments.jv_sweep import _compute_current


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


def run_impedance(
    stack: DeviceStack,
    frequencies: np.ndarray,
    V_dc: float = 0.9,
    delta_V: float = 0.01,
    N_grid: int = 60,
    n_cycles: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> ImpedanceResult:
    """Run small-signal impedance at each frequency."""
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

    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    # Pre-condition: illuminated steady state at V_dc (avoids dark→light transient)
    y_dc = solve_illuminated_ss(x, stack, V_app=V_dc, rtol=rtol, atol=atol)

    Z_arr = np.zeros(len(frequencies), dtype=complex)
    pts_per_cycle = 20  # integer pts per cycle ⇒ last cycle spans an exact period
    for k, f in enumerate(frequencies):
        T_period = 1.0 / f
        dt = T_period / pts_per_cycle
        n_intervals = n_cycles * pts_per_cycle
        t_eval = np.arange(n_intervals + 1, dtype=float) * dt

        def V_ac(t):
            return V_dc + delta_V * np.sin(2 * np.pi * f * t)

        y = y_dc.copy()
        J_t = np.zeros(n_intervals, dtype=float)
        for i in range(n_intervals):
            t_lo, t_hi = t_eval[i], t_eval[i + 1]
            V_i = V_ac(0.5 * (t_lo + t_hi))
            y_prev = y.copy()
            # Cap Radau's internal step so it cannot skip over the AC
            # excitation within a half-period — same rationale as jv_sweep
            # near V_bi, where an under-estimated error lets the solver
            # take a giant step and miss the small-signal response.
            sol = run_transient(x, y, (t_lo, t_hi), np.array([t_hi]),
                                stack, illuminated=True, V_app=V_i,
                                rtol=rtol, atol=atol,
                                max_step=(t_hi - t_lo) / 5.0)
            if not sol.success:
                raise RuntimeError(f"impedance transient failed at f={f:.3e} Hz, step {i}")
            y = sol.y[:, -1]
            J_t[i] = _compute_current(x, y, stack, V_i, y_prev=y_prev, dt=t_hi - t_lo)

        # Lock-in over the last full cycle. pts_per_cycle samples cover exactly
        # one period, so sin/cos references integrate to zero on any DC term —
        # but we still detrend J explicitly to guard against slow drift from
        # incompletely settled ion dynamics superimposed on the AC perturbation.
        J_ac = J_t[-pts_per_cycle:]
        t_mid = 0.5 * (t_eval[-pts_per_cycle - 1:-1] + t_eval[-pts_per_cycle:])
        I_ac = -(J_ac - J_ac.mean())  # passive sign convention + detrend
        sin_ref = np.sin(2 * np.pi * f * t_mid)
        cos_ref = np.cos(2 * np.pi * f * t_mid)
        I_in = 2.0 * np.mean(I_ac * sin_ref)
        I_quad = 2.0 * np.mean(I_ac * cos_ref)

        # Excitation ∝ sin(ωt); capacitive devices give Im(Z) < 0, so the
        # cosine lock-in term enters the phasor with a minus sign.
        delta_I = I_in - 1j * I_quad
        Z_arr[k] = delta_V / delta_I if abs(delta_I) > 0 else complex(np.inf, 0)

    return ImpedanceResult(frequencies=frequencies, Z=Z_arr)

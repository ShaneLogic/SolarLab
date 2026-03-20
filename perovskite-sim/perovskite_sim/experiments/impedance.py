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
    for k, f in enumerate(frequencies):
        T_period = 1.0 / f
        t_eval = np.linspace(0, n_cycles * T_period, n_cycles * 20)
        t_span = (0.0, t_eval[-1])

        def V_ac(t):
            return V_dc + delta_V * np.sin(2 * np.pi * f * t)

        # Step through time with piecewise constant V
        y = y_dc.copy()
        J_t = np.zeros_like(t_eval)
        for i, t_i in enumerate(t_eval[:-1]):
            V_i = V_ac(0.5 * (t_eval[i] + t_eval[i + 1]))
            sol = run_transient(x, y, (t_eval[i], t_eval[i + 1]),
                                np.array([t_eval[i + 1]]),
                                stack, illuminated=True, V_app=V_i,
                                rtol=rtol, atol=atol)
            y = sol.y[:, -1]
            J_t[i] = _compute_current(x, y, stack, V_i)

        # Lock-in extraction from last cycle (settled AC response).
        # J_t[i] is the current after integrating over [t_eval[i], t_eval[i+1]]
        # with voltage V_ac applied at the midpoint of that interval.
        # Use midpoint times for the sin/cos reference to match the actual
        # voltage phase, eliminating the half-step bias at high frequencies.
        pts = 20   # points per cycle
        t_mid_all = 0.5 * (t_eval[:-1] + t_eval[1:])   # midpoint of each interval
        t_ac = t_mid_all[-pts:]         # last cycle midpoint times  (len=pts)
        J_ac = J_t[:-1][-pts:]         # last cycle current values   (len=pts)
        sin_ref = np.sin(2 * np.pi * f * t_ac)
        cos_ref = np.cos(2 * np.pi * f * t_ac)
        J_in   = 2.0 * np.mean(J_ac * sin_ref)   # in-phase with V = δV·sin
        J_quad = 2.0 * np.mean(J_ac * cos_ref)   # quadrature
        delta_J = J_in + 1j * J_quad
        Z_arr[k] = delta_V / delta_J if abs(delta_J) > 0 else complex(np.inf, 0)

    return ImpedanceResult(frequencies=frequencies, Z=Z_arr)

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import solve_poisson
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    _build_layerwise_arrays, _equilibrium_bc, _build_carrier_params,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.constants import Q, V_T


@dataclass(frozen=True)
class JVMetrics:
    V_oc: float
    J_sc: float
    FF: float
    PCE: float


@dataclass(frozen=True)
class JVResult:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: JVMetrics
    metrics_rev: JVMetrics
    hysteresis_index: float


def compute_metrics(V: np.ndarray, J: np.ndarray) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²)."""
    J_sc = float(np.interp(0.0, V, J))
    # V_oc: where J crosses zero
    sign_changes = np.where(np.diff(np.sign(J)))[0]
    if len(sign_changes) == 0:
        return JVMetrics(V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0)
    idx = sign_changes[-1]
    V_oc = float(V[idx] - J[idx] * (V[idx+1] - V[idx]) / (J[idx+1] - J[idx]))
    P = V * J
    P_mpp = float(np.max(P[V <= V_oc]))
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    # PCE assuming 1000 W/m² AM1.5G
    PCE = P_mpp / 1000.0
    return JVMetrics(V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE)


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
) -> float:
    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev, J_rev)
    if m_rev.PCE == 0:
        return 0.0
    return (m_rev.PCE - m_fwd.PCE) / m_rev.PCE


def _compute_current(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
) -> float:
    """Extract total current density J [A/m²] via Scharfetter-Gummel fluxes.

    Uses the same per-layer D values and phi_right convention as assemble_rhs
    so that J_n + J_p is exactly conserved at steady state.  Averages over
    all inter-node faces for numerical robustness.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    N = len(x)
    sv = StateVec.unpack(y, N)
    dx = np.diff(x)

    eps_r, _, _, N_A, N_D, _ = _build_layerwise_arrays(x, stack)

    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    n = sv.n.copy(); n[0] = n_L; n[-1] = n_R
    p = sv.p.copy(); p[0] = p_L; p[-1] = p_R

    rho = Q * (p - n + sv.P - N_A + N_D)
    # Must match assemble_rhs convention: forward bias reduces the built-in field
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=stack.V_bi - V_app)

    carrier_p = _build_carrier_params(x, stack)
    D_n_face = carrier_p["D_n"]   # (N-1,)
    D_p_face = carrier_p["D_p"]   # (N-1,)

    xi = (phi[1:] - phi[:-1]) / V_T
    B_pos = bernoulli(xi)
    B_neg = bernoulli(-xi)

    J_n = Q * D_n_face / dx * (B_pos * n[1:] - B_neg * n[:-1])
    J_p = Q * D_p_face / dx * (B_pos * p[:-1] - B_neg * p[1:])

    # J_n + J_p = internal conventional current (negative at SC: photo-electrons
    # flow right → conventional current flows left).  Negate for external
    # circuit convention: J_sc > 0.
    return -float(np.mean(J_n + J_p))


def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,      # V/s
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> JVResult:
    """Run forward and reverse J-V sweeps."""
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_rate <= 0:
        raise ValueError(f"v_rate must be positive, got {v_rate}")
    for i, layer in enumerate(stack.layers):
        if layer.thickness <= 0:
            raise ValueError(
                f"layer {i} ({layer.name!r}) has non-positive thickness {layer.thickness}"
            )

    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    L = stack.total_thickness

    # Start from illuminated SC state: carriers equilibrated, ions at initial profile
    y_eq = solve_illuminated_ss(x, stack, V_app=0.0, rtol=rtol, atol=atol)

    def _sweep(V_start: float, V_end: float, y_init: np.ndarray):
        """Sweep from V_start to V_end, starting from carrier state y_init.

        Returns (V_arr, J_arr, y_final) so sweeps can be chained: the reverse
        sweep starts from the light-soaked state at the end of the forward sweep,
        matching how hysteresis is measured in the laboratory.
        """
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        y = y_init.copy()
        for k, V_k in enumerate(V_arr):
            # Each step integrates for the full dt so the state is always
            # self-consistent with the applied V_k before J is sampled.
            t_span = (t_points[k], t_points[k] + dt)
            sol = run_transient(x, y, t_span, np.array([t_span[-1]]),
                                stack, illuminated=True, V_app=V_k, rtol=rtol, atol=atol)
            y = sol.y[:, -1]
            J_arr[k] = _compute_current(x, y, stack, V_k)
        return V_arr, J_arr, y

    # Forward sweep: dark equilibrium → short circuit → open circuit
    V_fwd, J_fwd, y_oc = _sweep(0.0, stack.V_bi, y_eq)
    # Reverse sweep: continue from light-soaked OC state → short circuit
    V_rev, J_rev, _ = _sweep(stack.V_bi, 0.0, y_oc)

    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1])
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1])

    return JVResult(V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
                    metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI)

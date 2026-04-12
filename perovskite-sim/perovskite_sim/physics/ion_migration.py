from __future__ import annotations
import numpy as np

from perovskite_sim.discretization.fe_operators import bernoulli


def ion_flux_steric(
    phi: np.ndarray,   # [phi_i, phi_{i+1}]
    P: np.ndarray,     # [P_i,   P_{i+1}]
    h: float,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> float:
    """
    Sterically corrected positive-vacancy flux F_P [m⁻² s⁻¹].

    The vacancy drift term uses a Scharfetter-Gummel face discretisation so the
    flux has the correct sign for a positively charged mobile ion and remains
    conservative on strongly biased, non-uniform grids.
    """
    P_avg = 0.5 * (P[0] + P[1])
    steric = 1.0 / max(1.0 - np.clip(P_avg / P_lim, 0.0, 0.999999), 1e-6)
    xi = (phi[1] - phi[0]) / V_T
    D_eff = D_I * steric
    return float(D_eff / h * (bernoulli(np.array([xi]))[0] * P[0]
                              - bernoulli(np.array([-xi]))[0] * P[1]))


def ion_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    P: np.ndarray,
    D_I: np.ndarray | float,
    V_T: float,
    P_lim: np.ndarray | float,
) -> np.ndarray:
    """
    Vectorized dP/dt = -dF_P/dx for all nodes.
    Zero-flux BCs at both contacts.
    """
    P = np.asarray(P, dtype=float)
    dx = np.diff(x)                              # (N-1,)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    P_lim_face = np.broadcast_to(np.asarray(P_lim, dtype=float), dx.shape)
    P_avg = 0.5 * (P[:-1] + P[1:])              # (N-1,)
    steric = 1.0 / np.maximum(1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6)
    xi = (phi[1:] - phi[:-1]) / V_T             # (N-1,)
    D_eff = D_I_face * steric
    F_int = D_eff / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])

    # Zero-flux BCs: pad with 0 at both ends
    F_full = np.concatenate([[0.0], F_int, [0.0]])   # (N+1,)

    # Dual-grid cell widths
    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell


def ion_continuity_rhs_neg(
    x: np.ndarray,
    phi: np.ndarray,
    P_neg: np.ndarray,
    D_I: np.ndarray | float,
    V_T: float,
    P_lim: np.ndarray | float,
) -> np.ndarray:
    """dP_neg/dt for a negatively charged mobile ion species.

    Same SG discretization as the positive species but with reversed drift
    direction: the sign of xi is negated because q_neg = -q.
    Zero-flux BCs at both contacts.
    """
    P_neg = np.asarray(P_neg, dtype=float)
    dx = np.diff(x)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    P_lim_face = np.broadcast_to(np.asarray(P_lim, dtype=float), dx.shape)
    P_avg = 0.5 * (P_neg[:-1] + P_neg[1:])
    steric = 1.0 / np.maximum(1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6)
    # Reversed drift: negative charge → xi flipped
    xi = -(phi[1:] - phi[:-1]) / V_T
    D_eff = D_I_face * steric
    F_int = D_eff / dx * (bernoulli(xi) * P_neg[:-1] - bernoulli(-xi) * P_neg[1:])

    F_full = np.concatenate([[0.0], F_int, [0.0]])

    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell

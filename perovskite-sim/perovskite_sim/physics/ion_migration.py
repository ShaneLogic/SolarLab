from __future__ import annotations
import numpy as np


def ion_flux_steric(
    phi: np.ndarray,   # [phi_i, phi_{i+1}]
    P: np.ndarray,     # [P_i,   P_{i+1}]
    h: float,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> float:
    """
    Steric Blakemore ion vacancy flux F_P [m⁻² s⁻¹] from node i to i+1.
    F_P = -D_I * dP/dx / (1 - P_avg/P_lim)  +  (D_I/V_T) * P_avg * dφ/dx
    """
    P_avg = 0.5 * (P[0] + P[1])
    steric = 1.0 / (1.0 - P_avg / P_lim)
    grad_P = (P[1] - P[0]) / h
    grad_phi = (phi[1] - phi[0]) / h
    return float(-D_I * grad_P * steric + (D_I / V_T) * P_avg * grad_phi)


def ion_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    P: np.ndarray,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> np.ndarray:
    """
    Vectorized dP/dt = -dF_P/dx for all nodes.
    Zero-flux BCs at both contacts.
    """
    dx = np.diff(x)                              # (N-1,)
    P_avg = 0.5 * (P[:-1] + P[1:])              # (N-1,)
    steric = 1.0 / (1.0 - np.clip(P_avg / P_lim, 0.0, 0.9999))
    grad_P   = (P[1:]   - P[:-1])   / dx        # (N-1,)
    grad_phi = (phi[1:] - phi[:-1]) / dx        # (N-1,)
    F_int = -D_I * grad_P * steric + (D_I / V_T) * P_avg * grad_phi  # (N-1,)

    # Zero-flux BCs: pad with 0 at both ends
    F_full = np.concatenate([[0.0], F_int, [0.0]])   # (N+1,)

    # Dual-grid cell widths
    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell

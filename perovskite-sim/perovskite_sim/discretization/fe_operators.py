from __future__ import annotations
import numpy as np


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Bernoulli function B(x) = x / (exp(x) - 1), numerically stable."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1e-8
    large = ~small
    result[small] = 1.0 - x[small] / 2.0 + x[small]**2 / 12.0
    result[large] = x[large] / np.expm1(x[large])
    return result


def sg_flux_n(
    phi: np.ndarray,  # [phi_i, phi_{i+1}]  shape (2,)
    n: np.ndarray,    # [n_i,   n_{i+1}]    shape (2,)
    h: float,
    D_n: float,
    V_T: float,
) -> float:
    """Scharfetter-Gummel electron current J_n [A/m²] between node pair."""
    q = 1.602176634e-19
    xi = (phi[1] - phi[0]) / V_T
    return float(q * D_n / h * (bernoulli(np.array([xi]))[0] * n[1]
                                - bernoulli(np.array([-xi]))[0] * n[0]))


def sg_flux_p(
    phi: np.ndarray,  # [phi_i, phi_{i+1}]
    p: np.ndarray,    # [p_i,   p_{i+1}]
    h: float,
    D_p: float,
    V_T: float,
) -> float:
    """Scharfetter-Gummel hole current J_p [A/m²] between node pair."""
    q = 1.602176634e-19
    xi = (phi[1] - phi[0]) / V_T
    return float(q * D_p / h * (bernoulli(np.array([xi]))[0] * p[0]
                                - bernoulli(np.array([-xi]))[0] * p[1]))


# ── Vectorized versions: operate on all N-1 faces simultaneously ─────────────

def sg_fluxes_n(
    phi: np.ndarray,   # shape (N,)
    n: np.ndarray,     # shape (N,)
    dx: np.ndarray,    # shape (N-1,)  spacing between consecutive nodes
    D_n: float,
    V_T: float,
) -> np.ndarray:
    """Vectorized SG electron flux at all N-1 inter-node faces [A/m²]."""
    q = 1.602176634e-19
    xi = (phi[1:] - phi[:-1]) / V_T
    return q * D_n / dx * (bernoulli(xi) * n[1:] - bernoulli(-xi) * n[:-1])


def sg_fluxes_p(
    phi: np.ndarray,   # shape (N,)
    p: np.ndarray,     # shape (N,)
    dx: np.ndarray,    # shape (N-1,)
    D_p: float,
    V_T: float,
) -> np.ndarray:
    """Vectorized SG hole flux at all N-1 inter-node faces [A/m²]."""
    q = 1.602176634e-19
    xi = (phi[1:] - phi[:-1]) / V_T
    return q * D_p / dx * (bernoulli(xi) * p[:-1] - bernoulli(-xi) * p[1:])

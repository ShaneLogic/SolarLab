from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import K_B, Q


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Bernoulli function B(x) = x / (exp(x) - 1), numerically stable.

    For x > ~700 the analytic limit is B(x) → 0 but `np.expm1(x)` overflows
    to +inf and emits a RuntimeWarning. Branch the huge-positive case to the
    explicit zero limit so the warning never fires under aggressive 2D
    Newton iterations that produce out-of-range xi.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1e-8
    huge_pos = x > 700.0
    large = ~small & ~huge_pos
    result[small] = 1.0 - x[small] / 2.0 + x[small]**2 / 12.0
    result[huge_pos] = 0.0
    result[large] = x[large] / np.expm1(x[large])
    return result


def bernoulli_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of :func:`bernoulli`, evaluated without cancellation.

    The direct quotient derivative loses all useful digits near zero and
    overflows when its exponential terms are squared at large positive
    arguments.  The small-argument series and the ``exp(-x)`` form below
    cover those two regimes while retaining the analytic asymptotes
    ``B'(+inf)=0`` and ``B'(-inf)=-1``.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1.0e-4
    large_positive = x > 50.0
    regular = ~small & ~large_positive

    xs = x[small]
    result[small] = (
        -0.5
        + xs / 6.0
        - xs**3 / 180.0
        + xs**5 / 5040.0
    )

    xp = x[large_positive]
    decay = np.exp(-xp)
    result[large_positive] = (
        decay * (1.0 - xp - decay) / (1.0 - decay) ** 2
    )

    xr = x[regular]
    exp_x = np.exp(xr)
    denominator = np.expm1(xr)
    result[regular] = (
        denominator - xr * exp_x
    ) / denominator**2
    return result


@dataclass(frozen=True, slots=True)
class ScharfetterGummelFaceJacobian:
    """Local derivatives of one carrier current on every grid face."""

    flux: np.ndarray
    density_left_derivative: np.ndarray
    density_right_derivative: np.ndarray
    potential_left_derivative: np.ndarray
    potential_right_derivative: np.ndarray


def _validate_sg_jacobian_inputs(
    phi: np.ndarray,
    density: np.ndarray,
    dx: np.ndarray,
    diffusivity: np.ndarray | float,
    V_T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    potential = np.asarray(phi, dtype=float)
    carrier_density = np.asarray(density, dtype=float)
    spacing = np.asarray(dx, dtype=float)
    if (
        potential.ndim != 1
        or potential.size < 2
        or carrier_density.shape != potential.shape
        or spacing.shape != (potential.size - 1,)
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(carrier_density))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.isfinite(V_T)
        or V_T <= 0.0
    ):
        raise ValueError(
            "SG Jacobian inputs must be finite, shape matched, and use "
            "positive spacing and thermal voltage"
        )
    diffusion = np.broadcast_to(
        np.asarray(diffusivity, dtype=float), spacing.shape
    )
    if not np.all(np.isfinite(diffusion)) or np.any(diffusion < 0.0):
        raise ValueError("SG diffusivity must be finite and non-negative")
    return potential, carrier_density, spacing, diffusion


def sg_fluxes_n_jacobian(
    phi: np.ndarray,
    n: np.ndarray,
    dx: np.ndarray,
    D_n: np.ndarray | float,
    V_T: float,
) -> ScharfetterGummelFaceJacobian:
    """Analytic electron-current derivatives for fixed face diffusivity."""
    potential, density, spacing, diffusion = _validate_sg_jacobian_inputs(
        phi, n, dx, D_n, V_T
    )
    xi = (potential[1:] - potential[:-1]) / V_T
    B_pos = bernoulli(xi)
    B_neg = bernoulli(-xi)
    dB_pos = bernoulli_derivative(xi)
    dB_neg = bernoulli_derivative(-xi)
    prefactor = Q * diffusion / spacing
    xi_derivative = prefactor * (
        dB_pos * density[1:] + dB_neg * density[:-1]
    )
    return ScharfetterGummelFaceJacobian(
        flux=sg_fluxes_n(potential, density, spacing, diffusion, V_T),
        density_left_derivative=-prefactor * B_neg,
        density_right_derivative=prefactor * B_pos,
        potential_left_derivative=-xi_derivative / V_T,
        potential_right_derivative=xi_derivative / V_T,
    )


def sg_fluxes_p_jacobian(
    phi: np.ndarray,
    p: np.ndarray,
    dx: np.ndarray,
    D_p: np.ndarray | float,
    V_T: float,
) -> ScharfetterGummelFaceJacobian:
    """Analytic hole-current derivatives for fixed face diffusivity."""
    potential, density, spacing, diffusion = _validate_sg_jacobian_inputs(
        phi, p, dx, D_p, V_T
    )
    xi = (potential[1:] - potential[:-1]) / V_T
    B_pos = bernoulli(xi)
    B_neg = bernoulli(-xi)
    dB_pos = bernoulli_derivative(xi)
    dB_neg = bernoulli_derivative(-xi)
    prefactor = Q * diffusion / spacing
    xi_derivative = prefactor * (
        dB_pos * density[:-1] + dB_neg * density[1:]
    )
    return ScharfetterGummelFaceJacobian(
        flux=sg_fluxes_p(potential, density, spacing, diffusion, V_T),
        density_left_derivative=prefactor * B_pos,
        density_right_derivative=-prefactor * B_neg,
        potential_left_derivative=-xi_derivative / V_T,
        potential_right_derivative=xi_derivative / V_T,
    )


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


def thermionic_emission_flux(
    n_left: float, n_right: float, delta_E: float, T: float, A_star: float,
    N_dos: float | None = None,
) -> float:
    """Thermionic emission current [A/m^2] across a band offset.

    delta_E: band offset E_right - E_left [eV]. Positive = step-up barrier left->right.
    A_star: Richardson constant [A/(m^2 K^2)]
    N_dos: effective band-edge density of states of the emitting reservoir
        [m^-3] (N_C for electrons, N_V for holes). See the two forms below.

    Two normalizations are supported:

    * Legacy density-weighted bound (``N_dos is None``, the historical
      default). ``J = A*T^2 * (n_L e^(...) - n_R e^(...))``. This is
      dimensionally A/m^2 * m^-3 = A/m^5, so it is NOT the physical
      Richardson-Dushman current; it functions only as an empirical
      magnitude cap whose scale is tied to the SI unit system. In practice
      the density weighting makes ``|J_TE|`` enormous (~1e28-1e35 on a
      perovskite stack), so the ``min(|J_SG|, |J_TE|)`` cap almost never
      binds and the term is close to inert.

    * Physical emission-velocity form (``N_dos`` supplied). Dividing by the
      band-edge DOS converts the Richardson constant into an emission
      velocity ``v_R = A*T^2 / (q N_dos)`` [m/s], giving the dimensionally
      correct interface-limited current ``J = q v_R (n_L e^(...) -
      n_R e^(...))``. At real interface densities this lands in the same
      range as the drift-diffusion flux, so the cap binds meaningfully. A
      single ``N_dos`` scales both legs equally, so the equilibrium
      cancellation ``J = 0`` is preserved exactly.
    """
    v_t = K_B * T / Q
    left_term = n_left * math.exp(-max(delta_E, 0.0) / v_t)
    right_term = n_right * math.exp(-max(-delta_E, 0.0) / v_t)
    J = A_star * T**2 * (left_term - right_term)
    if N_dos is not None and N_dos > 0.0:
        J = J / N_dos
    return J

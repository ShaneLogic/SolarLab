"""Stable complete Fermi-Dirac integrals used by interface transport."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.special import expit, spence


_ETA_MIN = -40.0
_ETA_MAX = 20.0
_ETA_STEP = 5.0e-3


def _half_sommerfeld(eta: float) -> float:
    root = math.sqrt(eta)
    return (
        4.0 * eta * root / (3.0 * math.sqrt(math.pi))
        + math.pi**2 / (6.0 * math.sqrt(math.pi) * root)
    )


@lru_cache(maxsize=1)
def _half_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate normalized F_1/2 with deterministic Gaussian quadrature."""
    eta = np.linspace(
        _ETA_MIN,
        _ETA_MAX,
        int(round((_ETA_MAX - _ETA_MIN) / _ETA_STEP)) + 1,
    )
    nodes, weights = np.polynomial.legendre.leggauss(256)
    transformed = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    energy = transformed / (1.0 - transformed)
    jacobian = 1.0 / (1.0 - transformed) ** 2
    kernel = weights * np.sqrt(energy) * jacobian
    values = np.empty_like(eta)
    for start in range(0, len(eta), 256):
        stop = min(start + 256, len(eta))
        occupation = expit(eta[start:stop, None] - energy[None, :])
        values[start:stop] = (
            2.0 / math.sqrt(math.pi) * (occupation @ kernel)
        )
    return eta, values, np.log(values)


def fermi_dirac_half(eta: float) -> float:
    """Return normalized complete F_1/2(eta).

    The normalization is chosen so F_1/2(eta) approaches exp(eta) in the
    non-degenerate limit and carrier density is ``N_C * F_1/2(eta)``.
    """
    value = float(eta)
    if math.isnan(value):
        return math.nan
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return math.inf
    if value <= _ETA_MIN:
        return math.exp(value)
    if value >= _ETA_MAX:
        return _half_sommerfeld(value)
    eta_grid, _table, log_table = _half_table()
    return float(
        math.exp(np.interp(value, eta_grid, log_table))
    )


def fermi_dirac_half_log_derivative(eta: float) -> float:
    """Return ``d(log(F_1/2)) / d eta`` for the implemented approximation.

    Inside the tabulated interval this differentiates the piecewise-linear
    interpolation of ``log(F_1/2)`` used by :func:`fermi_dirac_half`.  The
    inverse-map derivative is therefore consistent with the constitutive law
    to machine precision instead of differentiating a different quadrature
    approximation.
    """
    value = float(eta)
    if math.isnan(value):
        return math.nan
    if value == -math.inf:
        return 1.0
    if value == math.inf:
        return 0.0
    if value <= _ETA_MIN:
        return 1.0
    if value >= _ETA_MAX:
        integral = _half_sommerfeld(value)
        root = math.sqrt(value)
        derivative = (
            2.0 * root / math.sqrt(math.pi)
            - math.pi**2
            / (12.0 * math.sqrt(math.pi) * value * root)
        )
        return derivative / integral
    eta_grid, _table, log_table = _half_table()
    position = (value - _ETA_MIN) / _ETA_STEP
    nearest = int(round(position))
    if (
        0 < nearest < len(eta_grid) - 1
        and abs(position - nearest) <= 1.0e-10
    ):
        return float(
            (log_table[nearest + 1] - log_table[nearest - 1])
            / (eta_grid[nearest + 1] - eta_grid[nearest - 1])
        )
    interval = min(int(position), len(eta_grid) - 2)
    return float(
        (log_table[interval + 1] - log_table[interval])
        / (eta_grid[interval + 1] - eta_grid[interval])
    )


def inverse_fermi_dirac_half(state_to_dos: float) -> float:
    """Return eta satisfying ``F_1/2(eta) = state_to_dos``."""
    ratio = float(state_to_dos)
    if math.isnan(ratio) or ratio < 0.0:
        raise ValueError("state_to_dos must be non-negative")
    if ratio == 0.0:
        return -math.inf
    if ratio == math.inf:
        return math.inf
    log_ratio = math.log(ratio)
    eta_grid, _table, log_table = _half_table()
    if log_ratio <= log_table[0]:
        return log_ratio
    if log_ratio <= log_table[-1]:
        return float(np.interp(log_ratio, log_table, eta_grid))

    coefficient = 4.0 / (3.0 * math.sqrt(math.pi))
    eta = max(_ETA_MAX, math.exp(2.0 / 3.0 * (log_ratio - math.log(coefficient))))
    for _ in range(6):
        root = math.sqrt(eta)
        residual = _half_sommerfeld(eta) - ratio
        derivative = (
            2.0 * root / math.sqrt(math.pi)
            - math.pi**2
            / (12.0 * math.sqrt(math.pi) * eta * root)
        )
        eta = max(_ETA_MAX, eta - residual / derivative)
    return eta


def fermi_dirac_one(eta: float) -> float:
    """Return normalized complete F_1(eta) for 3D thermionic supply."""
    value = float(eta)
    if math.isnan(value):
        return math.nan
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return math.inf
    if value < -4.0:
        activity = math.exp(value)
        total = 0.0
        power = activity
        for order in range(1, 13):
            term = power / (order * order)
            total += term if order % 2 else -term
            power *= activity
        return total
    if value > 40.0:
        return 0.5 * value * value + math.pi**2 / 6.0
    return -float(spence(1.0 + math.exp(value)))


def fermi_dirac_zero(eta: float) -> float:
    """Return normalized ``F_0(eta) = dF_1(eta)/deta`` stably."""
    value = float(eta)
    if math.isnan(value):
        return math.nan
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return math.inf
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


__all__ = [
    "fermi_dirac_half",
    "fermi_dirac_half_log_derivative",
    "fermi_dirac_one",
    "fermi_dirac_zero",
    "inverse_fermi_dirac_half",
]

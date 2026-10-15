"""Normalized complete Fermi-Dirac integral of order 1/2 (M class).

``F_1/2`` is normalized so that ``F_1/2(eta) -> exp(eta)`` in the
non-degenerate limit and the carrier density is ``N * F_1/2(eta)``. Inside
``[-40, 20]`` the value is the exponential of a piecewise-linear interpolation
of ``log F_1/2`` tabulated every ``5e-3`` with 256-point Gauss-Legendre
quadrature; below the table it is ``exp(eta)`` and above it the two-term
Sommerfeld expansion. The table, its log-derivative and the inverse are those
of the legacy engine, bit for bit. ``F_1`` and ``F_0`` (thermionic supply)
belong to the interface-transport slice and are not migrated here.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

__all__ = [
    "FERMI_DIRAC_HALF_TABLE_ETA_MAX", "FERMI_DIRAC_HALF_TABLE_ETA_MIN",
    "FERMI_DIRAC_HALF_TABLE_STEP", "fermi_dirac_half", "fermi_dirac_half_log_derivative",
    "fermi_dirac_half_table", "inverse_fermi_dirac_half",
]

FERMI_DIRAC_HALF_TABLE_ETA_MIN = -40.0
FERMI_DIRAC_HALF_TABLE_ETA_MAX = 20.0
FERMI_DIRAC_HALF_TABLE_STEP = 5.0e-3
_ETA_MIN = FERMI_DIRAC_HALF_TABLE_ETA_MIN
_ETA_MAX = FERMI_DIRAC_HALF_TABLE_ETA_MAX
_ETA_STEP = FERMI_DIRAC_HALF_TABLE_STEP


def _half_sommerfeld(eta: float) -> float:
    root = math.sqrt(eta)
    return (
        4.0 * eta * root / (3.0 * math.sqrt(math.pi))
        + math.pi**2 / (6.0 * math.sqrt(math.pi) * root)
    )


def _half_sommerfeld_derivative(eta: float) -> float:
    root = math.sqrt(eta)
    return (
        2.0 * root / math.sqrt(math.pi)
        - math.pi**2
        / (12.0 * math.sqrt(math.pi) * eta * root)
    )


@lru_cache(maxsize=1)
def _half_table() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    log_values = np.log(values)
    for array in (eta, values, log_values):
        array.setflags(write=False)
    return eta, values, log_values


def fermi_dirac_half_table() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Read-only ``(eta, F_1/2, log F_1/2)`` table behind the interpolation."""
    eta, values, log_values = _half_table()
    return eta.view(), values.view(), log_values.view()


def fermi_dirac_half(eta: float) -> float:
    """Return normalized complete ``F_1/2(eta)``."""
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
    """Return ``d log F_1/2 / d eta`` of the implemented approximation.

    Inside the table this differentiates the piecewise-linear interpolation of
    ``log F_1/2``; a value on a table node uses the centred difference. The
    inverse-map derivative is therefore consistent with the constitutive law
    to machine precision.
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
        return _half_sommerfeld_derivative(value) / integral
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
    """Return ``eta`` satisfying ``F_1/2(eta) = state_to_dos``."""
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
        residual = _half_sommerfeld(eta) - ratio
        eta = max(_ETA_MAX, eta - residual / _half_sommerfeld_derivative(eta))
    return eta

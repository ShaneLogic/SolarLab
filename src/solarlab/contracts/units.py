"""Exact physical dimensions for contract checks; no scale factors.

This is not the configuration unit table in :mod:`solarlab.units`, which
converts user quantities. A contract ``Unit`` only proves that assembled terms
share dimensions, so products and quotients are exact integer arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from solarlab.contracts.errors import ContractError

__all__ = [
    "AREA", "COULOMB", "ELEMENTARY_CHARGE_C", "KELVIN", "LENGTH", "ONE", "PARTICLE",
    "SECOND", "Unit", "VOLT", "VOLUME",
]

# Exact SI 2019 defining constant: converts particle counts/fluxes to charge,
# J_s = z_s * q * F_s (V1 section 15.1). Same binary64 value as the legacy engine.
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class Unit:
    """Dimensions (length, time, charge, number, mass, temperature); no scale."""

    powers: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.powers) != 6 or any(type(x) is not int for x in self.powers):
            raise ContractError("invalid_unit")

    def __mul__(self, other: Unit) -> Unit:
        return Unit(tuple(a + b for a, b in zip(self.powers, other.powers)))  # type: ignore[arg-type]

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(tuple(a - b for a, b in zip(self.powers, other.powers)))  # type: ignore[arg-type]


ONE = Unit((0, 0, 0, 0, 0, 0))
SECOND = Unit((0, 1, 0, 0, 0, 0))
LENGTH = Unit((1, 0, 0, 0, 0, 0))
VOLUME = Unit((3, 0, 0, 0, 0, 0))
AREA = Unit((2, 0, 0, 0, 0, 0))
PARTICLE = Unit((0, 0, 0, 1, 0, 0))
COULOMB = Unit((0, 0, 1, 0, 0, 0))
VOLT = Unit((2, -2, -1, 0, 1, 0))
KELVIN = Unit((0, 0, 0, 0, 0, 1))

"""Small SI/eV unit whitelist for configuration inputs.

Decimal text and rational conversion factors are combined before one float
conversion. Bare Python floats retain their original binary value. Nonzero
underflow is rejected: a tiny diffusion coefficient must not become a zero face.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
import re
from types import MappingProxyType

__all__ = ["UNIT_SCHEMA_VERSION", "QuantityError", "normalize_quantity", "supported_units"]

UNIT_SCHEMA_VERSION = "si-ev-1"


class QuantityError(ValueError):
    """A quantity cannot be represented in its declared canonical unit."""


@dataclass(frozen=True, slots=True)
class _Unit:
    canonical: str
    factor: Fraction


# These are unit definitions, not physical/model defaults. Energy remains eV.
_GROUPS = {
    "1": {"1": "1"},
    "m": {"m": "1", "cm": "1e-2", "mm": "1e-3", "um": "1e-6", "µm": "1e-6", "nm": "1e-9"},
    "s": {"s": "1", "ms": "1e-3", "us": "1e-6", "µs": "1e-6", "ns": "1e-9", "ps": "1e-12"},
    "m^-1": {"m^-1": "1", "cm^-1": "1e2"},
    "m^-2": {"m^-2": "1", "cm^-2": "1e4"},
    "m^-3": {"m^-3": "1", "cm^-3": "1e6"},
    "m^2": {"m^2": "1", "cm^2": "1e-4"},
    "m^2/(V s)": {"m^2/(V s)": "1", "m^2/Vs": "1", "cm^2/(V s)": "1e-4", "cm^2/Vs": "1e-4"},
    "m^2/s": {"m^2/s": "1", "cm^2/s": "1e-4"},
    "m^3/s": {"m^3/s": "1", "cm^3/s": "1e-6"},
    "m^6/s": {"m^6/s": "1", "cm^6/s": "1e-12"},
    "m/s": {"m/s": "1", "cm/s": "1e-2"},
    "K": {"K": "1"},
    "V": {"V": "1", "mV": "1e-3"},
    "eV": {"eV": "1", "meV": "1e-3"},
    "A/m^2": {"A/m^2": "1", "mA/cm^2": "10"},
    "ohm m^2": {"ohm m^2": "1", "ohm cm^2": "1e-4", "Ω m^2": "1", "Ω cm^2": "1e-4"},
}
_unit_values = {
    name: _Unit(canonical, Fraction(factor))
    for canonical, entries in _GROUPS.items()
    for name, factor in entries.items()
}
# SI's exact elementary-charge definition; no binary q multiplication/division.
_unit_values["J"] = _Unit("eV", 1 / Fraction("1.602176634e-19"))
_UNITS = MappingProxyType(_unit_values)
_CANONICAL = frozenset(_GROUPS)
del _unit_values

_QUANTITY = re.compile(
    r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*\Z"
)


def _fraction(value: object, path: str) -> Fraction:
    if type(value) is int:
        if value.bit_length() > 4096:
            raise QuantityError(f"{path}: quantity is outside finite float range")
        return Fraction(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise QuantityError(f"{path}: quantity must be finite")
        return Fraction.from_float(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise QuantityError(f"{path}: quantity must be finite")
        if value.is_zero():
            return Fraction(0)
        # Every whitelist factor is within 10**±28. This bound rejects only
        # unrepresentable results and avoids allocating enormous powers of ten.
        if abs(value.adjusted()) > 1024:
            raise QuantityError(f"{path}: quantity is outside finite float range")
        return Fraction(value)
    raise QuantityError(f"{path}: expected a number or unit string, not {type(value).__name__}")


def supported_units(canonical_unit: str) -> tuple[str, ...]:
    """Return the actual accepted spellings for one canonical unit."""
    if not isinstance(canonical_unit, str) or canonical_unit not in _CANONICAL:
        raise QuantityError(f"unknown canonical unit: {canonical_unit!r}")
    return tuple(sorted(name for name, unit in _UNITS.items() if unit.canonical == canonical_unit))


def normalize_quantity(
    value: object, canonical_unit: str, *, path: str = "value",
) -> float:
    """Resolve one finite input to SI (or eV), rejecting bools and wrong units.

    Signed zeros normalize to +0. Affine Celsius conversion and unlisted unit
    expressions are deliberately unsupported; callers must provide Kelvin.
    """
    supported_units(canonical_unit)
    source_unit = canonical_unit
    if isinstance(value, str):
        match = _QUANTITY.fullmatch(value)
        if match is None:
            raise QuantityError(f"{path}: invalid finite quantity {value!r}")
        text, suffix = match.groups()
        source_unit = suffix or canonical_unit
        try:
            value = Decimal(text)
        except InvalidOperation as error:
            raise QuantityError(f"{path}: invalid number") from error
    definition = _UNITS.get(source_unit)
    if definition is None or definition.canonical != canonical_unit:
        raise QuantityError(f"{path}: unit {source_unit!r} is not compatible with {canonical_unit!r}")
    exact = _fraction(value, path) * definition.factor
    try:
        result = float(exact)
    except OverflowError as error:
        raise QuantityError(f"{path}: quantity is outside finite float range") from error
    if not math.isfinite(result):
        raise QuantityError(f"{path}: quantity must be finite")
    if exact and result == 0.0:
        raise QuantityError(f"{path}: nonzero quantity underflows the resolved float representation")
    return result

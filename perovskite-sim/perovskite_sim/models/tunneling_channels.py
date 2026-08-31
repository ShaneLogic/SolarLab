"""Canonical contract for the D8 SCAPS-like WKB tunnelling family.

SolarLab already carries one tunnelling artefact: the static Padovani-Stratton
factor in ``physics/tunneling.py``. It is a single dimensionless number per
(face, carrier) folded into the Richardson constant — no transmission
spectrum, no direction, no trap coupling, no contact reach — and the D8 exit
condition explicitly forbids using such a scalar to claim parity for four
physically distinct channels. This module is therefore a NEW contract rather
than an extension of that one, and nothing here rescales ``A*``.

Four independent channels
-------------------------
``band_to_band``
    Valence-to-conduction (Zener) tunnelling across the gap under a field.
``intraband``
    Tunnelling through a band-edge spike at a heterointerface, staying in one
    band.
``interface_defect_assisted``
    Two-step tunnelling via an interface trap. It consumes the SAME interface
    occupancy the rest of the solver uses and may only be enabled where that
    occupancy is an explicit variable.
``contact``
    Field emission through a Schottky barrier at an outer contact.

Each carries its own enable flag, its own effective mass, its own energy
quadrature order and its own declared units, so no channel can be switched on
implicitly by another, and a frozen comparison can toggle them one at a time.

Everything defaults OFF. A document with every channel disabled is inert by
construction, which keeps the contract addable to any preset without moving a
single shipped number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Literal, Self


TUNNELLING_CHANNEL_SCHEMA_VERSION = "solarlab-wkb-tunnelling-channels-v1"

BAND_TO_BAND = "band_to_band"
INTRABAND = "intraband"
INTERFACE_DEFECT_ASSISTED = "interface_defect_assisted"
CONTACT = "contact"

CHANNEL_NAMES = (
    BAND_TO_BAND,
    INTRABAND,
    INTERFACE_DEFECT_ASSISTED,
    CONTACT,
)

ELECTRON = "electron"
HOLE = "hole"
BOTH_CARRIERS = "both"
_CARRIERS = {ELECTRON, HOLE, BOTH_CARRIERS}

LEFT = "left"
RIGHT = "right"
BOTH_SIDES = "both"
_SIDES = {LEFT, RIGHT, BOTH_SIDES}


class TunnellingChannelSchemaError(ValueError):
    """A tunnelling channel document violated the canonical contract."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise TunnellingChannelSchemaError(f"{field} must be finite")
    return result


def _positive(value: object, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise TunnellingChannelSchemaError(f"{field} must be positive")
    return result


def _nonnegative(value: object, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise TunnellingChannelSchemaError(f"{field} must be non-negative")
    return result


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise TunnellingChannelSchemaError(f"{field} must be >= {minimum}")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise TunnellingChannelSchemaError(
            f"{where} schema mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class BandToBandTunnellingChannel:
    """Zener tunnelling across the gap. Units: mass relative, energies eV."""

    enabled: bool = False
    reduced_effective_mass_rel: float = 0.1
    energy_quadrature_order: int = 32
    minimum_field_V_m: float = 1.0e6

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        object.__setattr__(
            self,
            "reduced_effective_mass_rel",
            _positive(self.reduced_effective_mass_rel, "reduced_effective_mass_rel"),
        )
        object.__setattr__(
            self,
            "energy_quadrature_order",
            _integer(
                self.energy_quadrature_order, "energy_quadrature_order", minimum=4
            ),
        )
        object.__setattr__(
            self,
            "minimum_field_V_m",
            _positive(self.minimum_field_V_m, "minimum_field_V_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reduced_effective_mass_rel": self.reduced_effective_mass_rel,
            "energy_quadrature_order": self.energy_quadrature_order,
            "minimum_field_V_m": self.minimum_field_V_m,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            value,
            {
                "enabled",
                "reduced_effective_mass_rel",
                "energy_quadrature_order",
                "minimum_field_V_m",
            },
            "band_to_band channel",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class IntrabandTunnellingChannel:
    """Tunnelling through a band spike, one band. Units: mass relative."""

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    carrier: Literal["electron", "hole", "both"] = BOTH_CARRIERS
    energy_quadrature_order: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field in (
            "electron_effective_mass_rel",
            "hole_effective_mass_rel",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        carrier = str(self.carrier).strip().lower()
        if carrier not in _CARRIERS:
            raise TunnellingChannelSchemaError(f"unknown carrier {carrier!r}")
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(
            self,
            "energy_quadrature_order",
            _integer(
                self.energy_quadrature_order, "energy_quadrature_order", minimum=4
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "electron_effective_mass_rel": self.electron_effective_mass_rel,
            "hole_effective_mass_rel": self.hole_effective_mass_rel,
            "carrier": self.carrier,
            "energy_quadrature_order": self.energy_quadrature_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            value,
            {
                "enabled",
                "electron_effective_mass_rel",
                "hole_effective_mass_rel",
                "carrier",
                "energy_quadrature_order",
            },
            "intraband channel",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class InterfaceDefectAssistedTunnellingChannel:
    """Trap-assisted tunnelling at an interface state.

    ``requires_explicit_occupancy`` is not a preference: the roadmap only
    permits this channel where the interface occupancy is an explicit solver
    variable, so the flag exists to make a consumer that cannot supply one
    fail closed rather than invent an occupancy.
    """

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    requires_explicit_occupancy: Literal[True] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field in (
            "electron_effective_mass_rel",
            "hole_effective_mass_rel",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.requires_explicit_occupancy is not True:
            raise TunnellingChannelSchemaError(
                "interface-defect-assisted tunnelling may not run against an "
                "algebraically eliminated occupancy"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "electron_effective_mass_rel": self.electron_effective_mass_rel,
            "hole_effective_mass_rel": self.hole_effective_mass_rel,
            "requires_explicit_occupancy": self.requires_explicit_occupancy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            value,
            {
                "enabled",
                "electron_effective_mass_rel",
                "hole_effective_mass_rel",
                "requires_explicit_occupancy",
            },
            "interface_defect_assisted channel",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ContactTunnellingChannel:
    """Field emission through a Schottky barrier at an outer contact.

    ``barrier_height_eV`` is the zero-field barrier the carrier sees from the
    metal; ``side`` selects which contact(s) the channel acts on.
    """

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    barrier_height_eV: float = 0.0
    side: Literal["left", "right", "both"] = BOTH_SIDES
    energy_quadrature_order: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field in (
            "electron_effective_mass_rel",
            "hole_effective_mass_rel",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        object.__setattr__(
            self,
            "barrier_height_eV",
            _nonnegative(self.barrier_height_eV, "barrier_height_eV"),
        )
        side = str(self.side).strip().lower()
        if side not in _SIDES:
            raise TunnellingChannelSchemaError(f"unknown contact side {side!r}")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "energy_quadrature_order",
            _integer(
                self.energy_quadrature_order, "energy_quadrature_order", minimum=4
            ),
        )
        if self.enabled and self.barrier_height_eV <= 0.0:
            raise TunnellingChannelSchemaError(
                "an enabled contact tunnelling channel needs a positive "
                "barrier_height_eV; a zero barrier is an ohmic contact"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "electron_effective_mass_rel": self.electron_effective_mass_rel,
            "hole_effective_mass_rel": self.hole_effective_mass_rel,
            "barrier_height_eV": self.barrier_height_eV,
            "side": self.side,
            "energy_quadrature_order": self.energy_quadrature_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            value,
            {
                "enabled",
                "electron_effective_mass_rel",
                "hole_effective_mass_rel",
                "barrier_height_eV",
                "side",
                "energy_quadrature_order",
            },
            "contact channel",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class TunnellingChannelDocument:
    """The four channels, each independently switchable."""

    schema_version: Literal["solarlab-wkb-tunnelling-channels-v1"] = (
        TUNNELLING_CHANNEL_SCHEMA_VERSION
    )
    band_to_band: BandToBandTunnellingChannel = None  # type: ignore[assignment]
    intraband: IntrabandTunnellingChannel = None  # type: ignore[assignment]
    interface_defect_assisted: InterfaceDefectAssistedTunnellingChannel = None  # type: ignore[assignment]
    contact: ContactTunnellingChannel = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.schema_version != TUNNELLING_CHANNEL_SCHEMA_VERSION:
            raise TunnellingChannelSchemaError(
                "unsupported tunnelling channel schema_version"
            )
        defaults = {
            "band_to_band": BandToBandTunnellingChannel,
            "intraband": IntrabandTunnellingChannel,
            "interface_defect_assisted": InterfaceDefectAssistedTunnellingChannel,
            "contact": ContactTunnellingChannel,
        }
        for field, factory in defaults.items():
            value = getattr(self, field)
            if value is None:
                object.__setattr__(self, field, factory())
            elif not isinstance(value, factory):
                raise TypeError(f"{field} must be a {factory.__name__}")

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        return tuple(name for name in CHANNEL_NAMES if getattr(self, name).enabled)

    @property
    def any_enabled(self) -> bool:
        return bool(self.enabled_channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **{name: getattr(self, name).to_dict() for name in CHANNEL_NAMES},
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise TunnellingChannelSchemaError(
                "tunnelling channel document must be a mapping"
            )
        _require_exact_keys(
            value,
            {"schema_version", *CHANNEL_NAMES},
            "tunnelling channel document",
        )
        return cls(
            schema_version=value["schema_version"],
            band_to_band=BandToBandTunnellingChannel.from_dict(value[BAND_TO_BAND]),
            intraband=IntrabandTunnellingChannel.from_dict(value[INTRABAND]),
            interface_defect_assisted=(
                InterfaceDefectAssistedTunnellingChannel.from_dict(
                    value[INTERFACE_DEFECT_ASSISTED]
                )
            ),
            contact=ContactTunnellingChannel.from_dict(value[CONTACT]),
        )


def tunnelling_channel_document_from_mapping(
    device: Mapping[str, Any],
) -> TunnellingChannelDocument | None:
    """Parse the optional ``tunnelling_channels`` block of a device mapping."""

    if not isinstance(device, Mapping):
        raise TunnellingChannelSchemaError("device configuration must be a mapping")
    raw = device.get("tunnelling_channels")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TunnellingChannelSchemaError("tunnelling_channels must be a mapping")
    payload: dict[str, Any] = {
        "schema_version": raw.get("schema_version", TUNNELLING_CHANNEL_SCHEMA_VERSION)
    }
    for name in CHANNEL_NAMES:
        channel = raw.get(name)
        if channel is None:
            payload[name] = {
                BAND_TO_BAND: BandToBandTunnellingChannel,
                INTRABAND: IntrabandTunnellingChannel,
                INTERFACE_DEFECT_ASSISTED: (InterfaceDefectAssistedTunnellingChannel),
                CONTACT: ContactTunnellingChannel,
            }[name]().to_dict()
        elif isinstance(channel, Mapping):
            payload[name] = _coerce_scalars(channel)
        else:
            raise TunnellingChannelSchemaError(
                f"tunnelling channel {name!r} must be a mapping"
            )
    return TunnellingChannelDocument.from_dict(payload)


def _coerce_scalars(value: Any) -> Any:
    """Normalize YAML 1.1 numeric scalars at the text boundary.

    Same reason as the v4 defect parser: PyYAML leaves ``1.0e6`` (unsigned
    exponent) as a string, and every other layer in this repository is written
    that way, so the coercion belongs at the boundary rather than forcing a
    different numeric style on tunnelling blocks.
    """

    if isinstance(value, Mapping):
        return {key: _coerce_scalars(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_scalars(item) for item in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


__all__ = [
    "BAND_TO_BAND",
    "BOTH_CARRIERS",
    "BOTH_SIDES",
    "CHANNEL_NAMES",
    "CONTACT",
    "ELECTRON",
    "HOLE",
    "INTERFACE_DEFECT_ASSISTED",
    "INTRABAND",
    "LEFT",
    "RIGHT",
    "TUNNELLING_CHANNEL_SCHEMA_VERSION",
    "BandToBandTunnellingChannel",
    "ContactTunnellingChannel",
    "InterfaceDefectAssistedTunnellingChannel",
    "IntrabandTunnellingChannel",
    "TunnellingChannelDocument",
    "TunnellingChannelSchemaError",
    "tunnelling_channel_document_from_mapping",
]

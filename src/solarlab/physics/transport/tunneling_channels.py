"""Canonical 1D WKB tunnelling-channel contracts, evaluators, and path Jacobians.

Migrates the four named WKB tunnelling channels under explicit quasi-static
capability gates:

1. ``band_to_band`` — Zener/Esaki interband two-band Kane tunnelling;
2. ``intraband`` — heterointerface band-spike path-resolved tunnelling;
3. ``interface_defect_assisted`` — trap-assisted tunnelling consuming an explicit
   interface occupancy ``f_t`` (refusing algebraically eliminated steady SRH);
4. ``contact`` — Schottky contact field emission.

For device steady-state coupling, ``intraband_path_flux_and_jacobian`` computes
both the conservative nonlocal face-flux profile ``particle_face_flux_m2_s`` and
its exact analytical Jacobians wrt ``band_edge_eV`` and ``quasi_fermi_eV``, plus
a ``WKBWindowSupport`` descriptor tracking active turning-point face intervals
``(a_k, b_k)`` against the preallocated reservoir superset or strict window
rebind signature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.physics.transport.wkb_tunneling import (
    MINIMUM_MEANINGFUL_ACTION,
    WKBTunnellingError,
    forbidden_run,
    piecewise_linear_wkb_segment_action_derivatives,
    piecewise_linear_wkb_segment_actions,
    reciprocal_net_flux,
    turning_point_levels,
    two_band_transmission,
    two_band_turning_point_levels,
    two_band_validity,
    windowed_wkb_transmission,
    wkb_validity,
)

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
    "PREALLOCATED_RESERVOIR_SUPERSET_V1",
    "RIGHT",
    "STRICT_WINDOW_REBIND_V1",
    "TUNNELLING_CHANNEL_DEVICE_VERSION",
    "TUNNELLING_CHANNEL_SCHEMA_VERSION",
    "WKB_SPARSITY_POLICIES",
    "BandToBandTunnellingChannel",
    "ChannelFlux",
    "CompiledTunnellingChannels",
    "ContactTunnellingChannel",
    "InterfaceDefectAssistedFlux",
    "InterfaceDefectAssistedTunnellingChannel",
    "IntrabandBarrierRegion",
    "IntrabandPathFlux",
    "IntrabandPathLinearization",
    "IntrabandTunnellingChannel",
    "TunnellingChannelCapabilityError",
    "TunnellingChannelDocument",
    "TunnellingChannelError",
    "TunnellingChannelEvaluation",
    "TunnellingChannelSchemaError",
    "WKBWindowSupport",
    "band_to_band_flux",
    "compile_tunnelling_channels",
    "conduction_band_eV",
    "contact_tunnelling_flux",
    "evaluate_tunnelling_channels",
    "interface_defect_assisted_rate",
    "intraband_flux",
    "intraband_path_flux",
    "intraband_path_flux_and_jacobian",
    "local_barrier_window",
    "physical_quasi_fermi_levels_eV",
    "tunnelling_channel_document_from_mapping",
    "valence_band_eV",
]

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]
Q: float = ELEMENTARY_CHARGE_C

TUNNELLING_CHANNEL_SCHEMA_VERSION = "solarlab-wkb-tunnelling-channels-v1"
TUNNELLING_CHANNEL_DEVICE_VERSION = "wkb-tunnelling-channel-device-v2"

BAND_TO_BAND = "band_to_band"
INTRABAND = "intraband"
INTERFACE_DEFECT_ASSISTED = "interface_defect_assisted"
CONTACT = "contact"
CHANNEL_NAMES = (BAND_TO_BAND, INTRABAND, INTERFACE_DEFECT_ASSISTED, CONTACT)

ELECTRON = "electron"
HOLE = "hole"
BOTH_CARRIERS = "both"
_CARRIERS = {ELECTRON, HOLE, BOTH_CARRIERS}

LEFT = "left"
RIGHT = "right"
BOTH_SIDES = "both"
_SIDES = {LEFT, RIGHT, BOTH_SIDES}

PREALLOCATED_RESERVOIR_SUPERSET_V1 = "preallocated_reservoir_superset_v1"
STRICT_WINDOW_REBIND_V1 = "strict_window_rebind_v1"
WKB_SPARSITY_POLICIES = (
    PREALLOCATED_RESERVOIR_SUPERSET_V1,
    STRICT_WINDOW_REBIND_V1,
)


class TunnellingChannelSchemaError(ValueError):
    """A tunnelling channel document violated the canonical contract."""


class TunnellingChannelError(RuntimeError):
    """A channel was asked for a flux outside its declared domain."""


class TunnellingChannelCapabilityError(RuntimeError):
    """A tunnelling channel was requested where it is not certified."""


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
    """Zener tunnelling across the gap."""

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
            _integer(self.energy_quadrature_order, "energy_quadrature_order", minimum=4),
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
    """Tunnelling through a heterointerface band spike."""

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    carrier: str = BOTH_CARRIERS
    energy_quadrature_order: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field_name in ("electron_effective_mass_rel", "hole_effective_mass_rel"):
            object.__setattr__(
                self, field_name, _positive(getattr(self, field_name), field_name)
            )
        carrier = str(self.carrier).strip().lower()
        if carrier not in _CARRIERS:
            raise TunnellingChannelSchemaError(f"unknown carrier {carrier!r}")
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(
            self,
            "energy_quadrature_order",
            _integer(self.energy_quadrature_order, "energy_quadrature_order", minimum=4),
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
    """Trap-assisted tunnelling requiring an explicit interface occupancy."""

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    requires_explicit_occupancy: Literal[True] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field_name in ("electron_effective_mass_rel", "hole_effective_mass_rel"):
            object.__setattr__(
                self, field_name, _positive(getattr(self, field_name), field_name)
            )
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
    """Field emission through a Schottky barrier at an outer contact."""

    enabled: bool = False
    electron_effective_mass_rel: float = 0.2
    hole_effective_mass_rel: float = 0.2
    barrier_height_eV: float = 0.0
    side: str = BOTH_SIDES
    energy_quadrature_order: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean(self.enabled, "enabled"))
        for field_name in ("electron_effective_mass_rel", "hole_effective_mass_rel"):
            object.__setattr__(
                self, field_name, _positive(getattr(self, field_name), field_name)
            )
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
            _integer(self.energy_quadrature_order, "energy_quadrature_order", minimum=4),
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
    """The four named WKB channels, each independently switchable."""

    schema_version: str = (
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
        defaults: dict[str, type] = {
            "band_to_band": BandToBandTunnellingChannel,
            "intraband": IntrabandTunnellingChannel,
            "interface_defect_assisted": InterfaceDefectAssistedTunnellingChannel,
            "contact": ContactTunnellingChannel,
        }
        for field_name, factory in defaults.items():
            value = getattr(self, field_name)
            if value is None:
                object.__setattr__(self, field_name, factory())
            elif not isinstance(value, factory):
                raise TypeError(f"{field_name} must be a {factory.__name__}")

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
            interface_defect_assisted=InterfaceDefectAssistedTunnellingChannel.from_dict(
                value[INTERFACE_DEFECT_ASSISTED]
            ),
            contact=ContactTunnellingChannel.from_dict(value[CONTACT]),
        )


def _coerce_scalars(value: Any) -> Any:
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


def tunnelling_channel_document_from_mapping(
    device: Mapping[str, Any],
) -> TunnellingChannelDocument | None:
    """Parse optional ``tunnelling_channels`` block (or direct channel mapping)."""
    if not isinstance(device, Mapping):
        raise TunnellingChannelSchemaError("device configuration must be a mapping")
    if "tunnelling_channels" in device:
        raw = device.get("tunnelling_channels")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TunnellingChannelSchemaError("tunnelling_channels must be a mapping")
    elif "schema_version" in device or any(name in device for name in CHANNEL_NAMES):
        raw = device
    else:
        return None

    defaults_map: dict[str, dict[str, Any]] = {
        BAND_TO_BAND: BandToBandTunnellingChannel().to_dict(),
        INTRABAND: IntrabandTunnellingChannel().to_dict(),
        INTERFACE_DEFECT_ASSISTED: InterfaceDefectAssistedTunnellingChannel().to_dict(),
        CONTACT: ContactTunnellingChannel().to_dict(),
    }
    payload: dict[str, Any] = {
        "schema_version": raw.get("schema_version", TUNNELLING_CHANNEL_SCHEMA_VERSION)
    }
    extra_top = set(raw.keys()) - {"schema_version", *CHANNEL_NAMES}
    if extra_top:
        raise TunnellingChannelSchemaError(
            f"tunnelling channel document unknown keys: {sorted(extra_top)}"
        )
    for name in CHANNEL_NAMES:
        channel = raw.get(name)
        if channel is None:
            payload[name] = dict(defaults_map[name])
        elif isinstance(channel, Mapping):
            coerced = _coerce_scalars(channel)
            unknown_sub = set(coerced.keys()) - set(defaults_map[name].keys())
            if unknown_sub:
                raise TunnellingChannelSchemaError(
                    f"{name} channel unknown keys: {sorted(unknown_sub)}"
                )
            payload[name] = {**defaults_map[name], **coerced}
        else:
            raise TunnellingChannelSchemaError(
                f"tunnelling channel {name!r} must be a mapping"
            )
    return TunnellingChannelDocument.from_dict(payload)


def _fermi(energy_eV: ArrayLike, level_eV: ArrayLike | float, thermal_voltage_V: float) -> Vector:
    reduced = (
        np.asarray(energy_eV, dtype=float) - np.asarray(level_eV, dtype=float)
    ) / float(thermal_voltage_V)
    out = np.empty_like(reduced, dtype=float)
    positive = reduced >= 0.0
    out[positive] = np.exp(-reduced[positive]) / (1.0 + np.exp(-reduced[positive]))
    out[~positive] = 1.0 / (1.0 + np.exp(reduced[~positive]))
    return out


def conduction_band_eV(potential_V: ArrayLike, affinity_eV: ArrayLike) -> Vector:
    """``E_C = -(phi + chi)`` [eV]."""
    return -(np.asarray(potential_V, dtype=float) + np.asarray(affinity_eV, dtype=float))


def valence_band_eV(
    potential_V: ArrayLike,
    affinity_eV: ArrayLike,
    band_gap_eV: ArrayLike,
) -> Vector:
    """``E_V = -(phi + chi + Eg)`` [eV]."""
    return -(
        np.asarray(potential_V, dtype=float)
        + np.asarray(affinity_eV, dtype=float)
        + np.asarray(band_gap_eV, dtype=float)
    )


def local_barrier_window(
    barrier_eV: ArrayLike,
    anchor_face: int,
    *,
    one_sided: bool = False,
) -> tuple[float, float]:
    """Peak and base of the barrier feature anchored on ``anchor_face``."""
    barrier = np.asarray(barrier_eV, dtype=float)
    face = int(anchor_face)
    if face < 0 or face >= barrier.size - 1:
        raise WKBTunnellingError("anchor_face is outside the transport faces")
    peak_index = face if barrier[face] >= barrier[face + 1] else face + 1
    while peak_index + 1 < barrier.size and barrier[peak_index + 1] > barrier[peak_index]:
        peak_index += 1
    while peak_index - 1 >= 0 and barrier[peak_index - 1] > barrier[peak_index]:
        peak_index -= 1
    left = peak_index
    while left - 1 >= 0 and barrier[left - 1] < barrier[left]:
        left -= 1
    right = peak_index
    while right + 1 < barrier.size and barrier[right + 1] < barrier[right]:
        right += 1
    peak = float(barrier[peak_index])
    if not one_sided:
        return peak, float(max(barrier[left], barrier[right]))
    return peak, float(min(barrier[left], barrier[right]))


def _turning_point_levels(
    barrier_eV: Vector,
    energies_eV: Vector,
    anchor_face: int,
    level_profile_eV: ArrayLike,
) -> tuple[Vector, Vector]:
    profile = np.asarray(level_profile_eV, dtype=float)
    left = np.empty_like(energies_eV)
    right = np.empty_like(energies_eV)
    for index, energy in enumerate(energies_eV):
        left[index], right[index] = turning_point_levels(
            barrier_eV, profile, float(energy), anchor_face
        )
    return left, right


def _two_band_turning_point_levels(
    conduction_eV: Vector,
    valence_eV: Vector,
    energies_eV: Vector,
    anchor_face: int,
    level_profile_eV: ArrayLike,
) -> tuple[Vector, Vector]:
    profile = np.asarray(level_profile_eV, dtype=float)
    left = np.empty_like(energies_eV)
    right = np.empty_like(energies_eV)
    for index, energy in enumerate(energies_eV):
        left[index], right[index] = two_band_turning_point_levels(
            conduction_eV, valence_eV, profile, float(energy), anchor_face
        )
    return left, right


@dataclass(frozen=True, slots=True)
class ChannelFlux:
    """One channel's net flux plus everything needed to audit it."""

    channel: str
    net_flux_m2_s: float
    forward_flux_m2_s: float
    reverse_flux_m2_s: float
    energies_eV: Vector
    transmission: Vector
    maximum_transmission: float
    minimum_transmission: float
    maximum_action: float
    valid: bool
    notes: tuple[str, ...]
    spectral_flux_m2_s_eV: Vector | None = None
    left_occupation: Vector | None = None
    right_occupation: Vector | None = None
    quadrature_weights_eV: Vector | None = None


def _channel_flux(
    channel: str,
    energies: Vector,
    transmission: Vector,
    left: Vector,
    right: Vector,
    prefactor: float,
    *,
    valid: bool,
    notes: tuple[str, ...],
    quadrature_weights_eV: Vector | None = None,
    actions: Vector | None = None,
) -> ChannelFlux:
    flux = reciprocal_net_flux(
        energies,
        transmission,
        left,
        right,
        prefactor,
        quadrature_weights_eV=quadrature_weights_eV,
    )
    least = float(np.min(flux.transmission))
    return ChannelFlux(
        channel=channel,
        net_flux_m2_s=flux.net_flux_m2_s,
        forward_flux_m2_s=flux.forward_flux_m2_s,
        reverse_flux_m2_s=flux.reverse_flux_m2_s,
        energies_eV=flux.energies_eV,
        transmission=flux.transmission,
        maximum_transmission=float(np.max(flux.transmission)),
        minimum_transmission=least,
        maximum_action=(
            float(np.max(actions))
            if actions is not None
            else float("inf") if least <= 0.0 else -0.5 * math.log(least)
        ),
        valid=valid,
        notes=notes,
        spectral_flux_m2_s_eV=flux.spectral_flux,
        left_occupation=flux.left_occupation,
        right_occupation=flux.right_occupation,
        quadrature_weights_eV=(
            frozen_array(quadrature_weights_eV) if quadrature_weights_eV is not None else None
        ),
    )


@dataclass(frozen=True, slots=True)
class IntrabandBarrierRegion:
    """One material barrier and its two adjacent reservoir node ranges."""

    identity: str
    left_start_node: int
    core_start_node: int
    core_stop_node: int
    right_stop_node: int
    core_left_m: float
    core_right_m: float

    def __post_init__(self) -> None:
        if not self.identity:
            raise TunnellingChannelError("barrier identity must be non-empty")
        if not (
            0
            <= self.left_start_node
            < self.core_start_node
            < self.core_stop_node
            < self.right_stop_node
        ):
            raise TunnellingChannelError(
                "barrier and reservoir node ranges must be ordered and non-empty"
            )
        if (
            not np.isfinite(self.core_left_m)
            or not np.isfinite(self.core_right_m)
            or self.core_left_m >= self.core_right_m
        ):
            raise TunnellingChannelError(
                "physical barrier boundaries must be finite and ordered"
            )


@dataclass(frozen=True, slots=True)
class WKBWindowSupport:
    """Active turning-point window vs preallocated reservoir superset."""

    reservoir_face_slice: tuple[int, int]
    reservoir_node_slice: tuple[int, int]
    turning_point_face_pairs: tuple[tuple[int, int], ...]
    active_face_bounds: tuple[int, int]
    active_node_bounds: tuple[int, int]
    contained_in_reservoir_superset: bool
    support_signature: str


@dataclass(frozen=True, slots=True)
class IntrabandPathFlux:
    """Path-resolved intraband barrier flux and turning-point support."""

    region: IntrabandBarrierRegion
    flux: ChannelFlux
    particle_face_flux_m2_s: Vector
    turning_points_m: Matrix
    actions: Vector
    low_action_supply_fraction: float | None
    support: WKBWindowSupport | None = None


@dataclass(frozen=True, slots=True)
class IntrabandPathLinearization:
    """Exact path flux and analytical Jacobians wrt ``band_edge_eV`` and ``quasi_fermi_eV``."""

    path: IntrabandPathFlux
    d_face_flux_d_barrier_eV: Matrix
    d_face_flux_d_quasi_fermi_eV: Matrix
    support: WKBWindowSupport


@lru_cache(maxsize=32)
def _legendre_energy_rule(order: int) -> tuple[Vector, Vector]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return frozen_array(nodes), frozen_array(weights)


def _build_window_support(
    region: IntrabandBarrierRegion,
    pairs: Sequence[tuple[int, int]],
    base_node: int,
    peak_node: int,
) -> WKBWindowSupport:
    res_faces = (region.left_start_node, region.right_stop_node - 1)
    res_nodes = (region.left_start_node, region.right_stop_node)
    tup_pairs = tuple((int(a), int(b)) for a, b in pairs)
    min_a = min(a for a, _ in tup_pairs)
    max_b = max(b for _, b in tup_pairs)
    active_faces = (min_a, max_b + 1)
    min_node = min(min_a, int(base_node), int(peak_node))
    max_node = max(max_b + 1, int(base_node), int(peak_node)) + 1
    active_nodes = (min_node, max_node)
    contained = (
        active_faces[0] >= res_faces[0]
        and active_faces[1] <= res_faces[1]
        and active_nodes[0] >= res_nodes[0]
        and active_nodes[1] <= res_nodes[1]
    )
    sig_payload = {
        "reservoir_faces": list(res_faces),
        "active_faces": list(active_faces),
        "active_nodes": list(active_nodes),
        "pairs": [list(p) for p in tup_pairs],
    }
    sig = hashlib.sha256(
        json.dumps(sig_payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return WKBWindowSupport(
        reservoir_face_slice=res_faces,
        reservoir_node_slice=res_nodes,
        turning_point_face_pairs=tup_pairs,
        active_face_bounds=active_faces,
        active_node_bounds=active_nodes,
        contained_in_reservoir_superset=contained,
        support_signature=sig,
    )


def intraband_path_flux_and_jacobian(
    positions_m: ArrayLike,
    band_edge_eV: ArrayLike,
    channel: IntrabandTunnellingChannel,
    *,
    region: IntrabandBarrierRegion,
    quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
    compute_jacobian: bool = True,
) -> IntrabandPathLinearization:
    """Reciprocal electron transfer across one resolved material barrier + exact Jacobians."""
    if not channel.enabled or channel.carrier != ELECTRON:
        raise TunnellingChannelError(
            "resolved intraband device transport currently requires one electron channel"
        )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(band_edge_eV, dtype=float)
    levels = np.asarray(quasi_fermi_eV, dtype=float)
    if (
        x.ndim != 1
        or x.size < 3
        or np.any(np.diff(x) <= 0.0)
        or barrier.shape != x.shape
        or levels.shape != x.shape
        or not all(np.all(np.isfinite(value)) for value in (x, barrier, levels))
    ):
        raise TunnellingChannelError(
            "barrier, physical levels and positions must share a finite increasing grid"
        )
    if region.right_stop_node > x.size:
        raise TunnellingChannelError("barrier region is outside the electrical grid")
    geometry_tolerance = 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(x))), float(np.ptp(x))
    )
    if (
        abs(float(x[region.core_start_node]) - region.core_left_m) > geometry_tolerance
        or abs(float(x[region.core_stop_node]) - region.core_right_m) > geometry_tolerance
    ):
        raise TunnellingChannelError(
            "barrier identity requires its physical boundaries on the nodal grid"
        )
    vt = float(thermal_voltage_V)
    if not np.isfinite(vt) or vt <= 0.0:
        raise TunnellingChannelError("thermal voltage must be finite and positive")

    start, stop = region.core_start_node, region.core_stop_node
    peak_node = start + int(np.argmax(barrier[start:stop]))
    peak = float(barrier[peak_node])
    left_min_node = region.left_start_node + int(
        np.argmin(barrier[region.left_start_node : start])
    )
    right_min_node = stop + int(np.argmin(barrier[stop : region.right_stop_node]))
    if float(barrier[left_min_node]) >= float(barrier[right_min_node]):
        base_node = left_min_node
    else:
        base_node = right_min_node
    base = float(barrier[base_node])
    if not peak > base:
        raise TunnellingChannelError(
            "identified barrier has no sub-barrier energy window between its reservoirs"
        )

    nodes_xi, rule_weights = _legendre_energy_rule(channel.energy_quadrature_order)
    energies = base + (peak - base) * (nodes_xi + 1.0) / 2.0
    weights = (peak - base) * rule_weights / 2.0
    if np.any(energies >= peak) or np.any(energies <= base):
        raise TunnellingChannelError(
            "sub-barrier energy interval is below floating-point resolution"
        )

    bounds = np.empty((energies.size, 2), dtype=int)
    fractions = np.empty((energies.size, 2), dtype=float)
    turning_points = np.empty((energies.size, 2), dtype=float)
    left_levels = np.empty(energies.size, dtype=float)
    right_levels = np.empty(energies.size, dtype=float)
    pairs: list[tuple[int, int]] = []

    for k, energy in enumerate(energies):
        run = forbidden_run(barrier, float(energy), peak_node)
        if run is None:
            raise TunnellingChannelError(
                "turning points do not close inside the declared reservoirs"
            )
        low, high = run
        if low <= region.left_start_node or high + 1 >= region.right_stop_node:
            raise TunnellingChannelError(
                "turning points do not close inside the declared reservoirs"
            )
        core_forbidden: NDArray[np.intp] = np.flatnonzero(barrier[start:stop] > energy) + start
        if np.any((core_forbidden < low) | (core_forbidden > high)):
            raise TunnellingChannelError(
                "multiple disconnected barriers inside one declared material region"
            )
        a, b = low - 1, high
        left_fraction = (float(energy) - float(barrier[a])) / float(
            barrier[a + 1] - barrier[a]
        )
        right_fraction = (float(barrier[b]) - float(energy)) / float(
            barrier[b] - barrier[b + 1]
        )
        bounds[k] = a, b
        pairs.append((a, b))
        fractions[k] = left_fraction, right_fraction
        turning_points[k] = (
            float(x[a]) + left_fraction * float(x[a + 1] - x[a]),
            float(x[b]) + right_fraction * float(x[b + 1] - x[b]),
        )
        left_levels[k] = float(levels[a]) + left_fraction * float(
            levels[a + 1] - levels[a]
        )
        right_levels[k] = float(levels[b]) + right_fraction * float(
            levels[b + 1] - levels[b]
        )

    support = _build_window_support(region, pairs, base_node, peak_node)
    if not support.contained_in_reservoir_superset:
        raise TunnellingChannelError("active WKB turning-point support escaped reservoir bounds")

    if compute_jacobian:
        seg_derivs = piecewise_linear_wkb_segment_action_derivatives(
            x, barrier, energies, channel.electron_effective_mass_rel
        )
        segment_actions = seg_derivs.segment_actions
    else:
        seg_derivs = None
        segment_actions = piecewise_linear_wkb_segment_actions(
            x, barrier, energies, channel.electron_effective_mass_rel
        )

    faces = np.arange(x.size - 1)[None, :]
    active = (faces >= bounds[:, :1]) & (faces <= bounds[:, 1:])
    actions = np.sum(np.where(active, segment_actions, 0.0), axis=1)
    transmission = np.exp(-2.0 * actions)
    left = _fermi(energies, left_levels, vt)
    right = _fermi(energies, right_levels, vt)
    flux = _channel_flux(
        "intraband_electron",
        energies,
        transmission,
        left,
        right,
        supply_prefactor_m2_s_eV,
        valid=bool(np.max(actions) >= MINIMUM_MEANINGFUL_ACTION),
        notes=(
            "resolved_material_barrier",
            "boltzmann_reservoir_levels",
            "leading_wkb_exponent",
            "supply_prefactor_not_externally_validated",
        ),
        quadrature_weights_eV=weights,
        actions=actions,
    )
    assert flux.spectral_flux_m2_s_eV is not None
    path_weights = ((faces > bounds[:, :1]) & (faces < bounds[:, 1:])).astype(float)
    rows = np.arange(energies.size)
    path_weights[rows, bounds[:, 0]] = 1.0 - fractions[:, 0]
    path_weights[rows, bounds[:, 1]] = fractions[:, 1]
    particle_faces = (weights * flux.spectral_flux_m2_s_eV) @ path_weights

    supply = weights * transmission * (left + right)
    denominator = float(np.sum(supply))
    low_action_fraction = (
        float(np.sum(supply[actions < 1.0])) / denominator if denominator > 0.0 else None
    )

    n_nodes = x.size
    n_faces = n_nodes - 1
    d_barrier: Matrix = np.zeros((n_faces, n_nodes), dtype=float) if compute_jacobian else np.zeros((0, 0), dtype=float)
    d_qf: Matrix = np.zeros((n_faces, n_nodes), dtype=float) if compute_jacobian else np.zeros((0, 0), dtype=float)

    if compute_jacobian:
        assert seg_derivs is not None
        prefactor = float(supply_prefactor_m2_s_eV)
        for k, (a, b) in enumerate(pairs):
            fl = float(fractions[k, 0])
            fr = float(fractions[k, 1])
            xi_k = float(nodes_xi[k])
            w_hat_k = float(rule_weights[k])
            w_k = float(weights[k])
            t_k = float(transmission[k])
            f_l = float(left[k])
            f_r = float(right[k])
            delta_occ = f_l - f_r
            jk = w_k * prefactor * t_k * delta_occ

            # 1. Energy & weight derivatives wrt U via base_node and peak_node.
            de_du = np.zeros(n_nodes, dtype=float)
            dw_du = np.zeros(n_nodes, dtype=float)
            de_du[base_node] += 0.5 * (1.0 - xi_k)
            de_du[peak_node] += 0.5 * (1.0 + xi_k)
            dw_du[base_node] -= 0.5 * w_hat_k
            dw_du[peak_node] += 0.5 * w_hat_k

            # 2. Active segment action derivatives over faces [a .. b].
            ds_du = np.zeros(n_nodes, dtype=float)
            dl_seg = seg_derivs.d_action_d_left_barrier_eV[k, a : b + 1]
            dr_seg = seg_derivs.d_action_d_right_barrier_eV[k, a : b + 1]
            ds_du[a : b + 1] += dl_seg
            ds_du[a + 1 : b + 2] += dr_seg
            ds_de = -float(np.sum(dl_seg) + np.sum(dr_seg))
            ds_du += ds_de * de_du
            dt_du = -2.0 * t_k * ds_du

            # 3. Crossing-fraction & turning-point Fermi derivatives.
            du_l = float(barrier[a + 1] - barrier[a])
            dfl_du = np.zeros(n_nodes, dtype=float)
            dfl_du[a] += (fl - 1.0) / du_l
            dfl_du[a + 1] -= fl / du_l
            dfl_du += de_du / du_l

            du_r = float(barrier[b + 1] - barrier[b])
            dfr_du = np.zeros(n_nodes, dtype=float)
            dfr_du[b] += (fr - 1.0) / du_r
            dfr_du[b + 1] -= fr / du_r
            dfr_du += de_du / du_r

            dmu_l_du = float(levels[a + 1] - levels[a]) * dfl_du
            dmu_r_du = float(levels[b + 1] - levels[b]) * dfr_du

            # 4. Occupation derivatives df = f*(1-f)/V_T * (dmu - dE).
            df_l_slope = f_l * (1.0 - f_l) / vt
            df_r_slope = f_r * (1.0 - f_r) / vt

            ddelta_dqf = np.zeros(n_nodes, dtype=float)
            ddelta_dqf[a] += df_l_slope * (1.0 - fl)
            ddelta_dqf[a + 1] += df_l_slope * fl
            ddelta_dqf[b] -= df_r_slope * (1.0 - fr)
            ddelta_dqf[b + 1] -= df_r_slope * fr

            ddelta_du = df_l_slope * (dmu_l_du - de_du) - df_r_slope * (dmu_r_du - de_du)

            djk_dqf = w_k * prefactor * t_k * ddelta_dqf
            djk_du = prefactor * (
                dw_du * t_k * delta_occ
                + w_k * dt_du * delta_occ
                + w_k * t_k * ddelta_du
            )

            # 5. Path-weight scatter across faces a .. b (note b >= a + 1 always).
            d_qf[a, :] += (1.0 - fl) * djk_dqf
            d_barrier[a, :] += (1.0 - fl) * djk_du - dfl_du * jk
            if b > a + 1:
                d_qf[a + 1 : b, :] += djk_dqf[None, :]
                d_barrier[a + 1 : b, :] += djk_du[None, :]
            d_qf[b, :] += fr * djk_dqf
            d_barrier[b, :] += fr * djk_du + dfr_du * jk

    path = IntrabandPathFlux(
        region=region,
        flux=flux,
        particle_face_flux_m2_s=frozen_array(particle_faces),
        turning_points_m=frozen_array(turning_points),
        actions=frozen_array(actions),
        low_action_supply_fraction=low_action_fraction,
        support=support,
    )
    return IntrabandPathLinearization(
        path=path,
        d_face_flux_d_barrier_eV=frozen_array(d_barrier),
        d_face_flux_d_quasi_fermi_eV=frozen_array(d_qf),
        support=support,
    )


def intraband_path_flux(
    positions_m: ArrayLike,
    band_edge_eV: ArrayLike,
    channel: IntrabandTunnellingChannel,
    *,
    region: IntrabandBarrierRegion,
    quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> IntrabandPathFlux:
    """Reciprocal electron transfer across one resolved material barrier."""
    return intraband_path_flux_and_jacobian(
        positions_m,
        band_edge_eV,
        channel,
        region=region,
        quasi_fermi_eV=quasi_fermi_eV,
        thermal_voltage_V=thermal_voltage_V,
        supply_prefactor_m2_s_eV=supply_prefactor_m2_s_eV,
        compute_jacobian=False,
    ).path


def band_to_band_flux(
    positions_m: ArrayLike,
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    channel: BandToBandTunnellingChannel,
    *,
    anchor_face: int,
    quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Zener tunnelling from the valence band to the conduction band."""
    if not channel.enabled:
        raise TunnellingChannelError("band_to_band channel is disabled")
    x = np.asarray(positions_m, dtype=float)
    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    if conduction.shape != x.shape or valence.shape != x.shape:
        raise WKBTunnellingError("band edges must match positions_m")
    if np.any(conduction <= valence):
        raise TunnellingChannelError(
            "band-to-band tunnelling requires a positive gap everywhere"
        )
    span = float(x[-1] - x[0])
    if span <= 0.0:
        raise WKBTunnellingError("positions_m must span a positive distance")
    field_v_m = abs(float(conduction[-1] - conduction[0])) / span
    notes: list[str] = []
    if field_v_m < channel.minimum_field_V_m:
        notes.append("field_below_channel_minimum")

    lower = float(valence[anchor_face])
    upper = float(conduction[anchor_face])
    energies = np.linspace(lower, upper, channel.energy_quadrature_order + 2)[1:-1]
    transmission = np.array(
        [
            two_band_transmission(
                x,
                conduction,
                valence,
                float(energy),
                channel.reduced_effective_mass_rel,
                anchor_face,
            )
            for energy in energies
        ],
        dtype=float,
    )
    mid = float(energies[len(energies) // 2])
    validity = two_band_validity(
        x,
        conduction,
        valence,
        mid,
        channel.reduced_effective_mass_rel,
        anchor_face,
    )
    left_levels, right_levels = _two_band_turning_point_levels(
        conduction, valence, energies, anchor_face, quasi_fermi_eV
    )
    left = _fermi(energies, left_levels, thermal_voltage_V)
    right = _fermi(energies, right_levels, thermal_voltage_V)
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    return _channel_flux(
        "band_to_band",
        energies,
        transmission,
        left,
        right,
        supply_prefactor_m2_s_eV,
        valid=validity.valid and not notes,
        notes=tuple(notes),
    )


def intraband_flux(
    positions_m: ArrayLike,
    band_edge_eV: ArrayLike,
    channel: IntrabandTunnellingChannel,
    *,
    anchor_face: int,
    carrier: str,
    quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Tunnelling through a band-edge spike without changing band."""
    if not channel.enabled:
        raise TunnellingChannelError("intraband channel is disabled")
    if channel.carrier not in (carrier, "both"):
        raise TunnellingChannelError(
            f"intraband channel is configured for {channel.carrier!r}, not {carrier!r}"
        )
    mass = (
        channel.electron_effective_mass_rel
        if carrier == ELECTRON
        else channel.hole_effective_mass_rel
    )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(band_edge_eV, dtype=float)
    if barrier.shape != x.shape:
        raise WKBTunnellingError("band edge must match positions_m")
    peak, base = local_barrier_window(barrier, anchor_face)
    notes: list[str] = []
    if not peak > base:
        raise TunnellingChannelError(
            "intraband tunnelling requires a local barrier spike at this face"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [
            windowed_wkb_transmission(x, barrier, float(energy), mass, anchor_face)
            for energy in energies
        ],
        dtype=float,
    )
    validity = wkb_validity(x, barrier, float(energies[0]), mass)
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    left_levels, right_levels = _turning_point_levels(
        barrier, energies, anchor_face, quasi_fermi_eV
    )
    left = _fermi(energies, left_levels, thermal_voltage_V)
    right = _fermi(energies, right_levels, thermal_voltage_V)
    return _channel_flux(
        f"intraband_{carrier}",
        energies,
        transmission,
        left,
        right,
        supply_prefactor_m2_s_eV,
        valid=validity.valid,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class InterfaceDefectAssistedFlux:
    """Trap-assisted tunnelling written against an explicit occupancy."""

    channel: str
    occupancy: float
    equilibrium_occupancy: float
    electron_transmission: float
    hole_transmission: float
    electron_net_rate_m2_s: float
    hole_net_rate_m2_s: float
    net_rate_m2_s: float
    occupancy_sensitivity_m2_s: float
    stationary_occupancy_residual: float
    valid: bool
    notes: tuple[str, ...]


def interface_defect_assisted_rate(
    positions_m: ArrayLike,
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    channel: InterfaceDefectAssistedTunnellingChannel,
    *,
    anchor_face: int,
    trap_energy_eV: float,
    occupancy: float | None,
    trap_density_m2: float,
    electron_capture_velocity_m_s: float,
    hole_capture_velocity_m_s: float,
    electron_density_m3: float,
    hole_density_m3: float,
    electron_reference_density_m3: float,
    hole_reference_density_m3: float,
) -> InterfaceDefectAssistedFlux:
    """Trap-assisted tunnelling consuming the solver's explicit trap occupancy."""
    if not channel.enabled:
        raise TunnellingChannelError("interface_defect_assisted channel is disabled")
    if occupancy is None:
        raise TunnellingChannelCapabilityError(
            "interface_defect_assisted requires an explicit interface occupancy; "
            "algebraically eliminated steady SRH cannot supply one"
        )
    if not 0.0 <= float(occupancy) <= 1.0:
        raise TunnellingChannelError("interface occupancy must lie in [0, 1]")
    x = np.asarray(positions_m, dtype=float)
    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    electron_transmission = windowed_wkb_transmission(
        x,
        conduction,
        float(trap_energy_eV),
        channel.electron_effective_mass_rel,
        anchor_face,
    )
    hole_transmission = windowed_wkb_transmission(
        x,
        -valence,
        -float(trap_energy_eV),
        channel.hole_effective_mass_rel,
        anchor_face,
    )
    notes: list[str] = []
    electron_validity = wkb_validity(
        x, conduction, float(trap_energy_eV), channel.electron_effective_mass_rel
    )
    if not electron_validity.valid:
        notes.append("electron_barrier_below_meaningful_action")

    occupied = float(occupancy)
    empty = 1.0 - occupied
    electron_net = (
        float(trap_density_m2)
        * float(electron_capture_velocity_m_s)
        * electron_transmission
        * (
            float(electron_density_m3) * empty
            - float(electron_reference_density_m3) * occupied
        )
    )
    hole_net = (
        float(trap_density_m2)
        * float(hole_capture_velocity_m_s)
        * hole_transmission
        * (
            float(hole_density_m3) * occupied
            - float(hole_reference_density_m3) * empty
        )
    )
    denominator = float(electron_capture_velocity_m_s) * electron_transmission * (
        float(electron_density_m3) + float(electron_reference_density_m3)
    ) + float(hole_capture_velocity_m_s) * hole_transmission * (
        float(hole_density_m3) + float(hole_reference_density_m3)
    )
    if denominator <= 0.0:
        raise TunnellingChannelError("trap-assisted tunnelling has no active capture leg")
    equilibrium = (
        float(electron_capture_velocity_m_s)
        * electron_transmission
        * float(electron_density_m3)
        + float(hole_capture_velocity_m_s)
        * hole_transmission
        * float(hole_reference_density_m3)
    ) / denominator
    sensitivity = float(trap_density_m2) * denominator
    residual = (
        abs(float(electron_net - hole_net)) / sensitivity
        if sensitivity > 0.0
        else math.inf
    )
    return InterfaceDefectAssistedFlux(
        channel="interface_defect_assisted",
        occupancy=occupied,
        equilibrium_occupancy=float(equilibrium),
        electron_transmission=float(electron_transmission),
        hole_transmission=float(hole_transmission),
        electron_net_rate_m2_s=float(electron_net),
        hole_net_rate_m2_s=float(hole_net),
        net_rate_m2_s=float(electron_net - hole_net),
        occupancy_sensitivity_m2_s=sensitivity,
        stationary_occupancy_residual=residual,
        valid=electron_validity.valid,
        notes=tuple(notes),
    )


def contact_tunnelling_flux(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    channel: ContactTunnellingChannel,
    *,
    anchor_face: int,
    carrier: str,
    metal_fermi_eV: float,
    quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    richardson_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Field emission through a Schottky barrier at an outer contact."""
    if not channel.enabled:
        raise TunnellingChannelError("contact channel is disabled")
    mass = (
        channel.electron_effective_mass_rel
        if carrier == ELECTRON
        else channel.hole_effective_mass_rel
    )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(barrier_eV, dtype=float)
    if barrier.shape != x.shape:
        raise WKBTunnellingError("barrier profile must match positions_m")
    peak, base = local_barrier_window(barrier, anchor_face, one_sided=True)
    if not peak > base:
        raise TunnellingChannelError(
            "contact tunnelling requires a barrier above the contact level"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [
            windowed_wkb_transmission(x, barrier, float(energy), mass, anchor_face)
            for energy in energies
        ],
        dtype=float,
    )
    validity = wkb_validity(x, barrier, float(energies[0]), mass)
    notes: list[str] = []
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    metal = _fermi(energies, metal_fermi_eV, thermal_voltage_V)
    _, semiconductor_levels = _turning_point_levels(
        barrier, energies, anchor_face, quasi_fermi_eV
    )
    semiconductor = _fermi(energies, semiconductor_levels, thermal_voltage_V)
    return _channel_flux(
        f"contact_{carrier}",
        energies,
        transmission,
        metal,
        semiconductor,
        richardson_prefactor_m2_s_eV,
        valid=validity.valid,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class CompiledTunnellingChannels:
    """Static part of the tunnelling family bound to one electrical grid."""

    document: TunnellingChannelDocument
    node_count: int
    interface_faces: tuple[int, ...]
    left_contact_face: int
    right_contact_face: int
    intraband_barrier: IntrabandBarrierRegion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, TunnellingChannelDocument):
            raise TypeError("document must be a TunnellingChannelDocument")
        if self.node_count < 3:
            raise TunnellingChannelCapabilityError(
                "tunnelling channels need at least three electrical nodes"
            )
        faces = tuple(int(face) for face in self.interface_faces)
        if any(face < 0 or face >= self.node_count - 1 for face in faces):
            raise TunnellingChannelCapabilityError(
                "interface face index is outside the transport faces"
            )
        object.__setattr__(self, "interface_faces", faces)

    @property
    def any_enabled(self) -> bool:
        return self.document.any_enabled

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        return self.document.enabled_channels

    @property
    def identity_sha256(self) -> str:
        return self.document.sha256

    def digest(self) -> str:
        payload = {
            "version": TUNNELLING_CHANNEL_DEVICE_VERSION,
            "document_sha256": self.document.sha256,
            "node_count": self.node_count,
            "interface_faces": list(self.interface_faces),
            "intraband_barrier": (
                None
                if self.intraband_barrier is None
                else {
                    "identity": self.intraband_barrier.identity,
                    "left_start_node": self.intraband_barrier.left_start_node,
                    "core_start_node": self.intraband_barrier.core_start_node,
                    "core_stop_node": self.intraband_barrier.core_stop_node,
                    "right_stop_node": self.intraband_barrier.right_stop_node,
                    "core_left_m": repr(self.intraband_barrier.core_left_m),
                    "core_right_m": repr(self.intraband_barrier.core_right_m),
                }
            ),
        }
        return _sha256(payload)


def compile_tunnelling_channels(
    document: TunnellingChannelDocument | None,
    *,
    node_count: int,
    interface_nodes: tuple[int, ...],
    positions_m: ArrayLike | None = None,
    physical_affinity_eV: ArrayLike | None = None,
    layer_names: tuple[str, ...] | None = None,
    layer_boundaries_m: ArrayLike | None = None,
    driver: str = "quasi_fermi_steady_state",
) -> CompiledTunnellingChannels | None:
    """Bind a channel document to an electrical grid, enforcing scope pre-solve."""
    if document is None or not document.any_enabled:
        return None
    if driver != "quasi_fermi_steady_state":
        raise TunnellingChannelCapabilityError(
            f"WKB tunnelling channels are certified only for driver='quasi_fermi_steady_state', got {driver!r}"
        )
    if document.enabled_channels != (INTRABAND,) or document.intraband.carrier != ELECTRON:
        raise TunnellingChannelCapabilityError(
            "device tunnelling v2 supports one intraband electron channel; "
            "hole, contact, band-to-band and defect-assisted occupancy coupling "
            "require their own conservative device validation"
        )
    faces = tuple(int(node) - 1 for node in interface_nodes)
    if not faces:
        raise TunnellingChannelCapabilityError(
            "interface-bound tunnelling channels need at least one heterointerface on this stack"
        )
    compiled = CompiledTunnellingChannels(
        document=document,
        node_count=int(node_count),
        interface_faces=faces,
        left_contact_face=0,
        right_contact_face=int(node_count) - 2,
    )
    if positions_m is not None:
        if any(value is None for value in (physical_affinity_eV, layer_names, layer_boundaries_m)):
            raise TunnellingChannelCapabilityError(
                "material barrier identity needs layer geometry and physical affinity"
            )
        assert physical_affinity_eV is not None
        assert layer_names is not None
        assert layer_boundaries_m is not None
        x = np.asarray(positions_m, dtype=float)
        affinity = np.asarray(physical_affinity_eV, dtype=float)
        boundaries = np.asarray(layer_boundaries_m, dtype=float)
        if (
            x.shape != (node_count,)
            or affinity.shape != x.shape
            or not np.all(np.isfinite(affinity))
            or np.any(np.diff(x) <= 0.0)
            or len(layer_names) != len(interface_nodes) + 1
            or boundaries.shape != (len(layer_names) + 1,)
        ):
            raise TunnellingChannelCapabilityError(
                "barrier geometry does not match the layer grid"
            )
        nodes = (0, *interface_nodes, node_count)
        if any(left_i >= right_i for left_i, right_i in zip(nodes[:-1], nodes[1:])):
            raise TunnellingChannelCapabilityError(
                "material layer node ranges must be strictly ordered"
            )
        candidates: list[IntrabandBarrierRegion] = []
        for layer_index in range(1, len(layer_names) - 1):
            start, stop = nodes[layer_index], nodes[layer_index + 1]
            if affinity[start] < affinity[start - 1] and affinity[stop - 1] < affinity[stop]:
                if stop - start < 3:
                    raise TunnellingChannelCapabilityError(
                        "resolved intraband barrier requires at least three barrier nodes"
                    )
                candidates.append(
                    IntrabandBarrierRegion(
                        identity=f"electrical_layer[{layer_index}]:{layer_names[layer_index]}",
                        left_start_node=nodes[layer_index - 1],
                        core_start_node=start,
                        core_stop_node=stop,
                        right_stop_node=nodes[layer_index + 2],
                        core_left_m=float(boundaries[layer_index]),
                        core_right_m=float(boundaries[layer_index + 1]),
                    )
                )
        if len(candidates) != 1:
            raise TunnellingChannelCapabilityError(
                "intraband device transport requires one isolated material conduction-band spike; "
                f"found {len(candidates)}"
            )
        compiled = replace(compiled, intraband_barrier=candidates[0])
    return compiled


@dataclass(frozen=True, slots=True)
class TunnellingChannelEvaluation:
    """Per-face tunnelling currents plus per-channel diagnostics and linearization."""

    identity_sha256: str
    electron_face_flux_m2_s: Vector
    electron_face_current_A_m2: Vector
    hole_face_current_A_m2: Vector
    electron_nodal_transfer_m2_s: Vector
    channel_names: tuple[str, ...]
    channel_net_flux_m2_s: tuple[float, ...]
    channel_maximum_transmission: tuple[float, ...]
    channel_minimum_transmission: tuple[float, ...]
    channel_valid: tuple[bool, ...]
    channel_notes: tuple[tuple[str, ...], ...]
    defect_assisted_occupancy: float | None = None
    defect_assisted_residual: float | None = None
    intraband_path: IntrabandPathFlux | None = None
    linearization: IntrabandPathLinearization | None = None
    implementation_version: str = TUNNELLING_CHANNEL_DEVICE_VERSION


def physical_quasi_fermi_levels_eV(
    *,
    potential_V: ArrayLike,
    affinity_eV: ArrayLike,
    band_gap_eV: ArrayLike,
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    conduction_dos_m3: ArrayLike | None,
    valence_dos_m3: ArrayLike | None,
    thermal_voltage_V: float,
) -> tuple[Vector, Vector]:
    """Physical ``(E_Fn, E_Fp)`` [eV] for Boltzmann carrier populations."""
    if conduction_dos_m3 is None or valence_dos_m3 is None:
        raise TunnellingChannelCapabilityError(
            "physical Nc/Nv are required for tunnelling occupations"
        )
    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (
            potential_V,
            affinity_eV,
            band_gap_eV,
            electron_density_m3,
            hole_density_m3,
            conduction_dos_m3,
            valence_dos_m3,
        )
    )
    potential, affinity, gap, electrons, holes, nc, nv = arrays
    if (
        potential.ndim != 1
        or not potential.size
        or any(value.shape != potential.shape for value in arrays)
    ):
        raise TunnellingChannelCapabilityError(
            "physical level inputs must share one nodal grid"
        )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise TunnellingChannelCapabilityError(
            "physical bands, densities and Nc/Nv must be finite"
        )
    if any(np.any(value <= 0.0) for value in (electrons, holes, nc, nv)):
        raise TunnellingChannelCapabilityError(
            "positive carrier densities and Nc/Nv are required"
        )
    thermal = float(thermal_voltage_V)
    if not math.isfinite(thermal) or thermal <= 0.0:
        raise TunnellingChannelCapabilityError("thermal voltage must be finite and positive")
    conduction = -(potential + affinity)
    electron_level = conduction + thermal * (np.log(electrons) - np.log(nc))
    hole_level = conduction - gap - thermal * (np.log(holes) - np.log(nv))
    return electron_level, hole_level


def evaluate_tunnelling_channels(
    compiled: CompiledTunnellingChannels,
    *,
    positions_m: ArrayLike,
    potential_V: ArrayLike,
    affinity_eV: ArrayLike,
    band_gap_eV: ArrayLike,
    electron_quasi_fermi_eV: ArrayLike,
    hole_quasi_fermi_eV: ArrayLike,
    thermal_voltage_V: float,
    derivatives: bool = False,
) -> TunnellingChannelEvaluation:
    """Evaluate the resolved electron path; other device couplings fail explicitly."""
    document = compiled.document
    if document.enabled_channels != (INTRABAND,) or document.intraband.carrier != ELECTRON:
        raise TunnellingChannelCapabilityError(
            "device tunnelling v2 supports one intraband electron channel; "
            "hole, contact, band-to-band and defect-assisted occupancy coupling "
            "require their own conservative device validation"
        )
    if compiled.intraband_barrier is None:
        raise TunnellingChannelCapabilityError(
            "intraband channel requires compiled material and reservoir geometry"
        )
    x = np.asarray(positions_m, dtype=float)
    potential = np.asarray(potential_V, dtype=float)
    affinity = np.asarray(affinity_eV, dtype=float)
    gap = np.asarray(band_gap_eV, dtype=float)
    qfn = np.asarray(electron_quasi_fermi_eV, dtype=float)
    qfp = np.asarray(hole_quasi_fermi_eV, dtype=float)
    if x.shape != (compiled.node_count,) or any(
        value.shape != x.shape for value in (potential, affinity, gap, qfn, qfp)
    ):
        raise TunnellingChannelCapabilityError(
            "physical tunnelling inputs must share the compiled electrical grid"
        )
    if not all(np.all(np.isfinite(value)) for value in (potential, affinity, gap, qfn, qfp)):
        raise TunnellingChannelCapabilityError(
            "physical tunnelling bands and levels must be finite"
        )
    try:
        lin = intraband_path_flux_and_jacobian(
            x,
            -(potential + affinity),
            document.intraband,
            region=compiled.intraband_barrier,
            quasi_fermi_eV=qfn,
            thermal_voltage_V=thermal_voltage_V,
            compute_jacobian=derivatives,
        )
    except TunnellingChannelError as exc:
        raise TunnellingChannelCapabilityError(
            f"resolved intraband channel failed: {exc}"
        ) from exc
    path = lin.path
    flux = path.flux
    if not flux.valid:
        raise TunnellingChannelCapabilityError(
            "identified barrier has no meaningful WKB action"
        )
    face_flux = path.particle_face_flux_m2_s
    electron_current = -Q * face_flux
    if not np.all(np.isfinite(electron_current)):
        raise TunnellingChannelCapabilityError(
            "tunnelling produced a non-finite face current"
        )
    nodal_transfer = np.zeros(x.size, dtype=float)
    nodal_transfer[:-1] -= face_flux
    nodal_transfer[1:] += face_flux
    return TunnellingChannelEvaluation(
        identity_sha256=compiled.identity_sha256,
        electron_face_flux_m2_s=face_flux,
        electron_face_current_A_m2=frozen_array(electron_current),
        hole_face_current_A_m2=frozen_array(np.zeros(x.size - 1)),
        electron_nodal_transfer_m2_s=frozen_array(nodal_transfer),
        channel_names=(flux.channel,),
        channel_net_flux_m2_s=(flux.net_flux_m2_s,),
        channel_maximum_transmission=(flux.maximum_transmission,),
        channel_minimum_transmission=(flux.minimum_transmission,),
        channel_valid=(flux.valid,),
        channel_notes=(flux.notes,),
        defect_assisted_occupancy=None,
        defect_assisted_residual=None,
        intraband_path=path,
        linearization=lin if derivatives else None,
    )

"""Canonical microscopic contract for single-level interface defects.

The compatibility solvers still carry resolved surface-recombination
velocities on ``DeviceStack.interfaces``. This module preserves the microscopic
areal population that produced those velocities so charged interface closures
can bind recombination, occupancy, and sheet charge to one physical inventory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Self


EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION = (
    "solarlab-explicit-interface-defects-v1"
)
ENERGY_BELOW_REFERENCE_CONDUCTION_BAND = (
    "below_reference_conduction_band"
)
REFERENCE_ABSORBER_ELSE_LOWER_GAP = "absorber_else_lower_gap"
INTEGRATED_AREAL_TOTAL = "integrated_areal_total"
EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY = (
    "equilibrium_referenced_electron_occupancy"
)


class ExplicitInterfaceDefectSchemaError(ValueError):
    """An interface-defect document is incomplete or ambiguous."""


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExplicitInterfaceDefectSchemaError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ExplicitInterfaceDefectSchemaError(
            f"{name} must be finite and non-negative"
        )
    return result


def _finite_positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ExplicitInterfaceDefectSchemaError(
            f"{name} must be finite and positive"
        )
    return result


def _decimal_scale(value: object, factor: str, name: str) -> float:
    try:
        scaled = Decimal(str(value)) * Decimal(factor)
        result = float(scaled)
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ExplicitInterfaceDefectSchemaError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ExplicitInterfaceDefectSchemaError(f"{name} must be finite")
    return result


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unknown:
            details.append(f"unknown keys: {unknown}")
        raise ExplicitInterfaceDefectSchemaError(
            f"{where} requires an exact schema ({'; '.join(details)})"
        )


@dataclass(frozen=True, slots=True)
class InterfaceDefectKinetics:
    """Carrier capture inputs in canonical SI units."""

    sigma_n_m2: float
    sigma_p_m2: float
    thermal_velocity_n_m_s: float
    thermal_velocity_p_m_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sigma_n_m2",
            _finite_nonnegative(self.sigma_n_m2, "sigma_n_m2"),
        )
        object.__setattr__(
            self,
            "sigma_p_m2",
            _finite_nonnegative(self.sigma_p_m2, "sigma_p_m2"),
        )
        object.__setattr__(
            self,
            "thermal_velocity_n_m_s",
            _finite_positive(
                self.thermal_velocity_n_m_s,
                "thermal_velocity_n_m_s",
            ),
        )
        object.__setattr__(
            self,
            "thermal_velocity_p_m_s",
            _finite_positive(
                self.thermal_velocity_p_m_s,
                "thermal_velocity_p_m_s",
            ),
        )

    @property
    def capture_coefficient_n_m3_s(self) -> float:
        return self.sigma_n_m2 * self.thermal_velocity_n_m_s

    @property
    def capture_coefficient_p_m3_s(self) -> float:
        return self.sigma_p_m2 * self.thermal_velocity_p_m_s

    def to_dict(self) -> dict[str, float]:
        return {
            "sigma_n_m2": self.sigma_n_m2,
            "sigma_p_m2": self.sigma_p_m2,
            "thermal_velocity_n_m_s": self.thermal_velocity_n_m_s,
            "thermal_velocity_p_m_s": self.thermal_velocity_p_m_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitInterfaceDefectSchemaError(
                "interface defect kinetics must be a mapping"
            )
        _require_exact_keys(
            value,
            {
                "sigma_n_m2",
                "sigma_p_m2",
                "thermal_velocity_n_m_s",
                "thermal_velocity_p_m_s",
            },
            "interface defect kinetics",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class InterfaceDefectDocument:
    """One energy-integrated, areal, monovalent interface-trap population."""

    schema_version: str
    energy_reference: str
    reference_selection: str
    density_normalization: str
    trap_depth_eV: float
    total_density_m2: float
    kinetics: InterfaceDefectKinetics
    charge_convention: str
    degeneracy: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION:
            raise ExplicitInterfaceDefectSchemaError(
                "unsupported interface defect schema_version "
                f"{self.schema_version!r}"
            )
        if self.energy_reference != ENERGY_BELOW_REFERENCE_CONDUCTION_BAND:
            raise ExplicitInterfaceDefectSchemaError(
                "energy_reference must be "
                f"{ENERGY_BELOW_REFERENCE_CONDUCTION_BAND!r}"
            )
        if self.reference_selection != REFERENCE_ABSORBER_ELSE_LOWER_GAP:
            raise ExplicitInterfaceDefectSchemaError(
                "reference_selection must be "
                f"{REFERENCE_ABSORBER_ELSE_LOWER_GAP!r}"
            )
        if self.density_normalization != INTEGRATED_AREAL_TOTAL:
            raise ExplicitInterfaceDefectSchemaError(
                "density_normalization must be "
                f"{INTEGRATED_AREAL_TOTAL!r}"
            )
        if self.charge_convention != EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY:
            raise ExplicitInterfaceDefectSchemaError(
                "charge_convention must be "
                f"{EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY!r}"
            )
        object.__setattr__(
            self,
            "trap_depth_eV",
            _finite_nonnegative(self.trap_depth_eV, "trap_depth_eV"),
        )
        object.__setattr__(
            self,
            "total_density_m2",
            _finite_positive(self.total_density_m2, "total_density_m2"),
        )
        object.__setattr__(
            self,
            "degeneracy",
            _finite_positive(self.degeneracy, "degeneracy"),
        )
        if not isinstance(self.kinetics, InterfaceDefectKinetics):
            raise ExplicitInterfaceDefectSchemaError(
                "kinetics must be an InterfaceDefectKinetics"
            )

    @property
    def capture_velocity_n_m_s(self) -> float:
        return self.kinetics.capture_coefficient_n_m3_s * self.total_density_m2

    @property
    def capture_velocity_p_m_s(self) -> float:
        return self.kinetics.capture_coefficient_p_m3_s * self.total_density_m2

    @property
    def capture_velocities_m_s(self) -> tuple[float, float]:
        return self.capture_velocity_n_m_s, self.capture_velocity_p_m_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "energy_reference": self.energy_reference,
            "reference_selection": self.reference_selection,
            "density_normalization": self.density_normalization,
            "trap_depth_eV": self.trap_depth_eV,
            "total_density_m2": self.total_density_m2,
            "kinetics": self.kinetics.to_dict(),
            "charge_convention": self.charge_convention,
            "degeneracy": self.degeneracy,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitInterfaceDefectSchemaError(
                "interface defect document must be a mapping"
            )
        expected = {
            "schema_version",
            "energy_reference",
            "reference_selection",
            "density_normalization",
            "trap_depth_eV",
            "total_density_m2",
            "kinetics",
            "charge_convention",
            "degeneracy",
        }
        _require_exact_keys(value, expected, "interface defect document")
        payload = dict(value)
        payload["kinetics"] = InterfaceDefectKinetics.from_dict(payload["kinetics"])
        return cls(**payload)

    @classmethod
    def from_scaps_cgs(
        cls,
        *,
        sigma_n_cm2: object,
        sigma_p_cm2: object,
        thermal_velocity_cm_s: object,
        total_density_cm2: object,
        trap_depth_eV_below_cb: object,
    ) -> Self:
        """Convert the existing SCAPS-friendly flat fields to canonical SI."""

        thermal_velocity_m_s = _decimal_scale(
            thermal_velocity_cm_s,
            "1e-2",
            "thermal_velocity_cm_s",
        )
        return cls(
            schema_version=EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION,
            energy_reference=ENERGY_BELOW_REFERENCE_CONDUCTION_BAND,
            reference_selection=REFERENCE_ABSORBER_ELSE_LOWER_GAP,
            density_normalization=INTEGRATED_AREAL_TOTAL,
            trap_depth_eV=_finite_nonnegative(
                trap_depth_eV_below_cb,
                "trap_depth_eV_below_cb",
            ),
            total_density_m2=_decimal_scale(
                total_density_cm2,
                "1e4",
                "total_density_cm2",
            ),
            kinetics=InterfaceDefectKinetics(
                sigma_n_m2=_decimal_scale(
                    sigma_n_cm2,
                    "1e-4",
                    "sigma_n_cm2",
                ),
                sigma_p_m2=_decimal_scale(
                    sigma_p_cm2,
                    "1e-4",
                    "sigma_p_cm2",
                ),
                thermal_velocity_n_m_s=thermal_velocity_m_s,
                thermal_velocity_p_m_s=thermal_velocity_m_s,
            ),
            charge_convention=EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY,
        )

    def to_scaps_cgs_fields(self) -> dict[str, float]:
        """Return the lossless flat adapter shape used by the workstation."""

        velocity_n = self.kinetics.thermal_velocity_n_m_s
        velocity_p = self.kinetics.thermal_velocity_p_m_s
        if velocity_n != velocity_p:
            raise ExplicitInterfaceDefectSchemaError(
                "the flat SCAPS adapter requires one shared thermal velocity"
            )
        return {
            "sigma_n_cm2": _decimal_scale(
                self.kinetics.sigma_n_m2,
                "1e4",
                "sigma_n_m2",
            ),
            "sigma_p_cm2": _decimal_scale(
                self.kinetics.sigma_p_m2,
                "1e4",
                "sigma_p_m2",
            ),
            "v_th_cm_s": _decimal_scale(
                velocity_n,
                "1e2",
                "thermal_velocity_m_s",
            ),
            "N_t_cm2": _decimal_scale(
                self.total_density_m2,
                "1e-4",
                "total_density_m2",
            ),
            "E_t_eV_below_cb": self.trap_depth_eV,
        }


__all__ = [
    "ENERGY_BELOW_REFERENCE_CONDUCTION_BAND",
    "EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY",
    "EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION",
    "ExplicitInterfaceDefectSchemaError",
    "INTEGRATED_AREAL_TOTAL",
    "InterfaceDefectDocument",
    "InterfaceDefectKinetics",
    "REFERENCE_ABSORBER_ELSE_LOWER_GAP",
]

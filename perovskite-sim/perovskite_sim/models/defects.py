"""Canonical contracts for explicit bulk-defect inputs.

The schema in this module records microscopic defect identity without changing
the production recombination path.  Standard SolarLab inputs use SI units;
SCAPS-shaped inputs are converted here before they reach ``MaterialParams``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Self


EXPLICIT_DEFECT_SCHEMA_VERSION = "solarlab-explicit-bulk-defects-v1"
EFFECTIVE_LIFETIME = "effective_lifetime"
EXPLICIT_QUASI_STEADY = "explicit_quasi_steady"
EXPLICIT_DYNAMIC = "explicit_dynamic"

SINGLE_LEVEL = "single_level"
GAUSSIAN = "gaussian"
INTEGRATED_TOTAL = "integrated_total"

NEUTRAL = "neutral"
ACCEPTOR = "acceptor"
DONOR = "donor"
UNRESOLVED = "unresolved"

NEUTRAL_ALL_OCCUPANCIES = "all_occupancies"
NEUTRAL_WHEN_EMPTY = "empty"
NEUTRAL_WHEN_FILLED = "filled"
NEUTRAL_REFERENCE_UNRESOLVED = "unresolved"

WIDTH_NOT_APPLICABLE = "not_applicable"
WIDTH_GAUSSIAN_SIGMA = "gaussian_standard_deviation"
WIDTH_SCAPS_CHARACTERISTIC = "scaps_characteristic_energy"
WIDTH_UNRESOLVED = "unresolved"

DefectModel = Literal["effective_lifetime", "explicit_quasi_steady"]
DefectDistributionKind = Literal["single_level", "gaussian"]
DefectChargeTransition = Literal[
    "neutral", "acceptor", "donor", "unresolved"
]
DefectNeutralReference = Literal[
    "all_occupancies", "empty", "filled", "unresolved"
]


class ExplicitDefectSchemaError(ValueError):
    """An explicit-defect document is incomplete or ambiguous."""


class ExplicitDefectCapabilityError(RuntimeError):
    """A valid defect document requested an execution path not yet enabled."""


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ExplicitDefectSchemaError(f"{field} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and positive"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ExplicitDefectSchemaError(f"{field} must be finite and positive")
    return number


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and non-negative"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and non-negative"
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and non-negative"
        )
    return number


def _decimal_scale(value: object, factor: str, field: str) -> float:
    """Scale a validated decimal input without adding a binary-float ULP."""

    number = _finite_nonnegative(value, field)
    return float(Decimal(str(number)) * Decimal(factor))


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(str(key) for key in actual - expected)
        raise ExplicitDefectSchemaError(
            f"{where} schema mismatch: missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True, slots=True)
class BulkDefectDistribution:
    """One energy distribution with an explicitly integrated density."""

    kind: DefectDistributionKind
    normalization: Literal["integrated_total"]
    total_density_m3: float
    center_eV_above_vb: float
    width_eV: float | None = None
    width_convention: str = WIDTH_NOT_APPLICABLE

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {SINGLE_LEVEL, GAUSSIAN}:
            raise ExplicitDefectSchemaError(
                "defect distribution kind must be 'single_level' or 'gaussian'"
            )
        normalization = str(self.normalization).strip().lower()
        if normalization != INTEGRATED_TOTAL:
            raise ExplicitDefectSchemaError(
                "defect density normalization must be 'integrated_total'"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(
            self,
            "total_density_m3",
            _finite_positive(self.total_density_m3, "total_density_m3"),
        )
        object.__setattr__(
            self,
            "center_eV_above_vb",
            _finite_nonnegative(
                self.center_eV_above_vb,
                "center_eV_above_vb",
            ),
        )
        convention = str(self.width_convention).strip().lower()
        if kind == SINGLE_LEVEL:
            if self.width_eV is not None or convention != WIDTH_NOT_APPLICABLE:
                raise ExplicitDefectSchemaError(
                    "single_level defects forbid width_eV and require "
                    "width_convention='not_applicable'"
                )
            object.__setattr__(self, "width_convention", convention)
            return
        if convention not in {
            WIDTH_GAUSSIAN_SIGMA,
            WIDTH_SCAPS_CHARACTERISTIC,
            WIDTH_UNRESOLVED,
        }:
            raise ExplicitDefectSchemaError(
                "gaussian width_convention must identify standard deviation, "
                "SCAPS characteristic energy, or unresolved source metadata"
            )
        if convention == WIDTH_UNRESOLVED:
            if self.width_eV is not None:
                raise ExplicitDefectSchemaError(
                    "an unresolved gaussian width must not carry width_eV"
                )
        else:
            object.__setattr__(
                self,
                "width_eV",
                _finite_positive(self.width_eV, "width_eV"),
            )
        object.__setattr__(self, "width_convention", convention)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        gap = _finite_positive(band_gap_eV, "band_gap_eV")
        if self.center_eV_above_vb > gap:
            raise ExplicitDefectSchemaError(
                "center_eV_above_vb must lie inside the material band gap"
            )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "normalization": self.normalization,
            "total_density_m3": self.total_density_m3,
            "center_eV_above_vb": self.center_eV_above_vb,
        }
        if self.kind == GAUSSIAN:
            value["width_eV"] = self.width_eV
            value["width_convention"] = self.width_convention
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError("distribution must be a mapping")
        kind = str(value.get("kind", "")).strip().lower()
        expected = {
            "kind",
            "normalization",
            "total_density_m3",
            "center_eV_above_vb",
        }
        if kind == GAUSSIAN:
            expected |= {"width_eV", "width_convention"}
        _require_exact_keys(value, expected, "bulk defect distribution")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class BulkDefectKinetics:
    """Carrier-capture inputs in canonical SI units."""

    sigma_n_m2: float
    sigma_p_m2: float
    thermal_velocity_n_m_s: float
    thermal_velocity_p_m_s: float

    def __post_init__(self) -> None:
        for name in ("sigma_n_m2", "sigma_p_m2"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in ("thermal_velocity_n_m_s", "thermal_velocity_p_m_s"):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), name),
            )

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
            raise ExplicitDefectSchemaError("kinetics must be a mapping")
        expected = {
            "sigma_n_m2",
            "sigma_p_m2",
            "thermal_velocity_n_m_s",
            "thermal_velocity_p_m_s",
        }
        _require_exact_keys(value, expected, "bulk defect kinetics")
        return cls(**dict(value))


_NEUTRAL_REFERENCE_BY_TRANSITION = {
    NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
    ACCEPTOR: NEUTRAL_WHEN_EMPTY,
    DONOR: NEUTRAL_WHEN_FILLED,
    UNRESOLVED: NEUTRAL_REFERENCE_UNRESOLVED,
}


@dataclass(frozen=True, slots=True)
class BulkDefectSpecies:
    """One microscopic defect species and its charge-state convention."""

    name: str | None
    distribution: BulkDefectDistribution
    charge_transition: DefectChargeTransition
    neutral_reference: DefectNeutralReference
    kinetics: BulkDefectKinetics
    degeneracy: float = 1.0

    def __post_init__(self) -> None:
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ExplicitDefectSchemaError(
                    "bulk defect name must be null or a non-empty string"
                )
            object.__setattr__(self, "name", self.name.strip())
        if not isinstance(self.distribution, BulkDefectDistribution):
            raise TypeError("distribution must be a BulkDefectDistribution")
        if not isinstance(self.kinetics, BulkDefectKinetics):
            raise TypeError("kinetics must be BulkDefectKinetics")
        transition = str(self.charge_transition).strip().lower()
        if transition not in _NEUTRAL_REFERENCE_BY_TRANSITION:
            raise ExplicitDefectSchemaError(
                "charge_transition must be neutral, acceptor, donor, or unresolved"
            )
        neutral_reference = str(self.neutral_reference).strip().lower()
        expected_reference = _NEUTRAL_REFERENCE_BY_TRANSITION[transition]
        if neutral_reference != expected_reference:
            raise ExplicitDefectSchemaError(
                f"charge_transition={transition!r} requires "
                f"neutral_reference={expected_reference!r}"
            )
        object.__setattr__(self, "charge_transition", transition)
        object.__setattr__(self, "neutral_reference", neutral_reference)
        object.__setattr__(
            self,
            "degeneracy",
            _finite_positive(self.degeneracy, "degeneracy"),
        )

    @property
    def explicit_ready(self) -> bool:
        return (
            self.name is not None
            and self.charge_transition != UNRESOLVED
            and self.distribution.kind == SINGLE_LEVEL
        )

    def validate_band_gap(self, band_gap_eV: object) -> None:
        self.distribution.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "distribution": self.distribution.to_dict(),
            "charge_transition": self.charge_transition,
            "neutral_reference": self.neutral_reference,
            "kinetics": self.kinetics.to_dict(),
            "degeneracy": self.degeneracy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError("bulk defect species must be a mapping")
        expected = {
            "name",
            "distribution",
            "charge_transition",
            "neutral_reference",
            "kinetics",
            "degeneracy",
        }
        _require_exact_keys(value, expected, "bulk defect species")
        return cls(
            name=value["name"],
            distribution=BulkDefectDistribution.from_dict(value["distribution"]),
            charge_transition=value["charge_transition"],
            neutral_reference=value["neutral_reference"],
            kinetics=BulkDefectKinetics.from_dict(value["kinetics"]),
            degeneracy=value["degeneracy"],
        )


@dataclass(frozen=True, slots=True)
class BulkDefectDocument:
    """Versioned per-layer defect selector and microscopic species list."""

    schema_version: str
    defect_model: DefectModel
    bulk_defects: tuple[BulkDefectSpecies, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EXPLICIT_DEFECT_SCHEMA_VERSION:
            raise ExplicitDefectSchemaError(
                "unsupported explicit defect schema_version "
                f"{self.schema_version!r}"
            )
        model = str(self.defect_model).strip().lower()
        if model not in {EFFECTIVE_LIFETIME, EXPLICIT_QUASI_STEADY}:
            if model == EXPLICIT_DYNAMIC:
                raise ExplicitDefectSchemaError(
                    "explicit_dynamic is reserved for a future schema"
                )
            raise ExplicitDefectSchemaError(f"unknown defect_model {model!r}")
        species = tuple(self.bulk_defects)
        if not all(isinstance(item, BulkDefectSpecies) for item in species):
            raise TypeError("bulk_defects must contain BulkDefectSpecies values")
        names = [item.name for item in species if item.name is not None]
        if len(names) != len(set(names)):
            raise ExplicitDefectSchemaError("bulk defect names must be unique")
        if model == EXPLICIT_QUASI_STEADY:
            if not species:
                raise ExplicitDefectSchemaError(
                    "explicit_quasi_steady requires at least one bulk defect"
                )
            not_ready = [
                item.name or f"species[{index}]"
                for index, item in enumerate(species)
                if not item.explicit_ready
            ]
            if not_ready:
                raise ExplicitDefectSchemaError(
                    "explicit_quasi_steady v1 requires named, charge-resolved "
                    f"single-level species; invalid={not_ready}"
                )
        object.__setattr__(self, "defect_model", model)
        object.__setattr__(self, "bulk_defects", species)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        for species in self.bulk_defects:
            species.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect_model": self.defect_model,
            "bulk_defects": [item.to_dict() for item in self.bulk_defects],
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
            raise ExplicitDefectSchemaError("bulk defect document must be a mapping")
        expected = {"schema_version", "defect_model", "bulk_defects"}
        _require_exact_keys(value, expected, "bulk defect document")
        raw_species = value["bulk_defects"]
        if (
            not isinstance(raw_species, Sequence)
            or isinstance(raw_species, (str, bytes, bytearray))
        ):
            raise ExplicitDefectSchemaError("bulk_defects must be a list")
        return cls(
            schema_version=value["schema_version"],
            defect_model=value["defect_model"],
            bulk_defects=tuple(
                BulkDefectSpecies.from_dict(item) for item in raw_species
            ),
        )


def bulk_defect_document_from_layer_mapping(
    layer: Mapping[str, Any],
) -> BulkDefectDocument | None:
    """Parse the strict flat layer keys used by standard SolarLab configs."""

    keys = {"defect_schema_version", "defect_model", "bulk_defects"}
    present = keys & set(layer)
    if not present:
        return None
    if present != keys:
        raise ExplicitDefectSchemaError(
            "explicit defect layer contract requires defect_schema_version, "
            f"defect_model, and bulk_defects together; missing={sorted(keys-present)}"
        )
    return BulkDefectDocument.from_dict(
        {
            "schema_version": layer["defect_schema_version"],
            "defect_model": layer["defect_model"],
            "bulk_defects": layer["bulk_defects"],
        }
    )


def bulk_defect_species_from_scaps_mapping(
    value: Mapping[str, Any],
    *,
    band_gap_eV: object,
    layer_thermal_velocity_m_s: object,
    where: str,
) -> BulkDefectSpecies:
    """Convert one strict SCAPS-cgs defect entry to canonical SI."""

    if not isinstance(value, Mapping):
        raise ExplicitDefectSchemaError(f"{where} must be a mapping")
    required = {"sigma_n_cm2", "sigma_p_cm2", "N_t_cm3"}
    optional = {
        "name",
        "E_t_eV_below_cb",
        "E_t_eV_above_vb",
        "distribution",
        "E_char_eV",
        "N_peak_cm3",
        "charge_transition",
        "neutral_reference",
        "degeneracy",
    }
    missing = sorted(required - set(value))
    unknown = sorted(str(key) for key in set(value) - required - optional)
    if missing or unknown:
        raise ExplicitDefectSchemaError(
            f"{where} schema mismatch: missing={missing}, unknown={unknown}"
        )
    has_below = "E_t_eV_below_cb" in value
    has_above = "E_t_eV_above_vb" in value
    if has_below == has_above:
        raise ExplicitDefectSchemaError(
            f"{where}: exactly one of E_t_eV_below_cb / "
            "E_t_eV_above_vb required"
        )
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    if has_below:
        depth = _finite_nonnegative(
            value["E_t_eV_below_cb"],
            f"{where}.E_t_eV_below_cb",
        )
        center = float(Decimal(str(gap)) - Decimal(str(depth)))
    else:
        center = _finite_nonnegative(
            value["E_t_eV_above_vb"],
            f"{where}.E_t_eV_above_vb",
        )
    if center < 0.0 or center > gap:
        raise ExplicitDefectSchemaError(f"{where}: trap level lies outside the gap")

    distribution_name = str(value.get("distribution", "single")).strip().lower()
    if distribution_name in {"single", SINGLE_LEVEL}:
        distribution = BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=(
                _decimal_scale(value["N_t_cm3"], "1e6", f"{where}.N_t_cm3")
            ),
            center_eV_above_vb=center,
        )
    elif distribution_name == GAUSSIAN:
        if "E_char_eV" in value:
            width = _finite_positive(value["E_char_eV"], f"{where}.E_char_eV")
            width_convention = WIDTH_SCAPS_CHARACTERISTIC
        else:
            width = None
            width_convention = WIDTH_UNRESOLVED
        distribution = BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=(
                _decimal_scale(value["N_t_cm3"], "1e6", f"{where}.N_t_cm3")
            ),
            center_eV_above_vb=center,
            width_eV=width,
            width_convention=width_convention,
        )
    else:
        raise ExplicitDefectSchemaError(
            f"{where}: unknown distribution {distribution_name!r}"
        )
    if "N_peak_cm3" in value:
        _finite_nonnegative(value["N_peak_cm3"], f"{where}.N_peak_cm3")

    has_transition = "charge_transition" in value
    has_reference = "neutral_reference" in value
    if has_transition != has_reference:
        raise ExplicitDefectSchemaError(
            f"{where}: charge_transition and neutral_reference must be "
            "declared together"
        )
    transition = value.get("charge_transition", UNRESOLVED)
    neutral_reference = value.get(
        "neutral_reference", NEUTRAL_REFERENCE_UNRESOLVED
    )
    thermal_velocity = _finite_positive(
        layer_thermal_velocity_m_s,
        "layer_thermal_velocity_m_s",
    )
    species = BulkDefectSpecies(
        name=value.get("name"),
        distribution=distribution,
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=(
                _decimal_scale(
                    value["sigma_n_cm2"], "1e-4", f"{where}.sigma_n_cm2"
                )
            ),
            sigma_p_m2=(
                _decimal_scale(
                    value["sigma_p_cm2"], "1e-4", f"{where}.sigma_p_cm2"
                )
            ),
            thermal_velocity_n_m_s=thermal_velocity,
            thermal_velocity_p_m_s=thermal_velocity,
        ),
        degeneracy=value.get("degeneracy", 1.0),
    )
    species.validate_band_gap(gap)
    return species


__all__ = [
    "ACCEPTOR",
    "DONOR",
    "EFFECTIVE_LIFETIME",
    "EXPLICIT_DEFECT_SCHEMA_VERSION",
    "EXPLICIT_DYNAMIC",
    "EXPLICIT_QUASI_STEADY",
    "GAUSSIAN",
    "INTEGRATED_TOTAL",
    "NEUTRAL",
    "NEUTRAL_ALL_OCCUPANCIES",
    "NEUTRAL_REFERENCE_UNRESOLVED",
    "NEUTRAL_WHEN_EMPTY",
    "NEUTRAL_WHEN_FILLED",
    "SINGLE_LEVEL",
    "UNRESOLVED",
    "BulkDefectDistribution",
    "BulkDefectDocument",
    "BulkDefectKinetics",
    "BulkDefectSpecies",
    "ExplicitDefectCapabilityError",
    "ExplicitDefectSchemaError",
    "bulk_defect_document_from_layer_mapping",
    "bulk_defect_species_from_scaps_mapping",
]

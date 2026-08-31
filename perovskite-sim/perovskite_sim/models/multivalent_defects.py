"""Canonical contracts for multivalent and metastable bulk defects.

This module is deliberately solver independent.  Version 4 represents one
physical defect as a normalized set of coupled charge states; it must not be
reduced to independent monovalent SRH centres.  Metastable preparation is a
separate, replayable contract because SCAPS first establishes a donor/acceptor
configuration distribution at an initial working point and then freezes that
distribution for the measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Self

from perovskite_sim.models.defects import (
    EXPLICIT_QUASI_STEADY,
    BulkDefectKinetics,
    ExplicitDefectSchemaError,
)


MULTIVALENT_DEFECT_SCHEMA_VERSION = "solarlab-explicit-bulk-defects-v4"
METASTABLE_DEFECT_SCHEMA_VERSION = "solarlab-metastable-bulk-defects-v1"
METASTABLE_PREPARATION_SCHEMA_VERSION = "solarlab-metastable-preparation-v1"
EXPLICIT_METASTABLE_FROZEN = "explicit_metastable_frozen"

SINGLE_DONOR = "single_donor"
SINGLE_ACCEPTOR = "single_acceptor"
DOUBLE_DONOR = "double_donor"
DOUBLE_ACCEPTOR = "double_acceptor"
AMPHOTERIC = "amphoteric"
CUSTOM_MULTILEVEL = "custom_multilevel"

SCAPS_BINOMIAL = "scaps_binomial"
UNITY = "unity"
EXPLICIT = "explicit"

DOUBLE_ELECTRON_CAPTURE = "double_electron_capture"
ELECTRON_CAPTURE_HOLE_EMISSION = "electron_capture_plus_hole_emission"
DOUBLE_HOLE_CAPTURE = "double_hole_capture"
HOLE_CAPTURE_ELECTRON_EMISSION = "hole_capture_plus_electron_emission"

STATIONARY_INFINITE_TIME = "stationary_infinite_time"
FROZEN_BEFORE_MEASUREMENT = "after_stationary_preparation_before_measurement"

MultivalentFamily = Literal[
    "single_donor",
    "single_acceptor",
    "double_donor",
    "double_acceptor",
    "amphoteric",
    "custom_multilevel",
]
DegeneracyConvention = Literal[
    "scaps_binomial",
    "unity",
    "explicit",
]

_FAMILY_CHARGES: dict[str, tuple[int, ...]] = {
    SINGLE_DONOR: (1, 0),
    SINGLE_ACCEPTOR: (0, -1),
    DOUBLE_DONOR: (2, 1, 0),
    DOUBLE_ACCEPTOR: (0, -1, -2),
    AMPHOTERIC: (1, 0, -1),
}
_FAMILIES = set(_FAMILY_CHARGES) | {CUSTOM_MULTILEVEL}
_DEGENERACY_CONVENTIONS = {SCAPS_BINOMIAL, UNITY, EXPLICIT}
_ELECTRON_CAPTURE_PATHS = {
    DOUBLE_ELECTRON_CAPTURE,
    ELECTRON_CAPTURE_HOLE_EMISSION,
}
_HOLE_CAPTURE_PATHS = {
    DOUBLE_HOLE_CAPTURE,
    HOLE_CAPTURE_ELECTRON_EMISSION,
}


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ExplicitDefectSchemaError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _positive(value: object, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ExplicitDefectSchemaError(f"{field} must be positive")
    return result


def _nonnegative(value: object, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ExplicitDefectSchemaError(f"{field} must be non-negative")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ExplicitDefectSchemaError(
            f"{field} must be greater than or equal to {minimum}"
        )
    return result


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplicitDefectSchemaError(f"{field} must be a non-empty string")
    return value.strip()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ExplicitDefectSchemaError(
            f"{where} schema mismatch: missing={missing}, unknown={unknown}"
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


def _validate_sha256(value: object, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ExplicitDefectSchemaError(f"{field} must be a SHA-256 hex digest")
    return digest


@dataclass(frozen=True, slots=True)
class MultivalentEnergyLevels:
    """Transition levels represented by a first level and correlation energies."""

    first_transition_eV_above_vb: float
    correlation_energies_eV: tuple[float, ...]
    energy_reference: Literal["above_valence_band"] = "above_valence_band"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "first_transition_eV_above_vb",
            _nonnegative(
                self.first_transition_eV_above_vb,
                "first_transition_eV_above_vb",
            ),
        )
        correlations = tuple(
            _finite(value, f"correlation_energies_eV[{index}]")
            for index, value in enumerate(self.correlation_energies_eV)
        )
        object.__setattr__(self, "correlation_energies_eV", correlations)
        if self.energy_reference != "above_valence_band":
            raise ExplicitDefectSchemaError(
                "multivalent energy_reference must be 'above_valence_band'"
            )

    @property
    def transition_energies_eV_above_vb(self) -> tuple[float, ...]:
        values = [self.first_transition_eV_above_vb]
        for correlation in self.correlation_energies_eV:
            values.append(values[-1] + correlation)
        return tuple(values)

    def validate_transition_count(self, transition_count: int) -> None:
        if len(self.transition_energies_eV_above_vb) != transition_count:
            raise ExplicitDefectSchemaError(
                "multivalent energy levels must provide one energy per "
                f"transition; expected={transition_count}, "
                f"actual={len(self.transition_energies_eV_above_vb)}"
            )

    def validate_band_gap(self, band_gap_eV: object) -> None:
        gap = _positive(band_gap_eV, "band_gap_eV")
        scale = max(gap, *map(abs, self.transition_energies_eV_above_vb), 1.0)
        tolerance = 16.0 * math.ulp(scale)
        invalid = [
            value
            for value in self.transition_energies_eV_above_vb
            if value < -tolerance or value > gap + tolerance
        ]
        if invalid:
            raise ExplicitDefectSchemaError(
                "all multivalent transition levels must lie inside the band gap"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_transition_eV_above_vb": self.first_transition_eV_above_vb,
            "correlation_energies_eV": list(self.correlation_energies_eV),
            "energy_reference": self.energy_reference,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "multivalent energy levels must be a mapping"
            )
        _require_exact_keys(
            value,
            {
                "first_transition_eV_above_vb",
                "correlation_energies_eV",
                "energy_reference",
            },
            "multivalent energy levels",
        )
        correlations = value["correlation_energies_eV"]
        if not isinstance(correlations, Sequence) or isinstance(
            correlations, (str, bytes, bytearray)
        ):
            raise ExplicitDefectSchemaError("correlation_energies_eV must be a list")
        return cls(
            first_transition_eV_above_vb=value["first_transition_eV_above_vb"],
            correlation_energies_eV=tuple(correlations),
            energy_reference=value["energy_reference"],
        )


@dataclass(frozen=True, slots=True)
class MultivalentDefectConfiguration:
    """Coupled charge states and adjacent carrier transitions of one defect."""

    family: MultivalentFamily
    charge_states_e: tuple[int, ...]
    degeneracy_convention: DegeneracyConvention
    state_degeneracies: tuple[float, ...]
    energy_levels: MultivalentEnergyLevels
    transition_kinetics: tuple[BulkDefectKinetics, ...]

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if family not in _FAMILIES:
            raise ExplicitDefectSchemaError(
                f"unknown multivalent defect family {family!r}"
            )
        charges = tuple(
            _integer(value, f"charge_states_e[{index}]", minimum=-3)
            for index, value in enumerate(self.charge_states_e)
        )
        if not 2 <= len(charges) <= 5 or any(value > 3 for value in charges):
            raise ExplicitDefectSchemaError(
                "multivalent defects require 2-5 charge states in [-3, 3]"
            )
        if any(right != left - 1 for left, right in zip(charges, charges[1:])):
            raise ExplicitDefectSchemaError(
                "charge_states_e must descend by one electron charge per level"
            )
        predefined = _FAMILY_CHARGES.get(family)
        if predefined is not None and charges != predefined:
            raise ExplicitDefectSchemaError(
                f"family={family!r} requires charge_states_e={predefined}"
            )

        convention = str(self.degeneracy_convention).strip().lower()
        if convention not in _DEGENERACY_CONVENTIONS:
            raise ExplicitDefectSchemaError(
                f"unknown degeneracy convention {convention!r}"
            )
        degeneracies = tuple(
            _positive(value, f"state_degeneracies[{index}]")
            for index, value in enumerate(self.state_degeneracies)
        )
        if len(degeneracies) != len(charges):
            raise ExplicitDefectSchemaError(
                "state_degeneracies must have one value per charge state"
            )
        if convention == SCAPS_BINOMIAL:
            expected = tuple(
                float(math.comb(len(charges) - 1, index))
                for index in range(len(charges))
            )
            if degeneracies != expected:
                raise ExplicitDefectSchemaError(
                    f"scaps_binomial requires binomial state degeneracies {expected}"
                )
        elif convention == UNITY and any(value != 1.0 for value in degeneracies):
            raise ExplicitDefectSchemaError(
                "unity degeneracy convention requires every value to equal 1"
            )

        if not isinstance(self.energy_levels, MultivalentEnergyLevels):
            raise TypeError("energy_levels must be MultivalentEnergyLevels")
        kinetics = tuple(self.transition_kinetics)
        if len(kinetics) != len(charges) - 1 or not all(
            isinstance(value, BulkDefectKinetics) for value in kinetics
        ):
            raise ExplicitDefectSchemaError(
                "transition_kinetics must contain one BulkDefectKinetics per "
                "adjacent charge-state transition"
            )
        for index, value in enumerate(kinetics):
            if value.sigma_n_m2 == 0.0 and value.sigma_p_m2 == 0.0:
                raise ExplicitDefectSchemaError(
                    "each multivalent transition needs at least one carrier "
                    f"capture leg; invalid transition={index}"
                )
        self.energy_levels.validate_transition_count(len(charges) - 1)

        object.__setattr__(self, "family", family)
        object.__setattr__(self, "charge_states_e", charges)
        object.__setattr__(self, "degeneracy_convention", convention)
        object.__setattr__(self, "state_degeneracies", degeneracies)
        object.__setattr__(self, "transition_kinetics", kinetics)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        self.energy_levels.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "charge_states_e": list(self.charge_states_e),
            "degeneracy_convention": self.degeneracy_convention,
            "state_degeneracies": list(self.state_degeneracies),
            "energy_levels": self.energy_levels.to_dict(),
            "transition_kinetics": [
                value.to_dict() for value in self.transition_kinetics
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "multivalent defect configuration must be a mapping"
            )
        _require_exact_keys(
            value,
            {
                "family",
                "charge_states_e",
                "degeneracy_convention",
                "state_degeneracies",
                "energy_levels",
                "transition_kinetics",
            },
            "multivalent defect configuration",
        )
        for name in (
            "charge_states_e",
            "state_degeneracies",
            "transition_kinetics",
        ):
            if not isinstance(value[name], Sequence) or isinstance(
                value[name], (str, bytes, bytearray)
            ):
                raise ExplicitDefectSchemaError(f"{name} must be a list")
        return cls(
            family=value["family"],
            charge_states_e=tuple(value["charge_states_e"]),
            degeneracy_convention=value["degeneracy_convention"],
            state_degeneracies=tuple(value["state_degeneracies"]),
            energy_levels=MultivalentEnergyLevels.from_dict(value["energy_levels"]),
            transition_kinetics=tuple(
                BulkDefectKinetics.from_dict(item)
                for item in value["transition_kinetics"]
            ),
        )


@dataclass(frozen=True, slots=True)
class MultivalentBulkDefectSpecies:
    """A named multivalent configuration with one shared total density."""

    name: str
    total_density_m3: float
    configuration: MultivalentDefectConfiguration

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "defect name"))
        object.__setattr__(
            self,
            "total_density_m3",
            _positive(self.total_density_m3, "total_density_m3"),
        )
        if not isinstance(self.configuration, MultivalentDefectConfiguration):
            raise TypeError("configuration must be MultivalentDefectConfiguration")

    def validate_band_gap(self, band_gap_eV: object) -> None:
        self.configuration.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_density_m3": self.total_density_m3,
            "configuration": self.configuration.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "multivalent bulk defect species must be a mapping"
            )
        _require_exact_keys(
            value,
            {"name", "total_density_m3", "configuration"},
            "multivalent bulk defect species",
        )
        return cls(
            name=value["name"],
            total_density_m3=value["total_density_m3"],
            configuration=MultivalentDefectConfiguration.from_dict(
                value["configuration"]
            ),
        )


@dataclass(frozen=True, slots=True)
class MultivalentBulkDefectDocument:
    """Version-4 quasi-steady multivalent bulk-defect input document."""

    schema_version: Literal["solarlab-explicit-bulk-defects-v4"]
    defect_model: Literal["explicit_quasi_steady"]
    bulk_defects: tuple[MultivalentBulkDefectSpecies, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MULTIVALENT_DEFECT_SCHEMA_VERSION:
            raise ExplicitDefectSchemaError(
                "unsupported multivalent bulk-defect schema_version"
            )
        if self.defect_model != EXPLICIT_QUASI_STEADY:
            raise ExplicitDefectSchemaError(
                "v4 multivalent defects require explicit_quasi_steady"
            )
        species = tuple(self.bulk_defects)
        if not species or not all(
            isinstance(value, MultivalentBulkDefectSpecies) for value in species
        ):
            raise ExplicitDefectSchemaError(
                "v4 bulk_defects must contain multivalent species"
            )
        names = [value.name for value in species]
        if len(names) != len(set(names)):
            raise ExplicitDefectSchemaError(
                "multivalent bulk-defect names must be unique"
            )
        object.__setattr__(self, "bulk_defects", species)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        for species in self.bulk_defects:
            species.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect_model": self.defect_model,
            "bulk_defects": [value.to_dict() for value in self.bulk_defects],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "multivalent bulk-defect document must be a mapping"
            )
        _require_exact_keys(
            value,
            {"schema_version", "defect_model", "bulk_defects"},
            "multivalent bulk-defect document",
        )
        raw = value["bulk_defects"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ExplicitDefectSchemaError("bulk_defects must be a list")
        return cls(
            schema_version=value["schema_version"],
            defect_model=value["defect_model"],
            bulk_defects=tuple(
                MultivalentBulkDefectSpecies.from_dict(item) for item in raw
            ),
        )


def _coerce_yaml_scalars(value: Any) -> Any:
    """Normalize YAML 1.1 numeric scalars at the layer-mapping boundary.

    PyYAML's 1.1 resolver leaves ``2.0e21`` (no exponent sign) as a string, so
    every standard config in this repository relies on the loader coercing
    such scalars. The canonical dataclasses stay strict for API callers; the
    coercion belongs here, at the text boundary, so a v4 layer does not have
    to be written in a different numeric style than every other layer.
    """

    if isinstance(value, Mapping):
        return {key: _coerce_yaml_scalars(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_coerce_yaml_scalars(item) for item in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def multivalent_bulk_defect_document_from_layer_mapping(
    layer: Mapping[str, Any],
) -> MultivalentBulkDefectDocument | None:
    """Parse the flat standard-layer representation of a canonical v4 document."""

    keys = {"defect_schema_version", "defect_model", "bulk_defects"}
    present = keys.intersection(layer)
    if not present:
        return None
    if present != keys:
        raise ExplicitDefectSchemaError(
            "explicit defect metadata must declare defect_schema_version, "
            f"defect_model, and bulk_defects together; missing={sorted(keys - present)}"
        )
    if layer["defect_schema_version"] != MULTIVALENT_DEFECT_SCHEMA_VERSION:
        raise ExplicitDefectSchemaError(
            "multivalent layer parser requires the canonical v4 schema"
        )
    return MultivalentBulkDefectDocument.from_dict(
        {
            "schema_version": layer["defect_schema_version"],
            "defect_model": layer["defect_model"],
            "bulk_defects": _coerce_yaml_scalars(layer["bulk_defects"]),
        }
    )


@dataclass(frozen=True, slots=True)
class MetastableConversionKinetics:
    """Resolved double-carrier conversion barriers and rate prefactors."""

    transition_energy_eV_above_vb: float
    electron_capture_activation_eV: float
    electron_emission_activation_eV: float
    hole_capture_activation_eV: float
    hole_emission_activation_eV: float
    electron_capture_path: Literal[
        "double_electron_capture",
        "electron_capture_plus_hole_emission",
    ]
    hole_capture_path: Literal[
        "double_hole_capture",
        "hole_capture_plus_electron_emission",
    ]
    capture_n_m3_s: float
    capture_p_m3_s: float
    phonon_frequency_Hz: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transition_energy_eV_above_vb",
            _nonnegative(
                self.transition_energy_eV_above_vb,
                "transition_energy_eV_above_vb",
            ),
        )
        for field in (
            "electron_capture_activation_eV",
            "electron_emission_activation_eV",
            "hole_capture_activation_eV",
            "hole_emission_activation_eV",
        ):
            object.__setattr__(self, field, _nonnegative(getattr(self, field), field))
        if self.electron_capture_path not in _ELECTRON_CAPTURE_PATHS:
            raise ExplicitDefectSchemaError(
                f"unsupported electron_capture_path {self.electron_capture_path!r}"
            )
        if self.hole_capture_path not in _HOLE_CAPTURE_PATHS:
            raise ExplicitDefectSchemaError(
                f"unsupported hole_capture_path {self.hole_capture_path!r}"
            )
        for field in (
            "capture_n_m3_s",
            "capture_p_m3_s",
            "phonon_frequency_Hz",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))

    def validate_detailed_balance(self, band_gap_eV: object) -> None:
        """Require the four barriers to meet at the declared transition energy."""

        gap = _positive(band_gap_eV, "band_gap_eV")
        transition = self.transition_energy_eV_above_vb
        if transition > gap:
            raise ExplicitDefectSchemaError(
                "metastable transition energy must lie inside the band gap"
            )
        if self.electron_capture_path == DOUBLE_ELECTRON_CAPTURE:
            expected_ee = self.electron_capture_activation_eV + 2.0 * (gap - transition)
        else:
            expected_ee = self.electron_capture_activation_eV + gap - 2.0 * transition
        if self.hole_capture_path == DOUBLE_HOLE_CAPTURE:
            expected_he = self.hole_capture_activation_eV + 2.0 * transition
        else:
            expected_he = self.hole_capture_activation_eV + 2.0 * transition - gap
        scale = max(
            gap,
            transition,
            abs(expected_ee),
            abs(expected_he),
            self.electron_emission_activation_eV,
            self.hole_emission_activation_eV,
            1.0,
        )
        tolerance = 64.0 * math.ulp(scale)
        if not math.isclose(
            self.electron_emission_activation_eV,
            expected_ee,
            rel_tol=0.0,
            abs_tol=tolerance,
        ) or not math.isclose(
            self.hole_emission_activation_eV,
            expected_he,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ExplicitDefectSchemaError(
                "metastable activation barriers violate detailed balance at "
                "the declared transition energy"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in (
                "transition_energy_eV_above_vb",
                "electron_capture_activation_eV",
                "electron_emission_activation_eV",
                "hole_capture_activation_eV",
                "hole_emission_activation_eV",
                "electron_capture_path",
                "hole_capture_path",
                "capture_n_m3_s",
                "capture_p_m3_s",
                "phonon_frequency_Hz",
            )
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "metastable conversion kinetics must be a mapping"
            )
        expected = {
            "transition_energy_eV_above_vb",
            "electron_capture_activation_eV",
            "electron_emission_activation_eV",
            "hole_capture_activation_eV",
            "hole_emission_activation_eV",
            "electron_capture_path",
            "hole_capture_path",
            "capture_n_m3_s",
            "capture_p_m3_s",
            "phonon_frequency_Hz",
        }
        _require_exact_keys(value, expected, "metastable conversion kinetics")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class MetastableDefectDefinition:
    """Two conventional configurations joined by a double-carrier process."""

    name: str
    total_density_m3: float
    donor_configuration: MultivalentDefectConfiguration
    acceptor_configuration: MultivalentDefectConfiguration
    donor_conversion_state_index: int
    acceptor_conversion_state_index: int
    conversion_kinetics: MetastableConversionKinetics

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "metastable name"))
        object.__setattr__(
            self,
            "total_density_m3",
            _positive(self.total_density_m3, "total_density_m3"),
        )
        if not isinstance(
            self.donor_configuration, MultivalentDefectConfiguration
        ) or not isinstance(
            self.acceptor_configuration, MultivalentDefectConfiguration
        ):
            raise TypeError(
                "metastable configurations must be multivalent configurations"
            )
        donor_index = _integer(
            self.donor_conversion_state_index,
            "donor_conversion_state_index",
        )
        acceptor_index = _integer(
            self.acceptor_conversion_state_index,
            "acceptor_conversion_state_index",
        )
        if donor_index >= len(self.donor_configuration.charge_states_e):
            raise ExplicitDefectSchemaError(
                "donor_conversion_state_index is outside the configuration"
            )
        if acceptor_index >= len(self.acceptor_configuration.charge_states_e):
            raise ExplicitDefectSchemaError(
                "acceptor_conversion_state_index is outside the configuration"
            )
        donor_charge = self.donor_configuration.charge_states_e[donor_index]
        acceptor_charge = self.acceptor_configuration.charge_states_e[acceptor_index]
        if donor_charge - acceptor_charge != 2:
            raise ExplicitDefectSchemaError(
                "metastable conversion states must differ by exactly two "
                "elementary charges"
            )
        if not isinstance(self.conversion_kinetics, MetastableConversionKinetics):
            raise TypeError("conversion_kinetics must be MetastableConversionKinetics")
        object.__setattr__(self, "donor_conversion_state_index", donor_index)
        object.__setattr__(self, "acceptor_conversion_state_index", acceptor_index)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        self.donor_configuration.validate_band_gap(band_gap_eV)
        self.acceptor_configuration.validate_band_gap(band_gap_eV)
        self.conversion_kinetics.validate_detailed_balance(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_density_m3": self.total_density_m3,
            "donor_configuration": self.donor_configuration.to_dict(),
            "acceptor_configuration": self.acceptor_configuration.to_dict(),
            "donor_conversion_state_index": self.donor_conversion_state_index,
            "acceptor_conversion_state_index": (self.acceptor_conversion_state_index),
            "conversion_kinetics": self.conversion_kinetics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "metastable defect definition must be a mapping"
            )
        _require_exact_keys(
            value,
            {
                "name",
                "total_density_m3",
                "donor_configuration",
                "acceptor_configuration",
                "donor_conversion_state_index",
                "acceptor_conversion_state_index",
                "conversion_kinetics",
            },
            "metastable defect definition",
        )
        return cls(
            name=value["name"],
            total_density_m3=value["total_density_m3"],
            donor_configuration=MultivalentDefectConfiguration.from_dict(
                value["donor_configuration"]
            ),
            acceptor_configuration=MultivalentDefectConfiguration.from_dict(
                value["acceptor_configuration"]
            ),
            donor_conversion_state_index=value["donor_conversion_state_index"],
            acceptor_conversion_state_index=value["acceptor_conversion_state_index"],
            conversion_kinetics=MetastableConversionKinetics.from_dict(
                value["conversion_kinetics"]
            ),
        )


@dataclass(frozen=True, slots=True)
class MetastableDefectDocument:
    """Canonical inventory of frozen-measurement metastable defects."""

    schema_version: Literal["solarlab-metastable-bulk-defects-v1"]
    defect_model: Literal["explicit_metastable_frozen"]
    metastable_defects: tuple[MetastableDefectDefinition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != METASTABLE_DEFECT_SCHEMA_VERSION:
            raise ExplicitDefectSchemaError(
                "unsupported metastable defect schema_version"
            )
        if self.defect_model != EXPLICIT_METASTABLE_FROZEN:
            raise ExplicitDefectSchemaError(
                "metastable v1 requires explicit_metastable_frozen"
            )
        defects = tuple(self.metastable_defects)
        if not defects or not all(
            isinstance(value, MetastableDefectDefinition) for value in defects
        ):
            raise ExplicitDefectSchemaError(
                "metastable_defects must contain at least one definition"
            )
        names = [value.name for value in defects]
        if len(names) != len(set(names)):
            raise ExplicitDefectSchemaError("metastable defect names must be unique")
        object.__setattr__(self, "metastable_defects", defects)

    def validate_band_gap(self, band_gap_eV: object) -> None:
        for defect in self.metastable_defects:
            defect.validate_band_gap(band_gap_eV)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect_model": self.defect_model,
            "metastable_defects": [
                value.to_dict() for value in self.metastable_defects
            ],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "metastable defect document must be a mapping"
            )
        _require_exact_keys(
            value,
            {"schema_version", "defect_model", "metastable_defects"},
            "metastable defect document",
        )
        raw = value["metastable_defects"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ExplicitDefectSchemaError("metastable_defects must be a list")
        return cls(
            schema_version=value["schema_version"],
            defect_model=value["defect_model"],
            metastable_defects=tuple(
                MetastableDefectDefinition.from_dict(item) for item in raw
            ),
        )


@dataclass(frozen=True, slots=True)
class MetastablePreparationNumerics:
    """Replayable nonlinear controls for the stationary initial state."""

    initial_donor_fraction_guess: float
    max_iterations: int
    relative_tolerance: float
    clamping_factor: float
    final_unclamped_refinement: Literal[True] = True

    def __post_init__(self) -> None:
        guess = _finite(
            self.initial_donor_fraction_guess,
            "initial_donor_fraction_guess",
        )
        if not 0.0 <= guess <= 1.0:
            raise ExplicitDefectSchemaError(
                "initial_donor_fraction_guess must lie in [0, 1]"
            )
        clamping = _positive(self.clamping_factor, "clamping_factor")
        if clamping > 1.0:
            raise ExplicitDefectSchemaError("clamping_factor must be <= 1")
        if self.final_unclamped_refinement is not True:
            raise ExplicitDefectSchemaError(
                "a clamped iterate cannot be accepted without final "
                "unclamped refinement"
            )
        object.__setattr__(self, "initial_donor_fraction_guess", guess)
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", minimum=1),
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            _positive(self.relative_tolerance, "relative_tolerance"),
        )
        object.__setattr__(self, "clamping_factor", clamping)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_donor_fraction_guess": self.initial_donor_fraction_guess,
            "max_iterations": self.max_iterations,
            "relative_tolerance": self.relative_tolerance,
            "clamping_factor": self.clamping_factor,
            "final_unclamped_refinement": self.final_unclamped_refinement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "metastable preparation numerics must be a mapping"
            )
        expected = {
            "initial_donor_fraction_guess",
            "max_iterations",
            "relative_tolerance",
            "clamping_factor",
            "final_unclamped_refinement",
        }
        _require_exact_keys(value, expected, "metastable preparation numerics")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class MetastablePreparationProtocol:
    """SCAPS-like initial working point followed by a frozen measurement."""

    schema_version: Literal["solarlab-metastable-preparation-v1"]
    preparation_limit: Literal["stationary_infinite_time"]
    preparation_temperature_K: float
    preparation_voltage_V: float
    preparation_illumination_suns: float
    voltage_continuation_steps: int
    illumination_continuation_steps: int
    measurement_temperature_K: float
    configuration_freeze_stage: Literal[
        "after_stationary_preparation_before_measurement"
    ]
    freeze_configuration_during_measurement: Literal[True]
    measurement_protocol_sha256: str
    numerics: MetastablePreparationNumerics

    def __post_init__(self) -> None:
        if self.schema_version != METASTABLE_PREPARATION_SCHEMA_VERSION:
            raise ExplicitDefectSchemaError(
                "unsupported metastable preparation schema_version"
            )
        if self.preparation_limit != STATIONARY_INFINITE_TIME:
            raise ExplicitDefectSchemaError(
                "metastable preparation v1 supports only the stationary "
                "infinite-time SCAPS-like limit"
            )
        if self.configuration_freeze_stage != FROZEN_BEFORE_MEASUREMENT:
            raise ExplicitDefectSchemaError(
                "metastable configuration must be frozen after preparation "
                "and before measurement"
            )
        if self.freeze_configuration_during_measurement is not True:
            raise ExplicitDefectSchemaError(
                "metastable preparation v1 requires a frozen measurement state"
            )
        for field in (
            "preparation_temperature_K",
            "measurement_temperature_K",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        object.__setattr__(
            self,
            "preparation_voltage_V",
            _finite(self.preparation_voltage_V, "preparation_voltage_V"),
        )
        object.__setattr__(
            self,
            "preparation_illumination_suns",
            _nonnegative(
                self.preparation_illumination_suns,
                "preparation_illumination_suns",
            ),
        )
        for field in (
            "voltage_continuation_steps",
            "illumination_continuation_steps",
        ):
            object.__setattr__(
                self, field, _integer(getattr(self, field), field, minimum=0)
            )
        object.__setattr__(
            self,
            "measurement_protocol_sha256",
            _validate_sha256(
                self.measurement_protocol_sha256,
                "measurement_protocol_sha256",
            ),
        )
        if not isinstance(self.numerics, MetastablePreparationNumerics):
            raise TypeError("numerics must be MetastablePreparationNumerics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preparation_limit": self.preparation_limit,
            "preparation_temperature_K": self.preparation_temperature_K,
            "preparation_voltage_V": self.preparation_voltage_V,
            "preparation_illumination_suns": (self.preparation_illumination_suns),
            "voltage_continuation_steps": self.voltage_continuation_steps,
            "illumination_continuation_steps": (self.illumination_continuation_steps),
            "measurement_temperature_K": self.measurement_temperature_K,
            "configuration_freeze_stage": self.configuration_freeze_stage,
            "freeze_configuration_during_measurement": (
                self.freeze_configuration_during_measurement
            ),
            "measurement_protocol_sha256": self.measurement_protocol_sha256,
            "numerics": self.numerics.to_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ExplicitDefectSchemaError(
                "metastable preparation protocol must be a mapping"
            )
        expected = {
            "schema_version",
            "preparation_limit",
            "preparation_temperature_K",
            "preparation_voltage_V",
            "preparation_illumination_suns",
            "voltage_continuation_steps",
            "illumination_continuation_steps",
            "measurement_temperature_K",
            "configuration_freeze_stage",
            "freeze_configuration_during_measurement",
            "measurement_protocol_sha256",
            "numerics",
        }
        _require_exact_keys(value, expected, "metastable preparation protocol")
        data = dict(value)
        data["numerics"] = MetastablePreparationNumerics.from_dict(data["numerics"])
        return cls(**data)


__all__ = [
    "AMPHOTERIC",
    "CUSTOM_MULTILEVEL",
    "DOUBLE_ACCEPTOR",
    "DOUBLE_DONOR",
    "DOUBLE_ELECTRON_CAPTURE",
    "DOUBLE_HOLE_CAPTURE",
    "ELECTRON_CAPTURE_HOLE_EMISSION",
    "EXPLICIT",
    "EXPLICIT_METASTABLE_FROZEN",
    "FROZEN_BEFORE_MEASUREMENT",
    "HOLE_CAPTURE_ELECTRON_EMISSION",
    "METASTABLE_DEFECT_SCHEMA_VERSION",
    "METASTABLE_PREPARATION_SCHEMA_VERSION",
    "MULTIVALENT_DEFECT_SCHEMA_VERSION",
    "SCAPS_BINOMIAL",
    "SINGLE_ACCEPTOR",
    "SINGLE_DONOR",
    "STATIONARY_INFINITE_TIME",
    "UNITY",
    "MetastableConversionKinetics",
    "MetastableDefectDefinition",
    "MetastableDefectDocument",
    "MetastablePreparationNumerics",
    "MetastablePreparationProtocol",
    "MultivalentBulkDefectDocument",
    "MultivalentBulkDefectSpecies",
    "MultivalentDefectConfiguration",
    "MultivalentEnergyLevels",
    "multivalent_bulk_defect_document_from_layer_mapping",
]

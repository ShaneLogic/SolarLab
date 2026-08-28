"""Strict SCAPS-shaped conversion for distributed bulk defects.

This adapter is intentionally separate from the legacy SCAPS YAML loader.
The legacy loader treats ``N_t_cm3`` as an already integrated density and
retains ambiguous Gaussian metadata without executing it.  This module only
accepts dimensionally explicit v2 inputs and produces a fully normalized
canonical species plus auditable conversion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
from typing import Any

from perovskite_sim.models.defects import (
    CONDUCTION_BAND_TAIL,
    ENERGY_ABOVE_VALENCE_BAND,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
    ExplicitDefectSchemaError,
)
from perovskite_sim.physics.defect_distributions import (
    distribution_shape_integral_eV,
    peak_density_from_integrated_density,
)
from perovskite_sim.physics.temperature import thermal_voltage


SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION = (
    "scaps-distributed-bulk-defect-v1"
)
SCAPS_DENSITY_RELATIVE_TOLERANCE = 1.0e-12

SCAPS_ENERGY_ABOVE_VALENCE_BAND = "above_valence_band"
SCAPS_ENERGY_BELOW_CONDUCTION_BAND = "below_conduction_band"
SCAPS_ENERGY_ABOVE_INTRINSIC_LEVEL = "above_intrinsic_level"

_DISTRIBUTIONS = frozenset(
    {
        GAUSSIAN,
        UNIFORM,
        CONDUCTION_BAND_TAIL,
        VALENCE_BAND_TAIL,
    }
)
_ENERGY_FIELD_TO_REFERENCE = {
    "E_t_eV_above_vb": SCAPS_ENERGY_ABOVE_VALENCE_BAND,
    "E_t_eV_below_cb": SCAPS_ENERGY_BELOW_CONDUCTION_BAND,
    "E_t_eV_above_intrinsic": SCAPS_ENERGY_ABOVE_INTRINSIC_LEVEL,
}
_DENSITY_FIELDS = frozenset({"N_total_cm3", "N_peak_cm3_eV"})
_REQUIRED_FIELDS = frozenset(
    {
        "name",
        "distribution",
        "sigma_n_cm2",
        "sigma_p_cm2",
        "charge_transition",
        "neutral_reference",
        "E_char_eV",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "support_width_multiplier",
        "degeneracy",
        *_ENERGY_FIELD_TO_REFERENCE,
        *_DENSITY_FIELDS,
    }
)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ExplicitDefectSchemaError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExplicitDefectSchemaError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ExplicitDefectSchemaError(f"{field} must be finite")
    return number


def _finite_positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and positive"
        )
    return number


def _finite_nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ExplicitDefectSchemaError(
            f"{field} must be finite and non-negative"
        )
    return number


def _decimal_scale(value: object, factor: str, field: str) -> float:
    number = _finite_nonnegative(value, field)
    return float(Decimal(str(number)) * Decimal(factor))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScapsDistributedDefectConversion:
    """Canonical species and complete evidence for one SCAPS conversion."""

    conversion_identity_sha256: str
    source_distribution_kind: str
    source_energy_reference: str
    source_energy_value_eV: float
    source_sigma_n_cm2: float
    source_sigma_p_cm2: float
    source_total_density_cm3: float | None
    source_peak_density_cm3_eV: float | None
    source_density_mode: str
    shape_integral_eV: float
    resolved_total_density_m3: float
    resolved_peak_density_m3_eV: float
    density_relative_mismatch: float | None
    band_gap_eV: float
    temperature_K: float
    effective_conduction_dos_cm3: float
    effective_valence_dos_cm3: float
    intrinsic_level_eV_above_vb: float
    layer_thermal_velocity_cm_s: float
    species: BulkDefectSpecies

    def __post_init__(self) -> None:
        digest = str(self.conversion_identity_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "conversion_identity_sha256 must be a SHA-256 hex"
            )
        object.__setattr__(self, "conversion_identity_sha256", digest)
        if not isinstance(self.species, BulkDefectSpecies):
            raise TypeError("species must be a BulkDefectSpecies")
        if self.source_distribution_kind not in _DISTRIBUTIONS:
            raise ValueError("source distribution kind is invalid")
        if (
            self.species.distribution.kind
            != self.source_distribution_kind
        ):
            raise ValueError("source and canonical distribution kinds differ")
        distribution = self.species.distribution
        expected_width_convention = (
            WIDTH_UNIFORM_FULL
            if self.source_distribution_kind == UNIFORM
            else WIDTH_SCAPS_CHARACTERISTIC
        )
        if (
            distribution.energy_reference != ENERGY_ABOVE_VALENCE_BAND
            or distribution.width_convention != expected_width_convention
            or self.species.degeneracy != 1.0
        ):
            raise ValueError("canonical SCAPS distribution contract is invalid")
        self.species.validate_band_gap(self.band_gap_eV)
        if self.source_energy_reference not in set(
            _ENERGY_FIELD_TO_REFERENCE.values()
        ):
            raise ValueError("source energy reference is invalid")
        if self.source_density_mode not in {
            "integrated_total",
            "peak_density",
            "integrated_total_and_peak_density",
        }:
            raise ValueError("source density mode is invalid")

        for field in (
            "shape_integral_eV",
            "resolved_total_density_m3",
            "resolved_peak_density_m3_eV",
            "band_gap_eV",
            "temperature_K",
            "effective_conduction_dos_cm3",
            "effective_valence_dos_cm3",
            "layer_thermal_velocity_cm_s",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        for field in ("source_sigma_n_cm2", "source_sigma_p_cm2"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.source_sigma_n_cm2 == self.source_sigma_p_cm2 == 0.0:
            raise ValueError("at least one capture cross section must be positive")
        for field in (
            "source_energy_value_eV",
            "intrinsic_level_eV_above_vb",
        ):
            if not math.isfinite(float(getattr(self, field))):
                raise ValueError(f"{field} must be finite")
        for field in (
            "source_total_density_cm3",
            "source_peak_density_cm3_eV",
        ):
            value = getattr(self, field)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be null or positive")
        has_total = self.source_total_density_cm3 is not None
        has_peak = self.source_peak_density_cm3_eV is not None
        if not has_total and not has_peak:
            raise ValueError("conversion evidence requires a source density")
        if not 0.0 < self.intrinsic_level_eV_above_vb < self.band_gap_eV:
            raise ValueError("intrinsic level must lie inside the band gap")

        expected_intrinsic = (
            0.5 * self.band_gap_eV
            + 0.5
            * thermal_voltage(self.temperature_K)
            * math.log(
                self.effective_valence_dos_cm3
                / self.effective_conduction_dos_cm3
            )
        )
        if self.intrinsic_level_eV_above_vb != expected_intrinsic:
            raise ValueError("intrinsic-level conversion evidence is invalid")
        if self.source_energy_reference == SCAPS_ENERGY_ABOVE_VALENCE_BAND:
            if self.source_energy_value_eV < 0.0:
                raise ValueError("above-VB source energy must be non-negative")
            expected_center = self.source_energy_value_eV
        elif (
            self.source_energy_reference
            == SCAPS_ENERGY_BELOW_CONDUCTION_BAND
        ):
            if self.source_energy_value_eV < 0.0:
                raise ValueError("below-CB source energy must be non-negative")
            expected_center = float(
                Decimal(str(self.band_gap_eV))
                - Decimal(str(self.source_energy_value_eV))
            )
        else:
            expected_center = (
                self.intrinsic_level_eV_above_vb
                + self.source_energy_value_eV
            )
        if (
            self.species.distribution.center_eV_above_vb
            != expected_center
        ):
            raise ValueError("energy-reference conversion evidence is invalid")

        expected_integral = distribution_shape_integral_eV(
            self.species.distribution
        )
        expected_peak = peak_density_from_integrated_density(
            self.species.distribution
        )
        if (
            self.shape_integral_eV != expected_integral
            or self.resolved_total_density_m3
            != self.species.distribution.total_density_m3
            or self.resolved_peak_density_m3_eV != expected_peak
        ):
            raise ValueError("conversion evidence is not canonical")

        expected_mode = (
            "integrated_total_and_peak_density"
            if has_total and has_peak
            else "integrated_total"
            if has_total
            else "peak_density"
        )
        if self.source_density_mode != expected_mode:
            raise ValueError("source density mode does not match its fields")
        source_total_m3 = (
            float(
                Decimal(str(self.source_total_density_cm3))
                * Decimal("1e6")
            )
            if has_total
            else None
        )
        source_peak_m3_eV = (
            float(
                Decimal(str(self.source_peak_density_cm3_eV))
                * Decimal("1e6")
            )
            if has_peak
            else None
        )
        peak_implied_total = (
            source_peak_m3_eV * expected_integral
            if source_peak_m3_eV is not None
            else None
        )
        expected_total = (
            source_total_m3
            if source_total_m3 is not None
            else peak_implied_total
        )
        if self.resolved_total_density_m3 != expected_total:
            raise ValueError("source density conversion evidence is invalid")

        mismatch = self.density_relative_mismatch
        both = has_total and has_peak
        if both:
            expected_mismatch = abs(
                source_total_m3 - peak_implied_total
            ) / max(abs(source_total_m3), abs(peak_implied_total))
            if (
                mismatch is None
                or not math.isfinite(float(mismatch))
                or float(mismatch) < 0.0
                or float(mismatch) > SCAPS_DENSITY_RELATIVE_TOLERANCE
                or float(mismatch) != expected_mismatch
            ):
                raise ValueError("density consistency evidence is invalid")
        elif mismatch is not None:
            raise ValueError(
                "density mismatch is only defined when both densities exist"
            )
        expected_sigma_n = float(
            Decimal(str(self.source_sigma_n_cm2)) * Decimal("1e-4")
        )
        expected_sigma_p = float(
            Decimal(str(self.source_sigma_p_cm2)) * Decimal("1e-4")
        )
        expected_velocity = float(
            Decimal(str(self.layer_thermal_velocity_cm_s)) * Decimal("1e-2")
        )
        if (
            self.species.kinetics.sigma_n_m2 != expected_sigma_n
            or self.species.kinetics.sigma_p_m2 != expected_sigma_p
            or self.species.kinetics.thermal_velocity_n_m_s
            != expected_velocity
            or self.species.kinetics.thermal_velocity_p_m_s
            != expected_velocity
        ):
            raise ValueError("kinetic unit-conversion evidence is invalid")
        if _canonical_sha256(self._unsigned_dict()) != digest:
            raise ValueError("conversion identity is inconsistent")

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "adapter": SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION,
            "source_distribution_kind": self.source_distribution_kind,
            "source_energy_reference": self.source_energy_reference,
            "source_energy_value_eV": self.source_energy_value_eV,
            "source_sigma_n_cm2": self.source_sigma_n_cm2,
            "source_sigma_p_cm2": self.source_sigma_p_cm2,
            "source_total_density_cm3": self.source_total_density_cm3,
            "source_peak_density_cm3_eV": (
                self.source_peak_density_cm3_eV
            ),
            "source_density_mode": self.source_density_mode,
            "shape_integral_eV": self.shape_integral_eV,
            "resolved_total_density_m3": self.resolved_total_density_m3,
            "resolved_peak_density_m3_eV": (
                self.resolved_peak_density_m3_eV
            ),
            "density_relative_mismatch": self.density_relative_mismatch,
            "band_gap_eV": self.band_gap_eV,
            "temperature_K": self.temperature_K,
            "effective_conduction_dos_cm3": (
                self.effective_conduction_dos_cm3
            ),
            "effective_valence_dos_cm3": self.effective_valence_dos_cm3,
            "intrinsic_level_eV_above_vb": (
                self.intrinsic_level_eV_above_vb
            ),
            "layer_thermal_velocity_cm_s": (
                self.layer_thermal_velocity_cm_s
            ),
            "species": self.species.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "conversion_identity_sha256": self.conversion_identity_sha256,
        }


def convert_scaps_distributed_bulk_defect(
    value: Mapping[str, Any],
    *,
    band_gap_eV: object,
    temperature_K: object,
    effective_conduction_dos_cm3: object,
    effective_valence_dos_cm3: object,
    layer_thermal_velocity_cm_s: object,
    where: str,
) -> ScapsDistributedDefectConversion:
    """Convert one dimensionally explicit SCAPS-cgs distributed defect.

    Exactly one energy reference and at least one density representation are
    required.  When both density forms are supplied, their finite-support
    analytic conversion must agree within the frozen relative tolerance.
    """

    if not isinstance(value, Mapping):
        raise ExplicitDefectSchemaError(f"{where} must be a mapping")
    actual = set(value)
    missing = sorted(_REQUIRED_FIELDS - actual)
    unknown = sorted(
        str(key) for key in actual - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    )
    if missing or unknown:
        raise ExplicitDefectSchemaError(
            f"{where} schema mismatch: missing={missing}, unknown={unknown}"
        )

    energy_fields = [field for field in _ENERGY_FIELD_TO_REFERENCE if field in value]
    if len(energy_fields) != 1:
        raise ExplicitDefectSchemaError(
            f"{where}: exactly one of {sorted(_ENERGY_FIELD_TO_REFERENCE)} "
            "is required"
        )
    density_fields = [field for field in _DENSITY_FIELDS if field in value]
    if not density_fields:
        raise ExplicitDefectSchemaError(
            f"{where}: at least one of {sorted(_DENSITY_FIELDS)} is required"
        )

    distribution_kind = str(value["distribution"]).strip().lower()
    if distribution_kind not in _DISTRIBUTIONS:
        raise ExplicitDefectSchemaError(
            f"{where}.distribution must be one of {sorted(_DISTRIBUTIONS)}"
        )
    support_field = "support_width_multiplier"
    if distribution_kind == UNIFORM:
        if support_field in value:
            raise ExplicitDefectSchemaError(
                f"{where}: uniform distribution forbids {support_field}"
            )
        support_multiplier = None
        width_convention = WIDTH_UNIFORM_FULL
    else:
        if support_field not in value:
            raise ExplicitDefectSchemaError(
                f"{where}: {distribution_kind} requires {support_field}"
            )
        support_multiplier = _finite_positive(
            value[support_field],
            f"{where}.{support_field}",
        )
        width_convention = WIDTH_SCAPS_CHARACTERISTIC

    degeneracy = _finite_positive(
        value.get("degeneracy", 1.0),
        f"{where}.degeneracy",
    )
    if degeneracy != 1.0:
        raise ExplicitDefectCapabilityError(
            f"{where}: D3-E2 supports only degeneracy=1.0"
        )

    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    temperature = _finite_positive(temperature_K, "temperature_K")
    conduction_dos = _finite_positive(
        effective_conduction_dos_cm3,
        "effective_conduction_dos_cm3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_cm3,
        "effective_valence_dos_cm3",
    )
    velocity_cm_s = _finite_positive(
        layer_thermal_velocity_cm_s,
        "layer_thermal_velocity_cm_s",
    )
    intrinsic_level = (
        0.5 * gap
        + 0.5
        * thermal_voltage(temperature)
        * math.log(valence_dos / conduction_dos)
    )
    if not 0.0 < intrinsic_level < gap:
        raise ExplicitDefectSchemaError(
            "layer intrinsic level lies outside the declared band gap"
        )

    energy_field = energy_fields[0]
    source_energy = _finite(value[energy_field], f"{where}.{energy_field}")
    energy_reference = _ENERGY_FIELD_TO_REFERENCE[energy_field]
    if energy_reference == SCAPS_ENERGY_ABOVE_VALENCE_BAND:
        if source_energy < 0.0:
            raise ExplicitDefectSchemaError(
                f"{where}.{energy_field} must be non-negative"
            )
        center = source_energy
    elif energy_reference == SCAPS_ENERGY_BELOW_CONDUCTION_BAND:
        if source_energy < 0.0:
            raise ExplicitDefectSchemaError(
                f"{where}.{energy_field} must be non-negative"
            )
        center = float(
            Decimal(str(gap)) - Decimal(str(source_energy))
        )
    else:
        center = intrinsic_level + source_energy

    width = _finite_positive(value["E_char_eV"], f"{where}.E_char_eV")
    sigma_n_cm2 = _finite_nonnegative(
        value["sigma_n_cm2"],
        f"{where}.sigma_n_cm2",
    )
    sigma_p_cm2 = _finite_nonnegative(
        value["sigma_p_cm2"],
        f"{where}.sigma_p_cm2",
    )
    if sigma_n_cm2 == sigma_p_cm2 == 0.0:
        raise ExplicitDefectCapabilityError(
            f"{where}: at least one capture cross section must be positive"
        )
    source_total = (
        _finite_positive(value["N_total_cm3"], f"{where}.N_total_cm3")
        if "N_total_cm3" in value
        else None
    )
    source_peak = (
        _finite_positive(
            value["N_peak_cm3_eV"],
            f"{where}.N_peak_cm3_eV",
        )
        if "N_peak_cm3_eV" in value
        else None
    )
    total_m3 = (
        _decimal_scale(source_total, "1e6", f"{where}.N_total_cm3")
        if source_total is not None
        else None
    )
    peak_m3_eV = (
        _decimal_scale(source_peak, "1e6", f"{where}.N_peak_cm3_eV")
        if source_peak is not None
        else None
    )

    provisional = BulkDefectDistribution(
        kind=distribution_kind,
        normalization=INTEGRATED_TOTAL,
        total_density_m3=total_m3 if total_m3 is not None else 1.0,
        center_eV_above_vb=center,
        width_eV=width,
        width_convention=width_convention,
        energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        support_width_multiplier=support_multiplier,
    )
    provisional.validate_band_gap(gap)
    shape_integral = distribution_shape_integral_eV(provisional)
    implied_total_m3 = (
        peak_m3_eV * shape_integral
        if peak_m3_eV is not None
        else None
    )
    mismatch = None
    if total_m3 is not None and implied_total_m3 is not None:
        mismatch = abs(total_m3 - implied_total_m3) / max(
            abs(total_m3),
            abs(implied_total_m3),
        )
        if not math.isclose(
            total_m3,
            implied_total_m3,
            rel_tol=SCAPS_DENSITY_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        ):
            raise ExplicitDefectSchemaError(
                f"{where}: N_total_cm3 and N_peak_cm3_eV are "
                "inconsistent with E_char_eV/support; "
                f"relative_mismatch={mismatch:.6e}, "
                f"limit={SCAPS_DENSITY_RELATIVE_TOLERANCE:.6e}"
            )
    resolved_total = (
        total_m3 if total_m3 is not None else implied_total_m3
    )
    if resolved_total is None:
        raise AssertionError("density presence was validated above")
    distribution = replace(provisional, total_density_m3=resolved_total)

    velocity_m_s = _decimal_scale(
        velocity_cm_s,
        "1e-2",
        "layer_thermal_velocity_cm_s",
    )
    species = BulkDefectSpecies(
        name=value["name"],
        distribution=distribution,
        charge_transition=value["charge_transition"],
        neutral_reference=value["neutral_reference"],
        kinetics=BulkDefectKinetics(
            sigma_n_m2=_decimal_scale(
                sigma_n_cm2,
                "1e-4",
                f"{where}.sigma_n_cm2",
            ),
            sigma_p_m2=_decimal_scale(
                sigma_p_cm2,
                "1e-4",
                f"{where}.sigma_p_cm2",
            ),
            thermal_velocity_n_m_s=velocity_m_s,
            thermal_velocity_p_m_s=velocity_m_s,
        ),
        degeneracy=degeneracy,
    )
    species.validate_band_gap(gap)

    source_density_mode = (
        "integrated_total_and_peak_density"
        if source_total is not None and source_peak is not None
        else "integrated_total"
        if source_total is not None
        else "peak_density"
    )
    values: dict[str, object] = {
        "source_distribution_kind": distribution_kind,
        "source_energy_reference": energy_reference,
        "source_energy_value_eV": source_energy,
        "source_sigma_n_cm2": sigma_n_cm2,
        "source_sigma_p_cm2": sigma_p_cm2,
        "source_total_density_cm3": source_total,
        "source_peak_density_cm3_eV": source_peak,
        "source_density_mode": source_density_mode,
        "shape_integral_eV": shape_integral,
        "resolved_total_density_m3": resolved_total,
        "resolved_peak_density_m3_eV": (
            peak_density_from_integrated_density(distribution)
        ),
        "density_relative_mismatch": mismatch,
        "band_gap_eV": gap,
        "temperature_K": temperature,
        "effective_conduction_dos_cm3": conduction_dos,
        "effective_valence_dos_cm3": valence_dos,
        "intrinsic_level_eV_above_vb": intrinsic_level,
        "layer_thermal_velocity_cm_s": velocity_cm_s,
        "species": species,
    }
    identity_payload = {
        "adapter": SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION,
        **{
            key: item.to_dict() if key == "species" else item
            for key, item in values.items()
        },
    }
    return ScapsDistributedDefectConversion(
        conversion_identity_sha256=_canonical_sha256(identity_payload),
        **values,
    )


__all__ = [
    "SCAPS_DENSITY_RELATIVE_TOLERANCE",
    "SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION",
    "SCAPS_ENERGY_ABOVE_INTRINSIC_LEVEL",
    "SCAPS_ENERGY_ABOVE_VALENCE_BAND",
    "SCAPS_ENERGY_BELOW_CONDUCTION_BAND",
    "ScapsDistributedDefectConversion",
    "convert_scaps_distributed_bulk_defect",
]

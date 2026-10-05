"""Typed defect inventories and preparation protocols, without a closure.

Density is per physical defect, integrated over energy. State populations,
quadrature, capture rates and stationary configurations are not evaluated here.
The two metastable documents describe preparation; they are not solved states.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from solarlab.materials.full_parameters import Nonempty, StableId, StructuredInput, quantity_field as q
from solarlab.materials.parameters import QuantityInput
from solarlab.units import normalize_quantity

MULTIVALENT_VERSION = "solarlab-explicit-bulk-defects-v4"
METASTABLE_VERSION = "solarlab-metastable-bulk-defects-v1"
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _number(value: object, unit: str = "1") -> float:
    return normalize_quantity(value, unit)


def _gap(value: object) -> float:
    gap = _number(value, "eV")
    if gap <= 0:
        raise ValueError("defect requires a positive band gap")
    return gap


class KineticsInput(StructuredInput):
    sigma_n_m2: QuantityInput = q("m^2", required=True)
    sigma_p_m2: QuantityInput = q("m^2", required=True)
    thermal_velocity_n_m_s: QuantityInput = q("m/s", positive=True, required=True)
    thermal_velocity_p_m_s: QuantityInput = q("m/s", positive=True, required=True)


class LegacyBulkTrapInput(StructuredInput):
    distribution: Literal["single_level", "gaussian"]
    total_density_m3: QuantityInput = q("m^-3", positive=True, required=True)
    center_eV_above_vb: QuantityInput = q("eV", required=True)
    energy_sigma_eV: QuantityInput | None = q("eV", positive=True)
    sigma_n_m2: QuantityInput = q("m^2", positive=True, required=True)
    sigma_p_m2: QuantityInput = q("m^2", positive=True, required=True)
    thermal_velocity_m_s: QuantityInput = q("m/s", positive=True, required=True)
    charge_transition: Literal["acceptor", "donor"]

    @model_validator(mode="after")
    def _width(self) -> LegacyBulkTrapInput:
        if self.distribution == "single_level" and "energy_sigma_eV" in self.model_fields_set:
            raise ValueError("single_level forbids energy_sigma_eV")
        if self.distribution == "gaussian" and self.energy_sigma_eV is None:
            raise ValueError("Gaussian requires positive energy_sigma_eV")
        return self

    def resolved_data(self, band_gap_eV: object) -> dict[str, Any]:
        data = self.normalized_data()
        gap = _gap(band_gap_eV)
        if data["center_eV_above_vb"] > gap:
            raise ValueError("bulk_trap_distribution center lies outside the band gap")
        return {**data, "band_gap_eV": gap, "density_normalization": "integrated_total",
                "density_unit": "m^-3", "continuous_density_unit": "m^-3/eV",
                "energy_reference": "above_valence_band"}


class MultivalentEnergyInput(StructuredInput):
    first_transition_eV_above_vb: QuantityInput = q("eV", required=True)
    correlation_energies_eV: tuple[QuantityInput, ...]
    energy_reference: Literal["above_valence_band"]

    @field_validator("correlation_energies_eV")
    @classmethod
    def _correlations(cls, values: tuple[QuantityInput, ...]) -> tuple[QuantityInput, ...]:
        for value in values:
            _number(value, "eV")  # Signed correlations are allowed.
        return values

    @property
    def transition_energies_eV(self) -> tuple[float, ...]:
        values = [_number(self.first_transition_eV_above_vb, "eV")]
        # Preserve the declared left-to-right correlation convention, including
        # its scalar rounding, rather than replacing it with a final sum.
        for correlation in self.correlation_energies_eV:
            values.append(values[-1] + _number(correlation, "eV"))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite cumulative transition energy")
        return tuple(values)

    def normalized_data(self) -> dict[str, Any]:
        return {**super().normalized_data(),
                "correlation_energies_eV": [_number(value, "eV") for value in self.correlation_energies_eV]}


class MultivalentConfigurationInput(StructuredInput):
    family: Literal["single_donor", "single_acceptor", "double_donor", "double_acceptor", "amphoteric", "custom_multilevel"]
    charge_states_e: tuple[int, ...]
    degeneracy_convention: Literal["scaps_binomial", "unity", "explicit"]
    state_degeneracies: tuple[QuantityInput, ...]
    energy_levels: MultivalentEnergyInput
    transition_kinetics: tuple[KineticsInput, ...]

    @model_validator(mode="after")
    def _transitions(self) -> MultivalentConfigurationInput:
        charges = self.charge_states_e
        if not 2 <= len(charges) <= 5 or any(type(value) is not int or not -3 <= value <= 3 for value in charges):
            raise ValueError("multivalent charge states require 2-5 integers in [-3,3]")
        if any(right != left - 1 for left, right in zip(charges, charges[1:])):
            raise ValueError("multivalent charge states must descend by one")
        named = {"single_donor": (1, 0), "single_acceptor": (0, -1),
                 "double_donor": (2, 1, 0), "double_acceptor": (0, -1, -2), "amphoteric": (1, 0, -1)}
        if self.family in named and charges != named[self.family]:
            raise ValueError("named family and charge states disagree")
        degeneracies = tuple(_number(value) for value in self.state_degeneracies)
        if len(degeneracies) != len(charges) or any(value <= 0 for value in degeneracies):
            raise ValueError("require one positive degeneracy per state")
        if self.degeneracy_convention == "unity" and any(value != 1 for value in degeneracies):
            raise ValueError("unity requires all state degeneracies equal to 1")
        if self.degeneracy_convention == "scaps_binomial" and degeneracies != tuple(math.comb(len(charges) - 1, i) for i in range(len(charges))):
            raise ValueError("scaps_binomial requires binomial state degeneracies")
        if len(self.energy_levels.transition_energies_eV) != len(charges) - 1 or len(self.transition_kinetics) != len(charges) - 1:
            raise ValueError("one energy and kinetics record is required per transition")
        if any(_number(kinetics.sigma_n_m2, "m^2") == 0 and _number(kinetics.sigma_p_m2, "m^2") == 0 for kinetics in self.transition_kinetics):
            raise ValueError("each transition requires at least one nonzero capture leg")
        return self

    def normalized_data(self) -> dict[str, Any]:
        return {**super().normalized_data(), "state_degeneracies": [_number(value) for value in self.state_degeneracies]}

    def resolved_data(self, band_gap_eV: object) -> dict[str, Any]:
        data = self.normalized_data()
        gap = _gap(band_gap_eV)
        levels = self.energy_levels.transition_energies_eV
        tolerance = 16.0 * math.ulp(max(gap, *map(abs, levels), 1.0))
        if any(value < -tolerance or value > gap + tolerance for value in levels):
            raise ValueError("multivalent transition energy lies outside the band gap")
        return {**data, "transition_energies_eV_above_vb": list(levels)}


class MultivalentDefectInput(StructuredInput):
    id: StableId
    name: Nonempty
    total_density_m3: QuantityInput = q("m^-3", positive=True, required=True)
    configuration: MultivalentConfigurationInput

    def resolved_data(self, band_gap_eV: object) -> dict[str, Any]:
        return {**self.normalized_data(), "configuration": self.configuration.resolved_data(band_gap_eV)}


class MetastableConversionInput(StructuredInput):
    transition_energy_eV_above_vb: QuantityInput = q("eV", required=True)
    electron_capture_activation_eV: QuantityInput = q("eV", required=True)
    electron_emission_activation_eV: QuantityInput = q("eV", required=True)
    hole_capture_activation_eV: QuantityInput = q("eV", required=True)
    hole_emission_activation_eV: QuantityInput = q("eV", required=True)
    electron_capture_path: Literal["double_electron_capture", "electron_capture_plus_hole_emission"]
    hole_capture_path: Literal["double_hole_capture", "hole_capture_plus_electron_emission"]
    capture_n_m3_s: QuantityInput = q("m^3/s", positive=True, required=True)
    capture_p_m3_s: QuantityInput = q("m^3/s", positive=True, required=True)
    phonon_frequency_Hz: QuantityInput = q("Hz", positive=True, required=True)

    def validate_gap(self, band_gap_eV: object) -> None:
        data = self.normalized_data()
        gap = _gap(band_gap_eV)
        transition = data["transition_energy_eV_above_vb"]
        if transition > gap:
            raise ValueError("metastable conversion energy lies outside the band gap")
        ec, hc = data["electron_capture_activation_eV"], data["hole_capture_activation_eV"]
        ee = ec + 2 * (gap - transition) if self.electron_capture_path == "double_electron_capture" else ec + gap - 2 * transition
        he = hc + 2 * transition if self.hole_capture_path == "double_hole_capture" else hc + 2 * transition - gap
        actual_ee, actual_he = data["electron_emission_activation_eV"], data["hole_emission_activation_eV"]
        if not all(math.isfinite(value) for value in (ee, he)):
            raise ValueError("nonfinite detailed-balance barrier")
        tolerance = 64 * math.ulp(max(gap, transition, abs(ee), abs(he), actual_ee, actual_he, 1.0))
        if abs(actual_ee - ee) > tolerance or abs(actual_he - he) > tolerance:
            raise ValueError("metastable activation barriers violate detailed balance")


class MetastableDefectInput(StructuredInput):
    id: StableId
    name: Nonempty
    total_density_m3: QuantityInput = q("m^-3", positive=True, required=True)
    donor_configuration: MultivalentConfigurationInput
    acceptor_configuration: MultivalentConfigurationInput
    donor_conversion_state_index: int = Field(ge=0)
    acceptor_conversion_state_index: int = Field(ge=0)
    conversion_kinetics: MetastableConversionInput

    @model_validator(mode="after")
    def _conversion(self) -> MetastableDefectInput:
        donor, acceptor = self.donor_configuration.charge_states_e, self.acceptor_configuration.charge_states_e
        i, j = self.donor_conversion_state_index, self.acceptor_conversion_state_index
        if i >= len(donor) or j >= len(acceptor):
            raise ValueError("metastable conversion state index outside configuration")
        if donor[i] - acceptor[j] != 2:
            raise ValueError("metastable conversion states must differ by exactly two charges")
        return self

    def resolved_data(self, band_gap_eV: object) -> dict[str, Any]:
        self.conversion_kinetics.validate_gap(band_gap_eV)
        return {**self.normalized_data(),
                "donor_configuration": self.donor_configuration.resolved_data(band_gap_eV),
                "acceptor_configuration": self.acceptor_configuration.resolved_data(band_gap_eV)}


class MetastableDocumentInput(StructuredInput):
    schema_version: Literal["solarlab-metastable-bulk-defects-v1"]
    defect_model: Literal["explicit_metastable_frozen"]
    metastable_defects: tuple[MetastableDefectInput, ...]

    @model_validator(mode="after")
    def _inventory(self) -> MetastableDocumentInput:
        ids = [item.id for item in self.metastable_defects]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("metastable inventory requires nonempty unique stable IDs")
        return self

    def resolved_data(self, band_gap_eV: object) -> dict[str, Any]:
        return {**self.normalized_data(), "metastable_defects": [item.resolved_data(band_gap_eV) for item in self.metastable_defects]}


class MetastableNumericsInput(StructuredInput):
    initial_donor_fraction_guess: QuantityInput = q("1", required=True)
    max_iterations: int = Field(gt=0)
    relative_tolerance: QuantityInput = q("1", positive=True, required=True)
    clamping_factor: QuantityInput = q("1", positive=True, required=True)
    final_unclamped_refinement: Literal[True]

    @field_validator("final_unclamped_refinement", mode="before")
    @classmethod
    def _true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("final_unclamped_refinement requires the boolean true")
        return value

    @model_validator(mode="after")
    def _bounds(self) -> MetastableNumericsInput:
        if _number(self.initial_donor_fraction_guess) > 1 or _number(self.clamping_factor) > 1:
            raise ValueError("fraction guess and clamping factor must not exceed 1")
        return self


class MetastablePreparationInput(StructuredInput):
    schema_version: Literal["solarlab-metastable-preparation-v1"]
    preparation_limit: Literal["stationary_infinite_time"]
    preparation_temperature_K: QuantityInput = q("K", positive=True, required=True)
    preparation_voltage_V: QuantityInput = q("V", signed=True, required=True)
    preparation_illumination_suns: QuantityInput = q("1", required=True)
    voltage_continuation_steps: int = Field(ge=0)
    illumination_continuation_steps: int = Field(ge=0)
    measurement_temperature_K: QuantityInput = q("K", positive=True, required=True)
    configuration_freeze_stage: Literal["after_stationary_preparation_before_measurement"]
    freeze_configuration_during_measurement: Literal[True]
    measurement_protocol_sha256: SHA256
    numerics: MetastableNumericsInput

    @field_validator("freeze_configuration_during_measurement", mode="before")
    @classmethod
    def _freeze(cls, value: object) -> object:
        if value is not True:
            raise ValueError("metastable v1 requires a frozen measurement")
        return value

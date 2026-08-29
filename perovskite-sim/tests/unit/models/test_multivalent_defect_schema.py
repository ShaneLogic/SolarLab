"""D7-P0 canonical multivalent and metastable input contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from perovskite_sim.models.defects import (
    EXPLICIT_QUASI_STEADY,
    BulkDefectDocument,
    BulkDefectKinetics,
    ExplicitDefectSchemaError,
)
from perovskite_sim.models.multivalent_defects import (
    AMPHOTERIC,
    CUSTOM_MULTILEVEL,
    DOUBLE_ACCEPTOR,
    DOUBLE_DONOR,
    DOUBLE_ELECTRON_CAPTURE,
    DOUBLE_HOLE_CAPTURE,
    ELECTRON_CAPTURE_HOLE_EMISSION,
    EXPLICIT,
    EXPLICIT_METASTABLE_FROZEN,
    FROZEN_BEFORE_MEASUREMENT,
    HOLE_CAPTURE_ELECTRON_EMISSION,
    METASTABLE_DEFECT_SCHEMA_VERSION,
    METASTABLE_PREPARATION_SCHEMA_VERSION,
    MULTIVALENT_DEFECT_SCHEMA_VERSION,
    SCAPS_BINOMIAL,
    SINGLE_ACCEPTOR,
    SINGLE_DONOR,
    STATIONARY_INFINITE_TIME,
    UNITY,
    MetastableConversionKinetics,
    MetastableDefectDefinition,
    MetastableDefectDocument,
    MetastablePreparationNumerics,
    MetastablePreparationProtocol,
    MultivalentBulkDefectDocument,
    MultivalentBulkDefectSpecies,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)


def _kinetics(
    *,
    sigma_n_m2: float = 2.0e-19,
    sigma_p_m2: float = 7.0e-20,
) -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=sigma_n_m2,
        sigma_p_m2=sigma_p_m2,
        thermal_velocity_n_m_s=1.3e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def _configuration(
    family: str = AMPHOTERIC,
    *,
    charges: tuple[int, ...] = (1, 0, -1),
    degeneracy_convention: str = SCAPS_BINOMIAL,
    degeneracies: tuple[float, ...] = (1.0, 2.0, 1.0),
    first_transition_eV: float = 0.45,
    correlations_eV: tuple[float, ...] = (0.20,),
) -> MultivalentDefectConfiguration:
    return MultivalentDefectConfiguration(
        family=family,
        charge_states_e=charges,
        degeneracy_convention=degeneracy_convention,
        state_degeneracies=degeneracies,
        energy_levels=MultivalentEnergyLevels(
            first_transition_eV_above_vb=first_transition_eV,
            correlation_energies_eV=correlations_eV,
        ),
        transition_kinetics=tuple(_kinetics() for _ in range(len(charges) - 1)),
    )


def _species() -> MultivalentBulkDefectSpecies:
    return MultivalentBulkDefectSpecies(
        name="absorber_amphoteric",
        total_density_m3=1.0e21,
        configuration=_configuration(),
    )


def _conversion() -> MetastableConversionKinetics:
    # VSe-VCu CuInSe2 values from Decock et al., JAP 111, 043703 (2012).
    return MetastableConversionKinetics(
        transition_energy_eV_above_vb=0.19,
        electron_capture_activation_eV=0.10,
        electron_emission_activation_eV=0.76,
        hole_capture_activation_eV=0.35,
        hole_emission_activation_eV=0.73,
        electron_capture_path=ELECTRON_CAPTURE_HOLE_EMISSION,
        hole_capture_path=DOUBLE_HOLE_CAPTURE,
        capture_n_m3_s=2.0e-14,
        capture_p_m3_s=3.0e-14,
        phonon_frequency_Hz=1.0e13,
    )


def _metastable_definition() -> MetastableDefectDefinition:
    donor = _configuration(
        SINGLE_DONOR,
        charges=(1, 0),
        degeneracy_convention=UNITY,
        degeneracies=(1.0, 1.0),
        first_transition_eV=1.00,
        correlations_eV=(),
    )
    acceptor = _configuration(
        DOUBLE_ACCEPTOR,
        charges=(0, -1, -2),
        first_transition_eV=0.06,
        correlations_eV=(0.79,),
    )
    return MetastableDefectDefinition(
        name="vse-vcu",
        total_density_m3=3.0e21,
        donor_configuration=donor,
        acceptor_configuration=acceptor,
        donor_conversion_state_index=0,
        acceptor_conversion_state_index=1,
        conversion_kinetics=_conversion(),
    )


def _preparation() -> MetastablePreparationProtocol:
    return MetastablePreparationProtocol(
        schema_version=METASTABLE_PREPARATION_SCHEMA_VERSION,
        preparation_limit=STATIONARY_INFINITE_TIME,
        preparation_temperature_K=330.0,
        preparation_voltage_V=-2.0,
        preparation_illumination_suns=0.0,
        voltage_continuation_steps=20,
        illumination_continuation_steps=0,
        measurement_temperature_K=200.0,
        configuration_freeze_stage=FROZEN_BEFORE_MEASUREMENT,
        freeze_configuration_during_measurement=True,
        measurement_protocol_sha256="a" * 64,
        numerics=MetastablePreparationNumerics(
            initial_donor_fraction_guess=0.5,
            max_iterations=250,
            relative_tolerance=1.0e-6,
            clamping_factor=0.05,
        ),
    )


@pytest.mark.parametrize(
    ("family", "charges"),
    (
        (SINGLE_DONOR, (1, 0)),
        (SINGLE_ACCEPTOR, (0, -1)),
        (DOUBLE_DONOR, (2, 1, 0)),
        (DOUBLE_ACCEPTOR, (0, -1, -2)),
        (AMPHOTERIC, (1, 0, -1)),
    ),
)
def test_scaps_families_freeze_charge_order_and_binomial_degeneracy(
    family,
    charges,
):
    state_count = len(charges)
    degeneracies = (1.0, 1.0) if state_count == 2 else (1.0, 2.0, 1.0)
    configuration = _configuration(
        family,
        charges=charges,
        degeneracies=degeneracies,
        correlations_eV=(() if state_count == 2 else (0.20,)),
    )

    assert configuration.charge_states_e == charges
    assert configuration.state_degeneracies == degeneracies
    assert len(configuration.transition_kinetics) == state_count - 1


def test_custom_multilevel_supports_five_shared_states_but_not_charge_gaps():
    configuration = _configuration(
        CUSTOM_MULTILEVEL,
        charges=(2, 1, 0, -1, -2),
        degeneracy_convention=EXPLICIT,
        degeneracies=(1.0, 3.0, 4.0, 3.0, 1.0),
        correlations_eV=(0.10, -0.05, 0.20),
    )

    assert configuration.energy_levels.transition_energies_eV_above_vb == (
        0.45,
        0.55,
        0.5,
        0.7,
    )
    with pytest.raises(ExplicitDefectSchemaError, match="descend by one"):
        replace(configuration, charge_states_e=(2, 0, -1, -2, -3))


def test_family_charge_degeneracy_and_transition_count_mismatches_fail_closed():
    with pytest.raises(ExplicitDefectSchemaError, match="requires charge_states"):
        _configuration(DOUBLE_DONOR, charges=(1, 0, -1))
    with pytest.raises(ExplicitDefectSchemaError, match="binomial"):
        _configuration(degeneracies=(1.0, 1.0, 1.0))
    with pytest.raises(ExplicitDefectSchemaError, match="one energy per"):
        _configuration(correlations_eV=())
    with pytest.raises(ExplicitDefectSchemaError, match="capture leg"):
        replace(
            _configuration(),
            transition_kinetics=(
                _kinetics(sigma_n_m2=0.0, sigma_p_m2=0.0),
                _kinetics(),
            ),
        )


def test_signed_correlation_energy_is_explicit_and_all_levels_stay_in_gap():
    configuration = _configuration(
        CUSTOM_MULTILEVEL,
        charges=(1, 0, -1, -2),
        degeneracy_convention=UNITY,
        degeneracies=(1.0, 1.0, 1.0, 1.0),
        first_transition_eV=0.8,
        correlations_eV=(-0.2, 0.3),
    )
    configuration.validate_band_gap(1.2)
    with pytest.raises(ExplicitDefectSchemaError, match="inside the band gap"):
        configuration.validate_band_gap(0.85)


def test_v4_document_roundtrip_hash_and_shared_density_are_stable():
    document = MultivalentBulkDefectDocument(
        schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_species(),),
    )
    restored = MultivalentBulkDefectDocument.from_dict(document.to_dict())

    assert restored == document
    assert restored.canonical_json() == document.canonical_json()
    assert restored.sha256 == document.sha256
    assert document.sha256 == (
        "3f29bf380c710b03834b9ef50ad4b95d403352e743345046f55922e6283f7e81"
    )
    assert (
        document.sha256
        != replace(
            document,
            bulk_defects=(replace(_species(), total_density_m3=2.0e21),),
        ).sha256
    )
    assert document.to_dict()["bulk_defects"][0]["total_density_m3"] == 1.0e21


def test_v4_unknown_keys_and_legacy_parser_fail_closed():
    document = MultivalentBulkDefectDocument(
        schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_species(),),
    )
    payload = document.to_dict()
    payload["claim"] = "production_ready"
    with pytest.raises(ExplicitDefectSchemaError, match="unknown=.*claim"):
        MultivalentBulkDefectDocument.from_dict(payload)
    with pytest.raises(
        ExplicitDefectSchemaError,
        match="schema mismatch|unsupported",
    ):
        BulkDefectDocument.from_dict(document.to_dict())


def test_metastable_definition_roundtrips_and_obeys_primary_source_balance():
    definition = _metastable_definition()
    definition.validate_band_gap(1.04)
    document = MetastableDefectDocument(
        schema_version=METASTABLE_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_METASTABLE_FROZEN,
        metastable_defects=(definition,),
    )
    document.validate_band_gap(1.04)

    assert MetastableDefectDocument.from_dict(document.to_dict()) == document
    assert document.sha256 == (
        "101ab83c111fbeb5c1ac89e4479360721255fd8d0aa4694ba3e8269d5c43900b"
    )
    donor_charge = definition.donor_configuration.charge_states_e[
        definition.donor_conversion_state_index
    ]
    acceptor_charge = definition.acceptor_configuration.charge_states_e[
        definition.acceptor_conversion_state_index
    ]
    assert donor_charge - acceptor_charge == 2


def test_metastable_barrier_or_conversion_charge_mismatch_fails_closed():
    definition = _metastable_definition()
    with pytest.raises(ExplicitDefectSchemaError, match="detailed balance"):
        replace(
            definition,
            conversion_kinetics=replace(
                definition.conversion_kinetics,
                electron_emission_activation_eV=0.75,
            ),
        ).validate_band_gap(1.04)
    with pytest.raises(ExplicitDefectSchemaError, match="exactly two"):
        replace(definition, acceptor_conversion_state_index=0)


def test_alternate_double_electron_and_mixed_hole_paths_obey_balance():
    kinetics = MetastableConversionKinetics(
        transition_energy_eV_above_vb=0.80,
        electron_capture_activation_eV=0.10,
        electron_emission_activation_eV=1.50,
        hole_capture_activation_eV=0.20,
        hole_emission_activation_eV=0.30,
        electron_capture_path=DOUBLE_ELECTRON_CAPTURE,
        hole_capture_path=HOLE_CAPTURE_ELECTRON_EMISSION,
        capture_n_m3_s=1.0e-14,
        capture_p_m3_s=2.0e-14,
        phonon_frequency_Hz=1.0e13,
    )

    kinetics.validate_detailed_balance(1.50)
    with pytest.raises(ExplicitDefectSchemaError, match="detailed balance"):
        replace(kinetics, hole_emission_activation_eV=0.31).validate_detailed_balance(
            1.50
        )


def test_preparation_protocol_roundtrip_hashes_every_execution_field():
    protocol = _preparation()
    restored = MetastablePreparationProtocol.from_dict(protocol.to_dict())
    assert restored == protocol
    assert restored.canonical_json() == protocol.canonical_json()
    assert restored.sha256 == protocol.sha256
    assert protocol.sha256 == (
        "7bba03a617e8420bdfd974b004e0422ab36debad66a0e6ff8085f7274de14ea6"
    )

    variants = (
        replace(protocol, preparation_temperature_K=331.0),
        replace(protocol, preparation_voltage_V=-1.9),
        replace(protocol, preparation_illumination_suns=0.1),
        replace(protocol, voltage_continuation_steps=21),
        replace(protocol, illumination_continuation_steps=1),
        replace(protocol, measurement_temperature_K=201.0),
        replace(protocol, measurement_protocol_sha256="b" * 64),
        replace(
            protocol,
            numerics=replace(
                protocol.numerics,
                initial_donor_fraction_guess=0.4,
            ),
        ),
        replace(
            protocol,
            numerics=replace(protocol.numerics, clamping_factor=0.04),
        ),
    )
    assert all(value.sha256 != protocol.sha256 for value in variants)


def test_preparation_v1_cannot_silently_update_during_measurement():
    protocol = _preparation()
    with pytest.raises(ExplicitDefectSchemaError, match="frozen measurement"):
        replace(protocol, freeze_configuration_during_measurement=False)
    with pytest.raises(ExplicitDefectSchemaError, match="stationary"):
        replace(protocol, preparation_limit="finite_time_dynamic")
    with pytest.raises(ExplicitDefectSchemaError, match="final unclamped"):
        replace(
            protocol.numerics,
            final_unclamped_refinement=False,
        )


def test_preparation_unknown_fields_and_invalid_digest_fail_closed():
    payload = _preparation().to_dict()
    payload["cooling_rate_K_s"] = 1.0
    with pytest.raises(ExplicitDefectSchemaError, match="unknown"):
        MetastablePreparationProtocol.from_dict(payload)
    with pytest.raises(ExplicitDefectSchemaError, match="SHA-256"):
        replace(_preparation(), measurement_protocol_sha256="unbound")

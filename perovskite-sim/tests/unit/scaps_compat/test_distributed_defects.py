"""D3-E2 strict SCAPS distributed-defect conversion fixtures."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    NEUTRAL_WHEN_EMPTY,
    UNIFORM,
    VALENCE_BAND_TAIL,
    BulkDefectDocument,
    ExplicitDefectCapabilityError,
    ExplicitDefectSchemaError,
)
from perovskite_sim.scaps_compat.distributed_defects import (
    SCAPS_DENSITY_RELATIVE_TOLERANCE,
    SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION,
    SCAPS_ENERGY_ABOVE_INTRINSIC_LEVEL,
    SCAPS_ENERGY_ABOVE_VALENCE_BAND,
    SCAPS_ENERGY_BELOW_CONDUCTION_BAND,
    convert_scaps_distributed_bulk_defect,
)


GAP_EV = 1.5
TEMPERATURE_K = 300.0
NC_CM3 = 2.0e19
NV_CM3 = 2.0e19
VELOCITY_CM_S = 1.0e7


def _mapping(kind: str = GAUSSIAN, **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "name": "acceptor_distribution",
        "distribution": kind,
        "sigma_n_cm2": 1.0e-15,
        "sigma_p_cm2": 2.0e-15,
        "charge_transition": ACCEPTOR,
        "neutral_reference": NEUTRAL_WHEN_EMPTY,
        "E_t_eV_above_vb": 0.75,
        "E_char_eV": 0.08,
        "N_total_cm3": 1.0e16,
    }
    if kind != UNIFORM:
        value["support_width_multiplier"] = 6.0
    if kind == UNIFORM:
        value["E_char_eV"] = 0.4
    elif kind == CONDUCTION_BAND_TAIL:
        value["E_t_eV_above_vb"] = 1.4
        value["E_char_eV"] = 0.1
        value["support_width_multiplier"] = 7.0
    elif kind == VALENCE_BAND_TAIL:
        value["E_t_eV_above_vb"] = 0.1
        value["E_char_eV"] = 0.1
        value["support_width_multiplier"] = 7.0
    value.update(updates)
    return value


def _convert(value: dict[str, object]):
    return convert_scaps_distributed_bulk_defect(
        value,
        band_gap_eV=GAP_EV,
        temperature_K=TEMPERATURE_K,
        effective_conduction_dos_cm3=NC_CM3,
        effective_valence_dos_cm3=NV_CM3,
        layer_thermal_velocity_cm_s=VELOCITY_CM_S,
        where="fixture",
    )


def _document_sha(conversion) -> str:
    return BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(conversion.species,),
    ).sha256


@pytest.mark.parametrize(
    (
        "kind",
        "expected_integral_eV",
        "expected_peak_m3_eV",
        "expected_conversion_sha256",
        "expected_document_sha256",
    ),
    (
        (
            GAUSSIAN,
            0.14179317572152342,
            7.05252558814229e22,
            "ebfd73e6785e77d4caa24f66e6f85ba6056038077bc522a1baf780c4ee0e31c2",
            "ca2724f5701bada98e0f0307f43128f2fce55c8ada416c3fd3f726e90a117da6",
        ),
        (
            UNIFORM,
            0.4,
            2.5e22,
            "2be71f3a68525c93fb618866edecd6d62ff5e38f1663818038d7263055133ae3",
            "80b880ee09033d012e0dc6f70efc6dc90dcf4a53c8fec81fcc8e42405d96cc74",
        ),
        (
            CONDUCTION_BAND_TAIL,
            0.09990881180344456,
            1.0009127142532215e23,
            "76c035c065e951aabe86007d81a8797830ba0d91257b3fd67957638f36e34d87",
            "c93b2931f5469fc5233a5000e34a68462634148b78a2028513b1043a054cd8c2",
        ),
        (
            VALENCE_BAND_TAIL,
            0.09990881180344456,
            1.0009127142532215e23,
            "1aa4ca63a14d5ae01e4da1051ed71118dd083176c4f98689eb07d4fca78c571a",
            "f1ed3a6bdd45a0af807f687e2e7d6de2858e650410942299269e82c9ef2501e9",
        ),
    ),
)
def test_literal_conversion_fixtures_freeze_values_and_hashes(
    kind,
    expected_integral_eV,
    expected_peak_m3_eV,
    expected_conversion_sha256,
    expected_document_sha256,
):
    conversion = _convert(_mapping(kind))

    assert conversion.shape_integral_eV == expected_integral_eV
    assert conversion.resolved_peak_density_m3_eV == expected_peak_m3_eV
    assert conversion.conversion_identity_sha256 == expected_conversion_sha256
    assert _document_sha(conversion) == expected_document_sha256


@pytest.mark.parametrize(
    ("kind", "expected_shape_integral_eV"),
    (
        (
            GAUSSIAN,
            0.08 * math.sqrt(math.pi) * math.erf(3.0),
        ),
        (UNIFORM, 0.4),
        (
            CONDUCTION_BAND_TAIL,
            0.1 * -math.expm1(-7.0),
        ),
        (
            VALENCE_BAND_TAIL,
            0.1 * -math.expm1(-7.0),
        ),
    ),
)
def test_integrated_total_fixtures_freeze_shape_units_and_kinetics(
    kind,
    expected_shape_integral_eV,
):
    conversion = _convert(_mapping(kind))
    species = conversion.species

    assert conversion.shape_integral_eV == pytest.approx(
        expected_shape_integral_eV,
        rel=2.0e-15,
    )
    assert conversion.source_density_mode == "integrated_total"
    assert conversion.resolved_total_density_m3 == 1.0e22
    assert conversion.resolved_peak_density_m3_eV == pytest.approx(
        1.0e22 / expected_shape_integral_eV,
        rel=2.0e-15,
    )
    assert species.kinetics.sigma_n_m2 == 1.0e-19
    assert species.kinetics.sigma_p_m2 == 2.0e-19
    assert species.kinetics.thermal_velocity_n_m_s == 1.0e5
    assert species.kinetics.thermal_velocity_p_m_s == 1.0e5
    assert len(conversion.conversion_identity_sha256) == 64
    assert len(_document_sha(conversion)) == 64


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_peak_only_and_both_density_forms_recover_the_total_fixture(kind):
    total = _convert(_mapping(kind))
    peak_cm3_eV = (
        total.resolved_peak_density_m3_eV / 1.0e6
    )
    peak_only_mapping = _mapping(kind)
    peak_only_mapping.pop("N_total_cm3")
    peak_only_mapping["N_peak_cm3_eV"] = peak_cm3_eV
    peak_only = _convert(peak_only_mapping)
    both = _convert(
        _mapping(kind, N_peak_cm3_eV=peak_cm3_eV)
    )

    assert peak_only.source_density_mode == "peak_density"
    assert both.source_density_mode == "integrated_total_and_peak_density"
    assert peak_only.species == total.species == both.species
    assert _document_sha(peak_only) == _document_sha(total) == _document_sha(both)
    assert both.density_relative_mismatch is not None
    assert both.density_relative_mismatch <= SCAPS_DENSITY_RELATIVE_TOLERANCE
    assert len(
        {
            total.conversion_identity_sha256,
            peak_only.conversion_identity_sha256,
            both.conversion_identity_sha256,
        }
    ) == 3


def test_three_energy_references_produce_the_same_canonical_species():
    above_vb = _convert(_mapping())
    below_cb_mapping = _mapping()
    below_cb_mapping.pop("E_t_eV_above_vb")
    below_cb_mapping["E_t_eV_below_cb"] = 0.75
    below_cb = _convert(below_cb_mapping)
    above_intrinsic_mapping = _mapping()
    above_intrinsic_mapping.pop("E_t_eV_above_vb")
    above_intrinsic_mapping["E_t_eV_above_intrinsic"] = 0.0
    above_intrinsic = _convert(above_intrinsic_mapping)

    assert above_vb.source_energy_reference == SCAPS_ENERGY_ABOVE_VALENCE_BAND
    assert below_cb.source_energy_reference == SCAPS_ENERGY_BELOW_CONDUCTION_BAND
    assert (
        above_intrinsic.source_energy_reference
        == SCAPS_ENERGY_ABOVE_INTRINSIC_LEVEL
    )
    assert above_vb.intrinsic_level_eV_above_vb == 0.75
    assert above_vb.species == below_cb.species == above_intrinsic.species
    assert _document_sha(above_vb) == _document_sha(below_cb)
    assert _document_sha(above_vb) == _document_sha(above_intrinsic)
    assert len(
        {
            above_vb.conversion_identity_sha256,
            below_cb.conversion_identity_sha256,
            above_intrinsic.conversion_identity_sha256,
        }
    ) == 3


def test_intrinsic_reference_uses_dos_asymmetry_and_temperature():
    value = _mapping()
    value.pop("E_t_eV_above_vb")
    value["E_t_eV_above_intrinsic"] = 0.04
    conversion = convert_scaps_distributed_bulk_defect(
        value,
        band_gap_eV=GAP_EV,
        temperature_K=320.0,
        effective_conduction_dos_cm3=4.0e19,
        effective_valence_dos_cm3=1.0e19,
        layer_thermal_velocity_cm_s=VELOCITY_CM_S,
        where="asymmetric fixture",
    )
    expected_intrinsic = (
        0.5 * GAP_EV
        + 0.5
        * 8.617333262145e-5
        * 320.0
        * math.log(1.0e19 / 4.0e19)
    )

    assert conversion.intrinsic_level_eV_above_vb == pytest.approx(
        expected_intrinsic,
        rel=2.0e-15,
    )
    assert conversion.species.distribution.center_eV_above_vb == pytest.approx(
        expected_intrinsic + 0.04,
        rel=2.0e-15,
    )


def test_inconsistent_total_and_peak_density_fail_closed():
    with pytest.raises(ExplicitDefectSchemaError, match="inconsistent"):
        _convert(_mapping(N_peak_cm3_eV=1.0e12))


@pytest.mark.parametrize(
    "mutation",
    (
        {"N_total_cm3": None},
        {"E_t_eV_below_cb": 0.2},
        {"N_t_cm3": 1.0e16},
        {"N_peak_cm3": 1.0e16},
        {"invented_normalization": "peak"},
    ),
)
def test_missing_ambiguous_and_unknown_fields_fail_closed(mutation):
    value = _mapping()
    if mutation == {"N_total_cm3": None}:
        value.pop("N_total_cm3")
    else:
        value.update(mutation)
    with pytest.raises(ExplicitDefectSchemaError):
        _convert(value)


def test_support_contract_is_distribution_specific_and_inside_gap():
    missing_support = _mapping(GAUSSIAN)
    missing_support.pop("support_width_multiplier")
    with pytest.raises(ExplicitDefectSchemaError, match="requires"):
        _convert(missing_support)

    with pytest.raises(ExplicitDefectSchemaError, match="forbids"):
        _convert(_mapping(UNIFORM, support_width_multiplier=2.0))

    with pytest.raises(ExplicitDefectSchemaError, match="support"):
        _convert(
            _mapping(
                CONDUCTION_BAND_TAIL,
                E_t_eV_above_vb=0.5,
            )
        )


def test_nonunit_degeneracy_remains_an_explicit_capability_error():
    with pytest.raises(ExplicitDefectCapabilityError, match="degeneracy=1.0"):
        _convert(_mapping(degeneracy=2.0))


def test_two_zero_capture_legs_fail_before_occupancy_is_constructed():
    with pytest.raises(
        ExplicitDefectCapabilityError,
        match="capture cross section",
    ):
        _convert(_mapping(sigma_n_cm2=0.0, sigma_p_cm2=0.0))


def test_conversion_payload_and_identity_are_fail_closed():
    conversion = _convert(_mapping())
    payload = conversion.to_dict()

    assert payload["adapter"] == SCAPS_DISTRIBUTED_DEFECT_ADAPTER_VERSION
    assert payload["conversion_identity_sha256"] == (
        conversion.conversion_identity_sha256
    )
    with pytest.raises(ValueError, match="canonical|identity"):
        replace(
            conversion,
            resolved_total_density_m3=(
                conversion.resolved_total_density_m3 * 2.0
            ),
        )
    with pytest.raises(ValueError, match="identity"):
        replace(conversion, conversion_identity_sha256="0" * 64)

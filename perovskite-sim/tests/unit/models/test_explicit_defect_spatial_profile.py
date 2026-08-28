"""D3-E4a canonical spatial-profile contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import (
    ACCEPTOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    LAYER_AVERAGE_UNITY,
    NEUTRAL_WHEN_EMPTY,
    NORMALIZED_LAYER_COORDINATE,
    PIECEWISE_LINEAR,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpatialKnot,
    BulkDefectSpatialProfile,
    BulkDefectSpecies,
    ExplicitDefectSchemaError,
    bulk_defect_document_from_layer_mapping,
    bulk_defect_species_at_layer_position,
)
from perovskite_sim.reproducibility import semantic_sha256


ROOT = Path(__file__).resolve().parents[3]


def _profile(
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> BulkDefectSpatialProfile:
    positions = (0.0, 0.5, 1.0)
    return BulkDefectSpatialProfile(
        coordinate=NORMALIZED_LAYER_COORDINATE,
        interpolation=PIECEWISE_LINEAR,
        density_normalization=LAYER_AVERAGE_UNITY,
        knots=tuple(
            BulkDefectSpatialKnot(position, multiplier)
            for position, multiplier in zip(positions, multipliers, strict=True)
        ),
    )


def _species(
    profile: BulkDefectSpatialProfile | None = None,
) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name="graded_acceptor",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=2.0e22,
            center_eV_above_vb=0.55,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
        charge_transition=ACCEPTOR,
        neutral_reference=NEUTRAL_WHEN_EMPTY,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=1.0e-19,
            sigma_p_m2=2.0e-19,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=1.2e5,
        ),
        degeneracy=1.0,
        spatial_profile=profile,
    )


def _document() -> BulkDefectDocument:
    return BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(_species(_profile()),),
    )


def test_v3_spatial_document_roundtrip_hash_and_flat_layer_parser():
    document = _document()
    payload = document.to_dict()

    restored = BulkDefectDocument.from_dict(payload)
    layer = {
        "defect_schema_version": payload["schema_version"],
        "defect_model": payload["defect_model"],
        "bulk_defects": payload["bulk_defects"],
    }

    assert restored == document
    assert restored.canonical_json() == document.canonical_json()
    assert restored.sha256 == document.sha256
    assert bulk_defect_document_from_layer_mapping(layer) == document
    assert document.bulk_defects[0].spatial_profile.sha256 == (
        "6260828621b21d54d434445db4225a746af80af556dfae75e3feecb38f77d59c"
    )


def test_piecewise_linear_profile_is_explicit_and_conservative():
    profile = _profile()

    assert profile.layer_average_multiplier == 1.0
    assert profile.density_multiplier_at(0.0) == 0.5
    assert profile.density_multiplier_at(0.25) == 0.75
    assert profile.density_multiplier_at(0.5) == 1.0
    assert profile.density_multiplier_at(0.75) == 1.25
    assert profile.density_multiplier_at(1.0) == 1.5
    assert not profile.is_uniform


def test_localization_preserves_energy_and_resolves_only_density():
    source = _species(_profile())

    front, middle, back = (
        bulk_defect_species_at_layer_position((source,), position)[0]
        for position in (0.0, 0.5, 1.0)
    )

    assert front.distribution.total_density_m3 == 1.0e22
    assert middle.distribution.total_density_m3 == 2.0e22
    assert back.distribution.total_density_m3 == 3.0e22
    assert all(
        item.distribution.center_eV_above_vb == 0.55
        for item in (front, middle, back)
    )
    assert all(item.spatial_profile is None for item in (front, middle, back))
    assert source.spatial_profile is not None


def test_unprofiled_species_retains_object_identity_when_localized():
    source = _species()

    assert bulk_defect_species_at_layer_position((source,), 0.37)[0] is source


def test_v1_v2_documents_cannot_acquire_v3_profile_meaning():
    source = _species(_profile())

    with pytest.raises(ExplicitDefectSchemaError, match="schema v2"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(source,),
        )


def test_v3_requires_at_least_one_profile():
    with pytest.raises(ExplicitDefectSchemaError, match="at least one"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(_species(),),
        )


def test_empty_v3_document_cannot_reserve_spatial_semantics():
    with pytest.raises(ExplicitDefectSchemaError, match="at least one"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
            defect_model="effective_lifetime",
            bulk_defects=(),
        )


def test_v3_roundtrips_through_standard_backend_device_payload():
    baseline = load_device_from_yaml(
        str(ROOT / "configs" / "distributed_defect_qf_dc_pn.yaml")
    )
    document = _document()
    first = baseline.layers[0]
    explicit = replace(
        baseline,
        layers=(
            replace(
                first,
                params=replace(
                    first.params,
                    defect_schema_version=document.schema_version,
                    defect_model=document.defect_model,
                    bulk_defects=document.bulk_defects,
                ),
            ),
            *baseline.layers[1:],
        ),
    )

    rebuilt = stack_from_dict(_stack_to_config_dict(explicit))

    assert rebuilt.layers[0].params.defect_document == document
    assert semantic_sha256(rebuilt) == semantic_sha256(explicit)
    assert semantic_sha256(rebuilt) != semantic_sha256(baseline)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"coordinate": "device_coordinate_m"}, "coordinate"),
        ({"interpolation": "cubic"}, "interpolation"),
        ({"density_normalization": "reference_value"}, "density_normalization"),
    ),
)
def test_profile_enum_fields_fail_closed(updates, message):
    values = {
        "coordinate": NORMALIZED_LAYER_COORDINATE,
        "interpolation": PIECEWISE_LINEAR,
        "density_normalization": LAYER_AVERAGE_UNITY,
        "knots": _profile().knots,
    }
    values.update(updates)

    with pytest.raises(ExplicitDefectSchemaError, match=message):
        BulkDefectSpatialProfile(**values)


@pytest.mark.parametrize(
    ("knots", "message"),
    (
        (((0.1, 0.5), (1.0, 1.5)), "exact endpoints"),
        (((0.0, 0.5), (0.7, 1.0), (0.7, 1.0), (1.0, 1.5)), "strictly"),
        (((0.0, 1.0), (1.0, 2.0)), "layer-average unity"),
        (((0.0, 1.0), (1.0, 0.0)), "positive"),
    ),
)
def test_profile_knots_and_normalization_fail_closed(knots, message):
    with pytest.raises(ExplicitDefectSchemaError, match=message):
        BulkDefectSpatialProfile(
            coordinate=NORMALIZED_LAYER_COORDINATE,
            interpolation=PIECEWISE_LINEAR,
            density_normalization=LAYER_AVERAGE_UNITY,
            knots=tuple(BulkDefectSpatialKnot(*item) for item in knots),
        )


def test_profile_mapping_is_strict_and_does_not_renormalize():
    payload = _profile().to_dict()
    payload["implicit_normalization"] = True

    with pytest.raises(ExplicitDefectSchemaError, match="unknown"):
        BulkDefectSpatialProfile.from_dict(payload)


def test_position_queries_fail_closed_outside_layer():
    profile = _profile()

    with pytest.raises(ExplicitDefectSchemaError, match=r"\[0, 1\]"):
        profile.density_multiplier_at(1.01)
    with pytest.raises(ExplicitDefectSchemaError, match="non-negative"):
        profile.density_multiplier_at(-0.01)


def test_spatial_profile_changes_identity_without_changing_energy_contract():
    document = _document()
    mirrored = replace(
        document,
        bulk_defects=(
            replace(
                document.bulk_defects[0],
                spatial_profile=_profile((1.5, 1.0, 0.5)),
            ),
        ),
    )

    assert mirrored.sha256 != document.sha256
    assert (
        mirrored.bulk_defects[0].distribution
        == document.bulk_defects[0].distribution
    )

"""DEF-0 canonical explicit-defect input and compatibility contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import (
    load_device_from_yaml,
    material_params_from_dict,
)
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    EFFECTIVE_LIFETIME,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_DYNAMIC,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    UNRESOLVED,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpecies,
    ExplicitDefectCapabilityError,
    ExplicitDefectSchemaError,
    bulk_defect_document_from_layer_mapping,
    bulk_defect_species_from_scaps_mapping,
)
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.bulk_traps import BulkTrapDistribution
from perovskite_sim.reproducibility import semantic_sha256
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import (
    StateVec,
    assemble_rhs,
    build_material_arrays,
)


ROOT = Path(__file__).resolve().parents[3]


def _species(
    *,
    name: str | None = "absorber_acceptor_1",
    transition: str = ACCEPTOR,
    neutral_reference: str = NEUTRAL_WHEN_EMPTY,
) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name=name,
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.60,
        ),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=1.0e-19,
            sigma_p_m2=2.0e-19,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=2.0e5,
        ),
        degeneracy=2.0,
    )


def _document(
    *,
    model: str = EXPLICIT_QUASI_STEADY,
    species: tuple[BulkDefectSpecies, ...] | None = None,
) -> BulkDefectDocument:
    return BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=model,
        bulk_defects=species if species is not None else (_species(),),
    )


def test_canonical_document_roundtrip_and_hash_are_stable():
    document = _document()

    restored = BulkDefectDocument.from_dict(document.to_dict())

    assert restored == document
    assert restored.canonical_json() == document.canonical_json()
    assert restored.sha256 == document.sha256
    assert len(document.sha256) == 64
    assert document.sha256 != replace(
        document,
        bulk_defects=(
            replace(
                _species(),
                distribution=replace(
                    _species().distribution,
                    total_density_m3=2.0e22,
                ),
            ),
        ),
    ).sha256


def test_v1_canonical_json_and_hash_remain_frozen_after_v2_is_added():
    document = _document()

    assert document.sha256 == (
        "22a204b0ecf9a1c456aa3de36bfe2d75915b30389af6990038374fc09e163617"
    )
    assert "energy_reference" not in document.canonical_json()
    assert "support_width_multiplier" not in document.canonical_json()


@pytest.mark.parametrize(
    ("config_name", "expected_semantic_sha256"),
    (
        (
            "scaps_defect_s0_neutral.yaml",
            "a44db13adb9d77e705dec93753a72162569860590f8df4f371caf1ad493e473e",
        ),
        (
            "scaps_defect_s1_acceptor_n.yaml",
            "513d87b2828a0b372477657b2744a88b434f6501c88566a6fd9b6933f515c3b6",
        ),
        (
            "scaps_defect_s2_donor_p.yaml",
            "9c5a68947ed42c9c13ab74403b5599785ff1708d7ad7413d4215ef960c90607e",
        ),
    ),
)
def test_v1_shipped_device_semantic_hashes_remain_frozen(
    config_name,
    expected_semantic_sha256,
):
    stack = load_device_from_yaml(str(ROOT / "configs" / config_name))

    assert semantic_sha256(stack) == expected_semantic_sha256


@pytest.mark.parametrize(
    "distribution",
    (
        BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.6,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
        BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.6,
            width_eV=0.08,
            width_convention=WIDTH_SCAPS_CHARACTERISTIC,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            support_width_multiplier=6.0,
        ),
        BulkDefectDistribution(
            kind=UNIFORM,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.6,
            width_eV=0.2,
            width_convention=WIDTH_UNIFORM_FULL,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
        BulkDefectDistribution(
            kind=CONDUCTION_BAND_TAIL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=1.4,
            width_eV=0.1,
            width_convention=WIDTH_SCAPS_CHARACTERISTIC,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            support_width_multiplier=7.0,
        ),
        BulkDefectDistribution(
            kind=VALENCE_BAND_TAIL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.1,
            width_eV=0.1,
            width_convention=WIDTH_SCAPS_CHARACTERISTIC,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            support_width_multiplier=7.0,
        ),
    ),
)
def test_v2_roundtrip_supports_all_scaps_distribution_families(distribution):
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(replace(_species(), distribution=distribution),),
    )

    document.validate_band_gap(1.5)
    assert BulkDefectDocument.from_dict(document.to_dict()) == document
    assert document.sha256 == BulkDefectDocument.from_dict(
        document.to_dict()
    ).sha256


def test_schema_versions_cannot_silently_exchange_distribution_semantics():
    explicit_single = replace(
        _species(),
        distribution=replace(
            _species().distribution,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
    )
    with pytest.raises(ExplicitDefectSchemaError, match="v1 forbids"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(explicit_single,),
        )

    legacy_single = _species()
    with pytest.raises(ExplicitDefectSchemaError, match="v2 requires"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(legacy_single,),
        )

    incomplete_gaussian = replace(
        explicit_single,
        distribution=BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.6,
            width_eV=0.08,
            width_convention=WIDTH_SCAPS_CHARACTERISTIC,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
    )
    with pytest.raises(ExplicitDefectSchemaError, match="finite support"):
        BulkDefectDocument(
            schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
            defect_model=EXPLICIT_QUASI_STEADY,
            bulk_defects=(incomplete_gaussian,),
        )


@pytest.mark.parametrize(
    ("transition", "neutral_reference", "wrong_reference"),
    [
        (NEUTRAL, NEUTRAL_ALL_OCCUPANCIES, NEUTRAL_WHEN_EMPTY),
        (ACCEPTOR, NEUTRAL_WHEN_EMPTY, NEUTRAL_WHEN_FILLED),
        (DONOR, NEUTRAL_WHEN_FILLED, NEUTRAL_WHEN_EMPTY),
    ],
)
def test_charge_transition_requires_its_physical_neutral_reference(
    transition,
    neutral_reference,
    wrong_reference,
):
    species = _species(
        transition=transition,
        neutral_reference=neutral_reference,
    )

    assert species.explicit_ready
    with pytest.raises(ExplicitDefectSchemaError, match="requires"):
        replace(species, neutral_reference=wrong_reference)


def test_unresolved_scaps_charge_is_preserved_but_not_explicit_ready():
    unresolved = _species(
        name=None,
        transition=UNRESOLVED,
        neutral_reference=UNRESOLVED,
    )

    legacy = _document(model=EFFECTIVE_LIFETIME, species=(unresolved,))
    assert legacy.bulk_defects == (unresolved,)
    with pytest.raises(ExplicitDefectSchemaError, match="charge-resolved"):
        _document(model=EXPLICIT_QUASI_STEADY, species=(unresolved,))


def test_v1_rejects_dynamic_and_gaussian_explicit_execution():
    with pytest.raises(ExplicitDefectSchemaError, match="reserved"):
        _document(model=EXPLICIT_DYNAMIC)

    gaussian = replace(
        _species(),
        distribution=BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.60,
            width_eV=0.08,
            width_convention="gaussian_standard_deviation",
        ),
    )
    with pytest.raises(ExplicitDefectSchemaError, match="single-level"):
        _document(species=(gaussian,))


def test_standard_layer_parser_is_strict_about_document_and_nested_keys():
    payload = _document().to_dict()
    layer = {
        "defect_schema_version": payload["schema_version"],
        "defect_model": payload["defect_model"],
        "bulk_defects": payload["bulk_defects"],
    }
    assert bulk_defect_document_from_layer_mapping(layer) == _document()

    with pytest.raises(ExplicitDefectSchemaError, match="requires"):
        bulk_defect_document_from_layer_mapping(
            {"defect_model": EXPLICIT_QUASI_STEADY, "bulk_defects": []}
        )
    bad_species = payload["bulk_defects"][0] | {"invented_level_eV": 0.3}
    with pytest.raises(ExplicitDefectSchemaError, match="unknown"):
        bulk_defect_document_from_layer_mapping(
            layer | {"bulk_defects": [bad_species]}
        )
    distribution = payload["bulk_defects"][0]["distribution"]
    bad_distribution = distribution | {"peak_density_m3": 1.0e20}
    bad_species = payload["bulk_defects"][0] | {
        "distribution": bad_distribution
    }
    with pytest.raises(ExplicitDefectSchemaError, match="peak_density_m3"):
        bulk_defect_document_from_layer_mapping(
            layer | {"bulk_defects": [bad_species]}
        )


def test_scaps_cgs_adapter_matches_equivalent_canonical_si_species():
    scaps = bulk_defect_species_from_scaps_mapping(
        {
            "name": "acceptor",
            "sigma_n_cm2": 1.0e-15,
            "sigma_p_cm2": 2.0e-15,
            "N_t_cm3": 1.0e16,
            "E_t_eV_above_vb": 0.60,
            "charge_transition": ACCEPTOR,
            "neutral_reference": NEUTRAL_WHEN_EMPTY,
            "degeneracy": 2.0,
        },
        band_gap_eV=1.50,
        layer_thermal_velocity_m_s=1.0e5,
        where="test defect",
    )

    expected = replace(
        _species(name="acceptor"),
        kinetics=replace(
            _species(name="acceptor").kinetics,
            thermal_velocity_p_m_s=1.0e5,
        ),
    )
    assert scaps == expected
    assert _document(species=(scaps,)).sha256 == _document(
        species=(expected,)
    ).sha256


def test_scaps_energy_reference_conversion_and_ambiguity_fail_closed():
    common = {
        "sigma_n_cm2": 1.0e-15,
        "sigma_p_cm2": 1.0e-15,
        "N_t_cm3": 1.0e12,
    }
    below = bulk_defect_species_from_scaps_mapping(
        common | {"E_t_eV_below_cb": 0.1},
        band_gap_eV=1.5,
        layer_thermal_velocity_m_s=1.0e5,
        where="below",
    )
    above = bulk_defect_species_from_scaps_mapping(
        common | {"E_t_eV_above_vb": 1.4},
        band_gap_eV=1.5,
        layer_thermal_velocity_m_s=1.0e5,
        where="above",
    )
    assert below.distribution.center_eV_above_vb == pytest.approx(1.4)
    assert below.distribution == above.distribution

    with pytest.raises(ExplicitDefectSchemaError, match="exactly one"):
        bulk_defect_species_from_scaps_mapping(
            common
            | {
                "E_t_eV_below_cb": 0.1,
                "E_t_eV_above_vb": 1.4,
            },
            band_gap_eV=1.5,
            layer_thermal_velocity_m_s=1.0e5,
            where="ambiguous",
        )
    with pytest.raises(ExplicitDefectSchemaError, match="declared together"):
        bulk_defect_species_from_scaps_mapping(
            common
            | {
                "E_t_eV_above_vb": 0.6,
                "charge_transition": ACCEPTOR,
            },
            band_gap_eV=1.5,
            layer_thermal_velocity_m_s=1.0e5,
            where="partial charge",
        )


def test_scaps_loader_preserves_species_without_changing_legacy_reduction():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror_v2.yaml")
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    document = absorber.params.defect_document

    assert document is not None
    assert document.defect_model == EFFECTIVE_LIFETIME
    assert [item.name for item in document.bulk_defects] == [
        "Perovskite-CB",
        "Perovskite-VB",
    ]
    assert all(
        item.charge_transition == UNRESOLVED
        for item in document.bulk_defects
    )
    legacy_inverse_tau = (
        1.0e-15 * 1.0e-4 * (1.0e7 * 1.0e-2) * (1.0e12 * 1.0e6)
    )
    assert absorber.params.tau_n == 1.0 / (2.0 * legacy_inverse_tau)
    assert absorber.params.tau_p == 1.0 / (2.0 * legacy_inverse_tau)


def test_backend_roundtrip_retains_canonical_scaps_document():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror_v2.yaml")
    payload = _stack_to_config_dict(stack)
    rebuilt = stack_from_dict(payload)

    for original, restored in zip(stack.layers, rebuilt.layers):
        original_document = original.params.defect_document
        restored_document = restored.params.defect_document
        assert restored_document == original_document
        if original_document is not None:
            assert restored_document.sha256 == original_document.sha256
        assert restored.params.tau_n == original.params.tau_n
        assert restored.params.tau_p == original.params.tau_p


def test_default_standard_serializer_does_not_invent_defect_species():
    stack = stack_from_dict(
        _stack_to_config_dict(
            load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
        )
    )
    stripped = replace(
        stack,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=None,
                    defect_model=EFFECTIVE_LIFETIME,
                    bulk_defects=(),
                ),
            )
            for layer in stack.layers
        ),
    )
    payload = _stack_to_config_dict(stripped)

    assert all("bulk_defects" not in layer for layer in payload["layers"])
    assert all("defect_model" not in layer for layer in payload["layers"])


def test_material_params_rejects_old_and_new_bulk_trap_schemas_together():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    params = next(layer.params for layer in stack.layers if layer.role == "absorber")
    old_trap = BulkTrapDistribution(
        distribution="single_level",
        total_density_m3=1.0e22,
        center_eV_above_vb=0.6,
        sigma_n_m2=1.0e-19,
        sigma_p_m2=1.0e-19,
        thermal_velocity_m_s=1.0e5,
        charge_transition="acceptor",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        replace(params, bulk_trap_distribution=old_trap)


def test_charged_explicit_model_remains_fail_closed_in_def1():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    layers = []
    for layer in stack.layers:
        if layer.role != "absorber":
            layers.append(layer)
            continue
        document = _document(species=(_species(),))
        layers.append(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=document.schema_version,
                    defect_model=document.defect_model,
                    bulk_defects=document.bulk_defects,
                ),
            )
        )
    explicit = replace(stack, layers=tuple(layers))
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 3) for layer in explicit.layers),
        alpha=2.0,
    )

    with pytest.raises(ExplicitDefectCapabilityError, match="neutral species only"):
        build_material_arrays(grid, explicit)
    assert semantic_sha256(explicit) != semantic_sha256(stack)


def test_neutral_single_level_model_compiles_without_poisson_charge():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    neutral = replace(
        _species(
            transition=NEUTRAL,
            neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
        ),
        degeneracy=1.0,
    )
    document = _document(species=(neutral,))
    explicit = replace(
        stack,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=document.schema_version,
                    defect_model=document.defect_model,
                    bulk_defects=document.bulk_defects,
                ),
            )
            if layer.role == "absorber"
            else layer
            for layer in stack.layers
        ),
    )
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 3) for layer in explicit.layers),
        alpha=2.0,
    )

    material = build_material_arrays(grid, explicit)

    assert material.neutral_bulk_defects is not None
    assert len(material.neutral_bulk_defects.species) == 1
    assert np.all(material.neutral_bulk_defects.explicit_node_mask == (
        material.Eg_phys == next(
            layer.params.Eg for layer in explicit.layers if layer.role == "absorber"
        )
    ))
    assert material.iface_state_charge == 0.0


def test_def1_rejects_nonunit_degeneracy_instead_of_ignoring_it():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    neutral = _species(
        transition=NEUTRAL,
        neutral_reference=NEUTRAL_ALL_OCCUPANCIES,
    )
    document = _document(species=(neutral,))
    explicit = replace(
        stack,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=document.schema_version,
                    defect_model=document.defect_model,
                    bulk_defects=document.bulk_defects,
                ),
            )
            if layer.role == "absorber"
            else layer
            for layer in stack.layers
        ),
    )
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 3) for layer in explicit.layers),
        alpha=2.0,
    )

    with pytest.raises(ExplicitDefectCapabilityError, match="degeneracy=1.0"):
        build_material_arrays(grid, explicit)


def test_inactive_species_metadata_is_exactly_rhs_inert():
    stripped = load_device_from_yaml(str(ROOT / "configs/cigs_baseline.yaml"))
    legacy_document = _document(
        model=EFFECTIVE_LIFETIME,
        species=(
            _species(
                name=None,
                transition=UNRESOLVED,
                neutral_reference=UNRESOLVED,
            ),
        ),
    )
    stack = replace(
        stripped,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=legacy_document.schema_version,
                    defect_model=legacy_document.defect_model,
                    bulk_defects=legacy_document.bulk_defects,
                ),
            )
            if layer.role == "absorber"
            else layer
            for layer in stripped.layers
        ),
    )
    grid = multilayer_grid(
        tuple(Layer(layer.thickness, 3) for layer in stack.layers),
        alpha=2.0,
    )
    with_metadata = build_material_arrays(grid, stack)
    without_metadata = build_material_arrays(grid, stripped)
    n = np.geomspace(with_metadata.n_L, with_metadata.n_R, grid.size)
    p = np.geomspace(with_metadata.p_L, with_metadata.p_R, grid.size)
    state = StateVec.pack(n, p, with_metadata.P_ion0.copy())

    rhs_with = assemble_rhs(
        0.0,
        state,
        grid,
        stack,
        with_metadata,
        illuminated=False,
    )
    rhs_without = assemble_rhs(
        0.0,
        state,
        grid,
        stripped,
        without_metadata,
        illuminated=False,
    )

    np.testing.assert_array_equal(rhs_with, rhs_without)
    assert semantic_sha256(stack) == semantic_sha256(stripped)


def test_material_parser_consumes_frontend_shaped_canonical_document():
    base = _stack_to_config_dict(
        load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    )["layers"][1]
    document = _document()
    payload = document.to_dict()
    base.update(
        defect_schema_version=payload["schema_version"],
        defect_model=payload["defect_model"],
        bulk_defects=payload["bulk_defects"],
    )

    params = material_params_from_dict(base)

    assert isinstance(params, MaterialParams)
    assert params.defect_document == document


def test_v2_single_level_roundtrips_through_standard_backend_payload():
    stack = load_scaps_yaml(ROOT / "configs/scaps_mirror.yaml")
    v2_species = replace(
        _species(),
        degeneracy=1.0,
        distribution=replace(
            _species().distribution,
            energy_reference=ENERGY_ABOVE_VALENCE_BAND,
        ),
    )
    document = BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=(v2_species,),
    )
    explicit = replace(
        stack,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=document.schema_version,
                    defect_model=document.defect_model,
                    bulk_defects=document.bulk_defects,
                ),
            )
            if layer.role == "absorber"
            else layer
            for layer in stack.layers
        ),
    )

    payload = _stack_to_config_dict(explicit)
    rebuilt = stack_from_dict(payload)
    restored = next(
        layer.params.defect_document
        for layer in rebuilt.layers
        if layer.role == "absorber"
    )

    assert restored == document
    assert restored.sha256 == document.sha256
    assert semantic_sha256(rebuilt) == semantic_sha256(explicit)
    assert semantic_sha256(rebuilt) != semantic_sha256(stack)

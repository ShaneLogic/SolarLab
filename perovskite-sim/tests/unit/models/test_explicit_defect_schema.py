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
    DONOR,
    EFFECTIVE_LIFETIME,
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


def test_explicit_model_loads_as_data_but_solver_fails_closed():
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

    with pytest.raises(ExplicitDefectCapabilityError, match="DEF-0"):
        build_material_arrays(grid, explicit)
    assert semantic_sha256(explicit) != semantic_sha256(stack)


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

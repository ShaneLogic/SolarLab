"""D3-E3 source dispatch and device-level distributed defect closure."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

import perovskite_sim.physics.distributed_defect_closure as distributed_closure

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_SCAPS_CHARACTERISTIC,
    WIDTH_UNIFORM_FULL,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectModel,
    MonovalentDefectClosureResult,
    MonovalentDefectRegion,
    evaluate_monovalent_bulk_defects,
    evaluate_monovalent_defect_closure,
    evaluate_monovalent_source_defect_closure,
    solve_monovalent_defect_charge_neutrality,
)
from perovskite_sim.physics.distributed_defect_closure import (
    EnergyDistributedDefectClosureResult,
    evaluate_energy_distributed_defect_closure,
)


GAP_EV = 1.5
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0


def _distribution(kind: str, *, density_m3: float) -> BulkDefectDistribution:
    values: dict[str, object] = {
        "kind": kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": density_m3,
        "center_eV_above_vb": 0.72,
        "energy_reference": ENERGY_ABOVE_VALENCE_BAND,
    }
    if kind == GAUSSIAN:
        values |= {
            "width_eV": 0.08,
            "width_convention": WIDTH_GAUSSIAN_SIGMA,
            "support_width_multiplier": 8.0,
        }
    elif kind == UNIFORM:
        values |= {
            "width_eV": 0.36,
            "width_convention": WIDTH_UNIFORM_FULL,
        }
    elif kind == CONDUCTION_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 1.45,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    elif kind == VALENCE_BAND_TAIL:
        values |= {
            "center_eV_above_vb": 0.05,
            "width_eV": 0.1,
            "width_convention": WIDTH_SCAPS_CHARACTERISTIC,
            "support_width_multiplier": 7.0,
        }
    return BulkDefectDistribution(**values)


def _species(
    name: str,
    kind: str,
    transition: str,
    *,
    density_m3: float = 3.0e21,
) -> BulkDefectSpecies:
    neutral_reference = {
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
    }[transition]
    return BulkDefectSpecies(
        name=name,
        distribution=_distribution(kind, density_m3=density_m3),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.3e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
    )


def _document_hash(*species: BulkDefectSpecies) -> str:
    return BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=species,
    ).sha256


def _region(
    species: tuple[BulkDefectSpecies, ...],
    *,
    order: int,
    active_nodes: np.ndarray | None = None,
) -> MonovalentDefectRegion:
    return MonovalentDefectRegion(
        identifier="absorber",
        document_sha256=_document_hash(*species),
        active_nodes=(
            np.asarray([True, True, False])
            if active_nodes is None
            else active_nodes
        ),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=species,
        schema_version=EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
        energy_quadrature_order=order,
    )


def _source_evaluate(n, p, species, *, order):
    return evaluate_monovalent_source_defect_closure(
        n,
        p,
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        energy_quadrature_order=order,
    )


def test_source_dispatch_preserves_the_exact_single_level_d2_result():
    source = _species("single", SINGLE_LEVEL, ACCEPTOR)
    n = np.asarray([2.0e17, 4.0e20, 7.0e23])
    p = np.asarray([8.0e22, 3.0e19, 5.0e15])

    dispatched = _source_evaluate(n, p, (source,), order=64)
    direct = evaluate_monovalent_defect_closure(
        n,
        p,
        (source,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    assert isinstance(dispatched, MonovalentDefectClosureResult)
    for item in fields(direct):
        left = getattr(dispatched, item.name)
        right = getattr(direct, item.name)
        if isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_region_expands_all_five_source_kinds_with_auditable_node_ranges():
    species = (
        _species("single", SINGLE_LEVEL, NEUTRAL),
        _species("gaussian", GAUSSIAN, ACCEPTOR),
        _species("uniform", UNIFORM, DONOR),
        _species("cb_tail", CONDUCTION_BAND_TAIL, ACCEPTOR),
        _species("vb_tail", VALENCE_BAND_TAIL, DONOR),
    )
    region = _region(species, order=12)

    assert region.has_distributed_species
    assert region.distribution_kinds == tuple(
        item.distribution.kind for item in species
    )
    assert region.source_energy_orders == (1, 12, 12, 12, 12)
    assert tuple(stop - start for start, stop in region.source_node_ranges) == (
        1,
        12,
        12,
        12,
        12,
    )
    assert len(region.execution_species) == 49
    assert tuple(map(len, region.source_node_identifiers)) == (
        1,
        12,
        12,
        12,
        12,
    )
    assert all(
        len(names) == len(set(names)) for names in region.source_node_identifiers
    )


def test_model_identity_binds_distributed_order_without_changing_single_identity():
    distributed = (_species("gaussian", GAUSSIAN, ACCEPTOR),)
    single = (_species("single", SINGLE_LEVEL, ACCEPTOR),)

    distributed_8 = MonovalentBulkDefectModel(
        regions=(_region(distributed, order=8),)
    )
    distributed_16 = MonovalentBulkDefectModel(
        regions=(_region(distributed, order=16),)
    )
    single_8 = MonovalentBulkDefectModel(regions=(_region(single, order=8),))
    single_64 = MonovalentBulkDefectModel(regions=(_region(single, order=64),))

    assert distributed_8.identity_sha256 != distributed_16.identity_sha256
    assert single_8.identity_sha256 == single_64.identity_sha256
    assert single_8.source_energy_orders == (1,)
    assert single_64.source_energy_orders == (1,)


def test_device_evaluation_matches_the_independent_source_aggregator_exactly():
    species = (
        _species("single", SINGLE_LEVEL, NEUTRAL),
        _species("gaussian", GAUSSIAN, ACCEPTOR),
        _species("cb_tail", CONDUCTION_BAND_TAIL, DONOR),
    )
    region = _region(species, order=20)
    model = MonovalentBulkDefectModel(regions=(region,))
    n = np.asarray([2.0e18, 7.0e21, 9.0e19])
    p = np.asarray([5.0e21, 8.0e17, 4.0e19])
    mask = region.active_nodes

    device = evaluate_monovalent_bulk_defects(n, p, model)
    direct = evaluate_energy_distributed_defect_closure(
        n[mask],
        p[mask],
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        energy_quadrature_order=20,
    )

    assert isinstance(direct, EnergyDistributedDefectClosureResult)
    for index, source in enumerate(direct.source_closures):
        np.testing.assert_array_equal(
            device.occupancy[index, mask], source.mean_occupancy
        )
        np.testing.assert_array_equal(
            device.charge_density_C_m3[index, mask],
            source.charge_density_C_m3,
        )
        np.testing.assert_array_equal(
            device.recombination_rate_m3_s[index, mask],
            source.recombination_rate_m3_s,
        )
        np.testing.assert_array_equal(
            device.kinetic_denominator_s1[index, mask],
            np.min(source.node_closure.kinetic_denominator_s1, axis=0),
        )
    np.testing.assert_array_equal(
        device.total_charge_density_C_m3[mask],
        direct.total_charge_density_C_m3,
    )
    np.testing.assert_array_equal(
        device.total_recombination_rate_m3_s[mask],
        direct.total_recombination_rate_m3_s,
    )
    np.testing.assert_array_equal(
        device.total_recombination_derivative_n_s1[mask],
        direct.total_recombination_derivative_n_s1,
    )
    np.testing.assert_array_equal(
        device.total_recombination_derivative_p_s1[mask],
        direct.total_recombination_derivative_p_s1,
    )
    np.testing.assert_array_equal(
        device.total_charge_derivative_fixed_qf_C_m3_V[mask],
        direct.total_charge_derivative_fixed_qf_C_m3_V,
    )
    assert device.minimum_occupancy == direct.minimum_occupancy
    assert device.maximum_occupancy == direct.maximum_occupancy
    assert device.minimum_kinetic_denominator_s1 == (
        direct.minimum_kinetic_denominator_s1
    )
    assert device.distribution_kinds == (
        SINGLE_LEVEL,
        GAUSSIAN,
        CONDUCTION_BAND_TAIL,
    )
    assert device.source_energy_orders == (1, 20, 20)
    assert device.source_node_identifiers == model.source_node_identifiers
    assert device.to_dict()["source_energy_orders"] == [1, 20, 20]
    assert np.all(device.occupancy[:, ~mask] == 0.0)


def test_distributed_charge_neutrality_uses_the_same_integrated_charge():
    species = (
        _species("gaussian", GAUSSIAN, ACCEPTOR, density_m3=2.0e21),
        _species("vb_tail", VALENCE_BAND_TAIL, DONOR, density_m3=8.0e20),
    )
    results = {
        order: solve_monovalent_defect_charge_neutrality(
            temperature_K=TEMPERATURE_K,
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            acceptor_density_m3=1.0e20,
            donor_density_m3=4.0e20,
            species=species,
            energy_quadrature_order=order,
        )
        for order in (16, 32, 64)
    }

    for result in results.values():
        assert isinstance(result.closure, EnergyDistributedDefectClosureResult)
        state = result.neutrality
        residual = (
            state.hole_density_m3
            - state.electron_density_m3
            + state.ionized_donor_density_m3
            - state.ionized_acceptor_density_m3
            + float(result.closure.total_charge_density_C_m3) / Q
        )
        scale = max(
            state.electron_density_m3,
            state.hole_density_m3,
            state.ionized_acceptor_density_m3,
            state.ionized_donor_density_m3,
            *(item.distribution.total_density_m3 for item in species),
        )
        assert abs(residual) / scale <= 1.0e-12
        assert state.normalized_charge_residual <= 1.0e-12

    n32 = results[32].neutrality.electron_density_m3
    n64 = results[64].neutrality.electron_density_m3
    assert abs(n64 - n32) / max(abs(n64), 1.0) < 5.0e-3


def test_compiled_device_and_contact_neutrality_reuse_energy_expansions(
    monkeypatch,
):
    species = (
        _species("gaussian", GAUSSIAN, ACCEPTOR, density_m3=2.0e21),
        _species("cb_tail", CONDUCTION_BAND_TAIL, DONOR, density_m3=8.0e20),
    )
    region = _region(species, order=16)
    model = MonovalentBulkDefectModel(regions=(region,))

    def reject_reexpansion(*args, **kwargs):
        raise AssertionError("compiled energy nodes must be reused")

    monkeypatch.setattr(
        distributed_closure,
        "expand_bulk_defect_species_energy",
        reject_reexpansion,
    )

    evaluated = evaluate_monovalent_bulk_defects(
        np.asarray([2.0e18, 7.0e21, 9.0e19]),
        np.asarray([5.0e21, 8.0e17, 4.0e19]),
        model,
    )
    neutrality = solve_monovalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=1.0e20,
        donor_density_m3=4.0e20,
        species=species,
        energy_quadrature_order=region.energy_quadrature_order,
        energy_expansions=region.source_expansions,
    )

    assert np.all(np.isfinite(evaluated.total_charge_density_C_m3))
    assert neutrality.neutrality.normalized_charge_residual <= 1.0e-12


@pytest.mark.parametrize(
    "updates",
    (
        {"source_energy_orders": (20, 20, 20)},
        {"source_node_identifiers": (("only-one",),) * 3},
        {"distribution_kinds": ()},
    ),
)
def test_distributed_device_metadata_tampering_fails_closed(updates):
    species = (
        _species("single", SINGLE_LEVEL, NEUTRAL),
        _species("gaussian", GAUSSIAN, ACCEPTOR),
        _species("cb_tail", CONDUCTION_BAND_TAIL, DONOR),
    )
    model = MonovalentBulkDefectModel(regions=(_region(species, order=20),))
    result = evaluate_monovalent_bulk_defects(
        np.asarray([2.0e18, 7.0e21, 9.0e19]),
        np.asarray([5.0e21, 8.0e17, 4.0e19]),
        model,
    )

    with pytest.raises(
        ValueError,
        match="(?:distributed|partial) bulk-defect metadata",
    ):
        replace(result, **updates)

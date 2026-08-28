"""D3-E1/E2 energy-distributed local closure and analytic tangents."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    ENERGY_ABOVE_VALENCE_BAND,
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
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.physics.bulk_traps import (
    BulkTrapDistribution,
    evaluate_bulk_trap_state,
)
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_defect_closure,
)
from perovskite_sim.physics.defect_distributions import (
    density_weighted_mean_occupancy,
    expand_bulk_defect_species_energy,
)
from perovskite_sim.physics.distributed_defect_closure import (
    ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION,
    EnergyDistributedDefectClosureCapabilityError,
    evaluate_energy_distributed_defect_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 1.5
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0
TOTAL_DENSITY_M3 = 3.0e21


def _distribution(kind: str, **updates: object) -> BulkDefectDistribution:
    values: dict[str, object] = {
        "kind": kind,
        "normalization": INTEGRATED_TOTAL,
        "total_density_m3": TOTAL_DENSITY_M3,
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
            "width_eV": 0.40,
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
    values.update(updates)
    return BulkDefectDistribution(**values)


def _species(
    name: str,
    kind: str,
    transition: str = ACCEPTOR,
    **distribution_updates: object,
) -> BulkDefectSpecies:
    neutral_reference = {
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
    }[transition]
    return BulkDefectSpecies(
        name=name,
        distribution=_distribution(kind, **distribution_updates),
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


def _evaluate(n, p, *species, order=32):
    return evaluate_energy_distributed_defect_closure(
        n,
        p,
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        energy_quadrature_order=order,
    )


def test_v2_single_level_is_exactly_the_d2_local_closure():
    species = _species("single", SINGLE_LEVEL)
    n = np.asarray([3.0e16, 2.0e20, 8.0e23])
    p = np.asarray([6.0e22, 4.0e19, 7.0e15])

    distributed = _evaluate(n, p, species, order=64)
    direct = evaluate_monovalent_defect_closure(
        n,
        p,
        (species,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    source = distributed.source_closures[0]

    assert source.node_closure.closure_identity_sha256 == (
        direct.closure_identity_sha256
    )
    assert source.quadrature.order == 1
    assert np.array_equal(source.mean_occupancy, direct.occupancy[0])
    np.testing.assert_array_equal(
        source.occupied_density_m3,
        direct.occupied_density_m3[0],
    )
    np.testing.assert_array_equal(
        distributed.total_charge_density_C_m3,
        direct.total_charge_density_C_m3,
    )
    np.testing.assert_array_equal(
        distributed.total_recombination_rate_m3_s,
        direct.total_recombination_rate_m3_s,
    )
    np.testing.assert_array_equal(
        distributed.total_recombination_derivative_n_s1,
        direct.total_recombination_derivative_n_s1,
    )
    np.testing.assert_array_equal(
        distributed.total_recombination_derivative_p_s1,
        direct.total_recombination_derivative_p_s1,
    )
    np.testing.assert_array_equal(
        distributed.total_charge_derivative_fixed_qf_C_m3_V,
        direct.total_charge_derivative_fixed_qf_C_m3_V,
    )


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_source_aggregation_is_the_exact_sum_of_d2_energy_nodes(kind):
    species = _species("source", kind)
    n = np.asarray([1.0e18, 2.0e21])
    p = np.asarray([3.0e21, 7.0e17])

    result = _evaluate(n, p, species, order=24)
    expansion = expand_bulk_defect_species_energy(
        species,
        band_gap_eV=GAP_EV,
        order=24,
    )
    nodes = evaluate_monovalent_defect_closure(
        n,
        p,
        expansion.node_species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    source = result.source_closures[0]

    assert source.node_closure.closure_identity_sha256 == (
        nodes.closure_identity_sha256
    )
    assert source.quadrature.to_dict() == expansion.quadrature.to_dict()
    np.testing.assert_array_equal(
        source.recombination_rate_m3_s,
        np.sum(nodes.recombination_rate_m3_s, axis=0),
    )
    np.testing.assert_array_equal(
        source.charge_density_C_m3,
        np.sum(nodes.charge_density_C_m3, axis=0),
    )
    np.testing.assert_array_equal(
        source.mean_occupancy,
        density_weighted_mean_occupancy(
            nodes.occupancy,
            expansion.quadrature.density_weights_m3,
            TOTAL_DENSITY_M3,
        ),
    )


def test_nearly_saturated_tail_mean_stays_bounded_without_clipping():
    result = _evaluate(
        1.0e34,
        1.0,
        _species("saturated_tail", VALENCE_BAND_TAIL, DONOR),
        order=16,
    )
    source = result.source_closures[0]

    assert np.all(source.node_closure.occupancy <= 1.0)
    assert np.all(source.mean_occupancy <= 1.0)
    np.testing.assert_array_equal(
        source.mean_occupancy,
        density_weighted_mean_occupancy(
            source.node_closure.occupancy,
            source.quadrature.density_weights_m3,
            TOTAL_DENSITY_M3,
        ),
    )


def test_prepared_energy_expansion_must_match_requested_order():
    species = _species("source", GAUSSIAN)
    expansion = expand_bulk_defect_species_energy(
        species,
        band_gap_eV=GAP_EV,
        order=8,
    )

    with pytest.raises(ValueError, match="source/order protocol"):
        evaluate_energy_distributed_defect_closure(
            1.0e20,
            2.0e20,
            (species,),
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=TEMPERATURE_K,
            energy_quadrature_order=16,
            energy_expansions=(expansion,),
        )


def test_gaussian_matches_the_preexisting_research_primitive_on_same_support():
    source = _species(
        "gaussian",
        GAUSSIAN,
        center_eV_above_vb=0.75,
        width_eV=0.125,
        support_width_multiplier=12.0,
    )
    source = replace(
        source,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=1.0e5,
        ),
    )
    legacy = BulkTrapDistribution(
        distribution=GAUSSIAN,
        total_density_m3=TOTAL_DENSITY_M3,
        center_eV_above_vb=0.75,
        sigma_n_m2=2.0e-19,
        sigma_p_m2=7.0e-20,
        thermal_velocity_m_s=1.0e5,
        charge_transition=ACCEPTOR,
        energy_sigma_eV=0.125,
    )
    n = np.asarray([2.0e18, 7.0e21])
    p = np.asarray([5.0e21, 8.0e17])

    result = _evaluate(n, p, source, order=32).source_closures[0]
    reference = evaluate_bulk_trap_state(
        n,
        p,
        legacy,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        quadrature_order=32,
    )

    np.testing.assert_allclose(result.mean_occupancy, reference.occupancy, rtol=2e-15)
    np.testing.assert_allclose(
        result.recombination_rate_m3_s,
        reference.recombination_rate_m3_s,
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        result.charge_density_C_m3,
        reference.charge_density_C_m3,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        result.recombination_derivative_n_s1,
        reference.recombination_derivative_n_s,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        result.recombination_derivative_p_s1,
        reference.recombination_derivative_p_s,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        result.charge_derivative_n_C / Q,
        reference.charge_number_derivative_n,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        result.charge_derivative_p_C / Q,
        reference.charge_number_derivative_p,
        rtol=2e-13,
    )


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_analytic_carrier_and_fixed_qf_tangents_match_centered_difference(kind):
    source = _species("source", kind)
    n = 4.0e20
    p = 8.0e19
    result = _evaluate(n, p, source, order=40)
    relative_step = 2.0e-6

    n_plus = _evaluate(n * (1.0 + relative_step), p, source, order=40)
    n_minus = _evaluate(n * (1.0 - relative_step), p, source, order=40)
    p_plus = _evaluate(n, p * (1.0 + relative_step), source, order=40)
    p_minus = _evaluate(n, p * (1.0 - relative_step), source, order=40)
    rate_n = (
        n_plus.total_recombination_rate_m3_s
        - n_minus.total_recombination_rate_m3_s
    ) / (2.0 * relative_step * n)
    rate_p = (
        p_plus.total_recombination_rate_m3_s
        - p_minus.total_recombination_rate_m3_s
    ) / (2.0 * relative_step * p)
    charge_n = (
        n_plus.total_charge_density_C_m3
        - n_minus.total_charge_density_C_m3
    ) / (2.0 * relative_step * n)
    charge_p = (
        p_plus.total_charge_density_C_m3
        - p_minus.total_charge_density_C_m3
    ) / (2.0 * relative_step * p)

    assert result.total_recombination_derivative_n_s1 == pytest.approx(
        rate_n,
        rel=2.0e-8,
    )
    assert result.total_recombination_derivative_p_s1 == pytest.approx(
        rate_p,
        rel=2.0e-8,
    )
    assert result.total_charge_derivative_n_C == pytest.approx(
        charge_n,
        rel=2.0e-8,
    )
    assert result.total_charge_derivative_p_C == pytest.approx(
        charge_p,
        rel=2.0e-8,
    )

    thermal = thermal_voltage(TEMPERATURE_K)
    potential_step = 1.0e-7
    factor = math.exp(potential_step / thermal)
    plus = _evaluate(n * factor, p / factor, source, order=40)
    minus = _evaluate(n / factor, p * factor, source, order=40)
    fixed_qf_charge = (
        plus.total_charge_density_C_m3
        - minus.total_charge_density_C_m3
    ) / (2.0 * potential_step)
    fixed_qf_rate = (
        plus.total_recombination_rate_m3_s
        - minus.total_recombination_rate_m3_s
    ) / (2.0 * potential_step)
    assert result.total_charge_derivative_fixed_qf_C_m3_V == pytest.approx(
        fixed_qf_charge,
        rel=2.0e-8,
    )
    assert (
        result.total_recombination_derivative_fixed_qf_m3_s_V
        == pytest.approx(fixed_qf_rate, rel=2.0e-8)
    )


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_equilibrium_detailed_balance_and_charge_signs(kind):
    thermal = thermal_voltage(TEMPERATURE_K)
    ni_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal)
    n = 2.0e19
    p = ni_sq / n
    acceptor = _evaluate(n, p, _species("acceptor", kind, ACCEPTOR), order=32)
    donor = _evaluate(n, p, _species("donor", kind, DONOR), order=32)
    neutral = _evaluate(n, p, _species("neutral", kind, NEUTRAL), order=32)

    scale = TOTAL_DENSITY_M3 * 2.0e-19 * 1.3e5 * max(n, p)
    assert abs(acceptor.total_recombination_rate_m3_s.item()) < 2.0e-14 * scale
    assert abs(donor.total_recombination_rate_m3_s.item()) < 2.0e-14 * scale
    assert acceptor.total_charge_density_C_m3.item() < 0.0
    assert donor.total_charge_density_C_m3.item() > 0.0
    assert neutral.total_charge_density_C_m3.item() == 0.0
    assert 0.0 <= acceptor.minimum_occupancy
    assert acceptor.maximum_occupancy <= 1.0


@pytest.mark.parametrize(
    "kind",
    (GAUSSIAN, UNIFORM, CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_narrow_distribution_recovers_single_level(kind):
    center = 0.68
    single = _species(
        "single",
        SINGLE_LEVEL,
        center_eV_above_vb=center,
    )
    narrow = _species(
        "narrow",
        kind,
        center_eV_above_vb=center,
        width_eV=1.0e-7,
    )
    n = 7.0e20
    p = 3.0e19

    reference = _evaluate(n, p, single, order=32)
    result = _evaluate(n, p, narrow, order=32)

    assert result.total_recombination_rate_m3_s == pytest.approx(
        reference.total_recombination_rate_m3_s,
        rel=2.0e-11,
    )
    assert result.total_charge_density_C_m3 == pytest.approx(
        reference.total_charge_density_C_m3,
        rel=2.0e-12,
    )


def test_energy_order_refinement_is_independent_and_below_half_percent():
    species = (
        _species("gaussian", GAUSSIAN, ACCEPTOR),
        _species("uniform", UNIFORM, DONOR, center_eV_above_vb=0.82),
    )
    n = 6.0e20
    p = 2.0e19
    results = {order: _evaluate(n, p, *species, order=order) for order in (8, 16, 32)}

    def relative_change(field: str, coarse: int, fine: int) -> float:
        left = float(np.asarray(getattr(results[coarse], field)))
        right = float(np.asarray(getattr(results[fine], field)))
        return abs(right - left) / max(abs(right), 1.0)

    for field in (
        "total_recombination_rate_m3_s",
        "total_charge_density_C_m3",
        "total_recombination_derivative_n_s1",
        "total_recombination_derivative_p_s1",
        "total_charge_derivative_fixed_qf_C_m3_V",
    ):
        assert relative_change(field, 16, 32) < 5.0e-3
        assert relative_change(field, 16, 32) <= relative_change(field, 8, 16)
    assert results[8].energy_orders == (8, 8)
    assert results[32].energy_orders == (32, 32)
    assert results[8].closure_identity_sha256 != results[32].closure_identity_sha256


def test_source_permutation_leaves_total_arrays_exactly_unchanged():
    sources = (
        _species("z_gaussian", GAUSSIAN, ACCEPTOR),
        _species("a_uniform", UNIFORM, DONOR, center_eV_above_vb=0.82),
        _species("m_single", SINGLE_LEVEL, NEUTRAL),
    )
    n = np.asarray([2.0e18, 5.0e20])
    p = np.asarray([6.0e21, 4.0e19])

    forward = _evaluate(n, p, *sources, order=20)
    reverse = _evaluate(n, p, *reversed(sources), order=20)

    assert forward.source_identifiers == tuple(item.name for item in sources)
    assert reverse.source_identifiers == tuple(
        item.name for item in reversed(sources)
    )
    assert forward.closure_identity_sha256 != reverse.closure_identity_sha256
    for field in (
        "total_charge_density_C_m3",
        "total_recombination_rate_m3_s",
        "total_recombination_derivative_n_s1",
        "total_recombination_derivative_p_s1",
        "total_charge_derivative_n_C",
        "total_charge_derivative_p_C",
        "total_charge_derivative_fixed_qf_C_m3_V",
        "total_recombination_derivative_fixed_qf_m3_s_V",
    ):
        np.testing.assert_array_equal(
            getattr(forward, field),
            getattr(reverse, field),
        )


@pytest.mark.parametrize(
    "kind",
    (CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_band_tail_closure_retains_source_and_node_evidence(kind):
    result = _evaluate(
        np.asarray([1.0e18, 1.0e20]),
        np.asarray([2.0e21, 3.0e19]),
        _species("tail", kind),
        order=24,
    )

    source = result.source_closures[0]
    assert source.distribution_kind == kind
    assert source.quadrature.order == 24
    assert source.quadrature.integrated_density_m3 == pytest.approx(
        TOTAL_DENSITY_M3,
        rel=8.0e-16,
    )
    assert len(source.node_closure.species_identifiers) == 24
    assert np.all(np.isfinite(result.total_recombination_rate_m3_s))
    assert np.all(np.isfinite(result.total_charge_density_C_m3))


@pytest.mark.parametrize(
    "kind",
    (CONDUCTION_BAND_TAIL, VALENCE_BAND_TAIL),
)
def test_band_tail_inverse_cdf_matches_direct_energy_quadrature(kind):
    source = _species("tail", kind)
    n = 4.0e20
    p = 7.0e19
    result = _evaluate(n, p, source, order=128).source_closures[0]

    lower, upper = source.distribution.support_bounds_eV()
    direct_nodes, direct_weights = np.polynomial.legendre.leggauss(256)
    energies = lower + 0.5 * (direct_nodes + 1.0) * (upper - lower)
    energy_weights = 0.5 * (upper - lower) * direct_weights
    center = source.distribution.center_eV_above_vb
    width = source.distribution.width_eV
    if kind == CONDUCTION_BAND_TAIL:
        shape = np.exp((energies - center) / width)
    else:
        shape = np.exp(-(energies - center) / width)
    density_weights = shape * energy_weights
    density_weights *= TOTAL_DENSITY_M3 / math.fsum(density_weights)
    density_weights[-1] += TOTAL_DENSITY_M3 - math.fsum(density_weights)
    direct_species = tuple(
        replace(
            source,
            name=f"direct::{index:03d}",
            distribution=BulkDefectDistribution(
                kind=SINGLE_LEVEL,
                normalization=INTEGRATED_TOTAL,
                total_density_m3=float(density),
                center_eV_above_vb=float(energy),
                energy_reference=ENERGY_ABOVE_VALENCE_BAND,
            ),
        )
        for index, (energy, density) in enumerate(
            zip(energies, density_weights, strict=True)
        )
    )
    direct = evaluate_monovalent_defect_closure(
        n,
        p,
        direct_species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    np.testing.assert_allclose(
        result.mean_occupancy,
        np.sum(direct.occupied_density_m3, axis=0) / TOTAL_DENSITY_M3,
        rtol=2.0e-10,
        atol=2.0e-13,
    )
    for integrated_field, node_field in (
        ("charge_density_C_m3", "charge_density_C_m3"),
        ("recombination_rate_m3_s", "recombination_rate_m3_s"),
        ("recombination_derivative_n_s1", "recombination_derivative_n_s1"),
        ("recombination_derivative_p_s1", "recombination_derivative_p_s1"),
        ("charge_derivative_n_C", "charge_derivative_n_C"),
        ("charge_derivative_p_C", "charge_derivative_p_C"),
        (
            "charge_derivative_fixed_qf_C_m3_V",
            "charge_derivative_fixed_qf_C_m3_V",
        ),
        (
            "recombination_derivative_fixed_qf_m3_s_V",
            "recombination_derivative_fixed_qf_m3_s_V",
        ),
    ):
        np.testing.assert_allclose(
            getattr(result, integrated_field),
            np.sum(getattr(direct, node_field), axis=0),
            rtol=8.0e-9,
            atol=0.0,
        )


def test_source_capability_failure_keeps_source_identity_and_reason():
    source = replace(_species("bad_gaussian", GAUSSIAN), degeneracy=2.0)

    with pytest.raises(
        EnergyDistributedDefectClosureCapabilityError,
        match="bad_gaussian.*degeneracy=1.0",
    ):
        _evaluate(1.0e20, 1.0e20, source)


def test_result_payload_retains_source_node_and_quadrature_evidence():
    result = _evaluate(
        4.0e20,
        7.0e19,
        _species("gaussian", GAUSSIAN),
        _species("uniform", UNIFORM, DONOR),
        order=12,
    )
    payload = result.to_dict()

    assert payload["closure"] == ENERGY_DISTRIBUTED_DEFECT_CLOSURE_VERSION
    assert payload["closure_identity_sha256"] == result.closure_identity_sha256
    assert payload["source_identifiers"] == ["gaussian", "uniform"]
    assert payload["distribution_kinds"] == [GAUSSIAN, UNIFORM]
    assert payload["energy_orders"] == [12, 12]
    assert len(payload["source_closures"]) == 2
    assert all(
        len(item["quadrature"]["energy_levels_eV_above_vb"]) == 12
        and len(item["node_closure"]["species_identifiers"]) == 12
        for item in payload["source_closures"]
    )


def test_result_contract_rejects_inconsistent_source_total_or_identity():
    result = _evaluate(
        4.0e20,
        7.0e19,
        _species("gaussian", GAUSSIAN),
        order=12,
    )
    source = result.source_closures[0]

    with pytest.raises(ValueError, match="exact node aggregate"):
        replace(
            source,
            charge_density_C_m3=source.charge_density_C_m3 + 1.0,
        )
    with pytest.raises(ValueError, match="stable source sum"):
        replace(
            result,
            total_charge_density_C_m3=(
                result.total_charge_density_C_m3 + 1.0
            ),
        )
    with pytest.raises(ValueError, match="identity is inconsistent"):
        replace(result, closure_identity_sha256="0" * 64)

"""DEF-2 tests for the solver-independent monovalent defect closure."""

from __future__ import annotations

from dataclasses import fields, replace
import json
import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    GAUSSIAN,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_ALL_OCCUPANCIES,
    NEUTRAL_REFERENCE_UNRESOLVED,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    UNRESOLVED,
    WIDTH_GAUSSIAN_SIGMA,
    BulkDefectDistribution,
    BulkDefectDocument,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.physics.bulk_traps import (
    BulkTrapDistribution,
    evaluate_bulk_trap_state,
)
from perovskite_sim.physics.defect_closure import (
    MONOVALENT_DEFECT_CLOSURE_VERSION,
    MonovalentBulkDefectModel,
    MonovalentDefectClosureCapabilityError,
    MonovalentDefectRegion,
    evaluate_monovalent_bulk_defects,
    evaluate_monovalent_defect_closure,
    solve_monovalent_defect_charge_neutrality,
)
from perovskite_sim.physics.recombination import srh_recombination_derivatives
from perovskite_sim.physics.statistics import solve_charge_neutrality
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 1.50
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0


def _species(
    name: str,
    transition: str,
    *,
    density_m3: float = 3.0e21,
    center_eV: float = 0.62,
    sigma_n_m2: float = 2.0e-19,
    sigma_p_m2: float = 7.0e-20,
    velocity_n_m_s: float = 1.3e5,
    velocity_p_m_s: float = 8.0e4,
    degeneracy: float = 1.0,
) -> BulkDefectSpecies:
    neutral_reference = {
        NEUTRAL: NEUTRAL_ALL_OCCUPANCIES,
        ACCEPTOR: NEUTRAL_WHEN_EMPTY,
        DONOR: NEUTRAL_WHEN_FILLED,
        UNRESOLVED: NEUTRAL_REFERENCE_UNRESOLVED,
    }[transition]
    return BulkDefectSpecies(
        name=name,
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=density_m3,
            center_eV_above_vb=center_eV,
        ),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=sigma_n_m2,
            sigma_p_m2=sigma_p_m2,
            thermal_velocity_n_m_s=velocity_n_m_s,
            thermal_velocity_p_m_s=velocity_p_m_s,
        ),
        degeneracy=degeneracy,
    )


def _evaluate(n, p, *species):
    return evaluate_monovalent_defect_closure(
        n,
        p,
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )


def _document_hash(*species: BulkDefectSpecies) -> str:
    return BulkDefectDocument(
        schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
        defect_model=EXPLICIT_QUASI_STEADY,
        bulk_defects=species,
    ).sha256


def test_acceptor_single_level_matches_closed_form_occupancy_rate_and_charge():
    species = _species("acceptor", ACCEPTOR)
    n = 4.0e19
    p = 7.0e18
    result = _evaluate(n, p, species)
    thermal = thermal_voltage(TEMPERATURE_K)
    n1 = NC_M3 * math.exp(-(GAP_EV - 0.62) / thermal)
    p1 = NV_M3 * math.exp(-0.62 / thermal)
    capture_n = 2.0e-19 * 1.3e5
    capture_p = 7.0e-20 * 8.0e4
    denominator = capture_n * (n + n1) + capture_p * (p + p1)
    occupancy = (capture_n * n + capture_p * p1) / denominator
    ni_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal)
    rate = (
        3.0e21
        * capture_n
        * capture_p
        * (n * p - ni_sq)
        / denominator
    )

    assert result.occupancy[0].item() == pytest.approx(occupancy)
    assert result.recombination_rate_m3_s[0].item() == pytest.approx(rate)
    assert result.charge_density_C_m3[0].item() == pytest.approx(
        -Q * 3.0e21 * occupancy
    )


def test_donor_and_acceptor_share_occupancy_but_use_distinct_charge_reference():
    acceptor = _species("acceptor", ACCEPTOR)
    donor = _species("donor", DONOR)
    result = _evaluate(3.0e20, 5.0e18, acceptor, donor)

    assert result.occupancy[0].item() == pytest.approx(result.occupancy[1].item())
    assert (
        result.signed_charge_number_density_m3[1].item()
        - result.signed_charge_number_density_m3[0].item()
    ) == pytest.approx(acceptor.distribution.total_density_m3)
    np.testing.assert_array_equal(
        result.charge_derivative_n_C[0],
        result.charge_derivative_n_C[1],
    )
    np.testing.assert_array_equal(
        result.charge_derivative_p_C[0],
        result.charge_derivative_p_C[1],
    )


@pytest.mark.parametrize("transition", [ACCEPTOR, DONOR])
def test_canonical_single_level_matches_existing_charged_trap_primitive(transition):
    species = _species(
        transition,
        transition,
        velocity_n_m_s=1.0e5,
        velocity_p_m_s=1.0e5,
    )
    n = np.asarray([2.0e17, 8.0e19, 4.0e21])
    p = np.asarray([3.0e21, 5.0e19, 9.0e17])
    canonical = _evaluate(n, p, species)
    legacy = evaluate_bulk_trap_state(
        n,
        p,
        BulkTrapDistribution(
            distribution=SINGLE_LEVEL,
            total_density_m3=species.distribution.total_density_m3,
            center_eV_above_vb=species.distribution.center_eV_above_vb,
            sigma_n_m2=species.kinetics.sigma_n_m2,
            sigma_p_m2=species.kinetics.sigma_p_m2,
            thermal_velocity_m_s=1.0e5,
            charge_transition=transition,
        ),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    np.testing.assert_allclose(canonical.occupancy[0], legacy.occupancy, rtol=2e-15)
    np.testing.assert_allclose(
        canonical.recombination_rate_m3_s[0],
        legacy.recombination_rate_m3_s,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        canonical.charge_density_C_m3[0],
        legacy.charge_density_C_m3,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        canonical.charge_derivative_fixed_qf_C_m3_V[0] / Q,
        legacy.charge_number_derivative_potential_m3_V,
        rtol=3e-15,
    )


def test_neutral_transition_recovers_def1_srh_and_has_exactly_zero_charge():
    species = _species("neutral", NEUTRAL)
    n = np.asarray([2.0e17, 8.0e19, 4.0e21])
    p = np.asarray([3.0e21, 5.0e19, 9.0e17])
    result = _evaluate(n, p, species)
    capture_n = result.capture_n_m3_s[0]
    capture_p = result.capture_p_m3_s[0]
    density = species.distribution.total_density_m3
    ni_sq = NC_M3 * NV_M3 * math.exp(
        -GAP_EV / thermal_voltage(TEMPERATURE_K)
    )
    def1 = srh_recombination_derivatives(
        n,
        p,
        ni_sq,
        1.0 / (capture_n * density),
        1.0 / (capture_p * density),
        result.n1_m3[0],
        result.p1_m3[0],
    )

    np.testing.assert_allclose(result.recombination_rate_m3_s[0], def1.rate, rtol=3e-15)
    np.testing.assert_allclose(
        result.recombination_derivative_n_s1[0],
        def1.electron_density_derivative,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        result.recombination_derivative_p_s1[0],
        def1.hole_density_derivative,
        rtol=1e-11,
    )
    np.testing.assert_array_equal(result.charge_density_C_m3[0], np.zeros(n.shape))
    np.testing.assert_array_equal(result.charge_derivative_n_C[0], np.zeros(n.shape))


def test_equilibrium_mass_action_gives_zero_recombination_without_clipping():
    ni_sq = NC_M3 * NV_M3 * math.exp(
        -GAP_EV / thermal_voltage(TEMPERATURE_K)
    )
    n = np.asarray([1.0e12, math.sqrt(ni_sq), 1.0e23])
    p = ni_sq / n
    result = _evaluate(
        n,
        p,
        _species("acceptor", ACCEPTOR),
        _species("donor", DONOR, center_eV=0.91),
    )

    scale = np.maximum(
        np.abs(result.recombination_derivative_n_s1 * n[None, :]),
        1.0,
    )
    assert float(np.max(np.abs(result.recombination_rate_m3_s) / scale)) < 2.0e-15
    assert 0.0 <= result.minimum_occupancy <= result.maximum_occupancy <= 1.0


@pytest.mark.parametrize(
    ("transition", "acceptors", "donors", "carrier_name", "charge_sign"),
    [
        (ACCEPTOR, 0.0, 2.0e20, "electron_density_m3", -1.0),
        (DONOR, 2.0e20, 0.0, "hole_density_m3", 1.0),
    ],
)
def test_common_fermi_contact_closure_closes_charge_and_mass_action(
    transition,
    acceptors,
    donors,
    carrier_name,
    charge_sign,
):
    defect = _species(
        transition,
        transition,
        density_m3=8.0e21,
        center_eV=0.73,
    )
    result = solve_monovalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
        species=(defect,),
    )
    state = result.neutrality
    signed_defect_charge = float(
        np.sum(result.closure.signed_charge_number_density_m3)
    )
    residual = (
        state.hole_density_m3
        - state.electron_density_m3
        + donors
        - acceptors
        + signed_defect_charge
    )
    charge_scale = max(
        state.electron_density_m3,
        state.hole_density_m3,
        acceptors,
        donors,
        defect.distribution.total_density_m3,
    )
    intrinsic_product = NC_M3 * NV_M3 * math.exp(
        -GAP_EV / thermal_voltage(TEMPERATURE_K)
    )
    recombination_scale = max(
        abs(
            result.closure.total_recombination_derivative_n_s1.item()
            * state.electron_density_m3
        ),
        1.0,
    )

    assert abs(residual) / charge_scale < 1.0e-12
    assert state.normalized_charge_residual < 1.0e-12
    assert state.electron_density_m3 * state.hole_density_m3 == pytest.approx(
        intrinsic_product,
        rel=2.0e-14,
    )
    assert (
        abs(result.closure.total_recombination_rate_m3_s.item())
        / recombination_scale
        < 2.0e-14
    )
    assert 0.0 < result.closure.occupancy.item() < 1.0
    assert charge_sign * signed_defect_charge > 0.0
    assert getattr(state, carrier_name) < 2.0e20


def test_neutral_species_contact_closure_recovers_legacy_neutrality():
    defect = _species("neutral", NEUTRAL, density_m3=8.0e21)
    explicit = solve_monovalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=3.0e20,
        donor_density_m3=0.0,
        species=(defect,),
    )
    legacy = solve_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=3.0e20,
        donor_density_m3=0.0,
    )

    assert explicit.neutrality.electron_density_m3 == pytest.approx(
        legacy.electron_density_m3,
        rel=2.0e-14,
    )
    assert explicit.neutrality.hole_density_m3 == pytest.approx(
        legacy.hole_density_m3,
        rel=2.0e-14,
    )
    assert explicit.neutrality.reduced_electron_fermi_level == pytest.approx(
        legacy.reduced_electron_fermi_level,
        abs=2.0e-14,
    )
    assert explicit.closure.total_charge_density_C_m3.item() == 0.0


def test_all_local_tangents_match_independent_centered_differences():
    species = (
        _species("acceptor", ACCEPTOR),
        _species("donor", DONOR, center_eV=0.93),
        _species("neutral", NEUTRAL, center_eV=0.74),
    )
    n = 2.0e20
    p = 8.0e18
    result = _evaluate(n, p, *species)
    relative_step = 2.0e-6
    n_plus = _evaluate(n * (1.0 + relative_step), p, *species)
    n_minus = _evaluate(n * (1.0 - relative_step), p, *species)
    p_plus = _evaluate(n, p * (1.0 + relative_step), *species)
    p_minus = _evaluate(n, p * (1.0 - relative_step), *species)
    finite_rate_n = (
        n_plus.recombination_rate_m3_s - n_minus.recombination_rate_m3_s
    ) / (2.0 * relative_step * n)
    finite_rate_p = (
        p_plus.recombination_rate_m3_s - p_minus.recombination_rate_m3_s
    ) / (2.0 * relative_step * p)
    finite_charge_n = (
        n_plus.charge_density_C_m3 - n_minus.charge_density_C_m3
    ) / (2.0 * relative_step * n)
    finite_charge_p = (
        p_plus.charge_density_C_m3 - p_minus.charge_density_C_m3
    ) / (2.0 * relative_step * p)
    finite_occupancy_n = (n_plus.occupancy - n_minus.occupancy) / (
        2.0 * relative_step * n
    )
    finite_occupancy_p = (p_plus.occupancy - p_minus.occupancy) / (
        2.0 * relative_step * p
    )

    np.testing.assert_allclose(result.recombination_derivative_n_s1, finite_rate_n, rtol=3e-7)
    np.testing.assert_allclose(result.recombination_derivative_p_s1, finite_rate_p, rtol=3e-7)
    np.testing.assert_allclose(result.charge_derivative_n_C, finite_charge_n, rtol=3e-7, atol=1e-40)
    np.testing.assert_allclose(result.charge_derivative_p_C, finite_charge_p, rtol=3e-7, atol=1e-40)
    np.testing.assert_allclose(result.occupancy_derivative_n_m3, finite_occupancy_n, rtol=3e-7, atol=1e-40)
    np.testing.assert_allclose(result.occupancy_derivative_p_m3, finite_occupancy_p, rtol=3e-7, atol=1e-40)

    potential_step = 1.0e-7
    thermal = thermal_voltage(TEMPERATURE_K)
    plus = _evaluate(
        n * math.exp(potential_step / thermal),
        p * math.exp(-potential_step / thermal),
        *species,
    )
    minus = _evaluate(
        n * math.exp(-potential_step / thermal),
        p * math.exp(potential_step / thermal),
        *species,
    )
    np.testing.assert_allclose(
        result.charge_derivative_fixed_qf_C_m3_V,
        (plus.charge_density_C_m3 - minus.charge_density_C_m3)
        / (2.0 * potential_step),
        rtol=3e-8,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        result.recombination_derivative_fixed_qf_m3_s_V,
        (plus.recombination_rate_m3_s - minus.recombination_rate_m3_s)
        / (2.0 * potential_step),
        rtol=3e-8,
    )


def test_multiple_species_totals_identity_and_serialization_are_closed():
    species = (
        _species("acceptor", ACCEPTOR),
        _species("donor", DONOR, center_eV=0.91),
        _species("neutral", NEUTRAL, center_eV=0.73),
    )
    result = _evaluate(np.asarray([1.0e18, 3.0e20]), 7.0e19, *species)

    np.testing.assert_array_equal(
        result.total_recombination_rate_m3_s,
        np.sum(result.recombination_rate_m3_s, axis=0),
    )
    np.testing.assert_array_equal(
        result.total_charge_density_C_m3,
        np.sum(result.charge_density_C_m3, axis=0),
    )
    changed = _evaluate(
        np.asarray([1.0e18, 3.0e20]),
        7.0e19,
        replace(
            species[0],
            distribution=replace(
                species[0].distribution,
                total_density_m3=4.0e21,
            ),
        ),
        *species[1:],
    )
    assert changed.closure_identity_sha256 != result.closure_identity_sha256
    payload = result.to_dict()
    assert payload["closure"] == MONOVALENT_DEFECT_CLOSURE_VERSION
    assert json.loads(json.dumps(payload))["species_identifiers"] == [
        "acceptor",
        "donor",
        "neutral",
    ]
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def test_device_model_aggregates_disjoint_regions_and_overlapping_species():
    left_mask = np.asarray([True, True, False, False])
    right_mask = ~left_mask
    left_species = (
        _species("acceptor", ACCEPTOR),
        _species("neutral", NEUTRAL, center_eV=0.73),
    )
    right_species = (_species("donor", DONOR, center_eV=0.91),)
    model = MonovalentBulkDefectModel(
        regions=(
            MonovalentDefectRegion(
                identifier="layer[0]/left",
                document_sha256=_document_hash(*left_species),
                active_nodes=left_mask,
                band_gap_eV=GAP_EV,
                effective_conduction_dos_m3=NC_M3,
                effective_valence_dos_m3=NV_M3,
                temperature_K=TEMPERATURE_K,
                species=left_species,
            ),
            MonovalentDefectRegion(
                identifier="layer[1]/right",
                document_sha256=_document_hash(*right_species),
                active_nodes=right_mask,
                band_gap_eV=GAP_EV,
                effective_conduction_dos_m3=NC_M3,
                effective_valence_dos_m3=NV_M3,
                temperature_K=TEMPERATURE_K,
                species=right_species,
            ),
        )
    )
    result = evaluate_monovalent_bulk_defects(
        np.geomspace(1.0e18, 1.0e21, 4),
        np.geomspace(1.0e21, 1.0e18, 4),
        model,
    )

    assert result.species_identifiers == (
        "layer[0]/left/acceptor",
        "layer[0]/left/neutral",
        "layer[1]/right/donor",
    )
    np.testing.assert_array_equal(result.active_nodes[0], left_mask)
    np.testing.assert_array_equal(result.active_nodes[1], left_mask)
    np.testing.assert_array_equal(result.active_nodes[2], right_mask)
    np.testing.assert_array_equal(
        result.total_charge_density_C_m3,
        np.sum(result.charge_density_C_m3, axis=0),
    )
    np.testing.assert_array_equal(
        result.total_recombination_rate_m3_s,
        np.sum(result.recombination_rate_m3_s, axis=0),
    )
    assert np.all(result.charge_density_C_m3[0, left_mask] < 0.0)
    assert np.all(result.charge_density_C_m3[1] == 0.0)
    assert np.all(result.charge_density_C_m3[2, right_mask] > 0.0)
    assert not result.active_nodes.flags.writeable
    assert result.to_dict()["model_identity_sha256"] == model.identity_sha256


def test_device_model_rejects_overlapping_material_regions():
    species = (_species("acceptor", ACCEPTOR),)
    region = MonovalentDefectRegion(
        identifier="layer[0]/left",
        document_sha256=_document_hash(*species),
        active_nodes=np.asarray([True, True, False]),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=species,
    )
    overlapping = replace(
        region,
        identifier="layer[1]/right",
        active_nodes=np.asarray([False, True, True]),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        MonovalentBulkDefectModel(regions=(region, overlapping))


def test_device_region_rejects_document_hash_that_does_not_match_species():
    with pytest.raises(ValueError, match="does not match its species"):
        MonovalentDefectRegion(
            identifier="layer[0]/bad",
            document_sha256="a" * 64,
            active_nodes=np.asarray([True, True]),
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=TEMPERATURE_K,
            species=(_species("acceptor", ACCEPTOR),),
        )


@pytest.mark.parametrize(
    ("sigma_n_m2", "sigma_p_m2"),
    [(0.0, 2.0e-19), (3.0e-19, 0.0)],
)
def test_one_zero_capture_leg_has_finite_occupancy_and_zero_recombination(
    sigma_n_m2,
    sigma_p_m2,
):
    result = _evaluate(
        np.asarray([0.0, 2.0e20]),
        np.asarray([0.0, 4.0e19]),
        _species(
            "one_leg",
            ACCEPTOR,
            sigma_n_m2=sigma_n_m2,
            sigma_p_m2=sigma_p_m2,
        ),
    )

    assert np.all(np.isfinite(result.occupancy))
    assert 0.0 <= result.minimum_occupancy <= result.maximum_occupancy <= 1.0
    np.testing.assert_array_equal(
        result.recombination_rate_m3_s,
        np.zeros_like(result.recombination_rate_m3_s),
    )


def test_density_to_zero_limit_is_linear_for_rate_and_charge():
    base_species = _species("acceptor", ACCEPTOR, density_m3=1.0e22)
    small_species = replace(
        base_species,
        distribution=replace(
            base_species.distribution,
            total_density_m3=1.0e10,
        ),
    )
    base = _evaluate(2.0e20, 7.0e18, base_species)
    small = _evaluate(2.0e20, 7.0e18, small_species)

    assert small.occupancy.item() == base.occupancy.item()
    np.testing.assert_allclose(
        small.recombination_rate_m3_s,
        1.0e-12 * base.recombination_rate_m3_s,
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        small.charge_density_C_m3,
        1.0e-12 * base.charge_density_C_m3,
        rtol=3e-15,
    )


def test_band_edge_and_zero_carrier_limits_remain_finite_and_bounded():
    result = _evaluate(
        np.asarray([0.0, 1.0e24]),
        np.asarray([0.0, 1.0e24]),
        _species("vb_acceptor", ACCEPTOR, center_eV=0.0),
        _species("cb_donor", DONOR, center_eV=GAP_EV),
    )

    for value in (
        result.n1_m3,
        result.p1_m3,
        result.kinetic_denominator_s1,
        result.occupancy,
        result.recombination_rate_m3_s,
        result.charge_density_C_m3,
    ):
        assert np.all(np.isfinite(value))
    assert 0.0 <= result.minimum_occupancy <= result.maximum_occupancy <= 1.0


@pytest.mark.parametrize(
    (
        "temperature_K",
        "center_eV",
        "electron_density_m3",
        "hole_density_m3",
        "sigma_n_m2",
        "sigma_p_m2",
    ),
    [
        (180.0, 0.0, 0.0, 0.0, 1.0e-30, 1.0e-12),
        (300.0, GAP_EV / 2.0, 1.0e-30, 1.0e30, 0.0, 1.0e-19),
        (300.0, GAP_EV / 2.0, 1.0e30, 1.0e-30, 1.0e-19, 0.0),
        (420.0, GAP_EV, 1.0e30, 1.0e30, 1.0e-12, 1.0e-30),
    ],
    ids=["cold-vb", "hole-injection", "electron-injection", "hot-cb"],
)
def test_extreme_local_states_remain_finite_bounded_and_sign_consistent(
    temperature_K,
    center_eV,
    electron_density_m3,
    hole_density_m3,
    sigma_n_m2,
    sigma_p_m2,
):
    species = (
        _species(
            "acceptor",
            ACCEPTOR,
            center_eV=center_eV,
            sigma_n_m2=sigma_n_m2,
            sigma_p_m2=sigma_p_m2,
        ),
        _species(
            "donor",
            DONOR,
            center_eV=center_eV,
            sigma_n_m2=sigma_n_m2,
            sigma_p_m2=sigma_p_m2,
        ),
        _species(
            "neutral",
            NEUTRAL,
            center_eV=center_eV,
            sigma_n_m2=sigma_n_m2,
            sigma_p_m2=sigma_p_m2,
        ),
    )
    result = evaluate_monovalent_defect_closure(
        electron_density_m3,
        hole_density_m3,
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=temperature_K,
    )

    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value))
    assert 0.0 <= result.minimum_occupancy <= result.maximum_occupancy <= 1.0
    assert result.charge_density_C_m3[0].item() <= 0.0
    assert result.charge_density_C_m3[1].item() >= 0.0
    assert result.charge_density_C_m3[2].item() == 0.0


def test_unsupported_or_ambiguous_constitutive_inputs_fail_closed():
    base = _species("acceptor", ACCEPTOR)
    gaussian = replace(
        base,
        distribution=BulkDefectDistribution(
            kind=GAUSSIAN,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=1.0e22,
            center_eV_above_vb=0.7,
            width_eV=0.1,
            width_convention=WIDTH_GAUSSIAN_SIGMA,
        ),
    )
    unresolved = _species("unresolved", UNRESOLVED)

    with pytest.raises(MonovalentDefectClosureCapabilityError, match="single-level"):
        _evaluate(1.0e20, 1.0e20, gaussian)
    with pytest.raises(MonovalentDefectClosureCapabilityError, match="transition"):
        _evaluate(1.0e20, 1.0e20, unresolved)
    with pytest.raises(MonovalentDefectClosureCapabilityError, match="degeneracy=1.0"):
        _evaluate(1.0e20, 1.0e20, replace(base, degeneracy=2.0))
    with pytest.raises(MonovalentDefectClosureCapabilityError, match="unique named"):
        _evaluate(1.0e20, 1.0e20, base, base)
    with pytest.raises(MonovalentDefectClosureCapabilityError, match="both capture"):
        _evaluate(
            1.0e20,
            1.0e20,
            _species("blocked", ACCEPTOR, sigma_n_m2=0.0, sigma_p_m2=0.0),
        )
    with pytest.raises(ValueError, match="carrier densities"):
        _evaluate(-1.0, 1.0e20, base)
    with pytest.raises(ValueError, match="must not be empty"):
        _evaluate(1.0e20, 1.0e20)

"""D7 local stationary master-equation physics and tangent tests."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.models.multivalent_defects import (
    AMPHOTERIC,
    CUSTOM_MULTILEVEL,
    DOUBLE_ACCEPTOR,
    DOUBLE_DONOR,
    EXPLICIT,
    SCAPS_BINOMIAL,
    SINGLE_ACCEPTOR,
    SINGLE_DONOR,
    UNITY,
    MultivalentBulkDefectSpecies,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_defect_closure,
)
from perovskite_sim.physics.multivalent_defect_closure import (
    MULTIVALENT_DEFECT_CLOSURE_VERSION,
    evaluate_multivalent_defect_closure,
)
from perovskite_sim.physics.temperature import thermal_voltage


GAP_EV = 1.50
NC_M3 = 2.4e25
NV_M3 = 1.1e25
TEMPERATURE_K = 300.0
DENSITY_M3 = 3.0e21


def _kinetics(
    sigma_n_m2: float = 2.0e-19,
    sigma_p_m2: float = 7.0e-20,
    velocity_n_m_s: float = 1.3e5,
    velocity_p_m_s: float = 8.0e4,
) -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=sigma_n_m2,
        sigma_p_m2=sigma_p_m2,
        thermal_velocity_n_m_s=velocity_n_m_s,
        thermal_velocity_p_m_s=velocity_p_m_s,
    )


def _multivalent_species(
    family: str = AMPHOTERIC,
    *,
    charges: tuple[int, ...] = (1, 0, -1),
    energies_eV: tuple[float, ...] = (0.55, 0.85),
    kinetics: tuple[BulkDefectKinetics, ...] | None = None,
    degeneracy_convention: str = SCAPS_BINOMIAL,
    degeneracies: tuple[float, ...] = (1.0, 2.0, 1.0),
    name: str = "multivalent",
) -> MultivalentBulkDefectSpecies:
    transition_kinetics = kinetics or tuple(
        _kinetics() for _ in range(len(charges) - 1)
    )
    correlations = tuple(
        right - left for left, right in zip(energies_eV, energies_eV[1:])
    )
    return MultivalentBulkDefectSpecies(
        name=name,
        total_density_m3=DENSITY_M3,
        configuration=MultivalentDefectConfiguration(
            family=family,
            charge_states_e=charges,
            degeneracy_convention=degeneracy_convention,
            state_degeneracies=degeneracies,
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=energies_eV[0],
                correlation_energies_eV=correlations,
            ),
            transition_kinetics=transition_kinetics,
        ),
    )


def _single_pair(transition: str):
    if transition == ACCEPTOR:
        family = SINGLE_ACCEPTOR
        charges = (0, -1)
        neutral_reference = NEUTRAL_WHEN_EMPTY
    else:
        family = SINGLE_DONOR
        charges = (1, 0)
        neutral_reference = NEUTRAL_WHEN_FILLED
    kinetics = _kinetics()
    multivalent = _multivalent_species(
        family,
        charges=charges,
        energies_eV=(0.62,),
        kinetics=(kinetics,),
        degeneracy_convention=UNITY,
        degeneracies=(1.0, 1.0),
        name=f"single_{transition}",
    )
    monovalent = BulkDefectSpecies(
        name=f"single_{transition}",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=DENSITY_M3,
            center_eV_above_vb=0.62,
        ),
        charge_transition=transition,
        neutral_reference=neutral_reference,
        kinetics=kinetics,
        degeneracy=1.0,
    )
    return multivalent, monovalent


def _evaluate(n, p, species=None, *, nc=NC_M3, nv=NV_M3, gap=GAP_EV):
    return evaluate_multivalent_defect_closure(
        n,
        p,
        species or _multivalent_species(),
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        temperature_K=TEMPERATURE_K,
    )


def test_probabilities_are_normalized_nonnegative_and_solve_master_equation():
    n = np.geomspace(1.0e14, 1.0e23, 19)
    p = np.geomspace(1.0e22, 1.0e15, 19)
    result = _evaluate(n, p)

    assert result.to_dict()["closure"] == MULTIVALENT_DEFECT_CLOSURE_VERSION
    assert result.minimum_state_probability >= 0.0
    assert result.maximum_state_probability <= 1.0
    assert result.maximum_probability_sum_error <= 3.0e-16
    np.testing.assert_allclose(
        np.sum(result.state_probability, axis=0),
        np.ones(n.shape),
        rtol=0.0,
        atol=3.0e-16,
    )
    scale = np.max(result.forward_state_rate_s1 + result.backward_state_rate_s1)
    assert result.maximum_master_residual_s1 <= 2.0e-15 * scale


def test_equilibrium_grand_partition_ratios_and_zero_recombination():
    species = _multivalent_species()
    thermal = thermal_voltage(TEMPERATURE_K)
    intrinsic_product = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal)
    n = 2.0e18
    p = intrinsic_product / n
    result = _evaluate(n, p, species)
    probabilities = result.state_probability
    energies = result.transition_energies_eV_above_vb
    degeneracies = np.asarray(result.state_degeneracies)
    n1 = NC_M3 * np.exp(-(GAP_EV - energies) / thermal)
    expected_ratio = degeneracies[1:] / degeneracies[:-1] * n / n1

    np.testing.assert_allclose(
        probabilities[1:] / probabilities[:-1],
        expected_ratio,
        rtol=3.0e-14,
    )
    capture_scale = DENSITY_M3 * max(
        np.max(result.capture_n_m3_s) * n,
        np.max(result.capture_p_m3_s) * p,
    )
    assert abs(result.total_recombination_rate_m3_s.item()) <= (2.0e-15 * capture_scale)


@pytest.mark.parametrize("transition", (ACCEPTOR, DONOR))
def test_single_transition_unity_limit_recovers_d2_exact_local_closure(transition):
    multivalent, monovalent = _single_pair(transition)
    n = np.asarray([2.0e17, 8.0e19, 4.0e21])
    p = np.asarray([3.0e21, 5.0e19, 9.0e17])
    actual = _evaluate(n, p, multivalent)
    expected = evaluate_monovalent_defect_closure(
        n,
        p,
        (monovalent,),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    np.testing.assert_allclose(
        actual.state_probability[1], expected.occupancy[0], rtol=3.0e-12
    )
    np.testing.assert_allclose(
        actual.charge_density_C_m3,
        expected.charge_density_C_m3[0],
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        actual.total_recombination_rate_m3_s,
        expected.recombination_rate_m3_s[0],
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        actual.total_recombination_derivative_n_s1,
        expected.recombination_derivative_n_s1[0],
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        actual.total_recombination_derivative_p_s1,
        expected.recombination_derivative_p_s1[0],
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        actual.charge_derivative_n_C,
        expected.charge_derivative_n_C[0],
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        actual.charge_derivative_p_C,
        expected.charge_derivative_p_C[0],
        rtol=3.0e-12,
    )


def test_analytic_density_tangent_matches_centered_difference_and_ift():
    species = _multivalent_species()
    n = 3.0e19
    p = 8.0e18
    result = _evaluate(n, p, species)
    assert result.closure_identity_sha256 == (
        "13194ab97a4ecb4a217f13a53851bf4d47033d1717fa6c086fe651a3fcb9d37c"
    )
    relative_step = 2.0e-5
    n_step = n * relative_step
    p_step = p * relative_step
    plus_n = _evaluate(n + n_step, p, species)
    minus_n = _evaluate(n - n_step, p, species)
    plus_p = _evaluate(n, p + p_step, species)
    minus_p = _evaluate(n, p - p_step, species)
    fd_probability_n = (plus_n.state_probability - minus_n.state_probability) / (
        2.0 * n_step
    )
    fd_probability_p = (plus_p.state_probability - minus_p.state_probability) / (
        2.0 * p_step
    )

    np.testing.assert_allclose(
        result.state_probability_derivative_n_m3,
        fd_probability_n,
        rtol=2.0e-8,
        atol=1.0e-34,
    )
    np.testing.assert_allclose(
        result.state_probability_derivative_p_m3,
        fd_probability_p,
        rtol=2.0e-8,
        atol=1.0e-34,
    )
    np.testing.assert_allclose(
        result.charge_derivative_n_C,
        (plus_n.charge_density_C_m3 - minus_n.charge_density_C_m3) / (2.0 * n_step),
        rtol=2.0e-8,
    )
    np.testing.assert_allclose(
        result.total_recombination_derivative_p_s1,
        (plus_p.total_recombination_rate_m3_s - minus_p.total_recombination_rate_m3_s)
        / (2.0 * p_step),
        rtol=3.0e-8,
    )

    ift_n = np.einsum(
        "ij...,j...->i...",
        result.master_matrix_s1,
        result.state_probability_derivative_n_m3,
    ) + np.einsum(
        "ij...,j...->i...",
        result.master_matrix_derivative_n_m3_s1,
        result.state_probability,
    )
    ift_p = np.einsum(
        "ij...,j...->i...",
        result.master_matrix_s1,
        result.state_probability_derivative_p_m3,
    ) + np.einsum(
        "ij...,j...->i...",
        result.master_matrix_derivative_p_m3_s1,
        result.state_probability,
    )
    assert np.max(np.abs(ift_n)) <= 3.0e-15 * np.max(result.capture_n_m3_s)
    assert np.max(np.abs(ift_p)) <= 3.0e-15 * np.max(result.capture_p_m3_s)
    assert abs(np.sum(result.state_probability_derivative_n_m3)) <= 1.0e-34
    assert abs(np.sum(result.state_probability_derivative_p_m3)) <= 1.0e-34


def test_electron_hole_mirror_reverses_state_and_charge_but_not_rate():
    original_kinetics = (
        _kinetics(3.0e-19, 5.0e-20, 1.1e5, 7.0e4),
        _kinetics(8.0e-20, 4.0e-19, 9.0e4, 1.4e5),
    )
    original = _multivalent_species(
        DOUBLE_DONOR,
        charges=(2, 1, 0),
        energies_eV=(0.35, 0.92),
        kinetics=original_kinetics,
        name="double_donor",
    )
    mirror_kinetics = tuple(
        _kinetics(
            value.sigma_p_m2,
            value.sigma_n_m2,
            value.thermal_velocity_p_m_s,
            value.thermal_velocity_n_m_s,
        )
        for value in original_kinetics[::-1]
    )
    mirror = _multivalent_species(
        DOUBLE_ACCEPTOR,
        charges=(0, -1, -2),
        energies_eV=(GAP_EV - 0.92, GAP_EV - 0.35),
        kinetics=mirror_kinetics,
        name="double_acceptor",
    )
    n = np.asarray([3.0e17, 4.0e20])
    p = np.asarray([2.0e21, 7.0e18])
    result = _evaluate(n, p, original)
    mirrored = _evaluate(p, n, mirror, nc=NV_M3, nv=NC_M3)

    np.testing.assert_allclose(
        mirrored.state_probability,
        result.state_probability[::-1],
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        mirrored.charge_density_C_m3,
        -result.charge_density_C_m3,
        rtol=3.0e-14,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        mirrored.total_recombination_rate_m3_s,
        result.total_recombination_rate_m3_s,
        rtol=4.0e-14,
    )


def test_custom_five_state_closure_shares_one_density_across_all_states():
    species = _multivalent_species(
        CUSTOM_MULTILEVEL,
        charges=(2, 1, 0, -1, -2),
        energies_eV=(0.25, 0.55, 0.80, 1.10),
        degeneracy_convention=EXPLICIT,
        degeneracies=(1.0, 3.0, 4.0, 3.0, 1.0),
    )
    result = _evaluate(7.0e19, 4.0e18, species)

    assert result.state_probability.shape == (5,)
    assert np.sum(result.state_probability).item() == pytest.approx(1.0)
    assert (
        np.sum(result.state_probability) * result.total_density_m3
    ).item() == pytest.approx(DENSITY_M3)
    assert result.transition_recombination_rate_m3_s.shape == (4,)


def test_two_independent_srh_centres_cannot_masquerade_as_one_amphoteric_defect():
    velocity = 1.0e5
    donor_kinetics = _kinetics(
        sigma_n_m2=1.0e-16 / velocity,
        sigma_p_m2=1.0e-12 / velocity,
        velocity_n_m_s=velocity,
        velocity_p_m_s=velocity,
    )
    acceptor_kinetics = _kinetics(
        sigma_n_m2=1.0e-12 / velocity,
        sigma_p_m2=1.0e-16 / velocity,
        velocity_n_m_s=velocity,
        velocity_p_m_s=velocity,
    )
    amphoteric = _multivalent_species(
        AMPHOTERIC,
        charges=(1, 0, -1),
        energies_eV=(0.45, 0.65),
        kinetics=(donor_kinetics, acceptor_kinetics),
        name="one_amphoteric_defect",
    )
    multivalent = _evaluate(
        8.0e20,
        2.0e17,
        amphoteric,
        gap=1.10,
    )
    independent = evaluate_monovalent_defect_closure(
        8.0e20,
        2.0e17,
        (
            BulkDefectSpecies(
                name="independent_donor",
                distribution=BulkDefectDistribution(
                    kind=SINGLE_LEVEL,
                    normalization=INTEGRATED_TOTAL,
                    total_density_m3=DENSITY_M3,
                    center_eV_above_vb=0.45,
                ),
                charge_transition=DONOR,
                neutral_reference=NEUTRAL_WHEN_FILLED,
                kinetics=donor_kinetics,
                degeneracy=1.0,
            ),
            BulkDefectSpecies(
                name="independent_acceptor",
                distribution=BulkDefectDistribution(
                    kind=SINGLE_LEVEL,
                    normalization=INTEGRATED_TOTAL,
                    total_density_m3=DENSITY_M3,
                    center_eV_above_vb=0.65,
                ),
                charge_transition=ACCEPTOR,
                neutral_reference=NEUTRAL_WHEN_EMPTY,
                kinetics=acceptor_kinetics,
                degeneracy=1.0,
            ),
        ),
        band_gap_eV=1.10,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    assert not math.isclose(
        multivalent.total_recombination_rate_m3_s.item(),
        independent.total_recombination_rate_m3_s.item(),
        rel_tol=1.0e-3,
    )
    assert not math.isclose(
        multivalent.charge_density_C_m3.item(),
        independent.total_charge_density_C_m3.item(),
        rel_tol=1.0e-3,
    )


@pytest.mark.parametrize(
    "kinetics",
    (
        _kinetics(sigma_n_m2=0.0, sigma_p_m2=7.0e-20),
        _kinetics(sigma_n_m2=2.0e-19, sigma_p_m2=0.0),
    ),
)
def test_one_missing_capture_leg_has_finite_state_and_zero_recombination(kinetics):
    species = _multivalent_species(
        SINGLE_ACCEPTOR,
        charges=(0, -1),
        energies_eV=(0.62,),
        kinetics=(kinetics,),
        degeneracy_convention=UNITY,
        degeneracies=(1.0, 1.0),
    )
    result = _evaluate(4.0e19, 7.0e18, species)

    assert np.all(np.isfinite(result.state_probability))
    assert np.sum(result.state_probability).item() == pytest.approx(1.0)
    assert result.total_recombination_rate_m3_s.item() == 0.0


def test_correlation_energy_changes_identity_and_physical_state_distribution():
    species = _multivalent_species()
    changed = replace(
        species,
        configuration=replace(
            species.configuration,
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=0.55,
                correlation_energies_eV=(0.35,),
            ),
        ),
    )
    baseline = _evaluate(5.0e19, 2.0e19, species)
    perturbed = _evaluate(5.0e19, 2.0e19, changed)

    assert baseline.closure_identity_sha256 != perturbed.closure_identity_sha256
    assert not np.array_equal(
        baseline.state_probability,
        perturbed.state_probability,
    )


def test_charge_state_order_has_correct_electron_and_hole_injection_limits():
    species = _multivalent_species()
    electron_rich = _evaluate(1.0e25, 1.0e8, species)
    hole_rich = _evaluate(1.0e8, 1.0e25, species)

    assert electron_rich.state_probability[-1] > 0.999
    assert electron_rich.charge_number_density_m3 < 0.0
    assert hole_rich.state_probability[0] > 0.999
    assert hole_rich.charge_number_density_m3 > 0.0
    assert electron_rich.charge_density_C_m3 == pytest.approx(
        Q * electron_rich.charge_number_density_m3
    )


def test_nonpositive_carriers_fail_closed_and_results_are_immutable():
    with pytest.raises(ValueError, match="finite and positive"):
        _evaluate(0.0, 1.0e20)
    with pytest.raises(ValueError, match="finite and positive"):
        _evaluate(1.0e20, -1.0)

    result = _evaluate(1.0e20, 1.0e19)
    with pytest.raises(ValueError, match="read-only"):
        result.state_probability[0] = 0.5

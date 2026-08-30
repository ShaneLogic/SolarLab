"""D7-E1 device aggregation for stationary multivalent bulk defects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import BulkDefectKinetics
from perovskite_sim.models.multivalent_defects import (
    MultivalentBulkDefectSpecies,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.physics.multivalent_defect_closure import (
    evaluate_multivalent_defect_closure,
)
from perovskite_sim.physics.multivalent_defect_device import (
    MultivalentBulkDefectModel,
    MultivalentDefectRegion,
    evaluate_multivalent_bulk_defects,
    evaluate_multivalent_source_defect_closure,
    solve_multivalent_defect_charge_neutrality,
)
from perovskite_sim.physics.temperature import thermal_voltage


TEMPERATURE_K = 300.0
GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23


def _kinetics(scale: float = 1.0) -> BulkDefectKinetics:
    return BulkDefectKinetics(
        sigma_n_m2=2.0e-19 * scale,
        sigma_p_m2=7.0e-20 * scale,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )


def _species(
    name: str = "double_donor_bulk",
    *,
    family: str = "double_donor",
    density_m3: float = 2.0e21,
    degeneracy_convention: str = "unity",
) -> MultivalentBulkDefectSpecies:
    charges = {
        "double_donor": (2, 1, 0),
        "double_acceptor": (0, -1, -2),
        "amphoteric": (1, 0, -1),
    }[family]
    degeneracies = (
        tuple(float(math.comb(len(charges) - 1, i)) for i in range(len(charges)))
        if degeneracy_convention == "scaps_binomial"
        else (1.0,) * len(charges)
    )
    return MultivalentBulkDefectSpecies(
        name=name,
        total_density_m3=density_m3,
        configuration=MultivalentDefectConfiguration(
            family=family,
            charge_states_e=charges,
            degeneracy_convention=degeneracy_convention,
            state_degeneracies=degeneracies,
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=0.30,
                correlation_energies_eV=(0.15,),
            ),
            transition_kinetics=(_kinetics(), _kinetics(0.5)),
        ),
    )


def _region(
    mask: np.ndarray,
    species: tuple[MultivalentBulkDefectSpecies, ...],
    *,
    identifier: str = "layer[0]/defective",
) -> MultivalentDefectRegion:
    return MultivalentDefectRegion(
        identifier=identifier,
        document_sha256="0" * 64,
        active_nodes=mask,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
        species=species,
    )


def _carriers(size: int) -> tuple[np.ndarray, np.ndarray]:
    intrinsic_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    n = np.geomspace(1.0e12, 1.0e20, size)
    return n, intrinsic_sq / n


def test_source_aggregate_is_the_sum_of_independent_species_closures():
    first = _species("first")
    second = _species("second", family="amphoteric", density_m3=5.0e20)
    n, p = _carriers(7)
    aggregate = evaluate_multivalent_source_defect_closure(
        n,
        p,
        (first, second),
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    independent = tuple(
        evaluate_multivalent_defect_closure(
            n,
            p,
            item,
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=TEMPERATURE_K,
        )
        for item in (first, second)
    )

    assert aggregate.species_names == ("first", "second")
    np.testing.assert_array_equal(
        aggregate.total_charge_density_C_m3,
        independent[0].charge_density_C_m3 + independent[1].charge_density_C_m3,
    )
    np.testing.assert_array_equal(
        aggregate.total_recombination_rate_m3_s,
        independent[0].total_recombination_rate_m3_s
        + independent[1].total_recombination_rate_m3_s,
    )
    np.testing.assert_array_equal(
        aggregate.total_charge_derivative_fixed_qf_C_m3_V,
        independent[0].charge_derivative_fixed_qf_C_m3_V
        + independent[1].charge_derivative_fixed_qf_C_m3_V,
    )
    assert not aggregate.total_charge_density_C_m3.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        evaluate_multivalent_source_defect_closure(
            n,
            p,
            (first, first),
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            temperature_K=TEMPERATURE_K,
        )


def test_model_masks_must_be_disjoint_on_one_shared_grid():
    left = np.zeros(9, dtype=bool)
    left[:4] = True
    right = np.zeros(9, dtype=bool)
    right[4:] = True
    model = MultivalentBulkDefectModel(
        regions=(
            _region(left, (_species("a"),), identifier="layer[0]/left"),
            _region(right, (_species("b"),), identifier="layer[1]/right"),
        )
    )
    assert model.node_count == 9
    assert model.species_identifiers == (
        "layer[0]/left/a",
        "layer[1]/right/b",
    )
    np.testing.assert_array_equal(model.explicit_node_mask, left | right)

    overlapping = np.zeros(9, dtype=bool)
    overlapping[3:] = True
    with pytest.raises(ValueError, match="disjoint"):
        MultivalentBulkDefectModel(
            regions=(
                _region(left, (_species("a"),), identifier="layer[0]/left"),
                _region(
                    overlapping,
                    (_species("b"),),
                    identifier="layer[1]/right",
                ),
            )
        )
    with pytest.raises(ValueError, match="share one device grid"):
        MultivalentBulkDefectModel(
            regions=(
                _region(left, (_species("a"),), identifier="layer[0]/left"),
                _region(
                    np.ones(5, dtype=bool),
                    (_species("b"),),
                    identifier="layer[1]/right",
                ),
            )
        )


def test_full_grid_evaluation_matches_the_region_source_closure():
    mask = np.zeros(8, dtype=bool)
    mask[2:6] = True
    species = (_species("a"), _species("b", family="double_acceptor"))
    model = MultivalentBulkDefectModel(regions=(_region(mask, species),))
    n, p = _carriers(8)
    evaluation = evaluate_multivalent_bulk_defects(n, p, model)
    local = evaluate_multivalent_source_defect_closure(
        n[mask],
        p[mask],
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )

    assert evaluation.model_identity_sha256 == model.identity_sha256
    assert evaluation.state_counts == (3, 3)
    np.testing.assert_array_equal(
        evaluation.total_charge_density_C_m3[mask],
        local.total_charge_density_C_m3,
    )
    np.testing.assert_array_equal(
        evaluation.total_recombination_rate_m3_s[mask],
        local.total_recombination_rate_m3_s,
    )
    np.testing.assert_array_equal(
        evaluation.total_charge_density_C_m3[~mask],
        np.zeros(int(np.sum(~mask))),
    )
    for probabilities in evaluation.state_probability:
        np.testing.assert_allclose(
            np.sum(probabilities[:, mask], axis=0),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        # Probabilities are a normalized distribution, so off-region they are
        # undefined rather than zero; zero columns would sum to 0, not 1.
        assert np.all(np.isnan(probabilities[:, ~mask]))
    with pytest.raises(ValueError, match="match the compiled grid"):
        evaluate_multivalent_bulk_defects(n[:5], p[:5], model)
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_multivalent_bulk_defects(np.zeros_like(n), p, model)
    with pytest.raises(TypeError, match="MultivalentBulkDefectModel"):
        evaluate_multivalent_bulk_defects(n, p, object())


def test_multi_region_evaluation_matches_independent_single_region_models():
    """The per-region row offset must not leak species across regions."""
    left = np.zeros(10, dtype=bool)
    left[1:4] = True
    right = np.zeros(10, dtype=bool)
    right[6:9] = True
    left_species = (_species("a"), _species("b", family="amphoteric"))
    right_species = (_species("c", family="double_acceptor"),)
    combined = MultivalentBulkDefectModel(
        regions=(
            _region(left, left_species, identifier="layer[0]/left"),
            _region(right, right_species, identifier="layer[1]/right"),
        )
    )
    n, p = _carriers(10)
    evaluation = evaluate_multivalent_bulk_defects(n, p, combined)

    assert evaluation.species_identifiers == (
        "layer[0]/left/a",
        "layer[0]/left/b",
        "layer[1]/right/c",
    )
    assert evaluation.state_counts == (3, 3, 3)
    # Row `k` must be active exactly on the nodes of the region owning it.
    np.testing.assert_array_equal(evaluation.active_nodes[0], left)
    np.testing.assert_array_equal(evaluation.active_nodes[1], left)
    np.testing.assert_array_equal(evaluation.active_nodes[2], right)

    for mask, species, rows in (
        (left, left_species, slice(0, 2)),
        (right, right_species, slice(2, 3)),
    ):
        isolated = MultivalentBulkDefectModel(
            regions=(_region(mask, species, identifier="layer[0]/isolated"),)
        )
        alone = evaluate_multivalent_bulk_defects(n, p, isolated)
        np.testing.assert_array_equal(
            evaluation.total_charge_density_C_m3[mask],
            alone.total_charge_density_C_m3[mask],
        )
        np.testing.assert_array_equal(
            evaluation.total_recombination_rate_m3_s[mask],
            alone.total_recombination_rate_m3_s[mask],
        )
        np.testing.assert_array_equal(
            evaluation.charge_density_C_m3[rows][:, mask],
            alone.charge_density_C_m3[:, mask],
        )

    unowned = ~(left | right)
    np.testing.assert_array_equal(
        evaluation.total_charge_density_C_m3[unowned],
        np.zeros(int(np.sum(unowned))),
    )


def test_state_degeneracy_convention_reaches_the_device_aggregate():
    """SCAPS binomial degeneracies must change the compiled device answer."""
    mask = np.ones(5, dtype=bool)
    unity = MultivalentBulkDefectModel(
        regions=(_region(mask, (_species("a", family="amphoteric"),)),)
    )
    binomial = MultivalentBulkDefectModel(
        regions=(
            _region(
                mask,
                (
                    _species(
                        "a",
                        family="amphoteric",
                        degeneracy_convention="scaps_binomial",
                    ),
                ),
            ),
        )
    )
    n, p = _carriers(5)
    unity_evaluation = evaluate_multivalent_bulk_defects(n, p, unity)
    binomial_evaluation = evaluate_multivalent_bulk_defects(n, p, binomial)

    assert unity.identity_sha256 != binomial.identity_sha256
    assert unity.state_counts == binomial.state_counts
    assert (
        np.max(
            np.abs(
                binomial_evaluation.total_charge_density_C_m3
                - unity_evaluation.total_charge_density_C_m3
            )
        )
        > 0.0
    )

    unity_root = solve_multivalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=0.0,
        donor_density_m3=0.0,
        species=(_species("a", family="amphoteric"),),
    ).neutrality
    binomial_root = solve_multivalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=0.0,
        donor_density_m3=0.0,
        species=(
            _species(
                "a",
                family="amphoteric",
                degeneracy_convention="scaps_binomial",
            ),
        ),
    ).neutrality
    assert unity_root.normalized_charge_residual <= 1.0e-12
    assert binomial_root.normalized_charge_residual <= 1.0e-12
    assert not np.isclose(
        unity_root.electron_density_m3,
        binomial_root.electron_density_m3,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_model_identity_binds_region_metadata():
    mask = np.ones(4, dtype=bool)
    base = MultivalentBulkDefectModel(regions=(_region(mask, (_species(),)),))
    denser = MultivalentBulkDefectModel(
        regions=(_region(mask, (_species(density_m3=4.0e21),)),)
    )
    assert base.identity_sha256 != denser.identity_sha256


def test_contact_neutrality_root_uses_the_same_master_equation_closure():
    species = (_species(),)
    result = solve_multivalent_defect_charge_neutrality(
        temperature_K=TEMPERATURE_K,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        acceptor_density_m3=3.0e21,
        donor_density_m3=0.0,
        species=species,
    )
    neutrality = result.neutrality
    assert neutrality.normalized_charge_residual <= 1.0e-12

    closure = evaluate_multivalent_source_defect_closure(
        neutrality.electron_density_m3,
        neutrality.hole_density_m3,
        species,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=TEMPERATURE_K,
    )
    residual = (
        neutrality.hole_density_m3
        - neutrality.electron_density_m3
        - 3.0e21
        + float(np.asarray(closure.total_charge_density_C_m3)) / Q
    )
    scale = max(
        neutrality.electron_density_m3,
        neutrality.hole_density_m3,
        3.0e21,
        species[0].total_density_m3,
    )
    assert abs(residual) / scale <= 1.0e-12
    # A donor-family defect must push the acceptor-doped root toward
    # compensation: fewer holes than the defect-free acceptor solution.
    intrinsic_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    defect_free_holes = 0.5 * (3.0e21 + math.sqrt(3.0e21**2 + 4.0 * intrinsic_sq))
    assert neutrality.hole_density_m3 < defect_free_holes


def test_mixed_derivative_dispatch_partitions_nodes_and_matches_finite_difference():
    """The multivalent branch of the derivative API is not yet on a solver path.

    D7-E2 (AC / dynamic occupancy) will be the first consumer, so the
    partition order and the analytic tangents are pinned here rather than
    left to execute for the first time inside a frequency-domain solve.
    """
    from perovskite_sim.physics.recombination import (
        srh_recombination,
        srh_recombination_derivatives,
    )

    node_count = 9
    mask = np.zeros(node_count, dtype=bool)
    mask[3:7] = True
    model = MultivalentBulkDefectModel(
        regions=(_region(mask, (_species("a"), _species("b", family="amphoteric"))),)
    )
    n, p = _carriers(node_count)
    ni_sq = NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    lifetime = dict(
        ni_sq=np.full(node_count, ni_sq),
        tau_n=np.full(node_count, 1.0e-6),
        tau_p=np.full(node_count, 2.0e-6),
        n1=np.full(node_count, math.sqrt(ni_sq)),
        p1=np.full(node_count, math.sqrt(ni_sq)),
    )

    rate = srh_recombination(
        n,
        p,
        lifetime["ni_sq"],
        lifetime["tau_n"],
        lifetime["tau_p"],
        lifetime["n1"],
        lifetime["p1"],
        multivalent_bulk_defects=model,
    )
    derivatives = srh_recombination_derivatives(
        n,
        p,
        lifetime["ni_sq"],
        lifetime["tau_n"],
        lifetime["tau_p"],
        lifetime["n1"],
        lifetime["p1"],
        multivalent_bulk_defects=model,
    )

    # The two dispatches must agree on the rate itself.
    np.testing.assert_allclose(derivatives.rate, rate, rtol=0.0, atol=0.0)
    # Explicit nodes carry the master-equation rate; the rest keep the
    # effective-lifetime law.
    explicit = evaluate_multivalent_bulk_defects(n, p, model)
    np.testing.assert_array_equal(
        rate[mask],
        explicit.total_recombination_rate_m3_s[mask],
    )
    legacy = (n[~mask] * p[~mask] - ni_sq) / (
        lifetime["tau_p"][~mask] * (n[~mask] + lifetime["n1"][~mask])
        + lifetime["tau_n"][~mask] * (p[~mask] + lifetime["p1"][~mask])
    )
    np.testing.assert_allclose(rate[~mask], legacy, rtol=1.0e-12, atol=0.0)

    def rate_at(n_values: np.ndarray, p_values: np.ndarray) -> np.ndarray:
        return srh_recombination(
            n_values,
            p_values,
            lifetime["ni_sq"],
            lifetime["tau_n"],
            lifetime["tau_p"],
            lifetime["n1"],
            lifetime["p1"],
            multivalent_bulk_defects=model,
        )

    step = 1.0e-6
    finite_n = (rate_at(n * (1.0 + step), p) - rate_at(n * (1.0 - step), p)) / (
        2.0 * step * n
    )
    finite_p = (rate_at(n, p * (1.0 + step)) - rate_at(n, p * (1.0 - step))) / (
        2.0 * step * p
    )
    np.testing.assert_allclose(
        derivatives.electron_density_derivative,
        finite_n,
        rtol=2.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        derivatives.hole_density_derivative,
        finite_p,
        rtol=2.0e-6,
        atol=0.0,
    )


def test_contact_neutrality_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="finite and non-negative"):
        solve_multivalent_defect_charge_neutrality(
            temperature_K=TEMPERATURE_K,
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            acceptor_density_m3=-1.0,
            donor_density_m3=0.0,
            species=(_species(),),
        )
    with pytest.raises(TypeError, match="multivalent defect species"):
        solve_multivalent_defect_charge_neutrality(
            temperature_K=TEMPERATURE_K,
            band_gap_eV=GAP_EV,
            effective_conduction_dos_m3=NC_M3,
            effective_valence_dos_m3=NV_M3,
            acceptor_density_m3=0.0,
            donor_density_m3=0.0,
            species=(object(),),
        )

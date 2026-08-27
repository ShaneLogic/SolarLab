"""DEF-1 exact neutral multi-species SRH constitutive tests."""

from __future__ import annotations

import numpy as np

from perovskite_sim.physics.recombination import (
    CompiledNeutralDefectSpecies,
    NeutralBulkDefectModel,
    evaluate_neutral_bulk_defects,
    srh_recombination,
    srh_recombination_derivatives,
)


_DOCUMENT_HASH = "a" * 64


def _compiled_species(
    identifier: str,
    *,
    active: np.ndarray,
    tau_n_s: float,
    tau_p_s: float,
    n1_m3: float,
    p1_m3: float,
) -> CompiledNeutralDefectSpecies:
    return CompiledNeutralDefectSpecies(
        identifier=identifier,
        document_sha256=_DOCUMENT_HASH,
        active_nodes=active,
        tau_n_s=tau_n_s,
        tau_p_s=tau_p_s,
        n1_m3=np.where(active, n1_m3, 0.0),
        p1_m3=np.where(active, p1_m3, 0.0),
    )


def _model(
    *species: CompiledNeutralDefectSpecies,
    explicit_mask: np.ndarray | None = None,
) -> NeutralBulkDefectModel:
    mask = (
        np.logical_or.reduce([item.active_nodes for item in species])
        if explicit_mask is None
        else explicit_mask
    )
    return NeutralBulkDefectModel(
        species=species,
        explicit_node_mask=mask,
        layer_document_sha256=(("absorber", _DOCUMENT_HASH),),
    )


def test_single_neutral_species_is_bitwise_legacy_srh_and_derivatives():
    n = np.asarray([2.0e16, 4.0e19, 8.0e21])
    p = np.asarray([3.0e21, 6.0e19, 1.0e17])
    ni_sq = np.asarray([1.0e28, 1.0e28, 1.0e28])
    active = np.ones(n.size, dtype=bool)
    tau_n = 2.5e-7
    tau_p = 7.0e-8
    n1 = 4.0e14
    p1 = 2.5e13
    model = _model(
        _compiled_species(
            "absorber/neutral",
            active=active,
            tau_n_s=tau_n,
            tau_p_s=tau_p,
            n1_m3=n1,
            p1_m3=p1,
        )
    )

    legacy = srh_recombination_derivatives(
        n, p, ni_sq, tau_n, tau_p, n1, p1
    )
    explicit = srh_recombination_derivatives(
        n,
        p,
        ni_sq,
        99.0,
        88.0,
        77.0,
        66.0,
        neutral_bulk_defects=model,
    )

    np.testing.assert_array_equal(explicit.rate, legacy.rate)
    np.testing.assert_array_equal(
        explicit.electron_density_derivative,
        legacy.electron_density_derivative,
    )
    np.testing.assert_array_equal(
        explicit.hole_density_derivative,
        legacy.hole_density_derivative,
    )


def test_two_identical_species_add_as_parallel_exact_srh_channels():
    n = np.asarray([1.0e19, 5.0e20, 2.0e22])
    p = np.asarray([3.0e22, 8.0e20, 2.0e19])
    ni_sq = np.full(n.shape, 1.0e28)
    active = np.ones(n.size, dtype=bool)
    first = _compiled_species(
        "absorber/a",
        active=active,
        tau_n_s=3.0e-7,
        tau_p_s=9.0e-8,
        n1_m3=1.0e15,
        p1_m3=1.0e13,
    )
    second = _compiled_species(
        "absorber/b",
        active=active,
        tau_n_s=3.0e-7,
        tau_p_s=9.0e-8,
        n1_m3=1.0e15,
        p1_m3=1.0e13,
    )

    one = evaluate_neutral_bulk_defects(n, p, ni_sq, _model(first))
    two = evaluate_neutral_bulk_defects(n, p, ni_sq, _model(first, second))

    np.testing.assert_array_equal(two.total.rate, 2.0 * one.total.rate)
    np.testing.assert_array_equal(
        two.total.electron_density_derivative,
        2.0 * one.total.electron_density_derivative,
    )
    np.testing.assert_array_equal(
        two.total.hole_density_derivative,
        2.0 * one.total.hole_density_derivative,
    )


def test_separated_levels_do_not_collapse_to_legacy_weighted_reference():
    n = np.asarray([1.8e14, 3.0e16, 8.0e20])
    p = np.asarray([1.8e14, 7.0e18, 4.0e16])
    ni_sq = np.full(n.shape, 1.0e28)
    active = np.ones(n.size, dtype=bool)
    low = _compiled_species(
        "absorber/low",
        active=active,
        tau_n_s=1.0e-8,
        tau_p_s=1.0e-4,
        n1_m3=1.0e8,
        p1_m3=1.0e20,
    )
    high = _compiled_species(
        "absorber/high",
        active=active,
        tau_n_s=1.0e-4,
        tau_p_s=1.0e-8,
        n1_m3=1.0e20,
        p1_m3=1.0e8,
    )
    exact = evaluate_neutral_bulk_defects(
        n, p, ni_sq, _model(low, high)
    ).total.rate
    reduced = srh_recombination(
        n,
        p,
        ni_sq,
        9.999000099990002e-9,
        9.999000099990002e-9,
        9.999000199980002e15,
        9.999000199980002e15,
    )

    relative = np.abs(exact - reduced) / np.maximum(np.abs(exact), 1.0)
    assert float(np.max(relative)) > 0.1


def test_zero_capture_leg_blocks_cycle_without_nonfinite_rate():
    n = np.asarray([1.0e17, 1.0e20])
    p = np.asarray([1.0e21, 1.0e18])
    active = np.ones(n.size, dtype=bool)
    blocked = _compiled_species(
        "absorber/blocked",
        active=active,
        tau_n_s=float("inf"),
        tau_p_s=1.0e-7,
        n1_m3=1.0e14,
        p1_m3=1.0e14,
    )

    result = evaluate_neutral_bulk_defects(
        n, p, np.full(n.shape, 1.0e28), _model(blocked)
    )

    np.testing.assert_array_equal(result.total.rate, np.zeros_like(n))
    np.testing.assert_array_equal(
        result.total.electron_density_derivative, np.zeros_like(n)
    )
    np.testing.assert_array_equal(
        result.total.hole_density_derivative, np.zeros_like(n)
    )
    assert result.minimum_denominator_s_m3 == (None,)


def test_density_to_zero_limit_vanishes_linearly():
    n = np.asarray([2.0e18, 7.0e20])
    p = np.asarray([5.0e21, 9.0e18])
    ni_sq = np.full(n.shape, 1.0e28)
    active = np.ones(n.size, dtype=bool)
    finite_density = _compiled_species(
        "absorber/base",
        active=active,
        tau_n_s=2.0e-7,
        tau_p_s=8.0e-8,
        n1_m3=1.0e14,
        p1_m3=1.0e14,
    )
    vanishing_density = _compiled_species(
        "absorber/vanishing",
        active=active,
        tau_n_s=2.0e5,
        tau_p_s=8.0e4,
        n1_m3=1.0e14,
        p1_m3=1.0e14,
    )

    base = evaluate_neutral_bulk_defects(
        n, p, ni_sq, _model(finite_density)
    ).total
    small = evaluate_neutral_bulk_defects(
        n, p, ni_sq, _model(vanishing_density)
    ).total

    np.testing.assert_allclose(small.rate, 1.0e-12 * base.rate, rtol=2.0e-15)
    np.testing.assert_allclose(
        small.electron_density_derivative,
        1.0e-12 * base.electron_density_derivative,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        small.hole_density_derivative,
        1.0e-12 * base.hole_density_derivative,
        rtol=1.0e-14,
    )


def test_exact_multi_species_derivatives_match_centered_density_difference():
    n = np.asarray([2.0e18, 9.0e20, 3.0e22])
    p = np.asarray([4.0e21, 7.0e19, 5.0e17])
    ni_sq = np.full(n.shape, 2.5e28)
    active = np.ones(n.size, dtype=bool)
    model = _model(
        _compiled_species(
            "absorber/one",
            active=active,
            tau_n_s=8.0e-7,
            tau_p_s=2.0e-7,
            n1_m3=1.0e12,
            p1_m3=2.5e16,
        ),
        _compiled_species(
            "absorber/two",
            active=active,
            tau_n_s=4.0e-8,
            tau_p_s=3.0e-6,
            n1_m3=7.0e17,
            p1_m3=3.0e10,
        ),
    )
    analytic = evaluate_neutral_bulk_defects(n, p, ni_sq, model).total
    relative_step = 2.0e-6
    finite_n = np.empty_like(n)
    finite_p = np.empty_like(p)
    for index in range(n.size):
        n_step = relative_step * n[index]
        n_plus = n.copy()
        n_minus = n.copy()
        n_plus[index] += n_step
        n_minus[index] -= n_step
        finite_n[index] = (
            evaluate_neutral_bulk_defects(n_plus, p, ni_sq, model).total.rate[index]
            - evaluate_neutral_bulk_defects(n_minus, p, ni_sq, model).total.rate[index]
        ) / (2.0 * n_step)
        p_step = relative_step * p[index]
        p_plus = p.copy()
        p_minus = p.copy()
        p_plus[index] += p_step
        p_minus[index] -= p_step
        finite_p[index] = (
            evaluate_neutral_bulk_defects(n, p_plus, ni_sq, model).total.rate[index]
            - evaluate_neutral_bulk_defects(n, p_minus, ni_sq, model).total.rate[index]
        ) / (2.0 * p_step)

    np.testing.assert_allclose(
        analytic.electron_density_derivative, finite_n, rtol=5.0e-6, atol=0.0
    )
    np.testing.assert_allclose(
        analytic.hole_density_derivative, finite_p, rtol=5.0e-6, atol=0.0
    )


def test_mixed_lifetime_and_explicit_nodes_dispatch_without_charge_payload():
    n = np.asarray([1.0e18, 3.0e19, 8.0e20])
    p = np.asarray([7.0e20, 4.0e19, 2.0e18])
    ni_sq = np.full(n.shape, 1.0e28)
    explicit_mask = np.asarray([False, True, True])
    model = _model(
        _compiled_species(
            "absorber/neutral",
            active=explicit_mask,
            tau_n_s=2.0e-7,
            tau_p_s=5.0e-8,
            n1_m3=9.0e15,
            p1_m3=2.0e12,
        ),
        explicit_mask=explicit_mask,
    )
    mixed = srh_recombination(
        n,
        p,
        ni_sq,
        4.0e-6,
        7.0e-6,
        3.0e14,
        4.0e14,
        neutral_bulk_defects=model,
    )
    expected_legacy = srh_recombination(
        n, p, ni_sq, 4.0e-6, 7.0e-6, 3.0e14, 4.0e14
    )
    expected_explicit = evaluate_neutral_bulk_defects(n, p, ni_sq, model)

    assert mixed[0] == expected_legacy[0]
    np.testing.assert_array_equal(
        mixed[explicit_mask], expected_explicit.total.rate[explicit_mask]
    )
    payload = expected_explicit.to_dict()
    assert payload["charge_density_C_m3"] is None
    assert payload["model_identity_sha256"] == model.identity_sha256

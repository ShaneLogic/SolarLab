"""Independent checks of the bulk recombination laws and the bound reaction.

Oracles are analytic limits, exact rational arithmetic and complex-step
differentiation of the value kernels; none of them re-uses the analytic
derivative expressions under test.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from solarlab.physics.recombination import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
    NeutralDefectSpecies,
    NeutralExplicitDefectModel,
    RecombinationDomainError,
    auger_recombination,
    lifetime_srh_recombination,
    radiative_recombination,
)

DIGEST = "ab" * 32


def _coefficients(n_nodes, **overrides):
    values = dict(
        ni_sq_m6=1.0e20,
        tau_n_s=2.0e-7,
        tau_p_s=5.0e-8,
        n1_m3=3.0e9,
        p1_m3=4.0e8,
        radiative_m3_s=2.0e-17,
        auger_n_m6_s=1.0e-40,
        auger_p_m6_s=3.0e-41,
    )
    values.update(overrides)
    return BulkRecombinationCoefficients(n_nodes, **values)


def _states(n_nodes, seed=7):
    rng = np.random.default_rng(seed)
    return 10.0 ** rng.uniform(8.0, 23.0, n_nodes), 10.0 ** rng.uniform(8.0, 23.0, n_nodes)


# --- equilibrium: net reaction --------------------------------------------


def test_every_channel_vanishes_exactly_at_mass_action():
    n, p = _states(40)
    ni_sq = n * p  # mass action holds exactly in binary64 by construction
    coefficients = BulkRecombinationCoefficients(
        40, ni_sq, 2e-7, 5e-8, np.sqrt(ni_sq) * 3.0, np.sqrt(ni_sq) / 3.0, 2e-17, 1e-40, 3e-41
    )
    evaluation = BulkRecombinationReaction(coefficients).evaluate(n, p)
    for term in (evaluation.srh, evaluation.radiative, evaluation.auger, evaluation.total):
        assert np.all(term.rate == 0.0)


@pytest.mark.parametrize("factor", [1.0e-3, 0.5, 2.0, 1.0e3])
def test_rate_sign_follows_detailed_balance(factor):
    n = np.full(5, 1.0e15)
    p = np.full(5, 1.0e5) * factor
    evaluation = BulkRecombinationReaction(_coefficients(5)).evaluate(n, p)
    expected = np.sign(factor - 1.0)
    for term in (evaluation.srh, evaluation.radiative, evaluation.auger, evaluation.total):
        assert np.all(np.sign(term.rate) == expected)


# --- low and high injection -------------------------------------------------


def test_low_injection_limits_in_p_type_material():
    tau_n, tau_p, ni_sq, n1, p1 = 1.0e-6, 1.0e-6, 1.0e20, 1.0e10, 1.0e10
    B, C_n, C_p = 2.0e-17, 1.0e-40, 3.0e-41
    p0 = 1.0e23
    n0 = ni_sq / p0
    dn = 1.0e14
    n = np.array([n0 + dn])
    p = np.array([p0 + dn])
    srh = lifetime_srh_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1).rate[0]
    rad = radiative_recombination(n, p, ni_sq, B).rate[0]
    aug = auger_recombination(n, p, ni_sq, C_n, C_p).rate[0]
    # Neglected relative terms of each expansion, from the parameters alone.
    eps = (n0 + dn) / p0 + (tau_p / tau_n) * (n[0] + n1) / p0 + (dn + p1) / p0
    assert srh == pytest.approx(dn / tau_n, rel=2.0 * eps)
    assert rad == pytest.approx(B * p0 * dn, rel=2.0 * (n0 + dn) / p0)
    assert aug == pytest.approx(C_p * p0**2 * dn, rel=3.0 * ((n0 + dn) / p0 + C_n * n[0] / (C_p * p0)))


def test_high_injection_limits():
    tau_n, tau_p, ni_sq, n1, p1 = 1.0e-6, 2.5e-7, 1.0e20, 1.0e10, 1.0e10
    B, C_n, C_p = 2.0e-17, 1.0e-40, 3.0e-41
    p0 = 1.0e16
    n0 = ni_sq / p0
    dn = 1.0e24
    n = np.array([n0 + dn])
    p = np.array([p0 + dn])
    eps = (p0 + n1 + p1) / dn
    srh = lifetime_srh_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1).rate[0]
    assert srh == pytest.approx(dn / (tau_n + tau_p), rel=3.0 * eps)
    assert radiative_recombination(n, p, ni_sq, B).rate[0] == pytest.approx(B * dn**2, rel=3.0 * eps)
    assert auger_recombination(n, p, ni_sq, C_n, C_p).rate[0] == pytest.approx(
        (C_n + C_p) * dn**3, rel=4.0 * eps
    )


# --- zero channels are strict identities -------------------------------------


def test_zero_band_to_band_channels_leave_srh_bitwise():
    n, p = _states(30)
    evaluation = BulkRecombinationReaction(
        _coefficients(30, radiative_m3_s=0.0, auger_n_m6_s=0.0, auger_p_m6_s=0.0)
    ).evaluate(n, p)
    assert np.array_equal(evaluation.total.rate, evaluation.srh.rate)
    assert np.array_equal(evaluation.total.d_rate_dn, evaluation.srh.d_rate_dn)
    assert np.array_equal(evaluation.total.d_rate_dp, evaluation.srh.d_rate_dp)


def test_blocked_capture_leg_switches_srh_off_exactly():
    n, p = _states(20)
    evaluation = BulkRecombinationReaction(_coefficients(20, tau_n_s=math.inf)).evaluate(n, p)
    assert np.all(evaluation.srh.rate == 0.0)
    assert np.all(evaluation.srh.d_rate_dn == 0.0)
    assert np.all(evaluation.srh.d_rate_dp == 0.0)


def test_unowned_nodes_are_bitwise_the_pure_lifetime_law():
    n, p = _states(12)
    coefficients = _coefficients(12)
    owned = np.zeros(12, dtype=bool)
    owned[4:8] = True
    species = NeutralDefectSpecies("layer[1]/x", DIGEST, owned, 1e-8, 2e-8,
                                   np.where(owned, 1e12, 0.0), np.where(owned, 1e8, 0.0))
    model = NeutralExplicitDefectModel((species,), owned, (("layer[1]/x", DIGEST),))
    mixed = BulkRecombinationReaction(coefficients, model).evaluate(n, p)
    pure = BulkRecombinationReaction(coefficients).evaluate(n, p)
    free = ~owned
    for name in ("rate", "d_rate_dn", "d_rate_dp"):
        assert getattr(mixed.total, name)[free].tobytes() == getattr(pure.total, name)[free].tobytes()


def test_explicit_nodes_replace_lifetime_srh_instead_of_adding_it():
    n, p = _states(10)
    owned = np.ones(10, dtype=bool)
    tau_n, tau_p, n1, p1 = 3e-8, 7e-9, 2e11, 5e9
    # One explicit species with the lifetime law's own parameters must give
    # exactly the lifetime law: if both were counted the rate would double.
    species = NeutralDefectSpecies("layer[0]/a", DIGEST, owned, tau_n, tau_p,
                                   np.full(10, n1), np.full(10, p1))
    model = NeutralExplicitDefectModel((species,), owned, (("layer[0]/a", DIGEST),))
    coefficients = _coefficients(10, tau_n_s=tau_n, tau_p_s=tau_p, n1_m3=n1, p1_m3=p1)
    explicit = BulkRecombinationReaction(coefficients, model).evaluate(n, p)
    lifetime = BulkRecombinationReaction(coefficients).evaluate(n, p)
    assert explicit.srh.rate.tobytes() == lifetime.srh.rate.tobytes()
    assert explicit.total.rate.tobytes() == lifetime.total.rate.tobytes()
    assert not np.any(explicit.lifetime_nodes)


def test_two_species_on_one_node_add_in_declaration_order():
    n, p = np.full(3, 1.0e18), np.full(3, 2.0e17)
    owned = np.ones(3, dtype=bool)
    a = NeutralDefectSpecies("l/a", DIGEST, owned, 1e-8, 2e-8, np.full(3, 1e10), np.full(3, 1e10))
    b = NeutralDefectSpecies("l/b", DIGEST, owned, 4e-9, 1e-9, np.full(3, 1e12), np.full(3, 1e8))
    model = NeutralExplicitDefectModel((a, b), owned, (("l", DIGEST),))
    evaluation = BulkRecombinationReaction(_coefficients(3), model).evaluate(n, p)
    single_a = lifetime_srh_recombination(n, p, 1e20, 1e-8, 2e-8, 1e10, 1e10).rate
    single_b = lifetime_srh_recombination(n, p, 1e20, 4e-9, 1e-9, 1e12, 1e8).rate
    assert evaluation.srh.rate.tobytes() == (single_a + single_b).tobytes()


# --- derivatives -------------------------------------------------------------


def _complex_step(function, n, p, *, wrt):
    h_n = 1.0e-20 * n if wrt == "n" else 0.0 * n
    h_p = 1.0e-20 * p if wrt == "p" else 0.0 * p
    value = function(n + 1j * h_n, p + 1j * h_p)
    return np.imag(value) / (h_n if wrt == "n" else h_p)


@pytest.mark.parametrize("law", ["srh", "radiative", "auger"])
def test_analytic_tangents_match_complex_step(law):
    n, p = _states(64, seed=11)
    ni_sq, tau_n, tau_p, n1, p1 = 1.0e20, 2e-7, 5e-8, 3e9, 4e8

    def value(nn, pp):
        if law == "srh":
            return lifetime_srh_recombination(nn, pp, ni_sq, tau_n, tau_p, n1, p1, derivatives=False).rate
        if law == "radiative":
            return radiative_recombination(nn, pp, ni_sq, 2e-17, derivatives=False).rate
        return auger_recombination(nn, pp, ni_sq, 1e-40, 3e-41, derivatives=False).rate

    eps = np.finfo(float).eps
    if law == "srh":
        exact = lifetime_srh_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1)
        # The historical quotient-rule form (p - R tau_p)/D cancels when
        # R tau_p ~ p; its honest error bound scales with the cancelling terms.
        D = tau_p * (n + n1) + tau_n * (p + p1)
        bound_n = 32 * eps * (p + np.abs(exact.rate) * tau_p) / D
        bound_p = 32 * eps * (n + np.abs(exact.rate) * tau_n) / D
    elif law == "radiative":
        exact = radiative_recombination(n, p, ni_sq, 2e-17)
        bound_n = bound_p = None
    else:
        exact = auger_recombination(n, p, ni_sq, 1e-40, 3e-41)
        numerator = np.abs(n * p - ni_sq)
        coefficient = 1e-40 * n + 3e-41 * p
        bound_n = 32 * eps * (1e-40 * numerator + coefficient * p)
        bound_p = 32 * eps * (3e-41 * numerator + coefficient * n)
    step_n = _complex_step(value, n, p, wrt="n")
    step_p = _complex_step(value, n, p, wrt="p")
    if bound_n is None:
        np.testing.assert_allclose(exact.d_rate_dn, step_n, rtol=4e-15, atol=0.0)
        np.testing.assert_allclose(exact.d_rate_dp, step_p, rtol=4e-15, atol=0.0)
    else:
        assert np.all(np.abs(exact.d_rate_dn - step_n) <= bound_n)
        assert np.all(np.abs(exact.d_rate_dp - step_p) <= bound_p)


def test_cancellation_regime_of_the_historical_srh_tangent_is_bounded_and_documented():
    """Where R tau_p ~ p the historical tangent keeps only its absolute accuracy.

    Measured, not asserted away: at n >> p the relative error of dR/dn can
    reach ~1e-5 while the absolute error stays at the round-off of the
    cancelling terms. Recorded as a conditioning property of the M-class
    expression; a cancellation-free form is a separate N decision.
    """

    n = np.array([1.0e23])
    p = np.array([1.0e9])
    exact = lifetime_srh_recombination(n, p, 1.0e10, 1e-6, 1e-6, 1e5, 1e5)
    F = Fraction
    D = F(1e-6) * (F(1e23) + F(1e5)) + F(1e-6) * (F(1e9) + F(1e5))
    N = F(1e23) * F(1e9) - F(1e10)
    # d/dn (N/D) = (p D - N tau_p) / D^2, evaluated exactly.
    true = (F(1e9) * D - N * F(1e-6)) / D**2
    scale = (F(1e9) + abs(F(exact.rate[0])) * F(1e-6)) / D
    assert abs(F(exact.d_rate_dn[0]) - true) <= 32 * scale * F(2) ** -52


def test_total_tangent_is_the_sum_of_channel_tangents():
    n, p = _states(25)
    evaluation = BulkRecombinationReaction(_coefficients(25)).evaluate(n, p)
    assert np.array_equal(
        evaluation.total.d_rate_dn,
        evaluation.srh.d_rate_dn + evaluation.radiative.d_rate_dn + evaluation.auger.d_rate_dn,
    )


def test_srh_value_matches_exact_rational_arithmetic():
    # Points far from mass action, so the float error is a few ulp of R.
    cases = [(1e17, 3e15), (2.5e21, 7e8), (4e9, 6e22), (1e12, 1e12)]
    ni_sq, tau_n, tau_p, n1, p1 = 1.0e10, 3e-7, 4e-8, 2e5, 5e4
    for n, p in cases:
        rate = lifetime_srh_recombination(np.array([n]), np.array([p]), ni_sq, tau_n, tau_p, n1, p1).rate[0]
        F = Fraction
        exact = (F(n) * F(p) - F(ni_sq)) / (F(tau_p) * (F(n) + F(n1)) + F(tau_n) * (F(p) + F(p1)))
        assert abs(F(rate) - exact) <= 8 * abs(exact) * F(2) ** -52


# --- reaction instance --------------------------------------------------------


def test_carrier_sources_come_from_one_rate_array():
    n, p = _states(9)
    evaluation = BulkRecombinationReaction(_coefficients(9)).evaluate(n, p)
    sources = evaluation.carrier_sources()
    assert sources.electron is sources.hole
    assert np.array_equal(sources.electron, -evaluation.total.rate)
    assert np.array_equal(sources.d_source_dn, -evaluation.total.d_rate_dn)
    assert sources.stoichiometry == (("electron", -1), ("hole", -1))
    assert sources.unit == "m^-3 s^-1"


def test_value_only_evaluation_matches_the_full_one():
    n, p = _states(15)
    reaction = BulkRecombinationReaction(_coefficients(15))
    assert reaction.evaluate(n, p, derivatives=False).total.rate.tobytes() == reaction.evaluate(n, p).total.rate.tobytes()
    assert reaction.evaluate(n, p, derivatives=False).total.d_rate_dn is None


def test_observed_denominators_and_domain_predicate():
    n, p = _states(6)
    evaluation = BulkRecombinationReaction(_coefficients(6)).evaluate(n, p)
    assert evaluation.srh_denominators_s_m3.shape == (6,)
    assert evaluation.minimum_srh_denominator_s_m3 == float(np.min(evaluation.srh_denominators_s_m3))
    evaluation.require_domain()
    negative = BulkRecombinationReaction(_coefficients(6)).evaluate(-n, p)
    assert not negative.in_domain()
    with pytest.raises(RecombinationDomainError):
        negative.require_domain()


def test_evaluation_outputs_are_read_only():
    n, p = _states(4)
    evaluation = BulkRecombinationReaction(_coefficients(4)).evaluate(n, p)
    with pytest.raises(ValueError):
        evaluation.total.rate[0] = 1.0
    with pytest.raises(ValueError):
        evaluation.srh_denominators_s_m3[0] = 1.0


@pytest.mark.parametrize(
    "override",
    [
        {"tau_n_s": 0.0},
        {"tau_p_s": -1e-9},
        {"tau_n_s": math.nan},
        {"radiative_m3_s": -1e-17},
        {"auger_n_m6_s": math.inf},
        {"n1_m3": -1.0},
        {"ni_sq_m6": math.nan},
        {"tau_n_s": np.ones(3)},
    ],
)
def test_invalid_coefficients_are_rejected_at_binding(override):
    with pytest.raises(ValueError):
        _coefficients(5, **override)


def test_state_shape_and_finiteness_are_checked():
    reaction = BulkRecombinationReaction(_coefficients(4))
    with pytest.raises(ValueError):
        reaction.evaluate(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        reaction.evaluate(np.array([1.0, np.nan, 1.0, 1.0]), np.ones(4))


def test_explicit_model_must_share_the_grid():
    owned = np.ones(3, dtype=bool)
    species = NeutralDefectSpecies("l/a", DIGEST, owned, 1e-8, 1e-8, np.ones(3), np.ones(3))
    model = NeutralExplicitDefectModel((species,), owned, (("l", DIGEST),))
    with pytest.raises(ValueError):
        BulkRecombinationReaction(_coefficients(4), model)


def test_neutral_model_rejects_inconsistent_ownership():
    owned = np.array([True, False, True])
    species = NeutralDefectSpecies("l/a", DIGEST, owned, 1e-8, 1e-8, np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        NeutralExplicitDefectModel((species,), np.ones(3, dtype=bool), (("l", DIGEST),))
    with pytest.raises(ValueError):
        NeutralExplicitDefectModel((species, species), owned, (("l", DIGEST),))


def test_finite_lifetime_overflow_is_not_a_disabled_srh_channel():
    value = np.array([float(2**600)])
    with np.errstate(over='ignore'):
        with pytest.raises(FloatingPointError, match='finite-lifetime SRH'):
            lifetime_srh_recombination(value, value, 1.0, value, value, 1.0, 1.0)


def test_nonfinite_reaction_output_cannot_pass_the_domain_guard():
    from dataclasses import replace
    coefficients = replace(_coefficients(1), radiative_m3_s=np.array([1e300]))
    with np.errstate(over='ignore', invalid='ignore'):
        evaluation = BulkRecombinationReaction(coefficients).evaluate(np.array([1e20]), np.array([1e20]))
    assert not evaluation.in_domain()
    with pytest.raises(ValueError, match='finite'):
        evaluation.require_domain()

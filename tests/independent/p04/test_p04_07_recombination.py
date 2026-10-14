"""P04-07 oracle: MB lifetime SRH, radiative and Auger (values, derivatives, limits)."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import EPS, gate
from p04_oracle_recombination import (
    BulkParameters,
    auger_exact,
    exponential_edge_profile_ref,
    gaussian_edge_profile_ref,
    low_injection_srh_ratio,
    radiative_exact,
    srh_exact,
    tau_from_trap_density_ref,
    total_exact,
    within,
)

NI = 2.89e10
PAR = BulkParameters(ni_sq=NI * NI, tau_n=3e-9, tau_p=3e-7, n1=NI, p1=NI,
                     B_rad=4e-17, C_n=2.8e-41, C_p=9.9e-42)
RNG = np.random.default_rng(2024)
N_STATE = 10.0 ** RNG.uniform(-30, 30, 64)
P_STATE = 10.0 ** RNG.uniform(-30, 30, 64)
N_STATE[:4] = (NI, 1e22, 1e10, 3e23)
P_STATE[:4] = (NI, 1e10, 1e22, 3e23)
CHANNELS = {"srh": srh_exact, "radiative": radiative_exact, "auger": auger_exact}


def _legacy_channels(n, p, par: BulkParameters, channels):
    rec = legacy.recombination()
    terms = []
    if "srh" in channels:
        terms.append(rec.srh_recombination_derivatives(n, p, par.ni_sq, par.tau_n, par.tau_p, par.n1, par.p1))
    if "radiative" in channels:
        terms.append(rec.radiative_recombination_derivatives(n, p, par.ni_sq, par.B_rad))
    if "auger" in channels:
        terms.append(rec.auger_recombination_derivatives(n, p, par.ni_sq, par.C_n, par.C_p))
    rate = sum(np.asarray(t.rate) for t in terms)
    dn = sum(np.asarray(t.electron_density_derivative) for t in terms)
    dp = sum(np.asarray(t.hole_density_derivative) for t in terms)
    return rate, dn, dp


SUBJECTS = {
    "legacy": _legacy_channels,
    "candidate": lambda n, p, par, ch: adapter.bulk_recombination(n, p, par, ch),
}


def _exact(n, p, par, channels):
    parts = [CHANNELS[c](n, p, par) for c in channels]
    out = parts[0]
    for extra in parts[1:]:
        out = type(out)(*(getattr(out, f) + getattr(extra, f) for f in out.__dataclass_fields__))
    return out


@pytest.mark.parametrize("subject", list(SUBJECTS))
@pytest.mark.parametrize("channels", (("srh",), ("radiative",), ("auger",), ("srh", "radiative", "auger")))
def test_rate_and_derivatives_against_exact_rational(subject, channels):
    gv, gd = gate("P0407-REC-VAL"), gate("P0407-REC-DER")
    with np.errstate(all="ignore"):
        rate, dn, dp = (np.asarray(a, dtype=float) for a in SUBJECTS[subject](N_STATE, P_STATE, PAR, channels))
    for i, (n, p) in enumerate(zip(N_STATE, P_STATE)):
        ex = _exact(n, p, PAR, channels)
        assert within(rate[i], ex.rate, ex.gross, gv.rel_limit / EPS), f"{subject} {channels} rate @ {n:.3g},{p:.3g}"
        assert within(dn[i], ex.d_n, ex.gross_d_n, gd.rel_limit / EPS), f"{subject} {channels} dR/dn @ {i}"
        assert within(dp[i], ex.d_p, ex.gross_d_p, gd.rel_limit / EPS), f"{subject} {channels} dR/dp @ {i}"


def test_exact_derivatives_match_rational_difference_quotients():
    """Independent check of the derivative formulas themselves (no binary64)."""

    h = Fraction(1, 10**40)
    for n, p in ((1e15, 3e16), (2.89e10, 1e22), (5e20, 5e20)):
        for fn in CHANNELS.values():
            ex = fn(n, p, PAR)
            fp = fn(float(n), float(p), PAR)
            assert fp.rate == ex.rate
            # central difference on the exact rational function
            n_q, p_q = Fraction(n), Fraction(p)

            def rate_at(nn, pp):
                return _rational_rate(fn, nn, pp)

            dn_fd = (rate_at(n_q + h, p_q) - rate_at(n_q - h, p_q)) / (2 * h)
            dp_fd = (rate_at(n_q, p_q + h) - rate_at(n_q, p_q - h)) / (2 * h)
            assert abs(dn_fd - ex.d_n) <= Fraction(1, 10**20) * (abs(ex.d_n) + ex.gross_d_n)
            assert abs(dp_fd - ex.d_p) <= Fraction(1, 10**20) * (abs(ex.d_p) + ex.gross_d_p)


def _rational_rate(fn, n: Fraction, p: Fraction) -> Fraction:
    ni2, tn, tp = Fraction(PAR.ni_sq), Fraction(PAR.tau_n), Fraction(PAR.tau_p)
    n1, p1 = Fraction(PAR.n1), Fraction(PAR.p1)
    U = n * p - ni2
    if fn is srh_exact:
        return U / (tp * (n + n1) + tn * (p + p1))
    if fn is radiative_exact:
        return Fraction(PAR.B_rad) * U
    return (Fraction(PAR.C_n) * n + Fraction(PAR.C_p) * p) * U


# ---------------------------------------------------------------- equilibrium


@pytest.mark.parametrize("subject", list(SUBJECTS))
def test_net_rate_vanishes_at_equilibrium(subject):
    g = gate("P0407-REC-EQ")
    # exactly representable mass action: n p == ni^2 in binary64
    a = np.array([-40, -3, 0, 12, 37, 60])
    n_eq, p_eq = 2.0 ** a, 2.0 ** (40 - a)
    par = BulkParameters(ni_sq=2.0 ** 40, tau_n=1e-6, tau_p=2e-6, n1=2.0 ** 20, p1=2.0 ** 20,
                         B_rad=1e-16, C_n=1e-40, C_p=2e-40)
    rate, dn, dp = SUBJECTS[subject](n_eq, p_eq, par, ("srh", "radiative", "auger"))
    for i in range(a.size):
        ex = total_exact(n_eq[i], p_eq[i], par)
        assert ex.rate == 0
        assert abs(Fraction(float(np.asarray(rate)[i]))) <= Fraction(g.rel_limit) * ex.gross
        assert within(np.asarray(dn)[i], ex.d_n, ex.gross_d_n, gate("P0407-REC-DER").rel_limit / EPS)
    # rounded equilibrium (n = ni e^u, p = ni e^-u)
    u = np.linspace(-30, 30, 13)
    n_r, p_r = NI * np.exp(u), NI * np.exp(-u)
    rate_r, _, _ = SUBJECTS[subject](n_r, p_r, PAR, ("srh", "radiative", "auger"))
    for i in range(u.size):
        gross = total_exact(n_r[i], p_r[i], PAR).gross
        assert abs(Fraction(float(np.asarray(rate_r)[i]))) <= Fraction(4 * g.rel_limit) * gross


def test_legacy_exact_equilibrium_is_bitwise_zero():
    """M-class expectation: the historical U = n p - ni^2 form gives 0.0 exactly."""

    a = np.array([-40, 0, 37])
    par = BulkParameters(2.0 ** 40, 1e-6, 2e-6, 2.0 ** 20, 2.0 ** 20, 1e-16, 1e-40, 2e-40)
    rate, _, _ = _legacy_channels(2.0 ** a, 2.0 ** (40 - a), par, ("srh", "radiative", "auger"))
    assert np.all(rate == 0.0)


# ---------------------------------------------------------------- limits


def test_injection_limits_closed_form():
    g = gate("P0407-REC-LIMIT")
    p0, n0 = 1e23, PAR.ni_sq / 1e23
    for delta in (1e12, 1e15, 1e17):  # low injection, Delta/p0 <= 1e-6
        ratio = low_injection_srh_ratio(n0, p0, delta, PAR)
        ex = srh_exact(n0 + delta, p0 + delta, BulkParameters(n0 * p0, PAR.tau_n, PAR.tau_p, PAR.n1, PAR.p1, 0, 0, 0))
        assert ex.rate * Fraction(PAR.tau_n) / Fraction(delta) - ratio <= Fraction(1, 10**12) * ratio
        predicted = (delta + PAR.p1 + (PAR.tau_p / PAR.tau_n) * (n0 + delta + PAR.n1)) / p0
        assert abs(float(ratio) - 1.0) <= g.rel_limit * predicted + g.abs_limit
        rad = radiative_exact(n0 + delta, p0 + delta, BulkParameters(n0 * p0, 1, 1, 0, 0, PAR.B_rad, 0, 0))
        assert abs(float(rad.rate) / (PAR.B_rad * p0 * delta) - 1.0) <= 2 * (n0 + delta) / p0 + 4 * EPS
        aug = auger_exact(n0 + delta, p0 + delta, BulkParameters(n0 * p0, 1, 1, 0, 0, 0, PAR.C_n, PAR.C_p))
        assert abs(float(aug.rate) / (PAR.C_p * p0 * p0 * delta) - 1.0) <= 4 * (delta / p0) * (1 + PAR.C_n / PAR.C_p) + 4 * EPS
    for delta in (1e18, 1e20, 1e22):  # high injection n = p = Delta >= 1e6 n1
        ex = srh_exact(delta, delta, PAR)
        predicted = (PAR.tau_p * PAR.n1 + PAR.tau_n * PAR.p1) / ((PAR.tau_n + PAR.tau_p) * delta) + PAR.ni_sq / delta**2
        assert abs(float(ex.rate) * (PAR.tau_n + PAR.tau_p) / delta - 1.0) <= g.rel_limit * predicted + g.abs_limit
        assert abs(float(radiative_exact(delta, delta, PAR).rate) / (PAR.B_rad * delta**2) - 1.0) <= 2 * PAR.ni_sq / delta**2 + 4 * EPS
        assert abs(float(auger_exact(delta, delta, PAR).rate) / ((PAR.C_n + PAR.C_p) * delta**3) - 1.0) <= 2 * PAR.ni_sq / delta**2 + 4 * EPS


@pytest.mark.parametrize("subject", list(SUBJECTS))
def test_low_injection_lifetime_limit_on_subject(subject):
    g = gate("P0407-REC-LIMIT")
    p0, n0 = 1e23, PAR.ni_sq / 1e23
    delta = np.array([1e12, 1e14, 1e16])
    par = BulkParameters(n0 * p0, PAR.tau_n, PAR.tau_p, PAR.n1, PAR.p1, 0.0, 0.0, 0.0)
    rate, _, _ = SUBJECTS[subject](n0 + delta, p0 + delta, par, ("srh",))
    predicted = (delta + PAR.p1 + (PAR.tau_p / PAR.tau_n) * (n0 + delta + PAR.n1)) / p0
    assert np.all(np.abs(np.asarray(rate) * PAR.tau_n / delta - 1.0) <= g.rel_limit * predicted + g.abs_limit)


# ---------------------------------------------------------------- zero channels


@pytest.mark.parametrize("subject", list(SUBJECTS))
def test_zero_coefficient_channels_are_strict_identities(subject):
    par0 = BulkParameters(PAR.ni_sq, PAR.tau_n, PAR.tau_p, PAR.n1, PAR.p1, 0.0, 0.0, 0.0)
    with np.errstate(all="ignore"):
        srh_only = SUBJECTS[subject](N_STATE, P_STATE, par0, ("srh",))
        everything = SUBJECTS[subject](N_STATE, P_STATE, par0, ("srh", "radiative", "auger"))
        rad = SUBJECTS[subject](N_STATE, P_STATE, par0, ("radiative",))
        aug = SUBJECTS[subject](N_STATE, P_STATE, par0, ("auger",))
    for a, b in zip(srh_only, everything):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "disabled channels must add exactly 0.0"
    for arr in (*rad, *aug):
        arr = np.asarray(arr)
        assert np.all(arr == 0.0) and not np.any(np.isnan(arr))


def test_legacy_infinite_lifetime_limit_documented():
    """LD-04: tau = inf gives rate 0 but NaN derivatives in the lifetime form.

    The rate limit is correct (blocked cycle); the derivative (p - R tau_p)/Den
    evaluates 0*inf.  Shipped presets never use tau = inf for the lifetime
    form (explicit species skip non-cycling legs), but a new implementation
    must either represent a disabled SRH channel as an absent instance or
    return exact zeros -- never NaN (gate P0407-REC-ZERO).
    """

    rec = legacy.recombination()
    with np.errstate(all="ignore"):
        d = rec.srh_recombination_derivatives(np.array([1e15]), np.array([1e16]), NI * NI, np.inf, np.inf, NI, NI)
    assert np.asarray(d.rate)[0] == 0.0
    assert np.isnan(np.asarray(d.electron_density_derivative)[0])


# ---------------------------------------------------------------- explicit defect


def _neutral_model(mask, tau_n, tau_p, n1, p1):
    rec = legacy.recombination()
    digest = hashlib.sha256(b"r3-oracle-neutral-defect").hexdigest()
    species = rec.CompiledNeutralDefectSpecies(
        identifier="r3_defect", document_sha256=digest, active_nodes=mask,
        tau_n_s=tau_n, tau_p_s=tau_p, n1_m3=np.full(mask.size, n1), p1_m3=np.full(mask.size, p1))
    return rec.NeutralBulkDefectModel(species=(species,), explicit_node_mask=mask,
                                      layer_document_sha256=(("absorber", digest),))


def test_legacy_explicit_defect_is_not_double_counted():
    g = gate("P0407-REC-NODOUBLE")
    n, p = N_STATE[:12], P_STATE[:12]
    mask = np.zeros(12, dtype=bool)
    mask[3:9] = True
    tau_n_d, tau_p_d, n1_d, p1_d = 5e-8, 7e-8, 1e12, NI * NI / 1e12
    model = _neutral_model(mask, tau_n_d, tau_p_d, n1_d, p1_d)
    rec = legacy.recombination()
    out = np.asarray(rec.srh_recombination(n, p, PAR.ni_sq, PAR.tau_n, PAR.tau_p, PAR.n1, PAR.p1,
                                           neutral_bulk_defects=model))
    defect = BulkParameters(PAR.ni_sq, tau_n_d, tau_p_d, n1_d, p1_d, 0, 0, 0)
    for i in range(12):
        ex = srh_exact(n[i], p[i], defect if mask[i] else PAR)
        assert within(out[i], ex.rate, ex.gross, g.rel_limit / EPS), f"node {i} (explicit={mask[i]})"


def test_candidate_explicit_defect_is_not_double_counted():
    g = gate("P0407-REC-NODOUBLE")
    n, p = N_STATE[:12], P_STATE[:12]
    mask = np.zeros(12, dtype=bool)
    mask[3:9] = True
    out = np.asarray(adapter.bulk_recombination_with_explicit_defect(n, p, PAR, mask, 5e-8, 7e-8, 1e12, NI * NI / 1e12))
    defect = BulkParameters(PAR.ni_sq, 5e-8, 7e-8, 1e12, NI * NI / 1e12, PAR.B_rad, PAR.C_n, PAR.C_p)
    for i in range(12):
        ex = total_exact(n[i], p[i], defect if mask[i] else PAR)
        assert within(out[i], ex.rate, ex.gross, g.rel_limit / EPS)


# ---------------------------------------------------------------- trap profiles


TRAP_X = np.linspace(0.0, 400e-9, 41)


def _legacy_trap(shape, x, d, nif, nb, length, edge):
    t = legacy.traps()
    fn = t.exponential_edge_profile if shape == "exponential" else t.gaussian_edge_profile
    return fn(x, d, nif, nb, length, edge)


@pytest.mark.parametrize("subject", ("legacy", "candidate"))
@pytest.mark.parametrize("shape", ("exponential", "gaussian"))
@pytest.mark.parametrize("edge", ("both", "left", "right"))
@pytest.mark.parametrize("length", (5e-9, 30e-9, 400e-9, 4e-6, 0.0))
def test_trap_profiles(subject, shape, edge, length):
    g = gate("P0407-TRAP")
    args = (TRAP_X, 400e-9, 1e23, 1e20, length, edge)
    out = (np.asarray(_legacy_trap(shape, *args)) if subject == "legacy"
           else np.asarray(adapter.trap_profile(shape, *args)))
    ref = (exponential_edge_profile_ref if shape == "exponential" else gaussian_edge_profile_ref)(*args)
    assert np.max(np.abs(out - ref) / np.abs(ref)) <= g.rel_limit
    if length == 0.0:
        assert np.all(out == 1e20)
    if shape == "exponential" and edge == "both" and length > 0:
        assert abs(out[0] / 1e23 - 1.0) <= 4 * EPS and abs(out[-1] / 1e23 - 1.0) <= 4 * EPS
        assert np.all(out <= 1e23 * (1 + 4 * EPS))


def test_tau_from_trap_density_floor():
    t = legacy.traps()
    N = np.array([1e19, 1e20, 5e20, 1e23])
    assert np.array_equal(t.tau_from_trap_density(1e-6, N, 1e20), tau_from_trap_density_ref(1e-6, N, 1e20))
    assert tau_from_trap_density_ref(1e-6, N, 1e20)[0] == 1e-6  # passivation floor

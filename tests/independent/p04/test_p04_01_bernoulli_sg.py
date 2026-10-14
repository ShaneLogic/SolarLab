"""P04-01 oracle: robust Bernoulli value/derivative and Scharfetter-Gummel flux.

Each check runs against three subjects:

* ``oracle_ref`` -- the independent binary64 reference (self-validation);
* ``legacy`` -- ``perovskite_sim.discretization.fe_operators`` (history);
* ``candidate_m`` -- W3a's historical (M-class) kernels through ``p04_adapter``;
* ``candidate_n`` -- W3a's new-arithmetic (N-class) kernels through ``p04_adapter``.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

import p04_adapter as adapter
import p04_legacy as legacy
from p04_gates import EPS, gate
from p04_oracle_bernoulli import (
    HP_DIGITS,
    bernoulli_derivative_hp,
    bernoulli_derivative_ref,
    bernoulli_hp,
    bernoulli_ref,
    sg_equilibrium_bound,
    sg_particle_flux_hp,
    sg_particle_flux_jacobian_ref,
    sg_particle_flux_ref,
)

Q = 1.602176634e-19


def _core_grid() -> np.ndarray:
    pos = np.unique(np.concatenate([
        np.logspace(-300, np.log10(700.0), 500),
        [1e-8, 9.999e-5, 1e-4, 1.0001e-4, 0.0499, 0.05, 0.0501, 1.0, 50.0, 699.9, 700.0],
    ]))
    pos = pos[pos <= 700.0]  # the gate scope is |x| <= 700 exactly
    return np.concatenate([-pos[::-1], [0.0], pos])


CORE_X = _core_grid()
TAIL_POS = np.array([700.0000000000001, 701.0, 705.0, 709.8, 710.0, 714.0, 745.0, 746.0, 1e3, 1e5])
TAIL_NEG = -np.array([700.5, 710.0, 745.0, 1e3, 1e5])


@lru_cache(maxsize=None)
def _hp_values(xs: tuple[float, ...]) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    return tuple(bernoulli_hp(x) for x in xs), tuple(bernoulli_derivative_hp(x) for x in xs)


def _legacy_b(x):
    with np.errstate(all="ignore"):
        return legacy.fe_operators().bernoulli(np.asarray(x, dtype=float))


def _legacy_db(x):
    with np.errstate(all="ignore"):
        return legacy.fe_operators().bernoulli_derivative(np.asarray(x, dtype=float))


SUBJECTS_B = {
    "oracle_ref": bernoulli_ref,
    "legacy": _legacy_b,
    "candidate_m": lambda x: adapter.bernoulli(x),
    "candidate_n": lambda x: adapter.bernoulli_stable(x),
}
SUBJECTS_DB = {
    "oracle_ref": bernoulli_derivative_ref,
    "legacy": _legacy_db,
    "candidate_m": lambda x: adapter.bernoulli_derivative(x),
    "candidate_n": lambda x: adapter.bernoulli_derivative_stable(x),
}


def _rel_errors(values, refs) -> np.ndarray:
    out = []
    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        for v, r in zip(np.asarray(values, dtype=float), refs):
            if not np.isfinite(v):
                out.append(np.inf)
            elif r == 0:
                out.append(abs(float(v)))
            else:
                out.append(float(abs((Decimal(float(v)) - r) / r)))
    return np.array(out)


# ------------------------------------------------------------- oracle itself


def test_hp_oracle_satisfies_exact_identities():
    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        for x in (1e-20, 1e-9, 0.3, 2.0, 50.0, 700.0, 1e4):
            ident = bernoulli_hp(-x) - bernoulli_hp(x) - Decimal(x)
            assert abs(ident) <= Decimal("1e-85") * (1 + Decimal(x))
            dident = bernoulli_derivative_hp(-x) + bernoulli_derivative_hp(x) + 1
            assert abs(dident) <= Decimal("1e-85")
    assert bernoulli_derivative_hp(0.0) == Decimal(-1) / 2


def test_hp_derivative_matches_decimal_central_difference():
    """Validates the analytic derivative independently of its own formula.

    With h = 1e-35 at 100 digits the quotient keeps >= 55 significant digits
    even where e^x - 1 cancels (x ~ 1e-6), so 1e-50 is a safe absolute bound.
    """

    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        h = Decimal("1e-35")
        for x in (-30.0, -1.0, -1e-3, 2e-6, 0.04, 0.7, 3.0, 40.0):
            d = Decimal(x)
            fd = ((d + h) / ((d + h).exp() - 1) - (d - h) / ((d - h).exp() - 1)) / (2 * h)
            assert abs(fd - bernoulli_derivative_hp(x)) <= Decimal("1e-50")


# ------------------------------------------------------------- value gates


@pytest.mark.parametrize("subject", list(SUBJECTS_B))
def test_bernoulli_value_core(subject):
    g = gate("P0401-BERN-VAL")
    B = np.asarray(SUBJECTS_B[subject](CORE_X), dtype=float)
    refs, _ = _hp_values(tuple(CORE_X))
    err = _rel_errors(B, refs)
    worst = int(np.argmax(err))
    assert err[worst] <= g.rel_limit, (
        f"{subject}: B({CORE_X[worst]!r}) rel error {err[worst]:.3e} > {g.rel_limit:.3e}"
    )
    # B(50) must survive as the small branch (plan text: no subtraction loss).
    b50 = float(np.asarray(SUBJECTS_B[subject](np.array([50.0])))[0])
    assert abs(b50 / float(bernoulli_hp(50.0)) - 1.0) <= g.rel_limit


@pytest.mark.parametrize("subject", list(SUBJECTS_B))
def test_bernoulli_value_tail(subject):
    g = gate("P0401-BERN-TAIL")
    Bp = np.asarray(SUBJECTS_B[subject](TAIL_POS), dtype=float)
    refs_p, _ = _hp_values(tuple(TAIL_POS))
    assert np.all(Bp >= 0.0) and np.all(np.isfinite(Bp))
    for x, b, r in zip(TAIL_POS, Bp, refs_p):
        assert abs(Decimal(float(b)) - r) <= Decimal(g.abs_limit), f"{subject}: B({x})={b}"
    Bn = np.asarray(SUBJECTS_B[subject](TAIL_NEG), dtype=float)
    refs_n, _ = _hp_values(tuple(TAIL_NEG))
    assert np.max(_rel_errors(Bn, refs_n)) <= g.rel_limit


def test_legacy_tail_deviation_is_the_documented_zero_branch():
    """LD-01: legacy returns exactly 0 for x > 700 (true value <= 6.9e-302)."""

    B = _legacy_b(TAIL_POS)
    assert np.all(B == 0.0)
    assert float(bernoulli_hp(float(TAIL_POS[0]))) < 7e-302


@pytest.mark.parametrize("subject", list(SUBJECTS_B))
def test_bernoulli_large_argument_identity(subject):
    g = gate("P0401-BERN-IDENT")
    xp = CORE_X[CORE_X > 0]
    Bm = np.asarray(SUBJECTS_B[subject](-xp), dtype=float)
    Bp = np.asarray(SUBJECTS_B[subject](xp), dtype=float)
    worst = 0.0
    for x, bm, bp in zip(xp, Bm, Bp):
        exact = abs(Fraction(float(bm)) - Fraction(float(bp)) - Fraction(float(x)))
        worst = max(worst, float(exact / Fraction(np.spacing(float(bm)))))
    assert worst <= g.rel_limit, f"{subject}: identity off by {worst:.2f} ulp(B(-x))"


@pytest.mark.parametrize("subject", list(SUBJECTS_DB))
def test_bernoulli_derivative_value(subject):
    g = gate("P0401-DBERN-VAL")
    dB = np.asarray(SUBJECTS_DB[subject](CORE_X), dtype=float)
    _, refs = _hp_values(tuple(CORE_X))
    err = []
    for v, r in zip(dB, refs):
        if abs(r) < Decimal("1e-300"):
            err.append(0.0 if abs(Decimal(float(v)) - r) <= Decimal(g.abs_limit) else np.inf)
        else:
            err.append(float(abs((Decimal(float(v)) - r) / r)) if np.isfinite(v) else np.inf)
    err = np.array(err)
    worst = int(np.argmax(err))
    assert err[worst] <= g.rel_limit, (
        f"{subject}: B'({CORE_X[worst]!r}) rel error {err[worst]:.3e} > {g.rel_limit:.1e}"
    )


@pytest.mark.parametrize("subject", list(SUBJECTS_DB))
def test_bernoulli_derivative_identity_and_shape(subject):
    gi, gs = gate("P0401-DBERN-IDENT"), gate("P0401-DBERN-SHAPE")
    xp = CORE_X[CORE_X > 0]
    f = SUBJECTS_DB[subject]
    ident = np.asarray(f(-xp)) + np.asarray(f(xp)) + 1.0
    assert np.max(np.abs(ident)) <= gi.abs_limit
    wide = np.sort(np.concatenate([CORE_X, TAIL_NEG, TAIL_POS[TAIL_POS <= 745.0]]))
    d = np.asarray(f(wide), dtype=float)
    assert np.all(np.isfinite(d)), gs.zero_rule
    assert np.all((d >= -1.0) & (d <= 0.0))
    assert np.all(np.diff(d) >= 0.0), "B' must be non-decreasing (B convex)"
    assert float(np.asarray(f(np.array([0.0])))[0]) == -0.5


def test_legacy_nonfinite_inputs_are_limits_or_nan():
    """Non-finite inputs may give the analytic limit or NaN, never another finite value."""

    B = _legacy_b(np.array([-np.inf, np.inf, np.nan]))
    assert B[0] == np.inf and B[1] == 0.0 and np.isnan(B[2])
    dB = _legacy_db(np.array([-np.inf, np.inf]))
    assert all(np.isnan(v) or v in (-1.0, 0.0) for v in dB)  # LD-02: legacy gives NaN


# ------------------------------------------------------------- SG flux


def _sg_case(seed: int, z: int):
    rng = np.random.default_rng(seed)
    n = 41
    h = rng.uniform(0.2e-9, 8e-9, n - 1)
    V_T = 0.025851999786435
    psi = np.cumsum(np.concatenate([[0.0], rng.uniform(-0.6, 0.6, n - 1)])) * 0.5
    psi = np.clip(psi, -4.5, 4.5)
    D = rng.uniform(1e-6, 2e-4, n - 1)
    c = 1e15 * np.exp(-z * psi / V_T)
    return psi, c, h, D, V_T


def _legacy_particle_flux(psi, c, h, D, V_T, z):
    fe = legacy.fe_operators()
    if z == -1:
        return -fe.sg_fluxes_n(psi, c, h, D, V_T) / Q
    return fe.sg_fluxes_p(psi, c, h, D, V_T) / Q


def _legacy_particle_jacobian(psi, c, h, D, V_T, z):
    fe = legacy.fe_operators()
    sign, jac = (-1.0, fe.sg_fluxes_n_jacobian) if z == -1 else (1.0, fe.sg_fluxes_p_jacobian)
    j = jac(psi, c, h, D, V_T)
    return {
        "flux": sign * j.flux / Q,
        "d_c_left": sign * j.density_left_derivative / Q,
        "d_c_right": sign * j.density_right_derivative / Q,
        "d_psi_left": sign * j.potential_left_derivative / Q,
        "d_psi_right": sign * j.potential_right_derivative / Q,
    }


SUBJECTS_SG = {
    "oracle_ref": sg_particle_flux_ref,
    "legacy": _legacy_particle_flux,
    "candidate_m": lambda *a: adapter.sg_particle_flux_historical(*a),
    "candidate_n": lambda *a: adapter.sg_particle_flux(*a),
}
SUBJECTS_JAC = {
    "oracle_ref": sg_particle_flux_jacobian_ref,
    "legacy": _legacy_particle_jacobian,
    "candidate_m": lambda *a: adapter.sg_particle_flux_jacobian_historical(*a),
    "candidate_n": lambda *a: adapter.sg_particle_flux_jacobian(*a),
}


def _leg_sum(psi, c, h, D, V_T, z):
    xi = (psi[1:] - psi[:-1]) / V_T
    return (D / h) * (bernoulli_ref(z * xi) * c[:-1] + bernoulli_ref(-z * xi) * c[1:])


@pytest.mark.parametrize("subject", list(SUBJECTS_SG))
@pytest.mark.parametrize("z", (-1, 1))
@pytest.mark.parametrize("seed", (1, 2, 3))
def test_sg_flux_vanishes_at_boltzmann_equilibrium(subject, z, seed):
    assert gate("P0401-SG-EQ").rel_limit == 8.0  # factor inside sg_equilibrium_bound
    psi, c, h, D, V_T = _sg_case(seed, z)
    F = np.asarray(SUBJECTS_SG[subject](psi, c, h, D, V_T, z), dtype=float)
    bound = sg_equilibrium_bound(psi[:-1], psi[1:], V_T)
    ratio = np.abs(F) / _leg_sum(psi, c, h, D, V_T, z)
    assert np.all(ratio <= bound), f"{subject}: worst {np.max(ratio / bound):.2f} x bound"


@pytest.mark.parametrize("subject", list(SUBJECTS_SG))
@pytest.mark.parametrize("z", (-1, 1))
def test_sg_flux_value_against_decimal(subject, z):
    g = gate("P0401-SG-VAL")
    rng = np.random.default_rng(7)
    n = 25
    V_T = 0.025851999786435
    psi = np.concatenate([[0.0], np.cumsum(rng.choice([-1, 1], n - 1) * rng.uniform(0, 5.0, n - 1))])
    c = 10.0 ** rng.uniform(5, 24, n)
    h = rng.uniform(1e-10, 1e-8, n - 1)
    D = rng.uniform(1e-7, 1e-3, n - 1)
    F = np.asarray(SUBJECTS_SG[subject](psi, c, h, D, V_T, z), dtype=float)
    legs = _leg_sum(psi, c, h, D, V_T, z)
    for i in range(n - 1):
        ref = sg_particle_flux_hp(psi[i], psi[i + 1], c[i], c[i + 1], h[i], D[i], V_T, z)
        assert abs(Decimal(float(F[i])) - ref) <= Decimal(g.rel_limit) * Decimal(float(legs[i])), (
            f"{subject}: face {i} xi={(psi[i + 1] - psi[i]) / V_T:.3g}"
        )


@pytest.mark.parametrize("z", (-1, 1))
def test_sg_limits_diffusion_and_upwind_drift(z):
    """Reference-only physics limits (Fick at xi=0; upwind drift at |xi|>>1)."""

    V_T, D, h = 0.0258, 1e-4, 1e-9
    c = np.array([3e20, 1e20])
    F0 = sg_particle_flux_ref(np.array([0.0, 0.0]), c, np.array([h]), D, V_T, z)[0]
    assert F0 == pytest.approx(-D * (c[1] - c[0]) / h, rel=4 * EPS)
    for dpsi in (40 * V_T, -40 * V_T):
        F = sg_particle_flux_ref(np.array([0.0, dpsi]), c, np.array([h]), D, V_T, z)[0]
        zxi = z * dpsi / V_T
        up = c[1] if zxi > 0 else c[0]
        assert F == pytest.approx(-(D / h) * zxi * up, rel=1e-15 + 50 * np.exp(-abs(zxi)))


@pytest.mark.parametrize("subject", list(SUBJECTS_JAC))
@pytest.mark.parametrize("z", (-1, 1))
def test_sg_jacobian_against_decimal_derivative(subject, z):
    g = gate("P0401-SG-JAC")
    rng = np.random.default_rng(11)
    n = 15
    V_T = 0.025851999786435
    psi = np.concatenate([[0.0], np.cumsum(rng.uniform(-3.0, 3.0, n - 1))])
    c = 10.0 ** rng.uniform(8, 22, n)
    h = rng.uniform(1e-10, 1e-8, n - 1)
    D = rng.uniform(1e-7, 1e-3, n - 1)
    J = SUBJECTS_JAC[subject](psi, c, h, D, V_T, z)
    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        for i in range(n - 1):
            xi = (Decimal(float(psi[i + 1])) - Decimal(float(psi[i]))) / Decimal(V_T)
            zx = z * xi
            pref = Decimal(float(D[i])) / Decimal(float(h[i]))
            b_pos = _b_dec(zx)
            b_neg = _b_dec(-zx)
            db_pos, db_neg = _db_dec(zx), _db_dec(-zx)
            ci, cj = Decimal(float(c[i])), Decimal(float(c[i + 1]))
            d_xi = pref * z * (db_pos * ci + db_neg * cj)
            ref = {
                "d_c_left": pref * b_pos,
                "d_c_right": -pref * b_neg,
                "d_psi_left": -d_xi / Decimal(V_T),
                "d_psi_right": d_xi / Decimal(V_T),
            }
            psi_scale = pref * (abs(db_pos) * ci + abs(db_neg) * cj) / Decimal(V_T)
            for key, r in ref.items():
                v = Decimal(float(np.asarray(J[key])[i]))
                scale = abs(r) + (psi_scale if "psi" in key else 0)
                assert abs(v - r) <= Decimal(g.rel_limit) * scale, f"{subject}: {key} face {i}"


def _b_dec(d: Decimal) -> Decimal:
    if d == 0:
        return Decimal(1)
    if abs(d) < Decimal("1e-15"):
        return 1 - d / 2 + d * d / 12
    if d > 0:
        t = (-d).exp()
        return d * t / (1 - t)
    return d / (d.exp() - 1)


def _db_dec(d: Decimal) -> Decimal:
    if d == 0:
        return Decimal(-1) / 2
    if abs(d) < Decimal("1e-15"):
        return Decimal(-1) / 2 + d / 6
    if d > 0:
        t = (-d).exp()
        return t * (1 - d - t) / (1 - t) ** 2
    e = d.exp()
    return (e - 1 - d * e) / (e - 1) ** 2


# ------------------------------------------------------------- face averages


def _legacy_harmonic(a, b):
    from perovskite_sim.solver.mol import _harmonic_face_average

    return _harmonic_face_average(np.stack([a, b]).T.ravel())[::2]


@pytest.mark.parametrize("subject", ("legacy", "candidate_guarded", "candidate_poisson"))
def test_harmonic_face_mean_is_exact_series_and_preserves_zero(subject):
    """Exact series value; the zero rule applies to the guarded (transport) mean.

    The Poisson-channel mean only ever sees positive permittivities (the
    operator rejects eps <= 0), so it is checked on positive pairs only.
    """

    g = gate("P0401-FACE-MEAN")
    a = np.array([3.0, 24.1, 1e-10, 2.5e-4, 0.0, 0.0, 7.0])
    b = np.array([24.1, 10.0, 6.62e-3, 2.5e-4, 5.0, 0.0, 7.0])
    if subject == "candidate_poisson":
        keep = (a > 0) & (b > 0)
        a, b = a[keep], b[keep]
        out = np.asarray(adapter.harmonic_face_mean(a, b, variant="poisson"), dtype=float)
    elif subject == "candidate_guarded":
        out = np.asarray(adapter.harmonic_face_mean(a, b, variant="guarded"), dtype=float)
    else:
        out = np.asarray(_legacy_harmonic(a, b), dtype=float)
    for ai, bi, oi in zip(a, b, out):
        if ai == 0.0 or bi == 0.0:
            assert oi == 0.0, "a zero conductivity must give exactly 0 (no NaN)"
            continue
        exact = 2 * Fraction(ai) * Fraction(bi) / (Fraction(ai) + Fraction(bi))
        assert abs(Fraction(float(oi)) - exact) <= Fraction(g.rel_limit) * exact


def test_stable_pair_differs_from_legacy_only_by_the_legacy_error():
    """N-class triangle check: |pair - legacy| <= |pair - hp| + |legacy - hp|.

    The fixed P04-GOLDEN-N tolerance does not apply to ``bernoulli_pair``: the
    historical reference itself carries the x > 700 zero branch (LD-01) and a
    3.4e-12 derivative error, so the difference must be explained by the two
    measured errors instead of a fixed ratio.
    """

    refs_b, refs_d = _hp_values(tuple(CORE_X))
    pair_b = np.asarray(adapter.bernoulli_stable(CORE_X))
    pair_d = np.asarray(adapter.bernoulli_derivative_stable(CORE_X))
    old_b, old_d = _legacy_b(CORE_X), _legacy_db(CORE_X)
    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        for new, old, refs in ((pair_b, old_b, refs_b), (pair_d, old_d, refs_d)):
            for v_new, v_old, r in zip(new, old, refs):
                dn, do = Decimal(float(v_new)), Decimal(float(v_old))
                bound = abs(dn - r) + abs(do - r)
                assert abs(dn - do) <= bound * (1 + Decimal("1e-60")) + Decimal("1e-320")

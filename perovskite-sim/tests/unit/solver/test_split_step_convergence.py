"""Operator-splitting convergence of ``solver/mol.py:split_step``.

Review item §7 #8 / finding F-12.

What is under test
------------------
``split_step`` advances the ion subsystem over a macro-step ``dt`` with the
carriers FROZEN (Poisson is re-solved inside the ion RHS, so the ions feel
their own space charge but not the carrier response), then re-equilibrates
the carriers with a short full-system transient.  That is a Lie--Trotter
(first-order) splitting of the ion operator against the carrier operator.
The manual asserts first order *from reading the code*; this file is the
measurement that makes the claim honest.

Method
------
Fix a total physical time ``T`` and an initial state ``y0``.  Compare

  (a) ``n = T/dt`` successive ``split_step(dt)`` calls, for
      ``dt = T/4, T/8, T/16`` (the same total ``T`` each time), against
  (b) a single fully-coupled ``run_transient`` over ``(0, T)`` — the
      reference, in which ions and carriers are integrated together.

Error norm: the thickness-weighted L2 norm of the mobile-ion profile
``P(x)`` restricted to the absorber (the only layer with ``D_ion > 0``),
normalised by the reference's OWN ion displacement ``||P_ref - P_0||``.
That makes ``E(dt)`` dimensionless and directly readable as "the splitting
error is this fraction of the ion redistribution the reference performed".
The terminal current ``J`` is carried as a second, independent norm.

MEASURED RESULT (this machine, 2026-07-26, N=31 nodes, rtol/atol 1e-6/1e-8)
--------------------------------------------------------------------------
``T = 0.2 s``, reference converged, ions moved ``max|dP|/P0 = 9.54e-2``:

    n= 2  dt=0.1000   E = 1.497671e-01   |dJ| = 3.781855e-02 A/m^2
    n= 4  dt=0.0500   E = 7.743821e-02   |dJ| = 1.799908e-02
    n= 8  dt=0.0250   E = 3.932132e-02   |dJ| = 8.861439e-03
    n=16  dt=0.0125   E = 1.976270e-02   |dJ| = 4.496056e-03
    n=32  dt=0.00625  E = 9.813944e-03   |dJ| = 2.354954e-03

Per-halving error ratios (P): 1.934, 1.969, 1.990, 2.014 — textbook
Lie--Trotter.  Log-log slope over the asserted ladder (n = 4, 8, 16):
**0.9851** for P and 1.0006 for J.  The manual's first-order claim is
CONFIRMED by measurement, not assumed.

Latent regression hazard (a real property of the code, stated not hidden)
-------------------------------------------------------------------------
``split_step`` re-equilibrates the carriers over a HARD-CODED
``t_eq = 1e-6 s`` of FULL-system ``run_transient`` (mol.py, end of
``split_step``) that does not scale with ``dt``.  The ions therefore advance
by ``dt + t_eq`` per macro-step, so an ``n``-step sequence over-integrates
the ions by ``n*t_eq = T*(t_eq/dt)`` — a bias that GROWS as ``dt`` shrinks.
At the finest step in this ladder it is ``16 us / 0.2 s = 8e-5`` of ``T``
and invisible, which is exactly why the ladder was chosen here.  Push this
test to much finer ``dt``, or raise ``t_eq``, and the convergence will stall
and then reverse.  If that happens, the finding is the ``t_eq`` bias — do
not widen the bounds below.

Band-bending clause
-------------------
The manual also claims the splitting error "grows with the ion-induced band
bending".  ``test_split_step_error_grows_with_ion_induced_band_bending``
measures both halves of that: the ABSOLUTE error does grow ~5x from
V_app = 0.9 V to V_app = 0.0 (5.217e18 vs 1.040e18), but the
MOTION-NORMALISED error is bias-flat (7.744e-2 vs 8.209e-2, ratio 1.06).
So the growth is entirely "the ions move further at higher field", not a
per-unit-motion degradation of the splitting.  The manual sentence is true
only in that trivial absolute sense and should be worded accordingly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _compute_current_ss
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec,
    build_material_arrays,
    run_transient,
    split_step,
)

pytestmark = pytest.mark.slow

CONFIG = "configs/ionmonger_benchmark.yaml"
NODES_PER_LAYER = 10          # N = 31, matches the split_step tests in test_mol.py

# Total physical time.  Derived from the ionic timescale, NOT from
# convenience: the analytic ionic dielectric-relaxation time of the absorber
# is  tau = eps_r*eps0*V_T / (q*P0*D_ion)
#         = 24.1*8.854e-12*0.02585 / (1.602e-19*1.6e25*1.01e-17) = 0.213 s
# (equivalently lambda_D^2/D_ion with lambda_D = 1.46 nm).  A measured
# saturation sweep (fully-coupled run to 3.2 s) gives tau_eff ~ 0.78 s on
# this grid, so T = 0.2 s is ~0.26 tau_eff: the ions demonstrably move
# (~22% of the full redistribution) while the reference still costs ~0.2 s.
T_TOTAL = 0.2

# Refinement ladder: n successive split_step calls of dt = T/n.
LADDER_N = (4, 8, 16)

# Tight ODE tolerances chosen for HEADROOM, not because they change the
# answer: rerunning the whole ladder at the split_step defaults
# (rtol 1e-4 / atol 1e-6) reproduces E to 5 significant figures
# (7.743803e-2 vs 7.743821e-2), i.e. the splitting error dominates the
# Radau error by ~5 orders.  See GUARD-C, which measures this directly.
RTOL = 1e-6
ATOL = 1e-8

# Second bias point for the band-bending leg.  V_bi = 1.1 V on this preset,
# so V_app = 0.9 V leaves (V_bi - V_app) = 0.2 V of band bending versus
# 1.1 V at short circuit — a 5.5x drive ratio.
V_FLAT = 0.9


# ── helpers ─────────────────────────────────────────────────────────────
def _l2(u: np.ndarray, xs: np.ndarray) -> float:
    """Thickness-weighted (trapezoidal) L2 norm of u on grid xs."""
    return float(math.sqrt(np.trapezoid(u**2, xs)))


def _observed_order(dts: np.ndarray, errs: np.ndarray) -> float:
    """Slope of log(err) vs log(dt) — empirical convergence order.

    Same construction as ``tests/unit/solver/test_mms.py:_observed_order``.
    """
    slope, _ = np.polyfit(np.log(dts), np.log(errs), 1)
    return float(slope)


def _absorber_mask(x: np.ndarray, stack) -> np.ndarray:
    """Nodes belonging to the absorber — the only layer with D_ion > 0."""
    lo = stack.layers[0].thickness
    hi = lo + stack.layers[1].thickness
    return (x >= lo - 1e-15) & (x <= hi + 1e-15)


def _ions(y: np.ndarray, N: int, mat) -> np.ndarray:
    return StateVec.unpack(y, N, N_iface_state=mat.N_iface_state).P


def _run_ladder(x, stack, mat, y0, V_app, n_steps):
    """Advance y0 over T_TOTAL with n_steps successive split_step calls."""
    dt = T_TOTAL / n_steps
    y = y0.copy()
    for k in range(n_steps):
        y, ok = split_step(
            x, y, dt=dt, stack=stack, V_app=V_app,
            rtol=RTOL, atol=ATOL, mat=mat,
        )
        assert ok, (
            f"split_step reported failure on macro-step {k + 1}/{n_steps} "
            f"(dt={dt:.6g} s, V_app={V_app} V) — the convergence measurement "
            f"below is meaningless without a successful ion advance."
        )
    return y


def _leg(x, stack, mat, mask, V_app, n_steps):
    """One bias point: reference + one split ladder rung.

    Returns (abs_err, motion, normalised_err) on the absorber ion profile.
    """
    N = len(x)
    xa = x[mask]
    y0 = solve_illuminated_ss(x, stack, V_app=V_app)
    P0 = _ions(y0, N, mat)
    ref = run_transient(
        x, y0, (0.0, T_TOTAL), np.array([T_TOTAL]), stack,
        illuminated=True, V_app=V_app, rtol=RTOL, atol=ATOL, mat=mat,
    )
    assert ref.success, (
        f"fully-coupled reference failed at V_app={V_app} V: {ref.message}"
    )
    P_ref = _ions(ref.y[:, -1], N, mat)
    motion = _l2((P_ref - P0)[mask], xa)
    y_split = _run_ladder(x, stack, mat, y0, V_app, n_steps)
    abs_err = _l2((_ions(y_split, N, mat) - P_ref)[mask], xa)
    return abs_err, motion, abs_err / motion


# ── module-scoped baseline (V_app = 0): reference + full ladder ─────────
@pytest.fixture(scope="module")
def splitting_baseline():
    """Reference run + the dt-refinement ladder at short circuit.

    Module-scoped so the two test functions share ~5 s of integration.
    """
    stack = load_device_from_yaml(CONFIG)
    x = multilayer_grid([Layer(l.thickness, NODES_PER_LAYER) for l in stack.layers])
    mat = build_material_arrays(x, stack)
    N = len(x)
    mask = _absorber_mask(x, stack)
    xa = x[mask]

    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    P0 = _ions(y0, N, mat)

    ref = run_transient(
        x, y0, (0.0, T_TOTAL), np.array([T_TOTAL]), stack,
        illuminated=True, V_app=0.0, rtol=RTOL, atol=ATOL, mat=mat,
    )
    # GUARD-A is asserted in the test body; carry the object through.
    y_ref = ref.y[:, -1] if ref.success else None
    P_ref = _ions(y_ref, N, mat) if ref.success else None
    motion = _l2((P_ref - P0)[mask], xa) if ref.success else float("nan")
    J_ref = (
        _compute_current_ss(x, y_ref, stack, 0.0, mat=mat) if ref.success else float("nan")
    )

    # GUARD-C material: same reference at 10x looser tolerances.
    ref_loose = run_transient(
        x, y0, (0.0, T_TOTAL), np.array([T_TOTAL]), stack,
        illuminated=True, V_app=0.0, rtol=10 * RTOL, atol=10 * ATOL, mat=mat,
    )
    ref_self_err = (
        _l2((_ions(ref_loose.y[:, -1], N, mat) - P_ref)[mask], xa)
        if (ref.success and ref_loose.success)
        else float("nan")
    )

    dts, errs, j_errs, abs_errs = [], [], [], []
    if ref.success:
        for n in LADDER_N:
            y_split = _run_ladder(x, stack, mat, y0, 0.0, n)
            a = _l2((_ions(y_split, N, mat) - P_ref)[mask], xa)
            dts.append(T_TOTAL / n)
            abs_errs.append(a)
            errs.append(a / motion)
            j_errs.append(
                abs(_compute_current_ss(x, y_split, stack, 0.0, mat=mat) - J_ref)
            )

    return {
        "stack": stack, "x": x, "mat": mat, "mask": mask,
        "ref": ref, "ref_loose": ref_loose,
        "P0": P0, "P_ref": P_ref, "motion": motion, "J_ref": J_ref,
        "ref_self_err": ref_self_err,
        "dts": np.array(dts), "errs": np.array(errs),
        "abs_errs": np.array(abs_errs), "j_errs": np.array(j_errs),
    }


# ── test 1: the convergence certificate ─────────────────────────────────
def test_split_step_converges_first_order_to_the_fully_coupled_solution(
    splitting_baseline,
):
    """Halving the ion macro-step must halve the error against the coupled run.

    This is the measurement behind the manual's "first-order operator
    splitting" sentence.  Three independent guards keep it from passing
    vacuously, then the core claim is asserted theory-free (strict monotone
    shrinkage) before the order band is applied.
    """
    b = splitting_baseline
    stack = b["stack"]

    # GUARD-A — the reference is a reference.
    assert b["ref"].success, (
        "fully-coupled reference run_transient failed; the whole comparison "
        f"is vacuous. message: {b['ref'].message}"
    )

    # GUARD-B — the ions actually moved over T.  Floor is a pure
    # "not a no-op" certificate set one decade below the T-implied
    # displacement, not fitted to it.
    P0_layer = stack.layers[1].params.P0        # 1.6e25 m^-3, uniform vacancies
    moved = float(np.max(np.abs((b["P_ref"] - b["P0"])[b["mask"]])) / P0_layer)
    assert moved >= 1e-2, (
        f"ions barely moved over T={T_TOTAL} s: max|dP|/P0 = {moved:.4e} < 1e-2. "
        "The splitting comparison would be measuring nothing. Lengthen T."
    )

    # GUARD-C — the reference is not itself tolerance-limited.  Its own ODE
    # error must sit well below the SMALLEST splitting error the ladder
    # resolves, otherwise the ladder is measuring Radau noise.
    assert b["ref_loose"].success, b["ref_loose"].message
    ref_noise = b["ref_self_err"] / b["motion"]
    assert ref_noise <= 1e-2, (
        f"reference self-consistency {ref_noise:.4e} (rtol {RTOL} vs {10*RTOL}) "
        "is not far enough below the splitting errors "
        f"{b['errs']} for the ladder to be meaningful."
    )

    dts, errs, j_errs = b["dts"], b["errs"], b["j_errs"]

    # CORE — strictly monotone shrinkage.  No tolerance: this is the honest
    # form of the manual's claim, and the theory-free part of the result.
    assert errs[0] > errs[1] > errs[2], (
        "split_step is NOT converging to the fully-coupled solution: "
        f"E(dt) = {errs} for dt = {dts} s (should shrink strictly). "
        "This CONTRADICTS the manual's first-order-splitting claim — report "
        "it rather than weakening this assertion."
    )

    order = _observed_order(dts, errs)

    # ORDER-LOWER — theory for Lie--Trotter is exactly 1.0.  The 0.2 margin
    # mirrors the sibling convention in test_mms.py (theory 2.0, assert 1.8).
    assert order >= 0.8, (
        f"observed splitting order {order:.4f} < 0.8 — the error shrinks but "
        f"far slower than first order. E(dt) = {errs}, dt = {dts} s. Something "
        "is polluting the ladder (candidate: the hard-coded t_eq = 1e-6 s "
        "carrier re-equilibration, whose ion over-integration bias grows as "
        "1/dt). Diagnose it; do not lower this bound."
    )

    # ORDER-UPPER — pins the *first*-order claim specifically. 1.5 is the
    # midpoint between first (1.0) and second (2.0) order.
    assert order <= 1.5, (
        f"observed splitting order {order:.4f} > 1.5 — split_step is doing "
        "BETTER than first order (Strang / symmetric splitting?). That is good "
        "news, but it makes the manual's 'first-order operator splitting' "
        "sentence STALE. UPDATE THE MANUAL CLAIM AND THIS BOUND — do not "
        f"loosen it to hide the improvement. E(dt) = {errs}, dt = {dts} s."
    )

    # CURRENT — a second, independent norm.  Monotonicity only: the terminal
    # current is a sum of large nearly-cancelling components (CLAUDE.md F-13 /
    # the EQE-plateau-noise finding), so no order band is asserted on J.
    assert j_errs[0] > j_errs[1] > j_errs[2], (
        "terminal-current error does not shrink monotonically under macro-step "
        f"refinement: |J_split - J_ref| = {j_errs} A/m^2 for dt = {dts} s."
    )


# ── test 2: the band-bending clause ─────────────────────────────────────
def test_split_step_error_grows_with_ion_induced_band_bending(splitting_baseline):
    """Does the splitting error grow with band bending? Both readings.

    BAND-BENDING-A: at fixed dt, the ABSOLUTE profile error at short circuit
    (full built-in field) is much larger than at V_app = 0.9 V (nearly flat
    band).  Ion drift velocity is linear in field and (V_bi - V_app) drops
    1.1 -> 0.2 V, so linear response predicts ~5.5x; the gate asserts only
    > 1.5, i.e. direction plus order of magnitude.

    BAND-BENDING-B: the MOTION-NORMALISED errors at the two biases agree
    within 1.5x.  That is the honest counterpoint — the growth in (A) is
    fully accounted for by there being more ion redistribution at higher
    field, NOT by a per-unit-motion degradation of the splitting.  So the
    manual's "error grows with the ion-induced band bending" is true only in
    the trivial absolute sense.
    """
    b = splitting_baseline
    assert b["ref"].success, b["ref"].message

    n_fixed = LADDER_N[0]                     # dt = T/4, the coarsest rung
    abs_sc = b["abs_errs"][0]                 # V_app = 0, reused from the fixture
    motion_sc = b["motion"]
    norm_sc = b["errs"][0]

    abs_fb, motion_fb, norm_fb = _leg(
        b["x"], b["stack"], b["mat"], b["mask"], V_FLAT, n_fixed,
    )

    # BAND-BENDING-A
    growth = abs_sc / abs_fb
    assert growth > 1.5, (
        "absolute splitting error does NOT grow with band bending: "
        f"||P_split - P_ref||_L2 = {abs_sc:.6e} at V=0 vs {abs_fb:.6e} at "
        f"V={V_FLAT} V (ratio {growth:.3f}). Linear response predicts ~5.5x "
        f"from the (V_bi - V_app) drive ratio. Ion motion itself scaled "
        f"{motion_sc / motion_fb:.3f}x."
    )

    # BAND-BENDING-B
    ratio = norm_sc / norm_fb
    assert 1.0 / 1.5 <= ratio <= 1.5, (
        "motion-normalised splitting error is NOT bias-flat: "
        f"E = {norm_sc:.6e} at V=0 vs {norm_fb:.6e} at V={V_FLAT} V "
        f"(ratio {ratio:.3f}). If this fails, the band-bending growth in "
        "BAND-BENDING-A is more than the linear-response scaling of the ion "
        "displacement, and the manual's clause is true in a stronger sense "
        "than currently documented — record the new reading."
    )

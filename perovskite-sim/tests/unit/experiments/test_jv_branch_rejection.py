"""Wrong-branch rejection in the illuminated forward power quadrant.

WHAT BROKE
----------
Radau's error estimator under-reports truncation error near flat band, where
the Jacobian is nearly singular, and can accept a step that landed on the
carrier-injection branch of the implicit system.  ``_integrate_step`` already
capped ``max_step`` at ``dt/20`` for exactly this reason, and its docstring
claimed that "removes the spikes" -- but every recovery path in that function
(bisection, the BDF fallback) is gated on ``sol.success == False``, and a
wrong-branch step **succeeds**.  Nothing fired.

Measured on ``ionmonger_benchmark`` at the shipped benchmark grid
(``N_grid=40, n_points=20``), whose V = 1.10526 sample sits essentially
exactly on ``V_bi`` = 1.1::

    [14] V=1.03158  J=  204.124
    [15] V=1.10526  J=  509.796   <-- 2.3x J_sc, between two sane neighbours
    [16] V=1.17895  J=   51.706

which produced FF = 2.132 and PCE = 0.563 -- both impossible, since they say
``P_mpp > V_oc * J_sc`` -- and HI = -1.740 against a |HI| < 0.05 gate.

WHY NOT JUST TIGHTEN max_step
-----------------------------
Measured, sweeping the divisor on the failing config::

    dt/20 -> 509.8   dt/100 -> 162.0   dt/200 -> 255.9
    dt/1000 -> 162.0   dt/2000 -> 162.0

Non-monotone: a *global* max_step change perturbs every step's trajectory and
merely relocates which one lands wrong.  Same lesson as the E9.3 clamp-shape
variants.  ``test_tighter_max_step_is_not_a_fix`` pins that this remains true,
so the cheaper-looking fix is not re-attempted.

THE FIX
-------
Require local subdivision convergence.  V=0 has no lower current bound, so a
4-leg probe detects both under- and over-reads and an independent 8-leg result
must confirm both current and state before replacement.  At positive bias the
protocol ceiling still triggers the 2/4/8 ladder, but two successive results
must agree; merely falling below the ceiling is not enough.  Probe failures
warn and preserve the successful single-leg result.

BLAS PINNING IS LOAD-BEARING HERE
--------------------------------
The wrong-branch landing is **BLAS-thread-dependent** — measured on the gate
config with the guard disabled::

    unpinned (12 threads)   J(1.10526) = 161.986   FF_fwd = 0.79687
    pinned to 1 thread      J(1.10526) = 509.796   FF_fwd = 2.13233

Different thread counts reassociate the LU reductions, which perturbs the
Radau trajectory just enough to change which branch the near-singular step
lands on.  That is why the defect reached ``main``: ``tests/conftest.py`` pins
BLAS only when the ``slow`` marker is selected, so the ``slow`` lane saw it
(``test_hysteresis_index_bounded`` failed) while a default ``pytest`` run was
legitimately green.  Same sensitivity class as the note on
``test_flat_band_contacts`` in CLAUDE.md.

This module therefore pins BLAS itself rather than relying on the marker, so
it is deterministic in either lane.  It is NOT marked ``slow``: the whole
lesson is that this bug class hid behind that marker.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep as JV
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml

# The configuration the defect was measured on.
_GATE = dict(N_grid=40, n_points=20, v_rate=5.0)
_SPIKE_V = 1.10526          # the sample that sits on V_bi = 1.1
_CONVERGED_J = 161.986      # agreed by 2/4/8 legs, BDF, and 4 refinements
_SINGLE_LEG_J = 509.796     # what the unguarded path returned
_ZERO_BIAS_J = 223.0        # agreed by 4/8 legs across the failing grids


@pytest.fixture(scope="module", autouse=True)
def _pin_blas():
    """Pin BLAS to one thread for this module (see the module docstring).

    numpy and scipy are already imported by the time this runs, which is the
    ordering ``threadpoolctl`` requires — it can only see backends that are
    already loaded.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        yield


@pytest.fixture(scope="module")
def gate_stack():
    return load_device_from_yaml("configs/ionmonger_benchmark.yaml")


@pytest.fixture(scope="module")
def gate_result(gate_stack):
    return run_jv_sweep(gate_stack, **_GATE)


@pytest.fixture(scope="module")
def zero_bias_result(gate_stack):
    """The measured N=41 under-read counterexample, after recovery."""
    return run_jv_sweep(
        gate_stack, N_grid=41, n_points=15, v_rate=0.1,
    )


def _run_fake_sweep(monkeypatch, gate_stack, make_state):
    """Exercise branch mechanics without depending on a BLAS branch landing."""
    calls = []

    def fake_illuminated_ss(x, *_args, **_kwargs):
        return np.zeros(3 * len(x), dtype=float)

    def fake_integrate(
        x, y, _stack, _mat, V_app, _t_lo, _t_hi, _rtol, _atol,
        max_bisect=10, illuminated=True, n_legs=1,
    ):
        del max_bisect, illuminated
        out = make_state(float(V_app), int(n_legs), y.copy(), len(x))
        calls.append((float(V_app), int(n_legs), y.copy(), out.copy()))
        return out

    monkeypatch.setattr(JV, "solve_illuminated_ss", fake_illuminated_ss)
    monkeypatch.setattr(JV, "_integrate_step", fake_integrate)
    monkeypatch.setattr(
        JV, "_compute_current", lambda _x, y, *_a, **_kw: float(y[0]),
    )
    result = run_jv_sweep(
        gate_stack, N_grid=6, n_points=3, V_max=0.2, v_rate=1.0,
    )
    return result, calls


def _fake_state(y, current, marker=0.0, ion=0.0):
    out = np.zeros_like(y)
    N = len(y) // 3
    out[0] = current
    out[1] = marker
    out[2 * N] = ion
    return out


# ---------------------------------------------------------------------------
# the physical contract
# ---------------------------------------------------------------------------

def test_forward_current_never_exceeds_jsc(gate_result):
    """The bound itself, asserted over the whole forward power quadrant.

    This is the statement the rejector enforces, checked independently of it:
    no forward-bias point may carry more current than short circuit.
    """
    V, J = gate_result.V_fwd, gate_result.J_fwd
    J_sc = float(J[0])
    fwd = V > 0.0
    worst = float(np.max(J[fwd]))
    assert worst <= J_sc * (1.0 + JV._J_BRANCH_EXCESS), (
        f"forward J peaks at {worst:.3f} A/m^2 against J_sc = {J_sc:.3f} "
        f"(ceiling {J_sc * (1.0 + JV._J_BRANCH_EXCESS):.3f}) -- a step landed "
        "on the carrier-injection branch"
    )


def test_spike_sample_lands_on_the_converged_branch(gate_result):
    """The specific sample that used to spike now reads the converged value."""
    i = int(np.argmin(np.abs(gate_result.V_fwd - _SPIKE_V)))
    J = float(gate_result.J_fwd[i])
    assert abs(J - _CONVERGED_J) < 5.0, (
        f"J(V={gate_result.V_fwd[i]:.5f}) = {J:.3f}, expected the converged "
        f"{_CONVERGED_J} (the unguarded path returned {_SINGLE_LEG_J})"
    )


def test_fill_factor_is_physical(gate_result):
    """FF > 1 means P_mpp > V_oc*J_sc, which no J-V curve can do.

    The pre-fix value was 2.132.  A plain ``FF <= 1`` bound is the honest
    assertion here -- it cannot be satisfied by a wrong-branch curve and it
    encodes no fitted number.
    """
    for tag, m in (("fwd", gate_result.metrics_fwd),
                   ("rev", gate_result.metrics_rev)):
        assert 0.0 < m.FF <= 1.0, f"{tag} FF = {m.FF:.5f} outside (0, 1]"
        assert 0.0 < m.PCE <= 1.0, f"{tag} PCE = {m.PCE:.5f} outside (0, 1]"


def test_hysteresis_index_is_bounded(gate_result):
    """HI collapsed to -1.740 when the forward branch spiked (PCE_fwd was
    2.7x PCE_rev).  ionmonger_benchmark is a weak-hysteresis stack."""
    assert abs(gate_result.hysteresis_index) < 0.05, (
        f"|HI| = {abs(gate_result.hysteresis_index):.4f}"
    )


# ---------------------------------------------------------------------------
# the guard's mechanics
# ---------------------------------------------------------------------------

def test_rejector_is_inert_on_a_healthy_sweep(monkeypatch):
    """A healthy sweep must retain its original single-leg output exactly.

    Raise the positive-bias ceiling and force the V=0 agreement check true so
    neither recovery can replace a state. The 4-leg diagnostic still executes,
    but its result is discarded just as it must be on the healthy path.
    """
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    kw = dict(N_grid=30, n_points=8, v_rate=5.0)
    r_on = run_jv_sweep(stack, **kw)
    monkeypatch.setattr(JV, "_J_BRANCH_EXCESS", np.inf)
    monkeypatch.setattr(JV, "_refinements_agree", lambda *_a, **_kw: True)
    r_off = run_jv_sweep(stack, **kw)
    np.testing.assert_array_equal(r_on.J_fwd, r_off.J_fwd)
    np.testing.assert_array_equal(r_on.J_rev, r_off.J_rev)


def test_guard_actually_fires_on_the_gate_config(gate_stack, monkeypatch):
    """Turning the guard OFF must bring the spike back.

    Without this the suite could pass because the spike had quietly gone away
    for some unrelated reason, leaving the rejector untested.
    """
    monkeypatch.setattr(JV, "_J_BRANCH_EXCESS", np.inf)
    r = run_jv_sweep(gate_stack, **_GATE)
    i = int(np.argmin(np.abs(r.V_fwd - _SPIKE_V)))
    assert float(r.J_fwd[i]) > 400.0, (
        "guard disabled but no spike appeared -- this test no longer "
        f"exercises the rejector (J = {float(r.J_fwd[i]):.3f})"
    )
    assert r.metrics_fwd.FF > 1.0


def test_zero_bias_underread_recovers_on_the_converged_branch(
    zero_bias_result,
):
    """N=41 used to read 19.755 A/m2 while 4/8 legs agree near 223."""
    r = zero_bias_result
    assert float(r.J_fwd[0]) == pytest.approx(_ZERO_BIAS_J, abs=1.0)
    assert float(r.J_rev[-1]) == pytest.approx(_ZERO_BIAS_J, abs=1.0)
    assert 0.0 < r.metrics_fwd.FF <= 1.0
    assert 0.0 < r.metrics_rev.FF <= 1.0


def test_zero_bias_confirmation_replaces_current_and_forwarded_state(
    gate_stack, monkeypatch,
):
    """The 8-leg confirmation, not the single/4-leg state, drives next V."""
    def make_state(V, legs, y, _N):
        if V == 0.0:
            current = {1: 467.55, 4: 223.01, 8: 223.00}[legs]
            return _fake_state(y, current, marker=legs * 0.01, ion=1.0)
        return _fake_state(y, 200.0, marker=99.0, ion=1.0)

    r, calls = _run_fake_sweep(monkeypatch, gate_stack, make_state)
    assert float(r.J_fwd[0]) == pytest.approx(223.0, abs=0.1)
    first_positive = next(c for c in calls if c[0] > 0.0 and c[1] == 1)
    assert first_positive[2][1] == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("failed_leg", "message"),
    ((4, "subdivision probe failed"), (8, "subdivision confirmation failed")),
)
def test_zero_bias_probe_failure_keeps_successful_single_leg(
    gate_stack, monkeypatch, failed_leg, message,
):
    def make_state(V, legs, y, _N):
        if V == 0.0 and legs == failed_leg:
            raise RuntimeError("synthetic retry failure")
        current = 10.0 if legs == 1 else 20.0
        return _fake_state(y, current, marker=legs * 0.01, ion=1.0)

    with pytest.warns(RuntimeWarning, match=message):
        r, calls = _run_fake_sweep(monkeypatch, gate_stack, make_state)
    assert float(r.J_fwd[0]) == 10.0
    first_positive = next(c for c in calls if c[0] > 0.0 and c[1] == 1)
    assert first_positive[2][1] == pytest.approx(0.01)


def test_zero_bias_matching_current_but_divergent_state_is_not_accepted(
    gate_stack, monkeypatch,
):
    def make_state(V, legs, y, _N):
        if V != 0.0:
            return _fake_state(y, 5.0, ion=1.0)
        ion = {1: 1.0, 4: 10.0, 8: 20.0}[legs]
        current = 10.0 if legs == 1 else 20.0
        return _fake_state(y, current, marker=legs * 0.01, ion=ion)

    with pytest.warns(RuntimeWarning, match="failed local subdivision convergence"):
        r, _ = _run_fake_sweep(monkeypatch, gate_stack, make_state)
    assert float(r.J_fwd[0]) == 10.0


def test_positive_retry_rejects_first_below_ceiling_and_forwards_confirmed_state(
    gate_stack, monkeypatch,
):
    """A low 2-leg landing is not accepted until 4/8 agree."""
    def make_state(V, legs, y, _N):
        if V == 0.0:
            return _fake_state(y, 100.0, marker=1.0, ion=1.0)
        current = {1: 200.0, 2: 80.0, 4: 100.0, 8: 100.0}[legs]
        return _fake_state(y, current, marker=legs * 0.01, ion=1.0)

    r, calls = _run_fake_sweep(monkeypatch, gate_stack, make_state)
    assert float(r.J_fwd[1]) == 100.0
    second_positive = [c for c in calls if c[0] > 0.0 and c[1] == 1][1]
    assert second_positive[2][1] == pytest.approx(0.08)


def test_positive_retry_skips_a_failed_leg_when_later_pair_converges(
    gate_stack, monkeypatch,
):
    def make_state(V, legs, y, _N):
        if V == 0.0:
            return _fake_state(y, 100.0, marker=1.0, ion=1.0)
        if legs == 2:
            raise RuntimeError("synthetic 2-leg failure")
        current = 200.0 if legs == 1 else 100.0
        return _fake_state(y, current, marker=legs * 0.01, ion=1.0)

    r, calls = _run_fake_sweep(monkeypatch, gate_stack, make_state)
    assert float(r.J_fwd[1]) == 100.0
    assert any(V > 0.0 and legs == 8 for V, legs, *_ in calls)


def test_tighter_max_step_is_not_a_fix(gate_stack):
    """Pin the measured non-monotonicity so the divisor fix is not retried.

    dt/200 must still land on a wrong branch: it is the counterexample that
    rules out "just make max_step smaller".
    """
    orig = JV.run_transient

    def scaled(*a, **kw):
        ms = kw.get("max_step", np.inf)
        if np.isfinite(ms):
            kw["max_step"] = ms / 10.0        # dt/20 -> dt/200
        return orig(*a, **kw)

    try:
        JV.run_transient = scaled
        # Guard off, so we observe the raw integrator behaviour.
        excess = JV._J_BRANCH_EXCESS
        JV._J_BRANCH_EXCESS = np.inf
        r = run_jv_sweep(gate_stack, **_GATE)
    finally:
        JV.run_transient = orig
        JV._J_BRANCH_EXCESS = excess

    i = int(np.argmin(np.abs(r.V_fwd - _SPIKE_V)))
    assert float(r.J_fwd[i]) > _CONVERGED_J + 50.0, (
        "dt/200 no longer produces a wrong branch; the measured "
        "non-monotonicity that rules out a max_step fix has changed and the "
        "rationale in _integrate_step's docstring needs re-measuring "
        f"(J = {float(r.J_fwd[i]):.3f})"
    )


def test_unrecoverable_violation_warns_and_does_not_hide_the_value(
    gate_stack, monkeypatch,
):
    """If the retry ladder cannot satisfy the bound, warn -- never silently
    keep a number known to be unphysical, and never silently substitute one.

    Forced by making the ceiling unsatisfiable (any positive current violates
    it), so every retry fails by construction.
    """
    monkeypatch.setattr(JV, "_J_BRANCH_EXCESS", -0.999999)
    monkeypatch.setattr(JV, "_J_BRANCH_RETRY_LEGS", (2,))
    with pytest.warns(RuntimeWarning, match="above the protocol current ceiling"):
        r = run_jv_sweep(gate_stack, N_grid=30, n_points=6, v_rate=5.0)
    assert np.all(np.isfinite(r.J_fwd))


def test_n_legs_chaining_is_equivalent_to_manual_subdivision(gate_stack):
    """``n_legs=k`` must equal k chained ``n_legs=1`` calls over the same
    interval -- the parameter is plumbing, not a different integration."""
    from perovskite_sim.discretization.grid import Layer as GL, multilayer_grid
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.solver.mol import build_material_arrays
    from perovskite_sim.solver.newton import solve_equilibrium

    x = multilayer_grid([
        GL(l.thickness, n) for l, n in
        zip(electrical_layers(gate_stack), JV._layer_node_counts(gate_stack, 20))
    ])
    mat = build_material_arrays(x, gate_stack)
    y0 = solve_equilibrium(x, gate_stack)
    t0, t1 = 0.0, 1e-4

    chained = JV._integrate_step(
        x, y0, gate_stack, mat, 0.4, t0, t1, 1e-4, 1e-6, n_legs=4,
    )
    manual = y0
    edges = np.linspace(t0, t1, 5)
    for a, b in zip(edges[:-1], edges[1:]):
        manual = JV._integrate_step(
            x, manual, gate_stack, mat, 0.4, float(a), float(b), 1e-4, 1e-6,
        )
    np.testing.assert_array_equal(chained, manual)

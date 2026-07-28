"""Phase E1.9 — ``run_jv_sweep`` auto-extends V_max when bracket fails.

Sprint 1a (2026-05-25) revealed that with Robin contacts enabled and low
ETL donor doping (N_D ≤ 1e12 cm⁻³), V_oc can climb above the default
V_max=1.6 V and the sweep reports ``voc_bracketed=False`` with sentinel
zero V_oc/FF/PCE. The user has no signal beyond the boolean flag, and
no automatic recovery.

E1.9 adds an opt-in ``v_max_max_attempts`` kwarg: when the first sweep
fails to bracket, retry with V_max bumped by 0.5 V per attempt (capped
at V_initial + 2.0 V). Conservative cost (1-2 extra sweeps),
predictable upper bound. Default value 1 preserves the legacy
no-retry behaviour bit-identically.

Contract:
1. ``v_max_max_attempts=1`` (default) → no retry; failed bracket
   produces sentinel zeros (current behaviour).
2. ``v_max_max_attempts=N`` (N > 1) → on failed bracket, retry up to
   N-1 times with V_max += 0.5 per attempt.
3. Successful first attempt → no retry regardless of attempts setting.
4. Exhausted attempts → still returns ``voc_bracketed=False`` (no
   exception); caller can inspect.
5. Legacy bit-identity: callers that pass default attempts get identical
   numerical results to pre-E1.9.
"""
from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml


def _scaps_mirror_robin_low_etl():
    """scaps_mirror.yaml with Robin S_n_right=1e5 and low ETL doping
    (N_D = 1e12 cm⁻³ = 1e18 m⁻³) → V_oc ≈ 1.43 V, above the default
    V_max on the low rungs of the ladders below.

    This is also the device the collapsed-current V_oc defect was
    diagnosed on; see tests/unit/experiments/test_voc_collapsed_current.py."""
    base = load_scaps_yaml("configs/scaps_mirror.yaml")
    robin = dataclasses.replace(
        base, mode="full",
        S_n_right=1.0e5, S_p_right=1.0e-4,
        S_p_left=1.0e5, S_n_left=1.0e-4,
    )
    # Replace ETL params with N_D=1e12 cm⁻³ = 1e18 m⁻³.
    layers = list(robin.layers)
    etl = layers[-1]
    etl_params = dataclasses.replace(etl.params, N_D=1.0e18)
    layers[-1] = dataclasses.replace(etl, params=etl_params)
    return dataclasses.replace(robin, layers=tuple(layers))


# ---------------------------------------------------------------------------
# V_max ladders — re-anchored 2026-07-27, previously skipped.
#
# These three tests used to be skipped because the fixture was not an oracle:
# the ladders were anchored to a Sprint-1a probe reading "V_oc ~ 1.64 V",
# which is ABOVE the fixture's own Eg = 1.53 eV and therefore
# thermodynamically impossible.  The cause was in `compute_metrics`, not in
# the fixture: past flat band (V_bi = 1.300 V) this device's photocurrent
# collapses to ~1.5e-3 A/m2 (7e-6 of J_sc), where the SIGN of J is numerical
# noise, and the old extractor waited for a sign CHANGE -- so the same device
# reported "unbracketed", 1.44 V or 2.41 V depending only on V_max and the
# sampling, and WHICH of these tests failed changed between a full-suite run
# and a single-file run.  (Not ion history: D_ion = 0 in every layer here.)
#
# `compute_metrics` now takes V_oc where the current REACHES the resolution
# floor and refuses any V_oc above Eg/q -- see
# tests/unit/experiments/test_voc_collapsed_current.py for the full diagnosis
# and the guard's contract.  V_oc on this fixture is now single-valued and
# V_max-independent, so the ladders below are anchored to it:
#
#   measured 2026-07-27, N_grid=30, n_points=20, v_rate=5.0
#     V_max = 0.5  -> no bracket (sentinel zeros)
#     V_max = 1.0  -> no bracket (sentinel zeros)
#     V_max = 1.2  -> no bracket (sentinel zeros)
#     V_max = 1.7  -> V_oc = 1.4316 V, FF 0.789, bracketed
#
# The ladders are built so the FAILING rungs sit below 1.43 V and the passing
# rung above it.  The bounds asserted are structural, not fitted: a value that
# bracketed at V_max = 1.7 but not at 1.2 must lie in (1.2, 1.7], and any
# V_oc must lie at or below Eg/q = 1.53 V.
# ---------------------------------------------------------------------------
_EG_ABSORBER_EV = 1.53   # SCAPS_PVK band gap => V_oc ceiling for this stack


def test_default_attempts_preserves_legacy_no_retry_behaviour():
    """``v_max_max_attempts=1`` (default) on a stack that fails to bracket
    at V_max=1.2 returns sentinel zeros — bit-identical to pre-E1.9."""
    stack = _scaps_mirror_robin_low_etl()
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.2)
    assert r.metrics_fwd.voc_bracketed is False
    assert r.metrics_fwd.V_oc == 0.0  # sentinel
    # J_sc stays meaningful even when the bracket fails.
    assert r.metrics_fwd.J_sc > 0.0


def test_auto_extend_v_max_succeeds_on_second_attempt():
    """``v_max_max_attempts=2`` retries once with V_max=1.7 → succeeds.

    V_oc on this stack is ~1.43 V: V_max=1.2 fails, V_max=1.2 + 0.5 = 1.7
    succeeds."""
    stack = _scaps_mirror_robin_low_etl()
    r = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.2, v_max_max_attempts=2,
    )
    assert r.metrics_fwd.voc_bracketed is True
    # Structural window: above the rung that failed, at or below both the
    # rung that succeeded and the band gap.
    assert 1.2 < r.metrics_fwd.V_oc <= min(1.7, _EG_ABSORBER_EV), (
        f"V_oc={r.metrics_fwd.V_oc:.4f} V outside the (1.2, 1.53] window "
        "implied by the ladder that produced it"
    )
    # The retry actually happened — the returned curve runs to the bumped
    # V_max, not the initial one.
    assert r.V_fwd[-1] == pytest.approx(1.7)


def test_already_bracketed_first_attempt_does_not_retry():
    """When the first sweep already brackets V_oc, ``v_max_max_attempts``
    has no effect — same result whether attempts=1 or attempts=5."""
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r1 = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    r2 = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.6, v_max_max_attempts=5,
    )
    assert r1.metrics_fwd.voc_bracketed is True
    assert r2.metrics_fwd.voc_bracketed is True
    # Identical V_oc — no retry happened so the second call's sweep is
    # bit-identical to the first.
    assert r1.metrics_fwd.V_oc == pytest.approx(r2.metrics_fwd.V_oc)


def test_exhausted_attempts_returns_unbracketed_no_exception():
    """When all attempts exhaust without bracketing, return
    ``voc_bracketed=False`` without raising — caller inspects the flag,
    never an exception."""
    stack = _scaps_mirror_robin_low_etl()
    # Force a too-low V_max + too-few attempts. This stack's V_oc is
    # ~1.43 V. Start at V_max=0.5 with 1 retry → tries 0.5 and 1.0, both
    # below V_oc → unbracketed result expected (no exception).
    r = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=0.5, v_max_max_attempts=2,
    )
    assert r.metrics_fwd.voc_bracketed is False
    assert r.metrics_fwd.V_oc == 0.0
    assert r.V_fwd[-1] == pytest.approx(1.0)   # the ladder did climb


# ---------------------------------------------------------------------------
# The ladder's own defects (2026-07-29). Both concern the RETRY path only —
# a first attempt that brackets is untouched, and so is attempts=1.
# ---------------------------------------------------------------------------


def test_retry_preserves_the_first_attempt_voltage_resolution():
    """A retry must not answer at a coarser voltage step than the attempt
    that failed.

    ``n_points`` used to be passed through unchanged while ``V_max`` grew, so
    the step dV = V_max/(n_points-1) got LARGER on every rung: here 1.2/19 =
    0.063 V on the attempt that failed, 1.7/19 = 0.089 V on the one that
    answered. V_oc is interpolated between the two samples that bracket it, so
    its error scales with that step (Stage 2a measured ~10 mV between a coarse
    and a fine grid on this class of stack) — the ladder was trading away the
    precision of the answer it exists to produce. Worse, the degradation is
    systematic in V_max, so on a design sweep it lands preferentially on
    whichever points needed the retry.
    """
    stack = _scaps_mirror_robin_low_etl()
    r = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.2, v_max_max_attempts=2,
    )
    assert r.metrics_fwd.voc_bracketed is True
    assert r.V_fwd[-1] == pytest.approx(1.7)      # the retry did happen

    dV_first_attempt = 1.2 / (20 - 1)
    steps = [b - a for a, b in zip(r.V_fwd[:-1], r.V_fwd[1:])]
    assert max(steps) == pytest.approx(dV_first_attempt, rel=0.05), (
        f"retry ran at dV={max(steps):.4f} V; the attempt that failed used "
        f"{dV_first_attempt:.4f} V. The ladder must hold the resolution and "
        f"grow n_points with V_max."
    )


def test_ladder_stops_once_v_max_reaches_its_cap(monkeypatch):
    """Exhausting the budget must not re-run the same capped sweep.

    The bail-out compared ``V_max_attempt`` (a float) with ``last_result``
    (a ``JVResult``), which is never equal, so the branch was dead: once
    V_max saturated at ``V_initial + 2.0`` every remaining attempt repeated
    that identical sweep. With v_rate low and a stiff stack each of those is
    minutes of wasted work for a result already known.

    The loop's control flow is what is under test, so ``compute_metrics`` is
    forced to report "not bracketed" rather than hunting for a stack whose
    V_oc outruns the cap — that would make the test a physics fixture again,
    which is exactly what this file's header says went wrong before.
    """
    from perovskite_sim.experiments import jv_sweep as jv

    real_compute_metrics = jv.compute_metrics

    def never_brackets(*args, **kwargs):
        m = real_compute_metrics(*args, **kwargs)
        return dataclasses.replace(m, voc_bracketed=False, V_oc=0.0)

    monkeypatch.setattr(jv, "compute_metrics", never_brackets)

    sweeps = []

    def count_sweeps(stage, current, total, message):
        if stage == "jv_init":
            sweeps.append(total)

    stack = _scaps_mirror_robin_low_etl()
    jv.run_jv_sweep(
        stack, N_grid=15, n_points=5, v_rate=5.0,
        V_max=1.0, v_max_max_attempts=10, progress=count_sweeps,
    )

    # V_max climbs 1.0 -> 1.5 -> 2.0 -> 2.5 -> 3.0 (cap = 1.0 + 2.0) and then
    # stops: a sixth rung would repeat the 3.0 sweep verbatim.
    assert len(sweeps) == 5, (
        f"ran {len(sweeps)} sweeps for a 5-rung ladder — the cap bail-out "
        "did not fire"
    )

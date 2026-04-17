"""Tests for `experiments/mott_schottky.py` — C-V + depletion-doping fit.

Two layers of coverage:

1. Pure-math unit tests on synthetic Mott-Schottky data (no solver) —
   these tightly bound the V_bi and N_eff the fitter must recover when
   given a curve that is guaranteed-linear in 1/C² vs V by construction.
2. An integration smoke test that runs the dark C-V on
   cSi_homojunction and verifies the pipeline wires together — that is,
   a populated MottSchottkyResult with finite C, 1/C², V_bi_fit, and
   N_eff_fit is produced end-to-end.

No numerical bound is placed on the integration-path V_bi_fit / N_eff
because simulated junctions in this repo are fully or nearly-fully
depleted at zero/reverse bias (the cSi preset's 180 µm base is deep
enough to exceed the classical depletion width at low doping), so
enforcing a textbook MS ballpark there would measure the preset, not
the fitter. Numerical fit fidelity is covered by the synthetic tests
above; the integration test here exists to ensure the run_impedance ->
C(V) -> fit pipeline survives refactors.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments.mott_schottky import (
    EPS_0,
    MottSchottkyResult,
    _fit_mott_schottky,
    _resolve_eps_r,
    _select_ms_window,
    run_mott_schottky,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


# ---------------------------------------------------------------------------
# Pure-math unit tests — synthetic Mott-Schottky data, no solver.
# ---------------------------------------------------------------------------

def _synthetic_cv(V, V_bi, N, eps_r):
    """Build a synthetic C(V) from the Mott-Schottky formula.

    C(V) = sqrt(q·ε·ε_0·N / (2·(V_bi − V))). Only valid for V < V_bi.
    """
    return np.sqrt(Q * eps_r * EPS_0 * N / (2.0 * (V_bi - V)))


def test_fit_recovers_known_vbi_and_doping():
    """Round-trip: synthetic C(V) → V_bi_fit and N_eff_fit."""
    V = np.linspace(-0.3, 0.4, 20)
    V_bi_true = 0.9
    N_true = 1e22  # m⁻³
    eps_r = 11.7
    C = _synthetic_cv(V, V_bi_true, N_true, eps_r)

    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(V, C, eps_r)
    assert abs(V_bi_fit - V_bi_true) < 0.01, (
        f"V_bi_fit={V_bi_fit:.4f} off from {V_bi_true}"
    )
    assert abs(np.log10(N_fit) - np.log10(N_true)) < 0.02, (
        f"N_eff_fit={N_fit:.3e} off from {N_true:.3e}"
    )
    assert V_lo < V_hi


def test_fit_rejects_non_linear_tail():
    """A strongly-curved tail must not pull V_bi_fit or N_eff_fit off.

    Build a pure-MS curve for V in [-0.2, 0.5] V and prepend 3 points on
    a curved branch (``1/C²`` 9x higher than MS would predict at those
    V — simulates a freeze-out / fully-depleted tail that bends sharply
    upward). The window selector should land on the linear segment so
    the extracted (V_bi, N_eff) are close to the true MS values.
    """
    V_good = np.linspace(-0.2, 0.5, 15)
    V_bad = np.array([-0.6, -0.5, -0.4])
    V = np.concatenate([V_bad, V_good])

    V_bi, N, eps_r = 1.0, 1e22, 11.7
    C_good = _synthetic_cv(V_good, V_bi, N, eps_r)
    # Bad tail: C suppressed by 3x → 1/C² inflated by 9x relative to the
    # MS prediction at these biases. A line through the bad+good data
    # has much worse RMS than one through the good segment alone.
    C_bad = _synthetic_cv(V_bad, V_bi, N, eps_r) / 3.0
    C = np.concatenate([C_bad, C_good])

    V_bi_fit, N_fit, V_lo, V_hi = _fit_mott_schottky(V, C, eps_r)
    assert abs(V_bi_fit - V_bi) < 0.03, (
        f"V_bi_fit={V_bi_fit:.3f} off from {V_bi:.3f} — tail leaked "
        "into window"
    )
    assert abs(np.log10(N_fit) - np.log10(N)) < 0.05, (
        f"N_eff_fit={N_fit:.3e} off from {N:.3e} — tail leaked into "
        "window"
    )
    # And the window must sit strictly past the bad tail.
    assert V_lo >= V_good[0] - 1e-9, (
        f"fit window V_lo={V_lo:.3f} started inside bad tail"
    )


def test_resolve_eps_r_picks_absorber_layer():
    """With an explicit 'absorber' role, that layer's ε_r must win."""
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    eps_r = _resolve_eps_r(stack)
    # c-Si homojunction: absorber is p_base with eps_r=11.7.
    assert eps_r == pytest.approx(11.7, rel=1e-3)


def test_fit_returns_nan_on_flat_curve():
    """Flat 1/C² (no information) must return NaN rather than blow up."""
    V = np.linspace(0.0, 0.3, 8)
    C = np.full_like(V, 1e-4)  # constant
    V_bi_fit, N_fit, *_ = _fit_mott_schottky(V, C, eps_r=11.7)
    assert not np.isfinite(V_bi_fit) or abs(V_bi_fit) > 1e6
    # Either NaN or absurdly large — both are acceptable "this fit is
    # meaningless" signals. An absurd V_bi also fails downstream
    # sanity checks.


# ---------------------------------------------------------------------------
# Integration: dark C-V on cSi_homojunction.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csi_stack():
    return load_device_from_yaml("configs/cSi_homojunction.yaml")


@pytest.fixture(scope="module")
def csi_cv_result(csi_stack):
    """3-point reverse-bias C-V — cheapest run that still admits a fit."""
    return run_mott_schottky(
        csi_stack,
        V_range=np.linspace(-0.2, 0.2, 4),
        frequency=1e5,
        N_grid=30,
        n_cycles=3, n_extract=1,
    )


def test_cv_returns_populated_dataclass(csi_cv_result):
    r = csi_cv_result
    assert isinstance(r, MottSchottkyResult)
    assert r.V.shape == r.C.shape == r.one_over_C2.shape
    assert np.all(r.C > 0.0), f"non-positive capacitance detected: {r.C}"
    assert np.all(np.isfinite(r.one_over_C2))
    assert r.frequency == pytest.approx(1e5)
    # 1/C² should be consistent with C
    np.testing.assert_allclose(r.one_over_C2, 1.0 / (r.C * r.C), rtol=1e-12)


def test_cv_pipeline_returns_finite_fit(csi_cv_result):
    """run_mott_schottky -> MottSchottkyResult must yield finite fit values.

    A full end-to-end smoke: run_impedance on each bias, build C(V),
    then feed that into the linear-fit helper. We do *not* bound V_bi
    or N_eff numerically here — the cSi preset's 180 µm lightly-doped
    (N_A = 1e22 m⁻³, ε_r = 11.7) p-base is near-fully depleted at low
    reverse bias, so the simulated C(V) deviates from the textbook
    1/sqrt(V_bi − V) form. The fitter's numerical accuracy is pinned
    down by the synthetic-data tests above; this test guards the
    pipeline wiring (signature, shapes, dataclass fields populated
    with finite numbers) against silent breakage.
    """
    r = csi_cv_result
    assert np.isfinite(r.V_bi_fit), (
        f"V_bi_fit={r.V_bi_fit!r} non-finite — pipeline produced a "
        "degenerate C(V) or the fit blew up"
    )
    assert np.isfinite(r.N_eff_fit), (
        f"N_eff_fit={r.N_eff_fit!r} non-finite"
    )
    # Fit window must be a real sub-range of the input sweep.
    assert r.V_fit_lo <= r.V_fit_hi
    assert r.V[0] - 1e-9 <= r.V_fit_lo <= r.V_fit_hi <= r.V[-1] + 1e-9


def test_rejects_sparse_v_range(csi_stack):
    """Need at least 3 V points for a meaningful fit."""
    with pytest.raises(ValueError, match="at least 3"):
        run_mott_schottky(
            csi_stack, V_range=[0.0, 0.1], N_grid=30, n_cycles=3, n_extract=1,
        )


def test_rejects_nonpositive_frequency(csi_stack):
    with pytest.raises(ValueError, match="frequency"):
        run_mott_schottky(
            csi_stack, V_range=[-0.1, 0.0, 0.1], frequency=0.0,
            N_grid=30, n_cycles=3, n_extract=1,
        )

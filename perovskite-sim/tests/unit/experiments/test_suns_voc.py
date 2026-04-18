"""Tests for `experiments/suns_voc.py` — Suns-V_oc + pseudo J-V extraction.

Key physical invariants being checked:

1. Dataclass is populated and arrays have matching shapes.
2. V_oc(suns) is monotone increasing — more light, higher V_oc.
3. V_oc vs ln(suns) is log-linear with slope n·V_T, where n is the
   effective ideality from recombination. For SRH-dominated perovskites
   n is in [1, 2]; the Suns-Voc slope should land in the same band.
4. J_sc(suns) scales near-linearly with suns (no extreme saturation).
5. Pseudo-FF is a physical value in (0, 1) when the 1-sun reference is
   included in the sweep.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.experiments.suns_voc import (
    SunsVocResult,
    run_suns_voc,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def nip_stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------

def test_run_suns_voc_rejects_empty_levels(nip_stack):
    with pytest.raises(ValueError, match="non-empty"):
        run_suns_voc(nip_stack, suns_levels=(), N_grid=30)


def test_run_suns_voc_rejects_nonpositive_levels(nip_stack):
    with pytest.raises(ValueError, match="positive"):
        run_suns_voc(nip_stack, suns_levels=(0.0, 1.0), N_grid=30)


# ---------------------------------------------------------------------------
# Structural invariants — short sweep on nip_MAPbI3.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_sweep(nip_stack):
    """Three-point sweep reused by structural tests to keep runtime low."""
    return run_suns_voc(
        nip_stack,
        suns_levels=(0.1, 1.0, 5.0),
        N_grid=30,
        t_settle=5e-4,
    )


def test_result_is_populated_dataclass(short_sweep):
    r = short_sweep
    assert isinstance(r, SunsVocResult)
    # Arrays sorted ascending in suns and same shape across fields.
    assert r.suns.ndim == 1
    assert r.V_oc.shape == r.suns.shape
    assert r.J_sc.shape == r.suns.shape
    assert r.J_pseudo_V.shape == r.suns.shape
    assert r.J_pseudo_J.shape == r.suns.shape
    assert np.all(np.diff(r.suns) > 0), "suns should be sorted ascending"
    assert np.all(np.isfinite(r.V_oc))
    assert np.all(np.isfinite(r.J_sc))


def test_voc_monotone_in_suns(short_sweep):
    """More light → higher V_oc. Diode V_oc rises as log(X)."""
    r = short_sweep
    dV = np.diff(r.V_oc)
    assert np.all(dV > 0), (
        f"V_oc not monotone in suns: V_oc={r.V_oc}, suns={r.suns}"
    )


def test_voc_in_physical_range(short_sweep):
    """V_oc should sit between 0 and V_bi across the whole sweep."""
    r = short_sweep
    assert np.all(r.V_oc > 0.0)
    # 1 V is a loose upper bound for MAPbI3 at <=5 suns.
    assert np.all(r.V_oc < 1.2)


def test_jsc_scales_with_suns(short_sweep):
    """J_sc should be near-linear in suns for low recombination.

    Within a decade span (0.1x → 1x → 5x suns) the ratio J_sc/suns
    should stay within a factor of ~2 of the 1-sun value. Larger
    deviations flag a broken generation-scaling path.
    """
    r = short_sweep
    idx1 = int(np.argmin(np.abs(r.suns - 1.0)))
    J_per_sun_1 = r.J_sc[idx1] / r.suns[idx1]
    for k in range(len(r.suns)):
        ratio = (r.J_sc[k] / r.suns[k]) / J_per_sun_1
        assert 0.5 < ratio < 2.0, (
            f"J_sc/suns at X={r.suns[k]:.3g} is {ratio:.2f}× the 1-sun "
            f"value — generation scaling looks wrong"
        )


def test_pseudo_jv_sorted_by_voltage(short_sweep):
    """Pseudo-JV (V, J) pairs must be sorted ascending by V."""
    r = short_sweep
    assert np.all(np.diff(r.J_pseudo_V) > 0), (
        f"pseudo-JV V not sorted: {r.J_pseudo_V}"
    )


def test_pseudo_ff_physical(short_sweep):
    """Pseudo-FF must be in (0, 1) when the 1-sun reference is present."""
    r = short_sweep
    assert np.isfinite(r.pseudo_FF)
    assert 0.0 < r.pseudo_FF < 1.0, (
        f"pseudo_FF={r.pseudo_FF:.3f} outside (0, 1)"
    )


# ---------------------------------------------------------------------------
# Physics consistency: Suns-Voc slope gives the diode ideality.
# ---------------------------------------------------------------------------

def test_voc_vs_ln_suns_gives_physical_ideality(nip_stack):
    """Slope of V_oc vs ln(suns) = n_eff · V_T. Must land in [1.0, 2.5].

    This is the textbook Suns-Voc ideality: under open circuit the net
    recombination current equals the generation current X·J_gen, and
    V_oc(X) = n·V_T·ln(X·J_gen/J_0). The log slope therefore isolates
    the recombination ideality free of series resistance. For
    nip_MAPbI3 (SRH-dominated) we expect n_eff in the ~1–2 band,
    consistent with the dark J-V ideality fit.
    """
    # Use 4 well-spaced suns levels to stabilise the log-slope fit.
    r = run_suns_voc(
        nip_stack,
        suns_levels=(0.1, 1.0, 3.0, 10.0),
        N_grid=30,
        t_settle=5e-4,
    )
    ln_X = np.log(r.suns)
    slope, _intercept = np.polyfit(ln_X, r.V_oc, 1)
    n_eff = slope / V_T
    assert 1.0 <= n_eff <= 2.5, (
        f"Suns-Voc ideality n_eff={n_eff:.3f} outside physical [1.0, 2.5] "
        f"(slope={slope:.4f} V/decade→{slope * np.log(10):.4f} V/decade_e, "
        f"V_oc={r.V_oc}, suns={r.suns})"
    )


def test_pseudo_jv_anchors_at_one_sun_voc(nip_stack):
    """At X=1, pseudo-JV anchors at (V_oc_1sun, 0).

    Standard Sinton convention: J_pseudo(V_oc(X)) = J_sc_ref − J_sc(X),
    so the 1-sun point sits exactly on the V-axis at V = V_oc_1sun.
    Points with X < 1 lift into the power quadrant; X > 1 drops below.
    """
    r = run_suns_voc(
        nip_stack,
        suns_levels=(0.5, 1.0, 2.0),
        N_grid=30,
        t_settle=5e-4,
    )
    idx1 = int(np.argmin(np.abs(r.suns - 1.0)))
    assert r.J_pseudo_V[idx1] == pytest.approx(r.V_oc[idx1])
    assert r.J_pseudo_J[idx1] == pytest.approx(0.0, abs=1e-9)
    # X < 1 → power-quadrant (J > 0, V < V_oc_ref)
    idx_low = int(np.argmin(np.abs(r.suns - 0.5)))
    assert r.J_pseudo_J[idx_low] > 0.0
    assert r.J_pseudo_V[idx_low] < r.V_oc[idx1]
    # X > 1 → injection quadrant (J < 0, V > V_oc_ref)
    idx_high = int(np.argmin(np.abs(r.suns - 2.0)))
    assert r.J_pseudo_J[idx_high] < 0.0
    assert r.J_pseudo_V[idx_high] > r.V_oc[idx1]


def test_progress_callback_invoked(nip_stack):
    """Progress callback fires exactly once per suns level."""
    events: list[tuple[str, int, int, str]] = []

    def cb(stage, cur, total, msg):
        events.append((stage, cur, total, msg))

    levels = (0.5, 1.0, 2.0)
    run_suns_voc(
        nip_stack, suns_levels=levels, N_grid=30,
        t_settle=5e-4, progress=cb,
    )
    assert len(events) == len(levels)
    assert all(ev[0] == "suns_voc" for ev in events)
    assert [ev[1] for ev in events] == [1, 2, 3]
    assert all(ev[2] == 3 for ev in events)

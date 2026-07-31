"""Lin 2019 Nature Energy tandem benchmark regression test.

Runs the full combined-TMM + series-matched J-V pipeline on the Lin 2019
all-perovskite tandem preset and asserts each figure of merit lies within
the ±tolerance_pct window declared in the config.

The optics use real Dasgupta-repository n,k data (FA-Cs-Pb-BrI 40 % Br for
the 1.77 eV top, MA-FA-Cs-Pb-Sn-I 50 % Sn for the 1.22 eV bottom), an
Ag back reflector for the second-pass IR, a manual V_bi = 1.42 V on the
wide-gap top sub-cell to match Lin's per-junction V_oc ~ 1.18 V, and
absorber thicknesses tuned for current matching.

STATUS (2026-07, after the tandem generation-grid fix)
-----------------------------------------------------
V_oc and FF still land inside the ±10 % window; J_sc and PCE no longer do,
and that is a RECOVERY rather than a regression.

The optics used to sample G on a uniform grid and hand it to a solver
integrating on the tanh-clustered one — matching shapes, mismatched positions
(105 nm of error on the 500 nm top cell, 409 nm on the 1300 nm bottom cell).
Measured consequence: the top sub-cell collected **176.315 A/m² against the
142.033 A/m² of photons actually absorbed in it — 124 % of its own photon
budget**, which is thermodynamically impossible.  The old agreement with Lin
2019 was propped up by that surplus, exactly as the 1D generation fix
(7807985) turned out to have been.

Post-fix figures, N_grid=40 / n_points=25::

                  before     after    target
    V_oc  [V]     2.0662    2.0530    1.9650   ok  (+4.5 %)
    FF            0.7605    0.8161    0.7900   ok  (+3.3 %)
    J_sc  [A/m²] 147.384   127.179   156.000   GAP (-18.5 %)
    PCE   [%]     23.160    21.309    24.800   GAP (-14.1 %)
    top J_sc     176.315   127.179            <- was 124 % of its budget
    bot J_sc     147.384   183.124

Collection efficiencies are now physical (top 89.5 %, bottom 99.7 %); the
bottom's pre-fix 70.1 % was an artifact of generation deposited at the wrong
depth, which is also why its J_sc ROSE while its total generation fell.  The
current-limiting sub-cell moved from bottom to top, i.e. the pre-fix tandem
was series-matched at the wrong operating point.

The residual J_sc gap has two identified components, neither closed here:
  1. the incoherent branch drops the multi-pass geometric series (measured up
     to 3.9 % with a metal back contact — and this preset ships an Ag back
     reflector specifically for the second-pass IR);
  2. the absorber thicknesses in the preset were tuned for current matching
     AGAINST the mis-positioned generation, so they no longer current-match
     now that G lands at the right depth.
Closing either is a separate change; widening the tolerance is not, because it
would re-manufacture the agreement this fix removed.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.data import load_am15g
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.models.tandem_config import load_tandem_from_yaml


@pytest.fixture(scope="module")
def lin2019():
    """One tandem solve, shared by every figure-of-merit assertion."""
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")
    assert cfg.benchmark is not None, "Lin 2019 preset must declare a benchmark block"

    # Span 300-1100 nm so the Sn-Pb 1.22 eV (~1016 nm) absorption tail is
    # captured. A 1000 nm cap clips ~3 mA/cm² off bottom J_sc.
    wavelengths_nm = np.linspace(300.0, 1100.0, 220)
    _, spectral_flux = load_am15g(wavelengths_nm)

    result = run_tandem_jv(
        cfg,
        wavelengths_m=wavelengths_nm * 1e-9,
        spectral_flux=spectral_flux,
        wavelengths_nm=wavelengths_nm,
        N_grid=40,
        n_points=25,
    )
    return cfg, result


@pytest.mark.slow
def test_lin2019_voc_and_ff_within_tolerance(lin2019):
    """The two figures of merit that survived the generation-grid fix.

    Kept as a live gate rather than folded into the xfail below: V_oc and FF
    still agree with Lin 2019 to +4.5 % and +3.3 %, and xfailing the whole
    benchmark would retire two working guards along with the two broken ones.
    """
    cfg, result = lin2019
    m = result.metrics
    tol = float(cfg.benchmark["tolerance_pct"]) / 100.0

    assert m.V_oc == pytest.approx(float(cfg.benchmark["target_voc_v"]), rel=tol), (
        f"V_oc {m.V_oc:.3f} V outside tolerance of "
        f"{float(cfg.benchmark['target_voc_v']):.3f}"
    )
    assert m.FF == pytest.approx(float(cfg.benchmark["target_ff"]), rel=tol), (
        f"FF {m.FF:.3f} outside tolerance of {float(cfg.benchmark['target_ff']):.3f}"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN OPTICS GAP, opened by the tandem generation-grid fix and left "
        "open deliberately. Pre-fix agreement was manufactured: the top "
        "sub-cell collected 176.315 A/m^2 against 142.033 A/m^2 of absorbed "
        "photons (124% of its own budget). With generation now landing at the "
        "correct depth, J_sc reads 127.179 vs target 156.000 (-18.5%) and PCE "
        "21.309% vs 24.800% (-14.1%). Two identified causes, neither closed: "
        "(1) the incoherent branch drops the multi-pass geometric series, "
        "measured up to 3.9% with a metal back contact and this preset ships "
        "an Ag back reflector; (2) the preset's absorber thicknesses were "
        "tuned for current matching against the mis-positioned generation. "
        "strict=True on purpose: when either is fixed this XPASSes and fails, "
        "forcing this marker to be revisited rather than silently outliving "
        "the gap. Do NOT widen tolerance_pct to make it pass -- that "
        "re-manufactures the agreement the fix removed."
    ),
)
def test_lin2019_jsc_and_pce_within_tolerance(lin2019):
    """Blocked on the residual optics gap -- see the module docstring."""
    cfg, result = lin2019
    m = result.metrics
    tol = float(cfg.benchmark["tolerance_pct"]) / 100.0

    # PCE in cfg.benchmark is in % (Lin 2019 reports 24.8). compute_metrics
    # returns it as a dimensionless fraction (P_mpp / 1000 W/m²), so divide.
    target_pce = float(cfg.benchmark["target_pce"]) / 100.0
    target_jsc = float(cfg.benchmark["target_jsc_ma_cm2"]) * 10.0  # mA/cm² -> A/m²

    assert m.PCE == pytest.approx(target_pce, rel=tol), (
        f"PCE {m.PCE * 100:.2f}% outside ±{tol * 100:.0f}% of target "
        f"{target_pce * 100:.2f}%"
    )
    assert abs(m.J_sc) == pytest.approx(target_jsc, rel=tol), (
        f"|J_sc| {abs(m.J_sc):.2f} A/m² outside tolerance of {target_jsc:.2f}"
    )


@pytest.mark.slow
def test_lin2019_subcells_respect_their_photon_budgets(lin2019):
    """No sub-cell may collect more current than photons absorbed in it.

    This is the invariant the pre-fix pipeline violated (top sub-cell at 124 %
    of its budget) and the reason the Lin 2019 agreement looked better than it
    was. Anchored on the optics, not on the solve, so it cannot be satisfied by
    a solver artifact -- the same external-anchor argument as
    tests/regression/test_physical_bounds.py.
    """
    from perovskite_sim.constants import Q
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.physics.generation import dual_cell_widths
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation

    cfg, result = lin2019
    wavelengths_nm = np.linspace(300.0, 1100.0, 220)
    _, spectral_flux = load_am15g(wavelengths_nm)
    x_top = build_electrical_grid(cfg.top_cell, 40)
    x_bot = build_electrical_grid(cfg.bottom_cell, 40)
    gen = compute_tandem_generation(
        cfg, wavelengths_nm * 1e-9, spectral_flux, wavelengths_nm,
        x_top=x_top, x_bot=x_bot,
    )

    for label, G, x, sub in (
        ("top", gen.G_top, x_top, result.top_result),
        ("bottom", gen.G_bot, x_bot, result.bot_result),
    ):
        budget = Q * float(np.sum(np.asarray(G) * dual_cell_widths(x)))
        j_sc = abs(sub.metrics_fwd.J_sc)
        assert j_sc <= budget, (
            f"{label} sub-cell collected {j_sc:.3f} A/m² against an absorbed-"
            f"photon budget of {budget:.3f} A/m² ({100.0 * j_sc / budget:.1f} % "
            "of it). A sub-cell cannot collect more carriers than photons "
            "absorbed to make them."
        )

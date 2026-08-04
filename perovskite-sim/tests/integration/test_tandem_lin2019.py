"""Lin 2019 Nature Energy tandem benchmark regression test.

Runs the full combined-TMM + series-matched J-V pipeline on the Lin 2019
all-perovskite tandem preset and asserts each figure of merit lies within
the ±tolerance_pct window declared in the config.

The absorber optical constants are rigid-shift placeholders derived from a
MAPbI3 dataset, and the simulated layer sequence is not yet a faithful
transcription of the Lin device.  The preset also uses a manual V_bi = 1.42 V
on the wide-gap top sub-cell.  This is therefore a partial external comparison,
not a predictive reproduction of the paper device.

STATUS (2026-08, paper-thickness provenance restored)
-----------------------------------------------------
Lin 2019 reports approximately 300 nm / 800 nm wide-/narrow-gap absorbers;
the preset had instead used fitted 200 nm / 1000 nm values.  Restoring the
reported thicknesses raises the top-cell photon budget enough that all four
tandem metrics again lie inside the declared ±10 % paper window without
violating either sub-cell's absorbed-photon budget.

The optics used to sample G on a uniform grid and hand it to a solver
integrating on the tanh-clustered one — matching shapes, mismatched positions
(105 nm of error on the 500 nm top cell, 409 nm on the 1300 nm bottom cell).
Measured consequence: the top sub-cell collected **176.315 A/m² against the
142.033 A/m² of photons actually absorbed in it — 124 % of its own photon
budget**, which is thermodynamically impossible.  The old agreement with Lin
2019 was propped up by that surplus, exactly as the 1D generation fix
(7807985) turned out to have been.

Current figures, N_grid=40 / n_points=25::

    V_oc  [V]       2.03834   target 1.9650   ok (+3.7 %)
    FF              0.81459   target 0.8100   ok (+0.6 %)
    J_sc  [A/m²]   144.564    target 156.000  ok (-7.3 %)
    PCE   [%]       24.0035    target 24.800   ok (-3.2 %)
    top J_sc        144.564    budget 159.337 (90.7 % collection)
    bottom J_sc     168.222    budget 168.659 (99.7 % collection)

Collection efficiencies remain physical.  The remaining mismatch is now
stated directly: top and bottom short-circuit currents differ by about 15.1 %,
well outside the 2 % current-match acceptance threshold.  That failure has its
own strict xfail below; it is no longer conflated with J_sc or PCE.

The remaining current-match gap is input/model provenance, not a licence to
tune transport numerics until the target is met.  The paper stack is
glass/ITO/PTAA/WBG/C60/ALD-SnO2/Au/PEDOT:PSS/NBG/C60/BCP/Cu, while several of
those materials are absent or represented by optical and electrical proxies
here and both absorber n,k files are rigid shifts.  The unequal collection
efficiencies also mean the 15.1 % mismatch cannot be assigned to optics alone.
Those inputs must be replaced and both sub-cell transport certificates checked
before adjusting thickness away from the paper values.  Widening a tolerance
would only conceal the mismatch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.constants import Q
from perovskite_sim.data import load_am15g
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.models.tandem_config import load_tandem_from_yaml
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.tandem_optics import compute_tandem_generation


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


def test_lin2019_absorber_thicknesses_follow_paper():
    """Keep the paper-reported ~300/800 nm thickness provenance executable."""
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")
    top_abs = next(layer for layer in cfg.top_cell.layers if layer.role == "absorber")
    bot_abs = next(layer for layer in cfg.bottom_cell.layers if layer.role == "absorber")
    assert top_abs.thickness == pytest.approx(300e-9)
    assert bot_abs.thickness == pytest.approx(800e-9)


def _lin_observables(cfg, result):
    wavelengths_nm = np.linspace(300.0, 1100.0, 220)
    _, spectral_flux = load_am15g(wavelengths_nm)
    x_top = build_electrical_grid(cfg.top_cell, 40)
    x_bot = build_electrical_grid(cfg.bottom_cell, 40)
    generation = compute_tandem_generation(
        cfg,
        wavelengths_nm * 1e-9,
        spectral_flux,
        wavelengths_nm,
        x_top=x_top,
        x_bot=x_bot,
    )
    top_budget = Q * float(
        np.sum(np.asarray(generation.G_top) * dual_cell_widths(x_top))
    )
    bottom_budget = Q * float(
        np.sum(np.asarray(generation.G_bot) * dual_cell_widths(x_bot))
    )
    top_jsc = abs(result.top_result.metrics_fwd.J_sc)
    bottom_jsc = abs(result.bot_result.metrics_fwd.J_sc)
    mismatch = abs(top_jsc - bottom_jsc) / (0.5 * (top_jsc + bottom_jsc))
    metrics = result.metrics
    return {
        "Voc_V": metrics.V_oc,
        "Jsc_A_m2": abs(metrics.J_sc),
        "FF": metrics.FF,
        "PCE_percent": 100.0 * metrics.PCE,
        "top_photon_budget_A_m2": top_budget,
        "bottom_photon_budget_A_m2": bottom_budget,
        "top_subcell_Jsc_A_m2": top_jsc,
        "bottom_subcell_Jsc_A_m2": bottom_jsc,
        "subcell_current_mismatch_percent": 100.0 * mismatch,
    }


@pytest.mark.slow
def test_lin2019_voc_and_ff_within_tolerance(lin2019):
    """V_oc and FF remain independent live paper-window gates."""
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


_LIN_CURRENT_MATCH_GAP = pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN CURRENT-MATCH GAP. Restoring Lin 2019's approximately 300/800 nm "
        "absorber thicknesses puts Voc, Jsc, FF, and PCE inside their declared "
        "paper windows and preserves both photon budgets, but the top/bottom "
        "short-circuit currents remain 144.564/168.222 A/m^2 (about 15.1% "
        "apart vs the 2% acceptance threshold). The remaining stack and n,k "
        "proxies need traceable optical and electrical inputs before re-tuning. "
        "strict=True makes a future physical current match fail as XPASS until "
        "the evidence label is deliberately reviewed."
    ),
)


def _lin_gap_targets(cfg):
    tol = float(cfg.benchmark["tolerance_pct"]) / 100.0
    # PCE in cfg.benchmark is in %, while compute_metrics returns a fraction.
    target_pce = float(cfg.benchmark["target_pce"]) / 100.0
    target_jsc = float(cfg.benchmark["target_jsc_ma_cm2"]) * 10.0
    return tol, target_pce, target_jsc


@pytest.mark.slow
def test_lin2019_pce_within_tolerance(lin2019):
    """PCE is a live gate after restoring paper-reported thicknesses."""
    cfg, result = lin2019
    m = result.metrics
    tol, target_pce, _ = _lin_gap_targets(cfg)

    assert m.PCE == pytest.approx(target_pce, rel=tol), (
        f"PCE {m.PCE * 100:.2f}% outside ±{tol * 100:.0f}% of target "
        f"{target_pce * 100:.2f}%"
    )


@pytest.mark.slow
def test_lin2019_jsc_within_tolerance(lin2019):
    """J_sc is a live gate after restoring paper-reported thicknesses."""
    cfg, result = lin2019
    m = result.metrics
    tol, _, target_jsc = _lin_gap_targets(cfg)

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
    cfg, result = lin2019
    observed = _lin_observables(cfg, result)

    target_jsc = float(cfg.benchmark["target_jsc_ma_cm2"]) * 10.0
    for label, budget, j_sc in (
        (
            "top",
            observed["top_photon_budget_A_m2"],
            observed["top_subcell_Jsc_A_m2"],
        ),
        (
            "bottom",
            observed["bottom_photon_budget_A_m2"],
            observed["bottom_subcell_Jsc_A_m2"],
        ),
    ):
        assert budget >= target_jsc, (
            f"{label} absorbed-photon budget {budget:.3f} A/m² cannot support "
            f"the paper target {target_jsc:.3f} A/m²"
        )
        assert j_sc <= budget, (
            f"{label} sub-cell collected {j_sc:.3f} A/m² against an absorbed-"
            f"photon budget of {budget:.3f} A/m² ({100.0 * j_sc / budget:.1f} % "
            "of it). A sub-cell cannot collect more carriers than photons "
            "absorbed to make them."
        )


@pytest.mark.slow
def test_lin2019_observations_match_reproducibility_registry(lin2019):
    """Keep exact local observations live without promoting external evidence."""
    cfg, result = lin2019
    matrix = yaml.safe_load(
        Path("reproducibility/config_benchmark_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = matrix["benchmarks"]["lin2019-tandem"]
    observed = contract["observed"]
    tolerance = contract["regression_tolerance"]
    actual = _lin_observables(cfg, result)

    assert set(actual) == set(observed) == set(tolerance)
    for metric, expected in observed.items():
        assert actual[metric] == pytest.approx(
            expected, abs=float(tolerance[metric])
        ), metric


@pytest.mark.slow
@_LIN_CURRENT_MATCH_GAP
def test_lin2019_subcell_currents_match_within_two_percent(lin2019):
    """Keep the remaining series-current mismatch isolated and quantitative."""
    _, result = lin2019
    top_jsc = abs(result.top_result.metrics_fwd.J_sc)
    bot_jsc = abs(result.bot_result.metrics_fwd.J_sc)
    mismatch = abs(top_jsc - bot_jsc) / (0.5 * (top_jsc + bot_jsc))
    assert mismatch <= 0.02, (
        f"sub-cell J_sc mismatch is {100.0 * mismatch:.2f}%: "
        f"top={top_jsc:.3f}, bottom={bot_jsc:.3f} A/m²"
    )

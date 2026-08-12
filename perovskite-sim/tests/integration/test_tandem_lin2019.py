"""Lin 2019 Nature Energy tandem benchmark regression test.

Runs the full combined-TMM + series-matched J-V pipeline on the Lin 2019
all-perovskite tandem preset and asserts each figure of merit lies within
the ±tolerance_pct window declared in the config.

The optical layer sequence and source-reported thicknesses are now transcribed
as glass/ITO/PTAA/WBG/C60/ALD-SnO2/Au/PEDOT:PSS/NBG/C60/BCP/Cu.  The absorber
optical constants remain rigid-shift placeholders derived from MAPbI3, several
transport/contact parameters are priors, and the top sub-cell uses manual
V_bi = 1.42 V.  This is therefore a partial external comparison, not a
predictive reproduction of the paper device.

STATUS (2026-08, source stack and thickness provenance restored)
----------------------------------------------------------------
Lin 2019 reports approximately 300 nm / 800 nm wide-/narrow-gap absorbers,
20 nm C60 layers, 20 nm ALD-SnO2, approximately 1 nm Au, 7 nm BCP, and
100 nm Cu.  The complete layer sequence and those reported thicknesses are
now executable; unreported ITO/PTAA/PEDOT:PSS thicknesses remain labelled
modelling assumptions.

The optics used to sample G on a uniform grid and hand it to a solver
integrating on the tanh-clustered one — matching shapes, mismatched positions
(105 nm of error on the 500 nm top cell, 409 nm on the 1300 nm bottom cell).
Measured consequence: the top sub-cell collected **176.315 A/m² against the
142.033 A/m² of photons actually absorbed in it — 124 % of its own photon
budget**, which is thermodynamically impossible.  The old agreement with Lin
2019 was propped up by that surplus, exactly as the 1D generation fix
(7807985) turned out to have been.

Current figures, N_grid=40 / n_points=25::

    V_oc  [V]       2.01585   target 1.9650   ok (+2.6 %)
    FF              0.77053   target 0.8100   ok (-4.9 %)
    J_sc  [A/m²]   145.747    target 156.000  ok (-6.6 %)
    PCE   [%]       22.6385    target 24.800   ok (-8.7 %)
    top J_sc        145.747    budget 152.719 (95.4 % collection)
    bottom J_sc     146.490    budget 152.763 (95.9 % collection)

Collection efficiencies remain physical and the sub-cell mismatch is 0.508%,
inside the pre-declared 2% acceptance gate.  This closes the recorded current-
matching regression gap without fitting absorber thicknesses.  It does not
remove the input/model-provenance limitation: composition-specific absorber,
PTAA, BCP, and contact data plus independent sub-cell transport validation are
still required before treating the result as predictive.  In particular, the
152.719/152.763 A/m² proxy-optics budgets support the declared ±10% comparison
window and the measured batch range, but remain about 2.1% below the champion
156 A/m² central value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.constants import Q
from perovskite_sim.data import load_am15g, load_nk
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


def test_lin2019_source_stack_and_reported_thicknesses():
    """Keep the source sequence and every reported contact thickness live."""
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")

    assert [layer.name for layer in cfg.top_cell.layers] == [
        "glass",
        "ITO",
        "PTAA",
        "FA_Cs_1p77",
        "C60",
    ]
    assert [layer.name for layer in cfg.junction_stack] == [
        "ALD_SnO2",
        "Au_nanoparticles",
    ]
    assert [layer.name for layer in cfg.bottom_cell.layers] == [
        "PEDOT_PSS",
        "SnPb_1p22",
        "C60",
        "BCP",
    ]
    assert cfg.back_reflector is not None
    assert cfg.back_reflector.name == "Cu_back_contact"

    top_c60 = cfg.top_cell.layers[-1]
    bottom_c60 = cfg.bottom_cell.layers[-2]
    bcp = cfg.bottom_cell.layers[-1]
    assert top_c60.thickness == pytest.approx(20e-9)
    assert cfg.junction_stack[0].thickness == pytest.approx(20e-9)
    assert cfg.junction_stack[1].thickness == pytest.approx(1e-9)
    assert bottom_c60.thickness == pytest.approx(20e-9)
    assert bcp.thickness == pytest.approx(7e-9)
    assert cfg.back_reflector.thickness == pytest.approx(100e-9)


def test_lin2019_cu_optics_cover_protocol_from_native_source():
    """The 300-1100 nm protocol must interpolate inside the Cu source table."""
    requested = np.array([300.0, 1100.0])
    wavelengths, n_cu, k_cu = load_nk("Cu", requested)
    assert wavelengths == pytest.approx(requested)
    assert np.all(np.isfinite(n_cu))
    assert np.all(np.isfinite(k_cu))
    assert np.all(n_cu > 0.0)
    assert np.all(k_cu > 0.0)


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
    tolerance = float(cfg.benchmark["tolerance_pct"]) / 100.0
    accepted_jsc_floor = target_jsc * (1.0 - tolerance)
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
        assert budget >= accepted_jsc_floor, (
            f"{label} absorbed-photon budget {budget:.3f} A/m² cannot support "
            f"the lower edge {accepted_jsc_floor:.3f} A/m² of the declared "
            f"±{100.0 * tolerance:.0f}% paper window"
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
def test_lin2019_subcell_currents_match_within_two_percent(lin2019):
    """Keep the source-stack series-current match isolated and quantitative."""
    _, result = lin2019
    top_jsc = abs(result.top_result.metrics_fwd.J_sc)
    bot_jsc = abs(result.bot_result.metrics_fwd.J_sc)
    mismatch = abs(top_jsc - bot_jsc) / (0.5 * (top_jsc + bot_jsc))
    assert mismatch <= 0.02, (
        f"sub-cell J_sc mismatch is {100.0 * mismatch:.2f}%: "
        f"top={top_jsc:.3f}, bottom={bot_jsc:.3f} A/m²"
    )

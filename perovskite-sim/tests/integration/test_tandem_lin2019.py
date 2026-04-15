"""Lin 2019 Nature Energy tandem benchmark regression test.

Runs the full combined-TMM + series-matched J-V pipeline on the Lin 2019
all-perovskite tandem preset and asserts each figure of merit lies within
the ±tolerance_pct window declared in the config.

The optics use real Dasgupta-repository n,k data (FA-Cs-Pb-BrI 40 % Br for
the 1.77 eV top, MA-FA-Cs-Pb-Sn-I 50 % Sn for the 1.22 eV bottom), an
Ag back reflector for the second-pass IR, a manual V_bi = 1.42 V on the
wide-gap top sub-cell to match Lin's per-junction V_oc ~ 1.18 V, and
absorber thicknesses tuned for current matching. All four FoMs (PCE, J_sc,
V_oc, FF) land inside the ±10 % benchmark window.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.data import load_am15g
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.models.tandem_config import load_tandem_from_yaml


@pytest.mark.slow
def test_lin2019_benchmark_within_tolerance():
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")
    assert cfg.benchmark is not None, "Lin 2019 preset must declare a benchmark block"
    tol = float(cfg.benchmark["tolerance_pct"]) / 100.0

    # Span 300-1100 nm so the Sn-Pb 1.22 eV (~1016 nm) absorption tail is
    # captured. A 1000 nm cap clips ~3 mA/cm² off bottom J_sc.
    wavelengths_nm = np.linspace(300.0, 1100.0, 220)
    _, spectral_flux = load_am15g(wavelengths_nm)
    wavelengths_m = wavelengths_nm * 1e-9

    result = run_tandem_jv(
        cfg,
        wavelengths_m=wavelengths_m,
        spectral_flux=spectral_flux,
        wavelengths_nm=wavelengths_nm,
        N_grid=40,
        n_points=25,
    )
    m = result.metrics

    # PCE in cfg.benchmark is in % (Lin 2019 reports 24.8). compute_metrics
    # returns it as a dimensionless fraction (P_mpp / 1000 W/m²), so divide.
    target_pce = float(cfg.benchmark["target_pce"]) / 100.0
    target_jsc = float(cfg.benchmark["target_jsc_ma_cm2"]) * 10.0  # mA/cm² -> A/m²
    target_voc = float(cfg.benchmark["target_voc_v"])
    target_ff = float(cfg.benchmark["target_ff"])

    assert m.PCE == pytest.approx(target_pce, rel=tol), (
        f"PCE {m.PCE * 100:.2f}% outside ±{tol * 100:.0f}% of target {target_pce * 100:.2f}%"
    )
    assert abs(m.J_sc) == pytest.approx(target_jsc, rel=tol), (
        f"|J_sc| {abs(m.J_sc):.2f} A/m² outside tolerance of {target_jsc:.2f}"
    )
    assert m.V_oc == pytest.approx(target_voc, rel=tol), (
        f"V_oc {m.V_oc:.3f} V outside tolerance of {target_voc:.3f}"
    )
    assert m.FF == pytest.approx(target_ff, rel=tol), (
        f"FF {m.FF:.3f} outside tolerance of {target_ff:.3f}"
    )

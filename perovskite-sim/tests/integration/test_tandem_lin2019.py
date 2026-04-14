"""Lin 2019 Nature Energy tandem benchmark regression test.

Runs the full combined-TMM + series-matched J-V pipeline on the Lin 2019
all-perovskite tandem preset and asserts each figure of merit lies within
the ±tolerance_pct window declared in the config.

Currently marked xfail: the FA_Cs_1p77 / SnPb_1p22 n,k CSVs shipped in
perovskite_sim/data/nk/ are rigid-shift stubs of MAPbI3 and will not
reproduce Lin 2019's spectral response. Replace those CSVs with Lin 2019
SI (or Saliba 2016 / Hao 2014) data before removing the xfail mark.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.data import load_am15g
from perovskite_sim.experiments.tandem_jv import run_tandem_jv
from perovskite_sim.models.tandem_config import load_tandem_from_yaml


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Stub n,k CSVs for FA_Cs_1p77 and SnPb_1p22 are rigid bandgap-shifts "
        "of MAPbI3, not real Lin 2019 absorbers. Replace with Lin 2019 SI "
        "data before removing this mark."
    ),
    strict=False,
)
def test_lin2019_benchmark_within_tolerance():
    cfg = load_tandem_from_yaml("configs/tandem_lin2019.yaml")
    assert cfg.benchmark is not None, "Lin 2019 preset must declare a benchmark block"
    tol = float(cfg.benchmark["tolerance_pct"]) / 100.0

    wavelengths_nm = np.linspace(300.0, 1000.0, 200)
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

    target_pce = float(cfg.benchmark["target_pce"])
    target_jsc = float(cfg.benchmark["target_jsc_ma_cm2"]) * 10.0  # mA/cm² -> A/m²
    target_voc = float(cfg.benchmark["target_voc_v"])
    target_ff = float(cfg.benchmark["target_ff"])

    assert m.PCE == pytest.approx(target_pce, rel=tol), (
        f"PCE {m.PCE:.2f}% outside ±{tol * 100:.0f}% of target {target_pce:.2f}%"
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

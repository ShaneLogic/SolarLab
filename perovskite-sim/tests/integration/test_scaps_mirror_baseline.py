"""SCAPS-mirror baseline J-V smoke test.

Runs the partner SCAPS configuration through the SolarLab solver via the
``scaps_compat`` loader. Pins the base J-V metrics to a window around the
SCAPS reference (V_oc=1.1676 V, J_sc=26.28 mA/cm², FF=86.99%, PCE=26.69%).

This test guards Phase B of the SCAPS validation work: the goal is for the
SCAPS parameter set to produce a SolarLab baseline within ~10% of the
partner numbers without any solver, boundary-condition, or initial-
condition change.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from perovskite_sim.experiments.jv_sweep import compute_metrics, run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml


# SCAPS partner reference (from 1D-SCAPS PDF page 2 / Parameters(1).xlsx).
SCAPS_VOC = 1.1676
SCAPS_JSC_A_m2 = 262.81994   # 26.28 mA/cm^2 -> A/m^2
SCAPS_FF = 0.8699
SCAPS_PCE = 0.2669


@pytest.fixture(scope="module")
def mirror_stack():
    path = Path("configs/scaps_mirror.yaml")
    if not path.exists():
        pytest.fail(
            "configs/scaps_mirror.yaml not present — Phase B config missing"
        )
    return load_scaps_yaml(path)


@pytest.fixture(scope="module")
def mirror_jv(mirror_stack):
    return run_jv_sweep(
        mirror_stack,
        N_grid=30,
        n_points=20,
        v_rate=5.0,
        V_max=1.6,
    )


def test_baseline_voc_bracketed(mirror_jv):
    assert mirror_jv.metrics_fwd.voc_bracketed, (
        "forward sweep did not bracket V_oc — expand V_max"
    )


def test_baseline_voc_within_scaps_window(mirror_jv):
    voc = mirror_jv.metrics_fwd.V_oc
    assert 1.05 <= voc <= 1.25, f"V_oc={voc:.4f} outside [1.05, 1.25] V window"


def test_baseline_jsc_within_scaps_window(mirror_jv):
    """SCAPS J_sc = 26.28 mA/cm^2 = 262.8 A/m^2.

    SolarLab TMM at 800 nm MAPbI3 produces ~24 mA/cm^2 (Fresnel reflection
    at the spiro/MAPbI3 boundary trims the SCAPS scalar-alpha integral),
    so accept the [230, 280] A/m^2 envelope (23.0 to 28.0 mA/cm^2). Tighter
    parity will require an SCAPS-matched optics path, which is out of scope
    for this validation lane.
    """
    jsc = mirror_jv.metrics_fwd.J_sc
    assert 230.0 <= jsc <= 280.0, f"J_sc={jsc:.2f} A/m^2 outside [230, 280] window"


def test_baseline_pce_within_scaps_window(mirror_jv):
    pce = mirror_jv.metrics_fwd.PCE
    assert 0.22 <= pce <= 0.30, f"PCE={pce:.4f} outside [22%, 30%] window"


def test_baseline_ff_reasonable(mirror_jv):
    ff = mirror_jv.metrics_fwd.FF
    assert 0.78 <= ff <= 0.92, f"FF={ff:.4f} outside [78%, 92%] window"

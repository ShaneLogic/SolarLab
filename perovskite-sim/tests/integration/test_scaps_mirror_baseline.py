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
    """SCAPS J_sc = 26.28 mA/cm^2 = 262.8 A/m^2; SolarLab now reports ~218.

    RE-ANCHORED 2026-07-28, and the parity gap genuinely WIDENED -- read this
    before treating the new window as a relaxation.

    The old window [230, 280] was set when the TMM generation was built by
    point-sampling the volumetric rate A(x,lam)*Phi at the nodes and handing
    that to the RHS's node-centred rectangle rule.  That quadrature is not
    photon-conserving on a mesh comparable to the absorption length: it
    over-counts the sharply-peaked front-of-absorber generation.  Measured on
    this config (uniform mesh, 300-900 nm), the point-sampled absorbed current
    oscillates 170.4 / 233.8 / 216.9 / 224.3 A/m^2 at 10 / 31 / 61 / 121 nodes
    and only settles to 218.36 by 8001 nodes -- while the cell-exact form
    reports 218.38 on EVERY mesh from 10 nodes up.  The N_grid=30 value the
    old window was drawn around (~232-234) was therefore the coarse-mesh bias.

    That the old number was not merely imprecise but IMPOSSIBLE is the decisive
    point: the electrical stack's absorbed-photon budget here is 219.01 A/m^2
    (the telescoped Poynting-flux drop, mesh-independent), so a J_sc of 231.7
    collected more carriers than there were photons absorbed to make them.
    The apparent 12 % agreement with SCAPS was manufactured by that surplus.

    So the honest gap to SCAPS is now -17 % (218.2 vs 262.8), not -12 %.  The
    cause is unchanged and still out of scope for this lane: SCAPS integrates a
    scalar alpha with no Fresnel reflection at the spiro/MAPbI3 boundary, and
    closing it needs an SCAPS-matched optics path.  What changed is only that
    SolarLab's side of the comparison is now the number its own optics implies.

    Window [205, 225]: the upper bound sits just under the 219.01 A/m^2 photon
    ceiling that no correct result can cross, so a re-introduced over-count
    fails here rather than passing as parity.
    """
    jsc = mirror_jv.metrics_fwd.J_sc
    assert 205.0 <= jsc <= 225.0, f"J_sc={jsc:.2f} A/m^2 outside [205, 225] window"


def test_baseline_pce_within_scaps_window(mirror_jv):
    pce = mirror_jv.metrics_fwd.PCE
    # Window widened from [22%, 30%] to [21%, 30%] after Phase E1.5
    # activated the PVK/ETL interface defect — the cross-carrier
    # recombination drops PCE by ~0.1 percentage point at the baseline
    # chi values, which is the magnitude trade for the cliff-direction
    # parity gain over the full ΔE_C sweep.
    assert 0.21 <= pce <= 0.30, f"PCE={pce:.4f} outside [21%, 30%] window"


def test_baseline_ff_reasonable(mirror_jv):
    ff = mirror_jv.metrics_fwd.FF
    assert 0.78 <= ff <= 0.92, f"FF={ff:.4f} outside [78%, 92%] window"

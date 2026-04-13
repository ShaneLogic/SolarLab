"""Pinned J_sc baselines for TMM-enabled presets.

If these numbers drift, the optical model changed — document the shift
in the commit message and update the pins intentionally. A drift without
an accompanying physics change is a signal that something silently broke
(e.g. am15g.csv, nk CSVs, TMM math in physics/optics.py).

Baselines measured against commit 329c79d on main, after Task 7.5
(am15g.csv regeneration from ASTM G-173) and Task 8 (pin_MAPbI3_tmm
preset with NiO constant-index fallback).

Run with: pytest -m slow tests/regression/test_tmm_baseline.py
"""
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

pytestmark = pytest.mark.slow

# Pinned measurements (A/m^2)
NIP_TMM_JSC_PINNED = 211.02
PIN_TMM_JSC_PINNED = 220.07

# Tolerance band. ±5 A/m² ≈ ±0.5 mA/cm² ≈ ±2.4% — tight enough to catch
# any non-trivial change in the TMM path, loose enough to absorb solver
# noise from Radau step-control and small grid-dependent rounding.
TOLERANCE = 5.0


@pytest.fixture(scope="module")
def nip_tmm_result():
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    return run_jv_sweep(stack, n_points=21)


@pytest.fixture(scope="module")
def pin_tmm_result():
    stack = load_device_from_yaml("configs/pin_MAPbI3_tmm.yaml")
    return run_jv_sweep(stack, n_points=21)


def test_nip_tmm_baseline_jsc(nip_tmm_result):
    j_sc = nip_tmm_result.metrics_fwd.J_sc
    delta = j_sc - NIP_TMM_JSC_PINNED
    assert abs(delta) < TOLERANCE, (
        f"nip_MAPbI3_tmm J_sc drifted: measured {j_sc:.2f}, "
        f"pinned {NIP_TMM_JSC_PINNED:.2f}, delta {delta:+.2f} A/m² "
        f"(tolerance ±{TOLERANCE})"
    )


def test_pin_tmm_baseline_jsc(pin_tmm_result):
    j_sc = pin_tmm_result.metrics_fwd.J_sc
    delta = j_sc - PIN_TMM_JSC_PINNED
    assert abs(delta) < TOLERANCE, (
        f"pin_MAPbI3_tmm J_sc drifted: measured {j_sc:.2f}, "
        f"pinned {PIN_TMM_JSC_PINNED:.2f}, delta {delta:+.2f} A/m² "
        f"(tolerance ±{TOLERANCE})"
    )

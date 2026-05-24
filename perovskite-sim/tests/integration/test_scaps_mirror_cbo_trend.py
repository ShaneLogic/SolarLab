"""SCAPS-mirror CBO trend test — GREEN as of Phase E1.5.

Pins SCAPS cliff/spike direction across ΔE_C ∈ {−0.5, −0.16, 0, +0.3}.
SCAPS shows V_oc rising ~420 mV from cliff (~0.83 V) through flat band
(~1.25 V) and plateauing for spike. Phase E1.5 cross-carrier
Pauwels-Vanhoutte sampling (n at ETL interior, p at PVK interior)
reproduces the direction with the PVK/ETL ``interfaces:`` block in
``configs/scaps_mirror.yaml`` (SRV=0.01 m/s, empirically calibrated for
SolarLab's bulk-interior discretized sampling).
"""
from __future__ import annotations

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)


@pytest.fixture(scope="module")
def voc_by_delta_ec():
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    out = {}
    for d in [-0.5, -0.16, 0.0, 0.3]:
        pt = SweepPoint("p", "c", f"{d:+.2f}", {"etl_delta_ec_eV": d})
        swept = apply_sweep_point(stack, pt)
        r = run_jv_sweep(swept, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
        out[d] = r.metrics_fwd.V_oc
    return out


def test_cbo_voc_drops_at_cliff(voc_by_delta_ec):
    """SCAPS: V_oc(ΔE_C=−0.5) significantly below V_oc(ΔE_C=0)."""
    assert voc_by_delta_ec[-0.5] < voc_by_delta_ec[0.0] - 0.10, (
        f"V_oc(-0.5)={voc_by_delta_ec[-0.5]:.3f} V not >100 mV below "
        f"V_oc(0.0)={voc_by_delta_ec[0.0]:.3f} V"
    )


def test_cbo_voc_range_at_least_200mV(voc_by_delta_ec):
    """SCAPS V_oc range across ΔE_C∈[−0.5,+0.3] is ~420 mV."""
    rng = max(voc_by_delta_ec.values()) - min(voc_by_delta_ec.values())
    assert rng >= 0.20, f"V_oc range {rng*1000:.0f} mV below 200 mV threshold"


def test_cbo_voc_holds_at_spike(voc_by_delta_ec):
    """SCAPS shows V_oc near-plateau for moderate spike (ΔE_C=+0.3).

    This test passes today on the Phase B baseline because V_oc is
    bulk-limited and roughly constant across the sweep -- the
    near-plateau is incidentally satisfied. Kept active so a future
    Phase E1 landing must preserve the spike-side behaviour while
    bringing the cliff-side direction into alignment.
    """
    assert voc_by_delta_ec[0.3] >= voc_by_delta_ec[0.0] - 0.05, (
        f"V_oc(+0.3)={voc_by_delta_ec[0.3]:.3f} V dropped >50 mV from "
        f"V_oc(0.0)={voc_by_delta_ec[0.0]:.3f} V"
    )

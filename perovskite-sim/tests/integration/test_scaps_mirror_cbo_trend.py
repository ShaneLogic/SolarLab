"""SCAPS-mirror CBO trend test -- XFAIL (Phase E1 work).

Pins the SCAPS cliff/spike direction so a future per-interface SRH
landing has a target to hit. SCAPS shows V_oc rising ~420 mV from
ΔE_C = −0.5 (cliff, V_oc ≈ 0.83 V) through ΔE_C ≈ 0 (V_oc ≈ 1.25 V)
and plateaus for positive ΔE_C up to +0.3 V.

Phase D2 (directional Phase 4a edge profile) reproduces the *magnitude*
of the V_oc response (~150 mV swing achievable) but produces a V-shape
with minimum at the design point rather than SCAPS' monotonic cliff →
spike rise. The directional edge profile is now a real new capability
(useful for CIGS/CdS, c-Si heterojunctions, any layer needing asymmetric
trap distribution), but cliff/spike *direction* parity requires interface
SRH with E_t-aware n1/p1 at the heterojunction node -- solver work in
``solver/mol.py:_apply_interface_recombination`` that is Phase E1, out
of scope here.

These tests are marked xfail so the SCAPS direction target stays
documented and the CI surfaces if/when Phase E1 lands and the gap
closes.
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


_XFAIL_REASON = (
    "Phase E1 (per-interface SRH with E_t-aware n1/p1 at heterojunction "
    "node) not landed yet; Phase 4a directional edge profile produces a "
    "V-shape rather than the SCAPS monotonic cliff→spike rise."
)


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
def test_cbo_voc_drops_at_cliff(voc_by_delta_ec):
    """SCAPS: V_oc(ΔE_C=−0.5) significantly below V_oc(ΔE_C=0)."""
    assert voc_by_delta_ec[-0.5] < voc_by_delta_ec[0.0] - 0.10, (
        f"V_oc(-0.5)={voc_by_delta_ec[-0.5]:.3f} V not >100 mV below "
        f"V_oc(0.0)={voc_by_delta_ec[0.0]:.3f} V"
    )


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
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

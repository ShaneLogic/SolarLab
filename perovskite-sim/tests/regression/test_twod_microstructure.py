"""Stage-B physics gate: a single absorber GB drops V_oc relative to the
laterally-uniform baseline. Quantitative window: 5 mV ≤ ΔV_oc ≤ 100 mV
(published MAPbI3 GB-induced V_oc penalty range)."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import compute_metrics
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack


def _freeze_ions(stack: DeviceStack) -> DeviceStack:
    return replace(stack, layers=tuple(
        replace(layer, params=replace(layer.params, D_ion=0.0))
        for layer in stack.layers
    ))


def _maybe_flip_sign(V: np.ndarray, J: np.ndarray) -> np.ndarray:
    if len(V) < 2:
        return J
    return -J if J[0] < 0 else J


@pytest.mark.regression
@pytest.mark.slow
def test_twod_singleGB_lowers_voc():
    """Stage-B physics gate. Single absorber GB must drop V_oc 5–100 mV
    versus the laterally-uniform baseline (BL preset, frozen ions)."""
    base = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_uniform.yaml"))
    gb = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml"))

    common = dict(
        lateral_length=500e-9, Nx=10,
        V_max=1.2, V_step=0.05,
        illuminated=True, lateral_bc="periodic",
        Ny_per_layer=10, settle_t=1e-3,
    )
    r_base = run_jv_sweep_2d(stack=base, microstructure=None, **common)
    r_gb = run_jv_sweep_2d(stack=gb, microstructure=None, **common)

    V_b = np.asarray(r_base.V)
    J_b = _maybe_flip_sign(V_b, np.asarray(r_base.J))
    V_g = np.asarray(r_gb.V)
    J_g = _maybe_flip_sign(V_g, np.asarray(r_gb.J))
    m_base = compute_metrics(V_b, J_b)
    m_gb = compute_metrics(V_g, J_g)

    print(f"\nbaseline: V_oc={m_base.V_oc * 1e3:.2f} mV  "
          f"GB: V_oc={m_gb.V_oc * 1e3:.2f} mV")
    drop_mV = (m_base.V_oc - m_gb.V_oc) * 1e3
    assert 5.0 <= drop_mV <= 100.0, \
        f"GB V_oc drop {drop_mV:.2f} mV outside [5, 100]"

    rel_jsc = abs(m_gb.J_sc - m_base.J_sc) / abs(m_base.J_sc)
    assert rel_jsc <= 0.10, \
        f"GB J_sc rel drift {rel_jsc:.3f} exceeded 10%"

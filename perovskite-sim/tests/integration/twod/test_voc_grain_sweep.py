"""Stage-B headline experiment: V_oc(L_g) trend on a single-GB MAPbI3 stack."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.twod.experiments.voc_grain_sweep import run_voc_grain_sweep


def _freeze_ions(stack: DeviceStack) -> DeviceStack:
    return replace(stack, layers=tuple(
        replace(layer, params=replace(layer.params, D_ion=0.0))
        for layer in stack.layers
    ))


@pytest.mark.regression
@pytest.mark.slow
def test_voc_grain_sweep_monotone_in_Lg():
    """V_oc should grow as L_g grows (more bulk-like behaviour: a single
    centred GB takes a smaller relative fraction of the lateral domain)."""
    stack = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml"))
    res = run_voc_grain_sweep(
        stack=stack,
        grain_sizes=(200e-9, 500e-9, 1000e-9),
        tau_gb=(1e-9, 1e-9), gb_width=10e-9,
        Nx=8, Ny_per_layer=8,
        V_max=1.2, V_step=0.05, settle_t=1e-3,
    )
    assert res.V_oc_V.shape == (3,)
    # Monotone: V_oc(L_g) should increase as L_g increases (allow 5 mV slack
    # for finite-grid noise on the small L_g end).
    assert np.all(np.diff(res.V_oc_V) >= -5e-3), \
        f"V_oc not monotone in L_g: V_oc={res.V_oc_V}"
    # Total spread should exceed 5 mV — otherwise the GB has no measurable
    # effect across the grain-size range.
    assert (res.V_oc_V[-1] - res.V_oc_V[0]) * 1e3 > 5.0

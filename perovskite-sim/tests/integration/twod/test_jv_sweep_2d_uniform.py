from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d, JV2DResult
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.models.config_loader import load_device_from_yaml


PRESET = "configs/nip_MAPbI3_tmm.yaml"


def test_jv_sweep_2d_returns_finite_result():
    stack = load_device_from_yaml(PRESET)
    result = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=8,
        V_max=1.0, V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
    )
    assert isinstance(result, JV2DResult)
    assert np.all(np.isfinite(result.V))
    assert np.all(np.isfinite(result.J))
    assert len(result.V) == len(result.J)
    # Layer 1+2: every JV2DResult now carries a JVMetrics. On this BL
    # preset (V_oc ≈ 0.91 V) the sweep to V_max=1.0 must bracket V_oc;
    # the metrics must be finite and on the J_sc-positive convention
    # (compute_metrics flipped the sign internally).
    m = result.metrics
    assert m.voc_bracketed is True
    assert m.J_sc > 0.0
    assert 0.0 < m.V_oc <= 1.0
    assert 0.0 < m.FF < 1.0
    assert np.isfinite(m.PCE)


def test_jv_sweep_2d_has_nontrivial_photocurrent():
    """Illuminated J(V=0) magnitude should be substantial (>10 A/m²) on a real
    perovskite preset. Sign convention is solver-dependent; we only check
    magnitude here. Stage-A validation gate (Task 11) will pin the sign by
    comparing to 1D."""
    stack = load_device_from_yaml(PRESET)
    result = run_jv_sweep_2d(
        stack=stack,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=8,
        V_max=0.5, V_step=0.25,    # short sweep — fast
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
    )
    j_at_zero = abs(float(result.J[0]))
    assert j_at_zero > 10.0, f"|J(0)| = {j_at_zero:.3e} A/m² is too small for illuminated"

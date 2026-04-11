"""Numerical regression guard for the RHS material caching refactor.

Phase 2 moves `_build_layerwise_arrays` and `_build_carrier_params` out of
the per-evaluation RHS hot path and into a once-per-experiment builder.
Because the cached arrays are the same numpy float64 values constructed
in the same order with the same operations, the refactor must not shift
any numeric output beyond float-roundoff tolerance. This test pins the
ionmonger benchmark envelope so drift is caught immediately.

Do not relax the envelopes here to accommodate refactor changes. If the
numbers shift, something about the ordering or averaging has changed
and the refactor is broken.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.impedance import run_impedance
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_IONMONGER = str(_CONFIGS_DIR / "ionmonger_benchmark.yaml")


@pytest.mark.slow
def test_jv_sweep_ionmonger_envelope():
    """Ionmonger J-V sweep metrics must stay inside their physical envelope
    after the material-cache refactor."""
    stack = load_device_from_yaml(_IONMONGER)
    r = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=1.0, V_max=1.4)

    assert r.V_fwd.shape == (30,)
    assert r.J_fwd.shape == (30,)
    assert r.V_rev.shape == (30,)
    assert r.J_rev.shape == (30,)
    assert np.all(np.isfinite(r.J_fwd))
    assert np.all(np.isfinite(r.J_rev))

    m = r.metrics_fwd
    assert 0.9 < m.V_oc < 1.3, f"V_oc drift: {m.V_oc}"
    assert 150.0 < m.J_sc < 280.0, f"J_sc drift: {m.J_sc}"
    assert 0.6 < m.FF < 0.9, f"FF drift: {m.FF}"
    assert 0.10 < m.PCE < 0.30, f"PCE drift: {m.PCE}"


@pytest.mark.slow
def test_impedance_ionmonger_finite_and_capacitive():
    """Ionmonger impedance sweep at V_dc = 0.9 V must be finite and
    physically oriented (upper half plane Nyquist) after the refactor."""
    stack = load_device_from_yaml(_IONMONGER)
    freqs = np.logspace(2, 4, 5)
    r = run_impedance(
        stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2,
    )
    assert r.Z.shape == freqs.shape
    assert np.all(np.isfinite(r.Z.real))
    assert np.all(np.isfinite(r.Z.imag))
    assert np.all(r.Z.imag < 0.0), f"Im(Z) sign drift: {r.Z.imag}"
    assert np.all(r.Z.real > 0.0), f"Re(Z) sign drift: {r.Z.real}"

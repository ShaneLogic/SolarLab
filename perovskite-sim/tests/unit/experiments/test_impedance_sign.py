"""Regression guard for the impedance lock-in sign convention.

For any passive, capacitive device, the imaginary part of the complex
impedance Z(omega) must be negative (energy storage, current leads voltage).
The simulator's dummy-mode RC reference already uses this convention, and
the experimental lock-in must agree.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.impedance import run_impedance, extract_impedance

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def test_dummy_rc_has_negative_imaginary_impedance():
    """The dummy-mode RC reference returns Z = R + 1/(j*omega*C),
    which has Im(Z) < 0 at all positive frequencies."""
    freqs = np.logspace(1, 5, 10)
    Z = extract_impedance(freqs, dummy_mode=True)
    assert np.all(np.isfinite(Z.real))
    assert np.all(np.isfinite(Z.imag))
    assert np.all(Z.imag < 0.0), f"dummy RC should have Im(Z) < 0, got {Z.imag}"
    assert np.all(Z.real > 0.0), f"dummy RC should have Re(Z) > 0, got {Z.real}"


@pytest.mark.slow
def test_ionmonger_impedance_is_capacitive():
    """A real perovskite device at V_dc in the operating range should look
    capacitive: Im(Z) < 0 across the swept frequency band. This test pins
    the lock-in sign against the simulated physics."""
    stack = load_device_from_yaml(str(_CONFIGS_DIR / "ionmonger_benchmark.yaml"))
    freqs = np.array([1e2, 1e3, 1e4])
    result = run_impedance(
        stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2,
    )
    assert result.Z.shape == freqs.shape
    assert np.all(np.isfinite(result.Z.real))
    assert np.all(np.isfinite(result.Z.imag))
    assert np.all(result.Z.imag < 0.0), (
        f"Im(Z) must be negative for a capacitive device, "
        f"got Im(Z)={result.Z.imag}"
    )
    assert np.all(result.Z.real > 0.0), (
        f"Re(Z) must be positive for a passive device, "
        f"got Re(Z)={result.Z.real}"
    )

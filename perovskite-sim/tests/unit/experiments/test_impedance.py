import numpy as np
import pytest
from perovskite_sim.experiments.impedance import extract_impedance


def test_extract_impedance_shape():
    freqs = np.logspace(0, 6, 10)
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert Z.shape == (len(freqs),)
    assert np.iscomplexobj(Z)


def test_extract_impedance_high_freq_real():
    """High-frequency Z should be real-dominated (resistive)."""
    freqs = np.array([1e6])
    Z = extract_impedance(freqs, delta_V=0.01, t_settle=1e-3, n_cycles=5,
                          dummy_mode=True)
    assert abs(Z[0].real) > 0


def test_dummy_rc_phase():
    """Dummy RC circuit: Z must have negative imaginary part (capacitive)
    and |angle| must decrease as frequency increases."""
    freqs = np.array([1e2, 1e4, 1e6])
    Z = extract_impedance(freqs, dummy_mode=True)
    # Capacitive: imaginary part must be negative
    assert np.all(Z.imag < 0), f"Expected negative Im(Z), got {Z.imag}"
    # Phase angle |θ| must decrease with frequency (more resistive at high f)
    angles = np.abs(np.angle(Z, deg=True))
    assert angles[0] > angles[1] > angles[2], (
        f"Phase angle should decrease with frequency: {angles}"
    )


def test_impedance_rejects_empty_frequencies():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="frequenc"):
        run_impedance(stack, np.array([]))


def test_impedance_rejects_small_n_grid():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_impedance(stack, np.array([1e3]), N_grid=2)


def test_impedance_rejects_zero_delta_v():
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="delta_V"):
        run_impedance(stack, np.array([1e3]), delta_V=0.0)

"""Experiments must forward progress events through the optional callback."""
from __future__ import annotations
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.impedance import run_impedance


@pytest.mark.slow
def test_jv_sweep_reports_progress():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    events: list[tuple[str, int, int]] = []
    run_jv_sweep(
        stack, N_grid=30, n_points=5, v_rate=1.0, V_max=1.4,
        progress=lambda stage, cur, tot, msg: events.append((stage, cur, tot)),
    )
    stages = {e[0] for e in events}
    assert "jv_forward" in stages
    assert "jv_reverse" in stages
    fwd_counts = [cur for stage, cur, _ in events if stage == "jv_forward"]
    assert fwd_counts == list(range(1, 6))


@pytest.mark.slow
def test_impedance_reports_progress():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    events: list[tuple[str, int, int]] = []
    freqs = np.array([1e3, 1e4, 1e5])
    run_impedance(
        stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2,
        progress=lambda stage, cur, tot, msg: events.append((stage, cur, tot)),
    )
    assert [cur for _, cur, _ in events] == [1, 2, 3]
    assert all(stage == "impedance" for stage, _, _ in events)

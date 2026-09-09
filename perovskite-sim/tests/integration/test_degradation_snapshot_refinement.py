"""Frozen-ion electronic observables under independent time/voltage refinement."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import scipy

from perovskite_sim.experiments.degradation import measure_frozen_ion_snapshot
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import StateVec, build_material_arrays
from perovskite_sim.solver.newton import solve_equilibrium


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def snapshot_curve(request):
    base = load_device_from_yaml("tests/fixtures/configs/tpv_physical_reference.yaml")
    layers = list(base.layers)
    layers[1] = replace(layers[1], params=replace(
        layers[1].params, P0=1e20, P0_neg=0.7e20,
        D_ion=2e-14, D_ion_neg=4e-14, P_lim_neg=1e26,
    ))
    stack = replace(base, layers=tuple(layers))
    x = build_electrical_grid(stack, 24)
    material = build_material_arrays(x, stack)
    state = solve_equilibrium(x, stack)
    view = StateVec.unpack(state, x.size)
    wave = np.sin(2 * np.pi * x / x[-1])
    for values, reference, factor in ((view.P, material.P_ion0, 0.2),
                                      (view.P_neg, material.P_ion0_neg, -0.2)):
        mask = reference > 0.0
        centered = wave[mask] - np.average(wave[mask], weights=material.dx_cell[mask])
        values[mask] = reference[mask] * (1 + factor * centered)
    cache = {}

    evidence_path = os.environ.get("SOLARLAB_F4_EVIDENCE_PATH")
    if evidence_path:
        def save_evidence():
            root = Path(__file__).resolve().parents[2]
            sources = (
                "perovskite_sim/experiments/degradation.py",
                "perovskite_sim/experiments/steady_state.py",
                "perovskite_sim/solver/mol.py",
                "tests/fixtures/configs/tpv_physical_reference.yaml",
                "tests/integration/test_degradation_snapshot_refinement.py",
            )
            payload = {
                "kind": "frozen_ion_snapshot_refinement_data",
                "acceptance_record": "paired pytest/JUnit result; this file captures numerical data",
                "expected_cases": 5, "captured_cases": len(cache),
                "environment": {"python": platform.python_version(),
                                "numpy": np.__version__, "scipy": scipy.__version__},
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in sources},
                "x_m": x, "initial_state": state,
                "state_blocks": ["electron_m3", "hole_m3", "positive_ion_m3", "negative_ion_m3"],
                "positive_ion_background_m3": material.P_ion0,
                "negative_ion_background_m3": material.P_ion0_neg,
                "settle_time_s": 1e-4, "rtol": 1e-6, "atol_m3": 1e-8,
                "cases": [dict(voltage_step_V=key[0], max_step_s=key[1], **asdict(value))
                          for key, value in sorted(cache.items())],
            }

            def serialize(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(f"unsupported evidence value {type(value).__name__}")

            destination = Path(evidence_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, default=serialize, allow_nan=False) + "\n")
        request.addfinalizer(save_evidence)

    def solve(voltage_step=0.02, max_step=2.5e-6):
        key = voltage_step, max_step
        if key not in cache:
            voltages = np.linspace(0.0, 1.2, int(round(1.2 / voltage_step)) + 1)
            cache[key] = measure_frozen_ion_snapshot(
                x, state, stack, voltages, 1e-4, 1e-6, 1e-8, max_step=max_step,
            )
        result = cache[key]
        assert np.all(np.isfinite(result.states))
        assert result.metrics.voc_bracketed
        assert np.max(result.carrier_current_bound_A_m2) <= 0.05
        assert np.max(result.current_spread_A_m2) <= 0.05
        for accepted in result.states:
            np.testing.assert_array_equal(accepted[2 * x.size:], state[2 * x.size:])
        return result

    return solve


def _compare_metrics(coarse, fine):
    assert abs(coarse.V_oc - fine.V_oc) < 1e-3
    assert abs(coarse.J_sc / fine.J_sc - 1.0) < 0.002
    assert abs(coarse.FF / fine.FF - 1.0) < 0.005


def test_snapshot_metrics_are_independent_of_preparation_step_size(snapshot_curve):
    results = [snapshot_curve(max_step=step) for step in (5e-6, 2.5e-6, 1.25e-6)]
    _compare_metrics(results[-2].metrics, results[-1].metrics)
    np.testing.assert_allclose(results[-2].J, results[-1].J, rtol=0.0, atol=0.01)


def test_snapshot_voltage_sampling_resolves_voc_jsc_and_ff(snapshot_curve):
    results = [snapshot_curve(voltage_step=step) for step in (0.02, 0.01, 0.005)]
    _compare_metrics(results[-2].metrics, results[-1].metrics)

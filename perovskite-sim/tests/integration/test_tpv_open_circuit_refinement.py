"""Independent time, tolerance and pulse-amplitude refinements at fixed history."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments.tpv import run_tpv
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", params=["no_ions", "frozen_dual_ions"])
def refinement(request):
    stack = load_device_from_yaml("tests/fixtures/configs/tpv_physical_reference.yaml")
    if request.param == "frozen_dual_ions":
        layers = list(stack.layers)
        params = replace(layers[1].params, P0=1e20, P0_neg=0.7e20,
                         D_ion=0.0, D_ion_neg=0.0, P_lim_neg=1e26)
        layers[1] = replace(layers[1], params=params)
        stack = replace(stack, layers=tuple(layers))
        material = build_material_arrays(build_electrical_grid(stack, 30), stack)
        assert material.has_dual_ions
        assert np.any(material.P_ion0 > 0.0) and np.any(material.P_ion0_neg > 0.0)
        np.testing.assert_array_equal(material.D_ion_face, 0.0)
        np.testing.assert_array_equal(material.D_ion_neg_face, 0.0)
    cache = {}

    def solve(max_step=1e-7, rtol=1e-6, amplitude=0.02):
        key = max_step, rtol, amplitude
        if key not in cache:
            cache[key] = run_tpv(
                stack, N_grid=30, delta_G_frac=amplitude, t_pulse=1e-6,
                t_decay=8e-6, n_points=120, max_step=max_step, rtol=rtol,
                voltage_atol=1e-10,
            )
        return cache[key]

    return solve


def _require_converged_waveform(coarse, fine):
    assert np.all(coarse.valid) and np.all(fine.valid)
    np.testing.assert_array_equal(coarse.t, fine.t)
    amplitude = float(np.max(np.abs(fine.delta_V)))
    assert amplitude > 1e-5
    assert np.max(np.abs(coarse.delta_V - fine.delta_V)) < 0.01 * amplitude
    assert abs(coarse.V_oc - fine.V_oc) < 1e-3
    assert np.max(fine.max_face_current_A_m2) < 0.05
    assert np.max(fine.interval_current_residual_A_m2) < 0.05
    assert fine.charge_voltage_error_V[-1] < min(1e-6, 0.01 * amplitude)
    if coarse.tau is not None and fine.tau is not None:
        assert abs(coarse.tau / fine.tau - 1.0) < 0.01
    else:
        assert coarse.fit.status == fine.fit.status


def test_three_time_levels_preserve_the_same_pulse_and_preparation(refinement):
    levels = [refinement(max_step=step) for step in (2e-7, 1e-7, 5e-8)]
    assert len({result.protocol.protocol_hash for result in levels}) == 1
    _require_converged_waveform(levels[-2], levels[-1])


def test_three_tolerance_levels_resolve_the_same_waveform(refinement):
    levels = [refinement(rtol=tolerance) for tolerance in (1e-4, 1e-5, 1e-6)]
    assert len({result.protocol.protocol_hash for result in levels}) == 1
    _require_converged_waveform(levels[-2], levels[-1])


def test_two_amplitude_halvings_recover_the_small_signal_limit(refinement):
    amplitudes = (0.02, 0.01, 0.005)
    levels = [refinement(amplitude=value) for value in amplitudes]
    for result in levels:
        assert result.protocol.illumination_history[2].duration_s == 1e-6
    normalized = [result.delta_V / amplitude for result, amplitude in zip(levels, amplitudes)]
    error_coarse = np.max(np.abs(normalized[0] - normalized[1]))
    error_fine = np.max(np.abs(normalized[1] - normalized[2]))
    assert error_fine < error_coarse
    assert error_fine < 0.01 * np.max(np.abs(normalized[-1]))

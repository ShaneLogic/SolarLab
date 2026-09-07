from dataclasses import asdict, replace

import numpy as np
import pytest

from perovskite_sim.experiments import waveform_jv as wf
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_widths


@pytest.fixture
def stack():
    return load_device_from_yaml("configs/calado2016_fig1f.yaml")


@pytest.mark.parametrize("change", [
    {"dark_seed_s": -1}, {"branch_dwell_s": 0}, {"turnaround_dark": "false"},
    {"schema_version": True}, {"uniform_generation_rate_m3_s": -1},
    {"uniform_generation_rate_m3_s": True}, {"start_voltage_V": float("nan")},
])
def test_waveform_rejects_invalid_fields(change):
    with pytest.raises((ValueError, TypeError)):
        wf.JVWaveform(**change)


def test_waveform_json_contract_preserves_zero_and_none():
    for rate in (None, 0, 2.5e27):
        value = wf.JVWaveform(uniform_generation_rate_m3_s=rate)
        assert wf.JVWaveform.from_dict(asdict(value)) == value
    with pytest.raises(ValueError, match="extra"):
        wf.JVWaveform.from_dict({**asdict(wf.JVWaveform()), "P0": 0})


def test_history_and_sampling_are_separate(stack):
    value = wf.JVWaveform(turnaround_s=3)
    coarse = wf.build_waveform_protocol(stack, value, n_points=111, v_rate=0.04, V_max=1.2)
    fine = wf.build_waveform_protocol(stack, value, n_points=441, v_rate=0.04, V_max=1.2)
    assert coarse.illumination_history == fine.illumination_history
    assert coarse.illumination_history[3].duration_s == pytest.approx(55)
    assert coarse.illumination_history[4].condition == "dark"
    assert coarse.illumination_history[4].duration_s == 3
    assert coarse.sampling.values[:111] == tuple(np.linspace(-1, 1.2, 111))
    assert coarse.sampling.values[111:] == tuple(np.linspace(-1, 1.2, 111)[::-1])
    assert coarse.protocol_hash != fine.protocol_hash


@pytest.mark.parametrize("size", [11, 31, 121])
def test_uniform_source_conserves_exact_absorber_budget(stack, size):
    x = np.linspace(0, 800e-9, size)
    generation = wf.uniform_absorber_generation(x, stack, 2.5e27)
    assert generation @ dual_cell_widths(x) == pytest.approx(2.5e27 * 400e-9, rel=2e-15)
    assert np.all(generation >= 0)
    assert generation[0] == generation[-1] == 0


def test_driver_keeps_declared_state_chain_and_physical_parameters(stack, monkeypatch):
    stack = replace(stack, layers=tuple(replace(layer, params=replace(layer.params, P0=0, D_ion=0)) for layer in stack.layers))
    calls = []
    def fixed(x, y, device, mat, voltage, start, stop, rtol, atol, *, illuminated, jacobian=None):
        assert atol == wf.DEFAULT_DENSITY_ATOL_M3 == 100.0
        assert callable(jacobian)
        calls.append((voltage, voltage, stop, illuminated, mat.P_ion0.copy()))
        return y.copy()
    def ramp(**kwargs):
        from types import SimpleNamespace
        assert kwargs["atol"] == wf.DEFAULT_DENSITY_ATOL_M3 == 100.0
        assert callable(kwargs["jacobian"])
        duration = kwargs["t_span"][1]
        bias = kwargs["V_app"]
        calls.append((bias(0), bias(duration), duration, kwargs["illuminated"], kwargs["mat"].P_ion0.copy()))
        return SimpleNamespace(success=True, y=kwargs["y0"][:, None])
    monkeypatch.setattr(wf.jv, "_integrate_step", fixed)
    monkeypatch.setattr(wf.jv, "run_transient", ramp)
    before = asdict(stack)
    value = wf.JVWaveform(dark_seed_s=2, dark_prep_s=3, branch_dwell_s=0.1, turnaround_s=4)
    result = wf.run_waveform_jv(stack, value, N_grid=15, n_points=3, v_rate=1, V_max=1)
    assert [(a, b, t, light) for a, b, t, light, _ in calls] == [
        (0, 0, 2, False), (-1, -1, 3, False), (-1, -1, 0.1, True),
        (-1, 0, 1, True), (0, 1, 1, True), (1, 1, 4, False),
        (1, 1, 0.1, True), (1, 0, 1, True), (0, -1, 1, True),
    ]
    assert asdict(stack) == before
    assert all(np.count_nonzero(population) == 0 for *_, population in calls)
    assert result.numerical_scope == "finite_time_diagnostic"
    assert result.jacobian_evaluator == "analytic_single_ion_density"
    assert result.jacobian_fallback_reason is None
    assert result.numerical_controls["rtol"] == 1e-4
    assert result.numerical_controls["atol_m3"] == 100.0
    assert not result.certified
    assert result.protocol.implicit_legacy_protocol is False
    assert result.waveform_protocol_sha256 == result.protocol.protocol_hash


def test_snapshot_retains_negative_species_without_aliasing(stack):
    layers = list(stack.layers)
    layers[1] = replace(layers[1], params=replace(layers[1].params, P0_neg=2e24, D_ion_neg=1e-18))
    stack = replace(stack, layers=tuple(layers), mode="full")
    x = wf.jv.build_electrical_grid(stack, 20)
    mat = wf.jv.build_material_arrays(x, stack)
    state = wf.jv.solve_equilibrium(x, stack)
    snapshot = wf.jv.extract_spatial_snapshot(x, state, stack, 0, mat=mat)
    np.testing.assert_array_equal(snapshot.P_neg, mat.P_ion0_neg)
    state[-len(x):] = 0
    assert snapshot.P_neg.max() == 2e24

"""Accepted Radau histories with an analytic RHS, never a device simulation."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmarks import hysteresis_reference as reference
from perovskite_sim.solver import mol


@pytest.fixture
def bridge(monkeypatch):
    calls = [0]

    def rhs(_time, values, *_args, **_kwargs):
        calls[0] += 1
        rates = np.zeros_like(values)
        rates[:4] = -values[:4]
        return rates

    monkeypatch.setattr(mol, "assemble_rhs", rhs)
    item = reference.ProductionBridge.__new__(reference.ProductionBridge)
    item.np = np
    item.jv = SimpleNamespace(run_transient=mol.run_transient)
    item.x = np.array([0., 1.])
    item.stack = object()
    item.mat = SimpleNamespace(
        P_ion0=np.array([5., 6.]), has_dual_ions=False, N_iface_state=0,
    )
    item.widths = np.array([.5, .5])
    item.initial_inventory = 5.5
    item.numerics = {"rtol": 1e-7, "atol_m3": 1e-9}
    item.jacobian = lambda _time, _values: np.diag([-1., -1., -1., -1., 0., 0.])
    return item, calls


def control(max_calls):
    return reference.Control("analytic", 0, Fraction(0), Fraction(1),
                             Fraction(0), Fraction(0), False, .02, max_calls)


def test_budget_failure_retains_native_accepted_prefix(bridge):
    item, _ = bridge
    state = (1., 2., 3., 4., 5., 6.)
    result = item.advance(state, control(20))
    assert not result.success
    assert len(result.states) >= 2
    assert result.states[0] == state
    assert 0 < result.local_times[-1] < 1
    assert result.local_times[0] == 0
    assert all(b > a for a, b in zip(result.local_times, result.local_times[1:]))
    for time, values in zip(result.local_times, result.states, strict=True):
        np.testing.assert_allclose(values[:4], np.asarray(state[:4])*np.exp(-time), rtol=1e-7)
        assert values[4:] == state[4:]


def test_default_engine_failure_behavior_is_unchanged(bridge):
    item, _ = bridge
    result = mol.run_transient(
        item.x, np.arange(1., 7.), (0., 1.), None, item.stack,
        mat=item.mat, max_step=.02, max_nfev=20,
        rtol=1e-7, atol=1e-9, jacobian=item.jacobian,
    )
    assert not result.success
    assert result.t.size == 0 and result.y.shape == (6, 0)


def test_observer_preserves_complete_radau_mesh_and_counts(bridge):
    item, calls = bridge
    state = (1., 2., 3., 4., 5., 6.)
    protocol = control(10000)
    original = mol.run_transient(
        item.x, np.asarray(state), (0., 1.), None, item.stack,
        mat=item.mat, illuminated=False, V_app=protocol.voltage,
        max_step=.02, max_nfev=10000, rtol=1e-7, atol=1e-9,
        jacobian=item.jacobian,
    )
    original_calls = calls[0]
    calls[0] = 0
    observed = []
    item.accepted_observer = lambda segment, time, values: observed.append((segment, time, values))
    result = item.advance(state, protocol)
    assert original.success and result.success
    np.testing.assert_array_equal(result.local_times, original.t)
    np.testing.assert_array_equal(np.asarray(result.states).T, original.y)
    assert calls[0] == original_calls
    assert result.diagnostics["nfev"] == original.nfev
    assert result.diagnostics["njev"] == original.njev
    assert result.diagnostics["nlu"] == original.nlu
    assert len(observed) == len(result.local_times)-1
    assert all(segment is protocol and isinstance(values, tuple) for segment, _, values in observed)
    assert [time for _, time, _ in observed] == list(result.local_times[1:])
    assert [values for _, _, values in observed] == list(result.states[1:])


def test_nonfinite_failure_retains_last_accepted_state(bridge, monkeypatch):
    item, _ = bridge
    original = mol.assemble_rhs

    def fail_after_acceptance(time, values, *args, **kwargs):
        if time > .05:
            raise mol._RhsNonFinite("controlled analytic failure")
        return original(time, values, *args, **kwargs)

    monkeypatch.setattr(mol, "assemble_rhs", fail_after_acceptance)
    state = (1., 2., 3., 4., 5., 6.)
    result = item.advance(state, control(10000))
    assert not result.success and len(result.states) >= 2
    assert 0 < result.local_times[-1] <= .05
    assert result.diagnostics["last_accepted_local_time_s"] == result.local_times[-1]
    assert "controlled analytic failure" in result.diagnostics["message"]
    np.testing.assert_allclose(result.states[-1][:4],
                               np.asarray(state[:4])*np.exp(-result.local_times[-1]), rtol=1e-7)


def test_accepted_stream_survives_interrupted_control(tmp_path, monkeypatch):
    protocol = control(10000)
    phases = (reference.Phase("analytic", (protocol,)),)
    bridge = SimpleNamespace(stack={}, mat={}, x=np.array([0., 1.]))
    monkeypatch.setattr(reference, "ProductionBridge", lambda *_args: bridge)
    monkeypatch.setattr(reference, "make_timeline", lambda *_args: phases)
    state = (1., 2., 3., 4., 5., 6.)

    def interrupted(item, _phases, _save_segment, _save_event):
        item.accepted_observer(protocol, .01, state)
        raise RuntimeError("controlled interruption after accepted state")

    monkeypatch.setattr(reference, "collect_history", interrupted)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        reference.execute({"numerics": {"max_step_level": 0}},
                          {"request": {}, "contract": {}}, tmp_path)
    path = tmp_path/"Raw/Interval0000Accepted.npy"
    with path.open("rb") as stream:
        np.testing.assert_array_equal(np.load(stream, allow_pickle=False), (.01, *state))
        assert stream.read() == b""

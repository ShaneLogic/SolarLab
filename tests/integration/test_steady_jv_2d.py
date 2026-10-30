"""Accepted-only2D rebasing, admission, failed prefix and saved metric checks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_2d import dc_assembly, jv as example
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_2d import CompiledDevice2D
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.experiments.jv import steady_2d as driver
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest
from solarlab.solvers.nonlinear.steady_2d import Steady2DResult


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_device_2d(load_device(), dc_assembly.spatial_input())
    _, u = dc_assembly.coordinates(device, lateral=True)
    return device, device.state(u, voltage_V=.002)


def test_reference_only_rebase_preserves_all2d_fields_and_dtype(case):
    device, state = case
    next_state = driver._rebase(device, state, .025)
    assert next_state.u.shape == (195,) and next_state.potential_V.shape == (13, 5)
    np.testing.assert_array_equal(next_state.potential_V, state.potential_V)
    for field in ("electron_density_m3", "hole_density_m3"):
        np.testing.assert_allclose(getattr(next_state, field), getattr(state, field), rtol=3e-14, atol=0.0)
    expected = device.column.rebase_coordinates(state.u.reshape(3, 13, 5)[:, :, 0].ravel(), .002, .025)
    np.testing.assert_array_equal(next_state.u.reshape(3, 13, 5)[:, :, 0].ravel(), expected)


def test_metrics_reuse_sampler_and_require_explicit_irradiance():
    v, j = np.asarray([0.0, .5, 1.0]), np.asarray([100.0, 50.0, -5.0])
    no_power = driver._metrics(v, j, None)
    explicit = driver._metrics(v, j, 1000.0)
    assert no_power.efficiency is None and explicit.efficiency == explicit.maximum_power_W_m2 / 1000.0
    assert no_power.open_circuit_voltage_V == explicit.open_circuit_voltage_V
    assert "piecewise_linear" in explicit.limitation
    assert driver._metrics(v[:1], j[:1], None).limitation == "insufficient_accepted_points"


def test_accepted_only_continuation_retains_failed_and_unattempted_prefix(case, monkeypatch, tmp_path):
    """Synthetic control-flow fixture; no numerical solve or PV claim."""
    device, initial_state = case
    initial = driver.SteadyJV2DInitial(initial_state, 1.0, "synthetic", hashlib.sha256(b"synthetic").hexdigest(), "synthetic")
    monkeypatch.setattr(driver, "_admit", lambda *args: (initial_state, 1.0, (GateOutcome("synthetic", True, 0.0, 1.0),)))
    monkeypatch.setattr(driver, "steady_diagnostics", lambda *args: {})
    monkeypatch.setattr(driver, "steady_checks", lambda *args: (GateOutcome("synthetic", True, 0.0, 1.0),))
    calls, accepted_states = [], []
    def solve(device, *, voltage_V, generation_scale, options, initial_state, initial_provenance):
        calls.append(initial_state)
        accepted = len(calls) < 3
        state = initial_state
        if not accepted:
            changed = state.u.copy()
            changed[:device.node_count] += .001
            state = device.state(changed, voltage_V=voltage_V)
        else:
            accepted_states.append(state)
        term = Termination.CONVERGED if accepted else Termination.BUDGET_EXHAUSTED
        newton = NewtonResult(state.u, accepted, term, 0.0 if accepted else 2.0, (), (), 1, 1,
                              state_digest(initial_state.u), "synthetic continuation fixture")
        data = {"original_rows": tuple(range(195)), "terminal_current_per_depth_A_m": np.array([-1e-5, -1e-5])}
        return Steady2DResult(state, accepted, term.value, newton, initial_state.u, state.u, {}, {}, data,
            (GateOutcome("synthetic", True, 0.0, 1.0),), device.identity, voltage_V, generation_scale,
            {"numerical": asdict(options)}, {}, {"residual": 1, "jacobian": 1}, 0.0)
    monkeypatch.setattr(driver, "solve_steady_state_2d", solve)
    requested = np.array([0.0, .025, .05, .075, .1])
    journal = tmp_path / "Journal"
    result = driver.run_steady_jv_2d(device, initial, requested, checkpoint_directory=journal)
    assert len(calls) == 3 and len(result.attempts) == 3 and not result.completed
    np.testing.assert_array_equal(result.voltage_V, requested[:2])
    np.testing.assert_array_equal(result.unattempted_voltage_V, requested[3:])
    np.testing.assert_array_equal(calls[2].u, driver._rebase(device, accepted_states[1], .05).u)
    assert result.attempts[-1].solution is not None and not result.attempts[-1].solution.accepted
    path = tmp_path / "Prefix.json"
    digest = driver.write_steady_jv_2d(result, path)
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("saved reader solved"))
    record = driver.read_steady_jv_2d(path, digest)
    assert not record["completed"] and len(record["result"]["attempts"]) == 3
    np.testing.assert_array_equal(record["unattempted_voltage_V"], requested[3:])
    manifest_sha = hashlib.sha256((journal / "Manifest.json").read_bytes()).hexdigest()
    checkpoint = driver.read_steady_jv_2d_checkpoint(journal, manifest_sha)
    assert checkpoint["finished"] and not checkpoint["completed"] and len(checkpoint["attempts"]) == 3
    with pytest.raises(ValueError, match="failed2D"):
        driver.admit_steady_jv_2d_checkpoint(device, journal, manifest_sha)
    with pytest.raises(ValueError, match="SHA-256"):
        driver.read_steady_jv_2d(path, "wrong")


def test_durable_interrupted_point_and_exact_resume_admission(case, monkeypatch, tmp_path):
    """Simulated external interruption retains accepted data; no actual solve."""
    device, initial_state = case
    initial = driver.SteadyJV2DInitial(initial_state, 1.0, "synthetic", hashlib.sha256(b"synthetic").hexdigest(), "synthetic")
    monkeypatch.setattr(driver, "_admit", lambda *a: (initial_state, 1.0, ()))
    monkeypatch.setattr(driver, "steady_diagnostics", lambda *a: {})
    monkeypatch.setattr(driver, "steady_checks", lambda *a: (GateOutcome("synthetic", True, 0.0, 1.0),))
    calls = []
    def solve(device, *, voltage_V, generation_scale, options, initial_state, initial_provenance):
        calls.append(initial_state)
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated external wall interruption")
        newton = NewtonResult(initial_state.u, True, Termination.CONVERGED, 0.0, (), (), 1, 1,
                              state_digest(initial_state.u), "synthetic checkpoint fixture")
        data = {"original_rows": tuple(range(195)), "terminal_current_per_depth_A_m": np.array([-1e-5, -1e-5])}
        return Steady2DResult(initial_state, True, "converged", newton, initial_state.u, initial_state.u, {}, {}, data,
            (GateOutcome("synthetic", True, 0.0, 1.0),), device.identity, voltage_V, generation_scale,
            {"numerical": asdict(options)}, {}, {"residual": 1, "jacobian": 1}, 0.0)
    monkeypatch.setattr(driver, "solve_steady_state_2d", solve)
    journal = tmp_path / "Interrupted"
    requested = np.array([0.0, .025, .05, .075, .1])
    with pytest.raises(KeyboardInterrupt):
        driver.run_steady_jv_2d(device, initial, requested, checkpoint_directory=journal)
    sha = hashlib.sha256((journal / "Manifest.json").read_bytes()).hexdigest()
    saved = driver.read_steady_jv_2d_checkpoint(journal, sha)
    assert not saved["finished"] and saved["active_status"] == "unresolved_or_interrupted"
    assert len(saved["attempts"]) == 2 and saved["active_point"]["index"] == 2
    np.testing.assert_array_equal(saved["unattempted_voltage_V"], requested[3:])
    resumed, index = driver.admit_steady_jv_2d_checkpoint(device, journal, sha)
    assert resumed.voltage_V == .025 and index == 2
    np.testing.assert_array_equal(resumed.u, calls[1].u)
    manifest_path, active_path = journal / "Manifest.json", journal / "Active.json"
    manifest_raw, active_raw = manifest_path.read_bytes(), active_path.read_bytes()
    member = journal / "Points/00001.json"
    member_raw = member.read_bytes()
    wrong = _decode(json.loads(member_raw))
    wrong["attempt"]["voltage_V"] = .026
    member.write_text(json.dumps(_encode(wrong)))
    manifest = _decode(json.loads(manifest_raw))
    manifest["points"][1]["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(_encode(manifest)))
    with pytest.raises(ValueError, match="request/prefix"):
        driver.read_steady_jv_2d_checkpoint(journal, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    member.write_bytes(member_raw)
    manifest_path.write_bytes(manifest_raw)
    active = _decode(json.loads(active_raw))
    active["voltage_V"] = .051
    active_path.write_text(json.dumps(_encode(active)))
    manifest = _decode(json.loads(manifest_raw))
    manifest["active"] = hashlib.sha256(active_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(_encode(manifest)))
    with pytest.raises(ValueError, match="exact requested"):
        driver.read_steady_jv_2d_checkpoint(journal, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    active_path.write_bytes(active_raw)
    manifest = _decode(json.loads(manifest_raw))
    manifest["finished"] = True
    manifest_path.write_text(json.dumps(_encode(manifest)))
    with pytest.raises(ValueError, match="finished2D"):
        driver.read_steady_jv_2d_checkpoint(journal, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    manifest_path.write_bytes(manifest_raw)
    member.write_bytes(member.read_bytes() + b" ")
    with pytest.raises(ValueError, match="member SHA-256"):
        driver.read_steady_jv_2d_checkpoint(journal, sha)


def test_parent_deadline_stops_before_solver_and_saves_wall_prefix(case, monkeypatch, tmp_path):
    import time
    device, state = case
    initial = driver.SteadyJV2DInitial(state, 1.0, "synthetic", hashlib.sha256(b"synthetic").hexdigest(), "synthetic")
    monkeypatch.setattr(driver, "_admit", lambda *a: (state, 1.0, ()))
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("expired deadline invoked Newton"))
    journal = tmp_path / "Expired"
    result = driver.run_steady_jv_2d(device, initial, [0.0, .025, .05], checkpoint_directory=journal,
        absolute_deadline_s=time.monotonic() - 1.0)
    assert not result.completed and result.solver_calls == 0
    assert result.attempts[0].termination == "wave_wall_budget_exhausted"
    sha = hashlib.sha256((journal / "Manifest.json").read_bytes()).hexdigest()
    saved = driver.read_steady_jv_2d_checkpoint(journal, sha)
    assert saved["finished"] and len(saved["attempts"]) == 1
    assert saved["active_point"] is None and not saved["attempts"][0]["solver_called"]


def test_initial_admission_rejects_failure_without_solver(case, monkeypatch):
    device, _state = case
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("admission solved"))
    text = json.dumps({"schema": "solarlab.classical_2d_steady.research.v1", "completed": False})
    with pytest.raises(ValueError, match="actually completed"):
        driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)


def test_actual_compiled_grain_requires_explicit_matching_initial_before_any_solver(monkeypatch):
    if "grain_recombination" not in CompiledDevice2D.__dataclass_fields__:
        pytest.skip("actual grain binding has not yet been integrated; no-grain base remains admitted")
    data = dc_assembly.spatial_input().editing_data()
    data["microstructure"] = {"grain_boundaries": [{"id": "test_grain", "layer_ids": ["absorber"],
        "x_position": "0.5 um", "width": "0.05 um", "tau_n": "1 ns", "tau_p": "1 ns"}]}
    device = compile_device_2d(load_device(), JV2DInput.model_validate(data))
    assert device.grain_recombination is not None
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("grain scope reached solver"))
    driver._scope(device)
    text = json.dumps({"schema": "solarlab.classical_2d_steady.research.v1", "completed": True})
    with pytest.raises(ValueError, match="explicit grain initial"):
        driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)


def test_source_bound_accepted_illuminated_initial_admits_without_solver(case, monkeypatch):
    device, _state = case
    path = os.environ.get("SOLARLAB_JV2D_INITIAL")
    if path is None:
        pytest.skip("bind archived accepted illumination explicitly; no cold substitute")
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("admission solved"))
    initial = driver.load_steady_jv_2d_initial(device, path,
        "af7dcf6b7f6adad7254666971c87c33357f7b91600ac5a2ca27e62b638784c05")
    assert initial.generation_scale == 1.0 and initial.state.voltage_V == .002
    assert initial.state.u.shape == (195,)


def test_real_large_initial_and_full_point_checkpoint_without_solver(case, monkeypatch, tmp_path):
    import time
    device, _state = case
    path = os.environ.get("SOLARLAB_JV2D_INITIAL")
    if path is None:
        pytest.skip("bind actual af7 input; no cold or synthetic replacement")
    initial = driver.load_steady_jv_2d_initial(device, path,
        "af7dcf6b7f6adad7254666971c87c33357f7b91600ac5a2ca27e62b638784c05")
    assert len(json.dumps({"initial_evidence_json": initial.evidence_json}).encode()) > 2**20
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("real payload checkpoint invoked Newton"))
    journal = tmp_path / "RealInitial"
    result = driver.run_steady_jv_2d(device, initial, [0.0, .025], checkpoint_directory=journal,
        absolute_deadline_s=time.monotonic() - 1.0)
    assert result.solver_calls == 0
    assert (journal / "Request.json").stat().st_size < 2**20
    assert (journal / "Initial.json").read_bytes() == initial.evidence_json.encode()
    manifest_sha = hashlib.sha256((journal / "Manifest.json").read_bytes()).hexdigest()
    checkpoint = driver.read_steady_jv_2d_checkpoint(journal, manifest_sha)
    assert checkpoint["finished"] and len(checkpoint["attempts"]) == 1
    old = _decode(json.loads(initial.evidence_json))["attempts"][0]["result_2d"]
    full_point = {"attempt": {"index": 0, "voltage_V": .002, "accepted": True,
                             "solution": old}, "source_sha256": initial.source_sha256}
    raw = json.dumps(_encode(full_point), sort_keys=True, separators=(",", ":")).encode()
    point = journal / "Points/SavedPoint.json"
    digest = driver._checkpoint_artifact(point, raw)
    assert hashlib.sha256(point.read_bytes()).hexdigest() == digest
    saved = _decode(json.loads(point.read_bytes()))["attempt"]["solution"]
    assert saved["state"]["u"].shape == (195,) and saved["newton"]["accepted"]


@pytest.fixture
def wall_prefix(case, monkeypatch, tmp_path):
    """Synthetic wall-stop control flow, without a numerical acceptance claim."""
    from support.provenance import checked_provenance_owner

    device, state = case
    initial_path = tmp_path / "SyntheticInitial.json"
    initial_path.write_text("synthetic")
    initial = driver.SteadyJV2DInitial(state, 1.0, str(initial_path),
        hashlib.sha256(initial_path.read_bytes()).hexdigest(), "synthetic")
    check = GateOutcome("synthetic", True, 0.0, 1.0)
    monkeypatch.setattr(driver, "_admit", lambda *a: (state, 1.0, (check,)))
    monkeypatch.setattr(driver, "steady_checks", lambda *a: (check,))
    monkeypatch.setattr(driver, "steady_diagnostics", lambda *a: {})
    calls = []
    def solve(device, *, voltage_V, generation_scale, options, initial_state, initial_provenance):
        calls.append(initial_state)
        accepted = len(calls) < 3
        candidate = initial_state.u.copy()
        if not accepted:
            candidate[:device.node_count] += .5
        newton = None if not accepted else NewtonResult(candidate, True, Termination.CONVERGED, 0.0,
            (), (), 1, 1, state_digest(initial_state.u), "synthetic wall-prefix fixture")
        data = {"original_rows": tuple(range(195)), "original_residual": np.zeros(195),
                "physical_carrier_balance_per_depth_m_s": np.zeros((2, 65)),
                "terminal_current_per_depth_A_m": np.array([-1e-5, -1e-5])}
        return Steady2DResult(initial_state if accepted else None, accepted,
            "converged" if accepted else "wall_budget_exhausted", newton, initial_state.u, candidate,
            initial_provenance, data, data if accepted else None, (check,) if accepted else (), device.identity,
            voltage_V, generation_scale, {"numerical": asdict(options), "hidden_seed_or_fallback_calls": 0},
            driver.admitted_source_pins(), {"residual": 1, "jacobian": 1}, 0.0,
            None if accepted else "synthetic wall stop")
    monkeypatch.setattr(driver, "solve_steady_state_2d", solve)
    with checked_provenance_owner():
        result = driver.run_steady_jv_2d(device, initial, np.array([0.0, .025, .05, .075, .1]))
    prefix = tmp_path / "WallPrefix.json"
    digest = driver.write_steady_jv_2d(result, prefix)
    return device, prefix, digest, result, solve, calls


def test_explicit_wall_tail_admits_last_accepted_without_replaying_prefix(wall_prefix, monkeypatch, tmp_path):
    device, prefix, digest, old, solve, calls = wall_prefix
    original_bytes = prefix.read_bytes()
    calls.clear()
    def accepted_solve(**kwargs):
        calls.clear()  # Reuse a synthetic accepted result, never invoke Newton.
        return solve(**kwargs)
    seeds = []
    def tail_solve(device, **kwargs):
        seeds.append(kwargs["initial_state"])
        return accepted_solve(device=device, **kwargs)
    monkeypatch.setattr(driver, "solve_steady_state_2d", tail_solve)
    result = driver.continue_steady_jv_2d(device, prefix, digest)
    assert len(seeds) == 3 and result.completed
    np.testing.assert_array_equal(result.requested_voltage_V, old.requested_voltage_V[2:])
    last = old.attempts[1].solution.state
    np.testing.assert_array_equal(seeds[0].u, driver._rebase(device, last, .05).u)
    assert not np.array_equal(seeds[0].u, old.attempts[-1].solution.candidate_coordinates)
    assert result.attempts[0].seed_provenance["kind"] == "accepted_wall_prefix"
    assert prefix.read_bytes() == original_bytes
    path = tmp_path / "Tail.json"
    sha = driver.write_steady_jv_2d(result, path)
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("linked reader solved"))
    saved = driver.read_steady_jv_2d(path, sha)
    linked = saved["continuation"]
    np.testing.assert_array_equal(linked["combined_voltage_V"], old.requested_voltage_V)
    assert linked["prefix_completed"] is False and linked["completed"] is True
    assert linked["preserved_prefix_failure"]["voltage_V"] == .05
    assert linked["total_actual_solver_calls"] == 6 and example.summary(saved)["accepted_points"] == 5
    linked["metrics"]["short_circuit_current_A_m2"] += 1.0
    path.write_text(json.dumps(_encode(saved)))
    with pytest.raises(ValueError, match="combined metrics"):
        driver.read_steady_jv_2d(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_wall_tail_refuses_numerical_failure_changed_controls_and_saved_source_tampering(wall_prefix, monkeypatch, tmp_path):
    device, prefix, digest, _old, _solve, _calls = wall_prefix
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("refusal reached solver"))
    controls = driver.SteadyJV2DOptions(max_points=5)
    with pytest.raises(ValueError, match="controls changed"):
        driver.continue_steady_jv_2d(device, prefix, digest, options=controls)
    changed = _decode(json.loads(prefix.read_bytes()))
    changed["result"]["attempts"][-1]["termination"] = "budget_exhausted"
    path = tmp_path / "NumericalFailure.json"
    path.write_text(json.dumps(_encode(changed)))
    with pytest.raises(ValueError, match="wall-stopped"):
        driver.continue_steady_jv_2d(device, path, hashlib.sha256(path.read_bytes()).hexdigest())
    changed = _decode(json.loads(prefix.read_bytes()))
    changed["result"]["source_sha256"]["solarlab.equations.dc_2d"] = "0" * 64
    path = tmp_path / "ChangedSource.json"
    path.write_text(json.dumps(_encode(changed)))
    with pytest.raises(ValueError, match="producer provenance|recorded producer"):
        driver.continue_steady_jv_2d(device, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_linked_tail_allows_only_final_endpoint_and_preserves_both_stops(wall_prefix, monkeypatch, tmp_path):
    device, prefix, digest, _old, solve, calls = wall_prefix
    prefix_bytes = prefix.read_bytes()
    calls.clear()
    partial = driver.continue_steady_jv_2d(device, prefix, digest)
    assert len(partial.voltage_V) == 2 and not partial.completed
    tail_path = tmp_path / "StoppedTail.json"
    tail_sha = driver.write_steady_jv_2d(partial, tail_path)
    tail_bytes = tail_path.read_bytes()
    last = partial.attempts[1].solution.state
    calls.clear()
    endpoint = driver.continue_steady_jv_2d(device, tail_path, tail_sha)
    assert len(calls) == 1 and endpoint.completed
    np.testing.assert_array_equal(endpoint.requested_voltage_V, np.array([.1]))
    np.testing.assert_array_equal(calls[0].u, driver._rebase(device, last, .1).u)
    assert prefix.read_bytes() == prefix_bytes and tail_path.read_bytes() == tail_bytes
    path = tmp_path / "Endpoint.json"
    sha = driver.write_steady_jv_2d(endpoint, path)
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("endpoint reader solved"))
    saved = driver.read_steady_jv_2d(path, sha)
    linked = saved["continuation"]
    assert linked["endpoint_only"] and linked["prefix_accepted_points"] == 4
    assert linked["preserved_prefix_failure"]["index"] == 4
    assert linked["preserved_earlier_failure"]["index"] == 2
    assert linked["total_actual_solver_calls"] == 7 and example.summary(saved)["accepted_points"] == 5
    np.testing.assert_array_equal(linked["combined_voltage_V"], np.array([0.0, .025, .05, .075, .1]))
    with pytest.raises(ValueError, match="wall-stopped"):
        driver.continue_steady_jv_2d(device, path, sha)
    prefix_record = driver.read_steady_jv_2d(prefix, digest)
    malformed = asdict(partial)
    malformed["voltage_V"] = malformed["voltage_V"][:1]
    malformed["output_current_A_m2"] = malformed["output_current_A_m2"][:1]
    malformed["terminal_current_per_depth_A_m"] = malformed["terminal_current_per_depth_A_m"][:1]
    linked = driver._linked_tail(prefix_record, malformed)
    with pytest.raises(ValueError, match="one final"):
        driver._linked_tail({"completed": False, "result": malformed, "continuation": linked}, asdict(endpoint))


def test_exact_saved_wall_prefix_admission_does_not_solve(case, monkeypatch):
    device, _state = case
    path = os.environ.get("SOLARLAB_JV2D_PREFIX")
    if path is None:
        pytest.skip("bind actual bc3 prefix under its unchanged physical source; no replay")
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("wall admission solved"))
    sha = "bc3d18fb81050751d4ed700947db957fdb36894c14bba3729e37e0688f30674c"
    initial, requested, readmit = driver._read_wall_prefix(device, path, sha, driver.SteadyJV2DOptions())
    state, generation, checks = readmit()
    prefix = driver.read_steady_jv_2d(path, sha)
    assert initial.state.voltage_V == .9750000000000001 and initial.generation_scale == generation == 1.0
    np.testing.assert_array_equal(requested, np.linspace(0.0, 1.25, 51)[40:])
    np.testing.assert_array_equal(initial.state.u, prefix["result"]["attempts"][39]["solution"]["state"]["u"])
    np.testing.assert_array_equal(state.u, initial.state.u)
    assert all(item.passed for item in checks)


def test_exact_saved_linked_endpoint_admission_does_not_solve(case, monkeypatch):
    device, _state = case
    path = os.environ.get("SOLARLAB_JV2D_ENDPOINT_PREFIX")
    if path is None:
        pytest.skip("bind actual e5 linked stopped tail under its unchanged physical source")
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("endpoint admission solved"))
    sha = "e5a94d45064d4110b163ffdffe21b46341c6585f2bf00ab68c2205f9985ef3f8"
    initial, requested, readmit = driver._read_wall_prefix(device, path, sha, driver.SteadyJV2DOptions())
    state, generation, checks = readmit()
    prefix = driver.read_steady_jv_2d(path, sha)
    assert initial.state.voltage_V == 1.225 and initial.generation_scale == generation == 1.0
    np.testing.assert_array_equal(requested, prefix["continuation"]["original_requested_voltage_V"][50:])
    np.testing.assert_array_equal(initial.state.u, prefix["result"]["attempts"][9]["solution"]["state"]["u"])
    np.testing.assert_array_equal(state.u, initial.state.u)
    assert all(item.passed for item in checks)


def test_saved_actual51point_request_retains_completed_or_failed_prefix_without_solver(monkeypatch):
    path, sha = os.environ.get("SOLARLAB_JV2D_RESULT"), os.environ.get("SOLARLAB_JV2D_SHA256")
    if path is None or sha is None:
        pytest.skip("bind the sole saved2D wave; never repeat its solve")
    monkeypatch.setattr(example, "compile_device_2d", lambda *a, **k: pytest.fail("reader compiled"))
    monkeypatch.setattr(example, "run_steady_jv_2d", lambda *a, **k: pytest.fail("reader solved"))
    record = driver.read_steady_jv_2d(Path(path), sha)
    result = record["result"]
    linked = record.get("continuation")
    offset = 0 if linked is None else linked["prefix_accepted_points"]
    requested = np.linspace(0.0, 1.25, 51)[offset:]
    np.testing.assert_array_equal(result["requested_voltage_V"], requested)
    accepted = len(result["voltage_V"])
    attempts = result["attempts"]
    assert accepted + offset >= 3  # The actual same-wave first-three checkpoint is retained.
    np.testing.assert_array_equal(result["voltage_V"], requested[:accepted])
    assert len(attempts) == accepted + (0 if record["completed"] else 1)
    np.testing.assert_array_equal(record["unattempted_voltage_V"], requested[len(attempts):])
    assert result["incident_power_W_m2"] is None and result["metrics"]["efficiency"] is None
    initial_sha = result["initial_provenance"]["source_sha256"] if linked is None else linked["original_initial_artifact"]["sha256"]
    assert initial_sha == "af7dcf6b7f6adad7254666971c87c33357f7b91600ac5a2ca27e62b638784c05"
    assert result["solver_calls"] == sum(item["solver_called"] for item in attempts)
    for attempt in attempts[:accepted]:
        assert attempt["accepted"] and all(check["passed"] for check in attempt["solution"]["checks"])
        assert attempt["solution"]["state"]["u"].shape == (195,)
        data = attempt["solution"]["diagnostics"]
        assert tuple(data["original_rows"]) == tuple(range(195))
        assert data["original_residual"].shape == (195,) and np.all(np.isfinite(data["original_residual"]))
        assert data["physical_carrier_balance_per_depth_m_s"].shape == (2, 65)
    if record["completed"]:
        assert accepted + offset == 51
    else:
        failure = attempts[-1]
        assert not failure["accepted"] and failure["termination"] != "converged"
        assert failure["error"] or failure["solution"] is not None
        if failure["solution"] is not None:
            assert not failure["solution"]["accepted"]
            assert failure["solution"]["candidate_coordinates"].shape == (195,)
            initial_data = failure["solution"]["initial_diagnostics"]
            if initial_data is not None:
                assert tuple(initial_data["original_rows"]) == tuple(range(195))
        if result["metrics"]["limitation"] == "open_circuit_not_bracketed":
            assert all(result["metrics"][name] is None for name in (
                "open_circuit_voltage_V", "maximum_power_W_m2", "fill_factor"))
    saved_summary = example.summary(record)
    assert saved_summary["accepted_points"] == accepted + offset and saved_summary["qualification"] is False


def test_source_changed_v2_tail_preserves_distinct_admissions_and_original_failure(wall_prefix, monkeypatch, tmp_path):
    from solarlab.reproducibility.job_provenance import provenance_owner
    device, prefix, digest, _old, solve, calls = wall_prefix
    original = prefix.read_bytes()
    def fake_success(device, **kwargs):
        calls.clear()
        return solve(device, **kwargs)
    monkeypatch.setattr(driver, "solve_steady_state_2d", fake_success)
    with provenance_owner(source_files=(("solarlab/experiments/jv/steady_2d.py", b"synthetic changed driver"),),
            git_commit="a" * 40, environment_lock=b"synthetic locked environment", source_attested=True):
        tail = driver.continue_steady_jv_2d(device, prefix, digest)
    path = tmp_path / "DifferentSourceTail.json"
    sha = driver.write_steady_jv_2d(tail, path)
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("readback solved"))
    saved = driver.read_steady_jv_2d(path, sha)
    assert saved["schema"] == "solarlab.steady_jv_2d.v2"
    linked = saved["continuation"]
    assert linked["source_comparison"] == "different" and linked["completed"]
    assert linked["producer_provenance"]["prefix"] != linked["producer_provenance"]["tail"]
    assert linked["prefix_completed"] is False and linked["preserved_prefix_failure"]["voltage_V"] == .05
    assert prefix.read_bytes() == original
    with pytest.raises(ValueError, match="strict_same_source"):
        driver.read_steady_jv_2d(path, sha, current_identity=linked["producer_provenance"]["prefix"], strict_same_source=True)


def test_v1_source_map_rules_are_preserved_and_never_retrofitted(wall_prefix, monkeypatch, tmp_path):
    from copy import deepcopy
    from support.provenance import checked_provenance_owner
    device, prefix, digest, _old, solve, calls = wall_prefix
    old = _decode(json.loads(prefix.read_bytes()))
    old["schema"] = "solarlab.steady_jv_2d.v1"
    old["result"].pop("job_provenance")
    legacy_path = tmp_path / "SyntheticV1.json"
    original = json.dumps(_encode(old)).encode()
    legacy_path.write_bytes(original)
    legacy_digest = hashlib.sha256(original).hexdigest()
    def fake_success(device, **kwargs):
        calls.clear()
        return solve(device, **kwargs)
    monkeypatch.setattr(driver, "solve_steady_state_2d", fake_success)
    with checked_provenance_owner():
        tail = asdict(driver.continue_steady_jv_2d(device, legacy_path, legacy_digest))
    tail.pop("job_provenance")
    linked = driver._linked_tail(old, tail)
    assert "producer_provenance" not in linked and "schema" not in linked
    broken = deepcopy(tail)
    broken["source_sha256"]["solarlab.equations.dc_2d"] = "0" * 64
    with pytest.raises(ValueError, match="v1 continuation changed"):
        driver._linked_tail(old, broken)
    loaded = driver.read_steady_jv_2d(legacy_path, legacy_digest)
    assert loaded["schema"] == "solarlab.steady_jv_2d.v1" and not loaded["completed"]
    assert legacy_path.read_bytes() == original and loaded["result"]["source_sha256"] == old["result"]["source_sha256"]

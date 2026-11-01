"""One self-contained native CI case, routing tests, and pinned failure readback.

The fresh fixture runs a cold dark state plus the actual fresh direct CLI once;
an explicit saved fresh result avoids local repetitions. Routing doubles make
no physical claim. The original failed wave is read only when explicitly bound.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from examples.research_1d.ionic_initialization import compile_shared_case
from examples.research_1d.ionic_steady import controls
from solarlab.experiments import ionic_initialization as driver
from solarlab.experiments.ionic_transient import IonicTransientInitial
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.steady_1d import SteadyStateError


@pytest.fixture(scope="module")
def compiled():
    return compile_shared_case()


@pytest.fixture
def routing_input(compiled, monkeypatch):
    # A numerical state only: synthetic admission is confined to this unit test.
    # It is never written as accepted physical evidence or used for a real solve.
    steady = controls(compiled.ion_node_count)
    state = compiled.initial_state(voltage_V=0.002)
    initial = IonicTransientInitial(state, 0.0, steady, "test-double-only",
                                   hashlib.sha256(b"{}").hexdigest(), "{}")
    monkeypatch.setattr(driver, "_admit_evidence", lambda *args: (state, 0.0, ()))
    return initial, driver.IonicInitializationOptions(steady, driver.POLICY, 25.0)


@pytest.mark.parametrize("policy", ["auto", "continuation", "compiled_cold"])
def test_initialization_policy_is_explicit_and_no_fallback_exists(compiled, policy):
    with pytest.raises(ValueError, match="explicit electronic_density_seed_v1"):
        driver.IonicInitializationOptions(controls(compiled.ion_node_count), policy, 25.0)


@pytest.mark.parametrize("wall_s", [0.0, -1.0, np.inf, np.nan])
def test_nonpositive_or_nonfinite_wall_control_is_refused(compiled, wall_s):
    with pytest.raises(ValueError, match="wall_s"):
        driver.IonicInitializationOptions(controls(compiled.ion_node_count), driver.POLICY, wall_s)


def test_first_density_failure_stops_before_either_newton(compiled, routing_input, monkeypatch):
    initial, options = routing_input
    calls = []

    def fail_density(problem, u):
        calls.append("density")
        assert problem.generation_scale == 1.0 and problem.voltage_V == 0.002
        np.testing.assert_array_equal(u, initial.state.electronic_state.u)
        raise ValueError("test double: finite-positive density preparation refused")

    def forbidden(*args, **kwargs):
        pytest.fail("later solver called after a failed preparation")

    monkeypatch.setattr(driver, 'density_basin_seed', fail_density)
    monkeypatch.setattr(driver, "solve_steady_state", forbidden)
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    result = driver.initialize_ionic_steady_state(compiled, initial=initial,
        voltage_V=0.002, generation_scale=1.0, options=options)
    assert calls == ["density"]
    assert not result.accepted and result.state is None
    assert result.termination == "density_preparation_failed"
    assert result.stages[-1]["accepted"] is False
    assert result.evaluation_counts["component_solver_calls"] == result.evaluation_counts["coupled_solver_calls"] == 0
    assert "test double" in result.error


def test_component_failure_is_retained_and_cannot_seed_coupled(compiled, routing_input, monkeypatch):
    initial, options = routing_input
    monkeypatch.setattr(driver, 'density_basin_seed', lambda problem, u: u.copy())
    failed = NewtonResult(initial.state.electronic_state.u, False, Termination.BUDGET_EXHAUSTED,
        100.0, (), (), 1, 0, state_digest(initial.state.electronic_state.u), "test-double budget")

    def fail_component(device, **kwargs):
        assert kwargs["initial_state"] is not None
        assert kwargs["options"] is options.steady.electronic
        assert kwargs["generation_scale"] == 1.0
        raise SteadyStateError("test-double component", failed)

    def forbidden(*args, **kwargs):
        pytest.fail("full ionic solver called after failed component")

    monkeypatch.setattr(driver, "solve_steady_state", fail_component)
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    result = driver.initialize_ionic_steady_state(compiled, initial=initial,
        voltage_V=0.002, generation_scale=1.0, options=options)
    assert result.component_failure is failed
    assert result.coupled_result is None and result.coupled_initial_state is None
    assert result.evaluation_counts["component_solver_calls"] == 1
    assert result.evaluation_counts["coupled_solver_calls"] == 0
    assert result.termination == "component_budget_exhausted"
    assert not result.accepted and result.qualification is False


def test_source_hash_refusal_happens_without_any_solver(compiled, monkeypatch):
    steady = controls(compiled.ion_node_count)
    initial = IonicTransientInitial(compiled.initial_state(voltage_V=0.002), 0.0, steady,
                                   "invalid-test-input", "0" * 64, "{}")

    def forbidden(*args, **kwargs):
        pytest.fail("a solver was called for an unbound input")

    monkeypatch.setattr(driver, 'density_basin_seed', forbidden)
    monkeypatch.setattr(driver, "solve_steady_state", forbidden)
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        driver.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002, generation_scale=1.0,
            options=driver.IonicInitializationOptions(steady, driver.POLICY, 25.0))


def test_mismatched_controls_refused_before_preparation(compiled, routing_input, monkeypatch):
    initial, options = routing_input
    initial = replace(initial, steady_controls=replace(initial.steady_controls, ion_logit_step=0.5))
    with pytest.raises(ValueError, match="identical steady controls"):
        driver.initialize_ionic_steady_state(compiled, initial=initial,
            voltage_V=0.002, generation_scale=1.0, options=options)


@pytest.fixture(scope="module")
def author_record():
    path = os.environ.get("SOLARLAB_IONIC_INITIALIZATION_RESULT")
    if not path:
        pytest.skip("set the explicit author artifact/hash to test native evidence without repeating the wave")
    digest = os.environ["SOLARLAB_IONIC_INITIALIZATION_SHA256"]
    return driver.read_ionic_initialization(path, digest), Path(path), digest


def test_retained_real_failure_stages_counts_and_physical_density_seed(author_record):
    record, _, _ = author_record
    result = record["result"]
    assert result["compiled_identity"] == "ce9f89b4b818dc5dcb3fb7c59e72bf353182525301ecbfb75b6ff33dd160cf0b"
    assert result["termination"] == "component_budget_exhausted" and not result["accepted"]
    assert result["evaluation_counts"] == {"density_preparation_calls": 1, "density_residual": 1,
        "density_jacobian": 1, "component_solver_calls": 1, "component_residual": 1002,
        "component_jacobian": 100, "coupled_solver_calls": 0, "coupled_residual": 0, "coupled_jacobian": 0}
    assert len(result["component_failure"]["iterations"]) == 100
    assert result["coupled_result"] is None and result["coupled_initial_state"] is None
    assert result["initial"]["source_sha256"] == "71a51353822f5469bd827604336b1a78a4a0043f8c07bd849cf42c6e515e342c"
    seed, original = result["density_seed"], result["initial_electronic_state"]
    assert seed["potential_V"].tobytes() == original["potential_V"].tobytes()
    assert all(np.all(seed[name] > 0) and np.all(np.isfinite(seed[name]))
               for name in ("electron_density_m3", "hole_density_m3"))
    assert seed["hole_density_m3"][24] > 1e13 and original["hole_density_m3"][24] < 1
    assert result["physical_certificate_status"] == "not_evaluated" and not result["qualification"]


def test_lossless_solver_free_read_and_republication(author_record, tmp_path, monkeypatch):
    record, path, digest = author_record

    def forbidden(*args, **kwargs):
        pytest.fail("saved reading or republication invoked a solver")

    for name in ("initialize_ionic_steady_state", "solve_steady_state", "solve_ionic_steady_state",
                 "density_basin_seed", "_admit_evidence"):
        monkeypatch.setattr(driver, name, forbidden)
    reread = driver.read_ionic_initialization(path, digest)
    out = tmp_path / "Readback.json"
    sha = driver.write_ionic_initialization(reread["result"], out)
    again = driver.read_ionic_initialization(out, sha)
    assert _encode(again["result"]) == _encode(record["result"])
    assert again["bindings"] == record["bindings"]
    with pytest.raises(FileExistsError):
        driver.write_ionic_initialization(reread["result"], out)


@pytest.mark.parametrize("field", ["voltage", "generation", "policy", "controls", "source", "initial_generation"])
def test_single_field_condition_or_source_relabel_is_refused(author_record, tmp_path, field):
    record, _, _ = author_record
    changed = deepcopy(record)
    result = changed["result"]
    if field == "voltage":
        result["voltage_V"] += 0.001
    elif field == "generation":
        result["generation_scale"] = 0.0
    elif field == "policy":
        result["solver_controls"]["policy"] = "pretend_policy"
    elif field == "controls":
        result["solver_controls"]["steady"]["electronic"]["continuity_tolerance_A_m2"] *= 10
    elif field == "source":
        result["source_sha256"]["solarlab.solvers.nonlinear.newton"] = "0" * 64
    else:
        result["initial"]["generation_scale"] = 1.0
    raw = (json.dumps(_encode(changed), allow_nan=False) + "\n").encode()
    path = tmp_path / "Contradiction.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="request/control/source identity"):
        driver.read_ionic_initialization(path, hashlib.sha256(raw).hexdigest())


def test_original_input_cross_binding_is_checked_even_with_new_request_hash(author_record, tmp_path):
    record, _, _ = author_record
    changed = deepcopy(record)
    changed["result"]["compiled_identity"] = "0" * 64
    changed["bindings"] = driver._request_bindings(changed["result"])
    raw = (json.dumps(_encode(changed), allow_nan=False) + "\n").encode()
    path = tmp_path / "WrongDevice.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="original initial/source/component"):
        driver.read_ionic_initialization(path, hashlib.sha256(raw).hexdigest())



def test_bad_supplied_direct_seed_cannot_fall_back_to_fresh(compiled, routing_input, monkeypatch):
    initial, options = routing_input
    options = replace(options, policy=driver.DIRECT_POLICY)
    artifact = driver.IonicDensitySeedArtifact("bad-test-source", "0" * 64, "{}")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid saved input triggered fresh preparation or a solver")

    monkeypatch.setattr(driver, 'density_basin_seed', forbidden)
    monkeypatch.setattr(driver, "solve_steady_state", forbidden)
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    with pytest.raises(ValueError, match="density seed artifact SHA-256"):
        driver.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
            generation_scale=1.0, options=options, density_seed_artifact=artifact)


def test_fresh_direct_failure_is_an_explicit_recorded_stage(compiled, routing_input, monkeypatch):
    initial, options = routing_input
    options = replace(options, policy=driver.DIRECT_POLICY)

    def failure(*args, **kwargs):
        raise ValueError("test-double positive-density refusal")

    def forbidden(*args, **kwargs):
        pytest.fail("fresh preparation failure started a later solver")

    monkeypatch.setattr(driver, 'density_basin_seed', failure)
    monkeypatch.setattr(driver, "solve_steady_state", forbidden)
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    result = driver.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
        generation_scale=1.0, options=options)
    assert result.termination == "density_preparation_failed" and not result.accepted
    assert result.density_seed_provenance["mode"] == "fresh"
    assert result.evaluation_counts["density_preparation_calls"] == 1
    assert result.evaluation_counts["component_solver_calls"] == result.evaluation_counts["coupled_solver_calls"] == 0


@pytest.fixture(scope="module")
def fresh_native(compiled, tmp_path_factory):
    """One real default CI wave; an explicit saved result avoids local repetition."""
    import contextlib
    import io
    import time
    from unittest.mock import patch
    import sys

    from examples.research_1d.ionic_initialization import main as cli_main
    from examples.research_1d.transient import json_value
    from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state

    cached = os.environ.get("SOLARLAB_IONIC_INITIALIZATION_FRESH_RESULT")
    if cached:
        digest = os.environ["SOLARLAB_IONIC_INITIALIZATION_FRESH_SHA256"]
        record = driver.read_ionic_initialization(cached, digest)
        return record, Path(cached), digest

    folder = (Path(os.environ["SOLARLAB_IONIC_INITIALIZATION_NATIVE_DIR"])
              if "SOLARLAB_IONIC_INITIALIZATION_NATIVE_DIR" in os.environ
              else tmp_path_factory.mktemp("fresh_ionic_initialization"))
    folder.mkdir(parents=True, exist_ok=True)
    path, dark_path = folder / "FreshInitialization.json", folder / "DarkInitial.json"
    if path.exists() or dark_path.exists():
        raise FileExistsError("native witness paths already exist; use explicit saved evidence to read them")
    started = time.monotonic()
    steady = controls(compiled.ion_node_count)
    dark = solve_ionic_steady_state(compiled, voltage_V=0.002, generation_scale=0.0, options=steady)
    dark_document = {"summary": {"accepted": dark.accepted, "compiled_identity": compiled.identity,
        "voltage_V": 0.002, "generation_scale": 0.0, "initial_policy": "compiled_cold",
        "qualification": False}, "result": dark}
    raw = (json.dumps(json_value(dark_document), indent=2, allow_nan=False) + "\n").encode()
    dark_path.write_bytes(raw)
    dark_sha = hashlib.sha256(raw).hexdigest()
    receipt = {"initial_policy": "one_explicit_cold_dark_coupled_solve",
        "dark_sha256": dark_sha, "dark_accepted": dark.accepted, "dark_solver_calls": 1,
        "dark_evaluation_counts": dark.evaluation_counts, "dark_wall_s": dark.wall_s,
        "state_size": compiled.state_size, "compiled_identity": compiled.identity}
    (folder / "NativeReceipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert dark.accepted, (dark.termination, dark.error, dark.checks)

    # Author-only same-input verification before illuminated nonlinear work.
    # CI needs no archived artifact: its accepted cold state is explicit above.
    reference = os.environ.get("SOLARLAB_IONIC_INITIALIZATION_EQUIVALENCE")
    if reference:
        from solarlab.experiments.transient_results import _decode
        proof = _decode(json.loads(Path(reference).read_bytes()))
        assert dark.state.electronic_state.u.tobytes() == proof["input_electronic_coordinates"].tobytes()
        receipt["dark_input_matches_preflight_words"] = True

    args = ["ionic_initialization", "solve", "--policy", driver.DIRECT_POLICY,
        "--initial-result", str(dark_path), "--initial-sha256", dark_sha, "--voltage", "0.002",
        "--generation-scale", "1", "--wall-s", "25", "--output", str(path),
        "--accepted-steady-output", str(folder / "AcceptedIlluminatedSteady.json")]
    stdout = io.StringIO()
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(stdout):
        code = cli_main()
    (folder / "CLI.stdout.txt").write_text(stdout.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = driver.read_ionic_initialization(path, digest)
    receipt.update(cli_argv=args, cli_exit_code=code, fresh_result_sha256=digest,
        fresh_evaluation_counts=record["result"]["evaluation_counts"],
        accepted=record["result"]["accepted"], wall_s=time.monotonic() - started)
    if reference:
        receipt["prepared_seed_matches_preflight_words"] = (
            record["result"]["density_seed"]["u"].tobytes() == proof["fresh_seed"]["u"].tobytes())
    (folder / "NativeReceipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert code == 0, record["result"]["termination"]
    assert record["result"]["accepted"]
    if reference:
        assert receipt["prepared_seed_matches_preflight_words"]
    return record, path, digest


def test_self_contained_fresh_cli_has_one_preparation_and_original_coupled_acceptance(fresh_native):
    record, _, _ = fresh_native
    r = record["result"]
    assert r["accepted"] and r["termination"] == "converged"
    assert r["solver_controls"]["policy"] == driver.DIRECT_POLICY
    assert r["density_seed_provenance"]["mode"] == "fresh"
    counts = r["evaluation_counts"]
    assert counts["density_preparation_calls"] == counts["density_residual"] == counts["density_jacobian"] == 1
    assert counts["component_solver_calls"] == counts["component_residual"] == counts["component_jacobian"] == 0
    assert counts["coupled_solver_calls"] == 1 and counts["coupled_residual"] > 0 and counts["coupled_jacobian"] > 0
    assert r["component_result"] is None and r["component_failure"] is None
    assert all(check["passed"] for check in r["coupled_result"]["checks"])
    assert r["physical_certificate_status"] == "not_evaluated" and not r["qualification"]
    assert r["coupled_result"]["solver_controls"] == r["solver_controls"]["steady"]


def test_fresh_join_preserves_profile_then_all_species_remain_active(fresh_native):
    import math

    r = fresh_native[0]["result"]
    initial, seed, joined, final = r["initial"]["state"], r["density_seed"], r["coupled_initial_state"], r["coupled_result"]
    n = seed["potential_V"].size
    assert n == 37 and joined["u"].size == 137
    assert seed["potential_V"].tobytes() == r["initial_electronic_state"]["potential_V"].tobytes()
    assert joined["u"][:3*n].tobytes() == seed["u"].tobytes()
    assert joined["u"][3*n:].tobytes() == initial["u"][3*n:].tobytes()
    weights = np.asarray(r["binding_manifest"]["ion_domain"]["cell_measures_m"])
    for species in ("positive", "negative"):
        key = species + "_ion_density_m3"
        assert joined[key].tobytes() == initial[key].tobytes()
        assert np.max(abs(final["state"][key] / joined[key] - 1.0)) > 0.0
        info = final["diagnostics"]["ion_species"][species]
        inventory = math.fsum((weights * final["state"][key]).tolist())
        assert inventory == info["inventory_m2"]
        assert info["inventory_relative_error"] <= 1e-12
        assert info["ion_balance_l1_m2_s"] <= 1e6 and info["ion_flux_max_m2_s"] <= 1e6
        assert info["original_dropped_balance_m2_s"] == final["diagnostics"]["original_residual"][info["constraint_row"]]
    assert final["diagnostics"]["shared_site"]["maximum_summed_site_fraction"] < 1
    assert final["diagnostics"]["carrier_balance_A_m2"] <= 1e-8
    assert final["diagnostics"]["current_spread_A_m2"] <= 1e-8
    assert final["diagnostics"]["total_current_density_A_m2"].shape == (38,)


def test_fresh_accepted_read_is_solver_free_and_rejects_relabel(fresh_native, tmp_path, monkeypatch):
    record, path, digest = fresh_native

    def forbidden(*args, **kwargs):
        pytest.fail("a read operation invoked numerical work")

    for name in ("initialize_ionic_steady_state", "solve_steady_state", "solve_ionic_steady_state",
                 "density_basin_seed", "_admit_evidence"):
        monkeypatch.setattr(driver, name, forbidden)
    read = driver.read_ionic_initialization(path, digest)
    assert _encode(read) == _encode(record)
    changed = deepcopy(record)
    changed["result"]["generation_scale"] = 2.0
    raw = (json.dumps(_encode(changed), allow_nan=False) + "\n").encode()
    wrong = tmp_path / "WrongGeneration.json"
    wrong.write_bytes(raw)
    with pytest.raises(ValueError, match="request/control/source identity"):
        driver.read_ionic_initialization(wrong, hashlib.sha256(raw).hexdigest())

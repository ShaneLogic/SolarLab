"""Explicit grain initial/model admission and saved-only shared-driver checks."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_2d import grain_jv, grains
from examples.research_2d.dc_assembly import coordinates, spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.experiments.jv import steady_2d as driver
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.physics.recombination.reaction import BulkRecombinationReaction

INITIAL_SHA = "314e3b497ef71a686110885bc2bca3d5e2e2f1d6c5220ca70c76a7caf07ad154"


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def device():
    return compile_device_2d(load_device(), grains.spatial_input())


@pytest.fixture(scope="module")
def initial_path():
    value = os.environ.get("SOLARLAB_GRAIN_JV_INITIAL")
    if value is None:
        pytest.skip("bind accepted314 grain initial explicitly; no cold substitute")
    return Path(value)


@pytest.fixture
def model_matched_fixture(device, initial_path):
    """Synthetic model relocation; preserve historical source labels and314 fields."""
    record = _decode(json.loads(initial_path.read_bytes()))
    # Example source paths participate in identity. Rebind this in-memory test
    # fixture after relocation; never change the original numerical artifact.
    record["compiled_identity"] = device.identity
    record["result_2d"]["compiled_identity"] = device.identity
    record["result_2d"]["state"]["device_sha256"] = device.identity
    record["fixture_scope"] = "pure source-admission fixture, not fresh numerical evidence"
    return record


def test_current_source_grain_admission_rechecks_full_rows_without_solver(device, model_matched_fixture, monkeypatch):
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("admission solved"))
    text = json.dumps(_encode(model_matched_fixture))
    state, generation, _ = driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)
    assert state.u.shape == (195,) and generation == 1.0 and state.voltage_V == .002
    model = driver._grain_model(device)
    assert model["model_id"] == device.grain_recombination.model_id
    np.testing.assert_array_equal(state.u, model_matched_fixture["result_2d"]["state"]["u"])
    assert model["regions"][0]["declaration"] == model_matched_fixture["grain_binding"]["regions"][0]["declaration"]


def test_historical_source_drift_keeps_same_model_physical_admission(device, model_matched_fixture, monkeypatch):
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("source drift admission solved"))
    saved_sources = dict(model_matched_fixture["result_2d"]["source_sha256"])
    text = json.dumps(_encode(model_matched_fixture))
    state, generation, checks = driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)
    assert state.u.shape == (195,) and generation == 1.0 and all(c.passed for c in checks)
    assert model_matched_fixture["result_2d"]["source_sha256"] == saved_sources


def test_grain_saved_source_tampering_refused_and_inventory_does_not_enable_grain(device, model_matched_fixture):
    model_matched_fixture["source_sha256"]["solarlab.device.grain_2d"] = "wrong"
    text = json.dumps(_encode(model_matched_fixture))
    with pytest.raises(ValueError, match="SHA-256|saved grain"):
        driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)
    ordinary = compile_device_2d(load_device(), spatial_input())
    from solarlab.reproducibility.job_provenance import job_provenance, admitted_source_pins
    from support.provenance import checked_provenance_owner
    with checked_provenance_owner(), job_provenance({"compiled_identity": ordinary.identity}, operator="DC2DProblem"):
        pins = admitted_source_pins()
        assert "solarlab.device.grain_2d" in pins
        assert driver._grain_model(ordinary) is None
        assert driver._grain_model(device)["model_id"] == device.grain_recombination.model_id



def test_grain_reference_rebase_preserves_density_and_full_lateral_fields(device, initial_path):
    raw = _decode(json.loads(initial_path.read_bytes()))
    state = device.state(raw["result_2d"]["state"]["u"], voltage_V=.002)
    rebased = driver._rebase(device, state, .025)
    np.testing.assert_array_equal(rebased.potential_V, state.potential_V)
    for name in ("electron_density_m3", "hole_density_m3"):
        np.testing.assert_allclose(getattr(rebased, name), getattr(state, name), rtol=3e-14, atol=0.0)
    assert rebased.device_sha256 == state.device_sha256 and rebased.u.shape == (195,)
    assert np.max(np.ptp(state.electron_density_m3, axis=1)) > 0.0


@pytest.mark.parametrize("change", ["model", "fraction", "failure", "auxiliary"])
def test_changed_grain_model_or_unaccepted_initial_refused_before_solver(device, model_matched_fixture, monkeypatch, change):
    monkeypatch.setattr(driver, "solve_steady_state_2d", lambda *a, **k: pytest.fail("invalid admission solved"))
    record = model_matched_fixture
    if change == "model":
        record["grain_binding"]["model_id"] = "wrong"
    elif change == "fraction":
        record["grain_binding"]["regions"][0]["fractions"][5, 1] += .001
    elif change == "failure":
        record["result_2d"]["accepted"] = False
    else:
        record["full_2d_newton_calls"] = 0
    text = json.dumps(_encode(record))
    with pytest.raises(ValueError):
        driver._admit(device, text, hashlib.sha256(text.encode()).hexdigest(), driver.SteadyJV2DOptions().steady)


def test_no_grain_device_refuses_grain_initial_and_explicit_clear_preserves_no_grain_path(initial_path):
    prepared = load_device()
    ordinary = compile_device_2d(prepared, spatial_input())
    cleared_data = grains.spatial_input().editing_data()
    cleared_data["microstructure"]["grain_boundaries"] = []
    from solarlab.experiments.two_dimensional.inputs import JV2DInput
    cleared = compile_device_2d(prepared, JV2DInput.model_validate(cleared_data))
    assert driver._grain_model(ordinary) is None and driver._grain_model(cleared) is None
    _, u = coordinates(ordinary, lateral=True)
    for voltage in (.002, .025):
        np.testing.assert_array_equal(DC2DProblem(ordinary, voltage, 1.0).residual(u),
                                      DC2DProblem(cleared, voltage, 1.0).residual(u))
    with pytest.raises(ValueError, match="neutral grain"):
        driver.load_steady_jv_2d_initial(ordinary, initial_path, INITIAL_SHA)


def test_zero_effect_grain_lifetimes_do_not_change_rate_or_density_partials(device):
    """Algebra-only uniform lifetime fixture, no numerical acceptance claim."""
    coeff = replace(device.recombination.coefficients, tau_n_s=np.full(65, .5e-6), tau_p_s=np.full(65, .25e-6))
    _, u = coordinates(device, lateral=True)
    _, n, p = device.fields(u, voltage_V=.002)
    bulk = BulkRecombinationReaction(coeff).evaluate(n, p)
    mixed = device.grain_recombination.evaluate(bulk, coeff)
    for name in ("rate", "d_rate_dn", "d_rate_dp"):
        np.testing.assert_array_equal(getattr(mixed, name), getattr(bulk.total, name))
    driver._scope(device)


def test_actual_grain_provenance_round_trip_and_wrong_fraction_refusal(device):
    g = device.geometry
    geometry = {"identity": g.sha256, "shape_Ny_Nx": g.shape,
        "cell_areas_m2": g.contract_geometry.volumes, "grain_model": driver._grain_model(device)}
    driver._read_grain_model(geometry)
    geometry = _decode(json.loads(json.dumps(_encode(geometry))))
    geometry["grain_model"]["regions"][0]["fractions"][5, 1] += .001
    with pytest.raises(ValueError, match="fraction/area"):
        driver._read_grain_model(geometry)


def test_saved_grain_curve_and_baseline_comparison_have_no_solver(device, monkeypatch):
    path, sha = os.environ.get("SOLARLAB_GRAIN_JV_RESULT"), os.environ.get("SOLARLAB_GRAIN_JV_SHA256")
    baseline, baseline_sha = os.environ.get("SOLARLAB_GRAIN_JV_BASELINE"), os.environ.get("SOLARLAB_GRAIN_JV_BASELINE_SHA256")
    if any(value is None for value in (path, sha, baseline, baseline_sha)):
        pytest.skip("bind the sole actual grain curve and accepted no-grain archive; no rerun")
    monkeypatch.setattr(grain_jv, "compile_device_2d", lambda *a, **k: pytest.fail("saved grain reader compiled"))
    monkeypatch.setattr(grain_jv, "run_steady_jv_2d", lambda *a, **k: pytest.fail("saved grain reader solved"))
    record = driver.read_steady_jv_2d(Path(path), sha)
    result = record["result"]
    assert result["geometry"]["grain_model"]["model_id"] == device.grain_recombination.model_id
    np.testing.assert_array_equal(result["requested_voltage_V"], np.linspace(0.0, 1.25, 51))
    initial = Path(result["initial_provenance"]["source_path"])
    assert result["initial_provenance"]["source_sha256"] == hashlib.sha256(initial.read_bytes()).hexdigest()
    assert result["initial_provenance"]["grain_source_admission"]["current_original_gate_revalidation"]
    assert result["initial_provenance"]["grain_source_admission"]["exact_source_match"]
    for attempt in result["attempts"]:
        if attempt["accepted"]:
            assert all(check["passed"] for check in attempt["solution"]["checks"])
            assert tuple(attempt["solution"]["diagnostics"]["original_rows"]) == tuple(range(195))
    value = grain_jv.summary(record)
    assert value["accepted_points"] == len(result["voltage_V"]) and value["qualification"] is False
    comparison = grain_jv.compare_saved(Path(path), sha, Path(baseline), baseline_sha)
    assert comparison["new_numerical_solves"] == 0 and comparison["qualification"] is False
    assert result["incident_power_W_m2"] is None and result["metrics"]["efficiency"] is None

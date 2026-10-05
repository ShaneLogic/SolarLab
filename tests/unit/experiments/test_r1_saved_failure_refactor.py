"""Portable input/guard regressions; no DC preparation or trajectory solve."""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perovskite-sim"))
SPEC = importlib.util.spec_from_file_location(
    "r1_saved_failure_diagnostic", ROOT / "scripts/benchmarks/diagnose_r1_saved_failure.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)

from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_local_carrier import LocalCarrierInputs
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import R1PreparedState, physical_preparation_identity
from perovskite_sim.physics.compensated import DD


@pytest.fixture(scope="module")
def frozen():
    return diagnostic.load_fixture()


def test_frozen_input_identity_is_pinned_and_historical_failures_remain(frozen):
    assert diagnostic.file_sha256(diagnostic.FIXTURE) == "29e4981bd4d5d11d19f3120febbd47d853d39bfb45a72a44ecec91566757db1d"
    assert frozen["source_commit"] == "12869a4f90d6d08af2d24af902838d8a4d5c00c4"
    assert len(frozen["request"]["certificate_limits"]) == 16
    assert len(frozen["operator_channels"]) == 13
    assert frozen["prefix"]["accepted_rows"] == 309
    assert frozen["prefix"]["finite_accepted_steps"] == 308
    assert frozen["prefix"]["expected_rows"] == frozen["request"]["expected_rows"] == 3727
    assert not frozen["prefix"]["bilateral_100s_qualified"]
    assert not frozen["prefix"]["compensated_admitted"]
    limit = frozen["request"]["certificate_limits"]["nonlinear_residual"]
    for name, sign in (("step183_failed", -1), ("step309_failed", 1)):
        residual = np.asarray(frozen["cases"][name]["saved_witness"]["scaled_residual_vector"])
        assert np.flatnonzero(np.abs(residual) > limit).tolist() == [895]
        assert np.sign(residual[895]) == sign
    accepted = frozen["cases"]["step183_accepted"]
    assert accepted["saved_row"]["solver_accepted"]
    assert accepted["saved_row"]["physics_reconstruction"]["scaled_nonlinear_residual"] == accepted["historical_validation"]["scaled_nonlinear_residual"]


def test_preparation_artifact_identity_is_distinct_from_shared_physics(frozen):
    first = R1PreparedState.from_dict(frozen["prepared_v52"])
    second = R1PreparedState.from_dict(frozen["prepared"])
    assert first.sha256 != second.sha256
    assert physical_preparation_identity(first) == physical_preparation_identity(second)
    assert first.to_dict()["source"] != second.to_dict()["source"]


def test_one_bit_changed_input_is_rejected_without_resealing(tmp_path, frozen):
    changed = copy.deepcopy(frozen)
    coordinate = changed["cases"]["step309_failed"]["saved_witness"]["coordinate"]
    coordinate[895] = float(np.nextafter(coordinate[895], np.inf))
    path = tmp_path / "CorruptedInput.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="content hash mismatch"):
        diagnostic.load_fixture(path)


@pytest.mark.parametrize("name", ["step183_failed", "step183_accepted", "step309_failed"])
def test_direct_predecessor_copies_original_words_without_reencoding(frozen, name):
    row = frozen["cases"][name]["previous_row"]
    state = diagnostic.direct_saved_reference(None, row)
    for attribute, field in (("n", "n_m3"), ("p", "p_m3"), ("positive", "positive_m3"),
                             ("phi", "phi_V"), ("occupancy", "occupancy")):
        assert getattr(state, attribute).tobytes() == np.asarray(row["state"][field], dtype=float).tobytes()
    assert state.dqfn.tobytes() == np.asarray(row["physics_reconstruction"]["electron_qf_increment_V"]).tobytes()
    assert state.local[0].state_m3.tobytes() == np.asarray(row["state"]["trace_state_m3"][0]).tobytes()
    assert state.negative is None
    with pytest.raises(AttributeError):
        _ = state.storage_jacobian  # An unstored Jacobian must not be fabricated.


def guard_case(frozen, sign):
    """Isolate the real production guard at an asymmetric binary64 bin boundary."""
    system = object.__new__(ControlledPhysicalInterfaceIonSystem)
    system.interface_count, system.dimension, system.local_slice = 1, 8, slice(2, 8)
    physical = np.array([1.e12, 2.e12, float(2**44), 3.e12])
    system._step_reference = SimpleNamespace(local=(SimpleNamespace(state_m3=physical.copy()),))
    inputs = LocalCarrierInputs(
        DD(physical), DD([-.04, -.04]), DD([1.e13] * 4), DD([-.03, -.05]), DD(.3),
    )
    state = SimpleNamespace(local=(SimpleNamespace(state_m3=physical, carrier_data=SimpleNamespace(inputs=inputs)),))
    coordinate, direction, residual = np.zeros(8), np.zeros(8), np.zeros(8)
    limit = frozen["request"]["certificate_limits"]["nonlinear_residual"]
    residual[6] = sign * 1.1 * limit
    target = np.nextafter(physical[2], np.inf if sign > 0 else -np.inf)
    direction[6] = np.log1p((target - physical[2]) / physical[2])
    return system, state, coordinate, direction, residual, [np.ones(1), np.ones(1), np.ones(6)], limit


@pytest.mark.parametrize("sign", [-1, 1])
def test_directional_neighbor_preserves_all_other_words(frozen, sign):
    system, state, coordinate, direction, residual, scales, limit = guard_case(frozen, sign)
    original = state.local[0].state_m3.copy()
    before_words = diagnostic.local_input_words(state)
    candidate = system.representable_line_search_candidate(state, state, coordinate, direction, residual, *scales, limit)
    assert candidate is not None
    assert np.flatnonzero(candidate != coordinate).tolist() == [6]
    expected = original.copy()
    expected[2] = np.nextafter(original[2], np.inf if sign > 0 else -np.inf)
    np.testing.assert_array_equal(system._trace_density_coordinates(candidate), [expected])
    np.testing.assert_array_equal(state.local[0].state_m3, original)
    assert diagnostic.local_input_words(state) == before_words
    assert not np.shares_memory(candidate, coordinate)
    upward = np.nextafter(original[2], np.inf) - original[2]
    downward = original[2] - np.nextafter(original[2], -np.inf)
    assert upward == 2 * downward  # A fixed, sign-independent ULP would be wrong here.


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("reason", [
    "zero", "negative_zero", "tiny", "large", "nan", "inf", "at_limit", "below_limit",
    "multiple_rows", "bulk_row", "no_reference", "two_interfaces", "coordinate_shape",
    "direction_shape", "residual_shape", "local_shape", "nonpositive_limit", "reference_zero",
    "wrong_map_direction", "reference_too_far",
])
def test_production_guard_rejects_ineligible_candidate(frozen, sign, reason):
    system, state, coordinate, direction, residual, scales, limit = guard_case(frozen, sign)
    if reason in ("zero", "negative_zero", "nan", "inf"):
        direction[6] = {"zero": 0., "negative_zero": -0., "nan": np.nan, "inf": np.inf}[reason]
    elif reason in ("tiny", "large"):
        direction[6] *= .1 if reason == "tiny" else 4.
    elif reason == "at_limit":
        residual[6] = sign * limit
    elif reason == "below_limit":
        residual[6] = sign * np.nextafter(limit, 0.)
    elif reason == "multiple_rows":
        residual[5] = 2 * limit
    elif reason == "bulk_row":
        residual[0] = 2 * limit
    elif reason == "no_reference":
        system._step_reference = None
    elif reason == "two_interfaces":
        system.interface_count = 2
    elif reason == "coordinate_shape":
        coordinate = coordinate[:-1]
    elif reason == "direction_shape":
        direction = direction[:-1]
    elif reason == "residual_shape":
        residual = residual[:-1]
    elif reason == "local_shape":
        scales[-1] = np.ones(5)
    elif reason == "nonpositive_limit":
        limit = 0.
    elif reason == "reference_zero":
        system._step_reference.local[0].state_m3[2] = 0.
    elif reason == "wrong_map_direction":
        coordinate[6] = 2 * direction[6]
    elif reason == "reference_too_far":
        system._step_reference.local[0].state_m3[2] *= .1
    assert system.representable_line_search_candidate(
        state, state, coordinate, direction, residual, *scales, limit,
    ) is None


@pytest.mark.parametrize("field", ["state_density", "trace_potential", "bulk_density", "bulk_potential", "occupancy"])
@pytest.mark.parametrize("word", ["hi", "lo"])
def test_every_support_field_including_low_word_must_match(frozen, field, word):
    system, state, coordinate, direction, residual, scales, limit = guard_case(frozen, 1)
    original = state.local[0].carrier_data.inputs
    value = getattr(original, field)
    high, low = value.hi.copy(), value.lo.copy()
    if word == "hi":
        high.reshape(-1)[0] = np.nextafter(high.reshape(-1)[0], np.inf)
    else:
        low.reshape(-1)[0] = 1.e-30
    changed = replace(original, **{field: DD._trusted_parts(high, low)})
    rejected = SimpleNamespace(local=(SimpleNamespace(carrier_data=SimpleNamespace(inputs=changed)),))
    assert system.representable_line_search_candidate(
        state, rejected, coordinate, direction, residual, *scales, limit,
    ) is None


@pytest.mark.parametrize("metric", ["nonlinear_residual", "charge_balance_relative", "all_face_current_relative", "interface_current_relative"])
def test_each_original_solver_limit_including_conservation_is_binding(frozen, metric):
    limits = frozen["request"]["certificate_limits"]
    settings = {
        "maximum_scaled_nonlinear_residual": limits["nonlinear_residual"],
        "maximum_charge_balance_relative_error": limits["charge_balance_relative"],
        "maximum_all_face_current_spread_relative": limits["all_face_current_relative"],
        "maximum_two_sided_interface_total_current_relative_error": limits["interface_current_relative"],
    }
    values = {name: 0. for name in ("nonlinear_residual", "charge_balance_relative", "all_face_current_relative", "interface_current_relative")}
    system = SimpleNamespace(
        solver_current_metrics=lambda *args: (None, None, None, None, values["all_face_current_relative"], values["interface_current_relative"]),
        charge_balance_metrics=lambda *args: (0., values["charge_balance_relative"]),
    )
    values[metric] = limits[metric]
    result = diagnostic.original_acceptance_metrics(system, None, None, 1., np.array([values["nonlinear_residual"]]), settings)
    assert result["all_original_solver_gates_passed"]
    values[metric] = float(np.nextafter(limits[metric], np.inf))
    result = diagnostic.original_acceptance_metrics(system, None, None, 1., np.array([values["nonlinear_residual"]]), settings)
    assert not result["all_original_solver_gates_passed"]
    assert result["passed"][metric] is False


def test_ulp_report_counts_signed_zero_and_adjacent_float_words():
    expected = np.array([0., 1., -1.])
    actual = np.array([-0., np.nextafter(1., np.inf), np.nextafter(-1., -np.inf)])
    compared = diagnostic.array_comparison(actual, expected)
    assert compared["different_words"] == 3
    assert compared["maximum_ulp"] == 1
    assert not compared["exact_bytes"]


def test_diagnostic_scope_rejects_accidental_dc_and_native_step_calls():
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as state
    from perovskite_sim.experiments import interface_defect_transient as transient

    with diagnostic.forbid_trajectory_work():
        for function in (state.prepare_common_state, state.solve_r1_dc, transient._solve_step):
            with pytest.raises(AssertionError, match="cannot prepare DC or advance time"):
                function()

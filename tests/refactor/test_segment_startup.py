"""Real-input request checks and synthetic constructor/reader discriminants.

No IDA import or native trajectory is used. Synthetic statistics test readback
rejection, not the physical or numerical validity of an initial step.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from scripts.benchmarks import coupled_device_prototype as device
from scripts.benchmarks.native_readback import (
    HistoryVerificationError, SegmentStartupCheck, segment_startup_controls,
)


@pytest.fixture(scope="module")
def requests():
    bound_inputs = os.environ.get("LIFT_PREVIOUS_REQUESTS")
    if bound_inputs is None:
        pytest.skip("real-input SegmentStartup checks require LIFT_PREVIOUS_REQUESTS; absence is not qualification")
    paths = json.loads(Path(bound_inputs).read_text())
    previous = json.loads(Path(paths["DynamicAcceptorIonPublicDeviceV1"]).read_text())
    model = device.AffineCoupledSlab(device.SlabDefinition(**previous["numeric_packet"]["definition"]), 8)
    mapping = device.AffineVoltageMap(model)
    segments = tuple(device.ProtocolSegment(**row) for row in previous["segments"])
    options = dict(nonlin_conv_coef=1e-8, nonlin_guard="first-correction-wrms-v1",
                   first_step=7.8125e-7, time_weight_kappa=1024)
    parent = device.prepare_voltage_lift_native_request(mapping, segments, previous, **options)
    selected = device.prepare_voltage_lift_native_request(
        mapping, segments, previous, **options,
        segment_startup_overrides={"slow_state_hold": {"first_step": 0.0}})
    return mapping, segments, previous, options, parent, selected


def test_absent_and_explicit_zero_preserve_the_physical_request(requests):
    mapping, segments, previous, options, parent, selected = requests
    default = device.prepare_voltage_lift_native_request(
        mapping, segments, previous, **options, segment_startup_overrides=None)
    assert default == parent and "segment_startup_policy" not in parent
    for request in (parent, selected):
        device.validate_voltage_lift_native_request(mapping, segments, request)
    changed = {key for key in selected if selected[key] != parent.get(key)}
    assert changed == {"segment_startup_policy", "preparation_context_sha256", "initial_preparation"}
    assert selected["controls"] == parent["controls"]
    assert selected["previous_controls"] == previous["controls"]
    assert selected["segments"][-1]["end"] == 9.2
    assert len(selected["controls"]["atol"]) == 45
    assert sum(map(len, selected["observation_times"].values())) == 131
    assert set(selected["quadrature"]) == {"8", "16", "32"}
    for key in ("point_identity", "raw_z_hex", "raw_zdot_hex", "inputs_hex", "input_rates_hex",
                "desired_physical_tangent_hex", "mapped_physical_rate_words_hex",
                "desired_tangent_residual_SI", "represented_rate_residual_SI"):
        assert selected["initial_preparation"][key] == parent["initial_preparation"][key], key
    assert device.digest(selected) != device.digest(parent)
    assert selected["initial_preparation"]["request_sha256"] == selected["preparation_context_sha256"]


@pytest.mark.parametrize("overrides", [
    {}, True, [], {"voltage_ramp": {"first_step": 0.0}},
    {"slow_state_hold": {}}, {"slow_state_hold": {"max_step": 0.0}},
    {"slow_state_hold": {"first_step": False}}, {"slow_state_hold": {"first_step": -0.0}},
    {"slow_state_hold": {"first_step": -1.0}}, {"slow_state_hold": {"first_step": 1e-10}},
    {"slow_state_hold": {"first_step": float("inf")}},
    {"slow_state_hold": {"first_step": "0.0"}},
    {"slow_state_hold": {"first_step": 0.0, "rtol": 1e-12}},
    {"slow_state_hold": {"first_step": 0.0}, "dark_equilibrium_hold": {"first_step": 0.0}},
])
def test_invalid_policy_rejected_by_constructor(requests, overrides):
    mapping, segments, previous, options, _, _ = requests
    with pytest.raises(device.ContractError, match="invalid_segment_startup"):
        device.prepare_voltage_lift_native_request(
            mapping, segments, previous, **options, segment_startup_overrides=overrides)


def test_tampered_policy_and_unrelated_controls_rejected(requests):
    mapping, segments, _, _, _, selected = requests
    policy = selected["segment_startup_policy"]
    for field in policy:
        altered = deepcopy(selected)
        altered["segment_startup_policy"][field] = {} if field == "overrides" else "forged"
        with pytest.raises(device.ContractError, match="segment_startup"):
            device.validate_voltage_lift_native_request(mapping, segments, altered)
        with pytest.raises(HistoryVerificationError, match="segment_startup"):
            SegmentStartupCheck(altered)
    for malformed in (None, {}, [], True):
        altered = deepcopy(selected)
        altered["segment_startup_policy"] = malformed
        with pytest.raises(device.ContractError, match="segment_startup"):
            device.validate_voltage_lift_native_request(mapping, segments, altered)
    altered = deepcopy(selected)
    altered["segment_startup_overrides"] = {"slow_state_hold": {"first_step": 0.0}}
    with pytest.raises(device.ContractError, match="unbound_override"):
        device.validate_voltage_lift_native_request(mapping, segments, altered)
    for field, value in (("first_step", 0.0), ("max_step", 0.2), ("rtol", 1e-12),
                         ("nonlin_conv_coef", 1e-8), ("max_order", 1)):
        altered = deepcopy(selected)
        altered["controls"][field] = value
        with pytest.raises(device.ContractError, match="native_controls_changed"):
            device.validate_voltage_lift_native_request(mapping, segments, altered)
    altered = deepcopy(selected)
    altered["controls"]["atol"][0] *= 2
    with pytest.raises(device.ContractError, match="native_controls_changed|time_weights_binding"):
        device.validate_voltage_lift_native_request(mapping, segments, altered)
    with pytest.raises(device.ContractError, match="unbound_segment"):
        device._voltage_lift_initialization_controls(
            selected, device.ProtocolSegment(**dict(asdict(segments[-1]), end=9.3)))


def constructor(request, segment, ordinal):
    captured = {}

    def fake_ida(residual, **kwargs):
        captured.update(kwargs)
        return captured

    marker = object()
    solver, receipt = device._voltage_lift_segment_solver(
        fake_ida, request, segment, ordinal, marker, marker, marker, [0])
    assert solver is captured
    assert captured.pop("jacfn") is marker and captured.pop("sparsity") is marker
    assert captured.pop("algebraic_idx") == [0]
    return captured, receipt


def stats(segment, ordinal, phase, steps=0, h0=0.0):
    return {"kind": "native_statistics", "segment_id": segment.id,
            "logical_initialization_index": ordinal, "phase": phase,
            "raw_statistics": {"observation_owner": "synthetic-owner", "observation_generation": ordinal,
                               "num_steps": steps, "initial_step": h0}}


def test_actual_constructor_kwargs_and_independent_reader(requests):
    _, segments, _, _, parent, selected = requests
    before = deepcopy(selected)
    check = SegmentStartupCheck(selected)
    for ordinal, segment in enumerate(segments, 1):
        original, absent = constructor(parent, segment, ordinal)
        assert absent is None and original == parent["controls"]
        effective, receipt = constructor(selected, segment, ordinal)
        expected = dict(parent["controls"], first_step=0.0 if ordinal == 3 else 7.8125e-7)
        assert effective == expected == segment_startup_controls(selected, asdict(segment))
        assert receipt["constructor_controls"] == effective
        check.consume(receipt)
        check.consume(stats(segment, ordinal, "initialization_return"))
        check.consume(stats(segment, ordinal, "after_onestep", 1, 1e-15 if ordinal == 3 else 7.8125e-7))
    assert selected == before
    result = check.finish(complete=True)
    assert len(result["initializations"]) == 3
    assert result["initializations"][-1]["requested_first_step_hex"] == "0x0.0p+0"
    assert result["native_requested_first_step_getter"] == "unavailable"


def test_reader_rejects_missing_forged_or_misapplied_controls(requests):
    _, segments, _, _, parent, selected = requests
    _, receipt = constructor(selected, segments[0], 1)
    for field, value in (("segment_id", "slow_state_hold"), ("constructor_controls_sha256", "0"*64),
                         ("requested_first_step_hex", "0x0.0p+0"),
                         ("segment_startup_policy_sha256", "0"*64)):
        altered = dict(receipt, **{field: value})
        with pytest.raises(HistoryVerificationError, match="constructor_binding"):
            SegmentStartupCheck(selected).consume(altered)
    altered = deepcopy(receipt)
    altered["constructor_controls"]["first_step"] = 0.0
    altered["constructor_controls_sha256"] = device.digest(altered["constructor_controls"])
    altered["requested_first_step_hex"] = "0x0.0p+0"
    with pytest.raises(HistoryVerificationError, match="constructor_binding"):
        SegmentStartupCheck(selected).consume(altered)
    with pytest.raises(HistoryVerificationError, match="unexpected_constructor"):
        SegmentStartupCheck(parent).consume(receipt)
    with pytest.raises(HistoryVerificationError, match="missing_constructor"):
        SegmentStartupCheck(selected).consume(stats(segments[0], 1, "initialization_return"))
    check = SegmentStartupCheck(selected)
    check.consume(receipt)
    check.consume(stats(segments[0], 1, "initialization_return"))
    with pytest.raises(HistoryVerificationError, match="explicit_step_not_applied"):
        check.consume(stats(segments[0], 1, "after_onestep", 1, 1e-15))
    check.consume(stats(segments[0], 1, "after_onestep", 0, 0.0))
    partial = check.finish(complete=False)
    assert partial["initializations"][0]["actual_initial_step_hex"] is None
    with pytest.raises(HistoryVerificationError, match="incomplete_application"):
        check.finish(complete=True)

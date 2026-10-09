"""Segment startup policy, constructor controls and independent readback."""

from dataclasses import asdict, replace
import json

import pytest


def _full_protocol_startup_case():
    """Manufactured binding shapes only; no model, Point or solver is made."""
    from scripts.benchmarks import coupled_device_prototype as d

    segments = tuple(d.ProtocolSegment(name, start, end, (0., 0.), (0., 0.))
                     for name, start, end in (("dark_equilibrium_hold", 0., .1),
                                              ("voltage_ramp", .1, .2),
                                              ("slow_state_hold", .2, 9.2)))
    request = dict(case_id="DynamicAcceptorIonPublicDeviceV1", prior_request_sha256="a"*64,
                   map_identity="manufactured-map", segments=[asdict(s) for s in segments],
                   controls=dict(first_step=7.8125e-7, rtol=1e-6,
                                 atol=[(i+1)*1e-10 for i in range(45)], max_step=.1,
                                 nonlin_conv_coef=1.024e-5, nonlin_guard="first-correction-wrms-v1",
                                 nonlin_trace_capacity=4096), time_weight_policy={"kappa": 1024})
    return d, json.loads(json.dumps(request)), segments


def _full_protocol_startup_constructor(d, request, segment, ordinal):
    captured, marker = {}, object()

    def constructor_spy(residual, **kwargs):
        assert residual is marker
        captured.update(kwargs)
        return marker

    returned, receipt = d._voltage_lift_segment_solver(
        constructor_spy, request, segment, ordinal, marker, marker, marker, [0])
    assert returned is marker
    assert captured.pop("jacfn") is marker and captured.pop("sparsity") is marker
    assert captured.pop("algebraic_idx") == [0]
    return captured, receipt


def _full_protocol_startup_stats(segment, ordinal, phase, h0=0., steps=0):
    return dict(kind="native_statistics", segment_id=segment.id, phase=phase,
                logical_initialization_index=ordinal,
                raw_statistics=dict(observation_owner="manufactured-owner", observation_generation=ordinal,
                                    num_steps=steps, initial_step=h0, last_step=2.0**-40))


def test_full_protocol_startup_absent_and_zero_identity():
    from scripts.benchmarks.native_readback import segment_startup_controls

    d, request, segments = _full_protocol_startup_case()
    before = d.digest(request)
    for ordinal, segment in enumerate(segments, 1):
        controls, receipt = _full_protocol_startup_constructor(d, request, segment, ordinal)
        assert receipt is None and controls == request["controls"]
        assert segment_startup_controls(request, request["segments"][ordinal-1]) == controls
    assert d.digest(request) == before
    zero = d._voltage_lift_segment_startup(request, {"slow_state_hold": {"first_step": 0.0}})
    expected = dict(schema="solarlab.voltage-lift-segment-startup.v1",
                    ancestor_request_sha256=request["prior_request_sha256"],
                    base_controls_sha256=d.digest(request["controls"]),
                    segments_sha256=d.digest(request["segments"]),
                    time_weight_policy_sha256=d.digest(request["time_weight_policy"]),
                    overrides={"slow_state_hold": {"first_step": 0.0}},
                    application="fresh_segment_initialization_before_first_solve")
    assert zero == expected and d.digest(zero) == d.digest(expected)
    request["segment_startup_policy"] = zero
    controls, receipt = _full_protocol_startup_constructor(d, request, segments[-1], 3)
    assert controls == dict(request["controls"], first_step=0.0)
    assert receipt["requested_first_step_hex"] == "0x0.0p+0"
    assert segment_startup_controls(request, request["segments"][-1]) == controls
    assert d.digest({k: v for k, v in request.items() if k != "segment_startup_policy"}) == before


def test_full_protocol_startup_explicit_hold_constructor_and_readback():
    from scripts.benchmarks.native_readback import SegmentStartupCheck, segment_startup_controls

    d, request, segments = _full_protocol_startup_case()
    policy = d._voltage_lift_segment_startup(request, {"slow_state_hold": {"first_step": 2.0**-32}})
    zero = d._voltage_lift_segment_startup(request, {"slow_state_hold": {"first_step": 0.0}})
    assert {k for k in policy if policy[k] != zero[k]} == {"overrides"}
    assert d.digest(policy) != d.digest(zero)
    request["segment_startup_policy"] = policy
    before, check = d.digest(request), SegmentStartupCheck(request)
    for ordinal, segment in enumerate(segments, 1):
        controls, receipt = _full_protocol_startup_constructor(d, request, segment, ordinal)
        expected = dict(request["controls"], first_step=2.0**-32 if ordinal == 3 else 7.8125e-7)
        assert controls == expected == segment_startup_controls(request, request["segments"][ordinal-1])
        assert receipt["request_sha256"] == before
        check.consume(receipt)
        check.consume(_full_protocol_startup_stats(segment, ordinal, "initialization_return"))
        check.consume(_full_protocol_startup_stats(segment, ordinal, "after_onestep", controls["first_step"], 1))
    result = check.finish(complete=True)
    assert d.digest(request) == before and len(result["initializations"]) == 3
    assert result["initializations"][-1]["actual_initial_step_hex"] == (2.0**-32).hex()
    assert result["native_requested_first_step_getter"] == "unavailable"


def test_full_protocol_startup_invalid_or_unbound_policy():
    from copy import deepcopy
    from math import inf, nan, nextafter
    from scripts.benchmarks.native_readback import HistoryVerificationError, segment_startup_controls

    d, request, segments = _full_protocol_startup_case()
    selected = {"slow_state_hold": {"first_step": 2.0**-32}}
    policy = d._voltage_lift_segment_startup(request, selected)
    invalid = [{"slow_state_hold": {"first_step": v}} for v in
               (False, -0.0, -1., 1e-10, nextafter(2.0**-32, inf), nan, inf, "0.0")]
    invalid += [{}, [], {"voltage_ramp": {"first_step": 2.0**-32}},
                {"slow_state_hold": {"first_step": 2.0**-32, "rtol": 1e-12}},
                dict(selected, dark_equilibrium_hold={"first_step": 2.0**-32})]
    for overrides in invalid:
        with pytest.raises(d.ContractError, match="invalid_segment_startup"):
            d._voltage_lift_segment_startup(request, overrides)
        altered = dict(request, segment_startup_policy=dict(policy, overrides=overrides))
        with pytest.raises(HistoryVerificationError, match="segment_startup"):
            segment_startup_controls(altered, altered["segments"][-1])
    request["segment_startup_policy"] = policy
    for field in ("schema", "ancestor_request_sha256", "base_controls_sha256", "segments_sha256",
                  "time_weight_policy_sha256", "application"):
        altered = deepcopy(request)
        altered["segment_startup_policy"][field] = "tampered"
        with pytest.raises(d.ContractError, match="segment_startup_binding"):
            d._voltage_lift_initialization_controls(altered, segments[-1])
        with pytest.raises(HistoryVerificationError, match="segment_startup"):
            segment_startup_controls(altered, altered["segments"][-1])
    for mutate in (lambda r: r.update(prior_request_sha256="b"*64),
                   lambda r: r["controls"].update(rtol=2e-6),
                   lambda r: r["time_weight_policy"].update(kappa=512),
                   lambda r: r["segments"][-1].update(end=9.3)):
        altered = deepcopy(request)
        mutate(altered)
        with pytest.raises(d.ContractError, match="segment_startup"):
            d._voltage_lift_initialization_controls(altered, segments[-1])
        with pytest.raises(HistoryVerificationError, match="segment_startup"):
            segment_startup_controls(altered, altered["segments"][-1])
    with pytest.raises(d.ContractError, match="unbound_segment"):
        d._voltage_lift_initialization_controls(request, replace(segments[-1], end=9.3))
    with pytest.raises(HistoryVerificationError, match="unbound_override"):
        segment_startup_controls(dict(request, segment_startup_overrides=selected), request["segments"][-1])


def test_full_protocol_startup_reader_rejects_control_or_actual_h0_mismatch():
    from copy import deepcopy
    from scripts.benchmarks.native_readback import HistoryVerificationError, SegmentStartupCheck

    d, request, segments = _full_protocol_startup_case()
    request["segment_startup_policy"] = d._voltage_lift_segment_startup(
        request, {"slow_state_hold": {"first_step": 2.0**-32}})
    for wrong_h0 in (0.0, 2.0**-40, 7.8125e-7):
        check = SegmentStartupCheck(request)
        for ordinal, segment in enumerate(segments, 1):
            controls, receipt = _full_protocol_startup_constructor(d, request, segment, ordinal)
            if ordinal == 3:
                forged = deepcopy(receipt)
                forged["constructor_controls"]["first_step"] = wrong_h0
                forged["constructor_controls_sha256"] = d.digest(forged["constructor_controls"])
                forged["requested_first_step_hex"] = wrong_h0.hex()
                with pytest.raises(HistoryVerificationError, match="constructor_binding"):
                    check.consume(forged)
            check.consume(receipt)
            check.consume(_full_protocol_startup_stats(segment, ordinal, "initialization_return"))
            row = _full_protocol_startup_stats(segment, ordinal, "after_onestep", controls["first_step"], 1)
            if ordinal == 3:
                row["raw_statistics"]["initial_step"] = wrong_h0
                with pytest.raises(HistoryVerificationError, match="actual_initial_step|explicit_step_not_applied"):
                    check.consume(row)
            else:
                check.consume(row)

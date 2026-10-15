"""PhysicalToleranceV1 metadata only; no model or execution authority.

The native callback and independent reader perform their own weight arithmetic.
This module shares only frozen profile construction and exact option binding.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import math

from perovskite_sim.constants import K_B, Q

SCHEMA = "solarlab.physical-tolerance.v1"
WEIGHT_SCHEMA = "sksundae.ida.physical-affine-weights.v1"
ROUNDING = "RN-exact-physical-X_then-unfused-physical-EWT-v1"
PROFILES = ((1e-4, 1e-8), (1e-5, 1e-8), (1e-6, 1e-10))
CHARGE_ALLOCATION = "physical_charge.budget_C/12 * actual_interval/full_protocol / 32 per integral"
NONLINEAR = dict(nonlin_conv_coef=1e-8, nonlin_guard="first-correction-wrms-v1",
                 nonlin_trace_capacity=4096, max_nonlin_iters=4, max_conv_fails=10)


def certificate_failure_policy(*, observation="online", t1=None):
    """Explicit new-history semantics; absence keeps every historical stop rule.

    A missing reviewed T1 definition is a pending proof, never a zero tolerance
    or a pass. This metadata does not admit a run or change IDA controls.
    """
    require(type(observation) is str and observation in {"online", "sealed-history"}, "certificate_observation_mode")
    if t1 is not None:
        require(type(t1) is dict and set(t1) == {"schema", "sha256"}
                and t1["schema"] in {"solarlab.g2-route.t1-gatespec.v2", "solarlab.g2-route.t1-gatespec.v3"}
                and type(t1["sha256"]) is str and len(t1["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in t1["sha256"]), "reviewed_T1_definition_required")
    return {"schema": "solarlab.certificate-failure-policy.v1",
            "scientific_failure": "latch-and-continue", "safety_failure": "stop",
            "observation": observation, "T1": t1,
            "legacy_charge_role": "retained-diagnostic",
            "physical_acceptance": "requires-all-proofs"}


def validate_certificate_policy(request):
    if "certificate_failure_policy" not in request:
        return None
    value = request["certificate_failure_policy"]
    require(type(value) is dict and digest(value) == digest(certificate_failure_policy(
        observation=value.get("observation"), t1=value.get("T1"))), "certificate_policy_binding")
    require("diagnostic_interval" not in request, "certificate_complete_protocol_only")
    return value


def run_window_policy(request, last_segment):
    """A declared prefix of the original protocol, ending on a retained side."""
    require(validate_certificate_policy(request) is not None, "window_requires_certificate_policy")
    segments = request["segments"]
    matches = [i for i, segment in enumerate(segments) if segment["id"] == last_segment]
    require(len(matches) == 1 and matches[0] < len(segments)-1, "window_segment_boundary")
    count = matches[0]+1
    return {"schema": "solarlab.protocol-run-window.v1", "last_segment": last_segment,
            "segment_count": count, "protocol_sha256": digest(segments),
            "start_hex": float(segments[0]["start"]).hex(), "end_hex": float(segments[count-1]["end"]).hex(),
            "full_end_hex": float(segments[-1]["end"]).hex(),
            "sample_records": sum(len(request["observation_times"][s["id"]]) for s in segments[:count]),
            "initial_state": "original-protocol-initial-state", "multistep_history_restored": False}


def run_segment_count(request):
    window = request.get("execution_window")
    if "execution_window" not in request:
        return len(request["segments"])
    require(type(window) is dict and window == run_window_policy(request, window.get("last_segment")),
            "window_binding")
    return window["segment_count"]


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError("physical_tolerance_" + reason)


def scalar(value, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value), "finite_scalar")
    require(not positive or value > 0, "positive_scalar")
    return float(value)


def make_policy(packet, profile):
    require(type(profile) is dict and set(profile) == {"r", "alpha"}, "profile_fields")
    r, alpha = (scalar(profile[k], positive=True) for k in ("r", "alpha"))
    require((r, alpha) in PROFILES, "unsupported_profile")
    scales = [scalar(v, positive=True) for v in packet["column_scaling"]]
    size = len(scales)
    require(size > 0, "empty_layout")
    offsets = packet["variable_offsets"]
    require(set(offsets) in ({"n_m3", "p_m3", "phi_V"}, {"c_m3", "f", "n_m3", "p_m3", "phi_V"}), "fields")
    fields = packet["physical_reference_fields"]
    reference = packet["initial_reference"]["payload"]["fields"]
    temperature = scalar(packet["definition"]["temperature"], positive=True)
    absolute, roots, covered = [None]*size, [[None]*size for _ in range(2)], set()
    for name, bounds in offsets.items():
        require(type(bounds) in (list, tuple) and len(bounds) == 2, "offsets")
        lo, hi = bounds
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= size, "offsets")
        require(not (set(range(lo, hi)) & covered), "overlapping_offsets")
        covered.update(range(lo, hi))
        pair = fields[name]
        require(set(pair) == {"high", "low"} and all(len(pair[k]) == hi-lo for k in pair), "reference_shape")
        for j, i in enumerate(range(lo, hi)):
            h, l = (scalar(pair[k][j]) for k in ("high", "low"))
            require(h.hex() == reference[name]["high"][j] and l.hex() == reference[name]["low"][j], "reference_words")
            roots[0][i], roots[1][i] = h.hex(), l.hex()
            value = Fraction(h) + Fraction(l)
            if name in ("n_m3", "p_m3", "c_m3"):
                require(value >= 0 and (name == "c_m3" or value > 0), "reference_density")
                floor = max(1e-6, float(Fraction(alpha)*value))
            elif name == "f":
                require(0 <= value <= 1, "reference_occupancy")
                floor = alpha
            else:
                floor = alpha * (K_B * temperature / Q)
            absolute[i] = scalar(floor, positive=True)
    require(covered == set(range(size)), "layout_coverage")
    return {"schema": SCHEMA, "profile": {"r": r, "alpha": alpha}, "size": size,
            "numeric_packet_sha256": digest(packet), "reference_words_hex": roots,
            "scale_hex": [v.hex() for v in scales], "atol_SI": absolute,
            "floor_basis": "frozen-initial-reference; fixed-electrode-gauge; no evolving-state floor",
            "nonlinear_controls": dict(NONLINEAR), "nonlinear_effective_threshold_equivalence": False,
            "native_execution_authorized": False,
            "signed_balance_integration": "source-bound-signed-physical-charge-v1"}


def controls(previous, policy):
    values = [a / float.fromhex(s) for a, s in zip(policy["atol_SI"], policy["scale_hex"], strict=True)]
    require(all(math.isfinite(a) and a > 0 for a in values), "scaled_absolute_range")
    result = dict(previous, rtol=policy["profile"]["r"], atol=values, **NONLINEAR)
    result.pop("constraints_idx", None)
    result.pop("constraints_type", None)
    return result


def route_controls_policy(request, *, case_id, source_commit, run_plan_sha256):
    """One prospective G2-R max-step change; all physical controls stay bound.

    The original preparation/previous-controls records remain historical data.
    Git/source and plan admission are checked by the route, not created here.
    """
    names = ("Coarse", "Main", "Strict")
    expected_ids = {f"B_N8_{name}_{stage}" for name in names for stage in ("Probe", "Full")}
    expected_ids.add("B_N16_Strict_Full")
    require(case_id in expected_ids, "route_case")
    mesh = 16 if "N16" in case_id else 8
    profile = dict(zip(("r", "alpha"), PROFILES[names.index(case_id.split("_")[2])]))
    require(request["numeric_packet"]["intervals"] == mesh
            and request["numeric_packet"]["nodes"] == mesh+1
            and request["physical_tolerance_policy"]["size"] == 5*(mesh+1)
            and digest(request["physical_tolerance_policy"]["profile"]) == digest(profile), "route_profile_geometry")
    for value, size in ((source_commit, 40), (run_plan_sha256, 64)):
        require(type(value) is str and len(value) == size and all(c in "0123456789abcdef" for c in value),
                "route_source_plan_pin")
    before = controls(request["previous_controls"], request["physical_tolerance_policy"])
    if "startup_step_policy" in request:
        before["first_step"] = request["startup_step_policy"]["value"]
    require(before.get("max_step") == (.02 if mesh == 16 else .1), "route_ancestral_max_step")
    return {"schema": "solarlab.g2-route.controls.v1", "case_id": case_id,
            "source_commit": source_commit, "run_plan_sha256": run_plan_sha256,
            "parent_controls_sha256": digest(before), "previous_max_step_s": before["max_step"],
            "max_step_s": .02, "other_controls_changed": False}


def validate_request(request):
    policy = request.get("physical_tolerance_policy")
    require(type(policy) is dict, "missing_policy")
    expected = make_policy(request["numeric_packet"], policy.get("profile"))
    require(policy == expected, "policy_binding")
    require(not any(k in request for k in ("time_weight_policy", "diagnostic_accuracy_policy",
                "nonlinear_control_refinement", "nonlinear_guard_policy")), "mixed_legacy_profile")
    applied = controls(request["previous_controls"], expected)
    if "startup_step_policy" in request:
        startup = request["startup_step_policy"]
        value = scalar(startup.get("value"), positive=True)
        require(startup.get("schema") == "solarlab.voltage-lift-startup-step.v1"
                and startup.get("ancestor_request_sha256") == request["prior_request_sha256"]
                and startup.get("previous_controls_sha256") == digest(request["previous_controls"])
                and startup.get("previous_value") == request["previous_controls"].get("first_step", 0.0)
                and startup.get("field") == "first_step"
                and startup.get("application") == "every_protocol_initialization"
                and (applied.get("max_step", 0) == 0 or value <= applied["max_step"]), "startup_binding")
        applied["first_step"] = value
    if "g2_route_controls" in request:
        route = request["g2_route_controls"]
        require(type(route) is dict, "route_controls_type")
        expected_route = route_controls_policy(request, case_id=route.get("case_id"),
            source_commit=route.get("source_commit"), run_plan_sha256=route.get("run_plan_sha256"))
        require(digest(route) == digest(expected_route), "route_controls_binding")
        applied["max_step"] = expected_route["max_step_s"]
    require(request["controls"] == applied, "controls_binding")
    require(request["weight_certificate"] == expected, "weight_certificate")
    require(request["budgets"] == request["original_budgets"], "historical_budgets_changed")
    require(request.get("native_admission") is False, "execution_authority")
    return expected


def require_balance_integration(request):
    """Require the new balance contract; exact-source Root admission is separate."""
    if "physical_tolerance_policy" in request:
        validate_request(request)
        observer = request.get("interval_observation")
        require(type(observer) is dict and observer.get("physical_charge") == physical_charge_policy(request)
                and observer.get("quadrature_error_allocation") == CHARGE_ALLOCATION
                and "diagnostic_interval" not in request
                and "segment_frame_policy" in request and "initial_segment_frame" in request,
                "signed_balance_integration_pending")
        forms = request.get("physical_charge_forms")
        require(type(forms) is dict and set(forms) == {"body_charge", "metal_charge"}
                and all(type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                        for v in forms.values()), "physical_charge_form_binding")


def weight_option(request, frame, segment):
    policy = validate_request(request)
    require(frame["size"] == policy["size"] and frame["parent_map_identity"] == request["map_identity"], "frame_binding")
    segment_id = digest(segment)
    require(frame["segment_sha256"] == segment_id
            and segment_id in {digest(row) for row in request["segments"]}, "segment_binding")
    mapping = request["voltage_lift_map"]
    require(mapping["columns_hex"] == policy["scale_hex"], "map_scale")
    start, end = scalar(segment["start"]), scalar(segment["end"])
    require(start < end and start <= float.fromhex(frame["t0_hex"]) <= end, "frame_time")
    pairs = [segment["voltage"], segment["photons"]]
    starts = [scalar(pair[0]) for pair in pairs]
    ends = [scalar(pair[1]) for pair in pairs]
    rates = [(b-a)/(end-start) for a, b in zip(starts, ends)]
    return {"schema": WEIGHT_SCHEMA, "frame_identity": digest(frame),
            **{k: frame[k] for k in ("t0_hex", "size", "q0_words_hex", "v0_words_hex")},
            "weight_rounding": ROUNDING, "profile_identity": digest(policy),
            "rtol_hex": policy["profile"]["r"].hex(),
            "atol_physical_hex": [v.hex() for v in policy["atol_SI"]],
            "scale_hex": policy["scale_hex"], "reference_words_hex": policy["reference_words_hex"],
            "lift_hex": mapping["lift_hex"], "reference_inputs_hex": mapping["reference_inputs_hex"],
            "protocol_start_hex": start.hex(), "protocol_end_hex": end.hex(),
            "inputs_start_hex": [v.hex() for v in starts], "inputs_end_hex": [v.hex() for v in ends],
            "input_rates_hex": [v.hex() for v in rates]}


def segment_startup_policy(request, overrides):
    policy = validate_request(request)
    require(type(overrides) is dict and set(overrides) == {"slow_state_hold"}
            and type(overrides["slow_state_hold"]) is dict
            and set(overrides["slow_state_hold"]) == {"first_step"}, "segment_startup_fields")
    value = overrides["slow_state_hold"]["first_step"]
    require(type(value) in (int, float) and value in (0, 2.0**-32)
            and math.copysign(1, value) == 1, "segment_startup_value")
    require(request["case_id"] == "DynamicAcceptorIonPublicDeviceV1"
            and [s["id"] for s in request["segments"]] == ["dark_equilibrium_hold", "voltage_ramp", "slow_state_hold"], "segment_startup_scope")
    return {"schema": "solarlab.physical-segment-startup.v1", "ancestor_request_sha256": request["prior_request_sha256"],
            "base_controls_sha256": digest(request["controls"]), "segments_sha256": digest(request["segments"]),
            "physical_tolerance_policy_sha256": digest(policy),
            "overrides": {"slow_state_hold": {"first_step": float(value)}},
            "application": "fresh_segment_initialization_before_first_solve"}


def physical_charge_policy(request):
    """New independent balance budget from frozen physical inputs, never old debt.

    Rational strings are serialization only; consumers use Fraction directly.
    Request/frame/operation linkage belongs to the existing observation record.
    """
    policy = validate_request(request)
    definition = request["numeric_packet"]["definition"]
    area = scalar(definition["area"], positive=True)
    length = scalar(definition["length"], positive=True)
    densities = [scalar(definition[k]) for k in ("ni", "ion_initial", "trap_density")]
    require(all(v >= 0 for v in densities) and max(densities) > 0, "charge_reference_density")
    qref = Fraction(Q)*Fraction(area)*Fraction(length)*max(map(Fraction, densities))
    budget = qref / 10**8
    return {"schema": "solarlab.physical-charge-policy.v1",
            "physical_tolerance_policy_sha256": digest(policy),
            "numeric_packet_sha256": digest(request["numeric_packet"]),
            "original_budgets_sha256": digest(request["original_budgets"]),
            "Qref_C": str(qref), "budget_C": str(budget), "budget_fraction": [1, 10**8],
            "channels": ["device", "left_metal", "right_metal"],
            "integration_uncertainty_fraction": [1, 3],
            "reference_basis": {"q_C_hex": float(Q).hex(), "area_m2_hex": area.hex(),
                                "length_m_hex": length.hex(),
                                "density_hex": [v.hex() for v in densities]},
            "historical_budgets_changed": False, "native_execution_authorized": False}


def physical_charge_budgets(request):
    policy = physical_charge_policy(request)
    return (Fraction(policy["budget_C"]),)*3


def physical_charge_error_allocation(request, duration):
    """Prospective allocation only; actual interval and cumulative errors gate."""
    duration = Fraction(duration)
    first, last = request["segments"][0]["start"], request["segments"][-1]["end"]
    full = Fraction(scalar(last))-Fraction(scalar(first))
    require(first == 0 and 0 < duration <= full, "physical_charge_allocation_clock")
    return physical_charge_budgets(request)[0]/12*duration/full/32

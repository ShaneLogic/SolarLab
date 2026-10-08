"""Bounded P02 mesh/control metadata; no model imports or execution authority.

Only the original two slab cases and seven N16-centered axes are described.
New per-field denominators are bounded by the minimum CURRENT N8 applied
physical denominator, then tightened by the declared axis factor. The old plan
rtol/atol numbers are provenance, never replacements for those applied weights.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import math
import sys

SCHEMA = "solarlab.p02-qualification-mesh-controls.v1"
ORIGINS = {
    "S0NeutralPublicDeviceV1": "bc24e4b4add18ffaaab5469bb9875ef7fb7d84b7ff92b48fd70e1354cd3a4e5d",
    "DynamicAcceptorIonPublicDeviceV1": "cd004d42e53e56b6a69e4ab7667848054bae55af33a6b1230865d761b230a3c0",
}
AXES = {
    "base": (16, 0, 0), "mesh32": (32, 0, 0), "mesh64": (64, 0, 0),
    "tolerance_middle": (16, 1, 0), "tolerance_tight": (16, 2, 0),
    "step_middle": (16, 0, 1), "step_tight": (16, 0, 2),
}
GRID_WORDS = {
    16: "9bf4f40d7d194571210a3b2e6767eb1519ced739af0b614f06fa1ea5901ee855",
    32: "c531d6319563c0f77460e5b6e7785f27e713d066192d2d80fcb9a93d2d65944d",
    64: "9c3d62a65d55a50ba51f82d4dc4f720daa99c0e37e3e88ddd195070ae8dc1bb1",
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError("qualification_" + reason)


def downward(value):
    value = Fraction(value)
    require(value > 0, "nonpositive_weight")
    rounded = float(value)
    require(math.isfinite(rounded) and rounded >= sys.float_info.min, "unrepresentable_weight")
    if Fraction(rounded) > value:
        rounded = math.nextafter(rounded, 0.0)
    require(rounded >= sys.float_info.min, "unrepresentable_weight")
    return rounded


def make_policy(origin, intervals, axis):
    require(type(origin) is dict and origin.get("case_id") in ORIGINS, "unknown_origin")
    require(digest(origin) == ORIGINS[origin["case_id"]], "unbound_origin")
    require(type(intervals) is int and type(axis) is str and axis in AXES
            and intervals == AXES[axis][0], "unsupported_mesh_axis")
    _, tolerance, step = AXES[axis]
    dynamic = origin["case_id"] == "DynamicAcceptorIonPublicDeviceV1"
    return {
        "schema": SCHEMA, "origin": json.loads(json.dumps(origin)),
        "intervals": intervals, "nodes": intervals + 1, "axis": axis,
        "coordinates": (5 if dynamic else 3)*(intervals + 1),
        "x_words_sha256": GRID_WORDS[intervals],
        "tolerance_factor": [1, 10**tolerance],
        "plan_basis": {"rtol": (1e-4, 1e-5, 1e-6)[tolerance],
                       "component_atol": (1e-8, 1e-9, 1e-10)[tolerance]},
        "max_step": ((0.02, 0.01, 0.005) if dynamic else (2e-8, 1e-8, 5e-9))[step],
        "denominator_rule": "per-field minimum current N8 S*atol and current rtol, times exact axis factor; existing affine/voltage triangle proofs may only tighten",
        "pre_time_multiplier": 1024 if dynamic else 1,
        "scientific_thresholds_changed": False,
        "native_execution_authorized": False,
    }


def validate_policy(policy):
    require(type(policy) is dict, "policy_type")
    expected = make_policy(policy.get("origin"), policy.get("intervals"), policy.get("axis"))
    require(digest(policy) == digest(expected), "policy_binding")
    return expected


def layout_offsets(policy):
    policy = validate_policy(policy)
    n = policy["nodes"]
    dynamic = policy["origin"]["case_id"] == "DynamicAcceptorIonPublicDeviceV1"
    fields = ("c_m3", "f", "n_m3", "p_m3", "phi_V") if dynamic else ("n_m3", "p_m3", "phi_V")
    variables = {name: [i*n, (i+1)*n] for i, name in enumerate(fields)}
    sizes = [("a_n", n-2), ("b_p", n-2)]
    if dynamic:
        sizes += [("c_ion", n), ("d_trap", n)]
    sizes += [("x_poisson", n-2), ("y_phi_contact", 2), ("z_n_contact", 2), ("zz_p_contact", 2)]
    rows, offset = {}, 0
    for name, count in sizes:
        rows[name] = [offset, offset+count]
        offset += count
    require(offset == policy["coordinates"], "nonsquare_layout")
    return variables, rows


def validate_layout(packet, policy):
    policy = validate_policy(policy)
    origin = policy["origin"]
    variables, rows = layout_offsets(policy)
    n, size = policy["nodes"], policy["coordinates"]
    require(type(packet) is dict and type(packet.get("intervals")) is int
            and packet["intervals"] == policy["intervals"]
            and type(packet.get("nodes")) is int and packet["nodes"] == n
            and packet.get("variable_offsets") == variables
            and packet.get("equation_offsets") == rows, "layout_binding")
    definition = {k: v for k, v in packet["definition"].items() if k != "source_path"}
    require(digest(definition) == digest(origin["definition"]), "physical_definition_changed")
    for key, length in (("x_m", n), ("cell_edges_m", n+1), ("volumes_m3", n),
                        ("face_areas_m2", n-1), ("column_scaling", size), ("row_scaling", size)):
        values = packet[key]
        require(type(values) is list and len(values) == length
                and all(type(v) in (int, float) and math.isfinite(v) for v in values), "array_shape_" + key)
    x, edges, area = packet["x_m"], packet["cell_edges_m"], definition["area"]
    require(digest(x) == policy["x_words_sha256"]
            and x[0] == 0 and x[-1] == definition["length"]
            and all(b > a for a, b in zip(x[:-1], x[1:])), "grid_domain")
    require(edges == [0.0] + [(u+v)/2 for u, v in zip(x[:-1], x[1:])] + [definition["length"]]
            and packet["volumes_m3"] == [area*(v-u) for u, v in zip(edges[:-1], edges[1:])]
            and packet["face_areas_m2"] == [area]*(n-1), "geometric_weights_changed")
    require(all(v > 0 for v in packet["row_scaling"] + packet["volumes_m3"]), "nonpositive_geometry_or_scale")
    for field, (lo, hi) in variables.items():
        require(all(float(v).hex() == origin["field_denominator_ceilings"][field]["scale_hex"]
                    for v in packet["column_scaling"][lo:hi]), "physical_column_scale_changed")
    return size


def raw_controls(policy, packet):
    size = validate_layout(packet, policy)
    origin = policy["origin"]
    factor = Fraction(*policy["tolerance_factor"])*policy["pre_time_multiplier"]
    rtol = downward(Fraction(float.fromhex(origin["applied_rtol_hex"]))*factor)
    atols = [None]*size
    for field, (lo, hi) in packet["variable_offsets"].items():
        limit = Fraction(origin["field_denominator_ceilings"][field]["applied_atol_SI_exact"])*factor
        for i in range(lo, hi):
            atols[i] = downward(limit/Fraction(packet["column_scaling"][i]))
    dynamic = origin["case_id"] == "DynamicAcceptorIonPublicDeviceV1"
    indices, types = [], []
    for name in ["n_m3", "p_m3"] + (["c_m3", "f"] if dynamic else []):
        lo, hi = packet["variable_offsets"][name]
        indices.extend(range(lo, hi)); types.extend([2 if name in ("n_m3", "p_m3") else 1]*(hi-lo))
    controls = {"rtol": rtol, "atol": atols, "max_step": policy["max_step"], "max_order": 5,
                "max_num_steps": 200000, "max_nonlin_iters": 4, "max_conv_fails": 10,
                "first_step": 0.0, "calc_initcond": None, "linsolver": "sparse", "nthreads": 1,
                "constraints_idx": indices, "constraints_type": types}
    if dynamic:
        controls["nonlin_conv_coef"] = origin["pre_time_nonlinear_ancestor"]
    return controls


def check_common(request, policy):
    validate_layout(request["numeric_packet"], policy)
    origin = policy["origin"]
    require(request["case_id"] == origin["case_id"], "case_changed")
    for key in ("segments", "observation_times", "quadrature", "budgets", "mandatory"):
        require(digest(request[key]) == digest(origin[key]), "original_" + key + "_changed")
    if request["schema"] in ("solarlab.affine-native-request.v1", "solarlab.voltage-lift-native-request.v1"):
        require(digest(request.get("physical_domain_policy")) == digest(origin["physical_domain_policy"]),
                "physical_domain_changed")
    require(digest(request.get("qualification_policy")) == digest(policy), "missing_policy_binding")


def check_ancestry(request):
    """Verify the complete bounded raw->affine->voltage control ancestry."""
    policy = validate_policy(request["qualification_policy"])
    check_common(request, policy)
    affine = request["qualification_ancestor"]
    raw = affine["qualification_ancestor"]
    require(raw["schema"] == "solarlab.real-device-native-request.v1"
            and affine["schema"] == "solarlab.affine-native-request.v1"
            and request["schema"] == "solarlab.voltage-lift-native-request.v1", "ancestor_schema")
    for row in (raw, affine):
        check_common(row, policy)
    require(affine["prior_request_sha256"] == digest(raw)
            and request["prior_request_sha256"] == digest(affine)
            and request["previous_controls"] == affine["controls"], "unbound_ancestor")
    require(raw["controls"] == raw_controls(policy, raw["numeric_packet"]), "raw_denominator_origin")
    # Independently reconstruct the existing affine triangle denominator from
    # saved physical reference words; this performs no physical evaluation.
    refs = []
    for field, _ in sorted(affine["numeric_packet"]["variable_offsets"].items(), key=lambda row: row[1][0]):
        words = affine["numeric_packet"]["physical_reference_fields"][field]
        refs.extend(Fraction(h)+Fraction(l) for h, l in zip(words["high"], words["low"], strict=True))
    scales = list(map(Fraction, affine["numeric_packet"]["column_scaling"]))
    require(len(refs) == len(scales), "reference_shape")
    upper = []
    for value in map(abs, refs):
        rounded = float(value)
        if Fraction(rounded) < value:
            rounded = math.nextafter(rounded, math.inf)
        upper.append(Fraction(rounded))
    amounts = [s*Fraction(at) for s, at in zip(scales, raw["controls"]["atol"], strict=True)]
    rr = downward(min([Fraction(raw["controls"]["rtol"])] + [a/(2*u) for a, u in zip(amounts, upper, strict=True) if u]))
    aa = [downward((a-Fraction(rr)*u)/s) for a, u, s in zip(amounts, upper, scales, strict=True)]
    expected = dict(raw["controls"], rtol=rr, atol=aa)
    expected.pop("constraints_idx"); expected.pop("constraints_type")
    require(affine["controls"] == expected, "affine_denominator_origin")
    require(affine["weight_certificate"]["old_controls_sha256"] == digest(raw["controls"])
            and affine["weight_certificate"]["rtol"] == rr
            and affine["weight_certificate"]["atol"] == aa, "affine_certificate_binding")
    return policy


def check_applied_ceiling(request):
    policy = check_ancestry(request)
    origin, controls, packet = policy["origin"], request["controls"], request["numeric_packet"]
    factor = Fraction(*policy["tolerance_factor"])
    require(type(controls["rtol"]) is float and math.isfinite(controls["rtol"])
            and 0 < Fraction(controls["rtol"]) <= Fraction(float.fromhex(origin["applied_rtol_hex"]))*factor,
            "applied_rtol_ceiling")
    require(len(controls["atol"]) == policy["coordinates"] and controls["max_step"] == policy["max_step"],
            "applied_shape_or_step")
    expected = dict(request["qualification_ancestor"]["controls"],
                    rtol=controls["rtol"], atol=controls["atol"])
    policy_names = ("nonlinear_control_refinement", "nonlinear_guard_policy", "startup_step_policy",
                    "time_weight_policy", "segment_startup_policy")
    if origin["case_id"] == "DynamicAcceptorIonPublicDeviceV1":
        expected.update(nonlin_conv_coef=1.024e-5, nonlin_guard="first-correction-wrms-v1",
                        nonlin_trace_capacity=4096, first_step=7.8125e-7)
        require(all(name in request for name in policy_names), "required_B_controls_missing")
        for name, schema in (("time_weight_policy", "solarlab.voltage-lift-time-weights.v2"),
                             ("segment_startup_policy", "solarlab.voltage-lift-segment-startup.v2")):
            require(request[name].get("schema") == schema
                    and request[name].get("qualification_policy_sha256") == digest(policy), "control_policy_binding")
    else:
        require(not any(name in request for name in policy_names), "unexpected_S0_control_override")
    require(controls == expected, "unbound_applied_control")
    for field, (lo, hi) in packet["variable_offsets"].items():
        limit = Fraction(origin["field_denominator_ceilings"][field]["applied_atol_SI_exact"])*factor
        for i in range(lo, hi):
            at = controls["atol"][i]
            require(type(at) is float and math.isfinite(at) and at > 0
                    and Fraction(at)*Fraction(packet["column_scaling"][i]) <= limit, "applied_atol_ceiling")
    require(request["weight_certificate"]["rtol"] == controls["rtol"]
            and request["weight_certificate"]["atol"] == controls["atol"], "applied_certificate")
    # The constructor checks denominators before it creates initial-input data.
    # Every finalized observer request must also bind the exact initialization.
    if any(key in request for key in ("z0", "zdot0", "initial_preparation", "interval_observation")):
        check_initialization(request, policy["coordinates"])
    return policy["coordinates"]


def check_initialization(request, size):
    names = ("z0", "zdot0", "actual_initial_identity", "initial_preparation", "preparation_context_sha256")
    require(all(name in request for name in names), "initial_input_missing")
    record = request["initial_preparation"]
    for name in ("z0", "zdot0"):
        vector = request[name]
        require(type(vector) is list and len(vector) == size
                and all(type(v) is float and math.isfinite(v) for v in vector), "initial_vector_shape")
        key = "raw_z_hex" if name == "z0" else "raw_zdot_hex"
        require(record[key] == [v.hex() for v in vector], "initial_vector_words")
    require(record["raw_z_hex"] == [0.0.hex()]*size, "initial_state_reference_changed")
    context = {k: v for k, v in request.items() if k not in (*names, "interval_observation")}
    require(digest(context) == request["preparation_context_sha256"] == record["request_sha256"],
            "initial_context_binding")
    require(record["record_sha256"] == digest({k: v for k, v in record.items() if k != "record_sha256"})
            and record["map_identity"] == request["map_identity"]
            and record["point_identity"] == request["actual_initial_identity"]
            and record["predecessor_identity"] == request["voltage_lift_map"]["physical_reference"]
            and record["segment_id"] == request["segments"][0]["id"]
            and record["segment_sha256"] == digest(request["segments"][0])
            and record["state_changed"] is False and record["native_initialization_performed"] is False
            and record["native_steps"] == 0, "initial_record_binding")
    for name in ("desired_tangent_residual_SI", "represented_rate_residual_SI"):
        for word in ("high_hex", "low_hex"):
            values = record[name][word]
            require(len(values) == size and all(math.isfinite(float.fromhex(v)) for v in values),
                    "initial_residual_shape")


def check_initialized_request(request):
    require(all(name in request for name in ("z0", "zdot0", "initial_preparation")), "initial_input_missing")
    return check_applied_ceiling(request)


def options(policy):
    policy = validate_policy(policy)
    if policy["origin"]["case_id"] == "DynamicAcceptorIonPublicDeviceV1":
        return {"nonlin_conv_coef": 1e-8, "nonlin_guard": "first-correction-wrms-v1",
                "nonlin_trace_capacity": 4096, "first_step": 7.8125e-7,
                "time_weight_kappa": 1024,
                "segment_startup_overrides": {"slow_state_hold": {"first_step": 0.0}}}
    return {"nonlin_conv_coef": None, "nonlin_guard": None, "nonlin_trace_capacity": 4096,
            "first_step": None, "time_weight_kappa": None, "segment_startup_overrides": None}

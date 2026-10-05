"""Bounded paired transitions: exact input algebra and saved S0 regression.

Fraction checks the finite binary inputs independently. The 1e-28 relative
arithmetic diagnostic is inherited from test_temporal_interface; it neither
changes the physical device gates nor certifies a native trajectory.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    PARTICLE, VOLT, VOLUME, ContractError, Layout, Point,
    StateIncrement, StateView, Support, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic, DoubleArray, PrimitiveDifference, PrimitiveExpansion,
    RelativeCoordinates, decode_point, encode_point,
)


ARITHMETIC_RTOL = Fraction(1, 10**28)


def rational(value):
    return Fraction.from_float(float(value))


def exact_primitive(value):
    if isinstance(value, PrimitiveDifference):
        return [a-b for a, b in zip(exact_primitive(value.current),
                                    exact_primitive(value.previous), strict=True)]
    return [sum((rational(word.ravel()[i]) for word in value.words), Fraction())
            for i in range(value.high.size)]


def exact_dd(value):
    return [rational(a)+rational(b) for a, b in
            zip(value.high.ravel(), value.low.ravel(), strict=True)]


def check_words(value, reference, *, exact=False):
    represented = exact_dd(value)
    errors = []
    for actual, expected in zip(represented, reference, strict=True):
        error = abs(actual-expected)
        assert (actual == 0) == (expected == 0)
        assert (actual > 0) == (expected > 0)
        assert error == 0 if exact else error <= abs(expected)*ARITHMETIC_RTOL
        errors.append(str(error))
    return {"actual": list(map(str, represented)), "reference": list(map(str, reference)),
            "absolute_error": errors, "relative_budget": str(ARITHMETIC_RTOL)}


def record_case(request, data):
    if target := os.environ.get("PRIMITIVE_CASE_LOG"):
        with Path(target).open("a") as stream:
            stream.write(json.dumps({"test": request.node.nodeid, "native_steps": 0,
                                     "passed": True, **data}, allow_nan=False)+"\n")


def linear_fixture(count=2):
    layout = Layout((Support("nodes", "global", (count,)),), (
        VariableSpec("n_m3", "carrier", "nodes", (count,), PARTICLE/VOLUME, lower=0),
        VariableSpec("phi_V", "electrostatics", "nodes", (count,), VOLT)), ())
    coordinates = RelativeCoordinates(layout, {v.id: "linear" for v in layout.variables})
    reference = coordinates.initial(StateView(layout, [
        ("n_m3", DoubleArray(np.ones(count))),
        ("phi_V", DoubleArray(np.zeros(count)))]), inputs=[0.])
    return coordinates, reference


def wide_endpoints(*, sign=1, face_change=True, density=False):
    """Each endpoint has <=4 words; its local difference has a fifth word."""
    previous = [np.zeros(4) for _ in range(4)]
    current = [np.zeros(4) for _ in range(4)]
    selected = slice(0, 2) if density else slice(2, 4)
    previous[0][selected] = sign*2.0**-216
    for target, word in zip(current, [1., 2.0**-54, 2.0**-108, 2.0**-162], strict=True):
        target[selected] = sign*word
    if face_change:
        current[3][selected.stop-1] *= 2
    return PrimitiveExpansion(previous), PrimitiveExpansion(current)


def test_pair_owns_two_fixed_endpoints_without_nesting_or_projection():
    a, b = wide_endpoints()
    pair = PrimitiveDifference(b, a)
    identity = pair.identity_bytes()
    assert pair.shape == (4,) and pair.immutable_copy() is pair
    assert not pair.is_zero() and PrimitiveDifference(a, a).is_zero()
    assert exact_primitive(pair.take_flat([3, 2])) == exact_primitive(pair)[3:1:-1]
    assert len(pair.current.words)+len(pair.previous.words) == 8
    view = pair.current.words[0]
    view.shape = (2, 2)
    view.dtype = np.int64
    assert pair.shape == (4,) and pair.identity_bytes() == identity
    with pytest.raises(ValueError):
        pair.current.words[0].setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        pair.current = a
    with pytest.raises(ContractError, match="requires_two_endpoints"):
        PrimitiveDifference(pair, a)
    with pytest.raises(ContractError, match="requires_two_endpoints"):
        PrimitiveDifference(a, b.take_flat([0]))
    with pytest.raises(ContractError, match="index_outside_support"):
        pair.take_flat([-1])
    with pytest.raises(ContractError):
        pair.take_flat([0.5])
    with pytest.raises(TypeError, match="projection"):
        np.asarray(pair)
    with pytest.raises(TypeError, match="projection"):
        float(pair)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("face_change", [False, True])
def test_paired_face_and_activity_contract_all_words_before_projection(sign, face_change, request):
    coordinates, root = linear_fixture()
    before, after = wide_endpoints(sign=sign, face_change=face_change)
    left, _ = coordinates.trial(root, before, 1., [0.])
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        coordinates.trial(root, after, 2., [0.], predecessor=left,
                          transition_representation="four-word-v1")
    right, increment = coordinates.trial(root, after, 2., [0.], predecessor=left)
    exact_delta = [b-a for a, b in zip(exact_primitive(before), exact_primitive(after), strict=True)]
    expected_face = exact_delta[3]-exact_delta[2]
    arithmetic, vt = DoubleArithmetic(), 0.025851999786435535
    checks = []
    for first, last in ((left, right),
                        (decode_point(encode_point(left), root.state.layout),
                         decode_point(encode_point(right), root.state.layout))):
        change = StateIncrement.from_points(first, last)
        checks.append(check_words(change.face_delta(first, last, "phi_V", [[0, 1]],
                                                     arithmetic=arithmetic), [expected_face], exact=True))
        checks.append(check_words(change.electrochemical_delta(
            first, last, "n_m3", "phi_V", [[0, 1]], vt,
            potential_sign=-1, arithmetic=arithmetic), [-expected_face/rational(vt)]))
        check_words(change.log_ratio(first, last, "n_m3"), [Fraction(), Fraction()], exact=True)
        check_words(change.field("phi_V"), exact_delta[2:])
        assert all(type(v) is PrimitiveDifference for v in last.state.authority.local_primitives.values())
    assert right.state.authority.payload()["schema"] == "solarlab.state-authority.v5"
    increment.validate(left, right)
    record_case(request, {"family": "paired_temporal_contraction", "checks": checks,
                          "source_word_count_per_operand": 4, "operand_count": 2})


def test_paired_affine_density_log_has_independent_decimal_reference(request):
    coordinates, root = linear_fixture()
    before, after = wide_endpoints(density=True)
    left, _ = coordinates.trial(root, before, 1., [0.])
    right, increment = coordinates.trial(root, after, 2., [0.], predecessor=left)
    reference = []
    with localcontext() as context:
        context.prec = 180
        decimal = lambda x: Decimal(x.numerator)/Decimal(x.denominator)
        old = [Fraction(1)+v for v in exact_primitive(before)[:2]]
        new = [Fraction(1)+v for v in exact_primitive(after)[:2]]
        reference = [(decimal(b)/decimal(a)).ln() for a, b in zip(old, new, strict=True)]
        actual = exact_dd(increment.log_ratio(left, right, "n_m3"))
        for got, expected in zip(actual, reference, strict=True):
            assert abs(decimal(got)-expected) <= abs(expected)*Decimal("1e-28")
        expected_face = reference[1]-reference[0]
        face = increment.electrochemical_delta(left, right, "n_m3", "phi_V", [[0, 1]],
                                               0.025, potential_sign=-1, arithmetic=DoubleArithmetic())
        assert expected_face > 0
        assert abs(decimal(exact_dd(face)[0])-expected_face) <= abs(expected_face)*Decimal("1e-28")
    record_case(request, {"family": "paired_affine_temporal_log", "Decimal_precision": 180,
                          "reference": list(map(str, reference)), "actual": list(map(str, actual)),
                          "face_reference": str(expected_face), "face_actual": str(exact_dd(face)[0])})


@pytest.mark.parametrize("policy,schema", [("four-word-v1", "v4"), ("paired-endpoints-v1", "v5")])
def test_versioned_trial_and_old_local_codec_roundtrip(policy, schema):
    coordinates, root = linear_fixture()
    old_local, _ = coordinates.advance(root, [0., 0., 0.1, 0.2], 1., [0.])
    assert old_local.state.authority.payload()["schema"] == "solarlab.state-authority.v3"
    assert encode_point(decode_point(encode_point(old_local), root.state.layout)) == encode_point(old_local)
    trial, _ = coordinates.trial(root, [0., 0., 0.1, 0.2], 1., [0.], transition_representation=policy)
    encoded = encode_point(trial)
    assert encoded["payload"]["authority"]["schema"] == "solarlab.state-authority."+schema
    assert encode_point(decode_point(json.loads(json.dumps(encoded)), root.state.layout)) == encoded
    with pytest.raises(ContractError, match="unknown_affine_transition_representation"):
        coordinates.trial(root, [0., 0., 0., 0.], 2., [0.], transition_representation="guess")


def resign(record):
    record["sha256"] = hashlib.sha256(json.dumps(record["payload"], sort_keys=True,
                                                 separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return record


@pytest.mark.parametrize("mutation", ["current", "previous", "shape", "nested", "policy", "schema", "missing_transition"])
def test_paired_codec_rejects_forgery_with_fresh_outer_digest(mutation):
    coordinates, root = linear_fixture()
    left, _ = coordinates.trial(root, [0., 0., 0.1, 0.2], 1., [0.])
    right, _ = coordinates.trial(root, [0., 0., 0.3, 0.4], 2., [0.], predecessor=left)
    data = deepcopy(encode_point(right))
    authority = data["payload"]["authority"]
    local = authority["transition"]["local_primitives"]["phi_V"]
    if mutation in {"current", "previous"}:
        local[mutation]["words"][0][0] = (0.5).hex()
    elif mutation == "shape":
        local["shape"] = [1, 2]
    elif mutation == "nested":
        local["current"] = deepcopy(local)
    elif mutation == "policy":
        authority["composition_policy"] = "exact-four-word-primitives-v1"
    elif mutation == "schema":
        authority["schema"] = "solarlab.state-authority.v4"
    else:
        authority["transition"] = None
    with pytest.raises(ContractError):
        decode_point(resign(data), root.state.layout)


def test_pair_checks_predecessor_words_and_forbids_mixed_local_modes():
    coordinates, root = linear_fixture()
    left, _ = coordinates.trial(root, [0., 0., 0.1, 0.2], 1., [0.])
    right, _ = coordinates.trial(root, [0., 0., 0.3, 0.4], 2., [0.], predecessor=left)
    authority = right.state.authority
    old = dict(authority.previous_primitives)
    old["phi_V"] = PrimitiveExpansion.from_value([0.2, 0.3])
    paired = {k: PrimitiveDifference(authority.primitives[k], old[k]) for k in old}
    forged = replace(authority, previous_primitives=old, local_primitives=paired)
    point = Point(right.time, right.y, right.inputs, StateView(root.state.layout, authority=forged),
                  right.coordinate_reference)
    with pytest.raises(ContractError, match="increment_transition_primitive_mismatch"):
        StateIncrement.from_points(left, point)
    with pytest.raises(ContractError, match="paired_transition_requires_affine_endpoints"):
        replace(authority, local_primitives={**paired, "n_m3": PrimitiveExpansion.from_value([0., 0.])})


@pytest.mark.parametrize("policy", [None, "four-word-v1", "paired-endpoints-v1"])
def test_empty_fixed_reference_keeps_original_v4_bytes(policy):
    layout = Layout((), (), ())
    coordinates = RelativeCoordinates(layout, {})
    root = coordinates.initial(StateView(layout, ()), inputs=[2.])
    kwargs = {} if policy is None else {"transition_representation": policy}
    point, increment = coordinates.trial(root, [], 1., [3.], **kwargs)
    golden_path = os.environ.get("EMPTY_LEGACY_CODEC")
    if golden_path is None:
        pytest.skip("original-source empty-layout codec receipt not supplied")
    golden = json.loads(Path(golden_path).read_text())
    assert encode_point(root) == golden["reference"]
    assert encode_point(point) == golden["point"]
    assert point.state.authority.payload()["schema"] == "solarlab.state-authority.v4"
    assert encode_point(decode_point(golden["point"], layout)) == golden["point"]
    increment.validate(root, point)
    bad = deepcopy(golden["point"])
    bad["payload"]["authority"].update(schema="solarlab.state-authority.v5",
                                      composition_policy="four-word-endpoints-paired-transition-v1")
    with pytest.raises(ContractError, match="precision_codec_empty_paired_transition"):
        decode_point(resign(bad), layout)


@pytest.fixture(scope="module")
def saved_s0_pair():
    """Restore original public records only; never initialize a time integrator."""
    from scripts.benchmarks.coupled_device_prototype import (
        AffineCoupledSlab, AffineVoltageMap, ProtocolSegment, SlabDefinition, VoltageLiftHistory,
    )
    source = os.environ.get("SAVED_PRIMITIVE_WITNESS")
    if source is None:
        pytest.skip("source-bound archived S0 witness not supplied")
    witness = json.loads(Path(source).read_text())
    for record in witness["extracted"].values():
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    data = lambda name: json.loads(Path(witness["extracted"][name]["path"]).read_text())
    native_request, failed, reference_record = (data(name) for name in
                                               ("NativeRequest.json", "SavedFailure.json", "Reference.json"))
    definition = SlabDefinition(**native_request["numeric_packet"]["definition"])
    assert hashlib.sha256(Path(definition.source_path).read_bytes()).hexdigest() == definition.source_sha256
    model = AffineCoupledSlab(definition, 8)
    mapping, previous = AffineVoltageMap(model), model.reference
    history = VoltageLiftHistory(mapping)
    # The source hash is part of the model/map identity. A changed source must
    # not impersonate the archived model, even when the physical map is equal.
    assert history.reference_record != reference_record["reference"]
    assert history.reference_record["physical_reference"] == reference_record["reference"]["physical_reference"]
    for name in ("columns_hex", "rows_hex", "lift_hex", "reference_inputs_hex"):
        assert mapping.payload()[name] == reference_record["reference"]["map"][name]
    raw_lines = Path(witness["extracted"]["AncestorRecords.jsonl"]["path"]).read_text().splitlines()
    # An isolated original-source reader emits only public codec records. It
    # performs no native initialization, solve, or hidden-history inference.
    if existing := os.environ.get("SAVED_PRIMITIVE_POINTS"):
        replay = json.loads(Path(existing).read_text())
    else:
        replayer = Path(os.environ["SAVED_PRIMITIVE_REPLAYER"])
        completed = subprocess.run([sys.executable, "-B", str(replayer)], capture_output=True, text=True,
                                   timeout=20, check=False)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        replay = json.loads((replayer.parent/"HistoricalPointsV1.json").read_text())
    assert len(replay["points"]) == len(raw_lines)
    identities = []
    for line, encoded in zip(raw_lines, replay["points"], strict=True):
        sample = json.loads(line)["history"]
        assert sample["schema"] == "solarlab.voltage-lift-sample.v1"
        previous = decode_point(encoded, model.layout)
        assert previous.identity == sample["point_identity"]
        assert encode_point(previous) == encoded
        identities.append(previous.identity)
    assert len(identities) == witness["exact_raw_ancestor_records"] == 28
    assert previous.identity == witness["actual_sampling_predecessor"]["point_identity"]
    with pytest.raises(ContractError, match="voltage_lift_history_binding_mismatch"):
        history.restore(reference_record["reference"], json.loads(raw_lines[0])["history"], model.reference)
    # A same-source legacy record still roundtrips, without relabeling any of
    # the original archive records as current-source records.
    first = json.loads(raw_lines[0])["history"]
    point, _, _, legacy = history.build_sample(
        [float.fromhex(v) for v in first["raw_solver_z_hex"]], float.fromhex(first["time_hex"]),
        [float.fromhex(v) for v in first["inputs_hex"]], model.reference,
        [float.fromhex(v) for v in first["raw_solver_zdot_hex"]],
        [float.fromhex(v) for v in first["input_rates_hex"]], origin="algebraic_probe",
        transition_representation="four-word-v1")
    assert history.restore(history.reference_record, legacy, model.reference)[0].identity == point.identity
    segment = next(ProtocolSegment(**s) for s in native_request["segments"] if s["id"] == failed["segment_id"])
    when = float.fromhex(failed["time_hex"])
    inputs, input_rate = segment.inputs(when)
    raw, raw_rate = (np.array([float.fromhex(v) for v in failed[key]]) for key in ("z_hex", "zdot_hex"))
    return model, mapping, history, previous, raw, raw_rate, when, inputs, input_rate, identities, native_request


def test_saved_s0_legacy_rejection_and_paired_public_history(saved_s0_pair, request):
    model, mapping, history, previous, raw, raw_rate, when, inputs, adot, identities, native_request = saved_s0_pair
    old_bytes = encode_point(previous)
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        history.build_sample(raw, when, inputs, previous, raw_rate, adot, origin="native",
                             transition_representation="four-word-v1")
    point, increment, rate, record = history.build_sample(raw, when, inputs, previous, raw_rate, adot, origin="native")
    assert record["schema"] == "solarlab.voltage-lift-sample.v2"
    restored, change, restored_rate, restored_adot = history.restore(history.reference_record, record, previous)
    assert restored.identity == point.identity and exact_primitive(restored_rate) == exact_primitive(rate)
    assert np.array_equal(restored_adot, adot) and encode_point(previous) == old_bytes
    assert record["raw_solver_z_hex"] == [float(v).hex() for v in raw]
    exact_map = [rational(mapping.columns[i])*rational(raw[i])+sum(
        (rational(mapping.lift[i, j])*(rational(inputs[j])-rational(mapping.reference_inputs[j])) for j in range(2)),
        Fraction()) for i in range(model.layout.size)]
    assert exact_primitive(mapping.physical_primitive(raw, inputs)) == exact_map
    changes = []
    for spec in model.layout.variables:
        pair = point.state.authority.local_primitives[spec.id]
        assert type(pair) is PrimitiveDifference
        expected = [a-b for a, b in zip(exact_primitive(pair.current), exact_primitive(pair.previous), strict=True)]
        changes.append({"field": spec.id, **check_words(increment.field(spec.id), expected)})
        assert exact_dd(increment.field(spec.id)) == exact_dd(change.field(spec.id))
    for mutation in ("schema", "transition_representation", "raw_solver_z_hex"):
        damaged = deepcopy(record)
        if mutation == "schema":
            damaged[mutation] = "solarlab.voltage-lift-sample.v1"
        elif mutation == "transition_representation":
            damaged[mutation] = "four-word-v1"
        else:
            damaged[mutation][19] = float(np.nextafter(raw[19], np.inf)).hex()
        with pytest.raises(ContractError):
            history.restore(history.reference_record, damaged, previous)
    record_case(request, {"family": "saved_S0_paired_history", "ancestor_point_identities": identities,
                          "new_point_identity": point.identity, "record_sha256": hashlib.sha256(
                              json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                          "changes": changes, "failed_saved_time_hex": float(when).hex(),
                          "full_protocol_s": native_request["segments"][-1]["end"],
                          "original_records_unchanged": True, "full_protocol_passed": False})


def test_saved_s0_temporal_physical_consumers(saved_s0_pair, request):
    from perovskite_sim.constants import Q
    model, mapping, history, previous, raw, raw_rate, when, inputs, adot, _, original_request = saved_s0_pair
    point, increment, _, _ = history.build_sample(raw, when, inputs, previous, raw_rate, adot, origin="native")
    changes = model.finite_physical_changes(previous, point, increment)
    delta = {v.id: exact_primitive(point.state.authority.local_primitives[v.id]) for v in model.layout.variables}
    dn, dp, dphi = (delta[name] for name in ("n_m3", "p_m3", "phi_V"))
    rho = [rational(Q)*(p-n) for n, p in zip(dn, dp, strict=True)]
    displacement = [-rational(model.definition.epsilon)*(dphi[i+1]-dphi[i])/rational(model.dx[i])
                    for i in range(model.count-1)]
    volume = list(map(rational, model.geometry.volumes))
    area = rational(model.definition.area)
    metal = [area*displacement[0]-volume[0]*rho[0], -area*displacement[-1]-volume[-1]*rho[-1]]
    body = [sum((v*r for v, r in zip(volume, rho, strict=True)), Fraction())]
    def measure(value, expected):
        actual = exact_dd(value)
        errors = [abs(a-b) for a, b in zip(actual, expected, strict=True)]
        passed = all((a == 0) == (b == 0) and (a > 0) == (b > 0)
                     and error <= abs(b)*ARITHMETIC_RTOL
                     for a, b, error in zip(actual, expected, errors, strict=True))
        return {"high_hex": [float(v).hex() for v in value.high.ravel()],
                "low_hex": [float(v).hex() for v in value.low.ravel()],
                "actual": list(map(str, actual)), "reference": list(map(str, expected)),
                "absolute_error": list(map(str, errors)),
                "relative_error": [str(error/abs(b)) if b else None for b, error in zip(expected, errors, strict=True)],
                "relative_diagnostic_budget": str(ARITHMETIC_RTOL), "passed": passed}

    evidence = {name: measure(changes[name], expected) for name, expected in (
        ("charge_density_C_m3", rho), ("displacement_C_m2", displacement),
        ("metal_charge_C", metal), ("body_charge_C", body))}
    # Interior carrier inventories use Geometry.volumes, which already include area.
    expected_storage = [volume[i]*d[i] for d in (dn, dp) for i in range(1, model.count-1)]
    evidence["storage"] = measure(changes["storage"], expected_storage)
    for spec in model.layout.variables:
        fraction = [sum((rational(x) for x in (model.reference.state.field(spec.id).high[i],
                                              model.reference.state.field(spec.id).low[i])), Fraction())
                    +v for i, v in enumerate(exact_primitive(point.state.authority.primitives[spec.id]))]
        evidence["endpoint_"+spec.id] = measure(point.state.field(spec.id), fraction)
    projected_n, projected_p = (exact_dd(increment.field(name)) for name in ("n_m3", "p_m3"))
    after_delta = sum((rational(Q)*v*(p-n) for v, n, p in
                       zip(volume, projected_n, projected_p, strict=True)), Fraction())
    after_rho = sum((v*r for v, r in zip(volume, exact_dd(changes["charge_density_C_m3"]), strict=True)), Fraction())
    actual_body = exact_dd(changes["body_charge_C"])[0]
    parts = {"local_density_DD_projection_C": after_delta-body[0],
             "nodal_charge_arithmetic_C": after_rho-after_delta,
             "weighted_product_and_sum_C": actual_body-after_rho}
    assert sum(parts.values(), Fraction()) == actual_body-body[0]
    total_variation = sum((abs(v*r) for v, r in zip(volume, rho, strict=True)), Fraction())
    physical_budget = rational(original_request["budgets"]["charge_C"])
    passed = all(v["passed"] for v in evidence.values())
    assert "sksundae" not in sys.modules
    record_case(request, {"family": "saved_S0_physical_consumers", "evidence": evidence,
                          "point_identity": point.identity, "native_acceptance_awarded": False,
                          "body_error_partition": {k: str(v) for k, v in parts.items()},
                          "body_sum_absolute_constituents_C": str(total_variation),
                          "body_cancellation_condition": str(total_variation/abs(body[0])),
                          "original_physical_charge_budget_C": str(physical_budget),
                          "single_sample_body_error_to_original_budget": str(abs(actual_body-body[0])/physical_budget),
                          "physical_budget_comparison_is_not_native_qualification": True,
                          "passed": passed})
    # Retain the original strict relative diagnostic. In particular, logging
    # a small absolute error must not silently turn its failed assertion green.
    assert passed, {name: v for name, v in evidence.items() if not v["passed"]}

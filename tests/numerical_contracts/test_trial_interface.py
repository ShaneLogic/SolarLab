"""Fixed-reference affine trials, separate from local-coordinate advances."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    PARTICLE, VOLT, BalanceTerms, ContractError, ImplicitSystem, Layout, Point,
    StateIncrement, StateView, StoragePartials, Support, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArithmetic, DoubleArray, PrimitiveExpansion, RelativeCoordinates,
    decode_point, encode_point,
)


def exact(value):
    return [Fraction.from_float(float(h))+Fraction.from_float(float(l))
            for h, l in zip(value.high.ravel(), value.low.ravel())]


def setup_reference(mapped=False):
    layout = Layout((Support("scalar", "global", (1,)),),
                    (VariableSpec("n", "population", "scalar", (1,), PARTICLE, lower=0),
                     VariableSpec("phi", "potential", "scalar", (1,), VOLT)), ())
    state = StateView(layout, [("n", DoubleArray([10.], [1e-17])),
                              ("phi", DoubleArray([0.25]))])
    resolved = Point(0, np.empty(0), np.array([0.5]), state, "original-physical-reference-v1")
    coordinates = RelativeCoordinates(layout, {"n": "linear", "phi": "linear"})
    if mapped:
        initial = coordinates.initial(state, inputs=[0.5])
        reference, _ = coordinates.advance(initial, [0.5, 0.125], 0.5, [0.5])
    else:
        reference = resolved
    return coordinates, reference, resolved


def roundtrip(point):
    return decode_point(json.loads(json.dumps(encode_point(point))), point.state.layout)


@pytest.mark.parametrize("mapped", [False, True])
def test_sibling_trials_bind_predecessor_with_fixed_cumulative_coordinates(mapped):
    coordinates, reference, resolved = setup_reference(mapped)
    first, _ = coordinates.trial(reference, [1., 0.25], 1, [0.75])
    second, change = coordinates.trial(reference, [1.5, -0.125], 2, [1.25], predecessor=first)
    np.testing.assert_array_equal(first.y, [1., 0.25])
    np.testing.assert_array_equal(second.y, [1.5, -0.125])
    assert exact(change.field("n")) == [Fraction(1, 2)]
    assert exact(change.field("phi")) == [Fraction(-3, 8)]
    assert second.state.authority.root_identity == resolved.state.authority.identity
    assert second.state.authority.fixed_reference.identity == reference.identity
    assert second.state.authority.coordinate_kind == "fixed-reference"
    for a, b in ((first, second), (roundtrip(first), roundtrip(second))):
        increment = StateIncrement.from_points(a, b)
        assert exact(increment.field("n")) == [Fraction(1, 2)]
        assert exact(increment.field("phi")) == [Fraction(-3, 8)]
    sibling, _ = coordinates.trial(reference, [1.5, -0.125], 2, [1.25])
    assert exact(sibling.state.field("n")) == exact(second.state.field("n"))
    with pytest.raises(ContractError, match="predecessor_point"):
        StateIncrement.from_points(first, sibling)


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("remainder", [[0., 0.], [-0.25, -0.125], [2., 0.5]])
def test_trial_physical_fields_keep_reference_and_valid_negative_remainders(mapped, remainder):
    coordinates, reference, _ = setup_reference(mapped)
    before = encode_point(reference)
    point, change = coordinates.trial(reference, remainder, 1, reference.inputs)
    for index, name in enumerate(("n", "phi")):
        expected = exact(reference.state.field(name))[0]+Fraction.from_float(remainder[index])
        assert exact(point.state.field(name)) == [expected]
        assert exact(change.field(name)) == [Fraction.from_float(remainder[index])]
    assert encode_point(reference) == before
    restored = roundtrip(point)
    assert restored.identity == point.identity
    assert restored.state.authority.fixed_reference.identity == reference.identity
    assert restored.state.authority.payload()["schema"] == "solarlab.state-authority.v4"
    assert restored.state.authority.payload()["coordinate_contract"]["solver_projection"] == "first-word"


def test_trial_keeps_dd_and_four_word_inputs_with_same_solver_projection():
    coordinates, reference, _ = setup_reference()
    first, _ = coordinates.trial(reference, DoubleArray([1e5, 0]), 1, [0.5])
    second, change = coordinates.trial(reference, DoubleArray([1e5, 0], [1e-17, 0]),
                                        1, [0.5], predecessor=first)
    np.testing.assert_array_equal(first.y, second.y)
    assert exact(change.field("n")) == [Fraction.from_float(1e-17)]
    assert StateIncrement.from_points(roundtrip(first), roundtrip(second)).field("n").identity_bytes() == change.field("n").identity_bytes()
    words = [np.array([v, 0.]) for v in (1., 2.**-54, 2.**-108, 2.**-162)]
    richer = PrimitiveExpansion(words)
    a, _ = coordinates.trial(reference, richer, 2, [0.5])
    smaller = PrimitiveExpansion([*words[:3], np.array([2.**-163, 0.])])
    b, increment = coordinates.trial(reference, smaller, 2, [0.5], predecessor=a)
    np.testing.assert_array_equal(a.y, b.y)
    assert exact(increment.field("n")) == [Fraction.from_float(-2.**-163)]
    restored = roundtrip(b)
    for actual, supplied in zip(restored.state.authority.primitives["n"].words, smaller.words):
        np.testing.assert_array_equal(actual, supplied[:1])
    assert exact(StateIncrement.from_points(roundtrip(a), restored).field("n")) == [Fraction.from_float(-2.**-163)]


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("form", ["array", "double", "expansion"])
@pytest.mark.parametrize("values", [[-0.0, 0.25], [0.25, -0.0]])
def test_trial_canonical_signed_zero_keeps_exact_solver_binding(mapped, form, values):
    coordinates, reference, _ = setup_reference(mapped)
    raw = np.asarray(values)
    value = (raw if form == "array" else DoubleArray(raw) if form == "double"
             else PrimitiveExpansion((raw, np.zeros(2), np.zeros(2), np.zeros(2))))
    point, increment = coordinates.trial(reference, value, 1, reference.inputs)
    canonical = raw.copy(); canonical[canonical == 0] = 0.0
    assert point.y.tobytes() == canonical.tobytes()
    increment.validate(reference, point)
    assert roundtrip(point).identity == point.identity
    with pytest.raises(ContractError, match="point_authority_binding_mismatch"):
        replace(point, y=raw)
    record = json.loads(json.dumps(encode_point(point)))
    index = int(np.flatnonzero(np.signbit(raw))[0])
    record["payload"]["y"][index] = float(-0.0).hex()
    record["payload"]["authority"]["coordinates"][index] = float(-0.0).hex()
    reseal(record)
    with pytest.raises(ContractError, match="authority_coordinate_projection_mismatch"):
        decode_point(record, reference.state.layout)


@pytest.mark.parametrize("replacement", ["time", "inputs", "y", "label", "physical_root"])
def test_trial_rejects_foreign_fixed_reference_even_when_physics_matches(replacement):
    coordinates, reference, _ = setup_reference()
    first, _ = coordinates.trial(reference, [1., 0.], 1, [0.5])
    if replacement == "time": other = replace(reference, time=3)
    elif replacement == "inputs": other = replace(reference, inputs=np.array([2.]))
    elif replacement == "y": other = replace(reference, y=np.array([1., 2.]))
    elif replacement == "label": other = replace(reference, coordinate_reference="another-reference")
    else:
        other = replace(reference, state=StateView(reference.state.layout,
                        [("n", DoubleArray([11.])), ("phi", DoubleArray([0.25]))]))
    with pytest.raises(ContractError, match="predecessor_reference"):
        coordinates.trial(other, [2., 0.], 2, [0.5], predecessor=first)
    with pytest.raises(ContractError, match="predecessor_reference"):
        coordinates.trial(reference, [2., 0.], 2, [0.5], predecessor=other)


def test_trial_rejects_rebasing_nonlinear_modes_bad_shapes_and_physical_domain():
    coordinates, reference, _ = setup_reference()
    first, _ = coordinates.trial(reference, [1., 0.], 1, [0.5])
    with pytest.raises(ContractError, match="rebase_forbidden"):
        coordinates.trial(first, [1., 0.], 2, [0.5])
    with pytest.raises(ContractError, match="linear_reference"):
        RelativeCoordinates(reference.state.layout, {"n": "log", "phi": "linear"}).trial(reference, [1., 0.], 1)
    for remainder in ([1.], [[1., 0.]]):
        with pytest.raises(ContractError):
            coordinates.trial(reference, remainder, 1)
    with pytest.raises(ArithmeticError, match="DD requires finite"):
        coordinates.trial(reference, [np.nan, 0.], 1)
    with pytest.raises(ContractError, match="physical_state_outside_domain"):
        coordinates.trial(reference, [-11., 0.], 1)
    with pytest.raises(ContractError, match="coordinate_reference_mismatch"):
        coordinates.advance(first, [1., 0.], 2)


def test_trial_exact_transition_capacity_is_bounded_and_nonmutating():
    coordinates, reference, _ = setup_reference()
    a = PrimitiveExpansion([np.array([v, 0.]) for v in (1., 2.**-54, 2.**-108, 2.**-162)])
    b = PrimitiveExpansion([np.array([v, 0.]) for v in (2., 2.**-100, 2.**-200, 2.**-300)])
    first, _ = coordinates.trial(reference, a, 1)
    before = encode_point(first)
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        coordinates.trial(reference, b, 2, predecessor=first)
    assert encode_point(first) == before


@pytest.mark.parametrize("tiny", [0.0, 1e-40, -1e-40])
@pytest.mark.parametrize("lower", [None, 0.0])
def test_trial_joint_affine_projection_preserves_depletion_sign_and_domain(tiny, lower):
    layout = Layout((Support("scalar", "global", (1,)),),
                    (VariableSpec("n", "affine-field", "scalar", (1,), PARTICLE, lower=lower),), ())
    state = StateView(layout, [("n", DoubleArray([1e22], [1e-18]))])
    reference = Point(0, np.empty(0), np.empty(0), state, "depletion-original-reference")
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    remainder = PrimitiveExpansion(tuple(np.array([value]) for value in (-1e22, -1e-18, tiny, 0.)))
    before = encode_point(reference)
    if lower == 0 and tiny < 0:
        with pytest.raises(ContractError, match="physical_state_outside_domain"):
            coordinates.trial(reference, remainder, 1)
    else:
        point, increment = coordinates.trial(reference, remainder, 1)
        assert exact(point.state.field("n")) == [Fraction.from_float(tiny)]
        assert exact(roundtrip(point).state.field("n")) == [Fraction.from_float(tiny)]
        assert point.state.project_field("n").evaluator_identity == "stateless-dd-affine-joint-v1"
        increment.validate(reference, point)
        if tiny > 0:
            damaged = json.loads(json.dumps(encode_point(point)))
            damaged["payload"]["fields"]["n"]["high"] = [float(0).hex()]
            damaged["payload"]["fields"]["n"]["low"] = [float(0).hex()]
            reseal(damaged)
            with pytest.raises(ContractError, match="precision_codec_projection_mismatch"):
                decode_point(damaged, layout)
    assert encode_point(reference) == before



@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("ratio", [1., 2., 0.5])
def test_trial_affine_faces_contract_anchor_and_sources_before_dd(reverse, ratio):
    layout = Layout((Support("pair", "cell", (2,)),),
                    (VariableSpec("n", "population", "pair", (2,), PARTICLE, lower=0),
                     VariableSpec("phi", "potential", "pair", (2,), VOLT)), ())
    reference = Point(0, np.empty(0), np.empty(0), StateView(layout, [
        ("n", DoubleArray([0., 1e22], [0., 1e-18])),
        ("phi", DoubleArray(np.zeros(2)))]), "joint-affine-face-reference")
    coordinates = RelativeCoordinates(layout, {"n": "linear", "phi": "linear"})
    epsilon = 1e-40
    primitive = PrimitiveExpansion((np.array([epsilon, -1e22, 0., 0.]),
                                    np.array([0., -1e-18, 0., 0.]),
                                    np.array([0., ratio*epsilon, 0., 0.]), np.zeros(4)))
    point, _ = coordinates.trial(reference, primitive, 1)
    source = [Fraction.from_float(epsilon), Fraction.from_float(ratio*epsilon)]
    pair = [[1, 0]] if reverse else [[0, 1]]
    i, j = pair[0]
    expected_drop = source[j]-source[i]
    with localcontext() as context:
        context.prec = 100
        expected_log = (Decimal(source[j].numerator)/Decimal(source[j].denominator)).ln()-(
            Decimal(source[i].numerator)/Decimal(source[i].denominator)).ln()
        for current in (point, roundtrip(point)):
            assert exact(current.state.field("n")) == source
            assert exact(current.state.face_difference("n", pair)) == [expected_drop]
            observed = exact(current.state.log_ratio("n", pair))[0]
            observed_log = Decimal(observed.numerator)/Decimal(observed.denominator)
            assert abs(observed_log-expected_log) < Decimal("1e-30")
            activity = current.state.electrochemical_difference("n", "phi", pair, 0.025,
                        potential_sign=-1, arithmetic=DoubleArithmetic())
            assert exact(activity) == [observed]
            # Zero electric drive leaves a unit-coefficient SG diffusion flux.
            # This reads the public face operation, not endpoint subtraction.
            flux = -current.state.face_difference("n", pair).as_dd()
            assert exact(DoubleArray.from_dd(flux)) == [-expected_drop]
            assert (flux != 0).all() == (ratio != 1.)


def test_historical_v3_projection_is_not_reinterpreted_as_joint_v4():
    layout = Layout((Support("scalar", "global", (1,)),),
                    (VariableSpec("n", "legacy-affine-field", "scalar", (1,), PARTICLE),), ())
    state = StateView(layout, [("n", DoubleArray([1e22], [1e-18]))])
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    reference = coordinates.initial(state)
    remainder = PrimitiveExpansion(tuple(np.array([value]) for value in (-1e22, -1e-18, 1e-40, 0.)))
    historical, _ = coordinates.advance(reference, remainder, 1)
    record = encode_point(historical)
    assert record["payload"]["authority"]["schema"] == "solarlab.state-authority.v3"
    assert exact(historical.state.field("n")) == [Fraction()]
    assert exact(roundtrip(historical).state.field("n")) == [Fraction()]
    assert historical.state.project_field("n").evaluator_identity == "stateless-dd-map-v1"
    current, _ = coordinates.trial(reference, remainder, 1)
    assert exact(current.state.field("n")) == [Fraction.from_float(1e-40)]


def reseal(record):
    record["sha256"] = hashlib.sha256(json.dumps(record["payload"], sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("corruption", ["kind", "projection", "reference", "local_y", "drop_authority"])
def test_trial_codec_rejects_changed_coordinate_meaning(corruption):
    coordinates, reference, _ = setup_reference()
    first, _ = coordinates.trial(reference, [1., 0.], 1, [0.5])
    point, _ = coordinates.trial(reference, [1.5, 0.], 2, [0.5], predecessor=first)
    record = json.loads(json.dumps(encode_point(point)))
    payload = record["payload"]; authority = payload["authority"]
    if corruption == "kind": authority["coordinate_contract"]["kind"] = "local"
    elif corruption == "projection": authority["coordinate_contract"]["solver_projection"] = "rounded-sum"
    elif corruption == "reference":
        authority["coordinate_contract"]["reference"] = encode_point(replace(reference, time=4))
    elif corruption == "local_y":
        payload["y"][0] = authority["coordinates"][0] = float(0.5).hex()
    else:
        payload["authority"] = None; payload["field_role"] = "resolved-input"
    reseal(record)
    with pytest.raises(ContractError):
        decode_point(record, reference.state.layout)


def test_advance_keeps_its_local_coordinates_and_v3_payload():
    coordinates, _, resolved = setup_reference()
    initial = coordinates.initial(resolved.state, inputs=[0.5])
    first, _ = coordinates.advance(initial, [1., 0.25], 1, [0.5])
    second, _ = coordinates.advance(first, [2., 0.125], 2, [0.5])
    np.testing.assert_array_equal(second.y, [2., 0.125])
    payload = encode_point(second)["payload"]["authority"]
    assert payload["schema"] == "solarlab.state-authority.v3"
    assert "coordinate_contract" not in payload
    assert second.state.authority.coordinate_kind == "local"
    assert second.state.authority.fixed_reference is None
    assert roundtrip(second).identity == second.identity


def test_trial_zero_remainder_retains_nonzero_reference_residual_and_affine_jacobian():
    coordinates, reference, _ = setup_reference()
    point, _ = coordinates.trial(reference, [0., 0.], reference.time, reference.inputs)

    class Storage:
        def value(self, p): return p.state.field("n")
        def delta(self, left, right, increment):
            increment.validate(left, right)
            return increment.field("n")
        def first(self, p):
            return StoragePartials(DoubleArray([[1., 0.]]), DoubleArray([[0.]]), DoubleArray([0.]))
        def rate_jvp(self, p, ydot, adot, dy, da, dt): return DoubleArray([0.])

    def terms(p):
        return BalanceTerms(DoubleArray.from_dd(p.state.field("n").as_dd()+DD([0.25])),
                            DoubleArray.from_dd(p.state.field("phi").as_dd()-p.inputs),
                            DoubleArray([[1., 0.]]), DoubleArray([[0.]]), DoubleArray([0.]),
                            DoubleArray([[0., 1.]]), DoubleArray([[-1.]]), DoubleArray([0.]))

    system = ImplicitSystem(Storage(), terms, DoubleArithmetic())
    residual = system.residual(point, np.zeros(2), np.zeros(1))
    assert exact(residual) == [-Fraction(41, 4)-Fraction.from_float(1e-17), -Fraction(1, 4)]
    assert exact(system.linearize(point, np.zeros(2), np.zeros(1)).ida_matrix(2)) == [Fraction(1), Fraction(), Fraction(), Fraction(1)]

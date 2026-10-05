"""Exact finite-input references for the shared affine physical action.

No device, integrator or native binding is constructed. Fraction/Decimal here
are independent references; the action uses guarded finite products and one
final row reduction. These tests do not certify source physics or integration.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    COULOMB, LENGTH, ONE, PARTICLE, SECOND, VOLT, VOLUME,
    BoundLinearSource, ContractError, EquationSpec, FloatArray, FloatArithmetic,
    ImplicitSystem, Layout, LinearCoordinates, LinearFactor, LinearSourceSpec,
    LinearTerm, PhysicalLinearForm, PhysicalStorage, Point, RateView,
    SparseLinearization, SparseStructure, StateIncrement, StateView, Support,
    ValidatedProblem, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic, DoubleArray, PrimitiveExpansion, RelativeCoordinates,
    decode_point, encode_point,
)


SOURCE = sha256(b"independent-linear-action-controls-v1").hexdigest()
MAP = sha256(b"explicit-synthetic-affine-rate-input-v1").hexdigest()
RTOL = Fraction(1, 10**28)


def f(value):
    return Fraction.from_float(float(value))


def fractions(value):
    if isinstance(value, DoubleArray):
        return [f(a) + f(b) for a, b in zip(value.high.flat, value.low.flat, strict=True)]
    return [f(v) for v in np.asarray(value).flat]


def assert_result(result, reference, *, exact=False):
    actual = fractions(result.value)
    bounds = list(map(f, result.absolute_error_bound.values))
    assert len(actual) == len(reference) == len(bounds)
    for got, want, bound in zip(actual, reference, bounds, strict=True):
        error = abs(got - want)
        assert error <= bound
        assert error == 0 if exact else error <= abs(want) * RTOL
        assert (got == 0) == (want == 0)
        assert (got > 0) == (want > 0)
    return {"represented": list(map(str, actual)), "reference": list(map(str, reference)),
            "absolute_error_bound": list(map(str, bounds)), "form": result.form_identity,
            "operand": result.operand_identity, "arithmetic_policy": result.arithmetic_policy}


def record(request, data):
    if target := os.environ.get("LINEAR_ACTION_CASE_LOG"):
        with Path(target).open("a") as stream:
            stream.write(json.dumps({"test": request.node.nodeid, "passed": True,
                                     "native_steps": 0, **data}) + "\n")


@pytest.mark.parametrize("sign", (-1, 0, 1))
def test_bound_four_word_source_keeps_cancelled_tail(sign, request):
    """A source used by the map/Point proof retains its complete four words."""
    _, point = fixture()
    source = PrimitiveExpansion(tuple(sign*np.array(word) for word in (
        [1., 1.], [2.0**-54, 2.0**-54], [2.0**-108, 2.0**-108],
        [2.0**-162, 2.0**-161])))
    original = source.identity_bytes()
    spec = LinearSourceSpec("mapped_input", "nodes", (2,), PARTICLE/VOLUME, SOURCE)
    contract = form(point.state.layout, (
        LinearTerm(0, spec.id, 1), LinearTerm(0, spec.id, 0, sign=-1)),
        PARTICLE/VOLUME, sources=(spec,))
    packet = BoundLinearSource.bind(point, ((spec, source),))
    result = point.state.linear_form(contract, point=point, sources=packet,
                                      arithmetic=DoubleArithmetic())
    reference = [sum((f(word[1])-f(word[0]) for word in source.words), Fraction())]
    assert reference == [sign*Fraction(1, 2**162)]
    evidence = assert_result(result, reference, exact=True)
    assert source.is_finite() and packet[0].value.is_finite()
    assert type(packet[0].value) is PrimitiveExpansion
    assert packet[0].value.identity_bytes() == original == source.identity_bytes()
    with pytest.raises(ValueError):
        packet[0].value.words[3][0] = 0.
    with pytest.raises(TypeError, match="implicit primitive projection"):
        np.asarray(packet[0].value)
    with pytest.raises(ContractError, match="linear_source_binding"):
        BoundLinearSource.bind(point, ((spec, source.reshape((1, 2))),))
    if sign:
        with pytest.raises(ContractError, match="primitive_projection_would_discard_remainder"):
            source.as_dd()
    record(request, {"family": "bound_four_word_source", "sign": sign,
                     "source_identity": packet[0].identity, "result": evidence})


def fixture(count=2):
    layout = Layout((Support("nodes", "cell", (count,)),), (
        VariableSpec("n", "electron", "nodes", (count,), PARTICLE / VOLUME, lower=0),
        VariableSpec("p", "hole", "nodes", (count,), PARTICLE / VOLUME, lower=0),
        VariableSpec("phi", "electrostatic", "nodes", (count,), VOLT),
    ), ())
    coordinates = RelativeCoordinates(layout, {spec.id: "linear" for spec in layout.variables})
    state = StateView(layout, [(spec.id, DoubleArray(np.zeros(count) if spec.id == "phi" else np.ones(count)))
                               for spec in layout.variables])
    return coordinates, coordinates.initial(state, inputs=[0.0])


def factor(name, value, unit=ONE):
    return LinearFactor(name, value, unit, SOURCE)


def form(layout, terms, unit, *, kind="state", sources=(), row_ids=("row",)):
    return PhysicalLinearForm(layout, row_ids, (unit,) * len(row_ids), tuple(terms),
                              SOURCE, kind, len(terms), tuple(sources))


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_factored_products_survive_cancellation_before_final_row(sign, request):
    _, point = fixture(1)
    a, b, c, d = [float(1 + k * 2.0**-52) for k in (1, 2, 3, 4)]
    factors = tuple(factor(str(i), value) for i, value in enumerate((a, b, c, d)))
    rounded = a * b * c * d
    exact_product = f(a) * f(b) * f(c) * f(d)
    left, right = ((factors, factors) if sign == 0 else
                   (factors, (factor("rounded-comparison-input", rounded),)) if sign == 1 else
                   ((factor("rounded-comparison-input", rounded),), factors))
    action = form(point.state.layout, [LinearTerm(0, "p", 0, left), LinearTerm(0, "n", 0, right, -1)],
                  PARTICLE / VOLUME)
    result = point.state.linear_form(action, arithmetic=DoubleArithmetic(), point=point)
    expected = Fraction() if sign == 0 else sign * (exact_product - f(rounded))
    # The comparison input is explicitly a separate supplied binary64 value;
    # it is never substituted for the four original coefficient factors.
    assert expected == 0 if sign == 0 else expected * sign > 0
    record(request, assert_result(result, [expected], exact=sign == 0))


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_full_endpoint_words_cross_fields_before_body_increment_rounding(sign, request):
    coordinates, root = fixture()
    layout, count = root.state.layout, 2
    before = [np.zeros(layout.size) for _ in range(4)]
    after = [np.zeros(layout.size) for _ in range(4)]
    for name in ("n", "p"):
        section = layout.offsets[name]
        for target, value in zip(after, (1.0, 2.0**-54, 2.0**-108, 0.0), strict=True):
            target[section] = value
    before[0][layout.offsets["p"]] = sign * 2.0**-216
    after[3][layout.offsets["p"]] = sign * 2.0**-162
    left, _ = coordinates.trial(root, PrimitiveExpansion(before), 1.0, [0.0])
    right, increment = coordinates.trial(root, PrimitiveExpansion(after), 2.0, [0.0], predecessor=left)
    original = encode_point(left), encode_point(right)
    q, volume = 1.602176634e-19, 0.037
    factors = (factor("q", q, COULOMB / PARTICLE), factor("cell-volume-including-area", volume, VOLUME))
    terms = [LinearTerm(0, name, i, factors, s) for i in range(count) for name, s in (("p", 1), ("n", -1))]
    action = form(layout, terms, COULOMB, kind="increment")
    expected = count * f(q) * f(volume) * sign * (f(2.0**-162) - f(2.0**-216))
    arithmetic = DoubleArithmetic()
    result = increment.linear_form(left, right, action, arithmetic=arithmetic)
    check = assert_result(result, [expected])
    new_left, new_right = (decode_point(record, layout) for record in original)
    decoded = StateIncrement.from_points(new_left, new_right)
    restored = decoded.linear_form(new_left, new_right, action, arithmetic=arithmetic)
    assert fractions(restored.value) == fractions(result.value)
    assert (encode_point(left), encode_point(right)) == original
    state_action = replace(action, kind="state")
    state_result = right.state.linear_form(state_action, arithmetic=arithmetic, point=right)
    assert_result(state_result, [count * f(q) * f(volume) * sign * f(2.0**-162)])
    with pytest.raises(ContractError, match="endpoint|predecessor"):
        increment.linear_form(root, right, action, arithmetic=arithmetic)
    damaged = StateIncrement(increment.left_identity, increment.right_identity,
                             {**increment.fields, "p": DoubleArray(np.zeros(count))})
    with pytest.raises(ContractError, match="authority_mismatch"):
        damaged.linear_form(left, right, action, arithmetic=arithmetic)
    record(request, {**check, "left": left.identity, "right": right.identity,
                     "left_codec": original[0]["payload"]["authority"]["schema"],
                     "right_codec": original[1]["payload"]["authority"]["schema"]})


def test_fixed_background_participates_in_state_action_not_rate_or_increment():
    coordinates, root = fixture(1)
    values = [np.zeros(3) for _ in range(4)]
    values[2][root.state.layout.offsets["p"]] = 2.0**-108
    # A canonical four-word input places its sole nonzero word first.
    values = [np.sum(values, axis=0), np.zeros(3), np.zeros(3), np.zeros(3)]
    point, _ = coordinates.trial(root, PrimitiveExpansion(values), 1.0, [0.0])
    q = factor("q", 1.602176634e-19, COULOMB / PARTICLE)
    density = factor("fixed-background", 1.0, PARTICLE / VOLUME)
    action = form(root.state.layout, [LinearTerm(0, "p", 0, (q,)),
                                     LinearTerm(0, None, 0, (q, density), -1)], COULOMB / VOLUME)
    assert_result(point.state.linear_form(action, arithmetic=DoubleArithmetic(), point=point),
                  [f(q.value) * f(2.0**-108)])
    for kind in ("rate", "increment"):
        with pytest.raises(ContractError, match="constant_requires_state|unit_mismatch"):
            replace(action, kind=kind)


def rate_source_fixture():
    _, point = fixture()
    layout = point.state.layout
    words = [np.zeros(layout.size) for _ in range(4)]
    for array, word in zip(words, (1.0, 2.0**-54, 2.0**-108, 2.0**-162), strict=True):
        array[layout.offsets["p"].start] = word
    primitive = PrimitiveExpansion(words)
    rate = RateView(point, primitive, [0.3], source_identity=SOURCE, mapping_identity=MAP,
                    origin="mapped-coordinate-rate", raw_coordinates=point.y, raw_rate=words[0])
    q, volume = factor("q", 1.602176634e-19, COULOMB / PARTICLE), factor("volume", 1.0, VOLUME)
    rn = LinearSourceSpec("Rn", "nodes", (2,), PARTICLE / SECOND, SOURCE)
    rp = LinearSourceSpec("Rp", "nodes", (2,), PARTICLE / SECOND, SOURCE)
    sources = BoundLinearSource.bind(point, (
        (rn, DoubleArray([-1.0, 0.0], [-2.0**-54, 0.0])),
        (rp, DoubleArray([2.0**-108, 0.0])),
    ))
    action = form(layout, [LinearTerm(0, "p", 0, (q, volume)), LinearTerm(0, "Rn", 0, (q,)),
                           LinearTerm(0, "Rp", 0, (q,), -1)], COULOMB / SECOND,
                  kind="rate", sources=(rn, rp))
    return point, rate, sources, action, primitive, q.value


def later_point(point):
    """Create a valid different binding through the public coordinate map."""
    coordinates = RelativeCoordinates(point.state.layout, {spec.id: "linear" for spec in point.state.layout.variables})
    return coordinates.advance(point, np.zeros(point.y.shape), point.time + 1.0, point.inputs)[0]


def test_full_rate_and_actual_source_packet_cancel_before_final_row(request):
    point, rate, sources, action, primitive, q = rate_source_fixture()
    result = rate.linear_form(action, point=point, arithmetic=DoubleArithmetic(), sources=sources)
    record(request, assert_result(result, [f(q) * f(2.0**-162)]))
    assert rate.values is primitive
    assert rate.values.identity_bytes() == primitive.identity_bytes()
    assert len(rate.words) == 4 and len(result.source_identities) == 2
    assert rate.field("p").shape == (2,)
    with pytest.raises(TypeError, match="projection"):
        np.asarray(rate)
    # Arbitrary finite negative rates remain valid despite positive populations.
    negative = RateView(point, DoubleArray(-np.ones(point.y.shape)), [0.3], source_identity=SOURCE,
                        mapping_identity=MAP, origin="physical-rate", raw_coordinates=point.y,
                        raw_rate=-np.ones(point.y.shape))
    plain = form(point.state.layout, [LinearTerm(0, "n", 0)], PARTICLE / VOLUME / SECOND, kind="rate")
    assert_result(negative.linear_form(plain, point=point, arithmetic=DoubleArithmetic()), [Fraction(-1)], exact=True)


@pytest.mark.parametrize("change", ["point", "input_rate", "source", "evaluation", "value", "missing", "unknown"])
def test_actual_rate_source_and_point_bindings_reject_tampering(change):
    point, rate, sources, action, _, _ = rate_source_fixture()
    other = later_point(point)
    with pytest.raises(ContractError):
        if change == "point":
            rate.linear_form(action, point=other, arithmetic=DoubleArithmetic(), sources=sources)
        elif change == "input_rate":
            rate.validate(point, [float(np.nextafter(0.3, 1.0))])
        elif change == "source":
            rate.validate(point, [0.3], "different-source")
        else:
            altered = list(sources)
            if change == "evaluation":
                altered = [replace(source, evaluation_identity="same-untrusted-label") for source in altered]
            elif change == "value":
                altered[0] = replace(altered[0], value=DoubleArray([1.0, 0.0]))
            elif change == "missing":
                altered.pop()
            else:
                altered[0] = replace(altered[0], spec=replace(altered[0].spec, id="other"))
            rate.linear_form(action, point=point, arithmetic=DoubleArithmetic(), sources=altered)


def test_same_point_cannot_mix_different_evaluation_packets():
    point, rate, sources, action, _, _ = rate_source_fixture()
    newer = BoundLinearSource.bind(point, ((sources[0].spec, DoubleArray([-2.0, 0.0])),
                                         (sources[1].spec, sources[1].value)))
    with pytest.raises(ContractError, match="evaluation_mismatch"):
        rate.linear_form(action, point=point, arithmetic=DoubleArithmetic(), sources=(sources[0], newer[1]))
    other = later_point(point)
    stale = BoundLinearSource.bind(other, ((source.spec, source.value) for source in sources))
    with pytest.raises(ContractError, match="point_mismatch"):
        rate.linear_form(action, point=point, arithmetic=DoubleArithmetic(), sources=stale)


@pytest.mark.parametrize("divisor", [3.0, -7.0, 0.3])
@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_explicit_quotient_groups_have_independent_fraction_and_decimal_bounds(divisor, sign, request):
    coordinates, root = fixture()
    words = [np.zeros(root.state.layout.size) for _ in range(4)]
    section = root.state.layout.offsets["phi"]
    for target, number in zip(words, (1.0, 2.0**-54, 2.0**-108, 0.0), strict=True):
        target[section] = number
    words[3][section.stop - 1] = sign * 2.0**-162
    point, increment = coordinates.trial(root, PrimitiveExpansion(words), 1.0, [0.0])
    epsilon = factor("epsilon", 0.7, COULOMB / VOLT / LENGTH)
    dx = factor("dx", divisor, LENGTH)
    action = form(root.state.layout, [LinearTerm(0, "phi", 1, (epsilon,), -1, dx),
                                     LinearTerm(0, "phi", 0, (epsilon,), 1, dx)], COULOMB / (LENGTH * LENGTH),
                  kind="increment")
    result = increment.linear_form(root, point, action, arithmetic=DoubleArithmetic())
    expected = -sign * f(epsilon.value) * f(2.0**-162) / f(divisor)
    check = assert_result(result, [expected], exact=sign == 0)
    with localcontext() as context:
        context.prec = 180
        independent = (-Decimal(sign) * Decimal.from_float(epsilon.value) * Decimal.from_float(2.0**-162)
                       / Decimal.from_float(divisor))
        got = fractions(result.value)[0]
        represented = Decimal(got.numerator) / Decimal(got.denominator)
        bound = Decimal.from_float(float(result.absolute_error_bound.values[0]))
        assert abs(represented - independent) <= bound if expected else represented == 0
    record(request, {**check, "Decimal_precision": 180, "divisor_hex": float(divisor).hex()})


def test_mixed_quotient_and_direct_terms_retain_explicit_error_bound():
    _, point = fixture(1)
    a, b = 0.7, 3.0
    action = form(point.state.layout, [LinearTerm(0, "p", 0, (factor("a", a),), divisor=factor("b", b)),
                                       LinearTerm(0, "n", 0, (factor("comparison", a / b),), -1)],
                  PARTICLE / VOLUME)
    expected = f(a) / f(b) - f(a / b)
    assert_result(point.state.linear_form(action, arithmetic=DoubleArithmetic()), [expected])


@pytest.mark.parametrize("value", [True, "1.0", 2**53 + 1, math.nan, math.inf])
def test_invalid_factor_inputs_are_not_implicitly_coerced(value):
    with pytest.raises(ContractError):
        factor("invalid", value)


def test_longdouble_checks_actual_storage_precision_instead_of_platform_name():
    value = np.longdouble(1)
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(ContractError, match="factor_type"):
            factor("wider-input", value)
    else:
        assert np.finfo(np.longdouble).nmant == 52
        assert factor("binary64-alias", value).value.hex() == (1.0).hex()


@pytest.mark.parametrize("mutation", ["nnz", "row", "field", "index", "bool_index", "fractional_index",
                                       "units", "source", "arity", "zero_divisor", "support"])
def test_layout_units_support_and_factor_contracts_are_checked(mutation):
    _, point = fixture(1)
    layout = point.state.layout
    action = form(layout, [LinearTerm(0, "p", 0)], PARTICLE / VOLUME)
    with pytest.raises(ContractError):
        if mutation == "nnz":
            replace(action, declared_nnz=0)
        elif mutation == "row":
            form(layout, [LinearTerm(1, "p", 0)], PARTICLE / VOLUME)
        elif mutation == "field":
            form(layout, [LinearTerm(0, "unknown", 0)], PARTICLE / VOLUME)
        elif mutation == "index":
            form(layout, [LinearTerm(0, "p", 1)], PARTICLE / VOLUME)
        elif mutation == "bool_index":
            LinearTerm(0, "p", True)
        elif mutation == "fractional_index":
            LinearTerm(0, "p", 0.5)
        elif mutation == "units":
            form(layout, [LinearTerm(0, "p", 0, (factor("wrong-q", 1.0, COULOMB),))], COULOMB / VOLUME)
        elif mutation == "source":
            form(layout, [LinearTerm(0, "p", 0, (replace(factor("x", 1.0), source_identity="foreign"),))], PARTICLE / VOLUME)
        elif mutation == "arity":
            LinearTerm(0, "p", 0, (factor("x", 1.0),) * 5)
        elif mutation == "zero_divisor":
            LinearTerm(0, "p", 0, divisor=factor("zero", 0.0))
        else:
            source = LinearSourceSpec("source", "missing", (1,), PARTICLE / VOLUME, SOURCE)
            form(layout, [LinearTerm(0, "source", 0)], PARTICLE / VOLUME, sources=(source,))


def test_zero_support_duplicate_contributions_and_order_are_structural():
    _, point = fixture(1)
    zero = factor("declared-zero", 0.0)
    terms = [LinearTerm(0, "p", 0, (zero,)), LinearTerm(0, "n", 0), LinearTerm(0, "n", 0, sign=-1)]
    action = form(point.state.layout, terms, PARTICLE / VOLUME)
    reordered = replace(action, terms=tuple(reversed(terms)))
    assert action.identity == reordered.identity and len(action.terms) == 3
    assert_result(point.state.linear_form(action, arithmetic=DoubleArithmetic()), [Fraction()], exact=True)
    moved = replace(action, terms=(LinearTerm(0, "n", 0, (zero,)), *terms[1:]))
    assert moved.identity != action.identity
    with pytest.raises(ContractError, match="layout_or_kind"):
        fixture(2)[1].state.linear_form(action, arithmetic=DoubleArithmetic())


def test_empty_layouts_and_empty_rows_are_explicit():
    layout = Layout((), (), ())
    point = Point(0.0, [], [], StateView(layout), "empty")
    action = PhysicalLinearForm(layout, (), (), (), SOURCE, "state", 0)
    for arithmetic in (FloatArithmetic(), DoubleArithmetic()):
        result = point.state.linear_form(action, arithmetic=arithmetic, point=point)
        assert result.value.shape == result.absolute_error_bound.shape == (0,)
    zero_row = form(layout, [], COULOMB)
    assert_result(point.state.linear_form(zero_row, arithmetic=DoubleArithmetic()), [Fraction()], exact=True)
    rate = RateView(point, PrimitiveExpansion([np.zeros(0)] * 4), [], source_identity=SOURCE,
                    mapping_identity=MAP, origin="mapped-coordinate-rate", raw_coordinates=[], raw_rate=[])
    assert rate.linear_form(replace(action, kind="rate"), point=point,
                            arithmetic=DoubleArithmetic()).value.shape == (0,)


def test_declared_support_has_no_unrelated_grid_size_ceiling():
    _, point = fixture(1)
    term = LinearTerm(0, "n", 0, (factor("declared-zero", 0.0),))
    action = form(point.state.layout, (term,) * 16385, PARTICLE / VOLUME)
    assert action.declared_nnz == len(action.terms) == 16385


@pytest.mark.parametrize("case", ["product_overflow", "product_underflow", "quotient_overflow", "quotient_underflow"])
def test_finite_range_failures_are_explicit(case):
    _, point = fixture(1)
    tiny = float(np.nextafter(0.0, 1.0))
    if case == "product_overflow":
        factors, divisor = (factor("max", np.finfo(float).max), factor("two", 2.0)), None
    elif case == "product_underflow":
        factors, divisor = (factor("tiny", tiny), factor("half", 0.5)), None
    elif case == "quotient_overflow":
        factors, divisor = (factor("max", np.finfo(float).max),), factor("half", 0.5)
    else:
        factors, divisor = (factor("tiny", tiny),), factor("two", 2.0)
    action = form(point.state.layout, [LinearTerm(0, "p", 0, factors, divisor=divisor)], PARTICLE / VOLUME)
    with pytest.raises(ContractError, match="overflow|underflow"):
        point.state.linear_form(action, arithmetic=DoubleArithmetic())


def test_output_keeps_existing_dd_range_limits_without_float_fallback():
    _, point = fixture(1)
    action = form(point.state.layout, [LinearTerm(0, "p", 0, (factor("large", 1e300),))], PARTICLE / VOLUME)
    with pytest.raises(ContractError, match="output_provider_range"):
        point.state.linear_form(action, arithmetic=DoubleArithmetic())


def test_immutable_rate_source_result_and_shape_metadata():
    point, rate, sources, action, primitive, _ = rate_source_fixture()
    identity, word_id = rate.identity, rate.values.identity_bytes()
    for array in (rate.words[0], rate.input_rate, rate.raw_coordinates, rate.raw_rate):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        array.shape = (1, array.size)
        array.dtype = np.int64
    assert rate.shape == point.y.shape and rate.input_rate.shape == point.inputs.shape
    assert rate.identity == identity and rate.values.identity_bytes() == word_id
    assert primitive.words[0].shape == point.y.shape
    with pytest.raises(FrozenInstanceError):
        rate.origin = "physical-rate"
    with pytest.raises(FrozenInstanceError):
        sources[0].evaluation_identity = "other"
    result = rate.linear_form(action, point=point, arithmetic=DoubleArithmetic(), sources=sources)
    view = result.absolute_error_bound.values
    view.shape = (1, 1)
    view.dtype = np.int64
    assert result.absolute_error_bound.values.shape == (1,)
    with pytest.raises(ValueError):
        result.value.high.setflags(write=True)


def test_float_api_is_explicit_and_unchanged_and_other_maps_fail():
    layout = Layout((Support("one", "global", (1,)),),
                    (VariableSpec("x", "owner", "one", (1,), ONE),), ())
    coordinates = LinearCoordinates(layout)
    point = coordinates.point([2.0], 0.0, [])
    action = form(layout, [LinearTerm(0, "x", 0, (factor("scale", 3.0),))], ONE)
    result = point.state.linear_form(action)
    assert isinstance(result.value, np.ndarray)
    np.testing.assert_array_equal(result.value, [6.0])
    raw = FloatArithmetic().array(np.array([1.0, -2.0]))
    np.testing.assert_array_equal(FloatArithmetic().weighted(raw, np.array([3.0, 4.0])), [3.0, -8.0])
    relative = RelativeCoordinates(layout, {"x": "log"})
    root = relative.initial(StateView(layout, [("x", DoubleArray([2.0]))]))
    mapped, _ = relative.advance(root, [0.25], 1.0, [])
    with pytest.raises(ContractError, match="affine_physical_si"):
        mapped.state.linear_form(action, arithmetic=DoubleArithmetic())
    with pytest.raises(ContractError, match="provider_required"):
        root.state.linear_form(action)


def callback_problem():
    layout = Layout((Support("one", "global", (1,)),),
                    (VariableSpec("charge", "owner", "one", (1,), COULOMB),),
                    (EquationSpec("balance", "owner", "one", (1,), COULOMB / SECOND,
                                  derivative_support=("charge",)),))
    point = LinearCoordinates(layout).point([0.0], 0.0, [0.0])
    arithmetic, calls = DoubleArithmetic(), []
    action = form(layout, [LinearTerm(0, "charge", 0)], COULOMB / SECOND, kind="rate")
    structure = SparseStructure((1, 1), [0], [0, 1], "one-structural-slot", layout.identity)
    storage = PhysicalStorage(lambda p: p.state.field("charge"), lambda l, r, d: d.field("charge"),
                              (COULOMB,), SOURCE, arithmetic)

    def residual(p, ydot, adot):
        calls.append(("residual", ydot))
        if isinstance(ydot, RateView):
            return ydot.linear_form(action, point=p, arithmetic=arithmetic).value
        return arithmetic.freeze(arithmetic.array(ydot))

    def linearize(p, ydot, adot):
        calls.append(("linearize", ydot))
        return SparseLinearization(structure.filled([0.0]), structure.filled([1.0]),
                                   np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), structure, SOURCE)

    problem = ValidatedProblem(layout, 1, 1, storage, structure, SOURCE, residual, linearize,
                               lambda l, r, d: d.field("charge"), lambda l, r: structure.filled([1.0]),
                               arithmetic=arithmetic, accepted_rate_mapping_identity=MAP)
    return problem, point, calls


def test_validated_problem_passes_full_rate_to_both_callbacks_without_projection():
    problem, point, calls = callback_problem()
    primitive = PrimitiveExpansion([np.array([v]) for v in (1.0, 2.0**-54, 2.0**-108, 2.0**-162)])
    rate = RateView(point, primitive, [0.5], source_identity=SOURCE, mapping_identity=MAP,
                    origin="mapped-coordinate-rate", raw_coordinates=point.y, raw_rate=[1.0])
    result = problem.residual(point, rate, [0.5])
    problem.linearize(point, rate, [0.5])
    assert calls == [("residual", rate), ("linearize", rate)]
    assert fractions(result)[0] != f(1.0)
    before = len(calls)
    for bad in (replace(problem, accepted_rate_mapping_identity=None),
                replace(problem, accepted_rate_mapping_identity="other-map")):
        with pytest.raises(ContractError, match="not_admitted"):
            bad.residual(point, rate, [0.5])
    with pytest.raises(ContractError, match="input_mismatch"):
        problem.linearize(point, rate, [0.0])
    assert len(calls) == before
    np.testing.assert_array_equal(problem.residual(point, np.array([3.0]), np.array([0.5])).high, [3.0])
    # Dense generic Qyy/Qya/Qyt terms cannot be assumed zero for a typed rate.
    with pytest.raises(ContractError, match="explicit_problem_callback"):
        ImplicitSystem(None, None).residual(point, rate, [0.5])

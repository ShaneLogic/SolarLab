"""NC08 independent Decimal references and public precision-boundary checks."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, KELVIN, ONE, PARTICLE, SECOND, VOLT, VOLUME,
    CellSource, ContractError, EquationSpec, FaceFlux, Geometry, Layout,
    Point, StateView, Support, TermSink, Unit, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArithmetic, DoubleArray, PrimitiveExpansion, RelativeCoordinates,
    bernoulli, decode_point, encode_point,
)


ROOT = Path(__file__).resolve().parents[2]
GATE = next(gate for gate in json.loads(
    Path(__file__).with_name("AnalyticGatesV1.json").read_text()
)["gates"] if gate["id"] == "NC08")


def decimal_words(value: DoubleArray, index=0):
    return (Decimal.from_float(float(value.high.ravel()[index]))
            + Decimal.from_float(float(value.low.ravel()[index])))


def expm1_series(value):
    return sum(value ** k / Decimal(math.factorial(k)) for k in range(1, 6))


def test_primitive_add_preserves_exact_voltage_product_and_remainder():
    theta = [0.00855663413738028, 0.8191527652292634]
    voltage = 0.010463084503981271
    products = [Fraction.from_float(x) * Fraction.from_float(voltage) for x in theta]
    high = np.array([float(value) for value in products])
    low = np.array([float(value - Fraction.from_float(float(word)))
                    for value, word in zip(products, high)])
    lift = PrimitiveExpansion.from_value(DoubleArray(high, low))
    departure = PrimitiveExpansion.from_value([-high[0], 2.0**-120])
    before = (lift.identity_bytes(), departure.identity_bytes())
    result = departure.add(lift)
    for index, product in enumerate(products):
        exact = sum((Fraction.from_float(float(word[index])) for word in result.words), Fraction())
        expected = product + Fraction.from_float(float(departure.high[index]))
        assert exact == expected
    assert result.words[0][0] != 0
    assert (lift.identity_bytes(), departure.identity_bytes()) == before
    assert all(not word.flags.writeable for word in result.words)


def test_primitive_add_cancellation_retains_three_authoritative_words():
    terms = (1.0, 2.0**-54, 2.0**-108, 2.0**-162)
    left = PrimitiveExpansion(tuple(np.array([value]) for value in terms))
    result = left.add([-1.0])
    exact = sum((Fraction.from_float(float(word[0])) for word in result.words), Fraction())
    assert exact == sum((Fraction.from_float(value) for value in terms[1:]), Fraction())
    assert result.words[2][0] != 0
    with pytest.raises(ContractError, match="primitive_projection_would_discard_remainder"):
        result.as_dd()


def test_primitive_add_capacity_and_shape_fail_without_mutation():
    left = PrimitiveExpansion(tuple(np.array([value]) for value in
                                    (1.0, 2.0**-54, 2.0**-108, 2.0**-162)))
    identity = left.identity_bytes()
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        left.add([2.0**-216])
    with pytest.raises(ContractError, match="primitive_shape_mismatch"):
        left.add([1.0, 2.0])
    assert left.identity_bytes() == identity


def test_primitive_add_rejects_finite_input_sum_overflow():
    largest = np.finfo(np.float64).max
    value = PrimitiveExpansion((np.array([largest]), np.zeros(1), np.zeros(1), np.zeros(1)))
    identity = value.identity_bytes()
    with pytest.raises(ContractError, match="primitive_expansion_sum_range"):
        value.add(value)
    assert value.identity_bytes() == identity


def assert_decimal_close(candidate, reference, atol, rtol):
    allowed = Decimal(str(atol)) + Decimal(str(rtol)) * abs(reference)
    assert abs(candidate - reference) <= allowed, (candidate, reference, allowed)


def density_layout(count=1):
    return Layout((Support("cells", "cell", (count,)),),
                  (VariableSpec("n", "carrier", "cells", (count,), PARTICLE / VOLUME, lower=0),), ())


def test_frame_input_old_codec_bytes_and_four_word_rejections_unchanged():
    """Hashes were obtained from the frozen c214679 qualification capsule."""
    from scripts.benchmarks.precision_prototype import FrameInputExpansion, PrimitiveDifference

    layout = density_layout(2)
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    reference = coordinates.initial(StateView(layout, [("n", DoubleArray([3., 3.], [2.**-60]*2))]))
    trial, _ = coordinates.trial(reference, PrimitiveExpansion.from_value([0.25, 0.5]), 1.0)
    for point, expected in ((reference, "dd38ea2ea38fd36e5ca945b2497e7f8c4a23e64e57a9b86e44b8dd0ac558c0b5"),
                            (trial, "b00dd60e645c7a7e6bcbac7c016cbaf5acea90b47c046522d37934724107ae52")):
        raw = json.dumps(encode_point(point), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert hashlib.sha256(raw).hexdigest() == expected
        assert encode_point(decode_point(encode_point(point), layout)) == encode_point(point)
    extended = FrameInputExpansion.from_value([0.0, 0.0])
    with pytest.raises(ContractError, match="primitive_word_count"):
        PrimitiveExpansion(extended.words)
    with pytest.raises(ContractError):
        PrimitiveDifference(extended, PrimitiveExpansion.from_value([0.0, 0.0]))


def test_frame_input_twelve_word_normalization_identity_and_capacity():
    from dataclasses import FrozenInstanceError
    from scripts.benchmarks.precision_prototype import FrameInputExpansion, FrameInputDifference

    terms = tuple(np.array([2.**(-54*i)]) for i in range(12))
    value = FrameInputExpansion(terms)
    assert len(value.words) == 12
    assert sum((Fraction(float(w[0])) for w in value.words), Fraction()) == sum(
        (Fraction(float(w[0])) for w in terms), Fraction())
    assert value.take_flat([0]).reshape((1,)).identity_bytes() == value.identity_bytes()
    assert all(not word.flags.writeable for word in value.words)
    for assignment in (lambda: setattr(value, "_words", ()), lambda: setattr(value, "other", 1)):
        with pytest.raises(FrozenInstanceError):
            assignment()
    with pytest.raises(TypeError):
        np.asarray(value)
    with pytest.raises(ContractError, match="discard_remainder"):
        value.as_dd()
    with pytest.raises(ContractError, match="frame_input_capacity_exceeded"):
        value.add([2.**-648])
    with pytest.raises(ContractError, match="frame_input_sum_range"):
        FrameInputExpansion.from_terms([np.array([np.finfo(float).max])]*2)
    for malformed in (terms[:4], (*terms, terms[-1])):
        with pytest.raises(ContractError, match="frame_input_word_count"):
            FrameInputExpansion(malformed)
    for malformed in ([np.array([2**53+1], dtype=np.int64)], [np.array([True])], [np.array(["1.0"]) ]):
        with pytest.raises(ContractError):
            FrameInputExpansion.from_terms(malformed)
    with pytest.raises(ContractError, match="frame_difference_requires_two_endpoints"):
        FrameInputDifference(value, FrameInputDifference(value, value))
    with pytest.raises(ContractError, match="frame_difference_requires_two_endpoints"):
        FrameInputDifference(value, PrimitiveExpansion.from_value([1.0]))


def test_frame_input_state_source_and_paired_difference_retain_final_word():
    from copy import deepcopy
    from scripts.benchmarks.contract_prototype import BoundLinearSource, LinearFactor, LinearSourceSpec, LinearTerm, PhysicalLinearForm
    from scripts.benchmarks.precision_prototype import FrameInputExpansion, _double_linear_increment_words, _double_linear_state_words

    layout = density_layout(2)
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    reference = coordinates.initial(StateView(layout, [("n", DoubleArray([3., 3.], [2.**-60]*2))]))
    identity = reference.identity
    terms = [np.full(2, 2.**(-54*i)) for i in range(12)]
    terms[-1] = np.array([2.**-594, 0.0])
    current = FrameInputExpansion(terms)
    point, _ = coordinates.trial(reference, current, 1.0)
    factors = tuple(LinearFactor(str(i), 2.0, ONE, "manufactured") for i in range(4))
    form = PhysicalLinearForm(layout, ("difference",), (PARTICLE/VOLUME,),
        (LinearTerm(0, "n", 0, factors), LinearTerm(0, "n", 1, factors, -1)), "manufactured", "state", 2)
    result = point.state.linear_form(form, arithmetic=DoubleArithmetic(), point=point)
    assert result.value.high[0] == 2.**-590 and result.value.low[0] == 0
    assert len(_double_linear_state_words(point.state, "n")) == 14
    spec = LinearSourceSpec("input", "cells", (2,), PARTICLE/VOLUME, "manufactured")
    source_form = PhysicalLinearForm(layout, ("difference",), (PARTICLE/VOLUME,),
        (LinearTerm(0, "input", 0), LinearTerm(0, "input", 1, sign=-1)), "manufactured", "state", 2, (spec,))
    result = point.state.linear_form(source_form, arithmetic=DoubleArithmetic(), point=point,
        sources=BoundLinearSource.bind(point, [(spec, current)]))
    assert result.value.high[0] == 2.**-594
    terms[-1] = np.array([2.**-593, 0.0])
    right, increment = coordinates.trial(reference, FrameInputExpansion(terms), 2.0, predecessor=point)
    assert len(_double_linear_increment_words(point.state, right.state, "n")) == 24
    delta_form = PhysicalLinearForm(layout, ("difference",), (PARTICLE/VOLUME,), form.terms, "manufactured", "increment", 2)
    result = increment.linear_form(point, right, delta_form, arithmetic=DoubleArithmetic())
    assert result.value.high[0] == 2.**-590 and result.value.low[0] == 0
    payload = encode_point(right)
    assert payload["payload"]["authority"]["schema"] == "solarlab.state-authority.v6"
    restored = decode_point(payload, layout)
    assert encode_point(restored) == payload and reference.identity == identity
    for mutation in (lambda p: p["authority"].update(schema="solarlab.state-authority.v5"),
                     lambda p: p["authority"]["primitives"]["n"]["words"].pop()):
        bad = deepcopy(payload); mutation(bad["payload"])
        bad["sha256"] = hashlib.sha256(json.dumps(bad["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with pytest.raises(ContractError):
            decode_point(bad, layout)
    with pytest.raises(ContractError, match="frame_input_requires_paired_transition"):
        coordinates.trial(reference, current, 3.0, transition_representation="four-word-v1")
    old, _ = coordinates.trial(reference, PrimitiveExpansion.from_value([0.0, 0.0]), 0.5)
    with pytest.raises(ContractError, match="frame_input_predecessor_profile"):
        coordinates.trial(reference, current, 3.0, predecessor=old)


@pytest.mark.parametrize("framed", [False, True])
def test_frame_input_standalone_state_linear_form(framed):
    from scripts.benchmarks.contract_prototype import LinearTerm, PhysicalLinearForm, RateView
    from scripts.benchmarks.precision_prototype import FrameInputExpansion

    layout = density_layout()
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    reference = coordinates.initial(StateView(layout, [("n", DoubleArray([3.]))]))
    kind = FrameInputExpansion if framed else PrimitiveExpansion
    point, _ = coordinates.trial(reference, kind.from_value([0.25]), 1.)
    form = PhysicalLinearForm(layout, ("n",), (PARTICLE/VOLUME,), (LinearTerm(0, "n", 0),),
                              "manufactured-edge", "state", 1)
    result = point.state.linear_form(form, arithmetic=DoubleArithmetic())
    assert result.value.high[0] == 3.25 and result.value.low[0] == 0.
    rate = RateView(point, kind.from_value([0.5]), [], source_identity="manufactured-edge",
                    mapping_identity="manufactured-map", origin="mapped-coordinate-rate",
                    raw_coordinates=[0.25], raw_rate=[0.5])
    rate_form = PhysicalLinearForm(layout, ("dn",), (PARTICLE/VOLUME/SECOND,), form.terms,
                                   "manufactured-edge", "rate", 1)
    result = rate.linear_form(rate_form, arithmetic=DoubleArithmetic(), point=point)
    assert result.value.high[0] == 0.5 and result.value.low[0] == 0.


@pytest.mark.parametrize("framed", [False, True])
def test_frame_input_same_point_zero_increment_ignores_past_transition(framed):
    from scripts.benchmarks.contract_prototype import LinearTerm, PhysicalLinearForm, StateIncrement
    from scripts.benchmarks.precision_prototype import FrameInputExpansion

    layout = density_layout()
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    reference = coordinates.initial(StateView(layout, [("n", DoubleArray([3.]))]))
    kind = FrameInputExpansion if framed else PrimitiveExpansion
    first, _ = coordinates.trial(reference, kind.from_value([0.25]), 1.)
    second, _ = coordinates.trial(reference, kind.from_value([0.5]), 2., predecessor=first)
    form = PhysicalLinearForm(layout, ("dn",), (PARTICLE/VOLUME,), (LinearTerm(0, "n", 0),),
                              "manufactured-edge", "increment", 1)
    for point in (first, second):
        increment = StateIncrement.from_points(point, point)
        result = increment.linear_form(point, point, form, arithmetic=DoubleArithmetic())
        assert result.value.high[0] == result.value.low[0] == 0.
        assert result.absolute_error_bound.values[0] == 0.


@pytest.mark.parametrize("text", GATE["decimal_inputs"]["delta_log_n"])
def test_nc08_weak_density_survives_state_increment_and_codec(text):
    layout = density_layout()
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    state = StateView(layout, [("n", DoubleArray([float(GATE["decimal_inputs"]["n0"])]))])
    left = coordinates.initial(state)
    right, increment = coordinates.advance(left, [float(text)], 1)
    increment.validate(left, right)
    with localcontext() as context:
        context.prec = 80
        n0, x = Decimal(GATE["decimal_inputs"]["n0"]), Decimal(text)
        reference = n0 * expm1_series(x)
        candidate = decimal_words(increment.field("n"))
        quantization = abs(n0 * (expm1_series(Decimal.from_float(float(text))) - expm1_series(x)))
        assert quantization <= Decimal(str(GATE["density_increment_atol"])) / 3
        assert_decimal_close(candidate, reference, GATE["density_increment_atol"], GATE["density_increment_rtol"])
        state_change = decimal_words(right.state.field("n")) - decimal_words(left.state.field("n"))
        assert_decimal_close(state_change, reference, GATE["density_increment_atol"], GATE["density_increment_rtol"])
        assert (candidate == 0) == (x == 0)
        assert right.state.field("n").high[0] == left.state.field("n").high[0]
        restored = decode_point(json.loads(json.dumps(encode_point(right))), layout)
        assert restored.identity == right.identity
        assert restored.state.field("n").identity_bytes() == right.state.field("n").identity_bytes()
        # Subsequent evaluation uses restored words, not the rounded high value.
        recovered_change = restored.state.field("n").difference(left.state.field("n"))
        assert_decimal_close(decimal_words(recovered_change), reference,
                             GATE["density_increment_atol"], GATE["density_increment_rtol"])


@pytest.mark.parametrize("text", GATE["decimal_inputs"]["delta_log_n"])
def test_nc08_weak_flux_and_capture_use_stable_physical_differences(text):
    layout = density_layout(2)
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    n0 = float(GATE["decimal_inputs"]["n0"])
    left = coordinates.initial(StateView(layout, [("n", DoubleArray([n0, n0]))]))
    right, increment = coordinates.advance(left, [0, float(text)], 1)
    face_difference = right.state.face_difference("n", [[0, 1]])
    weak = GATE["weak_flux"]
    flux = face_difference.as_dd() * (DD(float(weak["q_C"])) * weak["D_m2_s"] / weak["dx_m"])
    capture = GATE["capture"]
    reaction = (increment.field("n").as_dd()[1] * capture["capture_coefficient"]
                * capture["trap_density"] * (1 - capture["occupancy"]))
    with localcontext() as context:
        context.prec = 80
        dn = Decimal(GATE["decimal_inputs"]["n0"]) * expm1_series(Decimal(text))
        expected_flux = Decimal(weak["q_C"]) * Decimal(str(weak["D_m2_s"])) / Decimal(str(weak["dx_m"])) * dn
        assert_decimal_close(decimal_words(DoubleArray.from_dd(flux)), expected_flux,
                             weak["atol_A_m2"], weak["rtol"])
        expected_capture = (Decimal(str(capture["capture_coefficient"])) * Decimal(str(capture["trap_density"]))
                            * (1 - Decimal(str(capture["occupancy"]))) * dn)
        assert_decimal_close(decimal_words(DoubleArray.from_dd(reaction)), expected_capture,
                             capture["atol"], capture["rtol"])


@pytest.mark.parametrize("text", GATE["decimal_inputs"]["delta_log_n"])
def test_stable_bernoulli_resolves_weak_correction_on_density_scale(text):
    value = bernoulli(DoubleArray([float(text)]))
    with localcontext() as context:
        context.prec = 80
        x, n0 = Decimal(text), Decimal(GATE["decimal_inputs"]["n0"])
        # Independent Bernoulli series; next term is O(x^6), <1e-96 here.
        correction = -x / 2 + x*x / 12 - x**4 / 720
        candidate = n0 * (decimal_words(value) - 1)
        assert_decimal_close(candidate, n0 * correction,
                             GATE["density_increment_atol"], GATE["density_increment_rtol"])
    if text == "0":
        np.testing.assert_array_equal(value.high, [1])
        np.testing.assert_array_equal(value.low, [0])


def test_nc08_displacement_current_uses_increment_not_equal_high_words():
    fixture = GATE["displacement"]
    layout = Layout((Support("face", "face", (1,)),),
                    (VariableSpec("D", "electrostatics", "face", (1,), COULOMB / AREA),), ())
    coordinates = RelativeCoordinates(layout, {"D": "linear"})
    left = coordinates.initial(StateView(layout, [("D", DoubleArray([float(fixture["D0_C_m2"])]))]))
    right, increment = coordinates.advance(left, [float(fixture["increment_C_m2"])], float(fixture["dt_s"]))
    assert right.state.field("D").high[0] == left.state.field("D").high[0]
    current = DoubleArray.from_dd(increment.field("D").as_dd() / float(fixture["dt_s"]))
    with localcontext() as context:
        context.prec = 80
        assert_decimal_close(decimal_words(current), Decimal(fixture["expected_current_A_m2"]),
                             fixture["atol_A_m2"], fixture["rtol"])


def test_precision_reaches_geometry_weighting_and_cancellation():
    layout = Layout((Support("cells", "cell", (2,)), Support("faces", "face", (1,))), (),
                    (EquationSpec("balance", "assembly", "cells", (2,), PARTICLE / SECOND),))
    geometry = Geometry([2, 3], [[0, 1]], [5])
    sink = TermSink(layout, "balance", geometry, DoubleArithmetic())
    flux = DoubleArray.from_dd(DD([1e22], [1e5]))
    sink.add(FaceFlux("transport", "faces", flux, PARTICLE / AREA / SECOND))
    # Choose the counterterm using DD arithmetic so binary64 division error
    # does not contaminate the deliberately small cancellation witness.
    source = DoubleArray.from_dd(DD([1e22, -1e22]) * 5 / DD([2.0, 3.0]))
    sink.add(CellSource("counterterm", "cells", source, PARTICLE / VOLUME / SECOND))
    result = sink.value()
    with localcontext() as context:
        context.prec = 80
        assert_decimal_close(decimal_words(result, 0), Decimal("-5e5"), 1e-7, 1e-12)
        assert_decimal_close(decimal_words(result, 1), Decimal("5e5"), 1e-7, 1e-12)
        assert abs(decimal_words(result, 0) + decimal_words(result, 1)) <= Decimal("1e-7")
    with pytest.raises(TypeError, match="implicit precision rounding"):
        np.asarray(result)


def test_exact_zero_and_explicit_rebase_preserve_physical_history():
    layout = density_layout()
    inactive = RelativeCoordinates(layout, {"n": "inactive"})
    zero = inactive.initial(StateView(layout, [("n", DoubleArray([0]))]))
    right, increment = inactive.advance(zero, [0], 1)
    assert decimal_words(right.state.field("n")) == decimal_words(increment.field("n")) == 0
    with pytest.raises(ContractError, match="inactive_coordinate_change"):
        inactive.advance(zero, [1e-17], 1)
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    left = coordinates.initial(StateView(layout, [("n", DoubleArray([1e22]))]))
    plus, _ = coordinates.advance(left, [1e-17], 1)
    restored = decode_point(encode_point(plus), layout)
    back, _ = coordinates.advance(restored, [-1e-17], 2)
    assert back.coordinate_reference.endswith(restored.identity)
    with localcontext() as context:
        context.prec = 80
        assert abs(decimal_words(back.state.field("n")) - Decimal("1e22")) <= Decimal("1e-7")


def test_codec_preserves_zero_signs_and_rejects_tampered_words():
    layout = density_layout(2)
    state = StateView(layout, [("n", DoubleArray([0.0, -0.0], [-0.0, 0.0]))])
    point = RelativeCoordinates(layout, {"n": "inactive"}).initial(state)
    restored = decode_point(encode_point(point), layout)
    assert restored.state.field("n").identity_bytes() == point.state.field("n").identity_bytes()
    record = encode_point(point)
    record["payload"]["fields"]["n"]["low"][0] = (1e-30).hex()
    with pytest.raises(ContractError, match="precision_codec_digest_mismatch"):
        decode_point(record, layout)


@pytest.mark.parametrize("case_name", ["step183_failed", "step183_accepted", "step309_failed"])
def test_nc08_saved_r1_inputs_have_exact_public_representation(case_name):
    path = ROOT / "tests/fixtures/refactor/R1FailureWitnessV1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "29e4981bd4d5d11d19f3120febbd47d853d39bfb45a72a44ecec91566757db1d"
    case = json.loads(path.read_text())["cases"][case_name]
    row = case["saved_row"] if case_name == "step183_accepted" else None
    raw = row["state"] if row else case["saved_witness"]["attempted_state"]
    coordinates = row["physics_reconstruction"]["coordinate"] if row else case["saved_witness"]["coordinate"]
    units = {"n_m3": PARTICLE / VOLUME, "p_m3": PARTICLE / VOLUME,
             "positive_m3": PARTICLE / VOLUME, "phi_V": VOLT, "occupancy": ONE,
             "trace_state_m3": PARTICLE / VOLUME, "trace_potential_V": VOLT}
    supports, variables, fields = [], [], []
    for name, unit in units.items():
        value = np.asarray(raw[name], dtype=float)
        supports.append(Support(name, "global", value.shape))
        variables.append(VariableSpec(name, "frozen_r1_input", name, value.shape, unit))
        fields.append((name, DoubleArray(value)))
    layout = Layout(tuple(supports), tuple(variables), ())
    point = Point(0, np.asarray(coordinates), np.empty(0), StateView(layout, fields),
                  f"frozen-original-r1:{case_name}")
    restored = decode_point(json.loads(json.dumps(encode_point(point))), layout)
    assert point.identity == restored.identity
    assert point.y.tobytes() == restored.y.tobytes()
    for name in units:
        assert point.state.field(name).identity_bytes() == restored.state.field(name).identity_bytes()
        assert restored.state.field(name).high.tobytes() == np.asarray(raw[name], dtype=float).tobytes()
    # This is representation/codec evidence only. The changed Fermi table and
    # original 100s/R1 qualification remain explicitly outside this check.


def test_voltage_energy_and_temperature_dimensions_are_distinct():
    assert COULOMB * VOLT == Unit((2, -2, 0, 0, 1, 0))
    assert (COULOMB * VOLT / KELVIN) * KELVIN == COULOMB * VOLT
    assert VOLT != ONE


def test_core_uses_injected_arithmetic_without_importing_research_provider():
    import ast
    path = ROOT / "scripts/benchmarks/contract_prototype.py"
    modules = [node.module for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ("precision" in module or "compensated" in module
                              or module.startswith("perovskite_sim")) for module in modules)


@pytest.mark.parametrize("component", ["high", "low"])
def test_precision_field_metadata_cannot_change_owned_state(component):
    layout = density_layout(2)
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    point = coordinates.initial(StateView(layout, [("n", DoubleArray([1, 2]))]))
    value = point.state.field("n")
    before = value.identity_bytes()
    view = getattr(value, component)
    view.shape = (1, 2)
    view.dtype = np.int64
    assert value.shape == (2,)
    assert getattr(value, component).dtype == np.float64
    assert value.identity_bytes() == before
    assert decode_point(encode_point(point), layout).identity == point.identity


def test_public_float_history_arrays_own_metadata_as_well_as_bytes():
    from scripts.benchmarks.contract_prototype import AcceptedStep, LinearCoordinates, LinearStorage
    layout = density_layout(2)
    coordinates = LinearCoordinates(layout)
    left = coordinates.point([1, 2], inputs=[0, 1])
    right, increment = coordinates.advance(left, [0.1, -0.1], 1, [1, 2])
    step = AcceptedStep(left, right, increment, [0.1, -0.1], [0.1, -0.1], [0.1, -0.1], [1, 1],
                        "conservative-be-v1", "continuous", (("inventory_defect", 0.0),), True)
    storage = LinearStorage(np.eye(2))
    reads = [lambda: left.y, lambda: left.inputs, lambda: left.state.field("n").values,
             lambda: step.derivative, lambda: step.storage_delta, lambda: step.displacement_delta,
             lambda: step.input_rate, lambda: storage.matrix]
    for read in reads:
        original = read().copy()
        view = read()
        view.shape = (view.size, 1)
        view.dtype = np.int64
        assert read().shape == original.shape
        assert read().dtype == np.float64
        np.testing.assert_array_equal(read(), original)


@pytest.mark.parametrize("value", [[2**53 + 1], ["1.00000000000000000001"]])
def test_double_constructor_preserves_reused_arithmetic_input_guards(value):
    with pytest.raises((TypeError, ValueError)):
        DoubleArray(value)


@pytest.mark.parametrize("mode,value", [("linear", -1), ("linear", 3), ("log", 3)])
def test_initial_physical_anchor_must_satisfy_declared_bounds(mode, value):
    layout = Layout((Support("cells", "cell", (1,)),),
                    (VariableSpec("n", "carrier", "cells", (1,), PARTICLE / VOLUME, lower=0, upper=2),), ())
    coordinates = RelativeCoordinates(layout, {"n": mode})
    with pytest.raises(ContractError, match="physical_state_outside_domain"):
        coordinates.initial(StateView(layout, [("n", DoubleArray([value]))]))


def test_large_log_decrease_retains_positive_endpoint_separately_from_increment():
    layout = density_layout()
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    first = coordinates.initial(StateView(layout, [("n", DoubleArray([1]))]))
    weak, _ = coordinates.advance(first, [1e-17], 1)
    down, increment = coordinates.advance(weak, [-80], 2)
    with localcontext() as context:
        context.prec = 80
        expected = (Decimal.from_float(1e-17) - Decimal(80)).exp()
        result = decimal_words(down.state.field("n"))
        assert result > 0
        assert abs(result - expected) / expected < Decimal("1e-28")
        back, _ = coordinates.advance(down, [80], 3)
        assert abs(decimal_words(back.state.field("n")) - decimal_words(weak.state.field("n"))) < Decimal("1e-28")
        # A finite delta near -1 cannot encode a 1e-35 remainder at DD precision;
        # it must not replace the separately computed positive endpoint.
        assert decimal_words(increment.field("n")) < 0

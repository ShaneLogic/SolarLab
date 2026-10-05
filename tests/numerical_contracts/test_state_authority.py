"""Map-authority regressions with independent Decimal physical input oracles.

These tests adopt the explicit mapped input. They do not relabel the retained
old endpoint-word or stored-increment sign failures as passes.
"""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    ONE, PARTICLE, VOLT, VOLUME, AcceptedStep, ContractError,
    FloatArray, Layout, LinearCoordinates, PhysicalStorage, Point,
    StateIncrement, StateView, Support, TerminalPort, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArithmetic, DoubleArray, InputLiftCoordinates, MappedArray, PrimitiveExpansion,
    RelativeCoordinates, bernoulli, decode_point, encode_point,
)


TEXTS = ["0", "1e-17", "-1e-17", "1e-16", "-1e-16"]
VT = 0.025851999786435535


def words(value, index=0):
    return Decimal.from_float(float(value.high.ravel()[index]))+Decimal.from_float(float(value.low.ravel()[index]))


def state_fixture(gauge=0.0):
    layout = Layout((Support("cells", "cell", (2,)),),
                    (VariableSpec("n_m3", "carrier", "cells", (2,), PARTICLE / VOLUME, lower=0),
                     VariableSpec("phi_V", "electrostatics", "cells", (2,), VOLT)), ())
    coordinates = RelativeCoordinates(layout, {"n_m3": "log", "phi_V": "linear"})
    initial = coordinates.initial(StateView(layout, [("n_m3", DoubleArray([1e22, 1e22])),
                                                     ("phi_V", DoubleArray([gauge, gauge]))]), inputs=[0])
    return coordinates, initial


def mapped_hole_current(state):
    """SG evaluation through the public density/potential map operations."""
    log_density = state.log_ratio("n_m3", [[0, 1]]).as_dd()
    xi = state.face_difference("phi_V", [[0, 1]]).as_dd()/VT
    right = state.field("n_m3").take([1]).as_dd()
    bminus = bernoulli(DoubleArray.from_dd(-xi)).as_dd()
    affinity = -log_density-xi
    # All frozen weak controls use the stable equivalent backward-flux form.
    value = bminus*right*affinity.expm1()
    return DoubleArray.from_dd(DD(1.602176634e-19)*1e-4/1e-8*value)


@pytest.mark.parametrize("gauge", [0.0, 1.0])
@pytest.mark.parametrize("text", TEXTS)
def test_ten_mixed_hole_inputs_use_one_map_before_and_after_codec(gauge, text):
    coordinates, left = state_fixture(gauge)
    x, voltage = float(text), -VT*float(text)
    right, increment = coordinates.advance(left, [0, x, 0, voltage], 1, [0])
    restored = decode_point(json.loads(json.dumps(encode_point(right))), right.state.layout)
    assert isinstance(right.state.field("n_m3"), MappedArray)
    assert right.state.field("n_m3").authority.identity == right.state.authority.identity
    assert restored.identity == right.identity
    assert StateIncrement.from_points(left, restored).field("n_m3").identity_bytes() == increment.field("n_m3").identity_bytes()
    with localcontext() as context:
        context.prec = 100
        d = Decimal.from_float(x)
        xi = Decimal.from_float(voltage)/Decimal.from_float(VT)
        density = Decimal(10**22)*d.exp()
        bm = Decimal(1) if xi == 0 else (-xi)/((-xi).exp()-1)
        expected = (Decimal.from_float(1.602176634e-19)*Decimal.from_float(1e-4)
                    / Decimal.from_float(1e-8)*bm*density*((-d-xi).exp()-1))
        for state in (right.state, restored.state):
            current = words(mapped_hole_current(state))
            assert abs(current-expected) <= Decimal("1e-40")+abs(expected)*Decimal("1e-12")
            assert (current == 0) == (expected == 0)
            assert (current > 0) == (expected > 0)
        if text == "1e-16":
            assert expected < 0  # The retained rounded-endpoint oracle was positive.
        difference = right.state.face_difference("n_m3", [[0, 1]])
        assert abs(words(difference)-Decimal(10**22)*(d.exp()-1)) < Decimal("1e-20")
        assert words(right.state.log_ratio("n_m3", [[0, 1]])) == d


def test_increment_values_cannot_be_forged_for_the_same_float_or_mapped_pair():
    coordinates, left = state_fixture()
    right, increment = coordinates.advance(left, [0, 1e-17, 0, 0], 1, [0])
    fields = dict(increment.fields)
    fields["n_m3"] = DoubleArray.from_dd(fields["n_m3"].as_dd()+DD([1, 0]))
    with pytest.raises(ContractError, match="increment_authority_mismatch"):
        StateIncrement(left.identity, right.identity, fields).validate(left, right)
    linear = LinearCoordinates(left.state.layout)
    a = linear.point([1, 2, 0, 0])
    b, change = linear.advance(a, [1, 0, 0, 0], 1)
    fields = dict(change.fields)
    fields["n_m3"] = FloatArray([2, 0])
    with pytest.raises(ContractError, match="increment_authority_mismatch"):
        StateIncrement(a.identity, b.identity, fields).validate(a, b)


def test_map_binding_and_projection_cannot_silently_create_new_authority():
    coordinates, left = state_fixture()
    right, _ = coordinates.advance(left, [0, 1e-17, 0, 0], 1, [0])
    for replacement in ({"time": 2}, {"inputs": np.array([1.])}, {"y": np.zeros(4)},
                        {"coordinate_reference": "unresolved"}):
        with pytest.raises(ContractError, match="point_authority_binding_mismatch"):
            replace(right, **replacement)
    with pytest.raises(ContractError, match="mapped_field_requires_its_authority"):
        StateView(right.state.layout, right.state.fields.items())
    projection = right.state.project_field("n_m3")
    assert type(projection.value) is DoubleArray
    assert projection.authority_identity == right.state.authority.identity
    with pytest.raises(ContractError, match="projection_not_certified"):
        projection.require_error([1e10, 1e10])
    with pytest.raises(ContractError, match="projection_not_certified"):
        coordinates.rebase(right, {"n_m3": [1e10, 1e10], "phi_V": [1e10, 1e10]})
    rebased = coordinates.rebase(left, {"n_m3": [0, 0], "phi_V": [0, 0]})
    assert rebased.state.authority.root_identity == left.state.authority.root_identity


def test_accepted_composition_codec_and_depletion_keep_a_positive_direct_map_endpoint():
    layout = Layout((Support("cell", "global", (1,)),),
                    (VariableSpec("n", "carrier", "cell", (1,), ONE, lower=0),), ())
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    root = coordinates.initial(StateView(layout, [("n", DoubleArray([1]))]))
    weak, _ = coordinates.advance(root, [1e-17], 1)
    restored = decode_point(encode_point(weak), layout)
    down, delta = coordinates.advance(restored, [-80], 2)
    up, _ = coordinates.advance(decode_point(encode_point(down), layout), [80], 3)
    with localcontext() as context:
        context.prec = 100
        expected = (Decimal.from_float(1e-17)-80).exp()
        assert words(down.state.field("n")) > 0
        assert abs(words(down.state.field("n"))-expected) < expected*Decimal("1e-28")
        assert words(delta.field("n"))+words(weak.state.project_field("n").value) == 0
        assert abs(words(up.state.field("n"))-Decimal.from_float(1e-17).exp()) < Decimal("1e-28")
    assert down.state.authority.root_identity == root.state.authority.root_identity
    assert up.state.authority.root_identity == root.state.authority.root_identity
    point = root
    for step in range(1, 11):
        point, _ = coordinates.advance(point, [1e-17], step)
    for step in range(11, 21):
        point, _ = coordinates.advance(decode_point(encode_point(point), layout), [-1e-17], step)
    assert np.all(point.state.authority.primitives["n"].as_dd() == 0)
    assert words(point.state.field("n")) == 1


def test_fixed_primitive_expansion_retains_four_words_and_refuses_a_fifth():
    layout = Layout((Support("cell", "global", (1,)),),
                    (VariableSpec("n", "carrier", "cell", (1,), ONE),), ())
    coordinates = RelativeCoordinates(layout, {"n": "linear"})
    root = coordinates.initial(StateView(layout, [("n", DoubleArray([0]))]))
    first, _ = coordinates.advance(root, DoubleArray([1], [2.0**-54]), 1)
    second, delta = coordinates.advance(first, [2.0**-108], 2)
    third, _ = coordinates.advance(second, [2.0**-162], 3)
    np.testing.assert_array_equal([word[0] for word in third.state.authority.primitives["n"].words],
                                  [1, 2.0**-54, 2.0**-108, 2.0**-162])
    assert delta.field("n").high[0] == 2.0**-108 and delta.field("n").low[0] == 0
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        coordinates.advance(third, [2.0**-216], 4)
    with pytest.raises(ContractError, match="primitive_projection_would_discard_remainder"):
        third.state.authority.primitives["n"].as_dd()
    assert first.state.authority.local_primitives["n"].words[1][0] == 2.0**-54


def test_primitive_expansion_owns_word_metadata_and_rejects_invalid_encodings():
    value = PrimitiveExpansion(([1.0, 2.0], [2.0**-54, 2.0**-53], [2.0**-108]*2, [2.0**-162]*2))
    identity = value.identity_bytes()
    view = value.words[2]
    view.shape = (1, 2)
    view.dtype = np.int64
    assert value.words[2].shape == (2,) and value.words[2].dtype == np.float64
    assert value.identity_bytes() == identity
    with pytest.raises(ValueError):
        value.words[0].setflags(write=True)
    with pytest.raises(TypeError, match="primitive projection"):
        np.asarray(value)
    with pytest.raises(ContractError, match="primitive_word_count"):
        PrimitiveExpansion(([1.], [0.]))
    with pytest.raises(ContractError, match="unnormalized_primitive_words"):
        PrimitiveExpansion(([1.], [1.], [0.], [0.]))
    with pytest.raises(ContractError, match="primitive_word_dtype"):
        PrimitiveExpansion((["1"], [0.], [0.], [0.]))
    with pytest.raises(ContractError, match="inexact_primitive_integer"):
        PrimitiveExpansion(([2**53+1], [0.], [0.], [0.]))


def test_empty_precision_state_retains_prescribed_input_and_transition_identity():
    layout = Layout((), (), ())
    coordinates = RelativeCoordinates(layout, {})
    left = coordinates.initial(StateView(layout), inputs=[0])
    right, increment = coordinates.advance(left, [], 1, [2])
    assert not increment.fields
    restored = decode_point(encode_point(right), layout)
    assert restored.identity == right.identity
    assert restored.state.authority.previous_point_identity == left.identity
    np.testing.assert_array_equal(restored.inputs, [2])


def test_transition_must_resolve_the_actual_previous_primitives_not_only_its_ID():
    layout = Layout((Support("cell", "global", (1,)),),
                    (VariableSpec("x", "linear", "cell", (1,), ONE),), ())
    coordinates = RelativeCoordinates(layout, {"x": "linear"})
    root = coordinates.initial(StateView(layout, [("x", DoubleArray([0]))]))
    left, _ = coordinates.advance(root, [1], 1)
    right, _ = coordinates.advance(left, [1], 2)
    forged = replace(right.state.authority, previous_primitives={"x": DoubleArray([0])},
                     local_primitives={"x": DoubleArray([2])}, coordinates=np.array([2.]))
    point = Point(2, np.array([2.]), np.empty(0), StateView(layout, authority=forged), forged.reference)
    with pytest.raises(ContractError, match="increment_transition_primitive_mismatch"):
        StateIncrement.from_points(left, point)
    with pytest.raises(ContractError, match="authority_coordinate_projection_mismatch"):
        replace(right.state.authority, coordinates=np.array([9.]))


def _resign(record):
    record["sha256"] = hashlib.sha256(json.dumps(record["payload"], sort_keys=True,
                                                 separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return record


@pytest.mark.parametrize("part", ["anchor", "mode", "primitive", "transition", "previous_point", "version", "projection", "role"])
def test_codec_rejects_changed_meaning_even_with_a_recomputed_outer_digest(part):
    coordinates, left = state_fixture()
    right, _ = coordinates.advance(left, [0, 1e-17, 0, 0], 1, [0])
    record = json.loads(json.dumps(encode_point(right)))
    authority = record["payload"]["authority"]
    if part == "anchor": del authority["anchor"]["n_m3"]
    elif part == "mode": authority["modes"]["n_m3"] = "linear"
    elif part == "primitive": authority["primitives"]["n_m3"]["words"][1][1] = (1e-40).hex()
    elif part == "transition": authority["transition"]["previous_authority"] = "unresolved"
    elif part == "previous_point": authority["transition"]["previous_point"] = "0"*64
    elif part == "version": authority["version"] = 99
    elif part == "projection": record["payload"]["fields"]["n_m3"]["high"][1] = (2e22).hex()
    elif part == "role": record["payload"]["field_role"] = "resolved-input"
    with pytest.raises((ContractError, ValueError)):
        decode_point(_resign(record), right.state.layout)


def test_named_r1_lift_preserves_VT_full_lifts_zero_branches_and_coupled_activity():
    names = ["n_m3", "p_m3", "phi_V", "dqfn_V", "dqfp_V", "positive_m3", "occupancy",
             "trace_state_m3", "trace_potential_V"]
    volts = {"phi_V", "dqfn_V", "dqfp_V", "trace_potential_V"}
    layout = Layout((Support("nodes", "global", (2,)),), tuple(
        VariableSpec(name, "frozen-input", "nodes", (2,), VOLT if name in volts else ONE,
                     lower=0 if name not in volts else None, upper=1 if name == "occupancy" else None)
        for name in names), ())
    fields = {name: DoubleArray([0, 0] if name in volts or name == "positive_m3" else [1, 1]) for name in names}
    fields["occupancy"] = DoubleArray([0, 1])
    left = Point(0, np.zeros(2), np.array([0.]), StateView(layout, fields.items()), "explicit-r1-input-v1")
    coords = InputLiftCoordinates(layout, VT)
    drive = np.array([1e-17, -1e-17])
    bulk_lift = DoubleArray.from_dd(-DD(VT)*DD(drive))
    trace_lift = DoubleArray.from_dd(-DD(VT)*DD(drive[::-1]))
    right, increment = coords.advance(left, {"electron": -drive, "hole": drive, "potential": drive,
                                            "positive": drive, "occupancy": [-80, 80],
                                            "trace_density": drive, "trace_potential": drive[::-1]},
                                      1, bulk_lift, trace_lift, [0])
    for name in ("n_m3", "p_m3", "phi_V", "dqfn_V", "dqfp_V", "positive_m3", "occupancy", "trace_potential_V"):
        assert np.all(increment.field(name).as_dd() == 0), name
    restored = decode_point(json.loads(json.dumps(encode_point(right))), layout)
    assert restored.identity == right.identity
    assert restored.state.authority.parameters["thermal_voltage_V"] == VT
    assert restored.state.authority.local_primitives["voltage_lift"].identity_bytes() == PrimitiveExpansion.from_value(bulk_lift).identity_bytes()
    assert restored.state.authority.local_primitives["trace_voltage_lift"].identity_bytes() == PrimitiveExpansion.from_value(trace_lift).identity_bytes()
    assert np.all(restored.state.log_ratio("n_m3", [[0, 1]]).as_dd() == 0)
    assert np.all(restored.state.electrochemical_difference("n_m3", "phi_V", [[0, 1]], VT,
                                                         potential_sign=-1, arithmetic=DoubleArithmetic()).as_dd() == 0)
    for key in ("thermal_voltage_V", "voltage_lift", "trace_voltage_lift"):
        record = json.loads(json.dumps(encode_point(right)))
        if key == "thermal_voltage_V": record["payload"]["authority"]["parameters"][key] = (VT*2).hex()
        else: record["payload"]["authority"]["primitives"][key]["words"][0][0] = (1e-4).hex()
        with pytest.raises(ContractError): decode_point(_resign(record), layout)


@pytest.mark.parametrize("case_name", ["step183_failed", "step183_accepted", "step309_failed"])
def test_original_R1_physical_snapshots_roundtrip_the_named_map_without_private_state(case_name):
    path = Path(__file__).resolve().parents[1]/"fixtures/refactor/R1FailureWitnessV1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "29e4981bd4d5d11d19f3120febbd47d853d39bfb45a72a44ecec91566757db1d"
    case = json.loads(path.read_text())["cases"][case_name]
    raw = case["saved_row"]["state"] if case_name == "step183_accepted" else case["saved_witness"]["attempted_state"]
    units = {"n_m3": PARTICLE/VOLUME, "p_m3": PARTICLE/VOLUME, "positive_m3": PARTICLE/VOLUME,
             "phi_V": VOLT, "occupancy": ONE, "trace_state_m3": PARTICLE/VOLUME, "trace_potential_V": VOLT}
    supports, variables, fields = [], [], []
    for name, unit in units.items():
        value = np.asarray(raw[name], dtype=float)
        supports.append(Support(name, "global", value.shape))
        variables.append(VariableSpec(name, "original-physical-input", name, value.shape, unit))
        fields.append((name, DoubleArray(value)))
    layout = Layout(tuple(supports), tuple(variables), ())
    root = Point(0, np.empty(0), np.array([0.]), StateView(layout, fields), "original-physical-input:"+case_name)
    bulk = np.full(root.state.field("n_m3").shape, 1e-17)
    trace = np.full(root.state.field("trace_potential_V").shape, 1e-17)
    right, increment = InputLiftCoordinates(layout, VT).advance(
        root, {"electron": bulk, "potential": -bulk, "trace_potential": trace}, 1,
        DoubleArray.from_dd(DD(VT)*DD(bulk)), DoubleArray.from_dd(-DD(VT)*DD(trace)), [0])
    restored = decode_point(json.loads(json.dumps(encode_point(right))), layout)
    assert restored.identity == right.identity
    for name in units:
        assert restored.state.field(name).shape == root.state.field(name).shape
        assert restored.state.authority.anchor[name].identity_bytes() == root.state.field(name).identity_bytes()
    for name in ("n_m3", "phi_V", "positive_m3", "occupancy", "trace_potential_V"):
        assert np.all(increment.field(name).as_dd() == 0), name
    with localcontext() as context:
        context.prec = 80
        multiplier = Decimal.from_float(1e-17).exp()-1
        for index in range(root.state.field("p_m3").shape[0]):
            expected = words(root.state.field("p_m3"), index)*multiplier
            actual = words(increment.field("p_m3"), index)
            assert abs(actual-expected) <= Decimal("1e-7")+Decimal("1e-12")*abs(expected)
            assert (actual > 0) == (expected > 0)
    # The independent review's ordinary three-scale rejection, followed by an
    # admitted weaker drive, must now compose exactly in the bounded primitive
    # representation while the physical delta remains ordinary DD.
    point, total = root, Decimal(0)
    for step, magnitude in enumerate((0.1, 1e-8, 1e-20, 1e-40), 1):
        update = np.zeros(root.state.field("n_m3").shape)
        update[-1] = magnitude
        following, change = InputLiftCoordinates(layout, VT).advance(point, {"electron": update}, step)
        with localcontext() as context:
            context.prec = 150
            n0 = words(root.state.field("n_m3"), len(update)-1)
            expected = n0*total.exp()*(Decimal.from_float(magnitude).exp()-1)
            actual = words(change.field("n_m3"), len(update)-1)
            assert actual != 0 and abs(actual-expected) <= abs(expected)*Decimal("1e-12")
            total += Decimal.from_float(magnitude)
        restored_following = decode_point(json.loads(json.dumps(encode_point(following))), layout)
        assert restored_following.identity == following.identity
        point = restored_following
    assert np.any(point.state.authority.primitives["electron"].words[3] != 0)
    represented = sum((Fraction.from_float(float(word[-1])) for word in point.state.authority.primitives["electron"].words), Fraction())
    requested = sum((Fraction.from_float(value) for value in (0.1, 1e-8, 1e-20, 1e-40)), Fraction())
    assert represented == requested
    # These are kinematic input/codec controls, not native183/309 or13-channel
    # longwindow science qualification; the original failure records stay put.


def test_logit_weak_step_and_inactive_and_domain_failures_are_explicit():
    layout = Layout((Support("cell", "global", (3,)),),
                    (VariableSpec("f", "trap", "cell", (3,), ONE, lower=0, upper=1),), ())
    coords = RelativeCoordinates(layout, {"f": "logit"})
    root = coords.initial(StateView(layout, [("f", DoubleArray([0, 0.5, 1]))]))
    right, delta = coords.advance(root, [-80, 1e-17, 80], 1)
    assert words(delta.field("f"), 0) == words(delta.field("f"), 2) == 0
    with localcontext() as context:
        context.prec = 100
        z = Decimal.from_float(1e-17)
        expected = z.exp()/(1+z.exp())-Decimal("0.5")
        assert abs(words(delta.field("f"), 1)-expected) < Decimal("1e-45")
    inactive = RelativeCoordinates(layout, {"f": "inactive"})
    zero = inactive.initial(StateView(layout, [("f", DoubleArray([0, 0, 0]))]))
    with pytest.raises(ContractError, match="inactive_coordinate_change"):
        inactive.advance(zero, [0, 1e-17, 0], 1)
    logarithmic = RelativeCoordinates(layout, {"f": "log"})
    with pytest.raises(ContractError, match="positive_inventory"):
        logarithmic.initial(StateView(layout, [("f", DoubleArray([0, 0.5, 1]))]))
    with pytest.raises(ArithmeticError, match="supported precision range"):
        coords.advance(root, [0, 700, 0], 1)


@pytest.mark.parametrize("density,role,sign", [("n_m3", "electron", -1), ("p_m3", "hole", 1)])
@pytest.mark.parametrize("weak", [0.0, 1e-40, -1e-40])
@pytest.mark.parametrize("gauge", [0.0, 1.0])
@pytest.mark.parametrize("potential", [[0.1, 0.3], [1e-6, 0.017000000000000003],
                                       [0.123456789012345, 3.141592653589793]])
def test_localized_coupled_activity_cancels_the_shared_primitive_before_rounding(density, role, sign, weak, gauge, potential):
    layout = Layout((Support("nodes", "global", (2,)),),
                    (VariableSpec("n_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("p_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("phi_V", "field", "nodes", (2,), VOLT)), ())
    root = Point(0, np.empty(0), np.empty(0), StateView(layout, [
        ("n_m3", DoubleArray([1e22, 1e22])), ("p_m3", DoubleArray([1e22, 1e22])),
        ("phi_V", DoubleArray([gauge, gauge]))]), "localized-r1-physical-input-v1")
    right, _ = InputLiftCoordinates(layout, VT).advance(
        root, {"potential": potential, role: [0, weak]}, 1)
    for point in (right, decode_point(encode_point(right), layout)):
        actual = point.state.electrochemical_difference(density, "phi_V", [[0, 1]], VT,
                                                        potential_sign=sign, arithmetic=DoubleArithmetic())
        with localcontext() as context:
            context.prec = 100
            expected = Decimal.from_float(weak)
            assert abs(words(actual)-expected) <= abs(expected)*Decimal("1e-12"), (words(actual), expected)
            assert (words(actual) == 0) == (expected == 0)
            other_vt = point.state.electrochemical_difference(density, "phi_V", [[0, 1]], 2*VT,
                                                               potential_sign=sign, arithmetic=DoubleArithmetic())
            expected_other = expected-Decimal(sign)*(Decimal.from_float(potential[1])-Decimal.from_float(potential[0]))/2
            assert abs(words(other_vt)-expected_other) < Decimal("1e-29")


@pytest.mark.parametrize("density,role,coupled_sign", [("n_m3", "electron", 1), ("p_m3", "hole", -1)])
@pytest.mark.parametrize("weak", [0.0, 1e-40, -1e-40])
@pytest.mark.parametrize("potential", [[0.1, 0.3], [1e-6, 0.017000000000000003],
                                       [0.123456789012345, 3.141592653589793]])
def test_localized_full_voltage_lift_retains_weak_activity_through_all_product_words(density, role, coupled_sign, weak, potential):
    layout = Layout((Support("nodes", "global", (2,)),),
                    (VariableSpec("n_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("p_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("phi_V", "field", "nodes", (2,), VOLT)), ())
    root = Point(0, np.empty(0), np.empty(0), StateView(layout, [
        ("n_m3", DoubleArray([1e22, 1e22])), ("p_m3", DoubleArray([1e22, 1e22])),
        ("phi_V", DoubleArray([1, 1]))]), "localized-lift-physical-input-v1")
    primary = DoubleArray(-coupled_sign*np.asarray(potential), [0, weak])
    lift = DoubleArray.from_dd(-DD(VT)*DD(potential))
    right, _ = InputLiftCoordinates(layout, VT).advance(root, {"potential": potential, role: primary}, 1, lift)
    for point in (right, decode_point(encode_point(right), layout)):
        result = point.state.electrochemical_difference(density, "phi_V", [[0, 1]], VT,
                                                        potential_sign=-coupled_sign, arithmetic=DoubleArithmetic())
        with localcontext() as context:
            context.prec = 100
            expected = Decimal.from_float(weak)
            assert abs(words(result)-expected) <= abs(expected)*Decimal("1e-12")
            assert (words(result) == 0) == (expected == 0)


def test_accepted_step_and_physical_storage_preserve_the_same_DD_delta_words():
    coords, left = state_fixture()
    right, delta = coords.advance(left, [0, 1e-17, 0, 1e-22], 1, [0])
    a = DoubleArithmetic()
    storage = PhysicalStorage(lambda p: p.state.field("n_m3"), lambda l, r, d: d.field("n_m3"),
                               (PARTICLE / VOLUME, PARTICLE / VOLUME), "authority-example-v1", a)
    dq = storage.delta(left, right, delta)
    step = AcceptedStep(left, right, delta, dq, delta.field("phi_V"), np.zeros(4), np.zeros(1),
                        "nonnative-authority-control", "continuous", (("independent_control", 0),), True, a)
    assert step.storage_delta.identity_bytes() == dq.identity_bytes()
    current = TerminalPort("left", -1, 1).current(dq, DoubleArray.from_dd(-dq.as_dd()+DD([0, 1e-20])), arithmetic=a)
    with localcontext() as context:
        context.prec = 100
        assert abs(words(current, 1)-Decimal.from_float(1e-20)) < Decimal("1e-27")
    with pytest.raises(TypeError, match="implicit precision rounding"):
        np.asarray(step.storage_delta)


@pytest.mark.parametrize("changed", ["time", "inputs", "reference", "y"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_first_lift_binds_the_complete_predecessor_Point(changed, roundtrip):
    layout = Layout((Support("node", "global", (1,)),),
                    (VariableSpec("n_m3", "carrier", "node", (1,), ONE),
                     VariableSpec("p_m3", "carrier", "node", (1,), ONE),
                     VariableSpec("phi_V", "potential", "node", (1,), VOLT)), ())
    left = Point(0, np.array([0.]), np.array([0.]), StateView(layout, [
        ("n_m3", DoubleArray([1])), ("p_m3", DoubleArray([1])),
        ("phi_V", DoubleArray([0]))]), "first-r1-root-v1")
    right, original = InputLiftCoordinates(layout, VT).advance(left, {"electron": [0.01]}, 1, inputs=[0])
    if roundtrip:
        right = decode_point(encode_point(right), layout)
    assert right.state.authority.previous_point_identity == left.identity
    assert StateIncrement.from_points(left, right).field("n_m3").identity_bytes() == original.field("n_m3").identity_bytes()
    kwargs = {"time": {"time": 0.25}, "inputs": {"inputs": np.array([1.])},
              "reference": {"coordinate_reference": "different-root-source"}, "y": {"y": np.array([0.25])}}[changed]
    foreign = replace(left, **kwargs)
    assert foreign.state.authority.identity == left.state.authority.identity
    assert foreign.identity != left.identity
    with pytest.raises(ContractError, match="increment_predecessor_point_mismatch"):
        StateIncrement.from_points(foreign, right)
    forged = StateIncrement(foreign.identity, right.identity, original.fields)
    with pytest.raises(ContractError, match="increment_predecessor_point_mismatch"):
        forged.validate(foreign, right)
    with pytest.raises(ContractError, match="increment_predecessor_point_mismatch"):
        AcceptedStep(foreign, right, forged, [0], [0], np.zeros_like(right.y), np.zeros_like(right.inputs),
                     "binding-only-control", "continuous", (("binding", 0),), True)


@pytest.mark.parametrize("density,role,potential_sign", [("n_m3", "electron", 1), ("p_m3", "hole", -1)])
@pytest.mark.parametrize("weak", [1e-40, -1e-40, 1e-17, -1e-17])
def test_canonical_inventory_delta_uses_primitives_before_coupled_drive_rounding(density, role, potential_sign, weak):
    layout = Layout((Support("nodes", "global", (2,)),),
                    (VariableSpec("n_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("p_m3", "carrier", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("phi_V", "potential", "nodes", (2,), VOLT)), ())
    root = Point(0, np.empty(0), np.empty(0), StateView(layout, [
        ("n_m3", DoubleArray([1e22, 1e22])), ("p_m3", DoubleArray([1e22, 1e22])),
        ("phi_V", DoubleArray([0, 0]))]), "canonical-inventory-root-v1")
    coordinates = InputLiftCoordinates(layout, VT)
    first, _ = coordinates.advance(root, {"potential": [0.1, 0.3]}, 1)
    left, _ = coordinates.advance(first, {"potential": [1e-8, -1e-8]}, 2)
    right, increment = coordinates.advance(left, {role: [0, weak]}, 3)
    with localcontext() as context:
        context.prec = 140
        z0 = Decimal(potential_sign)*(Decimal.from_float(0.3)-Decimal.from_float(1e-8))
        expected = Decimal(10**22)*z0.exp()*(Decimal.from_float(weak).exp()-1)
        for point in (right, decode_point(encode_point(right), layout)):
            actual = StateIncrement.from_points(left, point).field(density)
            assert abs(words(actual, 1)-expected) <= abs(expected)*Decimal("1e-12")
            assert (words(actual, 1) > 0) == (expected > 0)
            assert words(actual, 1) != 0
            activity = point.state.electrochemical_difference(density, "phi_V", [[0, 1]], VT,
                                                               potential_sign=-potential_sign, arithmetic=DoubleArithmetic())
            assert abs(words(activity)-Decimal.from_float(weak)) <= abs(Decimal.from_float(weak))*Decimal("1e-12")
    assert increment.field(density).identity_bytes() == StateIncrement.from_points(left, right).field(density).identity_bytes()

"""Endpoint-bound temporal constitutive inputs with independent exact oracles.

No SG law or integrator is implemented here. Fraction linear forms check the
named binary inputs; Decimal logarithms independently check resolved words.
The 1e-28 contraction check is a local arithmetic diagnostic, not a gate change.
"""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    ONE, PARTICLE, VOLT, VOLUME, ContractError, FloatArray, Layout, Point,
    StateIncrement, StateView, Support, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic, DoubleArray, InputLiftCoordinates, RelativeCoordinates,
    decode_point, encode_point,
)
from scripts.benchmarks.storage_prototype import ExponentialStorage


VT = 0.025851999786435535
PAIRS = np.array([[0, 1], [1, 2]])
WEAK = [0.0, 1e-40, -1e-40, 1e-17, -1e-17]


def rational(value):
    return Fraction.from_float(float(value))


def exact_words(value):
    return [rational(h)+rational(l) for h, l in zip(value.high.ravel(), value.low.ravel())]


def check_linear_form(value, expected, *, exact=False):
    actual = exact_words(value)
    assert len(actual) == len(expected)
    for got, wanted in zip(actual, expected):
        assert (got == 0) == (wanted == 0)
        assert (got > 0) == (wanted > 0)
        if exact:
            assert got == wanted
        else:
            assert abs(got-wanted) <= abs(wanted)*Fraction(1, 10**28)


def root_point(*, zero_node=False, precision=True, trace=False, potential_unit=VOLT):
    supports = [Support("nodes", "global", (3,))]
    variables = [VariableSpec("n_m3", "carrier", "nodes", (3,), PARTICLE/VOLUME),
                 VariableSpec("p_m3", "carrier", "nodes", (3,), PARTICLE/VOLUME),
                 VariableSpec("phi_V", "electrostatics", "nodes", (3,), potential_unit)]
    array = DoubleArray if precision else FloatArray
    fields = [("n_m3", array([0 if zero_node else 1e22, 1.3e22, 2e22])),
              ("p_m3", array([0.7e22, 2e22, 3e22])), ("phi_V", array([0, 0.017, -0.02]))]
    if trace:
        supports.append(Support("trace", "global", (1, 2)))
        variables.append(VariableSpec("trace_state_m3", "trace-population", "trace", (1, 2), PARTICLE/VOLUME))
        fields.append(("trace_state_m3", array([[1e20, 2e20]])))
    layout = Layout(tuple(supports), tuple(variables), ())
    return Point(0, np.empty(0), np.array([0.]), StateView(layout, fields), "temporal-resolved-root-v1")


def roundtrip(point):
    return decode_point(json.loads(json.dumps(encode_point(point))), point.state.layout)


@pytest.mark.parametrize("stage", ["first_root", "mapped"])
@pytest.mark.parametrize("role,sign", [("electron", -1), ("hole", 1)])
@pytest.mark.parametrize("common", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("weak", WEAK)
def test_temporal_log_and_spatial_drive_use_named_inputs(stage, role, sign, common, weak):
    left = root_point()
    coordinates = InputLiftCoordinates(left.state.layout, VT)
    if stage == "mapped":
        left, _ = coordinates.advance(left, {"potential": [0.1, 0.3, -0.2],
                                              "electron": [0.02, -0.01, 0.03],
                                              "hole": [-0.01, 0.04, -0.07]}, 1)
    before = encode_point(left)
    right, change = coordinates.advance(left, {role: [0, weak, -weak], "potential": [common]*3}, left.time+1)
    density = "n_m3" if role == "electron" else "p_m3"
    temporal = [rational(common)*(-sign)+rational(weak), rational(common)*(-sign)-rational(weak)]
    driving = [rational(weak), -2*rational(weak)]
    for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
        increment = StateIncrement.from_points(a, b)
        check_linear_form(increment.log_ratio(a, b, density, indices=[1, 2]), temporal, exact=True)
        value = increment.electrochemical_delta(a, b, density, "phi_V", PAIRS, VT,
                                               potential_sign=sign, arithmetic=DoubleArithmetic())
        check_linear_form(value, driving, exact=True)
        check_linear_form(increment.face_delta(a, b, "phi_V", PAIRS, arithmetic=DoubleArithmetic()),
                          [Fraction(), Fraction()], exact=True)
    assert encode_point(left) == before
    change.validate(left, right)


@pytest.mark.parametrize("stage", ["first_root", "mapped"])
@pytest.mark.parametrize("weak", WEAK)
@pytest.mark.parametrize("observed_VT", [VT, 0.03])
def test_temporal_face_and_full_lift_contract_before_projection(stage, weak, observed_VT):
    left = root_point()
    coordinates = InputLiftCoordinates(left.state.layout, VT)
    if stage == "mapped":
        left, _ = coordinates.advance(left, {"potential": [0.17, -0.02, 0.31]}, 1)
    z = DoubleArray([0.3]*3, [0, weak, -weak])
    lift = DoubleArray([0.4]*3, [weak, -weak, 0])
    right, increment = coordinates.advance(left, {"potential": z, "electron": [weak, 0, -weak]},
                                            left.time+1, voltage_lift_V=lift)
    f, vt, measured = rational(weak), rational(VT), rational(observed_VT)
    expected_phi = [vt*f-2*f, -2*vt*f+f]
    # Independent joint physical input: log(n_j/n_i) changes by delta(e+z),
    # and phi_j-phi_i changes by VT*delta(z)+delta(lift).
    expected_drive = [-f+f-expected_phi[0]/measured,
                      -f-2*f-expected_phi[1]/measured]
    for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
        change = StateIncrement.from_points(a, b)
        check_linear_form(change.face_delta(a, b, "phi_V", PAIRS, arithmetic=DoubleArithmetic()), expected_phi)
        check_linear_form(change.electrochemical_delta(a, b, "n_m3", "phi_V", PAIRS, observed_VT,
                                                       potential_sign=-1, arithmetic=DoubleArithmetic()), expected_drive)
    increment.validate(left, right)


@pytest.mark.parametrize("weak", WEAK)
def test_relative_map_and_resolved_first_root_share_temporal_contract(weak):
    resolved = root_point()
    layout = resolved.state.layout
    coordinates = RelativeCoordinates(layout, {"n_m3": "log", "p_m3": "log", "phi_V": "linear"})
    left = coordinates.initial(resolved.state, inputs=resolved.inputs)
    right, change = coordinates.advance(left, [0, weak, -weak, 0, 0, 0, 0, -VT*weak, VT*weak], 1, [0])
    expected = [rational(weak)+rational(VT*weak)/rational(VT),
                -2*rational(weak)-2*rational(VT*weak)/rational(VT)]
    check_linear_form(change.electrochemical_delta(left, right, "n_m3", "phi_V", PAIRS, VT,
                                                 potential_sign=-1, arithmetic=DoubleArithmetic()), expected)
    check_linear_form(change.log_ratio(left, right, "n_m3"), [Fraction(), rational(weak), -rational(weak)], exact=True)


@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("size", [1e5, 1e-18])
def test_affine_density_temporal_activity_at_nonzero_affinity(direction, size):
    root = root_point()
    layout = root.state.layout
    coordinates = RelativeCoordinates(layout, {v.id: "linear" for v in layout.variables})
    left = coordinates.initial(root.state, inputs=[0])
    dn = direction*np.array([size, 2*size, -3*size])
    dp = direction*np.array([size*1e-24, -2*size*1e-24, 0])
    dy = np.zeros(layout.size)
    dy[layout.offsets["n_m3"]], dy[layout.offsets["phi_V"]] = dn, dp
    right, change = coordinates.advance(left, dy, 1, [0])
    with localcontext() as ctx:
        ctx.prec = 180
        dec = lambda v: Decimal.from_float(float(v))
        density = [dec(v) for v in [1e22, 1.3e22, 2e22]]
        temporal = [((n+dec(delta))/n).ln() for n, delta in zip(density, dn)]
        expected = [temporal[j]-temporal[i]-(dec(dp[j])-dec(dp[i]))/dec(VT) for i, j in PAIRS]
        for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
            initial = a.state.electrochemical_difference("n_m3", "phi_V", PAIRS, VT,
                                                       potential_sign=-1, arithmetic=DoubleArithmetic())
            final = b.state.electrochemical_difference("n_m3", "phi_V", PAIRS, VT,
                                                     potential_sign=-1, arithmetic=DoubleArithmetic())
            assert all(x != 0 and (x > 0) == (y > 0) for x, y in zip(exact_words(initial), exact_words(final)))
            value = StateIncrement.from_points(a, b).electrochemical_delta(
                a, b, "n_m3", "phi_V", PAIRS, VT, potential_sign=-1, arithmetic=DoubleArithmetic())
            for actual, wanted in zip(exact_words(value), expected):
                represented = Decimal(actual.numerator)/Decimal(actual.denominator)
                assert (represented == 0) == (wanted == 0)
                assert (represented > 0) == (wanted > 0)
                assert abs(represented-wanted) <= abs(wanted)*Decimal("1e-28")
    change.validate(left, right)


@pytest.mark.parametrize("weak", [1e-18, -1e-18])
@pytest.mark.parametrize("common", [1e21, -1e21])
@pytest.mark.parametrize("potential", [0.0, 0.017])
def test_affine_common_density_step_keeps_existing_weak_spatial_gradient(weak, common, potential):
    layout = Layout((Support("nodes", "global", (2,)),),
                    (VariableSpec("n", "population", "nodes", (2,), PARTICLE/VOLUME),
                     VariableSpec("phi", "potential", "nodes", (2,), VOLT)), ())
    root = StateView(layout, [("n", DoubleArray([1e22, 1e22], [0, weak])),
                              ("phi", DoubleArray([0, potential]))])
    coordinates = RelativeCoordinates(layout, {"n": "linear", "phi": "linear"})
    left = coordinates.initial(root)
    right, change = coordinates.advance(left, [common, common, 0, 0], 1)
    references = []
    for precision in (160, 200):
        with localcontext() as ctx:
            ctx.prec = precision
            base, delta, small = map(Decimal.from_float, [1e22, common, weak])
            references.append(((base+small+delta)/(base+delta)).ln()-((base+small)/base).ln())
    with localcontext() as ctx:
        ctx.prec = 200
        budget = abs(references[1])*Decimal("1e-28")
        assert abs(references[0]-references[1]) <= budget/3
        for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
            value = StateIncrement.from_points(a, b).electrochemical_delta(
                a, b, "n", "phi", [[0, 1]], VT, potential_sign=-1, arithmetic=DoubleArithmetic())
            fraction = exact_words(value)[0]
            actual = Decimal(fraction.numerator)/Decimal(fraction.denominator)
            assert actual != 0
            assert (actual > 0) == (references[1] > 0)
            assert abs(actual-references[1]) <= budget
    change.validate(left, right)


@pytest.mark.parametrize("density_steps", [[2e22, -1e22, 3e22], [-0.5e22, 1e22, -0.75e22]])
def test_large_affine_spatial_ratio_changes_keep_finite_log_branch(density_steps):
    root = root_point()
    layout = root.state.layout
    coordinates = RelativeCoordinates(layout, {v.id: "linear" for v in layout.variables})
    left = coordinates.initial(root.state)
    dy = np.zeros(layout.size); dy[layout.offsets["n_m3"]] = density_steps
    right, change = coordinates.advance(left, dy, 1)
    value = change.electrochemical_delta(left, right, "n_m3", "phi_V", PAIRS, VT,
                                        potential_sign=-1, arithmetic=DoubleArithmetic())
    with localcontext() as ctx:
        ctx.prec = 180
        initial = list(map(Decimal.from_float, [1e22, 1.3e22, 2e22]))
        final = [a+Decimal.from_float(b) for a, b in zip(initial, density_steps)]
        reference = [(final[j]/final[i]).ln()-(initial[j]/initial[i]).ln() for i, j in PAIRS]
        for fraction, expected in zip(exact_words(value), reference):
            actual = Decimal(fraction.numerator)/Decimal(fraction.denominator)
            assert abs(actual-expected) <= abs(expected)*Decimal("1e-28")


@pytest.mark.parametrize("mapped", [False, True])
def test_temporal_same_point_is_exact_zero(mapped):
    point = root_point()
    if mapped:
        point, _ = InputLiftCoordinates(point.state.layout, VT).advance(point, {"potential": [0.1, 0.3, -0.2]}, 1)
    change = StateIncrement.from_points(point, point)
    check_linear_form(change.log_ratio(point, point, "n_m3"), [Fraction()]*3, exact=True)
    check_linear_form(change.face_delta(point, point, "phi_V", PAIRS, arithmetic=DoubleArithmetic()), [Fraction()]*2, exact=True)
    check_linear_form(change.electrochemical_delta(point, point, "n_m3", "phi_V", PAIRS, VT,
                                                 potential_sign=-1, arithmetic=DoubleArithmetic()), [Fraction()]*2, exact=True)


def test_selected_positive_nodes_do_not_evaluate_inactive_zero_populations():
    left = root_point(zero_node=True)
    right, change = InputLiftCoordinates(left.state.layout, VT).advance(left, {"electron": [0, 1e-40, 0]}, 1)
    check_linear_form(change.log_ratio(left, right, "n_m3", indices=[1, 2]), [rational(1e-40), Fraction()], exact=True)
    check_linear_form(change.electrochemical_delta(left, right, "n_m3", "phi_V", [[1, 2]], VT,
                                                 potential_sign=-1, arithmetic=DoubleArithmetic()), [-rational(1e-40)], exact=True)
    with pytest.raises(ContractError, match="positive_fields"):
        change.log_ratio(left, right, "n_m3")
    with pytest.raises(ContractError, match="positive_fields"):
        change.electrochemical_delta(left, right, "n_m3", "phi_V", [[0, 1]], VT,
                                    potential_sign=-1, arithmetic=DoubleArithmetic())


def test_trace_log_ratio_keeps_shape_and_first_root_meaning():
    left = root_point(trace=True)
    right, change = InputLiftCoordinates(left.state.layout, VT).advance(left, {"trace_density": [[1e-40, -1e-40]]}, 1)
    for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
        increment = StateIncrement.from_points(a, b)
        value = increment.log_ratio(a, b, "trace_state_m3", indices=[0])
        assert value.shape == (1, 2)
        check_linear_form(value, [rational(1e-40), -rational(1e-40)], exact=True)
    change.validate(left, right)


def test_temporal_log_keeps_the_positive_depletion_endpoint():
    root = root_point()
    coordinates = InputLiftCoordinates(root.state.layout, VT)
    left, _ = coordinates.advance(root, {"electron": [0, 1e-17, 0]}, 1)
    right, change = coordinates.advance(left, {"electron": [0, -80, 0]}, 2)
    assert right.state.field("n_m3").as_dd()[1] > 0
    for a, b in ((left, right), (roundtrip(left), roundtrip(right))):
        check_linear_form(StateIncrement.from_points(a, b).log_ratio(a, b, "n_m3", indices=[1]),
                          [Fraction(-80)], exact=True)
    change.validate(left, right)


@pytest.mark.parametrize("precision", [False, True])
def test_explicit_resolved_fields_have_finite_temporal_meaning(precision):
    left = root_point(precision=precision)
    array = DoubleArray if precision else FloatArray
    values = dict(left.state.fields)
    values["n_m3"] = array([2e22, 1e22, 4e22])
    values["phi_V"] = array([0.1, 0.3, -0.4])
    right = Point(1, np.empty(0), np.array([0.]), StateView(left.state.layout, values.items()), "resolved-after-v1")
    change = StateIncrement.from_points(left, right)
    a = DoubleArithmetic() if precision else None
    log = change.log_ratio(left, right, "n_m3")
    drive = change.electrochemical_delta(left, right, "n_m3", "phi_V", PAIRS, VT, potential_sign=-1, arithmetic=a)
    face = change.face_delta(left, right, "phi_V", PAIRS, arithmetic=a)
    with localcontext() as ctx:
        ctx.prec = 180
        dec = lambda v: Decimal.from_float(float(v))
        before = [dec(v) for v in [1e22, 1.3e22, 2e22]]
        after = [dec(v) for v in [2e22, 1e22, 4e22]]
        truth = [(b/a).ln() for a, b in zip(before, after)]
        potential = [dec(0.1), dec(0.3)-dec(0.017), dec(-0.4)-dec(-0.02)]
        faces = [potential[j]-potential[i] for i, j in PAIRS]
        drives = [truth[j]-truth[i]-drop/dec(VT) for (i, j), drop in zip(PAIRS, faces)]
        for value, wanted in ((log, truth), (face, faces), (drive, drives)):
            got = [Decimal(v.numerator)/Decimal(v.denominator) for v in exact_words(value)] if precision else [dec(v) for v in (value.values if isinstance(value, FloatArray) else value)]
            for actual, expected in zip(got, wanted):
                assert abs(actual-expected) <= abs(expected)*Decimal("1e-28" if precision else "1e-12")


def test_float_storage_map_uses_its_canonical_increment_for_temporal_log():
    coordinates = ExponentialStorage(by=1.0, ba=0.0, bt=0.0)
    left = coordinates.point([0.])
    right, change = coordinates.advance(left, [1e-17], 1)
    np.testing.assert_array_equal(change.log_ratio(left, right, "storage").values, [1e-17])


def test_temporal_operations_reject_wrong_endpoints_forged_increments_and_roots():
    left = root_point()
    right, change = InputLiftCoordinates(left.state.layout, VT).advance(left, {"electron": [0, 1e-40, 0]}, 1)
    operations = [lambda inc, a, b: inc.log_ratio(a, b, "n_m3"),
                  lambda inc, a, b: inc.face_delta(a, b, "phi_V", PAIRS, arithmetic=DoubleArithmetic()),
                  lambda inc, a, b: inc.electrochemical_delta(a, b, "n_m3", "phi_V", PAIRS, VT,
                                                            potential_sign=-1, arithmetic=DoubleArithmetic())]
    fields = dict(change.fields); fields["n_m3"] = DoubleArray([0, 0, 0])
    forged = StateIncrement(left.identity, right.identity, fields)
    other = replace(left, time=2)
    other_fields = dict(left.state.fields); other_fields["n_m3"] = DoubleArray([2e22]*3)
    alien = replace(left, state=StateView(left.state.layout, other_fields.items()))
    for operation in operations:
        for before in (other, alien, right):
            with pytest.raises(ContractError, match="increment_endpoint_mismatch"):
                operation(change, before, right)
        with pytest.raises(ContractError, match="increment_authority_mismatch"):
            operation(forged, left, right)


def test_temporal_units_shapes_indices_and_implicit_rounding_are_guarded():
    left = root_point()
    right, change = InputLiftCoordinates(left.state.layout, VT).advance(left, {"potential": [0, 1e-40, 0]}, 1)
    for indices in ([-1], [3], [0.5], [True]):
        with pytest.raises(ContractError):
            change.log_ratio(left, right, "n_m3", indices=indices)
    for pairs in ([0, 1], [[0, 3]], [[-1, 0]], [[0.1, 1]], [[True, False]]):
        with pytest.raises(ContractError):
            change.face_delta(left, right, "phi_V", pairs, arithmetic=DoubleArithmetic())
    for name in ("missing",):
        with pytest.raises(ContractError, match="unknown_variable"):
            change.log_ratio(left, right, name)
    with pytest.raises(ContractError, match="affine_map"):
        change.face_delta(left, right, "n_m3", PAIRS, arithmetic=DoubleArithmetic())
    for vt, sign in ((0, -1), (np.nan, -1), (VT, 0)):
        with pytest.raises(ContractError, match="invalid_activity_parameters"):
            change.electrochemical_delta(left, right, "n_m3", "phi_V", PAIRS, vt, potential_sign=sign, arithmetic=DoubleArithmetic())
    with pytest.raises(TypeError, match="implicit precision rounding"):
        change.face_delta(left, right, "phi_V", PAIRS)
    wrong_unit = root_point(potential_unit=ONE)
    inc = StateIncrement.from_points(wrong_unit, wrong_unit)
    with pytest.raises(ContractError, match="activity_potential_unit_mismatch"):
        inc.electrochemical_delta(wrong_unit, wrong_unit, "n_m3", "phi_V", PAIRS, VT, potential_sign=-1, arithmetic=DoubleArithmetic())

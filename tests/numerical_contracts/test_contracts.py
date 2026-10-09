"""Independent contract checks; gates are frozen before candidate execution."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, ONE, PARTICLE, SECOND, VOLUME,
    AcceptedStep, AlgebraicTerm, BalanceTerms, CellSource, ContractError,
    EquationSpec, FaceFlux, FloatArray, Geometry, ImplicitSystem, Layout,
    LinearCoordinates, LinearStorage, Point, StateIncrement, StateView,
    Support, SurfaceCharge, TerminalPort, TermSink, Unit, VariableSpec,
)


GATES = {item["id"]: item for item in json.loads(
    Path(__file__).with_name("AnalyticGatesV1.json").read_text()
)["gates"]}


def cell_layout(count=2, face_count=1, *, reverse=False, dimensionless=True):
    density_unit = ONE if dimensionless else PARTICLE / VOLUME
    rate_unit = ONE / SECOND if dimensionless else PARTICLE / SECOND
    supports = (Support("cells", "cell", (count,)), Support("faces", "face", (face_count,)))
    variables = (VariableSpec("ions.n", "ions", "cells", (count,), density_unit, lower=0),)
    equations = (EquationSpec("ions.balance", "assembly", "cells", (count,), rate_unit,
                              derivative_support=("ions.n",)),)
    return Layout(supports[::-1] if reverse else supports,
                  variables[::-1] if reverse else variables,
                  equations[::-1] if reverse else equations)


def linear_terms(rate, matrix, inputs=0):
    n = len(rate)
    return BalanceTerms(np.asarray(rate), np.empty(0), np.asarray(matrix),
                        np.zeros((n, inputs)), np.zeros(n), np.empty((0, n)),
                        np.empty((0, inputs)), np.empty(0))


def test_frame_input_rate_retains_last_word_and_rejects_mixed_points():
    from scripts.benchmarks.contract_prototype import LinearTerm, PhysicalLinearForm, RateView
    from scripts.benchmarks.precision_prototype import DoubleArray, DoubleArithmetic, FrameInputExpansion, PrimitiveExpansion, RelativeCoordinates

    layout = cell_layout()
    coordinates = RelativeCoordinates(layout, {"ions.n": "linear"})
    reference = coordinates.initial(StateView(layout, [("ions.n", DoubleArray([3., 3.]))]))
    point, _ = coordinates.trial(reference, FrameInputExpansion.from_value([0., 0.]), 1.)
    terms = [np.full(2, 2.**(-54*i)) for i in range(12)]
    terms[-1] = np.array([2.**-594, 0.])
    inputs = dict(source_identity="manufactured", mapping_identity="mapped12", origin="mapped-coordinate-rate",
                  raw_coordinates=[0., 0.], raw_rate=[0., 0.])
    values = FrameInputExpansion(terms)
    rate = RateView(point, values, [], **inputs)
    form = PhysicalLinearForm(layout, ("difference",), (ONE/SECOND,),
        (LinearTerm(0, "ions.n", 0), LinearTerm(0, "ions.n", 1, sign=-1)), "manufactured", "rate", 2)
    result = rate.linear_form(form, arithmetic=DoubleArithmetic(), point=point)
    assert len(rate.words) == 12 and rate.field("ions.n").identity_bytes() == values.identity_bytes()
    assert result.value.high[0] == 2.**-594 and result.value.low[0] == 0
    terms[-1] = np.zeros(2)
    other = RateView(point, FrameInputExpansion(terms), [], **inputs)
    assert other.identity != rate.identity
    with pytest.raises(ContractError, match="linear_rate_word_shape_or_count"):
        RateView(reference, values, [], **inputs)
    with pytest.raises(ContractError, match="linear_rate_frame_input_profile"):
        RateView(point, PrimitiveExpansion.from_value([0., 0.]), [], **inputs)
    # A named physical tangent is still an ordinary physical array.
    tangent = RateView(point, DoubleArray([0., 0.]), [], **dict(inputs, origin="physical-rate"))
    assert len(tangent.words) == 2


def diffusion_system(case):
    volumes = np.array(case["volumes"], dtype=float)
    pairs = np.array(case["faces"], dtype=int)
    conductance = np.array(case["conductance"], dtype=float)
    layout = cell_layout(len(volumes), len(pairs))
    geometry = Geometry(volumes, pairs, np.ones(len(pairs)), ONE, ONE)
    matrix = np.zeros((len(volumes), len(volumes)))
    for (left, right), coefficient in zip(pairs, conductance):
        matrix[left, left] -= coefficient
        matrix[left, right] += coefficient
        matrix[right, left] += coefficient
        matrix[right, right] -= coefficient

    def terms(point):
        # Constitutive flux is oriented left -> right; sink owns its divergence.
        difference = point.state.face_difference("ions.n", pairs).values
        sink = TermSink(layout, "ions.balance", geometry)
        sink.add(FaceFlux("ions.diffusion", "faces", -conductance * difference, ONE / SECOND))
        return linear_terms(sink.value(), matrix)

    return LinearCoordinates(layout), ImplicitSystem(LinearStorage(np.diag(volumes)), terms), matrix


def test_registration_order_does_not_change_layout_or_solution():
    supports = (Support("nodes", "cell", (2,)), Support("contact", "global", (1,)))
    variables = (VariableSpec("p", "holes", "nodes", (2,), PARTICLE / VOLUME),
                 VariableSpec("n", "electrons", "nodes", (2,), PARTICLE / VOLUME),
                 VariableSpec("port", "contact", "contact", (1,), COULOMB, role="constraint"))
    equations = (EquationSpec("poisson", "electrostatics", "nodes", (2,), COULOMB,
                              role="constraint", derivative_support=("n", "p", "port")),)
    forward = Layout(supports, variables, equations)
    reverse = Layout(supports[::-1], variables[::-1], equations[::-1])
    assert forward.identity == reverse.identity
    assert forward.offsets == reverse.offsets
    assert tuple(forward.offsets) == ("n", "p", "port")
    assert LinearCoordinates(forward).point([1, 2, 3, 4, 5]).identity == LinearCoordinates(reverse).point([1, 2, 3, 4, 5]).identity


@pytest.mark.parametrize("change,reason", [
    (lambda x: replace(x, variables=x.variables + x.variables), "duplicate_owner"),
    (lambda x: replace(x, supports=x.supports + x.supports), "duplicate_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], support="missing"),)), "unknown_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], shape=(1,)),)), "support_shape_mismatch"),
    (lambda x: replace(x, equations=(replace(x.equations[0], derivative_support=("missing",)),)), "unknown_derivative_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], lower=2, upper=1),)), "invalid_feasible_domain"),
])
def test_invalid_layout_rejected_before_assembly(change, reason):
    with pytest.raises(ContractError, match=reason):
        change(cell_layout())


def test_state_ownership_shape_and_immutable_snapshots():
    layout = cell_layout()
    values = np.array([1.0, 2.0])
    state = StateView(layout, [("ions.n", FloatArray(values))])
    values[:] = 99
    np.testing.assert_array_equal(state.field("ions.n").values, [1, 2])
    with pytest.raises(ValueError):
        state.field("ions.n").values.setflags(write=True)
    with pytest.raises(TypeError):
        state.fields["ions.n"] = FloatArray([3, 4])
    with pytest.raises(ContractError, match="duplicate_state_write"):
        StateView(layout, [("ions.n", FloatArray([1, 2]))] * 2)
    with pytest.raises(ContractError, match="state_shape_mismatch"):
        StateView(layout, [("ions.n", FloatArray([1]))])
    with pytest.raises(ContractError, match="incomplete_state"):
        StateView(layout, [])


def test_typed_flux_and_source_apply_geometry_once():
    layout = cell_layout(dimensionless=False)
    geometry = Geometry([2, 5], [[0, 1]], [3])
    sink = TermSink(layout, "ions.balance", geometry)
    sink.add(FaceFlux("transport", "faces", [7], PARTICLE / AREA / SECOND))
    sink.add(CellSource("generation", "cells", [11, 13], PARTICLE / VOLUME / SECOND))
    np.testing.assert_array_equal(sink.value(), [-21 + 22, 21 + 65])
    with pytest.raises(ContractError, match="duplicate_contribution"):
        sink.add(CellSource("generation", "cells", [11, 13], PARTICLE / VOLUME / SECOND))


@pytest.mark.parametrize("term,reason", [
    (CellSource("bad", "cells", [1, 2], COULOMB), "unit_mismatch"),
    (CellSource("bad", "missing", [1, 2], ONE), "unknown_support"),
    (CellSource("bad", "cells", [1], ONE), "term_shape_mismatch"),
    (CellSource("bad", "faces", [1], ONE), "wrong_source_support"),
    (FaceFlux("bad", "cells", [1, 2], ONE), "wrong_flux_support"),
    (AlgebraicTerm("bad", "cells", [1, 2], ONE), "wrong_algebraic_support"),
])
def test_wrong_term_units_locations_and_broadcasts_rejected(term, reason):
    sink = TermSink(cell_layout(), "ions.balance", Geometry([1, 2], [[0, 1]], [1], ONE, ONE))
    with pytest.raises(ContractError, match=reason):
        sink.add(term)
    np.testing.assert_array_equal(sink.value(), [0, 0])


def test_surface_charge_and_algebraic_terms_have_distinct_geometry_rules():
    support = Support("surface", "face", (1,))
    layout = Layout((Support("cells", "cell", (2,)), support), (), (EquationSpec("gauss", "electrostatics", "surface", (1,),
                                                   COULOMB, role="constraint"),))
    sink = TermSink(layout, "gauss", Geometry([1, 1], [[0, 1]], [3], face_support="surface"))
    sink.add(SurfaceCharge("sheet", "surface", [2], COULOMB / AREA))
    sink.add(AlgebraicTerm("electrode", "surface", [-1], COULOMB))
    np.testing.assert_array_equal(sink.value(), [5])


@pytest.mark.parametrize("case", [c for c in GATES["NC01"]["fixed_inputs"] if c["id"].endswith("_be")],
                         ids=lambda c: c["id"])
def test_nc01_conservative_step_and_domain_inventory(case):
    coordinates, system, rate_matrix = diffusion_system(case)
    point = coordinates.point(case["initial"])
    volumes = np.asarray(case["volumes"])
    mass = np.diag(volumes)
    gate = GATES["NC01"]
    initial_inventory = [sum(volumes[domain] * point.y[domain]) for domain in case["domains"]]
    accumulated = np.zeros(len(case["domains"]))
    for h in case["step_sizes"]:
        candidate = np.linalg.solve(mass - h * rate_matrix, mass @ point.y)
        right, increment = coordinates.advance(point, candidate - point.y, point.time + h)
        np.testing.assert_allclose(system.conservative_residual(point, right, increment), 0,
                                   atol=gate["state_atol"], rtol=0)
        np.testing.assert_allclose(system.conservative_jacobian(point, right), mass - h * rate_matrix,
                                   atol=gate["state_atol"], rtol=gate["state_rtol"])
        delta = system.storage.delta(point, right, increment)
        for i, domain in enumerate(case["domains"]):
            defect = abs(sum(delta[domain]))
            accumulated[i] += defect
            scale = initial_inventory[i]
            allowed = (gate["inventory_relative_limit"] * scale if scale
                       else gate["zero_inventory_absolute_limit"])
            assert defect <= allowed
            assert abs(sum(volumes[domain] * right.y[domain]) - scale) <= allowed
            assert accumulated[i] <= (gate["cumulative_absolute_inventory_relative_limit"] * scale
                                       if scale else gate["zero_inventory_absolute_limit"])
        point = right
    if "expected_final" in case:
        expected = [float(Fraction(str(value))) for value in case["expected_final"]]
        np.testing.assert_allclose(point.y, expected, atol=gate["state_atol"], rtol=gate["state_rtol"])
    if case["id"] == "zero_be":
        np.testing.assert_array_equal(point.y, [0, 0])


def test_linear_residual_and_ida_matrix_share_the_physical_assembly():
    case = GATES["NC01"]["fixed_inputs"][1]
    coordinates, system, rate_matrix = diffusion_system(case)
    point = coordinates.point([1.2, 1.8])
    ydot, adot = np.array([0.6, -0.3]), np.empty(0)
    np.testing.assert_allclose(system.residual(point, ydot, adot), [0, 0], atol=1e-14)
    derivatives = system.linearize(point, ydot, adot)
    np.testing.assert_array_equal(derivatives.y, -rate_matrix)
    np.testing.assert_array_equal(derivatives.ydot, np.diag(case["volumes"]))
    for cj in (0, 0.1, 100):
        np.testing.assert_allclose(derivatives.ida_matrix(cj), -rate_matrix + cj * np.diag(case["volumes"]))


def test_invalid_equation_shapes_fail_instead_of_broadcasting():
    coordinates = LinearCoordinates(cell_layout())
    point = coordinates.point([1, 2])
    system = ImplicitSystem(LinearStorage([[1, 0]]), lambda _: linear_terms([0, 0], np.eye(2)))
    with pytest.raises(ContractError, match="equation_derivative_shape_mismatch"):
        system.residual(point, np.zeros(2), np.empty(0))
    system = ImplicitSystem(LinearStorage(np.eye(2)), lambda _: linear_terms([0, 0], np.eye(2)))
    with pytest.raises(ContractError, match="rate_shape_mismatch"):
        system.residual(point, np.zeros(1), np.empty(0))


def test_increment_cannot_be_reused_with_different_endpoint_or_layout():
    coordinates = LinearCoordinates(cell_layout())
    left = coordinates.point([1, 2])
    right, increment = coordinates.advance(left, [0.1, -0.1], 0.5)
    increment.validate(left, right)
    with pytest.raises(ContractError, match="increment_endpoint_mismatch"):
        increment.validate(left, coordinates.point([1.2, 1.8], 0.5))
    incomplete = StateIncrement(left.identity, right.identity, {})
    with pytest.raises(ContractError, match="incomplete_increment"):
        incomplete.validate(left, right)
    with pytest.raises(ContractError, match="coordinate_reference_mismatch"):
        replace(coordinates, reference="different").advance(left, [0, 0], 0.1)


def test_float_coordinate_increment_does_not_claim_erased_subulp_information():
    layout = cell_layout(count=1, face_count=0)
    coordinates = LinearCoordinates(layout)
    left = coordinates.point([1e22])
    right, increment = coordinates.advance(left, [1e5], 1)
    assert right.y[0] == left.y[0]
    assert increment.field("ions.n").values[0] == 0
    # NC08 will require an injected precision representation with a retained low
    # word; this ordinary binary64 map makes no such qualification claim.


@pytest.mark.parametrize("normal", [-1, 1])
def test_nc06_capacitor_feedthrough_without_internal_state(normal):
    gate = GATES["NC06"]
    inputs = gate["fixed_inputs"]["capacitor"]
    port = TerminalPort("left" if normal == -1 else "right", normal, inputs["area"])
    for slope in inputs["slopes"]:
        for time in inputs["times"]:
            voltage = inputs["V0"] + slope * time
            displacement = inputs["epsilon"] * voltage / inputs["length"]
            d_dot = inputs["epsilon"] * slope / inputs["length"]
            np.testing.assert_allclose(port.charge(displacement), -normal * inputs["C"] * voltage,
                                       atol=gate["algebraic_atol"], rtol=gate["algebraic_rtol"])
            np.testing.assert_allclose(port.current(0, d_dot), -normal * inputs["C"] * slope,
                                       atol=gate["algebraic_atol"], rtol=gate["algebraic_rtol"])


def test_accepted_step_owns_history_and_rejects_failed_physical_checks():
    coordinates = LinearCoordinates(cell_layout())
    left = coordinates.point([1, 2], inputs=[0])
    right, increment = coordinates.advance(left, [0.25, -0.25], 0.5, [0.1])
    derivative = np.array([0.5, -0.5])
    storage_delta = np.array([0.25, -0.25])
    step = AcceptedStep(left, right, increment, storage_delta, [0.2], derivative, [0.2],
                        "conservative-be-v1", "continuous", (("inventory_defect", 0.0),), True)
    derivative[:] = 999
    storage_delta[:] = 999
    np.testing.assert_array_equal(step.derivative, [0.5, -0.5])
    np.testing.assert_array_equal(step.storage_delta, [0.25, -0.25])
    with pytest.raises(ValueError):
        step.derivative.setflags(write=True)
    with pytest.raises(ContractError, match="physical_step_rejected"):
        replace(step, physical_checks_passed=False)
    with pytest.raises(ContractError, match="physical_step_rejected"):
        replace(step, acceptance_metrics=())
    with pytest.raises(ContractError, match="step_derivative_shape_mismatch"):
        replace(step, derivative=[0])


def test_core_contract_module_has_no_product_or_server_dependencies():
    import ast
    import scripts.benchmarks.contract_prototype as candidate
    tree = ast.parse(Path(candidate.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(module and module.split(".")[0] in
                   {"backend", "models", "experiments", "solarlab_server", "solarlab_research"}
                   for module in modules)


def test_units_are_exact_dimensions_not_silent_truncated_tuples():
    assert (PARTICLE / VOLUME / SECOND) * VOLUME == PARTICLE / SECOND
    assert (COULOMB / AREA) * AREA == COULOMB
    with pytest.raises(ContractError, match="invalid_unit"):
        Unit((1, 2))


@pytest.mark.parametrize("callback", ["delta", "rate_jvp"])
@pytest.mark.parametrize("value,reason", [([3], "storage_callback_shape_mismatch"),
                                         ([np.nan, np.nan], "nonfinite_storage_callback")])
def test_malformed_storage_provider_cannot_broadcast_or_publish_nan(callback, value, reason):
    class MalformedStorage(LinearStorage):
        def delta(self, *args):
            return np.asarray(value) if callback == "delta" else super().delta(*args)

        def rate_jvp(self, *args):
            return np.asarray(value) if callback == "rate_jvp" else super().rate_jvp(*args)

    coordinates = LinearCoordinates(cell_layout())
    left = coordinates.point([1, 2])
    right, increment = coordinates.advance(left, [0, 0], 1)
    system = ImplicitSystem(MalformedStorage(np.eye(2)), lambda _: linear_terms([0, 0], np.zeros((2, 2))))
    with pytest.raises(ContractError, match=reason):
        if callback == "delta":
            system.conservative_residual(left, right, increment)
        else:
            system.linearize(right, np.zeros(2), np.empty(0))


def test_equal_shape_does_not_authorize_a_different_face_support():
    layout = cell_layout()
    layout = replace(layout, supports=layout.supports + (Support("other_faces", "face", (1,)),))
    sink = TermSink(layout, "ions.balance", Geometry([1, 1], [[0, 1]], [1], ONE, ONE))
    with pytest.raises(ContractError, match="geometry_support_mismatch"):
        sink.add(FaceFlux("other-domain-flux", "other_faces", [9], ONE / SECOND))
    np.testing.assert_array_equal(sink.value(), [0, 0])


@pytest.mark.parametrize("indices", [[[0.9, 1.9]], [[0, np.nan]], [[0, np.inf]], [[0, 2**64]]])
def test_topology_cannot_silently_truncate_indices(indices):
    with pytest.raises(ContractError, match="invalid_index"):
        Geometry([1, 1], indices, [1], ONE, ONE)
    point = LinearCoordinates(cell_layout()).point([1, 2])
    with pytest.raises(ContractError, match="invalid_index"):
        point.state.face_difference("ions.n", indices)
    with pytest.raises(ContractError, match="invalid_index"):
        point.state.field("ions.n").take(np.asarray(indices).ravel())


@pytest.mark.parametrize("operation", ["value", "first", "rate_jvp", "residual", "linearize"])
def test_linear_storage_reference_checked_by_every_operation(operation):
    point = LinearCoordinates(cell_layout(), reference="other-reference").point([1, 2])
    storage = LinearStorage(np.eye(2))
    system = ImplicitSystem(storage, lambda _: linear_terms([0, 0], np.zeros((2, 2))))
    with pytest.raises(ContractError, match="linear_storage_coordinate_mismatch"):
        if operation in {"value", "first"}:
            getattr(storage, operation)(point)
        elif operation == "rate_jvp":
            storage.rate_jvp(point, np.zeros(2), np.empty(0), np.ones(2), np.empty(0), 0)
        else:
            getattr(system, operation)(point, np.zeros(2), np.empty(0))


def test_accepted_step_owns_nested_acceptance_evidence():
    coordinates = LinearCoordinates(cell_layout())
    left = coordinates.point([1, 2])
    right, increment = coordinates.advance(left, [0, 0], 1)
    metrics = [["inventory_defect", 0.0]]
    step = AcceptedStep(left, right, increment, [0, 0], [], [0, 0], [],
                        "conservative-be-v1", "continuous", metrics, True)
    metrics[0][1] = 999
    assert step.acceptance_metrics == (("inventory_defect", 0.0),)

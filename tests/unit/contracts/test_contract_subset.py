"""Route-independent public contract subset (P02-02) in ``solarlab.contracts``.

The prototype in ``scripts/benchmarks/contract_prototype.py`` remains the
qualified P02-02 reference; these checks bind the migrated subset to it
(bit-identical identities and assembly values: M class) and repeat the
negative contract checks that apply to the subset.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmarks.contract_prototype as prototype
import solarlab.contracts as contracts
from solarlab.contracts import (
    AREA, COULOMB, ONE, PARTICLE, SECOND, VOLT, VOLUME,
    AlgebraicTerm, CellSource, ContractError, EquationSpec, FaceFlux, FloatArray, Geometry,
    Layout, LinearCoordinates, StateIncrement, StateView, Support, SurfaceCharge,
    TerminalPort, TermSink, Unit, VariableSpec,
)
from solarlab.contracts import pending

SRC = Path(contracts.__file__).parent


def cell_layout(module=contracts, count=2, face_count=1, *, reverse=False, dimensionless=True):
    density_unit = module.ONE if dimensionless else module.PARTICLE / module.VOLUME
    rate_unit = module.ONE / module.SECOND if dimensionless else module.PARTICLE / module.SECOND
    supports = (module.Support("cells", "cell", (count,)),
                module.Support("faces", "face", (face_count,)))
    variables = (module.VariableSpec("ions.n", "ions", "cells", (count,), density_unit, lower=0),)
    equations = (module.EquationSpec("ions.balance", "assembly", "cells", (count,), rate_unit,
                                     derivative_support=("ions.n",)),)
    return module.Layout(supports[::-1] if reverse else supports,
                         variables[::-1] if reverse else variables,
                         equations[::-1] if reverse else equations)


def test_layout_and_point_identities_match_the_p02_02_prototype_bitwise():
    for reverse in (False, True):
        new, old = cell_layout(reverse=reverse), cell_layout(prototype, reverse=reverse)
        assert new.identity == old.identity
        assert dict(new.offsets) == dict(old.offsets)
        new_point = LinearCoordinates(new).point([1.5, 2.25], 0.5, [3.0])
        old_point = prototype.LinearCoordinates(old).point([1.5, 2.25], 0.5, [3.0])
        assert new_point.identity == old_point.identity
        new_right, new_inc = LinearCoordinates(new).advance(new_point, [0.25, -0.5], 1.0, [3.5])
        old_right, old_inc = prototype.LinearCoordinates(old).advance(old_point, [0.25, -0.5],
                                                                      1.0, [3.5])
        assert new_right.identity == old_right.identity
        assert (new_inc.field("ions.n").identity_bytes()
                == old_inc.field("ions.n").identity_bytes())


def test_term_sink_values_match_the_prototype_bitwise():
    rng = np.random.default_rng(20261014)
    volumes, measures = rng.uniform(0.1, 3.0, 5), rng.uniform(0.1, 3.0, 4)
    pairs = np.column_stack((np.arange(4), np.arange(1, 5)))
    flux, source = rng.normal(size=4), rng.normal(size=5)
    values = []
    for module in (contracts, prototype):
        layout = cell_layout(module, 5, 4, dimensionless=False)
        sink = module.TermSink(layout, "ions.balance", module.Geometry(volumes, pairs, measures))
        sink.add(module.FaceFlux("t", "faces", flux, module.PARTICLE / module.AREA / module.SECOND))
        sink.add(module.CellSource("g", "cells", source,
                                   module.PARTICLE / module.VOLUME / module.SECOND))
        values.append(np.asarray(sink.value()))
    assert values[0].tobytes() == values[1].tobytes()


def test_registration_order_does_not_change_layout_or_offsets():
    supports = (Support("nodes", "cell", (2,)), Support("contact", "global", (1,)))
    variables = (VariableSpec("p", "holes", "nodes", (2,), PARTICLE / VOLUME),
                 VariableSpec("n", "electrons", "nodes", (2,), PARTICLE / VOLUME),
                 VariableSpec("port", "contact", "contact", (1,), COULOMB, role="constraint"))
    equations = (EquationSpec("poisson", "electrostatics", "nodes", (2,), COULOMB,
                              role="constraint", derivative_support=("n", "p", "port")),)
    forward = Layout(supports, variables, equations)
    reverse = Layout(supports[::-1], variables[::-1], equations[::-1])
    assert forward.identity == reverse.identity
    assert tuple(forward.offsets) == ("n", "p", "port")


@pytest.mark.parametrize("change,reason", [
    (lambda x: replace(x, variables=x.variables + x.variables), "duplicate_owner"),
    (lambda x: replace(x, supports=x.supports + x.supports), "duplicate_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], support="missing"),)),
     "unknown_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], shape=(1,)),)),
     "support_shape_mismatch"),
    (lambda x: replace(x, equations=(replace(x.equations[0], derivative_support=("missing",)),)),
     "unknown_derivative_support"),
    (lambda x: replace(x, variables=(replace(x.variables[0], lower=2, upper=1),)),
     "invalid_feasible_domain"),
    (lambda x: replace(x, variables=(replace(x.variables[0], role="free"),)), "invalid_role"),
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
    for fields, reason in (([("ions.n", FloatArray([1, 2]))] * 2, "duplicate_state_write"),
                           ([("ions.n", FloatArray([1]))], "state_shape_mismatch"),
                           ([("other", FloatArray([1, 2]))], "unknown_variable"),
                           ([], "incomplete_state")):
        with pytest.raises(ContractError, match=reason):
            StateView(layout, fields)
    with pytest.raises(ContractError, match="nonfinite_array"):
        FloatArray([np.nan, 1.0])


def test_state_view_differences_and_electrochemical_difference():
    layout = Layout((Support("cells", "cell", (3,)),),
                    (VariableSpec("n", "carrier", "cells", (3,), PARTICLE / VOLUME, lower=0),
                     VariableSpec("phi", "electrostatics", "cells", (3,), VOLT)), ())
    state = StateView(layout, [("n", FloatArray([1e22, 2e22, 4e22])),
                               ("phi", FloatArray([0.0, 0.1, 0.3]))])
    pairs = [[0, 1], [1, 2]]
    np.testing.assert_array_equal(state.face_difference("phi", pairs).values,
                                  [0.1 - 0.0, 0.3 - 0.1])
    # Ratio 2 takes the log(a)-log(b) branch: error ~ eps*log(1e22)/log(2).
    np.testing.assert_allclose(state.log_ratio("n", pairs).values, np.log([2.0, 2.0]),
                               rtol=4e-14)
    vt = 0.025
    value = state.electrochemical_difference("n", "phi", pairs, vt, potential_sign=-1)
    np.testing.assert_allclose(value, np.log(2.0) - np.array([0.1, 0.2]) / vt, rtol=4e-14)
    with pytest.raises(ContractError, match="activity_potential_unit_mismatch"):
        state.electrochemical_difference("n", "n", pairs, vt, potential_sign=1)
    with pytest.raises(ContractError, match="invalid_activity_parameters"):
        state.electrochemical_difference("n", "phi", pairs, -1.0, potential_sign=1)
    with pytest.raises(ContractError, match="face_outside_support"):
        state.face_difference("phi", [[0, 3]])


@pytest.mark.parametrize("indices", [[[0.9, 1.9]], [[0, np.nan]], [[0, np.inf]], [[0, 2**64]]])
def test_topology_cannot_silently_truncate_indices(indices):
    with pytest.raises(ContractError, match="invalid_index"):
        Geometry([1, 1], indices, [1], ONE, ONE)
    point = LinearCoordinates(cell_layout()).point([1, 2])
    with pytest.raises(ContractError, match="invalid_index"):
        point.state.face_difference("ions.n", indices)


def test_increment_cannot_be_reused_or_forged():
    coordinates = LinearCoordinates(cell_layout())
    left = coordinates.point([1, 2])
    right, increment = coordinates.advance(left, [0.1, -0.1], 0.5)
    increment.validate(left, right)
    assert StateIncrement.from_points(left, right).field("ions.n").identity_bytes() == \
        increment.field("ions.n").identity_bytes()
    with pytest.raises(ContractError, match="increment_endpoint_mismatch"):
        increment.validate(left, coordinates.point([1.2, 1.8], 0.5))
    with pytest.raises(ContractError, match="incomplete_increment"):
        StateIncrement(left.identity, right.identity, {}).validate(left, right)
    forged = StateIncrement(left.identity, right.identity, {"ions.n": FloatArray([0.2, -0.1])})
    with pytest.raises(ContractError, match="increment_authority_mismatch"):
        forged.validate(left, right)
    with pytest.raises(ContractError, match="coordinate_reference_mismatch"):
        replace(coordinates, reference="different").advance(left, [0, 0], 0.1)


def test_temporal_log_ratio_and_face_delta():
    layout = Layout((Support("cells", "cell", (2,)),),
                    (VariableSpec("n", "carrier", "cells", (2,), PARTICLE / VOLUME, lower=0),
                     VariableSpec("phi", "electrostatics", "cells", (2,), VOLT)), ())
    coordinates = LinearCoordinates(layout)
    left = coordinates.point([1e22, 2e22, 0.0, 0.1])
    right, increment = coordinates.advance(left, [1e21, 0.0, 0.0, 0.05], 1.0)
    np.testing.assert_allclose(increment.log_ratio(left, right, "n").values,
                               [np.log1p(0.1), 0.0], rtol=1e-15)
    np.testing.assert_allclose(increment.face_delta(left, right, "phi", [[0, 1]]), [0.05],
                               rtol=1e-12)
    delta = increment.electrochemical_delta(left, right, "n", "phi", [[0, 1]], 0.025,
                                            potential_sign=1)
    np.testing.assert_allclose(delta, [-np.log1p(0.1) + 0.05 / 0.025], rtol=1e-12)


def test_typed_flux_and_source_apply_geometry_once():
    layout = cell_layout(dimensionless=False)
    sink = TermSink(layout, "ions.balance", Geometry([2, 5], [[0, 1]], [3]))
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
    layout = Layout((Support("cells", "cell", (2,)), Support("surface", "face", (1,))), (),
                    (EquationSpec("gauss", "electrostatics", "surface", (1,), COULOMB,
                                  role="constraint"),))
    sink = TermSink(layout, "gauss", Geometry([1, 1], [[0, 1]], [3], face_support="surface"))
    sink.add(SurfaceCharge("sheet", "surface", [2], COULOMB / AREA))
    sink.add(AlgebraicTerm("electrode", "surface", [-1], COULOMB))
    np.testing.assert_array_equal(sink.value(), [5])


def test_equal_shape_does_not_authorize_a_different_face_support():
    layout = cell_layout()
    layout = replace(layout, supports=layout.supports + (Support("other_faces", "face", (1,)),))
    sink = TermSink(layout, "ions.balance", Geometry([1, 1], [[0, 1]], [1], ONE, ONE))
    with pytest.raises(ContractError, match="geometry_support_mismatch"):
        sink.add(FaceFlux("other-domain-flux", "other_faces", [9], ONE / SECOND))


@pytest.mark.parametrize("normal", [-1, 1])
def test_port_capacitor_feedthrough_signs(normal):
    area, epsilon, length, voltage, slope = 2.0, 3.0, 0.5, 0.7, 4.0
    port = TerminalPort("left" if normal == -1 else "right", normal, area)
    capacitance = epsilon * area / length
    np.testing.assert_allclose(port.charge([epsilon * voltage / length]),
                               [-normal * capacitance * voltage], rtol=1e-15)
    np.testing.assert_allclose(port.current([0.0], [epsilon * slope / length]),
                               [-normal * capacitance * slope], rtol=1e-15)
    with pytest.raises(ContractError, match="port_unit_mismatch"):
        port.charge([1.0], unit=COULOMB)
    with pytest.raises(ContractError, match="port_shape_mismatch"):
        port.current([1.0], [1.0, 2.0])
    with pytest.raises(ContractError, match="invalid_port"):
        TerminalPort("bad", 0, 1.0)


def test_units_are_exact_dimensions():
    assert (PARTICLE / VOLUME / SECOND) * VOLUME == PARTICLE / SECOND
    assert (COULOMB / AREA) * AREA == COULOMB
    with pytest.raises(ContractError, match="invalid_unit"):
        Unit((1, 2))


@pytest.mark.parametrize("name", sorted(pending.PENDING_G2R))
def test_g2r_dependent_interfaces_are_named_placeholders_that_fail_closed(name):
    placeholder = getattr(pending, name)
    assert placeholder.reason == pending.PENDING_G2R[name]
    with pytest.raises(ContractError, match=f"pending_g2r:{name}"):
        placeholder()
    with pytest.raises(ContractError, match=f"pending_g2r:{name}"):
        placeholder.residual  # noqa: B018 - attribute access must fail closed
    assert getattr(placeholder, "__deepcopy__", None) is None  # protocol lookups stay normal
    assert name not in contracts.__all__


def test_contract_modules_import_no_product_legacy_or_server_layer():
    forbidden = {"perovskite_sim", "backend", "solarlab_server", "solarlab_research", "scripts"}
    forbidden_solarlab = {"config", "device", "materials", "experiments", "physics", "mesh",
                          "discretization", "equations", "solvers", "io", "sweeps"}
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names]
        for module in filter(None, modules):
            parts = module.split(".")
            assert parts[0] not in forbidden, (path.name, module)
            if parts[0] == "solarlab":
                assert parts[1] == "contracts", (path.name, module)
                assert parts[1] not in forbidden_solarlab

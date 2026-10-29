"""The Assembler on synthetic Terms: 1D, 2D and embedded node sets.

The toy binding is a nonlinear diffusion-reaction pair (``c`` with storage,
``phi`` algebraic) with a Robin port and a Dirichlet constraint, so every
contribution kind and the cross-block tangents are exercised without device
physics. Checks: finite-difference J and S_u, the TermSink (P02-02) residual
order, exact linearity in each measure (geometry applied once), discrete
conservation, constraint ownership and the declared-contract rejections.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from solarlab.contracts import (
    AREA, LENGTH, ONE, PARTICLE, SECOND, VOLUME, CellSource, EquationSpec, FaceFlux, Geometry,
    Layout, Support, TermSink, Unit,
)
from solarlab.contracts.errors import ContractError
from solarlab.equations.assembly import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, MEASURE, PORT_FLUX, STORAGE, Assembler, Block,
    ConstraintRows, EvaluationState, FaceValue, FieldValues, LocalValue, Mesh, NodeSet,
    OperatingInputs, Port, SupportGeometry, SystemLayout,
)
from solarlab.equations.assembly.terms import Contribution, TermValue
from solarlab.mesh.geometry import half_endpoint_geometry, integer_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D

Vector: TypeAlias = NDArray[np.float64]
FLUX = PARTICLE / (AREA * SECOND)
SOURCE = PARTICLE / (VOLUME * SECOND)
DENSITY = PARTICLE / VOLUME


@dataclass(frozen=True)
class Identity:
    """Field map returning the block coordinates themselves."""

    blocks: tuple[str, ...]
    identity: str = "toy.identity"

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        return FieldValues({block: layout.view(u, block) for block in self.blocks}, True)


@dataclass(frozen=True)
class ToyTerm:
    """A Term from plain functions; ``value(state, derivatives)`` returns its mapping."""

    id: str
    kind: str
    support: str
    equations: tuple[str, ...]
    dependencies: tuple[str, ...]
    unit: Unit
    value: Callable[[EvaluationState, bool], Mapping[str, TermValue]] = field(compare=False)
    port: str | None = None
    identity: str = "toy"
    increment_function: Callable[[EvaluationState, EvaluationState], Mapping[str, Contribution]] | None = \
        field(default=None, compare=False)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        return self.value(state, derivatives)

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        assert self.increment_function is not None
        return self.increment_function(state, previous)


def toy_terms(support: str, pairs: NDArray[np.int64], spacing: Vector, flux_unit: Unit,
              source_unit: Unit, port: tuple[str, tuple[int, ...]],
              dirichlet: tuple[int, ...]) -> list[ToyTerm]:
    left, right = pairs[:, 0], pairs[:, 1]
    k = 1.0 / spacing
    port_id, port_nodes = port

    def diffusion(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c, phi = state.field("c"), state.field("phi")
        a, b = c[left], c[right]
        drift = phi[right] - phi[left]
        flux = k * ((a - b) * (1.0 + a * b) + 0.3 * drift * (a + b))
        if not derivatives:
            return {"c": FaceValue(flux)}
        return {"c": FaceValue(flux, {"c": k * ((1.0 + a * b) + (a - b) * b + 0.3 * drift),
                                      "phi": -0.3 * k * (a + b)},
                               {"c": k * (-(1.0 + a * b) + (a - b) * a + 0.3 * drift),
                                "phi": 0.3 * k * (a + b)})}

    def reaction(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c, phi = state.field("c"), state.field("phi")
        source = 0.7 - c ** 3 + 0.2 * phi * c
        if not derivatives:
            return {"c": LocalValue(source)}
        # A weighted tangent keeps an explicit product order.
        return {"c": LocalValue(source, {"c": MEASURE * (-3.0 * c ** 2 + 0.2 * phi), "phi": 0.2 * c})}

    def potential(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        phi = state.field("phi")
        flux = -k * (phi[right] - phi[left])
        if not derivatives:
            return {"phi": FaceValue(flux)}
        return {"phi": FaceValue(flux, {"phi": k}, {"phi": -k})}

    def charge(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c = state.field("c")
        value = c - 0.5
        return {"phi": LocalValue(value, {"c": np.ones_like(c)} if derivatives else {})}

    def robin(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c = state.field("c")[np.asarray(port_nodes, dtype=np.intp)]
        value = -2.0 * (c - 0.25) * (1.0 + c)
        return {"c": LocalValue(value, {"c": -2.0 * ((1.0 + c) + (c - 0.25))} if derivatives else {})}

    def storage(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c = state.field("c")
        value = c + 0.5 * c * c
        return {"c": LocalValue(value, {"c": MEASURE * (1.0 + c)} if derivatives else {})}

    def storage_increment(state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        c, old = state.field("c"), previous.field("c")
        return {"c": (c - old) * (1.0 + 0.5 * (c + old))}

    terms = [
        ToyTerm("c.diffusion", FACE_FLUX, support, ("c",), ("c", "phi"), flux_unit, diffusion),
        ToyTerm("c.reaction", CELL_SOURCE, support, ("c",), ("c", "phi"), source_unit, reaction),
        ToyTerm("c.robin", PORT_FLUX, support, ("c",), ("c",), flux_unit, robin, port=port_id),
        ToyTerm("c.storage", STORAGE, support, ("c",), ("c",), DENSITY, storage,
                increment_function=storage_increment),
        ToyTerm("phi.potential", FACE_FLUX, support, ("phi",), ("phi",), flux_unit, potential),
        ToyTerm("phi.charge", CELL_SOURCE, support, ("phi",), ("c",), source_unit, charge),
    ]
    def fixed(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        phi = state.field("phi")[np.asarray(dirichlet, dtype=np.intp)]
        return {"phi": LocalValue(phi - 0.1, {"phi": np.ones(len(dirichlet))} if derivatives else {})}

    terms.append(ToyTerm("phi.dirichlet", CONSTRAINT, support, ("phi",), ("phi",), ONE, fixed))
    return terms


@dataclass(frozen=True)
class Toy:
    assembler: Assembler
    count: int


def toy_1d(nodes: Vector | None = None) -> Toy:
    x = np.array([0.0, 0.3, 0.45, 1.0, 1.7, 2.0]) if nodes is None else nodes
    geometry = integer_endpoint_geometry(x)
    contract = geometry.to_contract_geometry(cell_support="nodes", face_support="faces")
    count = x.size
    ports = (Port("left", (0,), -1, np.array([1.0]), ONE), Port("right", (count - 1,), 1, np.array([1.0]), ONE))
    layout = SystemLayout((NodeSet("nodes", count),),
                          (Block("phi", "nodes", ONE, FLUX), Block("c", "nodes", ONE, FLUX)),
                          (ConstraintRows("phi", (count - 1,), "phi.dirichlet", ONE),))
    terms = toy_terms("nodes", contract.pairs, np.diff(x), FLUX, SOURCE, ("left", (0,)), (count - 1,))
    return Toy(Assembler(layout, Mesh((SupportGeometry("nodes", contract, ports),)),
                         Identity(("phi", "c")), terms), count)


def toy_2d() -> Toy:
    rect = RectangularGeometry2D(np.array([0.0, 0.4, 1.0, 1.3]), half_endpoint_geometry(np.array([0.0, 0.5, 0.8, 1.6])))
    contract = rect.contract_geometry
    count = rect.node_count
    bottom, top = tuple(range(rect.nx)), tuple(range(count - rect.nx, count))
    ports = (Port("bottom", bottom, -1, rect.x_widths_m, LENGTH), Port("top", top, 1, rect.x_widths_m, LENGTH))
    balance = PARTICLE / (LENGTH * SECOND)
    layout = SystemLayout((NodeSet("nodes", count),),
                          (Block("phi", "nodes", ONE, balance), Block("c", "nodes", ONE, balance)),
                          (ConstraintRows("phi", top, "phi.dirichlet", ONE),))
    terms = toy_terms("nodes", contract.pairs, rect.face_spacing_m, FLUX, SOURCE, ("bottom", bottom), top)
    return Toy(Assembler(layout, Mesh((SupportGeometry("nodes", contract, ports),)),
                         Identity(("phi", "c")), terms), count)


def numerical_jacobian(function: Callable[[Vector], Vector], u: Vector, step: float = 1e-6) -> NDArray[np.float64]:
    columns = []
    for index in range(u.size):
        def central(h: float) -> Vector:
            plus, minus = u.copy(), u.copy()
            plus[index] += h
            minus[index] -= h
            result: Vector = (function(plus) - function(minus)) / (2.0 * h)
            return result
        columns.append((4.0 * central(0.5 * step) - central(step)) / 3.0)
    return np.column_stack(columns)


def state_vector(toy: Toy, seed: int) -> Vector:
    rng = np.random.default_rng(seed)
    result: Vector = 0.5 + 0.3 * rng.standard_normal(2 * toy.count)
    return result


@pytest.mark.parametrize("build", [toy_1d, toy_2d], ids=["1d", "2d"])
def test_jacobian_and_storage_tangent_match_finite_differences(build: Callable[[], Toy]) -> None:
    toy = build()
    inputs = OperatingInputs()
    for seed in range(3):
        u = state_vector(toy, seed)
        system = toy.assembler.evaluate(u, inputs, storage=True)
        assert system.J is not None and system.S_u is not None and system.M is not None
        F = lambda v: np.asarray(toy.assembler.evaluate(v, inputs, derivatives=False).F)  # noqa: E731
        S = lambda v: np.asarray(toy.assembler.evaluate(v, inputs, derivatives=False, storage=True).S)  # noqa: E731
        np.testing.assert_allclose(system.J.toarray(), numerical_jacobian(F, u), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(system.S_u.toarray(), numerical_jacobian(S, u), rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(system.M.toarray(), -system.S_u.toarray())
        old = state_vector(toy, 100 + seed)
        increment = toy.assembler.storage_increment(u, inputs, old, inputs)
        S_old = toy.assembler.evaluate(old, inputs, derivatives=False, storage=True).S
        assert system.S is not None and S_old is not None
        np.testing.assert_allclose(increment, system.S - S_old, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("build", [toy_1d, toy_2d], ids=["1d", "2d"])
def test_constraint_rows_are_owned_by_one_term(build: Callable[[], Toy]) -> None:
    toy = build()
    u = state_vector(toy, 5)
    system = toy.assembler.evaluate(u, OperatingInputs(), storage=True)
    assert system.J is not None and system.S is not None and system.S_u is not None
    rows = np.flatnonzero(system.boundary_rows)
    J = system.J.toarray()
    for row in rows:
        assert system.F[row] == u[row] - 0.1
        assert np.flatnonzero(J[row]).tolist() == [row] and J[row, row] == 1.0
    assert np.all(system.S[rows] == 0.0) and system.S_u.toarray()[rows].sum() == 0.0
    # The dropped balance stays observable as the boundary defect.
    assert np.any(system.balance[rows] != system.F[rows])


def test_residual_follows_the_term_sink_contract() -> None:
    """Face and cell values assemble exactly as the P02-02 TermSink orders them."""
    toy = toy_1d()
    u = state_vector(toy, 9)
    count = toy.count
    state = toy.assembler.state(u, OperatingInputs())
    assert state is not None
    terms = {term.id: term for term in toy.assembler.terms}
    flux = terms["c.diffusion"].evaluate(state, derivatives=False)["c"]
    source = terms["c.reaction"].evaluate(state, derivatives=False)["c"]
    assert isinstance(flux, FaceValue) and isinstance(source, LocalValue)
    support = toy.assembler.mesh.support("nodes")
    contract = Layout((Support("nodes", "cell", (count,)), Support("faces", "face", (count - 1,))), (),
                      (EquationSpec("c", "c", "nodes", (count,), FLUX),))
    sink = TermSink(contract, "c", support.geometry)
    sink.add(FaceFlux("flux", "faces", np.asarray(flux.flux), FLUX))
    sink.add(CellSource("source", "nodes", np.asarray(source.value), SOURCE))
    expected = np.array(sink.value(), copy=True)
    expected[0] += 1.0 * float(np.asarray(terms["c.robin"].evaluate(state, derivatives=False)["c"].value)[0])  # type: ignore[union-attr]
    balance = toy.assembler.evaluate(u, OperatingInputs(), derivatives=False).balance
    np.testing.assert_array_equal(balance[toy.assembler.layout.offsets["c"]], expected)


def test_each_measure_enters_exactly_once() -> None:
    """Doubling cell (face) measures doubles exactly the source (flux) pieces."""
    toy = toy_1d()
    u = state_vector(toy, 11)
    support = toy.assembler.mesh.support("nodes")
    g = support.geometry

    def rebuilt(volumes: Vector, faces: Vector) -> Assembler:
        geometry = Geometry(volumes, g.pairs, faces, g.volume_unit, g.face_unit, g.cell_support, g.face_support)
        return Assembler(toy.assembler.layout, Mesh((replace(support, geometry=geometry),)),
                         toy.assembler.fields, toy.assembler.terms)

    def pieces(assembler: Assembler) -> tuple[Vector, Vector]:
        state = assembler.state(u, OperatingInputs())
        assert state is not None
        only = {"c.diffusion": FACE_FLUX, "c.reaction": CELL_SOURCE}
        results = assembler._evaluate(state, False, tuple(only.values()))
        face = assembler._balance("c", [r for r in results if r[0].id == "c.diffusion"])
        cell = assembler._balance("c", [r for r in results if r[0].id == "c.reaction"])
        return face, cell

    face, cell = pieces(toy.assembler)
    face_cells, cell_cells = pieces(rebuilt(2.0 * g.volumes, g.face_measures))
    face_faces, cell_faces = pieces(rebuilt(g.volumes, 2.0 * g.face_measures))
    np.testing.assert_array_equal(cell_cells, 2.0 * cell)
    np.testing.assert_array_equal(face_cells, face)
    np.testing.assert_array_equal(face_faces, 2.0 * face)
    np.testing.assert_array_equal(cell_faces, cell)
    # J entries scale the same way (one cell measure in the weighted tangent).
    J = toy.assembler.evaluate(u, OperatingInputs()).J
    J2 = rebuilt(2.0 * g.volumes, g.face_measures).evaluate(u, OperatingInputs()).J
    assert J is not None and J2 is not None
    rows = toy.assembler.layout.offsets["c"]
    diagonal = np.arange(rows.start, rows.stop)
    reaction = J2.toarray()[diagonal, diagonal] - J.toarray()[diagonal, diagonal]
    state = toy.assembler.state(u, OperatingInputs())
    assert state is not None
    c, phi = state.field("c"), state.field("phi")
    np.testing.assert_allclose(reaction, g.volumes * (-3.0 * c ** 2 + 0.2 * phi), rtol=1e-12)


@pytest.mark.parametrize("build", [toy_1d, toy_2d], ids=["1d", "2d"])
def test_discrete_conservation_of_the_balance_block(build: Callable[[], Toy]) -> None:
    """Sum of c rows = integrated source + port inflow: face fluxes telescope."""
    toy = build()
    u = state_vector(toy, 13)
    assembler = toy.assembler
    state = assembler.state(u, OperatingInputs())
    assert state is not None
    support = assembler.mesh.support("nodes")
    terms = {term.id: term for term in assembler.terms}
    source = np.asarray(terms["c.reaction"].evaluate(state, derivatives=False)["c"].value)  # type: ignore[union-attr]
    port = next(p for p in support.ports if p.id == terms["c.robin"].port)
    exchange = np.asarray(terms["c.robin"].evaluate(state, derivatives=False)["c"].value)  # type: ignore[union-attr]
    expected = np.sum(source * support.geometry.volumes) + np.sum(-port.outward_normal * exchange * port.measures)
    total = np.sum(assembler.evaluate(u, OperatingInputs(), derivatives=False).balance[assembler.layout.offsets["c"]])
    assert total == pytest.approx(expected, rel=1e-13, abs=1e-13)


def embedded_toy() -> Assembler:
    """``q`` lives on an embedded domain with its own measures and reads ``c``."""
    x = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0])
    parent = integer_endpoint_geometry(x).to_contract_geometry(cell_support="nodes", face_support="faces")
    inside = (1, 2, 3, 4)
    child = half_endpoint_geometry(x[list(inside)]).to_contract_geometry(cell_support="domain",
                                                                          face_support="domain.faces")
    layout = SystemLayout((NodeSet("nodes", x.size), NodeSet("domain", len(inside), "nodes", inside)),
                          (Block("c", "nodes", ONE, FLUX), Block("q", "domain", ONE, FLUX)))
    pairs, spacing = child.pairs, np.diff(x[list(inside)])
    embedded = np.zeros(x.size, dtype=bool)
    embedded[list(inside)] = True

    def c_reaction(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c = state.field("c")
        q = np.zeros(x.size)
        q[list(inside)] = state.field("q")
        value = 1.0 - c * c - q * c
        tangents: dict[str, Contribution] = {"c": -2.0 * c - q, "q": np.where(embedded, -c, 0.0)}
        return {"c": LocalValue(value, tangents if derivatives else {})}

    def c_flux(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        c = state.field("c")
        k = 1.0 / np.diff(x)
        flux = k * (c[:-1] - c[1:])
        return {"c": FaceValue(flux, {"c": k}, {"c": -k}) if derivatives else FaceValue(flux)}

    def q_flux(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        q, c = state.field("q"), state.field("c")[list(inside)]
        a, b = pairs[:, 0], pairs[:, 1]
        k = 1.0 / spacing
        flux = k * (q[a] - q[b]) * (1.0 + c[a] * c[b])
        if not derivatives:
            return {"q": FaceValue(flux)}
        return {"q": FaceValue(flux, {"q": k * (1.0 + c[a] * c[b]), "c": k * (q[a] - q[b]) * c[b]},
                               {"q": -k * (1.0 + c[a] * c[b]), "c": k * (q[a] - q[b]) * c[a]})}

    def q_reaction(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        q, c = state.field("q"), state.field("c")[list(inside)]
        value = c - q ** 3
        return {"q": LocalValue(value, {"q": -3.0 * q ** 2, "c": np.ones_like(c)} if derivatives else {})}

    terms = [
        ToyTerm("c.flux", FACE_FLUX, "nodes", ("c",), ("c",), FLUX, c_flux),
        ToyTerm("c.reaction", CELL_SOURCE, "nodes", ("c",), ("c", "q"), SOURCE, c_reaction),
        ToyTerm("q.flux", FACE_FLUX, "domain", ("q",), ("q", "c"), FLUX, q_flux),
        ToyTerm("q.reaction", CELL_SOURCE, "domain", ("q",), ("q", "c"), SOURCE, q_reaction),
    ]
    mesh = Mesh((SupportGeometry("nodes", parent), SupportGeometry("domain", child)))
    return Assembler(layout, mesh, Identity(("c", "q")), terms)


def test_embedded_node_sets_couple_through_their_embedding() -> None:
    assembler = embedded_toy()
    assert assembler.layout.block_ranges() == {"c": (0, 6), "q": (6, 10)}
    np.testing.assert_array_equal(assembler.layout.column_map("domain", "c"), [1, 2, 3, 4])
    np.testing.assert_array_equal(assembler.layout.column_map("nodes", "q"), [-1, 0, 1, 2, 3, -1])
    rng = np.random.default_rng(3)
    u = 0.6 + 0.2 * rng.standard_normal(10)
    J = assembler.evaluate(u, OperatingInputs()).J
    assert J is not None
    F = lambda v: np.asarray(assembler.evaluate(v, OperatingInputs(), derivatives=False).F)  # noqa: E731
    np.testing.assert_allclose(J.toarray(), numerical_jacobian(F, u), rtol=1e-8, atol=1e-9)
    # The child's own (half-endpoint) measures weight its rows.
    child = assembler.mesh.support("domain").geometry
    assert child.volumes[0] != assembler.mesh.support("nodes").geometry.volumes[1]


def test_inadmissible_state_has_nan_residual_and_no_jacobian() -> None:
    toy = toy_1d()

    @dataclass(frozen=True)
    class Closed(Identity):
        def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
            values = super().evaluate(layout, u, inputs)
            return FieldValues(values.fields, bool(np.all(u > 0.0)))

    assembler = Assembler(toy.assembler.layout, toy.assembler.mesh, Closed(("phi", "c")), toy.assembler.terms)
    u = state_vector(toy, 0)
    u[3] = -1.0
    system = assembler.evaluate(u, OperatingInputs(), derivatives=False)
    assert not system.admissible and np.all(np.isnan(system.F))
    with pytest.raises(FloatingPointError):
        assembler.evaluate(u, OperatingInputs())


def test_declared_contract_violations_are_rejected() -> None:
    toy = toy_1d()
    layout, mesh, fields = toy.assembler.layout, toy.assembler.mesh, toy.assembler.fields
    terms = [cast(ToyTerm, term) for term in toy.assembler.terms]
    by_id = {term.id: term for term in terms}

    def build(changed: list[ToyTerm], new_layout: SystemLayout = layout) -> Assembler:
        return Assembler(new_layout, mesh, fields, changed)

    others = [term for term in terms if term.id != "c.reaction"]
    with pytest.raises(ContractError, match="unit_mismatch"):
        build(others + [replace(by_id["c.reaction"], unit=FLUX)])
    with pytest.raises(ContractError, match="duplicate_or_empty_term_id"):
        build(terms + [by_id["c.reaction"]])
    with pytest.raises(ContractError, match="unknown_dependency_block"):
        build(others + [replace(by_id["c.reaction"], dependencies=("c", "missing"))])
    with pytest.raises(ContractError, match="declared_constraint_rows_without_owner"):
        build([term for term in terms if term.kind != CONSTRAINT])
    with pytest.raises(ContractError, match="constraint_term_without_declared_rows"):
        build(terms, SystemLayout(layout.node_sets, layout.blocks, ()))
    with pytest.raises(ContractError, match="port_term_needs_port"):
        build([term for term in terms if term.id != "c.robin"] + [replace(by_id["c.robin"], port=None)])
    with pytest.raises(ContractError, match="unknown_port"):
        build([term for term in terms if term.id != "c.robin"] + [replace(by_id["c.robin"], port="top")])
    with pytest.raises(ContractError, match="storage_term_without_increment"):
        Assembler(layout, mesh, fields, [term for term in terms if term.id != "c.storage"] + [_NoIncrement()])
    u = state_vector(toy, 1)

    def undeclared(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        return {"c": LocalValue(state.field("c"), {"phi": np.ones(toy.count)})}

    with pytest.raises(ContractError, match="tangent_with_undeclared_dependency"):
        build(others + [replace(by_id["c.reaction"], dependencies=("c",), value=undeclared)]).evaluate(
            u, OperatingInputs())

    def twice(state: EvaluationState, derivatives: bool) -> Mapping[str, TermValue]:
        return {"c": LocalValue(MEASURE * MEASURE * state.field("c"))}

    with pytest.raises(ContractError, match="exactly_one_measure"):
        build(others + [replace(by_id["c.reaction"], value=twice)]).evaluate(
            u, OperatingInputs(), derivatives=False)


class _NoIncrement:
    """Storage-kind object without an increment method (rejected)."""

    id = "c.storage"
    kind = STORAGE
    support = "nodes"
    equations = ("c",)
    dependencies = ("c",)
    unit = DENSITY
    port = None
    identity = "toy"

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        return {"c": LocalValue(state.field("c"))}

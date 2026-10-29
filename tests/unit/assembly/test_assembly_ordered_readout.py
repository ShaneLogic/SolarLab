"""The ordered port readout against the original 2D contact readout.

The five 2D harness factories at their four probes (Core's 2D-gate probes,
cross-checked against the gate's frozen ``Baseline4c7fb353.npz`` states) are
read out through :meth:`Assembler.reaction_flux_ordered`. The Terms here
return the original constitutive values (SG/CT face fluxes, generation and
the bound recombination as two plain cell parts), so only the readout
arithmetic is under test, not a 2D F/J binding. Every fixed-reservoir port
must equal ``DC2DProblem.contact_particle_fluxes`` in every word, signed zeros
included. The plain ``reaction_flux`` quotient does not, which is why the
ordered observer exists.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import AREA, LENGTH, ONE, PARTICLE, SECOND, VOLUME, Unit
from solarlab.equations.assembly import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, MEASURE, PORT_FLUX, Assembler, Block, ConstraintRows,
    EvaluationState, FaceValue, FieldValues, LocalValue, Mesh, NodeSet, OperatingInputs, Port,
    SupportGeometry, SystemLayout,
)
from solarlab.equations.assembly.terms import TermValue
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from support.assembly_references import REFERENCE_FACTORIES

Vector: TypeAlias = NDArray[np.float64]
FACTORIES = ("dc2d:fixed", "dc2d:contacts", "dc2d:ct", "dc2d:grain", "dc2d:composed")
BLOCKS = ("potential", "electron", "hole")
FLUX = PARTICLE / (AREA * SECOND)
SOURCE = PARTICLE / (VOLUME * SECOND)
BALANCE = PARTICLE / (LENGTH * SECOND)
GATE = Path("/Users/bytedance/Intern/SolarLab/.worktrees/assembly_2d/artifacts/Assembly2DGateV1/Baseline4c7fb353.npz")


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@dataclass(frozen=True)
class Coordinates:
    identity: str = "test.coordinates"

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        return FieldValues({}, True)


@dataclass(frozen=True)
class Original:
    """A Term returning one original 2D constitutive value at the evaluated state."""

    id: str
    kind: str
    equations: tuple[str, ...]
    unit: Unit
    value: Callable[[DC2DProblem, Vector], Any] = field(compare=False)
    device: CompiledDevice2D = field(compare=False, repr=False)
    support: str = "nodes"
    dependencies: tuple[str, ...] = ()
    port: str | None = None
    identity: str = "test.original"

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        problem = DC2DProblem(self.device, state.inputs.voltage_V, state.inputs.generation_scale)
        values = self.value(problem, np.asarray(state.u))
        wrap = FaceValue if self.kind == FACE_FLUX else LocalValue
        return {block: wrap(item) for block, item in zip(self.equations, values)}


@dataclass(frozen=True)
class Zeros:
    """Placeholder constraint values: the readout reads only the balance rows."""

    size: int

    def __call__(self, problem: DC2DProblem, u: Vector) -> tuple[Vector]:
        return (np.zeros(self.size),)


def rate(problem: DC2DProblem, u: Vector) -> Vector:
    _, n, p = problem.device.fields(u, voltage_V=problem.voltage_V)
    result: Vector = np.asarray(problem.recombination_rate(n, p, derivatives=False).rate)
    return result


def readout_binding(device: CompiledDevice2D) -> Assembler:
    """The 2D carrier rows of ``device`` with its fixed channels as constraint rows."""
    g = device.geometry
    count = g.node_count
    sides = g.contact_nodes.reshape(2, g.nx)
    ports = tuple(Port(side, tuple(int(node) for node in nodes), normal, g.x_widths_m, LENGTH)
                  for side, nodes, normal in zip(("y_min", "y_max"), sides, (-1, 1)))
    constraints = []
    terms: list[Any] = [
        Original("carrier.transport", FACE_FLUX, ("electron", "hole"), FLUX,
                 lambda problem, u: (problem.fluxes(u).electron.flux, problem.fluxes(u).hole.flux), device),
        Original("carrier.generation", CELL_SOURCE, ("electron", "hole"), SOURCE,
                 lambda problem, u: (problem.generation_scale * device.generation_source_m3_s,) * 2, device),
        Original("carrier.recombination", CELL_SOURCE, ("electron", "hole"), SOURCE,
                 lambda problem, u: (-rate(problem, u),) * 2, device),
    ]
    for block, carrier in (("electron", "n"), ("hole", "p")):
        nodes = tuple(int(node) for side, part in zip(("left", "right"), sides)
                      if device.column.channel(carrier, side).kind == "fixed_reservoir" for node in part)
        if nodes:
            constraints.append(ConstraintRows(block, tuple(sorted(nodes)), block + ".reservoirs", ONE))
            terms.append(Original(block + ".reservoirs", CONSTRAINT, (block,), ONE, Zeros(len(nodes)), device))
    layout = SystemLayout((NodeSet("nodes", count),), tuple(Block(name, "nodes", ONE, BALANCE) for name in BLOCKS),
                          tuple(constraints))
    return Assembler(layout, Mesh((SupportGeometry("nodes", g.contract_geometry, ports),)), Coordinates(), terms)


def stencils(device: CompiledDevice2D, side: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The y faces leaving one contact row and every x face, in the global face order."""
    g = device.geometry
    first = g.x_face_count + (0 if side == 0 else (g.ny - 2) * g.nx)
    return np.arange(first, first + g.nx), np.arange(g.x_face_count)


def bits(value: object) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).tobytes()


def test_twenty_probes_read_out_every_fixed_port_bitwise(request: pytest.FixtureRequest) -> None:
    gate = np.load(GATE) if GATE.exists() else None
    compared = changed_plain = probes = 0
    for name in FACTORIES:
        case = REFERENCE_FACTORIES[name]()
        device = case.device
        assembler = readout_binding(device)
        for state in case.states:
            probes += 1
            if gate is not None:
                assert bits(gate[f"{name}__{state.name}_u"]) == bits(state.u)
            inputs = OperatingInputs(state.voltage_V, state.generation_scale)
            original = DC2DProblem(device, state.voltage_V, state.generation_scale).contact_particle_fluxes(state.u)
            system = assembler.evaluate(state.u, inputs, derivatives=False)
            for carrier, block, expected in zip(("n", "p"), ("electron", "hole"), original):
                for side, (port, channel_side) in enumerate((("y_min", "left"), ("y_max", "right"))):
                    if device.column.channel(carrier, channel_side).kind != "fixed_reservoir":
                        with pytest.raises(ContractError, match="requires_constraint_rows"):
                            assembler.reaction_flux_ordered(state.u, inputs, block, port,
                                                            normal_faces=stencils(device, side)[0],
                                                            transverse_faces=stencils(device, side)[1])
                        continue
                    normal, transverse = stencils(device, side)
                    flux = assembler.reaction_flux_ordered(state.u, inputs, block, port, normal_faces=normal,
                                                           transverse_faces=transverse)
                    assert bits(flux) == bits(expected[side])
                    # Only the faces incident to the port: the same words.
                    g = device.geometry
                    nodes = g.contact_nodes.reshape(2, g.nx)[side]
                    incident = np.flatnonzero(np.isin(g.contract_geometry.pairs[:g.x_face_count], nodes).any(axis=1))
                    local = assembler.reaction_flux_ordered(state.u, inputs, block, port, normal_faces=normal,
                                                            transverse_faces=incident)
                    assert bits(local) == bits(flux)
                    plain = assembler.reaction_flux(system, block, port)
                    changed_plain += int(np.count_nonzero(np.frombuffer(bits(plain), np.uint64)
                                                          != np.frombuffer(bits(expected[side]), np.uint64)))
                    compared += 1
    assert probes == 20 and compared == 56
    request.node.user_properties.append(("ordered_readout", {
        "probes": probes, "fixed_port_vectors": compared, "changed_words": 0,
        "plain_reaction_flux_changed_words": changed_plain, "gate_states_checked": gate is not None}))


def test_the_readout_identity_records_the_operation_and_both_stencils() -> None:
    device = REFERENCE_FACTORIES["dc2d:fixed"]().device
    assembler = readout_binding(device)
    normal, transverse = stencils(device, 0)
    first = assembler.reaction_flux_ordered_identity("electron", "y_min", normal_faces=normal,
                                                     transverse_faces=transverse)
    assert first == assembler.reaction_flux_ordered_identity("electron", "y_min", normal_faces=normal,
                                                             transverse_faces=transverse)
    g = device.geometry
    incident = np.flatnonzero(np.isin(g.contract_geometry.pairs[:g.x_face_count], g.contact_nodes[:g.nx]).any(axis=1))
    assert first != assembler.reaction_flux_ordered_identity("electron", "y_min", normal_faces=normal,
                                                             transverse_faces=incident)
    assert first != assembler.reaction_flux_ordered_identity("hole", "y_min", normal_faces=normal,
                                                             transverse_faces=transverse)


def test_invalid_stencils_and_bindings_are_refused() -> None:
    case = REFERENCE_FACTORIES["dc2d:fixed"]()
    device, state = case.device, case.states[3]
    g = device.geometry
    assembler = readout_binding(device)
    inputs = OperatingInputs(state.voltage_V, state.generation_scale)
    normal, transverse = stencils(device, 0)
    top, _ = stencils(device, 1)

    def read(normal_faces: Any, transverse_faces: Any, port: str = "y_min",
             binding: Assembler = assembler) -> Vector:
        return binding.reaction_flux_ordered(state.u, inputs, "electron", port, normal_faces=normal_faces,
                                             transverse_faces=transverse_faces)

    cases = {
        "normal_faces_need_one_face_per_port_node": (normal[:-1], transverse),
        "normal_faces_must_leave_the_port_nodes_in_port_order": (normal[::-1], transverse),
        "transverse_faces_must_keep_the_face_order": (normal, transverse[::-1]),
        "transverse_faces_overlap_normal_faces": (normal, np.r_[transverse, normal[:1]]),
        "transverse_faces_miss_an_incident_face": (normal, transverse[1:]),
        "invalid_face_stencil": (normal, np.r_[transverse, g.contract_geometry.pairs.shape[0]]),
    }
    for reason, (normal_faces, transverse_faces) in cases.items():
        with pytest.raises(ContractError, match=reason):
            read(normal_faces, transverse_faces)
    with pytest.raises(ContractError, match="normal_faces_must_leave"):
        read(top, transverse)
    with pytest.raises(ContractError, match="invalid_index"):
        read(normal + 0.5, transverse)
    with pytest.raises(ContractError, match="unknown_port"):
        read(normal, transverse, port="x_min")
    # Weighted parts and exchanges on the port are refused (chains: 1D test below).
    terms: list[Any] = list(assembler.terms)
    weighted = replace(terms[1], value=lambda problem, u: (MEASURE * np.ones(g.node_count),) * 2)
    binding = Assembler(assembler.layout, assembler.mesh, assembler.fields, [terms[0], weighted, *terms[2:]])
    with pytest.raises(ContractError, match="plain_contributions"):
        read(normal, transverse, binding=binding)
    exchange = Original("electron.exchange", PORT_FLUX, ("electron",), FLUX,
                        lambda problem, u: (np.zeros(g.nx),), device, port="y_min")
    binding = Assembler(assembler.layout, assembler.mesh, assembler.fields, [*terms, exchange])
    with pytest.raises(ContractError, match="port_has_an_exchange_term"):
        read(normal, transverse, binding=binding)
    # A port measured in another unit than the faces cannot be a face-flux quotient.
    support = assembler.mesh.support("nodes")
    ports = tuple(replace(port, measure_unit=ONE) if port.id == "y_min" else port for port in support.ports)
    mesh = Mesh((replace(support, ports=ports),))
    with pytest.raises(ContractError, match="face_unit_port_measures"):
        read(normal, transverse, binding=Assembler(assembler.layout, mesh, assembler.fields, terms))


@dataclass(frozen=True)
class Toy:
    """A plain or chain face Term on a 1D chain, for the dimension-free checks."""

    id: str
    kind: str
    value: Callable[[EvaluationState], TermValue] = field(compare=False)
    unit: Unit = FLUX
    divergence: str = "scatter"
    support: str = "nodes"
    equations: tuple[str, ...] = ("c",)
    dependencies: tuple[str, ...] = ()
    port: str | None = None
    identity: str = "test.toy"

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        return {"c": self.value(state)}


def chain(divergence: str) -> Assembler:
    from solarlab.mesh.geometry import integer_endpoint_geometry

    x = np.array([0.0, 0.3, 0.45, 1.0])
    contract = integer_endpoint_geometry(x).to_contract_geometry(cell_support="nodes", face_support="faces")
    ports = (Port("left", (0,), -1, np.array([1.0]), ONE), Port("right", (3,), 1, np.array([1.0]), ONE))
    layout = SystemLayout((NodeSet("nodes", 4),), (Block("c", "nodes", ONE, FLUX),),
                          (ConstraintRows("c", (0, 3), "c.ends", ONE),))
    terms = [Toy("c.flux", FACE_FLUX, lambda state: FaceValue(np.diff(state.u) ** 2 - 0.3), divergence=divergence),
             Toy("c.source", CELL_SOURCE, lambda state: LocalValue(1.0 + state.u), unit=SOURCE),
             Toy("c.ends", CONSTRAINT, lambda state: LocalValue(state.u[[0, 3]]), unit=ONE)]
    return Assembler(layout, Mesh((SupportGeometry("nodes", contract, ports),)), Coordinates(), terms)


def test_in_one_dimension_the_readout_closes_the_end_cells() -> None:
    """No transverse faces: ``f -+ c/w`` agrees with the balance quotient to rounding."""
    assembler = chain("scatter")
    u = np.array([0.2, -0.4, 0.9, 0.1])
    system = assembler.evaluate(u, OperatingInputs(), derivatives=False)
    for port, normal in (("left", [0]), ("right", [2])):
        ordered = assembler.reaction_flux_ordered(u, OperatingInputs(), "c", port, normal_faces=normal,
                                                  transverse_faces=[])
        np.testing.assert_allclose(ordered, assembler.reaction_flux(system, "c", port), rtol=1e-14, atol=0.0)
    with pytest.raises(ContractError, match="scatter_face_terms"):
        chain("chain").reaction_flux_ordered(u, OperatingInputs(), "c", "left", normal_faces=[0],
                                             transverse_faces=[])

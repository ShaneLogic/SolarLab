"""Classical ion-free 1D electronic DC as Terms on the shared Assembler.

The binding reuses the compiled device and its physics kernels unchanged:
the Poisson operator's displacement and capacitances, the charge ledger, the
stable SG electrochemical flux (with the optional smooth CT diffusivity), the
bound bulk recombination reaction plus an injected local law, the generation
profile, and the contact exchange kernel. Every Term writes the historical
expression of its value and tangent, so the assembled F, J, S and S_u equal
the existing :class:`~solarlab.equations.dc_1d.DC1DProblem` arrays bit for bit
(pinned by tests/unit/assembly).

:class:`AssembledDC1DProblem` presents the assembled system with the surface
the steady driver uses (``residual``, ``jacobian``, ``fluxes``, the current
observations), plus ``storage``, ``storage_tangent`` and ``storage_increment``
for storage-difference time steps, and the public observation kernels
``recombination_rate`` and ``contact_exchange`` that step accounting reads.
It is steady-only in the ``udot`` form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts.units import (
    AREA, COULOMB, ELEMENTARY_CHARGE_C, ONE, PARTICLE, SECOND, VOLT, VOLUME, Unit,
)
from solarlab.equations.assembly.assembler import (
    AssembledSystem, Assembler, Mesh, Port, SupportGeometry,
)
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.quasi_fermi import (
    ELECTRODE_QUASI_FERMI_1D_V1, ElectrodeQuasiFermiFields, quasi_fermi_potential_tangents,
)
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, PORT_FLUX, STORAGE, Contribution, EvaluationState,
    FaceValue, LocalValue, OperatingInputs, Term, TermValue,
)
from solarlab.equations.assembly.weighted import MEASURE
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.recombination.bulk import RecombinationRate, sum_recombination_rates
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes, carrier_particle_fluxes

__all__ = [
    "AssembledDC1DProblem", "BulkRecombination", "CarrierExchange", "CarrierGeneration",
    "CarrierReservoirs", "CarrierStorage", "CarrierTransport", "ClassicalDC1D",
    "GaussDisplacement", "PotentialContacts", "SpaceCharge", "assembled_problem_factory",
    "classical_dc_1d",
]

Q = ELEMENTARY_CHARGE_C
NODES = "nodes"
POTENTIAL, ELECTRON, HOLE = "potential", "electron", "hole"
CARRIERS = ((ELECTRON, "n", 1.0), (HOLE, "p", -1.0))
DISPLACEMENT_UNIT = COULOMB / AREA
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
DENSITY_UNIT = PARTICLE / VOLUME
TERM_SCHEMA = "solarlab.classical_dc_1d.terms.v1"

Vector: TypeAlias = NDArray[np.float64]


def _identity(*parts: object) -> str:
    return sha256(json.dumps((TERM_SCHEMA, *parts), separators=(",", ":")).encode()).hexdigest()


def _density(state: EvaluationState, block: str) -> Vector:
    return state.field("electron_density_m3" if block == ELECTRON else "hole_density_m3")


@dataclass(frozen=True)
class GaussDisplacement:
    """Displacement ``D = -C (phi_R - phi_L)`` [C/m2] on the Poisson channel's faces."""

    device: CompiledDevice1D
    id: str = "potential.displacement"
    kind: str = FACE_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = DISPLACEMENT_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        poisson = self.device.poisson
        flux = poisson.face_displacement(state.field("potential_V"))
        if not derivatives:
            return {POTENTIAL: FaceValue(flux)}
        capacitance = poisson.face_capacitance_F_m2
        return {POTENTIAL: FaceValue(flux, {POTENTIAL: capacitance}, {POTENTIAL: -capacitance})}


@dataclass(frozen=True)
class SpaceCharge:
    """Ledger space charge [C/m3]; quasi-Fermi tangents in the historical product order."""

    device: CompiledDevice1D
    id: str = "potential.space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        n, p = _density(state, ELECTRON), _density(state, HOLE)
        rho = self.device.charge_ledger(n, p).space_charge_C_m3()
        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}
        thermal = self.device.thermal_voltage_V
        return {POTENTIAL: LocalValue(rho, {
            POTENTIAL: -Q * MEASURE * (n + p) / thermal,
            ELECTRON: -Q * MEASURE * n,
            HOLE: -Q * MEASURE * p,
        })}


@dataclass(frozen=True)
class PotentialContacts:
    """Dirichlet rows ``phi - phi_contact`` [V] at both electrodes."""

    device: CompiledDevice1D
    id: str = "potential.contacts"
    kind: str = CONSTRAINT
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = VOLT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        phi = state.field("potential_V")
        left, right = self.device.contact_potentials(state.inputs.voltage_V)
        value = np.array([phi[0] - left, phi[-1] - right])
        return {POTENTIAL: LocalValue(value, {POTENTIAL: np.ones(2)} if derivatives else {})}


@dataclass(frozen=True)
class CarrierTransport:
    """SG particle fluxes [m-2 s-1] of both carriers from the quasi-Fermi steps."""

    device: CompiledDevice1D
    id: str = "carrier.transport"
    kind: str = FACE_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = FLUX_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def fluxes(self, state: EvaluationState) -> CarrierFaceFluxes:
        device = self.device
        phi = state.field("potential_V")
        electron_diffusivity = device.electron_diffusivity_faces_m2_s
        hole_diffusivity = device.hole_diffusivity_faces_m2_s
        if device.field_transport is not None:
            transport = device.field_transport.evaluate(phi)
            electron_diffusivity = transport.electron_diffusivity_m2_s
            hole_diffusivity = transport.hole_diffusivity_m2_s
        return carrier_particle_fluxes(
            device.nodes_m, phi, _density(state, ELECTRON), _density(state, HOLE),
            electron_diffusivity, hole_diffusivity, device.thermal_voltage_V,
            device.parameters.band_structure,
            electron_electrochemical_step=np.diff(state.block(ELECTRON)),
            hole_electrochemical_step=-np.diff(state.block(HOLE)),
        )

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        device = self.device
        fluxes = self.fluxes(state)
        if not derivatives:
            return {ELECTRON: FaceValue(fluxes.electron.flux), HOLE: FaceValue(fluxes.hole.flux)}
        thermal = device.thermal_voltage_V
        transport = None if device.field_transport is None else device.field_transport.evaluate(
            state.field("potential_V"))
        result: dict[str, TermValue] = {}
        for block, flux, sign in ((ELECTRON, fluxes.electron, 1.0), (HOLE, fluxes.hole, -1.0)):
            density = _density(state, block)
            drive = sign * np.diff(state.block(block))
            left_phi, right_phi = quasi_fermi_potential_tangents(flux, drive, sign, thermal)
            if transport is not None:
                diffusivity = (transport.electron_diffusivity_m2_s if block == ELECTRON
                               else transport.hole_diffusivity_m2_s)
                d_diffusivity = (transport.electron_d_diffusivity_d_field if block == ELECTRON
                                 else transport.hole_d_diffusivity_d_field)
                # SG is linear in positive D and E = -(phi_R - phi_L)/dx.
                field_chain = (flux.flux / diffusivity) * d_diffusivity / np.diff(device.nodes_m)
                left_phi = left_phi + field_chain
                right_phi = right_phi - field_chain
            left_z = sign * flux.left_density_derivative * density[:-1]
            right_z = sign * flux.right_density_derivative * density[1:]
            result[block] = FaceValue(flux.flux, {POTENTIAL: left_phi, block: left_z},
                                      {POTENTIAL: right_phi, block: right_z})
        return result


@dataclass(frozen=True)
class CarrierGeneration:
    """Optical pair generation ``g G`` [m-3 s-1]; ``g`` is the operating drive."""

    device: CompiledDevice1D
    id: str = "carrier.generation"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = ()
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.device.generation.binding_sha256)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        source = state.inputs.generation_scale * self.device.generation.source_m3_s
        return {ELECTRON: LocalValue(source), HOLE: LocalValue(source)}


@dataclass(frozen=True)
class BulkRecombination:
    """Net bulk recombination ``-R`` [m-3 s-1] on both carrier balances.

    The bound reaction plus the optional injected local law; the injected
    rate is added inside ``R`` so the pair source keeps one subtraction.
    """

    device: CompiledDevice1D
    id: str = "carrier.recombination"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        extra = self.device.additional_recombination
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1,
                         None if extra is None else extra.model_id)

    def rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        """Bound bulk reaction plus the injected local law, with exact partials."""
        evaluation = self.device.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        total = evaluation.total
        model = self.device.additional_recombination
        if model is not None:
            extra = model.evaluate(n, p)
            if not isinstance(extra, RecombinationRate) or not extra.has_derivatives:
                raise ValueError("a local recombination model must return rate and both exact partials")
            for value in (extra.rate, extra.d_rate_dn, extra.d_rate_dp):
                array = np.asarray(value)
                if array.shape != n.shape or not np.all(np.isfinite(array)):
                    raise ValueError("local recombination outputs must be finite on every node")
            total = sum_recombination_rates(total, extra)
        return total

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        n, p = _density(state, ELECTRON), _density(state, HOLE)
        rate = self.rate(n, p, derivatives=derivatives)
        loss = -np.asarray(rate.rate)
        if not derivatives:
            return {ELECTRON: LocalValue(loss), HOLE: LocalValue(loss)}
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / self.device.thermal_voltage_V
        tangents: dict[str, Contribution] = {POTENTIAL: source_phi, ELECTRON: source_n, HOLE: source_p}
        return {ELECTRON: LocalValue(loss, tangents), HOLE: LocalValue(loss, tangents)}


def _reference(device: CompiledDevice1D, block: str, voltage_V: float) -> Vector:
    references = device.carrier_reference_potentials(voltage_V)
    return references[0] if block == ELECTRON else references[1]


@dataclass(frozen=True)
class CarrierReservoirs:
    """Fixed-reservoir rows ``sign (shift + z)``: the carrier's log density excess."""

    device: CompiledDevice1D
    block: str
    nodes: tuple[int, ...]
    kind: str = CONSTRAINT
    support: str = NODES
    unit: Unit = ONE
    port: str | None = None

    @property
    def id(self) -> str:
        return self.block + ".reservoirs"

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, self.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, list(self.nodes), ELECTRODE_QUASI_FERMI_1D_V1)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        sign = 1.0 if self.block == ELECTRON else -1.0
        nodes: NDArray[np.intp] = np.asarray(self.nodes, dtype=np.intp)
        thermal = self.device.thermal_voltage_V
        reference = _reference(self.device, self.block, state.inputs.voltage_V)
        shift = (state.field("potential_V")[nodes] - reference[nodes]) / thermal
        value = sign * (shift + state.block(self.block)[nodes])
        tangents: dict[str, Contribution] = {}
        if derivatives:
            tangents = {POTENTIAL: np.full(nodes.size, sign / thermal), self.block: np.full(nodes.size, sign)}
        return {self.block: LocalValue(value, tangents)}


@dataclass(frozen=True)
class CarrierExchange:
    """Finite-rate or blocking exchange ``n_out S (c - c_eq)`` through one port."""

    device: CompiledDevice1D
    block: str
    channel: CarrierExchangeChannel
    node: int
    kind: str = PORT_FLUX
    support: str = NODES
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"{self.block}.exchange.{self.channel.side}"

    @property
    def port(self) -> str:
        return self.channel.side

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, self.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def exchange(self, potential: float, coordinate: float, voltage_V: float) -> tuple[float, float]:
        """``(n_out S (c - c_eq), n_out S)`` at the port node, from its own coordinates.

        The density excess is ``c_eq expm1(log(c/c_eq))``, never a difference
        of rounded densities.
        """
        thermal = self.device.thermal_voltage_V
        reference = _reference(self.device, self.block, voltage_V)
        shift = (potential - reference[self.node]) / thermal
        exponent = shift + coordinate if self.block == ELECTRON else -shift - coordinate
        channel = self.channel
        # The kernel supplies n_out * S, including exactly zero for blocking.
        slope = boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        return slope * excess, slope

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        sign = 1.0 if self.block == ELECTRON else -1.0
        thermal = self.device.thermal_voltage_V
        node = self.node
        flux, slope = self.exchange(state.block(POTENTIAL)[node], state.block(self.block)[node],
                                    state.inputs.voltage_V)
        value = np.array([flux])
        tangents: dict[str, Contribution] = {}
        if derivatives:
            density = _density(state, self.block)[node]
            tangents = {POTENTIAL: np.array([slope * sign * density / thermal]),
                        self.block: np.array([slope * sign * density])}
        return {self.block: LocalValue(value, tangents)}


@dataclass(frozen=True)
class CarrierStorage:
    """Physical carrier inventories ``S = w c`` and their cancellation-free increment."""

    device: CompiledDevice1D
    id: str = "carrier.storage"
    kind: str = STORAGE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        n, p = _density(state, ELECTRON), _density(state, HOLE)
        if not derivatives:
            return {ELECTRON: LocalValue(n), HOLE: LocalValue(p)}
        thermal = self.device.thermal_voltage_V
        return {
            ELECTRON: LocalValue(n, {POTENTIAL: MEASURE * n / thermal, ELECTRON: MEASURE * n}),
            HOLE: LocalValue(p, {POTENTIAL: -(MEASURE * p) / thermal, HOLE: -(MEASURE * p)}),
        }

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        """``S_old expm1(log(c_new/c_old))``: physical secant without subtraction."""
        device = self.device
        thermal = device.thermal_voltage_V
        delta = {block: state.block(block) - previous.block(block) for block in (POTENTIAL, ELECTRON, HOLE)}
        old = device.carrier_reference_energies(previous.inputs.voltage_V)
        new = device.carrier_reference_energies(state.inputs.voltage_V)
        log_n = (delta[POTENTIAL] + (new[0] - old[0])) / thermal + delta[ELECTRON]
        log_p = -(delta[POTENTIAL] + (new[1] - old[1])) / thermal - delta[HOLE]
        return {
            ELECTRON: _density(previous, ELECTRON) * MEASURE * np.expm1(log_n),
            HOLE: _density(previous, HOLE) * MEASURE * np.expm1(log_p),
        }


def _layout(device: CompiledDevice1D) -> SystemLayout:
    count = device.node_count
    constraints = [ConstraintRows(POTENTIAL, (0, count - 1), "potential.contacts", VOLT)]
    for block, carrier, _ in CARRIERS:
        nodes = tuple(node for side, node in (("left", 0), ("right", count - 1))
                      if device.channel(carrier, side).kind == "fixed_reservoir")
        if nodes:
            constraints.append(ConstraintRows(block, nodes, block + ".reservoirs", ONE))
    return SystemLayout(
        (NodeSet(NODES, count),),
        (Block(POTENTIAL, NODES, VOLT, DISPLACEMENT_UNIT),
         Block(ELECTRON, NODES, ONE, FLUX_UNIT),
         Block(HOLE, NODES, ONE, FLUX_UNIT)),
        tuple(constraints),
    )


def _mesh(device: CompiledDevice1D) -> Mesh:
    geometry = device.geometry
    if not np.array_equal(geometry.cell_measures_m[1:-1], device.poisson.interior_widths_m):
        raise CapabilityError("Gauss and carrier control volumes must agree on interior cells")
    contract = geometry.to_contract_geometry(cell_support=NODES, face_support=NODES + ".faces")
    # Per unit cross-section the port measure is the face area, unit ONE as
    # for the internal faces of the contract geometry.
    ports = tuple(Port(face.id, (face.node,), face.outward_normal, np.array([face.area_m2]), ONE)
                  for face in geometry.ports)
    return Mesh((SupportGeometry(NODES, contract, ports),))


@dataclass(frozen=True)
class ClassicalDC1D:
    """The assembled classical 1D system of one compiled device."""

    device: CompiledDevice1D
    assembler: Assembler = field(init=False, repr=False)
    transport: CarrierTransport = field(init=False, repr=False)
    recombination: BulkRecombination = field(init=False, repr=False)
    exchanges: Mapping[tuple[str, str], CarrierExchange] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        device = self.device
        layout = _layout(device)
        transport = CarrierTransport(device)
        recombination = BulkRecombination(device)
        terms: list[Term] = [
            GaussDisplacement(device), SpaceCharge(device), PotentialContacts(device),
            transport, CarrierGeneration(device), recombination, CarrierStorage(device),
        ]
        exchanges: dict[tuple[str, str], CarrierExchange] = {}
        for block, carrier, _ in CARRIERS:
            fixed = []
            for side, node in (("left", 0), ("right", device.node_count - 1)):
                channel = device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    fixed.append(node)
                else:
                    exchanges[(block, side)] = CarrierExchange(device, block, channel, node)
            if fixed:
                terms.append(CarrierReservoirs(device, block, tuple(fixed)))
        terms.extend(exchanges.values())
        object.__setattr__(self, "assembler", Assembler(
            layout, _mesh(device), ElectrodeQuasiFermiFields(device), terms))
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "recombination", recombination)
        object.__setattr__(self, "exchanges", exchanges)

    @property
    def layout(self) -> SystemLayout:
        return self.assembler.layout

    @property
    def identity(self) -> str:
        return self.assembler.identity


def classical_dc_1d(device: CompiledDevice1D) -> ClassicalDC1D:
    """Bind the classical electronic Terms of one compiled device."""
    if not isinstance(device, CompiledDevice1D):
        raise TypeError("classical_dc_1d requires a CompiledDevice1D")
    return ClassicalDC1D(device)


def assembled_problem_factory(
    device: CompiledDevice1D,
) -> Callable[[CompiledDevice1D, float, float], AssembledDC1DProblem]:
    """Constructor-compatible stand-in for ``DC1DProblem``: one binding, any operating point.

    The steady driver builds ``DC1DProblem(device, voltage_V, generation_scale)``
    by name; this factory has that signature and reuses one bound system.
    """
    system = classical_dc_1d(device)

    def factory(bound: CompiledDevice1D, voltage_V: float, generation_scale: float) -> AssembledDC1DProblem:
        return AssembledDC1DProblem(bound, voltage_V, generation_scale, system)

    return factory


@dataclass(frozen=True)
class AssembledDC1DProblem:
    """The assembled system with the DC1DProblem surface used by the steady driver."""

    device: CompiledDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    system: ClassicalDC1D | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.device.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")
        if self.system is None:
            object.__setattr__(self, "system", classical_dc_1d(self.device))
        elif self.system.device is not self.device:
            raise ValueError("the assembled system must be bound to this device")

    @property
    def binding(self) -> ClassicalDC1D:
        assert self.system is not None
        return self.system

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(self.voltage_V, self.generation_scale)

    def at(self, voltage_V: float, generation_scale: float) -> AssembledDC1DProblem:
        """Same bound system at another operating point (no rebinding)."""
        return AssembledDC1DProblem(self.device, voltage_V, generation_scale, self.system)

    def _steady_only(self, udot: ArrayLike | None) -> None:
        """Admit udot exactly as DC1DProblem does, then refuse any nonzero rate."""
        if udot is None:
            return
        derivative = np.asarray(udot, dtype=float)
        if derivative.shape != (3 * self.device.node_count,) or not np.all(np.isfinite(derivative)):
            raise ValueError("udot must be finite and match the coordinate vector")
        if np.any(derivative != 0.0):
            raise CapabilityError("the assembled system is time-stepped in storage-difference form, not by udot")

    def assemble(self, u: ArrayLike, *, derivatives: bool = True, storage: bool = False) -> AssembledSystem:
        return self.binding.assembler.evaluate(u, self.inputs, derivatives=derivatives, storage=storage)

    def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> Vector:
        self._steady_only(udot)
        return np.array(self.assemble(u, derivatives=False).F, copy=True)

    def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        self._steady_only(udot)
        system = self.assemble(u)
        assert system.J is not None
        if alpha == 0.0:
            return system.J
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        return csc_matrix(system.J + alpha * self.storage_jacobian(u))

    def storage(self, u: ArrayLike) -> Vector:
        return np.array(self.binding.assembler.storage(u, self.inputs, derivatives=False)[0], copy=True)

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        tangent = self.binding.assembler.storage(u, self.inputs)[1]
        assert tangent is not None
        return tangent

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        """Signed ``M = dR/dudot = -S_u`` (the historical name)."""
        return cast(csc_matrix, -self.storage_tangent(u))

    def storage_increment(self, u: ArrayLike, old_problem: AssembledDC1DProblem, old_u: ArrayLike) -> Vector:
        if old_problem.device.identity != self.device.identity:
            raise ValueError("a storage increment needs a previous state of the same device")
        return self.binding.assembler.storage_increment(u, self.inputs, old_u, old_problem.inputs)

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("state outside the field map's admissible domain")
        return self.binding.transport.fluxes(state)

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        """Net bulk recombination (bound reaction plus injected law) of the bound Term."""
        return self.binding.recombination.rate(n, p, derivatives=derivatives)

    def contact_exchange(self, u: ArrayLike, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        """Exchange flux ``n_out S (c - c_eq)`` and slope ``n_out S`` of one bound port channel.

        Defined for the finite-rate and blocking channels the binding owns; a
        fixed-reservoir channel is a constraint row, not an exchange.
        """
        values = np.asarray(u, dtype=float)
        count = self.device.node_count
        if values.shape != (3 * count,):
            raise ValueError("u must match the coordinate vector")
        block = ELECTRON if channel.carrier == "n" else HOLE
        exchange = self.binding.exchanges.get((block, channel.side))
        if exchange is None or exchange.channel != channel or exchange.node != node:
            raise ValueError("no bound exchange channel at this port node")
        offset = count if block == ELECTRON else 2 * count
        return exchange.exchange(values[node], values[offset + node], self.voltage_V)

    def current_density(self, u: ArrayLike) -> Vector:
        fluxes = self.fluxes(u)
        result: Vector = fluxes.electron_current() + fluxes.hole_current()
        return result

    def contact_current_density(self, u: ArrayLike) -> Vector:
        """Left/right terminal currents [A/m2] from each channel's own port flux."""
        binding = self.binding
        system = self.assemble(u, derivatives=False)
        state = binding.assembler.state(u, self.inputs)
        assert state is not None
        currents = []
        for side in ("left", "right"):
            total = 0.0
            for block, carrier, _ in CARRIERS:
                charge = -Q if carrier == "n" else Q
                exchange = binding.exchanges.get((block, side))
                if exchange is None:
                    flux = float(binding.assembler.reaction_flux(system, block, side)[0])
                else:
                    value = exchange.evaluate(state, derivatives=False)[block]
                    assert isinstance(value, LocalValue)
                    flux = np.asarray(value.value, dtype=float)[0]
                total += charge * flux
            currents.append(total)
        return np.asarray(currents)

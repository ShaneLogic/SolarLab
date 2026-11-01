"""Classical connected 2D DC on the unified Layout/Term/Assembler.

The compiled physical coefficients, electrode quasi-Fermi fields, SG/CT
normal-face laws, complete charge ledger and bound bulk/grain rate remain
unchanged. Terms return constitutive arrays and exact coordinate tangents;
the Assembler alone applies half-cell areas, face lengths, divergences and
constraint ownership. Generation and the complete recombination rate form
one plain source run, ``(g G - R) area``. Face-major Jacobian addition and
symbolic measures preserve the historical floating-point product order.

Fixed-reservoir contact observations use ``reaction_flux_ordered``. Robin
and blocking channels keep their actual exchange law, even on failed rows.
All currents are per unit out-of-plane depth; width normalization gives
A/m2. This binding is DC only: S, S_u and M are absent, as in DC2DProblem.
There is no new lateral-contact, ionic, transient or AC model here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts.units import (
    AREA, COULOMB, ELEMENTARY_CHARGE_C, LENGTH, ONE, PARTICLE, SECOND, VOLT, VOLUME, Unit,
)
from solarlab.equations.assembly.assembler import AssembledSystem, Assembler
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.quasi_fermi import quasi_fermi_potential_tangents
from solarlab.equations.assembly.rectangular_2d import RectangularSupport2D
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, PORT_FLUX, Contribution, EvaluationState,
    FaceValue, FieldValues, LocalValue, OperatingInputs, Term, TermValue,
)
from solarlab.equations.assembly.weighted import MEASURE
from solarlab.equations.dc_2d import CompiledDevice2D
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import (
    CarrierFaceFlux, CarrierFaceFluxes, stable_sg_particle_flux,
)

__all__ = [
    "AssembledDC2DProblem", "BulkRecombination2D", "CarrierExchange2D", "CarrierGeneration2D",
    "CarrierReservoirs2D", "CarrierTransport2D", "ClassicalDC2D", "ElectrodeQuasiFermiFields2D",
    "GaussDisplacement2D", "PotentialContacts2D", "SpaceCharge2D", "assembled_problem_factory_2d",
    "classical_dc_2d",
]

Q = ELEMENTARY_CHARGE_C
NODES = "nodes"
POTENTIAL, ELECTRON, HOLE = "potential", "electron", "hole"
BLOCKS = (POTENTIAL, ELECTRON, HOLE)
CARRIERS = ((ELECTRON, "n", 1.0), (HOLE, "p", -1.0))
SIDES = (("left", "y_min"), ("right", "y_max"))
DISPLACEMENT_UNIT = COULOMB / AREA
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
COORDINATE_FORM = "electrode_quasi_fermi_2d_v1"
TERM_SCHEMA = "solarlab.classical_dc_2d.terms.v1"

Vector: TypeAlias = NDArray[np.float64]


def _identity(*parts: object) -> str:
    return sha256(json.dumps((TERM_SCHEMA, *parts), separators=(",", ":")).encode()).hexdigest()


def _density(state: EvaluationState, block: str) -> Vector:
    return state.field("electron_density_m3" if block == ELECTRON else "hole_density_m3")


def _reference(device: CompiledDevice2D, block: str, voltage_V: float) -> Vector:
    references = device.reference_potentials(voltage_V)
    return references[0] if block == ELECTRON else references[1]


@dataclass(frozen=True)
class ElectrodeQuasiFermiFields2D:
    """The compiled 2D device's own field map; no state or drive is cached."""

    device: CompiledDevice2D

    @property
    def identity(self) -> str:
        return _identity(COORDINATE_FORM, self.device.identity)

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        values = np.concatenate([layout.view(u, block) for block in BLOCKS])
        phi, n, p = self.device.fields(values, voltage_V=inputs.voltage_V)
        admissible = bool(np.all(np.isfinite(n)) and np.all(np.isfinite(p))
                          and not np.any(n <= 0.0) and not np.any(p <= 0.0))
        return FieldValues({"potential_V": phi, "electron_density_m3": n,
                            "hole_density_m3": p}, admissible)


@dataclass(frozen=True)
class GaussDisplacement2D:
    """Directed displacement on x faces then y faces, in C/m2."""

    device: CompiledDevice2D
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
        left, right = self.device.geometry.contract_geometry.pairs.T
        phi = state.field("potential_V")
        capacitance = self.device.face_capacitance_F_m2
        flux = -capacitance * (phi[right] - phi[left])
        return {POTENTIAL: FaceValue(flux, {POTENTIAL: capacitance} if derivatives else {},
                                    {POTENTIAL: -capacitance} if derivatives else {})}


@dataclass(frozen=True)
class SpaceCharge2D:
    """The complete compiled charge ledger, with its original tangent order."""

    device: CompiledDevice2D
    id: str = "potential.space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = BLOCKS
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, COORDINATE_FORM)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        n, p = _density(state, ELECTRON), _density(state, HOLE)
        rho = self.device.charge_ledger(n, p).space_charge_C_m3()
        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}
        return {POTENTIAL: LocalValue(rho, {
            POTENTIAL: -Q * MEASURE * (n + p) / self.device.thermal_voltage_V,
            ELECTRON: -Q * MEASURE * n,
            HOLE: -Q * MEASURE * p,
        })}


@dataclass(frozen=True)
class PotentialContacts2D:
    """Unweighted potential constraints on both full stack contact strips."""

    device: CompiledDevice2D
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
        g = self.device.geometry
        phi = state.field("potential_V")
        bottom, top = self.device.column.contact_potentials(state.inputs.voltage_V)
        value = np.r_[phi[:g.nx] - bottom, phi[-g.nx:] - top]
        return {POTENTIAL: LocalValue(value, {POTENTIAL: np.ones(2 * g.nx)} if derivatives else {})}


@dataclass(frozen=True)
class CarrierTransport2D:
    """The original SG/CT face law and its cancellation-safe coordinate chain."""

    device: CompiledDevice2D
    id: str = "carrier.transport"
    kind: str = FACE_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = BLOCKS
    unit: Unit = FLUX_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, COORDINATE_FORM)

    def fluxes(self, state: EvaluationState) -> CarrierFaceFluxes:
        d = self.device
        phi = state.field("potential_V")
        n, p = _density(state, ELECTRON), _density(state, HOLE)
        left, right = d.geometry.contract_geometry.pairs.T
        potentials = d.band_structure.transport_potentials(phi)
        electron_step = state.block(ELECTRON)[right] - state.block(ELECTRON)[left]
        hole_step = -(state.block(HOLE)[right] - state.block(HOLE)[left])
        if d.field_transport is not None:
            return d.field_transport.fluxes(phi, n, p, potentials[0], potentials[1],
                electron_electrochemical_step=electron_step, hole_electrochemical_step=hole_step)
        fluxes = []
        for density, potential, diffusivity, block, sign in (
            (n, potentials[0], d.electron_diffusivity_faces_m2_s, ELECTRON, 1.0),
            (p, potentials[1], d.hole_diffusivity_faces_m2_s, HOLE, -1.0),
        ):
            step = -sign * (potential[right] - potential[left]) / d.thermal_voltage_V
            coefficient = diffusivity / d.geometry.face_spacing_m
            drive = sign * (state.block(block)[right] - state.block(block)[left])
            evaluated = stable_sg_particle_flux(density[left], density[right], step, coefficient,
                                                 electrochemical_step=drive)
            per_volt = -sign / d.thermal_voltage_V * evaluated.energy_step_derivative
            fluxes.append(CarrierFaceFlux(evaluated.flux, evaluated.left_density_derivative,
                evaluated.right_density_derivative, -per_volt, per_volt, step, coefficient))
        return CarrierFaceFluxes(fluxes[0], fluxes[1])

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        d = self.device
        fluxes = self.fluxes(state)
        if not derivatives:
            return {ELECTRON: FaceValue(fluxes.electron.flux), HOLE: FaceValue(fluxes.hole.flux)}
        left, right = d.geometry.contract_geometry.pairs.T
        thermal = d.thermal_voltage_V
        transport = None if d.field_transport is None else d.field_transport.evaluate(state.field("potential_V"))
        result: dict[str, TermValue] = {}
        for block, flux, sign in ((ELECTRON, fluxes.electron, 1.0), (HOLE, fluxes.hole, -1.0)):
            density = _density(state, block)
            drive = sign * (state.block(block)[right] - state.block(block)[left])
            phi_left, phi_right = quasi_fermi_potential_tangents(flux, drive, sign, thermal)
            if transport is not None:
                diffusivity = (transport.electron_diffusivity_m2_s if block == ELECTRON
                               else transport.hole_diffusivity_m2_s)
                slope = (transport.electron_d_diffusivity_d_field if block == ELECTRON
                         else transport.hole_d_diffusivity_d_field)
                field_chain = (flux.flux / diffusivity) * slope / d.geometry.face_spacing_m
                phi_left = phi_left + field_chain
                phi_right = phi_right - field_chain
            z_left = sign * flux.left_density_derivative * density[left]
            z_right = sign * flux.right_density_derivative * density[right]
            result[block] = FaceValue(flux.flux, {POTENTIAL: phi_left, block: z_left},
                                      {POTENTIAL: phi_right, block: z_right})
        return result


@dataclass(frozen=True)
class CarrierGeneration2D:
    """The compiled fixed or optical source on the actual 2D half cells."""

    device: CompiledDevice2D
    id: str = "carrier.generation"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = ()
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.device.generation_y.binding_sha256)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        source = state.inputs.generation_scale * self.device.generation_source_m3_s
        return {ELECTRON: LocalValue(source), HOLE: LocalValue(source)}


@dataclass(frozen=True)
class BulkRecombination2D:
    """The complete bound bulk/grain rate; grain mixes rates, never lifetimes."""

    device: CompiledDevice2D
    id: str = "carrier.recombination"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = BLOCKS
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, COORDINATE_FORM)

    def rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        evaluation = self.device.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        if self.device.grain_recombination is not None:
            return self.device.grain_recombination.evaluate(evaluation, self.device.recombination.coefficients)
        return evaluation.total

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


@dataclass(frozen=True)
class CarrierReservoirs2D:
    """Log-density excess constraints, owned only by fixed-reservoir channels."""

    device: CompiledDevice2D
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
        return _identity(self.id, self.device.identity, list(self.nodes), COORDINATE_FORM)

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
class CarrierExchange2D:
    """One Robin or blocking channel across a full stack port, along +y."""

    device: CompiledDevice2D
    block: str
    channel: CarrierExchangeChannel
    nodes: tuple[int, ...]
    kind: str = PORT_FLUX
    support: str = NODES
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"{self.block}.exchange.{self.channel.side}"

    @property
    def port(self) -> str:
        return "y_min" if self.channel.side == "left" else "y_max"

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, self.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, list(self.nodes), COORDINATE_FORM)

    def exchange(self, state: EvaluationState) -> tuple[Vector, float]:
        nodes: NDArray[np.intp] = np.asarray(self.nodes, dtype=np.intp)
        reference = _reference(self.device, self.block, state.inputs.voltage_V)
        shift = (state.block(POTENTIAL)[nodes] - reference[nodes]) / self.device.thermal_voltage_V
        exponent = shift + state.block(ELECTRON)[nodes] if self.block == ELECTRON else -shift - state.block(HOLE)[nodes]
        channel = self.channel
        slope = boundary_particle_flux(channel, channel.reservoir_density_m3).density_derivative
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        return slope * excess, slope

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        flux, slope = self.exchange(state)
        tangents: dict[str, Contribution] = {}
        if derivatives:
            sign = 1.0 if self.block == ELECTRON else -1.0
            density = _density(state, self.block)[np.asarray(self.nodes, dtype=np.intp)]
            tangents = {POTENTIAL: slope * sign * density / self.device.thermal_voltage_V * MEASURE,
                        self.block: slope * sign * density * MEASURE}
        return {self.block: LocalValue(flux, tangents)}


def _layout(device: CompiledDevice2D) -> SystemLayout:
    count, g = device.node_count, device.geometry
    constraints = [ConstraintRows(POTENTIAL, tuple(int(node) for node in g.contact_nodes),
                                  "potential.contacts", VOLT)]
    for block, carrier, _ in CARRIERS:
        fixed = tuple(int(node) for (side, _), nodes in zip(SIDES, g.contact_nodes.reshape(2, g.nx))
                      if device.column.channel(carrier, side).kind == "fixed_reservoir" for node in nodes)
        if fixed:
            constraints.append(ConstraintRows(block, fixed, block + ".reservoirs", ONE))
    return SystemLayout((NodeSet(NODES, count),), (
        Block(POTENTIAL, NODES, VOLT, COULOMB / LENGTH),
        Block(ELECTRON, NODES, ONE, PARTICLE / (LENGTH * SECOND)),
        Block(HOLE, NODES, ONE, PARTICLE / (LENGTH * SECOND)),
    ), tuple(constraints))


@dataclass(frozen=True)
class ClassicalDC2D:
    """One connected 2D Term binding; operating points reuse this declaration."""

    device: CompiledDevice2D
    assembler: Assembler = field(init=False, repr=False)
    geometry: RectangularSupport2D = field(init=False, repr=False)
    transport: CarrierTransport2D = field(init=False, repr=False)
    recombination: BulkRecombination2D = field(init=False, repr=False)
    exchanges: Mapping[tuple[str, str], CarrierExchange2D] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        device = self.device
        if not isinstance(device, CompiledDevice2D):
            raise TypeError("classical_dc_2d requires a CompiledDevice2D")
        geometry = RectangularSupport2D(device.geometry)
        transport = CarrierTransport2D(device)
        recombination = BulkRecombination2D(device)
        terms: list[Term] = [GaussDisplacement2D(device), SpaceCharge2D(device), PotentialContacts2D(device),
                             transport, CarrierGeneration2D(device), recombination]
        exchanges: dict[tuple[str, str], CarrierExchange2D] = {}
        for block, carrier, _ in CARRIERS:
            fixed: list[int] = []
            for (side, port), nodes in zip(SIDES, device.geometry.contact_nodes.reshape(2, device.geometry.nx)):
                channel = device.column.channel(carrier, side)
                indices = tuple(int(node) for node in nodes)
                if channel.kind == "fixed_reservoir":
                    fixed.extend(indices)
                else:
                    exchanges[(block, port)] = CarrierExchange2D(device, block, channel, indices)
            if fixed:
                terms.append(CarrierReservoirs2D(device, block, tuple(fixed)))
        terms.extend(exchanges.values())
        object.__setattr__(self, "assembler", Assembler(_layout(device), geometry.mesh,
                                                       ElectrodeQuasiFermiFields2D(device), terms))
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "recombination", recombination)
        object.__setattr__(self, "exchanges", MappingProxyType(exchanges))

    @property
    def layout(self) -> SystemLayout:
        return self.assembler.layout

    @property
    def identity(self) -> str:
        return self.assembler.identity


def classical_dc_2d(device: CompiledDevice2D) -> ClassicalDC2D:
    """Bind the existing supported 2D DC physics without changing the assembler API."""
    return ClassicalDC2D(device)


def assembled_problem_factory_2d(device: CompiledDevice2D) -> Callable[[CompiledDevice2D, float, float], AssembledDC2DProblem]:
    """A constructor-compatible replacement for the steady driver's DC2DProblem."""
    system = classical_dc_2d(device)

    def factory(bound: CompiledDevice2D, voltage_V: float, generation_scale: float) -> AssembledDC2DProblem:
        return AssembledDC2DProblem(bound, voltage_V, generation_scale, system)

    return factory


@dataclass(frozen=True)
class AssembledDC2DProblem:
    """The assembled F/J and unchanged physical observations used by steady_2d."""

    device: CompiledDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    system: ClassicalDC2D | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledDevice2D):
            raise TypeError("AssembledDC2DProblem requires a CompiledDevice2D")
        self.device.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")
        if self.system is None:
            object.__setattr__(self, "system", classical_dc_2d(self.device))
        elif self.system.device is not self.device:
            raise ValueError("the assembled system must be bound to this device")

    @property
    def binding(self) -> ClassicalDC2D:
        assert self.system is not None
        return self.system

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(self.voltage_V, self.generation_scale)

    def at(self, voltage_V: float, generation_scale: float) -> AssembledDC2DProblem:
        return AssembledDC2DProblem(self.device, voltage_V, generation_scale, self.system)

    def assemble(self, u: ArrayLike, *, derivatives: bool = True) -> AssembledSystem:
        """DC only: the result's S, S_u and M stay absent (no storage request)."""
        return self.binding.assembler.evaluate(u, self.inputs, derivatives=derivatives)

    def residual(self, u: ArrayLike) -> Vector:
        return np.array(self.assemble(u, derivatives=False).F, copy=True)

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        system = self.assemble(u)
        assert system.J is not None
        return system.J

    def _state(self, u: ArrayLike) -> EvaluationState:
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("state outside the field map's admissible domain")
        return state

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        return self.binding.transport.fluxes(self._state(u))

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        return self.binding.recombination.rate(n, p, derivatives=derivatives)

    def displacement(self, u: ArrayLike) -> Vector:
        state = self._state(u)
        value = GaussDisplacement2D(self.device).evaluate(state, derivatives=False)[POTENTIAL]
        assert isinstance(value, FaceValue)
        return np.asarray(value.flux, dtype=float)

    def current_density(self, u: ArrayLike) -> Vector:
        fluxes = self.fluxes(u)
        return fluxes.electron_current() + fluxes.hole_current()

    def cross_section_current_A_m(self, u: ArrayLike) -> Vector:
        g = self.device.geometry
        current_y = self.current_density(u)[g.x_face_count:].reshape(g.ny - 1, g.nx)
        return np.sum(current_y * g.x_widths_m[None, :], axis=1)

    def contact_particle_fluxes(self, u: ArrayLike) -> tuple[Vector, Vector]:
        binding = self.binding
        state = self._state(u)
        result = []
        for block, _, _ in CARRIERS:
            fluxes = []
            for _, port in SIDES:
                exchange = binding.exchanges.get((block, port))
                if exchange is not None:
                    fluxes.append(exchange.exchange(state)[0])
                else:
                    fluxes.append(binding.assembler.reaction_flux_ordered(
                        u, self.inputs, block, port, normal_faces=binding.geometry.normal_faces(port),
                        transverse_faces=binding.geometry.transverse_faces))
            result.append(np.stack(fluxes))
        return result[0], result[1]

    def terminal_current_A_m(self, u: ArrayLike) -> Vector:
        electron, hole = self.contact_particle_fluxes(u)
        return np.sum((-Q * electron + Q * hole) * self.device.geometry.x_widths_m[None, :], axis=1)

    def terminal_current_density_A_m2(self, u: ArrayLike) -> Vector:
        return self.terminal_current_A_m(u) / self.device.geometry.lateral_length_m

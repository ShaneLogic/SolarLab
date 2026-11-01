"""Finite-site ionic 1D (single and per-domain) as Terms on the shared Assembler.

Each ion domain is an embedded :class:`NodeSet` (``ion:<layer>``) with the
compiled half-endpoint ion geometry. Every species is one block
(``ion:<layer>:<species>``) after ``potential, electron, hole``, in the
compiled coordinate order. The electronic Terms are the classical ones bound
to the device's electronic component; only the space charge is replaced.
The binding reuses the compiled devices and the ion kernels unchanged:
``fields_by_species``/``fields_by_domain``, the full ``charge_ledger``,
``ion_face_flux``/``ion_face_flux_jacobian``, and the finite-site and
shared-site coordinate maps, Jacobians and secants.

Historical evaluation orders reproduced (review R03):

* Gauss rows: the displacement scatter, then ``rho * w`` with ``rho`` from
  ONE fold of the full charge ledger (electronic and every ion entry). Ion
  charge tangents are ``Q * w_e * (W_i/W_e) * dP/dy`` in that product order;
  the Mesh's embedding ratio ``W_i/W_e`` must equal the compiled deposition
  weights ``ion_charge_weight_ratio`` bit for bit.
* Ion rows: ``-diff(r_[0, flux, 0])``, assigned (``"chain"`` divergence).
  Face-major Jacobian with ``dF/dP * dP/dy`` per node; shared sites sum
  ``dF/dP * m_sc + dF/dP_other * m_oc`` per coordinate column first.
* Steady closure: single-domain inventory rows (K1) are
  ``fsum(W P) - fsum(W P0)``, per-domain inventory rows (KM) are
  ``fsum(W (P - P0))`` and immobile profile rows (Z) are ``W (P - P0)`` per
  cell. Their gradients are the physical storage rows ``W dP/dy``.

The adapters present the original class surfaces (residual, jacobian,
storage_jacobian, ion fluxes and currents, constrained-row bookkeeping) to
the unchanged drivers, steppers and AC operators, and refuse exactly the
inputs and nonfinite evaluations the original classes refuse. Like the
electronic facade, they are steady-only in the ``ydot`` form (decision Q2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts.units import AREA, COULOMB, ELEMENTARY_CHARGE_C, ONE, PARTICLE, SECOND, VOLUME, Unit
from solarlab.equations.assembly.assembler import AssembledSystem, Assembler, Mesh, SupportGeometry
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, ClassicalDC1D, classical_dc_1d
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.quasi_fermi import ELECTRODE_QUASI_FERMI_1D_V1
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CHAIN_DIVERGENCE, CONSTRAINT, FACE_FLUX, STORAGE, Contribution, EvaluationState,
    FaceValue, FieldValues, LocalValue, OperatingInputs, ReductionValue, Term, TermValue,
)
from solarlab.equations.assembly.weighted import MEASURE
from solarlab.equations.dc_1d import CapabilityError, CompiledDevice1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.equations.ionic_closure import declared_immobile_profiles, ion_mobility_partition
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D
from solarlab.physics.electrostatics.charge import ChargeLedger
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1, ion_face_flux, ion_face_flux_jacobian
from solarlab.physics.ions.shared_site import shared_site_coordinate_jacobian_m3, shared_site_secant_m3
from solarlab.solvers.time.ionic_backward_euler import finite_site_secant_m3

__all__ = [
    "AssembledConstrainedIonic1DProblem", "AssembledConstrainedMultipleIonic1DProblem",
    "AssembledIonic1DProblem", "AssembledMultipleIonic1DProblem", "ImmobileProfile",
    "InventoryReduction", "IonDomain", "IonStorage", "IonTransport", "IonicFields",
    "IonicSpaceCharge", "IonicSystem", "ionic_system", "multiple_ionic_system",
]

Q = ELEMENTARY_CHARGE_C
NODES = "nodes"
POTENTIAL, ELECTRON, HOLE = "potential", "electron", "hole"
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
DENSITY_UNIT = PARTICLE / VOLUME
INVENTORY_UNIT = PARTICLE / AREA
TERM_SCHEMA = "solarlab.ionic_1d.terms.v1"
SHARED_SITES = "shared_finite_sites"
K1 = "inventory_minus_reference_v1"
KM = "inventory_excess_v1"
RESIDUAL_ENVELOPE = "ionic residual is outside the finite numerical envelope"
MULTIPLE_ENVELOPE = "multiple-domain residual is outside the finite envelope"
JACOBIAN_ENVELOPE = "ionic Jacobian is outside the finite numerical envelope"

Vector: TypeAlias = NDArray[np.float64]
IonicDevice: TypeAlias = CompiledIonicDevice1D | CompiledMultipleIonicDevice1D


def _identity(*parts: object) -> str:
    return sha256(json.dumps((TERM_SCHEMA, *parts), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class IonDomain:
    """One closed ion domain: its compiled component and its layout names."""

    device: CompiledIonicDevice1D

    @property
    def name(self) -> str:
        return self.device.domain_layer_id

    @property
    def node_set(self) -> str:
        return "ion:" + self.name

    @property
    def species(self) -> tuple[str, ...]:
        return self.device.ion_species_names

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(f"{self.node_set}:{species}" for species in self.species)

    @property
    def shared(self) -> bool:
        return self.device.ion_site_contract == SHARED_SITES

    @property
    def mass_field(self) -> str:
        """Field name of the node-local ``dP/dy`` [node, species, coordinate]."""
        return "ion_coordinate_jacobian_m3:" + self.name

    def block(self, species: str) -> str:
        return self.blocks[self.species.index(species)]

    def capacity(self, species: str) -> Vector:
        result: Vector = getattr(self.device, species + "_ion_site_limit_m3")
        return result

    def reference(self, species: str) -> Vector:
        result: Vector = getattr(self.device, species + "_ion_reference_m3")
        return result

    def coordinate_jacobian(self, populations: tuple[Vector, ...]) -> Vector:
        """``dP/dy`` per node, in the compiled device's arithmetic."""
        device = self.device
        if self.shared:
            shared: Vector = shared_site_coordinate_jacobian_m3(
                populations[0], populations[1], device.positive_ion_site_limit_m3)
            return shared
        result = np.zeros((device.ion_node_count, device.ion_species_count, device.ion_species_count))
        for index, (species, density) in enumerate(zip(self.species, populations)):
            result[:, index, index] = density * (1.0 - density / self.capacity(species))
        return result


def _domains(device: IonicDevice) -> tuple[IonDomain, ...]:
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return tuple(IonDomain(domain) for domain in device.domains)
    return (IonDomain(device),)


def _populations(state: EvaluationState, domain: IonDomain) -> tuple[Vector, ...]:
    return tuple(state.field(block) for block in domain.blocks)


@dataclass(frozen=True)
class IonicFields:
    """Field map of the full ionic coordinates; inadmissible trials raise as compiled."""

    device: IonicDevice

    @property
    def identity(self) -> str:
        return _identity("finite_site_ionic_fields_v1", self.device.identity)

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        device = self.device
        populations: dict[str, tuple[Vector, ...]]
        if isinstance(device, CompiledMultipleIonicDevice1D):
            phi, n, p, populations = device.fields_by_domain(u, voltage_V=inputs.voltage_V)
        else:
            phi, n, p, single = device.fields_by_species(u, voltage_V=inputs.voltage_V)
            populations = {device.domain_layer_id: single}
        fields: dict[str, Vector] = {"potential_V": phi, "electron_density_m3": n, "hole_density_m3": p}
        for domain in _domains(device):
            values = populations[domain.name]
            fields.update(zip(domain.blocks, values))
            fields[domain.mass_field] = domain.coordinate_jacobian(values)
        return FieldValues(fields, True)


@dataclass(frozen=True)
class IonicSpaceCharge:
    """Gauss charge from ONE fold of the full ledger; ion tangents deposited on Gauss rows."""

    device: IonicDevice
    id: str = "potential.space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        ions = tuple(block for domain in _domains(self.device) for block in domain.blocks)
        return (POTENTIAL, ELECTRON, HOLE) + ions

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1, "full_ledger_single_fold")

    def ledger(self, state: EvaluationState) -> ChargeLedger:
        """The compiled full charge ledger of one state."""
        device = self.device
        n, p = state.field("electron_density_m3"), state.field("hole_density_m3")
        if isinstance(device, CompiledMultipleIonicDevice1D):
            return device.charge_ledger(n, p, {domain.name: _populations(state, domain)
                                               for domain in _domains(device)})
        values = _populations(state, IonDomain(device))
        return device.charge_ledger(n, p, values[0], values[1] if len(values) == 2 else None)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        rho = self.ledger(state).space_charge_C_m3()
        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}
        n, p = state.field("electron_density_m3"), state.field("hole_density_m3")
        thermal = self.device.electronic_component.thermal_voltage_V
        tangents: dict[str, Contribution] = {
            POTENTIAL: -Q * MEASURE * (n + p) / thermal,
            ELECTRON: -Q * MEASURE * n,
            HOLE: -Q * MEASURE * p,
        }
        for domain in _domains(self.device):
            nodes = domain.device.ion_node_indices
            ratio = np.zeros(n.size)
            ratio[nodes] = state.embedding_ratio(domain.node_set)
            mass = state.field(domain.mass_field)
            for column, block in enumerate(domain.blocks):
                partial = np.zeros(n.size)
                if domain.shared:
                    # Net charge derivative of the shared sites: dP+/dy - dP-/dy.
                    partial[nodes] = mass[:, 0, column] - mass[:, 1, column]
                    tangents[block] = Q * MEASURE * ratio * partial
                    continue
                partial[nodes] = mass[:, column, column]
                charge = Q * MEASURE * ratio * partial
                tangents[block] = charge if domain.species[column] == "positive" else -charge
        return {POTENTIAL: LocalValue(rho, tangents)}


@dataclass(frozen=True)
class IonTransport:
    """Blocking-domain particle flux of one species, in the historical chain order."""

    domain: IonDomain
    species: str
    kind: str = FACE_FLUX
    unit: Unit = FLUX_UNIT
    port: str | None = None
    divergence: str = CHAIN_DIVERGENCE

    @property
    def block(self) -> str:
        return self.domain.block(self.species)

    @property
    def id(self) -> str:
        return self.block + ".transport"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL,) + (self.domain.blocks if self.domain.shared else (self.block,))

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity, self.domain.device.steric_model)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        domain, device = self.domain, self.domain.device
        index = domain.species.index(self.species)
        populations = _populations(state, domain)
        density = populations[index]
        partner = populations[1 - index] if domain.shared else None
        phi = state.field("potential_V")[device.ion_node_indices]
        spacing = np.diff(device.ion_geometry.nodes_m)
        diffusivity = getattr(device, self.species + "_ion_diffusivity_faces_m2_s")
        thermal = device.electronic_component.thermal_voltage_V
        capacity = domain.capacity(self.species)
        steric = device.steric_model == DIFFUSION_ONLY_STERIC_V1
        sign = 1.0 if self.species == "positive" else -1.0
        flux = ion_face_flux(phi, density, spacing, diffusivity, thermal, capacity[:-1],
                             steric_diffusion_only=steric, P_lim_node=capacity,
                             P_other_node=partner, drift_sign=sign)
        if not derivatives:
            return {self.block: FaceValue(flux)}
        if domain.shared:
            # The historical shared-site Jacobian binds the common site capacity.
            limit = device.positive_ion_site_limit_m3
            tangent = ion_face_flux_jacobian(phi, density, spacing, diffusivity, thermal, limit[:-1],
                                             steric_diffusion_only=True, P_lim_node=limit,
                                             P_other_node=partner, drift_sign=sign)
        else:
            tangent = ion_face_flux_jacobian(phi, density, spacing, diffusivity, thermal, capacity[:-1],
                                             steric_diffusion_only=steric, P_lim_node=capacity,
                                             drift_sign=sign)
        if not np.all(tangent.differentiable_faces):
            raise ValueError("ionic Jacobian is undefined at a retained historical regularization kink")
        mass = state.field(domain.mass_field)
        left: dict[str, Contribution] = {POTENTIAL: tangent.potential_left_derivative}
        right: dict[str, Contribution] = {POTENTIAL: tangent.potential_right_derivative}
        if domain.shared:
            other = 1 - index
            for column, block in enumerate(domain.blocks):
                left[block] = (tangent.density_left_derivative * mass[:-1, index, column]
                               + tangent.partner_left_derivative * mass[:-1, other, column])
                right[block] = (tangent.density_right_derivative * mass[1:, index, column]
                                + tangent.partner_right_derivative * mass[1:, other, column])
        else:
            partial = mass[:, index, index]
            left[self.block] = tangent.density_left_derivative * partial[:-1]
            right[self.block] = tangent.density_right_derivative * partial[1:]
        return {self.block: FaceValue(flux, left, right)}


def _storage_tangents(domain: IonDomain, mass: Vector, index: int) -> dict[str, Contribution]:
    """``d(W P_index)/dy`` over every coordinate of the domain: the storage rows."""
    if domain.shared:
        return {block: MEASURE * mass[:, index, column] for column, block in enumerate(domain.blocks)}
    return {domain.blocks[index]: MEASURE * mass[:, index, index]}


@dataclass(frozen=True)
class IonStorage:
    """Physical ion inventories ``S = W P`` and the cancellation-free site secants."""

    domain: IonDomain
    kind: str = STORAGE
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def id(self) -> str:
        return self.domain.node_set + ".storage"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return self.domain.blocks

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.domain.blocks

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        populations = _populations(state, self.domain)
        if not derivatives:
            return {block: LocalValue(density) for block, density in zip(self.domain.blocks, populations)}
        mass = state.field(self.domain.mass_field)
        return {block: LocalValue(density, _storage_tangents(self.domain, mass, index))
                for index, (block, density) in enumerate(zip(self.domain.blocks, populations))}

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        """Physical site-law secants ``P_new - P_old`` from the coordinate differences."""
        domain, device = self.domain, self.domain.device
        if domain.shared:
            positive, negative = domain.blocks
            changes = shared_site_secant_m3(
                device.positive_ion_site_limit_m3,
                device.positive_ion_logit_reference, device.negative_ion_logit_reference,
                previous.block(positive), previous.block(negative), state.block(positive), state.block(negative),
            )
            return dict(zip(domain.blocks, changes))
        return {block: finite_site_secant_m3(
            domain.capacity(species), getattr(device, species + "_ion_logit_reference"),
            previous.block(block), state.block(block),
        ) for species, block in zip(domain.species, domain.blocks)}


@dataclass(frozen=True)
class InventoryReduction:
    """One mobile species' closed-domain inventory row and its whole-domain gradient."""

    domain: IonDomain
    species: str
    form: str
    kind: str = CONSTRAINT
    unit: Unit = DENSITY_UNIT
    port: str | None = None
    measure: str = "cell"

    @property
    def block(self) -> str:
        return self.domain.block(self.species)

    @property
    def id(self) -> str:
        return self.block + ".inventory"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.domain.blocks if self.domain.shared else (self.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity, self.form)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        index = self.domain.species.index(self.species)
        density, reference = state.field(self.block), self.domain.reference(self.species)
        if self.form == K1:
            summand, offset = MEASURE * density, MEASURE * reference
        elif self.form == KM:
            summand, offset = MEASURE * (density - reference), None
        else:
            raise ValueError("unknown inventory evaluation order")
        tangents = _storage_tangents(self.domain, state.field(self.domain.mass_field), index) if derivatives else {}
        return {self.block: ReductionValue(summand, offset, tangents)}


@dataclass(frozen=True)
class ImmobileProfile:
    """Declared per-cell rows ``W (P - P0)`` of one exactly zero-diffusivity species."""

    domain: IonDomain
    species: str
    kind: str = CONSTRAINT
    unit: Unit = DENSITY_UNIT
    port: str | None = None
    measure: str = "cell"

    @property
    def block(self) -> str:
        return self.domain.block(self.species)

    @property
    def id(self) -> str:
        return self.block + ".profile"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.domain.blocks if self.domain.shared else (self.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        index = self.domain.species.index(self.species)
        profile = declared_immobile_profiles(self.domain.device)[self.species]
        value = MEASURE * (state.field(self.block) - profile)
        tangents = _storage_tangents(self.domain, state.field(self.domain.mass_field), index) if derivatives else {}
        return {self.block: LocalValue(value, tangents)}


@dataclass(frozen=True)
class IonicSystem:
    """The assembled ionic system of one device, optionally with its steady closure.

    With ``inventory_row_node`` every mobile species' row at that node is its
    inventory (K1 single-domain, KM per domain) and every row of an immobile
    species is its declared profile; ``closure_rows`` marks those rows.
    """

    device: IonicDevice
    inventory_row_node: int | None = None
    assembler: Assembler = field(init=False, repr=False, compare=False)
    electronic: ClassicalDC1D = field(init=False, repr=False, compare=False)
    transports: Mapping[tuple[str, str], IonTransport] = field(init=False, repr=False, compare=False)
    closure_rows: NDArray[np.bool_] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        device = self.device
        component = device.electronic_component
        electronic = classical_dc_1d(component)
        domains = _domains(device)
        base = electronic.assembler.layout
        node_sets, blocks, constraints = list(base.node_sets), list(base.blocks), list(base.constraints)
        supports = list(electronic.assembler.mesh.supports)
        form = KM if isinstance(device, CompiledMultipleIonicDevice1D) else K1
        closure: list[ImmobileProfile | InventoryReduction] = []
        for domain in domains:
            ion = domain.device
            node_sets.append(NodeSet(domain.node_set, ion.ion_node_count, NODES,
                                     tuple(int(node) for node in ion.ion_node_indices)))
            blocks.extend(Block(block, domain.node_set, ONE, FLUX_UNIT) for block in domain.blocks)
            supports.append(SupportGeometry(domain.node_set, ion.ion_geometry.to_contract_geometry(
                cell_support=domain.node_set, face_support=domain.node_set + ".faces")))
            if self.inventory_row_node is None:
                continue
            immobile = ion_mobility_partition(ion)[1]
            for species in domain.species:
                term: ImmobileProfile | InventoryReduction
                if species in immobile:
                    term, rows = ImmobileProfile(domain, species), tuple(range(ion.ion_node_count))
                else:
                    term, rows = InventoryReduction(domain, species, form), (self.inventory_row_node,)
                constraints.append(ConstraintRows(term.block, rows, term.id, INVENTORY_UNIT))
                closure.append(term)
        layout = SystemLayout(tuple(node_sets), tuple(blocks), tuple(constraints))
        count = component.node_count
        compiled = ([device.ion_species_slices[d.name][s] for d in domains for s in d.species]
                    if isinstance(device, CompiledMultipleIonicDevice1D) else list(device.ion_species_slices))
        expected = [(0, count), (count, 2 * count), (2 * count, 3 * count)]
        expected += [(cast(int, part.start), cast(int, part.stop)) for part in compiled]
        if layout.size != device.state_size or list(layout.block_ranges().values()) != expected:
            raise CapabilityError("the ionic layout must reproduce the compiled coordinate order")
        transports = {(domain.name, species): IonTransport(domain, species)
                      for domain in domains for species in domain.species}
        terms: list[Term] = [IonicSpaceCharge(device) if term.id == "potential.space_charge" else term
                             for term in electronic.assembler.terms]
        terms += [*transports.values(), *(IonStorage(domain) for domain in domains), *closure]
        assembler = Assembler(layout, Mesh(tuple(supports)), IonicFields(device), terms)
        for domain in domains:
            ratio = assembler.embedding_ratio(domain.node_set)
            if ratio.tobytes() != domain.device.ion_charge_weight_ratio.tobytes():
                raise CapabilityError("the embedded charge ratio must equal the compiled deposition weights")
        closure_rows: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
        for term in closure:
            owned = layout.constraint_rows(term.id)
            closure_rows[layout.rows(owned.block, owned.nodes)] = True
        closure_rows.setflags(write=False)
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "electronic", electronic)
        object.__setattr__(self, "transports", transports)
        object.__setattr__(self, "closure_rows", closure_rows)

    @property
    def layout(self) -> SystemLayout:
        return self.assembler.layout

    @property
    def identity(self) -> str:
        return self.assembler.identity


def ionic_system(device: CompiledIonicDevice1D, *, inventory_row_node: int | None = None) -> IonicSystem:
    """The original single-domain rows, or with the K1 inventory / Z profile closure."""
    if not isinstance(device, CompiledIonicDevice1D):
        raise TypeError("ionic_system requires a CompiledIonicDevice1D")
    return IonicSystem(device, inventory_row_node)


def multiple_ionic_system(device: CompiledMultipleIonicDevice1D, *,
                          inventory_row_node: int | None = None) -> IonicSystem:
    """The original per-domain rows, or with the KM inventory / Z profile closure."""
    if not isinstance(device, CompiledMultipleIonicDevice1D):
        raise TypeError("multiple_ionic_system requires a CompiledMultipleIonicDevice1D")
    return IonicSystem(device, inventory_row_node)


def _check_drive(component: CompiledDevice1D, voltage_V: float, generation_scale: float) -> None:
    """The electronic problem's admission of the drive (as ``DC1DProblem``)."""
    component.contact_potentials(voltage_V)
    if not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("generation_scale must be finite and non-negative")


def _finite_scalar(value: float, name: str, *, nonnegative: bool = False) -> float:
    """The multiple-domain scalar admission (as ``multiple_ionic_1d``)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _admit_coordinates(device: IonicDevice, y: ArrayLike) -> None:
    """The multiple-domain device admits only real numeric arrays (single: any float cast)."""
    if isinstance(device, CompiledMultipleIonicDevice1D) and np.asarray(y).dtype.kind not in {"i", "u", "f"}:
        raise ValueError("multiple-domain coordinates and rates must be real numeric arrays")


def _admit_rate(device: IonicDevice, ydot: ArrayLike | None) -> None:
    """Admit ``ydot`` as the original class does, then refuse any nonzero rate (Q2)."""
    if ydot is None:
        return
    _admit_coordinates(device, ydot)
    derivative = np.asarray(ydot, dtype=float)
    if derivative.shape != (device.state_size,) or not np.all(np.isfinite(derivative)):
        raise ValueError("coordinates must contain 3N and all species of both ionic domains"
                         if isinstance(device, CompiledMultipleIonicDevice1D)
                         else "ydot must be finite and match the complete ionic coordinates")
    if np.any(derivative != 0.0):
        raise CapabilityError("the assembled system is time-stepped in storage-difference form, not by ydot")


def _residual(system: AssembledSystem, closure: NDArray[np.bool_], message: str) -> Vector:
    """F, refused exactly where the original-row residual is nonfinite.

    Closure-owned rows replace original balances; their ``balance`` is the
    original row value that the original classes check before replacement.
    """
    if not (np.all(np.isfinite(system.F[~closure])) and np.all(np.isfinite(system.balance[closure]))):
        raise FloatingPointError(message)
    return np.array(system.F, copy=True)


def _jacobian(system: AssembledSystem, closure: NDArray[np.bool_]) -> csc_matrix:
    """J, refused exactly where the original-row Jacobian has a nonfinite entry.

    The original classes check their full original Jacobian, then replace the
    closure rows without checking the replacement.
    """
    matrix = system.J
    assert matrix is not None and system.replaced_nonfinite_rows is not None
    original = ~closure[matrix.indices]
    if not np.all(np.isfinite(matrix.data[original])) or np.any(system.replaced_nonfinite_rows[closure]):
        raise FloatingPointError(JACOBIAN_ENVELOPE)
    return matrix


class _AssembledIonicRows:
    """Observation and storage surface shared by the original-row adapters."""

    device: IonicDevice
    voltage_V: float
    generation_scale: float
    system: IonicSystem | None

    @property
    def binding(self) -> IonicSystem:
        assert self.system is not None
        return self.system

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(self.voltage_V, self.generation_scale)

    @property
    def electronic_problem(self) -> AssembledDC1DProblem:
        electronic = self.binding.electronic
        return AssembledDC1DProblem(electronic.device, self.voltage_V, self.generation_scale, electronic)

    def assemble(self, y: ArrayLike, *, derivatives: bool = True, storage: bool = False) -> AssembledSystem:
        _admit_coordinates(self.device, y)
        return self.binding.assembler.evaluate(y, self.inputs, derivatives=derivatives, storage=storage)

    def _residual(self, y: ArrayLike, ydot: ArrayLike | None, message: str) -> Vector:
        _admit_coordinates(self.device, y)
        _admit_rate(self.device, ydot)
        return _residual(self.assemble(y, derivatives=False), self.binding.closure_rows, message)

    def _jacobian(self, y: ArrayLike, ydot: ArrayLike | None, alpha: float) -> csc_matrix:
        _admit_coordinates(self.device, y)
        _admit_rate(self.device, ydot)
        system = self.assemble(y)
        assert system.J is not None
        if alpha == 0.0:
            return _jacobian(system, self.binding.closure_rows)
        combined = csc_matrix(system.J + alpha * self.storage_jacobian(y))
        if not np.all(np.isfinite(combined.data)):
            raise FloatingPointError(JACOBIAN_ENVELOPE)
        return combined

    def storage(self, y: ArrayLike) -> Vector:
        """Equation storage S (zero on algebraic rows), particles/m^2."""
        _admit_coordinates(self.device, y)
        return np.array(self.binding.assembler.storage(y, self.inputs, derivatives=False)[0], copy=True)

    def physical_storage(self, y: ArrayLike) -> Vector:
        """The original readers' physical S: zero on the algebraic carrier rows."""
        return self.storage(y)

    def storage_tangent(self, y: ArrayLike) -> csc_matrix:
        _admit_coordinates(self.device, y)
        tangent = self.binding.assembler.storage(y, self.inputs)[1]
        assert tangent is not None
        return tangent

    def storage_jacobian(self, y: ArrayLike) -> csc_matrix:
        """Signed ``M = dR/dydot = -S_u`` (the historical name)."""
        return cast(csc_matrix, -self.storage_tangent(y))

    def storage_increment(self, y: ArrayLike, old_problem: _AssembledIonicRows, old_y: ArrayLike) -> Vector:
        """Cancellation-free ``S(y) - S(old_y)`` (carrier and site-law secants)."""
        if old_problem.device.identity != self.device.identity:
            raise ValueError("a storage increment needs a previous state of the same device")
        _admit_coordinates(self.device, y)
        _admit_coordinates(self.device, old_y)
        return self.binding.assembler.storage_increment(y, self.inputs, old_y, old_problem.inputs)

    def _domain_fluxes(self, y: ArrayLike, domain_id: str, species: str) -> Vector:
        transport = self.binding.transports.get((domain_id, species))
        if transport is None:
            raise ValueError("requested ion species is not assembled in this device")
        _admit_coordinates(self.device, y)
        state = self.binding.assembler.state(y, self.inputs)
        assert state is not None
        value = transport.evaluate(state, derivatives=False)[transport.block]
        assert isinstance(value, FaceValue)
        return np.asarray(value.flux, dtype=float)

    def _on_electronic_faces(self, y: ArrayLike, domain_id: str, species: str) -> Vector:
        flux = self._domain_fluxes(y, domain_id, species)
        domain = next(item for item in _domains(self.device) if item.name == domain_id)
        result = np.zeros(self.device.electronic_component.node_count - 1)
        result[domain.device.ion_node_indices[:-1]] = flux
        return result


@dataclass(frozen=True)
class AssembledIonic1DProblem(_AssembledIonicRows):
    """Assembled single-domain rows with the ``Ionic1DProblem`` surface."""

    device: CompiledIonicDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    system: IonicSystem | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledIonicDevice1D):
            raise TypeError("Ionic1DProblem requires the complete compiled ionic device")
        _check_drive(self.device.electronic_component, self.voltage_V, self.generation_scale)
        if self.system is None:
            object.__setattr__(self, "system", ionic_system(self.device))
        elif self.system.device is not self.device or self.system.inventory_row_node is not None:
            raise ValueError("the assembled system must be the original rows of this device")

    def at(self, voltage_V: float, generation_scale: float) -> AssembledIonic1DProblem:
        """Same bound system at another operating point (no rebinding)."""
        return AssembledIonic1DProblem(self.device, voltage_V, generation_scale, self.system)

    def residual(self, y: ArrayLike, ydot: ArrayLike | None = None) -> Vector:
        return self._residual(y, ydot, RESIDUAL_ENVELOPE)

    def jacobian(self, y: ArrayLike, ydot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        return self._jacobian(y, ydot, alpha)

    def ion_fluxes(self, y: ArrayLike, *, species: str = "positive") -> Vector:
        """Particle flux of one named species on the domain faces, along +x."""
        return self._domain_fluxes(y, self.device.domain_layer_id, species)

    def ion_fluxes_on_electronic_faces(self, y: ArrayLike, *, species: str = "positive") -> Vector:
        return self._on_electronic_faces(y, self.device.domain_layer_id, species)

    def ion_current_density_A_m2(self, y: ArrayLike) -> Vector:
        """All internal electrical faces: ``q (Gamma_positive - Gamma_negative)``."""
        positive = self.ion_fluxes_on_electronic_faces(y)
        if self.device.ion_species_count == 1:
            single: Vector = Q * positive
            return single
        dual: Vector = Q * (positive - self.ion_fluxes_on_electronic_faces(y, species="negative"))
        return dual


@dataclass(frozen=True)
class AssembledMultipleIonic1DProblem(_AssembledIonicRows):
    """Assembled per-domain rows with the ``MultipleIonic1DProblem`` surface."""

    device: CompiledMultipleIonicDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    system: IonicSystem | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledMultipleIonicDevice1D):
            raise TypeError("a full source-bound multiple-domain device is required")
        object.__setattr__(self, "voltage_V", _finite_scalar(self.voltage_V, "voltage_V"))
        object.__setattr__(self, "generation_scale",
                           _finite_scalar(self.generation_scale, "generation_scale", nonnegative=True))
        _check_drive(self.device.electronic_component, self.voltage_V, self.generation_scale)
        if self.system is None:
            object.__setattr__(self, "system", multiple_ionic_system(self.device))
        elif self.system.device is not self.device or self.system.inventory_row_node is not None:
            raise ValueError("the assembled system must be the original rows of this device")

    def at(self, voltage_V: float, generation_scale: float) -> AssembledMultipleIonic1DProblem:
        """Same bound system at another operating point (no rebinding)."""
        return AssembledMultipleIonic1DProblem(self.device, voltage_V, generation_scale, self.system)

    def residual(self, y: ArrayLike, ydot: ArrayLike | None = None) -> Vector:
        return self._residual(y, ydot, MULTIPLE_ENVELOPE)

    def jacobian(self, y: ArrayLike, ydot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        return self._jacobian(y, ydot, _finite_scalar(alpha, "alpha"))

    def ion_fluxes(self, y: ArrayLike, *, domain_id: str, species: str = "positive") -> Vector:
        if domain_id not in self.device.domain_ids:
            raise ValueError("requested ion domain is not assembled")
        return self._domain_fluxes(y, domain_id, species)

    def ion_fluxes_on_electronic_faces(self, y: ArrayLike, *, domain_id: str, species: str = "positive") -> Vector:
        if domain_id not in self.device.domain_ids:
            raise ValueError("requested ion domain is not assembled")
        return self._on_electronic_faces(y, domain_id, species)

    def ion_current_density_A_m2(self, y: ArrayLike) -> Vector:
        result = np.zeros(self.device.node_count - 1)
        for domain in self.device.domains:
            for species in domain.ion_species_names:
                result += (Q if species == "positive" else -Q) * self.ion_fluxes_on_electronic_faces(
                    y, domain_id=domain.domain_layer_id, species=species)
        return result

    def inventory_gradients(self, y: ArrayLike) -> dict[str, dict[str, Vector]]:
        """Full physical gradients, with shared-site cross terms and no row replacement."""
        storage = self.storage_jacobian(y)
        return {name: {species: -np.asarray(storage[block, :].sum(axis=0)).ravel()
                       for species, block in blocks.items()}
                for name, blocks in self.device.ion_species_slices.items()}


def _require_closed_domain(device: CompiledIonicDevice1D, inventory_row_node: int) -> None:
    """The constrained wrapper's fail-closed binding checks (``ConstrainedIonic1DProblem``)."""
    if type(inventory_row_node) is not int or not 0 <= inventory_row_node < device.ion_node_count:
        raise ValueError("inventory_row_node must name an ion-domain node")
    ion_mobility_partition(device)
    manifest = device.binding_manifest
    domain = manifest.get("ion_domain")
    component_binding = manifest.get("electronic_component")
    ion_model = manifest.get("ion_model")
    if not isinstance(domain, dict) or domain.get("contacts") != ["blocking", "blocking"]:
        raise CapabilityError("ionic steady closure requires the declared closed blocking domain")
    if not isinstance(component_binding, dict) or not isinstance(ion_model, dict):
        raise ValueError("complete component and ionic policy bindings are required")
    if (sha256(device.binding_manifest_json.encode()).hexdigest() != device.identity
            or manifest["original_prepared_sha256"] != device.original_prepared_sha256
            or component_binding["compiled_sha256"] != device.electronic_component.identity
            or ion_model["steric_model"] != device.steric_model
            or not np.array_equal(ion_model["face_diffusivity_m2_s"], device.positive_ion_diffusivity_faces_m2_s)):
        raise ValueError("the full ionic/component provenance binding is inconsistent")
    if (domain["layer"] != device.domain_layer_id
            or not np.array_equal(domain.get("electrical_node_indices"), device.ion_node_indices)
            or not np.array_equal(domain["nodes_m"], device.ion_geometry.nodes_m)
            or not np.array_equal(domain["cell_measures_m"], device.ion_geometry.cell_measures_m)
            or float(domain["reference_inventory_m2"]) != device.reference_inventory_m2):
        raise ValueError("physical inventory must match the original prepared domain and weights")
    if device.ion_species_count == 2:
        if (ion_model.get("site_contract") != device.ion_site_contract
                or device.ion_site_contract not in {"independent_finite_sites", SHARED_SITES}
                or not np.array_equal(ion_model.get("negative_face_diffusivity_m2_s"),
                                      device.negative_ion_diffusivity_faces_m2_s)):
            raise ValueError("the negative ionic provenance binding is inconsistent")
        if domain.get("negative_reference_inventory_m2") != device.negative_reference_inventory_m2:
            raise ValueError("negative physical inventory must match the original prepared domain and weights")


def _profile_rows(slices: Mapping[str, slice], immobile: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    return {species: tuple(range(cast(int, block.start), cast(int, block.stop)))
            for species, block in slices.items() if species in immobile}


@dataclass(frozen=True)
class AssembledConstrainedIonic1DProblem:
    """Assembled K1 inventory / Z profile closure with the constrained-wrapper surface."""

    original: AssembledIonic1DProblem
    inventory_row_node: int
    system: IonicSystem = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.original, AssembledIonic1DProblem):
            raise TypeError("the assembled original ionic rows are required")
        _require_closed_domain(self.original.device, self.inventory_row_node)
        object.__setattr__(self, "system", ionic_system(self.original.device,
                                                        inventory_row_node=self.inventory_row_node))

    @property
    def constraint_row(self) -> int:
        return 3 * self.original.device.node_count + self.inventory_row_node

    @property
    def immobile_species(self) -> tuple[str, ...]:
        return ion_mobility_partition(self.original.device)[1]

    @property
    def profile_constraint_rows(self) -> dict[str, tuple[int, ...]]:
        device = self.original.device
        return _profile_rows(dict(zip(device.ion_species_names, device.ion_species_slices)), self.immobile_species)

    @property
    def inventory_target_m2(self) -> float:
        return self.original.device.reference_inventory_m2

    @property
    def constraint_rows(self) -> dict[str, int]:
        device = self.original.device
        return {species: cast(int, block.start) + self.inventory_row_node
                for species, block in zip(device.ion_species_names, device.ion_species_slices)}

    @property
    def inventory_targets_m2(self) -> dict[str, float]:
        device = self.original.device
        targets = {"positive": self.inventory_target_m2}
        if device.ion_species_count == 2:
            targets["negative"] = device.negative_reference_inventory_m2
        return targets

    def assemble(self, y: ArrayLike, *, derivatives: bool = True) -> AssembledSystem:
        return self.system.assembler.evaluate(y, self.original.inputs, derivatives=derivatives)

    def residual(self, y: ArrayLike) -> Vector:
        return _residual(self.assemble(y, derivatives=False), self.system.closure_rows, RESIDUAL_ENVELOPE)

    def jacobian(self, y: ArrayLike) -> csc_matrix:
        return _jacobian(self.assemble(y), self.system.closure_rows)


@dataclass(frozen=True)
class AssembledConstrainedMultipleIonic1DProblem:
    """Assembled per-domain KM inventory / Z profile closure."""

    original: AssembledMultipleIonic1DProblem
    inventory_row_node: int
    system: IonicSystem = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.original, AssembledMultipleIonic1DProblem):
            raise TypeError("the original multiple-domain equations are required")
        for domain in self.original.device.domains:
            _require_closed_domain(domain, self.inventory_row_node)
        object.__setattr__(self, "system", multiple_ionic_system(
            self.original.device, inventory_row_node=self.inventory_row_node))

    @property
    def immobile_species(self) -> dict[str, tuple[str, ...]]:
        return {domain.domain_layer_id: ion_mobility_partition(domain)[1] for domain in self.original.device.domains}

    @property
    def profile_constraint_rows(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {name: _profile_rows(blocks, self.immobile_species[name])
                for name, blocks in self.original.device.ion_species_slices.items()}

    @property
    def constraint_rows(self) -> dict[str, dict[str, int]]:
        return {name: {species: cast(int, block.start) + self.inventory_row_node
                       for species, block in blocks.items() if species not in self.immobile_species[name]}
                for name, blocks in self.original.device.ion_species_slices.items()}

    def assemble(self, y: ArrayLike, *, derivatives: bool = True) -> AssembledSystem:
        _admit_coordinates(self.original.device, y)
        return self.system.assembler.evaluate(y, self.original.inputs, derivatives=derivatives)

    def residual(self, y: ArrayLike) -> Vector:
        return _residual(self.assemble(y, derivatives=False), self.system.closure_rows, MULTIPLE_ENVELOPE)

    def jacobian(self, y: ArrayLike) -> csc_matrix:
        return _jacobian(self.assemble(y), self.system.closure_rows)

"""2D classical storage and single-positive-ion finite-site assembly on Assembler.

Provides:
* :class:`ClassicalTransient2D` / :class:`AssembledStorage2D`: adds physical
  carrier ``STORAGE`` terms (``S = area * density``, cancellation-free ``expm1``
  increments) to the existing 2D electronic system without mutating DC-only
  :class:`ClassicalDC2D`.
* :class:`IonicSystem2D`, :class:`AssembledIonic2DProblem`, and
  :class:`AssembledConstrainedIonic2DProblem`: binds the four-side-blocking 2D
  ion support ``ion:<layer>`` onto the shared :class:`Assembler` with
  ``M = -S_u``, cancellation-free ``finite_site_secant_2d_m3``, steady inventory
  closure, and exact sparse tangents for optional absorber radiative
  reabsorption and two-sided cross-node interface SRH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts.terms import Geometry
from solarlab.contracts.units import (
    AREA, COULOMB, ELEMENTARY_CHARGE_C, LENGTH, ONE, PARTICLE, SECOND, VOLUME, Unit,
)
from solarlab.equations.assembly.assembler import AssembledSystem, Assembler, Mesh, SupportGeometry
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.two_dimensional import (
    BLOCKS, COORDINATE_FORM, ELECTRON, HOLE, NODES, POTENTIAL, SIDES,
    ClassicalDC2D, SpaceCharge2D, classical_dc_2d,
)
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, SCATTER_DIVERGENCE, STORAGE, Contribution,
    EvaluationState, FaceValue, FieldValues, LocalValue, OperatingInputs, ReductionValue,
    Term, TermValue,
)
from solarlab.equations.assembly.weighted import MEASURE
from solarlab.equations.dc_2d import CompiledDevice2D
from solarlab.equations.ionic_2d import (
    CompiledIonicDevice2D, RadiativeReabsorptionSpec2D, TwoSidedInterfaceSRHCoupling2D,
    _interface_srh_sheet_rate_and_partials, evaluate_two_sided_interface_srh_2d,
    finite_site_secant_2d_m3, ion_pair_flux_jacobian_2d, recompute_g_with_rad_2d,
    trapezoid_weights_1d,
)
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1, IonFaceFluxJacobian
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes

__all__ = [
    "AssembledConstrainedIonic2DProblem",
    "AssembledIonic2DProblem",
    "AssembledIonicDC2DProblem",
    "AssembledIonicTransient2DProblem",
    "AssembledStorage2D",
    "CarrierStorage2D",
    "ClassicalTransient2D",
    "InventoryReduction2D",
    "IonDomain2D",
    "IonStorage2D",
    "IonTransport2D",
    "IonicFields2D",
    "IonicSpaceCharge2D",
    "IonicSystem2D",
    "classical_transient_2d",
    "ionic_system_2d",
]

Q = ELEMENTARY_CHARGE_C
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
DENSITY_UNIT = PARTICLE / VOLUME
INVENTORY_2D_UNIT = PARTICLE / LENGTH
BALANCE_2D_UNIT = PARTICLE / (LENGTH * SECOND)
TERM_SCHEMA_2D = "solarlab.ionic_2d.terms.v1"

Vector: TypeAlias = NDArray[np.float64]


def _identity(*parts: object) -> str:
    return sha256(json.dumps((TERM_SCHEMA_2D, *parts), separators=(",", ":")).encode()).hexdigest()


def _density(state: EvaluationState, block: str) -> Vector:
    return state.field("electron_density_m3" if block == ELECTRON else "hole_density_m3")


@dataclass(frozen=True)
class CarrierStorage2D:
    """Physical 2D carrier inventories ``S = area * density`` and cancellation-free increment."""

    device: CompiledDevice2D
    id: str = "carrier.storage_2d"
    kind: str = STORAGE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = BLOCKS
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, COORDINATE_FORM)

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
        """``S_old * expm1(log(c_new / c_old))`` with exact voltage-reference shift."""
        device = self.device
        thermal = device.thermal_voltage_V
        delta = {block: state.block(block) - previous.block(block) for block in BLOCKS}
        old_ref = device.reference_potentials(previous.inputs.voltage_V)
        new_ref = device.reference_potentials(state.inputs.voltage_V)
        log_n = (delta[POTENTIAL] - (new_ref[0] - old_ref[0])) / thermal + delta[ELECTRON]
        log_p = -(delta[POTENTIAL] - (new_ref[1] - old_ref[1])) / thermal - delta[HOLE]
        return {
            ELECTRON: _density(previous, ELECTRON) * MEASURE * np.expm1(log_n),
            HOLE: _density(previous, HOLE) * MEASURE * np.expm1(log_p),
        }


@dataclass(frozen=True)
class ClassicalTransient2D:
    """Classical 2D DC assembly augmented with physical carrier ``STORAGE`` terms."""

    device: CompiledDevice2D
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = ()
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = ()
    dc: ClassicalDC2D = field(init=False, repr=False, compare=False)
    assembler: Assembler = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dc = classical_dc_2d(self.device)
        storage = CarrierStorage2D(self.device)
        extra: list[Term] = []
        if self.radiative_reabsorption or self.interface_srh_couplings:
            extra.append(
                CoupledSourceCorrection2D(
                    self.device, self.radiative_reabsorption, self.interface_srh_couplings
                )
            )
        terms = (*dc.assembler.terms, *extra, storage)
        assembler = Assembler(dc.assembler.layout, dc.assembler.mesh, dc.assembler.fields, terms)
        object.__setattr__(self, "dc", dc)
        object.__setattr__(self, "assembler", assembler)

    @property
    def identity(self) -> str:
        return self.assembler.identity


def classical_transient_2d(
    device: CompiledDevice2D,
    *,
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = (),
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = (),
) -> ClassicalTransient2D:
    return ClassicalTransient2D(device, radiative_reabsorption, interface_srh_couplings)


@dataclass(frozen=True)
class AssembledStorage2D:
    """Storage-enabled 2D classical electronic problem adapter for BDF/BE steppers."""

    device: CompiledDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = ()
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = ()
    binding: ClassicalTransient2D = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.device.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")
        object.__setattr__(
            self,
            "binding",
            classical_transient_2d(
                self.device,
                radiative_reabsorption=self.radiative_reabsorption,
                interface_srh_couplings=self.interface_srh_couplings,
            ),
        )

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(voltage_V=float(self.voltage_V), generation_scale=float(self.generation_scale))

    def evaluate(self, u: ArrayLike, *, derivatives: bool = True) -> AssembledSystem:
        return self.binding.assembler.evaluate(u, self.inputs, derivatives=derivatives)

    def residual(self, u: ArrayLike) -> Vector:
        system = self.evaluate(u, derivatives=False)
        if not system.admissible:
            return np.full(3 * self.device.node_count, np.nan)
        return system.F

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        system = self.evaluate(u, derivatives=True)
        if not system.admissible or system.J is None:
            raise FloatingPointError("2D electronic Jacobian evaluated outside admissible domain")
        ext = _electronic_extension_off_diagonal_jacobian(
            self.device,
            np.asarray(u, dtype=np.float64)[:3 * self.device.node_count],
            self.voltage_V,
            self.radiative_reabsorption,
            self.interface_srh_couplings,
            matrix_size=3 * self.device.node_count,
        )
        return system.J if ext is None else cast(csc_matrix, (system.J + ext).tocsc())

    def storage(self, u: ArrayLike) -> Vector:
        s_vec, _ = self.binding.assembler.storage(u, self.inputs, derivatives=False)
        return s_vec

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        _, s_u = self.binding.assembler.storage(u, self.inputs, derivatives=True)
        if s_u is None:
            raise FloatingPointError("2D electronic storage tangent evaluated outside admissible domain")
        return s_u

    def mass_matrix(self, u: ArrayLike) -> csc_matrix:
        return -self.storage_tangent(u)

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        return self.mass_matrix(u)

    def storage_increment(self, u: ArrayLike, old_problem: AssembledStorage2D, old_u: ArrayLike) -> Vector:
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("2D storage increment requires the same bound 2D system")
        return self.binding.assembler.storage_increment(u, self.inputs, old_u, old_problem.inputs, physical=False)

    def physical_storage(self, u: ArrayLike) -> Vector:
        return self.binding.assembler.physical_storage(u, self.inputs)

    def physical_increment(self, u: ArrayLike, old_problem: AssembledStorage2D, old_u: ArrayLike) -> Vector:
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("2D physical storage increment requires the same bound 2D system")
        return self.binding.assembler.storage_increment(u, self.inputs, old_u, old_problem.inputs, physical=True)


@dataclass(frozen=True)
class IonDomain2D:
    """Closed 4-side-blocking 2D positive-ion domain metadata."""

    device: CompiledIonicDevice2D

    @property
    def name(self) -> str:
        return self.device.domain_layer_id

    @property
    def node_set(self) -> str:
        return "ion:" + self.name

    @property
    def block(self) -> str:
        return self.node_set + ":positive"

    @property
    def mass_field(self) -> str:
        return "ion_coordinate_jacobian_m3:" + self.name

    def coordinate_jacobian(self, density: Vector) -> Vector:
        return density * (1.0 - density / self.device.positive_ion_site_limit_m3)


@dataclass(frozen=True)
class IonicFields2D:
    """Field map for ``(phi, z_n, z_p, y_ion)`` on the 2D ionic layout."""

    device: CompiledIonicDevice2D

    @property
    def identity(self) -> str:
        return _identity("finite_site_ionic_fields_2d_v1", self.device.identity)

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        domain = IonDomain2D(self.device)
        ordered = np.concatenate([layout.view(u, b) for b in (*BLOCKS, domain.block)])
        try:
            phi, n, p, ion_p = self.device.fields(ordered, voltage_V=inputs.voltage_V)
        except (ValueError, FloatingPointError):
            return FieldValues({}, False)
        admissible = bool(
            np.all(np.isfinite(n)) and np.all(np.isfinite(p))
            and np.all(n > 0.0) and np.all(p > 0.0)
            and np.all(np.isfinite(ion_p)) and np.all(ion_p > 0.0)
        )
        if not admissible:
            return FieldValues({}, False)
        return FieldValues({
            "potential_V": phi,
            "electron_density_m3": n,
            "hole_density_m3": p,
            domain.block: ion_p,
            domain.mass_field: domain.coordinate_jacobian(ion_p),
        }, True)


@dataclass(frozen=True)
class IonicSpaceCharge2D:
    """Full 2D space-charge density with embedded half-cell ion excess tangents."""

    device: CompiledIonicDevice2D
    id: str = "potential.space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (*BLOCKS, IonDomain2D(self.device).block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, COORDINATE_FORM)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        domain = IonDomain2D(self.device)
        n, p = state.field("electron_density_m3"), state.field("hole_density_m3")
        ion_p = state.field(domain.block)
        rho = self.device.charge_ledger(n, p, ion_p).space_charge_C_m3()
        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}
        thermal = self.device.electronic_component.thermal_voltage_V
        nodes = self.device.ion_node_indices
        ratio = np.zeros(n.size, dtype=float)
        ratio[nodes] = state.embedding_ratio(domain.node_set)
        partial = np.zeros(n.size, dtype=float)
        partial[nodes] = state.field(domain.mass_field)
        return {POTENTIAL: LocalValue(rho, {
            POTENTIAL: -Q * MEASURE * (n + p) / thermal,
            ELECTRON: -Q * MEASURE * n,
            HOLE: -Q * MEASURE * p,
            domain.block: Q * MEASURE * ratio * partial,
        })}


@dataclass(frozen=True)
class CoupledSourceCorrection2D:
    """Optional 2D absorber radiative reabsorption and cross-node interface SRH cell source."""

    electronic_component: CompiledDevice2D
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...]
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...]
    id: str = "carrier.coupled_2d_extensions"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = ()
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(
            self.id,
            self.electronic_component.identity,
            [(s.layer_id, s.escape_probability) for s in self.radiative_reabsorption],
            [c.interface_row for c in self.interface_srh_couplings],
        )

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        _ = derivatives
        comp = self.electronic_component
        n, p = state.field("electron_density_m3"), state.field("hole_density_m3")
        correction = np.zeros_like(n)
        shape = comp.geometry.shape
        if self.radiative_reabsorption:
            base = state.inputs.generation_scale * comp.generation_source_m3_s
            b_rad = comp.recombination.coefficients.radiative_m3_s.reshape(shape)
            ni_sq = comp.recombination.coefficients.ni_sq_m6.reshape(shape)
            augmented = recompute_g_with_rad_2d(
                G_optical=base.reshape(shape),
                n=n.reshape(shape),
                p=p.reshape(shape),
                B_rad=b_rad,
                ni_sq=ni_sq,
                x=comp.geometry.x_m,
                y=comp.geometry.y_geometry.nodes_m,
                absorber_y_ranges=tuple((s.y_start, s.y_stop) for s in self.radiative_reabsorption),
                absorber_p_esc=tuple(s.escape_probability for s in self.radiative_reabsorption),
                absorber_areas=tuple(s.area_m2 for s in self.radiative_reabsorption),
            )
            correction = correction + (augmented.ravel() - base)
        if self.interface_srh_couplings:
            report = evaluate_two_sided_interface_srh_2d(
                n.reshape(shape), p.reshape(shape),
                comp.geometry.y_geometry.nodes_m,
                self.interface_srh_couplings,
            )
            correction = correction - report.volumetric_sink_m3_s.ravel()
        return {ELECTRON: LocalValue(correction), HOLE: LocalValue(correction)}


@dataclass(frozen=True)
class IonTransport2D:
    """Four-side-blocking 2D SG positive-ion flux and exact coordinate tangents."""

    domain: IonDomain2D
    kind: str = FACE_FLUX
    unit: Unit = FLUX_UNIT
    port: str | None = None
    divergence: str = SCATTER_DIVERGENCE

    @property
    def id(self) -> str:
        return self.domain.block + ".transport"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.domain.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, self.domain.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity)

    def flux_evaluation(self, state: EvaluationState) -> IonFaceFluxJacobian:
        d = self.domain.device
        phi_ion = state.field("potential_V")[d.ion_node_indices]
        ion_p = state.field(self.domain.block)
        return ion_pair_flux_jacobian_2d(
            phi_ion,
            ion_p,
            d.ion_geometry.contract_geometry.pairs,
            d.ion_geometry.face_spacing_m,
            d.positive_ion_diffusivity_faces_m2_s,
            d.electronic_component.thermal_voltage_V,
            d.positive_ion_site_limit_m3,
            steric_diffusion_only=(d.steric_model == DIFFUSION_ONLY_STERIC_V1),
            drift_sign=1.0,
        )

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        evaluated = self.flux_evaluation(state)
        if not derivatives:
            return {self.domain.block: FaceValue(evaluated.flux)}
        d = self.domain.device
        pairs = d.ion_geometry.contract_geometry.pairs
        left, right = pairs[:, 0], pairs[:, 1]
        dp_dy = state.field(self.domain.mass_field)
        return {
            self.domain.block: FaceValue(
                evaluated.flux,
                {
                    POTENTIAL: evaluated.potential_left_derivative,
                    self.domain.block: evaluated.density_left_derivative * dp_dy[left],
                },
                {
                    POTENTIAL: evaluated.potential_right_derivative,
                    self.domain.block: evaluated.density_right_derivative * dp_dy[right],
                },
            )
        }


@dataclass(frozen=True)
class IonStorage2D:
    """Physical 2D ion storage ``S = area * P`` and cancellation-free site secants."""

    domain: IonDomain2D
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
        return (self.domain.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.domain.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        ion_p = state.field(self.domain.block)
        if not derivatives:
            return {self.domain.block: LocalValue(ion_p)}
        dp_dy = state.field(self.domain.mass_field)
        return {self.domain.block: LocalValue(ion_p, {self.domain.block: MEASURE * dp_dy})}

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        d = self.domain.device
        secant = finite_site_secant_2d_m3(
            d.positive_ion_site_limit_m3,
            d.positive_ion_logit_reference,
            previous.block(self.domain.block),
            state.block(self.domain.block),
        )
        return {self.domain.block: secant}


@dataclass(frozen=True)
class InventoryReduction2D:
    """Whole-domain 2D ion inventory closure row ``fsum(area * P) - fsum(area * P0)``."""

    domain: IonDomain2D
    kind: str = CONSTRAINT
    unit: Unit = DENSITY_UNIT
    port: str | None = None
    measure: str = "cell"

    @property
    def id(self) -> str:
        return self.domain.block + ".inventory"

    @property
    def support(self) -> str:
        return self.domain.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.domain.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.domain.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.domain.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        d = self.domain.device
        density = state.field(self.domain.block)
        reference = d.positive_ion_reference_m3
        summand = MEASURE * density
        offset = MEASURE * reference
        tangents: dict[str, Contribution] = {}
        if derivatives:
            tangents = {self.domain.block: MEASURE * state.field(self.domain.mass_field)}
        return {self.domain.block: ReductionValue(summand, offset, tangents)}


@dataclass(frozen=True)
class IonicSystem2D:
    """Unified Assembler binding for 2D single-mobile-ion transient & constrained steady."""

    device: CompiledIonicDevice2D
    inventory_row_node: int | None = None
    assembler: Assembler = field(init=False, repr=False, compare=False)
    electronic: ClassicalDC2D = field(init=False, repr=False, compare=False)
    ion_transport: IonTransport2D = field(init=False, repr=False, compare=False)
    closure_rows: NDArray[np.bool_] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        d = self.device
        comp = d.electronic_component
        electronic = classical_dc_2d(comp)
        domain = IonDomain2D(d)
        base = electronic.assembler.layout
        node_sets = [
            *base.node_sets,
            NodeSet(domain.node_set, d.ion_node_count, NODES, tuple(int(n) for n in d.ion_node_indices)),
        ]
        blocks = [
            *base.blocks,
            Block(domain.block, domain.node_set, ONE, BALANCE_2D_UNIT),
        ]
        constraints = list(base.constraints)
        closure_terms: list[Term] = []
        if self.inventory_row_node is not None:
            if type(self.inventory_row_node) is not int or not 0 <= self.inventory_row_node < d.ion_node_count:
                raise ValueError("inventory_row_node must name a 2D ion-domain node")
            inv_term = InventoryReduction2D(domain)
            constraints.append(ConstraintRows(domain.block, (self.inventory_row_node,), inv_term.id, INVENTORY_2D_UNIT))
            closure_terms.append(inv_term)

        ion_contract = d.ion_geometry.contract_geometry
        ion_support = SupportGeometry(
            domain.node_set,
            Geometry(
                ion_contract.volumes,
                ion_contract.pairs,
                ion_contract.face_measures,
                AREA,
                LENGTH,
                cell_support=domain.node_set,
                face_support=domain.node_set + ".faces",
            ),
        )
        mesh = Mesh((*electronic.assembler.mesh.supports, ion_support))
        layout = SystemLayout(tuple(node_sets), tuple(blocks), tuple(constraints))

        # Replace electronic SpaceCharge2D with IonicSpaceCharge2D, add optional coupled extensions,
        # CarrierStorage2D, IonTransport2D, IonStorage2D, and optional InventoryReduction2D.
        terms: list[Term] = []
        for term in electronic.assembler.terms:
            if isinstance(term, SpaceCharge2D):
                terms.append(IonicSpaceCharge2D(d))
            else:
                terms.append(term)
        if d.radiative_reabsorption or d.interface_srh_couplings:
            terms.append(
                CoupledSourceCorrection2D(comp, d.radiative_reabsorption, d.interface_srh_couplings)
            )
        ion_transport = IonTransport2D(domain)
        terms.extend((CarrierStorage2D(comp), ion_transport, IonStorage2D(domain), *closure_terms))

        assembler = Assembler(layout, mesh, IonicFields2D(d), tuple(terms))
        mask: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
        if self.inventory_row_node is not None:
            mask[3 * d.node_count + self.inventory_row_node] = True
        mask.setflags(write=False)

        object.__setattr__(self, "electronic", electronic)
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "ion_transport", ion_transport)
        object.__setattr__(self, "closure_rows", mask)

    @property
    def identity(self) -> str:
        return self.assembler.identity


def ionic_system_2d(device: CompiledIonicDevice2D, inventory_row_node: int | None = None) -> IonicSystem2D:
    return IonicSystem2D(device, inventory_row_node)


def _electronic_extension_off_diagonal_jacobian(
    comp: CompiledDevice2D,
    elec_u: ArrayLike,
    voltage_V: float,
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...],
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...],
    *,
    matrix_size: int,
) -> csc_matrix | None:
    """Exact sparse off-diagonal CSC contribution for optional reabsorption / cross-node SRH."""
    if not radiative_reabsorption and not interface_srh_couplings:
        return None
    count = comp.node_count
    _, n, p = comp.fields(elec_u, voltage_V=voltage_V)
    thermal = comp.thermal_voltage_V
    vol = comp.geometry.contract_geometry.volumes
    shape = comp.geometry.shape
    n_2d, p_2d = n.reshape(shape), p.reshape(shape)
    fixed_mask = comp.fixed_carrier_mask

    row_chunks: list[NDArray[np.intp]] = []
    col_chunks: list[NDArray[np.intp]] = []
    val_chunks: list[Vector] = []

    def emit_carrier(carrier_idx: int, offset: int, r: ArrayLike, c: ArrayLike, v: ArrayLike) -> None:
        r_arr = np.asarray(r, dtype=np.intp).ravel()
        c_arr = np.asarray(c, dtype=np.intp).ravel()
        v_arr = np.asarray(v, dtype=float).ravel()
        keep = ~fixed_mask[carrier_idx, r_arr]
        if np.any(keep):
            row_chunks.append(offset + r_arr[keep])
            col_chunks.append(c_arr[keep])
            val_chunks.append(v_arr[keep])

    if radiative_reabsorption:
        b_2d = comp.recombination.coefficients.radiative_m3_s.reshape(shape)
        ni_2d = comp.recombination.coefficients.ni_sq_m6.reshape(shape)
        wx_trap = trapezoid_weights_1d(comp.geometry.x_m)
        for spec in radiative_reabsorption:
            y_slice = slice(spec.y_start, spec.y_stop)
            emission = b_2d[y_slice, :] * (n_2d[y_slice, :] * p_2d[y_slice, :] - ni_2d[y_slice, :])
            r_tot = float(np.trapezoid(np.trapezoid(emission, comp.geometry.y_geometry.nodes_m[y_slice], axis=0),
                                       comp.geometry.x_m))
            if r_tot <= 0.0:
                continue
            wy_trap = trapezoid_weights_1d(comp.geometry.y_geometry.nodes_m[y_slice])
            quad_w = (wy_trap[:, None] * wx_trap[None, :]).ravel()
            abs_nodes = (np.arange(spec.y_start, spec.y_stop, dtype=np.intp)[:, None] * comp.geometry.nx
                         + np.arange(comp.geometry.nx, dtype=np.intp)[None, :]).ravel()
            grad_zn = (1.0 - spec.escape_probability) / spec.area_m2 * quad_w * b_2d[y_slice, :].ravel() * (
                n_2d[y_slice, :].ravel() * p_2d[y_slice, :].ravel()
            )
            rr: NDArray[np.intp]
            cc: NDArray[np.intp]
            rr, cc = np.meshgrid(abs_nodes, abs_nodes, indexing="ij")
            outer = vol[abs_nodes][:, None] * grad_zn[None, :]
            for carrier_idx, offset in ((0, count), (1, 2 * count)):
                emit_carrier(carrier_idx, offset, rr, count + cc, outer)
                emit_carrier(carrier_idx, offset, rr, 2 * count + cc, -outer)

    if interface_srh_couplings:
        nx = comp.geometry.nx
        normal_widths = trapezoid_weights_1d(comp.geometry.y_geometry.nodes_m)
        cols_x: NDArray[np.intp] = np.arange(nx, dtype=np.intp)
        for c in interface_srh_couplings:
            row_nodes: NDArray[np.intp] = c.interface_row * nx + cols_x
            left_nodes: NDArray[np.intp] = c.left_sample_row * nx + cols_x
            right_nodes: NDArray[np.intp] = c.right_sample_row * nx + cols_x
            scale = vol[row_nodes] / normal_widths[c.interface_row]

            raw_a, da_dn, da_dp = _interface_srh_sheet_rate_and_partials(
                n_2d[c.right_sample_row, :], p_2d[c.left_sample_row, :],
                c.pair_a_equilibrium_product_m6, c.n1_m3, c.p1_m3,
                c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
            )
            act_a = (raw_a > 0.0).astype(float)
            dn_r = -scale * act_a * da_dn * n_2d[c.right_sample_row, :]
            dp_l = -scale * act_a * da_dp * p_2d[c.left_sample_row, :]

            raw_b, db_dn, db_dp = _interface_srh_sheet_rate_and_partials(
                n_2d[c.left_sample_row, :], p_2d[c.right_sample_row, :],
                c.pair_b_equilibrium_product_m6, c.n1_m3, c.p1_m3,
                c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
            )
            act_b = (raw_b > 0.0).astype(float)
            dn_l = -scale * act_b * db_dn * n_2d[c.left_sample_row, :]
            dp_r = -scale * act_b * db_dp * p_2d[c.right_sample_row, :]

            for carrier_idx, offset in ((0, count), (1, 2 * count)):
                emit_carrier(carrier_idx, offset, row_nodes, right_nodes, (dn_r - dp_r) / thermal)
                emit_carrier(carrier_idx, offset, row_nodes, count + right_nodes, dn_r)
                emit_carrier(carrier_idx, offset, row_nodes, 2 * count + right_nodes, -dp_r)
                emit_carrier(carrier_idx, offset, row_nodes, left_nodes, (dn_l - dp_l) / thermal)
                emit_carrier(carrier_idx, offset, row_nodes, count + left_nodes, dn_l)
                emit_carrier(carrier_idx, offset, row_nodes, 2 * count + left_nodes, -dp_l)

    if not row_chunks:
        return None
    return csc_matrix(
        (np.concatenate(val_chunks), (np.concatenate(row_chunks), np.concatenate(col_chunks))),
        shape=(matrix_size, matrix_size),
    )


def _extension_off_diagonal_jacobian(device: CompiledIonicDevice2D, u: ArrayLike, voltage_V: float,
                                     generation_scale: float) -> csc_matrix | None:
    _ = generation_scale
    comp = device.electronic_component
    return _electronic_extension_off_diagonal_jacobian(
        comp,
        np.asarray(u, dtype=np.float64)[:3 * comp.node_count],
        voltage_V,
        device.radiative_reabsorption,
        device.interface_srh_couplings,
        matrix_size=device.state_size,
    )


@dataclass(frozen=True)
class AssembledIonic2DProblem:
    """Production Assembler-backed 2D single-mobile-ion DAE problem."""

    device: CompiledIonicDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    binding: IonicSystem2D = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.device.electronic_component.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")
        object.__setattr__(self, "binding", ionic_system_2d(self.device))

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(voltage_V=float(self.voltage_V), generation_scale=float(self.generation_scale))

    def evaluate(self, u: ArrayLike, *, derivatives: bool = True) -> AssembledSystem:
        return self.binding.assembler.evaluate(u, self.inputs, derivatives=derivatives)

    def residual(self, u: ArrayLike) -> Vector:
        system = self.evaluate(u, derivatives=False)
        if not system.admissible:
            return np.full(self.device.state_size, np.nan)
        return system.F

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        system = self.evaluate(u, derivatives=True)
        if not system.admissible or system.J is None:
            raise FloatingPointError("2D ionic Jacobian evaluated outside admissible domain")
        ext = _extension_off_diagonal_jacobian(self.device, u, self.voltage_V, self.generation_scale)
        return system.J if ext is None else cast(csc_matrix, (system.J + ext).tocsc())

    def storage(self, u: ArrayLike) -> Vector:
        s_vec, _ = self.binding.assembler.storage(u, self.inputs, derivatives=False)
        return s_vec

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        _, s_u = self.binding.assembler.storage(u, self.inputs, derivatives=True)
        if s_u is None:
            raise FloatingPointError("2D ionic storage tangent evaluated outside admissible domain")
        return s_u

    def mass_matrix(self, u: ArrayLike) -> csc_matrix:
        return -self.storage_tangent(u)

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        return self.mass_matrix(u)

    def storage_increment(self, u: ArrayLike, old_problem: AssembledIonic2DProblem, old_u: ArrayLike) -> Vector:
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("2D ionic storage increment requires the same bound 2D system")
        return self.binding.assembler.storage_increment(u, self.inputs, old_u, old_problem.inputs, physical=False)

    def physical_storage(self, u: ArrayLike) -> Vector:
        return self.binding.assembler.physical_storage(u, self.inputs)

    def physical_increment(self, u: ArrayLike, old_problem: AssembledIonic2DProblem, old_u: ArrayLike) -> Vector:
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("2D ionic physical increment requires the same bound 2D system")
        return self.binding.assembler.storage_increment(u, self.inputs, old_u, old_problem.inputs, physical=True)

    def carrier_fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("2D carrier fluxes evaluated outside admissible domain")
        return self.binding.electronic.transport.fluxes(state)

    def ion_flux_evaluation(self, u: ArrayLike) -> IonFaceFluxJacobian:
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("2D ion fluxes evaluated outside admissible domain")
        return self.binding.ion_transport.flux_evaluation(state)

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        return self.binding.electronic.recombination.rate(n, p, derivatives=derivatives)

    def contact_particle_fluxes(self, u: ArrayLike, *, storage_rate: Vector | None = None) -> tuple[Vector, Vector]:
        """Stack y_min/y_max carrier particle fluxes along +y closing contact cells."""
        system = self.evaluate(u, derivatives=False)
        if not system.admissible:
            raise FloatingPointError("2D contact particle fluxes evaluated outside admissible domain")
        assembler = self.binding.assembler
        result: list[Vector] = []
        for block in (ELECTRON, HOLE):
            port_rows = [
                assembler.reaction_flux(system, block, port_id, storage_rate=storage_rate)
                for _, port_id in SIDES
            ]
            result.append(np.stack(port_rows))
        return result[0], result[1]

    def contact_displacement_C_m2(self, u: ArrayLike) -> Vector:
        """Stack y_min/y_max electric displacement ``D_y`` along +y closing Gauss contact cells."""
        system = self.evaluate(u, derivatives=False)
        if not system.admissible:
            raise FloatingPointError("2D contact displacement evaluated outside admissible domain")
        assembler = self.binding.assembler
        return np.stack([
            assembler.reaction_flux(system, POTENTIAL, port_id)
            for _, port_id in SIDES
        ])


@dataclass(frozen=True)
class AssembledConstrainedIonic2DProblem:
    """Assembler-backed steady 2D ionic problem with domain-inventory closure."""

    original: AssembledIonic2DProblem
    inventory_row_node: int
    binding: IonicSystem2D = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        d = self.original.device
        if type(self.inventory_row_node) is not int or not 0 <= self.inventory_row_node < d.ion_node_count:
            raise ValueError("inventory_row_node must name a 2D ion-domain node")
        object.__setattr__(self, "binding", ionic_system_2d(d, self.inventory_row_node))

    @property
    def constraint_row(self) -> int:
        return 3 * self.original.device.node_count + self.inventory_row_node

    @property
    def inventory_target_m1(self) -> float:
        return self.original.device.reference_inventory_m1

    def residual(self, u: ArrayLike) -> Vector:
        system = self.binding.assembler.evaluate(u, self.original.inputs, derivatives=False)
        if not system.admissible:
            return np.full(self.original.device.state_size, np.nan)
        return system.F

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        system = self.binding.assembler.evaluate(u, self.original.inputs, derivatives=True)
        if not system.admissible or system.J is None:
            raise FloatingPointError("2D constrained ionic Jacobian evaluated outside admissible domain")
        ext = _extension_off_diagonal_jacobian(
            self.original.device, u, self.original.voltage_V, self.original.generation_scale
        )
        return system.J if ext is None else cast(csc_matrix, (system.J + ext).tocsc())


AssembledIonicDC2DProblem = AssembledIonic2DProblem
AssembledIonicTransient2DProblem = AssembledIonic2DProblem


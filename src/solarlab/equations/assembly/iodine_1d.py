"""Unified sparse 1D assembly of coupled ``I^- / I^0 / I_2`` reaction-diffusion.

Builds strictly on :class:`~solarlab.equations.assembly.layout.SystemLayout`,
:class:`~solarlab.equations.assembly.terms.Term`, and
:class:`~solarlab.equations.assembly.assembler.Assembler` without modifying core
assembly infrastructure or duplicating nonlinear/time solvers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts import ContractError, Vector, frozen_array
from solarlab.contracts.units import (
    AREA,
    COULOMB,
    ELEMENTARY_CHARGE_C,
    ONE,
    PARTICLE,
    SECOND,
    VOLT,
    VOLUME,
    Unit,
)
from solarlab.equations.assembly.assembler import (
    AssembledSystem,
    Assembler,
    Mesh,
    Port,
    SupportGeometry,
)
from solarlab.equations.assembly.classical_1d import (
    BulkRecombination,
    CarrierExchange,
    CarrierGeneration,
    CarrierReservoirs,
    CarrierStorage,
    CarrierTransport,
    GaussDisplacement,
    InterfaceCarrierPort,
    InterfaceGaussPort,
    PotentialContacts,
    populate_interface_fields,
)
from solarlab.equations.assembly.layout import (
    Block,
    ConstraintRows,
    NodeSet,
    SystemLayout,
)
from solarlab.equations.assembly.quasi_fermi import ELECTRODE_QUASI_FERMI_1D_V1
from solarlab.equations.assembly.terms import (
    CELL_SOURCE,
    CHAIN_DIVERGENCE,
    CONSTRAINT,
    FACE_FLUX,
    PORT_FLUX,
    STORAGE,
    Contribution,
    EvaluationState,
    FaceValue,
    FieldValues,
    LocalValue,
    OperatingInputs,
    ReductionValue,
    Term,
    TermValue,
)
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.iodine_1d import (
    IODINE_BLOCKS,
    CompiledIodineDevice1D,
    IodineBoundaryPortSpec,
)
from solarlab.physics.chemistry import (
    evaluate_iodine_kinetics,
    species_sg_face_flux,
)
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.electrostatics import contact_displacement
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFluxes

__all__ = [
    "AssembledConstrainedIodine1DProblem",
    "AssembledIodine1DProblem",
    "CHEM_NODES",
    "IODIDE",
    "IODINE_ATOM",
    "IODINE_MOLECULE",
    "IodineCarrierReactionTerm",
    "IodineContactReactionPortTerm",
    "IodineFields",
    "IodineHoleContactPortTerm",
    "IodineMoleculeEscapePortTerm",
    "IodineSpaceChargeTerm",
    "IodineSpeciesReactionTerm",
    "IodineSpeciesStorageTerm",
    "IodineSystem",
    "IodineTransportTerm",
    "SingleSpeciesInventoryTerm",
    "TotalIodineInventoryTerm",
    "iodine_system",
]

Q = ELEMENTARY_CHARGE_C
NODES = "nodes"
CHEM_NODES = "chem"
POTENTIAL, ELECTRON, HOLE = "potential", "electron", "hole"
IODIDE: Final = "iodide"
IODINE_ATOM: Final = "iodine_atom"
IODINE_MOLECULE: Final = "iodine_molecule"
CARRIERS = ((ELECTRON, "n", 1.0), (HOLE, "p", -1.0))

DISPLACEMENT_UNIT = COULOMB / AREA
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
DENSITY_UNIT = PARTICLE / VOLUME
INVENTORY_UNIT = PARTICLE / AREA

TERM_SCHEMA = "solarlab.iodine_1d.terms.v1"


def _identity(*parts: object) -> str:
    return sha256(
        json.dumps((TERM_SCHEMA, *parts), separators=(",", ":")).encode()
    ).hexdigest()


def _carrier_chain_tangents(
    dn_coeff: Vector,
    dp_coeff: Vector,
    n: Vector,
    p: Vector,
    thermal_V: float,
) -> tuple[Vector, Vector, Vector]:
    """Chain rule from ``(d/dn, d/dp)`` to ``(d/dphi, d/dzn, d/dzp)``."""
    t_zn = dn_coeff * n
    t_zp = -dp_coeff * p
    t_phi = (t_zn + t_zp) / thermal_V
    return t_phi, t_zn, t_zp


@dataclass(frozen=True)
class IodineFields:
    """Single-pass field and bulk-reaction evaluation per trial state."""

    device: CompiledIodineDevice1D

    @property
    def identity(self) -> str:
        return _identity("iodine_fields_v1", self.device.identity)

    def evaluate(
        self, layout: SystemLayout, u: Vector, inputs: OperatingInputs
    ) -> FieldValues:
        device = self.device
        try:
            (
                phi,
                n,
                p,
                cm,
                c0,
                c2,
                mm,
                m0,
                m2,
            ) = device.fields_and_tangents(u, voltage_V=inputs.voltage_V)
            idx = device.chem_node_indices
            rxn = evaluate_iodine_kinetics(
                n[idx], p[idx], cm, c0, c2, device.kinetics, derivatives=True
            )
        except (ValueError, FloatingPointError, ContractError):
            return FieldValues({}, False)

        assert rxn.dsources_dstate is not None
        fields: dict[str, Vector] = {
            "potential_V": phi,
            "electron_density_m3": n,
            "hole_density_m3": p,
            IODIDE: cm,
            IODINE_ATOM: c0,
            IODINE_MOLECULE: c2,
            "mass:iodide": mm,
            "mass:iodine_atom": m0,
            "mass:iodine_molecule": m2,
            "rxn:rates": rxn.rates_m3_s,
            "rxn:sources": rxn.sources_m3_s,
            "rxn:dsources": rxn.dsources_dstate,
        }
        component = device.electronic_component
        if component.interface_model is not None:
            ok = populate_interface_fields(
                component, u[: 3 * component.node_count], phi, n, p, fields
            )
            return FieldValues(fields, ok)
        return FieldValues(fields, True)


@dataclass(frozen=True)
class IodineSpaceChargeTerm:
    """Gauss space-charge cell source coupling ``(n, p, c_-, N_fix^+)``."""

    device: CompiledIodineDevice1D
    id: str = "potential.iodine_space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE, IODIDE)
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(
            self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1
        )

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        n = state.field("electron_density_m3")
        p = state.field("hole_density_m3")
        cm = state.field(IODIDE)
        rho = self.device.charge_ledger(n, p, cm).space_charge_C_m3()
        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}
        vt = self.device.electronic_component.thermal_voltage_V
        idx = self.device.chem_node_indices
        ratio = state.embedding_ratio(CHEM_NODES)
        d_iodide = np.zeros(n.size, dtype=np.float64)
        d_iodide[idx] = -Q * ratio * state.field("mass:iodide")
        tangents: dict[str, Contribution] = {
            POTENTIAL: -Q * (n + p) / vt,
            ELECTRON: -Q * n,
            HOLE: -Q * p,
            IODIDE: d_iodide,
        }
        return {POTENTIAL: LocalValue(rho, tangents)}


@dataclass(frozen=True)
class IodineSpeciesStorageTerm:
    """Physical storage ``c_j`` and subtraction-free BDF secants on ``chem``."""

    device: CompiledIodineDevice1D
    id: str = "chem.storage"
    kind: str = STORAGE
    support: str = CHEM_NODES
    equations: tuple[str, ...] = IODINE_BLOCKS
    dependencies: tuple[str, ...] = IODINE_BLOCKS
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        out: dict[str, TermValue] = {}
        for block in IODINE_BLOCKS:
            dens = state.field(block)
            if derivatives:
                out[block] = LocalValue(dens, {block: state.field("mass:" + block)})
            else:
                out[block] = LocalValue(dens)
        return out

    def increment(
        self, state: EvaluationState, previous: EvaluationState
    ) -> Mapping[str, Contribution]:
        dev = self.device
        return {
            IODIDE: dev.iodide_map.secant_increment_m3(
                state.block(IODIDE), previous.block(IODIDE)
            ),
            IODINE_ATOM: dev.atom_map.secant_increment_m3(
                state.block(IODINE_ATOM), previous.block(IODINE_ATOM)
            ),
            IODINE_MOLECULE: dev.mol_map.secant_increment_m3(
                state.block(IODINE_MOLECULE), previous.block(IODINE_MOLECULE)
            ),
        }


@dataclass(frozen=True)
class IodineTransportTerm:
    """SG drift-diffusion of ``I^-`` and Fick/partition diffusion of ``I^0, I_2``."""

    device: CompiledIodineDevice1D
    species: Literal["iodide", "iodine_atom", "iodine_molecule"]
    kind: str = FACE_FLUX
    support: str = CHEM_NODES
    unit: Unit = FLUX_UNIT
    port: str | None = None
    divergence: str = CHAIN_DIVERGENCE

    @property
    def id(self) -> str:
        return f"chem.transport.{self.species}"

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.species,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        if self.species == IODIDE:
            return (POTENTIAL, IODIDE)
        return (self.species,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def flux_evaluation(
        self, state: EvaluationState, *, derivatives: bool
    ) -> tuple[Vector, dict[str, Contribution], dict[str, Contribution]]:
        dev = self.device
        phi = state.field("potential_V")[dev.chem_node_indices]
        dx = np.diff(dev.chem_geometry.nodes_m)
        vt = dev.electronic_component.thermal_voltage_V
        dens = state.field(self.species)
        mass = state.field("mass:" + self.species)

        if self.species == IODIDE:
            diff = dev.iodide_diffusivity_faces_m2_s
            log_part = dev.log_partition_minus_faces
            charge = -1
            steric = dev.steric_site_limit_m3
        elif self.species == IODINE_ATOM:
            diff = dev.atom_diffusivity_faces_m2_s
            log_part = dev.log_partition_atom_faces
            charge = 0
            steric = None
        else:
            diff = dev.mol_diffusivity_faces_m2_s
            log_part = dev.log_partition_mol_faces
            charge = 0
            steric = None

        ev = species_sg_face_flux(
            phi,
            dens,
            dx,
            diff,
            vt,
            charge_number=charge,
            log_partition_step=log_part,
            steric_site_limit_m3=steric,
            steric_diffusion_only=dev.steric_diffusion_only,
            derivatives=derivatives,
        )
        if not derivatives:
            return ev.flux_m2_s, {}, {}
        assert (
            ev.dflux_dc_left is not None
            and ev.dflux_dc_right is not None
            and ev.dflux_dphi_left is not None
            and ev.dflux_dphi_right is not None
        )
        left: dict[str, Contribution] = {
            self.species: ev.dflux_dc_left * mass[:-1],
        }
        right: dict[str, Contribution] = {
            self.species: ev.dflux_dc_right * mass[1:],
        }
        if self.species == IODIDE:
            left[POTENTIAL] = ev.dflux_dphi_left
            right[POTENTIAL] = ev.dflux_dphi_right
        return ev.flux_m2_s, left, right

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        flux, left, right = self.flux_evaluation(state, derivatives=derivatives)
        if not derivatives:
            return {self.species: FaceValue(flux)}
        return {self.species: FaceValue(flux, left, right)}


@dataclass(frozen=True)
class IodineCarrierReactionTerm:
    """Embed bulk iodine chemical sources ``(S_n, S_p)`` onto electronic ``nodes``."""

    device: CompiledIodineDevice1D
    id: str = "carrier.iodine_reactions"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = (
        POTENTIAL,
        ELECTRON,
        HOLE,
        IODIDE,
        IODINE_ATOM,
        IODINE_MOLECULE,
    )
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        dev = self.device
        n_total = dev.node_count
        idx = dev.chem_node_indices
        ratio = state.embedding_ratio(CHEM_NODES)
        sources = state.field("rxn:sources")

        s_n: NDArray[np.float64] = np.zeros(n_total, dtype=np.float64)
        s_p: NDArray[np.float64] = np.zeros(n_total, dtype=np.float64)
        s_n[idx] = ratio * sources[0]
        s_p[idx] = ratio * sources[1]
        if not derivatives:
            return {ELECTRON: LocalValue(s_n), HOLE: LocalValue(s_p)}

        ds = state.field("rxn:dsources")
        n_chem = state.field("electron_density_m3")[idx]
        p_chem = state.field("hole_density_m3")[idx]
        vt = dev.electronic_component.thermal_voltage_V
        mm = state.field("mass:iodide")
        m0 = state.field("mass:iodine_atom")
        m2 = state.field("mass:iodine_molecule")

        out: dict[str, TermValue] = {}
        for sp_row, block, val in ((0, ELECTRON, s_n), (1, HOLE, s_p)):
            t_phi_c, t_zn_c, t_zp_c = _carrier_chain_tangents(
                ds[sp_row, 0], ds[sp_row, 1], n_chem, p_chem, vt
            )
            tangents: dict[str, Contribution] = {}
            for dep, chem_tan in (
                (POTENTIAL, t_phi_c),
                (ELECTRON, t_zn_c),
                (HOLE, t_zp_c),
                (IODIDE, ds[sp_row, 2] * mm),
                (IODINE_ATOM, ds[sp_row, 3] * m0),
                (IODINE_MOLECULE, ds[sp_row, 4] * m2),
            ):
                full: NDArray[np.float64] = np.zeros(n_total, dtype=np.float64)
                full[idx] = ratio * chem_tan
                tangents[dep] = full
            out[block] = LocalValue(val, tangents)
        return out


@dataclass(frozen=True)
class IodineSpeciesReactionTerm:
    """Bulk iodine chemical sources ``(S_-, S_0, S_2)`` on ``chem``."""

    device: CompiledIodineDevice1D
    id: str = "chem.iodine_reactions"
    kind: str = CELL_SOURCE
    support: str = CHEM_NODES
    equations: tuple[str, ...] = IODINE_BLOCKS
    dependencies: tuple[str, ...] = (
        POTENTIAL,
        ELECTRON,
        HOLE,
        IODIDE,
        IODINE_ATOM,
        IODINE_MOLECULE,
    )
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        dev = self.device
        sources = state.field("rxn:sources")
        if not derivatives:
            return {
                IODIDE: LocalValue(sources[2]),
                IODINE_ATOM: LocalValue(sources[3]),
                IODINE_MOLECULE: LocalValue(sources[4]),
            }

        idx = dev.chem_node_indices
        ds = state.field("rxn:dsources")
        n_chem = state.field("electron_density_m3")[idx]
        p_chem = state.field("hole_density_m3")[idx]
        vt = dev.electronic_component.thermal_voltage_V
        mm = state.field("mass:iodide")
        m0 = state.field("mass:iodine_atom")
        m2 = state.field("mass:iodine_molecule")

        out: dict[str, TermValue] = {}
        for sp_row, block in (
            (2, IODIDE),
            (3, IODINE_ATOM),
            (4, IODINE_MOLECULE),
        ):
            t_phi, t_zn, t_zp = _carrier_chain_tangents(
                ds[sp_row, 0], ds[sp_row, 1], n_chem, p_chem, vt
            )
            tangents: dict[str, Contribution] = {
                POTENTIAL: t_phi,
                ELECTRON: t_zn,
                HOLE: t_zp,
                IODIDE: ds[sp_row, 2] * mm,
                IODINE_ATOM: ds[sp_row, 3] * m0,
                IODINE_MOLECULE: ds[sp_row, 4] * m2,
            }
            out[block] = LocalValue(sources[sp_row], tangents)
        return out


@dataclass(frozen=True)
class IodineMoleculeEscapePortTerm:
    """Volatile surface ``I_2`` escape ``J_out = k_loss * c_2`` on one port."""

    device: CompiledIodineDevice1D
    spec: IodineBoundaryPortSpec
    kind: str = PORT_FLUX
    support: str = CHEM_NODES
    equations: tuple[str, ...] = (IODINE_MOLECULE,)
    dependencies: tuple[str, ...] = (IODINE_MOLECULE,)
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"chem.escape.{self.spec.side}"

    @property
    def port(self) -> str:
        return self.spec.side

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.spec.side)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        node = self.spec.chem_node
        n_out = float(self.spec.outward_normal)
        k_loss = self.spec.mol_escape_velocity_m_s
        c2 = float(state.field(IODINE_MOLECULE)[node])
        exchange = np.array([n_out * k_loss * c2], dtype=np.float64)
        if not derivatives:
            return {IODINE_MOLECULE: LocalValue(exchange)}
        m2 = float(state.field("mass:iodine_molecule")[node])
        return {
            IODINE_MOLECULE: LocalValue(
                exchange,
                {IODINE_MOLECULE: np.array([n_out * k_loss * m2], dtype=np.float64)},
            )
        }


def _contact_oxidation_rate_and_partials(
    device: CompiledIodineDevice1D,
    spec: IodineBoundaryPortSpec,
    state: EvaluationState,
) -> tuple[float, float, float, float]:
    """Return ``(J_ct, dJ/dp, dJ/dc_-, dJ/dc_0)`` at ``spec``."""
    k_ct = spec.contact_oxidation_rate_m4_s
    p_val = float(state.field("hole_density_m3")[spec.elec_node])
    cm_val = float(state.field(IODIDE)[spec.chem_node])
    c0_val = float(state.field(IODINE_ATOM)[spec.chem_node])
    if spec.reversible_contact:
        rate = k_ct * (cm_val * p_val - spec.p1_iodine_m3 * c0_val)
        return rate, k_ct * cm_val, k_ct * p_val, -k_ct * spec.p1_iodine_m3
    rate = k_ct * cm_val * p_val
    return rate, k_ct * cm_val, k_ct * p_val, 0.0


@dataclass(frozen=True)
class IodineContactReactionPortTerm:
    """Surface contact reaction port on ``iodide`` or ``iodine_atom``."""

    device: CompiledIodineDevice1D
    spec: IodineBoundaryPortSpec
    species: Literal["iodide", "iodine_atom"]
    kind: str = PORT_FLUX
    support: str = CHEM_NODES
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"chem.contact_reaction.{self.species}.{self.spec.side}"

    @property
    def port(self) -> str:
        return self.spec.side

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.species,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, HOLE, IODIDE, IODINE_ATOM)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        spec = self.spec
        # Outward loss is +J_ct for iodide (consumed) and -J_ct for iodine_atom (created)
        sign = 1.0 if self.species == IODIDE else -1.0
        n_out = float(spec.outward_normal) * sign
        rate, dj_dp, dj_dcm, dj_dc0 = _contact_oxidation_rate_and_partials(
            self.device, spec, state
        )
        exchange = np.array([n_out * rate], dtype=np.float64)
        if not derivatives:
            return {self.species: LocalValue(exchange)}
        p_val = float(state.field("hole_density_m3")[spec.elec_node])
        vt = self.device.electronic_component.thermal_voltage_V
        mm = float(state.field("mass:iodide")[spec.chem_node])
        m0 = float(state.field("mass:iodine_atom")[spec.chem_node])
        tangents: dict[str, Contribution] = {
            POTENTIAL: np.array([n_out * (-dj_dp * p_val / vt)], dtype=np.float64),
            HOLE: np.array([n_out * (-dj_dp * p_val)], dtype=np.float64),
            IODIDE: np.array([n_out * dj_dcm * mm], dtype=np.float64),
            IODINE_ATOM: np.array([n_out * dj_dc0 * m0], dtype=np.float64),
        }
        return {self.species: LocalValue(exchange, tangents)}


@dataclass(frozen=True)
class IodineHoleContactPortTerm:
    """Hole consumption ``-J_ct`` on ``nodes`` from contact oxidation ``I^- + h^+ -> I^0``."""

    device: CompiledIodineDevice1D
    spec: IodineBoundaryPortSpec
    kind: str = PORT_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (HOLE,)
    dependencies: tuple[str, ...] = (POTENTIAL, HOLE, IODIDE, IODINE_ATOM)
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"hole.iodine_contact_reaction.{self.spec.side}"

    @property
    def port(self) -> str:
        return f"chem_{self.spec.side}"

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        spec = self.spec
        n_out = float(spec.outward_normal)
        rate, dj_dp, dj_dcm, dj_dc0 = _contact_oxidation_rate_and_partials(
            self.device, spec, state
        )
        exchange = np.array([n_out * rate], dtype=np.float64)
        if not derivatives:
            return {HOLE: LocalValue(exchange)}
        p_val = float(state.field("hole_density_m3")[spec.elec_node])
        vt = self.device.electronic_component.thermal_voltage_V
        mm = float(state.field("mass:iodide")[spec.chem_node])
        m0 = float(state.field("mass:iodine_atom")[spec.chem_node])
        tangents: dict[str, Contribution] = {
            POTENTIAL: np.array([n_out * (-dj_dp * p_val / vt)], dtype=np.float64),
            HOLE: np.array([n_out * (-dj_dp * p_val)], dtype=np.float64),
            IODIDE: np.array([n_out * dj_dcm * mm], dtype=np.float64),
            IODINE_ATOM: np.array([n_out * dj_dc0 * m0], dtype=np.float64),
        }
        return {HOLE: LocalValue(exchange, tangents)}


@dataclass(frozen=True)
class TotalIodineInventoryTerm:
    """Closed-domain steady constraint ``sum w (c_- + c_0 + 2 c_2) - M_I,0 = 0``."""

    device: CompiledIodineDevice1D
    id: str = "chem.total_iodine_inventory"
    kind: str = CONSTRAINT
    support: str = CHEM_NODES
    equations: tuple[str, ...] = (IODIDE,)
    dependencies: tuple[str, ...] = IODINE_BLOCKS
    unit: Unit = DENSITY_UNIT
    port: str | None = None
    measure: str = "cell"

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        cm = state.field(IODIDE)
        c0 = state.field(IODINE_ATOM)
        c2 = state.field(IODINE_MOLECULE)
        summand = cm + c0 + 2.0 * c2
        u_ref = self.device.initial_chem_coordinates()
        m = self.device.chem_node_count
        ref = (
            self.device.iodide_map.density_m3(u_ref[:m])
            + self.device.atom_map.density_m3(u_ref[m : 2 * m])
            + 2.0 * self.device.mol_map.density_m3(u_ref[2 * m : 3 * m])
        )
        if not derivatives:
            return {IODIDE: ReductionValue(summand, reference=ref)}
        tangents: dict[str, Contribution] = {
            IODIDE: state.field("mass:iodide"),
            IODINE_ATOM: state.field("mass:iodine_atom"),
            IODINE_MOLECULE: 2.0 * state.field("mass:iodine_molecule"),
        }
        return {IODIDE: ReductionValue(summand, tangents=tangents, reference=ref)}


@dataclass(frozen=True)
class SingleSpeciesInventoryTerm:
    """Closed-domain steady inventory constraint for one unreactive species."""

    device: CompiledIodineDevice1D
    species: Literal["iodide", "iodine_atom", "iodine_molecule"]
    kind: str = CONSTRAINT
    support: str = CHEM_NODES
    unit: Unit = DENSITY_UNIT
    port: str | None = None
    measure: str = "cell"

    @property
    def id(self) -> str:
        return f"chem.inventory.{self.species}"

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.species,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.species,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(
        self, state: EvaluationState, *, derivatives: bool
    ) -> Mapping[str, TermValue]:
        dens = state.field(self.species)
        u_ref = self.device.initial_chem_coordinates()
        m = self.device.chem_node_count
        if self.species == IODIDE:
            ref = self.device.iodide_map.density_m3(u_ref[:m])
        elif self.species == IODINE_ATOM:
            ref = self.device.atom_map.density_m3(u_ref[m : 2 * m])
        else:
            ref = self.device.mol_map.density_m3(u_ref[2 * m : 3 * m])
        if not derivatives:
            return {self.species: ReductionValue(dens, reference=ref)}
        return {
            self.species: ReductionValue(
                dens,
                tangents={self.species: state.field("mass:" + self.species)},
                reference=ref,
            )
        }


def _build_layout_and_terms(
    device: CompiledIodineDevice1D,
    *,
    constrained_steady: bool,
    inventory_node: int = 0,
) -> tuple[
    SystemLayout,
    Mesh,
    list[Term],
    CarrierTransport,
    BulkRecombination,
    dict[tuple[str, str], CarrierExchange],
    dict[str, IodineTransportTerm],
]:
    comp = device.electronic_component
    n_count = comp.node_count
    m_count = device.chem_node_count
    if not 0 <= inventory_node < m_count:
        raise ValueError("inventory_node must index a chemical domain node")

    constraints: list[ConstraintRows] = [
        ConstraintRows(POTENTIAL, (0, n_count - 1), "potential.contacts", VOLT)
    ]
    for block, carrier, _ in CARRIERS:
        fixed_nodes = tuple(
            node
            for side, node in (("left", 0), ("right", n_count - 1))
            if comp.channel(carrier, side).kind == "fixed_reservoir"
        )
        if fixed_nodes:
            constraints.append(
                ConstraintRows(block, fixed_nodes, block + ".reservoirs", ONE)
            )

    if constrained_steady:
        if device.has_open_iodine_loss:
            raise CapabilityError(
                "open-boundary iodine escape or irreversible surface immobilization "
                "has no non-trivial closed steady inventory; use transient solving"
            )
        if device.is_unreactive_limit:
            for sp in IODINE_BLOCKS:
                constraints.append(
                    ConstraintRows(
                        sp, (inventory_node,), f"chem.inventory.{sp}", INVENTORY_UNIT
                    )
                )
        else:
            constraints.append(
                ConstraintRows(
                    IODIDE,
                    (inventory_node,),
                    "chem.total_iodine_inventory",
                    INVENTORY_UNIT,
                )
            )

    node_sets = (
        NodeSet(NODES, n_count),
        NodeSet(
            CHEM_NODES,
            m_count,
            parent=NODES,
            parent_indices=tuple(int(i) for i in device.chem_node_indices),
        ),
    )
    blocks = (
        Block(POTENTIAL, NODES, VOLT, DISPLACEMENT_UNIT),
        Block(ELECTRON, NODES, ONE, FLUX_UNIT),
        Block(HOLE, NODES, ONE, FLUX_UNIT),
        Block(IODIDE, CHEM_NODES, ONE, FLUX_UNIT),
        Block(IODINE_ATOM, CHEM_NODES, ONE, FLUX_UNIT),
        Block(IODINE_MOLECULE, CHEM_NODES, ONE, FLUX_UNIT),
    )
    layout = SystemLayout(node_sets, blocks, tuple(constraints))

    # Mesh supports & boundary ports
    elec_geom = comp.geometry.to_contract_geometry(
        cell_support=NODES, face_support=NODES + ".faces"
    )
    elec_ports = [
        Port(
            face.id,
            (face.node,),
            face.outward_normal,
            np.array([face.area_m2]),
            ONE,
        )
        for face in comp.geometry.ports
    ]
    if comp.interface_model is not None:
        interior = tuple(range(1, n_count - 1))
        elec_ports.append(Port("interfaces", interior, -1, np.ones(len(interior)), ONE))
    for bspec in (device.left_boundary, device.right_boundary):
        elec_ports.append(
            Port(
                f"chem_{bspec.side}",
                (bspec.elec_node,),
                bspec.outward_normal,
                np.array([1.0]),
                ONE,
            )
        )

    chem_geom = device.chem_geometry.to_contract_geometry(
        cell_support=CHEM_NODES, face_support=CHEM_NODES + ".faces"
    )
    chem_ports = (
        Port(
            "left",
            (device.left_boundary.chem_node,),
            device.left_boundary.outward_normal,
            np.array([1.0]),
            ONE,
        ),
        Port(
            "right",
            (device.right_boundary.chem_node,),
            device.right_boundary.outward_normal,
            np.array([1.0]),
            ONE,
        ),
    )
    mesh = Mesh(
        (
            SupportGeometry(NODES, elec_geom, tuple(elec_ports)),
            SupportGeometry(CHEM_NODES, chem_geom, chem_ports),
        )
    )

    carrier_transport = CarrierTransport(comp)
    recombination = BulkRecombination(comp)
    chem_transports: dict[str, IodineTransportTerm] = {
        IODIDE: IodineTransportTerm(device, IODIDE),
        IODINE_ATOM: IodineTransportTerm(device, IODINE_ATOM),
        IODINE_MOLECULE: IodineTransportTerm(device, IODINE_MOLECULE),
    }

    terms: list[Term] = [
        GaussDisplacement(comp),
        IodineSpaceChargeTerm(device),
        PotentialContacts(comp),
        carrier_transport,
        CarrierGeneration(comp),
        recombination,
        CarrierStorage(comp),
        IodineCarrierReactionTerm(device),
        IodineSpeciesStorageTerm(device),
        chem_transports[IODIDE],
        chem_transports[IODINE_ATOM],
        chem_transports[IODINE_MOLECULE],
        IodineSpeciesReactionTerm(device),
    ]
    if comp.interface_model is not None:
        if comp.interface_model.has_sheet_charge:
            terms.append(InterfaceGaussPort(comp))
        if comp.interface_model.has_carrier_port:
            terms.append(InterfaceCarrierPort(comp, ELECTRON))
            terms.append(InterfaceCarrierPort(comp, HOLE))

    exchanges: dict[tuple[str, str], CarrierExchange] = {}
    for block, carrier, _ in CARRIERS:
        fixed: list[int] = []
        for side, node in (("left", 0), ("right", n_count - 1)):
            channel = comp.channel(carrier, side)
            if channel.kind == "fixed_reservoir":
                fixed.append(node)
            else:
                exchanges[(block, side)] = CarrierExchange(comp, block, channel, node)
        if fixed:
            terms.append(CarrierReservoirs(comp, block, tuple(fixed)))
    terms.extend(exchanges.values())

    # Boundary escape and charge-conserving contact oxidation ports
    for bspec in (device.left_boundary, device.right_boundary):
        if bspec.mol_escape_velocity_m_s > 0.0:
            terms.append(IodineMoleculeEscapePortTerm(device, bspec))
        if (
            bspec.contact_reaction_mode != "none"
            and bspec.contact_oxidation_rate_m4_s > 0.0
        ):
            terms.append(IodineHoleContactPortTerm(device, bspec))
            terms.append(IodineContactReactionPortTerm(device, bspec, IODIDE))
            if bspec.contact_reaction_mode == "hole_oxidation_to_I0":
                terms.append(
                    IodineContactReactionPortTerm(device, bspec, IODINE_ATOM)
                )

    if constrained_steady:
        if device.is_unreactive_limit:
            terms.append(SingleSpeciesInventoryTerm(device, IODIDE))
            terms.append(SingleSpeciesInventoryTerm(device, IODINE_ATOM))
            terms.append(SingleSpeciesInventoryTerm(device, IODINE_MOLECULE))
        else:
            terms.append(TotalIodineInventoryTerm(device))

    return (
        layout,
        mesh,
        terms,
        carrier_transport,
        recombination,
        exchanges,
        chem_transports,
    )


@dataclass(frozen=True)
class IodineSystem:
    """Reusable bound ``Assembler`` for transient or steady iodine systems."""

    device: CompiledIodineDevice1D
    constrained_steady: bool = False
    inventory_node: int = 0
    assembler: Assembler = field(init=False, repr=False)
    carrier_transport: CarrierTransport = field(init=False, repr=False)
    recombination: BulkRecombination = field(init=False, repr=False)
    exchanges: Mapping[tuple[str, str], CarrierExchange] = field(init=False, repr=False)
    chem_transports: Mapping[str, IodineTransportTerm] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        (
            layout,
            mesh,
            terms,
            carrier_transport,
            recombination,
            exchanges,
            chem_transports,
        ) = _build_layout_and_terms(
            self.device,
            constrained_steady=self.constrained_steady,
            inventory_node=self.inventory_node,
        )
        assembler = Assembler(layout, mesh, IodineFields(self.device), terms)
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "carrier_transport", carrier_transport)
        object.__setattr__(self, "recombination", recombination)
        object.__setattr__(self, "exchanges", exchanges)
        object.__setattr__(self, "chem_transports", chem_transports)

    @property
    def layout(self) -> SystemLayout:
        return self.assembler.layout

    @property
    def identity(self) -> str:
        return self.assembler.identity


def iodine_system(
    device: CompiledIodineDevice1D,
    *,
    constrained_steady: bool = False,
    inventory_node: int = 0,
) -> IodineSystem:
    if not isinstance(device, CompiledIodineDevice1D):
        raise TypeError("iodine_system requires a CompiledIodineDevice1D")
    return IodineSystem(
        device=device,
        constrained_steady=constrained_steady,
        inventory_node=inventory_node,
    )


@dataclass(frozen=True)
class AssembledIodine1DProblem:
    """Unconstrained assembled iodine 1D DAE problem implementing ``StorageProblem``."""

    device: CompiledIodineDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    binding: IodineSystem | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        v = float(self.voltage_V)
        g = float(self.generation_scale)
        if not math.isfinite(v):
            raise ValueError("voltage_V must be finite")
        if not math.isfinite(g) or g < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")
        object.__setattr__(self, "voltage_V", v)
        object.__setattr__(self, "generation_scale", g)
        system = self.binding
        if system is None:
            system = iodine_system(self.device, constrained_steady=False)
        elif system.device is not self.device or system.constrained_steady:
            raise ValueError("binding must match device with constrained_steady=False")
        object.__setattr__(self, "binding", system)

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(
            voltage_V=self.voltage_V,
            generation_scale=self.generation_scale,
        )

    def evaluate_system(
        self, u: ArrayLike, *, derivatives: bool = True, storage: bool = False
    ) -> AssembledSystem:
        assert self.binding is not None
        return self.binding.assembler.evaluate(
            u, self.inputs, derivatives=derivatives, storage=storage
        )

    def residual(self, u: Vector) -> Vector:
        ev = self.evaluate_system(u, derivatives=False, storage=False)
        if not ev.admissible or not np.all(np.isfinite(ev.F)):
            raise FloatingPointError("iodine 1D residual is outside finite domain")
        return ev.F

    def jacobian(self, u: Vector) -> csc_matrix:
        ev = self.evaluate_system(u, derivatives=True, storage=False)
        if (
            not ev.admissible
            or ev.J is None
            or (
                ev.replaced_nonfinite_rows is not None
                and np.any(ev.replaced_nonfinite_rows)
            )
            or not np.all(np.isfinite(ev.J.data))
        ):
            raise FloatingPointError("iodine 1D Jacobian is outside finite domain")
        return ev.J

    def storage(self, u: Vector) -> Vector:
        assert self.binding is not None
        s_val, _ = self.binding.assembler.storage(u, self.inputs, derivatives=False)
        return s_val

    def storage_tangent(self, u: Vector) -> csc_matrix:
        assert self.binding is not None
        _, s_tan = self.binding.assembler.storage(u, self.inputs, derivatives=True)
        assert s_tan is not None
        return s_tan

    def storage_jacobian(self, u: Vector) -> csc_matrix:
        """Signed mass matrix ``M = -S_u`` for small-signal AC conventions."""
        return -self.storage_tangent(u)

    def storage_increment(
        self, u: Vector, old_problem: object, old_u: Vector
    ) -> Vector:
        if not isinstance(old_problem, AssembledIodine1DProblem):
            raise TypeError("storage_increment requires AssembledIodine1DProblem")
        assert self.binding is not None and old_problem.binding is not None
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("storage_increment requires identical system binding")
        return self.binding.assembler.storage_increment(
            u, self.inputs, old_u, old_problem.inputs, physical=False
        )

    def physical_storage(self, u: Vector) -> Vector:
        assert self.binding is not None
        return self.binding.assembler.physical_storage(u, self.inputs)

    def physical_increment(
        self, u: Vector, old_problem: AssembledIodine1DProblem, old_u: Vector
    ) -> Vector:
        assert self.binding is not None and old_problem.binding is not None
        if old_problem.binding.identity != self.binding.identity:
            raise ValueError("physical_increment requires identical system binding")
        return self.binding.assembler.storage_increment(
            u, self.inputs, old_u, old_problem.inputs, physical=True
        )

    def carrier_fluxes(self, u: Vector) -> CarrierFaceFluxes:
        assert self.binding is not None
        ev_state = self.binding.assembler.state(u, self.inputs)
        if ev_state is None:
            raise FloatingPointError("inadmissible state in carrier_fluxes")
        return self.binding.carrier_transport.fluxes(ev_state)

    def species_fluxes_m2_s(self, u: Vector) -> dict[str, Vector]:
        assert self.binding is not None
        ev_state = self.binding.assembler.state(u, self.inputs)
        if ev_state is None:
            raise FloatingPointError("inadmissible state in species_fluxes_m2_s")
        return {
            sp: frozen_array(
                self.binding.chem_transports[sp]
                .flux_evaluation(ev_state, derivatives=False)[0]
            )
            for sp in IODINE_BLOCKS
        }

    def boundary_rates_m2_s(self, u: Vector) -> dict[str, float]:
        """Return outward ``I_2`` escape and contact oxidation rates [m^-2 s^-1]."""
        assert self.binding is not None
        ev_state = self.binding.assembler.state(u, self.inputs)
        if ev_state is None:
            raise FloatingPointError("inadmissible state in boundary_rates_m2_s")
        c2 = ev_state.field(IODINE_MOLECULE)
        left_b = self.device.left_boundary
        right_b = self.device.right_boundary
        esc_left = left_b.mol_escape_velocity_m_s * float(c2[left_b.chem_node])
        esc_right = right_b.mol_escape_velocity_m_s * float(c2[right_b.chem_node])
        ct_left = (
            _contact_oxidation_rate_and_partials(self.device, left_b, ev_state)[0]
            if left_b.contact_reaction_mode != "none"
            and left_b.contact_oxidation_rate_m4_s > 0.0
            else 0.0
        )
        ct_right = (
            _contact_oxidation_rate_and_partials(self.device, right_b, ev_state)[0]
            if right_b.contact_reaction_mode != "none"
            and right_b.contact_oxidation_rate_m4_s > 0.0
            else 0.0
        )
        return {
            "mol_escape_left_m2_s": esc_left,
            "mol_escape_right_m2_s": esc_right,
            "contact_oxidation_left_m2_s": ct_left,
            "contact_oxidation_right_m2_s": ct_right,
        }

    def face_conduction_current_A_m2(self, u: Vector) -> Vector:
        """Total particle conduction current ``J_n + J_p + J_I-`` on every face."""
        cf = self.carrier_fluxes(u)
        j_face = Q * (cf.hole.flux - cf.electron.flux)
        j_iodide_chem = -Q * self.species_fluxes_m2_s(u)[IODIDE]
        if self.device.chem_node_count == self.device.node_count:
            return frozen_array(j_face + j_iodide_chem)
        start = int(self.device.chem_node_indices[0])
        full = j_face.copy()
        full[start : start + j_iodide_chem.size] += j_iodide_chem
        return frozen_array(full)

    def steady_current_density_A_m2(self, u: Vector) -> float:
        """Spatial mean steady conduction current density [A/m^2]."""
        dx = np.diff(self.device.electronic_component.nodes_m)
        j_faces = self.face_conduction_current_A_m2(u)
        return float(math.fsum((dx * j_faces).tolist()) / math.fsum(dx.tolist()))

    def contact_displacement_C_m2(self, u: Vector) -> tuple[float, float]:
        """Left and right electrode Gauss surface charges [C/m^2]."""
        phi, n, p, cm, _, _, _, _, _ = self.device.fields_and_tangents(
            u, voltage_V=self.voltage_V
        )
        rho = self.device.charge_ledger(n, p, cm).space_charge_C_m3()
        comp = self.device.electronic_component
        d_left, d_right = contact_displacement(
            comp.poisson.face_displacement(phi),
            rho,
            comp.geometry.cell_measures_m,
        )
        return float(d_left), float(d_right)

    def recombination_rate(
        self, n: Vector, p: Vector, *, derivatives: bool = True
    ) -> RecombinationRate:
        assert self.binding is not None
        return self.binding.recombination.rate(n, p, derivatives=derivatives)

    def contact_exchange(
        self, u: Vector, channel: CarrierExchangeChannel, node: int
    ) -> tuple[float, float]:
        assert self.binding is not None
        block = ELECTRON if channel.carrier == "n" else HOLE
        term = self.binding.exchanges[(block, channel.side)]
        n_count = self.device.node_count
        offset = n_count if block == ELECTRON else 2 * n_count
        return term.exchange(float(u[node]), float(u[offset + node]), self.voltage_V)


@dataclass(frozen=True)
class AssembledConstrainedIodine1DProblem:
    """Closed-domain steady iodine 1D system enforcing total iodine inventory."""

    device: CompiledIodineDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    inventory_node: int = 0
    binding: IodineSystem | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        v = float(self.voltage_V)
        g = float(self.generation_scale)
        if not math.isfinite(v):
            raise ValueError("voltage_V must be finite")
        if not math.isfinite(g) or g < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")
        object.__setattr__(self, "voltage_V", v)
        object.__setattr__(self, "generation_scale", g)
        system = self.binding
        if system is None:
            system = iodine_system(
                self.device,
                constrained_steady=True,
                inventory_node=self.inventory_node,
            )
        elif (
            system.device is not self.device
            or not system.constrained_steady
            or system.inventory_node != self.inventory_node
        ):
            raise ValueError("binding must match constrained steady configuration")
        object.__setattr__(self, "binding", system)

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(
            voltage_V=self.voltage_V,
            generation_scale=self.generation_scale,
        )

    def evaluate_system(
        self, u: ArrayLike, *, derivatives: bool = True
    ) -> AssembledSystem:
        assert self.binding is not None
        return self.binding.assembler.evaluate(
            u, self.inputs, derivatives=derivatives, storage=False
        )

    def residual(self, u: Vector) -> Vector:
        ev = self.evaluate_system(u, derivatives=False)
        if not ev.admissible or not np.all(np.isfinite(ev.F)):
            raise FloatingPointError("constrained iodine residual outside finite domain")
        return ev.F

    def jacobian(self, u: Vector) -> csc_matrix:
        ev = self.evaluate_system(u, derivatives=True)
        if (
            not ev.admissible
            or ev.J is None
            or (
                ev.replaced_nonfinite_rows is not None
                and np.any(ev.replaced_nonfinite_rows)
            )
            or not np.all(np.isfinite(ev.J.data))
        ):
            raise FloatingPointError(
                "constrained iodine Jacobian outside finite domain"
            )
        return ev.J

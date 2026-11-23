"""Ion-free classical 1D Poisson and carrier continuity.

Coordinates are u = (phi, z_n, z_p), with dimensionless quasi-Fermi offsets
z_n and z_p are measured from each carrier's majority electrode Fermi level.
At zero bias both references equal E_F0. Physical densities
come from one common equilibrium reference and the physical bands. SG faces
consume the differences of z directly, so a constant Fermi level gives an
exactly zero flux even across DOS and band offsets.

The implicit residual is F(u, udot) = transport + source - d(storage)/dt.
Only steady operation is admitted by the production driver. The explicit
storage partials retain that architecture; they are not transient qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags, lil_matrix

from solarlab.contracts import (
    AREA,
    LENGTH,
    PARTICLE,
    SECOND,
    VOLUME,
    CellSource,
    EquationSpec,
    FaceFlux,
    Geometry,
    Layout,
    Support,
    TermSink,
    frozen_array,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.stable_bernoulli import bernoulli_pair
from solarlab.equations.field_transport_1d import FieldTransport1D
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.physics.band_structure.binding import ParameterBinding
from solarlab.physics.contacts import CarrierExchangeChannel, junction_polarity
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.electrostatics.interface_traces import TraceElectrostatics
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    THERMIONIC_MODEL_VERSIONS,
    ThermionicCombinationMode,
    ThermionicModelVersion,
    TwoSidedBulkState,
    TwoSidedInterfacePhysics,
    cap_particle_flux_thermionic,
    combine_sg_and_reciprocal_thermionic_flux,
    reciprocal_thermionic_flux_m2_s,
    solve_charged_two_sided_interface_qss,
    solve_two_sided_interface_qss,
)
from solarlab.physics.recombination.bulk import (
    RecombinationRate,
    sum_recombination_rates,
)
from solarlab.physics.recombination.interface import (
    PlaneClosureParams,
    interface_srh_rate,
    solve_plane_closure,
)
from solarlab.physics.recombination.reaction import BulkRecombinationReaction
from solarlab.physics.transport.electrochemical_flux import (
    BAND_STEP_MODES,
    CarrierFaceFlux,
    CarrierFaceFluxes,
    LEGACY_SUMMED_POTENTIAL_V1,
    carrier_particle_fluxes,
)

__all__ = [
    "CapabilityError",
    "CompiledDevice1D",
    "DC1DProblem",
    "LocalRecombinationModel",
    "State1D",
    "quasi_fermi_potential_tangents",
]

Vector: TypeAlias = NDArray[np.float64]
Q = ELEMENTARY_CHARGE_C
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
assert SOURCE_UNIT * LENGTH == FLUX_UNIT


class CapabilityError(ValueError):
    """An enabled mechanism has no binding in this first 1D DC slice."""


class LocalRecombinationModel(Protocol):
    """An injected local net pair rate and its exact density partials.

    The model owns its bound coefficients and a stable, parameter-specific
    model_id. Its evaluate(n, p) returns production RecombinationRate arrays
    of shape (N,); no file loader, registry, or solver dependency is required.
    """

    model_id: str

    def evaluate(self, n: Vector, p: Vector) -> RecombinationRate: ...


@dataclass(frozen=True)
class State1D:
    """Immutable coordinates and physical fields on one compiled device."""

    u: Vector
    potential_V: Vector
    electron_density_m3: Vector
    hole_density_m3: Vector
    device_sha256: str
    voltage_V: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        n = self.potential_V.size
        if self.potential_V.ndim != 1:
            raise ValueError("state potential must be a vector on the nodes")
        if self.u.shape != (3 * n,):
            raise ValueError("state coordinates must have shape (3 * node_count,)")
        for density in (self.electron_density_m3, self.hole_density_m3):
            if density.shape != (n,) or np.any(density <= 0.0):
                raise ValueError("state densities must be positive on every node")
        if not self.device_sha256:
            raise ValueError("a state must identify its compiled device")
        if not math.isfinite(self.voltage_V):
            raise ValueError("the state coordinate reference voltage must be finite")


VECTOR_EXP_CAP = 40.0
TERegularizationName: TypeAlias = Literal["legacy_logistic", "compact_support"]


@dataclass(frozen=True)
class DeduplicatedBareSRHSpec:
    """Single-node heterointerface SRH specification on a deduplicated grid."""

    interface_index: int
    node: int
    v_n_m_s: float
    v_p_m_s: float
    ni_sq_m6: float
    n1_m3: float
    p1_m3: float


@dataclass(frozen=True)
class DeduplicatedDefectSRHSpec:
    """Cross-carrier or plane-closure heterointerface defect SRH on a deduplicated grid."""

    interface_index: int
    node: int
    v_n_m_s: float
    v_p_m_s: float
    ni_sq_eff_m6: float
    n1_m3: float
    p1_m3: float
    plane_params: PlaneClosureParams | None = None


@dataclass(frozen=True)
class DeduplicatedThermionicFaceSpec:
    """Direction-preserving 1D thermionic emission cap on an internal SG face."""

    interface_index: int
    face: int
    delta_Ec_eV: float
    delta_Ev_eV: float
    a_star_n_A_m2_K2: float
    a_star_p_A_m2_K2: float
    cap_electron: bool
    cap_hole: bool
    a_star_n_right_A_m2_K2: float | None = None
    a_star_p_right_A_m2_K2: float | None = None
    dos_n_left_m3: float | None = None
    dos_n_right_m3: float | None = None
    dos_p_left_m3: float | None = None
    dos_p_right_m3: float | None = None
    hole_band_offset_eV: float | None = None
    transmission: float = 1.0


@dataclass(frozen=True)
class TwoSidedInterfaceBindingSpec:
    """Locally eliminated two-sided trace transport/SRH/sheet-charge binding."""

    interface_index: int
    face: int
    left_node: int
    right_node: int
    trace_electrostatics: TraceElectrostatics
    physics: TwoSidedInterfacePhysics
    charge: EquilibriumReferencedSheetCharge | None = None
    weight_left: float = 0.0
    weight_right: float = 0.0


@dataclass(frozen=True)
class InterfaceEvaluation1D:
    """Single-pass constitutive evaluation of all 1D heterointerface channels."""

    fluxes: CarrierFaceFluxes
    e_left_phi: Vector
    e_right_phi: Vector
    e_left_zn: Vector
    e_right_zn: Vector
    e_left_zp: Vector
    e_right_zp: Vector
    h_left_phi: Vector
    h_right_phi: Vector
    h_left_zn: Vector
    h_right_zn: Vector
    h_left_zp: Vector
    h_right_zp: Vector
    e_port_val: Vector
    e_port_phi: Vector
    e_port_zn: Vector
    e_port_zp: Vector
    h_port_val: Vector
    h_port_phi: Vector
    h_port_zn: Vector
    h_port_zp: Vector
    g_port_val: Vector
    g_port_phi: Vector
    g_port_zn: Vector
    g_port_zp: Vector
    g_left_phi: Vector
    g_right_phi: Vector
    g_left_zn: Vector
    g_right_zn: Vector
    g_left_zp: Vector
    g_right_zp: Vector
    sheet_charges_C_m2: Vector
    occupancies: Vector


def _scatter_three_node_row_tangent(
    idx: int,
    g_left: float,
    g_mid: float,
    g_right: float,
    face_left: Vector,
    face_right: Vector,
    port_diag: Vector,
) -> None:
    """Scatter ``(g_left, g_mid, g_right)`` onto row ``idx`` via face + interior port."""
    if g_left != 0.0:
        face_left[idx - 1] += g_left
        port_diag[idx - 2] += g_left
    if g_mid != 0.0:
        port_diag[idx - 1] += g_mid
    if g_right != 0.0:
        face_right[idx] -= g_right
        port_diag[idx] += g_right


def _scatter_two_sided_carrier_tangent(
    face: int,
    g_ll: float,
    g_lr: float,
    g_rl: float,
    g_rr: float,
    face_left: Vector,
    face_right: Vector,
    port_diag: Vector,
) -> None:
    """Set 2x2 carrier Jacobian ``[[-g_ll, -g_lr], [-g_rl, -g_rr]]`` on ``(L, R)``."""
    face_left[face] = -g_rl
    face_right[face] = g_lr
    port_diag[face - 1] -= g_ll + g_rl
    port_diag[face] -= g_rr + g_lr


@dataclass(frozen=True)
class CompiledInterfaceModel1D:
    """Compiled 1D heterointerface transport, SRH, and sheet-charge closure."""

    topology: Literal["deduplicated", "two_sided_trace"]
    identity_sha256: str
    bare_srh: tuple[DeduplicatedBareSRHSpec, ...] = ()
    defect_srh: tuple[DeduplicatedDefectSRHSpec, ...] = ()
    thermionic_faces: tuple[DeduplicatedThermionicFaceSpec, ...] = ()
    two_sided: tuple[TwoSidedInterfaceBindingSpec, ...] = ()
    plane_projection: bool = False
    plane_closure: bool = False
    plane_generation: bool = False
    te_softness: float = 0.02
    te_regularization_mode: TERegularizationName = "legacy_logistic"
    thermionic_mode: ThermionicModelVersion = LEGACY_UNNORMALIZED_THERMIONIC_V1
    combination_mode: ThermionicCombinationMode | None = None

    def __post_init__(self) -> None:
        if self.thermionic_mode not in THERMIONIC_MODEL_VERSIONS:
            raise ValueError(f"unknown thermionic_mode {self.thermionic_mode!r}")
        if self.combination_mode is not None and self.combination_mode not in (
            "harmonic_series",
            "compact_support",
            "legacy_logistic",
            "hard",
        ):
            raise ValueError(f"unknown combination_mode {self.combination_mode!r}")

    @property
    def has_carrier_port(self) -> bool:
        return bool(self.bare_srh or self.defect_srh or self.two_sided)

    @property
    def has_sheet_charge(self) -> bool:
        return any(spec.charge is not None for spec in self.two_sided)

    def evaluate_state(
        self,
        device: CompiledDevice1D,
        u: Vector,
        phi: Vector,
        n: Vector,
        p: Vector,
        *,
        derivatives: bool = True,
    ) -> InterfaceEvaluation1D:
        """Evaluate all active heterointerface laws in a single constitutive pass."""
        count = device.node_count
        thermal = device.thermal_voltage_V
        temp = device.parameters.temperature_K
        electron_diffusivity = device.electron_diffusivity_faces_m2_s
        hole_diffusivity = device.hole_diffusivity_faces_m2_s
        transport = None if device.field_transport is None else device.field_transport.evaluate(phi)
        if transport is not None:
            electron_diffusivity = transport.electron_diffusivity_m2_s
            hole_diffusivity = transport.hole_diffusivity_m2_s

        zn = u[count:2 * count]
        zp = u[2 * count:]
        ec_n_steps = np.diff(zn)
        ec_p_steps = -np.diff(zp)
        base = carrier_particle_fluxes(
            device.nodes_m,
            phi,
            n,
            p,
            electron_diffusivity,
            hole_diffusivity,
            thermal,
            device.parameters.band_structure,
            electron_electrochemical_step=ec_n_steps,
            hole_electrochemical_step=ec_p_steps,
            band_step_mode=device.band_step_mode,
        )
        e_flux = np.array(base.electron.flux, dtype=float, copy=True)
        e_ld = np.array(base.electron.left_density_derivative, dtype=float, copy=True)
        e_rd = np.array(base.electron.right_density_derivative, dtype=float, copy=True)
        e_lpot = np.array(base.electron.left_potential_derivative, dtype=float, copy=True)
        e_rpot = np.array(base.electron.right_potential_derivative, dtype=float, copy=True)
        h_flux = np.array(base.hole.flux, dtype=float, copy=True)
        h_ld = np.array(base.hole.left_density_derivative, dtype=float, copy=True)
        h_rd = np.array(base.hole.right_density_derivative, dtype=float, copy=True)
        h_lpot = np.array(base.hole.left_potential_derivative, dtype=float, copy=True)
        h_rpot = np.array(base.hole.right_potential_derivative, dtype=float, copy=True)

        face_count = count - 1
        interior_count = count - 2
        e_port_val: Vector = np.zeros(interior_count, dtype=float)
        h_port_val: Vector = np.zeros(interior_count, dtype=float)
        g_port_val: Vector = np.zeros(interior_count, dtype=float)

        if derivatives:
            e_left_phi, e_right_phi = quasi_fermi_potential_tangents(
                base.electron, np.diff(zn), 1.0, thermal
            )
            h_left_phi, h_right_phi = quasi_fermi_potential_tangents(
                base.hole, -np.diff(zp), -1.0, thermal
            )
            e_left_phi = np.array(e_left_phi, dtype=float, copy=True)
            e_right_phi = np.array(e_right_phi, dtype=float, copy=True)
            h_left_phi = np.array(h_left_phi, dtype=float, copy=True)
            h_right_phi = np.array(h_right_phi, dtype=float, copy=True)
            if transport is not None:
                dx = np.diff(device.nodes_m)
                e_chain = (e_flux / transport.electron_diffusivity_m2_s) * transport.electron_d_diffusivity_d_field / dx
                h_chain = (h_flux / transport.hole_diffusivity_m2_s) * transport.hole_d_diffusivity_d_field / dx
                e_left_phi = e_left_phi + e_chain
                e_right_phi = e_right_phi - e_chain
                h_left_phi = h_left_phi + h_chain
                h_right_phi = h_right_phi - h_chain
            e_left_zn = np.array(e_ld * n[:-1], dtype=float, copy=True)
            e_right_zn = np.array(e_rd * n[1:], dtype=float, copy=True)
            e_left_zp: Vector = np.zeros(face_count, dtype=float)
            e_right_zp: Vector = np.zeros(face_count, dtype=float)
            h_left_zn: Vector = np.zeros(face_count, dtype=float)
            h_right_zn: Vector = np.zeros(face_count, dtype=float)
            h_left_zp = np.array(-h_ld * p[:-1], dtype=float, copy=True)
            h_right_zp = np.array(-h_rd * p[1:], dtype=float, copy=True)
            g_left_phi = np.array(device.poisson.face_capacitance_F_m2, dtype=float, copy=True)
            g_right_phi = -np.array(device.poisson.face_capacitance_F_m2, dtype=float, copy=True)
        else:
            e_left_phi = e_right_phi = e_left_zn = e_right_zn = e_left_zp = e_right_zp = np.empty(0)
            h_left_phi = h_right_phi = h_left_zn = h_right_zn = h_left_zp = h_right_zp = np.empty(0)
            g_left_phi = g_right_phi = np.empty(0)

        e_port_phi: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        e_port_zn: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        e_port_zp: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        h_port_phi: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        h_port_zn: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        h_port_zp: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        g_port_phi: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        g_port_zn: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        g_port_zp: Vector = np.zeros(interior_count, dtype=float) if derivatives else np.empty(0)
        g_left_zn: Vector = np.zeros(face_count, dtype=float) if derivatives else np.empty(0)
        g_right_zn: Vector = np.zeros(face_count, dtype=float) if derivatives else np.empty(0)
        g_left_zp: Vector = np.zeros(face_count, dtype=float) if derivatives else np.empty(0)
        g_right_zp: Vector = np.zeros(face_count, dtype=float) if derivatives else np.empty(0)

        iface_count = len(device.geometry.interfaces)
        sheet_charges: Vector = np.zeros(iface_count, dtype=float)
        occupancies: Vector = np.full(iface_count, np.nan, dtype=float)

        if self.topology == "deduplicated":
            comb_mode: ThermionicCombinationMode = (
                "harmonic_series" if self.combination_mode is None else self.combination_mode
            )
            for tf in self.thermionic_faces:
                f = tf.face
                if self.thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1:
                    if tf.cap_electron:
                        cap_e = cap_particle_flux_thermionic(
                            float(e_flux[f]),
                            n_left_m3=float(n[f]),
                            n_right_m3=float(n[f + 1]),
                            delta_barrier_eV=tf.delta_Ec_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_n_A_m2_K2,
                            te_softness=self.te_softness,
                            regularization_mode=self.te_regularization_mode,
                        )
                        e_flux[f] = cap_e.flux_m2_s
                        e_ld[f] = cap_e.d_flux_d_sg_flux * e_ld[f] + cap_e.d_flux_d_n_left_m_s
                        e_rd[f] = cap_e.d_flux_d_sg_flux * e_rd[f] + cap_e.d_flux_d_n_right_m_s
                        if derivatives:
                            e_left_phi[f] = (
                                cap_e.d_flux_d_sg_flux * e_left_phi[f]
                                + cap_e.d_flux_d_n_left_m_s * float(n[f]) / thermal
                            )
                            e_right_phi[f] = (
                                cap_e.d_flux_d_sg_flux * e_right_phi[f]
                                + cap_e.d_flux_d_n_right_m_s * float(n[f + 1]) / thermal
                            )
                            e_left_zn[f] = (
                                cap_e.d_flux_d_sg_flux * e_left_zn[f]
                                + cap_e.d_flux_d_n_left_m_s * float(n[f])
                            )
                            e_right_zn[f] = (
                                cap_e.d_flux_d_sg_flux * e_right_zn[f]
                                + cap_e.d_flux_d_n_right_m_s * float(n[f + 1])
                            )
                    if tf.cap_hole:
                        cap_h = cap_particle_flux_thermionic(
                            float(h_flux[f]),
                            n_left_m3=float(p[f]),
                            n_right_m3=float(p[f + 1]),
                            delta_barrier_eV=tf.delta_Ev_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_p_A_m2_K2,
                            te_softness=self.te_softness,
                            regularization_mode=self.te_regularization_mode,
                        )
                        h_flux[f] = cap_h.flux_m2_s
                        h_ld[f] = cap_h.d_flux_d_sg_flux * h_ld[f] + cap_h.d_flux_d_n_left_m_s
                        h_rd[f] = cap_h.d_flux_d_sg_flux * h_rd[f] + cap_h.d_flux_d_n_right_m_s
                        if derivatives:
                            h_left_phi[f] = (
                                cap_h.d_flux_d_sg_flux * h_left_phi[f]
                                - cap_h.d_flux_d_n_left_m_s * float(p[f]) / thermal
                            )
                            h_right_phi[f] = (
                                cap_h.d_flux_d_sg_flux * h_right_phi[f]
                                - cap_h.d_flux_d_n_right_m_s * float(p[f + 1]) / thermal
                            )
                            h_left_zp[f] = (
                                cap_h.d_flux_d_sg_flux * h_left_zp[f]
                                - cap_h.d_flux_d_n_left_m_s * float(p[f])
                            )
                            h_right_zp[f] = (
                                cap_h.d_flux_d_sg_flux * h_right_zp[f]
                                - cap_h.d_flux_d_n_right_m_s * float(p[f + 1])
                            )
                else:
                    if tf.cap_electron:
                        te_e = reciprocal_thermionic_flux_m2_s(
                            float(n[f]),
                            float(n[f + 1]),
                            carrier="electron",
                            phi_left_V=float(phi[f]),
                            phi_right_V=float(phi[f + 1]),
                            band_offset_eV=tf.delta_Ec_eV,
                            temperature_K=temp,
                            a_star_left_A_m2_K2=tf.a_star_n_A_m2_K2,
                            a_star_right_A_m2_K2=tf.a_star_n_right_A_m2_K2,
                            dos_left_m3=tf.dos_n_left_m3,
                            dos_right_m3=tf.dos_n_right_m3,
                            transmission=tf.transmission,
                            statistics="maxwell_boltzmann",
                            model_version=self.thermionic_mode,
                            thermal_voltage_V=thermal,
                            electrochemical_step=float(ec_n_steps[f]),
                        )
                        comb_e = combine_sg_and_reciprocal_thermionic_flux(
                            float(e_flux[f]),
                            d_sg_d_phi_left_m2_s_V=float(e_lpot[f]),
                            d_sg_d_phi_right_m2_s_V=float(e_rpot[f]),
                            d_sg_d_c_left_m_s=float(e_ld[f]),
                            d_sg_d_c_right_m_s=float(e_rd[f]),
                            c_left_m3=float(n[f]),
                            c_right_m3=float(n[f + 1]),
                            thermionic=te_e,
                            combination_mode=comb_mode,
                            te_softness=self.te_softness,
                        )
                        e_flux[f] = comb_e.flux_m2_s
                        e_ld[f] = comb_e.d_flux_d_c_left_m_s
                        e_rd[f] = comb_e.d_flux_d_c_right_m_s
                        e_lpot[f] = comb_e.d_flux_d_phi_left_m2_s_V
                        e_rpot[f] = comb_e.d_flux_d_phi_right_m2_s_V
                        if derivatives:
                            te_qf_phi_l = (te_e.right_barrier_weight / thermal) * te_e.flux_m2_s
                            te_qf_phi_r = (te_e.left_barrier_weight / thermal) * te_e.flux_m2_s
                            e_left_phi[f] = comb_e.sg_weight * e_left_phi[f] + comb_e.te_weight * te_qf_phi_l
                            e_right_phi[f] = comb_e.sg_weight * e_right_phi[f] + comb_e.te_weight * te_qf_phi_r
                            e_left_zn[f] = comb_e.d_flux_d_log_c_left_m2_s
                            e_right_zn[f] = comb_e.d_flux_d_log_c_right_m2_s
                    if tf.cap_hole:
                        h_offset = (
                            -tf.delta_Ev_eV if tf.hole_band_offset_eV is None else tf.hole_band_offset_eV
                        )
                        te_h = reciprocal_thermionic_flux_m2_s(
                            float(p[f]),
                            float(p[f + 1]),
                            carrier="hole",
                            phi_left_V=float(phi[f]),
                            phi_right_V=float(phi[f + 1]),
                            band_offset_eV=h_offset,
                            temperature_K=temp,
                            a_star_left_A_m2_K2=tf.a_star_p_A_m2_K2,
                            a_star_right_A_m2_K2=tf.a_star_p_right_A_m2_K2,
                            dos_left_m3=tf.dos_p_left_m3,
                            dos_right_m3=tf.dos_p_right_m3,
                            transmission=tf.transmission,
                            statistics="maxwell_boltzmann",
                            model_version=self.thermionic_mode,
                            thermal_voltage_V=thermal,
                            electrochemical_step=float(ec_p_steps[f]),
                        )
                        comb_h = combine_sg_and_reciprocal_thermionic_flux(
                            float(h_flux[f]),
                            d_sg_d_phi_left_m2_s_V=float(h_lpot[f]),
                            d_sg_d_phi_right_m2_s_V=float(h_rpot[f]),
                            d_sg_d_c_left_m_s=float(h_ld[f]),
                            d_sg_d_c_right_m_s=float(h_rd[f]),
                            c_left_m3=float(p[f]),
                            c_right_m3=float(p[f + 1]),
                            thermionic=te_h,
                            combination_mode=comb_mode,
                            te_softness=self.te_softness,
                        )
                        h_flux[f] = comb_h.flux_m2_s
                        h_ld[f] = comb_h.d_flux_d_c_left_m_s
                        h_rd[f] = comb_h.d_flux_d_c_right_m_s
                        h_lpot[f] = comb_h.d_flux_d_phi_left_m2_s_V
                        h_rpot[f] = comb_h.d_flux_d_phi_right_m2_s_V
                        if derivatives:
                            te_qf_phi_l = -(te_h.right_barrier_weight / thermal) * te_h.flux_m2_s
                            te_qf_phi_r = -(te_h.left_barrier_weight / thermal) * te_h.flux_m2_s
                            h_left_phi[f] = comb_h.sg_weight * h_left_phi[f] + comb_h.te_weight * te_qf_phi_l
                            h_right_phi[f] = comb_h.sg_weight * h_right_phi[f] + comb_h.te_weight * te_qf_phi_r
                            h_left_zp[f] = -comb_h.d_flux_d_log_c_left_m2_s
                            h_right_zp[f] = -comb_h.d_flux_d_log_c_right_m2_s

            for bare in self.bare_srh:
                idx = bare.node
                srh = interface_srh_rate(
                    float(n[idx]),
                    float(p[idx]),
                    v_n_m_s=bare.v_n_m_s,
                    v_p_m_s=bare.v_p_m_s,
                    ni_sq_m6=bare.ni_sq_m6,
                    n1_m3=bare.n1_m3,
                    p1_m3=bare.p1_m3,
                    clamp_nonnegative=False,
                )
                sink = -float(np.real(srh.rate_m2_s))
                e_port_val[idx - 1] += sink
                h_port_val[idx - 1] += sink
                if derivatives:
                    s_zn = -float(np.real(srh.d_rate_dn_m_s)) * float(n[idx])
                    s_zp = float(np.real(srh.d_rate_dp_m_s)) * float(p[idx])
                    s_phi = (s_zn + s_zp) / thermal
                    e_port_phi[idx - 1] += s_phi
                    e_port_zn[idx - 1] += s_zn
                    e_port_zp[idx - 1] += s_zp
                    h_port_phi[idx - 1] += s_phi
                    h_port_zn[idx - 1] += s_zn
                    h_port_zp[idx - 1] += s_zp

            for dspec in self.defect_srh:
                idx = dspec.node
                i_l, i_0, i_r = idx - 1, idx, idx + 1
                if self.plane_closure and dspec.plane_params is not None:
                    el_raw = (float(phi[i_0]) - float(phi[i_l])) / thermal
                    er_raw = (float(phi[i_0]) - float(phi[i_r])) / thermal
                    el = max(-VECTOR_EXP_CAP, min(VECTOR_EXP_CAP, el_raw))
                    er = max(-VECTOR_EXP_CAP, min(VECTOR_EXP_CAP, er_raw))
                    fl = math.exp(el)
                    fr = math.exp(er)
                    nl_p = float(n[i_l]) * fl
                    nr_p = float(n[i_r]) * fr
                    pl_p = float(p[i_l]) / fl
                    pr_p = float(p[i_r]) / fr
                    plane_res = solve_plane_closure(
                        nl_p,
                        nr_p,
                        pl_p,
                        pr_p,
                        dspec.plane_params,
                        dspec.v_n_m_s,
                        dspec.v_p_m_s,
                        allow_generation=self.plane_generation,
                    )
                    sink = -plane_res.rate_m2_s
                    e_port_val[idx - 1] += sink
                    h_port_val[idx - 1] += sink
                    if derivatives:
                        b_nl = plane_res.d_rate_d_n_left_proj_m_s * nl_p
                        b_nr = plane_res.d_rate_d_n_right_proj_m_s * nr_p
                        b_pl = plane_res.d_rate_d_p_left_proj_m_s * pl_p
                        b_pr = plane_res.d_rate_d_p_right_proj_m_s * pr_p
                        chi_l = 1.0 if -VECTOR_EXP_CAP < el_raw < VECTOR_EXP_CAP else 0.0
                        chi_r = 1.0 if -VECTOR_EXP_CAP < er_raw < VECTOR_EXP_CAP else 0.0
                        s_phi_l = -((1.0 - chi_l) * (b_nl - b_pl)) / thermal
                        s_phi_0 = -(chi_l * (b_nl - b_pl) + chi_r * (b_nr - b_pr)) / thermal
                        s_phi_r = -((1.0 - chi_r) * (b_nr - b_pr)) / thermal
                        s_zn_l = -b_nl
                        s_zn_r = -b_nr
                        s_zp_l = b_pl
                        s_zp_r = b_pr
                        for f_l, f_r, p_d, gl, gm, gr in (
                            (e_left_phi, e_right_phi, e_port_phi, s_phi_l, s_phi_0, s_phi_r),
                            (h_left_phi, h_right_phi, h_port_phi, s_phi_l, s_phi_0, s_phi_r),
                            (e_left_zn, e_right_zn, e_port_zn, s_zn_l, 0.0, s_zn_r),
                            (h_left_zn, h_right_zn, h_port_zn, s_zn_l, 0.0, s_zn_r),
                            (e_left_zp, e_right_zp, e_port_zp, s_zp_l, 0.0, s_zp_r),
                            (h_left_zp, h_right_zp, h_port_zp, s_zp_l, 0.0, s_zp_r),
                        ):
                            _scatter_three_node_row_tangent(idx, gl, gm, gr, f_l, f_r, p_d)
                else:
                    n_raw = float(n[i_r])
                    p_raw = float(p[i_l])
                    if self.plane_projection:
                        en_raw = (float(phi[i_0]) - float(phi[i_r])) / thermal
                        ep_raw = (float(phi[i_0]) - float(phi[i_l])) / thermal
                        en = max(-VECTOR_EXP_CAP, min(VECTOR_EXP_CAP, en_raw))
                        ep = max(-VECTOR_EXP_CAP, min(VECTOR_EXP_CAP, ep_raw))
                        fac_n = math.exp(en)
                        fac_p = math.exp(-ep)
                        n_eval = n_raw * fac_n
                        p_eval = p_raw * fac_p
                        ni_eval = dspec.ni_sq_eff_m6 * fac_n * fac_p
                    else:
                        en_raw = ep_raw = 0.0
                        n_eval = n_raw
                        p_eval = p_raw
                        ni_eval = dspec.ni_sq_eff_m6
                    srh = interface_srh_rate(
                        n_eval,
                        p_eval,
                        v_n_m_s=dspec.v_n_m_s,
                        v_p_m_s=dspec.v_p_m_s,
                        ni_sq_m6=ni_eval,
                        n1_m3=dspec.n1_m3,
                        p1_m3=dspec.p1_m3,
                        clamp_nonnegative=True,
                    )
                    sink = -float(np.real(srh.rate_m2_s))
                    e_port_val[idx - 1] += sink
                    h_port_val[idx - 1] += sink
                    if derivatives:
                        a_n = float(np.real(srh.d_rate_dn_m_s)) * n_eval
                        a_p = float(np.real(srh.d_rate_dp_m_s)) * p_eval
                        s_zn_r = -a_n
                        s_zp_l = a_p
                        if self.plane_projection:
                            chi_en = 1.0 if -VECTOR_EXP_CAP < en_raw < VECTOR_EXP_CAP else 0.0
                            chi_ep = 1.0 if -VECTOR_EXP_CAP < ep_raw < VECTOR_EXP_CAP else 0.0
                            denom = dspec.v_n_m_s * (n_eval + dspec.n1_m3) + dspec.v_p_m_s * (p_eval + dspec.p1_m3)
                            unclamped = (
                                dspec.v_n_m_s > 0.0
                                and dspec.v_p_m_s > 0.0
                                and denom > 1.0e-300
                                and (n_eval * p_eval - ni_eval) >= 0.0
                            )
                            a_ni = -(dspec.v_n_m_s * dspec.v_p_m_s / denom) * ni_eval if unclamped else 0.0
                            s_phi_l = -((-a_p * (1.0 - chi_ep) + a_ni * chi_ep) / thermal)
                            s_phi_0 = -(((a_n + a_ni) * chi_en - (a_p + a_ni) * chi_ep) / thermal)
                            s_phi_r = -((a_n * (1.0 - chi_en) - a_ni * chi_en) / thermal)
                        else:
                            s_phi_l = a_p / thermal
                            s_phi_0 = 0.0
                            s_phi_r = -a_n / thermal
                        for f_l, f_r, p_d, gl, gm, gr in (
                            (e_left_phi, e_right_phi, e_port_phi, s_phi_l, s_phi_0, s_phi_r),
                            (h_left_phi, h_right_phi, h_port_phi, s_phi_l, s_phi_0, s_phi_r),
                            (e_left_zn, e_right_zn, e_port_zn, 0.0, 0.0, s_zn_r),
                            (h_left_zn, h_right_zn, h_port_zn, 0.0, 0.0, s_zn_r),
                            (e_left_zp, e_right_zp, e_port_zp, s_zp_l, 0.0, 0.0),
                            (h_left_zp, h_right_zp, h_port_zp, s_zp_l, 0.0, 0.0),
                        ):
                            _scatter_three_node_row_tangent(idx, gl, gm, gr, f_l, f_r, p_d)

        else:
            for tspec in self.two_sided:
                k = tspec.interface_index
                f = tspec.face
                l_node = tspec.left_node
                r_node = tspec.right_node
                bulk = TwoSidedBulkState(
                    phi_left_V=float(phi[l_node]),
                    phi_right_V=float(phi[r_node]),
                    n_left_m3=float(n[l_node]),
                    p_left_m3=float(p[l_node]),
                    n_right_m3=float(n[r_node]),
                    p_right_m3=float(p[r_node]),
                    electron_electrochemical_step=float(zn[r_node] - zn[l_node]),
                    hole_electrochemical_step=float(-(zp[r_node] - zp[l_node])),
                )
                if tspec.charge is not None:
                    charged = solve_charged_two_sided_interface_qss(
                        tspec.trace_electrostatics,
                        tspec.physics,
                        bulk,
                        tspec.charge,
                    )
                    bflux = charged.bulk_flux_m2_s
                    j_bulk = charged.bulk_flux_jacobian_bulk
                    sigma = charged.incremental_sheet_charge_C_m2
                    j_sigma = charged.sheet_charge_jacobian_bulk
                    sheet_charges[k] = sigma
                    occupancies[k] = charged.occupancy
                    g_port_val[l_node - 1] += tspec.weight_left * sigma
                    g_port_val[r_node - 1] += tspec.weight_right * sigma
                    if derivatives:
                        s_phi_l = float(j_sigma[0] + (j_sigma[2] - j_sigma[3]) / thermal)
                        s_phi_r = float(j_sigma[1] + (j_sigma[4] - j_sigma[5]) / thermal)
                        s_zn_l = float(j_sigma[2])
                        s_zn_r = float(j_sigma[4])
                        s_zp_l = float(-j_sigma[3])
                        s_zp_r = float(-j_sigma[5])
                        wl, wr = tspec.weight_left, tspec.weight_right
                        w_sum = wl + wr
                        g_left_phi[f] += wr * s_phi_l
                        g_right_phi[f] -= wl * s_phi_r
                        g_port_phi[l_node - 1] += w_sum * s_phi_l
                        g_port_phi[r_node - 1] += w_sum * s_phi_r
                        g_left_zn[f] += wr * s_zn_l
                        g_right_zn[f] -= wl * s_zn_r
                        g_port_zn[l_node - 1] += w_sum * s_zn_l
                        g_port_zn[r_node - 1] += w_sum * s_zn_r
                        g_left_zp[f] += wr * s_zp_l
                        g_right_zp[f] -= wl * s_zp_r
                        g_port_zp[l_node - 1] += w_sum * s_zp_l
                        g_port_zp[r_node - 1] += w_sum * s_zp_r
                else:
                    uncharged = solve_two_sided_interface_qss(
                        tspec.trace_electrostatics,
                        tspec.physics,
                        bulk,
                    )
                    bflux = uncharged.bulk_flux_m2_s
                    j_bulk = uncharged.bulk_flux_jacobian_bulk
                    if uncharged.occupancy is not None:
                        occupancies[k] = uncharged.occupancy

                e_flux[f] = float(bflux[0])
                h_flux[f] = float(bflux[1])
                e_ld[f] = float(j_bulk[0, 2]) / float(n[l_node])
                e_rd[f] = float(j_bulk[0, 4]) / float(n[r_node])
                h_ld[f] = float(j_bulk[1, 3]) / float(p[l_node])
                h_rd[f] = float(j_bulk[1, 5]) / float(p[r_node])
                e_port_val[r_node - 1] -= float(bflux[0] + bflux[2])
                h_port_val[r_node - 1] -= float(bflux[1] + bflux[3])

                if derivatives:
                    for row_l, row_r, fl_phi, fr_phi, pd_phi, fl_zn, fr_zn, pd_zn, fl_zp, fr_zp, pd_zp in (
                        (0, 2, e_left_phi, e_right_phi, e_port_phi, e_left_zn, e_right_zn, e_port_zn, e_left_zp, e_right_zp, e_port_zp),
                        (1, 3, h_left_phi, h_right_phi, h_port_phi, h_left_zn, h_right_zn, h_port_zn, h_left_zp, h_right_zp, h_port_zp),
                    ):
                        gl_phi = float(j_bulk[row_l, 0] + (j_bulk[row_l, 2] - j_bulk[row_l, 3]) / thermal)
                        gr_phi = float(j_bulk[row_l, 1] + (j_bulk[row_l, 4] - j_bulk[row_l, 5]) / thermal)
                        hl_phi = float(j_bulk[row_r, 0] + (j_bulk[row_r, 2] - j_bulk[row_r, 3]) / thermal)
                        hr_phi = float(j_bulk[row_r, 1] + (j_bulk[row_r, 4] - j_bulk[row_r, 5]) / thermal)
                        _scatter_two_sided_carrier_tangent(f, gl_phi, gr_phi, hl_phi, hr_phi, fl_phi, fr_phi, pd_phi)
                        _scatter_two_sided_carrier_tangent(
                            f,
                            float(j_bulk[row_l, 2]),
                            float(j_bulk[row_l, 4]),
                            float(j_bulk[row_r, 2]),
                            float(j_bulk[row_r, 4]),
                            fl_zn,
                            fr_zn,
                            pd_zn,
                        )
                        _scatter_two_sided_carrier_tangent(
                            f,
                            float(-j_bulk[row_l, 3]),
                            float(-j_bulk[row_l, 5]),
                            float(-j_bulk[row_r, 3]),
                            float(-j_bulk[row_r, 5]),
                            fl_zp,
                            fr_zp,
                            pd_zp,
                        )

        replaced_fluxes = CarrierFaceFluxes(
            electron=replace(
                base.electron,
                flux=e_flux,
                left_density_derivative=e_ld,
                right_density_derivative=e_rd,
                left_potential_derivative=e_lpot,
                right_potential_derivative=e_rpot,
            ),
            hole=replace(
                base.hole,
                flux=h_flux,
                left_density_derivative=h_ld,
                right_density_derivative=h_rd,
                left_potential_derivative=h_lpot,
                right_potential_derivative=h_rpot,
            ),
        )
        return InterfaceEvaluation1D(
            fluxes=replaced_fluxes,
            e_left_phi=e_left_phi,
            e_right_phi=e_right_phi,
            e_left_zn=e_left_zn,
            e_right_zn=e_right_zn,
            e_left_zp=e_left_zp,
            e_right_zp=e_right_zp,
            h_left_phi=h_left_phi,
            h_right_phi=h_right_phi,
            h_left_zn=h_left_zn,
            h_right_zn=h_right_zn,
            h_left_zp=h_left_zp,
            h_right_zp=h_right_zp,
            e_port_val=e_port_val,
            e_port_phi=e_port_phi,
            e_port_zn=e_port_zn,
            e_port_zp=e_port_zp,
            h_port_val=h_port_val,
            h_port_phi=h_port_phi,
            h_port_zn=h_port_zn,
            h_port_zp=h_port_zp,
            g_port_val=g_port_val,
            g_port_phi=g_port_phi,
            g_port_zn=g_port_zn,
            g_port_zp=g_port_zp,
            g_left_phi=g_left_phi,
            g_right_phi=g_right_phi,
            g_left_zn=g_left_zn,
            g_right_zn=g_right_zn,
            g_left_zp=g_left_zp,
            g_right_zp=g_right_zp,
            sheet_charges_C_m2=sheet_charges,
            occupancies=occupancies,
        )


@dataclass(frozen=True)
class CompiledDevice1D:
    """Cold-compiled geometry, physical models, and contact reservoirs."""

    device_id: str
    sha256: str
    geometry: ControlVolumeGeometry1D
    parameters: ParameterBinding
    poisson: PoissonOperator1D
    electron_diffusivity_faces_m2_s: Vector
    hole_diffusivity_faces_m2_s: Vector
    generation: GenerationProfile
    recombination: BulkRecombinationReaction
    channels: tuple[CarrierExchangeChannel, ...]
    phi_left_V: float
    built_in_potential_V: float
    fermi_level_eV: float
    reference_potential_V: Vector
    reference_electron_density_m3: Vector
    reference_hole_density_m3: Vector
    additional_recombination: LocalRecombinationModel | None = None
    capabilities: tuple[str, ...] = (
        "ion_free",
        "classical_1d_dc",
        "maxwell_boltzmann",
        "semiconductor_work_function_contacts",
        "bulk_srh_radiative_auger",
    )
    field_transport: FieldTransport1D | None = None
    interface_model: CompiledInterfaceModel1D | None = None
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1
    _carrier_layout: Layout = field(init=False, repr=False)
    _carrier_geometry: Geometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.band_step_mode not in BAND_STEP_MODES:
            raise ValueError(f"unknown band_step_mode {self.band_step_mode!r}")
        for name in (
            "electron_diffusivity_faces_m2_s",
            "hole_diffusivity_faces_m2_s",
            "reference_potential_V",
            "reference_electron_density_m3",
            "reference_hole_density_m3",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        count = self.node_count
        for name in (
            "reference_potential_V",
            "reference_electron_density_m3",
            "reference_hole_density_m3",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} does not match the geometry")
        for name in ("electron_diffusivity_faces_m2_s", "hole_diffusivity_faces_m2_s"):
            if getattr(self, name).shape != (count - 1,):
                raise ValueError(f"{name} does not match the geometry faces")
        transport = self.field_transport
        if transport is not None and (
            not np.array_equal(transport.nodes_m, self.nodes_m)
            or transport.thermal_voltage_V != self.thermal_voltage_V
            or not np.array_equal(transport.base_electron_diffusivity_m2_s, self.electron_diffusivity_faces_m2_s)
            or not np.array_equal(transport.base_hole_diffusivity_m2_s, self.hole_diffusivity_faces_m2_s)
        ):
            raise ValueError("field transport must bind this device's grid, temperature and low-field diffusivities")
        for name in ("reference_electron_density_m3", "reference_hole_density_m3"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError("all reference densities must be finite and positive")
        if self.poisson.node_count != count or self.recombination.node_count != count:
            raise ValueError("all compiled operators must share the node count")
        expected = {(carrier, side) for carrier in ("n", "p") for side in ("left", "right")}
        if {(channel.carrier, channel.side) for channel in self.channels} != expected:
            raise ValueError("the compiled device needs all four contact channels")
        if len(self.channels) != 4:
            raise ValueError("duplicate contact channel")
        self.generation.require_inputs(self.generation.binding_sha256, self.geometry)
        if self.additional_recombination is not None:
            if not isinstance(self.additional_recombination.model_id, str) or not self.additional_recombination.model_id:
                raise ValueError("an additional recombination law needs a model_id")
        layout = Layout(
            (Support("cells", "cell", (count,)), Support("faces", "face", (count - 1,))),
            (),
            (
                EquationSpec("electron", "electron", "cells", (count,), FLUX_UNIT),
                EquationSpec("hole", "hole", "cells", (count,), FLUX_UNIT),
            ),
        )
        object.__setattr__(self, "_carrier_layout", layout)
        object.__setattr__(self, "_carrier_geometry", self.geometry.to_contract_geometry())

    @property
    def node_count(self) -> int:
        return self.geometry.node_count

    @property
    def identity(self) -> str:
        """Identity of the actual geometry and physical/model bindings."""
        return self.sha256

    @property
    def nodes_m(self) -> Vector:
        return self.geometry.nodes_m

    @property
    def thermal_voltage_V(self) -> float:
        return self.parameters.thermal_voltage_V

    @property
    def voltage_polarity(self) -> float:
        return junction_polarity(self.built_in_potential_V)

    @property
    def generation_limit_A_m2(self) -> float:
        return Q * self.generation.areal_generation_m2_s

    def contact_potentials(self, voltage_V: float) -> tuple[float, float]:
        voltage = float(voltage_V)
        if not math.isfinite(voltage):
            raise ValueError("applied voltage must be finite")
        return (
            self.phi_left_V,
            self.phi_left_V + self.built_in_potential_V - self.voltage_polarity * voltage,
        )

    def channel(self, carrier: str, side: str) -> CarrierExchangeChannel:
        return next(c for c in self.channels if c.carrier == carrier and c.side == side)

    def carrier_reference_energies(self, voltage_V: float) -> tuple[float, float]:
        """Majority-electrode Fermi offsets from the common dark reference."""
        bias = self.voltage_polarity * float(voltage_V)
        return (bias, 0.0) if self.voltage_polarity > 0.0 else (0.0, bias)

    def carrier_reference_potentials(self, voltage_V: float) -> tuple[Vector, Vector]:
        """Shift before subtraction so the majority-contact exponent is exact zero."""
        electron, hole = self.carrier_reference_energies(voltage_V)
        return self.reference_potential_V - electron, self.reference_potential_V - hole

    def rebase_coordinates(self, u: ArrayLike, old_voltage_V: float, new_voltage_V: float) -> Vector:
        """Preserve the physical state when its electrode reference changes."""
        values = np.array(u, dtype=float, copy=True)
        old = self.carrier_reference_energies(old_voltage_V)
        new = self.carrier_reference_energies(new_voltage_V)
        count = self.node_count
        for offset, before, after in zip((count, 2 * count), old, new):
            values[offset:offset + count] += (before - after) / self.thermal_voltage_V
        return values

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]:
        values = np.asarray(u, dtype=float)
        count = self.node_count
        if values.shape != (3 * count,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite with shape (3 * node_count,)")
        phi = values[:count]
        reference_n, reference_p = self.carrier_reference_potentials(voltage_V)
        shift_n = (phi - reference_n) / self.thermal_voltage_V
        shift_p = (phi - reference_p) / self.thermal_voltage_V
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            n = self.reference_electron_density_m3 * np.exp(shift_n + values[count:2 * count])
            p = self.reference_hole_density_m3 * np.exp(-shift_p - values[2 * count:])
        return phi, n, p

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> State1D:
        phi, n, p = self.fields(u, voltage_V=voltage_V)
        return State1D(np.asarray(u, dtype=float), phi, n, p, self.sha256, voltage_V)

    def charge_ledger(self, n: Vector, p: Vector) -> ChargeLedger:
        """Poisson reads the same explicit population ledger as its tangents."""
        return ChargeLedger(
            self.node_count,
            len(self.geometry.interfaces),
            (
                ChargeEntry("hole", "hole", 1, p),
                ChargeEntry("electron", "electron", -1, n),
                ChargeEntry("acceptor", "dopant", -1, self.parameters.acceptor_density_m3),
                ChargeEntry("donor", "dopant", 1, self.parameters.donor_density_m3),
            ),
        )

    def interface_sheet_charges_C_m2(self, u: ArrayLike, *, voltage_V: float = 0.0) -> Vector:
        """Eliminated incremental interface sheet charges [C/m^2] at state ``u``."""
        if self.interface_model is None or not self.interface_model.has_sheet_charge:
            return np.zeros(len(self.geometry.interfaces), dtype=float)
        values = np.asarray(u, dtype=float)
        phi, n, p = self.fields(values, voltage_V=voltage_V)
        return self.interface_model.evaluate_state(
            self, values, phi, n, p, derivatives=False
        ).sheet_charges_C_m2

    def interface_occupancies(self, u: ArrayLike, *, voltage_V: float = 0.0) -> Vector:
        """Eliminated interface trap occupancies (or NaN where absent) at state ``u``."""
        if self.interface_model is None:
            return np.full(len(self.geometry.interfaces), np.nan, dtype=float)
        values = np.asarray(u, dtype=float)
        phi, n, p = self.fields(values, voltage_V=voltage_V)
        return self.interface_model.evaluate_state(
            self, values, phi, n, p, derivatives=False
        ).occupancies


def quasi_fermi_potential_tangents(
    flux: CarrierFaceFlux, drive: Vector, carrier_sign: float, thermal: float
) -> tuple[Vector, Vector]:
    """Chain the SG kernel through density(phi, z) without equilibrium cancellation."""
    pair = bernoulli_pair(flux.energy_step)
    forward = np.divide(
        pair.forward_derivative, pair.forward,
        out=np.zeros_like(flux.flux), where=pair.forward != 0.0,
    )
    backward = np.divide(
        pair.backward_derivative, pair.backward,
        out=np.zeros_like(flux.flux), where=pair.backward != 0.0,
    )
    # An underflowed B(x), x >> 1, still has d(log B)/dx = 1/x - 1.
    tiny_forward = pair.forward == 0.0
    tiny_backward = pair.backward == 0.0
    forward[tiny_forward] = 1.0 / flux.energy_step[tiny_forward] - 1.0
    backward[tiny_backward] = -1.0 / flux.energy_step[tiny_backward] - 1.0
    from_left = drive < 0.0
    left_factor = np.where(from_left, 1.0 + forward, -backward)
    right_factor = np.where(from_left, -forward, 1.0 + backward)
    factor = carrier_sign * flux.flux / thermal
    return factor * left_factor, factor * right_factor


@dataclass(frozen=True)
class DC1DProblem:
    """Production F(u, udot), Fy, and Fudot for one operating point."""

    device: CompiledDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.device.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")

    def _derivative(self, udot: ArrayLike | None) -> Vector:
        if udot is None:
            return np.zeros(3 * self.device.node_count)
        derivative = np.asarray(udot, dtype=float)
        if derivative.shape != (3 * self.device.node_count,) or not np.all(np.isfinite(derivative)):
            raise ValueError("udot must be finite and match the coordinate vector")
        return derivative

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        device = self.device
        count = device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = device.fields(values, voltage_V=self.voltage_V)
        if device.interface_model is not None:
            return device.interface_model.evaluate_state(
                device, values, phi, n, p, derivatives=False
            ).fluxes
        electron_diffusivity = device.electron_diffusivity_faces_m2_s
        hole_diffusivity = device.hole_diffusivity_faces_m2_s
        if device.field_transport is not None:
            transport = device.field_transport.evaluate(phi)
            electron_diffusivity = transport.electron_diffusivity_m2_s
            hole_diffusivity = transport.hole_diffusivity_m2_s
        return carrier_particle_fluxes(
            device.nodes_m, phi, n, p,
            electron_diffusivity, hole_diffusivity,
            device.thermal_voltage_V,
            device.parameters.band_structure,
            electron_electrochemical_step=np.diff(values[count:2 * count]),
            hole_electrochemical_step=-np.diff(values[2 * count:]),
            band_step_mode=device.band_step_mode,
        )

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
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

    def contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        """Evaluate the production contact slope on an accurate density excess."""
        count = self.device.node_count
        references = self.device.carrier_reference_potentials(self.voltage_V)
        reference = references[0] if channel.carrier == "n" else references[1]
        shift = (u[node] - reference[node]) / self.device.thermal_voltage_V
        exponent = shift + u[count + node] if channel.carrier == "n" else -shift - u[2 * count + node]
        boundary = boundary_particle_flux(channel, channel.reservoir_density_m3)
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        # The boundary kernel supplies n_out * S, including exactly zero for blocking.
        return boundary.density_derivative * excess, boundary.density_derivative

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel, node: int) -> tuple[float, float]:
        """Delegate for the pending backward-Euler accounting migration."""
        return self.contact_exchange(u, channel, node)

    def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> Vector:
        device = self.device
        count = device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = device.fields(values, voltage_V=self.voltage_V)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
            return np.full(3 * count, np.nan)
        derivative = self._derivative(udot)
        iface_eval: InterfaceEvaluation1D | None = None
        if device.interface_model is not None:
            try:
                iface_eval = device.interface_model.evaluate_state(
                    device, values, phi, n, p, derivatives=False
                )
            except (RuntimeError, FloatingPointError, ValueError):
                return np.full(3 * count, np.nan)
            fluxes = iface_eval.fluxes
        else:
            fluxes = self.fluxes(values)
        rate = self.recombination_rate(n, p, derivatives=False)
        source = self.generation_scale * device.generation.source_m3_s - rate.rate
        result = np.empty(3 * count)
        left, right = device.contact_potentials(self.voltage_V)
        rho = device.charge_ledger(n, p).space_charge_C_m3()
        result[:count] = device.poisson.residual(phi, rho, left, right)
        if iface_eval is not None and device.interface_model is not None and device.interface_model.has_sheet_charge:
            result[1:count - 1] += iface_eval.g_port_val
        geometry = device._carrier_geometry
        for carrier, offset, density, flux, sign, port_val in (
            ("n", count, n, fluxes.electron, 1.0, None if iface_eval is None else iface_eval.e_port_val),
            ("p", 2 * count, p, fluxes.hole, -1.0, None if iface_eval is None else iface_eval.h_port_val),
        ):
            name = "electron" if carrier == "n" else "hole"
            sink = TermSink(device._carrier_layout, name, geometry)
            sink.add(FaceFlux(name + ".sg", "faces", flux.flux, FLUX_UNIT))
            sink.add(CellSource(name + ".net_pairs", "cells", source, SOURCE_UNIT))
            rows = np.array(sink.value(), copy=True)
            if port_val is not None:
                rows[1:count - 1] += port_val
            rate_of_density = sign * density * (
                derivative[:count] / device.thermal_voltage_V + derivative[offset:offset + count]
            )
            rows -= device.geometry.cell_measures_m * rate_of_density
            for side, node in (("left", 0), ("right", count - 1)):
                channel = device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    references = device.carrier_reference_potentials(self.voltage_V)
                    reference = references[0] if carrier == "n" else references[1]
                    shift = (phi[node] - reference[node]) / device.thermal_voltage_V
                    rows[node] = sign * (shift + values[offset + node])
                else:
                    exchange, _ = self.contact_exchange(values, channel, node)
                    rows[node] += (1.0 if side == "left" else -1.0) * exchange
            result[offset:offset + count] = rows
        return result

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        """Exact Fudot; contact constraints and Poisson rows have no storage."""
        device = self.device
        count = device.node_count
        _, n, p = device.fields(u, voltage_V=self.voltage_V)
        wn = device.geometry.cell_measures_m * n
        wp = device.geometry.cell_measures_m * p
        for carrier, weights in (("n", wn), ("p", wp)):
            for side, node in (("left", 0), ("right", count - 1)):
                if device.channel(carrier, side).kind == "fixed_reservoir":
                    weights[node] = 0.0
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        matrix[count:2 * count, :count] = diags(-wn / device.thermal_voltage_V)
        matrix[count:2 * count, count:2 * count] = diags(-wn)
        matrix[2 * count:, :count] = diags(wp / device.thermal_voltage_V)
        matrix[2 * count:, 2 * count:] = diags(wp)
        return matrix.tocsc()

    def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        """Analytic Fy + alpha Fudot in the production quasi-Fermi coordinates."""
        device = self.device
        count = device.node_count
        thermal = device.thermal_voltage_V
        values = np.asarray(u, dtype=float)
        phi, n, p = device.fields(values, voltage_V=self.voltage_V)
        derivative = self._derivative(udot)
        iface_eval: InterfaceEvaluation1D | None = None
        if device.interface_model is not None:
            if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
                raise FloatingPointError("state outside the field map's admissible domain")
            try:
                iface_eval = device.interface_model.evaluate_state(
                    device, values, phi, n, p, derivatives=True
                )
            except (RuntimeError, ValueError) as exc:
                raise FloatingPointError("interface closure failed at trial state") from exc
            fluxes = iface_eval.fluxes
        else:
            fluxes = self.fluxes(values)
        rate = self.recombination_rate(n, p)
        transport = None if device.field_transport is None else device.field_transport.evaluate(values[:count])
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        widths = device.geometry.cell_measures_m
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        interior = range(1, count - 1)
        charge_widths = device.poisson.interior_widths_m
        if iface_eval is not None and device.interface_model is not None and device.interface_model.has_sheet_charge:
            matrix[0, 0] = 1.0
            matrix[count - 1, count - 1] = 1.0
            for face in range(count - 1):
                for orientation, node in ((-1.0, face), (1.0, face + 1)):
                    if node == 0 or node == count - 1:
                        continue
                    matrix[node, face] += orientation * iface_eval.g_left_phi[face]
                    matrix[node, face + 1] += orientation * iface_eval.g_right_phi[face]
                    matrix[node, count + face] += orientation * iface_eval.g_left_zn[face]
                    matrix[node, count + face + 1] += orientation * iface_eval.g_right_zn[face]
                    matrix[node, 2 * count + face] += orientation * iface_eval.g_left_zp[face]
                    matrix[node, 2 * count + face + 1] += orientation * iface_eval.g_right_zp[face]
            for local, node in enumerate(interior):
                matrix[node, node] += -Q * charge_widths[local] * (n[node] + p[node]) / thermal
                matrix[node, count + node] += -Q * charge_widths[local] * n[node]
                matrix[node, 2 * count + node] += -Q * charge_widths[local] * p[node]
            for local, node in enumerate(interior):
                matrix[node, node] += iface_eval.g_port_phi[local]
                matrix[node, count + node] += iface_eval.g_port_zn[local]
                matrix[node, 2 * count + node] += iface_eval.g_port_zp[local]
        else:
            matrix[:count, :count] = device.poisson.jacobian()
            for local, node in enumerate(interior):
                matrix[node, node] += -Q * charge_widths[local] * (n[node] + p[node]) / thermal
                matrix[node, count + node] = -Q * charge_widths[local] * n[node]
                matrix[node, 2 * count + node] = -Q * charge_widths[local] * p[node]
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / thermal
        for carrier, offset, density, flux, sign in (
            ("n", count, n, fluxes.electron, 1.0),
            ("p", 2 * count, p, fluxes.hole, -1.0),
        ):
            p_phi: Vector | None = None
            p_zn: Vector | None = None
            p_zp: Vector | None = None
            if iface_eval is None:
                drive = sign * np.diff(values[offset:offset + count])
                left_phi, right_phi = quasi_fermi_potential_tangents(flux, drive, sign, thermal)
                if transport is not None:
                    diffusivity = (transport.electron_diffusivity_m2_s if carrier == "n"
                                   else transport.hole_diffusivity_m2_s)
                    d_diffusivity = (transport.electron_d_diffusivity_d_field if carrier == "n"
                                     else transport.hole_d_diffusivity_d_field)
                    # SG is linear in positive D. E = -(phi_R-phi_L)/dx, so
                    # this is the additional (dGamma/dD)(dD/dE)(dE/dphi) term.
                    field_chain = (flux.flux / diffusivity) * d_diffusivity / np.diff(device.nodes_m)
                    left_phi = left_phi + field_chain
                    right_phi = right_phi - field_chain
                left_z = sign * flux.left_density_derivative * density[:-1]
                right_z = sign * flux.right_density_derivative * density[1:]
                for face in range(count - 1):
                    for orientation, node in ((-1.0, face), (1.0, face + 1)):
                        row = offset + node
                        matrix[row, face] += orientation * left_phi[face]
                        matrix[row, face + 1] += orientation * right_phi[face]
                        matrix[row, offset + face] += orientation * left_z[face]
                        matrix[row, offset + face + 1] += orientation * right_z[face]
            else:
                if carrier == "n":
                    l_phi, r_phi = iface_eval.e_left_phi, iface_eval.e_right_phi
                    l_zn, r_zn = iface_eval.e_left_zn, iface_eval.e_right_zn
                    l_zp, r_zp = iface_eval.e_left_zp, iface_eval.e_right_zp
                    p_phi, p_zn, p_zp = iface_eval.e_port_phi, iface_eval.e_port_zn, iface_eval.e_port_zp
                else:
                    l_phi, r_phi = iface_eval.h_left_phi, iface_eval.h_right_phi
                    l_zn, r_zn = iface_eval.h_left_zn, iface_eval.h_right_zn
                    l_zp, r_zp = iface_eval.h_left_zp, iface_eval.h_right_zp
                    p_phi, p_zn, p_zp = iface_eval.h_port_phi, iface_eval.h_port_zn, iface_eval.h_port_zp
                for face in range(count - 1):
                    for orientation, node in ((-1.0, face), (1.0, face + 1)):
                        row = offset + node
                        matrix[row, face] += orientation * l_phi[face]
                        matrix[row, face + 1] += orientation * r_phi[face]
                        matrix[row, count + face] += orientation * l_zn[face]
                        matrix[row, count + face + 1] += orientation * r_zn[face]
                        matrix[row, 2 * count + face] += orientation * l_zp[face]
                        matrix[row, 2 * count + face + 1] += orientation * r_zp[face]
            density_rate = sign * density * (
                derivative[:count] / thermal + derivative[offset:offset + count]
            )
            for node in range(count):
                row = offset + node
                matrix[row, node] += widths[node] * (source_phi[node] - sign * density_rate[node] / thermal)
                matrix[row, count + node] += widths[node] * source_n[node]
                matrix[row, 2 * count + node] += widths[node] * source_p[node]
                matrix[row, offset + node] -= widths[node] * sign * density_rate[node]
            if (
                p_phi is not None
                and p_zn is not None
                and p_zp is not None
                and device.interface_model is not None
                and device.interface_model.has_carrier_port
            ):
                for local, node in enumerate(interior):
                    row = offset + node
                    matrix[row, node] += p_phi[local]
                    matrix[row, count + node] += p_zn[local]
                    matrix[row, 2 * count + node] += p_zp[local]
            for side, node in (("left", 0), ("right", count - 1)):
                row = offset + node
                channel = device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    matrix.rows[row] = [node, offset + node]
                    matrix.data[row] = [sign / thermal, sign]
                else:
                    _, slope = self.contact_exchange(values, channel, node)
                    inflow = 1.0 if side == "left" else -1.0
                    matrix[row, node] += inflow * slope * sign * density[node] / thermal
                    matrix[row, offset + node] += inflow * slope * sign * density[node]
        result = matrix.tocsc()
        if alpha != 0.0:
            if not math.isfinite(alpha):
                raise ValueError("alpha must be finite")
            result = result + alpha * self.storage_jacobian(values)
        return result

    def current_density(self, u: ArrayLike) -> Vector:
        fluxes = self.fluxes(u)
        return fluxes.electron_current() + fluxes.hole_current()

    def contact_current_density(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=float)
        fluxes = self.fluxes(values)
        _, n, p = self.device.fields(values, voltage_V=self.voltage_V)
        rate = self.recombination_rate(n, p, derivatives=False)
        pairs = self.device.geometry.cell_measures_m * (
            self.generation_scale * self.device.generation.source_m3_s - rate.rate
        )
        currents = []
        for side, node, face in (("left", 0, 0), ("right", self.device.node_count - 1, -1)):
            total = 0.0
            for carrier, charge, flux in (("n", -Q, fluxes.electron), ("p", Q, fluxes.hole)):
                channel = self.device.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    exchange = float(flux.flux[face] + (-1.0 if side == "left" else 1.0) * pairs[node])
                else:
                    exchange, _ = self.contact_exchange(values, channel, node)
                total += charge * exchange
            currents.append(total)
        return np.asarray(currents)

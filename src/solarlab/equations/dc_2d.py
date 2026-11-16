"""Classical 2D DC on one connected lateral/stack face graph.

Unknowns are (phi, z_n, z_p) using the accepted majority-electrode references.
Gauss and carrier balances use physical areas and face lengths per unit depth;
fixed-reservoir contact rows are unweighted potential/log-density constraints.
Finite exchange and blocking contacts retain conservative carrier rows.
There is no independent-column residual, nonlinear solver or transient route.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, lil_matrix

from solarlab.contracts import (
    AREA, COULOMB, LENGTH, PARTICLE, SECOND, VOLUME, CellSource, EquationSpec,
    FaceFlux, Layout, Support, TermSink, frozen_array,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.face_average import (
    DIELECTRIC_RULES, NODAL_HARMONIC_DIELECTRIC_V1, VACUUM_PERMITTIVITY_F_M,
)
from solarlab.equations.dc_1d import CompiledDevice1D, quasi_fermi_potential_tangents
from solarlab.equations.field_transport_2d import FieldTransport2D
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.contacts.exchange import CarrierExchangeChannel
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.electrostatics.poisson import rectangular_2d_face_capacitances
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.physics.interface_transport import (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    ThermionicCombinationMode,
    combine_sg_and_reciprocal_thermionic_flux,
    reciprocal_thermionic_flux_m2_s,
)
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients, BulkRecombinationEvaluation, BulkRecombinationReaction,
)
from solarlab.physics.statistics.band_potentials import BandStructure
from solarlab.physics.transport.electrochemical_flux import (
    CarrierFaceFlux,
    CarrierFaceFluxes,
    LEGACY_SUMMED_POTENTIAL_V1,
    carrier_transport_potential_steps,
    stable_sg_particle_flux,
)

Vector: TypeAlias = NDArray[np.float64]
Q = ELEMENTARY_CHARGE_C


class GrainRecombinationModel2D(Protocol):
    """A geometry-bound rate correction supplied by device assembly."""

    @property
    def geometry_sha256(self) -> str: ...

    def evaluate(self, evaluation: BulkRecombinationEvaluation,
                 coefficients: BulkRecombinationCoefficients) -> RecombinationRate: ...


@dataclass(frozen=True)
class State2D:
    u: Vector
    potential_V: Vector
    electron_density_m3: Vector
    hole_density_m3: Vector
    device_sha256: str
    voltage_V: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        shape = self.potential_V.shape
        if len(shape) != 2 or self.u.shape != (3 * self.potential_V.size,):
            raise ValueError("2D fields require (Ny,Nx) and a matching coordinate vector")
        for value in (self.electron_density_m3, self.hole_density_m3):
            if value.shape != shape or np.any(value <= 0.0):
                raise ValueError("2D carrier densities must be positive on every node")
        if not self.device_sha256 or not math.isfinite(self.voltage_V):
            raise ValueError("state requires device identity and finite reference voltage")


@dataclass(frozen=True)
class CompiledDevice2D:
    """Extruded physical coefficients; equations act on the full 2D face graph.

    column supplies only accepted physical coefficients/contact references.
    Its integer-endpoint geometry is not used for 2D conservation weights.
    generation_y is explicitly rebound to physical half-endpoint cells.
    """

    sha256: str
    column: CompiledDevice1D
    geometry: RectangularGeometry2D
    generation_y: GenerationProfile
    relative_permittivity_y: Vector
    spatial_declaration_json: bytes
    grain_recombination: GrainRecombinationModel2D | None = None
    column_projection_json: bytes | None = None
    field_transport: FieldTransport2D | None = None
    dielectric_rule: str = NODAL_HARMONIC_DIELECTRIC_V1
    layer_permittivity_y: tuple[float, ...] | None = None
    lateral_permittivity_x: Vector | None = None
    lateral_interfaces_m: tuple[float, ...] = ()
    lateral_layer_permittivity_x: tuple[float, ...] | None = None
    band_structure: BandStructure = field(init=False)
    recombination: BulkRecombinationReaction = field(init=False)
    generation_source_m3_s: Vector = field(init=False)
    electron_diffusivity_faces_m2_s: Vector = field(init=False)
    hole_diffusivity_faces_m2_s: Vector = field(init=False)
    face_capacitance_F_m2: Vector = field(init=False)
    layout: Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        column, geometry = self.column, self.geometry
        if self.grain_recombination is not None and self.grain_recombination.geometry_sha256 != geometry.sha256:
            raise ValueError("grain recombination must be bound to this exact 2D geometry")
        if not self.sha256 or not np.array_equal(column.nodes_m, geometry.y_geometry.nodes_m):
            raise ValueError("column coefficients must match the 2D stack grid and identity")
        if column.field_transport is not None and self.field_transport is None:
            raise ValueError("active CT requires its full2D normal-face binding")
        if column.recombination.explicit_defects is not None:
            raise ValueError("2D DC requires lifetime SRH")
        if column.additional_recombination is not None:
            raise ValueError("2D DC requires the bound bulk reaction")
        if all(c.kind == "blocking" for c in column.channels):
            raise ValueError("fully_blocking_contacts: a closed carrier inventory constraint is not bound in 2D DC")
        eps = frozen_array(self.relative_permittivity_y)
        if eps.shape != (geometry.ny,) or np.any(eps <= 0.0):
            raise ValueError("positive physical permittivity is required on every stack node")
        object.__setattr__(self, "relative_permittivity_y", eps)
        self.generation_y.require_inputs(self.generation_y.binding_sha256, geometry.y_geometry)
        object.__setattr__(self, "generation_source_m3_s", frozen_array(self.extrude(self.generation_y.source_m3_s)))
        bands = column.parameters.band_structure
        object.__setattr__(self, "band_structure", BandStructure(
            self.extrude(bands.physical_electron_affinity_eV), self.extrude(bands.physical_band_gap_eV),
            self.extrude(bands.transport_electron_affinity_eV), self.extrude(bands.transport_band_gap_eV),
            bands.dos_folded, bands.reference_layer,
        ))
        coefficients = column.recombination.coefficients
        names = ("ni_sq_m6", "tau_n_s", "tau_p_s", "n1_m3", "p1_m3", "radiative_m3_s", "auger_n_m6_s", "auger_p_m6_s")
        object.__setattr__(self, "recombination", BulkRecombinationReaction(
            BulkRecombinationCoefficients(self.node_count, *(self.extrude(getattr(coefficients, name)) for name in names))))
        for carrier in ("electron", "hole"):
            nodal = getattr(column.parameters, carrier + "_diffusivity_m2_s")
            vertical = getattr(column, carrier + "_diffusivity_faces_m2_s")
            values = np.r_[np.repeat(nodal, geometry.x_faces_per_row), np.repeat(vertical, geometry.nx)]
            if np.any(values <= 0.0):
                raise ValueError("2D DC requires positive finite carrier diffusivities")
            object.__setattr__(self, carrier + "_diffusivity_faces_m2_s", frozen_array(values))
        if self.field_transport is not None:
            model = self.field_transport
            if (model.geometry.sha256 != geometry.sha256 or model.thermal_voltage_V != self.thermal_voltage_V
                    or not np.array_equal(model.base_electron_diffusivity_m2_s, self.electron_diffusivity_faces_m2_s)
                    or not np.array_equal(model.base_hole_diffusivity_m2_s, self.hole_diffusivity_faces_m2_s)):
                raise ValueError("field transport must match the exact2D grid, temperature and baseD")
        if self.dielectric_rule not in DIELECTRIC_RULES:
            raise ValueError(f"unknown 2D dielectric rule {self.dielectric_rule!r}")
        if (
            self.dielectric_rule == NODAL_HARMONIC_DIELECTRIC_V1
            and self.lateral_permittivity_x is None
            and not self.lateral_interfaces_m
            and self.lateral_layer_permittivity_x is None
        ):
            dx_row = (np.diff(geometry.x_m) if geometry.lateral_bc == "neumann"
                      else geometry.face_spacing_m[:geometry.nx])
            horizontal_capacitance = (VACUUM_PERMITTIVITY_F_M * eps[:, None] / dx_row[None, :]).ravel()
            object.__setattr__(self, "face_capacitance_F_m2", frozen_array(np.r_[
                horizontal_capacitance, np.repeat(column.poisson.face_capacitance_F_m2, geometry.nx)]))
        else:
            if self.lateral_permittivity_x is not None:
                object.__setattr__(self, "lateral_permittivity_x", frozen_array(self.lateral_permittivity_x))
            object.__setattr__(
                self,
                "face_capacitance_F_m2",
                rectangular_2d_face_capacitances(
                    geometry,
                    eps,
                    dielectric_rule=self.dielectric_rule,
                    layer_eps_r_y=self.layer_permittivity_y,
                    lateral_eps_r_x=self.lateral_permittivity_x,
                    lateral_interfaces_m=self.lateral_interfaces_m,
                    lateral_layer_eps_r_x=self.lateral_layer_permittivity_x,
                ),
            )
        faces = geometry.face_spacing_m.size
        object.__setattr__(self, "layout", Layout(
            (Support("cells", "cell", (self.node_count,)), Support("faces", "face", (faces,))), (),
            (EquationSpec("gauss", "electrostatics", "cells", (self.node_count,), COULOMB / LENGTH, role="constraint"),
             EquationSpec("electron", "electron", "cells", (self.node_count,), PARTICLE / (LENGTH * SECOND)),
             EquationSpec("hole", "hole", "cells", (self.node_count,), PARTICLE / (LENGTH * SECOND))),
        ))

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.geometry.node_count

    @property
    def thermal_voltage_V(self) -> float:
        return self.column.thermal_voltage_V

    @property
    def fixed_carrier_mask(self) -> NDArray[np.bool_]:
        """Density-constraint rows, indexed by (electron/hole, node)."""
        mask: NDArray[np.bool_] = np.zeros((2, self.node_count), dtype=bool)
        sides = self.geometry.contact_nodes.reshape(2, self.geometry.nx)
        for carrier_index, carrier in enumerate(("n", "p")):
            for side, nodes in zip(("left", "right"), sides):
                if self.column.channel(carrier, side).kind == "fixed_reservoir":
                    mask[carrier_index, nodes] = True
        mask.flags.writeable = False
        return mask

    @property
    def generation_limit_A_m(self) -> float:
        return Q * float(np.sum(self.generation_source_m3_s * self.geometry.contract_geometry.volumes))

    def extrude(self, values_y: ArrayLike) -> Vector:
        values = np.asarray(values_y, dtype=float)
        if values.shape != (self.geometry.ny,):
            raise ValueError("extruded nodal coefficients must match the stack grid")
        return np.repeat(values, self.geometry.nx)

    def extrude_coordinates(self, u_1d: ArrayLike) -> Vector:
        values = np.asarray(u_1d, dtype=float)
        if values.shape != (3 * self.geometry.ny,) or not np.all(np.isfinite(values)):
            raise ValueError("reference coordinates must match the compiled column")
        return np.repeat(values.reshape(3, self.geometry.ny), self.geometry.nx, axis=1).ravel()

    def reference_potentials(self, voltage_V: float) -> tuple[Vector, Vector]:
        left, right = self.column.carrier_reference_potentials(voltage_V)
        return self.extrude(left), self.extrude(right)

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]:
        values = np.asarray(u, dtype=float)
        count = self.node_count
        if values.shape != (3 * count,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite and match the 2D state layout")
        phi = values[:count]
        ref_n, ref_p = self.reference_potentials(voltage_V)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            n = self.extrude(self.column.reference_electron_density_m3) * np.exp(
                (phi - ref_n) / self.thermal_voltage_V + values[count:2 * count])
            p = self.extrude(self.column.reference_hole_density_m3) * np.exp(
                -(phi - ref_p) / self.thermal_voltage_V - values[2 * count:])
        return phi, n, p

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> State2D:
        phi, n, p = self.fields(u, voltage_V=voltage_V)
        shape = self.geometry.shape
        return State2D(np.asarray(u, dtype=float), phi.reshape(shape), n.reshape(shape), p.reshape(shape), self.sha256, voltage_V)

    def charge_ledger(self, n: Vector, p: Vector) -> ChargeLedger:
        return ChargeLedger(self.node_count, 0, (
            ChargeEntry("hole", "hole", 1, p), ChargeEntry("electron", "electron", -1, n),
            ChargeEntry("acceptor", "dopant", -1, self.extrude(self.column.parameters.acceptor_density_m3)),
            ChargeEntry("donor", "dopant", 1, self.extrude(self.column.parameters.donor_density_m3)),
        ))


@dataclass(frozen=True)
class FaceTransportEvaluation2D:
    """2D face particle fluxes and optional quasi-Fermi tangents for P08 V2 channels."""

    fluxes: CarrierFaceFluxes
    e_phi_left: Vector
    e_phi_right: Vector
    e_z_left: Vector
    e_z_right: Vector
    h_phi_left: Vector
    h_phi_right: Vector
    h_z_left: Vector
    h_z_right: Vector


def has_prospective_2d_transport(device: CompiledDevice2D) -> bool:
    """True when 2D transport opts into P08-02c band steps or P08-03 reciprocal TE."""
    model = device.column.interface_model
    return (
        device.column.band_step_mode != LEGACY_SUMMED_POTENTIAL_V1
        or (
            model is not None
            and model.topology == "deduplicated"
            and model.thermionic_mode != LEGACY_UNNORMALIZED_THERMIONIC_V1
            and bool(model.thermionic_faces)
        )
    )


def evaluate_2d_carrier_face_fluxes(
    device: CompiledDevice2D,
    phi: Vector,
    n: Vector,
    p: Vector,
    z_n: Vector,
    z_p: Vector,
    *,
    derivatives: bool = False,
) -> FaceTransportEvaluation2D:
    """Evaluate 2D SG/CT + optional P08-02c/P08-03 reciprocal TE with exact QF tangents."""
    left, right = device.geometry.contract_geometry.pairs.T
    thermal = device.thermal_voltage_V
    temp = device.column.parameters.temperature_K
    dpsi_n, dpsi_p = carrier_transport_potential_steps(
        phi,
        device.band_structure,
        band_step_mode=device.column.band_step_mode,
        left_nodes=left,
        right_nodes=right,
    )
    ec_n = z_n[right] - z_n[left]
    ec_p = -(z_p[right] - z_p[left])
    if device.field_transport is not None:
        potentials = device.band_structure.transport_potentials(phi)
        base = device.field_transport.fluxes(
            phi,
            n,
            p,
            potentials[0],
            potentials[1],
            electron_electrochemical_step=ec_n,
            hole_electrochemical_step=ec_p,
            electron_potential_step_V=dpsi_n,
            hole_potential_step_V=dpsi_p,
        )
    else:
        pair_fluxes = []
        for density, dpsi, diffusivity, sign, drive in (
            (n, dpsi_n, device.electron_diffusivity_faces_m2_s, 1.0, ec_n),
            (p, dpsi_p, device.hole_diffusivity_faces_m2_s, -1.0, ec_p),
        ):
            step = -sign * dpsi / thermal
            coefficient = diffusivity / device.geometry.face_spacing_m
            evaluated = stable_sg_particle_flux(
                density[left], density[right], step, coefficient, electrochemical_step=drive
            )
            per_volt = -sign / thermal * evaluated.energy_step_derivative
            pair_fluxes.append(
                CarrierFaceFlux(
                    evaluated.flux,
                    evaluated.left_density_derivative,
                    evaluated.right_density_derivative,
                    -per_volt,
                    per_volt,
                    step,
                    coefficient,
                )
            )
        base = CarrierFaceFluxes(pair_fluxes[0], pair_fluxes[1])

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

    if derivatives:
        e_phi_l, e_phi_r = quasi_fermi_potential_tangents(base.electron, ec_n, 1.0, thermal)
        h_phi_l, h_phi_r = quasi_fermi_potential_tangents(base.hole, ec_p, -1.0, thermal)
        e_phi_l = np.array(e_phi_l, dtype=float, copy=True)
        e_phi_r = np.array(e_phi_r, dtype=float, copy=True)
        h_phi_l = np.array(h_phi_l, dtype=float, copy=True)
        h_phi_r = np.array(h_phi_r, dtype=float, copy=True)
        if device.field_transport is not None:
            tr_eval = device.field_transport.evaluate(phi)
            h_space = device.geometry.face_spacing_m
            e_chain = (
                (e_flux / tr_eval.electron_diffusivity_m2_s)
                * tr_eval.electron_d_diffusivity_d_field
                / h_space
            )
            h_chain = (
                (h_flux / tr_eval.hole_diffusivity_m2_s)
                * tr_eval.hole_d_diffusivity_d_field
                / h_space
            )
            e_phi_l = e_phi_l + e_chain
            e_phi_r = e_phi_r - e_chain
            h_phi_l = h_phi_l + h_chain
            h_phi_r = h_phi_r - h_chain
        e_z_l = np.array(e_ld * n[left], dtype=float, copy=True)
        e_z_r = np.array(e_rd * n[right], dtype=float, copy=True)
        h_z_l = np.array(-h_ld * p[left], dtype=float, copy=True)
        h_z_r = np.array(-h_rd * p[right], dtype=float, copy=True)
    else:
        e_phi_l = e_phi_r = e_z_l = e_z_r = np.empty(0)
        h_phi_l = h_phi_r = h_z_l = h_z_r = np.empty(0)

    model = device.column.interface_model
    if (
        model is not None
        and model.topology == "deduplicated"
        and model.thermionic_mode != LEGACY_UNNORMALIZED_THERMIONIC_V1
    ):
        comb_mode: ThermionicCombinationMode = (
            "harmonic_series" if model.combination_mode is None else model.combination_mode
        )
        x_offset = device.geometry.x_face_count
        nx = device.geometry.nx
        for tf in model.thermionic_faces:
            base_idx = x_offset + tf.face * nx
            for col in range(nx):
                f = base_idx + col
                a = int(left[f])
                b = int(right[f])
                if tf.cap_electron:
                    te_e = reciprocal_thermionic_flux_m2_s(
                        float(n[a]),
                        float(n[b]),
                        carrier="electron",
                        phi_left_V=float(phi[a]),
                        phi_right_V=float(phi[b]),
                        band_offset_eV=tf.delta_Ec_eV,
                        temperature_K=temp,
                        a_star_left_A_m2_K2=tf.a_star_n_A_m2_K2,
                        a_star_right_A_m2_K2=tf.a_star_n_right_A_m2_K2,
                        dos_left_m3=tf.dos_n_left_m3,
                        dos_right_m3=tf.dos_n_right_m3,
                        transmission=tf.transmission,
                        statistics="maxwell_boltzmann",
                        model_version=model.thermionic_mode,
                        thermal_voltage_V=thermal,
                        electrochemical_step=float(ec_n[f]),
                    )
                    comb_e = combine_sg_and_reciprocal_thermionic_flux(
                        float(e_flux[f]),
                        d_sg_d_phi_left_m2_s_V=float(e_lpot[f]),
                        d_sg_d_phi_right_m2_s_V=float(e_rpot[f]),
                        d_sg_d_c_left_m_s=float(e_ld[f]),
                        d_sg_d_c_right_m_s=float(e_rd[f]),
                        c_left_m3=float(n[a]),
                        c_right_m3=float(n[b]),
                        thermionic=te_e,
                        combination_mode=comb_mode,
                        te_softness=model.te_softness,
                    )
                    e_flux[f] = comb_e.flux_m2_s
                    e_ld[f] = comb_e.d_flux_d_c_left_m_s
                    e_rd[f] = comb_e.d_flux_d_c_right_m_s
                    e_lpot[f] = comb_e.d_flux_d_phi_left_m2_s_V
                    e_rpot[f] = comb_e.d_flux_d_phi_right_m2_s_V
                    if derivatives:
                        te_qf_l = (te_e.right_barrier_weight / thermal) * te_e.flux_m2_s
                        te_qf_r = (te_e.left_barrier_weight / thermal) * te_e.flux_m2_s
                        e_phi_l[f] = comb_e.sg_weight * e_phi_l[f] + comb_e.te_weight * te_qf_l
                        e_phi_r[f] = comb_e.sg_weight * e_phi_r[f] + comb_e.te_weight * te_qf_r
                        e_z_l[f] = comb_e.d_flux_d_log_c_left_m2_s
                        e_z_r[f] = comb_e.d_flux_d_log_c_right_m2_s
                if tf.cap_hole:
                    h_offset = (
                        -tf.delta_Ev_eV if tf.hole_band_offset_eV is None else tf.hole_band_offset_eV
                    )
                    te_h = reciprocal_thermionic_flux_m2_s(
                        float(p[a]),
                        float(p[b]),
                        carrier="hole",
                        phi_left_V=float(phi[a]),
                        phi_right_V=float(phi[b]),
                        band_offset_eV=h_offset,
                        temperature_K=temp,
                        a_star_left_A_m2_K2=tf.a_star_p_A_m2_K2,
                        a_star_right_A_m2_K2=tf.a_star_p_right_A_m2_K2,
                        dos_left_m3=tf.dos_p_left_m3,
                        dos_right_m3=tf.dos_p_right_m3,
                        transmission=tf.transmission,
                        statistics="maxwell_boltzmann",
                        model_version=model.thermionic_mode,
                        thermal_voltage_V=thermal,
                        electrochemical_step=float(ec_p[f]),
                    )
                    comb_h = combine_sg_and_reciprocal_thermionic_flux(
                        float(h_flux[f]),
                        d_sg_d_phi_left_m2_s_V=float(h_lpot[f]),
                        d_sg_d_phi_right_m2_s_V=float(h_rpot[f]),
                        d_sg_d_c_left_m_s=float(h_ld[f]),
                        d_sg_d_c_right_m_s=float(h_rd[f]),
                        c_left_m3=float(p[a]),
                        c_right_m3=float(p[b]),
                        thermionic=te_h,
                        combination_mode=comb_mode,
                        te_softness=model.te_softness,
                    )
                    h_flux[f] = comb_h.flux_m2_s
                    h_ld[f] = comb_h.d_flux_d_c_left_m_s
                    h_rd[f] = comb_h.d_flux_d_c_right_m_s
                    h_lpot[f] = comb_h.d_flux_d_phi_left_m2_s_V
                    h_rpot[f] = comb_h.d_flux_d_phi_right_m2_s_V
                    if derivatives:
                        te_qf_l = -(te_h.right_barrier_weight / thermal) * te_h.flux_m2_s
                        te_qf_r = -(te_h.left_barrier_weight / thermal) * te_h.flux_m2_s
                        h_phi_l[f] = comb_h.sg_weight * h_phi_l[f] + comb_h.te_weight * te_qf_l
                        h_phi_r[f] = comb_h.sg_weight * h_phi_r[f] + comb_h.te_weight * te_qf_r
                        h_z_l[f] = -comb_h.d_flux_d_log_c_left_m2_s
                        h_z_r[f] = -comb_h.d_flux_d_log_c_right_m2_s

    replaced = CarrierFaceFluxes(
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
    return FaceTransportEvaluation2D(
        fluxes=replaced,
        e_phi_left=e_phi_l,
        e_phi_right=e_phi_r,
        e_z_left=e_z_l,
        e_z_right=e_z_r,
        h_phi_left=h_phi_l,
        h_phi_right=h_phi_r,
        h_z_left=h_z_l,
        h_z_right=h_z_r,
    )


@dataclass(frozen=True)
class DC2DProblem:
    device: CompiledDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.device.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")

    def fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        d = self.device
        values = np.asarray(u, dtype=float)
        phi, n, p = d.fields(values, voltage_V=self.voltage_V)
        if has_prospective_2d_transport(d):
            return evaluate_2d_carrier_face_fluxes(
                d,
                phi,
                n,
                p,
                values[d.node_count:2 * d.node_count],
                values[2 * d.node_count:],
                derivatives=False,
            ).fluxes
        left, right = d.geometry.contract_geometry.pairs.T
        potentials = d.band_structure.transport_potentials(phi)
        if d.field_transport is not None:
            return d.field_transport.fluxes(phi, n, p, potentials[0], potentials[1],
                electron_electrochemical_step=values[d.node_count + right] - values[d.node_count + left],
                hole_electrochemical_step=-(values[2 * d.node_count + right] - values[2 * d.node_count + left]))
        fluxes = []
        for density, potential, diffusivity, offset, sign in (
            (n, potentials[0], d.electron_diffusivity_faces_m2_s, d.node_count, 1.0),
            (p, potentials[1], d.hole_diffusivity_faces_m2_s, 2 * d.node_count, -1.0),
        ):
            step = -sign * (potential[right] - potential[left]) / d.thermal_voltage_V
            coefficient = diffusivity / d.geometry.face_spacing_m
            drive = sign * (values[offset + right] - values[offset + left])
            evaluated = stable_sg_particle_flux(density[left], density[right], step, coefficient, electrochemical_step=drive)
            per_volt = -sign / d.thermal_voltage_V * evaluated.energy_step_derivative
            fluxes.append(CarrierFaceFlux(evaluated.flux, evaluated.left_density_derivative, evaluated.right_density_derivative,
                                         -per_volt, per_volt, step, coefficient))
        return CarrierFaceFluxes(fluxes[0], fluxes[1])

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        evaluation = self.device.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        if self.device.grain_recombination is not None:
            return self.device.grain_recombination.evaluate(evaluation, self.device.recombination.coefficients)
        return evaluation.total

    def displacement(self, u: ArrayLike) -> Vector:
        phi, _, _ = self.device.fields(u, voltage_V=self.voltage_V)
        left, right = self.device.geometry.contract_geometry.pairs.T
        return -self.device.face_capacitance_F_m2 * (phi[right] - phi[left])

    def contact_exchange(self, u: Vector, channel: CarrierExchangeChannel,
                         nodes: NDArray[np.int64]) -> tuple[Vector, float]:
        """The 1D contact law/expm1 construction on a whole y contact face.

        Left/right means y-min/y-max here. Gamma is directed along +y,
        in particles/(m2*s); its exact density slope is n_out*S.
        """
        d, count = self.device, self.device.node_count
        references = d.reference_potentials(self.voltage_V)
        reference = references[0] if channel.carrier == "n" else references[1]
        shift = (u[nodes] - reference[nodes]) / d.thermal_voltage_V
        exponent = shift + u[count + nodes] if channel.carrier == "n" else -shift - u[2 * count + nodes]
        boundary = boundary_particle_flux(channel, channel.reservoir_density_m3)
        excess = channel.reservoir_density_m3 * np.expm1(exponent)
        return boundary.density_derivative * excess, boundary.density_derivative

    def _contact_exchange(self, u: Vector, channel: CarrierExchangeChannel,
                          nodes: NDArray[np.int64]) -> tuple[Vector, float]:
        return self.contact_exchange(u, channel, nodes)

    def residual(self, u: ArrayLike) -> Vector:
        d, count = self.device, self.device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = d.fields(values, voltage_V=self.voltage_V)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
            return np.full(3 * count, np.nan)
        fluxes = self.fluxes(values)
        source = self.generation_scale * d.generation_source_m3_s - self.recombination_rate(n, p, derivatives=False).rate
        outputs = []
        for name, flux, flux_unit, cell_source, source_unit in (
            ("gauss", self.displacement(values), COULOMB / AREA, d.charge_ledger(n, p).space_charge_C_m3(), COULOMB / VOLUME),
            ("electron", fluxes.electron.flux, PARTICLE / (AREA * SECOND), source, PARTICLE / (VOLUME * SECOND)),
            ("hole", fluxes.hole.flux, PARTICLE / (AREA * SECOND), source, PARTICLE / (VOLUME * SECOND)),
        ):
            sink = TermSink(d.layout, name, d.geometry.contract_geometry)
            sink.add(FaceFlux(name + ".transport", "faces", flux, flux_unit))
            sink.add(CellSource(name + ".source", "cells", cell_source, source_unit))
            outputs.append(np.array(sink.value(), copy=True))
        bottom, top = d.column.contact_potentials(self.voltage_V)
        outputs[0][:d.geometry.nx] = phi[:d.geometry.nx] - bottom
        outputs[0][-d.geometry.nx:] = phi[-d.geometry.nx:] - top
        sides = d.geometry.contact_nodes.reshape(2, d.geometry.nx)
        for carrier, rows, offset, sign, reference in zip(("n", "p"), outputs[1:], (count, 2 * count),
                                                          (1.0, -1.0), d.reference_potentials(self.voltage_V)):
            for side, nodes, inflow in zip(("left", "right"), sides, (1.0, -1.0)):
                channel = d.column.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    rows[nodes] = sign * ((phi[nodes] - reference[nodes]) / d.thermal_voltage_V + values[offset + nodes])
                else:
                    exchange, _ = self._contact_exchange(values, channel, nodes)
                    rows[nodes] += inflow * exchange * d.geometry.x_widths_m
        return np.concatenate(outputs)

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        """Full analytic Newton Jacobian, including both spatial directions."""
        d, count = self.device, self.device.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p = d.fields(values, voltage_V=self.voltage_V)
        thermal = d.thermal_voltage_V
        geometry = d.geometry.contract_geometry
        left, right = geometry.pairs.T
        matrix = lil_matrix((3 * count, 3 * count), dtype=float)
        for a, b, capacitance in zip(left, right, d.face_capacitance_F_m2 * geometry.face_measures):
            matrix[a, a] -= capacitance
            matrix[a, b] += capacitance
            matrix[b, a] += capacitance
            matrix[b, b] -= capacitance
        for node, area in enumerate(geometry.volumes):
            matrix[node, node] -= Q * area * (n[node] + p[node]) / thermal
            matrix[node, count + node] = -Q * area * n[node]
            matrix[node, 2 * count + node] = -Q * area * p[node]
        prospective_eval = (
            evaluate_2d_carrier_face_fluxes(
                d,
                phi,
                n,
                p,
                values[count:2 * count],
                values[2 * count:],
                derivatives=True,
            )
            if has_prospective_2d_transport(d)
            else None
        )
        fluxes = self.fluxes(values) if prospective_eval is None else prospective_eval.fluxes
        rate = self.recombination_rate(n, p)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / thermal
        for offset, sign, density, flux in ((count, 1.0, n, fluxes.electron), (2 * count, -1.0, p, fluxes.hole)):
            if prospective_eval is not None:
                if offset == count:
                    phi_left, phi_right = prospective_eval.e_phi_left, prospective_eval.e_phi_right
                    z_left, z_right = prospective_eval.e_z_left, prospective_eval.e_z_right
                else:
                    phi_left, phi_right = prospective_eval.h_phi_left, prospective_eval.h_phi_right
                    z_left, z_right = prospective_eval.h_z_left, prospective_eval.h_z_right
            else:
                drive = sign * (values[offset + right] - values[offset + left])
                # Reuse the accepted cancellation-safe quasi-Fermi chain, without altering its source.
                phi_left, phi_right = quasi_fermi_potential_tangents(flux, drive, sign, thermal)
                if d.field_transport is not None:
                    evaluated_transport = d.field_transport.evaluate(values[:count])
                    diffusivity = (evaluated_transport.electron_diffusivity_m2_s if offset == count
                                   else evaluated_transport.hole_diffusivity_m2_s)
                    diffusivity_slope = (evaluated_transport.electron_d_diffusivity_d_field if offset == count
                             else evaluated_transport.hole_d_diffusivity_d_field)
                    field_chain = (flux.flux / diffusivity) * diffusivity_slope / d.geometry.face_spacing_m
                    phi_left = phi_left + field_chain
                    phi_right = phi_right - field_chain
                z_left = sign * flux.left_density_derivative * density[left]
                z_right = sign * flux.right_density_derivative * density[right]
            for face, (a, b) in enumerate(geometry.pairs):
                for orientation, node in ((-1.0, a), (1.0, b)):
                    weight = orientation * geometry.face_measures[face]
                    row = offset + node
                    matrix[row, a] += weight * phi_left[face]
                    matrix[row, b] += weight * phi_right[face]
                    matrix[row, offset + a] += weight * z_left[face]
                    matrix[row, offset + b] += weight * z_right[face]
            for node, area in enumerate(geometry.volumes):
                row = offset + node
                matrix[row, node] += area * source_phi[node]
                matrix[row, count + node] += area * source_n[node]
                matrix[row, 2 * count + node] += area * source_p[node]
        for node in d.geometry.contact_nodes:
            matrix.rows[node], matrix.data[node] = [int(node)], [1.0]
        sides = d.geometry.contact_nodes.reshape(2, d.geometry.nx)
        for carrier, offset, sign, density in (("n", count, 1.0, n), ("p", 2 * count, -1.0, p)):
            for side, nodes, inflow in zip(("left", "right"), sides, (1.0, -1.0)):
                channel = d.column.channel(carrier, side)
                if channel.kind == "fixed_reservoir":
                    for node in nodes:
                        matrix.rows[offset + node] = [int(node), int(offset + node)]
                        matrix.data[offset + node] = [sign / thermal, sign]
                else:
                    _, slope = self._contact_exchange(values, channel, nodes)
                    for node, width in zip(nodes, d.geometry.x_widths_m):
                        matrix[offset + node, node] += inflow * slope * sign * density[node] / thermal * width
                        matrix[offset + node, offset + node] += inflow * slope * sign * density[node] * width
        return matrix.tocsc()

    def current_density(self, u: ArrayLike) -> Vector:
        """Internal directed face currents [A/m2], x faces followed by y faces."""
        fluxes = self.fluxes(u)
        return fluxes.electron_current() + fluxes.hole_current()

    def cross_section_current_A_m(self, u: ArrayLike) -> Vector:
        g = self.device.geometry
        current_y = self.current_density(u)[g.x_face_count:].reshape(g.ny - 1, g.nx)
        return np.sum(current_y * g.x_widths_m[None, :], axis=1)

    def contact_particle_fluxes(self, u: ArrayLike) -> tuple[Vector, Vector]:
        """Y-min/y-max flux densities along +y, using each channel's own law.

        Fixed channels close the contact-cell balance, including lateral
        inflow and half-cell generation/recombination. Finite/blocking
        channels use their actual exchange; a failed row is never closed by
        substituting an inferred terminal flux.
        """
        d, g = self.device, self.device.geometry
        _, n, p = d.fields(u, voltage_V=self.voltage_V)
        source = self.generation_scale * d.generation_source_m3_s - self.recombination_rate(n, p, derivatives=False).rate
        pairs = (source * g.contract_geometry.volumes).reshape(g.shape)
        fluxes = self.fluxes(u)
        result = []
        horizontal_pairs = g.contract_geometry.pairs[:g.x_face_count]
        for carrier, flux in zip(("n", "p"), (fluxes.electron.flux, fluxes.hole.flux)):
            x_inflow = np.zeros(d.node_count)
            weighted = flux[:g.x_face_count] * g.contract_geometry.face_measures[:g.x_face_count]
            np.add.at(x_inflow, horizontal_pairs[:, 0], -weighted)
            np.add.at(x_inflow, horizontal_pairs[:, 1], weighted)
            x_inflow = x_inflow.reshape(g.shape)
            y_flux = flux[g.x_face_count:].reshape(g.ny - 1, g.nx)
            contact = np.stack((y_flux[0] - (x_inflow[0] + pairs[0]) / g.x_widths_m,
                                y_flux[-1] + (x_inflow[-1] + pairs[-1]) / g.x_widths_m))
            for side_index, (side, nodes) in enumerate(zip(("left", "right"), g.contact_nodes.reshape(2, g.nx))):
                channel = d.column.channel(carrier, side)
                if channel.kind != "fixed_reservoir":
                    contact[side_index], _ = self._contact_exchange(np.asarray(u, dtype=float), channel, nodes)
            result.append(contact)
        return result[0], result[1]

    def terminal_current_A_m(self, u: ArrayLike) -> Vector:
        """Integrated bottom/top currents along +y, per metre of out-of-plane depth."""
        electron, hole = self.contact_particle_fluxes(u)
        return np.sum((-Q * electron + Q * hole) * self.device.geometry.x_widths_m[None, :], axis=1)

    def terminal_current_density_A_m2(self, u: ArrayLike) -> Vector:
        return self.terminal_current_A_m(u) / self.device.geometry.lateral_length_m

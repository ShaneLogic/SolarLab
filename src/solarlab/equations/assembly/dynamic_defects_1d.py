"""Unified 1D dynamic-defect, two-sided interface-trap, and coupled defect-ion assembly.

All physics equations, storages, constraints, steady solves, BDF transient steps,
and small-signal AC linearizations share ONE :class:`~solarlab.equations.assembly.Assembler`
binding per compiled device. Nothing is assembled twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from solarlab.contracts import Geometry
from solarlab.contracts.units import (
    AREA,
    COULOMB,
    ELEMENTARY_CHARGE_C as Q,
    LENGTH,
    ONE,
    PARTICLE,
    SECOND,
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
    ClassicalDC1D,
    classical_dc_1d,
    populate_interface_fields,
)
from solarlab.equations.assembly.ionic_1d import (
    INVENTORY_UNIT,
    ImmobileProfile,
    InventoryReduction,
    IonDomain,
    IonStorage,
    IonTransport,
    K1,
)
from solarlab.equations.assembly.layout import (
    Block,
    ConstraintRows,
    NodeSet,
    SystemLayout,
)
from solarlab.equations.assembly.quasi_fermi import (
    ELECTRODE_QUASI_FERMI_1D_V1,
    quasi_fermi_potential_tangents,
)
from solarlab.equations.assembly.terms import (
    CELL_SOURCE,
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
from solarlab.equations.assembly.weighted import MEASURE
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dynamic_defect_1d import (
    CompiledDynamicDefectDevice1D,
    DynamicDefectState1D,
    DynamicInterfaceTrapBinding1D,
    StationaryDefectEvaluation1D,
    TwoSidedIonicComponent1D,
)
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.equations.ionic_closure import declared_immobile_profiles, ion_mobility_partition
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux
from solarlab.physics.defects import (
    DynamicBulkTrapEvaluation,
    DynamicBulkTrapMode,
    evaluate_dynamic_bulk_traps,
    logit_occupancy_secant,
)
from solarlab.physics.electrostatics import contact_displacement
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    cap_particle_flux_thermionic,
    solve_charged_two_sided_interface_qss,
    solve_fixed_occupancy_two_sided_interface,
    solve_two_sided_interface_qss,
)
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1, ion_face_flux, ion_face_flux_jacobian
from solarlab.physics.recombination.interface import (
    interface_srh_rate,
    solve_plane_closure,
)
from solarlab.physics.transport.electrochemical_flux import (
    CarrierFaceFluxes,
    carrier_particle_fluxes,
)
from solarlab.solvers.linear import (
    FrequencyOptions,
    FrequencySolution,
    FrequencySolveError,
    solve_frequency_systems,
)
from solarlab.solvers.nonlinear.problem import Matrix
from solarlab.solvers.time.bdf import BDFStep, HistoryPoint, StorageProblem

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]

NODES = "nodes"
POTENTIAL, ELECTRON, HOLE = "potential", "electron", "hole"
DISPLACEMENT_UNIT = COULOMB / AREA
CHARGE_UNIT = COULOMB / VOLUME
FLUX_UNIT = PARTICLE / (AREA * SECOND)
SOURCE_UNIT = PARTICLE / (VOLUME * SECOND)
DENSITY_UNIT = PARTICLE / VOLUME
TERM_SCHEMA = "solarlab.dynamic_defects_1d.terms.v1"

__all__ = [
    "AssembledDynamicDefect1DProblem",
    "DynamicBulkCarrierCaptureTerm",
    "DynamicBulkTrapRateTerm",
    "DynamicBulkTrapStorageTerm",
    "DynamicDefectACSweepPoint",
    "DynamicDefectACSweepResult",
    "DynamicDefectFields",
    "DynamicDefectSmallSignal1D",
    "DynamicDefectSpaceCharge",
    "DynamicDefectSteadyOptions",
    "DynamicDefectSteadyResult",
    "DynamicDefectStorage1D",
    "DynamicDefectSystem1D",
    "DynamicInterfaceTrapRateTerm",
    "DynamicInterfaceTrapStorageTerm",
    "DynamicInterfaceTrapTieTerm",
    "StationaryDefectRecombinationTerm",
    "dynamic_defect_step_accounting",
    "dynamic_defect_system_1d",
    "solve_dynamic_defect_steady_1d",
]


def _identity(*parts: object) -> str:
    return sha256(json.dumps((TERM_SCHEMA, *parts), separators=(",", ":")).encode()).hexdigest()


def _scatter_two_sided_tangent(
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


def _scatter_three_node_tangent(
    idx: int,
    g_left: float,
    g_mid: float,
    g_right: float,
    face_left: Vector,
    face_right: Vector,
    port_diag: Vector,
) -> None:
    if g_left != 0.0:
        face_left[idx - 1] += g_left
        port_diag[idx - 1] += g_left
    if g_mid != 0.0:
        port_diag[idx] += g_mid
    if g_right != 0.0:
        face_right[idx] -= g_right
        port_diag[idx] += g_right


def _populate_interfaces_with_dynamic_traps(
    dyn_device: CompiledDynamicDefectDevice1D,
    full_u: Vector,
    elec_u: Vector,
    phi: Vector,
    n: Vector,
    p: Vector,
    iface_occ: Vector,
    iface_omf: Vector,
    fields: dict[str, Any],
) -> bool:
    """Evaluate all static and dynamic heterointerfaces in one pass."""
    device = dyn_device.electronic_component
    model = device.interface_model
    if model is None and not dyn_device.dynamic_interface_traps:
        return True
    if not dyn_device.dynamic_interface_traps and (
        model is None or model.topology != "two_sided_trace"
    ):
        return populate_interface_fields(device, elec_u, phi, n, p, fields)

    count = device.node_count
    thermal = device.thermal_voltage_V
    electron_diffusivity = device.electron_diffusivity_faces_m2_s
    hole_diffusivity = device.hole_diffusivity_faces_m2_s
    transport = None if device.field_transport is None else device.field_transport.evaluate(phi)
    if transport is not None:
        electron_diffusivity = transport.electron_diffusivity_m2_s
        hole_diffusivity = transport.hole_diffusivity_m2_s

    zn = elec_u[count : 2 * count]
    zp = elec_u[2 * count : 3 * count]
    base = carrier_particle_fluxes(
        device.nodes_m,
        phi,
        n,
        p,
        electron_diffusivity,
        hole_diffusivity,
        thermal,
        device.parameters.band_structure,
        electron_electrochemical_step=np.diff(zn),
        hole_electrochemical_step=-np.diff(zp),
    )
    e_flux = np.array(base.electron.flux, dtype=float, copy=True)
    e_ld = np.array(base.electron.left_density_derivative, dtype=float, copy=True)
    e_rd = np.array(base.electron.right_density_derivative, dtype=float, copy=True)
    h_flux = np.array(base.hole.flux, dtype=float, copy=True)
    h_ld = np.array(base.hole.left_density_derivative, dtype=float, copy=True)
    h_rd = np.array(base.hole.right_density_derivative, dtype=float, copy=True)

    face_count = count - 1
    interior_count = count - 2
    e_port_val: Vector = np.zeros(interior_count, dtype=float)
    h_port_val: Vector = np.zeros(interior_count, dtype=float)
    g_port_val: Vector = np.zeros(interior_count, dtype=float)

    e_left_phi, e_right_phi = quasi_fermi_potential_tangents(base.electron, np.diff(zn), 1.0, thermal)
    h_left_phi, h_right_phi = quasi_fermi_potential_tangents(base.hole, -np.diff(zp), -1.0, thermal)
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

    e_port_phi: Vector = np.zeros(interior_count, dtype=float)
    e_port_zn: Vector = np.zeros(interior_count, dtype=float)
    e_port_zp: Vector = np.zeros(interior_count, dtype=float)
    h_port_phi: Vector = np.zeros(interior_count, dtype=float)
    h_port_zn: Vector = np.zeros(interior_count, dtype=float)
    h_port_zp: Vector = np.zeros(interior_count, dtype=float)
    g_port_phi: Vector = np.zeros(interior_count, dtype=float)
    g_port_zn: Vector = np.zeros(interior_count, dtype=float)
    g_port_zp: Vector = np.zeros(interior_count, dtype=float)
    g_left_zn: Vector = np.zeros(face_count, dtype=float)
    g_right_zn: Vector = np.zeros(face_count, dtype=float)
    g_left_zp: Vector = np.zeros(face_count, dtype=float)
    g_right_zp: Vector = np.zeros(face_count, dtype=float)

    dyn_by_idx = {
        itrap.interface_index: (pos, itrap)
        for pos, itrap in enumerate(dyn_device.dynamic_interface_traps)
    }

    try:
        if model is not None and model.topology == "two_sided_trace":
            for tspec in model.two_sided:
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
                if k in dyn_by_idx:
                    pos, itrap = dyn_by_idx[k]
                    occ_val = float(iface_occ[pos])
                    omf_val = float(iface_omf[pos])
                    inc_val = float(full_u[dyn_device.interface_trap_slices[pos].start])
                    df_eq_val = itrap.occupancy_minus_equilibrium_from_increment(inc_val)
                    eq_charge = EquilibriumReferencedSheetCharge(
                        equilibrium_occupancy=itrap.trap_spec.equilibrium_occupancy,
                        trap_density_m2=(
                            itrap.trap_density_m2 if itrap.has_sheet_charge else 0.0
                        ),
                    )
                    fixed_res = solve_fixed_occupancy_two_sided_interface(
                        itrap.trace_electrostatics,
                        itrap.physics,
                        bulk,
                        eq_charge,
                        occupancy=occ_val,
                        one_minus_occupancy=omf_val,
                        occupancy_minus_equilibrium=df_eq_val,
                    )
                    fields[f"dyn_iface:{k}"] = fixed_res
                    bflux = fixed_res.bulk_flux_m2_s
                    j_bulk_qf = fixed_res.bulk_flux_jacobian_qf
                    assert j_bulk_qf is not None
                    if itrap.has_sheet_charge:
                        sigma = fixed_res.incremental_sheet_charge_C_m2
                        g_port_val[l_node - 1] += itrap.weight_left * sigma
                        g_port_val[r_node - 1] += itrap.weight_right * sigma
                elif tspec.charge is not None:
                    charged = solve_charged_two_sided_interface_qss(
                        tspec.trace_electrostatics,
                        tspec.physics,
                        bulk,
                        tspec.charge,
                    )
                    bflux = charged.bulk_flux_m2_s
                    j_bulk_qf = charged.bulk_flux_jacobian_qf
                    sigma = charged.incremental_sheet_charge_C_m2
                    j_sigma_qf = charged.sheet_charge_jacobian_qf
                    assert j_bulk_qf is not None and j_sigma_qf is not None
                    g_port_val[l_node - 1] += tspec.weight_left * sigma
                    g_port_val[r_node - 1] += tspec.weight_right * sigma
                    s_phi_l = float(j_sigma_qf[0])
                    s_phi_r = float(j_sigma_qf[1])
                    s_zn_l = float(j_sigma_qf[2])
                    s_zn_r = float(j_sigma_qf[4])
                    s_zp_l = float(-j_sigma_qf[3])
                    s_zp_r = float(-j_sigma_qf[5])
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
                    j_bulk_qf = uncharged.bulk_flux_jacobian_qf
                    assert j_bulk_qf is not None

                e_flux[f] = float(bflux[0])
                h_flux[f] = float(bflux[1])
                e_ld[f] = float(j_bulk_qf[0, 2]) / float(n[l_node])
                e_rd[f] = float(j_bulk_qf[0, 4]) / float(n[r_node])
                h_ld[f] = float(j_bulk_qf[1, 3]) / float(p[l_node])
                h_rd[f] = float(j_bulk_qf[1, 5]) / float(p[r_node])
                e_port_val[r_node - 1] -= float(bflux[0] + bflux[2])
                h_port_val[r_node - 1] -= float(bflux[1] + bflux[3])

                for row_l, row_r, fl_phi, fr_phi, pd_phi, fl_zn, fr_zn, pd_zn, fl_zp, fr_zp, pd_zp in (
                    (0, 2, e_left_phi, e_right_phi, e_port_phi, e_left_zn, e_right_zn, e_port_zn, e_left_zp, e_right_zp, e_port_zp),
                    (1, 3, h_left_phi, h_right_phi, h_port_phi, h_left_zn, h_right_zn, h_port_zn, h_left_zp, h_right_zp, h_port_zp),
                ):
                    _scatter_two_sided_tangent(
                        f,
                        float(j_bulk_qf[row_l, 0]),
                        float(j_bulk_qf[row_l, 1]),
                        float(j_bulk_qf[row_r, 0]),
                        float(j_bulk_qf[row_r, 1]),
                        fl_phi, fr_phi, pd_phi,
                    )
                    _scatter_two_sided_tangent(
                        f,
                        float(j_bulk_qf[row_l, 2]),
                        float(j_bulk_qf[row_l, 4]),
                        float(j_bulk_qf[row_r, 2]),
                        float(j_bulk_qf[row_r, 4]),
                        fl_zn, fr_zn, pd_zn,
                    )
                    _scatter_two_sided_tangent(
                        f,
                        float(-j_bulk_qf[row_l, 3]),
                        float(-j_bulk_qf[row_l, 5]),
                        float(-j_bulk_qf[row_r, 3]),
                        float(-j_bulk_qf[row_r, 5]),
                        fl_zp, fr_zp, pd_zp,
                    )
    except (RuntimeError, FloatingPointError, ValueError):
        return False

    ef = replace(
        base.electron,
        flux=e_flux,
        left_density_derivative=e_ld,
        right_density_derivative=e_rd,
    )
    hf = replace(
        base.hole,
        flux=h_flux,
        left_density_derivative=h_ld,
        right_density_derivative=h_rd,
    )
    fields.update({
        "iface:e_flux": ef.flux,
        "iface:e_ld": ef.left_density_derivative,
        "iface:e_rd": ef.right_density_derivative,
        "iface:e_lpot": ef.left_potential_derivative,
        "iface:e_rpot": ef.right_potential_derivative,
        "iface:e_step": ef.energy_step,
        "iface:e_coeff": ef.coefficient,
        "iface:h_flux": hf.flux,
        "iface:h_ld": hf.left_density_derivative,
        "iface:h_rd": hf.right_density_derivative,
        "iface:h_lpot": hf.left_potential_derivative,
        "iface:h_rpot": hf.right_potential_derivative,
        "iface:h_step": hf.energy_step,
        "iface:h_coeff": hf.coefficient,
        "iface:e_left_phi": e_left_phi,
        "iface:e_right_phi": e_right_phi,
        "iface:e_left_zn": e_left_zn,
        "iface:e_right_zn": e_right_zn,
        "iface:e_left_zp": e_left_zp,
        "iface:e_right_zp": e_right_zp,
        "iface:h_left_phi": h_left_phi,
        "iface:h_right_phi": h_right_phi,
        "iface:h_left_zn": h_left_zn,
        "iface:h_right_zn": h_right_zn,
        "iface:h_left_zp": h_left_zp,
        "iface:h_right_zp": h_right_zp,
        "iface:e_port_val": e_port_val,
        "iface:e_port_phi": e_port_phi,
        "iface:e_port_zn": e_port_zn,
        "iface:e_port_zp": e_port_zp,
        "iface:h_port_val": h_port_val,
        "iface:h_port_phi": h_port_phi,
        "iface:h_port_zn": h_port_zn,
        "iface:h_port_zp": h_port_zp,
        "iface:g_port_val": g_port_val,
        "iface:g_port_phi": g_port_phi,
        "iface:g_port_zn": g_port_zn,
        "iface:g_port_zp": g_port_zp,
        "iface:g_left_phi": g_left_phi,
        "iface:g_right_phi": g_right_phi,
        "iface:g_left_zn": g_left_zn,
        "iface:g_right_zn": g_right_zn,
        "iface:g_left_zp": g_left_zp,
        "iface:g_right_zp": g_right_zp,
    })
    return True


@dataclass(frozen=True)
class DynamicDefectFields:
    """Single-pass constitutive evaluation for all active electronic, ionic, and defect physics."""

    device: CompiledDynamicDefectDevice1D

    @property
    def identity(self) -> str:
        return _identity("dynamic_defect_fields_v1", self.device.identity)

    def evaluate(self, layout: SystemLayout, u: Vector, inputs: OperatingInputs) -> FieldValues:
        dev = self.device
        comp = dev.electronic_component
        elec_u = u[dev.electronic_slice]

        try:
            if dev.ionic_component is not None:
                ion_u = u[: dev.ionic_component.state_size]
                phi, n, p, single = dev.ionic_component.fields_by_species(
                    ion_u, voltage_V=inputs.voltage_V
                )
            else:
                phi, n, p = comp.fields(elec_u, voltage_V=inputs.voltage_V)
                single = ()
        except (ValueError, FloatingPointError):
            return FieldValues({}, False)

        admissible = bool(
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(n))
            and np.all(np.isfinite(p))
            and not np.any(n <= 0.0)
            and not np.any(p <= 0.0)
        )
        fields: dict[str, Any] = {
            "potential_V": phi,
            "electron_density_m3": n,
            "hole_density_m3": p,
        }
        if not admissible:
            return FieldValues(fields, False)

        if dev.ionic_component is not None:
            domain = IonDomain(cast(CompiledIonicDevice1D, dev.ionic_component))
            fields.update(zip(domain.blocks, single))
            fields[domain.mass_field] = domain.coordinate_jacobian(single)

        if dev.stationary_defects is not None:
            try:
                fields["stationary_defects"] = dev.stationary_defects.evaluate(n, p)
            except (ValueError, FloatingPointError):
                return FieldValues(fields, False)

        if dev.dynamic_bulk_traps is not None:
            try:
                bulk_occ = dev.bulk_trap_occupancies(u)
                bulk_omf = dev.bulk_trap_one_minus_occupancies(u)
                fields["dynamic_bulk_traps"] = evaluate_dynamic_bulk_traps(
                    n,
                    p,
                    bulk_occ,
                    dev.dynamic_bulk_traps,
                    one_minus_occupancy=bulk_omf,
                )
            except (ValueError, FloatingPointError):
                return FieldValues(fields, False)

        iface_occ = dev.interface_trap_occupancies(u)
        iface_omf = dev.interface_trap_one_minus_occupancies(u)
        if dev.dynamic_interface_traps:
            fields["iface_trap_occ"] = iface_occ

        if comp.interface_model is not None or dev.dynamic_interface_traps:
            ok = _populate_interfaces_with_dynamic_traps(
                dev, u, elec_u, phi, n, p, iface_occ, iface_omf, fields
            )
            if not ok:
                return FieldValues(fields, False)

        return FieldValues(fields, True)


@dataclass(frozen=True)
class DynamicDefectSpaceCharge:
    """Total Gauss space charge folding carriers, dopants, ions, stationary defects, and bulk traps."""

    device: CompiledDynamicDefectDevice1D
    id: str = "potential.space_charge"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = CHARGE_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        deps: list[str] = [POTENTIAL, ELECTRON, HOLE]
        if self.device.ionic_component is not None:
            deps.extend(IonDomain(cast(CompiledIonicDevice1D, self.device.ionic_component)).blocks)
        if self.device.dynamic_bulk_traps is not None:
            deps.extend(
                f"trap:{idx}" for idx in range(len(self.device.dynamic_bulk_traps.modes))
            )
        return tuple(deps)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        dev = self.device
        comp = dev.electronic_component
        n = state.field("electron_density_m3")
        p = state.field("hole_density_m3")
        thermal = comp.thermal_voltage_V

        if dev.ionic_component is not None:
            domain = IonDomain(cast(CompiledIonicDevice1D, dev.ionic_component))
            pops = tuple(state.field(blk) for blk in domain.blocks)
            base_rho = dev.ionic_component.charge_ledger(
                n, p, pops[0], pops[1] if len(pops) == 2 else None
            ).space_charge_C_m3()
        else:
            base_rho = comp.charge_ledger(n, p).space_charge_C_m3()

        rho = np.array(base_rho, dtype=np.float64, copy=True)
        stat_ev: StationaryDefectEvaluation1D | None = None
        if dev.stationary_defects is not None:
            stat_ev = cast(StationaryDefectEvaluation1D, state.field("stationary_defects"))
            rho = rho + stat_ev.charge_density_C_m3

        trap_ev: DynamicBulkTrapEvaluation | None = None
        if dev.dynamic_bulk_traps is not None:
            trap_ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
            rho = rho + trap_ev.total_charge_density_C_m3

        if not derivatives:
            return {POTENTIAL: LocalValue(rho)}

        d_phi: Vector = -Q * (n + p) / thermal
        d_zn: Vector = -Q * n
        d_zp: Vector = -Q * p

        if stat_ev is not None:
            d_phi = d_phi + stat_ev.d_charge_dphi_fixed_qf_C_m3_V
            d_zn = d_zn + stat_ev.d_charge_dn_C * n
            d_zp = d_zp - stat_ev.d_charge_dp_C * p

        tangents: dict[str, Contribution] = {
            POTENTIAL: d_phi * MEASURE,
            ELECTRON: d_zn * MEASURE,
            HOLE: d_zp * MEASURE,
        }

        if dev.ionic_component is not None:
            domain = IonDomain(cast(CompiledIonicDevice1D, dev.ionic_component))
            ion_nodes = domain.device.ion_node_indices
            ratio = np.zeros(n.size, dtype=np.float64)
            ratio[ion_nodes] = state.embedding_ratio(domain.node_set)
            mass = state.field(domain.mass_field)
            for col, blk in enumerate(domain.blocks):
                partial = np.zeros(n.size, dtype=np.float64)
                if domain.shared:
                    partial[ion_nodes] = mass[:, 0, col] - mass[:, 1, col]
                    tangents[blk] = (Q * ratio * partial) * MEASURE
                else:
                    partial[ion_nodes] = mass[:, col, col]
                    sign = 1.0 if domain.species[col] == "positive" else -1.0
                    tangents[blk] = (sign * Q * ratio * partial) * MEASURE

        if dev.dynamic_bulk_traps is not None:
            assert trap_ev is not None
            f = trap_ev.occupancy
            omf = trap_ev.one_minus_occupancy
            df_du = f * omf
            for idx, mode in enumerate(dev.dynamic_bulk_traps.modes):
                sl = mode.flat_slice
                t_vec = np.zeros(n.size, dtype=np.float64)
                t_vec[mode.device_node_indices] = trap_ev.d_charge_df_C_m3[sl] * df_du[sl]
                tangents[f"trap:{idx}"] = t_vec

        return {POTENTIAL: LocalValue(rho, tangents)}


@dataclass(frozen=True)
class StationaryDefectRecombinationTerm:
    """Net recombination ``-R_stat(n, p)`` [m^-3 s^-1] from stationary QSS bulk defects."""

    device: CompiledDynamicDefectDevice1D
    id: str = "carrier.stationary_defect_recombination"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def identity(self) -> str:
        assert self.device.stationary_defects is not None
        return _identity(self.id, self.device.stationary_defects.identity_sha256)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        stat_ev = cast(StationaryDefectEvaluation1D, state.field("stationary_defects"))
        loss = -stat_ev.recombination_rate_m3_s
        if not derivatives:
            return {ELECTRON: LocalValue(loss), HOLE: LocalValue(loss)}
        n = state.field("electron_density_m3")
        p = state.field("hole_density_m3")
        thermal = self.device.thermal_voltage_V
        sn = -stat_ev.d_recombination_dn_s1 * n
        sp = stat_ev.d_recombination_dp_s1 * p
        sphi = (sn + sp) / thermal
        tangents: dict[str, Contribution] = {POTENTIAL: sphi, ELECTRON: sn, HOLE: sp}
        return {ELECTRON: LocalValue(loss, tangents), HOLE: LocalValue(loss, tangents)}


@dataclass(frozen=True)
class DynamicBulkCarrierCaptureTerm:
    """Non-equilibrium carrier capture ``(-C_n, -C_p)`` [m^-3 s^-1] into dynamic bulk traps."""

    device: CompiledDynamicDefectDevice1D
    id: str = "carrier.dynamic_bulk_trap_capture"
    kind: str = CELL_SOURCE
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        assert self.device.dynamic_bulk_traps is not None
        trap_blocks = tuple(
            f"trap:{idx}" for idx in range(len(self.device.dynamic_bulk_traps.modes))
        )
        return (POTENTIAL, ELECTRON, HOLE) + trap_blocks

    @property
    def identity(self) -> str:
        assert self.device.dynamic_bulk_traps is not None
        return _identity(self.id, self.device.dynamic_bulk_traps.model_identity_sha256)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        layout_bt = self.device.dynamic_bulk_traps
        assert layout_bt is not None
        ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
        loss_n = -ev.total_electron_capture_rate_m3_s
        loss_p = -ev.total_hole_capture_rate_m3_s
        if not derivatives:
            return {ELECTRON: LocalValue(loss_n), HOLE: LocalValue(loss_p)}

        n = state.field("electron_density_m3")
        p = state.field("hole_density_m3")
        thermal = self.device.thermal_voltage_V
        N = n.size

        dn_tot = np.zeros(N, dtype=np.float64)
        dp_tot = np.zeros(N, dtype=np.float64)
        np.add.at(dn_tot, layout_bt.device_node_indices, -ev.d_electron_capture_dn_s1)
        np.add.at(dp_tot, layout_bt.device_node_indices, -ev.d_hole_capture_dp_s1)

        e_zn = dn_tot * n
        e_phi = e_zn / thermal
        h_zp = -dp_tot * p
        h_phi = h_zp / thermal

        e_tan: dict[str, Contribution] = {
            POTENTIAL: e_phi,
            ELECTRON: e_zn,
            HOLE: np.zeros(N, dtype=np.float64),
        }
        h_tan: dict[str, Contribution] = {
            POTENTIAL: h_phi,
            ELECTRON: np.zeros(N, dtype=np.float64),
            HOLE: h_zp,
        }

        df_du = ev.occupancy * ev.one_minus_occupancy
        for idx, mode in enumerate(layout_bt.modes):
            sl = mode.flat_slice
            te = np.zeros(N, dtype=np.float64)
            th = np.zeros(N, dtype=np.float64)
            te[mode.device_node_indices] = -ev.d_electron_capture_df_m3_s[sl] * df_du[sl]
            th[mode.device_node_indices] = -ev.d_hole_capture_df_m3_s[sl] * df_du[sl]
            blk = f"trap:{idx}"
            e_tan[blk] = te
            h_tan[blk] = th

        return {
            ELECTRON: LocalValue(loss_n, e_tan),
            HOLE: LocalValue(loss_p, h_tan),
        }


@dataclass(frozen=True)
class DynamicBulkTrapRateTerm:
    """Net trap capture-minus-emission rate ``C_n - C_p`` [m^-3 s^-1] for one trap mode."""

    device: CompiledDynamicDefectDevice1D
    mode_index: int
    mode: DynamicBulkTrapMode
    kind: str = CELL_SOURCE
    unit: Unit = SOURCE_UNIT
    port: str | None = None

    @property
    def block(self) -> str:
        return f"trap:{self.mode_index}"

    @property
    def id(self) -> str:
        return f"{self.block}.rate"

    @property
    def support(self) -> str:
        return self.block

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, ELECTRON, HOLE, self.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.mode.mode_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
        sl = self.mode.flat_slice
        rate = ev.trap_storage_rate_m3_s[sl]
        if not derivatives:
            return {self.block: LocalValue(rate)}

        nodes = self.mode.device_node_indices
        n_loc = state.field("electron_density_m3")[nodes]
        p_loc = state.field("hole_density_m3")[nodes]
        thermal = self.device.thermal_voltage_V

        t_zn = ev.d_trap_storage_rate_dn_s1[sl] * n_loc
        t_zp = -ev.d_trap_storage_rate_dp_s1[sl] * p_loc
        t_phi = (t_zn + t_zp) / thermal
        df_du = ev.occupancy[sl] * ev.one_minus_occupancy[sl]
        t_u = ev.d_trap_storage_rate_df_m3_s[sl] * df_du

        return {
            self.block: LocalValue(
                rate,
                {
                    POTENTIAL: t_phi,
                    ELECTRON: t_zn,
                    HOLE: t_zp,
                    self.block: t_u,
                },
            )
        }


@dataclass(frozen=True)
class DynamicBulkTrapStorageTerm:
    """Occupied trap storage ``N_t * f`` [m^-3] and cancellation-free logit secant."""

    device: CompiledDynamicDefectDevice1D
    mode_index: int
    mode: DynamicBulkTrapMode
    kind: str = STORAGE
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def block(self) -> str:
        return f"trap:{self.mode_index}"

    @property
    def id(self) -> str:
        return f"{self.block}.storage"

    @property
    def support(self) -> str:
        return self.block

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.mode.mode_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
        sl = self.mode.flat_slice
        stored = ev.occupied_storage_m3[sl]
        if not derivatives:
            return {self.block: LocalValue(stored)}
        df_du = ev.occupancy[sl] * ev.one_minus_occupancy[sl]
        return {
            self.block: LocalValue(
                stored,
                {self.block: self.mode.population_density_m3 * df_du},
            )
        }

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        sl = self.mode.flat_slice
        ref = self.device.bulk_trap_reference_logit[sl]
        l_new = ref + state.block(self.block)
        l_old = ref + previous.block(self.block)
        df = logit_occupancy_secant(l_new, l_old)
        return {self.block: self.mode.population_density_m3 * df}


@dataclass(frozen=True)
class DynamicInterfaceTrapRateTerm:
    """Two-sided interface trap rate ``+(C_n - C_p)`` [m^-2 s^-1] on Row 0 of ``iface_trap:{k}``."""

    device: CompiledDynamicDefectDevice1D
    binding: DynamicInterfaceTrapBinding1D
    kind: str = FACE_FLUX
    unit: Unit = FLUX_UNIT
    port: str | None = None

    @property
    def id(self) -> str:
        return f"{self.binding.block}.rate"

    @property
    def support(self) -> str:
        return self.binding.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (POTENTIAL, ELECTRON, HOLE, self.binding.block)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.binding.interface_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        res = state.field(f"dyn_iface:{self.binding.interface_index}")
        flux = np.array([-float(res.trap_storage_rate_m2_s)], dtype=np.float64)
        if not derivatives:
            return {self.binding.block: FaceValue(flux)}

        j_qf = res.trap_storage_rate_jacobian_qf
        assert j_qf is not None
        sr_df = float(res.trap_storage_rate_derivative_occupancy)
        f_occ = float(res.occupancy)
        df_du = f_occ * float(res.one_minus_occupancy)

        # Face divergence scatters -flux to node 0 (left) and +flux to node 1 (right, constraint-dropped).
        # Thus -left_tangent lands in Row 0 vs left_node columns, and -right_tangent lands in Row 0 vs right_node columns!
        d_phi_l = float(j_qf[0])
        d_phi_r = float(j_qf[1])
        d_zn_l = float(j_qf[2])
        d_zn_r = float(j_qf[4])
        d_zp_l = float(-j_qf[3])
        d_zp_r = float(-j_qf[5])
        d_u_l = sr_df * df_du

        left: dict[str, Contribution] = {
            POTENTIAL: np.array([-d_phi_l]),
            ELECTRON: np.array([-d_zn_l]),
            HOLE: np.array([-d_zp_l]),
            self.binding.block: np.array([-d_u_l]),
        }
        right: dict[str, Contribution] = {
            POTENTIAL: np.array([-d_phi_r]),
            ELECTRON: np.array([-d_zn_r]),
            HOLE: np.array([-d_zp_r]),
            self.binding.block: np.array([0.0]),
        }
        return {self.binding.block: FaceValue(flux, left, right)}


@dataclass(frozen=True)
class DynamicInterfaceTrapTieTerm:
    """Constraint ``u_L - u_R == 0`` on Row 1 of ``iface_trap:{k}``."""

    device: CompiledDynamicDefectDevice1D
    binding: DynamicInterfaceTrapBinding1D
    kind: str = CONSTRAINT
    unit: Unit = ONE
    port: str | None = None

    @property
    def id(self) -> str:
        return f"{self.binding.block}.tie"

    @property
    def support(self) -> str:
        return self.binding.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.binding.interface_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        blk_u = state.block(self.binding.block)
        summand = np.array([blk_u[0], -blk_u[1]], dtype=np.float64)
        tangents: dict[str, Contribution] = (
            {self.binding.block: np.array([1.0, -1.0], dtype=np.float64)}
            if derivatives
            else {}
        )
        return {self.binding.block: ReductionValue(summand, None, tangents)}


@dataclass(frozen=True)
class DynamicInterfaceTrapStorageTerm:
    """Areal trap storage ``N_it * f`` [m^-2] on Row 0 of ``iface_trap:{k}``."""

    device: CompiledDynamicDefectDevice1D
    binding: DynamicInterfaceTrapBinding1D
    kind: str = STORAGE
    unit: Unit = DENSITY_UNIT
    port: str | None = None

    @property
    def id(self) -> str:
        return f"{self.binding.block}.storage"

    @property
    def support(self) -> str:
        return self.binding.node_set

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.binding.interface_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        res = state.field(f"dyn_iface:{self.binding.interface_index}")
        f_occ = float(res.occupancy)
        nt = self.binding.trap_density_m2
        # Support volumes on iface_trap:{k} are [1.0, 1.0] m, so [nt * f, 0.0] integrates to [nt * f, 0.0] m^-2.
        val = np.array([nt * f_occ, 0.0], dtype=np.float64)
        if not derivatives:
            return {self.binding.block: LocalValue(val)}
        df_du = f_occ * float(res.one_minus_occupancy)
        return {
            self.binding.block: LocalValue(
                val,
                {self.binding.block: np.array([nt * df_du, 0.0], dtype=np.float64)},
            )
        }

    def increment(self, state: EvaluationState, previous: EvaluationState) -> Mapping[str, Contribution]:
        ref = self.binding.reference_logit
        l_new = np.array([ref + float(state.block(self.binding.block)[0])])
        l_old = np.array([ref + float(previous.block(self.binding.block)[0])])
        df = float(logit_occupancy_secant(l_new, l_old)[0])
        nt = self.binding.trap_density_m2
        return {self.binding.block: np.array([nt * df, 0.0], dtype=np.float64)}


@dataclass(frozen=True)
class DynamicInterfaceCarrierFaceTangentTerm:
    """Deposits right-node bulk carrier tangent wrt ``u_L`` (and ``-right`` on ``l_node``)."""

    device: CompiledDynamicDefectDevice1D
    id: str = "carrier.dyn_iface_face_tangents"
    kind: str = FACE_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (ELECTRON, HOLE)
    unit: Unit = FLUX_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(itrap.block for itrap in self.device.dynamic_interface_traps)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        face_count = self.device.node_count - 1
        zero: Vector = np.zeros(face_count, dtype=np.float64)
        if not derivatives:
            return {ELECTRON: FaceValue(zero), HOLE: FaceValue(zero)}

        e_left: dict[str, Contribution] = {}
        e_right: dict[str, Contribution] = {}
        h_left: dict[str, Contribution] = {}
        h_right: dict[str, Contribution] = {}

        for itrap in self.device.dynamic_interface_traps:
            res = state.field(f"dyn_iface:{itrap.interface_index}")
            f_occ = float(res.occupancy)
            df_du = f_occ * float(res.one_minus_occupancy)
            dbf = res.bulk_flux_derivative_occupancy
            # Target derivatives on (l_node, r_node) wrt u_L are (A, B) = (-dbf_L * df_du, -dbf_R * df_du).
            # Face f=l_node scatters -left_val to l_node and +left_val to r_node in col(u_L)!
            b_e = -float(dbf[2]) * df_du
            b_h = -float(dbf[3]) * df_du
            ve: Vector = np.zeros(face_count, dtype=np.float64)
            vh: Vector = np.zeros(face_count, dtype=np.float64)
            ve[itrap.face] = b_e
            vh[itrap.face] = b_h
            e_left[itrap.block] = ve
            e_right[itrap.block] = zero
            h_left[itrap.block] = vh
            h_right[itrap.block] = zero

        return {
            ELECTRON: FaceValue(zero, e_left, e_right),
            HOLE: FaceValue(zero, h_left, h_right),
        }


@dataclass(frozen=True)
class DynamicInterfaceCarrierLeftPortTerm:
    """Completes ``A + B`` tangent on ``l_node`` wrt ``u_L`` so net tangent is ``(A, B)``."""

    device: CompiledDynamicDefectDevice1D
    binding: DynamicInterfaceTrapBinding1D
    block_name: str
    kind: str = PORT_FLUX
    support: str = NODES
    unit: Unit = FLUX_UNIT

    @property
    def id(self) -> str:
        return f"{self.block_name}.dyn_iface_left.{self.binding.interface_index}"

    @property
    def port(self) -> str:
        return self.binding.left_port

    @property
    def equations(self) -> tuple[str, ...]:
        return (self.block_name,)

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.binding.interface_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        zero: Vector = np.zeros(1, dtype=np.float64)
        if not derivatives:
            return {self.block_name: LocalValue(zero)}
        res = state.field(f"dyn_iface:{self.binding.interface_index}")
        f_occ = float(res.occupancy)
        df_du = f_occ * float(res.one_minus_occupancy)
        dbf = res.bulk_flux_derivative_occupancy
        if self.block_name == ELECTRON:
            a_plus_b = (-float(dbf[0]) - float(dbf[2])) * df_du
        else:
            a_plus_b = (-float(dbf[1]) - float(dbf[3])) * df_du
        return {
            self.block_name: LocalValue(
                zero,
                {self.binding.block: np.array([a_plus_b], dtype=np.float64)},
            )
        }


@dataclass(frozen=True)
class DynamicInterfaceGaussPortTerm:
    """Deposits ``g_port_val`` onto interior Gauss rows via ``interfaces`` port."""

    device: CompiledDynamicDefectDevice1D
    id: str = "potential.interfaces"
    kind: str = PORT_FLUX
    support: str = NODES
    port: str = "interfaces"
    equations: tuple[str, ...] = (POTENTIAL,)
    dependencies: tuple[str, ...] = (POTENTIAL, ELECTRON, HOLE)
    unit: Unit = DISPLACEMENT_UNIT

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, ELECTRODE_QUASI_FERMI_1D_V1)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        val = state.field("iface:g_port_val")
        if not derivatives:
            return {POTENTIAL: LocalValue(val)}
        return {
            POTENTIAL: LocalValue(
                val,
                {
                    POTENTIAL: state.field("iface:g_port_phi"),
                    ELECTRON: state.field("iface:g_port_zn"),
                    HOLE: state.field("iface:g_port_zp"),
                },
            )
        }


@dataclass(frozen=True)
class DynamicInterfaceGaussFaceTangentTerm:
    """Deposits right-node Gauss sheet-charge tangent ``w_R * d_sigma/du_L`` in ``col(u_L)``."""

    device: CompiledDynamicDefectDevice1D
    id: str = "potential.dyn_iface_face_tangents"
    kind: str = FACE_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = DISPLACEMENT_UNIT
    port: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(
            itrap.block
            for itrap in self.device.dynamic_interface_traps
            if itrap.has_sheet_charge
        )

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        face_count = self.device.node_count - 1
        zero: Vector = np.zeros(face_count, dtype=np.float64)
        if not derivatives:
            return {POTENTIAL: FaceValue(zero)}
        left: dict[str, Contribution] = {}
        right: dict[str, Contribution] = {}
        for itrap in self.device.dynamic_interface_traps:
            if not itrap.has_sheet_charge:
                continue
            res = state.field(f"dyn_iface:{itrap.interface_index}")
            f_occ = float(res.occupancy)
            d_sigma_du = float(res.d_sheet_charge_df_C_m2) * f_occ * float(res.one_minus_occupancy)
            b_phi = itrap.weight_right * d_sigma_du
            vg: Vector = np.zeros(face_count, dtype=np.float64)
            vg[itrap.face] = b_phi
            left[itrap.block] = vg
            right[itrap.block] = zero
        return {POTENTIAL: FaceValue(zero, left, right)}


@dataclass(frozen=True)
class DynamicInterfaceGaussLeftPortTerm:
    """Completes ``(w_L + w_R) * d_sigma/du_L`` on ``l_node`` so net is ``(w_L, w_R) * d_sigma/du_L``."""

    device: CompiledDynamicDefectDevice1D
    binding: DynamicInterfaceTrapBinding1D
    kind: str = PORT_FLUX
    support: str = NODES
    equations: tuple[str, ...] = (POTENTIAL,)
    unit: Unit = DISPLACEMENT_UNIT

    @property
    def id(self) -> str:
        return f"potential.dyn_iface_left.{self.binding.interface_index}"

    @property
    def port(self) -> str:
        return self.binding.left_port

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (self.binding.block,)

    @property
    def identity(self) -> str:
        return _identity(self.id, self.device.identity, self.binding.interface_id)

    def evaluate(self, state: EvaluationState, *, derivatives: bool) -> Mapping[str, TermValue]:
        zero: Vector = np.zeros(1, dtype=np.float64)
        if not derivatives:
            return {POTENTIAL: LocalValue(zero)}
        res = state.field(f"dyn_iface:{self.binding.interface_index}")
        f_occ = float(res.occupancy)
        d_sigma_du = float(res.d_sheet_charge_df_C_m2) * f_occ * float(res.one_minus_occupancy)
        a_plus_b = (self.binding.weight_left + self.binding.weight_right) * d_sigma_du
        return {
            POTENTIAL: LocalValue(
                zero,
                {self.binding.block: np.array([a_plus_b], dtype=np.float64)},
            )
        }


@dataclass(frozen=True)
class DynamicDefectSystem1D:
    """Unified ``Assembler`` binding for CompiledDynamicDefectDevice1D."""

    device: CompiledDynamicDefectDevice1D
    inventory_row_node: int | None = None
    assembler: Assembler = field(init=False, repr=False, compare=False)
    electronic: ClassicalDC1D = field(init=False, repr=False, compare=False)
    closure_rows: NDArray[np.bool_] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dev = self.device
        comp = dev.electronic_component
        electronic = classical_dc_1d(comp)

        base_layout = electronic.assembler.layout
        node_sets = list(base_layout.node_sets)
        blocks = list(base_layout.blocks)
        constraints = list(base_layout.constraints)

        nodes_support = electronic.assembler.mesh.support(NODES)
        nodes_ports = list(nodes_support.ports)
        existing_port_ids = {pt.id for pt in nodes_ports}
        if (comp.interface_model is not None or dev.dynamic_interface_traps) and (
            "interfaces" not in existing_port_ids
        ):
            interior = tuple(range(1, comp.node_count - 1))
            nodes_ports.append(Port("interfaces", interior, -1, np.ones(len(interior)), ONE))

        for itrap in dev.dynamic_interface_traps:
            nodes_ports.append(
                Port(itrap.left_port, (itrap.left_node,), -1, np.ones(1), ONE)
            )

        supports: list[SupportGeometry] = [
            SupportGeometry(NODES, nodes_support.geometry, tuple(nodes_ports))
        ]

        closure_terms: list[ImmobileProfile | InventoryReduction] = []
        ion_transports: list[Term] = []
        ion_storages: list[Term] = []

        if dev.ionic_component is not None:
            domain = IonDomain(cast(CompiledIonicDevice1D, dev.ionic_component))
            ion = domain.device
            node_sets.append(
                NodeSet(
                    domain.node_set,
                    ion.ion_node_count,
                    NODES,
                    tuple(int(nd) for nd in ion.ion_node_indices),
                )
            )
            blocks.extend(Block(blk, domain.node_set, ONE, FLUX_UNIT) for blk in domain.blocks)
            supports.append(
                SupportGeometry(
                    domain.node_set,
                    ion.ion_geometry.to_contract_geometry(
                        cell_support=domain.node_set,
                        face_support=domain.node_set + ".faces",
                    ),
                )
            )
            for sp in domain.species:
                ion_transports.append(IonTransport(domain, sp))
            ion_storages.append(IonStorage(domain))
            if self.inventory_row_node is not None:
                immobile = ion_mobility_partition(ion)[1]
                for sp in domain.species:
                    cterm: ImmobileProfile | InventoryReduction
                    if sp in immobile:
                        cterm, crows = ImmobileProfile(domain, sp), tuple(range(ion.ion_node_count))
                    else:
                        cterm, crows = InventoryReduction(domain, sp, K1), (self.inventory_row_node,)
                    constraints.append(ConstraintRows(cterm.block, crows, cterm.id, INVENTORY_UNIT))
                    closure_terms.append(cterm)

        trap_terms: list[Term] = []
        if dev.dynamic_bulk_traps is not None:
            cell_w = comp.geometry.cell_measures_m
            empty_pairs: NDArray[np.int64] = np.empty((0, 2), dtype=np.int64)
            empty_faces: Vector = np.empty(0, dtype=np.float64)
            for idx, mode in enumerate(dev.dynamic_bulk_traps.modes):
                ns_id = f"trap:{idx}"
                dev_indices = tuple(int(nd) for nd in mode.device_node_indices)
                node_sets.append(NodeSet(ns_id, len(dev_indices), NODES, dev_indices))
                blocks.append(Block(ns_id, ns_id, ONE, FLUX_UNIT))
                supports.append(
                    SupportGeometry(
                        ns_id,
                        Geometry(
                            volumes=cell_w[mode.device_node_indices],
                            pairs=empty_pairs,
                            face_measures=empty_faces,
                            volume_unit=LENGTH,
                            face_unit=ONE,
                            cell_support=ns_id,
                            face_support=ns_id + ".faces",
                        ),
                    )
                )
                trap_terms.append(DynamicBulkTrapRateTerm(dev, idx, mode))
                trap_terms.append(DynamicBulkTrapStorageTerm(dev, idx, mode))

        iface_terms: list[Term] = []
        if dev.dynamic_interface_traps:
            iface_terms.append(DynamicInterfaceCarrierFaceTangentTerm(dev))
            has_any_dyn_sheet = any(it.has_sheet_charge for it in dev.dynamic_interface_traps)
            if has_any_dyn_sheet:
                iface_terms.append(DynamicInterfaceGaussFaceTangentTerm(dev))
            pair_01 = np.array([[0, 1]], dtype=np.int64)
            unit_face: Vector = np.ones(1, dtype=np.float64)
            unit_vols: Vector = np.ones(2, dtype=np.float64)
            for itrap in dev.dynamic_interface_traps:
                ns_id = itrap.node_set
                node_sets.append(
                    NodeSet(ns_id, 2, NODES, (itrap.left_node, itrap.right_node))
                )
                blocks.append(Block(itrap.block, ns_id, ONE, FLUX_UNIT))
                constraints.append(
                    ConstraintRows(itrap.block, (1,), f"{itrap.block}.tie", ONE)
                )
                supports.append(
                    SupportGeometry(
                        ns_id,
                        Geometry(
                            volumes=unit_vols,
                            pairs=pair_01,
                            face_measures=unit_face,
                            volume_unit=LENGTH,
                            face_unit=ONE,
                            cell_support=ns_id,
                            face_support=ns_id + ".faces",
                        ),
                    )
                )
                iface_terms.append(DynamicInterfaceTrapRateTerm(dev, itrap))
                iface_terms.append(DynamicInterfaceTrapTieTerm(dev, itrap))
                iface_terms.append(DynamicInterfaceTrapStorageTerm(dev, itrap))
                iface_terms.append(DynamicInterfaceCarrierLeftPortTerm(dev, itrap, ELECTRON))
                iface_terms.append(DynamicInterfaceCarrierLeftPortTerm(dev, itrap, HOLE))
                if itrap.has_sheet_charge:
                    iface_terms.append(DynamicInterfaceGaussLeftPortTerm(dev, itrap))

        layout = SystemLayout(tuple(node_sets), tuple(blocks), tuple(constraints))
        if layout.size != dev.state_size:
            raise CapabilityError("assembled dynamic-defect layout size disagrees with compiled device")

        terms: list[Term] = []
        has_gauss_port = False
        for term in electronic.assembler.terms:
            if term.id == "potential.space_charge":
                terms.append(DynamicDefectSpaceCharge(dev))
            else:
                if term.id == "potential.interfaces":
                    has_gauss_port = True
                terms.append(term)

        if not has_gauss_port and any(it.has_sheet_charge for it in dev.dynamic_interface_traps):
            terms.append(DynamicInterfaceGaussPortTerm(dev))

        if dev.stationary_defects is not None:
            terms.append(StationaryDefectRecombinationTerm(dev))
        if dev.dynamic_bulk_traps is not None:
            terms.append(DynamicBulkCarrierCaptureTerm(dev))

        terms.extend(ion_transports)
        terms.extend(ion_storages)
        terms.extend(closure_terms)
        terms.extend(trap_terms)
        terms.extend(iface_terms)

        assembler = Assembler(layout, Mesh(tuple(supports)), DynamicDefectFields(dev), terms)
        closure_mask: NDArray[np.bool_] = np.zeros(layout.size, dtype=bool)
        for cterm in closure_terms:
            owned = layout.constraint_rows(cterm.id)
            closure_mask[layout.rows(owned.block, owned.nodes)] = True
        closure_mask.setflags(write=False)

        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "electronic", electronic)
        object.__setattr__(self, "closure_rows", closure_mask)

    @property
    def layout(self) -> SystemLayout:
        return self.assembler.layout

    @property
    def identity(self) -> str:
        return self.assembler.identity


def dynamic_defect_system_1d(
    device: CompiledDynamicDefectDevice1D,
    *,
    inventory_row_node: int | None = None,
) -> DynamicDefectSystem1D:
    """Build unified Assembler system for ``device``."""
    if not isinstance(device, CompiledDynamicDefectDevice1D):
        raise TypeError("dynamic_defect_system_1d requires CompiledDynamicDefectDevice1D")
    return DynamicDefectSystem1D(device, inventory_row_node)


@dataclass(frozen=True)
class AssembledDynamicDefect1DProblem:
    """Assembled residual, Jacobian, physical storage, and current observations."""

    device: CompiledDynamicDefectDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0
    inventory_row_node: int | None = None
    system: DynamicDefectSystem1D | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.device.electronic_component.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")
        if self.system is None:
            object.__setattr__(
                self,
                "system",
                dynamic_defect_system_1d(
                    self.device, inventory_row_node=self.inventory_row_node
                ),
            )

    @property
    def binding(self) -> DynamicDefectSystem1D:
        assert self.system is not None
        return self.system

    @property
    def inputs(self) -> OperatingInputs:
        return OperatingInputs(self.voltage_V, self.generation_scale)

    def assemble(
        self, u: ArrayLike, *, derivatives: bool = True, storage: bool = False
    ) -> AssembledSystem:
        return self.binding.assembler.evaluate(
            u, self.inputs, derivatives=derivatives, storage=storage
        )

    def residual(self, u: ArrayLike) -> Vector:
        assembled = self.assemble(u, derivatives=False)
        if not assembled.admissible or not np.all(np.isfinite(assembled.F)):
            raise FloatingPointError("dynamic-defect residual outside finite domain")
        return np.array(assembled.F, copy=True)

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        assembled = self.assemble(u, derivatives=True)
        if not assembled.admissible or assembled.J is None or not np.all(np.isfinite(assembled.J.data)):
            raise FloatingPointError("dynamic-defect Jacobian outside finite domain")
        return assembled.J

    def storage(self, u: ArrayLike) -> Vector:
        s, _ = self.binding.assembler.storage(u, self.inputs, derivatives=False)
        return np.array(s, copy=True)

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        _, su = self.binding.assembler.storage(u, self.inputs, derivatives=True)
        assert su is not None
        return su

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        """Signed DAE mass ``M = -S_u``."""
        return csc_matrix(-self.storage_tangent(u))

    def storage_increment(
        self,
        u: ArrayLike,
        old_problem: AssembledDynamicDefect1DProblem,
        old_u: ArrayLike,
    ) -> Vector:
        return self.binding.assembler.storage_increment(
            u, self.inputs, old_u, old_problem.inputs
        )

    def physical_storage(self, u: ArrayLike) -> Vector:
        return self.binding.assembler.physical_storage(u, self.inputs)

    def physical_increment(
        self,
        u: ArrayLike,
        old_problem: AssembledDynamicDefect1DProblem,
        old_u: ArrayLike,
    ) -> Vector:
        return self.binding.assembler.storage_increment(
            u, self.inputs, old_u, old_problem.inputs, physical=True
        )

    def carrier_face_fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("state outside admissible domain")
        return self.binding.electronic.transport.fluxes(state)

    def ion_face_fluxes(self, u: ArrayLike) -> dict[str, Vector]:
        if self.device.ionic_component is None:
            return {}
        state = self.binding.assembler.state(u, self.inputs)
        if state is None:
            raise FloatingPointError("state outside admissible domain")
        domain = IonDomain(cast(CompiledIonicDevice1D, self.device.ionic_component))
        ion = domain.device
        phi = state.field("potential_V")[ion.ion_node_indices]
        dx = np.diff(ion.ion_geometry.nodes_m)
        pops = tuple(state.field(blk) for blk in domain.blocks)
        steric = ion.steric_model == DIFFUSION_ONLY_STERIC_V1
        out: dict[str, Vector] = {}
        for idx, sp in enumerate(domain.species):
            cap = domain.capacity(sp)
            partner = pops[1 - idx] if domain.shared else None
            sign = 1.0 if sp == "positive" else -1.0
            out[sp] = ion_face_flux(
                phi,
                pops[idx],
                dx,
                getattr(ion, sp + "_ion_diffusivity_faces_m2_s"),
                self.device.thermal_voltage_V,
                cap[:-1],
                steric_diffusion_only=steric,
                P_lim_node=cap,
                P_other_node=partner,
                drift_sign=sign,
            )
        return out

    def dc_current_density_A_m2(self, u: ArrayLike) -> Vector:
        """All-face ``(N + 1,)`` DC conduction current density [A m^-2] along +x."""
        values = np.asarray(u, dtype=np.float64)
        comp = self.device.electronic_component
        N = comp.node_count
        state = self.binding.assembler.state(values, self.inputs)
        if state is None:
            raise FloatingPointError("state outside admissible domain")
        fluxes = self.binding.electronic.transport.fluxes(state)
        fn = fluxes.electron.flux
        fp = fluxes.hole.flux
        interior = -Q * fn + Q * fp

        n = state.field("electron_density_m3")
        p = state.field("hole_density_m3")
        rec = self.binding.electronic.recombination.rate(n, p, derivatives=False).rate
        gen = self.generation_scale * comp.generation.source_m3_s
        net_source_n = np.array(gen - rec, dtype=np.float64)
        net_source_p = np.array(gen - rec, dtype=np.float64)
        if self.device.stationary_defects is not None:
            stat_ev = cast(StationaryDefectEvaluation1D, state.field("stationary_defects"))
            net_source_n -= stat_ev.recombination_rate_m3_s
            net_source_p -= stat_ev.recombination_rate_m3_s
        if self.device.dynamic_bulk_traps is not None:
            trap_ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
            net_source_n -= trap_ev.total_electron_capture_rate_m3_s
            net_source_p -= trap_ev.total_hole_capture_rate_m3_s

        w = comp.geometry.cell_measures_m
        terminals: list[float] = []
        for side, node, face, orient in (("left", 0, 0, -1.0), ("right", N - 1, -1, 1.0)):
            j_term = 0.0
            for carrier, c_dens, c_flux, src, q_sign in (
                ("n", n, fn, net_source_n, -Q),
                ("p", p, fp, net_source_p, Q),
            ):
                ch = comp.channel(carrier, side)
                if ch.kind == "fixed_reservoir":
                    exch = float(c_flux[face]) + orient * float(w[node] * src[node])
                else:
                    exch_term = self.binding.electronic.exchanges[(carrier, side)]
                    offset = N if carrier == "n" else 2 * N
                    raw_exch, _ = exch_term.exchange(
                        float(values[node]), float(values[offset + node]), self.voltage_V
                    )
                    # CarrierExchange returns outward particle flux; orient along +x:
                    exch = -raw_exch if side == "left" else raw_exch
                j_term += q_sign * exch
            terminals.append(j_term)

        all_faces: Vector = np.r_[terminals[0], interior, terminals[1]]
        all_faces.setflags(write=False)
        return all_faces

    def terminal_current_density_A_m2(self, u: ArrayLike) -> float:
        """Signed terminal current density [A m^-2] in SolarLab passive/active J-V convention."""
        j_all = self.dc_current_density_A_m2(u)
        pos_idx = 0 if self.device.voltage_polarity > 0.0 else -1
        return float(self.device.voltage_polarity * j_all[pos_idx])


@dataclass(frozen=True)
class DynamicDefectStorage1D:
    """``StorageProblem`` adapter for BDF1/2 time integration of dynamic defects and ions."""

    problem: AssembledDynamicDefect1DProblem

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AssembledDynamicDefect1DProblem):
            raise TypeError("DynamicDefectStorage1D requires AssembledDynamicDefect1DProblem")
        if self.problem.inventory_row_node is not None:
            raise ValueError("transient storage problem must not carry steady inventory constraints")

    @property
    def device(self) -> CompiledDynamicDefectDevice1D:
        return self.problem.device

    @property
    def voltage_V(self) -> float:
        return self.problem.voltage_V

    @property
    def generation_scale(self) -> float:
        return self.problem.generation_scale

    def residual(self, u: Vector) -> Vector:
        return self.problem.residual(u)

    def jacobian(self, u: Vector) -> Matrix:
        return self.problem.jacobian(u)

    def storage(self, u: Vector) -> Vector:
        return self.problem.storage(u)

    def storage_tangent(self, u: Vector) -> Matrix:
        return self.problem.storage_tangent(u)

    def mass_matrix(self, u: Vector) -> Matrix:
        return self.problem.storage_jacobian(u)

    def storage_jacobian(self, u: Vector) -> Matrix:
        return self.mass_matrix(u)

    def _previous(self, old_problem: StorageProblem) -> AssembledDynamicDefect1DProblem:
        if not isinstance(old_problem, DynamicDefectStorage1D):
            raise TypeError("dynamic-defect storage history requires DynamicDefectStorage1D")
        old = old_problem.problem
        if old.binding.identity != self.problem.binding.identity:
            raise ValueError("dynamic-defect storage history requires identical bound system")
        return old

    def storage_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        return self.problem.storage_increment(u, self._previous(old_problem), old_u)

    def physical_storage(self, u: Vector) -> Vector:
        return self.problem.physical_storage(u)

    def physical_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        return self.problem.physical_increment(u, self._previous(old_problem), old_u)


def _displacement_profile_C_m2(
    device: CompiledDynamicDefectDevice1D, u: Vector, voltage_V: float
) -> Vector:
    """All-face ``(N + 1,)`` displacement profile [C m^-2], using ``D_left`` at two-sided planes."""
    comp = device.electronic_component
    N = comp.node_count
    phi, n, p = comp.fields(u[: 3 * N], voltage_V=voltage_V)
    rho = device.space_charge_density_C_m3(n, p, u)
    d_series = np.array(comp.poisson.face_displacement(phi), dtype=np.float64, copy=True)

    sheets = device.interface_sheet_charges_C_m2(u, voltage_V=voltage_V)
    w = comp.geometry.cell_measures_m
    cell_charge = np.array(rho * w, dtype=np.float64, copy=True)

    # Fold eliminated two-sided sheet charges into adjacent cells for contact_displacement
    # while recording D_left = D_series - w_L * sigma on the interface face itself!
    if comp.interface_model is not None and comp.interface_model.topology == "two_sided_trace":
        dyn_map = {it.interface_index: it for it in device.dynamic_interface_traps}
        for tspec in comp.interface_model.two_sided:
            k = tspec.interface_index
            sigma = float(sheets[k])
            if k in dyn_map:
                wl, wr = dyn_map[k].weight_left, dyn_map[k].weight_right
            else:
                wl, wr = tspec.weight_left, tspec.weight_right
            if wl != 0.0 or wr != 0.0:
                cell_charge[tspec.left_node] += wl * sigma
                cell_charge[tspec.right_node] += wr * sigma
                d_series[tspec.face] -= wl * sigma

    d_left_port = float(d_series[0] - cell_charge[0])
    d_right_port = float(d_series[-1] + cell_charge[-1])
    # Wait: if face 0 were a two-sided interface (never, since l_node >= 1), d_series[0] already has -wl*sigma.
    # On right port, face N-2: if face N-2 were a two-sided interface, D_right_side = D_series_orig + wr * sigma
    # = (d_series[N-2] + wl * sigma) + wr * sigma! Let's compute ports from original D_series + folded cell_charge!
    orig_series = comp.poisson.face_displacement(phi)
    ports = contact_displacement(orig_series, cell_charge / w, w)
    _ = (d_left_port, d_right_port)
    return np.r_[ports[0], d_series, ports[1]]


def dynamic_defect_step_accounting(step: BDFStep, u: ArrayLike) -> dict[str, Any]:
    """Conservative all-face Maxwell current accounting ``J_cond + J_ion + dD/dt`` on a BDFStep."""
    if not isinstance(step.endpoint, DynamicDefectStorage1D):
        raise TypeError("dynamic_defect_step_accounting requires DynamicDefectStorage1D")
    adapter = step.endpoint
    prob = adapter.problem
    dev = prob.device
    comp = dev.electronic_component
    N = comp.node_count
    values = np.asarray(u, dtype=np.float64)

    old = step.history[0]
    assert isinstance(old.problem, DynamicDefectStorage1D)
    coeffs = step.coefficients
    h = step.dt_s

    phys_rate = coeffs[0] * adapter.physical_increment(values, old.problem, old.u) / h
    if step.order == 2:
        older = step.history[1]
        assert isinstance(old.problem, DynamicDefectStorage1D)
        phys_rate = phys_rate - coeffs[2] * old.problem.physical_increment(old.u, older.problem, older.u) / h

    state = prob.binding.assembler.state(values, prob.inputs)
    if state is None:
        raise FloatingPointError("state outside admissible domain")
    fluxes = prob.binding.electronic.transport.fluxes(state)
    fn = fluxes.electron.flux
    fp = fluxes.hole.flux
    j_elec_interior = -Q * fn + Q * fp

    n = state.field("electron_density_m3")
    p = state.field("hole_density_m3")
    rec = prob.binding.electronic.recombination.rate(n, p, derivatives=False).rate
    gen = prob.generation_scale * comp.generation.source_m3_s
    net_src_n = np.array(gen - rec, dtype=np.float64)
    net_src_p = np.array(gen - rec, dtype=np.float64)
    if dev.stationary_defects is not None:
        stat_ev = cast(StationaryDefectEvaluation1D, state.field("stationary_defects"))
        net_src_n -= stat_ev.recombination_rate_m3_s
        net_src_p -= stat_ev.recombination_rate_m3_s
    if dev.dynamic_bulk_traps is not None:
        trap_ev = cast(DynamicBulkTrapEvaluation, state.field("dynamic_bulk_traps"))
        net_src_n -= trap_ev.total_electron_capture_rate_m3_s
        net_src_p -= trap_ev.total_hole_capture_rate_m3_s

    dn_dt_area = phys_rate[N : 2 * N]
    dp_dt_area = phys_rate[2 * N : 3 * N]
    w = comp.geometry.cell_measures_m

    terminals: list[float] = []
    for side, node, face, orient in (("left", 0, 0, -1.0), ("right", N - 1, -1, 1.0)):
        j_term = 0.0
        for carrier, c_flux, src, dstorage_dt, q_sign in (
            ("n", fn, net_src_n, dn_dt_area, -Q),
            ("p", fp, net_src_p, dp_dt_area, Q),
        ):
            ch = comp.channel(carrier, side)
            if ch.kind == "fixed_reservoir":
                exch = float(c_flux[face]) + orient * float(w[node] * src[node] - dstorage_dt[node])
            else:
                exch_term = prob.binding.electronic.exchanges[(carrier, side)]
                offset = N if carrier == "n" else 2 * N
                raw_exch, _ = exch_term.exchange(
                    float(values[node]), float(values[offset + node]), prob.voltage_V
                )
                exch = -raw_exch if side == "left" else raw_exch
            j_term += q_sign * exch
        terminals.append(j_term)

    j_cond: Vector = np.r_[terminals[0], j_elec_interior, terminals[1]]
    j_ion: Vector = np.zeros(N + 1, dtype=np.float64)
    if dev.ionic_component is not None:
        ion_fluxes = prob.ion_face_fluxes(values)
        ion_indices = dev.ionic_component.ion_node_indices
        for idx, sp in enumerate(dev.ionic_component.ion_species_names):
            sign = 1.0 if idx == 0 else -1.0
            j_ion[ion_indices[:-1] + 1] += sign * Q * ion_fluxes[sp]

    d_new = _displacement_profile_C_m2(dev, values, prob.voltage_V)
    d_old = _displacement_profile_C_m2(dev, old.u, old.problem.voltage_V)
    if step.order == 1:
        j_disp = coeffs[0] * (d_new - d_old) / h
    else:
        older = step.history[1]
        assert isinstance(older.problem, DynamicDefectStorage1D)
        d_older = _displacement_profile_C_m2(dev, older.u, older.problem.voltage_V)
        j_disp = (coeffs[0] * (d_new - d_old) - coeffs[2] * (d_old - d_older)) / h

    j_total = j_cond + j_ion + j_disp
    pos_idx = 0 if dev.voltage_polarity > 0.0 else -1
    j_terminal = float(dev.voltage_polarity * j_total[pos_idx])
    spread = float(np.ptp(j_total))
    return {
        "conduction_current_density_A_m2": j_cond,
        "ionic_current_density_A_m2": j_ion,
        "displacement_current_density_A_m2": j_disp,
        "total_current_density_A_m2": j_total,
        "terminal_current_density_A_m2": j_terminal,
        "maxwell_spread_A_m2": spread,
    }


@dataclass(frozen=True, slots=True)
class DynamicDefectSteadyOptions:
    """Damped/equilibrated Newton + continuation controls for steady dynamic-defect solves."""

    max_iterations: int = 80
    residual_tolerance: float = 1.0e-10
    step_tolerance: float = 1.0e-8
    max_logit_step: float = 2.0
    max_potential_step_V: float = 0.5
    inventory_row_node: int = 0


@dataclass(frozen=True)
class DynamicDefectSteadyResult:
    """Certified steady state of ``CompiledDynamicDefectDevice1D``."""

    state: DynamicDefectState1D
    problem: AssembledDynamicDefect1DProblem = field(repr=False)
    converged: bool
    iterations: int
    residual_inf_norm: float
    scaled_step_norm: float
    terminal_current_density_A_m2: float
    all_face_current_density_A_m2: Vector
    maxwell_spread_A_m2: float


def _damped_newton_solve(
    problem: AssembledDynamicDefect1DProblem,
    u_init: Vector,
    options: DynamicDefectSteadyOptions,
) -> tuple[Vector, bool, int, float, float]:
    N = problem.device.node_count
    u = np.array(u_init, dtype=np.float64, copy=True)

    # Enforce exact Dirichlet potential contacts on initial seed:
    phi_l, phi_r = problem.device.electronic_component.contact_potentials(problem.voltage_V)
    u[0] = phi_l
    u[N - 1] = phi_r

    res_norm = math.inf
    step_norm = math.inf
    for iteration in range(1, options.max_iterations + 1):
        sys = problem.assemble(u, derivatives=True)
        if not sys.admissible or sys.J is None or not np.all(np.isfinite(sys.F)):
            raise FloatingPointError("steady Newton encountered inadmissible initial/trial state")
        F = sys.F
        J = sys.J

        row_max = np.asarray(np.abs(J).max(axis=1).todense()).ravel()
        row_scale = np.where(row_max > 0.0, 1.0 / row_max, 1.0)
        f_scaled = row_scale * F
        res_norm = float(np.max(np.abs(f_scaled)))
        if res_norm <= options.residual_tolerance and (
            step_norm <= options.step_tolerance
            or res_norm <= 1.0e-4 * options.residual_tolerance
        ):
            return u, True, iteration - 1, res_norm, (0.0 if iteration == 1 else step_norm)

        j_row = diags(row_scale) @ J
        col_max = np.asarray(np.abs(j_row).max(axis=0).todense()).ravel()
        col_scale = np.where(col_max > 0.0, 1.0 / col_max, 1.0)
        try:
            factor = splu((j_row @ diags(col_scale)).tocsc())
            du = -col_scale * factor.solve(f_scaled)
        except RuntimeError:
            return u, False, iteration, res_norm, step_norm
        if not np.all(np.isfinite(du)):
            raise FloatingPointError("steady Newton linear step is nonfinite")

        # Clip giant initial potential / logit jumps gently before line search:
        max_dphi = float(np.max(np.abs(du[:N])))
        max_dlog = float(np.max(np.abs(du[N:]))) if du.size > N else 0.0
        clip = 1.0
        if max_dphi > options.max_potential_step_V:
            clip = min(clip, options.max_potential_step_V / max_dphi)
        if max_dlog > options.max_logit_step:
            clip = min(clip, options.max_logit_step / max_dlog)

        merit_0 = float(np.linalg.norm(f_scaled))
        alpha = clip
        accepted_u: Vector | None = None
        for _ in range(18):
            u_try = u + alpha * du
            try:
                sys_try = problem.assemble(u_try, derivatives=False)
            except FloatingPointError:
                alpha *= 0.5
                continue
            if not sys_try.admissible or not np.all(np.isfinite(sys_try.F)):
                alpha *= 0.5
                continue
            f_try_scaled = row_scale * sys_try.F
            merit_try = float(np.linalg.norm(f_try_scaled))
            if merit_try <= (1.0 - 1.0e-4 * alpha) * merit_0 or merit_try <= options.residual_tolerance:
                accepted_u = u_try
                break
            alpha *= 0.5

        if accepted_u is None:
            return u, False, iteration, res_norm, step_norm
        step_norm = float(np.max(np.abs(accepted_u - u)))
        u = accepted_u

        try:
            sys_end = problem.assemble(u, derivatives=True)
        except FloatingPointError:
            return u, False, iteration, res_norm, step_norm
        if sys_end.admissible and sys_end.J is not None and np.all(np.isfinite(sys_end.F)):
            rm = np.asarray(np.abs(sys_end.J).max(axis=1).todense()).ravel()
            rs = np.where(rm > 0.0, 1.0 / rm, 1.0)
            res_norm = float(np.max(np.abs(rs * sys_end.F)))
            if res_norm <= options.residual_tolerance and (
                step_norm <= options.step_tolerance or res_norm <= 1.0e-2 * options.residual_tolerance
            ):
                return u, True, iteration, res_norm, step_norm

    return u, False, options.max_iterations, res_norm, step_norm


def _scaled_newton(
    step: Any,
    u_init: Vector,
    *,
    tolerance: float = 1.0e-10,
    max_iterations: int = 40,
) -> tuple[Vector, bool, int]:
    """Row/column-equilibrated damped Newton on any object exposing ``residual`` / ``jacobian``."""
    u = np.array(u_init, dtype=np.float64, copy=True)
    for iteration in range(max_iterations + 1):
        try:
            f = np.asarray(step.residual(u), dtype=np.float64)
            jac = csc_matrix(step.jacobian(u))
        except (FloatingPointError, ValueError):
            return u, False, iteration
        row_max = np.asarray(np.abs(jac).max(axis=1).todense()).ravel()
        row_scale = np.where(row_max > 0.0, 1.0 / row_max, 1.0)
        fs = row_scale * f
        if float(np.max(np.abs(fs))) <= tolerance:
            return u, True, iteration
        if iteration == max_iterations:
            break
        j_row = diags(row_scale) @ jac
        col_max = np.asarray(np.abs(j_row).max(axis=0).todense()).ravel()
        col_scale = np.where(col_max > 0.0, 1.0 / col_max, 1.0)
        try:
            du = -col_scale * splu((j_row @ diags(col_scale)).tocsc()).solve(fs)
        except RuntimeError:
            return u, False, iteration
        if not np.all(np.isfinite(du)):
            return u, False, iteration
        merit = float(np.linalg.norm(fs))
        alpha = 1.0
        accepted: Vector | None = None
        for _ in range(24):
            u_try = u + alpha * du
            try:
                trial = row_scale * np.asarray(step.residual(u_try), dtype=np.float64)
            except (FloatingPointError, ValueError):
                alpha *= 0.5
                continue
            if np.all(np.isfinite(trial)) and float(np.linalg.norm(trial)) < merit:
                accepted = u_try
                break
            alpha *= 0.5
        if accepted is None:
            return u, False, iteration
        u = accepted
    return u, False, max_iterations


def _pseudo_transient_solve(
    problem: AssembledDynamicDefect1DProblem,
    u_init: Vector,
    options: DynamicDefectSteadyOptions,
) -> tuple[Vector, bool, int]:
    """Backward-Euler pseudo-time continuation to a steady state of ``problem``."""
    unconstrained_prob = (
        problem
        if problem.inventory_row_node is None
        else AssembledDynamicDefect1DProblem(
            problem.device,
            problem.voltage_V,
            problem.generation_scale,
            None,
            dynamic_defect_system_1d(problem.device, inventory_row_node=None),
        )
    )
    adapter = DynamicDefectStorage1D(unconstrained_prob)
    u = np.array(u_init, dtype=np.float64, copy=True)
    phi_l, phi_r = problem.device.electronic_component.contact_potentials(problem.voltage_V)
    u[0] = phi_l
    u[problem.device.node_count - 1] = phi_r
    try:
        _ = adapter.storage(u)
    except (FloatingPointError, ValueError):
        u = problem.device.initial_equilibrium_seed(voltage_V=problem.voltage_V)
    time_s = 0.0
    dt = 1.0e-12
    steps = 0
    while dt < 1.0e18 and steps < 400:
        previous = HistoryPoint(time_s, u, adapter.storage(u), adapter)
        step = BDFStep(adapter, (previous,), time_s + dt, 1)
        candidate, ok, _ = _scaled_newton(step, u, tolerance=1.0e-9, max_iterations=25)
        steps += 1
        if not ok:
            dt *= 0.25
            if dt < 1.0e-18:
                break
            continue
        u = candidate
        time_s += dt
        dt *= 3.0
        try:
            probe = problem.assemble(u, derivatives=True)
        except FloatingPointError:
            continue
        if probe.admissible and probe.J is not None and np.all(np.isfinite(probe.F)):
            row_max = np.asarray(np.abs(probe.J).max(axis=1).todense()).ravel()
            if float(np.max(np.abs(np.where(row_max > 0, 1.0 / row_max, 1.0) * probe.F))) <= 1.0e-8:
                return u, True, steps
    return u, False, steps


def _newton_or_pseudo_transient(
    problem: AssembledDynamicDefect1DProblem,
    u_init: Vector,
    options: DynamicDefectSteadyOptions,
) -> tuple[Vector, bool, int, float, float]:
    """Direct damped Newton first; pseudo-transient continuation and Newton polish on failure."""
    try:
        result = _damped_newton_solve(problem, u_init, options)
    except FloatingPointError:
        result = (np.array(u_init, dtype=np.float64, copy=True), False, 0, math.inf, math.inf)
    if result[1]:
        return result
    seed, reached, steps = _pseudo_transient_solve(problem, u_init, options)
    try:
        polished = _damped_newton_solve(problem, seed, options)
    except FloatingPointError:
        return seed, False, steps, math.inf, math.inf
    return polished[0], polished[1], steps + polished[2], polished[3], polished[4]


def solve_dynamic_defect_steady_1d(
    device: CompiledDynamicDefectDevice1D,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    initial_u: ArrayLike | None = None,
    options: DynamicDefectSteadyOptions | None = None,
    voltage_step_max_V: float = 0.04,
) -> DynamicDefectSteadyResult:
    """Solve steady DC working point (with automatic dark -> light -> bias continuation if needed)."""
    controls = DynamicDefectSteadyOptions() if options is None else options
    inv_row = controls.inventory_row_node if device.has_ions else None
    sys_binding = dynamic_defect_system_1d(device, inventory_row_node=inv_row)

    target_v = float(voltage_V)
    target_g = float(generation_scale)
    if not math.isfinite(target_v) or not math.isfinite(target_g) or target_g < 0.0:
        raise ValueError("voltage_V must be finite and generation_scale >= 0")

    def _advance_bias_interval(
        u_left: Vector, v_left: float, v_right: float, depth: int = 0
    ) -> tuple[Vector, bool, int, float, float]:
        u_seed = device.rebase_coordinates(u_left, v_left, v_right)
        phi_l, phi_r = device.electronic_component.contact_potentials(v_right)
        dv_contact = (phi_r - phi_l) - (u_seed[device.node_count - 1] - u_seed[0])
        nodes = device.nodes_m
        u_seed[: device.node_count] += dv_contact * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        prob_v = AssembledDynamicDefect1DProblem(
            device, v_right, target_g, inv_row, sys_binding
        )
        u_out, ok, step_iters, r_val, s_val = _newton_or_pseudo_transient(prob_v, u_seed, controls)
        if ok or depth >= 6 or abs(v_right - v_left) <= 1.0e-3:
            return u_out, ok, step_iters, r_val, s_val
        v_mid = 0.5 * (v_left + v_right)
        u_mid, ok1, it1, r1, s1 = _advance_bias_interval(u_left, v_left, v_mid, depth + 1)
        if not ok1:
            return u_mid, False, step_iters + it1, r1, s1
        u_end, ok2, it2, r2, s2 = _advance_bias_interval(u_mid, v_mid, v_right, depth + 1)
        return u_end, ok2, step_iters + it1 + it2, r2, s2

    total_iters = 0
    if initial_u is not None:
        u_curr = np.array(initial_u, dtype=np.float64, copy=True)
        prob_target = AssembledDynamicDefect1DProblem(
            device, target_v, target_g, inv_row, sys_binding
        )
        u_sol, conv, iters, r_norm, s_norm = _newton_or_pseudo_transient(prob_target, u_curr, controls)
        total_iters += iters
    else:
        u_curr = device.initial_equilibrium_seed(voltage_V=0.0)
        prob_eq = AssembledDynamicDefect1DProblem(device, 0.0, 0.0, inv_row, sys_binding)
        u_curr, conv, iters, r_norm, s_norm = _newton_or_pseudo_transient(prob_eq, u_curr, controls)
        total_iters += iters
        if not conv:
            raise RuntimeError(f"dark 0 V equilibrium failed to converge (res={r_norm:.3e})")

        if target_g > 0.0:
            for g_frac in (1.0e-3, 1.0e-1, 1.0) if target_g >= 0.1 else (1.0,):
                g_val = target_g * g_frac
                prob_g = AssembledDynamicDefect1DProblem(device, 0.0, g_val, inv_row, sys_binding)
                u_curr, conv, iters, r_norm, s_norm = _newton_or_pseudo_transient(prob_g, u_curr, controls)
                total_iters += iters
                if not conv:
                    raise RuntimeError(f"illumination continuation failed at G={g_val:.3g}")

        if abs(target_v) > 0.0:
            n_steps = max(1, int(math.ceil(abs(target_v) / max(voltage_step_max_V, 1.0e-3))))
            v_prev = 0.0
            for k in range(1, n_steps + 1):
                v_next = target_v * (k / n_steps)
                u_curr, conv, iters, r_norm, s_norm = _advance_bias_interval(u_curr, v_prev, v_next)
                total_iters += iters
                if not conv:
                    raise RuntimeError(f"voltage continuation failed at V={v_next:.4f} V")
                v_prev = v_next
        u_sol = u_curr
        prob_target = AssembledDynamicDefect1DProblem(
            device, target_v, target_g, inv_row, sys_binding
        )

    if not conv:
        raise RuntimeError(
            f"steady dynamic-defect solve did not converge: residual_inf_norm={r_norm:.3e}"
        )

    state_obj = device.state(u_sol, voltage_V=target_v)
    j_all = prob_target.dc_current_density_A_m2(u_sol)
    j_term = prob_target.terminal_current_density_A_m2(u_sol)
    spread = float(np.ptp(j_all))
    return DynamicDefectSteadyResult(
        state=state_obj,
        problem=prob_target,
        converged=True,
        iterations=total_iters,
        residual_inf_norm=r_norm,
        scaled_step_norm=s_norm,
        terminal_current_density_A_m2=j_term,
        all_face_current_density_A_m2=j_all,
        maxwell_spread_A_m2=spread,
    )


@dataclass(frozen=True)
class DynamicDefectACSweepPoint:
    """Single-frequency small-signal response and Maxwell/conservation diagnostics."""

    frequency_Hz: float
    linear_solution: FrequencySolution
    admittance_density_S_m2: complex
    impedance_area_ohm_m2: complex
    electronic_conduction_current_A_m2: ComplexVector
    ionic_conduction_current_A_m2: ComplexVector
    displacement_current_A_m2: ComplexVector
    total_current_A_m2: ComplexVector
    maxwell_spread_A_m2_V: float
    potential_perturbation_V: ComplexVector
    electron_density_perturbation_m3: ComplexVector
    hole_density_perturbation_m3: ComplexVector
    bulk_trap_occupancy_perturbation_per_V: ComplexVector
    bulk_trap_charge_perturbation_C_m3_per_V: ComplexVector
    interface_trap_occupancy_perturbation_per_V: ComplexVector
    interface_sheet_charge_perturbation_C_m2_per_V: ComplexVector
    ion_density_perturbation_m3: Mapping[str, ComplexVector]
    ion_inventory_perturbation_m2: Mapping[str, complex]
    ion_inventory_relative_per_V: Mapping[str, float]


@dataclass(frozen=True)
class DynamicDefectACSweepResult:
    """Frequency-sweep small-signal AC result on ``CompiledDynamicDefectDevice1D``."""

    frequencies_Hz: Vector
    voltage_phasor_V: complex
    admittance_density_S_m2: ComplexVector
    impedance_area_ohm_m2: ComplexVector
    maxwell_spread_A_m2_V: Vector
    points: tuple[DynamicDefectACSweepPoint, ...]
    active_species: tuple[str, ...]
    frozen_species: tuple[str, ...]
    operating_voltage_V: float
    generation_scale: float
    working_state_digest: str


@dataclass(frozen=True)
class DynamicDefectSmallSignal1D:
    """Exact small-signal AC linearization ``(J + i * omega * K) x = b`` about steady state.

    Uses the moving-reference boundary coordinate lift ``MOVING_LIFT`` so that
    prescribed contact perturbations vanish in the linear solve and heavily
    doped majority transport layers never suffer sparse Scharfetter-Gummel
    diagonal cancellation on either ``p-i-n`` or ``n-i-p`` polarity.
    """

    device: CompiledDynamicDefectDevice1D
    operating_state: DynamicDefectState1D
    generation_scale: float = 0.0
    problem: AssembledDynamicDefect1DProblem = field(init=False, repr=False)
    jacobian: csc_matrix = field(init=False, repr=False)
    storage_jacobian: csc_matrix = field(init=False, repr=False)
    reference_direction_per_V: Vector = field(init=False, repr=False)
    coordinate_lift_per_V: Vector = field(init=False, repr=False)
    working_state_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.operating_state.device_sha256 != self.device.sha256:
            raise ValueError("operating_state belongs to a different compiled device")
        # Unconstrained rows (inventory_row_node=None) so K = -S_u has full ion storage!
        prob = AssembledDynamicDefect1DProblem(
            self.device,
            voltage_V=self.operating_state.voltage_V,
            generation_scale=self.generation_scale,
            inventory_row_node=None,
        )
        u0 = self.operating_state.u
        object.__setattr__(self, "problem", prob)
        object.__setattr__(self, "jacobian", prob.jacobian(u0))
        object.__setattr__(self, "storage_jacobian", prob.storage_jacobian(u0))

        comp = self.device.electronic_component
        count = self.device.node_count
        vt = self.device.thermal_voltage_V
        before, after = comp.carrier_reference_energies(0.0), comp.carrier_reference_energies(1.0)
        direction: Vector = np.zeros(self.device.state_size, dtype=np.float64)
        for offset, zero, one in ((count, before[0], after[0]), (2 * count, before[1], after[1])):
            direction[offset : offset + count] = -(one - zero) / vt
        direction.setflags(write=False)
        object.__setattr__(self, "reference_direction_per_V", direction)

        lift: Vector = np.zeros(self.device.state_size, dtype=np.float64)
        lift[count - 1] = -comp.voltage_polarity
        for offset in (count, 2 * count):
            for node in (0, count - 1):
                lift[offset + node] = -lift[node] / vt + direction[offset + node]
        lift.setflags(write=False)
        object.__setattr__(self, "coordinate_lift_per_V", lift)

        object.__setattr__(
            self,
            "working_state_digest",
            sha256(np.ascontiguousarray(u0, dtype=np.float64).tobytes()).hexdigest(),
        )

    def _base_sg_carrier_particles(
        self, moving: ArrayLike, reference_drive: complex = 0.0
    ) -> tuple[ComplexVector, ComplexVector, CarrierFaceFluxes]:
        """Cancellation-free linearized SG particle fluxes across all faces."""
        comp = self.device.electronic_component
        N = self.device.node_count
        vt = self.device.thermal_voltage_V
        u0 = self.operating_state.u[: 3 * N]
        phi0 = self.operating_state.electronic_state.potential_V
        n0 = self.operating_state.electronic_state.electron_density_m3
        p0 = self.operating_state.electronic_state.hole_density_m3

        e_diff = comp.electron_diffusivity_faces_m2_s
        h_diff = comp.hole_diffusivity_faces_m2_s
        transport = None if comp.field_transport is None else comp.field_transport.evaluate(phi0)
        if transport is not None:
            e_diff = transport.electron_diffusivity_m2_s
            h_diff = transport.hole_diffusivity_m2_s

        base_fluxes = carrier_particle_fluxes(
            comp.nodes_m,
            phi0,
            n0,
            p0,
            e_diff,
            h_diff,
            vt,
            comp.parameters.band_structure,
            electron_electrochemical_step=np.diff(u0[N : 2 * N]),
            hole_electrochemical_step=-np.diff(u0[2 * N : 3 * N]),
        )
        before, after = comp.carrier_reference_energies(0.0), comp.carrier_reference_energies(1.0)
        values_x = np.asarray(moving, dtype=np.complex128)
        perts: list[ComplexVector] = []
        for offset, dens, flx, sign, ref in (
            (N, n0, base_fluxes.electron, 1.0, after[0] - before[0]),
            (2 * N, p0, base_fluxes.hole, -1.0, after[1] - before[1]),
        ):
            drive = sign * np.diff(u0[offset : offset + N])
            l_phi, _ = quasi_fermi_potential_tangents(flx, drive, sign, vt)
            if transport is not None:
                diff = (
                    transport.electron_diffusivity_m2_s
                    if offset == N
                    else transport.hole_diffusivity_m2_s
                )
                d_diff = (
                    transport.electron_d_diffusivity_d_field
                    if offset == N
                    else transport.hole_d_diffusivity_d_field
                )
                l_phi = l_phi + (flx.flux / diff) * d_diff / np.diff(comp.nodes_m)
            l_z = sign * flx.left_density_derivative * dens[:-1]
            z_blk = values_x[offset : offset + N]
            pert = (
                l_phi * (values_x[: N - 1] - values_x[1:N])
                + l_z * (z_blk[:-1] - z_blk[1:])
                + sign * flx.flux * (z_blk[1:] + (values_x[1:N] + ref * reference_drive) / vt)
            )
            perts.append(pert)
        return perts[0], perts[1], base_fluxes

    def _moving_voltage_partial(self) -> tuple[Vector, Vector]:
        """Grouped ``F_V + J (c - r)`` and ``-K (c - r)`` for ``c = coordinate_lift_per_V``."""
        dev = self.device
        comp = dev.electronic_component
        N = dev.node_count
        vt = dev.thermal_voltage_V
        w = comp.geometry.cell_measures_m
        u0 = self.operating_state.u
        moving = self.coordinate_lift_per_V

        state_eval = self.problem.binding.assembler.state(u0, self.problem.inputs)
        assert state_eval is not None

        phi0 = self.operating_state.electronic_state.potential_V
        n0 = self.operating_state.electronic_state.electron_density_m3
        p0 = self.operating_state.electronic_state.hole_density_m3
        before, after = comp.carrier_reference_energies(0.0), comp.carrier_reference_energies(1.0)
        references = (after[0] - before[0], after[1] - before[1])

        dlog_n: Vector = (moving[:N] + references[0]) / vt + moving[N : 2 * N]
        dlog_p: Vector = -((moving[:N] + references[1]) / vt + moving[2 * N : 3 * N])
        dlog_n[0] = dlog_n[N - 1] = 0.0
        dlog_p[0] = dlog_p[N - 1] = 0.0
        dn: Vector = n0 * dlog_n
        dp: Vector = p0 * dlog_p

        result: Vector = np.zeros(dev.state_size, dtype=np.float64)
        storage_drive: Vector = np.zeros(dev.state_size, dtype=np.float64)

        # 1. Gauss rows (0 .. N - 1):
        drho_stat: Vector = np.zeros(N, dtype=np.float64)
        d_rec_stat: Vector = np.zeros(N, dtype=np.float64)
        if dev.stationary_defects is not None:
            stat_ev = cast(StationaryDefectEvaluation1D, state_eval.field("stationary_defects"))
            drho_stat = stat_ev.d_charge_dn_C * dn + stat_ev.d_charge_dp_C * dp
            d_rec_stat = stat_ev.d_recombination_dn_s1 * dn + stat_ev.d_recombination_dp_s1 * dp

        result[:N] = comp.poisson.jacobian() @ moving[:N] + comp.poisson.charge_jacobian() * (
            Q * (dp - dn) + drho_stat
        )
        result[N - 1] += comp.voltage_polarity

        # 2. Dynamic bulk trap capture into carriers + trap rate rows:
        d_cap_n: Vector = np.zeros(N, dtype=np.float64)
        d_cap_p: Vector = np.zeros(N, dtype=np.float64)
        if dev.dynamic_bulk_traps is not None:
            trap_ev = cast(DynamicBulkTrapEvaluation, state_eval.field("dynamic_bulk_traps"))
            bt_nodes = dev.dynamic_bulk_traps.device_node_indices
            np.add.at(d_cap_n, bt_nodes, trap_ev.d_electron_capture_dn_s1 * dn[bt_nodes])
            np.add.at(d_cap_p, bt_nodes, trap_ev.d_hole_capture_dp_s1 * dp[bt_nodes])
            sl_bt = dev.bulk_trap_slice
            result[sl_bt] = w[bt_nodes] * (
                trap_ev.d_trap_storage_rate_dn_s1 * dn[bt_nodes]
                + trap_ev.d_trap_storage_rate_dp_s1 * dp[bt_nodes]
            )

        # 3. Carrier face fluxes and heterointerface channels:
        dfn_c, dfp_c, base_fluxes = self._base_sg_carrier_particles(moving, 1.0)
        dfn: Vector = np.real(dfn_c).copy()
        dfp: Vector = np.real(dfp_c).copy()
        e_iface_node: Vector = np.zeros(N, dtype=np.float64)
        h_iface_node: Vector = np.zeros(N, dtype=np.float64)

        if comp.interface_model is not None:
            if comp.interface_model.topology == "deduplicated":
                temp = float(comp.parameters.temperature_K)
                for tf in comp.interface_model.thermionic_faces:
                    f = tf.face
                    if tf.cap_electron:
                        cap_e = cap_particle_flux_thermionic(
                            float(base_fluxes.electron.flux[f]),
                            n_left_m3=float(n0[f]),
                            n_right_m3=float(n0[f + 1]),
                            delta_barrier_eV=tf.delta_Ec_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_n_A_m2_K2,
                            te_softness=comp.interface_model.te_softness,
                            regularization_mode=comp.interface_model.te_regularization_mode,
                        )
                        dfn[f] = (
                            cap_e.d_flux_d_sg_flux * dfn[f]
                            + cap_e.d_flux_d_n_left_m_s * dn[f]
                            + cap_e.d_flux_d_n_right_m_s * dn[f + 1]
                        )
                    if tf.cap_hole:
                        cap_h = cap_particle_flux_thermionic(
                            float(base_fluxes.hole.flux[f]),
                            n_left_m3=float(p0[f]),
                            n_right_m3=float(p0[f + 1]),
                            delta_barrier_eV=tf.delta_Ev_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_p_A_m2_K2,
                            te_softness=comp.interface_model.te_softness,
                            regularization_mode=comp.interface_model.te_regularization_mode,
                        )
                        dfp[f] = (
                            cap_h.d_flux_d_sg_flux * dfp[f]
                            + cap_h.d_flux_d_n_left_m_s * dp[f]
                            + cap_h.d_flux_d_n_right_m_s * dp[f + 1]
                        )
                for bare in comp.interface_model.bare_srh:
                    idx = bare.node
                    srh = interface_srh_rate(
                        float(n0[idx]),
                        float(p0[idx]),
                        v_n_m_s=bare.v_n_m_s,
                        v_p_m_s=bare.v_p_m_s,
                        ni_sq_m6=bare.ni_sq_m6,
                        n1_m3=bare.n1_m3,
                        p1_m3=bare.p1_m3,
                        clamp_nonnegative=False,
                    )
                    dsink = -(
                        float(np.real(srh.d_rate_dn_m_s)) * dn[idx]
                        + float(np.real(srh.d_rate_dp_m_s)) * dp[idx]
                    )
                    e_iface_node[idx] += dsink
                    h_iface_node[idx] += dsink
                for dspec in comp.interface_model.defect_srh:
                    idx = dspec.node
                    i_l, i_0, i_r = idx - 1, idx, idx + 1
                    if comp.interface_model.plane_closure and dspec.plane_params is not None:
                        el_raw = (float(phi0[i_0]) - float(phi0[i_l])) / vt
                        er_raw = (float(phi0[i_0]) - float(phi0[i_r])) / vt
                        fl = math.exp(max(-500.0, min(500.0, el_raw)))
                        fr = math.exp(max(-500.0, min(500.0, er_raw)))
                        nl_p = float(n0[i_l]) * fl
                        nr_p = float(n0[i_r]) * fr
                        pl_p = float(p0[i_l]) / fl
                        pr_p = float(p0[i_r]) / fr
                        plane_res = solve_plane_closure(
                            nl_p,
                            nr_p,
                            pl_p,
                            pr_p,
                            dspec.plane_params,
                            dspec.v_n_m_s,
                            dspec.v_p_m_s,
                            allow_generation=comp.interface_model.plane_generation,
                        )
                        chi_l = 1.0 if -500.0 < el_raw < 500.0 else 0.0
                        chi_r = 1.0 if -500.0 < er_raw < 500.0 else 0.0
                        d_el = chi_l * (moving[i_0] - moving[i_l]) / vt
                        d_er = chi_r * (moving[i_0] - moving[i_r]) / vt
                        dsink = -(
                            plane_res.d_rate_d_n_left_proj_m_s * nl_p * (dlog_n[i_l] + d_el)
                            + plane_res.d_rate_d_n_right_proj_m_s * nr_p * (dlog_n[i_r] + d_er)
                            + plane_res.d_rate_d_p_left_proj_m_s * pl_p * (dlog_p[i_l] - d_el)
                            + plane_res.d_rate_d_p_right_proj_m_s * pr_p * (dlog_p[i_r] - d_er)
                        )
                    else:
                        if comp.interface_model.plane_projection:
                            en_raw = (float(phi0[i_0]) - float(phi0[i_r])) / vt
                            ep_raw = (float(phi0[i_0]) - float(phi0[i_l])) / vt
                            fac_n = math.exp(max(-500.0, min(500.0, en_raw)))
                            fac_p = math.exp(-max(-500.0, min(500.0, ep_raw)))
                            chi_en = 1.0 if -500.0 < en_raw < 500.0 else 0.0
                            chi_ep = 1.0 if -500.0 < ep_raw < 500.0 else 0.0
                        else:
                            fac_n = fac_p = 1.0
                            chi_en = chi_ep = 0.0
                        n_eval = float(n0[i_r]) * fac_n
                        p_eval = float(p0[i_l]) * fac_p
                        ni_eval = dspec.ni_sq_eff_m6 * fac_n * fac_p
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
                        d_en = chi_en * (moving[i_0] - moving[i_r]) / vt
                        d_ep = chi_ep * (moving[i_0] - moving[i_l]) / vt
                        denom = dspec.v_n_m_s * (n_eval + dspec.n1_m3) + dspec.v_p_m_s * (
                            p_eval + dspec.p1_m3
                        )
                        unclamped = (
                            dspec.v_n_m_s > 0.0
                            and dspec.v_p_m_s > 0.0
                            and denom > 1.0e-300
                            and (n_eval * p_eval - ni_eval) >= 0.0
                        )
                        a_ni = (
                            -(dspec.v_n_m_s * dspec.v_p_m_s / denom) * ni_eval
                            if unclamped
                            else 0.0
                        )
                        dsink = -(
                            float(np.real(srh.d_rate_dn_m_s)) * n_eval * (dlog_n[i_r] + d_en)
                            + float(np.real(srh.d_rate_dp_m_s)) * p_eval * (dlog_p[i_l] - d_ep)
                            + a_ni * (d_en - d_ep)
                        )
                    e_iface_node[idx] += dsink
                    h_iface_node[idx] += dsink
            elif comp.interface_model.topology == "two_sided_trace":
                dyn_idx_map = {
                    it.interface_index: (pos, it)
                    for pos, it in enumerate(dev.dynamic_interface_traps)
                }
                com_vec = np.array(
                    [references[0] / vt, -references[1] / vt], dtype=np.float64
                )
                for tspec in comp.interface_model.two_sided:
                    k = tspec.interface_index
                    f = tspec.face
                    ln, rn = tspec.left_node, tspec.right_node
                    dfn[f] = 0.0
                    dfp[f] = 0.0
                    qf_rem = np.array(
                        [
                            moving[ln],
                            moving[rn],
                            moving[N + ln] if 0 < ln < N - 1 else (-moving[ln] / vt - com_vec[0]),
                            -moving[2 * N + ln] if 0 < ln < N - 1 else (moving[ln] / vt - com_vec[1]),
                            moving[N + rn] if 0 < rn < N - 1 else (-moving[rn] / vt - com_vec[0]),
                            -moving[2 * N + rn] if 0 < rn < N - 1 else (moving[rn] / vt - com_vec[1]),
                        ],
                        dtype=np.float64,
                    )
                    if k in dyn_idx_map:
                        pos, _ = dyn_idx_map[k]
                        res = state_eval.field(f"dyn_iface:{k}")
                        j_qf = res.bulk_flux_jacobian_qf
                        j_com = res.bulk_flux_common_qf
                        sr_qf = res.trap_storage_rate_jacobian_qf
                        sr_com = res.trap_storage_rate_common_qf
                        assert (
                            j_qf is not None
                            and j_com is not None
                            and sr_qf is not None
                            and sr_com is not None
                        )
                        dbflux = j_qf @ qf_rem + j_com @ com_vec
                        row_it = dev.interface_trap_slices[pos].start
                        result[row_it] = float(sr_qf @ qf_rem + sr_com @ com_vec)
                    else:
                        bulk = TwoSidedBulkState(
                            phi_left_V=float(u0[ln]),
                            phi_right_V=float(u0[rn]),
                            n_left_m3=float(n0[ln]),
                            p_left_m3=float(p0[ln]),
                            n_right_m3=float(n0[rn]),
                            p_right_m3=float(p0[rn]),
                            electron_electrochemical_step=float(u0[N + rn] - u0[N + ln]),
                            hole_electrochemical_step=float(-(u0[2 * N + rn] - u0[2 * N + ln])),
                        )
                        if tspec.charge is not None:
                            qres = solve_charged_two_sided_interface_qss(
                                tspec.trace_electrostatics, tspec.physics, bulk, tspec.charge
                            )
                            assert (
                                qres.bulk_flux_jacobian_qf is not None
                                and qres.bulk_flux_common_qf is not None
                                and qres.sheet_charge_jacobian_qf is not None
                                and qres.sheet_charge_common_qf is not None
                            )
                            dbflux = (
                                qres.bulk_flux_jacobian_qf @ qf_rem
                                + qres.bulk_flux_common_qf @ com_vec
                            )
                            dsig = float(
                                qres.sheet_charge_jacobian_qf @ qf_rem
                                + qres.sheet_charge_common_qf @ com_vec
                            )
                            if 0 < ln < N - 1:
                                result[ln] += tspec.weight_left * dsig
                            if 0 < rn < N - 1:
                                result[rn] += tspec.weight_right * dsig
                        else:
                            ures = solve_two_sided_interface_qss(
                                tspec.trace_electrostatics, tspec.physics, bulk
                            )
                            assert (
                                ures.bulk_flux_jacobian_qf is not None
                                and ures.bulk_flux_common_qf is not None
                            )
                            dbflux = (
                                ures.bulk_flux_jacobian_qf @ qf_rem
                                + ures.bulk_flux_common_qf @ com_vec
                            )
                    e_iface_node[ln] -= float(dbflux[0])
                    e_iface_node[rn] -= float(dbflux[2])
                    h_iface_node[ln] -= float(dbflux[1])
                    h_iface_node[rn] -= float(dbflux[3])

        # 4. Carrier continuity rows and carrier storage drive:
        rate = self.problem.binding.electronic.recombination.rate(n0, p0, derivatives=True)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        d_rec_base = rate.d_rate_dn * dn + rate.d_rate_dp * dp
        src_n = -(d_rec_base + d_rec_stat + d_cap_n)
        src_p = -(d_rec_base + d_rec_stat + d_cap_p)

        for carrier, offset, flux_v, src_v, iface_v, d_dens, sign, ref in (
            ("n", N, dfn, src_n, e_iface_node, dn, 1.0, references[0]),
            ("p", 2 * N, dfp, src_p, h_iface_node, dp, -1.0, references[1]),
        ):
            rows = -np.diff(np.r_[0.0, flux_v, 0.0]) + w * src_v + iface_v
            s_drive = w * d_dens
            for side, node in (("left", 0), ("right", N - 1)):
                if comp.channel(carrier, side).kind == "fixed_reservoir":
                    rows[node] = sign * ((moving[node] + ref) / vt + moving[offset + node])
                    s_drive[node] = 0.0
            result[offset : offset + N] = rows
            storage_drive[offset : offset + N] = s_drive

        # 5. Mobile-ion potential coupling at contact endpoint:
        if dev.ionic_component is not None:
            sl_ion = dev.ion_slice
            result[sl_ion] = self.jacobian[sl_ion, :N] @ moving[:N]

        return result, storage_drive

    def species_partition(
        self,
        *,
        freeze_ions: bool = False,
        freeze_bulk_traps: bool = False,
        freeze_interface_traps: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(active_species, frozen_species)`` ledger for diagnostic transparency."""
        active: list[str] = ["electrons", "holes"]
        frozen: list[str] = []
        if self.device.ionic_component is not None:
            immobile = set(
                declared_immobile_profiles(cast(CompiledIonicDevice1D, self.device.ionic_component))
            )
            for sp in self.device.ionic_component.ion_species_names:
                label = f"ion:{sp}"
                if freeze_ions or sp in immobile:
                    frozen.append(label)
                else:
                    active.append(label)
        if self.device.dynamic_bulk_traps is not None:
            for mode in self.device.dynamic_bulk_traps.modes:
                label = f"bulk_trap:{mode.mode_id}"
                if freeze_bulk_traps:
                    frozen.append(label)
                else:
                    active.append(label)
        for itrap in self.device.dynamic_interface_traps:
            label = f"interface_trap:{itrap.interface_id}"
            if freeze_interface_traps:
                frozen.append(label)
            else:
                active.append(label)
        return tuple(active), tuple(frozen)

    def solve_sweep(
        self,
        frequencies_Hz: ArrayLike,
        *,
        voltage_phasor_V: complex = 1.0,
        options: FrequencyOptions | None = None,
        freeze_ions: bool = False,
        freeze_bulk_traps: bool = False,
        freeze_interface_traps: bool = False,
    ) -> DynamicDefectACSweepResult:
        """Solve complex sparse small-signal AC sweep with selective species freezing.

        Note on diagnostic freezing closures:
        ``freeze_ions=True`` clamps mobile-ion perturbations ``dP = 0`` and
        remains a locally Maxwell-conserving PDE at every frequency. By
        contrast, ``freeze_bulk_traps=True`` and ``freeze_interface_traps=True``
        clamp trap occupancy perturbations ``df = 0`` while retaining linearized
        carrier capture perturbations ``dC_n != dC_p`` as ``omega -> inf``
        asymptotic reference operators rather than finite-frequency
        Maxwell-conserving PDEs.
        """
        controls = (
            FrequencyOptions(relative_tolerance=1.0e-14, max_refinements=4)
            if options is None
            else options
        )
        freqs = np.asarray(frequencies_Hz, dtype=np.float64)
        if (
            freqs.ndim != 1
            or freqs.size == 0
            or not np.all(np.isfinite(freqs))
            or np.any(freqs < 0.0)
            or np.unique(freqs).size != freqs.size
        ):
            raise ValueError("frequencies_Hz must be 1D distinct finite non-negative Hz")
        if freqs.size > controls.max_frequencies:
            controls = FrequencyOptions(
                max_frequencies=int(freqs.size),
                relative_tolerance=controls.relative_tolerance,
                max_refinements=controls.max_refinements,
            )
        fallback_controls = (
            FrequencyOptions(max_frequencies=controls.max_frequencies, max_refinements=4)
            if options is None
            else None
        )

        def _solve_freq_batch(
            j_mat: csc_matrix,
            k_mat: csc_matrix,
            rhs_vec: ComplexVector,
            freq_vec: Vector,
            rate_vec: ComplexVector,
        ) -> tuple[FrequencySolution, ...]:
            try:
                return solve_frequency_systems(
                    j_mat,
                    k_mat,
                    rhs_vec,
                    freq_vec,
                    options=controls,
                    fixed_coordinates=known,
                    rhs_rate=rate_vec,
                )
            except FrequencySolveError:
                if fallback_controls is None:
                    raise
                return solve_frequency_systems(
                    j_mat,
                    k_mat,
                    rhs_vec,
                    freq_vec,
                    options=fallback_controls,
                    fixed_coordinates=known,
                    rhs_rate=rate_vec,
                )

        voltage = complex(voltage_phasor_V)
        if not np.isfinite(voltage) or voltage == 0.0:
            raise ValueError("voltage_phasor_V must be finite and nonzero")

        dev = self.device
        comp = dev.electronic_component
        N = dev.node_count

        # Build base J, K with moving-boundary coordinate lift and any frozen species pinned to 0:
        j_base = self.jacobian.copy().tolil()
        k_base = self.storage_jacobian.copy().tolil()

        partial, storage_drive = self._moving_voltage_partial()
        rhs: ComplexVector = (-partial * voltage).astype(np.complex128)
        rhs_rate: ComplexVector = (storage_drive * voltage).astype(np.complex128)

        known: dict[int, complex] = {0: 0.0j, N - 1: 0.0j}
        for carrier, offset in (("n", N), ("p", 2 * N)):
            if comp.channel(carrier, "left").kind == "fixed_reservoir":
                known[offset] = 0.0j
            if comp.channel(carrier, "right").kind == "fixed_reservoir":
                known[offset + N - 1] = 0.0j

        frozen_indices: list[int] = []
        if dev.ionic_component is not None and freeze_ions:
            sl_ion = dev.ion_slice
            frozen_indices.extend(range(sl_ion.start, sl_ion.stop))
        if dev.dynamic_bulk_traps is not None and freeze_bulk_traps:
            sl_bt = dev.bulk_trap_slice
            frozen_indices.extend(range(sl_bt.start, sl_bt.stop))
        if dev.dynamic_interface_traps and freeze_interface_traps:
            sl_it = dev.interface_trap_slice
            frozen_indices.extend(range(sl_it.start, sl_it.stop))

        for idx in frozen_indices:
            j_base.rows[idx], j_base.data[idx] = [idx], [1.0]
            k_base.rows[idx], k_base.data[idx] = [], []
            known[idx] = 0.0j
            rhs[idx] = 0.0j
            rhs_rate[idx] = 0.0j

        has_mobile_ions = dev.ionic_component is not None and not freeze_ions
        if has_mobile_ions:
            ion_dev: CompiledIonicDevice1D | TwoSidedIonicComponent1D | None = dev.ionic_component
            assert ion_dev is not None
            immobile_profiles = declared_immobile_profiles(cast(CompiledIonicDevice1D, ion_dev))
            for sp, blk_sl in zip(ion_dev.ion_species_names, ion_dev.ion_species_slices):
                if sp in immobile_profiles:
                    for row in range(cast(int, blk_sl.start), cast(int, blk_sl.stop)):
                        grad = -self.storage_jacobian[row, :].toarray()[0]
                        nz = np.flatnonzero(grad)
                        j_base.rows[row], j_base.data[row] = nz.tolist(), grad[nz].tolist()
                        k_base.rows[row], k_base.data[row] = [], []
                        rhs[row] = 0.0j
                        rhs_rate[row] = 0.0j
                else:
                    diff_faces = getattr(ion_dev, sp + "_ion_diffusivity_faces_m2_s")
                    if np.all(diff_faces > 0.0):
                        row0 = cast(int, blk_sl.start)
                        grad = -np.asarray(self.storage_jacobian[blk_sl, :].sum(axis=0)).ravel()
                        nz = np.flatnonzero(grad)
                        j_base.rows[row0], j_base.data[row0] = nz.tolist(), grad[nz].tolist()
                        k_base.rows[row0], k_base.data[row0] = [], []
                        rhs[row0] = 0.0j
                        rhs_rate[row0] = 0.0j
                    elif np.any(freqs == 0.0):
                        raise CapabilityError(
                            "0 Hz mobile ions require positive diffusivity across domain"
                        )

        j_csc = j_base.tocsc()
        k_csc = k_base.tocsc()
        solutions_by_idx = list(
            _solve_freq_batch(
                j_csc,
                k_csc,
                rhs,
                freqs,
                rhs_rate,
            )
        )

        points: list[DynamicDefectACSweepPoint] = []
        admittances = np.empty(freqs.size, dtype=np.complex128)
        impedances = np.empty(freqs.size, dtype=np.complex128)
        spreads = np.empty(freqs.size, dtype=np.float64)

        for f_idx, sol in enumerate(solutions_by_idx):
            pt = self._observe_point(
                sol,
                voltage=voltage,
                freeze_ions=freeze_ions,
                freeze_bulk_traps=freeze_bulk_traps,
                freeze_interface_traps=freeze_interface_traps,
            )
            points.append(pt)
            admittances[f_idx] = pt.admittance_density_S_m2
            impedances[f_idx] = pt.impedance_area_ohm_m2
            spreads[f_idx] = pt.maxwell_spread_A_m2_V

        for arr in (admittances, impedances, spreads):
            arr.setflags(write=False)
        freqs_out = np.array(freqs, dtype=np.float64, copy=True)
        freqs_out.setflags(write=False)
        active_sp, frozen_sp = self.species_partition(
            freeze_ions=freeze_ions,
            freeze_bulk_traps=freeze_bulk_traps,
            freeze_interface_traps=freeze_interface_traps,
        )
        return DynamicDefectACSweepResult(
            frequencies_Hz=freqs_out,
            voltage_phasor_V=voltage,
            admittance_density_S_m2=admittances,
            impedance_area_ohm_m2=impedances,
            maxwell_spread_A_m2_V=spreads,
            points=tuple(points),
            active_species=active_sp,
            frozen_species=frozen_sp,
            operating_voltage_V=self.operating_state.voltage_V,
            generation_scale=self.generation_scale,
            working_state_digest=self.working_state_digest,
        )

    def _observe_point(
        self,
        sol: FrequencySolution,
        *,
        voltage: complex,
        freeze_ions: bool,
        freeze_bulk_traps: bool,
        freeze_interface_traps: bool,
    ) -> DynamicDefectACSweepPoint:
        dev = self.device
        comp = dev.electronic_component
        N = dev.node_count
        vt = dev.thermal_voltage_V
        u0 = self.operating_state.u
        x = sol.solution
        lift = self.coordinate_lift_per_V
        omega = 2.0 * math.pi * sol.frequency_Hz

        state_eval = self.problem.binding.assembler.state(u0, self.problem.inputs)
        assert state_eval is not None

        phi_x: ComplexVector = x[:N] + lift[:N] * voltage
        before, after = comp.carrier_reference_energies(0.0), comp.carrier_reference_energies(1.0)
        ref_n, ref_p = after[0] - before[0], after[1] - before[1]

        dlog_n: ComplexVector = np.empty(N, dtype=np.complex128)
        dlog_p: ComplexVector = np.empty(N, dtype=np.complex128)
        dlog_n[0] = x[0] / vt + x[N]
        dlog_n[N - 1] = x[N - 1] / vt + x[2 * N - 1]
        dlog_n[1 : N - 1] = (x[1 : N - 1] + ref_n * voltage) / vt + x[N + 1 : 2 * N - 1]
        dlog_p[0] = -(x[0] / vt + x[2 * N])
        dlog_p[N - 1] = -(x[N - 1] / vt + x[3 * N - 1])
        dlog_p[1 : N - 1] = -((x[1 : N - 1] + ref_p * voltage) / vt + x[2 * N + 1 : 3 * N - 1])

        n0 = self.operating_state.electronic_state.electron_density_m3
        p0 = self.operating_state.electronic_state.hole_density_m3
        dn: ComplexVector = n0 * dlog_n
        dp: ComplexVector = p0 * dlog_p

        # 1. Dynamic bulk trap perturbations:
        if dev.dynamic_bulk_traps is not None:
            trap_ev = cast(DynamicBulkTrapEvaluation, state_eval.field("dynamic_bulk_traps"))
            u_bt_x = x[dev.bulk_trap_slice]
            df_bt = (
                np.zeros(dev.dynamic_bulk_traps.size, dtype=np.complex128)
                if freeze_bulk_traps
                else (trap_ev.occupancy * trap_ev.one_minus_occupancy) * u_bt_x
            )
            drho_bt_local = trap_ev.d_charge_df_C_m3 * df_bt
            drho_bt: ComplexVector = np.zeros(N, dtype=np.complex128)
            np.add.at(drho_bt, dev.dynamic_bulk_traps.device_node_indices, drho_bt_local)
            d_cap_n_local = (
                trap_ev.d_electron_capture_dn_s1 * dn[dev.dynamic_bulk_traps.device_node_indices]
                + trap_ev.d_electron_capture_df_m3_s * df_bt
            )
            d_cap_p_local = (
                trap_ev.d_hole_capture_dp_s1 * dp[dev.dynamic_bulk_traps.device_node_indices]
                + trap_ev.d_hole_capture_df_m3_s * df_bt
            )
            d_cap_n: ComplexVector = np.zeros(N, dtype=np.complex128)
            d_cap_p: ComplexVector = np.zeros(N, dtype=np.complex128)
            np.add.at(d_cap_n, dev.dynamic_bulk_traps.device_node_indices, d_cap_n_local)
            np.add.at(d_cap_p, dev.dynamic_bulk_traps.device_node_indices, d_cap_p_local)
        else:
            df_bt = np.empty(0, dtype=np.complex128)
            drho_bt = np.zeros(N, dtype=np.complex128)
            d_cap_n = np.zeros(N, dtype=np.complex128)
            d_cap_p = np.zeros(N, dtype=np.complex128)

        # 2. Stationary defect perturbations:
        if dev.stationary_defects is not None:
            stat_ev = cast(StationaryDefectEvaluation1D, state_eval.field("stationary_defects"))
            drho_stat = stat_ev.d_charge_dn_C * dn + stat_ev.d_charge_dp_C * dp
            d_rec_stat = stat_ev.d_recombination_dn_s1 * dn + stat_ev.d_recombination_dp_s1 * dp
        else:
            drho_stat = np.zeros(N, dtype=np.complex128)
            d_rec_stat = np.zeros(N, dtype=np.complex128)

        # 3. Dynamic interface trap perturbations:
        if dev.dynamic_interface_traps:
            df_it: ComplexVector = np.zeros(len(dev.dynamic_interface_traps), dtype=np.complex128)
            dsigma_it: ComplexVector = np.zeros(len(dev.dynamic_interface_traps), dtype=np.complex128)
            for idx, (itrap, sl) in enumerate(
                zip(dev.dynamic_interface_traps, dev.interface_trap_slices)
            ):
                if not freeze_interface_traps:
                    res = state_eval.field(f"dyn_iface:{itrap.interface_index}")
                    f_occ = float(res.occupancy)
                    df_it[idx] = f_occ * float(res.one_minus_occupancy) * x[sl.start]
                    if itrap.has_sheet_charge:
                        dsigma_it[idx] = float(res.d_sheet_charge_df_C_m2) * df_it[idx]
        else:
            df_it = np.empty(0, dtype=np.complex128)
            dsigma_it = np.empty(0, dtype=np.complex128)

        # 4. Electronic face particle flux perturbations (grouping exact SG differences):
        dfn_solve, dfp_solve, base_fluxes = self._base_sg_carrier_particles(x, 0.0)
        dfn_lift, dfp_lift, _ = self._base_sg_carrier_particles(lift, 1.0)
        dfn: ComplexVector = dfn_solve + dfn_lift * voltage
        dfp: ComplexVector = dfp_solve + dfp_lift * voltage

        # Override thermionic / two-sided interface faces with their exact analytical tangents:
        if comp.interface_model is not None:
            if comp.interface_model.topology == "deduplicated":
                temp = float(comp.parameters.temperature_K)
                for tf in comp.interface_model.thermionic_faces:
                    f = tf.face
                    if tf.cap_electron:
                        cap_e = cap_particle_flux_thermionic(
                            float(base_fluxes.electron.flux[f]),
                            n_left_m3=float(n0[f]),
                            n_right_m3=float(n0[f + 1]),
                            delta_barrier_eV=tf.delta_Ec_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_n_A_m2_K2,
                            te_softness=comp.interface_model.te_softness,
                            regularization_mode=comp.interface_model.te_regularization_mode,
                        )
                        dfn[f] = (
                            cap_e.d_flux_d_sg_flux * dfn[f]
                            + cap_e.d_flux_d_n_left_m_s * dn[f]
                            + cap_e.d_flux_d_n_right_m_s * dn[f + 1]
                        )
                    if tf.cap_hole:
                        cap_h = cap_particle_flux_thermionic(
                            float(base_fluxes.hole.flux[f]),
                            n_left_m3=float(p0[f]),
                            n_right_m3=float(p0[f + 1]),
                            delta_barrier_eV=tf.delta_Ev_eV,
                            temperature_K=temp,
                            a_star_A_m2_K2=tf.a_star_p_A_m2_K2,
                            te_softness=comp.interface_model.te_softness,
                            regularization_mode=comp.interface_model.te_regularization_mode,
                        )
                        dfp[f] = (
                            cap_h.d_flux_d_sg_flux * dfp[f]
                            + cap_h.d_flux_d_n_left_m_s * dp[f]
                            + cap_h.d_flux_d_n_right_m_s * dp[f + 1]
                        )
            elif comp.interface_model.topology == "two_sided_trace":
                com_vec_obs = np.array([ref_n / vt, -ref_p / vt], dtype=np.float64)
                dyn_idx_map = {
                    it.interface_index: (pos, it)
                    for pos, it in enumerate(dev.dynamic_interface_traps)
                }
                for tspec in comp.interface_model.two_sided:
                    k = tspec.interface_index
                    f = tspec.face
                    ln, rn = tspec.left_node, tspec.right_node
                    u_solve_vec = np.array(
                        [x[ln], x[rn], x[N + ln], -x[2 * N + ln], x[N + rn], -x[2 * N + rn]],
                        dtype=np.complex128,
                    )
                    qf_lift_rem = np.array(
                        [
                            lift[ln],
                            lift[rn],
                            lift[N + ln] if 0 < ln < N - 1 else (-lift[ln] / vt - com_vec_obs[0]),
                            -lift[2 * N + ln] if 0 < ln < N - 1 else (lift[ln] / vt - com_vec_obs[1]),
                            lift[N + rn] if 0 < rn < N - 1 else (-lift[rn] / vt - com_vec_obs[0]),
                            -lift[2 * N + rn] if 0 < rn < N - 1 else (lift[rn] / vt - com_vec_obs[1]),
                        ],
                        dtype=np.float64,
                    )
                    if k in dyn_idx_map:
                        pos, _ = dyn_idx_map[k]
                        res = state_eval.field(f"dyn_iface:{k}")
                        jb_qf = res.bulk_flux_jacobian_qf
                        jb_com = res.bulk_flux_common_qf
                        assert jb_qf is not None and jb_com is not None
                        dbf = res.bulk_flux_derivative_occupancy
                        occ_term_n = float(dbf[0]) * df_it[pos]
                        occ_term_p = float(dbf[1]) * df_it[pos]
                    else:
                        bulk = TwoSidedBulkState(
                            phi_left_V=float(u0[ln]),
                            phi_right_V=float(u0[rn]),
                            n_left_m3=float(n0[ln]),
                            p_left_m3=float(p0[ln]),
                            n_right_m3=float(n0[rn]),
                            p_right_m3=float(p0[rn]),
                            electron_electrochemical_step=float(u0[N + rn] - u0[N + ln]),
                            hole_electrochemical_step=float(-(u0[2 * N + rn] - u0[2 * N + ln])),
                        )
                        if tspec.charge is not None:
                            qres = solve_charged_two_sided_interface_qss(
                                tspec.trace_electrostatics, tspec.physics, bulk, tspec.charge
                            )
                            jb_qf = qres.bulk_flux_jacobian_qf
                            jb_com = qres.bulk_flux_common_qf
                        else:
                            ures = solve_two_sided_interface_qss(
                                tspec.trace_electrostatics, tspec.physics, bulk
                            )
                            jb_qf = ures.bulk_flux_jacobian_qf
                            jb_com = ures.bulk_flux_common_qf
                        assert jb_qf is not None and jb_com is not None
                        occ_term_n = 0.0j
                        occ_term_p = 0.0j
                    lift_flux = jb_qf @ qf_lift_rem + jb_com @ com_vec_obs
                    dfn[f] = complex(
                        jb_qf[0] @ u_solve_vec + float(lift_flux[0]) * voltage + occ_term_n
                    )
                    dfp[f] = complex(
                        jb_qf[1] @ u_solve_vec + float(lift_flux[1]) * voltage + occ_term_p
                    )

        # Terminal electronic conduction currents:
        rate = self.problem.binding.electronic.recombination.rate(n0, p0, derivatives=True)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        d_rec_base = rate.d_rate_dn * dn + rate.d_rate_dp * dp
        d_src_n = -(d_rec_base + d_rec_stat + d_cap_n)
        d_src_p = -(d_rec_base + d_rec_stat + d_cap_p)
        w = comp.geometry.cell_measures_m

        terminals: list[complex] = []
        for side, node, face, orient in (("left", 0, 0, -1.0), ("right", N - 1, -1, 1.0)):
            j_term = 0.0j
            for carrier, d_dens, d_flux, d_src, q_sign in (
                ("n", dn, dfn, d_src_n, -Q),
                ("p", dp, dfp, d_src_p, Q),
            ):
                ch = comp.channel(carrier, side)
                if ch.kind == "fixed_reservoir":
                    exch = d_flux[face] + orient * w[node] * (d_src[node] - 1j * omega * d_dens[node])
                else:
                    slope = boundary_particle_flux(ch, ch.reservoir_density_m3).density_derivative
                    exch = slope * d_dens[node]
                j_term += q_sign * exch
            terminals.append(j_term)

        j_elec: ComplexVector = np.r_[terminals[0], -Q * dfn + Q * dfp, terminals[1]]

        # 5. Mobile-ion density, inventory, charge, and conduction current perturbations:
        j_ion: ComplexVector = np.zeros(N + 1, dtype=np.complex128)
        drho_ion: ComplexVector = np.zeros(N, dtype=np.complex128)
        ion_dens_map: dict[str, ComplexVector] = {}
        ion_inv_map: dict[str, complex] = {}
        ion_inv_rel_map: dict[str, float] = {}

        if dev.ionic_component is not None:
            ion_dev = dev.ionic_component
            ion_st = self.operating_state.ionic_state
            assert ion_st is not None
            mass = ion_dev.ion_coordinate_jacobian_m3(
                u0[: ion_dev.state_size], voltage_V=self.operating_state.voltage_V
            )
            coords = np.stack([x[sl] for sl in ion_dev.ion_species_slices], axis=1)
            dens_perts = np.sum(mass * coords[:, None, :], axis=2).T
            pops = (
                (ion_st.positive_ion_density_m3,)
                if ion_dev.ion_species_count == 1
                else (ion_st.positive_ion_density_m3, ion_st.negative_ion_density_m3)
            )
            w_ratio = ion_dev.ion_geometry.cell_measures_m / w[ion_dev.ion_node_indices]
            for s_idx, sp in enumerate(ion_dev.ion_species_names):
                sign = 1.0 if s_idx == 0 else -1.0
                d_ion = (
                    np.zeros(ion_dev.ion_node_count, dtype=np.complex128)
                    if freeze_ions
                    else dens_perts[s_idx]
                )
                ion_dens_map[sp] = d_ion
                weighted = ion_dev.ion_geometry.cell_measures_m * d_ion
                inv_c = complex(
                    math.fsum(weighted.real.tolist()), math.fsum(weighted.imag.tolist())
                )
                target_inv = math.fsum(
                    (
                        ion_dev.ion_geometry.cell_measures_m
                        * getattr(ion_dev, sp + "_ion_reference_m3")
                    ).tolist()
                )
                ion_inv_map[sp] = inv_c
                ion_inv_rel_map[sp] = abs(inv_c) / (target_inv * abs(voltage))
                drho_ion[ion_dev.ion_node_indices] += sign * Q * w_ratio * d_ion

                if not freeze_ions:
                    tan = ion_face_flux_jacobian(
                        ion_st.electronic_state.potential_V[ion_dev.ion_node_indices],
                        pops[s_idx],
                        np.diff(ion_dev.ion_geometry.nodes_m),
                        getattr(ion_dev, sp + "_ion_diffusivity_faces_m2_s"),
                        vt,
                        getattr(ion_dev, sp + "_ion_site_limit_m3")[:-1],
                        steric_diffusion_only=ion_dev.steric_model == DIFFUSION_ONLY_STERIC_V1,
                        P_lim_node=getattr(ion_dev, sp + "_ion_site_limit_m3"),
                        P_other_node=(
                            pops[1 - s_idx]
                            if ion_dev.ion_site_contract == "shared_finite_sites"
                            else None
                        ),
                        drift_sign=sign,
                    )
                    other = (
                        dens_perts[1 - s_idx]
                        if ion_dev.ion_species_count == 2
                        else np.zeros_like(d_ion)
                    )
                    j_int = (
                        tan.potential_left_derivative * phi_x[ion_dev.ion_node_indices[:-1]]
                        + tan.potential_right_derivative * phi_x[ion_dev.ion_node_indices[1:]]
                        + tan.density_left_derivative * d_ion[:-1]
                        + tan.density_right_derivative * d_ion[1:]
                        + tan.partner_left_derivative * other[:-1]
                        + tan.partner_right_derivative * other[1:]
                    )
                    j_ion[ion_dev.ion_node_indices[:-1] + 1] += sign * Q * j_int

        # 6. Displacement current across all N+1 faces (accounting for two-sided D_left):
        cell_charge = w * (Q * (dp - dn) + drho_ion + drho_stat + drho_bt)
        d_series_orig = -comp.poisson.face_capacitance_F_m2 * np.diff(phi_x)
        d_interior = np.array(d_series_orig, dtype=np.complex128, copy=True)

        if comp.interface_model is not None and comp.interface_model.topology == "two_sided_trace":
            dyn_idx_map = {
                it.interface_index: (pos, it)
                for pos, it in enumerate(dev.dynamic_interface_traps)
            }
            for tspec in comp.interface_model.two_sided:
                k = tspec.interface_index
                f = tspec.face
                ln, rn = tspec.left_node, tspec.right_node
                if k in dyn_idx_map:
                    pos, itrap = dyn_idx_map[k]
                    if itrap.has_sheet_charge:
                        dsig = dsigma_it[pos]
                        cell_charge[ln] += itrap.weight_left * dsig
                        cell_charge[rn] += itrap.weight_right * dsig
                        d_interior[f] -= itrap.weight_left * dsig
                elif tspec.charge is not None:
                    bulk = TwoSidedBulkState(
                        phi_left_V=float(u0[ln]),
                        phi_right_V=float(u0[rn]),
                        n_left_m3=float(n0[ln]),
                        p_left_m3=float(p0[ln]),
                        n_right_m3=float(n0[rn]),
                        p_right_m3=float(p0[rn]),
                        electron_electrochemical_step=float(u0[N + rn] - u0[N + ln]),
                        hole_electrochemical_step=float(-(u0[2 * N + rn] - u0[2 * N + ln])),
                    )
                    qres = solve_charged_two_sided_interface_qss(
                        tspec.trace_electrostatics, tspec.physics, bulk, tspec.charge
                    )
                    js_qf = qres.sheet_charge_jacobian_qf
                    js_com = qres.sheet_charge_common_qf
                    assert js_qf is not None and js_com is not None
                    com_l6 = np.array([ref_n / vt, -ref_p / vt], dtype=np.float64)
                    u_s6 = np.array(
                        [x[ln], x[rn], x[N + ln], -x[2 * N + ln], x[N + rn], -x[2 * N + rn]],
                        dtype=np.complex128,
                    )
                    qf_l6 = np.array(
                        [
                            lift[ln],
                            lift[rn],
                            lift[N + ln] if 0 < ln < N - 1 else (-lift[ln] / vt - com_l6[0]),
                            -lift[2 * N + ln] if 0 < ln < N - 1 else (lift[ln] / vt - com_l6[1]),
                            lift[N + rn] if 0 < rn < N - 1 else (-lift[rn] / vt - com_l6[0]),
                            -lift[2 * N + rn] if 0 < rn < N - 1 else (lift[rn] / vt - com_l6[1]),
                        ],
                        dtype=np.float64,
                    )
                    dsig = complex(
                        js_qf @ u_s6 + float(js_qf @ qf_l6 + js_com @ com_l6) * voltage
                    )
                    cell_charge[ln] += tspec.weight_left * dsig
                    cell_charge[rn] += tspec.weight_right * dsig
                    d_interior[f] -= tspec.weight_left * dsig

        d_left_contact = d_series_orig[0] - cell_charge[0]
        d_right_contact = d_series_orig[-1] + cell_charge[-1]
        d_all: ComplexVector = np.r_[d_left_contact, d_interior, d_right_contact]
        j_disp: ComplexVector = 1j * omega * d_all

        j_total: ComplexVector = j_elec + j_ion + j_disp
        pos_electrode = 0 if dev.voltage_polarity > 0.0 else -1
        j_in = dev.voltage_polarity * j_total[pos_electrode]
        y_val = complex(j_in / voltage)
        z_val = complex(1.0 / y_val) if y_val != 0.0 else complex(math.nan, math.nan)
        spread = float(np.max(np.abs((j_total - j_total[0]) / voltage)))

        return DynamicDefectACSweepPoint(
            frequency_Hz=sol.frequency_Hz,
            linear_solution=sol,
            admittance_density_S_m2=y_val,
            impedance_area_ohm_m2=z_val,
            electronic_conduction_current_A_m2=j_elec,
            ionic_conduction_current_A_m2=j_ion,
            displacement_current_A_m2=j_disp,
            total_current_A_m2=j_total,
            maxwell_spread_A_m2_V=spread,
            potential_perturbation_V=phi_x,
            electron_density_perturbation_m3=dn,
            hole_density_perturbation_m3=dp,
            bulk_trap_occupancy_perturbation_per_V=df_bt / voltage,
            bulk_trap_charge_perturbation_C_m3_per_V=drho_bt / voltage,
            interface_trap_occupancy_perturbation_per_V=df_it / voltage,
            interface_sheet_charge_perturbation_C_m2_per_V=dsigma_it / voltage,
            ion_density_perturbation_m3=ion_dens_map,
            ion_inventory_perturbation_m2=ion_inv_map,
            ion_inventory_relative_per_V=ion_inv_rel_map,
        )

"""Explicit frozen-step prototype retaining rebased inputs before local DD.

This is a numerical representation experiment, not the production pair backend.
``from_saved_step`` copies an already restored, rebased binary64 system/state;
it performs no evaluation, initialization, DC solve or nonlinear correction.
The original Newton step and acceptance policy remain the caller's responsibility.

Primary fields are lifted directly from the saved binary64 reference and the
binary64 Newton coordinates. Local carrier balances and their analytic chain,
SG carrier/ion fluxes, storage, charge and displacement consume these same words.
The existing bulk sparse Jacobian remains a binary64 approximation. Bulk source
assembly retains its existing rounding and receives an explicit low-input SRH/
radiative/Auger correction. The baseline incremental Gauss reference and its
harmonic-lift convention remain unchanged. No full-pair preparation is claimed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy import sparse

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from .interface_defect_ion_transient import _InterfaceIonDeviceState
from .interface_defect_transient import InterfaceDefectTransientError
from .one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from .one_dimensional_mechanism_r1_local_carrier import (
    assemble_r1_carrier_source, carrier_data, evaluate_interface_from_fine,
    fine_local_carrier_inputs, without_local_exchange,
)
# Only pure arithmetic utilities are reused. No CompensatedR1System method,
# precision initialization, context, solve wrapper or corrector is invoked.
from .one_dimensional_mechanism_r1_precision import (
    carrier_currents_pair, cat, diff, ion_flux_pair, number, put,
)

REPRESENTATION = "r1-rebased-input-lift-dd-v1"
PRIMARY_FIELDS = ("n_m3", "p_m3", "phi_V", "dqfn_V", "dqfp_V", "occupancy",
                  "trace_potential_V", "trace_state_m3", "positive_m3")


def _words(values):
    return {name: {"hi": value.hi.tolist(), "lo": value.lo.tolist()}
            for name, value in values.items()}


def _identity(values):
    text = json.dumps(_words(values), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(slots=True)
class InputLiftState(_InterfaceIonDeviceState):
    """Ordinary views plus immutable, explicitly named represented inputs."""

    input_lift: object = field(default_factory=lambda: MappingProxyType({}))
    coordinate_reference_identity: str = ""
    operator_representation: str = REPRESENTATION


def primary_inputs(state):
    if not isinstance(state, InputLiftState) or state.operator_representation != REPRESENTATION:
        raise TypeError("input-lift operations require an explicit InputLiftState")
    value = state.input_lift
    if any(not isinstance(value.get(name), DD) for name in PRIMARY_FIELDS):
        raise ValueError("input-lift state is missing primary DD fields")
    return value


def _storage(system, value):
    return cat(value["n_m3"][1:-1], value["p_m3"][1:-1],
               DD(system.trap_density)*value["occupancy"],
               value["positive_m3"][system.positive_nodes])


def _recombination(system, n, p):
    """Same supported bulk laws, retaining low input perturbations in DD."""
    mat = system.system.source_mat
    params = mat.carrier_params
    if params.get("degenerate_recombination_model") == "off":
        return DD(np.zeros(n.shape))
    shape = n.shape
    array = lambda value: np.broadcast_to(value, shape)
    excess = n*p-DD(array(mat.ni_sq))

    def srh(mask, tau_n, tau_p, n1, p1):
        tn, tp = array(tau_n), array(tau_p)
        # Infinite lifetime means an inactive legacy channel, exactly as in
        # the existing SRH law; DD deliberately rejects infinity as a number.
        active = np.asarray(mask, dtype=bool) & np.isfinite(tn) & np.isfinite(tp)
        result = DD(np.zeros(shape))
        if np.any(active):
            denominator = DD(tp[active])*(n[active]+DD(array(n1)[active]))
            denominator = denominator+DD(tn[active])*(p[active]+DD(array(p1)[active]))
            result = put(result, active, excess[active]/denominator)
        return result

    model = getattr(mat, "neutral_bulk_defects", None)
    legacy = np.ones(shape, dtype=bool) if model is None else ~model.explicit_node_mask
    result = srh(legacy, mat.tau_n, mat.tau_p, mat.n1, mat.p1)
    if model is not None:
        for species in model.species:
            if species.cycle_active:
                mask = np.zeros(shape, dtype=bool)
                mask[species.active_nodes] = True
                result = result+srh(mask, species.tau_n_s, species.tau_p_s,
                                   species.n1_m3, species.p1_m3)
    return result+(DD(mat.B_rad)+DD(mat.C_n)*n+DD(mat.C_p)*p)*excess


class RebasedInputLiftR1System(ControlledPhysicalInterfaceIonSystem):
    """Opt-in one-step operator; construction and trajectory rebasing are blocked."""

    operator_representation = REPRESENTATION

    def __init__(self, *args, **kwargs):
        raise TypeError("use from_saved_step with an already restored baseline step")

    def rebase(self, previous):
        raise TypeError("input-lift prototype is scoped to one explicitly restored step")

    def evaluate(self, coordinate, voltage):
        if self._input_lift_work is not None:
            raise RuntimeError("input-lift evaluation cannot be nested on the same system")
        try:
            return super().evaluate(coordinate, voltage)
        finally:
            # A later evaluation must never supply an earlier state's input.
            self._input_lift_work = None

    def _coordinates(self, coordinate, voltage):
        result = list(super()._coordinates(coordinate, voltage))
        z, ref, vt = DD(np.asarray(coordinate, dtype=float)), self._input_lift_reference, DD(self.thermal_voltage)
        zn, zp, zu = z[self.electron_slice], z[self.hole_slice], z[self.potential_slice]
        values = dict(ref)
        for name, slot, increment in (("dqfn_V", 0, zn), ("dqfp_V", 1, zp), ("phi_V", 2, zu)):
            value = put(ref[name], [0, -1], result[slot][[0, -1]])
            values[name] = put(value, slice(1, -1), ref[name][1:-1]+vt*increment)
        values["n_m3"] = put(ref["n_m3"], slice(1, -1), ref["n_m3"][1:-1]*(zn+zu).exp())
        values["p_m3"] = put(ref["p_m3"], slice(1, -1), ref["p_m3"][1:-1]*(zp-zu).exp())
        odds, f = z[self.trap_slice].expm1(), ref["occupancy"]
        values["occupancy"] = f+f*(1-f)*odds/(1+f*odds)
        trace_phi, trace_n = ref["trace_potential_V"], ref["trace_state_m3"]
        for index in range(self.interface_count):
            block = z[self._local_block_slice(index)]
            trace_phi = put(trace_phi, index, ref["trace_potential_V"][index]+vt*block[:2])
            trace_n = put(trace_n, index, ref["trace_state_m3"][index]*block[2:].exp())
        if hasattr(self, "_lift"):
            lift = DD(self._lift[1:-1])
            values["phi_V"] = put(values["phi_V"], slice(1, -1), values["phi_V"][1:-1]+lift)
            values["dqfn_V"] = put(values["dqfn_V"], slice(1, -1), values["dqfn_V"][1:-1]-lift)
            values["dqfp_V"] = put(values["dqfp_V"], slice(1, -1), values["dqfp_V"][1:-1]+lift)
            trace_phi = trace_phi+DD(self._trace_lift)
        values["trace_potential_V"], values["trace_state_m3"] = trace_phi, trace_n
        self._input_lift_work = values
        for name, slot in (("dqfn_V", 0), ("dqfp_V", 1), ("phi_V", 2),
                           ("n_m3", 3), ("p_m3", 4), ("occupancy", 5), ("trace_potential_V", 6)):
            result[slot] = values[name].hi.copy()
        result[7] = trace_n.log().hi.copy()
        return tuple(result)

    def _ion_coordinates(self, coordinate):
        ref = self._input_lift_reference["positive_m3"]
        value = put(ref, self.positive_nodes, ref[self.positive_nodes]*DD(coordinate[self.positive_slice]).exp())
        self._site_fraction(value.hi, None, reject=True)
        self._input_lift_work["positive_m3"] = value
        return value.hi.copy(), None

    def _trace_density_coordinates(self, coordinate):
        return self._input_lift_work["trace_state_m3"].hi.copy()

    def _local_carrier_inputs(self, index, n, p, phi, occupancy, trace_potential,
                              trace_log_state, *, trace_density_m3=None):
        return fine_local_carrier_inputs(self, index, n, p, phi, occupancy,
            trace_potential, trace_log_state, trace_density_m3=trace_density_m3,
            fine=self._input_lift_work)

    def _refresh_fine_constants(self):
        # The name is the pure carrier_currents_pair utility's coefficient
        # protocol, not a declaration that this is the full pair backend.
        mat = self.material
        constants = {"vt": DD(self.thermal_voltage), "chi": DD(mat.chi), "Eg": DD(mat.Eg),
                     "edge_n": DD(self.system.reference_edge_drop_n),
                     "edge_p": DD(self.system.reference_edge_drop_p)}
        self._fine_carrier_prefactors = (DD(Q)*DD(mat.D_n_face)/DD(np.diff(self.grid)),
                                        DD(Q)*DD(mat.D_p_face)/DD(np.diff(self.grid)))
        return constants

    def _source(self, n, p, phi, voltage, interface_qss):
        # Retain the original assembled source at its rounded high inputs.
        from .interface_defect_transient import _InterfaceTransientSystem
        source = _InterfaceTransientSystem._source(self, n, p, phi, voltage,
                                                    without_local_exchange(interface_qss))
        value = self._input_lift_work
        correction = _recombination(self, DD(n), DD(p))-_recombination(self, value["n_m3"], value["p_m3"])
        correction = put(correction, [0, -1], 0.)
        return assemble_r1_carrier_source(self, DD(source)+cat(correction, correction), interface_qss)

    def _currents(self, dqfn, dqfp, phi, n, p, local):
        values = carrier_currents_pair(self, self._input_lift_work, local)
        self._input_lift_work["electron_current_A_m2"] = values[2]
        self._input_lift_work["hole_current_A_m2"] = values[3]
        return values[0], values[1], values[2].hi.copy(), values[3].hi.copy()

    def _carrier_rate_fields(self, source, transport_n, transport_p, local):
        def divergence(flux):
            return cat(flux[0], diff(flux), -flux[-1])
        n = source[:self.node_count]+divergence(transport_n)/(DD(Q)*DD(self.widths))
        p = source[self.node_count:]-divergence(transport_p)/(DD(Q)*DD(self.widths))
        captures = [carrier_data(item).balance["capture_flux_m2_s"] for item in local]
        trap = cat(*(c[0]+c[2]-c[1]-c[3] for c in captures))
        self._input_lift_work["carrier_rate"] = cat(n[1:-1], p[1:-1], trap)
        return n.hi.copy(), p.hi.copy(), trap.hi.copy()

    def _ion_fields(self, phi, positive, negative):
        value = self._input_lift_work
        flux = (ion_flux_pair(value["phi_V"], value["positive_m3"], self.material, np.diff(self.grid))
                if self.controls.nu_I else DD(np.zeros(self.node_count-1)))
        rate = -diff(cat(0., flux, 0.))/DD(self.widths)
        value["positive_flux_m2_s"], value["positive_rate_m3_s"] = flux, rate
        return rate.hi.copy(), None, flux.hi.copy(), None

    def _with_step_electrostatics(self, state):
        value, previous = dict(self._input_lift_work), self._step_reference
        value["storage"] = _storage(self, value)
        value["rate"] = cat(value["carrier_rate"], value["positive_rate_m3_s"][self.positive_nodes])
        value["sheet_charge_C_m2"] = -DD(Q)*DD(self.trap_density)*(value["occupancy"]-DD(self.equilibrium_occupancy))
        delta = value["storage"]-primary_inputs(previous)["storage"]
        count = self.interior_count
        rho = put(DD(np.zeros(self.node_count)), slice(1, -1), DD(Q)*(delta[count:2*count]-delta[:count]))
        ion_start = 2*count+self.interface_count
        rho = put(rho, self.positive_nodes, rho[self.positive_nodes]+DD(Q)*delta[ion_start:])
        occupied = delta[2*count:ion_start]
        # Same baseline increment equations and exact harmonic-lift convention.
        dphi = value["phi_V"]-primary_inputs(previous)["phi_V"]
        if hasattr(self, "_lift"):
            dphi = dphi-DD(self._lift)
        poisson = DD(previous.poisson_residual)+diff(DD(self.material.poisson_factor.C)*diff(dphi))
        poisson = poisson+rho[1:-1]*DD(self.material.poisson_factor.h_cell)
        local_rows = []
        for index, (left, right, item) in enumerate(zip(self.left_nodes, self.right_nodes, state.local)):
            wl, wr = self._sheet_weights(index)
            poisson = put(poisson, left-1, poisson[left-1]-DD(wl)*DD(Q)*occupied[index])
            poisson = put(poisson, right-1, poisson[right-1]-DD(wr)*DD(Q)*occupied[index])
            trace = value["trace_potential_V"][index]-primary_inputs(previous)["trace_potential_V"][index]
            if hasattr(self, "_trace_lift"):
                trace = trace-DD(self._trace_lift[index])
            cl, cr = self._trace_capacitances(index)
            before = DD(previous.local_residual[6*index:6*index+2])
            electrostatic = before+cat(trace[1]-trace[0], DD(cl)*(trace[0]-dphi[left])
                                      +DD(cr)*(trace[1]-dphi[right])+DD(Q)*occupied[index])
            local_rows.append(cat(electrostatic, carrier_data(item).balance["residual_m2_s"]))
        value["poisson_residual_C_m2"], value["local_residual"] = poisson, cat(*local_rows)
        result = InputLiftState(**{f.name: getattr(state, f.name) for f in fields(_InterfaceIonDeviceState)},
                                input_lift=MappingProxyType(value),
                                coordinate_reference_identity=self._input_lift_reference_identity)
        result.storage, result.rate = value["storage"].hi.copy(), value["rate"].hi.copy()
        result.sheet_charge = value["sheet_charge_C_m2"].hi.copy()
        result.direct_poisson_residual = state.poisson_residual.copy()
        result.poisson_residual, result.local_residual = poisson.hi.copy(), value["local_residual"].hi.copy()
        result.local = tuple(replace(item, electrostatic_residual=result.local_residual[6*k:6*k+2].copy(),
                                     sheet_charge_C_m2=float(result.sheet_charge[k])) for k, item in enumerate(state.local))
        return result

    def residual_and_jacobian(self, coordinate, voltage, previous_state, dt,
                              storage_scale, poisson_scale, local_scale):
        state = self.evaluate(coordinate, voltage)
        current, previous = primary_inputs(state), primary_inputs(previous_state)
        storage_residual = current["storage"]-previous["storage"]-DD(dt)*current["rate"]
        residual = cat(storage_residual/DD(storage_scale),
                       current["poisson_residual_C_m2"]/DD(poisson_scale),
                       current["local_residual"]/DD(local_scale)).hi.copy()
        jacobian = sparse.vstack((sparse.diags(1/storage_scale)@(state.storage_jacobian-dt*state.rate_jacobian),
                                  sparse.diags(1/poisson_scale)@state.poisson_jacobian,
                                  sparse.diags(1/local_scale)@state.local_jacobian), format="csr")
        return residual, jacobian, state

    def storage_increment(self, state, previous):
        return (primary_inputs(state)["storage"]-primary_inputs(previous)["storage"]).hi.copy()

    def potential_increment(self, state, previous):
        return (primary_inputs(state)["phi_V"]-primary_inputs(previous)["phi_V"]).hi.copy()

    def _trace_capacitances(self, index):
        left, right = self.left_nodes[index], self.right_nodes[index]
        mat = self.material
        return (EPS_0*mat.eps_r[left]/mat.iface_qss_left_distances_m[index],
                EPS_0*mat.eps_r[right]/mat.iface_qss_right_distances_m[index])

    def interface_current_sides(self, state, previous=None, dt=None):
        value = primary_inputs(state)
        conduction, displacement = [], []
        for index, (left, right, face, item) in enumerate(zip(self.left_nodes, self.right_nodes, self.interface_faces, state.local)):
            flux = carrier_data(item).balance["bulk_flux_m2_s"]
            ionic = number(DD(Q)*value["positive_flux_m2_s"][face])
            conduction.append(self.polarity*(cat(DD(Q)*(-flux[0]+flux[1]), DD(Q)*(flux[2]-flux[3])).hi+ionic))
            changes = np.zeros(2)
            if previous is not None and dt is not None:
                before = primary_inputs(previous)
                phi, trace = value["phi_V"]-before["phi_V"], value["trace_potential_V"]-before["trace_potential_V"]
                cl, cr = self._trace_capacitances(index)
                changes = self.polarity*cat(-DD(cl)*(trace[index, 0]-phi[left])/DD(dt),
                                            DD(cr)*(trace[index, 1]-phi[right])/DD(dt)).hi
            displacement.append(changes)
        conduction, displacement = np.asarray(conduction), np.asarray(displacement)
        return conduction, displacement, conduction+displacement

    def transient_current_metrics(self, state, previous, dt):
        change = primary_inputs(state)["phi_V"]-primary_inputs(previous)["phi_V"]
        displacement = self.polarity*(DD(self.eps_face)*(-diff(change)/DD(np.diff(self.grid)))/DD(dt)).hi
        conduction, interface_displacement, interface = self.interface_current_sides(state, previous, dt)
        displacement = displacement.copy()
        for index, face in enumerate(self.interface_faces):
            displacement[face] = interface_displacement[index, 0]
        total = state.conduction+displacement
        spread = float(np.ptp(total))/max(float(np.max(np.abs(total))), 1e-20)
        interface_error = float(np.max(np.abs(interface[:, 0]-interface[:, 1])/np.maximum(np.max(np.abs(interface), axis=1), 1e-20)))
        return displacement, total, conduction, interface_displacement, spread, interface_error

    def solver_current_metrics(self, state, previous, dt):
        metrics = self.transient_current_metrics(state, previous, dt)
        current, before = primary_inputs(state), primary_inputs(previous)
        rho = DD(Q)*((current["p_m3"]-before["p_m3"])-(current["n_m3"]-before["n_m3"])
                     +current["positive_m3"]-before["positive_m3"])
        change = -DD(self.material.poisson_factor.C)*diff(current["phi_V"]-before["phi_V"])
        contact = cat(change[0]-rho[0]*DD(self.widths[0]), change[-1]+rho[-1]*DD(self.widths[-1]))/DD(dt)
        total = self.polarity*(state.current_n[[0, -1]]+state.current_p[[0, -1]]+contact.hi)
        combined = np.r_[metrics[1], total]
        spread = float(np.ptp(combined))/max(float(np.max(np.abs(combined))), 1e-20)
        return (*metrics[:4], spread, metrics[5])

    def integrated_charge(self, state):
        value = primary_inputs(state)
        rho = DD(Q)*(value["p_m3"]-value["n_m3"]+value["positive_m3"]-DD(self.material.P_ion0))
        return number((rho[1:-1]*DD(self.widths[1:-1])).sum()+value["sheet_charge_C_m2"].sum())

    def integrated_charge_increment(self, state, previous):
        current, before = primary_inputs(state), primary_inputs(previous)
        rho = DD(Q)*((current["p_m3"]-before["p_m3"])-(current["n_m3"]-before["n_m3"])
                     +(current["positive_m3"]-before["positive_m3"]))
        occupied = DD(self.trap_density)*(current["occupancy"]-before["occupancy"])
        return number((rho[1:-1]*DD(self.widths[1:-1])).sum()-DD(Q)*occupied.sum())

    def failure_evidence(self, state, previous, voltage, dt, residual, storage_scale,
                         poisson_scale, local_scale, diagnostics):
        return {"schema": "R1InputLiftFailureV1", "operator_representation": REPRESENTATION,
                "terminal_state_available": True, "state": snapshot(self, state),
                "previous": snapshot(self, previous), "voltage_V": float(voltage), "dt_s": float(dt),
                "scaled_residual_vector": residual.tolist(), "diagnostics": diagnostics}


def from_saved_step(system, previous):
    """Adapt an already rebased baseline step, without any physical evaluation."""
    if type(system) is not ControlledPhysicalInterfaceIonSystem:
        raise TypeError("input lift accepts only the explicit restored baseline system")
    if system._step_reference is not previous or np.any(previous.coordinate != 0):
        raise ValueError("input lift requires the exact zero-coordinate restored step reference")
    if getattr(previous, "fine", None) is not None or hasattr(previous, "input_lift"):
        raise TypeError("a fine or already lifted state cannot be relabeled as a binary64 seed")
    mat = system.material
    if (mat.has_dual_ions or not mat.ion_steric_diffusion_only or system.negative_nodes.size
            or previous.negative is not None):
        raise ValueError("input lift supports the frozen positive-ion steric model only")
    source = system.system.source_mat
    params = source.carrier_params
    if (params.get("carrier_statistics", "maxwell_boltzmann") != "maxwell_boltzmann"
            or params.get("degenerate_recombination_model", "maxwell_boltzmann")
            not in ("maxwell_boltzmann", "off")):
        raise ValueError("input lift supports Maxwell-Boltzmann bulk transport and its declared recombination laws only")
    if any(getattr(source, key, None) is not None for key in
           ("monovalent_bulk_defects", "multivalent_bulk_defects", "frozen_metastable_defects")):
        raise ValueError("input lift does not support additional charged bulk-defect source laws")
    if (getattr(source, "has_selective_contacts", False) or params.get("het_recomb_despike", 0.) != 0.
            or getattr(source, "has_radiative_reabsorption", False)):
        raise ValueError("input lift requires pinned contacts, unmodified bulk recombination densities and no radiative reabsorption")
    primary = {name: DD(getattr(previous, attribute)) for name, attribute in
               (("n_m3", "n"), ("p_m3", "p"), ("phi_V", "phi"), ("dqfn_V", "dqfn"),
                ("dqfp_V", "dqfp"), ("occupancy", "occupancy"), ("positive_m3", "positive"))}
    primary["trace_potential_V"] = DD(np.asarray([item.trace_potential for item in previous.local]))
    primary["trace_state_m3"] = DD(np.asarray([item.state_m3 for item in previous.local]))
    identity = _identity(primary)
    value = dict(primary)
    value["storage"] = _storage(system, value)
    value["sheet_charge_C_m2"] = -DD(Q)*DD(system.trap_density)*(value["occupancy"]-DD(system.equilibrium_occupancy))
    value["poisson_residual_C_m2"] = DD(previous.poisson_residual)
    value["local_residual"] = DD(previous.local_residual)
    for name, attribute in (("electron_current_A_m2", "current_n"), ("hole_current_A_m2", "current_p"),
                            ("positive_flux_m2_s", "positive_flux"), ("positive_rate_m3_s", "positive_rate"), ("rate", "rate")):
        value[name] = DD(getattr(previous, attribute))
    saved = InputLiftState(**{f.name: copy.deepcopy(getattr(previous, f.name)) for f in fields(_InterfaceIonDeviceState)},
                           input_lift=MappingProxyType(value), coordinate_reference_identity=identity)
    working = copy.copy(system)
    working.__class__ = RebasedInputLiftR1System
    working._input_lift_reference = MappingProxyType(primary)
    working._input_lift_reference_identity = identity
    working._input_lift_work = None
    working._step_reference = saved
    working.input_lift_contract = MappingProxyType({"operator_representation": REPRESENTATION,
        "seed_representation": "float64-baseline", "scope": "one_restored_step",
        "reference_identity": identity, "initialization_evaluations": 0,
        "bulk_jacobian": "existing_binary64_sparse_approximation", "newton_solver": "original_solve_step",
        "saved_primary_words": "exact_binary64_values_with_zero_low_words",
        "previous_derived_fields": "storage_and_sheet_recomputed_from_primary_words; saved_public_arrays_unchanged",
        "storage_high_matches_saved": bool(np.array_equal(value["storage"].hi, previous.storage)),
        "sheet_charge_high_matches_saved": bool(np.array_equal(value["sheet_charge_C_m2"].hi, previous.sheet_charge))})
    return working, saved


def snapshot(system, state):
    """Explicit prototype record; never emits the full-pair or baseline label."""
    from .one_dimensional_mechanism_r1_state import _snapshot_legacy
    return {"schema": "R1InputLiftStateV1", "operator_representation": REPRESENTATION,
            "coordinate_reference_identity": state.coordinate_reference_identity,
            "high_word_state": _snapshot_legacy(system, state),
            "represented_fields": _words(primary_inputs(state))}


def independent_currents(system, state):
    """Recompute from primary words, ignoring saved local/flux/current payloads."""
    value = primary_inputs(state)
    fluxes = tuple(evaluate_interface_from_fine(system, value, k).balance["bulk_flux_m2_s"]
                   for k in range(system.interface_count))
    _, _, electron, hole = carrier_currents_pair(system, value, (), interface_fluxes=fluxes)
    ion = (ion_flux_pair(value["phi_V"], value["positive_m3"], system.material, np.diff(system.grid))
           if system.controls.nu_I else DD(np.zeros(system.node_count-1)))
    interface = np.asarray([cat(DD(Q)*(-flux[0]+flux[1]), DD(Q)*(flux[2]-flux[3])).hi
                            +number(DD(Q)*ion[face]) for face, flux in zip(system.interface_faces, fluxes)])
    return electron.hi.copy(), hole.hi.copy(), ion.hi.copy(), interface


def independent_physics_row(system, state, previous, dt, *, reported=None):
    """Original current/charge limits, freshly assembled from lifted inputs."""
    from .one_dimensional_mechanism_r1_precision_physics import (
        _independent_physics_row_dd, SHARED_CONSTITUTIVE_DEPENDENCIES,
    )
    result = _independent_physics_row_dd(system, state, previous, dt, reported=reported,
        fields_of=primary_inputs, currents=independent_currents,
        dependencies=SHARED_CONSTITUTIVE_DEPENDENCIES)
    result["operator_representation"] = REPRESENTATION
    return result


__all__ = ["REPRESENTATION", "PRIMARY_FIELDS", "InputLiftState", "RebasedInputLiftR1System",
           "from_saved_step", "primary_inputs", "snapshot", "independent_currents", "independent_physics_row"]

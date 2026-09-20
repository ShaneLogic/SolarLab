"""Opt-in R1 V7 compensated-state trajectory prototype.

No default solver, physical equation, acceptance limit or legacy reader is
changed by importing this module.  The explicit context installs the bounded
research implementation, including its state reconstruction consumers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
import copy
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonDeviceState
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem

REPRESENTATION = "float64-pair-v1"
_ACTIVE = False


def cat(*items):
    items = [x if isinstance(x, DD) else DD(x) for x in items]
    return DD(np.concatenate([np.atleast_1d(x.hi) for x in items]),
              np.concatenate([np.atleast_1d(x.lo) for x in items]))


def put(value, index, part):
    part = part if isinstance(part, DD) else DD(part)
    hi, lo = value.hi.copy(), value.lo.copy()
    hi[index], lo[index] = part.hi, part.lo
    return DD(hi, lo)


def diff(value):
    return value[1:] - value[:-1]


def number(value):
    return float(np.asarray(value.to_float()))


def bernoulli_pair(x):
    """Exponential fitting without discarding the low part of its argument."""
    result = DD(np.ones(x.shape))
    nonzero = (x.hi != 0) | (x.lo != 0)
    if np.any(nonzero):
        result = put(result, nonzero, x[nonzero] / x[nonzero].expm1())
    return result


def bernoulli_derivative_pair(x):
    result=DD(np.full(x.shape,-.5))
    small=np.abs(x.hi)<1e-3
    if np.any(small):
        a=x[small]
        a2=a*a
        result=put(result,small,DD(-.5)+a*(DD(1)/6+a2*(-DD(1)/180+a2*(DD(1)/5040+a2*(-DD(1)/151200+a2*DD(1)/4790016)))))
    if np.any(~small):
        a=x[~small]; denominator=a.expm1()
        result=put(result,~small,(denominator-a*a.exp())/(denominator*denominator))
    return result


def ion_flux_pair(phi, population, material, spacing, *, fault="none"):
    """Same frozen one-positive-species SG law, evaluated with two words."""
    if material.has_dual_ions or not material.ion_steric_diffusion_only:
        raise ValueError("V7 prototype supports the frozen positive-ion steric model")
    theta = population / DD(material.P_lim_node)
    if np.any(theta.hi < 0) or np.any(theta.hi >= .999):
        raise ValueError("V7 prototype reached its original pre-clipping bound")
    chemical = -(DD(1) - theta).log()
    vt = DD(material.V_T_device) * (1.01 if fault == "thermal_voltage" else 1)
    sign = -1 if fault == "drift_sign" else 1
    xi = sign * diff(phi) / vt + diff(chemical)
    diffusion = DD(material.D_ion_face) * (1.01 if fault == "diffusion" else 1)
    # Both terms retain ~106 bits before cancellation.  A fault is applied
    # inside this constitutive calculation, never to a comparator's output.
    flux = diffusion / DD(spacing) * (
        bernoulli_pair(xi) * population[:-1]
        - bernoulli_pair(-xi) * population[1:])
    if fault == "omit_ion":
        flux = DD(np.zeros(flux.shape))
    if fault == "single_face_sign":
        active = np.flatnonzero(material.D_ion_face > 0)
        if active.size:
            flux = put(flux, int(active[0]), -flux[int(active[0])])
    if fault not in {"none", "thermal_voltage", "drift_sign", "diffusion", "omit_ion", "single_face_sign"}:
        raise ValueError("unknown implementation fault")
    return flux


def carrier_currents_pair(system, fine, local):
    """Fine bulk SG law with the original separately solved interface law."""
    mat, vt = system.material, DD(system.thermal_voltage)
    phi,n,p=fine["phi_V"],fine["n_m3"],fine["p_m3"]
    xi_n=diff(phi+DD(mat.chi))/vt
    xi_p=diff(phi+DD(mat.chi)+DD(mat.Eg))/vt
    dn=DD(system.system.reference_edge_drop_n)+diff(fine["dqfn_V"])/vt
    dp=-(DD(system.system.reference_edge_drop_p)+diff(fine["dqfp_V"])/vt)
    def stable(forward,backward,drive):
        positive=drive.hi>=0
        value=backward*drive.expm1()
        return put(value,positive,forward[positive]*(-(-drive[positive]).expm1()))
    jn=DD(Q)*DD(mat.D_n_face)/DD(np.diff(system.grid))*stable(
        bernoulli_pair(xi_n)*n[1:],bernoulli_pair(-xi_n)*n[:-1],dn)
    jp=DD(Q)*DD(mat.D_p_face)/DD(np.diff(system.grid))*stable(
        bernoulli_pair(xi_p)*p[:-1],bernoulli_pair(-xi_p)*p[1:],dp)
    transport_n,transport_p=jn.copy(),jp.copy()
    for k,face in enumerate(system.interface_faces):
        balance=local[k].tangent.balance
        transport_n,transport_p=put(transport_n,face,0),put(transport_p,face,0)
        jn=put(jn,face,-Q*balance.bulk_flux_m2_s[0])
        jp=put(jp,face,Q*balance.bulk_flux_m2_s[1])
    return transport_n,transport_p,jn,jp


@dataclass(slots=True)
class PrecisionState(_InterfaceIonDeviceState):
    fine: dict = field(default_factory=dict)


class CompensatedR1System(ControlledPhysicalInterfaceIonSystem):
    """Persistent state, independently reconstructed Poisson and fine SG."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use from_baseline with a validated common-state seed")

    @classmethod
    def from_baseline(cls,baseline,initial):
        from .one_dimensional_mechanism_r1_precision_initial import initialize_fine_reference
        from perovskite_sim.physics.two_sided_interface import _material_two_sided_interface_problem
        system=copy.copy(baseline)
        system.__class__=cls
        system._fine_work={}
        system.precision_fault="none"
        system.precision_constraint_corrections=0
        fine=initialize_fine_reference(baseline)
        geometry=[_material_two_sided_interface_problem(
            baseline.material,baseline.stack,initial.n,initial.p,initial.phi,k,
            cross_transmission=baseline.dark_reference.interface_transmission)[0]
            for k in range(baseline.interface_count)]
        if any(g.fixed_sheet_charge_C_m2 != 0. for g in geometry):
            raise ValueError("V7 prototype requires the frozen zero-static-sheet R1 model")
        system._prescribed_trace_jump=DD([g.potential_jump_right_minus_left_V for g in geometry])
        system.precision_initialization_diagnostics=copy.deepcopy(baseline.precision_initialization_diagnostics)
        system._fine_anchor={k:v.copy() for k,v in fine.items()}
        system._fine_reference=system._fine_anchor
        system._step_reference=None
        system._maximum_eliminated_operator_components={}
        state=system.evaluate(np.zeros(system.dimension),0.)
        return system,state

    def rebase(self, previous):
        working, local = super().rebase(previous)
        working._fine_reference = {k: v.copy() for k, v in previous.fine.items()}
        working._fine_work = {}
        return working, local

    def set_voltage_lift(self, voltage, previous):
        super().set_voltage_lift(voltage, previous)
        # The recorded original harmonic lift is an exact input to the pair
        # arithmetic; its finite Gauss defect is solved, never erased.
        self._fine_lift = DD(self._lift)
        self._fine_trace_lift = DD(self._trace_lift)

    def _coordinates(self, coordinate, voltage):
        result = list(super()._coordinates(coordinate, voltage))
        if not hasattr(self, "_fine_reference"):
            return tuple(result)
        z = DD(np.asarray(coordinate, dtype=float))
        ref = self._fine_reference
        phi = put(ref["phi_V"], [0, -1], result[2][[0, -1]])
        phi = put(phi, slice(1, -1), ref["phi_V"][1:-1] + DD(self.thermal_voltage) * z[self.potential_slice])
        qn = put(ref["dqfn_V"], [0, -1], result[0][[0, -1]])
        qp = put(ref["dqfp_V"], [0, -1], result[1][[0, -1]])
        qn = put(qn, slice(1, -1), ref["dqfn_V"][1:-1] + DD(self.thermal_voltage) * z[self.electron_slice])
        qp = put(qp, slice(1, -1), ref["dqfp_V"][1:-1] + DD(self.thermal_voltage) * z[self.hole_slice])
        n = put(ref["n_m3"], slice(1, -1), ref["n_m3"][1:-1] * (z[self.electron_slice] + z[self.potential_slice]).exp())
        p = put(ref["p_m3"], slice(1, -1), ref["p_m3"][1:-1] * (z[self.hole_slice] - z[self.potential_slice]).exp())
        odds = z[self.trap_slice].expm1()
        f = ref["occupancy"]
        occupancy = f + f * (1-f) * odds / (1 + f * odds)
        trace_phi, trace_n = ref["trace_potential_V"].copy(), ref["trace_state_m3"].copy()
        for k in range(self.interface_count):
            block = z[self._local_block_slice(k)]
            trace_phi = put(trace_phi, k, ref["trace_potential_V"][k] + DD(self.thermal_voltage) * block[:2])
            trace_n = put(trace_n, k, ref["trace_state_m3"][k] * block[2:].exp())
        if hasattr(self, "_fine_lift"):
            phi = put(phi, slice(1, -1), phi[1:-1] + self._fine_lift[1:-1])
            qn = put(qn, slice(1, -1), qn[1:-1] - self._fine_lift[1:-1])
            qp = put(qp, slice(1, -1), qp[1:-1] + self._fine_lift[1:-1])
            trace_phi = trace_phi + self._fine_trace_lift
        self._fine_work = {"phi_V": phi, "dqfn_V": qn, "dqfp_V": qp,
            "n_m3": n, "p_m3": p, "occupancy": occupancy,
            "trace_potential_V": trace_phi, "trace_state_m3": trace_n}
        for i, value in ((0, qn), (1, qp), (2, phi), (3, n), (4, p), (5, occupancy), (6, trace_phi)):
            result[i] = value.hi.copy()
        result[7] = trace_n.log().hi.copy()
        return tuple(result)

    def _trace_density_coordinates(self, coordinate):
        if self._fine_work:
            return self._fine_work["trace_state_m3"].hi.copy()
        return super()._trace_density_coordinates(coordinate)

    def _ion_coordinates(self, coordinate):
        if not hasattr(self, "_fine_reference"):
            return super()._ion_coordinates(coordinate)
        positive = self._fine_reference["positive_m3"]
        positive = put(positive, self.positive_nodes, positive[self.positive_nodes] * DD(coordinate[self.positive_slice]).exp())
        self._fine_work["positive_m3"] = positive
        self._site_fraction(positive.hi, None, reject=True)
        return positive.hi.copy(), None

    def _ion_fields(self, phi, positive, negative):
        if not self._fine_work or not hasattr(self, "precision_fault"):
            return super()._ion_fields(phi, positive, negative)
        # evaluate() supplies exactly the just-constructed arrays.  The
        # independent elimination below calls ion_flux_pair explicitly.
        fine = self._fine_work
        if not np.array_equal(phi, fine["phi_V"].hi) or not np.array_equal(positive, fine["positive_m3"].hi):
            return super()._ion_fields(phi, positive, negative)
        flux = (ion_flux_pair(fine["phi_V"], fine["positive_m3"], self.material,
                             np.diff(self.grid), fault=self.precision_fault)
                if self.controls.nu_I else DD(np.zeros(self.node_count-1)))
        boundary = DD(np.zeros(2))
        rate = -diff(cat(boundary[0], flux, boundary[1])) / DD(self.widths)
        fine["positive_flux_m2_s"], fine["positive_rate_m3_s"] = flux, rate
        fine["boundary_flux_m2_s"] = boundary
        return rate.hi.copy(), None, flux.hi.copy(), None

    def _currents(self,dqfn,dqfp,phi,n,p,local):
        if not self._fine_work or not hasattr(self,"_fine_reference"):
            return super()._currents(dqfn,dqfp,phi,n,p,local)
        values=carrier_currents_pair(self,self._fine_work,local)
        self._fine_work["electron_current_A_m2"]=values[2]
        self._fine_work["hole_current_A_m2"]=values[3]
        return tuple(value.hi.copy() for value in values)

    def _poisson_pair(self, phi, n, p, positive, sigma):
        rho = DD(Q) * (p - n + DD(self.material.N_D) - DD(self.material.N_A)
                       + positive - DD(self.material.P_ion0))
        residual = diff(DD(self.material.poisson_factor.C) * diff(phi))
        residual = residual + rho[1:-1] * DD(self.material.poisson_factor.h_cell)
        for k, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            wl, wr = self._sheet_weights(k)
            residual = put(residual, left-1, residual[left-1] + DD(wl) * sigma[k])
            residual = put(residual, right-1, residual[right-1] + DD(wr) * sigma[k])
        return residual

    def _ion_jacobians(self, phi, positive, negative):
        if not self._fine_work or not hasattr(self,"precision_fault") or not self.controls.nu_I:
            return super()._ion_jacobians(phi,positive,negative)
        f=self._fine_work; population=f["positive_m3"]
        mu=-(1-population/DD(self.material.P_lim_node)).log()
        vt=DD(self.thermal_voltage)*(1.01 if self.precision_fault=="thermal_voltage" else 1)
        sign=-1 if self.precision_fault=="drift_sign" else 1
        xi=sign*diff(f["phi_V"])/vt+diff(mu)
        prefactor=DD(self.material.D_ion_face)/DD(np.diff(self.grid))
        if self.precision_fault=="diffusion":prefactor=prefactor*1.01
        dxi=bernoulli_derivative_pair(xi)*population[:-1]+bernoulli_derivative_pair(-xi)*population[1:]
        dmu=1/(DD(self.material.P_lim_node)-population)
        dl=prefactor*(bernoulli_pair(xi)-dxi*dmu[:-1])
        dr=prefactor*(-bernoulli_pair(-xi)+dxi*dmu[1:])
        dp=prefactor*dxi*sign/vt
        if self.precision_fault=="omit_ion":dl,dr,dp=(DD(np.zeros(dl.shape)) for _ in range(3))
        if self.precision_fault=="single_face_sign":
            active=np.flatnonzero(self.material.D_ion_face>0)
            if active.size:
                i=int(active[0]);dl,dr,dp=(put(v,i,-v[i]) for v in (dl,dr,dp))
        density=self._density_jacobian(population.hi,self.positive_nodes,self.positive_slice,self.dimension)
        potential=sparse.lil_matrix((self.node_count,self.dimension))
        for j in range(self.interior_count):potential[j+1,self.potential_slice.start+j]=self.thermal_voltage
        potential=potential.tocsr()
        flux=(sparse.diags(dl.hi)@density[:-1]+sparse.diags(dr.hi)@density[1:]
              -sparse.diags(dp.hi)@potential[:-1]+sparse.diags(dp.hi)@potential[1:])
        rate=-sparse.diags(1/self.widths)@self._divergence@flux
        zeros=sparse.csr_matrix((self.node_count,self.dimension))
        return density,zeros,rate.tocsr(),zeros

    def _with_step_electrostatics(self, state):
        if not self._fine_work or not hasattr(self, "_fine_reference"):
            return super()._with_step_electrostatics(state)
        fine = {k: v.copy() for k, v in self._fine_work.items()}
        fine["sheet_charge_C_m2"] = DD(Q) * DD(self.trap_density) * (DD(self.equilibrium_occupancy) - fine["occupancy"])
        fine["storage"] = cat(fine["n_m3"][1:-1], fine["p_m3"][1:-1],
                              DD(self.trap_density) * fine["occupancy"], fine["positive_m3"][self.positive_nodes])
        poisson = self._poisson_pair(fine["phi_V"], fine["n_m3"], fine["p_m3"],
                                     fine["positive_m3"], fine["sheet_charge_C_m2"])
        fine["poisson_residual_C_m2"] = poisson
        result = PrecisionState(**{f.name: getattr(state, f.name) for f in fields(_InterfaceIonDeviceState)}, fine=fine)
        result.sheet_charge = fine["sheet_charge_C_m2"].hi.copy()
        result.storage = fine["storage"].hi.copy()
        result.poisson_residual = poisson.hi.copy()
        result.direct_poisson_residual = poisson.hi.copy()
        # Recompute the electrostatic trace residual with the same frozen
        # capacitances and prescribed trace jump as the original operator.
        locals_fine=[]
        for k, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            trace = fine["trace_potential_V"][k]
            cl = EPS_0*self.material.eps_r[left]/self.material.iface_qss_left_distances_m[k]
            cr = EPS_0*self.material.eps_r[right]/self.material.iface_qss_right_distances_m[k]
            result.local_residual[6*k] = number(trace[1] - trace[0] - self._prescribed_trace_jump[k])
            result.local_residual[6*k+1] = number(
                DD(cl)*(trace[0]-fine["phi_V"][left])
                + DD(cr)*(trace[1]-fine["phi_V"][right]) - fine["sheet_charge_C_m2"][k])
            locals_fine.append(replace(state.local[k],
                electrostatic_residual=result.local_residual[6*k:6*k+2].copy(),
                sheet_charge_C_m2=float(result.sheet_charge[k])))
        result.local=tuple(locals_fine)
        return result

    def storage_increment(self, state, previous):
        if not isinstance(state, PrecisionState) or not isinstance(previous, PrecisionState):
            return super().storage_increment(state, previous)
        return (state.fine["storage"] - previous.fine["storage"]).hi.copy()

    def potential_increment(self, state, previous):
        if not isinstance(state, PrecisionState) or not isinstance(previous, PrecisionState):
            return super().potential_increment(state, previous)
        return (state.fine["phi_V"] - previous.fine["phi_V"]).hi.copy()

    def interface_current_sides(self,state,previous=None,dt=None):
        if not isinstance(state,PrecisionState):
            return super().interface_current_sides(state,previous,dt)
        # Start with the original conduction construction; independently
        # replace the displacement from actual fine trace/bulk differences.
        conduction,_,_=super().interface_current_sides(state,None,None)
        displacement=np.zeros_like(conduction)
        if previous is not None and dt is not None:
            phi=state.fine["phi_V"]-previous.fine["phi_V"]
            trace=state.fine["trace_potential_V"]-previous.fine["trace_potential_V"]
            for k,(left,right) in enumerate(zip(self.left_nodes,self.right_nodes)):
                cl=EPS_0*self.material.eps_r[left]/self.material.iface_qss_left_distances_m[k]
                cr=EPS_0*self.material.eps_r[right]/self.material.iface_qss_right_distances_m[k]
                displacement[k]=self.polarity*np.array([
                    number(-DD(cl)*(trace[k,0]-phi[left])/dt),
                    number(DD(cr)*(trace[k,1]-phi[right])/dt)])
        return conduction,displacement,conduction+displacement

    def transient_current_metrics(self,state,previous,dt):
        if not isinstance(state,PrecisionState):
            return super().transient_current_metrics(state,previous,dt)
        delta=state.fine["phi_V"]-previous.fine["phi_V"]
        displacement=(DD(self.polarity)*DD(self.eps_face)*(-diff(delta)/DD(np.diff(self.grid)))/dt).hi.copy()
        ic,ide,im=self.interface_current_sides(state,previous,dt)
        for k,face in enumerate(self.interface_faces):displacement[face]=ide[k,0]
        total=state.conduction+displacement
        spread=float(np.ptp(total))/max(float(np.max(np.abs(total))),1e-20)
        interface_error=float(np.max(np.abs(im[:,0]-im[:,1])/np.maximum(np.max(np.abs(im),axis=1),1e-20)))
        return displacement,total,ic,ide,spread,interface_error

    def solver_current_metrics(self,state,previous,dt):
        if not isinstance(state,PrecisionState):return super().solver_current_metrics(state,previous,dt)
        from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
        metrics=self.transient_current_metrics(state,previous,dt)
        storage=self.storage_increment(state,previous)
        rho=self._increment_charge_density(storage)
        delta=state.fine["phi_V"]-previous.fine["phi_V"]
        displacement=(-DD(self.material.poisson_factor.C)*diff(delta)).hi.copy()
        contact=physical_contact_displacement(displacement,rho,self.widths)/dt
        total=self.polarity*(state.current_n[[0,-1]]+state.current_p[[0,-1]]+contact)
        combined=np.r_[metrics[1],total]
        spread=float(np.ptp(combined))/max(float(np.max(np.abs(combined))),1e-20)
        return (*metrics[:4],spread,metrics[5])

    def integrated_charge(self, state):
        if not isinstance(state, PrecisionState):
            return super().integrated_charge(state)
        f = state.fine
        return number((DD(Q)*(f["p_m3"][1:-1]-f["n_m3"][1:-1]
                       +f["positive_m3"][1:-1]-DD(self.material.P_ion0[1:-1]))
                       * DD(self.widths[1:-1])).sum() + f["sheet_charge_C_m2"].sum())

    def integrated_charge_increment(self, state, previous):
        if not isinstance(state, PrecisionState):
            return super().integrated_charge_increment(state, previous)
        f, p = state.fine, previous.fine
        charge = ((f["p_m3"]-p["p_m3"]) - (f["n_m3"]-p["n_m3"])
                  + (f["positive_m3"]-p["positive_m3"]))
        return number((DD(Q)*charge[1:-1]*DD(self.widths[1:-1])).sum()
                       + (f["sheet_charge_C_m2"]-p["sheet_charge_C_m2"]).sum())

    def newton_residual_target(self, previous, storage_scale, poisson_scale, local_scale):
        # A newly solved high precision initial field no longer needs a
        # deliberately inherited absolute Gauss offset in the search target.
        return np.zeros(len(storage_scale)+len(poisson_scale)+len(local_scale))

    def _independent_eliminated_phi(self, state, voltage):
        f = state.fine
        old = self.system.evaluate_quasi_fermi_increments_defect_ion_combined(
            f["dqfn_V"].hi, f["dqfp_V"].hi, 0.,
            positive_ion_density_m3=f["positive_m3"].hi,
            negative_ion_density_m3=None, dynamic_interface_occupancy=f["occupancy"].hi,
            V_app=float(voltage))
        phi = DD(old.phi)
        # This solves the charged, fixed-QF nonlinear Poisson problem anew.
        # The immutable preparation, never the direct solution, anchors n/p.
        for _ in range(12):
            change=phi-DD(self.system.phi0)
            n=(DD(self.system.log_n0)+(f["dqfn_V"]+change)/DD(self.thermal_voltage)).exp()
            p=(DD(self.system.log_p0)+(f["dqfp_V"]-change)/DD(self.thermal_voltage)).exp()
            n,p=put(n,[0,-1],self.reference_n[[0,-1]]),put(p,[0,-1],self.reference_p[[0,-1]])
            residual = self._poisson_pair(phi,n,p,f["positive_m3"],f["sheet_charge_C_m2"])
            jac = self._poisson_laplacian[:,1:-1] - sparse.diags(
                Q*(n.hi[1:-1]+p.hi[1:-1])*self.material.poisson_factor.h_cell/self.thermal_voltage)
            delta = np.asarray(spsolve(jac.tocsr(), -residual.hi))
            phi = put(phi, slice(1,-1), phi[1:-1]+DD(delta))
            if np.max(np.abs(delta), initial=0.) < 1e-28:
                return phi
        raise RuntimeError("independent compensated Poisson did not converge in 12 corrections")

    def eliminated_operator_diagnostics(self, state, voltage):
        values = super().eliminated_operator_diagnostics(state, voltage)
        if not isinstance(state, PrecisionState):
            return values
        phi = self._independent_eliminated_phi(state,voltage)
        flux = (ion_flux_pair(phi,state.fine["positive_m3"],self.material,np.diff(self.grid),fault=self.precision_fault)
                if self.controls.nu_I else DD(np.zeros(self.node_count-1)))
        rate = -diff(cat(0,flux,0))/DD(self.widths)
        for name,direct,other in (("positive_ion_flux",state.fine["positive_flux_m2_s"],flux),
                                  ("positive_ion_rate",state.fine["positive_rate_m3_s"],rate)):
            delta = (direct-other).to_float()
            left,right=direct.to_float(),other.to_float()
            floor=float(values[name]["normalization_floor"])
            peak=max(float(np.max(np.abs(left),initial=0.)),float(np.max(np.abs(right),initial=0.)))
            scale=max(peak,floor)
            absolute=float(np.max(np.abs(delta),initial=0.))
            values[name].update(direct=left,eliminated=right,difference=delta,
                direct_maximum_absolute=float(np.max(np.abs(left),initial=0.)),
                eliminated_maximum_absolute=float(np.max(np.abs(right),initial=0.)),
                maximum_absolute_difference=absolute,normalization_floor=floor,normalization_scale=scale,
                floor_active=peak<floor,relative_error=absolute/scale)
        return values


def precision_capabilities():
    return {"representation_id":REPRESENTATION,"state_low_fields_persisted":True,
            "low_bits_consumed":True,"replay_supported":True,
            "original_checks_preserved":True,"full_production_migration":False,
            "scope":"explicit_bounded_prototype_not_R1_2_qualification"}


def _precision_solve_step(original,system,coordinate,previous,voltage,dt,policy,*,check_jacobian,scaling_system=None):
    result=original(system,coordinate,previous,voltage,dt,policy,
                    check_jacobian=check_jacobian,scaling_system=scaling_system)
    if not isinstance(system,CompensatedR1System):return result
    from . import interface_defect_transient as transient
    reference=system if scaling_system is None else scaling_system
    scales=(reference.storage_scale(previous.storage,previous,dt,policy),
            reference.poisson_scale(policy),reference.local_algebraic_scale(policy))
    used=result[1]
    def fail(message,state,corrections,error):
        from .one_dimensional_mechanism_r1_state import snapshot,json_data
        exc=transient.InterfaceDefectTransientError(message)
        residual,_,_=system.residual_and_jacobian(state.coordinate,voltage,previous,dt,*scales)
        exc.result=json_data({"schema":"R1V7PrecisionCorrectionFailureV1",
            "dt_s":dt,"voltage_V":voltage,"corrections":corrections,
            "original_ion_gate":error,"accepted_trajectory":False,
            "previous_state":snapshot(system,previous),"attempted_state":snapshot(system,state),
            "coordinate":state.coordinate,"scaled_residual_vector":residual,
            "storage_scale":scales[0],"poisson_scale":scales[1],"local_scale":scales[2]})
        return exc
    # Additional joint Newton corrections happen BEFORE any state is handed
    # to the integrator/observer. All original residual/current limits are
    # rechecked; this is not a projection of a saved/accepted trajectory.
    for correction in range(9):
        state=result[0]
        diagnostics=system.eliminated_operator_diagnostics(state,voltage)
        error=max(diagnostics[k]["relative_error"] for k in ("positive_ion_flux","positive_ion_rate"))
        if error<=policy.maximum_eliminated_operator_relative_error:
            return result
        if correction==8 or used+1>=policy.maximum_newton_iterations:
            raise fail("V7 coupled precision corrector failed the original ion gate",state,correction,error)
        trial=state.coordinate.copy()
        residual,jacobian,state=system.residual_and_jacobian(trial,voltage,previous,dt,*scales)
        rhs=np.zeros_like(residual)
        start=len(scales[0]); stop=start+len(scales[1])
        rhs[start:stop]=residual[start:stop]
        for k in range(system.interface_count):rhs[stop+6*k:stop+6*k+2]=residual[stop+6*k:stop+6*k+2]
        delta=np.asarray(spsolve(jacobian,-rhs))
        following=trial+delta
        if not np.all(np.isfinite(following)) or np.array_equal(following,trial):
            raise fail("V7 within-step precision correction is unresolved in its coordinate",state,correction,error)
        # Use the existing nonlinear solver to accept this new full coupled
        # state, with its remaining original iteration budget.
        remaining=policy.maximum_newton_iterations-used-1
        refined=original(system,following,previous,voltage,dt,
            replace(policy,maximum_newton_iterations=remaining,
                    maximum_near_acceptance_nonmonotone_steps=policy.maximum_near_acceptance_nonmonotone_steps-result[5]),
            check_jacobian=check_jacobian,
            scaling_system=scaling_system)
        used+=1+refined[1]
        system.precision_constraint_corrections+=1
        result=(refined[0],used,refined[2],max(result[3],refined[3]),
                max(result[4],refined[4]),result[5]+refined[5])
        if result[5]>policy.maximum_near_acceptance_nonmonotone_steps:
            raise fail("V7 precision corrector exceeded the original nonmonotone budget",result[0],correction+1,error)
    raise AssertionError("unreachable precision-correction loop")


@contextmanager
def precision_context(mode="compensated"):
    global _ACTIVE
    if mode != "compensated" or _ACTIVE:
        raise ValueError("precision context requires an unnested compensated run")
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol  # noqa: F401
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation  # noqa: F401
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as independent
    from perovskite_sim.experiments import interface_defect_transient as transient
    from .one_dimensional_mechanism_r1_precision_physics import independent_physics_row_pair, SHARED_CONSTITUTIVE_DEPENDENCIES
    original_snapshot=states.snapshot
    def snapshot(system,state):
        record=original_snapshot(system,state)
        if isinstance(state,PrecisionState):
            for key,value in state.fine.items():
                record["precision_"+key+"_hi"]=value.hi.tolist()
                record["precision_"+key+"_lo"]=value.lo.tolist()
        return record
    changes=[]
    def patch(module,name,value):
        changes.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    original_prepare,original_verify=states.prepare_common_state,states.verify_prepared_physics
    original_currents,original_increments=independent._currents,independent._increments
    original_solve=transient._solve_step
    original_physics_row=independent.independent_physics_row
    @contextmanager
    def legacy_context():
        saved=[]
        for module,name,old in changes:
            saved.append((module,name,getattr(module,name)))
            setattr(module,name,old)
        try:yield
        finally:
            for module,name,value in saved:setattr(module,name,value)
    from .one_dimensional_mechanism_r1_precision_state import build_hooks
    prepare,verify=build_hooks(original_prepare,original_verify,legacy_context,CompensatedR1System.from_baseline)
    def independent_currents(system,state):
        if not isinstance(state,PrecisionState):return original_currents(system,state)
        # Rebuild the interface constitutive state rather than reading saved
        # current/rate arrays. Shared fine bulk laws are declared in scope.
        locals_new=[]
        for k,item in enumerate(state.local):
            balance=independent._interface_balance(system,state,k)
            tangent=replace(item.tangent,balance=balance)
            locals_new.append(replace(item,tangent=tangent))
        _,_,electron,hole=carrier_currents_pair(system,state.fine,locals_new)
        ion=(ion_flux_pair(state.fine["phi_V"],state.fine["positive_m3"],system.material,np.diff(system.grid))
             if system.controls.nu_I else DD(np.zeros(system.node_count-1)))
        interface=np.empty((system.interface_count,2))
        for k,face in enumerate(system.interface_faces):
            flux=locals_new[k].tangent.balance.bulk_flux_m2_s
            interface[k]=Q*np.array([-flux[0]+flux[1],flux[2]-flux[3]])+Q*ion.hi[face]
        return electron.hi.copy(),hole.hi.copy(),ion.hi.copy(),interface
    def independent_increments(system,state,previous):
        if not isinstance(state,PrecisionState):return original_increments(system,state,previous)
        f,p=state.fine,previous.fine
        rho=DD(Q)*((f["p_m3"]-p["p_m3"])-(f["n_m3"]-p["n_m3"])+(f["positive_m3"]-p["positive_m3"]))
        occupied=DD(system.trap_density)*(f["occupancy"]-p["occupancy"])
        return tuple(v.hi.copy() for v in (rho,occupied,f["phi_V"]-p["phi_V"],f["trace_potential_V"]-p["trace_potential_V"]))
    for module in list(sys.modules.values()):
        if module is None or not getattr(module,"__name__","").startswith("perovskite_sim.experiments"):
            continue
        if getattr(module,"snapshot",None) is original_snapshot:
            patch(module,"snapshot",snapshot)
        if getattr(module,"prepare_common_state",None) is original_prepare:
            patch(module,"prepare_common_state",prepare)
        if getattr(module,"verify_prepared_physics",None) is original_verify:
            patch(module,"verify_prepared_physics",verify)
        if getattr(module,"independent_physics_row",None) is original_physics_row:
            patch(module,"independent_physics_row",independent_physics_row_pair)
    patch(independent,"_currents",independent_currents)
    patch(independent,"_increments",independent_increments)
    patch(independent,"SHARED_CONSTITUTIVE_DEPENDENCIES",SHARED_CONSTITUTIVE_DEPENDENCIES)
    patch(transient,"_solve_step",lambda *a,**kw:_precision_solve_step(original_solve,*a,**kw))
    _ACTIVE=True
    try:
        yield
    finally:
        for module,name,old in reversed(changes):
            setattr(module,name,old)
        _ACTIVE=False

"""R1 compensated arithmetic for the explicit production backend.

Importing this module changes no default solver, equation, acceptance limit
or reader. The historical precision_context remains a V7/V8 compatibility
interface; V9 production execution calls the explicit helpers directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
import copy
import json
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
_ONE = DD(1)
_HALF_NEGATIVE = DD(-.5)
_CHARGE = DD(Q)
_BERNOULLI_DERIVATIVE_COEFFICIENTS = (
    _ONE / 6, -_ONE / 180, _ONE / 5040, -_ONE / 151200,
)
REFERENCE_FIELDS = (
    "phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
    "occupancy", "trace_potential_V", "trace_state_m3",
)


def pair_words(value):
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def fine_identity(fine):
    """Content identity of primary physical inputs, independent of caches."""
    from .one_dimensional_mechanism_r1_state import digest
    return digest({name: pair_words(fine[name]) for name in REFERENCE_FIELDS})


def cat(*items):
    items = [x if isinstance(x, DD) else DD(x) for x in items]
    return DD._trusted_parts(np.concatenate([np.atleast_1d(x.hi) for x in items]),
                             np.concatenate([np.atleast_1d(x.lo) for x in items]))


def put(value, index, part):
    part = part if isinstance(part, DD) else DD(part)
    hi, lo = value.hi.copy(), value.lo.copy()
    hi[index], lo[index] = part.hi, part.lo
    return DD._trusted_parts(hi, lo)


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


def _choose_pair(mask, left, right):
    return DD._trusted_parts(np.where(mask,left.hi,right.hi),np.where(mask,left.lo,right.lo))


def bernoulli_signed_pair(x):
    """Return B(x), B(-x) using one nonnegative exponential evaluation.

    B(-a)=B(a)+a for a>=0 avoids subtracting nearly equal positive values.
    The algebraic identity changes the DD rounding path, not the SG law.
    """
    magnitude=abs(x)
    forward=bernoulli_pair(magnitude)
    backward=forward+magnitude
    nonnegative=x.hi>=0
    return (_choose_pair(nonnegative,forward,backward),
            _choose_pair(nonnegative,backward,forward))


def bernoulli_derivative_pair(x):
    result=DD(np.full(x.shape,-.5))
    small=np.abs(x.hi)<1e-3
    if np.any(small):
        a=x[small]
        a2=a*a
        c1,c3,c5,c7=_BERNOULLI_DERIVATIVE_COEFFICIENTS
        result=put(result,small,_HALF_NEGATIVE+a*(c1+a2*(c3+a2*(c5+a2*(c7+a2*_ONE/4790016)))))
    if np.any(~small):
        a=x[~small]; denominator=a.expm1()
        result=put(result,~small,(denominator-a*(denominator+_ONE))/(denominator*denominator))
    return result


def bernoulli_derivative_signed_pair(x):
    """Return B'(x), B'(-x) using B'(-a)=-1-B'(a), a>=0."""
    forward=bernoulli_derivative_pair(abs(x))
    backward=-_ONE-forward
    nonnegative=x.hi>=0
    return (_choose_pair(nonnegative,forward,backward),
            _choose_pair(nonnegative,backward,forward))


def _ion_terms(phi, population, material, spacing, fault, constants=None):
    constants = {} if constants is None else constants
    if material.has_dual_ions or not material.ion_steric_diffusion_only:
        raise ValueError("V7 prototype supports the frozen positive-ion steric model")
    limit = constants.get("ion_limit")
    if limit is None:
        limit = DD(material.P_lim_node)
    theta = population / limit
    if np.any(theta.hi < 0) or np.any(theta.hi >= .999):
        raise ValueError("V7 prototype reached its original pre-clipping bound")
    chemical = -(_ONE - theta).log()
    vt = constants.get("ion_vt")
    if vt is None:
        vt = DD(material.V_T_device)
    vt = vt * (1.01 if fault == "thermal_voltage" else 1)
    sign = -1 if fault == "drift_sign" else 1
    xi = sign * diff(phi) / vt + diff(chemical)
    forward,backward=bernoulli_signed_pair(xi)
    return {"chemical": chemical, "xi": xi, "vt": vt, "sign": sign,
            "forward": forward, "backward": backward}


def ion_flux_pair(phi, population, material, spacing, *, fault="none", _terms=None):
    """Same frozen one-positive-species SG law, evaluated with two words.

    ``_terms`` is reserved for one production evaluation. Independent callers
    construct their own intermediates from their actual arguments.
    """
    terms = _ion_terms(phi,population,material,spacing,fault) if _terms is None else _terms
    diffusion = DD(material.D_ion_face) * (1.01 if fault == "diffusion" else 1)
    # Both terms retain ~106 bits before cancellation.  A fault is applied
    # inside this constitutive calculation, never to a comparator's output.
    flux = diffusion / DD(spacing) * (
        terms["forward"] * population[:-1]
        - terms["backward"] * population[1:])
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
    mat = system.material
    constants = system._refresh_fine_constants()
    vt = constants["vt"]
    phi,n,p=fine["phi_V"],fine["n_m3"],fine["p_m3"]
    xi_n=diff(phi+constants["chi"])/vt
    xi_p=diff(phi+constants["chi"]+constants["Eg"])/vt
    dn=constants["edge_n"]+diff(fine["dqfn_V"])/vt
    dp=-(constants["edge_p"]+diff(fine["dqfp_V"])/vt)
    def stable(forward,backward,drive):
        positive=drive.hi>=0
        value=DD(np.zeros(drive.shape))
        if np.any(~positive):
            value=put(value,~positive,backward[~positive]*drive[~positive].expm1())
        if np.any(positive):
            value=put(value,positive,forward[positive]*(-(-drive[positive]).expm1()))
        return value
    bn_forward,bn_backward=bernoulli_signed_pair(xi_n)
    bp_forward,bp_backward=bernoulli_signed_pair(xi_p)
    prefactor_n,prefactor_p=system._fine_carrier_prefactors
    jn=prefactor_n*stable(
        bn_forward*n[1:],bn_backward*n[:-1],dn)
    jp=prefactor_p*stable(
        bp_forward*p[:-1],bp_backward*p[1:],dp)
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


class EliminatedDiagnostics(dict):
    """Original comparison channels with same-call fine evidence attached."""

    def __init__(self, values, precision_evidence):
        super().__init__(values)
        self.precision_evidence = precision_evidence


_INDEPENDENT_FIXED_FIELDS = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy", "sheet_charge_C_m2")
_INDEPENDENT_PREPARATION_FIELDS = (
    "phi0_V", "log_n0", "log_p0", "contact_n_m3", "contact_p_m3", "thermal_voltage_V",
    "poisson_capacitance_F_m2", "poisson_width_m", "N_D_m3", "N_A_m3", "ion_background_m3",
    "interface_nodes", "sheet_weights", "boundary_phi_V",
)


@dataclass(frozen=True, slots=True)
class IndependentPoissonInputs:
    """Immutable allowlisted inputs, without direct phi, n/p, flux or caches.

Canonical JSON owns copies of the input words and coefficients. Unlike a
frozen object containing NumPy arrays or DD objects, its data cannot change
through an alias while an independent solve is in progress.
"""
    fixed_inputs_json: str
    preparation_json: str
    seed_phi_json: str
    seed_provenance: str

    def __post_init__(self):
        if set(json.loads(self.fixed_inputs_json)) != set(_INDEPENDENT_FIXED_FIELDS):
            raise ValueError("independent Poisson fixed-input field coverage mismatch")
        if set(json.loads(self.preparation_json)) != set(_INDEPENDENT_PREPARATION_FIELDS):
            raise ValueError("independent Poisson preparation field coverage mismatch")
        if set(json.loads(self.seed_phi_json)) != {"hi", "lo"}:
            raise ValueError("independent Poisson seed must retain both words")
        if self.seed_provenance not in (
                "independent_legacy_fixed_qf_evaluation",
                "same_diagnostics_call_legacy_potential_eliminated_channel"):
            raise ValueError("independent Poisson seed provenance is unavailable")

    def to_dict(self):
        return {"schema": "R1IndependentPoissonInputsV1",
                "fixed_inputs": json.loads(self.fixed_inputs_json),
                "preparation": json.loads(self.preparation_json),
                "seed_phi_V": json.loads(self.seed_phi_json),
                "seed_provenance": self.seed_provenance}


def solve_independent_poisson(inputs):
    """Solve fixed-QF Poisson solely from the immutable, isolated input DTO."""
    from .one_dimensional_mechanism_r1_state import digest
    if not isinstance(inputs, IndependentPoissonInputs):
        raise TypeError("independent Poisson requires an isolated input bundle")
    payload = inputs.to_dict()
    f = {name: DD(value["hi"], value["lo"]) for name, value in payload["fixed_inputs"].items()}
    a = payload["preparation"]
    seed = payload["seed_phi_V"]
    phi = put(DD(seed["hi"], seed["lo"]), [0, -1], a["boundary_phi_V"])
    phi0, log_n0, log_p0 = DD(a["phi0_V"]), DD(a["log_n0"]), DD(a["log_p0"])
    vt, c, widths = DD(a["thermal_voltage_V"]), DD(a["poisson_capacitance_F_m2"]), DD(a["poisson_width_m"])
    donors, acceptors, background = DD(a["N_D_m3"]), DD(a["N_A_m3"]), DD(a["ion_background_m3"])
    laplacian = sparse.diags((c.hi[1:-1], -(c.hi[:-1] + c.hi[1:]), c.hi[1:-1]),
                            (-1, 0, 1), format="csr")

    def populations(potential):
        change = potential - phi0
        n = (log_n0 + (f["dqfn_V"] + change) / vt).exp()
        p = (log_p0 + (f["dqfp_V"] - change) / vt).exp()
        return (put(n, [0, -1], a["contact_n_m3"]),
                put(p, [0, -1], a["contact_p_m3"]))

    def poisson(potential, n, p):
        rho = _CHARGE * (p - n + donors - acceptors + f["positive_m3"] - background)
        residual = diff(c * diff(potential)) + rho[1:-1] * widths
        for k, (nodes, weights) in enumerate(zip(a["interface_nodes"], a["sheet_weights"])):
            for node, weight in zip(nodes, weights):
                residual = put(residual, node - 1, residual[node - 1] + DD(weight) * f["sheet_charge_C_m2"][k])
        return residual

    for iteration in range(1, 13):
        n, p = populations(phi)
        residual = poisson(phi, n, p)
        jac = laplacian - sparse.diags(Q * (n.hi[1:-1] + p.hi[1:-1])
                                       * widths.hi / a["thermal_voltage_V"])
        delta = np.asarray(spsolve(jac.tocsr(), -residual.hi))
        phi = put(phi, slice(1, -1), phi[1:-1] + DD(delta))
        maximum = float(np.max(np.abs(delta), initial=0.))
        if maximum < 1e-28:
            n, p = populations(phi)
            residual = poisson(phi, n, p)
            solved = {"phi_V": phi, "n_m3": n, "p_m3": p,
                      "poisson_residual_C_m2": residual}
            return solved, {
                "kind": "independent_fixed_qf_poisson", "iterations": iteration,
                "last_correction_max_abs_V": maximum,
                "poisson_residual_max_abs_C_m2": float(np.max(np.abs(residual.to_float()), initial=0.)),
                "used_direct_phi": False, "converged": True,
                "seed_provenance": inputs.seed_provenance, "seed_phi_V": seed,
                "fixed_input_digest": digest(payload["fixed_inputs"]),
                "solve_inputs_digest": digest(payload),
                "solved_fields_digest": digest({key: pair_words(value) for key, value in solved.items()}),
            }
    raise RuntimeError("independent compensated Poisson did not converge in 12 corrections")


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
        system._fine_evaluation_cache={}
        system._fine_constants={}
        system._reference_quantization_enabled=False
        system.rebase_evidence=None
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
        system._precision_upstream = {
            "schema": "R1PrecisionArithmeticContextV1",
            "inputs_derived_from_final_state": False,
            "thermal_voltage_V": float(baseline.thermal_voltage),
            "qf_anchors": {
                "phi0_V": np.asarray(baseline.system.phi0).tolist(),
                "log_n0": np.asarray(baseline.system.log_n0).tolist(),
                "log_p0": np.asarray(baseline.system.log_p0).tolist(),
                "contact_n_m3": baseline.reference_n[[0,-1]].tolist(),
                "contact_p_m3": baseline.reference_p[[0,-1]].tolist(),
                "electron_reference_V": np.asarray(baseline.qfn_reference).tolist(),
                "hole_reference_V": np.asarray(baseline.qfp_reference).tolist(),
                "dqfn_V": np.asarray(baseline.dqfn_dc).tolist(),
                "dqfp_V": np.asarray(baseline.dqfp_dc).tolist(),
            },
            "sheet_inputs": {
                "trap_density_m2": np.asarray(baseline.trap_density).tolist(),
                "equilibrium_occupancy": np.asarray(baseline.equilibrium_occupancy).tolist(),
                "static_sheet_charge_C_m2": [g.fixed_sheet_charge_C_m2 for g in geometry],
            },
            "trace_geometry": {
                "capacitances_F_m2": [[EPS_0*g.eps_r_left/g.left_distance_m,
                                        EPS_0*g.eps_r_right/g.right_distance_m] for g in geometry],
                "jump_V": [g.potential_jump_right_minus_left_V for g in geometry],
                "sheet_weights": [list(baseline._sheet_weights(k)) for k in range(baseline.interface_count)],
            },
            "fixed_population_inputs": {
                "positive_m3": np.asarray(baseline.reference_positive).tolist(),
                "occupancy": np.asarray(baseline.reference_occupancy).tolist(),
            },
        }
        system.precision_initialization_diagnostics=copy.deepcopy(baseline.precision_initialization_diagnostics)
        system._fine_anchor={k:v.copy() for k,v in fine.items()}
        system._fine_reference=system._fine_anchor
        system._step_reference=None
        system._maximum_eliminated_operator_components={}
        state=system.evaluate(np.zeros(system.dimension),0.)
        return system,state

    def rebase(self, previous):
        working, local = super().rebase(previous)
        reference = {k: previous.fine[k].copy() for k in REFERENCE_FIELDS}
        working.rebase_evidence = None
        if self._reference_quantization_enabled:
            before_identity = fine_identity(previous.fine)
            quantized = {k: DD(v.hi) for k, v in reference.items()}
            # Only the coordinate reference is quantized. local is the real
            # physical previous state (with its coordinate reset by super).
            # All storage, charge and displacement consumers retain local.
            working.rebase_evidence = {
                "schema": "R1RebaseReferenceQuantizationV1",
                "fields": list(REFERENCE_FIELDS),
                "reference_before": {k: pair_words(v) for k,v in reference.items()},
                "reference_after": {k: pair_words(v) for k,v in quantized.items()},
                "reference_before_identity": before_identity,
                "reference_after_identity": fine_identity(quantized),
                "physical_previous_identity": fine_identity(local.fine),
                "physical_previous_unchanged": fine_identity(local.fine) == before_identity,
                "reference_role": "coordinate_reference_only",
                "physical_previous_role": "full_precision_accepted_state",
                "derived_fields": "recomputed_from_primary_reference_by_evaluate",
            }
            reference = quantized
        working._fine_reference = reference
        working._fine_work = {}
        working._fine_evaluation_cache = {}
        return working, local

    def enable_reference_quantization(self):
        """Enable the frozen intervention for future rebases only.

        The single-tier runner/replayer invokes this after its first regular
        accepted step. Initialization deliberately never enables it.
        """
        self._reference_quantization_enabled = True

    def precision_arithmetic_context(self):
        result = copy.deepcopy(self._precision_upstream)
        result["zero_minus_fine"] = {k: pair_words(v) for k,v in self._fine_anchor.items()}
        result["initialization"] = copy.deepcopy(self.precision_initialization_diagnostics)
        result["trace_state_arithmetic"] = copy.deepcopy(
            self.precision_initialization_diagnostics["trace_state_arithmetic"])
        return result

    def _refresh_fine_constants(self):
        """Rebuild after any material value, geometry or fault identity change."""
        mat = self.material
        inputs = {
            "vt": self.thermal_voltage, "ion_vt": mat.V_T_device,
            "ion_limit": mat.P_lim_node, "ion_diffusion": mat.D_ion_face,
            "spacing": np.diff(self.grid), "widths": self.widths,
            "chi": mat.chi, "Eg": mat.Eg, "Dn": mat.D_n_face, "Dp": mat.D_p_face,
            "edge_n": self.system.reference_edge_drop_n, "edge_p": self.system.reference_edge_drop_p,
            "ND": mat.N_D, "NA": mat.N_A, "ion_background": mat.P_ion0,
            "poisson_C": mat.poisson_factor.C, "poisson_h": mat.poisson_factor.h_cell,
            "trap_density": self.trap_density, "equilibrium_occupancy": self.equilibrium_occupancy,
            "phi0": self.system.phi0, "log_n0": self.system.log_n0, "log_p0": self.system.log_p0,
        }
        identity = (id(mat), id(self.system), self.precision_fault)
        cached = self._fine_constants
        if (getattr(self,"_fine_constant_identity",None) != identity
                or set(cached) != set(inputs)
                or any(not np.array_equal(cached[k].hi,np.asarray(v)) for k,v in inputs.items())):
            constants = {k: DD(v) for k,v in inputs.items()}
            # These are state-independent coefficients in exactly the same
            # multiplication/division order used by carrier_currents_pair.
            # Construct both before publishing a refreshed cache, so an
            # arithmetic failure cannot leave old prefactors with new inputs.
            prefactors = (_CHARGE*constants["Dn"]/constants["spacing"],
                          _CHARGE*constants["Dp"]/constants["spacing"])
            self._fine_constants = constants
            self._fine_carrier_prefactors = prefactors
            self._fine_constant_identity = identity
            self._fine_evaluation_cache = {}
        return self._fine_constants

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
        constants=self._refresh_fine_constants()
        self._fine_evaluation_cache = {}
        self._precision_contact_evaluation={name:result[index][[0,-1]].copy()
            for name,index in (("dqfn_V",0),("dqfp_V",1),("phi_V",2),("n_m3",3),("p_m3",4))}
        z = DD(np.asarray(coordinate, dtype=float))
        z_phi,z_n,z_p=z[self.potential_slice],z[self.electron_slice],z[self.hole_slice]
        ref = self._fine_reference
        phi = put(ref["phi_V"], [0, -1], result[2][[0, -1]])
        phi = put(phi, slice(1, -1), ref["phi_V"][1:-1] + constants["vt"] * z_phi)
        qn = put(ref["dqfn_V"], [0, -1], result[0][[0, -1]])
        qp = put(ref["dqfp_V"], [0, -1], result[1][[0, -1]])
        qn = put(qn, slice(1, -1), ref["dqfn_V"][1:-1] + constants["vt"] * z_n)
        qp = put(qp, slice(1, -1), ref["dqfp_V"][1:-1] + constants["vt"] * z_p)
        n = put(ref["n_m3"], slice(1, -1), ref["n_m3"][1:-1] * (z_n + z_phi).exp())
        p = put(ref["p_m3"], slice(1, -1), ref["p_m3"][1:-1] * (z_p - z_phi).exp())
        odds = z[self.trap_slice].expm1()
        f = ref["occupancy"]
        occupancy = f + f * (1-f) * odds / (1 + f * odds)
        trace_phi, trace_n = ref["trace_potential_V"].copy(), ref["trace_state_m3"].copy()
        for k in range(self.interface_count):
            block = z[self._local_block_slice(k)]
            trace_phi = put(trace_phi, k, ref["trace_potential_V"][k] + constants["vt"] * block[:2])
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
        if self.controls.nu_I:
            terms = _ion_terms(fine["phi_V"],fine["positive_m3"],self.material,
                               np.diff(self.grid),self.precision_fault,self._fine_constants)
            self._fine_evaluation_cache = {"fine": fine, "fault": self.precision_fault, "ion": terms}
            flux = ion_flux_pair(fine["phi_V"], fine["positive_m3"], self.material,
                                 np.diff(self.grid), fault=self.precision_fault, _terms=terms)
        else:
            flux = DD(np.zeros(self.node_count-1))
        boundary = self._ion_boundary_flux()
        rate = -diff(cat(boundary[0], flux, boundary[1])) / self._fine_constants["widths"]
        fine["positive_flux_m2_s"], fine["positive_rate_m3_s"] = flux, rate
        fine["boundary_flux_m2_s"] = boundary
        return rate.hi.copy(), None, flux.hi.copy(), None

    def _ion_boundary_flux(self):
        """Actual boundary inputs; a named hook supports the leakage control."""
        return DD(np.zeros(2))

    def _currents(self,dqfn,dqfp,phi,n,p,local):
        if not self._fine_work or not hasattr(self,"_fine_reference"):
            return super()._currents(dqfn,dqfp,phi,n,p,local)
        values=carrier_currents_pair(self,self._fine_work,local)
        self._fine_work["electron_current_A_m2"]=values[2]
        self._fine_work["hole_current_A_m2"]=values[3]
        return tuple(value.hi.copy() for value in values)

    def _poisson_pair(self, phi, n, p, positive, sigma, *, constants=None):
        c=constants if constants is not None else {
            "ND":DD(self.material.N_D), "NA":DD(self.material.N_A),
            "ion_background":DD(self.material.P_ion0),
            "poisson_C":DD(self.material.poisson_factor.C),
            "poisson_h":DD(self.material.poisson_factor.h_cell),
        }
        rho = _CHARGE * (p - n + c["ND"] - c["NA"] + positive - c["ion_background"])
        residual = diff(c["poisson_C"] * diff(phi))
        residual = residual + rho[1:-1] * c["poisson_h"]
        for k, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            wl, wr = self._sheet_weights(k)
            residual = put(residual, left-1, residual[left-1] + DD(wl) * sigma[k])
            residual = put(residual, right-1, residual[right-1] + DD(wr) * sigma[k])
        return residual

    def _ion_jacobians(self, phi, positive, negative):
        if not self._fine_work or not hasattr(self,"precision_fault") or not self.controls.nu_I:
            return super()._ion_jacobians(phi,positive,negative)
        f=self._fine_work; population=f["positive_m3"]
        self._refresh_fine_constants()
        cache=self._fine_evaluation_cache
        if cache.get("fine") is f and cache.get("fault")==self.precision_fault:
            terms=cache["ion"]
        else:
            terms=_ion_terms(f["phi_V"],population,self.material,np.diff(self.grid),self.precision_fault,self._fine_constants)
        vt,sign,xi=terms["vt"],terms["sign"],terms["xi"]
        prefactor=self._fine_constants["ion_diffusion"]/self._fine_constants["spacing"]
        if self.precision_fault=="diffusion":prefactor=prefactor*1.01
        derivative_forward,derivative_backward=bernoulli_derivative_signed_pair(xi)
        dxi=derivative_forward*population[:-1]+derivative_backward*population[1:]
        dmu=1/(self._fine_constants["ion_limit"]-population)
        dl=prefactor*(terms["forward"]-dxi*dmu[:-1])
        dr=prefactor*(-terms["backward"]+dxi*dmu[1:])
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
        constants=self._fine_constants
        fine["sheet_charge_C_m2"] = _CHARGE * constants["trap_density"] * (constants["equilibrium_occupancy"] - fine["occupancy"])
        fine["storage"] = cat(fine["n_m3"][1:-1], fine["p_m3"][1:-1],
                              constants["trap_density"] * fine["occupancy"], fine["positive_m3"][self.positive_nodes])
        poisson = self._poisson_pair(fine["phi_V"], fine["n_m3"], fine["p_m3"],
                                     fine["positive_m3"], fine["sheet_charge_C_m2"], constants=constants)
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

    def _newton_electrostatic_residuals(self, state):
        """Read the Poisson pair and reconstruct the two local electrostatic pairs.

        These are the same physical residuals used by evaluate. Keeping this
        temporary local pair outside ``state.fine`` preserves its fixed
        seventeen-field representation and never changes a physical state.
        """
        fine = state.fine
        local = []
        for k, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            trace = fine["trace_potential_V"][k]
            cl = EPS_0*self.material.eps_r[left]/self.material.iface_qss_left_distances_m[k]
            cr = EPS_0*self.material.eps_r[right]/self.material.iface_qss_right_distances_m[k]
            local.extend((trace[1]-trace[0]-self._prescribed_trace_jump[k],
                          DD(cl)*(trace[0]-fine["phi_V"][left])
                          + DD(cr)*(trace[1]-fine["phi_V"][right])-fine["sheet_charge_C_m2"][k]))
        return fine["poisson_residual_C_m2"], cat(*local) if local else DD(np.empty(0))

    def newton_residual_target(self, previous, storage_scale, poisson_scale, local_scale):
        """Preserve only the previous accepted state's allowed Gauss residuals.

        This supplies a direction target, not a changed residual or gate.
        The caller retains its original admissibility checks on both the
        actual residual and this normalized previous-only target.
        """
        if not isinstance(previous, PrecisionState):
            return super().newton_residual_target(previous, storage_scale, poisson_scale, local_scale)
        poisson, local = self._newton_electrostatic_residuals(previous)
        start, stop = len(storage_scale), len(storage_scale)+len(poisson_scale)
        target = np.zeros(stop+len(local_scale))
        target[start:stop] = (poisson/DD(poisson_scale)).to_float()
        for k in range(self.interface_count):
            rows = slice(6*k, 6*k+2)
            target[stop+rows.start:stop+rows.stop] = (local[2*k:2*k+2]/DD(local_scale[rows])).to_float()
        return target

    def newton_direction_rhs(self, state, previous, residual, target,
                             storage_scale, poisson_scale, local_scale):
        """Subtract accepted-state electrostatic residuals in DD before rounding.

        All dynamic and local carrier rows retain the original residual.
        This hook never changes the residual used for acceptance, Jacobian,
        currents, eliminated state, or physical snapshots. A different target
        cannot silently replace the previous-state direction contract.
        """
        if not isinstance(state, PrecisionState) or not isinstance(previous, PrecisionState):
            return np.asarray(residual, dtype=float)-np.asarray(target, dtype=float)
        expected = CompensatedR1System.newton_residual_target(
            self, previous, storage_scale, poisson_scale, local_scale)
        if not np.array_equal(np.asarray(target), expected):
            raise ValueError("pair Newton direction target differs from the previous accepted state")
        before_poisson, before_local = self._newton_electrostatic_residuals(previous)
        if not any(np.any(value.hi != 0.) or np.any(value.lo != 0.)
                   for value in (before_poisson, before_local)):
            # A genuinely zero prior residual must retain the original
            # binary64 direction, including its existing rounding boundary.
            return np.asarray(residual, dtype=float)-np.asarray(target, dtype=float)
        current_poisson, current_local = self._newton_electrostatic_residuals(state)
        result = np.asarray(residual, dtype=float).copy()
        start, stop = len(storage_scale), len(storage_scale)+len(poisson_scale)
        result[start:stop] = ((current_poisson-before_poisson)/DD(poisson_scale)).to_float()
        for k in range(self.interface_count):
            rows = slice(6*k, 6*k+2)
            difference = current_local[2*k:2*k+2]-before_local[2*k:2*k+2]
            result[stop+rows.start:stop+rows.stop] = (difference/DD(local_scale[rows])).to_float()
        return result

    def independent_poisson_inputs(self, fixed_inputs, voltage, *, seed_phi=None):
        """Copy fixed inputs and preparation anchors into the isolated DTO."""
        from .one_dimensional_mechanism_r1_state import canonical
        from perovskite_sim.solver.mol import poisson_right_boundary
        if set(fixed_inputs) != set(_INDEPENDENT_FIXED_FIELDS):
            raise ValueError("independent input adapter rejects direct outputs")
        f = fixed_inputs
        if seed_phi is None:
            old = self.system.evaluate_quasi_fermi_increments_defect_ion_combined(
                f["dqfn_V"].hi, f["dqfp_V"].hi, 0.,
                positive_ion_density_m3=f["positive_m3"].hi,
                negative_ion_density_m3=None, dynamic_interface_occupancy=f["occupancy"].hi,
                V_app=float(voltage))
            seed_phi = old.phi
            provenance = "independent_legacy_fixed_qf_evaluation"
        else:
            provenance = "same_diagnostics_call_legacy_potential_eliminated_channel"
        preparation = {
            "phi0_V": self.system.phi0, "log_n0": self.system.log_n0,
            "log_p0": self.system.log_p0,
            "contact_n_m3": self.reference_n[[0, -1]],
            "contact_p_m3": self.reference_p[[0, -1]],
            "thermal_voltage_V": self.thermal_voltage,
            "poisson_capacitance_F_m2": self.material.poisson_factor.C,
            "poisson_width_m": self.material.poisson_factor.h_cell,
            "N_D_m3": self.material.N_D, "N_A_m3": self.material.N_A,
            "ion_background_m3": self.material.P_ion0,
            "interface_nodes": list(zip(self.left_nodes, self.right_nodes)),
            "sheet_weights": [self._sheet_weights(k) for k in range(self.interface_count)],
            "boundary_phi_V": [0., poisson_right_boundary(self.material, float(voltage))],
        }
        return IndependentPoissonInputs(
            canonical({key: pair_words(value) for key, value in f.items()}),
            canonical(preparation), canonical(pair_words(DD(seed_phi))), provenance)

    def _independent_eliminated_state(self, inputs):
        # Compatibility name, with the same isolated-input contract as the
        # production pure function. No direct state is accepted here.
        return solve_independent_poisson(inputs)

    def _independent_eliminated_phi(self, state, voltage):
        # Historical convenience wrapper; production diagnostics never passes
        # the complete state to the independent constraint solver.
        inputs = self.independent_poisson_inputs(
            {key: state.fine[key] for key in _INDEPENDENT_FIXED_FIELDS}, voltage)
        fields, _ = solve_independent_poisson(inputs)
        return fields["phi_V"]

    def _eliminated_flux_potential(self, phi):
        """Named pre-constitutive hook for the frozen single-side path fault."""
        return phi

    def eliminated_operator_diagnostics(self, state, voltage):
        values = super().eliminated_operator_diagnostics(state, voltage)
        if not isinstance(state, PrecisionState):
            return values
        inputs = self.independent_poisson_inputs(
            {key: state.fine[key] for key in _INDEPENDENT_FIXED_FIELDS},
            voltage, seed_phi=values["potential"]["eliminated"])
        eliminated, solve = solve_independent_poisson(inputs)
        constraint_phi = eliminated["phi_V"]
        phi = self._eliminated_flux_potential(constraint_phi)
        flux = (ion_flux_pair(phi,state.fine["positive_m3"],self.material,np.diff(self.grid),fault=self.precision_fault)
                if self.controls.nu_I else DD(np.zeros(self.node_count-1)))
        boundary = self._ion_boundary_flux()
        rate = -diff(cat(boundary[0],flux,boundary[1]))/DD(self.widths)
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
        exported = {**eliminated, "phi_V": phi,
            "constraint_phi_V": constraint_phi,
            "dqfn_V": state.fine["dqfn_V"], "dqfp_V": state.fine["dqfp_V"],
            "positive_m3": state.fine["positive_m3"], "occupancy": state.fine["occupancy"],
            "sheet_charge_C_m2": state.fine["sheet_charge_C_m2"],
            "positive_flux_m2_s": flux, "positive_rate_m3_s": rate,
            "boundary_flux_m2_s": boundary,
        }
        from .one_dimensional_mechanism_r1_state import digest
        ion_inputs = {
            "phi_V": pair_words(phi), "positive_m3": pair_words(state.fine["positive_m3"]),
            "boundary_flux_m2_s": pair_words(boundary),
            "spacing_m": np.diff(self.grid).tolist(),
            "D_ion_face_m2_s": self.material.D_ion_face.tolist(),
            "P_lim_node_m3": self.material.P_lim_node.tolist(),
            "thermal_voltage_V": float(self.material.V_T_device),
            "nu_I": int(self.controls.nu_I), "fault": self.precision_fault,
        }
        solve["ion_input_digest"] = digest(ion_inputs)
        identity = fine_identity(state.fine)
        evidence = {
            "schema": "R1EliminatedPrecisionV1", "state_identity": identity,
            "fields": {k: pair_words(v) for k,v in exported.items()}, "solve": solve,
            "shared_inputs": {
                "input_state_identity": identity,
                "fields": ["dqfn_V","dqfp_V","positive_m3","occupancy","sheet_charge_C_m2"],
                "source": "actual_direct_state_fixed_physical_inputs_not_independently_solved",
            },
            "potential_roles": {"phi_V":"actual_ion_constitutive_input",
                                "constraint_phi_V":"independent_poisson_solution_for_n_and_p"},
            "trace_state_scope": "not_claimed_as_an_independent_fine_trace_carrier_solve",
        }
        if getattr(getattr(self, "_r1_backend", None), "is_pair", False):
            evidence["schema"] = "R1EliminatedPrecisionV2"
            evidence["shared_inputs"]["fields"] = json.loads(inputs.fixed_inputs_json)
            evidence["solve_inputs"] = inputs.to_dict()
            evidence["ion_inputs"] = ion_inputs
        return EliminatedDiagnostics(values,evidence)


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


def capture_initial_local_context(system, state, voltage):
    """Record actual local-solve inputs before rebasing the initial state."""
    context = system.precision_arithmetic_context()
    context["trace_state_arithmetic"] = {
        "kind": "recorded_log_update",
        "reference_m3": pair_words(system._fine_reference["trace_state_m3"]),
        "log_increment": np.asarray([state.coordinate[system._local_block_slice(k)][2:]
                                      for k in range(system.interface_count)]).tolist(),
        "inputs_derived_from_final_state": False,
        "scope": "actual_reference_and_coordinate_before_evaluation",
    }
    context["actual_update"] = {
        "reference_fields": {k: pair_words(system._fine_reference[k]) for k in REFERENCE_FIELDS},
        "coordinate": state.coordinate.tolist(),
        "voltage_lift_V": pair_words(system._fine_lift),
        "trace_voltage_lift_V": pair_words(system._fine_trace_lift),
        "contact_boundary_inputs": {
            "voltage_V": float(voltage),
            "built_in_voltage_V": float(system.material.V_bi_bc),
            "junction_polarity": int(system.material.junction_polarity),
        },
        "contact_evaluation": {
            "source": "original_binary64_coordinates_before_fine_endpoint_insertion",
            "inputs_derived_from_final_state": False,
            **{k: v.tolist() for k, v in system._precision_contact_evaluation.items()},
        },
        "coordinate_indices": {
            "potential": list(range(*system.potential_slice.indices(system.dimension))),
            "electron": list(range(*system.electron_slice.indices(system.dimension))),
            "hole": list(range(*system.hole_slice.indices(system.dimension))),
            "trap": list(range(*system.trap_slice.indices(system.dimension))),
            "positive": list(range(*system.positive_slice.indices(system.dimension))),
            "local": [list(range(*system._local_block_slice(k).indices(system.dimension)))
                      for k in range(system.interface_count)],
        },
        "qf_rule": "qn=reference_qn+VT*z_e-lift; qp=reference_qp+VT*z_h+lift",
        "potential_rule": "phi=reference_phi+VT*z_phi+lift; endpoints=original_contact_rule",
        "fine_state_identity": fine_identity(state.fine),
    }
    system._precision_zero_plus_context = context


def finalize_initial_precision(system, before, result, *, snapshot):
    context = system.precision_arithmetic_context()
    context["zero_minus_fine"] = {k: pair_words(v) for k, v in before.fine.items()}
    context["zero_minus_state"] = snapshot(system, before)
    context["zero_plus_state"] = snapshot(result.system, result.zero_plus)
    context["zero_minus_state_identity"] = fine_identity(before.fine)
    context["zero_plus_state_identity"] = fine_identity(result.zero_plus.fine)
    context["zero_plus_context"] = result.system._precision_zero_plus_context
    return replace(result, event={**result.event, "precision_arithmetic_context": context})


def independent_currents_pair(system, state):
    from .one_dimensional_mechanism_r1_independent_physics import _interface_balance
    locals_new = []
    for k, item in enumerate(state.local):
        balance = _interface_balance(system, state, k)
        locals_new.append(replace(item, tangent=replace(item.tangent, balance=balance)))
    _, _, electron, hole = carrier_currents_pair(system, state.fine, locals_new)
    ion = (ion_flux_pair(state.fine["phi_V"], state.fine["positive_m3"], system.material,
                         np.diff(system.grid)) if system.controls.nu_I
           else DD(np.zeros(system.node_count - 1)))
    interface = np.empty((system.interface_count, 2))
    for k, face in enumerate(system.interface_faces):
        flux = locals_new[k].tangent.balance.bulk_flux_m2_s
        interface[k] = Q * np.array([-flux[0] + flux[1], flux[2] - flux[3]]) + Q * ion.hi[face]
    return electron.hi.copy(), hole.hi.copy(), ion.hi.copy(), interface


def independent_increments_pair(system, state, previous):
    f, p = state.fine, previous.fine
    rho = DD(Q) * ((f["p_m3"] - p["p_m3"]) - (f["n_m3"] - p["n_m3"])
                   + (f["positive_m3"] - p["positive_m3"]))
    occupied = DD(system.trap_density) * (f["occupancy"] - p["occupancy"])
    return tuple(v.hi.copy() for v in (rho, occupied, f["phi_V"] - p["phi_V"],
                                       f["trace_potential_V"] - p["trace_potential_V"]))


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
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_step as initial_step
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
    original_initial=initial_step.build_initial_step
    original_local_solve=initial_step._solve_local_carriers
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
    def local_solve(system,*args,**kwargs):
        result=original_local_solve(system,*args,**kwargs)
        if isinstance(system,CompensatedR1System):
            capture_initial_local_context(system,result[0],args[0] if args else kwargs["voltage"])
        return result
    def build_initial(system,before,*args,**kwargs):
        result=original_initial(system,before,*args,**kwargs)
        if not isinstance(system,CompensatedR1System):
            return result
        return finalize_initial_precision(system,before,result,snapshot=snapshot)
    def independent_currents(system,state):
        if not isinstance(state,PrecisionState):return original_currents(system,state)
        return independent_currents_pair(system,state)
    def independent_increments(system,state,previous):
        if not isinstance(state,PrecisionState):return original_increments(system,state,previous)
        return independent_increments_pair(system,state,previous)
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
        if getattr(module,"build_initial_step",None) is original_initial:
            patch(module,"build_initial_step",build_initial)
    patch(independent,"_currents",independent_currents)
    patch(independent,"_increments",independent_increments)
    patch(independent,"SHARED_CONSTITUTIVE_DEPENDENCIES",SHARED_CONSTITUTIVE_DEPENDENCIES)
    patch(transient,"_solve_step",lambda *a,**kw:_precision_solve_step(original_solve,*a,**kw))
    patch(initial_step,"_solve_local_carriers",local_solve)
    _ACTIVE=True
    try:
        yield
    finally:
        for module,name,old in reversed(changes):
            setattr(module,name,old)
        _ACTIVE=False

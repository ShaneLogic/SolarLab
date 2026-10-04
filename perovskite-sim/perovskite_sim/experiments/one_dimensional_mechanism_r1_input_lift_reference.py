"""Isolated fixed-QF ion reference in the rebased carrier coordinates.

The accepted coordinate origin defines the carrier mapping before a live trial
exists. It is neither a global cold-start mapping nor a current trial output.
Only four current fixed inputs cross this boundary; an independently eliminated
legacy potential supplies a seed, followed by an absolute DD Poisson solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from perovskite_sim.constants import Q
from perovskite_sim.physics.compensated import DD
from perovskite_sim.solver.mol import poisson_right_boundary

__all__ = ["RebasedIonReferenceInputs", "make_inputs", "solve_rebased_ion_reference"]

_FIXED = ("dqfn_V", "dqfp_V", "positive_m3", "occupancy")
_ORIGIN = ("phi_V", "n_m3", "p_m3", "dqfn_V", "dqfp_V")
_ORIGIN_IDENTITY_FIELDS = _ORIGIN + (
    "positive_m3", "occupancy", "trace_potential_V", "trace_state_m3",
)
_COEFFICIENTS = (
    "thermal_voltage_V", "ion_thermal_voltage_V", "charge_C",
    "poisson_capacitance_F_m2", "poisson_width_m", "N_D_m3", "N_A_m3",
    "ion_background_m3", "trap_density_m2", "equilibrium_occupancy",
    "interface_nodes", "sheet_weights", "static_sheet_charge_C_m2", "trace_jump_V",
    "boundary_phi_V", "grid_spacing_m",
    "control_volume_width_m", "ion_capacity_m3", "ion_diffusion_m2_s",
    "positive_nodes", "ion_flux_enabled", "site_occupancy_ceiling", "model",
)
_MODEL = "one_positive_ion_steric_diffusion_only"
_SEED_SOURCE = "same_diagnostics_call_legacy_potential_eliminated_channel"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _words(value):
    return {"hi": value.hi.tolist(), "lo": value.lo.tolist()}


def _json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("reference JSON contains duplicate keys")
            result[key] = value
        return result
    if not isinstance(text, str):
        raise TypeError("reference DTO owns canonical JSON strings")
    return json.loads(text, object_pairs_hook=unique)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"reference {name} field coverage mismatch")


def _array(value, shape, name):
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"reference {name} must contain finite real numbers")
    # JSON's empty list has no trailing dimensions; restore a known empty shape.
    if result.size == 0 and np.prod(shape) == 0:
        result = result.reshape(shape)
    if result.shape != shape:
        raise ValueError(f"reference {name} must have shape {shape}")
    return result.astype(float)


def _pair(value, shape, name):
    _keys(value, ("hi", "lo"), name)
    return DD(_array(value["hi"], shape, name), _array(value["lo"], shape, name))


def _put(value, index, part):
    part = part if isinstance(part, DD) else DD(part)
    hi, lo = value.hi.copy(), value.lo.copy()
    hi[index], lo[index] = part.hi, part.lo
    return DD(hi, lo)


def _positive(value, name, *, zero=False):
    if np.any(value.hi < 0 if zero else value.hi <= 0):
        raise ValueError(f"reference {name} must be {'nonnegative' if zero else 'positive'}")


def _decode(payload):
    f, o, a = payload["fixed_inputs"], payload["coordinate_origin"], payload["coefficients"]
    _keys(f, _FIXED, "fixed inputs")
    _keys(o, _ORIGIN, "coordinate origin")
    _keys(a, _COEFFICIENTS, "coefficients")
    _keys(o["phi_V"], ("hi", "lo"), "origin potential")
    shape = np.asarray(o["phi_V"]["hi"]).shape
    if len(shape) != 1 or shape[0] < 3:
        raise ValueError("reference requires at least three potential nodes")
    count = shape[0]
    interface_count = len(a["trap_density_m2"])
    origin = {key: _pair(value, shape, key) for key, value in o.items()}
    fixed = {key: _pair(value, (interface_count,) if key == "occupancy" else shape, key)
             for key, value in f.items()}
    seed = _pair(payload["seed_phi_V"], shape, "independent seed potential")
    shapes = {
        "thermal_voltage_V": (), "ion_thermal_voltage_V": (), "charge_C": (),
        "poisson_capacitance_F_m2": (count-1,), "poisson_width_m": (count-2,),
        "N_D_m3": shape, "N_A_m3": shape, "ion_background_m3": shape,
        "trap_density_m2": (interface_count,), "equilibrium_occupancy": (interface_count,),
        "static_sheet_charge_C_m2": (interface_count,), "trace_jump_V": (interface_count,),
        "sheet_weights": (interface_count, 2), "boundary_phi_V": (2,),
        "grid_spacing_m": (count-1,), "control_volume_width_m": shape,
        "ion_capacity_m3": shape, "ion_diffusion_m2_s": (count-1,),
        "site_occupancy_ceiling": (),
    }
    coefficients = {key: DD(_array(a[key], expected, key)) for key, expected in shapes.items()}
    if any(np.any(coefficients[key].hi != 0) for key in ("static_sheet_charge_C_m2", "trace_jump_V")):
        raise ValueError("reference requires zero static sheet charge and zero prescribed trace jump")
    for key in ("thermal_voltage_V", "ion_thermal_voltage_V", "charge_C",
                "poisson_capacitance_F_m2", "poisson_width_m", "grid_spacing_m",
                "control_volume_width_m", "ion_capacity_m3", "site_occupancy_ceiling"):
        _positive(coefficients[key], key)
    for key in ("N_D_m3", "N_A_m3", "ion_background_m3", "trap_density_m2",
                "sheet_weights", "ion_diffusion_m2_s"):
        _positive(coefficients[key], key, zero=True)
    for key in ("n_m3", "p_m3"):
        _positive(origin[key], key)
    _positive(fixed["positive_m3"], "positive ion population", zero=True)
    for name, value in (("occupancy", fixed["occupancy"]),
                        ("equilibrium occupancy", coefficients["equilibrium_occupancy"])):
        if np.any(value.hi <= 0) or np.any((DD(1)-value).hi <= 0):
            raise ValueError(f"reference {name} must lie strictly inside (0, 1)")
    indices = _array(a["interface_nodes"], (interface_count, 2), "interface nodes")
    if (np.any(indices != np.floor(indices)) or np.any(indices < 1)
            or np.any(indices >= count-1) or np.any(indices[:, 1] != indices[:, 0]+1)):
        raise ValueError("reference interfaces must connect adjacent interior nodes")
    active = np.asarray(a["positive_nodes"])
    active = _array(active, (active.size,), "positive nodes")
    if (np.any(active != np.floor(active)) or np.any(active < 0) or np.any(active >= count)
            or len(np.unique(active)) != len(active)):
        raise ValueError("reference positive node indices are invalid")
    _positive(fixed["positive_m3"][active.astype(int)], "active positive ion population")
    ceiling = coefficients["site_occupancy_ceiling"]
    if (DD(.999)-ceiling).hi < 0:
        raise ValueError("reference site ceiling cannot exceed the original .999 bound")
    if np.any((ceiling-fixed["positive_m3"]/coefficients["ion_capacity_m3"]).hi <= 0):
        raise ValueError("reference ion population reached the original pre-clipping bound")
    if type(a["ion_flux_enabled"]) is not bool or a["model"] != _MODEL:
        raise ValueError("reference requires the frozen positive-ion steric model and flux switch")
    return fixed, origin, coefficients, seed, indices.astype(int), a["ion_flux_enabled"]


@dataclass(frozen=True, slots=True)
class RebasedIonReferenceInputs:
    """Immutable allowlisted copies, with no live state or derived-field alias."""

    fixed_inputs_json: str
    coordinate_origin_json: str
    coefficients_json: str
    seed_phi_json: str
    seed_provenance: str = _SEED_SOURCE

    def __post_init__(self):
        if self.seed_provenance != _SEED_SOURCE:
            raise ValueError("reference seed must be the independent legacy eliminated potential")
        for name in ("fixed_inputs_json", "coordinate_origin_json", "coefficients_json", "seed_phi_json"):
            object.__setattr__(self, name, _canonical(_json(getattr(self, name))))
        _decode(self.to_dict())

    def to_dict(self):
        return {"schema": "R1RebasedIonReferenceInputsV1",
                "fixed_inputs": _json(self.fixed_inputs_json),
                "coordinate_origin": _json(self.coordinate_origin_json),
                "coefficients": _json(self.coefficients_json),
                "seed_phi_V": _json(self.seed_phi_json), "seed_provenance": self.seed_provenance}


def make_inputs(system, fixed_inputs, seed_phi, voltage):
    """Freeze material data and the pre-trial coordinate origin, never a state.

    ``seed_phi`` must come from the same diagnostics call's independent legacy
    potential-eliminated channel. It has no role in defining the carrier map.
    """
    if set(fixed_inputs) != set(_FIXED):
        raise ValueError("reference fixed inputs reject current direct outputs")
    if any(not isinstance(fixed_inputs[key], DD) for key in _FIXED):
        raise TypeError("reference fixed inputs must retain full DD words")
    origin = system._input_lift_reference
    if (set(origin) != set(_ORIGIN_IDENTITY_FIELDS)
            or any(not isinstance(origin.get(key), DD) for key in _ORIGIN_IDENTITY_FIELDS)):
        raise TypeError("reference requires a frozen DD coordinate origin")
    if _digest({key: _words(origin[key]) for key in _ORIGIN_IDENTITY_FIELDS}) != getattr(
            system, "_input_lift_reference_identity", None):
        raise ValueError("reference coordinate origin identity is stale or unavailable")
    mat, source = system.material, system.system.source_mat
    if mat.has_dual_ions or not mat.ion_steric_diffusion_only or np.size(system.negative_nodes):
        raise ValueError("reference supports only the frozen positive-ion steric model")
    params = source.carrier_params
    if (params.get("carrier_statistics", "maxwell_boltzmann") != "maxwell_boltzmann"
            or params.get("degenerate_recombination_model", "maxwell_boltzmann")
            not in ("maxwell_boltzmann", "off")
            or getattr(source, "has_selective_contacts", False)
            or getattr(source, "has_radiative_reabsorption", False)
            or params.get("het_recomb_despike", 0.) != 0.):
        raise ValueError("reference requires the supported Maxwell-Boltzmann pinned-contact model")
    if any(getattr(owner, key, None) is not None for owner in (mat, source) for key in
           ("monovalent_bulk_defects", "multivalent_bulk_defects", "frozen_metastable_defects")):
        raise ValueError("reference does not support additional charged bulk defects")
    # The material problem builder exposes prescribed geometry. Its generated
    # bulk/transport data are discarded; no local solve or current state enters.
    from . import one_dimensional_mechanism_r1_local_carrier as local_carrier
    geometry = [local_carrier._material_two_sided_interface_problem(
        mat, system.stack, origin["n_m3"].hi, origin["p_m3"].hi, origin["phi_V"].hi,
        k, cross_transmission=system.dark_reference.interface_transmission)[0]
        for k in range(system.interface_count)]
    if any(g.fixed_sheet_charge_C_m2 != 0. or g.potential_jump_right_minus_left_V != 0.
           for g in geometry):
        raise ValueError("reference requires zero static sheet charge and zero prescribed trace jump")
    voltage = float(_array(voltage, (), "applied voltage"))
    switch = system.controls.nu_I
    if isinstance(switch, (float, np.floating)) or switch not in (0, 1):
        raise ValueError("reference ion dynamics switch must be zero or one")
    a = {
        "thermal_voltage_V": system.thermal_voltage, "ion_thermal_voltage_V": mat.V_T_device,
        "charge_C": Q, "poisson_capacitance_F_m2": mat.poisson_factor.C,
        "poisson_width_m": mat.poisson_factor.h_cell, "N_D_m3": mat.N_D, "N_A_m3": mat.N_A,
        "ion_background_m3": mat.P_ion0, "trap_density_m2": system.trap_density,
        "equilibrium_occupancy": system.equilibrium_occupancy,
        "interface_nodes": list(zip(system.left_nodes, system.right_nodes)),
        "sheet_weights": [system._sheet_weights(k) for k in range(system.interface_count)],
        "static_sheet_charge_C_m2": [g.fixed_sheet_charge_C_m2 for g in geometry],
        "trace_jump_V": [g.potential_jump_right_minus_left_V for g in geometry],
        "boundary_phi_V": [0., poisson_right_boundary(mat, voltage)],
        "grid_spacing_m": np.diff(system.grid), "control_volume_width_m": system.widths,
        "ion_capacity_m3": mat.P_lim_node, "ion_diffusion_m2_s": mat.D_ion_face,
        "positive_nodes": system.positive_nodes, "site_occupancy_ceiling": system.site_occupancy_ceiling,
    }
    a = {key: np.asarray(value).tolist() for key, value in a.items()}
    a.update(ion_flux_enabled=bool(switch), model=_MODEL)
    return RebasedIonReferenceInputs(
        _canonical({key: _words(fixed_inputs[key]) for key in _FIXED}),
        _canonical({key: _words(origin[key]) for key in _ORIGIN}),
        _canonical(a), _canonical(_words(DD(seed_phi))))


def _ion_flux(phi, population, coefficients, enabled):
    """Independent SG formula, evaluated only on nonzero diffusion faces."""
    diffusion = coefficients["ion_diffusion_m2_s"]
    result = DD(np.zeros(diffusion.shape))
    active = diffusion.hi > 0
    if not enabled or not np.any(active):
        return result
    chemical = -(DD(1)-population/coefficients["ion_capacity_m3"]).log()
    xi = ((phi[1:][active]-phi[:-1][active])/coefficients["ion_thermal_voltage_V"]
          + chemical[1:][active]-chemical[:-1][active])
    magnitude = abs(xi)
    forward = DD(np.ones(magnitude.shape))
    nonzero = magnitude.hi != 0
    if np.any(nonzero):
        forward = _put(forward, nonzero, magnitude[nonzero]/magnitude[nonzero].expm1())
    backward = forward+magnitude
    positive = xi.hi >= 0
    b_plus = DD(np.where(positive, forward.hi, backward.hi),
                np.where(positive, forward.lo, backward.lo))
    b_minus = DD(np.where(positive, backward.hi, forward.hi),
                 np.where(positive, backward.lo, forward.lo))
    flux = diffusion[active]/coefficients["grid_spacing_m"][active]*(
        b_plus*population[:-1][active]-b_minus*population[1:][active])
    return _put(result, active, flux)


def solve_rebased_ion_reference(inputs):
    """Solve absolute Poisson and evaluate ion transport using only the DTO."""
    if not isinstance(inputs, RebasedIonReferenceInputs):
        raise TypeError("reference solve requires RebasedIonReferenceInputs")
    payload = inputs.to_dict()
    fixed, origin, a, seed, interfaces, enabled = _decode(payload)
    phi = _put(seed, [0, -1], a["boundary_phi_V"])
    c, widths, vt, charge = (a[key] for key in
                            ("poisson_capacitance_F_m2", "poisson_width_m", "thermal_voltage_V", "charge_C"))
    sheet = -charge*a["trap_density_m2"]*(fixed["occupancy"]-a["equilibrium_occupancy"])
    laplacian = sparse.diags((c.hi[1:-1], -(c.hi[:-1]+c.hi[1:]), c.hi[1:-1]),
                            (-1, 0, 1), format="csr")

    def populations(potential):
        shift = potential-origin["phi_V"]
        n = origin["n_m3"]*((fixed["dqfn_V"]-origin["dqfn_V"]+shift)/vt).exp()
        p = origin["p_m3"]*((fixed["dqfp_V"]-origin["dqfp_V"]-shift)/vt).exp()
        n = _put(n, [0, -1], origin["n_m3"][[0, -1]])
        p = _put(p, [0, -1], origin["p_m3"][[0, -1]])
        _positive(n, "solved electron population")
        _positive(p, "solved hole population")
        return n, p

    def poisson(potential, n, p):
        rho = charge*(p-n+a["N_D_m3"]-a["N_A_m3"]+fixed["positive_m3"]-a["ion_background_m3"])
        displacement = c*(potential[1:]-potential[:-1])
        residual = displacement[1:]-displacement[:-1]+rho[1:-1]*widths
        for k, nodes in enumerate(interfaces):
            for side, node in enumerate(nodes):
                residual = _put(residual, node-1, residual[node-1]+a["sheet_weights"][k, side]*sheet[k])
        return residual

    for iteration in range(1, 13):
        n, p = populations(phi)
        residual = poisson(phi, n, p)
        jacobian = laplacian-sparse.diags(
            charge.hi*(n.hi[1:-1]+p.hi[1:-1])*widths.hi/vt.hi)
        correction = np.asarray(spsolve(jacobian.tocsr(), -residual.hi))
        correction = _array(correction, (phi.size-2,), "Poisson correction")
        phi = _put(phi, slice(1, -1), phi[1:-1]+DD(correction))
        maximum = float(np.max(np.abs(correction), initial=0.))
        if maximum < 1e-28:
            break
    else:
        raise RuntimeError("independent rebased Poisson did not converge in 12 corrections")

    n, p = populations(phi)
    residual = poisson(phi, n, p)
    flux = _ion_flux(phi, fixed["positive_m3"], a, enabled)
    bounded_flux = _put(DD(np.zeros(phi.size+1)), slice(1, -1), flux)
    rate = -(bounded_flux[1:]-bounded_flux[:-1])/a["control_volume_width_m"]
    fields = {"phi_V": phi, "n_m3": n, "p_m3": p, "poisson_residual_C_m2": residual,
              "positive_flux_m2_s": flux, "positive_rate_m3_s": rate}
    words = {key: _words(value) for key, value in fields.items()}
    return fields, {
        "schema": "R1RebasedIonReferenceSolveV1", "kind": "independent_rebased_fixed_qf_poisson",
        "converged": True, "iterations": iteration, "maximum_corrections": 12,
        "correction_tolerance_V": 1e-28, "last_correction_max_abs_V": maximum,
        "poisson_residual_max_abs_C_m2": float(np.max(np.abs(residual.to_float()), initial=0.)),
        "poisson_residual_scope": "absolute_equation_reassembled_after_last_correction",
        "fixed_input_digest": _digest(payload["fixed_inputs"]),
        "coordinate_origin_digest": _digest(payload["coordinate_origin"]),
        "solve_inputs_digest": _digest(payload), "solved_fields_digest": _digest(words),
        "solve_inputs": payload, "fields": words, "sheet_charge_C_m2": _words(sheet),
        "boundary_flux_m2_s": _words(DD([0., 0.])), "seed_provenance": inputs.seed_provenance,
        "coordinate_origin_source": "system._input_lift_reference_frozen_before_current_trial",
        "coordinate_origin_role": "rebased_carrier_mapping_not_global_cold_start",
        "fixed_input_source": "current_full_DD_dqfn_dqfp_positive_occupancy_only",
        "used_direct_phi": False, "used_direct_carriers": False,
        "used_direct_flux_or_rate": False, "used_direct_sheet_charge": False,
        "used_historical_residual": False, "used_current_trial_outputs": False,
        "ion_constitutive_implementation": "independent_DD_SG_and_zero_boundary_FV_divergence",
    }

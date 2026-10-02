"""Explicit pair consumers for the controlled DC and response equations.

The sparse Jacobians and linear solves retain their original binary64
contract. Persistent physical states, inventory sums, physical observations
and subtraction of pair-valued finite-difference operands retain both words
until the resulting coefficient is explicitly rounded for the sparse solve.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.compensated import DD, sum as dd_sum
from perovskite_sim.physics.recombination import total_recombination
from .one_dimensional_mechanism_r1_pair_codec import FINE_FIELDS, validate_snapshot
from .one_dimensional_mechanism_r1_precision import PrecisionState, cat, diff, put, pair_words
from .one_dimensional_mechanism_r1_state import digest


REPRESENTATION = "float64-pair-v1"
SCHEMAS = {
    "R1ControlledDCResponseV1": "R1ControlledDCResponseV2",
    "R1DCConductanceStudyV1": "R1DCConductanceStudyV2",
    "R1DCEndpointAmplitudeStudyV1": "R1DCEndpointAmplitudeStudyV2",
    "R1ControlledSmallSignalV1": "R1ControlledSmallSignalV2",
}


def numerical_contract():
    return {
        "schema": "R1PairResponseNumericsV1",
        "state": "all_seventeen_normalized_high_low_fields",
        "inventory": "pair_weighted_sum_with_original_binary64_inventory_target",
        "observations": "pair_currents_and_displacement_with_original_interface_and_contact_recombination_law",
        "finite_difference": "subtract_pair_operands_before_rounding_coefficients",
        "uncompensated_components": ["interface_capture_and_carrier_balance", "bulk_recombination"],
        "linear_algebra": "original_binary64_sparse_jacobians_and_linear_solve",
        "precision_scope": "pair_state_consumption_not_full_double_double_linear_response",
    }


def require_state(system, state):
    from .one_dimensional_mechanism_r1_backend import backend_for
    selected = backend_for(system)
    if not selected.is_pair or not isinstance(state, PrecisionState) or set(state.fine) != FINE_FIELDS:
        raise ValueError("pair response requires a complete compensated state and explicit matching backend")
    validate_snapshot(selected.snapshot(system, state))
    if state.negative is not None or state.negative_flux is not None:
        raise ValueError("pair response supports the declared single-positive-ion model only")
    return state.fine


def observations(system, state):
    """The original four physical-face observations, retaining their pair words."""
    fine = require_state(system, state)
    mat, widths = system.material, DD(system.widths)
    rho = DD(Q) * (fine["p_m3"] - fine["n_m3"] + DD(mat.N_D) - DD(mat.N_A)
                   + fine["positive_m3"] - DD(mat.P_ion0))
    face_d = -DD(mat.poisson_factor.C) * diff(fine["phi_V"])
    contact_d = cat(face_d[0] - rho[0] * widths[0], face_d[-1] + rho[-1] * widths[-1])
    # The original R1 operator evaluates this separately at pinned contact
    # populations. Keep the constitutive route rather than invent a new law.
    rec = DD(total_recombination(state.n, state.p, mat.ni_sq, mat.tau_n, mat.tau_p,
                                 mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p))
    jn, jp = fine["electron_current_A_m2"].copy(), fine["hole_current_A_m2"].copy()
    contact_n = jn[[0, -1]] + DD(Q) * rec[[0, -1]] * widths[[0, -1]] * DD([-1., 1.])
    contact_p = jp[[0, -1]] + DD(Q) * rec[[0, -1]] * widths[[0, -1]] * DD([1., -1.])
    ji = DD(Q) * fine["positive_flux_m2_s"]
    labels = [f"internal_face_{i}" for i in range(len(jn))]
    for k in reversed(range(system.interface_count)):
        face, left, right = system.interface_faces[k], system.left_nodes[k], system.right_nodes[k]
        flux = state.local[k].tangent.balance.bulk_flux_m2_s
        cl = EPS_0 * mat.eps_r[left] / mat.iface_qss_left_distances_m[k]
        cr = EPS_0 * mat.eps_r[right] / mat.iface_qss_right_distances_m[k]
        jn, jp = put(jn, face, -Q * flux[0]), put(jp, face, Q * flux[1])
        face_d = put(face_d, face, -DD(cl) * (fine["trace_potential_V"][k, 0] - fine["phi_V"][left]))
        jn = cat(jn[:face+1], DD(Q * flux[2]), jn[face+1:])
        jp = cat(jp[:face+1], DD(-Q * flux[3]), jp[face+1:])
        ji = cat(ji[:face+1], ji[face], ji[face+1:])
        face_d = cat(face_d[:face+1], DD(cr) * (fine["trace_potential_V"][k, 1] - fine["phi_V"][right]), face_d[face+1:])
        labels[face:face+1] = [f"interface_{k}_left", f"interface_{k}_right"]
    rows = (cat(contact_n[0], jn, contact_n[-1]), cat(contact_p[0], jp, contact_p[-1]),
            cat(DD(0.), ji, DD(0.)), cat(contact_d[0], face_d, contact_d[-1]))
    values = DD(np.stack([row.hi for row in rows]), np.stack([row.lo for row in rows])) * system.polarity
    return values, ["left_contact", *labels, "right_contact"]


def inventory_rows(system, state):
    fine = require_state(system, state)
    lookup = {int(node): k for k, node in enumerate(system.positive_nodes)}
    rows = []
    for nodes, target in zip(system.ion_layout.positive_components, system.positive_targets):
        nodes = np.asarray(nodes, dtype=int)
        weighted = DD(system.widths[nodes]) * fine["positive_m3"][nodes]
        derivative = np.zeros(system.dimension)
        columns = system.positive_slice.start + np.array([lookup[int(node)] for node in nodes])
        derivative[columns] = (weighted / target).to_float()
        equation = system.positive_slice.start + lookup[int(nodes[-1])]
        rows.append((equation, derivative, float((dd_sum(weighted) / target - 1.).to_float())))
    return rows


def dynamic_rate(system, state):
    """Retain the available pair ionic rate; other original rates are binary64."""
    rate = DD(state.rate)
    return put(rate, system.positive_slice, state.fine["positive_rate_m3_s"][system.positive_nodes])


def equation_vector(system, state):
    return cat(dynamic_rate(system, state), -state.fine["poisson_residual_C_m2"], -DD(state.local_residual))


def state_difference(up, down, name):
    field = {"n": "n_m3", "p": "p_m3", "positive": "positive_m3", "phi": "phi_V",
             "occupancy": "occupancy"}[name]
    return (up.fine[field] - down.fine[field]).to_float()


def dc_precision_evidence(system, state, policy):
    from dataclasses import asdict
    from .one_dimensional_mechanism_r1_backend import backend_for
    values, _ = observations(system, state)
    current = dd_sum(values[:3], axis=0)
    return {
        "representation": REPRESENTATION, "response_numerics": numerical_contract(),
        "response_policy": asdict(policy),
        "precision_observations": pair_words(values),
        "precision_current_A_m2": pair_words(current),
        "consumed_state_sha256": digest(backend_for(system).snapshot(system, state)),
    }


def terminal_current(record):
    words = record["precision_current_A_m2"]
    return DD(words["hi"], words["lo"])[0]


def stamp(record):
    record["schema"] = SCHEMAS[record["schema"]]
    record["representation"] = REPRESENTATION
    record["response_numerics"] = numerical_contract()
    return record


def validate_schema(record, expected=None):
    if (record.get("schema") not in SCHEMAS.values()
            or expected is not None and record.get("schema") != expected
            or record.get("representation") != REPRESENTATION
            or record.get("response_numerics") != numerical_contract()):
        raise ValueError("pair response schema, representation or numerical contract mismatch")
    return record


def tail_arrays(dc, initial, tail):
    """Validate complete saved pairs before comparing any physical low word."""
    from .one_dimensional_mechanism_r1_pair_codec import SNAPSHOT_FIELDS
    validate_schema(dc.evidence, "R1ControlledDCResponseV2")
    initial_state = {key: initial[key] for key in SNAPSHOT_FIELDS if key in initial}
    tail_state = {key: tail[key] for key in SNAPSHOT_FIELDS if key in tail}
    validate_snapshot(initial_state)
    validate_snapshot(tail_state)
    result = {}
    for name in ("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V", "trace_state_m3", "trace_potential_V"):
        def pair(record):
            return DD(record["precision_" + name + "_hi"], record["precision_" + name + "_lo"])
        result[name] = pair(tail_state), pair(initial_state), dc.state.fine[name]
    return result

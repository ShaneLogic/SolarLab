"""Bounded exact-input ion/Poisson investigation; never changes acceptance gates.

Candidate A evaluates the SG electrochemical force without product subtraction.
Candidate B independently projects each stored input onto its fixed-QF Poisson
constraint in Decimal.  Projection is diagnostic: it changes the stored DAE
state and must not be mistaken for a corrected accepted trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy.linalg  # Load both BLAS clients before limiting threads.
from threadpoolctl import threadpool_info, threadpool_limits

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.ion_migration import ion_face_flux, ion_face_flux_jacobian


def dec(value):
    """Preserve Decimal low parts; float inputs mean their exact binary value."""
    return value if isinstance(value, Decimal) else Decimal.from_float(float(value))


def peak(values):
    return max(map(abs, values), default=Decimal(0))


def decimal_sg_flux(phi, density, dx, diffusion, vt, limits, *, precision=80):
    with localcontext() as context:
        context.prec = precision
        one = Decimal(1)
        potentials, populations, bounds = [list(map(dec, x)) for x in (phi, density, limits)]
        if any(p <= 0 or p >= bound for p, bound in zip(populations, bounds)):
            raise ValueError("oracle requires positive, unclipped population")
        crowding = [-(one - p / bound).ln() for p, bound in zip(populations, bounds)]
        result = []
        for j, distance in enumerate(dx):
            x = ((potentials[j + 1] - potentials[j]) / dec(vt)
                 + crowding[j + 1] - crowding[j])
            b = one if x == 0 else x / (x.exp() - one)
            force = (populations[j + 1] / populations[j]).ln() + x
            # Decimal exp()-1 here is resolved at 50/80 digits, independently
            # of the NumPy Bernoulli and the candidate float implementation.
            result.append(dec(diffusion[j]) / dec(distance) * b * populations[j]
                          * (one - force.exp()))
        return result


def balanced_float_flux(phi, density, dx, diffusion, vt, limits):
    """Candidate A: stable local log ratios, fsum force, expm1 SG balance."""
    occupancy = density / limits
    difference = np.diff(density) / density[:-1]
    chemical = np.log1p(np.diff(occupancy) / (1 - occupancy[1:]))
    electric = np.diff(phi) / vt
    xi = electric + chemical
    force = np.asarray([math.fsum((float(a), float(b), math.log1p(float(c))))
                        for a, b, c in zip(electric, chemical, difference)])
    return diffusion / dx * bernoulli(xi) * density[:-1] * (-np.expm1(force))


def divergence(flux, widths):
    faces = [Decimal(0), *map(dec, flux), Decimal(0)]
    return [(faces[j] - faces[j + 1]) / dec(widths[j]) for j in range(len(widths))]


def comparison(first, second):
    a, b = list(map(dec, first)), list(map(dec, second))
    difference = peak(x - y for x, y in zip(a, b))
    scale = max(peak(a), peak(b), Decimal(1))
    return {"absolute": float(difference), "scale": float(scale),
            "relative": float(difference / scale), "passed": difference / scale <= dec(1e-6)}


def solve_tridiagonal(lower, diagonal, upper, rhs):
    diagonal, rhs = list(diagonal), list(rhs)
    for j in range(1, len(diagonal)):
        factor = lower[j - 1] / diagonal[j - 1]
        diagonal[j] -= factor * upper[j - 1]
        rhs[j] -= factor * rhs[j - 1]
    result = [Decimal(0)] * len(rhs)
    result[-1] = rhs[-1] / diagonal[-1]
    for j in range(len(rhs) - 2, -1, -1):
        result[j] = (rhs[j] - upper[j] * result[j + 1]) / diagonal[j]
    return result


def poisson_project(phi, n, p, population, sheet, material, *, precision=80):
    """Solve each side independently using its own represented carrier anchors.

    n(phi)=n_input*exp((phi-phi_input)/VT), and similarly p, define the
    conditional fixed-QF continuation. No missing saved QF low bits are
    inferred. The physical ion densities, fixed-occupancy sheets and boundary
    potentials remain those from that side of the saved comparison.
    """
    with localcontext() as context:
        context.prec = precision
        anchor_phi, anchor_n, anchor_p, ions = [list(map(dec, x))
                                               for x in (phi, n, p, population)]
        potential = list(anchor_phi)
        q, vt = dec(Q), dec(material.V_T_device)
        capacitance = list(map(dec, material.poisson_factor.C))
        width = list(map(dec, material.poisson_factor.h_cell))
        background = [dec(a) - dec(d) + dec(i)
                      for a, d, i in zip(material.N_A, material.N_D, material.P_ion0)]
        sheet_rhs = [Decimal(0)] * (len(potential) - 2)
        for index, (left, right) in enumerate(zip(material.iface_qss_left_nodes,
                                                  material.iface_qss_right_nodes)):
            cl = EPS_0 * material.eps_r[left] / material.iface_qss_left_distances_m[index]
            cr = EPS_0 * material.eps_r[right] / material.iface_qss_right_distances_m[index]
            wl, wr = cl / (cl + cr), cr / (cl + cr)
            sheet_rhs[left - 1] += dec(wl) * dec(sheet[index])
            sheet_rhs[right - 1] += dec(wr) * dec(sheet[index])

        def evaluate():
            carriers_n = [value * ((x - base) / vt).exp()
                          for value, x, base in zip(anchor_n, potential, anchor_phi)]
            carriers_p = [value * ((base - x) / vt).exp()
                          for value, x, base in zip(anchor_p, potential, anchor_phi)]
            rho = [q * (holes - electrons + ion - fixed)
                   for holes, electrons, ion, fixed in zip(carriers_p, carriers_n, ions, background)]
            raw, diagonal = [], []
            for row, j in enumerate(range(1, len(potential) - 1)):
                raw.append(capacitance[j - 1] * (potential[j - 1] - potential[j])
                           + capacitance[j] * (potential[j + 1] - potential[j])
                           + rho[j] * width[row] + sheet_rhs[row])
                diagonal.append(-capacitance[j - 1] - capacitance[j]
                                - q * (carriers_n[j] + carriers_p[j]) / vt * width[row])
            return raw, diagonal

        initial_residual, _ = evaluate()
        for iteration in range(20):
            raw, diagonal = evaluate()
            step = solve_tridiagonal(capacitance[1:-1], diagonal,
                                     capacitance[1:-1], [-x for x in raw])
            potential[1:-1] = [x + dx for x, dx in zip(potential[1:-1], step)]
            if peak(step) < Decimal(10) ** (15 - precision):
                break
        else:
            raise ValueError("bounded Decimal Poisson projection did not converge")
        final_residual, _ = evaluate()
        return {"phi": potential, "initial_residual": initial_residual,
                "final_residual": final_residual, "iterations": iteration + 1,
                "correction": [x - y for x, y in zip(potential, anchor_phi)]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_split_inputs(row, previous, material):
    """Retain exact saved reference/increment pairs for one rebased step.

    The production float representation is re-evaluated first and required to
    match exactly.  This recovers the low part of this saved step, not a new
    higher-precision trajectory accumulated through preceding accepted steps.
    """
    state = row["state"]
    phi, density, n, p = [np.asarray(state[name]) for name in
                          ("phi_V", "positive_m3", "n_m3", "p_m3")]
    if previous is None:
        return {"phi": list(map(dec, phi)), "density": list(map(dec, density)),
                "n": list(map(dec, n)), "p": list(map(dec, p)),
                "checked_saved_float_reconstruction": True, "initial_row": True}
    old = previous["state"]
    old_phi, old_density, old_n, old_p = [np.asarray(old[name]) for name in
                                        ("phi_V", "positive_m3", "n_m3", "p_m3")]
    active_faces = np.flatnonzero(material.D_ion_face > 0)
    nodes = np.unique(np.r_[active_faces, active_faces + 1])
    interior, interfaces = len(phi) - 2, len(material.iface_qss_left_nodes)
    ion_start = 2 * interior + interfaces
    potential_start = ion_start + len(nodes)
    coordinate = np.asarray(row["physics_reconstruction"]["coordinate"])
    if len(coordinate) != potential_start + interior + 6 * interfaces:
        raise ValueError("saved coordinate layout is not the declared single-ion system")
    potential = coordinate[potential_start:potential_start + interior]
    expected_phi = old_phi.copy()
    expected_phi[1:-1] += material.V_T_device * potential
    expected_density = old_density.copy()
    expected_density[nodes] *= np.exp(coordinate[ion_start:potential_start])
    expected_n, expected_p = old_n.copy(), old_p.copy()
    expected_n[1:-1] *= np.exp(coordinate[:interior] + potential)
    expected_p[1:-1] *= np.exp(coordinate[interior:2 * interior] - potential)
    for label, actual, expected in (("phi", phi, expected_phi), ("ion", density, expected_density),
                                    ("n", n, expected_n), ("p", p, expected_p)):
        if not np.array_equal(actual, expected):
            raise ValueError(f"saved one-step {label} reconstruction mismatch at {row['time_s']}")
    precise_phi, precise_density, precise_n, precise_p = [list(map(dec, x))
                                                       for x in (old_phi, old_density, old_n, old_p)]
    for local, node in enumerate(range(1, len(phi) - 1)):
        precise_phi[node] += dec(material.V_T_device) * dec(potential[local])
        precise_n[node] *= (dec(coordinate[local]) + dec(potential[local])).exp()
        precise_p[node] *= (dec(coordinate[interior + local]) - dec(potential[local])).exp()
    for local, node in enumerate(nodes):
        precise_density[node] *= dec(coordinate[ion_start + local]).exp()
    return {"phi": precise_phi, "density": precise_density, "n": precise_n, "p": precise_p,
            "checked_saved_float_reconstruction": True, "initial_row": False}


def retain_saved_increments(row, retained, material):
    """Counterfactual precision shadow using original saved Newton increments.

    This is deliberately separate from exact replay: changing prior low parts
    changes subsequent nonlinear equations, so this sequence is a diagnosis of
    accumulation only, not an accepted trajectory at improved precision.
    """
    if retained is None:
        return {key: list(map(dec, row["state"][name])) for key, name in
                (("phi", "phi_V"), ("density", "positive_m3"), ("n", "n_m3"), ("p", "p_m3"))}
    result = {key: list(values) for key, values in retained.items()}
    active_faces = np.flatnonzero(material.D_ion_face > 0)
    nodes = np.unique(np.r_[active_faces, active_faces + 1])
    interior, interfaces = len(result["phi"]) - 2, len(material.iface_qss_left_nodes)
    ion_start = 2 * interior + interfaces
    potential_start = ion_start + len(nodes)
    coordinate = list(map(dec, row["physics_reconstruction"]["coordinate"]))
    for local, node in enumerate(range(1, len(result["phi"]) - 1)):
        dphi = coordinate[potential_start + local]
        result["phi"][node] += dec(material.V_T_device) * dphi
        result["n"][node] *= (coordinate[local] + dphi).exp()
        result["p"][node] *= (coordinate[interior + local] - dphi).exp()
    for local, node in enumerate(nodes):
        result["density"][node] *= coordinate[ion_start + local].exp()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    diagnosis = json.loads(args.diagnosis.read_text())
    selected = {item["record_index"] for item in diagnosis["local_oracle"]["selected_rows"]}
    row_path = args.attempt / "AcceptedStepsV1.jsonl"
    request = json.loads((args.attempt / "RequestV1.json").read_text())
    manifest = json.loads((args.attempt / "ManifestV1.json").read_text())
    if sha(row_path) != manifest["AcceptedStepsV1.jsonl"]:
        raise ValueError("saved-row manifest mismatch")
    rows = [json.loads(line) for line in row_path.open()]
    stack = load_device_from_yaml(config)
    grid, material = build_r1_material(stack, request["intervals"])
    if not material.ion_steric_diffusion_only or material.has_dual_ions:
        raise ValueError("oracle scope is the frozen single-ion diffusion-only model")
    if any(getattr(material, name) is not None for name in
           ("monovalent_bulk_defects", "frozen_metastable_defects", "multivalent_bulk_defects")):
        raise ValueError("additional bulk charge is outside the conditional oracle")
    dx, widths = np.diff(grid), np.asarray(material.dx_cell)
    limits = np.broadcast_to(material.P_lim_node, grid.shape)
    counts = Counter()
    summary, details = [], []
    with threadpool_limits(1), localcontext() as context:
        context.prec = 90
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("all loaded numerical thread pools must use one thread")
        retained = None
        for index, row in enumerate(rows):
            parts = row["physics_reconstruction"]["eliminated_operator"]
            density = np.asarray(row["state"]["positive_m3"])
            candidate, exact, projection = {}, {}, {}
            split = direct_split_inputs(row, rows[index - 1] if index and row["dt_s"] > 0 else None,
                                        material)
            retained = retain_saved_increments(row, retained if row["dt_s"] > 0 else None, material)
            for side in ("direct", "eliminated"):
                phi = np.asarray(parts["potential"][side])
                actual = ion_face_flux(phi, density, dx, material.D_ion_face, material.V_T_device,
                                       material.P_lim_face, steric_diffusion_only=True,
                                       P_lim_node=material.P_lim_node)
                if not np.array_equal(actual, parts["positive_ion_flux"][side]):
                    raise ValueError(f"represented production flux mismatch at {index}/{side}")
                candidate[side] = balanced_float_flux(phi, density, dx, material.D_ion_face,
                                                       material.V_T_device, limits)
                if index in selected:
                    exact[side] = decimal_sg_flux(phi, density, dx, material.D_ion_face,
                                                 material.V_T_device, limits)
                    n, p = [parts[key][side] for key in ("electron_density", "hole_density")]
                    proj = poisson_project(phi, n, p, density, parts["sheet_charge"][side], material)
                    coarse = poisson_project(phi, n, p, density, parts["sheet_charge"][side],
                                             material, precision=50)
                    proj["precision_change_V"] = peak(a - b for a, b in zip(proj["phi"], coarse["phi"]))
                    proj["flux"] = decimal_sg_flux(proj["phi"], density, dx, material.D_ion_face,
                                                   material.V_T_device, limits)
                    projection[side] = proj
            original = {field: parts["positive_ion_" + field]["relative_error"]
                        for field in ("flux", "rate")}
            tested = {"flux": comparison(candidate["direct"], candidate["eliminated"]),
                      "rate": comparison(divergence(candidate["direct"], widths),
                                         divergence(candidate["eliminated"], widths))}
            for field in ("flux", "rate"):
                counts["original_" + field + "_failures"] += original[field] > 1e-6
                counts["candidate_a_" + field + "_failures"] += not tested[field]["passed"]
            counts["original_any_ion_failure_rows"] += any(value > 1e-6 for value in original.values())
            counts["candidate_a_any_ion_failure_rows"] += any(not value["passed"] for value in tested.values())
            summary.append({"row": index, "time_s": row["time_s"], "substeps": row["substeps"],
                            "original_relative": original, "candidate_a": tested})
            if index not in selected:
                continue
            input_comparison = {"flux": comparison(exact["direct"], exact["eliminated"]),
                                "rate": comparison(divergence(exact["direct"], widths),
                                                   divergence(exact["eliminated"], widths))}
            projected_comparison = {"flux": comparison(projection["direct"]["flux"], projection["eliminated"]["flux"]),
                                    "rate": comparison(divergence(projection["direct"]["flux"], widths),
                                                       divergence(projection["eliminated"]["flux"], widths))}
            split_flux = decimal_sg_flux(split["phi"], split["density"], dx, material.D_ion_face,
                                         material.V_T_device, limits)
            split_eliminated_flux = decimal_sg_flux(parts["potential"]["eliminated"], split["density"],
                                                    dx, material.D_ion_face, material.V_T_device, limits)
            split_comparison = {"flux": comparison(split_flux, split_eliminated_flux),
                                "rate": comparison(divergence(split_flux, widths),
                                                   divergence(split_eliminated_flux, widths))}
            shadow_projection = poisson_project(retained["phi"], retained["n"], retained["p"],
                                                retained["density"], parts["sheet_charge"]["direct"], material)
            side_details = {}
            for side in ("direct", "eliminated"):
                proj = projection[side]
                side_details[side] = {
                    "original_flux_evaluation_error_m2_s": float(peak(dec(a) - b for a, b in zip(parts["positive_ion_flux"][side], exact[side]))),
                    "candidate_a_flux_evaluation_error_m2_s": float(peak(dec(a) - b for a, b in zip(candidate[side], exact[side]))),
                    "fixed_qf_poisson_residual_C_m2": [str(x) for x in proj["initial_residual"]],
                    "max_fixed_qf_poisson_residual_C_m2": float(peak(proj["initial_residual"])),
                    "projection_correction_V": [str(x) for x in proj["correction"]],
                    "max_projection_correction_V": float(peak(proj["correction"])),
                    "max_projected_poisson_residual_C_m2": float(peak(proj["final_residual"])),
                    "projection_50_80_change_V": float(proj["precision_change_V"]),
                    "projected_flux_m2_s": [str(x) for x in proj["flux"]],
                    "same_input_decimal_flux_m2_s": [str(x) for x in exact[side]],
                }
            details.append({"row": index, "time_s": row["time_s"], "substeps": row["substeps"],
                            "same_input_exact_comparison": input_comparison,
                            "candidate_a": tested,
                            "candidate_a_saved_split_inputs": {
                                "reconstructed_float_states_match": split["checked_saved_float_reconstruction"],
                                "max_one_step_phi_assembly_loss_V": float(peak(a - dec(b) for a, b in zip(split["phi"], row["state"]["phi_V"]))),
                                "max_one_step_ion_assembly_loss_m3": float(peak(a - dec(b) for a, b in zip(split["density"], density))),
                                "split_flux_comparison": split_comparison,
                                "direct_split_flux_m2_s": [str(x) for x in split_flux],
                                "shared_split_ion_eliminated_flux_m2_s": [str(x) for x in split_eliminated_flux]},
                            "candidate_a_persistent_precision_shadow": {
                                "accepted_trajectory": False,
                                "scope": "Original saved increments accumulated with retained low parts; no re-solve of changed nonlinear states.",
                                "max_accumulated_potential_difference_V": float(peak(a - dec(b) for a, b in zip(retained["phi"], row["state"]["phi_V"]))),
                                "max_accumulated_ion_difference_m3": float(peak(a - dec(b) for a, b in zip(retained["density"], density))),
                                "max_fixed_qf_poisson_residual_C_m2": float(peak(shadow_projection["initial_residual"])),
                                "max_zero_constraint_correction_V": float(peak(shadow_projection["correction"]))},
                            "candidate_b_independent_zero_constraint_projection": projected_comparison,
                            "saved_incremental_poisson_residual_C_m2": row["state"]["poisson_residual_C_m2"],
                            "sides": side_details})
        identity = {str(path.relative_to(project)): sha(path) for path in
                    (Path(__file__), config, project / "perovskite_sim/physics/ion_migration.py",
                     project / "perovskite_sim/experiments/interface_defect_transient.py",
                     project / "perovskite_sim/experiments/interface_defect_ion_transient.py",
                     project / "perovskite_sim/experiments/quasi_fermi_steady_state.py")}
    report = {
        "schema": "R1V6IonPrecisionProbeV1", "date": "2026-09-20", "source_hashes": identity,
        "source_attempt": str(args.attempt.resolve()), "request": request,
        "source_rows_sha256": sha(row_path), "source_diagnosis_sha256": sha(args.diagnosis),
        "runtime_threadpools": pools, "row_count": len(rows), "counts": dict(counts),
        "selected_row_count": len(details), "all_rows": summary, "selected_rows": details,
        "production_source_changed": False, "acceptance_threshold_changed": False,
        "saved_float_state_reconstruction_passed_rows": len(rows),
        "production_fix_implemented": False, "long_sentinel_gate_restored": False,
        "candidate_a_verdict": "rejected_as_production_fix_no_saved_gate_recovery",
        "candidate_b_verdict": "positive_conditional_projection_evidence_requires_end_to_end_persistent_state_implementation",
        "new_trajectories": 0,
        "candidate_a_scope": "Same represented states, stable SG force evaluation; original 1e-6 comparisons retained.",
        "candidate_b_scope": "Independent high-precision zero-Poisson projections with each side's own saved carrier anchors and sheets. This changes DAE states; not a replay pass or production remedy.",
        "missing_evidence": ["The saved coordinate and prior state reconstruct each float step exactly; retained low parts describe this one step, not a new higher-precision historical trajectory.",
                             "A projected potential is not an accepted coupled time step; no response or full-trajectory error bound is inferred.",
                             "No proposal to share two paths' potentials, residual targets, or fluxes is implemented."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"row_count": len(rows), "counts": dict(counts),
                      "selected": [{"row": r["row"], "exact": r["same_input_exact_comparison"],
                                    "projection": r["candidate_b_independent_zero_constraint_projection"],
                                    "correction": {s: r["sides"][s]["max_projection_correction_V"]
                                                   for s in ("direct", "eliminated")}} for r in details]}, indent=2))


if __name__ == "__main__":
    main()

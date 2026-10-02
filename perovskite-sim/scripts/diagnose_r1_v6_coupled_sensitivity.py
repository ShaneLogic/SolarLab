"""Conditional full-DAE response to a saved ion-operator discrepancy.

Reconstructs recorded states from their exact saved local coordinates.  It
does not integrate a new trajectory or change a production acceptance gate.
The tangent holds the previous accepted state and the baseline residual fixed;
it is not a bound on accumulated trajectory or discretisation error.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_manifest_coverage(manifest):
    required = {"FailureV1.json", "AcceptedStepsV1.jsonl", "CompletionV1.json"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("manifest omits a required sensitivity input")


def validate_recorded_source(recorded, actual):
    if (not recorded.get("files") or recorded["files"] != actual.get("files")
            or recorded.get("study_input") != actual.get("study_input")):
        raise ValueError("saved trajectory and executed equations differ")


def solve_conditional_tangent(jacobian, rate_difference, storage_scale, dt,
                              ion_rows):
    """Solve J dx = dt d(rate)/scale with all algebraic rows retained."""
    import numpy as np
    from scipy.sparse.linalg import spsolve

    difference = np.asarray(rate_difference, dtype=float)
    scales = np.asarray(storage_scale, dtype=float)
    raw_indices = np.asarray(ion_rows)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise ValueError("ion forcing rows must be integer indices")
    indices = raw_indices.astype(int)
    if (not np.isfinite(dt) or dt <= 0 or difference.shape != indices.shape
            or scales.ndim != 1 or np.any(scales <= 0)
            or not np.all(np.isfinite(scales))
            or not np.all(np.isfinite(difference))
            or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= scales.size)
            or jacobian.shape[0] != jacobian.shape[1]
            or jacobian.shape[0] < scales.size):
        raise ValueError("invalid finite-step ion forcing or DAE dimensions")
    rhs = np.zeros(jacobian.shape[0])
    rhs[indices] = dt * difference / scales[indices]
    direction = np.asarray(spsolve(jacobian.tocsc(), rhs))
    if not np.all(np.isfinite(direction)):
        raise ValueError("singular or nonfinite coupled tangent")
    residual = np.asarray(jacobian @ direction - rhs)
    scale = max(float(np.max(np.abs(rhs))),
                float(np.max(np.asarray(abs(jacobian) @ np.abs(direction)))),
                np.finfo(float).tiny)
    return direction, rhs, float(np.max(np.abs(residual))) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+",
                        default=[108, 109, 133, 400, 829, 830, 850, 851, 933])
    args = parser.parse_args()
    project, attempt = args.project.resolve(), args.attempt.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = set(args.rows)
    if not 1 <= len(selected) <= 16 or any(i < 1 for i in selected):
        raise ValueError("select between one and sixteen finite saved rows")
    git = lambda *v: subprocess.check_output(
        ["git", *v], cwd=project, text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain"):
        raise ValueError("source must be the specified clean frozen checkout")
    if digest(attempt / "ManifestV1.json") != args.manifest_sha256:
        raise ValueError("saved-attempt manifest differs from caller anchor")
    manifest = read(attempt / "ManifestV1.json")
    validate_manifest_coverage(manifest)
    for name, expected in manifest.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt) or digest(path) != expected:
            raise ValueError("saved-attempt manifest mismatch: " + name)

    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(project))
    import numpy as np
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import physical_step_record
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        R1PreparedState, _preparation_policy, canonical, snapshot,
        verify_prepared_physics, json_data, execution_source)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    from perovskite_sim.models.config_loader import load_device_from_yaml

    record = read(attempt / "FailureV1.json")["partial_result"]
    actual_source = execution_source()
    validate_recorded_source(record["source"], actual_source)
    rows = [json.loads(line) for line in (attempt / "AcceptedStepsV1.jsonl").open()]
    if rows != record["accepted_steps"] or max(selected) >= len(rows):
        raise ValueError("selected rows disagree with saved result coverage")
    if record["control_label"] != "D":
        raise ValueError("this diagnostic requires the fully coupled D control")
    prepared = R1PreparedState.from_dict(read(args.prepared))
    if prepared.sha256 != record["prepared_sha256"]:
        raise ValueError("saved trajectory and supplied preparation differ")
    policy = _preparation_policy(record["policy"])
    fixture = project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    binding_path = project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"
    stack, binding = load_device_from_yaml(fixture), read(binding_path)
    if binding["sha256"] != record["reference_sha256"]:
        raise ValueError("saved trajectory and fixed reference differ")
    report = {
        "schema": "R1V6ConditionalCoupledSensitivityV1",
        "scope": "saved_finite_step_full_DAE_tangent_previous_state_and_baseline_residual_fixed",
        "source_commit": args.expected_commit,
        "source_project": str(project), "script_sha256": digest(__file__),
        "recorded_source": record["source"], "executed_source": actual_source,
        "recorded_and_executed_equations_identical": True,
        "manifest_sha256": args.manifest_sha256,
        "prepared_sha256": digest(args.prepared),
        "fixture_sha256": digest(fixture), "binding_sha256": digest(binding_path),
        "original_run_status": read(attempt / "CompletionV1.json")["status"],
        "new_trajectory_integrated": False, "acceptance_limits_changed": False,
        "trajectory_error_bound": None,
        "unresolved_terms": ["previous_state_error_and_accumulation",
                             "spatial_and_time_discretisation", "model_uncertainty",
                             "nonlinear_corrected_trajectory", "baseline_residual_correction",
                             "correlated_carrier_algebraic_and_direct_observation_changes"],
        "selected_rows": sorted(selected), "cases": [],
    }
    with threadpool_limits(limits=1, user_api="blas"):
        report["blas"] = threadpool_info()
        if not report["blas"] or any(p["num_threads"] != 1 for p in report["blas"]):
            raise ValueError("BLAS single-thread setting not effective")
        system, before = verify_prepared_physics(prepared, stack, binding, policy=policy)
        initial = build_initial_step(system, before, record["amplitude_V"], policy=policy)
        previous = None
        for index, row in enumerate(rows):
            saved = row["physics_reconstruction"]
            coordinate = np.asarray(saved["coordinate"], dtype=float)
            if row["phase"] == "0+":
                working, local_previous = initial.system, None
                state = replace(initial.zero_plus, coordinate=coordinate.copy())
            else:
                if previous is None:
                    raise ValueError("saved finite row has no previous state")
                working, local_previous = initial.system.rebase(previous)
                working.set_voltage_lift(saved["voltage_V"], local_previous)
                state = working.evaluate(coordinate, saved["voltage_V"])
            if canonical(snapshot(working, state)) != canonical(row["state"]):
                raise ValueError(f"exact saved-coordinate reconstruction mismatch at row {index}")
            if index in selected:
                if local_previous is None or row["dt_s"] <= 0:
                    raise ValueError("selected row is not a finite step")
                dt, voltage = row["dt_s"], saved["voltage_V"]
                ss = np.asarray(saved["storage_scale"])
                ps = np.asarray(saved["poisson_scale_C_m2"])
                ls = np.asarray(saved["local_algebraic_scale"])
                baseline, jacobian, evaluated = working.residual_and_jacobian(
                    coordinate, voltage, local_previous, dt, ss, ps, ls)
                np.testing.assert_array_equal(baseline, saved["scaled_residual_vector"])
                diagnostic = saved["eliminated_operator"]["positive_ion_rate"]
                drate = (np.asarray(diagnostic["eliminated"])
                         - np.asarray(diagnostic["direct"]))[working.positive_nodes]
                start = 2 * working.interior_count + working.interface_count
                ion_rows = np.arange(start, start + len(working.positive_nodes))
                direction, rhs, backward = solve_conditional_tangent(
                    jacobian, drate, ss, dt, ion_rows)
                norm = float(np.max(np.abs(direction)))
                case = {"record_index": index, "time_s": row["time_s"],
                        "dt_s": dt, "substeps": row["substeps"],
                        "original_operator_error": saved["eliminated_operator_error"],
                        "full_DAE_dimension": working.dimension,
                        "scaled_baseline_residual": float(np.max(np.abs(baseline))),
                        "ion_rate_discrepancy_max_m3_s": float(np.max(np.abs(drate))),
                        "maximum_coordinate_tangent": norm,
                        "linear_backward_error": backward, "direction": direction,
                        "previous_state_fixed": True, "finite_difference_probes": []}
                if norm > 0:
                    # Amplify only for derivative measurement. These trial
                    # states are never written as accepted production states.
                    for step in (1e-4, 3e-5, 1e-5, 3e-6):
                        gain = step / norm
                        rp, _, plus = working.residual_and_jacobian(
                            coordinate + gain * direction, voltage, local_previous, dt, ss, ps, ls)
                        rm, _, minus = working.residual_and_jacobian(
                            coordinate - gain * direction, voltage, local_previous, dt, ss, ps, ls)
                        fp = physical_step_record(working, plus, local_previous, dt)
                        fm = physical_step_record(working, minus, local_previous, dt)
                        current = (np.asarray(fp["contact_maxwell_A_m2"])
                                   - np.asarray(fm["contact_maxwell_A_m2"])) / (2 * gain)
                        action = (rp - rm) / (2 * gain)
                        mismatch = float(np.max(np.abs(action - rhs))) / max(
                            float(np.max(np.abs(rhs))), np.finfo(float).tiny)
                        case["finite_difference_probes"].append({
                            "maximum_coordinate_probe": step, "amplification": gain,
                            "conditional_contact_current_change_A_m2": current,
                            "conditional_potential_change_V": (plus.phi - minus.phi) / (2 * gain),
                            "conditional_occupancy_change": (plus.occupancy - minus.occupancy) / (2 * gain),
                            "conditional_ion_density_change_m3": (plus.positive - minus.positive) / (2 * gain),
                            "full_DAE_directional_derivative_relative_error": mismatch,
                        })
                report["cases"].append(case)
                print(json.dumps({"row": index, "time_s": row["time_s"],
                                  "coordinate_tangent": norm, "linear_backward_error": backward}), flush=True)
            previous = state
        report["reconstructed_row_count"] = len(rows)
    if git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain"):
        raise ValueError("source changed while computing sensitivity")
    if actual_source != execution_source():
        raise ValueError("executed equations changed while computing sensitivity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_data(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

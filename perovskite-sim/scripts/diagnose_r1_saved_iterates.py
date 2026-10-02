"""Bounded re-evaluation of four archived R1 failed Newton states, no integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType
from decimal import Decimal, localcontext

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info, threadpool_limits
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    R1PreparedState, verify_prepared_physics, _make_system, snapshot, _preparation_policy, json_data)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
from perovskite_sim.models.config_loader import load_device_from_yaml

CASES = ("D_N256_F0p01_T1", "A_N256_F0p01_T4", "D_N256_F1p0_T2", "B_N256_F1p0_T4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(base):
    """Rebuild accepted prefix using the exact recorded incremental coordinates."""
    result = json.loads((base / "Result.json").read_text())
    failure = json.loads((base / "SolverTrace.json").read_text())["failures"][0]
    prepared = R1PreparedState.from_dict(json.loads((base / "Prepared.json").read_text()))
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((PROJECT / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    policy = _preparation_policy(result["policy"])
    # Deliberate historical physical re-evaluation, never relabel the old
    # preparation's execution identity as this candidate's preparation.
    baseline, before = verify_prepared_physics(prepared, stack, binding, policy=policy)
    controls = R1DynamicsControls.from_label(result["control_label"])
    system = _make_system(stack, baseline.grid, baseline.material, baseline.common_dc_state,
                          binding, controls, policy)
    controlled = system.evaluate(system.initial_coordinate(), 0.)
    for field in ("n", "p", "positive", "occupancy", "phi", "sheet_charge"):
        np.testing.assert_array_equal(getattr(controlled, field), getattr(before, field))
    before = controlled
    initial = build_initial_step(system, before, .005, policy=policy)
    previous = None
    for row in result["accepted_steps"]:
        if row["phase"] == "0+":
            previous = initial.zero_plus
        else:
            working, local_previous = initial.system.rebase(previous)
            working.set_voltage_lift(.005, local_previous)
            previous = working.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]), .005)
        assert json_data(snapshot(initial.system, previous)) == row["state"]
    working, previous = initial.system.rebase(previous)
    working.set_voltage_lift(.005, previous)
    args = (.005, previous, failure["dt_s"], np.asarray(failure["storage_scale"]),
            np.asarray(failure["poisson_scale"]), np.asarray(failure["local_scale"]))
    return working, previous, np.asarray(failure["coordinate"]), args, failure, policy


def trace_density(system, coordinate, mode):
    reference = np.asarray([item.state_m3 for item in system._step_reference.local])
    increment = np.asarray([coordinate[system._local_block_slice(k)][2:] for k in range(system.interface_count)])
    if mode == "binary_exp":
        return reference * np.exp(increment)
    if mode == "binary_expm1":
        return reference + reference * np.expm1(increment)
    with localcontext() as ctx:
        ctx.prec = 80
        return np.asarray([[float(Decimal.from_float(float(r)) * Decimal.from_float(float(z)).exp())
                            for r, z in zip(rr, zz)] for rr, zz in zip(reference, increment)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = {str(path.relative_to(PROJECT)): sha(path) for path in (PROJECT / "perovskite_sim").rglob("*.py")}
    report = {"schema": "R1SavedFailedIterateDiagnosticV4", "scope": "four_saved_iterates_no_time_integration_no_new_acceptance",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "actual_source_sha256": source, "script_sha256": sha(__file__),
        "working_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=PROJECT)).hexdigest(),
        "runtime": {"python": sys.version, "executable": sys.executable, "blas": threadpool_info()}, "cases": []}
    with threadpool_limits(limits=1, user_api="blas"):
        for name in CASES:
            base = args.source / name
            system, previous, coordinate, residual_args, saved, policy = rebuild(base)
            residual, jacobian, state = system.residual_and_jacobian(coordinate, *residual_args)
            np.testing.assert_array_equal(residual, np.asarray(saved["scaled_residual_vector"]))
            case = {"case": name, "input_sha256": {filename: sha(base / filename) for filename in ("Result.json", "Prepared.json", "SolverTrace.json")},
                "accepted_prefix_state_reconstruction_exact": True, "failed_residual_reconstruction_exact": True,
                "original_metrics": independent_physics_row(system, state, previous, saved["dt_s"])["metrics"], "variants": {}}
            for mode in ("binary_exp", "binary_expm1", "decimal80_rounded"):
                system._trace_density_coordinates = MethodType(lambda self, x, mode=mode: trace_density(self, x, mode), system)
                residual, jacobian, state = system.residual_and_jacobian(coordinate, *residual_args)
                direction = np.asarray(spsolve(jacobian, -residual))
                linear = np.asarray(jacobian @ direction + residual)
                samples = []
                for damping in (1., .5, .25, .125, .0625, .03125):
                    trial, _, trial_state = system.residual_and_jacobian(coordinate + damping * direction, *residual_args)
                    metrics = independent_physics_row(system, trial_state, previous, saved["dt_s"])["metrics"]
                    samples.append({"damping": damping, "residual": float(np.max(np.abs(trial))), "metrics": metrics,
                        "meets_required_residual_and_current": bool(np.max(np.abs(trial)) <= policy.maximum_scaled_nonlinear_residual
                            and metrics["contact_internal_current_spread_relative"] <= 2e-6
                            and metrics["interface_current_spread_relative"] <= 2e-6
                            and metrics["charge_balance_normalized"] <= 1e-10)})
                case["variants"][mode] = {"residual": float(np.max(np.abs(residual))),
                    "linear_residual_max": float(np.max(np.abs(linear))), "local_residual": state.local_residual.tolist(),
                    "newton_direction_unchanged_coordinates": int(np.count_nonzero(coordinate + direction == coordinate)),
                    "samples": samples}
            report["cases"].append(case)
            args.output.write_text(json.dumps(json_data(report), indent=2, allow_nan=False) + "\n")
            print(name, {mode: (v["residual"], min(row["residual"] for row in v["samples"]),
                                any(row["meets_required_residual_and_current"] for row in v["samples"]))
                         for mode, v in case["variants"].items()}, flush=True)


if __name__ == "__main__":
    main()

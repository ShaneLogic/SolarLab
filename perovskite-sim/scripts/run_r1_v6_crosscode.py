"""Bounded V6 cross-code diagnostics with an explicit production source.

This is development evidence, not an acceptance runner.  Freeze the contract
before computing, retain all refinement levels, and never infer an error budget
from the observed difference between solvers.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"

PROBES_NM = (25, 50, 75, 130, 150, 175)
TIMES_S = (0., 1e-9, 1e-8, 1e-6, 1e-4)
Q = 1.602176634e-19


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contract():
    return {
        "schema": "R1V6CrossCodeContractV1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_diagnostic_not_scientific_acceptance", "times_s": TIMES_S,
        "probes_nm": PROBES_NM, "current_units": "A/cm^2", "current_frame": "+x",
        "primary_candidates": ["Jion_nm25 at 1e-8 and 1e-6 s", "g_ion_nm25=Jion(1e-6)/Jion(1e-8)"],
        "analytic_control": "Jion_nm50 at 1e-8 s is a parameter/equilibrium check only",
        "electron_candidates": ["Jn_nm50 at 1e-8 s", "Jn_nm130 at 1e-9 and 1e-8 s"],
        "budget_rule": {
            "extrapolation": "none: compare finest actually computed values",
            "each_ladder": "max(abs(fine-medium), abs(medium-coarse)) in the observable's units",
            "combination": "sum of separate R1 space/time/nonlinear and DF space/time/tolerance/ramp/model terms",
            "incomplete_ladder": "missing axes are unknown, never zero; result cannot pass",
            "nonconvergence": "retain observed envelope but mark unestablished bound if refinements grow or oscillate materially",
            "ratio": "evaluate raw ratios on every rung and separately require abs(denominator)>10*its summed budget",
            "fault": "healthy discrepancy <= combined budget AND faulty discrepancy > combined budget; sensitivity alone is not detection",
            "prohibited": "no minimum cross-code residual, fitted acceptance threshold, or unchanged-state subtraction across fault preparations",
        },
        "faults": {"Dn_p1": "multiply D_n_face by 1.01 throughout material construction, preparation and both A/B controls",
                   "Dion_p1": "DF mu_c*1.01 with new common equilibrium and both controls"},
        "constants": {"eps0_F_m": 8.8541878128e-12, "q_C": Q,
                      "source": "NIST CODATA 2018 wall chart, https://physics.nist.gov/cuu/pdf/wall_2018.pdf",
                      "reason": "production R1 constants are frozen to CODATA 2018; no update to a different CODATA release",
                      "df_input_eps": "eps_r * (eps0_F_m / q_C / 100) / pc.epp0"},
        "reference_extraction": "reconstruct df.m F_n/F_p/F_c at mesh midpoints, retain c_max steric term, multiply mobseti; no calcJ/calcJdd observable",
        "protocol": "R1 ideal 0-/0+; DF soleq.ion -> ions-frozen finite ramp -> dwell. Ramp spread is explicit.",
        "limitations": ["DF finite interface and finite contact velocities remain model differences", "SRH and interface kinetics coverage not claimed"],
    }


def extract_rows(prepared, accepted):
    import numpy as np
    faces = np.asarray(prepared["grid"]["faces_m"])[1:-1]
    out = []
    for row in accepted:
        pr = row.get("physics_reconstruction") or {}
        if "positive_ion_flux_m2_s" not in pr:
            continue
        item = {"phase": row["phase"], "t_s": row["time_s"], "substeps": row.get("substeps"),
                "dt_s": row.get("dt_s")}
        for name, vals in (("Jion", Q*np.asarray(pr["positive_ion_flux_m2_s"])),
                           ("Jn", np.asarray(pr["electron_current_A_m2"])),
                           ("Jp", np.asarray(pr["hole_current_A_m2"]))):
            if vals.shape != faces.shape:
                raise ValueError("current and face coordinates differ")
            for nm in PROBES_NM:
                item[f"{name}_nm{nm}"] = float(np.interp(nm*1e-9, faces, vals)/1e4)
            item[name+"_absmax"] = float(np.max(np.abs(vals))/1e4)
        out.append(item)
    return out


def run_r1(args):
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, json_data, execution_source
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, r1_policy
    from perovskite_sim.models.config_loader import load_device_from_yaml
    import perovskite_sim.experiments.one_dimensional_mechanism_r1 as material_module
    import perovskite_sim.experiments.one_dimensional_mechanism_r1_state as state_module

    source = execution_source()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    fixture = project/"tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    binding_path = project/"tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"
    if args.output.exists():
        raise FileExistsError("do not overwrite an existing computation")
    if args.contract is None or json.loads(args.contract.read_text())["schema"] != "R1V6CrossCodeContractV1":
        raise ValueError("a prior frozen contract is required")
    args.output.mkdir(parents=True)
    if args.fault == "Dn_p1":
        original = material_module.build_r1_material
        def changed_material(*a, **k):
            grid, material = original(*a, **k)
            return grid, replace(material, D_n_face=material.D_n_face*1.01)
        material_module.build_r1_material = changed_material
        state_module.build_r1_material = changed_material
    stack = load_device_from_yaml(fixture)
    binding = json.loads(binding_path.read_text())
    summary = {"schema": "R1V6CrossCodeRunV1", "source_commit": commit,
               "project": str(project), "actual_source": source, "script_sha256": digest(__file__),
               "contract_sha256": digest(args.contract), "fixture_sha256": digest(fixture),
               "binding_sha256": digest(binding_path), "fault": args.fault,
               "fault_scope": "all preparations and A/B material assemblies" if args.fault != "none" else "none",
               "scope": "development_diagnostic_not_R1_2_qualification", "cases": []}
    write(args.output/"Summary.json", summary)
    with threadpool_limits(limits=1, user_api="blas"):
        summary["runtime"] = {"python": sys.version, "numpy": np.__version__, "blas": threadpool_info()}
        for intervals in args.intervals:
            started = time.monotonic()
            prepared = prepare_common_state(stack, intervals, binding, policy=r1_policy())
            for control in args.controls:
                label = f"{control}_N{intervals}_F{str(args.factor).replace('.', 'p')}_T{args.tier}_{args.fault}"
                directory = args.output/label
                directory.mkdir()
                write(directory/"Prepared.json", prepared.to_dict())
                rows = []
                def observe(row):
                    raw = json_data(row)
                    rows.append(raw)
                    with (directory/"AcceptedStepsV1.jsonl").open("a") as stream:
                        stream.write(json.dumps(raw, allow_nan=False)+"\n")
                entry = {"case": label, "intervals": intervals, "control": control,
                         "nonlinear_factor": args.factor, "time_substeps": [args.tier, 2*args.tier, 4*args.tier]}
                try:
                    result = run_r1_step(stack, intervals, binding, prepared, control=control,
                        policy=r1_policy(args.factor, time_substeps=tuple(entry["time_substeps"])),
                        times_s=list(TIMES_S), physics_evidence=True, accepted_step_observer=observe)
                    entry["status"] = "completed"
                except Exception as exc:
                    result = getattr(exc, "result", {})
                    entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                write(directory/"Result.json", json_data(result))
                entry["protocol_certified"] = result.get("certificate", {}).get("certified", False)
                entry["elapsed_s"] = time.monotonic()-started
                entry["rows"] = extract_rows(prepared.to_dict(), rows)
                write(directory/"ObservablesV1.json", entry)
                summary["cases"].append({k:v for k,v in entry.items() if k != "rows"})
                write(args.output/"Summary.json", summary)
                print(label, entry["status"], len(rows), flush=True)
    summary["source_unchanged"] = source == execution_source()
    write(args.output/"Summary.json", summary)
    return 0 if all(c["status"] == "completed" for c in summary["cases"]) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "r1", "extract"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--intervals", type=int, nargs="+", choices=(32, 64, 128, 256), default=[64])
    parser.add_argument("--controls", nargs="+", choices=("A", "B"), default=["B"])
    parser.add_argument("--factor", type=float, choices=(1., .1, .01, .001), default=.1)
    parser.add_argument("--tier", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--fault", choices=("none", "Dn_p1"), default="none")
    parser.add_argument("--case", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "freeze":
        if args.output.exists():
            raise FileExistsError("contract already frozen")
        write(args.output, contract())
        return 0
    if args.operation == "r1":
        return run_r1(args)
    prepared = json.loads((args.case/"Prepared.json").read_text())
    rows = [json.loads(line) for line in (args.case/"AcceptedStepsV1.jsonl").read_text().splitlines()]
    write(args.output, {"source_case": str(args.case), "prepared_sha256": digest(args.case/"Prepared.json"),
                        "rows_sha256": digest(args.case/"AcceptedStepsV1.jsonl"), "rows": extract_rows(prepared, rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

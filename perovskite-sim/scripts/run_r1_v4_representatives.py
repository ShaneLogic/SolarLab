"""Run exactly the seven authorized V4 development representatives on a frozen tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, json_data, execution_source
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.models.config_loader import load_device_from_yaml

CASES = ((64, 1., 1, "D"), (32, .01, 1, "D"), (64, .01, 1, "D"),
         (256, .01, 1, "D"), (256, .01, 4, "A"), (256, 1., 2, "D"), (256, 1., 4, "B"))
TIMES = [0., 1e-9, 1e-8, 1e-6, 1e-4]


def write(path, value):
    path.write_text(json.dumps(json_data(value), indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT):
        raise ValueError("development representatives require a clean frozen source")
    context = require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit)
    args.output.mkdir(parents=True)
    source = execution_source()
    summary = {"schema": "R1V4SevenRepresentatives", "scope": "development_numerical_evidence_not_R1_2_acceptance",
        "base_commit": args.expected_commit, "actual_source": source,
        "source_content_sha256": context.source_content_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "times_s": TIMES, "amplitude_V": .005, "cases": [],
        "runtime": {"executable": sys.executable, "python": sys.version, "blas": threadpool_info()}}
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((PROJECT / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    preparations = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for n, factor, level, control in CASES:
            require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                                expected_source_sha256=context.source_content_sha256)
            if execution_source() != source:
                raise ValueError("source changed between representative cases")
            name = f"{control}_N{n}_F{str(factor).replace('.', 'p')}_T{level}"
            directory = args.output / name
            directory.mkdir()
            if n not in preparations:
                preparations[n] = prepare_common_state(stack, n, binding, policy=r1_policy())
            prepared = preparations[n]
            write(directory / "Prepared.json", prepared.to_dict())
            policy = r1_policy(factor, time_substeps=(level, 2*level, 4*level))
            entry = {"case": name, "intervals": n, "factor": factor, "control": control,
                     "time_substeps": list(policy.refinement_substeps)}
            start = time.monotonic()
            def observe(row):
                with (directory / "AcceptedStepsV1.jsonl").open("a") as stream:
                    stream.write(json.dumps(json_data(row), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            try:
                result = run_r1_step(stack, n, binding, prepared, control=control,
                    policy=policy, times_s=TIMES, physics_evidence=True, accepted_step_observer=observe)
                entry["status"] = "completed"
            except Exception as exc:
                result = getattr(exc, "result", {})
                entry.update(status="failed", error_type=type(exc).__name__, message=str(exc))
            entry["elapsed_s"] = time.monotonic() - start
            write(directory / "Result.json", result)
            entry["accepted_rows"] = len(result.get("accepted_steps", []))
            entry["certified"] = result.get("certificate", {}).get("certified", False)
            try:
                replay = verify_r1_step_physics(stack, n, binding, prepared, result,
                                               allow_incomplete=entry["status"] != "completed")
                write(directory / "PhysicsReplay.json", replay)
                entry["physics_replay_certified"] = replay.get("certified")
            except Exception as exc:
                entry["physics_replay_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if entry["status"] == "failed":
                try:
                    terminal = rebuild_failure_witness(stack, n, binding, prepared, result)
                    write(directory / "FailureReplay.json", terminal)
                    entry["terminal_replay"] = terminal
                except Exception as exc:
                    entry["terminal_replay_error"] = {"type": type(exc).__name__, "message": str(exc)}
            summary["cases"].append(entry)
            write(args.output / "Summary.json", summary)
            print(name, entry["status"], entry["accepted_rows"], entry.get("terminal_replay", {}).get("metrics"), flush=True)


if __name__ == "__main__":
    main()

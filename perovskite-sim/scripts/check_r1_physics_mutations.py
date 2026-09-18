"""Bounded physical-contract mutations with a green unmutated control."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
TARGET = PROJECT / "perovskite_sim/experiments/one_dimensional_mechanism_r1_independent_physics.py"
TESTS = ["tests/unit/experiments/test_r1_independent_physics.py",
         "tests/unit/experiments/test_r1_physics_boundaries_v4.py",
         "tests/unit/experiments/test_r1_regular_independence_v4.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-law-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = TARGET.read_text()
    environment = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    command = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "--tb=short", *TESTS]
    def run(name):
        with (args.output / (name + ".log")).open("w") as stream:
            completed = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        return completed.returncode
    baseline = run("Baseline")
    report = {"schema": "R1PhysicsMutationCheckV4", "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "command": command, "baseline_exit": baseline, "mutations": []}
    if baseline:
        raise RuntimeError("unmutated control must pass before testing mutations")
    if args.global_law_only:
        target = PROJECT / "perovskite_sim/discretization/fe_operators.py"
        saved = target.read_text()
        prefix, suffix = saved.split("@dataclass(frozen=True, slots=True)", 1)
        if prefix.count("return result") != 2:
            raise ValueError("expected the Bernoulli and derivative return sites")
        changed = prefix.replace("return result", "return 1.001 * result") + "@dataclass(frozen=True, slots=True)" + suffix
        patch = "".join(difflib.unified_diff(saved.splitlines(True), changed.splitlines(True),
            fromfile=str(target.relative_to(PROJECT)), tofile=str(target.relative_to(PROJECT))))
        (args.output / "GlobalLaw.patch").write_text(patch)
        try:
            target.write_text(changed)
            code = run("GlobalLaw")
        finally:
            target.write_text(saved)
        report["mutations"] = [{"name": "global_B_and_Bprime_times_1p001", "exit": code, "killed": code == 1,
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}]
        (args.output / "Summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if code != 1:
            raise RuntimeError("shared constitutive mutant was not rejected by analytic tests")
        return
    mutations = []
    for name, value in (("internal_face_current_spread_relative", "2e-6"),
                        ("contact_internal_current_spread_relative", "2e-6"),
                        ("interface_current_spread_relative", "2e-6"),
                        ("charge_balance_normalized", "1e-10"),
                        ("inventory_relative_drift", "1e-10")):
        mutations.append((name, '"' + name + '": ' + value + ',', '"' + name + '": ' + ("2e-3" if value == "2e-6" else "1e-7") + ','))
    mutations.extend([
        ("range_to_std", "np.ptp(values)", "np.std(values)"),
        ("spread_floor", "SPREAD_FLOOR_A_M2 = 1e-20", "SPREAD_FLOOR_A_M2 = 1e-17"),
        ("charge_floor", "CHARGE_SCALE_FLOOR_A_M2 = 1.", "CHARGE_SCALE_FLOOR_A_M2 = 10."),
        ("site_ceiling", "ION_SITE_OCCUPANCY_CEILING = .999", "ION_SITE_OCCUPANCY_CEILING = 1.001"),
        ("inclusive_boundary", "value <= METRIC_LIMITS[name]", "value < METRIC_LIMITS[name]"),
        ("inventory_applicability", '"inventory_relative_drift": "all_saved_rows"', '"inventory_relative_drift": "finite_step"'),
        ("shared_bernoulli_scale", "bernoulli(xi_n)*state.n[1:]", "1.001*bernoulli(xi_n)*state.n[1:]"),
    ])
    try:
        for name, before, after in mutations:
            if original.count(before) != 1:
                raise ValueError("mutation does not match one source location: " + name)
            changed = original.replace(before, after, 1)
            patch = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                fromfile=str(TARGET.relative_to(PROJECT)), tofile=str(TARGET.relative_to(PROJECT))))
            (args.output / (name + ".patch")).write_text(patch)
            TARGET.write_text(changed)
            code = run(name)
            report["mutations"].append({"name": name, "exit": code, "killed": code == 1,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()})
            TARGET.write_text(original)
            (args.output / "Summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, code, flush=True)
    finally:
        TARGET.write_text(original)
    if not all(item["killed"] for item in report["mutations"]):
        raise RuntimeError("non-equivalent mutation survived or tests did not execute normally")


if __name__ == "__main__":
    main()

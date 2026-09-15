#!/usr/bin/env python3
"""Run R1-1 common preparation, A-D zero checks, or a short ideal step."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from run_one_dimensional_mechanism_r1 import (
    json_ready, manifest, sha256, source_record, write_json,
)

INPUT_PATH = PROJECT / "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
CONTRACT_PATH = PROJECT / "docs/OneDimensionalMechanismR1DynamicsV1.md"
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
COMMON_ARTIFACTS = frozenset({
    "CompletionV1.json", "StudyInputV1.json", "SourceFixtureV1.yaml",
    "ExecutionContractV1.md", "EnvironmentV1.json", "ResolvedStackV1.json",
    "ProtocolV1.json", "ReferenceBindingV1.json", "PreparedStateV1.json",
    "SourceV1.zip", "SourceManifestV1.json", "SourceChangesV1.patch",
})


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant, object_pairs_hook=_unique_pairs,
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_output(output):
    """Check a sealed bundle's bytes; do not promote it to new physics evidence."""
    output = output.resolve(strict=True)
    entries = read_json(output / "ManifestV1.json")
    if not isinstance(entries, dict) or "CompletionV1.json" not in entries:
        raise ValueError("manifest lacks the completion record")
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
        and path.relative_to(output).as_posix() != "ManifestV1.json"
    }
    if observed != set(entries):
        raise ValueError("manifest does not cover exactly the saved files")
    for name, entry in entries.items():
        path = (output / name).resolve()
        if (
            Path(name).is_absolute() or not path.is_relative_to(output)
            or not path.is_file() or not isinstance(entry, dict)
            or path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"artifact identity mismatch: {name}")
    completion = read_json(output / "CompletionV1.json")
    if (
        completion.get("schema") != "R1StageOneCompletionV1"
        or completion.get("stage_scope") != "R1-1"
        or completion.get("stage") not in ("prepare", "zero-check", "step")
        or completion.get("status") not in ("passed", "failed")
        or (completion["status"] == "passed") != (completion.get("failure") is None)
    ):
        raise ValueError("invalid R1-1 completion record")
    if completion["status"] == "passed":
        required = set(COMMON_ARTIFACTS)
        if completion["stage"] == "zero-check":
            required.add("ZeroExcitationV1.json")
        if completion["stage"] == "step":
            required.update(("StepResultV1.json", "AcceptedStepsV1.json"))
        if not required <= entries.keys():
            raise ValueError("passed completion lacks required R1-1 evidence")
    elif "FailureV1.json" not in entries:
        raise ValueError("failed completion lacks its failure record")
    return completion, len(entries)


def _record_failure_result(output, result):
    """Keep invalid numeric evidence without emitting non-standard JSON."""
    try:
        write_json(output / "FailedResultV1.json", result)
    except ValueError:
        import numpy as np

        def tag_nonfinite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return {"nonfinite": repr(value)}
            if isinstance(value, dict):
                return {key: tag_nonfinite(child) for key, child in value.items()}
            if isinstance(value, list):
                return [tag_nonfinite(child) for child in value]
            return value

        # The original array payload remains lossless in NPZ, including NaN/Inf.
        arrays = {}

        def collect(value, key="data"):
            import dataclasses

            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    collect(getattr(value, field.name), key + "." + field.name)
            elif isinstance(value, dict):
                for name, child in value.items():
                    collect(child, key + "." + str(name))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    collect(child, key + "." + str(index))

        collect(result)
        write_json(output / "FailedResultV1.json", tag_nonfinite(json_ready(result)))
        if arrays:
            np.savez_compressed(output / "FailedResultV1.npz", **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "zero-check", "step", "verify"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--control", choices=tuple("ABCD"))
    parser.add_argument("--intervals", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--amplitude", type=float, default=0.005, metavar="VOLTS")
    parser.add_argument("--nonlinear-factor", type=float, choices=(1.0, 0.1, 0.01, 0.001), default=0.1)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    args = parser.parse_args(argv)
    if args.stage == "list":
        print(
            "prepare: shared dark zero-bias D equilibrium from a fixed R1-0 reference\n"
            "zero-check: remaining equations for A-D from the same prepared state\n"
            "step: 0-/0+, impulse charge, and short regular response for one control\n"
            "verify: saved-file integrity and recorded completion status\n"
            "Scope: R1-1 only; long windows and the convergence matrix belong to R1-2."
        )
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if args.stage == "verify":
        try:
            completion, count = verify_output(output)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"verify failed: {exc}", file=sys.stderr)
            return 1
        print(f"verified {count} artifact identities; recorded {completion['stage']} status: {completion['status']}")
        return int(completion["status"] != "passed")
    if args.reference is None:
        parser.error("--reference is required; R1-1 does not prepare a new f_ref")
    if args.stage in ("zero-check", "step") and args.prepared is None:
        parser.error("--prepared is required; controls must share one D preparation")
    if args.stage == "step" and args.control is None:
        parser.error("--control is required for a step")
    if args.stage == "prepare" and (args.control is not None or args.prepared is not None):
        parser.error("prepare creates the common D state; --control/--prepared are not applicable")
    if not math.isfinite(args.amplitude) or not 0.0 < args.amplitude < 0.02:
        parser.error("--amplitude must be finite, positive, and below 0.02 V")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir already exists; use a fresh evidence directory")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    started = time.monotonic()
    failure = None
    accepted = []
    try:
        source_record(output)
        study = read_json(args.input)
        if _canonical(study) != _canonical(read_json(INPUT_PATH)):
            raise ValueError("R1-1 input differs from the supported versioned protocol")
        shutil.copyfile(args.input, output / "StudyInputV1.json")
        shutil.copyfile(CONTRACT_PATH, output / "ExecutionContractV1.md")
        fixture = PROJECT / study["fixture"]
        if sha256(fixture) != study["fixture_sha256"]:
            raise ValueError("source fixture mismatch")
        shutil.copyfile(fixture, output / "SourceFixtureV1.yaml")
        shutil.copyfile(args.reference, output / "ReferenceBindingV1.json")
        binding = read_json(output / "ReferenceBindingV1.json")
        if args.prepared is not None:
            shutil.copyfile(args.prepared, output / "PreparedStateV1.json")

        import numpy as np
        import scipy
        import scipy.linalg
        import perovskite_sim
        from threadpoolctl import threadpool_info, threadpool_limits
        from perovskite_sim.models.config_loader import load_device_from_yaml
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import (
            validate_r1_study_binding,
        )
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
            R1PreparedState, prepare_common_state,
        )
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
            check_zero_excitation, r1_policy, run_r1_step,
        )

        if not Path(perovskite_sim.__file__).resolve().is_relative_to(PROJECT):
            raise RuntimeError("package import did not resolve to this execution source")
        stack = load_device_from_yaml(fixture)
        policy = r1_policy(nonlinear_factor=args.nonlinear_factor)
        write_json(output / "ResolvedStackV1.json", stack)
        with threadpool_limits(limits=1, user_api="blas"):
            backends = threadpool_info()
            write_json(output / "EnvironmentV1.json", {
                "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "numpy": np.__version__,
                "scipy": scipy.__version__, "blas": backends,
                "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
                "parent_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
                ).strip(),
                "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
                "started_utc": datetime.now(timezone.utc).isoformat(),
            })
            if not backends or any(item["num_threads"] != 1 for item in backends):
                raise RuntimeError("R1 requires observed single-thread BLAS")
            validate_r1_study_binding(binding, stack)
            write_json(output / "ProtocolV1.json", {
                "stage_scope": "R1-1", "stage": args.stage,
                "intervals": args.intervals, "policy": policy,
                "control": "D" if args.stage == "prepare" else args.control or "ABCD",
                "control_definitions": study["controls"],
                "amplitude_V": args.amplitude if args.stage == "step" else 0.0,
                "times_s": study["functional_times_s"] if args.stage == "step" else None,
                "reference_file_sha256": sha256(output / "ReferenceBindingV1.json"),
                "reference_binding_sha256": binding.get("sha256"),
                "approved_reference_binding_sha256": study["fixed_reference_binding_sha256"],
                "prepared_file_sha256": (
                    None if args.prepared is None else sha256(output / "PreparedStateV1.json")
                ),
                "input_sha256": sha256(output / "StudyInputV1.json"),
                "contract_sha256": sha256(output / "ExecutionContractV1.md"),
                "claims_excluded": study["claims_excluded"],
            })
            if args.stage == "prepare":
                prepared = prepare_common_state(stack, args.intervals, binding, policy=policy)
                write_json(output / "PreparedStateV1.json", prepared.to_dict())
            else:
                prepared = R1PreparedState.from_dict(read_json(output / "PreparedStateV1.json"))
                if args.stage == "zero-check":
                    result = check_zero_excitation(
                        stack, args.intervals, binding, prepared,
                        controls=args.control or "ABCD", policy=policy,
                    )
                    write_json(output / "ZeroExcitationV1.json", result)
                else:
                    write_json(output / "AcceptedStepsV1.json", accepted)

                    def observe(record):
                        accepted.append(record)
                        write_json(output / "AcceptedStepsV1.json", accepted)

                    result = run_r1_step(
                        stack, args.intervals, binding, prepared, control=args.control,
                        amplitude_V=args.amplitude, times_s=np.asarray(study["functional_times_s"]),
                        policy=policy, accepted_step_observer=observe,
                    )
                    write_json(output / "StepResultV1.json", result)
                if not result["certificate"]["certified"]:
                    error = RuntimeError(f"R1-1 certification failed: {result['certificate']['reasons']}")
                    error.result = result
                    raise error
    except Exception as exc:
        failure = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "FailureV1.json", failure)
        if getattr(exc, "result", None) is not None:
            try:
                _record_failure_result(output, exc.result)
            except Exception as serialization_error:
                # A serialization failure must still leave the run failed/sealed.
                write_json(output / "FailureSerializationV1.json", {
                    "type": type(serialization_error).__name__,
                    "message": str(serialization_error),
                })
    write_json(output / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-1",
        "stage": args.stage, "status": "failed" if failure else "passed",
        "duration_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "failure": failure,
        "accepted_record_count": len(accepted),
    })
    manifest(output)
    print(f"R1-1 {args.stage}: {'failed' if failure else 'passed'}: {output}")
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())

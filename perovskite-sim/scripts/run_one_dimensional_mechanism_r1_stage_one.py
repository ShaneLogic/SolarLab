#!/usr/bin/env python3
"""Run R1 common-state controls; opt into R1-2 numerical axes in development."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
if not (sys.flags.isolated and sys.flags.no_site):
    sys.path.insert(0, str(PROJECT))
from run_one_dimensional_mechanism_r1 import (
    json_ready, sha256, source_record, write_json,
)

INPUT_PATH = PROJECT / "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
CONTRACT_PATH = PROJECT / "docs/OneDimensionalMechanismR1DynamicsV1.md"
CRITERION_PATH = PROJECT / "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md"
ADDITIONAL_FAILURES_PATH = PROJECT / "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import (
    COMMON_ARTIFACTS, read_json, _read_json_bytes, _canonical,
    verify_output, verify_acceptance,
)


def manifest(output):
    """Seal all artifacts, including the archived parent's own manifest."""
    entries = {
        path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.relative_to(output).as_posix() != "ManifestV1.json"
    }
    write_json(output / "ManifestV1.json", entries)


def _write_evidence(path, result):
    """Atomically replace each evidence file, tagging nonfinite JSON values."""
    import dataclasses
    import numpy as np

    path = Path(path)
    def tagged(value):
        if dataclasses.is_dataclass(value):
            return {field.name: tagged(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, np.ndarray):
            return tagged(value.tolist())
        if isinstance(value, np.generic):
            return tagged(value.item())
        if isinstance(value, complex):
            return {"real": tagged(value.real), "imag": tagged(value.imag)}
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": repr(value)}
        if isinstance(value, dict):
            return {key: tagged(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [tagged(child) for child in value]
        return value

    arrays = {}
    def collect(value, key="data"):
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
            arrays[key] = np.asarray(value)
        elif isinstance(value, (complex, np.complexfloating)) and not np.isfinite(value):
            arrays[key] = np.asarray(value)
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
    text = json.dumps(tagged(result), indent=2, allow_nan=False) + "\n"
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".r1-evidence-", delete=False) as stream:
            json_temp = Path(stream.name)
            temporary.append(json_temp)
            stream.write(text)
        if arrays:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=".r1-arrays-", delete=False) as stream:
                array_temp = Path(stream.name)
                temporary.append(array_temp)
                np.savez_compressed(stream, **arrays)
            os.replace(array_temp, path.with_suffix(".npz"))
        os.replace(json_temp, path)
    finally:
        for temporary_path in temporary:
            temporary_path.unlink(missing_ok=True)


def _record_failure_result(output, result):
    _write_evidence(Path(output) / "FailedResultV1.json", result)


def _record_execution_source(output):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
        record_frozen_source, require_r1_checkout,
    )
    context = require_r1_checkout(project=PROJECT, runner=Path(__file__))
    if not INPUT_PATH.samefile(context.project / "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"):
        raise ValueError("CLI study input does not identify the tracked checkout policy")
    record_frozen_source(output, context)
    return context


def _import_preparation(output, prepared_path, expected_manifest, context):
    """Snapshot and check a whole parent bundle before trusting its metadata."""
    parent = Path(prepared_path).resolve(strict=True).parent
    if Path(prepared_path).name != "PreparedStateV1.json":
        raise ValueError("formal preparation must be PreparedStateV1.json in its sealed bundle")
    if not expected_manifest:
        raise ValueError("formal preparation import requires --prepared-manifest-sha256")
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("preparation and consuming output must be separate bundle directories")
    if sha256(parent / "ManifestV1.json") != expected_manifest:
        raise ValueError("preparation manifest differs from the external preparation anchor")
    # Check path/coverage before copying, then validate the immutable copied
    # bytes against the external anchor. A change during copying is rejected.
    verify_output(parent)
    archived = output / "PreparationV1"
    archived.mkdir()
    for name in ["ManifestV1.json", *read_json(parent / "ManifestV1.json")]:
        src, dest = parent / name, archived / name
        if src.is_symlink():
            raise ValueError("preparation bundle must not contain symbolic links")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    completion, _ = verify_acceptance(
        archived, expected_manifest_sha256=expected_manifest,
        expected_source_commit=context.source_commit, source_repository=context.root,
        required_evidence_revision=5,
    )
    if completion["stage"] != "prepare" or completion["status"] != "passed":
        raise ValueError("formal import requires a passed formal preparation bundle")
    for filename in ("StudyInputV1.json", "SourceFixtureV1.yaml", "ExecutionContractV1.md",
                     "OperatorCriterionDecisionV2.md", "AdditionalFailuresV1.json",
                     "ReferenceBindingV1.json"):
        if (archived / filename).read_bytes() != (output / filename).read_bytes():
            raise ValueError("preparation and consuming inputs disagree: " + filename)
    raw = (archived / "PreparedStateV1.json").read_bytes()
    (output / "PreparedStateV1.json").write_bytes(raw)
    execution = read_json(archived / "ExecutionSourceV1.json")
    preparation = {
        "manifest_sha256": expected_manifest, "source_commit": execution["source_commit"],
        "source_content_sha256": execution["source_content_sha256"],
        "run_class": completion["run_class"], "evidence_revision": completion["evidence_revision"],
    }
    return preparation, _read_json_bytes(raw)["sha256"]


def _merge_additional_failures(study, additional):
    """Merge pinned historical observations without altering the canonical input."""
    if additional.get("schema") != "R1AdditionalFailuresV1":
        raise ValueError("unsupported supplemental R1 failure registry")
    semantics = additional.get("known_failure_semantics", {})
    if any(semantics.get(key) is not False for key in (
        "waives_checks", "skip_computation", "changes_acceptance_thresholds",
    )):
        raise ValueError("supplemental historical failures cannot waive execution or checks")
    entries = additional.get("known_physical_gate_failures")
    if not isinstance(entries, list) or not entries:
        raise ValueError("supplemental R1 failure registry requires physical-gate observations")
    known = [*study.get("known_nonconvergence", []), *study.get("known_physical_gate_failures", [])]
    identities = [entry["case_id"] for entry in [*known, *entries]]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate case identity in supplemental R1 failure registry")
    return {**study, "known_physical_gate_failures": [*study.get("known_physical_gate_failures", []), *entries]}


def _historical_case_observation(study, args, policy, failure):
    """Report previous measurements without treating them as waivers."""
    if args.stage != "step" or study is None or policy is None:
        return None
    matched = []
    for category in ("known_nonconvergence", "known_physical_gate_failures"):
        for entry in study.get(category, []):
            if any(entry.get(key) != actual for key, actual in (
                ("intervals", args.intervals), ("control", args.control),
                ("nonlinear_factor", args.nonlinear_factor), ("amplitude_V", args.amplitude),
                ("times_s", getattr(args, "resolved_times_s", study["functional_times_s"])),
                ("refinement_substeps", list(policy.refinement_substeps)),
            )):
                continue
            matched.append({
                "case_id": entry["case_id"], "category": category,
                "observed_source_commit": entry["observed_source_commit"],
                "historical_failure": entry["failure"],
                "outcome": ("previously_failed_case_now_passed" if failure is None else
                            "historical_signature_recurred" if entry["failure"] in failure["message"] else
                            "different_failure_signature"),
            })
    return {"matching_historical_cases": matched, "waives_checks": False,
            "signature_comparison": "failure_message_substring_only; recorded metric values are not compared",
            "note": "Historical observations do not determine current acceptance or exit status"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "zero-check", "step", "verify"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-manifest-sha256",
                        help="externally selected parent preparation manifest digest for formal import/verification")
    parser.add_argument("--control", choices=tuple("ABCD"))
    parser.add_argument("--intervals", type=int, choices=(16, 32, 64, 128, 256), default=16)
    parser.add_argument("--amplitude", type=float, default=0.005, metavar="VOLTS")
    parser.add_argument("--nonlinear-factor", type=float, choices=(1.0, 0.1, 0.01, 0.001), default=0.1)
    parser.add_argument("--time-substeps", type=int, nargs=3, default=(1, 2, 4),
                        metavar=("COARSE", "MIDDLE", "FINE"),
                        help="step time-axis setting: 1 2 4 through 16 32 64; nondefault requires --development")
    parser.add_argument("--window", choices=("functional", "full"), default="functional",
                        help="step output grid: legacy short functional grid or section-6 full logarithmic grid")
    parser.add_argument("--first-time-s", type=float,
                        help="first positive full-grid time: 1e-9, 1e-10, 1e-11 or 1e-12 s")
    parser.add_argument("--last-time-s", type=float,
                        help="full-grid endpoint: 1e2, 1e3, 1e4 or 1e5 s")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--mode", choices=("integrity", "acceptance"), default="integrity")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-source-commit",
                        help="full Git commit selected outside the bundle for current acceptance")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--ledger-sha256")
    parser.add_argument("--run-id")
    parser.add_argument("--required-evidence-revision", type=int, choices=(1, 2, 3, 4, 5), default=5,
                        help="acceptance format chosen outside the bundle; 1/2/3/4 are explicit legacy verification")
    parser.add_argument("--development", action="store_true",
                        help="explicit unverified development execution; not formal study evidence")
    args = parser.parse_args(argv)
    if args.stage == "list":
        print(
            "prepare: shared dark zero-bias D equilibrium from a fixed R1-0 reference\n"
            "zero-check: remaining equations for A-D from the same prepared state\n"
            "step: 0-/0+, impulse charge, and regular response for one control\n"
            "verify: byte consistency or comparison to a supplied external acceptance anchor\n"
            "Default scope: R1-1 short functional window.\n"
            "R1-2 development: --time-substeps selects an independent nested time setting;\n"
            "--intervals supports 16/32/64/128/256; --window full uses 0+ and 1e-9..1e2 s,\n"
            "12 intervals per decade; --first-time-s/--last-time-s select declared extensions.\n"
            "Nondefault time settings, extended spatial grids and full windows require --development.\n"
            "Each invocation runs one setting; three-axis convergence and full-window certification are not asserted."
        )
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if args.stage == "verify":
        try:
            if args.mode == "acceptance":
                completion, count = verify_acceptance(
                    output, expected_manifest_sha256=args.expected_manifest_sha256,
                    ledger=args.ledger, ledger_sha256=args.ledger_sha256, run_id=args.run_id,
                    required_evidence_revision=args.required_evidence_revision,
                    expected_source_commit=args.expected_source_commit,
                    expected_prepared_manifest_sha256=args.prepared_manifest_sha256,
                    source_repository=PROJECT.parent,
                )
            else:
                if any((args.expected_manifest_sha256, args.expected_source_commit,
                        args.prepared_manifest_sha256, args.ledger, args.ledger_sha256, args.run_id)):
                    raise ValueError("acceptance anchors require --mode acceptance")
                completion, count = verify_output(output)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"verify failed: {exc}", file=sys.stderr)
            return 1
        scope = ("supplied external anchor matched; no independent approval or physics rerun asserted"
                 if args.mode == "acceptance" else "provenance not authenticated")
        print(f"checksums consistent for {count} files; {scope}; recorded {completion['stage']} status: {completion['status']}")
        if completion.get("verification", {}).get("legacy"):
            print("legacy verification limits: " + str(completion["verification"]["limits"]))
        return int(completion["status"] != "passed")
    if args.mode != "integrity" or any((args.expected_manifest_sha256, args.expected_source_commit,
                                       args.ledger, args.ledger_sha256, args.run_id)):
        parser.error("verification options apply only to verify")
    if args.reference is None:
        parser.error("--reference is required; R1-1 does not prepare a new f_ref")
    if args.stage in ("zero-check", "step") and args.prepared is None:
        parser.error("--prepared is required; controls must share one D preparation")
    if args.stage == "step" and args.control is None:
        parser.error("--control is required for a step")
    if args.stage == "prepare" and (args.control is not None or args.prepared is not None
                                   or args.prepared_manifest_sha256 is not None):
        parser.error("prepare creates the common D state; --control/--prepared are not applicable")
    if not math.isfinite(args.amplitude) or not 0.0 < args.amplitude < 0.02:
        parser.error("--amplitude must be finite, positive, and below 0.02 V")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
        observation_times, validate_time_substeps,
    )
    try:
        args.time_substeps = validate_time_substeps(args.time_substeps)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.stage != "step" and (
        args.time_substeps != (1, 2, 4) or args.window != "functional"
        or args.first_time_s is not None or args.last_time_s is not None
    ):
        parser.error("time-axis and observation-window options apply only to step")
    if args.window == "functional" and (args.first_time_s is not None or args.last_time_s is not None):
        parser.error("--first-time-s/--last-time-s require --window full")
    args.resolved_times_s = None
    if args.window == "full":
        try:
            args.resolved_times_s = observation_times(
                first_time_s=1e-9 if args.first_time_s is None else args.first_time_s,
                last_time_s=1e2 if args.last_time_s is None else args.last_time_s,
            ).tolist()
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    stage_two = args.intervals > 64 or args.time_substeps != (1, 2, 4) or args.window != "functional"
    if stage_two and not args.development:
        parser.error("R1-2 axes/windows require --development; the formal R1-1 contract is unchanged")
    stage_scope = "R1-2-development" if stage_two else "R1-1"
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir already exists; use a fresh evidence directory")
    started = time.monotonic()
    failure = None
    execution_context = None
    study = None
    policy = None
    accepted = []
    persisted_count = 0
    preparation = None
    expected_prepared_sha256 = None
    try:
        execution_context = _record_execution_source(output)
        controlled = execution_context is not None and execution_context.run_class == "formal"
        if not controlled and not args.development:
            raise ValueError("formal R1 computation requires the controlled -I -S launcher; use --development for unverified development")
        if controlled and args.development:
            raise ValueError("controlled execution cannot be relabelled as development")
        canonical_input = (execution_context.read_bytes(INPUT_PATH) if execution_context is not None
                           else INPUT_PATH.read_bytes())
        supplied_input = (canonical_input if args.input.resolve() == INPUT_PATH.resolve()
                          else args.input.read_bytes())
        study = _read_json_bytes(supplied_input)
        if _canonical(study) != _canonical(_read_json_bytes(canonical_input)):
            raise ValueError("R1-1 input differs from the supported versioned protocol")
        # Execute and identify the canonical policy even when a caller provides
        # a semantically equivalent input with different JSON formatting.
        (output / "StudyInputV1.json").write_bytes(canonical_input)
        contract = (execution_context.read_bytes(CONTRACT_PATH) if execution_context is not None
                    else CONTRACT_PATH.read_bytes())
        (output / "ExecutionContractV1.md").write_bytes(contract)
        criterion = (execution_context.read_bytes(CRITERION_PATH) if execution_context is not None
                     else CRITERION_PATH.read_bytes())
        (output / "OperatorCriterionDecisionV2.md").write_bytes(criterion)
        additional_failures = (execution_context.read_bytes(ADDITIONAL_FAILURES_PATH)
                               if execution_context is not None else ADDITIONAL_FAILURES_PATH.read_bytes())
        (output / "AdditionalFailuresV1.json").write_bytes(additional_failures)
        fixture = PROJECT / study["fixture"]
        fixture_bytes = (execution_context.read_bytes(fixture) if execution_context is not None
                         else fixture.read_bytes())
        if hashlib.sha256(fixture_bytes).hexdigest() != study["fixture_sha256"]:
            raise ValueError("source fixture mismatch")
        (output / "SourceFixtureV1.yaml").write_bytes(fixture_bytes)
        shutil.copyfile(args.reference, output / "ReferenceBindingV1.json")
        binding = read_json(output / "ReferenceBindingV1.json")
        if args.prepared is not None and controlled:
            preparation, expected_prepared_sha256 = _import_preparation(
                output, args.prepared, args.prepared_manifest_sha256, execution_context,
            )
        elif args.prepared is not None:
            shutil.copyfile(args.prepared, output / "PreparedStateV1.json")
            if args.prepared_manifest_sha256 is not None:
                raise ValueError("development import cannot claim an anchored formal preparation chain")

        import numpy as np
        import scipy
        import scipy.linalg
        import perovskite_sim
        from threadpoolctl import threadpool_info, threadpool_limits
        from perovskite_sim.models.config_loader import load_device_from_yaml
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import (
            STUDY_INPUT_PATH, validate_r1_study_binding,
            execution_contract_identity, operator_criterion_identity, additional_failures_identity,
        )
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
            R1PreparedState, prepare_common_state,
        )
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
            check_zero_excitation, r1_policy, run_r1_step,
        )

        if not Path(perovskite_sim.__file__).resolve().is_relative_to(PROJECT):
            raise RuntimeError("package import did not resolve to this execution source")
        if not INPUT_PATH.samefile(STUDY_INPUT_PATH):
            raise RuntimeError("CLI and imported binding policy must be the same tracked file")
        execution_contract_identity()
        operator_criterion_identity()
        additional_failures_identity()
        study = _merge_additional_failures(study, _read_json_bytes(additional_failures))
        stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
        policy = r1_policy(nonlinear_factor=args.nonlinear_factor, time_substeps=args.time_substeps)
        if args.stage == "step" and args.resolved_times_s is None:
            args.resolved_times_s = study["functional_times_s"]
        write_json(output / "ResolvedStackV1.json", stack)
        with threadpool_limits(limits=1, user_api="blas"):
            backends = threadpool_info()
            write_json(output / "EnvironmentV1.json", {
                "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "numpy": np.__version__,
                "scipy": scipy.__version__, "blas": backends,
                "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
                "parent_commit": execution_context.commit if execution_context is not None else None,
                "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "checkout": execution_context.to_dict() if execution_context is not None else None,
                "run_class": "formal" if controlled else "development",
                "independent_approval": "not_asserted_by_execution",
                "controlled_runtime": execution_context.runtime if controlled else None,
            })
            if not backends or any(item["num_threads"] != 1 for item in backends):
                raise RuntimeError("R1 requires observed single-thread BLAS")
            validate_r1_study_binding(binding, stack)
            write_json(output / "ProtocolV1.json", {
                "stage_scope": stage_scope, "stage": args.stage,
                "source_commit": execution_context.source_commit if execution_context is not None else None,
                "source_content_sha256": execution_context.source_content_sha256 if execution_context is not None else None,
                "run_class": "formal" if controlled else "development",
                "preparation": preparation,
                "intervals": args.intervals, "policy": policy,
                "nonlinear_factor": args.nonlinear_factor,
                "time_substeps": list(policy.refinement_substeps),
                "observation_window": ({
                    "kind": args.window, "first_positive_time_s": args.resolved_times_s[1],
                    "last_time_s": args.resolved_times_s[-1],
                    "output_point_count": len(args.resolved_times_s),
                    "logarithmic_intervals_per_decade": 12 if args.window == "full" else None,
                } if args.stage == "step" else None),
                "numerical_validation_scope": "single setting; no three-axis convergence, amplitude linearity or dual-domain acceptance asserted",
                "control": "D" if args.stage == "prepare" else args.control or "ABCD",
                "control_definitions": study["controls"],
                "amplitude_V": args.amplitude if args.stage == "step" else 0.0,
                "times_s": args.resolved_times_s if args.stage == "step" else None,
                "reference_file_sha256": sha256(output / "ReferenceBindingV1.json"),
                "reference_binding_sha256": binding.get("sha256"),
                "approved_reference_binding_sha256": study["fixed_reference_binding_sha256"],
                "prepared_file_sha256": (
                    None if args.prepared is None else sha256(output / "PreparedStateV1.json")
                ),
                "input_sha256": sha256(output / "StudyInputV1.json"),
                "supplied_input_sha256": hashlib.sha256(supplied_input).hexdigest(),
                "contract_sha256": sha256(output / "ExecutionContractV1.md"),
                "criterion_sha256": sha256(output / "OperatorCriterionDecisionV2.md"),
                "additional_failures_sha256": sha256(output / "AdditionalFailuresV1.json"),
                "historical_failure_case_count": sum(len(study.get(key, [])) for key in (
                    "known_nonconvergence", "known_physical_gate_failures",
                )),
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
                        expected_prepared_sha256=expected_prepared_sha256,
                    )
                    write_json(output / "ZeroExcitationV1.json", result)
                else:
                    _write_evidence(output / "AcceptedStepsV1.json", accepted)

                    def observe(record):
                        nonlocal persisted_count
                        accepted.append(record)
                        _write_evidence(output / "AcceptedStepsV1.json", accepted)
                        persisted_count = len(accepted)

                    result = run_r1_step(
                        stack, args.intervals, binding, prepared, control=args.control,
                        amplitude_V=args.amplitude, times_s=np.asarray(args.resolved_times_s),
                        policy=policy, accepted_step_observer=observe,
                        expected_prepared_sha256=expected_prepared_sha256,
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
        "schema": "R1StageOneCompletionV1", "stage_scope": stage_scope,
        "stage": args.stage, "status": "failed" if failure else "passed",
        "duration_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "failure": failure,
        "evidence_revision": 5,
        "run_class": (execution_context.run_class if execution_context is not None else
                      "development" if args.development else "rejected_before_execution"),
        "independent_approval": "not_asserted_by_execution",
        "historical_case_observation": _historical_case_observation(study, args, policy, failure),
        "accepted_record_count": persisted_count,
        "observed_record_count": len(accepted), "persisted_record_count": persisted_count,
        "persisted_finite_step_count": sum(r.get("phase") == "accepted_regular_step" for r in accepted[:persisted_count]),
        "physical_passed_finite_step_count": sum(r.get("phase") == "accepted_regular_step" and r.get("physical_checks_passed") is True for r in accepted[:persisted_count]),
        "count_semantics": "record counts include 0+; accepted_record_count is the persisted record count, not a claim that every physical check passed",
    })
    manifest(output)
    print(f"{stage_scope} {args.stage}: {'failed' if failure else 'passed'}: {output}")
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay the fixed R0 case. See docs/OneDimensionalMechanismR0ProtocolV1.md."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile


SOURCE_COMMIT = "5b744b023ac708e50b6035616ca6f07cee3c9eb1"
CONFIG_PATH = "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
CONFIG_SHA256 = "f617f230b2d9c144573394e38fcc313225dc84c7b38f7670b97e9a0a7cc12a24"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_ready(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_baseline(project: Path, output: Path) -> dict:
    # Import the archived source before an editable installation can resolve it.
    if not (sys.flags.isolated and sys.flags.no_site):
        sys.path.insert(0, str(project))
    import numpy as np
    import scipy
    import scipy.linalg
    import perovskite_sim
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.dynamic_defect_transient import (
        build_dynamic_defect_transient_protocol,
        run_dynamic_defect_transient,
    )
    from perovskite_sim.experiments.interface_defect_ion_transient import (
        run_interface_defect_ion_device_transient,
    )
    from perovskite_sim.experiments.jv_sweep import build_electrical_grid
    from perovskite_sim.experiments.quasi_fermi_steady_state import (
        build_two_sided_trace_grid,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.physics.contacts import resolved_contact_velocities

    if not Path(perovskite_sim.__file__).resolve().is_relative_to(project):
        raise RuntimeError("package import did not resolve to the frozen source")
    fixture = project / CONFIG_PATH
    if sha256(fixture) != CONFIG_SHA256:
        raise RuntimeError("R0 source fixture hash mismatch")
    stack = load_device_from_yaml(fixture)
    grid = build_two_sided_trace_grid(build_electrical_grid(stack, 4), stack)
    protocol = build_dynamic_defect_transient_protocol(
        stack, grid, (0.0, 1.0e-8, 1.0e-6, 1.0e-4),
        (0.0, 0.05, 0.05, 0.05), requested_grid_intervals=4,
    )
    shutil.copyfile(fixture, output / "SourceFixtureV1.yaml")
    write_json(output / "ProtocolV1.json", protocol.to_dict())
    write_json(output / "ResolvedInputsV1.json", {
        "stack": stack,
        "effective_contact_velocities_m_s": resolved_contact_velocities(stack),
        "grid_m": grid,
        "grid_note": "4 total intervals before the two-sided trace transformation",
        "source_fixture_sha256": CONFIG_SHA256,
    })
    with threadpool_limits(limits=1, user_api="blas"):
        backends = threadpool_info()
        write_json(output / "EnvironmentV1.json", {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "blas": backends,
            "thread_environment": {key: os.environ[key] for key in THREAD_VARIABLES},
            "packages": dict(sorted(
                (item.metadata["Name"], item.version)
                for item in importlib.metadata.distributions()
                if item.metadata["Name"]
            )),
        })
        if not backends or any(item["num_threads"] != 1 for item in backends):
            raise RuntimeError("R0 requires an observed single-thread BLAS backend")
        print("Solving the source-bound research baseline", flush=True)
        raw = run_interface_defect_ion_device_transient(
            grid, stack, protocol.times_s, protocol.voltage_V,
            illuminated=False, policy=protocol.solver_policy,
            require_certificate=True,
        )
        write_json(output / "RawBaselineV1.json", raw)
        print("Replaying through the public protocol entry", flush=True)
        public = run_dynamic_defect_transient(grid, stack, protocol)
        write_json(output / "PublicReplayV1.json", public)

    compared_fields = (
        "electron_density_m3", "hole_density_m3", "positive_ion_density_m3",
        "electrostatic_potential_V", "interface_occupancy",
        "total_current_faces_A_m2", "interface_total_current_A_m2",
    )
    comparison = {}
    for name in compared_fields:
        first, second = np.asarray(getattr(raw, name)), np.asarray(getattr(public, name))
        comparison[name] = {
            "bitwise_equal": bool(np.array_equal(first, second)),
            "maximum_absolute_difference": float(np.max(np.abs(first - second))),
        }
    checks = {
        "raw_certificate": bool(raw.certificate.certified),
        "public_certificate": bool(public.evidence.certified),
        "dark_reference": bool(raw.dark_reference.dc_state.certificate.certified),
        "operating_point": bool(raw.dc_state.certificate.certified),
        "contact_thermodynamics": bool(raw.dc_state.certificate.contact_thermodynamics.certified),
        "four_effective_contacts_pinned": resolved_contact_velocities(stack) == (None,) * 4,
        "same_grid_repeatability": all(item["bitwise_equal"] for item in comparison.values()),
        "trap_motion_resolved_in_existing_gate": public.evidence.maximum_interface_occupancy_motion > 1.0e-9,
        "ion_motion_resolved_in_existing_gate": public.evidence.maximum_positive_ion_relative_motion > 1.0e-5,
        "no_clipping": not raw.certificate.clipping_used,
    }
    summary = {
        "schema": "one-dimensional-mechanism-r0-evidence-v1",
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol.protocol_hash,
        "checks": checks, "replay_comparison": comparison,
        "engine_certificate": raw.certificate,
        "public_evidence": public.evidence,
        "scope": "fixed dark short-transient baseline and same-grid replay only",
        "r1_campaign_executed": False,
        "new_refinement_matrix_executed": False,
        "all_accepted_step_states_exported": False,
    }
    write_json(output / "BaselineChecksV1.json", summary)
    if not all(checks.values()):
        raise RuntimeError("R0 baseline failed: " + ", ".join(
            name for name, passed in checks.items() if not passed
        ))
    return summary


def run_checks(project: Path, output: Path) -> dict:
    selected = [
        "tests/unit/experiments/test_dynamic_defect_transient.py",
        "tests/unit/twod/test_contact_boundary_semantics.py::test_every_resolved_contact_matches_one_dimension",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *selected,
               "--junitxml=" + str(output / "TargetedTestsV1.xml")]
    environment = dict(os.environ, PYTHONPATH=str(project))
    with (output / "TargetedTestsV1.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=project, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    root = ET.parse(output / "TargetedTestsV1.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {
        field: sum(int(suite.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    result = {"selection": selected, "exit_code": completed.returncode, **counts}
    write_json(output / "TargetedTestSummaryV1.json", result)
    if completed.returncode or counts["failures"] or counts["errors"] or not counts["tests"]:
        raise RuntimeError("R0 focused contract tests failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study-spec", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    specification = args.study_spec.resolve(strict=True)
    repo = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=False)
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    started = time.monotonic()
    status = "failed"
    failure = None
    with (output / "RunV1.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                write_json(output / "StartRecordV1.json", {
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "source_commit": SOURCE_COMMIT,
                    "study_spec_sha256": sha256(specification),
                    "runner_sha256": sha256(Path(__file__)),
                })
                shutil.copyfile(specification, output / "StudySpecV1.md")
                shutil.copyfile(__file__, output / "ReplayR0V1.py")
                shutil.copyfile(repo / "perovskite-sim/docs/OneDimensionalMechanismR0ProtocolV1.md",
                                output / "ExecutionContractV1.md")
                subprocess.run(
                    ["git", "archive", "--format=zip", "--output=" + str(output / "SourceV1.zip"), SOURCE_COMMIT],
                    cwd=repo, check=True, stdout=log, stderr=log,
                )
                with tempfile.TemporaryDirectory(prefix="solarlab_r0_") as scratch:
                    source = Path(scratch).resolve()
                    with zipfile.ZipFile(output / "SourceV1.zip") as archive:
                        members = [item for item in archive.infolist() if not item.is_dir()]
                        manifest = {
                            item.filename: {"bytes": item.file_size,
                                            "sha256": hashlib.sha256(archive.read(item)).hexdigest()}
                            for item in members
                        }
                        archive.extractall(source)
                    write_json(output / "SourceManifestV1.json", {
                        "source_commit": SOURCE_COMMIT, "file_count": len(manifest),
                        "files": manifest,
                    })
                    project = source / "perovskite-sim"
                    run_baseline(project, output)
                    run_checks(project, output)
                    status = "r0_baseline_verified"
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
                write_json(output / "FailureV1.json", failure)
                if getattr(exc, "result", None) is not None:
                    try:
                        write_json(output / "FailedResultV1.json", exc.result)
                    except (TypeError, ValueError):
                        traceback.print_exc()
                traceback.print_exc()
    write_json(output / "CompletionV1.json", {
        "status": status, "source_commit": SOURCE_COMMIT,
        "duration_s": time.monotonic() - started, "failure": failure,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(output / "ArtifactManifestV1.json", {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    })
    print(f"{status}: {output}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage-scoped R1-0 execution; never overwrite an earlier evidence directory."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile
import shutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from run_one_dimensional_mechanism_r0 import sha256


def json_ready(value):
    if dataclasses.is_dataclass(value):
        return {f.name: json_ready(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    import numpy as np
    arrays = {}
    def collect(item, prefix):
        if isinstance(item, np.ndarray):
            arrays[prefix] = item
        elif dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                collect(getattr(item, field.name), prefix+"."+field.name)
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(child, prefix+"."+str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                collect(child, prefix+"."+str(index))
    collect(value, "data")
    if arrays:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)


def source_record(output):
    repo = PROJECT.parent
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
    ).decode().split("\0")
    entries = {}
    with zipfile.ZipFile(output / "SourceV1.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            path = repo / name
            if not name or not path.is_file() or name.startswith("debug-"):
                continue
            entries[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            archive.write(path, name)
    write_json(output / "SourceManifestV1.json", entries)
    (output / "SourceChangesV1.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo))


def manifest(output):
    write_json(output / "ManifestV1.json", {
        p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "ManifestV1.json"
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["list", "geometry", "prepare", "coupled", "ac", "r0-regression", "verify"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--geometry-evidence", type=Path)
    parser.add_argument("--r0-baseline", type=Path)
    parser.add_argument("--intervals", type=int, default=16)
    parser.add_argument("--nonlinear-factor", type=float, choices=[1., .1, .01, .001], default=1.)
    parser.add_argument("--input", type=Path, default=PROJECT / "reproducibility/OneDimensionalMechanismR1InputV1.json")
    args = parser.parse_args()
    if args.stage == "list":
        print("geometry: GEO-01..06\nprepare: ReferenceBindingV1\ncoupled: GEO-07\nac: coupled storage geometry\nverify: byte manifest")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if args.stage == "verify":
        expected = json.loads((output / "ManifestV1.json").read_text())
        for name, entry in expected.items():
            if sha256(output / name) != entry["sha256"]:
                raise ValueError(f"artifact mismatch: {name}")
        print(f"verified {len(expected)} artifacts")
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    started = time.monotonic()
    failure = None
    try:
        source_record(output)
        canonical_path = PROJECT / "reproducibility/OneDimensionalMechanismR1InputV1.json"
        study_input = json.loads(args.input.read_text())
        if study_input != json.loads(canonical_path.read_text()):
            raise ValueError("R1-0 input differs from the supported versioned protocol")
        shutil.copyfile(args.input, output / "StudyInputV1.json")
        shutil.copyfile(PROJECT / "docs/OneDimensionalMechanismR1GeometryV1.md", output / "ExecutionContractV1.md")
        if sha256(PROJECT / study_input["fixture"]) != study_input["fixture_sha256"]:
            raise ValueError("source fixture mismatch")
        import numpy as np
        import scipy
        import scipy.linalg
        from threadpoolctl import threadpool_info, threadpool_limits
        from perovskite_sim.models.config_loader import load_device_from_yaml
        from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
        write_json(output / "EnvironmentV1.json", {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "blas": threadpool_info(),
            "parent_commit": source, "command": sys.argv,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        stack = load_device_from_yaml(PROJECT / study_input["fixture"])
        write_json(output / "ResolvedStackV1.json", stack)
        with threadpool_limits(limits=1, user_api="blas"):
            if not threadpool_info() or any(b["num_threads"] != 1 for b in threadpool_info()):
                raise RuntimeError("R1 requires observed single-thread BLAS")
            if args.stage == "geometry":
                write_json(output / "GeometryV1.json", [{
                    "intervals": n, "grid_m": x, "faces_m": m.physical_cell_faces_m,
                    "widths_m": m.dx_cell,
                    "left_inventory_m2": np.dot(m.dx_cell[x < 1e-7], m.P_ion0[x < 1e-7]),
                    "right_background_m2": np.dot(m.dx_cell[x > 1e-7], m.P_ion0[x > 1e-7]),
                } for n in [4, 16, 32, 64] for x, m in [build_r1_material(stack, n)]])
                with (output / "TestsV1.log").open("w") as log:
                    code = subprocess.call([
                        sys.executable, "-m", "pytest", "-q",
                        "tests/unit/experiments/test_one_dimensional_mechanism_r1_geometry.py",
                        f"--junitxml={output / 'TestsV1.xml'}",
                    ], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    raise RuntimeError(f"GEO-01..06 tests exited {code}")
            elif args.stage == "r0-regression":
                from run_one_dimensional_mechanism_r0 import run_baseline, run_checks
                if args.r0_baseline is None:
                    raise ValueError("--r0-baseline must point to the independent parent replay")
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    os.environ[key] = "1"
                # This call imports this worktree, unlike the frozen R0 main().
                run_baseline(PROJECT, output)
                run_checks(PROJECT, output)
                before = json.loads((args.r0_baseline / "RawBaselineV1.json").read_text())
                after = json.loads((output / "RawBaselineV1.json").read_text())
                fields = ["electron_density_m3", "hole_density_m3", "positive_ion_density_m3",
                          "electrostatic_potential_V", "interface_occupancy",
                          "total_current_faces_A_m2", "interface_total_current_A_m2"]
                comparison = {key: bool(np.array_equal(before[key], after[key])) for key in fields}
                write_json(output / "ParentComparisonV1.json", {
                    "bitwise_equal": comparison,
                    "note": "R0 runner labels identify its frozen protocol; SourceManifestV1 identifies actual modified execution source",
                })
                if not all(comparison.values()):
                    raise RuntimeError("modified-source R0 differs from the parent arrays")
            elif args.stage == "prepare":
                from perovskite_sim.experiments.one_dimensional_mechanism_r1 import prepare_reference
                if args.geometry_evidence is None:
                    raise ValueError("--geometry-evidence is required before reference binding")
                completion = json.loads((args.geometry_evidence / "CompletionV1.json").read_text())
                if completion["status"] != "passed":
                    raise ValueError("geometry gates have not passed")
                for name, entry in json.loads((args.geometry_evidence / "ManifestV1.json").read_text()).items():
                    if sha256(args.geometry_evidence / name) != entry["sha256"]:
                        raise ValueError("geometry evidence identity mismatch")
                prepare_reference(stack, output, write_json)
            elif args.stage == "coupled":
                from perovskite_sim.experiments.one_dimensional_mechanism_r1 import run_coupled
                if args.reference is None:
                    raise ValueError("--reference is required")
                binding = json.loads(args.reference.read_text())
                run_coupled(stack, args.intervals, binding, output, write_json,
                            nonlinear_factor=args.nonlinear_factor)
            elif args.stage == "ac":
                from perovskite_sim.experiments.defect_ion_combined_impedance import run_defect_ion_combined_impedance
                from perovskite_sim.constants import Q
                if args.reference is None:
                    raise ValueError("--reference is required")
                binding = json.loads(args.reference.read_text())
                x, mat = build_r1_material(stack, args.intervals)
                result = run_defect_ion_combined_impedance(
                    x, stack, np.logspace(-3, 8, 45), delta_V=.005,
                    refinement_factors=(1., .5, .25), research_binding=binding,
                )
                write_json(output / "RawAcV1.json", result)
                metric = mat.V_T_device*abs(result.positive_ion_storage_response_F_m2/Q)/1e15
                write_json(output / "AcInventoryV1.json", {
                    "V_T_V": mat.V_T_device, "inventory_m2": 1e15,
                    "absolute_complex_inventory_per_V": result.positive_ion_storage_response_F_m2/Q,
                    "normalized": metric, "limit": 1e-10,
                })
                if max(metric) > 1e-10:
                    raise RuntimeError("R1 AC inventory response exceeds 1e-10")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "FailureV1.json", failure)
        if getattr(exc, "result", None) is not None:
            write_json(output / "FailedResultV1.json", exc.result)
    write_json(output / "CompletionV1.json", {
        "stage": args.stage, "status": "failed" if failure else "passed",
        "duration_s": time.monotonic()-started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "failure": failure,
    })
    manifest(output)
    print(f"{args.stage}: {'failed' if failure else 'passed'}: {output}")
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())

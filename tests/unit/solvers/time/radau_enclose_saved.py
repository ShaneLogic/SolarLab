"""Enclose the pinned stage-2b polynomials; never prepare or integrate a state.

Explicit invocation only. The two saved fine grids, PWL drives and initial
inventories are immutable inputs. Failed inclusion is saved as blocked with
no error bound. A closed comparison tube still needs the reference review.

Invoke with runpy from the worktree root so the existing research-example
namespace is available alongside the explicit PYTHONPATH pointing to src.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any
from zipfile import ZipFile

from flint import ctx
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.solvers.time.radau_enclosure import (
    NonlinearInventoryEnclosure,
    PolynomialPath,
    PWLDrive,
    positive_polynomial_tube,
)

BASE = "cd4b4d286e1974b9cccfd8c22e2b2cfee545fe6b"
RESULT_SHA256 = "f1e109ccfbab8c2a991d0867fe8843972da2c3e31b19d9b2978b3a7ecf31a506"
PACKET_SHA256 = "053b825ceff415d537cd73deb623629bfcfee7b42a39495f00ddb738e3c7279b"
HISTORY_SHA256 = {
    "E_L": "bad2dc554066e59fd35d483c8a81fe6cdf950200d2e4835070d38d3154611fed",
    "I_V": "6bb2d63f162952fc7be9e083c3bb03a14cac60812ea794ceec6139cb1e22c6ec",
}
CHANGED_SOURCE = "src/solarlab/solvers/time/radau_enclosure.py"


def pin(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    result = pin(path)
    if result["sha256"] != expected:
        raise ValueError(f"saved_source_hash_mismatch:{path}")
    return result


def load_path(path: Path, case: dict[str, Any]) -> PolynomialPath:
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("saved_history_CRC_failure")
    with np.load(path, allow_pickle=False) as data:
        result = PolynomialPath(
            data["step_times"],
            data["coefficients_high"],
            data["coefficients_low"],
            data["initial_inventory_high"],
            data["initial_inventory_low"],
            data["initial_u"],
            data["endpoint_u"],
            data["component_scale"],
        )
    for actual, key in (
        (result.initial_high, "initial_inventory_high"),
        (result.initial_low, "initial_inventory_low"),
        (result.original_scale, "component_scale"),
    ):
        expected = np.asarray(case[key], dtype=float)
        if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError(f"saved_history_packet_mismatch:{key}")
    # The packet stores prepared coordinates. The pinned split producer
    # separately reconstructs its initial point from these same inventory
    # words (radau_split.py, integrate_split_radau); the NPZ stores that point.
    if result.initial_u.shape != np.asarray(case["initial"]["u"]).shape:
        raise ValueError("saved_history_initial_coordinate_shape_mismatch")
    grid = np.linspace(
        float(case["short_window_s"][0]), float(case["short_window_s"][1]), 9
    )
    if not np.array_equal(result.step_times_s, np.stack((grid[:-1], grid[1:]), axis=1)):
        raise ValueError("saved_history_grid_mismatch")
    return result


def run_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    scope = case["scope"]
    fine = next(row for row in case["histories"] if row["steps"] == 8)
    history = root / ".artifacts/RadauReferenceV1/Stage2b/Short03" / f"{scope}_8.npz"
    saved = require_hash(history, HISTORY_SHA256[scope])
    if fine["sha256"] != saved["sha256"]:
        raise ValueError("saved_history_packet_pin_mismatch")
    path = load_path(history, case)
    drive = PWLDrive(**{key: tuple(value) for key, value in case["drive"].items()})
    if {
        key: [value.hex() for value in values] for key, values in asdict(drive).items()
    } != case["drive_hex"]:
        raise ValueError("saved_drive_encoding_mismatch")
    inputs = drive(float(path.step_times_s[0, 0]))
    if scope == "E_L":
        electronic = compile_device_1d(load_device(), intervals_per_layer=(12, 12, 12))
        problem: AssembledDC1DProblem | AssembledIonic1DProblem = AssembledDC1DProblem(
            electronic, inputs.voltage_V, inputs.generation_scale
        )
    elif scope == "I_V":
        ionic = compile_ionic_device_1d(
            prepare_confined_ion_device(),
            intervals_per_layer=(12, 12, 12),
            ion_domain_layer="absorber",
            ion_domain_endpoints=("blocking", "blocking"),
        )
        problem = AssembledIonic1DProblem(
            ionic, inputs.voltage_V, inputs.generation_scale
        )
    else:
        raise ValueError("unapproved_saved_scope")
    if problem.device.identity != case["device_identity"]:
        raise ValueError("saved_compiled_device_identity_mismatch")
    enclosure = NonlinearInventoryEnclosure(
        problem, drive, path.original_scale, branch_aware=True
    )
    result = positive_polynomial_tube(
        enclosure,
        path,
        initial_log_error=0.0,
        log_radius=1e-6,
        subdivisions=2,
    )
    rows = asdict(result)
    for slab in rows["slabs"]:
        index = slab.get("dominant_log_row_approx")
        if index is not None:
            slab["dominant_block"] = (
                "electron" if index < 35 else "hole" if index < 70 else "positive_ion"
            )
            slab["dominant_block_index"] = (
                index if index < 35 else index - 35 if index < 70 else index - 70
            )
    return {
        "scope": scope,
        "electrical_nodes": 37,
        "ion_nodes": case.get("ion_nodes", 0),
        "window_s": case["short_window_s"],
        "saved_history": saved,
        "device_identity": problem.device.identity,
        "observed_refinement_delta": case["observed_refinement_delta"],
        "observation_meaning": case["observations"]["meaning"],
        "initial_error_semantics": case["initial_error_semantics"],
        "initial_coordinates_origin": "saved reconstructed split point",
        "prepared_to_reconstructed_coordinate_max_abs": float(
            np.max(np.abs(path.initial_u - np.asarray(case["initial"]["u"])))
        ),
        "bound_attempt": rows,
        "status": result.status,
        "reason": result.reason,
        "wall_s": perf_counter() - start,
        "new_steady_solves": 0,
        "new_transient_solves": 0,
        "admission_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    output = arguments.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output_must_stay_in_reference_worktree")
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    context: dict[str, Any] = {
        "schema": "solarlab.radau.saved-positive-enclosure.v1",
        "base_commit": BASE,
        "cases": [],
        "new_steady_solves": 0,
        "new_transient_solves": 0,
        "admission_runs": 0,
        "qualification": False,
        "tube": {
            "coordinates": "log(z_i/p_i)",
            "log_radius": 1e-6,
            "log_weights": "all_one",
            "subdivisions": 2,
            "initial_log_error": 0.0,
            "initial_error_scope": "exact encoded initial-inventory IVP only",
            "retained_branch_difference": "separately enclosed forcing",
        },
    }
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if head != BASE:
            raise ValueError("reference_base_commit_mismatch")
        subprocess.run(
            ["git", "diff", "--exit-code", BASE, "--", "src", "examples"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        prior = root / ".artifacts/RadauReferenceV1/Stage2b"
        context["previous_result"] = require_hash(prior / "Result.json", RESULT_SHA256)
        packet_path = prior / "Short03/ShortWindows.json"
        context["saved_packet"] = require_hash(packet_path, PACKET_SHA256)
        packet = json.loads(packet_path.read_text())
        context["saved_trajectory_source_pins"] = packet["source_pins"]
        for file, expected in packet["source_pins"].items():
            if file != CHANGED_SOURCE:
                require_hash(root / file, expected)
        context["enclosure_source_pins"] = {
            file: pin(root / file)["sha256"]
            for file in (
                CHANGED_SOURCE,
                str(Path(__file__).resolve().relative_to(root)),
            )
        }
        ctx.threads = 1
        with threadpool_limits(limits=1), ctx.workprec(192):
            context["threadpools"] = threadpool_info()
            for case in packet["cases"]:
                result = run_case(root, case)
                context["cases"].append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
        context["input_pins_unchanged_after_evaluation"] = all(
            pin(Path(row["saved_history"]["path"]))["sha256"]
            == row["saved_history"]["sha256"]
            for row in context["cases"]
        )
    except Exception as error:
        context["fatal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        context["whole_wall_s"] = perf_counter() - start
        (output / "SavedEnclosures.json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()

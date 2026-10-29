"""One bounded diagonal-weight check of pinned saved Radau polynomials.

Invoke through runpy from the reference worktree root. No steady or time
integrator is called. Floating eigen/linear algebra only proposes weights;
the propagation bounds use full-slab Arb enclosures and exact weight words.

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
import json
from pathlib import Path
import runpy
import subprocess
from time import perf_counter
from typing import Any

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
    PWLDrive,
    weighted_positive_polynomial_tube,
)

_reader = runpy.run_path(str(Path(__file__).with_name("radau_enclose_saved.py")))
pin, require_hash, load_path = (
    _reader["pin"],
    _reader["require_hash"],
    _reader["load_path"],
)
BASE: str = _reader["BASE"]
DELIVERY_SHA256 = "18908fe2241838551c95c0db4eba8375d7f0d90098fef5d0467ae15b71036fa2"
READER_SHA256 = "e13d01969b92417e27857dbdec7021ba02c4cac8ba0c8ed6cf81c8061caf1908"


def run_case(root: Path, case: dict[str, Any], output: Path) -> dict[str, Any]:
    start = perf_counter()
    scope = case["scope"]
    history = root / ".artifacts/RadauReferenceV1/Stage2b/Short03" / f"{scope}_8.npz"
    saved = require_hash(history, _reader["HISTORY_SHA256"][scope])
    fine = next(row for row in case["histories"] if row["steps"] == 8)
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
    field = NonlinearInventoryEnclosure(
        problem, drive, path.original_scale, branch_aware=True
    )
    bound = weighted_positive_polynomial_tube(
        field,
        path,
        initial_log_error=0.0,
        log_radius=1e-6,
        subdivisions=2,
    )
    data = asdict(bound)
    matrices = data.pop("comparison_matrices")
    matrix_path = output / f"{scope}_comparisons.npz"
    np.savez_compressed(
        matrix_path,
        comparison_matrices=matrices,
        selected_weights=np.array(bound.selected_weights),
        step_times=path.step_times_s,
        original_scale=path.original_scale,
    )
    result = {
        "scope": scope,
        "electrical_nodes": 37,
        "ion_nodes": case.get("ion_nodes", 0),
        "window_s": case["short_window_s"],
        "saved_history": saved,
        "device_identity": problem.device.identity,
        "observed_refinement_delta": case["observed_refinement_delta"],
        "observation_meaning": case["observations"]["meaning"],
        "initial_error_semantics": case["initial_error_semantics"],
        "bound_attempt": data,
        "comparison_artifact": pin(matrix_path),
        "status": bound.status,
        "reason": bound.reason,
        "cap_cleared": bound.best_exponent_upper is not None
        and bound.best_exponent_upper <= 50,
        "whole_window_error_bound": None
        if bound.tube_bound is None
        else bound.tube_bound.error_upper,
        "wall_s": perf_counter() - start,
        "new_steady_solves": 0,
        "new_transient_solves": 0,
        "admission_run": False,
    }
    print(
        json.dumps(
            {
                "scope": scope,
                "status": result["status"],
                "reason": result["reason"],
                "candidate": bound.selected_candidate,
                "best_exponent_upper": bound.best_exponent_upper,
                "best_first_slab_exponent_upper": bound.best_first_slab_exponent_upper,
                "enclosed_slabs": bound.enclosed_slabs,
                "wall_s": result["wall_s"],
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return result


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
        "schema": "solarlab.radau.saved-weighted-enclosure.v1",
        "base_commit": BASE,
        "cases": [],
        "new_steady_solves": 0,
        "new_transient_solves": 0,
        "admission_runs": 0,
        "qualification": False,
        "norm": "constant D=diag(1/w) on log-inventory deviations",
        "tube": {
            "log_radius": 1e-6,
            "subdivisions": 2,
            "initial_log_error": 0.0,
            "exponent_cap": 50,
        },
        "matrix_scope": "whole polynomial subslab and entire positive outer tube; no point-only Jacobians",
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
        prior = root / ".artifacts/RadauReferenceV1/Stage2c/Delivery.json"
        context["prior_delivery"] = require_hash(prior, DELIVERY_SHA256)
        delivery = json.loads(prior.read_text())
        context["prior_result"] = require_hash(
            Path(delivery["result"]["path"]), delivery["result"]["sha256"]
        )
        reader_path = Path(__file__).with_name("radau_enclose_saved.py")
        context["saved_reader"] = require_hash(reader_path, READER_SHA256)
        for row in delivery["all_files_to_integrate"]:
            if Path(row["path"]) != root / _reader["CHANGED_SOURCE"]:
                require_hash(Path(row["path"]), row["sha256"])
        packet_path = (
            root / ".artifacts/RadauReferenceV1/Stage2b/Short03/ShortWindows.json"
        )
        context["saved_packet"] = require_hash(packet_path, _reader["PACKET_SHA256"])
        packet = json.loads(packet_path.read_text())
        context["saved_trajectory_source_pins"] = packet["source_pins"]
        context["enclosure_source_pins"] = {
            file: pin(root / file)["sha256"]
            for file in (
                _reader["CHANGED_SOURCE"],
                str(Path(__file__).resolve().relative_to(root)),
            )
        }
        ctx.threads = 1
        with threadpool_limits(limits=1), ctx.workprec(192):
            context["threadpools"] = threadpool_info()
            for case in packet["cases"]:
                context["cases"].append(run_case(root, case, output))
                (output / "SavedWeights.json").write_text(
                    json.dumps(context, indent=2, allow_nan=False) + "\n"
                )
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
        (output / "SavedWeights.json").write_text(
            json.dumps(context, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()

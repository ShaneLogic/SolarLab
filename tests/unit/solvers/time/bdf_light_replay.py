"""One N-method attempt of the original frozen adaptive LIGHT protocol.

The exact saved initial is read, physically admitted by the public driver
and never cold-solved. The M failure is retained under its original SHA256.
No waveform, controller, Newton or physical tolerance is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.experiments.transient import solve_transient
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time.bdf import BDFOptions

ORIGINAL_SHA256 = "4e04d2812a8aa5a2f7d54c50acbd5bdd4d60562e8dbf709890432a8d1f60b87d"


def source_pins() -> dict[str, str]:
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("solarlab.", "examples.research_1d.")):
            filename = getattr(module, "__file__", None)
            if filename:
                result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    result[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the first N light attempt")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    raw = args.original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA256
    saved = json.loads(raw)
    old = saved["adaptive_legacy_1.0"]
    assert old["completed"] is False and old["termination"] == "rejection_budget_exhausted"
    assert sum(item["accepted"] for item in old["attempts"]) == 7
    assert sum(not item["accepted"] for item in old["attempts"]) == 15
    restore = runpy.run_path("tests/unit/solvers/time/bdf_robustness_witness.py")["restore_initial"]
    protocol = runpy.run_path("tests/unit/solvers/time/test_assembled_bdf.py")["protocol"]
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    saved_initial = saved["initial_1.0"]
    admission = SteadyOptions()
    warm_admission = None
    if saved_initial["state"]["device_sha256"] == device.identity:
        initial = restore(device, saved_initial)
    else:
        # The historical acceptance stays with its old identity. Obtain
        # actual current-source acceptance from the exact numerical fields;
        # no cold preparation, coordinate change or new tolerance is allowed.
        previous_u = np.array(saved_initial["state"]["u"])
        guess = device.state(previous_u, voltage_V=saved_initial["voltage_V"])
        for field in ("potential_V", "electron_density_m3", "hole_density_m3"):
            assert getattr(guess, field).tobytes() == np.array(saved_initial["state"][field]).tobytes()
        warm_started = time.monotonic()
        initial = solve_steady_state(device, voltage_V=saved_initial["voltage_V"],
            generation_scale=saved_initial["generation_scale"], options=admission, initial_state=guess)
        warm_admission = {"kind": "actual current-source steady acceptance of historical numerical coordinates",
            "historical_compiled_identity": saved_initial["state"]["device_sha256"],
            "current_compiled_identity": device.identity, "result": initial,
            "wall_s": time.monotonic() - warm_started, "cold_preparation_calls": 0,
            "historical_acceptance_transferred": False,
            "initial_coordinates_unchanged": initial.state.u.tobytes() == previous_u.tobytes()}
        with (args.output / "WarmAdmission.json").open("x") as stream:
            json.dump(json_value(warm_admission), stream, indent=2, allow_nan=False)
            stream.write("\n")
        assert warm_admission["initial_coordinates_unchanged"], "fresh acceptance changed the frozen initial; no light replay"
    requested, drive = protocol(1.0, np.array([0.0, .13, .31, 1.0]) * 4e-7)
    physical = controls()
    options = BDFOptions(1e-7, max_step_s=2e-7, relative_tolerance=1e-5)
    assert asdict(physical) == old["controls"]["physical_controls"]
    assert asdict(admission) == old["controls"]["initial_controls"]
    assert json_value(drive) == old["controls"]["drive"]
    assert requested.times_s.tobytes() == np.array(old["requested_output_times_s"]).tobytes()
    for name, value in asdict(options).items():
        if name not in {"wall_s", "max_steps"}:
            assert value == old["controls"][name]
    pins = source_pins()
    receipt: dict[str, Any] = {"schema": "solarlab.bdf_n_original_light_attempt.v1",
        "method_class": "N: accepted slope-knot restart and algebraic-zero handling",
        "original_artifact": str(args.original.resolve()), "original_sha256": ORIGINAL_SHA256,
        "original_M": {"completed": False, "termination": old["termination"],
                       "accepted_steps": 7, "rejected_steps": 15},
        "compiled_identity": device.identity, "initial": initial,
        "initial_state_sha256": hashlib.sha256(initial.state.u.tobytes()).hexdigest(),
        "physical_controls": physical, "initial_controls": admission, "BDF_options": options,
        "requested": requested, "drive": drive, "source_sha256": pins,
        "warm_admission": warm_admission, "cold_preparation_calls": 0, "new_adaptive_light_calls": 0,
        "physical_certificate_status": "not_evaluated", "qualification": False}
    with (args.output / "Started.json").open("x") as stream:
        json.dump(json_value(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Exactly one invocation. A failed result is saved without a retry.
    result = solve_transient(device, initial, requested, options=physical, initial_options=admission,
        method="bdf", bdf_options=options, bdf_drive=drive)
    receipt.update(result=result, new_adaptive_light_calls=1, source_pins_unchanged=source_pins() == pins,
        process_wall_s=time.monotonic() - started)
    with (args.output / "LightAttempt.json").open("x") as stream:
        json.dump(json_value(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {"completed": result.completed, "termination": result.termination,
        "accepted_steps": len(result.accepted_steps), "rejected_steps": len(result.rejected_steps),
        "accepted_orders": [item.order for item in result.accepted_steps],
        "restarts_s": [event.time_s for event in result.restarts], "samples": len(result.samples),
        "accepted_final_s": float(result.accepted_times_s[-1]), "requested_final_s": result.final_time_s,
        "solver_wall_s": result.wall_s, "process_wall_s": receipt["process_wall_s"],
        "nonlinear_calls": result.nonlinear_calls, "new_adaptive_light_calls": 1,
        "warm_admission_calls": int(warm_admission is not None),
        "warm_admission_wall_s": 0.0 if warm_admission is None else warm_admission["wall_s"],
        "cold_preparation_calls": 0, "original_M_failure_sha256": ORIGINAL_SHA256,
        "result_sha256": hashlib.sha256((args.output / "LightAttempt.json").read_bytes()).hexdigest(),
        "error": result.error, "qualification": False}
    with (args.output / "Summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    assert receipt["source_pins_unchanged"]
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

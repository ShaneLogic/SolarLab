"""One N-method replay of the frozen dark drive, using its saved DC initial."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import numpy as np


def restore_initial(device, raw):
    from solarlab.solvers.nonlinear.newton import IterationRecord, NewtonResult, Termination
    from solarlab.solvers.nonlinear.problem import GateOutcome
    from solarlab.solvers.nonlinear.steady_1d import SteadyState1D
    def newton(value):
        return NewtonResult(**{**value, "state": np.array(value["state"]),
            "termination": Termination(value["termination"]),
            "gates": tuple(GateOutcome(**gate) for gate in value["gates"]),
            "iterations": tuple(IterationRecord(**row) for row in value["iterations"])})
    data = raw["state"]
    assert data["device_sha256"] == device.identity
    state = device.state(np.array(data["u"]), voltage_V=data["voltage_V"])
    for field in ("potential_V", "electron_density_m3", "hole_density_m3"):
        assert getattr(state, field).tobytes() == np.array(data[field]).tobytes()
    return SteadyState1D(**{**raw, "state": state, "newton": newton(raw["newton"]),
        "preparation": tuple(newton(value) for value in raw["preparation"]),
        "current_density_A_m2": np.array(raw["current_density_A_m2"]),
        "contact_current_density_A_m2": np.array(raw["contact_current_density_A_m2"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    from solarlab.solvers.time.bdf import BDFOptions
    from solarlab.experiments.transient import solve_transient
    from examples.research_1d.transient import controls
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
    raw_bytes = (args.root / "NativeParity01.json").read_bytes()
    raw = json.loads(raw_bytes)
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial = restore_initial(device, raw["initial_0.0"])
    module = runpy.run_path("tests/unit/solvers/time/test_assembled_bdf.py")
    output = args.root / "NMethod/DarkReplay01.json"
    if output.exists():
        raise FileExistsError("preserve the first N-method dark attempt")
    os.environ["SOLARLAB_BDF_ASSEMBLY_OUTPUT"] = str(output.resolve())
    module["record"]("initial_provenance", {
        "original_raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "compiled_identity": device.identity, "new_cold_solves": 0,
        "method_class": "N: slope-knot history restart and algebraic residual words",
        "original_failure_preserved": True})
    requested, drive = module["protocol"](0.0, np.array([0.0, .13, .31, 1.0]) * 4e-7)
    settings = BDFOptions(1e-7, max_step_s=2e-7, relative_tolerance=1e-5)
    original = solve_transient(device, initial, requested, options=controls(), initial_options=SteadyOptions(),
        method="bdf", bdf_options=settings, bdf_drive=drive)
    module["record"]("N_legacy_dark", original)
    assembled = module["run_assembled"](device, initial, requested, drive, settings)
    module["record"]("N_assembled_dark", assembled)
    module["assert_bdf"](original, assembled, device, drive)
    summary = {"completed": assembled.completed, "termination": assembled.termination,
        "accepted_steps": len(assembled.accepted_steps), "rejected_steps": len(assembled.rejected_steps),
        "accepted_orders": [step.order for step in assembled.accepted_steps],
        "restarts": [{"time_s": event.time_s, "name": event.name,
                      "unchanged_state": event.left_u.tobytes() == event.right_u.tobytes()}
                     for event in assembled.restarts],
        "dense_samples": len(assembled.samples), "identical_legacy_assembled": True,
        "new_cold_solves": 0, "qualification": False}
    module["record"]("N_summary", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

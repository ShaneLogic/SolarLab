"""Bounded R1 numerical probes; diagnostic output never grants acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import time

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for name in THREAD_VARIABLES:
    os.environ[name] = "1"

import numpy as np
import scipy
import perovskite_sim
from threadpoolctl import threadpool_info

from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy, run_r1_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.models.config_loader import load_device_from_yaml


class SolverTrace:
    """Observe solver locals without replacing equations or acceptance tests."""
    def __init__(self):
        self.failures = []

    def trace(self, frame, event, arg):
        try:
            return self._trace(frame, event, arg)
        except Exception as exc:
            # Instrumentation must never replace the numerical failure.
            self.failures.append({"function": frame.f_code.co_name,
                                  "diagnostic_error": str(exc),
                                  "original_solver_error": str(arg[1]) if event == "exception" else None})
            return self.trace

    def _trace(self, frame, event, arg):
        if frame.f_code.co_name not in ("_solve_local_carriers", "_solve_step"):
            return None
        if event == "exception":
            values = frame.f_locals
            message = str(arg[1])
            if "stalled" not in message and "exceeded" not in message:
                return self.trace
            residual = np.asarray(values["residual"])
            scale = values.get("scale", values.get("local_scale"))
            state = values.get("state")
            coordinate = values["coordinate"] if frame.f_code.co_name == "_solve_local_carriers" else values["trial"]
            step = values.get("step", values.get("delta"))
            largest = np.argsort(np.abs(residual))[-8:][::-1]
            record = {"function": frame.f_code.co_name, "message": message,
                      "iteration": values.get("iteration"),
                      "largest_residual_rows": largest.tolist(),
                      "largest_residual_values": residual[largest].tolist(),
                      "coordinate_spacing_max": float(np.max(np.spacing(np.abs(coordinate)))),
                      "step_abs_max": float(np.max(np.abs(step))),
                      "step_abs_min_nonzero": float(np.min(np.abs(step)[step != 0.])),
                      "local_scale": np.asarray(scale).tolist(),
                      "local_residual": state.local_residual.tolist(),
                      "trace_densities": [s.state_m3.tolist() for s in state.local],
                      "trace_log_spacing": [np.spacing(s.log_state).tolist() for s in state.local],
                      "local_newton_step": np.asarray(step)[-6:].tolist(),
                      "dt_s": values.get("dt")}
            self.failures.append(record)
        return self.trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="N:factor:time-base:control")
    parser.add_argument("--trace", action="store_true", help="Record failed Newton frame diagnostics")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to replace existing diagnostic: {args.output}")
    root = Path(perovskite_sim.__file__).resolve().parents[1]
    stack = load_device_from_yaml(root / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = json.loads((root / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    cases = args.case or ["64:0.01:1:D", "32:0.01:1:D", "64:1:1:D"]
    preparations = {}
    report = {"schema": "R1NumericalDiagnosisV1", "scope": "development_diagnostic_only", "cases": []}
    report["runtime"] = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                         "platform": platform.platform(), "blas": threadpool_info(),
                         "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES}}
    report["source_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                               for name in (
                                   "perovskite_sim/physics/two_sided_interface.py",
                                   "perovskite_sim/experiments/interface_defect_transient.py",
                                   "perovskite_sim/experiments/interface_defect_ion_transient.py",
                                   "perovskite_sim/experiments/one_dimensional_mechanism_r1_dynamics.py",
                               )}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in cases:
        n, factor, base, control = key.split(":")
        n, factor, base = int(n), float(factor), int(base)
        if n not in preparations:
            preparations[n] = prepare_common_state(stack, n, binding, policy=r1_policy())
        start = time.monotonic()
        policy = r1_policy(factor, time_substeps=(base, 2*base, 4*base))
        row = {"case": key, "intervals": n, "nonlinear_factor": factor,
               "time_substeps": list(policy.refinement_substeps), "control": control}
        try:
            observer = SolverTrace()
            if args.trace:
                sys.settrace(observer.trace)
            result = run_r1_step(stack, n, binding, preparations[n], control=control, policy=policy)
            row.update(status="completed", certificate=result["certificate"])
        except Exception as exc:
            result = getattr(exc, "result", {}) or {}
            row.update(status="failed", error_type=type(exc).__name__, message=str(exc),
                       failure=result.get("failure"))
        finally:
            sys.settrace(None)
        row["solver_failures"] = observer.failures
        records = result.get("accepted_steps", [])
        row["accepted_count"] = len(records)
        row["last_rows_by_substeps"] = {}
        for count in policy.refinement_substeps:
            subset = [s for s in records if s.get("substeps") == count]
            if subset:
                last = subset[-1]
                row["last_rows_by_substeps"][str(count)] = {
                    k: last.get(k) for k in ("time_s", "dt_s", "scaled_nonlinear_residual",
                                             "physical_checks_passed", "physical_failure_reasons")}
                row["last_rows_by_substeps"][str(count)]["physical_checks"] = last.get("physical_checks")
        row["elapsed_s"] = time.monotonic() - start
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: row[k] for k in ("case", "status", "accepted_count", "elapsed_s")}) +
              (" " + row["message"] if "message" in row else ""), flush=True)


if __name__ == "__main__":
    main()

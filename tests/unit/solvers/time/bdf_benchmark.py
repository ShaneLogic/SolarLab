"""Bounded reproducible fit data and work-precision evidence (no legacy solve).

Run one mode per supervised command with BLAS1. Each result is updated after
every solve, preserving the first failure and all completed measurements.
"""

from dataclasses import fields, replace
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls as electronic_controls, json_value
from examples.research_1d.ionic_transient import controls as ionic_controls
from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_steady import controls as steady_ionic_controls
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.experiments.ionic_transient import load_ionic_steady_initial, solve_ionic_transient
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.time.bdf import BDFOptions, solve_bdf
from solarlab.solvers.time.bdf_1d import BDFDrive, ElectronicStorage1D, IonicStorage1D
from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.newton import NewtonResult, IterationRecord, Termination
from solarlab.solvers.nonlinear.problem import ResidualScaling, StepControl, SolveBudget, GateOutcome


def generic_controls(step):
    count = len(step.history[0].u)
    return NewtonSettings(ResidualScaling(np.ones(count)), 1e-12,
        StepControl(np.full(count, 1e30)), SolveBudget(20, 100, 20, 10), (), False)


def baseline_driver(name):
    """Actual base-commit driver source with byte-verified unchanged BE laws."""
    base = "c7b3fa61e2011b534774a49b0562ad437acb2253"
    for file in ("backward_euler.py", "ionic_backward_euler.py", "multiple_ionic_backward_euler.py"):
        path = "src/solarlab/solvers/time/" + file
        raw = subprocess.check_output(["git", "show", base + ":" + path])
        assert raw == Path(path).read_bytes(), "BE baseline source changed"
    path = "src/solarlab/experiments/" + name + ".py"
    raw = subprocess.check_output(["git", "show", base + ":" + path])
    module = types.ModuleType("base_c7_" + name)
    module.__file__ = "c7b3fa61:" + path
    sys.modules[module.__name__] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(raw).hexdigest()


def restore_electronic_initial(device, raw):
    def newton(item):
        data = dict(item)
        data["state"] = np.asarray(data["state"])
        data["termination"] = Termination(data["termination"])
        data["gates"] = tuple(GateOutcome(**g) for g in data["gates"])
        data["iterations"] = tuple(IterationRecord(**g) for g in data["iterations"])
        return NewtonResult(**data)
    from solarlab.solvers.nonlinear.steady_1d import SteadyState1D
    data = dict(raw)
    state = device.state(np.asarray(data["state"]["u"]), voltage_V=data["voltage_V"])
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        np.testing.assert_array_equal(getattr(state, name), data["state"][name])
    data["state"] = state
    data["newton"] = newton(data["newton"])
    data["preparation"] = tuple(newton(g) for g in data["preparation"])
    return SteadyState1D(**data)


def write(path, record):
    path.write_text(json.dumps(json_value(record), indent=2, allow_nan=False) + "\n")


def fit(ns, values):
    h = 1 / np.asarray(ns, dtype=float)
    error = np.asarray(values, dtype=float)
    return {"h_relative": h, "errors": error, "observed_order": float(np.polyfit(np.log(h), np.log(error), 1)[0])}


def analytic(output):
    class Modes:
        weights = np.array([1.0, 2.0, .5])
        rates = np.array([1.0, 3.0, 7.0])
        def storage(self, u):
            return self.weights * u
        def storage_tangent(self, u):
            return np.diag(self.weights)
        def residual(self, u):
            return -self.rates * self.storage(u)
        def jacobian(self, u):
            return -np.diag(self.rates * self.weights)
    model = Modes()
    record = {"case": "three analytic RC/diffusion modes", "rates_s1": model.rates,
              "capacity_weights": model.weights, "runs": [], "fits": {}, "qualification": False}
    for order in (1, 2):
        errors, ns = [], [16, 32, 64, 128]
        for n in ns:
            result = solve_bdf(lambda t, side: model, np.ones(3), (0.0, 1.0),
                options=BDFOptions(1 / n, max_order=order, adaptive=False), settings=generic_controls)
            error = float(np.max(abs(result.accepted_coordinates[-1] - np.exp(-model.rates))))
            errors.append(error)
            record["runs"].append({"method": "BDF" + str(order), "n": n, "error": error, "result": result})
            write(output, record)
            if not result.completed:
                raise RuntimeError("first analytic failure retained")
        record["fits"][str(order)] = fit(ns, errors)
    write(output, record)
    return record


class QuadraticDrive(BDFDrive):
    def values(self, t, side="right"):
        return 0.0, 1.0 + .01 * (t / 4e-7)**2


class CubicDiagnosticDrive(BDFDrive):
    def values(self, t, side="right"):
        return 0.0, 1.0 + .1 * (t / 4e-7)**3


def diagnostic(output):
    original_path = output.with_name("Electronic.json")
    original = json.loads(original_path.read_bytes())
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial = restore_electronic_initial(device, original["initial"])
    admission = SteadyOptions()
    controls = replace(electronic_controls(), max_steps=4096, wall_s=90.0)
    drive = CubicDiagnosticDrive(np.array([0, 4e-7]), np.array([[0, 1], [0, 1.1]]), np.array([[0, 1], [0, 1.1]]))
    request = TransientProtocol([0, 4e-7], [0, 0], [1, 1.1])
    record = {"case": "same37-node400ns device; labelled smooth cubic diagnostic", "runs": [],
        "diagnostic_manufactured_forcing": "generation_scale=1+0.1*(t/4e-7)^3; not a product protocol",
        "initial_layer_policy": "reuse accepted steady g=1; g_prime(0)=g_second(0)=0",
        "initial_source_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "new_initial_solves": 0, "physical_controls": controls, "fits": {}, "qualification": False}
    finals = []
    ns = [2, 4, 8, 16, 32, 64, 128]
    for n in ns:
        result = solve_transient(device, initial, request, options=controls, initial_options=admission,
            method="bdf", bdf_options=BDFOptions(4e-7 / n, adaptive=False, max_order=2,
                                                wall_s=90, min_step_s=1e-23), bdf_drive=drive)
        value = ElectronicStorage1D(DC1DProblem(device, 0, 1.1)).physical_storage(result.accepted_coordinates[-1])
        finals.append(value)
        record["runs"].append({"n": n, "final_inventory": value, "result": result})
        write(output, record)
        if not result.completed:
            raise RuntimeError("first diagnostic failure retained: " + result.termination)
    differences = [float(np.max(abs(a - b))) for a, b in zip(finals[:-1], finals[1:])]
    floor = max(float(original["fits"]["2"]["errors"][-1]), differences[-1])
    eligible = [index for index, error in enumerate(differences) if error >= 10 * floor]
    record["precision_floor_inventory_units"] = floor
    record["all_refinement_differences"] = differences
    record["eligible_step_counts"] = [ns[index] for index in eligible]
    record["order_interpretation_gate"] = "differences must exceed measured floor by>=10x"
    if len(eligible) >= 3:
        record["fits"]["2"] = fit([ns[index] for index in eligible], [differences[index] for index in eligible])
    else:
        record["order_unresolved"] = "fewer than three differences above10x measured floor"
    record["original_quadratic_status"] = "order_unresolved: final precision floor and startup/stiff reduction; neither passed nor failed"
    baseline, source_sha = baseline_driver("transient")
    t = np.linspace(0, 4e-7, 5)
    protocol = baseline.TransientProtocol(t, np.zeros(5), 1 + .01 * (t / t[-1])**2)
    old_controls = baseline.TransientOptions(**{f.name: getattr(controls, f.name) for f in fields(controls)})
    old = baseline.solve_transient(device, initial, protocol, options=old_controls, initial_options=admission)
    candidate_coordinates = original["BE_parity"]["candidate"]["accepted_coordinates"]
    record["c7_electronic_BE_parity"] = {"baseline_driver_sha256": source_sha,
        "steppers_byte_verified_unchanged": True,
        "maximum_coordinate_difference": max(float(np.max(abs(a.u - np.asarray(b)))) for a, b in zip(old.states, candidate_coordinates)),
        "bitwise_coordinates": all(a.u.tobytes() == np.asarray(b,dtype=float).tobytes() for a,b in zip(old.states,candidate_coordinates)),
        "baseline": old}
    write(output, record)
    return record


def electronic(output):
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    admission = SteadyOptions()
    initial = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0, options=admission)
    control = replace(electronic_controls(), max_steps=4096, wall_s=90.0)
    drive = QuadraticDrive(np.array([0.0, 4e-7]), np.array([[0, 1], [0, 1.01]]), np.array([[0, 1], [0, 1.01]]))
    record = {"case": "existing37-node electronic400ns light transient", "initial": initial,
        "compiled_identity": device.identity, "exact_generation_law": "1+0.01*(t/4e-7)^2 evaluated at every endpoint/trial",
        "physical_controls": control, "runs": [], "fits": {}, "qualification": False}
    finals = {1: [], 2: []}
    ns = [8, 16, 32, 64]
    for order in (1, 2):
        for n in ns:
            requested = TransientProtocol([0.0, 4e-7], [0, 0], [1, 1.01])
            result = solve_transient(device, initial, requested, options=control, initial_options=admission,
                method="bdf", bdf_options=BDFOptions(4e-7 / n, max_order=order, adaptive=False,
                                                    min_step_s=1e-23, wall_s=90), bdf_drive=drive)
            adapter = ElectronicStorage1D(DC1DProblem(device, 0.0, 1.01))
            final = adapter.physical_storage(result.accepted_coordinates[-1])
            finals[order].append(final)
            record["runs"].append({"method": "BDF" + str(order), "n": n, "final_physical_inventory": final, "result": result})
            write(output, record)
            if not result.completed:
                raise RuntimeError("first electronic failure retained: " + result.termination)
        differences = [float(np.max(abs(a - b))) for a, b in zip(finals[order][:-1], finals[order][1:])]
        record["fits"][str(order)] = fit(ns[:-1], differences)
    times = np.linspace(0, 4e-7, 5)
    requested = TransientProtocol(times, np.zeros(5), 1 + .01 * (times / times[-1])**2)
    baseline = solve_transient(device, initial, requested, options=control, initial_options=admission)
    candidate = solve_transient(device, initial, requested, options=control, initial_options=admission,
        method="bdf", bdf_options=BDFOptions(1e-7, max_order=1, adaptive=False))
    differences = [float(np.max(abs(a.u - b))) for a, b in zip(baseline.states, candidate.accepted_coordinates)]
    record["BE_parity"] = {"maximum_coordinate_difference": max(differences),
        "bitwise_coordinates": all(a.u.tobytes() == b.tobytes() for a, b in zip(baseline.states, candidate.accepted_coordinates)),
        "baseline": baseline, "candidate": candidate}
    write(output, record)
    return record


def ionic(output):
    device = compile_ionic_device_1d(prepare_confined_ion_device(diffusion_only=True), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    admission = steady_ionic_controls(device.ion_node_count)
    point = solve_ionic_steady_state(device, voltage_V=.002, generation_scale=0.0, options=admission)
    raw = (json.dumps(json_value({"summary": {"accepted": point.accepted, "compiled_identity": device.identity,
        "voltage_V": .002, "generation_scale": 0.0}, "result": point}), allow_nan=False) + "\n").encode()
    initial_path = output.with_name("IonicInitial.json")
    with initial_path.open("xb") as stream:
        stream.write(raw)
    initial = load_ionic_steady_initial(device, initial_path, hashlib.sha256(raw).hexdigest(), controls=admission)
    control = replace(ionic_controls(), electronic=replace(ionic_controls().electronic, max_steps=4096, wall_s=90.0))
    drive = BDFDrive(np.array([0.0, 1.0]), np.array([[.002, 0], [.004, 0]]), np.array([[.002, 0], [.004, 0]]))
    sample = TransientProtocol([0.0, 1.0], [.002, .004], [0, 0])
    adapter = IonicStorage1D(Ionic1DProblem(device, .004, 0.0))
    record = {"case": "existing37/13 confined-ion2-to4mV/1s transient", "compiled_identity": device.identity,
        "initial_path": str(initial_path), "initial_sha256": hashlib.sha256(raw).hexdigest(),
        "physical_controls": control, "references": [], "work_precision": [], "qualification": False}
    baseline, baseline_sha = baseline_driver("ionic_transient")
    baseline_initial = baseline.IonicTransientInitial(**{f.name: getattr(initial, f.name) for f in fields(initial)})
    baseline_control = baseline.IonicTransientOptions(control.electronic, control.ion_balance_tolerance_m2_s, control.ion_logit_step)
    record["BE_baseline"] = {"commit": "c7b3fa61e2011b534774a49b0562ad437acb2253", "driver_sha256": baseline_sha,
                             "steppers_verified_byte_unchanged": True}
    references = []
    for n in (64, 128):
        result = solve_ionic_transient(device, initial, sample, options=control, method="bdf",
            bdf_options=BDFOptions(1 / n, adaptive=False, max_order=2, wall_s=90, min_step_s=1e-23), bdf_drive=drive)
        final = adapter.physical_storage(result.accepted_coordinates[-1])
        references.append(final)
        record["references"].append({"method": "fixed BDF2", "n": n, "final_inventory": final, "result": result})
        write(output, record)
        if not result.completed:
            raise RuntimeError("first ionic reference failure retained: " + result.termination)
    reference = references[-1]
    scales = np.maximum(abs(reference), 1.0)
    record["reference_refinement_difference_relative"] = float(np.max(abs(references[-1] - references[-2]) / scales))
    record["reference_kind"] = "accepted128-step BDF2;64/128 refinement uncertainty retained, no extrapolated state accepted"
    for n in (4, 8, 16, 32):
        t = np.linspace(0, 1, n + 1)
        protocol = TransientProtocol(t, .002 + .002 * t, np.zeros(n + 1))
        result = baseline.solve_ionic_transient(device, baseline_initial, protocol, options=baseline_control)
        final = adapter.physical_storage(result.states[-1].u)
        error = float(np.max(abs(final - reference) / scales))
        record["work_precision"].append({"method": "fixed BE", "requested_steps": n,
            "accepted_steps": len(result.states) - 1, "wall_s": result.wall_s, "achieved_relative_inventory_error": error,
            "result": result})
        write(output, record)
        if not result.completed:
            raise RuntimeError("first fixed BE comparison failure retained")
        if n == 4:
            parity = solve_ionic_transient(device, initial, protocol, options=control, method="bdf",
                bdf_options=BDFOptions(.25, adaptive=False, max_order=1))
            record["BE_parity"] = {"maximum_coordinate_difference": max(float(np.max(abs(a.u - b)))
                for a, b in zip(result.states, parity.accepted_coordinates)), "bitwise_coordinates": all(a.u.tobytes() == b.tobytes()
                for a, b in zip(result.states, parity.accepted_coordinates)), "candidate": parity}
    for tolerance in (1e-4, 1e-5, 1e-6):
        result = solve_ionic_transient(device, initial, sample, options=control, method="bdf",
            bdf_options=BDFOptions(.1, relative_tolerance=tolerance, max_step_s=.5, wall_s=90), bdf_drive=drive)
        final = adapter.physical_storage(result.accepted_coordinates[-1])
        error = float(np.max(abs(final - reference) / scales))
        record["work_precision"].append({"method": "adaptive BDF1/2", "error_rtol": tolerance,
            "accepted_steps": len(result.accepted_steps), "rejected_steps": len(result.rejected_steps),
            "nonlinear_calls": result.nonlinear_calls, "wall_s": result.wall_s,
            "achieved_relative_inventory_error": error, "result": result})
        write(output, record)
        if not result.completed:
            raise RuntimeError("first adaptive ionic comparison failure retained: " + result.termination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analytic", "electronic", "ionic", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the first benchmark result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    with threadpool_limits(limits=1):
        value = {"analytic": analytic, "electronic": electronic, "ionic": ionic, "diagnostic": diagnostic}[args.mode](args.output)
    print(json.dumps({"case": value["case"], "fits": json_value(value.get("fits", {})),
        "work_precision": [{k: v for k, v in row.items() if k != "result"} for row in value.get("work_precision", [])],
        "BE_parity": {k: v for k, v in value.get("BE_parity", {}).items() if k not in ("baseline", "candidate")},
        "reference_uncertainty": value.get("reference_refinement_difference_relative"),
        "wall_s": time.monotonic() - t}, indent=2))


if __name__ == "__main__":
    main()

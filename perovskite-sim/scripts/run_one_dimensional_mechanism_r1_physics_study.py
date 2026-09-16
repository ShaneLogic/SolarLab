#!/usr/bin/env python3
"""Resumable DEVELOPMENT study of R1 physical consistency, with durable failures.

Use explicit --section values. Matrix cases use the short diagnostic window;
their completion is not the full-window 27-case R1-2 acceptance. This runner
does not claim the controlled launcher's formal source-execution assurance.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from collections.abc import Mapping

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
    require_r1_checkout, record_frozen_source,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    R1Response, base_convergence_cases, observation_times, compare_responses,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
    DEFAULT_TIMES_S, check_zero_excitation, r1_policy, run_r1_step,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    R1PreparedState, execution_source, prepare_common_state, restore_common_state,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    solve_controlled_dc, dc_conductance_study, small_signal_response, compare_transient_tail,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import (
    spatial_responses_from_step, compare_spatial_responses,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


SECTIONS = ("prepare", "zero", "short", "matrix", "long", "dc", "ac", "amplitude", "compare")
FREQUENCIES = np.r_[0., np.logspace(-3, 8, 45)]


class CaseBudgetExhausted(RuntimeError):
    """Explicit bounded work stopped before a required new case."""


def ready(value):
    if dataclasses.is_dataclass(value):
        return {f.name: ready(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, np.generic):
        return ready(value.item())
    if isinstance(value, complex):
        return {"real": ready(value.real), "imag": ready(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return {"value": None, "reason": "nonfinite_numeric_evidence", "representation": repr(value)}
    return value


def raw_json(value):
    return (json.dumps(ready(value), indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def write_json(path, value):
    path = Path(path)
    raw = raw_json(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(directory):
    write_json(directory/"ManifestV1.json", {
        file.relative_to(directory).as_posix(): sha(file)
        for file in sorted(directory.rglob("*"))
        if file.is_file() and file.name != "ManifestV1.json"
    })


def checked_read(directory, filename):
    manifest = json.loads((directory/"ManifestV1.json").read_text())
    for relative, expected in manifest.items():
        path = directory/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("saved case manifest mismatch: "+str(path))
    if filename not in manifest:
        raise ValueError("saved case does not bind "+filename)
    return json.loads((directory/filename).read_text())


class Study:
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        self.context = require_r1_checkout(project=PROJECT)
        self.source = execution_source()
        self.binding = json.loads(args.reference.read_text())
        self.stack = load_device_from_yaml(args.fixture)
        self.started = datetime.now(timezone.utc).isoformat()
        self.rows, self.attempted = [], 0
        self.preparations = {}
        self.dc_baselines = {}
        request = {
            "schema": "R1PhysicsStudyRequestV1", "run_class": "development",
            "source": self.source, "runner_sha256": sha(Path(__file__)),
            "fixture_sha256": sha(args.fixture), "reference_sha256": sha(args.reference),
            "short_times_s": DEFAULT_TIMES_S, "full_times_s": observation_times(),
            "frequency_Hz": FREQUENCIES,
            "short_amplitude_ladder_V": [.01, .005, .0025],
            "study_exit_passed": False,
            "scope": "development_equation_recomputation_and_bounded_physics_studies",
            "matrix_scope": "27 independent settings in short window; full-window convergence remains required",
        }
        if self.output.exists():
            if not args.resume:
                raise ValueError("output exists; use --resume to verify and extend it")
            existing = json.loads((self.output/"StudyRequestV1.json").read_text())
            if existing != ready(request):
                raise ValueError("resume source, runner, inputs or fixed study request changed")
        else:
            self.output.mkdir(parents=True)
            write_json(self.output/"StudyRequestV1.json", request)
            write_json(self.output/"EnvironmentV1.json", {
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "threads": {key: os.environ[key] for key in THREAD_VARIABLES},
            })
            record_frozen_source(self.output, self.context)

    def source_unchanged(self):
        if execution_source() != self.source:
            raise RuntimeError("source changed during the study; start a new evidence directory")

    def selected(self, key):
        return not self.args.case_filter or self.args.case_filter in key

    def latest(self, key):
        directory = self.output/key
        attempts = sorted(directory.glob("AttemptV*"), key=lambda p: int(p.name.removeprefix("AttemptV")))
        return attempts[-1] if attempts else None

    def saved(self, key):
        directory = self.latest(key)
        if directory is None:
            raise ValueError("required study case has not run: "+key)
        completion = checked_read(directory, "CompletionV1.json")
        if completion["status"] != "completed":
            raise ValueError("required study case failed: "+key)
        return checked_read(directory, "ResultV1.json")

    def case(self, key, request, operation, *, dependency=False):
        if not dependency and not self.selected(key):
            return None
        prior = self.latest(key)
        if prior is not None:
            sealed = (prior/"ManifestV1.json").is_file() and (prior/"CompletionV1.json").is_file()
            old_request = (checked_read(prior, "RequestV1.json") if sealed
                           else json.loads((prior/"RequestV1.json").read_text()))
            if old_request != ready(request):
                raise ValueError("resume case request changed: "+key)
            completion = checked_read(prior, "CompletionV1.json") if sealed else {"status": "interrupted"}
            if sealed and (completion["status"] == "completed" or not self.args.retry_failed):
                self.rows.append({"case": key, "status": completion["status"], "resumed": True,
                                  "scientific_checks_passed": completion.get("scientific_checks_passed"),
                                  "directory": str(prior.relative_to(self.output))})
                return checked_read(prior, "ResultV1.json") if completion["status"] == "completed" else None
            if not sealed:
                self.rows.append({"case": key, "status": "interrupted_attempt_preserved",
                                  "directory": str(prior.relative_to(self.output))})
        if self.args.max_cases is not None and self.attempted >= self.args.max_cases:
            self.rows.append({"case": key, "status": "not_started_case_budget"})
            return None
        dependency_error = None
        if not key.startswith("Preparation/") and "intervals" in request:
            try:
                self.prepared(request["intervals"])
            except Exception as exc:
                dependency_error = exc
            if self.args.max_cases is not None and self.attempted >= self.args.max_cases:
                self.rows.append({"case": key, "status": "not_started_case_budget"})
                return None
        self.source_unchanged()
        attempt = 1 if prior is None else int(prior.name.removeprefix("AttemptV"))+1
        directory = self.output/key/f"AttemptV{attempt}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory/"RequestV1.json", request)
        start = time.monotonic()
        self.attempted += 1
        print("START", key, flush=True)
        try:
            if dependency_error is not None:
                raise dependency_error
            result = operation(directory)
            self.source_unchanged()
            write_json(directory/"ResultV1.json", result)
            status, failure = ("unavailable" if isinstance(result, dict) and result.get("comparison_available") is False
                               else "completed"), None
        except Exception as exc:
            status = "failed"
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "partial_result": getattr(exc, "result", None)}
            write_json(directory/"FailureV1.json", failure)
            result = None
        completion = {
            "schema": "R1PhysicsStudyCaseV1", "case": key, "status": status,
            "duration_s": time.monotonic()-start, "run_class": "development",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "study_exit_passed": False, "scope": request.get("scope"),
            "scientific_checks_passed": scientific_checks(result) if status in ("completed", "unavailable") else False,
            "failure": None if failure is None else {k: v for k, v in failure.items() if k != "partial_result"},
        }
        write_json(directory/"CompletionV1.json", completion)
        seal(directory)
        self.rows.append({"case": key, "status": status, "duration_s": completion["duration_s"],
                          "scientific_checks_passed": completion["scientific_checks_passed"],
                          "directory": str(directory.relative_to(self.output))})
        print(status.upper(), key, f"{completion['duration_s']:.3f}s", flush=True)
        return result

    def prepared(self, intervals):
        if intervals not in self.preparations:
            result = self.case(f"Preparation/N{intervals}", {"intervals": intervals, "scope": "common_D_equilibrium"},
                               lambda directory: prepare_common_state(self.stack, intervals, self.binding).to_dict(),
                               dependency=True)
            if result is None:
                if self.args.max_cases is not None and self.attempted >= self.args.max_cases:
                    raise CaseBudgetExhausted("new-case budget exhausted before required preparation")
                raise ValueError("preparation unavailable within selected case budget")
            self.preparations[intervals] = R1PreparedState.from_dict(result)
        return self.preparations[intervals]

    def step(self, directory, intervals, control, substeps, factor, times, amplitude=.005):
        prepared = self.prepared(intervals)
        trajectory = directory/"AcceptedStepsV1.jsonl"
        def observe(row):
            # A complete row is durably appended before control returns to
            # the solver. This file survives solver errors and interruption.
            with trajectory.open("ab") as stream:
                stream.write(json.dumps(ready(row), separators=(",", ":"), allow_nan=False).encode()+b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        try:
            result = run_r1_step(self.stack, intervals, self.binding, prepared, control=control,
                                 amplitude_V=amplitude, times_s=times,
                                 policy=r1_policy(factor, time_substeps=substeps),
                                 accepted_step_observer=observe, physics_evidence=True)
        except Exception as exc:
            partial = getattr(exc, "result", None)
            if isinstance(partial, dict) and partial.get("physics_reconstruction") and partial.get("accepted_steps"):
                try:
                    audit = verify_r1_step_physics(self.stack, intervals, self.binding, prepared,
                                                   partial, allow_incomplete=True)
                    write_json(directory/"FailedPrefixPhysicsAuditV1.json", audit)
                except Exception as audit_exc:
                    write_json(directory/"FailedPrefixPhysicsAuditFailureV1.json", {
                        "type": type(audit_exc).__name__, "message": str(audit_exc),
                        "partial_result": getattr(audit_exc, "result", None),
                    })
            raise
        write_json(directory/"PhysicsRecomputationV1.json", verify_r1_step_physics(
            self.stack, intervals, self.binding, prepared, result))
        return result

    def run(self, sections):
        if "prepare" in sections:
            for n in (16, 32, 64):
                if self.selected(f"Preparation/N{n}"):
                    self.prepared(n)
        if "zero" in sections:
            for n in (16, 32, 64):
                self.case(f"Zero/N{n}", {"intervals": n, "controls": "ABCD", "scope": "zero_excitation_A_D"},
                          lambda directory, n=n: check_zero_excitation(self.stack, n, self.binding, self.prepared(n)))
        if "short" in sections:
            for n in (16, 32, 64):
                for control in "ABCD":
                    self.case(f"Short/N{n}/{control}", {"intervals": n, "control": control,
                              "times_s": DEFAULT_TIMES_S, "time_substeps": (1, 2, 4), "nonlinear_factor": .1,
                              "scope": "short_controlled_physics_recomputation"},
                              lambda directory, n=n, c=control: self.step(directory, n, c, (1, 2, 4), .1, DEFAULT_TIMES_S))
        if "matrix" in sections:
            for item in base_convergence_cases():
                key = f"Matrix/N{item.intervals}/T{item.time_substeps[0]}/F{str(item.nonlinear_factor).replace('.', 'p')}"
                self.case(key, {**dataclasses.asdict(item), "times_s": DEFAULT_TIMES_S,
                               "scope": "short_window_27_axis_functional_case_not_full_window_acceptance"},
                          lambda directory, item=item: self.step(directory, item.intervals, "D", item.time_substeps,
                                                                 item.nonlinear_factor, DEFAULT_TIMES_S))
        if "long" in sections:
            self.case("Long/N16/D", {"intervals": 16, "control": "D", "time_substeps": (1, 2, 4),
                      "nonlinear_factor": .1, "times_s": observation_times(), "scope": "full_100s_window_attempt"},
                      lambda directory: self.step(directory, 16, "D", (1, 2, 4), .1, observation_times()))
        if "dc" in sections:
            for n in (16, 32, 64):
                for control in "ABCD":
                    self.case(f"DC/N{n}/{control}", {"intervals": n, "control": control, "voltage_V": .005,
                              "scope": "same_control_5mV_dc_and_three_step_zero_bias_conductance"},
                              lambda directory, n=n, c=control: self.dc_record(n, c))
        if "ac" in sections:
            for n in (16, 32, 64):
                self.case(f"AC/N{n}/D", {"intervals": n, "control": "D", "frequency_Hz": FREQUENCIES,
                          "scope": "direct_zero_bias_ac_three_derivative_levels_not_window_coverage"},
                          lambda directory, n=n: small_signal_response(solve_controlled_dc(
                              self.stack, n, self.binding, self.prepared(n)), FREQUENCIES))
        if "amplitude" in sections:
            for amplitude, label in ((.01, "A10mV"), (.005, "A5mV"), (.0025, "A2p5mV")):
                self.case(f"Amplitude/N16/{label}", {"intervals": 16, "control": "D", "amplitude_V": amplitude,
                          "times_s": DEFAULT_TIMES_S, "time_substeps": (1, 2, 4), "nonlinear_factor": .1,
                          "scope": "short_small_signal_amplitude_ladder"},
                          lambda directory, a=amplitude: self.step(directory, 16, "D", (1, 2, 4), .1, DEFAULT_TIMES_S, a))
            self.case("Amplitude/ComparisonN16", {"scope": "normalized_current_comparison_with_unknown_absolute_error"},
                      lambda directory: self.amplitude_record())
        if "compare" in sections:
            for control in "ABCD":
                for left, right in ((16, 32), (32, 64)):
                    self.case(f"Compare/{control}/N{left}N{right}", {"left": left, "right": right, "control": control,
                              "scope": "short_fixed_position_spatial_comparison"},
                              lambda directory, l=left, r=right, c=control: self.compare_available(
                                  [f"Short/N{l}/{c}", f"Short/N{r}/{c}"],
                                  lambda: compare_spatial_responses(
                                      spatial_responses_from_step(self.prepared(l), self.saved(f"Short/N{l}/{c}")),
                                      spatial_responses_from_step(self.prepared(r), self.saved(f"Short/N{r}/{c}")))))
            for left, right in ((16, 32), (32, 64)):
                self.case(f"Compare/AC/N{left}N{right}", {"left": left, "right": right, "control": "D",
                          "scope": "direct_ac_real_and_imaginary_mesh_comparison"},
                          lambda directory, l=left, r=right: self.compare_available(
                              [f"AC/N{l}/D", f"AC/N{r}/D"], lambda: self.ac_mesh_comparison(l, r)))
            self.case("Compare/LongTailN16D", {"scope": "finite_accepted_tail_vs_5mV_dc_no_infinite_tail_bound"},
                      lambda directory: self.tail_record())
            self.matrix_comparisons()

    def dc_record(self, n, control):
        prepared = self.prepared(n)
        target = solve_controlled_dc(self.stack, n, self.binding, prepared, control=control, voltage_V=.005)
        return {"target_bias": target.evidence,
                "conductance": dc_conductance_study(self.stack, n, self.binding, prepared, control=control)}

    def amplitude_record(self):
        coarse, fine = self.saved("Amplitude/N16/A5mV"), self.saved("Amplitude/N16/A2p5mV")
        baseline = self.zero_dc(16, "D").evidence["terminal_current_A_m2"]
        curves = []
        for record in (coarse, fine):
            curves.append((np.array([x["report_contact_current_A_m2"][0] for x in record["regular_currents"]])-baseline)
                          / record["amplitude_V"])
        return {"scope": "two_smallest_amplitudes_normalized_regular_current_only",
                "times_s": fine["times_s"], "amplitudes_V": [coarse["amplitude_V"], fine["amplitude_V"]],
                "normalized_current_S_m2": curves, "absolute_difference_S_m2": np.abs(curves[0]-curves[1]),
                "one_percent_signal_scale_S_m2": .01*np.maximum(np.abs(curves[0]), np.abs(curves[1])),
                "independent_absolute_current_error_A_m2": None,
                "linearity_certified": False,
                "reason": "independent numerical current errors and their one-percent signal budget remain unestablished"}

    @staticmethod
    def matrix_key(item):
        return f"Matrix/N{item.intervals}/T{item.time_substeps[0]}/F{str(item.nonlinear_factor).replace('.', 'p')}"

    def matrix_comparisons(self):
        cases = base_convergence_cases()
        fields = ("intervals", "time_substeps", "nonlinear_factor")
        for axis in fields:
            groups = {}
            for item in cases:
                other = tuple(getattr(item, key) for key in fields if key != axis)
                groups.setdefault(other, []).append(item)
            for group in groups.values():
                group = sorted(group, key=lambda item: getattr(item, axis), reverse=axis == "nonlinear_factor")
                for left, right in zip(group[:-1], group[1:]):
                    left_key, right_key = self.matrix_key(left), self.matrix_key(right)
                    label = left_key.removeprefix("Matrix/").replace("/", "")+"To"+right_key.removeprefix("Matrix/").replace("/", "")
                    self.case(f"MatrixCompare/{axis}/{label}", {
                        "axis": axis, "left_case": left_key, "right_case": right_key,
                        "scope": "one_axis_short_window_comparison_with_other_two_axes_fixed"},
                        lambda directory, l=left, r=right: self.compare_available(
                            [self.matrix_key(l), self.matrix_key(r)], lambda: self.matrix_comparison(l, r)))

    def compare_available(self, keys, operation):
        missing = []
        for key in keys:
            directory = self.latest(key)
            if directory is None:
                missing.append({"case": key, "status": "not_run"})
            elif not (directory/"CompletionV1.json").is_file() or not (directory/"ManifestV1.json").is_file():
                missing.append({"case": key, "status": "interrupted"})
            else:
                completion = checked_read(directory, "CompletionV1.json")
                if completion["status"] != "completed":
                    missing.append({"case": key, "status": completion["status"]})
        if missing:
            return {"comparison_available": False, "missing_dependencies": missing,
                    "scope": "comparison_unavailable_due_to_missing_successful_input",
                    "reason": "no numerical agreement or disagreement was inferred"}
        return operation()

    def ac_mesh_comparison(self, left, right):
        a, b = self.saved(f"AC/N{left}/D"), self.saved(f"AC/N{right}/D")
        for key in ("reference_sha256", "control", "voltage_V"):
            if a[key] != b[key]:
                raise ValueError("AC physical experiment differs: "+key)
        def response(record):
            values = [complex(x["real"], x["imag"]) for x in record["admittance_S_m2"]]
            return R1Response(np.asarray(values), {"frequency_Hz": np.asarray(record["frequency_Hz"])})
        report = compare_responses("admittance", response(a), response(b))
        eligible = np.asarray(a["numerically_eligible_frequency_points"]) & np.asarray(b["numerically_eligible_frequency_points"])
        return {"component_comparison": report, "within_compared_budgets": report["passed"] and bool(np.all(eligible)),
                "both_numeric_checks_passed": eligible,
                "left_numeric_frequency_checks": a["numerically_eligible_frequency_points"],
                "right_numeric_frequency_checks": b["numerically_eligible_frequency_points"],
                "scope": "direct_ac_mesh_component_comparison_only",
                "frequency_window_coverage_certified": False, "double_domain_consistent": False}

    def matrix_comparison(self, left, right):
        a, b = self.saved(self.matrix_key(left)), self.saved(self.matrix_key(right))
        spatial = compare_spatial_responses(
            spatial_responses_from_step(self.prepared(left.intervals), a),
            spatial_responses_from_step(self.prepared(right.intervals), b))
        baseline_left = self.zero_dc(left.intervals, left.control)
        baseline_right = self.zero_dc(right.intervals, right.control)
        electrical = compare_step_current_charge(a, b, baseline_left.evidence["current_A_m2"][[0, -1]],
                                                baseline_right.evidence["current_A_m2"][[0, -1]])
        return {"spatial": spatial, "electrical": electrical,
                "within_compared_budgets": spatial["within_compared_budgets"] and electrical["within_compared_budgets"],
                "scope": "one_axis_short_window_response_comparison_only",
                "full_window_convergence_certified": False}

    def zero_dc(self, intervals, control):
        key = intervals, control
        if key not in self.dc_baselines:
            self.dc_baselines[key] = solve_controlled_dc(self.stack, intervals, self.binding,
                                                        self.prepared(intervals), control=control)
        return self.dc_baselines[key]

    def tail_record(self):
        prepared = self.prepared(16)
        attempt = self.latest("Long/N16/D")
        if attempt is None:
            return {"comparison_available": False, "reason": "full-window attempt not available"}
        completion = checked_read(attempt, "CompletionV1.json")
        record = (checked_read(attempt, "ResultV1.json") if completion["status"] == "completed"
                  else checked_read(attempt, "FailureV1.json")["partial_result"])
        rows = [row for row in record.get("accepted_steps", []) if row.get("physical_checks_passed") and row["dt_s"] > 0]
        if not rows:
            return {"comparison_available": False, "reason": "no physically accepted finite tail is available"}
        row = max(rows, key=lambda r: (r["time_s"], r["substeps"]))
        state = {**row["state"], "prepared_sha256": prepared.sha256, "control": "D", "voltage_V": .005}
        dc = solve_controlled_dc(self.stack, 16, self.binding, prepared, voltage_V=.005)
        # A finite-step average is not the instantaneous regular current.
        # Reconstruct this saved state using its exact solver coordinates.
        if "output_states" not in record or row["time_s"] != record["times_s"][-1]:
            return {"scope": "failed_window_prefix_only", "last_accepted_time_s": row["time_s"],
                    "target_dc": dc.evidence, "prefix_state": state,
                    "tail_vs_dc_comparison": None,
                    "reason": "requested tail unavailable; do not substitute a finite-step average for regular current",
                    "infinite_tail_integral_bound_F_m2": None}
        regular = record["regular_currents"][-1]["report_contact_current_A_m2"][0]
        return compare_transient_tail(dc, initial_state=prepared.to_dict()["state"], tail_state=state,
                                       tail_regular_current_A_m2=regular, time_s=row["time_s"])

    def finish(self):
        directory = self.output/"Invocations"
        directory.mkdir(exist_ok=True)
        index = len(list(directory.glob("InvocationV*.json")))+1
        summary = {"schema": "R1PhysicsStudyInvocationV1", "started_utc": self.started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                   "cases": self.rows, "attempted_cases": self.attempted,
                   "study_exit_passed": False,
                   "missing_for_R1_2_exit": ["full_window_27_case_convergence", "finite_amplitude_linearity",
                                              "earlier_start_and_extended_tail", "bounded_time_to_frequency_comparison",
                                              "frequency_window_coverage", "independent_acceptance"],
                   "scope": "development_case_outcomes_do_not_imply_study_acceptance"}
        write_json(directory/f"InvocationV{index}.json", summary)
        write_json(self.output/"StudySummaryV1.json", summary)
        failures, unavailable = [], []
        for completion_file in sorted(self.output.rglob("CompletionV1.json")):
            case_dir = completion_file.parent
            if not (case_dir/"ManifestV1.json").is_file():
                failures.append({"directory": str(case_dir.relative_to(self.output)),
                                 "failure": "interrupted before final case manifest"})
                continue
            completion = checked_read(case_dir, "CompletionV1.json")
            if completion["status"] == "unavailable":
                unavailable.append({"directory": str(case_dir.relative_to(self.output)), "completion": completion,
                                    "conditions": checked_read(case_dir, "RequestV1.json"),
                                    "result": checked_read(case_dir, "ResultV1.json")})
            if completion["status"] == "failed" or completion.get("scientific_checks_passed") is False:
                failures.append({"directory": str(case_dir.relative_to(self.output)), "completion": completion,
                                 "conditions": checked_read(case_dir, "RequestV1.json"),
                                 "failure": (checked_read(case_dir, "FailureV1.json").get("message")
                                             if completion["status"] == "failed" else "returned scientific comparisons failed")})
        write_json(self.output/"FailureIndexV1.json", {"schema": "R1PhysicsStudyFailureIndexV1", "cases": failures})
        write_json(self.output/"UnavailableComparisonIndexV1.json", {"schema": "R1UnavailableComparisonIndexV1", "cases": unavailable})
        return 1 if any(row["status"] == "failed" or row.get("scientific_checks_passed") is False for row in self.rows) else 0


def scientific_checks(result):
    """Return scoped pass/fail, or None when a required uncertainty is absent."""
    if not isinstance(result, dict):
        return None
    for key in ("certificate", "preparation_checks"):
        if key in result:
            return bool(result[key]["certified"])
    if "within_compared_budgets" in result:
        return bool(result["within_compared_budgets"])
    if "numerically_eligible_frequency_points" in result:
        return bool(np.all(result["numerically_eligible_frequency_points"]))
    if "target_bias" in result:
        return bool(result["target_bias"]["certified"] and result["conductance"]["finest_pair_agrees"])
    if "all_observables_agree" in result:
        return bool(result["all_observables_agree"])
    return None


def compare_step_current_charge(left, right, left_baseline, right_baseline):
    """Compare regular currents and cumulative charge at exact physical times.

    Each grid supplies independently evaluated same-control 0- currents.
    Charge includes the impulse and subtracts baseline current times time.
    The individual backward-Euler averages are not relabeled regular current.
    """
    responses = []
    for record, baseline in ((left, left_baseline), (right, right_baseline)):
        times = np.asarray(record["times_s"], dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        current = np.asarray([item["report_contact_current_A_m2"] for item in record["regular_currents"]])
        if current.shape != (len(times), 2) or baseline.shape != (2,):
            raise ValueError("regular current and physical contacts must align with exact output times")
        finest = max(record["policy"]["refinement_substeps"])
        rows = [row for row in record["accepted_steps"] if row["substeps"] == finest]
        charge = []
        for t in times:
            matching = [row for row in rows if row["time_s"] == t]
            if len(matching) != 1:
                raise ValueError("integrated charge requires exactly one finest row at each output time")
            charge.append(record["initial_event"]["impulse_charge_C_m2"]+matching[0]["regular_integrated_charge_C_m2"]-baseline[0]*t)
        responses.append((R1Response(current-baseline, {"time_s": times}, ("left_contact", "right_contact")),
                          R1Response(np.asarray(charge), {"time_s": times})))
    current_report = compare_responses("regular_current_response", responses[0][0], responses[1][0])
    charge_report = compare_responses("integrated_charge_response", responses[0][1], responses[1][1])
    return {"scope": "regular_current_and_impulse_inclusive_charge_response_comparison",
            "regular_current": current_report, "integrated_charge": charge_report,
            "baseline_contact_current_A_m2": [left_baseline, right_baseline],
            "within_compared_budgets": current_report["passed"] and charge_report["passed"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=PROJECT/"tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json")
    parser.add_argument("--fixture", type=Path, default=PROJECT/"tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    parser.add_argument("--section", action="append", required=True, choices=(*SECTIONS, "dc-ac", "all"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="preserve prior failure and create a new attempt")
    parser.add_argument("--case-filter", default="", help="substring of case key; required preparations run as dependencies")
    parser.add_argument("--max-cases", type=int, help="maximum newly started cases in this invocation")
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    study = None
    try:
        study = Study(args)
        sections = SECTIONS if "all" in args.section else [part for section in args.section
                                                          for part in (("dc", "ac") if section == "dc-ac" else (section,))]
        study.run(sections)
        return study.finish()
    except CaseBudgetExhausted:
        return study.finish()
    except Exception as exc:
        print(f"STUDY STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if study is not None:
            study.rows.append({"case": "invocation", "status": "failed", "reason": str(exc)})
            study.finish()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

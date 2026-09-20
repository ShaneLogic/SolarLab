#!/usr/bin/env python3
"""Versioned R1 physical study with explicit source, scope and independent verification.

Formal production and verification run through the trusted controlled launcher
with --runner physics-study. Development calculations remain labelled as such.
Use --window full for the logarithmic time grid; functional is a short diagnosis.
"""

from __future__ import annotations

import argparse
import copy
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
    require_r1_checkout, record_frozen_source, current_execution_context,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    R1Response, base_convergence_cases, observation_times, compare_responses,
    convergence_cases, AMPLITUDES_V, compare_amplitude_halving,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
    DEFAULT_TIMES_S, check_zero_excitation, r1_policy, run_r1_step,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    R1PreparedState, execution_source, prepare_common_state, restore_common_state,
    verify_prepared_physics,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    solve_controlled_dc, dc_conductance_study, small_signal_response, compare_transient_tail,
    assess_small_signal_response, dc_amplitude_endpoint_study, restore_controlled_dc,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import (
    spatial_responses_from_step, compare_spatial_responses, compare_convergence_responses,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import (
    frequency_window_report, reconstruct_study_response,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import R1AdmittanceErrors
from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_request import (
    build_execution_plan, load_execution_plan, inventory_coverage, load_qualification_inputs,
    verify_invocation_attempts,
)


SECTIONS = ("prepare", "zero", "short", "matrix", "long", "dc", "amplitude-dc", "ac", "amplitude", "compare", "windows", "reconstruct")
FREQUENCIES = np.r_[0., np.logspace(-3, 8, 45)]


def expanded_sections(values):
    selected = set(SECTIONS) if "all" in values else {
        part for section in values for part in (("dc", "ac") if section == "dc-ac" else (section,))}
    return tuple(section for section in SECTIONS if section in selected)


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


def pair_ready(value):
    """Keep legacy serialization unchanged while sharing pair failure tags."""
    def convert(item):
        if isinstance(item, dict):
            if (set(item) == {"value","reason","representation"}
                    and item["reason"] == "nonfinite_numeric_evidence"):
                return {"nonfinite":item["representation"]}
            return {key:convert(child) for key,child in item.items()}
        if isinstance(item,list):
            return [convert(child) for child in item]
        return item
    return convert(ready(value))


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
        if file.is_file() and file != directory/"ManifestV1.json"
    })


def checked_read(directory, filename):
    directory = Path(directory)
    manifest = json.loads((directory/"ManifestV1.json").read_text())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p != directory/"ManifestV1.json"}
    if set(manifest) != actual:
        raise ValueError("saved case manifest coverage mismatch: "+str(directory))
    for relative, expected in manifest.items():
        path = directory/relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.is_file() or sha(path) != expected):
            raise ValueError("saved case manifest mismatch: "+str(path))
    if filename not in manifest:
        raise ValueError("saved case does not bind "+filename)
    return json.loads((directory/filename).read_text())


def verify_study_source(directory, context, *, require_read_receipt):
    """Check provenance claims against the selected source, not one another."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import (
        _verify_archived_source_identity, verify_source_reads,
    )
    selected = context.to_dict()
    execution = checked_read(directory, "ExecutionSourceV1.json")
    provenance = {"repository_root", "dirty", "runtime"}
    if (set(execution) != set(selected) or any(execution[key] != selected[key]
            for key in selected if key not in provenance)):
        raise ValueError("study execution source differs from the caller-selected frozen source")
    if (not isinstance(execution["repository_root"], str)
            or not isinstance(execution["dirty"], dict)
            or any(type(value) is not bool for value in execution["dirty"].values())):
        raise ValueError("study source advisory metadata is invalid")
    runtime, expected_runtime = execution.get("runtime"), selected.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(expected_runtime, dict):
        raise ValueError("study verification requires controlled runtime provenance")
    path_fields = {"launcher", "python_executable", "dependency_paths"}
    if set(runtime) != set(expected_runtime) or any(runtime[key] != expected_runtime[key]
                                                  for key in runtime if key not in path_fields):
        raise ValueError("recorded study runtime differs from controlled verification")
    if (any(not isinstance(runtime[name], str) or not Path(runtime[name]).is_absolute()
            for name in ("launcher", "python_executable"))
            or not isinstance(runtime["dependency_paths"], list)
            or any(not isinstance(path, str) or not Path(path).is_absolute()
                   for path in runtime["dependency_paths"])):
        raise ValueError("invalid recorded runtime paths")
    _verify_archived_source_identity(Path(directory), context._source_bytes)
    if (Path(directory)/"SourceChangesV1.patch").read_bytes() != context._source_changes:
        raise ValueError("source changes differ from the selected frozen execution")
    if require_read_receipt:
        verify_source_reads(directory, execution, checked_read(directory, "SourceManifestV1.json"))
    return {"caller_source_and_all_archived_bytes_verified": True,
            "provenance_only": ["repository_root", "dirty", "runtime_paths"]}


class Study:
    @property
    def numerics(self):
        return get_backend(getattr(self, "_backend", None))

    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        self._backend = get_backend(getattr(args, "backend", None))
        self.context = require_r1_checkout(project=PROJECT)
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import standard_binding_record
        self.standard = standard_binding_record(self.context,
            approved_standard_sha256=getattr(args, "approved_standard_sha256", None))
        self.run_class = self.context.run_class
        if getattr(args, "formal", False) and self.run_class != "formal":
            raise ValueError("formal study requires the trusted controlled launcher")
        self.verifying = getattr(args, "verify", False)
        if self.verifying and not self.output.is_dir():
            raise ValueError("verification requires an existing sealed collection")
        if self.verifying and self.run_class != "formal":
            raise ValueError("scientific study verification requires controlled source execution")
        self.grids = tuple(getattr(args, "grids", (16, 32, 64)))
        self.window = getattr(args, "window", "functional")
        if self.window == "diagnostic" and not self.numerics.is_pair:
            raise ValueError("the fixed diagnostic window requires explicit pair backend")
        self.times = ((0., 1e-9) if self.window == "diagnostic" else DEFAULT_TIMES_S if self.window == "functional" else
                      observation_times(first_time_s=args.first_time_s, last_time_s=args.last_time_s))
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_window import build_window_spec
        self.window_spec = build_window_spec(self.times) if self.window == "full" else None
        self.frequencies = (np.r_[0., np.logspace(-6, 10, 65)]
                            if getattr(args, "extended_frequency", False) else FREQUENCIES)
        self.controls = tuple(getattr(args, "matrix_controls", ("D",)))
        self.amplitudes = tuple(getattr(args, "amplitudes", None) or AMPLITUDES_V)
        self.verified_cases = {}
        self.input_dependencies = {}
        self.source = execution_source()
        fixture_raw = self.context.read_bytes(args.fixture) if self.run_class == "formal" else args.fixture.read_bytes()
        reference_raw = self.context.read_bytes(args.reference) if self.run_class == "formal" else args.reference.read_bytes()
        self.binding = json.loads(reference_raw)
        with tempfile.TemporaryDirectory(prefix="r1-study-input-") as temporary:
            fixture_path = Path(temporary)/"SourceFixtureV1.yaml"
            fixture_path.write_bytes(fixture_raw)
            self.stack = load_device_from_yaml(fixture_path)
        self.started = datetime.now(timezone.utc).isoformat()
        self.rows, self.attempted = [], 0
        self.invocation_attempts = []
        self.preparations = {}
        self.dc_baselines = {}
        self.qualification_inputs = load_qualification_inputs(
            getattr(args, "qualification_file", None), getattr(args, "qualification_sha256", None),
            result_directory=self.output)
        request = {
            "schema": "R1PhysicsStudyRequestV5" if self.numerics.is_pair else "R1PhysicsStudyRequestV4", "run_class": self.run_class,
            "source": self.source, "runner_sha256": sha(Path(__file__)),
            "fixture_sha256": hashlib.sha256(fixture_raw).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_raw).hexdigest(),
            "short_times_s": self.times if self.window == "diagnostic" else DEFAULT_TIMES_S, "full_times_s": observation_times(),
            "frequency_Hz": self.frequencies,
            "amplitude_ladder_V": list(AMPLITUDES_V),
            "transient_amplitudes_V": list(self.amplitudes),
            "grids": self.grids, "matrix_controls": self.controls,
            "window": self.window, "times_s": self.times,
            "window_spec": self.window_spec,
            "window_amplitude_V": getattr(args, "window_amplitude", .005),
            "linearity_case": getattr(args, "linearity_case", None),
            "qualification_inputs_sha256": getattr(args, "qualification_sha256", None),
            "standard_binding": self.standard,
            "window_extensions": {"first_time_s": args.first_time_s, "last_time_s": args.last_time_s,
                                   "extended_last_time_s": min(args.last_time_s*10, 1e5),
                                   "earlier_first_time_s": max(args.first_time_s/10, 1e-12)},
            "scope": "single_frozen_study_settings_not_independent_R1_2_acceptance",
            "matrix_scope": "base_27_plus_declared_extensions_and_control_axis_crosses",
        }
        if self.numerics.is_pair:
            request["representation"] = self.numerics.representation_id
            request["pair_scope"] = "prepared_zero_transient_and_replay_only_DC_AC_not_migrated"
        self.request = ready(request)
        if getattr(args, "plan_only", False):
            self.plan = self.make_plan(expanded_sections(args.section))
            plan_file = args.plan_file.resolve()
            if plan_file.is_relative_to(self.output):
                raise ValueError("pre-execution plan must be written outside the not-yet-created result directory")
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            if plan_file.exists() and plan_file.read_bytes() != raw_json(self.plan):
                raise ValueError("plan path already holds a different request; choose a new version")
            if not plan_file.exists():
                write_json(plan_file, self.plan)
            self.plan_sha256 = sha(plan_file)
            return
        supplied_plan = getattr(args, "plan_file", None)
        if self.run_class == "formal" and supplied_plan is None:
            raise ValueError("formal study requires a pre-execution plan and external study request digest")
        if supplied_plan is not None:
            self.plan = load_execution_plan(supplied_plan, getattr(args, "request_sha256", None),
                                            result_directory=self.output)
            if self.plan["study_settings"] != self.request:
                raise ValueError("source, standard or study settings differ from the externally anchored request")
            expected_plan = self.make_plan(self.plan["sections"], self.plan["selection"])
            if self.plan != expected_plan:
                raise ValueError("external study plan differs from source-defined case requests")
            self.plan_sha256 = sha(supplied_plan)
        else:
            self.plan = self.make_plan(expanded_sections(args.section))
            self.plan_sha256 = None
        self.planned_cases = self.plan["cases"]
        if self.run_class == "formal" and getattr(args, "retry_failed", False):
            raise ValueError("formal retries require a new versioned study request; preserve the failed study")
        if self.output.exists():
            if not args.resume and not self.verifying:
                raise ValueError("output exists; use --resume to verify and extend it")
            if self.run_class == "formal":
                anchor = getattr(args, "manifest_sha256", None)
                if not anchor or sha(self.output/"ManifestV1.json") != anchor:
                    raise ValueError("formal resume/verify requires the external study manifest digest")
                checked_read(self.output, "StudyRequestV1.json")
                if self.verifying:
                    verify_study_source(self.output, self.context, require_read_receipt=True)
            existing = json.loads((self.output/"StudyRequestV1.json").read_text())
            if existing != ready(request):
                raise ValueError("resume source, runner, inputs or fixed study request changed")
            if checked_read(self.output, "StudyPlanV1.json") != self.plan:
                raise ValueError("archived study plan differs from externally anchored request")
            existing_inventory = checked_read(self.output, "CaseInventoryV2.json")
            inventory_coverage(self.plan, existing_inventory, {})
            if checked_read(self.output, "QualificationInputsV1.json") != self.qualification_inputs:
                raise ValueError("archived qualification inputs differ from caller-held inputs")
            if self.run_class == "formal":
                self.verify_recorded_invocations()
        else:
            self.output.mkdir(parents=True)
            write_json(self.output/"StudyRequestV1.json", request)
            write_json(self.output/"StudyPlanV1.json", self.plan)
            write_json(self.output/"CaseInventoryV2.json", self.planned_cases)
            write_json(self.output/"QualificationInputsV1.json", self.qualification_inputs)
            write_json(self.output/"EnvironmentV1.json", {
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "threads": {key: os.environ[key] for key in THREAD_VARIABLES},
            })
            record_frozen_source(self.output, self.context)
        if self.run_class == "formal":
            for path in (args.fixture, args.reference):
                if self.context.read_bytes(path) != path.read_bytes():
                    raise ValueError("study input differs from controlled snapshot: "+str(path))
        self.original_inventory = (json.loads((self.output/"CaseInventoryV2.json").read_text())
                                   if (self.output/"CaseInventoryV2.json").is_file() else {})

    def make_plan(self, sections, selection=None):
        """Enumerate the real runner's requests without executing operations."""
        if tuple(sections) != expanded_sections(sections):
            raise ValueError("plan sections must use the source-defined ordered section set")
        selection = selection or {"case_filter": self.args.case_filter,
                                  "cases": sorted(getattr(self.args, "case", None) or [])}
        if set(selection) != {"case_filter", "cases"}:
            raise ValueError("invalid study plan selection")
        collector = Study.__new__(Study)
        collector.__dict__.update(self.__dict__)
        collector.planning = True
        records = {}
        exact = set(selection["cases"])
        def selected(key):
            return (not exact or key in exact) and (not selection["case_filter"] or selection["case_filter"] in key)
        def capture(key, request, operation, *, dependency=False):
            if not dependency and not selected(key):
                return None
            request = ready(dict(request))
            if all(field in request for field in ("intervals", "control", "time_substeps", "nonlinear_factor", "times_s")):
                request.setdefault("amplitude_V", .005)
            if key in records and records[key] != request:
                raise ValueError("case has conflicting pre-execution requests: " + key)
            records[key] = request
            if not key.startswith("Preparation/"):
                for field in ("intervals", "grid", "left", "right"):
                    if field in request:
                        prepare(request[field])
                for field in ("left_case", "right_case"):
                    if field in request:
                        grid = next(int(part[1:]) for part in request[field].split("/") if part.startswith("N"))
                        prepare(grid)
            if key.startswith("AC/"):
                baseline(request["intervals"], request.get("control", "D"))
            if key.startswith(("Compare/", "MatrixCompare/", "Linearity/")):
                grids = {request[name] for name in ("intervals", "grid", "left", "right") if name in request}
                for name in ("left_case", "right_case"):
                    if name in request:
                        grids.add(next(int(p[1:]) for p in request[name].split("/") if p.startswith("N")))
                for n in sorted(grids or set(self.grids)):
                    baseline(n, request.get("control", "D"))
            if key.startswith("Window/"):
                n, control = request["intervals"], request["control"]
                capture(self.target_dc_key(n, control, request["amplitude_V"]),
                        self.dc_state_request(n, control, request["amplitude_V"]), None, dependency=True)
                capture(f"DC/N{n}/{control}", self.dc_study_request(n, control), None, dependency=True)
            return None
        def prepare(n):
            capture(f"Preparation/N{n}", {"intervals": n, "scope": "common_D_equilibrium"}, None, dependency=True)
        def baseline(n, control):
            capture(f"DCBaseline/N{n}/{control}", self.dc_state_request(n, control, 0.), None, dependency=True)
        collector.case, collector.prepared, collector.selected = capture, prepare, selected
        collector.qualified_amplitude = lambda: (self.request["window_amplitude_V"], self.request["linearity_case"])
        collector.run(tuple(sections))
        if exact - set(records):
            raise ValueError("selected cases are not defined in the requested sections: " + ", ".join(sorted(exact-set(records))))
        return build_execution_plan(self.request, sections, selection, records)

    def source_unchanged(self):
        if execution_source() != self.source:
            raise RuntimeError("source changed during the study; start a new evidence directory")
        if load_qualification_inputs(getattr(self.args, "qualification_file", None),
                getattr(self.args, "qualification_sha256", None), result_directory=self.output) != self.qualification_inputs:
            raise RuntimeError("qualification inputs changed during the study")

    def selected(self, key):
        if hasattr(self, "planned_cases") and key not in self.planned_cases:
            return False
        exact = getattr(self.args, "case", None)
        return bool(getattr(self, "verifying", False)) or (
            (not exact or key in exact) and (not self.args.case_filter or self.args.case_filter in key))

    def latest(self, key):
        directory = self.output/key
        attempts = sorted(directory.glob("AttemptV*"), key=lambda p: int(p.name.removeprefix("AttemptV")))
        return attempts[-1] if attempts else None

    def saved(self, key, *, require_scientific=True):
        directory = self.latest(key)
        if directory is None:
            raise ValueError("required study case has not run: "+key)
        completion = checked_read(directory, "CompletionV1.json")
        if completion["status"] != "completed":
            raise ValueError("required study case failed: "+key)
        result = checked_read(directory, "ResultV1.json")
        request = checked_read(directory, "RequestV1.json")
        self.audit_saved(key, directory, request, result, completion,
                         operation=getattr(self, "operations", {}).get(key))
        if require_scientific and completion.get("scientific_checks_passed") is not True:
            raise ValueError("required study case lacks scientific eligibility: "+key)
        self.record_dependency(key)
        return result

    def record_dependency(self, key):
        owner = getattr(self, "active_case", None)
        if owner is None or owner == key:
            return
        if not hasattr(self, "input_dependencies"):
            self.input_dependencies = {}
        directory = self.latest(key)
        if directory is None:
            raise ValueError("required dependency has no sealed attempt: "+key)
        self.input_dependencies.setdefault(owner, {})[key] = {
            "attempt": str(directory.relative_to(self.output)),
            "manifest_sha256": sha(directory/"ManifestV1.json"),
        }

    def audit_saved(self, key, directory, request, result, completion, operation=None):
        """Recompute physical contents, not the booleans of a self-sealed report."""
        if not hasattr(self, "verified_cases"):
            self.verified_cases = {}
        identity = sha(directory/"ManifestV1.json")
        cached = self.verified_cases.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1]
        if completion.get("case") != key or completion.get("scope") != request.get("scope"):
            raise ValueError("case completion differs from requested identity or scope")
        if self.numerics.is_pair:
            if (completion.get("schema") != "R1PhysicsStudyCaseV2"
                    or completion.get("representation") != self.numerics.representation_id):
                raise ValueError("pair case completion representation/schema differs from request")
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import verify_numeric_sidecar
            verify_numeric_sidecar(directory/"StateArraysV1.npz", result)
        elif completion.get("schema") == "R1PhysicsStudyCaseV2" or "representation" in completion:
            raise ValueError("legacy study cannot inherit pair case evidence")
        if hasattr(self, "run_class") and completion.get("run_class") != self.run_class:
            raise ValueError("case execution class differs from the study")
        if (directory/"InputBindingsV2.json").is_file():
            for parent, binding in checked_read(directory, "InputBindingsV2.json").items():
                current = self.latest(parent)
                if (current is None or str(current.relative_to(self.output)) != binding["attempt"]
                        or sha(current/"ManifestV1.json") != binding["manifest_sha256"]):
                    raise ValueError("case input differs from bound parent evidence: "+parent)
        audit = self.audit_result(key, request, result, operation=operation)
        actual = scientific_checks(result)
        if audit.get("certified") is False:
            actual = False
        if completion.get("scientific_checks_passed") is not actual:
            raise ValueError("case scientific verdict differs from recomputed contents: "+key)
        self.verified_cases[key] = (identity, audit)
        return audit

    def audit_result(self, key, request, result, *, operation=None):
        if not isinstance(result, dict):
            raise ValueError("study result must be a record")
        if key.startswith("Preparation/"):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import verify_prepared_metadata
            verify_prepared_metadata(result)
            prepared = self.numerics.decode_prepared(result)
            if result.get("source") != self.source or result.get("intervals") != request["intervals"]:
                raise ValueError("preparation source/grid differs from the study")
            if result.get("preparation_policy") != ready(r1_policy()):
                raise ValueError("preparation policy differs from declared R1 policy")
            verify_prepared_physics(prepared, self.stack, self.binding, policy=r1_policy(), backend=self.numerics)
            return {"certified": bool(result["preparation_checks"]["certified"]),
                    "content_matches_recomputed": True}
        if result.get("schema") in ("R1ControlledStepV1", "R1ControlledStepV2"):
            n = request["intervals"]
            if (result.get("intervals") != n or result.get("control_label") != request.get("control", "D")
                    or result.get("amplitude_V") != request.get("amplitude_V", .005)
                    or result.get("times_s") != ready(request["times_s"])
                    or result.get("policy") != ready(r1_policy(request["nonlinear_factor"],
                                                               time_substeps=request["time_substeps"]))):
                raise ValueError("step result differs from the requested numerical/physical axes")
            return verify_r1_step_physics(self.stack, n, self.binding, self.prepared(n), result, backend=self.numerics)
        if key.startswith(("DC/", "DCBaseline/", "TargetDC/", "AC/", "AmplitudeDC/")):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import verify_response_content
            return verify_response_content(result, stack=self.stack, intervals=request["intervals"],
                                           binding=self.binding, prepared=self.prepared(request["intervals"]),
                                           request=request, backend=self.numerics)
        if key.startswith("Zero/"):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import verify_zero_metadata
            def same(actual, expected, label):
                if ready(actual) != ready(expected):
                    raise ValueError(label + " differs from the requested zero-excitation contract")
            verify_zero_metadata(result, same=same)
            p = self.prepared(request["intervals"])
            fresh = check_zero_excitation(self.stack, request["intervals"], self.binding, p,
                                          expected_prepared_sha256=p.sha256, backend=self.numerics)
        elif operation is not None and key.startswith(("Compare/", "MatrixCompare/", "Amplitude/Comparison",
                                                       "Linearity/", "Frequency/", "DoubleDomain/")):
            fresh = operation(None)
        elif getattr(self, "run_class", "development") == "formal":
            raise ValueError("formal result type has no independent replay operation: "+key)
        else:
            return {"certified": scientific_checks(result), "content_matches_recomputed": False}
        if ready(fresh) != ready(result):
            raise ValueError("study result differs from independent replay: "+key)
        return {"certified": scientific_checks(fresh), "content_matches_recomputed": True}

    def case(self, key, request, operation, *, dependency=False):
        request = ready(dict(request))
        if hasattr(self, "planned_cases") and key not in self.planned_cases:
            return None
        if (getattr(self, "verifying", False) and key not in getattr(self, "original_inventory", {})
                and self.latest(key) is None):
            return None
        if all(field in request for field in ("intervals", "control", "time_substeps", "nonlinear_factor", "times_s")):
            request.setdefault("amplitude_V", .005)
        if hasattr(self, "planned_cases") and request != self.planned_cases[key]:
            raise ValueError("runtime case request differs from pre-execution plan: " + key)
        if not hasattr(self, "expected_cases"):
            self.expected_cases = {}
        if not hasattr(self, "operations"):
            self.operations = {}
        self.operations[key] = operation
        self.expected_cases[key] = ready(request)
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
            if sealed and completion["status"] in ("completed", "unavailable"):
                self.audit_saved(key, prior, request, checked_read(prior, "ResultV1.json"), completion,
                                 operation=operation)
            if getattr(self, "verifying", False):
                if not sealed:
                    raise ValueError("cannot verify an interrupted unsealed case: "+key)
                if completion["status"] == "failed":
                    audit = self.audit_failed(key, prior, request)
                    self.verified_cases[key] = (sha(prior/"ManifestV1.json"), audit)
                self.rows.append({"case": key, "status": completion["status"],
                                  "scientific_checks_passed": completion.get("scientific_checks_passed"),
                                  "independently_verified": self.verified_cases.get(key, (None, {}))[1].get("content_matches_recomputed") is True,
                                  "failure_scope": (self.verified_cases[key][1].get("failure_scope")
                                                    if completion["status"] == "failed" else None)})
                return checked_read(prior, "ResultV1.json") if completion["status"] == "completed" else None
            if sealed and (completion["status"] == "completed" or not self.args.retry_failed):
                self.rows.append({"case": key, "status": completion["status"], "resumed": True,
                                  "scientific_checks_passed": completion.get("scientific_checks_passed"),
                                  "directory": str(prior.relative_to(self.output))})
                return checked_read(prior, "ResultV1.json") if completion["status"] == "completed" else None
            if not sealed:
                self.rows.append({"case": key, "status": "interrupted_attempt_preserved",
                                  "directory": str(prior.relative_to(self.output))})
                if getattr(self, "run_class", "development") == "formal":
                    return None
        if getattr(self, "verifying", False):
            self.rows.append({"case": key, "status": "not_run", "scientific_checks_passed": None})
            return None
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
        if not hasattr(self, "invocation_attempts"):
            self.invocation_attempts = []
        self.invocation_attempts.append({"case": key, "directory": directory.relative_to(self.output).as_posix()})
        print("START", key, flush=True)
        previous_owner = getattr(self, "active_case", None)
        self.active_case = key
        try:
            if dependency_error is not None:
                raise dependency_error
            result = operation(directory)
            self.source_unchanged()
            write_json(directory/"ResultV1.json", result)
            if self.numerics.is_pair:
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import write_numeric_sidecar, verify_numeric_sidecar
                write_numeric_sidecar(directory/"StateArraysV1.npz", result)
                verify_numeric_sidecar(directory/"StateArraysV1.npz", result)
            status, failure = ("unavailable" if isinstance(result, dict) and result.get("comparison_available") is False
                               else "completed"), None
        except Exception as exc:
            status = "failed"
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "partial_result": getattr(exc, "result", None)}
            write_json(directory/"FailureV1.json", pair_ready(failure) if self.numerics.is_pair else failure)
            partial = failure["partial_result"]
            if self.numerics.is_pair and isinstance(partial, dict):
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import write_numeric_sidecar, numeric_arrays
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import nonfinite_numeric_paths
                try:
                    if nonfinite_numeric_paths(partial):
                        np.savez_compressed(directory/"StateArraysV1.npz", **numeric_arrays(partial, allow_nonfinite=True))
                    else:
                        write_numeric_sidecar(directory/"StateArraysV1.npz", partial)
                except Exception as sidecar_error:
                    write_json(directory/"NumericSidecarFailureV1.json", {
                        "type": type(sidecar_error).__name__, "message": str(sidecar_error)})
            result = None
        finally:
            self.active_case = previous_owner
        write_json(directory/"InputBindingsV2.json", getattr(self, "input_dependencies", {}).get(key, {}))
        historical = None
        if all(field in request for field in ("intervals", "control", "time_substeps", "nonlinear_factor", "times_s", "amplitude_V")):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_registry import historical_observation
            historical = historical_observation({**request, "source_commit": getattr(self, "source", {}).get("source_commit")}, failure)
        completion = {
            "schema": "R1PhysicsStudyCaseV2" if self.numerics.is_pair else "R1PhysicsStudyCaseV1", "case": key, "status": status,
            "duration_s": time.monotonic()-start, "run_class": getattr(self, "run_class", "development"),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "study_exit_passed": False, "scope": request.get("scope"),
            "scientific_checks_passed": scientific_checks(result) if status in ("completed", "unavailable") else False,
            "failure": None if failure is None else {k: v for k, v in failure.items() if k != "partial_result"},
            "historical_observation": historical,
        }
        if self.numerics.is_pair:
            completion["representation"] = self.numerics.representation_id
        write_json(directory/"CompletionV1.json", completion)
        if failure is not None:
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness
            persisted = directory/"AcceptedStepsV1.jsonl"
            rows = [json.loads(line) for line in persisted.read_text().splitlines()] if persisted.is_file() else []
            write_json(directory/"FailureWitnessV1.json", build_failure_witness(
                source_commit=getattr(self, "source", {}).get("source_commit"),
                protocol=self.failure_protocol(request), failure=pair_ready(failure) if self.numerics.is_pair else ready(failure),
                result=pair_ready(failure.get("partial_result")) if self.numerics.is_pair else ready(failure.get("partial_result")), persisted_rows=rows))
        seal(directory)
        if not hasattr(self, "produced_cases"):
            self.produced_cases = {}
        self.produced_cases[key] = sha(directory/"ManifestV1.json")
        self.rows.append({"case": key, "status": status, "duration_s": completion["duration_s"],
                          "scientific_checks_passed": completion["scientific_checks_passed"],
                          "directory": str(directory.relative_to(self.output))})
        print(status.upper(), key, f"{completion['duration_s']:.3f}s", flush=True)
        return result

    @staticmethod
    def failure_protocol(request):
        policy = (ready(r1_policy(request["nonlinear_factor"], time_substeps=request["time_substeps"]))
                  if "nonlinear_factor" in request and "time_substeps" in request else None)
        return {"stage": "physics-study", **request, "policy": policy}

    def audit_failed(self, key, directory, request):
        failure = checked_read(directory, "FailureV1.json")
        completion = checked_read(directory, "CompletionV1.json")
        if self.numerics.is_pair and (completion.get("schema") != "R1PhysicsStudyCaseV2"
                or completion.get("representation") != self.numerics.representation_id):
            raise ValueError("failed pair case representation/schema differs from request")
        if not self.numerics.is_pair and (completion.get("schema") == "R1PhysicsStudyCaseV2" or "representation" in completion):
            raise ValueError("legacy study cannot inherit failed pair case evidence")
        if completion.get("scientific_checks_passed") is not False:
            raise ValueError("failed case cannot have scientific eligibility")
        if completion.get("failure") != {k: v for k, v in failure.items() if k != "partial_result"}:
            raise ValueError("failed case reason differs from completion")
        partial = failure.get("partial_result")
        persisted = directory/"AcceptedStepsV1.jsonl"
        rows = [json.loads(line) for line in persisted.read_text().splitlines()] if persisted.is_file() else []
        witness_check = None
        if (directory/"FailureWitnessV1.json").exists():
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import verify_failure_witness
            witness_check = verify_failure_witness(checked_read(directory, "FailureWitnessV1.json"),
                source_commit=getattr(self, "source", {}).get("source_commit"),
                protocol=self.failure_protocol(request), failure=failure, result=partial, persisted_rows=rows)
        elif getattr(self, "request", {}).get("schema") in ("R1PhysicsStudyRequestV4", "R1PhysicsStudyRequestV5"):
            raise ValueError("V4 failed study case lacks its saved termination witness")
        if partial is not None and not isinstance(partial, dict):
            raise ValueError("failed scientific payload has no classified record schema")
        if isinstance(partial, dict) and "accepted_steps" in partial and not isinstance(partial["accepted_steps"], list):
            raise ValueError("failed accepted states must be a classified row list")
        if self.numerics.is_pair and isinstance(partial,dict):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import contains_nonfinite_tags, verify_failed_numeric_sidecar
            if contains_nonfinite_tags(partial):
                verify_failed_numeric_sidecar(directory/"StateArraysV1.npz", partial)
                return {"certified":False,"content_matches_recomputed":None,
                    "physical_limits_satisfied":False,"failure_record_checked":True,
                    "numeric_sidecar_checked":True,"finite_state_validity":False,
                    "failure_scope":{"failure_witness":witness_check,"saved_row_count":len(rows)},
                    "reason":"raw nonfinite failure retained; finite equations cannot be certified"}
            if partial.get("schema") == "R1ZeroExcitationV2":
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import _pair_zero_physics
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import verify_numeric_sidecar
                verify_numeric_sidecar(directory/"StateArraysV1.npz", partial)
                _pair_zero_physics(directory, partial, self.prepared(request["intervals"]).to_dict(), failed=True,
                    stack=self.stack,binding=self.binding,protocol={"control":request.get("controls","ABCD"),
                        "nonlinear_factor":.1,"time_substeps":[1,2,4],"policy":ready(r1_policy())})
                return {"certified":False,"content_matches_recomputed":True,
                    "physical_limits_satisfied":False,"saved_zero_controls_reconstructed":sorted(partial["controls"]),
                    "scope":"saved_failed_zero_controls_only_not_scientific_acceptance"}
        if (self.numerics.is_pair and isinstance(partial, dict)
                and partial.get("schema") == "R1FailedPreparationPairV2" and "raw_preparation" in partial):
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import verify_pair_state, verify_numeric_sidecar
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import _legacy_verify
            if (set(partial) != {"schema","representation","raw_preparation","certified","reasons"}
                    or partial.get("representation") != self.numerics.representation_id
                    or partial.get("certified") is not False):
                raise ValueError("failed pair preparation identity differs from its contract")
            raw = partial["raw_preparation"]
            if raw.get("intervals") != request["intervals"] or raw.get("source") != self.source:
                raise ValueError("failed pair preparation grid/source differs from the study")
            verify_pair_state(raw, self.stack, self.binding, policy=r1_policy(), require_pass=False,
                legacy_verify=_legacy_verify, fine_factory=self.numerics.fine_factory,
                snapshot=self.numerics.snapshot)
            verify_numeric_sidecar(directory/"StateArraysV1.npz", partial)
            if (raw["preparation_checks"].get("certified") is not False
                    or partial["reasons"] != raw["preparation_checks"].get("reasons")):
                raise ValueError("failed pair preparation gates contradict saved failure")
            return {"certified":False,"content_matches_recomputed":True,
                "physical_limits_satisfied":False,"failed_preparation_reconstructed":True,
                "scope":"failed_preparation_identity_and_equations_not_scientific_acceptance"}
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import RESULT_ARRAY_FIELDS
        contains_state = (bool(rows) or (isinstance(partial, dict) and (
            bool(partial.get("accepted_steps")) or any(name in partial for name in RESULT_ARRAY_FIELDS))))
        if contains_state:
            if not isinstance(partial, dict) or not partial.get("physics_reconstruction"):
                raise ValueError("present failed states require exact physical reconstruction evidence")
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import failure_scope_report
            if (partial.get("times_s") != ready(request.get("times_s"))
                    or partial.get("control_label") != request.get("control", "D")
                    or partial.get("amplitude_V") != request.get("amplitude_V", .005)
                    or partial.get("policy") != ready(r1_policy(request["nonlinear_factor"],
                                                               time_substeps=request["time_substeps"]))):
                raise ValueError("failed step prefix differs from the planned request")
            audit = verify_r1_step_physics(self.stack, request["intervals"], self.binding,
                                          self.prepared(request["intervals"]), partial, allow_incomplete=True,
                                          backend=self.numerics)
            if self.numerics.is_pair:
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import verify_numeric_sidecar
                verify_numeric_sidecar(directory/"StateArraysV1.npz", partial)
            raw_rows = partial["accepted_steps"]
            persistence = partial.get("persistence_failure")
            matches = rows == raw_rows
            if not matches and persistence:
                # The verifier above has checked all copies and bound this
                # failure to the final observed row. A durable prefix may be
                # missing that one row, or only its post-callback annotation.
                matches = (len(rows) == len(raw_rows)-1 and rows == raw_rows[:-1])
                if len(rows) == len(raw_rows) and rows:
                    last = {name: value for name, value in raw_rows[-1].items() if name != "persistence_failure"}
                    matches = rows[:-1] == raw_rows[:-1] and rows[-1] == last
            if not matches:
                raise ValueError("failed saved rows differ from the raw persisted prefix")
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
            terminal = rebuild_failure_witness(self.stack, request["intervals"], self.binding,
                                              self.prepared(request["intervals"]), partial, backend=self.numerics)
            audit["failure_scope"] = failure_scope_report(partial, audit, persisted_rows=rows,
                failure=completion["failure"], terminal_reconstruction=terminal)
            audit["failure_scope"]["last_observed_row_persisted"] = len(rows) == len(raw_rows)
            audit["failure_scope"]["post_callback_persistence_annotation_replayed"] = False
            audit["terminal_state_reconstruction"] = terminal
            audit["failure_scope"]["failure_witness"] = witness_check
            audit["saved_prefix_physical_limits_satisfied"] = audit["physical_limits_satisfied"]
            audit["physical_limits_satisfied"] = False
            audit["certified"] = False
            return audit
        return {"certified": False, "content_matches_recomputed": None,
                "physical_limits_satisfied": False, "saved_prefix_physical_limits_satisfied": None,
                "failure_record_checked": True,
                "failure_scope": {"saved_row_count": 0, "persisted_row_count": 0,
                                  "failure_witness": witness_check,
                                  "failure_origin_status": "not_reconstructed_from_saved_prefix",
                                  "original_execution_extent_verified": False,
                                  "unverified_present_fields": sorted(partial) if isinstance(partial, dict) else []},
                "reason": "no complete accepted state for physical replay; failed identity retained"}

    def prepared(self, intervals):
        if intervals not in self.preparations:
            result = self.case(f"Preparation/N{intervals}", {"intervals": intervals, "scope": "common_D_equilibrium"},
                               lambda directory: prepare_common_state(self.stack, intervals, self.binding,
                                                                        policy=r1_policy(), backend=self.numerics).to_dict(),
                               dependency=True)
            if result is None:
                if self.args.max_cases is not None and self.attempted >= self.args.max_cases:
                    raise CaseBudgetExhausted("new-case budget exhausted before required preparation")
                raise ValueError("preparation unavailable within selected case budget")
            self.preparations[intervals] = self.numerics.decode_prepared(result)
        self.record_dependency(f"Preparation/N{intervals}")
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
                                 accepted_step_observer=observe, physics_evidence=True,
                                 expected_prepared_sha256=prepared.sha256, backend=self.numerics)
        except Exception as exc:
            partial = getattr(exc, "result", None)
            if isinstance(partial, dict) and partial.get("physics_reconstruction") and partial.get("accepted_steps"):
                try:
                    audit = verify_r1_step_physics(self.stack, intervals, self.binding, prepared,
                                                   partial, allow_incomplete=True, backend=self.numerics)
                    write_json(directory/"FailedPrefixPhysicsAuditV1.json", audit)
                except Exception as audit_exc:
                    write_json(directory/"FailedPrefixPhysicsAuditFailureV1.json", {
                        "type": type(audit_exc).__name__, "message": str(audit_exc),
                        "partial_result": getattr(audit_exc, "result", None),
                    })
            raise
        audit = verify_r1_step_physics(self.stack, intervals, self.binding, prepared, result, backend=self.numerics)
        write_json(directory/"PhysicsRecomputationV1.json", audit)
        if audit.get("certified") is not True:
            error = ValueError("completed trajectory fails reconstructed physical gates")
            error.result = result
            raise error
        return result

    def run(self, sections):
        if "prepare" in sections:
            for n in self.grids:
                if self.selected(f"Preparation/N{n}"):
                    self.prepared(n)
        if "zero" in sections:
            for n in self.grids:
                self.case(f"Zero/N{n}", {"intervals": n, "controls": "ABCD", "scope": "zero_excitation_A_D"},
                          lambda directory, n=n: check_zero_excitation(self.stack, n, self.binding, self.prepared(n),
                              expected_prepared_sha256=self.prepared(n).sha256, backend=self.numerics))
        if "short" in sections:
            short_times = self.times if getattr(self, "window", "functional") == "diagnostic" else DEFAULT_TIMES_S
            for n in self.grids:
                for control in "ABCD":
                    self.case(f"Short/N{n}/{control}", {"intervals": n, "control": control,
                              "times_s": short_times, "time_substeps": (1, 2, 4), "nonlinear_factor": .1,
                              "scope": "short_controlled_physics_recomputation"},
                              lambda directory, n=n, c=control: self.step(directory, n, c, (1, 2, 4), .1, short_times))
        if "matrix" in sections:
            for item in self.matrix_cases():
                self.case(self.matrix_key(item), {**dataclasses.asdict(item), "times_s": self.times,
                               "scope": self.window+"_window_independent_axis_case"},
                          lambda directory, item=item: self.step(directory, item.intervals, item.control, item.time_substeps,
                                                                 item.nonlinear_factor, self.times))
        if "long" in sections:
            self.case("Long/N16/D", {"intervals": 16, "control": "D", "time_substeps": (1, 2, 4),
                      "nonlinear_factor": .1, "times_s": observation_times(), "scope": "full_100s_window_attempt"},
                      lambda directory: self.step(directory, 16, "D", (1, 2, 4), .1, observation_times()))
        if "dc" in sections:
            for n in self.grids:
                for control in "ABCD":
                    self.case(f"DCBaseline/N{n}/{control}", self.dc_state_request(n, control, 0.),
                              lambda directory, n=n, c=control: self.solve_dc_state(n, c, 0.))
                    self.case(f"DC/N{n}/{control}", self.dc_study_request(n, control),
                              lambda directory, n=n, c=control: self.dc_record(n, c))
        if "amplitude-dc" in sections:
            for n in self.grids:
                for control in self.controls:
                    self.case(f"AmplitudeDC/N{n}/{control}", {
                        "kind": "dc_amplitude_endpoints", "intervals": n, "control": control,
                        "operating_voltage_V": 0., "amplitudes_V": list(AMPLITUDES_V),
                        "nonlinear_factor": .1, "time_substeps": (1, 2, 4),
                        "scope": "dc_endpoint_ladder_without_transient_linearity_acceptance"},
                        lambda directory, n=n, c=control: dc_amplitude_endpoint_study(
                            self.stack, n, self.binding, self.prepared(n), control=c,
                            expected_prepared_sha256=self.prepared(n).sha256))
        if "ac" in sections:
            for n in self.grids:
                self.case(f"AC/N{n}/D", {"intervals": n, "control": "D", "frequency_Hz": self.frequencies,
                          "scope": "direct_zero_bias_ac_three_derivative_levels_not_window_coverage"},
                          lambda directory, n=n: small_signal_response(self.zero_dc(n, "D"), self.frequencies))
        if "amplitude" in sections:
            previous = None
            for amplitude in getattr(self, "amplitudes", AMPLITUDES_V):
                for item in self.amplitude_cases(amplitude):
                    self.case(self.amplitude_key(item), {**dataclasses.asdict(item), "times_s": self.times,
                              "scope": self.window+"_amplitude_independent_axis_case"},
                              lambda directory, item=item: self.step(directory, item.intervals, "D", item.time_substeps,
                                  item.nonlinear_factor, self.times, item.amplitude_V))
                if previous is not None:
                    result = self.case(f"Linearity/A{previous}ToA{amplitude}", {"kind": "amplitude_linearity", "coarse_amplitude_V": previous,
                              "fine_amplitude_V": amplitude, "scope": self.window+"_normalized_current_and_refinement_error"},
                              lambda directory, a=previous, b=amplitude: self.amplitude_record(a, b))
                previous = amplitude
        if "compare" in sections:
            for control in "ABCD":
                for left, right in zip(self.grids[:-1], self.grids[1:]):
                    self.case(f"Compare/{control}/N{left}N{right}", {"left": left, "right": right, "control": control,
                              "scope": "short_fixed_position_spatial_comparison"},
                              lambda directory, l=left, r=right, c=control: self.compare_available(
                                  [f"Short/N{l}/{c}", f"Short/N{r}/{c}"],
                                  lambda: self.short_comparison(l, r, c)))
            for left, right in zip(self.grids[:-1], self.grids[1:]):
                self.case(f"Compare/AC/N{left}N{right}", {"left": left, "right": right, "control": "D",
                          "scope": "direct_ac_real_and_imaginary_mesh_comparison"},
                          lambda directory, l=left, r=right: self.compare_available(
                              [f"AC/N{l}/D", f"AC/N{r}/D"], lambda: self.ac_mesh_comparison(l, r)))
            self.case("Compare/LongTailN16D", {"scope": "finite_accepted_tail_vs_5mV_dc_no_infinite_tail_bound"},
                      lambda directory: self.tail_record())
            self.matrix_comparisons()
        if "windows" in sections:
            n = self.grids[-1]
            amplitude, linearity = self.qualified_amplitude()
            start, end = self.args.first_time_s, self.args.last_time_s
            windows = (("Base", start, end), ("Extended", start, min(end*10, 1e5)),
                       ("Earlier", max(start/10, 1e-12), end))
            for label, first, last in windows:
                if label == "Extended" and last == end or label == "Earlier" and first == start:
                    continue
                times = observation_times(first_time_s=first, last_time_s=last)
                if (not getattr(self, "planning", False)
                        and self.selected(f"Window/{label}/N{n}/A{amplitude}")):
                    self.target_dc(n, "D", amplitude)
                    self.case(f"DC/N{n}/D", self.dc_study_request(n, "D"),
                              lambda directory, n=n: self.dc_record(n, "D"), dependency=True)
                self.case(f"Window/{label}/N{n}/A{amplitude}", {"intervals": n, "control": "D", "amplitude_V": amplitude,
                          "time_substeps": (4, 8, 16), "nonlinear_factor": .01, "times_s": times,
                          "linearity_case": linearity, "scope": "declared_full_window_extension"},
                          lambda directory, n=n, times=times, a=amplitude: self.step(directory, n, "D", (4, 8, 16), .01, times, a))
        if "reconstruct" in sections:
            n = self.grids[-1]
            for grid in self.grids:
                self.case(f"Frequency/N{grid}", {"kind": "frequency_coverage", "grid": grid,
                          "scope": "numerically_valid_bands_and_uncovered_turnovers"},
                          lambda directory, grid=grid: self.compare_available([f"AC/N{grid}/D"],
                              lambda: self.frequency_record(grid),
                              require_scientific=False))
            self.case(f"DoubleDomain/N{n}/D", {"kind": "double_domain", "grid": n,
                      "scope": "derived_double_domain_eligibility_with_explicit_unknown_errors"},
                      lambda directory: self.reconstruction_record(n))

    def reconstruction_record(self, n):
        amplitude, linearity_key = self.qualified_amplitude()
        keys = [f"Window/Base/N{n}/A{amplitude}", f"AC/N{n}/D", f"DC/N{n}/D"]
        def reconstruct():
            step, ac, dc = (self.saved(key) for key in keys)
            prepared = self.prepared(n)
            scope = self.response_scope(step, n)
            external = self.qualification_inputs["double_domain_evidence"].get(f"DoubleDomain/N{n}/D", {})
            if set(external) - {"errors", "prerequisites"}:
                raise ValueError("unknown externally supplied double-domain fields")
            qualification_args = {"expected_scope": scope, "window_spec": self.window_spec,
                "qualification_evidence": external.get("prerequisites"),
                "trusted_evidence": self.qualification_inputs["trusted_evidence"]}
            prerequisites = {"input_trajectory_certified": self.verified_cases[keys[0]][1].get("certified") is True,
                "ac_content_verified": self.verified_cases[keys[1]][1].get("content_matches_recomputed") is True,
                "finite_amplitude_linearity": False,
                "frequency_window_coverage": self.frequency_record(n)["device_frequency_window_certified"]}
            error_fields, uncertainty = {}, {}
            if linearity_key is not None and self.window == "full":
                linearity = self.saved(linearity_key, require_scientific=False)
                prerequisites["finite_amplitude_linearity"] = linearity["linearity_certified"]
                _, numerical_error, uncertainty = self.amplitude_error(amplitude)
                if np.array_equal(numerical_error.coordinates["time_s"], step["times_s"]):
                    raw_errors = []
                    for axis in uncertainty["axes"].values():
                        left, right = (self.saved(key) for key in axis["cases"][-2:])
                        a = np.asarray([row["report_contact_current_A_m2"] for row in left["regular_currents"]])
                        b = np.asarray([row["report_contact_current_A_m2"] for row in right["regular_currents"]])
                        raw_errors.append(np.abs(a-b))
                    error_fields["current_A_m2"] = np.sum(raw_errors, axis=0)[1:, 0]
                prerequisites["single_axis_convergence"] = uncertainty["axes_passed"]
                center = next(c for c in self.amplitude_cases(amplitude) if c.intervals == n
                              and c.time_substeps == (4, 8, 16) and c.nonlinear_factor == .01)
                neighbors = [c for c in self.amplitude_cases(amplitude) if c.intervals < n
                             and c.time_substeps == center.time_substeps and c.nonlinear_factor == center.nonlinear_factor]
                if neighbors:
                    neighbor = max(neighbors, key=lambda c: c.intervals)
                    error_fields["baseline_current_A_m2"] = abs(dc["conductance"]["baseline"]["terminal_current_A_m2"]
                        - self.zero_dc(neighbor.intervals, "D").evidence["terminal_current_A_m2"])
                    adjacent_dc = self.saved(f"DC/N{neighbor.intervals}/D")["conductance"]
                    error_fields["dc_conductance_S_m2"] = (dc["conductance"]["finest_pair_absolute_difference_S_m2"]
                        + abs(dc["conductance"]["conductance_S_m2"][-1]-adjacent_dc["conductance_S_m2"][-1]))
            # Refinement differences above remain estimates. Only separately
            # reviewed inputs can supply reconstruction bounds, and their
            # precise values are included in the qualification application.
            approved_error_fields = external.get("errors", {})
            errors = R1AdmittanceErrors(**approved_error_fields)
            reference = reconstruct_study_response(step, prepared, dc["conductance"], ac, errors=errors,
                                                   prerequisites=prerequisites, **qualification_args)
            tight = reconstruct_study_response(step, prepared, dc["conductance"], ac, errors=errors,
                prerequisites=prerequisites, quadrature_absolute_tolerance_F_m2=1e-13, quadrature_relative_tolerance=1e-11,
                **qualification_args)
            comparisons = {"stricter_integration": self.reconstruction_comparison(reference, tight)}
            prerequisites["stricter_integration"] = comparisons["stricter_integration"]["passed"]
            for label, gate in (("Extended", "window_extension"), ("Earlier", "earlier_start")):
                key = f"Window/{label}/N{n}/A{amplitude}"
                available = self.compare_available([key], lambda key=key: self.saved(key))
                if available.get("comparison_available") is False:
                    comparisons[gate] = available
                    prerequisites[gate] = False
                    continue
                other = reconstruct_study_response(available, prepared, dc["conductance"], ac)
                comparisons[gate] = self.reconstruction_comparison(reference, other)
                prerequisites[gate] = comparisons[gate]["passed"]
            # The measured tail state is compared against an independently
            # solved DC at the actual selected amplitude.
            target = self.target_dc(n, "D", amplitude)
            row = next(r for r in reversed(step["accepted_steps"]) if r["time_s"] == step["times_s"][-1]
                       and r["substeps"] == max(step["policy"]["refinement_substeps"]))
            tail = {**row["state"], "prepared_sha256": step["prepared_sha256"],
                    "control": step["control_label"], "voltage_V": step["amplitude_V"]}
            tail_report = compare_transient_tail(target, initial_state=prepared.to_dict()["state"], tail_state=tail,
                tail_regular_current_A_m2=step["regular_currents"][-1]["report_contact_current_A_m2"][0], time_s=row["time_s"])
            prerequisites["tail_dc_agreement"] = tail_report["all_observables_agree"]
            result = reconstruct_study_response(step, prepared, dc["conductance"], ac, errors=errors,
                                                prerequisites=prerequisites, **qualification_args)
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import device_response_gate
            stage_gate = device_response_gate(result, expected_scope=scope, window_spec=self.window_spec,
                required_frequency_Hz=getattr(self, "required_frequency_Hz", self.frequencies))
            return {**result, "device_stage_gate": stage_gate,
                    "linearity_case": linearity_key, "prerequisite_evidence": comparisons,
                    "tail_dc_evidence": tail_report, "current_uncertainty_evidence": uncertainty,
                    "provided_error_estimates": ready(error_fields),
                    "error_estimate_classification": "estimate_only",
                    "externally_supplied_error_bounds": ready(approved_error_fields),
                    "unestablished_error_budgets": [name for name in R1AdmittanceErrors.__dataclass_fields__
                                                     if name not in approved_error_fields]}
        return self.compare_available(keys, reconstruct)

    @staticmethod
    def reconstruction_comparison(left, right):
        def response(record):
            reconstruction = record["reconstruction"]
            values = np.array([complex(x["real"], x["imag"]) for x in reconstruction["admittance_S_m2"]])
            return R1Response(values, {"frequency_Hz": np.asarray(reconstruction["frequency_Hz"])})
        return compare_responses("admittance", response(left), response(right))

    def qualified_amplitude(self):
        # The chosen diagnostic amplitude is fixed before any computation.
        # Eligibility is evaluated separately; it cannot change the case universe.
        if hasattr(self, "plan") or getattr(self.args, "plan_only", False):
            amplitude = getattr(self.args, "window_amplitude", .005)
            key = getattr(self.args, "linearity_case", None)
            if key is not None and getattr(self, "qualification_analysis", False):
                report = self.saved(key)
                from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import (
                    evidence_digest, select_response_amplitude,
                )
                qualification = report.get("qualification", {})
                prepared = self.prepared(self.grids[-1]).to_dict()
                scope = self.response_scope(prepared, self.grids[-1])
                selection = select_response_amplitude([qualification], expected_scope=scope,
                    verified_report_digests=[evidence_digest(qualification)], diagnostic_amplitude_V=amplitude,
                    window_spec=self.window_spec, declared_amplitudes_V=self.amplitudes)
                if selection["device_qualified"] is not True or selection["qualified_amplitude_V"] != amplitude:
                    raise ValueError("requested window amplitude lacks the specified verified linearity evidence")
            return amplitude, key
        candidates = []
        for directory in sorted((self.output/"Linearity").glob("*/AttemptV*")):
            if not (directory/"CompletionV1.json").is_file():
                continue
            completion = checked_read(directory, "CompletionV1.json")
            key = completion["case"]
            if directory != self.latest(key) or completion.get("scientific_checks_passed") is not True:
                continue
            request = checked_read(directory, "RequestV1.json")
            if not hasattr(self, "operations"):
                self.operations = {}
            self.operations[key] = lambda ignored, r=request: self.amplitude_record(r["coarse_amplitude_V"], r["fine_amplitude_V"])
            result = self.saved(key)
            if result.get("linearity_certified") is True:
                candidates.append((request["fine_amplitude_V"], key))
        return min(candidates) if candidates else (.005, None)

    def dc_record(self, n, control):
        prepared = self.prepared(n)
        target = solve_controlled_dc(self.stack, n, self.binding, prepared, control=control, voltage_V=.005,
                                     expected_prepared_sha256=prepared.sha256)
        return {"target_bias": target.evidence,
                "conductance": dc_conductance_study(self.stack, n, self.binding, prepared, control=control,
                                                    expected_prepared_sha256=prepared.sha256)}

    @staticmethod
    def dc_state_request(n, control, voltage):
        return {"intervals": n, "control": control, "voltage_V": voltage,
                "scope": "sealed_same_control_dc_state_for_derived_analysis"}

    @staticmethod
    def dc_study_request(n, control):
        return {"intervals": n, "control": control, "voltage_V": .005,
                "scope": "same_control_5mV_dc_and_three_step_zero_bias_conductance"}

    @staticmethod
    def target_dc_key(n, control, amplitude):
        return f"TargetDC/N{n}/{control}/A{amplitude}"

    def solve_dc_state(self, n, control, voltage):
        prepared = self.prepared(n)
        return solve_controlled_dc(self.stack, n, self.binding, prepared, control=control,
            voltage_V=voltage, expected_prepared_sha256=prepared.sha256).evidence

    def target_dc(self, n, control, amplitude):
        key = self.target_dc_key(n, control, amplitude)
        record = self.case(key, self.dc_state_request(n, control, amplitude),
            lambda directory: self.solve_dc_state(n, control, amplitude), dependency=True)
        if record is None:
            raise ValueError("required target DC has not been collected: " + key)
        return restore_controlled_dc(record, self.stack, n, self.binding, self.prepared(n))

    @staticmethod
    def amplitude_key(item):
        return (f"Amplitude/A{item.amplitude_V}/N{item.intervals}/T{item.time_substeps[0]}"
                f"/F{str(item.nonlinear_factor).replace('.', 'p')}")

    def amplitude_cases(self, amplitude):
        choices = convergence_cases(intervals=self.grids[-3:], amplitude_V=amplitude)
        return tuple(c for c in choices if sum((c.intervals != self.grids[-1],
            c.time_substeps != (4, 8, 16), c.nonlinear_factor != .01)) <= 1)

    def current_response(self, item, key):
        record = self.saved(key)
        baseline = self.zero_dc(item.intervals, item.control).evidence["current_A_m2"][[0, -1]]
        values = np.asarray([row["report_contact_current_A_m2"] for row in record["regular_currents"]])-baseline
        return R1Response(values, {"time_s": np.asarray(record["times_s"])}, ("left_contact", "right_contact"))

    def amplitude_error(self, amplitude):
        choices = self.amplitude_cases(amplitude)
        center = next(c for c in choices if c.intervals == self.grids[-1]
                      and c.time_substeps == (4, 8, 16) and c.nonlinear_factor == .01)
        response = self.current_response(center, self.amplitude_key(center))
        axes, differences = {}, []
        for axis in ("intervals", "time_substeps", "nonlinear_factor"):
            group = [c for c in choices if all(getattr(c, name) == getattr(center, name)
                     for name in ("intervals", "time_substeps", "nonlinear_factor") if name != axis)]
            group.sort(key=lambda c: getattr(c, axis), reverse=axis == "nonlinear_factor")
            if len(group) != 3:
                raise ValueError("amplitude numerical-error estimate requires three levels on every independent axis")
            responses = [self.current_response(c, self.amplitude_key(c)) for c in group]
            reports = [compare_responses("regular_current_response", a, b)
                       for a, b in zip(responses[:-1], responses[1:])]
            differences.append(np.abs(responses[-1].values-responses[-2].values))
            axes[axis] = {"cases": [self.amplitude_key(c) for c in group], "comparisons": reports,
                          "fine_difference_A_m2": differences[-1], "passed": reports[-1]["passed"],
                          "coarser_comparison_is_diagnostic": True}
        error = np.sum(differences, axis=0)
        return response, R1Response(error, response.coordinates, response.components), {
            "method": "sum of separately measured finest-pair absolute differences on three axes",
            "interpretation": "empirical numerical uncertainty estimate; not a rigorous continuum error bound",
            "axes": axes, "axes_passed": all(a["passed"] for a in axes.values()),
            "error_A_m2": error}

    def amplitude_record(self, coarse=.005, fine=.0025):
        keys = [self.amplitude_key(c) for a in (coarse, fine) for c in self.amplitude_cases(a)]
        def compare():
            a, error_a, audit_a = self.amplitude_error(coarse)
            b, error_b, audit_b = self.amplitude_error(fine)
            report = compare_amplitude_halving(a, b, coarse_amplitude_V=coarse, fine_amplitude_V=fine,
                                               coarse_current_error=error_a, fine_current_error=error_b)
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import (
                assess_transient_linearity, evidence_digest,
            )
            centers = [next(c for c in self.amplitude_cases(amplitude)
                            if c.intervals == self.grids[-1] and c.time_substeps == (4, 8, 16)
                            and c.nonlinear_factor == .01) for amplitude in (coarse, fine)]
            center_keys = [self.amplitude_key(c) for c in centers]
            steps = [self.saved(key) for key in center_keys]
            scope = self.response_scope(steps[0], self.grids[-1])
            trusted = dict(self.qualification_inputs["trusted_evidence"])
            trusted.update({"verified_step:"+key: evidence_digest(step) for key, step in zip(center_keys, steps)})
            baseline = self.zero_dc(self.grids[-1], "D").evidence["current_A_m2"][[0, -1]]
            qualification = assess_transient_linearity(*steps, coarse_baseline_A_m2=baseline,
                fine_baseline_A_m2=baseline, expected_scope=scope, expected_times_s=self.times,
                coarse_budget=self.qualification_inputs["current_budgets"].get(center_keys[0]),
                fine_budget=self.qualification_inputs["current_budgets"].get(center_keys[1]), trusted_evidence=trusted,
                window_spec=self.window_spec, declared_amplitudes_V=self.amplitudes,
                verified_record_digests={"coarse_step": evidence_digest(steps[0]), "fine_step": evidence_digest(steps[1])},
                measured_evidence={"coarse_current_error": error_a, "fine_current_error": error_b,
                    "coarse_axes_passed": audit_a["axes_passed"], "fine_axes_passed": audit_b["axes_passed"],
                    "comparison": report},
                reconciliation_evidence=self.qualification_inputs.get("numerical_reconciliations", {}).get(
                    f"Linearity/A{coarse}ToA{fine}"))
            qualified = (audit_a["axes_passed"] and audit_b["axes_passed"]
                         and qualification["device_qualified"] is True)
            return {"schema": "R1AmplitudeLinearityV3", "comparison": report,
                    "coarse_amplitude_V": coarse, "fine_amplitude_V": fine,
                    "coarse_error_evidence": audit_a, "fine_error_evidence": audit_b,
                    "qualification": qualification, "linearity_certified": qualified,
                    "within_compared_budgets": qualified,
                    "empirical_comparison_classification": "estimate_only",
                    "scope": self.window+"_window_two_amplitude_diagnostics_and_externally_bound_qualification"}
        return self.compare_available(keys, compare)

    def response_scope(self, record, intervals):
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import evidence_digest, qualification_scope
        prepared = self.prepared(intervals).to_dict()
        return qualification_scope(record, state_sha256=evidence_digest(prepared["state"]), domain="device",
                                   window_spec=self.window_spec)

    def frequency_record(self, intervals):
        key = f"AC/N{intervals}/D"
        ac = self.saved(key, require_scientific=False)
        return frequency_window_report(ac, expected_scope=self.response_scope(ac, intervals),
            turnover_evidence=self.qualification_inputs["turnover_evidence"].get(key),
            trusted_evidence=self.qualification_inputs["trusted_evidence"], window_spec=self.window_spec)

    def short_comparison(self, left, right, control):
        keys = (f"Short/N{left}/{control}", f"Short/N{right}/{control}")
        a, b = (self.saved(key) for key in keys)
        electrical = compare_step_current_charge(a, b, self.zero_dc(left, control).evidence["current_A_m2"][[0, -1]],
                                                self.zero_dc(right, control).evidence["current_A_m2"][[0, -1]])
        return compare_convergence_responses(spatial_responses_from_step(self.prepared(left), a),
            spatial_responses_from_step(self.prepared(right), b), axis="intervals", electrical=electrical,
            input_verifications=tuple(self.verified_cases[key][1] for key in keys))

    @staticmethod
    def matrix_key(item):
        prefix = "Matrix" if item.control == "D" else "Matrix/"+item.control
        return f"{prefix}/N{item.intervals}/T{item.time_substeps[0]}/F{str(item.nonlinear_factor).replace('.', 'p')}"

    def matrix_cases(self):
        cases = []
        for control in self.controls:
            choices = convergence_cases(intervals=self.grids, control=control)
            if control != "D":
                choices = tuple(c for c in choices if sum((c.intervals != self.grids[-1],
                    c.time_substeps != (4, 8, 16), c.nonlinear_factor != .01)) <= 1)
            cases.extend(choices)
        return tuple(cases)

    def matrix_comparisons(self):
        cases = self.matrix_cases()
        fields = ("intervals", "time_substeps", "nonlinear_factor")
        for axis in fields:
            groups = {}
            for item in cases:
                other = (item.control, *(getattr(item, key) for key in fields if key != axis))
                groups.setdefault(other, []).append(item)
            for group in groups.values():
                group = sorted(group, key=lambda item: getattr(item, axis), reverse=axis == "nonlinear_factor")
                for left, right in zip(group[:-1], group[1:]):
                    left_key, right_key = self.matrix_key(left), self.matrix_key(right)
                    label = left_key.removeprefix("Matrix/").replace("/", "")+"To"+right_key.removeprefix("Matrix/").replace("/", "")
                    self.case(f"MatrixCompare/{axis}/{label}", {
                        "axis": axis, "left_case": left_key, "right_case": right_key,
                        "control": left.control,
                        "required_finest_pair": (right.intervals == self.grids[-1]
                            and right.time_substeps == (4, 8, 16) and right.nonlinear_factor == .01),
                        "scope": self.window+"_window_comparison_with_other_two_axes_fixed"},
                        lambda directory, l=left, r=right: self.compare_available(
                            [self.matrix_key(l), self.matrix_key(r)], lambda: self.matrix_comparison(l, r)))

    def compare_available(self, keys, operation, *, require_scientific=True):
        missing = []
        for key in keys:
            directory = self.latest(key)
            if directory is None:
                missing.append({"case": key, "status": "not_run"})
            elif not (directory/"CompletionV1.json").is_file() or not (directory/"ManifestV1.json").is_file():
                missing.append({"case": key, "status": "interrupted"})
            else:
                completion = checked_read(directory, "CompletionV1.json")
                if completion["status"] != "completed" or require_scientific and completion.get("scientific_checks_passed") is not True:
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
        baseline_left = self.zero_dc(left.intervals, left.control)
        baseline_right = self.zero_dc(right.intervals, right.control)
        electrical = compare_step_current_charge(a, b, baseline_left.evidence["current_A_m2"][[0, -1]],
                                                baseline_right.evidence["current_A_m2"][[0, -1]])
        axes = [field for field in ("intervals", "time_substeps", "nonlinear_factor")
                if getattr(left, field) != getattr(right, field)]
        if len(axes) != 1:
            raise ValueError("convergence comparison must change exactly one axis")
        return compare_convergence_responses(
            spatial_responses_from_step(self.prepared(left.intervals), a),
            spatial_responses_from_step(self.prepared(right.intervals), b), axis=axes[0], electrical=electrical,
            input_verifications=(self.verified_cases[self.matrix_key(left)][1],
                                 self.verified_cases[self.matrix_key(right)][1]))

    def zero_dc(self, intervals, control):
        key = intervals, control
        if key not in self.dc_baselines:
            record = self.case(f"DCBaseline/N{intervals}/{control}", self.dc_state_request(intervals, control, 0.),
                lambda directory: self.solve_dc_state(intervals, control, 0.), dependency=True)
            if record is None:
                raise ValueError("required baseline DC has not been collected")
            self.dc_baselines[key] = restore_controlled_dc(record, self.stack, intervals, self.binding,
                                                          self.prepared(intervals))
        return self.dc_baselines[key]

    def tail_record(self):
        prepared = self.prepared(16)
        attempt = self.latest("Long/N16/D")
        if attempt is None:
            return {"comparison_available": False, "reason": "full-window attempt not available"}
        completion = checked_read(attempt, "CompletionV1.json")
        if completion["status"] == "completed":
            record = self.saved("Long/N16/D", require_scientific=False)
        else:
            audit = self.audit_failed("Long/N16/D", attempt, checked_read(attempt, "RequestV1.json"))
            self.record_dependency("Long/N16/D")
            if audit.get("content_matches_recomputed") is not True:
                return {"comparison_available": False, "reason": "failed tail has no verified physical reconstruction"}
            record = checked_read(attempt, "FailureV1.json")["partial_result"]
        if not isinstance(record, dict):
            return {"comparison_available": False, "reason": "failed window has no saved physical state"}
        rows = [row for row in record.get("accepted_steps", []) if row.get("physical_checks_passed") and row["dt_s"] > 0]
        if not rows:
            return {"comparison_available": False, "reason": "no physically accepted finite tail is available"}
        row = max(rows, key=lambda r: (r["time_s"], r["substeps"]))
        if (record.get("prepared_sha256") != prepared.sha256 or record.get("control_label") != "D"
                or record.get("amplitude_V") != .005 or record.get("reference_sha256") != self.binding["sha256"]):
            raise ValueError("long-window tail identity differs from its verified preparation")
        state = {**row["state"], "prepared_sha256": record["prepared_sha256"],
                 "control": record["control_label"], "voltage_V": record["amplitude_V"]}
        dc = solve_controlled_dc(self.stack, 16, self.binding, prepared, voltage_V=.005,
                                expected_prepared_sha256=prepared.sha256)
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

    def verify_recorded_invocations(self):
        """Verify each sealed invocation's local count without conflating resumes."""
        recorded = {}
        for directory in sorted(self.output.rglob("AttemptV*")):
            if directory.is_dir() and (directory / "RequestV1.json").is_file():
                recorded[directory.relative_to(self.output).as_posix()] = json.loads(
                    (directory / "RequestV1.json").read_text())
        claimed = set()
        invocations = sorted((self.output / "Invocations").glob("InvocationV*.json"),
                             key=lambda path: int(path.stem.removeprefix("InvocationV")))
        if not invocations:
            raise ValueError("formal study has no recorded invocation accounting")
        for path in invocations:
            value = checked_read(self.output, path.relative_to(self.output).as_posix())
            if value.get("study_request_sha256") != self.plan_sha256:
                raise ValueError("recorded invocation study_request_sha256 differs from external request")
            current = verify_invocation_attempts(value, planned_cases=self.planned_cases,
                                                recorded_attempts=recorded)
            if claimed & current:
                raise ValueError("formal attempt is claimed by more than one invocation")
            claimed.update(current)
        published = checked_read(self.output, "StudySummaryV1.json")
        verify_invocation_attempts(published, planned_cases=self.planned_cases, recorded_attempts=recorded)
        for name in ("attempted_cases", "attempt_count_scope", "invocation_attempts"):
            if published.get(name) != value.get(name):
                raise ValueError("published invocation accounting differs from its latest recorded invocation")

    def finish(self):
        failures, unavailable, active, historical = [], [], {}, []
        interrupted = []
        for attempt_dir in sorted(self.output.rglob("AttemptV*")):
            if not attempt_dir.is_dir():
                continue
            key = attempt_dir.parent.relative_to(self.output).as_posix()
            if getattr(self, "run_class", "development") == "formal" and hasattr(self, "plan"):
                if key not in self.planned_cases or attempt_dir.name != "AttemptV1":
                    raise ValueError("formal study contains an unplanned case or repeated attempt: " + str(attempt_dir))
            if not (attempt_dir/"CompletionV1.json").is_file():
                item = {"case": key, "directory": str(attempt_dir.relative_to(self.output)),
                        "status": "interrupted", "scientific_checks_passed": False,
                        "failure": "attempt has no completed result; original extent unknown"}
                interrupted.append(item)
                failures.append(item)
        for completion_file in sorted(self.output.rglob("CompletionV1.json")):
            case_dir = completion_file.parent
            if not (case_dir/"ManifestV1.json").is_file():
                failures.append({"directory": str(case_dir.relative_to(self.output)),
                                 "failure": "interrupted before final case manifest"})
                continue
            completion = checked_read(case_dir, "CompletionV1.json")
            key = completion["case"]
            row = {"case": key, "directory": str(case_dir.relative_to(self.output)),
                   "completion": completion, "conditions": checked_read(case_dir, "RequestV1.json")}
            if case_dir != self.latest(key):
                historical.append(row)
                continue
            active[key] = row
            if completion["status"] == "unavailable":
                unavailable.append({"directory": str(case_dir.relative_to(self.output)), "completion": completion,
                                    "conditions": checked_read(case_dir, "RequestV1.json"),
                                    "result": checked_read(case_dir, "ResultV1.json")})
            if completion["status"] == "failed" or completion.get("scientific_checks_passed") is False:
                failures.append({"directory": str(case_dir.relative_to(self.output)), "completion": completion,
                                 "conditions": checked_read(case_dir, "RequestV1.json"),
                                 "failure": (checked_read(case_dir, "FailureV1.json").get("message")
                                             if completion["status"] == "failed" else "returned scientific comparisons failed")})
        inventory_path = self.output/"CaseInventoryV2.json"
        inventory = json.loads(inventory_path.read_text()) if inventory_path.is_file() else {}
        for key, value in getattr(self, "expected_cases", {}).items():
            if key in inventory and inventory[key] != value:
                raise ValueError("frozen case inventory request changed: "+key)
            inventory[key] = value
        missing = sorted(set(inventory)-set(active))
        if hasattr(self, "plan"):
            missing = inventory_coverage(self.plan, inventory, active)
        unknown = [key for key, row in active.items() if row["completion"].get("scientific_checks_passed") is None]
        unchecked = []
        if getattr(self, "run_class", "development") == "formal":
            for key, row in active.items():
                digest = sha(self.output/row["directory"]/"ManifestV1.json")
                verification = getattr(self, "verified_cases", {}).get(key)
                produced = getattr(self, "produced_cases", {}).get(key)
                if (verification is None or verification[0] != digest) and produced != digest:
                    unchecked.append(key)
        local_fail = any(row["status"] == "failed" for row in self.rows)
        required_pairs = [row for row in active.values() if row["conditions"].get("required_finest_pair") is True]
        pair_coverage = {(row["conditions"].get("control"), row["conditions"].get("axis")) for row in required_pairs}
        expected_pairs = {(control, axis) for control in getattr(self, "controls", ("D",))
                          for axis in ("intervals", "time_substeps", "nonlinear_factor")}
        finest_passed = expected_pairs <= pair_coverage and all(
            row["completion"].get("scientific_checks_passed") is True for row in required_pairs)
        if hasattr(self, "plan"):
            planned_finest = set(self.plan["required_finest_cases"])
            finest_passed = bool(planned_finest) and all(
                key in active and active[key]["completion"].get("scientific_checks_passed") is True
                for key in planned_finest)
        requirements = {
            "formal_source_execution": getattr(self, "run_class", "development") == "formal",
            "all_recorded_cases_checked": not unchecked,
            "verification_completed_without_error": not any(row.get("case") == "invocation"
                and row.get("status") == "failed" for row in self.rows),
            "declared_cases_available": not missing and bool(inventory),
            "required_finest_comparisons_passed": finest_passed,
            "full_window_base_27": getattr(self, "window", "functional") == "full" and all(
                self.matrix_key(c) in active for c in base_convergence_cases()),
            "three_independent_axis_comparisons": all(any(k.startswith("MatrixCompare/"+axis+"/") for k in active)
                for axis in ("intervals", "time_substeps", "nonlinear_factor")),
            "amplitude_linearity": any(row["conditions"].get("kind") == "amplitude_linearity"
                and row["completion"].get("scientific_checks_passed") is True for row in active.values()),
            "window_and_double_domain": any(row["conditions"].get("kind") == "double_domain"
                and row["completion"].get("scientific_checks_passed") is True for row in active.values()),
        }
        if hasattr(self, "plan"):
            requirements["external_case_set_verified"] = self.plan_sha256 is not None
            requirements["caller_approved_standard_bound"] = getattr(self, "standard", {}).get("matches_external_approval") is True
            requirements["all_four_control_axes_available"] = {
                (control, axis) for control in "ABCD"
                for axis in ("intervals", "time_substeps", "nonlinear_factor")} <= pair_coverage
            requirements["full_window_base_27_available"] = requirements.pop("full_window_base_27")
            requirements["three_axes_recorded"] = requirements.pop("three_independent_axis_comparisons")
        summary = {"schema": "R1PhysicsStudySummaryV3" if self.numerics.is_pair else "R1PhysicsStudySummaryV2", "started_utc": self.started,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                   "cases": self.rows, "attempted_cases": self.attempted,
                   "attempt_count_scope": "new_attempts_in_this_invocation",
                   "invocation_attempts": getattr(self, "invocation_attempts", []),
                   "active_case_count": len(active), "historical_attempt_count": len(historical),
                   "diagnostic_failure_count": len(failures),
                   "required_finest_pair_count": len(required_pairs),
                   "missing_cases": missing, "unresolved_cases": unknown,
                   "not_recomputed_in_this_invocation": unchecked,
                   "requirements": requirements, "study_exit_passed": all(requirements.values()),
                   "missing_for_R1_2_exit": [k for k, v in requirements.items() if not v],
                   "independent_acceptance": "not_asserted",
                   "scope": "numerical_study_criteria_do_not_replace_independent_review"}
        if hasattr(self, "plan"):
            summary.update(planned_case_count=len(self.planned_cases),
                           interrupted_attempts=interrupted,
                           study_request_sha256=self.plan_sha256,
                           diagnostic_failed_case_count=len(failures),
                           comparison_axes_recorded=sorted({row["conditions"].get("axis") for row in active.values()
                               if row["case"].startswith("MatrixCompare/")}),
                           availability_does_not_imply_convergence=True,
                           verification_does_not_imply_scientific_acceptance=True)
        if self.numerics.is_pair:
            summary["representation"] = self.numerics.representation_id
            summary["pair_scope"] = self.request["pair_scope"]
        code = 1 if failures or local_fail else 2 if missing or unavailable or unknown or unchecked else 0
        summary["exit_code"] = code
        self.last_summary = summary
        if getattr(self, "verifying", False):
            if getattr(self, "run_class", "development") == "formal" and hasattr(self, "plan"):
                self.verify_recorded_invocations()
            published = checked_read(self.output, "StudySummaryV1.json")
            if hasattr(self, "context"):
                verify_study_source(self.output, self.context, require_read_receipt=True)
            if hasattr(self, "plan"):
                if published.get("study_request_sha256") != self.plan_sha256:
                    raise ValueError("published summary study_request_sha256 differs from external request")
                for invocation in sorted((self.output/"Invocations").glob("InvocationV*.json")):
                    recorded = checked_read(self.output, invocation.relative_to(self.output).as_posix())
                    if recorded.get("study_request_sha256") != self.plan_sha256:
                        raise ValueError("recorded invocation study_request_sha256 differs from external request")
            if (published.get("schema") != summary["schema"]
                    or published.get("representation") != summary.get("representation")
                    or published.get("pair_scope") != summary.get("pair_scope")
                    or type(published.get("study_exit_passed")) is not bool
                    or set(published.get("requirements", {})) != set(requirements)):
                raise ValueError("published study summary has an invalid scientific contract")
            if published["active_case_count"] != len(active) or published["historical_attempt_count"] != len(historical):
                raise ValueError("published study counts differ from recorded attempts")
            if hasattr(self, "plan"):
                for name in ("diagnostic_failure_count", "diagnostic_failed_case_count", "planned_case_count"):
                    if type(published.get(name)) is not int or published[name] != summary[name]:
                        raise ValueError("published study count differs from recorded extent: " + name)
                for name in ("missing_cases", "unresolved_cases"):
                    if published.get(name) != summary[name]:
                        raise ValueError("published study extent differs from recorded cases: " + name)
            for key, value in published["requirements"].items():
                if type(value) is not bool:
                    raise ValueError("published study requirement must be boolean: "+key)
                # A legacy unplanned invocation may enlarge its scope. A
                # fixed external plan cannot change the expected case set.
                if (key == "declared_cases_available" and hasattr(self, "plan")
                        and value != requirements[key]):
                    raise ValueError("published study requirement differs from the fixed request: " + key)
                if key != "declared_cases_available" and value and not requirements[key]:
                    raise ValueError("published study requirement is not supported: "+key)
            if published["study_exit_passed"] and not summary["study_exit_passed"]:
                raise ValueError("published full-study acceptance is not supported by physical evidence")
            for row in published.get("cases", []):
                actual = active.get(row.get("case"))
                if (actual is not None and row.get("scientific_checks_passed") is True
                        and actual["completion"].get("scientific_checks_passed") is not True):
                    raise ValueError("published passing case contradicts its physical result")
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
            return code
        directory = self.output/"Invocations"
        directory.mkdir(exist_ok=True)
        index = len(list(directory.glob("InvocationV*.json")))+1
        write_json(directory/f"InvocationV{index}.json", summary)
        write_json(self.output/"StudySummaryV1.json", summary)
        write_json(inventory_path, inventory)
        write_json(self.output/"FailureIndexV1.json", {"schema": "R1PhysicsStudyFailureIndexV1", "cases": failures})
        write_json(self.output/"UnavailableComparisonIndexV1.json", {"schema": "R1UnavailableComparisonIndexV1", "cases": unavailable})
        write_json(self.output/"HistoricalAttemptsV2.json", {"cases": historical})
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import record_source_reads
        record_source_reads(self.output, self.context)
        seal(self.output)
        print("STUDY_MANIFEST_SHA256", sha(self.output/"ManifestV1.json"), flush=True)
        return code


def scientific_checks(result):
    """Return scoped pass/fail, or None when a required uncertainty is absent."""
    if not isinstance(result, dict):
        return None
    if result.get("comparison_available") is False:
        return None
    if result.get("schema") in ("R1FrequencyWindowReportV1", "R1FrequencyWindowReportV2"):
        return bool(result["numeric_checks_passed"])
    if result.get("schema") == "R1DCEndpointAmplitudeStudyV1":
        return (result.get("dc_states_certified") is True and result.get("linearity_certified") is False
                and result.get("full_transient_linearity_certified") is False)
    if result.get("schema") == "R1ControlledDCResponseV1":
        return result.get("certified") is True
    if "double_domain_consistent" in result and "reconstruction" in result:
        return result.get("device_stage_gate", {}).get("device_qualified") is True
    for key in ("certificate", "preparation_checks"):
        if key in result:
            return bool(result[key]["certified"])
    if "within_compared_budgets" in result:
        return bool(result["within_compared_budgets"])
    if "numerically_eligible_frequency_points" in result:
        try:
            return bool(assess_small_signal_response(result)["certified"])
        except (ValueError, KeyError, TypeError):
            return False
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
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import compare_step_current_charge as compare
    return compare(left, right, left_baseline, right_baseline)


class QualificationStudy(Study):
    """Derived-only view of a fully checked, immutable numerical collection."""

    def __init__(self, collection, args, request, receipt):
        self.__dict__.update(collection.__dict__)
        self.collection = collection
        self.args = copy.copy(collection.args)
        self.args.linearity_case = request["linearity_case"]
        self.args.qualification_file = args.qualification_file
        self.args.qualification_sha256 = args.qualification_sha256
        self.qualification_analysis = True
        self.analysis_request, self.verification_receipt = request, receipt
        self.required_frequency_Hz = request["required_frequency_Hz"]
        self.qualification_inputs = load_qualification_inputs(args.qualification_file,
            args.qualification_sha256, result_directory=self.output)
        self.derived_results, self.derived_rows, self.derived_dependencies = {}, {}, {}
        self.verified_cases = dict(collection.verified_cases)
        self.active_case = None

    def selected(self, key):
        return key in self.analysis_request["cases"]

    def solve_dc_state(self, *args, **kwargs):
        raise RuntimeError("post-execution qualification cannot create a new physical DC solution")

    def step(self, *args, **kwargs):
        raise RuntimeError("post-execution qualification cannot integrate a new trajectory")

    def prepared(self, intervals):
        if intervals not in self.collection.preparations:
            raise ValueError("required preparation was not checked in the collection")
        self.record_dependency(f"Preparation/N{intervals}")
        return self.collection.preparations[intervals]

    def record_dependency(self, key):
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import evidence_digest
        owner = self.active_case
        if owner is None or owner == key:
            return
        if key in self.derived_results:
            value = {"kind": "derived", "result_sha256": evidence_digest(self.derived_results[key])}
        else:
            directory = self.collection.latest(key)
            if directory is None:
                raise ValueError("analysis dependency is absent: " + key)
            value = {"kind": "collection", "attempt": directory.relative_to(self.output).as_posix(),
                     "manifest_sha256": sha(directory/"ManifestV1.json")}
        self.derived_dependencies.setdefault(owner, {})[key] = value

    def saved(self, key, *, require_scientific=True):
        if key.startswith(("Linearity/", "Frequency/", "DoubleDomain/")):
            if key not in self.derived_results:
                raise ValueError("required post-execution qualification has not run: " + key)
            result = self.derived_results[key]
            if require_scientific and scientific_checks(result) is not True:
                raise ValueError("derived dependency lacks scientific eligibility: " + key)
        else:
            result = self.collection.saved(key, require_scientific=require_scientific)
            if key in self.collection.verified_cases:
                self.verified_cases[key] = self.collection.verified_cases[key]
        self.record_dependency(key)
        return result

    def qualified_amplitude(self):
        # An unqualified selected amplitude remains diagnostic. The actual
        # linearity record is checked in reconstruction_record, not invented.
        return self.analysis_request["window_amplitude_V"], self.analysis_request["linearity_case"]

    def case(self, key, request, operation, *, dependency=False):
        if not key.startswith(("Linearity/", "Frequency/", "DoubleDomain/")):
            if key.startswith(("DCBaseline/", "TargetDC/", "DC/")):
                return self.saved(key)
            return None
        if not self.selected(key):
            return None
        expected = self.analysis_request["cases"][key]
        if any(ready(request.get(name)) != value for name, value in expected.items()):
            raise ValueError("derived operation differs from the external analysis request")
        previous = self.active_case
        self.active_case = key
        try:
            self.source_unchanged()
            result = ready(operation(None))
            self.source_unchanged()
            self.derived_results[key] = result
            if result.get("comparison_available") is False:
                status = "unavailable"
            else:
                status = "completed"
            self.derived_rows[key] = {"case": key, "status": status,
                "scientific_checks_passed": scientific_checks(result), "request": ready(request)}
            return result
        except Exception as exc:
            self.derived_rows[key] = {"case": key, "status": "failed", "request": ready(request),
                "scientific_checks_passed": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            return None
        finally:
            self.active_case = previous

    def result_record(self):
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import physical_preparation_identity
        qualifications = {}
        for key in self.analysis_request["cases"]:
            result = self.derived_results.get(key, {})
            if key.startswith("Frequency/"):
                qualifies = result.get("device_frequency_window_certified") is True
            elif key.startswith("Linearity/"):
                qualifies = (result.get("linearity_certified") is True
                             and result.get("qualification", {}).get("device_qualified") is True)
            else:
                qualifies = result.get("device_stage_gate", {}).get("device_qualified") is True
            qualifications[key] = qualifies
        requirements = dict(self.verification_receipt["requirements"])
        requirements["caller_approved_standard_bound"] = True
        requirements["amplitude_linearity"] = any(v for k, v in qualifications.items() if k.startswith("Linearity/"))
        requirements["window_and_double_domain"] = any(v for k, v in qualifications.items() if k.startswith("DoubleDomain/"))
        missing = sorted(set(self.analysis_request["cases"]) - set(self.derived_rows))
        failed = [key for key, row in self.derived_rows.items() if row["status"] == "failed"]
        complete = not missing and not failed
        return {"schema": "R1PostExecutionQualificationV1", "request": self.analysis_request,
            "verification_receipt": self.verification_receipt,
            "physical_preparation_identities": {str(n): physical_preparation_identity(p)
                                                for n, p in self.collection.preparations.items()},
            "cases": self.derived_rows, "results": self.derived_results,
            "input_bindings": self.derived_dependencies, "missing_cases": missing,
            "failed_cases": failed, "analysis_completed": complete,
            "device_qualifications": qualifications,
            "requested_device_qualifications_passed": complete and all(qualifications.values()),
            "requirements": requirements,
            "study_exit_passed": complete and all(requirements.values()),
            "new_physical_solves_in_derivation": 0,
            "independent_acceptance": "not_asserted",
            "scope": "derived_qualification_of_fixed_collection; verification_is_not_scientific_approval"}


def run_post_execution_analysis(args):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification_workflow import (
        analysis_cases, analysis_request, load_analysis_request, verify_collection_receipt, require_digest,
        read_only_derivation,
    )
    collection_args = copy.copy(args)
    collection_args.verify = True
    collection_args.plan_only = False
    collection_args.qualification_file = None
    collection_args.qualification_sha256 = None
    collection = Study(collection_args)
    destination = args.analysis_dir.resolve()
    if destination.is_relative_to(collection.output) or collection.output.is_relative_to(destination):
        raise ValueError("analysis and numerical collection must have separate directories")
    if args.qualification_file.resolve().is_relative_to(destination):
        raise ValueError("reviewed qualification inputs must be retained outside the analysis output")
    sections = tuple(name for name in ("amplitude", "reconstruct")
                     if name in (args.analysis_section or ("amplitude", "reconstruct")))
    cases = analysis_cases(collection.grids, collection.amplitudes, sections, args.analysis_case or ())
    request = analysis_request(collection_manifest_sha256=args.manifest_sha256,
        calculation_request_sha256=collection.plan_sha256,
        qualification_sha256=args.qualification_sha256,
        candidate_standard_sha256=collection.standard["candidate_standard_sha256"],
        approved_standard_sha256=args.analysis_approved_standard_sha256,
        source=collection.source, window_spec=collection.window_spec, cases=cases,
        amplitude_ladder_V=collection.amplitudes, window_amplitude_V=collection.args.window_amplitude,
        linearity_case=args.analysis_linearity_case,
        required_frequency_Hz=args.analysis_frequencies or collection.frequencies.tolist())
    if not np.all(np.isin(request["required_frequency_Hz"], collection.frequencies)):
        raise ValueError("analysis frequency intersection contains uncollected frequencies")
    # Validate the approval file now, before recording a request referencing it.
    approval = load_qualification_inputs(args.qualification_file, args.qualification_sha256,
                                        result_directory=collection.output)
    plan_path = args.analysis_plan_file.resolve()
    if plan_path.is_relative_to(collection.output) or plan_path.is_relative_to(destination):
        raise ValueError("analysis request must be retained outside both result directories")
    if args.analysis_plan_only:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if plan_path.exists() and plan_path.read_bytes() != raw_json(request):
            raise ValueError("analysis request path already contains a different request")
        if not plan_path.exists():
            write_json(plan_path, request)
        print("ANALYSIS_REQUEST_SHA256", sha(plan_path), flush=True)
        return 0
    load_analysis_request(plan_path, args.analysis_request_sha256, request)
    if args.verify_analysis:
        require_digest(args.analysis_manifest_sha256, "analysis manifest")
        if sha(destination/"ManifestV1.json") != args.analysis_manifest_sha256:
            raise ValueError("analysis manifest differs from external anchor")
        if checked_read(destination, "AnalysisRequestV1.json") != request:
            raise ValueError("saved analysis request differs from the caller request")
    elif destination.exists():
        raise ValueError("analysis output already exists; verify it or select a new version")
    collection.run(SECTIONS)
    collection.finish()
    receipt = verify_collection_receipt(collection.last_summary)
    receipt["collection_manifest_sha256"] = args.manifest_sha256
    receipt["verifier_source"] = collection.source
    analysis = QualificationStudy(collection, args, request, receipt)
    with read_only_derivation() as production_attempts:
        analysis.run(sections)
    result = ready(analysis.result_record())
    result["forbidden_physical_production_attempts"] = production_attempts
    # Recheck both immutable input anchors after all derived operations.
    if sha(collection.output/"ManifestV1.json") != args.manifest_sha256:
        raise ValueError("numerical collection changed during qualification")
    checked_read(collection.output, "StudyRequestV1.json")
    load_analysis_request(plan_path, args.analysis_request_sha256, request)
    if load_qualification_inputs(args.qualification_file, args.qualification_sha256,
                                result_directory=collection.output) != approval:
        raise ValueError("qualification inputs changed during analysis")
    if args.verify_analysis:
        if checked_read(destination, "AnalysisResultV1.json") != result:
            raise ValueError("saved qualification differs from recomputed analysis")
        if checked_read(destination, "VerificationReceiptV1.json") != receipt:
            raise ValueError("saved verification receipt differs from the collection verification")
        if checked_read(destination, "QualificationInputsV1.json") != approval:
            raise ValueError("saved approval differs from the externally selected inputs")
        verify_study_source(destination, collection.context, require_read_receipt=False)
    else:
        destination.mkdir(parents=True)
        write_json(destination/"AnalysisRequestV1.json", request)
        write_json(destination/"QualificationInputsV1.json", approval)
        write_json(destination/"AnalysisResultV1.json", result)
        write_json(destination/"VerificationReceiptV1.json", receipt)
        record_frozen_source(destination, collection.context)
        seal(destination)
        print("ANALYSIS_MANIFEST_SHA256", sha(destination/"ManifestV1.json"), flush=True)
    print(json.dumps({"analysis_completed": result["analysis_completed"],
                      "requested_device_qualifications_passed": result["requested_device_qualifications_passed"],
                      "study_exit_passed": result["study_exit_passed"]}), flush=True)
    return 1 if result["failed_cases"] else 0 if result["requested_device_qualifications_passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=PROJECT/"tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json")
    parser.add_argument("--fixture", type=Path, default=PROJECT/"tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    parser.add_argument("--section", action="append", required=True, choices=(*SECTIONS, "dc-ac", "all"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="preserve prior failure and create a new attempt")
    parser.add_argument("--case-filter", default="", help="substring of case key; required preparations run as dependencies")
    parser.add_argument("--case", action="append", help="exact planned case key; repeat for a bounded set")
    parser.add_argument("--max-cases", type=int, help="maximum newly started cases in this invocation")
    parser.add_argument("--formal", action="store_true", help="require the controlled source launcher")
    parser.add_argument("--backend", choices=("legacy", "pair"), default="legacy",
                        help="explicit per-run representation; pair DC/AC consumers are not yet qualified")
    parser.add_argument("--verify", action="store_true", help="recompute archived evidence without writing it")
    parser.add_argument("--manifest-sha256", help="external root manifest anchor for formal resume/verify")
    parser.add_argument("--plan-only", action="store_true", help="write exact case requests without solving or creating results")
    parser.add_argument("--plan-file", type=Path, help="caller-held pre-execution request file")
    parser.add_argument("--request-sha256", help="independently retained pre-execution request SHA256")
    parser.add_argument("--qualification-file", type=Path, help="caller-held independently reviewed response evidence")
    parser.add_argument("--qualification-sha256", help="independently retained qualification input SHA256")
    parser.add_argument("--qualify", action="store_true", help="derive qualification from an immutable checked collection")
    parser.add_argument("--analysis-dir", type=Path, help="separate output for post-execution qualification")
    parser.add_argument("--analysis-plan-file", type=Path, help="caller-held post-execution analysis request")
    parser.add_argument("--analysis-plan-only", action="store_true")
    parser.add_argument("--analysis-request-sha256")
    parser.add_argument("--analysis-approved-standard-sha256")
    parser.add_argument("--analysis-section", action="append", choices=("amplitude", "reconstruct"))
    parser.add_argument("--analysis-case", action="append")
    parser.add_argument("--analysis-linearity-case")
    parser.add_argument("--analysis-frequencies", type=float, nargs="+", help="explicit required intersection of collected frequencies")
    parser.add_argument("--verify-analysis", action="store_true")
    parser.add_argument("--analysis-manifest-sha256")
    parser.add_argument("--approved-standard-sha256", help="optional caller-held approved standard digest; candidate pins alone are not approval")
    parser.add_argument("--grids", type=int, nargs="+", default=(16, 32, 64), choices=(16, 32, 64, 128, 256))
    parser.add_argument("--matrix-controls", nargs="+", default=("D",), choices=tuple("ABCD"))
    parser.add_argument("--window", choices=("functional", "full", "diagnostic"), default="functional")
    parser.add_argument("--first-time-s", type=float, default=1e-9)
    parser.add_argument("--last-time-s", type=float, default=1e2)
    parser.add_argument("--extended-frequency", action="store_true", help="include protocol limit band 1e-6..1e10 Hz")
    parser.add_argument("--amplitudes", type=float, nargs="+", choices=AMPLITUDES_V,
                        help="preselected subset of the prescribed transient amplitude ladder")
    parser.add_argument("--window-amplitude", type=float, choices=AMPLITUDES_V, default=.005,
                        help="fixed diagnostic window amplitude; does not assert linearity")
    parser.add_argument("--linearity-case", help="explicit verified linearity case for the chosen window amplitude")
    args = parser.parse_args(argv)
    if args.qualify:
        if (not args.analysis_dir or not args.analysis_plan_file or not args.qualification_file
                or not args.qualification_sha256 or not args.manifest_sha256
                or args.plan_only or args.retry_failed or args.resume or args.verify):
            parser.error("--qualify requires collection/approval anchors and separate analysis paths; no collection mutation options")
        try:
            return run_post_execution_analysis(args)
        except Exception as exc:
            print(f"QUALIFICATION STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.analysis_plan_only or args.verify_analysis:
        parser.error("analysis options require --qualify")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if list(args.grids) != sorted(set(args.grids)):
        parser.error("--grids must be distinct and increasing")
    if args.verify and args.retry_failed:
        parser.error("verification cannot retry or modify recorded computations")
    if args.plan_only and (args.plan_file is None or args.verify or args.resume):
        parser.error("--plan-only requires --plan-file and cannot resume or verify")
    if args.amplitudes and (len(set(args.amplitudes)) != len(args.amplitudes)
                           or args.amplitudes != sorted(args.amplitudes, reverse=True)):
        parser.error("--amplitudes must be distinct and decreasing")
    study = None
    try:
        study = Study(args)
        if args.plan_only:
            print("STUDY_REQUEST_SHA256", study.plan_sha256, flush=True)
            print(json.dumps({"planned_case_count": len(study.plan["cases"]), "physical_cases_executed": 0}))
            return 0
        sections = expanded_sections(args.section)
        if args.verify:
            # --section must never hide a different archived result from an
            # otherwise successful scientific verification.
            sections = SECTIONS
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

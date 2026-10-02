"""Collect explicit D zero-voltage baselines, or analyze sealed V5 short cases.

Both commands require an external request plus its SHA256. Common fields:
schema, source_commit, source_content_sha256, batches. Each batch contains
directory, manifest_file (outside directory), manifest_sha256. collect-dc uses
schema R1V5DCBaselineRequestV1 and selections [{intervals, case}] for N64/128/256.
analyze uses schema R1V5ResponseAnalysisRequestV1 and dc_bundle with the same
three manifest fields. Optional chains maps the three axes to three explicit
case names each, sharing one finest endpoint; no new cases are executed.
Optional methodology contains file and sha256 for an externally held JSON
method assessment, retained without automatic approval. Analysis never
collects a preparation, DC or trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
TIMES = [0., 1e-9, 1e-8, 1e-6, 1e-4]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_text(value, size=64):
    if not isinstance(value, str) or len(value) != size or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a full external digest is required")
    return value


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def write(path, value):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(directory):
    write(directory/"ManifestV1.json", {p.relative_to(directory).as_posix(): {
        "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*")) if p.is_file() and p != directory/"ManifestV1.json"})


class SealedBundle:
    """Read only files covered by an independently held exact manifest."""
    def __init__(self, descriptor):
        if not isinstance(descriptor, dict) or set(descriptor) != {"directory", "manifest_file", "manifest_sha256"}:
            raise ValueError("bundle requires directory and external manifest path/digest")
        self.directory = Path(descriptor["directory"]).resolve()
        external = Path(descriptor["manifest_file"]).resolve()
        if external.is_relative_to(self.directory):
            raise ValueError("manifest must be held outside its result directory")
        if sha(external) != digest_text(descriptor["manifest_sha256"]):
            raise ValueError("external manifest digest mismatch")
        self.raw_manifest = external.read_bytes()
        self.manifest = parse(self.raw_manifest)
        self.descriptor = dict(descriptor)
        if not isinstance(self.manifest, dict) or not self.manifest:
            raise ValueError("empty or invalid manifest")
        self.verify()

    def verify(self):
        if (self.directory/"ManifestV1.json").read_bytes() != self.raw_manifest:
            raise ValueError("internal manifest differs from the external anchor")
        paths = list(self.directory.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("sealed bundles cannot contain symlinks")
        actual = {p.relative_to(self.directory).as_posix() for p in paths
                  if p.is_file() and p != self.directory/"ManifestV1.json"}
        if actual != set(self.manifest):
            raise ValueError("manifest coverage differs from actual files")
        for name in self.manifest:
            self.checked_path(name)

    def checked_path(self, name):
        if not isinstance(name, str):
            raise ValueError("invalid manifest path type")
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts
                or str(path) != name or name not in self.manifest):
            raise ValueError("invalid or unbound manifest path")
        expected = self.manifest[name]
        full = self.directory/name
        if (not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"}
                or type(expected["bytes"]) is not int or expected["bytes"] < 0
                or full.stat().st_size != expected["bytes"] or sha(full) != digest_text(expected["sha256"])):
            raise ValueError("sealed file mismatch: " + name)
        return full

    def read(self, name):
        return parse(self.checked_path(name).read_bytes())


def load_request(path, expected_sha256, output, command):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.is_relative_to(output) or sha(path) != digest_text(expected_sha256):
        raise ValueError("request must match its external digest and be outside output")
    raw = path.read_bytes()
    request = parse(raw)
    common = {"schema", "source_commit", "source_content_sha256", "batches"}
    extra, schema = (("selections", "R1V5DCBaselineRequestV1") if command == "collect-dc"
                     else ("dc_bundle", "R1V5ResponseAnalysisRequestV1"))
    optional = {"chains", "methodology"} if command == "analyze" else set()
    if (not isinstance(request, dict) or not common | {extra} <= set(request)
            or set(request)-common-{extra}-optional or request["schema"] != schema):
        raise ValueError("request schema or fields differ from the selected command")
    digest_text(request["source_commit"], 40)
    digest_text(request["source_content_sha256"])
    if not isinstance(request["batches"], list) or not 1 <= len(request["batches"]) <= 8:
        raise ValueError("require one to eight explicit case batches")
    if command == "collect-dc":
        selected = request["selections"]
        if (not isinstance(selected, list) or len(selected) != 3
                or any(not isinstance(s, dict) or set(s) != {"intervals", "case"} for s in selected)
                or any(type(s["intervals"]) is not int or not isinstance(s["case"], str) for s in selected)
                or sorted(s["intervals"] for s in selected) != [64, 128, 256]):
            raise ValueError("DC collection requires exactly one selected case for N64, N128 and N256")
    else:
        validate_chains(request.get("chains", response_chains()))
        if "methodology" in request:
            method = request["methodology"]
            if not isinstance(method, dict) or set(method) != {"file", "sha256"}:
                raise ValueError("methodology requires an external file and digest")
            if Path(method["file"]).resolve().is_relative_to(output) or sha(method["file"]) != digest_text(method["sha256"]):
                raise ValueError("methodology differs from its external anchor")
            if not isinstance(parse(Path(method["file"]).read_bytes()), dict):
                raise ValueError("methodology must be a JSON object")
    return request, raw


def source_context(request):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import execution_source
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True)
    if status:
        raise ValueError("the source must be a clean committed checkout")
    context = require_r1_checkout(project=PROJECT, formal=True, source_commit=request["source_commit"],
                                  expected_source_sha256=request["source_content_sha256"])
    return context, execution_source()


def case_name(case):
    return f"{case['control']}_N{case['intervals']}_F{str(float(case['nonlinear_factor'])).replace('.', 'p')}_T{case['time_substeps'][0]}"


def load_cases(request, actual_source):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import R1PreparedState
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    bundles, cases = [], {}
    for descriptor in request["batches"]:
        bundle = SealedBundle(descriptor)
        receipt, summary, requested = [bundle.read(name) for name in ("SourceReceiptV1.json", "Summary.json", "RequestV1.json")]
        if (receipt.get("source_commit") != request["source_commit"]
                or receipt.get("source_content_sha256") != request["source_content_sha256"]
                or receipt.get("actual_source") != actual_source
                or summary.get("schema") != "R1V5CaseRunV1" or summary.get("source_files_unchanged") is not True
                or summary.get("source_commit") != request["source_commit"]
                or summary.get("request_sha256") != bundle.manifest["RequestV1.json"]["sha256"]
                or requested.get("schema") != "R1V5CaseRequestV1"):
            raise ValueError("case batch source, request or completion identity is invalid")
        items = summary.get("cases")
        if not isinstance(items, list) or [item.get("request") for item in items] != requested.get("cases"):
            raise ValueError("case accounting differs from the sealed request")
        for item in items:
            spec, name = item["request"], item["case"]
            if name != case_name(spec) or name in cases:
                raise ValueError("ambiguous, duplicate or mislabeled case: " + name)
            entry = {"name": name, "request": spec, "status": item["status"], "available": False,
                     "reason": item.get("reason"), "bundle": bundle, "record": item}
            cases[name] = entry
            if item.get("short_case_checks_passed") is not True:
                entry["reason"] = item.get("reason") or "case_not_completed_with_verified_physics"
                continue
            if (item.get("status") != "completed" or item.get("execution_status") != "completed"
                    or item.get("source_identity_unchanged") is not True or item.get("observer_matches_result") is not True
                    or bundle.read(name+"/RecordV1.json") != item or bundle.read(name+"/RequestV1.json") != spec):
                raise ValueError("successful case accounting is inconsistent: " + name)
            prepared, result, wrapper = [bundle.read(name+"/"+file) for file in ("Prepared.json", "Result.json", "PhysicsReplay.json")]
            R1PreparedState.from_dict(prepared)
            replay = wrapper.get("reconstruction", {})
            if (wrapper.get("schema") != "R1V5CasePhysicsReplayV1" or wrapper.get("run_completed") is not True
                    or wrapper.get("scope") != "complete_short_case"
                    or any(replay.get(key) is not True for key in (
                        "certified", "content_matches_recomputed", "physical_limits_satisfied", "independent_physics_passed"))
                    or result.get("schema") != "R1ControlledStepV1" or result.get("certificate", {}).get("certified") is not True
                    or result.get("source") != actual_source or prepared.get("source") != actual_source
                    or result.get("prepared_sha256") != prepared["sha256"]
                    or result.get("intervals") != spec["intervals"] or prepared.get("intervals") != spec["intervals"]
                    or result.get("control_label") != spec["control"] or result.get("amplitude_V") != spec["amplitude_V"]
                    or result.get("times_s") != TIMES or spec["times_s"] != TIMES or spec["amplitude_V"] != .005
                    or result.get("policy") != json.loads(json.dumps(asdict(r1_policy(spec["nonlinear_factor"], time_substeps=spec["time_substeps"]))))) :
                raise ValueError("saved case physics, source or numerical axes differ: " + name)
            observer = [parse(line) for line in bundle.checked_path(name+"/AcceptedStepsV1.jsonl").read_bytes().splitlines()]
            if observer != result["accepted_steps"] or replay.get("checked_row_count") != len(observer):
                raise ValueError("saved case observer/replay coverage mismatch: " + name)
            entry.update(available=True, prepared=prepared, result=result, replay=replay)
        bundles.append(bundle)
    return bundles, cases


def collect_dc(request, cases, output, actual_source, context):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import solve_controlled_dc, restore_controlled_dc
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        json_data, restore_common_state, physical_preparation_identity)
    from perovskite_sim.models.config_loader import load_device_from_yaml
    fixture = context.read_bytes("tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding = parse(context.read_bytes("tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"))
    (output/"SourceFixtureV1.yaml").write_bytes(fixture)
    write(output/"ReferenceBindingV1.json", binding)
    stack = load_device_from_yaml(output/"SourceFixtureV1.yaml")
    summary = {"schema": "R1V5DCBaselineCollectionV1", "source_commit": request["source_commit"],
               "source_content_sha256": request["source_content_sha256"], "actual_source": actual_source,
               "entries": [], "all_completed": False, "scientifically_qualified": False}
    for selection in request["selections"]:
        n, name = selection["intervals"], selection["case"]
        item = {"intervals": n, "case": name, "status": "not_run", "voltage_V": 0., "control": "D"}
        summary["entries"].append(item)
        entry = cases.get(name)
        if entry is None or not entry["available"] or entry["request"]["control"] != "D" or entry["request"]["intervals"] != n:
            item.update(status="unavailable", reason="selected_verified_D_preparation_missing")
            continue
        directory = output/f"D_N{n}"
        directory.mkdir()
        prepared = entry["prepared"]
        write(directory/"Prepared.json", prepared)
        item["prepared_sha256"] = prepared["sha256"]
        item["physical_preparation_identity"] = physical_preparation_identity(prepared)
        try:
            restore_common_state(prepared, stack, n, binding, policy=r1_policy(),
                                 expected_prepared_sha256=prepared["sha256"])
            dc = solve_controlled_dc(stack, n, binding, prepared, control="D", voltage_V=0., policy=r1_policy(),
                                     expected_prepared_sha256=prepared["sha256"])
            write(directory/"DC.json", json_data(dc.evidence))
            restored = restore_controlled_dc(parse((directory/"DC.json").read_bytes()), stack, n, binding, prepared)
            write(directory/"DCReplayV1.json", {"schema": "R1V5DCBaselineReplayV1", "content_matches_recomputed": True,
                "certified": restored.evidence["certified"], "dc_file_sha256": sha(directory/"DC.json"),
                "prepared_sha256": prepared["sha256"], "source": actual_source,
                "physical_preparation_identity": item["physical_preparation_identity"],
                "scope": "saved_coordinate_recomputed_without_a_second_DC_root_solve"})
            item.update(status="completed", dc_file_sha256=sha(directory/"DC.json"))
        except Exception as exc:
            item.update(status="failed", reason=str(exc), exception_type=type(exc).__name__)
            write(directory/"FailureV1.json", {**item, "result": json_data(getattr(exc, "result", None))})
        write(output/"Summary.json", summary)
    summary["all_completed"] = all(item["status"] == "completed" for item in summary["entries"])
    write(output/"Summary.json", summary)
    return 0 if summary["all_completed"] else 2


def load_baselines(request, actual_source):
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import R1PreparedState, physical_preparation_identity
    bundle = SealedBundle(request["dc_bundle"])
    summary = bundle.read("Summary.json")
    completion = bundle.read("CompletionV1.json")
    if (completion.get("completed") is not True or completion.get("command") != "collect-dc"
            or completion.get("input_bytes_and_source_unchanged") is not True):
        raise ValueError("DC collection is not completed and sealed under stable inputs")
    if (summary.get("schema") != "R1V5DCBaselineCollectionV1" or summary.get("source_commit") != request["source_commit"]
            or summary.get("source_content_sha256") != request["source_content_sha256"]
            or summary.get("actual_source") != actual_source):
        raise ValueError("DC baseline source identity differs")
    baselines = {}
    entries = summary.get("entries", [])
    if sorted(item["intervals"] for item in entries) != [64, 128, 256]:
        raise ValueError("DC baseline accounting must retain all three requested grids")
    for item in entries:
        if item["status"] != "completed":
            continue
        n = item["intervals"]
        dc, replay, prepared = [bundle.read(f"D_N{n}/"+name) for name in ("DC.json", "DCReplayV1.json", "Prepared.json")]
        R1PreparedState.from_dict(prepared)
        physical_identity = physical_preparation_identity(prepared)
        if (dc.get("schema") != "R1ControlledDCResponseV1" or dc.get("control") != "D" or dc.get("voltage_V") != 0.
                or dc.get("intervals") != n or dc.get("source") != actual_source or dc.get("certified") is not True
                or dc.get("stop_reason") != "converged" or dc.get("checks", {}).get("certified") is not True
                or dc.get("prepared_sha256") != prepared.get("sha256") or prepared.get("sha256") != item.get("prepared_sha256")
                or replay.get("schema") != "R1V5DCBaselineReplayV1" or replay.get("certified") is not True
                or replay.get("content_matches_recomputed") is not True or replay.get("source") != actual_source
                or replay.get("prepared_sha256") != prepared["sha256"]
                or replay.get("physical_preparation_identity") != physical_identity
                or item.get("physical_preparation_identity") != physical_identity
                or replay.get("dc_file_sha256") != bundle.manifest[f"D_N{n}/DC.json"]["sha256"]
                or item.get("dc_file_sha256") != replay["dc_file_sha256"]):
            raise ValueError("DC baseline lacks matching saved physical verification")
        currents = np.asarray(dc["current_A_m2"])
        if currents.dtype.kind not in "fiu" or currents.ndim != 1 or len(currents) < 2 or not np.all(np.isfinite(currents)):
            raise ValueError("DC baseline currents are invalid")
        baselines[n] = {"currents": currents[[0, -1]], "prepared_sha256": prepared["sha256"],
                        "physical_preparation_identity": physical_identity, "dc": dc}
    return bundle, baselines


def response_chains():
    def name(n, factor=.01, time=4):
        return case_name({"control": "D", "intervals": n, "nonlinear_factor": factor, "time_substeps": [time]})
    return {"intervals": [name(n) for n in (64, 128, 256)],
            "time_substeps": [name(256, time=t) for t in (1, 2, 4)],
            "nonlinear_factor": [name(256, factor=f) for f in (1., .1, .01)]}


def validate_chains(chains):
    if (not isinstance(chains, dict) or set(chains) != {"intervals", "time_substeps", "nonlinear_factor"}
            or any(not isinstance(names, list) or len(names) != 3 or len(set(names)) != 3
                   or any(not isinstance(name, str) or not name.startswith("D_") for name in names)
                   for names in chains.values())
            or len({names[-1] for names in chains.values()}) != 1):
        raise ValueError("three explicit D-axis chains must share the same finest endpoint")
    return chains


def require_separate_output(output, bundles):
    if any(output.is_relative_to(bundle.directory) for bundle in bundles):
        raise ValueError("derived output cannot modify a sealed input directory")


def pair_report(left, right, axis, baselines):
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
        compare_step_current_charge, step_current_charge_responses)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_spatial import (
        spatial_responses_from_step, compare_convergence_responses)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import physical_preparation_identity
    a, b = left["result"], right["result"]
    currents, baseline_applications = [], []
    for entry in (left, right):
        baseline = baselines[entry["request"]["intervals"]]
        physical_identity = physical_preparation_identity(entry["prepared"])
        if baseline["physical_preparation_identity"] != physical_identity:
            raise ValueError("DC baseline differs from the physical case preparation")
        currents.append(baseline["currents"])
        baseline_applications.append({"case_prepared_sha256": entry["prepared"]["sha256"],
            "baseline_prepared_sha256": baseline["prepared_sha256"], "physical_preparation_identity": physical_identity,
            "scope": "same_physical_preparation_with_original_artifact_identities_retained"})
    electrical = compare_step_current_charge(a, b, *currents, expected_times_s=TIMES)
    comparison = compare_convergence_responses(spatial_responses_from_step(left["prepared"], a),
        spatial_responses_from_step(right["prepared"], b), axis=axis, electrical=electrical,
        input_verifications=(left["replay"], right["replay"]))
    responses = [step_current_charge_responses(record, base, expected_times_s=TIMES)
                 for record, base in zip((a, b), currents)]
    absolute = {}
    for index, key in enumerate(("regular_current_A_m2", "integrated_charge_C_m2")):
        values = [item[index].values for item in responses]
        absolute[key] = {"left": values[0].tolist(), "right": values[1].tolist(),
                         "absolute_difference": np.abs(values[0]-values[1]).tolist()}
    return {"available": True, "comparison": comparison, "absolute_responses": absolute,
            "baseline_applications": baseline_applications,
            "times_s": TIMES, "current_components": ["left_contact", "right_contact"],
            "charge_definition": "left_contact_regular_integral_plus_ideal_step_impulse_minus_DC_times_t",
            "verification_scope": "externally_anchored_saved_physics_replays; no_new_physical_solve"}


def axis_estimate(coarse, fine):
    """Report empirical differences and contraction, without an error bound."""
    import numpy as np
    if not coarse["available"] or not fine["available"]:
        return {"classification": "unknown", "reason": "three_adjacent_completed_levels_required"}
    terms = {}
    for key in ("regular_current_A_m2", "integrated_charge_C_m2"):
        earlier = np.asarray(coarse["absolute_responses"][key]["absolute_difference"])
        latest = np.asarray(fine["absolute_responses"][key]["absolute_difference"])
        contracted = (earlier > 0) & (latest < earlier)
        ratio = np.divide(latest, earlier, out=np.zeros_like(latest), where=earlier > 0)
        terms[key] = {"coarse_adjacent_difference": earlier.tolist(), "fine_adjacent_difference": latest.tolist(),
            "observed_contraction": contracted.tolist(),
            "difference_ratio": np.where(earlier > 0, ratio, 0.).tolist(),
            "ratio_defined": (earlier > 0).tolist(), "absolute_error_bound": None}
    return {"classification": "estimate_only", "method": "finest_adjacent_absolute_difference_with_three_level_contraction_diagnostic",
            "terms": terms, "validated_estimate": False, "extrapolated_error_bound": None,
            "both_adjacent_response_thresholds_passed": all(item["comparison"]["convergence_passed"] for item in (coarse, fine)),
            "assumptions_not_established": ["asymptotic_error_expansion", "axis_independence",
                                            "unperturbed_shared_initial_reference_and_arithmetic_errors"]}


def analyze(request, cases, baselines):
    reports, estimates, missing = {}, {}, []
    chains = validate_chains(request.get("chains", response_chains()))
    for axis, names in chains.items():
        pairs = []
        for index, (left, right) in enumerate(zip(names, names[1:])):
            dependencies = []
            for name in (left, right):
                entry = cases.get(name)
                if entry is None or not entry["available"]:
                    dependencies.append({"case": name, "reason": "missing_case" if entry is None else entry["reason"]})
                elif entry["request"]["intervals"] not in baselines:
                    dependencies.append({"case": name, "reason": "zero_voltage_DC_baseline_unavailable"})
            result = ({"available": False, "missing_dependencies": dependencies} if dependencies
                      else pair_report(cases[left], cases[right], axis, baselines))
            result.update(axis=axis, left=left, right=right, required_finest_pair=index == 1)
            pairs.append(result)
        reports[axis] = pairs
        estimates[axis] = axis_estimate(*pairs)
        if not pairs[-1]["available"]:
            missing.append(axis+"_finest_pair_unavailable")
        elif not pairs[-1]["comparison"]["convergence_passed"]:
            missing.append(axis+"_finest_pair_response_threshold_failed")
    covered_axes = [axis for axis, items in reports.items() if all(item["available"] for item in items)]
    methodology = (parse(Path(request["methodology"]["file"]).read_bytes()) if "methodology" in request else None)
    missing.extend(("shared_initial_state_and_fixed_reference_assessment_required",
                    "shared_constitutive_arithmetic_assessment_required", "complete_budget_methodology_assessment_required"))
    return {"schema": "R1V5DResponseAnalysisV1", "source_commit": request["source_commit"],
        "source_content_sha256": request["source_content_sha256"], "chains": chains, "pairs": reports,
        "requested_case_statuses": [{"case": name, "status": item["status"], "available": item["available"], "reason": item["reason"]}
                                    for name, item in sorted(cases.items())],
        "empirical_axis_estimates": estimates, "complete_budget": {"classification": "unknown",
            "current_error_A_m2": None, "charge_error_C_m2": None,
            "status": "requires_methodology_assessment", "axis_differences_summed": False, "independently_validated": False,
            "coverage": {
                "previous_state_propagation_under_axis_changes": {"covered_by": covered_axes,
                    "scope": "computed_full_coupled_trajectory_differences_for_the_named_axis_perturbations"},
                "coupled_feedback_under_axis_changes": {"covered_by": covered_axes,
                    "scope": "computed_full_coupled_trajectory_differences_not_a_separate_additive_error_term"},
                "shared_initial_state_and_fixed_reference": {"status": "requires_methodology_assessment"},
                "shared_constitutive_arithmetic": {"status": "requires_methodology_assessment"},
                "unsampled_times_and_settings": {"status": "outside_the_computed_evidence_scope"},
                "material_parameter_uncertainty": {"status": "outside_this_frozen_parameter_numerical_question"}}},
        "external_methodology": methodology, "external_methodology_identity": request.get("methodology"),
        "external_methodology_approved_by_tool": False,
        "necessary_finest_pairs_passed": all(items[-1]["available"] and items[-1]["comparison"]["convergence_passed"]
                                              for items in reports.values()),
        "D2_satisfied": False, "D2_assessment_status": "not_assessed_by_this_tool",
        "D2_remaining_requirements": missing, "long_window_authorized": False,
        "scope": "D_control_five_sample_short_window_only_not_R1_2_or_long_window_qualification"}


def run(args):
    output = args.output.resolve()
    request, raw = load_request(args.request_file, args.request_sha256, output, args.command)
    if output.exists():
        raise FileExistsError("output already exists; select a new run directory")
    for key in THREADS:
        os.environ[key] = "1"
    sys.path.insert(0, str(PROJECT))
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification_workflow import read_only_derivation
    context, actual_source = source_context(request)
    bundles, cases = load_cases(request, actual_source)
    if args.command == "analyze":
        dc_bundle, baselines = load_baselines(request, actual_source)
        bundles.append(dc_bundle)
    require_separate_output(output, bundles)
    output.mkdir(parents=True)
    (output/"RequestV1.json").write_bytes(raw)
    write(output/"SourceReceiptV1.json", {"source_commit": request["source_commit"],
        "source_content_sha256": request["source_content_sha256"], "actual_source": actual_source,
        "script_sha256": sha(__file__)})
    with threadpool_limits(limits=1, user_api="blas"):
        import numpy as np
        import scipy
        runtime = {"executable": sys.executable, "python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site),
            "threads": {key: os.environ[key] for key in THREADS}, "blas": threadpool_info()}
        if any(item["num_threads"] != 1 for item in runtime["blas"]):
            raise ValueError("BLAS single-thread limit did not take effect")
        write(output/"EnvironmentV1.json", runtime)
        if args.command == "collect-dc":
            code = collect_dc(request, cases, output, actual_source, context)
        else:
            with read_only_derivation() as attempted:
                report = analyze(request, cases, baselines)
            report["new_physical_solution_attempts"] = attempted
            write(output/"ResponseAnalysisV1.json", json_data(report))
            code = 0
    for bundle in bundles:
        bundle.verify()
    source_context(request)
    if sha(args.request_file) != args.request_sha256:
        raise ValueError("external request changed during execution")
    if "methodology" in request and sha(request["methodology"]["file"]) != request["methodology"]["sha256"]:
        raise ValueError("external methodology changed during analysis")
    write(output/"CompletionV1.json", {"completed": True, "command": args.command, "exit_code": code,
        "input_bytes_and_source_unchanged": True, "D2_qualification_asserted": False})
    seal(output)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect-dc", "analyze"))
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        return run(parser.parse_args(argv))
    except Exception as exc:
        print(f"V5 RESPONSE STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

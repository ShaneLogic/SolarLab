"""Exact historical failure observations shared by controlled and study runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CATEGORIES = ("known_nonconvergence", "known_physical_gate_failures")
CONDITION_FIELDS = ("control", "intervals", "nonlinear_factor", "amplitude_V", "times_s", "refinement_substeps")


def exact_conditions(request):
    """A missing axis is unknown, never a wildcard or a default measurement."""
    steps = request.get("time_substeps", request.get("refinement_substeps"))
    if "time_substeps" in request and "refinement_substeps" in request and list(request["time_substeps"]) != list(request["refinement_substeps"]):
        raise ValueError("conflicting failure registry time triples")
    result = {key: request[key] for key in CONDITION_FIELDS if key != "refinement_substeps"}
    if (result["control"] not in "ABCD" or len(result["control"]) != 1
            or type(result["intervals"]) is not int or result["intervals"] <= 0):
        raise ValueError("invalid failure registry control or interval")
    for key in ("nonlinear_factor", "amplitude_V"):
        value = result[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("invalid failure registry numerical condition: " + key)
        result[key] = float(value)
    times = list(result["times_s"])
    if (len(times) < 2 or times[0] != 0.0 or any(type(v) not in (float, int) or not math.isfinite(v) for v in times)
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("invalid failure registry observation times")
    if (not isinstance(steps, (tuple, list)) or len(steps) != 3
            or any(type(v) is not int or v <= 0 for v in steps)
            or any(right <= left for left, right in zip(steps, steps[1:]))):
        raise ValueError("invalid failure registry time triple")
    result.update(times_s=[float(value) for value in times], refinement_substeps=list(steps))
    return result


def merge_failure_registries(study, additional):
    if additional.get("schema") not in ("R1AdditionalFailuresV1", "R1AdditionalFailuresV2"):
        raise ValueError("unsupported supplemental R1 failure registry")
    semantics = additional.get("known_failure_semantics", {})
    if any(semantics.get(key) is not False for key in ("waives_checks", "skip_computation", "changes_acceptance_thresholds")):
        raise ValueError("supplemental historical failures cannot waive execution or checks")
    for category in CATEGORIES:
        if not isinstance(additional.get(category, []), list):
            raise ValueError("supplemental R1 failure observations must be lists")
    if not any(additional.get(category) for category in CATEGORIES):
        raise ValueError("supplemental R1 failure registry requires observations")
    combined = {category: [*study.get(category, []), *additional.get(category, [])] for category in CATEGORIES}
    entries = [entry for category in CATEGORIES for entry in combined[category]]
    ids = [entry["case_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case identity in supplemental R1 failure registry")
    keys = [(entry["observed_source_commit"], json.dumps(exact_conditions(entry), sort_keys=True)) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate exact conditions in supplemental R1 failure registry")
    return {**study, **combined}


def load_failure_registry():
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_binding as binding
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import current_execution_context
    context = current_execution_context()
    project = Path(__file__).resolve().parents[2]
    def read(path, digest):
        raw = context.read_bytes(path) if context is not None else (project / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("failure registry differs from its pinned digest: " + path)
        return json.loads(raw)
    study = read(binding.STUDY_INPUT_RELATIVE_PATH, binding.PINNED_STUDY_INPUT_SHA256)
    for path, digest in ((binding.ADDITIONAL_FAILURES_RELATIVE_PATH, binding.PINNED_ADDITIONAL_FAILURES_SHA256),
                         (binding.ADDITIONAL_FAILURES_V2_RELATIVE_PATH, binding.PINNED_ADDITIONAL_FAILURES_V2_SHA256)):
        study = merge_failure_registries(study, read(path, digest))
    return study


def historical_observation(request, failure, *, study=None):
    """Describe historical recurrence without deciding current acceptance."""
    study = load_failure_registry() if study is None else study
    conditions = exact_conditions(request)
    current_source = request.get("source_commit")
    message = failure.get("message", "") if isinstance(failure, dict) else failure
    if message is not None and not isinstance(message, str):
        raise ValueError("failure message must be text or absent")
    matched = []
    for category in CATEGORIES:
        for entry in study.get(category, []):
            if exact_conditions(entry) != conditions:
                continue
            source = entry["observed_source_commit"]
            matched.append({"case_id": entry["case_id"], "category": category,
                "observed_source_commit": source, "current_source_commit": current_source,
                "source_relation": "unknown" if current_source is None else "same_source" if current_source == source else "different_source",
                "historical_failure": entry["failure"],
                "outcome": "previously_failed_case_now_passed" if failure is None else
                    "historical_signature_recurred" if entry["failure"] in message else "different_failure_signature"})
    return {"matching_historical_cases": matched, "matched_conditions": conditions,
        "waives_checks": False, "changes_acceptance_thresholds": False,
        "signature_comparison": "failure_message_substring_only; recorded metric values are not compared",
        "coverage": "measured_conditions_matched" if matched else "no_historical_observation_at_exact_conditions",
        "note": "Historical observations do not determine current acceptance or exit status"}

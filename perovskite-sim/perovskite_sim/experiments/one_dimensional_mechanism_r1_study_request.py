"""A caller-held, pre-execution R1 case-set anchor, separate from result hashes.

Changing this anchor is a new research request, not verification of the old one.
It authenticates neither arbitrary source code nor an independent scientific review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


SCHEMA = "R1StudyExecutionPlanV1"


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _case_key(key):
    if not isinstance(key, str):
        raise ValueError("invalid planned case key")
    path = PurePosixPath(key)
    if (not key or path.is_absolute()
            or ".." in path.parts or "." in path.parts or path.as_posix() != key
            or "\\" in key):
        raise ValueError("invalid planned case key")


def build_execution_plan(settings, sections, selection, cases):
    if not cases:
        raise ValueError("study selection contains no planned cases")
    for key, request in cases.items():
        _case_key(key)
        if not isinstance(request, dict) or not isinstance(request.get("scope"), str):
            raise ValueError("every planned case requires its exact request and scope")
    finest = sorted(key for key, request in cases.items()
                    if request.get("required_finest_pair") is True)
    value = {
        "schema": SCHEMA,
        "study_settings": settings,
        "sections": list(sections),
        "selection": selection,
        "cases": dict(sorted(cases.items())),
        "required_finest_cases": finest,
        "scope": "caller_frozen_case_set_not_scientific_approval",
        "history_scope": "single_attempt_per_case; retries_require_a_new_study_request",
    }
    # Reject nonfinite inputs and detach mutable/numpy-free caller containers.
    return json.loads(canonical_bytes(value))


def load_execution_plan(path, expected_sha256, *, result_directory=None):
    path = Path(path).resolve()
    if result_directory is not None and path.is_relative_to(Path(result_directory).resolve()):
        raise ValueError("external study request must be held outside the result directory")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("external study request digest is required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("external study request digest mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported study execution plan")
    expected = build_execution_plan(value["study_settings"], value["sections"],
                                    value["selection"], value["cases"])
    if value != expected:
        raise ValueError("planned cases or required finest set differ from request")
    return value


def inventory_coverage(plan, stored_inventory, active_cases):
    """A repacked result cannot redefine the pre-execution case universe."""
    expected = plan["cases"]
    if stored_inventory != expected:
        raise ValueError("archived inventory differs from externally anchored study request")
    unexpected = sorted(set(active_cases) - set(expected))
    if unexpected:
        raise ValueError("unplanned archived study cases: " + ", ".join(unexpected))
    for key, row in active_cases.items():
        request = row.get("conditions", row)
        if request != expected[key]:
            raise ValueError("archived case differs from externally planned request: " + key)
    return sorted(set(expected) - set(active_cases))


def verify_invocation_attempts(record, *, planned_cases, recorded_attempts):
    """Check one invocation's attempts against retained directories.

    This count is local to the invocation, not the cumulative active-case
    count. Missing planned cases and retained interrupted attempts are legal.
    The external manifest protects history; this check rejects inconsistent
    self-resealed records without inventing erased execution history.
    """
    count = record.get("attempted_cases")
    attempts = record.get("invocation_attempts")
    if (type(count) is not int or count < 0 or not isinstance(attempts, list)
            or record.get("attempt_count_scope") != "new_attempts_in_this_invocation"
            or count != len(attempts)):
        raise ValueError("invocation attempt count differs from its recorded attempts")
    seen = set()
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != {"case", "directory"}:
            raise ValueError("invalid invocation attempt identity")
        key, directory = attempt["case"], attempt["directory"]
        if (key not in planned_cases or directory != key + "/AttemptV1"
                or directory in seen):
            raise ValueError("invocation attempt differs from the fixed case request")
        seen.add(directory)
        if directory not in recorded_attempts:
            raise ValueError("recorded invocation attempt is missing: " + directory)
        if recorded_attempts[directory] != planned_cases[key]:
            raise ValueError("recorded invocation attempt request differs: " + directory)
    for row in record.get("cases", []):
        if (row.get("directory") and not row.get("resumed")
                and row.get("status") in {"completed", "failed", "unavailable"}
                and row["directory"] not in seen):
            raise ValueError("invocation result is absent from its recorded attempts")
    return seen


def load_qualification_inputs(path, expected_sha256, *, result_directory):
    """Load caller-approved inputs through an anchor outside the result bundle.

    The caller is responsible for independent review; a digest establishes
    the selected inputs, not whether the scientific approval was warranted.
    No missing input is replaced by a zero bound or an approval.
    """
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("qualification digest requires an external input file")
        return {"schema": "R1StudyQualificationInputsV1", "current_budgets": {},
                "turnover_evidence": {}, "double_domain_evidence": {}, "trusted_evidence": {}}
    path = Path(path).resolve()
    if path.is_relative_to(Path(result_directory).resolve()):
        raise ValueError("qualification inputs must be held outside the result directory")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("external qualification input digest is required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("external qualification input digest mismatch")
    value = json.loads(raw)
    fields = {"schema", "current_budgets", "turnover_evidence", "double_domain_evidence", "trusted_evidence"}
    if isinstance(value, dict) and value.get("schema") == "R1StudyQualificationInputsV2":
        fields.add("numerical_reconciliations")
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema"] not in ("R1StudyQualificationInputsV1", "R1StudyQualificationInputsV2")
            or any(not isinstance(value[name], dict) for name in fields - {"schema"})):
        raise ValueError("invalid external qualification input schema")
    for name, digest in value["trusted_evidence"].items():
        if (not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid externally retained qualification evidence digest")
    return json.loads(canonical_bytes(value))

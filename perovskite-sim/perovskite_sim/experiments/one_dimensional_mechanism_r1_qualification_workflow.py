"""Post-execution qualification requests bind an immutable computation.

The external request and approval hashes remain caller responsibilities.
This module does not sign or invent scientific approval. Changing any input
creates a new analysis request; it never edits the numerical collection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .one_dimensional_mechanism_r1_study_request import canonical_bytes


def require_digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(label + " requires an external SHA256")
    return value


def analysis_request(*, collection_manifest_sha256, calculation_request_sha256,
                     qualification_sha256, candidate_standard_sha256,
                     approved_standard_sha256, source, window_spec, cases,
                     amplitude_ladder_V, window_amplitude_V, linearity_case):
    for label, value in (("collection manifest", collection_manifest_sha256),
                         ("calculation request", calculation_request_sha256),
                         ("qualification inputs", qualification_sha256),
                         ("candidate standard", candidate_standard_sha256),
                         ("approved standard", approved_standard_sha256)):
        require_digest(value, label)
    if approved_standard_sha256 != candidate_standard_sha256:
        raise ValueError("analysis standard differs from independently held approval")
    if not cases or any(not key.startswith(("Linearity/", "Frequency/", "DoubleDomain/")) for key in cases):
        raise ValueError("analysis requires explicit derived cases only")
    if window_spec is not None:
        from .one_dimensional_mechanism_r1_window import validate_window_spec
        window_spec = validate_window_spec(window_spec)
    return json.loads(canonical_bytes({
        "schema": "R1PostExecutionQualificationRequestV1",
        "collection_manifest_sha256": collection_manifest_sha256,
        "calculation_request_sha256": calculation_request_sha256,
        "qualification_inputs_sha256": qualification_sha256,
        "candidate_standard_sha256": candidate_standard_sha256,
        "approved_standard_sha256": approved_standard_sha256,
        "source": source, "window_spec": window_spec, "cases": dict(sorted(cases.items())),
        "amplitude_ladder_V": list(amplitude_ladder_V),
        "window_amplitude_V": window_amplitude_V, "linearity_case": linearity_case,
        "scope": "post_execution_derived_analysis_no_new_physical_solutions",
    }))


def load_analysis_request(path, expected_sha256, expected):
    require_digest(expected_sha256, "analysis request")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("external analysis request digest mismatch")
    value = json.loads(raw)
    if value != expected:
        raise ValueError("analysis inputs or scope differ from externally frozen request")
    return value


def analysis_cases(grids, amplitudes, sections, selected=()):
    """Enumerate analysis independently of whether a result will pass."""
    cases = {}
    if "amplitude" in sections:
        for coarse, fine in zip(amplitudes[:-1], amplitudes[1:]):
            cases[f"Linearity/A{coarse}ToA{fine}"] = {
                "kind": "amplitude_linearity", "coarse_amplitude_V": coarse, "fine_amplitude_V": fine}
    if "reconstruct" in sections:
        for n in grids:
            cases[f"Frequency/N{n}"] = {"kind": "frequency_coverage", "grid": n}
        cases[f"DoubleDomain/N{grids[-1]}/D"] = {"kind": "double_domain", "grid": grids[-1]}
    if selected:
        if set(selected) - set(cases):
            raise ValueError("requested analysis case is not in the frozen analysis sections")
        cases = {key: cases[key] for key in selected}
    return cases


def verify_collection_receipt(summary):
    """Recorded physical failures may remain, but incomplete checking cannot."""
    requirements = summary.get("requirements", {})
    if (requirements.get("all_recorded_cases_checked") is not True
            or requirements.get("verification_completed_without_error") is not True
            or requirements.get("external_case_set_verified") is not True
            or summary.get("not_recomputed_in_this_invocation")):
        raise ValueError("post-execution analysis requires a complete collection verification receipt")
    return {"schema": "R1CollectionVerificationReceiptV1",
            "study_request_sha256": summary["study_request_sha256"],
            "diagnostic_failure_count": summary["diagnostic_failure_count"],
            "missing_cases": summary["missing_cases"],
            "requirements": requirements,
            "scientific_acceptance_inferred": False,
            "scope": "all_recorded_cases_checked_not_all_cases_scientifically_passed"}

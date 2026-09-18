"""Frozen calculation and review identities cannot substitute for each other."""
import copy
import hashlib

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification_workflow import (
    analysis_request, analysis_cases, load_analysis_request, verify_collection_receipt,
    read_only_derivation, require_collection_phase,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_request import canonical_bytes


def request():
    return analysis_request(collection_manifest_sha256="a"*64, calculation_request_sha256="b"*64,
        qualification_sha256="c"*64, candidate_standard_sha256="d"*64,
        approved_standard_sha256="d"*64, source={"sha256": "e"*64}, window_spec=None,
        cases=analysis_cases([16], [.01, .005, .0025], ["reconstruct"], ["Frequency/N16"]),
        amplitude_ladder_V=[.01, .005, .0025], window_amplitude_V=.0025, linearity_case=None)


@pytest.mark.parametrize("field", ["collection_manifest_sha256", "calculation_request_sha256",
    "qualification_inputs_sha256", "source", "cases", "window_amplitude_V"])
def test_resealed_analysis_cannot_replace_any_caller_frozen_input(tmp_path, field):
    original = request()
    path = tmp_path/"AnalysisRequestV1.json"
    path.write_bytes(canonical_bytes(original))
    anchor = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_analysis_request(path, anchor, original) == original
    altered = copy.deepcopy(original)
    altered[field] = "changed"
    path.write_bytes(canonical_bytes(altered))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_analysis_request(path, anchor, original)
    with pytest.raises(ValueError, match="inputs or scope"):
        load_analysis_request(path, hashlib.sha256(path.read_bytes()).hexdigest(), original)


def test_analysis_selection_never_starts_a_physical_case():
    cases = analysis_cases([16, 32], [.01, .005, .0025], ["amplitude", "reconstruct"])
    assert set(cases) == {"Linearity/A0.005ToA0.0025",
                          "Frequency/N16", "Frequency/N32", "DoubleDomain/N32/D"}
    with pytest.raises(ValueError, match="not in the frozen"):
        analysis_cases([16], [.01, .005], ["reconstruct"], ["Matrix/N16/T1/F1p0"])


def test_derived_phase_rejects_real_preparation_and_dc_entry_points_before_solving():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import solve_controlled_dc
    with read_only_derivation() as attempts:
        with pytest.raises(RuntimeError, match="prepare_common_state"):
            prepare_common_state(None, 16, None)
        with pytest.raises(RuntimeError, match="solve_controlled_dc"):
            solve_controlled_dc(None, 16, None, None)
    assert attempts == ["prepare_common_state", "solve_controlled_dc"]
    require_collection_phase("allowed_after_context_exit")


def test_complete_verification_receipt_preserves_physical_failures_and_missing_cases():
    summary = {"requirements": {"all_recorded_cases_checked": True,
        "verification_completed_without_error": True, "external_case_set_verified": True},
        "not_recomputed_in_this_invocation": [], "study_request_sha256": "b"*64,
        "diagnostic_failure_count": 2, "missing_cases": ["not-run"]}
    receipt = verify_collection_receipt(summary)
    assert receipt["diagnostic_failure_count"] == 2 and receipt["missing_cases"] == ["not-run"]
    assert receipt["scientific_acceptance_inferred"] is False
    for field in summary["requirements"]:
        altered = copy.deepcopy(summary)
        altered["requirements"][field] = False
        with pytest.raises(ValueError, match="complete collection verification"):
            verify_collection_receipt(altered)

"""Real saved-state baselines for the adjacent R8 record-boundary counterexamples."""
from copy import deepcopy
import json

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import check_zero_excitation
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import verify_result_records
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.fixtures.r1_failure_evidence import refresh_stage_one_failure_witness
from tests.integration.test_one_dimensional_mechanism_r1_physics_result_records import (
    physical_case, bundle, fail_prefix, FIXTURE,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner, PROJECT

pytestmark = pytest.mark.slow


def save(path, record, *, sealed=False):
    record = deepcopy(record)
    if sealed:
        record["sha256"] = common.digest({k: v for k, v in record.items() if k != "sha256"})
    path.write_text(json.dumps(record, allow_nan=False))


def test_real_step_baseline_keeps_physics_acceptance_and_scope(bundle):
    result = verify_result_records(*bundle)
    assert result["scientifically_accepted"]
    assert result["range_only"] == ["historical_iteration_maximum_jacobian_nnz"]
    assert result["failure_scope"] is None


@pytest.mark.parametrize("field,value", [
    ("converged", True), ("solver_iterations", 3), ("spatial_convergence_certified", True),
])
def test_unclassified_accepted_row_claim_is_rejected_by_row_guard(bundle, field, value):
    output, completion = bundle
    result = read_json(output / "StepResultV1.json")
    for row in result["accepted_steps"]:
        row[field] = value
    save(output / "StepResultV1.json", result, sealed=True)
    save(output / "AcceptedStepsV1.json", result["accepted_steps"])
    with pytest.raises(ValueError, match="unclassified accepted row"):
        verify_result_records(output, completion)


def test_persistence_failure_cannot_hide_inside_a_successful_step(bundle):
    output, completion = bundle
    result = read_json(output / "StepResultV1.json")
    result["persistence_failure"] = {"type": "OSError", "message": "invented"}
    save(output / "StepResultV1.json", result, sealed=True)
    with pytest.raises(ValueError, match="passed result contains persistence_failure"):
        verify_result_records(output, completion)


@pytest.mark.parametrize("container", [(), ("dc_state",), ("dc_state", "certificate")])
def test_prepared_claim_containers_are_closed(bundle, container):
    output, completion = bundle
    prepared = read_json(output / "PreparedStateV1.json")
    target = prepared
    for key in container:
        target = target[key]
    target["mechanism_identified"] = True
    save(output / "PreparedStateV1.json", prepared, sealed=True)
    completion = dict(completion, stage="prepare")
    with pytest.raises(ValueError, match="unclassified prepared"):
        verify_result_records(output, completion)


@pytest.fixture
def zero_bundle(bundle, runner):
    output, completion = bundle
    prepared = common.R1PreparedState.from_dict(read_json(output / "PreparedStateV1.json"))
    result = check_zero_excitation(load_device_from_yaml(PROJECT / FIXTURE), 16,
        approved_r1_binding(), prepared, controls="D")
    runner.write_json(output / "ZeroExcitationV1.json", result)
    return output, dict(completion, stage="zero-check")


def test_zero_and_prepare_baselines_have_scoped_metadata_disclosure(zero_bundle):
    output, completion = zero_bundle
    for stage in ("prepare", "zero-check"):
        result = verify_result_records(output, dict(completion, stage=stage))
        assert result["scientifically_accepted"]
        assert "historical_metadata_not_replayed" in result["reconstruction_scope"]
        assert "prepared.dc_state.certificate.optimizer_nfev" in result["provenance_only"]


@pytest.mark.parametrize("path,value,message", [
    (("scope",), "full_three_axis_converged", "zero metadata scope"),
    (("version",), "r1-2-qualified", "zero metadata version"),
    (("certificate", "scope"), "mechanism_identified", "zero certificate scope"),
    (("certificate", "relative_dynamic_current_certified"), True, "zero dynamic-current disclaimer"),
    (("certificate", "finite_numeric_evidence", "scope"), "forged", "zero finite numeric certificate"),
    (("convergence_certified",), True, "unclassified zero-excitation"),
    (("controls", "D", "converged"), True, "unclassified zero control"),
])
def test_zero_claims_are_bound_without_disturbing_equilibrium(zero_bundle, path, value, message):
    output, completion = zero_bundle
    result = read_json(output / "ZeroExcitationV1.json")
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    save(output / "ZeroExcitationV1.json", result, sealed=True)
    with pytest.raises(ValueError, match=message):
        verify_result_records(output, completion)


def test_clean_failed_prefix_does_not_claim_the_failure_origin_was_reproduced(bundle, runner):
    fail_prefix(bundle, runner)
    result = verify_result_records(*bundle)
    scope = result["failure_scope"]
    assert not result["scientifically_accepted"]
    assert result["physical_limits_satisfied"] is False
    assert result["saved_prefix_physical_limits_satisfied"] is True
    assert scope["saved_schedule_checked"]
    assert scope["saved_row_count"] == scope["persisted_row_count"] == 2
    assert scope["expected_row_count"] == 10
    assert not scope["complete_requested_schedule"]
    assert not scope["original_execution_extent_verified"]
    assert scope["failure_origin_status"] == "not_reconstructed_from_saved_prefix"


def test_failure_label_cannot_claim_a_missing_physical_witness(bundle, runner):
    result = fail_prefix(bundle, runner)
    output, completion = bundle
    failure = {"type": "PhysicalCheckFailure", "message": "gate witness erased",
               "record_index": len(result["accepted_steps"]) - 1, "reasons": []}
    result["failure"] = failure
    result["certificate"].update(failed_record_index=failure["record_index"],
                                  physical_checks=result["accepted_steps"][-1]["physical_checks"])
    completion["failure"] = failure
    runner.write_json(output / "FailureV1.json", failure)
    runner._record_failure_result(output, result)
    refresh_stage_one_failure_witness(output)
    with pytest.raises(ValueError, match="no violating saved row"):
        verify_result_records(output, completion)

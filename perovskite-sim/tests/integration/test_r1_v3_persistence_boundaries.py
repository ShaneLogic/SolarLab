"""Real saved-state replay still accepts an honest durable-write failure."""
from copy import deepcopy

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_evidence import read_json
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import verify_result_records
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.integration.test_one_dimensional_mechanism_r1_physics_result_records import physical_case, bundle, FIXTURE
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner, PROJECT

pytestmark = pytest.mark.slow


@pytest.fixture
def persistence_bundle(bundle, runner):
    output, completion = bundle
    prepared = common.R1PreparedState.from_dict(read_json(output / "PreparedStateV1.json"))
    persisted = []

    def interrupted_writer(row):
        if row["dt_s"] > 0:
            raise OSError("injected durable-write failure")
        persisted.append(deepcopy(row))

    with pytest.raises(protocol.R1RunError, match="accepted_step_persistence_failed") as failure:
        protocol.run_r1_step(load_device_from_yaml(PROJECT / FIXTURE), 16,
            read_json(output / "ReferenceBindingV1.json"), prepared, times_s=[0., 1e-9],
            policy=protocol.r1_policy(), physics_evidence=True,
            accepted_step_observer=interrupted_writer)
    record = failure.value.result
    for name in ("StepResultV1.json", "StepResultV1.npz", "AcceptedStepsV1.npz"):
        (output / name).unlink(missing_ok=True)
    runner._record_failure_result(output, record)
    runner.write_json(output / "AcceptedStepsV1.json", persisted)
    runner.write_json(output / "FailureV1.json", record["failure"])
    completion.update(status="failed", failure=record["failure"],
        accepted_record_count=len(persisted), observed_record_count=len(record["accepted_steps"]),
        persisted_record_count=len(persisted), persisted_finite_step_count=0,
        physical_passed_finite_step_count=0)
    return output, completion


def test_real_write_failure_keeps_raw_observed_row_separate_from_durable_prefix(persistence_bundle):
    result = verify_result_records(*persistence_bundle)
    assert result["content_matches_recomputed"]
    assert not result["scientifically_accepted"]
    scope = result["failure_scope"]
    assert scope["saved_row_count"] == 2
    assert scope["persisted_row_count"] == 1
    assert scope["expected_row_count"] == 10
    assert not scope["complete_requested_schedule"]
    assert not scope["original_execution_extent_verified"]
    assert scope["failure_origin_status"] == "not_reconstructed_from_saved_prefix"


def test_persistence_coordinates_cannot_be_shifted_to_a_different_raw_row(persistence_bundle, runner):
    output, completion = persistence_bundle
    record = read_json(output / "FailedResultV1.json")
    record["persistence_failure"]["record_index"] = 0
    record["accepted_steps"][-1]["persistence_failure"]["record_index"] = 0
    record["failure"]["record_index"] = 0
    runner._record_failure_result(output, record)
    with pytest.raises(ValueError, match="persistence failure must identify the last saved raw row"):
        verify_result_records(output, completion)


@pytest.mark.parametrize("mutation", ["none", "claim", "mismatch"])
def test_nested_persistence_copy_is_classified_and_bound(persistence_bundle, runner, mutation):
    output, completion = persistence_bundle
    record = read_json(output / "FailedResultV1.json")
    nested = deepcopy(record["persistence_failure"])
    record["failure"]["persistence_failure"] = nested
    if mutation == "claim":
        nested["mechanism_identified"] = True
    elif mutation == "mismatch":
        nested["message"] = "different historical failure"
    runner._record_failure_result(output, record)
    if mutation == "none":
        assert verify_result_records(output, completion)["content_matches_recomputed"]
    else:
        message = "unclassified nested persistence" if mutation == "claim" else "failure persistence provenance"
        with pytest.raises(ValueError, match=message):
            verify_result_records(output, completion)

"""A saved prefix cannot prove the missing termination state passed physics."""
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness, verify_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import failure_scope_report


def inputs():
    rows = [{"time_s": 1e-9, "dt_s": 1e-9, "substeps": 1, "phase": "accepted_regular_step",
             "solver_accepted": True, "physical_checks_passed": True, "state": {"coordinate": [1.]}}]
    return dict(source_commit="source", protocol={"intervals": 32, "control": "D", "times_s": [0., 1e-9, 1e-8],
        "policy": {"maximum_newton_iterations": 100}}, failure={"type": "R1RunError", "message": "stalled"},
        result={"accepted_steps": rows, "failure": {"type": "InterfaceDefectTransientError", "message": "stalled"}},
        persisted_rows=rows)


def test_honest_unwitnessed_failure_retains_unknown_terminal_physics():
    arguments = inputs()
    witness = build_failure_witness(**arguments)
    checked = verify_failure_witness(witness, **arguments)
    assert checked["termination_binding_verified"]
    assert checked["terminal_state_physics_verified"] is None
    assert not checked["terminal_state_available"]
    scope = failure_scope_report(arguments["result"], {"checked_row_count": 1, "expected_row_count": 3,
                                                      "physical_limits_satisfied": True})
    assert scope["saved_prefix_physical_limits_satisfied"]
    assert scope["terminal_state_physics_verified"] is None
    assert not scope["whole_failed_run_physical_limits_satisfied"]


@pytest.mark.parametrize("change", ["failure_label", "last_row", "policy", "terminal_status"])
def test_failure_witness_binds_termination_extent_settings_and_availability(change):
    arguments = inputs()
    witness = build_failure_witness(**arguments)
    if change == "failure_label":
        arguments["result"]["failure"]["type"] = "OtherFailure"
    elif change == "last_row":
        arguments["result"]["accepted_steps"] = []
    elif change == "policy":
        arguments["protocol"]["policy"]["maximum_newton_iterations"] = 1000
    else:
        witness["terminal_evidence"]["available"] = True
    with pytest.raises(ValueError, match="failure witness differs"):
        verify_failure_witness(witness, **arguments)


def test_physical_violation_witness_remains_a_failed_run():
    arguments = inputs()
    arguments["result"]["failure"]["type"] = "PhysicalCheckFailure"
    arguments["result"]["accepted_steps"][-1].update(physical_checks_passed=False,
                                                    physical_failure_reasons=["charge_balance"])
    witness = build_failure_witness(**arguments)
    assert witness["terminal_evidence"]["kind"] == "saved_physical_failure_row"
    scope = failure_scope_report(arguments["result"], {"physical_limit_violations": [{"row": 0}],
                                                      "physical_limits_satisfied": False})
    assert scope["saved_last_row_physics_verified"] is True
    assert scope["saved_last_row_physics_passed"] is False
    assert scope["terminal_state_physics_verified"] is None
    assert not scope["scientifically_accepted"]

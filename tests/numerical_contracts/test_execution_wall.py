"""Execution wall policy regressions independent of device plans and solvers."""

import json

from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.coupled_device_prototype import digest


def _execution_wall_test_input():
    request = {"budgets": {"wall_s": 4200, "charge_C": 1e-12, "current_A": 1e-3,
                           "native_steps": 200000, "residual_calls": 2000000},
               "controls": {"rtol": 1e-12, "first_step": 2**-32}}
    admission = {"authority": "Root", "voltage_lift_native_authorized": True,
                 "operation_id": "manufactured-wall-test", "coordinator_message": "msg_0123456789ab"}
    admission["execution_wall_limit"] = {
        "schema": "solarlab.native-execution-wall.v1", "request_sha256": digest(request),
        "original_wall_s": 4200, "wall_s": 10800, "operation_id": admission["operation_id"],
        "coordinator_message": admission["coordinator_message"]}
    return request, admission


def test_execution_wall_default_preserves_problem_budget():
    from scripts.benchmarks.coupled_device_prototype import voltage_lift_execution_wall_policy

    request, admission = _execution_wall_test_input()
    del admission["execution_wall_limit"]
    before = json.dumps([request, admission], sort_keys=True)
    policy = voltage_lift_execution_wall_policy(request, admission)
    assert policy == {"schema": "solarlab.native-execution-wall-observed.v1",
                      "original_wall_s": 4200, "effective_wall_s": 4200, "admission_binding": None}
    assert json.dumps([request, admission], sort_keys=True) == before


def test_execution_wall_authorized_override_preserves_inputs():
    from scripts.benchmarks.coupled_device_prototype import voltage_lift_execution_wall_policy

    request, admission = _execution_wall_test_input()
    before = json.dumps([request, admission], sort_keys=True)
    policy = voltage_lift_execution_wall_policy(request, admission)
    assert policy["original_wall_s"] == 4200 and policy["effective_wall_s"] == 10800
    assert policy["admission_binding"] == admission["execution_wall_limit"]
    assert json.dumps([request, admission], sort_keys=True) == before
    assert voltage_lift_execution_wall_policy(request, json.loads(json.dumps(admission))) == policy
    admission["execution_wall_limit"]["wall_s"] = 1
    assert policy["effective_wall_s"] == policy["admission_binding"]["wall_s"] == 10800


def _check_execution_wall_rejection(case):
    from scripts.benchmarks.coupled_device_prototype import voltage_lift_execution_wall_policy

    request, admission = _execution_wall_test_input()
    grant = admission["execution_wall_limit"]
    invalid_numbers = {"zero": 0, "negative": -1, "infinite": float("inf"),
                       "nan": float("nan"), "boolean": True}
    if case in invalid_numbers:
        grant["wall_s"] = invalid_numbers[case]
    elif case == "null":
        admission["execution_wall_limit"] = None
    elif case == "bad_schema":
        grant["schema"] = "unbound-wall-limit"
    elif case == "missing_binding":
        del grant["operation_id"]
    elif case == "foreign_request":
        grant["request_sha256"] = "0"*64
    elif case == "foreign_operation":
        grant["operation_id"] = "another-operation"
    elif case == "foreign_message":
        grant["coordinator_message"] = "msg_abcdef"
    elif case == "stale_original_wall":
        grant["original_wall_s"] = 4201
    elif case == "scientific_extra":
        grant["charge_C"] = 1.0
    elif case == "not_root":
        admission["authority"] = "worker"
    elif case == "not_authorized":
        admission["voltage_lift_native_authorized"] = False
    else:
        assert case == "tampered_request"
        request["controls"]["rtol"] = 2e-12
    try:
        voltage_lift_execution_wall_policy(request, admission)
    except ContractError as error:
        assert str(error) == "voltage_lift_execution_wall_binding", case
    else:
        raise AssertionError("unbound execution wall accepted: " + case)


def test_execution_wall_rejects_malformed_or_stale_admission():
    for case in (
        "zero", "negative", "infinite", "nan", "boolean", "null", "bad_schema",
        "missing_binding", "foreign_request", "foreign_operation", "foreign_message",
        "stale_original_wall", "scientific_extra", "not_root", "not_authorized", "tampered_request",
    ):
        _check_execution_wall_rejection(case)

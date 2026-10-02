"""Actual R1 state reconstruction supplements the assembly-level counterexamples."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy,run_r1_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state,restore_common_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as validation
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import canonical
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE

pytestmark=pytest.mark.slow


@pytest.fixture(scope="module")
def actual():
    stack,binding=load_device_from_yaml(FIXTURE),approved_r1_binding()
    policy=r1_policy()
    prepared=prepare_common_state(stack,16,binding,policy=policy)
    result=run_r1_step(stack,16,binding,prepared,policy=policy,physics_evidence=True)
    return stack,binding,prepared,result


def test_actual_all_nested_levels_have_independent_physical_checks(actual):
    stack,binding,prepared,result=actual
    rows=result["accepted_steps"]
    assert {row["substeps"] for row in rows}=={1,2,4}
    assert all(row["physics_reconstruction"]["independent_physics"]["passed"] for row in rows)
    audit=verify_r1_step_physics(stack,16,binding,prepared,result)
    assert audit["independent_physics_passed"] is True
    assert audit["checked_row_count"]==31


def test_actual_shared_solver_and_publisher_current_error_is_detected(actual):
    stack,binding,prepared,result=actual
    system,before=restore_common_state(prepared,stack,16,binding,policy=r1_policy())
    initial=build_initial_step(system,before,.005,policy=r1_policy())
    working,previous=initial.system.rebase(initial.zero_plus)
    working.set_voltage_lift(.005,previous)
    row=result["accepted_steps"][1]
    state=working.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]),.005)
    honest=independent_physics_row(working,state,previous,row["dt_s"],reported=row["physics_reconstruction"])
    assert honest["passed"]
    wrong=replace(state,current_n=2*state.current_n,current_p=2*state.current_p)
    published=deepcopy(row["physics_reconstruction"])
    published["electron_current_A_m2"]=wrong.current_n
    published["hole_current_A_m2"]=wrong.current_p
    checked=independent_physics_row(working,wrong,previous,row["dt_s"],reported=published)
    assert checked["checks"]["electron_current_A_m2_state_content_matches"]["passed"] is False
    assert checked["checks"]["electron_current_A_m2_reported_content_matches"]["passed"] is False
    assert checked["passed"] is False


def test_coarse_level_independent_rejection_vetoes_complete_trace(actual,monkeypatch):
    """Inject a failed audit witness to test propagation, not the constitutive law."""
    stack,binding,prepared,result=actual
    original=validation.capture_r1_physics_row
    def inject(*args,**kwargs):
        value=original(*args,**kwargs)
        if args[3] is not None and args[5]==1e-9:
            evidence=value["independent_physics"]
            evidence["metrics"]["internal_face_current_spread_relative"]=3e-6
            evidence["checks"]["internal_face_current_spread_relative"]["passed"]=False
            evidence["reasons"]=["internal_face_current_spread_relative"]
            evidence["passed"]=False
        return value
    record=deepcopy(result)
    witness=record["accepted_steps"][1]["physics_reconstruction"]["independent_physics"]
    witness["metrics"]["internal_face_current_spread_relative"]=3e-6
    witness["checks"]["internal_face_current_spread_relative"]["passed"]=False
    witness["reasons"]=["internal_face_current_spread_relative"]
    witness["passed"]=False
    failed_row=record["accepted_steps"][1]
    # Keep the row's published gate status consistent with its failed witness.
    # A metadata mismatch must not preempt the all-level certificate check.
    checks=protocol._physical_step_checks(failed_row["physical"],finite_step=True,
                                          policy=r1_policy(),evidence=failed_row)
    assert checks["reasons"]==["independent_physics_failed"]
    failed_row.update(physical_checks=checks,physical_checks_passed=False,
                      physical_failure_reasons=checks["reasons"])
    assert failed_row["substeps"]==1
    monkeypatch.setattr(validation,"capture_r1_physics_row",inject)
    audit=verify_r1_step_physics(stack,16,binding,prepared,record)
    assert audit["checked_row_count"]==31
    assert not audit["independent_physics_passed"]
    assert not audit["physical_limits_satisfied"]
    assert not audit["certified"]
    assert any(item.get("row")==1 and item.get("substeps")==1
               for item in audit["physical_limit_violations"])


@pytest.mark.parametrize("persistence_fails",[False,True])
def test_independent_failure_is_serialized_before_protocol_rejection(actual,monkeypatch,tmp_path,persistence_fails):
    """Observe a real solver-accepted row with a deliberately failed audit."""
    stack,binding,prepared,_=actual
    original=validation.capture_r1_physics_row
    def inject(*args,**kwargs):
        value=original(*args,**kwargs)
        if args[3] is not None:
            independent=value["independent_physics"]
            independent["metrics"]["internal_face_current_spread_relative"]=3e-6
            independent["checks"]["internal_face_current_spread_relative"]["passed"]=False
            independent["reasons"]=["internal_face_current_spread_relative"]
            independent["passed"]=False
        return value
    journal=tmp_path/"AcceptedStepsV1.jsonl"
    def observe(row):
        # Serialize now: an append(row) list would alias the later mutation.
        with journal.open("a") as stream:
            stream.write(canonical(row)+"\n")
            stream.flush()
        if row["solver_accepted"] and persistence_fails:
            raise OSError("injected failure after a complete serialized row")
    monkeypatch.setattr(validation,"capture_r1_physics_row",inject)
    with pytest.raises(protocol.R1RunError,match="independent_physics_failed") as error:
        protocol.run_r1_step(stack,16,binding,prepared,policy=r1_policy(),
                             accepted_step_observer=observe,physics_evidence=True)
    partial=error.value.result
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows)==len(partial["accepted_steps"])==2
    failed=partial["accepted_steps"][-1]
    assert failed["substeps"]==1 and failed["solver_accepted"]
    assert failed["physical_checks_passed"] is False
    assert failed["physical_failure_reasons"]==["independent_physics_failed"]
    assert failed["physical_checks"]["checks"]["independent_physics"]["passed"] is False
    assert partial["certificate"]["certified"] is False
    assert partial["certificate"]["reasons"]==["independent_physics_failed"]
    assert partial["failure"]["type"]=="PhysicalCheckFailure"
    expected=deepcopy(partial["accepted_steps"])
    if persistence_fails:
        persistence=expected[-1].pop("persistence_failure")
        assert persistence==partial["persistence_failure"]==partial["failure"]["persistence_failure"]
        assert persistence["record_index"]==1
        assert partial["certificate"]["secondary_reasons"]==["accepted_step_persistence_failed"]
        assert "persistence_failure" not in rows[-1]
    else:
        assert "persistence_failure" not in partial
    assert canonical(rows)==canonical(expected)

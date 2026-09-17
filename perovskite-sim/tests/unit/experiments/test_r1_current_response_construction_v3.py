"""The actual runner construction must retain early and interior current samples."""

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    compare_step_current_charge, step_current_charge_responses,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_physics_study import runner  # noqa: F401


TIMES = [0., 1e-9, 1e-8, 1e-6, 1e-4]


def record():
    return {"times_s": TIMES.copy(), "policy": {"refinement_substeps": [1, 2, 4]},
            "regular_currents": [{"report_contact_current_A_m2": [1e-5, 1e-5]} for _ in TIMES],
            "initial_event": {"impulse_charge_C_m2": 2e-6},
            "accepted_steps": [{"substeps": 4, "time_s": time, "regular_integrated_charge_C_m2": 1e-5*time}
                               for time in TIMES]}


@pytest.mark.parametrize("owner", ["helper", "runner"])
@pytest.mark.parametrize("index", [0, 2, 3])
@pytest.mark.parametrize("contact", [0, 1])
def test_current_construction_rejects_an_early_or_interior_failure_with_a_matching_final_point(runner, owner, index, contact):
    compare = compare_step_current_charge if owner == "helper" else runner.compare_step_current_charge
    left, right = record(), record()
    baseline = compare(left, right, [0., 0.], [0., 0.])
    assert baseline["within_compared_budgets"]
    assert baseline["regular_current"]["scalar_comparison_count"] == 10
    right["regular_currents"][index]["report_contact_current_A_m2"][contact] += 1e-6
    changed = compare(left, right, [0., 0.], [0., 0.])
    assert left["regular_currents"][-1] == right["regular_currents"][-1]
    assert not changed["within_compared_budgets"]
    assert not changed["regular_current"]["passed"]
    assert changed["integrated_charge"]["passed"]
    failure = changed["regular_current"]["failures"][0]
    assert failure["coordinates"]["time_s"] == TIMES[index]
    assert failure["component"] == ("left_contact", "right_contact")[contact]


def test_caller_fixed_time_axis_detects_removal_of_an_interior_sample():
    shortened = record()
    shortened["times_s"].pop(2)
    shortened["regular_currents"].pop(2)
    with pytest.raises(ValueError, match="complete requested time axis"):
        step_current_charge_responses(shortened, [0., 0.], expected_times_s=TIMES)


def test_regular_current_is_not_replaced_by_backward_euler_average_or_baseline():
    left, right = record(), record()
    for item in right["regular_currents"]:
        item["report_contact_current_A_m2"] = [1.1e-5, 1.1e-5]
    for row in right["accepted_steps"]:
        row["regular_integrated_charge_C_m2"] = 1.1e-5*row["time_s"]
        row["contact_maxwell_A_m2"] = [-999., -999.]
    assert compare_step_current_charge(left, right, [0., 0.], [1e-6, 1e-6])["within_compared_budgets"]

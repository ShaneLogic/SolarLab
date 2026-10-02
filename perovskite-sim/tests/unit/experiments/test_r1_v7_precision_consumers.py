"""Real N16 DD state/consumer checks; no fake precision backend."""
from decimal import Decimal

import pytest
from threadpoolctl import threadpool_limits

from scripts.check_r1_v7_precision_consumers import (
    absolute_response_report, prepare_real_system, run_checks,
)


@pytest.fixture(scope="module")
def actual_report():
    with threadpool_limits(1):
        system, state, frozen, preparation = prepare_real_system()
        assert type(system).__name__ == "CompensatedR1System"
        assert type(state).__name__ == "PrecisionState"
        assert preparation["prepared_sha256"]
        return run_checks(system, state, frozen)


def test_actual_dd_initial_and_controlled_states_match_independent_oracle(actual_report):
    assert len(actual_report["cases"]) == 3
    for case in actual_report["cases"]:
        assert case["oracle"]["qualified"], case["name"]


@pytest.mark.parametrize("fault", ["diffusion", "thermal_voltage", "drift_sign", "omit_ion", "single_face_sign"])
def test_actual_dd_shared_faults_are_detected_by_each_side_oracle(actual_report, fault):
    cases = [item for item in actual_report["faults"] if item["fault"] == fault]
    assert len(cases) == 3
    for case in cases:
        assert case["original_shared_difference_passed"]
        assert case["detected"], (fault, case["case"])


@pytest.mark.parametrize("consumer", [
    "rebase_potential", "potential_increment", "ion_flux_phi_low", "interface_displacement_phi_low",
    "rebase_hole", "storage_hole_low", "charge_hole_low", "poisson_hole_low",
    "rebase_ion", "storage_ion_low", "charge_ion_low", "poisson_ion_low", "ion_flux_population_low",
    "jacobian",
])
def test_actual_state_consumer_uses_the_low_part(actual_report, consumer):
    assert actual_report["consumers"][consumer]["passed"], actual_report["consumers"][consumer]


def test_actual_dd_boundary_leak_is_checked_before_and_after_divergence(actual_report):
    assert actual_report["boundary_fault"]["actual_calls"]
    assert actual_report["boundary_fault"]["detected"]
    assert not actual_report["boundary_fault"]["result"]["qualified"]


def test_consumer_report_does_not_promote_operator_states_to_transient_acceptance(actual_report):
    assert actual_report["run_class"] == "development"
    assert actual_report["actual_accepted_transient_steps"] == 0
    assert not actual_report["P1_qualified"]


@pytest.mark.parametrize("consumer", ["poisson_hole_low", "poisson_ion_low"])
def test_absolute_roundoff_budget_is_resolvable_and_rejects_ignored_low_bits(actual_report, consumer):
    check = actual_report["consumers"][consumer]
    assert check["criterion"] == "absolute_DD_roundoff_bound_and_resolvable_nonzero_response"
    assert check["response_interval_excludes_zero"]
    assert Decimal(check["minimum_signal_to_bound"]) > 1
    assert check["bound_derivation"]["not_a_physical_acceptance_limit"]
    # Comparator counterexample, explicitly not an implementation fault run:
    # ignoring the low input gives zero response and must fail this same bound.
    ignored = absolute_response_report([0] * len(check["expected"]), check["expected"],
                                       check["absolute_error_bounds"])
    assert not ignored["passed"]
    assert not ignored["actual_nonzero_correct_direction"]


def test_unresolvable_poisson_difference_fails_closed():
    report = absolute_response_report(["1e-40"], ["1e-40"], ["2e-40"])
    assert not report["passed"]
    assert not report["response_interval_excludes_zero"]

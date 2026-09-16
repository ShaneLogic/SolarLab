"""Controlled common-state runs retain their initial charge and all steps."""

from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE
from tests.fixtures.r1_reference import approved_r1_binding


@pytest.fixture(scope="module")
def shared():
    stack = load_device_from_yaml(FIXTURE)
    reference = approved_r1_binding()
    policy = protocol.r1_policy()
    prepared = prepare_common_state(stack, 16, reference, policy=policy)
    return stack, reference, policy, prepared


@pytest.mark.slow
@pytest.mark.parametrize("control", list("ABCD"))
def test_controlled_step_retains_common_state_and_separates_impulse(shared, control):
    stack, reference, policy, prepared = shared
    observed = []
    result = protocol.run_r1_step(stack, 16, reference, prepared,
                                 control=control, policy=policy,
                                 accepted_step_observer=observed.append)
    assert result["certificate"]["certified"]
    assert not result["certificate"]["zero_plus_is_equilibrium"]
    assert result["prepared_sha256"] == prepared.sha256
    assert result["reference_sha256"] == reference["sha256"]
    assert len(observed) == len(result["accepted_steps"]) == 31
    initial = result["initial_event"]
    original = prepared.to_dict()["state"]
    for name, key in (("electron_density_m3", "n_m3"), ("hole_density_m3", "p_m3"),
                      ("positive_ion_density_m3", "positive_m3"), ("interface_occupancy", "occupancy")):
        np.testing.assert_array_equal(initial["zero_minus"][name], original[key])
        np.testing.assert_array_equal(initial["zero_plus"][name], original[key])
    assert np.count_nonzero(original["sheet_charge_C_m2"])
    expected_impulse = EPS_0 * 10.0 / 2e-7 * 0.005
    assert initial["impulse_charge_C_m2"] == pytest.approx(expected_impulse, rel=2e-15)
    for substeps in policy.refinement_substeps:
        rows = [r for r in result["accepted_steps"] if r["substeps"] == substeps and r["dt_s"] > 0]
        integrated = result["junction_polarity"] * sum(
            r["physical"]["contact_maxwell_A_m2"][0] * r["dt_s"] for r in rows)
        assert result["charge_integral"]["regular_by_substeps_C_m2"][substeps] == pytest.approx(integrated, rel=2e-15)
        assert result["charge_integral"]["complete_by_substeps_C_m2"][substeps] == pytest.approx(expected_impulse+integrated, rel=2e-15)
    assert all(result["certificate"]["exact_freeze_checks"].values())
    if control in "AC":
        for row in result["accepted_steps"]:
            np.testing.assert_array_equal(row["state"]["positive_m3"], original["positive_m3"])
    if control in "AB":
        for row in result["accepted_steps"]:
            np.testing.assert_array_equal(row["state"]["occupancy"], original["occupancy"])
            np.testing.assert_array_equal(row["state"]["capture_m2_s"], 0.0)
    assert result["accepted_state_arrays"]["positive_m3"].shape[0] == len(observed)
    assert result["version"] == "r1-1-controls-and-initial-charge-v4"
    trap_summary = result["certificate"]["trap_storage"]
    assert trap_summary["checked_finite_step_count"] == 28
    assert sum(trap_summary["branch_counts"]["all"].values()) == 28
    assert sum(trap_summary["branch_counts"]["active_trap"].values()) == (28 if control in "CD" else 0)
    assert result["certificate"]["metrics"]["trap_storage_normalized_error"] <= 1.
    for row in observed:
        check = row["physical"]["trap_storage_check"]
        assert row["physical_checks_passed"]
        assert row["physical_checks"]["passed"]
        assert row["physical_failure_reasons"] == []
        assert row["physical_checks"]["nonfinite_numeric_paths"] == []
        if row["dt_s"] == 0:
            assert not check["applicable"] and check["certified"] is None
            assert row["physical"]["trap_storage_error_A_m2"] == 0.
            assert check["dt_s"] == 0. and check["legacy_error_is_placeholder"]
            assert not row["solver_accepted"]
            assert row["physical_checks"]["checks"]["trap_storage"]["passed"] is None
            assert row["physical_checks"]["checks"]["charge_balance_normalized"]["passed"] is None
            assert row["physical_checks"]["checks"]["regular_right_limit_charge_balance_normalized"]["passed"]
            assert row["physical"]["initial_algebraic_certificate"] == initial["algebraic_certificate"]
        else:
            assert check["applicable"] and check["certified"]
            assert row["solver_accepted"]
            assert check["normalized_error"] <= check["normalized_limit"] == 1.
            assert check["charge_error_C_m2"] == row["physical"]["trap_storage_error_A_m2"] * row["dt_s"]
            assert check["normalized_error"] == max(check["newton_consistency_ratio"], check["local_charge_ratio"])
            assert check["dominant_budget"] in ("newton", "local_charge", "equal")
            assert check["trap_dynamics_active"] is (control in "CD")
            assert check["nonlinear_tolerances"] == {key: getattr(policy, key) for key in protocol._TOLERANCE_FIELDS}
            assert check["charge_scale_floor_A_m2"] == 1.
            assert check["charge_scale_floor_active"] is (check["charge_scale_before_floor_A_m2"] < 1.)
    assert result["certificate"]["finite_numeric_evidence"]["passed"]
    assert protocol.nonfinite_numeric_paths(result) == []


def test_zero_excitation_labels_equation_check_without_relative_current_claim(shared):
    stack, reference, policy, prepared = shared
    result = protocol.check_zero_excitation(stack, 16, reference, prepared, policy=policy)
    assert result["certificate"]["certified"]
    assert not result["certificate"]["relative_dynamic_current_certified"]
    assert set(result["controls"]) == set("ABCD")
    for row in result["controls"].values():
        assert row["remaining_equations"]["certified"]
        assert row["initial_event"]["impulse_charge_C_m2"] == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("substeps", [(2, 4, 8), (4, 8, 16)])
def test_independent_time_settings_execute_each_nested_level(shared, substeps):
    stack, reference, _, prepared = shared
    policy = protocol.r1_policy(time_substeps=substeps)
    result = protocol.run_r1_step(stack, 16, reference, prepared, control="D", policy=policy,
                                  times_s=np.array([0., 1e-9]))
    assert result["certificate"]["certified"]
    assert result["prepared_sha256"] == prepared.sha256
    assert result["execution_axes"] == {"intervals": 16, "time_substeps": list(substeps)}
    for count in substeps:
        rows = [row for row in result["accepted_steps"] if row["substeps"] == count]
        assert len(rows) == count + 1
        assert rows[0]["phase"] == "0+" and rows[-1]["time_s"] == 1e-9
        assert sum(row["dt_s"] for row in rows) == pytest.approx(1e-9)
        assert all(row["physical_checks_passed"] for row in rows)
    assert "no full-window or three-axis convergence claim" in result["scope_note"]


def test_failed_physical_step_retains_accepted_partial_evidence(shared, monkeypatch):
    stack, reference, policy, prepared = shared
    original = protocol.physical_step_record

    def over_limit(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[2] is not None:
            value["gauss_normalized"] = 1e-6
        return value

    monkeypatch.setattr(protocol, "physical_step_record", over_limit)
    with pytest.raises(protocol.R1RunError, match="gauss_normalized_exceeds_limit") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy)
    partial = error.value.result
    assert not partial["certificate"]["certified"]
    assert partial["prepared_sha256"] == prepared.sha256
    assert partial["initial_event"]["certified"]
    assert partial["accepted_steps"][-1]["physical"]["gauss_normalized"] == 1e-6
    assert partial["certificate"]["reasons"] == ["gauss_normalized_exceeds_limit"]


@pytest.mark.parametrize("injected", ["over_budget", "nan", "inf"])
def test_trap_storage_failure_retains_coarse_step_and_distinct_certificate(shared, monkeypatch, injected):
    stack, reference, policy, prepared = shared
    original = protocol._trap_storage_check
    observed = []

    def inject(scaling, state, previous, dt, active_policy, physical):
        check = original(scaling, state, previous, dt, active_policy, physical)
        if previous is not None:
            physical["trap_storage_error_A_m2"] = (
                np.nextafter(check["current_error_limit_A_m2"], np.inf)
                if injected == "over_budget" else float(injected)
            )
            return original(scaling, state, previous, dt, active_policy, physical)
        return check

    monkeypatch.setattr(protocol, "_trap_storage_check", inject)
    with pytest.raises(protocol.R1RunError, match="trap_storage_balance_exceeds_budget") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy,
                             accepted_step_observer=observed.append)
    partial = error.value.result
    assert partial["prepared_sha256"] == prepared.sha256
    assert partial["reference_sha256"] == reference["sha256"]
    assert partial["initial_event"]["certified"]
    assert partial["certificate"]["reasons"] == ["trap_storage_balance_exceeds_budget"]
    assert not partial["certificate"]["certified"]
    failed = partial["accepted_steps"][-1]
    assert failed["substeps"] == policy.refinement_substeps[0]
    assert failed["dt_s"] > 0 and failed["state"]["occupancy"]
    assert not failed["physical"]["trap_storage_check"]["certified"]
    assert failed["physical"]["gauss_normalized"] <= 1e-10
    assert failed["physical"]["charge_balance_normalized"] <= 1e-10
    # Every solver-accepted row reaches the observer with complete gate status.
    assert len(observed) == 2 and observed[0]["phase"] == "0+"
    assert observed[-1] is failed
    assert failed["solver_accepted"] and not failed["physical_checks_passed"]
    assert failed["physical_failure_reasons"] == ["trap_storage_balance_exceeds_budget"]
    if injected == "nan":
        assert np.isnan(failed["physical"]["trap_storage_error_A_m2"])
    elif injected == "inf":
        assert np.isinf(failed["physical"]["trap_storage_error_A_m2"])


@pytest.mark.parametrize("observer_fails", [False, True])
def test_all_physical_failures_reach_observer_and_survive_persistence_error(shared, monkeypatch, observer_fails):
    stack, reference, policy, prepared = shared
    original_physical = protocol.physical_step_record
    original_trap = protocol._trap_storage_check
    observed = []

    def several_failures(*args, **kwargs):
        physical = original_physical(*args, **kwargs)
        if args[2] is not None:
            for metric, limit in protocol._PHYSICAL_LIMITS:
                physical[metric] = limit * 2.
        return physical

    def trap_failure(scaling, state, previous, dt, active_policy, physical):
        check = original_trap(scaling, state, previous, dt, active_policy, physical)
        if previous is not None:
            physical["trap_storage_error_A_m2"] = check["current_error_limit_A_m2"] * 2.
            check = original_trap(scaling, state, previous, dt, active_policy, physical)
        return check

    def observer(row):
        observed.append(row)
        if row["solver_accepted"] and observer_fails:
            raise OSError("injected accepted-step write failure")

    monkeypatch.setattr(protocol, "physical_step_record", several_failures)
    monkeypatch.setattr(protocol, "_trap_storage_check", trap_failure)
    with pytest.raises(protocol.R1RunError, match="physical gates") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy,
                             accepted_step_observer=observer)
    partial = error.value.result
    expected = [metric + "_exceeds_limit" for metric, _ in protocol._PHYSICAL_LIMITS]
    expected.append("trap_storage_balance_exceeds_budget")
    assert partial["certificate"]["reasons"] == expected
    assert partial["failure"]["reasons"] == expected
    failed = partial["accepted_steps"][-1]
    assert len(observed) == 2 and observed[-1] is failed
    assert failed["solver_accepted"] and not failed["physical_checks_passed"]
    assert failed["physical_failure_reasons"] == expected
    assert failed["physical_checks"]["reasons"] == expected
    if observer_fails:
        assert partial["certificate"]["secondary_reasons"] == ["accepted_step_persistence_failed"]
        assert partial["persistence_failure"]["type"] == "OSError"
        assert partial["persistence_failure"]["message"] == "injected accepted-step write failure"
        assert partial["failure"]["type"] == "PhysicalCheckFailure"
    else:
        assert "persistence_failure" not in partial


def test_observer_only_failure_has_output_reason_and_retains_passed_physics(shared):
    stack, reference, policy, prepared = shared
    observed = []

    def observer(row):
        observed.append(row)
        if row["solver_accepted"]:
            raise OSError("injected output-only failure")

    with pytest.raises(protocol.R1RunError, match="accepted_step_persistence_failed") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy,
                             accepted_step_observer=observer)
    partial = error.value.result
    assert partial["certificate"]["reasons"] == ["accepted_step_persistence_failed"]
    assert partial["failure"]["type"] == "OSError"
    assert len(observed) == 2
    assert observed[-1]["physical_checks_passed"]
    assert observed[-1]["physical_failure_reasons"] == []
    assert observed[-1] is partial["accepted_steps"][-1]


@pytest.mark.parametrize("finite_step", [False, True])
@pytest.mark.parametrize("observer_fails", [False, True])
def test_nonfinite_interior_charge_is_physical_failure_before_digest(shared, monkeypatch, finite_step, observer_fails):
    stack, reference, policy, prepared = shared
    original = protocol.physical_step_record
    observed = []

    def inject(*args, **kwargs):
        physical = original(*args, **kwargs)
        if (args[2] is not None) is finite_step:
            physical["interior_charge_C_m2"] = np.nan
        return physical

    def observer(row):
        observed.append(row)
        if not row["physical_checks_passed"] and observer_fails:
            raise OSError("injected nonfinite-row write failure")

    monkeypatch.setattr(protocol, "physical_step_record", inject)
    with pytest.raises(protocol.R1RunError, match="physical gates: nonfinite_numeric_evidence") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy, accepted_step_observer=observer)
    partial = error.value.result
    failed = partial["accepted_steps"][-1]
    assert len(observed) == (2 if finite_step else 1)
    assert observed[-1] is failed
    assert failed["solver_accepted"] is finite_step
    assert failed["phase"] == ("accepted_regular_step" if finite_step else "0+")
    assert np.isnan(failed["physical"]["interior_charge_C_m2"])
    assert not failed["physical_checks_passed"]
    assert failed["physical_failure_reasons"] == ["nonfinite_numeric_evidence"]
    assert failed["physical_checks"]["nonfinite_numeric_paths"] == ["physical.interior_charge_C_m2"]
    assert partial["certificate"]["nonfinite_numeric_paths"] == ["physical.interior_charge_C_m2"]
    assert partial["certificate"]["reasons"] == ["nonfinite_numeric_evidence"]
    assert partial["failure"]["type"] == "PhysicalCheckFailure"
    assert partial["failure"]["nonfinite_numeric_paths"] == ["physical.interior_charge_C_m2"]
    assert not partial["certificate"]["certified"]
    assert "sha256" not in partial
    if observer_fails:
        assert partial["certificate"]["secondary_reasons"] == ["accepted_step_persistence_failed"]
        assert partial["persistence_failure"]["message"] == "injected nonfinite-row write failure"
    else:
        assert "persistence_failure" not in partial


@pytest.mark.parametrize("target", ["state", "physical_complex", "initial_current", "initial_charge"])
def test_nested_nonfinite_evidence_reports_full_path_on_initial_row(shared, monkeypatch, target):
    stack, reference, policy, prepared = shared
    observed = []
    if target == "state":
        original = protocol.snapshot

        def inject(*args, **kwargs):
            result = original(*args, **kwargs)
            result["n_m3"][0] = -np.inf
            return result

        monkeypatch.setattr(protocol, "snapshot", inject)
        expected = ["state.n_m3[0]"]
    elif target == "physical_complex":
        original = protocol.physical_step_record

        def inject(*args, **kwargs):
            result = original(*args, **kwargs)
            result["nested_current"] = {"vector": np.array([1 + 2j, complex(np.nan, np.inf)])}
            return result

        monkeypatch.setattr(protocol, "physical_step_record", inject)
        expected = ["physical.nested_current.vector[1].real", "physical.nested_current.vector[1].imag"]
    else:
        original = protocol.build_initial_step

        def inject(*args, **kwargs):
            result = original(*args, **kwargs)
            if target == "initial_current":
                result.event["regular_current"]["extra_contact_current"] = np.array([0., np.inf])
            else:
                result.event["impulse_charge_C_m2"] = np.nan
            return result

        monkeypatch.setattr(protocol, "build_initial_step", inject)
        expected = (["physical.regular_right_limit.extra_contact_current[1]",
                     "initial_event.regular_current.extra_contact_current[1]"]
                    if target == "initial_current" else ["initial_event.impulse_charge_C_m2"])
    with pytest.raises(protocol.R1RunError, match="nonfinite_numeric_evidence") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy, accepted_step_observer=observed.append)
    partial = error.value.result
    assert len(observed) == 1
    failed = observed[0]
    assert failed is partial["accepted_steps"][0]
    assert failed["phase"] == "0+" and not failed["physical_checks_passed"]
    assert failed["physical_checks"]["nonfinite_numeric_paths"] == expected
    assert partial["certificate"]["reasons"] == ["nonfinite_numeric_evidence"]
    assert partial["failure"]["type"] == "PhysicalCheckFailure"
    assert "sha256" not in partial


def test_final_regular_current_nonfinite_evidence_cannot_reach_digest(shared, monkeypatch):
    stack, reference, policy, prepared = shared
    original = protocol.regular_current_at_state
    observed = []

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        evidence = dict(result.evidence)
        evidence["extra_vector"] = np.array([complex(0., np.inf)])
        return SimpleNamespace(evidence=evidence)

    monkeypatch.setattr(protocol, "regular_current_at_state", inject)
    with pytest.raises(protocol.R1RunError, match="nonfinite_numeric_evidence") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy, accepted_step_observer=observed.append)
    partial = error.value.result
    expected = [f"regular_currents[{index}].extra_vector[0].imag" for index in range(5)]
    assert len(observed) == 31 and all(row["physical_checks_passed"] for row in observed)
    assert partial["certificate"]["reasons"] == ["nonfinite_numeric_evidence"]
    assert partial["certificate"]["nonfinite_numeric_paths"] == expected
    assert not partial["certificate"]["certified"]
    assert not partial["certificate"]["finite_numeric_evidence"]["passed"]
    assert partial["failure"]["type"] == "PhysicalCheckFailure"
    assert partial["failure"]["scope"] == "completed_result"
    assert np.isinf(partial["regular_currents"][0]["extra_vector"][0].imag)
    assert "sha256" not in partial


@pytest.mark.parametrize("field,value,suffix", [
    ("trap_storage_error_A_m2", np.nan, ""),
    ("contact_maxwell_A_m2", [np.inf, 0.], "[0]"),
    ("contact_internal_current_spread_relative", complex(0., -np.inf), ".imag"),
])
def test_initial_placeholder_replacement_cannot_hide_nonfinite_raw_evidence(shared, monkeypatch, field, value, suffix):
    stack, reference, policy, prepared = shared
    original = protocol.physical_step_record
    observed = []

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        result[field] = value
        return result

    monkeypatch.setattr(protocol, "physical_step_record", inject)
    with pytest.raises(protocol.R1RunError, match="nonfinite_numeric_evidence") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy, accepted_step_observer=observed.append)
    partial = error.value.result
    assert len(observed) == 1 and observed[0] is partial["accepted_steps"][0]
    failed = observed[0]
    assert failed["phase"] == "0+" and not failed["physical_checks_passed"]
    assert protocol.nonfinite_numeric_paths(failed["raw_initial_physical"]) == [field + suffix]
    assert failed["physical_checks"]["nonfinite_numeric_paths"] == ["raw_initial_physical." + field + suffix]
    assert partial["certificate"]["reasons"] == ["nonfinite_numeric_evidence"]
    assert partial["failure"]["type"] == "PhysicalCheckFailure"
    assert "sha256" not in partial


def test_zero_excitation_nonfinite_population_retains_raw_evidence(shared, monkeypatch):
    stack, reference, policy, prepared = shared
    original = protocol.snapshot

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        result["p_m3"][0] = np.nan
        return result

    monkeypatch.setattr(protocol, "snapshot", inject)
    with pytest.raises(protocol.R1RunError, match="nonfinite_numeric_evidence") as error:
        protocol.check_zero_excitation(stack, 16, reference, prepared, policy=policy)
    partial = error.value.result
    assert partial["certificate"]["reasons"] == ["nonfinite_numeric_evidence"]
    assert partial["failure"]["type"] == "PhysicalCheckFailure"
    assert partial["certificate"]["nonfinite_numeric_paths"] == [
        f"controls.{control}.population_identity.p_m3[0]" for control in "ABCD"
    ]
    assert np.isnan(partial["controls"]["A"]["population_identity"]["p_m3"][0])
    assert not partial["certificate"]["certified"]
    assert "sha256" not in partial


def test_initial_numeric_failure_retains_underlying_residuals(shared, monkeypatch):
    from perovskite_sim.experiments.interface_defect_ion_transient import InterfaceDefectIonTransientError
    stack, reference, policy, prepared = shared

    def fail(*args, **kwargs):
        error = InterfaceDefectIonTransientError("initial algebraic residual failed")
        error.result = {"poisson_residual_C_m2": np.array([2e-7]), "limit": 1e-14}
        raise error

    monkeypatch.setattr(protocol, "build_initial_step", fail)
    with pytest.raises(protocol.R1RunError) as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy)
    partial = error.value.result
    assert not partial["certificate"]["certified"]
    assert partial["failure"]["type"] == "InterfaceDefectIonTransientError"
    np.testing.assert_array_equal(partial["failure"]["numerical_evidence"]["poisson_residual_C_m2"], [2e-7])


def test_zero_check_failure_retains_completed_controls(shared, monkeypatch):
    from perovskite_sim.experiments.interface_defect_ion_transient import InterfaceDefectIonTransientError
    stack, reference, policy, prepared = shared
    original = protocol.build_initial_step
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            error = InterfaceDefectIonTransientError("second control failed")
            error.result = {"residual": 0.5}
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(protocol, "build_initial_step", fail_second)
    with pytest.raises(protocol.R1RunError) as error:
        protocol.check_zero_excitation(stack, 16, reference, prepared, policy=policy)
    partial = error.value.result
    assert not partial["certificate"]["certified"]
    assert set(partial["controls"]) == {"A"}
    assert partial["failure"]["numerical_evidence"] == {"residual": 0.5}


@pytest.mark.parametrize("amplitude", [0., .02, -.02, float("nan")])
def test_step_rejects_non_step_or_unsupported_amplitude(shared, amplitude):
    stack, reference, policy, prepared = shared
    with pytest.raises(ValueError, match="amplitude"):
        protocol.run_r1_step(stack, 16, reference, prepared, amplitude_V=amplitude, policy=policy)

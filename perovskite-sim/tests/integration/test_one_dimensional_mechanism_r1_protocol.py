"""Controlled common-state runs retain their initial charge and all steps."""

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
    assert result["version"] == "r1-1-controls-and-initial-charge-v2"
    assert result["certificate"]["trap_storage"]["checked_finite_step_count"] == 28
    assert result["certificate"]["metrics"]["trap_storage_normalized_error"] <= 1.
    for row in observed:
        check = row["physical"]["trap_storage_check"]
        if row["dt_s"] == 0:
            assert not check["applicable"] and check["certified"] is None
            assert "trap_storage_error_A_m2" not in row["physical"]
        else:
            assert check["applicable"] and check["certified"]
            assert check["normalized_error"] <= check["normalized_limit"] == 1.
            assert check["charge_error_C_m2"] == row["physical"]["trap_storage_error_A_m2"] * row["dt_s"]


def test_zero_excitation_labels_equation_check_without_relative_current_claim(shared):
    stack, reference, policy, prepared = shared
    result = protocol.check_zero_excitation(stack, 16, reference, prepared, policy=policy)
    assert result["certificate"]["certified"]
    assert not result["certificate"]["relative_dynamic_current_certified"]
    assert set(result["controls"]) == set("ABCD")
    for row in result["controls"].values():
        assert row["remaining_equations"]["certified"]
        assert row["initial_event"]["impulse_charge_C_m2"] == 0.0


def test_failed_physical_step_retains_accepted_partial_evidence(shared, monkeypatch):
    stack, reference, policy, prepared = shared
    original = protocol.physical_step_record

    def over_limit(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[2] is not None:
            value["gauss_normalized"] = 1e-6
        return value

    monkeypatch.setattr(protocol, "physical_step_record", over_limit)
    with pytest.raises(protocol.R1RunError, match="physical contact/charge") as error:
        protocol.run_r1_step(stack, 16, reference, prepared, policy=policy)
    partial = error.value.result
    assert not partial["certificate"]["certified"]
    assert partial["prepared_sha256"] == prepared.sha256
    assert partial["initial_event"]["certified"]
    assert partial["accepted_steps"][-1]["physical"]["gauss_normalized"] == 1e-6


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
    # The failure row remains in the result before an ordinary JSON observer
    # can mask the explicit error with an unrelated nonfinite serialization.
    assert len(observed) == 1 and observed[0]["phase"] == "0+"
    if injected == "nan":
        assert np.isnan(failed["physical"]["trap_storage_error_A_m2"])
    elif injected == "inf":
        assert np.isinf(failed["physical"]["trap_storage_error_A_m2"])


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

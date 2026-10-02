"""Deterministic checks of the single-interface storage charge budget."""

from dataclasses import dataclass, replace
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol


def budget_case(*, storage_scale=7.0, dt=1e-6):
    previous = SimpleNamespace(storage=np.arange(6, dtype=float))
    state = SimpleNamespace(conduction=np.array([.2, -.1]))
    calls = []

    def scale(storage, old, timestep, policy):
        assert storage is previous.storage
        assert old is previous
        calls.append((timestep, policy))
        return np.array([1., 2., 3., 4., storage_scale, 6.])

    system = SimpleNamespace(interior_count=2, interface_count=1, storage_scale=scale,
                             controls=SimpleNamespace(nu_t=1))
    physical = {"trap_storage_error_A_m2": 0., "charge_rate_A_m2": .3,
                "contact_conduction_A_m2": [.05, .02]}
    return system, state, previous, dt, protocol.r1_policy(), physical, calls


@pytest.mark.parametrize("side,accepted", [(-1, True), (0, True), (1, False)])
def test_trap_budget_boundary_is_deterministic(side, accepted):
    system, state, previous, dt, policy, physical, calls = budget_case()
    check = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
    limit = check["current_error_limit_A_m2"]
    error = limit if side == 0 else np.nextafter(limit, -np.inf if side < 0 else np.inf)
    physical["trap_storage_error_A_m2"] = error
    checked = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
    assert checked["certified"] is accepted
    assert checked["current_error_limit_A_m2"] == limit
    assert checked["charge_error_C_m2"] == error * dt
    assert checked["normalized_limit"] == 1.
    assert checked["newton_consistency_ratio"] == checked["normalized_error"]
    assert checked["local_charge_ratio"] == error / (checked["conservation_charge_budget_C_m2"] / dt)
    assert checked["dominant_budget"] == "newton"
    assert physical["trap_storage_error_A_m2"] == error
    assert len(calls) == 2
    if not accepted:
        assert checked["failure_reason"] == "trap_storage_balance_exceeds_budget"


def test_storage_budget_preserves_actual_scale_and_time_units():
    # The returned scale can include an existing rounding contribution. The
    # gate must use that whole value rather than reconstructing nominal atol.
    args = budget_case(storage_scale=7. + .25)
    system, state, previous, dt, policy, physical, calls = args
    first = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
    second = protocol._trap_storage_check(system, state, previous, dt*10, policy, physical)
    expected_charge_budget = Q * policy.maximum_scaled_nonlinear_residual * 7.25
    assert first["storage_scale_m2"] == 7.25
    assert first["nonlinear_tolerances"] == {
        key: getattr(policy, key) for key in protocol._TOLERANCE_FIELDS
    }
    assert first["trap_dynamics_active"]
    assert first["charge_scale_before_floor_A_m2"] == .3
    assert first["charge_scale_floor_A_m2"] == 1.
    assert first["charge_scale_floor_active"]
    assert first["nonlinear_charge_budget_C_m2"] == expected_charge_budget
    assert second["nonlinear_charge_budget_C_m2"] == expected_charge_budget
    assert first["charge_error_limit_C_m2"] == second["charge_error_limit_C_m2"]
    assert second["current_error_limit_A_m2"] == pytest.approx(first["current_error_limit_A_m2"] / 10)
    assert [call[0] for call in calls] == [dt, dt*10]


@pytest.mark.parametrize("policy_limit,expected", [(1e-3, 1e-10), (1e-10, 1e-10), (1e-12, 1e-12)])
def test_conservation_budget_retains_physical_cap_and_original_current_scale(policy_limit, expected):
    system, state, previous, dt, policy, physical, _ = budget_case(storage_scale=1e12)
    policy = replace(policy, maximum_charge_balance_relative_error=policy_limit)
    state.conduction = np.array([-4., 2.])
    physical["charge_rate_A_m2"] = 3.
    physical["contact_conduction_A_m2"] = [7., 2.]
    check = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
    assert check["charge_balance_scale_A_m2"] == 5.
    assert check["charge_balance_relative_limit"] == expected
    assert check["conservation_charge_budget_C_m2"] == dt * expected * 5.
    assert check["current_error_limit_A_m2"] == pytest.approx(expected * 5.)
    assert check["dominant_budget"] == "local_charge"
    assert not check["charge_scale_floor_active"]
    assert check["charge_scale_before_floor_A_m2"] == 5.


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -float("inf"), -1.])
def test_nonfinite_or_negative_reported_trap_error_rejected(error):
    system, state, previous, dt, policy, physical, _ = budget_case()
    physical["trap_storage_error_A_m2"] = error
    check = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
    assert not check["certified"]
    assert np.isinf(check["normalized_error"])
    assert check["failure_reason"] == "trap_storage_balance_exceeds_budget"


def test_initial_right_limit_has_no_finite_step_storage_certificate():
    check = protocol._trap_storage_check(None, None, None, 0., None, {})
    assert not check["applicable"] and check["certified"] is None
    assert check["dt_s"] == 0.
    assert check["reason"] == "no_positive_duration_time_step"
    assert check["legacy_error_is_placeholder"]
    assert "no finite-step error was computed" in check["legacy_error_note"]
    assert "normalized_error" not in check


@pytest.mark.parametrize("metric,limit", protocol._PHYSICAL_LIMITS)
@pytest.mark.parametrize("error", [float("nan"), float("inf"), -float("inf")])
def test_each_old_physical_gate_rejects_nonfinite_values(metric, limit, error):
    physical = {key: 0. for key, _ in protocol._PHYSICAL_LIMITS}
    physical[metric] = error
    physical["trap_storage_check"] = {
        "applicable": True, "normalized_error": 0., "certified": True,
        "failure_reason": None,
    }
    checks = protocol._physical_step_checks(physical, finite_step=True, policy=protocol.r1_policy())
    assert not checks["passed"]
    assert checks["reasons"] == [metric + "_nonfinite"]
    assert not checks["checks"][metric]["passed"]
    assert checks["checks"][metric]["limit"] == limit
    assert checks["nonfinite_numeric_paths"] == ["physical." + metric]
    assert not checks["checks"]["finite_numeric_evidence"]["passed"]


def test_numeric_evidence_walk_handles_nested_arrays_complex_and_na_without_mutation():
    @dataclass
    class Evidence:
        samples: object
        unavailable: object = None

    value = {
        "finite": [0., np.float32(1.), 10**1000, Decimal("2.5"), True, None, "N/A"],
        "nested": Evidence(np.array([[1 + 2j, complex(np.nan, np.inf)]])),
        "object_vector": np.array([{"charge": -np.inf}, None], dtype=object),
        "scalar_array": np.array(np.nan),
        "decimal": Decimal("Infinity"),
    }
    assert protocol.nonfinite_numeric_paths(value) == [
        "nested.samples[0][1].real", "nested.samples[0][1].imag",
        "object_vector[0].charge", "scalar_array", "decimal",
    ]
    assert np.isnan(value["nested"].samples[0, 1].real)
    assert np.isposinf(value["nested"].samples[0, 1].imag)
    assert value["nested"].unavailable is None
    assert protocol.nonfinite_numeric_paths(value["finite"]) == []


@pytest.mark.parametrize("value,paths", [
    (complex(np.nan, np.inf), ["physical.gauss_normalized.real", "physical.gauss_normalized.imag"]),
    (np.array([0., np.nan]), ["physical.gauss_normalized[1]"]),
])
def test_existing_scalar_gate_rejects_nonfinite_components_before_scalar_conversion(value, paths):
    physical = {key: 0. for key, _ in protocol._PHYSICAL_LIMITS}
    physical["gauss_normalized"] = value
    physical["trap_storage_check"] = {
        "applicable": True, "normalized_error": 0., "certified": True,
        "failure_reason": None,
    }
    checks = protocol._physical_step_checks(physical, finite_step=True, policy=protocol.r1_policy())
    assert checks["reasons"] == ["gauss_normalized_nonfinite"]
    assert checks["nonfinite_numeric_paths"] == paths
    assert not checks["passed"]
    assert not checks["checks"]["gauss_normalized"]["passed"]


def test_uncovered_numeric_evidence_aggregates_with_existing_physical_failure():
    physical = {key: 0. for key, _ in protocol._PHYSICAL_LIMITS}
    physical["gauss_normalized"] = 2e-10
    physical["interior_charge_C_m2"] = np.nan
    physical["current_vector"] = [0., -np.inf]
    physical["trap_storage_check"] = {
        "applicable": True, "normalized_error": 0., "certified": True,
        "failure_reason": None,
    }
    checks = protocol._physical_step_checks(physical, finite_step=True, policy=protocol.r1_policy())
    assert checks["reasons"] == ["gauss_normalized_exceeds_limit", "nonfinite_numeric_evidence"]
    assert not checks["passed"]
    assert checks["nonfinite_numeric_paths"] == [
        "physical.interior_charge_C_m2", "physical.current_vector[1]",
    ]


def test_completed_result_preserves_existing_metric_failure_without_duplicate_reason():
    result = {"certificate": {"certified": False, "reasons": ["nonlinear_residual"],
                              "metrics": {"nonlinear_residual": np.nan}}}
    with pytest.raises(protocol.R1RunError, match="physical gates") as error:
        protocol._require_finite_result(result)
    assert error.value.result is result
    assert result["certificate"]["reasons"] == ["nonlinear_residual"]
    assert result["certificate"]["nonfinite_numeric_paths"] == ["certificate.metrics.nonlinear_residual"]
    assert not result["certificate"]["finite_numeric_evidence"]["passed"]
    assert np.isnan(result["certificate"]["metrics"]["nonlinear_residual"])


def certificate_case(monkeypatch, errors_and_limits):
    state = SimpleNamespace(positive=np.array([1.]), negative=None,
                            conduction=np.array([1.]), carrier_conduction=np.array([1.]),
                            positive_current=np.array([0.]))
    level = SimpleNamespace(states=[state], maximum_nnz=1)
    for name in ("maximum_scaled_residual", "maximum_local_carrier_residual",
                 "maximum_local_gauss_residual", "maximum_jacobian_error",
                 "maximum_charge_balance_error", "maximum_face_spread",
                 "maximum_interface_current_error", "maximum_operator_error",
                 "maximum_charge_balance_absolute_error", "maximum_poisson_residual"):
        setattr(level, name, 0.)
    system = SimpleNamespace(ion_layout=SimpleNamespace(positive_components=[(0,)]),
                             widths=np.ones(1), positive_targets=np.ones(1), dimension=3,
                             controls=SimpleNamespace(nu_I=1, nu_t=1),
                             _site_fraction=lambda *args, **kwargs: .5)
    monkeypatch.setattr(protocol, "_refinement_changes", lambda *args: (0., 0.))
    records = []
    for error, limit in errors_and_limits:
        records.append({
            "gauss_normalized": 0., "charge_balance_normalized": 0.,
            "contact_internal_current_spread_relative": 0.,
            "trap_storage_error_A_m2": error,
            "trap_storage_check": {"applicable": True, "normalized_error": error/limit,
                                   "error_A_m2": error,
                                   "charge_error_C_m2": error * 1e-6,
                                   "current_error_limit_A_m2": limit,
                                   "newton_consistency_ratio": error/limit,
                                   "local_charge_ratio": error/(limit * 2),
                                   "dominant_budget": "newton",
                                   "trap_dynamics_active": True,
                                   "charge_scale_floor_active": True,
                                   "certified": bool(np.isfinite(error) and error <= limit)},
        })
    return protocol._trace_certificate(system, state, [level, level], protocol.r1_policy(), records)


def test_final_certificate_cannot_lose_coarse_only_violation(monkeypatch):
    # First record is the coarse level; every later (fine) check passes.
    certificate = certificate_case(monkeypatch, [(2e-14, 1e-14), (1e-16, 1e-14)])
    assert not certificate["certified"]
    assert certificate["reasons"] == ["trap_storage_balance_exceeds_budget"]
    assert certificate["metrics"]["trap_storage_normalized_error"] == 2.
    assert certificate["limits"]["trap_storage_normalized_error"] == 1.
    assert certificate["trap_storage"]["checked_finite_step_count"] == 2


def test_raw_error_maximum_is_not_compared_with_another_steps_limit(monkeypatch):
    certificate = certificate_case(monkeypatch, [(1e-8, 2e-8), (1e-20, 2e-20)])
    assert certificate["certified"]
    assert certificate["maximum_trap_storage_error_A_m2"] == 1e-8
    assert certificate["metrics"]["trap_storage_normalized_error"] == .5
    assert certificate["trap_storage"]["minimum_current_error_limit_A_m2"] == 2e-20


@pytest.mark.parametrize("error", [float("nan"), float("inf")])
def test_final_certificate_rejects_nonfinite_coarse_error(monkeypatch, error):
    certificate = certificate_case(monkeypatch, [(error, 1e-14), (0., 1e-14)])
    assert not certificate["certified"]
    assert "trap_storage_balance_exceeds_budget" in certificate["reasons"]


def test_summary_separates_budget_branches_and_nontrivial_active_trap_evidence():
    records = []
    for storage_scale, active, ratio in ((7., 0, 0.), (7., 1, .5), (1e12, 0, 0.), (1e12, 1, .25)):
        system, state, previous, dt, policy, physical, _ = budget_case(storage_scale=storage_scale)
        system.controls.nu_t = active
        check = protocol._trap_storage_check(system, state, previous, dt, policy, physical)
        physical["trap_storage_error_A_m2"] = ratio * check["current_error_limit_A_m2"]
        physical["trap_storage_check"] = protocol._trap_storage_check(
            system, state, previous, dt, policy, physical,
        )
        records.append(physical)
    summary = protocol._trap_storage_summary(records)
    assert summary["checked_finite_step_count"] == summary["charge_scale_floor_active_step_count"] == 4
    assert summary["branch_counts"]["all"] == {"newton": 2, "local_charge": 2, "equal": 0, "invalid": 0}
    for subset in ("active_trap", "nontrivial", "active_trap_nontrivial"):
        assert summary["branch_counts"][subset] == {"newton": 1, "local_charge": 1, "equal": 0, "invalid": 0}
    maxima = summary["branch_maxima"]["active_trap_nontrivial"]
    assert maxima["newton"]["maximum_newton_consistency_ratio"] == .5
    assert maxima["local_charge"]["maximum_local_charge_ratio"] == .25
    assert maxima["equal"]["maximum_normalized_error"] is None

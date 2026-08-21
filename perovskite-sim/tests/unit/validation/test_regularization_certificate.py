from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from perovskite_sim.physics.regularization import RHSRegularization
from perovskite_sim.validation.regularization_certificate import (
    AppliedRunContext,
    MetricSpec,
    MetricValue,
    ObservableSpec,
    QualityGateSpec,
    REGULARIZATION_LADDER_FACTORS,
    RegularizationCertificate,
    RegularizationCertificateError,
    RegularizationMeasurement,
    RegularizationRung,
    RegularizationStudy,
    SolverWork,
    build_regularization_certificate,
    policy_for_factor,
    run_regularization_ladder,
)


@pytest.fixture
def study() -> RegularizationStudy:
    return RegularizationStudy.from_values(
        base_policy=RHSRegularization(
            poole_frenkel_field_width_V_m=4.0e3,
            interface_density_width_m3=8.0e16,
            te_cap_relative_width=0.04,
        ),
        protocol={"schema_version": 1, "temperature_K": 300.0},
        config={"device": "regularization-test", "generation": True},
        grid={"kind": "tanh", "intervals": 60},
        tolerances={"rtol": 1.0e-6, "atol_density_m3": 1.0e10},
        observables=(ObservableSpec("current_density", "A m^-2", 1.0e-12),),
        residuals=(MetricSpec("nonlinear_residual", "s^-1", 10.0),),
        conservation_errors=(MetricSpec("current_spread", "A m^-2", 10.0),),
        physical_health_gates=(QualityGateSpec("inventory_error", "1", "le", 1.0e-6),),
        state_blocks=("n", "p"),
    )


def _measurement(
    study: RegularizationStudy,
    observable: float | list[float],
    *,
    factor: float = 0.25,
    applied_factor: float | None = None,
    protocol_sha256: str | None = None,
    config_sha256: str | None = None,
    grid_sha256: str | None = None,
    tolerances_sha256: str | None = None,
    residual: float = 1.0,
    conservation: float = 1.0,
    trial_minimum: float = 1.0,
    terminal_minimum: float = 1.0,
    health_value: float = 0.0,
    solver_accepted: bool = True,
    negative_trials: int = 0,
    nonfinite_events: int = 0,
    nfev: int = 100,
    wall_time_s: float = 1.0,
) -> RegularizationMeasurement:
    applied = AppliedRunContext(
        policy=policy_for_factor(
            study.base_policy,
            factor if applied_factor is None else applied_factor,
        ),
        protocol_sha256=protocol_sha256 or study.protocol.sha256,
        config_sha256=config_sha256 or study.config.sha256,
        grid_sha256=grid_sha256 or study.grid.sha256,
        tolerances_sha256=tolerances_sha256 or study.tolerances.sha256,
    )
    return RegularizationMeasurement.from_values(
        study,
        applied=applied,
        observables={"current_density": observable},
        residuals={"nonlinear_residual": residual},
        conservation_errors={"current_spread": conservation},
        minimum_trial_state_m3={"n": trial_minimum, "p": trial_minimum},
        terminal_minimum_state_m3={
            "n": terminal_minimum,
            "p": terminal_minimum,
        },
        physical_health={"inventory_error": health_value},
        solver_accepted=solver_accepted,
        negative_trial_count=negative_trials,
        nonfinite_event_count=nonfinite_events,
        nfev=nfev,
        njev=4,
        nlu=4,
        wall_time_s=wall_time_s,
    )


def _complete_rungs(
    study: RegularizationStudy,
    *,
    observables: tuple[float, float, float, float] = (100.8, 100.4, 100.1, 100.0),
    residuals: tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.0),
    conservation: tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.0),
) -> list[RegularizationRung]:
    return [
        RegularizationRung.completed(
            factor,
            _measurement(
                study,
                observable,
                factor=factor,
                residual=residual,
                conservation=conservation_error,
            ),
        )
        for factor, observable, residual, conservation_error in zip(
            REGULARIZATION_LADDER_FACTORS,
            observables,
            residuals,
            conservation,
            strict=True,
        )
    ]


def test_fixed_runner_uses_exact_policy_ladder_and_certifies(study):
    observed = []
    values = dict(zip(REGULARIZATION_LADDER_FACTORS, (100.8, 100.4, 100.1, 100.0)))

    def executor(request):
        observed.append(request)
        return _measurement(study, values[request.factor], factor=request.factor)

    certificate = run_regularization_ladder(study, executor)

    assert certificate.status == "certified"
    assert (
        tuple(request.factor for request in observed) == REGULARIZATION_LADDER_FACTORS
    )
    assert tuple(request.policy for request in observed) == (
        study.base_policy,
        study.base_policy.refined(0.5),
        study.base_policy.refined(0.25),
        RHSRegularization(),
    )
    assert not observed[-1].policy.active
    assert all(rung.measurement.work.wall_time_s >= 0.0 for rung in certificate.rungs)


def test_runner_rejects_executor_that_reports_zero_policy_for_every_rung(study):
    values = dict(zip(REGULARIZATION_LADDER_FACTORS, (100.8, 100.4, 100.1, 100.0)))

    def executor(request):
        return _measurement(
            study,
            values[request.factor],
            factor=request.factor,
            applied_factor=0.0,
        )

    certificate = run_regularization_ladder(study, executor)

    assert certificate.status == "failed"
    policy_check = next(
        check for check in certificate.checks if check.name == "fixed_width_ladder"
    )
    assert not policy_check.passed


def test_runner_rejects_actual_context_hash_drift(study):
    values = dict(zip(REGULARIZATION_LADDER_FACTORS, (100.8, 100.4, 100.1, 100.0)))

    def executor(request):
        return _measurement(
            study,
            values[request.factor],
            factor=request.factor,
            protocol_sha256="0" * 64 if request.factor == 0.25 else None,
        )

    certificate = run_regularization_ladder(study, executor)

    assert certificate.status == "failed"
    context_check = next(
        check for check in certificate.checks if check.name == "fixed_context"
    )
    assert not context_check.passed


def test_every_rung_records_full_fixed_context_and_solver_work(study):
    certificate = build_regularization_certificate(study, _complete_rungs(study))

    for factor, rung in zip(
        REGULARIZATION_LADDER_FACTORS, certificate.rungs, strict=True
    ):
        assert rung.policy == policy_for_factor(study.base_policy, factor)
        assert rung.protocol_sha256 == study.protocol.sha256
        assert rung.config_sha256 == study.config.sha256
        assert rung.grid_sha256 == study.grid.sha256
        assert rung.tolerances_sha256 == study.tolerances.sha256
        assert rung.measurement.work == SolverWork(100, 4, 4, 1.0)


def test_missing_rung_is_materialized_and_fails(study):
    rungs = _complete_rungs(study)
    del rungs[2]

    certificate = build_regularization_certificate(study, rungs)

    assert certificate.status == "failed"
    assert certificate.rungs[2].outcome == "missing"
    assert not certificate.checks[0].passed


def test_failed_rung_fails_even_when_other_rungs_converge(study):
    rungs = _complete_rungs(study)
    rungs[2] = RegularizationRung.failed(
        study,
        0.25,
        "Radau failed",
        work=SolverWork(900, 12, 12, 2.0),
    )

    certificate = build_regularization_certificate(study, rungs)

    assert certificate.status == "failed"
    assert certificate.rungs[2].failed_work.nfev == 900


def test_executor_exception_is_retained_as_failed_rung(study):
    def executor(request):
        if request.factor == 0.25:
            raise RuntimeError("solver stopped")
        values = {1.0: 100.8, 0.5: 100.4, 0.0: 100.0}
        return _measurement(study, values[request.factor], factor=request.factor)

    certificate = run_regularization_ladder(study, executor)

    assert certificate.status == "failed"
    assert certificate.rungs[2].outcome == "failed"
    assert certificate.rungs[2].failure_reason == "RuntimeError: solver stopped"
    assert certificate.rungs[2].failed_work.wall_time_s >= 0.0


@pytest.mark.parametrize(
    "rung_update",
    [
        {"protocol_sha256": "0" * 64},
        {"config_sha256": "1" * 64},
        {"grid_sha256": "2" * 64},
        {"tolerances_sha256": "3" * 64},
        {"policy": RHSRegularization(poole_frenkel_field_width_V_m=1.0)},
    ],
)
def test_context_or_policy_drift_fails(study, rung_update):
    rungs = _complete_rungs(study)
    rungs[1] = replace(rungs[1], **rung_update)

    assert build_regularization_certificate(study, rungs).status == "failed"


@pytest.mark.parametrize(
    ("observables", "residuals", "conservation"),
    [
        ((101.0, 100.0, 99.5, 99.4), (4.0, 3.0, 2.0, 5.0), (4.0, 3.0, 2.0, 5.0)),
        ((100.8, 100.4, 100.1, 100.0), (4.0, 2.0, 2.1, 5.0), (4.0, 3.0, 2.0, 5.0)),
        ((100.8, 100.4, 100.1, 100.0), (4.0, 3.0, 2.0, 5.0), (4.0, 2.0, 2.1, 5.0)),
        ((100.1, 100.4, 100.2, 100.0), (4.0, 3.0, 2.0, 5.0), (4.0, 3.0, 2.0, 5.0)),
        ((103.0, 101.0, 100.6, 100.0), (4.0, 3.0, 2.0, 5.0), (4.0, 3.0, 2.0, 5.0)),
    ],
    ids=(
        "exactly-0.5-percent",
        "residual-worsens",
        "conservation-worsens",
        "zero-trend-reverses",
        "quarter-to-zero-does-not-close",
    ),
)
def test_complete_healthy_nonconverged_ladder_is_partial(
    study, observables, residuals, conservation
):
    certificate = build_regularization_certificate(
        study,
        _complete_rungs(
            study,
            observables=observables,
            residuals=residuals,
            conservation=conservation,
        ),
    )

    assert certificate.status == "partial"


def test_zero_width_residual_worsening_is_partial(study):
    certificate = build_regularization_certificate(
        study,
        _complete_rungs(study, residuals=(4.0, 3.0, 2.0, 3.0)),
    )

    assert certificate.status == "partial"
    residual = next(
        check for check in certificate.checks if check.name == "residual_non_worsening"
    )
    assert not residual.passed


def test_speedup_alone_cannot_certify(study):
    rungs = _complete_rungs(
        study,
        observables=(101.0, 100.0, 99.5, 99.4),
    )
    rungs[1] = replace(
        rungs[1],
        measurement=replace(
            rungs[1].measurement,
            work=SolverWork(10_000, 100, 100, 10.0),
        ),
    )
    rungs[2] = replace(
        rungs[2],
        measurement=replace(
            rungs[2].measurement,
            work=SolverWork(1, 0, 0, 0.001),
        ),
    )

    assert build_regularization_certificate(study, rungs).status == "partial"


def test_slowdown_does_not_block_an_otherwise_certified_ladder(study):
    rungs = _complete_rungs(study)
    rungs[2] = replace(
        rungs[2],
        measurement=replace(
            rungs[2].measurement,
            work=SolverWork(100_000, 1_000, 1_000, 100.0),
        ),
    )

    assert build_regularization_certificate(study, rungs).status == "certified"


def test_absolute_residual_or_conservation_failure_is_failed(study):
    rungs = _complete_rungs(study)
    rungs[-1] = replace(
        rungs[-1],
        measurement=replace(
            rungs[-1].measurement,
            residuals=(
                MetricValue.from_value("nonlinear_residual", 1.0e300, units="s^-1"),
            ),
            conservation_errors=(
                MetricValue.from_value("current_spread", 1.0e300, units="A m^-2"),
            ),
        ),
    )

    certificate = build_regularization_certificate(study, rungs)

    assert certificate.status == "failed"
    absolute = next(
        check
        for check in certificate.checks
        if check.name == "absolute_residual_conservation"
    )
    assert not absolute.passed


def test_absolute_gate_boundaries_are_inclusive(study):
    rungs = _complete_rungs(
        study,
        residuals=(10.0, 10.0, 10.0, 10.0),
        conservation=(10.0, 10.0, 10.0, 10.0),
    )
    rungs = [
        replace(
            rung,
            measurement=replace(
                rung.measurement,
                physical_health=(
                    MetricValue.from_value("inventory_error", 1.0e-6, units="1"),
                ),
            ),
        )
        for rung in rungs
    ]

    assert build_regularization_certificate(study, rungs).status == "certified"


@pytest.mark.parametrize(
    "case",
    ("solver-rejected", "nonfinite", "nonpositive-terminal", "absolute-health-gate"),
)
def test_physical_health_failure_is_failed(study, case):
    if case == "solver-rejected":
        measurement_update = {"solver_accepted": False}
    elif case == "nonfinite":
        measurement_update = {"nonfinite_event_count": 1}
    elif case == "nonpositive-terminal":
        measurement_update = {
            "terminal_minimum_state_m3": (
                MetricValue.from_value("n", 0.0, units="m^-3"),
                MetricValue.from_value("p", 1.0, units="m^-3"),
            )
        }
    else:
        measurement_update = {
            "physical_health": (
                MetricValue.from_value("inventory_error", 2.0e-6, units="1"),
            )
        }
    rungs = _complete_rungs(study)
    rungs[2] = replace(
        rungs[2],
        measurement=replace(rungs[2].measurement, **measurement_update),
    )

    assert build_regularization_certificate(study, rungs).status == "failed"


def test_negative_intermediate_trials_are_diagnostic_not_a_failure(study):
    rungs = _complete_rungs(study)
    rungs[2] = replace(
        rungs[2],
        measurement=replace(
            rungs[2].measurement,
            minimum_trial_state_m3=(
                MetricValue.from_value("n", -2.0, units="m^-3"),
                MetricValue.from_value("p", -1.0, units="m^-3"),
            ),
            negative_trial_count=7,
        ),
    )

    certificate = build_regularization_certificate(study, rungs)

    assert certificate.status == "certified"
    assert certificate.rungs[2].measurement.negative_trial_count == 7


def test_missing_pre_registered_metric_fails_closed(study):
    rungs = _complete_rungs(study)
    rungs[1] = replace(
        rungs[1],
        measurement=replace(rungs[1].measurement, residuals=()),
    )

    certificate = build_regularization_certificate(study, rungs)

    assert certificate.status == "failed"
    evidence = next(
        check for check in certificate.checks if check.name == "evidence_complete"
    )
    assert not evidence.passed


def test_missing_pre_registered_state_block_fails_closed(study):
    rungs = _complete_rungs(study)
    rungs[1] = replace(
        rungs[1],
        measurement=replace(
            rungs[1].measurement,
            minimum_trial_state_m3=(MetricValue.from_value("n", 1.0, units="m^-3"),),
            terminal_minimum_state_m3=(MetricValue.from_value("n", 1.0, units="m^-3"),),
        ),
    )

    assert build_regularization_certificate(study, rungs).status == "failed"


def test_array_observable_shape_change_fails_closed(study):
    rungs = _complete_rungs(study)
    rungs[0] = replace(
        rungs[0],
        measurement=replace(
            rungs[0].measurement,
            observables=(
                MetricValue.from_value(
                    "current_density", np.array([100.8, 100.7]), units="A m^-2"
                ),
            ),
        ),
    )

    assert build_regularization_certificate(study, rungs).status == "failed"


def test_shuffled_rungs_have_same_canonical_certificate(study):
    ordered = _complete_rungs(study)
    shuffled = [ordered[2], ordered[0], ordered[3], ordered[1]]

    first = build_regularization_certificate(study, ordered)
    second = build_regularization_certificate(study, shuffled)

    assert first.canonical_json() == second.canonical_json()
    assert first.certificate_sha256 == second.certificate_sha256


def test_certificate_round_trip_recomputes_checks_and_hash(study):
    certificate = build_regularization_certificate(study, _complete_rungs(study))
    raw = json.loads(certificate.canonical_json())

    restored = RegularizationCertificate.from_dict(raw)

    assert restored.canonical_json() == certificate.canonical_json()
    raw["status"] = "partial"
    with pytest.raises(RegularizationCertificateError, match="status"):
        RegularizationCertificate.from_dict(raw)


def test_direct_constructor_cannot_self_report_certified(study):
    certificate = build_regularization_certificate(study, _complete_rungs(study))

    with pytest.raises(RegularizationCertificateError, match="status"):
        replace(certificate, status="partial", certificate_sha256="")


def test_mapping_order_does_not_change_study_hash(study):
    reordered = RegularizationStudy.from_values(
        base_policy=study.base_policy,
        protocol={"temperature_K": 300.0, "schema_version": 1},
        config={"generation": True, "device": "regularization-test"},
        grid={"intervals": 60, "kind": "tanh"},
        tolerances={"atol_density_m3": 1.0e10, "rtol": 1.0e-6},
        observables=study.observables,
        residuals=study.residuals,
        conservation_errors=study.conservation_errors,
        physical_health_gates=study.physical_health_gates,
        state_blocks=tuple(reversed(study.state_blocks)),
    )

    assert reordered.definition_sha256 == study.definition_sha256


def test_measurement_set_order_and_signed_zero_do_not_change_hash(study):
    ordered = _complete_rungs(study)
    reordered = list(ordered)
    zero = reordered[-1]
    reordered[-1] = replace(
        zero,
        factor=-0.0,
        measurement=replace(
            zero.measurement,
            minimum_trial_state_m3=tuple(
                reversed(zero.measurement.minimum_trial_state_m3)
            ),
            terminal_minimum_state_m3=tuple(
                reversed(zero.measurement.terminal_minimum_state_m3)
            ),
        ),
    )

    first = build_regularization_certificate(study, ordered)
    second = build_regularization_certificate(study, reordered)

    assert first.certificate_sha256 == second.certificate_sha256


def test_gate_threshold_changes_study_definition_hash(study):
    changed = replace(
        study,
        physical_health_gates=(replace(study.physical_health_gates[0], limit=2.0e-6),),
    )

    assert changed.definition_sha256 != study.definition_sha256


@pytest.mark.parametrize(
    "base_policy",
    [
        RHSRegularization(),
        RHSRegularization(poole_frenkel_field_width_V_m=np.nextafter(0.0, 1.0)),
    ],
    ids=("all-zero", "collapsed-positive-rungs"),
)
def test_study_rejects_degenerate_width_ladders(study, base_policy):
    with pytest.raises(RegularizationCertificateError):
        replace(study, base_policy=base_policy)


def test_study_rejects_dummy_or_incomplete_state_blocks(study):
    with pytest.raises(RegularizationCertificateError, match="known solver blocks"):
        replace(study, state_blocks=("dummy",))


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, True, 1.0 + 2.0j, "not-a-number"]
)
def test_measurement_rejects_invalid_numeric_evidence(study, bad_value):
    with pytest.raises(RegularizationCertificateError):
        _measurement(study, bad_value)


def test_negative_error_metric_fails_closed(study):
    rungs = _complete_rungs(study)
    rungs[1] = replace(
        rungs[1],
        measurement=replace(
            rungs[1].measurement,
            residuals=(
                MetricValue.from_value("nonlinear_residual", -1.0, units="s^-1"),
            ),
        ),
    )

    assert build_regularization_certificate(study, rungs).status == "failed"


def test_duplicate_rung_is_rejected(study):
    rungs = _complete_rungs(study)
    with pytest.raises(RegularizationCertificateError, match="duplicate"):
        build_regularization_certificate(study, [*rungs, rungs[0]])

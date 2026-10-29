"""Device-free BDF moments, physical storage, orders and event semantics."""

from dataclasses import dataclass

import numpy as np
import pytest

from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.problem import GateOutcome, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.time.bdf import (
    Assessment, BDFEvent, BDFOptions, BDFStep, HistoryPoint, bdf_coefficients, solve_bdf,
)


@dataclass
class RC:
    logarithmic: bool = False

    def residual(self, u):
        return -self.storage(u)

    def jacobian(self, u):
        return -self.storage_tangent(u)

    def storage(self, u):
        return np.exp(u) if self.logarithmic else np.array(u, copy=True)

    def storage_tangent(self, u):
        return np.diag(np.exp(u)) if self.logarithmic else np.eye(len(u))


def controls(step):
    n = len(step.history[0].u)
    return NewtonSettings(ResidualScaling(np.ones(n)), 1e-12, StepControl(np.full(n, 1e30)),
                          SolveBudget(20, 100, 20, 10), (), False)


def fixed(n, order, logarithmic=False, output=None):
    problem = RC(logarithmic)
    return solve_bdf(lambda t, side: problem, np.array([0.0 if logarithmic else 1.0]), (0.0, 1.0),
        options=BDFOptions(1 / n, max_step_s=1 / n, max_order=order, adaptive=False),
        settings=controls, output_times_s=output)


@pytest.mark.parametrize("ratio", [.25, .7, 1.0, 2.0])
def test_variable_step_coefficients_differentiate_degree_two_exactly(ratio):
    h = .2
    previous = h / ratio
    nodes = np.array([1.0, 1.0 - h, 1.0 - h - previous])
    coefficients = bdf_coefficients(h, previous, 2)
    for degree in (0, 1, 2):
        assert coefficients @ nodes**degree / h == pytest.approx(float(degree), abs=2e-14)


def test_residual_differences_physical_storage_and_full_tangent():
    p = RC(logarithmic=True)
    old = HistoryPoint(0.0, np.array([.7]), p.storage(np.array([.7])), p)
    step = BDFStep(p, (old,), .1, 1)
    u = np.array([.8])
    expected = -np.exp(u) - (np.exp(u) - np.exp(old.u)) / .1
    np.testing.assert_array_equal(step.residual(u), expected)
    np.testing.assert_array_equal(step.jacobian(u), np.diag(-np.exp(u) * 11))
    assert abs(step.residual(u)[0] - (-np.exp(u)[0] - (u[0] - old.u[0]) / .1)) > 1.0


@pytest.mark.parametrize("order", [1, 2])
def test_observed_rc_order_and_logarithmic_physical_storage(order):
    ns = np.array([16, 32, 64])
    errors = []
    for n in ns:
        result = fixed(int(n), order)
        assert result.completed, (result.termination, result.error)
        errors.append(abs(result.accepted_coordinates[-1][0] - np.exp(-1)))
    observed = np.polyfit(np.log(1 / ns), np.log(errors), 1)[0]
    assert abs(observed - order) < .15, (observed, errors)
    transformed = fixed(32, order, logarithmic=True)
    assert transformed.completed
    plain = fixed(32, order)
    assert np.exp(transformed.accepted_coordinates[-1][0]) == pytest.approx(plain.accepted_coordinates[-1][0], rel=2e-12)


def test_error_rejection_and_adaptive_order_selection_preserve_accepted_prefix():
    p = RC()
    result = solve_bdf(lambda t, side: p, np.array([1.0]), (0.0, 1.0),
        options=BDFOptions(.7, relative_tolerance=1e-4), settings=controls)
    assert result.completed, (result.termination, result.error)
    assert result.rejected_steps and {step.order for step in result.accepted_steps} == {1, 2}
    assert len(result.accepted_times_s) == len(result.accepted_steps) + 1
    assert all(step.error_norm <= 1 for step in result.accepted_steps)
    assert result.accepted_coordinates[-1][0] == pytest.approx(np.exp(-1), rel=2e-3)
    for rejected in result.rejected_steps:
        assert rejected.candidate_coordinates is not result.accepted_coordinates[0]


def test_samples_are_storage_interpolants_and_do_not_restart_the_trajectory():
    endpoints = fixed(16, 2)
    dense = fixed(16, 2, output=np.array([0.0, .13, .31, .77, 1.0]))
    assert dense.completed, (dense.termination, dense.error)
    np.testing.assert_array_equal(dense.accepted_times_s, endpoints.accepted_times_s)
    for a, b in zip(dense.accepted_coordinates, endpoints.accepted_coordinates):
        np.testing.assert_array_equal(a, b)
    assert len(dense.samples) == 5 and len(dense.accepted_steps) == 16
    assert all(abs(sample.u[0] - np.exp(-sample.time_s)) < .003 for sample in dense.samples)
    assert not dense.restarts
    assert dense.evaluation_counts["sample_residual"] == sum(
        sample.projection.residual_evaluations for sample in dense.samples if sample.projection is not None)
    assert dense.evaluation_counts["sample_jacobian"] == sum(
        sample.projection.jacobian_evaluations for sample in dense.samples if sample.projection is not None)


def test_nonlinear_storage_dense_samples_evaluate_projection_jacobian():
    output = np.array([0.0, .13, .31, 1.0])
    dense = fixed(8, 2, logarithmic=True, output=output)
    assert dense.completed, (dense.termination, dense.error)
    assert len(dense.accepted_steps) == 8 and len(dense.samples) == 4
    projected = [sample for sample in dense.samples if not sample.accepted_step_endpoint]
    assert all(sample.projection is not None and sample.projection.jacobian_evaluations > 0
               for sample in projected)
    plain = fixed(8, 2, output=output)
    assert plain.completed
    for transformed, physical in zip(dense.samples, plain.samples):
        np.testing.assert_allclose(transformed.storage_value, physical.storage_value, rtol=2e-12, atol=0)
    projections = [sample.projection for sample in projected]
    main = [result for attempt in dense.attempts
            for result in (attempt.newton, *attempt.auxiliary_newton) if result is not None]
    for key, attribute in (("residual", "residual_evaluations"), ("jacobian", "jacobian_evaluations")):
        assert dense.evaluation_counts["sample_" + key] == sum(getattr(result, attribute) for result in projections)
        assert dense.evaluation_counts[key] == sum(getattr(result, attribute) for result in (*main, *projections))


def test_voltage_discontinuity_has_exact_left_right_state_and_order_restart():
    class Driven:
        def __init__(self, voltage):
            self.voltage = voltage
        def residual(self, u):
            return np.array([self.voltage - u[0], u[1] - self.voltage])
        def jacobian(self, u):
            return np.diag([-1.0, 1.0])
        def storage(self, u):
            return np.array([u[0], 0.0])
        def storage_tangent(self, u):
            return np.diag([1.0, 0.0])
    def factory(t, side):
        return Driven(0.0 if t < .5 or (t == .5 and side == "left") else 1.0)
    def restart(old, new, u):
        value = u.copy()
        value[1] = new.voltage
        return value, Assessment({"electrode_charge_impulse": new.voltage - old.voltage},
            (GateOutcome("frozen_inventory", value[0] == u[0], 0.0, 0.0),))
    result = solve_bdf(factory, np.array([0.0, 0.0]), (0.0, 1.0),
        options=BDFOptions(.125, max_step_s=.125, adaptive=False), settings=controls,
        events=(BDFEvent(.5, restart, "voltage_step"),), output_times_s=np.array([0, .5, 1]))
    assert result.completed, (result.termination, result.error)
    event = result.restarts[0]
    assert event.time_s == .5 and event.left_u[1] == 0 and event.right_u[1] == 1
    np.testing.assert_array_equal(event.left_storage, event.right_storage)
    assert event.assessment.accounting["electrode_charge_impulse"] == 1
    first_after = next(step for step in result.accepted_steps if step.start_s == .5)
    assert first_after.order == 1
    assert result.samples[1].u[1] == 1


def test_physical_gate_failure_is_recorded_and_never_retried_as_lte():
    p = RC()
    result = solve_bdf(lambda t, side: p, np.array([1.0]), (0.0, 1.0),
        options=BDFOptions(.1), settings=controls,
        assess=lambda step, u: Assessment({}, (GateOutcome("injected_physical_refusal", False, 2.0, 1.0),)))
    assert not result.completed and result.termination == "physical_gate_rejected"
    assert len(result.attempts) == 1 and len(result.accepted_times_s) == 1


@pytest.mark.parametrize("order", [1, 2])
def test_sample_interpolant_has_the_integrator_convergence_order(order):
    ns = np.array([16, 32, 64])
    output = np.array([.13, .31, .77])
    errors = []
    for n in ns:
        result = fixed(int(n), order, output=output)
        assert result.completed
        errors.append(max(abs(sample.storage_value[0] - np.exp(-sample.time_s)) for sample in result.samples))
    observed = np.polyfit(np.log(1 / ns), np.log(errors), 1)[0]
    assert abs(observed - order) < .15, (observed, errors)


def test_failed_sample_preserves_requested_samples_and_complete_step_history():
    class BadSample(RC):
        def jacobian(self, u):
            return np.array([[np.nan]])
        def storage_tangent(self, u):
            return np.array([[np.nan]])
    good = RC(logarithmic=True)
    result = solve_bdf(lambda t, side: BadSample(logarithmic=True) if t == .31 else good, np.array([0.0]), (0, 1),
        options=BDFOptions(.125, adaptive=False), settings=controls, output_times_s=np.array([0, .13, .31, 1]))
    assert not result.completed and result.termination == "sample_projection_failed"
    assert result.accepted_times_s[-1] == 1 and len(result.accepted_steps) == 8
    np.testing.assert_array_equal(result.requested_output_times_s, [0, .13, .31, 1])
    assert result.sample_failure is not None and result.sample_failure["time_s"] == .31
    assert len(result.samples) == 2
    assert result.samples[1].projection is not None and result.samples[1].projection.jacobian_evaluations > 0
    assert result.sample_failure["newton"] is None

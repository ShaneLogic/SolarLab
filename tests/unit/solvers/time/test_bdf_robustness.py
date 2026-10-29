"""N-method contracts: algebraic words and consistent slope-knot restarts."""

import numpy as np

from examples.research_1d.transient import controls
from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.problem import ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.time.bdf import Assessment, BDFEvent, BDFOptions, BDFStep, HistoryPoint, solve_bdf
from solarlab.solvers.time.bdf_1d import BDFDrive, _events


def test_algebraic_residual_signed_zero_is_preserved():
    class Problem:
        def residual(self, u):
            return np.array([-u[0], u[1]])
        def jacobian(self, u):
            return np.diag([-1.0, 1.0])
        def storage(self, u):
            return np.array([u[0], 0.0])
        def storage_tangent(self, u):
            return np.diag([1.0, 0.0])
    problem = Problem()
    old_u, u = np.array([1.0, -0.0]), np.array([.9, -0.0])
    old = HistoryPoint(0.0, old_u, problem.storage(old_u), problem)
    result = BDFStep(problem, (old,), .1, 1).residual(u)
    assert result[1:].tobytes() == problem.residual(u)[1:].tobytes()
    assert np.signbit(result[1])


def test_nonzero_history_rate_is_retained_at_a_zero_storage_tangent():
    class Degenerate:
        def residual(self, u):
            return -np.ones(1)
        def jacobian(self, u):
            return np.zeros((1, 1))
        def storage(self, u):
            return u**3
        def storage_tangent(self, u):
            return np.diag(3 * u**2)
    problem = Degenerate()
    old_u, u = np.ones(1), np.zeros(1)
    history = HistoryPoint(0.0, old_u, problem.storage(old_u), problem)
    step = BDFStep(problem, (history,), .1, 1)
    np.testing.assert_array_equal(step.residual(u), [9.0])


def test_slope_knots_are_explicit_events_without_duplicate_jump_events():
    values = np.array([[0.0, 0.0], [.5, 0.0], [1.5, 0.0]])
    drive = BDFDrive(np.array([0.0, .5, 1.0]), values, values)
    events = _events(drive, None, controls())
    assert drive.slope_breakpoints_s == (.5,)
    assert [(e.time_s, e.name) for e in events] == [(.5, "drive_slope_discontinuity")]
    linear = np.array([[0.0, 0.0], [.5, 0.0], [1.0, 0.0]])
    assert not BDFDrive(drive.times_s, linear, linear).slope_breakpoints_s
    right = values.copy()
    right[1, 0] += .1
    jumped = BDFDrive(drive.times_s, values, right)
    assert not jumped.slope_breakpoints_s
    assert [e.name for e in _events(jumped, None, controls())] == ["voltage_generation_protocol_jump"]


def test_generic_restart_at_a_continuous_slope_knot_keeps_state_and_order_one():
    values = np.array([[0.0, 0.0], [.5, 0.0], [1.5, 0.0]])
    drive = BDFDrive(np.array([0.0, .5, 1.0]), values, values)
    class Driven:
        def __init__(self, value):
            self.value = value
        def residual(self, u):
            return self.value - u
        def jacobian(self, u):
            return -np.eye(1)
        def storage(self, u):
            return u.copy()
        def storage_tangent(self, u):
            return np.eye(1)
    def settings(step):
        return NewtonSettings(ResidualScaling(np.ones(1)), 1e-12,
            StepControl(np.full(1, 1e30)), SolveBudget(20, 100, 20, 10), (), False)
    def restart(old, new, u):
        return u.copy(), Assessment()
    result = solve_bdf(lambda t, side: Driven(drive.values(t, side)[0]), np.zeros(1), (0.0, 1.0),
        options=BDFOptions(.125, max_step_s=.125, adaptive=False), settings=settings,
        events=(BDFEvent(.5, restart, "drive_slope_discontinuity"),),
        output_times_s=np.array([0.0, .5, 1.0]))
    assert result.completed
    event = result.restarts[0]
    assert event.left_u.tobytes() == event.right_u.tobytes()
    assert event.left_storage.tobytes() == event.right_storage.tobytes()
    assert next(a for a in result.accepted_steps if a.start_s == .5).order == 1

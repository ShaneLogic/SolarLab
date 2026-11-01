"""Accepted-coordinate records use the same jump side as history and samples."""

import numpy as np
import pytest

from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.problem import GateOutcome, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.time.bdf import Assessment, BDFEvent, BDFOptions, solve_bdf


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


def settings(step):
    return NewtonSettings(ResidualScaling(np.ones(2)), 1e-12,
        StepControl(np.full(2, 1e30)), SolveBudget(20, 100, 20, 10), (), False)


@pytest.mark.parametrize("event_time", [0.0, .5])
def test_accepted_coordinate_at_value_jump_is_the_right_state_without_changing_left_record(event_time):
    def factory(t, side):
        return Driven(0.0 if t < event_time or (t == event_time and side == "left") else 1.0)

    def restart(old, new, u):
        right = u.copy()
        right[1] = new.voltage
        return right, Assessment({}, (GateOutcome("frozen_inventory", right[0] == u[0], 0.0, 0.0),))

    result = solve_bdf(factory, np.zeros(2), (0, 1), options=BDFOptions(.125, adaptive=False),
        settings=settings, events=(BDFEvent(event_time, restart),),
        output_times_s=np.array([0.0, event_time, 1.0]))
    assert result.completed, (result.termination, result.error)
    event = result.restarts[0]
    index = int(np.flatnonzero(result.accepted_times_s == event_time)[0])
    assert result.accepted_coordinates[index].tobytes() == event.right_u.tobytes()
    assert event.left_u[1] == 0 and event.right_u[1] == 1
    assert event.left_storage.tobytes() == event.right_storage.tobytes()
    first_after = next(step for step in result.accepted_steps if step.start_s == event_time)
    assert first_after.order == 1 and first_after.initial_coordinates.tobytes() == event.right_u.tobytes()
    assert all(s.u.tobytes() == event.right_u.tobytes() for s in result.samples if s.time_s == event_time)


def test_failed_restart_keeps_the_left_accepted_coordinate():
    def factory(t, side):
        return Driven(0.0 if t < .5 or (t == .5 and side == "left") else 1.0)

    def restart(old, new, u):
        right = u.copy()
        right[1] = new.voltage
        return right, Assessment({}, (GateOutcome("explicit_physical_refusal", False, 2.0, 1.0),))

    result = solve_bdf(factory, np.zeros(2), (0, 1), options=BDFOptions(.125, adaptive=False),
        settings=settings, events=(BDFEvent(.5, restart),))
    assert not result.completed and result.termination == "restart_physical_gate_rejected"
    assert not result.restarts[0].accepted
    assert result.accepted_coordinates[-1].tobytes() == result.restarts[0].left_u.tobytes()
    assert result.accepted_coordinates[-1].tobytes() != result.restarts[0].right_u.tobytes()

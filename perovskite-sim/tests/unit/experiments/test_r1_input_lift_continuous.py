"""Continuation refuses gaps and stops on the first original-solver failure."""
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift_continuous as continuous
from perovskite_sim.experiments.one_dimensional_mechanism_r1_input_lift import RebasedInputLiftR1System


def test_original_substep_arithmetic_allowed_but_schedule_gaps_rejected():
    dt = (0.1-0.08254041852680186)/4
    points = [0.08254041852680186+k*dt for k in range(5)]
    schedule = [{"previous_time_s": a, "time_s": b, "dt_s": dt, "voltage_V": .005}
                for a, b in zip(points, points[1:])]
    assert continuous.validate_schedule(schedule)[-1]["time_s"] == .1
    schedule[1]["previous_time_s"] += .00001
    with pytest.raises(ValueError):
        continuous.validate_schedule(schedule)


def test_original_rejection_stops_window_without_retry(monkeypatch):
    calls = []
    system = object.__new__(RebasedInputLiftR1System)
    system.dimension = 1
    previous = SimpleNamespace(storage=np.ones(1))
    monkeypatch.setattr(RebasedInputLiftR1System, "rebase", lambda self, state: (self, state))
    monkeypatch.setattr(RebasedInputLiftR1System, "set_voltage_lift", lambda *args: None)
    scales = SimpleNamespace(storage_scale=lambda *args: np.ones(1),
                             poisson_scale=lambda *args: np.ones(1),
                             local_algebraic_scale=lambda *args: np.ones(1))
    def rejected(*args, **kwargs):
        calls.append((args, kwargs))
        raise continuous.transient.InterfaceDefectTransientError("original rejection")
    monkeypatch.setattr(continuous.transient, "_solve_step", rejected)
    steps = [{"previous_time_s": a, "time_s": a+1., "dt_s": 1., "voltage_V": .005}
             for a in (0., 1.)]
    with pytest.raises(continuous.transient.InterfaceDefectTransientError, match="original rejection"):
        list(continuous.advance_steps(system, previous, steps, object(), scaling_system=scales))
    assert len(calls) == 1
    assert calls[0][1]["scaling_system"] is scales


def test_all_steps_checked_before_any_solver_entry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid schedule entered a scientific solver")
    monkeypatch.setattr(continuous, "advance_step", forbidden)
    with pytest.raises(ValueError):
        list(continuous.advance_steps(None, None, [
            {"previous_time_s": 0., "time_s": 1., "dt_s": 1., "voltage_V": .005},
            {"previous_time_s": 2., "time_s": 3., "dt_s": 1., "voltage_V": .005},
        ], None, scaling_system=None))

"""Endpoint-only observations must not hide an invalid intervening state."""

from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from tests.accepted_trajectory import AcceptedTrajectoryMonitor, observe_accepted_radau


def test_negative_accepted_density_is_detected_between_positive_output_samples():
    module = SimpleNamespace(solve_ivp=solve_ivp)
    initial = np.ones(2)
    monitor = AcceptedTrajectoryMonitor(1, initial)
    with observe_accepted_radau(module, monitor):
        result = module.solve_ivp(
            lambda t, _y: np.array([-2 * np.pi * np.sin(2 * np.pi * t), 0.0]),
            (0.0, 1.0),
            initial,
            method="Radau",
            t_eval=[0.0, 1.0],
            max_step=0.03,
            rtol=1e-8,
            atol=1e-10,
        )
    assert result.success and np.all(result.y > 0.0)
    report = monitor.report()
    assert not report["passed"]
    assert "electron_nonpositive" in report["violations"]
    assert report["minimum_density_m3"][0] < -0.99


def test_capacity_excursion_is_detected_between_admissible_output_samples():
    module = SimpleNamespace(solve_ivp=solve_ivp)
    initial = np.array([1.0, 1.0, 0.1])
    monitor = AcceptedTrajectoryMonitor(
        1, initial, weights=np.ones(1), ion_capacities=(np.ones(1),)
    )
    with observe_accepted_radau(module, monitor):
        result = module.solve_ivp(
            lambda t, _y: np.array([0.0, 0.0, 2 * np.pi * np.sin(2 * np.pi * t)]),
            (0.0, 1.0),
            initial,
            method="Radau",
            t_eval=[0.0, 1.0],
            max_step=0.03,
            rtol=1e-8,
            atol=1e-10,
        )
    assert result.success and np.all((result.y[2] > 0.0) & (result.y[2] < 1.0))
    report = monitor.report()
    assert not report["passed"]
    assert "ion_0_capacity" in report["violations"]
    assert report["maximum_ion_site_fraction"][0] > 2.0


def test_observing_accepted_states_preserves_samples_steps_and_rhs_counts():
    module = SimpleNamespace(solve_ivp=solve_ivp)
    initial = np.array([1.0, 2.0, 0.4])
    rhs = lambda _t, y: np.array([-y[0], -y[1], 0.0])
    settings = dict(
        method="Radau",
        t_eval=np.linspace(0.0, 1.0, 9),
        max_step=0.05,
        rtol=1e-7,
        atol=1e-9,
    )
    plain = solve_ivp(rhs, (0.0, 1.0), initial, **settings)
    monitor = AcceptedTrajectoryMonitor(
        1, initial, weights=np.ones(1), ion_capacities=(np.ones(1),)
    )
    with observe_accepted_radau(module, monitor):
        observed = module.solve_ivp(rhs, (0.0, 1.0), initial, **settings)
    monitor.assert_valid()
    np.testing.assert_array_equal(observed.t, plain.t)
    np.testing.assert_array_equal(observed.y, plain.y)
    assert (observed.nfev, observed.njev, observed.nlu) == (
        plain.nfev,
        plain.njev,
        plain.nlu,
    )
    assert monitor.report()["accepted_step_count"] > observed.t.size

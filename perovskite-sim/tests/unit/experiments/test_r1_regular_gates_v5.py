"""Physical counterexamples reach regular-current metric checks directly.

These checks do not change source identities, standard bytes or final flags.
The displacement defect reaches independent Gauss/current assembly; no
reported arrays are supplied, so content agreement cannot mask a lost gate.
"""
import numpy as np

from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_regular as regular
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from tests.unit.experiments.test_r1_independent_physics import fixture
from tests.unit.experiments.test_r1_regular_independence_v4 import system_state


def test_actual_regular_displacement_fault_is_rejected_by_metric_limit(system_state, monkeypatch):
    system, state = system_state
    state.phi = 1e-3 * system.grid
    state.dqfn, state.dqfp = -state.phi, state.phi
    baseline = regular.independent_regular_current(system, state, policy=r1_policy())
    assert baseline["passed"]
    magnitude = float(np.max(np.abs(baseline["arrays"]["internal_maxwell_A_m2"])))
    real = regular._gauss_derivative
    computed = []
    def defective(*args):
        phi, trace, displacement, interface, contact = real(*args)
        displacement = displacement.copy()
        displacement[1] += magnitude * 1e-4
        computed.append(displacement.copy())
        return phi, trace, displacement, interface, contact
    monkeypatch.setattr(regular, "_gauss_derivative", defective)
    report = regular.independent_regular_current(system, state, policy=r1_policy())
    assert len(computed) == 1
    metric = "internal_face_current_spread_relative"
    assert 2e-6 < report["metrics"][metric] < 1e-3
    assert report["limits"][metric] == 2e-6
    assert report["checks"][metric]["passed"] is False
    assert not report["passed"]
    assert set(report["reasons"]) == {metric, "contact_internal_current_spread_relative"}
    assert not any("content_matches" in key for key in report["checks"])

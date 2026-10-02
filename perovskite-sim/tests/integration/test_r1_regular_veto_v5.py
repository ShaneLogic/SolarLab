"""A real prepared state reaches the production independent-regular veto."""
import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_regular as regular
from perovskite_sim.experiments import one_dimensional_mechanism_r1_step as step
from perovskite_sim.experiments.interface_defect_ion_transient import InterfaceDefectIonTransientError
from tests.integration.test_r1_regular_independence_v4 import initial

pytestmark = pytest.mark.slow


def test_real_regular_displacement_fault_reaches_the_production_veto(initial, monkeypatch):
    system, before, policy = initial
    built = step.build_initial_step(system, before, .005, policy=policy)
    baseline = step.regular_current_at_state(built.system, built.zero_plus, policy=policy)
    assert baseline.evidence["independent_physics"]["passed"]
    real = regular._gauss_derivative
    calls = []
    magnitude = float(np.max(np.abs(baseline.evidence["internal_maxwell_A_m2"])))
    def defective(*args):
        phi, trace, displacement, interface, contact = real(*args)
        displacement = displacement.copy()
        displacement[len(displacement) // 2] += magnitude * 1e-3
        calls.append(True)
        return phi, trace, displacement, interface, contact
    monkeypatch.setattr(regular, "_gauss_derivative", defective)
    with pytest.raises(InterfaceDefectIonTransientError, match="regular current failed independent reconstruction") as error:
        step.regular_current_at_state(built.system, built.zero_plus, policy=policy)
    assert calls == [True]
    evidence = error.value.result
    report = evidence["independent_physics"]
    assert report["metrics"]["internal_face_current_spread_relative"] > 2e-6
    assert report["checks"]["internal_face_current_spread_relative"]["passed"] is False
    assert report["passed"] is False
    assert evidence["limits"] == baseline.evidence["limits"]
    assert evidence["internal_maxwell_A_m2"] == baseline.evidence["internal_maxwell_A_m2"]


def test_actual_published_regular_limit_relaxation_is_rejected(initial, monkeypatch):
    system, before, policy = initial
    built = step.build_initial_step(system, before, .005, policy=policy)
    real = step.effective_regular_limits
    called = []
    def relaxed(*args, **kwargs):
        limits = real(*args, **kwargs)
        limits["face_current_spread_relative"] *= 1000.
        called.append(True)
        return limits
    monkeypatch.setattr(step, "effective_regular_limits", relaxed)
    with pytest.raises(ValueError, match="execution standard: published_regular_relative"):
        step.regular_current_at_state(built.system, built.zero_plus, policy=policy)
    assert called == [True]

"""Effective execution limits are bound separately from physical formulas."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from perovskite_sim.experiments import interface_defect_transient as newton
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments import one_dimensional_mechanism_r1_standard as standard
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_regular as regular
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import EXECUTION_STANDARD_RELATIVE_PATH
from tests.unit.experiments.test_r1_independent_physics import fixture
from tests.unit.experiments.test_r1_regular_independence_v4 import system_state

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def execution():
    return standard.load_execution_standard(SimpleNamespace(read_bytes=lambda path: (PROJECT / path).read_bytes()))


@pytest.mark.parametrize("factor", (1., .1, .01, .001))
def test_every_supported_policy_retains_the_registered_execution_limits(execution, factor):
    runtime = standard.runtime_execution_contract(protocol.r1_policy(factor))
    assert standard.verify_runtime_execution_standard(execution, runtime=runtime)["constants_match"]


@pytest.mark.parametrize("name", (
    "_DEFAULT_MAXIMUM_CHARGE_BALANCE_RELATIVE_ERROR",
    "_DEFAULT_MAXIMUM_ALL_FACE_CURRENT_SPREAD_RELATIVE",
    "_DEFAULT_MAXIMUM_INTERFACE_CURRENT_RELATIVE_ERROR",
))
def test_effective_newton_maximum_cannot_hide_a_changed_solver_default(execution, monkeypatch, name):
    monkeypatch.setattr(newton, name, getattr(newton, name) * 1000.)
    with pytest.raises(ValueError, match="execution standard: newton_acceptance"):
        standard.verify_runtime_execution_standard(execution)


@pytest.mark.parametrize("name,value", (
    ("maximum_scaled_nonlinear_residual", .5),
    ("maximum_charge_balance_relative_error", 1e-7),
    ("maximum_all_face_current_spread_relative", 2e-3),
    ("maximum_two_sided_interface_total_current_relative_error", 2e-3),
    ("maximum_newton_iterations", 101),
    ("maximum_line_search_steps", 41),
    ("maximum_near_acceptance_nonmonotone_steps", 3),
))
def test_actual_execution_policy_cannot_silently_change_acceptance_or_caps(name, value):
    with pytest.raises(ValueError, match="execution standard"):
        standard.verify_execution_policy(replace(protocol.r1_policy(), **{name: value}))


@pytest.mark.parametrize("finite_step", (True, False))
def test_each_row_checks_its_actual_published_limits(monkeypatch, finite_step):
    physical = {key: 0. for key, _ in protocol._PHYSICAL_LIMITS}
    physical["trap_storage_check"] = {"applicable": finite_step, "certified": True}
    physical["regular_right_limit"] = {"charge_balance_normalized": 0.,
                                        "contact_internal_current_spread_relative": 0.}
    policy = protocol.r1_policy()
    baseline = protocol._physical_step_checks(physical, finite_step=finite_step, policy=policy)
    assert baseline["passed"]
    real = protocol.effective_physical_row_limits
    used = []
    def relaxed(*args, **kwargs):
        limits = real(*args, **kwargs)
        limits["contact_internal_current_spread_relative"] = .01
        used.append(limits)
        return limits
    monkeypatch.setattr(protocol, "effective_physical_row_limits", relaxed)
    with pytest.raises(ValueError, match="execution standard: protocol_"):
        protocol._physical_step_checks(physical, finite_step=finite_step, policy=policy)
    assert len(used) == 1


@pytest.mark.parametrize("relative", (True, False))
def test_independent_regular_checks_the_limits_after_actual_calculation(system_state, monkeypatch, relative):
    system, state = system_state
    policy = protocol.r1_policy()
    baseline = regular.independent_regular_current(system, state, policy=policy, require_relative_closure=relative)
    assert baseline["passed"]
    real_limits, real_gauss = regular.effective_regular_limits, regular._gauss_derivative
    computed = []
    def relaxed(*args, **kwargs):
        limits = real_limits(*args, **kwargs)
        limits["charge_balance_normalized"] *= 1000.
        return limits
    def gauss(*args):
        value = real_gauss(*args)
        computed.append(value)
        return value
    monkeypatch.setattr(regular, "effective_regular_limits", relaxed)
    monkeypatch.setattr(regular, "_gauss_derivative", gauss)
    with pytest.raises(ValueError, match="execution standard: independent_regular_"):
        regular.independent_regular_current(system, state, policy=policy, require_relative_closure=relative)
    assert len(computed) == 1


def test_registered_execution_bytes_are_checked_and_loaded_objects_cannot_change_cached_limits(execution):
    execution["effective_limits"]["newton_acceptance"]["maximum_scaled_nonlinear_residual"] = 1.
    standard.verify_execution_policy(protocol.r1_policy())
    raw = (PROJECT / EXECUTION_STANDARD_RELATIVE_PATH).read_bytes().replace(b"0.05", b"0.5")
    with pytest.raises(ValueError, match="execution standard differs from its pinned digest"):
        standard.load_execution_standard(SimpleNamespace(read_bytes=lambda path: raw))


def test_each_controlled_context_reads_its_own_registered_execution_dependency(monkeypatch):
    raw = (PROJECT / EXECUTION_STANDARD_RELATIVE_PATH).read_bytes()
    reads = []
    monkeypatch.setattr(standard, "current_execution_context", lambda: SimpleNamespace(
        read_bytes=lambda path: reads.append(path) or raw))
    standard.verify_execution_policy(protocol.r1_policy())
    assert reads == [EXECUTION_STANDARD_RELATIVE_PATH] * 2
    raw = raw.replace(b"0.05", b"0.5")
    with pytest.raises(ValueError, match="pinned digest"):
        standard.verify_execution_policy(protocol.r1_policy())


def test_development_rows_do_not_rescan_git_for_each_limit_check(monkeypatch):
    calls = []
    monkeypatch.setattr(standard, "current_execution_context", lambda: None)
    monkeypatch.setattr(standard, "require_r1_checkout", lambda: calls.append(True) or SimpleNamespace(
        read_bytes=lambda path: (PROJECT / path).read_bytes()))
    standard._development_execution_bytes.cache_clear()
    try:
        for _ in range(4):
            standard.verify_execution_policy(protocol.r1_policy())
        assert calls == [True]
    finally:
        standard._development_execution_bytes.cache_clear()

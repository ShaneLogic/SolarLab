"""Failure-only density recovery, original budgets, history and gate contracts."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.sparse import diags

from solarlab.solvers.nonlinear.newton import IterationRecord, NewtonResult, NewtonSettings, Termination
from solarlab.solvers.nonlinear.problem import GateOutcome, ResidualScaling, SolveBudget, StepControl, state_digest
from solarlab.solvers.time import bdf as module
from solarlab.solvers.time import carrier_seed as recovery
from solarlab.solvers.time.bdf import Assessment, BDFOptions, solve_bdf
from solarlab.solvers.time.carrier_seed import DensitySeedContext, recover_density_seed, stagnation_basin_signature


class Device:
    node_count = 1

    def fields(self, u, *, voltage_V=0.0):
        return u[:1], np.exp(u[1:2]), np.exp(-u[2:3])


class Mass:
    def __init__(self, source=1.0):
        self.source = source

    def residual(self, u):
        return np.array([u[0], self.source, self.source])

    def jacobian(self, u):
        return diags([1.0, 0.0, 0.0], format="csc")

    def storage(self, u):
        return np.r_[0.0, np.exp(u[1]), np.exp(-u[2])]

    def storage_tangent(self, u):
        return diags([0.0, np.exp(u[1]), -np.exp(-u[2])], format="csc")


class Target:
    def __init__(self, matrix="normal", source=1.0):
        self.matrix, self.source = matrix, source

    def residual(self, u):
        return np.r_[u[0], np.exp(u[1]) - 1.0 - self.source, np.exp(-u[2]) - 1.0 - self.source]

    def jacobian(self, u):
        if self.matrix == "singular":
            return diags([1.0, 0.0, 0.0], format="csc")
        if self.matrix == "nonfinite":
            return diags([1.0, np.nan, 1.0], format="csc")
        if self.matrix == "invalid":
            return np.zeros((1, 1))
        return diags([1.0, np.exp(u[1]), -np.exp(-u[2])], format="csc")


def controls(budget=None, gates=()):
    return NewtonSettings(ResidualScaling(np.ones(3)), 1e-10, StepControl(np.full(3, 2.0)),
                          SolveBudget(50, 1000, 50, 24) if budget is None else budget, gates, False)


def failed(u, termination=Termination.STAGNATED):
    rows = tuple(IterationRecord(i, 10.0, 1e18, 2e-18, 0, True) for i in (1, 2, 3))
    return NewtonResult(u.copy(), False, termination, 10.0, (), rows, 4, 3, state_digest(u), "recorded plateau")


def context(guard=lambda: "unchanged", validate=lambda: None, checkpoint=lambda: None):
    return DensitySeedContext(Device(), 0.0, {"test_scope": True}, validate, guard, checkpoint)


@pytest.mark.parametrize("termination", [t for t in Termination if t != Termination.STAGNATED])
def test_non_stagnation_results_are_never_recoverable(termination):
    assert not stagnation_basin_signature(failed(np.zeros(3), termination), 1.0)


@pytest.mark.parametrize("change", ["accepted", "backtracking", "stale", "large_alpha", "converged", "short"])
def test_signature_is_specific_to_the_frozen_machine_precision_plateau(change):
    f = failed(np.zeros(3))
    if change == "accepted":
        f = replace(f, accepted=True)
    elif change == "short":
        f = replace(f, iterations=f.iterations[:2])
    else:
        kwargs = {"backtracking": {"backtracks": 1}, "stale": {"fresh_jacobian": False},
                  "large_alpha": {"damping": 2 * np.finfo(float).eps}, "converged": {"scaled_residual": .5}}[change]
        f = replace(f, iterations=tuple(replace(row, **kwargs) for row in f.iterations))
    assert not stagnation_basin_signature(f, 1.0)


def test_original_failure_seed_retry_and_remaining_budget_are_separate():
    u = np.zeros(3)
    original = failed(u)
    record = {}
    result = recover_density_seed(Target(), u, original, controls(), context(), record)
    assert result.accepted
    assert record["original_newton"] is original and record["retry_newton"] is result
    assert record["remaining_budget"] == asdict(SolveBudget(47, 995, 46, 24))
    assert record["seed_work"] == {"residual": 1, "jacobian": 1, "field_map": 1,
                                  "factorization": 1, "linear_solve": 1, "conversion": 1}
    assert record["history_before"] == record["history_after"] == "unchanged"
    np.testing.assert_array_equal(u, np.zeros(3))
    assert np.max(abs(Target().residual(result.state))) <= controls().tolerance


@pytest.mark.parametrize("budget", [SolveBudget(3, 1000, 50, 24), SolveBudget(50, 5, 50, 24), SolveBudget(50, 1000, 4, 24)])
def test_no_seed_or_retry_after_remaining_work_is_exhausted(budget):
    u = np.zeros(3)
    original = failed(u)
    record = {}
    assert recover_density_seed(Target(), u, original, controls(budget), context(), record) is original
    assert record["status"] == "remaining_budget_exhausted" and not record["retry_attempted"]
    assert sum(record["seed_work"].values()) == 0


@pytest.mark.parametrize("matrix", ["singular", "nonfinite", "invalid"])
def test_seed_matrix_failures_are_retained_without_an_alternate_initializer(matrix):
    u = np.zeros(3)
    original = failed(u)
    record = {}
    assert recover_density_seed(Target(matrix), u, original, controls(), context(), record) is original
    assert record["status"] == "seed_or_binding_failed" and record["error"]
    assert not record["retry_attempted"] and record["retry_newton"] is None
    assert record["seed_work"]["residual"] == record["seed_work"]["jacobian"] == 1


@pytest.mark.parametrize("log_value", [-1000.0, 1000.0, np.nan])
def test_nonfinite_or_underflowed_input_population_cannot_be_clipped(log_value):
    u = np.array([0.0, log_value, 0.0])
    original = failed(u)
    record = {}
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        assert recover_density_seed(Target(), u, original, controls(), context(), record) is original
    assert record["status"] == "seed_or_binding_failed" and not record["retry_attempted"]


def test_nonpositive_seed_candidate_fails_without_newton():
    u = np.zeros(3)
    original = failed(u)
    record = {}
    assert recover_density_seed(Target(source=-2.0), u, original, controls(), context(), record) is original
    assert record["seed_work"]["factorization"] == 1 and record["seed_work"]["conversion"] == 0
    assert not record["retry_attempted"]


@pytest.mark.parametrize("reason", ["device mismatch", "voltage reference mismatch", "not a complete target"])
def test_invalid_binding_refuses_before_seed_work(reason):
    def refuse():
        raise ValueError(reason)
    record = {}
    u = np.zeros(3)
    original = failed(u)
    assert recover_density_seed(Target(), u, original, controls(), context(validate=refuse), record) is original
    assert reason in record["error"] and sum(record["seed_work"].values()) == 0


def test_deadline_is_not_reset():
    def expired():
        raise TimeoutError("original deadline")
    record = {}
    with pytest.raises(TimeoutError, match="original deadline"):
        recover_density_seed(Target(), np.zeros(3), failed(np.zeros(3)), controls(), context(checkpoint=expired), record)
    assert record["status"] == "wall_budget_exhausted" and not record["retry_attempted"]


@pytest.mark.parametrize("phase", ["before_seed", "during_seed", "during_retry", "after_retry"])
def test_actual_be_deadline_preserves_exception_failure_work_and_history(monkeypatch, phase):
    from solarlab.experiments.transient import _WallBudgetExceeded

    u = np.zeros(3)
    before = u.tobytes()
    original = failed(u)
    record = {}
    deadline = _WallBudgetExceeded("original BE deadline")
    elapsed = phase == "before_seed"
    calls = {"seed": 0, "retry": 0}
    following = None

    def checkpoint():
        if elapsed:
            raise deadline

    def seed(problem, initial, *, device, voltage_V, work_observer):
        nonlocal elapsed
        calls["seed"] += 1
        work_observer("field_map")
        problem.residual(initial)
        if phase == "during_seed":
            elapsed = True
            work_observer("checkpoint")
        problem.jacobian(initial)
        for work in ("factorization", "linear_solve", "conversion"):
            work_observer(work)
        return np.array([0.0, np.log(2.0), -np.log(2.0)])

    def retry(problem, initial, settings, **kwargs):
        nonlocal elapsed, following
        calls["retry"] += 1
        assert settings.budget == SolveBudget(47, 995, 46, 24)
        assert settings.tolerance == controls().tolerance
        assert settings.gates == controls().gates
        elapsed = True
        if phase == "during_retry":
            checkpoint()
        following = replace(failed(initial), accepted=True, termination=Termination.CONVERGED)
        return following

    monkeypatch.setattr(recovery, "transient_density_basin_seed", seed)
    monkeypatch.setattr(recovery, "solve_newton", retry)
    bound = replace(context(checkpoint=checkpoint),
                    deadline_exceptions=(TimeoutError, _WallBudgetExceeded))
    with pytest.raises(_WallBudgetExceeded) as caught:
        recover_density_seed(Target(), u, original, controls(), bound, record)
    assert caught.value is deadline
    assert isinstance(caught.value, RuntimeError) and not isinstance(caught.value, TimeoutError)
    assert record["status"] == "wall_budget_exhausted"
    assert record["original_newton"] is original and not original.accepted
    assert record["history_before"] == record["history_after"] == "unchanged"
    assert u.tobytes() == record["original_initial"].tobytes() == before
    assert record["retry_attempted"] == (phase in ("during_retry", "after_retry"))
    assert record["retry_newton"] is following
    if phase == "before_seed":
        assert calls == {"seed": 0, "retry": 0}
        assert sum(record["seed_work"].values()) == 0 and record["seed"] is None
    elif phase == "during_seed":
        assert calls == {"seed": 1, "retry": 0}
        assert record["seed_work"] == {"residual": 1, "jacobian": 0, "field_map": 1,
                                       "factorization": 0, "linear_solve": 0, "conversion": 0}
        assert record["seed"] is None
    else:
        assert calls == {"seed": 1, "retry": 1}
        assert all(work == 1 for work in record["seed_work"].values())
        assert record["remaining_budget"] == asdict(SolveBudget(47, 995, 46, 24))
        assert record["seed"].tobytes() != before


@pytest.mark.parametrize("be_deadline_binding", [False, True])
def test_history_mutation_terminates_before_any_acceptance(be_deadline_binding):
    from solarlab.experiments.transient import _WallBudgetExceeded

    values = iter(["before", "after", "after"])
    record = {}
    bound = context(guard=lambda: next(values))
    if be_deadline_binding:
        bound = replace(bound, deadline_exceptions=(TimeoutError, _WallBudgetExceeded))
    with pytest.raises(RuntimeError, match="frozen initial/history"):
        recover_density_seed(Target(), np.zeros(3), failed(np.zeros(3)), controls(), bound, record)
    assert record["status"] == "frozen_history_violation" and not record["retry_attempted"]


@pytest.mark.parametrize("termination", [Termination.BUDGET_EXHAUSTED, Termination.LINE_SEARCH_FAILED, Termination.GATE_REJECTED])
def test_positive_seed_does_not_promote_a_failed_retry(monkeypatch, termination):
    def reject(problem, u, settings, **kwargs):
        assert settings.gates == (gate,)
        return failed(u, termination)
    def gate(u):
        return GateOutcome("original", False, 2.0, 1.0)
    monkeypatch.setattr(recovery, "solve_newton", reject)
    record = {}
    result = recover_density_seed(Target(), np.zeros(3), failed(np.zeros(3)), controls(gates=(gate,)), context(), record)
    assert not result.accepted and record["status"] == "retry_rejected"


@pytest.mark.parametrize("mode", ["accepted", "physical_gate", "LTE"])
def test_bdf_history_and_original_gates_follow_recovered_main_and_auxiliary_calls(monkeypatch, mode):
    def stagnant(problem, u, settings, **kwargs):
        for _ in range(4):
            problem.residual(u)
        for _ in range(3):
            problem.jacobian(u)
        return failed(u)
    monkeypatch.setattr(module, "solve_newton", stagnant)
    u = np.zeros(3)
    budget = controls()
    opts = BDFOptions(1.0, relative_tolerance=1e-10, absolute_tolerance=1e-10,
                      max_steps=1, max_attempts=2, max_consecutive_rejections=1)
    def binding(step, original):
        return context(guard=lambda: state_digest(step.history[0].u) + state_digest(step.history[0].storage_value))
    result = solve_bdf(lambda t, side: Mass(t if mode == "LTE" else 1.0), u, (0.0, 1.0),
        options=opts, settings=lambda step: budget, density_seed_context=binding,
        assess=lambda step, value: Assessment({}, (GateOutcome("original physical", mode != "physical_gate", 0, 0),)),
        output_times_s=np.array([0.0, 1.0]))
    records = result.controls["density_seed_recoveries"]
    assert records and all(r["original_newton"].termination == Termination.STAGNATED for r in records)
    assert all(r["seed"] is not None and r["retry_newton"].accepted for r in records)
    assert result.nonlinear_calls == 2 * len(records)
    if mode == "accepted":
        assert result.completed and len(records) == 3
        assert len(result.attempts[0].auxiliary_newton) == 2 and result.attempts[0].error_norm <= 1
        assert {r["stage"] for r in records} == {"main", "startup_half_1", "startup_half_2"}
    else:
        assert not result.completed and len(result.accepted_times_s) == 1
        assert result.accepted_coordinates[0].tobytes() == u.tobytes()
        assert result.termination == ("physical_gate_rejected" if mode == "physical_gate" else "rejection_budget_exhausted")


@pytest.fixture(scope="module")
def compiled_device():
    from examples.research_1d.steady import load_device
    from solarlab.device.compile_1d import compile_device_1d
    return compile_device_1d(load_device(), intervals_per_layer=2)


@pytest.mark.parametrize("mode", ["valid", "BDF2", "ramp", "already_light", "voltage", "away_from_knot", "wrong_generation"])
def test_real_adapter_scope_only_admits_complete_fixed_voltage_dark_jump_BDF1(compiled_device, mode):
    from solarlab.equations.dc_1d import DC1DProblem
    from solarlab.solvers.time.bdf import BDFStep, HistoryPoint
    from solarlab.solvers.time.bdf_1d import BDFDrive, ElectronicStorage1D, _generation_density_context
    values = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    right = values.copy()
    right[1, 1] = 1.0
    if mode == "ramp":
        values[1] = right[1]
    if mode == "already_light":
        values[1, 1], right[1, 1] = 1.0, 2.0
    if mode == "voltage":
        right[1, 0] = .1
    drive = BDFDrive(np.array([0.0, .5, 1.0]), values, right)
    old = ElectronicStorage1D(DC1DProblem(compiled_device, float(right[1, 0]), float(right[1, 1])))
    endpoint = ElectronicStorage1D(DC1DProblem(compiled_device, 0.0, .5 if mode == "wrong_generation" else 1.0))
    u = np.zeros(3 * compiled_device.node_count)
    point = HistoryPoint(.4 if mode == "away_from_knot" else .5, u, old.storage(u), old)
    history = (point, HistoryPoint(.3, u, old.storage(u), old)) if mode == "BDF2" else (point,)
    step = BDFStep(endpoint, history, .6, 2 if mode == "BDF2" else 1)
    scoped = _generation_density_context(drive, step)
    if mode == "valid":
        assert scoped is not None
        scoped.validate()
        assert scoped.scope["device_identity"] == compiled_device.identity
        assert scoped.history_fingerprint() == scoped.history_fingerprint()
    else:
        assert scoped is None

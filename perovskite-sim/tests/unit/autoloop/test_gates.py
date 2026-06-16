import pytest
from perovskite_sim.autoloop.types import LadderResult, ParityScore, SweepScore
from perovskite_sim.autoloop.gates import (
    run_gate_stack, gate_g4_deferred, gate_g5_deferred,
)


def _score(overall, closure):
    return ParityScore(overall=overall, base_deltas={"V_oc": 0.0},
                       per_sweep={"CHI_ETL": SweepScore("CHI_ETL", closure, 10, 8)})


def test_gates_pass_when_no_regression():
    res = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.80, 80.0), details={})
    verdicts = run_gate_stack(res, baseline=_score(0.78, 78.0), regression_tol=0.01)
    assert all(v.passed for v in verdicts)


def test_g1_fails_when_l0_red():
    res = LadderResult(l0_pass=False, l1_pass=False, score=None, details={})
    verdicts = {v.name: v for v in run_gate_stack(res, baseline=_score(0.78, 78.0))}
    assert verdicts["G1_numerics"].passed is False
    assert verdicts["G3_scorecard"].passed is False   # cannot improve without a score


def test_g3_fails_on_parity_regression():
    res = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.70, 70.0), details={})
    verdicts = {v.name: v for v in run_gate_stack(res, baseline=_score(0.80, 80.0),
                                                  regression_tol=0.01)}
    assert verdicts["G3_scorecard"].passed is False    # 0.70 < 0.80 - 0.01


def test_g4_g5_are_deferred_stubs():
    with pytest.raises(NotImplementedError):
        gate_g4_deferred(mechanism=None, residual=None)
    with pytest.raises(NotImplementedError):
        gate_g5_deferred()

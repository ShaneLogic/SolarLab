# tests/unit/autoloop/test_search_optimizer.py
import pytest
from perovskite_sim.autoloop.search import DesignKnob, RandomSearchOptimizer


SPACE = [DesignKnob("a", 0.0, 10.0, "linear"), DesignKnob("b", 1e2, 1e6, "log")]


def _obj(design):
    # PCE proxy = value of knob "a"; always bracketed
    return (design["a"], True)


def test_seeded_determinism():
    o1 = RandomSearchOptimizer(seed=7).optimize(_obj, SPACE, budget=8)
    o2 = RandomSearchOptimizer(seed=7).optimize(_obj, SPACE, budget=8)
    assert [t.design for t in o1] == [t.design for t in o2]


def test_samples_within_bounds_and_sorted():
    trials = RandomSearchOptimizer(seed=3).optimize(_obj, SPACE, budget=20)
    assert len(trials) == 20
    for t in trials:
        assert 0.0 <= t.design["a"] <= 10.0
        assert 1e2 <= t.design["b"] <= 1e6      # log knob stays in range
    pces = [t.pce for t in trials]
    assert pces == sorted(pces, reverse=True)    # sorted by PCE desc


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        RandomSearchOptimizer().optimize(_obj, SPACE, budget=0)

# tests/unit/autoloop/test_search_orchestration.py
import json
import pytest
from perovskite_sim.autoloop.search import (
    run_design_search, DesignKnob, SearchNotTrusted, RandomSearchOptimizer,
)

SPACE = [DesignKnob("a", 0.0, 10.0, "linear")]


def test_refuses_when_not_parity_trusted(tmp_path):
    with pytest.raises(SearchNotTrusted):
        run_design_search(
            config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
            outputs_root=tmp_path / "out", timestamp="t", space=SPACE, budget=5,
            parity_target=0.9, parity_fn=lambda: 0.5,           # below target
            objective=lambda d: (d["a"], True), optimizer=RandomSearchOptimizer(seed=1))


def test_runs_when_trusted_and_writes_report(tmp_path):
    result = run_design_search(
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        outputs_root=tmp_path / "out", timestamp="2026-06-16T00:00:00Z",
        space=SPACE, budget=12, parity_target=0.9,
        parity_fn=lambda: 0.95,                                  # trusted
        objective=lambda d: (d["a"], True), optimizer=RandomSearchOptimizer(seed=1))
    assert result.n_evaluated == 12
    assert result.parity_overall == 0.95
    assert result.best.pce == max(t.pce for t in result.trials)  # best = top PCE
    # advisory report written, nothing applied
    report = tmp_path / "out" / "search-2026-06-16T00:00:00Z" / "result.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["budget"] == 12 and data["n_evaluated"] == 12


def test_all_unbracketed_does_not_crash(tmp_path):
    result = run_design_search(
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        outputs_root=tmp_path / "out", timestamp="t", space=SPACE, budget=4,
        parity_target=0.5, parity_fn=lambda: 0.9,
        objective=lambda d: (0.0, False), optimizer=RandomSearchOptimizer(seed=1))
    assert result.best.pce == 0.0
    assert all(not t.bracketed for t in result.trials)

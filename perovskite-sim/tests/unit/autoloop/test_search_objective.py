# tests/unit/autoloop/test_search_objective.py
from perovskite_sim.autoloop import search


class _Metrics:
    def __init__(self, pce, bracketed):
        self.PCE = pce
        self.voc_bracketed = bracketed


class _Result:
    def __init__(self, m):
        self.metrics_fwd = m


def test_objective_returns_pce_when_bracketed(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE_STACK")
    monkeypatch.setattr(search, "apply_sweep_point", lambda base, sp: "SWEPT")
    monkeypatch.setattr(search, "run_jv_sweep", lambda stack, **kw: _Result(_Metrics(0.27, True)))
    obj = search.make_design_objective(tmp_path / "c.yaml", {"N_grid": 30})
    pce, bracketed = obj({"etl_doping_cm3": 1e17})
    assert pce == 0.27 and bracketed is True


def test_objective_zero_when_unbracketed(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE")
    monkeypatch.setattr(search, "apply_sweep_point", lambda base, sp: "SWEPT")
    monkeypatch.setattr(search, "run_jv_sweep", lambda stack, **kw: _Result(_Metrics(0.0, False)))
    obj = search.make_design_objective(tmp_path / "c.yaml", {})
    assert obj({"x": 1.0}) == (0.0, False)


def test_objective_zero_and_logged_on_exception(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE")
    def _boom(base, sp): raise RuntimeError("solver diverged")
    monkeypatch.setattr(search, "apply_sweep_point", _boom)
    obj = search.make_design_objective(tmp_path / "c.yaml", {})
    import logging
    with caplog.at_level(logging.WARNING):
        pce, bracketed = obj({"x": 1.0})
    assert pce == 0.0 and bracketed is False
    assert "design eval failed" in caplog.text          # logged, not swallowed

# tests/unit/autoloop/test_ladder.py
from perovskite_sim.autoloop.ladder import run_l0, build_run_callables, run_ladder
from perovskite_sim.autoloop.types import LadderResult


def test_run_l0_reports_pass_on_green_subprocess(monkeypatch):
    class _CP:
        returncode = 0
        stdout = "5 passed"
        stderr = ""
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.subprocess.run",
                        lambda *a, **k: _CP())
    ok, detail = run_l0(["tests/unit/autoloop"])
    assert ok is True
    assert "passed" in detail


def test_run_l0_reports_fail_on_red_subprocess(monkeypatch):
    class _CP:
        returncode = 1
        stdout = "1 failed"
        stderr = ""
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.subprocess.run",
                        lambda *a, **k: _CP())
    ok, _ = run_l0(["tests/unit/autoloop"])
    assert ok is False


def test_run_ladder_short_circuits_when_l0_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.run_l0", lambda paths: (False, "1 failed"))
    res = run_ladder(reference_path=tmp_path / "ref.json", config_path=tmp_path / "c.yaml",
                     l0_paths=["tests/unit/autoloop"])
    assert isinstance(res, LadderResult)
    assert res.l0_pass is False
    assert res.score is None        # L2 not reached

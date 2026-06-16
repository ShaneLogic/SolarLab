# tests/unit/autoloop/test_orchestrator_implement.py
import subprocess
from pathlib import Path
import pytest
from perovskite_sim.autoloop.types import Gap, Hypothesis, GateVerdict
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import implement_top_confirmed, commit_promotion

_YAML = "name: x\ndevice:\n  mode: fast\nlayers:\n  - a\n"


def _gap(gid="trend:Nd_ETL:V_oc", mag=0.4, status="open"):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _confirmed_hyp(gid="trend:Nd_ETL:V_oc"):
    return Hypothesis(gap_id=gid, cause="physics",
                      mechanism="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed")


def _setup(tmp_path, *, status="open", hyp=True):
    cfg = tmp_path / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap(status=status))
    if hyp:
        led.add_hypothesis(_confirmed_hyp())
    led.save()
    return cfg


def _green_gates(edit, gap, hyp):
    return [GateVerdict("G1_numerics", True, ""), GateVerdict("G0_legacy_bit_identical", True, "")]


def _red_gates(edit, gap, hyp):
    return [GateVerdict("G1_numerics", True, ""), GateVerdict("G0_legacy_bit_identical", False, "boom")]


def test_no_confirmed_returns_status(tmp_path):
    cfg = _setup(tmp_path, hyp=False)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates)
    assert r.status == "no_confirmed"


def test_not_promotable_when_lever_has_no_key(tmp_path):
    cfg = tmp_path / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.add_hypothesis(Hypothesis(gap_id="trend:Nd_ETL:V_oc", cause="physics",
                                  mechanism="flag SOLARLAB_INTERFACE_PLANE_STATE term",
                                  verdict="confirmed"))
    led.save()
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates)
    assert r.status == "not_promotable"


def test_dry_run_reverts_and_reports(tmp_path):
    cfg = _setup(tmp_path)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates, apply=False)
    assert r.status == "dry_run"
    assert cfg.read_text(encoding="utf-8") == _YAML       # working tree restored


def test_gates_failed_reverts_and_adds_negative(tmp_path):
    cfg = _setup(tmp_path)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_red_gates, apply=True)
    assert r.status == "gates_failed"
    assert cfg.read_text(encoding="utf-8") == _YAML
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("flag SOLARLAB_IFACE_PROJ term")  # refuted -> won't retry


def test_apply_commits_in_tmp_git_repo(tmp_path):
    # real git repo so commit_promotion runs for real
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "autoloop/test"], cwd=repo, check=True)
    cfg = repo / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led_dir = repo / "ledger"
    led = Ledger(root=led_dir); led.add_gap(_gap()); led.add_hypothesis(_confirmed_hyp()); led.save()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)

    r = implement_top_confirmed(ledger_root=led_dir, outputs_root=repo / "out",
                                config_path=cfg, reference_path=repo / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates, apply=True, git_cwd=repo)
    assert r.status == "applied"
    assert r.committed_sha
    assert "interface_plane_projection: true" in cfg.read_text(encoding="utf-8")
    led2 = Ledger.load(led_dir)
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "closed" and g.mechanism == "flag SOLARLAB_IFACE_PROJ term"


def test_commit_promotion_refuses_on_main(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    cfg = repo / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    from perovskite_sim.autoloop.types import ConfigEdit
    edit = ConfigEdit(config_path=str(cfg), device_key="interface_plane_projection",
                      new_value=True, old_text=_YAML)
    with pytest.raises(RuntimeError, match="main"):
        commit_promotion(edit, _gap(), _confirmed_hyp(), [], git_cwd=repo)

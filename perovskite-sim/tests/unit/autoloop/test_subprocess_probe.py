# tests/unit/autoloop/test_subprocess_probe.py
import json
from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner


def _gap():
    return Gap(id="trend:Nd_ETL:V_oc", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_runner_parses_metric_from_worker_stdout(monkeypatch, tmp_path):
    captured = {}

    class _CP:
        returncode = 0
        stdout = json.dumps({"metric": 12.5})
        stderr = ""

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env_has_flag"] = kw.get("env", {}).get("SOLARLAB_IFACE_PROJ")
        captured["cwd"] = kw.get("cwd")
        return _CP()

    monkeypatch.setattr("perovskite_sim.autoloop.subprocess_probe.subprocess.run", _fake_run)
    runner = SubprocessProbeRunner(config_path=tmp_path / "c.yaml",
                                   reference_path=tmp_path / "r.json", gap=_gap())
    val = runner.run({"env_flags": {"SOLARLAB_IFACE_PROJ": "1"},
                      "jv_overrides": {}, "measure": "gap"})
    assert val == 12.5
    assert captured["env_has_flag"] == "1"                 # flag injected into env
    assert "_probe_worker" in " ".join(captured["cmd"])
    from perovskite_sim.autoloop.ladder import _PKG_ROOT
    assert captured["cwd"] == str(_PKG_ROOT)               # cwd = package root


def test_runner_raises_on_worker_failure(monkeypatch, tmp_path):
    class _CP:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("perovskite_sim.autoloop.subprocess_probe.subprocess.run",
                        lambda cmd, **kw: _CP())
    runner = SubprocessProbeRunner(config_path=tmp_path / "c.yaml",
                                   reference_path=tmp_path / "r.json", gap=_gap())
    import pytest
    with pytest.raises(RuntimeError):
        runner.run({"env_flags": {}, "jv_overrides": {}, "measure": "gap"})

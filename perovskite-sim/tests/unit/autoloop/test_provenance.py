# tests/unit/autoloop/test_provenance.py
import subprocess

from perovskite_sim.autoloop.provenance import _git, config_hash, stamp


def test_config_hash_is_stable_and_content_addressed(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    h1 = config_hash(p)
    h2 = config_hash(p)
    assert h1 == h2 and len(h1) == 64        # sha256 hex
    p.write_text("a: 2\n", encoding="utf-8")
    assert config_hash(p) != h1


def test_stamp_captures_git_and_flags(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    prov = stamp(
        run_id="run-test",
        config_path=cfg,
        flags={"SOLARLAB_DOS_BAND": "1"},
        seed=1234,
        timestamp="2026-06-16T00:00:00Z",
    )
    assert prov.run_id == "run-test"
    assert prov.seed == 1234
    assert prov.timestamp == "2026-06-16T00:00:00Z"
    assert prov.flags == {"SOLARLAB_DOS_BAND": "1"}
    assert isinstance(prov.git_sha, str) and len(prov.git_sha) >= 7


def test_git_timeout_is_conservative_and_does_not_block(monkeypatch, tmp_path):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("perovskite_sim.autoloop.provenance.subprocess.run", _timeout)
    assert _git("status", "--porcelain") is None

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    prov = stamp(
        run_id="timeout",
        config_path=cfg,
        flags={},
        seed=0,
        timestamp="2026-08-07T00:00:00Z",
    )
    assert prov.git_sha == "unknown"
    assert prov.git_dirty is True

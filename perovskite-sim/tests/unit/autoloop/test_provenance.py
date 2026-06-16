# tests/unit/autoloop/test_provenance.py
from perovskite_sim.autoloop.provenance import stamp, config_hash


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

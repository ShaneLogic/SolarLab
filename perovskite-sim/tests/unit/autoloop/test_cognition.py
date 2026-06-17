import json
import subprocess
import pytest
from perovskite_sim.autoloop import cognition
from perovskite_sim.autoloop.cognition import ClaudeCliRuntime, FakeRuntime, _validate

SCHEMA = {"required": ["cause", "mechanism"], "cause_enum": ["bug", "numerics", "physics", "data"]}


def _envelope(text):
    class _CP:
        returncode = 0
        stdout = json.dumps({"type": "result", "result": text})
        stderr = ""
    return _CP()


def test_validate_accepts_good_and_rejects_bad():
    _validate({"cause": "physics", "mechanism": "x"}, SCHEMA)        # no raise
    with pytest.raises(ValueError):
        _validate({"mechanism": "x"}, SCHEMA)                        # missing cause
    with pytest.raises(ValueError):
        _validate({"cause": "vibes", "mechanism": "x"}, SCHEMA)      # bad enum


def test_fake_runtime_returns_canned():
    assert FakeRuntime({"cause": "bug", "mechanism": "m"}).complete("p", SCHEMA)["cause"] == "bug"


def test_claude_runtime_parses_envelope(monkeypatch):
    captured = {}
    def _run(cmd, **kw):
        captured["cmd"] = cmd
        return _envelope('{"cause": "physics", "mechanism": "missing Auger term"}')
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    out = ClaudeCliRuntime(model="sonnet").complete("diagnose", SCHEMA)
    assert out["mechanism"] == "missing Auger term"
    assert "--output-format" in captured["cmd"] and "json" in captured["cmd"]
    assert "sonnet" in captured["cmd"]


def test_claude_runtime_strips_markdown_fence(monkeypatch):
    monkeypatch.setattr(cognition.subprocess, "run",
                        lambda cmd, **kw: _envelope('```json\n{"cause":"bug","mechanism":"m"}\n```'))
    assert ClaudeCliRuntime().complete("p", SCHEMA)["cause"] == "bug"


def test_claude_runtime_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def _run(cmd, **kw):
        calls["n"] += 1
        return _envelope("not json" if calls["n"] == 1 else '{"cause":"data","mechanism":"m"}')
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    assert ClaudeCliRuntime().complete("p", SCHEMA)["cause"] == "data"
    assert calls["n"] == 2                                          # retried once


def test_claude_runtime_raises_after_retry(monkeypatch):
    monkeypatch.setattr(cognition.subprocess, "run", lambda cmd, **kw: _envelope("never json"))
    with pytest.raises(RuntimeError):
        ClaudeCliRuntime().complete("p", SCHEMA)


def test_claude_runtime_raises_on_timeout(monkeypatch):
    def _run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    with pytest.raises(RuntimeError):
        ClaudeCliRuntime(timeout_s=1).complete("p", SCHEMA)

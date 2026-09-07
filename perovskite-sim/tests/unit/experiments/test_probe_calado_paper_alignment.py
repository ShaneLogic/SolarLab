"""Reject invalid diagnostic scan endpoints before any simulation is started."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def probe(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "scripts/probe_calado_paper_alignment.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("calado_alignment_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arguments", [
    ["--scan-end-voltage", "nan"],
    ["--scan-end-voltage", "inf"],
    ["--scan-start-voltage", "nan"],
    ["--scan-end-voltage", "0"],
    ["--scan-end-voltage", "-1"],
    ["--scan-start-voltage", "1.1", "--scan-end-voltage", "1.08"],
    ["--scan-start-voltage", "1.08", "--scan-end-voltage", "1.08"],
    ["--scan-start-voltage", "1.079", "--scan-end-voltage", "1.08"],
])
def test_invalid_scan_interval_does_not_start_or_write(probe, arguments, monkeypatch, tmp_path):
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["probe", *arguments, "--out-dir", str(output)])
    def unexpected_load(*args):
        pytest.fail("Invalid endpoints must be rejected before loading a device")
    monkeypatch.setattr(probe, "load_device_from_yaml", unexpected_load)
    with pytest.raises(SystemExit) as raised:
        probe.main()
    assert raised.value.code == 2
    assert not output.exists()

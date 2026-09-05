"""Unit contracts for scripts/plot_calado_fig1f_scan_rate.py (no solver run)."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest


def _load(name: str):
    path = Path(f"scripts/{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load("plot_calado_fig1f_scan_rate")


def test_display_rates_are_slowest_peak_fastest(mod):
    hi = {0.01: 0.10, 0.1: 0.50, 1.0: 0.20, 10.0: 0.05}
    assert mod.select_display_rates(hi) == (0.01, 0.1, 10.0)


def test_display_rates_collapse_when_the_peak_is_an_end(mod):
    hi = {0.01: 0.50, 0.1: 0.20, 1.0: 0.05}
    assert mod.select_display_rates(hi) == (0.01, 1.0)


def test_rate_tag_is_filename_safe_and_unique(mod):
    tags = {mod.rate_tag(r) for r in mod.DEFAULT_RATES}
    assert len(tags) == len(mod.DEFAULT_RATES)
    assert all(ch.isalnum() for tag in tags for ch in tag)


def test_base_protocol_exposes_scan_rate_and_hold(mod):
    params = inspect.signature(mod.base.run_protocol).parameters
    assert params["scan_rate"].default == mod.base.SCAN_RATE_V_S
    assert params["hold_s"].default == mod.base.HOLD_S
    assert mod.DEFAULT_HOLD_S == 0.0

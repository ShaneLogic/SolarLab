"""Tests that the tandem sub-cell preset YAMLs load without error and carry
the expected absorber bandgaps.

These are load-correctness tests only — benchmark accuracy is the job of Task 8.
The n/k CSVs referenced by optical_material are stubs (rigid bandgap-shifted
MAPbI3 data) and are not expected to produce accurate optics.
"""
from __future__ import annotations

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml


def _absorber(stack):
    """Return the absorber LayerSpec by role attribute."""
    for layer in stack.layers:
        if getattr(layer, "role", None) == "absorber":
            return layer
    # Fallback: middle layer (index 1 in HTL/absorber/ETL order)
    return stack.layers[1]


def test_wideGap_preset_loads():
    stack = load_device_from_yaml("configs/nip_wideGap_FACs_1p77.yaml")
    absorber = _absorber(stack)
    assert absorber.params.Eg == pytest.approx(1.77, abs=1e-9)


def test_SnPb_preset_loads():
    stack = load_device_from_yaml("configs/nip_SnPb_1p22.yaml")
    absorber = _absorber(stack)
    assert absorber.params.Eg == pytest.approx(1.22, abs=1e-9)

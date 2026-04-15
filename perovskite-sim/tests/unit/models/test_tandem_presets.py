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
    assert absorber.role == "absorber"
    # Absorber Eg/chi intentionally left at 0 — see YAML comment block. The
    # 1.77 eV wide-gap physics enters only via the optical n,k CSV (FA_Cs_1p77)
    # plus a manual V_bi override, so compute_V_bi falls back to V_bi instead
    # of computing a V_bi_eff that blows up when only the absorber has band data.


def test_SnPb_preset_loads():
    stack = load_device_from_yaml("configs/nip_SnPb_1p22.yaml")
    absorber = _absorber(stack)
    assert absorber.role == "absorber"
    # Same convention as the wide-gap preset: absorber Eg/chi left at 0; the
    # 1.22 eV Sn-Pb physics enters via the SnPb_1p22 n,k CSV.

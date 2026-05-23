"""Tests for SCAPS-style material parameter adapters.

SCAPS specifies the conduction- and valence-band effective DOS (N_C, N_V) and
the bandgap E_g. SolarLab consumes the intrinsic carrier density ``ni``.
These tests pin ``ni_from_dos`` against the standard semiconductor relation
``ni^2 = N_C * N_V * exp(-E_g / kT)``.
"""
from __future__ import annotations

import math

import pytest


def test_ni_from_dos_matches_textbook_relation():
    """ni^2 = N_C * N_V * exp(-E_g / kT) for the SCAPS PVK layer."""
    from perovskite_sim.scaps_compat import ni_from_dos

    # SCAPS PVK: N_C = N_V = 1e19 cm^-3 = 1e25 m^-3, E_g = 1.53 eV, T = 300 K
    N_C = 1.0e25
    N_V = 1.0e25
    Eg = 1.53
    kT = 0.025852  # eV at 300 K
    expected_ni = math.sqrt(N_C * N_V * math.exp(-Eg / kT))

    ni = ni_from_dos(N_C_m3=N_C, N_V_m3=N_V, E_g_eV=Eg, T=300.0)
    assert ni == pytest.approx(expected_ni, rel=1.0e-4)


def test_ni_from_dos_rises_with_temperature():
    from perovskite_sim.scaps_compat import ni_from_dos

    ni_cold = ni_from_dos(1.0e25, 1.0e25, 1.53, T=250.0)
    ni_warm = ni_from_dos(1.0e25, 1.0e25, 1.53, T=350.0)
    assert ni_warm > ni_cold


def test_ni_from_dos_rejects_zero_bandgap():
    from perovskite_sim.scaps_compat import ni_from_dos

    with pytest.raises(ValueError, match="E_g_eV"):
        ni_from_dos(1.0e25, 1.0e25, 0.0, T=300.0)

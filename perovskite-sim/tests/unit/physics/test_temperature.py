"""Unit tests for perovskite_sim.physics.temperature scaling utilities."""
from __future__ import annotations

import math
import numpy as np
import pytest

from perovskite_sim.constants import K_B, Q
from perovskite_sim.physics.temperature import (
    thermal_voltage,
    ni_at_T,
    mu_at_T,
    D_ion_at_T,
    T_REF,
)


class TestThermalVoltage:
    def test_reference_value(self):
        """V_T(300K) ~ 0.02585 V."""
        v = thermal_voltage(300.0)
        assert v == pytest.approx(K_B * 300.0 / Q, rel=1e-12)
        assert v == pytest.approx(0.02585, abs=1e-4)

    def test_linear_in_T(self):
        """V_T is linear in T."""
        assert thermal_voltage(600.0) == pytest.approx(2.0 * thermal_voltage(300.0))

    def test_zero_temperature(self):
        assert thermal_voltage(0.0) == 0.0


class TestNiAtT:
    def test_identity_at_reference(self):
        """ni(T=300, Eg=1.5) returns ni_300 unchanged."""
        ni_300 = 1.0e13
        assert ni_at_T(ni_300, 1.5, 300.0) == ni_300

    def test_increases_with_temperature(self):
        """Intrinsic density grows with T (exponential Boltzmann factor)."""
        ni_300 = 1.0e13
        Eg = 1.5
        ni_350 = ni_at_T(ni_300, Eg, 350.0)
        ni_250 = ni_at_T(ni_300, Eg, 250.0)
        assert ni_350 > ni_300 > ni_250

    def test_arrhenius_slope(self):
        """Check the activation factor matches exp(-Eg/2kT)."""
        ni_300 = 1.0e13
        Eg = 1.5
        T_hi = 400.0
        ni_hi = ni_at_T(ni_300, Eg, T_hi)
        # expected ratio (simplified formula): (T/300)^1.5 * exp(-Eg*q/(2k)*(1/T - 1/300))
        ratio_th = (T_hi / 300.0) ** 1.5 * math.exp(
            -Eg * Q / (2.0 * K_B) * (1.0 / T_hi - 1.0 / 300.0)
        )
        assert ni_hi / ni_300 == pytest.approx(ratio_th, rel=1e-10)

    def test_explicit_dos_form(self):
        """When Nc300/Nv300 are given, uses sqrt(Nc*Nv)*exp(-Eg/2kT)."""
        Nc300 = 4.5e24
        Nv300 = 1.0e25
        Eg = 1.55
        T = 350.0
        ni = ni_at_T(1.0, Eg, T, Nc300=Nc300, Nv300=Nv300)
        ratio = T / 300.0
        expected = math.sqrt(Nc300 * ratio ** 1.5 * Nv300 * ratio ** 1.5) * math.exp(
            -Eg * Q / (2.0 * K_B * T)
        )
        assert ni == pytest.approx(expected, rel=1e-10)


class TestMuAtT:
    def test_identity_at_reference(self):
        assert mu_at_T(1.0e-4, 300.0) == 1.0e-4

    def test_power_law_exponent(self):
        """Default gamma=-1.5: mu(T) = mu_300 * (T/300)^-1.5."""
        mu_300 = 2.0e-4
        mu_600 = mu_at_T(mu_300, 600.0)
        assert mu_600 == pytest.approx(mu_300 * 2.0 ** -1.5, rel=1e-12)

    def test_custom_gamma(self):
        mu_300 = 1.0e-4
        assert mu_at_T(mu_300, 600.0, gamma=-2.0) == pytest.approx(mu_300 * 0.25)

    def test_mobility_decreases_with_heating(self):
        """Phonon scattering lowers mobility at higher T."""
        assert mu_at_T(1.0, 400.0) < mu_at_T(1.0, 300.0)


class TestDIonAtT:
    def test_identity_at_reference(self):
        assert D_ion_at_T(1.0e-16, 300.0, 0.58) == 1.0e-16

    def test_zero_returns_zero(self):
        """D_ion=0 stays 0 (non-ion-conducting layer)."""
        assert D_ion_at_T(0.0, 350.0, 0.58) == 0.0

    def test_arrhenius_activation(self):
        D0 = 1.0e-16
        Ea = 0.58
        T_hi = 350.0
        D_hi = D_ion_at_T(D0, T_hi, Ea)
        expected = D0 * math.exp(-Ea * Q / K_B * (1.0 / T_hi - 1.0 / 300.0))
        assert D_hi == pytest.approx(expected, rel=1e-10)
        # Higher T -> larger D_ion (exothermic hop)
        assert D_hi > D0

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
    B_rad_at_T,
    eg_at_T,
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


class TestBRadAtT:
    """Phase 4b: temperature-scaled band-to-band radiative coefficient."""

    def test_identity_at_reference(self):
        assert B_rad_at_T(4e-17, 300.0, gamma=-1.5) == 4e-17

    def test_zero_gamma_is_identity_everywhere(self):
        """gamma=0 (the default) keeps B unchanged at any T."""
        assert B_rad_at_T(4e-17, 200.0, gamma=0.0) == 4e-17
        assert B_rad_at_T(4e-17, 400.0, gamma=0.0) == 4e-17

    def test_zero_input_returns_zero(self):
        """B_300=0 stays 0 regardless of T or gamma."""
        assert B_rad_at_T(0.0, 400.0, gamma=-1.5) == 0.0

    def test_power_law_exponent(self):
        """B(T) = B_300 · (T/300)^gamma."""
        B0 = 4e-17
        gamma = -1.5
        T = 400.0
        expected = B0 * (T / 300.0) ** gamma
        assert B_rad_at_T(B0, T, gamma) == pytest.approx(expected, rel=1e-12)

    def test_default_gamma_is_zero(self):
        """Default gamma=0 so pre-Phase-4b configs are bit-identical."""
        assert B_rad_at_T(4e-17, 350.0) == 4e-17

    def test_detailed_balance_sign_of_temperature_dependence(self):
        """gamma=-1.5 → B decreases with T (phonon-suppressed radiative rate)."""
        B0 = 4e-17
        assert B_rad_at_T(B0, 400.0, gamma=-1.5) < B0
        assert B_rad_at_T(B0, 250.0, gamma=-1.5) > B0


class TestEgAtT:
    """Phase 4b: Varshni bandgap shift referenced to 300 K."""

    def test_identity_at_reference(self):
        assert eg_at_T(1.6, 300.0, alpha=-3e-4, beta=200.0) == 1.6

    def test_zero_alpha_is_identity_everywhere(self):
        """alpha=0 (default) keeps Eg unchanged at any T."""
        assert eg_at_T(1.6, 200.0, alpha=0.0) == 1.6
        assert eg_at_T(1.6, 400.0, alpha=0.0) == 1.6

    def test_default_alpha_is_zero(self):
        assert eg_at_T(1.55, 350.0) == 1.55

    def test_shift_matches_varshni_formula(self):
        """Explicit check of the referenced Varshni: shift_T = α·[T²/(T+β) − T_REF²/(T_REF+β)]."""
        Eg0 = 1.60
        alpha = -3e-4    # MAPbI3-like: Eg increases with T
        beta = -200.0
        T = 250.0
        shift = alpha * (T * T / (T + beta) - 300.0 * 300.0 / (300.0 + beta))
        expected = Eg0 - shift
        assert eg_at_T(Eg0, T, alpha, beta) == pytest.approx(expected, rel=1e-12)

    def test_negative_alpha_increases_eg_with_heating(self):
        """MAPbI3-like Varshni (negative α, positive β): heating widens Eg."""
        Eg0 = 1.60
        Eg_hot = eg_at_T(Eg0, 400.0, alpha=-3e-4, beta=200.0)
        Eg_cold = eg_at_T(Eg0, 200.0, alpha=-3e-4, beta=200.0)
        assert Eg_hot > Eg0 > Eg_cold

    def test_positive_alpha_decreases_eg_with_heating(self):
        """Silicon-like Varshni (positive alpha): heating narrows Eg."""
        Eg0 = 1.12
        # Silicon: α ≈ 4.73e-4 eV/K, β ≈ 636 K
        Eg_hot = eg_at_T(Eg0, 400.0, alpha=4.73e-4, beta=636.0)
        assert Eg_hot < Eg0

    def test_degenerate_denominator_returns_reference(self):
        """T + β = 0 is a pathological Varshni input — return Eg_300 rather than divide by zero."""
        # β = -T makes denom_T = 0
        assert eg_at_T(1.55, 250.0, alpha=-3e-4, beta=-250.0) == 1.55

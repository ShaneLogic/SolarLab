"""Tests for SCAPS-style defect parameter adapters.

The SCAPS PDF specifies bulk and interface defects with a microscopic triplet
``(sigma, v_th, N_t)``. SolarLab consumes SRH lifetimes ``tau`` and interface
surface recombination velocities ``v_eff``. These tests pin the conversion.
"""
from __future__ import annotations

import math

import pytest


def test_srh_lifetime_microscopic_identity():
    """tau = 1 / (sigma * v_th * N_t) for the SCAPS PVK bulk defect."""
    from perovskite_sim.scaps_compat import srh_lifetime

    # SCAPS PVK bulk default: sigma=1e-15 cm^2, v_th=1e7 cm/s, N_t=1e12 cm^-3
    # Converted to SI: sigma=1e-19 m^2, v_th=1e5 m/s, N_t=1e18 m^-3
    tau = srh_lifetime(sigma_m2=1.0e-19, v_th_m_s=1.0e5, N_t_m3=1.0e18)
    assert tau == pytest.approx(1.0e-4)


def test_srh_lifetime_halves_when_density_doubles():
    from perovskite_sim.scaps_compat import srh_lifetime

    tau_base = srh_lifetime(sigma_m2=1.0e-19, v_th_m_s=1.0e5, N_t_m3=1.0e18)
    tau_dense = srh_lifetime(sigma_m2=1.0e-19, v_th_m_s=1.0e5, N_t_m3=2.0e18)
    assert tau_dense == pytest.approx(tau_base / 2.0)


def test_srh_lifetime_zero_density_returns_infinity():
    """SCAPS Nt=0 means no SRH centre. tau diverges; sentinel is +inf."""
    from perovskite_sim.scaps_compat import srh_lifetime

    tau = srh_lifetime(sigma_m2=1.0e-19, v_th_m_s=1.0e5, N_t_m3=0.0)
    assert math.isinf(tau) and tau > 0


def test_interface_surface_velocity_microscopic_identity():
    """v_eff = sigma * v_th * N_t for the SCAPS HTL/PVK interface defect."""
    from perovskite_sim.scaps_compat import interface_surface_velocity

    # SCAPS HTL/PVK interface: sigma=1e-19 cm^2, v_th=1e7 cm/s, N_t=1e12 cm^-3
    # Converted to SI: sigma=1e-23 m^2, v_th=1e5 m/s, N_t=1e18 m^-3
    v_eff = interface_surface_velocity(
        sigma_m2=1.0e-23, v_th_m_s=1.0e5, N_t_m3=1.0e18,
    )
    assert v_eff == pytest.approx(1.0)


def test_interface_surface_velocity_linear_in_density():
    from perovskite_sim.scaps_compat import interface_surface_velocity

    v_base = interface_surface_velocity(1.0e-19, 1.0e5, 1.0e18)
    v_high = interface_surface_velocity(1.0e-19, 1.0e5, 1.0e21)
    assert v_high == pytest.approx(v_base * 1.0e3)

import numpy as np
import pytest
from perovskite_sim.physics.recombination import (
    srh_recombination, radiative_recombination, auger_recombination,
    total_recombination,
)

NI = 3.2e13   # m⁻³  (MAPbI₃ intrinsic carrier density)
NI2 = NI**2


def test_srh_zero_at_equilibrium():
    n = p = NI
    R = srh_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6, n1=NI, p1=NI)
    assert abs(R) < 1e-10 * NI


def test_radiative_zero_at_equilibrium():
    n = p = NI
    R = radiative_recombination(n, p, NI2, B_rad=5e-22)
    assert abs(R) < 1e-30


def test_auger_zero_at_equilibrium():
    n = p = NI
    R = auger_recombination(n, p, NI2, C_n=1e-42, C_p=1e-42)
    assert abs(R) < 1e-30


def test_total_positive_under_injection():
    n = 1e22; p = 1e22  # strong injection
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R > 0


def test_total_negative_for_depletion():
    n = 0.01 * NI; p = 0.01 * NI  # below equilibrium (generation)
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R < 0

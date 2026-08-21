import numpy as np

from perovskite_sim.physics.recombination import (
    auger_recombination,
    bulk_srh_denominator,
    interface_srh_denominator,
    radiative_recombination,
    srh_recombination,
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
    n = 1e22
    p = 1e22  # strong injection
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R > 0


def test_total_negative_for_depletion():
    n = 0.01 * NI
    p = 0.01 * NI  # below equilibrium (generation)
    R = total_recombination(n, p, NI2, tau_n=1e-6, tau_p=1e-6,
                            n1=NI, p1=NI, B_rad=5e-22, C_n=1e-42, C_p=1e-42)
    assert R < 0


def test_bulk_srh_denominator_is_the_exact_production_expression():
    n = np.array([1.0, 3.0])
    p = np.array([2.0, 4.0])
    expected = 7.0 * (n + 11.0) + 5.0 * (p + 13.0)

    denominator = bulk_srh_denominator(
        n, p, tau_n=5.0, tau_p=7.0, n1=11.0, p1=13.0
    )
    rate = srh_recombination(
        n,
        p,
        ni_sq=17.0,
        tau_n=5.0,
        tau_p=7.0,
        n1=11.0,
        p1=13.0,
    )

    np.testing.assert_array_equal(denominator, expected)
    np.testing.assert_array_equal(rate, (n * p - 17.0) / expected)


def test_interface_srh_denominator_keeps_distinct_surface_units():
    denominator = interface_srh_denominator(
        n=2.0,
        p=3.0,
        n1=5.0,
        p1=7.0,
        v_n=11.0,
        v_p=13.0,
    )
    assert denominator == (2.0 + 5.0) / 13.0 + (3.0 + 7.0) / 11.0

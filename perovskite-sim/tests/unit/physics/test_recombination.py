import numpy as np

from perovskite_sim.physics.recombination import (
    auger_recombination,
    auger_recombination_derivatives,
    bulk_srh_denominator,
    interface_recombination,
    interface_recombination_derivatives,
    interface_srh_denominator,
    radiative_recombination,
    radiative_recombination_derivatives,
    srh_recombination,
    srh_recombination_derivatives,
    total_recombination,
    total_recombination_derivatives,
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


def test_interface_recombination_derivatives_match_density_complex_step():
    for n, p, ni_sq in (
        (2.0e19, 5.0e17, 3.0e28),
        (2.0e10, 5.0e9, 3.0e28),
    ):
        args = (ni_sq, 7.0e13, 9.0e14, 2.0e3, 4.0e2)
        derivatives = interface_recombination_derivatives(n, p, *args)
        n_step = n * 1.0e-30
        p_step = p * 1.0e-30
        complex_n = np.imag(
            interface_recombination(n + 1j * n_step, p, *args)
        ) / n_step
        complex_p = np.imag(
            interface_recombination(n, p + 1j * p_step, *args)
        ) / p_step

        assert derivatives.rate == interface_recombination(n, p, *args)
        assert np.sign(derivatives.rate) == np.sign(n * p - ni_sq)
        np.testing.assert_allclose(
            derivatives.electron_density_derivative,
            complex_n,
            rtol=2.0e-13,
            atol=0.0,
        )
        np.testing.assert_allclose(
            derivatives.hole_density_derivative,
            complex_p,
            rtol=2.0e-13,
            atol=0.0,
        )


def test_interface_recombination_derivatives_preserve_blocked_cycle_limit():
    for velocities in ((0.0, 2.0e3), (4.0e2, 0.0), (0.0, 0.0)):
        derivatives = interface_recombination_derivatives(
            2.0e19,
            5.0e17,
            3.0e28,
            7.0e13,
            9.0e14,
            *velocities,
        )

        assert derivatives.rate == 0.0
        assert derivatives.electron_density_derivative == 0.0
        assert derivatives.hole_density_derivative == 0.0


def test_component_derivative_rates_match_production_formulas():
    n = np.array([2.0e14, 3.0e19, 7.0e22])
    p = np.array([5.0e20, 4.0e17, 9.0e21])
    parameters = {
        "ni_sq": np.array([1.0e28, 2.0e30, 3.0e32]),
        "tau_n": np.array([2.0e-7, 3.0e-6, 5.0e-8]),
        "tau_p": np.array([7.0e-7, 2.0e-6, 4.0e-8]),
        "n1": np.array([8.0e13, 9.0e14, 2.0e15]),
        "p1": np.array([6.0e14, 5.0e13, 4.0e15]),
        "B_rad": np.array([1.0e-17, 2.0e-18, 3.0e-19]),
        "C_n": np.array([2.0e-41, 3.0e-42, 4.0e-43]),
        "C_p": np.array([5.0e-41, 6.0e-42, 7.0e-43]),
    }

    srh = srh_recombination_derivatives(
        n,
        p,
        parameters["ni_sq"],
        parameters["tau_n"],
        parameters["tau_p"],
        parameters["n1"],
        parameters["p1"],
    )
    radiative = radiative_recombination_derivatives(
        n, p, parameters["ni_sq"], parameters["B_rad"]
    )
    auger = auger_recombination_derivatives(
        n,
        p,
        parameters["ni_sq"],
        parameters["C_n"],
        parameters["C_p"],
    )

    np.testing.assert_array_equal(
        srh.rate,
        srh_recombination(
            n,
            p,
            parameters["ni_sq"],
            parameters["tau_n"],
            parameters["tau_p"],
            parameters["n1"],
            parameters["p1"],
        ),
    )
    np.testing.assert_array_equal(
        radiative.rate,
        radiative_recombination(n, p, parameters["ni_sq"], parameters["B_rad"]),
    )
    np.testing.assert_array_equal(
        auger.rate,
        auger_recombination(
            n,
            p,
            parameters["ni_sq"],
            parameters["C_n"],
            parameters["C_p"],
        ),
    )


def test_total_recombination_derivatives_match_density_complex_step():
    n = np.array([2.0e14, 3.0e19, 7.0e22])
    p = np.array([5.0e20, 4.0e17, 9.0e21])
    args = (
        np.array([1.0e28, 2.0e30, 3.0e32]),
        np.array([2.0e-7, 3.0e-6, 5.0e-8]),
        np.array([7.0e-7, 2.0e-6, 4.0e-8]),
        np.array([8.0e13, 9.0e14, 2.0e15]),
        np.array([6.0e14, 5.0e13, 4.0e15]),
        np.array([1.0e-17, 2.0e-18, 3.0e-19]),
        np.array([2.0e-41, 3.0e-42, 4.0e-43]),
        np.array([5.0e-41, 6.0e-42, 7.0e-43]),
    )
    derivatives = total_recombination_derivatives(n, p, *args)
    n_step = n * 1.0e-30
    p_step = p * 1.0e-30
    finite_n = np.imag(
        total_recombination(n.astype(complex) + 1j * n_step, p, *args)
    ) / n_step
    finite_p = np.imag(
        total_recombination(n, p.astype(complex) + 1j * p_step, *args)
    ) / p_step

    np.testing.assert_array_equal(
        derivatives.rate,
        total_recombination(n, p, *args),
    )
    np.testing.assert_allclose(
        derivatives.electron_density_derivative,
        finite_n,
        rtol=2.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        derivatives.hole_density_derivative,
        finite_p,
        rtol=2.0e-13,
        atol=0.0,
    )


def test_srh_derivative_at_mass_action_keeps_denominator_response():
    n = np.array([2.0e16])
    p = np.array([5.0e14])
    ni_sq = n * p
    derivative = srh_recombination_derivatives(
        n,
        p,
        ni_sq,
        tau_n=3.0e-6,
        tau_p=7.0e-7,
        n1=4.0e13,
        p1=8.0e14,
    )
    denominator = bulk_srh_denominator(
        n,
        p,
        tau_n=3.0e-6,
        tau_p=7.0e-7,
        n1=4.0e13,
        p1=8.0e14,
    )

    np.testing.assert_array_equal(derivative.rate, np.zeros(1))
    np.testing.assert_array_equal(
        derivative.electron_density_derivative,
        p / denominator,
    )
    np.testing.assert_array_equal(
        derivative.hole_density_derivative,
        n / denominator,
    )

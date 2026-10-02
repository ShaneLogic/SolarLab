"""Analytic checks for the diagnostic Decimal ion-flux oracle."""
from decimal import Decimal

import numpy as np

from scripts.diagnose_one_dimensional_mechanism_r1_weak_flux import decimal_flux


def test_equal_population_zero_field_is_exact_equilibrium():
    result = decimal_flux([0., 0.], [1e22, 1e22], [1e-8], [1e-12], .025,
                          [2e22, 2e22], precision=80)
    assert result == [Decimal(0)]


def test_equal_crowding_zero_field_matches_fick_law():
    result = decimal_flux([0., 0.], [2., 1.], [4.], [2.], 1., [4., 2.], precision=80)
    assert result == [Decimal('.5')]


def test_equal_density_flux_has_exact_field_sign_and_precision_converges():
    # B(x)-B(-x)=-x: this analytic identity is independent of the
    # implementation's exponential quotient and tests cancellation accuracy.
    for field in (1e-12, -1e-12, np.spacing(.025)):
        args = ([0., field], [1., 1.], [1.], [1.], 1., [2., 2.])
        expected = -Decimal.from_float(float(field))
        low = decimal_flux(*args, precision=50)[0]
        high = decimal_flux(*args, precision=80)[0]
        assert abs(high-expected) < Decimal('1e-27')
        assert abs(low-high) < Decimal('1e-31')
        assert high*Decimal.from_float(float(field)) < 0

"""Independent Decimal checks of the arithmetic, without solver/flux imports."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from perovskite_sim.physics.compensated import DD, exp, expm1, log, log1p, sum


def decimal_pairs(value):
    with localcontext() as context:
        context.prec = 110
        return [
            Decimal.from_float(float(h)) + Decimal.from_float(float(l))
            for h, l in zip(value.hi.flat, value.lo.flat)
        ]


def assert_decimal_close(actual, expected, rtol="3e-30", atol="0"):
    with localcontext() as context:
        context.prec = 110
        for got, want in zip(decimal_pairs(actual), expected, strict=True):
            error = abs(got - want)
            assert error <= Decimal(atol) + abs(want) * Decimal(rtol), (got, want, error)


def test_normalization_broadcast_slice_and_explicit_rounding():
    x = DD([[1.0], [1e22]], [1e-23, -1e-23])
    assert x.shape == (2, 2)
    assert x[0].shape == (2,)
    assert x[0, 0].shape == ()
    assert not x.hi.flags.writeable
    assert not x.lo.flags.writeable
    np.testing.assert_array_equal(x[0].hi, [1.0, 1.0])
    np.testing.assert_array_equal(x[0].lo, [1e-23, -1e-23])
    np.testing.assert_array_equal(x.to_float(), x.hi)
    with pytest.raises(TypeError, match="implicit"):
        np.asarray(x)
    with pytest.raises(TypeError, match="implicit"):
        float(x[0, 0])
    with pytest.raises(ValueError):
        x.hi[0, 0] = 2


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_arithmetic_against_decimal_mixed_scales(operation):
    a = DD([0.095, 1e22, -1e-20, 3.7, 1e-120, 1e120],
           [1e-23, 1e5, 3e-39, -2e-17, 1e-139, -1e101])
    b = DD([0.094, 3e21, 2e-20, -1.9, 3e-110, 2e100],
           [-2e-23, -7e4, -1e-39, 2e-18, 1e-129, 1e81])
    with localcontext() as context:
        context.prec = 110
        da, db = decimal_pairs(a), decimal_pairs(b)
        if operation == "add":
            actual, expected = a + b, [x + y for x, y in zip(da, db)]
        elif operation == "subtract":
            actual, expected = a - b, [x - y for x, y in zip(da, db)]
        elif operation == "multiply":
            actual, expected = a * b, [x * y for x, y in zip(da, db)]
        else:
            actual, expected = a / b, [x / y for x, y in zip(da, db)]
        assert_decimal_close(actual, expected, rtol="8e-32")


def test_sub_ulp_drive_survives_accumulation_and_cancellation():
    x = DD(0.095)
    increment = DD(1e-23)
    for _ in range(1000):
        x = x + increment
    change = x - DD(0.095)
    with localcontext() as context:
        context.prec = 110
        expected = Decimal.from_float(1e-23) * 1000
        # The stored full potential has double-double absolute accuracy.
        # Subtracting the O(0.1) reference is ill-conditioned and does not
        # create another 106 bits of relative accuracy in the tiny change.
        assert_decimal_close(x, [Decimal.from_float(0.095) + expected], rtol="3e-31")
        assert_decimal_close(change, [expected], rtol="0", atol="2e-32")
    assert float(0.095 + 1000 * 1e-23) == 0.095
    assert change.hi != 0


def test_sum_preserves_small_terms_between_large_opposites():
    x = DD([[1e22, 1.0, -1e22], [-1e22, 2.0, 1e22]])
    np.testing.assert_array_equal(sum(x, axis=1).to_float(), [1.0, 2.0])
    assert sum(x).to_float() == 3
    assert sum(x, axis=(0, 1), keepdims=True).shape == (1, 1)
    assert sum(DD(np.empty((0, 3))), axis=0).shape == (3,)


@pytest.mark.parametrize("function", [exp, expm1])
def test_exponentials_against_decimal(function):
    x = DD([-600, -51, -1, -0.3, -1e-23, 0, 1e-23, 0.3, 1, 51, 600],
           [1e-16, 2e-16, 3e-18, 1e-20, 1e-42, 0, -1e-42, -1e-20, 3e-18, -2e-16, -1e-16])
    with localcontext() as context:
        context.prec = 110
        expected = [a.exp() - (1 if function is expm1 else 0) for a in decimal_pairs(x)]
        assert_decimal_close(function(x), expected)


@pytest.mark.parametrize("function,values", [
    (log, DD([1e-200, 0.75, 0.999, 1, 1.001, 1.25, 1e22, 1e200],
             [1e-219, 1e-20, -1e-20, 1e-23, 1e-20, -1e-20, 1e5, 1e181])),
    (log1p, DD([-1, -0.7, -0.25, -1e-23, 0, 1e-23, 0.25, 1e22],
               [1e-23, 1e-20, -1e-20, 1e-42, 0, -1e-42, 1e-20, 1e5])),
])
def test_logarithms_against_decimal(function, values):
    with localcontext() as context:
        context.prec = 110
        expected = [(a + (1 if function is log1p else 0)).ln() for a in decimal_pairs(values)]
        assert_decimal_close(function(values), expected)


def test_small_driving_force_round_trip_and_direction():
    x = DD([1e-23, -1e-23, 1e-40, -1e-40])
    y = x.expm1()
    assert np.all(y.hi * x.hi > 0)
    assert_decimal_close(y.log1p(), decimal_pairs(x), rtol="2e-31")
    assert np.all(np.exp(x.hi) - 1 == 0)


def test_near_equal_populations_have_nonzero_log_ratio():
    population = DD(1e22, 1e5)
    neighbor = DD(1e22, 1e5 + 0.25)
    drive = ((neighbor - population) / population).log1p()
    with localcontext() as context:
        context.prec = 110
        want = (decimal_pairs(neighbor)[0] / decimal_pairs(population)[0]).ln()
        assert_decimal_close(drive, [want], rtol="2e-31")
    assert np.log(neighbor.hi / population.hi) == 0


def test_random_physical_scale_functions_against_decimal():
    rng = np.random.default_rng(731)
    x = DD(rng.uniform(-80.0, 80.0, 64), rng.uniform(-1e-17, 1e-17, 64))
    p = DD(10.0 ** rng.uniform(-40.0, 40.0, 64))
    with localcontext() as context:
        context.prec = 110
        assert_decimal_close(x.exp(), [a.exp() for a in decimal_pairs(x)], rtol="8e-31")
        assert_decimal_close(p.log(), [a.ln() for a in decimal_pairs(p)], rtol="8e-31")


def test_reflected_numpy_arithmetic_and_component_round_trip():
    x = DD([1.0, 2.0], [1e-23, -1e-23])
    weights = np.array([2.0, 3.0])
    assert np.all(weights + x == x + weights)
    assert np.all(weights * x == x * weights)
    assert np.all(weights - x == -(x - weights))
    assert np.all(DD(x.hi, x.lo) == x)
    np.testing.assert_array_equal(x != x[0], [False, True])
    assert np.all(DD(0.0, [1e-23, -1e-23]).hi == [1e-23, -1e-23])


def test_scalar_zero_and_empty_elementwise_functions():
    assert DD(0).exp() == 1
    assert DD(0).expm1() == 0
    assert DD(1).log() == 0
    assert DD(0).log1p() == 0
    for function in (exp, expm1, log, log1p):
        assert function(DD(np.empty((0, 3)))).shape == (0, 3)
    assert DD(-1.0, 1e-23).log1p().hi < 0
    with pytest.raises(ValueError, match="integer"):
        DD(2**60 + 1)


@pytest.mark.parametrize("value", [np.inf, np.nan, 1e300, 1e-300])
def test_unsupported_range_is_explicit(value):
    with pytest.raises(ArithmeticError):
        DD(value)


def test_invalid_domains_and_underflow_do_not_fall_back():
    with pytest.raises(ZeroDivisionError):
        DD([1, 2]) / [1, 0]
    with pytest.raises(ValueError):
        log(DD(0))
    with pytest.raises(ValueError):
        log1p(DD(-1))
    with pytest.raises(ArithmeticError):
        exp(DD(700))
    with pytest.raises(ArithmeticError):
        DD(1e-200) * DD(1e-200)
    with pytest.raises(TypeError):
        DD(Decimal("1.1"))
    with pytest.raises(TypeError):
        DD(1 + 2j)

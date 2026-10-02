"""V8 immutable ownership at the trusted DD selection boundary."""

import numpy as np
import pytest
from decimal import Decimal, localcontext

from perovskite_sim.physics.compensated import DD


def assert_immutable_storage(array):
    current = array
    while isinstance(current, np.ndarray):
        assert not current.flags.writeable
        with pytest.raises(ValueError):
            current.flags.writeable = True
        current = current.base
    assert isinstance(current, bytes)


def test_external_arrays_cannot_mutate_the_pair_or_a_shared_copy():
    high = np.array([[.095, 1e22], [2., -3.]])
    low = np.array([[1e-23, 1e5], [1e-19, -1e-20]])
    pair = DD(high, low)
    duplicate = pair.copy()
    assert duplicate is not pair
    expected_high, expected_low = pair.hi.copy(), pair.lo.copy()
    high[:] = np.inf
    low[:] = np.nan
    for value in (pair, duplicate):
        np.testing.assert_array_equal(value.hi, expected_high)
        np.testing.assert_array_equal(value.lo, expected_low)
        assert_immutable_storage(value.hi)
        assert_immutable_storage(value.lo)


def test_all_supported_views_and_advanced_selections_keep_immutable_words():
    pair = DD(np.arange(12.).reshape(3,4), 1e-30)
    for index in (slice(None,None,-1), (1,slice(None)), (1,2),
                  np.array([True,False,True]), ([2,0],slice(None))):
        selected = pair[index]
        assert selected.shape == np.asarray(pair.hi[index]).shape
        np.testing.assert_array_equal(selected.hi, pair.hi[index])
        np.testing.assert_array_equal(selected.lo, pair.lo[index])
        assert_immutable_storage(selected.hi)
        assert_immutable_storage(selected.lo)
    for transformed in (pair.reshape(4,3),pair.ravel(),pair[::-1].ravel()):
        assert transformed.size == pair.size
        assert_immutable_storage(transformed.hi)
        assert_immutable_storage(transformed.lo)


def test_public_component_metadata_cannot_reshape_a_pair_or_its_copy():
    value=DD(np.arange(6.).reshape(2,3),1e-30)
    expected=value.hi.copy()
    copied=value.copy()
    high,low=value.hi,value.lo
    high.shape=(6,)
    low.shape=(3,2)
    assert value.shape==(2,3)
    assert copied.shape==(2,3)
    assert value.lo.shape==value.hi.shape
    # Base-buffer metadata is equally unable to alter a DD's own views.
    if isinstance(high.base,np.ndarray):
        high.base.shape=(1,6)
    assert value.shape==(2,3)
    assert copied.shape==(2,3)
    np.testing.assert_array_equal(value.hi,expected)


def test_external_construction_and_arithmetic_still_validate_range():
    with pytest.raises(ArithmeticError, match="finite"):
        DD([np.inf])
    with pytest.raises(ArithmeticError, match="below"):
        DD(np.ldexp(1.,-971))
    with pytest.raises(ArithmeticError, match="above"):
        DD(np.ldexp(1.,970))*2
    with pytest.raises(ValueError, match="integer"):
        DD(1)+(2**54+1)
    with pytest.raises(ZeroDivisionError):
        DD(1)/0


def test_arithmetic_results_also_cannot_become_mutable():
    initial=DD([.095,1e22], [1e-23,1e5])
    for result in (initial+1,initial*2,initial/3,-initial):
        assert_immutable_storage(result.hi)
        assert_immutable_storage(result.lo)


def test_series_boundary_and_tiny_signed_drives_against_decimal():
    threshold=np.ldexp(1.,-10)
    values=np.array([-1.,-threshold,np.nextafter(-threshold,0.),-1e-23,-1e-80,
                     0.,1e-80,1e-23,np.nextafter(threshold,0.),threshold,1.])
    pair=DD(values,values*1e-18)
    with localcontext() as context:
        context.prec=160
        exact=[Decimal.from_float(float(h))+Decimal.from_float(float(l))
               for h,l in zip(pair.hi,pair.lo)]
        for actual,wanted in ((pair.exp(),[v.exp() for v in exact]),
                              (pair.expm1(),[v.exp()-1 for v in exact])):
            for high,low,want in zip(actual.hi,actual.lo,wanted):
                got=Decimal.from_float(float(high))+Decimal.from_float(float(low))
                assert abs(got-want)<=abs(want)*Decimal("8e-31"), (got,want)
        nonzero=values!=0
        assert np.all(pair.expm1().hi[nonzero]*values[nonzero]>0)


def test_bernoulli_value_and_derivative_at_small_domain_edges():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import (
        bernoulli_pair,bernoulli_derivative_pair,
    )
    pair=DD([-1.,-.001,-2**-10,-1e-23,0.,1e-23,2**-10,.001,1.],1e-40)
    values,slopes=bernoulli_pair(pair),bernoulli_derivative_pair(pair)
    with localcontext() as context:
        context.prec=160
        for h,l,vh,vl,dh,dl in zip(pair.hi,pair.lo,values.hi,values.lo,slopes.hi,slopes.lo):
            x=Decimal.from_float(float(h))+Decimal.from_float(float(l))
            e=x.exp()
            expected_value=x/(e-1)
            expected_slope=((e-1)-x*e)/(e-1)**2
            value=Decimal.from_float(float(vh))+Decimal.from_float(float(vl))
            slope=Decimal.from_float(float(dh))+Decimal.from_float(float(dl))
            assert abs(value-expected_value)<=Decimal("1e-29")
            assert abs(slope-expected_slope)<=Decimal("1e-25")


def test_signed_bernoulli_values_and_derivatives_against_decimal():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import (
        bernoulli_signed_pair,bernoulli_derivative_signed_pair,
    )
    values=np.array([-20.,-1.,-.001,-2**-10,-1e-23,0.,1e-23,2**-10,.001,1.,20.])
    pair=DD(values,values*1e-18)
    forward,backward=bernoulli_signed_pair(pair)
    forward_d,backward_d=bernoulli_derivative_signed_pair(pair)
    with localcontext() as context:
        context.prec=160
        for sign,b,derivative in ((1,forward,forward_d),(-1,backward,backward_d)):
            for h,l,bh,bl,dh,dl in zip(pair.hi,pair.lo,b.hi,b.lo,derivative.hi,derivative.lo):
                x=sign*(Decimal.from_float(float(h))+Decimal.from_float(float(l)))
                if x==0:
                    want,want_d=Decimal(1),Decimal("-.5")
                else:
                    e=x.exp()
                    want=x/(e-1)
                    want_d=((e-1)-x*e)/(e-1)**2
                got=Decimal.from_float(float(bh))+Decimal.from_float(float(bl))
                got_d=Decimal.from_float(float(dh))+Decimal.from_float(float(dl))
                assert abs(got-want)<=max(abs(want),Decimal(1))*Decimal("1e-29")
                assert abs(got_d-want_d)<=Decimal("1e-25")

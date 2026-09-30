from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.dynamic_storage import (
    DynamicStorageIncrementError,
    log_density_increment,
    log_density_update,
    logit_occupancy_increment,
)


def test_log_density_increment_resolves_sub_ulp_relative_motion():
    previous = np.array([1.0e22])
    coordinate_increment = np.array([1.0e-16])

    direct = previous * np.exp(coordinate_increment) - previous
    stable = log_density_increment(previous, coordinate_increment)

    assert direct[0] == 0.0
    assert stable[0] == pytest.approx(1.0e6, rel=2.0e-16)


def test_logit_increment_matches_reconstructed_occupancy_change():
    previous = np.array([1.0e-4, 0.3, 0.9])
    coordinate_increment = np.array([1.0e-7, -0.2, 0.4])
    previous_logit = np.log(previous) - np.log1p(-previous)
    reconstructed = 1.0 / (1.0 + np.exp(-(previous_logit + coordinate_increment)))

    np.testing.assert_allclose(
        logit_occupancy_increment(previous, coordinate_increment),
        reconstructed - previous,
        rtol=2.0e-9,
        atol=1.0e-17,
    )


def test_zero_coordinate_increment_is_exactly_zero():
    np.testing.assert_array_equal(
        log_density_increment(np.array([1.0, 1.0e30]), np.zeros(2)),
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        logit_occupancy_increment(np.array([0.1, 0.9]), np.zeros(2)),
        np.zeros(2),
    )


@pytest.mark.parametrize(
    ("function", "previous", "increment"),
    (
        (log_density_increment, [0.0], [0.0]),
        (log_density_increment, [1.0], [np.nan]),
        (logit_occupancy_increment, [0.0], [0.0]),
        (logit_occupancy_increment, [0.5], [np.inf]),
        (log_density_increment, [1.0, 2.0], [0.0]),
    ),
)
def test_invalid_increment_inputs_fail_closed(function, previous, increment):
    with pytest.raises(DynamicStorageIncrementError):
        function(previous, increment)


# Two frozen V33 failure coordinates, with independent 90-digit reference
# rounding from V34 TraceMapLinuxV1 (not outputs computed by this helper).
TRACE_MAP_REFERENCE_VECTORS = [(('0x1.8d3fe19d0982dp+38', '0x1.764f192cf3ac1p+43', '0x1.28451963606b1p+44', '0x1.f620866214489p+37'), ('-0x1.105f9b3f57cbap-18', '0x1.1188716fd1043p-18', '-0x1.10ba59229b794p-18', '0x1.1136b3009b24cp-18'), ('0x1.8d3f77f308bb7p+38', '0x1.764f7d2977757p+43', '0x1.2844ca7b25d79p+44', '0x1.f6210c5b24793p+37')), (('0x1.8d3fe19d0982dp+38', '0x1.764f192cf3ac1p+43', '0x1.28451963606b1p+44', '0x1.f620866214489p+37'), ('-0x1.105f9b3f21fa8p-18', '0x1.1188716fe0aa4p-18', '-0x1.10ba5922610d2p-18', '0x1.1136b3009a80ap-18'), ('0x1.8d3f77f308bb8p+38', '0x1.764f7d2977757p+43', '0x1.2844ca7b25d7ap+44', '0x1.f6210c5b24793p+37'))]


@pytest.mark.parametrize("anchor_hex, coordinate_hex, expected_hex", TRACE_MAP_REFERENCE_VECTORS)
def test_log_density_update_matches_frozen_independent_trace_reference(
    anchor_hex, coordinate_hex, expected_hex,
):
    anchor = np.array([[float.fromhex(v) for v in anchor_hex]])
    coordinate = np.array([[float.fromhex(v) for v in coordinate_hex]])
    expected = np.array([[float.fromhex(v) for v in expected_hex]])
    np.testing.assert_array_equal(log_density_update(anchor, coordinate), expected)


def test_log_density_update_zero_scalar_shape_and_input_ownership():
    anchor = np.array([[1e-100, 1., 1e22, 1e100], [3., 7., 11., 13.]])
    before = anchor.copy()
    result = log_density_update(anchor, np.zeros_like(anchor))
    np.testing.assert_array_equal(result, before)
    assert result.dtype == np.float64 and result.shape == anchor.shape
    result[0, 0] = 9.
    np.testing.assert_array_equal(anchor, before)
    scalar = log_density_update(7., 0.)
    assert scalar.shape == () and scalar == 7.


def test_log_density_update_fixed_branch_and_large_negative_input(monkeypatch):
    import perovskite_sim.physics.dynamic_storage as storage

    exp, expm1 = np.exp, np.expm1
    calls = {"exp": [], "expm1": []}
    def traced_exp(x, **kwargs):
        calls["exp"].append((x.copy(), kwargs["where"].copy()))
        return exp(x, **kwargs)
    def traced_expm1(x, **kwargs):
        calls["expm1"].append((x.copy(), kwargs["where"].copy()))
        return expm1(x, **kwargs)
    monkeypatch.setattr(storage.np, "exp", traced_exp)
    monkeypatch.setattr(storage.np, "expm1", traced_expm1)
    z = np.array([-700., np.nextafter(-.5, -np.inf), -.5,
                  np.nextafter(-.5, 0.), np.nextafter(.5, 0.), .5,
                  np.nextafter(.5, np.inf), 10.])
    mapped = log_density_update(np.ones_like(z), z)
    np.testing.assert_array_equal(calls["exp"][0][0], z)
    np.testing.assert_array_equal(calls["expm1"][0][0], z)
    np.testing.assert_array_equal(calls["exp"][0][1], [True, True, False, False, False, False, True, True])
    np.testing.assert_array_equal(calls["expm1"][0][1], [False, False, True, True, True, True, False, False])
    assert mapped[0] > 0.
    np.testing.assert_allclose(mapped, exp(z), rtol=3e-16, atol=0.)


@pytest.mark.parametrize("anchor, coordinate", [
    ([0.], [0.]), ([-1.], [0.]), ([np.nan], [0.]), ([np.inf], [0.]),
    ([1.], [np.nan]), ([1.], [np.inf]), ([1.], [-np.inf]),
    ([1., 2.], [0.]), ([1.], [1000.]), ([1.], [-1000.]),
])
def test_log_density_update_rejects_invalid_or_unrepresentable_density(anchor, coordinate):
    with pytest.raises(DynamicStorageIncrementError):
        log_density_update(anchor, coordinate)

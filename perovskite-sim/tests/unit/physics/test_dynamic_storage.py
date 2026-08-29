from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.dynamic_storage import (
    DynamicStorageIncrementError,
    log_density_increment,
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

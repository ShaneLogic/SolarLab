from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.regularization import (
    RHSRegularization,
    compact_positive_part,
    compact_sqrt_abs,
    direction_preserving_magnitude_min,
)


def test_zero_width_is_bit_identical_to_historical_expressions():
    values = np.array([-1.0e6, -1.0, 0.0, 1.0, 1.0e6])
    np.testing.assert_array_equal(compact_sqrt_abs(values), np.sqrt(np.abs(values)))
    np.testing.assert_array_equal(
        compact_positive_part(values), np.maximum(values, 0.0)
    )


def test_compact_sqrt_is_exact_outside_transition():
    values = np.array([-20.0, -10.0, 10.0, 20.0])
    np.testing.assert_array_equal(
        compact_sqrt_abs(values, width=10.0),
        np.sqrt(np.abs(values)),
    )


def test_compact_sqrt_has_finite_zero_field_derivative():
    width = 100.0
    step = 1.0e-4
    derivative = (
        float(compact_sqrt_abs(step, width)) - float(compact_sqrt_abs(-step, width))
    ) / (2.0 * step)
    assert derivative == pytest.approx(0.0, abs=1.0e-14)


def test_compact_sqrt_width_ladder_converges_to_original():
    value = 10.0
    exact = np.sqrt(value)
    errors = [
        abs(float(compact_sqrt_abs(value, width)) - exact)
        for width in (100.0, 50.0, 25.0)
    ]
    assert errors[2] < errors[1] < errors[0]


def test_positive_part_is_exact_outside_transition_and_nonnegative():
    values = np.array([-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0])
    result = compact_positive_part(values, width=10.0)
    assert np.all(result >= 0.0)
    np.testing.assert_array_equal(
        result[[0, 1, 5, 6]], np.array([0.0, 0.0, 10.0, 20.0])
    )


def test_magnitude_min_preserves_direction_and_is_exact_away_from_kink():
    assert direction_preserving_magnitude_min(-10.0, +2.0, 0.1) == -2.0
    assert direction_preserving_magnitude_min(+1.0, -10.0, 0.1) == 1.0


def test_magnitude_min_width_ladder_converges_to_hard_cap():
    hard = direction_preserving_magnitude_min(1.01, 1.0, 0.0)
    errors = [
        abs(direction_preserving_magnitude_min(1.01, 1.0, width) - hard)
        for width in (0.2, 0.1, 0.05)
    ]
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"poole_frenkel_field_width_V_m": -1.0},
        {"interface_density_width_m3": np.nan},
        {"te_cap_relative_width": np.inf},
        {"te_cap_relative_width": True},
        {"te_cap_relative_width": np.bool_(False)},
    ],
)
def test_policy_rejects_invalid_widths(kwargs):
    with pytest.raises(ValueError, match="finite and non-negative"):
        RHSRegularization(**kwargs)


def test_policy_refinement_scales_every_width():
    policy = RHSRegularization(100.0, 1.0e18, 0.02)
    refined = policy.refined(0.1)
    assert refined == RHSRegularization(10.0, 1.0e17, 0.002)
    assert policy.active
    assert not RHSRegularization().active


@pytest.mark.parametrize("factor", [0.0, -1.0, 1.1, np.nan, True, np.bool_(False)])
def test_policy_rejects_invalid_refinement_factor(factor):
    with pytest.raises(ValueError, match="factor"):
        RHSRegularization().refined(factor)


def test_policy_normalizes_signed_zero_for_canonical_hashing():
    policy = RHSRegularization(
        poole_frenkel_field_width_V_m=-0.0,
        interface_density_width_m3=-0.0,
        te_cap_relative_width=-0.0,
    )

    assert np.signbit(
        [
            policy.poole_frenkel_field_width_V_m,
            policy.interface_density_width_m3,
            policy.te_cap_relative_width,
        ]
    ).tolist() == [False, False, False]

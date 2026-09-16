"""Sub-ULP absolute-log updates remain resolvable physical density updates."""
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from perovskite_sim.physics.two_sided_interface import (
    InterfaceTracePotentials, fixed_occupancy_carrier_tangent,
    fixed_occupancy_carrier_tangent_from_density,
)
from tests.unit.physics.test_two_sided_interface import _bulk, _geometry, _physics


def _assert_identical(left, right):
    if is_dataclass(left):
        for field in fields(left):
            _assert_identical(getattr(left, field.name), getattr(right, field.name))
    else:
        np.testing.assert_array_equal(left, right)


def test_density_entry_preserves_legacy_tangent_at_identical_density():
    logs = np.log([2e20, 8e19, 6e19, 3e20])
    args = (_geometry(), _physics(), _bulk(), .31)
    legacy = fixed_occupancy_carrier_tangent(logs, *args)
    resolved = fixed_occupancy_carrier_tangent_from_density(np.exp(logs), *args)
    _assert_identical(legacy, resolved)


def test_zero_field_flux_resolves_update_erased_by_absolute_log():
    density = float(np.exp(30.))
    increment = 1e-15
    assert 30. + increment == 30.
    updated = density * np.exp(increment)
    assert updated != density
    state = np.full(4, density)
    state[0] = updated
    geometry = _geometry(left_distance_m=1., right_distance_m=1.)
    physics = _physics(D_n_left_m2_s=1., transmission=0.)
    bulk = _bulk(phi_left_V=0., phi_right_V=0., n_left_m3=density)
    tangent = fixed_occupancy_carrier_tangent_from_density(
        state, geometry, physics, bulk, .5, InterfaceTracePotentials(0., 0.))
    # Independent zero-field Fick law and its derivative; no logarithm enters
    # the oracle, so replacing the density entry by exp(log(n)) is detected.
    assert tangent.balance.bulk_flux_m2_s[0] == density - updated
    assert tangent.balance.bulk_flux_m2_s[0] != 0.
    assert tangent.bulk_flux_jacobian_log_state_m2_s[0, 0] == -updated
    assert tangent.balance.state_m3[0] == updated


@pytest.mark.parametrize("bad", [[0., 1., 1., 1.], [-1., 1., 1., 1.], [np.nan]*4, [1., 2.]])
def test_density_entry_rejects_unphysical_input(bad):
    with pytest.raises(ValueError, match="finite positive"):
        fixed_occupancy_carrier_tangent_from_density(bad, _geometry(), _physics(), _bulk(), .5)

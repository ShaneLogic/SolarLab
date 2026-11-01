"""Unified contact boundary particle flux (N class).

``F = n_out S (c - c_eq)`` for both carriers, with the historical currents
recovered through the output-channel signs ``J_n = -q F_n`` and
``J_p = +q F_p`` (one rounding of the product order apart). Blocking returns an
exact zero; a fixed reservoir is a constraint, not a flux.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.physics.contacts import CarrierExchangeChannel, legacy_selective_contact_current
from solarlab.physics.contacts.boundary_flux import boundary_particle_flux, reservoir_constraint

Q = 1.602176634e-19
EPS = float(np.finfo(float).eps)


@pytest.mark.parametrize("carrier", ["n", "p"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_unified_boundary_flux_matches_the_historical_robin_current(carrier, side):
    rng = np.random.default_rng(29)
    charge = -Q if carrier == "n" else Q
    for _ in range(300):
        S = float(10.0 ** rng.uniform(-3.0, 9.0))
        density = float(10.0 ** rng.uniform(-5.0, 26.0))
        reservoir = float(10.0 ** rng.uniform(-5.0, 26.0))
        channel = CarrierExchangeChannel(carrier, side, S, reservoir)
        flux = boundary_particle_flux(channel, density)
        legacy = legacy_selective_contact_current(density, reservoir, S, carrier=carrier,
                                                  side=side)
        assert abs(charge * flux.flux - legacy) <= 4.0 * EPS * abs(legacy)
        assert flux.density_derivative == channel.outward_normal * S


def test_finite_exchange_relaxes_the_boundary_toward_its_reservoir():
    """Excess carriers leave through the contact face; deficits are refilled."""
    for side in ("left", "right"):
        channel = CarrierExchangeChannel("n", side, 1.0e4, 1.0e20)
        excess = boundary_particle_flux(channel, 3.0e20).flux
        deficit = boundary_particle_flux(channel, 1.0e19).flux
        inflow_sign = 1.0 if side == "left" else -1.0   # +F enters at the left face
        assert inflow_sign * excess < 0.0 and inflow_sign * deficit > 0.0
        assert boundary_particle_flux(channel, 1.0e20).flux == 0.0


def test_blocking_and_fixed_reservoir_channels():
    blocked = CarrierExchangeChannel("p", "right", 0.0, 1.0e10)
    result = boundary_particle_flux(blocked, 7.0e22)
    assert result.flux == 0.0 and not math.copysign(1.0, result.flux) < 0
    assert result.density_derivative == 0.0
    fixed = CarrierExchangeChannel("n", "left", None, 2.0e18)
    assert reservoir_constraint(fixed, 2.0e18) == 0.0
    assert reservoir_constraint(fixed, 3.0e18) == 1.0e18
    with pytest.raises(ValueError, match="constraint"):
        boundary_particle_flux(fixed, 1.0)
    with pytest.raises(ValueError, match="pins"):
        reservoir_constraint(blocked, 1.0)


def test_large_exchange_velocity_approaches_the_fixed_reservoir():
    """S -> infinity: the boundary balance forces c -> c_eq, the gap closing as 1/S."""
    supply = 1.0e22   # particle flux the interior delivers to the right contact face
    reservoir = 1.0e16
    gaps = []
    for S in (1e2, 1e4, 1e6, 1e8):
        # DC balance of the contact cell: supply = F_out = S (c - c_eq).
        density = reservoir + supply / S
        channel = CarrierExchangeChannel("n", "right", S, reservoir)
        assert boundary_particle_flux(channel, density).flux == pytest.approx(supply, rel=1e-12)
        gaps.append(density - reservoir)
    ratios = np.asarray(gaps[:-1]) / np.asarray(gaps[1:])
    np.testing.assert_allclose(ratios, 100.0, rtol=1e-6)

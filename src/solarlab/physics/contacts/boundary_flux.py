"""Unified contact boundary particle flux (N class).

Along the outward normal ``n_out`` (-1 at the left contact, +1 at the right),
a channel exchanges

    F = n_out S (c - c_eq)                      [m^-2 s^-1, along +x]

for both carriers; ``dF/dc = n_out S``. A blocking channel returns exactly
zero and a fixed reservoir is the algebraic constraint ``c - c_eq``. The
historical currents follow from the output channels' own signs,
``J_n = -q F_n`` and ``J_p = +q F_p``; they equal
:func:`~solarlab.physics.contacts.exchange.legacy_selective_contact_current`
up to the product order of ``q S (c - c_eq)`` (one rounding) and the sign of
an exact zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from solarlab.physics.contacts.exchange import (
    BLOCKING, FIXED_RESERVOIR, CarrierExchangeChannel,
)

__all__ = ["BoundaryParticleFlux", "boundary_particle_flux", "reservoir_constraint"]


@dataclass(frozen=True, slots=True)
class BoundaryParticleFlux:
    """Particle flux through one contact face and its density derivative."""

    flux: float
    density_derivative: float


def boundary_particle_flux(channel: CarrierExchangeChannel, density_m3: float) -> BoundaryParticleFlux:
    """Exchange flux of a blocking or Robin channel at its boundary density."""
    if channel.kind == FIXED_RESERVOIR:
        raise ValueError(
            "a fixed-reservoir channel is the constraint c - c_eq, not a flux; "
            "use reservoir_constraint"
        )
    if channel.kind == BLOCKING:
        return BoundaryParticleFlux(0.0, 0.0)
    assert channel.velocity_m_s is not None
    rate = channel.outward_normal * float(channel.velocity_m_s)
    excess = float(density_m3) - float(channel.reservoir_density_m3)
    return BoundaryParticleFlux(rate * excess, rate)


def reservoir_constraint(channel: CarrierExchangeChannel, density_m3: float) -> float:
    """Dirichlet constraint value ``c - c_eq`` of a fixed-reservoir channel."""
    if channel.kind != FIXED_RESERVOIR:
        raise ValueError("only a fixed-reservoir channel pins its boundary density")
    return float(density_m3) - float(channel.reservoir_density_m3)

"""Carrier exchange at the outer contacts, one channel per carrier and side (M class).

Each of the four channels (electron/hole x left/right) independently selects
its exchange kinetics:

* ``fixed_reservoir`` (velocity ``None``) -- the boundary density is pinned to
  its reservoir (a Dirichlet constraint row ``c - c_eq``);
* ``blocking`` (velocity ``0``) -- zero carrier exchange;
* ``robin`` (finite velocity ``S > 0``) -- finite exchange ``S (c - c_eq)``.

The reservoir density and its source (doping equilibrium, statistics closure,
Schottky barrier or calibrated metal floor) are declared with the channel and
never select the Poisson potential. ``legacy_selective_contact_current`` is the
historical Robin current with its carrier/side sign table, signed zero
included; the unified boundary particle flux is
:mod:`solarlab.physics.contacts.boundary_flux` (N class).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C

__all__ = [
    "BLOCKING", "CONTACT_SIDES", "CarrierExchangeChannel", "DOPING_EQUILIBRIUM",
    "EXCHANGE_KINDS", "FIXED_RESERVOIR", "FLAT_BAND_CONTACT_VELOCITY_M_S", "METAL_FLOOR",
    "RESERVOIR_SOURCES", "ROBIN", "SCHOTTKY_BARRIER", "STATISTICS_CLOSURE",
    "exchange_kind", "legacy_selective_contact_current", "resolve_exchange_velocities",
]

Q = ELEMENTARY_CHARGE_C
FIXED_RESERVOIR = "fixed_reservoir"
BLOCKING = "blocking"
ROBIN = "robin"
EXCHANGE_KINDS = (FIXED_RESERVOIR, BLOCKING, ROBIN)
DOPING_EQUILIBRIUM = "doping_equilibrium"
STATISTICS_CLOSURE = "statistics_closure"
SCHOTTKY_BARRIER = "schottky_barrier"
METAL_FLOOR = "metal_floor"
RESERVOIR_SOURCES = (DOPING_EQUILIBRIUM, STATISTICS_CLOSURE, SCHOTTKY_BARRIER, METAL_FLOOR)
CONTACT_SIDES = ("left", "right")
# Historical finite exchange velocity of ``flat_band_contacts`` [m/s].
FLAT_BAND_CONTACT_VELOCITY_M_S = 1.0e5


def resolve_exchange_velocities(
    velocities: tuple[float | None, float | None, float | None, float | None],
    *,
    flat_band_contacts: bool,
    selective_contacts_enabled: bool,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Effective (n-left, p-left, n-right, p-right) exchange velocities.

    ``None`` pins a reservoir density and zero blocks that carrier. The
    historical ``flat_band_contacts`` flag enables finite exchange on every
    unset channel regardless of tier; otherwise the tier gate decides.
    """
    if flat_band_contacts:
        return tuple(  # type: ignore[return-value]
            FLAT_BAND_CONTACT_VELOCITY_M_S if value is None else value
            for value in velocities
        )
    if not selective_contacts_enabled:
        return (None, None, None, None)
    return velocities


def exchange_kind(velocity_m_s: float | None) -> str:
    """Exchange kinetics of one channel from its velocity."""
    if velocity_m_s is None:
        return FIXED_RESERVOIR
    velocity = float(velocity_m_s)
    if not math.isfinite(velocity) or velocity < 0.0:
        raise ValueError("a contact exchange velocity must be finite and non-negative")
    return BLOCKING if velocity == 0.0 else ROBIN


@dataclass(frozen=True, slots=True)
class CarrierExchangeChannel:
    """One carrier/side contact channel: kinetics plus its reservoir."""

    carrier: str
    side: str
    velocity_m_s: float | None
    reservoir_density_m3: float
    reservoir_source: str = DOPING_EQUILIBRIUM

    def __post_init__(self) -> None:
        if self.carrier not in ("n", "p"):
            raise ValueError(f"carrier must be 'n' or 'p', got {self.carrier!r}")
        if self.side not in CONTACT_SIDES:
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")
        if self.reservoir_source not in RESERVOIR_SOURCES:
            raise ValueError(f"unknown reservoir source {self.reservoir_source!r}")
        density = float(self.reservoir_density_m3)
        if not math.isfinite(density) or density < 0.0:
            raise ValueError("a contact reservoir density must be finite and non-negative")
        exchange_kind(self.velocity_m_s)

    @property
    def kind(self) -> str:
        return exchange_kind(self.velocity_m_s)

    @property
    def outward_normal(self) -> float:
        """Outward normal of the contact face along x: -1 left, +1 right."""
        return -1.0 if self.side == "left" else 1.0


def legacy_selective_contact_current(
    density: float | ArrayLike,
    density_eq: float | ArrayLike,
    S: float,
    *,
    carrier: str,
    side: str,
) -> float | NDArray[np.float64]:
    """Historical Robin boundary current density [A/m^2] along +x.

    ``J_n = +q S (n - n_eq)`` and ``J_p = -q S (p - p_eq)`` at the left
    contact; the right contact flips both. ``S = 0`` returns a signed zero
    exactly as before.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if carrier not in ("n", "p"):
        raise ValueError(f"carrier must be 'n' or 'p', got {carrier!r}")

    excess = np.asarray(density, dtype=float) - np.asarray(density_eq, dtype=float)
    mag = Q * float(S) * excess
    J = mag if carrier == "n" else -mag
    if side == "right":
        J = -J
    return float(J) if np.isscalar(density) else J

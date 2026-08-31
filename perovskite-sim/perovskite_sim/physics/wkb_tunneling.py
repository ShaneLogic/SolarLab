"""WKB transmission primitives for the D8 tunnelling family.

SolarLab already carries a *scalar* thermionic-field-emission enhancement
(``physics/tunneling.py``): one number, built once from geometry and doping,
folded into the Richardson constant. It cannot represent four physically
distinct tunnelling channels, and the D8 exit condition forbids claiming that
it does. This module is the shared, channel-independent foundation the four
channels are built on — it computes transmission from an actual barrier
profile, not from a fitted enhancement.

Core quantities
---------------
For a carrier of energy ``E`` moving through a barrier profile ``U(x)``, the
WKB action over the classically forbidden region is

.. math::

    S(E) = \\int_{U(x) > E} \\kappa(x)\\,dx,
    \\qquad
    \\kappa = \\frac{\\sqrt{2 m^{*} (U - E)}}{\\hbar},

and the transmission probability is ``T = exp(-2 S)``. Energies are handled in
eV at the interface and converted internally; ``kappa`` is returned in
1/m so the action is dimensionless.

Reciprocity
-----------
Every channel builds its net flux as

.. math::

    J = C \\int T(E)\\,[\\,f_{\\rm left}(E) - f_{\\rm right}(E)\\,]\\,dE ,

with one transmission shared by both directions. At equilibrium the two
occupation factors are the same function of energy, so the integrand vanishes
pointwise and the net flux is **structurally** zero — not zero by cancellation
of two separately computed currents. :func:`reciprocal_net_flux` is the only
sanctioned way to combine a transmission with occupations, so no channel can
accidentally break that property.

Validity
--------
WKB is an asymptotic result: it is trustworthy when the action is large
compared with unity. :func:`wkb_validity` reports that, and channels are
expected to refuse (rather than silently return a number) when a barrier is
too thin or too shallow for the approximation to mean anything. The
local-wavelength condition is reported alongside it but is not a gate — see
:meth:`WKBValidity.valid` for why.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from perovskite_sim.constants import Q


# Reduced Planck constant [J s] and free-electron mass [kg]. physics/tunneling.py
# defines the same two constants for the scalar TFE model; they are repeated
# here so this module does not depend on the model D8 is replacing.
HBAR_J_S = 1.054571817e-34
ELECTRON_MASS_KG = 9.1093837015e-31

# Below this action the exponential is not an asymptotic statement about a
# barrier any more; channels treat it as "no meaningful barrier" and must say
# so rather than reporting a transmission near unity as a tunnelling result.
MINIMUM_MEANINGFUL_ACTION = 1.0e-3


class WKBTunnellingError(ValueError):
    """A barrier profile or effective mass was outside the WKB domain."""


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise WKBTunnellingError(f"{field} must be finite and positive")
    return result


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def decay_constant_per_m(
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
) -> np.ndarray:
    """Return ``kappa(x)`` in 1/m, zero where the carrier is not forbidden."""

    mass = _positive(effective_mass_rel, "effective_mass_rel")
    barrier = np.asarray(barrier_eV, dtype=float)
    if not np.all(np.isfinite(barrier)):
        raise WKBTunnellingError("barrier profile must be finite")
    if not math.isfinite(float(energy_eV)):
        raise WKBTunnellingError("energy_eV must be finite")
    forbidden = np.maximum(barrier - float(energy_eV), 0.0)
    return np.sqrt(2.0 * mass * ELECTRON_MASS_KG * forbidden * Q) / HBAR_J_S


def wkb_action(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
) -> float:
    """Integrate ``kappa`` over the classically forbidden region.

    The integrand is clipped at the turning points rather than the integration
    limits being solved for, so the result is a plain quadrature of a
    continuous, non-negative function. That is deliberately simple; the price
    is the square-root turning-point behaviour, whose mesh convergence the
    channel tests measure instead of assuming.
    """

    x = np.asarray(positions_m, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise WKBTunnellingError("positions_m must be a finite 1-D grid")
    if np.any(np.diff(x) <= 0.0):
        raise WKBTunnellingError("positions_m must strictly increase")
    kappa = decay_constant_per_m(barrier_eV, energy_eV, effective_mass_rel)
    if kappa.shape != x.shape:
        raise WKBTunnellingError("barrier profile must match positions_m")
    return float(np.trapezoid(kappa, x))


def wkb_transmission(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
) -> float:
    """Transmission ``exp(-2 S)`` for one energy, bounded by construction."""

    action = wkb_action(positions_m, barrier_eV, energy_eV, effective_mass_rel)
    # exp(-2S) with S >= 0 is already in (0, 1]; the guard is against a
    # non-finite action rather than against the range.
    if not math.isfinite(action) or action < 0.0:
        raise WKBTunnellingError("WKB action must be finite and non-negative")
    return math.exp(-2.0 * action)


def triangular_barrier_action(
    height_eV: float,
    width_m: float,
    effective_mass_rel: float,
    energy_eV: float = 0.0,
) -> float:
    """Closed-form action for a linear barrier, used as an exact reference.

    For ``U(x) = U0 (1 - x / w)`` and a carrier at ``E``,

    .. math::

        S = \\frac{2}{3}\\,\\frac{\\sqrt{2 m^{*}}}{\\hbar}\\,
            \\frac{(U_0 - E)^{3/2}}{U_0}\\,w ,

    which is the Fowler-Nordheim exponent. Channels compare their numerical
    quadrature against this to show the integrator is right before any device
    physics is layered on top.
    """

    height = _positive(height_eV, "height_eV")
    width = _positive(width_m, "width_m")
    mass = _positive(effective_mass_rel, "effective_mass_rel")
    energy = float(energy_eV)
    if not math.isfinite(energy) or energy >= height:
        raise WKBTunnellingError("energy_eV must be finite and below the barrier")
    prefactor = math.sqrt(2.0 * mass * ELECTRON_MASS_KG * Q) / HBAR_J_S
    return (2.0 / 3.0) * prefactor * (height - energy) ** 1.5 / height * width


@dataclass(frozen=True, slots=True)
class WKBValidity:
    """Diagnostics describing whether WKB means anything for this barrier."""

    action: float
    transmission: float
    maximum_wavelength_gradient: float
    forbidden_width_m: float
    meaningful_barrier: bool
    slowly_varying: bool

    @property
    def valid(self) -> bool:
        """Whether the WKB exponent is a meaningful statement here.

        Only ``meaningful_barrier`` gates this. ``slowly_varying`` is reported
        but deliberately NOT gated: for any smooth barrier the local-wavelength
        criterion fails as the turning point is approached, however large the
        action is, because ``1/kappa`` diverges there. That failure is the
        textbook one the Airy connection formula resolves and which the
        ``exp(-2S)`` normalisation already carries, so gating on it would
        reject every physical barrier while telling us nothing. It stays as a
        diagnostic because it does still flag an unresolved or artificially
        abrupt profile.
        """

        return bool(self.meaningful_barrier)


def wkb_validity(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
    *,
    maximum_wavelength_gradient: float = 1.0,
    turning_point_fraction: float = 0.1,
) -> WKBValidity:
    """Report the two conditions under which WKB is an approximation at all.

    ``meaningful_barrier`` fails for a barrier so thin or shallow that the
    action is negligible: the carrier is essentially free and calling the
    result "tunnelling" would be a category error rather than an inaccuracy.

    ``slowly_varying`` is the usual ``|d(1/kappa)/dx| << 1`` condition, but it
    is evaluated only where ``kappa`` is at least ``turning_point_fraction`` of
    its maximum. WKB always fails *at* a turning point, where ``kappa -> 0``
    and ``1/kappa`` diverges; that breakdown is handled by the standard Airy
    connection formula, which is already absorbed into the ``exp(-2S)`` form.
    Including the turning-point neighbourhood would therefore report every
    physical barrier as invalid and the diagnostic would carry no information.
    """

    x = np.asarray(positions_m, dtype=float)
    kappa = decay_constant_per_m(barrier_eV, energy_eV, effective_mass_rel)
    action = wkb_action(positions_m, barrier_eV, energy_eV, effective_mass_rel)
    forbidden = kappa > 0.0
    width = float(np.sum(np.diff(x)[forbidden[:-1] & forbidden[1:]]))
    gradient = 0.0
    peak = float(np.max(kappa)) if kappa.size else 0.0
    if peak > 0.0:
        fraction = float(turning_point_fraction)
        if not 0.0 < fraction < 1.0:
            raise WKBTunnellingError("turning_point_fraction must lie in (0, 1)")
        # Exclude the turning-point neighbourhood; see the docstring.
        core = kappa >= fraction * peak
        interior = core[:-1] & core[1:]
        if np.any(interior):
            wavelength = np.zeros_like(kappa)
            wavelength[core] = 1.0 / kappa[core]
            gradient = float(
                np.max(np.abs(np.diff(wavelength)[interior] / np.diff(x)[interior]))
            )
    return WKBValidity(
        action=action,
        transmission=math.exp(-2.0 * action),
        maximum_wavelength_gradient=gradient,
        forbidden_width_m=width,
        meaningful_barrier=bool(action >= MINIMUM_MEANINGFUL_ACTION),
        slowly_varying=bool(gradient <= float(maximum_wavelength_gradient)),
    )


@dataclass(frozen=True, slots=True)
class ReciprocalFlux:
    """One channel's net flux plus the pieces that make it reciprocal."""

    energies_eV: np.ndarray
    transmission: np.ndarray
    left_occupation: np.ndarray
    right_occupation: np.ndarray
    spectral_flux: np.ndarray
    net_flux_m2_s: float
    forward_flux_m2_s: float
    reverse_flux_m2_s: float


def reciprocal_net_flux(
    energies_eV: np.ndarray,
    transmission: np.ndarray,
    left_occupation: np.ndarray,
    right_occupation: np.ndarray,
    prefactor_m2_s_eV: float,
) -> ReciprocalFlux:
    """Combine one transmission with two occupations in detailed-balance form.

    The same ``transmission`` multiplies both directions, so equal occupations
    give an identically zero integrand at every energy. Channels must route
    every net current through this function; that is what makes "zero net
    tunnelling current at equilibrium" a structural property rather than a
    numerical coincidence.
    """

    energies = np.asarray(energies_eV, dtype=float)
    values = [
        np.asarray(item, dtype=float)
        for item in (transmission, left_occupation, right_occupation)
    ]
    if energies.ndim != 1 or energies.size < 2:
        raise WKBTunnellingError("energies_eV must be a 1-D grid with >= 2 points")
    if np.any(np.diff(energies) <= 0.0):
        raise WKBTunnellingError("energies_eV must strictly increase")
    for name, array in zip(
        ("transmission", "left_occupation", "right_occupation"), values, strict=True
    ):
        if array.shape != energies.shape:
            raise WKBTunnellingError(f"{name} must match energies_eV")
        if not np.all(np.isfinite(array)):
            raise WKBTunnellingError(f"{name} must be finite")
    transmission_values, left, right = values
    if np.any(transmission_values < 0.0) or np.any(transmission_values > 1.0):
        raise WKBTunnellingError("transmission must lie in [0, 1]")
    prefactor = float(prefactor_m2_s_eV)
    if not math.isfinite(prefactor) or prefactor < 0.0:
        raise WKBTunnellingError("prefactor_m2_s_eV must be finite and non-negative")
    spectral = prefactor * transmission_values * (left - right)
    return ReciprocalFlux(
        energies_eV=_readonly(energies),
        transmission=_readonly(transmission_values),
        left_occupation=_readonly(left),
        right_occupation=_readonly(right),
        spectral_flux=_readonly(spectral),
        net_flux_m2_s=float(np.trapezoid(spectral, energies)),
        forward_flux_m2_s=float(
            np.trapezoid(prefactor * transmission_values * left, energies)
        ),
        reverse_flux_m2_s=float(
            np.trapezoid(prefactor * transmission_values * right, energies)
        ),
    )


__all__ = [
    "ELECTRON_MASS_KG",
    "HBAR_J_S",
    "MINIMUM_MEANINGFUL_ACTION",
    "ReciprocalFlux",
    "WKBTunnellingError",
    "WKBValidity",
    "decay_constant_per_m",
    "reciprocal_net_flux",
    "triangular_barrier_action",
    "wkb_action",
    "wkb_transmission",
    "wkb_validity",
]

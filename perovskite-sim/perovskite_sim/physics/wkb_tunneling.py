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


def forbidden_run(
    barrier_eV: np.ndarray,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int] | None:
    """Return the connected forbidden interval that contains a given face.

    A device grid generally holds several barriers — a heterojunction spike,
    the band bending at each contact — and the classically forbidden set at a
    given energy is therefore disconnected. Integrating all of it would merge
    unrelated barriers into one fictitious path, which is both wrong and
    silently plausible. A channel bound to one interface must integrate only
    the run bracketing that interface, which is what this returns as an
    inclusive ``(low, high)`` node pair. ``None`` means neither node of the
    face is forbidden at this energy: the carrier is not blocked there.
    """

    barrier = np.asarray(barrier_eV, dtype=float)
    if barrier.ndim != 1 or barrier.size < 2:
        raise WKBTunnellingError("barrier profile must be a 1-D grid")
    face = int(anchor_face)
    if face < 0 or face >= barrier.size - 1:
        raise WKBTunnellingError("anchor_face is outside the transport faces")
    forbidden = barrier > float(energy_eV)
    if not (forbidden[face] or forbidden[face + 1]):
        return None
    start = face if forbidden[face] else face + 1
    low = start
    while low - 1 >= 0 and forbidden[low - 1]:
        low -= 1
    high = start
    while high + 1 < barrier.size and forbidden[high + 1]:
        high += 1
    return low, high


def turning_point_nodes(
    barrier_eV: np.ndarray,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int]:
    """The allowed nodes flanking the forbidden run that contains a face.

    These are the two places a tunnelling carrier actually is: it leaves the
    allowed region at one turning point and arrives at the other. Their
    occupations are what drive the channel.

    Why not the two nodes of the anchor face: that difference is taken across
    ONE grid cell, so it shrinks with the mesh and the resulting flux vanishes
    as ``dx -> 0`` — a measured factor of ~2 per grid doubling, i.e. the
    channel current would be an artifact of the discretisation rather than a
    property of the barrier. Turning points converge to fixed physical
    positions instead, which is what makes the flux mesh-convergent.

    When the energy is above the barrier the carrier is not blocked; the two
    nodes of the face are returned so the caller still gets a well-defined
    pair, and the transmission is 1 there anyway.
    """

    barrier = np.asarray(barrier_eV, dtype=float)
    face = int(anchor_face)
    bounds = forbidden_run(barrier, energy_eV, face)
    if bounds is None:
        return face, face + 1
    low, high = bounds
    return max(low - 1, 0), min(high + 1, barrier.size - 1)


def two_band_turning_point_nodes(
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int]:
    """The same, for the in-gap run a band-to-band carrier crosses.

    The forbidden set here is "inside the gap" rather than "above one band
    edge", so the run is bracketed by turning points on opposite bands.
    """

    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    if conduction.shape != valence.shape:
        raise WKBTunnellingError("band edges must share one grid")
    face = int(anchor_face)
    if face < 0 or face >= conduction.size - 1:
        raise WKBTunnellingError("anchor_face is outside the transport faces")
    energy = float(energy_eV)
    inside = (conduction > energy) & (valence < energy)
    if not (inside[face] or inside[face + 1]):
        return face, face + 1
    start = face if inside[face] else face + 1
    low = start
    while low - 1 >= 0 and inside[low - 1]:
        low -= 1
    high = start
    while high + 1 < conduction.size and inside[high + 1]:
        high += 1
    return max(low - 1, 0), min(high + 1, conduction.size - 1)


def _interpolate_at_crossing(
    barrier: np.ndarray,
    profile: np.ndarray,
    energy: float,
    allowed: int,
    forbidden: int,
) -> float:
    """Read ``profile`` where the barrier actually crosses ``energy``.

    The turning point is defined by ``U(x) = E`` and generally falls BETWEEN
    two nodes. Snapping it to the nearer node costs O(h) in position and
    therefore O(h) in whatever is read there, which drags the whole channel
    flux down to first-order convergence. Linear interpolation restores the
    second order the rest of the discretisation has.
    """

    low_value = float(barrier[allowed])
    high_value = float(barrier[forbidden])
    span = high_value - low_value
    if not np.isfinite(span) or span == 0.0:
        return float(profile[allowed])
    fraction = (float(energy) - low_value) / span
    fraction = min(max(fraction, 0.0), 1.0)
    return float(
        profile[allowed] + fraction * (profile[forbidden] - profile[allowed])
    )


def turning_point_levels(
    barrier_eV: np.ndarray,
    profile_eV: np.ndarray,
    energy_eV: float,
    anchor_face: int,
) -> tuple[float, float]:
    """``profile_eV`` evaluated at the two true turning points of the run.

    Interpolated rather than snapped to nodes; see
    :func:`_interpolate_at_crossing` for why that matters.
    """

    barrier = np.asarray(barrier_eV, dtype=float)
    profile = np.asarray(profile_eV, dtype=float)
    if barrier.shape != profile.shape:
        raise WKBTunnellingError("profile must share the barrier grid")
    face = int(anchor_face)
    bounds = forbidden_run(barrier, energy_eV, face)
    if bounds is None:
        return float(profile[face]), float(profile[face + 1])
    low, high = bounds
    energy = float(energy_eV)
    left = (
        _interpolate_at_crossing(barrier, profile, energy, low - 1, low)
        if low - 1 >= 0
        else float(profile[low])
    )
    right = (
        _interpolate_at_crossing(barrier, profile, energy, high + 1, high)
        if high + 1 < barrier.size
        else float(profile[high])
    )
    return left, right


def two_band_turning_point_levels(
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    profile_eV: np.ndarray,
    energy_eV: float,
    anchor_face: int,
) -> tuple[float, float]:
    """The same for the in-gap run, whose turning points sit on opposite bands.

    The left crossing is where the carrier leaves the conduction band
    (``E_C = E``) and the right is where it enters the valence band
    (``E_V = E``), so each side interpolates against its own edge.
    """

    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    profile = np.asarray(profile_eV, dtype=float)
    if not (conduction.shape == valence.shape == profile.shape):
        raise WKBTunnellingError("band edges and profile must share one grid")
    low, high = two_band_turning_point_nodes(
        conduction, valence, energy_eV, anchor_face
    )
    energy = float(energy_eV)
    # `low`/`high` are already the allowed nodes flanking the in-gap run.
    left = (
        _interpolate_at_crossing(conduction, profile, energy, low, min(low + 1, conduction.size - 1))
        if low + 1 < conduction.size
        else float(profile[low])
    )
    right = (
        _interpolate_at_crossing(valence, profile, energy, high, max(high - 1, 0))
        if high - 1 >= 0
        else float(profile[high])
    )
    return left, right


def windowed_wkb_action(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
    anchor_face: int,
) -> float:
    """WKB action over the forbidden run containing ``anchor_face`` only."""

    bounds = forbidden_run(barrier_eV, energy_eV, anchor_face)
    if bounds is None:
        return 0.0
    low, high = bounds
    x = np.asarray(positions_m, dtype=float)
    if high - low < 1:
        # A single forbidden node carries no width on this grid.
        return 0.0
    return wkb_action(
        x[low : high + 1],
        np.asarray(barrier_eV, dtype=float)[low : high + 1],
        energy_eV,
        effective_mass_rel,
    )


def two_band_decay_constant_per_m(
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    energy_eV: float,
    reduced_effective_mass_rel: float,
) -> np.ndarray:
    """Kane two-band decay constant for a carrier inside the forbidden gap.

    The single-band form ``sqrt(2 m (U - E))`` has no meaning for band-to-band
    tunnelling: the particle is not blocked by one band edge but by the gap,
    and its turning points are set by *different* bands (it leaves the valence
    band where ``E = E_V`` and enters the conduction band where ``E = E_C``).
    Using ``E_C`` alone as the barrier would place both turning points on the
    same edge and integrate the wrong region entirely.

    The two-band dispersion gives

    ``kappa(x) = sqrt(2 m_r (E_C - E)(E - E_V) / E_g) / h_bar``

    which vanishes at *both* turning points, as a decay constant must. Under a
    uniform field this integrates in closed form to the textbook Zener result
    ``T = exp(-pi sqrt(m_r) E_g^{3/2} / (2 sqrt(2) h_bar q F))`` — the identity
    the tests check, so the prefactor here is pinned rather than asserted.
    """

    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    if conduction.shape != valence.shape:
        raise WKBTunnellingError("band edges must share one grid")
    mass = _positive(reduced_effective_mass_rel, "reduced_effective_mass_rel")
    gap = conduction - valence
    if np.any(gap <= 0.0):
        raise WKBTunnellingError("the gap must be positive on the whole grid")
    energy = float(energy_eV)
    inside = (conduction > energy) & (valence < energy)
    kappa = np.zeros_like(conduction)
    if not np.any(inside):
        return kappa
    numerator = (
        (conduction[inside] - energy) * (energy - valence[inside]) / gap[inside]
    )
    kappa[inside] = (
        np.sqrt(2.0 * mass * ELECTRON_MASS_KG * numerator * Q)
        / HBAR_J_S
    )
    return kappa


def two_band_action(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    energy_eV: float,
    reduced_effective_mass_rel: float,
    anchor_face: int,
) -> float:
    """Kane action over the in-gap run that contains ``anchor_face``.

    The forbidden set here is "inside the gap", not "above one band edge", so
    the run is bracketed by the two turning points on opposite bands.
    """

    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    x = np.asarray(positions_m, dtype=float)
    if conduction.shape != x.shape or valence.shape != x.shape:
        raise WKBTunnellingError("band edges must match positions_m")
    if x.ndim != 1 or x.size < 2:
        raise WKBTunnellingError("positions_m must be a 1-D grid")
    if np.any(np.diff(x) <= 0.0):
        raise WKBTunnellingError("positions_m must strictly increase")
    face = int(anchor_face)
    if face < 0 or face >= x.size - 1:
        raise WKBTunnellingError("anchor_face is outside the transport faces")
    energy = float(energy_eV)
    inside = (conduction > energy) & (valence < energy)
    if not (inside[face] or inside[face + 1]):
        return 0.0
    start = face if inside[face] else face + 1
    low = start
    while low - 1 >= 0 and inside[low - 1]:
        low -= 1
    high = start
    while high + 1 < x.size and inside[high + 1]:
        high += 1
    if high - low < 1:
        return 0.0
    kappa = two_band_decay_constant_per_m(
        conduction[low : high + 1],
        valence[low : high + 1],
        energy,
        reduced_effective_mass_rel,
    )
    return float(np.trapezoid(kappa, x[low : high + 1]))


def two_band_transmission(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    energy_eV: float,
    reduced_effective_mass_rel: float,
    anchor_face: int,
) -> float:
    """Two-band transmission ``exp(-2 S)`` for band-to-band tunnelling."""

    return math.exp(
        -2.0
        * two_band_action(
            positions_m,
            conduction_edge_eV,
            valence_edge_eV,
            energy_eV,
            reduced_effective_mass_rel,
            anchor_face,
        )
    )


def kane_uniform_field_action(
    band_gap_eV: float,
    field_V_m: float,
    reduced_effective_mass_rel: float,
) -> float:
    """Closed-form Kane action for a uniform field, used to check the quadrature.

    ``S = pi sqrt(2 m_r) E_g^{3/2} / (8 h_bar q F)``, so ``exp(-2 S)`` is the
    textbook Zener exponent.
    """

    gap = _positive(band_gap_eV, "band_gap_eV")
    field = _positive(field_V_m, "field_V_m")
    mass = _positive(reduced_effective_mass_rel, "reduced_effective_mass_rel")
    gap_J = gap * Q
    return (
        math.pi
        * math.sqrt(2.0 * mass * ELECTRON_MASS_KG)
        * gap_J**1.5
        / (8.0 * HBAR_J_S * Q * field)
    )


def windowed_wkb_transmission(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    energy_eV: float,
    effective_mass_rel: float,
    anchor_face: int,
) -> float:
    """Transmission through the forbidden run containing ``anchor_face``."""

    action = windowed_wkb_action(
        positions_m, barrier_eV, energy_eV, effective_mass_rel, anchor_face
    )
    return math.exp(-2.0 * action)


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

    kappa = decay_constant_per_m(barrier_eV, energy_eV, effective_mass_rel)
    action = wkb_action(positions_m, barrier_eV, energy_eV, effective_mass_rel)
    return _validity_from_kappa(
        positions_m,
        kappa,
        action,
        maximum_wavelength_gradient=maximum_wavelength_gradient,
        turning_point_fraction=turning_point_fraction,
    )


def _validity_from_kappa(
    positions_m: np.ndarray,
    kappa: np.ndarray,
    action: float,
    *,
    maximum_wavelength_gradient: float,
    turning_point_fraction: float,
) -> WKBValidity:
    """Assemble the diagnostics from an already-computed decay constant.

    Shared by the single-band and two-band reports so the two cannot drift:
    only the decay constant differs between them, never the criteria.
    """

    x = np.asarray(positions_m, dtype=float)
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
        action=float(action),
        transmission=math.exp(-2.0 * float(action)),
        maximum_wavelength_gradient=gradient,
        forbidden_width_m=width,
        meaningful_barrier=bool(action >= MINIMUM_MEANINGFUL_ACTION),
        slowly_varying=bool(gradient <= float(maximum_wavelength_gradient)),
    )


def two_band_validity(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    energy_eV: float,
    reduced_effective_mass_rel: float,
    anchor_face: int,
    *,
    maximum_wavelength_gradient: float = 1.0,
    turning_point_fraction: float = 0.1,
) -> WKBValidity:
    """The same diagnostics for the two-band (band-to-band) decay constant.

    Reporting the single-band validity for a Zener channel would describe a
    barrier that channel never crosses, so the action and the forbidden width
    would both be wrong — hence a separate entry point rather than a shared
    one with a different barrier argument.
    """

    kappa = two_band_decay_constant_per_m(
        conduction_edge_eV,
        valence_edge_eV,
        energy_eV,
        reduced_effective_mass_rel,
    )
    action = two_band_action(
        positions_m,
        conduction_edge_eV,
        valence_edge_eV,
        energy_eV,
        reduced_effective_mass_rel,
        anchor_face,
    )
    return _validity_from_kappa(
        positions_m,
        kappa,
        action,
        maximum_wavelength_gradient=maximum_wavelength_gradient,
        turning_point_fraction=turning_point_fraction,
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
    "turning_point_levels",
    "two_band_turning_point_levels",
    "turning_point_nodes",
    "two_band_turning_point_nodes",
    "two_band_validity",
    "kane_uniform_field_action",
    "two_band_action",
    "two_band_decay_constant_per_m",
    "two_band_transmission",
    "ELECTRON_MASS_KG",
    "HBAR_J_S",
    "MINIMUM_MEANINGFUL_ACTION",
    "ReciprocalFlux",
    "WKBTunnellingError",
    "WKBValidity",
    "decay_constant_per_m",
    "forbidden_run",
    "reciprocal_net_flux",
    "triangular_barrier_action",
    "wkb_action",
    "wkb_transmission",
    "wkb_validity",
    "windowed_wkb_action",
    "windowed_wkb_transmission",
]

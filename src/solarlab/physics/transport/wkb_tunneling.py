"""One-dimensional WKB barrier integrals, analytical tangents, and TFE scaling.

Provides two distinct, explicitly separated quantum-tunnelling primitives:

1. ``tfe_gamma``: static Padovani-Stratton thermionic-field-emission (TFE)
   Richardson-constant enhancement ``Gamma >= 1`` built once from geometry and
   doping (never representing nonlocal WKB barrier transmission).
2. Resolved 1D WKB and two-band Kane action / transmission / turning-point /
   reciprocal-flux kernels, together with cancellation-free analytical
   derivatives ``piecewise_linear_wkb_segment_action_derivatives`` of every
   piecewise-linear segment action wrt the nodal barrier potential.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C

__all__ = [
    "ELECTRON_MASS_KG",
    "HBAR_J_S",
    "MINIMUM_MEANINGFUL_ACTION",
    "VACUUM_PERMITTIVITY_F_M",
    "ReciprocalFlux",
    "WKBSegmentActionDerivatives",
    "WKBTunnellingError",
    "WKBValidity",
    "decay_constant_per_m",
    "forbidden_run",
    "kane_uniform_field_action",
    "piecewise_linear_wkb_segment_action_derivatives",
    "piecewise_linear_wkb_segment_actions",
    "reciprocal_net_flux",
    "tfe_gamma",
    "triangular_barrier_action",
    "turning_point_levels",
    "turning_point_nodes",
    "two_band_action",
    "two_band_decay_constant_per_m",
    "two_band_transmission",
    "two_band_turning_point_levels",
    "two_band_turning_point_nodes",
    "two_band_validity",
    "windowed_wkb_action",
    "windowed_wkb_transmission",
    "wkb_action",
    "wkb_transmission",
    "wkb_validity",
]

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]

Q: float = ELEMENTARY_CHARGE_C
HBAR_J_S: float = 1.054571817e-34
ELECTRON_MASS_KG: float = 9.1093837015e-31
VACUUM_PERMITTIVITY_F_M: float = 8.854187817e-12
MINIMUM_MEANINGFUL_ACTION: float = 1.0e-3


class WKBTunnellingError(ValueError):
    """A barrier profile or effective mass was outside the WKB domain."""


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a real number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise WKBTunnellingError(f"{field} must be finite and positive")
    return result


def _coth_positive(z: float) -> float:
    if z > 40.0:
        return 1.0
    if z < 1.0e-8:
        return 1.0 / z
    return 1.0 / math.tanh(z)


def tfe_gamma(
    delta_E_eV: float,
    N_iface: float,
    m_eff_rel: float,
    eps_r: float,
    V_T: float,
) -> float:
    """Static Padovani-Stratton TFE Richardson enhancement ``Gamma >= 1``."""
    d_e = abs(float(delta_E_eV))
    n_if = float(N_iface)
    m_rel = float(m_eff_rel)
    eps = float(eps_r)
    vt = float(V_T)
    if n_if <= 0.0 or m_rel <= 0.0 or eps <= 0.0 or d_e <= 0.0 or vt <= 0.0:
        return 1.0
    eps_s = eps * VACUUM_PERMITTIVITY_F_M
    e00_j = (Q * HBAR_J_S / 2.0) * math.sqrt(n_if / (m_rel * ELECTRON_MASS_KG * eps_s))
    e00 = e00_j / Q
    e0 = e00 * _coth_positive(e00 / vt)
    delta_tun = d_e * (1.0 - vt / e0)
    if delta_tun <= 0.0:
        return 1.0
    if delta_tun >= d_e:
        delta_tun = d_e * (1.0 - 1.0e-9)
    return math.exp(delta_tun / vt)


def decay_constant_per_m(
    barrier_eV: ArrayLike,
    energy_eV: float,
    effective_mass_rel: float,
) -> Vector:
    """Return ``kappa(x)`` in 1/m, zero where the carrier is not forbidden."""
    mass = _positive(effective_mass_rel, "effective_mass_rel")
    barrier = np.asarray(barrier_eV, dtype=float)
    if not np.all(np.isfinite(barrier)):
        raise WKBTunnellingError("barrier profile must be finite")
    if not math.isfinite(float(energy_eV)):
        raise WKBTunnellingError("energy_eV must be finite")
    forbidden = np.maximum(barrier - float(energy_eV), 0.0)
    return np.sqrt(2.0 * mass * ELECTRON_MASS_KG * forbidden * Q) / HBAR_J_S


def piecewise_linear_wkb_segment_actions(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    energies_eV: ArrayLike,
    effective_mass_rel: float,
) -> Matrix:
    """Exact actions per energy/segment for the linearly interpolated band."""
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(barrier_eV, dtype=float)
    energies = np.asarray(energies_eV, dtype=float)
    if (
        x.ndim != 1
        or x.size < 2
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0.0)
        or not np.all(np.isfinite(np.diff(x)))
    ):
        raise WKBTunnellingError("positions_m must be finite and strictly increase on a 1-D grid")
    if barrier.shape != x.shape or not np.all(np.isfinite(barrier)):
        raise WKBTunnellingError("finite barrier profile must match positions_m")
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise WKBTunnellingError("energies_eV must be a finite 1-D vector")
    mass = _positive(effective_mass_rel, "effective_mass_rel")
    left = barrier[:-1][None, :] - energies[:, None]
    right = barrier[1:][None, :] - energies[:, None]
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise WKBTunnellingError("barrier-energy differences must be finite")
    root_left = np.sqrt(np.maximum(left, 0.0))
    root_right = np.sqrt(np.maximum(right, 0.0))
    root_sum = root_left + root_right
    ratio = np.divide(root_right, root_sum, out=np.zeros_like(root_sum), where=root_sum > 0.0)
    mean_root = (2.0 / 3.0) * (root_sum - root_left * ratio)
    crossing = ((left < 0.0) & (right > 0.0)) | ((left > 0.0) & (right < 0.0))
    positive = np.maximum(left[crossing], right[crossing])
    negative = -np.minimum(left[crossing], right[crossing])
    with np.errstate(over="ignore"):
        forbidden_fraction = 1.0 / (1.0 + negative / positive)
    mean_root[crossing] *= forbidden_fraction
    prefactor = math.sqrt(2.0 * mass * ELECTRON_MASS_KG * Q) / HBAR_J_S
    return prefactor * mean_root * np.diff(x)[None, :]


@dataclass(frozen=True)
class WKBSegmentActionDerivatives:
    """Segment actions ``I_{k,m}`` and exact partials ``dI_{k,m}/dU_m``, ``dI_{k,m}/dU_{m+1}`` [eV^-1]."""

    segment_actions: Matrix
    d_action_d_left_barrier_eV: Matrix
    d_action_d_right_barrier_eV: Matrix


def piecewise_linear_wkb_segment_action_derivatives(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    energies_eV: ArrayLike,
    effective_mass_rel: float,
) -> WKBSegmentActionDerivatives:
    """Exact cancellation-free segment actions and derivatives wrt nodal ``barrier_eV``."""
    actions = piecewise_linear_wkb_segment_actions(
        positions_m, barrier_eV, energies_eV, effective_mass_rel
    )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(barrier_eV, dtype=float)
    energies = np.asarray(energies_eV, dtype=float)
    mass = _positive(effective_mass_rel, "effective_mass_rel")

    prefactor = math.sqrt(2.0 * mass * ELECTRON_MASS_KG * Q) / HBAR_J_S
    scale = np.broadcast_to((prefactor * np.diff(x))[None, :], actions.shape)
    u_left = barrier[:-1][None, :] - energies[:, None]
    u_right = barrier[1:][None, :] - energies[:, None]

    d_left: Matrix = np.zeros_like(actions)
    d_right: Matrix = np.zeros_like(actions)

    both = (u_left > 0.0) & (u_right > 0.0)
    if np.any(both):
        a = np.sqrt(u_left[both])
        b = np.sqrt(u_right[both])
        apb = a + b
        inv_denom = 1.0 / (3.0 * apb * apb)
        d_left[both] = scale[both] * (a + 2.0 * b) * inv_denom
        d_right[both] = scale[both] * (b + 2.0 * a) * inv_denom

    left_cross = (u_left <= 0.0) & (u_right > 0.0)
    if np.any(left_cross):
        l_val = u_left[left_cross]
        r_val = u_right[left_cross]
        b = np.sqrt(r_val)
        rml = r_val - l_val
        inv_sq = 1.0 / (3.0 * rml * rml)
        d_left[left_cross] = 2.0 * scale[left_cross] * (r_val * b) * inv_sq
        d_right[left_cross] = scale[left_cross] * b * (r_val - 3.0 * l_val) * inv_sq

    right_cross = (u_left > 0.0) & (u_right <= 0.0)
    if np.any(right_cross):
        l_val = u_left[right_cross]
        r_val = u_right[right_cross]
        a = np.sqrt(l_val)
        lmr = l_val - r_val
        inv_sq = 1.0 / (3.0 * lmr * lmr)
        d_left[right_cross] = scale[right_cross] * a * (l_val - 3.0 * r_val) * inv_sq
        d_right[right_cross] = 2.0 * scale[right_cross] * (l_val * a) * inv_sq

    return WKBSegmentActionDerivatives(actions, d_left, d_right)


def wkb_action(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    energy_eV: float,
    effective_mass_rel: float,
) -> float:
    """Integrate ``kappa`` over the classically forbidden region."""
    actions = piecewise_linear_wkb_segment_actions(
        positions_m, barrier_eV, np.array([float(energy_eV)]), effective_mass_rel
    )
    return float(np.sum(actions[0]))


def forbidden_run(
    barrier_eV: ArrayLike,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int] | None:
    """Return the connected forbidden interval ``(low, high)`` containing ``anchor_face``."""
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
    barrier_eV: ArrayLike,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int]:
    """Allowed nodes flanking the forbidden run containing ``anchor_face``."""
    barrier = np.asarray(barrier_eV, dtype=float)
    face = int(anchor_face)
    bounds = forbidden_run(barrier, energy_eV, face)
    if bounds is None:
        return face, face + 1
    low, high = bounds
    return max(low - 1, 0), min(high + 1, barrier.size - 1)


def two_band_turning_point_nodes(
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    energy_eV: float,
    anchor_face: int,
) -> tuple[int, int]:
    """Allowed nodes flanking the in-gap run containing ``anchor_face``."""
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
    barrier: Vector,
    profile: Vector,
    energy: float,
    allowed: int,
    forbidden: int,
) -> float:
    low_value = float(barrier[allowed])
    high_value = float(barrier[forbidden])
    span = high_value - low_value
    if not math.isfinite(span) or span == 0.0:
        return float(profile[allowed])
    fraction = min(max((float(energy) - low_value) / span, 0.0), 1.0)
    return float(profile[allowed] + fraction * (profile[forbidden] - profile[allowed]))


def turning_point_levels(
    barrier_eV: ArrayLike,
    profile_eV: ArrayLike,
    energy_eV: float,
    anchor_face: int,
) -> tuple[float, float]:
    """``profile_eV`` linearly interpolated at the two turning points of ``anchor_face``."""
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
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    profile_eV: ArrayLike,
    energy_eV: float,
    anchor_face: int,
) -> tuple[float, float]:
    """``profile_eV`` interpolated at the opposite-band turning points of ``anchor_face``."""
    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    profile = np.asarray(profile_eV, dtype=float)
    if not (conduction.shape == valence.shape == profile.shape):
        raise WKBTunnellingError("band edges and profile must share one grid")
    low, high = two_band_turning_point_nodes(conduction, valence, energy_eV, anchor_face)
    energy = float(energy_eV)
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
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
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
    low = max(low - 1, 0)
    high = min(high + 1, x.size - 1)
    return wkb_action(
        x[low : high + 1],
        np.asarray(barrier_eV, dtype=float)[low : high + 1],
        energy_eV,
        effective_mass_rel,
    )


def two_band_decay_constant_per_m(
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    energy_eV: float,
    reduced_effective_mass_rel: float,
) -> Vector:
    """Kane two-band decay constant inside the forbidden gap."""
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
    numerator = (conduction[inside] - energy) * (energy - valence[inside]) / gap[inside]
    kappa[inside] = np.sqrt(2.0 * mass * ELECTRON_MASS_KG * numerator * Q) / HBAR_J_S
    return kappa


def two_band_action(
    positions_m: ArrayLike,
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    energy_eV: float,
    reduced_effective_mass_rel: float,
    anchor_face: int,
) -> float:
    """Kane action over the in-gap run containing ``anchor_face``."""
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
    positions_m: ArrayLike,
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
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
    """Closed-form Kane action ``pi sqrt(2 m_r) E_g^{3/2} / (8 hbar q F)``."""
    gap = _positive(band_gap_eV, "band_gap_eV")
    field = _positive(field_V_m, "field_V_m")
    mass = _positive(reduced_effective_mass_rel, "reduced_effective_mass_rel")
    gap_j = gap * Q
    return (
        math.pi
        * math.sqrt(2.0 * mass * ELECTRON_MASS_KG)
        * gap_j**1.5
        / (8.0 * HBAR_J_S * Q * field)
    )


def windowed_wkb_transmission(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
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
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    energy_eV: float,
    effective_mass_rel: float,
) -> float:
    """Transmission ``exp(-2 S)`` for one energy, bounded in ``(0, 1]``."""
    action = wkb_action(positions_m, barrier_eV, energy_eV, effective_mass_rel)
    if not math.isfinite(action) or action < 0.0:
        raise WKBTunnellingError("WKB action must be finite and non-negative")
    return math.exp(-2.0 * action)


def triangular_barrier_action(
    height_eV: float,
    width_m: float,
    effective_mass_rel: float,
    energy_eV: float = 0.0,
) -> float:
    """Closed-form Fowler-Nordheim action for ``U(x) = U0 (1 - x / w)``."""
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
    """Diagnostics describing whether WKB is meaningful for this barrier."""

    action: float
    transmission: float
    maximum_wavelength_gradient: float
    forbidden_width_m: float
    meaningful_barrier: bool
    slowly_varying: bool

    @property
    def valid(self) -> bool:
        return bool(self.meaningful_barrier)


def _validity_from_kappa(
    positions_m: ArrayLike,
    kappa: Vector,
    action: float,
    *,
    maximum_wavelength_gradient: float,
    turning_point_fraction: float,
) -> WKBValidity:
    x = np.asarray(positions_m, dtype=float)
    forbidden = kappa > 0.0
    width = float(np.sum(np.diff(x)[forbidden[:-1] & forbidden[1:]]))
    gradient = 0.0
    peak = float(np.max(kappa)) if kappa.size else 0.0
    if peak > 0.0:
        fraction = float(turning_point_fraction)
        if not 0.0 < fraction < 1.0:
            raise WKBTunnellingError("turning_point_fraction must lie in (0, 1)")
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


def wkb_validity(
    positions_m: ArrayLike,
    barrier_eV: ArrayLike,
    energy_eV: float,
    effective_mass_rel: float,
    *,
    maximum_wavelength_gradient: float = 1.0,
    turning_point_fraction: float = 0.1,
) -> WKBValidity:
    """Semiclassical validity diagnostics for a single-band barrier."""
    kappa = decay_constant_per_m(barrier_eV, energy_eV, effective_mass_rel)
    action = wkb_action(positions_m, barrier_eV, energy_eV, effective_mass_rel)
    return _validity_from_kappa(
        positions_m,
        kappa,
        action,
        maximum_wavelength_gradient=maximum_wavelength_gradient,
        turning_point_fraction=turning_point_fraction,
    )


def two_band_validity(
    positions_m: ArrayLike,
    conduction_edge_eV: ArrayLike,
    valence_edge_eV: ArrayLike,
    energy_eV: float,
    reduced_effective_mass_rel: float,
    anchor_face: int,
    *,
    maximum_wavelength_gradient: float = 1.0,
    turning_point_fraction: float = 0.1,
) -> WKBValidity:
    """Semiclassical validity diagnostics for the two-band Kane decay constant."""
    kappa = two_band_decay_constant_per_m(
        conduction_edge_eV, valence_edge_eV, energy_eV, reduced_effective_mass_rel
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

    energies_eV: Vector
    transmission: Vector
    left_occupation: Vector
    right_occupation: Vector
    spectral_flux: Vector
    net_flux_m2_s: float
    forward_flux_m2_s: float
    reverse_flux_m2_s: float


def reciprocal_net_flux(
    energies_eV: ArrayLike,
    transmission: ArrayLike,
    left_occupation: ArrayLike,
    right_occupation: ArrayLike,
    prefactor_m2_s_eV: float,
    *,
    quadrature_weights_eV: ArrayLike | None = None,
) -> ReciprocalFlux:
    """Combine one transmission with two occupations in detailed-balance form."""
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
    if quadrature_weights_eV is None:
        def integrate(arr: Vector) -> float:
            return float(np.trapezoid(arr, energies))
    else:
        weights = np.asarray(quadrature_weights_eV, dtype=float)
        if weights.shape != energies.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise WKBTunnellingError(
                "quadrature weights must be finite, nonnegative and match energies"
            )

        def integrate(arr: Vector) -> float:
            return float(np.dot(weights, arr))

    spectral = prefactor * transmission_values * (left - right)
    return ReciprocalFlux(
        energies_eV=frozen_array(energies),
        transmission=frozen_array(transmission_values),
        left_occupation=frozen_array(left),
        right_occupation=frozen_array(right),
        spectral_flux=frozen_array(spectral),
        net_flux_m2_s=integrate(spectral),
        forward_flux_m2_s=integrate(prefactor * transmission_values * left),
        reverse_flux_m2_s=integrate(prefactor * transmission_values * right),
    )

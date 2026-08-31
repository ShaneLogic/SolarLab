"""The four D8 tunnelling channels, implemented separately.

Each channel below computes its own transmission from an actual barrier
profile and turns it into a net flux through
:func:`~perovskite_sim.physics.wkb_tunneling.reciprocal_net_flux`, so every
channel is zero at equilibrium by construction and none of them is obtained by
rescaling another quantity. That separation is the point: the D8 exit
condition forbids one scalar enhancement standing in for four channels.

Band edges follow the repository convention, in eV:

    E_C(x) = -(phi(x) + chi(x)),    E_V(x) = -(phi(x) + chi(x) + Eg(x))

An electron in the conduction band is forbidden where ``E < E_C``; a hole in
the valence band is forbidden where ``E > E_V``, which is why the hole barrier
is built as ``+E_V`` (the particle barrier for a positive charge) before the
shared WKB machinery is applied.

Declared limitations
--------------------
* Prefactors are supply-function estimates, not fitted to any reference. The
  transmission, the reciprocity and the band-to-band exponent are asserted by
  tests; absolute channel magnitudes are explicitly not validated against
  SCAPS here.
* Every channel is anchored to one face and integrates only the connected
  forbidden run containing it, because a device grid holds several barriers at
  once and merging them would be wrong in a way that still looks plausible.
* The band-to-band channel uses the two-band (Kane) exponent, not the
  single-band one: its two turning points sit on different bands, which the
  single-band form cannot express at any prefactor. Pinned against the
  closed-form uniform-field Zener result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from perovskite_sim.models.tunneling_channels import (
    BandToBandTunnellingChannel,
    ContactTunnellingChannel,
    InterfaceDefectAssistedTunnellingChannel,
    IntrabandTunnellingChannel,
)
from perovskite_sim.physics.wkb_tunneling import (
    WKBTunnellingError,
    reciprocal_net_flux,
    two_band_transmission,
    two_band_validity,
    wkb_validity,
    windowed_wkb_transmission,
)


class TunnellingChannelError(RuntimeError):
    """A channel was asked for a flux outside its declared domain."""


def _fermi(
    energy_eV: np.ndarray, level_eV: float, thermal_voltage_V: float
) -> np.ndarray:
    """Fermi-Dirac occupation, written to avoid overflow on either tail."""

    reduced = (np.asarray(energy_eV, dtype=float) - float(level_eV)) / float(
        thermal_voltage_V
    )
    out = np.empty_like(reduced)
    positive = reduced >= 0.0
    out[positive] = np.exp(-reduced[positive]) / (1.0 + np.exp(-reduced[positive]))
    out[~positive] = 1.0 / (1.0 + np.exp(reduced[~positive]))
    return out


def conduction_band_eV(potential_V: np.ndarray, affinity_eV: np.ndarray) -> np.ndarray:
    """``E_C = -(phi + chi)`` on the repository's sign convention."""

    return -(
        np.asarray(potential_V, dtype=float) + np.asarray(affinity_eV, dtype=float)
    )


def valence_band_eV(
    potential_V: np.ndarray,
    affinity_eV: np.ndarray,
    band_gap_eV: np.ndarray,
) -> np.ndarray:
    """``E_V = -(phi + chi + Eg)``."""

    return -(
        np.asarray(potential_V, dtype=float)
        + np.asarray(affinity_eV, dtype=float)
        + np.asarray(band_gap_eV, dtype=float)
    )



def local_barrier_window(
    barrier_eV: np.ndarray,
    anchor_face: int,
    *,
    one_sided: bool = False,
) -> tuple[float, float]:
    """Peak and base of the barrier feature that sits on ``anchor_face``.

    Walking downhill from the face in both directions finds the local minima
    that bound the feature; the higher of those two minima is the energy above
    which a carrier is no longer blocked. Using the *local* feature rather
    than the whole profile is what keeps a heterojunction spike from being
    confused with the device-wide band bending between the contacts.

    ``one_sided`` is for a barrier whose peak sits at a grid endpoint, which
    is what a Schottky contact looks like: the barrier is highest *at* the
    metal and decays into the semiconductor. There is no second minimum to
    take the higher of, so the base is the one on the interior side. Applying
    the two-sided rule to that shape would return ``peak == base`` and reject
    every real contact barrier, so the caller must say which shape it has
    rather than have this function guess from the data.
    """

    barrier = np.asarray(barrier_eV, dtype=float)
    face = int(anchor_face)
    if face < 0 or face >= barrier.size - 1:
        raise WKBTunnellingError("anchor_face is outside the transport faces")
    peak_index = face if barrier[face] >= barrier[face + 1] else face + 1
    # Climb to the local maximum first: the face may sit on the flank.
    while (
        peak_index + 1 < barrier.size and barrier[peak_index + 1] > barrier[peak_index]
    ):
        peak_index += 1
    while peak_index - 1 >= 0 and barrier[peak_index - 1] > barrier[peak_index]:
        peak_index -= 1
    left = peak_index
    while left - 1 >= 0 and barrier[left - 1] < barrier[left]:
        left -= 1
    right = peak_index
    while right + 1 < barrier.size and barrier[right + 1] < barrier[right]:
        right += 1
    peak = float(barrier[peak_index])
    if not one_sided:
        return peak, float(max(barrier[left], barrier[right]))
    # The interior side is whichever direction actually descends; if the peak
    # is interior after all, both do, and the deeper minimum is the reach of
    # the barrier the carrier has to cross.
    return peak, float(min(barrier[left], barrier[right]))


@dataclass(frozen=True, slots=True)
class ChannelFlux:
    """One channel's net flux plus everything needed to audit it."""

    channel: str
    net_flux_m2_s: float
    forward_flux_m2_s: float
    reverse_flux_m2_s: float
    energies_eV: np.ndarray
    transmission: np.ndarray
    maximum_transmission: float
    minimum_transmission: float
    maximum_action: float
    valid: bool
    notes: tuple[str, ...]


def _channel_flux(
    channel: str,
    energies: np.ndarray,
    transmission: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    prefactor: float,
    *,
    valid: bool,
    notes: tuple[str, ...],
) -> ChannelFlux:
    """Turn a transmission spectrum into a reciprocal net flux plus its audit.

    The audit pair is taken at the *opaque* end of the window. The transparent
    end is worthless as a diagnostic: every window here runs up to the barrier
    top, where ``T`` is 1 and the action is 0 by construction, so reporting
    those would look like a measurement while carrying no information about
    the barrier at all.
    """

    flux = reciprocal_net_flux(energies, transmission, left, right, prefactor)
    least = float(np.min(flux.transmission))
    return ChannelFlux(
        channel=channel,
        net_flux_m2_s=flux.net_flux_m2_s,
        forward_flux_m2_s=flux.forward_flux_m2_s,
        reverse_flux_m2_s=flux.reverse_flux_m2_s,
        energies_eV=flux.energies_eV,
        transmission=flux.transmission,
        maximum_transmission=float(np.max(flux.transmission)),
        minimum_transmission=least,
        maximum_action=float("inf") if least <= 0.0 else -0.5 * math.log(least),
        valid=valid,
        notes=notes,
    )


def band_to_band_flux(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    channel: BandToBandTunnellingChannel,
    *,
    anchor_face: int,
    left_fermi_eV: float,
    right_fermi_eV: float,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Zener tunnelling from the valence band to the conduction band.

    The particle inside the gap is not blocked by one band edge: it leaves the
    valence band where ``E = E_V`` and enters the conduction band where
    ``E = E_C``, so its two turning points sit on *different* bands. The
    single-band exponent cannot express that — it would put both turning
    points on the same edge — so this channel uses the two-band (Kane) decay
    constant, which vanishes at both. Under a uniform field the quadrature
    reproduces the closed-form Zener exponent, which is what the tests pin.
    """

    if not channel.enabled:
        raise TunnellingChannelError("band_to_band channel is disabled")
    x = np.asarray(positions_m, dtype=float)
    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    if conduction.shape != x.shape or valence.shape != x.shape:
        raise WKBTunnellingError("band edges must match positions_m")
    if np.any(conduction <= valence):
        raise TunnellingChannelError(
            "band-to-band tunnelling requires a positive gap everywhere"
        )
    span = float(x[-1] - x[0])
    if span <= 0.0:
        raise WKBTunnellingError("positions_m must span a positive distance")
    # The field that tilts the bands is what opens the channel at all.
    field_V_m = abs(float(conduction[-1] - conduction[0])) / span
    notes: list[str] = []
    if field_V_m < channel.minimum_field_V_m:
        notes.append("field_below_channel_minimum")

    # The tunnelling window is the gap straddling the anchor: an energy inside
    # it is in the valence band on one side of the junction and in the
    # conduction band on the other, which is exactly the Zener channel. The
    # endpoints are excluded because kappa vanishes there (T = 1, no barrier).
    lower = float(valence[anchor_face])
    upper = float(conduction[anchor_face])
    energies = np.linspace(lower, upper, channel.energy_quadrature_order + 2)[1:-1]
    transmission = np.array(
        [
            two_band_transmission(
                x,
                conduction,
                valence,
                energy,
                channel.reduced_effective_mass_rel,
                anchor_face,
            )
            for energy in energies
        ]
    )
    mid = float(energies[len(energies) // 2])
    validity = two_band_validity(
        x,
        conduction,
        valence,
        mid,
        channel.reduced_effective_mass_rel,
        anchor_face,
    )
    left = _fermi(energies, left_fermi_eV, thermal_voltage_V)
    right = _fermi(energies, right_fermi_eV, thermal_voltage_V)
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    return _channel_flux(
        "band_to_band",
        energies,
        transmission,
        left,
        right,
        supply_prefactor_m2_s_eV,
        valid=validity.valid and not notes,
        notes=tuple(notes),
    )


def intraband_flux(
    positions_m: np.ndarray,
    band_edge_eV: np.ndarray,
    channel: IntrabandTunnellingChannel,
    *,
    anchor_face: int,
    carrier: str,
    left_fermi_eV: float,
    right_fermi_eV: float,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Tunnelling through a band-edge spike without changing band.

    ``band_edge_eV`` is the particle barrier for the requested carrier: pass
    ``E_C`` for electrons and ``-E_V`` for holes, both of which this module's
    helpers produce, so the same WKB machinery applies to either sign.
    """

    if not channel.enabled:
        raise TunnellingChannelError("intraband channel is disabled")
    if channel.carrier not in (carrier, "both"):
        raise TunnellingChannelError(
            f"intraband channel is configured for {channel.carrier!r}, not {carrier!r}"
        )
    mass = (
        channel.electron_effective_mass_rel
        if carrier == "electron"
        else channel.hole_effective_mass_rel
    )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(band_edge_eV, dtype=float)
    if barrier.shape != x.shape:
        raise WKBTunnellingError("band edge must match positions_m")
    # The spike is what the carrier tunnels through: energies from the higher
    # of the two asymptotes up to the peak.
    peak, base = local_barrier_window(barrier, anchor_face)
    notes: list[str] = []
    if not peak > base:
        raise TunnellingChannelError(
            "intraband tunnelling requires a local barrier spike at this face"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [
            windowed_wkb_transmission(x, barrier, energy, mass, anchor_face)
            for energy in energies
        ]
    )
    validity = wkb_validity(x, barrier, float(energies[0]), mass)
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    left = _fermi(energies, left_fermi_eV, thermal_voltage_V)
    right = _fermi(energies, right_fermi_eV, thermal_voltage_V)
    return _channel_flux(
        f"intraband_{carrier}",
        energies,
        transmission,
        left,
        right,
        supply_prefactor_m2_s_eV,
        valid=validity.valid,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class InterfaceDefectAssistedFlux:
    """Trap-assisted tunnelling written against an explicit occupancy."""

    channel: str
    occupancy: float
    equilibrium_occupancy: float
    electron_transmission: float
    hole_transmission: float
    electron_net_rate_m2_s: float
    hole_net_rate_m2_s: float
    net_rate_m2_s: float
    occupancy_sensitivity_m2_s: float
    stationary_occupancy_residual: float
    valid: bool
    notes: tuple[str, ...]


def interface_defect_assisted_rate(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    channel: InterfaceDefectAssistedTunnellingChannel,
    *,
    anchor_face: int,
    trap_energy_eV: float,
    occupancy: float,
    trap_density_m2: float,
    electron_capture_velocity_m_s: float,
    hole_capture_velocity_m_s: float,
    electron_density_m3: float,
    hole_density_m3: float,
    electron_reference_density_m3: float,
    hole_reference_density_m3: float,
) -> InterfaceDefectAssistedFlux:
    """Trap-assisted tunnelling that consumes the solver's own occupancy.

    The occupancy is an input, never invented here: the roadmap requires this
    channel to consume the same interface occupancy the rest of the solver
    uses, so a caller that has only an algebraically eliminated occupancy has
    nothing to pass and must not enable the channel.

    Tunnelling enters as a transmission at the trap energy that multiplies the
    capture legs; the emission legs carry the same factor, so the trap's
    detailed balance — and therefore zero net rate at the equilibrium
    occupancy — is preserved exactly.
    """

    if not channel.enabled:
        raise TunnellingChannelError("interface_defect_assisted channel is disabled")
    if not 0.0 <= float(occupancy) <= 1.0:
        raise TunnellingChannelError("interface occupancy must lie in [0, 1]")
    x = np.asarray(positions_m, dtype=float)
    conduction = np.asarray(conduction_edge_eV, dtype=float)
    valence = np.asarray(valence_edge_eV, dtype=float)
    electron_transmission = windowed_wkb_transmission(
        x,
        conduction,
        float(trap_energy_eV),
        channel.electron_effective_mass_rel,
        anchor_face,
    )
    hole_transmission = windowed_wkb_transmission(
        x,
        -valence,
        -float(trap_energy_eV),
        channel.hole_effective_mass_rel,
        anchor_face,
    )
    notes: list[str] = []
    electron_validity = wkb_validity(
        x, conduction, float(trap_energy_eV), channel.electron_effective_mass_rel
    )
    if not electron_validity.valid:
        notes.append("electron_barrier_below_meaningful_action")

    occupied = float(occupancy)
    empty = 1.0 - occupied
    # Capture and emission share one transmission per carrier, so the
    # equilibrium occupancy makes each carrier's net rate vanish identically.
    electron_net = (
        trap_density_m2
        * electron_capture_velocity_m_s
        * electron_transmission
        * (
            float(electron_density_m3) * empty
            - float(electron_reference_density_m3) * occupied
        )
    )
    hole_net = (
        trap_density_m2
        * hole_capture_velocity_m_s
        * hole_transmission
        * (float(hole_density_m3) * occupied - float(hole_reference_density_m3) * empty)
    )
    denominator = electron_capture_velocity_m_s * electron_transmission * (
        float(electron_density_m3) + float(electron_reference_density_m3)
    ) + hole_capture_velocity_m_s * hole_transmission * (
        float(hole_density_m3) + float(hole_reference_density_m3)
    )
    if denominator <= 0.0:
        raise TunnellingChannelError(
            "trap-assisted tunnelling has no active capture leg"
        )
    equilibrium = (
        electron_capture_velocity_m_s
        * electron_transmission
        * float(electron_density_m3)
        + hole_capture_velocity_m_s
        * hole_transmission
        * float(hole_reference_density_m3)
    ) / denominator
    # d(net)/df is enormous here (the legs are ~1e20 while f is O(1)), so a
    # bare net rate cannot be compared against zero: representing f in double
    # precision alone perturbs the net by |d net/df| * eps. Report the residual
    # as the equivalent occupancy offset, which is scale free and is what a
    # test can meaningfully bound.
    sensitivity = float(trap_density_m2) * denominator
    residual = (
        abs(float(electron_net - hole_net)) / sensitivity
        if sensitivity > 0.0
        else math.inf
    )
    return InterfaceDefectAssistedFlux(
        channel="interface_defect_assisted",
        occupancy=occupied,
        equilibrium_occupancy=float(equilibrium),
        electron_transmission=float(electron_transmission),
        hole_transmission=float(hole_transmission),
        electron_net_rate_m2_s=float(electron_net),
        hole_net_rate_m2_s=float(hole_net),
        occupancy_sensitivity_m2_s=sensitivity,
        stationary_occupancy_residual=residual,
        net_rate_m2_s=float(electron_net - hole_net),
        valid=electron_validity.valid,
        notes=tuple(notes),
    )


def contact_tunnelling_flux(
    positions_m: np.ndarray,
    barrier_eV: np.ndarray,
    channel: ContactTunnellingChannel,
    *,
    anchor_face: int,
    carrier: str,
    metal_fermi_eV: float,
    semiconductor_fermi_eV: float,
    thermal_voltage_V: float,
    richardson_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Field emission through a Schottky barrier at an outer contact.

    ``barrier_eV`` is the particle barrier profile measured from the metal
    into the semiconductor. Equal Fermi levels give exactly zero net current,
    which is the zero-bias statement for a contact.
    """

    if not channel.enabled:
        raise TunnellingChannelError("contact channel is disabled")
    mass = (
        channel.electron_effective_mass_rel
        if carrier == "electron"
        else channel.hole_effective_mass_rel
    )
    x = np.asarray(positions_m, dtype=float)
    barrier = np.asarray(barrier_eV, dtype=float)
    if barrier.shape != x.shape:
        raise WKBTunnellingError("barrier profile must match positions_m")
    peak, base = local_barrier_window(barrier, anchor_face, one_sided=True)
    if not peak > base:
        raise TunnellingChannelError(
            "contact tunnelling requires a barrier above the contact level"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [
            windowed_wkb_transmission(x, barrier, energy, mass, anchor_face)
            for energy in energies
        ]
    )
    validity = wkb_validity(x, barrier, float(energies[0]), mass)
    notes: list[str] = []
    if not validity.valid:
        notes.append("wkb_action_below_meaningful_barrier")
    metal = _fermi(energies, metal_fermi_eV, thermal_voltage_V)
    semiconductor = _fermi(energies, semiconductor_fermi_eV, thermal_voltage_V)
    return _channel_flux(
        f"contact_{carrier}",
        energies,
        transmission,
        metal,
        semiconductor,
        richardson_prefactor_m2_s_eV,
        valid=validity.valid,
        notes=tuple(notes),
    )


__all__ = [
    "ChannelFlux",
    "InterfaceDefectAssistedFlux",
    "TunnellingChannelError",
    "band_to_band_flux",
    "local_barrier_window",
    "conduction_band_eV",
    "contact_tunnelling_flux",
    "interface_defect_assisted_rate",
    "intraband_flux",
    "valence_band_eV",
]

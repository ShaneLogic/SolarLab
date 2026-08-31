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
* The band-to-band channel uses the single-band (parabolic) WKB exponent with
  a reduced effective mass. The two-band Kane dispersion differs in the
  numerical prefactor of the exponent; that correction is NOT applied and is
  not claimed.
* Prefactors are supply-function estimates, not fitted to any reference. Only
  the transmission and the reciprocity are asserted by tests; absolute channel
  magnitudes are explicitly not validated against SCAPS here.
* Every channel is local to the structure handed to it. None of them is wired
  into the solver at this checkpoint.
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
    wkb_transmission,
    wkb_validity,
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
    minimum_action: float
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
    minimum_action: float,
) -> ChannelFlux:
    flux = reciprocal_net_flux(energies, transmission, left, right, prefactor)
    return ChannelFlux(
        channel=channel,
        net_flux_m2_s=flux.net_flux_m2_s,
        forward_flux_m2_s=flux.forward_flux_m2_s,
        reverse_flux_m2_s=flux.reverse_flux_m2_s,
        energies_eV=flux.energies_eV,
        transmission=flux.transmission,
        maximum_transmission=float(np.max(flux.transmission)),
        minimum_action=minimum_action,
        valid=valid,
        notes=notes,
    )


def band_to_band_flux(
    positions_m: np.ndarray,
    conduction_edge_eV: np.ndarray,
    valence_edge_eV: np.ndarray,
    channel: BandToBandTunnellingChannel,
    *,
    left_fermi_eV: float,
    right_fermi_eV: float,
    thermal_voltage_V: float,
    supply_prefactor_m2_s_eV: float = 1.0e24,
) -> ChannelFlux:
    """Zener tunnelling from the valence band to the conduction band.

    The tunnelling particle inside the gap sees the *gap* as its barrier: at
    energy ``E`` the forbidden region is where ``E_V(x) < E < E_C(x)``. The
    barrier profile handed to the WKB integrator is therefore ``E_C`` for the
    electron-like branch, evaluated only on the energies that lie inside the
    local gap somewhere along the path.
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

    # Energies that are inside the gap somewhere: between the highest valence
    # edge and the lowest conduction edge is the fully forbidden window.
    lower = float(np.max(valence))
    upper = float(np.min(conduction))
    if not upper > lower:
        raise TunnellingChannelError(
            "no energy is inside the gap along the whole path; the structure "
            "does not support band-to-band tunnelling"
        )
    energies = np.linspace(lower, upper, channel.energy_quadrature_order)
    transmission = np.array(
        [
            wkb_transmission(x, conduction, energy, channel.reduced_effective_mass_rel)
            for energy in energies
        ]
    )
    validity = wkb_validity(
        x,
        conduction,
        float(energies[len(energies) // 2]),
        channel.reduced_effective_mass_rel,
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
        minimum_action=float(-0.5 * np.log(np.max(transmission))),
    )


def intraband_flux(
    positions_m: np.ndarray,
    band_edge_eV: np.ndarray,
    channel: IntrabandTunnellingChannel,
    *,
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
    base = max(float(barrier[0]), float(barrier[-1]))
    peak = float(np.max(barrier))
    notes: list[str] = []
    if not peak > base:
        raise TunnellingChannelError(
            "intraband tunnelling requires a barrier spike above both sides"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [wkb_transmission(x, barrier, energy, mass) for energy in energies]
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
        minimum_action=float(-0.5 * np.log(np.max(transmission))),
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
    electron_transmission = wkb_transmission(
        x, conduction, float(trap_energy_eV), channel.electron_effective_mass_rel
    )
    hole_transmission = wkb_transmission(
        x, -valence, -float(trap_energy_eV), channel.hole_effective_mass_rel
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
    peak = float(np.max(barrier))
    base = min(float(barrier[0]), float(barrier[-1]))
    if not peak > base:
        raise TunnellingChannelError(
            "contact tunnelling requires a barrier above the contact level"
        )
    energies = np.linspace(base, peak, channel.energy_quadrature_order)
    transmission = np.array(
        [wkb_transmission(x, barrier, energy, mass) for energy in energies]
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
        minimum_action=float(-0.5 * np.log(np.max(transmission))),
    )


__all__ = [
    "ChannelFlux",
    "InterfaceDefectAssistedFlux",
    "TunnellingChannelError",
    "band_to_band_flux",
    "conduction_band_eV",
    "contact_tunnelling_flux",
    "interface_defect_assisted_rate",
    "intraband_flux",
    "valence_band_eV",
]

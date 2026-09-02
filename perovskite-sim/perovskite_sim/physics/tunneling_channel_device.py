"""Device wiring for the D8 tunnelling channels on the guarded QF/DC lane.

The four channels in ``physics/tunneling_channels.py`` are local: they take a
barrier profile and return a flux. This module binds them to a device.

Why the binding cannot be a build-time cache
--------------------------------------------
The defect models compiled in D7 are static: their inputs are material
constants, so ``build_material_arrays`` can evaluate them once. A tunnelling
barrier is not — it is ``E_C(x) = -(phi(x) + chi(x))``, and ``phi`` is solved
for. So the compiled object here carries only the *static* part (which faces,
which contacts, the masses, the quadrature orders) and the transmission is
evaluated per residual call from the live potential, in the same way the
radiative-reabsorption hook already integrates over space per call.

Declared scope for D8-E1
------------------------
* Interface-bound channels (band-to-band, intraband, defect-assisted) are
  handed an interface face and integrate only the *connected* classically
  forbidden run containing it. That matters: a device grid holds several
  barriers at once (each heterojunction spike, the band bending at each
  contact), so integrating the whole forbidden set would merge unrelated
  barriers into one fictitious path — wrong, and silently plausible.
* **D8 defines no barrier IDENTITY, and `anchor_face` is not validated
  against the stack's interfaces (D8-P1).** `local_barrier_window` walks to a
  local maximum and then to the bounding minima, so on a device profile it
  returns the same window from a large basin of faces — measured, 49 of 72 on
  the registered lane config. The flux is correspondingly a smooth function of
  where the anchor is dropped: a plain interior face seven cells from the
  interface reproduces 99.5 % of the interface-anchored flux, while the other
  real heterointerface reports 0.19 %. Anchors that lie *inside* a forbidden
  run at a given energy do all agree; the qualifier is the limitation. Pinned
  by `tests/unit/physics/test_barrier_anchor_locality.py`. Any per-interface
  loop must supply that missing primitive first, or it will report N smoothly
  varying numbers rather than N barriers.
* The defect-assisted channel additionally requires an explicit interface
  occupancy, which exists only on the two-sided-trace interface lane.
* Every channel adds its net flux to the carrier face current at its own face.
  Nothing rescales the Richardson constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.tunneling_channels import (
    CHANNEL_NAMES,
    TunnellingChannelDocument,
)
from perovskite_sim.physics.tunneling_channels import (
    ChannelFlux,
    TunnellingChannelError,
    band_to_band_flux,
    contact_tunnelling_flux,
    interface_defect_assisted_rate,
    intraband_flux,
)


TUNNELLING_CHANNEL_DEVICE_VERSION = "wkb-tunnelling-channel-device-v1"


class TunnellingChannelCapabilityError(RuntimeError):
    """A tunnelling channel was requested where it is not certified."""


@dataclass(frozen=True, slots=True)
class CompiledTunnellingChannels:
    """Static part of the tunnelling family bound to one electrical grid."""

    document: TunnellingChannelDocument
    node_count: int
    interface_faces: tuple[int, ...]
    left_contact_face: int
    right_contact_face: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, TunnellingChannelDocument):
            raise TypeError("document must be a TunnellingChannelDocument")
        if self.node_count < 3:
            raise TunnellingChannelCapabilityError(
                "tunnelling channels need at least three electrical nodes"
            )
        faces = tuple(int(face) for face in self.interface_faces)
        if any(face < 0 or face >= self.node_count - 1 for face in faces):
            raise TunnellingChannelCapabilityError(
                "interface face index is outside the transport faces"
            )
        object.__setattr__(self, "interface_faces", faces)

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        return self.document.enabled_channels

    @property
    def identity_sha256(self) -> str:
        return self.document.sha256


def compile_tunnelling_channels(
    document: TunnellingChannelDocument | None,
    *,
    node_count: int,
    interface_nodes: tuple[int, ...],
) -> CompiledTunnellingChannels | None:
    """Bind a channel document to a grid, or return None when it is inert."""

    if document is None or not document.any_enabled:
        return None
    faces = tuple(int(node) - 1 for node in interface_nodes)
    interface_bound = tuple(
        name
        for name in ("band_to_band", "intraband", "interface_defect_assisted")
        if getattr(document, name).enabled
    )
    if interface_bound and not faces:
        raise TunnellingChannelCapabilityError(
            "interface-bound tunnelling channels need at least one "
            "heterointerface on this stack"
        )
    return CompiledTunnellingChannels(
        document=document,
        node_count=int(node_count),
        interface_faces=faces,
        left_contact_face=0,
        right_contact_face=int(node_count) - 2,
    )


@dataclass(frozen=True, slots=True)
class TunnellingChannelEvaluation:
    """Per-face tunnelling currents plus per-channel diagnostics."""

    identity_sha256: str
    electron_face_current_A_m2: np.ndarray
    hole_face_current_A_m2: np.ndarray
    channel_names: tuple[str, ...]
    channel_net_flux_m2_s: tuple[float, ...]
    channel_maximum_transmission: tuple[float, ...]
    channel_minimum_transmission: tuple[float, ...]
    channel_valid: tuple[bool, ...]
    channel_notes: tuple[tuple[str, ...], ...]
    defect_assisted_occupancy: float | None
    defect_assisted_residual: float | None


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def evaluate_tunnelling_channels(
    compiled: CompiledTunnellingChannels,
    *,
    positions_m: np.ndarray,
    potential_V: np.ndarray,
    affinity_eV: np.ndarray,
    band_gap_eV: np.ndarray,
    electron_quasi_fermi_eV: np.ndarray,
    hole_quasi_fermi_eV: np.ndarray,
    thermal_voltage_V: float,
    interface_occupancy: float | None = None,
    interface_trap_energy_eV: float | None = None,
    interface_trap_density_m2: float | None = None,
    interface_electron_velocity_m_s: float | None = None,
    interface_hole_velocity_m_s: float | None = None,
    electron_density_m3: np.ndarray | None = None,
    hole_density_m3: np.ndarray | None = None,
) -> TunnellingChannelEvaluation:
    """Evaluate every enabled channel at the current self-consistent state."""

    x = np.asarray(positions_m, dtype=float)
    potential = np.asarray(potential_V, dtype=float)
    affinity = np.asarray(affinity_eV, dtype=float)
    gap = np.asarray(band_gap_eV, dtype=float)
    if not (x.shape == potential.shape == affinity.shape == gap.shape):
        raise TunnellingChannelCapabilityError(
            "tunnelling inputs must share the electrical grid"
        )
    conduction = -(potential + affinity)
    valence = -(potential + affinity + gap)
    face_count = x.size - 1
    electron_current = np.zeros(face_count, dtype=float)
    hole_current = np.zeros(face_count, dtype=float)

    names: list[str] = []
    fluxes: list[float] = []
    transmissions: list[float] = []
    least_transmissions: list[float] = []
    valid: list[bool] = []
    notes: list[tuple[str, ...]] = []
    occupancy_out: float | None = None
    residual_out: float | None = None

    document = compiled.document
    # Quasi-Fermi levels on the two sides of the tunnelling path set the
    # occupation difference; equal levels give exactly zero net flux.
    #
    # The whole profile goes in; each channel reads it at each energy's own
    # turning points. A channel is a conduction path across ONE barrier and is
    # driven by the quasi-Fermi drop across THAT barrier — not across the whole
    # device (which would inflate every interface channel by the full applied
    # bias), and not across one grid cell (which would make the flux
    # proportional to dx so that it vanishes under refinement — measured as a
    # factor of ~2 per grid doubling before this was corrected).
    qfn = np.asarray(electron_quasi_fermi_eV, dtype=float)
    qfp = np.asarray(hole_quasi_fermi_eV, dtype=float)
    if qfn.shape != x.shape or qfp.shape != x.shape:
        raise TunnellingChannelCapabilityError(
            "quasi-Fermi levels must share the electrical grid"
        )

    def _record(
        flux: ChannelFlux, face: int, carrier: str, *, also_record: bool = True
    ) -> None:
        """Log one channel and add its charge to the carrier it moves.

        ``also_record`` is False for band-to-band, which moves TWO carriers
        from one event and adds the second leg at the call site; the
        diagnostics stay one entry per channel either way.
        """
        names.append(flux.channel)
        fluxes.append(flux.net_flux_m2_s)
        transmissions.append(flux.maximum_transmission)
        least_transmissions.append(flux.minimum_transmission)
        valid.append(flux.valid)
        notes.append(flux.notes)
        if carrier == "electron":
            electron_current[face] += -Q * flux.net_flux_m2_s
        else:
            hole_current[face] += Q * flux.net_flux_m2_s

    if document.band_to_band.enabled:
        face = compiled.interface_faces[0]
        try:
            flux = band_to_band_flux(
                x,
                conduction,
                valence,
                document.band_to_band,
                anchor_face=face,
                quasi_fermi_eV=qfn,
                thermal_voltage_V=thermal_voltage_V,
            )
        except TunnellingChannelError as exc:
            raise TunnellingChannelCapabilityError(
                f"band-to-band channel cannot run on this structure: {exc}"
            ) from exc
        # A Zener transition takes one electron from the valence band to the
        # conduction band, so it creates an electron AND a hole. Recording
        # only the electron leg was a flat charge-conservation violation: the
        # hole array was never touched by this channel.
        _record(flux, face, "electron", also_record=False)
        hole_current[face] += Q * flux.net_flux_m2_s

    if document.intraband.enabled:
        face = compiled.interface_faces[0]
        carriers = (
            ("electron", "hole")
            if document.intraband.carrier == "both"
            else (document.intraband.carrier,)
        )
        for carrier in carriers:
            barrier = conduction if carrier == "electron" else -valence
            # `qfp` is ALREADY in the hole particle-energy convention that
            # matches the `-valence` barrier: the solver builds it as
            # `V_T*ln(p) + (phi + chi + Eg)`, i.e. `-E_V + V_T*ln(p)`.
            # Negating it here put the hole drive 12.9 eV BELOW its barrier
            # instead of 0.72 eV above, which made the hole flux underflow by
            # ~212 orders. The only device test that touched this branch
            # asserted channel names and tuple lengths, so it could not see it.
            profile = qfn if carrier == "electron" else qfp
            try:
                flux = intraband_flux(
                    x,
                    barrier,
                    document.intraband,
                    anchor_face=face,
                    carrier=carrier,
                    quasi_fermi_eV=profile,
                    thermal_voltage_V=thermal_voltage_V,
                )
            except TunnellingChannelError as exc:
                raise TunnellingChannelCapabilityError(
                    f"intraband channel cannot run on this structure: {exc}"
                ) from exc
            _record(flux, face, carrier)

    if document.interface_defect_assisted.enabled:
        required = (
            interface_occupancy,
            interface_trap_energy_eV,
            interface_trap_density_m2,
            interface_electron_velocity_m_s,
            interface_hole_velocity_m_s,
            electron_density_m3,
            hole_density_m3,
        )
        if any(value is None for value in required):
            raise TunnellingChannelCapabilityError(
                "interface-defect-assisted tunnelling requires an explicit "
                "interface occupancy and its trap parameters; the default "
                "lane eliminates the occupancy algebraically and cannot "
                "supply them"
            )
        face = compiled.interface_faces[0]
        electrons = np.asarray(electron_density_m3, dtype=float)
        holes = np.asarray(hole_density_m3, dtype=float)
        assisted = interface_defect_assisted_rate(
            x,
            conduction,
            valence,
            document.interface_defect_assisted,
            anchor_face=face,
            trap_energy_eV=float(interface_trap_energy_eV),
            occupancy=float(interface_occupancy),
            trap_density_m2=float(interface_trap_density_m2),
            electron_capture_velocity_m_s=float(interface_electron_velocity_m_s),
            hole_capture_velocity_m_s=float(interface_hole_velocity_m_s),
            electron_density_m3=float(electrons[face]),
            hole_density_m3=float(holes[face]),
            electron_reference_density_m3=float(electrons[face + 1]),
            hole_reference_density_m3=float(holes[face + 1]),
        )
        names.append(assisted.channel)
        fluxes.append(assisted.net_rate_m2_s)
        transmissions.append(
            max(assisted.electron_transmission, assisted.hole_transmission)
        )
        least_transmissions.append(
            min(assisted.electron_transmission, assisted.hole_transmission)
        )
        valid.append(assisted.valid)
        notes.append(assisted.notes)
        occupancy_out = assisted.occupancy
        residual_out = assisted.stationary_occupancy_residual
        electron_current[face] += -Q * assisted.electron_net_rate_m2_s
        hole_current[face] += Q * assisted.hole_net_rate_m2_s

    if document.contact.enabled:
        sides = (
            ("left", "right")
            if document.contact.side == "both"
            else (document.contact.side,)
        )
        for side in sides:
            face = (
                compiled.left_contact_face
                if side == "left"
                else compiled.right_contact_face
            )
            # ONE energy frame for all three inputs. The barrier profile an
            # electron sees at a Schottky contact is the conduction band
            # itself, in the same absolute frame as the metal level and the
            # quasi-Fermi profile.
            #
            # The previous form built `conduction - reference + phi_B`, a
            # BARRIER-RELATIVE profile (values around +0.3), and compared its
            # energies against an absolute `metal` (around -4.3) and an
            # absolute quasi-Fermi profile. Measured, that mismatch moved the
            # flux by ~47 orders of magnitude and made the channel a
            # structural no-op in every configuration. It also meant
            # `barrier_height_eV` shifted the whole profile by a constant,
            # which cannot change a barrier's SHAPE and so never affected the
            # WKB action at all.
            #
            # `barrier_height_eV` now does only its physical job: it places
            # the metal Fermi level below the contact conduction edge.
            reference = float(conduction[0] if side == "left" else conduction[-1])
            barrier = conduction
            metal = reference - document.contact.barrier_height_eV
            # The semiconductor side is read at each energy's own turning
            # point inside contact_tunnelling_flux, so the whole profile goes
            # in rather than one endpoint.
            try:
                flux = contact_tunnelling_flux(
                    x,
                    barrier,
                    document.contact,
                    anchor_face=face,
                    carrier="electron",
                    metal_fermi_eV=metal,
                    quasi_fermi_eV=qfn,
                    thermal_voltage_V=thermal_voltage_V,
                )
            except TunnellingChannelError as exc:
                raise TunnellingChannelCapabilityError(
                    f"contact channel cannot run on this structure: {exc}"
                ) from exc
            names.append(f"{flux.channel}_{side}")
            fluxes.append(flux.net_flux_m2_s)
            transmissions.append(flux.maximum_transmission)
            least_transmissions.append(flux.minimum_transmission)
            valid.append(flux.valid)
            notes.append(flux.notes)
            electron_current[face] += -Q * flux.net_flux_m2_s

    if not np.all(np.isfinite(electron_current)) or not np.all(
        np.isfinite(hole_current)
    ):
        raise TunnellingChannelCapabilityError(
            "tunnelling channels produced a non-finite face current"
        )
    return TunnellingChannelEvaluation(
        identity_sha256=compiled.identity_sha256,
        electron_face_current_A_m2=_readonly(electron_current),
        hole_face_current_A_m2=_readonly(hole_current),
        channel_names=tuple(names),
        channel_net_flux_m2_s=tuple(fluxes),
        channel_maximum_transmission=tuple(transmissions),
        channel_minimum_transmission=tuple(least_transmissions),
        channel_valid=tuple(valid),
        channel_notes=tuple(notes),
        defect_assisted_occupancy=occupancy_out,
        defect_assisted_residual=residual_out,
    )


__all__ = [
    "CHANNEL_NAMES",
    "TUNNELLING_CHANNEL_DEVICE_VERSION",
    "CompiledTunnellingChannels",
    "TunnellingChannelCapabilityError",
    "TunnellingChannelEvaluation",
    "compile_tunnelling_channels",
    "evaluate_tunnelling_channels",
]

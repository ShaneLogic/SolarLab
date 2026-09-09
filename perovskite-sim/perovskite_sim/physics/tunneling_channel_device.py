"""Physical-energy, conservative single-barrier tunnelling for QF/DC.

The device scope is one isolated conduction-band material spike between two
adjacent semiconductor reservoirs. Its layer identity is fixed at construction;
the live electrostatic potential determines its energy-dependent turning points.
Local formulas for the other tunnelling families remain in tunneling_channels,
but their device injection and occupancy coupling are not certified here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.tunneling_channels import (
    CHANNEL_NAMES,
    TunnellingChannelDocument,
)
from perovskite_sim.physics.tunneling_channels import (
    IntrabandBarrierRegion,
    IntrabandPathFlux,
    TunnellingChannelError,
    intraband_path_flux,
)


TUNNELLING_CHANNEL_DEVICE_VERSION = "wkb-tunnelling-channel-device-v2"


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
    intraband_barrier: IntrabandBarrierRegion | None = None

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
    positions_m: np.ndarray | None = None,
    physical_affinity_eV: np.ndarray | None = None,
    layer_names: tuple[str, ...] | None = None,
    layer_boundaries_m: np.ndarray | None = None,
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
    compiled = CompiledTunnellingChannels(
        document=document,
        node_count=int(node_count),
        interface_faces=faces,
        left_contact_face=0,
        right_contact_face=int(node_count) - 2,
    )
    if document.intraband.enabled and document.intraband.carrier == "electron" and positions_m is not None:
        if any(value is None for value in (physical_affinity_eV, layer_names, layer_boundaries_m)):
            raise TunnellingChannelCapabilityError("material barrier identity needs layer geometry and physical affinity")
        x = np.asarray(positions_m, dtype=float)
        affinity = np.asarray(physical_affinity_eV, dtype=float)
        boundaries = np.asarray(layer_boundaries_m, dtype=float)
        if (x.shape != (node_count,) or affinity.shape != x.shape
                or not np.all(np.isfinite(affinity)) or np.any(np.diff(x) <= 0.0)
                or len(layer_names) != len(interface_nodes) + 1
                or boundaries.shape != (len(layer_names) + 1,)):
            raise TunnellingChannelCapabilityError("barrier geometry does not match the layer grid")
        nodes = (0, *interface_nodes, node_count)
        if any(left >= right for left, right in zip(nodes[:-1], nodes[1:])):
            raise TunnellingChannelCapabilityError("material layer node ranges must be strictly ordered")
        candidates = []
        for layer_index in range(1, len(layer_names) - 1):
            start, stop = nodes[layer_index], nodes[layer_index + 1]
            if affinity[start] < affinity[start - 1] and affinity[stop - 1] < affinity[stop]:
                if stop - start < 3:
                    raise TunnellingChannelCapabilityError("resolved intraband barrier requires at least three barrier nodes")
                candidates.append(IntrabandBarrierRegion(
                    identity=f"electrical_layer[{layer_index}]:{layer_names[layer_index]}",
                    left_start_node=nodes[layer_index - 1], core_start_node=start,
                    core_stop_node=stop, right_stop_node=nodes[layer_index + 2],
                    core_left_m=float(boundaries[layer_index]),
                    core_right_m=float(boundaries[layer_index + 1]),
                ))
        if len(candidates) != 1:
            raise TunnellingChannelCapabilityError(
                "intraband device transport requires one isolated material conduction-band spike; "
                f"found {len(candidates)}"
            )
        compiled = replace(compiled, intraband_barrier=candidates[0])
    return compiled


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
    intraband_path: IntrabandPathFlux | None = None
    implementation_version: str = TUNNELLING_CHANNEL_DEVICE_VERSION


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def physical_quasi_fermi_levels_eV(
    *,
    potential_V: np.ndarray,
    affinity_eV: np.ndarray,
    band_gap_eV: np.ndarray,
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    conduction_dos_m3: np.ndarray,
    valence_dos_m3: np.ndarray,
    thermal_voltage_V: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical E_Fn and E_Fp for the Boltzmann density solver.

    Both returned levels use the electron-energy reference of E_C and E_V.
    A hole particle channel therefore uses -E_Fp with its -E_V barrier.
    Physical, temperature-scaled DOS are required even if TE normalization
    or the DOS-folded transport potential is disabled.
    """
    if conduction_dos_m3 is None or valence_dos_m3 is None:
        raise TunnellingChannelCapabilityError("physical Nc/Nv are required for tunnelling occupations")
    arrays = tuple(np.asarray(value, dtype=float) for value in (
        potential_V, affinity_eV, band_gap_eV, electron_density_m3,
        hole_density_m3, conduction_dos_m3, valence_dos_m3,
    ))
    potential, affinity, gap, electrons, holes, nc, nv = arrays
    if potential.ndim != 1 or not potential.size or any(value.shape != potential.shape for value in arrays):
        raise TunnellingChannelCapabilityError("physical level inputs must share one nodal grid")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise TunnellingChannelCapabilityError("physical bands, densities and Nc/Nv must be finite")
    if any(np.any(value <= 0.0) for value in (electrons, holes, nc, nv)):
        raise TunnellingChannelCapabilityError("positive carrier densities and Nc/Nv are required")
    thermal = float(thermal_voltage_V)
    if not np.isfinite(thermal) or thermal <= 0.0:
        raise TunnellingChannelCapabilityError("thermal voltage must be finite and positive")
    conduction = -(potential + affinity)
    electron_level = conduction + thermal * (np.log(electrons) - np.log(nc))
    hole_level = conduction - gap - thermal * (np.log(holes) - np.log(nv))
    return electron_level, hole_level


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
    """Evaluate the resolved electron path; other device couplings fail explicitly."""
    document = compiled.document
    if document.enabled_channels != ("intraband",) or document.intraband.carrier != "electron":
        raise TunnellingChannelCapabilityError(
            "device tunnelling v2 supports one intraband electron channel; "
            "hole, contact, band-to-band and defect-assisted occupancy coupling "
            "require their own conservative device validation"
        )
    if compiled.intraband_barrier is None:
        raise TunnellingChannelCapabilityError("intraband channel requires compiled material and reservoir geometry")
    x = np.asarray(positions_m, dtype=float)
    potential = np.asarray(potential_V, dtype=float)
    affinity = np.asarray(affinity_eV, dtype=float)
    gap = np.asarray(band_gap_eV, dtype=float)
    qfn = np.asarray(electron_quasi_fermi_eV, dtype=float)
    qfp = np.asarray(hole_quasi_fermi_eV, dtype=float)
    if x.shape != (compiled.node_count,) or any(value.shape != x.shape for value in (potential, affinity, gap, qfn, qfp)):
        raise TunnellingChannelCapabilityError("physical tunnelling inputs must share the compiled electrical grid")
    if not all(np.all(np.isfinite(value)) for value in (potential, affinity, gap, qfn, qfp)):
        raise TunnellingChannelCapabilityError("physical tunnelling bands and levels must be finite")
    try:
        path = intraband_path_flux(
            x, -(potential + affinity), document.intraband,
            region=compiled.intraband_barrier, quasi_fermi_eV=qfn,
            thermal_voltage_V=thermal_voltage_V,
        )
    except TunnellingChannelError as exc:
        raise TunnellingChannelCapabilityError(f"resolved intraband channel failed: {exc}") from exc
    flux = path.flux
    if not flux.valid:
        raise TunnellingChannelCapabilityError("identified barrier has no meaningful WKB action")
    electron_current = -Q * path.particle_face_flux_m2_s
    if not np.all(np.isfinite(electron_current)):
        raise TunnellingChannelCapabilityError("tunnelling produced a non-finite face current")
    return TunnellingChannelEvaluation(
        identity_sha256=compiled.identity_sha256,
        electron_face_current_A_m2=_readonly(electron_current),
        hole_face_current_A_m2=_readonly(np.zeros(x.size - 1)),
        channel_names=(flux.channel,),
        channel_net_flux_m2_s=(flux.net_flux_m2_s,),
        channel_maximum_transmission=(flux.maximum_transmission,),
        channel_minimum_transmission=(flux.minimum_transmission,),
        channel_valid=(flux.valid,), channel_notes=(flux.notes,),
        defect_assisted_occupancy=None, defect_assisted_residual=None,
        intraband_path=path,
    )


__all__ = [
    "CHANNEL_NAMES",
    "TUNNELLING_CHANNEL_DEVICE_VERSION",
    "CompiledTunnellingChannels",
    "TunnellingChannelCapabilityError",
    "TunnellingChannelEvaluation",
    "compile_tunnelling_channels",
    "physical_quasi_fermi_levels_eV",
    "evaluate_tunnelling_channels",
]

"""Physical band edges kept apart from DOS-folded transport potentials (M class).

Under Boltzmann statistics the heterostructure-correct drift potentials carry
``V_T ln(N_C)`` and ``-V_T ln(N_V)`` besides the physical band edges. The
historical engine folds these terms, relative to one reference layer, into its
cached affinity and gap arrays::

    dC = V_T ln(N_C / N_C,ref)    dV = V_T ln(N_V / N_V,ref)
    chi_t = chi + dC              Eg_t = Eg - dC - dV

The folded arrays are transport potentials only: thermionic emission, band
diagrams, contact quasi-Fermi levels and every statistics evaluation read the
physical edges. ``BandStructure`` therefore carries both on the same nodes and
names which is which; neither substitutes for the other.

Layers whose DOS is missing are skipped, and nothing is folded when the
reference layer lacks DOS. Node ownership must be a partition; the historical
1D owner is :data:`solarlab.mesh.RIGHT_LAYER_WINS`, under which the shared
interface node takes the right layer's correction, so the effective material
interface stays at ``x_I - h_{I-1}/2`` (LD-05, kept on purpose).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import ImmutableArrays, Vector, frozen_array
from solarlab.mesh.grid import RIGHT_LAYER_WINS, layer_node_masks

__all__ = [
    "ABSORBER_ROLE", "BandStructure", "LayerBandInput", "dos_fold_reference_layer",
    "fold_dos_into_transport_bands", "layered_band_structure",
]

ABSORBER_ROLE = "absorber"


@dataclass(frozen=True)
class BandStructure(ImmutableArrays):
    """Physical band edges and transport potentials on one set of nodes [eV]."""

    physical_electron_affinity_eV: Vector
    physical_band_gap_eV: Vector
    transport_electron_affinity_eV: Vector
    transport_band_gap_eV: Vector
    dos_folded: bool
    reference_layer: int | None

    def __post_init__(self) -> None:
        names = (
            "physical_electron_affinity_eV", "physical_band_gap_eV",
            "transport_electron_affinity_eV", "transport_band_gap_eV",
        )
        arrays = [frozen_array(object.__getattribute__(self, name)) for name in names]
        if arrays[0].ndim != 1 or any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("band arrays must be 1-D and share one node set")
        for name, array in zip(names, arrays):
            object.__setattr__(self, name, array)
        if not self.dos_folded and (
            arrays[0].tobytes() != arrays[2].tobytes()
            or arrays[1].tobytes() != arrays[3].tobytes()
        ):
            raise ValueError("an unfolded band structure must reuse the physical edges")

    @classmethod
    def without_dos_fold(cls, electron_affinity_eV: ArrayLike,
                         band_gap_eV: ArrayLike) -> BandStructure:
        """Transport potentials equal to the physical edges (fold disabled)."""
        chi = np.asarray(electron_affinity_eV, dtype=float)
        gap = np.asarray(band_gap_eV, dtype=float)
        return cls(chi, gap, chi, gap, dos_folded=False, reference_layer=None)

    def transport_potentials(self, phi: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Historical SG drift potentials ``(phi + chi_t, phi + chi_t + Eg_t)`` [V]."""
        potential = np.asarray(phi, dtype=float)
        if potential.shape != self.transport_electron_affinity_eV.shape:
            raise ValueError("phi must live on the band-structure nodes")
        electron = potential + self.transport_electron_affinity_eV
        hole = potential + self.transport_electron_affinity_eV + self.transport_band_gap_eV
        return electron, hole

    def physical_band_edges(self, phi: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Physical ``(E_C, E_V)`` below vacuum [eV]: ``E_C = -phi - chi``."""
        potential = np.asarray(phi, dtype=float)
        if potential.shape != self.physical_electron_affinity_eV.shape:
            raise ValueError("phi must live on the band-structure nodes")
        conduction = -potential - self.physical_electron_affinity_eV
        return conduction, conduction - self.physical_band_gap_eV


def dos_fold_reference_layer(roles: Sequence[str]) -> int:
    """Historical fold reference: the first absorber layer, else layer 0."""
    if not roles:
        raise ValueError("at least one electrical layer is required")
    return next((index for index, role in enumerate(roles) if role == ABSORBER_ROLE), 0)


def _has_dos(conduction: float | None, valence: float | None) -> bool:
    # Truthiness as in the historical engine: None and 0.0 both mean "absent".
    return bool(conduction and valence)


def fold_dos_into_transport_bands(
    electron_affinity_eV: ArrayLike,
    band_gap_eV: ArrayLike,
    layer_masks: Sequence[ArrayLike],
    layer_dos_m3: Sequence[tuple[float | None, float | None]],
    *,
    reference_layer: int,
    thermal_voltage_V: float,
) -> BandStructure:
    """Fold per-layer ``V_T ln(N/N_ref)`` into the transport potentials.

    ``electron_affinity_eV`` and ``band_gap_eV`` are the physical node arrays;
    ``layer_masks`` must be mutually exclusive because the fold accumulates.
    """
    chi = np.array(electron_affinity_eV, dtype=float, copy=True)
    gap = np.array(band_gap_eV, dtype=float, copy=True)
    if chi.ndim != 1 or gap.shape != chi.shape:
        raise ValueError("affinity and gap must be 1-D node arrays of one shape")
    masks = [np.asarray(mask, dtype=bool) for mask in layer_masks]
    if len(masks) != len(layer_dos_m3) or not masks:
        raise ValueError("one DOS pair is required per layer mask")
    if any(mask.shape != chi.shape for mask in masks):
        raise ValueError("layer masks must live on the band nodes")
    if np.any(np.sum(masks, axis=0) > 1):
        raise ValueError("an accumulating fold requires a node partition")
    if not 0 <= reference_layer < len(masks):
        raise ValueError("reference_layer is outside the layer list")
    thermal = float(thermal_voltage_V)
    if not math.isfinite(thermal) or thermal <= 0.0:
        raise ValueError("thermal_voltage_V must be finite and positive")
    physical_chi, physical_gap = chi.copy(), gap.copy()
    reference_nc, reference_nv = layer_dos_m3[reference_layer]
    if not _has_dos(reference_nc, reference_nv):
        return BandStructure(physical_chi, physical_gap, chi, gap,
                             dos_folded=False, reference_layer=reference_layer)
    for (conduction, valence), mask in zip(layer_dos_m3, masks):
        if _has_dos(conduction, valence):
            shift_c = thermal * math.log(conduction / reference_nc)  # type: ignore[operator]
            shift_v = thermal * math.log(valence / reference_nv)  # type: ignore[operator]
            chi[mask] = chi[mask] + shift_c
            gap[mask] = gap[mask] - shift_c - shift_v
    return BandStructure(physical_chi, physical_gap, chi, gap,
                         dos_folded=True, reference_layer=reference_layer)


@dataclass(frozen=True, slots=True)
class LayerBandInput:
    """One uniform electrical layer for :func:`layered_band_structure`."""

    thickness_m: float
    electron_affinity_eV: float
    band_gap_eV: float
    conduction_dos_m3: float | None = None
    valence_dos_m3: float | None = None
    role: str = ""


def layered_band_structure(
    x: ArrayLike,
    layers: Sequence[LayerBandInput],
    *,
    thermal_voltage_V: float,
    fold_dos: bool,
    convention: str = RIGHT_LAYER_WINS,
) -> BandStructure:
    """Paint uniform layers onto nodes, then optionally fold their DOS.

    Assignment and fold share one ownership partition, as in the historical
    builder; graded or temperature-shifted edges are prepared by the caller
    and passed to :func:`fold_dos_into_transport_bands` directly.
    """
    grid = np.asarray(x, dtype=float)
    thicknesses = [layer.thickness_m for layer in layers]
    masks = layer_node_masks(grid, thicknesses, convention)
    chi = np.zeros(grid.shape)
    gap = np.zeros(grid.shape)
    for layer, mask in zip(layers, masks):
        chi[mask] = layer.electron_affinity_eV
        gap[mask] = layer.band_gap_eV
    if not fold_dos:
        return BandStructure.without_dos_fold(chi, gap)
    return fold_dos_into_transport_bands(
        chi, gap, masks,
        [(layer.conduction_dos_m3, layer.valence_dos_m3) for layer in layers],
        reference_layer=dos_fold_reference_layer([layer.role for layer in layers]),
        thermal_voltage_V=thermal_voltage_V,
    )

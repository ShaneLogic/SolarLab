"""Optical binding of a device stack: the inputs a TMM generation depends on.

``OpticalStackBinding`` freezes the full optical stack (including optical-only
layers such as a glass substrate), the wavelength grid, the spectral photon
flux, the bounding media, the incidence side and any declared temperature /
composition dependence, and identifies them by one SHA-256. A generation
profile carries that identity, so changing any input forces recomputation.

Coordinates. The optical coordinate starts at the illuminated front face of
``layers[0]``. Optical-only layers must form a contiguous front prefix (the
existing substrate rule); the electrical coordinate starts at the first
electrical layer, so ``optical = electrical + electrical_offset_m``.
Incidence is the front face only -- the single existing behaviour; any other
side is rejected rather than approximated.

Photon budget. Per wavelength, ``R + T + sum_layers A + dropped = 1`` where
``dropped`` is the documented incoherent-bypass multi-pass term (zero for a
coherent stack). Electrical-layer absorption is the generation budget;
optical-only (parasitic) absorption never generates carriers.

The arithmetic of the generation path is the historical one
(``_compute_tmm_generation``), so profiles reproduce bit for bit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

from solarlab.physics.generation._validation import (
    array_digest,
    require_cell_faces,
    require_node_coordinates,
)
from solarlab.physics.generation.profile import (
    ControlVolumes,
    GenerationProfile,
    control_volume_digest,
    require_control_volumes,
)
from solarlab.physics.generation.spectrum import SpectralPhotonFlux, wavelength_grid_m
from solarlab.physics.generation.transfer_matrix import (
    TMMLayer,
    dropped_multipass_fraction,
    tmm_absorbed_photon_flux_per_cell,
    tmm_absorption_profile,
    tmm_cumulative_absorptance,
    tmm_reflectance,
    tmm_transmittance,
)

__all__ = [
    "INCIDENCE_SIDES",
    "OpticalLayerSpec",
    "OpticalStackBinding",
    "PhotonBudget",
    "constant_index_optical_constants",
    "graded_slice_centres",
]

INCIDENCE_SIDES = ("front",)
_WINDOW_TOLERANCE_M = 1e-15


@dataclass(frozen=True)
class OpticalLayerSpec:
    """One optical layer (or one slice of a graded physical layer).

    ``n`` and ``k`` are sampled on the binding's wavelength grid. Slices of
    one physical layer share ``name`` and must be contiguous.
    """

    name: str
    thickness_m: float
    n: np.ndarray
    k: np.ndarray
    incoherent: bool = False
    electrical: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("optical layer requires a name")
        thickness = float(self.thickness_m)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(f"{self.name}: thickness_m must be finite and positive")
        n = np.array(self.n, dtype=float, copy=True)
        k = np.array(self.k, dtype=float, copy=True)
        if n.ndim != 1 or k.shape != n.shape:
            raise ValueError(f"{self.name}: n and k must be 1-D arrays of one length")
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(k)) or np.any(n <= 0.0) or np.any(k < 0.0):
            raise ValueError(f"{self.name}: n must be finite positive and k finite non-negative")
        n.setflags(write=False)
        k.setflags(write=False)
        object.__setattr__(self, "thickness_m", thickness)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "incoherent", bool(self.incoherent))
        object.__setattr__(self, "electrical", bool(self.electrical))


def constant_index_optical_constants(
    wavelengths_nm: np.ndarray,
    *,
    absorption_coefficient_m1: float,
    refractive_index: float | None = None,
    relative_permittivity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, k)`` of a layer without tabulated optical constants.

    ``n`` is the declared constant index, or ``sqrt(eps_r)`` when none is
    declared; ``k = alpha lam / (4 pi)`` reproduces the layer's scalar
    absorption coefficient at every wavelength (historical fallback).
    """

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if refractive_index is not None:
        n = np.full(len(wavelengths), refractive_index)
    elif relative_permittivity is not None:
        n = np.full(len(wavelengths), np.sqrt(relative_permittivity))
    else:
        raise ValueError("a constant-index layer needs refractive_index or relative_permittivity")
    k = absorption_coefficient_m1 * wavelength_grid_m(wavelengths) / (4.0 * np.pi)
    return n, k


def graded_slice_centres(thickness_m: float, slices: int) -> tuple[np.ndarray, float]:
    """Slice-centre positions and the common slice thickness of a graded layer.

    The composition (and hence n, k) of each slice is evaluated at these
    centres by the composition binding (P04-11); this is only the geometry.
    """

    if type(slices) is not int or slices < 1:
        raise ValueError("slices must be a positive integer")
    thickness = float(thickness_m)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness_m must be finite and positive")
    centres = (np.arange(slices, dtype=float) + 0.5) * (thickness / slices)
    return centres, thickness / slices


def _canonical_dependencies(value: Mapping[str, object] | tuple) -> tuple[tuple[str, str], ...]:
    items = dict(value) if isinstance(value, Mapping) else dict(value)
    canonical: list[tuple[str, str]] = []
    for key in sorted(items):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("dependency names must be non-empty strings")
        raw = items[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"dependency {key!r} must be a number or a string")
        if isinstance(raw, float) and not math.isfinite(raw):
            raise ValueError(f"dependency {key!r} must be finite")
        canonical.append((key, repr(float(raw)) if isinstance(raw, (int, float)) else raw))
    return tuple(canonical)


@dataclass(frozen=True)
class PhotonBudget:
    """Spectral and integrated photon bookkeeping of one optical binding.

    Integrated values are photon fluxes [m^-2 s^-1] (trapezoid over the
    metre wavelength grid, the same rule the generation uses).
    """

    wavelengths_nm: np.ndarray
    incident_spectral_m2_s_m: np.ndarray
    reflectance: np.ndarray
    transmittance: np.ndarray
    layer_absorptance: np.ndarray
    dropped_multipass: np.ndarray
    layer_names: tuple[str, ...]
    layer_electrical: tuple[bool, ...]

    def _integrate(self, fraction: np.ndarray) -> float:
        return float(np.trapezoid(self.incident_spectral_m2_s_m * fraction, x=wavelength_grid_m(self.wavelengths_nm)))

    @property
    def incident_m2_s(self) -> float:
        return self._integrate(np.ones_like(self.reflectance))

    @property
    def reflected_m2_s(self) -> float:
        return self._integrate(self.reflectance)

    @property
    def transmitted_m2_s(self) -> float:
        return self._integrate(self.transmittance)

    @property
    def dropped_m2_s(self) -> float:
        return self._integrate(self.dropped_multipass)

    @property
    def layer_absorbed_m2_s(self) -> np.ndarray:
        return np.array([self._integrate(row) for row in self.layer_absorptance])

    @property
    def electrical_absorbed_m2_s(self) -> float:
        mask: np.ndarray = np.asarray(self.layer_electrical, dtype=bool)
        return self._integrate(np.sum(self.layer_absorptance[mask], axis=0))

    @property
    def parasitic_absorbed_m2_s(self) -> float:
        mask: np.ndarray = ~np.asarray(self.layer_electrical, dtype=bool)
        if not np.any(mask):
            return 0.0
        return self._integrate(np.sum(self.layer_absorptance[mask], axis=0))

    def spectral_closure_residual(self) -> np.ndarray:
        """``1 - R - T - sum A - dropped`` per wavelength (round-off only)."""

        return 1.0 - self.reflectance - self.transmittance - np.sum(self.layer_absorptance, axis=0) - self.dropped_multipass


@dataclass(frozen=True)
class OpticalStackBinding:
    """Frozen optical inputs of one device and their content identity."""

    layers: tuple[OpticalLayerSpec, ...]
    spectrum: SpectralPhotonFlux
    ambient_index: float = 1.0
    exit_index: float = 1.0
    incidence: str = "front"
    dependencies: tuple[tuple[str, str], ...] = ()
    sha256: str = field(init=False)
    boundaries_m: np.ndarray = field(init=False, repr=False)
    electrical_offset_m: float = field(init=False)
    physical_layer_slices: tuple[tuple[str, slice], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers or not all(isinstance(item, OpticalLayerSpec) for item in layers):
            raise ValueError("optical binding requires OpticalLayerSpec layers")
        if not isinstance(self.spectrum, SpectralPhotonFlux):
            raise TypeError("spectrum must be a SpectralPhotonFlux")
        n_wl = self.spectrum.wavelengths_nm.shape[0]
        if any(item.n.shape != (n_wl,) for item in layers):
            raise ValueError("every layer's n, k must be sampled on the spectrum's wavelength grid")
        if any(item.incoherent for item in layers[1:]):
            raise ValueError("only the first optical layer may be incoherent")
        seen_electrical = False
        for item in layers:
            if item.electrical:
                seen_electrical = True
            elif seen_electrical:
                raise ValueError("optical-only layers must form a contiguous front prefix")
        if not seen_electrical:
            raise ValueError("optical binding requires at least one electrical layer")
        if self.incidence not in INCIDENCE_SIDES:
            raise ValueError(f"incidence must be one of {INCIDENCE_SIDES}; only front illumination exists")
        for name in ("ambient_index", "exit_index"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        groups: list[tuple[str, slice]] = []
        start = 0
        for index in range(1, len(layers) + 1):
            if index == len(layers) or layers[index].name != layers[start].name:
                groups.append((layers[start].name, slice(start, index)))
                start = index
        if len({name for name, _ in groups}) != len(groups):
            raise ValueError("slices of one physical layer must be contiguous")
        dependencies = _canonical_dependencies(self.dependencies)
        boundaries: np.ndarray = np.empty(len(layers) + 1, dtype=float)
        boundaries[0] = 0.0
        boundaries[1:] = np.cumsum([item.thickness_m for item in layers])
        boundaries.setflags(write=False)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "boundaries_m", boundaries)
        object.__setattr__(self, "electrical_offset_m", sum(item.thickness_m for item in layers if not item.electrical))
        object.__setattr__(self, "physical_layer_slices", tuple(groups))
        object.__setattr__(self, "sha256", self._identity())

    def _identity(self) -> str:
        digest = hashlib.sha256()
        header = {
            "model": "transfer_matrix_front_incidence_v1",
            "ambient_index": repr(self.ambient_index),
            "exit_index": repr(self.exit_index),
            "incidence": self.incidence,
            "dependencies": [list(item) for item in self.dependencies],
            "spectrum": {"source_id": self.spectrum.source_id, "unit": self.spectrum.unit,
                         "sha256": self.spectrum.sha256},
            "layers": [
                {"name": item.name, "thickness_m": repr(item.thickness_m),
                 "incoherent": item.incoherent, "electrical": item.electrical}
                for item in self.layers
            ],
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
        digest.update(array_digest(*[array for item in self.layers for array in (item.n, item.k)]).encode("ascii"))
        return digest.hexdigest()

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return self.spectrum.wavelengths_nm

    @property
    def wavelengths_m(self) -> np.ndarray:
        return wavelength_grid_m(self.spectrum.wavelengths_nm)

    @property
    def tmm_layers(self) -> tuple[TMMLayer, ...]:
        return tuple(TMMLayer(d=item.thickness_m, n=item.n, k=item.k, incoherent=item.incoherent) for item in self.layers)

    @property
    def electrical_thickness_m(self) -> float:
        return float(self.boundaries_m[-1]) - self.electrical_offset_m

    def _optical_faces(self, electrical_faces_m: np.ndarray) -> np.ndarray:
        faces = require_cell_faces(electrical_faces_m)
        optical = faces + self.electrical_offset_m
        if (
            float(faces[0]) < -_WINDOW_TOLERANCE_M
            or float(optical[-1]) > float(self.boundaries_m[-1]) + _WINDOW_TOLERANCE_M
        ):
            raise ValueError("cell faces must lie inside the electrical window of the optical stack")
        return optical

    def absorbed_photon_flux_per_cell(self, electrical_faces_m: np.ndarray) -> np.ndarray:
        """Exact photons absorbed in each electrical cell [m^-2 s^-1]."""

        absorbed = tmm_absorbed_photon_flux_per_cell(
            list(self.tmm_layers),
            self.wavelengths_m,
            self.spectrum.flux_m2_s_m,
            self._optical_faces(electrical_faces_m),
            self.boundaries_m,
            self.ambient_index,
            self.exit_index,
        )
        absorbed.setflags(write=False)
        return absorbed

    def generation_profile(self, control_volumes: ControlVolumes) -> GenerationProfile:
        """Photon-conserving TMM cell source ``absorbed / w`` on named control volumes.

        ``control_volumes`` live in the electrical coordinate; their faces
        are shifted by ``electrical_offset_m`` into the optical coordinate.
        """

        _, faces, measures = require_control_volumes(control_volumes)
        absorbed = self.absorbed_photon_flux_per_cell(faces)
        return GenerationProfile(
            method="transfer_matrix",
            geometry_name=str(control_volumes.name),
            source_m3_s=absorbed / measures,
            absorbed_m2_s=absorbed,
            binding_sha256=self.sha256,
            control_volume_sha256=control_volume_digest(control_volumes),
        )

    def absorption_profile(self, x_electrical_m: np.ndarray) -> np.ndarray:
        """Point-sampled ``A(x, lambda)`` [m^-1] at electrical positions (diagnostic)."""

        x = require_node_coordinates(x_electrical_m)
        return tmm_absorption_profile(
            list(self.tmm_layers),
            self.wavelengths_m,
            x + self.electrical_offset_m,
            self.boundaries_m,
            self.ambient_index,
            self.exit_index,
        )

    def photon_budget(self) -> PhotonBudget:
        layers = list(self.tmm_layers)
        wavelengths = self.wavelengths_m
        C = tmm_cumulative_absorptance(
            layers, wavelengths, self.boundaries_m, self.boundaries_m, self.ambient_index, self.exit_index
        )
        return PhotonBudget(
            wavelengths_nm=self.wavelengths_nm,
            incident_spectral_m2_s_m=self.spectrum.flux_m2_s_m,
            reflectance=tmm_reflectance(layers, wavelengths, self.ambient_index, self.exit_index),
            transmittance=tmm_transmittance(layers, wavelengths, self.ambient_index, self.exit_index),
            layer_absorptance=np.diff(C, axis=0),
            dropped_multipass=dropped_multipass_fraction(layers, wavelengths, self.ambient_index, self.exit_index),
            layer_names=tuple(item.name for item in self.layers),
            layer_electrical=tuple(item.electrical for item in self.layers),
        )

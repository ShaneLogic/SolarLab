"""Single-pass spectral control-volume Beer-Lambert optics (P08-04).

Uses the same material ``k(lambda)``, wavelength grid, front-prefix substrate
coordinate convention, and ``SpectralPhotonFlux`` as ``OpticalStackBinding``,
without Fresnel reflection, coherent wave interference, or TMM-derived photon
recycling.

Per-layer attenuation uses ``alpha_l(lambda) = 4 * pi * k_l(lambda) / lambda``.
For cumulative optical depth ``tau(z, lambda)``, exact control-volume cell
absorptance is evaluated without thin-cell cancellation via:

    absorbed(cell, lambda) = exp(-tau[:-1]) * -expm1(-diff(tau, axis=0))

so per-wavelength and trapezoid-integrated photon accounting close to machine
precision: ``transmitted + sum_layers absorbed == incident`` with zero
reflection and zero dropped multipass fraction. Optical-only front layers
(``electrical=False``, e.g. glass substrate) attenuate the spectrum entering
the electrical stack without creating electrical carriers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from solarlab.physics.generation._validation import (
    array_digest,
    require_cell_faces,
    require_node_coordinates,
)
from solarlab.physics.generation.optical_binding import (
    INCIDENCE_SIDES,
    OpticalLayerSpec,
    OpticalStackBinding,
    PhotonBudget,
)
from solarlab.physics.generation.profile import (
    ControlVolumes,
    GenerationProfile,
    control_volume_digest,
    require_control_volumes,
)
from solarlab.physics.generation.spectrum import SpectralPhotonFlux, wavelength_grid_m

__all__ = [
    "SPECTRAL_BEER_LAMBERT_MODEL_ID",
    "SpectralBeerLambertBinding",
    "spectral_beer_lambert_absorbed_photon_flux_per_cell",
    "spectral_beer_lambert_absorption_coefficients",
    "spectral_beer_lambert_absorption_profile",
    "spectral_beer_lambert_optical_depth",
]

SPECTRAL_BEER_LAMBERT_MODEL_ID = "beer_lambert_spectral_front_incidence_v1"
_WINDOW_TOLERANCE_M = 1e-15


def _canonical_dependencies(
    value: Mapping[str, object] | Sequence[tuple[str, object]],
) -> tuple[tuple[str, str], ...]:
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
        rendered = repr(float(raw)) if isinstance(raw, (int, float)) else raw
        canonical.append((key, rendered))
    return tuple(canonical)


def _require_wavelengths_m(wavelengths_m: NDArray[np.floating]) -> NDArray[np.float64]:
    wl = np.asarray(wavelengths_m, dtype=float)
    if wl.ndim != 1 or wl.shape[0] < 2:
        raise ValueError(
            f"wavelengths_m must be 1-D with at least 2 entries, got {wl.shape}"
        )
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0) or np.any(np.diff(wl) <= 0.0):
        raise ValueError(
            "wavelengths_m must be finite, positive, and strictly increasing"
        )
    return wl


def spectral_beer_lambert_absorption_coefficients(
    layers: Sequence[OpticalLayerSpec],
    wavelengths_m: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return per-layer ``alpha_l(lambda) = 4 * pi * k_l / lambda`` [m^-1]."""

    if not layers:
        raise ValueError("at least one OpticalLayerSpec is required")
    wl = _require_wavelengths_m(wavelengths_m)
    rows: list[NDArray[np.float64]] = []
    for layer in layers:
        if not isinstance(layer, OpticalLayerSpec):
            raise TypeError("layers must contain OpticalLayerSpec instances")
        if layer.k.shape != wl.shape:
            raise ValueError(
                f"{layer.name}: k shape {layer.k.shape} does not match {wl.shape}"
            )
        alpha = 4.0 * np.pi * layer.k / wl
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0):
            raise ValueError(
                f"{layer.name}: Beer-Lambert absorption coefficients must be "
                "finite and nonnegative"
            )
        rows.append(alpha)
    return np.stack(rows, axis=0)


def spectral_beer_lambert_optical_depth(
    optical_positions_m: NDArray[np.floating],
    layers: Sequence[OpticalLayerSpec],
    boundaries_m: NDArray[np.floating],
    wavelengths_m: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Cumulative optical depth ``tau(z, lambda)`` at optical coordinates ``z``."""

    positions = np.asarray(optical_positions_m, dtype=float)
    if positions.ndim != 1 or positions.shape[0] < 1 or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError("optical_positions_m must be a finite non-empty 1-D array")
    bounds = np.asarray(boundaries_m, dtype=float)
    if bounds.shape != (len(layers) + 1,):
        raise ValueError("boundaries_m must have length len(layers) + 1")
    if (
        float(np.min(positions)) < float(bounds[0]) - _WINDOW_TOLERANCE_M
        or float(np.max(positions)) > float(bounds[-1]) + _WINDOW_TOLERANCE_M
    ):
        raise ValueError("optical positions must lie within the optical stack bounds")
    alphas = spectral_beer_lambert_absorption_coefficients(layers, wavelengths_m)
    tau: NDArray[np.float64] = np.zeros(
        (positions.shape[0], alphas.shape[1]), dtype=float
    )
    for index, layer in enumerate(layers):
        distance = np.clip(positions - bounds[index], 0.0, layer.thickness_m)
        tau += distance[:, None] * alphas[index][None, :]
    return tau


def spectral_beer_lambert_absorbed_photon_flux_per_cell(
    layers: Sequence[OpticalLayerSpec],
    wavelengths_m: NDArray[np.floating],
    spectral_flux_m2_s_m: NDArray[np.floating],
    optical_faces_m: NDArray[np.floating],
    boundaries_m: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Exact single-pass photons absorbed per control volume [m^-2 s^-1]."""

    faces = require_cell_faces(optical_faces_m)
    wl = _require_wavelengths_m(wavelengths_m)
    flux = np.asarray(spectral_flux_m2_s_m, dtype=float)
    if flux.shape != wl.shape or not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
        raise ValueError(
            "spectral_flux_m2_s_m must be finite, nonnegative, and match wavelengths_m"
        )
    tau = spectral_beer_lambert_optical_depth(faces, layers, boundaries_m, wl)
    absorbed_fraction = np.exp(-tau[:-1]) * -np.expm1(-np.diff(tau, axis=0))
    integrated = np.trapezoid(
        absorbed_fraction * flux[None, :],
        x=wl,
        axis=1,
    )
    return np.asarray(integrated, dtype=float)


def spectral_beer_lambert_absorption_profile(
    layers: Sequence[OpticalLayerSpec],
    wavelengths_m: NDArray[np.floating],
    x_optical_m: NDArray[np.floating],
    boundaries_m: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Point-sampled absorptance density ``alpha(x, lambda) * exp(-tau(x, lambda))``."""

    x = require_node_coordinates(x_optical_m)
    wl = _require_wavelengths_m(wavelengths_m)
    bounds = np.asarray(boundaries_m, dtype=float)
    alphas = spectral_beer_lambert_absorption_coefficients(layers, wl)
    tau = spectral_beer_lambert_optical_depth(x, layers, bounds, wl)
    # Right-layer ownership on interior planes, clamping the back endpoint to L-1.
    layer_idx = np.clip(
        np.searchsorted(bounds[1:], x, side="right"),
        0,
        len(layers) - 1,
    )
    return np.asarray(alphas[layer_idx] * np.exp(-tau), dtype=float)


@dataclass(frozen=True)
class SpectralBeerLambertBinding:
    """Frozen single-pass spectral Beer-Lambert optical stack and identity."""

    layers: tuple[OpticalLayerSpec, ...]
    spectrum: SpectralPhotonFlux
    incidence: str = "front"
    dependencies: tuple[tuple[str, str], ...] = ()
    sha256: str = field(init=False)
    boundaries_m: NDArray[np.float64] = field(init=False, repr=False)
    electrical_offset_m: float = field(init=False)
    physical_layer_slices: tuple[tuple[str, slice], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers or not all(
            isinstance(item, OpticalLayerSpec) for item in layers
        ):
            raise ValueError(
                "spectral Beer-Lambert binding requires OpticalLayerSpec layers"
            )
        if not isinstance(self.spectrum, SpectralPhotonFlux):
            raise TypeError("spectrum must be a SpectralPhotonFlux")
        n_wl = self.spectrum.wavelengths_nm.shape[0]
        if any(item.n.shape != (n_wl,) or item.k.shape != (n_wl,) for item in layers):
            raise ValueError(
                "every layer's n, k must be sampled on the spectrum's wavelength grid"
            )
        if any(item.incoherent for item in layers[1:]):
            raise ValueError("only the first optical layer may be incoherent")
        seen_electrical = False
        for item in layers:
            if item.electrical:
                seen_electrical = True
            elif seen_electrical:
                raise ValueError(
                    "optical-only layers must form a contiguous front prefix"
                )
        if not seen_electrical:
            raise ValueError(
                "spectral Beer-Lambert binding requires at least one electrical layer"
            )
        if self.incidence not in INCIDENCE_SIDES:
            raise ValueError(
                f"incidence must be one of {INCIDENCE_SIDES}; "
                "only front illumination exists"
            )
        spectral_beer_lambert_absorption_coefficients(
            layers, wavelength_grid_m(self.spectrum.wavelengths_nm)
        )
        groups: list[tuple[str, slice]] = []
        start = 0
        for index in range(1, len(layers) + 1):
            if index == len(layers) or layers[index].name != layers[start].name:
                groups.append((layers[start].name, slice(start, index)))
                start = index
        if len({name for name, _ in groups}) != len(groups):
            raise ValueError("slices of one physical layer must be contiguous")
        dependencies = _canonical_dependencies(self.dependencies)
        boundaries: NDArray[np.float64] = np.empty(len(layers) + 1, dtype=float)
        boundaries[0] = 0.0
        boundaries[1:] = np.cumsum([item.thickness_m for item in layers])
        boundaries.setflags(write=False)
        offset = float(sum(item.thickness_m for item in layers if not item.electrical))
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "boundaries_m", boundaries)
        object.__setattr__(self, "electrical_offset_m", offset)
        object.__setattr__(self, "physical_layer_slices", tuple(groups))
        object.__setattr__(self, "sha256", self._identity())

    @classmethod
    def from_optical_stack(
        cls,
        stack: OpticalStackBinding,
        *,
        dependencies: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    ) -> SpectralBeerLambertBinding:
        """Construct single-pass Beer-Lambert binding from an OpticalStackBinding."""

        if not isinstance(stack, OpticalStackBinding):
            raise TypeError("stack must be an OpticalStackBinding")
        deps = stack.dependencies if dependencies is None else _canonical_dependencies(
            dependencies
        )
        return cls(
            layers=stack.layers,
            spectrum=stack.spectrum,
            incidence=stack.incidence,
            dependencies=deps,
        )

    def _identity(self) -> str:
        digest = hashlib.sha256()
        header = {
            "model": SPECTRAL_BEER_LAMBERT_MODEL_ID,
            "incidence": self.incidence,
            "dependencies": [list(item) for item in self.dependencies],
            "spectrum": {
                "source_id": self.spectrum.source_id,
                "unit": self.spectrum.unit,
                "sha256": self.spectrum.sha256,
            },
            "layers": [
                {
                    "name": item.name,
                    "thickness_m": repr(item.thickness_m),
                    "electrical": item.electrical,
                }
                for item in self.layers
            ],
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(
            array_digest(*[item.k for item in self.layers]).encode("ascii")
        )
        return digest.hexdigest()

    @property
    def wavelengths_nm(self) -> NDArray[np.float64]:
        return self.spectrum.wavelengths_nm

    @property
    def wavelengths_m(self) -> NDArray[np.float64]:
        return wavelength_grid_m(self.spectrum.wavelengths_nm)

    @property
    def electrical_thickness_m(self) -> float:
        return float(self.boundaries_m[-1]) - self.electrical_offset_m

    def _optical_faces(
        self, electrical_faces_m: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        faces = require_cell_faces(electrical_faces_m)
        optical = faces + self.electrical_offset_m
        if (
            float(faces[0]) < -_WINDOW_TOLERANCE_M
            or float(optical[-1]) > float(self.boundaries_m[-1]) + _WINDOW_TOLERANCE_M
        ):
            raise ValueError(
                "cell faces must lie inside the electrical window of the optical stack"
            )
        return optical

    def absorbed_photon_flux_per_cell(
        self, electrical_faces_m: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Exact photons absorbed in each electrical cell [m^-2 s^-1]."""

        absorbed = spectral_beer_lambert_absorbed_photon_flux_per_cell(
            self.layers,
            self.wavelengths_m,
            self.spectrum.flux_m2_s_m,
            self._optical_faces(electrical_faces_m),
            self.boundaries_m,
        )
        absorbed.setflags(write=False)
        return absorbed

    def generation_profile(self, control_volumes: ControlVolumes) -> GenerationProfile:
        """Photon-conserving spectral Beer-Lambert cell source on control volumes."""

        _, faces, measures = require_control_volumes(control_volumes)
        absorbed = self.absorbed_photon_flux_per_cell(faces)
        return GenerationProfile(
            method="beer_lambert_spectral",
            geometry_name=str(control_volumes.name),
            source_m3_s=absorbed / measures,
            absorbed_m2_s=absorbed,
            binding_sha256=self.sha256,
            control_volume_sha256=control_volume_digest(control_volumes),
        )

    def absorption_profile(
        self, x_electrical_m: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Point-sampled ``A(x, lambda)`` [m^-1] at electrical positions."""

        x = require_node_coordinates(x_electrical_m)
        optical = x + self.electrical_offset_m
        if (
            float(x[0]) < -_WINDOW_TOLERANCE_M
            or float(optical[-1]) > float(self.boundaries_m[-1]) + _WINDOW_TOLERANCE_M
        ):
            raise ValueError(
                "node coordinates must lie inside the electrical window "
                "of the optical stack"
            )
        return spectral_beer_lambert_absorption_profile(
            self.layers,
            self.wavelengths_m,
            optical,
            self.boundaries_m,
        )

    def photon_budget(self) -> PhotonBudget:
        """Exact single-pass photon bookkeeping (zero reflection and multipass drop)."""

        wl = self.wavelengths_m
        tau_bounds = spectral_beer_lambert_optical_depth(
            self.boundaries_m, self.layers, self.boundaries_m, wl
        )
        layer_absorptance = np.exp(-tau_bounds[:-1]) * -np.expm1(
            -np.diff(tau_bounds, axis=0)
        )
        transmittance = np.exp(-tau_bounds[-1])
        zeros = np.zeros_like(wl)
        return PhotonBudget(
            wavelengths_nm=self.wavelengths_nm,
            incident_spectral_m2_s_m=self.spectrum.flux_m2_s_m,
            reflectance=zeros,
            transmittance=transmittance,
            layer_absorptance=layer_absorptance,
            dropped_multipass=zeros.copy(),
            layer_names=tuple(item.name for item in self.layers),
            layer_electrical=tuple(item.electrical for item in self.layers),
        )

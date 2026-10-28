"""Tabulated optical constants and spectral photon flux on a wavelength grid.

Tables are supplied as arrays by the caller (no path discovery here). The
sampling is the historical one: linear interpolation on the nanometre axis,
and a hard failure instead of ``np.interp``'s silent edge clamping for any
requested wavelength outside the table -- clamping ``k`` or the AM1.5G edge
value is exactly the historical spurious-absorption / inflated-J_sc bug.

Spectral unit: the photon flux per unit wavelength is in m^-2 s^-1 m^-1 and
is integrated over wavelength in metres, so ``int Phi dlam`` is a photon
flux in m^-2 s^-1. Irradiance (W m^-2 nm^-1) is a different quantity and is
rejected by unit name rather than converted implicitly.

Interface switch point (S5): WS2's ``materials.resources.ResourceTable``
stores wavelengths in metres; reaching these tables through it must keep the
nanometre sampling axis (or be recorded as an N-class change).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from solarlab.physics.generation._validation import array_digest

__all__ = [
    "SPECTRAL_PHOTON_FLUX_UNIT",
    "SpectralPhotonFlux",
    "sample_nk_table",
    "sample_tabulated",
    "wavelength_grid_m",
]

SPECTRAL_PHOTON_FLUX_UNIT = "m^-2 s^-1 m^-1"


def wavelength_grid_m(wavelengths_nm: np.ndarray) -> np.ndarray:
    """The metre grid every optical integral uses (historical ``nm * 1e-9``)."""

    return np.asarray(wavelengths_nm, dtype=float) * 1e-9


def _require_table_axis(table_wavelength_nm: np.ndarray) -> np.ndarray:
    axis = np.asarray(table_wavelength_nm, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("table wavelength axis must be 1-D with at least two rows")
    if not np.all(np.isfinite(axis)) or np.any(axis <= 0.0) or np.any(np.diff(axis) <= 0.0):
        raise ValueError("table wavelengths must be finite, positive and strictly increasing")
    return axis


def sample_tabulated(
    table_wavelength_nm: np.ndarray,
    table_values: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Interpolate one tabulated column onto ``wavelengths_nm``; never extrapolate."""

    axis = _require_table_axis(table_wavelength_nm)
    values = np.asarray(table_values, dtype=float)
    if values.shape != axis.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: table values must be finite and match the wavelength axis")
    wl_req = np.asarray(wavelengths_nm, dtype=float)
    lo, hi = float(axis[0]), float(axis[-1])
    if wl_req.size and (wl_req.min() < lo or wl_req.max() > hi):
        raise ValueError(
            f"Requested wavelengths [{wl_req.min():.2f}, {wl_req.max():.2f}] nm "
            f"are outside the native range of {name!r} [{lo:.2f}, {hi:.2f}] nm."
        )
    return np.interp(wl_req, axis, values)


def sample_nk_table(
    table_wavelength_nm: np.ndarray,
    table_n: np.ndarray,
    table_k: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    material: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, k)`` of one material on ``wavelengths_nm`` (historical ``load_nk``)."""

    n = sample_tabulated(table_wavelength_nm, table_n, wavelengths_nm, name=material)
    k = sample_tabulated(table_wavelength_nm, table_k, wavelengths_nm, name=material)
    if np.any(n <= 0.0) or np.any(k < 0.0):
        raise ValueError(f"{material}: n must be positive and k non-negative")
    return n, k


@dataclass(frozen=True)
class SpectralPhotonFlux:
    """Incident spectral photon flux sampled on one wavelength grid."""

    wavelengths_nm: np.ndarray
    flux_m2_s_m: np.ndarray
    source_id: str
    unit: str = SPECTRAL_PHOTON_FLUX_UNIT
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.unit != SPECTRAL_PHOTON_FLUX_UNIT:
            raise ValueError(
                f"spectral photon flux must be in {SPECTRAL_PHOTON_FLUX_UNIT}, got {self.unit!r}"
            )
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("spectrum requires an explicit source_id")
        wavelengths = np.array(self.wavelengths_nm, dtype=float, copy=True)
        flux = np.array(self.flux_m2_s_m, dtype=float, copy=True)
        if wavelengths.ndim != 1 or wavelengths.size < 2:
            raise ValueError("spectrum wavelengths must be 1-D with at least two points")
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0) or np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("spectrum wavelengths must be finite, positive and strictly increasing")
        if flux.shape != wavelengths.shape or not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
            raise ValueError("spectral flux must be finite, non-negative and match the wavelengths")
        wavelengths.setflags(write=False)
        flux.setflags(write=False)
        object.__setattr__(self, "wavelengths_nm", wavelengths)
        object.__setattr__(self, "flux_m2_s_m", flux)
        object.__setattr__(self, "sha256", array_digest(wavelengths, flux))

    @classmethod
    def from_table(
        cls,
        table_wavelength_nm: np.ndarray,
        table_flux_m2_s_m: np.ndarray,
        wavelengths_nm: np.ndarray,
        *,
        source_id: str,
    ) -> SpectralPhotonFlux:
        flux = sample_tabulated(table_wavelength_nm, table_flux_m2_s_m, wavelengths_nm, name=source_id)
        return cls(np.asarray(wavelengths_nm, dtype=float), flux, source_id)

    @property
    def wavelengths_m(self) -> np.ndarray:
        return wavelength_grid_m(self.wavelengths_nm)

    def integrated_photon_flux_m2_s(self) -> float:
        """``int Phi dlam`` over this grid (trapezoid in metres) [m^-2 s^-1]."""

        return float(np.trapezoid(self.flux_m2_s_m, x=self.wavelengths_m))

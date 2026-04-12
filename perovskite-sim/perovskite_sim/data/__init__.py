"""Optical material data and spectral resources.

Provides loaders for n(lambda)/k(lambda) optical constants and AM1.5G
spectral photon flux data shipped with the package.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent


def load_nk(material: str, wavelengths_nm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load optical constants n(lambda), k(lambda) for a material.

    Args:
        material: one of "MAPbI3", "TiO2", "spiro_OMeTAD"
        wavelengths_nm: if provided, interpolate to these wavelengths [nm].
                        Otherwise return the native grid.

    Returns:
        (wavelengths_nm, n, k) — all shape (n_wl,)
    """
    path = _DATA_DIR / "nk" / f"{material}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No optical data for {material!r} at {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    wl_native = data[:, 0]
    n_native = data[:, 1]
    k_native = data[:, 2]
    if wavelengths_nm is None:
        return wl_native, n_native, k_native
    n_interp = np.interp(wavelengths_nm, wl_native, n_native)
    k_interp = np.interp(wavelengths_nm, wl_native, k_native)
    return wavelengths_nm, n_interp, k_interp


def load_am15g(wavelengths_nm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load AM1.5G spectral photon flux.

    Args:
        wavelengths_nm: if provided, interpolate to these wavelengths [nm].

    Returns:
        (wavelengths_nm, spectral_flux) where spectral_flux is in [m^-2 s^-1 m^-1]
    """
    path = _DATA_DIR / "am15g.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    wl_native = data[:, 0]
    flux_native = data[:, 1]
    if wavelengths_nm is None:
        return wl_native, flux_native
    flux_interp = np.interp(wavelengths_nm, wl_native, flux_native)
    return wavelengths_nm, flux_interp

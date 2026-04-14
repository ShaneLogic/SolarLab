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
        material: name of a CSV under ``data/nk/`` (e.g. "MAPbI3", "TiO2",
            "spiro_OMeTAD", "FTO", ...). See ``data/nk/`` for the full list.
        wavelengths_nm: if provided, interpolate to these wavelengths [nm].
            All requested wavelengths must lie within the native file range;
            out-of-range wavelengths raise ``ValueError`` rather than silently
            clamping. Otherwise return the native grid.

    Returns:
        (wavelengths_nm, n, k) — all shape (n_wl,)

    Raises:
        FileNotFoundError: if no CSV is shipped for ``material``.
        ValueError: if any requested wavelength falls outside the native
            range of the material's CSV. This is a hard failure because
            ``np.interp`` silently clamps edge values, and clamping the
            imaginary index ``k`` extends spurious absorption (or spurious
            transparency) into a region where the material was never
            measured -- historically the same class of bug the AM1.5G
            loader already guards against.
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
    wl_req = np.asarray(wavelengths_nm, dtype=float)
    lo, hi = float(wl_native[0]), float(wl_native[-1])
    if wl_req.size and (wl_req.min() < lo or wl_req.max() > hi):
        raise ValueError(
            f"Requested wavelengths [{wl_req.min():.2f}, {wl_req.max():.2f}] nm "
            f"are outside the native range of {material!r} "
            f"[{lo:.2f}, {hi:.2f}] nm. Extend the CSV in data/nk/{material}.csv "
            "or narrow the TMM wavelength grid (lam_min/lam_max in "
            "_compute_tmm_generation)."
        )
    n_interp = np.interp(wl_req, wl_native, n_native)
    k_interp = np.interp(wl_req, wl_native, k_native)
    return wl_req, n_interp, k_interp


def load_am15g(wavelengths_nm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load AM1.5G spectral photon flux.

    Args:
        wavelengths_nm: if provided, interpolate to these wavelengths [nm].
            All requested wavelengths must lie within the native file range;
            extrapolation raises ``ValueError`` rather than silently clamping.

    Returns:
        (wavelengths_nm, spectral_flux) where spectral_flux is in [m^-2 s^-1 m^-1]

    Raises:
        ValueError: if any requested wavelength falls outside the native
            range of the shipped AM1.5G file. This is a hard failure to
            prevent the historical bug where ``np.interp`` silently clamped
            the inflated edge value through the IR, inflating J_sc.
    """
    path = _DATA_DIR / "am15g.csv"
    # The file has an arbitrary number of leading '#' comment lines followed
    # by a single column-name header row, then numeric data. np.loadtxt's
    # skiprows= is applied *before* comment stripping, so we read rows and
    # drop the header row manually after comment filtering.
    data = np.loadtxt(path, delimiter=",", comments="#")
    wl_native = data[:, 0]
    flux_native = data[:, 1]
    if wavelengths_nm is None:
        return wl_native, flux_native
    wl_req = np.asarray(wavelengths_nm, dtype=float)
    lo, hi = float(wl_native[0]), float(wl_native[-1])
    if wl_req.size and (wl_req.min() < lo or wl_req.max() > hi):
        raise ValueError(
            f"Requested wavelengths [{wl_req.min():.2f}, {wl_req.max():.2f}] nm "
            f"are outside the native AM1.5G range [{lo:.2f}, {hi:.2f}] nm. "
            "Extend the source file or narrow your TMM wavelength grid."
        )
    flux_interp = np.interp(wl_req, wl_native, flux_native)
    return wl_req, flux_interp

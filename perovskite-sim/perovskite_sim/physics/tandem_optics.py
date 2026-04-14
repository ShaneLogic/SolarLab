"""Combined-TMM absorption partitioning for 2T monolithic tandem cells.

Runs a single TMM over the full stack (top sub-cell + junction + bottom
sub-cell), then splits the per-layer absorption profiles into per-sub-cell
generation rate arrays G_top(x) and G_bot(x) [m^-3 s^-1].

The junction layers between the two sub-cells act as recombination layers;
photons absorbed there are counted as parasitic_absorption and excluded from
both sub-cell generation profiles.

Design contract:
    - wavelengths are always in METRES throughout this module.
    - load_nk is called with wavelengths in NANOMETRES (the existing data API).
    - TMMLayer field names match physics/optics.py: d, n, k, incoherent.
    - DeviceStack layers expose .params.optical_material (or .params.n_optical /
      .params.alpha / .params.eps_r as fallbacks) — identical to mol.py.
    - JunctionLayer exposes .optical_material directly (no nested .params).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perovskite_sim.physics.optics import TMMLayer, tmm_absorption_profile
from perovskite_sim.models.tandem_config import TandemConfig, JunctionLayer
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.data import load_nk


@dataclass(frozen=True)
class TandemGeneration:
    """Immutable result bundle from compute_tandem_generation.

    Attributes:
        G_top: generation profile for the top sub-cell [m^-3 s^-1],
               shape (N_top,)
        G_bot: generation profile for the bottom sub-cell [m^-3 s^-1],
               shape (N_bot,)
        parasitic_absorption: fraction of incident photon flux absorbed
                              in the junction / recombination layers
        top_layer_slice: slice into the combined x-grid that spans G_top
        bottom_layer_slice: slice into the combined x-grid that spans G_bot
    """
    G_top: np.ndarray
    G_bot: np.ndarray
    parasitic_absorption: float
    top_layer_slice: slice
    bottom_layer_slice: slice


def partition_absorption(
    A: np.ndarray,              # (N, n_wl)  absorption rate [m^-1]
    x: np.ndarray,              # (N,)       spatial grid [m]
    wavelengths: np.ndarray,    # (n_wl,)    metres
    spectral_flux: np.ndarray,  # (n_wl,)    photon flux [m^-2 s^-1 m^-1]
    top_slice: slice,
    junction_slice: slice,
    bottom_slice: slice,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split combined-stack absorption into per-sub-cell generation profiles.

    Integrates A(x, λ) * Φ(λ) over wavelength to get G(x), then extracts
    the top-cell, junction, and bottom-cell sub-arrays.

    The parasitic fraction is defined as the residual:

        parasitic = (full_absorbed - top_absorbed - bot_absorbed) / total_incident

    where each term is a trapezoid integral of G over the appropriate sub-range
    of x.  This definition ensures that::

        np.trapezoid(G_top, x[top_slice])
        + parasitic * np.trapezoid(spectral_flux, wavelengths)
        + np.trapezoid(G_bot, x[bottom_slice])
        == np.trapezoid(G_full, x)

    exactly (to floating-point precision), which is the conservation identity
    required by the test suite.

    The residual differs from a naive junction-only integral because
    np.trapezoid on disjoint sub-ranges does not partition the full integral:
    boundary grid points at the top/junction and junction/bottom interfaces
    contribute half their trapezoid weight to both adjacent sub-ranges, so a
    direct junction integral would under-account for those shared contributions.
    The residual formulation captures them correctly.

    Args:
        A: spectral absorption rate [m^-1], shape (N, n_wl)
        x: spatial grid [m], shape (N,)
        wavelengths: wavelength array [m], shape (n_wl,)
        spectral_flux: spectral photon flux [m^-2 s^-1 m^-1], shape (n_wl,)
        top_slice: index slice for the top sub-cell region of x
        junction_slice: index slice for the junction / recombination layers
        bottom_slice: index slice for the bottom sub-cell region of x

    Returns:
        G_top: generation rate profile [m^-3 s^-1] for the top sub-cell
        G_bot: generation rate profile [m^-3 s^-1] for the bottom sub-cell
        parasitic_fraction: dimensionless residual fraction in [0, 1)
    """
    integrand = A * spectral_flux[None, :]                  # (N, n_wl)
    G_full = np.trapezoid(integrand, wavelengths, axis=1)   # (N,)

    G_top = G_full[top_slice]
    G_bot = G_full[bottom_slice]

    total_incident = float(np.trapezoid(spectral_flux, wavelengths))
    if total_incident <= 0:
        return G_top, G_bot, 0.0

    full_absorbed = float(np.trapezoid(G_full, x))
    top_absorbed = float(np.trapezoid(G_top, x[top_slice]))
    bot_absorbed = float(np.trapezoid(G_bot, x[bottom_slice]))
    parasitic = (full_absorbed - top_absorbed - bot_absorbed) / total_incident

    return G_top, G_bot, max(0.0, parasitic)


def _tmm_layer_from_stack_layer(layer, wavelengths_nm: np.ndarray) -> TMMLayer:
    """Build a TMMLayer from a DeviceStack LayerSpec.

    Mirrors the adapter in solver/mol.py:_compute_tmm_generation so that
    both paths use the same material priority order:
      1. optical_material CSV (n, k from file)
      2. n_optical constant (k derived from scalar alpha)
      3. fallback: sqrt(eps_r) for n, alpha-derived k

    Args:
        layer: a LayerSpec (has .thickness and .params)
        wavelengths_nm: wavelength grid in nanometres
    """
    p = layer.params
    n_wl = len(wavelengths_nm)
    wavelengths_m = wavelengths_nm * 1e-9

    if p is not None and p.optical_material is not None:
        _, n_arr, k_arr = load_nk(p.optical_material, wavelengths_nm)
    elif p is not None and p.n_optical is not None:
        n_arr = np.full(n_wl, p.n_optical)
        k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
    elif p is not None:
        n_arr = np.full(n_wl, np.sqrt(p.eps_r))
        k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
    else:
        # Layer has no params — transparent placeholder
        n_arr = np.ones(n_wl)
        k_arr = np.zeros(n_wl)

    incoherent = bool(p.incoherent) if p is not None and hasattr(p, "incoherent") else False
    return TMMLayer(d=layer.thickness, n=n_arr, k=k_arr, incoherent=incoherent)


def _tmm_layer_from_junction_layer(
    jlayer: JunctionLayer,
    wavelengths_nm: np.ndarray,
) -> TMMLayer:
    """Build a TMMLayer from a JunctionLayer (tandem_config.py).

    JunctionLayer exposes optical_material directly (no nested .params).
    """
    _, n_arr, k_arr = load_nk(jlayer.optical_material, wavelengths_nm)
    return TMMLayer(
        d=jlayer.thickness,
        n=n_arr,
        k=k_arr,
        incoherent=jlayer.incoherent,
    )


def _build_tmm_layers_from_stack(
    stack: DeviceStack,
    wavelengths_nm: np.ndarray,
) -> list[TMMLayer]:
    """Convert every layer in a DeviceStack to a TMMLayer for the TMM solver."""
    return [_tmm_layer_from_stack_layer(layer, wavelengths_nm) for layer in stack.layers]


def compute_tandem_generation(
    cfg: TandemConfig,
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    wavelengths_nm: np.ndarray,
    N_top: int,
    N_bot: int,
) -> TandemGeneration:
    """Run combined-TMM and partition absorption into per-sub-cell profiles.

    Constructs one TMM stack covering top_cell + junction_stack + bottom_cell,
    calls tmm_absorption_profile once, then delegates to partition_absorption
    to split the result.

    Args:
        cfg: tandem device configuration
        wavelengths: wavelength array in METRES, shape (n_wl,) — passed to TMM
        spectral_flux: photon flux [m^-2 s^-1 m^-1], shape (n_wl,)
        wavelengths_nm: same wavelengths in NANOMETRES — used by load_nk
        N_top: number of spatial grid points in the top sub-cell
        N_bot: number of spatial grid points in the bottom sub-cell

    Returns:
        TandemGeneration with G_top, G_bot, parasitic_absorption and slice info.
    """
    # --- Build per-section TMMLayer lists ---
    top_tmm = _build_tmm_layers_from_stack(cfg.top_cell, wavelengths_nm)
    bot_tmm = _build_tmm_layers_from_stack(cfg.bottom_cell, wavelengths_nm)
    junc_tmm = [
        _tmm_layer_from_junction_layer(j, wavelengths_nm)
        for j in cfg.junction_stack
    ]

    combined = top_tmm + junc_tmm + bot_tmm
    n_top = len(top_tmm)
    n_junc = len(junc_tmm)

    # --- Build cumulative layer boundaries ---
    thicknesses = np.array([L.d for L in combined])
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    total_thickness = float(boundaries[-1])

    top_end = float(boundaries[n_top])
    junc_end = float(boundaries[n_top + n_junc])

    # --- Spatial grid: top / junction interior / bottom ---
    x_top = np.linspace(0.0, top_end, N_top)
    n_junc_pts_full = max(3, n_junc * 3)
    x_junc = np.linspace(top_end, junc_end, n_junc_pts_full)
    x_bot = np.linspace(junc_end, total_thickness, N_bot)

    # Drop duplicate boundary points at top/junc and junc/bot interfaces.
    x = np.concatenate([x_top, x_junc[1:-1], x_bot])
    n_junc_pts = n_junc_pts_full - 2

    # --- Combined TMM absorption ---
    A = tmm_absorption_profile(
        combined, wavelengths, x, boundaries,
        n_ambient=1.0, n_substrate=1.0,
    )

    # --- Slice assignments ---
    top_slice = slice(0, N_top)
    junction_slice = slice(N_top, N_top + n_junc_pts)
    bottom_slice = slice(N_top + n_junc_pts, N_top + n_junc_pts + N_bot)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux,
        top_slice, junction_slice, bottom_slice,
    )

    return TandemGeneration(
        G_top=G_top,
        G_bot=G_bot,
        parasitic_absorption=parasitic,
        top_layer_slice=top_slice,
        bottom_layer_slice=bottom_slice,
    )

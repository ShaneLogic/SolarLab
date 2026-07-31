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

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.physics.generation import dual_cell_faces, dual_cell_widths
from perovskite_sim.physics.optics import (
    TMMLayer,
    tmm_absorbed_photon_flux_per_cell,
    tmm_absorption_profile,
)
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

    .. deprecated::
        UNUSED as of the tandem grid fix, and kept only so its two unit tests
        still document the old behaviour.  Do not wire it back in.  It
        point-samples ``A(x, lambda)`` and states its conservation identity in
        terms of ``np.trapezoid(G, x)`` -- and ``physics/generation.py`` is
        explicit that ``G`` is a CELL-AVERAGED quantity which must never be
        integrated with trapezoid.  ``compute_tandem_generation`` now uses
        ``_generation_on_grid``, which differences the closed-form cumulative
        absorptance across dual-cell faces and is photon-exact on any mesh.

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
    G_full = trapezoid(integrand, wavelengths, axis=1)      # (N,)

    G_top = G_full[top_slice]
    G_bot = G_full[bottom_slice]

    total_incident = float(trapezoid(spectral_flux, wavelengths))
    if total_incident <= 0:
        return G_top, G_bot, 0.0

    full_absorbed = float(trapezoid(G_full, x))
    top_absorbed = float(trapezoid(G_top, x[top_slice]))
    bot_absorbed = float(trapezoid(G_bot, x[bottom_slice]))
    parasitic = (full_absorbed - top_absorbed - bot_absorbed) / total_incident

    # Note: junction_slice is accepted for API symmetry; the junction contribution
    # is computed as the residual (full - top - bot) to make photon conservation
    # exact under disjoint-trapezoid quadrature at shared grid boundaries.
    return G_top, G_bot, min(1.0, max(0.0, parasitic))


def _generation_on_grid(
    combined: list[TMMLayer],
    wavelengths: np.ndarray,
    spectral_flux: np.ndarray,
    boundaries: np.ndarray,
    x_local: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Cell-exact generation [m^-3 s^-1] on one sub-cell's electrical grid.

    Mirrors ``solver/mol.py:_compute_tmm_generation``: every dual cell receives
    its OWN exactly-absorbed photon count from the closed-form cumulative
    absorptance, divided by the very weight ``carrier_continuity_rhs``
    multiplies back in.  ``sum(G * dx_cell)`` therefore telescopes to the
    photons absorbed between the sub-cell's outer faces on ANY mesh — the
    conservation is structural, not a refinement limit.

    ``x_local`` is measured from the sub-cell's own front face; ``offset``
    places it inside the combined tandem stack.
    """
    faces = dual_cell_faces(x_local) + offset
    absorbed = tmm_absorbed_photon_flux_per_cell(
        combined, wavelengths, spectral_flux, faces, boundaries,
        n_ambient=1.0, n_substrate=1.0,
    )
    G = absorbed / dual_cell_widths(x_local)
    if not np.all(np.isfinite(G)):
        raise ValueError(
            "tandem generation is not finite on this sub-cell grid "
            f"(offset {offset:.3e} m, {len(x_local)} nodes). The previous "
            "implementation silently replaced non-finite entries with zero, "
            "which hid the condition rather than reporting it."
        )
    return G


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
    # Enforce a tiny k floor: exact k=0 over multi-layer stacks can produce
    # singular cumulative transfer matrices and NaN field profiles.
    k_arr = np.maximum(k_arr, 1e-6)
    return TMMLayer(d=layer.thickness, n=n_arr, k=k_arr, incoherent=incoherent)


def _tmm_layer_from_junction_layer(
    jlayer: JunctionLayer,
    wavelengths_nm: np.ndarray,
) -> TMMLayer:
    """Build a TMMLayer from a JunctionLayer (tandem_config.py).

    JunctionLayer exposes optical_material directly (no nested .params).
    """
    _, n_arr, k_arr = load_nk(jlayer.optical_material, wavelengths_nm)
    k_arr = np.maximum(k_arr, 1e-6)
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
    x_top: np.ndarray,
    x_bot: np.ndarray,
) -> TandemGeneration:
    """Run combined-TMM and build per-sub-cell generation on the SOLVER's grid.

    Constructs one TMM stack covering top_cell + junction_stack + bottom_cell,
    then evaluates each sub-cell's generation with the cell-exact quadrature on
    the grid that sub-cell's drift-diffusion solve will actually integrate on.

    Why the grids are arguments and not node counts
    -----------------------------------------------
    ``G`` is a spatial DENSITY: index ``i`` is meaningless without the position
    of node ``i``.  This function used to take ``N_top`` / ``N_bot`` and sample
    on ``np.linspace(0, top_end, N_top, endpoint=False)`` — a UNIFORM grid —
    while its only consumer, ``run_jv_sweep(fixed_generation=...)``, integrates
    on the TANH-CLUSTERED multilayer grid.  The shape contract was enforced
    (``tandem_jv.py`` derives the counts from ``_grid_node_count``); the
    POSITION contract was not, so every value landed at the wrong depth.

    Measured on ``tandem_lin2019`` at ``N_grid=60``: the node-position mismatch
    peaked at 105 nm on the 500 nm top cell (21 % of its thickness) and 409 nm
    on the 1300 nm bottom cell (31 %), and because ``G(x)`` is steepest exactly
    where the two grids disagree most, the sub-cells received **+36.8 %** and
    **+18.2 %** more photons than the optics had computed as absorbed.

    Taking the grids removes the failure mode instead of documenting it: there
    is no longer a second grid that could disagree.

    Photon conservation
    -------------------
    Generation is built the way ``solver/mol.py:_compute_tmm_generation`` does
    it — each dual cell gets its OWN exactly-absorbed photon count from the
    closed-form cumulative absorptance, divided by the very weight the RHS
    multiplies back in.  ``sum(G * dx_cell)`` then telescopes to the photons
    absorbed between the sub-cell's outer faces on ANY mesh, so conservation is
    structural rather than asymptotic.  The previous route point-sampled
    ``A(x, lambda)`` and integrated with ``trapezoid``, which
    ``physics/generation.py`` explicitly forbids for a cell-averaged quantity.

    Args:
        cfg: tandem device configuration
        wavelengths: wavelength array in METRES, shape (n_wl,) — passed to TMM
        spectral_flux: photon flux [m^-2 s^-1 m^-1], shape (n_wl,)
        wavelengths_nm: same wavelengths in NANOMETRES — used by load_nk
        x_top: the top sub-cell's ELECTRICAL grid [m], measured from its own
            front face (i.e. starting at 0.0), exactly as built by
            ``run_jv_sweep``
        x_bot: the bottom sub-cell's ELECTRICAL grid [m], likewise from its own
            front face; it is offset internally by the top + junction thickness

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

    # Optional back reflector — optical-only, sits behind the bottom sub-cell
    # in the combined TMM stack. It never appears on the spatial x grid, so
    # its only effect is on the cumulative transfer matrix (i.e. it bounces
    # near-IR photons back into the bottom absorber for a second pass).
    back_tmm: list[TMMLayer] = []
    if cfg.back_reflector is not None:
        back_tmm = [_tmm_layer_from_junction_layer(cfg.back_reflector, wavelengths_nm)]

    combined = top_tmm + junc_tmm + bot_tmm + back_tmm
    n_top = len(top_tmm)
    n_junc = len(junc_tmm)
    n_bot = len(bot_tmm)

    # --- Build cumulative layer boundaries ---
    thicknesses = np.array([L.d for L in combined])
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    total_thickness = float(boundaries[-1])

    top_end = float(boundaries[n_top])
    junc_end = float(boundaries[n_top + n_junc])
    # x grid must not enter the back reflector — it is electrically absent.
    bot_end = float(boundaries[n_top + n_junc + n_bot])

    # --- Per-sub-cell generation, on the grid each solve actually uses ---
    # The top sub-cell occupies [0, top_end] of the combined stack; the bottom
    # occupies [junc_end, junc_end + its own thickness].  The back reflector is
    # optically present but electrically absent, so it never carries nodes.
    x_top = np.asarray(x_top, dtype=float)
    x_bot = np.asarray(x_bot, dtype=float)

    G_top = _generation_on_grid(
        combined, wavelengths, spectral_flux, boundaries, x_top, 0.0,
    )
    G_bot = _generation_on_grid(
        combined, wavelengths, spectral_flux, boundaries, x_bot, junc_end,
    )

    # --- Parasitic absorption: exact, not a trapezoid residual ---
    # One differencing of the closed-form cumulative absorptance across the
    # three section boundaries gives the photons absorbed in each section
    # directly, so the junction share needs no residual construction.
    total_incident = float(trapezoid(spectral_flux, wavelengths))
    if total_incident > 0.0:
        section_faces = np.array([0.0, top_end, junc_end, bot_end], dtype=float)
        per_section = tmm_absorbed_photon_flux_per_cell(
            combined, wavelengths, spectral_flux, section_faces, boundaries,
            n_ambient=1.0, n_substrate=1.0,
        )
        parasitic = float(per_section[1]) / total_incident
    else:
        parasitic = 0.0

    top_slice = slice(0, len(x_top))
    bottom_slice = slice(len(x_top), len(x_top) + len(x_bot))

    return TandemGeneration(
        G_top=G_top,
        G_bot=G_bot,
        parasitic_absorption=min(1.0, max(0.0, parasitic)),
        top_layer_slice=top_slice,
        bottom_layer_slice=bottom_slice,
    )

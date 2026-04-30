"""Self-consistent radiative reabsorption recompute for the 2D solver
(Stage B(c.3)).

On every RHS call, for each absorber layer:
  R_tot_2D = ∬ B(y,x) · n(y,x) · p(y,x) dy dx     over absorber rows × all x
  area     = thickness × lateral_length            (precomputed at build time)
  G_rad    = R_tot_2D · (1 − P_esc) / area         (uniform over absorber 2D area)
  G[absorber_y_range, :] += G_rad

Bit-equivalent to 1D Phase 3.1b in the lateral-uniform limit. Optical-profile-
weighted redistribution is explicitly deferred — see
docs/superpowers/specs/2026-04-30-2d-stage-b-c3-radiative-reabsorption-design.md.

The cached G_optical is never mutated. The helper returns a NEW (Ny, Nx)
array equal to G_optical augmented per absorber.
"""
from __future__ import annotations
import numpy as np


def _check_2d_shape(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.3) shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny}, {Nx})."
        )


def recompute_g_with_rad_2d(
    *,
    G_optical: np.ndarray,                              # (Ny, Nx)
    n: np.ndarray,                                      # (Ny, Nx)
    p: np.ndarray,                                      # (Ny, Nx)
    B_rad: np.ndarray,                                  # (Ny, Nx)
    x: np.ndarray,                                      # (Nx,)
    y: np.ndarray,                                      # (Ny,)
    absorber_y_ranges: tuple[tuple[int, int], ...],
    absorber_p_esc: tuple[float, ...],
    absorber_areas: tuple[float, ...],
) -> np.ndarray:
    """Return a NEW (Ny, Nx) array equal to G_optical augmented with the
    self-consistent radiative reabsorption source per absorber.

    Per-absorber operations:
      1. Slice (n, p, B_rad) along the absorber's y-range.
      2. Integrate B·n·p over y first (axis=0), then over x → scalar R_tot.
      3. Skip if R_tot ≤ 0, area ≤ 0, P_esc ≥ 1, or fewer than 2 nodes
         on either axis (matches 1D mol.py:874-895 safety guards).
      4. G_rad = R_tot · (1 − P_esc) / area   (uniform over absorber area).
      5. G_with_rad[y_lo:y_hi, :] += G_rad.

    The lengths of ``absorber_y_ranges``, ``absorber_p_esc``, and
    ``absorber_areas`` must match (one entry per absorber).
    """
    if y.ndim != 1:
        raise ValueError(
            f"Stage B(c.3): y must be 1-D, got shape {y.shape}"
        )
    if x.ndim != 1:
        raise ValueError(
            f"Stage B(c.3): x must be 1-D, got shape {x.shape}"
        )
    Ny = y.shape[0]
    Nx = x.shape[0]
    _check_2d_shape("G_optical", G_optical, Ny, Nx)
    _check_2d_shape("n", n, Ny, Nx)
    _check_2d_shape("p", p, Ny, Nx)
    _check_2d_shape("B_rad", B_rad, Ny, Nx)
    if not (len(absorber_y_ranges) == len(absorber_p_esc) == len(absorber_areas)):
        raise ValueError(
            f"Stage B(c.3): per-absorber tuple length mismatch — "
            f"y_ranges={len(absorber_y_ranges)}, "
            f"p_esc={len(absorber_p_esc)}, "
            f"areas={len(absorber_areas)}"
        )

    G_with_rad = G_optical.copy()
    for (y_lo, y_hi), p_esc, area in zip(
        absorber_y_ranges, absorber_p_esc, absorber_areas
    ):
        if area <= 0.0 or p_esc >= 1.0:
            continue
        if y_hi - y_lo < 2 or Nx < 2:
            continue
        emission = B_rad[y_lo:y_hi, :] * n[y_lo:y_hi, :] * p[y_lo:y_hi, :]   # (n_y_abs, Nx)
        # Integrate over y first (axis=0), giving (Nx,), then over x → scalar.
        emission_x = np.trapezoid(emission, y[y_lo:y_hi], axis=0)            # (Nx,)
        R_tot = float(np.trapezoid(emission_x, x))                            # scalar
        if R_tot <= 0.0:
            continue
        G_rad = R_tot * (1.0 - p_esc) / area
        G_with_rad[y_lo:y_hi, :] += G_rad
    return G_with_rad


__all__ = ["recompute_g_with_rad_2d"]

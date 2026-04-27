from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from perovskite_sim.twod.grid_2d import Grid2D


@dataclass(frozen=True)
class GrainBoundary:
    """A vertical grain boundary, modelled as a band of nodes in the absorber
    layer with reduced SRH lifetimes (τ_n, τ_p) and width δ."""
    x_position: float       # m, lateral coordinate of GB centre
    width: float            # m, GB band width (typical 5e-9)
    tau_n: float            # s, electron SRH lifetime inside GB
    tau_p: float            # s, hole SRH lifetime inside GB
    layer_role: str = "absorber"


@dataclass(frozen=True)
class Microstructure:
    """Container for spatially-varying defect features in a 2D simulation."""
    grain_boundaries: tuple[GrainBoundary, ...] = ()


def build_tau_field(
    grid: Grid2D,
    ustruct: Microstructure,
    tau_n_bulk_per_y: np.ndarray,
    tau_p_bulk_per_y: np.ndarray,
    layer_role_per_y: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (τ_n, τ_p) on the (Ny, Nx) grid.

    Stage A: ustruct is empty → returns the uniform extrusion of the 1D
    bulk τ along x. Stage B: GBs in the absorber layer override τ inside
    bands of width `gb.width` centred at `gb.x_position`.
    """
    Nx, Ny = grid.Nx, grid.Ny
    tau_n = np.broadcast_to(tau_n_bulk_per_y[:, None], (Ny, Nx)).copy()
    tau_p = np.broadcast_to(tau_p_bulk_per_y[:, None], (Ny, Nx)).copy()

    for gb in ustruct.grain_boundaries:
        mask_x = np.abs(grid.x - gb.x_position) < gb.width / 2.0      # (Nx,)
        mask_y = np.array([role == gb.layer_role for role in layer_role_per_y])
        mask_2d = np.outer(mask_y, mask_x)                            # (Ny, Nx)
        tau_n[mask_2d] = gb.tau_n
        tau_p[mask_2d] = gb.tau_p

    return tau_n, tau_p

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid, tanh_grid


@dataclass(frozen=True)
class Grid2D:
    """Tensor-product rectilinear mesh on (x, y).

    x is the lateral coordinate (length Nx, points 0..Nx-1).
    y is the vertical (stack) coordinate (length Ny, points 0..Ny-1).
    Total node count is Nx * Ny; the linear node index is
    idx(i, j) = j * Nx + i  (y-major / row-major over (j, i)).
    """
    x: np.ndarray
    y: np.ndarray

    @property
    def Nx(self) -> int:
        return int(self.x.size)

    @property
    def Ny(self) -> int:
        return int(self.y.size)

    @property
    def n_nodes(self) -> int:
        return self.Nx * self.Ny


def build_grid_2d(
    layers: list[Layer],
    lateral_length: float,
    Nx: int,
    *,
    alpha_y: float = 3.0,
    alpha_x: float = 2.0,
    lateral_uniform: bool = False,
) -> Grid2D:
    """Build a tensor-product (x, y) grid.

    y is the existing 1D multilayer tanh grid (clustered at layer interfaces
    and contacts via the same `multilayer_grid` used by the 1D solver).
    x is either uniform (Nx+1 evenly spaced points on [0, lateral_length])
    when `lateral_uniform=True`, or tanh-clustered toward x=0 and x=L_x
    otherwise. Stage A defaults to `lateral_uniform=True` because the
    validation problem has no internal x-features to cluster around.
    """
    y = multilayer_grid(layers, alpha=alpha_y)
    if lateral_uniform:
        x = np.linspace(0.0, lateral_length, Nx + 1)
    else:
        x = tanh_grid(Nx, lateral_length, alpha=alpha_x)
    return Grid2D(x=np.asarray(x, dtype=float),
                  y=np.asarray(y, dtype=float))

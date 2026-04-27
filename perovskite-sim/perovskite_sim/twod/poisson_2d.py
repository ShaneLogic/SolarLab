from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu, SuperLU

from perovskite_sim.constants import EPS_0
from perovskite_sim.twod.grid_2d import Grid2D


@dataclass(frozen=True)
class Poisson2DFactor:
    """Pre-factored 2D Poisson operator on a tensor-product mesh.

    The operator depends only on (Grid2D, eps_r(x,y), lateral_bc), all of
    which are constant across a transient. Building it once and reusing
    `splu` for every RHS evaluation collapses the hot path.

    Lateral BC is either "periodic" (couples i=Nx-1 to i=0) or "neumann"
    (zero-flux at i=0 and i=Nx-1). Vertical BC is always Dirichlet at
    j=0 (bottom contact) and j=Ny-1 (top contact); those rows are
    eliminated from the unknown set and absorbed into the RHS at solve time.
    """
    grid: Grid2D
    lateral_bc: str
    lu: SuperLU
    C_x: np.ndarray         # (Ny, Nx-1) for neumann or (Ny, Nx) for periodic — face conductances ε₀ε_face/h
    C_y: np.ndarray         # (Ny-1, Nx)
    cell_area: np.ndarray   # (Ny-2, Nx)
    C_y_top: np.ndarray     # (Nx,)
    C_y_bot: np.ndarray     # (Nx,)


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


def build_poisson_2d_factor(
    grid: Grid2D,
    eps_r: np.ndarray,
    lateral_bc: str = "periodic",
) -> Poisson2DFactor:
    """Assemble the 5-point sparse Poisson operator and factor it once."""
    assert eps_r.shape == (grid.Ny, grid.Nx), "eps_r must be shape (Ny, Nx)"
    assert lateral_bc in ("periodic", "neumann")

    Nx, Ny = grid.Nx, grid.Ny
    x, y = grid.x, grid.y
    dx = np.diff(x)                            # (Nx-1,)
    dy = np.diff(y)                            # (Ny-1,)
    # Dual-grid widths (one per node — index i in 0..Nx-1)
    hx_cell = np.empty(Nx)
    hx_cell[0] = dx[0] / 2.0
    hx_cell[-1] = dx[-1] / 2.0
    hx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    # For periodic, the boundary cells share the wrap face; we use full dx[0] / 2 + dx[-1] / 2
    if lateral_bc == "periodic":
        hx_cell[0] = 0.5 * (dx[0] + dx[-1])
        hx_cell[-1] = 0.5 * (dx[0] + dx[-1])
    hy_cell_int = 0.5 * (dy[:-1] + dy[1:])     # (Ny-2,)

    # x-face permittivities
    eps_face_x_int = _harmonic_mean(eps_r[:, :-1], eps_r[:, 1:])   # (Ny, Nx-1)
    if lateral_bc == "periodic":
        eps_face_wrap = _harmonic_mean(eps_r[:, -1], eps_r[:, 0])  # (Ny,)
        # face spacing for the wrap edge — half-cell on each side
        dx_wrap = 0.5 * (dx[0] + dx[-1])
        C_x_int = EPS_0 * eps_face_x_int / dx[None, :]            # (Ny, Nx-1)
        C_x_wrap = EPS_0 * eps_face_wrap / dx_wrap                 # (Ny,)
        C_x = np.concatenate([C_x_int, C_x_wrap[:, None]], axis=1) # (Ny, Nx)
    else:
        C_x = EPS_0 * eps_face_x_int / dx[None, :]                 # (Ny, Nx-1)

    # y-face permittivities
    eps_face_y = _harmonic_mean(eps_r[:-1, :], eps_r[1:, :])
    C_y = EPS_0 * eps_face_y / dy[:, None]                          # (Ny-1, Nx)

    # Store Gy_bot/Gy_top scaled by hx_cell so solve_poisson_2d can absorb
    # Dirichlet BCs consistently with the matrix (which uses Gy = C_y * dx_c).
    C_y_bot = C_y[0, :] * hx_cell          # (Nx,)  Gy at bottom face
    C_y_top = C_y[-1, :] * hx_cell         # (Nx,)  Gy at top face

    cell_area = hy_cell_int[:, None] * hx_cell[None, :]             # (Ny-2, Nx)

    # Build sparse matrix on interior unknowns: j = 1..Ny-2, i = 0..Nx-1
    #
    # Full 2D FV flux-balance per dual cell (dy_cell × dx_cell):
    #   x-fluxes are multiplied by dy_cell (the perpendicular width)
    #   y-fluxes are multiplied by dx_cell (the perpendicular width)
    #   RHS = -rho * (dx_cell * dy_cell) = -rho * cell_area
    #
    # Effective conductances entering the matrix:
    #   Gx = C_x * dy_cell   (x-face ε₀ε_r/dx_face * dy_cell)
    #   Gy = C_y * dx_cell   (y-face ε₀ε_r/dy_face * dx_cell)
    n_int_rows = Ny - 2
    n_unknowns = n_int_rows * Nx
    A = lil_matrix((n_unknowns, n_unknowns), dtype=float)

    def idx(j_int: int, i: int) -> int:
        return j_int * Nx + i

    for j_int in range(n_int_rows):
        j = j_int + 1
        dy_c = hy_cell_int[j_int]   # dual-cell height for interior row j_int
        for i in range(Nx):
            dx_c = hx_cell[i]       # dual-cell width for column i

            # Effective conductances (face conductance × perpendicular cell width)
            if lateral_bc == "periodic":
                ileft  = (i - 1) % Nx
                iright = (i + 1) % Nx
                Gx_left  = C_x[j, (i - 1) % Nx] * dy_c
                Gx_right = C_x[j, i] * dy_c
            else:
                ileft  = i - 1
                iright = i + 1
                Gx_left  = (C_x[j, i - 1] if i > 0 else 0.0) * dy_c
                Gx_right = (C_x[j, i] if i < Nx - 1 else 0.0) * dy_c

            Gy_below = C_y[j - 1, i] * dx_c
            Gy_above = C_y[j, i] * dx_c

            diag = -(Gx_left + Gx_right + Gy_below + Gy_above)
            A[idx(j_int, i), idx(j_int, i)] = diag

            if lateral_bc == "periodic":
                A[idx(j_int, i), idx(j_int, ileft)]  = Gx_left
                A[idx(j_int, i), idx(j_int, iright)] = Gx_right
            else:
                if 0 <= ileft < Nx:
                    A[idx(j_int, i), idx(j_int, ileft)]  = Gx_left
                if 0 <= iright < Nx:
                    A[idx(j_int, i), idx(j_int, iright)] = Gx_right

            if j_int - 1 >= 0:
                A[idx(j_int, i), idx(j_int - 1, i)] = Gy_below
            if j_int + 1 < n_int_rows:
                A[idx(j_int, i), idx(j_int + 1, i)] = Gy_above

    A = A.tocsc()
    lu = splu(A)
    return Poisson2DFactor(
        grid=grid, lateral_bc=lateral_bc, lu=lu,
        C_x=C_x, C_y=C_y, cell_area=cell_area,
        C_y_top=C_y_top, C_y_bot=C_y_bot,
    )


def solve_poisson_2d(
    fac: Poisson2DFactor,
    rho: np.ndarray,
    phi_bottom: float | np.ndarray,
    phi_top: float | np.ndarray,
) -> np.ndarray:
    """Solve A phi = -rho * cell_area, with Dirichlet rows absorbed into RHS.

    `phi_bottom` and `phi_top` may each be either a scalar (constant along x)
    or a 1D array of length Nx. The returned phi has shape (Ny, Nx) with
    phi[0,:]=phi_bottom and phi[-1,:]=phi_top filled in.
    """
    assert rho.shape == (fac.grid.Ny, fac.grid.Nx)
    Nx, Ny = fac.grid.Nx, fac.grid.Ny
    n_int_rows = Ny - 2

    phi_bot_arr = np.broadcast_to(np.asarray(phi_bottom, dtype=float), (Nx,)).copy()
    phi_top_arr = np.broadcast_to(np.asarray(phi_top, dtype=float), (Nx,)).copy()

    rhs = -rho[1:-1, :] * fac.cell_area      # (Ny-2, Nx)
    rhs[0, :]  -= fac.C_y_bot * phi_bot_arr
    rhs[-1, :] -= fac.C_y_top * phi_top_arr

    x_sol = fac.lu.solve(rhs.flatten())
    phi = np.empty((Ny, Nx), dtype=float)
    phi[0, :]  = phi_bot_arr
    phi[-1, :] = phi_top_arr
    phi[1:-1, :] = x_sol.reshape((n_int_rows, Nx))
    return phi

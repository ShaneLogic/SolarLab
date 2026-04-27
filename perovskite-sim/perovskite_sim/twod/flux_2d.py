from __future__ import annotations
import numpy as np

from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.constants import Q


def sg_fluxes_2d_n(
    phi_n: np.ndarray,        # (Ny, Nx)
    n: np.ndarray,            # (Ny, Nx)
    x: np.ndarray,            # (Nx,)
    y: np.ndarray,            # (Ny,)
    D_n: np.ndarray,          # (Ny, Nx) — node-resolved; face avg taken inside
    V_T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG electron current density on horizontal edges (J_x, shape
    (Ny, Nx-1)) and vertical edges (J_y, shape (Ny-1, Nx)).

    Conventions (matching 1D fe_operators.sg_fluxes_n):
      J_x[j, i]  : flux from (i,j) to (i+1,j); positive = electron current
                   flowing in +x direction.
      J_y[j, i]  : flux from (i,j) to (i,j+1); positive = +y direction.
    """
    dx = np.diff(x)                              # (Nx-1,)
    dy = np.diff(y)                              # (Ny-1,)

    # Harmonic-mean face averaging (matches 1D physics/poisson harmonic eps_r
    # face average and 1D MaterialArrays harmonic D_n_face). Required at
    # heterointerfaces where D varies by orders of magnitude across a face.
    _eps = 1e-300
    D_face_x = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)  # (Ny, Nx-1)
    xi_x = (phi_n[:, 1:] - phi_n[:, :-1]) / V_T  # (Ny, Nx-1)
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * n[:, 1:] - bernoulli(-xi_x) * n[:, :-1]
    )

    D_face_y = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)  # (Ny-1, Nx)
    xi_y = (phi_n[1:, :] - phi_n[:-1, :]) / V_T  # (Ny-1, Nx)
    Jy = (Q * D_face_y / dy[:, None]) * (
        bernoulli(xi_y) * n[1:, :] - bernoulli(-xi_y) * n[:-1, :]
    )
    return Jx, Jy


def sg_fluxes_2d_p(
    phi_p: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    D_p: np.ndarray,
    V_T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG hole current density on horizontal and vertical edges.
    Sign convention matches the 1D `sg_fluxes_p`."""
    dx = np.diff(x)
    dy = np.diff(y)

    _eps = 1e-300
    D_face_x = 2.0 * D_p[:, :-1] * D_p[:, 1:] / (D_p[:, :-1] + D_p[:, 1:] + _eps)
    xi_x = (phi_p[:, 1:] - phi_p[:, :-1]) / V_T
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * p[:, :-1] - bernoulli(-xi_x) * p[:, 1:]
    )

    D_face_y = 2.0 * D_p[:-1, :] * D_p[1:, :] / (D_p[:-1, :] + D_p[1:, :] + _eps)
    xi_y = (phi_p[1:, :] - phi_p[:-1, :]) / V_T
    Jy = (Q * D_face_y / dy[:, None]) * (
        bernoulli(xi_y) * p[:-1, :] - bernoulli(-xi_y) * p[1:, :]
    )
    return Jx, Jy

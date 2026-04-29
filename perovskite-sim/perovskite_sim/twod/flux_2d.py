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
    *,
    D_n_x_face: np.ndarray | None = None,   # (Ny, Nx-1)  Stage B(c.2) override
    D_n_y_face: np.ndarray | None = None,   # (Ny-1, Nx)  Stage B(c.2) override
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG electron current density on horizontal edges (J_x, shape
    (Ny, Nx-1)) and vertical edges (J_y, shape (Ny-1, Nx)).

    Conventions (matching 1D fe_operators.sg_fluxes_n):
      J_x[j, i]  : flux from (i,j) to (i+1,j); positive = electron current
                   flowing in +x direction.
      J_y[j, i]  : flux from (i,j) to (i,j+1); positive = +y direction.

    Per-face D override (Stage B(c.2)): when ``D_n_x_face`` / ``D_n_y_face``
    are provided, they replace the internal harmonic-mean computation. This
    is how field-dependent μ(E) injects per-face effective diffusion. When
    both are ``None`` (default) the harmonic-mean path runs unchanged.
    """
    dx = np.diff(x)                              # (Nx-1,)
    dy = np.diff(y)                              # (Ny-1,)

    # Harmonic-mean face averaging (matches 1D physics/poisson harmonic eps_r
    # face average and 1D MaterialArrays harmonic D_n_face). Required at
    # heterointerfaces where D varies by orders of magnitude across a face.
    _eps = 1e-300
    if D_n_x_face is None:
        D_face_x = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)  # (Ny, Nx-1)
    else:
        D_face_x = D_n_x_face
    xi_x = (phi_n[:, 1:] - phi_n[:, :-1]) / V_T  # (Ny, Nx-1)
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * n[:, 1:] - bernoulli(-xi_x) * n[:, :-1]
    )

    if D_n_y_face is None:
        D_face_y = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)  # (Ny-1, Nx)
    else:
        D_face_y = D_n_y_face
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
    *,
    D_p_x_face: np.ndarray | None = None,   # (Ny, Nx-1)  Stage B(c.2) override
    D_p_y_face: np.ndarray | None = None,   # (Ny-1, Nx)  Stage B(c.2) override
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG hole current density on horizontal and vertical edges.
    Sign convention matches the 1D `sg_fluxes_p`.

    Per-face D override (Stage B(c.2)): see ``sg_fluxes_2d_n`` for semantics.
    """
    dx = np.diff(x)
    dy = np.diff(y)

    _eps = 1e-300
    if D_p_x_face is None:
        D_face_x = 2.0 * D_p[:, :-1] * D_p[:, 1:] / (D_p[:, :-1] + D_p[:, 1:] + _eps)
    else:
        D_face_x = D_p_x_face
    xi_x = (phi_p[:, 1:] - phi_p[:, :-1]) / V_T
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * p[:, :-1] - bernoulli(-xi_x) * p[:, 1:]
    )

    if D_p_y_face is None:
        D_face_y = 2.0 * D_p[:-1, :] * D_p[1:, :] / (D_p[:-1, :] + D_p[1:, :] + _eps)
    else:
        D_face_y = D_p_y_face
    xi_y = (phi_p[1:, :] - phi_p[:-1, :]) / V_T
    Jy = (Q * D_face_y / dy[:, None]) * (
        bernoulli(xi_y) * p[:-1, :] - bernoulli(-xi_y) * p[1:, :]
    )
    return Jx, Jy

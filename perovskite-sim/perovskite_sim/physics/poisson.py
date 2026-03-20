from __future__ import annotations
import numpy as np
try:
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
except ImportError:
    from scipy_shim import diags, spsolve

from perovskite_sim.constants import EPS_0


def solve_poisson(
    x: np.ndarray,
    eps_r: np.ndarray,
    rho: np.ndarray,      # charge density [C/m³]
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """
    Solve  d/dx( ε₀ εᵣ dφ/dx ) = -ρ  on the 1-D grid x.

    Finite-Volume discretisation
    ────────────────────────────
    For internal node i (1 ≤ i ≤ N-2), integrate over the dual cell
    [x_{i-½}, x_{i+½}]:

        C_R·(φ_{i+1} - φ_i) - C_L·(φ_i - φ_{i-1}) = -ρ_i · h_cell_i

    where
        C_{i±½} = ε₀ · ε̃_{i±½} / h_{i±½}    [C V⁻¹ m⁻²]

    Face permittivities use the HARMONIC mean of adjacent nodal values,

        ε̃_{i+½} = 2 ε_r[i] ε_r[i+1] / (ε_r[i] + ε_r[i+1])

    which is the exact series-capacitor result for a sharp dielectric
    interface anywhere inside the face cell.  Nodal-value interpolation
    (the previous implementation) incorrectly equates E-fields across
    interfaces and concentrates all the voltage drop in the last layer.

    Dirichlet BCs:  φ(x[0]) = phi_left,  φ(x[-1]) = phi_right.

    Returns φ on all N nodes.
    """
    N = len(x)
    h = np.diff(x)                                # h[j] = x[j+1]-x[j],   (N-1,)

    # Harmonic-mean face permittivities and face conductances
    eps_face = 2.0 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])  # (N-1,)
    C = EPS_0 * eps_face / h                      # (N-1,)  [C V⁻¹ m⁻²]

    # Dual-grid cell widths for internal nodes 1..N-2
    h_cell = 0.5 * (h[:-1] + h[1:])              # (N-2,)

    # Tridiagonal system for internal unknowns (j = i-1, j = 0..N-3)
    #   A[j, j-1] = C[j]            (sub,  j ≥ 1)
    #   A[j, j]   = -(C[j]+C[j+1]) (main, j = 0..N-3)
    #   A[j, j+1] = C[j+1]          (sup,  j ≤ N-4)
    dia = -(C[:-1] + C[1:])          # (N-2,)  main diagonal
    off = C[1:-1]                    # (N-3,)  off-diagonal (same for sub and sup)

    A = diags(
        [off, dia, off],
        offsets=[-1, 0, 1],
        shape=(N - 2, N - 2),
        format="csr",
    )

    # Right-hand side: -ρ · h_cell
    rhs = -rho[1:-1] * h_cell

    # Absorb Dirichlet boundary conditions into the RHS
    # (contribution of φ[0] to equation j=0, and φ[N-1] to equation j=N-3)
    rhs[0]  -= C[0]  * phi_left
    rhs[-1] -= C[-1] * phi_right

    phi_int = spsolve(A, rhs)

    phi = np.empty(N)
    phi[0]    = phi_left
    phi[-1]   = phi_right
    phi[1:-1] = phi_int
    return phi

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
try:
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    from scipy.linalg.lapack import dgttrf, dgttrs
except ImportError:
    from perovskite_sim._compat.scipy_shim import diags, spsolve
    dgttrf = None
    dgttrs = None

from perovskite_sim.constants import EPS_0


@dataclass(frozen=True)
class PoissonFactor:
    """Precomputed LAPACK LU factorisation of the Poisson tridiagonal.

    The discrete Poisson matrix depends only on the grid `x` and the per-node
    permittivity `eps_r`, both of which are constant across every RHS
    evaluation in a transient. Factorising once (via LAPACK `dgttrf`) and
    calling `dgttrs` for each RHS collapses the hot path: no sparse CSR
    construction, no general-sparse dispatch, pure O(N) BLAS.

    Fields
    ------
    C       : (N-1,) face conductances ε₀ε̃/h, reused to absorb the Dirichlet
              BCs into the RHS each call.
    h_cell  : (N-2,) dual-grid cell widths for the RHS scaling.
    dl, d, du, du2, ipiv : LAPACK `dgttrf` output — the in-place LU of the
              tridiagonal factored once at build time.
    N       : grid size.
    """
    C: np.ndarray
    h_cell: np.ndarray
    dl: np.ndarray
    d: np.ndarray
    du: np.ndarray
    du2: np.ndarray
    ipiv: np.ndarray
    N: int


def factor_poisson(x: np.ndarray, eps_r: np.ndarray) -> PoissonFactor:
    """Precompute the LAPACK LU of the Poisson operator for constant (x, eps_r).

    Matches `solve_poisson`'s finite-volume discretisation exactly:
    harmonic-mean face permittivities, dual-grid cell widths, and the same
    tridiagonal pattern A[j,j-1]=C[j], A[j,j]=-(C[j]+C[j+1]), A[j,j+1]=C[j+1]
    on the interior unknowns j = 0 .. N-3.
    """
    if dgttrf is None:
        raise RuntimeError("scipy.linalg.lapack.dgttrf is unavailable")
    N = len(x)
    h = np.diff(x)
    eps_face = 2.0 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])
    C = EPS_0 * eps_face / h                      # (N-1,)
    h_cell = 0.5 * (h[:-1] + h[1:])               # (N-2,)

    dia = -(C[:-1] + C[1:])                       # (N-2,)  main diagonal
    off = C[1:-1].copy()                          # (N-3,)  sub- and super-diagonals

    # LAPACK dgttrf expects three vectors (sub, main, super) and returns the
    # factored (dl, d, du, du2, ipiv) for use by dgttrs.
    dl_in = off.astype(np.float64)
    d_in = dia.astype(np.float64)
    du_in = off.astype(np.float64)
    dl, d, du, du2, ipiv, info = dgttrf(dl_in, d_in, du_in)
    if info != 0:
        raise RuntimeError(f"dgttrf failed with info={info}")
    return PoissonFactor(
        C=C, h_cell=h_cell,
        dl=dl, d=d, du=du, du2=du2, ipiv=ipiv, N=N,
    )


def solve_poisson_prefactored(
    fac: PoissonFactor,
    rho: np.ndarray,
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """Solve the Poisson system using the cached LAPACK LU factorisation.

    This is the hot-path entry point used by `assemble_rhs`. All work is a
    single `dgttrs` call plus a tiny RHS assembly — no sparse construction,
    no Python loops over grid nodes.
    """
    N = fac.N
    C = fac.C

    # Build RHS for the interior system, absorbing Dirichlet BCs
    rhs = -rho[1:-1] * fac.h_cell                 # (N-2,)
    rhs[0]  -= C[0]  * phi_left
    rhs[-1] -= C[-1] * phi_right

    # dgttrs takes a 2-D RHS; reshape to column, solve, flatten.
    b = rhs.reshape(-1, 1)
    phi_int, info = dgttrs(fac.dl, fac.d, fac.du, fac.du2, fac.ipiv, b)
    if info != 0:
        raise RuntimeError(f"dgttrs failed with info={info}")

    phi = np.empty(N)
    phi[0]    = phi_left
    phi[-1]   = phi_right
    phi[1:-1] = phi_int[:, 0]
    return phi


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

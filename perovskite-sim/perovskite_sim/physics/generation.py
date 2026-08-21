"""Beer-Lambert optical generation on the solver's node-centred dual mesh.

Photon conservation is STRUCTURAL here, not asymptotic
-----------------------------------------------------
``carrier_continuity_rhs`` adds ``G[i]`` to ``dn[i]``/``dp[i]`` and the
flux-divergence term it sits next to is ``(J[i+1]-J[i]) / (Q*dx_cell[i])``.
Multiplying the whole balance by ``Q*dx_cell[i]`` and summing over the grid
telescopes the flux term to the two terminal currents, so the areal
generation the device actually receives is exactly::

    q * sum_i G[i] * dx_cell[i]

i.e. the solver applies a **node-centred rectangle rule** with the dual-cell
weights ``dx_cell``.  Point-sampling ``G = Phi*alpha*exp(-tau(x))`` at the
nodes and handing that to a rectangle rule is not photon-conserving: on a
mesh that is coarse relative to the absorption length ``1/alpha`` the rule
over-counts the sharply-peaked front-of-absorber generation and the device
creates more electron-hole pairs than there are incident photons.  Measured
on ``configs/ionmonger_benchmark.yaml`` (1/alpha = 76.9 nm, q*Phi = 224.305
A/m^2) the point-sampled form gave 237.497 A/m^2 at N_grid = 30 (+5.9 % over
the incident flux), 226.797 at N_grid = 60 and 224.447 at N_grid = 100 — the
last two being the meshes the shipped regression suite and ``run_jv_sweep``
default to.

So the node value is instead defined as the **exact absorbed fraction of its
own dual cell, divided by the volume weight the RHS assigns to that node**::

    G[i] = Phi * (exp(-tau(b_i)) - exp(-tau(b_{i+1}))) / dx_cell[i]

with ``b`` the dual-cell faces.  The numerators telescope, so for ANY mesh::

    sum_i G[i]*dx_cell[i] = Phi * (1 - exp(-tau_total))     (to machine eps)

which is bounded by ``Phi`` by construction.  Refinement no longer buys
photon conservation — it only sharpens where inside the absorber the photons
land — so J_sc becomes mesh-independent instead of converging to its own
budget from above.

Note on the boundary cells (deliberate, and the reason ``dx_cell`` rather
than the geometric half-width appears in the denominator): the solver's
``dx_cell`` is FULL width at nodes 0 and N-1 (``dx_cell[0] = dx[0]``, not
``dx[0]/2``), so ``sum(dx_cell)`` overshoots the device thickness by
``(dx[0]+dx[-1])/2``.  Dividing the exact per-cell absorbed photon count by
that same over-wide weight is what keeps ``sum(G*dx_cell)`` exact; the
consequence is that at an absorbing OUTER node the returned ``G`` is the
cell-integrated rate spread over the solver's wider cell rather than the
local volumetric rate.  Every shipped preset has alpha = 0 in both outer
layers, so their boundary nodes carry G = 0 either way.  Consumers that want
a pointwise volumetric rate, or that integrate with a rule other than
``sum(.*dx_cell)`` (e.g. ``np.trapezoid``), must account for this.
"""
import numpy as np


def dual_cell_widths(x: np.ndarray) -> np.ndarray:
    """Node-centred dual-cell widths, matching the solver exactly.

    Mirrors ``solver/mol.py:build_material_arrays`` and
    ``physics/continuity.py:carrier_continuity_rhs``: interior nodes own
    half of each neighbouring interval, the two boundary nodes own the FULL
    adjacent interval.  Kept identical on purpose — the generation
    quadrature is only photon-exact if it divides by the very weights the
    RHS multiplies back in.
    """
    dx = np.diff(x)
    w = np.empty(x.shape[0], dtype=float)
    w[0] = dx[0]
    w[-1] = dx[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def dual_cell_integral(x: np.ndarray, values: np.ndarray) -> float:
    """Integrate node values with the weights used by conservation laws.

    This is the discrete invariant paired with the node-centred divergence
    operators.  In particular, zero-flux ion continuity conserves this sum;
    a geometric trapezoid rule does not use the same endpoint weights.
    """

    coordinates = np.asarray(x, dtype=float)
    samples = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or samples.ndim != 1:
        raise ValueError("x and values must be one-dimensional")
    if coordinates.shape != samples.shape:
        raise ValueError("x and values must have identical shapes")
    if coordinates.size < 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError("x must contain at least two finite points")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must be finite")
    return float(np.sum(samples * dual_cell_widths(coordinates)))


def dual_cell_faces(x: np.ndarray) -> np.ndarray:
    """The ``N+1`` faces bounding the node-centred dual cells.

    Interior faces are the interval midpoints; the two outer faces are the
    device faces themselves, because no absorption happens outside the
    grid.  Paired with :func:`dual_cell_widths` this is the complete
    definition of the solver's dual mesh::

        cell i spans [faces[i], faces[i+1]]   and is weighted by dx_cell[i]

    Note the deliberate mismatch at the two boundary cells: their geometric
    extent is ``dx[0]/2`` / ``dx[-1]/2`` but the RHS weight is the FULL
    ``dx[0]`` / ``dx[-1]``.  Any photon-conserving quadrature must divide
    the exactly-absorbed photons of ``[faces[i], faces[i+1]]`` by
    ``dx_cell[i]`` — see the module docstring.

    Used by the TMM path (``solver/mol.py:_compute_tmm_generation``), which
    needs the face POSITIONS to evaluate the transfer-matrix cumulative
    absorptance.  The Beer-Lambert path below reaches the same faces
    implicitly: its optical depth is piecewise linear between nodes, so the
    face value is just the mean of the two node values.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError(
            f"dual cells need at least 2 nodes, got {x.shape[0]}"
        )
    faces = np.empty(x.shape[0] + 1, dtype=float)
    faces[0] = x[0]
    faces[-1] = x[-1]
    faces[1:-1] = 0.5 * (x[:-1] + x[1:])
    return faces


def _cumulative_optical_depth(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """tau(x_i) = integral of alpha from x[0] to x[i], at the nodes.

    Array alpha uses the same mid-face trapezoidal accumulation as before
    this module was made photon-conserving, so the optical depth itself is
    unchanged and the two versions differ ONLY in the quadrature.
    """
    if alpha.ndim == 0:
        # Legacy scalar semantics: tau = alpha * x (x[0] = 0 on every grid
        # the solver builds, so this is the integral from the front face).
        return float(alpha) * x
    dx = np.diff(x)
    alpha_mid = 0.5 * (alpha[:-1] + alpha[1:])   # (N-1,) face-average
    cum = np.zeros_like(x)
    cum[1:] = np.cumsum(alpha_mid * dx)
    return cum


def beer_lambert_generation(
    x: np.ndarray, alpha, Phi: float
) -> np.ndarray:
    """Photon-conserving Beer-Lambert generation [m^-3 s^-1] at the nodes.

    ``G[i]`` is the exact number of photons absorbed inside node ``i``'s dual
    cell per unit time and area, divided by ``dx_cell[i]`` — so that the
    rectangle rule the RHS performs, ``sum(G*dx_cell)``, reproduces the
    closed-form absorbed-photon budget ``Phi*(1 - exp(-tau_total))`` on every
    mesh, coarse or fine.  See the module docstring for why this is the
    correct discretisation and what it costs at absorbing outer nodes.

    Supports scalar alpha (uniform device) or array alpha (layered device).
    For array alpha the cumulative optical depth is integrated from ``x[0]``
    with the mid-face trapezoidal rule, so generation is exactly zero in
    non-absorbing layers (both faces of the cell carry the same optical
    depth, so the difference of exponentials is an exact floating-point
    zero) and peaks at the front of the absorbing region.
    """
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if alpha.ndim > 1 or (alpha.ndim == 1 and alpha.shape != x.shape):
        raise ValueError(
            f"alpha must be scalar or shape {x.shape}, got {alpha.shape}"
        )
    if x.shape[0] < 2:
        # No dual cell exists on a single node — nothing to integrate over.
        # Fall back to the point-sampled rate rather than raising, so callers
        # probing degenerate grids behave as they did before.
        return float(Phi) * alpha * np.exp(-_cumulative_optical_depth(x, alpha))

    cum = _cumulative_optical_depth(x, alpha)

    # Optical depth at the dual-cell faces. tau is piecewise linear between
    # nodes under the trapezoidal accumulation, so the face value is the
    # arithmetic mean of the two node values; the outer faces are the device
    # faces themselves (no absorption happens outside the grid).
    tau_face = np.empty(x.shape[0] + 1, dtype=float)
    tau_face[0] = cum[0]
    tau_face[-1] = cum[-1]
    tau_face[1:-1] = 0.5 * (cum[:-1] + cum[1:])

    # Photons absorbed per unit area per unit time inside each dual cell.
    # This difference telescopes across the grid, which is the whole point.
    absorbed = float(Phi) * (np.exp(-tau_face[:-1]) - np.exp(-tau_face[1:]))
    return absorbed / dual_cell_widths(x)

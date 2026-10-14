"""Independent electrostatics oracle for P04-01 (Poisson as an algebraic residual).

Continuous problem (1D, unit cross-section, SI)::

    -d/dx( eps0 eps_r(x) dphi/dx ) = rho(x),   phi(0) = phi_L,  phi(L) = phi_R

Gauss form: D = -eps dphi/dx, dD/dx = rho, and D jumps by a sheet charge
sigma at an interface plane.  For piecewise-constant eps_k, rho_k on layers
[b_k, b_{k+1}] the exact solution is piecewise quadratic::

    D(x)   = D_k + rho_k (x - b_k)
    phi(x) = phi(b_k) - [D_k (x - b_k) + rho_k (x - b_k)^2 / 2] / eps_k
    D_k    = D0 + sum_{j<k} rho_j d_j + (sheet charges on planes left of k)

and phi(L) = phi_R fixes D0 (``layered_constant_charge_solution``).  With
rho = 0 this is the series capacitor, C = 1 / sum_k d_k/eps_k (``series``).

Finite-volume reference.  For nodes x_i, faces f (between nodes) and cells
[f_i, f_{i+1}] the oracle uses::

    C_f (phi_{i+1} - phi_i) - C_{f-1} (phi_i - phi_{i-1}) = -Q_i     (FV)

C_f = 1 / int_{x_i}^{x_{i+1}} dx/eps (exact series face capacitance) and Q_i
the cell charge.  Two exactness facts are derived in ``Report.md``:

* interface NODES + exact C_f + exact cell-integrated charge reproduce the
  piecewise-quadratic solution exactly at the nodes on any non-uniform mesh
  (the secant slope of a quadratic is its midpoint derivative);
* for rho = 0 any geometry with exact C_f reproduces the series capacitor
  exactly (D is constant), including the R1 two-sided geometry.

The historical 1D rule (``solver/mol.py`` + ``physics/poisson.py``): nodal
eps_r and rho owned by the RIGHT layer at an interface node, face
eps = harmonic mean of the two nodal values, Q_i = rho_i (h_{i-1}+h_i)/2.
This equals (FV) with exact data for the SHIFTED structure whose material
planes sit at x_I - h_{I-1}/2.  It is therefore exact for that shifted
structure when rho = 0 and first order in h at a dielectric contrast that
carries displacement.  ``legacy_rule_*`` re-derives that rule independently
so the M-class expectation survives the retirement of the legacy engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: perovskite_sim/constants.py (CODATA 2006 value).  An M-class migration
#: must keep it; CODATA 2018 (8.8541878128e-12) is a P-class change.
EPS_0_LEGACY = 8.854187817e-12


@dataclass(frozen=True)
class LayeredStack:
    thicknesses_m: tuple[float, ...]
    eps_r: tuple[float, ...]
    rho_C_m3: tuple[float, ...] = ()
    sheet_charge_C_m2: tuple[float, ...] = ()  # one per internal plane

    @property
    def bounds(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.thicknesses_m)))

    def rho(self) -> tuple[float, ...]:
        return self.rho_C_m3 or (0.0,) * len(self.thicknesses_m)

    def sheets(self) -> tuple[float, ...]:
        return self.sheet_charge_C_m2 or (0.0,) * (len(self.thicknesses_m) - 1)


def series_capacitance(stack: LayeredStack, eps0: float = EPS_0_LEGACY) -> float:
    """C = 1 / sum_k d_k/(eps0 eps_k)  [F/m^2]."""

    return 1.0 / math.fsum(
        d / (eps0 * e) for d, e in zip(stack.thicknesses_m, stack.eps_r)
    )


@dataclass(frozen=True)
class LayeredSolution:
    stack: LayeredStack
    eps0: float
    phi_left: float
    phi_right: float
    D0: float  # displacement at x = 0 [C/m^2]

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.stack.bounds
        rho, sig = self.stack.rho(), self.stack.sheets()
        D_k = self.D0
        for k in range(len(rho)):
            if k:
                D_k += rho[k - 1] * self.stack.thicknesses_m[k - 1] + sig[k - 1]
            m = (x >= b[k]) & (x <= b[k + 1]) if k == 0 else (x > b[k]) & (x <= b[k + 1])
            out[m] = D_k + rho[k] * (x[m] - b[k])
        return out

    def potential(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        b = self.stack.bounds
        rho, sig = self.stack.rho(), self.stack.sheets()
        phi_k, D_k = self.phi_left, self.D0
        for k in range(len(rho)):
            eps = self.eps0 * self.stack.eps_r[k]
            if k:
                d_prev = self.stack.thicknesses_m[k - 1]
                eps_prev = self.eps0 * self.stack.eps_r[k - 1]
                phi_k = phi_k - (D_k * d_prev + rho[k - 1] * d_prev**2 / 2) / eps_prev
                D_k += rho[k - 1] * d_prev + sig[k - 1]
            m = (x >= b[k]) & (x <= b[k + 1]) if k == 0 else (x > b[k]) & (x <= b[k + 1])
            u = x[m] - b[k]
            out[m] = phi_k - (D_k * u + rho[k] * u**2 / 2) / eps
        return out


def layered_constant_charge_solution(
    stack: LayeredStack,
    phi_left: float,
    phi_right: float,
    eps0: float = EPS_0_LEGACY,
) -> LayeredSolution:
    """Exact piecewise-quadratic solution; D0 from phi(L) = phi_right."""

    rho, sig = stack.rho(), stack.sheets()
    d = stack.thicknesses_m
    eps = [eps0 * e for e in stack.eps_r]
    S, s_terms, q_terms = 0.0, [], []
    for k in range(len(d)):
        if k:
            S += rho[k - 1] * d[k - 1] + sig[k - 1]
        s_terms.append(S * d[k] / eps[k])
        q_terms.append(rho[k] * d[k] ** 2 / (2 * eps[k]))
    inv_c = math.fsum(dk / ek for dk, ek in zip(d, eps))
    D0 = (phi_left - phi_right - math.fsum(s_terms) - math.fsum(q_terms)) / inv_c
    return LayeredSolution(stack, eps0, phi_left, phi_right, D0)


# --------------------------------------------------------------------------
# Finite-volume references
# --------------------------------------------------------------------------


def exact_face_capacitance(
    x: np.ndarray, stack: LayeredStack, eps0: float = EPS_0_LEGACY
) -> np.ndarray:
    """C_f = 1/int dx/(eps0 eps_r) over every interval (true planes)."""

    x = np.asarray(x, dtype=float)
    b = stack.bounds
    out = np.empty(x.size - 1)
    for i in range(x.size - 1):
        lo, hi = x[i], x[i + 1]
        parts = []
        for k in range(len(stack.thicknesses_m)):
            a, c = max(lo, b[k]), min(hi, b[k + 1])
            if c > a:
                parts.append((c - a) / (eps0 * stack.eps_r[k]))
        out[i] = 1.0 / math.fsum(parts)
    return out


def exact_cell_charge(
    faces: np.ndarray, stack: LayeredStack, x_nodes: np.ndarray | None = None
) -> np.ndarray:
    """Q_i = int over [faces_i, faces_{i+1}] of rho (+ sheets inside the cell).

    A sheet charge on a plane that coincides with a cell face is ambiguous for
    a single-trace FV scheme and is rejected (the two-sided Gauss jump needs
    its own interface unknowns).
    """

    faces = np.asarray(faces, dtype=float)
    b = stack.bounds
    rho, sig = stack.rho(), stack.sheets()
    out = np.empty(faces.size - 1)
    for i in range(faces.size - 1):
        lo, hi = faces[i], faces[i + 1]
        parts = []
        for k in range(len(rho)):
            a, c = max(lo, b[k]), min(hi, b[k + 1])
            if c > a:
                parts.append(rho[k] * (c - a))
        for j, s in enumerate(sig):
            plane = b[j + 1]
            if s != 0.0 and plane in (lo, hi):
                raise ValueError("sheet charge on a cell face needs a two-sided trace")
            if s != 0.0 and lo < plane < hi:
                parts.append(s)
        out[i] = math.fsum(parts)
    return out


def solve_fv_dirichlet(
    face_capacitance: np.ndarray,
    cell_charge: np.ndarray,
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """Solve (FV) for interior nodes with a dense LAPACK gesv (reference).

    ``cell_charge`` has one entry per node; the two Dirichlet entries are
    ignored.  The dense path is deliberately different from the legacy
    banded ``dgttrf``/``dgttrs`` factorisation.
    """

    C = np.asarray(face_capacitance, dtype=float)
    Q = np.asarray(cell_charge, dtype=float)
    n = C.size + 1
    m = n - 2
    A = np.zeros((m, m))
    rhs = -Q[1:-1].astype(float).copy()
    for j in range(m):
        A[j, j] = -(C[j] + C[j + 1])
        if j > 0:
            A[j, j - 1] = C[j]
        if j < m - 1:
            A[j, j + 1] = C[j + 1]
    rhs[0] -= C[0] * phi_left
    rhs[-1] -= C[-1] * phi_right
    phi = np.empty(n)
    phi[0], phi[-1] = phi_left, phi_right
    phi[1:-1] = np.linalg.solve(A, rhs)
    return phi


def fv_residual(
    phi: np.ndarray, face_capacitance: np.ndarray, cell_charge: np.ndarray
) -> np.ndarray:
    """Interior (FV) residual in C/m^2 with the oracle's sign convention."""

    phi = np.asarray(phi, dtype=float)
    C = np.asarray(face_capacitance, dtype=float)
    flux = C * np.diff(phi)
    return flux[1:] - flux[:-1] + np.asarray(cell_charge, dtype=float)[1:-1]


def fv_residual_scale(
    phi: np.ndarray, face_capacitance: np.ndarray, cell_charge: np.ndarray
) -> np.ndarray:
    """Sum of absolute terms of each residual row (rounding scale)."""

    C = np.asarray(face_capacitance, dtype=float)
    flux = np.abs(C * np.diff(np.asarray(phi, dtype=float)))
    return flux[1:] + flux[:-1] + np.abs(np.asarray(cell_charge, dtype=float)[1:-1])


def contact_displacement_from_fv(
    phi: np.ndarray,
    face_capacitance: np.ndarray,
    cell_charge_left: float,
    cell_charge_right: float,
) -> tuple[float, float]:
    """D at x=0 and x=L by Gauss over the two end (half) cells."""

    C = np.asarray(face_capacitance, dtype=float)
    d_face = -C * np.diff(np.asarray(phi, dtype=float))
    return float(d_face[0] - cell_charge_left), float(d_face[-1] + cell_charge_right)


# --------------------------------------------------------------------------
# Historical 1D rule (independent re-derivation; M-class expectation)
# --------------------------------------------------------------------------


def legacy_rule_face_capacitance(
    eps_node: np.ndarray, x: np.ndarray, eps0: float = EPS_0_LEGACY
) -> np.ndarray:
    eps_node = np.asarray(eps_node, dtype=float)
    eps_face = 2.0 * eps_node[:-1] * eps_node[1:] / (eps_node[:-1] + eps_node[1:])
    return eps0 * eps_face / np.diff(np.asarray(x, dtype=float))


def legacy_rule_cell_charge(rho_node: np.ndarray, x: np.ndarray) -> np.ndarray:
    h = np.diff(np.asarray(x, dtype=float))
    w = np.zeros(len(x))
    w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return np.asarray(rho_node, dtype=float) * w


def shifted_stack_for_legacy_rule(
    x: np.ndarray, stack: LayeredStack
) -> LayeredStack:
    """Structure whose exact FV data equals the legacy 1D rule.

    Each internal plane moves from the interface node x_I to the dual face
    x_I - h_{I-1}/2; the outer faces are unchanged.
    """

    x = np.asarray(x, dtype=float)
    b = stack.bounds
    planes = []
    for position in b[1:-1]:
        node = int(np.argmin(np.abs(x - position)))
        if abs(x[node] - position) > 1e-12 * b[-1]:
            raise ValueError("legacy 1D rule requires interface nodes")
        planes.append(0.5 * (x[node - 1] + x[node]))
    edges = [0.0, *planes, float(b[-1])]
    return LayeredStack(
        tuple(float(e1 - e0) for e0, e1 in zip(edges[:-1], edges[1:])),
        stack.eps_r,
        stack.rho_C_m3,
        stack.sheet_charge_C_m2,
    )


# --------------------------------------------------------------------------
# Manufactured solution (smooth rho, piecewise eps, optional sheet charge)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManufacturedPoisson:
    """D(x) = F0 cos(pi x / L) + F1 + sum_j sigma_j H(x - b_j).

    rho = dD/dx = -F0 (pi/L) sin(pi x/L) away from the planes; the sheet
    charges are the Gauss jumps.  phi = phi_L - int_0^x D/eps.
    """

    thicknesses_m: tuple[float, ...]
    eps_r: tuple[float, ...]
    F0: float
    F1: float
    phi_left: float
    sheet_charge_C_m2: tuple[float, ...] = ()
    eps0: float = EPS_0_LEGACY

    @property
    def bounds(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.thicknesses_m)))

    @property
    def length(self) -> float:
        return float(self.bounds[-1])

    def _sheets(self) -> tuple[float, ...]:
        return self.sheet_charge_C_m2 or (0.0,) * (len(self.thicknesses_m) - 1)

    def smooth_flux(self, x: np.ndarray) -> np.ndarray:
        return self.F0 * np.cos(np.pi * np.asarray(x, dtype=float) / self.length) + self.F1

    def rho(self, x: np.ndarray) -> np.ndarray:
        L = self.length
        return -self.F0 * (np.pi / L) * np.sin(np.pi * np.asarray(x, dtype=float) / L)

    def _antiderivative(self, x: float) -> float:
        L = self.length
        return self.F0 * (L / np.pi) * math.sin(math.pi * x / L) + self.F1 * x

    def potential(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        b = self.bounds
        sig = self._sheets()
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            total, jump = [], 0.0
            for k in range(len(self.thicknesses_m)):
                if k:
                    jump += sig[k - 1]
                lo, hi = b[k], min(b[k + 1], xi)
                if hi <= lo:
                    break
                eps = self.eps0 * self.eps_r[k]
                total.append(
                    (self._antiderivative(hi) - self._antiderivative(lo) + jump * (hi - lo))
                    / eps
                )
            out[i] = self.phi_left - math.fsum(total)
        return out

    @property
    def phi_right(self) -> float:
        return float(self.potential(np.array([self.length]))[0])

    def cell_charge(self, faces: np.ndarray) -> np.ndarray:
        """Exact int rho over each cell, plus sheets strictly inside it."""

        faces = np.asarray(faces, dtype=float)
        D = self.smooth_flux(faces)
        out = D[1:] - D[:-1]
        b = self.bounds
        for j, s in enumerate(self._sheets()):
            if s == 0.0:
                continue
            plane = b[j + 1]
            inside = (faces[:-1] < plane) & (faces[1:] > plane)
            if not np.any(inside) and np.any(faces == plane):
                raise ValueError("sheet charge on a cell face needs a two-sided trace")
            out[inside] += s
        return out

    def as_layered_stack(self) -> LayeredStack:
        return LayeredStack(self.thicknesses_m, self.eps_r)


def observed_orders(h: np.ndarray, err: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])

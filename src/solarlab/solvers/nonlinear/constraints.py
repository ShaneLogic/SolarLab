"""Steady-state constraints: domain inventories, frozen fields, the DC port (N class).

* :class:`InventoryConstrainedProblem` -- with blocking boundaries a mobile
  species' steady continuity rows sum to zero in every closed domain, so the
  DC system is rank deficient by one per domain. One declared row of each
  domain is replaced by ``sum_i w_i c_i(u) - N_domain``; the caller supplies
  ``c(u)`` and ``dc/du`` in its own coordinates.
* :class:`FrozenVariables` -- a frozen species keeps its complete profile and
  background bit for bit; only the free variables are solved, and the rows of
  the frozen equations are dropped.
* :class:`DcPortProblem` -- direct ``V_oc``: the bias becomes an unknown and
  the DC terminal-current constraint ``J_dc(u, V) = 0`` closes the system.
  Only the DC total terminal current is a valid port constraint here; a zero
  transient conduction current (the TPV open-circuit condition) is a dynamic
  port constraint and is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, hstack, issparse, vstack

from solarlab.solvers.nonlinear.problem import Matrix, SteadyProblem, Vector

__all__ = [
    "DC_TERMINAL_CURRENT", "DcPortProblem", "DomainInventory", "FrozenVariables",
    "InventoryConstrainedProblem", "PortConstraintError", "TRANSIENT_CONDUCTION_CURRENT",
    "augmented_start",
]

DC_TERMINAL_CURRENT = "dc_terminal_current"
TRANSIENT_CONDUCTION_CURRENT = "transient_conduction_current"


def _dense(matrix: Matrix) -> NDArray[np.float64]:
    return np.asarray(matrix.toarray() if issparse(matrix) else matrix, dtype=float)


def _indices(values: ArrayLike, size: int | None = None) -> NDArray[np.intp]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError("index sets must be non-empty 1-D integer arrays")
    result = array.astype(np.intp)
    if np.unique(result).size != result.size or np.any(result < 0) or (
            size is not None and np.any(result >= size)):
        raise ValueError("index sets must be unique and inside the system")
    return result


@dataclass(frozen=True)
class DomainInventory:
    """``sum_i weights_i c_i(u) = inventory`` over the domain's ``indices``."""

    name: str
    indices: NDArray[np.intp]
    weights: Vector
    inventory: float
    replaced_row: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _indices(self.indices))
        weights = np.array(self.weights, dtype=float, copy=True)
        if weights.shape != self.indices.shape or not np.all(np.isfinite(weights)) \
                or np.any(weights <= 0.0):
            raise ValueError(f"{self.name}: weights must be positive, one per index")
        object.__setattr__(self, "weights", weights)
        if not (np.isfinite(self.inventory) and self.inventory > 0.0):
            raise ValueError(f"{self.name}: inventory must be finite and positive")
        if int(self.replaced_row) not in set(self.indices.tolist()):
            raise ValueError(f"{self.name}: the replaced row must belong to the domain")


@dataclass(frozen=True)
class InventoryConstrainedProblem:
    """Replace one continuity row per closed domain by its inventory."""

    base: SteadyProblem
    domains: tuple[DomainInventory, ...]
    density: Callable[[Vector], Vector]
    density_derivative: Callable[[Vector], Vector]

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(self.domains))
        if not self.domains:
            raise ValueError("at least one inventory domain is required")
        seen: set[int] = set()
        for domain in self.domains:
            members = set(domain.indices.tolist())
            if seen & members:
                raise ValueError("inventory domains must not overlap")
            seen |= members

    def residual(self, u: Vector) -> Vector:
        values = np.array(self.base.residual(u), dtype=float, copy=True)
        density = np.asarray(self.density(u), dtype=float)
        for domain in self.domains:
            values[domain.replaced_row] = (
                float(np.dot(domain.weights, density[domain.indices])) - domain.inventory)
        return values

    def jacobian(self, u: Vector) -> NDArray[np.float64]:
        matrix = _dense(self.base.jacobian(u)).copy()
        slope = np.asarray(self.density_derivative(u), dtype=float)
        for domain in self.domains:
            matrix[domain.replaced_row, :] = 0.0
            matrix[domain.replaced_row, domain.indices] = domain.weights * slope[domain.indices]
        return matrix

    def inventories(self, u: Vector) -> dict[str, float]:
        density = np.asarray(self.density(u), dtype=float)
        return {domain.name: float(np.dot(domain.weights, density[domain.indices]))
                for domain in self.domains}


@dataclass(frozen=True)
class FrozenVariables:
    """Solve only the free variables; the frozen entries are held exactly."""

    base: SteadyProblem
    frozen: NDArray[np.intp]
    frozen_values: Vector
    size: int
    free: NDArray[np.intp] = field(init=False)

    def __post_init__(self) -> None:
        frozen = _indices(self.frozen, self.size)
        values = np.array(self.frozen_values, dtype=float, copy=True)
        if values.shape != frozen.shape or not np.all(np.isfinite(values)):
            raise ValueError("one finite frozen value is required per frozen index")
        values.setflags(write=False)
        object.__setattr__(self, "frozen", frozen)
        object.__setattr__(self, "frozen_values", values)
        mask: NDArray[np.bool_] = np.ones(self.size, dtype=bool)
        mask[frozen] = False
        if not np.any(mask):
            raise ValueError("at least one variable must stay free")
        object.__setattr__(self, "free", np.flatnonzero(mask))

    def expand(self, free_values: Vector) -> Vector:
        full: Vector = np.empty(self.size, dtype=float)
        full[self.frozen] = self.frozen_values
        full[self.free] = np.asarray(free_values, dtype=float)
        return full

    def restrict(self, full: Vector) -> Vector:
        result: Vector = np.asarray(full, dtype=float)[self.free].copy()
        return result

    def residual(self, u: Vector) -> Vector:
        result: Vector = np.asarray(self.base.residual(self.expand(u)), dtype=float)[self.free]
        return result

    def jacobian(self, u: Vector) -> Matrix:
        matrix = self.base.jacobian(self.expand(u))
        if issparse(matrix):
            return matrix.tocsr()[self.free][:, self.free]
        return np.asarray(matrix, dtype=float)[np.ix_(self.free, self.free)]


class PortConstraintError(ValueError):
    """A port constraint of the wrong kind was requested."""


@dataclass(frozen=True)
class DcPortProblem:
    """Unknowns ``(u, V)``; equations ``F(u, V) = 0`` and ``J_dc(u, V) / J_scale = 0``.

    ``problem_at(V)`` builds the steady problem at bias ``V``; ``dresidual_dV``
    is ``dF/dV``; ``terminal_current`` and its two derivatives describe the DC
    total terminal current. ``current_scale`` [A/m^2] makes the port row
    dimensionless (no default).
    """

    problem_at: Callable[[float], SteadyProblem]
    dresidual_dV: Callable[[Vector, float], Vector]
    terminal_current: Callable[[Vector, float], float]
    terminal_current_gradient: Callable[[Vector, float], Vector]
    terminal_current_dV: Callable[[Vector, float], float]
    current_scale: float
    kind: str = DC_TERMINAL_CURRENT

    def __post_init__(self) -> None:
        if self.kind == TRANSIENT_CONDUCTION_CURRENT:
            raise PortConstraintError(
                "a zero transient conduction current is the dynamic (TPV) open-circuit "
                "constraint; the DC solver only closes on the DC terminal current")
        if self.kind != DC_TERMINAL_CURRENT:
            raise PortConstraintError(f"unknown port constraint kind {self.kind!r}")
        if not (np.isfinite(self.current_scale) and self.current_scale > 0.0):
            raise ValueError("current_scale must be finite and positive")

    @staticmethod
    def split(z: Vector) -> tuple[Vector, float]:
        return np.asarray(z[:-1], dtype=float), float(z[-1])

    def residual(self, z: Vector) -> Vector:
        u, V = self.split(z)
        values = np.asarray(self.problem_at(V).residual(u), dtype=float)
        return np.append(values, self.terminal_current(u, V) / self.current_scale)

    def jacobian(self, z: Vector) -> Matrix:
        u, V = self.split(z)
        base = self.problem_at(V).jacobian(u)
        column = np.asarray(self.dresidual_dV(u, V), dtype=float)[:, None]
        row = np.append(np.asarray(self.terminal_current_gradient(u, V), dtype=float),
                        self.terminal_current_dV(u, V)) / self.current_scale
        if issparse(base):
            top = hstack([base, csr_matrix(column)])
            return vstack([top, csr_matrix(row[None, :])]).tocsr()
        return np.vstack([np.hstack([np.asarray(base, dtype=float), column]), row[None, :]])


def augmented_start(u: Vector, V: float) -> Vector:
    """``(u, V)`` as one vector for :class:`DcPortProblem`."""
    return np.append(np.asarray(u, dtype=float), float(V))

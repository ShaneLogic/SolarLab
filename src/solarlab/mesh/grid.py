"""Historical 1D grid constructors, node ownership and interface stencils.

``tanh_grid`` and ``multilayer_grid`` reproduce the legacy constructors
bit-for-bit. Node ownership at a shared layer-boundary node differs between
the historical consumers and is therefore explicit:

* ``RIGHT_LAYER_WINS`` -- 1D method-of-lines material arrays: inclusive spans
  with a roundoff padding, assigned in layer order so the right layer owns the
  boundary node. The 2D material arrays are extruded from these, so they are
  right-owned too.
* ``LEFT_LAYER_WINS`` -- 2D role tagging only (absorber range, grain-boundary
  painting): the first layer whose cumulative right boundary (plus 1e-15 m) is
  not below the node owns it (upstream). It never selects 2D material values.

Two-sided interface geometries remove the shared node instead, so each
material keeps strictly interior reservoirs on both sides of the true plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "InterfaceStencil", "LEFT_LAYER_WINS", "RIGHT_LAYER_WINS", "build_interface_stencils",
    "layer_boundary_padding", "layer_node_masks", "multilayer_grid", "node_owner",
    "remove_shared_interface_nodes", "tanh_grid",
]

RIGHT_LAYER_WINS = "right_layer_wins"
LEFT_LAYER_WINS = "left_layer_wins"
_TWOD_BOUNDARY_TOLERANCE_M = 1e-15


def tanh_grid(N: int, L: float, alpha: float = 3.0) -> NDArray[np.float64]:
    """N+1 points on [0, L] with boundary concentration parameter alpha."""
    xi = np.linspace(-1.0, 1.0, N + 1)
    x: NDArray[np.float64] = L * (1.0 + np.tanh(alpha * xi) / np.tanh(alpha)) / 2.0
    return x


def multilayer_grid(
    layers: Sequence[tuple[float, int]],
    alpha: float | Sequence[float] = 3.0,
) -> NDArray[np.float64]:
    """Concatenated tanh grid for ``(thickness, intervals)`` layers.

    ``alpha`` may be one scalar shared by every layer or one value per layer.
    The shared boundary node is kept once (the deduplicated historical grid).
    """
    alpha_array: NDArray[np.float64] = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 0:
        layer_alphas = (float(alpha_array),) * len(layers)
    elif alpha_array.ndim == 1 and len(alpha_array) == len(layers):
        layer_alphas = tuple(float(value) for value in alpha_array)
    else:
        raise ValueError("alpha must be a scalar or contain exactly one value per layer")

    segments = []
    offset = 0.0
    for k, ((thickness, intervals), layer_alpha) in enumerate(zip(layers, layer_alphas)):
        x_seg = tanh_grid(intervals, thickness, layer_alpha) + offset
        if k > 0:
            x_seg = x_seg[1:]   # drop duplicate interface point
        segments.append(x_seg)
        offset += thickness
    return np.concatenate(segments)


def layer_boundary_padding(x: ArrayLike, thicknesses: Sequence[float]) -> float:
    """Roundoff allowance that cannot consume a resolved interface interval."""
    grid = np.asarray(x, dtype=float)
    scale = max(abs(float(grid[0])), abs(float(grid[-1])), math.fsum(thicknesses))
    roundoff = 2 * (len(thicknesses) + 1) * np.spacing(scale)
    return float(min(roundoff, 0.25 * float(np.min(np.diff(grid)))))


def node_owner(x: ArrayLike, thicknesses: Sequence[float], convention: str) -> NDArray[np.intp]:
    """Owning layer index of every node under one named historical convention."""
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a finite 1-D array with at least two nodes")
    if not thicknesses:
        raise ValueError("at least one layer is required")
    if convention == RIGHT_LAYER_WINS:
        padding = layer_boundary_padding(grid, thicknesses)
        owner = np.full(grid.shape, -1, dtype=np.intp)
        offset = 0.0
        for index, thickness in enumerate(thicknesses):
            span = (grid >= offset - padding) & (grid <= offset + thickness + padding)
            owner[span] = index
            offset += thickness
        return owner
    if convention == LEFT_LAYER_WINS:
        boundaries = [0.0]
        for thickness in thicknesses:
            boundaries.append(boundaries[-1] + thickness)
        result = np.empty(grid.shape, dtype=np.intp)
        for node, value in enumerate(grid):
            for k in range(len(thicknesses)):
                if value <= boundaries[k + 1] + _TWOD_BOUNDARY_TOLERANCE_M:
                    result[node] = k
                    break
            else:
                result[node] = len(thicknesses) - 1
        return result
    raise ValueError(f"unknown node ownership convention {convention!r}")


def layer_node_masks(x: ArrayLike, thicknesses: Sequence[float],
                     convention: str = RIGHT_LAYER_WINS) -> list[NDArray[np.bool_]]:
    """One mutually exclusive boolean mask per layer."""
    owner = node_owner(x, thicknesses, convention)
    return [owner == index for index in range(len(thicknesses))]


@dataclass(frozen=True)
class InterfaceStencil:
    """Grid indices bracketing a material boundary with interior reservoirs.

    ``shared_boundary_node`` records the node of the deduplicated grid that
    lies on the plane, when present. It is never used as a reservoir, because
    its distance to the physical interface is zero.
    """

    interface_position_m: float
    left_bulk_node: int
    right_bulk_node: int
    left_distance_m: float
    right_distance_m: float
    shared_boundary_node: int | None


def _coordinate_tolerance(grid: NDArray[np.float64]) -> float:
    return float(
        32.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(float(grid[0])),
            abs(float(grid[-1])),
            abs(float(grid[-1] - grid[0])),
        )
    )


def _strict_grid_and_positions(
    grid_m: ArrayLike, interface_positions_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    grid = np.asarray(grid_m, dtype=float)
    positions = np.asarray(interface_positions_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("interface_positions_m must be a finite vector")
    if positions.size and np.any(np.diff(positions) <= 0.0):
        raise ValueError("interface_positions_m must be strictly increasing")
    return grid, positions


def build_interface_stencils(
    grid_m: ArrayLike,
    interface_positions_m: ArrayLike,
) -> tuple[InterfaceStencil, ...]:
    """Map physical interfaces to the nearest strict left/right bulk nodes."""
    grid, positions = _strict_grid_and_positions(grid_m, interface_positions_m)
    tolerance = _coordinate_tolerance(grid)
    stencils: list[InterfaceStencil] = []
    for position_value in positions:
        position = float(position_value)
        if not float(grid[0]) < position < float(grid[-1]):
            raise ValueError("each interface position must lie inside grid_m")
        nearest = int(np.argmin(np.abs(grid - position)))
        shared = nearest if abs(float(grid[nearest]) - position) <= tolerance else None
        if shared is None:
            right = int(np.searchsorted(grid, position, side="right"))
            left = right - 1
        else:
            left = shared - 1
            right = shared + 1
        if left < 0 or right >= grid.size:
            raise ValueError("interface lacks strict left/right bulk nodes")
        left_distance = position - float(grid[left])
        right_distance = float(grid[right]) - position
        if left_distance <= 0.0 or right_distance <= 0.0:
            raise ValueError("interface stencil contains a zero-distance half-cell")
        stencils.append(
            InterfaceStencil(
                interface_position_m=position,
                left_bulk_node=left,
                right_bulk_node=right,
                left_distance_m=left_distance,
                right_distance_m=right_distance,
                shared_boundary_node=shared,
            )
        )
    return tuple(stencils)


def remove_shared_interface_nodes(
    grid_m: ArrayLike,
    interface_positions_m: ArrayLike,
) -> NDArray[np.float64]:
    """Return a strict grid with deduplicated material-boundary nodes removed."""
    grid, positions = _strict_grid_and_positions(grid_m, interface_positions_m)
    tolerance = _coordinate_tolerance(grid)
    keep = np.ones(grid.size, dtype=bool)
    for position_value in positions:
        position = float(position_value)
        if not float(grid[0]) < position < float(grid[-1]):
            raise ValueError("each interface position must lie inside grid_m")
        nearest = int(np.argmin(np.abs(grid - position)))
        if abs(float(grid[nearest]) - position) <= tolerance:
            keep[nearest] = False
    reduced = grid[keep]
    if reduced.size < 3:
        raise ValueError("removing interface nodes leaves fewer than three nodes")
    stencils = build_interface_stencils(reduced, positions)
    if any(stencil.shared_boundary_node is not None for stencil in stencils):
        raise RuntimeError("shared interface node remained after grid reduction")
    return reduced

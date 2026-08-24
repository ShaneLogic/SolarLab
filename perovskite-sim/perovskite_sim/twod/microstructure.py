from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from perovskite_sim.twod.grid_2d import Grid2D


@dataclass(frozen=True)
class GrainBoundary:
    """A finite-width vertical grain-boundary band.

    ``width`` and ``x_position`` describe physical geometry in metres. The
    lifetime parameters apply only to the fraction of each lateral control
    volume covered by that band; they are not node-centred paint values.
    """

    x_position: float
    width: float
    tau_n: float
    tau_p: float
    layer_role: str = "absorber"

    def __post_init__(self) -> None:
        scalar_fields = {
            "x_position": self.x_position,
            "width": self.width,
            "tau_n": self.tau_n,
            "tau_p": self.tau_p,
        }
        for name, value in scalar_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"grain boundary {name} must be finite")
        if self.width <= 0.0:
            raise ValueError("grain boundary width must be positive")
        if self.tau_n <= 0.0 or self.tau_p <= 0.0:
            raise ValueError("grain boundary lifetimes must be positive")
        if not self.layer_role.strip():
            raise ValueError("grain boundary layer_role must be non-empty")


@dataclass(frozen=True)
class Microstructure:
    """Container for spatially varying defect features in a 2D simulation."""

    grain_boundaries: tuple[GrainBoundary, ...] = ()


@dataclass(frozen=True)
class GrainBoundaryRegion2D:
    """Immutable finite-volume representation of one grain-boundary band."""

    x_position: float
    physical_width: float
    tau_n: float
    tau_p: float
    layer_role: str
    x_overlap_fraction: np.ndarray
    y_mask: np.ndarray


_GB_KEYS = frozenset({"x_position", "width", "tau_n", "tau_p", "layer_role"})
_MICROSTRUCTURE_KEYS = frozenset({"grain_boundaries"})


def load_microstructure_from_yaml_block(block: Mapping[str, Any] | None) -> Microstructure:
    """Parse a strict YAML ``microstructure:`` block."""
    if not block:
        return Microstructure()
    unknown_block = set(block) - _MICROSTRUCTURE_KEYS
    if unknown_block:
        raise ValueError(f"microstructure unknown key(s): {sorted(unknown_block)}")
    raw_gbs = block.get("grain_boundaries") or ()
    if not isinstance(raw_gbs, Sequence) or isinstance(raw_gbs, (str, bytes)):
        raise ValueError("microstructure.grain_boundaries must be a sequence")

    gbs: list[GrainBoundary] = []
    for index, entry in enumerate(raw_gbs):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"microstructure.grain_boundaries[{index}] must be a mapping"
            )
        unknown = set(entry) - _GB_KEYS
        if unknown:
            raise ValueError(
                f"microstructure.grain_boundaries unknown key(s): {sorted(unknown)}"
            )
        missing = {"x_position", "width", "tau_n", "tau_p"} - set(entry)
        if missing:
            raise ValueError(
                "microstructure.grain_boundaries missing key(s): "
                f"{sorted(missing)}"
            )
        gbs.append(
            GrainBoundary(
                x_position=float(entry["x_position"]),
                width=float(entry["width"]),
                tau_n=float(entry["tau_n"]),
                tau_p=float(entry["tau_p"]),
                layer_role=str(entry.get("layer_role", "absorber")),
            )
        )
    return Microstructure(grain_boundaries=tuple(gbs))


def lateral_dual_cell_bounds(x: np.ndarray) -> np.ndarray:
    """Return finite-volume bounds for a non-periodic nodal grid."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size < 2:
        raise ValueError("lateral grid must be a one-dimensional array of size >= 2")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("lateral grid coordinates must be finite")
    if np.any(np.diff(x_arr) <= 0.0):
        raise ValueError("lateral grid coordinates must be strictly increasing")
    bounds = np.empty(x_arr.size + 1, dtype=float)
    bounds[0] = x_arr[0]
    bounds[-1] = x_arr[-1]
    bounds[1:-1] = 0.5 * (x_arr[:-1] + x_arr[1:])
    return bounds


def lateral_dual_cell_widths(x: np.ndarray) -> np.ndarray:
    """Return non-periodic lateral control-volume widths."""
    return np.diff(lateral_dual_cell_bounds(x))


def _immutable_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_nonoverlapping_bands(
    grain_boundaries: tuple[GrainBoundary, ...],
) -> None:
    for i, left in enumerate(grain_boundaries):
        left_lo = left.x_position - 0.5 * left.width
        left_hi = left.x_position + 0.5 * left.width
        for right in grain_boundaries[i + 1 :]:
            if left.layer_role != right.layer_role:
                continue
            right_lo = right.x_position - 0.5 * right.width
            right_hi = right.x_position + 0.5 * right.width
            if min(left_hi, right_hi) > max(left_lo, right_lo):
                raise ValueError(
                    "overlapping grain-boundary bands in the same layer role "
                    "are not supported"
                )


def build_grain_boundary_regions(
    grid: Grid2D,
    ustruct: Microstructure,
    layer_role_per_y: Sequence[str],
    *,
    lateral_bc: str,
) -> tuple[GrainBoundaryRegion2D, ...]:
    """Map physical GB bands to exact lateral control-volume fractions.

    The current periodic 2D topology keeps duplicate nodes at both ``x=0``
    and ``x=L`` and therefore has no unique physical control-volume partition.
    A non-empty microstructure is consequently admitted only with Neumann
    lateral boundaries until that topology is replaced. Empty microstructures
    retain the existing periodic path unchanged.
    """
    if not ustruct.grain_boundaries:
        return ()
    if lateral_bc != "neumann":
        raise ValueError(
            "grain-boundary area closure requires lateral_bc='neumann'; "
            "the duplicate-endpoint periodic topology is not area-certified"
        )
    if len(layer_role_per_y) != grid.Ny:
        raise ValueError("layer_role_per_y length must match the vertical grid size")

    bounds = lateral_dual_cell_bounds(grid.x)
    widths = np.diff(bounds)
    domain_lo = float(bounds[0])
    domain_hi = float(bounds[-1])
    domain_width = domain_hi - domain_lo
    _validate_nonoverlapping_bands(ustruct.grain_boundaries)

    regions: list[GrainBoundaryRegion2D] = []
    roles = np.asarray(tuple(layer_role_per_y), dtype=object)
    for gb in ustruct.grain_boundaries:
        band_lo = gb.x_position - 0.5 * gb.width
        band_hi = gb.x_position + 0.5 * gb.width
        scale = max(1.0, abs(domain_lo), abs(domain_hi), abs(gb.x_position))
        tolerance = 16.0 * np.finfo(float).eps * scale
        if gb.width > domain_width + tolerance:
            raise ValueError("grain boundary width exceeds the lateral domain")
        if band_lo < domain_lo - tolerance or band_hi > domain_hi + tolerance:
            raise ValueError(
                "grain-boundary band must lie fully inside the Neumann domain"
            )
        band_lo = max(band_lo, domain_lo)
        band_hi = min(band_hi, domain_hi)

        overlap = np.maximum(
            0.0,
            np.minimum(bounds[1:], band_hi) - np.maximum(bounds[:-1], band_lo),
        )
        fractions = overlap / widths
        integrated_width = float(np.dot(fractions, widths))
        if not np.isclose(
            integrated_width,
            gb.width,
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps * max(gb.width, domain_width),
        ):
            raise RuntimeError(
                "grain-boundary control-volume overlap does not preserve width"
            )
        y_mask = roles == gb.layer_role
        if not np.any(y_mask):
            raise ValueError(
                f"grain-boundary layer_role {gb.layer_role!r} matches no grid rows"
            )
        regions.append(
            GrainBoundaryRegion2D(
                x_position=gb.x_position,
                physical_width=gb.width,
                tau_n=gb.tau_n,
                tau_p=gb.tau_p,
                layer_role=gb.layer_role,
                x_overlap_fraction=_immutable_array(fractions, dtype=np.dtype(float)),
                y_mask=_immutable_array(y_mask, dtype=np.dtype(bool)),
            )
        )
    return tuple(regions)


def build_tau_field(
    grid: Grid2D,
    ustruct: Microstructure,
    tau_n_bulk_per_y: np.ndarray,
    tau_p_bulk_per_y: np.ndarray,
    layer_role_per_y: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude bulk lifetimes without node-painting a finite-width GB.

    This helper is retained for the empty-microstructure compatibility path.
    A non-empty descriptor must use :func:`build_grain_boundary_regions` and
    the recombination mixture in ``solver_2d``; assigning a whole nodal cell a
    GB lifetime makes the effective defect area grid dependent.
    """
    del layer_role_per_y
    if ustruct.grain_boundaries:
        raise ValueError(
            "node-painted GB lifetimes are not area-conservative; build finite-volume "
            "grain-boundary regions instead"
        )
    Nx, Ny = grid.Nx, grid.Ny
    tau_n = np.broadcast_to(tau_n_bulk_per_y[:, None], (Ny, Nx)).copy()
    tau_p = np.broadcast_to(tau_p_bulk_per_y[:, None], (Ny, Nx)).copy()
    return tau_n, tau_p


__all__ = [
    "GrainBoundary",
    "GrainBoundaryRegion2D",
    "Microstructure",
    "build_grain_boundary_regions",
    "build_tau_field",
    "lateral_dual_cell_bounds",
    "lateral_dual_cell_widths",
    "load_microstructure_from_yaml_block",
]

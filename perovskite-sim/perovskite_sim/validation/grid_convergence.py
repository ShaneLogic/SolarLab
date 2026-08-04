"""Grid-convergence arithmetic that uses actual interval counts."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.solver.mol import (
    StateVec,
    _layer_node_masks,
    build_material_arrays,
)


@dataclass(frozen=True)
class ConvergenceSeries:
    """Three-rung scalar convergence series on geometrically refined grids."""

    intervals: tuple[int, int, int]
    values: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(n <= 0 for n in self.intervals):
            raise ValueError("interval counts must be positive")
        if not self.intervals[0] < self.intervals[1] < self.intervals[2]:
            raise ValueError("interval counts must be strictly increasing")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("convergence values must be finite")
        r1 = self.intervals[1] / self.intervals[0]
        r2 = self.intervals[2] / self.intervals[1]
        if not math.isclose(r1, r2, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("the three interval counts must share one refinement ratio")

    @property
    def refinement_ratio(self) -> float:
        return self.intervals[1] / self.intervals[0]

    @property
    def differences(self) -> tuple[float, float]:
        return self.values[1] - self.values[0], self.values[2] - self.values[1]

    @property
    def contraction(self) -> float:
        coarse, fine = self.differences
        if coarse == 0.0:
            return 0.0 if fine == 0.0 else math.inf
        return abs(fine / coarse)

    @property
    def contracts(self) -> bool:
        return self.contraction < 1.0

    @property
    def observed_order(self) -> float:
        rho = self.contraction
        if rho <= 0.0 or not math.isfinite(rho):
            return math.inf if rho == 0.0 else math.nan
        return -math.log(rho) / math.log(self.refinement_ratio)

    @property
    def richardson_residual(self) -> float:
        """Estimated remaining finest-grid error for an asymptotic series."""
        rho = self.contraction
        if rho >= 1.0:
            return math.inf
        if rho == 0.0:
            return 0.0
        return abs(self.differences[1]) * rho / (1.0 - rho)


def prolong_packed_state(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    stack: DeviceStack,
) -> np.ndarray:
    """Prolong a packed drift-diffusion state onto a finer device grid.

    Carrier densities use shape-preserving cubic interpolation on the shared
    nodal field. Mobile ions are interpolated layer by layer as deviations
    from each grid's neutral ionic background. This prevents an absorber
    ``P0`` discontinuity from being smeared into an ion-blocking transport
    layer. Interface-plane states are zero-dimensional unknowns, so their
    trailing block is copied exactly.

    This only constructs a warm start. Callers must certify the target state
    with the steady-state residual and current checks.
    """
    source = np.asarray(x_source, dtype=float)
    target = np.asarray(x_target, dtype=float)
    state = np.asarray(y_source, dtype=float)
    for name, grid in (("source", source), ("target", target)):
        if (
            grid.ndim != 1
            or len(grid) < 2
            or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError(f"{name} grid must be finite and strictly increasing")
    if len(target) <= len(source):
        raise ValueError("target grid must contain more nodes than source grid")
    if not (
        math.isclose(source[0], target[0], rel_tol=0.0, abs_tol=1.0e-18)
        and math.isclose(
            source[-1], target[-1], rel_tol=1.0e-12, abs_tol=1.0e-18,
        )
    ):
        raise ValueError("source and target grids must span the same device")
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("source state must be a finite one-dimensional array")

    source_mat = build_material_arrays(source, stack)
    target_mat = build_material_arrays(target, stack)
    if (
        source_mat.has_dual_ions != target_mat.has_dual_ions
        or source_mat.N_iface_state != target_mat.N_iface_state
    ):
        raise ValueError("source and target grids resolved different state layouts")
    source_bulk_blocks = 4 if source_mat.has_dual_ions else 3
    expected_source = (
        source_bulk_blocks * len(source) + 4 * source_mat.N_iface_state
    )
    if state.size != expected_source:
        raise ValueError(
            f"source state has size {state.size}; expected {expected_source}"
        )

    source_state = StateVec.unpack(
        state, len(source), source_mat.N_iface_state,
    )

    def _carrier(field: np.ndarray) -> np.ndarray:
        if np.any(field <= 0.0):
            raise ValueError("carrier densities must be strictly positive")
        prolonged = PchipInterpolator(source, field, extrapolate=False)(target)
        if not np.all(np.isfinite(prolonged)) or np.any(prolonged < 0.0):
            raise ValueError("carrier prolongation produced an invalid density")
        # PCHIP is shape-preserving, but subtraction between a contact value
        # near 1e-24 and its 1e5 neighbour can round an interior result to
        # exact zero. Project only that round-off floor to the smallest
        # positive value already present in the certified source state.
        prolonged = np.maximum(prolonged, float(np.min(field)))
        return np.asarray(prolonged, dtype=float)

    layers = electrical_layers(stack)
    source_masks = _layer_node_masks(source, layers)
    target_masks = _layer_node_masks(target, layers)

    def _ion(
        field: np.ndarray,
        source_background: np.ndarray,
        target_background: np.ndarray,
        target_limit: np.ndarray,
    ) -> np.ndarray:
        prolonged = np.asarray(target_background, dtype=float).copy()
        for source_mask, target_mask in zip(source_masks, target_masks):
            delta = field[source_mask] - source_background[source_mask]
            prolonged[target_mask] += np.interp(
                target[target_mask], source[source_mask], delta,
            )
        if not np.all(np.isfinite(prolonged)):
            raise ValueError("ion prolongation produced a nonfinite density")
        target_limit = np.asarray(target_limit, dtype=float)
        allowance = 1.0e-12 * np.maximum(target_limit, 1.0)
        if (
            np.any(prolonged < -allowance)
            or np.any(prolonged > target_limit + allowance)
        ):
            raise ValueError("prolonged ion density violates its occupancy bounds")
        return np.clip(prolonged, 0.0, target_limit)

    n_target = _carrier(source_state.n)
    p_target = _carrier(source_state.p)
    p_ion_target = _ion(
        source_state.P,
        source_mat.P_ion0,
        target_mat.P_ion0,
        target_mat.P_lim_node,
    )
    p_neg_target = None
    if source_mat.has_dual_ions:
        if source_state.P_neg is None:
            raise ValueError("dual-ion material state is missing its negative-ion block")
        p_neg_target = _ion(
            source_state.P_neg,
            source_mat.P_ion0_neg,
            target_mat.P_ion0_neg,
            target_mat.P_lim_neg_node,
        )
        if target_mat.ion_steric_shared_site:
            shared_limit = np.minimum(
                target_mat.P_lim_node, target_mat.P_lim_neg_node,
            )
            if np.any(
                p_ion_target + p_neg_target > shared_limit * (1.0 + 1.0e-12)
            ):
                raise ValueError("prolonged dual ions violate the shared-site limit")

    interface_state = (
        None
        if source_state.iface_state is None
        else np.asarray(source_state.iface_state, dtype=float).copy()
    )
    return StateVec.pack(
        n_target,
        p_target,
        p_ion_target,
        p_neg_target,
        iface_state=interface_state,
    )

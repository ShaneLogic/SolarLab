from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Literal

import numpy as np

from perovskite_sim.constants import EPS_0, K_B, Q
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.physics.doping import doping_at_position
from perovskite_sim.physics.grading import (
    band_gap_profile,
    grade_ni_sq,
    grading_coordinate,
    has_grading_params,
)
from perovskite_sim.physics.temperature import T_REF, eg_at_T, ni_at_T

if TYPE_CHECKING:
    from perovskite_sim.models.device import DeviceStack, LayerSpec


# This guard is deliberately scoped to extreme multi-scale layers.  Thin-film
# smoke meshes can be intentionally coarse, whereas a wafer-scale layer that
# spans thousands of screening lengths can silently return catastrophic J-V
# spikes even when the tanh grid looks dense in absolute node count.
MIN_GUARDED_LAYER_DEBYE_SPAN = 1.0e3
MAX_INTERFACE_CELL_DEBYE_RATIO = 1.5


@dataclass(frozen=True)
class Layer:
    thickness: float   # metres
    N: int             # number of grid intervals in this layer


@dataclass(frozen=True)
class InterfaceGridDiagnostic:
    """Resolution of one interface-adjacent cell against its local Debye scale.

    This is a necessary electrostatic-resolution diagnostic, not a mesh-
    convergence certificate.  A passing value says nothing about bulk optical
    resolution, terminal-current conservation, or J-V convergence.
    """

    interface_index: int
    side: Literal["left", "right"]
    layer_index: int
    layer_name: str
    cell_width: float
    debye_length: float
    cell_debye_ratio: float
    layer_debye_span: float


class GridResolutionError(ValueError):
    """A J-V request cannot resolve a thick layer's interface boundary scale."""

    def __init__(
        self,
        diagnostic: InterfaceGridDiagnostic,
        *,
        N_grid: int,
        max_cell_debye_ratio: float,
    ) -> None:
        self.diagnostic = diagnostic
        self.N_grid = int(N_grid)
        self.max_cell_debye_ratio = float(max_cell_debye_ratio)
        super().__init__(
            "under-resolved electrical grid before integration: "
            f"N_grid={self.N_grid}, interface {diagnostic.interface_index} "
            f"{diagnostic.side} cell in layer {diagnostic.layer_name!r} has "
            f"dx={diagnostic.cell_width:.6g} m = "
            f"{diagnostic.cell_debye_ratio:.3f} local Debye lengths "
            f"(limit {self.max_cell_debye_ratio:.3f}); the layer spans "
            f"{diagnostic.layer_debye_span:.3g} Debye lengths. Increase N_grid "
            "or set allow_underresolved_grid=True only for an explicitly "
            "non-physical diagnostic run."
        )


def _solver_intrinsic_density_at_endpoint(
    layer: "LayerSpec",
    stack: "DeviceStack",
    *,
    endpoint: Literal["front", "back"],
) -> tuple[float, float]:
    """Return ``(ni, T)`` using the same mode/temperature/grading law as MoL."""
    if layer.params is None:
        return 0.0, T_REF
    params = layer.params
    sim_mode = resolve_mode(getattr(stack, "mode", "full"))
    temperature = float(stack.T) if sim_mode.use_temperature_scaling else T_REF
    if not np.isfinite(temperature) or temperature <= 0.0:
        return float("nan"), temperature

    if sim_mode.use_temperature_scaling:
        eg_front = eg_at_T(
            params.Eg,
            temperature,
            params.varshni_alpha,
            params.varshni_beta,
        )
    else:
        eg_front = params.Eg
    ni_front = ni_at_T(
        params.ni,
        eg_front,
        temperature,
        params.Nc300,
        params.Nv300,
    )

    grading_enabled = (
        bool(getattr(stack, "band_grading", False))
        or os.environ.get("SOLARLAB_BAND_GRADING") == "1"
    ) and sim_mode.name != "legacy"
    if not grading_enabled or not has_grading_params(params):
        return float(ni_front), temperature

    eg_back = params.Eg_back if params.Eg_back is not None else params.Eg
    if sim_mode.use_temperature_scaling:
        eg_back = eg_at_T(
            eg_back,
            temperature,
            params.varshni_alpha,
            params.varshni_beta,
        )
    x_local = np.array([0.0 if endpoint == "front" else layer.thickness])
    composition = grading_coordinate(
        x_local,
        layer.thickness,
        params.grading_profile,
        params.grading_char_length,
        params.grading_direction,
    )
    local_eg = band_gap_profile(
        composition,
        eg_front,
        eg_back,
        params.grading_bowing,
    )
    ni_sq = grade_ni_sq(
        ni_front ** 2,
        local_eg,
        eg_front,
        K_B * temperature / Q,
    )
    return float(np.sqrt(ni_sq[0])), temperature


def _layer_debye_length(
    layer: "LayerSpec",
    stack: "DeviceStack",
    *,
    endpoint: Literal["front", "back"],
) -> float:
    """Small-signal carrier Debye length from the solver's local dark state."""
    if layer.params is None:
        return float("inf")
    params = layer.params
    intrinsic_density, temperature = _solver_intrinsic_density_at_endpoint(
        layer,
        stack,
        endpoint=endpoint,
    )
    # At mass-action equilibrium n + p = hypot(N_D - N_A, 2 n_i).  Using the
    # mobile-carrier sum handles intrinsic and compensated layers without the
    # cancellation in max(N_D-N_A, n_i).
    position = 0.0 if endpoint == "front" else float(layer.thickness)
    local_N_A, local_N_D = doping_at_position(
        params, position, float(layer.thickness)
    )
    screening_density = float(np.hypot(
        local_N_D - local_N_A,
        2.0 * intrinsic_density,
    ))
    if (
        not np.isfinite(screening_density)
        or screening_density <= 0.0
        or not np.isfinite(params.eps_r)
        or params.eps_r <= 0.0
        or not np.isfinite(temperature)
        or temperature <= 0.0
    ):
        return float("inf")
    return float(
        np.sqrt(params.eps_r * EPS_0 * K_B * temperature / (Q * Q * screening_density))
    )


def interface_grid_diagnostics(
    x: np.ndarray,
    stack: "DeviceStack",
) -> tuple[InterfaceGridDiagnostic, ...]:
    """Measure both cells adjacent to every electrical-layer interface."""
    from perovskite_sim.models.device import electrical_layers

    grid = np.asarray(x, dtype=float)
    layers = electrical_layers(stack)
    if len(layers) < 2:
        return ()
    if grid.ndim != 1 or len(grid) < 3 or not np.all(np.diff(grid) > 0.0):
        raise ValueError("electrical grid must be a strictly increasing 1-D array")

    diagnostics: list[InterfaceGridDiagnostic] = []
    interface_position = 0.0
    scale = max(float(grid[-1] - grid[0]), np.finfo(float).tiny)
    atol = 16.0 * np.finfo(float).eps * scale
    for interface_index, (left, right) in enumerate(zip(layers[:-1], layers[1:])):
        interface_position += left.thickness
        node = int(np.argmin(np.abs(grid - interface_position)))
        if (
            node <= 0
            or node >= len(grid) - 1
            or abs(float(grid[node]) - interface_position) > atol
        ):
            raise ValueError(
                f"electrical grid does not contain interface {interface_index} "
                f"at x={interface_position:.6g} m"
            )
        for side, layer_index, layer, endpoint, cell_width in (
            (
                "left", interface_index, left, "back",
                float(grid[node] - grid[node - 1]),
            ),
            (
                "right", interface_index + 1, right, "front",
                float(grid[node + 1] - grid[node]),
            ),
        ):
            debye_length = _layer_debye_length(
                layer,
                stack,
                endpoint=endpoint,
            )
            diagnostics.append(InterfaceGridDiagnostic(
                interface_index=interface_index,
                side=side,
                layer_index=layer_index,
                layer_name=layer.name,
                cell_width=cell_width,
                debye_length=debye_length,
                cell_debye_ratio=cell_width / debye_length,
                layer_debye_span=layer.thickness / debye_length,
            ))
    return tuple(diagnostics)


def require_thick_layer_interface_resolution(
    x: np.ndarray,
    stack: "DeviceStack",
    *,
    N_grid: int,
    allow_underresolved_grid: bool = False,
    min_layer_debye_span: float = MIN_GUARDED_LAYER_DEBYE_SPAN,
    max_cell_debye_ratio: float = MAX_INTERFACE_CELL_DEBYE_RATIO,
) -> tuple[InterfaceGridDiagnostic, ...]:
    """Fail before integration when an extreme multi-scale layer is unresolved.

    The guard is intentionally necessary-but-not-sufficient.  It activates
    only for a layer spanning at least ``min_layer_debye_span`` screening
    lengths, then requires its interface-adjacent cell to be no wider than
    ``max_cell_debye_ratio`` local Debye lengths.  This catches the shipped
    180 um c-Si wafer at the generic N_grid=100 default without turning coarse
    thin-film smoke protocols into an undocumented global convergence claim.
    """
    diagnostics = interface_grid_diagnostics(x, stack)
    offenders = tuple(
        item for item in diagnostics
        if item.layer_debye_span >= min_layer_debye_span
        and item.cell_debye_ratio > max_cell_debye_ratio
    )
    if offenders and not allow_underresolved_grid:
        worst = max(offenders, key=lambda item: item.cell_debye_ratio)
        raise GridResolutionError(
            worst,
            N_grid=N_grid,
            max_cell_debye_ratio=max_cell_debye_ratio,
        )
    return diagnostics


def tanh_grid(N: int, L: float, alpha: float = 3.0) -> np.ndarray:
    """N+1 points on [0, L] with boundary concentration parameter alpha."""
    xi = np.linspace(-1.0, 1.0, N + 1)
    x = L * (1.0 + np.tanh(alpha * xi) / np.tanh(alpha)) / 2.0
    return x


def multilayer_grid(
    layers: list[Layer],
    alpha: float | Sequence[float] = 3.0,
) -> np.ndarray:
    """Concatenated tanh grid for a stack of layers.

    ``alpha`` may be one scalar shared by every layer or one value per layer.
    The scalar path preserves the historical grid exactly. Per-layer values
    let a thick/thin stack balance interface resolution without forcing the
    thin layer onto unnecessarily small cells.
    """
    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 0:
        layer_alphas = (float(alpha_array),) * len(layers)
    elif alpha_array.ndim == 1 and len(alpha_array) == len(layers):
        layer_alphas = tuple(float(value) for value in alpha_array)
    else:
        raise ValueError(
            "alpha must be a scalar or contain exactly one value per layer"
        )

    segments = []
    offset = 0.0
    for k, (layer, layer_alpha) in enumerate(zip(layers, layer_alphas)):
        x_seg = tanh_grid(layer.N, layer.thickness, layer_alpha) + offset
        if k > 0:
            x_seg = x_seg[1:]   # drop duplicate interface point
        segments.append(x_seg)
        offset += layer.thickness
    return np.concatenate(segments)

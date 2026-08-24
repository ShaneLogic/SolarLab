from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.discretization.fe_operators import bernoulli


@dataclass(frozen=True, slots=True)
class IonFluxes2D:
    """Positive-ion particle fluxes on x and y faces [m^-2 s^-1]."""

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class MobileIonDiagnostics2D:
    """Terminal conservation and physical-bound report for one transient."""

    initial_inventory_m1: float
    terminal_inventory_m1: float
    relative_inventory_drift: float
    terminal_min_electron_density_m3: float
    terminal_min_hole_density_m3: float
    terminal_min_density_m3: float
    terminal_max_site_fraction: float
    inventory_rtol: float
    passed: bool
    violations: tuple[str, ...]


def _axis_control_volume_widths(coordinates: np.ndarray) -> np.ndarray:
    spacing = np.diff(coordinates)
    widths = np.empty(coordinates.size, dtype=float)
    widths[0] = 0.5 * spacing[0]
    widths[-1] = 0.5 * spacing[-1]
    if coordinates.size > 2:
        widths[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return widths


def control_volume_areas_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return Neumann-domain nodal control-volume areas [m^2]."""
    x_arr = _validated_axis("x", x)
    y_arr = _validated_axis("y", y)
    return np.outer(
        _axis_control_volume_widths(y_arr),
        _axis_control_volume_widths(x_arr),
    )


def ion_inventory_2d(
    x: np.ndarray,
    y: np.ndarray,
    density: np.ndarray,
) -> float:
    """Return the discrete ion inventory per out-of-plane length [m^-1]."""
    values = np.asarray(density, dtype=float)
    expected = (np.asarray(y).size, np.asarray(x).size)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(
            f"ion density must be finite with shape {expected}; got {values.shape}"
        )
    return float(np.sum(values * control_volume_areas_2d(x, y)))


def _validated_axis(name: str, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if (
        axis.ndim != 1
        or axis.size < 2
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, 1-D, and strictly increasing")
    return axis


def _harmonic_faces_x(values: np.ndarray) -> np.ndarray:
    denominator = values[:, :-1] + values[:, 1:]
    return np.divide(
        2.0 * values[:, :-1] * values[:, 1:],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def _harmonic_faces_y(values: np.ndarray) -> np.ndarray:
    denominator = values[:-1, :] + values[1:, :]
    return np.divide(
        2.0 * values[:-1, :] * values[1:, :],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )


def positive_ion_fluxes_2d(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    density: np.ndarray,
    diffusion: np.ndarray,
    thermal_voltage: float,
    site_limit: np.ndarray,
    *,
    steric_diffusion_only: bool = False,
) -> IonFluxes2D:
    """Return positive-ion SG particle fluxes on a tensor-product grid."""
    x_arr = _validated_axis("x", x)
    y_arr = _validated_axis("y", y)
    shape = (y_arr.size, x_arr.size)
    potential = np.asarray(phi, dtype=float)
    ions = np.asarray(density, dtype=float)
    diffusivity = np.asarray(diffusion, dtype=float)
    limit = np.asarray(site_limit, dtype=float)
    arrays = {
        "phi": potential,
        "density": ions,
        "diffusion": diffusivity,
        "site_limit": limit,
    }
    for name, array in arrays.items():
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(diffusivity < 0.0):
        raise ValueError("ionic diffusion must be non-negative")
    if np.any(limit <= 0.0):
        raise ValueError("ionic site limits must be positive")
    if not np.isfinite(thermal_voltage) or thermal_voltage <= 0.0:
        raise ValueError("thermal voltage must be finite and positive")

    dx = np.diff(x_arr)[None, :]
    dy = np.diff(y_arr)[:, None]
    diffusion_x = _harmonic_faces_x(diffusivity)
    diffusion_y = _harmonic_faces_y(diffusivity)

    if steric_diffusion_only:
        occupancy = ions / limit
        chemical_potential = -np.log1p(-np.clip(occupancy, 0.0, 0.999999))
        xi_x = (
            (potential[:, 1:] - potential[:, :-1]) / thermal_voltage
            + chemical_potential[:, 1:]
            - chemical_potential[:, :-1]
        )
        xi_y = (
            (potential[1:, :] - potential[:-1, :]) / thermal_voltage
            + chemical_potential[1:, :]
            - chemical_potential[:-1, :]
        )
        effective_x = diffusion_x
        effective_y = diffusion_y
    else:
        limit_x = 0.5 * (limit[:, :-1] + limit[:, 1:])
        limit_y = 0.5 * (limit[:-1, :] + limit[1:, :])
        occupancy_x = 0.5 * (ions[:, :-1] + ions[:, 1:]) / limit_x
        occupancy_y = 0.5 * (ions[:-1, :] + ions[1:, :]) / limit_y
        steric_x = 1.0 / np.maximum(
            1.0 - np.clip(occupancy_x, 0.0, 0.999999), 1.0e-6
        )
        steric_y = 1.0 / np.maximum(
            1.0 - np.clip(occupancy_y, 0.0, 0.999999), 1.0e-6
        )
        xi_x = (potential[:, 1:] - potential[:, :-1]) / thermal_voltage
        xi_y = (potential[1:, :] - potential[:-1, :]) / thermal_voltage
        effective_x = diffusion_x * steric_x
        effective_y = diffusion_y * steric_y

    flux_x = effective_x / dx * (
        bernoulli(xi_x) * ions[:, :-1]
        - bernoulli(-xi_x) * ions[:, 1:]
    )
    flux_y = effective_y / dy * (
        bernoulli(xi_y) * ions[:-1, :]
        - bernoulli(-xi_y) * ions[1:, :]
    )
    if not np.all(np.isfinite(flux_x)) or not np.all(np.isfinite(flux_y)):
        raise ValueError("2D ionic flux produced non-finite values")
    return IonFluxes2D(x=flux_x, y=flux_y)


def positive_ion_continuity_rhs_2d(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    density: np.ndarray,
    diffusion: np.ndarray,
    thermal_voltage: float,
    site_limit: np.ndarray,
    *,
    lateral_bc: str,
    steric_diffusion_only: bool = False,
) -> np.ndarray:
    """Return conservative ``dP/dt`` with blocking boundaries on all sides."""
    if lateral_bc != "neumann":
        raise ValueError(
            "2D mobile-ion transport requires lateral_bc='neumann'; "
            "periodic-x is not topology-certified"
        )
    x_arr = _validated_axis("x", x)
    y_arr = _validated_axis("y", y)
    fluxes = positive_ion_fluxes_2d(
        x_arr,
        y_arr,
        phi,
        density,
        diffusion,
        thermal_voltage,
        site_limit,
        steric_diffusion_only=steric_diffusion_only,
    )
    hx = _axis_control_volume_widths(x_arr)
    hy = _axis_control_volume_widths(y_arr)
    derivative = np.zeros_like(np.asarray(density, dtype=float))

    derivative[:, 0] -= fluxes.x[:, 0] / hx[0]
    derivative[:, -1] += fluxes.x[:, -1] / hx[-1]
    if x_arr.size > 2:
        derivative[:, 1:-1] -= (
            fluxes.x[:, 1:] - fluxes.x[:, :-1]
        ) / hx[None, 1:-1]

    derivative[0, :] -= fluxes.y[0, :] / hy[0]
    derivative[-1, :] += fluxes.y[-1, :] / hy[-1]
    if y_arr.size > 2:
        derivative[1:-1, :] -= (
            fluxes.y[1:, :] - fluxes.y[:-1, :]
        ) / hy[1:-1, None]
    return derivative


def assess_mobile_ion_terminal_2d(
    x: np.ndarray,
    y: np.ndarray,
    initial_density: np.ndarray,
    terminal_density: np.ndarray,
    site_limit: np.ndarray,
    *,
    terminal_electron_density: np.ndarray,
    terminal_hole_density: np.ndarray,
    inventory_rtol: float,
) -> MobileIonDiagnostics2D:
    """Assess terminal bounds and the exact discrete inventory invariant."""
    if not np.isfinite(inventory_rtol) or inventory_rtol < 0.0:
        raise ValueError("inventory_rtol must be finite and non-negative")
    initial = np.asarray(initial_density, dtype=float)
    terminal = np.asarray(terminal_density, dtype=float)
    limit = np.asarray(site_limit, dtype=float)
    electrons = np.asarray(terminal_electron_density, dtype=float)
    holes = np.asarray(terminal_hole_density, dtype=float)
    if (
        initial.shape != terminal.shape
        or terminal.shape != limit.shape
        or electrons.shape != terminal.shape
        or holes.shape != terminal.shape
    ):
        raise ValueError("carrier, ion, and site-limit shapes must match")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(limit)):
        raise ValueError("initial density and site limits must be finite")
    if np.any(initial < 0.0) or np.any(initial > limit) or np.any(limit <= 0.0):
        raise ValueError("initial ion density must lie within positive site limits")

    violations: list[str] = []
    if not np.all(np.isfinite(electrons)):
        violations.append("nonfinite_terminal_electron_density")
        terminal_min_electron = float("nan")
    else:
        terminal_min_electron = float(np.min(electrons))
        if terminal_min_electron < 0.0:
            violations.append("negative_terminal_electron_density")
    if not np.all(np.isfinite(holes)):
        violations.append("nonfinite_terminal_hole_density")
        terminal_min_hole = float("nan")
    else:
        terminal_min_hole = float(np.min(holes))
        if terminal_min_hole < 0.0:
            violations.append("negative_terminal_hole_density")
    if not np.all(np.isfinite(terminal)):
        violations.append("nonfinite_terminal_density")
        terminal_min = float("nan")
        max_fraction = float("nan")
        terminal_inventory = float("nan")
        relative_drift = float("inf")
    else:
        terminal_min = float(np.min(terminal))
        max_fraction = float(np.max(terminal / limit))
        if terminal_min < 0.0:
            violations.append("negative_terminal_density")
        if max_fraction > 1.0:
            violations.append("terminal_site_limit_exceeded")
        terminal_inventory = ion_inventory_2d(x, y, terminal)
        initial_inventory = ion_inventory_2d(x, y, initial)
        scale = max(abs(initial_inventory), np.finfo(float).tiny)
        relative_drift = abs(terminal_inventory - initial_inventory) / scale
        if relative_drift > inventory_rtol:
            violations.append("inventory_drift_exceeded")
    initial_inventory = ion_inventory_2d(x, y, initial)
    return MobileIonDiagnostics2D(
        initial_inventory_m1=initial_inventory,
        terminal_inventory_m1=terminal_inventory,
        relative_inventory_drift=relative_drift,
        terminal_min_electron_density_m3=terminal_min_electron,
        terminal_min_hole_density_m3=terminal_min_hole,
        terminal_min_density_m3=terminal_min,
        terminal_max_site_fraction=max_fraction,
        inventory_rtol=float(inventory_rtol),
        passed=not violations,
        violations=tuple(violations),
    )


__all__ = [
    "IonFluxes2D",
    "MobileIonDiagnostics2D",
    "assess_mobile_ion_terminal_2d",
    "control_volume_areas_2d",
    "ion_inventory_2d",
    "positive_ion_continuity_rhs_2d",
    "positive_ion_fluxes_2d",
]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.twod.ion_migration_2d import positive_ion_fluxes_2d
from perovskite_sim.twod.microstructure import lateral_dual_cell_widths
from perovskite_sim.twod.poisson_2d import solve_poisson_2d
from perovskite_sim.twod.snapshot import SpatialSnapshot2D


@dataclass(frozen=True, slots=True)
class MobileIonCurrentComponents2D:
    """Instantaneous vertical current decomposition in the 2D mobile-ion lane."""

    electron_y_A_m2: np.ndarray
    hole_y_A_m2: np.ndarray
    positive_ion_y_A_m2: np.ndarray
    displacement_y_A_m2: np.ndarray
    total_y_A_m2: np.ndarray
    lateral_average_electron_A_m2: np.ndarray
    lateral_average_hole_A_m2: np.ndarray
    lateral_average_positive_ion_A_m2: np.ndarray
    lateral_average_displacement_A_m2: np.ndarray
    lateral_average_total_A_m2: np.ndarray
    terminal_electron_A_m2: float
    terminal_hole_A_m2: float
    terminal_positive_ion_A_m2: float
    terminal_displacement_A_m2: float
    terminal_total_A_m2: float
    max_face_spread_A_m2: float
    max_relative_face_spread: float
    applied_voltage_rate_V_s: float


def _read_only(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _validated_snapshot(snapshot: SpatialSnapshot2D, material) -> tuple[int, int]:
    x = np.asarray(snapshot.x, dtype=float)
    y = np.asarray(snapshot.y, dtype=float)
    if (
        x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 3
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
        or np.any(np.diff(y) <= 0.0)
    ):
        raise ValueError("mobile-ion current requires finite increasing 2D axes")
    if material.poisson_factor.lateral_bc != "neumann":
        raise ValueError(
            "mobile-ion current requires the Neumann-x control-volume topology"
        )
    if not getattr(material, "has_mobile_ions", False):
        raise ValueError("mobile-ion current requires an active 2D ion state")
    if getattr(material, "has_selective_contacts", False):
        raise ValueError(
            "mobile-ion terminal current does not yet support Robin contacts"
        )
    if snapshot.P_ion is None:
        raise ValueError("mobile-ion current snapshot is missing its ion density")
    if material.D_ion_2d is None or material.P_lim_2d is None:
        raise ValueError("mobile-ion current material arrays are incomplete")
    if not np.array_equal(x, np.asarray(material.grid.x, dtype=float)) or not np.array_equal(
        y,
        np.asarray(material.grid.y, dtype=float),
    ):
        raise ValueError("mobile-ion current snapshot and material grids differ")

    Ny, Nx = y.size, x.size
    nodal_shape = (Ny, Nx)
    face_shape = (Ny - 1, Nx)
    nodal = {
        "potential": snapshot.phi,
        "electron density": snapshot.n,
        "hole density": snapshot.p,
        "ion density": snapshot.P_ion,
    }
    faces = {
        "electron current": snapshot.Jy_n,
        "hole current": snapshot.Jy_p,
    }
    for name, values in nodal.items():
        array = np.asarray(values, dtype=float)
        if array.shape != nodal_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {nodal_shape}")
    for name, values in faces.items():
        array = np.asarray(values, dtype=float)
        if array.shape != face_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {face_shape}")
    contact_pairs = (
        ("left electron", snapshot.n[0, :], material.n_eq_left),
        ("right electron", snapshot.n[-1, :], material.n_eq_right),
        ("left hole", snapshot.p[0, :], material.p_eq_left),
        ("right hole", snapshot.p[-1, :], material.p_eq_right),
    )
    for name, actual, expected in contact_pairs:
        if not np.allclose(actual, expected, rtol=1.0e-12, atol=0.0):
            raise ValueError(
                f"mobile-ion current requires the ohmic {name} reservoir state"
            )
    return Ny, Nx


def _lateral_average(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    widths = lateral_dual_cell_widths(x)
    domain_width = float(x[-1] - x[0])
    averaged = np.asarray(values, dtype=float) @ widths / domain_width
    return np.asarray(averaged, dtype=float)


def evaluate_mobile_ion_current_components_2d(
    snapshot: SpatialSnapshot2D,
    state_derivative: np.ndarray,
    material,
    *,
    applied_voltage_rate_V_s: float = 0.0,
) -> MobileIonCurrentComponents2D:
    """Evaluate the instantaneous semidiscrete Maxwell-current decomposition.

    ``state_derivative`` must be the physical-density RHS at ``snapshot.V``.
    Poisson is differentiated with that charge rate, so displacement current
    is instantaneous rather than a secant between voltage samples.
    """
    Ny, Nx = _validated_snapshot(snapshot, material)
    voltage_rate = float(applied_voltage_rate_V_s)
    if not np.isfinite(voltage_rate):
        raise ValueError("applied_voltage_rate_V_s must be finite")

    derivative = np.asarray(state_derivative, dtype=float)
    block_size = Ny * Nx
    if derivative.shape != (3 * block_size,) or not np.all(np.isfinite(derivative)):
        raise ValueError(
            "mobile-ion current derivative must be finite with three 2D blocks"
        )
    dn = derivative[:block_size].reshape(Ny, Nx)
    dp = derivative[block_size : 2 * block_size].reshape(Ny, Nx)
    dP = derivative[2 * block_size :].reshape(Ny, Nx)

    charge_rate = Q * (dp - dn + dP)
    potential_rate = solve_poisson_2d(
        material.poisson_factor,
        charge_rate,
        phi_bottom=0.0,
        phi_top=-float(material.junction_polarity) * voltage_rate,
    )
    dy = np.diff(np.asarray(snapshot.y, dtype=float))[:, None]
    field_rate_y = -(potential_rate[1:, :] - potential_rate[:-1, :]) / dy
    eps_top = np.asarray(material.eps_r[:-1, :], dtype=float)
    eps_bottom = np.asarray(material.eps_r[1:, :], dtype=float)
    eps_face_y = np.divide(
        2.0 * eps_top * eps_bottom,
        eps_top + eps_bottom,
        out=np.zeros_like(eps_top),
        where=(eps_top + eps_bottom) > 0.0,
    )
    displacement_y = EPS_0 * eps_face_y * field_rate_y

    ion_flux = positive_ion_fluxes_2d(
        snapshot.x,
        snapshot.y,
        snapshot.phi,
        snapshot.P_ion,
        material.D_ion_2d,
        material.V_T,
        material.P_lim_2d,
        steric_diffusion_only=material.ion_steric_diffusion_only,
    )
    electron_y = np.asarray(snapshot.Jy_n, dtype=float)
    hole_y = np.asarray(snapshot.Jy_p, dtype=float)
    positive_ion_y = Q * ion_flux.y
    total_y = electron_y + hole_y + positive_ion_y + displacement_y
    fields = {
        "electron current": electron_y,
        "hole current": hole_y,
        "positive-ion current": positive_ion_y,
        "displacement current": displacement_y,
        "total current": total_y,
    }
    if not all(np.all(np.isfinite(values)) for values in fields.values()):
        raise ValueError("mobile-ion current decomposition produced non-finite values")

    x = np.asarray(snapshot.x, dtype=float)
    average_electron = _lateral_average(electron_y, x)
    average_hole = _lateral_average(hole_y, x)
    average_ion = _lateral_average(positive_ion_y, x)
    average_displacement = _lateral_average(displacement_y, x)
    average_total = _lateral_average(total_y, x)
    terminal_index = -1
    terminal_components = (
        float(average_electron[terminal_index]),
        float(average_hole[terminal_index]),
        float(average_ion[terminal_index]),
        float(average_displacement[terminal_index]),
    )
    terminal_total = float(average_total[terminal_index])
    decomposition_total = sum(terminal_components)
    scale = max(abs(terminal_total), abs(decomposition_total), 1.0e-30)
    if abs(terminal_total - decomposition_total) > 32.0 * np.finfo(float).eps * scale:
        raise RuntimeError("mobile-ion terminal-current decomposition is inconsistent")
    spread = float(np.ptp(average_total))
    face_scale = max(float(np.max(np.abs(average_total))), 1.0e-30)

    return MobileIonCurrentComponents2D(
        electron_y_A_m2=_read_only(electron_y),
        hole_y_A_m2=_read_only(hole_y),
        positive_ion_y_A_m2=_read_only(positive_ion_y),
        displacement_y_A_m2=_read_only(displacement_y),
        total_y_A_m2=_read_only(total_y),
        lateral_average_electron_A_m2=_read_only(average_electron),
        lateral_average_hole_A_m2=_read_only(average_hole),
        lateral_average_positive_ion_A_m2=_read_only(average_ion),
        lateral_average_displacement_A_m2=_read_only(average_displacement),
        lateral_average_total_A_m2=_read_only(average_total),
        terminal_electron_A_m2=terminal_components[0],
        terminal_hole_A_m2=terminal_components[1],
        terminal_positive_ion_A_m2=terminal_components[2],
        terminal_displacement_A_m2=terminal_components[3],
        terminal_total_A_m2=terminal_total,
        max_face_spread_A_m2=spread,
        max_relative_face_spread=spread / face_scale,
        applied_voltage_rate_V_s=voltage_rate,
    )


__all__ = [
    "MobileIonCurrentComponents2D",
    "evaluate_mobile_ion_current_components_2d",
]

"""Concrete production bindings for the ion-free classical 1D DC system.

Adapters consume the equations-layer compiled object and provide the residual,
analytic sparse linearization, physical row scaling, and coordinate limits.
No G2 qualification or unsupported time/ion capability is inferred from them.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.solvers.nonlinear.problem import ResidualScaling, StepControl

__all__ = [
    "HISTORICAL_MAX_ROW_V1", "POISSON_CONTACT_BALANCE_V1", "production_convergence_norm",
    "production_coordinates", "production_residual_scaling",
    "production_steady_problem", "production_steady_linearization",
]

HISTORICAL_MAX_ROW_V1 = "historical_max_row_v1"
POISSON_CONTACT_BALANCE_V1 = "poisson_contact_balance_v1"


def production_convergence_norm(
    device: CompiledDevice1D, *, convergence_policy: str = HISTORICAL_MAX_ROW_V1,
) -> Callable[[np.ndarray], float] | None:
    """Optional convergence norm of the unchanged production-scaled rows.

    Poisson and fixed-reservoir log rows retain their individual limits.
    Other carrier rows use q*sum(abs(balance))/current_tol, obtained by
    dividing their existing 2*N*q/current_tol scaling by 2*N. Finite-rate
    and blocking boundary cells are included. The Newton merit is untouched.
    """
    if convergence_policy == HISTORICAL_MAX_ROW_V1:
        return None
    if convergence_policy != POISSON_CONTACT_BALANCE_V1:
        raise ValueError("unknown production convergence policy")
    count = device.node_count
    fixed = np.zeros(3 * count, dtype=bool)
    fixed[:count] = True
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                fixed[offset + node] = True

    def norm(scaled_residual: np.ndarray) -> float:
        values = np.asarray(scaled_residual, dtype=float)
        if values.shape != (3 * count,):
            raise ValueError("production convergence rows must match the compiled device")
        if not np.all(np.isfinite(values)):
            return math.inf
        absolute = np.abs(values)
        carrier_rows = absolute[count:].copy()
        carrier_rows[fixed[count:]] = 0.0
        balance = (
            float(np.sum(carrier_rows[:count])) + float(np.sum(carrier_rows[count:]))
        ) / (2 * count)
        return max(float(np.max(absolute[fixed])), balance)

    return norm


def poisson_row_weights(device: CompiledDevice1D) -> np.ndarray:
    """Inverse Gauss scale q N_ref w + (C_left + C_right) V_ref."""
    count = device.node_count
    voltage_ref = max(abs(device.built_in_potential_V), device.thermal_voltage_V)
    density_ref = np.maximum(
        np.maximum(device.parameters.acceptor_density_m3[1:-1], device.parameters.donor_density_m3[1:-1]),
        1.0,
    )
    capacitance = device.poisson.face_capacitance_F_m2
    scale = (
        ELEMENTARY_CHARGE_C * density_ref * device.poisson.interior_widths_m
        + (capacitance[:-1] + capacitance[1:]) * voltage_ref
    )
    weights = np.full(count, 1.0 / voltage_ref)
    weights[1:-1] = 1.0 / scale
    return weights


def production_steady_problem(
    device: CompiledDevice1D, *, voltage_V: float = 0.0, generation_scale: float = 0.0
) -> DC1DProblem:
    return DC1DProblem(device, voltage_V, generation_scale)


def production_steady_linearization(
    problem: DC1DProblem, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0
) -> csc_matrix:
    return problem.jacobian(u, udot, alpha=alpha)


def production_residual_scaling(
    device: CompiledDevice1D,
    *,
    poisson_tolerance: float = 1.0e-11,
    continuity_tolerance_A_m2: float = 1.0e-8,
    contact_log_tolerance: float = 1.0e-11,
) -> ResidualScaling:
    """Physical bound per row; use a Newton tolerance of 1.

    Bounding each carrier row by current_tol/(2 N q) bounds the sum of
    their absolute balances independently of density or mesh clustering.
    """
    for value in (poisson_tolerance, continuity_tolerance_A_m2, contact_log_tolerance):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("physical residual tolerances must be finite and positive")
    count = device.node_count
    weights = np.full(3 * count, 2 * count * ELEMENTARY_CHARGE_C / continuity_tolerance_A_m2)
    weights[:count] = poisson_row_weights(device) / poisson_tolerance
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                weights[offset + node] = 1.0 / contact_log_tolerance
    return ResidualScaling(weights)


def production_coordinates(
    device: CompiledDevice1D, *, potential_step_V: float = 0.1, quasi_fermi_step: float = 2.0
) -> StepControl:
    """Whole-direction step limits in phi and dimensionless quasi-Fermi offsets."""
    if not all(math.isfinite(value) and value > 0.0 for value in (potential_step_V, quasi_fermi_step)):
        raise ValueError("coordinate step limits must be finite and positive")
    return StepControl(np.r_[
        np.full(device.node_count, potential_step_V),
        np.full(2 * device.node_count, quasi_fermi_step),
    ])

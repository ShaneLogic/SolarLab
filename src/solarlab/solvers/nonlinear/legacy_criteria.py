"""Historical steady-state acceptance quantities reproduced bit for bit (M class).

* :func:`legacy_steady_state_tolerances` -- the effective residual and stall
  bounds of ``experiments.steady_state.solve_steady_state``: the configured
  peak-density-relative rate tolerances, each capped by the continuity-current
  limit ``max_error / (q n_ref W)``.
* :func:`legacy_continuity_current_bound` -- the reported certificate
  ``q n_ref W * residual`` [A/m^2].
* :func:`legacy_normalized_poisson_residual` -- the dimensionless Poisson
  residual of the dark degenerate and bulk-trap pn equilibrium lanes.

The acceptance labels are those of the historical result objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from solarlab.contracts.units import ELEMENTARY_CHARGE_C

__all__ = [
    "CURRENT_BOUNDED_STALL", "LEGACY_ACCEPTANCE_LABELS", "LegacySteadyStateCriteriaError",
    "LegacySteadyStateTolerances", "RESIDUAL_CONVERGED", "legacy_continuity_current_bound",
    "legacy_normalized_poisson_residual", "legacy_steady_state_tolerances",
]

Q = ELEMENTARY_CHARGE_C
RESIDUAL_CONVERGED = "residual_converged"
CURRENT_BOUNDED_STALL = "current_bounded_stall"
LEGACY_ACCEPTANCE_LABELS = (RESIDUAL_CONVERGED, CURRENT_BOUNDED_STALL)


class LegacySteadyStateCriteriaError(RuntimeError):
    """The historical continuity-current scale cannot be formed."""


@dataclass(frozen=True, slots=True)
class LegacySteadyStateTolerances:
    """Effective bounds of the historical steady-state Newton."""

    continuity_current_scale_A_m2: float
    current_limited_tolerance: float
    residual_tolerance: float
    stall_acceptance_tolerance: float


def legacy_steady_state_tolerances(
    *,
    tol: float,
    tol_accept: float,
    max_continuity_current_error: float,
    n_ref: float,
    device_width_m: float,
) -> LegacySteadyStateTolerances:
    """``tol``/``tol_accept`` capped by ``max_error / (q n_ref W)``."""
    if (not np.isfinite(max_continuity_current_error)
            or max_continuity_current_error <= 0.0):
        raise ValueError(
            "max_continuity_current_error must be finite and positive, got "
            f"{max_continuity_current_error}"
        )
    continuity_current_scale = Q * n_ref * device_width_m
    if not np.isfinite(continuity_current_scale) or continuity_current_scale <= 0.0:
        raise LegacySteadyStateCriteriaError(
            "cannot construct a finite continuity-current convergence scale"
        )
    current_limited_tol = (
        max_continuity_current_error / continuity_current_scale
    )
    return LegacySteadyStateTolerances(
        continuity_current_scale_A_m2=continuity_current_scale,
        current_limited_tolerance=current_limited_tol,
        residual_tolerance=min(tol, current_limited_tol),
        stall_acceptance_tolerance=min(tol_accept, current_limited_tol),
    )


def legacy_continuity_current_bound(tolerances: LegacySteadyStateTolerances,
                                    residual: float) -> float:
    """Reported continuity-current certificate ``q n_ref W * residual`` [A/m^2]."""
    return float(tolerances.continuity_current_scale_A_m2 * residual)


def legacy_normalized_poisson_residual(
    residual: ArrayLike,
    *,
    face_capacitance_F_m2: ArrayLike,
    interior_widths_m: ArrayLike,
    acceptor_density_m3: ArrayLike,
    donor_density_m3: ArrayLike,
    V_bi_bc: float,
    thermal_voltage_V: float,
    trap_density_m3: float | None = None,
) -> float:
    """``max |r| / (q N_ref h + (C_L + C_R) V_ref)`` over the interior nodes.

    ``N_ref = max(N_A, N_D, 1)`` (and the bulk-trap density when given) and
    ``V_ref = max(|V_bi_bc|, V_T)``.
    """
    C = np.asarray(face_capacitance_F_m2)
    N_A = np.asarray(acceptor_density_m3)
    N_D = np.asarray(donor_density_m3)
    if trap_density_m3 is None:
        density_reference = np.maximum(
            np.maximum(N_A[1:-1], N_D[1:-1]),
            1.0,
        )
    else:
        density_reference = np.maximum.reduce((
            np.asarray(N_A[1:-1], dtype=float),
            np.asarray(N_D[1:-1], dtype=float),
            np.full(N_A.size - 2, float(trap_density_m3)),
            np.ones(N_A.size - 2),
        ))
    voltage_reference = max(abs(float(V_bi_bc)), thermal_voltage_V)
    scale = (
        Q * density_reference * np.asarray(interior_widths_m)
        + (C[:-1] + C[1:]) * voltage_reference
    )
    return float(np.max(np.abs(np.asarray(residual)) / scale))

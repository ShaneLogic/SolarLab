"""Historical damped-Newton loops reproduced bit for bit (M class).

``legacy_damped_fd_newton`` is ``experiments.steady_state._damped_newton``:
a forward-difference Jacobian, a 4-step ridge fallback for a singular matrix,
a componentwise clip of the step to +-5 and an Armijo backtracking search on
the 2-norm (12 halvings). It returns ``(z, max|F|)`` and never certifies
convergence; the caller decides.

``legacy_newton_stage`` is ``experiments.quasi_fermi_steady_state.
_solve_newton_stage``: the same structure with a fixed finite-difference step,
an optional equilibrated least-squares step for rank-deficient systems, a +-5
clip, 30 halvings and the near-tolerance stagnation exit of intermediate
continuation stages. The residual callback takes ``(z, stage_value)``; the
historical system object and its edge-coordinate variant are the caller's
concern. Failures raise :class:`LegacyNewtonStageError` with the historical
messages.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "LEGACY_FD_EPS", "LEGACY_LN_STEP_CAP", "LegacyNewtonStageError",
    "legacy_damped_fd_newton", "legacy_newton_stage",
]

LEGACY_FD_EPS = 1.0e-7        # relative FD perturbation of _damped_newton
LEGACY_LN_STEP_CAP = 5.0      # componentwise step clip of both loops


class LegacyNewtonStageError(RuntimeError):
    """A historical Newton stage failed (same messages as the legacy error)."""


def legacy_damped_fd_newton(
    F: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    z0: ArrayLike,
    n_unk: int,
    *,
    max_it: int,
    tol: float,
    tol_step: float = 1e-9,
) -> tuple[NDArray[np.float64], float]:
    """Historical compact damped FD-Jacobian Newton; returns ``(z, res)``."""
    z = np.array(z0, dtype=float, copy=True)
    f = F(z)
    res = float(np.max(np.abs(f)))
    for _ in range(max_it):
        if res < tol:
            break
        J = np.empty((n_unk, n_unk))
        for j in range(n_unk):
            dz = LEGACY_FD_EPS * max(1.0, abs(z[j]))
            zj = z.copy()
            zj[j] += dz
            J[:, j] = (F(zj) - f) / dz
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            lam_r = 1e-10 * float(np.max(np.abs(np.diag(J))) or 1.0)
            step = None
            for _r in range(4):
                try:
                    step = np.linalg.solve(J + lam_r * np.eye(n_unk), -f)
                    break
                except np.linalg.LinAlgError:
                    lam_r *= 1e3
            if step is None:
                break
        if float(np.max(np.abs(step))) < tol_step:
            break
        step = np.clip(step, -LEGACY_LN_STEP_CAP, LEGACY_LN_STEP_CAP)
        nrm = float(np.linalg.norm(f))
        lam, accepted = 1.0, False
        for _ls in range(12):
            f_try = F(z + lam * step)
            if float(np.linalg.norm(f_try)) < nrm * (1.0 - 1.0e-4 * lam):
                z = z + lam * step
                f = f_try
                res = float(np.max(np.abs(f)))
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
    return z, res


def legacy_newton_stage(
    evaluate: Callable[[NDArray[np.float64], float], ArrayLike],
    z0: ArrayLike,
    stage_value: float,
    *,
    finite_difference_step: float,
    residual_tolerance: float,
    max_iterations: int,
    rank_deficient: bool = False,
    near_tolerance_stagnation_factor: float = 1.0,
) -> tuple[NDArray[np.float64], int]:
    """Historical QF Newton stage; returns ``(z, iterations)`` or raises."""
    z = np.asarray(z0, dtype=float).copy()
    size = z.size

    for iteration in range(max_iterations + 1):
        residual = np.asarray(evaluate(z, stage_value))
        max_residual = float(np.max(np.abs(residual)))
        if max_residual < residual_tolerance:
            return z, iteration
        if iteration == max_iterations:
            break

        jacobian = np.empty((size, size), dtype=float)
        for column in range(size):
            trial = z.copy()
            trial[column] += finite_difference_step
            jacobian[:, column] = (
                np.asarray(evaluate(trial, stage_value)) - residual
            ) / finite_difference_step
        if rank_deficient:
            column_scale = np.max(np.abs(jacobian), axis=0)
            active = column_scale > 1.0e-30
            if not np.any(active):
                raise LegacyNewtonStageError(
                    "QF interface Jacobian has no active transport columns"
                )
            equilibrated = jacobian[:, active] / column_scale[active]
            scaled_step, *_ = np.linalg.lstsq(
                equilibrated,
                -residual,
                rcond=1.0e-14,
            )
            step = np.zeros(size, dtype=float)
            step[active] = scaled_step / column_scale[active]
        else:
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError as exc:
                raise LegacyNewtonStageError(
                    "singular QF Newton Jacobian at "
                    f"illumination={stage_value:g}"
                ) from exc
        step = np.clip(step, -LEGACY_LN_STEP_CAP, LEGACY_LN_STEP_CAP)
        norm = float(np.linalg.norm(residual))
        for line_search_iteration in range(30):
            damping = 0.5**line_search_iteration
            candidate = z + damping * step
            try:
                candidate_norm = float(
                    np.linalg.norm(np.asarray(evaluate(candidate, stage_value)))
                )
            except (RuntimeError, ValueError):
                if rank_deficient:
                    continue
                raise
            if candidate_norm < norm * (1.0 - 1.0e-4 * damping):
                z = candidate
                break
        else:
            if (
                near_tolerance_stagnation_factor > 1.0
                and max_residual
                <= near_tolerance_stagnation_factor * residual_tolerance
            ):
                return z, iteration
            raise LegacyNewtonStageError(
                "QF Newton line search failed at "
                f"illumination={stage_value:g}, "
                f"max normalized residual={max_residual:.6g}"
            )
    raise LegacyNewtonStageError(
        "QF Newton iteration limit reached at "
        f"illumination={stage_value:g}, "
        f"max normalized residual={max_residual:.6g}"
    )

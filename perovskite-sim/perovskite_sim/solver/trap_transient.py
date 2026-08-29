"""Certified local integration of one monovalent trap occupancy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:
    from perovskite_sim._compat.scipy_shim import solve_ivp

from perovskite_sim.physics.trap_kinetics import (
    TrapReservoirKinetics,
    TrapReservoirState,
)
from perovskite_sim.physics.trap_transient import (
    ConstantReservoirTrapTrace,
    TrapTransientError,
    constant_reservoir_trap_trace,
    evaluate_trap_transient,
    linearize_trap_transient,
    occupancy_from_logit,
    occupancy_logit,
)


class LocalTrapTransientCertificationError(RuntimeError):
    """The local implicit solve or its invariant checks failed closed."""


@dataclass(frozen=True, slots=True)
class LocalTrapTransientPolicy:
    """Numerical and certification policy for the isolated local solve."""

    rtol: float = 1.0e-8
    atol_logit: float = 1.0e-10
    max_step_s: float | None = None
    max_closed_form_absolute_error: float = 1.0e-7
    max_charge_balance_relative_error: float = 1.0e-12
    method: str = "Radau"

    def __post_init__(self) -> None:
        for name in (
            "rtol",
            "atol_logit",
            "max_closed_form_absolute_error",
            "max_charge_balance_relative_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.max_step_s is not None:
            step = float(self.max_step_s)
            if not math.isfinite(step) or step <= 0.0:
                raise ValueError("max_step_s must be None or finite and positive")
            object.__setattr__(self, "max_step_s", step)
        method = str(self.method).strip()
        if method not in {"Radau", "BDF"}:
            raise ValueError("method must be 'Radau' or 'BDF'")
        object.__setattr__(self, "method", method)


@dataclass(frozen=True, slots=True, eq=False)
class LocalTrapTransientResult:
    """Immutable physical trace and certification evidence."""

    times_s: np.ndarray
    occupancy: np.ndarray
    logit_coordinate: np.ndarray
    occupancy_rate_s1: np.ndarray
    trap_charge_C: np.ndarray
    trap_charge_rate_C_s: np.ndarray
    carrier_charge_rate_C_s: np.ndarray
    charge_balance_residual_C_s: np.ndarray
    exact_trace: ConstantReservoirTrapTrace
    maximum_closed_form_absolute_error: float
    maximum_charge_balance_relative_error: float
    nfev: int
    njev: int
    nlu: int
    analytic_jacobian_used: bool
    state_coordinate: str
    certified: bool

    def __post_init__(self) -> None:
        times = np.array(self.times_s, dtype=float, copy=True)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise TrapTransientError(
                "times_s must be finite, one-dimensional, and strictly increasing"
            )
        times.setflags(write=False)
        object.__setattr__(self, "times_s", times)
        for name in (
            "occupancy",
            "logit_coordinate",
            "occupancy_rate_s1",
            "trap_charge_C",
            "trap_charge_rate_C_s",
            "carrier_charge_rate_C_s",
            "charge_balance_residual_C_s",
        ):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != times.shape or not np.all(np.isfinite(values)):
                raise TrapTransientError(f"{name} must be finite and match times_s")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if np.any((self.occupancy <= 0.0) | (self.occupancy >= 1.0)):
            raise TrapTransientError("occupancy must remain strictly inside (0, 1)")
        if not isinstance(self.exact_trace, ConstantReservoirTrapTrace):
            raise TypeError("exact_trace must be a ConstantReservoirTrapTrace")
        for name in (
            "maximum_closed_form_absolute_error",
            "maximum_charge_balance_relative_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TrapTransientError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in ("nfev", "njev", "nlu"):
            value = int(getattr(self, name))
            if value < 0:
                raise TrapTransientError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.analytic_jacobian_used, (bool, np.bool_)):
            raise TypeError("analytic_jacobian_used must be boolean")
        if not bool(self.analytic_jacobian_used):
            raise TrapTransientError("local certified solve requires an analytic Jacobian")
        object.__setattr__(self, "analytic_jacobian_used", True)
        if self.state_coordinate != "logit":
            raise TrapTransientError("local dynamic occupancy must use logit coordinates")
        if not isinstance(self.certified, (bool, np.bool_)):
            raise TypeError("certified must be boolean")
        object.__setattr__(self, "certified", bool(self.certified))


def solve_local_trap_transient(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    initial_occupancy: object,
    times_s: object,
    *,
    charge_transition: str,
    policy: LocalTrapTransientPolicy | None = None,
    require_certified: bool = True,
) -> LocalTrapTransientResult:
    """Integrate one constant-reservoir trap in logit coordinates.

    This D6-E0 slice is intentionally local.  The exact constant-reservoir
    solution is retained as an independent oracle; carrier, Poisson, and
    terminal-current coupling belong to subsequent device checkpoints.
    """
    if not isinstance(kinetics, TrapReservoirKinetics):
        raise TypeError("kinetics must be TrapReservoirKinetics")
    if not isinstance(state, TrapReservoirState):
        raise TypeError("state must be TrapReservoirState")
    resolved_policy = policy or LocalTrapTransientPolicy()
    if not isinstance(resolved_policy, LocalTrapTransientPolicy):
        raise TypeError("policy must be a LocalTrapTransientPolicy or None")
    if not isinstance(require_certified, (bool, np.bool_)):
        raise TypeError("require_certified must be boolean")
    exact = constant_reservoir_trap_trace(
        kinetics,
        state,
        initial_occupancy,
        times_s,
        charge_transition=charge_transition,
    )
    initial_logit = occupancy_logit(initial_occupancy)

    def rhs(_time: float, coordinate: np.ndarray) -> np.ndarray:
        occupancy = occupancy_from_logit(coordinate[0])
        evaluation = evaluate_trap_transient(
            kinetics,
            state,
            occupancy,
            charge_transition=charge_transition,
        )
        return np.array([evaluation.logit_rate_s1])

    def jacobian(_time: float, coordinate: np.ndarray) -> np.ndarray:
        occupancy = occupancy_from_logit(coordinate[0])
        tangent = linearize_trap_transient(
            kinetics,
            state,
            occupancy,
            charge_transition=charge_transition,
        )
        return np.array([[tangent.logit_rate_logit_derivative_s1]])

    maximum_step = (
        np.inf
        if resolved_policy.max_step_s is None
        else resolved_policy.max_step_s
    )
    try:
        solution = solve_ivp(
            rhs,
            (float(exact.times_s[0]), float(exact.times_s[-1])),
            np.array([initial_logit]),
            t_eval=exact.times_s,
            method=resolved_policy.method,
            rtol=resolved_policy.rtol,
            atol=resolved_policy.atol_logit,
            max_step=maximum_step,
            jac=jacobian,
            dense_output=False,
        )
    except (FloatingPointError, OverflowError, TrapTransientError) as exc:
        raise LocalTrapTransientCertificationError(
            f"local logit integration failed closed: {exc}"
        ) from exc
    if not bool(getattr(solution, "success", False)):
        raise LocalTrapTransientCertificationError(
            "local logit integration did not converge: "
            f"{getattr(solution, 'message', 'unknown solver failure')}"
        )
    coordinates = np.asarray(solution.y, dtype=float)
    if coordinates.shape != (1, exact.times_s.size) or not np.all(
        np.isfinite(coordinates)
    ):
        raise LocalTrapTransientCertificationError(
            "local logit integration returned an invalid trace"
        )
    occupancy = np.array(
        [occupancy_from_logit(value) for value in coordinates[0]],
        dtype=float,
    )
    evaluations = tuple(
        evaluate_trap_transient(
            kinetics,
            state,
            value,
            charge_transition=charge_transition,
        )
        for value in occupancy
    )
    occupancy_rate = np.array([value.occupancy_rate_s1 for value in evaluations])
    trap_charge = np.array([value.trap_charge_C for value in evaluations])
    trap_charge_rate = np.array(
        [value.trap_charge_rate_C_s for value in evaluations]
    )
    carrier_charge_rate = np.array(
        [value.carrier_charge_rate_C_s for value in evaluations]
    )
    residual = np.array(
        [value.charge_balance_residual_C_s for value in evaluations]
    )
    closed_form_error = float(np.max(np.abs(occupancy - exact.occupancy)))
    charge_error = max(
        value.charge_balance_relative_error for value in evaluations
    )
    certified = bool(
        closed_form_error <= resolved_policy.max_closed_form_absolute_error
        and charge_error <= resolved_policy.max_charge_balance_relative_error
    )
    result = LocalTrapTransientResult(
        times_s=exact.times_s,
        occupancy=occupancy,
        logit_coordinate=coordinates[0],
        occupancy_rate_s1=occupancy_rate,
        trap_charge_C=trap_charge,
        trap_charge_rate_C_s=trap_charge_rate,
        carrier_charge_rate_C_s=carrier_charge_rate,
        charge_balance_residual_C_s=residual,
        exact_trace=exact,
        maximum_closed_form_absolute_error=closed_form_error,
        maximum_charge_balance_relative_error=charge_error,
        nfev=int(getattr(solution, "nfev", 0)),
        njev=int(getattr(solution, "njev", 0)),
        nlu=int(getattr(solution, "nlu", 0)),
        analytic_jacobian_used=True,
        state_coordinate="logit",
        certified=certified,
    )
    if require_certified and not certified:
        raise LocalTrapTransientCertificationError(
            "local trap transient did not certify: "
            f"closed_form_error={closed_form_error:.6g} "
            f"(limit={resolved_policy.max_closed_form_absolute_error:.6g}), "
            f"charge_balance_error={charge_error:.6g} "
            f"(limit={resolved_policy.max_charge_balance_relative_error:.6g})"
        )
    return result


__all__ = [
    "LocalTrapTransientCertificationError",
    "LocalTrapTransientPolicy",
    "LocalTrapTransientResult",
    "solve_local_trap_transient",
]

"""Local monovalent trap dynamics in a bounded occupancy coordinate.

This module is deliberately device-solver agnostic.  It evaluates the same
electron/hole capture convention used by :mod:`trap_kinetics`, exposes exact
analytic tangents, and supplies the constant-reservoir relaxation trace used
as an independent time-domain oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import ACCEPTOR, DONOR
from perovskite_sim.physics.trap_kinetics import (
    TrapReservoirKinetics,
    TrapReservoirState,
    evaluate_trap_dc_operating_point,
)


DYNAMIC_TRAP_TRANSIENT_VERSION = "monovalent-local-trap-transient-v1"
SUPPORTED_DYNAMIC_TRANSITIONS = frozenset({ACCEPTOR, DONOR})


class TrapTransientError(ValueError):
    """A local dynamic-trap state or trace is not physically admissible."""


def _readonly(value: object, *, dtype: object = float) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_open_occupancy(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("occupancy must be a real number")
    try:
        occupancy = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("occupancy must be a real number") from exc
    if not math.isfinite(occupancy) or not 0.0 < occupancy < 1.0:
        raise TrapTransientError("dynamic occupancy must lie strictly inside (0, 1)")
    return occupancy


def _transition(value: object) -> str:
    transition = str(value).strip().lower()
    if transition not in SUPPORTED_DYNAMIC_TRANSITIONS:
        raise TrapTransientError(
            "dynamic charge closure requires an acceptor or donor transition"
        )
    return transition


def occupancy_logit(occupancy: object) -> float:
    """Map one strictly interior occupancy to its unbounded coordinate."""
    value = _finite_open_occupancy(occupancy)
    return math.log(value) - math.log1p(-value)


def occupancy_from_logit(coordinate: object) -> float:
    """Map one finite logit coordinate without clipping or endpoint repair."""
    if isinstance(coordinate, (bool, np.bool_)):
        raise TypeError("logit coordinate must be a real number")
    try:
        value = float(coordinate)
    except (TypeError, ValueError) as exc:
        raise TypeError("logit coordinate must be a real number") from exc
    if not math.isfinite(value):
        raise TrapTransientError("logit coordinate must be finite")
    if value >= 0.0:
        occupancy = 1.0 / (1.0 + math.exp(-value))
    else:
        exponential = math.exp(value)
        occupancy = exponential / (1.0 + exponential)
    if not 0.0 < occupancy < 1.0:
        raise TrapTransientError(
            "logit transform saturated outside resolvable occupancy bounds"
        )
    return occupancy


@dataclass(frozen=True, slots=True, eq=False)
class TrapTransientEvaluation:
    """Non-equilibrium capture, storage, and charge rates for one trap."""

    kinetics_sha256: str
    state_sha256: str
    charge_transition: str
    occupancy: float
    quasi_steady_occupancy: float
    filled_rate_s1: float
    empty_rate_s1: float
    relaxation_rate_s1: float
    occupancy_rate_s1: float
    logit_rate_s1: float
    electron_capture_rates_s1: np.ndarray
    hole_capture_rates_s1: np.ndarray
    trap_charge_C: float
    trap_charge_rate_C_s: float
    carrier_charge_rate_C_s: float
    charge_balance_residual_C_s: float
    charge_balance_relative_error: float
    version: str = DYNAMIC_TRAP_TRANSIENT_VERSION

    def __post_init__(self) -> None:
        for name in ("kinetics_sha256", "state_sha256"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise TrapTransientError(f"{name} must be a lowercase SHA-256")
        object.__setattr__(self, "charge_transition", _transition(self.charge_transition))
        object.__setattr__(self, "occupancy", _finite_open_occupancy(self.occupancy))
        object.__setattr__(
            self,
            "quasi_steady_occupancy",
            float(self.quasi_steady_occupancy),
        )
        for name in (
            "filled_rate_s1",
            "empty_rate_s1",
            "relaxation_rate_s1",
            "occupancy_rate_s1",
            "logit_rate_s1",
            "trap_charge_C",
            "trap_charge_rate_C_s",
            "carrier_charge_rate_C_s",
            "charge_balance_residual_C_s",
            "charge_balance_relative_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise TrapTransientError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.quasi_steady_occupancy <= 1.0:
            raise TrapTransientError("quasi_steady_occupancy must lie in [0, 1]")
        if self.filled_rate_s1 < 0.0 or self.empty_rate_s1 < 0.0:
            raise TrapTransientError("filled and empty rates must be non-negative")
        if self.relaxation_rate_s1 <= 0.0:
            raise TrapTransientError("relaxation_rate_s1 must be positive")
        if self.charge_balance_relative_error < 0.0:
            raise TrapTransientError("charge balance error must be non-negative")
        for name in ("electron_capture_rates_s1", "hole_capture_rates_s1"):
            values = _readonly(getattr(self, name))
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise TrapTransientError(f"{name} must be a non-empty finite vector")
            object.__setattr__(self, name, values)
        tolerance = 64.0 * np.finfo(float).eps
        if not math.isclose(
            self.relaxation_rate_s1,
            self.filled_rate_s1 + self.empty_rate_s1,
            rel_tol=tolerance,
        ):
            raise TrapTransientError("relaxation rate must equal filled plus empty")
        if not math.isclose(
            self.quasi_steady_occupancy,
            self.filled_rate_s1 / self.relaxation_rate_s1,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapTransientError("quasi-steady occupancy is inconsistent")
        capture_rate = float(
            np.sum(self.electron_capture_rates_s1)
            - np.sum(self.hole_capture_rates_s1)
        )
        scale = max(
            abs(capture_rate),
            abs(self.occupancy_rate_s1),
            self.relaxation_rate_s1,
        )
        if abs(capture_rate - self.occupancy_rate_s1) > tolerance * scale:
            raise TrapTransientError("occupancy rate must equal capture imbalance")
        expected_logit_rate = self.occupancy_rate_s1 / (
            self.occupancy * (1.0 - self.occupancy)
        )
        if not math.isclose(
            self.logit_rate_s1,
            expected_logit_rate,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapTransientError("logit rate is inconsistent with occupancy rate")
        expected_charge = (
            -Q * self.occupancy
            if self.charge_transition == ACCEPTOR
            else Q * (1.0 - self.occupancy)
        )
        if not math.isclose(
            self.trap_charge_C,
            expected_charge,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapTransientError("trap charge is inconsistent with transition")
        if not math.isclose(
            self.trap_charge_rate_C_s,
            -Q * self.occupancy_rate_s1,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapTransientError("trap charge rate is inconsistent")
        if not math.isclose(
            self.carrier_charge_rate_C_s,
            Q * self.occupancy_rate_s1,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapTransientError("carrier charge rate is inconsistent")
        if self.version != DYNAMIC_TRAP_TRANSIENT_VERSION:
            raise TrapTransientError("unsupported dynamic trap transient version")


@dataclass(frozen=True, slots=True, eq=False)
class TrapTransientTangent:
    """Exact local derivatives at one non-equilibrium logit state."""

    evaluation: TrapTransientEvaluation
    occupancy_rate_occupancy_derivative_s1: float
    occupancy_rate_electron_density_derivative_m3_s: np.ndarray
    occupancy_rate_hole_density_derivative_m3_s: np.ndarray
    logit_rate_logit_derivative_s1: float
    logit_rate_electron_density_derivative_m3_s: np.ndarray
    logit_rate_hole_density_derivative_m3_s: np.ndarray
    electron_capture_occupancy_derivative_s1: np.ndarray
    hole_capture_occupancy_derivative_s1: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, TrapTransientEvaluation):
            raise TypeError("evaluation must be a TrapTransientEvaluation")
        for name in (
            "occupancy_rate_occupancy_derivative_s1",
            "logit_rate_logit_derivative_s1",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise TrapTransientError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "occupancy_rate_electron_density_derivative_m3_s",
            "occupancy_rate_hole_density_derivative_m3_s",
            "logit_rate_electron_density_derivative_m3_s",
            "logit_rate_hole_density_derivative_m3_s",
            "electron_capture_occupancy_derivative_s1",
            "hole_capture_occupancy_derivative_s1",
        ):
            values = _readonly(getattr(self, name))
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise TrapTransientError(f"{name} must be a non-empty finite vector")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True, eq=False)
class ConstantReservoirTrapTrace:
    """Exact constant-reservoir occupancy and charge-conservation trace."""

    kinetics_sha256: str
    state_sha256: str
    charge_transition: str
    times_s: np.ndarray
    occupancy: np.ndarray
    logit_coordinate: np.ndarray
    occupancy_rate_s1: np.ndarray
    trap_charge_C: np.ndarray
    trap_charge_rate_C_s: np.ndarray
    carrier_charge_rate_C_s: np.ndarray
    charge_balance_residual_C_s: np.ndarray
    maximum_charge_balance_relative_error: float
    quasi_steady_occupancy: float
    relaxation_rate_s1: float

    def __post_init__(self) -> None:
        for name in ("kinetics_sha256", "state_sha256"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise TrapTransientError(f"{name} must be a lowercase SHA-256")
        object.__setattr__(self, "charge_transition", _transition(self.charge_transition))
        times = _readonly(self.times_s)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise TrapTransientError(
                "times_s must be finite, one-dimensional, and strictly increasing"
            )
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
            values = _readonly(getattr(self, name))
            if values.shape != times.shape or not np.all(np.isfinite(values)):
                raise TrapTransientError(f"{name} must be finite and match times_s")
            object.__setattr__(self, name, values)
        if np.any((self.occupancy <= 0.0) | (self.occupancy >= 1.0)):
            raise TrapTransientError("trace occupancy must remain strictly inside (0, 1)")
        for name in (
            "maximum_charge_balance_relative_error",
            "quasi_steady_occupancy",
            "relaxation_rate_s1",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise TrapTransientError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.maximum_charge_balance_relative_error < 0.0:
            raise TrapTransientError("charge balance error must be non-negative")
        if self.relaxation_rate_s1 <= 0.0:
            raise TrapTransientError("relaxation rate must be positive")


def evaluate_trap_transient(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    occupancy: object,
    *,
    charge_transition: str,
) -> TrapTransientEvaluation:
    """Evaluate exact non-equilibrium capture and local charge conservation."""
    if not isinstance(kinetics, TrapReservoirKinetics):
        raise TypeError("kinetics must be TrapReservoirKinetics")
    if not isinstance(state, TrapReservoirState):
        raise TypeError("state must be TrapReservoirState")
    transition = _transition(charge_transition)
    value = _finite_open_occupancy(occupancy)
    steady = evaluate_trap_dc_operating_point(kinetics, state)
    c_n = kinetics.electron_capture_coefficients_m3_s
    c_p = kinetics.hole_capture_coefficients_m3_s
    n = state.electron_densities_m3
    p = state.hole_densities_m3
    n1 = kinetics.electron_reference_densities_m3
    p1 = kinetics.hole_reference_densities_m3
    electron = c_n * (n * (1.0 - value) - n1 * value)
    hole = c_p * (p * value - p1 * (1.0 - value))
    occupancy_rate = float(np.sum(electron) - np.sum(hole))
    logit_rate = occupancy_rate / (value * (1.0 - value))
    trap_charge = -Q * value if transition == ACCEPTOR else Q * (1.0 - value)
    trap_charge_rate = -Q * occupancy_rate
    carrier_charge_rate = Q * occupancy_rate
    residual = trap_charge_rate + carrier_charge_rate
    scale = abs(trap_charge_rate) + abs(carrier_charge_rate)
    relative = abs(residual) / scale if scale > 0.0 else 0.0
    return TrapTransientEvaluation(
        kinetics_sha256=kinetics.sha256,
        state_sha256=state.sha256,
        charge_transition=transition,
        occupancy=value,
        quasi_steady_occupancy=steady.occupancy,
        filled_rate_s1=steady.filled_rate_s1,
        empty_rate_s1=steady.empty_rate_s1,
        relaxation_rate_s1=steady.relaxation_rate_s1,
        occupancy_rate_s1=occupancy_rate,
        logit_rate_s1=logit_rate,
        electron_capture_rates_s1=electron,
        hole_capture_rates_s1=hole,
        trap_charge_C=trap_charge,
        trap_charge_rate_C_s=trap_charge_rate,
        carrier_charge_rate_C_s=carrier_charge_rate,
        charge_balance_residual_C_s=residual,
        charge_balance_relative_error=relative,
    )


def linearize_trap_transient(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    occupancy: object,
    *,
    charge_transition: str,
) -> TrapTransientTangent:
    """Return exact derivatives for occupancy and logit residuals."""
    evaluation = evaluate_trap_transient(
        kinetics,
        state,
        occupancy,
        charge_transition=charge_transition,
    )
    value = evaluation.occupancy
    c_n = kinetics.electron_capture_coefficients_m3_s
    c_p = kinetics.hole_capture_coefficients_m3_s
    n = state.electron_densities_m3
    p = state.hole_densities_m3
    n1 = kinetics.electron_reference_densities_m3
    p1 = kinetics.hole_reference_densities_m3
    filled = evaluation.filled_rate_s1
    empty = evaluation.empty_rate_s1
    return TrapTransientTangent(
        evaluation=evaluation,
        occupancy_rate_occupancy_derivative_s1=-evaluation.relaxation_rate_s1,
        occupancy_rate_electron_density_derivative_m3_s=c_n * (1.0 - value),
        occupancy_rate_hole_density_derivative_m3_s=-c_p * value,
        logit_rate_logit_derivative_s1=(
            -filled * (1.0 - value) / value
            - empty * value / (1.0 - value)
        ),
        logit_rate_electron_density_derivative_m3_s=c_n / value,
        logit_rate_hole_density_derivative_m3_s=-c_p / (1.0 - value),
        electron_capture_occupancy_derivative_s1=-c_n * (n + n1),
        hole_capture_occupancy_derivative_s1=c_p * (p + p1),
    )


def constant_reservoir_trap_trace(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    initial_occupancy: object,
    times_s: object,
    *,
    charge_transition: str,
) -> ConstantReservoirTrapTrace:
    """Return the exact isolated-trap relaxation for constant reservoirs."""
    transition = _transition(charge_transition)
    initial = _finite_open_occupancy(initial_occupancy)
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise TrapTransientError(
            "times_s must be finite, one-dimensional, and strictly increasing"
        )
    steady = evaluate_trap_dc_operating_point(kinetics, state)
    elapsed = times - times[0]
    occupancy = steady.occupancy + (initial - steady.occupancy) * np.exp(
        -steady.relaxation_rate_s1 * elapsed
    )
    occupancy[0] = initial
    if not np.all(np.isfinite(occupancy)) or np.any(
        (occupancy <= 0.0) | (occupancy >= 1.0)
    ):
        raise TrapTransientError(
            "constant-reservoir trace reached an unresolvable occupancy endpoint"
        )
    evaluations = tuple(
        evaluate_trap_transient(
            kinetics,
            state,
            value,
            charge_transition=transition,
        )
        for value in occupancy
    )
    logit = np.array([occupancy_logit(value) for value in occupancy])
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
    return ConstantReservoirTrapTrace(
        kinetics_sha256=kinetics.sha256,
        state_sha256=state.sha256,
        charge_transition=transition,
        times_s=times,
        occupancy=occupancy,
        logit_coordinate=logit,
        occupancy_rate_s1=occupancy_rate,
        trap_charge_C=trap_charge,
        trap_charge_rate_C_s=trap_charge_rate,
        carrier_charge_rate_C_s=carrier_charge_rate,
        charge_balance_residual_C_s=residual,
        maximum_charge_balance_relative_error=max(
            value.charge_balance_relative_error for value in evaluations
        ),
        quasi_steady_occupancy=steady.occupancy,
        relaxation_rate_s1=steady.relaxation_rate_s1,
    )


__all__ = [
    "DYNAMIC_TRAP_TRANSIENT_VERSION",
    "SUPPORTED_DYNAMIC_TRANSITIONS",
    "ConstantReservoirTrapTrace",
    "TrapTransientError",
    "TrapTransientEvaluation",
    "TrapTransientTangent",
    "constant_reservoir_trap_trace",
    "evaluate_trap_transient",
    "linearize_trap_transient",
    "occupancy_from_logit",
    "occupancy_logit",
]

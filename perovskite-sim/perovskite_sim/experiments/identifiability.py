"""Fail-closed synthetic identifiability analysis for interface SRH inputs.

This module deliberately separates a valid identifiability analysis from a
claim that every fitted parameter is identifiable.  The built-in benchmark
uses the production interface-SRH kinetic identity and the research-only
equilibrium-referenced sheet-charge primitive.  It therefore exposes the exact
``N_t * sigma * calibration_factor`` confounding already present in the device
schema instead of hiding it behind a well-conditioned toy polynomial.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import least_squares

from perovskite_sim.constants import Q
from perovskite_sim.physics.interface_plane import (
    equilibrium_referenced_interface_trap_charge,
)
from perovskite_sim.physics.recombination import interface_recombination


ParameterName = Literal[
    "trap_density_cm2",
    "capture_cross_section_scale",
    "calibration_factor",
]
ObservableFamily = Literal[
    "recombination_only",
    "recombination_plus_charge",
]

PARAMETER_NAMES: tuple[ParameterName, ...] = (
    "trap_density_cm2",
    "capture_cross_section_scale",
    "calibration_factor",
)
OBSERVABLE_FAMILIES: tuple[ObservableFamily, ...] = (
    "recombination_only",
    "recombination_plus_charge",
)


class IdentifiabilityError(ValueError):
    """Base class for invalid protocols or evidence."""


class IdentifiabilityForwardError(IdentifiabilityError):
    """Raised when the frozen forward model cannot be evaluated."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


def _positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer >= {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer >= {minimum}") from exc
    if integer != value or integer < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return integer


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identifiability document contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "identifiability document contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _exact_keys(payload: Mapping[str, Any], cls: type, label: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise IdentifiabilityError(
            f"{label} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class IdentifiabilityParameter:
    """One bounded base-10 logarithmic synthetic parameter."""

    name: ParameterName
    lower_log10: float
    upper_log10: float
    truth_log10: float
    estimated: bool = True

    def __post_init__(self) -> None:
        if self.name not in PARAMETER_NAMES:
            raise ValueError(f"unsupported identifiability parameter {self.name!r}")
        lower = _finite(self.lower_log10, "lower_log10")
        upper = _finite(self.upper_log10, "upper_log10")
        truth = _finite(self.truth_log10, "truth_log10")
        if not lower < upper:
            raise ValueError("parameter lower_log10 must be below upper_log10")
        if not lower < truth < upper:
            raise ValueError("parameter truth_log10 must lie strictly inside bounds")
        if not isinstance(self.estimated, bool):
            raise TypeError("parameter estimated flag must be boolean")
        object.__setattr__(self, "lower_log10", lower)
        object.__setattr__(self, "upper_log10", upper)
        object.__setattr__(self, "truth_log10", truth)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("identifiability parameter must be a mapping")
        _exact_keys(payload, cls, "identifiability parameter")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class InterfaceCarrierCondition:
    """One positive carrier-density pair sampled by the synthetic model."""

    electron_density_m3: float
    hole_density_m3: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electron_density_m3",
            _positive(self.electron_density_m3, "electron_density_m3"),
        )
        object.__setattr__(
            self,
            "hole_density_m3",
            _positive(self.hole_density_m3, "hole_density_m3"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("interface carrier condition must be a mapping")
        _exact_keys(payload, cls, "interface carrier condition")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class InterfaceSRHIdentifiabilityProtocol:
    """Frozen synthetic recovery and local-identifiability contract."""

    parameters: tuple[IdentifiabilityParameter, ...]
    carrier_conditions: tuple[InterfaceCarrierCondition, ...]
    intrinsic_density_squared_m6: float
    trap_electron_density_m3: float
    trap_hole_density_m3: float
    base_sigma_n_cm2: float
    base_sigma_p_cm2: float
    thermal_velocity_cm_s: float
    equilibrium_occupancy: float
    observable_family: ObservableFamily
    rate_relative_standard_deviation: float
    charge_relative_standard_deviation: float
    standard_deviation_floor_fraction: float
    synthetic_noise_sigma_multiplier: float
    noise_seed: int
    finite_difference_step_log10: float
    svd_relative_threshold: float
    expected_rank: int
    condition_number_limit: float
    truth_recovery_tolerance_log10: float
    multistart_fractions: tuple[tuple[float, ...], ...]
    profile_grid_count: int
    least_squares_max_nfev: int
    failure_standardized_residual: float
    schema_version: Literal[1] = 1
    model: Literal["production_interface_srh_scaling"] = (
        "production_interface_srh_scaling"
    )
    model_version: Literal[1] = 1
    coordinate_system: Literal["base10_log_parameters"] = "base10_log_parameters"
    optimizer: Literal["scipy_least_squares_trf"] = "scipy_least_squares_trf"
    profile_method: Literal["fixed_parameter_reoptimization"] = (
        "fixed_parameter_reoptimization"
    )
    forward_failure_policy: Literal["penalize_and_invalidate"] = (
        "penalize_and_invalidate"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.model_version != 1:
            raise ValueError("identifiability schema/model version must be 1")
        if self.model != "production_interface_srh_scaling":
            raise ValueError("unsupported identifiability forward model")
        if self.coordinate_system != "base10_log_parameters":
            raise ValueError("identifiability parameters must use base-10 logs")
        if self.optimizer != "scipy_least_squares_trf":
            raise ValueError("unsupported identifiability optimizer")
        if self.profile_method != "fixed_parameter_reoptimization":
            raise ValueError("unsupported profile method")
        if self.forward_failure_policy != "penalize_and_invalidate":
            raise ValueError(
                "forward failures must be penalized and invalidate evidence"
            )
        if self.observable_family not in OBSERVABLE_FAMILIES:
            raise ValueError("unsupported observable family")

        parameters = tuple(self.parameters)
        if tuple(parameter.name for parameter in parameters) != PARAMETER_NAMES:
            raise ValueError(
                "parameters must contain the canonical trap/capture/calibration order"
            )
        if not any(parameter.estimated for parameter in parameters):
            raise ValueError("at least one identifiability parameter must be estimated")
        object.__setattr__(self, "parameters", parameters)

        conditions = tuple(self.carrier_conditions)
        if len(conditions) < 3 or not all(
            isinstance(item, InterfaceCarrierCondition) for item in conditions
        ):
            raise ValueError("at least three carrier conditions are required")
        condition_pairs = {
            (item.electron_density_m3.hex(), item.hole_density_m3.hex())
            for item in conditions
        }
        if len(condition_pairs) != len(conditions):
            raise ValueError("carrier conditions must be unique")
        object.__setattr__(self, "carrier_conditions", conditions)

        for name in (
            "intrinsic_density_squared_m6",
            "trap_electron_density_m3",
            "trap_hole_density_m3",
            "base_sigma_n_cm2",
            "base_sigma_p_cm2",
            "thermal_velocity_cm_s",
            "rate_relative_standard_deviation",
            "charge_relative_standard_deviation",
            "standard_deviation_floor_fraction",
            "finite_difference_step_log10",
            "svd_relative_threshold",
            "condition_number_limit",
            "truth_recovery_tolerance_log10",
            "failure_standardized_residual",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        occupancy = _finite(self.equilibrium_occupancy, "equilibrium_occupancy")
        if not 0.0 <= occupancy <= 1.0:
            raise ValueError("equilibrium_occupancy must lie in [0, 1]")
        object.__setattr__(self, "equilibrium_occupancy", occupancy)
        noise = _nonnegative(
            self.synthetic_noise_sigma_multiplier,
            "synthetic_noise_sigma_multiplier",
        )
        object.__setattr__(self, "synthetic_noise_sigma_multiplier", noise)
        if self.standard_deviation_floor_fraction >= 1.0:
            raise ValueError("standard_deviation_floor_fraction must be below 1")
        if self.svd_relative_threshold >= 1.0:
            raise ValueError("svd_relative_threshold must be below 1")

        seed = _integer(self.noise_seed, "noise_seed", minimum=0)
        object.__setattr__(self, "noise_seed", seed)
        estimated_count = len(self.estimated_parameters)
        rank = _integer(self.expected_rank, "expected_rank", minimum=1)
        if rank > estimated_count:
            raise ValueError("expected_rank exceeds estimated parameter count")
        object.__setattr__(self, "expected_rank", rank)
        profile_count = _integer(
            self.profile_grid_count,
            "profile_grid_count",
            minimum=5,
        )
        if profile_count % 2 == 0:
            raise ValueError("profile_grid_count must be odd")
        object.__setattr__(self, "profile_grid_count", profile_count)
        object.__setattr__(
            self,
            "least_squares_max_nfev",
            _integer(
                self.least_squares_max_nfev,
                "least_squares_max_nfev",
                minimum=10,
            ),
        )

        starts = tuple(
            tuple(float(value) for value in row) for row in self.multistart_fractions
        )
        if len(starts) < 3 or any(len(row) != estimated_count for row in starts):
            raise ValueError(
                "multistart_fractions require at least three dimension-matched rows"
            )
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for row in starts
            for value in row
        ):
            raise ValueError("multistart fractions must lie strictly inside (0, 1)")
        if len(set(starts)) != len(starts):
            raise ValueError("multistart fractions must be unique")
        object.__setattr__(self, "multistart_fractions", starts)

    @property
    def estimated_parameters(self) -> tuple[IdentifiabilityParameter, ...]:
        return tuple(parameter for parameter in self.parameters if parameter.estimated)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    def canonical_json(self) -> str:
        return _canonical_json(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("identifiability protocol must be a mapping")
        _exact_keys(payload, cls, "identifiability protocol")
        values = dict(payload)
        raw_parameters = values["parameters"]
        raw_conditions = values["carrier_conditions"]
        if not isinstance(raw_parameters, Sequence) or isinstance(
            raw_parameters, (str, bytes)
        ):
            raise TypeError("protocol parameters must be a sequence")
        if not isinstance(raw_conditions, Sequence) or isinstance(
            raw_conditions, (str, bytes)
        ):
            raise TypeError("protocol carrier_conditions must be a sequence")
        values["parameters"] = tuple(
            IdentifiabilityParameter.from_dict(item) for item in raw_parameters
        )
        values["carrier_conditions"] = tuple(
            InterfaceCarrierCondition.from_dict(item) for item in raw_conditions
        )
        values["multistart_fractions"] = tuple(
            tuple(row) for row in values["multistart_fractions"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class IdentifiabilityFitAttempt:
    """One bounded least-squares attempt from one declared start."""

    start_log10: tuple[float, ...]
    solution_log10: tuple[float, ...]
    chi_square: float
    standardized_residual_linf: float
    success: bool
    status: int
    function_evaluations: int

    def __post_init__(self) -> None:
        start = tuple(_finite(value, "start_log10") for value in self.start_log10)
        solution = tuple(
            _finite(value, "solution_log10") for value in self.solution_log10
        )
        if not start or len(start) != len(solution):
            raise ValueError("fit attempt coordinates must be nonempty and aligned")
        object.__setattr__(self, "start_log10", start)
        object.__setattr__(self, "solution_log10", solution)
        object.__setattr__(
            self, "chi_square", _nonnegative(self.chi_square, "chi_square")
        )
        object.__setattr__(
            self,
            "standardized_residual_linf",
            _nonnegative(
                self.standardized_residual_linf,
                "standardized_residual_linf",
            ),
        )
        if not isinstance(self.success, bool):
            raise TypeError("fit attempt success must be boolean")
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(
            self,
            "function_evaluations",
            _integer(self.function_evaluations, "function_evaluations", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class IdentifiabilityProfile:
    """Fixed-parameter profile with all remaining parameters re-optimized."""

    parameter_name: ParameterName
    parameter_values_log10: tuple[float, ...]
    chi_square: tuple[float, ...]
    successful: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.parameter_name not in PARAMETER_NAMES:
            raise ValueError("profile parameter is not supported")
        values = tuple(
            _finite(value, "parameter_values_log10")
            for value in self.parameter_values_log10
        )
        costs = tuple(
            _nonnegative(value, "profile chi_square") for value in self.chi_square
        )
        flags = tuple(self.successful)
        if not values or len(values) != len(costs) or len(values) != len(flags):
            raise ValueError("profile arrays must be nonempty and aligned")
        if any(not isinstance(flag, bool) for flag in flags):
            raise TypeError("profile successful values must be boolean")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("profile parameter values must be strictly increasing")
        object.__setattr__(self, "parameter_values_log10", values)
        object.__setattr__(self, "chi_square", costs)
        object.__setattr__(self, "successful", flags)


def _result_unsigned_document(
    result: "InterfaceSRHIdentifiabilityResult",
) -> dict[str, Any]:
    return {
        field.name: _json_ready(getattr(result, field.name))
        for field in dataclasses.fields(result)
        if field.name != "mapping_sha256"
    }


@dataclass(frozen=True, slots=True)
class InterfaceSRHIdentifiabilityResult:
    """Immutable synthetic fit, Fisher/SVD, nullspace, and profile evidence."""

    protocol: InterfaceSRHIdentifiabilityProtocol
    protocol_sha256: str
    observable_labels: tuple[str, ...]
    observable_units: tuple[str, ...]
    observed_values: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    truth_values: tuple[float, ...]
    best_fit_log10: tuple[float, ...]
    best_fit_values: tuple[float, ...]
    best_chi_square: float
    best_standardized_residual_linf: float
    weighted_jacobian: tuple[tuple[float, ...], ...]
    fisher_information: tuple[tuple[float, ...], ...]
    fisher_correlation: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    numerical_rank: int
    condition_number: float | None
    nullspace_vectors: tuple[tuple[float, ...], ...]
    fit_attempts: tuple[IdentifiabilityFitAttempt, ...]
    profiles: tuple[IdentifiabilityProfile, ...]
    forward_failure_count: int
    rank_expectation_met: bool
    truth_recovered: bool
    all_multistarts_converged: bool
    profiles_completed: bool
    parameters_identifiable: bool
    analysis_certified: bool
    mapping_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, InterfaceSRHIdentifiabilityProtocol):
            raise TypeError("protocol must be InterfaceSRHIdentifiabilityProtocol")
        protocol_hash = _sha256(self.protocol_sha256, "protocol_sha256")
        if protocol_hash != self.protocol.sha256:
            raise ValueError("protocol_sha256 does not match protocol")
        object.__setattr__(self, "protocol_sha256", protocol_hash)

        labels = tuple(self.observable_labels)
        units = tuple(self.observable_units)
        observed = tuple(
            _finite(value, "observed_values") for value in self.observed_values
        )
        deviations = tuple(
            _positive(value, "standard_deviations")
            for value in self.standard_deviations
        )
        if (
            not labels
            or len(set(labels)) != len(labels)
            or len(labels) != len(units)
            or len(labels) != len(observed)
            or len(labels) != len(deviations)
        ):
            raise ValueError("observable evidence arrays must be unique and aligned")
        object.__setattr__(self, "observable_labels", labels)
        object.__setattr__(self, "observable_units", units)
        object.__setattr__(self, "observed_values", observed)
        object.__setattr__(self, "standard_deviations", deviations)

        estimated = self.protocol.estimated_parameters
        count = len(estimated)
        truth_values = tuple(
            _positive(value, "truth_values") for value in self.truth_values
        )
        best_log = tuple(
            _finite(value, "best_fit_log10") for value in self.best_fit_log10
        )
        best_values = tuple(
            _positive(value, "best_fit_values") for value in self.best_fit_values
        )
        if (
            len(truth_values) != count
            or len(best_log) != count
            or len(best_values) != count
        ):
            raise ValueError(
                "parameter result arrays do not match estimated parameters"
            )
        if any(
            not math.isclose(value, 10.0**coordinate, rel_tol=2.0e-15)
            for value, coordinate in zip(best_values, best_log, strict=True)
        ):
            raise ValueError("best_fit_values do not match best_fit_log10")
        object.__setattr__(self, "truth_values", truth_values)
        object.__setattr__(self, "best_fit_log10", best_log)
        object.__setattr__(self, "best_fit_values", best_values)
        object.__setattr__(
            self,
            "best_chi_square",
            _nonnegative(self.best_chi_square, "best_chi_square"),
        )
        object.__setattr__(
            self,
            "best_standardized_residual_linf",
            _nonnegative(
                self.best_standardized_residual_linf,
                "best_standardized_residual_linf",
            ),
        )

        jacobian = _finite_matrix(
            self.weighted_jacobian,
            rows=len(labels),
            columns=count,
            field="weighted_jacobian",
        )
        fisher = _finite_matrix(
            self.fisher_information,
            rows=count,
            columns=count,
            field="fisher_information",
        )
        correlation = _finite_matrix(
            self.fisher_correlation,
            rows=count,
            columns=count,
            field="fisher_correlation",
        )
        object.__setattr__(self, "weighted_jacobian", jacobian)
        object.__setattr__(self, "fisher_information", fisher)
        object.__setattr__(self, "fisher_correlation", correlation)
        singular = tuple(
            _nonnegative(value, "singular_values") for value in self.singular_values
        )
        if len(singular) != count or any(
            right > left for left, right in zip(singular, singular[1:], strict=False)
        ):
            raise ValueError("singular_values must be descending and dimension matched")
        object.__setattr__(self, "singular_values", singular)
        rank = _integer(self.numerical_rank, "numerical_rank", minimum=1)
        if rank > count:
            raise ValueError("numerical_rank exceeds parameter count")
        object.__setattr__(self, "numerical_rank", rank)
        if self.condition_number is not None:
            condition = _positive(self.condition_number, "condition_number")
            if rank != count:
                raise ValueError("rank-deficient result must use condition_number=None")
            object.__setattr__(self, "condition_number", condition)
        elif rank == count:
            raise ValueError("full-rank result requires a finite condition number")
        nullspace = _finite_matrix(
            self.nullspace_vectors,
            rows=count - rank,
            columns=count,
            field="nullspace_vectors",
        )
        object.__setattr__(self, "nullspace_vectors", nullspace)

        attempts = tuple(self.fit_attempts)
        profiles = tuple(self.profiles)
        if len(attempts) != len(self.protocol.multistart_fractions) or not all(
            isinstance(item, IdentifiabilityFitAttempt) for item in attempts
        ):
            raise ValueError("fit attempts do not match protocol starts")
        if tuple(item.parameter_name for item in profiles) != tuple(
            parameter.name for parameter in estimated
        ):
            raise ValueError("profiles do not match estimated parameter order")
        object.__setattr__(self, "fit_attempts", attempts)
        object.__setattr__(self, "profiles", profiles)
        failures = _integer(
            self.forward_failure_count,
            "forward_failure_count",
            minimum=0,
        )
        object.__setattr__(self, "forward_failure_count", failures)

        expected_rank_match = rank == self.protocol.expected_rank
        full_rank = rank == count
        expected_flags = {
            "rank_expectation_met": expected_rank_match,
            "all_multistarts_converged": all(item.success for item in attempts),
            "profiles_completed": all(all(item.successful) for item in profiles),
            "parameters_identifiable": (
                full_rank
                and self.condition_number is not None
                and self.condition_number <= self.protocol.condition_number_limit
            ),
        }
        for name, expected in expected_flags.items():
            actual = getattr(self, name)
            if not isinstance(actual, bool) or actual != expected:
                raise ValueError(f"{name} does not match identifiability evidence")
        truth_error = max(
            abs(coordinate - parameter.truth_log10)
            for coordinate, parameter in zip(best_log, estimated, strict=True)
        )
        expected_recovery = (
            expected_flags["parameters_identifiable"]
            and truth_error <= self.protocol.truth_recovery_tolerance_log10
        )
        if (
            not isinstance(self.truth_recovered, bool)
            or self.truth_recovered != expected_recovery
        ):
            raise ValueError("truth_recovered does not match fit evidence")
        expected_certified = (
            expected_rank_match
            and expected_flags["all_multistarts_converged"]
            and expected_flags["profiles_completed"]
            and failures == 0
            and (expected_recovery if full_rank else True)
        )
        if (
            not isinstance(self.analysis_certified, bool)
            or self.analysis_certified != expected_certified
        ):
            raise ValueError("analysis_certified does not match evidence")

        mapping = _sha256(self.mapping_sha256, "mapping_sha256")
        expected_mapping = hashlib.sha256(
            _canonical_json(_result_unsigned_document(self)).encode("ascii")
        ).hexdigest()
        if mapping != expected_mapping:
            raise ValueError("mapping_sha256 does not match result evidence")
        object.__setattr__(self, "mapping_sha256", mapping)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


def _finite_matrix(
    values: Sequence[Sequence[float]],
    *,
    rows: int,
    columns: int,
    field: str,
) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(_finite(value, field) for value in row) for row in values)
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise ValueError(f"{field} must have shape ({rows}, {columns})")
    return matrix


def _default_multistarts(dimension: int) -> tuple[tuple[float, ...], ...]:
    corners = tuple(
        tuple(0.2 if bit == 0 else 0.8 for bit in bits)
        for bits in itertools.product((0, 1), repeat=dimension)
    )
    center = (0.5,) * dimension
    return (center, *corners)


def _expected_structural_rank(
    estimated_names: tuple[ParameterName, ...],
    observable_family: ObservableFamily,
) -> int:
    rate_row = np.ones((1, len(estimated_names)), dtype=float)
    rows = [rate_row]
    if observable_family == "recombination_plus_charge":
        charge_row = np.array(
            [[1.0 if name == "trap_density_cm2" else 0.0 for name in estimated_names]],
            dtype=float,
        )
        if np.any(charge_row):
            rows.append(charge_row)
    return int(np.linalg.matrix_rank(np.vstack(rows)))


def build_interface_srh_identifiability_protocol(
    *,
    observable_family: ObservableFamily = "recombination_plus_charge",
    estimated_parameters: Sequence[ParameterName] = PARAMETER_NAMES,
    carrier_condition_count: int = 7,
    finite_difference_step_log10: float = 1.0e-4,
    synthetic_noise_sigma_multiplier: float = 0.0,
    noise_seed: int = 1729,
) -> InterfaceSRHIdentifiabilityProtocol:
    """Build the canonical production-formula synthetic benchmark."""

    if observable_family not in OBSERVABLE_FAMILIES:
        raise ValueError("unsupported observable family")
    requested = tuple(estimated_parameters)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("estimated_parameters must be nonempty and unique")
    if any(name not in PARAMETER_NAMES for name in requested):
        raise ValueError("estimated_parameters contains an unsupported name")
    count = _integer(carrier_condition_count, "carrier_condition_count", minimum=3)
    electron_axis = np.geomspace(1.0e16, 1.0e22, count)
    hole_axis = electron_axis[::-1]
    conditions = tuple(
        InterfaceCarrierCondition(float(n), float(p))
        for n, p in zip(electron_axis, hole_axis, strict=True)
    )
    parameters = (
        IdentifiabilityParameter(
            "trap_density_cm2",
            lower_log10=8.0,
            upper_log10=14.0,
            truth_log10=12.0,
            estimated="trap_density_cm2" in requested,
        ),
        IdentifiabilityParameter(
            "capture_cross_section_scale",
            lower_log10=-2.0,
            upper_log10=2.0,
            truth_log10=0.0,
            estimated="capture_cross_section_scale" in requested,
        ),
        IdentifiabilityParameter(
            "calibration_factor",
            lower_log10=-6.0,
            upper_log10=0.0,
            truth_log10=-3.0,
            estimated="calibration_factor" in requested,
        ),
    )
    estimated_order = tuple(
        parameter.name for parameter in parameters if parameter.estimated
    )
    expected_rank = _expected_structural_rank(estimated_order, observable_family)
    return InterfaceSRHIdentifiabilityProtocol(
        parameters=parameters,
        carrier_conditions=conditions,
        intrinsic_density_squared_m6=1.0e32,
        trap_electron_density_m3=1.0e16,
        trap_hole_density_m3=1.0e16,
        base_sigma_n_cm2=1.0e-15,
        base_sigma_p_cm2=3.0e-16,
        thermal_velocity_cm_s=1.0e7,
        equilibrium_occupancy=0.5,
        observable_family=observable_family,
        rate_relative_standard_deviation=0.02,
        charge_relative_standard_deviation=0.02,
        standard_deviation_floor_fraction=1.0e-6,
        synthetic_noise_sigma_multiplier=synthetic_noise_sigma_multiplier,
        noise_seed=noise_seed,
        finite_difference_step_log10=finite_difference_step_log10,
        svd_relative_threshold=1.0e-7,
        expected_rank=expected_rank,
        condition_number_limit=1.0e8,
        truth_recovery_tolerance_log10=5.0e-5,
        multistart_fractions=_default_multistarts(len(estimated_order)),
        profile_grid_count=9,
        least_squares_max_nfev=400,
        failure_standardized_residual=1.0e6,
    )


def _full_parameter_log_values(
    protocol: InterfaceSRHIdentifiabilityProtocol,
    estimated_log_values: np.ndarray,
) -> dict[ParameterName, float]:
    values: dict[ParameterName, float] = {}
    cursor = 0
    for parameter in protocol.parameters:
        if parameter.estimated:
            values[parameter.name] = float(estimated_log_values[cursor])
            cursor += 1
        else:
            values[parameter.name] = parameter.truth_log10
    if cursor != len(estimated_log_values):
        raise ValueError("estimated coordinate vector has the wrong size")
    return values


def _predict_interface_srh_observables(
    protocol: InterfaceSRHIdentifiabilityProtocol,
    estimated_log_values: np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    log_values = _full_parameter_log_values(protocol, estimated_log_values)
    trap_density_cm2 = 10.0 ** log_values["trap_density_cm2"]
    capture_scale = 10.0 ** log_values["capture_cross_section_scale"]
    calibration = 10.0 ** log_values["calibration_factor"]
    common = protocol.thermal_velocity_cm_s * trap_density_cm2 * 1.0e-2
    velocity_n = protocol.base_sigma_n_cm2 * capture_scale * common * calibration
    velocity_p = protocol.base_sigma_p_cm2 * capture_scale * common * calibration
    if not (
        math.isfinite(velocity_n)
        and math.isfinite(velocity_p)
        and velocity_n > 0.0
        and velocity_p > 0.0
    ):
        raise IdentifiabilityForwardError("synthetic capture velocities are invalid")

    labels: list[str] = []
    units: list[str] = []
    values: list[float] = []
    occupancies: list[float] = []
    for index, condition in enumerate(protocol.carrier_conditions):
        n = condition.electron_density_m3
        p = condition.hole_density_m3
        rate = interface_recombination(
            n,
            p,
            protocol.intrinsic_density_squared_m6,
            protocol.trap_electron_density_m3,
            protocol.trap_hole_density_m3,
            velocity_n,
            velocity_p,
        )
        current = Q * float(rate)
        denominator = velocity_n * (n + protocol.trap_electron_density_m3) + (
            velocity_p * (p + protocol.trap_hole_density_m3)
        )
        occupancy = (
            velocity_n * n + velocity_p * protocol.trap_hole_density_m3
        ) / denominator
        if not (
            math.isfinite(current)
            and current > 0.0
            and math.isfinite(occupancy)
            and 0.0 <= occupancy <= 1.0
        ):
            raise IdentifiabilityForwardError(
                "production interface-SRH formula left its valid envelope"
            )
        labels.append(f"interface_current_condition_{index}_A_m2")
        units.append("A m-2")
        values.append(current)
        occupancies.append(occupancy)

    if protocol.observable_family == "recombination_plus_charge":
        trap_density_m2 = trap_density_cm2 * 1.0e4
        charge = equilibrium_referenced_interface_trap_charge(
            np.asarray(occupancies),
            protocol.equilibrium_occupancy,
            trap_density_m2,
        )
        for index, value in enumerate(np.asarray(charge, dtype=float)):
            labels.append(f"incremental_sheet_charge_condition_{index}_C_m2")
            units.append("C m-2")
            values.append(float(value))
    predicted = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(predicted)):
        raise IdentifiabilityForwardError("synthetic observables are non-finite")
    return tuple(labels), tuple(units), predicted


def _block_standard_deviations(
    protocol: InterfaceSRHIdentifiabilityProtocol,
    truth: np.ndarray,
) -> np.ndarray:
    count = len(protocol.carrier_conditions)
    standard_deviations = np.empty_like(truth)
    blocks = [
        (slice(0, count), protocol.rate_relative_standard_deviation),
    ]
    if protocol.observable_family == "recombination_plus_charge":
        blocks.append(
            (slice(count, 2 * count), protocol.charge_relative_standard_deviation)
        )
    for block, relative in blocks:
        values = np.abs(truth[block])
        scale = float(np.max(values))
        if not math.isfinite(scale) or scale <= 0.0:
            raise IdentifiabilityForwardError("observable block has no finite scale")
        floor = protocol.standard_deviation_floor_fraction * scale
        standard_deviations[block] = relative * np.maximum(values, floor)
    return standard_deviations


class _ResidualEvaluator:
    def __init__(
        self,
        protocol: InterfaceSRHIdentifiabilityProtocol,
        observed: np.ndarray,
        standard_deviations: np.ndarray,
    ) -> None:
        self.protocol = protocol
        self.observed = observed
        self.standard_deviations = standard_deviations
        self.forward_failure_count = 0

    def residual(self, coordinates: np.ndarray) -> np.ndarray:
        try:
            _labels, _units, predicted = _predict_interface_srh_observables(
                self.protocol,
                np.asarray(coordinates, dtype=float),
            )
        except Exception:
            self.forward_failure_count += 1
            return np.full(
                self.observed.shape,
                self.protocol.failure_standardized_residual,
                dtype=float,
            )
        return (predicted - self.observed) / self.standard_deviations

    def jacobian(
        self,
        coordinates: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        jacobian = np.empty((self.observed.size, values.size), dtype=float)
        step = self.protocol.finite_difference_step_log10
        base = self.residual(values)
        for column in range(values.size):
            left = values.copy()
            right = values.copy()
            left[column] = max(lower[column], values[column] - step)
            right[column] = min(upper[column], values[column] + step)
            if left[column] < values[column] and right[column] > values[column]:
                jacobian[:, column] = (self.residual(right) - self.residual(left)) / (
                    right[column] - left[column]
                )
            elif right[column] > values[column]:
                jacobian[:, column] = (self.residual(right) - base) / (
                    right[column] - values[column]
                )
            elif left[column] < values[column]:
                jacobian[:, column] = (base - self.residual(left)) / (
                    values[column] - left[column]
                )
            else:
                raise IdentifiabilityForwardError(
                    "finite-difference coordinate is pinned at both bounds"
                )
        return jacobian


def _fit(
    evaluator: _ResidualEvaluator,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> IdentifiabilityFitAttempt:
    result = least_squares(
        evaluator.residual,
        start,
        jac=lambda values: evaluator.jacobian(values, lower, upper),
        bounds=(lower, upper),
        method="trf",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=evaluator.protocol.least_squares_max_nfev,
    )
    residual = evaluator.residual(result.x)
    return IdentifiabilityFitAttempt(
        start_log10=tuple(float(value) for value in start),
        solution_log10=tuple(float(value) for value in result.x),
        chi_square=float(np.dot(residual, residual)),
        standardized_residual_linf=float(np.max(np.abs(residual))),
        success=bool(result.success and np.all(np.isfinite(result.x))),
        status=int(result.status),
        function_evaluations=int(result.nfev),
    )


def _profile_parameter(
    protocol: InterfaceSRHIdentifiabilityProtocol,
    evaluator: _ResidualEvaluator,
    best: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    index: int,
) -> IdentifiabilityProfile:
    grid = np.linspace(lower[index], upper[index], protocol.profile_grid_count)
    remaining = np.array(
        [item for item in range(best.size) if item != index], dtype=int
    )
    costs: list[float] = []
    successes: list[bool] = []
    for fixed in grid:
        if remaining.size == 0:
            candidate = np.array([fixed], dtype=float)
            residual = evaluator.residual(candidate)
            costs.append(float(np.dot(residual, residual)))
            successes.append(bool(np.all(np.isfinite(residual))))
            continue

        reduced_lower = lower[remaining]
        reduced_upper = upper[remaining]
        reduced_start = np.clip(best[remaining], reduced_lower, reduced_upper)

        def expand(reduced: np.ndarray) -> np.ndarray:
            candidate = best.copy()
            candidate[index] = fixed
            candidate[remaining] = reduced
            return candidate

        def residual(reduced: np.ndarray) -> np.ndarray:
            return evaluator.residual(expand(reduced))

        def jacobian(reduced: np.ndarray) -> np.ndarray:
            expanded = expand(reduced)
            full = evaluator.jacobian(expanded, lower, upper)
            return full[:, remaining]

        result = least_squares(
            residual,
            reduced_start,
            jac=jacobian,
            bounds=(reduced_lower, reduced_upper),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=protocol.least_squares_max_nfev,
        )
        profile_residual = residual(result.x)
        costs.append(float(np.dot(profile_residual, profile_residual)))
        successes.append(bool(result.success and np.all(np.isfinite(result.x))))
    return IdentifiabilityProfile(
        parameter_name=protocol.estimated_parameters[index].name,
        parameter_values_log10=tuple(float(value) for value in grid),
        chi_square=tuple(costs),
        successful=tuple(successes),
    )


def _canonical_nullspace(vectors: np.ndarray) -> tuple[tuple[float, ...], ...]:
    canonical: list[tuple[float, ...]] = []
    for vector in vectors:
        values = np.asarray(vector, dtype=float).copy()
        pivot = int(np.argmax(np.abs(values)))
        if values[pivot] < 0.0:
            values *= -1.0
        canonical.append(tuple(float(value) for value in values))
    return tuple(canonical)


def _correlation_from_fisher(fisher: np.ndarray) -> np.ndarray:
    covariance = np.linalg.pinv(fisher, rcond=1.0e-15, hermitian=True)
    diagonal = np.maximum(np.diag(covariance), 0.0)
    scale = np.sqrt(diagonal)
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def run_interface_srh_identifiability(
    protocol: InterfaceSRHIdentifiabilityProtocol,
) -> InterfaceSRHIdentifiabilityResult:
    """Run the frozen synthetic multi-start, Fisher/SVD, and profile analysis."""

    if not isinstance(protocol, InterfaceSRHIdentifiabilityProtocol):
        raise TypeError("protocol must be InterfaceSRHIdentifiabilityProtocol")
    estimated = protocol.estimated_parameters
    truth_log = np.asarray([item.truth_log10 for item in estimated], dtype=float)
    labels, units, truth_prediction = _predict_interface_srh_observables(
        protocol,
        truth_log,
    )
    standard_deviations = _block_standard_deviations(protocol, truth_prediction)
    rng = np.random.default_rng(protocol.noise_seed)
    observed = truth_prediction + (
        protocol.synthetic_noise_sigma_multiplier
        * standard_deviations
        * rng.standard_normal(truth_prediction.shape)
    )
    evaluator = _ResidualEvaluator(protocol, observed, standard_deviations)
    lower = np.asarray([item.lower_log10 for item in estimated], dtype=float)
    upper = np.asarray([item.upper_log10 for item in estimated], dtype=float)
    starts = tuple(
        lower + np.asarray(fractions, dtype=float) * (upper - lower)
        for fractions in protocol.multistart_fractions
    )
    attempts = tuple(_fit(evaluator, start, lower, upper) for start in starts)
    best_attempt = min(
        attempts,
        key=lambda item: (item.chi_square, item.standardized_residual_linf),
    )
    best = np.asarray(best_attempt.solution_log10, dtype=float)
    weighted_jacobian = evaluator.jacobian(best, lower, upper)
    _u, singular_values, vh = np.linalg.svd(weighted_jacobian, full_matrices=True)
    threshold = protocol.svd_relative_threshold * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank < 1:
        raise IdentifiabilityForwardError("weighted Jacobian has zero numerical rank")
    fisher = weighted_jacobian.T @ weighted_jacobian
    correlation = _correlation_from_fisher(fisher)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if rank == len(estimated)
        else None
    )
    nullspace = _canonical_nullspace(vh[rank:])
    profiles = tuple(
        _profile_parameter(protocol, evaluator, best, lower, upper, index)
        for index in range(len(estimated))
    )
    truth_values = tuple(10.0**item.truth_log10 for item in estimated)
    best_values = tuple(10.0**value for value in best)
    parameters_identifiable = (
        rank == len(estimated)
        and condition_number is not None
        and condition_number <= protocol.condition_number_limit
    )
    truth_error = float(np.max(np.abs(best - truth_log)))
    truth_recovered = (
        parameters_identifiable
        and truth_error <= protocol.truth_recovery_tolerance_log10
    )
    rank_expectation_met = rank == protocol.expected_rank
    all_multistarts_converged = all(item.success for item in attempts)
    profiles_completed = all(all(item.successful) for item in profiles)
    analysis_certified = (
        rank_expectation_met
        and all_multistarts_converged
        and profiles_completed
        and evaluator.forward_failure_count == 0
        and (truth_recovered if rank == len(estimated) else True)
    )

    values: dict[str, Any] = {
        "protocol": protocol,
        "protocol_sha256": protocol.sha256,
        "observable_labels": labels,
        "observable_units": units,
        "observed_values": tuple(float(value) for value in observed),
        "standard_deviations": tuple(float(value) for value in standard_deviations),
        "truth_values": truth_values,
        "best_fit_log10": tuple(float(value) for value in best),
        "best_fit_values": best_values,
        "best_chi_square": best_attempt.chi_square,
        "best_standardized_residual_linf": best_attempt.standardized_residual_linf,
        "weighted_jacobian": tuple(
            tuple(float(value) for value in row) for row in weighted_jacobian
        ),
        "fisher_information": tuple(
            tuple(float(value) for value in row) for row in fisher
        ),
        "fisher_correlation": tuple(
            tuple(float(value) for value in row) for row in correlation
        ),
        "singular_values": tuple(float(value) for value in singular_values),
        "numerical_rank": rank,
        "condition_number": condition_number,
        "nullspace_vectors": nullspace,
        "fit_attempts": attempts,
        "profiles": profiles,
        "forward_failure_count": evaluator.forward_failure_count,
        "rank_expectation_met": rank_expectation_met,
        "truth_recovered": truth_recovered,
        "all_multistarts_converged": all_multistarts_converged,
        "profiles_completed": profiles_completed,
        "parameters_identifiable": parameters_identifiable,
        "analysis_certified": analysis_certified,
    }
    unsigned = _json_ready(values)
    values["mapping_sha256"] = hashlib.sha256(
        _canonical_json(unsigned).encode("ascii")
    ).hexdigest()
    return InterfaceSRHIdentifiabilityResult(**values)


__all__ = [
    "IdentifiabilityError",
    "IdentifiabilityFitAttempt",
    "IdentifiabilityForwardError",
    "IdentifiabilityParameter",
    "IdentifiabilityProfile",
    "InterfaceCarrierCondition",
    "InterfaceSRHIdentifiabilityProtocol",
    "InterfaceSRHIdentifiabilityResult",
    "OBSERVABLE_FAMILIES",
    "PARAMETER_NAMES",
    "build_interface_srh_identifiability_protocol",
    "run_interface_srh_identifiability",
]

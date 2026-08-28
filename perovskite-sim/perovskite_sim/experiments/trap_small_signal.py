"""Canonical pure-local frequency protocol for one monovalent trap state.

This module certifies the constitutive trap response only. It does not expose a
device impedance route and does not add trap charge to Poisson, displacement
current, or carrier continuity. Those couplings require a separate device-level
certificate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Mapping, Self

import numpy as np

from perovskite_sim.models.defects import ACCEPTOR, DONOR
from perovskite_sim.physics.trap_kinetics import (
    TRAP_PHASOR_CONVENTION,
    TrapDCOperatingPoint,
    TrapFrequencyResponse,
    TrapKineticsCertificationError,
    TrapReservoirKinetics,
    TrapReservoirState,
    solve_trap_frequency_response,
)


TRAP_SMALL_SIGNAL_PROTOCOL_SCHEMA = "solarlab-local-trap-small-signal-v1"
TRAP_PERTURBATION_SCHEMA = "solarlab-local-trap-perturbation-v1"
TRAP_RESPONSE_NORMALIZATION = "per_applied_volt"
DYNAMIC_CHARGED_TRAP_TRANSITIONS = frozenset({ACCEPTOR, DONOR})


class TrapSmallSignalProtocolError(ValueError):
    """A local trap protocol or perturbation violates its exact schema."""


class TrapSmallSignalCertificationError(RuntimeError):
    """A finite local trap response failed one or more declared gates."""

    def __init__(self, message: str, result: "TrapSmallSignalResult") -> None:
        self.result = result
        super().__init__(message)


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise TrapSmallSignalProtocolError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise TrapSmallSignalProtocolError(f"{name} must be positive")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise TrapSmallSignalProtocolError(f"{name} must be non-negative")
    return number


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrapSmallSignalProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exact_schema(payload: Mapping[str, Any], cls: type, where: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    actual = set(payload)
    if actual != expected:
        raise TrapSmallSignalProtocolError(
            f"{where} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True)
class TrapSmallSignalPerturbation:
    """Real carrier-density phasors normalized to one applied volt."""

    electron_density_amplitude_m3_per_V: tuple[float, ...]
    hole_density_amplitude_m3_per_V: tuple[float, ...]
    response_normalization: Literal["per_applied_volt"] = TRAP_RESPONSE_NORMALIZATION
    phasor_convention: Literal["exp(+i*omega*t)"] = TRAP_PHASOR_CONVENTION
    schema_version: Literal["solarlab-local-trap-perturbation-v1"] = (
        TRAP_PERTURBATION_SCHEMA
    )

    def __post_init__(self) -> None:
        for name in (
            "electron_density_amplitude_m3_per_V",
            "hole_density_amplitude_m3_per_V",
        ):
            try:
                values = tuple(
                    _finite(item, f"{name}[{index}]")
                    for index, item in enumerate(getattr(self, name))
                )
            except TypeError as exc:
                raise TypeError(f"{name} must be an iterable") from exc
            if not values:
                raise TrapSmallSignalProtocolError(f"{name} must be non-empty")
            object.__setattr__(self, name, values)
        if not any(
            value != 0.0
            for value in (
                *self.electron_density_amplitude_m3_per_V,
                *self.hole_density_amplitude_m3_per_V,
            )
        ):
            raise TrapSmallSignalProtocolError(
                "the local carrier perturbation must not be identically zero"
            )
        if self.response_normalization != TRAP_RESPONSE_NORMALIZATION:
            raise TrapSmallSignalProtocolError(
                "unsupported trap response normalization"
            )
        if self.phasor_convention != TRAP_PHASOR_CONVENTION:
            raise TrapSmallSignalProtocolError("unsupported trap phasor convention")
        if self.schema_version != TRAP_PERTURBATION_SCHEMA:
            raise TrapSmallSignalProtocolError("unsupported trap perturbation schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["electron_density_amplitude_m3_per_V"] = list(
            self.electron_density_amplitude_m3_per_V
        )
        payload["hole_density_amplitude_m3_per_V"] = list(
            self.hole_density_amplitude_m3_per_V
        )
        return payload

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("trap perturbation must be a mapping")
        _exact_schema(payload, cls, "trap perturbation")
        values = dict(payload)
        values["electron_density_amplitude_m3_per_V"] = tuple(
            values["electron_density_amplitude_m3_per_V"]
        )
        values["hole_density_amplitude_m3_per_V"] = tuple(
            values["hole_density_amplitude_m3_per_V"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("trap perturbation JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class TrapSmallSignalProtocol:
    """Immutable local model identity, frequency request, and acceptance gates."""

    kinetics_sha256: str
    state_sha256: str
    operating_point_sha256: str
    perturbation_sha256: str
    charge_transition: Literal["acceptor", "donor"]
    frequencies_Hz: tuple[float, ...]
    max_operating_point_normalized_residual: float = 1.0e-12
    max_linear_solve_backward_error: float = 1.0e-12
    max_local_charge_conservation_relative_error: float = 1.0e-12
    max_conjugate_symmetry_relative_error: float = 1.0e-12
    max_low_frequency_relative_error: float = 2.0e-2
    max_high_frequency_frozen_ratio: float = 2.0e-2
    frequency_branch_margin_decades: float = 2.0
    max_frequency_sampling_gap_decades: float = 0.5
    response_normalization: Literal["per_applied_volt"] = TRAP_RESPONSE_NORMALIZATION
    phasor_convention: Literal["exp(+i*omega*t)"] = TRAP_PHASOR_CONVENTION
    schema_version: Literal["solarlab-local-trap-small-signal-v1"] = (
        TRAP_SMALL_SIGNAL_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        for name in (
            "kinetics_sha256",
            "state_sha256",
            "operating_point_sha256",
            "perturbation_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        transition = str(self.charge_transition).strip().lower()
        if transition not in DYNAMIC_CHARGED_TRAP_TRANSITIONS:
            raise TrapSmallSignalProtocolError(
                "certified dynamic trap response requires an acceptor or donor "
                f"charge transition, got {transition!r}"
            )
        object.__setattr__(self, "charge_transition", transition)
        try:
            frequencies = tuple(
                _positive(value, f"frequencies_Hz[{index}]")
                for index, value in enumerate(self.frequencies_Hz)
            )
        except TypeError as exc:
            raise TypeError("frequencies_Hz must be an iterable") from exc
        if len(frequencies) < 3 or any(
            right <= left for left, right in zip(frequencies, frequencies[1:])
        ):
            raise TrapSmallSignalProtocolError(
                "frequencies_Hz must contain at least three increasing values"
            )
        object.__setattr__(self, "frequencies_Hz", frequencies)
        for name in (
            "max_operating_point_normalized_residual",
            "max_linear_solve_backward_error",
            "max_local_charge_conservation_relative_error",
            "max_conjugate_symmetry_relative_error",
            "max_low_frequency_relative_error",
            "max_high_frequency_frozen_ratio",
            "max_frequency_sampling_gap_decades",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "frequency_branch_margin_decades",
            _nonnegative(
                self.frequency_branch_margin_decades,
                "frequency_branch_margin_decades",
            ),
        )
        if self.response_normalization != TRAP_RESPONSE_NORMALIZATION:
            raise TrapSmallSignalProtocolError(
                "unsupported trap response normalization"
            )
        if self.phasor_convention != TRAP_PHASOR_CONVENTION:
            raise TrapSmallSignalProtocolError("unsupported trap phasor convention")
        if self.schema_version != TRAP_SMALL_SIGNAL_PROTOCOL_SCHEMA:
            raise TrapSmallSignalProtocolError("unsupported trap protocol schema")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["frequencies_Hz"] = list(self.frequencies_Hz)
        return payload

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("trap small-signal protocol must be a mapping")
        _exact_schema(payload, cls, "trap small-signal protocol")
        values = dict(payload)
        values["frequencies_Hz"] = tuple(values["frequencies_Hz"])
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("trap protocol JSON must contain an object")
        return cls.from_dict(parsed)


def build_trap_small_signal_protocol(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    operating_point: TrapDCOperatingPoint,
    perturbation: TrapSmallSignalPerturbation,
    frequencies_Hz: object,
    *,
    charge_transition: Literal["acceptor", "donor"],
    **gates: Any,
) -> TrapSmallSignalProtocol:
    """Bind a local frequency request to exact kinetics, state, and forcing."""
    if not isinstance(kinetics, TrapReservoirKinetics):
        raise TypeError("kinetics must be TrapReservoirKinetics")
    if not isinstance(state, TrapReservoirState):
        raise TypeError("state must be TrapReservoirState")
    if not isinstance(operating_point, TrapDCOperatingPoint):
        raise TypeError("operating_point must be TrapDCOperatingPoint")
    if not isinstance(perturbation, TrapSmallSignalPerturbation):
        raise TypeError("perturbation must be TrapSmallSignalPerturbation")
    if not operating_point.certified:
        raise TrapKineticsCertificationError(
            "trap protocol requires a certified DC operating point"
        )
    if operating_point.kinetics_sha256 != kinetics.sha256:
        raise TrapSmallSignalProtocolError(
            "operating point does not match trap kinetics"
        )
    if operating_point.state_sha256 != state.sha256:
        raise TrapSmallSignalProtocolError("operating point does not match trap state")
    if len(perturbation.electron_density_amplitude_m3_per_V) != (
        kinetics.electron_reservoir_count
    ) or len(perturbation.hole_density_amplitude_m3_per_V) != (
        kinetics.hole_reservoir_count
    ):
        raise TrapSmallSignalProtocolError(
            "perturbation reservoir counts do not match trap kinetics"
        )
    return TrapSmallSignalProtocol(
        kinetics_sha256=kinetics.sha256,
        state_sha256=state.sha256,
        operating_point_sha256=operating_point.sha256,
        perturbation_sha256=perturbation.sha256,
        charge_transition=charge_transition,
        frequencies_Hz=tuple(np.asarray(frequencies_Hz, dtype=float).tolist()),
        **gates,
    )


@dataclass(frozen=True, slots=True)
class TrapFrequencyWindowAssessment:
    """Whether the sampled frequencies resolve both sides of trap relaxation."""

    relaxation_frequency_Hz: float
    minimum_frequency_Hz: float
    maximum_frequency_Hz: float
    required_low_frequency_Hz: float
    required_high_frequency_Hz: float
    maximum_sampling_gap_decades: float
    low_frequency_limit_covered: bool
    high_frequency_limit_covered: bool
    relaxation_bracketed: bool
    sampling_density_passed: bool
    certified: bool
    warnings: tuple[str, ...]


def assess_trap_frequency_window(
    frequencies_Hz: object,
    relaxation_rate_s1: float,
    *,
    branch_margin_decades: float,
    max_sampling_gap_decades: float,
) -> TrapFrequencyWindowAssessment:
    """Assess actual trap timescale coverage without a hard-coded lower bound."""
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size < 3
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise TrapSmallSignalProtocolError(
            "frequency assessment requires at least three increasing values"
        )
    relaxation = _positive(relaxation_rate_s1, "relaxation_rate_s1")
    margin = _nonnegative(branch_margin_decades, "branch_margin_decades")
    gap_limit = _positive(max_sampling_gap_decades, "max_sampling_gap_decades")
    corner = relaxation / (2.0 * np.pi)
    required_low = corner / (10.0**margin)
    required_high = corner * (10.0**margin)
    maximum_gap = float(np.max(np.diff(np.log10(frequencies))))
    low_covered = bool(frequencies[0] <= required_low)
    high_covered = bool(frequencies[-1] >= required_high)
    bracketed = bool(frequencies[0] < corner < frequencies[-1])
    comparison_slack = 16.0 * np.finfo(float).eps * max(1.0, gap_limit)
    sampled = bool(maximum_gap <= gap_limit + comparison_slack)
    warnings: list[str] = []
    if not low_covered:
        warnings.append("low_frequency_limit_not_covered")
    if not high_covered:
        warnings.append("high_frequency_limit_not_covered")
    if not bracketed:
        warnings.append("trap_relaxation_not_bracketed")
    if not sampled:
        warnings.append("frequency_sampling_gap_exceeds_limit")
    certified = low_covered and high_covered and bracketed and sampled
    return TrapFrequencyWindowAssessment(
        relaxation_frequency_Hz=corner,
        minimum_frequency_Hz=float(frequencies[0]),
        maximum_frequency_Hz=float(frequencies[-1]),
        required_low_frequency_Hz=required_low,
        required_high_frequency_Hz=required_high,
        maximum_sampling_gap_decades=maximum_gap,
        low_frequency_limit_covered=low_covered,
        high_frequency_limit_covered=high_covered,
        relaxation_bracketed=bracketed,
        sampling_density_passed=sampled,
        certified=certified,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class TrapSmallSignalCertificate:
    """Local numerical, asymptotic, and frequency-window evidence."""

    operating_point_certified: bool
    numerical_response_certified: bool
    low_frequency_limit_certified: bool
    high_frequency_limit_certified: bool
    frequency_window_certified: bool
    certified: bool
    max_linear_solve_backward_error: float
    max_local_charge_conservation_relative_error: float
    max_conjugate_symmetry_relative_error: float
    low_frequency_relative_error: float
    high_frequency_frozen_ratio: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrapSmallSignalResult:
    """Protocol-bound pure-local response; not a device admittance."""

    protocol: TrapSmallSignalProtocol
    perturbation: TrapSmallSignalPerturbation
    response: TrapFrequencyResponse
    frequency_window: TrapFrequencyWindowAssessment
    certificate: TrapSmallSignalCertificate


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    scale = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0e-300)
    return float(np.max(np.abs(left_values - right_values) / scale))


def _conjugate_symmetry_error(
    positive: TrapFrequencyResponse,
    negative: TrapFrequencyResponse,
) -> float:
    pairs = (
        (positive.occupancy_response_per_V, negative.occupancy_response_per_V),
        (
            positive.electron_capture_response_s1_per_V,
            negative.electron_capture_response_s1_per_V,
        ),
        (
            positive.hole_capture_response_s1_per_V,
            negative.hole_capture_response_s1_per_V,
        ),
        (
            positive.charge_per_trap_response_C_per_V,
            negative.charge_per_trap_response_C_per_V,
        ),
    )
    return max(_relative_error(left, np.conjugate(right)) for left, right in pairs)


def run_local_trap_small_signal(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    operating_point: TrapDCOperatingPoint,
    perturbation: TrapSmallSignalPerturbation,
    protocol: TrapSmallSignalProtocol,
    *,
    require_certificate: bool = True,
) -> TrapSmallSignalResult:
    """Run and certify the analytic local trap response."""
    if not isinstance(protocol, TrapSmallSignalProtocol):
        raise TypeError("protocol must be TrapSmallSignalProtocol")
    identities = (
        ("kinetics", protocol.kinetics_sha256, kinetics.sha256),
        ("state", protocol.state_sha256, state.sha256),
        (
            "operating point",
            protocol.operating_point_sha256,
            operating_point.sha256,
        ),
        ("perturbation", protocol.perturbation_sha256, perturbation.sha256),
    )
    mismatched = [name for name, expected, actual in identities if expected != actual]
    if mismatched:
        raise TrapSmallSignalProtocolError(
            "trap protocol identity mismatch: " + ", ".join(mismatched)
        )
    if len(perturbation.electron_density_amplitude_m3_per_V) != (
        kinetics.electron_reservoir_count
    ) or len(perturbation.hole_density_amplitude_m3_per_V) != (
        kinetics.hole_reservoir_count
    ):
        raise TrapSmallSignalProtocolError(
            "perturbation reservoir counts do not match trap kinetics"
        )
    frequencies = np.asarray(protocol.frequencies_Hz, dtype=float)
    electron_amplitude = np.asarray(
        perturbation.electron_density_amplitude_m3_per_V,
        dtype=float,
    )
    hole_amplitude = np.asarray(
        perturbation.hole_density_amplitude_m3_per_V,
        dtype=float,
    )
    response = solve_trap_frequency_response(
        kinetics,
        state,
        operating_point,
        frequencies,
        electron_amplitude,
        hole_amplitude,
        charge_transition=protocol.charge_transition,
    )
    negative = solve_trap_frequency_response(
        kinetics,
        state,
        operating_point,
        -frequencies,
        electron_amplitude,
        hole_amplitude,
        charge_transition=protocol.charge_transition,
    )
    window = assess_trap_frequency_window(
        frequencies,
        operating_point.relaxation_rate_s1,
        branch_margin_decades=protocol.frequency_branch_margin_decades,
        max_sampling_gap_decades=protocol.max_frequency_sampling_gap_decades,
    )
    max_backward = float(np.max(response.linear_solve_backward_error))
    omega = 2.0 * np.pi * frequencies
    balance_scale = (
        np.sum(np.abs(response.electron_capture_response_s1_per_V), axis=1)
        + np.sum(np.abs(response.hole_capture_response_s1_per_V), axis=1)
        + np.abs(1j * omega * response.occupancy_response_per_V)
    )
    balance_error = float(
        np.max(
            np.divide(
                np.abs(response.occupancy_balance_residual_s1_per_V),
                balance_scale,
                out=np.zeros_like(balance_scale),
                where=balance_scale > 0.0,
            )
        )
    )
    symmetry_error = _conjugate_symmetry_error(response, negative)
    quasistatic_scale = np.abs(response.quasistatic_occupancy_response_per_V)
    if not np.any(quasistatic_scale > 0.0):
        raise TrapSmallSignalProtocolError(
            "carrier perturbation has zero net occupancy forcing"
        )
    low_error = float(
        np.abs(
            response.occupancy_response_per_V[0]
            - response.quasistatic_occupancy_response_per_V[0]
        )
        / quasistatic_scale[0]
    )
    high_ratio = float(
        np.abs(response.occupancy_response_per_V[-1]) / quasistatic_scale[-1]
    )
    operating_certified = bool(
        operating_point.certified
        and operating_point.normalized_residual
        <= protocol.max_operating_point_normalized_residual
    )
    numerical = bool(
        max_backward <= protocol.max_linear_solve_backward_error
        and balance_error <= protocol.max_local_charge_conservation_relative_error
        and symmetry_error <= protocol.max_conjugate_symmetry_relative_error
    )
    low_certified = low_error <= protocol.max_low_frequency_relative_error
    high_certified = high_ratio <= protocol.max_high_frequency_frozen_ratio
    reasons: list[str] = []
    if not operating_certified:
        reasons.append("dc_operating_point_not_certified")
    if max_backward > protocol.max_linear_solve_backward_error:
        reasons.append("linear_solve_backward_error_exceeds_limit")
    if balance_error > protocol.max_local_charge_conservation_relative_error:
        reasons.append("local_charge_conservation_exceeds_limit")
    if symmetry_error > protocol.max_conjugate_symmetry_relative_error:
        reasons.append("conjugate_symmetry_exceeds_limit")
    if not low_certified:
        reasons.append("low_frequency_limit_not_converged")
    if not high_certified:
        reasons.append("high_frequency_frozen_limit_not_converged")
    reasons.extend(window.warnings)
    certified = bool(
        operating_certified
        and numerical
        and low_certified
        and high_certified
        and window.certified
    )
    certificate = TrapSmallSignalCertificate(
        operating_point_certified=operating_certified,
        numerical_response_certified=numerical,
        low_frequency_limit_certified=low_certified,
        high_frequency_limit_certified=high_certified,
        frequency_window_certified=window.certified,
        certified=certified,
        max_linear_solve_backward_error=max_backward,
        max_local_charge_conservation_relative_error=balance_error,
        max_conjugate_symmetry_relative_error=symmetry_error,
        low_frequency_relative_error=low_error,
        high_frequency_frozen_ratio=high_ratio,
        reasons=tuple(dict.fromkeys(reasons)),
    )
    result = TrapSmallSignalResult(
        protocol=protocol,
        perturbation=perturbation,
        response=response,
        frequency_window=window,
        certificate=certificate,
    )
    if require_certificate and not certified:
        raise TrapSmallSignalCertificationError(
            "local trap small-signal certificate failed: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "DYNAMIC_CHARGED_TRAP_TRANSITIONS",
    "TRAP_PERTURBATION_SCHEMA",
    "TRAP_RESPONSE_NORMALIZATION",
    "TRAP_SMALL_SIGNAL_PROTOCOL_SCHEMA",
    "TrapFrequencyWindowAssessment",
    "TrapSmallSignalCertificate",
    "TrapSmallSignalCertificationError",
    "TrapSmallSignalPerturbation",
    "TrapSmallSignalProtocol",
    "TrapSmallSignalProtocolError",
    "TrapSmallSignalResult",
    "assess_trap_frequency_window",
    "build_trap_small_signal_protocol",
    "run_local_trap_small_signal",
]

"""Local monovalent trap kinetics and analytic frequency response.

The same electron-occupancy state covers a bulk single level and a shared
two-sided interface level. A bulk trap has one electron and one hole
reservoir; an interface trap can have one reservoir of each carrier on both
sides. Capture coefficients are microscopic ``sigma * v_th`` values in
``m^3/s``. Surface recombination velocities ``sigma * v_th * N_t`` are not
accepted here because their denominator is a flux rather than an occupancy
relaxation rate.

The phasor convention is ``exp(+i * omega * t)``. At a certified DC point,

``df/dt = sum(r_n) - sum(r_p)``

and the scalar frequency-domain operator is

``(i * omega + lambda) * delta_f = B_n delta_n + B_p delta_p``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Mapping, Self

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import ACCEPTOR, DONOR, NEUTRAL


TRAP_PHASOR_CONVENTION = "exp(+i*omega*t)"
TRAP_RESERVOIR_KINETICS_SCHEMA = "solarlab-trap-reservoir-kinetics-v1"
TRAP_RESERVOIR_STATE_SCHEMA = "solarlab-trap-reservoir-state-v1"
SUPPORTED_TRAP_CHARGE_TRANSITIONS = frozenset({NEUTRAL, ACCEPTOR, DONOR})


class TrapKineticsError(ValueError):
    """A local trap input or operating point is not physically admissible."""


class TrapKineticsCertificationError(RuntimeError):
    """A finite local trap operating point failed its residual contract."""


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise TrapKineticsError(f"{name} must be finite")
    return number


def _finite_positive(value: object, name: str) -> float:
    number = _finite_real(value, name)
    if number <= 0.0:
        raise TrapKineticsError(f"{name} must be positive")
    return number


def _finite_nonnegative(value: object, name: str) -> float:
    number = _finite_real(value, name)
    if number < 0.0:
        raise TrapKineticsError(f"{name} must be non-negative")
    return number


def _readonly_vector(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise TrapKineticsError(f"{name} must be a non-empty finite 1-D array")
    if nonnegative and np.any(array < 0.0):
        raise TrapKineticsError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _readonly_complex(value: object) -> np.ndarray:
    array = np.array(value, dtype=complex, copy=True)
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise TrapKineticsError(
            f"{where} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True, eq=False)
class TrapReservoirKinetics:
    """Microscopic capture coefficients and emission reference densities."""

    identifier: str
    electron_capture_coefficients_m3_s: np.ndarray
    hole_capture_coefficients_m3_s: np.ndarray
    electron_reference_densities_m3: np.ndarray
    hole_reference_densities_m3: np.ndarray
    schema_version: Literal["solarlab-trap-reservoir-kinetics-v1"] = (
        TRAP_RESERVOIR_KINETICS_SCHEMA
    )

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise TrapKineticsError("identifier must be non-empty")
        object.__setattr__(self, "identifier", identifier)
        for name in (
            "electron_capture_coefficients_m3_s",
            "hole_capture_coefficients_m3_s",
            "electron_reference_densities_m3",
            "hole_reference_densities_m3",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), name, nonnegative=True),
            )
        if (
            self.electron_capture_coefficients_m3_s.shape
            != self.electron_reference_densities_m3.shape
        ):
            raise TrapKineticsError(
                "electron capture coefficients and reference densities must align"
            )
        if (
            self.hole_capture_coefficients_m3_s.shape
            != self.hole_reference_densities_m3.shape
        ):
            raise TrapKineticsError(
                "hole capture coefficients and reference densities must align"
            )
        if not (
            np.any(self.electron_capture_coefficients_m3_s > 0.0)
            or np.any(self.hole_capture_coefficients_m3_s > 0.0)
        ):
            raise TrapKineticsError(
                "at least one electron or hole capture coefficient must be positive"
            )
        if self.schema_version != TRAP_RESERVOIR_KINETICS_SCHEMA:
            raise TrapKineticsError("unsupported trap reservoir kinetics schema")

    @property
    def electron_reservoir_count(self) -> int:
        return int(self.electron_capture_coefficients_m3_s.size)

    @property
    def hole_reservoir_count(self) -> int:
        return int(self.hole_capture_coefficients_m3_s.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "electron_capture_coefficients_m3_s": (
                self.electron_capture_coefficients_m3_s.tolist()
            ),
            "hole_capture_coefficients_m3_s": (
                self.hole_capture_coefficients_m3_s.tolist()
            ),
            "electron_reference_densities_m3": (
                self.electron_reference_densities_m3.tolist()
            ),
            "hole_reference_densities_m3": (self.hole_reference_densities_m3.tolist()),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("trap reservoir kinetics must be a mapping")
        _require_exact_keys(
            payload,
            {
                "identifier",
                "electron_capture_coefficients_m3_s",
                "hole_capture_coefficients_m3_s",
                "electron_reference_densities_m3",
                "hole_reference_densities_m3",
                "schema_version",
            },
            "trap reservoir kinetics",
        )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("trap reservoir kinetics JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True, eq=False)
class TrapReservoirState:
    """Carrier densities at the reservoirs coupled to one occupancy state."""

    electron_densities_m3: np.ndarray
    hole_densities_m3: np.ndarray
    schema_version: Literal["solarlab-trap-reservoir-state-v1"] = (
        TRAP_RESERVOIR_STATE_SCHEMA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electron_densities_m3",
            _readonly_vector(
                self.electron_densities_m3,
                "electron_densities_m3",
                nonnegative=True,
            ),
        )
        if self.schema_version != TRAP_RESERVOIR_STATE_SCHEMA:
            raise TrapKineticsError("unsupported trap reservoir state schema")
        object.__setattr__(
            self,
            "hole_densities_m3",
            _readonly_vector(
                self.hole_densities_m3,
                "hole_densities_m3",
                nonnegative=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "electron_densities_m3": self.electron_densities_m3.tolist(),
            "hole_densities_m3": self.hole_densities_m3.tolist(),
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("trap reservoir state must be a mapping")
        _require_exact_keys(
            payload,
            {
                "electron_densities_m3",
                "hole_densities_m3",
                "schema_version",
            },
            "trap reservoir state",
        )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("trap reservoir state JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True, eq=False)
class TrapDCOperatingPoint:
    """One local steady occupancy and its residual evidence."""

    kinetics_sha256: str
    state_sha256: str
    occupancy: float
    filled_rate_s1: float
    empty_rate_s1: float
    relaxation_rate_s1: float
    electron_capture_rates_s1: np.ndarray
    hole_capture_rates_s1: np.ndarray
    occupancy_rate_residual_s1: float
    normalized_residual: float
    certified: bool

    def __post_init__(self) -> None:
        for name in ("kinetics_sha256", "state_sha256"):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise TrapKineticsError(f"{name} must be a lowercase SHA-256 digest")
        occupancy = _finite_real(self.occupancy, "occupancy")
        if not 0.0 <= occupancy <= 1.0:
            raise TrapKineticsError("occupancy must lie in [0, 1]")
        object.__setattr__(self, "occupancy", occupancy)
        for name in ("filled_rate_s1", "empty_rate_s1"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "relaxation_rate_s1",
            _finite_positive(self.relaxation_rate_s1, "relaxation_rate_s1"),
        )
        object.__setattr__(
            self,
            "occupancy_rate_residual_s1",
            _finite_real(
                self.occupancy_rate_residual_s1,
                "occupancy_rate_residual_s1",
            ),
        )
        object.__setattr__(
            self,
            "normalized_residual",
            _finite_nonnegative(self.normalized_residual, "normalized_residual"),
        )
        for name in ("electron_capture_rates_s1", "hole_capture_rates_s1"):
            object.__setattr__(self, name, _readonly_vector(getattr(self, name), name))
        if not isinstance(self.certified, (bool, np.bool_)):
            raise TypeError("certified must be boolean")
        object.__setattr__(self, "certified", bool(self.certified))
        tolerance = 32.0 * np.finfo(float).eps
        if not math.isclose(
            self.relaxation_rate_s1,
            self.filled_rate_s1 + self.empty_rate_s1,
            rel_tol=tolerance,
            abs_tol=0.0,
        ):
            raise TrapKineticsError(
                "relaxation_rate_s1 must equal filled_rate_s1 + empty_rate_s1"
            )
        capture_residual = float(
            np.sum(self.electron_capture_rates_s1) - np.sum(self.hole_capture_rates_s1)
        )
        residual_scale = max(
            abs(capture_residual),
            abs(self.occupancy_rate_residual_s1),
            self.relaxation_rate_s1,
        )
        if abs(capture_residual - self.occupancy_rate_residual_s1) > (
            tolerance * residual_scale
        ):
            raise TrapKineticsError(
                "occupancy residual must equal electron minus hole capture"
            )
        expected_normalized = abs(self.occupancy_rate_residual_s1) / (
            self.relaxation_rate_s1
        )
        if not math.isclose(
            self.normalized_residual,
            expected_normalized,
            rel_tol=tolerance,
            abs_tol=np.finfo(float).tiny,
        ):
            raise TrapKineticsError(
                "normalized_residual does not match the occupancy residual"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinetics_sha256": self.kinetics_sha256,
            "state_sha256": self.state_sha256,
            "occupancy": self.occupancy,
            "filled_rate_s1": self.filled_rate_s1,
            "empty_rate_s1": self.empty_rate_s1,
            "relaxation_rate_s1": self.relaxation_rate_s1,
            "electron_capture_rates_s1": self.electron_capture_rates_s1.tolist(),
            "hole_capture_rates_s1": self.hole_capture_rates_s1.tolist(),
            "occupancy_rate_residual_s1": self.occupancy_rate_residual_s1,
            "normalized_residual": self.normalized_residual,
            "certified": self.certified,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True, eq=False)
class TrapLinearization:
    """Analytic derivatives of occupancy and carrier capture at one DC point."""

    operating_point: TrapDCOperatingPoint
    electron_density_forcing_m3_s: np.ndarray
    hole_density_forcing_m3_s: np.ndarray
    electron_capture_occupancy_derivative_s1: np.ndarray
    hole_capture_occupancy_derivative_s1: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.operating_point, TrapDCOperatingPoint):
            raise TypeError("operating_point must be TrapDCOperatingPoint")
        for name in (
            "electron_density_forcing_m3_s",
            "hole_density_forcing_m3_s",
            "electron_capture_occupancy_derivative_s1",
            "hole_capture_occupancy_derivative_s1",
        ):
            object.__setattr__(self, name, _readonly_vector(getattr(self, name), name))


@dataclass(frozen=True, slots=True, eq=False)
class TrapFrequencyResponse:
    """Per-trap complex response for supplied carrier-density phasors."""

    frequencies_Hz: np.ndarray
    relaxation_frequency_Hz: float
    occupancy_response_per_V: np.ndarray
    quasistatic_occupancy_response_per_V: np.ndarray
    electron_capture_response_s1_per_V: np.ndarray
    hole_capture_response_s1_per_V: np.ndarray
    charge_per_trap_response_C_per_V: np.ndarray
    occupancy_balance_residual_s1_per_V: np.ndarray
    linear_solve_backward_error: np.ndarray
    charge_transition: str
    phasor_convention: str = TRAP_PHASOR_CONVENTION

    def __post_init__(self) -> None:
        frequencies = np.array(self.frequencies_Hz, dtype=float, copy=True)
        if (
            frequencies.ndim != 1
            or frequencies.size == 0
            or not np.all(np.isfinite(frequencies))
            or np.any(frequencies == 0.0)
        ):
            raise TrapKineticsError(
                "frequencies_Hz must be a non-empty finite 1-D array without zero"
            )
        frequencies.setflags(write=False)
        object.__setattr__(self, "frequencies_Hz", frequencies)
        object.__setattr__(
            self,
            "relaxation_frequency_Hz",
            _finite_positive(self.relaxation_frequency_Hz, "relaxation_frequency_Hz"),
        )
        vector_fields = (
            "occupancy_response_per_V",
            "quasistatic_occupancy_response_per_V",
            "charge_per_trap_response_C_per_V",
            "occupancy_balance_residual_s1_per_V",
        )
        for name in vector_fields:
            value = _readonly_complex(getattr(self, name))
            if value.shape != frequencies.shape or not np.all(np.isfinite(value)):
                raise TrapKineticsError(f"{name} must be finite and match frequencies")
            object.__setattr__(self, name, value)
        for name in (
            "electron_capture_response_s1_per_V",
            "hole_capture_response_s1_per_V",
        ):
            value = _readonly_complex(getattr(self, name))
            if (
                value.ndim != 2
                or value.shape[0] != frequencies.size
                or value.shape[1] == 0
                or not np.all(np.isfinite(value))
            ):
                raise TrapKineticsError(
                    f"{name} must be a finite frequency-by-reservoir array"
                )
            object.__setattr__(self, name, value)
        backward = np.array(self.linear_solve_backward_error, dtype=float, copy=True)
        if (
            backward.shape != frequencies.shape
            or not np.all(np.isfinite(backward))
            or np.any(backward < 0.0)
        ):
            raise TrapKineticsError(
                "linear_solve_backward_error must be finite, non-negative, and "
                "match frequencies"
            )
        backward.setflags(write=False)
        object.__setattr__(self, "linear_solve_backward_error", backward)
        transition = str(self.charge_transition).strip().lower()
        if transition not in SUPPORTED_TRAP_CHARGE_TRANSITIONS:
            raise TrapKineticsError(f"unsupported charge transition {transition!r}")
        object.__setattr__(self, "charge_transition", transition)
        if self.phasor_convention != TRAP_PHASOR_CONVENTION:
            raise TrapKineticsError("unsupported trap phasor convention")


def _validate_alignment(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
) -> None:
    if state.electron_densities_m3.shape != (kinetics.electron_reservoir_count,):
        raise TrapKineticsError("electron state does not match kinetics reservoirs")
    if state.hole_densities_m3.shape != (kinetics.hole_reservoir_count,):
        raise TrapKineticsError("hole state does not match kinetics reservoirs")


def evaluate_trap_dc_operating_point(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    *,
    occupancy: float | None = None,
    max_normalized_residual: float = 1.0e-12,
    require_certified: bool = True,
) -> TrapDCOperatingPoint:
    """Evaluate or verify a steady local trap occupancy without clipping."""
    if not isinstance(kinetics, TrapReservoirKinetics):
        raise TypeError("kinetics must be TrapReservoirKinetics")
    if not isinstance(state, TrapReservoirState):
        raise TypeError("state must be TrapReservoirState")
    _validate_alignment(kinetics, state)
    tolerance = _finite_positive(max_normalized_residual, "max_normalized_residual")
    c_n = kinetics.electron_capture_coefficients_m3_s
    c_p = kinetics.hole_capture_coefficients_m3_s
    n1 = kinetics.electron_reference_densities_m3
    p1 = kinetics.hole_reference_densities_m3
    n = state.electron_densities_m3
    p = state.hole_densities_m3
    filled = float(np.dot(c_n, n) + np.dot(c_p, p1))
    empty = float(np.dot(c_n, n1) + np.dot(c_p, p))
    relaxation = filled + empty
    if not math.isfinite(relaxation) or relaxation <= 0.0:
        raise TrapKineticsError("trap relaxation rate must be finite and positive")
    resolved_occupancy = (
        filled / relaxation
        if occupancy is None
        else _finite_real(occupancy, "occupancy")
    )
    if not 0.0 <= resolved_occupancy <= 1.0:
        raise TrapKineticsError("occupancy must lie in [0, 1]")
    electron_capture = c_n * (n * (1.0 - resolved_occupancy) - n1 * resolved_occupancy)
    hole_capture = c_p * (p * resolved_occupancy - p1 * (1.0 - resolved_occupancy))
    residual = float(np.sum(electron_capture) - np.sum(hole_capture))
    normalized = abs(residual) / relaxation
    certified = bool(math.isfinite(normalized) and normalized <= tolerance)
    result = TrapDCOperatingPoint(
        kinetics_sha256=kinetics.sha256,
        state_sha256=state.sha256,
        occupancy=resolved_occupancy,
        filled_rate_s1=filled,
        empty_rate_s1=empty,
        relaxation_rate_s1=relaxation,
        electron_capture_rates_s1=electron_capture,
        hole_capture_rates_s1=hole_capture,
        occupancy_rate_residual_s1=residual,
        normalized_residual=normalized,
        certified=certified,
    )
    if require_certified and not certified:
        raise TrapKineticsCertificationError(
            "trap DC occupancy residual did not certify: "
            f"normalized_residual={normalized:.6g}, limit={tolerance:.6g}"
        )
    return result


def linearize_trap_kinetics(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    operating_point: TrapDCOperatingPoint,
) -> TrapLinearization:
    """Return exact local derivatives at a certified DC occupancy."""
    if not isinstance(operating_point, TrapDCOperatingPoint):
        raise TypeError("operating_point must be TrapDCOperatingPoint")
    _validate_alignment(kinetics, state)
    if not operating_point.certified:
        raise TrapKineticsCertificationError(
            "frequency response requires a certified trap DC operating point"
        )
    if operating_point.kinetics_sha256 != kinetics.sha256:
        raise TrapKineticsError("operating point does not match trap kinetics")
    if operating_point.state_sha256 != state.sha256:
        raise TrapKineticsError("operating point does not match reservoir state")
    verified = evaluate_trap_dc_operating_point(
        kinetics,
        state,
        occupancy=operating_point.occupancy,
    )
    if verified.sha256 != operating_point.sha256:
        raise TrapKineticsError(
            "operating point content does not match the declared kinetics and state"
        )
    occupancy = operating_point.occupancy
    c_n = kinetics.electron_capture_coefficients_m3_s
    c_p = kinetics.hole_capture_coefficients_m3_s
    n1 = kinetics.electron_reference_densities_m3
    p1 = kinetics.hole_reference_densities_m3
    n = state.electron_densities_m3
    p = state.hole_densities_m3
    return TrapLinearization(
        operating_point=operating_point,
        electron_density_forcing_m3_s=c_n * (1.0 - occupancy),
        hole_density_forcing_m3_s=-c_p * occupancy,
        electron_capture_occupancy_derivative_s1=-c_n * (n + n1),
        hole_capture_occupancy_derivative_s1=c_p * (p + p1),
    )


def fixed_quasi_fermi_density_response_per_potential(
    state: TrapReservoirState,
    thermal_voltage_V: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``dn/dphi`` and ``dp/dphi`` at fixed carrier QF levels."""
    thermal = _finite_positive(thermal_voltage_V, "thermal_voltage_V")
    return (
        np.asarray(state.electron_densities_m3) / thermal,
        -np.asarray(state.hole_densities_m3) / thermal,
    )


def _broadcast_density_response(
    value: object,
    *,
    frequencies: np.ndarray,
    reservoir_count: int,
    name: str,
) -> np.ndarray:
    response = np.asarray(value, dtype=complex)
    if response.shape == (reservoir_count,):
        response = np.broadcast_to(response, (frequencies.size, reservoir_count))
    if response.shape != (frequencies.size, reservoir_count) or not np.all(
        np.isfinite(response)
    ):
        raise TrapKineticsError(
            f"{name} must be finite with reservoir or frequency-by-reservoir shape"
        )
    return response


def solve_trap_frequency_response(
    kinetics: TrapReservoirKinetics,
    state: TrapReservoirState,
    operating_point: TrapDCOperatingPoint,
    frequencies_Hz: object,
    electron_density_response_m3_per_V: object,
    hole_density_response_m3_per_V: object,
    *,
    charge_transition: str,
) -> TrapFrequencyResponse:
    """Solve the exact scalar occupancy response for arbitrary carrier phasors."""
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies == 0.0)
    ):
        raise TrapKineticsError(
            "frequencies_Hz must be finite, one-dimensional, and nonzero"
        )
    transition = str(charge_transition).strip().lower()
    if transition not in SUPPORTED_TRAP_CHARGE_TRANSITIONS:
        raise TrapKineticsError(f"unsupported charge transition {transition!r}")
    linearization = linearize_trap_kinetics(kinetics, state, operating_point)
    electron_response = _broadcast_density_response(
        electron_density_response_m3_per_V,
        frequencies=frequencies,
        reservoir_count=kinetics.electron_reservoir_count,
        name="electron_density_response_m3_per_V",
    )
    hole_response = _broadcast_density_response(
        hole_density_response_m3_per_V,
        frequencies=frequencies,
        reservoir_count=kinetics.hole_reservoir_count,
        name="hole_density_response_m3_per_V",
    )
    forcing = (
        electron_response @ linearization.electron_density_forcing_m3_s
        + hole_response @ linearization.hole_density_forcing_m3_s
    )
    omega = 2.0 * np.pi * frequencies
    relaxation = operating_point.relaxation_rate_s1
    operator = relaxation + 1j * omega
    occupancy_response = forcing / operator
    quasistatic_response = forcing / relaxation
    electron_capture_response = (
        electron_response * linearization.electron_density_forcing_m3_s[np.newaxis, :]
        + occupancy_response[:, np.newaxis]
        * linearization.electron_capture_occupancy_derivative_s1[np.newaxis, :]
    )
    hole_capture_response = (
        hole_response
        * (kinetics.hole_capture_coefficients_m3_s * operating_point.occupancy)[
            np.newaxis, :
        ]
        + occupancy_response[:, np.newaxis]
        * linearization.hole_capture_occupancy_derivative_s1[np.newaxis, :]
    )
    occupancy_balance_residual = (
        np.sum(electron_capture_response, axis=1)
        - np.sum(hole_capture_response, axis=1)
        - 1j * omega * occupancy_response
    )
    equation_residual = forcing - operator * occupancy_response
    residual_scale = np.abs(forcing) + np.abs(operator) * np.abs(occupancy_response)
    backward_error = np.divide(
        np.abs(equation_residual),
        residual_scale,
        out=np.zeros_like(residual_scale, dtype=float),
        where=residual_scale > 0.0,
    )
    charge_response = (
        np.zeros_like(occupancy_response)
        if transition == NEUTRAL
        else -Q * occupancy_response
    )
    return TrapFrequencyResponse(
        frequencies_Hz=frequencies,
        relaxation_frequency_Hz=relaxation / (2.0 * np.pi),
        occupancy_response_per_V=occupancy_response,
        quasistatic_occupancy_response_per_V=quasistatic_response,
        electron_capture_response_s1_per_V=electron_capture_response,
        hole_capture_response_s1_per_V=hole_capture_response,
        charge_per_trap_response_C_per_V=charge_response,
        occupancy_balance_residual_s1_per_V=occupancy_balance_residual,
        linear_solve_backward_error=backward_error,
        charge_transition=transition,
    )


__all__ = [
    "SUPPORTED_TRAP_CHARGE_TRANSITIONS",
    "TRAP_PHASOR_CONVENTION",
    "TRAP_RESERVOIR_KINETICS_SCHEMA",
    "TRAP_RESERVOIR_STATE_SCHEMA",
    "TrapDCOperatingPoint",
    "TrapFrequencyResponse",
    "TrapKineticsCertificationError",
    "TrapKineticsError",
    "TrapLinearization",
    "TrapReservoirKinetics",
    "TrapReservoirState",
    "evaluate_trap_dc_operating_point",
    "fixed_quasi_fermi_density_response_per_potential",
    "linearize_trap_kinetics",
    "solve_trap_frequency_response",
]

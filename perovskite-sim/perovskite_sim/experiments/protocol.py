"""Canonical, immutable experiment-history protocols.

An experiment protocol is evidence about the state and history that produced a
result.  It is deliberately separate from solver tolerances and numerical
certificates: the same equations and tolerances can represent different
physical experiments when pre-bias, illumination, dwell, or sampling changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Self


ExperimentKind = Literal["jv_hysteresis", "tpv", "eqe", "suns_voc", "impedance"]
ProtocolMode = Literal["compatibility", "research_strict"]

_EXPERIMENT_KINDS = {"jv_hysteresis", "tpv", "eqe", "suns_voc", "impedance"}
_ILLUMINATION_CONDITIONS = {
    "dark",
    "baseline",
    "scaled",
    "monochromatic",
    "pulse",
}
_SCAN_AXES = {"voltage_V", "time_s", "wavelength_nm", "suns", "frequency_Hz"}
_SCAN_DIRECTIONS = {
    "ascending",
    "descending",
    "ascending_then_descending",
    "declared_order",
    "forward_time",
}
_SAMPLING_MODES = {"linear", "log", "declared", "piecewise_linear"}
_SETTLE_KINDS = {
    "finite_time",
    "residual_certified",
    "finite_time_with_certificate",
    "not_applicable",
}
_INITIAL_STATE_SOURCES = {
    "dark_equilibrium",
    "dark_equilibrium_each_sample",
    "finite_time_illuminated_preconditioned",
    "finite_time_dc_preconditioned",
    "qf_dc_candidate",
    "user_supplied_state",
}


class ExperimentProtocolError(ValueError):
    """Base class for invalid or incompatible experiment protocols."""


class ImplicitProtocolError(ExperimentProtocolError):
    """Research-strict execution was requested with implicit history."""


class ProtocolMismatchError(ExperimentProtocolError):
    """A supplied protocol does not describe the requested execution."""


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    numeric = float(value)
    if not (-float("inf") < numeric < float("inf")):
        raise ValueError(f"{field} must be finite")
    return numeric


def _optional_finite_float(value: object | None, field: str) -> float | None:
    return None if value is None else _finite_float(value, field)


def _nonnegative_optional(value: object | None, field: str) -> float | None:
    numeric = _optional_finite_float(value, field)
    if numeric is not None and numeric < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _positive_optional(value: object | None, field: str) -> float | None:
    numeric = _optional_finite_float(value, field)
    if numeric is not None and numeric <= 0.0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ExperimentProtocolError(
            f"{label} keys do not match schema; missing={missing}, extra={extra}"
        )


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise ValueError("experiment protocol contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "experiment protocol contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class IlluminationStep:
    """One declared dark/light segment in the device history."""

    phase: str
    condition: Literal["dark", "baseline", "scaled", "monochromatic", "pulse"]
    duration_s: float | None = None
    intensity_suns: float | None = None
    photon_flux_m2_s: float | None = None
    relative_generation_change: float | None = None
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("illumination phase must be a non-empty string")
        if self.condition not in _ILLUMINATION_CONDITIONS:
            raise ValueError(f"unknown illumination condition {self.condition!r}")
        object.__setattr__(
            self,
            "duration_s",
            _nonnegative_optional(self.duration_s, "illumination duration_s"),
        )
        object.__setattr__(
            self,
            "intensity_suns",
            _nonnegative_optional(self.intensity_suns, "illumination intensity_suns"),
        )
        object.__setattr__(
            self,
            "photon_flux_m2_s",
            _positive_optional(self.photon_flux_m2_s, "illumination photon_flux_m2_s"),
        )
        object.__setattr__(
            self,
            "relative_generation_change",
            _optional_finite_float(
                self.relative_generation_change,
                "illumination relative_generation_change",
            ),
        )
        if self.source_reference is not None and (
            not isinstance(self.source_reference, str)
            or not self.source_reference.strip()
        ):
            raise ValueError("illumination source_reference must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ScanProtocol:
    """Direction and rate of the controlled experimental axis."""

    axis: Literal["voltage_V", "time_s", "wavelength_nm", "suns", "frequency_Hz"]
    direction: Literal[
        "ascending",
        "descending",
        "ascending_then_descending",
        "declared_order",
        "forward_time",
    ]
    start: float
    stop: float
    rate_V_s: float | None = None

    def __post_init__(self) -> None:
        if self.axis not in _SCAN_AXES:
            raise ValueError(f"unknown scan axis {self.axis!r}")
        if self.direction not in _SCAN_DIRECTIONS:
            raise ValueError(f"unknown scan direction {self.direction!r}")
        object.__setattr__(self, "start", _finite_float(self.start, "scan start"))
        object.__setattr__(self, "stop", _finite_float(self.stop, "scan stop"))
        rate = _positive_optional(self.rate_V_s, "scan rate_V_s")
        if self.axis == "voltage_V" and rate is None:
            raise ValueError("a voltage scan requires rate_V_s")
        if self.axis != "voltage_V" and rate is not None:
            raise ValueError("rate_V_s is only valid for a voltage scan")
        object.__setattr__(self, "rate_V_s", rate)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ACExcitation:
    """Small-signal excitation and lock-in sampling contract."""

    dc_bias_V: float
    amplitude_V: float
    waveform: Literal["sine"] = "sine"
    cycles: int | None = None
    extraction_cycles: int | None = None
    points_per_cycle: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dc_bias_V", _finite_float(self.dc_bias_V, "AC dc_bias_V")
        )
        amplitude = _positive_optional(self.amplitude_V, "AC amplitude_V")
        assert amplitude is not None
        object.__setattr__(self, "amplitude_V", amplitude)
        if self.waveform != "sine":
            raise ValueError(f"unsupported AC waveform {self.waveform!r}")
        for name in ("cycles", "extraction_cycles", "points_per_cycle"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"AC {name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (
            self.cycles is not None
            and self.extraction_cycles is not None
            and self.extraction_cycles > self.cycles
        ):
            raise ValueError("AC extraction_cycles cannot exceed cycles")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class DCSettleCriterion:
    """Declared DC-state acceptance rule before a measurement starts."""

    kind: Literal[
        "finite_time",
        "residual_certified",
        "finite_time_with_certificate",
        "not_applicable",
    ]
    duration_s: float | None = None
    max_carrier_area_rate_A_m2: float | None = None
    max_ion_area_rate_A_m2: float | None = None
    max_ionic_face_current_A_m2: float | None = None
    max_face_current_spread_A_m2: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SETTLE_KINDS:
            raise ValueError(f"unknown DC settle kind {self.kind!r}")
        object.__setattr__(
            self,
            "duration_s",
            _positive_optional(self.duration_s, "DC settle duration_s"),
        )
        for name in (
            "max_carrier_area_rate_A_m2",
            "max_ion_area_rate_A_m2",
            "max_ionic_face_current_A_m2",
            "max_face_current_spread_A_m2",
        ):
            object.__setattr__(
                self,
                name,
                _positive_optional(getattr(self, name), f"DC settle {name}"),
            )
        if self.kind == "finite_time" and self.duration_s is None:
            raise ValueError("finite_time DC settling requires duration_s")
        if self.kind == "not_applicable" and any(
            getattr(self, name) is not None
            for name in (
                "duration_s",
                "max_carrier_area_rate_A_m2",
                "max_ion_area_rate_A_m2",
                "max_ionic_face_current_A_m2",
                "max_face_current_spread_A_m2",
            )
        ):
            raise ValueError("not_applicable DC settling cannot carry thresholds")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class SamplingProtocol:
    """Exact requested output axis and sampling strategy."""

    axis: Literal["voltage_V", "time_s", "wavelength_nm", "suns", "frequency_Hz"]
    mode: Literal["linear", "log", "declared", "piecewise_linear"]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.axis not in _SCAN_AXES:
            raise ValueError(f"unknown sampling axis {self.axis!r}")
        if self.mode not in _SAMPLING_MODES:
            raise ValueError(f"unknown sampling mode {self.mode!r}")
        try:
            values = tuple(
                _finite_float(value, f"sampling values[{index}]")
                for index, value in enumerate(self.values)
            )
        except TypeError as exc:
            raise TypeError("sampling values must be an iterable of finite numbers") from exc
        if not values:
            raise ValueError("sampling values must be non-empty")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        data = dict(payload)
        data["values"] = tuple(data["values"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class VocSearchProtocol:
    """State-advancing finite-time search used to establish open circuit."""

    coarse_start_V: float = 0.0
    coarse_upper_guess_factor: float = 1.5
    minimum_guess_V: float = 0.0
    coarse_points: int = 20
    coarse_dwell_s: float = 1.0e-4
    bisection_tolerance_V: float = 1.0e-3
    bisection_max_steps: int = 15
    bisection_dwell_s: float = 1.0e-4
    final_settle_s: float = 1.0e-4
    fallback: Literal["minimum_absolute_current"] = "minimum_absolute_current"
    warm_start: Literal[
        "coarse_continuation_then_lower_bracket_state"
    ] = "coarse_continuation_then_lower_bracket_state"

    def __post_init__(self) -> None:
        for name in ("coarse_start_V", "minimum_guess_V"):
            value = _finite_float(getattr(self, name), f"Voc search {name}")
            if value < 0.0:
                raise ValueError(f"Voc search {name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "coarse_upper_guess_factor",
            "coarse_dwell_s",
            "bisection_tolerance_V",
            "bisection_dwell_s",
            "final_settle_s",
        ):
            value = _positive_optional(getattr(self, name), f"Voc search {name}")
            assert value is not None
            object.__setattr__(self, name, value)
        for name, minimum in (("coarse_points", 2), ("bisection_max_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"Voc search {name} must be an integer")
            if value < minimum:
                raise ValueError(f"Voc search {name} must be >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.fallback != "minimum_absolute_current":
            raise ValueError(f"unsupported Voc search fallback {self.fallback!r}")
        if self.warm_start != "coarse_continuation_then_lower_bracket_state":
            raise ValueError(f"unsupported Voc search warm_start {self.warm_start!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Complete physical history and measurement sampling for one run."""

    experiment: ExperimentKind
    initial_state_source: str
    pre_bias_V: float | None
    soak_duration_s: float | None
    dwell_duration_s: float | None
    illumination_history: tuple[IlluminationStep, ...]
    temperature_K: float
    scan: ScanProtocol
    ac_excitation: ACExcitation | None
    dc_settle: DCSettleCriterion
    sampling: SamplingProtocol
    voc_search: VocSearchProtocol | None = None
    implicit_legacy_protocol: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.experiment not in _EXPERIMENT_KINDS:
            raise ValueError(f"unknown experiment kind {self.experiment!r}")
        if self.initial_state_source not in _INITIAL_STATE_SOURCES:
            raise ValueError(
                f"unknown initial_state_source {self.initial_state_source!r}"
            )
        object.__setattr__(
            self,
            "pre_bias_V",
            _optional_finite_float(self.pre_bias_V, "pre_bias_V"),
        )
        object.__setattr__(
            self,
            "soak_duration_s",
            _nonnegative_optional(self.soak_duration_s, "soak_duration_s"),
        )
        object.__setattr__(
            self,
            "dwell_duration_s",
            _nonnegative_optional(self.dwell_duration_s, "dwell_duration_s"),
        )
        history = tuple(self.illumination_history)
        if not history or not all(isinstance(step, IlluminationStep) for step in history):
            raise TypeError("illumination_history must contain IlluminationStep values")
        object.__setattr__(self, "illumination_history", history)
        temperature = _finite_float(self.temperature_K, "temperature_K")
        if temperature <= 0.0:
            raise ValueError("temperature_K must be positive")
        object.__setattr__(self, "temperature_K", temperature)
        if not isinstance(self.scan, ScanProtocol):
            raise TypeError("scan must be a ScanProtocol")
        if self.ac_excitation is not None and not isinstance(
            self.ac_excitation, ACExcitation
        ):
            raise TypeError("ac_excitation must be an ACExcitation or None")
        if not isinstance(self.dc_settle, DCSettleCriterion):
            raise TypeError("dc_settle must be a DCSettleCriterion")
        if not isinstance(self.sampling, SamplingProtocol):
            raise TypeError("sampling must be a SamplingProtocol")
        if self.voc_search is not None and not isinstance(
            self.voc_search, VocSearchProtocol
        ):
            raise TypeError("voc_search must be a VocSearchProtocol or None")
        if self.experiment in ("tpv", "suns_voc") and self.voc_search is None:
            raise ValueError(f"{self.experiment} protocol requires voc_search")
        if self.experiment not in ("tpv", "suns_voc") and self.voc_search is not None:
            raise ValueError(f"{self.experiment} protocol cannot carry voc_search")
        if not isinstance(self.implicit_legacy_protocol, bool):
            raise TypeError("implicit_legacy_protocol must be boolean")
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported experiment protocol schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Return plain JSON-compatible values without mutable references."""

        return _json_ready(self)

    def canonical_json(self) -> str:
        """Return the stable byte-for-byte representation used for hashing."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json(self) -> str:
        """Alias for the canonical JSON representation."""

        return self.canonical_json()

    @property
    def protocol_hash(self) -> str:
        """SHA-256 of :meth:`canonical_json`."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def sha256(self) -> str:
        """Conventional short alias for :attr:`protocol_hash`."""

        return self.protocol_hash

    def as_explicit(self) -> Self:
        """Acknowledge a generated compatibility protocol for strict replay."""

        return dataclasses.replace(self, implicit_legacy_protocol=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("experiment protocol payload must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_exact_keys(payload, expected, cls.__name__)
        data = dict(payload)
        history = data["illumination_history"]
        if not isinstance(history, (list, tuple)):
            raise TypeError("illumination_history must be a JSON array")
        data["illumination_history"] = tuple(
            IlluminationStep.from_dict(step) for step in history
        )
        data["scan"] = ScanProtocol.from_dict(data["scan"])
        if data["ac_excitation"] is not None:
            data["ac_excitation"] = ACExcitation.from_dict(data["ac_excitation"])
        data["dc_settle"] = DCSettleCriterion.from_dict(data["dc_settle"])
        data["sampling"] = SamplingProtocol.from_dict(data["sampling"])
        if data["voc_search"] is not None:
            data["voc_search"] = VocSearchProtocol.from_dict(data["voc_search"])
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("experiment protocol JSON must contain an object")
        return cls.from_dict(parsed)


def resolve_experiment_protocol(
    supplied: ExperimentProtocol | None,
    expected_legacy: ExperimentProtocol,
    *,
    mode: ProtocolMode = "compatibility",
) -> ExperimentProtocol:
    """Validate a declaration against the actual execution and apply its gate.

    Compatibility calls receive ``expected_legacy`` and are visibly marked as
    implicit.  An explicit protocol must match every execution-defining field;
    only the provenance flag may differ.  This prevents a caller from attaching
    a scientifically attractive history to a calculation that did not execute
    that history.
    """

    if mode not in ("compatibility", "research_strict"):
        raise ValueError(
            "protocol mode must be 'compatibility' or 'research_strict', got "
            f"{mode!r}"
        )
    if not isinstance(expected_legacy, ExperimentProtocol):
        raise TypeError("expected_legacy must be an ExperimentProtocol")
    if not expected_legacy.implicit_legacy_protocol:
        raise ValueError("expected_legacy must carry implicit_legacy_protocol=True")

    resolved = expected_legacy if supplied is None else supplied
    if not isinstance(resolved, ExperimentProtocol):
        raise TypeError("experiment_protocol must be an ExperimentProtocol")
    if supplied is not None:
        expected = expected_legacy.to_dict()
        actual = supplied.to_dict()
        expected.pop("implicit_legacy_protocol")
        actual.pop("implicit_legacy_protocol")
        mismatches = tuple(
            name for name in sorted(expected) if actual[name] != expected[name]
        )
        if mismatches:
            raise ProtocolMismatchError(
                "experiment_protocol does not match the requested execution; "
                f"mismatched fields: {', '.join(mismatches)}"
            )
    if mode == "research_strict" and resolved.implicit_legacy_protocol:
        raise ImplicitProtocolError(
            "research_strict protocol mode requires an explicit experiment "
            "history; the compatibility call generated "
            "implicit_legacy_protocol=True"
        )
    return resolved


__all__ = [
    "ACExcitation",
    "DCSettleCriterion",
    "ExperimentProtocol",
    "ExperimentProtocolError",
    "IlluminationStep",
    "ImplicitProtocolError",
    "ProtocolMismatchError",
    "ProtocolMode",
    "SamplingProtocol",
    "ScanProtocol",
    "VocSearchProtocol",
    "resolve_experiment_protocol",
]

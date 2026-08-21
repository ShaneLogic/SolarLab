"""Fail-closed certificates for RHS regularization-width ladders.

The ladder is intentionally separate from grid/tolerance certification.  It
holds the device, physical protocol, grid, and solver tolerances fixed while
only the declared RHS transition widths change.  Solver speed is recorded as
evidence but is never a promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from perovskite_sim.physics.regularization import RHSRegularization
from perovskite_sim.validation.numerical_certificate import (
    canonical_json_bytes as _shared_canonical_json_bytes,
    content_sha256 as _shared_content_sha256,
)


REGULARIZATION_CERTIFICATE_SCHEMA = "rhs-regularization-certificate-v1"
REGULARIZATION_STUDY_SCHEMA = "rhs-regularization-study-v1"
REGULARIZATION_EVALUATOR_VERSION = "rhs-regularization-evaluator-v1"
REGULARIZATION_LADDER_FACTORS = (1.0, 0.5, 0.25, 0.0)
OBSERVABLE_RELATIVE_LIMIT = 0.005

CertificateStatus = Literal["failed", "partial", "certified"]
RungOutcome = Literal["completed", "failed", "missing"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATE_BLOCKS = {"n", "p", "P", "P_neg", "interface_state"}


class RegularizationCertificateError(ValueError):
    """A regularization study, rung, or certificate is malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return _shared_canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise RegularizationCertificateError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def _content_sha256(value: Any) -> str:
    try:
        return _shared_content_sha256(value)
    except (TypeError, ValueError) as exc:
        raise RegularizationCertificateError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def _identifier(value: Any, where: str) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        raise RegularizationCertificateError(
            f"{where} must match {_IDENTIFIER.pattern!r}, got {text!r}"
        )
    return text


def _sha256(value: Any, where: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise RegularizationCertificateError(
            f"{where} must be a lowercase SHA-256 digest"
        )
    return text


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RegularizationCertificateError(f"{where} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RegularizationCertificateError(f"{where} must be finite") from exc
    if not math.isfinite(number):
        raise RegularizationCertificateError(f"{where} must be finite")
    return 0.0 if number == 0.0 else number


def _nonnegative(value: Any, where: str) -> float:
    number = _finite(value, where)
    if number < 0.0:
        raise RegularizationCertificateError(f"{where} must be finite and non-negative")
    return number


def _count(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RegularizationCertificateError(f"{where} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RegularizationCertificateError(
            f"{where} must be a non-negative integer"
        ) from exc
    if number != value or number < 0:
        raise RegularizationCertificateError(f"{where} must be a non-negative integer")
    return number


def _optional_count(value: Any, where: str) -> int | None:
    return None if value is None else _count(value, where)


def _optional_nonnegative(value: Any, where: str) -> float | None:
    return None if value is None else _nonnegative(value, where)


def _policy_to_dict(policy: RHSRegularization) -> dict[str, float]:
    return {
        "interface_density_width_m3": policy.interface_density_width_m3,
        "poole_frenkel_field_width_V_m": (policy.poole_frenkel_field_width_V_m),
        "te_cap_relative_width": policy.te_cap_relative_width,
    }


def _policy_from_dict(raw: Mapping[str, Any]) -> RHSRegularization:
    expected = {
        "interface_density_width_m3",
        "poole_frenkel_field_width_V_m",
        "te_cap_relative_width",
    }
    if set(raw) != expected:
        raise RegularizationCertificateError(
            "regularization policy must contain exactly " + ", ".join(sorted(expected))
        )
    if any(isinstance(raw[name], bool) for name in expected):
        raise RegularizationCertificateError(
            "regularization policy widths cannot be boolean"
        )
    try:
        return RHSRegularization(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise RegularizationCertificateError(
            f"invalid regularization policy: {exc}"
        ) from exc


def policy_for_factor(
    base_policy: RHSRegularization,
    factor: float,
) -> RHSRegularization:
    """Return the exact policy required at one fixed ladder factor."""

    if not isinstance(base_policy, RHSRegularization):
        raise RegularizationCertificateError("base_policy must be an RHSRegularization")
    factor = _finite(factor, "regularization factor")
    if factor not in REGULARIZATION_LADDER_FACTORS:
        raise RegularizationCertificateError(
            f"regularization factor must be one of {REGULARIZATION_LADDER_FACTORS}"
        )
    if factor == 0.0:
        return RHSRegularization()
    return base_policy.refined(factor)


@dataclass(frozen=True)
class CanonicalInput:
    """A JSON object retained together with its canonical content hash."""

    canonical_json: str
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise RegularizationCertificateError("canonical input must be JSON text")
        try:
            value = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise RegularizationCertificateError(
                "canonical input is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise RegularizationCertificateError(
                "canonical input must contain a JSON object"
            )
        canonical = _canonical_json_bytes(value).decode("ascii")
        digest = _content_sha256(value)
        if self.sha256 and _sha256(self.sha256, "canonical input SHA-256") != digest:
            raise RegularizationCertificateError("canonical input SHA-256 mismatch")
        object.__setattr__(self, "canonical_json", canonical)
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_value(cls, value: Any) -> "CanonicalInput":
        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise RegularizationCertificateError(
                "canonical study inputs must be mappings or expose to_dict()"
            )
        return cls(_canonical_json_bytes(dict(value)).decode("ascii"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CanonicalInput":
        if set(raw) != {"payload", "sha256"}:
            raise RegularizationCertificateError(
                "canonical input must contain exactly payload and sha256"
            )
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise RegularizationCertificateError(
                "canonical input payload must be a mapping"
            )
        return cls(
            _canonical_json_bytes(dict(payload)).decode("ascii"),
            str(raw["sha256"]),
        )

    @property
    def value(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.value, "sha256": self.sha256}


@dataclass(frozen=True)
class MetricSpec:
    """Absolute and comparative gates for an error-like scalar metric."""

    name: str
    units: str
    upper_limit: float
    non_worsening_atol: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "metric name"))
        if not isinstance(self.units, str) or not self.units:
            raise RegularizationCertificateError(
                f"metric {self.name}: units must be a non-empty string"
            )
        object.__setattr__(
            self,
            "upper_limit",
            _nonnegative(self.upper_limit, f"metric {self.name}.upper_limit"),
        )
        object.__setattr__(
            self,
            "non_worsening_atol",
            _nonnegative(
                self.non_worsening_atol,
                f"metric {self.name}.non_worsening_atol",
            ),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricSpec":
        if set(raw) != {"name", "non_worsening_atol", "units", "upper_limit"}:
            raise RegularizationCertificateError(
                "metric specification has incomplete or unknown fields"
            )
        return cls(
            name=str(raw["name"]),
            units=str(raw["units"]),
            upper_limit=raw["upper_limit"],
            non_worsening_atol=raw["non_worsening_atol"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "non_worsening_atol": self.non_worsening_atol,
            "units": self.units,
            "upper_limit": self.upper_limit,
        }


@dataclass(frozen=True)
class QualityGateSpec:
    """A reproducible absolute physical-health gate."""

    name: str
    units: str
    operator: Literal["le", "lt", "ge", "gt", "eq"]
    limit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "quality gate name"))
        if not isinstance(self.units, str) or not self.units:
            raise RegularizationCertificateError(
                f"quality gate {self.name}: units must be a non-empty string"
            )
        if self.operator not in {"le", "lt", "ge", "gt", "eq"}:
            raise RegularizationCertificateError(
                f"quality gate {self.name}: unsupported operator {self.operator!r}"
            )
        object.__setattr__(
            self, "limit", _finite(self.limit, f"quality gate {self.name}.limit")
        )

    def passes(self, value: float) -> bool:
        if self.operator == "le":
            return value <= self.limit
        if self.operator == "lt":
            return value < self.limit
        if self.operator == "ge":
            return value >= self.limit
        if self.operator == "gt":
            return value > self.limit
        return value == self.limit

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QualityGateSpec":
        if set(raw) != {"limit", "name", "operator", "units"}:
            raise RegularizationCertificateError(
                "quality gate specification has incomplete or unknown fields"
            )
        return cls(
            name=str(raw["name"]),
            units=str(raw["units"]),
            operator=str(raw["operator"]),  # type: ignore[arg-type]
            limit=raw["limit"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "name": self.name,
            "operator": self.operator,
            "units": self.units,
        }


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    units: str
    relative_floor: float = 1.0e-30

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "observable name"))
        if not isinstance(self.units, str) or not self.units:
            raise RegularizationCertificateError(
                f"observable {self.name}: units must be a non-empty string"
            )
        floor = _finite(self.relative_floor, f"observable {self.name}.relative_floor")
        if floor <= 0.0:
            raise RegularizationCertificateError(
                f"observable {self.name}.relative_floor must be positive"
            )
        object.__setattr__(self, "relative_floor", floor)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservableSpec":
        if set(raw) != {"name", "relative_floor", "units"}:
            raise RegularizationCertificateError(
                "observable specification has incomplete or unknown fields"
            )
        return cls(
            name=str(raw["name"]),
            units=str(raw["units"]),
            relative_floor=raw["relative_floor"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relative_floor": self.relative_floor,
            "units": self.units,
        }


@dataclass(frozen=True)
class RegularizationStudy:
    """Pre-registered fixed context and gates for one width ladder."""

    base_policy: RHSRegularization
    protocol: CanonicalInput
    config: CanonicalInput
    grid: CanonicalInput
    tolerances: CanonicalInput
    observables: tuple[ObservableSpec, ...]
    residuals: tuple[MetricSpec, ...]
    conservation_errors: tuple[MetricSpec, ...]
    physical_health_gates: tuple[QualityGateSpec, ...]
    state_blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_policy, RHSRegularization):
            raise RegularizationCertificateError(
                "base_policy must be an RHSRegularization"
            )
        if not self.base_policy.active:
            raise RegularizationCertificateError(
                "base_policy must enable at least one positive transition width"
            )
        widths = tuple(_policy_to_dict(self.base_policy).values())
        if any(value == 0.0 and math.copysign(1.0, value) < 0.0 for value in widths):
            raise RegularizationCertificateError(
                "base_policy cannot contain signed negative zero widths"
            )
        positive_widths = tuple(value for value in widths if value > 0.0)
        if any(
            not 0.0 < value * 0.25 < value * 0.5 < value for value in positive_widths
        ):
            raise RegularizationCertificateError(
                "base_policy is too small for distinct positive ladder rungs"
            )
        for name in ("protocol", "config", "grid", "tolerances"):
            if not isinstance(getattr(self, name), CanonicalInput):
                raise RegularizationCertificateError(
                    f"study {name} must be a CanonicalInput"
                )
        raw_specs = {
            "observables": tuple(self.observables),
            "residuals": tuple(self.residuals),
            "conservation_errors": tuple(self.conservation_errors),
        }
        for name, specs in raw_specs.items():
            expected_type = ObservableSpec if name == "observables" else MetricSpec
            if any(not isinstance(spec, expected_type) for spec in specs):
                raise RegularizationCertificateError(
                    f"study {name} contains an invalid metric specification"
                )
            object.__setattr__(
                self, name, tuple(sorted(specs, key=lambda item: item.name))
            )
        raw_gates = tuple(self.physical_health_gates)
        if any(not isinstance(gate, QualityGateSpec) for gate in raw_gates):
            raise RegularizationCertificateError(
                "physical_health_gates must contain QualityGateSpec values"
            )
        gates = tuple(sorted(raw_gates, key=lambda item: item.name))
        object.__setattr__(self, "physical_health_gates", gates)
        blocks = tuple(
            sorted(_identifier(value, "state block") for value in self.state_blocks)
        )
        object.__setattr__(self, "state_blocks", blocks)
        for name, specs in (
            ("observables", self.observables),
            ("residuals", self.residuals),
            ("conservation_errors", self.conservation_errors),
        ):
            if not specs:
                raise RegularizationCertificateError(
                    f"study must pre-register at least one {name} metric"
                )
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise RegularizationCertificateError(
                    f"study {name} metric names must be unique"
                )
        gate_names = [gate.name for gate in gates]
        if not gates or len(gate_names) != len(set(gate_names)):
            raise RegularizationCertificateError(
                "physical health gate names must be non-empty and unique"
            )
        if not blocks or len(blocks) != len(set(blocks)):
            raise RegularizationCertificateError(
                "state block names must be non-empty and unique"
            )
        unknown_blocks = set(blocks) - _STATE_BLOCKS
        if unknown_blocks or not {"n", "p"}.issubset(blocks):
            raise RegularizationCertificateError(
                "state blocks must use known solver blocks and include n and p"
            )

    @classmethod
    def from_values(
        cls,
        *,
        base_policy: RHSRegularization,
        protocol: Any,
        config: Any,
        grid: Any,
        tolerances: Any,
        observables: Sequence[ObservableSpec],
        residuals: Sequence[MetricSpec],
        conservation_errors: Sequence[MetricSpec],
        physical_health_gates: Sequence[QualityGateSpec],
        state_blocks: Sequence[str],
    ) -> "RegularizationStudy":
        return cls(
            base_policy=base_policy,
            protocol=CanonicalInput.from_value(protocol),
            config=CanonicalInput.from_value(config),
            grid=CanonicalInput.from_value(grid),
            tolerances=CanonicalInput.from_value(tolerances),
            observables=tuple(observables),
            residuals=tuple(residuals),
            conservation_errors=tuple(conservation_errors),
            physical_health_gates=tuple(physical_health_gates),
            state_blocks=tuple(state_blocks),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegularizationStudy":
        expected = {
            "base_policy",
            "config",
            "conservation_errors",
            "evaluator_version",
            "grid",
            "ladder_factors",
            "observables",
            "observable_relative_limit",
            "physical_health_gates",
            "protocol",
            "residuals",
            "schema",
            "state_blocks",
            "tolerances",
        }
        if set(raw) != expected:
            raise RegularizationCertificateError(
                "regularization study has incomplete or unknown fields"
            )
        if raw["schema"] != REGULARIZATION_STUDY_SCHEMA:
            raise RegularizationCertificateError(
                "unsupported regularization study schema"
            )
        if raw["evaluator_version"] != REGULARIZATION_EVALUATOR_VERSION:
            raise RegularizationCertificateError(
                "unsupported regularization evaluator version"
            )
        if tuple(raw["ladder_factors"]) != REGULARIZATION_LADDER_FACTORS:
            raise RegularizationCertificateError(
                "regularization ladder factors changed"
            )
        if raw["observable_relative_limit"] != OBSERVABLE_RELATIVE_LIMIT:
            raise RegularizationCertificateError(
                "regularization observable threshold changed"
            )
        return cls(
            base_policy=_policy_from_dict(_mapping(raw["base_policy"], "base_policy")),
            protocol=CanonicalInput.from_dict(_mapping(raw["protocol"], "protocol")),
            config=CanonicalInput.from_dict(_mapping(raw["config"], "config")),
            grid=CanonicalInput.from_dict(_mapping(raw["grid"], "grid")),
            tolerances=CanonicalInput.from_dict(
                _mapping(raw["tolerances"], "tolerances")
            ),
            observables=tuple(
                ObservableSpec.from_dict(_mapping(item, "observable"))
                for item in _sequence(raw["observables"], "observables")
            ),
            residuals=tuple(
                MetricSpec.from_dict(_mapping(item, "residual"))
                for item in _sequence(raw["residuals"], "residuals")
            ),
            conservation_errors=tuple(
                MetricSpec.from_dict(_mapping(item, "conservation error"))
                for item in _sequence(raw["conservation_errors"], "conservation_errors")
            ),
            physical_health_gates=tuple(
                QualityGateSpec.from_dict(_mapping(item, "physical health gate"))
                for item in _sequence(
                    raw["physical_health_gates"], "physical_health_gates"
                )
            ),
            state_blocks=tuple(
                str(item) for item in _sequence(raw["state_blocks"], "state_blocks")
            ),
        )

    @property
    def definition_sha256(self) -> str:
        return _content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_policy": _policy_to_dict(self.base_policy),
            "config": self.config.to_dict(),
            "conservation_errors": [
                spec.to_dict() for spec in self.conservation_errors
            ],
            "evaluator_version": REGULARIZATION_EVALUATOR_VERSION,
            "grid": self.grid.to_dict(),
            "ladder_factors": list(REGULARIZATION_LADDER_FACTORS),
            "observables": [spec.to_dict() for spec in self.observables],
            "observable_relative_limit": OBSERVABLE_RELATIVE_LIMIT,
            "physical_health_gates": [
                gate.to_dict() for gate in self.physical_health_gates
            ],
            "protocol": self.protocol.to_dict(),
            "residuals": [spec.to_dict() for spec in self.residuals],
            "schema": REGULARIZATION_STUDY_SCHEMA,
            "state_blocks": list(self.state_blocks),
            "tolerances": self.tolerances.to_dict(),
        }


@dataclass(frozen=True)
class MetricValue:
    """A finite scalar or shaped numeric observable."""

    name: str
    values: tuple[float, ...]
    shape: tuple[int, ...] = ()
    units: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "metric name"))
        shape = tuple(self.shape)
        if any(
            isinstance(item, bool) or int(item) != item or int(item) < 1
            for item in shape
        ):
            raise RegularizationCertificateError(
                f"metric {self.name}: shape entries must be positive integers"
            )
        shape = tuple(int(item) for item in shape)
        expected = math.prod(shape) if shape else 1
        values = tuple(_finite(item, f"metric {self.name}") for item in self.values)
        if len(values) != expected:
            raise RegularizationCertificateError(
                f"metric {self.name}: shape {shape} requires {expected} values"
            )
        if not isinstance(self.units, str) or not self.units:
            raise RegularizationCertificateError(
                f"metric {self.name}: units must be a non-empty string"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        units: str,
    ) -> "MetricValue":
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise RegularizationCertificateError(
                f"metric {name}: must be numeric"
            ) from exc
        if np.issubdtype(array.dtype, np.bool_) or np.iscomplexobj(array):
            raise RegularizationCertificateError(
                f"metric {name}: must be real numeric evidence"
            )
        if array.ndim:
            if array.size == 0:
                raise RegularizationCertificateError(
                    f"metric {name}: arrays cannot be empty"
                )
            try:
                numeric = np.asarray(array, dtype=float)
            except (TypeError, ValueError) as exc:
                raise RegularizationCertificateError(
                    f"metric {name}: must be numeric"
                ) from exc
            return cls(
                name=name,
                values=tuple(float(item) for item in numeric.ravel(order="C")),
                shape=tuple(int(item) for item in numeric.shape),
                units=units,
            )
        try:
            scalar = float(array)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RegularizationCertificateError(
                f"metric {name}: must be real numeric evidence"
            ) from exc
        return cls(name=name, values=(scalar,), units=units)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricValue":
        if set(raw) != {"name", "shape", "units", "values"}:
            raise RegularizationCertificateError(
                "metric value has incomplete or unknown fields"
            )
        return cls(
            name=str(raw["name"]),
            values=tuple(_sequence(raw["values"], "metric values")),
            shape=tuple(_sequence(raw["shape"], "metric shape")),
            units=str(raw["units"]),
        )

    @property
    def scalar(self) -> float:
        if self.shape:
            raise RegularizationCertificateError(
                f"metric {self.name} must be scalar for this gate"
            )
        return self.values[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "units": self.units,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class AppliedRunContext:
    """Policy and hashes reported by the result that actually ran."""

    policy: RHSRegularization
    protocol_sha256: str
    config_sha256: str
    grid_sha256: str
    tolerances_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RHSRegularization):
            raise RegularizationCertificateError(
                "applied policy must be an RHSRegularization"
            )
        for name in (
            "protocol_sha256",
            "config_sha256",
            "grid_sha256",
            "tolerances_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AppliedRunContext":
        expected = {
            "config_sha256",
            "grid_sha256",
            "policy",
            "protocol_sha256",
            "tolerances_sha256",
        }
        if set(raw) != expected:
            raise RegularizationCertificateError(
                "applied run context has incomplete or unknown fields"
            )
        return cls(
            policy=_policy_from_dict(_mapping(raw["policy"], "applied policy")),
            protocol_sha256=str(raw["protocol_sha256"]),
            config_sha256=str(raw["config_sha256"]),
            grid_sha256=str(raw["grid_sha256"]),
            tolerances_sha256=str(raw["tolerances_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "grid_sha256": self.grid_sha256,
            "policy": _policy_to_dict(self.policy),
            "protocol_sha256": self.protocol_sha256,
            "tolerances_sha256": self.tolerances_sha256,
        }


@dataclass(frozen=True)
class SolverWork:
    """Solver effort; missing counts are permitted only for failed/missing rungs."""

    nfev: int | None
    njev: int | None
    nlu: int | None
    wall_time_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nfev", _optional_count(self.nfev, "nfev"))
        object.__setattr__(self, "njev", _optional_count(self.njev, "njev"))
        object.__setattr__(self, "nlu", _optional_count(self.nlu, "nlu"))
        object.__setattr__(
            self,
            "wall_time_s",
            _optional_nonnegative(self.wall_time_s, "wall_time_s"),
        )

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (self.nfev, self.njev, self.nlu, self.wall_time_s)
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SolverWork":
        if set(raw) != {"nfev", "njev", "nlu", "wall_time_s"}:
            raise RegularizationCertificateError(
                "solver work has incomplete or unknown fields"
            )
        return cls(
            nfev=raw["nfev"],
            njev=raw["njev"],
            nlu=raw["nlu"],
            wall_time_s=raw["wall_time_s"],
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True)
class RegularizationMeasurement:
    """Complete output evidence from one successfully executed rung."""

    applied: AppliedRunContext
    observables: tuple[MetricValue, ...]
    residuals: tuple[MetricValue, ...]
    conservation_errors: tuple[MetricValue, ...]
    minimum_trial_state_m3: tuple[MetricValue, ...]
    terminal_minimum_state_m3: tuple[MetricValue, ...]
    physical_health: tuple[MetricValue, ...]
    solver_accepted: bool
    negative_trial_count: int
    nonfinite_event_count: int
    work: SolverWork

    def __post_init__(self) -> None:
        if not isinstance(self.applied, AppliedRunContext):
            raise RegularizationCertificateError(
                "measurement applied context must be AppliedRunContext"
            )
        for name in (
            "observables",
            "residuals",
            "conservation_errors",
            "minimum_trial_state_m3",
            "terminal_minimum_state_m3",
            "physical_health",
        ):
            metrics = tuple(getattr(self, name))
            if any(not isinstance(metric, MetricValue) for metric in metrics):
                raise RegularizationCertificateError(
                    f"{name} must contain MetricValue evidence"
                )
            names = [metric.name for metric in metrics]
            if len(names) != len(set(names)):
                raise RegularizationCertificateError(
                    f"{name} metric names must be unique"
                )
            object.__setattr__(
                self, name, tuple(sorted(metrics, key=lambda metric: metric.name))
            )
        if not isinstance(self.solver_accepted, bool):
            raise RegularizationCertificateError("solver_accepted must be boolean")
        object.__setattr__(
            self,
            "negative_trial_count",
            _count(self.negative_trial_count, "negative_trial_count"),
        )
        object.__setattr__(
            self,
            "nonfinite_event_count",
            _count(self.nonfinite_event_count, "nonfinite_event_count"),
        )
        if not isinstance(self.work, SolverWork) or not self.work.complete:
            raise RegularizationCertificateError(
                "completed measurement must record nfev, njev, nlu, and wall_time_s"
            )
        for name in ("minimum_trial_state_m3", "terminal_minimum_state_m3"):
            metrics = getattr(self, name)
            if not metrics:
                raise RegularizationCertificateError(f"{name} must be non-empty")
            for metric in metrics:
                if metric.shape or metric.units != "m^-3":
                    raise RegularizationCertificateError(
                        f"{name} metrics must be scalar and use m^-3"
                    )

    @classmethod
    def from_values(
        cls,
        study: RegularizationStudy,
        *,
        applied: AppliedRunContext,
        observables: Mapping[str, Any],
        residuals: Mapping[str, Any],
        conservation_errors: Mapping[str, Any],
        minimum_trial_state_m3: Mapping[str, Any],
        terminal_minimum_state_m3: Mapping[str, Any],
        physical_health: Mapping[str, Any],
        solver_accepted: bool,
        negative_trial_count: int,
        nonfinite_event_count: int,
        nfev: int,
        njev: int,
        nlu: int,
        wall_time_s: float,
    ) -> "RegularizationMeasurement":
        return cls(
            applied=applied,
            observables=_metrics_from_values(study.observables, observables),
            residuals=_metrics_from_values(study.residuals, residuals),
            conservation_errors=_metrics_from_values(
                study.conservation_errors, conservation_errors
            ),
            minimum_trial_state_m3=tuple(
                MetricValue.from_value(name, value, units="m^-3")
                for name, value in sorted(minimum_trial_state_m3.items())
            ),
            terminal_minimum_state_m3=tuple(
                MetricValue.from_value(name, value, units="m^-3")
                for name, value in sorted(terminal_minimum_state_m3.items())
            ),
            physical_health=_metrics_from_values(
                study.physical_health_gates, physical_health
            ),
            solver_accepted=solver_accepted,
            negative_trial_count=negative_trial_count,
            nonfinite_event_count=nonfinite_event_count,
            work=SolverWork(nfev, njev, nlu, wall_time_s),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegularizationMeasurement":
        expected = {
            "applied",
            "conservation_errors",
            "minimum_trial_state_m3",
            "negative_trial_count",
            "nonfinite_event_count",
            "observables",
            "physical_health",
            "residuals",
            "solver_accepted",
            "terminal_minimum_state_m3",
            "work",
        }
        if set(raw) != expected:
            raise RegularizationCertificateError(
                "regularization measurement has incomplete or unknown fields"
            )
        return cls(
            applied=AppliedRunContext.from_dict(_mapping(raw["applied"], "applied")),
            observables=_metrics_from_dict(raw["observables"], "observables"),
            residuals=_metrics_from_dict(raw["residuals"], "residuals"),
            conservation_errors=_metrics_from_dict(
                raw["conservation_errors"], "conservation_errors"
            ),
            minimum_trial_state_m3=_metrics_from_dict(
                raw["minimum_trial_state_m3"], "minimum_trial_state_m3"
            ),
            terminal_minimum_state_m3=_metrics_from_dict(
                raw["terminal_minimum_state_m3"], "terminal_minimum_state_m3"
            ),
            physical_health=tuple(
                MetricValue.from_dict(_mapping(item, "physical health"))
                for item in _sequence(raw["physical_health"], "physical_health")
            ),
            solver_accepted=raw["solver_accepted"],
            negative_trial_count=raw["negative_trial_count"],
            nonfinite_event_count=raw["nonfinite_event_count"],
            work=SolverWork.from_dict(_mapping(raw["work"], "work")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied.to_dict(),
            "conservation_errors": [
                metric.to_dict() for metric in self.conservation_errors
            ],
            "minimum_trial_state_m3": [
                metric.to_dict() for metric in self.minimum_trial_state_m3
            ],
            "negative_trial_count": self.negative_trial_count,
            "nonfinite_event_count": self.nonfinite_event_count,
            "observables": [metric.to_dict() for metric in self.observables],
            "physical_health": [metric.to_dict() for metric in self.physical_health],
            "residuals": [metric.to_dict() for metric in self.residuals],
            "solver_accepted": self.solver_accepted,
            "terminal_minimum_state_m3": [
                metric.to_dict() for metric in self.terminal_minimum_state_m3
            ],
            "work": self.work.to_dict(),
        }


@dataclass(frozen=True)
class RegularizationRung:
    """One width factor plus its immutable declared context and evidence."""

    factor: float
    policy: RHSRegularization
    protocol_sha256: str
    config_sha256: str
    grid_sha256: str
    tolerances_sha256: str
    outcome: RungOutcome
    measurement: RegularizationMeasurement | None
    failure_reason: str | None = None
    failed_work: SolverWork | None = None

    def __post_init__(self) -> None:
        factor = _finite(self.factor, "rung factor")
        if factor not in REGULARIZATION_LADDER_FACTORS:
            raise RegularizationCertificateError(
                f"rung factor must be one of {REGULARIZATION_LADDER_FACTORS}"
            )
        object.__setattr__(self, "factor", factor)
        if not isinstance(self.policy, RHSRegularization):
            raise RegularizationCertificateError(
                "rung policy must be an RHSRegularization"
            )
        for name in (
            "protocol_sha256",
            "config_sha256",
            "grid_sha256",
            "tolerances_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.outcome not in {"completed", "failed", "missing"}:
            raise RegularizationCertificateError(
                f"unknown rung outcome {self.outcome!r}"
            )
        if self.outcome == "completed":
            if not isinstance(self.measurement, RegularizationMeasurement):
                raise RegularizationCertificateError(
                    "completed rung requires a complete measurement"
                )
            if self.failure_reason is not None or self.failed_work is not None:
                raise RegularizationCertificateError(
                    "completed rung cannot carry failure evidence"
                )
        else:
            if self.measurement is not None:
                raise RegularizationCertificateError(
                    "failed or missing rung cannot carry completed measurement"
                )
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise RegularizationCertificateError(
                    "failed or missing rung requires a failure reason"
                )
            if self.failed_work is not None and not isinstance(
                self.failed_work, SolverWork
            ):
                raise RegularizationCertificateError(
                    "failed_work must be SolverWork or None"
                )

    @classmethod
    def completed(
        cls,
        factor: float,
        measurement: RegularizationMeasurement,
    ) -> "RegularizationRung":
        if not isinstance(measurement, RegularizationMeasurement):
            raise RegularizationCertificateError(
                "completed rung requires RegularizationMeasurement"
            )
        applied = measurement.applied
        return cls(
            factor=factor,
            policy=applied.policy,
            protocol_sha256=applied.protocol_sha256,
            config_sha256=applied.config_sha256,
            grid_sha256=applied.grid_sha256,
            tolerances_sha256=applied.tolerances_sha256,
            outcome="completed",
            measurement=measurement,
        )

    @classmethod
    def failed(
        cls,
        study: RegularizationStudy,
        factor: float,
        reason: str,
        *,
        work: SolverWork | None = None,
    ) -> "RegularizationRung":
        return cls(
            factor=factor,
            policy=policy_for_factor(study.base_policy, factor),
            protocol_sha256=study.protocol.sha256,
            config_sha256=study.config.sha256,
            grid_sha256=study.grid.sha256,
            tolerances_sha256=study.tolerances.sha256,
            outcome="failed",
            measurement=None,
            failure_reason=reason,
            failed_work=work,
        )

    @classmethod
    def missing(cls, study: RegularizationStudy, factor: float) -> "RegularizationRung":
        return cls(
            factor=factor,
            policy=policy_for_factor(study.base_policy, factor),
            protocol_sha256=study.protocol.sha256,
            config_sha256=study.config.sha256,
            grid_sha256=study.grid.sha256,
            tolerances_sha256=study.tolerances.sha256,
            outcome="missing",
            measurement=None,
            failure_reason="rung result is missing",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegularizationRung":
        expected = {
            "config_sha256",
            "factor",
            "failed_work",
            "failure_reason",
            "grid_sha256",
            "measurement",
            "outcome",
            "policy",
            "protocol_sha256",
            "tolerances_sha256",
        }
        if set(raw) != expected:
            raise RegularizationCertificateError(
                "regularization rung has incomplete or unknown fields"
            )
        measurement = raw["measurement"]
        failed_work = raw["failed_work"]
        return cls(
            factor=raw["factor"],
            policy=_policy_from_dict(_mapping(raw["policy"], "rung policy")),
            protocol_sha256=str(raw["protocol_sha256"]),
            config_sha256=str(raw["config_sha256"]),
            grid_sha256=str(raw["grid_sha256"]),
            tolerances_sha256=str(raw["tolerances_sha256"]),
            outcome=str(raw["outcome"]),  # type: ignore[arg-type]
            measurement=(
                None
                if measurement is None
                else RegularizationMeasurement.from_dict(
                    _mapping(measurement, "measurement")
                )
            ),
            failure_reason=(
                None if raw["failure_reason"] is None else str(raw["failure_reason"])
            ),
            failed_work=(
                None
                if failed_work is None
                else SolverWork.from_dict(_mapping(failed_work, "failed_work"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "factor": self.factor,
            "failed_work": (
                None if self.failed_work is None else self.failed_work.to_dict()
            ),
            "failure_reason": self.failure_reason,
            "grid_sha256": self.grid_sha256,
            "measurement": (
                None if self.measurement is None else self.measurement.to_dict()
            ),
            "outcome": self.outcome,
            "policy": _policy_to_dict(self.policy),
            "protocol_sha256": self.protocol_sha256,
            "tolerances_sha256": self.tolerances_sha256,
        }


@dataclass(frozen=True)
class CertificateCheck:
    name: str
    passed: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "check name"))
        if not isinstance(self.passed, bool):
            raise RegularizationCertificateError(
                f"check {self.name}: passed must be boolean"
            )
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise RegularizationCertificateError(
                f"check {self.name}: reasons must be non-empty strings"
            )
        if self.passed and reasons:
            raise RegularizationCertificateError(
                f"check {self.name}: passing checks cannot carry failure reasons"
            )
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RegularizationCertificate:
    """Content-addressed result of a fixed four-rung regularization study."""

    study: RegularizationStudy
    rungs: tuple[RegularizationRung, ...]
    checks: tuple[CertificateCheck, ...]
    status: CertificateStatus
    certificate_sha256: str = ""
    schema: str = REGULARIZATION_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.study, RegularizationStudy):
            raise RegularizationCertificateError(
                "certificate study must be a RegularizationStudy"
            )
        rungs = tuple(self.rungs)
        checks = tuple(self.checks)
        object.__setattr__(self, "rungs", rungs)
        object.__setattr__(self, "checks", checks)
        if tuple(rung.factor for rung in rungs) != REGULARIZATION_LADDER_FACTORS:
            raise RegularizationCertificateError(
                "certificate rungs must use the fixed factor order "
                f"{REGULARIZATION_LADDER_FACTORS}"
            )
        if self.status not in {"failed", "partial", "certified"}:
            raise RegularizationCertificateError(
                f"unknown certificate status {self.status!r}"
            )
        if self.schema != REGULARIZATION_CERTIFICATE_SCHEMA:
            raise RegularizationCertificateError(
                f"certificate schema must be {REGULARIZATION_CERTIFICATE_SCHEMA!r}"
            )
        expected_checks, expected_status = _derive_checks_and_status(self.study, rungs)
        if checks != expected_checks:
            raise RegularizationCertificateError(
                "certificate checks are not reproducible from rung evidence"
            )
        if self.status != expected_status:
            raise RegularizationCertificateError(
                "certificate status is not reproducible from rung evidence"
            )
        digest = _content_sha256(self._unsigned_dict())
        if self.certificate_sha256:
            if _sha256(self.certificate_sha256, "certificate_sha256") != digest:
                raise RegularizationCertificateError("certificate SHA-256 mismatch")
        else:
            object.__setattr__(self, "certificate_sha256", digest)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegularizationCertificate":
        expected = {
            "certificate_sha256",
            "checks",
            "rungs",
            "schema",
            "status",
            "study",
            "study_definition_sha256",
        }
        if set(raw) != expected:
            raise RegularizationCertificateError(
                "regularization certificate has incomplete or unknown fields"
            )
        if raw["schema"] != REGULARIZATION_CERTIFICATE_SCHEMA:
            raise RegularizationCertificateError("unsupported certificate schema")
        study = RegularizationStudy.from_dict(_mapping(raw["study"], "study"))
        if (
            _sha256(raw["study_definition_sha256"], "study_definition_sha256")
            != study.definition_sha256
        ):
            raise RegularizationCertificateError("study definition SHA-256 mismatch")
        rungs = tuple(
            RegularizationRung.from_dict(_mapping(item, "rung"))
            for item in _sequence(raw["rungs"], "rungs")
        )
        rebuilt = build_regularization_certificate(study, rungs)
        claimed_checks = _sequence(raw["checks"], "checks")
        if claimed_checks != [check.to_dict() for check in rebuilt.checks]:
            raise RegularizationCertificateError(
                "certificate checks are not reproducible"
            )
        if raw["status"] != rebuilt.status:
            raise RegularizationCertificateError(
                "certificate status is not reproducible"
            )
        if _sha256(raw["certificate_sha256"], "certificate_sha256") != (
            rebuilt.certificate_sha256
        ):
            raise RegularizationCertificateError("certificate SHA-256 mismatch")
        return rebuilt

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "rungs": [rung.to_dict() for rung in self.rungs],
            "schema": self.schema,
            "status": self.status,
            "study": self.study.to_dict(),
            "study_definition_sha256": self.study.definition_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._unsigned_dict()
        value["certificate_sha256"] = self.certificate_sha256
        return value

    def canonical_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("ascii")


@dataclass(frozen=True)
class RegularizationRungRequest:
    """The only execution-varying input is ``policy``/``factor``."""

    study: RegularizationStudy
    factor: float
    policy: RHSRegularization

    def __post_init__(self) -> None:
        if not isinstance(self.study, RegularizationStudy):
            raise RegularizationCertificateError(
                "rung request study must be a RegularizationStudy"
            )
        expected = policy_for_factor(self.study.base_policy, self.factor)
        if self.policy != expected:
            raise RegularizationCertificateError(
                "rung request policy does not match its fixed ladder factor"
            )

    @classmethod
    def for_factor(
        cls, study: RegularizationStudy, factor: float
    ) -> "RegularizationRungRequest":
        return cls(
            study=study,
            factor=factor,
            policy=policy_for_factor(study.base_policy, factor),
        )


RegularizationExecutor = Callable[
    [RegularizationRungRequest], RegularizationMeasurement
]


def run_regularization_ladder(
    study: RegularizationStudy,
    executor: RegularizationExecutor,
) -> RegularizationCertificate:
    """Execute every fixed rung and retain failures instead of dropping cells."""

    if not isinstance(study, RegularizationStudy):
        raise RegularizationCertificateError("study must be a RegularizationStudy")
    if not callable(executor):
        raise RegularizationCertificateError("executor must be callable")
    rungs: list[RegularizationRung] = []
    for factor in REGULARIZATION_LADDER_FACTORS:
        request = RegularizationRungRequest.for_factor(study, factor)
        started = time.perf_counter()
        try:
            measurement = executor(request)
            if not isinstance(measurement, RegularizationMeasurement):
                raise RegularizationCertificateError(
                    "executor must return RegularizationMeasurement"
                )
            elapsed = time.perf_counter() - started
            measurement = replace(
                measurement,
                work=replace(measurement.work, wall_time_s=elapsed),
            )
            rungs.append(RegularizationRung.completed(factor, measurement))
        except Exception as exc:  # noqa: BLE001 - a failed rung is certificate evidence
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}".strip()
            rungs.append(
                RegularizationRung.failed(
                    study,
                    factor,
                    reason,
                    work=SolverWork(None, None, None, elapsed),
                )
            )
    return build_regularization_certificate(study, rungs)


def build_regularization_certificate(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
) -> RegularizationCertificate:
    """Evaluate execution, health, and convergence without using speed as a gate."""

    if not isinstance(study, RegularizationStudy):
        raise RegularizationCertificateError("study must be a RegularizationStudy")
    by_factor: dict[float, RegularizationRung] = {}
    for rung in rungs:
        if not isinstance(rung, RegularizationRung):
            raise RegularizationCertificateError(
                "rungs must contain RegularizationRung values"
            )
        if rung.factor in by_factor:
            raise RegularizationCertificateError(
                f"duplicate regularization rung factor {rung.factor}"
            )
        by_factor[rung.factor] = rung
    ordered = tuple(
        by_factor.get(factor, RegularizationRung.missing(study, factor))
        for factor in REGULARIZATION_LADDER_FACTORS
    )

    checks, status = _derive_checks_and_status(study, ordered)
    return RegularizationCertificate(study, ordered, checks, status)


def _derive_checks_and_status(
    study: RegularizationStudy,
    ordered: tuple[RegularizationRung, ...],
) -> tuple[tuple[CertificateCheck, ...], CertificateStatus]:
    """Derive status so neither constructors nor deserializers can self-certify."""

    execution = _execution_check(ordered)
    context = _context_check(study, ordered)
    policies = _policy_check(study, ordered)
    evidence = _evidence_check(study, ordered)
    health = _health_check(study, ordered, evidence.passed)
    absolute_quality = _absolute_quality_check(study, ordered, evidence.passed)
    observable = _observable_check(study, ordered, evidence.passed)
    zero_closure = _zero_closure_check(study, ordered, evidence.passed)
    residual = _non_worsening_check(study, ordered, "residuals", evidence.passed)
    conservation = _non_worsening_check(
        study, ordered, "conservation_errors", evidence.passed
    )
    zero_limit = _zero_limit_check(study, ordered, evidence.passed)
    checks = (
        execution,
        context,
        policies,
        evidence,
        health,
        absolute_quality,
        observable,
        zero_closure,
        residual,
        conservation,
        zero_limit,
    )

    fatal = (execution, context, policies, evidence, health, absolute_quality)
    convergence = (observable, zero_closure, residual, conservation, zero_limit)
    if not all(check.passed for check in fatal):
        status: CertificateStatus = "failed"
    elif all(check.passed for check in convergence):
        status = "certified"
    else:
        status = "partial"
    return checks, status


def _execution_check(rungs: Sequence[RegularizationRung]) -> CertificateCheck:
    reasons = tuple(
        f"factor {rung.factor:g}: {rung.outcome}: {rung.failure_reason}"
        for rung in rungs
        if rung.outcome != "completed"
    )
    return CertificateCheck("ladder_complete", not reasons, reasons)


def _context_check(
    study: RegularizationStudy, rungs: Sequence[RegularizationRung]
) -> CertificateCheck:
    expected = {
        "protocol_sha256": study.protocol.sha256,
        "config_sha256": study.config.sha256,
        "grid_sha256": study.grid.sha256,
        "tolerances_sha256": study.tolerances.sha256,
    }
    reasons = tuple(
        f"factor {rung.factor:g}: {name} differs from fixed study context"
        for rung in rungs
        for name, digest in expected.items()
        if getattr(rung, name) != digest
    )
    return CertificateCheck("fixed_context", not reasons, reasons)


def _policy_check(
    study: RegularizationStudy, rungs: Sequence[RegularizationRung]
) -> CertificateCheck:
    reasons = tuple(
        f"factor {rung.factor:g}: policy does not equal base_policy * factor"
        for rung in rungs
        if rung.policy != policy_for_factor(study.base_policy, rung.factor)
    )
    return CertificateCheck("fixed_width_ladder", not reasons, reasons)


def _evidence_check(
    study: RegularizationStudy, rungs: Sequence[RegularizationRung]
) -> CertificateCheck:
    reasons: list[str] = []
    observable_shapes: dict[str, tuple[int, ...]] = {}
    for rung in rungs:
        if rung.measurement is None:
            reasons.append(f"factor {rung.factor:g}: completed evidence is unavailable")
            continue
        measurement = rung.measurement
        applied = measurement.applied
        if (
            applied.policy != rung.policy
            or applied.protocol_sha256 != rung.protocol_sha256
            or applied.config_sha256 != rung.config_sha256
            or applied.grid_sha256 != rung.grid_sha256
            or applied.tolerances_sha256 != rung.tolerances_sha256
        ):
            reasons.append(
                f"factor {rung.factor:g}: rung provenance differs from applied result"
            )
        reasons.extend(
            _metric_contract_reasons(
                rung.factor, "observable", study.observables, measurement.observables
            )
        )
        reasons.extend(
            _metric_contract_reasons(
                rung.factor,
                "residual",
                study.residuals,
                measurement.residuals,
                scalar_required=True,
                scalar_nonnegative=True,
            )
        )
        reasons.extend(
            _metric_contract_reasons(
                rung.factor,
                "conservation",
                study.conservation_errors,
                measurement.conservation_errors,
                scalar_required=True,
                scalar_nonnegative=True,
            )
        )
        reasons.extend(
            _metric_contract_reasons(
                rung.factor,
                "physical health",
                study.physical_health_gates,
                measurement.physical_health,
                scalar_required=True,
            )
        )
        if not measurement.work.complete:
            reasons.append(
                f"factor {rung.factor:g}: nfev/njev/nlu/wall evidence is incomplete"
            )
        trial_names = {metric.name for metric in measurement.minimum_trial_state_m3}
        terminal_names = {
            metric.name for metric in measurement.terminal_minimum_state_m3
        }
        expected_state_names = set(study.state_blocks)
        if trial_names != expected_state_names:
            reasons.append(
                f"factor {rung.factor:g}: trial state blocks do not match study"
            )
        if terminal_names != expected_state_names:
            reasons.append(
                f"factor {rung.factor:g}: terminal state blocks do not match study"
            )
        for metric in measurement.observables:
            previous = observable_shapes.setdefault(metric.name, metric.shape)
            if previous != metric.shape:
                reasons.append(
                    f"factor {rung.factor:g}: observable {metric.name} shape changed"
                )
    return CertificateCheck("evidence_complete", not reasons, tuple(reasons))


def _health_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    evidence_complete: bool,
) -> CertificateCheck:
    if not evidence_complete:
        return CertificateCheck(
            "physical_health",
            False,
            ("physical health is not evaluable without complete evidence",),
        )
    reasons: list[str] = []
    for rung in rungs:
        measurement = _measurement(rung)
        if not measurement.solver_accepted:
            reasons.append(f"factor {rung.factor:g}: solver acceptance gate failed")
        if measurement.nonfinite_event_count != 0:
            reasons.append(
                f"factor {rung.factor:g}: nonfinite_event_count="
                f"{measurement.nonfinite_event_count}"
            )
        nonpositive = [
            metric.name
            for metric in measurement.terminal_minimum_state_m3
            if metric.scalar <= 0.0
        ]
        if nonpositive:
            reasons.append(
                f"factor {rung.factor:g}: non-positive terminal state blocks "
                + ", ".join(sorted(nonpositive))
            )
        health = _metric_map(measurement.physical_health)
        for gate in study.physical_health_gates:
            value = health[gate.name].scalar
            if not gate.passes(value):
                reasons.append(
                    f"factor {rung.factor:g}: physical gate {gate.name} value "
                    f"{value:.12g} failed {gate.operator} {gate.limit:.12g}"
                )
    return CertificateCheck("physical_health", not reasons, tuple(reasons))


def _absolute_quality_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    evidence_complete: bool,
) -> CertificateCheck:
    if not evidence_complete:
        return CertificateCheck(
            "absolute_residual_conservation",
            False,
            ("absolute error gates are not evaluable without complete evidence",),
        )
    reasons: list[str] = []
    for rung in rungs:
        measurement = _measurement(rung)
        for category, specs in (
            ("residuals", study.residuals),
            ("conservation_errors", study.conservation_errors),
        ):
            values = _metric_map(getattr(measurement, category))
            for spec in specs:
                value = values[spec.name].scalar
                if value > spec.upper_limit:
                    reasons.append(
                        f"factor {rung.factor:g}: {spec.name}={value:.12g} exceeds "
                        f"absolute limit {spec.upper_limit:.12g}"
                    )
    return CertificateCheck(
        "absolute_residual_conservation", not reasons, tuple(reasons)
    )


def _observable_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    evidence_complete: bool,
) -> CertificateCheck:
    if not evidence_complete:
        return CertificateCheck(
            "final_positive_observable_change",
            False,
            ("observable change is not evaluable without complete evidence",),
        )
    coarse = _metric_map(_measurement(rungs[1]).observables)
    fine = _metric_map(_measurement(rungs[2]).observables)
    reasons: list[str] = []
    for spec in study.observables:
        change = _relative_linf(coarse[spec.name], fine[spec.name], spec.relative_floor)
        if not change < OBSERVABLE_RELATIVE_LIMIT:
            reasons.append(
                f"{spec.name}: relative change {change:.12g} is not < "
                f"{OBSERVABLE_RELATIVE_LIMIT:.12g}"
            )
    return CertificateCheck(
        "final_positive_observable_change", not reasons, tuple(reasons)
    )


def _zero_closure_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    evidence_complete: bool,
) -> CertificateCheck:
    if not evidence_complete:
        return CertificateCheck(
            "zero_width_observable_closure",
            False,
            ("zero-width closure is not evaluable without complete evidence",),
        )
    fine = _metric_map(_measurement(rungs[2]).observables)
    zero = _metric_map(_measurement(rungs[3]).observables)
    reasons: list[str] = []
    for spec in study.observables:
        closure = _relative_linf(fine[spec.name], zero[spec.name], spec.relative_floor)
        if not closure < OBSERVABLE_RELATIVE_LIMIT:
            reasons.append(
                f"{spec.name}: quarter-to-zero relative closure {closure:.12g} "
                f"is not < {OBSERVABLE_RELATIVE_LIMIT:.12g}"
            )
    return CertificateCheck(
        "zero_width_observable_closure", not reasons, tuple(reasons)
    )


def _non_worsening_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    category: Literal["residuals", "conservation_errors"],
    evidence_complete: bool,
) -> CertificateCheck:
    check_name = (
        "residual_non_worsening"
        if category == "residuals"
        else "conservation_non_worsening"
    )
    if not evidence_complete:
        return CertificateCheck(
            check_name,
            False,
            (f"{category} are not evaluable without complete evidence",),
        )
    specs = study.residuals if category == "residuals" else study.conservation_errors
    values = [_metric_map(getattr(_measurement(rung), category)) for rung in rungs]
    reasons: list[str] = []
    for spec in specs:
        for index in range(len(rungs) - 1):
            previous = values[index][spec.name].scalar
            narrowed = values[index + 1][spec.name].scalar
            if narrowed > previous + spec.non_worsening_atol:
                reasons.append(
                    f"{spec.name}: factor {rungs[index + 1].factor:g} value "
                    f"{narrowed:.12g} worsened from {previous:.12g} beyond atol "
                    f"{spec.non_worsening_atol:.12g}"
                )
    return CertificateCheck(check_name, not reasons, tuple(reasons))


def _zero_limit_check(
    study: RegularizationStudy,
    rungs: Sequence[RegularizationRung],
    evidence_complete: bool,
) -> CertificateCheck:
    if not evidence_complete:
        return CertificateCheck(
            "zero_width_convergence",
            False,
            ("zero-width convergence is not evaluable without complete evidence",),
        )
    maps = [_metric_map(_measurement(rung).observables) for rung in rungs]
    reasons: list[str] = []
    for spec in study.observables:
        reference = maps[3][spec.name]
        errors = [
            _absolute_linf(maps[index][spec.name], reference) for index in range(3)
        ]
        if not errors[2] <= errors[1] <= errors[0]:
            reasons.append(
                f"{spec.name}: distances to zero-width rung are not "
                f"non-increasing ({errors[0]:.12g}, {errors[1]:.12g}, "
                f"{errors[2]:.12g})"
            )
    return CertificateCheck("zero_width_convergence", not reasons, tuple(reasons))


def _metric_contract_reasons(
    factor: float,
    category: str,
    specs: Sequence[MetricSpec | ObservableSpec | QualityGateSpec],
    metrics: Sequence[MetricValue],
    *,
    scalar_required: bool = False,
    scalar_nonnegative: bool = False,
) -> list[str]:
    reasons: list[str] = []
    expected = {spec.name: spec for spec in specs}
    actual = _metric_map(metrics)
    if set(actual) != set(expected):
        reasons.append(f"factor {factor:g}: {category} metric names do not match study")
        return reasons
    for name, spec in expected.items():
        metric = actual[name]
        if metric.units != spec.units:
            reasons.append(
                f"factor {factor:g}: {category} {name} units differ from study"
            )
        if scalar_required and metric.shape:
            reasons.append(f"factor {factor:g}: {category} {name} must be scalar")
        elif scalar_nonnegative and metric.scalar < 0.0:
            reasons.append(f"factor {factor:g}: {category} {name} must be non-negative")
    return reasons


def _metrics_from_values(
    specs: Sequence[MetricSpec | ObservableSpec | QualityGateSpec],
    values: Mapping[str, Any],
) -> tuple[MetricValue, ...]:
    spec_by_name = {spec.name: spec for spec in specs}
    return tuple(
        MetricValue.from_value(
            name,
            value,
            units=spec_by_name[name].units if name in spec_by_name else "1",
        )
        for name, value in sorted(values.items())
    )


def _metrics_from_dict(value: Any, where: str) -> tuple[MetricValue, ...]:
    return tuple(
        MetricValue.from_dict(_mapping(item, where)) for item in _sequence(value, where)
    )


def _metric_map(metrics: Sequence[MetricValue]) -> dict[str, MetricValue]:
    result: dict[str, MetricValue] = {}
    for metric in metrics:
        if metric.name in result:
            raise RegularizationCertificateError(
                f"duplicate metric value {metric.name!r}"
            )
        result[metric.name] = metric
    return result


def _measurement(rung: RegularizationRung) -> RegularizationMeasurement:
    if rung.measurement is None:
        raise RegularizationCertificateError(
            f"factor {rung.factor:g} has no completed measurement"
        )
    return rung.measurement


def _relative_linf(
    first: MetricValue,
    second: MetricValue,
    floor: float,
) -> float:
    if first.shape != second.shape:
        raise RegularizationCertificateError(
            f"observable {first.name}: shape differs across rungs"
        )
    return max(
        abs(left - right) / max(abs(left), abs(right), floor)
        for left, right in zip(first.values, second.values, strict=True)
    )


def _absolute_linf(first: MetricValue, second: MetricValue) -> float:
    if first.shape != second.shape:
        raise RegularizationCertificateError(
            f"observable {first.name}: shape differs across rungs"
        )
    return max(
        abs(left - right)
        for left, right in zip(first.values, second.values, strict=True)
    )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegularizationCertificateError(f"{where} must be a mapping")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegularizationCertificateError(f"{where} must be a JSON array")
    return value


__all__ = [
    "AppliedRunContext",
    "CanonicalInput",
    "CertificateCheck",
    "MetricSpec",
    "MetricValue",
    "OBSERVABLE_RELATIVE_LIMIT",
    "ObservableSpec",
    "QualityGateSpec",
    "REGULARIZATION_CERTIFICATE_SCHEMA",
    "REGULARIZATION_EVALUATOR_VERSION",
    "REGULARIZATION_LADDER_FACTORS",
    "REGULARIZATION_STUDY_SCHEMA",
    "RegularizationCertificate",
    "RegularizationCertificateError",
    "RegularizationMeasurement",
    "RegularizationRung",
    "RegularizationRungRequest",
    "RegularizationStudy",
    "SolverWork",
    "build_regularization_certificate",
    "policy_for_factor",
    "run_regularization_ladder",
]

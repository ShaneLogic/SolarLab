"""Immutable contracts for tolerance-by-grid numerical certification.

The certificate deliberately separates execution completeness from numerical
convergence.  Missing or failed matrix cells produce ``failed``; a complete
matrix whose pre-registered gates do not close produces ``partial``; only a
complete matrix that closes both refinement axes is ``certified``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import yaml


CERTIFICATE_SCHEMA = "numerical-certificate-v1"
REGISTRY_SCHEMA = "numerical-refinement-registry-v1"
CELL_SCHEMA = "numerical-refinement-cell-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NumericalCertificateError(ValueError):
    """A numerical-certificate or lane contract is malformed."""


def _require_mapping_keys(
    raw: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    where: str,
) -> None:
    """Fail closed when a versioned mapping drifts from its declared schema."""
    keys = set(raw)
    unknown = keys - allowed
    if unknown:
        rendered = ", ".join(sorted(repr(item) for item in unknown))
        raise NumericalCertificateError(f"{where} has unknown keys: {rendered}")
    missing = required - keys
    if missing:
        rendered = ", ".join(sorted(repr(item) for item in missing))
        raise NumericalCertificateError(f"{where} is missing required keys: {rendered}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for all hashes."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise NumericalCertificateError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_identifier(value: Any, where: str) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        raise NumericalCertificateError(
            f"{where} must match {_IDENTIFIER.pattern!r}, got {text!r}"
        )
    return text


def _require_sha256(value: Any, where: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise NumericalCertificateError(f"{where} must be a lowercase SHA-256 digest")
    return text


def _finite_positive(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise NumericalCertificateError(f"{where} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NumericalCertificateError(f"{where} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise NumericalCertificateError(f"{where} must be finite and positive")
    return number


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NumericalCertificateError(f"{where} must be finite") from exc
    if not math.isfinite(number):
        raise NumericalCertificateError(f"{where} must be finite")
    return number


def _integer_at_least(value: Any, minimum: int, where: str) -> int:
    if isinstance(value, bool):
        raise NumericalCertificateError(f"{where} must be an integer >= {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericalCertificateError(
            f"{where} must be an integer >= {minimum}"
        ) from exc
    if integer != value or integer < minimum:
        raise NumericalCertificateError(f"{where} must be an integer >= {minimum}")
    return integer


@dataclass(frozen=True, order=True)
class MatrixPoint:
    """One cell of the Cartesian grid-by-tolerance matrix."""

    grid: int
    tolerance_factor: float

    def __post_init__(self) -> None:
        if isinstance(self.grid, bool) or int(self.grid) != self.grid or self.grid < 1:
            raise NumericalCertificateError("matrix grid must be a positive integer")
        object.__setattr__(self, "grid", int(self.grid))
        object.__setattr__(
            self,
            "tolerance_factor",
            _finite_positive(self.tolerance_factor, "matrix tolerance_factor"),
        )

    @property
    def key(self) -> str:
        tolerance = format(self.tolerance_factor, ".15g")
        return f"grid={self.grid};tolerance={tolerance}"

    def to_dict(self) -> dict[str, int | float]:
        return {
            "grid": self.grid,
            "tolerance_factor": self.tolerance_factor,
        }


@dataclass(frozen=True)
class MetricValue:
    """A finite scalar or shaped numeric observable stored without mutability."""

    name: str
    values: tuple[float, ...]
    shape: tuple[int, ...] = ()
    units: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, "metric name"))
        if any(
            isinstance(item, bool) or int(item) != item or item < 1
            for item in self.shape
        ):
            raise NumericalCertificateError(
                f"metric {self.name}: shape entries must be positive integers"
            )
        shape = tuple(int(item) for item in self.shape)
        object.__setattr__(self, "shape", shape)
        expected = math.prod(shape) if shape else 1
        if len(self.values) != expected:
            raise NumericalCertificateError(
                f"metric {self.name}: shape {self.shape} requires {expected} values, "
                f"got {len(self.values)}"
            )
        normalized = tuple(
            _finite_number(value, f"metric {self.name}") for value in self.values
        )
        object.__setattr__(self, "values", normalized)
        if not isinstance(self.units, str) or not self.units:
            raise NumericalCertificateError(
                f"metric {self.name}: units must be a non-empty string"
            )

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        units: str = "1",
    ) -> "MetricValue":
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise NumericalCertificateError(
                f"metric {name}: values must be real-valued"
            )
        if array.ndim == 0:
            return cls(name=name, values=(float(array),), units=units)
        if array.size == 0:
            raise NumericalCertificateError(f"metric {name}: arrays cannot be empty")
        try:
            numeric = np.asarray(array, dtype=float)
        except (TypeError, ValueError) as exc:
            raise NumericalCertificateError(f"metric {name}: must be numeric") from exc
        return cls(
            name=name,
            values=tuple(float(item) for item in numeric.ravel(order="C")),
            shape=tuple(int(item) for item in numeric.shape),
            units=units,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricValue":
        keys = {"name", "shape", "units", "values"}
        _require_mapping_keys(
            raw,
            allowed=keys,
            required=keys,
            where="metric",
        )
        return cls(
            name=str(raw["name"]),
            values=tuple(raw["values"] or ()),
            shape=tuple(raw["shape"] or ()),
            units=str(raw["units"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "units": self.units,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class ObservableGate:
    """Pre-registered convergence limit for one reported observable."""

    metric: str
    comparison: Literal[
        "absolute_linf",
        "relative_linf",
        "pointwise_relative_linf",
    ]
    limit: float
    units: str = "1"
    relative_floor: float = 1.0e-30

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric", _require_identifier(self.metric, "observable metric")
        )
        if self.comparison not in {
            "absolute_linf",
            "relative_linf",
            "pointwise_relative_linf",
        }:
            raise NumericalCertificateError(
                f"observable {self.metric}: unsupported comparison {self.comparison!r}"
            )
        object.__setattr__(
            self, "limit", _finite_positive(self.limit, f"{self.metric}.limit")
        )
        object.__setattr__(
            self,
            "relative_floor",
            _finite_positive(
                self.relative_floor,
                f"{self.metric}.relative_floor",
            ),
        )
        if not isinstance(self.units, str) or not self.units:
            raise NumericalCertificateError(
                f"observable {self.metric}: units must be non-empty"
            )

    @classmethod
    def from_mapping(cls, metric: str, raw: Mapping[str, Any]) -> "ObservableGate":
        _require_mapping_keys(
            raw,
            allowed={"comparison", "limit", "relative_floor", "units"},
            required={"comparison", "limit", "units"},
            where=f"observable {metric}",
        )
        return cls(
            metric=metric,
            comparison=str(raw.get("comparison", "")),  # type: ignore[arg-type]
            limit=raw.get("limit"),
            units=str(raw.get("units", "1")),
            relative_floor=raw.get("relative_floor", 1.0e-30),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "comparison": self.comparison,
            "limit": self.limit,
            "relative_floor": self.relative_floor,
            "units": self.units,
        }


@dataclass(frozen=True)
class QualityGate:
    """Per-cell physical/numerical gate applied to every matrix cell."""

    metric: str
    operator: Literal["le", "ge", "eq"]
    limit: float
    units: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric", _require_identifier(self.metric, "quality metric")
        )
        if self.operator not in {"le", "ge", "eq"}:
            raise NumericalCertificateError(
                f"quality {self.metric}: unsupported operator {self.operator!r}"
            )
        object.__setattr__(
            self, "limit", _finite_number(self.limit, f"quality {self.metric}.limit")
        )
        if not isinstance(self.units, str) or not self.units:
            raise NumericalCertificateError(
                f"quality {self.metric}: units must be non-empty"
            )

    @classmethod
    def from_mapping(cls, metric: str, raw: Mapping[str, Any]) -> "QualityGate":
        _require_mapping_keys(
            raw,
            allowed={"operator", "limit", "units"},
            required={"operator", "limit", "units"},
            where=f"quality gate {metric}",
        )
        return cls(
            metric=metric,
            operator=str(raw.get("operator", "")),  # type: ignore[arg-type]
            limit=raw.get("limit"),
            units=str(raw.get("units", "1")),
        )

    def passes(self, value: float) -> bool:
        if self.operator == "le":
            return value <= self.limit
        if self.operator == "ge":
            return value >= self.limit
        return value == self.limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "limit": self.limit,
            "units": self.units,
        }


@dataclass(frozen=True)
class LaneDefinition:
    """Deeply immutable, pre-registered refinement-lane definition."""

    lane_id: str
    claim_level: str
    config_path: str
    config_sha256: str
    grid_parameter: str
    grid_values: tuple[int, ...]
    tolerance_parameter: str
    tolerance_factors: tuple[float, ...]
    observables: tuple[ObservableGate, ...]
    quality_gates: tuple[QualityGate, ...]
    executor: str | None = None
    executor_version: str = "1"
    options_json: str = "{}"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lane_id", _require_identifier(self.lane_id, "lane_id")
        )
        object.__setattr__(
            self, "claim_level", _require_identifier(self.claim_level, "claim_level")
        )
        if not self.config_path or Path(self.config_path).is_absolute():
            raise NumericalCertificateError(
                f"lane {self.lane_id}: config_path must be repository-relative"
            )
        if ".." in Path(self.config_path).parts:
            raise NumericalCertificateError(
                f"lane {self.lane_id}: config_path cannot traverse parents"
            )
        object.__setattr__(
            self,
            "config_sha256",
            _require_sha256(self.config_sha256, f"lane {self.lane_id}.config_sha256"),
        )
        object.__setattr__(
            self,
            "grid_parameter",
            _require_identifier(
                self.grid_parameter, f"lane {self.lane_id}.grid_parameter"
            ),
        )
        if (
            len(self.grid_values) < 2
            or any(
                isinstance(item, bool) or int(item) != item or item < 1
                for item in self.grid_values
            )
            or any(a >= b for a, b in zip(self.grid_values, self.grid_values[1:]))
        ):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: grid_values must contain at least two "
                "strictly increasing positive integers"
            )
        object.__setattr__(
            self, "grid_values", tuple(int(item) for item in self.grid_values)
        )
        object.__setattr__(
            self,
            "tolerance_parameter",
            _require_identifier(
                self.tolerance_parameter,
                f"lane {self.lane_id}.tolerance_parameter",
            ),
        )
        factors = tuple(
            _finite_positive(item, f"lane {self.lane_id}.tolerance_factors")
            for item in self.tolerance_factors
        )
        if len(factors) < 2 or any(a <= b for a, b in zip(factors, factors[1:])):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: tolerance_factors must contain at least "
                "two strictly decreasing positive values"
            )
        object.__setattr__(self, "tolerance_factors", factors)
        observables = tuple(self.observables)
        quality_gates = tuple(self.quality_gates)
        limitations = tuple(self.limitations)
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "quality_gates", quality_gates)
        object.__setattr__(self, "limitations", limitations)
        if not observables:
            raise NumericalCertificateError(
                f"lane {self.lane_id}: at least one observable gate is required"
            )
        observable_names = [item.metric for item in observables]
        quality_names = [item.metric for item in quality_gates]
        if len(observable_names) != len(set(observable_names)):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: duplicate observable gates"
            )
        if len(quality_names) != len(set(quality_names)):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: duplicate quality gates"
            )
        if self.executor is not None and ":" not in self.executor:
            raise NumericalCertificateError(
                f"lane {self.lane_id}: executor must be module:function"
            )
        _require_identifier(
            self.executor_version, f"lane {self.lane_id}.executor_version"
        )
        try:
            options = json.loads(self.options_json)
        except json.JSONDecodeError as exc:
            raise NumericalCertificateError(
                f"lane {self.lane_id}: options_json is invalid"
            ) from exc
        if not isinstance(options, dict):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: options must be a mapping"
            )
        object.__setattr__(
            self,
            "options_json",
            canonical_json_bytes(options).decode("ascii"),
        )
        if any(not isinstance(item, str) or not item for item in limitations):
            raise NumericalCertificateError(
                f"lane {self.lane_id}: limitations must be non-empty strings"
            )

    @classmethod
    def from_mapping(
        cls,
        lane_id: str,
        raw: Mapping[str, Any],
    ) -> "LaneDefinition":
        lane_keys = {
            "claim_level",
            "config",
            "config_sha256",
            "executor",
            "executor_version",
            "grid",
            "limitations",
            "observables",
            "options",
            "quality_gates",
            "tolerance",
        }
        _require_mapping_keys(
            raw,
            allowed=lane_keys,
            required=lane_keys,
            where=f"lane {lane_id}",
        )
        grid = raw.get("grid")
        tolerance = raw.get("tolerance")
        observables = raw.get("observables")
        quality = raw.get("quality_gates", {})
        if not isinstance(grid, Mapping) or not isinstance(tolerance, Mapping):
            raise NumericalCertificateError(
                f"lane {lane_id}: grid and tolerance must be mappings"
            )
        if not isinstance(observables, Mapping) or not isinstance(quality, Mapping):
            raise NumericalCertificateError(
                f"lane {lane_id}: observables and quality_gates must be mappings"
            )
        _require_mapping_keys(
            grid,
            allowed={"parameter", "values"},
            required={"parameter", "values"},
            where=f"lane {lane_id} grid",
        )
        _require_mapping_keys(
            tolerance,
            allowed={"parameter", "factors"},
            required={"parameter", "factors"},
            where=f"lane {lane_id} tolerance",
        )
        for name, value in observables.items():
            if not isinstance(value, Mapping):
                raise NumericalCertificateError(
                    f"lane {lane_id}: observable {name!r} must be a mapping"
                )
        for name, value in quality.items():
            if not isinstance(value, Mapping):
                raise NumericalCertificateError(
                    f"lane {lane_id}: quality gate {name!r} must be a mapping"
                )
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise NumericalCertificateError(
                f"lane {lane_id}: options must be a mapping"
            )
        return cls(
            lane_id=lane_id,
            claim_level=str(raw.get("claim_level", "")),
            config_path=str(raw.get("config", "")),
            config_sha256=str(raw.get("config_sha256", "")),
            grid_parameter=str(grid.get("parameter", "")),
            grid_values=tuple(grid.get("values") or ()),
            tolerance_parameter=str(tolerance.get("parameter", "")),
            tolerance_factors=tuple(tolerance.get("factors") or ()),
            observables=tuple(
                ObservableGate.from_mapping(str(name), value)
                for name, value in sorted(observables.items())
            ),
            quality_gates=tuple(
                QualityGate.from_mapping(str(name), value)
                for name, value in sorted(quality.items())
            ),
            executor=(
                None if raw.get("executor") in {None, ""} else str(raw["executor"])
            ),
            executor_version=str(raw.get("executor_version", "1")),
            options_json=canonical_json_bytes(dict(options)).decode("ascii"),
            limitations=tuple(str(item) for item in raw.get("limitations") or ()),
        )

    @property
    def definition_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def matrix_points(self) -> tuple[MatrixPoint, ...]:
        return tuple(
            MatrixPoint(grid, factor)
            for grid in self.grid_values
            for factor in self.tolerance_factors
        )

    @property
    def options(self) -> dict[str, Any]:
        return json.loads(self.options_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_level": self.claim_level,
            "config": self.config_path,
            "config_sha256": self.config_sha256,
            "executor": self.executor,
            "executor_version": self.executor_version,
            "grid": {
                "parameter": self.grid_parameter,
                "values": list(self.grid_values),
            },
            "lane_id": self.lane_id,
            "limitations": list(self.limitations),
            "observables": [item.to_dict() for item in self.observables],
            "options": json.loads(self.options_json),
            "quality_gates": [item.to_dict() for item in self.quality_gates],
            "tolerance": {
                "factors": list(self.tolerance_factors),
                "parameter": self.tolerance_parameter,
            },
        }


@dataclass(frozen=True)
class RefinementRegistry:
    schema_version: str
    certificate_schema: str
    lanes: tuple[LaneDefinition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lanes", tuple(self.lanes))
        if self.schema_version != REGISTRY_SCHEMA:
            raise NumericalCertificateError(
                f"registry schema_version must be {REGISTRY_SCHEMA!r}"
            )
        if self.certificate_schema != CERTIFICATE_SCHEMA:
            raise NumericalCertificateError(
                f"certificate_schema must be {CERTIFICATE_SCHEMA!r}"
            )
        lane_ids = [lane.lane_id for lane in self.lanes]
        if not lane_ids or len(lane_ids) != len(set(lane_ids)):
            raise NumericalCertificateError("registry lane IDs must be unique")

    def lane(self, lane_id: str) -> LaneDefinition:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        raise KeyError(lane_id)


def load_refinement_registry(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_config_hashes: bool = True,
) -> RefinementRegistry:
    registry_path = Path(path)
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise NumericalCertificateError(
            f"cannot load refinement registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("lanes"), Mapping):
        raise NumericalCertificateError(
            "refinement registry must contain a lanes mapping"
        )
    registry_keys = {"schema_version", "certificate_schema", "policy", "lanes"}
    _require_mapping_keys(
        raw,
        allowed=registry_keys,
        required=registry_keys,
        where="refinement registry",
    )
    policy = raw["policy"]
    if not isinstance(policy, Mapping):
        raise NumericalCertificateError("refinement registry policy must be a mapping")
    policy_keys = {
        "terminal_pairs",
        "incomplete_matrix_status",
        "complete_nonconverged_status",
        "complete_converged_status",
        "threshold_change_policy",
        "result_policy",
    }
    _require_mapping_keys(
        policy,
        allowed=policy_keys,
        required=policy_keys,
        where="refinement registry policy",
    )
    terminal_pairs = policy["terminal_pairs"]
    if not isinstance(terminal_pairs, Mapping):
        raise NumericalCertificateError(
            "refinement registry terminal_pairs must be a mapping"
        )
    _require_mapping_keys(
        terminal_pairs,
        allowed={"grid", "tolerance"},
        required={"grid", "tolerance"},
        where="refinement registry terminal_pairs",
    )
    registry = RefinementRegistry(
        schema_version=str(raw.get("schema_version", "")),
        certificate_schema=str(raw.get("certificate_schema", "")),
        lanes=tuple(
            LaneDefinition.from_mapping(str(lane_id), lane_raw)
            for lane_id, lane_raw in sorted(raw["lanes"].items())
            if isinstance(lane_raw, Mapping)
        ),
    )
    if len(registry.lanes) != len(raw["lanes"]):
        raise NumericalCertificateError("every registry lane must be a mapping")
    if verify_config_hashes:
        root = (
            Path(project_root)
            if project_root is not None
            else registry_path.parent.parent
        )
        root = root.resolve()
        for lane in registry.lanes:
            config_path = (root / lane.config_path).resolve()
            try:
                config_path.relative_to(root)
            except ValueError as exc:
                raise NumericalCertificateError(
                    f"lane {lane.lane_id}: config escapes project root"
                ) from exc
            if not config_path.is_file():
                raise NumericalCertificateError(
                    f"lane {lane.lane_id}: missing config {lane.config_path}"
                )
            actual = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if actual != lane.config_sha256:
                raise NumericalCertificateError(
                    f"lane {lane.lane_id}: config SHA-256 drift "
                    f"({actual} != {lane.config_sha256})"
                )
    return registry


@dataclass(frozen=True)
class CellResult:
    """Immutable outcome for one matrix cell."""

    point: MatrixPoint
    status: Literal["completed", "failed"]
    observables: tuple[MetricValue, ...] = ()
    quality: tuple[MetricValue, ...] = ()
    wall_time_s: float = 0.0
    metadata_json: str = "{}"
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise NumericalCertificateError(f"unknown cell status {self.status!r}")
        wall_time = _finite_number(self.wall_time_s, "cell wall_time_s")
        if wall_time < 0.0:
            raise NumericalCertificateError("cell wall_time_s must be non-negative")
        object.__setattr__(self, "wall_time_s", wall_time)
        observables = tuple(self.observables)
        quality = tuple(self.quality)
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "quality", quality)
        for collection_name, collection in (
            ("observables", observables),
            ("quality", quality),
        ):
            names = [item.name for item in collection]
            if len(names) != len(set(names)):
                raise NumericalCertificateError(
                    f"cell {self.point.key}: duplicate {collection_name} metrics"
                )
        if any(item.shape for item in quality):
            raise NumericalCertificateError("quality metrics must be scalar")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise NumericalCertificateError("cell metadata_json is invalid") from exc
        if not isinstance(metadata, dict):
            raise NumericalCertificateError("cell metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata_json",
            canonical_json_bytes(metadata).decode("ascii"),
        )
        if self.status == "completed":
            if not self.observables:
                raise NumericalCertificateError(
                    f"completed cell {self.point.key} has no observables"
                )
            if self.error_type is not None or self.error_message is not None:
                raise NumericalCertificateError(
                    "completed cells cannot carry an execution error"
                )
        elif not self.error_type or not self.error_message:
            raise NumericalCertificateError(
                f"failed cell {self.point.key} must record error type and message"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CellResult":
        cell_keys = {
            "error_message",
            "error_type",
            "metadata",
            "observables",
            "point",
            "quality",
            "status",
            "wall_time_s",
        }
        _require_mapping_keys(
            raw,
            allowed=cell_keys,
            required=cell_keys,
            where="cell",
        )
        point = raw.get("point")
        if not isinstance(point, Mapping):
            raise NumericalCertificateError("cell point must be a mapping")
        _require_mapping_keys(
            point,
            allowed={"grid", "tolerance_factor"},
            required={"grid", "tolerance_factor"},
            where="cell point",
        )
        observables = raw["observables"]
        quality = raw["quality"]
        if (
            not isinstance(observables, Sequence)
            or isinstance(observables, (str, bytes))
            or any(not isinstance(item, Mapping) for item in observables)
        ):
            raise NumericalCertificateError("cell observables must be mappings")
        if (
            not isinstance(quality, Sequence)
            or isinstance(quality, (str, bytes))
            or any(not isinstance(item, Mapping) for item in quality)
        ):
            raise NumericalCertificateError("cell quality metrics must be mappings")
        return cls(
            point=MatrixPoint(
                grid=point.get("grid"),
                tolerance_factor=point.get("tolerance_factor"),
            ),
            status=str(raw.get("status", "")),  # type: ignore[arg-type]
            observables=tuple(
                MetricValue.from_dict(item) for item in observables
            ),
            quality=tuple(
                MetricValue.from_dict(item) for item in quality
            ),
            wall_time_s=raw.get("wall_time_s", 0.0),
            metadata_json=canonical_json_bytes(raw.get("metadata", {})).decode("ascii"),
            error_type=raw.get("error_type"),
            error_message=raw.get("error_message"),
        )

    def metric(self, name: str, *, quality: bool = False) -> MetricValue:
        collection = self.quality if quality else self.observables
        for item in collection:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_message": self.error_message,
            "error_type": self.error_type,
            "metadata": json.loads(self.metadata_json),
            "observables": [item.to_dict() for item in self.observables],
            "point": self.point.to_dict(),
            "quality": [item.to_dict() for item in self.quality],
            "status": self.status,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True)
class ConvergenceCheck:
    dimension: Literal["grid", "tolerance", "quality"]
    metric: str
    passed: bool
    observed: float
    limit: float
    comparison: str
    cells: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dimension not in {"grid", "tolerance", "quality"}:
            raise NumericalCertificateError(
                f"unknown convergence dimension {self.dimension!r}"
            )
        object.__setattr__(
            self,
            "metric",
            _require_identifier(self.metric, "convergence-check metric"),
        )
        if not isinstance(self.passed, bool):
            raise NumericalCertificateError("convergence-check passed must be bool")
        object.__setattr__(
            self,
            "observed",
            _finite_number(self.observed, f"check {self.metric}.observed"),
        )
        object.__setattr__(
            self,
            "limit",
            _finite_number(self.limit, f"check {self.metric}.limit"),
        )
        if not isinstance(self.comparison, str) or not self.comparison:
            raise NumericalCertificateError(
                "convergence-check comparison must be non-empty"
            )
        cells = tuple(str(item) for item in self.cells)
        if not cells or any(not item for item in cells):
            raise NumericalCertificateError("convergence-check cells must be non-empty")
        object.__setattr__(self, "cells", cells)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConvergenceCheck":
        keys = {
            "cells",
            "comparison",
            "dimension",
            "limit",
            "metric",
            "observed",
            "passed",
        }
        _require_mapping_keys(
            raw,
            allowed=keys,
            required=keys,
            where="convergence check",
        )
        return cls(
            dimension=str(raw.get("dimension", "")),  # type: ignore[arg-type]
            metric=str(raw.get("metric", "")),
            passed=raw.get("passed"),
            observed=raw.get("observed"),
            limit=raw.get("limit"),
            comparison=str(raw.get("comparison", "")),
            cells=tuple(raw.get("cells") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": list(self.cells),
            "comparison": self.comparison,
            "dimension": self.dimension,
            "limit": self.limit,
            "metric": self.metric,
            "observed": self.observed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class NumericalCertificate:
    """Immutable final assessment tied to one manifest snapshot."""

    run_id: str
    lane_id: str
    lane_definition_sha256: str
    config_path: str
    config_sha256: str
    source_commit: str
    source_fingerprint_sha256: str
    environment_json: str
    manifest_sha256: str
    protocol_sha256: str | None
    status: Literal["certified", "partial", "failed"]
    expected_cells: int
    completed_cells: int
    failed_cells: tuple[str, ...]
    missing_cells: tuple[str, ...]
    unconverged_dimensions: tuple[str, ...]
    checks: tuple[ConvergenceCheck, ...]
    cell_artifact_sha256: tuple[str, ...]
    limitations: tuple[str, ...]
    certificate_sha256: str = field(default="")
    schema_version: str = field(default=CERTIFICATE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        failed_cells = tuple(str(item) for item in self.failed_cells)
        missing_cells = tuple(str(item) for item in self.missing_cells)
        unconverged = tuple(str(item) for item in self.unconverged_dimensions)
        checks = tuple(self.checks)
        artifact_hashes = tuple(str(item) for item in self.cell_artifact_sha256)
        limitations = tuple(str(item) for item in self.limitations)
        object.__setattr__(self, "failed_cells", failed_cells)
        object.__setattr__(self, "missing_cells", missing_cells)
        object.__setattr__(self, "unconverged_dimensions", unconverged)
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "cell_artifact_sha256", artifact_hashes)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(
            self,
            "expected_cells",
            _integer_at_least(self.expected_cells, 1, "certificate expected_cells"),
        )
        object.__setattr__(
            self,
            "completed_cells",
            _integer_at_least(
                self.completed_cells,
                0,
                "certificate completed_cells",
            ),
        )
        _require_sha256(self.run_id, "certificate run_id")
        _require_identifier(self.lane_id, "certificate lane_id")
        for name in (
            "lane_definition_sha256",
            "config_sha256",
            "source_fingerprint_sha256",
            "manifest_sha256",
        ):
            _require_sha256(getattr(self, name), f"certificate {name}")
        if self.protocol_sha256 is not None:
            object.__setattr__(
                self,
                "protocol_sha256",
                _require_sha256(
                    self.protocol_sha256,
                    "certificate protocol_sha256",
                ),
            )
        if self.source_commit != "unknown" and not re.fullmatch(
            r"[0-9a-f]{40}", self.source_commit
        ):
            raise NumericalCertificateError(
                "certificate source_commit must be a Git SHA-1 or 'unknown'"
            )
        if self.status not in {"certified", "partial", "failed"}:
            raise NumericalCertificateError(
                f"unknown certificate status {self.status!r}"
            )
        if self.completed_cells > self.expected_cells:
            raise NumericalCertificateError("invalid certificate cell counts")
        if len(failed_cells) != len(set(failed_cells)) or len(missing_cells) != len(
            set(missing_cells)
        ):
            raise NumericalCertificateError("certificate cell lists must be unique")
        if set(failed_cells) & set(missing_cells):
            raise NumericalCertificateError(
                "certificate failed and missing cells must be disjoint"
            )
        if (
            self.completed_cells + len(failed_cells) + len(missing_cells)
            != self.expected_cells
        ):
            raise NumericalCertificateError(
                "completed, failed, and missing counts must cover the expected matrix"
            )
        failed_checks = tuple(check for check in checks if not check.passed)
        if self.status == "certified" and (
            self.completed_cells != self.expected_cells
            or failed_cells
            or missing_cells
            or unconverged
            or failed_checks
            or not checks
        ):
            raise NumericalCertificateError(
                "certified status requires a complete, fully passing matrix"
            )
        if self.status == "partial" and (
            self.completed_cells != self.expected_cells
            or failed_cells
            or missing_cells
            or not unconverged
            or not failed_checks
        ):
            raise NumericalCertificateError(
                "partial status requires a complete matrix with failed gates"
            )
        if self.status == "failed" and not unconverged:
            raise NumericalCertificateError(
                "failed status requires an explicit failed contract dimension"
            )
        try:
            environment = json.loads(self.environment_json)
        except json.JSONDecodeError as exc:
            raise NumericalCertificateError(
                "certificate environment_json is invalid"
            ) from exc
        if not isinstance(environment, dict):
            raise NumericalCertificateError("certificate environment must be a mapping")
        object.__setattr__(
            self,
            "environment_json",
            canonical_json_bytes(environment).decode("ascii"),
        )
        for digest in artifact_hashes:
            _require_sha256(digest, "cell artifact SHA-256")
        expected_artifacts = self.completed_cells + len(failed_cells)
        if len(artifact_hashes) != expected_artifacts:
            raise NumericalCertificateError(
                "cell artifact SHA-256 count must match completed and failed cells"
            )
        if len(artifact_hashes) != len(set(artifact_hashes)):
            raise NumericalCertificateError("cell artifact SHA-256 values must be unique")
        expected_digest = content_sha256(self._unsigned_dict())
        if self.certificate_sha256:
            _require_sha256(self.certificate_sha256, "certificate_sha256")
            if self.certificate_sha256 != expected_digest:
                raise NumericalCertificateError(
                    "certificate SHA-256 does not match content"
                )
        else:
            object.__setattr__(self, "certificate_sha256", expected_digest)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "cell_artifact_sha256": list(self.cell_artifact_sha256),
            "checks": [item.to_dict() for item in self.checks],
            "completed_cells": self.completed_cells,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "environment": json.loads(self.environment_json),
            "expected_cells": self.expected_cells,
            "failed_cells": list(self.failed_cells),
            "lane_definition_sha256": self.lane_definition_sha256,
            "lane_id": self.lane_id,
            "limitations": list(self.limitations),
            "manifest_sha256": self.manifest_sha256,
            "missing_cells": list(self.missing_cells),
            "protocol_sha256": self.protocol_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "status": self.status,
            "unconverged_dimensions": list(self.unconverged_dimensions),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._unsigned_dict()
        value["certificate_sha256"] = self.certificate_sha256
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NumericalCertificate":
        """Restore a certificate while re-running every schema invariant."""
        certificate_keys = {
            "cell_artifact_sha256",
            "certificate_sha256",
            "checks",
            "completed_cells",
            "config_path",
            "config_sha256",
            "environment",
            "expected_cells",
            "failed_cells",
            "lane_definition_sha256",
            "lane_id",
            "limitations",
            "manifest_sha256",
            "missing_cells",
            "protocol_sha256",
            "run_id",
            "schema_version",
            "source_commit",
            "source_fingerprint_sha256",
            "status",
            "unconverged_dimensions",
        }
        _require_mapping_keys(
            raw,
            allowed=certificate_keys,
            required=certificate_keys,
            where="numerical certificate",
        )
        if raw.get("schema_version") != CERTIFICATE_SCHEMA:
            raise NumericalCertificateError(
                f"certificate schema_version must be {CERTIFICATE_SCHEMA!r}"
            )
        environment = raw.get("environment")
        checks = raw.get("checks")
        if not isinstance(environment, Mapping):
            raise NumericalCertificateError("certificate environment must be a mapping")
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            raise NumericalCertificateError("certificate checks must be a sequence")
        if any(not isinstance(item, Mapping) for item in checks):
            raise NumericalCertificateError("certificate checks must be mappings")
        return cls(
            run_id=str(raw.get("run_id", "")),
            lane_id=str(raw.get("lane_id", "")),
            lane_definition_sha256=str(raw.get("lane_definition_sha256", "")),
            config_path=str(raw.get("config_path", "")),
            config_sha256=str(raw.get("config_sha256", "")),
            source_commit=str(raw.get("source_commit", "")),
            source_fingerprint_sha256=str(raw.get("source_fingerprint_sha256", "")),
            environment_json=canonical_json_bytes(dict(environment)).decode("ascii"),
            manifest_sha256=str(raw.get("manifest_sha256", "")),
            protocol_sha256=(
                None
                if raw.get("protocol_sha256") is None
                else str(raw["protocol_sha256"])
            ),
            status=str(raw.get("status", "")),  # type: ignore[arg-type]
            expected_cells=raw.get("expected_cells"),
            completed_cells=raw.get("completed_cells"),
            failed_cells=tuple(raw.get("failed_cells") or ()),
            missing_cells=tuple(raw.get("missing_cells") or ()),
            unconverged_dimensions=tuple(raw.get("unconverged_dimensions") or ()),
            checks=tuple(ConvergenceCheck.from_dict(item) for item in checks),
            cell_artifact_sha256=tuple(raw.get("cell_artifact_sha256") or ()),
            limitations=tuple(raw.get("limitations") or ()),
            certificate_sha256=str(raw.get("certificate_sha256", "")),
        )


def _observable_delta(
    left: MetricValue,
    right: MetricValue,
    gate: ObservableGate,
) -> float:
    a = np.asarray(left.values, dtype=float)
    b = np.asarray(right.values, dtype=float)
    absolute = float(np.max(np.abs(b - a)))
    if gate.comparison == "absolute_linf":
        return absolute
    if gate.comparison == "pointwise_relative_linf":
        scale = np.maximum(
            np.maximum(np.abs(a), np.abs(b)),
            gate.relative_floor,
        )
        return float(np.max(np.abs(b - a) / scale))
    scale = max(
        float(np.max(np.abs(a))),
        float(np.max(np.abs(b))),
        gate.relative_floor,
    )
    return absolute / scale


def _protocol_contract(
    lane: LaneDefinition,
    expected: Mapping[str, MatrixPoint],
    cells: Mapping[str, CellResult],
) -> tuple[list[str], str | None]:
    """Validate a lane-stable canonical protocol on every completed cell."""
    required = lane.options.get("require_protocol", False)
    if not isinstance(required, bool):
        return ["protocol:require_protocol_not_boolean"], None
    errors: list[str] = []
    hashes: set[str] = set()
    completed = 0
    valid = 0
    for key in expected:
        cell = cells.get(key)
        if cell is None or cell.status != "completed":
            continue
        completed += 1
        metadata = json.loads(cell.metadata_json)
        present = {
            name
            for name in ("protocol", "protocol_hash", "protocol_schema")
            if name in metadata
        }
        if not present:
            if required:
                errors.append(f"protocol:missing@{key}")
            continue
        if present != {"protocol", "protocol_hash", "protocol_schema"}:
            errors.append(f"protocol:incomplete@{key}")
            continue
        protocol = metadata["protocol"]
        protocol_hash = metadata["protocol_hash"]
        protocol_schema = metadata["protocol_schema"]
        if (
            not isinstance(protocol, Mapping)
            or not isinstance(protocol_hash, str)
            or not isinstance(protocol_schema, str)
        ):
            errors.append(f"protocol:malformed@{key}")
            continue
        if protocol.get("schema_version") != protocol_schema:
            errors.append(f"protocol:schema@{key}")
            continue
        try:
            _require_sha256(protocol_hash, f"protocol hash at {key}")
        except NumericalCertificateError:
            errors.append(f"protocol:hash_format@{key}")
            continue
        if content_sha256(dict(protocol)) != protocol_hash:
            errors.append(f"protocol:hash_content@{key}")
            continue
        hashes.add(protocol_hash)
        valid += 1
    if len(hashes) > 1:
        errors.append("protocol:inconsistent_across_matrix")
    if required and completed and valid != completed:
        errors.append("protocol:invalid_completed_cells")
    return errors, next(iter(hashes)) if len(hashes) == 1 else None


def _matrix_metric_contract(
    lane: LaneDefinition,
    expected: Mapping[str, MatrixPoint],
    cells: Mapping[str, CellResult],
) -> list[str]:
    """Validate every completed cell against the pre-registered metric schema."""
    errors: list[str] = []
    observable_gates = {gate.metric: gate for gate in lane.observables}
    quality_gates = {gate.metric: gate for gate in lane.quality_gates}
    observable_shapes: dict[str, set[tuple[int, ...]]] = {
        name: set() for name in observable_gates
    }
    for key in expected:
        cell = cells.get(key)
        if cell is None or cell.status != "completed":
            continue
        observables = {metric.name: metric for metric in cell.observables}
        quality = {metric.name: metric for metric in cell.quality}
        for name in sorted(set(observable_gates) - set(observables)):
            errors.append(f"observable:{name}:missing@{key}")
        for name in sorted(set(observables) - set(observable_gates)):
            errors.append(f"observable:{name}:unexpected@{key}")
        for name in sorted(set(observable_gates) & set(observables)):
            metric = observables[name]
            if metric.units != observable_gates[name].units:
                errors.append(f"observable:{name}:units@{key}")
            observable_shapes[name].add(metric.shape)
        for name in sorted(set(quality_gates) - set(quality)):
            errors.append(f"quality:{name}:missing@{key}")
        for name in sorted(set(quality) - set(quality_gates)):
            errors.append(f"quality:{name}:unexpected@{key}")
        for name in sorted(set(quality_gates) & set(quality)):
            if quality[name].units != quality_gates[name].units:
                errors.append(f"quality:{name}:units@{key}")
    for name, shapes in observable_shapes.items():
        if len(shapes) > 1:
            errors.append(f"observable:{name}:shape_across_matrix")
    return errors


def evaluate_numerical_certificate(
    lane: LaneDefinition,
    cells: Sequence[CellResult],
    *,
    run_id: str,
    source_commit: str,
    source_fingerprint_sha256: str,
    environment: Mapping[str, Any],
    manifest_sha256: str,
    cell_artifact_sha256: Sequence[str],
) -> NumericalCertificate:
    """Evaluate exactly the pre-registered terminal grid/tolerance pairs."""
    expected = {point.key: point for point in lane.matrix_points}
    by_key: dict[str, CellResult] = {}
    duplicate_keys: set[str] = set()
    for cell in cells:
        if cell.point.key in by_key:
            duplicate_keys.add(cell.point.key)
        by_key[cell.point.key] = cell

    missing = tuple(sorted(set(expected) - set(by_key)))
    out_of_contract = tuple(sorted(set(by_key) - set(expected)))
    failed = tuple(
        sorted(
            key
            for key, cell in by_key.items()
            if key in expected and cell.status == "failed"
        )
    )
    execution_dimensions = [f"matrix:{key}" for key in missing]
    execution_dimensions.extend(f"execution:{key}" for key in failed)
    execution_dimensions.extend(f"duplicate:{key}" for key in sorted(duplicate_keys))
    execution_dimensions.extend(f"unexpected:{key}" for key in out_of_contract)

    checks: list[ConvergenceCheck] = []
    contract_errors, protocol_sha256 = _protocol_contract(lane, expected, by_key)
    contract_errors.extend(_matrix_metric_contract(lane, expected, by_key))
    if not execution_dimensions:
        fine_grid = lane.grid_values[-1]
        grid_pair = lane.grid_values[-2:]
        fine_tolerance = lane.tolerance_factors[-1]
        tolerance_pair = lane.tolerance_factors[-2:]

        for gate in lane.observables:
            pairs = (
                (
                    "grid",
                    MatrixPoint(grid_pair[0], fine_tolerance),
                    MatrixPoint(grid_pair[1], fine_tolerance),
                ),
                (
                    "tolerance",
                    MatrixPoint(fine_grid, tolerance_pair[0]),
                    MatrixPoint(fine_grid, tolerance_pair[1]),
                ),
            )
            for dimension, left_point, right_point in pairs:
                try:
                    left = by_key[left_point.key].metric(gate.metric)
                    right = by_key[right_point.key].metric(gate.metric)
                except KeyError:
                    contract_errors.append(f"{dimension}:{gate.metric}:missing")
                    continue
                if left.units != gate.units or right.units != gate.units:
                    contract_errors.append(f"{dimension}:{gate.metric}:units")
                    continue
                if left.shape != right.shape:
                    contract_errors.append(f"{dimension}:{gate.metric}:shape")
                    continue
                observed = _observable_delta(left, right, gate)
                checks.append(
                    ConvergenceCheck(
                        dimension=dimension,  # type: ignore[arg-type]
                        metric=gate.metric,
                        passed=math.isfinite(observed) and observed <= gate.limit,
                        observed=observed,
                        limit=gate.limit,
                        comparison=gate.comparison,
                        cells=(left_point.key, right_point.key),
                    )
                )

        for gate in lane.quality_gates:
            for point in lane.matrix_points:
                try:
                    metric = by_key[point.key].metric(gate.metric, quality=True)
                except KeyError:
                    contract_errors.append(f"quality:{gate.metric}:missing@{point.key}")
                    continue
                if metric.units != gate.units:
                    contract_errors.append(f"quality:{gate.metric}:units@{point.key}")
                    continue
                value = metric.values[0]
                checks.append(
                    ConvergenceCheck(
                        dimension="quality",
                        metric=gate.metric,
                        passed=gate.passes(value),
                        observed=value,
                        limit=gate.limit,
                        comparison=gate.operator,
                        cells=(point.key,),
                    )
                )

    failed_dimensions = [
        f"{check.dimension}:{check.metric}" for check in checks if not check.passed
    ]
    unconverged = tuple(
        dict.fromkeys(execution_dimensions + contract_errors + failed_dimensions)
    )
    if execution_dimensions or contract_errors:
        status: Literal["certified", "partial", "failed"] = "failed"
    elif failed_dimensions:
        status = "partial"
    else:
        status = "certified"

    completed_count = sum(
        cell.status == "completed" and key in expected for key, cell in by_key.items()
    )
    return NumericalCertificate(
        run_id=run_id,
        lane_id=lane.lane_id,
        lane_definition_sha256=lane.definition_sha256,
        config_path=lane.config_path,
        config_sha256=lane.config_sha256,
        source_commit=source_commit,
        source_fingerprint_sha256=source_fingerprint_sha256,
        environment_json=canonical_json_bytes(dict(environment)).decode("ascii"),
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        status=status,
        expected_cells=len(expected),
        completed_cells=completed_count,
        failed_cells=failed,
        missing_cells=missing,
        unconverged_dimensions=unconverged,
        checks=tuple(checks),
        cell_artifact_sha256=tuple(cell_artifact_sha256),
        limitations=lane.limitations,
    )

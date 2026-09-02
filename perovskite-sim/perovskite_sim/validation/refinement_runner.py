"""Resumable, content-addressed tolerance-by-grid matrix execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Mapping, Protocol

import numpy as np
import scipy
import yaml

from .numerical_certificate import (
    CELL_SCHEMA,
    CellResult,
    LaneDefinition,
    MatrixPoint,
    MetricValue,
    NumericalCertificate,
    NumericalCertificateError,
    canonical_json_bytes,
    content_sha256,
    evaluate_numerical_certificate,
)


MANIFEST_SCHEMA = "numerical-refinement-manifest-v1"
STATE_SCHEMA = "numerical-refinement-state-v1"
DEFAULT_OUTPUT_ROOT = Path("outputs/numerical-refinement")
_PROTECTED_OUTPUTS = (
    Path("reproducibility/baselines"),
    Path("perovskite_sim/data/references"),
)
_BEHAVIOR_ENVIRONMENT_VARIABLES = (
    "PEROVSKITE_RHS_FINITE_CHECK",
    "SOLARLAB_AUTOLOOP_GEN",
    "SOLARLAB_BAND_GRADING",
    "SOLARLAB_DOS_BAND",
    "SOLARLAB_IFACE_ALLOW_GEN",
    "SOLARLAB_IFACE_PLANE",
    "SOLARLAB_IFACE_PLANE_GEN",
    "SOLARLAB_IFACE_PROJ",
    "SOLARLAB_IFACE_QSS",
    "SOLARLAB_IFACE_SHARED_OCC",
    "SOLARLAB_IFACE_TUNNEL",
    "SOLARLAB_IFACE_TWOSIDED",
    "SOLARLAB_INTERFACE_PLANE_STATE",
    "SOLARLAB_ION_STERIC_DIFF",
    "SOLARLAB_QSS_VTH",
    "SOLARLAB_SS_GUMMEL",
    "SOLARLAB_SS_JAC_REUSE",
    "SOLARLAB_TE_PHYSICAL",
)
_MANIFEST_KEYS = {
    "artifacts",
    "environment",
    "executor",
    "executor_source_sha256",
    "expected_cells",
    "failed_cells",
    "lane_definition_sha256",
    "lane_id",
    "pending_cells",
    "protocols",
    "run_id",
    "schema_version",
    "source",
    "status",
}
_STATE_KEYS = {
    "latest_certificate_sha256",
    "latest_manifest_sha256",
    "run_id",
    "schema_version",
}
_CELL_ARTIFACT_KEYS = {
    "cell",
    "lane_definition_sha256",
    "run_id",
    "schema_version",
}
_ARTIFACT_REFERENCE_KEYS = {"path", "point_key", "sha256", "status"}


class RefinementRunnerError(RuntimeError):
    """The runner could not preserve its content-addressed contract."""


def _require_exact_keys(
    raw: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    keys = set(raw)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        rendered = ", ".join(sorted(repr(item) for item in unknown))
        raise RefinementRunnerError(f"{where} has unknown keys: {rendered}")
    if missing:
        rendered = ", ".join(sorted(repr(item) for item in missing))
        raise RefinementRunnerError(f"{where} is missing required keys: {rendered}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class CellMeasurement:
    """Successful executor payload before the runner adds provenance."""

    observables: tuple[MetricValue, ...]
    quality: tuple[MetricValue, ...] = ()
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        observables = tuple(self.observables)
        quality = tuple(self.quality)
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "quality", quality)
        if not observables:
            raise NumericalCertificateError("cell measurement has no observables")
        for label, metrics in (("observable", observables), ("quality", quality)):
            names = [item.name for item in metrics]
            if len(names) != len(set(names)):
                raise NumericalCertificateError(
                    f"cell measurement has duplicate {label} metrics"
                )
        if any(item.shape for item in quality):
            raise NumericalCertificateError(
                "measurement quality metrics must be scalar"
            )
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise NumericalCertificateError(
                "measurement metadata_json is invalid"
            ) from exc
        if not isinstance(metadata, dict):
            raise NumericalCertificateError("measurement metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata_json",
            canonical_json_bytes(metadata).decode("ascii"),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CellMeasurement":
        allowed = {"metadata", "observables", "quality", "units"}
        unknown = set(raw) - allowed
        if unknown:
            rendered = ", ".join(sorted(repr(item) for item in unknown))
            raise NumericalCertificateError(
                f"executor result has unknown keys: {rendered}"
            )
        observables = raw.get("observables")
        quality = raw.get("quality", {})
        units = raw.get("units", {})
        metadata = raw.get("metadata", {})
        if not isinstance(observables, Mapping) or not isinstance(quality, Mapping):
            raise NumericalCertificateError(
                "executor result must contain observables and quality mappings"
            )
        if not isinstance(units, Mapping) or not isinstance(metadata, Mapping):
            raise NumericalCertificateError(
                "executor result units and metadata must be mappings"
            )
        metric_names = set(observables) | set(quality)
        unknown_units = set(units) - metric_names
        if unknown_units:
            rendered = ", ".join(sorted(repr(item) for item in unknown_units))
            raise NumericalCertificateError(
                f"executor result units contain unknown metrics: {rendered}"
            )
        return cls(
            observables=tuple(
                MetricValue.from_value(
                    str(name), value, units=str(units.get(name, "1"))
                )
                for name, value in sorted(observables.items())
            ),
            quality=tuple(
                MetricValue.from_value(
                    str(name), value, units=str(units.get(name, "1"))
                )
                for name, value in sorted(quality.items())
            ),
            metadata_json=canonical_json_bytes(dict(metadata)).decode("ascii"),
        )


class RefinementExecutor(Protocol):
    def __call__(
        self,
        lane: LaneDefinition,
        point: MatrixPoint,
        project_root: Path,
    ) -> CellMeasurement | Mapping[str, Any]: ...


@dataclass(frozen=True)
class ArtifactReference:
    point_key: str
    status: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.point_key:
            raise RefinementRunnerError("artifact point_key must be non-empty")
        if self.status not in {"completed", "failed"}:
            raise RefinementRunnerError(f"invalid artifact status {self.status!r}")
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts:
            raise RefinementRunnerError("artifact path must be run-relative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise RefinementRunnerError("artifact SHA-256 is malformed")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ArtifactReference":
        _require_exact_keys(raw, _ARTIFACT_REFERENCE_KEYS, "artifact reference")
        return cls(
            point_key=str(raw.get("point_key", "")),
            status=str(raw.get("status", "")),
            path=str(raw.get("path", "")),
            sha256=str(raw.get("sha256", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "point_key": self.point_key,
            "sha256": self.sha256,
            "status": self.status,
        }


@dataclass(frozen=True)
class RunOutcome:
    certificate: NumericalCertificate
    run_directory: Path
    manifest_path: Path
    certificate_path: Path
    executed_cells: int
    reused_cells: int


@dataclass(frozen=True)
class RefinementPlan:
    """Read-only execution identity returned by ``plan_refinement``."""

    run_id: str
    lane_id: str
    lane_definition_sha256: str
    config_path: str
    config_sha256: str
    executor_id: str
    executor_source_sha256: str
    source_json: str
    environment_json: str
    expected_cells: tuple[str, ...]
    project_root: Path
    output_root: Path
    run_directory: Path

    @property
    def source(self) -> dict[str, Any]:
        return json.loads(self.source_json)

    @property
    def environment(self) -> dict[str, Any]:
        return json.loads(self.environment_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "environment": self.environment,
            "executor": self.executor_id,
            "executor_source_sha256": self.executor_source_sha256,
            "expected_cells": list(self.expected_cells),
            "lane_definition_sha256": self.lane_definition_sha256,
            "lane_id": self.lane_id,
            "matrix_cells": len(self.expected_cells),
            "output_root": str(self.output_root),
            "project_root": str(self.project_root),
            "run_directory": str(self.run_directory),
            "run_id": self.run_id,
            "source": self.source,
        }


def load_executor(specification: str) -> RefinementExecutor:
    """Load a local ``module:function`` executor without evaluating strings."""
    if specification.count(":") != 1:
        raise RefinementRunnerError("executor must be specified as module:function")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise RefinementRunnerError("executor must be specified as module:function")
    try:
        module = importlib.import_module(module_name)
        executor = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise RefinementRunnerError(
            f"cannot load executor {specification!r}: {exc}"
        ) from exc
    if not callable(executor):
        raise RefinementRunnerError(f"executor {specification!r} is not callable")
    return executor


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(root: Path, arguments: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def source_provenance(project_root: Path) -> dict[str, Any]:
    """Hash tracked and untracked source changes that can affect a lane run."""
    root = project_root.resolve()
    include_paths = (
        "perovskite_sim",
        "scripts/run_numerical_refinement.py",
        "pyproject.toml",
        # D9.4a: the SCAPS reference importers. A lane whose config or suite
        # manifest was produced by one of these is only as reproducible as the
        # transform that produced it, and an UNCOMMITTED edit to an importer
        # would otherwise leave the fingerprint — and therefore the run id —
        # unchanged. The reproducibility payload hashes configs, raw exports,
        # decks and manifests; the code that reads them was the one artifact
        # in that chain with no hash of its own.
        #
        # This pins WHICH transform ran, not that it is correct. It closes no
        # part of the external-validation gap, which needs a real SCAPS deck.
        "scripts/extract_scaps_reference.py",
        "scripts/import_scaps_cbo_reference.py",
        "scripts/import_scaps_defect_reference.py",
    )
    try:
        commit = _run_git(root, ["rev-parse", "HEAD"])
        tracked_output = _run_git(
            root,
            [
                "diff",
                "--relative",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                "HEAD",
                "--",
                *include_paths,
            ],
        )
        untracked_output = _run_git(
            root,
            [
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *include_paths,
            ],
        )
        changed_paths = sorted(
            set(tracked_output.splitlines()) | set(untracked_output.splitlines())
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        commit = "unknown"
        changed_paths = sorted(
            path.relative_to(root).as_posix()
            for base in include_paths
            for path in [root / base]
            if path.is_file()
        )
        changed_paths.extend(
            sorted(
                path.relative_to(root).as_posix()
                for path in (root / "perovskite_sim").rglob("*.py")
            )
        )
        changed_paths = sorted(set(changed_paths))

    changes: list[dict[str, str]] = []
    for relative in changed_paths:
        path = root / relative
        if path.is_file():
            changes.append({"path": relative, "sha256": _file_sha256(path)})
        else:
            changes.append({"path": relative, "sha256": "deleted"})
    payload = {
        "commit": commit,
        "source_changes": changes,
        "source_scope": list(include_paths),
    }
    return {
        **payload,
        "fingerprint_sha256": content_sha256(payload),
    }


def runtime_environment() -> dict[str, Any]:
    thread_variables = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    return {
        "behavior_variables": {
            name: os.environ.get(name, "unset")
            for name in _BEHAVIOR_ENVIRONMENT_VARIABLES
        },
        "blas_threads": {
            name: os.environ.get(name, "unset") for name in thread_variables
        },
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyyaml": yaml.__version__,
        "scipy": scipy.__version__,
    }


def _executor_source_sha256(executor: RefinementExecutor) -> str:
    try:
        source = inspect.getsource(executor)
    except (OSError, TypeError):
        source = (
            f"{executor.__module__}:{getattr(executor, '__qualname__', repr(executor))}"
        )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _executor_identity(
    lane: LaneDefinition,
    executor: RefinementExecutor,
    executor_id: str | None,
) -> str:
    identity = executor_id or lane.executor
    if identity is None:
        module = getattr(executor, "__module__", type(executor).__module__)
        name = (
            getattr(executor, "__qualname__", None)
            or getattr(executor, "__name__", None)
            or type(executor).__qualname__
        )
        identity = f"{module}:{name}"
    if identity.count(":") != 1:
        raise RefinementRunnerError(
            "executor_id must be a stable module:function identifier"
        )
    return identity


def _safe_output_root(output_root: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    output = output_root if output_root.is_absolute() else root / output_root
    output = output.resolve()
    if output == root:
        raise RefinementRunnerError("output root cannot be the project root")
    for protected_relative in _PROTECTED_OUTPUTS:
        protected = (root / protected_relative).resolve()
        if output == protected or protected in output.parents:
            raise RefinementRunnerError(
                f"refinement output cannot be written under historical "
                f"reference directory {protected_relative}"
            )
    return output


def _write_immutable_json(
    directory: Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    data = canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(data).hexdigest()
    path = directory / f"{digest}.json"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RefinementRunnerError(
                f"immutable artifact collision or corruption at {path}"
            )
    return path, digest


def _write_named_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RefinementRunnerError(
                f"immutable artifact collision or corruption at {path}"
            )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefinementRunnerError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RefinementRunnerError(f"{path}: expected a JSON object")
    return raw


def _validate_measurement_contract(
    lane: LaneDefinition,
    measurement: CellMeasurement,
) -> None:
    observable_gates = {gate.metric: gate for gate in lane.observables}
    quality_gates = {gate.metric: gate for gate in lane.quality_gates}
    observables = {metric.name: metric for metric in measurement.observables}
    quality = {metric.name: metric for metric in measurement.quality}
    if set(observables) != set(observable_gates):
        raise NumericalCertificateError(
            "executor observables do not exactly match the registered lane contract"
        )
    if set(quality) != set(quality_gates):
        raise NumericalCertificateError(
            "executor quality metrics do not exactly match the registered lane contract"
        )
    for name, gate in observable_gates.items():
        if observables[name].units != gate.units:
            raise NumericalCertificateError(
                f"executor observable {name} units do not match the lane contract"
            )
    for name, gate in quality_gates.items():
        if quality[name].units != gate.units:
            raise NumericalCertificateError(
                f"executor quality metric {name} units do not match the lane contract"
            )


def _manifest_protocol_records(
    run_directory: Path,
    references: Mapping[str, ArtifactReference],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for reference in references.values():
        if reference.status != "completed":
            continue
        artifact = _read_json(run_directory / reference.path)
        cell_raw = artifact.get("cell")
        if not isinstance(cell_raw, Mapping):
            continue
        metadata = cell_raw.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        protocol = metadata.get("protocol")
        protocol_hash = metadata.get("protocol_hash")
        protocol_schema = metadata.get("protocol_schema")
        if (
            not isinstance(protocol, Mapping)
            or not isinstance(protocol_hash, str)
            or not isinstance(protocol_schema, str)
            or protocol.get("schema_version") != protocol_schema
            or content_sha256(dict(protocol)) != protocol_hash
        ):
            continue
        record = {
            "protocol": dict(protocol),
            "protocol_hash": protocol_hash,
            "protocol_schema": protocol_schema,
        }
        existing = records.get(protocol_hash)
        if existing is not None and existing != record:
            raise RefinementRunnerError("protocol content-hash collision")
        records[protocol_hash] = record
    return [records[key] for key in sorted(records)]


def _manifest_payload(
    *,
    run_id: str,
    lane: LaneDefinition,
    executor_id: str,
    executor_source_sha256: str,
    source: Mapping[str, Any],
    environment: Mapping[str, Any],
    references: Mapping[str, ArtifactReference],
    run_directory: Path,
) -> dict[str, Any]:
    expected = [point.key for point in lane.matrix_points]
    pending = [key for key in expected if key not in references]
    failures = [
        key
        for key in expected
        if key in references and references[key].status == "failed"
    ]
    if pending:
        status = "running"
    elif failures:
        status = "complete_with_failures"
    else:
        status = "complete"
    return {
        "artifacts": [
            references[key].to_dict() for key in expected if key in references
        ],
        "environment": dict(environment),
        "executor": executor_id,
        "executor_source_sha256": executor_source_sha256,
        "expected_cells": expected,
        "failed_cells": failures,
        "lane_definition_sha256": lane.definition_sha256,
        "lane_id": lane.lane_id,
        "pending_cells": pending,
        "protocols": _manifest_protocol_records(run_directory, references),
        "run_id": run_id,
        "schema_version": MANIFEST_SCHEMA,
        "source": dict(source),
        "status": status,
    }


def _persist_manifest(
    run_directory: Path,
    payload: Mapping[str, Any],
    *,
    latest_certificate_sha256: str | None = None,
) -> tuple[Path, str]:
    manifest_path, manifest_sha = _write_immutable_json(
        run_directory / "manifests",
        payload,
    )
    state = {
        "latest_certificate_sha256": latest_certificate_sha256,
        "latest_manifest_sha256": manifest_sha,
        "run_id": payload["run_id"],
        "schema_version": STATE_SCHEMA,
    }
    _atomic_write_json(run_directory / "state.json", state)
    return manifest_path, manifest_sha


def _load_existing_references(
    run_directory: Path,
    *,
    plan: RefinementPlan,
    lane: LaneDefinition,
) -> tuple[dict[str, ArtifactReference], Path | None, str | None]:
    state_path = run_directory / "state.json"
    if not state_path.exists():
        return {}, None, None
    state = _read_json(state_path)
    _require_exact_keys(state, _STATE_KEYS, "resume state")
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("run_id") != plan.run_id
    ):
        raise RefinementRunnerError("resume state does not match this run")
    manifest_sha = str(state.get("latest_manifest_sha256", ""))
    if not _is_sha256(manifest_sha):
        raise RefinementRunnerError("latest manifest SHA-256 is malformed")
    latest_certificate_sha = state.get("latest_certificate_sha256")
    if latest_certificate_sha is not None and not _is_sha256(latest_certificate_sha):
        raise RefinementRunnerError("latest certificate SHA-256 is malformed")
    manifest_path = run_directory / "manifests" / f"{manifest_sha}.json"
    if not manifest_path.is_file() or _file_sha256(manifest_path) != manifest_sha:
        raise RefinementRunnerError("latest manifest is missing or hash-invalid")
    manifest = _read_json(manifest_path)
    _require_exact_keys(manifest, _MANIFEST_KEYS, "refinement manifest")
    identity = {
        "environment": plan.environment,
        "executor": plan.executor_id,
        "executor_source_sha256": plan.executor_source_sha256,
        "lane_definition_sha256": lane.definition_sha256,
        "lane_id": lane.lane_id,
        "run_id": plan.run_id,
        "schema_version": MANIFEST_SCHEMA,
        "source": plan.source,
    }
    if any(manifest.get(name) != value for name, value in identity.items()):
        raise RefinementRunnerError("latest manifest identity violates the run contract")
    expected = [point.key for point in lane.matrix_points]
    if manifest.get("expected_cells") != expected:
        raise RefinementRunnerError("latest manifest matrix does not match the lane")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RefinementRunnerError("manifest artifacts must be a list")
    references: dict[str, ArtifactReference] = {}
    for raw_reference in raw_artifacts:
        if not isinstance(raw_reference, Mapping):
            raise RefinementRunnerError("manifest artifact reference must be a mapping")
        reference = ArtifactReference.from_mapping(raw_reference)
        if reference.point_key not in expected:
            raise RefinementRunnerError("manifest references an unexpected matrix cell")
        if reference.point_key in references:
            raise RefinementRunnerError("manifest contains duplicate matrix cells")
        if reference.path != f"cells/{reference.sha256}.json":
            raise RefinementRunnerError(
                "manifest cell path is not addressed by its content SHA-256"
            )
        artifact_path = (run_directory / reference.path).resolve()
        try:
            artifact_path.relative_to(run_directory.resolve())
        except ValueError as exc:
            raise RefinementRunnerError("artifact path escapes run directory") from exc
        if (
            not artifact_path.is_file()
            or _file_sha256(artifact_path) != reference.sha256
        ):
            raise RefinementRunnerError(
                f"artifact for {reference.point_key} is missing or hash-invalid"
            )
        artifact = _read_json(artifact_path)
        _require_exact_keys(artifact, _CELL_ARTIFACT_KEYS, "cell artifact")
        if (
            artifact.get("schema_version") != CELL_SCHEMA
            or artifact.get("run_id") != plan.run_id
            or artifact.get("lane_definition_sha256") != lane.definition_sha256
        ):
            raise RefinementRunnerError(
                f"artifact for {reference.point_key} violates the run contract"
            )
        cell_raw = artifact.get("cell")
        if not isinstance(cell_raw, Mapping):
            raise RefinementRunnerError("cell artifact payload is malformed")
        cell = CellResult.from_dict(cell_raw)
        if cell.point.key != reference.point_key or cell.status != reference.status:
            raise RefinementRunnerError("artifact reference does not match its cell")
        references[reference.point_key] = reference
    pending = [key for key in expected if key not in references]
    failures = [
        key
        for key in expected
        if key in references and references[key].status == "failed"
    ]
    status = (
        "running" if pending else "complete_with_failures" if failures else "complete"
    )
    if (
        manifest.get("pending_cells") != pending
        or manifest.get("failed_cells") != failures
        or manifest.get("status") != status
        or manifest.get("protocols")
        != _manifest_protocol_records(run_directory, references)
    ):
        raise RefinementRunnerError("manifest completion state is inconsistent")
    return references, manifest_path, manifest_sha


def _load_cells(
    run_directory: Path,
    references: Mapping[str, ArtifactReference],
    lane: LaneDefinition,
) -> tuple[list[CellResult], list[str]]:
    cells: list[CellResult] = []
    hashes: list[str] = []
    for point in lane.matrix_points:
        reference = references.get(point.key)
        if reference is None:
            continue
        artifact = _read_json(run_directory / reference.path)
        cell_raw = artifact.get("cell")
        if not isinstance(cell_raw, Mapping):
            raise RefinementRunnerError("cell artifact payload is malformed")
        cells.append(CellResult.from_dict(cell_raw))
        hashes.append(reference.sha256)
    return cells, hashes


def _execute_cell(
    executor: RefinementExecutor,
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellResult:
    started = time.perf_counter()
    try:
        raw = executor(lane, point, project_root)
        measurement = (
            raw
            if isinstance(raw, CellMeasurement)
            else CellMeasurement.from_mapping(raw)
        )
        _validate_measurement_contract(lane, measurement)
    except Exception as exc:  # each failed cell must remain independently resumable
        return CellResult(
            point=point,
            status="failed",
            wall_time_s=time.perf_counter() - started,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
        )
    return CellResult(
        point=point,
        status="completed",
        observables=measurement.observables,
        quality=measurement.quality,
        wall_time_s=time.perf_counter() - started,
        metadata_json=measurement.metadata_json,
    )


def plan_refinement(
    lane: LaneDefinition,
    executor: RefinementExecutor,
    *,
    project_root: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    executor_id: str | None = None,
) -> RefinementPlan:
    """Resolve and validate a run identity without writing or executing cells."""
    root = Path(project_root).resolve()
    if not (root / lane.config_path).is_file():
        raise RefinementRunnerError(f"missing lane config {lane.config_path}")
    actual_config_sha = _file_sha256(root / lane.config_path)
    if actual_config_sha != lane.config_sha256:
        raise RefinementRunnerError(
            f"lane config hash drift: {actual_config_sha} != {lane.config_sha256}"
        )

    identity = _executor_identity(lane, executor, executor_id)
    executor_source_sha = _executor_source_sha256(executor)
    source = source_provenance(root)
    environment = runtime_environment()
    run_identity = {
        "config_sha256": lane.config_sha256,
        "environment": environment,
        "executor": identity,
        "executor_source_sha256": executor_source_sha,
        "lane_definition_sha256": lane.definition_sha256,
        "source_fingerprint_sha256": source["fingerprint_sha256"],
    }
    run_id = content_sha256(run_identity)
    destination = _safe_output_root(Path(output_root), root)
    return RefinementPlan(
        run_id=run_id,
        lane_id=lane.lane_id,
        lane_definition_sha256=lane.definition_sha256,
        config_path=lane.config_path,
        config_sha256=lane.config_sha256,
        executor_id=identity,
        executor_source_sha256=executor_source_sha,
        source_json=canonical_json_bytes(source).decode("ascii"),
        environment_json=canonical_json_bytes(environment).decode("ascii"),
        expected_cells=tuple(point.key for point in lane.matrix_points),
        project_root=root,
        output_root=destination,
        run_directory=destination / lane.lane_id / run_id,
    )


def run_refinement(
    lane: LaneDefinition,
    executor: RefinementExecutor,
    *,
    project_root: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    executor_id: str | None = None,
    max_cells: int | None = None,
    retry_failed: bool = False,
) -> RunOutcome:
    """Run or resume a lane without overwriting any scientific artifact."""
    if max_cells is not None and (
        isinstance(max_cells, bool) or int(max_cells) != max_cells or max_cells < 1
    ):
        raise RefinementRunnerError("max_cells must be a positive integer")
    plan = plan_refinement(
        lane,
        executor,
        project_root=project_root,
        output_root=output_root,
        executor_id=executor_id,
    )
    root = plan.project_root
    identity = plan.executor_id
    executor_source_sha = plan.executor_source_sha256
    source = plan.source
    environment = plan.environment
    run_id = plan.run_id
    run_directory = plan.run_directory
    references, manifest_path, manifest_sha = _load_existing_references(
        run_directory,
        plan=plan,
        lane=lane,
    )
    initial_references = set(references)
    executed = 0
    retried = 0

    if manifest_path is None:
        manifest = _manifest_payload(
            run_id=run_id,
            lane=lane,
            executor_id=identity,
            executor_source_sha256=executor_source_sha,
            source=source,
            environment=environment,
            references=references,
            run_directory=run_directory,
        )
        manifest_path, manifest_sha = _persist_manifest(run_directory, manifest)

    for point in lane.matrix_points:
        reference = references.get(point.key)
        if reference is not None and not (
            retry_failed and reference.status == "failed"
        ):
            continue
        if max_cells is not None and executed >= max_cells:
            break
        if reference is not None:
            retried += 1
        cell = _execute_cell(executor, lane, point, root)
        artifact_payload = {
            "cell": cell.to_dict(),
            "lane_definition_sha256": lane.definition_sha256,
            "run_id": run_id,
            "schema_version": CELL_SCHEMA,
        }
        artifact_path, artifact_sha = _write_immutable_json(
            run_directory / "cells",
            artifact_payload,
        )
        references[point.key] = ArtifactReference(
            point_key=point.key,
            status=cell.status,
            path=artifact_path.relative_to(run_directory).as_posix(),
            sha256=artifact_sha,
        )
        executed += 1
        manifest = _manifest_payload(
            run_id=run_id,
            lane=lane,
            executor_id=identity,
            executor_source_sha256=executor_source_sha,
            source=source,
            environment=environment,
            references=references,
            run_directory=run_directory,
        )
        manifest_path, manifest_sha = _persist_manifest(run_directory, manifest)

    if manifest_path is None or manifest_sha is None:
        raise RefinementRunnerError("runner failed to persist a manifest")
    cells, artifact_hashes = _load_cells(run_directory, references, lane)
    certificate = evaluate_numerical_certificate(
        lane,
        cells,
        run_id=run_id,
        source_commit=str(source["commit"]),
        source_fingerprint_sha256=str(source["fingerprint_sha256"]),
        environment=environment,
        manifest_sha256=manifest_sha,
        cell_artifact_sha256=artifact_hashes,
    )
    certificate_path = (
        run_directory / "certificates" / f"{certificate.certificate_sha256}.json"
    )
    _write_named_immutable_json(certificate_path, certificate.to_dict())
    state = _read_json(run_directory / "state.json")
    state["latest_certificate_sha256"] = certificate.certificate_sha256
    _atomic_write_json(run_directory / "state.json", state)
    return RunOutcome(
        certificate=certificate,
        run_directory=run_directory,
        manifest_path=manifest_path,
        certificate_path=certificate_path,
        executed_cells=executed,
        reused_cells=len(initial_references) - retried,
    )

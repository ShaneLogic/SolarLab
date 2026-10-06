"""Spawn transport and trusted infrastructure workers, not scientific executors.

No solver, database, closure or native solver handle crosses this boundary.
Request documents belong to the future P05 authority. A source/schema match
only admits this transport; it does not validate a physical RunSpec. Progress
is replaceable UI metadata. Required state history must be an artifact.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import math
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
from typing import Any, Callable

from solarlab.io.artifacts import ArtifactPayload, _json_document, _parse_document
from solarlab.materials.source import SourceDocument

__all__ = ["WorkerRequest", "WorkerBinding", "ResourceLimits", "WorkerResult", "WorkerContext", "CooperativeStop"]

PROGRESS_BYTES = 4096
REQUEST_BYTES = 262144
TERMINAL_BYTES = 262144


def _name(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("expected a bounded identifier, not a module or path supplied in payload")
    return value


def _sha(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("an explicit SHA-256 identity is required")
    return value


def _object(value: Any) -> str:
    if type(value) is not dict:
        raise ValueError("an explicit strict JSON object is required")
    return _json_document(value)


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request: SourceDocument
    registry: SourceDocument
    schema: SourceDocument
    execution_identity: SourceDocument
    H_input: str
    H_physics: str

    def __post_init__(self) -> None:
        for document in (self.request, self.registry, self.schema, self.execution_identity):
            if type(document) is not SourceDocument:
                raise ValueError("all four actual immutable upstream content documents are required")
        _sha(self.H_input)
        _sha(self.H_physics)
        if len(self.to_bytes()) > REQUEST_BYTES:
            raise ValueError("request exceeds bounded inline transport; a future large-document adapter is required")

    def to_document(self) -> dict[str, Any]:
        return {"transport": "solarlab.worker-request.v1", "status": "prepared_pending_dependencies",
                "H_input": self.H_input, "H_physics": self.H_physics,
                "H_execution": self.execution_identity.sha256,
                "documents": {name: {"id": document.id, "sha256": document.sha256,
                    "base64": base64.b64encode(document.content).decode("ascii")}
                    for name in ("request", "registry", "schema", "execution_identity")
                    for document in (getattr(self, name),)}}

    def to_bytes(self) -> bytes:
        return _object(self.to_document()).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> WorkerRequest:
        if type(raw) is not bytes or len(raw) > REQUEST_BYTES:
            raise ValueError("invalid bounded request transport")
        value = _parse_document(raw.decode())
        if type(value) is not dict or set(value) != {"transport", "status", "H_input", "H_physics", "H_execution", "documents"}:
            raise ValueError("unknown request transport fields")
        documents = value["documents"]
        if type(documents) is not dict or set(documents) != {"request", "registry", "schema", "execution_identity"}:
            raise ValueError("incomplete request documents")
        decoded = {}
        for name, item in documents.items():
            if type(item) is not dict or set(item) != {"id", "sha256", "base64"}:
                raise ValueError("invalid source document transport")
            source = SourceDocument(item["id"], base64.b64decode(item["base64"], validate=True))
            if source.sha256 != item["sha256"]:
                raise ValueError("source document digest mismatch")
            decoded[name] = source
        result = cls(decoded["request"], decoded["registry"], decoded["schema"], decoded["execution_identity"],
                     value["H_input"], value["H_physics"])
        if result.to_bytes() != raw:
            raise ValueError("request transport identity/schema differs from canonical content")
        return result


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    wall_seconds: float
    rss_bytes: int
    output_bytes: int
    accepted_steps: int | None
    calls: int | None

    def __post_init__(self) -> None:
        if type(self.wall_seconds) not in {int, float} or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("a positive finite owned execution wall limit is required")
        for value in (self.rss_bytes, self.output_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("explicit positive RSS and output limits are required")
        if self.output_bytes < 16384:
            raise ValueError("output allowance must include at least 16 KiB for bounded control records")
        for cap in (self.accepted_steps, self.calls):
            if cap is not None and (type(cap) is not int or not 0 <= cap < 2**63):
                raise ValueError("counter cap must be a nonnegative int64 or explicitly unrequested")


@dataclass(frozen=True, slots=True)
class WorkerBinding:
    """Trusted parent-code allowlist entry; user request data never creates one."""

    id: str
    module: str
    symbol: str
    source_sha256: str
    registry_sha256: str
    schema_sha256: str

    def __post_init__(self) -> None:
        _name(self.id)
        if not self.module or self.module == "__main__" or any(not p.isidentifier() for p in self.module.split(".")) or not self.symbol.isidentifier():
            raise ValueError("worker must be an importable named module-level function")
        for value in (self.source_sha256, self.registry_sha256, self.schema_sha256):
            _sha(value)

    @classmethod
    def from_function(
        cls, id: str, function: Callable[..., Any], *, registry_sha256: str, schema_sha256: str,
    ) -> WorkerBinding:
        if (not inspect.isfunction(function) or function.__closure__ is not None
                or function.__qualname__ != function.__name__ or function.__module__ == "__main__"):
            raise ValueError("closures, nested functions and anonymous workers are not transportable")
        module = importlib.import_module(function.__module__)
        if getattr(module, function.__name__, None) is not function:
            raise ValueError("worker is not the registered module-level function")
        path = Path(inspect.getfile(module))
        if path.suffix != ".py":
            raise ValueError("this preparation requires an explicit Python source worker")
        return cls(id, function.__module__, function.__name__, hashlib.sha256(path.read_bytes()).hexdigest(),
                   registry_sha256, schema_sha256)

    def validate_request(self, request: WorkerRequest) -> None:
        if request.registry.sha256 != self.registry_sha256 or request.schema.sha256 != self.schema_sha256:
            raise ValueError("worker registry/schema content mismatch")


@dataclass(frozen=True, slots=True)
class WorkerResult:
    result_json: str
    artifacts: tuple[tuple[str, ArtifactPayload], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_json", _object(_parse_document(self.result_json)))
        artifacts = tuple(self.artifacts)
        if len(artifacts) > 16 or len({name for name, _ in artifacts}) != len(artifacts):
            raise ValueError("result requires at most sixteen uniquely named artifacts")
        for name, payload in artifacts:
            _name(name)
            if type(payload) is not ArtifactPayload:
                raise ValueError("result artifacts require validated immutable ArtifactPayload records")
        object.__setattr__(self, "artifacts", artifacts)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, artifacts: tuple[tuple[str, ArtifactPayload], ...] = ()) -> WorkerResult:
        return cls(_object(value), artifacts)


class CooperativeStop(BaseException):
    """A requested soft cancellation, independent of worker exceptions."""


class _LimitExceeded(BaseException):
    def __init__(self, reason: str, observed: int) -> None:
        self.reason, self.observed = reason, observed


class WorkerContext:
    """One child only; counters are explicit cumulative observations, not estimates.

    Reporting no counters leaves them unknown. They are never fabricated from
    progress or reset at an output point. IPC primitives here are transport
    synchronization, not native solver handles. A worker cannot create a nested
    multiprocessing pool because its process is daemonic.
    """

    def __init__(self, stop: Any, lock: Any, header: Any, buffer: Any, limits: ResourceLimits) -> None:
        self._stop, self._lock, self._header, self._buffer = stop, lock, header, buffer
        self._limits = limits
        self._counts: tuple[int | None, int | None] = (None, None)
        self._progress = b""
        self._revision = 0

    @property
    def cancelled(self) -> bool:
        return bool(self._stop.is_set())

    def check_cancel(self) -> None:
        if self.cancelled:
            raise CooperativeStop()

    def _publish(self) -> None:
        self._revision += 1
        # A killed peer must never leave the control loop waiting on this lock.
        if self._lock.acquire(timeout=0.05):
            try:
                self._buffer[:len(self._progress)] = self._progress
                self._header[:] = (self._revision, len(self._progress),
                                   *(-1 if x is None else x for x in self._counts))
            finally:
                self._lock.release()

    def progress(self, value: dict[str, Any]) -> None:
        raw = _object(value).encode()
        if len(raw) > PROGRESS_BYTES:
            raise ValueError("replaceable progress exceeds its bounded slot; use an artifact for history")
        self._progress = raw
        self._publish()

    def counters(self, *, accepted_steps: int | None, calls: int | None) -> None:
        values = (accepted_steps, calls)
        for old, new in zip(self._counts, values):
            if new is not None and (type(new) is not int or not 0 <= new < 2**63):
                raise ValueError("counters require cumulative nonnegative int64 observations")
            if old is not None and (new is None or new < old):
                raise ValueError("cumulative counters cannot decrease or become unobserved")
        self._counts = values
        self._publish()
        for name, observed, cap in zip(("accepted_steps", "calls"), values, (self._limits.accepted_steps, self._limits.calls)):
            if observed is not None and cap is not None and observed > cap:
                raise _LimitExceeded(name + "_limit", observed)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = _object(value).encode()
    partial = path.with_suffix(path.suffix + ".part")
    fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _process_info(pids: list[int], *, include_descendants: bool = False) -> dict[int, dict[str, Any]]:
    """Read supported macOS/Linux ps fields; missing measurements remain missing.

    lstart has seconds precision: a mismatch proves an old PID exited, while
    equality never authorizes signalling a restored/unowned process.
    """
    if not pids:
        return {}
    fields = "pid=,ppid=,lstart=,rss="
    argv = ["ps", "-axo", fields] if include_descendants else ["ps", "-p", ",".join(str(p) for p in pids), "-o", fields]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=0.5)
    found = {}
    for line in result.stdout.splitlines():
        words = line.split()
        if len(words) == 8 and words[0].isdigit() and words[1].isdigit() and words[-1].isdigit():
            found[int(words[0])] = {"pid": int(words[0]), "ppid": int(words[1]), "start_lstart": " ".join(words[2:-1]),
                                    "start_precision": "seconds", "rss_bytes": int(words[-1]) * 1024}
    selected = set(pids)
    if include_descendants:
        while True:
            extended = selected | {pid for pid, item in found.items() if item["ppid"] in selected}
            if extended == selected:
                break
            selected = extended
    return {pid: item for pid, item in found.items() if pid in selected}


def _regular_bytes(path: Path, maximum: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("unsafe or oversized generated file")
        return stream.read(maximum + 1)


def _directory_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        info = file.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("worker output must not contain links")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def _child_main(
    raw: bytes, binding: WorkerBinding, limits: ResourceLimits, directory: str,
    token: dict[str, Any], stop: Any, lock: Any, header: Any, buffer: Any,
) -> None:
    """Importable spawn entry. The allowlist binding comes only from parent code."""
    path = Path(directory)
    context = WorkerContext(stop, lock, header, buffer, limits)
    state = "failed"
    result: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    try:
        os.chdir(path)
        log = os.open(path / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.close(log)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes))
        identity = _process_info([os.getpid()]).get(os.getpid())
        _atomic_json(path / "started.json", {"token": token, "pid": os.getpid(), "pgid": os.getpgrp(), "os_identity": identity})
        request = WorkerRequest.from_bytes(raw)
        binding.validate_request(request)
        spec = importlib.util.find_spec(binding.module)
        if spec is None or spec.origin is None or Path(spec.origin).suffix != ".py" or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != binding.source_sha256:
            raise ValueError("worker source content changed since trusted registration")
        module = importlib.import_module(binding.module)
        function = getattr(module, binding.symbol)
        if not inspect.isfunction(function) or function.__qualname__ != binding.symbol or function.__module__ != binding.module:
            raise ValueError("registered worker is not a module-level function")
        context.check_cancel()
        output = function(request, context)
        if type(output) is not WorkerResult:
            raise ValueError("worker must return a validated WorkerResult")
        output = WorkerResult(output.result_json, output.artifacts)
        for name, observed, cap in zip(("accepted_steps", "calls"), context._counts, (limits.accepted_steps, limits.calls)):
            if cap is not None and observed is None:
                raise ValueError("requested counter is unobserved: " + name)
        for index, (name, payload) in enumerate(output.artifacts):
            if _directory_bytes(path) + 2 * len(payload.data) > limits.output_bytes:
                raise _LimitExceeded("output_limit_before_publication", _directory_bytes(path) + 2 * len(payload.data))
            filename = f"artifact-{index}.bin"
            fd = os.open(path / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload.data)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts.append({"name": name, "file": filename, "sha256": payload.sha256,
                              "bytes": len(payload.data), "format": payload.format, "metadata_json": payload.metadata_json})
        state, result = "succeeded", _parse_document(output.result_json)
    except CooperativeStop:
        state, result, artifacts = "cancelled", {"reason": "cooperative_stop"}, []
    except _LimitExceeded as error:
        result, artifacts = {"reason": error.reason, "observed": error.observed}, []
    except BaseException as error:
        result, artifacts = {"reason": "worker_exception", "exception_type": type(error).__name__, "detail": str(error)[:2048]}, []
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest = {"token": token, "worker_source_sha256": binding.source_sha256, "state": state,
                "result": result, "artifacts": artifacts,
                "counters": dict(zip(("accepted_steps", "calls"), context._counts)),
                "child_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
    if len(_object(manifest).encode()) > TERMINAL_BYTES:
        manifest.update(state="failed", result={"reason": "terminal_transport_limit"}, artifacts=[])
    _atomic_json(path / "terminal.json", manifest)

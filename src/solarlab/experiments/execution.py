"""Shared bounded spawn supervision against the real RunStore.

This is prepared_pending_dependencies, not a P05 executor or scheduler service.
One parent monitor writes the store; children receive immutable document bytes
and transport synchronization only. All submitters use the same active/pending
capacity and accounting. No scientific registry, endpoint or cache is activated.

Wall/RSS/disk limits are monitored limits, not a hard realtime sandbox. Receipts
include observed overruns and cleanup time. Counter completeness depends on the
trusted worker's explicit cumulative instrumentation, never UI progress.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid

from solarlab.io.artifacts import ArtifactPayload, _parse_document
from solarlab.io.run_store import AttemptToken, RunStore, StateConflict
from solarlab.experiments.worker import (
    PROGRESS_BYTES, TERMINAL_BYTES, ManifestBinding, ResourceLimits, RunSpec, WorkerBinding, WorkerRequest,
    _atomic_json, _child_main, _directory_bytes, _name, _object, _process_info, _regular_bytes, _sha,
)

__all__ = ["ProcessSupervisor", "JobHandle", "ExecutionReceipt", "QueueFull", "ScanPoint", "ScanSpec", "ScanRunner"]

SCAN_BYTES = 1024**2  # Match the shared strict metadata codec's 1 MiB bound.
MAX_SCAN_POINTS = 1024
_TERMINAL = {"succeeded", "failed", "cancelled"}
_RSS_MAX_MISSING_SAMPLES = 3
_RSS_MAX_GAP_SECONDS = 2.0


class _RSSUnavailable(RuntimeError):
    """The bounded window for enforcing the RSS budget has expired."""


class _AdmissionLimit(RuntimeError):
    """A measured shared limit leaves no capacity to start queued work."""


class QueueFull(RuntimeError):
    """No active reservation or bounded pending position remains."""


@dataclass(frozen=True, slots=True)
class JobHandle:
    store_id: str
    run_id: str
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    document_json: str

    @property
    def data(self) -> dict[str, Any]:
        return _parse_document(self.document_json)


@dataclass(frozen=True, slots=True)
class ScanPoint:
    point_id: str
    run_id: str
    coordinates_json: str
    spec: RunSpec

    def __post_init__(self) -> None:
        _name(self.point_id)
        _name(self.run_id)
        if type(self.spec) is not RunSpec:
            raise ValueError("a scan point requires an explicit RunSpec")
        if type(self.coordinates_json) is not str or len(self.coordinates_json.encode()) > 16384:
            raise ValueError("scan coordinates require a bounded JSON object")
        object.__setattr__(self, "coordinates_json", _object(_parse_document(self.coordinates_json)))

    def to_document(self) -> dict[str, Any]:
        return {"point_id": self.point_id, "run_id": self.run_id,
                "coordinates": _parse_document(self.coordinates_json), "spec": self.spec.to_document()}


@dataclass(frozen=True, slots=True)
class ScanSpec:
    """An ordered declaration of at most 1 MiB; unsubmitted points use no slots."""

    scan_id: str
    points: tuple[ScanPoint, ...]

    def __post_init__(self) -> None:
        _name(self.scan_id)
        points = tuple(self.points)
        if not 1 <= len(points) <= MAX_SCAN_POINTS or any(type(p) is not ScanPoint for p in points):
            raise ValueError("a scan requires between 1 and 1024 explicit point declarations")
        if len({p.point_id for p in points}) != len(points) or len({p.run_id for p in points}) != len(points):
            raise ValueError("scan point and run IDs must be unique within the declaration")
        object.__setattr__(self, "points", points)
        if sum(len(p.spec.to_bytes()) + len(p.coordinates_json.encode()) for p in points) > SCAN_BYTES:
            raise ValueError("scan declaration exceeds 1 MiB; it was not truncated")
        if len(self.to_bytes()) > SCAN_BYTES:
            raise ValueError("scan declaration exceeds 1 MiB; it was not truncated")

    def to_bytes(self) -> bytes:
        return _object({"schema": "solarlab.scan-spec.v1", "scan_id": self.scan_id,
                        "points": [p.to_document() for p in self.points]}).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> ScanSpec:
        if type(raw) is not bytes or len(raw) > SCAN_BYTES:
            raise ValueError("invalid bounded scan transport")
        value = _parse_document(raw.decode())
        if (type(value) is not dict or set(value) != {"schema", "scan_id", "points"}
                or value["schema"] != "solarlab.scan-spec.v1" or type(value["points"]) is not list
                or not 1 <= len(value["points"]) <= MAX_SCAN_POINTS):
            raise ValueError("unknown or incomplete scan declaration")
        points = []
        for point in value["points"]:
            if type(point) is not dict or set(point) != {"point_id", "run_id", "coordinates", "spec"}:
                raise ValueError("invalid scan point fields")
            points.append(ScanPoint(point["point_id"], point["run_id"], _object(point["coordinates"]),
                                    RunSpec.from_bytes(_object(point["spec"]).encode())))
        result = cls(value["scan_id"], tuple(points))
        if result.to_bytes() != raw:
            raise ValueError("scan differs from its canonical declaration")
        return result


@dataclass
class _Job:
    handle: JobHandle
    generation: int
    request: WorkerRequest
    binding: WorkerBinding
    limits: ResourceLimits
    enqueued: float
    execution: ManifestBinding | None = None
    token: AttemptToken | None = None
    process: Any = None
    directory: Path | None = None
    pid: int | None = None
    pgid: int | None = None
    started: float | None = None
    stop: Any = None
    lock: Any = None
    header: Any = None
    buffer: Any = None
    progress_revision: int = 0
    counts: dict[str, int | None] = field(default_factory=lambda: {"accepted_steps": None, "calls": None})
    peak_rss: int | None = None
    output_peak: int = 0
    os_identity: dict[str, Any] | None = None
    stop_reason: str | None = None
    stop_state: str = "failed"
    stop_at: float | None = None
    terminate_at: float | None = None
    kill_at: float | None = None
    signals: list[str] = field(default_factory=list)
    setup_error: str | None = None


class ProcessSupervisor:
    """Explicit owner of a finite pool, local work area and RunStore writer.

    Construction starts only the parent monitor thread. ``submit`` requires an
    allowlisted worker ID; a request cannot supply Python modules or commands.
    ``close`` retains queued rows for explicit matching resubmission and stops
    running children. It succeeds only after the owned children are joined.

    Recovered intents reserve only their affected slots. Explicit reconciliation
    fences their recorded tokens, never all store runs, and never signals an
    unowned/restored PID. Absence or a changed OS start identity proves that old
    process exited; equal coarse timestamps remain uncertain.
    """

    def __init__(
        self, store: RunStore, work_root: str | Path, workers: tuple[WorkerBinding, ...], *,
        max_active: int, max_pending: int, shared_rss_bytes: int, shared_output_bytes: int,
        soft_grace_seconds: float = 0.2, terminate_grace_seconds: float = 0.2,
        poll_seconds: float = 0.025,
    ) -> None:
        if multiprocessing.current_process().daemon:
            raise ValueError("workers cannot create nested supervisors or scans")
        if not isinstance(store, RunStore) or not workers or any(type(w) is not WorkerBinding for w in workers):
            raise ValueError("a real store and explicit trusted worker allowlist are required")
        if len({w.id for w in workers}) != len(workers):
            raise ValueError("duplicate trusted worker ID")
        if type(max_active) is not int or max_active < 1 or type(max_pending) is not int or max_pending < 0:
            raise ValueError("finite positive active and nonnegative pending capacities are required")
        for value in (shared_rss_bytes, shared_output_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("explicit shared resource caps are required")
        for interval in (soft_grace_seconds, terminate_grace_seconds, poll_seconds):
            if type(interval) not in {int, float} or not math.isfinite(interval) or interval <= 0:
                raise ValueError("bounded positive monitor and stop intervals are required")
        root = Path(work_root)
        if (not root.is_absolute() or root.resolve() != root or root == store.root
                or root.is_relative_to(store.root) or store.root.is_relative_to(root)):
            raise ValueError("work root must be a canonical local path disjoint from the store")
        root.mkdir(mode=0o700, exist_ok=True)
        self._lock_fd = os.open(root / ".supervisor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self._lock_fd)
            raise
        self.store, self.work_root = store, root
        self.owner_id = uuid.uuid4().hex
        self._workers = {w.id: w for w in workers}
        self.max_active, self.max_pending = max_active, max_pending
        self._rss_cap, self._output_cap = shared_rss_bytes, shared_output_bytes
        self._soft, self._terminate, self._poll = soft_grace_seconds, terminate_grace_seconds, poll_seconds
        self._condition = threading.Condition(threading.RLock())
        self._context = multiprocessing.get_context("spawn")
        self._pending: deque[_Job] = deque()
        self._active: dict[str, _Job] = {}
        self._known: dict[str, _Job] = {}
        self._done: dict[str, ExecutionReceipt] = {}
        self._recovery: dict[str, dict[str, Any]] = {}
        self._control_recovery: dict[str, dict[str, Any]] = {}
        self._unknown_history: set[str] = set()
        self._scans: dict[str, ScanRunner] = {}
        self._closing = False
        self._closed = False
        self._fatal: BaseException | None = None
        self._parent_rss: int | None = None
        self._shared_rss: int | None = None
        self._known_family_rss: int | None = None
        self._unlocated_prior_intents = 0
        self._sampled_family_pids: list[int] = []
        self._shared_peak_rss: int | None = None
        self._rss_sample_at: float | None = None
        self._rss_missing_since: float | None = None
        self._rss_missing_count = 0
        self._rss_failures_total = 0
        self._rss_sample_error: str | None = None
        self._control_cleanup_errors: list[str] = []
        self._control_audit_errors: dict[str, list[str]] = {}
        self._control_retention: dict[str, list[str]] = {}
        try:
            self._store_baseline = _directory_bytes(store.root)
            self._shared_output_peak = _directory_bytes(root)
            for directory in sorted(root.glob(store.store_id + "-*")):
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("unsafe previous owned work directory")
                receipt = self._discover_control_failure(directory)
                if receipt is not None:
                    if (directory / "finalization.json").exists():
                        audit = self._read_json(directory / "finalization.json")
                        if audit.get("token") != receipt.get("token"):
                            raise ValueError("finalization audit belongs to a different attempt")
                        self._shared_output_peak = max(self._shared_output_peak, audit["shared_output_peak_bytes"])
                        receipt = {**receipt, "finalization": audit}
                        if (directory / "finalization_overrun.json").exists():
                            overrun = self._read_json(directory / "finalization_overrun.json")
                            if overrun.get("token") != receipt.get("token"):
                                raise ValueError("finalization overrun belongs to a different attempt")
                            self._shared_output_peak = max(self._shared_output_peak, overrun["observed_bytes"])
                            receipt = {**receipt, "finalization_overrun": overrun}
                    self._done[receipt["attempt_id"]] = ExecutionReceipt(_object(receipt))
                    if receipt.get("terminal_committed") and (receipt.get("process_started") is False
                            or receipt.get("process_exit_observed") is True and receipt.get("process_joined") is True):
                        continue
                if (directory / "launch.json").exists():
                    launch = self._read_json(directory / "launch.json")
                    token = AttemptToken(**launch["token"])
                    if token.store_id != store.store_id or directory.name != store.store_id + "-" + token.attempt_id:
                        raise ValueError("previous process intent belongs to a different store/attempt")
                    if (directory / "reconciled.json").exists():
                        recovered = self._read_json(directory / "reconciled.json")
                        if recovered.get("token") != asdict(token) or recovered.get("evidence", {}).get("process_exit_observed") is not True:
                            raise ValueError("invalid prior process reconciliation")
                        self._unknown_history.add(token.attempt_id)
                    else:
                        self._recovery[token.attempt_id] = {"directory": directory, "launch": launch}
                else:
                    entry = self._control_recovery.get(directory.name[len(store.store_id) + 1:])
                    if entry is not None and entry["intent"] is not None:
                        process = entry["intent"]["result"]["process"]
                        if process.get("process_started") and process.get("token") is not None:
                            token = AttemptToken(**process["token"])
                            self._recovery[token.attempt_id] = {"directory": directory,
                                "launch": {"token": asdict(token)}, "process": process}
            self._thread = threading.Thread(target=self._loop, name="solarlab-process-supervisor", daemon=True)
            self._thread.start()
        except BaseException:
            os.close(self._lock_fd)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = _parse_document(_regular_bytes(path, TERMINAL_BYTES).decode())
        _object(value)
        return value

    def _control_error(self, attempt_id: str, operation: str, error: Exception) -> None:
        message = operation + ": " + type(error).__name__ + ": " + str(error)[:512]
        errors = self._control_audit_errors.setdefault(attempt_id, [])
        if message not in errors:
            errors.append(message)
            self._control_cleanup_errors.append(attempt_id + "/" + message)

    def _control_binding(self, directory: Path, intent: dict[str, Any]) -> tuple[JobHandle, AttemptToken | None]:
        handle = JobHandle(**intent["handle"])
        generation, owner = intent["generation"], intent["owner_id"]
        if (intent["schema"] != "solarlab.supervisor-control-failure.v1"
                or handle.store_id != self.store.store_id
                or directory.name != handle.store_id + "-" + handle.attempt_id
                or type(generation) is not int or generation < 0
                or owner is not None and (type(owner) is not str or not owner)
                or intent["result"]["worker_result"]["reason"] != "supervisor_control_failure"):
            raise ValueError("invalid prior control-failure intent")
        token = AttemptToken(**intent["token"]) if intent["token"] is not None else None
        if token is not None and (token.store_id != handle.store_id or token.run_id != handle.run_id
                or token.attempt_id != handle.attempt_id or token.owner_id != owner or token.generation != generation):
            raise ValueError("control-failure token differs from its retained binding")
        process_token = intent["result"]["process"].get("token")
        if process_token is not None:
            saved = AttemptToken(**process_token)
            if (saved.store_id != handle.store_id or saved.run_id != handle.run_id
                    or saved.attempt_id != handle.attempt_id or saved.owner_id != owner
                    or saved.generation != generation + (token is None)):
                raise ValueError("control-failure process differs from its retained binding")
        return handle, token

    def _fallback_control_intent(self, directory: Path, record: dict[str, Any]) -> dict[str, Any]:
        """Read historical failure evidence; never reconstruct or run a request."""
        if "control_intent" in record:
            return dict(record["control_intent"])
        saved_token = record.get("token")
        if saved_token is None:
            try:
                saved_token = self._read_json(directory / "launch.json")["token"]
            except FileNotFoundError:
                pass
        token = AttemptToken(**saved_token) if saved_token is not None else None
        run_id = record.get("run_id", token.run_id if token else None)
        attempt_id = record.get("attempt_id", token.attempt_id if token else None)
        if type(run_id) is not str or type(attempt_id) is not str:
            raise ValueError("failure evidence lacks its run/attempt binding")
        handle = JobHandle(self.store.store_id, run_id, attempt_id)
        prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == handle.attempt_id)
        if token is None and (prior["owner_id"] is not None or prior["started_ns"] is not None):
            raise StateConflict("failure evidence lacks the claimed owner/generation; administrative fencing required")
        process = {key: record.get(key) for key in (
            "pid", "pgid", "os_identity", "exitcode", "signals", "process_started",
            "process_joined", "process_exit_observed")}
        process["token"] = asdict(token) if token is not None else None
        if "process_started" not in record:
            process["process_started"] = type(process["pid"]) is int and process["pid"] > 0
        resources = record.get("resources", {key: None for key in (
            "owned_wall_seconds", "queue_seconds", "child_peak_rss_bytes", "output_bytes", "accepted_steps", "calls")})
        failure = record.get("control_failure") or record.get("worker_result") or {
            "reason": "supervisor_control_failure", "control_reason": "retained_control_exit_evidence"}
        return {"schema": "solarlab.supervisor-control-failure.v1", "handle": asdict(handle),
                "generation": token.generation if token else prior["generation"],
                "owner_id": token.owner_id if token else None, "token": process["token"],
                "result": {"status": "prepared_pending_dependencies", "worker_result": failure,
                    "process": process, "resources": resources,
                    "counter_observation_kind": "worker_reported_or_null_unobserved"}}

    def _discover_control_failure(self, directory: Path) -> dict[str, Any] | None:
        """Load all available evidence, including complete interrupted writes.

        Nonterminal receipts are recovery input, never completed submissions.
        A damaged record is retained and reported without hiding other attempts.
        """
        attempt_id = directory.name[len(self.store.store_id) + 1:]
        receipt: dict[str, Any] | None = None
        intent: dict[str, Any] | None = None
        selected_rank = 4
        control = attempt_id in self._control_recovery
        paths = [directory / name for name in ("control-failure.json", "receipt.json", "exit.json")]
        for name in ("control-failure.json", "receipt.json", "exit.json"):
            paths.append(directory / (name + ".part"))
            paths.extend(sorted(directory.glob(name + ".recovery-*")))
        for path in paths:
            try:
                record = self._read_json(path)
            except FileNotFoundError:
                continue
            except Exception as error:
                self._control_error(attempt_id, path.name, error)
                control = True
                continue
            if path.name == "receipt.json":
                receipt = record
            is_intent = path.name.startswith("control-failure.json")
            if not (is_intent or record.get("control_failure") or record.get("control_intent")
                    or record.get("reason") == "supervisor_control_failure"
                    or record.get("worker_result", {}).get("reason") == "supervisor_control_failure"):
                continue
            control = True
            try:
                candidate = record if is_intent else self._fallback_control_intent(directory, record)
                handle, token = self._control_binding(directory, candidate)
                prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == handle.attempt_id)
                if prior["state"] not in _TERMINAL:
                    if prior["state"] == "queued":
                        if token is not None or candidate["generation"] != prior["generation"]:
                            raise StateConflict("failure evidence differs from the queued generation")
                    elif (candidate["owner_id"] is None or candidate["owner_id"] != prior["owner_id"]
                            or candidate["generation"] + (token is None) != prior["generation"]):
                        raise StateConflict("failure evidence differs from the claimed owner/generation")
            except Exception as error:
                self._control_error(attempt_id, path.name, error)
                continue
            retained = self._control_retention.setdefault(attempt_id, [])
            if path.name not in retained:
                retained.append(path.name)
            rank = 0 if is_intent else 1 if "control_intent" in record else 2 if path.name.startswith("receipt.json") else 3
            if rank < selected_rank:
                intent = candidate
                selected_rank = rank
        if not control:
            return receipt
        if receipt is not None and intent is not None:
            handle, _ = self._control_binding(directory, intent)
            prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == attempt_id)
            if (receipt.get("terminal_committed") is True and prior["state"] in _TERMINAL
                    and receipt.get("run_id") == handle.run_id and receipt.get("attempt_id") == attempt_id
                    and receipt.get("store_state") == prior["state"]
                    and receipt.get("worker_result") == prior["result"].get("worker_result", prior["result"])):
                return receipt  # Preserve an already current terminal receipt's bytes.
        committed = False
        if intent is not None:
            handle, token = self._control_binding(directory, intent)
            prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == attempt_id)
            committed = prior["state"] in _TERMINAL
            if committed:
                self._done[attempt_id] = ExecutionReceipt(_object(self._control_receipt(intent, prior, token)))
        self._control_recovery[attempt_id] = {"directory": directory, "intent": intent,
                                             "terminal_committed": committed}
        return None

    @staticmethod
    def _control_receipt(intent: dict[str, Any], prior: dict[str, Any], token: AttemptToken | None) -> dict[str, Any]:
        result, handle = intent["result"], intent["handle"]
        return {"status": "prepared_pending_dependencies", "run_id": handle["run_id"],
            "attempt_id": handle["attempt_id"], **result["process"],
            "token": asdict(token) if token is not None else result["process"].get("token"),
            "store_state": prior["state"], "proposed_state": "failed", "terminal_committed": True,
            "commit_error": None, "resources": result["resources"],
            "worker_result": prior["result"].get("worker_result", prior["result"]),
            "terminal_artifacts": [a["artifact_id"] for a in prior["terminal"]["artifacts"]],
            "publication_error": None, "control_failure": result["worker_result"], "control_intent": intent}

    def _check_open(self) -> None:
        if self._fatal is not None:
            raise RuntimeError("supervisor control failure; inspect retained process records") from self._fatal
        if self._closing or self._closed:
            raise RuntimeError("supervisor is closing or closed")
        if any(not entry["terminal_committed"] for entry in self._control_recovery.values()):
            raise RuntimeError("uncommitted control failures require explicit reconciliation before submission")

    def _submission(self, worker_id: str, request: WorkerRequest, limits: ResourceLimits,
                    execution: ManifestBinding | None) -> tuple[dict[str, Any], dict[str, Any]]:
        if type(request) is not WorkerRequest or type(limits) is not ResourceLimits:
            raise ValueError("explicit immutable request documents and resource limits are required")
        if worker_id not in self._workers:
            raise ValueError("worker ID is not in the trusted code allowlist")
        binding = self._workers[worker_id]
        binding.validate_request(request)
        if limits.rss_bytes > self._rss_cap or limits.output_bytes > self._output_cap:
            raise ValueError("request resource limit exceeds the explicit shared capacity")
        document = request.to_document()
        metadata = {"transport": document, "worker_binding": asdict(binding), "limits": asdict(limits)}
        if execution is not None:
            if type(execution) is not ManifestBinding:
                raise ValueError("explicit trusted manifest binding required")
            execution.validate_request(request)
            if len(execution.manifest.canonical_json) * 4 + 16384 > limits.output_bytes:
                raise ValueError("output allowance cannot retain the complete execution manifest")
            metadata["execution_manifest"] = execution.summary()
        identity = {key: document[key] for key in ("H_input", "H_physics", "H_execution")}
        return identity, metadata

    def submit(self, run_id: str, worker_id: str, request: WorkerRequest, limits: ResourceLimits, *,
               execution: ManifestBinding | None = None) -> JobHandle:
        identity, metadata = self._submission(worker_id, request, limits, execution)
        binding = self._workers[worker_id]
        with self._condition:
            self._check_open()
            try:
                previous = self.store.get_run(run_id)
            except KeyError:
                previous = None
            if previous is not None:
                # Matching duplicates are idempotent even when the queue is full.
                self.store.enqueue(run_id, opaque_identity=identity, input_metadata=metadata)
                old = previous["current_attempt_id"]
                if old in self._known or old in self._done:
                    return JobHandle(self.store.store_id, run_id, old)
                if previous["state"] != "queued":
                    raise StateConflict("only an explicitly queued attempt can be resumed; retries belong to RunStore")
            if len(self._active) + len(self._pending) + len(self._recovery) >= self.max_active + self.max_pending:
                raise QueueFull("shared active reservations and pending queue are full")
            record = self.store.enqueue(run_id, opaque_identity=identity, input_metadata=metadata)
            handle = JobHandle(self.store.store_id, run_id, record["current_attempt_id"])
            job = _Job(handle, record["generation"], request, binding, limits, time.monotonic(), execution)
            self._known[handle.attempt_id] = job
            self._pending.append(job)
            self._condition.notify_all()
            return handle

    def scan(self, spec: ScanSpec, *, executions: tuple[ManifestBinding, ...], max_in_flight: int) -> ScanRunner:
        """Parent-side coordination using this exact server/sweep resource owner.

        No coordinator process or second pool is created. The caller drives
        ``advance`` or bounded ``wait``; only admitted points occupy capacity.
        """
        with self._condition:
            self._check_open()
            if type(spec) is not ScanSpec:
                raise ValueError("an explicit immutable scan declaration is required")
            previous = self._scans.get(spec.scan_id)
            if previous is not None:
                if previous.spec != spec or previous.executions != tuple(executions) or previous.max_in_flight != max_in_flight:
                    raise StateConflict("scan coordinator already exists with different inputs or capacity")
                return previous
            runner = ScanRunner(self, spec, tuple(executions), max_in_flight=max_in_flight)
            self._scans[spec.scan_id] = runner
            return runner

    def cancel(self, handle: JobHandle, *, reason: str = "cancel_requested") -> bool:
        if type(reason) is not str or not reason.strip():
            raise ValueError("cancellation requires an explicit reason")
        with self._condition:
            try:
                return self._cancel(handle, reason)
            except (StateConflict, KeyError):
                raise
            except BaseException as error:
                self._abort_control(error)
                raise RuntimeError("supervisor failed during cancellation") from error

    def _cancel(self, handle: JobHandle, reason: str) -> bool:
        """Unexpected cancellation side-effect failures abort the owned pool."""
        with self._condition:
            if handle.store_id != self.store.store_id:
                raise StateConflict("foreign store handle")
            if handle.attempt_id in self._done:
                return False
            job = self._known[handle.attempt_id]
            if job.handle != handle:
                raise StateConflict("stale run/attempt handle")
            if job in self._pending:
                self.store.cancel(handle.run_id, expected_attempt_id=handle.attempt_id,
                                  expected_generation=job.generation, reason=reason)
                self._pending.remove(job)
                self._done[handle.attempt_id] = ExecutionReceipt(_object({
                    "status": "prepared_pending_dependencies", "run_id": handle.run_id, "attempt_id": handle.attempt_id,
                    "store_state": "cancelled", "terminal_committed": True, "reason": reason,
                    "process_started": False, "process_exit_observed": None, "process_joined": False,
                    "resources": {"owned_wall_seconds": 0.0, "queue_seconds": time.monotonic() - job.enqueued,
                                  "accepted_steps": None, "calls": None, "child_peak_rss_bytes": None, "output_bytes": 0}}))
                directory = self.work_root / (self.store.store_id + "-" + handle.attempt_id)
                directory.mkdir(mode=0o700)
                _atomic_json(directory / "receipt.json", self._done[handle.attempt_id].data)
                self._condition.notify_all()
                return True
            if job.stop_reason is not None:
                return False
            self._request_stop(job, reason, "cancelled")
            self._condition.notify_all()
            return True

    def wait(self, handle: JobHandle, timeout: float) -> ExecutionReceipt:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("wait requires a positive finite timeout")
        deadline = time.monotonic() + timeout
        with self._condition:
            recovery = self._control_recovery.get(handle.attempt_id)
            if self._fatal is None and recovery is not None and not recovery["terminal_committed"]:
                raise RuntimeError("uncommitted control failure requires explicit reconciliation before readback")
            while handle.attempt_id not in self._done:
                if self._fatal is not None:
                    raise RuntimeError("supervisor failed; child cleanup is still owned until close returns") from self._fatal
                if self._closed:
                    raise RuntimeError("queued work was retained; explicitly resubmit its matching documents after restart")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("attempt has not produced a joined-process receipt")
                self._condition.wait(remaining)
            receipt = self._done[handle.attempt_id]
            if handle.store_id != self.store.store_id or receipt.data["run_id"] != handle.run_id:
                raise StateConflict("foreign/stale result handle")
            return receipt

    def _request_stop(self, job: _Job, reason: str, state: str) -> None:
        if job.stop_reason is not None:
            return
        assert job.token is not None and job.directory is not None
        job.stop_reason, job.stop_state, job.stop_at = reason, state, time.monotonic()
        decision = {"token": asdict(job.token), "reason": reason, "state": state, "requested_monotonic": job.stop_at}
        if job.setup_error is not None:
            decision["setup_error"] = job.setup_error
        _atomic_json(job.directory / "stop.json", decision)
        job.stop.set()
        job.signals.append("soft_stop")
        try:
            self.store.append_event(job.token, event_id="stop_requested", payload=decision)
        except StateConflict:
            pass  # Revoked authority is retained; signalling still targets our owned child.

    def _start(self, job: _Job) -> None:
        job.token = self.store.claim(job.handle.run_id, expected_attempt_id=job.handle.attempt_id,
                                     expected_generation=job.generation, owner_id=self.owner_id)
        job.generation = job.token.generation
        job.started = time.monotonic()
        job.directory = self.work_root / (self.store.store_id + "-" + job.handle.attempt_id)
        job.directory.mkdir(mode=0o700)
        if job.execution is not None:
            # This owned launch input remains available even if the child
            # crashes before returning or refuses the actual environment.
            with (job.directory / "execution-manifest.json").open("xb") as stream:
                stream.write(job.execution.manifest.canonical_json)
                stream.flush()
                os.fsync(stream.fileno())
        _atomic_json(job.directory / "launch.json", {"schema": "solarlab.process-intent.v1", "token": asdict(job.token),
            "request_sha256": hashlib.sha256(job.request.to_bytes()).hexdigest(), "binding": asdict(job.binding),
            "limits": asdict(job.limits), "started_monotonic": job.started, "owner_pid": os.getpid()})
        job.stop = self._context.Event()
        job.lock = self._context.Lock()
        job.header = self._context.RawArray("q", (0, 0, -1, -1))
        job.buffer = self._context.RawArray("B", PROGRESS_BYTES)
        job.process = self._context.Process(target=_child_main, args=(job.request.to_bytes(), job.binding, job.limits,
            str(job.directory), asdict(job.token), job.stop, job.lock, job.header, job.buffer, job.execution), daemon=True)
        self._active[job.handle.attempt_id] = job
        job.process.start()
        job.pid = job.process.pid
        assert job.pid is not None
        try:
            job.pgid = os.getpgid(job.pid)
        except ProcessLookupError:
            pass
        _atomic_json(job.directory / "spawned.json", {"token": asdict(job.token), "pid": job.pid, "pgid": job.pgid})

    def _progress(self, job: _Job) -> None:
        if not job.lock.acquire(False):
            return
        try:
            revision, length, steps, calls = job.header[:]
            raw = bytes(job.buffer[:length]) if 0 <= length <= PROGRESS_BYTES else b""
        finally:
            job.lock.release()
        if revision < job.progress_revision or not 0 <= length <= PROGRESS_BYTES:
            raise ValueError("corrupt shared progress header")
        for name, value in (("accepted_steps", steps), ("calls", calls)):
            old = job.counts[name]
            observed = None if value == -1 else value
            if observed is not None and (observed < 0 or old is not None and observed < old):
                raise ValueError("corrupt decreasing cumulative counter")
            if observed is not None:
                job.counts[name] = observed
        if revision > job.progress_revision and raw:
            assert job.token is not None
            try:
                self.store.append_event(job.token, event_id=f"ui-{revision}", payload={"ui_progress": _parse_document(raw.decode()),
                                        "coalesced_revision": revision})
            except StateConflict:
                pass
        job.progress_revision = revision

    def _unstarted(self, job: _Job, error: BaseException) -> None:
        if job.process is not None and job.process.pid is not None:
            job.pid = job.process.pid
            job.setup_error = type(error).__name__ + ": " + str(error)[:1024]
            self._request_stop(job, "spawn_setup_failed", "failed")
            return  # Keep this exact child and slot owned through normal bounded stop/join.
        committed = False
        reason = str(error) if isinstance(error, _AdmissionLimit) else "spawn_failed"
        result = {"reason": reason, "detail": str(error)[:1024], "process_started": False}
        try:
            if job.token is not None:
                self.store.complete(job.token, state="failed", result=result)
            else:
                self.store.fail_queued(job.handle.run_id, expected_attempt_id=job.handle.attempt_id,
                                       expected_generation=job.generation, result=result)
            committed = True
        except StateConflict:
            pass
        prior = next(a for a in self.store.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)
        if prior["state"] == "running" and prior["owner_id"] == self.owner_id:
            # A claim can commit before returning its token. Do not strand our
            # own generation in the expected spawn-error/conflict fallback.
            raise RuntimeError("unstarted owned claim has no terminal state") from error
        receipt = {"status": "prepared_pending_dependencies", "run_id": job.handle.run_id,
            "attempt_id": job.handle.attempt_id, "store_state": prior["state"], "terminal_committed": committed,
            "reason": reason if isinstance(error, _AdmissionLimit) else "spawn_failed_or_queued_authority_revoked",
            "worker_result": result, "detail": str(error)[:1024],
            "process_started": False, "process_exit_observed": None, "process_joined": False,
            "resources": {"owned_wall_seconds": time.monotonic() - job.started if job.started is not None else 0.0,
                "queue_seconds": (job.started or time.monotonic()) - job.enqueued, "accepted_steps": None,
                "calls": None, "child_peak_rss_bytes": None, "output_bytes": 0}}
        self._done[job.handle.attempt_id] = ExecutionReceipt(_object(receipt))
        self._active.pop(job.handle.attempt_id, None)
        if job.directory is not None and job.directory.is_dir() and not (job.directory / "receipt.json").exists():
            _atomic_json(job.directory / "receipt.json", receipt)
        if job.process is not None:
            job.process.close()
        self._condition.notify_all()

    def _resource_reason(self, job: _Job, now: float) -> str | None:
        assert job.started is not None and job.directory is not None
        job.output_peak = max(job.output_peak, _directory_bytes(job.directory))
        if now - job.started > job.limits.wall_seconds:
            return "wall_limit"
        if job.peak_rss is not None and job.peak_rss > job.limits.rss_bytes:
            return "rss_limit"
        if job.output_peak > job.limits.output_bytes:
            return "output_limit"
        for name, cap in (("accepted_steps", job.limits.accepted_steps), ("calls", job.limits.calls)):
            observed = job.counts[name]
            if cap is not None and observed is not None and observed > cap:
                return name + "_limit"
        return None

    def _sample(self) -> None:
        self._shared_output()
        if (self._rss_missing_since is not None
                and time.monotonic() - self._rss_missing_since >= _RSS_MAX_GAP_SECONDS):
            raise _RSSUnavailable("RSS measurement gap exceeded its bounded interval")
        pids = [os.getpid(), *(j.pid for j in self._active.values() if j.pid is not None)]
        self._unlocated_prior_intents = 0
        for recovered in self._recovery.values():
            located = False
            for name in ("started.json", "spawned.json"):
                path = recovered["directory"] / name
                if path.exists():
                    old = self._read_json(path)
                    if old.get("token") != recovered["launch"]["token"] or type(old.get("pid")) is not int or old["pid"] <= 0:
                        raise ValueError("invalid unresolved process identity")
                    pids.append(old["pid"])
                    located = True
                    break
            if not located:
                self._unlocated_prior_intents += 1
        # Include spawn's resource tracker and auxiliary descendants. This is a
        # conservative parent-family charge, not only a sum of worker processes.
        started = time.monotonic()
        try:
            information = _process_info(pids, include_descendants=True)
        except (subprocess.SubprocessError, OSError, UnicodeError) as error:
            self._missing_rss_sample(started, type(error).__name__ + ": " + str(error)[:512])
            return
        live_pids = [os.getpid(), *(j.pid for j in self._active.values()
                                   if j.pid is not None and j.process.is_alive())]
        if any(pid not in information or information[pid]["rss_bytes"] <= 0 for pid in live_pids):
            self._missing_rss_sample(started, "RSS is missing for a live owned process or its parent")
            return
        if (self._rss_missing_since is not None
                and time.monotonic() - self._rss_missing_since >= _RSS_MAX_GAP_SECONDS):
            raise _RSSUnavailable("RSS measurement resumed after its bounded interval expired")
        self._rss_sample_at = started
        self._rss_missing_since = None
        self._rss_missing_count = 0
        self._rss_sample_error = None
        self._sampled_family_pids = sorted(information)
        self._parent_rss = information.get(os.getpid(), {}).get("rss_bytes")
        self._known_family_rss = sum(info["rss_bytes"] for info in information.values()) if self._parent_rss is not None else None
        self._shared_rss = self._known_family_rss if self._unlocated_prior_intents == 0 else None
        if self._known_family_rss is not None:
            self._shared_peak_rss = max(self._shared_peak_rss or 0, self._known_family_rss)
        for job in self._active.values():
            info = information.get(job.pid) if job.pid is not None else None
            if info is not None:
                job.os_identity = info
                job.peak_rss = max(job.peak_rss or 0, info["rss_bytes"])
        self._shared_output_peak = max(self._shared_output_peak, _directory_bytes(self.work_root)
                                      + max(0, _directory_bytes(self.store.root) - self._store_baseline))

    def _missing_rss_sample(self, started: float, detail: str) -> None:
        if self._rss_missing_since is None:
            self._rss_missing_since = (self._rss_sample_at
                if self._active and self._rss_sample_at is not None else started)
        self._rss_missing_count += 1
        self._rss_failures_total += 1
        self._rss_sample_error = detail
        # No stale measurement is represented as current and missing is never
        # charged as zero. Historical peaks remain available for accounting.
        self._parent_rss = self._known_family_rss = self._shared_rss = None
        self._sampled_family_pids = []
        if (self._rss_missing_count >= _RSS_MAX_MISSING_SAMPLES
                or time.monotonic() - self._rss_missing_since >= _RSS_MAX_GAP_SECONDS):
            raise _RSSUnavailable("RSS measurement unavailable: " + detail)

    def _rss_observation(self) -> dict[str, Any]:
        return {"last_sample_monotonic": self._rss_sample_at,
                "missing_since_monotonic": self._rss_missing_since,
                "consecutive_missing_samples": self._rss_missing_count,
                "missing_samples_total": self._rss_failures_total,
                "max_missing_samples": _RSS_MAX_MISSING_SAMPLES,
                "max_missing_seconds": _RSS_MAX_GAP_SECONDS,
                "last_error": self._rss_sample_error}

    def _failure_commit(self, writer: RunStore, job: _Job, result: dict[str, Any]) -> dict[str, Any]:
        """Fence a failure to this admitted attempt; never relabel a terminal."""
        if writer.store_id != job.handle.store_id:
            raise StateConflict("control failure cannot write to a different store")
        prior = next(a for a in writer.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)
        if prior["state"] in _TERMINAL:
            return prior
        current = writer.get_run(job.handle.run_id)
        if current["current_attempt_id"] != job.handle.attempt_id:
            raise StateConflict("control failure cannot replace a newer attempt")
        if job.token is None and prior["state"] == "running":
            # Claim may have committed before a subsequent launch/setup error.
            # Recover only our own exact claimed generation, never another owner.
            if prior["owner_id"] != self.owner_id or prior["generation"] != job.generation + 1:
                raise StateConflict("control failure cannot replace another owner")
            job.token = AttemptToken(self.store.store_id, job.handle.run_id, job.handle.attempt_id,
                                     self.owner_id, prior["generation"])
            job.generation = job.token.generation
            result["process"]["token"] = asdict(job.token)
        try:
            if job.token is None:
                writer.fail_queued(job.handle.run_id, expected_attempt_id=job.handle.attempt_id,
                                   expected_generation=job.generation, result=result)
            else:
                writer.complete(job.token, state="failed", result=result)
        except StateConflict:
            prior = next(a for a in writer.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)
            if prior["state"] not in _TERMINAL:
                raise
            return prior
        return next(a for a in writer.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)

    @staticmethod
    def _sync_audit(path: Path) -> None:
        for target, flags in ((path, os.O_RDONLY | os.O_NOFOLLOW),
                              (path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)):
            fd = os.open(target, flags)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _control_audit(self, attempt_id: str, directory: Path | None, name: str,
                       value: dict[str, Any], *, preserve_existing: bool = False) -> bool:
        """Publish independently, preserving partial evidence and reporting durability.

        Only recovery audits use an alternate staging name after a collision;
        ordinary producer publication retains its original exclusive-write rule.
        """
        try:
            assert directory is not None
            if directory.is_symlink() or directory.resolve() != directory:
                raise ValueError("unsafe control-failure audit directory")
            directory.mkdir(mode=0o700, exist_ok=True)
            path = directory / name
            if name == "receipt.json":
                value["control_audit_errors"] = list(self._control_audit_errors.get(attempt_id, []))
            if path.exists():
                saved = self._read_json(path)
                if (name == "exit.json" or preserve_existing and saved.get("terminal_committed") is True
                        or _object(saved) == _object(value)):
                    self._sync_audit(path)
                    return True
            try:
                _atomic_json(path, value)
            except FileExistsError as error:
                # Never truncate, unlink or overwrite interrupted-write evidence.
                # Refuse unsafe/oversized partials and isolate that error too.
                _regular_bytes(path.with_suffix(path.suffix + ".part"), TERMINAL_BYTES)
                self._control_error(attempt_id, name, error)
                if name == "receipt.json":
                    value["control_audit_errors"] = list(self._control_audit_errors[attempt_id])
                staging = directory / (name + ".recovery-" + uuid.uuid4().hex)
                _atomic_json(staging, value)
                os.replace(staging, path)
                self._sync_audit(path)
            return True
        except Exception as error:
            self._control_error(attempt_id, name, error)
            return False

    def _abort_control(self, error: BaseException) -> None:
        """Stop execution, then commit one fenced failure for every owned job.

        This path deliberately avoids progress parsing, RSS sampling, artifact
        publication and normal finalization: any of those may have caused the
        control failure. A terminal already committed remains historical truth.
        """
        if self._fatal is not None:
            return
        self._fatal = error
        exits: dict[str, dict[str, Any]] = {}
        # Signal all owned processes before any potentially slow store work.
        for job in list(self._active.values()):
            try:
                if job.process is not None and job.process.pid is not None and job.process.is_alive():
                    job.process.kill()
                    job.signals.append("kill")
            except Exception as cleanup:
                self._control_cleanup_errors.append("kill: " + type(cleanup).__name__ + ": " + str(cleanup)[:512])
        for job in self._known.values():
            previous = self._done.get(job.handle.attempt_id)
            old = previous.data if previous is not None else {}
            evidence = {"token": asdict(job.token) if job.token is not None else None,
                        "pid": job.pid, "pgid": job.pgid, "os_identity": job.os_identity,
                        "process_started": job.pid is not None,
                        "process_exit_observed": old.get("process_exit_observed", False),
                        "process_joined": old.get("process_joined", False),
                        "exitcode": old.get("exitcode"), "signals": list(job.signals),
                        "reason": "supervisor_control_failure"}
            if job.handle.attempt_id in self._active and not evidence["process_joined"]:
                try:
                    if job.process is not None and job.process.pid is not None:
                        job.pid = job.process.pid
                        evidence.update(process_started=True, pid=job.pid)
                        job.process.join(timeout=1.0)
                        joined = job.process.exitcode is not None and not job.process.is_alive()
                        evidence.update(process_exit_observed=joined, process_joined=joined,
                                        exitcode=job.process.exitcode)
                except Exception as cleanup:
                    self._control_cleanup_errors.append("join: " + type(cleanup).__name__ + ": " + str(cleanup)[:512])
            exits[job.handle.attempt_id] = evidence
        for job in self._known.values():
            evidence = exits[job.handle.attempt_id]
            previous = self._done.get(job.handle.attempt_id)
            if previous is not None and previous.data.get("terminal_committed"):
                if evidence["process_joined"] or not evidence["process_started"]:
                    self._active.pop(job.handle.attempt_id, None)
                continue  # Already completed receipts remain byte-for-byte historical.
            job.directory = job.directory or self.work_root / (self.store.store_id + "-" + job.handle.attempt_id)
            now = time.monotonic()
            resources = {"owned_wall_seconds": now - job.started if job.started is not None else 0.0,
                         "queue_seconds": (job.started if job.started is not None else now) - job.enqueued,
                         "child_peak_rss_bytes": job.peak_rss, "output_bytes": job.output_peak,
                         **job.counts, "rss_observation": self._rss_observation()}
            failure = {"reason": "supervisor_control_failure", "exception_type": type(error).__name__,
                       "detail": str(error)[:2048],
                       "control_reason": "rss_measurement_unavailable" if isinstance(error, _RSSUnavailable)
                           else "unexpected_control_exception"}
            result = {"status": "prepared_pending_dependencies", "worker_result": failure,
                      "process": evidence, "resources": resources,
                      "counter_observation_kind": "worker_reported_or_null_unobserved"}
            intent = {"schema": "solarlab.supervisor-control-failure.v1", "handle": asdict(job.handle),
                      "generation": job.generation, "owner_id": self.owner_id,
                      "token": evidence["token"], "result": result}
            # If storage is unavailable, retain the failure intent independently
            # for explicit recovery. File failures never gate the DB commit.
            retained = []
            if self._control_audit(job.handle.attempt_id, job.directory, "control-failure.json", intent):
                retained.append("control-failure.json")
            prior: dict[str, Any] | None = None
            commit_error: str | None = None
            try:
                prior = self._failure_commit(self.store, job, result)
            except StateConflict as conflict:
                commit_error = str(conflict)
            except Exception as failed_writer:
                # A broken writer connection must not strand a readable store.
                # Use the same fenced API and no artifact references on one
                # fresh, bounded local writer; never bypass ownership with SQL.
                try:
                    with RunStore(self.store.root, timeout=0.5) as repair:
                        prior = self._failure_commit(repair, job, result)
                except Exception as repair_error:
                    commit_error = (type(failed_writer).__name__ + ": " + str(failed_writer)[:512]
                                    + "; repair: " + type(repair_error).__name__ + ": " + str(repair_error)[:512])
                    self._control_cleanup_errors.append("terminal commit: " + commit_error)
            state = prior["state"] if prior is not None else "unknown"
            committed = state in _TERMINAL
            if not committed:
                self._control_recovery[job.handle.attempt_id] = {"directory": job.directory,
                    "intent": intent, "terminal_committed": False}
            saved_result = prior["result"] if prior is not None and prior["result"] is not None else result
            terminal = prior.get("terminal") if prior is not None else None
            receipt = {"status": "prepared_pending_dependencies", "run_id": job.handle.run_id,
                       "attempt_id": job.handle.attempt_id, "token": asdict(job.token) if job.token is not None else None,
                       "store_state": state, "proposed_state": "failed", "terminal_committed": committed,
                       "commit_error": commit_error, **evidence, "resources": resources,
                       "worker_result": saved_result.get("worker_result", saved_result),
                       "terminal_artifacts": [a["artifact_id"] for a in terminal["artifacts"]] if terminal else [],
                       "publication_error": None, "control_failure": failure, "control_intent": intent}
            if evidence["process_exit_observed"]:
                if self._control_audit(job.handle.attempt_id, job.directory, "exit.json", evidence):
                    retained.append("exit.json")
            if self._control_audit(job.handle.attempt_id, job.directory, "receipt.json", receipt,
                                   preserve_existing=previous is None):
                retained.append("receipt.json")
            self._control_retention[job.handle.attempt_id] = retained
            receipt["control_recovery_retention"] = {"confirmed_records": retained,
                "memory_only": not retained, "administrative_fencing_required": not committed and not retained}
            self._done[job.handle.attempt_id] = ExecutionReceipt(_object(receipt))
            if not evidence["process_started"] or evidence["process_joined"]:
                if job.handle.attempt_id in self._active and job.process is not None:
                    try:
                        job.process.close()
                    except Exception as cleanup:
                        self._control_cleanup_errors.append("close: " + type(cleanup).__name__ + ": " + str(cleanup)[:512])
                self._active.pop(job.handle.attempt_id, None)
        self._pending.clear()
        self._condition.notify_all()

    def _shared_output(self) -> int:
        observed = _directory_bytes(self.work_root) + max(0, _directory_bytes(self.store.root) - self._store_baseline)
        self._shared_output_peak = max(self._shared_output_peak, observed)
        return observed

    def _completion_allowance(self, job: _Job, result: dict[str, Any]) -> int:
        # RunStore writes result/terminal/event JSON plus SQLite pages and an
        # independent audit receipt. Reserve conservatively; this is not an OS
        # quota or a claim that concurrent filesystem writes are transactional.
        others = sum(other.limits.output_bytes + 131072 for other in self._active.values() if other is not job)
        return 131072 + 8 * (len(_object(result).encode()) + 4096) + others

    def _output_guard(self, extra_bytes: int, allowance: int) -> dict[str, Any]:
        current = self._shared_output()
        return {"observed_bytes": current, "charged_peak_bytes": self._shared_output_peak,
                "additional_payload_bytes": extra_bytes, "reserved_completion_bytes": allowance,
                "limit_bytes": self._output_cap,
                "would_exceed": self._shared_output_peak + extra_bytes + allowance > self._output_cap}

    @staticmethod
    def _checked_artifacts(job: _Job, items: Any) -> list[tuple[str, ArtifactPayload]]:
        """Validate all untrusted terminal entries before touching the store."""
        if type(items) is not list or len(items) > 16:
            raise ValueError("invalid artifact manifest collection")
        assert job.directory is not None
        checked = []
        names: set[str] = set()
        total = 0
        for index, item in enumerate(items):
            if type(item) is not dict or set(item) != {"name", "file", "sha256", "bytes", "format", "metadata_json"}:
                raise ValueError("malformed artifact manifest entry")
            name = _name(item["name"])
            if name in names or item["file"] != f"artifact-{index}.bin":
                raise ValueError("duplicate artifact or invalid generated filename")
            names.add(name)
            _sha(item["sha256"])
            if type(item["bytes"]) is not int or item["bytes"] < 0 or type(item["format"]) is not str or type(item["metadata_json"]) is not str:
                raise ValueError("invalid artifact size/format/metadata type")
            total += item["bytes"]
            if total > job.limits.output_bytes:
                raise ValueError("artifact manifest exceeds the output budget")
            data = _regular_bytes(job.directory / item["file"], job.limits.output_bytes)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("artifact payload digest/size mismatch")
            checked.append((name, ArtifactPayload(data, item["format"], item["metadata_json"])))
        return checked

    def _finish(self, job: _Job) -> None:
        assert job.token is not None and job.directory is not None and job.started is not None
        job.process.join(timeout=0)
        exitcode = job.process.exitcode
        if exitcode is None or job.process.is_alive():
            raise RuntimeError("a slot cannot be released without positive process exit/join")
        self._progress(job)
        exit_evidence = {"token": asdict(job.token), "pid": job.pid, "pgid": job.pgid, "os_identity": job.os_identity,
                         "exitcode": exitcode, "process_joined": True, "process_exit_observed": True,
                         "signals": list(job.signals)}
        _atomic_json(job.directory / "exit.json", exit_evidence)
        manifest: dict[str, Any] = {}
        payloads: list[tuple[str, ArtifactPayload]] = []
        publishable = False
        state, result = "failed", {"reason": "child_crash", "exitcode": exitcode}
        try:
            if (job.directory / "terminal.json").exists():
                manifest = self._read_json(job.directory / "terminal.json")
                if (set(manifest) != {"token", "worker_source_sha256", "state", "result", "artifacts", "counters", "child_peak_rss_bytes"}
                        or type(manifest["token"]) is not dict or AttemptToken(**manifest["token"]) != job.token
                        or manifest["worker_source_sha256"] != job.binding.source_sha256
                        or type(manifest["state"]) is not str or manifest["state"] not in {"succeeded", "failed", "cancelled"}):
                    raise ValueError("terminal identity/schema mismatch")
                _object(manifest["result"])
                if type(manifest["child_peak_rss_bytes"]) is not int or manifest["child_peak_rss_bytes"] <= 0:
                    raise ValueError("invalid child RSS measurement")
                job.peak_rss = max(job.peak_rss or 0, manifest["child_peak_rss_bytes"])
                if type(manifest["counters"]) is not dict or set(manifest["counters"]) != set(job.counts):
                    raise ValueError("incomplete counter measurement fields")
                for name, count in manifest["counters"].items():
                    if count is not None and (type(count) is not int or count < 0 or count >= 2**63):
                        raise ValueError("invalid cumulative terminal counter")
                    old = job.counts[name]
                    if old is not None and (count is None or count < old):
                        raise ValueError("terminal counters lost a prior observation")
                    job.counts[name] = count
                payloads = self._checked_artifacts(job, manifest["artifacts"])
                if exitcode == 0:
                    state, result = manifest["state"], manifest["result"]
                    # Exceptions/cooperative stops carry no payloads. Only a
                    # validated clean return may publish a numerical failure.
                    publishable = state == "succeeded" or state == "failed" and bool(payloads)
            elif exitcode == 0:
                result = {"reason": "missing_terminal"}
        except (ValueError, TypeError, KeyError, OSError) as error:
            state, result = "failed", {"reason": "invalid_terminal", "detail": str(error)[:1024]}
            payloads = []
            publishable = False
        reason = self._resource_reason(job, time.monotonic())
        if reason is not None:
            state, result = "failed", {"reason": reason}
            publishable = False
        if publishable:
            missing = [name for name, cap in (("accepted_steps", job.limits.accepted_steps), ("calls", job.limits.calls))
                       if cap is not None and job.counts[name] is None]
            if missing or job.peak_rss is None:
                state, result = "failed", {"reason": "required_measurement_unobserved", "counters": missing}
                publishable = False
        if job.stop_reason is not None:
            state, result = job.stop_state, {"reason": job.stop_reason, "discarded_terminal_state": manifest.get("state")}
            publishable = False
            if job.setup_error is not None:
                result["setup_error"] = job.setup_error
        references: list[str] = []
        published_bytes = 0
        publication_error: str | None = None
        if publishable:
            try:
                for name, payload in payloads:
                    if _directory_bytes(job.directory) + published_bytes + len(payload.data) > job.limits.output_bytes:
                        raise ValueError("output limit before parent publication")
                    guard = self._output_guard(len(payload.data), self._completion_allowance(job, result))
                    if guard["would_exceed"]:
                        state, result, references = "failed", {"reason": "shared_output_publication_reservation", "budget": guard}, []
                        publishable = False
                        break
                    published = self.store.publish(job.token, name, payload)
                    references.append(published["artifact_id"])
                    published_bytes += len(payload.data)
                    guard = self._output_guard(0, self._completion_allowance(job, result))
                    if guard["would_exceed"]:
                        state, result, references = "failed", {"reason": "shared_output_after_publication", "budget": guard}, []
                        publishable = False
                        break
            except (ValueError, OSError, StateConflict) as error:
                publication_error = str(error)[:1024]
                state, result, references = "failed", {"reason": "artifact_publication", "detail": publication_error}, []
                publishable = False
        if time.monotonic() - job.started > job.limits.wall_seconds and publishable:
            state, result, references = "failed", {"reason": "wall_limit_during_publication"}, []
            publishable = False
        retained_output = _directory_bytes(job.directory) + published_bytes
        job.output_peak = max(job.output_peak, retained_output)
        resources: dict[str, Any] = {"owned_wall_seconds": time.monotonic() - job.started, "queue_seconds": job.started - job.enqueued,
                     "child_peak_rss_bytes": job.peak_rss, "parent_rss_last_observed_bytes": self._parent_rss,
                     "output_bytes": job.output_peak, "retained_output_bytes_before_terminal_commit": retained_output,
                     "wall_measurement_phase": "before_terminal_commit", **job.counts}
        resources["observed_limit_violations"] = [name for name, observed, cap in (
            ("wall_seconds", resources["owned_wall_seconds"], job.limits.wall_seconds),
            ("rss_bytes", job.peak_rss, job.limits.rss_bytes), ("output_bytes", job.output_peak, job.limits.output_bytes),
            ("accepted_steps", job.counts["accepted_steps"], job.limits.accepted_steps),
            ("calls", job.counts["calls"], job.limits.calls)) if observed is not None and cap is not None and observed > cap]
        final_result = {"status": "prepared_pending_dependencies", "worker_result": result,
                        "process": exit_evidence, "resources": resources,
                        "counter_observation_kind": "worker_reported_or_null_unobserved"}
        guard = self._output_guard(0, self._completion_allowance(job, final_result))
        resources["shared_output_before_terminal_commit"] = guard
        if publishable and guard["would_exceed"]:
            state, result, references = "failed", {"reason": "shared_output_finalization_reservation", "budget": guard}, []
            publishable = False
            final_result["worker_result"] = result
        committed, commit_error = False, None
        storage_error: sqlite3.Error | None = None
        try:
            self.store.complete(job.token, state=state, result=final_result, artifacts=references)
            committed = True
        except StateConflict as error:
            commit_error = str(error)  # Late output cannot overwrite a terminal or a new generation.
            result = {"reason": "late_result_rejected", "discarded_terminal_state": manifest.get("state")}
        except sqlite3.Error as error:
            storage_error = error
            commit_error = type(error).__name__ + ": " + str(error)[:1024]
        prior = next(a for a in self.store.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)
        resources = {**resources, "owned_wall_seconds": time.monotonic() - job.started,
                     "wall_measurement_phase": "after_terminal_commit_before_audit_fsync",
                     "shared_output_after_terminal_commit_bytes": self._shared_output()}
        receipt = {"status": "prepared_pending_dependencies", "run_id": job.handle.run_id, "attempt_id": job.handle.attempt_id,
                   "token": asdict(job.token), "store_state": prior["state"], "proposed_state": state,
                   "terminal_committed": committed, "commit_error": commit_error, "process_started": True,
                   "process_exit_observed": True, "process_joined": True, "exitcode": exitcode, "pid": job.pid, "pgid": job.pgid,
                   "signals": list(job.signals), "resources": resources, "worker_result": result,
                   "terminal_artifacts": references if committed else [], "publication_error": publication_error}
        _atomic_json(job.directory / "receipt.json", receipt)
        after_audit = self._shared_output()
        finalization = {"token": asdict(job.token), "shared_output_after_audit_bytes": after_audit,
                        "shared_output_peak_bytes": self._shared_output_peak, "shared_limit_bytes": self._output_cap,
                        "owned_wall_after_audit_seconds": time.monotonic() - job.started,
                        "finalization_overrun": self._shared_output_peak > self._output_cap,
                        "measurement_phase": "after_receipt_fsync_before_finalization_record",
                        "store_terminal_is_historical": True}
        _atomic_json(job.directory / "finalization.json", finalization)
        finalization_record_bytes = (job.directory / "finalization.json").stat().st_size
        finalization["finalization_record_bytes"] = finalization_record_bytes
        finalization["shared_output_after_finalization_record_bytes"] = self._shared_output()
        finalization["finalization_overrun"] = self._shared_output_peak > self._output_cap
        if finalization["finalization_overrun"] and after_audit <= self._output_cap:
            _atomic_json(job.directory / "finalization_overrun.json", {"token": asdict(job.token),
                "observed_bytes": self._shared_output_peak, "limit_bytes": self._output_cap,
                "measurement_phase": "after_finalization_record_write", "store_terminal_is_historical": True})
            self._shared_output()
        self._done[job.handle.attempt_id] = ExecutionReceipt(_object({**receipt, "finalization": finalization}))
        job.process.close()
        del self._active[job.handle.attempt_id]
        self._condition.notify_all()
        if storage_error is not None:
            raise RuntimeError("parent RunStore terminal commit failed; joined-process result remains in local records") from storage_error

    def _loop(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._fatal is not None:
                        break
                    if self._closing:
                        self._pending.clear()  # SQLite queued rows remain available for explicit resubmission.
                        for job in list(self._active.values()):
                            self._request_stop(job, "supervisor_shutdown", "cancelled")
                    if self._active or self._pending:
                        self._sample()
                    now = time.monotonic()
                    for job in list(self._active.values()):
                        self._progress(job)
                        assert job.token is not None
                        current = self.store.get_run(job.handle.run_id)
                        if (current["current_attempt_id"] != job.handle.attempt_id or current["generation"] != job.generation
                                or current["state"] != "running"):
                            self._request_stop(job, "authority_revoked", "failed")
                        if self._known_family_rss is not None and self._known_family_rss > self._rss_cap:
                            self._request_stop(job, "shared_rss_limit", "failed")
                        if self._shared_output_peak > self._output_cap:
                            self._request_stop(job, "shared_output_limit", "failed")
                        reason = self._resource_reason(job, now)
                        if reason is not None:
                            self._request_stop(job, reason, "failed")
                        if not job.process.is_alive():
                            if self._rss_missing_since is not None:
                                continue
                            self._finish(job)
                            continue
                        if job.stop_at is not None and now - job.stop_at >= self._soft and job.terminate_at is None:
                            job.process.terminate()  # Only this still-owned, unreaped multiprocessing child.
                            job.terminate_at = now
                            job.signals.append("terminate")
                        elif job.terminate_at is not None and now - job.terminate_at >= self._terminate and job.kill_at is None:
                            job.process.kill()
                            job.kill_at = now
                            job.signals.append("kill")
                    if not self._active and self._pending and self._rss_missing_since is None:
                        unavailable = ("shared_rss_limit" if self._known_family_rss is not None
                            and self._known_family_rss > self._rss_cap else
                            "shared_output_limit" if self._shared_output_peak > self._output_cap else None)
                        if unavailable is not None:
                            while self._pending:
                                self._unstarted(self._pending.popleft(), _AdmissionLimit(unavailable))
                    while (not self._closing and self._pending and self._rss_missing_since is None
                           and self._shared_output_peak <= self._output_cap
                           and len(self._active) + len(self._recovery) < self.max_active):
                        queued = self._pending.popleft()
                        try:
                            self._start(queued)
                        except (StateConflict, OSError) as error:
                            self._unstarted(queued, error)
                    if self._closing and not self._active:
                        break
                    self._condition.wait(self._poll)
        except BaseException as error:
            with self._condition:
                self._abort_control(error)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def _reconcile_control_failures(self) -> list[str]:
        """Replay only a retained failure intent, never the worker request."""
        invalidated = []
        for attempt_id, entry in list(self._control_recovery.items()):
            try:
                directory = entry["directory"]
                if entry["intent"] is None:
                    receipt = self._discover_control_failure(directory)
                    if receipt is not None:
                        self._done[attempt_id] = ExecutionReceipt(_object(receipt))
                        del self._control_recovery[attempt_id]
                        continue
                    entry = self._control_recovery[attempt_id]
                    if entry["intent"] is None:
                        raise StateConflict("incomplete control evidence; administrative fencing required")
                intent = entry["intent"]
                handle, token = self._control_binding(directory, intent)
                prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == attempt_id)
                if prior["state"] not in _TERMINAL:
                    if token is None and prior["state"] == "running":
                        if (intent["owner_id"] is None or prior["owner_id"] != intent["owner_id"]
                                or prior["generation"] != intent["generation"] + 1):
                            raise StateConflict("control-failure recovery cannot replace another owner")
                        token = AttemptToken(handle.store_id, handle.run_id, attempt_id,
                                             intent["owner_id"], prior["generation"])
                    if token is None:
                        self.store.fail_queued(handle.run_id, expected_attempt_id=attempt_id,
                            expected_generation=intent["generation"], result=intent["result"])
                    else:
                        self.store.complete(token, state="failed", result=intent["result"])
                    invalidated.append(attempt_id)
                    prior = next(a for a in self.store.attempts(handle.run_id) if a["attempt_id"] == attempt_id)
                entry["terminal_committed"] = True
                receipt = self._control_receipt(intent, prior, token)
                # Store truth is current even if both audits below are unavailable.
                self._done[attempt_id] = ExecutionReceipt(_object(receipt))
                retained = self._control_retention.setdefault(attempt_id, [])
                if self._control_audit(attempt_id, directory, "control-failure.json", intent):
                    if "control-failure.json" not in retained:
                        retained.append("control-failure.json")
                published = self._control_audit(attempt_id, directory, "receipt.json", receipt)
                self._done[attempt_id] = ExecutionReceipt(_object(receipt))
                if published:
                    if "receipt.json" not in retained:
                        retained.append("receipt.json")
                    del self._control_recovery[attempt_id]
            except Exception as error:
                self._control_error(attempt_id, "reconcile control failure", error)
        return invalidated

    def reconcile_interrupted(self) -> dict[str, Any]:
        """Explicitly invalidate affected tokens and observe, never kill, old PIDs."""
        with self._condition:
            released, unresolved, invalidated = [], [], self._reconcile_control_failures()
            for attempt_id, item in list(self._recovery.items()):
                control = self._control_recovery.get(attempt_id)
                if control is not None and not control["terminal_committed"]:
                    unresolved.append({"attempt_id": attempt_id, "process_exit_observed": False,
                        "method": "control_failure_requires_reconciliation"})
                    continue  # Never relabel an unresolved control failure as interrupted.
                try:
                    evidence, changed = self._reconcile_process(item)
                    if changed:
                        invalidated.append(attempt_id)
                    if evidence["process_exit_observed"]:
                        if self._control_audit(attempt_id, item["directory"], "reconciled.json",
                                {"token": item["launch"]["token"], "evidence": evidence}):
                            released.append(attempt_id)
                            del self._recovery[attempt_id]
                            if attempt_id not in self._done:
                                self._unknown_history.add(attempt_id)
                            continue
                        evidence = {**evidence, "method": "exit_observed_audit_pending"}
                    unresolved.append({"attempt_id": attempt_id, **evidence})
                except Exception as error:
                    self._control_error(attempt_id, "reconcile process", error)
                    unresolved.append({"attempt_id": attempt_id, "process_exit_observed": False,
                        "method": "process_reconciliation_failed", "error": type(error).__name__ + ": " + str(error)[:512]})
            self._condition.notify_all()
            return {"invalidated_owned_attempts": invalidated, "released_owned_reservations": released,
                    "unresolved_owned_slots": unresolved, "unowned_processes_signalled": False,
                    "uncommitted_control_failures": [key for key, entry in self._control_recovery.items()
                                                    if not entry["terminal_committed"]],
                    "pending_control_audits": [key for key, entry in self._control_recovery.items()
                                               if entry["terminal_committed"]],
                    "control_audit_errors": {key: list(value) for key, value in self._control_audit_errors.items()},
                    "queued_policy": "explicit_matching_resubmission; no automatic scientific rerun"}

    def _reconcile_process(self, item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Observe one old process; publication errors cannot bypass its exit proof."""
        directory, launch = item["directory"], item["launch"]
        token = AttemptToken(**launch["token"])
        evidence: dict[str, Any] = {"process_exit_observed": False, "method": "unknown_intent", "pid": None}
        if (directory / "exit.json").exists():
            old = self._read_json(directory / "exit.json")
            if old.get("token") != asdict(token) or old.get("process_joined") is not True or type(old.get("exitcode")) is not int:
                raise ValueError("invalid prior positive exit receipt")
            evidence = {"process_exit_observed": True, "method": "prior_owned_join", "pid": old["pid"], "exitcode": old["exitcode"]}
        else:
            ready: dict[str, Any] = item.get("process", {})
            for name in ("started.json", "spawned.json"):
                if (directory / name).exists():
                    ready = self._read_json(directory / name)
                    break
            if ready:
                pid = ready.get("pid")
                if ready.get("token") != asdict(token) or type(pid) is not int or pid <= 0:
                    raise ValueError("invalid prior process identity")
                evidence.update(pid=pid, method="unowned_pid_still_uncertain")
                try:
                    os.kill(pid, 0)  # Presence check only, never a cancellation signal.
                except ProcessLookupError:
                    evidence.update(process_exit_observed=True, method="observed_pid_absence")
                except PermissionError:
                    pass
                else:
                    try:
                        current = _process_info([pid]).get(pid)
                    except (subprocess.SubprocessError, OSError, UnicodeError) as error:
                        current = None
                        evidence.update(method="unowned_pid_measurement_unavailable",
                            measurement_error=type(error).__name__ + ": " + str(error)[:512])
                    previous = ready.get("os_identity")
                    if current is not None and previous is not None and current["start_lstart"] != previous["start_lstart"]:
                        evidence.update(process_exit_observed=True, method="observed_pid_reuse", old_identity=previous, current_identity=current)
        current_run = self.store.get_run(token.run_id)
        if (current_run["current_attempt_id"] == token.attempt_id and current_run["generation"] == token.generation
                and current_run["state"] == "running"):
            self.store.complete(token, state="failed", result={"reason": "interrupted", "ownership_invalidated": True,
                "process_exit_observed": evidence["process_exit_observed"], "process_evidence": evidence})
            return evidence, True
        return evidence, False

    def accounting(self) -> dict[str, Any]:
        with self._condition:
            self._shared_output_peak = max(self._shared_output_peak, _directory_bytes(self.work_root)
                                          + max(0, _directory_bytes(self.store.root) - self._store_baseline))
            finished = [r.data["resources"] for r in self._done.values()]
            active = [{**j.counts, "owned_wall_seconds": time.monotonic() - j.started if j.started is not None else 0.0}
                      for j in self._active.values()]
            rows = finished + active
            counts = {name: sum(r[name] for r in rows) if rows and not self._unknown_history and not self._recovery
                      and all(r[name] is not None for r in rows) else None
                      for name in ("accepted_steps", "calls")}
            return {"status": "prepared_pending_dependencies", "active": len(self._active), "pending": len(self._pending),
                    "unresolved_owned_slots": len(self._recovery), "completed_receipts": len(finished),
                    "uncommitted_control_failures": sum(not e["terminal_committed"] for e in self._control_recovery.values()),
                    "pending_control_audits": sum(e["terminal_committed"] for e in self._control_recovery.values()),
                    "control_audit_errors": {key: list(value) for key, value in self._control_audit_errors.items()},
                    "control_recovery_records": {key: list(value) for key, value in self._control_retention.items()},
                    "administrative_fencing_required": [key for key, entry in self._control_recovery.items()
                        if not entry["terminal_committed"] and (entry["intent"] is None or not self._control_retention.get(key))],
                    "known_owned_wall_seconds": sum(r["owned_wall_seconds"] for r in rows if r["owned_wall_seconds"] is not None),
                    "counter_totals": counts, "unknown_historical_attempts": len(self._recovery) + len(self._unknown_history),
                    "shared_rss_last_observed_bytes": self._shared_rss, "shared_rss_peak_bytes": self._shared_peak_rss,
                    "known_family_rss_last_observed_bytes": self._known_family_rss,
                    "rss_scope": "conservative_parent_process_family_and_located_prior_intents",
                    "sampled_family_pids": list(self._sampled_family_pids),
                    "rss_observation": self._rss_observation(),
                    "control_cleanup_errors": list(self._control_cleanup_errors),
                    "unlocated_prior_intents": self._unlocated_prior_intents,
                    "retained_output_peak_bytes": self._shared_output_peak, "queued_recovery_is_explicit": True}

    def close(self, *, timeout: float = 3.0) -> None:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("close requires a positive finite join bound")
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive() or any(j.process is not None and j.process.pid is not None and j.process.is_alive()
                                         for j in self._active.values()):
            raise TimeoutError("owned process/monitor exit is not yet observed; capacity is not released")
        # A fatal cleanup may have timed out before a killed child finally
        # exited. Close its native handle only after this later positive join.
        if self._fatal is not None:
            for attempt_id, job in list(self._active.items()):
                if job.process is not None:
                    if job.process.pid is not None:
                        job.process.join(timeout=0)
                        self._control_audit(attempt_id, job.directory, "exit.json", {"token": asdict(job.token) if job.token else None,
                            "pid": job.pid, "exitcode": job.process.exitcode, "process_joined": True,
                            "process_exit_observed": True, "reason": "supervisor_control_failure"})
                    job.process.close()
                del self._active[attempt_id]
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1
        if self._fatal is not None:
            raise RuntimeError("supervisor failed; preserved process intents require reconciliation") from self._fatal

    def __enter__(self) -> ProcessSupervisor:
        return self

    def __exit__(self, *error: Any) -> None:
        self.close()


class ScanRunner:
    """Bounded parent coordination, persisted declarations and explicit retries.

    RunStore remains the sole attempt/terminal authority. A scan's status is a
    derived view of those rows, never a second mutable terminal. Every point
    starts a fresh worker with its own manifest; state and numerical bindings
    are not carried from a preceding point. Scientific qualification belongs
    to the worker's retained result and is never inferred from process success.
    """

    def __init__(self, supervisor: ProcessSupervisor, spec: ScanSpec, executions: tuple[ManifestBinding, ...], *,
                 max_in_flight: int) -> None:
        if multiprocessing.current_process().daemon:
            raise ValueError("workers cannot create nested supervisors or scans")
        if type(supervisor) is not ProcessSupervisor or type(spec) is not ScanSpec:
            raise ValueError("scans must share an existing ProcessSupervisor")
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= supervisor.max_active + supervisor.max_pending:
            raise ValueError("scan in-flight bound must fit the shared finite capacity")
        if len(executions) != len(spec.points):
            raise ValueError("every point requires its independently prepared manifest binding")
        self.supervisor, self.spec, self.executions = supervisor, spec, tuple(executions)
        self.max_in_flight = max_in_flight
        self._submitted: set[str] = set()
        self._reused: set[str] = set()
        self._metadata = []
        self._path = supervisor.work_root / ("scan-" + supervisor.store.store_id + "-" + spec.scan_id + ".json")
        self._cancel_path = self._path.with_suffix(".cancel.json")
        self._sha256 = hashlib.sha256(spec.to_bytes()).hexdigest()
        with supervisor._condition:
            supervisor._check_open()
            # Check the whole declaration before persisting or starting any point.
            for point, binding in zip(spec.points, self.executions):
                point.spec.validate(binding)
                self._metadata.append(supervisor._submission(point.spec.worker_id, point.spec.request, point.spec.limits, binding))
            for index in range(len(spec.points)):
                self._existing(index)
            known = {p["point_id"]: p for p in supervisor.store.sweep_points(spec.scan_id)}
            if set(known) - {p.point_id for p in spec.points}:
                raise StateConflict("stored scan has points outside this declaration")
            for point in spec.points:
                previous = known.get(point.point_id)
                if previous is not None and (previous["run_id"] != point.run_id
                        or previous["opaque_execution_identity"] != point.spec.request.execution_identity.sha256
                        or _object(previous["coordinates"]) != point.coordinates_json):
                    raise StateConflict("stored scan association differs from this declaration")
            self._persist(self._path, _parse_document(spec.to_bytes().decode()))
            self._cancel_reason()  # Refuse malformed cancellation intent before scheduling.

    @property
    def declaration_path(self) -> Path:
        return self._path

    def _persist(self, path: Path, value: dict[str, Any]) -> None:
        raw = _object(value).encode()
        if path.exists() or path.is_symlink():
            if _regular_bytes(path, SCAN_BYTES) != raw:
                raise StateConflict("scan declaration or cancellation intent already has different content")
            return
        guard = self.supervisor._output_guard(len(raw), 65536)
        if guard["would_exceed"]:
            raise ValueError("shared output capacity cannot retain scan metadata")
        _atomic_json(path, value)

    def _cancel_reason(self) -> str | None:
        if not self._cancel_path.exists() and not self._cancel_path.is_symlink():
            return None
        value = self.supervisor._read_json(self._cancel_path)
        if (set(value) != {"schema", "scan_sha256", "reason"} or value["schema"] != "solarlab.scan-cancel.v1"
                or value["scan_sha256"] != self._sha256 or type(value["reason"]) is not str or not value["reason"].strip()):
            raise ValueError("invalid persisted scan cancellation intent")
        return value["reason"]

    def _existing(self, index: int) -> dict[str, Any] | None:
        point = self.spec.points[index]
        try:
            row = self.supervisor.store.get_run(point.run_id)
        except KeyError:
            return None
        identity, metadata = self._metadata[index]
        expected = {"identity_kind": "opaque_upstream", "identity": identity, "input_metadata": metadata}
        if _object(row["request"]) != _object(expected):
            raise StateConflict("scan point differs from the complete stored execution request, manifest, worker or budgets")
        return row

    def _owned(self, run_id: str) -> bool:
        pool = self.supervisor
        return (any(job.handle.run_id == run_id for job in pool._active.values())
                or any(item["launch"]["token"]["run_id"] == run_id for item in pool._recovery.values()))

    def _bind(self, point: ScanPoint) -> None:
        self.supervisor.store.bind_sweep_point(self.spec.scan_id, point.point_id, run_id=point.run_id,
                                               coordinates=_parse_document(point.coordinates_json))

    def _verify_success(self, index: int, row: dict[str, Any]) -> None:
        """Reuse requires actual committed artifacts and a joined local receipt."""
        point, binding = self.spec.points[index], self.executions[index]
        pool = self.supervisor
        receipt = pool._done.get(row["current_attempt_id"])
        attempt = pool.store.attempts(point.run_id)[-1]
        if receipt is None:
            raise StateConflict("completed point has no retained owned-process receipt")
        evidence = receipt.data
        token = {"store_id": pool.store.store_id, "run_id": point.run_id,
                 "attempt_id": attempt["attempt_id"], "owner_id": attempt["owner_id"], "generation": attempt["generation"]}
        # Restart retains a post-write overrun separately from the earlier audit.
        # Its presence fences reuse even when that earlier audit still says False.
        if (evidence.get("token") != token or evidence.get("store_state") != "succeeded"
                or evidence.get("terminal_committed") is not True or evidence.get("process_exit_observed") is not True
                or evidence.get("process_joined") is not True or evidence.get("exitcode") != 0
                or evidence.get("finalization", {}).get("finalization_overrun") is True
                or "finalization_overrun" in evidence):
            raise StateConflict("completed point lacks a successful closed resource receipt")
        audit = evidence.get("finalization", {})
        resources = evidence.get("resources", {})
        if audit.get("token") != token or audit.get("finalization_overrun") is not False:
            raise StateConflict("completed point lacks its complete final resource audit")
        for name, value, limit in (
            ("wall_seconds", audit.get("owned_wall_after_audit_seconds"), point.spec.limits.wall_seconds),
            ("rss_bytes", resources.get("child_peak_rss_bytes"), point.spec.limits.rss_bytes),
            ("output_bytes", resources.get("output_bytes"), point.spec.limits.output_bytes),
            ("accepted_steps", resources.get("accepted_steps"), point.spec.limits.accepted_steps),
            ("calls", resources.get("calls"), point.spec.limits.calls),
        ):
            if limit is not None and (type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= limit):
                raise StateConflict("completed point lacks an in-budget measurement: " + name)
        result = attempt["result"]
        if (result["process"]["token"] != token or result["process"]["process_joined"] is not True
                or result["process"]["process_exit_observed"] is not True or result["process"]["exitcode"] != 0
                or _object(result["worker_result"]) != _object(evidence["worker_result"])):
            raise StateConflict("completed point result differs from its owned receipt")
        references = attempt["terminal"]["artifacts"]
        if [item["artifact_id"] for item in references] != evidence.get("terminal_artifacts"):
            raise StateConflict("completed point terminal differs from its owned receipt")
        manifest_count = 0
        for item in references:
            record = pool.store.artifact(item["artifact_id"])
            content = pool.store.read_artifact(item["artifact_id"])
            if (record["run_id"] != point.run_id or record["attempt_id"] != attempt["attempt_id"]
                    or record["generation"] != attempt["generation"] or record["manifest_sha256"] != item["manifest_sha256"]):
                raise StateConflict("completed artifact belongs to another attempt or generation")
            if record["name"] == "execution_manifest":
                if content != binding.manifest.canonical_json:
                    raise StateConflict("completed point has a different complete execution manifest")
                manifest_count += 1
        if manifest_count != 1:
            raise StateConflict("completed point requires exactly one verified execution manifest artifact")

    def advance(self) -> dict[str, Any]:
        """Admit only available points, or finish an explicit durable cancellation."""
        pool = self.supervisor
        with pool._condition:
            pool._check_open()
            reason = self._cancel_reason()
            in_flight = sum(job.handle.run_id in {p.run_id for p in self.spec.points}
                            for job in (*pool._active.values(), *pool._pending))
            for index, point in enumerate(self.spec.points):
                row = self._existing(index)
                if reason is not None:
                    if row is None:
                        identity, metadata = self._metadata[index]
                        row = pool.store.enqueue(point.run_id, opaque_identity=identity, input_metadata=metadata)
                    self._bind(point)
                    if row["state"] not in _TERMINAL:
                        handle = JobHandle(pool.store.store_id, point.run_id, row["current_attempt_id"])
                        if handle.attempt_id in pool._known:
                            pool.cancel(handle, reason=reason)
                        else:
                            # A declared/unstarted point needs no process slot.
                            # Recovered running ownership is fenced, never signalled here.
                            pool.store.cancel(point.run_id, expected_attempt_id=handle.attempt_id,
                                              expected_generation=row["generation"], reason=reason)
                    continue
                if row is not None and row["state"] in _TERMINAL:
                    self._bind(point)
                    if row["state"] == "succeeded" and not self._owned(point.run_id):
                        self._verify_success(index, row)
                        if row["current_attempt_id"] not in self._submitted:
                            self._reused.add(row["current_attempt_id"])
                    continue
                if row is not None and row["current_attempt_id"] in pool._known:
                    continue
                if row is not None and row["state"] == "running":
                    raise StateConflict("interrupted point needs explicit owned-process reconciliation before resume")
                if in_flight >= self.max_in_flight:
                    continue
                try:
                    handle = pool.submit(point.run_id, point.spec.worker_id, point.spec.request,
                                         point.spec.limits, execution=self.executions[index])
                except QueueFull:
                    break  # Server/other scans own the remaining shared capacity.
                self._submitted.add(handle.attempt_id)
                try:
                    self._bind(point)
                except (StateConflict, sqlite3.Error, OSError):
                    # The monitor still shares this lock: revoke the queued
                    # child before a failed association can reach execution.
                    try:
                        pool.cancel(handle, reason="scan_association_failed")
                    finally:
                        job = pool._known[handle.attempt_id]
                        if job in pool._pending:
                            pool._pending.remove(job)  # Retain the DB row if storage itself failed.
                    raise
                in_flight += 1
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Detached aggregate view; original numerical and qualification fields survive."""
        pool = self.supervisor
        with pool._condition:
            points = []
            for index, point in enumerate(self.spec.points):
                row = self._existing(index)
                value: dict[str, Any] = {"point_id": point.point_id, "run_id": point.run_id,
                    "coordinates": _parse_document(point.coordinates_json), "limits": asdict(point.spec.limits),
                    "initial_state": {"id": point.spec.initial_state_id, "sha256": point.spec.initial_state_sha256},
                    "manifest_sha256": point.spec.manifest_sha256, **self._metadata[index][0],
                    "state": "declared", "reason": "waiting_for_capacity", "attempt_id": None,
                    "reused": False, "closure_pending": False, "result": None, "attempts": []}
                if row is not None:
                    attempts = pool.store.attempts(point.run_id)
                    attempt = attempts[-1]
                    result = attempt["result"]
                    reason = row["state"]
                    if result is not None:
                        reason = result.get("worker_result", result).get("reason", "completed" if row["state"] == "succeeded" else row["state"])
                    value.update(state=row["state"], reason=reason, attempt_id=attempt["attempt_id"],
                        generation=attempt["generation"], reused=attempt["attempt_id"] in self._reused,
                        closure_pending=self._owned(point.run_id), result=result,
                        attempts=[{"attempt_id": a["attempt_id"], "state": a["state"], "generation": a["generation"]} for a in attempts])
                points.append(value)
            complete = all(p["state"] in _TERMINAL and not p["closure_pending"] for p in points)
            state = "succeeded" if all(p["state"] == "succeeded" for p in points) else "failed"
            if self._cancel_reason() is not None:
                state = "cancelled"
            return {"schema": "solarlab.scan-status.v1", "scan_id": self.spec.scan_id, "scan_sha256": self._sha256,
                    "state": state if complete else "running", "complete": complete, "points": points,
                    "max_in_flight": self.max_in_flight, "coordinator_uses_process_slot": False,
                    "scientific_qualification_inferred": False}

    def wait(self, timeout: float) -> dict[str, Any]:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("scan wait requires a positive finite timeout")
        pool = self.supervisor
        deadline = time.monotonic() + timeout
        with pool._condition:
            while True:
                result = self.advance()
                if result["complete"]:
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("scan remains resumable; caller wait timeout is not execution cancellation")
                pool._condition.wait(min(remaining, pool._poll))

    def cancel(self, *, reason: str) -> bool:
        if type(reason) is not str or not reason.strip() or len(reason.encode()) > 4096:
            raise ValueError("scan cancellation requires a bounded explicit reason")
        pool = self.supervisor
        with pool._condition:
            pool._check_open()
            if self.snapshot()["complete"]:
                return False
            self._persist(self._cancel_path, {"schema": "solarlab.scan-cancel.v1", "scan_sha256": self._sha256, "reason": reason})
            self.advance()
            return True

    def retry(self, point_id: str, *, expected_attempt_id: str) -> JobHandle:
        """Explicit one-point retry; old failures survive and no retry loop exists."""
        pool = self.supervisor
        with pool._condition:
            pool._check_open()
            if self._cancel_reason() is not None:
                raise StateConflict("a cancelled scan stays cancelled; retry explicitly under a new scan declaration")
            index = next((i for i, p in enumerate(self.spec.points) if p.point_id == point_id), None)
            if index is None:
                raise KeyError(point_id)
            row = self._existing(index)
            point = self.spec.points[index]
            if row is None or row["current_attempt_id"] != expected_attempt_id or row["state"] not in {"failed", "cancelled"}:
                raise StateConflict("retry requires the exact current failed/cancelled attempt")
            if self._owned(point.run_id):
                raise StateConflict("retry cannot precede the owned process's observed exit")
            attempt = pool.store.attempts(point.run_id)[-1]
            receipt = pool._done.get(expected_attempt_id)
            closed = receipt is not None and receipt.data.get("process_exit_observed") is True
            reconciled = (pool.work_root / (pool.store.store_id + "-" + expected_attempt_id) / "reconciled.json").exists()
            if attempt["started_ns"] is not None and not closed and not reconciled:
                raise StateConflict("retry requires positive previous process closure")
            queued = pool.store.retry(point.run_id, expected_attempt_id=expected_attempt_id, expected_generation=row["generation"])
            return JobHandle(pool.store.store_id, point.run_id, queued["current_attempt_id"])

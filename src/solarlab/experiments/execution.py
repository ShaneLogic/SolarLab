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
        self._store_baseline = _directory_bytes(store.root)
        self._shared_output_peak = _directory_bytes(root)
        try:
            for directory in sorted(root.glob(store.store_id + "-*")):
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("unsafe previous owned work directory")
                if (directory / "receipt.json").exists():
                    receipt = self._read_json(directory / "receipt.json")
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
                    if receipt.get("terminal_committed"):
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

    def _check_open(self) -> None:
        if self._fatal is not None:
            raise RuntimeError("supervisor control failure; inspect retained process records") from self._fatal
        if self._closing or self._closed:
            raise RuntimeError("supervisor is closing or closed")

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
        if job.token is not None:
            try:
                self.store.complete(job.token, state="failed", result={"reason": "spawn_failed",
                    "detail": str(error)[:1024], "process_started": False})
                committed = True
            except StateConflict:
                pass
        prior = next(a for a in self.store.attempts(job.handle.run_id) if a["attempt_id"] == job.handle.attempt_id)
        receipt = {"status": "prepared_pending_dependencies", "run_id": job.handle.run_id,
            "attempt_id": job.handle.attempt_id, "store_state": prior["state"], "terminal_committed": committed,
            "reason": "spawn_failed_or_queued_authority_revoked", "detail": str(error)[:1024],
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
        information = _process_info(pids, include_descendants=True)
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
            elif exitcode == 0:
                result = {"reason": "missing_terminal"}
        except (ValueError, TypeError, KeyError, OSError) as error:
            state, result = "failed", {"reason": "invalid_terminal", "detail": str(error)[:1024]}
            payloads = []
        reason = self._resource_reason(job, time.monotonic())
        if reason is not None:
            state, result = "failed", {"reason": reason}
        if state == "succeeded":
            missing = [name for name, cap in (("accepted_steps", job.limits.accepted_steps), ("calls", job.limits.calls))
                       if cap is not None and job.counts[name] is None]
            if missing or job.peak_rss is None:
                state, result = "failed", {"reason": "required_measurement_unobserved", "counters": missing}
        if job.stop_reason is not None:
            state, result = job.stop_state, {"reason": job.stop_reason, "discarded_terminal_state": manifest.get("state")}
            if job.setup_error is not None:
                result["setup_error"] = job.setup_error
        references: list[str] = []
        published_bytes = 0
        publication_error: str | None = None
        if state == "succeeded":
            try:
                for name, payload in payloads:
                    if _directory_bytes(job.directory) + published_bytes + len(payload.data) > job.limits.output_bytes:
                        raise ValueError("output limit before parent publication")
                    guard = self._output_guard(len(payload.data), self._completion_allowance(job, result))
                    if guard["would_exceed"]:
                        state, result, references = "failed", {"reason": "shared_output_publication_reservation", "budget": guard}, []
                        break
                    published = self.store.publish(job.token, name, payload)
                    references.append(published["artifact_id"])
                    published_bytes += len(payload.data)
                    guard = self._output_guard(0, self._completion_allowance(job, result))
                    if guard["would_exceed"]:
                        state, result, references = "failed", {"reason": "shared_output_after_publication", "budget": guard}, []
                        break
            except (ValueError, OSError, StateConflict) as error:
                publication_error = str(error)[:1024]
                state, result, references = "failed", {"reason": "artifact_publication", "detail": publication_error}, []
        if time.monotonic() - job.started > job.limits.wall_seconds and state == "succeeded":
            state, result, references = "failed", {"reason": "wall_limit_during_publication"}, []
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
        if state == "succeeded" and guard["would_exceed"]:
            state, result, references = "failed", {"reason": "shared_output_finalization_reservation", "budget": guard}, []
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
                    if self._closing:
                        self._pending.clear()  # SQLite queued rows remain available for explicit resubmission.
                        for job in list(self._active.values()):
                            self._request_stop(job, "supervisor_shutdown", "cancelled")
                    if self._active:
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
                    while (not self._closing and self._pending and self._shared_output_peak <= self._output_cap
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
                self._fatal = error
                # Failure cleanup still owns the exact children. Never signal recovered PIDs.
                for job in self._active.values():
                    if job.process is not None and job.process.pid is not None:
                        if job.process.is_alive():
                            job.process.kill()
                        job.process.join(timeout=1.0)
                        if job.process.exitcode is not None and job.directory is not None:
                            path = job.directory / "exit.json"
                            if not path.exists():
                                _atomic_json(path, {"token": asdict(job.token) if job.token else None, "pid": job.pid,
                                    "exitcode": job.process.exitcode, "process_joined": True, "process_exit_observed": True,
                                    "reason": "supervisor_control_failure"})
                self._condition.notify_all()
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def reconcile_interrupted(self) -> dict[str, Any]:
        """Explicitly invalidate affected tokens and observe, never kill, old PIDs."""
        with self._condition:
            released, unresolved, invalidated = [], [], []
            for attempt_id, item in list(self._recovery.items()):
                directory, launch = item["directory"], item["launch"]
                token = AttemptToken(**launch["token"])
                evidence: dict[str, Any] = {"process_exit_observed": False, "method": "unknown_intent", "pid": None}
                if (directory / "exit.json").exists():
                    old = self._read_json(directory / "exit.json")
                    if old.get("token") != asdict(token) or old.get("process_joined") is not True or type(old.get("exitcode")) is not int:
                        raise ValueError("invalid prior positive exit receipt")
                    evidence = {"process_exit_observed": True, "method": "prior_owned_join", "pid": old["pid"], "exitcode": old["exitcode"]}
                else:
                    ready: dict[str, Any] = {}
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
                            current = _process_info([pid]).get(pid)
                            previous = ready.get("os_identity")
                            if current is not None and previous is not None and current["start_lstart"] != previous["start_lstart"]:
                                evidence.update(process_exit_observed=True, method="observed_pid_reuse", old_identity=previous, current_identity=current)
                current_run = self.store.get_run(token.run_id)
                if (current_run["current_attempt_id"] == token.attempt_id and current_run["generation"] == token.generation
                        and current_run["state"] == "running"):
                    self.store.complete(token, state="failed", result={"reason": "interrupted", "ownership_invalidated": True,
                        "process_exit_observed": evidence["process_exit_observed"], "process_evidence": evidence})
                    invalidated.append(attempt_id)
                if evidence["process_exit_observed"]:
                    _atomic_json(directory / "reconciled.json", {"token": asdict(token), "evidence": evidence})
                    released.append(attempt_id)
                    del self._recovery[attempt_id]
                    if attempt_id not in self._done:
                        self._unknown_history.add(attempt_id)
                else:
                    unresolved.append({"attempt_id": attempt_id, **evidence})
            self._condition.notify_all()
            return {"invalidated_owned_attempts": invalidated, "released_owned_reservations": released,
                    "unresolved_owned_slots": unresolved, "unowned_processes_signalled": False,
                    "queued_policy": "explicit_matching_resubmission; no automatic scientific rerun"}

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
                    "known_owned_wall_seconds": sum(r["owned_wall_seconds"] for r in rows),
                    "counter_totals": counts, "unknown_historical_attempts": len(self._recovery) + len(self._unknown_history),
                    "shared_rss_last_observed_bytes": self._shared_rss, "shared_rss_peak_bytes": self._shared_peak_rss,
                    "known_family_rss_last_observed_bytes": self._known_family_rss,
                    "rss_scope": "conservative_parent_process_family_and_located_prior_intents",
                    "sampled_family_pids": list(self._sampled_family_pids),
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

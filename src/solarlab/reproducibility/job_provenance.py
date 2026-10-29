"""Job-owned source capture and read-only provenance comparison.

Only admission reads the execution source tree. Numerical code may carry the
active record's reference, but never discovers or rechecks Python files. A
historical reader consumes recorded data and an optional supplied current
record; it does not discover the current installation.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Literal


SCHEMA = "solarlab.job-provenance.v1"
DirtyPolicy = Literal["record", "refuse"]
_LIMIT = 4 * 1024**2
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
_ACTIVE: ContextVar[JobProvenance | None] = ContextVar("solarlab_job_provenance", default=None)
_OWNER: ContextVar[dict[str, Any] | None] = ContextVar("solarlab_job_provenance_owner", default=None)


def canonical(value: Any) -> bytes:
    """Canonical metadata bytes, separate from original numerical array bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("provenance requires a SHA-256 content digest")


def _commit(value: Any) -> None:
    if value is not None and (not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None):
        raise ValueError("provenance commit must be an immutable commit or unknown")


@dataclass(frozen=True, slots=True)
class JobProvenance:
    """Immutable canonical admission metadata; no acceptance is inferred."""

    canonical_json: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not bytes or len(self.canonical_json) > _LIMIT:
            raise ValueError("provenance must be bounded immutable metadata")
        value = json.loads(self.canonical_json)
        fields = {"schema", "git_commit", "source_state", "dirty_policy", "dirty_paths",
                  "source_tree_sha256", "source_scope", "source_files", "input_sha256",
                  "environment_lock_sha256", "runtime_observed", "assembler", "attempt_ref", "execution_manifest_sha256"}
        if not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA:
            raise ValueError("unsupported job provenance schema")
        if canonical(value) != self.canonical_json:
            raise ValueError("job provenance is not canonical")
        _commit(value["git_commit"])
        for name in ("input_sha256", "source_tree_sha256"):
            _sha(value[name])
        _sha(value["environment_lock_sha256"], optional=True)
        _sha(value["execution_manifest_sha256"], optional=True)
        if value["dirty_policy"] not in {"record", "refuse"}:
            raise ValueError("explicit dirty-source policy is required")
        if value["source_state"] not in {"clean", "dirty_recorded", "unknown"}:
            raise ValueError("unsupported source state")
        paths = value["dirty_paths"]
        if (not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths)
                or paths != sorted(set(paths))):
            raise ValueError("dirty paths must be unique recorded paths")
        if value["source_state"] == "clean" and (paths or value["git_commit"] is None):
            raise ValueError("clean provenance cannot conceal dirty or unknown source")
        if value["source_state"] == "dirty_recorded" and (not paths or value["dirty_policy"] != "record"):
            raise ValueError("dirty provenance needs explicit recording")
        files = value["source_files"]
        if not isinstance(files, list):
            raise ValueError("source inventory must be a list")
        names = []
        for item in files:
            if (not isinstance(item, dict) or set(item) != {"path", "mode", "sha256"}
                    or not isinstance(item["path"], str) or not item["path"]
                    or item["mode"] not in {"file", "executable", "supplied"}):
                raise ValueError("invalid admitted source record")
            _sha(item["sha256"])
            names.append(item["path"])
        if names != sorted(set(names)) or digest(files) != value["source_tree_sha256"]:
            raise ValueError("source tree digest differs from the admitted inventory")
        if value["source_state"] == "clean" and not files:
            raise ValueError("clean provenance requires an actual source inventory")
        if not isinstance(value["source_scope"], str) or not value["source_scope"]:
            raise ValueError("source inventory scope is required")
        assembler = value["assembler"]
        if (not isinstance(assembler, dict) or set(assembler) != {"operator", "unified_assembler", "identity", "analysis_identity"}
                or not isinstance(assembler["operator"], str) or not assembler["operator"]):
            raise ValueError("actual assembler/operator description is required")
        if type(assembler["unified_assembler"]) is not bool:
            raise ValueError("assembler use must be explicit")
        if assembler["unified_assembler"] != (assembler["identity"] is not None):
            raise ValueError("an assembled operation requires its actual Assembler identity")
        _sha(assembler["identity"], optional=True)
        _sha(assembler["analysis_identity"], optional=True)
        if not isinstance(value["runtime_observed"], dict):
            raise ValueError("observed runtime metadata is required")
        attempt = value["attempt_ref"]
        if attempt is not None and (not isinstance(attempt, dict)
                or any(not isinstance(k, str) or not isinstance(v, (str, int)) for k, v in attempt.items())):
            raise ValueError("invalid provenance attempt reference")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> JobProvenance:
        return cls(canonical(dict(record)))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json).hexdigest()

    def export(self) -> dict[str, Any]:
        return dict(json.loads(self.canonical_json))

    def envelope(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "record": self.export()}


def read_provenance(value: Mapping[str, Any]) -> JobProvenance:
    """Check recorded provenance content without examining current source."""
    if set(value) != {"sha256", "record"} or not isinstance(value["record"], dict):
        raise ValueError("invalid provenance envelope")
    record = JobProvenance.from_record(value["record"])
    if record.sha256 != value["sha256"]:
        raise ValueError("producer provenance content digest mismatch")
    return record


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise ValueError("execution source is not an available Git checkout")
    return result.stdout


def execution_checkout() -> Path | None:
    """Locate this imported package's source checkout, never a caller's cwd."""
    candidate = Path(__file__).resolve().parents[3]
    if not (candidate / "pyproject.toml").is_file():
        return None
    try:
        actual = Path(_git(candidate, "rev-parse", "--show-toplevel").strip()).resolve()
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None
    return actual if actual == candidate else None


def _runtime() -> dict[str, Any]:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(),
            "packages": dict(sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                                    for d in importlib.metadata.distributions() if d.metadata["Name"])),
            "settings": {name: os.environ.get(name) for name in _THREADS}}


def capture_job_provenance(
    inputs: Mapping[str, Any], *, dirty_policy: DirtyPolicy, operator: str,
    source_root: Path | None = None, git_commit: str | None = None,
    source_files: Sequence[tuple[str, bytes]] | None = None,
    environment_lock: bytes | None = None, source_attested: bool = False,
    source_dirty_paths: Sequence[str] = (),
    assembler_identity: str | None = None,
    analysis_identity: str | None = None, attempt_ref: Mapping[str, str | int] | None = None,
    execution_manifest_sha256: str | None = None,
) -> JobProvenance:
    """Capture once at a trusted job boundary; supplied bytes require authority.

    A Git checkout is inspected only when explicitly supplied. An installed
    source bundle can supply its existing attested commit/content/lock instead;
    without that attestation its source state remains unknown.
    """
    if dirty_policy not in {"record", "refuse"}:
        raise ValueError("explicit dirty-source policy is required")
    _commit(git_commit)
    _sha(assembler_identity, optional=True)
    _sha(analysis_identity, optional=True)
    if assembler_identity is not None and analysis_identity is None:
        raise ValueError("an actual assembler requires explicit analysis-operation identity")
    dirty: list[str] = sorted(set(source_dirty_paths))
    files: list[dict[str, str]] = []
    state = "dirty_recorded" if dirty else "clean" if source_attested and git_commit is not None else "unknown"
    scope = "supplied-content-v1" if source_files is not None else "unrecorded"
    if source_root is not None:
        if source_dirty_paths:
            raise ValueError("checkout changes are observed, never supplied by payload")
        root = source_root.resolve(strict=True)
        actual = _git(root, "rev-parse", "HEAD").strip()
        if git_commit is not None and git_commit != actual:
            raise ValueError("job source commit differs from the selected checkout")
        git_commit = actual
        changed = _git(root, "diff", "--name-only", "-z", "HEAD", "--", "src", "pyproject.toml", "uv.lock")
        new = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", "src", "pyproject.toml", "uv.lock")
        dirty = sorted(set(filter(None, (changed + new).split("\0"))))
        state = "dirty_recorded" if dirty else "clean"
        scope = "checkout-first-party-src/path-mode-sha256-v1"
        observations: dict[Path, os.stat_result] = {}
        inventory = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "src")
        for name in sorted(set(filter(None, inventory.split("\0")))):
            path = root / name
            if not path.exists():
                continue  # Deletions are retained in dirty_paths.
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError("source snapshot contains an escaping or non-file path")
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                continue
            observations[path] = path.stat()
            files.append({"path": name, "mode": "executable" if path.stat().st_mode & 0o111 else "file",
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if environment_lock is None and (root / "uv.lock").is_file():
            observations[root / "uv.lock"] = (root / "uv.lock").stat()
            environment_lock = (root / "uv.lock").read_bytes()
        def stamp(info: os.stat_result) -> tuple[int, int, int, int]:
            return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        if (any(not path.is_file() or stamp(path.stat()) != stamp(before) for path, before in observations.items())
                or _git(root, "rev-parse", "HEAD").strip() != actual
                or _git(root, "diff", "--name-only", "-z", "HEAD", "--", "src", "pyproject.toml", "uv.lock") != changed
                or _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", "src", "pyproject.toml", "uv.lock") != new
                or _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "src") != inventory):
            raise ValueError("execution source snapshot changed during admission; freeze it before running")
    if source_files is not None:
        if source_root is not None:
            raise ValueError("choose the execution checkout or supplied source snapshot explicitly")
        files = [{"path": name, "mode": "supplied", "sha256": hashlib.sha256(content).hexdigest()}
                 for name, content in sorted(source_files)]
    if dirty_policy == "refuse" and state != "clean":
        raise ValueError("strict source admission requires a clean attested source snapshot")
    return JobProvenance.from_record({
        "schema": SCHEMA, "git_commit": git_commit, "source_state": state,
        "dirty_policy": dirty_policy, "dirty_paths": dirty, "source_scope": scope,
        "source_files": files, "source_tree_sha256": digest(files), "input_sha256": digest(dict(inputs)),
        "environment_lock_sha256": hashlib.sha256(environment_lock).hexdigest() if environment_lock is not None else None,
        "runtime_observed": _runtime(), "assembler": {"operator": operator,
            "unified_assembler": assembler_identity is not None,
            "identity": assembler_identity, "analysis_identity": analysis_identity or digest({"operator": operator})},
        "attempt_ref": dict(attempt_ref) if attempt_ref is not None else None,
        "execution_manifest_sha256": execution_manifest_sha256,
    })


def current_provenance() -> dict[str, Any] | None:
    """Return detached admission metadata; no capture or filesystem access."""
    active = _ACTIVE.get()
    return None if active is None else active.envelope()


def current_provenance_ref() -> str | None:
    active = _ACTIVE.get()
    return None if active is None else active.sha256


def admitted_source_pins(*, path_keys: bool = False) -> dict[str, str]:
    """Compatibility metadata from the one admitted snapshot, never a rehash.

    Bare numerical objects have no producer claim and expose an empty map.
    Full scientific provenance is the enclosing job record, not this legacy
    display field. Snapshot membership does not imply an enabled physical law.
    """
    active = _ACTIVE.get()
    if active is None:
        return {}
    result = {}
    for item in active.export()["source_files"]:
        name = item["path"].removeprefix("src/")
        if name.startswith("solarlab/") and name.endswith(".py"):
            key = name if path_keys else name[:-3].replace("/", ".")
            result[key] = item["sha256"]
    return result


@contextmanager
def provenance_owner(**options: Any) -> Iterator[dict[str, Any]]:
    """Supply trusted job/package capture options to nested producer calls."""
    owner = dict(options)
    token = _OWNER.set(owner)
    active_token = _ACTIVE.set(None)
    try:
        yield owner
    finally:
        _ACTIVE.reset(active_token)
        _OWNER.reset(token)


@contextmanager
def job_provenance(
    inputs: Mapping[str, Any], *, operator: str | None, assembler_identity: str | None = None,
    analysis_identity: str | None = None,
) -> Iterator[JobProvenance]:
    """Admit once and reject reuse under a different declared binding.

    ``operator=None`` explicitly inherits an existing unassembled job's record
    for a preparation substage; it cannot declare or conceal an assembler.
    """
    _sha(assembler_identity, optional=True)
    _sha(analysis_identity, optional=True)
    if assembler_identity is not None and analysis_identity is None:
        raise ValueError("provenance binding requires explicit analysis-operation identity")
    owner = _OWNER.get()
    existing = _ACTIVE.get() or (None if owner is None else owner.get("_record"))
    if existing is not None:
        binding = existing.export()["assembler"]
        if operator is None:
            if (assembler_identity is not None or analysis_identity is not None
                    or binding["unified_assembler"]):
                raise ValueError("an assembled operation requires an explicit provenance binding")
        elif binding != {"operator": operator,
                "unified_assembler": assembler_identity is not None,
                "identity": assembler_identity,
                "analysis_identity": analysis_identity or digest({"operator": operator})}:
            raise ValueError("declared provenance binding differs from the admitted job")
        active_token = _ACTIVE.set(existing)
        try:
            yield existing
        finally:
            _ACTIVE.reset(active_token)
        return
    if operator is None:
        raise ValueError("inherited provenance binding requires an admitted unassembled job")
    options = dict(owner or {})
    actual_inputs = options.pop("input_record", inputs)
    if "source_files" not in options:
        options.setdefault("source_root", execution_checkout())
    options.setdefault("dirty_policy", "record")
    record = capture_job_provenance(actual_inputs, operator=operator, assembler_identity=assembler_identity,
                                    analysis_identity=analysis_identity, **options)
    if owner is not None:
        owner["_record"] = record
    token = _ACTIVE.set(record)
    try:
        yield record
    finally:
        _ACTIVE.reset(token)


def compare_provenance(
    producer: Mapping[str, Any] | None, current: JobProvenance | Mapping[str, Any] | None = None,
    *, strict_same_source: bool = False,
) -> dict[str, Any]:
    """Compare supplied records only; source drift is informational by default."""
    old = None if producer is None else read_provenance(producer)
    now = current if isinstance(current, JobProvenance) else None if current is None else read_provenance(current)
    left = None if old is None else old.export()
    right = None if now is None else now.export()
    keys = ("git_commit", "source_tree_sha256", "source_scope", "environment_lock_sha256", "assembler", "runtime_observed")
    differences = [] if left is None or right is None else [name for name in keys if left[name] != right[name]]
    known = (left is not None and right is not None
             and all(value["source_state"] != "unknown" and value["git_commit"] is not None
                     and value["environment_lock_sha256"] is not None
                     and value["assembler"]["analysis_identity"] is not None
                     and value["source_scope"] in {"supplied-content-v1", "checkout-first-party-src/path-mode-sha256-v1"}
                     for value in (left, right)))
    complete = bool(known and left is not None and right is not None
                    and left["source_state"] == right["source_state"] == "clean")
    status = "unknown" if not known else "different" if differences else "same"
    if strict_same_source and (not complete or differences):
        raise ValueError("strict_same_source requires matching clean producer/current source, lock and assembler evidence")
    return {"producer_identity": None if old is None else old.envelope(),
            "current_identity": None if now is None else now.envelope(),
            "status": status, "different_fields": differences, "complete_same_source_evidence": bool(complete and not differences),
            "strict_same_source": strict_same_source, "scientific_qualification": False}


def result_provenance(
    record: Mapping[str, Any], current: JobProvenance | Mapping[str, Any] | None = None,
    *, strict_same_source: bool = False, expected_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a standalone result; process claims need their RunStore reader."""
    producer = record.get("job_provenance")
    if producer is not None:
        value = read_provenance(producer).export()
        if value["attempt_ref"] is not None or value["execution_manifest_sha256"] is not None:
            raise ValueError("process provenance requires its verified RunStore attempt and execution manifest")
        if expected_inputs is not None and value["input_sha256"] != digest(dict(expected_inputs)):
            raise ValueError("saved producer provenance belongs to different inputs")
    return _result_provenance_comparison(record, current, strict_same_source=strict_same_source)


def validate_recorded_sources(record: Mapping[str, Any]) -> None:
    """Check recorded source content only, without asserting input admission.

    Store readers use this for inner payloads before independently verifying
    the outer job's attempt, execution manifest, input and publication record.
    """
    producer = record.get("job_provenance")
    admitted = None if producer is None else read_provenance(producer).export()
    legacy = record.get("source_sha256", record.get("sources"))
    if isinstance(legacy, dict):
        for value in legacy.values():
            _sha(value)
    if admitted is not None and isinstance(legacy, dict):
        pins = {item["path"].removeprefix("src/"): item["sha256"] for item in admitted["source_files"]}
        for key, value in legacy.items():
            path = key if "/" in key else key.replace(".", "/") + ".py"
            if pins.get(path) != value:
                raise ValueError("saved source metadata differs from its producer provenance")


def _result_provenance_comparison(
    record: Mapping[str, Any], current: JobProvenance | Mapping[str, Any] | None,
    *, strict_same_source: bool,
) -> dict[str, Any]:
    validate_recorded_sources(record)
    result = compare_provenance(record.get("job_provenance"), current, strict_same_source=strict_same_source)
    result["legacy_source_sha256"] = record.get("source_sha256")
    result["legacy_source_identity"] = record.get("source_identity")
    return result


def verify_loaded_source(content: bytes, filename: str | Path, *, label: str) -> None:
    """Admission consistency for a trusted installed consumer, never readback."""
    if Path(filename).read_bytes() != content:
        raise ValueError("loaded " + label + " differs from the bound source bytes")


def worker_input_record(transport: Mapping[str, Any], execution_document: Mapping[str, Any] | None) -> dict[str, Any]:
    """Actual admitted inputs; source/attempt identity is outside this digest."""
    value = {"request_sha256": transport["documents"]["request"]["sha256"],
             **{k: transport[k] for k in ("H_input", "H_physics")}}
    if execution_document is not None:
        value["execution_inputs"] = {name: execution_document["content_roles"][name]["sha256"]
            for name in ("protocol", "initial_state", "grid", "numeric_controls")}
    return value


def worker_result_provenance(
    record: Mapping[str, Any], run: Mapping[str, Any], attempt: Mapping[str, Any], *, store_id: str,
    current: JobProvenance | Mapping[str, Any] | None = None, strict_same_source: bool = False,
    provenance_ref: str | None = None, execution_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a job receipt against saved transport and attempt; never recapture."""
    producer = record.get("job_provenance")
    if producer is not None:
        value = read_provenance(producer).export()
        reference = value["attempt_ref"]
        expected = {"store_id": store_id, "run_id": run["run_id"],
                    "attempt_id": attempt["attempt_id"], "owner_id": attempt["owner_id"], "generation": attempt["generation"]}
        if reference is None or any(reference.get(k) != v for k, v in expected.items()):
            raise ValueError("saved job provenance belongs to another attempt")
        if provenance_ref != read_provenance(producer).sha256:
            raise ValueError("saved job provenance differs from its publication reference")
        manifest_sha = run["request"]["input_metadata"]["execution_manifest"]["manifest_sha256"]
        if value["execution_manifest_sha256"] != manifest_sha or record.get("manifest_sha256") != manifest_sha:
            raise ValueError("saved job provenance belongs to a different execution manifest")
        transport = run["request"]["input_metadata"]["transport"]
        if execution_document is None:
            raise ValueError("saved job provenance requires its verified execution content")
        inputs = worker_input_record(transport, execution_document)
        if value["input_sha256"] != digest(inputs):
            raise ValueError("saved job provenance input differs from transport")
    return _result_provenance_comparison(record, current, strict_same_source=strict_same_source)


def publication_binding(producer: Mapping[str, Any], *, owner: Mapping[str, Any],
                        request_identity: Mapping[str, Any], result_sha256: str) -> dict[str, Any]:
    """Seal an already captured computation to a later publication token.

    The original admission record is unchanged. This digest excludes both the
    later publication-manifest digest and its own checksum, avoiding a cycle.
    """
    _sha(result_sha256)
    value = {"schema": "solarlab.provenance-publication.v1",
             "admission_sha256": read_provenance(producer).sha256,
             "owner": dict(owner), "request_identity": dict(request_identity),
             "result_sha256": result_sha256}
    return {"sha256": digest(value), "record": value}

"""Content-bound R1 snapshots; trusted startup and approval remain separate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping
import zipfile


PROJECT_RELATIVE_PATH = "perovskite-sim"
STUDY_INPUT_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
_CHECKOUT_MODULE = "perovskite_sim/experiments/one_dimensional_mechanism_r1_checkout.py"
_RUNNER = "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
_LAUNCHER = "scripts/run_one_dimensional_mechanism_r1_controlled.py"
_ANCHORS = (
    _CHECKOUT_MODULE,
    _RUNNER,
    _LAUNCHER,
    "scripts/run_one_dimensional_mechanism_r1.py",
    "scripts/run_one_dimensional_mechanism_r0.py",
    STUDY_INPUT_RELATIVE_PATH,
    "docs/OneDimensionalMechanismR1DynamicsV1.md",
    "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
)
LEGACY_SOURCE_ANCHORS = _ANCHORS
REQUIRED_SOURCE_ANCHORS = (
    *_ANCHORS, "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md",
    "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json",
    "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV1.md",
    "docs/OneDimensionalMechanismR1EvidenceV5.md",
    "docs/OneDimensionalMechanismR1PhysicsProtocolV1.md",
)
_CURRENT_CONTEXT = None


class R1CheckoutError(ValueError):
    """Formal R1 execution lacks the required tracked-checkout provenance."""


@dataclass(frozen=True)
class R1CheckoutContext:
    root: Path
    project: Path
    commit: str
    tracked_paths: tuple[str, ...]
    required_sources: dict[str, dict[str, str | int]]
    dirty: dict[str, bool]
    run_class: str = "development"
    source_commit: str | None = None
    source_content_sha256: str | None = None
    content_mismatches: tuple[str, ...] = ()
    runtime: dict | None = None
    _source_bytes: Mapping[str, bytes] = field(default_factory=dict, repr=False, compare=False)
    _source_changes: bytes = field(default=b"", repr=False, compare=False)

    @property
    def study_input(self):
        return self.project / STUDY_INPUT_RELATIVE_PATH

    def read_bytes(self, path):
        """Return frozen bytes, never reopen a possibly changed source."""
        path = Path(path)
        if not path.is_absolute():
            path = (self.root if path.parts[:1] == (PROJECT_RELATIVE_PATH,) else self.project) / path
        try:
            name = Path(os.path.abspath(path)).relative_to(self.root).as_posix()
            return self._source_bytes[name]
        except (ValueError, KeyError) as exc:
            raise R1CheckoutError(f"path is absent from the frozen R1 source snapshot: {path}") from exc

    def package_source_hashes(self):
        prefix = PROJECT_RELATIVE_PATH + "/perovskite_sim/"
        return {name[len(prefix):]: hashlib.sha256(raw).hexdigest()
                for name, raw in sorted(self._source_bytes.items())
                if name.startswith(prefix) and name.endswith(".py")}

    def to_dict(self):
        return {
            "schema": "R1ExecutionSourceV2",
            "repository_root": str(self.root),
            "project_relative_path": PROJECT_RELATIVE_PATH,
            "observed_commit": self.commit,
            "source_commit": self.source_commit,
            "source_content_sha256": self.source_content_sha256,
            "run_class": self.run_class,
            "dirty": dict(self.dirty),
            "dirty_note": "Git status is advisory; required-source content is compared separately",
            "status_scope": "advisory working-tree observation at capture",
            "required_source_snapshot_matches_commit": not self.content_mismatches,
            "archived_source_is_commit_snapshot": self.run_class == "formal",
            "required_source_content_mismatches": list(self.content_mismatches),
            "tracked_path_count": len(self.tracked_paths),
            "required_sources": {name: dict(entry) for name, entry in self.required_sources.items()},
            "runtime": self.runtime,
            "scope": ("caller-anchored committed source snapshot; approval is separate"
                      if self.run_class == "formal" else
                      "development source snapshot; no controlled execution or independent approval asserted"),
        }


def current_execution_context():
    """The trusted startup path installs this object directly, not through env."""
    return _CURRENT_CONTEXT


def _install_controlled_context(context, runtime):
    global _CURRENT_CONTEXT
    if not (sys.flags.isolated and sys.flags.no_site):
        raise R1CheckoutError("controlled R1 execution must start with Python -I -S")
    if not isinstance(context, R1CheckoutContext) or context.run_class != "formal":
        raise R1CheckoutError("controlled startup requires a committed source context")
    if _CURRENT_CONTEXT is not None:
        raise R1CheckoutError("a controlled R1 context is already installed")
    # Arbitrary replacement of Python objects or the launcher is not the threat
    # model. There is deliberately no environment-variable authorization token.
    _CURRENT_CONTEXT = replace(context, runtime=dict(runtime))
    return _CURRENT_CONTEXT


def git_environment():
    """Ignore ambient redirects, config paths and object replacement refs."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_SYSTEM": os.devnull,
                        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1",
                        "GIT_OPTIONAL_LOCKS": "0"})
    return environment


def _git(directory, *arguments, input=None):
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise R1CheckoutError("R1 research execution requires Git and a source checkout")
    try:
        completed = subprocess.run(
            [executable, "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull,
             "-c", "core.excludesFile=" + os.devnull, "-C", str(directory), *arguments],
            input=input, capture_output=True, check=False, env=git_environment(),
        )
    except OSError as exc:
        raise R1CheckoutError("R1 research execution requires Git and a source checkout") from exc
    if completed.returncode:
        raise R1CheckoutError(
            "R1 research execution requires a Git checkout with its tracked study input; "
            "package-only and wheel installations are not formal execution contexts"
        )
    return completed.stdout


def _commit_tree(root, commit):
    entries = {}
    for entry in _git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            entries[name.decode("utf-8")] = tuple(metadata.decode("ascii").split())
    return entries


def _commit_bytes(root, tree):
    objects = list(dict.fromkeys(oid for mode, kind, oid in tree.values()
                                if kind == "blob" and mode in ("100644", "100755")))
    raw = _git(root, "cat-file", "--batch", input=("\n".join(objects) + "\n").encode())
    blobs, cursor = {}, 0
    for expected in objects:
        end = raw.index(b"\n", cursor)
        oid, kind, length = raw[cursor:end].decode("ascii").split()
        length = int(length)
        if oid != expected or kind != "blob":
            raise R1CheckoutError("Git returned an unexpected source object")
        cursor = end + 1
        blobs[oid] = raw[cursor:cursor + length]
        if len(blobs[oid]) != length or raw[cursor + length:cursor + length + 1] != b"\n":
            raise R1CheckoutError("Git source object is truncated")
        cursor += length + 1
    return {name: blobs[oid] for name, (mode, kind, oid) in tree.items()
            if kind == "blob" and mode in ("100644", "100755")}


def _identity(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def source_content_digest(entries):
    """Canonical digest of the required repo-relative {sha256, bytes} map."""
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_patch(original, captured):
    pieces = []
    for name in sorted(set(original) | set(captured)):
        before, after = original.get(name, b""), captured.get(name, b"")
        if before != after:
            try:
                pieces.extend(difflib.unified_diff(
                    before.decode("utf-8").splitlines(keepends=True),
                    after.decode("utf-8").splitlines(keepends=True),
                    fromfile="a/" + name, tofile="b/" + name))
            except UnicodeDecodeError:
                pieces.append(f"Binary source differs: {name}\n")
    return "".join(pieces).encode("utf-8")


def _file_identity(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _same_file(actual, expected):
    try:
        actual = Path(actual)
        expected = Path(expected)
        return actual.resolve() == expected.resolve() and actual.samefile(expected)
    except (OSError, TypeError, ValueError):
        return False


def _require_exact_path(actual, expected, label):
    if not _same_file(actual, expected):
        raise R1CheckoutError(f"{label} must use the expected tracked checkout path: {expected}")


def _require_import_origins(project, tracked):
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == "perovskite_sim" or name.startswith("perovskite_sim.")):
            continue
        module_path = project.joinpath(*name.split("."))
        package_paths = getattr(module, "__path__", None)
        expected = module_path / "__init__.py" if package_paths is not None else module_path.with_suffix(".py")
        actual = getattr(module, "__file__", None)
        if not _same_file(actual, expected):
            raise R1CheckoutError(f"unexpected imported package origin: {name}")
        relative = expected.relative_to(project.parent).as_posix()
        if relative not in tracked:
            raise R1CheckoutError(f"imported package source is not tracked: {relative}")
        if package_paths is not None:
            paths = tuple(package_paths)
            if len(paths) != 1 or not _same_file(paths[0], expected.parent):
                raise R1CheckoutError(f"unexpected imported package search path: {name}")
    for name in ("run_one_dimensional_mechanism_r1", "run_one_dimensional_mechanism_r0"):
        module = sys.modules.get(name)
        if module is not None:
            _require_exact_path(getattr(module, "__file__", None), project / "scripts" / (name + ".py"), name)


def require_r1_checkout(*, project=None, runner=None, formal=False, source_commit=None,
                        expected_source_sha256=None):
    """Capture development bytes or check content against a caller's commit.

    Formal CLI output additionally requires controlled startup. New development
    files must be declared to Git (intent-to-add is sufficient). Neither Git
    index flags nor status authorize formal content or imply approval.
    """
    active = current_execution_context()
    if active is not None:
        if project is not None and Path(project).resolve() != active.project:
            raise R1CheckoutError("project differs from the controlled source snapshot")
        if runner is not None and Path(runner).resolve() != active.project / _RUNNER:
            raise R1CheckoutError("runner differs from the controlled source snapshot")
        if source_commit is not None and source_commit != active.source_commit:
            raise R1CheckoutError("source commit differs from the controlled source snapshot")
        if expected_source_sha256 is not None and expected_source_sha256 != active.source_content_sha256:
            raise R1CheckoutError("source content differs from the caller anchor")
        return active
    module_file = Path(__file__).resolve()
    root = Path(_git(module_file.parent, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    expected_project = root / PROJECT_RELATIVE_PATH
    _require_exact_path(module_file, expected_project / _CHECKOUT_MODULE, "R1 checkout helper")
    if expected_project.is_symlink() or not expected_project.resolve().is_relative_to(root):
        raise R1CheckoutError("R1 project must be a physical directory in its tracked checkout")
    if project is not None:
        _require_exact_path(project, expected_project, "R1 project")
    if runner is not None:
        _require_exact_path(runner, expected_project / _RUNNER, "R1 runner")
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if source_commit is not None and (
        not isinstance(source_commit, str) or len(source_commit) not in (40, 64)
        or any(c not in "0123456789abcdef" for c in source_commit)
    ):
        raise R1CheckoutError("caller source commit must be a full lowercase Git object id")
    if formal and source_commit is None:
        raise R1CheckoutError("formal R1 execution requires an explicit caller source commit")
    selected = source_commit or head
    if formal and selected != head:
        raise R1CheckoutError("checkout HEAD differs from the caller source commit")
    tree = _commit_tree(root, selected)
    committed = _commit_bytes(root, tree)
    tracked = set(tree) if formal else {
        value for value in _git(root, "ls-files", "--cached", "--full-name", "-z").decode().split("\0")
        if value
    }
    package = expected_project / "perovskite_sim"
    package_sources = set(package.rglob("*.py"))
    prefix = PROJECT_RELATIVE_PATH + "/perovskite_sim/"
    package_sources.update(root / name for name in committed
                           if name.startswith(prefix) and name.endswith(".py"))
    if not package_sources:
        raise R1CheckoutError("R1 checkout has no package source coverage")
    required_paths = set(package_sources) | {expected_project / name for name in REQUIRED_SOURCE_ANCHORS}
    required, disk = {}, {}
    for path in sorted(required_paths):
        relative = path.relative_to(root).as_posix()
        if relative not in tracked or not path.is_file():
            raise R1CheckoutError(f"required R1 execution source is not a tracked file: {relative}")
        if path.is_symlink() or not path.resolve().is_relative_to(expected_project):
            raise R1CheckoutError(f"R1 execution source escapes its canonical checkout path: {relative}")
        disk[relative] = path.read_bytes()
        required[relative] = _identity(disk[relative])
    mismatches = tuple(name for name in sorted(required) if committed.get(name) != disk[name])
    if formal and mismatches:
        raise R1CheckoutError("required R1 source differs from caller commit blob: " + ", ".join(mismatches))
    content_sha256 = source_content_digest(required)
    if expected_source_sha256 is not None and content_sha256 != expected_source_sha256:
        raise R1CheckoutError("required R1 source content differs from the caller SHA-256 anchor")
    _require_import_origins(expected_project, tracked)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal").decode().splitlines()
    dirty = {
        "staged": any(line[:1] not in (" ", "?") for line in status),
        "unstaged": any(line[1:2] not in (" ", "?") for line in status),
        "untracked": any(line.startswith("?? ") for line in status),
    }
    if formal:
        captured, patch = committed, b""
    else:
        captured = {name: (root / name).read_bytes() for name in sorted(tracked)
                    if (root / name).is_file() and not (root / name).is_symlink()}
        captured.update(disk)
        patch = _source_patch(committed, captured)
        dirty["unstaged"] = bool(patch) or dirty["unstaged"]
    return R1CheckoutContext(
        root, expected_project, head, tuple(sorted(tracked)), required, dirty,
        "formal" if formal else "development", selected, content_sha256, mismatches,
        _source_bytes=MappingProxyType(dict(captured)), _source_changes=patch,
    )


def _atomic_bytes(path, raw):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".r1-source-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_frozen_source(output, context):
    """Save the captured/compiled byte map, without reopening disk sources."""
    if not isinstance(context, R1CheckoutContext) or not context._source_bytes:
        raise R1CheckoutError("a frozen source context is required")
    output = Path(output)
    entries = {name: _identity(raw) for name, raw in sorted(context._source_bytes.items())}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=".r1-source-", delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, raw in sorted(context._source_bytes.items()):
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, raw)
        os.replace(temporary, output / "SourceV1.zip")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _atomic_bytes(output / "SourceManifestV1.json", (json.dumps(entries, indent=2) + "\n").encode())
    _atomic_bytes(output / "SourceChangesV1.patch", context._source_changes)
    _atomic_bytes(output / "ExecutionSourceV1.json", (json.dumps(context.to_dict(), indent=2) + "\n").encode())
    validate_source_coverage(output, context)
    return context


def validate_source_coverage(output, context):
    """Match captured evidence to the frozen required execution bytes."""
    if not isinstance(context, R1CheckoutContext):
        raise TypeError("context must come from require_r1_checkout")
    if not context.required_sources:
        raise R1CheckoutError("R1 checkout context has no required source coverage")
    output = Path(output)
    manifest_path = output / "SourceManifestV1.json"
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise R1CheckoutError("R1 source manifest is missing or invalid") from exc
    if not isinstance(entries, dict) or not entries:
        raise R1CheckoutError("R1 source manifest must be a nonempty object")
    missing = set(context.required_sources) - entries.keys()
    if missing:
        raise R1CheckoutError("source coverage lacks required execution sources: " + ", ".join(sorted(missing)))
    try:
        with zipfile.ZipFile(output / "SourceV1.zip") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(entries):
                raise R1CheckoutError("source archive and manifest coverage differ")
            for name, expected in context.required_sources.items():
                if entries[name] != expected:
                    raise R1CheckoutError(f"source manifest identity mismatch: {name}")
                if context._source_bytes and _identity(context.read_bytes(name)) != expected:
                    raise R1CheckoutError(f"frozen execution source identity mismatch: {name}")
                info = archive.getinfo(name)
                if info.file_size != expected["bytes"]:
                    raise R1CheckoutError(f"source archive identity mismatch: {name}")
                if hashlib.sha256(archive.read(name)).hexdigest() != expected["sha256"]:
                    raise R1CheckoutError(f"source archive identity mismatch: {name}")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise R1CheckoutError("R1 source archive is missing or invalid") from exc
    return {
        "source_manifest_sha256": _file_identity(manifest_path)["sha256"],
        "source_file_count": len(entries),
        "required_execution_source_count": len(context.required_sources),
        "certified": True,
        "scope": "captured bytes match the frozen source snapshot; approval is separate",
    }


__all__ = ["R1CheckoutError", "R1CheckoutContext", "git_environment", "require_r1_checkout",
           "current_execution_context", "record_frozen_source", "validate_source_coverage",
           "REQUIRED_SOURCE_ANCHORS", "source_content_digest"]

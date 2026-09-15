"""Bind formal R1 execution to the actual, tracked SolarLab worktree."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


PROJECT_RELATIVE_PATH = "perovskite-sim"
STUDY_INPUT_RELATIVE_PATH = "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
_CHECKOUT_MODULE = "perovskite_sim/experiments/one_dimensional_mechanism_r1_checkout.py"
_RUNNER = "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
_ANCHORS = (
    _CHECKOUT_MODULE,
    _RUNNER,
    "scripts/run_one_dimensional_mechanism_r1.py",
    "scripts/run_one_dimensional_mechanism_r0.py",
    STUDY_INPUT_RELATIVE_PATH,
    "docs/OneDimensionalMechanismR1DynamicsV1.md",
    "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
)


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

    @property
    def study_input(self):
        return self.project / STUDY_INPUT_RELATIVE_PATH

    def to_dict(self):
        return {
            "repository_root": str(self.root),
            "project_relative_path": PROJECT_RELATIVE_PATH,
            "observed_commit": self.commit,
            "dirty": dict(self.dirty),
            "tracked_path_count": len(self.tracked_paths),
            "required_sources": {name: dict(entry) for name, entry in self.required_sources.items()},
            "scope": "tracked_execution_checkout; commit approval is checked separately",
        }


def git_environment():
    """Use filesystem checkout discovery for every formal Git operation."""
    return {
        key: value for key, value in os.environ.items()
        if key not in (
            "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
            "GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_PREFIX",
        ) and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }


def _git(directory, *arguments):
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True, check=False, env=git_environment(),
        )
    except OSError as exc:
        raise R1CheckoutError("R1 research execution requires Git and a source checkout") from exc
    if completed.returncode:
        raise R1CheckoutError(
            "R1 research execution requires a Git checkout with its tracked study input; "
            "package-only and wheel installations are not formal execution contexts"
        )
    return completed.stdout


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


def require_r1_checkout(*, project=None, runner=None):
    """Validate source location/tracking, without treating HEAD as approved.

    The root comes from Git's actual worktree discovery, not from a copied
    package's parent directory. Development changes may be present: their
    bytes and dirty flags are recorded, never relabelled as a clean commit.
    """
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
    tracked = {
        value for value in _git(root, "ls-files", "--cached", "--full-name", "-z").decode().split("\0")
        if value
    }
    package = expected_project / "perovskite_sim"
    package_sources = list(package.rglob("*.py"))
    if not package_sources:
        raise R1CheckoutError("R1 checkout has no package source coverage")
    required_paths = set(package_sources) | {expected_project / name for name in _ANCHORS}
    required = {}
    for path in sorted(required_paths):
        relative = path.relative_to(root).as_posix()
        if relative not in tracked or not path.is_file():
            raise R1CheckoutError(f"required R1 execution source is not a tracked file: {relative}")
        if path.is_symlink() or not path.resolve().is_relative_to(expected_project):
            raise R1CheckoutError(f"R1 execution source escapes its canonical checkout path: {relative}")
        required[relative] = _file_identity(path)
    _require_import_origins(expected_project, tracked)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal").decode().splitlines()
    dirty = {
        "staged": any(line[:1] not in (" ", "?") for line in status),
        "unstaged": any(line[1:2] not in (" ", "?") for line in status),
        "untracked": any(line.startswith("?? ") for line in status),
    }
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    return R1CheckoutContext(root, expected_project, commit, tuple(sorted(tracked)), required, dirty)


def validate_source_coverage(output, context):
    """Match nonempty captured evidence to every actual required source file."""
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
                path = context.root / name
                if _file_identity(path) != expected:
                    raise R1CheckoutError(f"execution source changed after checkout capture: {name}")
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
        "scope": "captured bytes match the tracked execution checkout; approval is separate",
    }


__all__ = ["R1CheckoutError", "R1CheckoutContext", "git_environment", "require_r1_checkout", "validate_source_coverage"]

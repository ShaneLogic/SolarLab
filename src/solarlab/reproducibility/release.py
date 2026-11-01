"""Immutable release content and installed-source verification.

The build authority supplies a frozen source directory and its actual source
commit. This module does not infer that association from maintenance HEAD.
Generated schema files are explicitly distinguished from source-commit files.
Verification needs no Git checkout and grants no scientific qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from solarlab.config.identity import BuildSource, RuntimeEnvironment
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.sources import relative_path
from solarlab.reproducibility.job_provenance import JobProvenance, capture_job_provenance, compare_provenance

__all__ = ["ReleaseFile", "ReleaseBundle", "snapshot_release_bundle", "write_release_bundle",
           "read_release_bundle", "verify_release_installation"]

_PACKAGES = ("solarlab", "solarlab_research", "solarlab_server")
_ROLES = {"package", "validator", "contract", "resource", "dependency_lock", "generated_schema", "build_configuration"}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _commit(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise ValueError("release requires an explicit immutable build-source commit")


def _files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()
        and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"} and path.name != ".DS_Store"))


def _contained(root: Path, name: str) -> Path:
    path = root / relative_path(name)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(f"release path escapes its explicit root: {name}")
    return path


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    source_path: str
    path: str
    role: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        relative_path(self.source_path)
        relative_path(self.path)
        if self.role not in _ROLES or type(self.content) is not bytes or len(self.content) > 2**24:
            raise ValueError("release requires a known content role and bounded immutable bytes")
        prefix = next((value for value in ("src/", "perovskite-sim/") if self.source_path.startswith(value)
                       and self.source_path[len(value):].split("/")[0] in (*_PACKAGES, "perovskite_sim")), None)
        expected = "site/" + self.source_path[len(prefix):] if prefix else "share/" + self.source_path
        if self.path != expected or self.role == "package" and prefix is None:
            raise ValueError("release source-to-install mapping is not the declared package/data layout")

    def record(self) -> dict[str, Any]:
        return dict(source_path=self.source_path, path=self.path, role=self.role, bytes=len(self.content),
                    sha256=hashlib.sha256(self.content).hexdigest(),
                    origin="generated_from_build_source" if self.role == "generated_schema" else "build_source")


@dataclass(frozen=True, slots=True)
class ReleaseBundle:
    """Complete versioned content snapshot, separate from release acceptance.

    Empty markers are retained in the bundle even though SourceDocument (used
    by existing scientific source identities) intentionally excludes them.
    ``transition`` keeps the one old engine; ``target`` excludes it. Neither
    profile silently changes an existing certificate's source algorithm.
    """
    build_source_commit: str
    profile: str
    files: tuple[ReleaseFile, ...]
    canonical_json: bytes = field(init=False, repr=False)
    sha256: str = field(init=False)
    can_execute: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _commit(self.build_source_commit)
        if self.profile not in {"transition", "target"}:
            raise ValueError("release profile must explicitly be transition or target")
        files = tuple(self.files)
        if not files or any(type(item) is not ReleaseFile for item in files):
            raise ValueError("release requires actual file content")
        if len({item.path for item in files}) != len(files) or len({item.source_path for item in files}) != len(files):
            raise ValueError("duplicate release path/source mapping")
        if sum(len(item.content) for item in files) > 2**26:
            raise ValueError("release content exceeds the bounded 64 MiB contract")
        by_path = {item.path: item for item in files}
        required = {f"site/{package}/{leaf}" for package in _PACKAGES for leaf in ("__init__.py", "py.typed")}
        required.update(("share/uv.lock", "share/pyproject.toml", "site/solarlab/reproducibility/PublicAPI.json"))
        if not required <= set(by_path) or not _ROLES <= {item.role for item in files}:
            raise ValueError("release is missing packages, validators, contracts, resources, lock, schema or build configuration")
        if by_path["share/uv.lock"].role != "dependency_lock" or by_path["share/pyproject.toml"].role != "build_configuration":
            raise ValueError("release lock/build source roles do not match their actual files")
        has_legacy = any(item.path.startswith("site/perovskite_sim/") for item in files)
        if has_legacy != (self.profile == "transition"):
            raise ValueError("legacy engine content differs from the explicit release profile")
        files = tuple(sorted(files, key=lambda item: item.path))
        value = {"schema": "solarlab.release-bundle.v1", "build_source_commit": self.build_source_commit,
                 "profile": self.profile, "files": [item.record() for item in files],
                 "historical_source_algorithms": "unchanged", "scientific_qualification": False}
        content = _json(value)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "canonical_json", content)
        object.__setattr__(self, "sha256", hashlib.sha256(content).hexdigest())

    def export(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    def implementation(self) -> BuildSource:
        """Bind the full release, including schema and empty-file records.

        The generated manifest explicitly distinguishes source-commit files
        from derived schema. It is a new source record, never a replacement
        interpretation of any historical source identity.
        """
        return BuildSource(self.build_source_commit, (
            SourceDocument("release:content-manifest.v1", self.canonical_json),
            *(SourceDocument(item.source_path, item.content) for item in self.files if item.content),
        ))

    def job_provenance(self, inputs: dict[str, Any], *, operator: str,
                       assembler_identity: str | None = None,
                       analysis_identity: str | None = None) -> JobProvenance:
        """Admission from this trusted bundle; full release file pins stay intact.

        The caller still verifies the installed bundle before executing. This
        record is provenance only and never grants scientific qualification.
        """
        lock = next(item.content for item in self.files if item.path == "share/uv.lock")
        return capture_job_provenance(inputs, dirty_policy="refuse", operator=operator,
            git_commit=self.build_source_commit, source_files=tuple((item.source_path, item.content) for item in self.files),
            environment_lock=lock, source_attested=True, assembler_identity=assembler_identity,
            analysis_identity=analysis_identity)

    def verify_job_provenance(
        self, producer: dict[str, Any], current: JobProvenance, *, installation_root: Path,
        expected_bundle_sha256: str, module_origins: tuple[tuple[str, str], ...],
        required_modules: tuple[str, ...], environment: RuntimeEnvironment,
    ) -> dict[str, Any]:
        """Strict prerequisite checks; no scientific qualification is awarded.

        The trusted release launcher supplies module origins and the environment
        from its existing locked execution contract. Matching metadata alone is
        insufficient: retain the full installed-file and actual environment checks.
        """
        from solarlab.experiments.worker import observe_environment
        installation = verify_release_installation(self, installation_root,
            expected_sha256=expected_bundle_sha256, module_origins=module_origins,
            required_modules=required_modules)
        observed = observe_environment(tuple(name for name, _ in environment.packages),
                                       tuple(name for name, _ in environment.settings))
        if observed != environment:
            raise ValueError("strict release environment differs from its admitted locked contract")
        value = current.export()
        expected = self.job_provenance({}, operator=value["assembler"]["operator"],
            assembler_identity=value["assembler"]["identity"], analysis_identity=value["assembler"]["analysis_identity"])
        compare_provenance(expected.envelope(), current, strict_same_source=True)
        return {**compare_provenance(producer, current, strict_same_source=True),
                "installation": installation, "environment_admitted": True,
                "scientific_qualification": False}


def snapshot_release_bundle(
    source_root: Path, *, build_source_commit: str, profile: str,
    required_files: tuple[tuple[str, str], ...],
) -> ReleaseBundle:
    """Read all first-party package bytes and the caller's required data closure.

    Use a frozen directory exported from build_source_commit, not an editable
    checkout labelled with HEAD. The build authority must bind that export and
    generate schema with its own source. This helper never edits the source.
    """
    if not source_root.is_absolute():
        raise ValueError("an absolute frozen source root is required")
    root = source_root.resolve(strict=True)
    required = dict(required_files)
    if len(required) != len(required_files):
        raise ValueError("duplicate required release file")
    roots = [(f"src/{name}", name) for name in _PACKAGES]
    if profile == "transition":
        roots.append(("perovskite-sim/perovskite_sim", "perovskite_sim"))
    values = {}
    for prefix, package in roots:
        folder = root / prefix
        if not folder.is_dir():
            raise ValueError(f"missing complete package source: {prefix}")
        for path in _files(folder):
            source = path.relative_to(root).as_posix()
            target = f"site/{package}/" + path.relative_to(folder).as_posix()
            values[source] = ReleaseFile(source, target, required.pop(source, "package"), _contained(root, source).read_bytes())
    for source, role in required.items():
        values[source] = ReleaseFile(source, "share/" + relative_path(source), role, _contained(root, source).read_bytes())
    return ReleaseBundle(build_source_commit, profile, tuple(values.values()))


def write_release_bundle(bundle: ReleaseBundle, destination: Path) -> None:
    """Materialize immutable content into a fresh directory; no installation."""
    if not destination.is_absolute():
        raise ValueError("an absolute new release directory is required")
    destination.mkdir(parents=True, exist_ok=False)
    for item in bundle.files:
        path = destination / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.content)
    (destination / "ReleaseBundle.json").write_bytes(bundle.canonical_json)


def read_release_bundle(manifest: SourceDocument, root: Path, *, expected_sha256: str) -> ReleaseBundle:
    """Read exactly the pinned manifest/content without discovering a checkout."""
    if manifest.sha256 != expected_sha256 or not root.is_absolute():
        raise ValueError("release manifest/root differs from the expected binding")
    value = json.loads(manifest.content)
    if not isinstance(value, dict) or set(value) != {"schema", "build_source_commit", "profile", "files", "historical_source_algorithms", "scientific_qualification"}:
        raise ValueError("invalid release manifest schema")
    root = root.resolve(strict=True)
    files = tuple(ReleaseFile(item["source_path"], item["path"], item["role"], _contained(root, item["path"]).read_bytes()) for item in value["files"])
    result = ReleaseBundle(value["build_source_commit"], value["profile"], files)
    if result.canonical_json != manifest.content:
        raise ValueError("release source content/schema differs from the complete pinned manifest")
    return result


def verify_release_installation(
    bundle: ReleaseBundle, root: Path, *, expected_sha256: str,
    module_origins: tuple[tuple[str, str], ...], required_modules: tuple[str, ...],
    current_repository_head: str | None = None,
) -> dict[str, Any]:
    """Verify actual installed bytes and the isolated process's module origins.

    Collect *all* first-party module __file__ values in the isolated process,
    after importing the required entry/validator modules. That trusted launcher
    also fixes the expected digest and required modules; payload claims cannot
    choose them. Dependencies are pinned by uv.lock and runtime identity, not
    claimed as first-party files by this contract.
    """
    if type(bundle) is not ReleaseBundle or bundle.sha256 != expected_sha256 or not root.is_absolute():
        raise ValueError("installed release differs from the expected bundle/root")
    if current_repository_head is not None:
        _commit(current_repository_head)
    root = root.resolve(strict=True)
    for item in bundle.files:
        if _contained(root, item.path).read_bytes() != item.content:
            raise ValueError(f"installed release file changed: {item.path}")
    packages = (*_PACKAGES, "perovskite_sim")
    # Even bytecode is unbound executable content. The isolated launcher uses
    # -B and an installation with no compiled cache, rather than trusting it.
    actual = {path.relative_to(root).as_posix() for package in packages
              for path in (root / "site" / package).rglob("*") if path.is_file()}
    expected = {item.path for item in bundle.files if item.path.startswith("site/")}
    if actual != expected:
        raise ValueError("installed package inventory has missing or unbound content")
    origins = dict(module_origins)
    if not required_modules or len(origins) != len(module_origins) or not set(required_modules) <= set(origins):
        raise ValueError("isolated installation must report every required module origin")
    for name, origin in origins.items():
        if name.split(".")[0] not in packages:
            raise ValueError("only first-party module origins belong to this verification")
        base = "site/" + name.replace(".", "/")
        candidates = {base + ".py", base + "/__init__.py"} & expected
        path = Path(origin)
        if not path.is_absolute() or not any(path == root / item and path.resolve(strict=True) == root / item for item in candidates):
            raise ValueError(f"module is imported from an editable/unbound location: {name}")
    return dict(schema="solarlab.release-installation-check.v1", bundle_sha256=bundle.sha256,
                build_source_commit=bundle.build_source_commit, current_repository_head=current_repository_head,
                files_verified=len(bundle.files), module_origins=dict(sorted(origins.items())),
                source_installation_verified=True, scientific_qualification=False)

"""Development provenance distinguishes checkout base, executed bytes and tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import _commit_bytes, _commit_tree, _git


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_record(path):
    path = Path(path).resolve(strict=True)
    root = Path(_git(path.parent, "rev-parse", "--show-toplevel").decode().strip())
    base = _git(root, "rev-parse", "HEAD").decode().strip()
    relative = path.relative_to(root).as_posix()
    committed = _commit_bytes(root, _commit_tree(root, base))
    raw = path.read_bytes()
    return {"path": str(path), "repository": str(root), "relative_path": relative,
            "base_commit": base, "sha256": _sha(raw), "bytes": len(raw),
            "matches_base_commit": committed.get(relative) == raw}


def capture_run_source(project, *, test_paths=()):
    """Snapshot exact tracked/source/test bytes; never label a dirty tree a commit."""
    project = Path(project).resolve(strict=True)
    root = Path(_git(project, "rev-parse", "--show-toplevel").decode().strip())
    base = _git(root, "rev-parse", "HEAD").decode().strip()
    committed = _commit_bytes(root, _commit_tree(root, base))
    tracked = {name for name in _git(root, "ls-files", "--cached", "--full-name", "-z").decode().split("\0") if name}
    package = project / "perovskite_sim"
    names = tracked | {path.relative_to(root).as_posix() for path in package.rglob("*.py")}
    current = {name: (root / name).read_bytes() for name in names
               if (root / name).is_file() and not (root / name).is_symlink()}
    hashes = {name: _sha(raw) for name, raw in sorted(current.items())}
    mismatches = sorted(name for name in set(current) | set(committed) if current.get(name) != committed.get(name))
    return {"schema": "R1DevelopmentSourceRecordV1", "repository": str(root), "project": str(project),
            "base_commit": base, "source_matches_base_commit": not mismatches,
            "executed_source_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
            "source_file_hashes": hashes, "base_content_mismatches": mismatches,
            "untracked_execution_sources": sorted(set(current) - tracked),
            "working_tree_patch": _git(root, "diff", "--binary", "HEAD", "--", ".").decode(),
            "test_sources": [_file_record(path) for path in test_paths],
            "scope": "development_observation; not_controlled_execution_or_scientific_approval"}


def compare_run_snapshots(before, after):
    """During-run stability and correspondence to a Git commit are separate."""
    return {"source_files_unchanged_during_run": before["source_file_hashes"] == after["source_file_hashes"],
            "test_files_unchanged_during_run": before["test_sources"] == after["test_sources"],
            "base_commit_unchanged_during_run": before["base_commit"] == after["base_commit"],
            "source_matched_base_commit_before_run": before["source_matches_base_commit"],
            "source_matched_base_commit_after_run": after["source_matches_base_commit"]}

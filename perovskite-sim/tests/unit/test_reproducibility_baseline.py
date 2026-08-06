from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest
import yaml

from perovskite_sim.reproducibility import verify_baseline


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=text,
    )


def test_baseline_reverse_recovers_dataless_base_and_target_blobs(tmp_path: Path):
    repo = tmp_path / "repo"
    project = repo / "project"
    project.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Baseline Test")

    frozen = project / "frozen.txt"
    frozen.write_text("base content\n", encoding="utf-8")
    _git(repo, "add", "project/frozen.txt")
    _git(repo, "commit", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    base_object = _git(
        repo, "rev-parse", f"{base_commit}:project/frozen.txt",
    ).stdout.strip()

    frozen.write_text("frozen P0 content\n", encoding="utf-8")
    target_object = _git(
        repo, "hash-object", "-w", "project/frozen.txt",
    ).stdout.strip()
    patch_data = _git(
        repo,
        "diff",
        "--full-index",
        base_commit,
        "--",
        "project/frozen.txt",
        text=False,
    ).stdout

    baseline = project / "reproducibility/baselines/test-baseline"
    baseline.mkdir(parents=True)
    patch_path = baseline / "p0.patch"
    patch_path.write_bytes(patch_data)
    target_data = frozen.read_bytes()
    snapshots = baseline / "target_blobs"
    snapshots.mkdir()
    snapshot_path = snapshots / f"{target_object}.blob"
    snapshot_path.write_bytes(target_data)
    manifest = {
        "schema_version": 1,
        "baseline_id": "test-baseline",
        "base_commit": base_commit,
        "patch": {
            "path": "reproducibility/baselines/test-baseline/p0.patch",
            "sha256": hashlib.sha256(patch_data).hexdigest(),
            "size_bytes": len(patch_data),
        },
        "target_blob_snapshots": [
            {
                "object_id": target_object,
                "path": (
                    "reproducibility/baselines/test-baseline/target_blobs/"
                    f"{target_object}.blob"
                ),
                "sha256": hashlib.sha256(target_data).hexdigest(),
            }
        ],
        "files": [
            {
                "path": "frozen.txt",
                "sha256": hashlib.sha256(target_data).hexdigest(),
            }
        ],
    }
    (baseline / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    assert target_object in patch_data.decode("utf-8")
    loose_objects = [
        repo / ".git/objects" / object_id[:2] / object_id[2:]
        for object_id in (base_object, target_object)
    ]
    for loose_object in loose_objects:
        compressed_size = loose_object.stat().st_size
        loose_object.unlink()
        with loose_object.open("wb") as stream:
            stream.truncate(compressed_size)
    if any(loose_object.stat().st_blocks != 0 for loose_object in loose_objects):
        pytest.skip("test filesystem did not create a sparse placeholder")

    report = verify_baseline("test-baseline", project)

    assert report["reconstruction_verified"] is True
    assert report["checked_files"] == 1
    assert report["reverse_recovered_files"] == 1
    assert report["target_snapshot_files"] == 1

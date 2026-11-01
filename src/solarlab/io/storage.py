"""Local active-storage guard; completed archives have a separate writer."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def require_local_storage(root: Path) -> None:
    """Check the actual mount and reject known managed cloud-sync locations.

    Arbitrary third-party synchronization cannot be discovered here; callers
    must still choose a private unsynchronized root.
    """
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("active store requires a canonical absolute local path")
    if any(part in str(root).casefold() + "/" for part in ("/library/mobile documents/", "/library/cloudstorage/", "/dropbox/", "/onedrive/")):
        raise ValueError("active database and WAL cannot use a cloud-sync directory")
    mounts: list[tuple[Path, bool]] = []
    if sys.platform == "darwin":
        output = subprocess.run(["/sbin/mount"], capture_output=True, text=True, check=True, timeout=5).stdout
        for line in output.splitlines():
            match = re.fullmatch(r".+ on (.+) \(([^)]+)\)", line)
            if match:
                mounts.append((Path(_unescape(match[1])), "local" in match[2].split(", ")))
    elif sys.platform.startswith("linux"):
        local = {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "ramfs", "overlay", "zfs", "f2fs"}
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            split = fields.index("-")
            mounts.append((Path(_unescape(fields[4])), fields[split + 1] in local))
    else:
        raise NotImplementedError("local WAL storage currently supports macOS and Linux")
    matches = [(len(path.parts), is_local) for path, is_local in mounts if root.is_relative_to(path)]
    depth = max((depth for depth, _ in matches), default=-1)
    if not matches or not all(local for size, local in matches if size == depth):
        raise ValueError("active database and WAL require a verified local filesystem")

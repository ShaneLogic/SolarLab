"""Input checks shared by the generation modules; they never alter a value."""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "array_digest",
    "require_cell_faces",
    "require_node_coordinates",
]


def require_node_coordinates(x_nodes_m: np.ndarray) -> np.ndarray:
    """Finite, strictly increasing 1D node coordinates with at least two nodes.

    A single node has no control volume, so it has no photon-conserving
    generation either; it is rejected instead of point-sampled.
    """

    x = np.asarray(x_nodes_m, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("a control-volume generation needs at least two nodes")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("node coordinates must be finite and strictly increasing")
    return x


def require_cell_faces(faces_m: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces_m, dtype=float)
    if faces.ndim != 1 or faces.shape[0] < 2:
        raise ValueError(f"faces must be 1-D with at least 2 entries, got shape {faces.shape}")
    if not np.all(np.isfinite(faces)) or np.any(np.diff(faces) <= 0.0):
        raise ValueError("cell faces must be finite and strictly increasing")
    return faces


def array_digest(*arrays: np.ndarray) -> str:
    """SHA-256 of little-endian float64 bytes and shapes, in argument order."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()

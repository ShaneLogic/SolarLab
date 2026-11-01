"""Explicitly supplied fixed generation arrays (the existing tandem path).

A caller that already owns a cell-averaged generation profile on the
electrical nodes (e.g. the sub-cell split of a combined tandem TMM) injects
it verbatim. The array must be finite and have one value per node; its
provenance string is the historical ``sha256:`` digest of its shape and
little-endian float64 bytes, so protocol identities are unchanged. There is
deliberately no file or coordinate-table loader here: none existed, and a
new profile-file entry is not a migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from solarlab.physics.generation.profile import (
    ControlVolumes,
    GenerationProfile,
    control_volume_digest,
    require_control_volumes,
)

__all__ = ["BASELINE_GENERATION_REFERENCE", "FixedGenerationProfile", "fixed_generation_reference"]

#: Provenance label of a run that keeps the stack's own optical generation.
BASELINE_GENERATION_REFERENCE = "stack_baseline_generation"


def fixed_generation_reference(generation: np.ndarray | None) -> str:
    """Historical provenance string of a supplied generation array (or its absence)."""

    if generation is None:
        return BASELINE_GENERATION_REFERENCE
    values = np.ascontiguousarray(np.asarray(generation, dtype="<f8"))
    if np.any(~np.isfinite(values)):
        raise ValueError("fixed_generation must contain only finite values")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class FixedGenerationProfile:
    """A supplied cell-averaged generation [m^-3 s^-1] on ``node_count`` nodes."""

    values_m3_s: np.ndarray
    node_count: int
    source_reference: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.node_count) is not int or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        values = np.array(self.values_m3_s, dtype=float, copy=True)
        if values.shape != (self.node_count,):
            raise ValueError(
                f"fixed_generation shape {values.shape} != expected ({self.node_count},)"
            )
        reference = fixed_generation_reference(values)
        values.setflags(write=False)
        object.__setattr__(self, "values_m3_s", values)
        object.__setattr__(self, "source_reference", reference)

    def generation_profile(self, control_volumes: ControlVolumes) -> GenerationProfile:
        """The supplied source verbatim; its cell photons follow from the measures."""

        _, _, measures = require_control_volumes(control_volumes)
        if measures.shape != (self.node_count,):
            raise ValueError("control volumes do not match the supplied generation's node count")
        return GenerationProfile(
            method="fixed",
            geometry_name=str(control_volumes.name),
            source_m3_s=self.values_m3_s,
            absorbed_m2_s=self.values_m3_s * measures,
            binding_sha256=self.source_reference,
            control_volume_sha256=control_volume_digest(control_volumes),
        )

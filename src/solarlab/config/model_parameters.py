"""Bind an actual resolved layer's full scalar values to model metadata."""

from __future__ import annotations

from solarlab.device.resolved import PreparedLayer
from solarlab.physics.registry import EvidenceRef, metadata_digest
from solarlab.physics.selection import ParameterSource

__all__ = ["layer_parameter_source"]


def layer_parameter_source(layer: PreparedLayer) -> ParameterSource:
    if not isinstance(layer, PreparedLayer):
        raise ValueError("a validated prepared layer is required")
    digest = metadata_digest(layer.to_mapping())
    return ParameterSource(layer.id, layer.parameters,
                           EvidenceRef("prepared.layer:" + layer.id, digest, "source"), "full_layer")

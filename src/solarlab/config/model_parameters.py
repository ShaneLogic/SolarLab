"""Bind an actual resolved layer's full scalar values to model metadata."""

from __future__ import annotations

from solarlab.device.resolved import PreparedLayer
from solarlab.physics.registry import EvidenceRef, metadata_digest
from solarlab.physics.selection import ParameterSource
from solarlab.config.structured_parameters import (
    device_topology, resolve_structured_edit, structured_parameter_source, structured_parameter_sources,
    update_structured_source,
)

__all__ = ["layer_parameter_source", "device_topology", "structured_parameter_source",
           "structured_parameter_sources", "update_structured_source", "resolve_structured_edit"]


def layer_parameter_source(layer: PreparedLayer) -> ParameterSource:
    if not isinstance(layer, PreparedLayer):
        raise ValueError("a validated prepared layer is required")
    digest = metadata_digest(layer.to_mapping())
    return ParameterSource(layer.id, layer.parameters,
                           EvidenceRef("prepared.layer:" + layer.id, digest, "source"), "full_layer")

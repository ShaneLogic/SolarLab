"""Source-bound model projections for identity preparation, never cache grants.

Both rules retain all physical model values, including temperature exponents.
The only reduced rule removes the documentary display label from a pinned
declaration/selection API. There is no caller-defined field exclusion list or
pointwise-value equivalence flag.
This module depends on model declarations, not devices, solvers or config.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelDefinition, ModelRef, metadata_value
from solarlab.physics.selection import SelectedModel

__all__ = ["SemanticProjection"]

_DISPLAY_LABEL_EVIDENCE = {
    "src/solarlab/physics/registry.py": "0a638307d4eec988d6920030750155970f4d7c9a2623f8de6b2b8a8f9e57ba1b",
    "src/solarlab/physics/selection.py": "dab0a6453ac64401a41f5a4028a49086074f063968d1fb22cafa3db0e297455c",
}


@dataclass(frozen=True, slots=True)
class SemanticProjection:
    """One model/version and actual immutable source documents.

    ``all_values_v1`` binds the complete declared model and effective values.
    ``without_display_label_v1`` removes only ModelDefinition.label, which the
    pinned selector never forwards to SelectedModel or uses to bind values.
    A changed proof source needs review, not an automatic fallback. Equality
    of a physical value at one temperature never permits omitting its exponent.
    """

    model: ModelRef
    rule: str
    sources: tuple[SourceDocument, ...]
    projection_evidence: tuple[SourceDocument, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelRef) or self.rule not in {"all_values_v1", "without_display_label_v1"}:
            raise ValueError("unknown model-owned semantic projection")
        sources = tuple(self.sources)
        if not sources or any(type(item) is not SourceDocument for item in sources):
            raise ValueError("semantic projection requires actual immutable source bytes")
        if len({item.id for item in sources}) != len(sources):
            raise ValueError("duplicate semantic source ID")
        object.__setattr__(self, "sources", tuple(sorted(sources, key=lambda item: item.id)))
        proof = tuple(self.projection_evidence)
        if any(type(item) is not SourceDocument for item in proof) or len({item.id for item in proof}) != len(proof):
            raise ValueError("invalid/duplicate documentary projection evidence")
        if self.rule == "without_display_label_v1":
            if {item.id: item.sha256 for item in proof} != _DISPLAY_LABEL_EVIDENCE:
                raise ValueError("display-label projection requires the exact audited declaration and selector sources")
        elif proof:
            raise ValueError("lossless model projection has no documentary omission proof")
        object.__setattr__(self, "projection_evidence", tuple(sorted(proof, key=lambda item: item.id)))

    def _evidence(self, definition: ModelDefinition) -> list[dict[str, str]]:
        evidence = [item for item in definition.capability.evidence if item.kind == "source"]
        documents = {item.id: item for item in self.sources}
        if not evidence or set(documents) != {item.id.partition("#")[0] for item in evidence}:
            raise ValueError("model source evidence must exactly match supplied source documents")
        for item in evidence:
            if documents[item.id.partition("#")[0]].sha256 != item.sha256:
                raise ValueError("model semantic source bytes disagree with the registry evidence")
        return [{"id": item.id, "sha256": item.sha256} for item in self.sources]

    def project(
        self, definition: ModelDefinition, instance: SelectedModel,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Return a fresh declaration and the layer parameters it accounts for.

        The configuration producer retains every other effective layer value.
        Thus a narrow reduction here cannot silently drop unrelated parameters.
        """
        if definition.ref != self.model or instance.model != self.model:
            raise ValueError("model/projection semantic-version mismatch")
        sources = self._evidence(definition)
        declaration = metadata_value(definition)
        if instance.structured is None:
            values: Any = dict(instance.parameters)
            if set(values) != set(definition.parameters):
                raise ValueError("projection requires the complete declared parameter set")
            claimed = tuple(definition.parameters)
        else:
            data = instance.structured.data
            reference = json.loads(data.resolved_reference_json)
            if definition.structured_schema != data.schema:
                raise ValueError("structured projection schema mismatch")
            values = {
                "normalized_supplied": json.loads(data.normalized_json),
                "resolved": reference["projection"], "context": reference["context"],
            }
            claimed = ()
        if self.rule == "without_display_label_v1":
            declaration.pop("label")
        return {
            "model": declaration, "rule": self.rule, "sources": sources,
            "instance_id": instance.id, "scope": metadata_value(instance.scope),
            "owned_variables": metadata_value(instance.owned_variables),
            "read_variables": list(instance.read_variables),
            "effective_values": values,
            "projection_evidence": [{"id": item.id, "sha256": item.sha256} for item in self.projection_evidence],
        }, claimed

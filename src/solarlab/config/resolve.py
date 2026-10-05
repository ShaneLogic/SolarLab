"""Explicit template loading and immutable layer resolution.

No source path is guessed and no catalog is copied into the new core. Callers
provide the existing document bytes; current production entrypoints are intact.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from typing import Any

from solarlab.config.inputs import LayerInput
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.layer import ResolvedLayer, ResolvedMaterial, ValueOrigin
from solarlab.materials.library import LayerTemplate, MaterialLibrary
from solarlab.materials.parameters import LayerValuesInput

__all__ = ["load_template_library", "load_layer_input", "layer_input_mapping", "resolve_layer", "resolve_layers"]


def load_template_library(content: str | bytes, *, source_id: str) -> MaterialLibrary:
    """Validate the actual layer-template document and bind its exact bytes."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    document = load_yaml_mapping(payload)
    fields = {"role", "optical_material", "description", "source", "defaults"}
    templates = []
    for template_id, record in document.items():
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError(f"templates.{template_id}: expected exactly {sorted(fields)}")
        if not isinstance(record["defaults"], dict):
            raise ValueError(f"templates.{template_id}.defaults: expected a mapping")
        if "optical_material" in record["defaults"]:
            raise ValueError(f"templates.{template_id}: optical_material has two competing sources")
        values = {**record["defaults"], "optical_material": record["optical_material"]}
        try:
            normalized = LayerValuesInput.model_validate(values).normalized_items(complete=True)
            templates.append(LayerTemplate(
                template_id, record["role"], record["description"], record["source"], normalized,
            ))
        except ValueError as error:
            raise ValueError(f"templates.{template_id}.defaults: {error}") from error
    return MaterialLibrary(source_id, hashlib.sha256(payload).hexdigest(), tuple(templates))


def load_layer_input(content: str | bytes) -> LayerInput:
    return LayerInput.model_validate(load_yaml_mapping(content))


def layer_input_mapping(layer: LayerInput) -> dict[str, Any]:
    """A detached JSON-compatible editing document; omitted fields stay omitted."""
    checked = LayerInput.model_validate(layer)
    return checked.model_dump(mode="json", exclude_unset=True)


def resolve_layer(layer: LayerInput, library: MaterialLibrary) -> ResolvedLayer:
    """Apply explicit overrides; zero is a value and nullable optical null clears."""
    if not isinstance(layer, LayerInput) or not isinstance(library, MaterialLibrary):
        raise ValueError("resolve_layer requires LayerInput and MaterialLibrary")
    layer.overrides.normalized_items()  # Validate constructed/copy-bypassed instances.
    if set(layer.__dict__) - set(LayerInput.model_fields):
        raise ValueError("unknown layer input fields")
    layer = LayerInput.model_validate(layer)
    template = library.get(layer.template)
    overrides = dict(layer.overrides.normalized_items())
    values = {**template.values, **overrides}
    values = dict(LayerValuesInput.model_validate(values).normalized_items(complete=True))
    material = ResolvedMaterial(template.id, tuple(
        (name, value) for name, value in values.items() if name not in {"thickness", "N_A", "N_D"}
    ))
    origins = tuple(ValueOrigin(
        name, "override" if name in overrides else "template",
        layer.id if name in overrides else f"{library.source_id}#{template.id}",
    ) for name in values)
    return ResolvedLayer(
        layer.id, layer.name, template.role,
        values["thickness"], values["N_A"], values["N_D"],  # type: ignore[arg-type]
        material, library.source_id, library.source_sha256, library.content_sha256, origins,
    )


def resolve_layers(layers: Sequence[LayerInput], library: MaterialLibrary) -> tuple[ResolvedLayer, ...]:
    resolved = tuple(resolve_layer(layer, library) for layer in layers)
    if not resolved or len({layer.id for layer in resolved}) != len(resolved):
        raise ValueError("layers require nonempty, unique stable IDs; display names may repeat")
    return resolved

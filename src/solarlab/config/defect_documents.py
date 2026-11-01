"""Explicit data-only import of the existing metastable documents.

This is a preparation API for the programmatic document consumers. It does not
add metastable keys to the legacy device YAML, backend or frontend routes.
"""

from __future__ import annotations

from solarlab.config.device_import import _keys, _mapping, _sequence
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defects import MetastableDocumentInput, MetastablePreparationInput
from solarlab.materials.source import SourceDocument


def import_metastable_document(source: SourceDocument) -> MetastableDocumentInput:
    raw = load_yaml_mapping(source.content)
    fields = {"schema_version", "defect_model", "metastable_defects"}
    _keys(raw, fields, fields, "metastable_document")
    species = []
    for i, value in enumerate(_sequence(raw["metastable_defects"], "metastable_defects")):
        item = _mapping(value, f"metastable_defects[{i}]")
        species.append({"id": f"metastable_{i}", **item})
    names = [item.get("name") for item in species]
    if len(names) != len(set(names)):
        raise ValueError("legacy metastable species names must be unique")
    return MetastableDocumentInput.model_validate({**raw, "metastable_defects": species})


def import_metastable_preparation(source: SourceDocument) -> MetastablePreparationInput:
    return MetastablePreparationInput.model_validate(load_yaml_mapping(source.content))

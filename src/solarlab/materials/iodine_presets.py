"""Packaged immutable YAML presets for 1D iodine chemistry & aging studies.

Reads exclusively from ``importlib.resources`` under
``solarlab.materials.data.iodine_presets`` without importing higher
configuration, device, physics, or experiment packages.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml

from solarlab.materials.source import SourceDocument

__all__ = [
    "PACKAGED_IODINE_PRESET_NAMES",
    "list_packaged_iodine_preset_summaries",
    "load_packaged_iodine_preset_bytes",
    "load_packaged_iodine_preset_mapping",
    "load_packaged_iodine_preset_source",
]

_IODINE_PRESET_PACKAGE = "solarlab.materials.data.iodine_presets"

PACKAGED_IODINE_PRESET_NAMES: tuple[str, ...] = (
    "bitton_tessler_2023_blocking",
    "bitton_tessler_2023_permeable",
    "bitton_tessler_2023_outgassing",
    "bitton_tessler_2023_ag_corrosion",
    "bitton_tessler_2023_self_healing",
)


class _UniqueKeyYamlLoader(yaml.SafeLoader):
    """YAML SafeLoader rejecting duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyYamlLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in mapping:
            raise ValueError("packaged iodine YAML mapping keys must be unique strings")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyYamlLoader.add_constructor(
    "tag:yaml.org,2002:map", _construct_unique_mapping
)


def _normalize_preset_stem(name: str) -> str:
    stem = name.strip().removesuffix(".yaml").removesuffix(".yml")
    if stem not in PACKAGED_IODINE_PRESET_NAMES:
        raise KeyError(
            f"unknown packaged iodine preset {name!r}; "
            f"available: {list(PACKAGED_IODINE_PRESET_NAMES)}"
        )
    return stem


def load_packaged_iodine_preset_bytes(name: str) -> bytes:
    """Return raw UTF-8 YAML bytes for a packaged iodine deck preset."""
    stem = _normalize_preset_stem(name)
    filename = f"{stem}.yaml"
    node = resources.files(_IODINE_PRESET_PACKAGE).joinpath(filename)
    if not node.is_file():
        raise FileNotFoundError(f"missing packaged iodine preset asset: {filename}")
    return node.read_bytes()


def load_packaged_iodine_preset_source(name: str) -> SourceDocument:
    """Return a checksummed ``SourceDocument`` for a packaged iodine preset."""
    stem = _normalize_preset_stem(name)
    raw = load_packaged_iodine_preset_bytes(stem)
    return SourceDocument(
        f"solarlab/materials/data/iodine_presets/{stem}.yaml",
        raw,
    )


def load_packaged_iodine_preset_mapping(name: str) -> dict[str, Any]:
    """Parse and return a packaged iodine preset YAML document as a dict."""
    raw = load_packaged_iodine_preset_bytes(name)
    doc = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyYamlLoader)
    if not isinstance(doc, dict):
        raise ValueError(f"packaged iodine preset {name!r} must decode to a mapping")
    return doc


def list_packaged_iodine_preset_summaries() -> list[dict[str, Any]]:
    """Return metadata summaries and parsed decks for all packaged iodine presets."""
    items: list[dict[str, Any]] = []
    for stem in PACKAGED_IODINE_PRESET_NAMES:
        source = load_packaged_iodine_preset_source(stem)
        doc = load_packaged_iodine_preset_mapping(stem)
        items.append(
            {
                "id": stem,
                "name": str(doc.get("name", stem)),
                "description": str(doc.get("description", "")),
                "reference": str(doc.get("reference", "")),
                "source_id": source.id,
                "sha256": source.sha256,
                "deck": doc,
            }
        )
    return items

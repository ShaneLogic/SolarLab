"""Packaged immutable default catalog, optical resources, templates, and presets.

Reads solely from ``importlib.resources`` package assets under
``solarlab.materials.data``, requiring no git checkout or ``perovskite_sim``
directory when installed from the release wheel.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import resources
import json
from typing import TYPE_CHECKING, Any

import yaml

from solarlab.materials.library import LayerTemplate, MaterialLibrary
from solarlab.materials.parameters import LayerValuesInput
from solarlab.materials.source import SourceDocument

if TYPE_CHECKING:
    from solarlab.materials.resources import ResourceLibrary, ResourceTable

__all__ = [
    "PACKAGED_NK_MATERIALS",
    "PACKAGED_PRESET_NAMES",
    "load_packaged_am15g_table",
    "load_packaged_default_catalog",
    "load_packaged_default_catalog_mapping",
    "load_packaged_layer_templates_bytes",
    "load_packaged_nk_manifest_bytes",
    "load_packaged_nk_table",
    "load_packaged_preset_device_mapping",
    "load_packaged_preset_sources",
    "load_packaged_preset_v2_documents",
    "load_packaged_resource_library",
    "load_packaged_template_library",
]

_DATA_PACKAGE = "solarlab.materials.data"

_DEFAULT_SOURCE_IDS: tuple[str, ...] = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)

_MODEL_SOURCE_IDS: tuple[str, ...] = (
    "physics/cigs_optics.py",
    "models/tunneling_channels.py",
)

PACKAGED_NK_MATERIALS: tuple[str, ...] = (
    "Ag",
    "Au",
    "C60",
    "Cu",
    "FA_Cs_1p77",
    "FTO",
    "ITO",
    "MAPbI3",
    "NiOx",
    "PCBM",
    "PEDOT_PSS",
    "SnO2",
    "SnPb_1p22",
    "TiO2",
    "glass",
    "spiro_OMeTAD",
)

PACKAGED_PRESET_NAMES: tuple[str, ...] = (
    "classical_pin",
    "scaps_pvk_xy",
    "scaps_mirror_v2",
    "calado2016_ion_sweep",
    "calado2016_fig1f",
)

_SHIPPED_YAML_PRESETS: tuple[str, ...] = (
    "scaps_pvk_xy.yaml",
    "scaps_mirror_v2.yaml",
    "calado2016_ion_sweep.yaml",
    "calado2016_fig1f.yaml",
)


class _StrictYamlMappingLoader(yaml.SafeLoader):
    """YAML SafeLoader rejecting duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictYamlMappingLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in mapping:
            raise ValueError("packaged YAML mapping keys must be unique strings")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictYamlMappingLoader.add_constructor(
    "tag:yaml.org,2002:map", _construct_unique_mapping
)


def _read_data_bytes(*parts: str) -> bytes:
    node = resources.files(_DATA_PACKAGE)
    for part in parts:
        node = node.joinpath(part)
    if not node.is_file():
        joined = "/".join(parts)
        raise FileNotFoundError(f"missing packaged material data resource: {joined}")
    return node.read_bytes()


def load_packaged_default_catalog_mapping(profile: str = "full") -> dict[str, Any]:
    """Return the frozen ``solarlab.default-catalog.v1`` mapping for ``profile``.

    Supported profiles:
    - ``"full"``: complete 15-source catalog including ``complex_defaults`` and
      ``experiment_defaults`` (for ``solarlab_server`` and CLI preparation).
    - ``"matrix_v2"``: 11-source catalog matching ``ConfigBenchmarkMatrixV2``
      ``preparation_context``.
    - ``"device_only"``: 9-source base device/material catalog.
    """
    raw = json.loads(_read_data_bytes("default_catalog_v1.json").decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid packaged default catalog payload")
    if profile == "full":
        return raw
    if profile == "matrix_v2":
        keep = set(_DEFAULT_SOURCE_IDS) | set(_MODEL_SOURCE_IDS)
        filtered = {k: v for k, v in raw.items() if k != "experiment_defaults"}
        filtered["evidence"] = [pair for pair in raw["evidence"] if pair[0] in keep]
        return filtered
    if profile == "device_only":
        keep = set(_DEFAULT_SOURCE_IDS)
        filtered = {
            k: v
            for k, v in raw.items()
            if k not in {"complex_defaults", "experiment_defaults"}
        }
        filtered["evidence"] = [pair for pair in raw["evidence"] if pair[0] in keep]
        return filtered
    raise ValueError(
        f"unknown packaged default catalog profile {profile!r}; "
        "expected 'full', 'matrix_v2', or 'device_only'"
    )


def load_packaged_default_catalog(profile: str = "full") -> Any:
    """Instantiate an immutable ``DefaultCatalog`` from packaged data."""
    defaults_module = importlib.import_module("solarlab.device.defaults")
    return defaults_module.DefaultCatalog.from_mapping(
        load_packaged_default_catalog_mapping(profile=profile)
    )


def load_packaged_nk_table(
    material: str, *, source_id_prefix: str = "solarlab/materials/data"
) -> ResourceTable:
    """Load one packaged optical ``(wavelength, n, k)`` ``ResourceTable``."""
    from solarlab.materials.resources import ResourceTable

    if material not in PACKAGED_NK_MATERIALS:
        raise FileNotFoundError(f"unknown packaged optical material: {material!r}")
    prefix = source_id_prefix.rstrip("/")
    source_id = f"{prefix}/nk/{material}.csv" if prefix else f"nk/{material}.csv"
    content = _read_data_bytes("nk", f"{material}.csv")
    return ResourceTable(material, "nk", SourceDocument(source_id, content))


def load_packaged_am15g_table(
    *, source_id_prefix: str = "solarlab/materials/data"
) -> ResourceTable:
    """Load the packaged AM1.5G spectral photon-flux ``ResourceTable``."""
    from solarlab.materials.resources import ResourceTable

    prefix = source_id_prefix.rstrip("/")
    source_id = f"{prefix}/am15g.csv" if prefix else "am15g.csv"
    content = _read_data_bytes("am15g.csv")
    return ResourceTable("am15g", "spectrum", SourceDocument(source_id, content))


def load_packaged_resource_library(
    *,
    include_spectrum: bool = True,
    source_id_prefix: str = "solarlab/materials/data",
) -> ResourceLibrary:
    """Load the packaged optical ``n,k`` (and optional AM1.5G) ``ResourceLibrary``."""
    from solarlab.materials.resources import ResourceLibrary

    tables: list[ResourceTable] = [
        load_packaged_nk_table(name, source_id_prefix=source_id_prefix)
        for name in PACKAGED_NK_MATERIALS
    ]
    if include_spectrum:
        tables.append(load_packaged_am15g_table(source_id_prefix=source_id_prefix))
    return ResourceLibrary(tuple(tables))


def load_packaged_nk_manifest_bytes() -> bytes:
    """Return raw bytes of packaged ``nk/manifest.yaml``."""
    return _read_data_bytes("nk", "manifest.yaml")


def load_packaged_layer_templates_bytes() -> bytes:
    """Return raw bytes of packaged ``layer_templates.yaml``."""
    return _read_data_bytes("layer_templates.yaml")


def load_packaged_template_library(
    *, source_id: str = "solarlab/materials/data/layer_templates.yaml"
) -> MaterialLibrary:
    """Load and validate the packaged scalar ``MaterialLibrary``."""
    payload = load_packaged_layer_templates_bytes()
    document = yaml.load(payload.decode("utf-8"), Loader=_StrictYamlMappingLoader)
    if not isinstance(document, dict):
        raise ValueError("packaged layer_templates.yaml must contain a mapping")
    expected_fields = {
        "role",
        "optical_material",
        "description",
        "source",
        "defaults",
    }
    templates: list[LayerTemplate] = []
    for template_id, record in document.items():
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(
                f"templates.{template_id}: expected {sorted(expected_fields)}"
            )
        if not isinstance(record["defaults"], dict):
            raise ValueError(f"templates.{template_id}.defaults: expected a mapping")
        if "optical_material" in record["defaults"]:
            raise ValueError(
                f"templates.{template_id}: competing optical_material keys"
            )
        values = {**record["defaults"], "optical_material": record["optical_material"]}
        normalized = LayerValuesInput.model_validate(values).normalized_items(
            complete=True
        )
        templates.append(
            LayerTemplate(
                template_id,
                record["role"],
                record["description"],
                record["source"],
                normalized,
            )
        )
    return MaterialLibrary(
        source_id, hashlib.sha256(payload).hexdigest(), tuple(templates)
    )


def load_packaged_preset_sources() -> tuple[SourceDocument, ...]:
    """Return ``SourceDocument`` records for all packaged preset configurations."""
    docs: list[SourceDocument] = []
    for filename in _SHIPPED_YAML_PRESETS:
        docs.append(SourceDocument(filename, _read_data_bytes("presets", filename)))
    for stem in PACKAGED_PRESET_NAMES:
        dev_name = f"{stem}.device.json"
        docs.append(SourceDocument(dev_name, _read_data_bytes("presets", dev_name)))
    return tuple(docs)


def load_packaged_preset_v2_documents() -> dict[str, SourceDocument]:
    """Return migrated ``solarlab.configuration.v2`` documents keyed by preset stem."""
    result: dict[str, SourceDocument] = {}
    for filename in _SHIPPED_YAML_PRESETS:
        stem = filename.removesuffix(".yaml")
        v2_name = f"{stem}.v2.json"
        result[stem] = SourceDocument(v2_name, _read_data_bytes("presets", v2_name))
    return result


def load_packaged_preset_device_mapping(name: str) -> dict[str, Any]:
    """Load the V2 ``DeviceInput`` JSON mapping for a packaged preset."""
    stem = name.removesuffix(".yaml").removesuffix(".device.json").removesuffix(".json")
    if stem not in PACKAGED_PRESET_NAMES:
        raise KeyError(f"unknown packaged preset {name!r}")
    data = json.loads(_read_data_bytes("presets", f"{stem}.device.json").decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid packaged preset device mapping: {stem}")
    return data

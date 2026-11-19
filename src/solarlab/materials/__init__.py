"""Material and layer-template data, independent of numerical engines."""

from solarlab.materials.packaged import (
    PACKAGED_NK_MATERIALS,
    PACKAGED_PRESET_NAMES,
    load_packaged_am15g_table,
    load_packaged_default_catalog,
    load_packaged_default_catalog_mapping,
    load_packaged_layer_templates_bytes,
    load_packaged_nk_manifest_bytes,
    load_packaged_nk_table,
    load_packaged_preset_device_mapping,
    load_packaged_preset_sources,
    load_packaged_preset_v2_documents,
    load_packaged_resource_library,
    load_packaged_template_library,
)

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

"""One explicit legacy-to-v2 data migration and the existing reverse adapter.

The v2 reader accepts only the new envelope and resolves its typed input.
Original sources remain evidence, not a fallback input format. Neither a data
roundtrip nor choosing the v2 reader authorizes a scientific execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, Literal

from pydantic import ValidationError

from solarlab.config.behavior import _mapping
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_wire import LegacyWire, prepare_legacy_wire
from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

__all__ = ["CONFIGURATION_VERSION", "ConfigurationMigration", "migrate_configuration",
           "read_configuration_v2", "prepare_v2_legacy_wire"]

CONFIGURATION_VERSION = "solarlab.configuration.v2"


def _binding(source: SourceDocument) -> dict[str, str]:
    if type(source) is not SourceDocument:
        raise ValueError("explicit immutable source documents are required")
    return {"id": source.id, "sha256": source.sha256}


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(
    document: SourceDocument, defaults: DefaultCatalog, resources: ResourceLibrary, sources: tuple[SourceDocument, ...],
) -> tuple[DeviceInput | TandemInput, tuple[SourceDocument, ...], SourceDocument, dict[str, SourceDocument]]:
    _binding(document)
    if type(defaults) is not DefaultCatalog or type(resources) is not ResourceLibrary:
        raise ValueError("actual default and resource catalogs are required")
    # v2 files are JSON: keep numeric tokens as their actual int/float types.
    # The legacy YAML reader deliberately returns Decimal; routing this JSON
    # through it would turn some typed interface fields into decimal strings.
    data = _mapping(document)
    if data.get("schema_version") != CONFIGURATION_VERSION:
        raise ValueError("the formal reader accepts only solarlab.configuration.v2; use the migration tool for v1")
    if set(data) != {"schema_version", "input", "source", "sources", "references", "default_catalog_sha256", "resource_library_sha256"}:
        raise ValueError("unknown or missing v2 configuration field")
    if (data["default_catalog_sha256"] != defaults.content_sha256
            or data["resource_library_sha256"] != resources.content_sha256):
        raise ValueError("v2 default/resource content binding mismatch")
    trusted = {item.id: item for item in sources if type(item) is SourceDocument}
    if len(trusted) != len(sources):
        raise ValueError("v2 sources must have unique IDs and actual immutable bytes")

    def source(binding: Any) -> SourceDocument:
        if not isinstance(binding, dict) or set(binding) != {"id", "sha256"} or type(binding["id"]) is not str:
            raise ValueError("a v2 source binding requires exactly id and sha256")
        actual = trusted.get(binding["id"])
        if actual is None or binding != _binding(actual):
            raise ValueError("v2 original-source content binding mismatch")
        return actual

    if not isinstance(data["sources"], list) or not isinstance(data["references"], dict):
        raise ValueError("v2 sources/references have invalid shapes")
    original = source(data["source"])
    selected = tuple(source(item) for item in data["sources"])
    references = {name: source(item) for name, item in data["references"].items()}
    raw = data["input"]
    if not isinstance(raw, dict):
        raise ValueError("v2 input must be a typed declaration object")
    declaration: DeviceInput | TandemInput
    if raw.get("schema_version") == "solarlab.device-preparation.v1":
        declaration = DeviceInput.model_validate(raw)
        if references or selected != (original,):
            raise ValueError("a device v2 file requires exactly its original source and no references")
    elif raw.get("schema_version") == "solarlab.tandem-preparation.v1":
        declaration = TandemInput.model_validate(raw)
        names = (declaration.top_cell_reference, declaration.bottom_cell_reference)
        if set(references) != set(names) or selected != tuple(dict.fromkeys((original, *(references[name] for name in names)))):
            raise ValueError("tandem v2 references must cover the exact ordered original subcells")
    else:
        raise ValueError("unsupported v2 typed input schema")
    return declaration, selected, original, references


def read_configuration_v2(
    document: SourceDocument, *, defaults: DefaultCatalog, resources: ResourceLibrary,
    sources: tuple[SourceDocument, ...],
) -> PreparedDevice | PreparedTandem:
    """Read the formal v2 declaration with exact caller-supplied provenance.

    The nested preparation schemas keep their existing versions. The outer
    configuration version does not rename their units, defaults or semantics.
    """
    declaration, selected, _, _ = _decode(document, defaults, resources, sources)
    if isinstance(declaration, DeviceInput):
        return resolve_device(declaration, defaults, resources, sources=selected)
    return resolve_tandem(declaration, defaults, resources, sources=selected)


def prepare_v2_legacy_wire(
    document: SourceDocument, *, defaults: DefaultCatalog, resources: ResourceLibrary,
    sources: tuple[SourceDocument, ...],
) -> LegacyWire:
    """Use the single existing reverse adapter; never implement a second one."""
    declaration, _, original, references = _decode(document, defaults, resources, sources)
    read_configuration_v2(document, defaults=defaults, resources=resources, sources=sources)
    return prepare_legacy_wire(declaration, source=original, defaults=defaults, references=references)


@dataclass(frozen=True, slots=True)
class ConfigurationMigration:
    document: SourceDocument | None
    _report_json: bytes

    @property
    def supported(self) -> bool:
        return self.document is not None

    @property
    def can_execute(self) -> bool:
        return False

    def export(self) -> dict[str, Any]:
        return json.loads(self._report_json)


def migrate_configuration(
    source: SourceDocument, *, format: Literal["standard-device-v1", "scaps-device-v1", "tandem-v1"],
    input_id: str, defaults: DefaultCatalog, resources: ResourceLibrary,
    references: Mapping[str, SourceDocument] | None = None,
) -> ConfigurationMigration:
    """Dry-run one forward migration, with unchanged existing field readers.

    Failed or unsupported inputs have no proposed document. This function
    writes nothing; publishing the returned data is a distinct caller action.
    """
    report: dict[str, Any] = {"schema": "solarlab.configuration-migration-report.v1", "source": _binding(source),
        "format": format, "input_id": input_id, "supported": False, "can_execute": False,
        "evidence": "source_verified_preparation_only", "scientific_gates_awarded": [], "issues": []}
    try:
        if type(defaults) is not DefaultCatalog or type(resources) is not ResourceLibrary:
            raise ValueError("actual default and resource catalogs are required")
        supplied = dict(references or {})
        for value in supplied.values():
            _binding(value)
        input: DeviceInput | TandemInput
        prepared: PreparedDevice | PreparedTandem
        if format == "tandem-v1":
            input = import_tandem(source, id=input_id, references=supplied, defaults=defaults)
            names = (input.top_cell_reference, input.bottom_cell_reference)
            if set(supplied) != set(names):
                raise ValueError("supply exactly the declared tandem references")
            originals = tuple(dict.fromkeys((source, *(supplied[name] for name in names))))
            prepared = resolve_tandem(input, defaults, resources, sources=originals)
        elif format in {"standard-device-v1", "scaps-device-v1"}:
            if supplied:
                raise ValueError("a device migration has no external input references")
            reader = import_standard_device if format == "standard-device-v1" else import_scaps_device
            input = reader(source, id=input_id, defaults=defaults)
            originals = (source,)
            prepared = resolve_device(input, defaults, resources, sources=originals)
        else:
            raise ValueError("an explicit supported legacy format is required; no schema guessing")
        document = SourceDocument(input_id + ".configuration.v2", _encode({
            "schema_version": CONFIGURATION_VERSION, "input": json.loads(prepared.input_json),
            "source": _binding(source), "sources": [_binding(item) for item in originals],
            "references": {name: _binding(item) for name, item in supplied.items()},
            "default_catalog_sha256": defaults.content_sha256, "resource_library_sha256": resources.content_sha256,
        }))
        reopened = read_configuration_v2(document, defaults=defaults, resources=resources, sources=originals)
        if (reopened.input_json != prepared.input_json or reopened.to_mapping() != prepared.to_mapping()
                or reopened.content_sha256 != prepared.content_sha256):
            raise ValueError("v2 declaration did not retain the complete prepared data")
        wire = prepare_v2_legacy_wire(document, defaults=defaults, resources=resources, sources=originals)
        report["reverse_adapter"] = wire.export()
        if not wire.supported or wire.document != source or dict(wire.references) != supplied:
            raise ValueError("the existing reverse adapter cannot retain all original bytes; keep the old path")
        report.update(supported=True, proposed=_binding(document), prepared_content_sha256=prepared.content_sha256,
                      input_declaration=json.loads(prepared.input_json), diagnostics=prepared.to_mapping().get("diagnostics", []),
                      capability_gaps=prepared.to_mapping().get("capability_gaps", []),
                      preparation_roundtrip_exact=True, original_wire_bytes_exact=True,
                      runtime_behavior="existing source-bound driver context still required; v2 data is not execution admission")
        return ConfigurationMigration(document, _encode(report))
    except ValueError as error:
        if isinstance(error, ValidationError):
            report["issues"] = error.errors(include_url=False, include_context=False, include_input=False)
        else:
            report["issues"] = [{"loc": [], "type": "configuration_migration", "msg": str(error)}]
        return ConfigurationMigration(None, _encode(report))

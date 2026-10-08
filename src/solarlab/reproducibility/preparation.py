"""Exercise declared configuration mappings through existing public readers.

The result is a prepared declaration, never a migrated scientific executor or
an admission decision. Missing legacy input contracts remain explicit errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import posixpath

from pydantic import ValidationError

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.registry import ConfigurationPreparation, MigrationRegistry, PreparationIssue
from solarlab.reproducibility.sources import SourceSet, relative_path


def read_preparation_context(registry: MigrationRegistry) -> tuple[DefaultCatalog, ResourceLibrary]:
    """Use explicitly bound source data; no legacy module import or discovery."""
    context = registry.spec.preparation_context
    if context is None:
        raise ValueError("use the configuration matrix's explicit preparation context")
    catalog = read_legacy_default_catalog(
        tuple(registry.sources.document(id) for id in context.default_sources),
        model_sources=tuple(registry.sources.document(id) for id in context.model_sources))
    resources = ResourceLibrary(tuple(ResourceTable(PurePosixPath(registry.sources.reference(id).path).stem, "nk", registry.sources.document(id))
                                      for id in context.resource_sources))
    if catalog.content_sha256 != context.default_catalog_sha256 or resources.content_sha256 != context.resource_library_sha256:
        raise ValueError("prepared default/resource context identity mismatch")
    return catalog, resources


def prepare_original(source: SourceDocument, *, input_id: str, schema: str,
                     sources: SourceSet, defaults: DefaultCatalog,
                     resources: ResourceLibrary) -> PreparedDevice | PreparedTandem:
    """Prepare one verified V1 input without changing its unsupported fields."""
    if sources.document(source.id) != source:
        raise ValueError("configuration source is not in the verified bundle")
    if schema == "standard-device-v1":
        return resolve_device(import_standard_device(source, id=input_id, defaults=defaults), defaults, resources, sources=(source,))
    if schema == "scaps-device-v1":
        return resolve_device(import_scaps_device(source, id=input_id, defaults=defaults), defaults, resources, sources=(source,))
    if schema != "tandem-v1":
        raise ValueError("unsupported registered configuration schema")
    original = load_yaml_mapping(source.content)
    parent = PurePosixPath(sources.reference(source.id).path).parent
    references = {}
    for field in ("top_cell", "bottom_cell"):
        name = original["tandem"][field]
        if not isinstance(name, str) or PurePosixPath(name).is_absolute():
            raise ValueError("tandem reference must be a supplied relative document path")
        path = relative_path(posixpath.normpath(str(parent / name)))
        references[name] = sources.at_path(path)
    value = import_tandem(source, id=input_id, references=references, defaults=defaults)
    return resolve_tandem(value, defaults, resources, sources=(source, *references.values()))


def preparation_issues(error: ValueError) -> tuple[PreparationIssue, ...]:
    if isinstance(error, ValidationError):
        return tuple(PreparationIssue(path=tuple(item["loc"]), type=item["type"], message=item["msg"])
                     for item in error.errors(include_url=False, include_context=False, include_input=False))
    return (PreparationIssue(path=(), type="configuration_resolution", message=str(error)),)


@dataclass(frozen=True, slots=True)
class PreparationResult:
    declaration: ConfigurationPreparation
    prepared: PreparedDevice | PreparedTandem | None

    @property
    def can_execute(self) -> bool:
        return False


def prepare_configuration(registry: MigrationRegistry, id: str, *, defaults: DefaultCatalog,
                          resources: ResourceLibrary) -> PreparationResult:
    entry = registry.entry(id)
    context = registry.spec.preparation_context
    expected = entry.preparation
    if entry.section != "configs" or context is None or expected is None:
        raise ValueError("only a mapped configuration has a public preparation path")
    if defaults.content_sha256 != context.default_catalog_sha256 or resources.content_sha256 != context.resource_library_sha256:
        raise ValueError("prepared default/resource context identity mismatch")
    row = registry.original(id)
    source = registry.sources.at_path(registry.spec.original_path_root + "/" + row["path"])
    prepared = None
    try:
        prepared = prepare_original(source, input_id=expected.input_id, schema=row["schema"], sources=registry.sources, defaults=defaults, resources=resources)
    except ValueError as error:
        actual = ConfigurationPreparation(status="blocked", input_id=expected.input_id, loader=expected.loader, resolver=expected.resolver,
                                          content_sha256=None, issues=preparation_issues(error))
    else:
        actual = ConfigurationPreparation(status="prepared_only", input_id=expected.input_id, loader=expected.loader, resolver=expected.resolver,
                                          content_sha256=prepared.content_sha256, issues=())
    if actual != expected:
        raise ValueError(f"registered preparation no longer matches its actual public reader: {id}")
    return PreparationResult(actual, prepared)

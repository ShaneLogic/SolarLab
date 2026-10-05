"""Public preparation resolution from explicit input, catalog and resource data."""

from __future__ import annotations

import json

from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

__all__ = ["resolve_device", "resolve_tandem"]


def resolve_device(input: DeviceInput, defaults: DefaultCatalog, resources: ResourceLibrary,
                   *, sources: tuple[SourceDocument, ...] = ()) -> PreparedDevice:
    checked = DeviceInput.model_validate(input)
    document = json.dumps(checked.model_dump(mode="json", exclude_unset=True), sort_keys=True, allow_nan=False).encode()
    return PreparedDevice(document, defaults, resources, sources)


def resolve_tandem(input: TandemInput, defaults: DefaultCatalog, resources: ResourceLibrary,
                   *, sources: tuple[SourceDocument, ...] = ()) -> PreparedTandem:
    checked = TandemInput.model_validate(input)
    document = json.dumps(checked.model_dump(mode="json", exclude_unset=True), sort_keys=True, allow_nan=False).encode()
    return PreparedTandem(document, defaults, resources, sources)

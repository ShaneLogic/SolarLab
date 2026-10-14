"""Complete current preparation inventory, immutable resources and typed edges."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import json
import struct

import numpy as np
import pytest

from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.device.settings import DeviceSettingsInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.resources import ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.preparation import prepare_configuration, read_preparation_context
from solarlab.units import normalize_quantity
from test_registry_migration import registries as registries


def encoded(value):
    if type(value) is float:
        return ["binary64", struct.pack(">d", value).hex()]
    if isinstance(value, dict):
        return {k: encoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    return value


def test_all_55_inputs_reopen_without_aliases_or_changed_words(registries):
    registry = registries[0]
    defaults, resources = read_preparation_context(registry)
    checked = []
    for entry in registry.spec.entries:
        if entry.section != "configs":
            continue
        prepared = prepare_configuration(registry, entry.id, defaults=defaults, resources=resources).prepared
        assert prepared is not None
        before = encoded(prepared.to_mapping())
        original_input = prepared.to_input()
        resolver = resolve_tandem if original_input.schema_version == "solarlab.tandem-preparation.v1" else resolve_device
        reopened = resolver(original_input, defaults, resources, sources=prepared.sources)
        assert encoded(reopened.to_mapping()) == before
        assert reopened.input_json == prepared.input_json
        assert reopened.content_sha256 == prepared.content_sha256
        # Every returned editing/result object is detached from the immutable
        # prepared record, including tandem, defects and microstructure inputs.
        exported = prepared.to_mapping()
        exported.clear()
        detached = original_input.editing_data()
        detached.clear()
        original_input.id = "detached_copy"
        assert prepared.to_input().id != "detached_copy"
        assert encoded(prepared.to_mapping()) == before
        assert not prepared.can_execute
        checked.append(entry.id)
    assert len(checked) == 55


def test_all_actual_resources_own_immutable_exact_words(registries):
    _, library = read_preparation_context(registries[0])
    for table in library.tables:
        first, second = table.array, table.array
        before = first.tobytes()
        assert first.shape == table.shape and first.dtype == np.dtype("<f8")
        with pytest.raises(ValueError):
            first.setflags(write=True)
        with pytest.raises(ValueError):
            first.flat[0] = 0
        copy = first.copy()
        copy.flat[0] = 0
        assert second.tobytes() == table.array.tobytes() == before
        rebuilt = ResourceTable(table.name, table.kind, table.source)
        assert rebuilt.array.tobytes() == before
        assert rebuilt.content_sha256 == table.content_sha256
    assert len(library.tables) == 16


@pytest.mark.parametrize(("value", "unit", "expected"), [
    ("0.1 cm^2/(V s)", "m^2/(V s)", Fraction("0.1") / 10_000),
    ("1.602176634e-19 J", "eV", Fraction(1)),
    ("2.585e-18 m^2/s", "m^2/s", Fraction("2.585e-18")),
    (Decimal("300.00000000000000000000001"), "K", Fraction("300.00000000000000000000001")),
])
def test_exact_conversion_before_the_single_binary64_round(value, unit, expected):
    assert struct.pack(">d", normalize_quantity(value, unit)) == struct.pack(">d", float(expected))


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf"), "1 nm"])
def test_full_input_copies_still_reject_invalid_temperature_and_energy(value):
    settings = DeviceSettingsInput()
    with pytest.raises(ValueError):
        settings.model_copy(update={"T": value})
    with pytest.raises(ValueError):
        FullParameterInput.model_validate({"Eg": value})


@pytest.mark.parametrize("change", [
    lambda x: x.update(extra=1),
    lambda x: x.update(coordinate_m=[0, 0]),
    lambda x: x.update(generation_m3_s=[1]),
    lambda x: x.update(generation_m3_s=[1, True]),
    lambda x: x.update(generation_m3_s=[1, float("inf")]),
])
def test_fixed_generation_shapes_and_types_are_not_silently_repaired(change):
    data = {"coordinate_m": [0, 1], "generation_m3_s": [0, 1]}
    changed = deepcopy(data)
    change(changed)
    with pytest.raises(ValueError):
        ResourceTable("fixed", "fixed_generation", SourceDocument("fixed.json", json.dumps(changed).encode()))

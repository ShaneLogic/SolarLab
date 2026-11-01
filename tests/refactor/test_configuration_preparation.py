"""Real layer-template roundtrips, strict inputs and immutable SI/eV values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from solarlab.config.inputs import LayerInput
from solarlab.config.resolve import (
    layer_input_mapping, load_layer_input, load_template_library,
    resolve_layer, resolve_layers,
)
from solarlab.config.yaml import load_yaml_mapping
from solarlab.materials.library import MaterialLibrary
from solarlab.materials.parameters import LayerValuesInput
from solarlab.units import QuantityError, normalize_quantity

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "perovskite-sim/perovskite_sim/data/layer_templates.yaml"


@pytest.fixture
def library() -> MaterialLibrary:
    return load_template_library(CATALOG.read_bytes(), source_id="shipped.layer_templates")


def layer_input(**overrides: object) -> LayerInput:
    return LayerInput.model_validate({
        "id": "absorber_1", "name": "absorber", "template": "MAPbI3_absorber",
        "overrides": overrides,
    })


@pytest.mark.parametrize(("value", "unit", "expected"), [
    ("400 nm", "m", "4e-7"), ("4e-7 m", "m", "4e-7"),
    ("1e17 cm^-3", "m^-3", "1e23"), ("1e10 cm^-2", "m^-2", "1e14"),
    ("1 us", "s", "1e-6"), ("2 cm^2/Vs", "m^2/(V s)", "2e-4"),
    ("1e-12 cm^2/s", "m^2/s", "1e-16"), ("2 cm^3/s", "m^3/s", "2e-6"),
    ("2 cm^6/s", "m^6/s", "2e-12"), ("1 ohm cm^2", "ohm m^2", "1e-4"),
    ("1 mA/cm^2", "A/m^2", "10"), ("1000 meV", "eV", "1"),
    ("1.602176634e-19 J", "eV", "1"), ("1e-320 m^2/s", "m^2/s", "1e-320"),
])
def test_exact_unit_conversion(value: str, unit: str, expected: str) -> None:
    # Independent decimal literals, not the implementation's conversion table.
    assert normalize_quantity(value, unit).hex() == float(Decimal(expected)).hex()


@pytest.mark.parametrize("value", [True, False, None, float("inf"), float("nan"),
                                   Decimal("NaN"), "nan", "1e999999999 m", "1e-999999999 m",
                                   "1e-400 m", "1e999 m", [], {}, "1 kg", "20 degC"])
def test_invalid_or_unrepresentable_quantity_is_rejected(value: object) -> None:
    with pytest.raises(QuantityError):
        normalize_quantity(value, "m")


def test_bare_float_is_preserved_and_negative_zero_is_canonical() -> None:
    value = math.nextafter(4e-7, 1.0)
    assert normalize_quantity(value, "m").hex() == value.hex()
    assert normalize_quantity(-0.0, "m").hex() == 0.0.hex()
    with pytest.raises(QuantityError):
        normalize_quantity("1 cm", "cm")


def test_every_actual_template_retains_its_scalar_values(library: MaterialLibrary) -> None:
    original = yaml.safe_load(CATALOG.read_bytes())
    assert {item.id for item in library.templates} == set(original)
    assert library.source_sha256 == hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    for index, (name, record) in enumerate(original.items()):
        resolved = resolve_layer(LayerInput(id=f"layer_{index}", name=name, template=name), library)
        values = {**resolved.material.values, "thickness": resolved.thickness,
                  "N_A": resolved.N_A, "N_D": resolved.N_D}
        for key, value in record["defaults"].items():
            if key == "incoherent":
                assert values[key] is value
            else:
                assert type(values[key]) is float
                assert values[key].hex() == float(value).hex()
        assert values["optical_material"] == record["optical_material"]
        assert all(origin.kind == "template" for origin in resolved.origins)


def test_unit_equivalent_edit_documents_resolve_identically(library: MaterialLibrary) -> None:
    left = resolve_layer(layer_input(thickness="400 nm", N_A="1e17 cm^-3"), library)
    right = resolve_layer(layer_input(thickness="4e-7 m", N_A="1e23 m^-3"), library)
    assert left.to_mapping() == right.to_mapping()
    assert left.content_sha256 == right.content_sha256
    assert left.to_mapping()["units"]["chi"] == "eV"
    assert {"thickness", "N_A", "N_D"}.isdisjoint(left.material.values)


def test_zero_null_and_absence_have_distinct_semantics(library: MaterialLibrary) -> None:
    inherited = resolve_layer(layer_input(), library)
    zero = resolve_layer(layer_input(mu_n=0), library)
    cleared = resolve_layer(layer_input(optical_material=None), library)
    assert inherited.material.values["mu_n"] > 0
    assert zero.material.values["mu_n"] == 0.0
    assert inherited.material.values["optical_material"] == "MAPbI3"
    assert cleared.material.values["optical_material"] is None
    assert next(item.kind for item in zero.origins if item.parameter == "mu_n") == "override"
    assert next(item.kind for item in inherited.origins if item.parameter == "mu_n") == "template"
    assert layer_input_mapping(layer_input())["overrides"] == {}
    assert layer_input_mapping(layer_input(optical_material=None))["overrides"] == {"optical_material": None}
    with pytest.raises(ValidationError, match="omit it to inherit"):
        layer_input(mu_n=None)


@pytest.mark.parametrize("overrides", [
    {"mu_typo": 1}, {"mu_n": True}, {"tau_n": float("nan")}, {"tau_n": -1},
    {"thickness": 0}, {"incoherent": "yes"}, {"incoherent": 1},
    {"optical_material": ""}, {"mu_n": [1, 2]}, {"profile_file": "unimplemented.csv"},
    {"D_ion": "1e-400 m^2/s"},
])
def test_invalid_overrides_fail_with_a_field_location(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as error:
        layer_input(**overrides)
    assert error.value.errors()[0]["loc"][0] == "overrides"


def test_edits_and_copy_updates_revalidate(library: MaterialLibrary) -> None:
    editable = layer_input(thickness="400 nm")
    resolved = resolve_layer(editable, library)
    editable.overrides.thickness = "450 nm"
    assert resolved.thickness == 4e-7
    assert resolve_layer(editable, library).thickness == 4.5e-7
    with pytest.raises(ValidationError):
        editable.overrides.thickness = -1
    with pytest.raises(ValidationError):
        editable.overrides.model_copy(update={"tau_n": -1})
    with pytest.raises(ValidationError):
        editable.model_copy(update={"unknown": 1})
    broken = LayerInput.model_construct(
        id="broken", name="broken", template="MAPbI3_absorber",
        overrides=LayerValuesInput.model_construct(mu_n=float("nan")),
    )
    with pytest.raises(ValidationError):
        resolve_layer(broken, library)


def test_typed_input_nesting_preserves_absent_fields(library: MaterialLibrary) -> None:
    overrides = LayerValuesInput(mu_n=0)
    assert LayerValuesInput.model_validate(overrides).model_fields_set == {"mu_n"}
    editable = LayerInput(id="a", name="a", template="MAPbI3_absorber", overrides=overrides)
    assert editable.overrides.model_fields_set == {"mu_n"}
    assert LayerInput.model_validate(editable).overrides.model_fields_set == {"mu_n"}
    assert resolve_layer(editable, library).material.values["mu_n"] == 0.0


def test_editing_default_nested_overrides_is_not_lost(library: MaterialLibrary) -> None:
    editable = LayerInput(id="a", name="a", template="MAPbI3_absorber")
    assert "overrides" not in layer_input_mapping(editable)
    editable.overrides.mu_n = 0
    assert resolve_layer(editable, library).material.values["mu_n"] == 0.0
    restored = LayerInput.model_validate(layer_input_mapping(editable))
    assert restored.overrides.model_fields_set == {"mu_n"}
    assert editable.validated_update({"name": "renamed"}).overrides.mu_n == 0
    assert editable.model_copy().overrides.model_fields_set == {"mu_n"}
    copied = editable.model_copy(update={"name": "copied"}, deep=True)
    assert resolve_layer(copied, library).material.values["mu_n"] == 0.0


def test_input_schema_has_no_fake_physical_defaults_or_numeric_null() -> None:
    properties = LayerValuesInput.model_json_schema()["properties"]
    assert all("default" not in definition for definition in properties.values())
    for name, definition in properties.items():
        allows_null = {"type": "null"} in definition.get("anyOf", [])
        assert allows_null is (name == "optical_material")


def test_resolved_values_are_deeply_immutable_and_detached(library: MaterialLibrary) -> None:
    raw = {"id": "a", "name": "absorber", "template": "MAPbI3_absorber",
           "overrides": {"thickness": "400 nm"}}
    editable = LayerInput.model_validate(raw)
    resolved = resolve_layer(editable, library)
    digest = resolved.content_sha256
    raw["overrides"]["thickness"] = "999 nm"
    detached = resolved.to_mapping()
    detached["material"]["parameters"]["mu_n"] = 999
    assert resolved.content_sha256 == digest
    with pytest.raises(FrozenInstanceError):
        resolved.thickness = 1
    with pytest.raises(TypeError):
        resolved.material.values["mu_n"] = 1
    with pytest.raises(TypeError):
        library.get("MAPbI3_absorber").values["mu_n"] = 1
    with pytest.raises(ValidationError):
        replace(resolved, N_A=float("nan"))


def test_resolved_unit_metadata_is_owned(library: MaterialLibrary, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = resolve_layer(layer_input(), library)
    digest = resolved.content_sha256
    metadata = LayerValuesInput.model_fields["mu_n"].json_schema_extra
    assert isinstance(metadata, dict)
    monkeypatch.setitem(metadata, "unit", "cm^2/(V s)")
    assert resolved.content_sha256 == digest
    assert resolved.to_mapping()["units"]["mu_n"] == "m^2/(V s)"


def test_display_names_may_repeat_but_stable_layer_ids_may_not(library: MaterialLibrary) -> None:
    first = layer_input()
    second = first.validated_update({"id": "absorber_2"})
    assert len(resolve_layers([first, second], library)) == 2
    with pytest.raises(ValueError, match="unique stable IDs"):
        resolve_layers([first, first], library)


def test_source_replacement_changes_identity_even_when_values_match(library: MaterialLibrary) -> None:
    altered = load_template_library(CATALOG.read_bytes() + b"\n# source update\n", source_id=library.source_id)
    assert altered.get("MAPbI3_absorber").parameters == library.get("MAPbI3_absorber").parameters
    assert altered.source_sha256 != library.source_sha256
    assert resolve_layer(layer_input(), altered).content_sha256 != resolve_layer(layer_input(), library).content_sha256


def test_yaml12_and_json_edit_roundtrip(library: MaterialLibrary) -> None:
    document = """
id: absorber_1
name: absorber
template: MAPbI3_absorber
overrides:
  thickness: 400 nm
  N_A: 1e17 cm^-3
  mu_n: 0
  optical_material: null
  incoherent: false
"""
    original = load_layer_input(document)
    restored = load_layer_input(json.dumps(layer_input_mapping(original)))
    assert resolve_layer(original, library).to_mapping() == resolve_layer(restored, library).to_mapping()
    values = load_yaml_mapping("a: 1e-9\nb: 012\nc: 0o12\nd: 0x12\ne: yes\nf: on\ng: true\nh: null\n")
    assert values == {"a": Decimal("1e-9"), "b": 12, "c": 10, "d": 18, "e": "yes", "f": "on", "g": True, "h": None}
    assert yaml.safe_load("old: yes")["old"] is True  # Do not mutate the legacy loader.


@pytest.mark.parametrize("document", [
    "x: 1\nx: 2", "true: value", "x: .nan", "x: .inf", "x: !!bool yes",
    "x: !!null nope", "x: !!float 1_000", "x: !!int 1:20", "[]",
    "x: !!python/object/new:builtins.object {}", "x: &x {self: *x}",
    "x: &x [*x]", "x: !!set {a: null}",
])
def test_ambiguous_unsafe_and_cyclic_yaml_is_rejected(document: str) -> None:
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_yaml_mapping(document)


def test_catalog_unknown_missing_and_competing_fields_reject() -> None:
    document = yaml.safe_load(CATALOG.read_bytes())
    document["MAPbI3_absorber"]["defaults"]["extra"] = 1
    with pytest.raises(ValueError, match="MAPbI3_absorber.defaults"):
        load_template_library(yaml.safe_dump(document), source_id="invalid")
    del document["MAPbI3_absorber"]["defaults"]["extra"]
    del document["MAPbI3_absorber"]["defaults"]["N_A"]
    with pytest.raises(ValueError, match="missing template parameters"):
        load_template_library(yaml.safe_dump(document), source_id="invalid")
    document["MAPbI3_absorber"]["defaults"]["optical_material"] = None
    with pytest.raises(ValueError, match="competing sources"):
        load_template_library(yaml.safe_dump(document), source_id="invalid")

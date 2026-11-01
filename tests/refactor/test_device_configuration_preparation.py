"""Real configuration data, source identity, edits and immutable preparation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import layer_parameter_source
from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, FullLayerInput, NamedMaterialInput
from solarlab.device.settings import DeviceSettingsInput
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityContext, CapabilityRule, ModelDefinition, ModelRef, ModelRegistry, Scope
from solarlab.physics.selection import ModelChoice, Topology, select_models

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
CONFIGS = ROOT / "perovskite-sim/tests/fixtures/configs"
CASES = {
    "standard": CONFIGS / "nip_MAPbI3.yaml",
    "scaps": ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml",
    "defect": CONFIGS / "scaps_defect_s1_acceptor_n.yaml",
    "grain": CONFIGS / "twod/nip_MAPbI3_singleGB.yaml",
    "distributed": CONFIGS / "distributed_defect_qf_dc_pn.yaml",
    "spatial": CONFIGS / "graded_distributed_defect_qf_dc_pn.yaml",
}
DEFAULT_SOURCES = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                   "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                   "twod/microstructure.py", "models/tandem_config.py")


@pytest.fixture(scope="module")
def defaults():
    return read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES))


@pytest.fixture(scope="module")
def resources():
    return ResourceLibrary(tuple(ResourceTable(path.stem, "nk", SourceDocument(str(path.relative_to(ROOT)), path.read_bytes()))
                                 for path in sorted((LEGACY / "data/nk").glob("*.csv"))))


def imported(case, defaults):
    path = CASES[case]
    source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    loader = import_scaps_device if case == "scaps" else import_standard_device
    return loader(source, id=case, defaults=defaults), source


def prepared(case, defaults, resources):
    input, source = imported(case, defaults)
    return resolve_device(input, defaults, resources, sources=(source,))


def edited_source_bytes(raw):
    def decimal_string(value):
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError(type(value).__name__)
    return json.dumps(raw, default=decimal_string).encode()


@pytest.mark.parametrize("case", CASES)
def test_actual_device_inputs_resolve_roundtrip_and_preserve_fields(case, defaults, resources):
    input, source = imported(case, defaults)
    output = resolve_device(input, defaults, resources, sources=(source,))
    assert resolve_device(output.to_input(), defaults, resources, sources=(source,)).content_sha256 == output.content_sha256
    assert not output.can_execute
    raw = load_yaml_mapping(source.content)
    assert len(output.layers) == len(raw["layers"])
    if case != "scaps":
        for declared, layer in zip(raw["layers"], output.layers):
            assert layer.name == declared["name"] and layer.role == declared["role"]
            assert layer.thickness.hex() == float(declared["thickness"]).hex()
            values = dict(layer.parameters)
            for name in FullParameterInput.model_fields:
                if name in declared:
                    expected = float(declared[name]) if isinstance(declared[name], Decimal) else declared[name]
                    assert values[name] == expected, name
    assert all(len(layer.field_origins) == len(layer.parameters) for layer in output.layers)
    for layer in output.to_mapping()["layers"]:
        assert all("effective_value" in origin for origin in layer["provenance"])


def test_scaps_units_defects_and_glass_diagnostic_are_explicit(defaults, resources):
    output = prepared("scaps", defaults, resources)
    glass, htl, pvk, etl = output.layers
    assert glass.thickness == 1e-3
    assert glass.material.values["incoherent"] is True
    assert htl.thickness == 20e-9 and etl.thickness == 25e-9
    assert pvk.material.values["mu_n"] == 0.002
    assert len(pvk.defects) == 2
    assert pvk.local_parameters["tau_n"] == pytest.approx(5e-5, rel=1e-15)
    assert {defect.values["charge_transition"] for defect in pvk.defects} == {"unresolved"}
    with localcontext() as ctx:
        ctx.prec = 80
        voltage = Decimal("1.380649e-23") * 300 / Decimal("1.602176634e-19")
        independent = (Decimal("1e25") ** 2 * (-Decimal("1.53") / voltage).exp()).sqrt()
    assert pvk.material.values["ni"] == pytest.approx(float(independent), rel=3e-14)
    diagnostic, = output.values["diagnostics"]
    assert diagnostic["path"] == "layers.layer_0.parameters.incoherent"
    assert diagnostic["declared"] is True and diagnostic["historical_effective"] is False
    assert diagnostic["loader_bindings"] and diagnostic["source_bindings"]
    assert not output.interfaces[0].electrical
    assert output.interfaces[1].values["left"] == htl.id
    assert output.interfaces[1].values["right"] == pvk.id
    assert output.interfaces[2].values["defect"]["partner_metadata"]["distribution"] == "gaussian"


def test_scaps_edits_recompute_dos_lifetimes_and_referenced_energy(defaults, resources):
    input, source = imported("scaps", defaults)
    before = resolve_device(input, defaults, resources, sources=(source,))
    input.layers[2].parameters.Eg = "1.6 eV"
    input.layers[2].bulk_defects[0].kinetics.sigma_n_m2 = "2e-19 m^2"
    after = resolve_device(input, defaults, resources, sources=(source,))
    pvk = after.layers[2]
    assert pvk.defects[0].values["distribution"]["center_eV_above_vb"] == 1.5
    assert pvk.defects[1].values["distribution"]["center_eV_above_vb"] == 0.1
    assert pvk.material.values["ni"] != before.layers[2].material.values["ni"]
    assert pvk.local_parameters["tau_n"] == pytest.approx(1 / 30000, rel=1e-15)
    assert before.layers[2].defects[0].values["distribution"]["center_eV_above_vb"] == 1.43
    input.layers[2].parameters.ni = 1.0
    with pytest.raises(ValueError, match="SCAPS-derived"):
        resolve_device(input, defaults, resources)


def test_scaps_legacy_comparison_pins_equal_values_and_exact_conversion_differences(defaults, resources):
    # This imports the old data loader only as a comparison oracle. It does
    # not build numerical material arrays or call an equilibrium/trajectory.
    from perovskite_sim.scaps_compat.loader import load_scaps_yaml

    historical = load_scaps_yaml(CASES["scaps"])
    current = prepared("scaps", defaults, resources)
    # These eight differences are explicitly retained W6 exact-unit results,
    # not historical M execution parity. P03-04 owns that later projection.
    expected = {
        ("layer_1", "mu_n"): ("0x1.ad7f29abcaf49p-24", "0x1.ad7f29abcaf48p-24"),
        ("layer_1", "mu_p"): ("0x1.ad7f29abcaf49p-24", "0x1.ad7f29abcaf48p-24"),
        ("layer_2", "B_rad"): ("0x1.2725dd1d243abp-60", "0x1.2725dd1d243acp-60"),
        ("layer_2", "tau_n"): ("0x1.a36e2eb1c432bp-15", "0x1.a36e2eb1c432dp-15"),
        ("layer_2", "tau_p"): ("0x1.a36e2eb1c432bp-15", "0x1.a36e2eb1c432dp-15"),
        ("layer_2", "thickness"): ("0x1.ad7f29abcaf49p-21", "0x1.ad7f29abcaf48p-21"),
        ("layer_3", "mu_n"): ("0x1.0c6f7a0b5ed8ep-20", "0x1.0c6f7a0b5ed8dp-20"),
        ("layer_3", "thickness"): ("0x1.ad7f29abcaf49p-26", "0x1.ad7f29abcaf48p-26"),
    }
    mismatches = {}
    unchanged = 0
    for original, layer in zip(historical.layers, current.layers):
        pairs = [(name, getattr(original.params, name), value) for name, value in layer.parameters]
        pairs.append(("thickness", original.thickness, layer.thickness))
        for name, before, value in pairs:
            if layer.role == "substrate" and name == "incoherent":
                assert before is False and value is True
                continue
            before_word = before.hex() if isinstance(before, float) else before
            value_word = value.hex() if isinstance(value, float) else value
            if type(value) is not type(before) or value_word != before_word:
                mismatches[(layer.id, name)] = (before_word, value_word)
            else:
                unchanged += 1
        for field in ("ni", "n1", "p1"):
            assert dict(layer.parameters)[field].hex() == getattr(original.params, field).hex()
    assert unchanged == 271
    assert mismatches == expected


def test_named_material_inheritance_zero_null_and_duplicate_display_names(defaults, resources):
    input, source = imported("standard", defaults)
    original = resolve_device(input, defaults, resources)
    material_values = dict(original.layers[0].material.parameters)
    material_values["optical_material"] = "MAPbI3"
    input.materials = (NamedMaterialInput(id="shared", name="Same material", parameters=FullParameterInput.model_validate(material_values)),)
    for layer in input.layers[:2]:
        layer.name = "Same display name"
        layer.material = "shared"
    data = input.layers[0].parameters.editing_data()
    data.pop("mu_n")
    input.layers[0].parameters = FullParameterInput.model_validate(data)
    inherited = resolve_device(input, defaults, resources, sources=(source,))
    assert inherited.layers[0].material.values["mu_n"] == original.layers[0].material.values["mu_n"]
    assert inherited.layers[0].optical.name == "MAPbI3"
    input.layers[0].parameters.mu_n = 0
    input.layers[0].parameters.optical_material = None
    changed = resolve_device(input, defaults, resources, sources=(source,))
    assert changed.layers[0].material.values["mu_n"] == 0
    assert changed.layers[0].optical is None
    assert changed.layers[0].id != changed.layers[1].id
    assert inherited.layers[0].optical.name == "MAPbI3"
    with pytest.raises(ValueError):
        input.layers[0].parameters.mu_n = None


def test_default_child_edits_inside_layer_tuples_survive_validation_and_copy(defaults, resources):
    input, _ = imported("standard", defaults)
    data = input.layers[0].editing_data()
    parameters = data.pop("parameters")
    input.layers = (FullLayerInput.model_validate(data), *input.layers[1:])
    for name, value in parameters.items():
        setattr(input.layers[0].parameters, name, value)
    input.layers[0].parameters.mu_n = 0
    copied = input.model_copy()
    assert dict(resolve_device(copied, defaults, resources).layers[0].parameters)["mu_n"] == 0
    copied.layers[0].parameters.mu_n = "1 cm^2/(V s)"
    assert input.layers[0].parameters.mu_n == 0
    assert dict(resolve_device(copied, defaults, resources).layers[0].parameters)["mu_n"] == 1e-4


def test_unit_schema_matches_null_and_omission_rules(defaults):
    fields = FullParameterInput.model_json_schema()["properties"]
    for name in ("mu_n", "incoherent", "carrier_statistics", "grading_N_mult"):
        assert "default" not in fields[name]
        assert {"type": "null"} not in fields[name]["anyOf"]
        with pytest.raises(ValueError):
            FullParameterInput.model_validate({name: None})
    for name in ("optical_material", "Nc300", "trap_decay_length"):
        assert {"type": "null"} in fields[name]["anyOf"]
        assert dict(FullParameterInput.model_validate({name: None}).normalized_items())[name] is None
    setting = DeviceSettingsInput.model_json_schema()["properties"]
    assert {"type": "null"} not in setting["Phi"]["anyOf"]
    assert {"type": "null"} in setting["built_in_potential_mode"]["anyOf"]


def test_explicit_zero_trap_length_and_copied_material_constraints(defaults, resources):
    input, _ = imported("standard", defaults)
    input.layers[1].parameters.trap_decay_length = 0
    output = resolve_device(input, defaults, resources)
    assert output.layers[1].local_parameters["trap_decay_length"] == 0
    original = output.layers[1].material
    for replacements in ({"bgn_conduction_band_fraction": 2}, {"P0": 2, "P_lim": 1},
                         {"carrier_statistics": "fermi_dirac", "Nc300": None}):
        values = {**dict(original.parameters), **replacements}
        with pytest.raises(ValueError):
            replace(original, parameters=tuple(values.items()))


def test_grain_overlap_and_source_default_role_have_explicit_layer_ownership(defaults, resources):
    raw = load_yaml_mapping(CASES["grain"].read_bytes())
    raw["microstructure"]["grain_boundaries"][0].pop("layer_role")
    input = import_standard_device(SourceDocument("grain-with-default", edited_source_bytes(raw)), id="grain", defaults=defaults)
    original = resolve_device(input, defaults, resources)
    grain = input.grain_boundaries[0]
    assert original.values["grain_boundaries"][0]["source_layer_role"] == dict(defaults.structural)["grain_boundary_layer_role"]
    input.grain_boundaries = (grain, grain.model_copy(update={"id": "second"}))
    with pytest.raises(ValueError, match="overlapping grain-boundary"):
        resolve_device(input, defaults, resources)
    input.grain_boundaries[1].layer_ids = (input.layers[0].id,)
    assert len(resolve_device(input, defaults, resources).values["grain_boundaries"]) == 2


def test_tandem_uses_supplied_documents_and_keeps_optical_only_layers(defaults, resources):
    source = SourceDocument("tandem_lin2019.yaml", (CONFIGS / "tandem_lin2019.yaml").read_bytes())
    refs = {name: SourceDocument(name, (CONFIGS / name).read_bytes()) for name in ("nip_wideGap_FACs_1p77.yaml", "nip_SnPb_1p22.yaml")}
    input = import_tandem(source, id="lin2019", references=refs, defaults=defaults)
    output = resolve_tandem(input, defaults, resources, sources=(source, *refs.values()))
    data = output.to_mapping()
    assert len(output.top_cell.layers) == 5 and len(output.bottom_cell.layers) == 4
    assert len(data["optical_layers"]) == 3
    assert data["optical_layers"][-1]["optical_material"] == "Cu"
    assert data["benchmark"]["target_jsc_A_m2"] == 156
    assert data["benchmark"]["target_pce_fraction"] == pytest.approx(0.248)
    assert data["junction_model"] == "ideal_ohmic"
    assert not output.can_execute
    assert resolve_tandem(output.to_input(), defaults, resources, sources=(source, *refs.values())).content_sha256 == output.content_sha256
    with pytest.raises(ValueError, match="missing explicitly supplied"):
        import_tandem(source, id="lin", references={}, defaults=defaults)
    raw = load_yaml_mapping(source.content)
    raw["benchmark"] = None
    raw["tandem"].pop("light_direction")
    edited = import_tandem(replace(source, content=edited_source_bytes(raw)), id="lin", references=refs, defaults=defaults)
    assert edited.benchmark is None and edited.light_direction == dict(defaults.structural)["tandem_light_direction"]


def test_immutable_nested_values_arrays_and_actual_resource_identity(defaults, resources):
    output = prepared("scaps", defaults, resources)
    with pytest.raises(TypeError):
        output.values["settings"]["T"] = 500
    with pytest.raises(TypeError):
        output.layers[2].defects[0].values["kinetics"]["sigma_n_m2"] = 1
    with pytest.raises(FrozenInstanceError):
        output.layers[0].thickness = 2
    with pytest.raises((TypeError, ValueError)):
        replace(output, layers=())
    table = resources.get("MAPbI3", "nk")
    view = table.array
    before = view.copy()
    with pytest.raises(ValueError):
        view[0, 1] = 3
    with pytest.raises(ValueError):
        view.setflags(write=True)
    view.shape = (view.size,)
    view.dtype = np.int64
    np.testing.assert_array_equal(table.array, before)
    with pytest.raises(FrozenInstanceError):
        table.shape = (1, 2)
    with pytest.raises((TypeError, ValueError)):
        replace(table, shape=(1, 2))
    replacement = ResourceTable(table.name, table.kind, replace(table.source, content=table.source.content.replace(b"2.6214", b"2.7000", 1)))
    assert replacement.source.id == table.source.id
    assert replacement.content_sha256 != table.content_sha256
    other = ResourceLibrary(tuple(replacement if item.name == table.name else item for item in resources.tables))
    changed = resolve_device(output.to_input(), defaults, other, sources=output.sources)
    assert changed.layers[2].material.content_sha256 != output.layers[2].material.content_sha256
    assert changed.content_sha256 != output.content_sha256


def test_fixed_generation_and_spectrum_bind_only_supplied_data(defaults, resources):
    source = SourceDocument("provided-generation", b'{"coordinate_m":[0,0.0000007],"generation_m3_s":[1e25,2e25]}')
    generation = ResourceTable("measured-generation", "fixed_generation", source)
    spectrum_path = LEGACY / "data/am15g.csv"
    spectrum = ResourceTable("am15g", "spectrum", SourceDocument(str(spectrum_path.relative_to(ROOT)), spectrum_path.read_bytes()))
    input, _ = imported("standard", defaults)
    input.fixed_generation = generation.name
    input.spectrum = spectrum.name
    output = resolve_device(input, defaults, ResourceLibrary((*resources.tables, generation, spectrum)))
    assert {item["kind"] for item in output.values["resource_bindings"]} == {"fixed_generation", "spectrum"}
    assert generation.array.shape == (2, 2)
    assert generation.array[1, 1] == 2e25
    assert spectrum.units == ("m", "m^-3/s")
    with pytest.raises(ValueError, match="missing supplied"):
        resolve_device(input, defaults, resources)


@pytest.mark.parametrize("content", [
    b'{"coordinate_m":[0,1],"generation_m3_s":[1]}',
    b'{"coordinate_m":[0,true],"generation_m3_s":[1,2]}',
    b'{"coordinate_m":[0,0],"generation_m3_s":[1,2]}',
    b'{"coordinate_m":[0,1],"generation_m3_s":[1,NaN]}',
    b'{"coordinate_m":[0,1],"generation_m3_s":[1,2],"profile_file":"missing"}',
    b'{"coordinate_m":[0,1],"coordinate_m":[0,2],"generation_m3_s":[1,2]}',
])
def test_invalid_resource_shape_coordinate_and_unknown_keys_reject(content):
    with pytest.raises(ValueError):
        ResourceTable("bad", "fixed_generation", SourceDocument("provided", content))


@pytest.mark.parametrize("field,value", [
    ("mu_n", True), ("mu_n", float("nan")), ("mu_n", float("inf")),
    ("tau_n", 0), ("mu_n", "1 kg"), ("grading_N_mult", True),
    ("grading_N_mult", 1.5), ("incoherent", "false"), ("Nc300", 0),
])
def test_full_parameters_reject_invalid_units_numbers_and_boolean_aliases(field, value):
    with pytest.raises(ValueError):
        FullParameterInput.model_validate({field: value})


def test_unknown_nested_paths_and_copied_resolved_inputs_reject(defaults, resources):
    input, source = imported("standard", defaults)
    document = input.model_dump(mode="json", exclude_unset=True)
    document["layers"][0]["parameters"]["profile_file"] = "absent.csv"
    with pytest.raises(ValueError, match="profile_file"):
        DeviceInput.model_validate(document)
    with pytest.raises(ValueError, match="profile_file"):
        PreparedDevice(json.dumps(document).encode(), defaults, resources)
    output = resolve_device(input, defaults, resources, sources=(source,))
    bad = output.to_input().model_dump(mode="json", exclude_unset=True)
    bad["layers"][0]["parameters"]["tau_n"] = 0
    with pytest.raises(ValueError):
        replace(output, input_json=json.dumps(bad).encode())
    with pytest.raises(ValueError):
        replace(output.layers[0], parameters=tuple((name, value) for name, value in output.layers[0].parameters if name != "mu_n"))
    with pytest.raises(ValueError):
        input.model_copy(update={"extra_field": 1})
    with pytest.raises(ValueError):
        input.settings.model_copy(update={"T": False})


def test_conflicting_contact_aliases_and_explicit_null_or_zero(defaults, resources):
    raw = load_yaml_mapping(CASES["standard"].read_bytes())
    raw["device"].update(S_n_left=0, contacts={"left": {"S_n": None}})
    source = SourceDocument("conflict", edited_source_bytes(raw))
    with pytest.raises(ValueError, match="conflicting flat/nested"):
        import_standard_device(source, id="conflict", defaults=defaults)
    raw["device"]["contacts"]["left"]["S_n"] = 0
    input = import_standard_device(replace(source, content=edited_source_bytes(raw)), id="equal", defaults=defaults)
    output = resolve_device(input, defaults, resources)
    assert output.values["contacts"][0]["S_n"] == 0
    input.contacts[0].S_n = None
    assert resolve_device(input, defaults, resources).values["contacts"][0]["S_n"] is None


def test_scope_and_ownership_failures_are_not_hidden_by_ids(defaults, resources):
    input, _ = imported("grain", defaults)
    input.grain_boundaries[0].layer_ids = ("missing",)
    with pytest.raises(ValueError, match="grain boundaries"):
        resolve_device(input, defaults, resources)
    input, _ = imported("standard", defaults)
    input.interfaces[0].left, input.interfaces[0].right = input.interfaces[0].right, input.interfaces[0].left
    with pytest.raises(ValueError, match="reversed"):
        resolve_device(input, defaults, resources)
    input, _ = imported("standard", defaults)
    input.layers[1].id = input.layers[0].id
    with pytest.raises(ValueError, match="unique layer IDs"):
        resolve_device(input, defaults, resources)
    input, _ = imported("scaps", defaults)
    input.layers[2].scaps_defect_metadata[0].defect_id = "missing"
    with pytest.raises(ValueError, match="unknown defect ID"):
        resolve_device(input, defaults, resources)


def test_resolved_declaration_keeps_all_typed_partner_fields(defaults, resources):
    output = prepared("scaps", defaults, resources)
    declaration = output.to_mapping()["declaration"]
    assert declaration == output.to_input().normalized_data()
    assert declaration["layers"][2]["scaps_defect_metadata"][0]["energy_reference"] == "below_conduction_band"
    assert declaration["interfaces"][2]["defect"]["partner_metadata"]["E_char_eV"] == 0.1
    assert "P03_04_legacy_unit_conversion_bytes_pending" in output.values["capability_gaps"]


@pytest.mark.parametrize("case,version", [("distributed", "v1"), ("spatial", "v2")])
def test_defect_version_rejects_incompatible_existing_distribution_and_spatial_fields(case, version, defaults, resources):
    input, _ = imported(case, defaults)
    input.layers[0].defect_schema_version = "solarlab-explicit-bulk-defects-" + version
    with pytest.raises(ValueError, match=version):
        resolve_device(input, defaults, resources)


def test_empty_explicit_inventory_and_invalid_finite_support_reject(defaults, resources):
    input, _ = imported("defect", defaults)
    layer = resolve_device(input, defaults, resources).layers[0]
    with pytest.raises(ValueError, match="nonempty defect inventory"):
        replace(layer, defects=())
    with pytest.raises(ValueError, match="requires a schema version"):
        replace(layer, defect_schema_version=None)
    input.layers[0].bulk_defects = ()
    with pytest.raises(ValueError, match="nonempty defect inventory"):
        resolve_device(input, defaults, resources)
    input, _ = imported("distributed", defaults)
    input.layers[0].bulk_defects[0].distribution.width_eV = 2
    with pytest.raises(ValueError, match="support lies outside"):
        resolve_device(input, defaults, resources)
    input, _ = imported("spatial", defaults)
    input.layers[0].bulk_defects[0].spatial_profile.knots[0].density_multiplier = 100
    with pytest.raises(ValueError, match="layer-average unity"):
        resolve_device(input, defaults, resources)


def test_catalog_has_effective_values_and_source_adapter_rejects_code(defaults):
    document = defaults.to_mapping()
    copied = DefaultCatalog.from_mapping(document)
    assert copied.content_sha256 == defaults.content_sha256
    document["material"]["mu_T_gamma"] = -2
    changed = DefaultCatalog.from_mapping(document)
    assert changed.evidence == defaults.evidence
    assert changed.content_sha256 != defaults.content_sha256
    assert dict(defaults.material)["mu_T_gamma"] == -1.5
    sources = tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES)
    bad = tuple(replace(source, content=source.content.replace(b"mu_T_gamma: float = -1.5", b"mu_T_gamma: float = forbidden_call()"))
                if source.id == "models/parameters.py" else source for source in sources)
    with pytest.raises(ValueError, match="unsupported source default expression"):
        read_legacy_default_catalog(bad)


def test_real_full_layer_values_feed_versioned_registry_without_scalar_escape(defaults, resources):
    layer = prepared("scaps", defaults, resources).layers[2]
    source = layer_parameter_source(layer)
    ref = ModelRef("full.material_transport", 1)
    rule = CapabilityRule("full_declaration_only", (1,), ("dc",), (source.evidence,))
    definition = ModelDefinition(ref, "Full typed material data", "transport", ("layer",), "face",
                                 ("mu_n", "mu_p", "Nc300", "Nv300", "Eg"), (), (), rule,
                                 parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    chosen = select_models(registry, Topology((layer.id,)), CapabilityContext(1, "dc", "metadata_only", "maxwell_boltzmann"),
                           {"layer": (ModelChoice("carrier", ref, parameter_schema="full_layer"),)},
                           {Scope("layer", (layer.id,)): source})
    assert dict(chosen.instances[0].parameters)["Nc300"] == layer.material.values["Nc300"]
    assert source.parameter_schema == "full_layer"
    with pytest.raises(ValueError, match="parameter schema"):
        replace(definition, parameter_schema="scalar_layer")
    with pytest.raises(ValueError, match="schema binding mismatch"):
        select_models(registry, Topology((layer.id,)), chosen.context,
                      {"layer": (ModelChoice("carrier", ref),)}, {Scope("layer", (layer.id,)): source})


def test_public_resolution_needs_no_source_reader_or_legacy_installation(defaults, resources, tmp_path):
    input, _ = imported("standard", defaults)
    catalog_path = tmp_path / "catalog.json"
    input_path = tmp_path / "input.json"
    catalog_path.write_text(json.dumps(defaults.to_mapping()))
    input_path.write_text(input.model_dump_json(exclude_unset=True))
    code = '''import importlib.abc,json,sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(('perovskite_sim','solarlab_server','solarlab_research','backend','solarlab.config.legacy_defaults')):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Boundary())
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.config.resolve_device import resolve_device
from solarlab.materials.resources import ResourceLibrary
catalog=DefaultCatalog.from_mapping(json.loads(Path(sys.argv[1]).read_text()))
input=DeviceInput.model_validate_json(Path(sys.argv[2]).read_text())
result=resolve_device(input,catalog,ResourceLibrary())
assert len(result.layers)==3 and result.can_execute is False
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(catalog_path), str(input_path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

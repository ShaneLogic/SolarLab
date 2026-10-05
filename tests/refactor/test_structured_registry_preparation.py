"""Actual closed DTOs in canonical model selection; no model is executed.

An injected accepted input packet avoids rebuilding prior source/default
evidence in the supervised run. A checkout can instead use the same existing
fixtures and data-only importers. Neither path calls a solver or equilibrium.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from solarlab.config.defect_documents import import_metastable_document, import_metastable_preparation
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import (
    device_topology, resolve_structured_edit, structured_parameter_source,
    structured_parameter_sources, update_structured_source,
)
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.presets import NamedPreset, select_preset
from solarlab.physics.registry import (
    CapabilityContext, CapabilityRule, ModelDefinition, ModelRef, ModelRegistry,
    OwnedVariable, Scope,
)
from solarlab.physics.selection import ModelChoice, select_models, validate_selection

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
FIXTURES = ROOT / "perovskite-sim/tests/fixtures/configs"
COMPLEX = (
    "scaps_defect_m1_double_donor_p.yaml", "scaps_defect_m2_double_acceptor_n.yaml",
    "scaps_defect_m3_amphoteric_i.yaml", "csi_gaussian_bulk_trap_pn_research.yaml",
    "cigs_graded_optics.yaml", "wkb_resolved_electron_barrier.yaml",
    "wkb_tunnelling_intraband_spike.yaml", "programmatic_metastable",
)
MONOVALENT = ("scaps_defect_s1_acceptor_n.yaml", "distributed_defect_qf_dc_pn.yaml",
              "graded_distributed_defect_qf_dc_pn.yaml", "scaps_mirror_v2.yaml")


@pytest.fixture(scope="module")
def prepared_case():
    packet_path = os.environ.get("SOLARLAB_STRUCTURED_INPUT_PACKET")
    packet = json.loads(Path(packet_path).read_text()) if packet_path else None
    if packet is not None:
        defaults = DefaultCatalog.from_mapping(packet["catalog"])
    else:
        paths = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                 "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                 "twod/microstructure.py", "models/tandem_config.py")
        defaults = read_legacy_default_catalog(
            tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in paths),
            model_sources=tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in
                                ("physics/cigs_optics.py", "models/tunneling_channels.py")))
    resources = ResourceLibrary(tuple(ResourceTable(path.stem, "nk", SourceDocument(str(path.relative_to(ROOT)), path.read_bytes()))
                                     for path in sorted((LEGACY / "data/nk").glob("*.csv"))))

    @lru_cache(maxsize=None)
    def prepare(name):
        if packet is not None and name in packet["cases"]:
            entry = packet["cases"][name]
            sources = tuple(SourceDocument(item["id"], item["content"].encode()) for item in entry["sources"])
            assert [item.sha256 for item in sources] == [item["sha256"] for item in entry["sources"]]
            result = resolve_device(DeviceInput.model_validate(entry["input"]), defaults, resources, sources=sources)
            # Reuse, rather than rerun, the accepted DTO/legacy equality gate.
            assert result.content_sha256 == entry["resolved_sha256"]
            return result
        if name == "programmatic_metastable":
            path = ROOT / "perovskite-sim/tests/unit/models/test_multivalent_defect_schema.py"
            spec = importlib.util.spec_from_file_location("structured_metastable_data", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            doc = module.MetastableDefectDocument(schema_version=module.METASTABLE_DEFECT_SCHEMA_VERSION,
                  defect_model=module.EXPLICIT_METASTABLE_FROZEN, metastable_defects=(module._metastable_definition(),))
            sources = (SourceDocument("metastable_document", doc.canonical_json().encode()),
                       SourceDocument("preparation_protocol", module._preparation().canonical_json().encode()))
            data = prepare(COMPLEX[0]).to_input().editing_data()
            layer = data["layers"][0]
            for key in ("bulk_defects", "defect_schema_version", "defect_model"):
                layer.pop(key)
            layer["parameters"]["Eg"] = "1.04 eV"
            layer["metastable_document"] = import_metastable_document(sources[0])
            layer["metastable_preparation"] = import_metastable_preparation(sources[1])
            return resolve_device(DeviceInput.model_validate(data), defaults, resources, sources=sources)
        path = ROOT / "perovskite-sim/configs" / name if name == "scaps_mirror_v2.yaml" else FIXTURES / name
        source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
        loader = import_scaps_device if name == "scaps_mirror_v2.yaml" else import_standard_device
        return resolve_device(loader(source, id="device", defaults=defaults), defaults, resources, sources=(source,))

    return prepare


def selection_for(prepared, sources=None):
    sources = structured_parameter_sources(prepared) if sources is None else tuple(sources)
    definitions, overrides = {}, {}
    for source in sources:
        record = source.structured.data
        schema = record.schema
        ref = ModelRef("parameters." + schema.family, schema.version)
        # This necessary metadata context says nothing about physical support.
        rule = CapabilityRule("data_preparation_only", (1,), ("dc",), (schema.source,),
                              backends=("metadata_only",), required_facts=("data_preparation",))
        definitions[ref] = ModelDefinition(ref, schema.family, schema.family, (schema.scope_kind,),
                          {"layer": "node", "interface": "interface_left", "contact": "port", "device": "global"}[schema.scope_kind],
                          schema.parameter_names, (), (), rule, composition="additive",
                          parameter_schema=schema.id, structured_schema=schema)
        families = overrides.setdefault(record.scope, {})
        families.setdefault(schema.family, []).append(ModelChoice(record.local_id, ref, parameter_schema=schema.id))
    registry = ModelRegistry(tuple(definitions.values()))
    context = CapabilityContext(1, "dc", "metadata_only", "maxwell_boltzmann", facts=("data_preparation",))
    per_instance = {source.structured.data.instance_id: source for source in sources}
    selected = select_models(registry, device_topology(prepared), context, {}, {}, overrides,
                             instance_parameter_sources=per_instance)
    return registry, selected, overrides, per_instance


def source_for(prepared, family):
    return next(source for source in structured_parameter_sources(prepared) if source.structured.data.schema.family == family)


def by_id(value, path):
    for item in path:
        value = next(row for row in value if row["id"] == item) if isinstance(value, list) else value.get(item)
    return value


@pytest.mark.parametrize("name", (*COMPLEX, *MONOVALENT))
def test_actual_structured_documents_select_and_export_without_lost_fields(name, prepared_case):
    prepared = prepared_case(name)
    registry, selected, _, _ = selection_for(prepared)
    exported = selected.export()
    assert exported["status"] == "prepared_pending_G2" and not exported["can_execute"]
    assert validate_selection(selected, registry) == ()
    assert list(exported["parameter_schemas"]) == sorted(exported["parameter_schemas"])
    assert [item.id for item in selected.instances] == sorted(item.id for item in selected.instances)
    for instance in selected.instances:
        record = instance.structured
        output = record.export()
        assert output["resolved_reference"]["projection"] == by_id(prepared.to_mapping(), output["resolved_reference"]["path_by_id"])
        assert output["source"]["sha256"] == prepared.content_sha256
        assert set(output["capability_gaps"]) == set(prepared.to_mapping()["capability_gaps"])
        assert not output["can_execute"]
        assert record.data.schema.parameter_names == registry.get(instance.model).parameters
        # Input/projection are separate: referenced energies and extra derived
        # fields must never be relabelled as the same editing schema.
        assert record.data.schema.input_json != record.data.schema.normalized_json
        if record.data.presence == "value":
            original = by_id(prepared.to_input().model_dump(mode="json", exclude_unset=True), output["resolved_reference"]["path_by_id"])
            assert output["input"] == original
    assert ModelRegistry(tuple(reversed(registry.definitions))).export() == registry.export()
    assert json.loads(json.dumps(exported, sort_keys=True)) == exported


def test_multiple_real_species_keep_ids_scopes_and_independent_sources(prepared_case):
    prepared = prepared_case("scaps_mirror_v2.yaml")
    editing = prepared.to_input()
    editing.layers[2].bulk_defects = tuple(item.validated_update({"name": "same display name"}) for item in editing.layers[2].bulk_defects)
    prepared = resolve_device(editing, prepared.defaults, prepared.resources, sources=prepared.sources)
    registry, selected, overrides, sources = selection_for(prepared)
    defects = [item for item in selected.instances if item.structured.data.schema.family == "bulk_defect" and item.scope.ids == ("layer_2",)]
    assert len(defects) == 2 and len({item.model for item in defects}) == 1
    assert len({item.id for item in defects}) == len({item.parameter_source_sha256 for item in defects}) == 2
    source = sources[defects[0].id]
    data = source.structured.export()["input"]
    data["kinetics"]["sigma_n_m2"] = "0 m^2"
    updated = update_structured_source(source, data)
    mixed = dict(sources)
    mixed[defects[0].id] = updated
    with pytest.raises(ValueError, match="distinct prepared-device"):
        select_models(registry, selected.topology, selected.context, {}, {}, overrides, instance_parameter_sources=mixed)
    fresh = resolve_structured_edit(source, data)
    _, rebound, _, _ = selection_for(fresh)
    assert [item.id for item in rebound.instances] == [item.id for item in selected.instances]
    target = next(item for item in rebound.instances if item.id == defects[0].id)
    assert target.structured.export()["normalized_parameters"]["kinetics"]["sigma_n_m2"] == 0.0
    assert source.structured.export()["normalized_parameters"]["kinetics"]["sigma_n_m2"] > 0


def test_units_preserve_input_identity_separately_from_normalized_identity(prepared_case):
    prepared = prepared_case(COMPLEX[0])
    source = source_for(prepared, "bulk_defect")
    data = source.structured.export()["input"]
    data["total_density_m3"] = "2e15 cm^-3"
    data["configuration"]["energy_levels"]["correlation_energies_eV"] = ["150 meV"]
    data["configuration"]["state_degeneracies"] = [1, 2, 1]
    edited = update_structured_source(source, data)
    assert edited.structured.input_sha256 != source.structured.input_sha256
    assert edited.structured.normalized_sha256 == source.structured.normalized_sha256
    assert edited.structured.export()["normalized_parameters"]["total_density_m3"] == 2e21
    normalized = json.loads(edited.structured.data.schema.normalized_json)
    energy = normalized["$defs"]["MultivalentEnergyInput"]["properties"]["correlation_energies_eV"]
    assert energy["items"] == {"type": "number", "unit": "eV"}
    degeneracy = normalized["$defs"]["MultivalentConfigurationInput"]["properties"]["state_degeneracies"]
    assert degeneracy["items"]["type"] == "number" and degeneracy["items"]["exclusiveMinimum"] == 0
    assert source.structured.data.schema.source.sha256 == hashlib.sha256(
        json.dumps(json.loads(source.structured.data.schema.input_json), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("mutation", ("unknown", "boolean", "infinite", "gap", "transition", "variant", "rename", "zero_legs"))
def test_nested_edits_revalidate_closed_variant_and_parent_constraints(mutation, prepared_case):
    source = source_for(prepared_case(COMPLEX[0]), "bulk_defect")
    data = source.structured.export()["input"]
    configuration = data["configuration"]
    if mutation == "unknown":
        configuration["energy_levels"]["profile_file"] = "ignored.csv"
    elif mutation == "boolean":
        data["total_density_m3"] = True
    elif mutation == "infinite":
        configuration["energy_levels"]["correlation_energies_eV"] = [float("inf")]
    elif mutation == "gap":
        configuration["energy_levels"]["first_transition_eV_above_vb"] = "3 eV"
    elif mutation == "transition":
        configuration["charge_states_e"] = [2, 0, -1]
    elif mutation == "variant":
        data = source_for(prepared_case(MONOVALENT[0]), "bulk_defect").structured.export()["input"]
    elif mutation == "rename":
        data["id"] = "different_species"
    else:
        configuration["transition_kinetics"][0].update(sigma_n_m2=0, sigma_p_m2=0)
    before = source.content_sha256
    with pytest.raises(ValueError):
        update_structured_source(source, data)
    assert source.content_sha256 == before


def test_cigs_inheritance_null_clear_and_reinherit_keep_one_default_authority(prepared_case):
    prepared = prepared_case("cigs_graded_optics.yaml")
    data = prepared.to_input().editing_data()
    layer = next(row for row in data["layers"] if row.get("cigs_graded_optics"))
    cigs = layer.pop("cigs_graded_optics")
    layer["material"] = "graded_material"
    data["materials"] = [{"id": "graded_material", "name": "CIGS", "parameters": {}, "cigs_graded_optics": cigs}]
    data["settings"]["graded_optics"] = False
    inherited = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    source = source_for(inherited, "cigs_graded_optics")
    assert source.structured.data.presence == "absent"
    assert source.structured.export()["input"] is None
    assert source.structured.export()["normalized_parameters"]["slices"] == 25
    cleared = update_structured_source(source, None)
    assert cleared.structured.data.presence == "null" and cleared.structured.export()["normalized_parameters"] is None
    restored = update_structured_source(cleared)
    assert restored.structured.content_sha256 == source.structured.content_sha256
    explicit = update_structured_source(source, {"ggi_front": 0, "ggi_back": "0.6", "cgi": "0.9"})
    assert explicit.structured.export()["normalized_parameters"]["ggi_front"] == 0.0
    assert explicit.structured.export()["normalized_parameters"]["slices"] == dict(prepared.defaults.model_defaults("cigs_graded_optics"))["slices"]


def test_tunnelling_false_zero_omission_and_clear_are_distinct(prepared_case):
    source = source_for(prepared_case("wkb_tunnelling_intraband_spike.yaml"), "tunnelling_channels")
    update = update_structured_source(source, {"intraband": {"enabled": False}, "contact": {"barrier_height_eV": 0}})
    exported = update.structured.export()
    assert exported["normalized_parameters"]["intraband"]["enabled"] is False
    assert exported["normalized_parameters"]["contact"]["barrier_height_eV"] == 0.0
    assert set(exported["input"]) == {"intraband", "contact"}
    assert len(exported["normalized_parameters"]) == 5
    assert update_structured_source(source, None).structured.data.presence == "null"
    assert update_structured_source(source).structured.data.presence == "absent"
    # The accepted document uses a null channel as a reset to its supplied
    # defaults; null on the channel's enabled field itself is not allowed.
    reset = update_structured_source(source, {"intraband": None})
    omitted = update_structured_source(source, {})
    assert reset.structured.export()["input"]["intraband"] is None
    assert reset.structured.normalized_sha256 == omitted.structured.normalized_sha256
    assert reset.structured.input_sha256 != omitted.structured.input_sha256
    for invalid in ({"intraband": {"enabled": 0}}, {"intraband": {"enabled": None}}, {"unexpected": {}}):
        with pytest.raises(ValueError):
            update_structured_source(source, invalid)


def test_metastable_protocol_and_legacy_energy_density_remain_data(prepared_case):
    prepared = prepared_case("programmatic_metastable")
    source = source_for(prepared, "metastable_preparation")
    data = source.structured.export()["input"]
    data["voltage_continuation_steps"] = 0
    data["preparation_voltage_V"] = "0 mV"
    updated = update_structured_source(source, data)
    assert updated.structured.export()["normalized_parameters"]["preparation_voltage_V"] == 0.0
    assert "metastable_state_not_prepared_no_stationary_or_dynamic_evaluation" in updated.structured.data.capability_gaps
    assert update_structured_source(source, None).structured.data.presence == "null"
    data["freeze_configuration_during_measurement"] = False
    with pytest.raises(ValueError):
        update_structured_source(source, data)
    trap = source_for(prepared_case("csi_gaussian_bulk_trap_pn_research.yaml"), "bulk_trap_distribution")
    projection = trap.structured.export()["resolved_reference"]["projection"]
    assert projection["density_normalization"] == "integrated_total" and projection["continuous_density_unit"] == "m^-3/eV"
    data = trap.structured.export()["input"]
    data["distribution"] = "single_level"
    data.pop("energy_sigma_eV")
    single = update_structured_source(trap, data)
    assert single.structured.export()["normalized_parameters"]["distribution"] == "single_level"
    assert "energy_sigma_eV" not in single.structured.export()["normalized_parameters"]


def test_topology_canonical_ids_orientation_and_scope_rejections(prepared_case):
    prepared = prepared_case("scaps_mirror_v2.yaml")
    registry, selected, overrides, sources = selection_for(prepared)
    topology = selected.topology
    assert topology.interface_ids == (("interface_1", "layer_1", "layer_2"), ("interface_2", "layer_2", "layer_3"))
    assert topology.contact_sides == (("contact_left", "left"), ("contact_right", "right"))
    for wrong in (replace(topology, device_id="other"), replace(topology, interface_ids=()), replace(topology, contact_sides=())):
        with pytest.raises(ValueError):
            select_models(registry, wrong, selected.context, {}, {}, overrides, instance_parameter_sources=sources)
    with pytest.raises(ValueError):
        replace(topology, interface_ids=(("interface_1", "layer_2", "layer_1"),))
    source = next(item for item in sources.values() if item.structured.data.schema.family == "interface")
    data = source.structured.export()["input"]
    data["left"], data["right"] = data["right"], data["left"]
    with pytest.raises(ValueError):
        update_structured_source(source, data)
    with pytest.raises(ValueError, match="optical-only"):
        structured_parameter_source(prepared, "interface", "interface_0")
    with pytest.raises(ValueError, match="unknown layer"):
        structured_parameter_source(prepared, "bulk_defect", "missing", instance_id="bulk_0")
    editing = prepared.to_input()
    editing.layers[0].bulk_trap_distribution = None
    optical = resolve_device(editing, prepared.defaults, prepared.resources, sources=prepared.sources)
    with pytest.raises(ValueError, match="optical-only layers"):
        structured_parameter_sources(optical)


def test_validated_records_cannot_be_copied_with_forged_resolved_values(prepared_case):
    prepared = prepared_case(COMPLEX[0])
    registry, selected, _, _ = selection_for(prepared)
    source = source_for(prepared, "bulk_defect")
    structured = source.structured
    before = structured.content_sha256
    exported = structured.export()
    exported["normalized_parameters"]["configuration"]["state_degeneracies"][0] = 123
    assert structured.content_sha256 == before
    with pytest.raises(FrozenInstanceError):
        structured.data.normalized_json = "{}"
    with pytest.raises((ValueError, TypeError), match="init=False"):
        replace(structured, data=replace(structured.data, normalized_json="{}"))
    with pytest.raises(ValueError, match="stale"):
        replace(next(item for item in selected.instances if item.structured.data.schema.family == "bulk_defect"), parameter_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="schema/family/scope"):
        replace(registry.get(ModelRef("parameters.bulk_defect", 4)), scopes=("device",))
    with pytest.raises(ValueError, match="complete closed DTO"):
        replace(registry.get(ModelRef("parameters.bulk_defect", 4)), parameters=("id",))
    with pytest.raises(ValueError):
        replace(source, parameter_schema="scalar_layer")


def test_presets_inherit_complete_instances_and_explicitly_clear_families(prepared_case):
    prepared = prepared_case(COMPLEX[0])
    source = source_for(prepared, "bulk_defect")
    registry, selected, _, sources = selection_for(prepared, (source,))
    item, = selected.instances
    choice = ModelChoice(item.local_id, item.model, parameter_schema=item.parameter_schema)
    preset = NamedPreset("actual_m1", 1, "M1 declarations", (("layer", (choice,)),), (source.evidence,))
    base = select_preset(preset, registry, selected.topology, selected.context, {}, instance_parameter_sources=sources)
    assert base.selection.content_sha256 == selected.content_sha256
    document = source.structured.export()["input"]
    document["total_density_m3"] = "2e15 cm^-3"
    updated = update_structured_source(source, document)
    explicit = replace(choice, structured=updated.structured)
    replaced = select_preset(preset, registry, selected.topology, selected.context, {},
                             {item.scope: {"bulk_defect": (explicit,)}}, instance_parameter_sources=sources)
    replaced_item, = replaced.selection.instances
    assert replaced_item.overridden_parameters == registry.get(item.model).parameters
    assert replaced_item.structured.input_sha256 == updated.structured.input_sha256
    empty = select_preset(preset, registry, selected.topology, selected.context, {},
                          {item.scope: {"bulk_defect": ()}}, instance_parameter_sources=sources)
    assert empty.selection.instances == () and not empty.selection.export()["can_execute"]
    for wrong in ({item.scope: {"bulk_defect": None}}, {item.scope: {"missing": ()}}):
        with pytest.raises(ValueError):
            select_preset(preset, registry, selected.topology, selected.context, {}, wrong, instance_parameter_sources=sources)


def test_multiple_species_ownership_conflict_is_checked(prepared_case):
    prepared = prepared_case("scaps_mirror_v2.yaml")
    sources = tuple(item for item in structured_parameter_sources(prepared)
                    if item.structured.data.scope == Scope("layer", ("layer_2",)) and item.structured.data.schema.family == "bulk_defect")
    registry, selected, overrides, per_instance = selection_for(prepared, sources)
    definition, = registry.definitions
    with_ownership = ModelRegistry((replace(definition, owns=(OwnedVariable("occupancy", "1", "node", ("node",)),)),))
    scope = sources[0].structured.data.scope
    choices = overrides[scope]["bulk_defect"]
    same_owner = {scope: {"bulk_defect": tuple(replace(choice, field_bindings=(("occupancy", "one_physical_inventory"),)) for choice in choices)}}
    with pytest.raises(ValueError, match="duplicate ownership"):
        select_models(with_ownership, selected.topology, selected.context, {}, {}, same_owner, instance_parameter_sources=per_instance)


def test_unsupported_versions_and_instance_source_keys_reject(prepared_case):
    prepared = prepared_case(COMPLEX[0])
    registry, selected, overrides, sources = selection_for(prepared)
    source = source_for(prepared, "bulk_defect")
    with pytest.raises(ValueError, match="canonical instance"):
        select_models(registry, selected.topology, selected.context, {}, {}, overrides,
                      instance_parameter_sources={"wrong": source})
    instance = source.structured.data
    with pytest.raises(ValueError, match="unknown model"):
        select_models(registry, selected.topology, selected.context, {}, {},
                      {instance.scope: {"bulk_defect": (ModelChoice(instance.local_id, ModelRef("parameters.bulk_defect", 99), parameter_schema=source.parameter_schema),)}},
                      instance_parameter_sources=sources)
    with pytest.raises(ValueError, match="unavailable structured"):
        structured_parameter_source(prepared, "spectral_profile", "device")


def test_delimiters_in_valid_ids_cannot_alias_parameter_sources(prepared_case):
    prepared = prepared_case(COMPLEX[0])
    data = prepared.to_input().editing_data()
    first, second = deepcopy(data["layers"][0]), deepcopy(data["layers"][0])
    first["id"], first["bulk_defects"][0]["id"] = "layer:a", "b"
    second["id"], second["bulk_defects"][0]["id"] = "layer", "a:b"
    data.update(layers=(first, second), contacts=())
    device = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    records = [item for item in structured_parameter_sources(device) if item.structured.data.schema.family == "bulk_defect"]
    assert len({item.id for item in records}) == 2
    assert len({item.structured.data.instance_id for item in records}) == 2
    _, selected, _, _ = selection_for(device)
    assert len({item.id for item in selected.instances}) == len(selected.instances)

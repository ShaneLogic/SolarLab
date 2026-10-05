"""Actual complex declarations, checked against independent old data loaders.

No test calls a closure, material-array builder, equilibrium or trajectory.
Old model constructors are used only to obtain their existing data contracts.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from solarlab.config.defect_documents import import_metastable_document, import_metastable_preparation
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.defects import LegacyBulkTrapInput
from solarlab.device.inputs import FullLayerInput, NamedMaterialInput
from solarlab.device.tunnelling import TunnellingInput, validate_resolved_tunnelling
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
FIXTURES = ROOT / "perovskite-sim/tests/fixtures/configs"
DEFAULT_PATHS = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                 "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                 "twod/microstructure.py", "models/tandem_config.py")
MODEL_PATHS = ("physics/cigs_optics.py", "models/tunneling_channels.py")
CASES = (
    "scaps_defect_m1_double_donor_p.yaml", "scaps_defect_m2_double_acceptor_n.yaml",
    "scaps_defect_m3_amphoteric_i.yaml", "csi_gaussian_bulk_trap_pn_research.yaml",
    "cigs_graded_optics.yaml", "wkb_resolved_electron_barrier.yaml", "wkb_tunnelling_intraband_spike.yaml",
)


@pytest.fixture(scope="module")
def defaults():
    return read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS),
        model_sources=tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in MODEL_PATHS))


def source(name):
    return SourceDocument(name, (FIXTURES / name).read_bytes())


def imported(name, defaults):
    item = source(name)
    return import_standard_device(item, id="device", defaults=defaults), item


def prepare(input, defaults, sources=()):
    return resolve_device(input, defaults, ResourceLibrary(), sources=sources)


def words(value):
    """Exact scalar comparison, including bool/int/float and low ULP changes."""
    if isinstance(value, dict):
        return {name: words(part) for name, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [words(part) for part in value]
    return [type(value).__name__, value.hex() if type(value) is float else value]


@pytest.mark.parametrize("name", CASES)
def test_actual_documents_roundtrip_and_match_independent_data_loaders(name, defaults):
    from perovskite_sim.models.config_loader import material_params_from_dict
    from perovskite_sim.models.tunneling_channels import tunnelling_channel_document_from_mapping

    input, item = imported(name, defaults)
    output = prepare(input, defaults, (item,))
    assert prepare(output.to_input(), defaults, (item,)).content_sha256 == output.content_sha256
    assert not output.can_execute and output.values["can_execute"] is False
    raw = yaml.safe_load(item.content)
    for row, declaration, layer in zip(raw["layers"], input.layers, output.layers):
        old = material_params_from_dict(row)
        assert words(dict(layer.parameters)) == words({field: getattr(old, field) for field in FullParameterInput.model_fields})
        if "bulk_defects" in row:
            assert len(old.bulk_defects) == len(layer.defects)
            for canonical, prior, resolved in zip(declaration.bulk_defects, old.bulk_defects, layer.defects):
                actual = canonical.normalized_data()
                actual.pop("id")
                assert words(actual) == words(prior.to_dict())
                assert words(resolved.values["configuration"]["transition_energies_eV_above_vb"]) == words(prior.configuration.energy_levels.transition_energies_eV_above_vb)
        if "bulk_trap_distribution" in row:
            declared = declaration.bulk_trap_distribution.normalized_data()
            assert words(declared) == words(asdict(old.bulk_trap_distribution))
            assert layer.bulk_trap_distribution["density_normalization"] == "integrated_total"
            assert layer.bulk_trap_distribution["continuous_density_unit"] == "m^-3/eV"
        if "cigs_graded_optics" in row:
            assert words(dict(layer.material.cigs_graded_optics)) == words(asdict(old.cigs_graded_optics))
            assert layer.material.cigs_source_binding in defaults.evidence
    if "tunnelling_channels" in raw["device"]:
        prior = tunnelling_channel_document_from_mapping(raw["device"])
        assert words(output.to_mapping()["tunnelling_channels"]) == words(prior.to_dict())


@pytest.fixture(scope="module")
def metastable_sources():
    # Reuse the actual programmatic canonical fixture. Loading this model-only
    # test module defines functions; only its two data constructors are called.
    path = ROOT / "perovskite-sim/tests/unit/models/test_multivalent_defect_schema.py"
    spec = importlib.util.spec_from_file_location("metastable_data_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definition = module._metastable_definition()
    protocol = module._preparation()
    document = module.MetastableDefectDocument(
        schema_version=module.METASTABLE_DEFECT_SCHEMA_VERSION,
        defect_model=module.EXPLICIT_METASTABLE_FROZEN,
        metastable_defects=(definition,))
    return (SourceDocument("metastable_document", document.canonical_json().encode()),
            SourceDocument("preparation_protocol", protocol.canonical_json().encode()))


def metastable_input(defaults, metastable_sources):
    input, _ = imported(CASES[0], defaults)
    layer = input.layers[0].editing_data()
    layer.pop("bulk_defects")
    layer.pop("defect_schema_version")
    layer.pop("defect_model")
    layer["parameters"]["Eg"] = "1.04 eV"
    layer["metastable_document"] = import_metastable_document(metastable_sources[0])
    layer["metastable_preparation"] = import_metastable_preparation(metastable_sources[1])
    return input.validated_update({"layers": (FullLayerInput.model_validate(layer),)})


def test_actual_metastable_documents_keep_every_field_and_source(defaults, metastable_sources):
    from perovskite_sim.models.multivalent_defects import MetastableDefectDocument, MetastablePreparationProtocol

    input = metastable_input(defaults, metastable_sources)
    output = prepare(input, defaults, metastable_sources)
    document = input.layers[0].metastable_document.normalized_data()
    for defect in document["metastable_defects"]:
        defect.pop("id")
    original_document = MetastableDefectDocument.from_dict(json.loads(metastable_sources[0].content))
    original_protocol = MetastablePreparationProtocol.from_dict(json.loads(metastable_sources[1].content))
    original_document.validate_band_gap(1.04)
    assert words(document) == words(original_document.to_dict())
    assert words(input.layers[0].metastable_preparation.normalized_data()) == words(original_protocol.to_dict())
    assert prepare(output.to_input(), defaults, metastable_sources).content_sha256 == output.content_sha256
    layer = output.to_mapping()["layers"][0]
    assert layer["metastable_state"] is None
    assert layer["defect_model"] == "explicit_metastable_frozen"
    assert layer["metastable_document"]["metastable_defects"][0]["total_density_m3"] == 3e21
    assert "metastable_measurement_protocol_bytes_not_supplied" in output.values["capability_gaps"]
    assert not output.can_execute


def test_protocol_hash_is_a_reference_until_real_bytes_are_supplied(defaults, metastable_sources):
    input = metastable_input(defaults, metastable_sources)
    measurement = SourceDocument("measurement", b'{"bias_V":[0,0.01]}')
    input.layers[0].metastable_preparation.measurement_protocol_sha256 = measurement.sha256
    unbound = prepare(input, defaults)
    bound = prepare(input, defaults, (*metastable_sources, measurement))
    assert unbound.content_sha256 != bound.content_sha256
    assert bound.values["metastable_measurement_protocol_bindings"][0]["sha256"] == measurement.sha256
    assert "metastable_measurement_protocol_bytes_not_supplied" not in bound.values["capability_gaps"]
    assert bound.values["layers"][0]["metastable_state"] is None


@pytest.mark.parametrize("update", [
    {"charge_states_e": (2, 0, -1)}, {"charge_states_e": (True, 1, 0)},
    {"state_degeneracies": (1, 1, 1)}, {"state_degeneracies": (1, float("nan"), 1)},
    {"family": "single_donor"}, {"transition_kinetics": ()}, {"unknown": 1},
])
def test_multivalent_transitions_and_unknown_fields_reject(update, defaults):
    input, _ = imported(CASES[0], defaults)
    with pytest.raises(ValueError):
        input.layers[0].bulk_defects[0].configuration.validated_update(update)


def test_multivalent_zero_leg_negative_correlation_and_readonly_values(defaults):
    input, _ = imported(CASES[0], defaults)
    defect = input.layers[0].bulk_defects[0]
    defect.configuration.transition_kinetics[0].sigma_n_m2 = 0
    defect.configuration.energy_levels.correlation_energies_eV = ("-0.1 eV",)
    output = prepare(input, defaults)
    actual = output.layers[0].defects[0].values
    assert actual["configuration"]["transition_kinetics"][0]["sigma_n_m2"] == 0
    assert actual["configuration"]["transition_energies_eV_above_vb"] == (0.3, 0.3 - 0.1)
    with pytest.raises(TypeError):
        actual["configuration"]["charge_states_e"][0] = 4
    defect.configuration.transition_kinetics[0].sigma_p_m2 = 0
    with pytest.raises(ValueError, match="capture leg"):
        prepare(input, defaults)
    assert output.layers[0].defects[0].values["configuration"]["transition_kinetics"][0]["sigma_p_m2"] > 0


def test_multivalent_gap_edit_and_copied_resolved_object_revalidate(defaults):
    input, _ = imported(CASES[0], defaults)
    original = prepare(input, defaults)
    with pytest.raises(ValueError, match="outside"):
        replace(original.layers[0].defects[0], band_gap_eV=0.2)
    with pytest.raises(ValueError, match="v4"):
        replace(original.layers[0], defects=())
    with pytest.raises(ValueError, match="unique"):
        replace(original.layers[0], defects=original.layers[0].defects * 2)
    input.layers[0].parameters.Eg = 0.2
    with pytest.raises(ValueError, match="outside"):
        prepare(input, defaults)


def test_duplicate_display_names_are_distinct_stable_instances(defaults):
    input, _ = imported(CASES[0], defaults)
    defect = input.layers[0].bulk_defects[0]
    input.layers[0].bulk_defects = (defect, defect.validated_update({"id": "second"}))
    output = prepare(input, defaults)
    assert [value.values["id"] for value in output.layers[0].defects] == ["bulk_0", "second"]
    assert len({value.values["name"] for value in output.layers[0].defects}) == 1


@pytest.mark.parametrize(("family", "charges", "degeneracies"), [
    ("single_donor", (1, 0), (1, 1)), ("single_acceptor", (0, -1), (1, 1)),
    ("double_donor", (2, 1, 0), (1, 2, 1)), ("double_acceptor", (0, -1, -2), (1, 2, 1)),
    ("amphoteric", (1, 0, -1), (1, 2, 1)), ("custom_multilevel", (2, 1, 0, -1, -2), (1, 4, 6, 4, 1)),
])
def test_all_declared_multivalent_families_match_data_contract(family, charges, degeneracies, defaults):
    from perovskite_sim.models.multivalent_defects import MultivalentDefectConfiguration

    input, _ = imported(CASES[0], defaults)
    original = input.layers[0].bulk_defects[0].configuration
    data = original.normalized_data()
    data.update(family=family, charge_states_e=charges, state_degeneracies=degeneracies,
                transition_kinetics=[data["transition_kinetics"][0]] * (len(charges) - 1))
    data["energy_levels"]["correlation_energies_eV"] = [0.0] * (len(charges) - 2)
    current = type(original).model_validate(data)
    prior = MultivalentDefectConfiguration.from_dict(data)
    prior.validate_band_gap(0.8)
    assert words(current.normalized_data()) == words(prior.to_dict())
    assert words(current.resolved_data(0.8)["transition_energies_eV_above_vb"]) == words(prior.energy_levels.transition_energies_eV_above_vb)


@pytest.mark.parametrize("update", [
    {"total_density_m3": 0}, {"total_density_m3": "1e22 m^-3/eV"},
    {"energy_sigma_eV": 0}, {"energy_sigma_eV": None}, {"sigma_n_m2": False},
    {"charge_transition": "amphoteric"}, {"normalization": "peak"},
])
def test_bulk_trap_density_width_kinetics_and_normalization_are_strict(update, defaults):
    input, _ = imported(CASES[3], defaults)
    with pytest.raises(ValueError):
        input.layers[0].bulk_trap_distribution.validated_update(update)


def test_single_level_bulk_trap_forbids_width_and_inventories_are_exclusive(defaults):
    input, _ = imported(CASES[3], defaults)
    data = input.layers[0].bulk_trap_distribution.editing_data()
    data["distribution"] = "single_level"
    with pytest.raises(ValueError, match="forbids"):
        LegacyBulkTrapInput.model_validate(data)
    data.pop("energy_sigma_eV")
    input.layers[0].bulk_trap_distribution = LegacyBulkTrapInput.model_validate(data)
    original = prepare(input, defaults)
    assert original.layers[0].bulk_trap_distribution["distribution"] == "single_level"
    input.layers[0].bulk_trap_distribution = None
    assert prepare(input, defaults).layers[0].bulk_trap_distribution is None
    explicit, _ = imported(CASES[0], defaults)
    explicit.layers[0].bulk_trap_distribution = LegacyBulkTrapInput.model_validate(data)
    with pytest.raises(ValueError, match="exclusive"):
        prepare(explicit, defaults)


def test_complex_field_units_normalize_before_resolution(defaults, metastable_sources):
    input, _ = imported(CASES[3], defaults)
    original = prepare(input, defaults)
    trap = input.layers[0].bulk_trap_distribution
    trap.total_density_m3 = "1e16 cm^-3"
    trap.sigma_n_m2 = "1e-15 cm^2"
    trap.energy_sigma_eV = "80 meV"
    assert words(dict(prepare(input, defaults).layers[0].bulk_trap_distribution)) == words(dict(original.layers[0].bulk_trap_distribution))
    meta = metastable_input(defaults, metastable_sources)
    original_meta = prepare(meta, defaults)
    conversion = meta.layers[0].metastable_document.metastable_defects[0].conversion_kinetics
    conversion.capture_n_m3_s = "2e-8 cm^3/s"
    conversion.phonon_frequency_Hz = "1e10 kHz"
    assert prepare(meta, defaults).to_mapping()["layers"][0]["metastable_document"] == original_meta.to_mapping()["layers"][0]["metastable_document"]


@pytest.mark.parametrize("update", [{"slices": True}, {"slices": 0}, {"slices": 513}, {"kk_quadrature_order": 47},
                                   {"kk_quadrature_order": 2049}, {"cgi": 0.74}, {"ggi_front": float("inf")},
                                   {"ggi_back": 1.1}, {"model": None}, {"unknown": 0}])
def test_cigs_ranges_and_explicit_controls_validate(update, defaults):
    input, _ = imported(CASES[4], defaults)
    with pytest.raises(ValueError):
        input.layers[-1].cigs_graded_optics.validated_update(update)


def test_cigs_defaults_material_inheritance_clear_and_source_identity(defaults):
    input, _ = imported(CASES[4], defaults)
    layer = input.layers[-1]
    cigs = layer.cigs_graded_optics
    data = {name: value for name, value in cigs.editing_data().items() if name in {"ggi_front", "ggi_back", "cgi"}}
    layer.cigs_graded_optics = type(cigs).model_validate(data)
    first = prepare(input, defaults)
    assert dict(first.layers[-1].material.cigs_graded_optics) == cigs.normalized_data()
    original_sha = first.layers[-1].material.content_sha256
    model_sources = tuple(SourceDocument(name, (LEGACY / name).read_bytes() + b'\n# source identity control\n') for name in MODEL_PATHS)
    changed_defaults = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS), model_sources=model_sources)
    assert changed_defaults.complex_defaults == defaults.complex_defaults
    assert prepare(input, changed_defaults).layers[-1].material.content_sha256 != original_sha
    input.materials = (NamedMaterialInput(id="CIGS", name="CIGS", parameters=FullParameterInput(), cigs_graded_optics=cigs),)
    data = layer.editing_data()
    data.pop("cigs_graded_optics")
    data["material"] = "CIGS"
    input.layers = (*input.layers[:-1], FullLayerInput.model_validate(data))
    assert prepare(input, defaults).layers[-1].material.cigs_graded_optics is not None
    input.layers[-1].cigs_graded_optics = None
    input.settings.graded_optics = False
    assert prepare(input, defaults).layers[-1].material.cigs_graded_optics is None
    assert first.layers[-1].material.content_sha256 == original_sha


@pytest.mark.parametrize("change", ["legacy", "no_grade", "no_endpoint", "wrong_role", "no_model", "n_optical"])
def test_cigs_activation_and_resource_conflicts_are_precise(change, defaults):
    input, _ = imported(CASES[4], defaults)
    if change == "legacy":
        input.settings.mode = "legacy"
    elif change == "no_grade":
        input.settings.band_grading = False
    elif change == "no_endpoint":
        input.layers[-1].parameters.Eg_back = None
        input.layers[-1].parameters.chi_back = None
    elif change == "wrong_role":
        input.layers[-1].role = "ETL"
    elif change == "no_model":
        input.layers[-1].cigs_graded_optics = None
    else:
        input.layers[-1].parameters.n_optical = 2
    with pytest.raises(ValueError, match="CIGS|cigs|graded_optics"):
        prepare(input, defaults)


def test_inert_cigs_document_does_not_claim_optical_activation(defaults):
    input, _ = imported(CASES[4], defaults)
    input.settings.graded_optics = False
    input.settings.band_grading = False
    input.settings.mode = "legacy"
    input.layers[-1].cigs_graded_optics.ggi_front = 0
    output = prepare(input, defaults)
    assert output.layers[-1].material.cigs_graded_optics is not None
    assert output.layers[-1].material.cigs_graded_optics["ggi_front"] == 0
    assert output.values["settings"]["graded_optics"] is False
    assert not output.can_execute


def test_tunnelling_defaults_false_zero_clear_and_independent_channels(defaults):
    input, _ = imported(CASES[5], defaults)
    input.tunnelling_channels = TunnellingInput()
    off = prepare(input, defaults)
    channels = off.to_mapping()["tunnelling_channels"]
    for name in ("band_to_band", "intraband", "interface_defect_assisted", "contact"):
        assert channels[name]["enabled"] is False
    assert channels["contact"]["barrier_height_eV"] == 0
    input.tunnelling_channels = TunnellingInput.model_validate({"intraband": {"enabled": True, "carrier": "electron"}})
    active = prepare(input, defaults)
    assert active.values["tunnelling_channels"]["intraband"]["enabled"] is True
    input.tunnelling_channels.intraband.enabled = False
    assert prepare(input, defaults).values["tunnelling_channels"]["intraband"]["enabled"] is False
    input.tunnelling_channels.intraband = None
    assert prepare(input, defaults).values["tunnelling_channels"]["intraband"]["enabled"] is False
    input.tunnelling_channels = None
    assert "tunnelling_channels" not in prepare(input, defaults).to_mapping()


@pytest.mark.parametrize("data", [
    {"contact": {"enabled": True, "barrier_height_eV": 0}},
    {"intraband": {"enabled": 1}}, {"intraband": {"enabled": None}},
    {"intraband": {"energy_quadrature_order": 3}}, {"intraband": {"electron_effective_mass_rel": False}},
    {"interface_defect_assisted": {"requires_explicit_occupancy": False}},
    {"interface_defect_assisted": {"requires_explicit_occupancy": 1}},
    {"band_to_band": {"minimum_field_V_m": float("nan")}}, {"typo": {}},
])
def test_channel_invalid_types_and_controls_fail(data):
    with pytest.raises(ValueError):
        TunnellingInput.model_validate(data)


def test_copied_tunnelling_payload_cannot_omit_validation(defaults):
    input, _ = imported(CASES[5], defaults)
    data = prepare(input, defaults).to_mapping()["tunnelling_channels"]
    data["contact"].pop("barrier_height_eV")
    with pytest.raises(ValueError, match="incomplete"):
        validate_resolved_tunnelling(data)
    input.tunnelling_channels = TunnellingInput.model_validate({"contact": {"enabled": True}})
    with pytest.raises(ValueError, match="positive barrier"):
        prepare(input, defaults)


def test_valid_unqualified_channels_are_preserved_with_explicit_gaps(defaults):
    input, _ = imported(CASES[5], defaults)
    input.tunnelling_channels = TunnellingInput.model_validate({"intraband": {"enabled": True, "carrier": "hole"},
        "contact": {"enabled": True, "barrier_height_eV": "200 meV"}, "band_to_band": {"enabled": True},
        "interface_defect_assisted": {"enabled": True}})
    output = prepare(input, defaults)
    for name in ("intraband", "contact", "band_to_band", "interface_defect_assisted"):
        assert "device_tunnelling_coupling_unavailable:" + name in output.values["capability_gaps"]
        assert output.values["tunnelling_channels"][name]["enabled"] is True
    assert not output.can_execute


@pytest.mark.parametrize("change", ["barrier", "charge", "state_index", "gap", "second_inventory", "protocol_without_document"])
def test_metastable_relations_and_ownership_reject(change, defaults, metastable_sources):
    input = metastable_input(defaults, metastable_sources)
    layer = input.layers[0]
    with pytest.raises(ValueError):
        if change == "barrier":
            layer.metastable_document.metastable_defects[0].conversion_kinetics.electron_emission_activation_eV = 0.8
        elif change == "charge":
            layer.metastable_document.metastable_defects[0].donor_conversion_state_index = 1
        elif change == "state_index":
            layer.metastable_document.metastable_defects[0].donor_conversion_state_index = 2
        elif change == "gap":
            layer.parameters.Eg = 0.5
        elif change == "second_inventory":
            original, _ = imported(CASES[0], defaults)
            layer.bulk_defects = original.layers[0].bulk_defects
        else:
            layer.metastable_document = None
        prepare(input, defaults)


@pytest.mark.parametrize("update", [
    {"preparation_limit": "finite_time"}, {"freeze_configuration_during_measurement": False},
    {"freeze_configuration_during_measurement": 1}, {"preparation_temperature_K": False},
    {"measurement_protocol_sha256": "not-a-sha"}, {"voltage_continuation_steps": -1}, {"unknown": 0},
])
def test_metastable_protocol_revalidates_copies(update, metastable_sources):
    protocol = import_metastable_preparation(metastable_sources[1])
    with pytest.raises(ValueError):
        protocol.model_copy(update=update)


def test_nested_metastable_edits_change_identity_without_mutable_alias(defaults, metastable_sources):
    input = metastable_input(defaults, metastable_sources)
    before = prepare(input, defaults)
    copied = input.model_copy()
    copied.layers[0].metastable_preparation.numerics.initial_donor_fraction_guess = 0
    copied.layers[0].metastable_preparation.preparation_illumination_suns = 0
    after = prepare(copied, defaults)
    assert before.content_sha256 != after.content_sha256
    assert input.layers[0].metastable_preparation.numerics.initial_donor_fraction_guess == 0.5
    assert after.values["layers"][0]["metastable_preparation"]["numerics"]["initial_donor_fraction_guess"] == 0
    with pytest.raises(TypeError):
        before.layers[0].metastable_document["metastable_defects"][0]["donor_configuration"]["charge_states_e"][0] = 3
    damaged = json.loads(before.input_json)
    damaged["layers"][0]["metastable_preparation"]["numerics"]["clamping_factor"] = 2
    with pytest.raises(ValueError):
        replace(before, input_json=json.dumps(damaged).encode())


def test_data_catalog_roundtrip_effective_values_and_source_bytes(defaults):
    assert DefaultCatalog.from_mapping(defaults.to_mapping()).content_sha256 == defaults.content_sha256
    data = defaults.to_mapping()
    data["complex_defaults"]["intraband"]["electron_effective_mass_rel"] = 0.3
    changed = DefaultCatalog.from_mapping(data)
    assert changed.evidence == defaults.evidence and changed.content_sha256 != defaults.content_sha256
    data["complex_defaults"]["intraband"]["enabled"] = True
    with pytest.raises(ValueError, match="enable"):
        DefaultCatalog.from_mapping(data)


def test_runtime_resolution_uses_only_serialized_injected_data(tmp_path, defaults):
    input, item = imported(CASES[4], defaults)
    catalog_path = tmp_path / "catalog.json"
    input_path = tmp_path / "input.json"
    catalog_path.write_text(json.dumps(defaults.to_mapping()))
    input_path.write_text(input.model_dump_json(exclude_unset=True))
    code = '''import json,sys
from pathlib import Path
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.config.resolve_device import resolve_device
from solarlab.materials.resources import ResourceLibrary
catalog=DefaultCatalog.from_mapping(json.loads(Path(sys.argv[1]).read_text()))
document=DeviceInput.model_validate_json(Path(sys.argv[2]).read_text())
output=resolve_device(document,catalog,ResourceLibrary())
assert not output.can_execute
assert not any(name.startswith(("perovskite_sim","solarlab_research","solarlab_server","backend")) for name in sys.modules)
assert "solarlab.config.legacy_defaults" not in sys.modules
print(output.content_sha256)
'''
    result = subprocess.run([sys.executable, "-c", code, str(catalog_path), str(input_path)], text=True, capture_output=True, check=True, cwd=tmp_path)
    assert result.stdout.strip() == prepare(input, defaults).content_sha256


def test_legacy_device_route_does_not_acquire_metastable_fields(defaults, metastable_sources):
    raw = yaml.safe_load(source(CASES[0]).content)
    raw["layers"][0]["metastable_document"] = json.loads(metastable_sources[0].content)
    with pytest.raises(ValueError, match="unknown"):
        import_standard_device(SourceDocument("unavailable_route", json.dumps(raw).encode()), id="bad", defaults=defaults)


def test_real_yaml_merge_precedence_and_edit_alias_independence(defaults):
    text = "first: &a {mu_n: 1, optical_material: CIGS}\nsecond: &b {mu_n: 3, eps_r: 4}\nmerged: {mu_n: 0, <<: [*a, *b]}\ninherited: {<<: [*a, *b]}"
    assert load_yaml_mapping(text) == yaml.safe_load(text)
    assert load_yaml_mapping(text)["merged"] == {"mu_n": 0, "optical_material": "CIGS", "eps_r": 4}
    assert load_yaml_mapping(text)["inherited"]["mu_n"] == 1
    shallow = "template: &t {parameters: {mu_n: 1, eps_r: 2}}\nlayer: {<<: *t, parameters: {mu_n: 0}}"
    assert load_yaml_mapping(shallow)["layer"]["parameters"] == {"mu_n": 0}
    input, _ = imported(CASES[5], defaults)
    original = input.layers[0].parameters.mu_n
    input.layers[1].parameters.mu_n = 0
    assert input.layers[0].parameters.mu_n == original
    assert prepare(input, defaults).layers[1].material.values["mu_n"] == 0


@pytest.mark.parametrize("text", [
    "a: &a {x: 1, x: 2}\nb: {<<: *a}", "a: &a {x: 1}\nb: {<<: *a, <<: *a}",
    "a: {<<: 2}", "a: {<<: [1, 2]}", "a: &a {<<: *a}",
    "a: &a {x: 1}\nb: {<<: *a, x: 2, x: 3}",
])
def test_merge_does_not_hide_duplicates_bad_shapes_or_cycles(text):
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_yaml_mapping(text)

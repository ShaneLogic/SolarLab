"""Legacy input history and data-only reverse wire/dry-run preparation."""
from __future__ import annotations

from decimal import Decimal
import builtins
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess

import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.legacy_wire import LEGACY_WIRE_VERSION, prepare_legacy_wire
from solarlab.config.legacy_fields import (
    INLINE_LOADER_BINDING, JV_HINT_CONSUMER_BINDINGS, STANDARD_LOADER_BINDING,
)
from solarlab.config.resolve_device import resolve_device
from solarlab.config.schema import export_configuration_schema
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.inputs import DeviceInput
from solarlab.device.settings import DeviceSettingsInput
from solarlab.experiments.jv.declarations import JVSweepHintsInput, JVInput
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.jv.preparation import prepare_jv_experiment
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
CASES = {
    "tpv": "perovskite-sim/tests/fixtures/configs/tpv_physical_reference.yaml",
    "driftfusion": "perovskite-sim/tests/fixtures/configs/driftfusion_benchmark_tmm.yaml",
    "ionmonger": "perovskite-sim/tests/fixtures/configs/ionmonger_benchmark_tmm.yaml",
    "calado": "perovskite-sim/configs/calado2016_ion_sweep.yaml",
}
DEFAULTS = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
            "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
            "twod/microstructure.py", "models/tandem_config.py")


@pytest.fixture(scope="module")
def defaults():
    return read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULTS),
        jv_sources=(SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes()),
                    *(SourceDocument(name, (LEGACY / name).read_bytes()) for name in (
                        "experiments/jv_sweep.py", "experiments/waveform_jv.py",
                        "experiments/dark_jv.py", "experiments/protocol.py"))))


@pytest.fixture(scope="module")
def resources():
    return ResourceLibrary(tuple(ResourceTable(p.stem, "nk", SourceDocument(str(p.relative_to(ROOT)), p.read_bytes()))
                                 for p in sorted((LEGACY / "data/nk").glob("*.csv"))))


def raw_json(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: raw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [raw_json(item) for item in value]
    return value


def source(case, edit=None):
    path = ROOT / CASES[case]
    data = path.read_bytes()
    if edit is not None:
        raw = raw_json(load_yaml_mapping(data))
        edit(raw)
        data = json.dumps(raw, allow_nan=False).encode()
    return SourceDocument(CASES[case], data)


def prepared(case, defaults, resources, edit=None):
    original = source(case, edit)
    input = import_standard_device(original, id=case, defaults=defaults)
    return resolve_device(input, defaults, resources, sources=(original,))


def diagnostic(value, code):
    return next(item for item in value.to_mapping()["diagnostics"] if item["code"] == code)


@pytest.mark.parametrize("case", CASES)
def test_original_four_import_resolve_and_reopen_exact_source(case, defaults, resources):
    output = prepared(case, defaults, resources)
    assert output.sources[0].content == (ROOT / CASES[case]).read_bytes()
    assert output.sources[0].sha256 == hashlib.sha256(output.sources[0].content).hexdigest()
    assert not output.can_execute
    reopened = resolve_device(output.to_input(), defaults, resources, sources=output.sources)
    assert reopened.input_json == output.input_json
    assert reopened.content_sha256 == output.content_sha256
    assert reopened.to_mapping() == output.to_mapping()
    assert output.to_mapping()["settings"]["T"] == 300


@pytest.mark.parametrize("ignored", [0, -0.0, False, None, "900 K", "not an effective temperature"])
def test_ignored_temperature_never_becomes_T(ignored, defaults, resources, tmp_path):
    from perovskite_sim.models.config_loader import load_device_from_yaml

    output = prepared("tpv", defaults, resources, lambda raw: raw["device"].update(temperature=ignored, T=310))
    historical = tmp_path / "temperature.yaml"
    historical.write_bytes(output.sources[0].content)
    # This is only the actual data loader, not a material-array or state builder.
    assert load_device_from_yaml(str(historical)).T == 310
    record = diagnostic(output, "legacy_standard_ignored_temperature")
    assert record["effective_setting_T"] == 310 and record["T_origin"] == "input.settings.T"
    assert record["historical_effective"] == "ignored; only device.T is read"
    value = output.to_input().legacy_fields.temperature
    assert type(value) is type(ignored) and value == ignored
    if type(ignored) is float:
        assert value.hex() == ignored.hex()
    assert "temperature" not in output.to_mapping()["settings"]
    assert record["source_binding"]["bytes_supplied"]


def test_temperature_omission_and_effective_T_validation_remain_distinct(defaults, resources):
    value = prepared("tpv", defaults, resources)
    assert "T" not in value.to_input().settings.editing_data()
    assert diagnostic(value, "legacy_standard_ignored_temperature")["T_origin"] == "default_catalog.device.T"
    omitted = prepared("tpv", defaults, resources, lambda raw: raw["device"].pop("temperature"))
    assert "legacy_fields" not in omitted.to_input().editing_data()
    assert omitted.to_mapping()["settings"] == value.to_mapping()["settings"]
    for invalid in (0, None, "300 s"):
        with pytest.raises(ValueError):
            prepared("tpv", defaults, resources, lambda raw: raw["device"].update(T=invalid))
    with pytest.raises(ValueError, match="temperature"):
        DeviceSettingsInput.model_validate({"temperature": 300})
    with pytest.raises(ValueError, match="source_sha256"):
        resolve_device(value.to_input(), defaults, resources,
                       sources=(SourceDocument(value.sources[0].id, value.sources[0].content + b"\n"),))
    tampered = value.to_input()
    tampered.legacy_fields.temperature = 999
    with pytest.raises(ValueError, match="original source bytes"):
        resolve_device(tampered, defaults, resources, sources=value.sources)
    altered = value.to_input().editing_data()
    altered["legacy_fields"]["schema_version"] = "corrected-temperature-alias"
    with pytest.raises(ValueError, match="schema_version"):
        DeviceInput.model_validate(altered)


@pytest.mark.parametrize("case,magnitude", [("driftfusion", 1000), ("ionmonger", 300000)])
def test_tmm_slots_follow_full_raw_layers_and_actual_loader(case, magnitude, defaults, resources):
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_interfaces

    output = prepared(case, defaults, resources)
    old = load_device_from_yaml(str(ROOT / CASES[case]))
    expected = ((magnitude, .1), (.1, magnitude), (0., 0.))
    assert old.interfaces == expected
    assert electrical_interfaces(old) == expected[1:]
    assert len(output.layers) == 4 and output.layers[0].role == "substrate"
    rows = [interface.to_mapping() for interface in output.interfaces]
    assert [(row["v_n"], row["v_p"]) for row in rows] == list(expected)
    assert [row["electrical"] for row in rows] == [False, True, True]
    assert [(row["left"], row["right"]) for row in rows] == [("layer_0", "layer_1"), ("layer_1", "layer_2"), ("layer_2", "layer_3")]
    assert output.to_mapping()["contacts"][0]["layer"] == "layer_1"
    note = diagnostic(output, "legacy_standard_interface_layout")
    assert len(note["declared"]["interfaces"]) == 2
    assert note["historical_effective"][-1]["origin"] == "legacy_trailing_zero_padding"
    # An explicit canonical edit does not rewrite the original raw list history.
    input = output.to_input()
    input.interfaces[1].v_n = "2 cm/s"
    changed = resolve_device(input, defaults, resources, sources=output.sources)
    assert changed.interfaces[1].values["v_n"] == .02
    assert changed.to_input().legacy_fields == output.to_input().legacy_fields


@pytest.mark.parametrize("declaration", [None, [], [[0, -0.0]], [["2 cm/s", "0 m/s"]]])
def test_short_pairs_null_empty_zero_units_remain_declared(declaration, defaults, resources):
    output = prepared("driftfusion", defaults, resources, lambda raw: raw["device"].update(interfaces=declaration))
    retained = json.loads(output.input_json)["legacy_fields"]["interfaces"]
    assert retained == declaration
    assert type(retained) is type(declaration)
    if declaration and declaration[0] == [0, -0.0]:
        assert math.copysign(1, retained[0][1]) == -1
    if declaration:
        assert output.interfaces[-1].values["v_n"] == 0
        assert output.interfaces[0].values["v_n"] == (.02 if declaration == [["2 cm/s", "0 m/s"]] else 0)
    else:
        assert all("v_n" not in interface.values for interface in output.interfaces)
        note = diagnostic(output, "legacy_standard_interface_layout")
        assert note["historical_effective"] == [] and note["historical_return_shape"] == "empty_tuples"


@pytest.mark.parametrize("invalid", [[[1]], [[False, 0]], [["1 s", 0]], [[1, 2]] * 4])
def test_malformed_or_unused_interface_input_rejects_without_reindexing(invalid, defaults, resources):
    with pytest.raises(ValueError):
        prepared("driftfusion", defaults, resources, lambda raw: raw["device"].update(interfaces=invalid))


def test_short_pairs_and_defect_override_retain_independent_raw_slots(defaults, resources, tmp_path):
    from perovskite_sim.models.config_loader import load_device_from_yaml

    def edit(raw):
        raw["device"]["interfaces"] = [[4, 5]]
        raw["device"]["interface_defects"] = [None, dict(
            sigma_n_cm2=1e-15, sigma_p_cm2=2e-15, v_th_cm_s=1e7,
            N_t_cm2=1e10, E_t_eV_below_cb=.4)]
    output = prepared("tpv", defaults, resources, edit)
    path = tmp_path / "defect.yaml"
    path.write_bytes(output.sources[0].content)
    old = load_device_from_yaml(str(path))
    assert old.interfaces[0] == (4, 5)
    assert old.interfaces[1] == pytest.approx((1, 2))
    assert output.interfaces[0].values["v_n"] == 4
    # The canonical declaration retains both authorities and labels precedence;
    # it never overwrites the original bare-pair history with capture values.
    assert output.interfaces[1].values["v_n"] == 0
    assert output.interfaces[1].values["microscopic_capture_velocities_m_s"] == pytest.approx((1, 2))
    note = diagnostic(output, "legacy_standard_interface_layout")
    assert note["declared"]["interface_defect_count"] == 2
    assert note["historical_effective"][1]["pair_before_defect_override"] == [0, 0]


@pytest.mark.parametrize("path", [
    "dynamic_interface_defect_ion_transient.yaml",
    "dynamic_interface_defect_ion_transient_absorber_only.yaml",
    "interface_charge_jv_research.yaml", "interface_charge_research.yaml",
    "solarscale_nip_band_aligned_iface.yaml", "twod/combined_mobile_interface_research.yaml",
])
def test_full_defect_inventory_preserves_bare_pair_omission(path, defaults, resources):
    file = ROOT / "perovskite-sim/tests/fixtures/configs" / path
    source = SourceDocument(str(file.relative_to(ROOT)), file.read_bytes())
    raw = load_yaml_mapping(source.content)
    assert "interfaces" not in raw["device"]
    assert len(raw["device"]["interface_defects"]) == len(raw["layers"]) - 1
    input = import_standard_device(source, id="full_defects", defaults=defaults)
    value = resolve_device(input, defaults, resources, sources=(source,))
    assert all(not {"v_n", "v_p"} & row.model_fields_set for row in value.to_input().interfaces)
    assert all("v_n" not in row.values and "v_p" not in row.values for row in value.interfaces)
    original_capture = value.interfaces[0].values.get("microscopic_capture_velocities_m_s")
    identities = {value.content_sha256}
    for declared in (0, None):
        edited = value.to_input()
        edited.interfaces[0].v_n = declared
        edited.interfaces[0].v_p = declared
        prepared = resolve_device(edited, defaults, resources, sources=(source,))
        assert {"v_n", "v_p"} <= prepared.to_input().interfaces[0].model_fields_set
        assert prepared.interfaces[0].values["v_n"] == declared
        assert prepared.interfaces[0].values.get("microscopic_capture_velocities_m_s") == original_capture
        identities.add(prepared.content_sha256)
    assert len(identities) == 3  # Omission, explicit zero and clearing stay distinct.


def test_complete_calado_history_is_advisory_until_explicit_composition(defaults, resources):
    device = prepared("calado", defaults, resources)
    raw = load_yaml_mapping(device.sources[0].content)["simulation_hints"]["jv_sweep"]
    hints = device.to_input().simulation_hints.jv_sweep
    assert json.loads(device.input_json)["simulation_hints"]["jv_sweep"] == raw_json(raw)
    assert diagnostic(device, "legacy_jv_sweep_advisory")["implicit_experiment_created"] is False
    assert diagnostic(device, "legacy_jv_sweep_advisory")["physical_state_created"] is False
    def experiment(fields):
        return prepare_jv_experiment(JVExperimentInput.model_validate({
            "schema_version": "solarlab.experiment-preparation.v1", "id": "calado_history",
            "device": device.to_input().editing_data(), "experiment": {"kind": "jv", **fields},
        }), defaults, resources, sources=device.sources)
    ordinary = experiment({"n_points": 4, "V_max": 1.0}).to_mapping()["experiment"]
    assert ordinary["controls"]["n_points"] == 4 and ordinary["waveform"] is None
    assert ordinary["history_kind"] == "staircase_forward_reverse"
    explicit = experiment(hints.editing_data())
    values = explicit.to_mapping()["experiment"]
    assert values["controls"]["N_grid"] == 60 and values["controls"]["n_points"] == 111
    assert values["controls"]["v_rate"] == .04 and values["controls"]["V_max"] == 1.2
    assert values["numerical_controls"] == {"rtol": .0001, "atol_m3": 1.0}
    assert defaults.experiment_defaults_for("jv_waveform_controls")["atol_m3"] == 100
    protocol = values["protocol"]["declaration"]
    assert [row["phase"] for row in protocol["illumination_history"]] == [
        "dark_seed_at_0V", "dark_prebias_at_scan_start", "forward_start_dwell",
        "forward_continuous_ramp", "turnaround_at_scan_stop", "reverse_start_dwell", "reverse_continuous_ramp"]
    assert [row["duration_s"] for row in protocol["illumination_history"]] == pytest.approx([120, 30, .5, 55, 3, .5, 55])
    assert [row["condition"] for row in protocol["illumination_history"]] == ["dark", "dark", "baseline", "baseline", "dark", "baseline", "baseline"]
    samples = protocol["sampling"]["values"]
    assert len(samples) == 222 and samples[:111] == list(reversed(samples[111:]))
    assert samples[0] == -1 and samples[110] == 1.2
    assert values["waveform"]["uniform_generation_rate_m3_s"] == 2.5e27
    assert values["state_history"]["continuation"] == "one_state_through_forward_turnaround_reverse"
    assert values["state_history"]["generated"] is False and not explicit.can_execute
    reopened = prepare_jv_experiment(explicit.to_input(), defaults, resources, sources=device.sources)
    assert reopened.input_json == explicit.input_json and reopened.content_sha256 == explicit.content_sha256


def test_hint_units_zero_null_omission_and_ion_inventory_are_independent(defaults, resources):
    def edit(raw):
        hints = raw["simulation_hints"]["jv_sweep"]
        hints["v_rate"] = "40 mV/s"
        hints["waveform"].update(dark_seed_s=0, dark_prep_s=-0.0, turnaround_s="0 s",
                                 turnaround_dark=False, uniform_generation_rate_m3_s=0)
        raw["layers"][1]["D_ion"] = 0
    output = prepared("calado", defaults, resources, edit)
    hints = output.to_input().simulation_hints.jv_sweep
    assert hints.editing_data()["v_rate"] == "40 mV/s"
    assert hints.normalized_data()["v_rate"] == .04
    assert hints.waveform.dark_seed_s == 0 and math.copysign(1, hints.waveform.dark_prep_s) == -1
    assert hints.waveform.turnaround_dark is False and hints.waveform.uniform_generation_rate_m3_s == 0
    assert dict(output.layers[1].parameters)["D_ion"] == 0
    assert dict(output.layers[1].parameters)["P0"] == 1e25
    depleted = prepared("calado", defaults, resources, lambda raw: raw["layers"][1].update(P0=0))
    assert dict(depleted.layers[1].parameters)["D_ion"] == 2.585e-18
    assert dict(depleted.layers[1].parameters)["P0"] == 0
    assert JVSweepHintsInput.model_validate({}).editing_data() == {}
    null = JVSweepHintsInput.model_validate({"waveform": None, "waveform_controls": None, "V_max": None})
    assert null.editing_data() == {"waveform": None, "waveform_controls": None, "V_max": None}
    unit_null = hints.waveform.validated_update({"uniform_generation_rate_m3_s": None})
    assert unit_null.editing_data()["uniform_generation_rate_m3_s"] is None


@pytest.mark.parametrize("path,value", [(("v_rate",), 0), (("v_rate",), "1 K"), (("N_grid",), False),
    (("waveform", "dark_seed_s"), -1), (("waveform", "turnaround_dark"), 0),
    (("waveform", "schema_version"), True), (("waveform_controls", "atol_m3"), 0),
    (("unknown_evidence",), "retain by error, never silently delete")])
def test_invalid_history_fields_are_reported_without_default_repair(path, value, defaults, resources):
    def edit(raw):
        target = raw["simulation_hints"]["jv_sweep"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    original = source("calado", edit)
    snapshot = original.content
    with pytest.raises(ValueError, match=path[-1]):
        import_standard_device(original, id="invalid", defaults=defaults)
    assert original.content == snapshot


def test_source_bindings_and_generated_hints_metadata_have_actual_authorities():
    for binding in (STANDARD_LOADER_BINDING, INLINE_LOADER_BINDING, *JV_HINT_CONSUMER_BINDINGS):
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    device = export_configuration_schema()["dto_schemas"]["DeviceInput"]["schema"]
    fields = device["$defs"]["JVSweepHintsInput"]["properties"]
    assert fields["v_rate"] == JVInput.model_json_schema()["properties"]["v_rate"]
    assert fields["v_rate"]["unit"] == "V/s"
    assert "kind" not in fields and "request_api" not in fields
    assert "temperature" not in device["$defs"]["DeviceSettingsInput"]["properties"]
    assert device["$defs"]["LegacyDeviceFieldsInput"]["properties"]["schema_version"]["const"] == "solarlab.standard-loader-fields.v1"


@pytest.fixture
def wire_guard(monkeypatch):
    """Only new data-only tests use this guard; old loader tests stay intact."""
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy", "sksundae")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture(scope="module")
def wire_catalog():
    from solarlab.reproducibility.preparation import prepare_configuration, read_preparation_context
    from solarlab.reproducibility.registry import RegistryDocument, load_registry, mapping
    from solarlab.reproducibility.sources import SourceSet, read_sources
    path = ROOT / "reproducibility/ConfigBenchmarkMatrixV2.yaml"
    matrix_source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    specification = RegistryDocument.model_validate(mapping(matrix_source))
    documentary_path = "tests/refactor/test_legacy_input_migration.py"
    historical, = (ref for ref in specification.sources if ref.path == documentary_path)
    # This ONE source is documentary input, selected before unchanged SHA
    # verification. Matching current bytes also work in shallow CI checkouts;
    # development edits use the explicitly declared historical Git object.
    blob = (ROOT / historical.path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != historical.sha256:
        blob = subprocess.run(["git", "show", historical.source_commit + ":" + historical.path],
                              cwd=ROOT, capture_output=True, check=True, timeout=5).stdout
    current = read_sources(ROOT, tuple(ref for ref in specification.sources if ref != historical))
    bundle = SourceSet(specification.sources, (*current.documents, SourceDocument(historical.id, blob)))
    matrix = load_registry(matrix_source, bundle.documents)
    catalog, tables = read_preparation_context(matrix)
    cases = []
    for entry in matrix.spec.entries:
        if entry.section != "configs":
            continue
        prepared = prepare_configuration(matrix, entry.id, defaults=catalog, resources=tables).prepared
        assert prepared is not None and not prepared.can_execute
        original = matrix.sources.at_path(matrix.spec.original_path_root + "/" + entry.original_key)
        references = {}
        if prepared.to_input().schema_version == "solarlab.tandem-preparation.v1":
            parent = PurePosixPath(matrix.sources.reference(original.id).path).parent
            raw = load_yaml_mapping(original.content)
            for side in ("top_cell", "bottom_cell"):
                ref = raw["tandem"][side]
                references[ref] = matrix.sources.at_path(posixpath.normpath(str(parent / ref)))
        cases.append((entry.original_key, prepared, original, references))
    return catalog, cases


def test_wire_all_55_catalog_sources_roundtrip_without_changing_originals(wire_catalog, wire_guard, record_property):
    from solarlab.config.scaps_input import import_scaps_device
    from solarlab.config.tandem_input import import_tandem
    catalog, cases = wire_catalog
    assert len(cases) == 55
    formats = {}
    environment = dict(os.environ)
    def public_data(value):
        data = value.normalized_data()
        if isinstance(value, DeviceInput):
            if value.legacy_fields is not None:
                # Original wire bytes are checked above. Retained evidence uses
                # its established JSON serializer across preparation/reopen.
                data["legacy_fields"] = value.legacy_fields.model_dump(mode="json", exclude_unset=True)
        else:
            for side in ("top_cell", "bottom_cell"):
                data[side] = public_data(getattr(value, side))
        return data
    for name, prepared, original, references in cases:
        declaration = prepared.to_input()
        snapshot = declaration.model_dump_json(exclude_unset=True)
        result = prepare_legacy_wire(declaration, source=original, defaults=catalog, references=references)
        assert result.supported, (name, result.export()["unsupported"])
        assert result.document.content == original.content, (name, result.export()["differences"][:8], result.export()["wire_differences"][:8])
        assert result.original_sources[0] == original and not result.can_execute
        assert declaration.model_dump_json(exclude_unset=True) == snapshot
        if declaration.schema_version == "solarlab.tandem-preparation.v1":
            assert dict(result.references) == references
            reopened = import_tandem(result.document, id=declaration.id, references=dict(result.references), defaults=catalog)
            kind = "tandem"
        else:
            importer = import_standard_device if declaration.source_format == "standard" else import_scaps_device
            reopened = importer(result.document, id=declaration.id, defaults=catalog)
            kind = declaration.source_format
        assert public_data(reopened) == public_data(declaration)
        report = result.export()
        assert report["schema"] == LEGACY_WIRE_VERSION and report["strict_roundtrip_passed"]
        assert report["default_catalog_sha256"] == catalog.content_sha256
        assert report["differences"] == [] and report["wire_differences"] == []
        assert report["behavior"]["binding"] == "not_supplied"
        formats[kind] = formats.get(kind, 0) + 1
    assert dict(os.environ) == environment
    record_property("source_catalog_roundtrips", len(cases))
    record_property("wire_formats", json.dumps(formats, sort_keys=True))


def test_wire_material_device_ion_units_presence_and_inheritance_edits(defaults, wire_guard):
    original = source("calado")
    declaration = import_standard_device(original, id="wire_edit", defaults=defaults)
    before = original.content
    declaration.layers[1].parameters.mu_n = "3 cm^2/(V s)"
    declaration.layers[1].parameters.D_ion = -0.0
    declaration.layers[1].parameters.P0 = 0
    declaration.layers[1].parameters.optical_material = None
    declaration.layers[1].parameters.incoherent = False
    declaration.layers[1].thickness = "750 nm"
    parameters = declaration.layers[1].parameters.editing_data()
    parameters.pop("mu_p", None)
    declaration.layers[1].parameters = type(declaration.layers[1].parameters).model_validate(parameters)
    declaration.settings.phi_left = "-0 V"
    declaration.settings.T = "315 K"
    declaration.settings.dos_band_potentials = False
    snapshot = declaration.model_dump_json(exclude_unset=True)
    environment = dict(os.environ)
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)
    assert raw["layers"][1]["mu_n"] == Decimal("0.0003")
    assert raw["layers"][1]["thickness"] == Decimal("7.5e-7")
    assert raw["layers"][1]["D_ion"].is_signed() and raw["layers"][1]["D_ion"] == 0
    assert raw["layers"][1]["P0"] == 0 and not raw["layers"][1]["P0"].is_signed()
    assert raw["layers"][1]["optical_material"] is None and raw["layers"][1]["incoherent"] is False
    assert "mu_p" not in raw["layers"][1]
    assert raw["device"]["T"] == 315 and raw["device"]["phi_left"].is_signed()
    assert raw["device"]["dos_band_potentials"] is False
    assert result.document.content != before and original.content == before
    assert declaration.model_dump_json(exclude_unset=True) == snapshot and dict(os.environ) == environment
    report = result.export()
    assert "mu_p" in report["inheritance"][declaration.id]["undeclared_layer_parameters"][declaration.layers[1].id]
    assert report["differences"] and report["wire_differences"]
    report["differences"].clear()
    assert result.export()["differences"]  # Detached report mutation cannot alter the result.

    # Quantity vectors also cross the editing/prepared JSON boundary. A real
    # vector edit must reach the wire after unchanged vectors retain their bytes.
    path = ROOT / "perovskite-sim/tests/fixtures/configs/scaps_defect_m1_double_donor_p.yaml"
    vector_source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    vector_input = import_standard_device(vector_source, id="wire_vector", defaults=defaults)
    energy = vector_input.layers[0].bulk_defects[0].configuration.energy_levels
    energy.correlation_energies_eV = ("0.2 eV",)
    vector_result = prepare_legacy_wire(vector_input, source=vector_source, defaults=defaults)
    assert vector_result.supported, vector_result.export()["unsupported"]
    emitted = load_yaml_mapping(vector_result.document.content)["layers"][0]["bulk_defects"][0]
    assert emitted["configuration"]["energy_levels"]["correlation_energies_eV"] == [Decimal("0.2")]
    original_vector = load_yaml_mapping(vector_source.content)["layers"][0]["bulk_defects"][0]
    assert original_vector["configuration"]["energy_levels"]["correlation_energies_eV"] == [Decimal("0.15")]


@pytest.mark.parametrize("ignored", [0, -0.0, False, None, "not a temperature"])
def test_wire_ignored_temperature_keeps_raw_type_and_word(ignored, defaults, wire_guard):
    original = source("tpv", lambda raw: raw["device"].update(temperature=ignored))
    declaration = import_standard_device(original, id="wire_temperature", defaults=defaults)
    declaration.settings.T = "310 K"
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)
    if type(ignored) is float:
        assert float(raw["device"]["temperature"]).hex() == ignored.hex()
    else:
        assert type(raw["device"]["temperature"]) is type(ignored)
        assert raw["device"]["temperature"] == ignored
    assert raw["device"]["T"] == 310
    assert result.export()["retained_legacy_evidence"]["source_sha256"] == original.sha256
    assert result.original_sources[0].content == original.content


@pytest.mark.parametrize("case", ["driftfusion", "ionmonger"])
def test_wire_interface_edit_preserves_substrate_slot_and_original_short_list(case, defaults, wire_guard):
    original = source(case)
    declaration = import_standard_device(original, id="wire_interfaces", defaults=defaults)
    retained = declaration.legacy_fields.model_dump(mode="json", exclude_unset=True)
    original_pairs = load_yaml_mapping(original.content)["device"]["interfaces"]
    declaration.interfaces[1].v_n = "2 cm/s"
    declaration.interfaces[1].v_p = -0.0
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)
    assert len(raw["device"]["interfaces"]) == len(original_pairs) == 2
    assert raw["device"]["interfaces"][0] == original_pairs[0]
    assert raw["device"]["interfaces"][1][0] == Decimal("0.02")
    assert raw["device"]["interfaces"][1][1].is_signed()
    assert declaration.legacy_fields.model_dump(mode="json", exclude_unset=True) == retained
    reopened = import_standard_device(result.document, id=declaration.id, defaults=defaults)
    assert reopened.interfaces[0].left == declaration.layers[0].id
    assert reopened.interfaces[-1].v_n == 0  # Original full-layer trailing padding.


def test_wire_contact_aliases_and_manual_potential_edits(defaults, wire_guard):
    def setup(raw):
        raw["device"].pop("V_bi", None)
        raw["device"]["built_in_potential_mode"] = "legacy_manual"
        raw["device"]["V_bi_override"] = .5
        raw["device"]["S_n_left"] = 4
        raw["device"]["contacts"] = {"left": {"S_n": 4}}
    original = source("tpv", setup)
    declaration = import_standard_device(original, id="wire_aliases", defaults=defaults)
    declaration.settings.V_bi = "600 mV"
    declaration.contacts[0].S_n = None
    declaration.contacts[1].S_p = 0
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    dev = load_yaml_mapping(result.document.content)["device"]
    assert dev["V_bi_override"] == Decimal("0.6") and "V_bi" not in dev
    assert dev["S_n_left"] is None and dev["contacts"]["left"]["S_n"] is None
    assert dev["S_p_right"] == 0


def test_wire_interface_defect_edit_keeps_independent_bare_pair(defaults, wire_guard):
    def setup(raw):
        raw["device"]["interfaces"] = [[4, 5]]
        raw["device"]["interface_defects"] = [None, {
            "sigma_n_cm2": 1e-15, "sigma_p_cm2": 2e-15, "v_th_cm_s": 1e7,
            "N_t_cm2": 1e10, "E_t_eV_below_cb": .4}]
    original = source("tpv", setup)
    declaration = import_standard_device(original, id="wire_defect", defaults=defaults)
    declaration.interfaces[1].defect.kinetics.sigma_n_m2 = "2e-15 cm^2"
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)["device"]
    assert raw["interfaces"] == [[4, 5]] and raw["interface_defects"][0] is None
    assert raw["interface_defects"][1]["sigma_n_cm2"] == Decimal("2e-15")
    reopened = import_standard_device(result.document, id=declaration.id, defaults=defaults)
    assert reopened.interfaces[1].defect.kinetics.normalized_data()["sigma_n_m2"] == 2e-19
    declaration.interfaces[1].defect.kinetics.thermal_velocity_p_m_s = "2e7 cm/s"
    rejected = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert not rejected.supported
    assert rejected.export()["unsupported"][0]["code"] == "single_thermal_velocity"


def test_wire_scaps_unit_and_signed_zero_edits_report_existing_normalization(defaults, wire_guard):
    from solarlab.config.scaps_input import import_scaps_device
    path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    original = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    declaration = import_scaps_device(original, id="wire_scaps", defaults=defaults)
    declaration.layers[2].parameters.mu_n = "3 cm^2/(V s)"
    declaration.layers[2].parameters.D_ion = -0.0
    declaration.layers[2].thickness = "500 nm"
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)
    assert raw["layers"][2]["mu_n_cm2"] == 3 and raw["layers"][2]["thickness_nm"] == 500
    assert raw["layers"][2]["D_ion_m2_s"].is_signed()
    reopened = import_scaps_device(result.document, id=declaration.id, defaults=defaults)
    assert reopened.layers[2].parameters.mu_n == .0003
    assert math.copysign(1, reopened.layers[2].parameters.D_ion) == 1
    assert any(row["path"].endswith("parameters.D_ion") for row in result.export()["existing_import_normalization"])


def test_wire_protocol_hints_retain_complete_history_and_explicit_values(defaults, wire_guard):
    original = source("calado")
    declaration = import_standard_device(original, id="wire_protocol", defaults=defaults)
    hints = declaration.simulation_hints.jv_sweep
    hints.v_rate = "20 mV/s"
    hints.waveform.dark_seed_s = 0
    hints.waveform.dark_prep_s = -0.0
    hints.waveform.turnaround_dark = False
    hints.waveform.uniform_generation_rate_m3_s = None
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    raw = load_yaml_mapping(result.document.content)["simulation_hints"]["jv_sweep"]
    assert raw["n_points"] == 111 and raw["v_rate"] == Decimal("0.02")
    assert raw["waveform"]["dark_seed_s"] == 0
    assert raw["waveform"]["dark_prep_s"].is_signed()
    assert raw["waveform"]["turnaround_dark"] is False
    assert raw["waveform"]["uniform_generation_rate_m3_s"] is None
    original_hints = load_yaml_mapping(original.content)["simulation_hints"]["jv_sweep"]
    assert raw["waveform"]["turnaround_s"] == original_hints["waveform"]["turnaround_s"]
    assert raw["waveform_controls"] == original_hints["waveform_controls"]
    assert not result.can_execute


def test_wire_tandem_edit_reaches_referenced_document(wire_catalog, wire_guard):
    from solarlab.config.tandem_input import import_tandem
    catalog, cases = wire_catalog
    _, prepared, original, references = next(row for row in cases if row[1].to_input().schema_version == "solarlab.tandem-preparation.v1")
    declaration = prepared.to_input()
    declaration.top_cell.layers[1].parameters.D_ion = 0
    declaration.top_cell.layers[1].thickness = "900 nm"
    result = prepare_legacy_wire(declaration, source=original, defaults=catalog, references=references)
    assert result.supported, result.export()["unsupported"]
    emitted = dict(result.references)
    assert emitted[declaration.top_cell_reference].content != references[declaration.top_cell_reference].content
    assert emitted[declaration.bottom_cell_reference] == references[declaration.bottom_cell_reference]
    reopened = import_tandem(result.document, id=declaration.id, references=emitted, defaults=catalog)
    assert reopened.top_cell.layers[1].parameters.D_ion == 0
    assert reopened.top_cell.layers[1].normalized_data()["thickness"] == 9e-7
    assert result.export()["reference_differences"][declaration.top_cell_reference]


def test_wire_canonical_flat_input_uses_strict_source_family(defaults, wire_guard):
    original = source("calado")
    declaration = import_standard_device(original, id="wire_canonical", defaults=defaults)
    declaration.source_format = "canonical"
    declaration.layers[1].parameters.D_ion = 0
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert result.supported, result.export()["unsupported"]
    assert result.export()["source_format_mapping"]["declared"] == "canonical"
    assert result.export()["source_format_mapping"]["wire"] == "standard"
    assert load_yaml_mapping(result.document.content)["layers"][1]["D_ion"] == 0


def test_wire_shared_tandem_reference_cannot_flatten_distinct_instances(wire_catalog, wire_guard):
    from solarlab.config.tandem_input import import_tandem
    catalog, cases = wire_catalog
    _, prepared, original, references = next(row for row in cases if row[1].to_input().schema_version == "solarlab.tandem-preparation.v1")
    raw = raw_json(load_yaml_mapping(original.content))
    raw["tandem"]["bottom_cell"] = raw["tandem"]["top_cell"]
    shared_source = SourceDocument("shared-tandem-reference", json.dumps(raw).encode())
    declaration = import_tandem(shared_source, id="shared_tandem", references=references, defaults=catalog)
    declaration.top_cell.layers[1].parameters.mu_n = "17 cm^2/(V s)"
    result = prepare_legacy_wire(declaration, source=shared_source, references=references, defaults=catalog)
    assert not result.supported and result.references == ()
    assert result.export()["unsupported"][0]["code"] == "shared_reference_conflict"


@pytest.mark.parametrize("change,code", [
    ("named_material", "material_inheritance"), ("extra_layer", "source_topology_changed"),
    ("reindexed_interface", "interface_identity"), ("unpaired_velocity", "velocity_pair_representation"),
    ("spectrum", "roundtrip_loss"), ("retained_temperature", "retained_evidence_changed"),
    ("grid", "grid_layer_identity"),
])
def test_wire_rejects_unrepresentable_edits_with_concrete_differences(change, code, defaults, wire_guard):
    from solarlab.device.inputs import GridLayerInput, NamedMaterialInput
    original = source("tpv")
    declaration = import_standard_device(original, id="wire_reject", defaults=defaults)
    if change == "named_material":
        declaration.materials = (NamedMaterialInput.model_validate({"id": "shared", "name": "Shared", "parameters": {"mu_n": 0}}),)
        declaration.layers[1].material = "shared"
    elif change == "extra_layer":
        declaration.layers = (*declaration.layers, declaration.layers[-1].validated_update({"id": "extra"}))
    elif change == "reindexed_interface":
        declaration.interfaces[0].left = declaration.layers[-1].id
    elif change == "unpaired_velocity":
        declaration.interfaces[0].v_n = None
    elif change == "spectrum":
        declaration.spectrum = "unencoded-external-spectrum"
    elif change == "grid":
        declaration.electrical_grid = (GridLayerInput.model_validate({"layer": "unknown", "interval_weight": 1, "alpha": 1}),)
    else:
        declaration.legacy_fields.temperature = 999
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert not result.supported and result.document is None and result.references == ()
    assert result.export()["unsupported"][0]["code"] == code
    assert result.export()["differences"] and result.original_sources[0] == original


def test_wire_scaps_rejects_missing_negative_ion_field(defaults, wire_guard):
    from solarlab.config.scaps_input import import_scaps_device
    path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    original = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    declaration = import_scaps_device(original, id="wire_scaps_gap", defaults=defaults)
    declaration.layers[2].parameters.D_ion_neg = 1e-18
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults)
    assert not result.supported
    assert result.export()["unsupported"][0]["code"] == "scaps_parameter_not_encodable"
    assert result.export()["unsupported"][0]["path"].endswith("parameters.D_ion_neg")


def test_wire_behavior_context_stays_source_bound_and_never_mutates_environment(defaults, resources, wire_guard):
    from solarlab.config.behavior import behavior_source_requirements
    from solarlab.config.scaps_input import import_scaps_device
    from test_behavior_context import context
    path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    original = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    declaration = import_scaps_device(original, id="wire_behavior", defaults=defaults)
    device = resolve_device(declaration, defaults, resources, sources=(original,))
    sources = tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name, _ in behavior_source_requirements())
    historical = context(device, sources, environment=(("SOLARLAB_DOS_BAND", "0"), ("PEROVSKITE_RHS_FINITE_CHECK", "1")))
    before, environment = historical.export(), dict(os.environ)
    result = prepare_legacy_wire(declaration, source=original, defaults=defaults, behavior=historical)
    assert result.supported, result.export()["unsupported"]
    assert result.export()["behavior"]["context"] == before
    assert result.export()["behavior"]["binding"] == "edited_declaration"
    assert result.export()["behavior"]["selected_models_encoded_as_legacy_controls"] is False
    declaration.settings.Phi = 0
    changed = prepare_legacy_wire(declaration, source=original, defaults=defaults, behavior=historical)
    assert changed.supported, changed.export()["unsupported"]
    assert changed.export()["behavior"]["binding"] == "original_only_requires_reinspection"
    assert changed.export()["unsupported_execution"] and not changed.can_execute
    assert historical.export() == before and dict(os.environ) == environment
    wrong_source = SourceDocument(original.id, original.content + b"\n")
    wrong = prepare_legacy_wire(declaration, source=wrong_source, defaults=defaults, behavior=historical)
    assert not wrong.supported and wrong.export()["unsupported"][0]["code"] == "behavior_source_binding"

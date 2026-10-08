"""Four real legacy gaps: retained evidence, actual loader effect, no solving."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
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

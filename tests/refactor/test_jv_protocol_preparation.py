"""Source and archived-request declarations only; no model solver execution."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.experiments.jv.inputs import JVExperimentInput, JVWaveformInput
from solarlab.experiments.jv.preparation import prepare_jv_experiment
from solarlab.materials.source import SourceDocument
from test_device_configuration_preparation import DEFAULT_SOURCES, LEGACY, ROOT, imported
from test_device_configuration_preparation import resources as resources

REFERENCES = json.loads((Path(__file__).parent / "fixtures/jv_reference_requests.json").read_text())
JV_SOURCES = ("experiments/jv_sweep.py", "experiments/waveform_jv.py", "experiments/dark_jv.py", "experiments/protocol.py")


@pytest.fixture(scope="module")
def defaults():
    return read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES),
        jv_sources=(SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes()),
                    *(SourceDocument(name, (LEGACY / name).read_bytes()) for name in JV_SOURCES)))


def request(defaults, **controls):
    device, source = imported("standard", defaults)
    return dict(schema_version="solarlab.experiment-preparation.v1", id="jv_test",
                device=device.model_dump(mode="json", exclude_unset=True), experiment={"kind": "jv", **controls}), (source,)


def reference(defaults, index=0):
    item = REFERENCES[index]
    raw = item["request"]
    source = SourceDocument(item["provenance"]["source"], json.dumps(raw["device"]).encode())
    device = import_standard_device(source, id=item["provenance"]["label"], defaults=defaults)
    return dict(schema_version="solarlab.experiment-preparation.v1", id=item["provenance"]["label"],
                device=device.model_dump(mode="json", exclude_unset=True), experiment={"kind": "jv", **deepcopy(raw["params"])}), (source,)


def prepare(raw, defaults, resources, sources=()):
    return prepare_jv_experiment(JVExperimentInput.model_validate(raw), defaults, resources, sources=sources)


def words(value):
    if isinstance(value, dict):
        return {key: words(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [words(item) for item in value]
    return [type(value).__name__, value.hex() if type(value) is float else value]


def test_distinct_endpoint_dark_and_driver_defaults_from_actual_source(defaults, resources):
    assert DefaultCatalog.from_mapping(defaults.to_mapping()) == defaults
    raw, sources = request(defaults)
    job = prepare(raw, defaults, resources, sources).to_mapping()["experiment"]
    assert job["controls"]["N_grid"] == 60 and job["controls"]["n_points"] == 30
    assert job["protocol"]["status"] == "pending_operating_contact_potential"
    assert job["protocol"]["declaration"] is None
    raw["experiment"]["request_api"] = "jv_endpoint"
    endpoint = prepare(raw, defaults, resources).to_mapping()["experiment"]
    assert endpoint["controls"]["N_grid"] == 80 and endpoint["controls"]["n_points"] == 40
    raw["experiment"] = {"kind": "dark_jv"}
    dark = prepare(raw, defaults, resources).to_mapping()["experiment"]
    assert dark["controls"]["n_points"] == 60 and dark["controls"]["V_max"] == 1.2
    assert dark["controls"]["illuminated"] is False
    assert dark["legacy_result_scope"] == "forward_curve_and_diode_fit_only_from_transient_dark_sweep"
    assert job["numerical_controls"] == dark["numerical_controls"] == {"rtol": 1e-4, "atol": 1e-6}
    assert defaults.experiment_defaults_for("jv_waveform_controls")["atol_m3"] == 100
    assert set(JV_SOURCES) <= set(dict(defaults.evidence))


def test_endpoint_and_job_defaults_are_independently_source_bound(defaults):
    source = (ROOT / "perovskite-sim/backend/main.py").read_bytes()
    original = b'    N_grid: int = 80\n    n_points: int = 40\n'
    assert source.count(original) == 1
    altered = source.replace(original, b'    N_grid: int = 81\n    n_points: int = 41\n')
    # Change only the JVRequest maximum, leaving start_job's explicit None.
    begin = altered.index(b'class JVRequest(')
    before, after = altered[:begin], altered[begin:]
    altered = before + after.replace(b'    V_max: Optional[float] = None', b'    V_max: Optional[float] = 1.1', 1)
    catalog = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES),
        jv_sources=(SourceDocument("backend/main.py", altered), *(SourceDocument(name, (LEGACY / name).read_bytes()) for name in JV_SOURCES)))
    assert catalog.experiment_defaults_for("jv_endpoint")["V_max"] == 1.1
    assert catalog.experiment_defaults_for("jv_endpoint")["N_grid"] == 81
    assert catalog.experiment_defaults_for("jv_jobs") == defaults.experiment_defaults_for("jv_jobs")
    assert catalog.content_sha256 != defaults.content_sha256


@pytest.mark.parametrize("index", [0, 1, 2])
def test_archived_waveform_words_numerical_settings_and_device_identity(index, defaults, resources):
    raw, sources = reference(defaults, index)
    before = words(raw)
    result = prepare(raw, defaults, resources, sources)
    data = result.to_mapping()
    assert words(result.to_input().editing_data()) == before
    device = resolve_device(result.to_input().device, defaults, resources, sources=sources)
    assert words(data["device"]) == words(device.to_mapping())
    assert data["experiment"]["numerical_controls"] == {"rtol": 0.0001, "atol_m3": 1}
    assert data["experiment"]["controls"]["v_rate"] == 0.04
    history = data["experiment"]["protocol"]["declaration"]["illumination_history"]
    assert [phase["duration_s"] for phase in history] == [120, 30, 0.5, 55, 3, 0.5, 55]
    assert [phase["condition"] for phase in history] == ["dark", "dark", "baseline", "baseline", "dark", "baseline", "baseline"]
    absorber = next(layer for layer in data["device"]["layers"] if layer["role"] == "absorber")
    values = {**absorber["material_parameters"], **absorber["layer_parameters"]}
    assert values["D_ion"] == (0 if index == 1 else 2.585e-18)
    assert values["P0"] == (0 if index == 2 else 1e25)
    assert data["experiment"]["state_history"]["required"] is True
    assert not data["experiment"]["state_history"]["generated"]
    assert data["identity"]["execution_identity"] is None and not result.can_execute
    assert prepare(result.to_input(), defaults, resources, sources).content_sha256 == result.content_sha256
    raw["device"]["layers"].clear()
    data["experiment"]["controls"]["v_rate"] = 999
    assert result.to_mapping()["experiment"]["controls"]["v_rate"] == 0.04


def test_genuine_declaration_oracle_preserves_waveform_and_staircase_semantics(defaults, resources, monkeypatch):
    from perovskite_sim.experiments import jv_sweep, waveform_jv
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical operation forbidden in declaration tests")
    for name in ("build_electrical_grid", "build_material_arrays", "solve_equilibrium", "run_transient", "run_jv_sweep"):
        monkeypatch.setattr(jv_sweep, name, forbidden)
    for raw, sources in (reference(defaults), request(defaults, V_max=1.2, illuminated=False), request(defaults, V_max=-0.0)):
        result = prepare(raw, defaults, resources, sources).to_mapping()
        values = result["experiment"]["controls"]
        stack = SimpleNamespace(T=result["device"]["settings"]["T"])
        if "waveform" in raw["experiment"]:
            waveform = waveform_jv.JVWaveform.from_dict(raw["experiment"]["waveform"])
            assert asdict(waveform) == JVWaveformInput.model_validate(raw["experiment"]["waveform"]).normalized_data()
            expected = waveform_jv.build_waveform_protocol(stack, waveform, n_points=values["n_points"], v_rate=values["v_rate"], V_max=values["V_max"], illuminated=values["illuminated"])
        else:
            expected = jv_sweep.build_jv_experiment_protocol(stack, n_points=values["n_points"], v_rate=values["v_rate"], V_max=values["V_max"], illuminated=values["illuminated"], implicit_legacy_protocol=True)
        assert words(result["experiment"]["protocol"]["declaration"]) == words(expected.to_dict())
        assert result["experiment"]["protocol"]["declaration_sha256"] == expected.protocol_hash


def test_zero_null_omission_equivalent_units_and_tolerance_inheritance(defaults, resources):
    raw, _ = reference(defaults)
    raw["experiment"]["v_rate"] = "40 mV/s"
    raw["experiment"]["waveform"].update(start_voltage_V=-0.0, dark_seed_s=0, dark_prep_s="0 s", turnaround_s=0,
                                        turnaround_dark=False, uniform_generation_rate_m3_s="0 m^-3/s")
    result = prepare(raw, defaults, resources)
    assert words(result.to_input().editing_data()) == words(raw)
    resolved = result.to_mapping()["experiment"]
    assert resolved["controls"]["v_rate"] == 0.04
    assert resolved["waveform"]["uniform_generation_rate_m3_s"] == 0
    assert resolved["protocol"]["declaration"]["dc_settle"]["kind"] == "not_applicable"
    ids = []
    for value in ("omitted", None):
        if value == "omitted":
            raw["experiment"].pop("waveform_controls")
        else:
            raw["experiment"]["waveform_controls"] = None
        result = prepare(raw, defaults, resources)
        assert result.to_mapping()["experiment"]["numerical_controls"]["atol_m3"] == 100
        ids.append(result.content_sha256)
    assert ids[0] != ids[1]
    raw["experiment"]["waveform"]["uniform_generation_rate_m3_s"] = None
    assert prepare(raw, defaults, resources).to_mapping()["experiment"]["protocol"]["declaration"]["illumination_history"][2]["source_reference"] == "device_optics"


@pytest.mark.parametrize("field,value", [("v_rate", 0), ("v_rate", "1 nm"), ("N_grid", 0), ("N_grid", True), ("n_points", 1), ("illuminated", None)])
def test_invalid_request_field_paths(field, value, defaults, resources):
    raw, _ = request(defaults, **{field: value})
    with pytest.raises(ValidationError) as error:
        prepare(raw, defaults, resources)
    assert ("experiment", field) in [row["loc"] for row in error.value.errors()]


def test_waveform_branch_rules_and_unsupported_dark_input_remain_visible(defaults, resources):
    raw, _ = reference(defaults)
    for patch, path in [({"solver": "steady_state"}, "solver"), ({"V_max": None}, "V_max"), ({"V_max": 0}, "V_max"),
                        ({"iface_states": True}, "iface_states"), ({"interface_boundary": True}, "interface_boundary")]:
        bad = deepcopy(raw)
        bad["experiment"].update(patch)
        with pytest.raises(ValidationError) as error:
            prepare(bad, defaults, resources)
        assert ("experiment", path) in [row["loc"] for row in error.value.errors()]
    bad = deepcopy(raw)
    bad["experiment"]["kind"] = "dark_jv"
    with pytest.raises(ValidationError) as error:
        prepare(bad, defaults, resources)
    assert ("experiment", "waveform") in [row["loc"] for row in error.value.errors()]
    bad = deepcopy(raw)
    bad["experiment"]["waveform"] = None
    with pytest.raises(ValidationError, match="requires an explicit waveform"):
        prepare(bad, defaults, resources)


def test_explicit_protocol_order_and_history_mismatch_are_not_repaired(defaults, resources):
    raw, _ = reference(defaults)
    raw["experiment"]["protocol_mode"] = "research_strict"
    explicit = prepare(raw, defaults, resources).to_mapping()["experiment"]["protocol"]["declaration"]
    raw["experiment"]["experiment_protocol"] = explicit
    assert prepare(raw, defaults, resources).to_mapping()["experiment"]["protocol"]["request_history_compared"]
    explicit["illumination_history"][0], explicit["illumination_history"][1] = explicit["illumination_history"][1], explicit["illumination_history"][0]
    with pytest.raises(ValidationError) as error:
        prepare(raw, defaults, resources)
    assert error.value.errors()[0]["loc"][:3] == ("experiment", "experiment_protocol", "illumination_history")
    raw, _ = request(defaults, V_max=1.2, protocol_mode="research_strict")
    with pytest.raises(ValidationError, match="explicit history"):
        prepare(raw, defaults, resources)


def test_preserves_nontransient_choice_without_claiming_hysteresis(defaults, resources):
    raw, _ = request(defaults, solver="steady_state", V_max=1.2)
    output = prepare(raw, defaults, resources).to_mapping()["experiment"]
    assert output["history_kind"] == "single_steady_curve" and output["numerical_controls"] is None
    assert not output["state_history"]["required"]
    raw["experiment"]["request_api"] = "jv_endpoint"
    raw["experiment"]["illuminated"] = False
    with pytest.raises(ValidationError, match="fixes illuminated=True"):
        prepare(raw, defaults, resources)


def test_immutable_source_identity_and_no_legacy_import_in_preparation(defaults, resources, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    raw, sources = reference(defaults)
    first = prepare(raw, defaults, resources, sources)
    other = prepare(raw, defaults, resources, (*sources, SourceDocument("provenance", b"different")))
    assert first.content_sha256 != other.content_sha256 and not other.can_execute
    empty = replace(defaults, experiment_defaults=())
    with pytest.raises(ValueError, match="not supplied"):
        prepare(raw, empty, resources)
    with pytest.raises(ValidationError):
        prepare({**raw, "defaults": defaults.to_mapping()}, defaults, resources)

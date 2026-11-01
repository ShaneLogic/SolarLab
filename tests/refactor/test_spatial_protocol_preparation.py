"""Data-only 2D preparation against source defaults and protocol declarations."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import sys

import pytest
from pydantic import ValidationError

from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.experiments.two_dimensional.inputs import (
    ComponentwiseAtolInput, GrainSweepInput, JV2DInput, JV2DProtocolInput, SpatialExperimentInput,
)
from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment, prepare_spatial_experiment
from solarlab.materials.source import SourceDocument
from solarlab.units import normalize_quantity
from test_device_configuration_preparation import DEFAULT_SOURCES, LEGACY, ROOT, imported, resources


@pytest.fixture(scope="module")
def defaults():
    return read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES),
        experiment_sources=(SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes()),
                            SourceDocument("solver/tolerances.py", (LEGACY / "solver/tolerances.py").read_bytes())))


def request(defaults, case="standard", **controls):
    device, source = imported(case, defaults)
    return {"schema_version": "solarlab.experiment-preparation.v1", "id": "spatial",
            "device": device.model_dump(mode="json", exclude_unset=True),
            "experiment": {"kind": "jv_2d", **controls}}, (source,)


def prepare(data, defaults, resources, sources=()):
    return prepare_spatial_experiment(SpatialExperimentInput.model_validate(data), defaults, resources, sources=sources)


def words(value):
    if isinstance(value, dict):
        return {name: words(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [words(item) for item in value]
    return [type(value).__name__, value.hex() if type(value) is float else value]


def locations(error):
    return [entry["loc"] for entry in error.value.errors()]


def test_actual_source_defaults_distinct_controls_and_catalog_roundtrip(defaults, resources):
    catalog = DefaultCatalog.from_mapping(defaults.to_mapping())
    assert catalog == defaults
    jv, sweep = catalog.experiment_defaults_for("jv_2d"), catalog.experiment_defaults_for("voc_grain_sweep")
    assert jv["lateral_length"] == 5e-7 and jv["Nx"] == sweep["Nx"] == 10
    assert jv["Ny_per_layer"] == 20 and sweep["Ny_per_layer"] == 10
    assert jv["settle_t"] == 1e-7 and sweep["settle_t"] == 1e-3
    assert "backend/main.py" in dict(defaults.evidence)
    assert "solver/tolerances.py" in dict(defaults.evidence)
    untouched = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_SOURCES))
    assert "experiment_defaults" not in untouched.to_mapping()
    data, _ = request(defaults)
    with pytest.raises(ValueError, match="experiment defaults not supplied"):
        prepare(data, untouched, resources)
    with pytest.raises(ValueError, match="incomplete experiment defaults"):
        replace(defaults, experiment_defaults=(("jv_2d", ()), *defaults.experiment_defaults[1:]))


@pytest.mark.parametrize("case", ["standard", "scaps", "grain"])
def test_actual_device_composition_declared_counts_no_mesh_or_execution(case, defaults, resources):
    data, sources = request(defaults, case)
    result = prepare(data, defaults, resources, sources)
    resolved = result.to_mapping()
    device = resolve_device(DeviceInput.model_validate(data["device"]), defaults, resources, sources=sources)
    assert words(resolved["device"]) == words(device.to_mapping())
    assert words(result.to_input().editing_data()) == words(data)
    geometry = resolved["experiment"]["geometry"]
    assert geometry["lateral_interval_count"] == 10 and geometry["expected_lateral_node_count"] == 11
    assert geometry["expected_vertical_node_count"] == 20 * len(geometry["electrical_layer_ids"]) + 1
    assert geometry["x_coordinates_m"] is geometry["y_coordinates_m"] is None
    assert not geometry["mesh_generated"] and not geometry["device_electrical_grid_applied"]
    assert not result.can_execute and resolved["identity"]["execution_identity"] is None
    assert set(device.to_mapping()["capability_gaps"]) <= set(resolved["capability_gaps"])
    assert prepare(result.to_input(), defaults, resources, sources).content_sha256 == result.content_sha256


def test_exact_input_signed_zero_null_empty_and_detached_output(defaults, resources):
    data, sources = request(defaults, V_max=-0.0, illuminated=False, save_snapshots=False,
                            lateral_length="500.000000000000000000 nm", lateral_bc=None, microstructure={})
    data["device"]["settings"]["phi_left"] = -0.0
    before = words(data)
    result = prepare(data, defaults, resources, sources)
    assert words(result.to_input().editing_data()) == before
    data["experiment"]["Nx"] = 12
    output = result.to_mapping()
    output["experiment"]["controls"]["Nx"] = 200
    output["device"]["layers"].clear()
    assert result.to_mapping()["experiment"]["controls"]["Nx"] == 10
    assert "Nx" not in result.to_input().editing_data()["experiment"]
    assert result.to_mapping()["experiment"]["controls"]["V_max"] == 0
    assert math.copysign(1, result.to_input().experiment.V_max) == -1
    with pytest.raises(FrozenInstanceError):
        result.can_execute = True


@pytest.mark.parametrize("controls,path", [
    ({"lateral_length": 0}, ("experiment", "lateral_length")),
    ({"lateral_length": "5 ns"}, ("experiment", "lateral_length")),
    ({"Nx": 0}, ("experiment", "Nx")), ({"Nx": True}, ("experiment", "Nx")),
    ({"Ny_per_layer": 1.5}, ("experiment", "Ny_per_layer")),
    ({"Nx": "10"}, ("experiment", "Nx")), ({"V_step": 0}, ("experiment", "V_step")),
    ({"illuminated": "false"}, ("experiment", "illuminated")),
    ({"rtol": None}, ("experiment", "rtol")),
    ({"lateral_bc": "open"}, ("experiment", "lateral_bc")),
    ({"alpha_x": 2}, ("experiment", "alpha_x")),
])
def test_strict_errors_locate_supplied_field(controls, path, defaults):
    data, _ = request(defaults, **controls)
    with pytest.raises(ValidationError) as error:
        SpatialExperimentInput.model_validate(data)
    assert path in locations(error)


def test_empty_override_preserves_legacy_truthy_policy_and_domain_validation(defaults, resources):
    outputs = []
    for extra in ({}, {"microstructure": None}, {"microstructure": {}}, {"microstructure": {"grain_boundaries": []}}):
        data, _ = request(defaults, "grain", **extra)
        output = prepare(data, defaults, resources)
        outputs.append(output)
        assert words(output.to_input().editing_data()) == words(data)
    for output in outputs[:3]:
        resolved = output.to_mapping()["experiment"]
        assert resolved["microstructure"]["grain_boundaries"]
        assert resolved["controls"]["lateral_bc"] == "neumann"
        assert resolved["microstructure"]["effective_origin"] == "device.grain_boundaries"
    assert not outputs[3].to_mapping()["experiment"]["microstructure"]["grain_boundaries"]
    assert outputs[3].to_mapping()["experiment"]["controls"]["lateral_bc"] == "periodic"
    assert len({output.content_sha256 for output in outputs}) == 4
    data, _ = request(defaults, "grain", lateral_length="10 nm")
    # Device-only preparation has no upper lateral bound; context adds it.
    resolve_device(DeviceInput.model_validate(data["device"]), defaults, resources)
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("device", "grain_boundaries", 0, "width") in locations(error)
    data["experiment"]["lateral_length"] = "500 nm"
    data["experiment"]["lateral_bc"] = "periodic"
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("experiment", "lateral_bc") in locations(error)


def test_override_reference_duplicate_and_overlap_errors(defaults, resources):
    data, _ = request(defaults, "grain")
    row = deepcopy(data["device"]["grain_boundaries"][0])
    data["experiment"]["microstructure"] = {"grain_boundaries": [row]}
    prepare(data, defaults, resources)
    for field, value in (("layer_ids", ["missing"]), ("width", "1000 nm")):
        bad = deepcopy(data)
        bad["experiment"]["microstructure"]["grain_boundaries"][0][field] = value
        with pytest.raises(ValidationError) as error:
            prepare(bad, defaults, resources)
        assert locations(error)[0][:5] == ("experiment", "microstructure", "grain_boundaries", 0, field)
    data["experiment"]["microstructure"]["grain_boundaries"].append(deepcopy(row))
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert locations(error)[0][-1] == "id"
    data["experiment"]["microstructure"]["grain_boundaries"][1]["id"] = "other"
    with pytest.raises(ValidationError, match="overlapping"):
        prepare(data, defaults, resources)


def test_sweep_units_alias_empty_precedence_and_distinct_defaults(defaults, resources):
    data, _ = request(defaults)
    data["experiment"] = {"kind": "voc_grain_sweep", "grain_sizes_nm": [500, "1 um"], "grain_sizes": [777]}
    output = prepare(data, defaults, resources).to_mapping()["experiment"]
    assert output["controls"]["grain_sizes_m"] == [5e-7, 1e-6]
    assert output["controls"]["Ny_per_layer"] == 10
    assert output["controls"]["lateral_bc"] == "neumann"
    assert output["sweep_domains"][0]["band_declaration"]["x_position_m"] == 2.5e-7
    data["experiment"]["grain_sizes_nm"] = []
    assert prepare(data, defaults, resources).to_mapping()["experiment"]["controls"]["grain_sizes_m"] == [777e-9]
    data["experiment"]["grain_sizes"] = None
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("experiment", "grain_sizes_nm") in locations(error)
    for sizes in ([0], [500, "5 s"]):
        with pytest.raises(ValidationError):
            GrainSweepInput(kind="voc_grain_sweep", grain_sizes_nm=sizes)


def declared_protocol(defaults, resources):
    data, _ = request(defaults, Nx=2, illuminated=False)
    controls = prepare(data, defaults, resources).to_mapping()["experiment"]
    values = controls["controls"]
    return {
        "schema_version": "jv-2d-execution-protocol-v1", "temperature_K": dict(defaults.device)["T"],
        "illuminated": False, "illumination_source": None,
        "initial_state_source": "one_dimensional_dark_equilibrium", "initial_state_voltage_V": -0.0,
        "initial_state_settle_s": None, "voltage_values_V": [0, "100 mV"],
        "dwell_time_per_voltage_s": values["settle_t"], "state_topology": "frozen_ion_background",
        "ion_boundary_condition": "frozen", "carrier_boundary_condition": controls["carrier_boundary_condition"],
        "interface_srh": "off", "lateral_bc": "periodic", "x_coordinates_m": [0, "250 nm", "500 nm"],
        "y_coordinates_m": [0, "200 nm", "400 nm"], "grain_boundaries": [],
        "current_composition": "electron_hole_conduction", "current_sampling": "instantaneous_dwell_endpoint",
        "applied_voltage_rate_at_sampling_V_s": "-0 mV/s", "solver_method": "Radau",
        "solver_rtol": values["rtol"], "solver_atol": values["solver_atol"], "solver_max_step_divisor": 50,
        "max_nfev_per_solve": values["max_nfev_per_solve"], "max_bisect": values["max_bisect"],
        "ion_inventory_rtol": values["ion_inventory_rtol"], "save_snapshots": values["save_snapshots"],
        "implicit_legacy_protocol": False,
    }


def test_protocol_matches_genuine_declaration_constructor_without_building_grid(defaults, resources):
    # Constructor validates only supplied arrays/scalars; no grid builder,
    # initial state, material assembly, solver or trajectory is called.
    from perovskite_sim.twod.experiments.jv_protocol_2d import JV2DProtocol
    raw = declared_protocol(defaults, resources)
    normalized = JV2DProtocolInput.model_validate(raw).normalized_data()
    expected = JV2DProtocol.from_dict(normalized)
    assert words(expected.to_dict()) == words(normalized)
    data, _ = request(defaults, Nx=2, illuminated=False, protocol_mode="research_strict", jv_2d_protocol=raw)
    result = prepare(data, defaults, resources).to_mapping()["experiment"]["protocol"]
    assert result["declaration_sha256"] == expected.protocol_hash
    assert result["status"] == "supplied_declaration_only" and "pending" in result["execution_binding"]
    assert normalize_quantity("1 mV/s", "V/s") == 0.001
    for field, value in (("applied_voltage_rate_at_sampling_V_s", "1 mV/s"), ("x_coordinates_m", [0, 1, 1]),
                         ("initial_state_settle_s", 1), ("ion_boundary_condition", "blocking")):
        bad = {**raw, field: value}
        with pytest.raises(ValidationError) as error:
            JV2DProtocolInput.model_validate(bad)
        assert (field,) in locations(error)
    data["experiment"]["jv_2d_protocol"]["solver_rtol"] = 0.01
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("experiment", "jv_2d_protocol", "solver_rtol") in locations(error)


def test_extended_topology_and_source_identity_do_not_grant_execution(defaults, resources):
    data, _ = request(defaults, ion_dynamics="single_mobile", lateral_bc="periodic")
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("experiment", "lateral_bc") in locations(error)
    data, _ = request(defaults, protocol_mode="research_strict")
    with pytest.raises(ValidationError) as error:
        prepare(data, defaults, resources)
    assert ("experiment", "jv_2d_protocol") in locations(error)
    data, _ = request(defaults)
    first = prepare(data, defaults, resources)
    second = prepare(data, defaults, resources, (SourceDocument("declaration-source", b"declared"),))
    assert first.content_sha256 != second.content_sha256
    assert second.to_mapping()["identity"]["input_source_bindings"][0]["sha256"] == hashlib.sha256(b"declared").hexdigest()
    assert second.to_mapping()["identity"]["execution_identity"] is None
    with pytest.raises(ValidationError):
        JV2DInput(kind="jv_2d", atol=1, componentwise_atol=defaults.experiment_defaults_for("jv_2d_componentwise_atol"))


def test_preparation_imports_no_legacy_or_numerical_execution(defaults, resources, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    data, _ = request(defaults)
    assert prepare(data, defaults, resources).can_execute is False


def test_declared_vertical_count_skips_substrate_prefix_and_retains_grid(defaults, resources):
    data, _ = request(defaults, Nx=3, Ny_per_layer=4)
    layers = data["device"]["layers"]
    electrical = [row["id"] for row in layers if row["role"] != "substrate"]
    substrate = {**deepcopy(layers[0]), "id": "optical_only", "role": "substrate"}
    data["device"]["layers"] = [substrate, *layers]
    data["device"]["electrical_grid"] = [{"layer": id, "interval_weight": "7", "alpha": 9} for id in electrical]
    output = prepare(data, defaults, resources).to_mapping()["experiment"]["geometry"]
    assert output["electrical_layer_ids"] == electrical
    assert output["expected_vertical_node_count"] == len(electrical) * 4 + 1
    assert output["expected_lateral_node_count"] == 4
    assert output["retained_electrical_grid"] and not output["device_electrical_grid_applied"]
    data["device"]["layers"] = [*layers, substrate]
    with pytest.raises(ValueError, match="substrates.*prefix"):
        prepare(data, defaults, resources)


def test_strict_api_differences_and_direct_json_guard_are_explicit(defaults, resources):
    data, _ = request(defaults, "grain")
    output = prepare(data, defaults, resources).to_mapping()
    assert output["validation_contract"]["scope"] == "strict_new_preparation_input_not_legacy_request_equivalence"
    assert output["validation_contract"]["legacy_numerical_sources_changed"] is False
    row = deepcopy(data["device"]["grain_boundaries"][0])
    row.update(x_position=4.99999999999999e-9, width=1e-8)
    data["experiment"]["microstructure"] = {"grain_boundaries": [row]}
    with pytest.raises(ValidationError, match="strict declaration") as error:
        prepare(data, defaults, resources)
    assert locations(error)[0] == ("experiment", "microstructure", "grain_boundaries", 0, "width")
    raw = json.dumps(data).replace('"kind": "jv_2d"', '"kind": "jv_2d", "kind": "jv_2d"')
    with pytest.raises(ValueError, match="duplicate experiment JSON"):
        PreparedSpatialExperiment(raw.encode(), defaults, resources)

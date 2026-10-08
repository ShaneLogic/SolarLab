"""Stable serial declarations against actual SCAPS coordinates and resolvers."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
from pydantic import ValidationError
from solarlab.device.inputs import DeviceInput
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import prepare_sweep, prepare_point, digest
from solarlab.sweeps.scaps import scaps_sweeps
from solarlab.sweeps.reference import baseline_signature
from test_device_configuration_preparation import ROOT, imported
from test_device_configuration_preparation import resources as resources
from test_jv_protocol_preparation import defaults as defaults, reference, words


@pytest.fixture(scope="module")
def scaps(defaults):
    device, source = imported("scaps", defaults)
    manifest = SourceDocument("P00.reference_manifest", (ROOT / "reproducibility/RefactorReferenceManifestV1.json").read_bytes())
    contract = SourceDocument("P00.scaps_contract", (ROOT / "reproducibility/ScapsComparisonContractV1.json").read_bytes())
    sweeps, refs = scaps_sweeps(device, manifest, contract, interface_density_interpretation="historical_areal_input")
    return device, (source,), sweeps, refs


def single(base, family="layer_parameter", owner_id="layer_2", parameter=("D_ion",), values=(0,), **target):
    return dict(schema_version="solarlab.sweep-preparation.v1", id="scan", base=base.model_dump(mode="json", exclude_unset=True),
        axes=[dict(id="axis", target=dict(family=family, parameter=parameter, **({"owner_id": owner_id} if owner_id is not None else {}), **target),
            coordinates=[dict(kind="value", value=value) for value in values])])


def prepare(raw, defaults, resources, sources=(), references=()):
    return prepare_sweep(SweepInput.model_validate(raw), defaults, resources, sources=sources, references=references)


def test_all_thirteen_original_groups_and_missing_coordinates_prepare_without_solver(scaps, defaults, resources, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy.integrate")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    device, sources, sweeps, refs = scaps
    results = {sweep.id: prepare_sweep(sweep, defaults, resources, sources=sources, references=refs).to_mapping() for sweep in sweeps}
    assert len(results) == 13
    assert sum(len(value["points"]) for value in results.values()) == 261
    assert sum(point["reference"]["status"] == "reference_missing" for value in results.values() for point in value["points"]) == 14
    for value in results.values():
        assert not value["can_execute"] and value["expansion"]["numerical_calls"] == 0
        assert value["summary"]["unresolved"] == 0, value["points"][0]
        assert not value["expansion"]["truncated"]
        assert all(point["simulation_status"] == "not_started" and point["execution_identity"] is None for point in value["points"])
        interpretation = value["reference_scope"]["interpretation"]
        assert interpretation["interface_density_choice"] == "historical_areal_input"
        density_axes = interpretation["interface_density_axes"]
        assert bool(density_axes) == (value["id"] in {"scaps_nt_htl_pvk", "scaps_nt_pvk_etl", "scaps_nt_et", "scaps_nt_cbo"})
        for axis in density_axes:
            assert axis == dict(axis_id="Nt", reported_unit="cm^-3", input_unit="cm^-2", resolved_unit="m^-2", thickness_conversion=False, native_input_verified=False)
    assert results["scaps_nt_et"]["expansion"]["declared_points"] == 7 * 8
    pair = results["scaps_nt_cbo"]
    assert pair["expansion"]["declared_points"] == 7 * 16
    assert sum(point["reference"]["status"] == "available" for point in pair["points"]) == 100
    missing = [point for point in pair["points"] if point["reference"]["status"] == "reference_missing"]
    assert all(point["reference"]["reason"] == "four_blank_metric_cells_in_original_workbook" for point in missing)
    assert all(point["reference"]["solver_execution_status"] == "unknown" for point in missing)
    output = json.dumps(results)
    assert not any(metric in output for metric in ("PCE_percent", "Voc_V", "Jsc_mA_cm2", "FF_percent"))
    assert device.model_dump(mode="json", exclude_unset=True) == pair["points"][0]["applied_input"] | {"layers": device.model_dump(mode="json", exclude_unset=True)["layers"], "interfaces": device.model_dump(mode="json", exclude_unset=True)["interfaces"]}


@pytest.mark.parametrize("group", ["scaps_nt_pvk_cb", "scaps_nt_pvk_vb", "scaps_et_pvk_cb", "scaps_et_pvk_vb", "scaps_nt_htl_pvk", "scaps_nt_pvk_etl", "scaps_et_htl_pvk", "scaps_et_pvk_etl"])
def test_four_defect_locations_update_only_their_named_field(group, scaps, defaults, resources):
    _, sources, sweeps, refs = scaps
    raw = next(sweep for sweep in sweeps if sweep.id == group).model_dump(mode="json", exclude_unset=True)
    raw["axes"][0]["coordinates"] = raw["axes"][0]["coordinates"][:1]
    baseline = deepcopy(raw["base"])
    result = prepare(raw, defaults, resources, sources, refs).to_mapping()
    point = result["points"][0]
    expected = prepare_point(point["applied_input"], defaults, resources, sources)
    assert point["content_sha256"] == expected.content_sha256
    def differences(left, right, path=()):
        if isinstance(left, dict) and isinstance(right, dict):
            assert left.keys() == right.keys()
            return [d for key in left for d in differences(left[key], right[key], (*path, key))]
        if isinstance(left, list) and isinstance(right, list):
            assert len(left) == len(right)
            return [d for i, (a, b) in enumerate(zip(left, right)) for d in differences(a, b, (*path, i))]
        return [] if type(left) is type(right) and left == right else [path]
    delta = differences(baseline, point["applied_input"])
    assert len(delta) == 1
    if "pvk_cb" in group or "pvk_vb" in group:
        expected_index = 0 if "pvk_cb" in group else 1
        assert delta[0][:4] == ("layers", 2, "bulk_defects", expected_index)
        assert point["applied_input"]["layers"][2]["scaps_defect_metadata"] == baseline["layers"][2]["scaps_defect_metadata"]
    else:
        expected_index = 1 if "htl_pvk" in group else 2
        assert delta[0][:3] == ("interfaces", expected_index, "defect")
        assert point["applied_input"]["interfaces"][expected_index]["defect"]["partner_metadata"] == baseline["interfaces"][expected_index]["defect"]["partner_metadata"]


def test_coordinate_and_axis_order_do_not_retarget_reference_or_point_identity(scaps, defaults, resources):
    _, sources, sweeps, refs = scaps
    raw = next(s for s in sweeps if s.id == "scaps_nt_et").model_dump(mode="json", exclude_unset=True)
    for axis in raw["axes"]:
        axis["coordinates"] = axis["coordinates"][:2]
    first = prepare(raw, defaults, resources, sources, refs).to_mapping()
    raw["axes"].reverse()
    for axis in raw["axes"]:
        axis["coordinates"].reverse()
    second = prepare(raw, defaults, resources, sources, refs).to_mapping()
    assert {p["id"]: p for p in first["points"]} == {p["id"]: p for p in second["points"]}
    raw["base"]["layers"][2]["bulk_defects"].reverse()
    target = single(DeviceInput.model_validate(raw["base"]), family="bulk_defect", parameter=("energy_level", "value_eV"), values=("50 meV",), local_id="bulk_0")
    point = prepare(target, defaults, resources).to_mapping()["points"][0]
    assert point["applied_input"]["layers"][2]["bulk_defects"][1]["id"] == "bulk_0"
    assert point["applied_input"]["layers"][2]["bulk_defects"][1]["energy_level"]["value_eV"] == "50 meV"


def test_cbo_uses_effective_pvk_affinity_and_rejects_ambiguous_dependencies(scaps, defaults, resources):
    device, _, sweeps, _ = scaps
    raw = next(s for s in sweeps if s.id == "scaps_cbo").model_dump(mode="json", exclude_unset=True)
    raw["axes"][0]["coordinates"] = [dict(kind="value", value="-200 meV")]
    raw["base"]["layers"][2]["parameters"]["chi"] = "4.0 eV"
    output = prepare(raw, defaults, resources).to_mapping()["points"][0]
    assert output["effective_targets"]["CBO"]["value"] == 4.2
    assert output["effective_targets"]["CBO"]["offset_eV"] == -0.2
    assert output["applied_input"]["settings"] == raw["base"]["settings"]
    raw["axes"].append(single(device, parameter=("chi",), values=(4.1,))["axes"][0])
    with pytest.raises(ValidationError, match="reference affinity"):
        prepare(raw, defaults, resources)
    raw["axes"][1] = deepcopy(raw["axes"][0])
    raw["axes"][1]["id"] = "alias"
    with pytest.raises(ValidationError, match="same parameter"):
        prepare(raw, defaults, resources)


def test_d0_c0_null_omission_and_unit_words_remain_distinct(scaps, defaults, resources):
    device = scaps[0]
    raw = single(device, values=(-0.0, "1e-18 m^2/s"))
    original = words(raw)
    result = prepare(raw, defaults, resources)
    assert words(result.to_input().model_dump(mode="json", exclude_unset=True)) == words(SweepInput.model_validate(raw).model_dump(mode="json", exclude_unset=True))
    assert words(raw) == original
    points = result.to_mapping()["points"]
    assert words(points[0]["applied_input"]["layers"][2]["parameters"]["D_ion"]) == words(-0.0)
    assert points[1]["applied_input"]["layers"][2]["parameters"]["D_ion"] == "1e-18 m^2/s"
    assert all(p["applied_input"]["layers"][2]["parameters"].get("P0") == raw["base"]["layers"][2]["parameters"].get("P0") for p in points)
    c0 = single(device, parameter=("P0",), values=(0,))
    assert prepare(c0, defaults, resources).to_mapping()["points"][0]["id"] != points[0]["id"]
    raw = single(device, family="setting", owner_id=None, parameter=("work_function_left_eV",), values=(None,))
    raw["axes"][0]["coordinates"].append({"kind": "omit"})
    points = prepare(raw, defaults, resources).to_mapping()["points"]
    assert points[0]["applied_input"]["settings"]["work_function_left_eV"] is None
    assert "work_function_left_eV" not in points[1]["applied_input"]["settings"]
    assert points[0]["id"] != points[1]["id"]


@pytest.mark.parametrize("values", [(0, -0.0), ("1000 nm", "1 um")])
def test_duplicate_equivalent_coordinates_are_rejected(values, scaps, defaults, resources):
    field = "D_ion" if values[0] == 0 else "trap_decay_length"
    with pytest.raises(ValidationError, match="duplicate physical coordinate"):
        prepare(single(scaps[0], parameter=(field,), values=values), defaults, resources)


def test_invalid_coordinates_targets_bounds_and_reference_states(scaps, defaults, resources):
    raw = single(scaps[0], values=("1 nm",))
    point = prepare(raw, defaults, resources).to_mapping()["points"][0]
    assert point["status"] == "unresolved" and point["field_errors"][0]["loc"] == ["layers", 2, "parameters", "D_ion"]
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            prepare(single(scaps[0], values=(value,)), defaults, resources)
    for owner in ("missing", "layer_0"):
        raw = single(scaps[0], family="bulk_defect", owner_id=owner, local_id="bulk_0", parameter=("energy_level", "value_eV"))
        with pytest.raises(ValidationError, match="target"):
            prepare(raw, defaults, resources)
    raw = single(scaps[0], values=tuple(range(129)))
    with pytest.raises(ValidationError, match="not truncated"):
        prepare(raw, defaults, resources)
    raw = single(scaps[0])
    raw["reference_id"] = "unknown"
    assert prepare(raw, defaults, resources).to_mapping()["points"][0]["reference"]["status"] == "reference_unavailable"
    raw["reference_id"] = "scaps_nt_pvk_cb"
    assert prepare(raw, defaults, resources, references=scaps[3]).to_mapping()["points"][0]["reference"]["status"] == "target_mismatch"


def test_jv_base_protocol_and_unrelated_history_are_retained(defaults, resources):
    raw_jv, sources = reference(defaults)
    base = JVExperimentInput.model_validate(raw_jv)
    raw = single(base, family="jv", owner_id=None, parameter=("v_rate",), values=("40 mV/s", "0 V/s"))
    result = prepare(raw, defaults, resources, sources)
    points = result.to_mapping()["points"]
    assert points[0]["status"] == "prepared_pending_dependencies" and points[1]["status"] == "unresolved"
    assert points[0]["effective_targets"]["axis"]["value"] == 0.04
    assert points[0]["applied_input"]["experiment"]["waveform"] == raw_jv["experiment"]["waveform"]
    direct = prepare_point(points[0]["applied_input"], defaults, resources, sources)
    assert points[0]["resolved_sha256"] == digest(direct.to_mapping())
    assert prepare(result.to_input(), defaults, resources, sources).content_sha256 == result.content_sha256


def test_reference_metadata_is_source_bound_and_contains_no_result_answers(scaps, defaults):
    manifest = SourceDocument("manifest", (ROOT / "reproducibility/RefactorReferenceManifestV1.json").read_bytes())
    contract = SourceDocument("contract", (ROOT / "reproducibility/ScapsComparisonContractV1.json").read_bytes())
    changed = json.loads(manifest.content)
    changed["scaps"]["groups"][0]["points"][0]["coordinates"] = [999]
    with pytest.raises(ValueError, match="frozen comparison"):
        scaps_sweeps(scaps[0], replace(manifest, content=json.dumps(changed).encode()), contract, interface_density_interpretation="historical_areal_input")
    for ref in scaps[3]:
        assert "values" not in ref.to_mapping()
        assert "PCE" not in ref.document_json.decode()


def test_scaps_requires_an_explicit_known_interface_density_channel(scaps):
    manifest = SourceDocument("manifest", (ROOT / "reproducibility/RefactorReferenceManifestV1.json").read_bytes())
    contract = SourceDocument("contract", (ROOT / "reproducibility/ScapsComparisonContractV1.json").read_bytes())
    with pytest.raises(TypeError, match="interface_density_interpretation"):
        scaps_sweeps(scaps[0], manifest, contract)
    for choice in (None, "", "native_volume", "thickness_conversion"):
        with pytest.raises(ValueError, match="explicit historical_areal_input"):
            scaps_sweeps(scaps[0], manifest, contract, interface_density_interpretation=choice)


def test_named_material_inheritance_conflicts_and_duplicate_owner_ids(scaps, defaults, resources):
    raw = single(scaps[0], parameter=("mu_n",), values=("2 cm^2/(V s)",))
    raw["base"]["materials"] = [{"id": "named", "name": "Declared material", "parameters": {"mu_n": "3 cm^2/(V s)", "chi": "4 eV"}}]
    raw["base"]["layers"][2]["material"] = "named"
    raw["base"]["layers"][2]["parameters"].pop("mu_n")
    raw["axes"][0]["coordinates"].append({"kind": "omit"})
    points = prepare(raw, defaults, resources).to_mapping()["points"]
    assert points[1]["effective_targets"]["axis"]["value"] == 0.0003
    assert points[1]["effective_targets"]["axis"]["origin"] == "material:named"
    raw["base"]["layers"].append(deepcopy(raw["base"]["layers"][2]))
    with pytest.raises(ValidationError, match="ambiguous"):
        prepare(raw, defaults, resources)
    raw = next(s for s in scaps[2] if s.id == "scaps_cbo").model_dump(mode="json", exclude_unset=True)
    raw["base"]["materials"] = [{"id": "named", "name": "Declared material", "parameters": {"chi": "4 eV"}}]
    raw["base"]["layers"][2]["material"] = "named"
    raw["base"]["layers"][2]["parameters"].pop("chi")
    raw["axes"][0]["coordinates"] = raw["axes"][0]["coordinates"][:1]
    raw["axes"].append(dict(id="affinity", target=dict(family="material_parameter", owner_id="named", parameter=["chi"]), coordinates=[dict(kind="value", value="4.1 eV")]))
    with pytest.raises(ValidationError, match="reference affinity"):
        prepare(raw, defaults, resources)


def test_reference_baseline_difference_does_not_claim_equivalent_comparison(scaps, defaults, resources):
    raw = next(s for s in scaps[2] if s.id == "scaps_cbo").model_dump(mode="json", exclude_unset=True)
    raw["axes"][0]["coordinates"] = raw["axes"][0]["coordinates"][:1]
    raw["base"]["layers"][2]["parameters"]["chi"] = "4 eV"
    ref = prepare(raw, defaults, resources, references=scaps[3]).to_mapping()["points"][0]["reference"]
    assert ref["status"] == "available" and ref["baseline_matches"] is False
    assert ref["comparison_qualified"] is False and ref["scope"] == "coordinate_and_target_coverage_only"


def test_reference_baseline_survives_json_number_spelling_without_erasing_input_types(scaps, defaults, resources):
    raw = next(s for s in scaps[2] if s.id == "scaps_cbo").model_dump(mode="json", exclude_unset=True)
    raw["axes"][0]["coordinates"] = raw["axes"][0]["coordinates"][:1]
    def browser_numbers(value):
        if isinstance(value, dict):
            return {key: browser_numbers(part) for key, part in value.items()}
        if isinstance(value, list):
            return [browser_numbers(part) for part in value]
        if type(value) is float and value.is_integer() and (value != 0 or math.copysign(1, value) > 0):
            return int(value)
        return value
    transported = browser_numbers(raw)
    before = words(transported)
    assert baseline_signature(raw["base"]) == baseline_signature(transported["base"])
    ref = prepare(transported, defaults, resources, references=scaps[3]).to_mapping()["points"][0]["reference"]
    assert ref["baseline_matches"] is True and ref["comparison_qualified"] is False
    assert words(transported) == before
    for left, right in [(-0.0, 0), (False, 0), (None, ""), (1, "1"), ("1 eV", "1000 meV"), (9007199254740993, float(9007199254740993))]:
        assert baseline_signature({"value": left}) != baseline_signature({"value": right})
    assert baseline_signature({"value": None}) != baseline_signature({})

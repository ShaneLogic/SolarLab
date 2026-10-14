"""Actual saved inputs and R1 preparation bytes; these checks run no model."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
import subprocess
import sys

import pytest

from solarlab.config.legacy_execution import LegacyExecutionContext, R1ExecutionContext, r1_execution_source_requirements
from solarlab.materials.source import SourceDocument
from test_behavior_context import LEGACY, ROOT, context, document
from test_behavior_context import defaults as defaults, resources as resources
from test_behavior_context import scaps as scaps, ion as ion, sources as sources
from test_behavior_context import r1_material_inputs as r1_material_inputs


def source(path):
    return SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())


@pytest.fixture
def r1(r1_material_inputs):
    device, _ = r1_material_inputs
    documents = tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name, _ in r1_execution_source_requirements())
    prepared = source(ROOT / "tests/refactor/fixtures/r1_common_state_preparation.json")
    policy = source(ROOT / "tests/refactor/fixtures/r1_execution_policy.json")
    assert prepared.sha256 == "94017d880119bb151b22e72118f2668721537ad9eb02186c55dcf07adf5a3d46"
    assert policy.sha256 == "82c1d4c1115f0ddd4caa4c9fe2885ba36bd394937b3ec6c9a9a0d488d83b87db"
    return R1ExecutionContext(
        device=device, input_source=device.sources[0], sources=documents, environment=(),
        arguments=document("r1_explicit_short_call", {"intervals": 4, "times_s": [0., 1e-9], "backend": "legacy", "physics_evidence": True}),
        reference=source(ROOT / "perovskite-sim/tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"),
        study_input=source(ROOT / "perovskite-sim/reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"),
        prepared=prepared, policy=policy,
    )


@pytest.mark.parametrize("driver", ("scaps_transient", "scaps_steady_state"))
def test_original_scaps_input_and_supplied_arguments_are_unchanged(scaps, sources, driver):
    behavior = context(scaps, sources, driver=driver)
    invocation = LegacyExecutionContext(behavior, scaps.sources[0])
    actual = invocation.export()
    assert invocation.input_source.content == scaps.sources[0].content
    assert actual["arguments"] == json.loads(behavior.arguments.content)
    assert actual["environment"] == behavior.export()["environment_effective_snapshot"]
    assert actual["behavior"]["source_inspection"]["optics"]["incoherence"][0]["historical_loader_value"] is False
    assert actual["existing_driver_validation_required"] and not actual["scientific_qualification"]
    with pytest.raises(ValueError, match="original historical input"):
        LegacyExecutionContext(behavior, replace(scaps.sources[0], content=scaps.sources[0].content + b"\n"))


def test_waveform_bridge_retains_original_controls_and_full_history(ion, sources):
    device, original = ion
    behavior = context(device, sources, driver="ion_hysteresis", arguments=original)
    raw = next(item for item in device.sources if item.id.endswith("#request.device"))
    invocation = LegacyExecutionContext(behavior, raw)
    result = invocation.export()
    supplied = json.loads(original.content)
    assert result["arguments"]["waveform"] == supplied["waveform"]
    assert result["arguments"]["rtol"] == supplied["waveform_controls"]["rtol"]
    assert result["arguments"]["atol"] == supplied["waveform_controls"]["atol_m3"]
    assert invocation.behavior.arguments.content == original.content
    assert "solver" not in result["arguments"] and "waveform_controls" not in result["arguments"]
    inconsistent = document("inconsistent_control", {**supplied, "rtol": 0.1})
    with pytest.raises(ValueError, match="ambiguous waveform"):
        LegacyExecutionContext(context(device, sources, driver="ion_hysteresis", arguments=inconsistent), raw)


def test_r1_preserves_the_complete_actual_preparation_reference_policy_and_sources(r1):
    record = r1.export()
    original = json.loads(r1.prepared.content)
    assert record["prepared_sha256"] == original["sha256"]
    assert record["history"] == original["history"]
    assert record["preparation_created_utc"] == original["created_utc"]
    assert record["preparation_environment"] == original["environment"]
    assert record["arguments"]["control"] == "D"
    assert record["arguments"]["controls"] == {"nu_I": 1, "nu_t": 1}
    assert record["arguments"]["nonlinear_factor"] == .1
    assert record["arguments"]["time_substeps"] == [1, 2, 4]
    assert record["material_behavior"]["declared_interface_charge_closure"] == "equilibrium_referenced"
    assert record["material_behavior"]["material_interface_charge_closure"] == "off"
    assert record["existing_reader_and_policy_verification_required"]
    assert not record["scientific_qualification"]


def test_real_pair_preparation_keeps_all_high_low_words_and_original_seed(r1):
    paired = source(ROOT / "tests/refactor/fixtures/r1_pair_common_state_preparation.json")
    result = replace(r1, prepared=paired,
        arguments=document("pair_call", {**json.loads(r1.arguments.content), "backend": "pair"}))
    raw = json.loads(paired.content)
    assert result.prepared.content == paired.content
    assert result.export()["material_behavior"]["backend_representation"] == "float64-pair-v1"
    assert raw["seed_preparation_sha256"] == raw["seed_preparation"]["sha256"]
    assert len([key for key in raw["state"] if key.startswith("precision_")]) == 34
    changed = json.loads(paired.content)
    changed["state"]["precision_phi_V_lo"].pop()
    changed["sha256"] = hashlib.sha256(json.dumps({key: value for key, value in changed.items() if key != "sha256"},
                                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="shapes differ"):
        replace(result, prepared=document("wrong_pair_shape", changed))


@pytest.mark.parametrize("control,pair", [("A", (0, 0)), ("B", (1, 0)), ("C", (0, 1)), ("D", (1, 1))])
def test_r1_controls_reuse_exact_common_state_and_keep_inventory(r1, control, pair):
    args = {**json.loads(r1.arguments.content), "control": control}
    result = replace(r1, arguments=document("control", args))
    assert result.prepared is r1.prepared
    assert result.reference is r1.reference
    assert result.export()["arguments"]["controls"] == dict(zip(("nu_I", "nu_t"), pair))
    assert result.export()["material_behavior"]["ions"] == r1.export()["material_behavior"]["ions"]


def test_contexts_are_deeply_detached_and_do_not_change_environment(r1, scaps, sources):
    before = dict(os.environ)
    original = r1.content_sha256
    a = replace(r1, environment=(("SOLARLAB_SS_GUMMEL", "0"),))
    b = replace(r1, environment=(("SOLARLAB_SS_GUMMEL", "1"),))
    assert a.export()["environment"]["SOLARLAB_SS_GUMMEL"] == "0"
    assert b.export()["environment"]["SOLARLAB_SS_GUMMEL"] == "1"
    assert a.prepared is b.prepared
    mutable = a.export()
    mutable["arguments"]["times_s"].append(1.0)
    mutable["prepared"].clear()
    assert a.export()["arguments"]["times_s"] == [0., 1e-9]
    assert r1.content_sha256 == original
    with pytest.raises(FrozenInstanceError):
        r1.prepared = r1.policy
    old = LegacyExecutionContext(context(scaps, sources), scaps.sources[0])
    value = old.export()
    value["arguments"].clear()
    assert old.export()["arguments"]
    assert dict(os.environ) == before


@pytest.mark.parametrize("key,value", [
    ("intervals", True), ("control", "Z"), ("control", False), ("amplitude_V", 0.),
    ("amplitude_V", float("nan")), ("times_s", [0., 0.]), ("times_s", [False, 1e-9]),
    ("times_s", [0., 1e-13]), ("time_substeps", [1, 2]), ("time_substeps", [1, 2, 5]),
    ("nonlinear_factor", .2), ("nonlinear_factor", True), ("physics_evidence", 1),
])
def test_r1_invalid_protocol_is_refused(r1, key, value):
    with pytest.raises(ValueError):
        replace(r1, arguments=document("invalid", {**json.loads(r1.arguments.content), key: value}))


def test_r1_refuses_changed_source_input_reference_and_backend(r1):
    with pytest.raises(ValueError, match="exact R1"):
        replace(r1, sources=r1.sources[:-1])
    with pytest.raises(ValueError, match="exact R1"):
        replace(r1, sources=(replace(r1.sources[0], content=r1.sources[0].content+b"\n"), *r1.sources[1:]))
    with pytest.raises(ValueError, match="original historical input"):
        replace(r1, input_source=replace(r1.input_source, content=r1.input_source.content+b"\n"))
    changed = json.loads(r1.reference.content)
    changed["f_ref"][0] += .01
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(r1, reference=document("changed_reference", changed))
    with pytest.raises(ValueError, match="pair|binding mismatch"):
        replace(r1, arguments=document("wrong_backend", {**json.loads(r1.arguments.content), "backend": "pair"}))


@pytest.mark.parametrize("mutation", ("shape", "bool", "nan", "missing_state", "history", "uncertified", "grid"))
def test_resealed_wrong_state_shapes_and_preparation_claims_are_refused(r1, mutation):
    data = json.loads(r1.prepared.content)
    if mutation == "shape":
        data["state"]["n_m3"].pop()
    elif mutation == "bool":
        data["state"]["n_m3"][0] = True
    elif mutation == "nan":
        data["state"]["n_m3"][0] = float("nan")
    elif mutation == "missing_state":
        del data["state"]["capture_m2_s"]
    elif mutation == "history":
        data["history"]["voltage_V"] = .1
    elif mutation == "uncertified":
        data["preparation_checks"]["certified"] = False
    elif mutation == "grid":
        data["grid"]["coordinates_m"].pop()
    body = {key: value for key, value in data.items() if key != "sha256"}
    data["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError):
        replace(r1, prepared=document("resealed_bad_preparation", data))


def test_clean_import_does_not_load_an_old_engine_or_numerical_solver(tmp_path):
    code = "import sys; import solarlab.config.legacy_execution; assert not set(sys.modules) & {'perovskite_sim','scipy','sksundae','flint','solarlab_server'}"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr

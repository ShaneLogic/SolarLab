"""Complete manifest consumption through real data preparers; no solver calls.

Initial-state/grid/preparation documents deliberately use the existing labelled
nonexecuted fixtures. Passing these tests grants no physical-state qualification.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import random

import pytest

from solarlab.config.identity import (
    ExecutionEngine, ExecutionManifest, execution_identity, execution_manifest,
    verify_execution_manifest,
)
from solarlab.config.resolve_device import resolve_device
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.jv.preparation import consume_jv_manifest, prepare_jv_experiment
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import PreparedJVExperiment, prepare_point, prepare_sweep
from test_device_configuration_preparation import resources as resources
from test_execution_identity_preparation import execution as execution, identity_inputs, engine_behavior
from test_jv_protocol_preparation import defaults as defaults, reference, request, words
from test_sweep_preparation import single


def binding(prepared, execution):
    device = resolve_device(prepared.to_input().device, prepared.defaults, prepared.resources, sources=prepared.sources)
    inputs = identity_inputs(device)
    protocol, controls = prepared.execution_documents()
    execution = replace(execution, protocol=protocol, numeric_controls=controls)
    engine = ExecutionEngine("solarlab")
    return inputs, execution, engine, execution_manifest(inputs, execution, engine=engine)


@pytest.fixture
def prepared(defaults, resources):
    raw, sources = reference(defaults)
    return prepare_jv_experiment(JVExperimentInput.model_validate(raw), defaults, resources, sources=sources)


def consume(prepared, inputs, execution, selected_engine, manifest, **changes):
    args = dict(manifest=manifest, inputs=inputs, execution=execution, engine=selected_engine,
                environment=execution.environment, expected_sha256=manifest.sha256, sources=prepared.sources)
    args.update(changes)
    return consume_jv_manifest(prepared.to_input(), prepared.defaults, prepared.resources, **args)


def test_complete_manifest_retains_every_upstream_byte_and_exact_engine(prepared, execution):
    inputs, description, engine, manifest = binding(prepared, execution)
    payload = manifest.export()["document"]
    for name in ("protocol", "preparation", "initial_state", "grid", "numeric_controls"):
        original = getattr(description, name)
        record = payload["content_roles"][name]
        assert bytes.fromhex(record["content_hex"]) == original.content
        assert record["sha256"] == original.sha256 and record["bytes"] == len(original.content)
    assert payload["execution_identity"]["payload"]["engine"] == engine.export()
    assert payload["execution_identity"]["payload"]["implementation"]["build_source_commit"] == description.implementation.build_source_commit
    assert manifest.execution_sha256 == execution_identity(inputs, description, engine=engine).sha256
    assert hashlib.sha256(manifest.canonical_json).hexdigest() == manifest.sha256
    assert ExecutionManifest(manifest.canonical_json, description.current_repository_head) == manifest
    assert not manifest.can_execute and not manifest.cache_eligible
    with pytest.raises(FrozenInstanceError):
        manifest.sha256 = "0" * 64
    payload["content_roles"]["initial_state"]["content_hex"] = "00"
    assert manifest.export()["document"] != payload


def test_real_jv_consumer_retains_full_input_protocol_and_numerical_payload(prepared, execution, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy.integrate")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    inputs, description, engine, manifest = binding(prepared, execution)
    result = consume(prepared, inputs, description, engine, manifest)
    assert result is not prepared
    assert result.input_json == prepared.input_json
    assert result._resolved_json == prepared._resolved_json
    assert words(result.to_mapping()) == words(prepared.to_mapping())
    assert result.content_sha256 == prepared.content_sha256 and not result.can_execute
    assert result.to_mapping()["experiment"]["state_history"]["generated"] is False


@pytest.mark.parametrize("role", ["protocol", "preparation", "initial_state", "grid", "numeric_controls"])
def test_changed_upstream_role_rejected_even_if_its_new_document_is_well_formed(prepared, execution, role):
    inputs, description, engine, manifest = binding(prepared, execution)
    old = getattr(description, role)
    changed = replace(description, **{role: SourceDocument(old.id, old.content + b"\n")})
    with pytest.raises(ValueError, match="differ"):
        verify_execution_manifest(manifest, inputs, changed, engine=engine,
                                  environment=description.environment, expected_sha256=manifest.sha256)


def test_environment_engine_expected_pin_and_build_are_independently_checked(prepared, execution):
    inputs, description, engine, manifest = binding(prepared, execution)
    with pytest.raises(ValueError, match="frozen"):
        consume(prepared, inputs, description, engine, manifest, expected_sha256="0" * 64)
    environment = replace(description.environment, settings=(("threads", "2"),))
    with pytest.raises(ValueError, match="environment"):
        consume(prepared, inputs, description, engine, manifest, environment=environment)
    with pytest.raises(ValueError, match="engine"):
        consume(prepared, inputs, description, engine, manifest, engine=ExecutionEngine("legacy", engine_behavior(inputs)))
    build = replace(description.implementation, build_source_commit="1" * 40)
    with pytest.raises(ValueError, match="implementation"):
        consume(prepared, inputs, replace(description, implementation=build), engine, manifest)
    source = description.implementation.files[0]
    build = replace(description.implementation, files=(SourceDocument(source.id, source.content + b"\n"),))
    with pytest.raises(ValueError, match="implementation"):
        consume(prepared, inputs, replace(description, implementation=build), engine, manifest)


def test_maintained_head_does_not_relabel_build_source_or_change_execution(prepared, execution):
    inputs, description, engine, manifest = binding(prepared, execution)
    moved = replace(description, current_repository_head="2" * 40)
    assert execution_manifest(inputs, moved, engine=engine).sha256 == manifest.sha256
    verified = verify_execution_manifest(manifest, inputs, moved, engine=engine,
                                         environment=moved.environment, expected_sha256=manifest.sha256)
    assert verified.current_repository_head == moved.current_repository_head
    assert verified.sha256 == manifest.execution_sha256 and not verified.cache_eligible


@pytest.mark.parametrize("change", ["missing_role", "missing_engine", "changed_content", "extra_role", "wrong_count", "unknown_version"])
def test_incomplete_or_self_inconsistent_serialized_manifest_refuses(prepared, execution, change):
    _, _, _, manifest = binding(prepared, execution)
    value = manifest.export()["document"]
    if change == "missing_role":
        del value["content_roles"]["initial_state"]
    elif change == "missing_engine":
        del value["execution_identity"]["payload"]["engine"]
    elif change == "changed_content":
        value["content_roles"]["initial_state"]["content_hex"] = b"other state".hex()
    elif change == "extra_role":
        value["content_roles"]["fallback"] = value["content_roles"]["protocol"]
    elif change == "wrong_count":
        value["content_roles"]["grid"]["bytes"] += 1
    else:
        value["schema"] = "solarlab.execution-manifest.v2"
    with pytest.raises(ValueError):
        ExecutionManifest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), manifest.current_repository_head)


def test_new_history_or_device_cannot_consume_another_jv_manifest(prepared, execution):
    inputs, description, engine, manifest = binding(prepared, execution)
    for target in ("history", "device"):
        raw = deepcopy(prepared.to_input().editing_data())
        if target == "history":
            raw["experiment"]["waveform"]["turnaround_s"] = "4 s"
        else:
            raw["device"]["settings"]["T"] = "301 K"
        with pytest.raises(ValueError, match="J-V"):
            consume_jv_manifest(JVExperimentInput.model_validate(raw), prepared.defaults, prepared.resources,
                manifest=manifest, inputs=inputs, execution=description, engine=engine,
                environment=description.environment, expected_sha256=manifest.sha256, sources=prepared.sources)


def test_pending_operating_voltage_and_steady_driver_are_not_full_transient_manifests(defaults, resources):
    for controls in ({}, {"solver": "quasi_fermi", "V_max": "1 V"}):
        raw, sources = request(defaults, **controls)
        prepared = prepare_jv_experiment(JVExperimentInput.model_validate(raw), defaults, resources, sources=sources)
        with pytest.raises(ValueError, match="complete transient"):
            prepared.execution_documents()


def test_real_sweep_consumption_is_fresh_complete_and_order_independent(prepared, execution):
    raw = single(prepared.to_input(), family="jv", owner_id=None, parameter=("v_rate",), values=("20 mV/s", "40 mV/s", "60 mV/s"))
    sweep = prepare_sweep(SweepInput.model_validate(raw), prepared.defaults, prepared.resources, sources=prepared.sources)
    points = sweep.to_mapping()["points"]
    original = sweep._resolved_json
    records = {}
    for point in points:
        direct = prepare_point(point["applied_input"], prepared.defaults, prepared.resources, prepared.sources)
        assert isinstance(direct, PreparedJVExperiment)
        records[point["id"]] = direct, binding(direct, execution)
    order = list(records)
    random.Random(1847).shuffle(order)
    for point_id in [*records, *order]:
        direct, (inputs, description, engine, manifest) = records[point_id]
        result = sweep.consume_point_manifest(point_id, manifest, inputs, description, engine=engine,
            environment=description.environment, expected_sha256=manifest.sha256)
        assert result is not direct and result.input_json == direct.input_json
        assert result._resolved_json == direct._resolved_json
        assert not result.can_execute
    assert sweep._resolved_json == original
    _, (inputs, description, engine, manifest) = records[points[0]["id"]]
    with pytest.raises(ValueError, match="J-V protocol"):
        sweep.consume_point_manifest(points[1]["id"], manifest, inputs, description, engine=engine,
            environment=description.environment, expected_sha256=manifest.sha256)
    with pytest.raises(ValueError, match="resolved declared"):
        sweep.consume_point_manifest("missing", manifest, inputs, description, engine=engine,
            environment=description.environment, expected_sha256=manifest.sha256)

"""Draft schema/serialization checks only; no state or model is executed."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from solarlab.experiments.protocol import Protocol, WorkingPoint, protocol_schema, working_point_schema
from solarlab.experiments.results import (
    EVIDENCE_LABELS, ExperimentResult, array_schema, decode_array, encode_array, result_schema,
)


def ref(name):
    return {"id": "serialization-fixture:" + name, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def protocol():
    return {"schema": "solarlab.protocol.v1", "id": "schema-test", "version": 1, "experiment_id": "jv",
        "preparation": {"schema": "test.preparation.v1", "hold_s": 1e-6, "initial_voltage_V": -0.0},
        "waveform": {"schema": "test.waveform.v1", "nodes": [[0., 0.], [1e-6, .02]], "illuminated": True},
        "switches": {"schema": "test.switches.v1", "events": []},
        "sampling": {"schema": "test.sampling.v1", "times_s": [0., 1e-6]},
        "control_mode": "voltage", "phases": ["preparation", "scan"], "transfers": [{
            "from": "preparation", "to": "scan", "working_point": ref("initial-working-point"),
            "preserve": ["state", "inventories", "coordinate_reference", "materials", "damage", "grid", "history"]}]}


def point():
    return {"schema": "solarlab.working-point.v1", "id": "initial", "engine": "legacy", "owner_epoch": "test.epoch1",
        "state_encoding": "test.raw-f64", "physical_identity": ref("physical")["sha256"],
        **{name: ref(name) for name in ("state", "inventories", "coordinate_reference", "materials", "damage", "grid", "history", "qualification")}}


def result(outcome="completed", label="FIRST_SLICE"):
    return {"schema": "solarlab.experiment-result.v1", "header": {
        "run_id": "schema-test", "attempt_id": "attempt1", "experiment_id": "impedance", "engine": "legacy",
        "manifest_sha256": ref("manifest")["sha256"], "outcome": outcome,
        "reason": None if outcome == "completed" else "explicit_test_failure",
        "bindings": {key: ref(key) for key in ("source", "schema", "registry", "physics", "protocol", "initial", "numerics", "resources")},
        "evidence": [{"label": label, "scope": ["serialization_fixture"], "sources": [ref("test-contract")],
                      "limitations": ["No physical execution or qualification"]}]},
        "payload_schema": "test.impedance.v1", "payload": {"raw_unknown_experiment_field": {"present": None},
            "impedance": encode_array(np.array([1 - 2j, complex(-0., 0.)]), unit="ohm m^2")},
        "metrics": {"tau": {"status": "unidentified", "value": None, "reason": "no_identified_peak"},
                    "voltage": {"status": "identified", "value": -0.0, "reason": None}}}


def test_protocol_roundtrip_keeps_all_sections_presence_and_signed_zero():
    raw = protocol()
    item = Protocol.from_document(raw)
    Draft202012Validator(protocol_schema()).validate(item.export())
    assert Protocol(item.canonical_json) == item
    assert item.export() == raw
    raw["sampling"]["times_s"].append(1.)
    assert item.export()["sampling"]["times_s"] == [0., 1e-6]
    assert np.signbit(item.export()["preparation"]["initial_voltage_V"])
    changed = item.export()
    changed["preparation"]["initial_voltage_V"] = 0.0
    assert Protocol.from_document(changed).sha256 != item.sha256
    with pytest.raises(FrozenInstanceError):
        item.canonical_json = "{}"


@pytest.mark.parametrize("edit", [
    lambda x: x.pop("switches"), lambda x: x.update(version=True), lambda x: x.update(schema="unknown"),
    lambda x: x["transfers"][0].update(to="unknown_phase"),
    lambda x: x["transfers"][0].update(preserve=["automatically_reset_to_steady_state"]),
    lambda x: x["preparation"].update(hold_s=float("inf")),
])
def test_bad_protocol_declarations_refuse(edit):
    raw = protocol()
    edit(raw)
    with pytest.raises(ValueError):
        Protocol.from_document(raw)


def test_working_point_keeps_physical_and_encoding_ids_and_damage_history_separate():
    a = WorkingPoint.from_document(point())
    Draft202012Validator(working_point_schema()).validate(a.export())
    other = a.export()
    other.update(state=ref("alternate-encoded-words"), state_encoding="test.pair-words")
    b = WorkingPoint.from_document(other)
    assert a.sha256 != b.sha256
    assert a.export()["physical_identity"] == b.export()["physical_identity"]
    # Identity equality alone performs no physical equivalence/state transfer.
    assert not hasattr(a, "advance") and not hasattr(a, "rebind")
    for field in ("materials", "damage", "grid", "history", "owner_epoch"):
        changed = a.export()
        changed[field] = "epoch2" if field == "owner_epoch" else ref("changed-" + field)
        assert WorkingPoint.from_document(changed).sha256 != a.sha256
    missing = point()
    del missing["inventories"]
    with pytest.raises(ValueError):
        WorkingPoint.from_document(missing)


@pytest.mark.parametrize("outcome", ("completed", "failed", "partial"))
@pytest.mark.parametrize("label", EVIDENCE_LABELS)
def test_results_preserve_original_scoped_labels_and_partial_unknown_metrics(outcome, label):
    raw = result(outcome, label)
    item = ExperimentResult.from_document(raw)
    Draft202012Validator(result_schema()).validate(item.export())
    assert item.export() == raw
    assert ExperimentResult(item.canonical_json) == item
    assert item.export()["header"]["evidence"][0]["label"] == label
    raw["payload"].clear()
    assert item.export()["payload"]
    assert np.signbit(item.export()["metrics"]["voltage"]["value"])


@pytest.mark.parametrize("edit", [
    lambda x: x["header"]["bindings"].pop("initial"),
    lambda x: x["header"]["evidence"][0].update(label="PASSED"),
    lambda x: x["header"]["evidence"][0].update(scope=[]),
    lambda x: x["header"].update(outcome="failed", reason=None),
    lambda x: x["metrics"]["tau"].update(value=1.),
    lambda x: x["metrics"]["voltage"].update(value=float("nan")),
])
def test_false_or_incomplete_result_claims_refuse(edit):
    raw = result()
    edit(raw)
    with pytest.raises(ValueError):
        ExperimentResult.from_document(raw)


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8", "<c8", ">c8", "<c16", ">c16", "<i8", ">u8", "|b1"])
def test_array_roundtrip_is_bitwise_and_endian_exact(dtype):
    array = np.array([0, 1, 2, 3], dtype=dtype).reshape(2, 2)
    if array.dtype.kind in "fc":
        array.real[0, 0] = -0.
        array.real[0, 1] = np.inf
        array.real[1, 0] = np.nan
        if array.dtype.kind == "c":
            array.imag[1, 1] = -np.inf
            array.imag[0, 0] = -0.
    encoded = encode_array(array, unit="1")
    Draft202012Validator(array_schema()).validate(encoded)
    decoded = decode_array(json.loads(json.dumps(encoded, allow_nan=False)))
    assert decoded.dtype == array.dtype and decoded.shape == array.shape
    assert decoded.tobytes() == array.tobytes()


def test_nonfinite_payload_words_empty_arrays_and_noncontiguous_inputs_survive():
    array = np.array([0x7FF8000000000001, 0xFFF8000000000123, 0x8000000000000000, 0], dtype="<u8").view("<f8")
    for original in (array, array[::2], np.empty((0, 3), dtype="<c16"), np.array(-0.)):
        encoded = encode_array(original, unit="V")
        decoded = decode_array(encoded)
        assert decoded.tobytes() == original.tobytes() and decoded.shape == original.shape
    assert encode_array(array, unit="V")["real"]["values"][:2] == [None, None]


def test_noncanonical_boolean_words_refuse_before_logical_normalization():
    original = np.frombuffer(bytes([0, 2, 255]), dtype="?")
    for array in (original, original[::-1], original[1::2]):
        before = array.tobytes()
        with pytest.raises(ValueError, match="noncanonical boolean words"):
            encode_array(array, unit="1")
        assert array.tobytes() == before
    for array in (np.array([False, True]), np.empty((0, 3), dtype="?"), np.array(True)):
        encoded = encode_array(array, unit="1")
        Draft202012Validator(array_schema()).validate(encoded)
        assert decode_array(json.loads(json.dumps(encoded))).tobytes() == array.tobytes()


@pytest.mark.parametrize("rank", [0, 1, 32, 33])
def test_encoder_schema_and_decoder_share_the_rank_boundary(rank):
    array = np.zeros((1,) * rank, dtype="<f8")
    if rank > array_schema()["properties"]["shape"]["maxItems"]:
        with pytest.raises(ValueError, match="rank"):
            encode_array(array, unit="1")
    else:
        encoded = encode_array(array, unit="1")
        Draft202012Validator(array_schema()).validate(encoded)
        decoded = decode_array(json.loads(json.dumps(encoded)))
        assert decoded.shape == array.shape and decoded.tobytes() == array.tobytes()


@pytest.mark.parametrize("edit", [
    lambda x: x.update(shape=[100000000000]), lambda x: x.update(dtype="|O"),
    lambda x: x["real"]["mask"].__setitem__(0, False),
    lambda x: x["real"]["reasons"].__setitem__(0, None),
    lambda x: x["real"]["nonfinite_words"].__setitem__(0, "0000000000000000"),
    lambda x: x["real"]["values"].__setitem__(1, None),
    lambda x: x.update(imag=deepcopy(x["real"])),
])
def test_inconsistent_array_masks_reasons_shapes_and_words_refuse(edit):
    encoded = encode_array(np.array([np.nan, 1.]), unit="V")
    edit(encoded)
    with pytest.raises((ValueError, TypeError)):
        decode_array(encoded)


def test_exported_schemas_are_valid_draft202012_documents():
    for schema in (protocol_schema(), working_point_schema(), result_schema(), array_schema()):
        Draft202012Validator.check_schema(schema)

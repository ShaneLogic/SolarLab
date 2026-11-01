"""Source drift is metadata; artifact corruption and strict checks still fail."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from solarlab.reproducibility import job_provenance as provenance


def admitted(content: bytes = b"source", **changes: Any) -> provenance.JobProvenance:
    options = {"git_commit": "a" * 40, "source_files": (("solarlab/source.py", content),),
               "environment_lock": b"locked environment", "source_attested": True,
               "operator": "test.bound_operator", "assembler_identity": "b" * 64, "analysis_identity": "c" * 64,
               "dirty_policy": "record"}
    options.update(changes)
    return provenance.capture_job_provenance({"input": "original"}, **options)


def test_record_is_canonical_immutable_and_keeps_separate_input_and_tree_digests():
    record = admitted()
    envelope = record.envelope()
    assert provenance.read_provenance(envelope) == record
    assert envelope["sha256"] == hashlib.sha256(record.canonical_json).hexdigest()
    assert envelope["record"]["input_sha256"] == provenance.digest({"input": "original"})
    assert envelope["record"]["source_tree_sha256"] == provenance.digest(envelope["record"]["source_files"])
    envelope["record"]["source_files"].clear()
    assert record.export()["source_files"]


def test_source_drift_never_relabels_record_or_looks_up_files_during_comparison(monkeypatch):
    producer, current = admitted(), admitted(b"refactored source")
    original = producer.canonical_json
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("readback consulted current source"))
    monkeypatch.setattr(provenance, "_git", lambda *_: pytest.fail("readback called Git"))
    result = provenance.compare_provenance(producer.envelope(), current)
    assert result["status"] == "different"
    assert "source_tree_sha256" in result["different_fields"]
    assert producer.canonical_json == original
    with pytest.raises(ValueError, match="strict_same_source"):
        provenance.compare_provenance(producer.envelope(), current, strict_same_source=True)
    assert provenance.compare_provenance(producer.envelope(), producer, strict_same_source=True)["status"] == "same"


@pytest.mark.parametrize("mutation", ["outer_digest", "source_record", "tree_digest", "commit"])
def test_producer_record_tampering_is_not_treated_as_current_source_drift(mutation):
    value = admitted().envelope()
    if mutation == "outer_digest":
        value["sha256"] = "0" * 64
    elif mutation == "source_record":
        value["record"]["source_files"][0]["sha256"] = "0" * 64
    elif mutation == "tree_digest":
        value["record"]["source_tree_sha256"] = "0" * 64
    else:
        value["record"]["git_commit"] = "guessed-HEAD"
    with pytest.raises(ValueError):
        provenance.compare_provenance(value)


def test_unrecorded_historical_identity_and_missing_lock_are_unknown_not_certified():
    assert provenance.compare_provenance(None)["status"] == "unknown"
    for record in (admitted(environment_lock=None), admitted(source_attested=False)):
        assert provenance.compare_provenance(record.envelope(), record)["status"] == "unknown"
        with pytest.raises(ValueError, match="strict_same_source"):
            provenance.compare_provenance(record.envelope(), record, strict_same_source=True)
    with pytest.raises(ValueError, match="actual source inventory"):
        admitted(source_files=())


def test_observed_runtime_mismatch_is_reported_and_refused_only_in_strict_mode():
    producer = admitted()
    data = producer.export()
    data["runtime_observed"]["python"] = "different"
    current = provenance.JobProvenance.from_record(data)
    assert provenance.compare_provenance(producer.envelope(), current)["different_fields"] == ["runtime_observed"]
    with pytest.raises(ValueError, match="strict_same_source"):
        provenance.compare_provenance(producer.envelope(), current, strict_same_source=True)


def test_dirty_admission_records_actual_bytes_or_refuses_and_nested_capture_is_once(tmp_path, monkeypatch):
    source = tmp_path / "src/solarlab/a.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    (tmp_path / "uv.lock").write_bytes(b"locked")

    def git(root, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[0] == "diff":
            return "src/solarlab/a.py\0"
        if "--cached" in args:
            return "src/solarlab/a.py\0"
        return ""

    monkeypatch.setattr(provenance, "_git", git)
    with pytest.raises(ValueError, match="clean attested"):
        provenance.capture_job_provenance({}, source_root=tmp_path, dirty_policy="refuse", operator="test")
    captures = []
    original_capture = provenance.capture_job_provenance

    def capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        captures.append(result)
        return result

    monkeypatch.setattr(provenance, "capture_job_provenance", capture)
    with provenance.provenance_owner(source_root=tmp_path, dirty_policy="record"):
        with provenance.job_provenance({"job": 1}, operator="outer") as original:
            assert original.export()["source_state"] == "dirty_recorded"
            assert provenance.current_provenance_ref() == original.sha256
            envelope = deepcopy(provenance.current_provenance())
            source.write_bytes(b"edited after admission")
            with provenance.job_provenance({"nested": True}, operator=None) as inner:
                assert inner is original
                assert provenance.current_provenance() == envelope
            assert len(captures) == 1
    assert provenance.current_provenance() is None


def test_supplied_installed_snapshot_does_not_require_a_checkout(monkeypatch):
    monkeypatch.setattr(provenance, "execution_checkout", lambda: pytest.fail("installed snapshot consulted checkout"))
    with provenance.provenance_owner(source_files=(("installed/a.py", b"body"),),
            git_commit="a" * 40, environment_lock=b"lock", source_attested=True, dirty_policy="refuse"):
        with provenance.job_provenance({"input": 1}, operator="installed") as record:
            assert record.export()["source_state"] == "clean"


def test_outer_owner_captures_once_for_sequential_producers(monkeypatch):
    record = admitted(operator="initial", assembler_identity=None, analysis_identity=None)
    captures = []
    def capture(inputs, **kwargs):
        captures.append((inputs, kwargs))
        return record
    monkeypatch.setattr(provenance, "capture_job_provenance", capture)
    with provenance.provenance_owner(input_record={"whole_job": True}):
        with provenance.job_provenance({"stage": 1}, operator="initial") as first:
            pass
        with provenance.job_provenance({"stage": 2}, operator=None) as second:
            assert provenance.current_provenance_ref() == first.sha256
    assert first == second and len(captures) == 1
    assert captures[0][0] == {"whole_job": True}
    assert provenance.current_provenance() is None


def test_incomplete_or_inconsistent_assembler_identity_never_passes_strict():
    data = admitted().export()
    data["assembler"]["identity"] = None
    with pytest.raises(ValueError, match="Assembler identity"):
        provenance.JobProvenance.from_record(data)
    data = admitted().export()
    data["assembler"]["analysis_identity"] = None
    incomplete = provenance.JobProvenance.from_record(data)
    with pytest.raises(ValueError, match="strict_same_source"):
        provenance.compare_provenance(incomplete.envelope(), incomplete, strict_same_source=True)


def test_job_receipt_is_bound_to_unchanged_transport_and_attempt():
    transport = {"H_input": "1" * 64, "H_physics": "2" * 64,
                 "documents": {"request": {"sha256": "3" * 64}}}
    token = {"store_id": "s", "run_id": "r", "attempt_id": "a", "owner_id": "owner", "generation": 1}
    value = admitted().export()
    value["attempt_ref"] = token
    value["execution_manifest_sha256"] = "5" * 64
    document = {"content_roles": {k: {"sha256": "9" * 64} for k in ("protocol", "initial_state", "grid", "numeric_controls")}}
    value["input_sha256"] = provenance.digest(provenance.worker_input_record(transport, document))
    producer = {"job_provenance": provenance.JobProvenance.from_record(value).envelope(), "manifest_sha256": "5" * 64}
    run = {"run_id": "r", "request": {"input_metadata": {"transport": transport,
        "execution_manifest": {"manifest_sha256": "5" * 64}}}}
    attempt = {"attempt_id": "a", "owner_id": "owner", "generation": 1}
    assert provenance.worker_result_provenance(producer, run, attempt, store_id="s", provenance_ref=producer["job_provenance"]["sha256"], execution_document=document)["status"] == "unknown"
    assert provenance.worker_result_provenance(producer, run, attempt, store_id="s",
        provenance_ref=producer["job_provenance"]["sha256"], execution_document=document,
        current=producer["job_provenance"], strict_same_source=True)["status"] == "same"
    with pytest.raises(ValueError, match="RunStore"):
        provenance.result_provenance(producer, producer["job_provenance"], strict_same_source=True)
    with pytest.raises(ValueError, match="another attempt"):
        provenance.worker_result_provenance(producer, run, {**attempt, "generation": 2}, store_id="s", provenance_ref=producer["job_provenance"]["sha256"], execution_document=document)
    transport["documents"]["request"]["sha256"] = "4" * 64
    with pytest.raises(ValueError, match="input differs"):
        provenance.worker_result_provenance(producer, run, attempt, store_id="s", provenance_ref=producer["job_provenance"]["sha256"], execution_document=document)


def test_saved_legacy_pin_cannot_be_relabelled_under_a_valid_new_envelope():
    record = admitted()
    original = {"source_sha256": {"solarlab.source": hashlib.sha256(b"source").hexdigest()},
                "job_provenance": record.envelope()}
    assert provenance.result_provenance(original)["producer_identity"] == record.envelope()
    original["source_sha256"]["solarlab.source"] = "0" * 64
    with pytest.raises(ValueError, match="producer provenance"):
        provenance.result_provenance(original)


def test_capture_refuses_a_mutating_snapshot_before_computation(tmp_path, monkeypatch):
    path = tmp_path / "src/solarlab/example.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"before")
    def git(root, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if "--cached" in args:
            return "src/solarlab/example.py\0"
        return ""
    monkeypatch.setattr(provenance, "_git", git)
    original = Path.read_bytes
    def changing(self):
        data = original(self)
        if self == path:
            path.write_bytes(b"changed during capture")
        return data
    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(ValueError, match="changed during admission"):
        provenance.capture_job_provenance({}, operator="test", source_root=tmp_path, dirty_policy="record")


def test_missing_git_makes_implicit_execution_checkout_unknown(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("git is not installed")
    monkeypatch.setattr(provenance, "_git", missing)
    assert provenance.execution_checkout() is None
    with provenance.job_provenance({}, operator="ordinary published experiment") as record:
        assert record.export()["source_state"] == "unknown"


def test_late_dirty_source_cannot_be_mislabelled_clean(tmp_path, monkeypatch):
    path = tmp_path / "src/solarlab/example.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"snapshot")
    calls = []
    def git(root, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[0] == "diff":
            calls.append(args)
            return "" if len(calls) == 1 else "src/solarlab/example.py\0"
        if "--cached" in args:
            return "src/solarlab/example.py\0"
        return ""
    monkeypatch.setattr(provenance, "_git", git)
    with pytest.raises(ValueError, match="changed during admission"):
        provenance.capture_job_provenance({}, operator="test", source_root=tmp_path, dirty_policy="record")


@pytest.mark.parametrize("change", ["owner", "generation", "manifest", "inputs", "swapped_valid_record"])
def test_valid_provenance_cannot_be_swapped_across_publication_bindings(change):
    record = admitted().export()
    token = {"store_id": "s", "run_id": "r", "attempt_id": "a", "owner_id": "owner", "generation": 1}
    transport = {"H_input": "1" * 64, "H_physics": "2" * 64, "documents": {"request": {"sha256": "3" * 64}}}
    document = {"content_roles": {k: {"sha256": "9" * 64} for k in ("protocol", "initial_state", "grid", "numeric_controls")}}
    inputs = provenance.worker_input_record(transport, document)
    record.update(attempt_ref=token.copy(), execution_manifest_sha256="4" * 64, input_sha256=provenance.digest(inputs))
    original = provenance.JobProvenance.from_record(record)
    if change in {"owner", "generation"}:
        record["attempt_ref"]["owner_id" if change == "owner" else "generation"] = "different" if change == "owner" else 2
    elif change == "manifest":
        record["execution_manifest_sha256"] = "5" * 64
    elif change == "inputs":
        record["input_sha256"] = "6" * 64
    else:
        record["git_commit"] = "d" * 40
    changed = provenance.JobProvenance.from_record(record)
    result = {"job_provenance": changed.envelope(), "manifest_sha256": "4" * 64}
    run = {"run_id": "r", "request": {"input_metadata": {"transport": transport,
           "execution_manifest": {"manifest_sha256": "4" * 64}}}}
    # Even a freshly recomputed local envelope checksum cannot replace the
    # reference sealed into the independently checked execution artifact.
    reference = original.sha256 if change == "swapped_valid_record" else changed.sha256
    with pytest.raises(ValueError):
        provenance.worker_result_provenance(result, run, token, store_id="s", provenance_ref=reference, execution_document=document)


def test_standalone_producer_inputs_and_publication_owner_are_separate():
    record = admitted()
    producer = {"job_provenance": record.envelope()}
    provenance.result_provenance(producer, expected_inputs={"input": "original"})
    with pytest.raises(ValueError, match="different inputs"):
        provenance.result_provenance(producer, expected_inputs={"input": "changed"})
    owner = {"store_id": "s", "run_id": "r", "attempt_id": "a", "owner_id": "o", "generation": 1}
    first = provenance.publication_binding(record.envelope(), owner=owner,
        request_identity={"H_input": "original"}, result_sha256="a" * 64)
    second = provenance.publication_binding(record.envelope(), owner={**owner, "owner_id": "changed"},
        request_identity={"H_input": "original"}, result_sha256="a" * 64)
    assert first["sha256"] != second["sha256"]
    assert first["record"]["admission_sha256"] == second["record"]["admission_sha256"] == record.sha256
    assert provenance.digest(first["record"]) == first["sha256"]


def test_strict_source_requires_known_scope_and_explicit_assembled_analysis():
    value = admitted().export()
    value["source_scope"] = "unrecognized-future-algorithm"
    unknown = provenance.JobProvenance.from_record(value)
    with pytest.raises(ValueError, match="strict_same_source"):
        provenance.compare_provenance(unknown.envelope(), unknown, strict_same_source=True)
    with pytest.raises(ValueError, match="analysis-operation"):
        admitted(analysis_identity=None)
    legacy = admitted(assembler_identity=None, analysis_identity=None)
    assert legacy.export()["assembler"]["unified_assembler"] is False
    assert legacy.export()["assembler"]["identity"] is None
    changed = admitted(analysis_identity="d" * 64)
    with pytest.raises(ValueError, match="strict_same_source"):
        provenance.compare_provenance(admitted().envelope(), changed, strict_same_source=True)


def test_installed_overlays_are_explicitly_dirty_not_committed_clean():
    record = admitted(source_dirty_paths=("solarlab/source.py",))
    assert record.export()["source_state"] == "dirty_recorded"
    assert record.export()["dirty_paths"] == ["solarlab/source.py"]
    with pytest.raises(ValueError, match="clean attested"):
        admitted(source_dirty_paths=("solarlab/source.py",), dirty_policy="refuse")


def test_public_steady_2d_sources_only_returns_admitted_metadata(monkeypatch):
    from solarlab.solvers.nonlinear.steady_2d import sources

    assert sources() == {}
    with provenance.provenance_owner(input_record={"input": "original"},
            git_commit="a" * 40, source_files=(("solarlab/source.py", b"source"),),
            environment_lock=b"locked environment", source_attested=True, dirty_policy="record"):
        with provenance.job_provenance({}, operator="test.bound_operator"):
            expected = provenance.admitted_source_pins()
            assert expected
            monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("public accessor inspected source"))
            monkeypatch.setattr(provenance, "capture_job_provenance", lambda *a, **k: pytest.fail("public accessor recaptured provenance"))
            assert sources() == expected
            emitted = sources()
            emitted.clear()
            assert sources() == expected
    assert sources() == {}


def test_fixture_admission_refuses_foreign_installed_bytes_before_capture(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from support import provenance as fixture

    foreign = tmp_path / "solarlab" / "__init__.py"
    foreign.parent.mkdir()
    foreign.write_text("# unrelated installed source\n")
    real_find_spec = fixture.importlib.util.find_spec
    monkeypatch.setattr(fixture.importlib.util, "find_spec", lambda name:
        SimpleNamespace(origin=str(foreign)) if name == "solarlab" else real_find_spec(name))
    monkeypatch.setattr(fixture, "provenance_owner", lambda **kw: pytest.fail("unverified admission"))
    with pytest.raises(ValueError, match="installed fixture source differs"):
        with fixture.checked_provenance_owner():
            pytest.fail("foreign installed source was admitted")

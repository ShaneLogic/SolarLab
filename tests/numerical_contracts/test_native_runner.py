"""Infrastructure-only tests of the real save bodies and maintained runner.

The scientific controller boundary is injected; no controller initialization,
IDA, native solver, trajectory, equation or physical qualification is executed.
"""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmarks.native_history import HistoryLimitError, HistoryReader, HistoryWriter, METADATA_RESERVE, emit_record
from scripts.benchmarks import native_runner

ROOT = Path(__file__).resolve().parents[2]
OWNERS = ("run_affine_native_pilot", "run_voltage_lift_native_pilot")


class BoundaryContractError(ValueError):
    """Injected exception boundary; the extracted real save body is unchanged."""


def save_body(owner, emit, cap):
    path = ROOT / "scripts/benchmarks/coupled_device_prototype.py"
    tree = ast.parse(path.read_text())
    controller = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == owner)
    save = next(n for n in controller.body if isinstance(n, ast.FunctionDef) and n.name == "save")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "emit_record"
               for n in ast.walk(save)), "actual controller save path is not integrated"
    counts = {"history_bytes": 0, "native_steps": 0}
    env = {"emit": emit, "budgets": {"total_output_bytes": cap}, "counts": counts,
           "emit_record": emit_record, "HistoryLimitError": HistoryLimitError,
           "ContractError": BoundaryContractError, "json": json}
    exec(compile(ast.Module(body=[save], type_ignores=[]), str(path), "exec"), env)
    return env["save"], counts


def accepted_record():
    folder = ROOT / "tests/fixtures/native_history"
    spec = json.loads((folder / "manifest.json").read_text())["accepted"]
    raw = (folder / spec["path"]).read_bytes()
    if "stored_sha256" in spec:
        assert hashlib.sha256(raw).hexdigest() == spec["stored_sha256"]
    if spec.get("encoding") == "gzip":
        raw = gzip.decompress(raw)
    assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
    return json.loads(raw)


def read_rows(path):
    with HistoryReader(path, encoding="gzip", max_record_bytes=131072, max_logical_bytes=1048576) as reader:
        rows = list(reader.records())
        assert reader.container_complete is True
        return rows


@pytest.mark.parametrize("owner", OWNERS)
def test_actual_save_paths_raw_callback_and_explicit_writer(tmp_path, owner):
    record = {"kind": "infrastructure_only", "repeat": "x" * 8192, "zero": 0, "false": False, "null": None}
    raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    captured = []
    save, counts = save_body(owner, captured.append, METADATA_RESERVE + len(raw))
    save(record)
    assert captured == [record] and captured[0] is record
    assert counts == {"history_bytes": len(raw), "native_steps": 0}
    refused, untouched = save_body(owner, captured.append, METADATA_RESERVE + len(raw) - 1)
    with pytest.raises(BoundaryContractError, match="history_budget"):
        refused(record)
    assert untouched["history_bytes"] == 0 and len(captured) == 1

    cap = METADATA_RESERVE + 1000
    folder = tmp_path / "gzip"; folder.mkdir()
    with HistoryWriter(folder / "history.gz", encoding="gzip", total_output_bytes=cap) as writer:
        save, counts = save_body(owner, writer, cap)
        save(record)  # This raw size could not pass the unchanged legacy raw gate.
        assert counts["history_bytes"] == len(raw) > 1000
        assert counts["history_encoded_bytes"] == writer.encoded_bytes < 1000
        assert counts["native_steps"] == 0
    assert read_rows(writer.path) == [record]

    folder = tmp_path / "wrapped"; folder.mkdir()
    with HistoryWriter(folder / "history.gz", encoding="gzip", total_output_bytes=cap) as writer:
        save, counts = save_body(owner, lambda value: writer(value), cap)
        with pytest.raises(BoundaryContractError, match="history_budget"):
            save(record)
        assert writer.logical_bytes == counts["history_bytes"] == 0


@pytest.mark.parametrize("owner", OWNERS)
def test_actual_save_first_failure_immediate_and_write_refusal(tmp_path, owner):
    first = {"kind": "first_callback_failure", "exception": "InfrastructureSentinel", "reason": "first", "zero": 0, "null": None}
    cap = METADATA_RESERVE + 4096
    with HistoryWriter(tmp_path / "history.gz", encoding="gzip", total_output_bytes=cap) as writer:
        save, counts = save_body(owner, writer, cap)
        save(first)
        assert json.loads((tmp_path / "FirstFailure.json").read_text()) == first
        save({**first, "reason": "later"})
        assert json.loads((tmp_path / "FirstFailure.json").read_text()) == first
        assert counts["history_bytes"] == writer.logical_bytes
    assert len(read_rows(writer.path)) == 2

    folder = tmp_path / "refused"; folder.mkdir()
    cap = METADATA_RESERVE + 20  # Header and reserved footer only; no record fits.
    writer = HistoryWriter(folder / "history.gz", encoding="gzip", total_output_bytes=cap)
    save, counts = save_body(owner, writer, cap)
    with pytest.raises(BoundaryContractError, match="history_budget"):
        save(first)
    assert json.loads((folder / "FirstFailure.json").read_text()) == first
    assert writer.closed and writer.first_exception is not None
    assert writer.logical_bytes == counts["history_bytes"] == 0
    assert writer.path.stat().st_size == 10
    assert sum(p.stat().st_size for p in folder.iterdir()) <= cap


def test_runner_finishes_before_successful_result_and_binds_actual_record(tmp_path, monkeypatch):
    events = []
    original_finish = HistoryWriter.finish
    original_write = native_runner._write_document
    def finish(writer):
        position = original_finish(writer)
        assert read_rows(writer.path) == [record]
        events.append("finished")
        return position
    def publish(folder, name, value, **kwargs):
        if name == "NativeResult.json":
            assert events == ["finished"]
            assert value["history"]["container_complete"] is True
            events.append("result")
        return original_write(folder, name, value, **kwargs)
    monkeypatch.setattr(HistoryWriter, "finish", finish)
    monkeypatch.setattr(native_runner, "_write_document", publish)
    record = accepted_record()
    raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    cap = 2 * METADATA_RESERVE
    def controller(writer):
        assert isinstance(writer, HistoryWriter)
        save, counts = save_body(OWNERS[1], writer, cap)
        save(record)
        return {"status": native_runner.COMPLETED_STATUS, "complete_protocol": True,
                "counts": counts, "fixture_scope": "nonnative recording test"}
    result = native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert events == ["finished", "result"]
    stored = json.loads((tmp_path / "NativeResult.json").read_text())
    assert stored == result and stored["counts"]["history_bytes"] == len(raw)
    assert stored["counts"]["history_encoded_bytes"] == (tmp_path / "NativeHistory.jsonl.gz").stat().st_size
    assert stored["history"]["logical_sha256"] == hashlib.sha256(raw).hexdigest()
    pointer = json.loads((tmp_path / "LastAccepted.json").read_text())
    assert pointer["right"] == record["right"] and pointer["frame"] == record["coefficient_frame_identity"]
    assert pointer["pointer"]["history_byte_offset_after_record"] == len(raw)
    assert native_runner.output_bytes(tmp_path, ()) <= cap


def test_runner_finalization_failure_never_publishes_success(tmp_path, monkeypatch):
    failure = OSError("injected finalization failure")
    cap = 2 * METADATA_RESERVE
    def cannot_finish(_writer):
        raise failure
    monkeypatch.setattr(HistoryWriter, "finish", cannot_finish)
    def controller(writer):
        save, counts = save_body(OWNERS[1], writer, cap)
        save({"kind": "infrastructure_only", "value": 0})
        return {"status": native_runner.COMPLETED_STATUS, "complete_protocol": True, "counts": counts}
    with pytest.raises(OSError) as caught:
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert caught.value is failure and not (tmp_path / "NativeResult.json").exists()
    assert json.loads((tmp_path / "RunnerFailure.json").read_text())["reason"] == str(failure)
    rows = []
    with HistoryReader(tmp_path / "NativeHistory.jsonl.gz", encoding="gzip", max_record_bytes=1024, max_logical_bytes=4096) as reader:
        with pytest.raises(EOFError):
            for line in reader:
                rows.append(json.loads(line))
        assert reader.container_complete is False
    assert rows == [{"kind": "infrastructure_only", "value": 0}]


def test_runner_preserves_first_callback_and_primary_exception_on_secondary_failure(tmp_path, monkeypatch):
    first = {"kind": "first_callback_failure", "reason": "first callback", "zero": 0, "null": None}
    primary = RuntimeError("primary controller exception")
    original_write = native_runner._write_document
    def reject_failure(folder, name, value, **kwargs):
        if name == "RunnerFailure.json":
            raise OSError("secondary report failure")
        return original_write(folder, name, value, **kwargs)
    monkeypatch.setattr(native_runner, "_write_document", reject_failure)
    cap = 2 * METADATA_RESERVE
    def controller(writer):
        save, _counts = save_body(OWNERS[1], writer, cap)
        save(first)
        assert json.loads((tmp_path / "FirstFailure.json").read_text()) == first
        save({**first, "reason": "second callback"})
        raise primary
    with pytest.raises(RuntimeError) as caught:
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert caught.value is primary
    assert any("secondary report failure" in note for note in primary.__notes__)
    assert json.loads((tmp_path / "FirstFailure.json").read_text()) == first
    assert not (tmp_path / "NativeResult.json").exists()


def test_runner_total_artifact_cap_before_result_publication(tmp_path):
    cap = METADATA_RESERVE + 4096
    def controller(writer):
        save, counts = save_body(OWNERS[1], writer, cap)
        save({"kind": "infrastructure_only", "value": 0})
        return {"status": native_runner.COMPLETED_STATUS, "complete_protocol": True,
                "counts": counts, "oversized_fixture": "x" * (2 * METADATA_RESERVE)}
    with pytest.raises(HistoryLimitError, match="total output cap"):
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert not (tmp_path / "NativeResult.json").exists()
    assert native_runner.output_bytes(tmp_path, ()) <= cap
    assert read_rows(tmp_path / "NativeHistory.jsonl.gz") == [{"kind": "infrastructure_only", "value": 0}]
    assert json.loads((tmp_path / "RunnerFailure.json").read_text())["history"]["container_complete"] is True


def test_failed_controller_retains_explicit_incomplete_result(tmp_path):
    cap = METADATA_RESERVE + 20
    def controller(writer):
        save, counts = save_body(OWNERS[1], writer, cap)
        try:
            save({"kind": "first_callback_failure", "reason": "original callback"})
        except BoundaryContractError as error:
            return {"status": "failed_bounded_voltage_lift_native_pilot", "complete_protocol": False,
                    "counts": counts, "reason": str(error), "first_failure": {"reason": "later fallback"}}
        raise AssertionError("the bounded record unexpectedly fit")
    result = native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert result["complete_protocol"] is False and result["history"]["container_complete"] is False
    assert json.loads((tmp_path / "FirstFailure.json").read_text())["reason"] == "original callback"
    assert native_runner.output_bytes(tmp_path, ()) <= cap


def test_maintained_runner_import_is_isolated_from_scientific_backends():
    code = ("import sys; sys.path.insert(0,sys.argv[1]); import scripts.benchmarks.native_runner; "
            "assert not any(name in sys.modules for name in ('numpy','scipy','flint','sksundae'))")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT)],
                            capture_output=True, text=True, timeout=2)
    assert result.returncode == 0, result.stderr

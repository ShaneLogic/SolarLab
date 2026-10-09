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


@pytest.mark.parametrize("policy", [None, {}, {"profile": "frame-input-expansion12-v1"}])
def test_native_producer_binds_request_input_profile(policy):
    tree = ast.parse(Path(native_runner.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "run_recorded")
    request = {} if policy is None else {"frame_input_policy": policy}
    expected_profile = None if policy is None else policy.get("profile")
    model, segments, admission, timing, emitter = (object() for _ in range(5))
    observed = []

    def map_factory(actual_model, *, mapped_input_profile=None):
        assert actual_model is model
        observed.append(mapped_input_profile)
        return object()

    def controller(mapping, actual_segments, actual_request, actual_admission, emit, **kwargs):
        assert observed == [expected_profile]
        assert actual_segments is segments and actual_request is request
        assert actual_admission is admission and emit is emitter
        assert kwargs == {"timing": timing}
        return mapping

    env = {"AffineVoltageMap": map_factory, "run_voltage_lift_native_pilot": controller,
           "model": model, "segments": segments, "request": request,
           "admission": admission, "timing": timing}
    producer = eval(compile(ast.Expression(call.args[1]), native_runner.__file__, "eval"), env)
    assert producer(emitter) is not None


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
           "ContractError": BoundaryContractError, "json": json,
           "timing": native_runner.RuntimeTiming()}
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


def test_failed_controller_retains_primary_writer_error_and_small_envelope(tmp_path):
    cap = METADATA_RESERVE + 20
    def controller(writer):
        save, counts = save_body(OWNERS[1], writer, cap)
        try:
            save({"kind": "first_callback_failure", "reason": "original callback"})
        except BoundaryContractError as error:
            return {"status": "failed_bounded_voltage_lift_native_pilot", "complete_protocol": False,
                    "counts": counts, "reason": str(error), "first_failure": {"reason": "later fallback"}}
        raise AssertionError("the bounded record unexpectedly fit")
    with pytest.raises(HistoryLimitError, match="reservation exceeds cap"):
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=cap, input_names=())
    assert not (tmp_path / "NativeResult.json").exists()
    assert json.loads((tmp_path / "PrimaryFailure.json").read_text())["controller"]["status"].startswith("failed")
    assert json.loads((tmp_path / "FirstFailure.json").read_text())["reason"] == "original callback"
    assert native_runner.output_bytes(tmp_path, ()) <= cap


def test_decimal_failure_is_not_masked_by_result_publication(tmp_path):
    enormous = 10**4400
    def controller(writer):
        try:
            writer.append({"kind": "interval_fixture", "large": enormous})
        except ValueError as error:
            return {"status": "failed_bounded_voltage_lift_native_pilot", "complete_protocol": False,
                    "reason": str(error), "exception": type(error).__name__,
                    "counts": {"history_bytes": writer.logical_bytes},
                    "last_attempt": {"phase": "native_return", "success": True, "status": 0},
                    "first_failure": {"phase": "native_return", "success": True},
                    "unpublished_cumulative_integer": enormous}
        raise AssertionError("legacy decimal limit did not fail")
    with pytest.raises(ValueError, match="4300 digits"):
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=2*METADATA_RESERVE, input_names=())
    primary = json.loads((tmp_path / "PrimaryFailure.json").read_text())
    failure = json.loads((tmp_path / "RunnerFailure.json").read_text())
    assert "4300 digits" in primary["controller"]["reason"]
    assert primary["last_native_context"]["success"] is True
    assert "native_history.py" in failure["traceback"]
    assert "_write_document" not in failure["traceback"]
    assert not (tmp_path / "NativeResult.json").exists()


def test_runner_records_applied_policy_and_publishes_large_exact_result(tmp_path):
    from scripts.benchmarks.native_history import decode_integer_values
    policy = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
              "decimal_digits": 4300, "max_integer_bits": 131072}
    large = 10**4400 + 1
    def controller(writer):
        observed = json.loads((tmp_path / "IntegerIOObserved.json").read_text())
        assert observed["actual_decimal_digits"] == 4300 and observed["policy"] == policy
        writer.append({"kind": "integer_fixture", "rational": [large, large+2]})
        return {"status": native_runner.COMPLETED_STATUS, "complete_protocol": True,
                "counts": {"history_bytes": writer.logical_bytes}, "exact_integer": -large}
    result = native_runner.run_recorded(tmp_path, controller, total_output_bytes=2*METADATA_RESERVE,
                                        input_names=(), integer_io_policy=policy)
    stored = decode_integer_values(json.loads((tmp_path / "NativeResult.json").read_text()), policy)
    assert stored == result and stored["exact_integer"] == -large
    assert stored["history"]["integer_io"]["actual_decimal_digits"] == 4300


@pytest.mark.parametrize("explicit_limit", [False, True])
def test_runner_main_record_limit_forwarding_and_exact_large_record(tmp_path, explicit_limit):
    # Evaluate the maintained entry's real recording keywords without executing
    # its model/native construction. The production writer and reader are real.
    tree = ast.parse(Path(native_runner.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "run_recorded")
    cap = 4 * METADATA_RESERVE
    freeze = {"watchdog": {"input_file_names": []}}
    if explicit_limit:
        freeze["writer_limits"] = {"max_record_bytes": 16 * METADATA_RESERVE}
    env = {"freeze": freeze, "request": {"budgets": {"total_output_bytes": cap}},
           "timing": native_runner.RuntimeTiming()}
    kwargs = {kw.arg: eval(compile(ast.Expression(kw.value), native_runner.__file__, "eval"), env)
              for kw in call.keywords}
    record = {"kind": "infrastructure_large_record", "high": [1.0, -0.0],
              "low": [2.0 ** -100, -2.0 ** -100], "payload": "x" * METADATA_RESERVE}
    raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    assert len(raw) > METADATA_RESERVE
    before_footer = []
    def controller(writer):
        assert writer.metadata_reserve_bytes == METADATA_RESERVE
        writer(record)
        before_footer.append(writer.encoded_bytes)
        return {"status": native_runner.COMPLETED_STATUS,
                "counts": {"history_bytes": writer.logical_bytes}}
    if not explicit_limit:
        with pytest.raises(HistoryLimitError):
            native_runner.run_recorded(tmp_path, controller, **kwargs)
        assert not before_footer and not (tmp_path / "NativeResult.json").exists()
        failure = json.loads((tmp_path / "RunnerFailure.json").read_text())
        assert failure["history"]["max_record_bytes"] == METADATA_RESERVE
        assert failure["history"]["records"] == 0
        return
    result = native_runner.run_recorded(tmp_path, controller, **kwargs)
    path = tmp_path / "NativeHistory.jsonl.gz"
    with HistoryReader(path, encoding="gzip", max_record_bytes=16 * METADATA_RESERVE,
                       max_logical_bytes=2 * METADATA_RESERVE) as reader:
        assert list(reader) == [raw]  # Includes every numeric word and signed zero.
        assert reader.container_complete is True  # Validates EOF, CRC and trailer.
    assert result["history"]["max_record_bytes"] == 16 * METADATA_RESERVE
    assert result["history"]["logical_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["history"]["logical_bytes"] == len(raw)
    assert result["history"]["records"] == 1
    assert result["counts"]["history_encoded_bytes"] == path.stat().st_size > before_footer[0]
    pointer = json.loads((tmp_path / "LastRecord.json").read_text())
    assert pointer["history_byte_offset_after_record"] == len(raw)
    assert native_runner.output_bytes(tmp_path, ()) <= cap


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "16777216", None])
def test_runner_rejects_invalid_record_limits_before_controller(tmp_path, limit):
    def controller(_writer):
        raise AssertionError("invalid resource limit reached the controller")
    with pytest.raises(ValueError, match="max_record_bytes"):
        native_runner.run_recorded(tmp_path, controller, total_output_bytes=2 * METADATA_RESERVE,
                                   input_names=(), max_record_bytes=limit)
    assert not (tmp_path / "NativeResult.json").exists()
    assert not (tmp_path / "NativeHistory.jsonl.gz").exists()


def test_maintained_runner_import_is_isolated_from_scientific_backends():
    code = ("import sys; sys.path.insert(0,sys.argv[1]); import scripts.benchmarks.native_runner; "
            "assert not any(name in sys.modules for name in ('numpy','scipy','flint','sksundae'))")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT)],
                            capture_output=True, text=True, timeout=2)
    assert result.returncode == 0, result.stderr

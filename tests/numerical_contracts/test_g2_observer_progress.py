"""Exact saved-prefix IO and stop handling; no producer or real history replay."""
from copy import deepcopy
import gzip
import hashlib
import json
import sys
from types import ModuleType

import pytest

from scripts.benchmarks import g2_observer_progress as progress
from scripts.benchmarks.native_history import HistoryLimitError, HistoryWriter, _line, decode_integer_values


POLICY = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
          "decimal_digits": 4300, "max_integer_bits": 131072}


def rows():
    return [
        {"kind": "sealed_g2_context", "request_sha256": "a"*64, "evidence": {"case_id": "fixture"}},
        {"kind": "sealed_voltage_lift_charge", "request_sha256": "a"*64,
         "source_interval_sha256": "b"*64,
         "interval": {"segment_id": "fixture", "observation": {"center": [1 << 15000, 7],
            "radius": [1, 1 << 16000], "ball_words": [[[234567, -123], [9, -160]], [[0, 0], [1, -191]]]}}},
        {"kind": "t1_fixed_window_piece", "evidence": {"piece_id": "c"*64, "window_index": 0,
            "segment_id": "fixture", "t_left_hex": "0x0.0p+0", "t_right_hex": "0x1.0p-4"}},
    ]


def writer(tmp_path, **kwargs):
    return progress.PrefixProgress(tmp_path/"new", binding={"fixture": True, "integer_io_policy": POLICY},
        artifact_bytes=8*progress.MIB, max_record_bytes=kwargs.get("max_record_bytes", 65536), integer_io_policy=POLICY)


@pytest.mark.parametrize("kind", ["deadline", "cancelled", "resource_limit"])
def test_operational_stop_closes_only_output_prefix_and_keeps_exact_words(tmp_path, kind):
    sink = writer(tmp_path)
    originals = deepcopy(rows())
    for row in originals:
        sink.append(row)
    stop = progress._stop(progress.saved.ObservationStopped(kind, "fixture_stop"))
    outcome = sink.finish_prefix(stop=stop)
    assert outcome["status"] == "bounded_stop" and outcome["stop"]["classification"] == kind
    assert outcome["records"] == 3 and outcome["output_prefix"]["writer_container_complete"]
    assert not outcome["input_verifier_returned"] and not outcome["resume_eligible"]
    assert not outcome["input_EOF_CRC_coverage_asserted"] and not outcome["T1_pass_asserted"]
    assert not (sink.directory/"RunResult.json").exists() and not (sink.directory/"RecoveryReceipt.json").exists()
    control = tmp_path/"old-writer"
    control.mkdir()
    with HistoryWriter(control/"History.gz", encoding="gzip", total_output_bytes=4*progress.MIB,
                       max_record_bytes=65536, integer_io_policy=POLICY) as original:
        for row in originals:
            original.append(row)
    # Same writer/codec/operation order: even encoded bytes are unchanged.
    assert (sink.directory/"ChargePrefix.jsonl.gz").read_bytes() == (control/"History.gz").read_bytes()
    decoded = [decode_integer_values(json.loads(line), POLICY)
               for line in gzip.decompress((control/"History.gz").read_bytes()).splitlines()]
    assert decoded == originals == rows()
    check = progress.read_progress(sink.directory)
    assert check["read_checks"]["output_prefix_EOF_CRC_verified"]
    assert not check["read_checks"]["input_EOF_CRC_coverage_asserted"]
    assert not list(sink.directory.glob("*.tmp"))


def test_atomic_progress_matches_each_complete_callback(tmp_path):
    sink = writer(tmp_path)
    combined = b""
    for index, row in enumerate(rows()):
        before = deepcopy(row)
        sink.append(row)
        combined += _line(row, POLICY)
        metadata = json.loads((sink.directory/"Progress.json").read_bytes())
        assert metadata["records"] == index+1
        assert metadata["last_record"]["index"] == index
        assert metadata["last_record"]["record_sha256"] == hashlib.sha256(_line(row, POLICY)).hexdigest()
        assert metadata["output_prefix"]["logical_sha256"] == hashlib.sha256(combined).hexdigest()
        assert not metadata["output_prefix"]["writer_container_complete"]
        assert row == before
    sink.finish_prefix(stop=progress._stop(RuntimeError("injected failure")))
    assert json.loads((sink.directory/"PrefixOutcome.json").read_bytes())["stop"]["classification"] == "unclassified_error"


def test_failed_record_write_never_acquires_closed_prefix_or_crc_claim(tmp_path):
    sink = writer(tmp_path, max_record_bytes=65536)
    sink.append(rows()[0])
    with pytest.raises(HistoryLimitError) as captured:
        sink.append({"kind": "sealed_g2_sample", "evidence": {"large": "x"*65536}})
    outcome = sink.finish_prefix(stop=progress._stop(captured.value))
    assert outcome["records"] == 1 and outcome["stop"]["classification"] == "resource_limit"
    assert outcome["output_prefix"]["writer_error"] is not None
    assert not outcome["output_prefix"]["writer_container_complete"]
    assert not progress.read_progress(sink.directory)["read_checks"]["output_prefix_EOF_CRC_verified"]


def test_same_error_causes_remain_distinguishable(tmp_path):
    for error, expected in ((ValueError("unknown"), "unclassified_error"),
                            (gzip.BadGzipFile("container"), "artifact_decode_error"),
                            (HistoryLimitError("cap"), "resource_limit")):
        try:
            try:
                raise error
            except ValueError as cause:
                raise ValueError("reader wrapper") from cause
        except Exception as wrapper:
            evidence = progress._stop(wrapper)
        assert evidence["classification"] == expected
        assert any(item["type"] == type(error).__name__ for item in evidence["chain"])


@pytest.mark.parametrize("damage", ["compressed_tail", "digest", "counts", "last_hash", "claim", "byte_count"])
def test_saved_reader_refuses_corruption_and_overclaim(tmp_path, damage):
    sink = writer(tmp_path)
    sink.append(rows()[0])
    outcome = sink.finish_prefix(stop=progress._stop(progress.saved.ObservationStopped("deadline", "test")))
    path = sink.directory/"PrefixOutcome.json"
    if damage == "compressed_tail":
        raw = (sink.directory/"ChargePrefix.jsonl.gz").read_bytes()[:-3]
        (sink.directory/"ChargePrefix.jsonl.gz").write_bytes(raw)
        outcome["output_prefix"]["encoded_bytes"] = len(raw)
    elif damage == "digest":
        outcome["output_prefix"]["logical_sha256"] = "0"*64
    elif damage == "counts":
        outcome["counts_by_kind"]["sealed_g2_context"] = 2
    elif damage == "last_hash":
        outcome["last_record"]["record_sha256"] = "0"*64
    elif damage == "claim":
        outcome["input_EOF_CRC_coverage_asserted"] = True
    else:
        outcome["output_prefix"]["encoded_bytes"] += 1
    path.write_text(json.dumps(outcome))
    with pytest.raises((ValueError, OSError, EOFError)):
        progress.read_progress(sink.directory)


def test_existing_output_is_never_reopened(tmp_path):
    sink = writer(tmp_path)
    sink.append(rows()[0])
    sink.finish_prefix(stop=progress._stop(RuntimeError("original")))
    before = {p.name: p.read_bytes() for p in sink.directory.iterdir()}
    with pytest.raises(FileExistsError):
        writer(tmp_path)
    assert {p.name: p.read_bytes() for p in sink.directory.iterdir()} == before


def test_unknown_callback_and_first_failure_are_retained(tmp_path):
    sink = writer(tmp_path)
    sink.append(rows()[0])
    with pytest.raises(ValueError, match="unsupported") as error:
        sink.append({"kind": "not-an-observer-record"})
    sink.finish_prefix(stop=progress._stop(error.value))
    failure = (sink.directory/"FirstFailure.json").read_bytes()
    sink.writer.publish_first_failure({"reason": "later"})
    assert (sink.directory/"FirstFailure.json").read_bytes() == failure
    assert progress.read_progress(sink.directory)["progress"]["records"] == 1


def test_reader_uses_no_observer_or_native_solver(tmp_path, monkeypatch):
    sink = writer(tmp_path)
    sink.append(rows()[0])
    sink.finish_prefix(stop=progress._stop(RuntimeError("stopped")))
    monkeypatch.setattr(progress, "run", lambda *args: pytest.fail("reader entered observer"))
    assert progress.main(["--read", str(sink.directory)]) == 0


@pytest.mark.parametrize("tampered", [False, True])
def test_producer_validation_keeps_original_entry_binding(tmp_path, tampered):
    entry = tmp_path/"scripts/benchmarks/g2_route.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("import hashlib\nfrom pathlib import Path\n"
        "def validate(path, *, executable, case_id):\n"
        "    assert executable and case_id == 'fixture'\n"
        "    return {'plan_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}\n")
    pin = hashlib.sha256(entry.read_bytes()).hexdigest()
    plan = tmp_path/"Producer.json"
    plan.write_text(json.dumps({"cases": [{"id": "fixture", "execution": {"cwd": str(tmp_path)}}],
        "source_freeze": {"source_sha256": {str(entry): pin}}}))
    reference = {"path": str(plan), "bytes": plan.stat().st_size, "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
    if tampered:
        entry.write_text("raise AssertionError('changed entry must not be imported')\n")
        with pytest.raises(ValueError, match="producer_metadata_entry_changed"):
            progress.producer_metadata(reference, "fixture")
    else:
        parent, case = progress.producer_metadata(reference, "fixture")
        assert parent["source_freeze"]["source_sha256"][str(entry)] == pin
        assert case["execution"]["cwd"] == str(tmp_path)


@pytest.mark.parametrize("problem", ["missing", "mismatch"])
def test_required_arb_preflight_rejects_missing_or_unpinned_runtime(tmp_path, monkeypatch, problem):
    # The real BallIntegrator constructor is used; its numeric methods are
    # never invoked and no reader/model is constructed by the preflight.
    module = ModuleType("flint")
    module.__version__ = "0.8.0"
    module.__FLINT_VERSION__ = "3.3.1"
    module_path = tmp_path/"flint.py"
    module_path.write_text("# explicit runtime fixture\n")
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, "flint", module if problem != "missing" else None)
    reference = {"path": str(module_path), "bytes": module_path.stat().st_size,
                 "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()}
    lock = {"python_version": sys.version, "flint_versions": {"__version__": "0.8.0", "__FLINT_VERSION__": "3.3.1"},
            "module_files": {"flint": reference}}
    lock_path = tmp_path/"Environment.json"
    lock_path.write_text(json.dumps(lock))
    lock_ref = {"path": str(lock_path), "bytes": lock_path.stat().st_size,
                "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}
    request = {"interval_observation": {"arithmetic": {"bits": 256, "evaluations": 2048, "depth": 16}}}
    if problem == "mismatch":
        module_path.write_text("# changed runtime fixture\n")
    with pytest.raises(progress.saved.ObservationStopped) as error:
        progress.backend_preflight(request, lock_ref)
    assert progress._stop(error.value)["classification"] == "runtime_binding_failure"

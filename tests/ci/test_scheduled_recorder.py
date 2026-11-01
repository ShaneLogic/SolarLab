"""Real pytest processes exercise recorder isolation and fail-closed receipts."""
import hashlib
import json
from pathlib import Path
import random
import runpy
import struct
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from scripts.ci import scheduled_suite as scheduled


JSON_PROBE = """
import json
from pathlib import Path
import pytest

@pytest.fixture(scope="module", autouse=True)
def shared_json():
    def forbidden(*args, **kwargs):
        raise AssertionError("test replaced the shared JSON callable")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(json, "dumps", forbidden)
        patch.setattr(json, "loads", forbidden)
        yield

def test_monkeypatch():
    assert True

def test_canary():
    Path("canary").write_text("ran")
"""


JSON_INTERNAL_ATTRIBUTES = (
    "json.encoder.encode_basestring_ascii", "json.encoder.encode_basestring",
    "json.encoder.c_make_encoder", "json.encoder._make_iterencode", "json.encoder.INFINITY",
    "json.encoder.sorted", "json.encoder.TypeError",
    "json.JSONEncoder.encode", "json.JSONEncoder.iterencode", "json.JSONEncoder.default",
    "json.JSONEncoder.__init__", "json._default_encoder", "json.JSONEncoder",
    "json.encoder.JSONEncoder", "json.dump", "json.dumps", "json.encoder",
)
REQUIRED_JSON_INTERNAL_ATTRIBUTES = (
    "json.encoder.encode_basestring_ascii", "json.encoder.c_make_encoder", "json.JSONEncoder.iterencode",
)


JSON_INTERNALS_PROBE = '''
import json, sys, pytest
from pathlib import Path

def test_monkeypatch(monkeypatch, request):
    targets = []
    for path in ATTRIBUTES:
        parts = path.split(".")
        owner = json
        for part in parts[1:-1]:
            owner = getattr(owner, part)
        targets.append((owner, parts[-1]))
    def forbidden(*args, **kwargs):
        pytest.fail("shared JSON internals were called")
    for owner, name in targets:
        monkeypatch.setattr(owner, name, forbidden, raising=False)
    recorder = next(p for p in request.config.pluginmanager.get_plugins()
                    if type(p).__name__ == "_PytestRecorder")
    # Exercise snapshots while the patch is active as well as the call journal.
    # The parent assertions require an error-free receipt for this operation.
    recorder.snapshot()

def test_canary():
    Path("canary").write_text("ran")
'''.replace("ATTRIBUTES", repr(REQUIRED_JSON_INTERNAL_ATTRIBUTES))
JSON_ALL_INTERNALS_PROBE = JSON_INTERNALS_PROBE.replace(
    repr(REQUIRED_JSON_INTERNAL_ATTRIBUTES), repr(JSON_INTERNAL_ATTRIBUTES),
)


def patch_json_internals(monkeypatch):
    targets = []
    for path in JSON_INTERNAL_ATTRIBUTES:
        parts = path.split(".")
        owner = json
        for part in parts[1:-1]:
            owner = getattr(owner, part)
        targets.append((owner, parts[-1]))

    def forbidden(*args, **kwargs):
        pytest.fail("shared JSON internals were called")

    for owner, name in targets:
        monkeypatch.setattr(owner, name, forbidden, raising=False)


def v3_recorder_file():
    path = Path(__file__).parent / "fixtures" / "scheduled_suite_v3.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7a81306030d131c102b160b8d6a9260e18e25b808ef4e561d44f748c9729cb31"
    )
    return path


def serialization_samples():
    values = [None, True, False, 0, 1, -1, 2**256, -(2**256),
              0.0, -0.0, 0.1, -0.1, 1e-300, 1e300, float("nan"), float("inf"), -float("inf"),
              "", 'quotes " and backslashes \\', "\b\f\n\r\t", "".join(map(chr, range(32))),
              "é Ω 汉字 🌊 \u2028\u2029", "\ud800\udfff", [], {}, (),
              {"z": [None, True, 1.5, {"nested": "Ω"}], "a": (1, 2), "empty": {}},
              {3: "three", 1: "one", 2: "two"},
              {"event": "report", "nodeid": 'test_Ω.py::test_case["x"]', "when": "call",
               "outcome": "failed", "duration": 0.123456789, "wasxfail": None,
               "longrepr": 'failure\nwith quotes " and a backslash \\'},
              {"status": "running", "items": [{"nodeid": "test.py::test_x", "markers": []}],
               "reports": [], "seconds": -0.0}]
    rng = random.Random(41201981031)
    for _ in range(256):
        values.append(struct.unpack("!d", rng.getrandbits(64).to_bytes(8, "big"))[0])
    for _ in range(64):
        values.append({"z": [rng.choice(values[:26]), rng.choice(values[:26])],
                       "a": {"Ω": rng.choice(values[:26]), "b": [rng.randrange(-10000, 10000)]}})
    return values


def test_private_serialization_matches_v3_bytes_under_all_json_patches(monkeypatch):
    encode_v3 = runpy.run_path(str(v3_recorder_file()))["_RECORDER_DUMPS"]
    cases = [(value, {"indent": indent, "sort_keys": sort})
             for value in serialization_samples() for indent in (None, 0, 2, "\t") for sort in (False, True)]
    expected = [encode_v3(value, **options) for value, options in cases]
    patch_json_internals(monkeypatch)
    assert [scheduled._RECORDER_DUMPS(value, **options) for value, options in cases] == expected


def test_private_serialization_preserves_invalid_input_failures(monkeypatch):
    patch_json_internals(monkeypatch)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="Circular reference detected"):
        scheduled._RECORDER_DUMPS(cycle)
    with pytest.raises(TypeError, match="not JSON serializable"):
        scheduled._RECORDER_DUMPS(object())
    with pytest.raises(TypeError, match="keys must be"):
        scheduled._RECORDER_DUMPS({object(): "unsupported key"})


def run_recorder(tmp_path, source, *, conftest="", extra=(), runner=None):
    (tmp_path / "test_probe.py").write_text(textwrap.dedent(source))
    if conftest:
        (tmp_path / "conftest.py").write_text(textwrap.dedent(conftest))
    output = tmp_path / "actual.json"
    command = [sys.executable, str(runner or Path(scheduled.__file__)), "_pytest", "--out", str(output),
               "--", "test_probe.py", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(tmp_path / "junit.xml"), *extra]
    env = scheduled.pinned_environment()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    data = json.loads(output.read_text())
    journal = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    errors = [json.loads(line) for line in output.with_suffix(".recorder-errors.jsonl").read_text().splitlines()]
    xml = ET.parse(tmp_path / "junit.xml")
    return result, data, journal, errors, xml


def assert_complete_two_tests(result, data, journal, errors, xml):
    assert result.returncode == 0, result.stdout + result.stderr
    assert data["status"] == "completed" and data["execution_complete"]
    assert data["exit_code"] == data["pytest_exit_code"] == 0
    assert data["recorder_errors"] == errors == []
    assert data["not_started_nodeids"] == data["incomplete_nodeids"] == []
    assert data["started_nodeids"] == ["test_probe.py::test_canary", "test_probe.py::test_monkeypatch"]
    phases = [(r["nodeid"], r["when"], r["outcome"]) for r in data["reports"]]
    assert phases == [("test_probe.py::" + name, phase, "passed")
                      for name in ("test_monkeypatch", "test_canary")
                      for phase in ("setup", "call", "teardown")]
    assert [(r["nodeid"], r["when"], r["outcome"]) for r in journal if r["event"] == "report"] == phases
    assert len(xml.findall(".//testcase")) == 2
    assert not xml.findall(".//failure") and not xml.findall(".//error") and not xml.findall(".//skipped")


def test_real_recorder_survives_shared_json_monkeypatch_and_runs_canary(tmp_path):
    observed = run_recorder(tmp_path, JSON_PROBE)
    assert_complete_two_tests(*observed)
    assert (tmp_path / "canary").read_text() == "ran"


@pytest.mark.parametrize("probe", [JSON_INTERNALS_PROBE, JSON_ALL_INTERNALS_PROBE], ids=["required", "all"])
def test_real_recorder_survives_json_internals_and_keeps_v3_bytes(tmp_path, probe):
    observed = run_recorder(tmp_path, probe)
    assert_complete_two_tests(*observed)
    assert (tmp_path / "canary").read_text() == "ran"
    encode_v3 = runpy.run_path(str(v3_recorder_file()))["_RECORDER_DUMPS"]
    data, journal = observed[1:3]
    assert (tmp_path / "actual.jsonl").read_text() == "".join(encode_v3(row) + "\n" for row in journal)
    assert (tmp_path / "actual.json").read_text() == encode_v3(data, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("probe", [JSON_INTERNALS_PROBE, JSON_ALL_INTERNALS_PROBE], ids=["required", "all"])
def test_v3_negative_control_records_json_internals_failure(tmp_path, probe):
    result, data, journal, errors, xml = run_recorder(tmp_path, probe, runner=v3_recorder_file())
    assert result.returncode == 1 and data["pytest_exit_code"] == 0
    assert data["status"] == "recorder_error" and data["execution_complete"]
    assert len(data["reports"]) == 6 and len(xml.findall(".//testcase")) == 2
    assert data["recorder_errors"] == errors and errors
    assert any(row["event"] == "recorder_error" for row in journal)
    assert (tmp_path / "canary").read_text() == "ran"
    assert "SCHEDULED RECORDER" in result.stderr and "INTERNALERROR" not in result.stdout


def test_8a0bf115_recorder_file_negative_control_aborts_before_canary(tmp_path):
    # This is the complete git-show 8a0bf115:scripts/ci/scheduled_suite.py file.
    # Vendor exact bytes so a shallow checkout still runs the historical plugin.
    runner = Path(__file__).parent / "fixtures" / "scheduled_suite_8a0bf115.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
        "9ac0502f157bf0c5d7ce0dea8286db21af423cf8321e40ecd8a3f47136fc1045"
    )
    (tmp_path / "test_probe.py").write_text(textwrap.dedent(JSON_PROBE))
    env = scheduled.pinned_environment()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run([
        sys.executable, str(runner), "_pytest", "--out", str(tmp_path / "legacy.json"),
        "--", "test_probe.py", "-q", "-p", "no:cacheprovider", "--junitxml=legacy.xml",
    ], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "test replaced the shared JSON callable" in result.stdout + result.stderr
    assert "INTERNALERROR" in result.stdout and not (tmp_path / "canary").exists()
    assert ET.parse(tmp_path / "legacy.xml").findall(".//error")
    data = json.loads((tmp_path / "legacy.json").read_text())
    assert len(data["reports"]) < 6


def test_module_scoped_patches_leave_clock_io_and_decoder_bound(tmp_path):
    source = '''
        import builtins, json, os, sys, time
        from pathlib import Path
        import pytest
        @pytest.fixture(scope="module", autouse=True)
        def shared_stdlib():
            def forbidden(*args, **kwargs):
                pytest.fail("recorder used a test-patched stdlib callable")
            with pytest.MonkeyPatch.context() as patch:
                for obj, name in [(json, "dumps"), (json, "loads"), (time, "monotonic"),
                                  (os, "getcwd"), (os, "replace"), (os, "fsync"), (builtins, "open")]:
                    patch.setattr(obj, name, forbidden)
                yield
        def test_monkeypatch(request):
            recorder = next(p for p in request.config.pluginmanager.get_plugins()
                            if type(p).__name__ == "_PytestRecorder")
            module = sys.modules[type(recorder).__module__]
            assert recorder.snapshot()
            assert module.read(recorder.output)["status"] == "running"
        def test_canary():
            Path("canary").write_text("ran")
    '''
    observed = run_recorder(tmp_path, source)
    assert_complete_two_tests(*observed)
    assert (tmp_path / "canary").read_text() == "ran"


def test_session_entry_and_finish_keep_their_original_clock_and_cwd(tmp_path, monkeypatch):
    import builtins
    import os
    import time

    def collection_only(arguments, plugins):
        # Isolate pytest's own use of cwd/open from the runner's entry/finalization.
        plugins[0].pytest_collection_finish(SimpleNamespace(
            config=SimpleNamespace(option=SimpleNamespace(collectonly=True)), items=[],
        ))
        return 0

    def forbidden(*args, **kwargs):
        pytest.fail("session used a test-patched stdlib callable")

    output = tmp_path / "actual.json"
    with monkeypatch.context() as patch:
        patch.setattr(pytest, "main", collection_only)
        patch.setattr(sys, "path", sys.path.copy())
        for obj, name in [(json, "dumps"), (json, "loads"), (time, "monotonic"),
                          (os, "getcwd"), (os, "replace"), (os, "fsync"), (builtins, "open")]:
            patch.setattr(obj, name, forbidden)
        code = scheduled.pytest_session([], output)
    assert code == 0
    data = json.loads(output.read_text())
    assert code == 0 and data["status"] == "collected"
    assert data["recorder_errors"] == [] and data["seconds"] >= 0
    assert output.with_suffix(".jsonl").exists()
    assert output.with_suffix(".recorder-errors.jsonl").read_text() == ""


def test_diagnostic_keeps_original_stderr_when_os_write_is_patched(monkeypatch, capfd):
    import os

    def forbidden(*args, **kwargs):
        pytest.fail("shared os.write was called")

    monkeypatch.setattr(os, "write", forbidden)
    scheduled._recorder_diagnostic({"event": "recorder_error", "stage": "stderr-probe"})
    assert "stderr-probe" in capfd.readouterr().err


@pytest.mark.parametrize("stage", ["session_clock", "error_sink_flush"])
def test_finalization_baseexceptions_name_the_failed_operation(tmp_path, monkeypatch, stage):
    def broken(*args, **kwargs):
        pytest.fail("finalization fault")

    output = tmp_path / "actual.json"
    with output.with_suffix(".jsonl").open("w") as journal, \
            output.with_suffix(".recorder-errors.jsonl").open("w") as errors:
        recorder = scheduled._PytestRecorder(output, None, journal, errors, tmp_path)
        recorder.pytest_collection_finish(SimpleNamespace(
            config=SimpleNamespace(option=SimpleNamespace(collectonly=True)), items=[],
        ))
        if stage == "session_clock":
            monkeypatch.setattr(scheduled, "_RECORDER_MONOTONIC", broken)
        else:
            monkeypatch.setattr(recorder, "error_flush", broken)
        assert recorder.finish(0) == 1
    data = json.loads(output.read_text())
    assert data["status"] == "recorder_error" and data["pytest_exit_code"] == 0
    assert [error["stage"] for error in data["recorder_errors"]] == [stage]


def test_unexpected_pytest_baseexception_is_finalized(tmp_path, monkeypatch):
    def interrupted(arguments, plugins):
        plugins[0].pytest_collection_finish(SimpleNamespace(
            config=SimpleNamespace(option=SimpleNamespace(collectonly=False)), items=[],
        ))
        pytest.fail("unexpected pytest failure")

    monkeypatch.setattr(pytest, "main", interrupted)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    output = tmp_path / "actual.json"
    assert scheduled.pytest_session([], output) == 1
    data = json.loads(output.read_text())
    assert data["status"] == "recorder_error" and data["exit_code"] == 1
    assert [row["stage"] for row in data["recorder_errors"]] == ["pytest_session"]


def test_file_setup_baseexception_still_emits_an_explicit_failure(tmp_path, monkeypatch, capfd):
    class Unprintable(BaseException):
        def __str__(self):
            pytest.fail("unprintable exception")

    def broken(*args, **kwargs):
        raise Unprintable()

    monkeypatch.setattr(scheduled, "_RECORDER_OPEN", broken)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    assert scheduled.pytest_session([], tmp_path / "actual.json") == 1
    diagnostic = capfd.readouterr().err
    assert "recorder_files" in diagnostic and "Recorder exception could not be formatted" in diagnostic


def test_error_serializer_keeps_bound_string_quoting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("shared string quoting was called")

    monkeypatch.setattr(json.encoder, "encode_basestring_ascii", forbidden)
    event = {"event": "recorder_error", "stage": "quoting", "nodeid": None,
             "message": 'quoted "text"\nUnicode: Ω\tbackslash: \\', "errors": 1}
    assert json.loads(scheduled._recorder_error_line(event)) == event


def test_recorder_error_protects_unprintable_exception_messages(tmp_path):
    class Unprintable(BaseException):
        def __str__(self):
            pytest.fail("unprintable exception")

    output = tmp_path / "actual.json"
    with output.with_suffix(".jsonl").open("w") as journal, \
            output.with_suffix(".recorder-errors.jsonl").open("w") as errors:
        recorder = scheduled._PytestRecorder(output, None, journal, errors, tmp_path)
        recorder.error("probe", Unprintable(), "test_probe.py::test_probe")
        assert recorder.finish(0) == 1
    data = json.loads(output.read_text())
    recorded = [json.loads(line) for line in output.with_suffix(".recorder-errors.jsonl").read_text().splitlines()]
    assert recorded == data["recorder_errors"] and len(recorded) == 1
    assert recorded[0]["error_type"] == "Unprintable" and recorded[0]["stage"] == "probe"
    assert recorded[0]["message"] == "Recorder exception could not be formatted"
    assert recorded[0]["nodeid"] == "test_probe.py::test_probe"


@pytest.mark.parametrize("method", ["encode", "iterencode"])
@pytest.mark.parametrize("exception", ["RuntimeError", "pytest.fail"])
def test_patched_encoder_methods_fail_receipts_without_aborting_tests(tmp_path, method, exception):
    # Keep the scheduled node identities; V4 strengthens this to clean receipts.
    source = '''
        import json, pytest
        from pathlib import Path
        @pytest.fixture(scope="module", autouse=True)
        def broken_encoder():
            def broken(*args, **kwargs):
                RAISE
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(json.JSONEncoder, METHOD, broken)
                yield
        def test_monkeypatch():
            assert True
        def test_canary():
            Path("canary").write_text("ran")
    '''.replace("METHOD", repr(method)).replace(
        "RAISE", 'raise RuntimeError("encoder fault")' if exception == "RuntimeError" else
        'pytest.fail("encoder fault")',
    )
    observed = run_recorder(tmp_path, source)
    assert_complete_two_tests(*observed)
    assert (tmp_path / "canary").read_text() == "ran"


@pytest.mark.parametrize("fault", ["sinks", "quoting"])
def test_error_reporting_baseexceptions_do_not_abort_the_canary(tmp_path, fault):
    source = '''
        import pytest, sys
        from pathlib import Path
        def test_monkeypatch(monkeypatch, request):
            recorder = next(p for p in request.config.pluginmanager.get_plugins()
                            if type(p).__name__ == "_PytestRecorder")
            module = sys.modules[type(recorder).__module__]
            def broken(*args, **kwargs):
                pytest.fail("error-path fault")
            if FAULT == "sinks":
                monkeypatch.setattr(recorder, "journal_write", broken)
                monkeypatch.setattr(recorder, "error_write", broken)
                monkeypatch.setattr(module, "_RECORDER_STDERR", broken)
            else:
                monkeypatch.setattr(module, "_RECORDER_DUMPS", broken)
                monkeypatch.setattr(module, "_RECORDER_ERROR_QUOTE", broken)
        def test_canary():
            Path("canary").write_text("ran")
    '''.replace("FAULT", repr(fault))
    result, data, journal, errors, xml = run_recorder(tmp_path, source)
    assert result.returncode == 1, result.stdout + result.stderr
    assert data["status"] == "recorder_error" and data["pytest_exit_code"] == 0
    assert data["recorder_errors"] and data["execution_complete"]
    assert len(data["reports"]) == 6 and (tmp_path / "canary").read_text() == "ran"
    assert len(xml.findall(".//testcase")) == 2 and not xml.findall(".//error")
    assert "SCHEDULED RECORDER" in result.stderr and "INTERNALERROR" not in result.stdout
    if fault == "quoting":
        assert errors and all(row == {"event": "recorder_error", "stage": "error_record_encoding"}
                              for row in errors)
    else:
        assert errors == []  # Both sinks failed, but the result and stderr still fail explicitly.


@pytest.mark.parametrize("hook", ["pytest_collectreport", "pytest_collection_finish",
                                  "pytest_runtest_logstart", "pytest_runtest_logreport", "finish"])
def test_hook_body_baseexceptions_become_recorder_errors(tmp_path, hook):
    class Broken:
        def __getattr__(self, name):
            pytest.fail("hook property fault")

        def __getitem__(self, name):
            pytest.fail("hook item fault")

    output = tmp_path / "actual.json"
    with output.with_suffix(".jsonl").open("w") as journal, \
            output.with_suffix(".recorder-errors.jsonl").open("w") as errors:
        recorder = scheduled._PytestRecorder(output, None, journal, errors, tmp_path)
        if hook == "pytest_runtest_logstart":
            recorder.started_nodes = Broken()
            recorder.pytest_runtest_logstart("test_probe.py::test_probe", None)
            recorder.started_nodes = set()
        elif hook == "finish":
            recorder.data["items"] = [Broken()]
            assert recorder.finish(0) == 1
            recorder.data["items"] = []
        elif hook == "pytest_collection_finish":
            with pytest.raises(pytest.UsageError, match="scheduled collection could not be verified"):
                recorder.pytest_collection_finish(Broken())
        else:
            getattr(recorder, hook)(Broken())
        assert recorder.data["recorder_errors"]
        assert recorder.finish(0) == 1
    lines = [json.loads(line) for line in output.with_suffix(".recorder-errors.jsonl").read_text().splitlines()]
    assert lines and all(row["event"] == "recorder_error" for row in lines)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_control_exceptions_propagate_from_hooks_and_error_paths(tmp_path, monkeypatch, exception):
    def interrupted(*args, **kwargs):
        raise exception()

    output = tmp_path / "actual.json"
    with output.with_suffix(".jsonl").open("w") as journal, \
            output.with_suffix(".recorder-errors.jsonl").open("w") as errors:
        recorder = scheduled._PytestRecorder(output, None, journal, errors, tmp_path)
        monkeypatch.setattr(scheduled, "_RECORDER_DUMPS", interrupted)
        with pytest.raises(exception):
            recorder.pytest_runtest_logreport(SimpleNamespace(
                nodeid="test_probe.py::test_probe", when="call", outcome="passed", duration=0, failed=False,
            ))
        with pytest.raises(exception):
            recorder.snapshot()
        monkeypatch.setattr(scheduled, "_RECORDER_MONOTONIC", interrupted)
        with pytest.raises(exception):
            recorder.pytest_runtest_logstart("test_probe.py::test_probe", None)
        monkeypatch.setattr(scheduled, "_RECORDER_STDERR", interrupted)
        with pytest.raises(exception):
            scheduled._recorder_diagnostic({"event": "recorder_error"})
        monkeypatch.setattr(scheduled, "_RECORDER_ERROR_QUOTE", interrupted)
        with pytest.raises(exception):
            recorder.error("interrupted", ValueError("failure"))


@pytest.mark.parametrize("sink", ["journal", "snapshot", "encoder"])
@pytest.mark.parametrize("exception", ["OSError", "pytest.fail"])
def test_recorder_io_failure_keeps_tests_running_but_fails_receipt(tmp_path, sink, exception):
    source = '''
        import json, os, sys, pytest
        from pathlib import Path
        def test_monkeypatch(monkeypatch, request):
            recorder = next(p for p in request.config.pluginmanager.get_plugins()
                            if type(p).__name__ == "_PytestRecorder")
            module = sys.modules[type(recorder).__module__]
            def broken(*args, **kwargs):
                RAISE
            def forbidden(*args, **kwargs):
                raise AssertionError("error reporting used a shared callable")
            for obj, name in [(json, "dumps"), (json, "loads"), (os, "write")]:
                monkeypatch.setattr(obj, name, forbidden)
            monkeypatch.setattr(json.JSONEncoder, "iterencode", forbidden)
            if SINK == "journal":
                monkeypatch.setattr(recorder, "journal_write", broken)
            elif SINK == "encoder":
                monkeypatch.setattr(module, "_RECORDER_DUMPS", broken)
            else:
                monkeypatch.setattr(module, "_RECORDER_REPLACE", broken)
                assert not recorder.snapshot()
        def test_canary():
            Path("canary").write_text("ran")
    '''.replace("SINK", repr(sink)).replace(
        "RAISE", 'raise OSError("injected recorder I/O failure")' if exception == "OSError" else
        'pytest.fail("injected recorder I/O failure")',
    )
    result, data, journal, errors, xml = run_recorder(tmp_path, source)
    assert result.returncode == 1, result.stdout + result.stderr
    assert data["pytest_exit_code"] == 0 and data["exit_code"] == 1
    assert data["status"] == "recorder_error" and data["execution_complete"]
    assert len(data["reports"]) == 6 and (tmp_path / "canary").read_text() == "ran"
    assert errors and errors == data["recorder_errors"]
    assert all(r["event"] == "recorder_error" and "injected recorder I/O failure" in r["message"] for r in errors)
    assert {row["stage"] for row in errors} == ({"snapshot"} if sink == "snapshot" else {"journal"})
    assert "SCHEDULED RECORDER" in result.stderr and "INTERNALERROR" not in result.stdout
    if sink != "journal":
        assert any(r["event"] == "recorder_error" for r in journal)
    assert len(xml.findall(".//testcase")) == 2 and not xml.findall(".//error")


def test_final_snapshot_failure_cannot_leave_a_passing_receipt(tmp_path):
    conftest = '''
        import sys
        def pytest_sessionfinish(session, exitstatus):
            recorder = next(p for p in session.config.pluginmanager.get_plugins()
                            if type(p).__name__ == "_PytestRecorder")
            module = sys.modules[type(recorder).__module__]
            replace = module._RECORDER_REPLACE
            def fail_once(*args, **kwargs):
                module._RECORDER_REPLACE = replace
                raise OSError("injected final snapshot failure")
            module._RECORDER_REPLACE = fail_once
    '''
    result, data, journal, errors, xml = run_recorder(tmp_path, JSON_PROBE, conftest=conftest)
    assert result.returncode == 1, result.stdout + result.stderr
    assert data["pytest_exit_code"] == 0 and data["exit_code"] == 1
    assert data["status"] == "recorder_error" and data["execution_complete"]
    assert len(data["reports"]) == 6 and len(xml.findall(".//testcase")) == 2
    assert errors == data["recorder_errors"] and len(errors) == 1
    assert errors[0]["stage"] == "snapshot"
    assert "injected final snapshot failure" in result.stderr
    assert any(r["event"] == "recorder_error" for r in journal)


def test_incomplete_zero_exit_is_failed_and_names_unexecuted_cases(tmp_path):
    source = '''
        import pytest
        def test_first():
            pytest.exit("explicit early exit", returncode=0)
        def test_canary():
            assert False, "must not run"
    '''
    result, data, journal, errors, xml = run_recorder(tmp_path, source)
    assert result.returncode == 1, result.stdout + result.stderr
    assert data["pytest_exit_code"] == 0 and data["exit_code"] == 1
    assert data["status"] == "incomplete" and not data["execution_complete"]
    assert data["not_started_nodeids"] == ["test_probe.py::test_canary"]
    assert data["incomplete_nodeids"] == ["test_probe.py::test_first"]
    assert "incomplete" in result.stderr


def test_unverifiable_external_collection_refuses_before_test_body(tmp_path):
    cwd, external = tmp_path / "cwd", tmp_path / "external"
    cwd.mkdir()
    external.mkdir()
    sentinel = tmp_path / "body-ran"
    probe = external / "test_probe.py"
    probe.write_text("from pathlib import Path\ndef test_probe():\n"
                     f"    Path({str(sentinel)!r}).write_text('ran')\n")
    expected, output = tmp_path / "expected.json", tmp_path / "actual.json"
    expected.write_text(json.dumps(["test_probe.py::test_probe"]))
    env = scheduled.pinned_environment()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = [sys.executable, str(Path(scheduled.__file__)), "_pytest", "--out", str(output),
               "--expected", str(expected), "--", str(probe), "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(tmp_path / "junit.xml")]
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 4, result.stdout + result.stderr
    assert not sentinel.exists()
    assert "scheduled collection could not be verified" in result.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "incomplete" and data["pytest_exit_code"] == data["exit_code"] == 4
    assert not data["collection_complete"] and not data["execution_complete"]
    assert data["reports"] == data["started_nodeids"] == []
    assert [row["stage"] for row in data["recorder_errors"]] == ["collection"]
    errors = [json.loads(line) for line in output.with_suffix(".recorder-errors.jsonl").read_text().splitlines()]
    assert errors == data["recorder_errors"]
    journal = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert all(row["event"] == "recorder_error" for row in journal)
    assert not ET.parse(tmp_path / "junit.xml").findall(".//testcase")


def test_started_node_without_any_report_is_incomplete_not_unstarted(tmp_path):
    conftest = '''
        import pytest
        @pytest.hookimpl(trylast=True)
        def pytest_runtest_logstart(nodeid, location):
            pytest.exit("exit after recorder start but before setup", returncode=0)
    '''
    result, data, journal, errors, xml = run_recorder(tmp_path, JSON_PROBE, conftest=conftest)
    assert result.returncode == 1, result.stdout + result.stderr
    assert data["status"] == "incomplete" and data["pytest_exit_code"] == 0
    assert data["reports"] == [] and not data["execution_complete"]
    assert data["started_nodeids"] == ["test_probe.py::test_monkeypatch"]
    assert data["incomplete_nodeids"] == ["test_probe.py::test_monkeypatch"]
    assert data["not_started_nodeids"] == ["test_probe.py::test_canary"]
    assert [row["nodeid"] for row in journal if row["event"] == "start"] == data["started_nodeids"]


def test_internal_error_is_incomplete_and_preserves_pytest_exit_code(tmp_path):
    conftest = '''
        def pytest_runtest_logreport(report):
            if report.when == "call":
                raise RuntimeError("external plugin aborted test execution")
    '''
    result, data, journal, errors, xml = run_recorder(tmp_path, JSON_PROBE, conftest=conftest)
    assert result.returncode == data["pytest_exit_code"] == data["exit_code"] == 3
    assert data["status"] == "incomplete" and not data["execution_complete"]
    assert data["not_started_nodeids"] == ["test_probe.py::test_canary"]
    assert data["incomplete_nodeids"] == ["test_probe.py::test_monkeypatch"]
    assert "INTERNALERROR" in result.stdout and "incomplete" in result.stderr
    assert not (tmp_path / "canary").exists()


def test_collect_only_has_explicit_collected_status(tmp_path):
    result, data, journal, errors, xml = run_recorder(tmp_path, JSON_PROBE, extra=("--collect-only",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert data["status"] == "collected" and data["collection_complete"]
    assert not data["execution_complete"] and data["reports"] == []
    assert data["recorder_errors"] == errors == []
    assert not (tmp_path / "canary").exists()


def test_setup_error_and_setup_skip_are_complete_terminal_outcomes(tmp_path):
    source = '''
        import pytest
        @pytest.fixture
        def broken():
            raise ValueError("fixture failure")
        def test_error(broken):
            assert False
        @pytest.mark.skip(reason="explicit test fixture skip")
        def test_skip():
            assert False
    '''
    result, data, journal, errors, xml = run_recorder(tmp_path, source)
    assert result.returncode == 1
    assert data["status"] == "completed" and data["execution_complete"]
    assert data["not_started_nodeids"] == data["incomplete_nodeids"] == []
    assert not errors and not data["recorder_errors"]
    assert len(xml.findall(".//error")) == len(xml.findall(".//skipped")) == 1


def run_coverage_shard(tmp_path, monkeypatch, source, expected):
    (tmp_path / "test_probe.py").write_text(textwrap.dedent(source))
    (tmp_path / "uv.lock").write_text("frozen synthetic lock\n")
    monkeypatch.setattr(scheduled, "ROOT", tmp_path)
    monkeypatch.setattr(scheduled, "SUITES", {"root": tmp_path})
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    monkeypatch.setattr(scheduled, "verify_installation", lambda: {})
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    shard = {"id": "probe", "suite": "root", "paths": ["test_probe.py"],
             "markers": "not slow", "ignored": [], "nodeids": expected}
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "lock_sha256": scheduled.sha256(tmp_path / "uv.lock"),
                           "selected_shards": {"coverage": [shard]}})
    out = tmp_path / "out"
    code = scheduled.run_shard(SimpleNamespace(plan=plan, kind="coverage", shard="probe", out=out))
    return code, scheduled.read(out / "result.json"), out


def test_coverage_hard_zero_exit_fails_and_names_missing_nodes(tmp_path, monkeypatch):
    source = '''
        import os
        def test_first():
            assert True
        def test_exit():
            os._exit(0)
        def test_canary():
            assert False, "must not run"
    '''
    expected = ["test_probe.py::" + name for name in ("test_first", "test_exit", "test_canary")]
    code, result, out = run_coverage_shard(tmp_path, monkeypatch, source, expected)
    assert code == result["exit_code"] == 1 and result["status"] == "failed"
    assert result["pytest_process_exit_code"] == 0 and not result["receipt_complete"]
    assert result["not_started_nodeids"] == [expected[2]]
    assert result["incomplete_nodeids"] == [expected[1]]
    assert scheduled.read(out / "tests.json")["status"] == "running"
    log = (out / "pytest.log").read_text()
    assert "SCHEDULED INCOMPLETE RECEIPT" in log and expected[1] in log and expected[2] in log


def test_coverage_complete_receipt_accepts_worker_local_parameter_ids(tmp_path, monkeypatch):
    source = '''
        import pytest
        @pytest.mark.parametrize("value", [1, 2], ids=["local-one", "local-two"])
        def test_probe(value):
            assert value > 0
    '''
    expected = ["test_probe.py::test_probe[plan-one]", "test_probe.py::test_probe[plan-two]"]
    code, result, out = run_coverage_shard(tmp_path, monkeypatch, source, expected)
    assert code == 0 and result["status"] == "passed" and result["receipt_complete"]
    assert result["receipt_errors"] == result["not_started_nodeids"] == result["incomplete_nodeids"] == []
    data = scheduled.read(out / "tests.json")
    assert [row["nodeid"] for row in data["items"]] == [
        "test_probe.py::test_probe[local-one]", "test_probe.py::test_probe[local-two]",
    ]
    assert len(data["reports"]) == 6 and len(ET.parse(out / "junit.xml").findall(".//testcase")) == 2


@pytest.mark.parametrize("corruption", ["no-final-status", "collection-flag", "execution-flag",
                                       "missing-phase", "extra-node", "recorder-error", "torn-journal",
                                       "function-count", "exit-code"])
def test_coverage_receipt_rejects_false_completeness(tmp_path, corruption):
    nodeid = "test_probe.py::test_probe"
    reports = [{"nodeid": nodeid, "when": phase, "outcome": "passed"}
               for phase in ("setup", "call", "teardown")]
    data = {"items": [{"nodeid": nodeid}], "reports": reports, "status": "completed",
            "exit_code": 0, "collection_complete": True, "execution_complete": True,
            "recorder_errors": [], "started_nodeids": [nodeid]}
    if corruption == "no-final-status":
        data["status"] = "running"
    elif corruption == "collection-flag":
        data["collection_complete"] = False
    elif corruption == "execution-flag":
        data["execution_complete"] = False
    elif corruption == "missing-phase":
        reports.pop()
    elif corruption == "extra-node":
        reports.append({"nodeid": "test_probe.py::unplanned", "when": "call", "outcome": "passed"})
    elif corruption == "recorder-error":
        data["recorder_errors"] = [{"event": "recorder_error", "stage": "journal"}]
    elif corruption == "exit-code":
        data["exit_code"] = 1
    scheduled.write(tmp_path / "tests.json", data)
    journal = "".join(json.dumps({"event": "report", **row}) + "\n" for row in reports)
    (tmp_path / "tests.jsonl").write_text(journal + ('{"interrupted":' if corruption == "torn-journal" else ""))
    expected = [nodeid, "test_probe.py::test_missing"] if corruption == "function-count" else [nodeid]
    result = scheduled._coverage_receipt(tmp_path, expected, 0)
    assert not result["receipt_complete"] and result["receipt_errors"]
    if corruption == "missing-phase":
        assert result["incomplete_nodeids"] == [nodeid]


def test_slow_child_recorder_error_has_its_own_failure_label(tmp_path, monkeypatch):
    (tmp_path / "test_probe.py").write_text(textwrap.dedent('''
        import pytest, sys
        def test_probe(monkeypatch, request):
            recorder = next(p for p in request.config.pluginmanager.get_plugins()
                            if type(p).__name__ == "_PytestRecorder")
            def broken(*args, **kwargs):
                pytest.fail("slow recorder fault")
            monkeypatch.setattr(sys.modules[type(recorder).__module__], "_RECORDER_DUMPS", broken)
    '''))
    out = tmp_path / "out"
    out.mkdir()
    nodeid = "test_probe.py::test_probe"
    scheduled.write(out / "expected.json", [nodeid])
    monkeypatch.setattr(scheduled, "SUITES", {"legacy": tmp_path})
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    shard = {"id": "probe", "suite": "legacy", "paths": ["test_probe.py"],
             "markers": "not slow", "ignored": [], "node_wall_seconds": 10}
    assert scheduled.run_slow_nodes(shard, out, scheduled.pinned_environment(), {}) == 1
    child = scheduled.read(out / "nodes/0000/tests.json")
    assert child["status"] == "recorder_error" and child["execution_complete"]
    result = scheduled.read(out / "tests.json")
    assert result["node_outcomes"][0]["outcome"] == "recorder_error"
    failure = ET.parse(out / "junit.xml").find(".//failure")
    assert failure is not None and failure.get("type") == "recorder_error"
    assert failure.get("message") == "Scheduled node recorder error"
    assert ET.parse(out / "nodes/0000/junit.xml").findall(".//testcase")

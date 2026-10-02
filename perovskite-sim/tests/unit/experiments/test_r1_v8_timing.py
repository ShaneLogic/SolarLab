"""Exclusive timing must not double count children or change trajectory results."""
import pytest
import ast
from scripts import r1_v8_timing as timing


def test_original_persistence_phases_remain_separate():
    source = '\n'.join('write(directory / "' + name + '", {})' for name in
        ("PreparedV1.json", "ResultV1.json", "PhysicsReplayV1.json"))
    tree = timing.CoarseInstrumentation().visit(ast.parse(source))
    assert [node.value.args[0].value for node in tree.body] == [
        "preparation_write", "result_write", "replay_write"]


def test_nested_time_is_exclusive(monkeypatch):
    clock = iter([0., 2., 5., 10.])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(clock))
    timer = timing.ExclusiveTimer()
    with timer.phase("integrate"):
        with timer.phase("observer"):
            pass
    assert timer.entries["integrate"]["exclusive_s"] == 7.
    assert timer.entries["observer"]["exclusive_s"] == 3.
    assert timer.report(10.)["exclusive_sum_s"] == 10.


def test_exception_closes_scopes_and_preserves_failure(monkeypatch):
    clock = iter([1., 4.])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(clock))
    timer = timing.ExclusiveTimer()
    with pytest.raises(ValueError, match="numerical failure"):
        timer.call("prepare", lambda: (_ for _ in ()).throw(ValueError("numerical failure")))
    assert timer.stack == []
    assert timer.entries["prepare"]["exclusive_s"] == 3.
    with pytest.raises(ValueError, match="boundary"):
        timer.report(2.)


def representative(api):
    prepared = api["prepare"]()
    rows = []
    def observe(row):
        rows.append(api["json_data"](row))
    result = api["run"](prepared, observe)
    return {"result": result, "rows": rows, "replay": api["replay"](prepared, result, False)}


def test_instrumentation_preserves_work_results_and_observer():
    calls = []
    def run(prepared, observer):
        calls.append("run")
        observer({"state": prepared + 1})
        return prepared * 2
    api = {"prepare": lambda: 3, "run": run, "json_data": dict,
           "replay": lambda prepared, result, incomplete: (prepared, result, incomplete)}
    expected = representative(api)
    timer = timing.ExclusiveTimer()
    actual = timing.instrument(representative, timer)(api)
    assert actual == expected
    assert calls == ["run", "run"]
    assert timer.entries["integrate_compute_and_checks"]["calls"] == 1
    assert timer.entries["observer_io_and_bookkeeping"]["calls"] == 1
    assert not timer.stack

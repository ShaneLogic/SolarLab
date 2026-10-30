"""Real process, store and SSE contracts for unavailable resource sampling."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import pytest

import control_workers
import test_event_service as events
import test_execution as fixtures
import solarlab.experiments.execution as execution
import solarlab.experiments.worker as worker_module
from solarlab.experiments.worker import WorkerBinding
from solarlab.io.run_store import RunStore, StateConflict


# These are observation budgets for real spawn/fsync/reap work on busy runners,
# not changes to the supervisor's RSS, sampling or escalation limits.
WAIT_SECONDS = 30.0


@contextmanager
def runtime(tmp_path, *, fatal=False):
    binding = WorkerBinding.from_function("barrier", control_workers.barrier,
        registry_sha256=fixtures.REGISTRY.sha256, schema_sha256=fixtures.SCHEMA.sha256)
    with RunStore(tmp_path / "store") as store:
        pool = execution.ProcessSupervisor(store, tmp_path / "work", (binding,), max_active=1, max_pending=3,
            shared_rss_bytes=768 * 1024**2, shared_output_bytes=16 * 1024**2,
            poll_seconds=0.01)
        try:
            yield store, pool
        except BaseException as error:
            # A setup/assertion failure may precede the intended fault. Keep its
            # evidence instead of replacing it with a false DID NOT RAISE.
            try:
                pool.close(timeout=WAIT_SECONDS)
            except BaseException as cleanup:
                error.add_note(f"supervisor cleanup: {type(cleanup).__name__}: {cleanup}")
            raise
        else:
            if fatal:
                with pytest.raises(RuntimeError, match="supervisor failed"):
                    pool.close(timeout=WAIT_SECONDS)
            else:
                pool.close(timeout=WAIT_SECONDS)


def submit(pool, name):
    return pool.submit(name, "barrier", fixtures.request(), fixtures.limits(wall_seconds=4 * WAIT_SECONDS))


def ready(pool, handle):
    directory = pool.work_root / (handle.store_id + "-" + handle.attempt_id)
    deadline = time.monotonic() + WAIT_SECONDS
    with pool._condition:
        while True:
            receipt = pool._done.get(handle.attempt_id)
            if receipt is not None:
                value = receipt.data
                pytest.fail("owned child ended before the durable barrier: " + json.dumps({
                    key: value.get(key) for key in ("store_state", "worker_result", "resources")
                }))
            path = directory / "ready.json"
            if path.exists():
                assert json.loads(path.read_text())["pid"] == pool._known[handle.attempt_id].pid
                return directory
            remaining = deadline - time.monotonic()
            assert remaining > 0, "owned child did not reach the durable barrier"
            # The file is the child's fsynced acknowledgement, not coalescible
            # progress; polling yields the lock to the real monitor each time.
            pool._condition.wait(min(remaining, 0.01))


def timeout_sample(*args, **kwargs):
    raise subprocess.TimeoutExpired(["ps", "-axo", "pid=,ppid=,lstart=,rss="], 0.5)


def test_early_resource_stop_is_reported_without_masking_the_test_failure(tmp_path, monkeypatch):
    original = execution._process_info

    def over_budget(pids, **kwargs):
        information = original(pids, **kwargs)
        information[pids[0]]["rss_bytes"] = 768 * 1024**2 + 1
        return information

    monkeypatch.setattr(execution, "_process_info", over_budget)
    # The real resource stop precedes fault arming: preserve that diagnosis even
    # in a fixture whose successful body would require fatal supervisor cleanup.
    with pytest.raises(pytest.fail.Exception, match="shared_rss_limit"):
        with runtime(tmp_path, fatal=True) as (store, pool):
            ready(pool, submit(pool, "early-stop"))
    with RunStore(tmp_path / "store", read_only=True) as store:
        assert store.get_run("early-stop")["state"] == "failed"
        assert len(fixtures.terminal_events(store, "early-stop")) == 1


def failed(store, pool, handle, *, started=True):
    receipt = pool.wait(handle, WAIT_SECONDS).data
    assert receipt["store_state"] == store.get_run(handle.run_id)["state"] == "failed"
    assert receipt["terminal_committed"]
    assert receipt["worker_result"]["reason"] == "supervisor_control_failure"
    assert receipt["process_started"] is started
    terminal = fixtures.terminal_events(store, handle.run_id)
    assert len(terminal) == 1 and terminal[0]["payload"]["state"] == "failed"
    assert terminal[0]["payload"]["result"]["worker_result"]["reason"] == "supervisor_control_failure"
    if started:
        fixtures.assert_reaped(pool.wait(handle, 1))
    return receipt


def test_repeated_sampler_timeout_commits_one_terminal_and_ends_live_sse(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        first = submit(pool, "running")
        ready(pool, first)
        queued = submit(pool, "queued")
        with events._HTTPServer(store.root) as server, server.client() as client:
            observer = server.observe("control-failure")
            url = f"/runs/{first.run_id}/attempts/{first.attempt_id}/events"
            with client.stream("GET", url, headers={"X-Test-Subscriber": "control-failure"}) as response:
                assert observer.started.wait(WAIT_SECONDS)
                blocks = events._blocks(response)
                assert [events._next_event(blocks)["kind"] for _ in range(2)] == ["queued", "running"]
                with pool._condition:
                    monkeypatch.setattr(execution, "_process_info", timeout_sample)
                    pool._condition.notify_all()
                value = failed(store, pool, first)
                failed(store, pool, queued, started=False)
                assert store.attempts("queued")[0]["started_ns"] is None
                assert value["worker_result"]["control_reason"] == "rss_measurement_unavailable"
                assert value["resources"]["rss_observation"]["consecutive_missing_samples"] == 3
                streamed = [json.loads(block["data"]) for block in blocks if "data" in block]
                terminal = [record for record in streamed if record["kind"] == "terminal"]
                assert terminal == fixtures.terminal_events(store, "running")
                assert observer.finished.wait(WAIT_SECONDS)
            assert client.get(url, headers={"Last-Event-ID": str(terminal[0]["sequence"])}).status_code == 204


@pytest.mark.parametrize("misses", [1, 2])
def test_bounded_missing_samples_retry_before_launch_and_recover(tmp_path, monkeypatch, misses):
    original_sample = execution._process_info
    calls = []
    launched_after = []
    original_start = execution.ProcessSupervisor._start

    def sample(*args, **kwargs):
        calls.append(True)
        if len(calls) <= misses:
            timeout_sample()
        return original_sample(*args, **kwargs)

    def start(pool, job):
        launched_after.append(len(calls))
        return original_start(pool, job)

    monkeypatch.setattr(execution, "_process_info", sample)
    monkeypatch.setattr(execution.ProcessSupervisor, "_start", start)
    with runtime(tmp_path) as (store, pool):
        handle = submit(pool, "recovers")
        directory = ready(pool, handle)
        assert launched_after == [misses + 1]
        (directory / "release").touch()
        value = fixtures.assert_reaped(pool.wait(handle, WAIT_SECONDS))
        assert value["store_state"] == "succeeded" and value["signals"] == []
        observation = pool.accounting()["rss_observation"]
        assert observation["missing_samples_total"] == misses
        assert observation["consecutive_missing_samples"] == 0
        assert observation["missing_since_monotonic"] is None
        assert len(fixtures.terminal_events(store, handle.run_id)) == 1


@pytest.mark.parametrize("kind", ["empty", "unavailable", "failed-command"])
def test_impossible_measurement_fails_closed_without_launch(tmp_path, monkeypatch, kind):
    calls = []

    def missing(*args, **kwargs):
        calls.append(True)
        if kind == "unavailable":
            raise FileNotFoundError("no resource sampler")
        if kind == "failed-command":
            raise subprocess.CalledProcessError(1, ["ps"])
        return {}

    monkeypatch.setattr(execution, "_process_info", missing)
    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "no-measurement")
        value = failed(store, pool, handle, started=False)
        assert len(calls) == 3
        assert value["resources"]["child_peak_rss_bytes"] is None
        assert store.attempts(handle.run_id)[0]["started_ns"] is None
        assert not list(pool.work_root.glob("*/started.json"))
        assert pool.accounting()["known_family_rss_last_observed_bytes"] is None


@pytest.mark.parametrize("late_success", [False, True])
def test_elapsed_measurement_bound_cannot_be_reset_by_a_late_sample(tmp_path, monkeypatch, late_success):
    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "gap")
        ready(pool, handle)
        original = execution._process_info

        def late(*args, **kwargs):
            pool._rss_missing_since = time.monotonic() - execution._RSS_MAX_GAP_SECONDS - 0.01
            return original(*args, **kwargs)

        with pool._condition:
            if late_success:
                monkeypatch.setattr(execution, "_process_info", late)
            else:
                pool._rss_missing_since = time.monotonic() - execution._RSS_MAX_GAP_SECONDS - 0.01
            pool._condition.notify_all()
        value = failed(store, pool, handle)
        assert value["worker_result"]["control_reason"] == "rss_measurement_unavailable"
        assert "interval" in value["worker_result"]["detail"]


@pytest.mark.parametrize("point", ["before-claim", "after-claim", "after-spawn", "progress", "resource", "finalize"])
def test_unexpected_control_paths_terminalize_active_and_pending_attempts(tmp_path, monkeypatch, point):
    def broken(*args, **kwargs):
        raise RuntimeError("injected " + point)

    with runtime(tmp_path, fatal=True) as (store, pool):
        with pool._condition:
            if point == "before-claim":
                monkeypatch.setattr(pool, "_start", broken)
            elif point == "after-claim":
                monkeypatch.setattr(pool._context, "Event", broken)
            elif point == "after-spawn":
                original = execution._atomic_json

                def audit(path, value):
                    if path.name == "spawned.json":
                        broken()
                    return original(path, value)

                monkeypatch.setattr(execution, "_atomic_json", audit)
            first = submit(pool, "first")
            second = submit(pool, "second")
        started = point not in {"before-claim", "after-claim"}
        if point in {"progress", "resource", "finalize"}:
            directory = ready(pool, first)
            method = {"progress": "_progress", "resource": "_resource_reason", "finalize": "_output_guard"}[point]
            with pool._condition:
                monkeypatch.setattr(pool, method, broken)
                if point == "finalize":
                    (directory / "release").touch()
                pool._condition.notify_all()
        failed(store, pool, first, started=started)
        failed(store, pool, second, started=False)
        assert len(store.attempts("first")) == len(store.attempts("second")) == 1


def test_broken_original_writer_uses_fenced_fresh_connection(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "writer")
        directory = ready(pool, handle)

        def broken(*args, **kwargs):
            raise sqlite3.OperationalError("original writer connection failed")

        with pool._condition:
            monkeypatch.setattr(store, "complete", broken)
            (directory / "release").touch()
            pool._condition.notify_all()
        failed(store, pool, handle)
        with RunStore(store.root, read_only=True) as reopened:
            assert reopened.get_run(handle.run_id)["state"] == "failed"
            assert len(fixtures.terminal_events(reopened, handle.run_id)) == 1


def test_failure_audit_errors_do_not_block_store_terminal(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "audit")
        directory = ready(pool, handle)

        def broken(*args, **kwargs):
            raise OSError("audit filesystem unavailable")

        monkeypatch.setattr(execution, "_atomic_json", broken)
        with pytest.raises(RuntimeError, match="during cancellation"):
            pool.cancel(handle)
        failed(store, pool, handle)
        assert not (directory / "receipt.json").exists()
        assert pool.accounting()["control_cleanup_errors"]


def test_receipt_stat_failure_cannot_strand_other_admitted_runs(tmp_path, monkeypatch):
    original = Path.exists

    def denied(path, *args, **kwargs):
        if path.name == "receipt.json":
            raise PermissionError("receipt directory cannot be inspected")
        return original(path, *args, **kwargs)

    with runtime(tmp_path, fatal=True) as (store, pool):
        first = submit(pool, "first")
        ready(pool, first)
        queued = submit(pool, "queued")
        with pool._condition:
            monkeypatch.setattr(Path, "exists", denied)
            monkeypatch.setattr(execution, "_process_info", timeout_sample)
            pool._condition.notify_all()
        failed(store, pool, first)
        failed(store, pool, queued, started=False)
        assert any("receipt.json: PermissionError" in error for error in pool.accounting()["control_cleanup_errors"])


def test_post_commit_failure_preserves_terminal_and_saved_receipt_bytes(tmp_path, monkeypatch):
    original = execution._atomic_json
    captured = {}

    def audit(path, value):
        original(path, value)
        if path.name == "receipt.json" and value["store_state"] == "succeeded":
            captured["receipt"] = path.read_bytes()
            captured["exit"] = (path.parent / "exit.json").read_bytes()
            raise RuntimeError("after successful terminal and receipt publication")

    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "completed")
        directory = ready(pool, handle)
        with pool._condition:
            monkeypatch.setattr(execution, "_atomic_json", audit)
            (directory / "release").touch()
            pool._condition.notify_all()
        result = pool.wait(handle, WAIT_SECONDS).data
        assert result["store_state"] == store.get_run(handle.run_id)["state"] == "succeeded"
        assert store.attempts(handle.run_id)[0]["result"]["worker_result"] == {"ok": True}
        assert len(fixtures.terminal_events(store, handle.run_id)) == 1
        assert (directory / "receipt.json").read_bytes() == captured["receipt"]
        assert (directory / "exit.json").read_bytes() == captured["exit"]


def test_control_failure_does_not_replace_cancelled_or_newer_attempt(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        first = submit(pool, "retry")
        ready(pool, first)
        with pool._condition:
            current = store.get_run("retry")
            store.cancel("retry", expected_attempt_id=first.attempt_id,
                         expected_generation=current["generation"], reason="external cancel")
            following = store.retry("retry", expected_attempt_id=first.attempt_id,
                                    expected_generation=current["generation"])
            monkeypatch.setattr(execution, "_process_info", timeout_sample)
            pool._condition.notify_all()
        result = fixtures.assert_reaped(pool.wait(first, WAIT_SECONDS))
        assert result["store_state"] == "cancelled"
        assert store.get_run("retry") == following
        assert len(fixtures.terminal_events(store, "retry")) == 1


def test_fail_queued_is_idempotent_and_cannot_bypass_claimed_owner(tmp_path):
    with RunStore(tmp_path / "store") as store:
        queued = store.enqueue("queued", opaque_identity=events.IDENTITY, input_metadata={})
        result = {"reason": "supervisor_control_failure"}
        args = {"expected_attempt_id": queued["current_attempt_id"], "expected_generation": queued["generation"]}
        sequence = store.fail_queued("queued", result=result, **args)
        assert store.fail_queued("queued", result=result, **args) == sequence
        assert store.attempts("queued")[0]["started_ns"] is None
        assert len(fixtures.terminal_events(store, "queued")) == 1
        another = store.retry("queued", **args)
        token = store.claim("queued", expected_attempt_id=another["current_attempt_id"],
                            expected_generation=another["generation"], owner_id="another-owner")
        with pytest.raises(StateConflict):
            store.fail_queued("queued", expected_attempt_id=token.attempt_id,
                              expected_generation=token.generation, result=result)
        with pytest.raises(StateConflict):
            store.fail_queued("queued", result=result, **args)
        assert store.get_run("queued")["state"] == "running"


def test_reconciliation_measurement_failure_still_fences_prior_run_without_releasing_slot(tmp_path, monkeypatch):
    with RunStore(tmp_path / "store") as store:
        work = tmp_path / "work"
        token = fixtures.make_uncertain_intent(store, work, reused=False)
        monkeypatch.setattr(execution, "_process_info", timeout_sample)
        with fixtures.supervisor(store, work, max_active=2) as pool:
            recovery = pool.reconcile_interrupted()
            assert recovery["invalidated_owned_attempts"] == [token.attempt_id]
            assert recovery["released_owned_reservations"] == []
            assert not recovery["unowned_processes_signalled"]
            assert recovery["unresolved_owned_slots"][0]["method"] == "unowned_pid_measurement_unavailable"
            assert store.get_run(token.run_id)["state"] == "failed"
            assert len(fixtures.terminal_events(store, token.run_id)) == 1
            assert pool.accounting()["unresolved_owned_slots"] == 1


def test_ps_failure_rejects_partial_family_rows_and_keeps_the_half_second_cap(monkeypatch):
    observed = []

    def failed_query(argv, **kwargs):
        observed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1, "1 0 Fri Oct 30 00:00:00 2026 10\n", "partial result")

    monkeypatch.setattr(worker_module.subprocess, "run", failed_query)
    with pytest.raises(subprocess.CalledProcessError):
        worker_module._process_info([1], include_descendants=True)
    assert observed[0][1]["timeout"] == 0.5


def test_recovered_sample_still_enforces_the_unchanged_shared_rss_budget(tmp_path, monkeypatch):
    with runtime(tmp_path) as (store, pool):
        handle = submit(pool, "rss-limit")
        ready(pool, handle)
        original = execution._process_info
        count = 0

        def sample(pids, **kwargs):
            nonlocal count
            count += 1
            if count <= 2:
                timeout_sample()
            found = original(pids, **kwargs)
            found[pids[0]]["rss_bytes"] = 768 * 1024**2 + 1
            return found

        with pool._condition:
            monkeypatch.setattr(execution, "_process_info", sample)
            pool._condition.notify_all()
        receipt = fixtures.assert_reaped(pool.wait(handle, WAIT_SECONDS))
        assert receipt["store_state"] == "failed"
        assert receipt["worker_result"]["reason"] == "shared_rss_limit"
        assert len(fixtures.terminal_events(store, handle.run_id)) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, StateConflict])
def test_claim_commit_followed_by_exception_fails_the_exact_owned_generation(tmp_path, monkeypatch, error_type):
    with runtime(tmp_path, fatal=True) as (store, pool):
        claim = store.claim

        def broken(*args, **kwargs):
            claim(*args, **kwargs)
            raise error_type("claim returned no token after its transaction committed")

        monkeypatch.setattr(store, "claim", broken)
        handle = submit(pool, "claimed")
        receipt = failed(store, pool, handle, started=False)
        assert receipt["token"]["generation"] == store.attempts(handle.run_id)[0]["generation"] == 1
        assert receipt["token"]["owner_id"] == pool.owner_id


def test_unwritable_store_retains_failure_intents_and_blocks_replay_until_reconciled(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        first = submit(pool, "running")
        ready(pool, first)
        queued = submit(pool, "queued")

        def unavailable(*args, **kwargs):
            raise sqlite3.OperationalError("storage remains unavailable on both writer connections")

        with monkeypatch.context() as faults:
            with pool._condition:
                faults.setattr(RunStore, "complete", unavailable)
                faults.setattr(RunStore, "fail_queued", unavailable)
                faults.setattr(execution, "_process_info", timeout_sample)
                pool._condition.notify_all()
            one = pool.wait(first, WAIT_SECONDS).data
            two = pool.wait(queued, WAIT_SECONDS).data
            assert one["process_joined"] and not two["process_started"]
            assert not one["terminal_committed"] and not two["terminal_committed"]
            assert "storage remains unavailable" in one["commit_error"]
            assert pool.accounting()["uncommitted_control_failures"] == 2
    with runtime(tmp_path) as (store, recovered):
        assert recovered.accounting()["uncommitted_control_failures"] == 2
        with pytest.raises(RuntimeError, match="explicit reconciliation"):
            submit(recovered, "queued")
        outcome = recovered.reconcile_interrupted()
        assert set(outcome["invalidated_owned_attempts"]) == {first.attempt_id, queued.attempt_id}
        assert outcome["released_owned_reservations"] == [first.attempt_id]
        for handle in (first, queued):
            assert store.get_run(handle.run_id)["state"] == "failed"
            terminal = fixtures.terminal_events(store, handle.run_id)
            assert len(terminal) == 1
            assert terminal[0]["payload"]["result"]["worker_result"]["reason"] == "supervisor_control_failure"
        assert recovered.accounting()["uncommitted_control_failures"] == 0
        assert recovered.reconcile_interrupted()["invalidated_owned_attempts"] == []


def test_terminal_failure_does_not_release_an_unobserved_native_exit_on_reopen(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        handle = submit(pool, "unreaped")
        ready(pool, handle)
        process = pool._active[handle.attempt_id].process
        try:
            with monkeypatch.context() as faults:
                with pool._condition:
                    faults.setattr(process, "kill", lambda: None)
                    faults.setattr(process, "join", lambda timeout=None: None)
                    faults.setattr(execution, "_process_info", timeout_sample)
                    pool._condition.notify_all()
                receipt = pool.wait(handle, WAIT_SECONDS).data
                assert receipt["store_state"] == "failed" and receipt["terminal_committed"]
                assert not receipt["process_joined"] and not receipt["process_exit_observed"]
                assert process.is_alive() and pool.accounting()["active"] == 1
        finally:
            if process.is_alive():
                process.kill()
            process.join(timeout=WAIT_SECONDS)
    with runtime(tmp_path) as (store, recovered):
        assert recovered.accounting()["unresolved_owned_slots"] == 1
        outcome = recovered.reconcile_interrupted()
        assert outcome["released_owned_reservations"] == [handle.attempt_id]
        assert len(fixtures.terminal_events(store, handle.run_id)) == 1


@pytest.mark.parametrize("resource", ["rss", "output"])
def test_measured_shared_limit_before_launch_fails_queue_instead_of_stranding_it(tmp_path, monkeypatch, resource):
    with runtime(tmp_path) as (store, pool):
        original = execution._process_info

        def sample(pids, **kwargs):
            information = original(pids, **kwargs)
            information[pids[0]]["rss_bytes"] = pool._rss_cap + 1
            return information

        with pool._condition:
            if resource == "rss":
                monkeypatch.setattr(execution, "_process_info", sample)
            else:
                pool._shared_output_peak = pool._output_cap + 1
            handle = submit(pool, "no-capacity")
        value = pool.wait(handle, WAIT_SECONDS).data
        assert value["store_state"] == "failed" and not value["process_started"]
        assert value["worker_result"]["reason"] == f"shared_{resource}_limit"
        assert store.attempts(handle.run_id)[0]["started_ns"] is None
        assert len(fixtures.terminal_events(store, handle.run_id)) == 1


def test_emergency_writer_is_bound_to_the_original_store(tmp_path):
    with runtime(tmp_path) as (store, pool), RunStore(tmp_path / "foreign") as foreign:
        handle = submit(pool, "owned")
        directory = ready(pool, handle)
        with pytest.raises(StateConflict, match="different store"):
            pool._failure_commit(foreign, pool._known[handle.attempt_id], {})
        assert foreign.events() == []
        (directory / "release").touch()
        assert pool.wait(handle, WAIT_SECONDS).data["store_state"] == "succeeded"


def retain_control_outage(tmp_path, monkeypatch, *, denied_intent=False, outage_partials=False):
    """Science's two-writer outage: one reaped child and one admitted queue row."""
    with runtime(tmp_path, fatal=True) as (store, pool):
        active = submit(pool, "active")
        ready(pool, active)
        queued = submit(pool, "queued")
        atomic = execution._atomic_json

        def unavailable(*args, **kwargs):
            raise sqlite3.OperationalError("both control-failure writers are unavailable")

        def audit(path, value):
            if denied_intent and path.name == "control-failure.json":
                raise PermissionError("control intent publication denied")
            if outage_partials and path.name == "receipt.json":
                path.with_suffix(".json.part").write_bytes(b'{"interrupted":')
            return atomic(path, value)

        with monkeypatch.context() as fault:
            # Publish the complete outage before the next sampler call can
            # trigger cleanup; no monitor can see a partially armed fault.
            with pool._condition:
                fault.setattr(RunStore, "complete", unavailable)
                fault.setattr(RunStore, "fail_queued", unavailable)
                fault.setattr(execution, "_atomic_json", audit)
                fault.setattr(execution, "_process_info", timeout_sample)
                pool._condition.notify_all()
            receipts = [pool.wait(handle, WAIT_SECONDS).data for handle in (active, queued)]
            assert not any(receipt["terminal_committed"] for receipt in receipts)
            assert receipts[0]["process_joined"] and not receipts[1]["process_started"]
            assert pool.accounting()["uncommitted_control_failures"] == 2
            assert len(list(pool.work_root.glob("*/receipt.json"))) == 2
            if denied_intent:
                assert not list(pool.work_root.glob("*/control-failure.json"))
    return active, queued


@pytest.mark.parametrize("denied_intent", [False, True])
@pytest.mark.parametrize("stale_partial", [False, True])
def test_recovery_isolates_audits_and_discovers_fallback_receipts(
        tmp_path, monkeypatch, denied_intent, stale_partial):
    """RTC-B1/B2, including their combined fault and an unchanged positive control."""
    handles = retain_control_outage(tmp_path, monkeypatch, denied_intent=denied_intent)
    with runtime(tmp_path) as (store, pool):
        assert pool.accounting()["uncommitted_control_failures"] == 2
        with pytest.raises(RuntimeError, match="explicit reconciliation"):
            submit(pool, "queued")
        with pytest.raises(RuntimeError, match="explicit reconciliation"):
            pool.wait(handles[1], 1)
        entry = next(iter(pool._control_recovery.values()))
        partial = entry["directory"] / "receipt.json.part"
        if stale_partial:
            partial.write_bytes(b'{"interrupted":')
        for _ in range(2):
            pool.reconcile_interrupted()
            for handle in handles:
                failed(store, pool, handle, started=handle == handles[0])
                assert len(store.attempts(handle.run_id)) == 1
        if stale_partial:
            assert partial.read_bytes() == b'{"interrupted":'
            assert any("FileExistsError" in error for error in
                       pool.accounting()["control_audit_errors"][partial.parent.name[len(store.store_id) + 1:]])
        assert pool.accounting()["uncommitted_control_failures"] == 0
        duplicate = submit(pool, "queued")
        assert duplicate == handles[1]
        assert pool.wait(duplicate, 1).data["terminal_committed"]
        assert len(list(pool.work_root.glob("*/started.json"))) == 1
    with runtime(tmp_path) as (store, pool):
        assert pool.reconcile_interrupted()["invalidated_owned_attempts"] == []
        for handle in handles:
            failed(store, pool, handle, started=handle == handles[0])


@pytest.mark.parametrize("fault_kind", ["write", "stat", "fsync", "replace"])
def test_recovery_publication_failures_are_per_attempt_and_retryable(tmp_path, monkeypatch, fault_kind):
    handles = retain_control_outage(tmp_path, monkeypatch)
    with runtime(tmp_path) as (store, pool):
        attempt_id, entry = next(iter(pool._control_recovery.items()))
        directory = entry["directory"]
        receipt_path = directory / "receipt.json"
        atomic, exists, fsync, replace = execution._atomic_json, Path.exists, os.fsync, os.replace
        directory_inode = directory.stat().st_ino

        def write(path, value):
            if path == receipt_path:
                raise PermissionError("recovery receipt write denied")
            return atomic(path, value)

        def stat(path):
            if path == receipt_path:
                raise PermissionError("recovery receipt stat denied")
            return exists(path)

        def sync(fd):
            if os.fstat(fd).st_ino == directory_inode:
                raise OSError("recovery directory fsync failed")
            return fsync(fd)

        def publish(source, target):
            if Path(target) == receipt_path:
                raise OSError("recovery receipt rename failed")
            return replace(source, target)

        with monkeypatch.context() as fault:
            target, attribute, replacement = {
                "write": (execution, "_atomic_json", write),
                "stat": (Path, "exists", stat),
                "fsync": (os, "fsync", sync),
                "replace": (os, "replace", publish),
            }[fault_kind]
            fault.setattr(target, attribute, replacement)
            for _ in range(2):
                outcome = pool.reconcile_interrupted()
                assert outcome["uncommitted_control_failures"] == []
                assert outcome["pending_control_audits"] == [attempt_id]
                for handle in handles:
                    failed(store, pool, handle, started=handle == handles[0])
            assert pool.accounting()["control_audit_errors"][attempt_id]
        partial = receipt_path.with_suffix(".json.part")
        partial_bytes = partial.read_bytes() if partial.exists() else None
        outcome = pool.reconcile_interrupted()
        assert outcome["invalidated_owned_attempts"] == []
        assert outcome["pending_control_audits"] == []
        if partial_bytes is not None:
            assert partial.read_bytes() == partial_bytes
    with runtime(tmp_path) as (store, pool):
        assert pool.reconcile_interrupted()["invalidated_owned_attempts"] == []
        assert pool.accounting()["uncommitted_control_failures"] == 0
        for handle in handles:
            failed(store, pool, handle, started=handle == handles[0])


@pytest.mark.parametrize("evidence", ["v1_receipts", "receipt_part", "intent_part", "recovery_stage", "exit_only"])
def test_reopen_discovers_bound_fallback_and_partial_control_evidence(tmp_path, monkeypatch, evidence):
    handles = retain_control_outage(tmp_path, monkeypatch, denied_intent=evidence != "intent_part")
    expected_resources = {}
    for handle in handles:
        directory = tmp_path / "work" / (handle.store_id + "-" + handle.attempt_id)
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text())
        expected_resources[handle.attempt_id] = receipt["resources"]
        if evidence == "v1_receipts":
            del receipt["control_intent"]
            path.write_text(json.dumps(receipt))
        elif evidence in {"receipt_part", "recovery_stage"}:
            path.rename(directory / ("receipt.json.part" if evidence == "receipt_part"
                                     else "receipt.json.recovery-crash.part"))
        elif evidence == "intent_part":
            (directory / "control-failure.json").rename(directory / "control-failure.json.part")
            path.rename(directory / "receipt.fixture-hidden")
        elif handle == handles[0]:
            path.rename(directory / "receipt.fixture-hidden")
    with runtime(tmp_path) as (store, pool):
        assert pool.accounting()["uncommitted_control_failures"] == 2
        assert not pool._done
        pool.reconcile_interrupted()
        for handle in handles:
            receipt = failed(store, pool, handle, started=handle == handles[0])
            if evidence == "exit_only" and handle == handles[0]:
                assert all(value is None for value in receipt["resources"].values())
                assert pool.accounting()["counter_totals"] == {"accepted_steps": None, "calls": None}
            else:
                assert receipt["resources"] == expected_resources[handle.attempt_id]
        assert pool.accounting()["uncommitted_control_failures"] == 0


def test_unresolved_control_failure_is_not_relabelled_by_interrupted_recovery(tmp_path, monkeypatch):
    handles = retain_control_outage(tmp_path, monkeypatch, denied_intent=True)
    complete = RunStore.complete

    def unavailable(store, token, *, state, result):
        if result.get("worker_result", {}).get("reason") == "supervisor_control_failure":
            raise sqlite3.OperationalError("control terminal remains unavailable")
        return complete(store, token, state=state, result=result)

    with runtime(tmp_path) as (store, pool):
        with monkeypatch.context() as fault:
            fault.setattr(RunStore, "complete", unavailable)
            outcome = pool.reconcile_interrupted()
            assert outcome["invalidated_owned_attempts"] == [handles[1].attempt_id]
            assert outcome["uncommitted_control_failures"] == [handles[0].attempt_id]
            assert store.get_run("active")["state"] == "running"
            assert fixtures.terminal_events(store, "active") == []
            assert pool.accounting()["unresolved_owned_slots"] == 1
            failed(store, pool, handles[1], started=False)
        pool.reconcile_interrupted()
        failed(store, pool, handles[0])


def test_recovery_audit_does_not_release_an_uncertain_native_slot(tmp_path, monkeypatch):
    active, queued = retain_control_outage(tmp_path, monkeypatch)
    with runtime(tmp_path) as (store, pool):
        directory = pool._recovery[active.attempt_id]["directory"]
        partial = directory / "reconciled.json.part"
        partial.write_bytes(b'{"exit":')
        atomic = execution._atomic_json

        def denied(path, value):
            if path.name.startswith("reconciled.json"):
                raise PermissionError("exit audit temporarily unavailable")
            return atomic(path, value)

        with monkeypatch.context() as fault:
            fault.setattr(execution, "_atomic_json", denied)
            result = pool.reconcile_interrupted()
            assert result["released_owned_reservations"] == []
            assert pool.accounting()["unresolved_owned_slots"] == 1
            failed(store, pool, active)
            failed(store, pool, queued, started=False)
        assert pool.reconcile_interrupted()["released_owned_reservations"] == [active.attempt_id]
        assert partial.read_bytes() == b'{"exit":'
        assert pool.reconcile_interrupted()["released_owned_reservations"] == []


@pytest.mark.parametrize("change", ["success", "retry", "foreign_claim"])
def test_recovery_preserves_existing_terminals_and_cannot_take_foreign_authority(tmp_path, monkeypatch, change):
    active, queued = retain_control_outage(tmp_path, monkeypatch)
    with RunStore(tmp_path / "store") as store:
        if change == "foreign_claim":
            current = store.get_run("queued")
            store.claim("queued", expected_attempt_id=queued.attempt_id,
                        expected_generation=current["generation"], owner_id="different-supervisor")
        elif change == "retry":
            current = store.get_run("active")
            store.cancel("active", expected_attempt_id=active.attempt_id,
                         expected_generation=current["generation"], reason="external terminal")
            store.retry("active", expected_attempt_id=active.attempt_id, expected_generation=current["generation"])
        else:
            path = tmp_path / "work" / (active.store_id + "-" + active.attempt_id) / "control-failure.json"
            token = execution.AttemptToken(**json.loads(path.read_text())["token"])
            store.complete(token, state="succeeded", result={"external": "already committed"})
        preserved = store.get_run("queued" if change == "foreign_claim" else "active")
        history = fixtures.terminal_events(store, "active")
    with runtime(tmp_path) as (store, pool):
        for _ in range(2):
            pool.reconcile_interrupted()
            assert store.get_run(preserved["run_id"]) == preserved
            if change == "foreign_claim":
                failed(store, pool, active)
                assert fixtures.terminal_events(store, "queued") == []
                assert pool.accounting()["administrative_fencing_required"] == [queued.attempt_id]
                with pytest.raises(RuntimeError, match="explicit reconciliation"):
                    submit(pool, "queued")
                with pytest.raises(RuntimeError, match="explicit reconciliation"):
                    pool.wait(queued, 1)
            else:
                assert fixtures.terminal_events(store, "active") == history
                failed(store, pool, queued, started=False)
                assert pool.wait(active, 1).data["store_state"] == ("succeeded" if change == "success" else "cancelled")


def test_no_durable_channel_reports_memory_only_recovery(tmp_path, monkeypatch):
    with runtime(tmp_path, fatal=True) as (store, pool):
        active = submit(pool, "active")
        ready(pool, active)
        queued = submit(pool, "queued")

        def denied(*args, **kwargs):
            raise OSError("no durable writer available")

        with monkeypatch.context() as fault:
            with pool._condition:
                fault.setattr(RunStore, "complete", denied)
                fault.setattr(RunStore, "fail_queued", denied)
                fault.setattr(execution, "_atomic_json", denied)
                fault.setattr(execution, "_process_info", timeout_sample)
                pool._condition.notify_all()
            for handle in (active, queued):
                receipt = pool.wait(handle, WAIT_SECONDS).data
                assert not receipt["terminal_committed"]
                assert receipt["control_recovery_retention"] == {
                    "confirmed_records": [], "memory_only": True, "administrative_fencing_required": True}
            assert set(pool.accounting()["administrative_fencing_required"]) == {active.attempt_id, queued.attempt_id}
        pool.reconcile_interrupted()  # The live owner can still use its in-memory fenced intent.
        failed(store, pool, active)
        failed(store, pool, queued, started=False)


def test_constructor_census_error_releases_the_work_area_lock(tmp_path, monkeypatch):
    census = execution._directory_bytes
    calls = 0

    def denied(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("initial directory census denied")
        return census(path)

    with monkeypatch.context() as fault:
        fault.setattr(execution, "_directory_bytes", denied)
        with pytest.raises(PermissionError, match="census denied"), runtime(tmp_path):
            pass
    with runtime(tmp_path) as (store, pool):
        handle = submit(pool, "reopened")
        (ready(pool, handle) / "release").touch()
        assert pool.wait(handle, WAIT_SECONDS).data["store_state"] == "succeeded"


@pytest.mark.parametrize("denied_intent", [False, True])
def test_outage_retains_fallback_receipts_despite_existing_partial_writes(tmp_path, monkeypatch, denied_intent):
    handles = retain_control_outage(tmp_path, monkeypatch, denied_intent=denied_intent, outage_partials=True)
    with runtime(tmp_path) as (store, pool):
        assert pool.accounting()["uncommitted_control_failures"] == 2
        pool.reconcile_interrupted()
        for handle in handles:
            failed(store, pool, handle, started=handle == handles[0])
            directory = pool.work_root / (handle.store_id + "-" + handle.attempt_id)
            assert (directory / "receipt.json.part").read_bytes() == b'{"interrupted":'


def test_recovered_control_failures_finish_both_live_sse_streams(tmp_path, monkeypatch):
    active, queued = retain_control_outage(tmp_path, monkeypatch, denied_intent=True)
    with runtime(tmp_path) as (store, pool), events._HTTPServer(store.root) as server:
        entry = next(iter(pool._control_recovery.values()))
        (entry["directory"] / "receipt.json.part").write_bytes(b'{"interrupted":')
        active_observer, queued_observer = server.observe("active-recovery"), server.observe("queued-recovery")
        active_url = f"/runs/{active.run_id}/attempts/{active.attempt_id}/events"
        queued_url = f"/runs/{queued.run_id}/attempts/{queued.attempt_id}/events"
        with server.client() as first, server.client() as second:
            with first.stream("GET", active_url, headers={"X-Test-Subscriber": "active-recovery"}) as active_response:
                with second.stream("GET", queued_url, headers={"X-Test-Subscriber": "queued-recovery"}) as queued_response:
                    assert active_observer.started.wait(WAIT_SECONDS) and queued_observer.started.wait(WAIT_SECONDS)
                    active_blocks, queued_blocks = events._blocks(active_response), events._blocks(queued_response)
                    assert [events._next_event(active_blocks)["kind"] for _ in range(2)] == ["queued", "running"]
                    assert events._next_event(queued_blocks)["kind"] == "queued"
                    pool.reconcile_interrupted()
                    for handle, blocks, observer in ((active, active_blocks, active_observer),
                                                      (queued, queued_blocks, queued_observer)):
                        records = [json.loads(block["data"]) for block in blocks if "data" in block]
                        assert [record for record in records if record["kind"] == "terminal"] == fixtures.terminal_events(store, handle.run_id)
                        assert observer.finished.wait(WAIT_SECONDS)
            for handle, url in ((active, active_url), (queued, queued_url)):
                sequence = fixtures.terminal_events(store, handle.run_id)[0]["sequence"]
                assert first.get(url, headers={"Last-Event-ID": str(sequence)}).status_code == 204
        assert pool.reconcile_interrupted()["invalidated_owned_attempts"] == []


def test_reopen_refreshes_committed_store_truth_even_when_the_receipt_write_failed(tmp_path, monkeypatch):
    handles = retain_control_outage(tmp_path, monkeypatch)
    atomic = execution._atomic_json

    def denied(path, value):
        if path.name == "receipt.json":
            raise PermissionError("terminal receipt publication denied")
        return atomic(path, value)

    with runtime(tmp_path) as (store, pool), monkeypatch.context() as fault:
        fault.setattr(execution, "_atomic_json", denied)
        pool.reconcile_interrupted()
        assert pool.accounting()["uncommitted_control_failures"] == 0
        assert pool.accounting()["pending_control_audits"] == 2
    with runtime(tmp_path) as (store, pool):
        assert pool.accounting()["uncommitted_control_failures"] == 0
        assert pool.accounting()["pending_control_audits"] == 2
        assert submit(pool, "queued") == handles[1]
        for handle in handles:
            failed(store, pool, handle, started=handle == handles[0])
        assert pool.reconcile_interrupted()["invalidated_owned_attempts"] == []
        assert pool.accounting()["pending_control_audits"] == 0

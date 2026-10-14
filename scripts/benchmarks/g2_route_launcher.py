"""Commander-owned, per-case POSIX watchdog for the G2 route.

All limits come from the selected RunPlan execution/supervision fields.  The
watchdog does not grant scientific admission or interpret numerical results.
The entry must adopt the inherited channel FD; opening a second flock would
conflict with the watchdog's lock.  See ``inherited_channel`` for that contract.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
CHANNEL_FD_ENV = "SOLARLAB_G2_ROUTE_CHANNEL_FD"
CHANNEL_PROTOCOL = "solarlab.g2-route.inherited-channel.v1"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)


class LauncherError(ValueError):
    """An invalid plan, occupied slot, or incomplete supervision operation."""


def require(condition, message):
    if not condition:
        raise LauncherError(message)


def positive_number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    require(valid, "invalid " + name)
    return float(value)


def positive_integer(value, name):
    require(type(value) is int and value > 0, "invalid " + name)
    return value


@dataclass(frozen=True)
class Limits:
    wall_s: float
    rss_bytes: int
    artifact_bytes: int
    poll_s: float
    term_grace_s: float
    kill_wait_s: float
    receipt_reserve_s: float

    @classmethod
    def from_execution(cls, execution):
        supervision = execution["supervision"]
        result = cls(
            positive_number(execution["wall_s"], "wall_s"),
            positive_integer(supervision["rss_bytes"], "rss_bytes"),
            positive_integer(supervision["artifact_bytes"], "artifact_bytes"),
            *(positive_number(supervision[key], key) for key in
              ("poll_s", "term_grace_s", "kill_wait_s", "receipt_reserve_s")),
        )
        require(result.wall_s > result.cleanup_s + result.poll_s,
                "wall_s must include work, termination, closure and receipt time")
        return result

    @property
    def cleanup_s(self):
        return self.term_grace_s + self.kill_wait_s + self.receipt_reserve_s


def _slot_path(directory, channel):
    require(type(channel) is int and 0 <= channel < 4, "channel must be 0..3")
    root = Path(directory).resolve()
    require(root.is_dir(), "Commander channel directory must already exist")
    return root / f"{channel}.lock"


@contextmanager
def inherited_channel(directory, channel):
    """Use the watchdog's exact locked open-file description without unlocking it.

    The consumer checks inode identity through the inherited FD, without opening
    the slot again.  The watchdog retains its FD until group closure.
    This function intentionally does not issue LOCK_UN, including on exceptions.
    """
    value = os.environ.get(CHANNEL_FD_ENV)
    require(value is not None and value.isdecimal(), "inherited channel FD missing")
    fd = int(value)
    path = _slot_path(directory, channel)
    actual = os.fstat(fd)
    expected = path.stat()
    require(stat.S_ISREG(actual.st_mode) and
            (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
            "inherited channel inode mismatch")
    yield fd


@contextmanager
def _owned_channel(directory, channel):
    path = _slot_path(directory, channel)
    quarantine = path.with_suffix(".quarantine.json")
    require(not quarantine.exists(), "channel quarantined after incomplete closure")
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LauncherError("channel occupied; no case started") from error
        require(not quarantine.exists(), "channel quarantined after incomplete closure")
        try:
            yield stream.fileno(), quarantine
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _orphan_reaper():
    """A standalone Linux launcher can reap its own orphaned grandchildren.

    macOS launchd performs this reaping.  Only explicitly observed PIDs in this
    case's process group are waited on; unrelated children are never reaped.
    """
    libc = previous = None
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    try:
        yield
    finally:
        if libc is not None:
            if libc.prctl(36, previous.value, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "restore child subreaper")


def _process_table(timeout_s):
    """ps RSS is KiB on both supported hosts: macOS and Linux."""
    require(timeout_s > 0, "process inventory deadline exhausted")
    output = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="],
        check=True, capture_output=True, text=True, timeout=timeout_s,
    ).stdout
    result = {}
    for line in output.splitlines():
        pid, ppid, pgid, rss, state = line.split()
        result[int(pid)] = {"pid": int(pid), "ppid": int(ppid),
                            "pgid": int(pgid), "rss_bytes": int(rss)*1024,
                            "state": state}
    require(os.getpid() in result, "process inventory omitted the supervisor")
    return result


def _artifact_size(paths, deadline):
    """Logical file bytes, without following links outside named ownership."""
    total = 0
    pending = list(dict.fromkeys(Path(path).absolute() for path in paths))
    seen = set()
    while pending:
        require(time.monotonic() < deadline, "artifact inventory deadline exhausted")
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode), "artifact symlink is outside bounded ownership")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                for entry in entries:
                    require(time.monotonic() < deadline, "artifact inventory deadline exhausted")
                    pending.append(Path(entry.path))
        else:
            raise LauncherError("unsupported artifact file type")
    return total


def _absent(identifier, *, group=False):
    try:
        (os.killpg if group else os.kill)(identifier, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OverflowError):
        return False
    return False


def _presence(pgid, observed, table):
    checks = [{"pid": pid, "pid_absent": pid not in table and _absent(pid),
               "same_id_process_group_absent": _absent(pid, group=True)}
              for pid in sorted(observed)]
    members = [row for row in table.values() if row["pgid"] == pgid]
    remaining_owned = {row["pid"]: row for row in members}
    remaining_owned.update({pid: table[pid] for pid in observed if pid in table})
    absent = not members and _absent(pgid, group=True)
    return {"at": datetime.now(timezone.utc).isoformat(), "pgid": pgid, "checks": checks,
            "remaining_group_members": members, "remaining_owned_processes": list(remaining_owned.values()),
            "process_group_absent": absent,
            "all_named_owned_processes_and_groups_absent":
                absent and all(
                    row["pid_absent"] and row["same_id_process_group_absent"] for row in checks)}


def _closure(pgid, observed, returncode, table):
    evidence = _presence(pgid, observed, table)
    evidence.update(schema="solarlab.owned-tool-process-closure.v1", after_tool_exit_code=returncode)
    evidence["all_named_owned_processes_and_groups_absent"] &= returncode is not None
    return evidence


def _remaining_commands(pgid, members, deadline):
    """Read command lines for this group only, and timestamp the observation."""
    when = datetime.now(timezone.utc).isoformat()
    pids = [row["pid"] for row in members]
    if not pids:
        return {"at": when, "command": None, "processes": []}
    args = ["/bin/ps", "-ww", "-p", ",".join(map(str, pids)), "-o", "pid=,pgid=,stat=,command="]
    try:
        require(time.monotonic() < deadline, "remaining-process inspection budget exhausted")
        result = subprocess.run(args, capture_output=True, text=True, timeout=deadline-time.monotonic())
        require(result.returncode in (0, 1), "remaining-process inspection failed")
        rows = []
        for line in result.stdout.splitlines():
            pid, group, state, command = line.strip().split(None, 3)
            if int(pid) in pids:
                rows.append({"pid": int(pid), "pgid": int(group), "state": state, "command_line": command})
        return {"at": when, "command": args, "exit_code": result.returncode, "processes": rows}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"at": when, "command": args, "processes": None, "error": str(error)}


@contextmanager
def _cancellation_signals():
    requested = {"signal": None}
    def receive(number, frame):
        requested["signal"] = number
    previous = {number: signal.signal(number, receive) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _encode_receipt(result, artifact_bytes, limit):
    """Include this JSON's own bytes in the reported artifact total."""
    previous = -1
    size = 0
    while size != previous:
        previous = size
        result["receipt_bytes"] = size
        result["artifact_bytes"] = None if artifact_bytes is None else artifact_bytes+size
        if result["artifact_bytes"] is not None:
            result["peak_artifact_bytes"] = max(result["peak_artifact_bytes"], result["artifact_bytes"])
            if result["artifact_bytes"] > limit:
                if result["process_outcome"] == "completed":
                    result["process_outcome"] = "artifact_limit"
                result["artifact_limit_exceeded"] = True
        encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        size = len(encoded)
    return encoded


def supervise(*, command, execution, channel_directory, case_id, plan_sha256,
              started_monotonic=None):
    """Supervise one case in this process; no scientific entry is imported here."""
    started = time.monotonic() if started_monotonic is None else started_monotonic
    started_at = datetime.now(timezone.utc).isoformat()
    limits = Limits.from_execution(execution)
    policy = execution["supervision"]
    require(type(command) in (list, tuple) and command and
            all(type(value) is str and value and "\0" not in value for value in command),
            "command must be a nonempty argv; shell execution is not supported")
    directory = Path(execution["directory"]).resolve()
    receipt_path = Path(policy["closure_path"]).absolute()
    cancel_path = Path(policy["cancel_path"]).absolute()
    log_path = receipt_path.with_suffix(".log")
    require(not receipt_path.is_relative_to(directory),
            "closure_path must be outside the fresh case directory")
    require(receipt_path != cancel_path and log_path != cancel_path,
            "cancel and output paths must differ")
    require(not receipt_path.exists() and not log_path.exists(), "supervision output already exists")
    require(not cancel_path.exists(), "case cancelled before launch")
    deadline = started + limits.wall_s
    work_deadline = deadline - limits.cleanup_s
    require(time.monotonic() < work_deadline, "no time left for case bootstrap")
    env = dict(os.environ)
    env.update({name: "1" for name in THREAD_VARIABLES})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    samples, observed, signals_sent = [], set(), []
    peak_rss = peak_bytes = 0
    rss_peak_sample = artifact_peak_sample = None
    reason = failure = None
    term_at = kill_at = None
    process = None
    table = {}
    with _owned_channel(channel_directory, execution["channel"]) as (fd, quarantine), \
            _orphan_reaper(), _cancellation_signals() as cancellation:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        env[CHANNEL_FD_ENV] = str(fd)
        with log_path.open("xb", buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(fd,))
            pgid = process.pid
            observed.add(pgid)
            session_id = os.getsid(pgid)
            require(pgid != os.getpgrp() and os.getpgid(pgid) == session_id == pgid,
                    "case process group not isolated")

            def stop(number):
                require(pgid != os.getpgrp(), "refuse to signal the supervisor group")
                try:
                    os.killpg(pgid, number)
                except ProcessLookupError:
                    return
                signals_sent.append({"signal": signal.Signals(number).name,
                                     "elapsed_s": time.monotonic()-started, "pgid": pgid})

            while True:
                now = time.monotonic()
                if now >= deadline - limits.receipt_reserve_s:
                    reason = reason or "wall_limit"
                    if term_at is None:
                        stop(signal.SIGTERM)
                    stop(signal.SIGKILL)
                    break
                if reason is None and (cancellation["signal"] is not None or cancel_path.exists()):
                    reason = "cancelled"
                if reason is None and now >= work_deadline:
                    reason = "wall_limit"
                if reason is not None and term_at is None:
                    stop(signal.SIGTERM)
                    term_at = now
                term_deadline = (None if term_at is None else min(
                    term_at+limits.term_grace_s, deadline-limits.kill_wait_s-limits.receipt_reserve_s))
                if term_deadline is not None and kill_at is None and now >= term_deadline:
                    stop(signal.SIGKILL)
                    kill_at = now
                kill_deadline = (None if kill_at is None else min(
                    kill_at+limits.kill_wait_s, deadline-limits.receipt_reserve_s))
                if kill_deadline is not None and now >= kill_deadline:
                    break
                inspection_deadline = (kill_deadline if kill_at is not None else
                                       term_deadline if term_at is not None else work_deadline)
                try:
                    table = _process_table(inspection_deadline-now)
                    members = [row for row in table.values() if row["pgid"] == pgid]
                    observed.update(row["pid"] for row in members)
                    # Keep the unreaped leader as a PID/PGID lease until all
                    # descendants are gone; do not poll/reap it in the loop.
                    for row in members:
                        if row["pid"] != pgid and "Z" in row["state"]:
                            try:
                                os.waitpid(row["pid"], os.WNOHANG)
                            except ChildProcessError:
                                pass
                    others = [row for row in members if row["pid"] != pgid]
                    leader = table.get(pgid)
                    if (leader is None or "Z" in leader["state"]) and not others:
                        process.wait(timeout=max(0, deadline-time.monotonic()-limits.receipt_reserve_s))
                        table = _process_table(deadline-time.monotonic()-limits.receipt_reserve_s)
                        # Never send another group signal after reaping the
                        # leader: its PID lease is now released. Closure below
                        # must independently succeed, or the slot is quarantined.
                        break
                    rss = sum(row["rss_bytes"] for row in members)
                    size = _artifact_size((directory, log_path, receipt_path), inspection_deadline)
                    sample = {"elapsed_s": time.monotonic()-started, "rss_bytes": rss,
                              "artifact_bytes": size, "members": members}
                    if rss_peak_sample is None or rss > peak_rss:
                        rss_peak_sample = sample
                    if artifact_peak_sample is None or size > peak_bytes:
                        artifact_peak_sample = sample
                    peak_rss, peak_bytes = max(peak_rss, rss), max(peak_bytes, size)
                    # Only resource maxima and PID observations are needed;
                    # bounded receipt size does not grow with every poll.
                    samples.append(sample)
                    samples = samples[-2:]
                    if reason is None:
                        if cancellation["signal"] is not None or cancel_path.exists():
                            reason = "cancelled"
                        elif rss > limits.rss_bytes:
                            reason = "rss_limit"
                        elif size > limits.artifact_bytes:
                            reason = "artifact_limit"
                        elif time.monotonic() >= work_deadline:
                            reason = "wall_limit"
                        elif leader is None or "Z" in leader["state"]:
                            reason = "descendants_after_leader_exit"
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    reason = reason or ("wall_limit" if time.monotonic() >= work_deadline else "supervision_failure")
                    failure = type(error).__name__ + ": " + str(error)
                now = time.monotonic()
                if reason is not None and term_at is None:
                    stop(signal.SIGTERM)
                    term_at = now
                next_event = work_deadline if term_at is None else (
                    term_at+limits.term_grace_s if kill_at is None else kill_at+limits.kill_wait_s)
                pause = min(limits.poll_s, max(0, next_event-time.monotonic()),
                            max(0, deadline-limits.receipt_reserve_s-time.monotonic()))
                if pause:
                    time.sleep(pause)

        # A final process inventory is positive evidence; a missing/failed
        # observation is not interpreted as an empty process group.
        try:
            table = _process_table(max(0, deadline-time.monotonic()))
            remaining = [row for row in table.values() if row["pgid"] == pgid]
            if all(row["pid"] == pgid and "Z" in row["state"] for row in remaining):
                process.wait(timeout=max(0, deadline-time.monotonic()))
                table = _process_table(max(0, deadline-time.monotonic()))
            # No more group signals follow this point, so the leader may now
            # be reaped even if independent closure evidence remains incomplete.
            process.poll()
            closure = _closure(pgid, observed, process.returncode, table)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            process.poll()
            closure = {"schema": "solarlab.owned-tool-process-closure.v1", "pgid": pgid,
                       "at": datetime.now(timezone.utc).isoformat(),
                       "after_tool_exit_code": process.returncode,
                       "all_named_owned_processes_and_groups_absent": False,
                       "error": type(error).__name__+": "+str(error)}
        closed = closure["all_named_owned_processes_and_groups_absent"]
        reason = reason or ("closure_failure" if not closed else
                            "completed" if process.returncode == 0 else "child_failed")
        try:
            artifact_bytes = _artifact_size((directory, log_path), deadline)
        except (OSError, LauncherError) as error:
            artifact_bytes = None
            failure = failure or type(error).__name__+": "+str(error)
            reason = "supervision_failure" if reason == "completed" else reason
        if artifact_bytes is not None:
            peak_bytes = max(peak_bytes, artifact_bytes)
            if artifact_bytes > limits.artifact_bytes and reason == "completed":
                reason = "artifact_limit"
        result = {"schema": "solarlab.g2-route.launcher-result.v1", "case_id": case_id,
                  "plan_sha256": plan_sha256, "command": list(command), "cwd": str(REPOSITORY),
                  "pid": process.pid, "pgid": pgid, "session_id": session_id, "channel": execution["channel"],
                  "started_at_utc": started_at,
                  "channel_protocol": CHANNEL_PROTOCOL, "limits": vars(limits),
                  "process_outcome": reason if closed else "closure_failure",
                  "termination_reason": reason, "exit_code": process.returncode,
                  "exit_signal": (-process.returncode if process.returncode is not None and process.returncode < 0 else None),
                  "cancellation_signal": cancellation["signal"], "signals_sent": signals_sent,
                  "peak_group_rss_bytes": peak_rss, "peak_artifact_bytes": peak_bytes,
                  "rss_peak_sample": rss_peak_sample, "artifact_peak_sample": artifact_peak_sample,
                  "artifact_bytes_before_receipt": artifact_bytes, "last_samples": samples,
                  "observed_pids": sorted(observed), "closure": closure, "error": failure,
                  "wall_s": time.monotonic()-started,
                  "wall_includes": "bootstrap, command, observer/readback, I/O, termination and closure; post-receipt timing is also returned on stdout",
                  "rss_measurement": "sampled sum of process-group RSS; ps KiB converted to bytes",
                  "thread_environment": {name: env[name] for name in THREAD_VARIABLES},
                  "scientific_status": "not_evaluated", "scientific_pass": None}
        if result["wall_s"] >= limits.wall_s and reason == "completed":
            result["process_outcome"] = "wall_limit"
        if not closed:
            remaining = closure.get("remaining_owned_processes", [
                row for row in table.values() if row["pgid"] == pgid])
            commands = _remaining_commands(pgid, remaining, deadline)
            marker = {"schema": "solarlab.g2-route.channel-quarantine.v1",
                      "case_id": case_id, "channel": execution["channel"],
                      "at": datetime.now(timezone.utc).isoformat(),
                      "pgid": pgid, "observed_pids": sorted(observed),
                      "remaining_group_members": remaining, "remaining_commands": commands,
                      "plan_sha256": plan_sha256, "closure_path": str(receipt_path),
                      "original_exit_code": process.returncode,
                      "reason": "incomplete closure; Commander must record verification before releasing slot"}
            with quarantine.open("x") as stream:
                json.dump(marker, stream, sort_keys=True, indent=2)
            result.update(quarantine_path=str(quarantine), remaining_commands=commands,
                          channel_reusable=False)
        write_started = time.monotonic()
        with receipt_path.open("xb+") as stream:
            stream.write(_encode_receipt(result, artifact_bytes, limits.artifact_bytes))
            stream.flush()
            result["receipt_initial_write_s"] = time.monotonic()-write_started
            result["wall_s"] = time.monotonic()-started
            result["wall_measurement"] = "after initial receipt write, before final metadata commit"
            if result["wall_s"] >= limits.wall_s and result["process_outcome"] == "completed":
                result["process_outcome"] = "wall_limit"
            stream.seek(0)
            stream.write(_encode_receipt(result, artifact_bytes, limits.artifact_bytes))
            stream.truncate()
            stream.flush()
            if time.monotonic() >= deadline and result["process_outcome"] == "completed":
                result["process_outcome"] = "wall_limit"
                result["wall_s"] = time.monotonic()-started
                stream.seek(0)
                stream.write(_encode_receipt(result, artifact_bytes, limits.artifact_bytes))
                stream.truncate()
        result["wall_through_receipt_s"] = time.monotonic()-started
        return result


def _quarantine_context(plan_path, case_id):
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw)
    matches = [case for case in plan["cases"] if case["id"] == case_id]
    require(len(matches) == 1, "case must be unique")
    execution = matches[0]["execution"]
    limits = Limits.from_execution(execution)
    slot = _slot_path(plan["artifacts"]["channel_directory"], execution["channel"])
    return execution, limits, slot, hashlib.sha256(raw).hexdigest()


def verify_quarantine(plan_path, case_id):
    """Read-only presence check. It does not improve the original failed result."""
    execution, limits, slot, plan_digest = _quarantine_context(plan_path, case_id)
    marker_path = slot.with_suffix(".quarantine.json")
    raw = marker_path.read_bytes()
    marker = json.loads(raw)
    require(marker.get("schema") == "solarlab.g2-route.channel-quarantine.v1" and
            marker.get("case_id") == case_id and marker.get("channel") == execution["channel"] and
            marker.get("plan_sha256") == plan_digest and
            Path(marker["closure_path"]).resolve() == Path(execution["supervision"]["closure_path"]).resolve(),
            "quarantine identity mismatch")
    pgid = positive_integer(marker["pgid"], "quarantine pgid")
    pids = {positive_integer(pid, "quarantine pid") for pid in marker["observed_pids"]} | {pgid}
    deadline = time.monotonic()+limits.kill_wait_s
    table = _process_table(max(0, deadline-time.monotonic()))
    presence = _presence(pgid, pids, table)
    return {"schema": "solarlab.g2-route.quarantine-verification.v1",
            "case_id": case_id, "channel": execution["channel"], "plan_sha256": plan_digest,
            "quarantine_path": str(marker_path), "quarantine_sha256": hashlib.sha256(raw).hexdigest(),
            "quarantine": marker, "verification_only": True,
            "original_case_outcome": "closure_failure", "scientific_status": "not_evaluated",
            "command": [str(Path(sys.executable).resolve()), "-m", "scripts.benchmarks.g2_route_launcher",
                        "verify-quarantine", "--plan", str(Path(plan_path).resolve()), "--case", case_id],
            "process_inventory_command": ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="],
            "remaining_commands": _remaining_commands(pgid, presence["remaining_owned_processes"], deadline),
            **presence}


def release_quarantine(plan_path, case_id, record_path):
    """Commander-only workflow: require a retained record and recheck before unlink.

    The caller owns creation of the release record. No failed result, RunPlan,
    or supplied record is rewritten. Filesystem access is the authority boundary.
    """
    require(record_path is not None, "Commander release record is required")
    execution, limits, slot, plan_digest = _quarantine_context(plan_path, case_id)
    marker_path = slot.with_suffix(".quarantine.json")
    record_path = Path(record_path).resolve()
    require(record_path != marker_path.resolve(), "release record cannot be the quarantine marker")
    record_raw = record_path.read_bytes()
    record = json.loads(record_raw)
    require(record.get("schema") == "solarlab.g2-route.quarantine-release-record.v1" and
            record.get("actor") == "coordinator", "Commander release record schema/actor required")
    saved = record["verification_output"]
    recorded_at, checked_at = datetime.fromisoformat(record["at"]), datetime.fromisoformat(saved["at"])
    require(recorded_at.tzinfo is not None and checked_at.tzinfo is not None and
            checked_at <= recorded_at <= datetime.now(timezone.utc), "release record time invalid")
    require(saved.get("schema") == "solarlab.g2-route.quarantine-verification.v1" and
            saved.get("all_named_owned_processes_and_groups_absent") is True and
            saved.get("case_id") == case_id and saved.get("channel") == execution["channel"] and
            saved.get("plan_sha256") == plan_digest and
            type(saved.get("command")) is list and bool(saved["command"]) and
            record.get("verification_command") == saved.get("command"),
            "release record must retain a successful verification command and output")
    # This operation is the sole explicit exception to the normal acquisition
    # refusal: it locks only to verify and remove a quarantine, never to run work.
    with slot.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = marker_path.read_bytes()
        require(record_path != Path(json.loads(raw)["closure_path"]).resolve(),
                "release record must not overwrite the original failure receipt")
        require(hashlib.sha256(raw).hexdigest() == saved.get("quarantine_sha256") and
                json.loads(raw) == saved.get("quarantine"), "quarantine changed since retained verification")
        fresh = verify_quarantine(plan_path, case_id)
        require(saved["command"] == fresh["command"], "verification command differs from this plan/case entry")
        require(fresh["all_named_owned_processes_and_groups_absent"], "owned PIDs or process group remain")
        require(marker_path.read_bytes() == raw and record_path.read_bytes() == record_raw,
                "quarantine or release record changed during verification")
        marker_path.unlink()
    return {"schema": "solarlab.g2-route.quarantine-release.v1", "case_id": case_id,
            "channel": execution["channel"], "at": datetime.now(timezone.utc).isoformat(),
            "quarantine_removed": str(marker_path), "retained_record": str(record_path),
            "record_sha256": hashlib.sha256(record_raw).hexdigest(), "fresh_verification": fresh,
            "original_case_outcome": "closure_failure", "scientific_status": "not_evaluated"}


def load_case(plan_path, case_id, action):
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw)
    require(plan.get("status") == "admitted", "RunPlan is not admitted")
    resources = plan["resources"]
    require(resources["WS1_channels"] == 4 and resources["threads_per_case"] == 1,
            "RunPlan must declare four single-threaded channels")
    matches = [case for case in plan["cases"] if case["id"] == case_id]
    require(len(matches) == 1, "case must be unique")
    execution = matches[0]["execution"]
    limits = Limits.from_execution(execution)
    require(limits.rss_bytes <= resources["rss_bytes_per_case"] and
            limits.artifact_bytes <= resources["combined_persisted_case_bytes"],
            "case supervision exceeds RunPlan resource ceiling")
    _slot_path(plan["artifacts"]["channel_directory"], execution["channel"])
    for other in plan["cases"]:
        if other["id"] == case_id or type(other.get("execution")) is not dict:
            continue
        theirs = other["execution"]
        if "directory" in theirs:
            left, right = Path(execution["directory"]).resolve(), Path(theirs["directory"]).resolve()
            require(not left.is_relative_to(right) and not right.is_relative_to(left),
                    "case artifact directories overlap")
        for name in ("closure_path", "cancel_path"):
            if name in theirs.get("supervision", {}):
                require(Path(execution["supervision"][name]).resolve() != Path(theirs["supervision"][name]).resolve(),
                        "case supervision paths overlap: "+name)
    command = execution.get("command")
    if command is None:
        require(plan["authority"].get("scientific_execution_authorized") is True,
                "RunPlan does not authorize scientific execution")
        command = [str(REPOSITORY/".venv/bin/python"), "-m", "scripts.benchmarks.g2_route", action,
                   "--plan", str(Path(plan_path).resolve()), "--case", case_id]
    return dict(command=command, execution=execution, channel_directory=plan["artifacts"]["channel_directory"],
                case_id=case_id, plan_sha256=hashlib.sha256(raw).hexdigest())


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "observe", "verify-quarantine", "release-quarantine"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "verify-quarantine":
            result = verify_quarantine(args.plan, args.case)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0 if result["all_named_owned_processes_and_groups_absent"] else 1
        if args.action == "release-quarantine":
            result = release_quarantine(args.plan, args.case, args.record)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0
        result = supervise(**load_case(args.plan, args.case, args.action), started_monotonic=started)
        print(json.dumps({key: result[key] for key in
                          ("case_id", "process_outcome", "exit_code", "scientific_status", "wall_through_receipt_s")}))
        return 0 if result["process_outcome"] == "completed" else 1
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

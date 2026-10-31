"""Scheduled-only pytest sharding and report-only coverage, without source edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import sysconfig
import time
from typing import Any, TextIO
import xml.etree.ElementTree as ET

# Tests deliberately patch these shared modules. Recorder calls must keep the
# originals, including the independent error encoder and already-open streams.
_RECORDER_DUMPS = json.dumps
_RECORDER_LOADS = json.loads
_RECORDER_ERROR_QUOTE = json.encoder.encode_basestring_ascii
_RECORDER_MONOTONIC = time.monotonic
_RECORDER_GETCWD = os.getcwd
_RECORDER_OPEN = open
_RECORDER_REPLACE = os.replace
_RECORDER_FSYNC = os.fsync
_RECORDER_STDERR = os.write

ROOT = Path(__file__).resolve().parents[2]
SUITES = {"root": ROOT, "legacy": ROOT / "perovskite-sim"}
PACKAGE_ROOTS = {
    "solarlab": "src/solarlab",
    "solarlab_research": "src/solarlab_research",
    "solarlab_server": "src/solarlab_server",
    "perovskite_sim": "perovskite-sim/perovskite_sim",
}
# These files contain individually expensive protocols or several old producers.
# Split by test function; parameter reprs are not portable selector identities.
FUNCTION_SHARDS = {
    "tests/integration/test_r1_v4_historical_evidence.py",
    "tests/integration/test_physical_interface_cbo_scan.py",
    "tests/regression/test_twod_validation.py",
    "tests/regression/test_grading_cigs_notch.py",
}
# The integration directory took 56.8 min in run 40966847092 and timed out at
# 110 min in run 40986149015. Its slowest completed file took 11.5 min: retain
# module fixtures within each file and leave ample headroom between files.
FILE_COVERAGE_SHARDS = {"tests/integration"}
# Target 24 slow cases per ordinary shard, but never split a parameterized
# function. A function above this target runs alone; this is not a test limit.
MAX_SLOW_NODES = 24
# Completed per-node durations from runs 40966847092 and 40986149015, rounded
# up to seconds. Use the file maximum for every node because separate processes
# recreate shared fixtures. Smaller files receive the 20-minute floor.
MEASURED_SLOW_NODE_SECONDS = {
    "tests/integration/test_degradation_snapshot_refinement.py": 323,
    "tests/integration/test_foundation_mobile_refinement.py": 1208,
    "tests/integration/test_foundation_tpv_spatial_refinement.py": 2011,
    "tests/integration/test_physical_interface_cbo_scan.py": 2898,
    "tests/integration/test_r1_evidence_accounting_v5.py": 465,
    "tests/integration/test_r1_v4_historical_evidence.py": 613,
    "tests/regression/test_csi_quasi_fermi_convergence.py": 508,
    "tests/regression/test_interface_balance_ridge.py": 696,
    "tests/regression/test_interface_srv_grid_reference.py": 476,
    "tests/regression/test_mesh_convergence.py": 304,
    "tests/regression/test_symmetry.py": 317,
    "tests/regression/test_twod_microstructure.py": 452,
    "tests/regression/test_twod_validation.py": 1864,
}
THREADS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
# These tests observe real child-process deadlines. Instrument the pytest
# parent, but leave the measured children in the same mode as push CI.
PARENT_ONLY_COVERAGE = {
    "tests/numerical_contracts/test_g2_route_launcher.py":
        "Synthetic process-group startup/closeout is measured inside a fixed wall budget.",
    "tests/runtime/test_ionic_jv_job_service.py":
        "The HTTP worker has a fixed 30 s wall limit; child tracing changes the measured workload.",
}


def read(path: str | Path) -> Any:
    with _RECORDER_OPEN(path, encoding="utf-8") as stream:
        return _RECORDER_LOADS(stream.read())


def write(path: str | Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_nodeid(nodeid: str) -> str:
    """Keep file/class/function identity, without machine-dependent case reprs."""
    return nodeid.partition("[")[0]


def function_counts(nodeids: Iterable[str]) -> Counter[str]:
    return Counter(function_nodeid(nodeid) for nodeid in nodeids)


def function_groups(items: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[function_nodeid(item["nodeid"])].append(item)
    return [groups[name] for name in sorted(groups)]


def slow_node_seconds(suite: str, path: str) -> int:
    measured = MEASURED_SLOW_NODE_SECONDS.get(path, 0) if suite == "legacy" else 0
    # Four times the measured maximum, rounded to minutes, within 20-60 min.
    # The largest completed case is 48.3 min; a global 4x cap would outlive the
    # unchanged 110-minute workflow step and could not close a stalled node.
    return max(1200, min(3600, ((4 * measured + 59) // 60) * 60))


def commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def pinned_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update({name: "1" for name in THREADS})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Each child receives this runner's explicit coverage policy, never an
    # outer caller's automatic subprocess tracing configuration.
    env.pop("COVERAGE_PROCESS_CONFIG", None)
    env.pop("COVERAGE_PROCESS_START", None)
    return env


def verify_installation() -> dict[str, dict[str, Any]]:
    """Reject a stale local-project wheel before tracing it as current source."""
    packages = {}
    for name, relative in PACKAGE_ROOTS.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError("installed package has no source origin: " + name)
        installed = Path(spec.origin).resolve().parent
        if not installed.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("scheduled CI requires a non-editable installation: " + name)
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", relative], cwd=ROOT
        ).decode().split("\0")
        checked = 0
        for filename in filter(None, names):
            source = ROOT / filename
            target = installed / source.relative_to(ROOT / relative)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError("installed package differs from checkout: " + filename)
            checked += 1
        if not checked:
            raise ValueError("empty package source inventory: " + name)
        packages[name] = {"origin": str(installed), "files_verified": checked}
    return packages


def _recorder_error_message(exc: BaseException) -> str:
    try:
        return str(exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return "Recorder exception could not be formatted"


def _recorder_error_line(event: dict[str, Any]) -> str:
    """Encode flat diagnostic records without consulting JSONEncoder methods."""
    try:
        fields = []
        for key, value in sorted(event.items()):
            if value is None:
                encoded = "null"
            elif type(value) is int:
                encoded = str(value)
            else:
                encoded = _RECORDER_ERROR_QUOTE(str(value))
            fields.append(_RECORDER_ERROR_QUOTE(key) + ": " + encoded)
        return "{" + ", ".join(fields) + "}\n"
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return '{"event":"recorder_error","stage":"error_record_encoding"}\n'


def _recorder_diagnostic(event: dict[str, Any]) -> None:
    """A failed evidence sink must still produce a non-recursive diagnostic."""
    try:
        _RECORDER_STDERR(2, ("SCHEDULED RECORDER: " + _recorder_error_line(event)).encode())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        pass  # The caller still fails the run if even stderr is unavailable.


def _execution_gaps(items: list[dict[str, Any]], rows: list[dict[str, Any]],
                    started: set[str]) -> tuple[list[str], list[str]]:
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in rows:
        reports[report["nodeid"]].append(report)
    missing, incomplete = [], []
    for item in items:
        nodeid = item["nodeid"]
        phases = {row["when"] for row in reports[nodeid]}
        if nodeid not in started and not phases:
            missing.append(nodeid)
            continue
        setup_stopped = any(row["when"] == "setup" and row["outcome"] in {"failed", "skipped"}
                            for row in reports[nodeid])
        required = {"setup", "teardown"} | (set() if setup_stopped else {"call"})
        if not required <= phases:
            incomplete.append(nodeid)
    return missing, incomplete


class _PytestRecorder:
    def __init__(self, output: Path, expected: list[str] | None, journal: TextIO,
                 errors: TextIO, cwd: Path) -> None:
        self.output = str(output)
        self.temporary = str(output.with_suffix(output.suffix + ".tmp"))
        self.cwd = cwd
        self.expected = expected
        self.journal_write, self.journal_flush = journal.write, journal.flush
        self.error_write, self.error_flush = errors.write, errors.flush
        self.error_fileno = errors.fileno
        self.started = _RECORDER_MONOTONIC()
        self.started_nodes: set[str] = set()
        self.collection_complete = False
        self.collect_only = False
        self.data: dict[str, Any] = {
            "items": [], "collection_reports": [], "reports": [],
            "recorder_errors": [], "status": "collecting",
        }

    def error(self, stage: str, exc: BaseException, nodeid: str | None = None) -> None:
        event = {"event": "recorder_error", "stage": stage, "nodeid": nodeid,
                 "error_type": type(exc).__name__, "message": _recorder_error_message(exc)}
        self.data["recorder_errors"].append(event)
        # Do not retry the operation that failed or use the normal event path.
        # The separate sink remains usable if the main journal itself failed.
        line = _recorder_error_line(event)
        for write_line, flush in ((self.journal_write, self.journal_flush),
                                  (self.error_write, self.error_flush)):
            try:
                write_line(line)
                flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                pass
        _recorder_diagnostic(event)

    def event(self, value: dict[str, Any]) -> None:
        try:
            self.journal_write(_RECORDER_DUMPS(value) + "\n")
            self.journal_flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("journal", exc, value.get("nodeid"))

    def snapshot(self) -> bool:
        try:
            with _RECORDER_OPEN(self.temporary, "w", encoding="utf-8") as stream:
                stream.write(_RECORDER_DUMPS(self.data, indent=2, sort_keys=True) + "\n")
                stream.flush()
                _RECORDER_FSYNC(stream.fileno())
            _RECORDER_REPLACE(self.temporary, self.output)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("snapshot", exc)
            return False
        return True

    def pytest_collectreport(self, report: Any) -> None:
        try:
            if report.failed or report.skipped:
                self.data["collection_reports"].append({
                    "nodeid": report.nodeid, "outcome": report.outcome,
                    "reason": str(report.longrepr),
                })
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("collection_report", exc)

    def pytest_collection_finish(self, session: Any) -> None:
        try:
            import pytest

            self.collect_only = session.config.option.collectonly
            self.data["items"] = [
                {"nodeid": item.nodeid,
                 "path": item.path.relative_to(self.cwd).as_posix(),
                 "markers": sorted({mark.name for mark in item.iter_markers()})}
                for item in session.items
            ]
            selection_error = None
            if self.expected is not None:
                actual = function_counts(item["nodeid"] for item in self.data["items"])
                wanted = function_counts(self.expected)
                if actual != wanted:
                    selection_error = pytest.UsageError(
                        "scheduled selection drift: missing="
                        + repr(dict(sorted((wanted - actual).items())))
                        + "; extra="
                        + repr(dict(sorted((actual - wanted).items())))
                    )
            if selection_error is None:
                self.collection_complete = True
                self.data["status"] = "running"
                self.snapshot()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("collection", exc)
            raise pytest.UsageError("scheduled collection could not be verified") from exc
        if selection_error is not None:
            # Preserve the established intentional selection refusal (exit 4).
            raise selection_error

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        try:
            self.started_nodes.add(nodeid)
            self.event({"event": "start", "nodeid": nodeid,
                        "seconds": _RECORDER_MONOTONIC() - self.started})
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("test_start", exc, nodeid)

    def pytest_runtest_logreport(self, report: Any) -> None:
        try:
            recorded = {
                "nodeid": report.nodeid, "when": report.when,
                "outcome": report.outcome, "duration": report.duration,
                "wasxfail": getattr(report, "wasxfail", None),
            }
            if report.failed:
                recorded["longrepr"] = str(report.longrepr)
            self.data["reports"].append(recorded)
            self.event({"event": "report", **recorded})
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("test_report", exc)

    def finish(self, pytest_code: int) -> int:
        try:
            return self._finish(pytest_code)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("session_finish", exc)
            code = pytest_code or 1
            self.data.update(status="recorder_error", exit_code=code,
                             pytest_exit_code=pytest_code, execution_complete=False)
            self.snapshot()
            return code

    def _finish(self, pytest_code: int) -> int:
        missing: list[str] = []
        incomplete: list[str] = []
        if not self.collect_only:
            missing, incomplete = _execution_gaps(self.data["items"], self.data["reports"], self.started_nodes)
        complete = self.collection_complete and not missing and not incomplete
        self.data.update(
            pytest_exit_code=pytest_code, not_started_nodeids=missing,
            incomplete_nodeids=incomplete, collection_complete=self.collection_complete,
            execution_complete=complete and not self.collect_only,
            started_nodeids=sorted(self.started_nodes),
        )
        try:
            self.data["seconds"] = _RECORDER_MONOTONIC() - self.started
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("session_clock", exc)
        try:
            self.error_flush()
            _RECORDER_FSYNC(self.error_fileno())
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.error("error_sink_flush", exc)
        code = pytest_code
        if not complete:
            status = "incomplete"
            code = code or 1
            _recorder_diagnostic({"event": "incomplete", "not_started": len(missing),
                                  "incomplete": len(incomplete), "pytest_exit_code": pytest_code})
        elif self.data["recorder_errors"]:
            status, code = "recorder_error", code or 1
        else:
            status = "collected" if self.collect_only else "completed"
        self.data.update(status=status, exit_code=code)
        if not self.snapshot():
            code = code or 1
            self.data.update(status="incomplete" if not complete else "recorder_error", exit_code=code)
            self.snapshot()  # A transient final-write failure must not leave a passing snapshot.
        if self.data["recorder_errors"]:
            # Test-time diagnostics may be captured even when the test passes.
            # Repeat the failing result after pytest has released its capture.
            _recorder_diagnostic({"event": "recorder_error", "stage": "session_result",
                                  "errors": len(self.data["recorder_errors"]), "exit_code": code})
        return code


def pytest_session(arguments: list[str], output: Path, expected: list[str] | None = None) -> int:
    """Preserve pytest results; recorder errors fail the receipt, not the test loop."""
    import pytest

    try:
        cwd = Path(_RECORDER_GETCWD())
        sys.path.insert(0, str(cwd))
        output = Path(output).absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Both handles are opened before any test code or test monkeypatch runs.
        with _RECORDER_OPEN(output.with_suffix(".jsonl"), "w", encoding="utf-8") as journal, \
                _RECORDER_OPEN(output.with_suffix(".recorder-errors.jsonl"), "w", encoding="utf-8") as errors:
            recorder = _PytestRecorder(output, expected, journal, errors, cwd)
            try:
                result = pytest.main(arguments, plugins=[recorder])
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                recorder.error("pytest_session", exc)
                return recorder.finish(1)
            return recorder.finish(int(result))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        _recorder_diagnostic({"event": "recorder_error", "stage": "recorder_files",
                              "error_type": type(exc).__name__, "message": _recorder_error_message(exc)})
        return 1


def collect(suite: str, directory: Path) -> dict[str, Any]:
    output = directory / (suite + "-collection.json")
    command = [
        sys.executable, str(Path(__file__).resolve()), "_collect", "--out",
        str(output), "--", "tests", "--collect-only", "-q", "-p",
        "no:cacheprovider", "-m", "slow or not slow",
    ]
    with (directory / (suite + "-collection.log")).open("w") as log:
        result = subprocess.run(
            command, cwd=SUITES[suite], env=pinned_environment(),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{suite} collection failed; inspect {output.with_suffix('.log')}")
    return read(output)


def make_shard(kind: str, suite: str, paths: Sequence[str], markers: str,
               items: list[dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    key = json.dumps([kind, suite, paths], separators=(",", ":"))
    identifier = name or f"{kind}-{suite}-" + hashlib.sha256(key.encode()).hexdigest()[:10]
    shard: dict[str, Any] = {
        "id": identifier, "suite": suite, "paths": list(paths),
        "markers": markers, "nodeids": [item["nodeid"] for item in items],
        "function_counts": dict(sorted(function_counts(item["nodeid"] for item in items).items())),
        "count": len(items), "ignored": [],
    }
    if kind == "slow":
        shard["node_wall_seconds"] = max(slow_node_seconds(suite, item["path"]) for item in items)
    return shard


def verify_partition(shards: list[dict[str, Any]], expected: Iterable[tuple[str, str]]) -> None:
    actual = Counter((shard["suite"], node) for shard in shards for node in shard["nodeids"])
    wanted = Counter(expected)
    if actual != wanted:
        raise ValueError(f"partition mismatch: missing={wanted - actual}; extra={actual - wanted}")
    owners: dict[tuple[str, str], str] = {}
    for shard in shards:
        for function in function_counts(shard["nodeids"]):
            key = (shard["suite"], function)
            if key in owners:
                raise ValueError(f"test function split across shards: {key}: {owners[key]}, {shard['id']}")
            owners[key] = shard["id"]
    if len({shard["id"] for shard in shards}) != len(shards):
        raise ValueError("duplicate shard ID")
    if len(shards) > 256:
        raise ValueError("GitHub matrix limit exceeded; split the workflow explicitly")


def build_shards(collections: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    policy = runpy.run_path(str(ROOT / "tests/ci/policy.py"))
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    slow, coverage = [], []
    expected_slow: list[tuple[str, str]] = []
    expected_default: list[tuple[str, str]] = []
    for suite, inventory in collections.items():
        by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        default_items = []
        for item in inventory["items"]:
            is_slow = "slow" in item["markers"]
            (expected_slow if is_slow else expected_default).append((suite, item["nodeid"]))
            if is_slow:
                by_file[item["path"]].append(item)
            else:
                default_items.append(item)
        for file, items in sorted(by_file.items()):
            if suite == "legacy" and (file in FUNCTION_SHARDS or file in MEASURED_SLOW_NODE_SECONDS):
                chunks = function_groups(items)
            elif len(items) > MAX_SLOW_NODES:
                chunks = []
                chunk: list[dict[str, Any]] = []
                for group in function_groups(items):
                    if chunk and len(chunk) + len(group) > MAX_SLOW_NODES:
                        chunks.append(chunk)
                        chunk = []
                    chunk.extend(group)
                if chunk:
                    chunks.append(chunk)
            else:
                slow.append(make_shard("slow", suite, [file], "slow", items))
                continue
            for chunk in chunks:
                paths = sorted(function_counts(item["nodeid"] for item in chunk))
                slow.append(make_shard("slow", suite, paths, "slow", chunk))
        if suite == "root":
            for selection in policy["workflow_selections"](workflow):
                selected = [
                    item for item in default_items
                    if any(item["path"] == path or item["path"].startswith(path + "/")
                           for path in selection.directories)
                    and item["path"] not in selection.ignored_files
                    and policy["matches"](selection.markers, set(item["markers"]))
                ]
                suffix = "-" + selection.isolated_group if selection.isolated_group else ""
                shard = make_shard(
                    "coverage", suite, selection.directories, selection.markers,
                    selected, "coverage-root-" + selection.job + suffix,
                )
                shard["ignored"] = list(selection.ignored_files)
                shard["ci_job"] = selection.job
                isolated = {
                    path: [item for item in selected if item["path"] == path]
                    for path in PARENT_ONLY_COVERAGE
                    if any(item["path"] == path for item in selected)
                }
                if isolated:
                    shard["nodeids"] = [item["nodeid"] for item in selected
                                        if item["path"] not in isolated]
                    shard["count"] = len(shard["nodeids"])
                    shard["function_counts"] = dict(sorted(function_counts(shard["nodeids"]).items()))
                    shard["ignored"] += sorted(isolated)
                if shard["count"]:
                    coverage.append(shard)
                for path, items in sorted(isolated.items()):
                    separate = make_shard(
                        "coverage", suite, [path], selection.markers, items,
                        shard["id"] + "-" + Path(path).stem.removeprefix("test_"),
                    )
                    separate["ci_job"] = selection.job
                    separate["subprocess_coverage"] = False
                    separate["coverage_limitation"] = PARENT_ONLY_COVERAGE[path]
                    coverage.append(separate)
        else:
            groups = defaultdict(list)
            for item in default_items:
                groups[Path(item["path"]).parent.as_posix()].append(item)
            for parent, items in sorted(groups.items()):
                # Exact files avoid covering nested directories a second time.
                files = sorted({item["path"] for item in items})
                name = "coverage-legacy-" + parent.removeprefix("tests/").replace("/", "-")
                if parent in FILE_COVERAGE_SHARDS:
                    for file in files:
                        coverage.append(make_shard(
                            "coverage", suite, [file], "not slow",
                            [item for item in items if item["path"] == file],
                            name + "-" + Path(file).stem.removeprefix("test_"),
                        ))
                else:
                    coverage.append(make_shard("coverage", suite, files, "not slow", items, name))
    verify_partition(slow, expected_slow)
    verify_partition(coverage, expected_default)
    return {"slow": slow, "coverage": coverage}


def choose(shards: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    if requested == "all":
        return list(shards)
    names = requested.split(",")
    if not names or any(not name.strip() for name in names):
        raise ValueError("use all or comma-separated exact shard IDs")
    names = [name.strip() for name in names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate requested shard ID")
    known = {shard["id"]: shard for shard in shards}
    if set(names) - known.keys():
        raise ValueError(f"unknown shards: {sorted(set(names) - known.keys())}")
    return [known[name] for name in names]


def plan(args: argparse.Namespace) -> None:
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    installation = verify_installation()
    collections = {suite: collect(suite, directory) for suite in SUITES}
    shards = build_shards(collections)
    selected = {}
    for kind in shards:
        selected[kind] = choose(shards[kind], getattr(args, kind + "_shards"))
        if getattr(args, "run_" + kind) == "false":
            selected[kind] = []
    value = {
        "schema": "solarlab.scheduled-ci-plan.v2", "commit": commit(),
        "selection_identity": "test function with exact per-function case counts; raw case IDs retained as evidence",
        "lock_sha256": sha256(ROOT / "uv.lock"), "thread_count": 1,
        "installed_source_verification": installation,
        "all_shards": shards, "selected_shards": selected,
        "counts": {kind: sum(s["count"] for s in rows) for kind, rows in shards.items()},
        "collection_reports": {suite: v["collection_reports"] for suite, v in collections.items()},
    }
    write(directory / "plan.json", value)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write(f"commit={value['commit']}\n")
            for kind, rows in selected.items():
                out.write(f"{kind}_enabled={str(bool(rows)).lower()}\n")
                # A dummy row keeps the matrix valid when its job is disabled.
                matrix = {"include": [{"id": s["id"], "suite": s["suite"]} for s in rows]
                          or [{"id": "disabled", "suite": "root"}]}
                out.write(f"{kind}_matrix={json.dumps(matrix, separators=(',', ':'))}\n")
    print(json.dumps(value["counts"], sort_keys=True))
    for kind, rows in shards.items():
        for row in rows:
            print(kind, row["id"], row["count"], *row["paths"])


def coverage_config(path: Path, *, subprocesses: bool = True) -> None:
    """Trace only this checkout and its installed packages, not old producer clones."""
    site = Path(sysconfig.get_paths()["purelib"])
    pairs = {
        "solarlab": [ROOT / "src/solarlab", site / "solarlab"],
        "perovskite_sim": [ROOT / "perovskite-sim/perovskite_sim", site / "perovskite_sim"],
    }
    text = "[run]\nparallel = true\n"
    if subprocesses:
        text += "patch = subprocess\n"
    text += "sigterm = true\nsource =\n"
    text += "".join(f"    {p}\n" for paths in pairs.values() for p in paths)
    text += "\n[paths]\n"
    for name, paths in pairs.items():
        text += name + " =\n" + "".join(f"    {p}\n" for p in paths)
    text += "\n[report]\nshow_missing = true\n"
    path.write_text(text)


def run_with_watchdog(command: Sequence[str], *, cwd: Path, env: dict[str, str],
                      log: Path, seconds: float) -> dict[str, Any]:
    """Bound one owned process group; a timeout is never an expected failure."""
    started = time.monotonic()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            # A descendant may ignore TERM even when its parent has exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {"returncode": process.returncode, "timeout": timed_out,
            "seconds": time.monotonic() - started}


def watchdog_failure(suites: ET.Element, nodeid: str, outcome: str, seconds: float,
                     message: str) -> dict[str, Any]:
    suite = ET.SubElement(suites, "testsuite", name="scheduled-watchdog", tests="1",
                          failures="1", errors="0", skipped="0", time=str(seconds))
    case = ET.SubElement(suite, "testcase", classname=nodeid.partition("::")[0],
                         name=nodeid.partition("::")[2], time=str(seconds))
    ET.SubElement(case, "failure", type=outcome, message=message).text = message
    return {"nodeid": nodeid, "when": "watchdog", "outcome": outcome,
            "duration": seconds, "wasxfail": None, "longrepr": message}


def run_slow_nodes(shard: dict[str, Any], out: Path, env: dict[str, str],
                   result: dict[str, Any]) -> int:
    """Verify function counts, then run each worker-local case under a watchdog."""
    seconds = shard["node_wall_seconds"]
    cwd = SUITES[shard["suite"]]
    runner = [sys.executable, str(Path(__file__).resolve())]
    selection = [*shard["paths"], "-m", shard["markers"], "-q", "-ra", "-p", "no:cacheprovider"]
    for ignored in shard["ignored"]:
        selection += ["--ignore", ignored]
    # Only locally collected parameter IDs are executable selectors. The plan
    # is checked first using every function and its exact number of cases.
    collected = out / "collection.json"
    preflight = run_with_watchdog(
        [*runner, "_collect", "--out", str(collected), "--expected", str(out / "expected.json"),
         "--", *selection, "--collect-only"], cwd=cwd, env=env,
        log=out / "pytest.log", seconds=seconds,
    )
    suites = ET.Element("testsuites")
    if preflight["timeout"] or preflight["returncode"]:
        outcome = "timeout" if preflight["timeout"] else "failed"
        report = watchdog_failure(suites, shard["id"] + "::<collection>", outcome,
                                  preflight["seconds"], "Scheduled slow collection did not complete successfully")
        write(out / "tests.json", {"items": [], "reports": [report], "status": "incomplete",
                                   "collection_complete": False, "execution_complete": False,
                                   "not_started_nodeids": read(out / "expected.json"),
                                   "exit_code": preflight["returncode"] or 1})
        ET.ElementTree(suites).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)
        return preflight["returncode"] or 1
    inventory = read(collected)
    aggregate: dict[str, Any] = {
        "items": inventory["items"], "collection_reports": inventory["collection_reports"],
        "reports": [], "node_outcomes": [], "status": "running",
    }
    result["watchdog"] = {"node_wall_seconds": seconds, "scope": "scheduled slow only",
                          "isolation": "one process per locally collected case; shared fixtures recreated"}
    write(out / "result.json", result)
    failed = False
    started = time.monotonic()
    with (out / "tests.jsonl").open("w") as journal, (out / "pytest.log").open("a") as log:
        for index, item in enumerate(inventory["items"]):
            nodeid = item["nodeid"]
            node_out = out / "nodes" / f"{index:04d}"
            node_out.mkdir(parents=True)
            write(node_out / "expected.json", [nodeid])
            journal.write(json.dumps({"event": "start", "nodeid": nodeid}) + "\n")
            journal.flush()
            executed = run_with_watchdog(
                [*runner, "_pytest", "--out", str(node_out / "tests.json"),
                 "--expected", str(node_out / "expected.json"), "--", nodeid,
                 "-m", shard["markers"], "-q", "-ra", "-p", "no:cacheprovider",
                 "--junitxml=" + str(node_out / "junit.xml")],
                cwd=cwd, env=env, log=node_out / "pytest.log", seconds=seconds,
            )
            log.write(f"\n=== {nodeid} ===\n" + (node_out / "pytest.log").read_text())
            reports, receipt_errors = [], []
            if (node_out / "tests.jsonl").is_file():
                for number, line in enumerate((node_out / "tests.jsonl").read_text().splitlines(), 1):
                    try:
                        row = json.loads(line)
                        if row.get("event") == "report":
                            reports.append(row)
                    except (ValueError, AttributeError) as exc:
                        # The watchdog may interrupt a write. Preserve the
                        # timeout and continue; a torn receipt is never a pass.
                        receipt_errors.append(f"journal line {number}: {exc}")
            child, child_xml = None, None
            if not executed["timeout"]:
                try:
                    child = read(node_out / "tests.json")
                    child_xml = ET.parse(node_out / "junit.xml").getroot()
                    if child.get("status") != "completed" or child.get("exit_code") != executed["returncode"]:
                        raise ValueError("incomplete or inconsistent node receipt")
                except (OSError, ValueError, ET.ParseError) as exc:
                    receipt_errors.append(str(exc))
            if executed["timeout"]:
                outcome = "timeout"
                message = f"Scheduled node exceeded its {seconds:g} s wall cap (startup/setup/call/teardown)"
                reports.append(watchdog_failure(suites, nodeid, outcome, executed["seconds"], message))
                log.write(f"TIMEOUT {nodeid}: {message}\n")
                failed = True
            elif receipt_errors:
                recorder_failed = child is not None and child.get("status") == "recorder_error"
                outcome = "recorder_error" if recorder_failed else "failed"
                reports.append(watchdog_failure(suites, nodeid, outcome, executed["seconds"],
                    "Scheduled node recorder error" if recorder_failed else
                    "Scheduled node exited without complete test receipts"))
                failed = True
            else:
                assert child is not None and child_xml is not None
                reports = child["reports"]
                suites.extend(list(child_xml) if child_xml.tag == "testsuites" else [child_xml])
                outcome = "failed" if executed["returncode"] else "passed"
                if not executed["returncode"] and any(r["outcome"] == "skipped" for r in reports):
                    outcome = "xfailed" if any(r.get("wasxfail") for r in reports) else "skipped"
                failed |= bool(executed["returncode"])
            node_result = {"nodeid": nodeid, "outcome": outcome,
                           "receipt_errors": receipt_errors, **executed}
            aggregate["reports"].extend(reports)
            aggregate["node_outcomes"].append(node_result)
            journal.write(json.dumps({"event": "outcome", **node_result}) + "\n")
            journal.flush()
            log.flush()
            write(out / "tests.json", aggregate)
            ET.ElementTree(suites).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)
    aggregate.update(status="completed", exit_code=int(failed), seconds=time.monotonic() - started)
    write(out / "tests.json", aggregate)
    return int(failed)


def _coverage_receipt(out: Path, expected: list[str], exit_code: int) -> dict[str, Any]:
    """A clean process exit cannot certify an absent or unfinished test ledger."""
    errors = []
    items = [{"nodeid": nodeid} for nodeid in expected]
    reports: list[dict[str, Any]] = []
    started: set[str] = set()
    try:
        child = read(out / "tests.json")
        collected = child["items"]
        if function_counts(item["nodeid"] for item in collected) != function_counts(expected):
            errors.append("planned versus collected function counts differ")
        else:
            items = collected
        reports = child["reports"]
        if child.get("status") != "completed" or child.get("exit_code") != exit_code:
            errors.append("missing or inconsistent final recorder status/exit code")
        if not child.get("collection_complete") or not child.get("execution_complete"):
            errors.append("recorder did not certify collection and execution completeness")
        if child.get("recorder_errors"):
            errors.append("recorder errors were reported")
        started.update(child.get("started_nodeids", []))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append("test receipt: " + str(exc))
    try:
        journal_reports = []
        for line in (out / "tests.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "start":
                started.add(row["nodeid"])
            elif row.get("event") == "report":
                journal_reports.append(row)
        # A hard exit may leave the JSON snapshot at collection while the
        # journal still distinguishes started/partial from never-started nodes.
        if errors:
            reports = journal_reports
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append("test journal: " + str(exc))
    try:
        missing, incomplete = _execution_gaps(items, reports, started)
        unexpected = sorted({row["nodeid"] for row in reports} - {item["nodeid"] for item in items})
        if unexpected:
            errors.append("unplanned executed nodes: " + repr(unexpected))
    except (ValueError, TypeError, KeyError) as exc:
        errors.append("test phases: " + str(exc))
        missing, incomplete = expected, []
    if missing or incomplete:
        errors.append("missing complete outcomes for planned nodes")
    return {"receipt_complete": not errors, "receipt_errors": errors,
            "not_started_nodeids": missing, "incomplete_nodeids": incomplete,
            "pytest_process_exit_code": exit_code}


def run_shard(args: argparse.Namespace) -> int:
    value = read(args.plan)
    if value["commit"] != commit() or value["lock_sha256"] != sha256(ROOT / "uv.lock"):
        raise ValueError("plan source/lock does not match this immutable job checkout")
    shard = next(row for row in value["selected_shards"][args.kind] if row["id"] == args.shard)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write(out / "expected.json", shard["nodeids"])
    result: dict[str, Any] = {
        "shard": shard, "commit": value["commit"], "status": "started",
        "plan_sha256": sha256(args.plan), "coverage_data": False,
    }
    write(out / "result.json", result)
    result["installed_source_verification"] = verify_installation()
    write(out / "result.json", result)
    env = pinned_environment()
    if shard.get("ci_job"):
        env["PYTEST_CI_JOB"] = shard["ci_job"]
    if args.kind == "coverage" and shard.get("ci_job") == "core":
        # Live reference loaders require classes from this exact checkout.
        # Audit the non-editable installation above before binding only the
        # pytest child to the same bytes, as the core push job already does.
        env["PYTHONPATH"] = str(ROOT / "src")
        result["execution_source"] = str(ROOT / "src")
    pytest_args = [
        *shard["paths"], "-m", shard["markers"], "-q", "-ra", "-p",
        "no:cacheprovider", "--durations=0", "--junitxml=" + str(out / "junit.xml"),
    ]
    for ignored in shard["ignored"]:
        pytest_args += ["--ignore", ignored]
    command = [sys.executable]
    if args.kind == "coverage":
        config = out / "coverage.ini"
        result["subprocess_coverage"] = shard.get("subprocess_coverage", True)
        coverage_config(config, subprocesses=result["subprocess_coverage"])
        env["COVERAGE_FILE"] = str(out / "coverage.data")
        command += ["-m", "coverage", "run", "--rcfile=" + str(config),
                    "--parallel-mode", "--context=" + shard["id"]]
    command += [
        str(Path(__file__).resolve()), "_pytest", "--out", str(out / "tests.json"),
        "--expected", str(out / "expected.json"), "--", *pytest_args,
    ]
    started = time.monotonic()
    if args.kind == "slow":
        returncode = run_slow_nodes(shard, out, env, result)
    else:
        with (out / "pytest.log").open("w") as log:
            executed = subprocess.run(command, cwd=SUITES[shard["suite"]], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, check=False)
        returncode = executed.returncode
        receipt = _coverage_receipt(out, shard["nodeids"], returncode)
        result.update(receipt)
        if not receipt["receipt_complete"]:
            returncode = returncode or 1
            with (out / "pytest.log").open("a") as log:
                log.write("\nSCHEDULED INCOMPLETE RECEIPT: " + json.dumps(receipt, sort_keys=True) + "\n")
    result.update(
        status="passed" if returncode == 0 else "failed",
        exit_code=returncode, seconds=time.monotonic() - started,
        coverage_data=bool(list(out.glob("coverage.data.*"))),
    )
    write(out / "result.json", result)
    print((out / "pytest.log").read_text())
    return returncode


def combine(args: argparse.Namespace) -> None:
    from coverage import Coverage

    value = read(args.plan)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    installation = verify_installation()
    directories = sorted({path.parent for path in args.data.rglob("coverage.data.*")})
    found = {read(path)["shard"]["id"]: read(path) for path in args.data.rglob("result.json")}
    selected = value["selected_shards"]["coverage"]
    complete = (
        len(selected) == len(value["all_shards"]["coverage"])
        and all(found.get(row["id"], {}).get("status") == "passed"
                and found[row["id"]].get("coverage_data") for row in selected)
    )
    status = {"schema": "solarlab.coverage-report-scope.v1", "commit": value["commit"],
              "installed_source_verification": installation,
              "report_only": True, "fail_under": None, "full_default_run_passed": complete,
              "selected_shards": [row["id"] for row in selected], "shard_results": found,
              "limits": ["Python processes using -S may disable startup instrumentation.",
                         "Historical producer clones and temporary release installations are outside the source filter.",
                         "Original explicit skips and strict expected failures remain visible in JUnit."]}
    write(out / "scope.json", status)
    if not directories:
        raise RuntimeError("no coverage data was produced; the report is incomplete")
    if value["commit"] != commit():
        raise ValueError("coverage report checkout differs from plan source")
    for row in found.values():
        if row["commit"] != value["commit"] or row["plan_sha256"] != sha256(args.plan):
            raise ValueError("cannot combine data from a different source or plan")
    config = out / "coverage.ini"
    coverage_config(config)
    cov = Coverage(data_file=str(out / "coverage.data"), config_file=str(config))
    cov.combine(data_paths=[str(path) for path in directories], strict=True, keep=True)
    cov.save()
    cov.xml_report(outfile=str(out / "coverage.xml"))
    cov.json_report(outfile=str(out / "coverage.json"))
    cov.html_report(directory=str(out / "html"))
    with (out / "coverage.txt").open("w") as stream:
        cov.report(file=stream)
    print((out / "coverage.txt").read_text())
    print(json.dumps({"full_default_run_passed": complete, "report_only": True}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("plan")
    command.add_argument("--out", type=Path, required=True)
    for kind in ("slow", "coverage"):
        command.add_argument("--run-" + kind, choices=("true", "false"), default="true")
        command.add_argument("--" + kind + "-shards", default="all")
    command = sub.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--kind", choices=("slow", "coverage"), required=True)
    command.add_argument("--shard", required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("combine")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    for name in ("_collect", "_pytest"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--expected", type=Path)
        command.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command in ("_collect", "_pytest"):
        arguments = args.pytest_args
        if arguments and arguments[0] == "--":
            arguments = arguments[1:]
        return pytest_session(arguments, args.out, read(args.expected) if args.expected else None)
    if args.command == "plan":
        plan(args)
    elif args.command == "run":
        return run_shard(args)
    else:
        combine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

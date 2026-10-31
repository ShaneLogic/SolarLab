"""Scheduled-only pytest sharding and report-only coverage, without source edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
import xml.etree.ElementTree as ET

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


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_nodeid(nodeid):
    """Keep file/class/function identity, without machine-dependent case reprs."""
    return nodeid.partition("[")[0]


def function_counts(nodeids):
    return Counter(function_nodeid(nodeid) for nodeid in nodeids)


def function_groups(items):
    groups = defaultdict(list)
    for item in items:
        groups[function_nodeid(item["nodeid"])].append(item)
    return [groups[name] for name in sorted(groups)]


def slow_node_seconds(suite, path):
    measured = MEASURED_SLOW_NODE_SECONDS.get(path, 0) if suite == "legacy" else 0
    # Four times the measured maximum, rounded to minutes, within 20-60 min.
    # The largest completed case is 48.3 min; a global 4x cap would outlive the
    # unchanged 110-minute workflow step and could not close a stalled node.
    return max(1200, min(3600, ((4 * measured + 59) // 60) * 60))


def commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def pinned_environment():
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


def verify_installation():
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


def pytest_session(arguments, output, expected=None):
    """Collect actual inherited/parameterized nodes and preserve raw outcomes."""
    import pytest

    data = {"items": [], "collection_reports": [], "reports": [], "status": "collecting"}
    started = time.monotonic()

    class Recorder:
        def pytest_collectreport(self, report):
            if report.failed or report.skipped:
                data["collection_reports"].append({
                    "nodeid": report.nodeid, "outcome": report.outcome,
                    "reason": str(report.longrepr),
                })

        def pytest_collection_finish(self, session):
            data["items"] = [
                {"nodeid": item.nodeid,
                 "path": item.path.relative_to(Path.cwd()).as_posix(),
                 "markers": sorted({mark.name for mark in item.iter_markers()})}
                for item in session.items
            ]
            if expected is not None:
                actual = function_counts(item["nodeid"] for item in data["items"])
                wanted = function_counts(expected)
                if actual != wanted:
                    raise pytest.UsageError(
                        "scheduled selection drift: missing="
                        + repr(dict(sorted((wanted - actual).items())))
                        + "; extra="
                        + repr(dict(sorted((actual - wanted).items())))
                    )
            # Retain the collection even if the outer job later times out.
            data["status"] = "running"
            write(output, data)

        def pytest_runtest_logstart(self, nodeid, location):
            journal.write(json.dumps({"event": "start", "nodeid": nodeid,
                                      "seconds": time.monotonic() - started}) + "\n")
            journal.flush()

        def pytest_runtest_logreport(self, report):
            recorded = {
                "nodeid": report.nodeid, "when": report.when,
                "outcome": report.outcome, "duration": report.duration,
                "wasxfail": getattr(report, "wasxfail", None),
            }
            if report.failed:
                recorded["longrepr"] = str(report.longrepr)
            data["reports"].append(recorded)
            journal.write(json.dumps({"event": "report", **recorded}) + "\n")
            journal.flush()

    sys.path.insert(0, str(Path.cwd()))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).with_suffix(".jsonl").open("w") as journal:
        result = pytest.main(arguments, plugins=[Recorder()])
    data.update(status="completed", exit_code=int(result), seconds=time.monotonic() - started)
    write(output, data)
    return int(result)


def collect(suite, directory):
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


def make_shard(kind, suite, paths, markers, items, name=None):
    key = json.dumps([kind, suite, paths], separators=(",", ":"))
    identifier = name or f"{kind}-{suite}-" + hashlib.sha256(key.encode()).hexdigest()[:10]
    shard = {
        "id": identifier, "suite": suite, "paths": list(paths),
        "markers": markers, "nodeids": [item["nodeid"] for item in items],
        "function_counts": dict(sorted(function_counts(item["nodeid"] for item in items).items())),
        "count": len(items), "ignored": [],
    }
    if kind == "slow":
        shard["node_wall_seconds"] = max(slow_node_seconds(suite, item["path"]) for item in items)
    return shard


def verify_partition(shards, expected):
    actual = Counter((shard["suite"], node) for shard in shards for node in shard["nodeids"])
    wanted = Counter(expected)
    if actual != wanted:
        raise ValueError(f"partition mismatch: missing={wanted - actual}; extra={actual - wanted}")
    owners = {}
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


def build_shards(collections):
    policy = runpy.run_path(str(ROOT / "tests/ci/policy.py"))
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    slow, coverage = [], []
    expected_slow, expected_default = [], []
    for suite, inventory in collections.items():
        by_file = defaultdict(list)
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
                chunks, chunk = [], []
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


def choose(shards, requested):
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


def plan(args):
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


def coverage_config(path, *, subprocesses=True):
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


def run_with_watchdog(command, *, cwd, env, log, seconds):
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


def watchdog_failure(suites, nodeid, outcome, seconds, message):
    suite = ET.SubElement(suites, "testsuite", name="scheduled-watchdog", tests="1",
                          failures="1", errors="0", skipped="0", time=str(seconds))
    case = ET.SubElement(suite, "testcase", classname=nodeid.partition("::")[0],
                         name=nodeid.partition("::")[2], time=str(seconds))
    ET.SubElement(case, "failure", type=outcome, message=message).text = message
    return {"nodeid": nodeid, "when": "watchdog", "outcome": outcome,
            "duration": seconds, "wasxfail": None, "longrepr": message}


def run_slow_nodes(shard, out, env, result):
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
        write(out / "tests.json", {"items": [], "reports": [report], "status": "completed",
                                   "exit_code": preflight["returncode"] or 1})
        ET.ElementTree(suites).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)
        return preflight["returncode"] or 1
    inventory = read(collected)
    aggregate = {"items": inventory["items"], "collection_reports": inventory["collection_reports"],
                 "reports": [], "node_outcomes": [], "status": "running"}
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
                outcome = "failed"
                reports.append(watchdog_failure(suites, nodeid, outcome, executed["seconds"],
                                               "Scheduled node exited without complete test receipts"))
                failed = True
            else:
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


def run_shard(args):
    value = read(args.plan)
    if value["commit"] != commit() or value["lock_sha256"] != sha256(ROOT / "uv.lock"):
        raise ValueError("plan source/lock does not match this immutable job checkout")
    shard = next(row for row in value["selected_shards"][args.kind] if row["id"] == args.shard)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write(out / "expected.json", shard["nodeids"])
    result = {"shard": shard, "commit": value["commit"], "status": "started",
              "plan_sha256": sha256(args.plan), "coverage_data": False}
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
    result.update(
        status="passed" if returncode == 0 else "failed",
        exit_code=returncode, seconds=time.monotonic() - started,
        coverage_data=bool(list(out.glob("coverage.data.*"))),
    )
    write(out / "result.json", result)
    print((out / "pytest.log").read_text())
    return returncode


def combine(args):
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


def main():
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

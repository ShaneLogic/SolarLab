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
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[2]
SUITES = {"root": ROOT, "legacy": ROOT / "perovskite-sim"}
PACKAGE_ROOTS = {
    "solarlab": "src/solarlab",
    "solarlab_research": "src/solarlab_research",
    "solarlab_server": "src/solarlab_server",
    "perovskite_sim": "perovskite-sim/perovskite_sim",
}
# These files contain individually expensive protocols or several old producers.
# Split their collected nodes; every other file retains its shared module fixtures.
NODE_SHARDS = {
    "tests/integration/test_r1_v4_historical_evidence.py",
    "tests/integration/test_physical_interface_cbo_scan.py",
    "tests/regression/test_twod_validation.py",
}
# First measurement budget: at most 24 slow nodes per ordinary shard. The
# initial sample's longest completed case was 82 s; 3 min/node leaves room
# for slower Linux hosts. This is a scheduling estimate, not a test threshold.
MAX_SLOW_NODES = 24
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

    data = {"items": [], "collection_reports": [], "reports": []}
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
                actual = [item["nodeid"] for item in data["items"]]
                if Counter(actual) != Counter(expected):
                    raise pytest.UsageError(
                        "scheduled selection drift: missing="
                        + repr(sorted((Counter(expected) - Counter(actual)).elements()))
                        + "; extra="
                        + repr(sorted((Counter(actual) - Counter(expected)).elements()))
                    )

        def pytest_runtest_logreport(self, report):
            data["reports"].append({
                "nodeid": report.nodeid, "when": report.when,
                "outcome": report.outcome, "duration": report.duration,
                "wasxfail": getattr(report, "wasxfail", None),
            })

    sys.path.insert(0, str(Path.cwd()))
    result = pytest.main(arguments, plugins=[Recorder()])
    data.update(exit_code=int(result), seconds=time.monotonic() - started)
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
    return {
        "id": identifier, "suite": suite, "paths": list(paths),
        "markers": markers, "nodeids": [item["nodeid"] for item in items],
        "count": len(items), "ignored": [],
    }


def verify_partition(shards, expected):
    actual = Counter((shard["suite"], node) for shard in shards for node in shard["nodeids"])
    wanted = Counter(expected)
    if actual != wanted:
        raise ValueError(f"partition mismatch: missing={wanted - actual}; extra={actual - wanted}")
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
            if suite == "legacy" and file in NODE_SHARDS:
                for item in items:
                    slow.append(make_shard("slow", suite, [item["nodeid"]], "slow", [item]))
            elif len(items) > MAX_SLOW_NODES:
                for start in range(0, len(items), MAX_SLOW_NODES):
                    chunk = items[start:start + MAX_SLOW_NODES]
                    slow.append(make_shard(
                        "slow", suite, [item["nodeid"] for item in chunk], "slow", chunk
                    ))
            else:
                slow.append(make_shard("slow", suite, [file], "slow", items))
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
        "schema": "solarlab.scheduled-ci-plan.v1", "commit": commit(),
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
    with (out / "pytest.log").open("w") as log:
        executed = subprocess.run(command, cwd=SUITES[shard["suite"]], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
    result.update(
        status="passed" if executed.returncode == 0 else "failed",
        exit_code=executed.returncode, seconds=time.monotonic() - started,
        coverage_data=bool(list(out.glob("coverage.data.*"))),
    )
    write(out / "result.json", result)
    print((out / "pytest.log").read_text())
    return executed.returncode


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

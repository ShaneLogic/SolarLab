"""Freeze and execute a source-bound R1 regression plan with exact ID accounting.

``freeze`` only collects tests. ``run`` executes one or more precollected shards.
Both require an explicit project and the same isolated dependency environment.
The V5 full non-slow plus affected-slow scope is retained, with explicit extra
files allowed for newly affected paths. No numerical acceptance limit is set
or changed here. ``response-requests`` binds completed short-case bundles to
the existing DC/response analyzer without executing new physics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
STATUSES = ("passed", "failure", "error", "skipped", "incomplete", "not_run")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def content_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def snapshot(project, expected_commit=None):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()
    root = Path(git("rev-parse", "--show-toplevel"))
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("project commit differs from the requested full commit")
    if status:
        raise ValueError("verification requires a clean frozen checkout: " + status)
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {name: sha(root / name) for name in sorted(names) if name}
    return {"project": str(project), "root": str(root), "source_commit": commit,
            "file_hashes": files, "all_tracked_content_sha256": content_digest(files)}


def phase_status(phases):
    if not phases:
        return "not_run"
    if any(phases.get(name, {}).get("outcome") == "failed" for name in ("setup", "teardown")):
        return "error"
    if phases.get("call", {}).get("outcome") == "failed":
        return "failure"
    if any(phase.get("outcome") == "skipped" for phase in phases.values()):
        return "skipped"
    if phases.get("call", {}).get("outcome") == "passed" and "teardown" in phases:
        return "passed"
    return "incomplete"


class VerificationPlugin:
    """Loaded inside the isolated child, so pytest belongs to that environment."""
    def __init__(self, affected, output, expected=None):
        self.affected = set(affected)
        self.output = Path(output)
        self.expected = None if expected is None else set(expected)
        self.items, self.phases, self.collection_errors = {}, {}, []

    def pytest_collection_modifyitems(self, session, config, items):
        import pytest
        kept, excluded = [], []
        for item in items:
            file = item.path.resolve().relative_to(config.rootpath.resolve()).as_posix()
            affected, nonslow = file in self.affected, item.get_closest_marker("slow") is None
            if not (affected or nonslow):
                excluded.append(item)
                continue
            kept.append(item)
            item.user_properties.append(("raw_nodeid", item.nodeid))
            prefix = item.nodeid.removesuffix("::" + item.name)
            base, _, classes = prefix.partition("::")
            classname = base.removesuffix(".py").replace("/", ".")
            if classes:
                classname += "." + classes.replace("::", ".")
            self.items[item.nodeid] = {"file": file, "affected": affected, "nonslow": nonslow,
                                       "junit_key": [classname, item.name]}
        items[:] = kept
        if excluded:
            config.hook.pytest_deselected(items=excluded)
        write(self.output / "CollectedV1.json", {"items": self.items,
              "selected_count": len(self.items), "deselected_slow_count": len(excluded)})
        if self.expected is not None and self.expected != set(self.items):
            write(self.output / "SelectionMismatchV1.json", {
                "missing": sorted(self.expected - set(self.items)),
                "unexpected": sorted(set(self.items) - self.expected)})
            raise pytest.UsageError("collected IDs differ from the externally frozen shard")

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({"nodeid": report.nodeid, "detail": str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        phase = {"outcome": report.outcome, "duration_s": report.duration}
        if report.failed or report.skipped:
            phase["detail"] = str(report.longrepr)
        self.phases.setdefault(report.nodeid, {})[report.when] = phase
        if report.failed:
            with (self.output / "LiveFailuresV1.jsonl").open("a") as stream:
                stream.write(json.dumps({"nodeid": report.nodeid, "phase": report.when, **phase}) + "\n")

    def pytest_sessionfinish(self, session, exitstatus):
        from threadpoolctl import threadpool_info
        results = {nodeid: {**item, "phases": self.phases.get(nodeid, {}),
                           "status": phase_status(self.phases.get(nodeid, {}))}
                   for nodeid, item in self.items.items()}
        write(self.output / "RawResultsV1.json", {
            "schema": "R1V6RawTestResultsV1", "pytest_exit_code": int(exitstatus),
            "collection_only": bool(session.config.option.collectonly),
            "collection_errors": self.collection_errors, "results": results,
            "counts": {status: sum(row["status"] == status for row in results.values()) for status in STATUSES},
            "started_count": len(self.phases), "selected_count": len(self.items),
            "observed_threadpools_at_finish": threadpool_info()})


def child_code(args, pytest_args, affected, output, expected):
    return (
        "import sys,importlib.util;sys.path[:0]=" + repr([str(args.project), *args.dependency_path]) + ";"
        "import numpy,scipy,scipy.linalg;from threadpoolctl import threadpool_info,threadpool_limits;"
        "_limit=threadpool_limits(limits=1,user_api='blas');"
        "assert threadpool_info() and all(p['num_threads']==1 for p in threadpool_info());"
        "s=importlib.util.spec_from_file_location('r1_v6_verification'," + repr(str(Path(__file__).resolve())) + ");"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "p=m.VerificationPlugin(" + repr(affected) + "," + repr(str(output)) + "," + repr(expected) + ");"
        "import pytest;raise SystemExit(pytest.main(" + repr(pytest_args) + ",plugins=[p]))")


def execute_pytest(args, output, pytest_args, affected, expected, before):
    if output.exists():
        raise FileExistsError("verification output already exists: " + str(output))
    output.mkdir(parents=True)
    code = child_code(args, pytest_args, affected, output, expected)
    command = [args.python, "-I", "-S", "-B", "-c", code]
    helper_hash = sha(__file__)
    env = {**os.environ, **{key: "1" for key in THREADS}, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    write(output / "CommandV1.json", {"command": command, "cwd": str(args.project),
          "source_identity": before, "helper_file": str(Path(__file__).resolve()),
          "helper_sha256": helper_hash, "thread_environment": {key: env[key] for key in THREADS},
          "dependency_paths": args.dependency_path,
          "plugin_autoload_disabled": True, "started_utc": datetime.now(timezone.utc).isoformat()})
    start = time.monotonic()
    with (output / "RunV1.log").open("w") as stream:
        process = subprocess.run(command, cwd=args.project, env=env, stdout=stream, stderr=subprocess.STDOUT)
    unchanged, source_error = False, None
    try:
        unchanged = snapshot(args.project, before["source_commit"]) == before
    except (ValueError, subprocess.CalledProcessError) as exc:
        source_error = str(exc)
    result = {"process_exit_code": process.returncode, "duration_s": time.monotonic() - start,
              "source_files_unchanged": unchanged, "source_error": source_error,
              "helper_unchanged": sha(__file__) == helper_hash,
              "all_tracked_content_sha256": before["all_tracked_content_sha256"]}
    if (output / "RawResultsV1.json").is_file():
        raw = read(output / "RawResultsV1.json")
        result.update(counts=raw["counts"], selected_count=raw["selected_count"],
                      started_count=raw["started_count"], collection_errors=raw["collection_errors"])
    else:
        result["missing_raw_results"] = True
    write(output / "ResultV1.json", result)
    return result


def freeze(args):
    before = snapshot(args.project, args.expected_commit)
    prior = read(args.prior_plan)
    affected = sorted(set(read(args.affected)["arguments"]) | set(args.extra_file))
    unknown = [name for name in affected if not (args.project / name).is_file()]
    if unknown:
        raise ValueError("affected test files are missing: " + repr(unknown))
    args.output.mkdir(parents=True, exist_ok=True)
    command = ["--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "--tb=short",
               *prior["dependency_exclusions"], "tests"]
    receipt = execute_pytest(args, args.output / "Collection", command, affected, None, before)
    if receipt["process_exit_code"] != 0 or not receipt["source_files_unchanged"] or not receipt["helper_unchanged"]:
        raise ValueError("collection failed or source changed; no final plan was produced")
    items = read(args.output / "Collection/CollectedV1.json")["items"]
    file_assignment = {file: name for name, part in prior["shards"].items() for file in part["files"]}
    shards = {name: {"files": [], "expected_raw_ids": []} for name in prior["shards"]}
    for file in sorted({row["file"] for row in items.values()}):
        name = file_assignment.get(file)
        if name is None:
            name = min(shards, key=lambda key: len(shards[key]["expected_raw_ids"]))
        shards[name]["files"].append(file)
        shards[name]["expected_raw_ids"].extend(sorted(key for key, row in items.items() if row["file"] == file))
    prior_ids = set(prior["expected_raw_ids"])
    plan = {"schema": "R1V6FrozenVerificationPlanV1", "source_identity": before,
            "helper_sha256": sha(__file__), "affected_files": affected,
            "extra_affected_files": args.extra_file, "dependency_exclusions": prior["dependency_exclusions"],
            "dependency_exclusions_are_not_passes": True, "shards": shards,
            "expected_raw_ids": sorted(items), "added_raw_ids": sorted(set(items) - prior_ids),
            "retired_raw_ids": sorted(prior_ids - set(items)),
            "prior_plan_sha256": sha(args.prior_plan), "affected_input_sha256": sha(args.affected),
            "python": args.python, "dependency_paths": args.dependency_path,
            "scope": "all non-slow plus prior affected and explicitly added affected slow; no verdict-based selection"}
    write(args.output / "PlanV1.json", plan)
    print(json.dumps({"plan": str(args.output / "PlanV1.json"), "plan_sha256": sha(args.output / "PlanV1.json"),
                      "source_sha256": before["all_tracked_content_sha256"], "raw_ids": len(items),
                      "added": len(plan["added_raw_ids"]), "retired": len(plan["retired_raw_ids"])}))


def junit_assessment(path, expected):
    if not path.exists():
        return {"present": False, "missing_ids": sorted(expected), "unexpected_ids": [], "consistent": False}
    reverse = {tuple(row["junit_key"]): key for key, row in expected.items()}
    actual, unmapped = {}, []
    for case in ET.parse(path).findall(".//testcase"):
        properties = {p.get("name"): p.get("value") for p in case.findall("./properties/property")}
        nodeid = properties.get("raw_nodeid") or reverse.get((case.get("classname", ""), case.get("name", "")))
        status = "error" if case.find("error") is not None else "failure" if case.find("failure") is not None else "skipped" if case.find("skipped") is not None else "passed"
        if nodeid is None:
            unmapped.append({"classname": case.get("classname"), "name": case.get("name"), "status": status})
            continue
        # pytest can emit separate call and teardown XML rows for one raw ID.
        old = actual.get(nodeid)
        priorities = {"passed": 0, "skipped": 1, "failure": 2, "error": 3}
        if old is None or priorities[status] > priorities[old]:
            actual[nodeid] = status
    mismatches = [key for key in set(actual) & set(expected) if actual[key] != expected[key]["status"]]
    missing, unexpected = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    return {"present": True, "unique_raw_ids": len(actual), "missing_ids": missing,
            "unexpected_ids": unexpected, "unmapped_rows": unmapped, "status_mismatches": mismatches,
            "consistent": not (missing or unexpected or unmapped or mismatches)}


def run(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("plan differs from the externally supplied digest")
    plan = read(args.plan)
    before = snapshot(args.project, plan["source_identity"]["source_commit"])
    if before != plan["source_identity"] or before["all_tracked_content_sha256"] != args.expected_source_sha256:
        raise ValueError("executed source differs from frozen source identity")
    if sha(__file__) != plan["helper_sha256"]:
        raise ValueError("verification helper changed after collection")
    if args.python != plan["python"] or args.dependency_path != plan["dependency_paths"]:
        raise ValueError("interpreter or dependency search path changed after collection")
    names = args.shard or list(plan["shards"])
    if len(names) != len(set(names)) or any(name not in plan["shards"] for name in names):
        raise ValueError("duplicate or unknown shard")
    failed = False
    for name in names:
        if snapshot(args.project, before["source_commit"]) != before:
            raise ValueError("source changed between shards")
        output, part = args.output / name, plan["shards"][name]
        command = ["-q", "-o", "addopts=", "-p", "no:cacheprovider", "--tb=short",
                   *plan["dependency_exclusions"], *part["files"], "--junitxml=" + str(output / "JUnitV1.xml")]
        receipt = execute_pytest(args, output, command, plan["affected_files"], part["expected_raw_ids"], before)
        if (output / "RawResultsV1.json").exists():
            raw = read(output / "RawResultsV1.json")
            assessment = junit_assessment(output / "JUnitV1.xml", raw["results"])
            write(output / "JUnitAssessmentV1.json", assessment)
            receipt["junit_consistent"] = assessment["consistent"]
        else:
            receipt["junit_consistent"] = False
        write(output / "ResultV1.json", receipt)
        print(json.dumps({"shard": name, **{k: v for k, v in receipt.items() if k != "collection_errors"}}), flush=True)
        failed |= receipt["process_exit_code"] != 0 or not receipt["junit_consistent"]
        if not receipt["source_files_unchanged"] or not receipt["helper_unchanged"]:
            raise ValueError("source/helper changed during verification")
    return 1 if failed else 0


def summarize(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("plan digest mismatch")
    plan, results, shards = read(args.plan), {}, {}
    for name, part in plan["shards"].items():
        folder = args.output / name
        if not (folder / "RawResultsV1.json").exists():
            shards[name] = {"status": "not_run", "expected_count": len(part["expected_raw_ids"])}
            continue
        raw, receipt = read(folder / "RawResultsV1.json"), read(folder / "ResultV1.json")
        if set(results) & set(raw["results"]):
            raise ValueError("duplicate execution across shards")
        results.update(raw["results"])
        shards[name] = {"status": "reported", "receipt": receipt,
                        "exact_ids": set(raw["results"]) == set(part["expected_raw_ids"])}
    for nodeid in set(plan["expected_raw_ids"]) - set(results):
        results[nodeid] = {"status": "not_run", "phases": {}}
    counter = Counter(row["status"] for row in results.values())
    previous = read(args.baseline_results)["results"] if args.baseline_results else {}
    changes = {key: {"before": previous[key]["status"], "after": value["status"]}
               for key, value in results.items() if key in previous and previous[key]["status"] != value["status"]}
    integrity = all(value["status"] == "reported" and value["exact_ids"]
                    and value["receipt"].get("source_files_unchanged") is True
                    and value["receipt"].get("helper_unchanged") is True
                    and value["receipt"].get("junit_consistent") is True
                    and value["receipt"].get("all_tracked_content_sha256") == plan["source_identity"]["all_tracked_content_sha256"]
                    for value in shards.values())
    payload = {"schema": "R1V6VerificationSummaryV1", "plan_sha256": args.plan_sha256,
               "source_sha256": plan["source_identity"]["all_tracked_content_sha256"],
               "shards": shards, "counts": {status: counter[status] for status in STATUSES},
               "expected_count": len(plan["expected_raw_ids"]), "actual_started_count": sum(bool(v["phases"]) for v in results.values()),
               "all_planned_ids_reported": not counter["not_run"] and not counter["incomplete"],
               "all_selected_passed": counter["passed"] == len(plan["expected_raw_ids"]),
               "verification_integrity_passed": integrity,
               "baseline_status_changes": changes, "results": results,
               "dependency_exclusions": plan["dependency_exclusions"],
               "qualification": "software verification only; numerical D1/D2/D3 require separate evidence"}
    write(args.output / "VerificationSummaryV1.json", payload)
    print(json.dumps({k: payload[k] for k in ("counts", "expected_count", "actual_started_count", "all_planned_ids_reported", "all_selected_passed")}))


def response_requests(args):
    descriptors, receipts = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, directory in enumerate(args.batch):
        directory = directory.resolve()
        manifest = directory / "ManifestV1.json"
        manifest_copy = args.output / f"Batch{index + 1}ExternalManifestV1.json"
        manifest_copy.write_bytes(manifest.read_bytes())
        if manifest_copy.resolve().is_relative_to(directory):
            raise ValueError("external batch manifest must be outside the result bundle")
        descriptors.append({"directory": str(directory), "manifest_file": str(manifest_copy),
                            "manifest_sha256": sha(manifest_copy)})
        receipts.append(read(directory / "SourceReceiptV1.json"))
    identities = {(row["source_commit"], row["source_content_sha256"]) for row in receipts}
    if len(identities) != 1:
        raise ValueError("response batches do not share one source")
    commit, source_hash = identities.pop()
    common = {"source_commit": commit, "source_content_sha256": source_hash, "batches": descriptors}
    selections = [{"intervals": n, "case": f"D_N{n}_F0p01_T4"} for n in (64, 128, 256)]
    write(args.output / "DCRequestV1.json", {**common, "schema": "R1V5DCBaselineRequestV1", "selections": selections})
    if args.dc_bundle is not None:
        dc = args.dc_bundle.resolve()
        external = args.output / "DCExternalManifestV1.json"
        external.write_bytes((dc / "ManifestV1.json").read_bytes())
        if external.resolve().is_relative_to(dc):
            raise ValueError("external DC manifest must be outside the result bundle")
        request = {**common, "schema": "R1V5ResponseAnalysisRequestV1", "dc_bundle": {
            "directory": str(dc), "manifest_file": str(external), "manifest_sha256": sha(external)}}
        if args.methodology:
            request["methodology"] = {"file": str(args.methodology.resolve()), "sha256": sha(args.methodology)}
        write(args.output / "ResponseRequestV1.json", request)
    print(json.dumps({"source_commit": commit, "source_content_sha256": source_hash,
                      "dc_request_sha256": sha(args.output / "DCRequestV1.json"),
                      "response_request_created": args.dc_bundle is not None}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "run"):
        child = commands.add_parser(name)
        child.add_argument("--project", type=Path, required=True)
        child.add_argument("--python", required=True)
        child.add_argument("--dependency-path", action="append", required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "freeze":
            child.add_argument("--expected-commit", required=True)
            child.add_argument("--prior-plan", type=Path, required=True)
            child.add_argument("--affected", type=Path, required=True)
            child.add_argument("--extra-file", action="append", default=[])
        else:
            child.add_argument("--plan", type=Path, required=True)
            child.add_argument("--plan-sha256", required=True)
            child.add_argument("--expected-source-sha256", required=True)
            child.add_argument("--shard", action="append")
    summary = commands.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    summary.add_argument("--plan-sha256", required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--baseline-results", type=Path)
    response = commands.add_parser("response-requests")
    response.add_argument("--batch", type=Path, action="append", required=True)
    response.add_argument("--output", type=Path, required=True)
    response.add_argument("--dc-bundle", type=Path)
    response.add_argument("--methodology", type=Path)
    args = parser.parse_args()
    if hasattr(args, "project"):
        args.project = args.project.resolve()
    return {"freeze": freeze, "run": run, "summarize": summarize,
            "response-requests": response_requests}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())

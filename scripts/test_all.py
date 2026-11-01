"""Run the ordinary root CI pytest selections in separate developer processes.

The parent imports no pytest or numerical packages: pool RSS includes its
process family. CI's policy owns selection, marker and isolation decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
THREADS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
OUTCOMES = ("passed", "failed", "errors", "skipped", "xfailed")
LEGACY = "legacy suite: not part of the green claim"


def selections(root: Path) -> tuple[Any, ...]:
    import yaml

    policy = runpy.run_path(str(root / "tests/ci/policy.py"))
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    groups = policy["workflow_selections"](workflow)
    exclusions = json.loads((root / "tests/ci/exclusions.json").read_text())
    audit = policy["audit"](root, groups, exclusions)
    if not groups or audit["uncovered_files"]:
        raise ValueError("CI selections leave root tests uncovered: " + repr(audit["uncovered_files"]))
    return groups


def environment(selection: Any) -> dict[str, str]:
    result = dict(os.environ)
    # An enclosing pytest command or shell must not alter CI selection/ownership
    # or redirect imports away from the developer's active installation.
    for name in ("PYTEST_ADDOPTS", "PYTEST_CI_JOB", "PYTEST_CI_ISOLATED_GROUP", "PYTHONPATH", "PYTHONHOME"):
        result.pop(name, None)
    result.update({name: "1" for name in THREADS})
    result.update(PYTEST_CI_JOB=selection.job, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if selection.isolated_group is not None:
        result["PYTEST_CI_ISOLATED_GROUP"] = selection.isolated_group
    return result


def result_counts(path: Path) -> dict[str, int]:
    counts = dict.fromkeys(OUTCOMES, 0)
    # Count top-level cases, not testsuite.tests, which also includes subtests
    # in pytest 9. Xfails (strict marks and runtime xfails) remain visible as their own outcome.
    for case in ET.parse(path).getroot().iter("testcase"):
        skipped = case.find("skipped")
        if case.find("error") is not None:
            outcome = "errors"
        elif case.find("failure") is not None:
            outcome = "failed"
        elif skipped is not None:
            outcome = "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"
        else:
            outcome = "passed"
        counts[outcome] += 1
    return counts


def run_groups(root: Path, groups: tuple[Any, ...], output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, group in enumerate(groups, 1):
        label = group.job + ("/" + group.isolated_group if group.isolated_group else "")
        report = output / f"selection_{index}.xml"
        report.unlink(missing_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider",
                   *group.directories, "-m", group.markers, "--durations=20", "--junitxml=" + str(report)]
        for path in group.ignored_files:
            command.extend(("--ignore", path))
        print(f"\n[{index}/{len(groups)}] {label}: {shlex.join(command)}", flush=True)
        started = time.monotonic()
        row: dict[str, Any] = {"selection": asdict(group), "command": command, "junitxml": str(report)}
        try:
            result = subprocess.run(command, cwd=root, env=environment(group), check=False)
            row["exit_code"] = result.returncode
            row["counts"] = result_counts(report)
            row["status"] = "passed" if (result.returncode == 0
                and not row["counts"]["failed"] and not row["counts"]["errors"]) else "failed"
        except (OSError, ET.ParseError) as error:
            row.update(status="failed", error=str(error))
        row["seconds"] = time.monotonic() - started
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], *, error: str | None = None) -> dict[str, Any]:
    jobs: dict[str, dict[str, Any]] = {}
    for row in rows:
        job = jobs.setdefault(row["selection"]["job"], {"status": "passed", "counts": Counter(), "processes": 0})
        job["counts"].update(row.get("counts", {}))
        job["processes"] += 1
        if row["status"] != "passed":
            job["status"] = "failed"
    status = "passed" if rows and error is None and all(row["status"] == "passed" for row in rows) else "failed"
    totals: Counter[str] = Counter()
    print("\nRoot CI pytest summary", flush=True)
    for name, job in jobs.items():
        totals.update(job["counts"])
        counts = ", ".join(f"{job['counts'][key]} {key}" for key in OUTCOMES)
        print(f"{name}: {job['status'].upper()} — {counts} ({job['processes']} processes)")
    if error is not None:
        print(error)
    for row in rows:
        if row.get("error"):
            print(f"{row['selection']['job']}: {row['error']}")
    print(f"Overall: {status.upper()} — " + ", ".join(f"{totals[key]} {key}" for key in OUTCOMES))
    print(LEGACY, flush=True)
    return {"status": status, "python": sys.executable, "python_version": sys.version,
            "counts": dict(totals), "jobs": jobs, "selections": rows, "error": error, "legacy": LEGACY}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/test_all",
                        help="JUnit reports and the final summary.json (default: outputs/test_all)")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        rows = run_groups(ROOT, selections(ROOT), output)
    except (OSError, ValueError) as error:
        summary = summarize([], error=str(error))
    else:
        summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

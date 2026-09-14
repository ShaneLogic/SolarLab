#!/usr/bin/env python3
"""Verify and archive R1-0 evidence without overwriting historical artifacts."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from run_one_dimensional_mechanism_r1 import PROJECT, manifest, sha256, source_record, write_json


def camel(name):
    return "".join(part[:1].upper()+part[1:] for part in name.split("_"))


def counts(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    return {key: sum(int(s.get(key, 0)) for s in suites) for key in ["tests", "failures", "errors", "skipped"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    r0 = archive / "results/OneDimensionalMechanism/R0V1"
    original = json.loads((r0 / "DeliveryManifestV1.json").read_text())["files"]
    checked = {}
    for name, entry in original.items():
        checked[name] = sha256(r0 / name) == entry["sha256"]
    if not all(checked.values()):
        raise RuntimeError("historical R0 delivery manifest mismatch")
    restored = {
        "ReplayR0V1.py": PROJECT / "scripts/run_one_dimensional_mechanism_r0.py",
        "ExecutionContractV1.md": PROJECT / "docs/OneDimensionalMechanismR0ProtocolV1.md",
    }
    if not all(sha256(path) == original[name]["sha256"] for name, path in restored.items()):
        raise RuntimeError("restored R0 files lost byte identity")
    write_json(output / "R0SourceAuditV1.json", checked)
    source_record(output)
    write_json(output / "GitIdentityV1.json", {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=PROJECT, text=True),
        "source_manifest_sha256": sha256(output / "SourceManifestV1.json"),
        "note": "Per-run SourceManifestV1 identifies the exact execution source, including any uncommitted changes; HEAD alone is not its identity.",
    })
    runs = PROJECT / "outputs/one_dimensional_mechanism_r1"
    mapping = {}
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir() or directory == output or directory.name == "debug":
            continue
        completion = directory / "CompletionV1.json"
        if not completion.exists():
            continue
        status = json.loads(completion.read_text())
        if status["status"] == "failed":
            target = Path("FailedRuns") / camel(directory.name)
        elif directory.name == "geometry_final_v1":
            target = Path("Geometry/RunV1")
        elif directory.name == "preparation_final_v1":
            target = Path("Preparation/RunV1")
        elif "_delivery_" in directory.name:
            target = Path("EquilibriumResponse") / camel(directory.name)
        elif directory.name == "r0_regression_final_v1":
            target = Path("Validation/R0ModifiedReplayV1")
        else:
            target = Path("Validation/DevelopmentRuns") / camel(directory.name)
        if (directory / "ManifestV1.json").exists():
            for name, entry in json.loads((directory / "ManifestV1.json").read_text()).items():
                if sha256(directory / name) != entry["sha256"]:
                    raise RuntimeError(f"changed run artifact: {directory.name}/{name}")
        shutil.copytree(directory, output / target)
        mapping[directory.name] = target.as_posix()
    for directory in sorted((PROJECT / "outputs/one_dimensional_mechanism_r0").iterdir()):
        if not directory.is_dir():
            continue
        target = Path("Validation/R0ParentReplayV1") if directory.name == "run_v2" else Path("FailedRuns/R0EnvironmentV1")
        shutil.copytree(directory, output / target)
    for path in sorted(runs.glob("*.*")):
        if path.suffix in [".xml", ".log"]:
            target = output / "Validation" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write_json(output / "RunMapV1.json", mapping)
    write_json(output / "TestSummaryV1.json", {
        p.name: counts(p) for p in sorted((output / "Validation").rglob("*.xml"))
    })
    source_spec = archive / "plans/Specs/OneDimensionalMechanismR1StudySpecV1.md"
    if sha256(source_spec) != "f140505c3a7ec7c52fe38c46f36d53b876a87c2ee50512568cb62671b05c6a73":
        raise RuntimeError("R1 specification changed")
    shutil.copyfile(source_spec, output / "StudySpecV1.md")
    manifest(output)
    destination = archive / "results/OneDimensionalMechanism/R1V1"
    shutil.copytree(output, destination)
    for name, entry in json.loads((destination / "ManifestV1.json").read_text()).items():
        if sha256(destination / name) != entry["sha256"]:
            raise RuntimeError(f"archive copy mismatch: {name}")
    print(f"archived and verified: {destination}")


if __name__ == "__main__":
    main()

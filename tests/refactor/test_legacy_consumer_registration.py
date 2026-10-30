"""Retained consumers must not grow silently beyond the reviewed registry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from support.legacy_consumers import (
    Consumer,
    SCRIPT_SCOPE,
    registration_violations,
    scan_consumers,
)


ROOT = Path(__file__).resolve().parents[2]


def write(root: Path, path: str, source: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(source)


def registry(root: Path) -> dict[str, Any]:
    path = "src/solarlab_server/legacy_engine/__init__.py"
    write(root, path, "")
    return {
        "execution_facade": {"path": path, "owner": "WS4", "exit_tasks": ["P06-06"]},
        "consumer_groups": [],
    }


def p00() -> dict[str, Any]:
    return {"sources": [], "capabilities": []}


@pytest.fixture(scope="module")
def current() -> tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]]:
    policy = json.loads((ROOT / "reproducibility/RefactorTransitionBoundariesV1.json").read_text())
    previous = json.loads((ROOT / "reproducibility/RefactorCoverageV1.json").read_text())
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    lanes = yaml.safe_load(
        (ROOT / "perovskite-sim/reproducibility/NumericalRefinementRegistry.yaml").read_text()
    )["lanes"]
    return policy, previous, scan_consumers(ROOT, paths), lanes


def test_all_current_direct_consumers_have_live_registration(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
) -> None:
    policy, previous, consumers, lanes = current
    assert consumers  # An empty/broken discovery pass is not coverage.
    assert registration_violations(ROOT, policy, previous, consumers, lanes) == []


def test_current_inventory_agrees_with_the_accepted_audit(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
) -> None:
    policy, previous, consumers, lanes = current
    # Independent AcceptanceV2 / UnregisteredLegacyConsumersV1 inventory:
    # 98 direct files (excluding backend-only support), 47 lane/executor pairs.
    # Audit sha256: 31c7bbbf8a081000036333de6deb27c735441344470083679390ab9933ea0b55.
    # A deliberate inventory change needs a refreshed audit; CI needs no archive.
    inventory = {
        "consumer_paths": sorted({consumer.path for consumer in consumers}),
        "refinement_lanes": sorted((name, lane["executor"]) for name, lane in lanes.items()),
    }
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "9d6fa3dadbe97bce10c996ef63117574b93bb6c326bbbe28ef2d4c26941a697c"
    )
    assert registration_violations(ROOT, policy, previous, consumers, lanes) == []


@pytest.mark.parametrize("group_id", [
    "legacy-notebooks", "retained-research-benchmark-tools", "legacy-browser-wire",
])
def test_removing_an_accepted_group_exposes_every_affected_consumer(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
    group_id: str,
) -> None:
    policy, previous, consumers, lanes = current
    changed = deepcopy(policy)
    group = next(g for g in changed["consumer_groups"] if g["id"] == group_id)
    changed["consumer_groups"].remove(group)
    errors = registration_violations(ROOT, changed, previous, consumers, lanes)
    assert sorted(errors) == sorted(
        f"unregistered direct legacy consumer: {path}" for path in group["paths"]
    )


@pytest.mark.parametrize(("path", "source"), [
    ("src/solarlab/new.py", "import perovskite_sim.models.device as old\n"),
    ("scripts/new.py", "def later():\n    from perovskite_sim import experiments\n"),
    ("examples/new.py", "import backend.main\n"),
    ("scripts/new.py", "import importlib\nimportlib.import_module('perovskite_sim.models')\n"),
    ("scripts/new.py", "from importlib import import_module as load\nload('backend.main')\n"),
    ("scripts/new.py", "__import__('perovskite_sim')\n"),
    ("scripts/new.py", "raise AssertionError('must never execute')\nimport perovskite_sim\n"),
    ("perovskite-sim/notebooks/New.ipynb", json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# A notebook"]},
        {"cell_type": "code", "source": ["%matplotlib inline\n", "import perovskite_sim\n"]},
    ]})),
    ("web/src/new.ts", "const url = '/api/jobs';\nfetch(\n  url,\n)\n"),
    ("web/src/new.ts", "new EventSource(\n '/api/jobs/x/events'\n)\n"),
    ("web/src/new.ts", "window.fetch('/api/jv')\n"),
])
def test_new_direct_consumer_is_rejected_until_registered(
    tmp_path: Path, path: str, source: str,
) -> None:
    policy = registry(tmp_path)
    write(tmp_path, path, source)
    consumers = scan_consumers(tmp_path, [path])
    assert consumers
    assert registration_violations(tmp_path, policy, p00(), consumers) == [
        f"unregistered direct legacy consumer: {path}",
    ]
    policy["consumer_groups"].append({
        "id": "retained-new", "paths": [path], "owner": "WS4", "exit_tasks": ["P09-02"],
    })
    assert registration_violations(tmp_path, policy, p00(), consumers) == []


@pytest.mark.parametrize("path", [
    "tests/new.py", "perovskite-sim/tests/new.py",
    "perovskite-sim/perovskite_sim/new.py",
    "src/solarlab_server/legacy_engine/new.py",
    "web/src/new.test.ts", "web/src/__tests__/new.ts",
])
def test_reviewed_source_exclusions(tmp_path: Path, path: str) -> None:
    source = "fetch('/api/jobs')" if path.endswith(".ts") else "import perovskite_sim"
    write(tmp_path, path, source)
    assert scan_consumers(tmp_path, [path]) == ()


@pytest.mark.parametrize("source", [
    "/** Mirrors /api/jobs and fetch(url). */\nexport type Id = string;\n",
    "// fetch('/api/jobs')\nexport const value = 1;\n",
    "export const description = '/api/jobs';\n",  # A wire type/constant is not transport.
])
def test_browser_comments_and_types_do_not_become_execution(
    tmp_path: Path, source: str,
) -> None:
    path = "web/src/types.ts"
    write(tmp_path, path, source)
    assert scan_consumers(tmp_path, [path]) == ()


def test_backend_internal_support_is_not_a_new_library_consumer(tmp_path: Path) -> None:
    path = "perovskite-sim/backend/jobs.py"
    write(tmp_path, path, "from backend.progress import ProgressEvent\n")
    assert scan_consumers(tmp_path, [path]) == ()


def test_p00_umbrella_does_not_cover_a_new_script_or_a_notebook(tmp_path: Path) -> None:
    policy = registry(tmp_path)
    script = "scripts/old.py"
    notebook = "perovskite-sim/notebooks/New.ipynb"
    write(tmp_path, script, "import perovskite_sim\n")
    write(tmp_path, notebook, json.dumps({"cells": [
        {"cell_type": "code", "source": ["import perovskite_sim\n"]},
    ]}))
    policy["consumer_groups"].append({
        "id": "legacy-cli-and-registries", "paths": [script],
        "owner": "WS2/WS3", "exit_tasks": ["P09-03"], "inventory_scope": SCRIPT_SCOPE,
    })
    old = p00()
    old["sources"] = [{"current_path": "scripts/known.py"}, {"current_path": notebook}]
    write(tmp_path, "scripts/known.py", "import perovskite_sim\n")
    write(tmp_path, "scripts/new.py", "import perovskite_sim\n")
    consumers = scan_consumers(tmp_path, [script, notebook, "scripts/known.py", "scripts/new.py"])
    assert registration_violations(tmp_path, policy, old, consumers) == [
        f"unregistered direct legacy consumer: {notebook}",
        "unregistered direct legacy consumer: scripts/new.py",
    ]


@pytest.mark.parametrize("historical_marker", [{}, {"historical": True}, {"status": "historical"}])
def test_missing_registered_path_requires_explicit_historical_status(
    tmp_path: Path, historical_marker: dict[str, Any],
) -> None:
    policy = registry(tmp_path)
    policy["consumer_groups"].append({
        "id": "retained-old", "paths": ["scripts/gone.py"], "owner": "RS",
        "exit_tasks": ["P09-03"], **historical_marker,
    })
    errors = registration_violations(tmp_path, policy, p00(), [])
    assert errors == ([] if historical_marker else [
        "retained-old: registered path missing: scripts/gone.py",
    ])


def test_historical_row_cannot_authorize_a_live_consumer(tmp_path: Path) -> None:
    policy = registry(tmp_path)
    path = "scripts/reactivated.py"
    write(tmp_path, path, "import perovskite_sim\n")
    policy["consumer_groups"].append({
        "id": "retired", "paths": [path], "owner": "RS",
        "exit_tasks": ["P09-03"], "historical": True,
    })
    assert registration_violations(tmp_path, policy, p00(), scan_consumers(tmp_path, [path])) == [
        f"unregistered direct legacy consumer: {path}",
    ]


@pytest.mark.parametrize("field", ["owner", "exit_tasks"])
def test_an_incomplete_registration_does_not_authorize_access(tmp_path: Path, field: str) -> None:
    policy = registry(tmp_path)
    path = "scripts/new.py"
    write(tmp_path, path, "import perovskite_sim\n")
    group: dict[str, Any] = {
        "id": "incomplete", "paths": [path], "owner": "RS", "exit_tasks": ["P09-03"],
    }
    del group[field]
    policy["consumer_groups"].append(group)
    errors = registration_violations(tmp_path, policy, p00(), scan_consumers(tmp_path, [path]))
    assert errors == [
        "incomplete: missing owner or exit tasks",
        f"unregistered direct legacy consumer: {path}",
    ]


@pytest.mark.parametrize("executor", [
    "perovskite_sim.validation.new:run", "backend.main:run_jv", "other_engine.tasks:run",
])
def test_new_serialized_refinement_executor_needs_registration(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
    executor: str,
) -> None:
    policy, previous, consumers, lanes = current
    changed = {**lanes, "new-lane": {"executor": executor}}
    assert registration_violations(
        ROOT, policy, previous, consumers, changed,
    ) == [f"unregistered direct legacy refinement executor: new-lane: {executor}"]


@pytest.mark.parametrize("executor", [
    "perovskite_sim.validation.new:run", "backend.main:run_jv", "other_engine.tasks:run",
])
def test_retargeting_only_a_registered_lane_executor_needs_registration(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
    executor: str,
) -> None:
    policy, previous, consumers, lanes = current
    name = "scaps-mirror-frozen-ion-ss"
    changed = deepcopy(lanes)
    changed[name]["executor"] = executor
    assert changed == {**lanes, name: {**lanes[name], "executor": executor}}
    assert registration_violations(ROOT, policy, previous, consumers, changed) == [
        f"unregistered direct legacy refinement executor: {name}: {executor}",
    ]


def test_registered_refinement_lane_still_passes(
    current: tuple[dict[str, Any], dict[str, Any], tuple[Consumer, ...], dict[str, Any]],
) -> None:
    policy, previous, consumers, lanes = current
    name = "scaps-mirror-frozen-ion-ss"
    assert registration_violations(
        ROOT, policy, previous, consumers, {name: lanes[name]},
    ) == []


def test_malformed_notebook_python_is_not_silently_dropped(tmp_path: Path) -> None:
    path = "perovskite-sim/notebooks/New.ipynb"
    write(tmp_path, path, json.dumps({"cells": [
        {"cell_type": "code", "source": ["import perovskite_sim\n", "if (\n"]},
    ]}))
    with pytest.raises(SyntaxError):
        scan_consumers(tmp_path, [path])

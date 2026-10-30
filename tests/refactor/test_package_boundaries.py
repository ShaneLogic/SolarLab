"""P03-01 public API, future low-level layers and one legacy source tree."""

from __future__ import annotations

import ast
import hashlib
import importlib
from importlib import resources
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("solarlab", "solarlab_research", "solarlab_server")


def policy() -> dict:
    return json.loads(
        resources.files("solarlab.reproducibility").joinpath("PublicAPI.json").read_text()
    )


def declared_exports(source: str) -> list[str] | None:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            return list(ast.literal_eval(node.value))
    return None


def public_import_violations(source: str, allowed: dict) -> list[str]:
    """Check named imports/members; module objects cannot escape this check."""
    tree = ast.parse(source)
    result = []
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "solarlab" or node.module.startswith("solarlab.")
        ):
            result.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name not in allowed.get(node.module, ())
            )
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        if isinstance(node, ast.Import):
            result.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("solarlab.") and alias.name not in allowed
            )
            for alias in node.names:
                if alias.name == "solarlab" or alias.name.startswith("solarlab."):
                    bindings[alias.asname or "solarlab"] = alias.name if alias.asname else "solarlab"

    def qualified(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            base = qualified(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        name = qualified(node)
        if not name:
            continue
        if name in allowed:
            parent = parents.get(node)
            if not isinstance(parent, ast.Attribute) or parent.value is not node:
                # Passing/storing a module, including getattr/vars calls,
                # would permit untracked aliases and dynamic private access.
                result.append("module object escapes public-symbol checking: " + name)
            continue
        if isinstance(node, ast.Attribute):
            modules = [module for module in allowed if name.startswith(module + ".")]
            if not modules:
                result.append(name)
                continue
            module = max(modules, key=len)
            members = name[len(module) + 1:].split(".")
            if members[0] not in allowed[module] or any(member.startswith("_") for member in members[1:]):
                result.append(name)
    return result


def test_public_api_policy_is_a_real_package_resource():
    value = policy()
    assert value["schema"] == "solarlab.public-api-policy.v1"
    assert value["core_root_exports"] == []
    assert value["future_contracts"]["owner"] == "WS3"
    assert "solarlab.config.behavior" in value["public_imports"]


def test_ws2_declared_exports_match_the_closed_public_interface_list():
    for module, expected in policy()["declared_exports"].items():
        path = ROOT / "src" / module.replace(".", "/")
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        assert declared_exports(path.read_text()) == expected, str(path)


def test_companion_consumers_only_use_explicit_public_symbols():
    value = policy()
    violations = []
    for package in value["checked_consumers"]:
        for path in (ROOT / "src" / package).rglob("*.py"):
            violations.extend(
                f"{path.relative_to(ROOT)}: {item}"
                for item in public_import_violations(path.read_text(), value["public_imports"])
            )
    assert violations == []


@pytest.mark.parametrize("source", [
    "from solarlab.config.legacy_wire import _patch",
    "from solarlab.config.behavior import *",
    "from solarlab import _private",
    "import solarlab.config.unregistered_internal",
])
def test_unlisted_cross_package_symbols_are_rejected(source):
    assert public_import_violations(source, policy()["public_imports"])


@pytest.mark.parametrize("source", [
    "import solarlab.config.legacy_wire as wire\nwire._patch",
    "import solarlab.config.legacy_wire\nsolarlab.config.legacy_wire._patch",
    "import solarlab.config.legacy_wire as wire\nother = wire\nother._patch",
    "import solarlab.config.legacy_wire as wire\ngetattr(wire, '_patch')",
    "import solarlab.config.legacy_wire as wire\nwire.prepare_legacy_wire.__globals__",
    "from solarlab.config.legacy_wire import LegacyWire as Wire\nWire._private",
])
def test_module_aliases_cannot_bypass_the_public_symbol_boundary(source):
    assert public_import_violations(source, policy()["public_imports"])


@pytest.mark.parametrize("source", [
    "import solarlab.config.legacy_wire as wire\nwire.prepare_legacy_wire",
    "import solarlab.config.legacy_wire\nsolarlab.config.legacy_wire.prepare_legacy_wire",
    "from solarlab.config.legacy_wire import prepare_legacy_wire as prepare\nprepare",
])
def test_explicit_public_alias_access_is_supported(source):
    assert public_import_violations(source, policy()["public_imports"]) == []


def test_registered_legacy_boundaries_have_real_consumers_and_explicit_exits():
    value = json.loads(
        (ROOT / "reproducibility/RefactorTransitionBoundariesV1.json").read_text()
    )
    assert value["schema"] == "solarlab.refactor-transition-boundaries.v1"
    legacy = value["legacy_package"]
    assert (ROOT / legacy["source_root"]).is_dir()
    assert not (ROOT / legacy["excluded_alternative"]).exists()
    assert legacy["source_copies"] == 1
    facade = value["execution_facade"]
    assert facade["owner"] == "WS4"
    assert facade["module"] == "solarlab_server.legacy_engine"
    assert facade["status"] == "implemented_W03"
    module = importlib.import_module(facade["module"])
    assert (ROOT / facade["path"]).is_file()
    assert facade["symbol"] == "prepare_legacy_job"
    assert facade["symbol"] in module.__all__
    assert tuple(inspect.signature(getattr(module, facade["symbol"])).parameters) == (
        "invocation", "inputs", "execution", "engine", "manifest", "expected_sha256",
    )
    assert facade["submission"]["symbol"] == "LegacyJob.submit"
    assert "LegacyJob" in module.__all__ and callable(module.LegacyJob.submit)
    registration = facade["worker"]
    assert registration["factory"] in module.__all__
    binding = getattr(module, registration["factory"])()
    assert (binding.id, binding.module, binding.symbol) == (
        registration["id"], facade["module"], registration["symbol"],
    )
    assert binding.id == "legacy" and binding.symbol == "_run_legacy"
    assert tuple(inspect.signature(getattr(module, binding.symbol)).parameters) == ("request", "worker")
    assert binding.source_sha256 == hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
    assert registration["registry"]["document"] == {
        "drivers": ["ion_hysteresis", "scaps_steady_state", "scaps_transient"],
        "engine": "legacy", "fallback": "forbidden",
    }
    for name, digest in (("registry", binding.registry_sha256), ("request_schema", binding.schema_sha256)):
        record = registration[name]
        raw = json.dumps(record["document"], sort_keys=True, separators=(",", ":")).encode()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest() == digest
    planning = facade["historical_planning"]
    assert planning["status"] == "planned_for_M1_not_implemented_by_WS2"
    assert planning["symbol"] == "run" and planning["source_commit"]
    assert set(facade["exit_tasks"]) == {"P06-06", "P09-03"}
    for adapter in value["data_adapters"]:
        assert (ROOT / adapter["path"]).is_file()
        module = importlib.import_module(adapter["module"])
        assert adapter["symbol"] in module.__all__
        assert callable(getattr(module, adapter["symbol"]))
        assert adapter["exit_tasks"]
    for group in value["consumer_groups"]:
        if group.get("historical") is True or group.get("status") == "historical":
            continue
        assert group["owner"] and group["exit_tasks"]
        assert all((ROOT / path).is_file() for path in group["paths"])
    # Registration is deliberately not an execution or deletion certificate.
    assert value["consumer_inventory"]["discovery_record"] == "discovery_evidence"


def test_registered_legacy_submission_passes_the_same_request_and_manifest():
    from unittest.mock import Mock

    value = json.loads((ROOT / "reproducibility/RefactorTransitionBoundariesV1.json").read_text())
    facade = value["execution_facade"]
    module = importlib.import_module(facade["module"])
    request, manifest, limits, handle = (object() for _ in range(4))
    supervisor = Mock(spec=["submit"])
    supervisor.submit.return_value = handle
    job = module.LegacyJob(request, manifest)
    assert job.submit(supervisor, "registered-run", limits) is handle
    supervisor.submit.assert_called_once_with(
        "registered-run", facade["worker"]["id"], request, limits, execution=manifest,
    )


def create_package(root: Path, name: str) -> Path:
    path = root
    for part in name.split("."):
        path /= part
        path.mkdir(exist_ok=True)
        (path / "__init__.py").touch(exist_ok=True)
    return path


@pytest.mark.parametrize(("source", "forbidden", "indirect"), [
    ("solarlab.physics", "solarlab.equations", False),
    ("solarlab.physics", "solarlab.equations", True),
    ("solarlab.solvers", "solarlab.device", False),
    ("solarlab.solvers", "solarlab.device", True),
    ("solarlab.solvers", "solarlab.experiments.two_dimensional", False),
    ("solarlab.mesh", "solarlab.experiments.two_dimensional", False),
    ("solarlab.mesh", "solarlab.solvers", False),
    ("solarlab.contracts", "solarlab.physics", False),
    ("solarlab.contracts", "solarlab.device", True),
    ("solarlab.contracts", "solarlab_server", False),
])
def test_actual_linter_rejects_current_and_future_upward_dependencies(
    tmp_path, source, forbidden, indirect,
):
    for package in PACKAGES:
        shutil.copytree(ROOT / "src" / package, tmp_path / package)
    source_path = create_package(tmp_path, source)
    create_package(tmp_path, forbidden)
    if indirect:
        (tmp_path / "solarlab/boundary_probe.py").write_text(f"import {forbidden}\n")
        code = "from solarlab import boundary_probe\n"
    else:
        code = f"import {forbidden}\n"
    (source_path / "p03_probe.py").write_text(code)
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [str(Path(sys.executable).with_name("lint-imports")), "--config",
         str(ROOT / "pyproject.toml"), "--no-cache"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "broken" in result.stdout.lower(), result.stdout + result.stderr

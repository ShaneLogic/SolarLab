"""Solver imports stay below experiment orchestration in a fresh process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROBE = """
import importlib
import json
from pathlib import Path
import pkgutil
import sys

import solarlab

expected = Path(sys.argv[1]).resolve()
actual = Path(solarlab.__file__).resolve()
assert actual == expected, (actual, expected)

import solarlab.solvers as solvers

if len(sys.argv) > 2:
    solvers.__path__.append(sys.argv[2])

modules = []
for module in pkgutil.walk_packages(solvers.__path__, solvers.__name__ + "."):
    importlib.import_module(module.name)
    modules.append(module.name)

forbidden = sorted(name for name in sys.modules if name.startswith("solarlab.experiments"))
print(json.dumps({"solarlab": str(actual), "modules": modules, "forbidden": forbidden}))
assert not forbidden, "solver imports loaded experiment modules: " + repr(forbidden)
"""


def run_guard(extra_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-B", "-c", PROBE, str(ROOT / "src/solarlab/__init__.py")]
    if extra_path is not None:
        command.append(str(extra_path))
    # Keep the caller's interpreter and installation binding; -I discards
    # environment-based checkout bindings used by ordinary CI.
    return subprocess.run(command, cwd=ROOT, env=dict(os.environ), text=True,
                          capture_output=True, timeout=60, check=False)


def test_solver_imports_do_not_load_experiments():
    result = run_guard()
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout)
    assert document["modules"]
    assert document["forbidden"] == []


def test_solver_import_guard_detects_injected_experiment_import(tmp_path):
    (tmp_path / "boundary_probe.py").write_text("import solarlab.experiments\n")
    result = run_guard(tmp_path)
    assert result.returncode != 0
    document = json.loads(result.stdout)
    assert "solarlab.solvers.boundary_probe" in document["modules"]
    assert "solarlab.experiments" in document["forbidden"]
    assert "solver imports loaded experiment modules" in result.stderr


def test_reproducibility_exports_are_lazy_and_keep_identity():
    probe = """
import importlib
from pathlib import Path
import sys

import solarlab
import solarlab.reproducibility as package

assert Path(solarlab.__file__).resolve() == Path(sys.argv[1]).resolve()
assert "solarlab.reproducibility.registry" not in sys.modules
assert "solarlab.reproducibility.sources" not in sys.modules
assert not any(name.startswith("solarlab.experiments") for name in sys.modules)
assert set(package.__all__) <= set(dir(package))
providers = {
    "MigrationRegistry": "registry",
    "load_registry": "registry",
    "validate_registry_pair": "registry",
    "SourceRef": "sources",
    "SourceSet": "sources",
    "read_sources": "sources",
}
assert package.__all__ == list(providers)
for name, provider in providers.items():
    value = getattr(package, name)
    assert value is getattr(importlib.import_module(package.__name__ + "." + provider), name)
    assert package.__dict__[name] is value
namespace = {}
exec("from solarlab.reproducibility import *", namespace)
assert all(namespace[name] is getattr(package, name) for name in providers)
assert not hasattr(package, "missing_export")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, str(ROOT / "src/solarlab/__init__.py")],
        cwd=ROOT, env=dict(os.environ), text=True, capture_output=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

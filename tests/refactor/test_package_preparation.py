"""P03 package boundaries; production numerical migration is a separate gate."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("solarlab", "solarlab_research", "solarlab_server")


def test_transition_lock_preserves_every_existing_dependency_record():
    old = tomllib.loads((ROOT / "perovskite-sim/uv.lock").read_text())
    new = tomllib.loads((ROOT / "uv.lock").read_text())
    old_packages = [p for p in old["package"] if p["name"] != "perovskite-sim"]
    new_packages = [p for p in new["package"] if p["name"] != "solarlab"]
    assert new_packages == old_packages
    assert new["requires-python"] == old["requires-python"]
    assert new["resolution-markers"] == old["resolution-markers"]
    old_metadata = tomllib.loads((ROOT / "perovskite-sim/pyproject.toml").read_text())
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert metadata["project"]["dependencies"] == [
        *old_metadata["project"]["dependencies"], "pydantic==2.13.5",
    ]
    assert metadata["project"]["optional-dependencies"] == old_metadata["project"]["optional-dependencies"]
    assert "scripts" not in metadata["project"]


def test_only_the_declared_packages_are_selected_for_the_transition_wheel():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert set(wheel["only-include"]) == {
        *(f"src/{name}" for name in PACKAGES),
        "perovskite-sim/perovskite_sim",
    }
    assert set(wheel["sources"]) == {"src", "perovskite-sim"}
    assert not (ROOT / "src/perovskite_sim").exists()
    for name in PACKAGES:
        assert (ROOT / "src" / name / "py.typed").is_file()


def test_actual_core_source_import_rejects_companion_and_legacy_dependencies(tmp_path):
    script = """
import importlib.abc
import sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'perovskite_sim', 'backend', 'solarlab_server', 'solarlab_research'}:
            raise AssertionError('core crossed package boundary: '+fullname)
sys.meta_path.insert(0, Boundary())
sys.path.insert(0, sys.argv[1])
import solarlab
assert solarlab.__all__ == ()
assert not any(name.startswith(('perovskite_sim', 'backend', 'solarlab_server', 'solarlab_research')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_import_contracts(package_root: Path, cwd: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    executable = Path(sys.executable).parent / "lint-imports"
    return subprocess.run(
        [str(executable), "--config", str(ROOT / "pyproject.toml"), "--no-cache"],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
    )


def test_import_linter_accepts_actual_package_sources(tmp_path):
    result = run_import_contracts(ROOT / "src", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("source", "forbidden", "indirect"),
    [
        ("solarlab", "solarlab_server", False),
        ("solarlab", "solarlab_research", False),
        ("solarlab", "perovskite_sim", False),
        ("solarlab", "backend", False),
        ("solarlab", "perovskite_sim", True),
        ("solarlab_research", "solarlab_server", False),
        ("solarlab_server", "solarlab_research", False),
        ("solarlab.materials", "solarlab.config", False),
        ("solarlab.materials", "solarlab.device", False),
        ("solarlab.units", "solarlab.materials", False),
    ],
)
def test_import_linter_rejects_actual_direct_and_indirect_violations(
    tmp_path, source, forbidden, indirect,
):
    for name in PACKAGES:
        shutil.copytree(ROOT / "src" / name, tmp_path / name)
    target = tmp_path.joinpath(*source.split("."))
    if target.with_suffix(".py").is_file():
        target.with_suffix(".py").write_text(f"import {forbidden}\n")
    elif indirect:
        (target / "boundary_probe.py").write_text(f"import {forbidden}\n")
        (target / "__init__.py").write_text(f"from {source} import boundary_probe\n")
    else:
        (target / "__init__.py").write_text(f"import {forbidden}\n")
    result = run_import_contracts(tmp_path, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "broken" in result.stdout.lower(), result.stdout + result.stderr

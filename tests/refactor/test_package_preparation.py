"""P03/P09 package boundaries, target wheel/sdist exclusions, and console scripts."""

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
EXPECTED_SCRIPTS = {
    "solarlab": "solarlab.cli:main",
    "solarlab-server": "solarlab.cli:server_main",
    "solarlab-cbo-scan": "solarlab_research.scripts.cbo_scan:main",
    "solarlab-r1-short-window": "solarlab_research.scripts.r1_short_window:main",
    "solarlab-r1-verify-sources": "solarlab_research.scripts.r1_verify_sources:main",
}


def test_release_lock_and_console_scripts_preserve_locked_dependencies():
    new = tomllib.loads((ROOT / "uv.lock").read_text())
    new_packages = [p for p in new["package"] if p["name"] != "solarlab"]
    added = [p for p in new_packages if p["name"] == "python-flint"]
    assert len(added) == 1 and added[0]["version"] == "0.8.0"
    legacy_lock = ROOT / "perovskite-sim/uv.lock"
    if legacy_lock.is_file():
        old = tomllib.loads(legacy_lock.read_text())
        old_packages = [p for p in old["package"] if p["name"] != "perovskite-sim"]
        assert [p for p in new_packages if p["name"] != "python-flint"] == old_packages
        assert new["requires-python"] == old["requires-python"]
        assert new["resolution-markers"] == old["resolution-markers"]

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert metadata["project"]["scripts"] == EXPECTED_SCRIPTS

    legacy_pyproject = ROOT / "perovskite-sim/pyproject.toml"
    if legacy_pyproject.is_file():
        old_metadata = tomllib.loads(legacy_pyproject.read_text())
        assert metadata["project"]["dependencies"] == [
            *old_metadata["project"]["dependencies"],
            "pydantic==2.13.5",
        ]
        previous_extras = old_metadata["project"]["optional-dependencies"]
        extras = metadata["project"]["optional-dependencies"]
        expected_dev = list(previous_extras["dev"])
        expected_dev.insert(
            expected_dev.index("httpx==0.28.1") + 1, "jsonschema==4.26.0"
        )
        assert extras == {
            **previous_extras,
            "dev": expected_dev,
            "arb": ["python-flint==0.8.0"],
        }


def test_only_first_party_packages_and_web_assets_are_selected_for_the_release_wheel():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert set(wheel["only-include"]) == {f"src/{name}" for name in PACKAGES}
    assert wheel["sources"] == ["src"]
    assert (
        wheel["hooks"]["custom"]["path"]
        == "src/solarlab/materials/wheel_hook.py"
    )
    sdist = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert not any(
        entry.startswith(("perovskite-sim", "perovskite_sim"))
        for entry in sdist["only-include"]
    )
    assert (
        sdist["hooks"]["custom"]["path"]
        == "src/solarlab/materials/wheel_hook.py"
    )
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
import solarlab.materials
import solarlab.physics.registry
import solarlab.physics.selection
import solarlab.physics.presets
import solarlab.experiments.registry
assert solarlab.__all__ == ()
assert solarlab.materials.load_packaged_default_catalog().content_sha256
assert not any(name.startswith(('perovskite_sim', 'backend', 'solarlab_server', 'solarlab_research')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_import_contracts(package_root: Path, cwd: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    executable = Path(sys.executable).parent / "lint-imports"
    return subprocess.run(
        [str(executable), "--config", str(ROOT / "pyproject.toml"), "--no-cache"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
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
        ("solarlab_research", "perovskite_sim", False),
        ("solarlab_research", "backend", False),
        ("solarlab_research", "solarlab_server", False),
        ("solarlab_server", "perovskite_sim", False),
        ("solarlab_server", "backend", False),
        ("solarlab_server", "perovskite_sim", True),
        ("solarlab_server", "solarlab_research", False),
        ("solarlab.materials", "solarlab.config", False),
        ("solarlab.materials", "solarlab.device", False),
        ("solarlab.units", "solarlab.materials", False),
        ("solarlab.materials", "solarlab.physics", False),
        ("solarlab.physics", "solarlab.config", False),
        ("solarlab.physics", "solarlab.device", False),
        ("solarlab.physics", "solarlab.experiments", True),
    ],
)
def test_import_linter_rejects_actual_direct_and_indirect_violations(
    tmp_path, source, forbidden, indirect
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

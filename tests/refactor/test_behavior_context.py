"""SCAPS optical coherence diagnostics and isolated behavior import boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from solarlab.config.behavior import (
    behavior_source_requirements,
    inspect_historical_behavior,
    inspect_scaps_optical_coherence,
)
from solarlab.materials.packaged import load_packaged_default_catalog
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def defaults():
    return load_packaged_default_catalog("device_only")


def test_import_boundary_has_no_legacy_or_scientific_solver(tmp_path):
    module_path = Path(sys.modules[inspect_historical_behavior.__module__].__file__).resolve()
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import solarlab.config.behavior as behavior
forbidden = {"perovskite_sim", "scipy", "sksundae", "flint", "backend", "solarlab_server", "solarlab_research"}
assert not {name.split(".")[0] for name in sys.modules}.intersection(forbidden)
assert Path(behavior.__file__).resolve() == Path(sys.argv[2])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(module_path.parents[2]), str(module_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_behavior_source_requirements_pin_audited_domains():
    base_reqs = behavior_source_requirements()
    r1_reqs = behavior_source_requirements(driver="r1_material")
    assert len(base_reqs) == 22 and len(r1_reqs) == 27
    with pytest.raises(ValueError, match="unknown"):
        behavior_source_requirements(driver="unknown_driver")


def test_scaps_optical_coherence_surfaces_declared_vs_effective_without_changing_physics(defaults):
    path = ROOT / "src/solarlab/materials/data/presets/scaps_mirror_v2.yaml"
    raw_source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    rows = inspect_scaps_optical_coherence(raw_source, defaults=defaults)
    assert len(rows) == 4
    glass = rows[0]
    assert glass == {
        "layer": "layer_0",
        "name": "glass",
        "role": "substrate",
        "parameterization": "scaps",
        "declared_present": True,
        "declared": True,
        "normalized_configured": True,
        "effective": False,
        "historical_loader_value": False,
        "ignored_by_legacy_loader": True,
        "prospective_fix_owner": "P08-04",
    }
    for item in rows[1:]:
        assert item["declared_present"] is False and item["declared"] is None
        assert item["normalized_configured"] is False and item["effective"] is False
        assert item["ignored_by_legacy_loader"] is False and item["prospective_fix_owner"] is None

    parsed = yaml.safe_load(raw_source.content)
    parsed["layers"][0]["incoherent"] = False
    explicit_false = SourceDocument("explicit_false.yaml", yaml.safe_dump(parsed, sort_keys=False).encode())
    false_glass = inspect_scaps_optical_coherence(explicit_false, defaults=defaults)[0]
    assert false_glass["declared_present"] is True and false_glass["declared"] is False
    assert false_glass["effective"] is False and false_glass["ignored_by_legacy_loader"] is False
    assert false_glass["prospective_fix_owner"] is None

    for invalid in ("yes", 1):
        parsed["layers"][0]["incoherent"] = invalid
        bad_source = SourceDocument("bad_incoherent.yaml", yaml.safe_dump(parsed, sort_keys=False).encode())
        with pytest.raises(ValueError, match=r"layers\[0\]\.incoherent: expected boolean"):
            inspect_scaps_optical_coherence(bad_source, defaults=defaults)

"""Current preset inventory and loading checks, without historical lane claims."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from backend.main import get_config, list_configs, stack_from_dict
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.reproducibility import semantic_sha256, sha256_file
from perovskite_sim.scaps_compat import load_scaps_yaml


ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "scaps_mirror_v2.yaml": ("fast", load_scaps_yaml),
    "calado2016_fig1f.yaml": ("legacy", load_device_from_yaml),
}


def test_only_two_bundled_presets_remain():
    paths = {
        path.relative_to(ROOT / "configs").as_posix()
        for pattern in ("*.yaml", "*.yml")
        for path in (ROOT / "configs").rglob(pattern)
        if "user" not in path.relative_to(ROOT / "configs").parts[:-1]
    }
    assert paths == set(PRESETS)


def test_api_lists_only_current_bundled_presets():
    entries = list_configs()["configs"]
    shipped = [entry for entry in entries if entry["namespace"] == "shipped"]
    assert {entry["name"] for entry in shipped} == set(PRESETS)
    assert len(shipped) == len(PRESETS)


@pytest.mark.parametrize("name", PRESETS)
def test_retained_preset_bytes_and_solver_semantics_are_unchanged(name):
    with (ROOT / "reproducibility/config_benchmark_matrix.yaml").open() as stream:
        historical = yaml.safe_load(stream)
    entry = next(item for item in historical["configs"] if item["path"] == f"configs/{name}")
    path = ROOT / "configs" / name
    mode, loader = PRESETS[name]
    stack = loader(str(path))
    assert sha256_file(path) == entry["sha256"]
    assert semantic_sha256(stack) == entry["semantic_sha256"]
    assert stack.mode == mode


@pytest.mark.parametrize("name", PRESETS)
def test_current_preset_api_round_trip_preserves_mode_and_semantics(name):
    payload = get_config(name)
    assert payload["status"] == "ok"
    mode, loader = PRESETS[name]
    assert payload["config"]["device"]["mode"] == mode
    original = loader(str(ROOT / "configs" / name))
    rebuilt = stack_from_dict(payload["config"])
    assert semantic_sha256(rebuilt) == semantic_sha256(original)


def test_research_ion_populations_remain_distinct():
    scaps = load_scaps_yaml(str(ROOT / "configs/scaps_mirror_v2.yaml"))
    assert all(layer.params.D_ion == 0.0 and layer.params.P0 == 0.0 for layer in scaps.layers)
    calado = load_device_from_yaml(str(ROOT / "configs/calado2016_fig1f.yaml"))
    absorber = next(layer.params for layer in calado.layers if layer.role == "absorber")
    assert absorber.D_ion == pytest.approx(2.585e-18, rel=1e-12, abs=0.0)
    assert absorber.P0 == pytest.approx(1e25)


@pytest.mark.parametrize("name", [
    "ionmonger_benchmark.yaml",
    "scaps_mirror_v2_robin_strong.yaml",
    "driftfusion_calado2016_repro.yaml",
    "nip_MAPbI3_singleGB.yaml",
    "tandem_lin2019.yaml",
])
def test_deleted_presets_are_not_available_through_api(name):
    with pytest.raises(HTTPException) as caught:
        get_config(name)
    assert caught.value.status_code == 404

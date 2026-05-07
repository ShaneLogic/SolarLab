"""Integration tests for the ``configs/twod/`` discovery patch (Phase 6
acceptance-test fix).

The Phase 6 shipped 2D presets (``nip_MAPbI3_uniform.yaml``,
``nip_MAPbI3_singleGB.yaml``, ``bcx_combined_demo.yaml``) live under
``configs/twod/``. Pre-fix, ``GET /api/configs`` and ``GET /api/configs/{name}``
only scanned the top-level ``configs/`` and ``configs/user/`` directories,
so the workstation device-picker dropdown could not see them and the
get-config endpoint 404'd. The fix adds a third subdir to both lookups
while preserving top-level precedence on basename collision.

These tests pin:
1. listing — all three twod presets appear in /api/configs.
2. retrieval — /api/configs/{name} returns each twod preset as a dict
   with the expected device + layers shape.
3. backwards-compat — the existing top-level presets still list and
   resolve correctly.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# /api/configs listing — Phase 6 twod presets must be visible
# ---------------------------------------------------------------------------

def _config_names() -> list[str]:
    r = client.get("/api/configs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    return [c["name"] for c in body["configs"]]


def test_list_configs_includes_twod_uniform_preset():
    names = _config_names()
    assert "nip_MAPbI3_uniform.yaml" in names, (
        "Stage A 2D baseline preset (configs/twod/nip_MAPbI3_uniform.yaml) "
        "is missing from /api/configs — workstation dropdown cannot reach it."
    )


def test_list_configs_includes_twod_singleGB_preset():
    names = _config_names()
    assert "nip_MAPbI3_singleGB.yaml" in names, (
        "Stage B(a) microstructure preset (configs/twod/nip_MAPbI3_singleGB.yaml) "
        "is missing from /api/configs."
    )


def test_list_configs_includes_twod_bcx_combined_demo():
    names = _config_names()
    assert "bcx_combined_demo.yaml" in names, (
        "T7 B(c.x) combined demo (configs/twod/bcx_combined_demo.yaml) is "
        "missing from /api/configs — Phase 6 acceptance-test blocker."
    )


def test_list_configs_keeps_top_level_presets():
    """Backwards compat: every preset that appeared pre-fix must still
    appear post-fix. Spot-check the canonical top-level presets."""
    names = _config_names()
    for canonical in [
        "nip_MAPbI3.yaml",
        "nip_MAPbI3_tmm.yaml",
        "selective_contacts_demo.yaml",
        "field_mobility_demo.yaml",
    ]:
        assert canonical in names, (
            f"Top-level preset '{canonical}' disappeared after the twod-discovery "
            "patch — backwards-compat regression."
        )


def test_list_configs_twod_presets_namespace_is_shipped():
    """twod presets are tagged ``namespace='shipped'`` so the existing
    optgroup rendering surfaces them next to the top-level shipped
    presets without a frontend change. ``namespace='user'`` would land
    them in the wrong group and the dropdown would mislabel them."""
    r = client.get("/api/configs")
    by_name = {c["name"]: c for c in r.json()["configs"]}
    for twod_name in [
        "nip_MAPbI3_uniform.yaml",
        "nip_MAPbI3_singleGB.yaml",
        "bcx_combined_demo.yaml",
    ]:
        assert by_name[twod_name]["namespace"] == "shipped"


# ---------------------------------------------------------------------------
# /api/configs/{name} retrieval — twod basenames must resolve
# ---------------------------------------------------------------------------

def _get_config(name: str) -> dict:
    r = client.get(f"/api/configs/{name}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["name"] == name
    return body["config"]


def test_get_config_resolves_twod_uniform():
    cfg = _get_config("nip_MAPbI3_uniform.yaml")
    # Stage A baseline — same materials as nip_MAPbI3.yaml.
    assert cfg["device"]["V_bi"] == 1.1
    assert len(cfg["layers"]) == 3
    assert any(L["role"] == "absorber" for L in cfg["layers"])


def test_get_config_resolves_twod_singleGB():
    cfg = _get_config("nip_MAPbI3_singleGB.yaml")
    # Stage B(a) — the absorber GB block must round-trip through the loader.
    assert "microstructure" in cfg
    gbs = cfg["microstructure"]["grain_boundaries"]
    assert isinstance(gbs, list) and len(gbs) >= 1
    assert gbs[0]["layer_role"] == "absorber"


def test_get_config_resolves_twod_bcx_combined_demo():
    cfg = _get_config("bcx_combined_demo.yaml")
    # T7 — Robin S + per-layer μ(E) + microstructure all in one preset.
    dev = cfg["device"]
    assert dev["S_p_left"] == 1.0e3
    assert dev["S_n_right"] == 1.0e3
    abs_layer = next(L for L in cfg["layers"] if L["role"] == "absorber")
    assert abs_layer["v_sat_n"] == 1.0e5
    assert abs_layer["ct_beta_n"] == 2.0
    assert "microstructure" in cfg


def test_get_config_resolves_top_level_preset():
    """Top-level resolution is unchanged — pin a canonical case."""
    cfg = _get_config("nip_MAPbI3.yaml")
    assert cfg["device"]["V_bi"] == 1.1
    assert len(cfg["layers"]) == 3


def test_get_config_404_on_missing_basename():
    r = client.get("/api/configs/does_not_exist_anywhere.yaml")
    assert r.status_code == 404

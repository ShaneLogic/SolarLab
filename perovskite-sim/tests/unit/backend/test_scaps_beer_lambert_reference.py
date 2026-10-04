import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as bm
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.validation.scaps_beer_lambert import reference_catalog


def test_reference_covers_all_workbook_rows_and_keeps_missing_values():
    catalog = reference_catalog()
    assert len(catalog["sweeps"]) == 11
    points = [p for s in catalog["sweeps"].values() for p in s["points"]]
    assert sum(p.get("origin") != "solarlab_extension" for p in points) == 93
    assert len(points) == 95
    assert catalog["source_sha256"] == hashlib.sha256(Path(catalog["source"]).read_bytes()).hexdigest()
    assert catalog["sweeps"]["CHI_ETL"]["points"][-1]["reference"]["V_oc"] is None


@pytest.mark.parametrize("value", [1e19, 1e20])
def test_pvk_extension_is_loadable_and_has_no_fabricated_scaps_reference(value):
    catalog = reference_catalog()
    point = next(p for p in catalog["sweeps"]["Nd_PVK"]["points"] if p["x"] == value)
    assert point["origin"] == "solarlab_extension"
    assert point["row"] is None
    assert all(v is None for v in point["reference"].values())
    response = TestClient(bm.app).get("/api/reference-studies/scaps-beer-lambert", params={"sweep": "Nd_PVK", "value": value})
    assert response.status_code == 200, response.text
    config = response.json()["config"]
    absorber = next(layer for layer in config["layers"] if layer["role"] == "absorber")
    assert absorber["N_D"] == pytest.approx(value * 1e6)
    assert config["simulation_hints"]["reference_point"]["scope"] == "high_doping_extrapolation"
    assert config["simulation_hints"]["jv_sweep"]["N_grid"] == 90
    assert config["simulation_hints"]["min_N_grid"] == 90
    from perovskite_sim.discretization.grid import require_thick_layer_interface_resolution
    stack = bm.stack_from_dict(config)
    require_thick_layer_interface_resolution(build_electrical_grid(stack, 90), stack, N_grid=90)


def test_pvk_extension_still_rejects_values_beyond_the_declared_interval():
    assert TestClient(bm.app).get("/api/reference-studies/scaps-beer-lambert", params={"sweep": "Nd_PVK", "value": 1e21}).status_code == 422


@pytest.mark.parametrize("sheet", list(reference_catalog()["sweeps"]))
def test_each_family_loads_through_http_and_roundtrips_generation(sheet):
    entry = reference_catalog()["sweeps"][sheet]
    response = TestClient(bm.app).get("/api/reference-studies/scaps-beer-lambert", params={"sweep": sheet, "value": entry["points"][0]["x"]})
    assert response.status_code == 200, response.text
    config = response.json()["config"]
    assert config["device"]["optical_model"] == "beer_lambert_spectral"
    assert config["simulation_hints"]["jv_sweep"] == {"N_grid": 30, "n_points": 40, "v_rate": 5.0, "V_max": 1.6}
    stack = bm.stack_from_dict(config)
    rebuilt = bm.stack_from_dict(bm._stack_to_config_dict(stack))
    x = build_electrical_grid(stack, 30)
    np.testing.assert_array_equal(build_material_arrays(x, stack).G_optical, build_material_arrays(x, rebuilt).G_optical)


@pytest.mark.parametrize("params", [{"sweep": "typo", "value": 0}, {"sweep": "CHI_ETL"}, {"sweep": "CHI_ETL", "value": 10}, {"sweep": "CHI_ETL", "value": "nan"}])
def test_bad_reference_requests_fail_closed(params):
    assert TestClient(bm.app).get("/api/reference-studies/scaps-beer-lambert", params=params).status_code == 422


def test_preset_keeps_optics_and_standard_scan_hints():
    config = bm._config_dict_from_path("configs/scaps_mirror_v2.yaml")
    assert config["simulation_hints"]["reference_study"] == "scaps_beer_lambert_v1"
    assert config["device"]["mode"] == "fast"
    assert config["device"]["optical_model"] == "beer_lambert_spectral"
    summary = bm._describe_active_physics(bm.stack_from_dict(config))
    assert "Beer-Lambert" in summary
    assert "TMM" not in summary and "photon recycling" not in summary

"""Internal report replay through the same POST/SSE route as the frontend."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as bm

CASES = json.loads((Path(__file__).parents[2] / "fixtures/ScapsBeerLambertReplayV1.json").read_text())["cases"]


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[row["sweep"] for row in CASES])
def test_reference_point_replays_through_frontend_job_route(case):
    client = TestClient(bm.app)
    config = client.get("/api/reference-studies/scaps-beer-lambert", params={"sweep": case["sweep"], "value": case["x"]}).json()["config"]
    params = {**config["simulation_hints"]["jv_sweep"], "solver": "transient", "illuminated": True}
    started = client.post("/api/jobs", json={"kind": "jv", "device": config, "params": params})
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    status, result, error = bm._JOB_REGISTRY.wait(job_id, timeout=120)
    assert error is None
    assert result is not None
    assert "Beer-Lambert" in result["active_physics"]
    assert "TMM" not in result["active_physics"]
    for key in ("V_fwd", "J_fwd", "V_rev", "J_rev"):
        np.testing.assert_allclose(result[key], case["result"][key], rtol=5e-6, atol=5e-6)
    for branch in ("metrics_fwd", "metrics_rev"):
        assert result[branch]["voc_bracketed"] is True
        for key in ("V_oc", "J_sc", "FF", "PCE"):
            assert result[branch][key] == pytest.approx(case["result"][branch][key], rel=5e-6, abs=5e-6)
    stream = client.get(f"/api/jobs/{job_id}/events")
    assert "event: result" in stream.text
    assert "event: done" in stream.text

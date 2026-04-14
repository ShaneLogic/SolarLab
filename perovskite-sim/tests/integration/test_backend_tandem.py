"""Integration tests for the POST /api/tandem endpoint.

Uses FastAPI's TestClient (synchronous) so no uvicorn process is required.
The test uses the shipped tandem_lin2019.yaml which in turn references the
wide-gap and Sn-Pb sub-cell YAML presets — all relative paths are resolved
by load_tandem_from_yaml relative to the YAML file itself, so they work
regardless of CWD as long as configs/ is present on the filesystem.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# Absolute path so the test is CWD-independent when run from any directory.
_TANDEM_CONFIG = str(
    (Path(__file__).parent.parent.parent / "configs" / "tandem_lin2019.yaml").resolve()
)


def test_tandem_endpoint_returns_metrics(client: TestClient) -> None:
    """POST /api/tandem returns 200 with J-V arrays and four key metrics.

    Uses deliberately coarse grid (N_grid=20, n_points=8) to keep wall time
    under ~60 s on a laptop.  Numbers are physically meaningless at this
    resolution; we test the HTTP contract, not physical accuracy.
    """
    payload = {
        "config_path": _TANDEM_CONFIG,
        "N_grid": 20,
        "n_points": 8,
    }
    r = client.post("/api/tandem", json=payload)
    assert r.status_code == 200, f"Unexpected status {r.status_code}: {r.text}"

    data = r.json()
    assert "metrics" in data
    assert "V" in data and "J" in data
    assert len(data["V"]) == len(data["J"]) > 0
    for key in ("V_oc", "J_sc", "FF", "PCE"):
        assert key in data["metrics"], f"Missing metric: {key}"


def test_tandem_endpoint_missing_config(client: TestClient) -> None:
    """POST /api/tandem returns 404 when config_path does not exist."""
    payload = {"config_path": "/nonexistent/path/tandem.yaml"}
    r = client.post("/api/tandem", json=payload)
    assert r.status_code == 404


def test_tandem_endpoint_includes_v_top_v_bot(client: TestClient) -> None:
    """POST /api/tandem response includes per-sub-cell voltage arrays."""
    payload = {
        "config_path": _TANDEM_CONFIG,
        "N_grid": 20,
        "n_points": 8,
    }
    r = client.post("/api/tandem", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "V_top" in data and "V_bot" in data
    assert len(data["V_top"]) > 0
    assert len(data["V_bot"]) > 0

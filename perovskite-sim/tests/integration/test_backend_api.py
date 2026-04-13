"""Integration tests for thin FastAPI endpoints in ``backend.main``.

These exercise the HTTP surface with ``TestClient`` so the public contract
(status codes, JSON shape) is locked down without spinning up uvicorn.
"""

from fastapi.testclient import TestClient

from backend.main import app


def test_optical_materials_endpoint():
    """GET /api/optical-materials auto-scans data/nk/ and returns sorted names."""
    client = TestClient(app)
    resp = client.get("/api/optical-materials")
    assert resp.status_code == 200

    body = resp.json()
    assert "materials" in body

    materials = body["materials"]
    # Canonical set shipped with the repo — these must always be present.
    for expected in ("MAPbI3", "FTO", "glass"):
        assert expected in materials, f"{expected} missing from {materials}"

    # Endpoint contract: list is returned already sorted.
    assert materials == sorted(materials)

    # All 12 shipped n,k CSVs should round-trip through the scan.
    assert len(materials) >= 12

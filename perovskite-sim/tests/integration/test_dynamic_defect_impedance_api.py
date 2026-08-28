from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend


def test_real_dynamic_defect_impedance_api_serializes_protocol_and_evidence():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/impedance",
            json={
                "config_path": "dynamic_defect_ion_impedance.yaml",
                "N_grid": 8,
                "V_dc": 0.1,
                "n_freq": 19,
                "f_min": 1.0e-3,
                "f_max": 1.0e6,
                "illuminated": False,
                "method": "dynamic_defect_frequency_certified",
                "defect_energy_quadrature_order": 32,
            },
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result["Z_real"]) == 19
    assert len(result["Z_imag"]) == 19
    assert result["protocol"]["method"] == (
        "dynamic_defect_frequency_certified"
    )
    evidence = result["dynamic_defect_evidence"]
    assert evidence["capability"] == "bulk_defect_plus_ions"
    assert evidence["certified"] is True
    assert evidence["protocol"] == (
        result["protocol"]["dynamic_defect_protocol"]
    )
    assert len(evidence["protocol_sha256"]) == 64

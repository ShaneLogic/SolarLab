from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend


def test_real_dynamic_defect_transient_api_serializes_protocol_and_evidence():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/dynamic-defect-transient",
            json={
                "config_path": (
                    "dynamic_interface_defect_ion_transient_absorber_only.yaml"
                ),
                "N_grid": 4,
                "times_s": [0.0, 1.0e-8, 1.0e-6, 1.0e-4],
                "voltage_V": [0.0, 0.05, 0.05, 0.05],
                "illuminated": False,
                "method": "dynamic_defect_transient_certified",
            },
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["protocol"]["method"] == "dynamic_defect_transient_certified"
    assert result["protocol"]["capability"] == ("interface_defect_plus_positive_ions")
    assert result["protocol"]["time_step_refinement_factor"] == 1.0
    assert result["protocol"]["solver_policy"]["refinement_substeps"] == [1, 2, 4]
    assert result["evidence"]["certified"] is True
    assert result["evidence"]["reasons"] == []
    assert result["evidence"]["protocol"] == result["protocol"]
    assert len(result["evidence"]["protocol_sha256"]) == 64
    assert len(result["terminal_total_current_A_m2"]) == 4
    assert len(result["interface_occupancy"]) == 4
    assert len(result["positive_ion_centroid_shift_m"]) == 4

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# 示例J-V仿真请求体
jv_payload = {
    "config_path": "../configs/nip_MAPbI3.yaml"
}

def test_jv_api():
    resp = client.post("/api/jv", json=jv_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    result = data["result"]
    # 检查关键字段
    assert "V_fwd" in result and "J_fwd" in result
    assert "V_rev" in result and "J_rev" in result
    assert "metrics_fwd" in result and "metrics_rev" in result
    assert "hysteresis_index" in result

# 示例阻抗仿真请求体
is_payload = {
    "config_path": "../configs/nip_MAPbI3.yaml"
}

def test_impedance_api():
    resp = client.post("/api/impedance", json=is_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    result = data["result"]
    assert "frequencies" in result and "Z_real" in result and "Z_imag" in result

# 示例degradation仿真请求体
deg_payload = {
    "config_path": "../configs/nip_MAPbI3.yaml"
}

def test_degradation_api():
    resp = client.post("/api/degradation", json=deg_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    result = data["result"]
    assert "times" in result and "PCE" in result

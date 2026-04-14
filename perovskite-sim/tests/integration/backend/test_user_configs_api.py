"""Integration tests for Phase 2b backend endpoints."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestLayerTemplatesEndpoint:
    def test_returns_dict_of_templates(self) -> None:
        r = client.get("/api/layer-templates")
        assert r.status_code == 200
        body = r.json()
        assert "templates" in body
        templates = body["templates"]
        assert isinstance(templates, dict)
        for required in [
            "TiO2_ETL",
            "spiro_HTL",
            "MAPbI3_absorber",
            "glass_substrate",
            "Au_back_contact",
        ]:
            assert required in templates, f"missing template: {required}"

    def test_each_template_has_required_fields(self) -> None:
        r = client.get("/api/layer-templates")
        for name, tmpl in r.json()["templates"].items():
            assert "role" in tmpl, f"{name} missing role"
            assert "description" in tmpl, f"{name} missing description"
            assert "source" in tmpl, f"{name} missing source"
            assert "defaults" in tmpl, f"{name} missing defaults"
            assert "thickness" in tmpl["defaults"], (
                f"{name} defaults missing thickness"
            )


import pytest
import shutil
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
USER_DIR = CONFIGS_DIR / "user"


@pytest.fixture
def clean_user_dir():
    if USER_DIR.exists():
        shutil.rmtree(USER_DIR)
    yield
    if USER_DIR.exists():
        shutil.rmtree(USER_DIR)


class TestListConfigsNamespace:
    def test_returns_entries_with_namespace(self, clean_user_dir) -> None:
        r = client.get("/api/configs")
        assert r.status_code == 200
        body = r.json()
        assert "configs" in body
        configs = body["configs"]
        assert all(isinstance(c, dict) for c in configs)
        assert all("name" in c and "namespace" in c for c in configs)
        shipped = [c for c in configs if c["namespace"] == "shipped"]
        assert any(c["name"].startswith("nip_MAPbI3") for c in shipped)

    def test_includes_user_presets(self, clean_user_dir) -> None:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / "my_test.yaml").write_text("device: {V_bi: 1.0}\nlayers: []\n")
        r = client.get("/api/configs")
        configs = r.json()["configs"]
        user_entries = [c for c in configs if c["namespace"] == "user"]
        assert any(c["name"] == "my_test.yaml" for c in user_entries)

    def test_user_dir_missing_does_not_break_listing(
        self, clean_user_dir
    ) -> None:
        # USER_DIR removed by fixture; endpoint must still 200.
        r = client.get("/api/configs")
        assert r.status_code == 200


class TestPostUserConfig:
    def test_save_new_user_preset(self, clean_user_dir) -> None:
        body = {
            "name": "post_test_stack",
            "config": {"device": {"V_bi": 1.1}, "layers": []},
        }
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 200
        assert r.json()["saved"] == "post_test_stack"
        assert (USER_DIR / "post_test_stack.yaml").exists()

    def test_collision_with_shipped_returns_409(self, clean_user_dir) -> None:
        body = {"name": "nip_MAPbI3", "config": {"device": {"V_bi": 1.0}, "layers": []}}
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 409
        assert "reserved" in r.json()["detail"].lower() or "shipped" in r.json()["detail"].lower()

    def test_invalid_filename_returns_400(self, clean_user_dir) -> None:
        body = {"name": "../etc/passwd", "config": {}}
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 400

    def test_overwrite_protection(self, clean_user_dir) -> None:
        first = {"name": "dup_test", "config": {"device": {"V_bi": 1.0}, "layers": []}}
        client.post("/api/configs/user", json=first)
        # Second save without overwrite must 409.
        r = client.post("/api/configs/user", json=first)
        assert r.status_code == 409
        # With overwrite: 200.
        body = {**first, "overwrite": True}
        r2 = client.post("/api/configs/user", json=body)
        assert r2.status_code == 200

    def test_missing_name_returns_400(self, clean_user_dir) -> None:
        r = client.post("/api/configs/user", json={"config": {}})
        assert r.status_code in (400, 422)

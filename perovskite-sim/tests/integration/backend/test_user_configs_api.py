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

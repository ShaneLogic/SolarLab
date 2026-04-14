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

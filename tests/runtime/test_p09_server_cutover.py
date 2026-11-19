"""P09-02 atomic server/UI/worker cutover verification (zero legacy imports, SPA serving, historical readback)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import pytest

from solarlab.experiments.execution import JobHandle
from solarlab.io.artifacts import bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab_server.cli import enforce_server_environment
from solarlab_server.job_service import canonical, legacy_job_id

ROOT = Path(__file__).resolve().parents[2]


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "NEWCORE_EQUATIONS"}
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_server_executes_1d_transient_and_2d_and_serves_spa_with_legacy_packages_blocked(tmp_path: Path) -> None:
    """Prove solarlab_server starts, serves SPA assets, and executes 1D/2D workers with perovskite_sim blocked."""
    dist_dir = tmp_path / "_web_dist"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html><html><body><div id=\"app\">SolarLab V2</div></body></html>")
    (dist_dir / "favicon.svg").write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>")
    (dist_dir / "assets" / "app-v2.js").write_text("console.log('solarlab-v2');")

    script = """
import importlib.abc
import json
from pathlib import Path
import sys

BLOCKED = {"perovskite_sim", "backend", "solarlab_server.legacy_engine"}

class BlockLegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED or fullname.startswith(("perovskite_sim.", "backend.", "solarlab_server.legacy_engine.")):
            raise ImportError(f"forbidden legacy import in V2 server: {fullname}")
        return None

sys.meta_path.insert(0, BlockLegacyFinder())

from fastapi.testclient import TestClient
from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.io.run_store import RunStore
from solarlab_server.app import create_app
from solarlab_server.cli import build_default_server_components
from solarlab_server.job_service import JobService

tmp = Path(sys.argv[1]).resolve()
dist = Path(sys.argv[2]).resolve()
store_root = tmp / "store"
work_root = tmp / "work"

preview, catalog = build_default_server_components()
headers = {
    "X-Solarlab-Registry": catalog.document.sha256,
    "X-Solarlab-Job-Schema": catalog.schema.sha256,
}

with RunStore(store_root) as store, ProcessSupervisor(
    store,
    work_root,
    catalog.worker_bindings(),
    max_active=1,
    max_pending=3,
    shared_rss_bytes=3 * 1024**3,
    shared_output_bytes=384 * 1024**2,
    poll_seconds=0.01,
) as supervisor:
    app = create_app(
        store.root,
        configuration_preview=preview,
        job_service=JobService(catalog, supervisor),
        web_dist_dir=dist,
    )
    with TestClient(app) as http:
        # 1. SPA static serving does not shadow API routes or swallow unknown 404s.
        root_html = http.get("/")
        assert root_html.status_code == 200 and "SolarLab V2" in root_html.text
        view_html = http.get("/?view=steady-jv")
        assert view_html.status_code == 200 and "SolarLab V2" in view_html.text
        js_asset = http.get("/assets/app-v2.js")
        assert js_asset.status_code == 200 and "solarlab-v2" in js_asset.text
        fav = http.get("/favicon.svg")
        assert fav.status_code == 200 and "<svg" in fav.text
        assert http.get("/nonexistent-api-route").status_code == 404

        # 2. Catalog advertises only solarlab engine bindings with scientific_qualification=False.
        exp_resp = http.get("/experiments")
        assert exp_resp.status_code == 200
        exp_doc = exp_resp.json()
        assert "legacy_facade_sha256" not in exp_doc
        assert {b["experiment_id"] for b in exp_doc["bindings"]} == {"steady_jv", "transient_1d", "steady_2d"}
        assert all(b["engine"] == "solarlab" and b["scientific_qualification"] is False for b in exp_doc["bindings"])

        # 3. Legacy POST routes and engine="legacy" submissions are refused before enqueue.
        assert http.post("/api/jobs", json={"kind": "jv"}).status_code == 410
        assert http.post("/api/jv", json={}).status_code == 410
        bad_engine = http.post(
            "/jobs",
            headers=headers,
            json={
                "schema_version": "solarlab.job-submission.v1",
                "run_id": "refused-legacy",
                "experiment_id": "steady_jv",
                "version": 1,
                "engine": "legacy",
                "input": {
                    "device_id": "research_1d",
                    "voltage_V": [0.0, 0.05],
                    "intervals_per_layer": 6,
                    "generation_scale": 1.0,
                    "incident_power_W_m2": None,
                },
                "limits": {"wall_seconds": 30.0, "rss_bytes": 1024**3, "output_bytes": 64 * 1024**2, "accepted_steps": None, "calls": None},
            },
        )
        assert bad_engine.status_code == 422
        assert store.events() == []

        # 4. Execute 1D steady_jv, 1D transient_1d, and 2D steady_2d end-to-end.
        steady_2d_entry = next(b for b in exp_doc["bindings"] if b["experiment_id"] == "steady_2d")
        policy_2d = steady_2d_entry["preparation_context"]["fixed_numerical_policy"]
        requests = [
            (
                "cutover-steady-1d",
                "steady_jv",
                {
                    "device_id": "research_1d",
                    "voltage_V": [0.0, 0.05],
                    "intervals_per_layer": 6,
                    "generation_scale": 1.0,
                    "incident_power_W_m2": None,
                },
            ),
            (
                "cutover-transient-1d",
                "transient_1d",
                {
                    "device_id": "research_1d",
                    "times_s": [0.0, 1.0e-6, 2.0e-6],
                    "voltage_V": [0.0, 0.02, 0.02],
                    "generation_scale": [0.0, 0.0, 0.0],
                    "intervals_per_layer": 6,
                    "controls": {
                        "max_steps": 4,
                        "wall_s": 20.0,
                        "newton_budget": {
                            "max_iterations": 25,
                            "max_residual_evaluations": 200,
                            "max_jacobian_evaluations": 25,
                            "max_backtracks": 12,
                        },
                    },
                },
            ),
            (
                "cutover-steady-2d",
                "steady_2d",
                {
                    "device_id": "research_1d",
                    "voltage_V": policy_2d["voltage_V"],
                    "generation_scales": [0.0],
                    "spatial": policy_2d["spatial"],
                    "controls": policy_2d["controls"],
                    "initial": policy_2d["initial"],
                },
            ),
        ]
        outcomes = {}
        for run_id, exp_id, inp in requests:
            sub = http.post(
                "/jobs",
                headers=headers,
                json={
                    "schema_version": "solarlab.job-submission.v1",
                    "run_id": run_id,
                    "experiment_id": exp_id,
                    "version": 1,
                    "engine": "solarlab",
                    "input": inp,
                    "limits": {
                        "wall_seconds": 30.0,
                        "rss_bytes": 1536 * 1024**2,
                        "output_bytes": 96 * 1024**2,
                        "accepted_steps": None,
                        "calls": None,
                    },
                },
            )
            assert sub.status_code == 202, sub.text
            acc = sub.json()
            receipt = supervisor.wait(JobHandle(store.store_id, acc["run_id"], acc["attempt_id"]), 60).data
            assert receipt["store_state"] == "succeeded", receipt
            res_doc = http.get(acc["result_url"]).json()
            assert res_doc["state"] == "succeeded"
            assert res_doc["result"]["worker_result"]["scientific_qualification"] is False
            outcomes[exp_id] = len(res_doc["artifacts"])

assert not ({"perovskite_sim", "backend", "solarlab_server.legacy_engine"} & set(sys.modules))
print(json.dumps({"outcomes": outcomes, "legacy_loaded": False}))
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path), str(dist_dir)],
        cwd=tmp_path,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["legacy_loaded"] is False
    assert summary["outcomes"] == {"steady_jv": 9, "transient_1d": 16, "steady_2d": 14}


def test_cli_enforces_single_thread_blas_and_refuses_legacy_equation_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from solarlab_server.cli import main as cli_main

    monkeypatch.setenv("NEWCORE_EQUATIONS", "legacy")
    with pytest.raises(ValueError, match="NEWCORE_EQUATIONS='assembled'"):
        enforce_server_environment()

    monkeypatch.delenv("NEWCORE_EQUATIONS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    enforce_server_environment()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["NEWCORE_EQUATIONS"] == "assembled"

    rc = cli_main(["--store-dir", str(tmp_path / "cli-check-store"), "--no-web-dist", "--check-only"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "ok"
    assert report["engine"] == "solarlab"
    assert set(report["installed_experiments"]) == {"steady_jv", "transient_1d", "steady_2d"}


def test_solver_free_historical_runstore_readback_with_solvers_and_legacy_blocked(tmp_path: Path) -> None:
    store_dir = (tmp_path / "historical-store").resolve()
    with RunStore(store_dir) as store:
        queued = store.enqueue(
            "archived-legacy-run",
            opaque_identity={"H_input": "h1", "H_physics": "h2", "H_execution": "h3"},
            input_metadata={"origin": "historical_w03_run"},
        )
        token = store.claim(
            "archived-legacy-run",
            expected_attempt_id=queued["current_attempt_id"],
            expected_generation=0,
            owner_id="historical-writer",
        )
        raw_payload = canonical(
            {
                "schema": "solarlab.legacy-raw-result.v1",
                "driver": "ion_hysteresis",
                "result": {
                    "V_fwd": {"dtype": "<f8", "shape": [3], "raw_hex": struct.pack("<3d", 0.0, 0.5, 1.0).hex()},
                    "metrics_fwd": {
                        "V_oc": {"binary64_be": struct.pack(">d", 0.92).hex()},
                        "PCE": {"binary64_be": struct.pack(">d", 0.165).hex()},
                    },
                },
            }
        )
        art = store.publish(token, "legacy_result", bytes_artifact(raw_payload, metadata={"scientific_qualification": False}))
        store.complete(
            token,
            state="succeeded",
            result={"worker_result": {"engine": "legacy", "driver": "ion_hysteresis", "scientific_qualification": False}},
            artifacts=[art["artifact_id"]],
        )
        job_id = legacy_job_id(JobHandle(store.store_id, "archived-legacy-run", token.attempt_id))
        attempt_id = token.attempt_id

    reader_script = """
import importlib.abc
import json
from pathlib import Path
import sys

FORBIDDEN = {"perovskite_sim", "backend", "scipy", "sksundae", "flint", "solarlab_server.legacy_engine"}

class BlockSolversAndLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN or any(fullname.startswith(p + ".") for p in FORBIDDEN):
            raise ImportError(f"solver/legacy import during historical readback: {fullname}")
        return None

sys.meta_path.insert(0, BlockSolversAndLegacy())

from fastapi.testclient import TestClient
from solarlab_server.app import create_app

store_dir = Path(sys.argv[1]).resolve()
attempt_id = sys.argv[2]
job_id = sys.argv[3]

with TestClient(create_app(store_dir)) as http:
    res = http.get(f"/results/archived-legacy-run/attempts/{attempt_id}")
    assert res.status_code == 200
    assert res.json()["state"] == "succeeded"

    decoded = http.get(f"/api/jobs/{job_id}/result")
    assert decoded.status_code == 200
    body = decoded.json()
    assert body["V_fwd"] == [0.0, 0.5, 1.0]
    assert body["metrics_fwd"]["V_oc"] == 0.92
    assert body["execution"]["worker_result"]["scientific_qualification"] is False

    sse = http.get(f"/api/jobs/{job_id}/events")
    assert sse.status_code == 200
    assert "event: result" in sse.text and "event: done" in sse.text

assert not (FORBIDDEN & set(sys.modules))
print(json.dumps({"solver_free_readback": True}))
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", reader_script, str(store_dir), attempt_id, job_id],
        cwd=tmp_path,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip())["solver_free_readback"] is True


def test_dockerfile_compose_and_production_web_sources_are_cut_over() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    for text in (dockerfile, compose):
        assert "solarlab_server.cli" in text
        assert "OMP_NUM_THREADS" in text
        assert "NEWCORE_EQUATIONS" in text
        assert "Dockerfile.backend" not in text
        assert "backend.main:app" not in text

    assert not (ROOT / "src/solarlab_server/legacy_engine").exists()
    assert not (ROOT / "web/src/api.ts").exists()
    assert not (ROOT / "web/src/job-stream.ts").exists()
    import_legacy_re = re.compile(r"""from\s+['"](?:\.\.?/)*(?:api|job-stream)['"]""")
    offenders: list[str] = []
    for ts_file in sorted((ROOT / "web/src").rglob("*.ts")):
        if import_legacy_re.search(ts_file.read_text()):
            offenders.append(str(ts_file.relative_to(ROOT)))
    assert offenders == []


def test_pure_wheel_server_startup_without_examples_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise build_default_server_components() and cli_main when examples/research_1d/device.json is absent."""
    from solarlab_server.cli import build_default_server_components, main as cli_main

    assert not (tmp_path / "examples" / "research_1d" / "device.json").is_file()
    preview, catalog = build_default_server_components(repo_root=tmp_path)
    assert any(doc.id == "research_1d.yaml" for doc in preview.sources)
    assert "examples/research_1d/device.json" not in {doc.id for doc in preview.sources}
    assert {defn.id for defn in catalog.registry.definitions} == {"steady_jv", "transient_1d", "steady_2d"}

    orig_is_file = Path.is_file

    def _hide_example_json(self: Path) -> bool:
        if self.parts[-3:] == ("examples", "research_1d", "device.json"):
            return False
        return orig_is_file(self)

    monkeypatch.setattr(Path, "is_file", _hide_example_json)
    rc = cli_main(["--store-dir", str(tmp_path / "wheel-check-store"), "--no-web-dist", "--check-only"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "ok"
    assert set(report["installed_experiments"]) == {"steady_jv", "transient_1d", "steady_2d"}

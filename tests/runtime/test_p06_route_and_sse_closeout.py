"""P06-06 + P03-07 + P06-04 + P09-02 closeout: accepted YAML -> GET -> no-op editor save -> POST -> SSE -> readback."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from examples.research_1d.service import committed_implementation
from examples.research_1d.steady import load_device
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.legacy_wire import prepare_legacy_wire
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.schema import configuration_schema_representation
from solarlab.device.inputs import DeviceInput
from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.jv.worker import read_steady_jv_job, validate_steady_jv_scope
from solarlab.experiments.worker import ResourceLimits, WorkerBinding
from solarlab.io.artifacts import bytes_artifact
from solarlab.io.run_bundle import RunBundle
from solarlab.io.run_store import RunStore
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab_server.configuration import ConfigurationPreviewContext
from solarlab_server.job_service import ExecutionCatalog, JobService, JobSubmission
from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog
from test_event_service import _HTTPServer as _BaseHTTPServer, _all_events, _blocks, _next_event

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"

DEFAULT_PATHS = (
    "models/parameters.py",
    "models/device.py",
    "models/defects.py",
    "physics/statistics.py",
    "physics/band_gap_narrowing.py",
    "scaps_compat/loader.py",
    "constants.py",
    "twod/microstructure.py",
    "models/tandem_config.py",
)


class _HTTPServer(_BaseHTTPServer):
    def client(self):
        client = super().client()
        client.timeout = 30.0
        return client


def _slot_gate(request, context):
    """Non-numerical worker holding the single active slot for deterministic SSE cursor staging."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "slot_gate_ready"})
    while True:
        context.check_cancel()
        time.sleep(0.005)


@pytest.fixture(scope="module")
def legacy_defaults_and_resources():
    defaults = read_legacy_default_catalog(
        tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS)
    )
    resources = ResourceLibrary(
        tuple(
            ResourceTable(path.stem, "nk", SourceDocument(path.name, path.read_bytes()))
            for path in sorted((LEGACY / "data/nk").glob("*.csv"))
        )
    )
    return defaults, resources


@pytest.fixture(scope="module")
def standard_steady_bundle(legacy_defaults_and_resources):
    defaults, resources = legacy_defaults_and_resources
    fixture_path = ROOT / "perovskite-sim/tests/fixtures/configs/nip_MAPbI3.yaml"
    seed = SourceDocument("nip_steady.yaml", fixture_path.read_bytes())
    imported = import_standard_device(seed, id="nip_steady", defaults=defaults)
    editing = imported.editing_data()
    editing["settings"].pop("V_bi", None)
    editing["settings"]["built_in_potential_mode"] = "semiconductor_work_function"
    editing["settings"]["ion_steric_diffusion_only"] = False
    editing["settings"]["ion_steric_shared_site"] = False
    for dst_layer, src_layer in zip(editing["layers"], load_device().layers, strict=True):
        dst_layer["parameters"] = {
            k: v for k, v in dict(src_layer.parameters).items() if v is not None
        }
    exported = prepare_legacy_wire(DeviceInput.model_validate(editing), source=seed, defaults=defaults)
    assert exported.document is not None and exported.export()["strict_roundtrip_passed"] is True
    yaml_source = exported.document
    reimported = import_standard_device(yaml_source, id="nip_steady", defaults=defaults)
    prepared = resolve_device(reimported, defaults, resources, sources=(yaml_source,))
    validate_steady_jv_scope(prepared)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    steady_ctx = SteadyPreparationContext((("nip_steady", prepared),), committed_implementation(head), head)
    ceiling = ResourceLimits(60.0, 1536 * 1024**2, 96 * 1024**2, None, None)
    catalog = steady_execution_catalog(steady_ctx, ceiling=ceiling)
    preview_ctx = ConfigurationPreviewContext(defaults, resources, (yaml_source,))
    return yaml_source, prepared, preview_ctx, catalog, ceiling


def _job_headers(catalog: ExecutionCatalog) -> dict[str, str]:
    return {
        "X-Solarlab-Registry": catalog.document.sha256,
        "X-Solarlab-Job-Schema": catalog.schema.sha256,
    }


def _wire_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Solarlab-Configuration-Schema": configuration_schema_representation()[1],
    }


def test_standard_yaml_get_noop_editor_save_post_steady_job_two_sse_cursors_and_restart_readback(
    tmp_path, standard_steady_bundle
):
    yaml_source, prepared, preview_ctx, catalog, ceiling = standard_steady_bundle
    database = (tmp_path / "steady-chain-store").resolve()
    work = (tmp_path / "steady-chain-work").resolve()
    gate_binding = WorkerBinding.from_function(
        "slot_gate_fixture",
        _slot_gate,
        registry_sha256=catalog.document.sha256,
        schema_sha256=catalog.schema.sha256,
    )

    def payload(
        run_id: str = "yaml-steady-e2e",
        *,
        source_sha256: str | None = None,
        prepared_sha256: str | None = None,
    ) -> dict:
        return {
            "schema_version": "solarlab.job-submission.v1",
            "run_id": run_id,
            "experiment_id": "steady_jv",
            "version": 1,
            "engine": "solarlab",
            "input": {
                "device_id": "nip_steady",
                "voltage_V": [0.0, 0.05, 0.1],
                "intervals_per_layer": 8,
                "generation_scale": 1.0,
                "incident_power_W_m2": None,
                "expected_source_sha256": yaml_source.sha256 if source_sha256 is None else source_sha256,
                "expected_prepared_sha256": prepared.content_sha256 if prepared_sha256 is None else prepared_sha256,
            },
            "limits": {
                "wall_seconds": 60.0,
                "rss_bytes": 1536 * 1024**2,
                "output_bytes": 96 * 1024**2,
                "accepted_steps": None,
                "calls": None,
            },
        }

    with RunStore(database) as store, ProcessSupervisor(
        store,
        work,
        (catalog.worker_binding("solarlab"), gate_binding),
        max_active=1,
        max_pending=2,
        shared_rss_bytes=3 * 1024**3,
        shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=0.1,
        terminate_grace_seconds=0.1,
        poll_seconds=0.01,
    ) as supervisor:
        service = JobService(catalog, supervisor)
        with _HTTPServer(database, configuration_preview=preview_ctx, job_service=service) as server:
            with server.client() as http, server.client() as sub_a_client, server.client() as sub_b_client:
                # 1. GET /configs & GET /configs/{id}
                listing = http.get("/configs")
                assert listing.status_code == 200
                assert listing.json()["sources"] == [
                    {"id": yaml_source.id, "sha256": yaml_source.sha256, "bytes": len(yaml_source.content)}
                ]
                fetched = http.get(f"/configs/{yaml_source.id}")
                assert fetched.status_code == 200
                assert fetched.headers["etag"] == f'"{yaml_source.sha256}"'
                assert fetched.content == yaml_source.content

                # 2. Import YAML -> POST /configuration-preview/device
                imported = import_standard_device(
                    SourceDocument(yaml_source.id, fetched.content), id="nip_steady", defaults=preview_ctx.defaults
                )
                dto_wire = DeviceInput.model_validate(imported).model_dump(mode="json", exclude_unset=True)
                dev_preview = http.post("/configuration-preview/device", json=dto_wire)
                assert dev_preview.status_code == 200, dev_preview.text
                dev_json = dev_preview.json()
                assert dev_json["can_execute"] is False and dev_json["status"] == "prepared_pending_dependencies"
                preview_prepared_sha = dev_json["identity"]["content_sha256"]
                assert preview_prepared_sha == prepared.content_sha256

                # 3. No-op editor save via POST /configuration-preview/legacy-wire
                wire_req = {
                    "schema_version": "solarlab.legacy-wire-preview-request.v1",
                    "input": dev_json["input"],
                    "source": {"id": yaml_source.id, "sha256": yaml_source.sha256},
                    "references": {},
                }
                assert http.post("/configuration-preview/legacy-wire", json=wire_req).status_code == 428
                assert (
                    http.post(
                        "/configuration-preview/legacy-wire",
                        json=wire_req,
                        headers={**_wire_headers(), "X-Solarlab-Configuration-Schema": "0" * 64},
                    ).status_code
                    == 409
                )
                wire_resp = http.post("/configuration-preview/legacy-wire", json=wire_req, headers=_wire_headers())
                assert wire_resp.status_code == 200, wire_resp.text
                wire_json = wire_resp.json()
                assert wire_json["can_execute"] is False and wire_json["status"] == "prepared_data_only"
                assert wire_json["report"]["strict_roundtrip_passed"] is True
                assert wire_json["report"]["differences"] == []
                assert len(wire_json["documents"]) == 1
                saved_doc = wire_json["documents"][0]
                assert saved_doc["sha256"] == yaml_source.sha256
                assert saved_doc["utf8"].encode("utf-8") == yaml_source.content

                reimported = import_standard_device(
                    SourceDocument(saved_doc["source_id"], saved_doc["utf8"].encode("utf-8")),
                    id="nip_steady",
                    defaults=preview_ctx.defaults,
                )
                repreview = http.post(
                    "/configuration-preview/device",
                    json=DeviceInput.model_validate(reimported).model_dump(mode="json", exclude_unset=True),
                )
                assert repreview.status_code == 200
                assert repreview.json()["identity"]["content_sha256"] == preview_prepared_sha
                assert store.events() == []

                # 4. Pre-enqueue schema / registry / source / prepared SHA-256 mismatch refusals
                assert http.post("/jobs", json=payload()).status_code == 428
                for hdr in _job_headers(catalog):
                    bad_hdr = {**_job_headers(catalog), hdr: "0" * 64}
                    assert http.post("/jobs", json=payload(), headers=bad_hdr).status_code == 409
                bad_src = http.post("/jobs", json=payload(source_sha256="0" * 64), headers=_job_headers(catalog))
                assert bad_src.status_code == 422 and bad_src.json()["detail"]["code"] == "capability_or_binding_refused"
                bad_prep = http.post("/jobs", json=payload(prepared_sha256="f" * 64), headers=_job_headers(catalog))
                assert bad_prep.status_code == 422 and bad_prep.json()["detail"]["code"] == "capability_or_binding_refused"
                assert store.events() == []

                # 5. Stage slot gate so two live SSE subscribers attach before compute finishes
                spec, execution = catalog.prepare(JobSubmission.model_validate(payload()))
                gate_handle = supervisor.submit("slot-gate", gate_binding.id, spec.request, ceiling, execution=execution)
                until = time.monotonic() + 10
                while time.monotonic() < until:
                    if any(
                        e["payload"].get("ui_progress", {}).get("stage") == "slot_gate_ready"
                        for e in store.events(run_id="slot-gate", limit=100)
                    ):
                        break
                    time.sleep(0.005)
                else:
                    pytest.fail("slot_gate did not reach ready barrier")

                accepted_resp = http.post("/jobs", json=payload(), headers=_job_headers(catalog))
                assert accepted_resp.status_code == 202, accepted_resp.text
                accepted = accepted_resp.json()
                assert accepted["scientific_qualification"] is False
                dup_resp = http.post("/jobs", json=payload(), headers=_job_headers(catalog))
                assert dup_resp.status_code == 202 and dup_resp.json()["attempt_id"] == accepted["attempt_id"]

                obs_a = server.observe("sub-a")
                obs_b = server.observe("sub-b")
                with sub_a_client.stream(
                    "GET", accepted["events_url"], headers={"X-Test-Subscriber": "sub-a"}
                ) as stream_a:
                    with sub_b_client.stream(
                        "GET", accepted["events_url"], headers={"X-Test-Subscriber": "sub-b"}
                    ) as stream_b:
                        assert obs_a.started.wait(3) and obs_b.started.wait(3)
                        blocks_a, blocks_b = _blocks(stream_a), _blocks(stream_b)
                        queued_a = _next_event(blocks_a)
                        queued_b = _next_event(blocks_b)
                        assert queued_a == queued_b and queued_a["kind"] == "queued"

                        stream_a.close()
                        assert obs_a.finished.wait(3)
                        assert store.get_run(accepted["run_id"])["state"] == "queued"

                        assert supervisor.cancel(gate_handle, reason="release_slot_for_e2e")
                        assert supervisor.wait(gate_handle, 10).data["store_state"] == "cancelled"

                        remaining_b = []
                        for block in blocks_b:
                            if "data" in block:
                                rec = json.loads(block["data"])
                                assert int(block["id"]) == rec["sequence"]
                                remaining_b.append(rec)
                        all_b = [queued_b, *remaining_b]
                        assert all_b[-1]["kind"] == "terminal" and all_b[-1]["payload"]["state"] == "succeeded"
                        assert sum(e["kind"] == "terminal" for e in all_b) == 1

                with server.client() as reconnect_client:
                    with reconnect_client.stream(
                        "GET",
                        accepted["events_url"],
                        headers={"Last-Event-ID": str(queued_a["sequence"])},
                    ) as reconnected:
                        replayed_a = _all_events(reconnected)
                    assert [queued_a, *replayed_a] == all_b
                    terminal_seq = all_b[-1]["sequence"]
                    ack = reconnect_client.get(
                        accepted["events_url"], headers={"Last-Event-ID": str(terminal_seq)}
                    )
                    assert ack.status_code == 204 and ack.content == b""

                job_handle = JobHandle(store.store_id, accepted["run_id"], accepted["attempt_id"])
                receipt = supervisor.wait(job_handle, 10).data
                assert receipt["store_state"] == "succeeded" and receipt["process_joined"]
                result_json = http.get(accepted["result_url"]).json()
                assert result_json["state"] == "succeeded" and len(result_json["artifacts"]) == 9
                assert result_json["result"]["worker_result"]["scientific_qualification"] is False
                loaded = read_steady_jv_job(store, accepted["run_id"], attempt_id=accepted["attempt_id"])
                for item in result_json["artifacts"]:
                    blob = http.get(item["content_url"])
                    assert blob.status_code == 200
                    assert hashlib.sha256(blob.content).hexdigest() == item["sha256"]
                    if item["name"] == "execution_manifest":
                        assert item["sha256"] == result_json["result"]["worker_result"]["manifest_sha256"]
                    else:
                        decoded = np.load(io.BytesIO(blob.content), allow_pickle=False)
                        assert decoded.tobytes() == loaded["arrays"][item["name"]].tobytes()

    with _HTTPServer(database) as readonly_server, readonly_server.client() as reader:
        assert reader.get(accepted["result_url"]).json() == result_json
        with reader.stream("GET", accepted["events_url"]) as full_stream:
            assert _all_events(full_stream) == all_b
        with reader.stream(
            "GET", accepted["events_url"], headers={"Last-Event-ID": str(queued_a["sequence"])}
        ) as tail_stream:
            assert [queued_a, *_all_events(tail_stream)] == all_b
        assert reader.post("/jobs", json=payload(), headers=_job_headers(catalog)).status_code == 503


def test_scaps_yaml_preview_legacy_refusal_solarlab_job_and_orphan_recovery_readback(
    tmp_path, legacy_defaults_and_resources, standard_steady_bundle
):
    defaults, resources = legacy_defaults_and_resources
    yaml_source, steady_prepared, _, catalog, _ = standard_steady_bundle
    scaps_source = SourceDocument(
        "scaps-original.yaml", (ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml").read_bytes()
    )
    scaps_prepared = resolve_device(
        import_scaps_device(scaps_source, id="legacy_job", defaults=defaults),
        defaults,
        resources,
        sources=(scaps_source,),
    )
    preview_ctx = ConfigurationPreviewContext(defaults, resources, (scaps_source,))

    database = (tmp_path / "scaps-chain-store").resolve()
    work = (tmp_path / "scaps-chain-work").resolve()

    def steady_submission(
        run_id: str = "solarlab-bundle-e2e",
        *,
        engine: str = "solarlab",
        source_sha256: str | None = None,
        prepared_sha256: str | None = None,
    ) -> dict:
        return {
            "schema_version": "solarlab.job-submission.v1",
            "run_id": run_id,
            "experiment_id": "steady_jv",
            "version": 1,
            "engine": engine,
            "input": {
                "device_id": "nip_steady",
                "voltage_V": [0.0, 0.05],
                "intervals_per_layer": 6,
                "generation_scale": 1.0,
                "incident_power_W_m2": None,
                "expected_source_sha256": yaml_source.sha256 if source_sha256 is None else source_sha256,
                "expected_prepared_sha256": steady_prepared.content_sha256 if prepared_sha256 is None else prepared_sha256,
            },
            "limits": {
                "wall_seconds": 60.0,
                "rss_bytes": 1536 * 1024**2,
                "output_bytes": 96 * 1024**2,
                "accepted_steps": None,
                "calls": None,
            },
        }

    with RunStore(database) as store, ProcessSupervisor(
        store,
        work,
        (catalog.worker_binding("solarlab"),),
        max_active=1,
        max_pending=2,
        shared_rss_bytes=3 * 1024**3,
        shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=0.1,
        terminate_grace_seconds=0.1,
        poll_seconds=0.01,
    ) as supervisor:
        service = JobService(catalog, supervisor)
        with _HTTPServer(database, configuration_preview=preview_ctx, job_service=service) as server:
            with server.client() as http:
                # 1. GET /configs/{id} -> import_scaps_device -> preview -> no-op legacy-wire save
                fetched = http.get(f"/configs/{scaps_source.id}")
                assert fetched.status_code == 200 and fetched.content == scaps_source.content
                assert fetched.headers["etag"] == f'"{scaps_source.sha256}"'

                editable = import_scaps_device(
                    SourceDocument(scaps_source.id, fetched.content), id="legacy_job", defaults=defaults
                )
                dev_resp = http.post(
                    "/configuration-preview/device",
                    json=DeviceInput.model_validate(editable).model_dump(mode="json", exclude_unset=True),
                )
                assert dev_resp.status_code == 200
                dev_json = dev_resp.json()
                assert dev_json["identity"]["content_sha256"] == scaps_prepared.content_sha256

                wire_resp = http.post(
                    "/configuration-preview/legacy-wire",
                    json={
                        "schema_version": "solarlab.legacy-wire-preview-request.v1",
                        "input": dev_json["input"],
                        "source": {"id": scaps_source.id, "sha256": scaps_source.sha256},
                        "references": {},
                    },
                    headers=_wire_headers(),
                )
                assert wire_resp.status_code == 200, wire_resp.text
                wire_json = wire_resp.json()
                assert wire_json["report"]["strict_roundtrip_passed"] is True
                assert wire_json["report"]["differences"] == []
                assert wire_json["documents"][0]["sha256"] == scaps_source.sha256
                assert wire_json["documents"][0]["utf8"].encode("utf-8") == scaps_source.content

                # 2. Legacy engine & POST /api/jobs are refused before enqueue, as are SHA mismatches
                assert (
                    http.post("/jobs", json=steady_submission(engine="legacy"), headers=_job_headers(catalog)).status_code
                    == 422
                )
                assert http.post("/api/jobs", json={"kind": "jv", "device": None}).status_code == 410
                assert (
                    http.post("/jobs", json=steady_submission(source_sha256="0" * 64), headers=_job_headers(catalog)).status_code
                    == 422
                )
                assert (
                    http.post("/jobs", json=steady_submission(prepared_sha256="0" * 64), headers=_job_headers(catalog)).status_code
                    == 422
                )
                assert store.events() == []

                # 3. Valid V2 solarlab POST /jobs -> SSE -> verified readback
                sub_resp = http.post("/jobs", json=steady_submission(), headers=_job_headers(catalog))
                assert sub_resp.status_code == 202, sub_resp.text
                accepted = sub_resp.json()
                with http.stream("GET", accepted["events_url"]) as stream:
                    events = _all_events(stream)
                assert events[-1]["kind"] == "terminal" and events[-1]["payload"]["state"] == "succeeded"
                assert sum(e["kind"] == "terminal" for e in events) == 1
                assert (
                    http.get(accepted["events_url"], headers={"Last-Event-ID": str(events[-1]["sequence"])}).status_code
                    == 204
                )
                receipt = supervisor.wait(JobHandle(store.store_id, accepted["run_id"], accepted["attempt_id"]), 10).data
                assert receipt["process_joined"] and receipt["process_exit_observed"]
                result_json = http.get(accepted["result_url"]).json()
                assert result_json["state"] == "succeeded"
                assert result_json["result"]["worker_result"]["scientific_qualification"] is False

        # 4. P06-04 crash-injection + orphan-only recovery + RunBundle + missing-committed-file HTTP 409 check
        bundle_path = (tmp_path / "completed.slrun").resolve()
        bundle_pub = store.archive_run(accepted["run_id"], bundle_path)
        assert not bundle_pub["contains_active_database"]

        store.enqueue(
            "crashed-writer",
            opaque_identity={"H_input": "crash:i", "H_physics": "crash:p", "H_execution": "crash:e"},
            input_metadata={"kind": "crash_injection"},
        )
        queued = store.get_run("crashed-writer")
        crash_token = store.claim(
            "crashed-writer",
            expected_attempt_id=queued["current_attempt_id"],
            expected_generation=queued["generation"],
            owner_id="crashed-owner",
        )
        orphan_pub = store.publish(
            crash_token, "uncommitted_blob", bytes_artifact(b"orphan-published-before-db-commit", metadata={})
        )
        half_part = database / "partial" / (orphan_pub["artifact_id"] + ".part")
        half_part.write_bytes(b"half-written")
        orphan_obj = database / orphan_pub["relative_path"]
        assert orphan_obj.is_file() and half_part.is_file()

        interrupted = store.recover_interrupted()
        assert interrupted["interrupted_attempts"] == [crash_token.attempt_id]
        collected = store.recover_artifacts(cleanup=True)
        assert collected["issues"] == []
        assert orphan_pub["relative_path"] in collected["removed_paths"]
        assert f"partial/{orphan_pub['artifact_id']}.part" in collected["removed_paths"]
        assert not orphan_obj.exists() and not half_part.exists()

    with RunBundle(bundle_path, expected_sha256=bundle_pub["sha256"]) as bundle:
        assert bundle.get_run()["state"] == "succeeded"
        assert len(bundle.events()) == len(events)

    with _HTTPServer(database) as readonly_server, readonly_server.client() as reader:
        assert reader.get(accepted["result_url"]).json() == result_json
        for item in result_json["artifacts"]:
            resp = reader.get(item["content_url"])
            assert resp.status_code == 200 and hashlib.sha256(resp.content).hexdigest() == item["sha256"]

    target_artifact = result_json["artifacts"][0]
    committed_path = database / target_artifact["relative_path"]
    saved_bytes = committed_path.read_bytes()
    committed_path.unlink()
    try:
        with RunStore(database, read_only=True) as audit_store:
            audit = audit_store.recover_artifacts(cleanup=False)
            assert any(issue["artifact_id"] == target_artifact["artifact_id"] for issue in audit["issues"])
        with _HTTPServer(database) as broken_server, broken_server.client() as reader:
            res_err = reader.get(accepted["result_url"])
            assert res_err.status_code == 409 and res_err.json()["detail"]["code"] == "artifact_integrity"
            art_err = reader.get(target_artifact["content_url"])
            assert art_err.status_code == 409 and art_err.json()["detail"]["code"] == "artifact_integrity"
    finally:
        committed_path.write_bytes(saved_bytes)

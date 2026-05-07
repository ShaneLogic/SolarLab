"""Backend tests for Stage B(c.x) advanced-physics dispatch (Phase 6 frontend
wiring).

Two surfaces matter:

1. ``backend.main.stack_from_dict`` — turns a JSON device dict (the path the
   frontend uses when sending the live editor state) into a ``DeviceStack``.
   Stage B(c.1) Robin S fields and Stage B(c.2) per-layer μ(E) fields must
   round-trip; if they are silently dropped here, the frontend editor
   becomes a UI-only placebo.

2. ``POST /api/jobs`` with ``kind="jv_2d"`` and either a ``config_path`` to a
   B(c.x)-enabled preset OR an inline ``device`` JSON — must dispatch
   without error and return a job_id. This mirrors the existing
   ``test_voc_grain_sweep_api.py`` smoke pattern: we do not await the
   worker thread, only the synchronous handshake.

The unit-level round-trip test is where regressions show up first; the
HTTP smoke tests are belt-and-braces against the dispatch surface.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app, stack_from_dict


client = TestClient(app)


# ---------------------------------------------------------------------------
# unit-level: stack_from_dict round-trip for Stage B(c.x) parameters
# ---------------------------------------------------------------------------

def _baseline_layer_dict(name: str, role: str, **overrides) -> dict:
    """Minimal-valid layer schema for stack_from_dict. Numbers chosen to match
    the nip_MAPbI3 absorber / spiro_HTL / TiO2_ETL stack so the test reads
    like a real preset."""
    base = {
        "name": name,
        "role": role,
        "thickness": 400e-9,
        "eps_r": 24.1,
        "mu_n": 2e-4,
        "mu_p": 2e-4,
        "ni": 3.2e13,
        "N_D": 0.0,
        "N_A": 0.0,
        "D_ion": 1e-16,
        "P_lim": 1.6e27,
        "P0": 1.6e24,
        "tau_n": 1e-6,
        "tau_p": 1e-6,
        "n1": 3.2e13,
        "p1": 3.2e13,
        "B_rad": 5e-22,
        "C_n": 1e-42,
        "C_p": 1e-42,
        "alpha": 1.3e7,
    }
    base.update(overrides)
    return base


def test_stack_from_dict_propagates_robin_S_fields():
    """device.S_n_left/p_left/n_right/p_right must round-trip onto DeviceStack.

    Stage B(c.1) Robin / selective contacts. Without this plumbing the
    frontend's Robin panel produces a silent no-op on inline-device runs.
    """
    cfg = {
        "device": {
            "V_bi": 1.1,
            "Phi": 2.5e21,
            "mode": "full",
            "S_n_left": 1.0e-3,
            "S_p_left": 1.0e3,
            "S_n_right": 1.0e3,
            "S_p_right": 1.0e-3,
        },
        "layers": [
            _baseline_layer_dict("HTL", "HTL", thickness=200e-9),
            _baseline_layer_dict("ABS", "absorber"),
            _baseline_layer_dict("ETL", "ETL", thickness=100e-9, N_D=1e24),
        ],
    }
    stack = stack_from_dict(cfg)
    assert stack.S_n_left == 1.0e-3, "S_n_left dropped by stack_from_dict"
    assert stack.S_p_left == 1.0e3, "S_p_left dropped by stack_from_dict"
    assert stack.S_n_right == 1.0e3, "S_n_right dropped by stack_from_dict"
    assert stack.S_p_right == 1.0e-3, "S_p_right dropped by stack_from_dict"


def test_stack_from_dict_robin_S_absent_means_None():
    """Omitting S_* keys must yield None (= ohmic Dirichlet) on DeviceStack —
    the documented 'absent / disabled' sentinel that distinguishes itself
    from an explicit 0 (Neumann blocking)."""
    cfg = {
        "device": {"V_bi": 1.1, "Phi": 2.5e21, "mode": "full"},
        "layers": [
            _baseline_layer_dict("HTL", "HTL", thickness=200e-9),
            _baseline_layer_dict("ABS", "absorber"),
            _baseline_layer_dict("ETL", "ETL", thickness=100e-9, N_D=1e24),
        ],
    }
    stack = stack_from_dict(cfg)
    assert stack.S_n_left is None
    assert stack.S_p_left is None
    assert stack.S_n_right is None
    assert stack.S_p_right is None


def test_stack_from_dict_propagates_field_mobility_layer_fields():
    """Per-layer v_sat_{n,p}, ct_beta_{n,p}, pf_gamma_{n,p} must round-trip
    onto MaterialParams. Stage B(c.2) field-dependent mobility μ(E)."""
    abs_layer = _baseline_layer_dict(
        "ABS", "absorber",
        v_sat_n=1.0e5, v_sat_p=2.0e5,
        ct_beta_n=2.0, ct_beta_p=1.0,
        pf_gamma_n=1.0e-4, pf_gamma_p=3.0e-4,
    )
    cfg = {
        "device": {"V_bi": 1.1, "Phi": 2.5e21, "mode": "full"},
        "layers": [
            _baseline_layer_dict("HTL", "HTL", thickness=200e-9),
            abs_layer,
            _baseline_layer_dict("ETL", "ETL", thickness=100e-9, N_D=1e24),
        ],
    }
    stack = stack_from_dict(cfg)
    p = stack.layers[1].params  # absorber
    assert p.v_sat_n == 1.0e5
    assert p.v_sat_p == 2.0e5
    assert p.ct_beta_n == 2.0
    assert p.ct_beta_p == 1.0
    assert p.pf_gamma_n == 1.0e-4
    assert p.pf_gamma_p == 3.0e-4


def test_stack_from_dict_field_mobility_default_is_zero():
    """Layers that omit μ(E) fields must default to 0 (= disabled)."""
    cfg = {
        "device": {"V_bi": 1.1, "Phi": 2.5e21, "mode": "full"},
        "layers": [_baseline_layer_dict("ABS", "absorber")],
    }
    stack = stack_from_dict(cfg)
    p = stack.layers[0].params
    assert p.v_sat_n == 0.0
    assert p.v_sat_p == 0.0
    assert p.pf_gamma_n == 0.0
    assert p.pf_gamma_p == 0.0


# ---------------------------------------------------------------------------
# Stage B(a) — microstructure must round-trip on the inline-device path
# (mirror of load_device_from_yaml; the frontend's startJob always sends
# device: <full cfg>, never config_path:, so this is the only path that
# matters at runtime)
# ---------------------------------------------------------------------------

def test_stack_from_dict_propagates_microstructure_block():
    """Inline-device payloads carrying a top-level ``microstructure:`` block
    must produce a populated DeviceStack.microstructure. Loading the
    shipped configs/twod/nip_MAPbI3_singleGB.yaml in the workstation and
    submitting via ``device:`` is exactly this code path."""
    cfg = {
        "device": {"V_bi": 1.1, "Phi": 2.5e21, "mode": "full"},
        "layers": [
            _baseline_layer_dict("HTL", "HTL", thickness=200e-9),
            _baseline_layer_dict("ABS", "absorber"),
            _baseline_layer_dict("ETL", "ETL", thickness=100e-9, N_D=1e24),
        ],
        "microstructure": {
            "grain_boundaries": [
                {
                    "x_position": 250e-9,
                    "width": 5e-9,
                    "tau_n": 5e-8,
                    "tau_p": 5e-8,
                    "layer_role": "absorber",
                }
            ]
        },
    }
    stack = stack_from_dict(cfg)
    gbs = stack.microstructure.grain_boundaries
    assert len(gbs) == 1, "GB block silently dropped by stack_from_dict"
    gb = gbs[0]
    assert gb.x_position == 250e-9
    assert gb.width == 5e-9
    assert gb.tau_n == 5e-8
    assert gb.tau_p == 5e-8
    assert gb.layer_role == "absorber"


def test_stack_from_dict_microstructure_absent_means_empty():
    """Configs without a ``microstructure:`` key must produce an empty
    Microstructure() so Stage A lateral-uniform runs are bit-identical
    to the pre-Phase 6 path."""
    cfg = {
        "device": {"V_bi": 1.1, "Phi": 2.5e21, "mode": "full"},
        "layers": [_baseline_layer_dict("ABS", "absorber")],
    }
    stack = stack_from_dict(cfg)
    assert stack.microstructure is not None
    assert stack.microstructure.grain_boundaries == ()


def _consume_sse_until_done(client_, job_id: str, timeout: float = 600.0):
    """Read /api/jobs/{id}/events to completion. Returns the LAST
    ``result`` event payload as a dict, or raises if an ``error`` event
    fires. Mirrors the SSE consumer the live frontend uses (see
    ``frontend/src/job-stream.ts``); the difference is we read every
    line explicitly so we can fail fast on a worker crash, instead of
    a generic JS console.error.
    """
    import json as _json
    cur = None
    result = None
    error_msg = None
    with client_.stream("GET", f"/api/jobs/{job_id}/events", timeout=timeout) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            if not raw:
                cur = None
                continue
            if raw.startswith(":"):
                continue
            if raw.startswith("event:"):
                cur = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:"):
                payload = raw.split(":", 1)[1].strip()
                if cur == "result":
                    result = _json.loads(payload)
                elif cur == "error":
                    error_msg = payload
                elif cur == "done":
                    break
    if error_msg is not None:
        raise AssertionError(
            f"worker thread emitted an SSE error event before completing: "
            f"{error_msg[:400]}"
        )
    return result


def test_jv_2d_sse_result_event_carries_metrics_dict():
    """Catches the exact serializer regression that escaped Layer 2: the
    backend ``kind="jv_2d"`` worker MUST emit a final ``result`` event
    that carries a ``metrics`` dict with the JVMetrics fields. The
    pre-existing dispatch tests only assert a 200 + job_id on the POST
    handshake; a worker-thread crash inside ``_run`` would surface as
    an ``event: error`` on the SSE stream, not a non-200 HTTP code, so
    those tests cannot see it.

    Mirrors what the live frontend's ``streamJobEvents`` consumer does
    (``frontend/src/job-stream.ts:streamJobEvents``).
    """
    cfg_resp = client.get("/api/configs/nip_MAPbI3_uniform.yaml")
    assert cfg_resp.status_code == 200
    cfg = cfg_resp.json()["config"]

    post = client.post(
        "/api/jobs",
        json={
            "kind": "jv_2d",
            "device": cfg,
            "params": {
                "lateral_length": 500e-9,
                "Nx": 4,
                "Ny_per_layer": 5,
                "V_max": 0.4,    # 3 V points, fast — bracket flag will be False
                "V_step": 0.2,
                "settle_t": 1e-7,
                "illuminated": True,
                "save_snapshots": False,
                "lateral_bc": "periodic",
            },
        },
    )
    assert post.status_code == 200, post.text
    job_id = post.json()["job_id"]

    result = _consume_sse_until_done(client, job_id)
    assert result is not None, "no SSE result event emitted"

    # Layer 2 wire-through: ``metrics`` must be present as a dict and
    # carry every JVMetrics field. Bracket flag is expected False here
    # (V_max=0.4 is below V_oc) but the test does NOT assert physics —
    # it only asserts SHAPE so a serializer regression fails loudly.
    assert "metrics" in result, (
        "SSE result event is missing ``metrics`` — Layer 2 regression: "
        "either the worker thread crashed before reaching the JVMetrics "
        "serializer, or the field was renamed. Live frontend would render "
        "no metric card for any 2D run."
    )
    m = result["metrics"]
    for key in ("V_oc", "J_sc", "FF", "PCE", "voc_bracketed"):
        assert key in m, f"metrics dict is missing {key!r}: keys = {sorted(m)}"
    assert isinstance(m["voc_bracketed"], bool)


def test_jv_2d_dispatches_inline_device_with_microstructure_from_singleGB():
    """Frontend-realistic dispatch: load configs/twod/nip_MAPbI3_singleGB.yaml
    via the YAML loader, hand the resulting dict back as the ``device:``
    payload, and confirm the dispatch handshake succeeds. Mirrors the
    workstation flow: GET /api/configs/{name} → device cache → POST
    /api/jobs with device:<cfg>. The handshake passing here means
    stack_from_dict picked up the microstructure block (the worker
    thread does the heavy lifting after the 200 OK)."""
    import yaml
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cfg_path = os.path.join(repo_root, "configs", "twod", "nip_MAPbI3_singleGB.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Sanity: the bundled preset really does carry a GB block.
    assert "microstructure" in cfg
    assert cfg["microstructure"]["grain_boundaries"]

    # Stack the cfg through stack_from_dict and confirm GB survives.
    stack = stack_from_dict(cfg)
    assert len(stack.microstructure.grain_boundaries) >= 1, (
        "configs/twod/nip_MAPbI3_singleGB.yaml microstructure dropped "
        "by stack_from_dict — workstation submit would silently lose "
        "Stage B(a) physics."
    )

    resp = client.post(
        "/api/jobs",
        json={
            "kind": "jv_2d",
            "device": cfg,  # frontend always sends this
            "params": {
                "lateral_length": 500e-9,
                "Nx": 4,
                "Ny_per_layer": 5,
                "V_max": 0.2,
                "V_step": 0.2,
                "settle_t": 1e-7,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["job_id"], str)


# ---------------------------------------------------------------------------
# HTTP-level: dispatch smoke for the bcx-enabled presets
# ---------------------------------------------------------------------------

def test_jv_2d_dispatches_bcx_combined_demo():
    """POST /api/jobs with kind='jv_2d' on the new bcx_combined_demo preset
    must return 200 + job_id. The preset activates Robin S + μ(E) +
    microstructure simultaneously; this test pins that the dispatch surface
    accepts the combined-physics config, not that the worker thread
    succeeds (the worker is async and not awaited here)."""
    resp = client.post(
        "/api/jobs",
        json={
            "kind": "jv_2d",
            "config_path": "configs/twod/bcx_combined_demo.yaml",
            "params": {
                "lateral_length": 500e-9,
                "Nx": 4,
                "Ny_per_layer": 5,
                "V_max": 0.2,
                "V_step": 0.2,
                "settle_t": 1e-7,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["job_id"], str)


def test_jv_2d_dispatches_inline_device_with_bcx_fields():
    """POST /api/jobs with kind='jv_2d' and an inline device JSON carrying
    S_n_left + v_sat_n must dispatch without error. This exercises the
    same code path the frontend uses when the user has an unsaved edit
    in the device editor."""
    inline_device = {
        "device": {
            "V_bi": 1.1,
            "Phi": 2.5e21,
            "mode": "full",
            "S_n_left": 1.0e-3,
            "S_p_left": 1.0e3,
            "S_n_right": 1.0e3,
            "S_p_right": 1.0e-3,
        },
        "layers": [
            _baseline_layer_dict("HTL", "HTL", thickness=200e-9, pf_gamma_p=3.0e-4),
            _baseline_layer_dict("ABS", "absorber",
                                 v_sat_n=1.0e5, v_sat_p=1.0e5,
                                 ct_beta_n=2.0, ct_beta_p=1.0),
            _baseline_layer_dict("ETL", "ETL", thickness=100e-9, N_D=1e24),
        ],
    }
    resp = client.post(
        "/api/jobs",
        json={
            "kind": "jv_2d",
            "device": inline_device,
            "params": {
                "lateral_length": 500e-9,
                "Nx": 4,
                "Ny_per_layer": 5,
                "V_max": 0.2,
                "V_step": 0.2,
                "settle_t": 1e-7,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["job_id"], str)

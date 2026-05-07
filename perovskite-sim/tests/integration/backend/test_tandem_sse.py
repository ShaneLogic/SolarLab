"""SSE-consuming dispatch test for ``kind="tandem"`` on POST /api/jobs.

The streaming tandem ``_run`` block in ``backend/main.py`` serialises the
``run_tandem_jv`` result into the SSE ``result`` event. Pre-fix, that path
used module-qualified ``dataclasses.asdict(...)`` even though the worker
function's local namespace only had ``asdict`` from the module-top
``from dataclasses import asdict`` import — so a successful tandem run
crashed at the metrics serialiser with ``NameError: name 'dataclasses'
is not defined`` and the user saw an SSE ``error`` event instead of a
``result`` event.

The pre-existing tandem tests cover the legacy blocking endpoint
``POST /api/tandem`` (which has its own local ``import dataclasses``
inside ``run_tandem``) and the synchronous handshake on ``/api/jobs``,
but neither consumes the SSE event stream — so the worker-thread crash
on the streaming path stayed dark.

This test monkeypatches ``run_tandem_jv`` to skip the ~60 s tandem
sweep AND the stub n,k "Sub-cell J ranges do not overlap" failure
mode that ships with ``configs/tandem_lin2019.yaml``. The monkeypatched
function returns a synthetic ``TandemJVResult`` with sensible JVMetrics
so the test exercises the SSE serialiser path end-to-end without
running any real physics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app


_TANDEM_CONFIG = str(
    (Path(__file__).parent.parent.parent.parent / "configs" / "tandem_lin2019.yaml").resolve()
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _consume_sse_until_done(client_, job_id: str, timeout: float = 60.0):
    """Read /api/jobs/{id}/events to completion. Returns the LAST
    ``result`` event payload as a dict, or raises if an ``error`` event
    fires. Mirror of the helper in ``test_jv_2d_advanced_physics.py``;
    intentionally duplicated here so the tandem test file is self-
    contained and not coupled to the 2D test module."""
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
            f"tandem worker emitted an SSE error event before completing: "
            f"{error_msg[:400]}"
        )
    return result


def _build_fake_tandem_result():
    """Synthetic ``TandemJVResult`` with valid-looking JVMetrics. Avoids
    the ~60 s real sweep and the stub-n,k overlap failure that ships
    with the reference tandem_lin2019.yaml."""
    from perovskite_sim.experiments.jv_sweep import JVMetrics, JVResult
    from perovskite_sim.experiments.tandem_jv import TandemJVResult

    V = np.array([0.0, 0.5, 1.0, 1.5])
    J = np.array([200.0, 100.0, 0.0, -100.0])
    tandem_metrics = JVMetrics(
        V_oc=1.0, J_sc=200.0, FF=0.7, PCE=0.14, voc_bracketed=True,
    )
    sub_metrics = JVMetrics(
        V_oc=0.7, J_sc=200.0, FF=0.7, PCE=0.10, voc_bracketed=True,
    )
    sub_jv = JVResult(
        V_fwd=V, J_fwd=J,
        V_rev=V[::-1].copy(), J_rev=J[::-1].copy(),
        metrics_fwd=sub_metrics, metrics_rev=sub_metrics,
        hysteresis_index=0.0,
    )
    return TandemJVResult(
        V=V, J=J,
        V_top=V * 0.6, V_bot=V * 0.4,
        metrics=tandem_metrics,
        top_result=sub_jv, bot_result=sub_jv,
    )


def test_tandem_sse_result_event_carries_metrics_dict(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker thread must complete the streaming tandem dispatch and
    emit a final ``result`` event with the ``metrics`` dict populated.
    Catches the same regression class as the 2D fix in f29b190."""
    fake = _build_fake_tandem_result()

    # Patch the source module attribute. ``backend/main.py`` imports
    # ``run_tandem_jv`` lazily inside the ``elif kind == "tandem"``
    # block, so the lookup happens after this monkeypatch lands.
    import perovskite_sim.experiments.tandem_jv as _tandem_mod
    monkeypatch.setattr(_tandem_mod, "run_tandem_jv", lambda *a, **kw: fake)

    post = client.post(
        "/api/jobs",
        json={
            "kind": "tandem",
            "config_path": _TANDEM_CONFIG,
            "params": {"N_grid": 8, "n_points": 4},
        },
    )
    assert post.status_code == 200, post.text
    job_id = post.json()["job_id"]

    result = _consume_sse_until_done(client, job_id)
    assert result is not None, "no SSE result event emitted"

    # Layer 2 / regression-fix wire-through: ``metrics`` dict must be
    # present and carry every JVMetrics field. The shape — not the
    # numbers — is what we assert; values come from _build_fake_tandem_result.
    assert "metrics" in result, (
        "SSE result event is missing ``metrics`` — tandem serialiser "
        "regression: worker thread crashed before reaching the metrics "
        "dict, or the field was renamed."
    )
    m = result["metrics"]
    for key in ("V_oc", "J_sc", "FF", "PCE", "voc_bracketed"):
        assert key in m, f"metrics dict is missing {key!r}: keys = {sorted(m)}"
    assert isinstance(m["voc_bracketed"], bool)

    # Result envelope must also carry the rest of the tandem-specific
    # payload — V, J, V_top, V_bot — to confirm the serialiser as a
    # whole still works after the dataclasses-vs-asdict fix.
    for key in ("V", "J", "V_top", "V_bot", "benchmark"):
        assert key in result, (
            f"tandem result envelope missing {key!r}: keys = {sorted(result)}"
        )

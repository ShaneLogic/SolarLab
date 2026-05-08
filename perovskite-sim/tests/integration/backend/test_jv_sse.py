"""SSE-consuming dispatch test for ``kind="jv"`` on POST /api/jobs.

Pins the ``to_serializable`` bool-vs-int ordering: Python ``bool``
subclasses ``int``, so before the bool branch was added in
``backend/main.py:to_serializable`` the ``isinstance(obj, int)``
branch silently coerced ``True``/``False`` to ``1``/``0`` and the
frontend's strict-equality picker (``=== true``) failed to match,
producing missing 1D J-V publication-mode annotations + autoranged
y/x even when the sweep brackets V_oc fine.

The 2D and tandem dispatches use raw ``asdict(result.metrics)`` and
were never affected — this gap motivated the regression test for
the 1D path specifically.

The fixture monkeypatches ``run_jv_sweep`` to skip the real ~10 s
1D Radau sweep and return a synthetic ``JVResult`` with valid
``JVMetrics(voc_bracketed=True)`` so the test exercises the SSE
serializer path end-to-end without running any physics.
"""
from __future__ import annotations

import json as _json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app


_PRESET_PATH = str(
    (Path(__file__).parent.parent.parent.parent / "configs" / "nip_MAPbI3.yaml").resolve()
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _consume_sse_until_done(client_, job_id: str, timeout: float = 30.0):
    """Read /api/jobs/{id}/events to completion and return the parsed
    final ``result`` event dict, or raise on an SSE ``error`` event.
    Mirrors the helper in ``test_tandem_sse.py`` / ``test_jv_2d_advanced_physics.py``;
    intentionally inlined to keep this test file self-contained."""
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
            f"jv worker emitted an SSE error event before completing: {error_msg[:400]}"
        )
    return result


def _build_fake_jv_result():
    """Synthetic ``JVResult`` with ``voc_bracketed=True``. The bool
    is what the test cares about — the V/J arrays are placeholder."""
    from perovskite_sim.experiments.jv_sweep import JVMetrics, JVResult

    V = np.array([0.0, 0.4, 0.8, 1.0])
    J_fwd = np.array([+400.0, +200.0, +50.0, -1500.0])
    J_rev = J_fwd[::-1].copy()
    m = JVMetrics(
        V_oc=0.91, J_sc=400.0, FF=0.79, PCE=0.29, voc_bracketed=True,
    )
    return JVResult(
        V_fwd=V, J_fwd=J_fwd,
        V_rev=V[::-1].copy(), J_rev=J_rev,
        metrics_fwd=m, metrics_rev=m,
        hysteresis_index=0.0,
    )


def test_jv_sse_voc_bracketed_is_json_boolean_not_integer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``voc_bracketed`` MUST round-trip through the SSE serializer
    as a JSON boolean, not as integer ``1``/``0``. Pre-fix the
    ``to_serializable`` int branch caught Python ``bool`` (because
    ``bool`` subclasses ``int``) and silently coerced the field,
    breaking the frontend's ``=== true`` strict-equality check."""
    fake = _build_fake_jv_result()

    # Patch the function the dispatch closure looks up. The 1D dispatch
    # imports ``jv_sweep`` at module load and resolves
    # ``jv_sweep.run_jv_sweep`` at call time, so a setattr on the source
    # module is the correct hook.
    import perovskite_sim.experiments.jv_sweep as _jv_mod
    monkeypatch.setattr(_jv_mod, "run_jv_sweep", lambda *a, **kw: fake)

    post = client.post(
        "/api/jobs",
        json={
            "kind": "jv",
            "config_path": _PRESET_PATH,
            "params": {"N_grid": 8, "n_points": 4, "v_rate": 1.0, "V_max": 1.0, "illuminated": True},
        },
    )
    assert post.status_code == 200, post.text
    job_id = post.json()["job_id"]

    result = _consume_sse_until_done(client, job_id)
    assert result is not None, "no SSE result event emitted"

    # Both metric blocks must be present and carry voc_bracketed.
    for key in ("metrics_fwd", "metrics_rev"):
        assert key in result, f"SSE result missing {key!r}: keys = {sorted(result)}"
        m = result[key]
        assert "voc_bracketed" in m, (
            f"{key} missing 'voc_bracketed': fields = {sorted(m)}"
        )
        # The strict invariant — must be Python ``bool``, NOT
        # an ``int`` masquerading as one. ``isinstance(1, bool)`` is
        # False in Python, so this assertion fires on the pre-fix
        # behaviour where the int branch coerced True → 1.
        assert isinstance(m["voc_bracketed"], bool), (
            f"{key}.voc_bracketed should be a JSON boolean (Python bool), "
            f"got {type(m['voc_bracketed']).__name__}={m['voc_bracketed']!r}. "
            f"This is the to_serializable bool-vs-int ordering trap — "
            f"insert the bool branch before the int branch in "
            f"backend/main.py:to_serializable."
        )
        # And it should be ``True`` for our bracketed fixture.
        assert m["voc_bracketed"] is True

    # Hysteresis index — also a float in the JVResult dataclass — must
    # survive the to_serializable path without coercion.
    assert "hysteresis_index" in result
    assert isinstance(result["hysteresis_index"], float)

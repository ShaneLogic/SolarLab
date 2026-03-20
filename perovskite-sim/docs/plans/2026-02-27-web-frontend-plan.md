# Web Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a React + FastAPI web UI so users can edit device parameters, submit simulations, and view interactive J-V / Impedance / Degradation charts.

**Architecture:** `backend/` (FastAPI + Celery + Redis) and `frontend/` (React 18 + Vite + Plotly.js) live alongside the existing `perovskite_sim/` package. A `docker-compose.yml` at the repo root orchestrates all services. The Celery worker imports `perovskite_sim` directly — no changes to the simulator.

**Tech Stack:** Python 3.11, FastAPI ≥0.110, Celery ≥5.3, Redis 7, Pydantic v2, React 18, TypeScript, Vite 5, react-plotly.js, Tailwind CSS 3, Axios, Vitest, React Testing Library, Docker Compose v2.

---

## Reference: Existing Simulator API

The Celery tasks will call these functions from `perovskite_sim/`:

```python
# Build a DeviceStack from the request dict:
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams

# Run J-V sweep:
from perovskite_sim.experiments.jv_sweep import run_jv_sweep  # → JVResult

# Run impedance:
from perovskite_sim.experiments.impedance import run_impedance  # → ImpedanceResult

# Run degradation:
from perovskite_sim.experiments.degradation import run_degradation  # → DegradationResult
```

`JVResult` has: `V_fwd, J_fwd, V_rev, J_rev` (np arrays), `metrics_fwd, metrics_rev` (JVMetrics with V_oc, J_sc, FF, PCE), `hysteresis_index`.

---

## Task 1: Backend Scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py` (empty)
- Create: `backend/app/api/__init__.py` (empty)
- Create: `backend/app/tasks/__init__.py` (empty)
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/tests/conftest.py`

**Step 1: Create directory tree**

```bash
mkdir -p backend/app/api backend/app/tasks backend/tests
touch backend/app/__init__.py backend/app/api/__init__.py \
      backend/app/tasks/__init__.py backend/tests/__init__.py
```

**Step 2: Write `backend/requirements.txt`**

```
fastapi>=0.110
uvicorn[standard]>=0.27
celery[redis]>=5.3
redis>=5.0
pydantic>=2.5
httpx>=0.26
pytest>=8.0
pytest-asyncio>=0.23
numpy>=1.26
scipy>=1.12
pyyaml>=6.0
matplotlib>=3.8
```

**Step 3: Write `backend/tests/conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)
```

**Step 4: Install backend dependencies**

```bash
cd backend
pip install -r requirements.txt
pip install -e ..   # installs perovskite_sim from repo root
```

Expected: no errors.

**Step 5: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend directory structure"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas import LayerIn, SimulateRequest, DeviceIn, JVParams

NIP_LAYER_HTL = dict(
    name="spiro_HTL", role="HTL", thickness=200e-9,
    eps_r=3.0, mu_n=1e-10, mu_p=1e-8, ni=1.0,
    N_A=2e23, N_D=0.0, D_ion=0.0, P_lim=1e30, P0=0.0,
    tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
    B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0,
)
NIP_LAYER_ABS = dict(
    name="MAPbI3", role="absorber", thickness=400e-9,
    eps_r=24.1, mu_n=2e-4, mu_p=2e-4, ni=3.2e13,
    N_A=0.0, N_D=0.0, D_ion=1e-16, P_lim=1.6e27, P0=1.6e24,
    tau_n=1e-6, tau_p=1e-6, n1=3.2e13, p1=3.2e13,
    B_rad=5e-22, C_n=1e-42, C_p=1e-42, alpha=1.3e7,
)
NIP_LAYER_ETL = dict(
    name="TiO2_ETL", role="ETL", thickness=100e-9,
    eps_r=10.0, mu_n=1e-7, mu_p=1e-10, ni=1.0,
    N_A=0.0, N_D=1e24, D_ion=0.0, P_lim=1e30, P0=0.0,
    tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
    B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0,
)

def test_layer_in_valid():
    layer = LayerIn(**NIP_LAYER_HTL)
    assert layer.thickness == 200e-9
    assert layer.role == "HTL"

def test_simulate_request_valid():
    req = SimulateRequest(
        device=DeviceIn(V_bi=1.1, Phi=2.5e21),
        layers=[NIP_LAYER_HTL, NIP_LAYER_ABS, NIP_LAYER_ETL],
        sim_type="jv",
        sim_params={"N_grid": 60, "n_points": 20, "v_rate": 5.0},
    )
    assert req.sim_type == "jv"
    assert len(req.layers) == 3

def test_layer_negative_thickness_invalid():
    bad = {**NIP_LAYER_HTL, "thickness": -1e-7}
    with pytest.raises(ValidationError):
        LayerIn(**bad)

def test_layer_invalid_role():
    bad = {**NIP_LAYER_HTL, "role": "BUFFER"}
    with pytest.raises(ValidationError):
        LayerIn(**bad)
```

**Step 2: Run to verify it fails**

```bash
cd backend
python -m pytest tests/test_schemas.py -v
```

Expected: `ImportError: cannot import name 'LayerIn' from 'app.schemas'`

**Step 3: Write `backend/app/schemas.py`**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class LayerIn(BaseModel):
    name: str
    role: Literal["HTL", "absorber", "ETL"]
    thickness: float = Field(gt=0)
    eps_r: float = Field(gt=0)
    mu_n: float = Field(gt=0)
    mu_p: float = Field(gt=0)
    ni: float = Field(gt=0)
    N_A: float = Field(ge=0)
    N_D: float = Field(ge=0)
    D_ion: float = Field(ge=0)
    P_lim: float = Field(gt=0)
    P0: float = Field(ge=0)
    tau_n: float = Field(gt=0)
    tau_p: float = Field(gt=0)
    n1: float = Field(gt=0)
    p1: float = Field(gt=0)
    B_rad: float = Field(ge=0)
    C_n: float = Field(ge=0)
    C_p: float = Field(ge=0)
    alpha: float = Field(ge=0)


class DeviceIn(BaseModel):
    V_bi: float = Field(gt=0)
    Phi: float = Field(gt=0)


class SimulateRequest(BaseModel):
    device: DeviceIn
    layers: list[LayerIn] = Field(min_length=1, max_length=10)
    sim_type: Literal["jv", "impedance", "degradation"]
    sim_params: dict = Field(default_factory=dict)


class JVMetricsOut(BaseModel):
    V_oc: float
    J_sc: float
    FF: float
    PCE: float


class JVResultOut(BaseModel):
    V_fwd: list[float]
    J_fwd: list[float]
    V_rev: list[float]
    J_rev: list[float]
    metrics_fwd: JVMetricsOut
    metrics_rev: JVMetricsOut
    hysteresis_index: float


class ImpedanceResultOut(BaseModel):
    frequencies: list[float]
    Z_real: list[float]
    Z_imag: list[float]


class DegradationResultOut(BaseModel):
    times: list[float]
    PCE: list[float]
    V_oc: list[float]
    J_sc: list[float]


class JobSubmitted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: Literal["pending", "running", "done", "failed"]
    result: dict | None = None
    error: str | None = None
```

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/test_schemas.py -v
```

Expected: 4 passed.

**Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_schemas.py
git commit -m "feat: add Pydantic schemas for simulation API"
```

---

## Task 3: FastAPI App + Presets Endpoint

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/simulate.py`
- Create: `backend/tests/test_api_presets.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_api_presets.py
def test_get_presets(client):
    response = client.get("/presets")
    assert response.status_code == 200
    presets = response.json()
    assert "nip_MAPbI3" in presets
    assert "pin_MAPbI3" in presets

def test_get_preset_nip(client):
    response = client.get("/presets/nip_MAPbI3")
    assert response.status_code == 200
    data = response.json()
    assert data["device"]["V_bi"] == 1.1
    assert len(data["layers"]) == 3
    roles = [l["role"] for l in data["layers"]]
    assert "HTL" in roles
    assert "absorber" in roles
    assert "ETL" in roles

def test_get_preset_not_found(client):
    response = client.get("/presets/nonexistent_device")
    assert response.status_code == 404
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_api_presets.py -v
```

Expected: `ImportError` — `app.main` doesn't exist yet.

**Step 3: Write `backend/app/api/simulate.py`** (presets only for now)

```python
from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from perovskite_sim.models.config_loader import load_device_from_yaml

router = APIRouter()

CONFIGS_DIR = Path(os.environ.get(
    "CONFIGS_DIR",
    Path(__file__).parent.parent.parent.parent / "configs"
))


def _available_presets() -> list[str]:
    return [p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml"))]


def _load_preset(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    stack = load_device_from_yaml(str(path))
    layers = []
    for lyr in stack.layers:
        p = lyr.params
        layers.append({
            "name": lyr.name, "role": lyr.role, "thickness": lyr.thickness,
            "eps_r": p.eps_r, "mu_n": p.mu_n, "mu_p": p.mu_p,
            "ni": p.ni, "N_A": p.N_A, "N_D": p.N_D,
            "D_ion": p.D_ion, "P_lim": p.P_lim, "P0": p.P0,
            "tau_n": p.tau_n, "tau_p": p.tau_p,
            "n1": p.n1, "p1": p.p1,
            "B_rad": p.B_rad, "C_n": p.C_n, "C_p": p.C_p,
            "alpha": p.alpha,
        })
    return {"device": {"V_bi": stack.V_bi, "Phi": stack.Phi}, "layers": layers}


@router.get("/presets")
def list_presets() -> list[str]:
    return _available_presets()


@router.get("/presets/{name}")
def get_preset(name: str) -> dict:
    return _load_preset(name)
```

**Step 4: Write `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.simulate import router

app = FastAPI(title="Perovskite Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

**Step 5: Run to verify tests pass**

```bash
python -m pytest tests/test_api_presets.py -v
```

Expected: 3 passed.

**Step 6: Commit**

```bash
git add backend/app/main.py backend/app/api/simulate.py backend/tests/test_api_presets.py
git commit -m "feat: add FastAPI app with presets endpoints"
```

---

## Task 4: Celery Worker + Simulation Tasks

**Files:**
- Create: `backend/celery_worker.py`
- Create: `backend/app/tasks/run_sim.py`
- Create: `backend/tests/test_tasks.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_tasks.py
import pytest
from app.tasks.run_sim import _build_device_stack, _run_jv

NIP_REQUEST = {
    "device": {"V_bi": 1.1, "Phi": 2.5e21},
    "layers": [
        dict(name="spiro_HTL", role="HTL", thickness=200e-9,
             eps_r=3.0, mu_n=1e-10, mu_p=1e-8, ni=1.0,
             N_A=2e23, N_D=0.0, D_ion=0.0, P_lim=1e30, P0=0.0,
             tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
             B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0),
        dict(name="MAPbI3", role="absorber", thickness=400e-9,
             eps_r=24.1, mu_n=2e-4, mu_p=2e-4, ni=3.2e13,
             N_A=0.0, N_D=0.0, D_ion=1e-16, P_lim=1.6e27, P0=1.6e24,
             tau_n=1e-6, tau_p=1e-6, n1=3.2e13, p1=3.2e13,
             B_rad=5e-22, C_n=1e-42, C_p=1e-42, alpha=1.3e7),
        dict(name="TiO2_ETL", role="ETL", thickness=100e-9,
             eps_r=10.0, mu_n=1e-7, mu_p=1e-10, ni=1.0,
             N_A=0.0, N_D=1e24, D_ion=0.0, P_lim=1e30, P0=0.0,
             tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
             B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0),
    ],
    "sim_params": {"N_grid": 30, "n_points": 3, "v_rate": 20.0},
}


def test_build_device_stack():
    stack = _build_device_stack(NIP_REQUEST)
    assert len(stack.layers) == 3
    assert stack.V_bi == 1.1
    assert stack.layers[0].role == "HTL"


def test_run_jv_returns_expected_keys():
    result = _run_jv(NIP_REQUEST)
    assert "V_fwd" in result
    assert "J_fwd" in result
    assert "metrics_fwd" in result
    assert len(result["V_fwd"]) == 3
    assert result["metrics_fwd"]["J_sc"] >= 0
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_tasks.py -v
```

Expected: `ImportError: cannot import name '_build_device_stack'`

**Step 3: Write `backend/celery_worker.py`**

```python
from celery import Celery
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "perovskite_sim",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# Register tasks
celery_app.autodiscover_tasks(["app.tasks"])
```

**Step 4: Write `backend/app/tasks/run_sim.py`**

```python
from __future__ import annotations
import sys
from pathlib import Path

# Ensure perovskite_sim is importable when running as a Celery worker
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from celery_worker import celery_app
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.impedance import run_impedance
from perovskite_sim.experiments.degradation import run_degradation


def _build_device_stack(req: dict) -> DeviceStack:
    layers = []
    for l in req["layers"]:
        params = MaterialParams(
            eps_r=l["eps_r"], mu_n=l["mu_n"], mu_p=l["mu_p"],
            D_ion=l["D_ion"], P_lim=l["P_lim"], P0=l["P0"],
            ni=l["ni"], tau_n=l["tau_n"], tau_p=l["tau_p"],
            n1=l["n1"], p1=l["p1"], B_rad=l["B_rad"],
            C_n=l["C_n"], C_p=l["C_p"], alpha=l["alpha"],
            N_A=l["N_A"], N_D=l["N_D"],
        )
        layers.append(LayerSpec(
            name=l["name"], thickness=l["thickness"],
            params=params, role=l["role"],
        ))
    return DeviceStack(
        layers=layers,
        V_bi=req["device"]["V_bi"],
        Phi=req["device"]["Phi"],
    )


def _run_jv(req: dict) -> dict:
    stack = _build_device_stack(req)
    p = req.get("sim_params", {})
    result = run_jv_sweep(
        stack,
        N_grid=int(p.get("N_grid", 60)),
        n_points=int(p.get("n_points", 20)),
        v_rate=float(p.get("v_rate", 5.0)),
    )
    return {
        "V_fwd": result.V_fwd.tolist(),
        "J_fwd": result.J_fwd.tolist(),
        "V_rev": result.V_rev.tolist(),
        "J_rev": result.J_rev.tolist(),
        "metrics_fwd": {
            "V_oc": result.metrics_fwd.V_oc,
            "J_sc": result.metrics_fwd.J_sc,
            "FF": result.metrics_fwd.FF,
            "PCE": result.metrics_fwd.PCE,
        },
        "metrics_rev": {
            "V_oc": result.metrics_rev.V_oc,
            "J_sc": result.metrics_rev.J_sc,
            "FF": result.metrics_rev.FF,
            "PCE": result.metrics_rev.PCE,
        },
        "hysteresis_index": result.hysteresis_index,
    }


def _run_impedance(req: dict) -> dict:
    stack = _build_device_stack(req)
    p = req.get("sim_params", {})
    freqs = np.logspace(
        np.log10(float(p.get("freq_min", 1.0))),
        np.log10(float(p.get("freq_max", 1e6))),
        int(p.get("n_freq", 10)),
    )
    result = run_impedance(
        stack, frequencies=freqs,
        V_dc=float(p.get("V_dc", 0.9)),
        N_grid=int(p.get("N_grid", 40)),
    )
    return {
        "frequencies": result.frequencies.tolist(),
        "Z_real": result.Z.real.tolist(),
        "Z_imag": result.Z.imag.tolist(),
    }


def _run_degradation(req: dict) -> dict:
    stack = _build_device_stack(req)
    p = req.get("sim_params", {})
    result = run_degradation(
        stack,
        t_end=float(p.get("t_end", 3600.0)),
        n_snapshots=int(p.get("n_snapshots", 10)),
        N_grid=int(p.get("N_grid", 40)),
        store_ion_profiles=False,
    )
    return {
        "times": (result.t / 3600.0).tolist(),   # convert s → hours
        "PCE": result.PCE.tolist(),
        "V_oc": result.V_oc.tolist(),
        "J_sc": result.J_sc.tolist(),
    }


_RUNNERS = {
    "jv": _run_jv,
    "impedance": _run_impedance,
    "degradation": _run_degradation,
}


@celery_app.task(name="run_simulation", bind=True)
def run_simulation(self, request_dict: dict) -> dict:
    sim_type = request_dict.get("sim_type", "jv")
    runner = _RUNNERS.get(sim_type)
    if runner is None:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    return runner(request_dict)
```

**Step 5: Run to verify tests pass**

```bash
python -m pytest tests/test_tasks.py -v
```

Expected: 2 passed. (Note: `test_run_jv_returns_expected_keys` will take ~10–30s — it runs a real simulation with coarse grid.)

**Step 6: Commit**

```bash
git add backend/celery_worker.py backend/app/tasks/run_sim.py backend/tests/test_tasks.py
git commit -m "feat: add Celery worker and simulation tasks"
```

---

## Task 5: POST /simulate + GET /job/{id} Endpoints

**Files:**
- Modify: `backend/app/api/simulate.py`
- Create: `backend/tests/test_api_simulate.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_api_simulate.py
from unittest.mock import patch, MagicMock

VALID_REQUEST = {
    "device": {"V_bi": 1.1, "Phi": 2.5e21},
    "layers": [
        dict(name="spiro_HTL", role="HTL", thickness=200e-9,
             eps_r=3.0, mu_n=1e-10, mu_p=1e-8, ni=1.0,
             N_A=2e23, N_D=0.0, D_ion=0.0, P_lim=1e30, P0=0.0,
             tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
             B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0),
        dict(name="MAPbI3", role="absorber", thickness=400e-9,
             eps_r=24.1, mu_n=2e-4, mu_p=2e-4, ni=3.2e13,
             N_A=0.0, N_D=0.0, D_ion=1e-16, P_lim=1.6e27, P0=1.6e24,
             tau_n=1e-6, tau_p=1e-6, n1=3.2e13, p1=3.2e13,
             B_rad=5e-22, C_n=1e-42, C_p=1e-42, alpha=1.3e7),
        dict(name="TiO2_ETL", role="ETL", thickness=100e-9,
             eps_r=10.0, mu_n=1e-7, mu_p=1e-10, ni=1.0,
             N_A=0.0, N_D=1e24, D_ion=0.0, P_lim=1e30, P0=0.0,
             tau_n=1e-9, tau_p=1e-9, n1=1.0, p1=1.0,
             B_rad=1e-30, C_n=1e-42, C_p=1e-42, alpha=0.0),
    ],
    "sim_type": "jv",
    "sim_params": {"N_grid": 60, "n_points": 20, "v_rate": 5.0},
}


def test_post_simulate_returns_job_id(client):
    mock_task = MagicMock()
    mock_task.id = "test-job-abc"
    with patch("app.api.simulate.run_simulation") as mock_run:
        mock_run.delay.return_value = mock_task
        response = client.post("/simulate", json=VALID_REQUEST)
    assert response.status_code == 202
    assert response.json()["job_id"] == "test-job-abc"


def test_post_simulate_invalid_role(client):
    bad = dict(VALID_REQUEST)
    bad["layers"] = [{**VALID_REQUEST["layers"][0], "role": "BUFFER"},
                     *VALID_REQUEST["layers"][1:]]
    response = client.post("/simulate", json=bad)
    assert response.status_code == 422


def test_get_job_pending(client):
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    with patch("app.api.simulate.AsyncResult", return_value=mock_result):
        response = client.get("/job/some-job-id")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_get_job_done(client):
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {"V_fwd": [0.0, 1.1], "J_fwd": [200.0, 0.0]}
    with patch("app.api.simulate.AsyncResult", return_value=mock_result):
        response = client.get("/job/some-job-id")
    assert response.json()["status"] == "done"
    assert "V_fwd" in response.json()["result"]


def test_get_job_failed(client):
    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.result = Exception("solver diverged")
    with patch("app.api.simulate.AsyncResult", return_value=mock_result):
        response = client.get("/job/some-job-id")
    data = response.json()
    assert data["status"] == "failed"
    assert "solver diverged" in data["error"]
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_api_simulate.py -v
```

Expected: `ImportError` — `run_simulation` and `AsyncResult` not in simulate.py.

**Step 3: Extend `backend/app/api/simulate.py`** — add the two new endpoints

Append to the existing `simulate.py` (after the presets routes):

```python
import uuid
from fastapi import Response
from celery.result import AsyncResult
from app.schemas import SimulateRequest, JobSubmitted, JobStatus
from app.tasks.run_sim import run_simulation
from celery_worker import celery_app


@router.post("/simulate", status_code=202)
def submit_simulation(req: SimulateRequest, response: Response) -> JobSubmitted:
    task = run_simulation.delay(req.model_dump())
    return JobSubmitted(job_id=task.id)


_STATE_MAP = {
    "PENDING": "pending",
    "RECEIVED": "pending",
    "STARTED": "running",
    "RETRY": "running",
    "SUCCESS": "done",
    "FAILURE": "failed",
    "REVOKED": "failed",
}


@router.get("/job/{job_id}")
def get_job_status(job_id: str) -> JobStatus:
    ar = AsyncResult(job_id, app=celery_app)
    status = _STATE_MAP.get(ar.state, "pending")
    if status == "done":
        return JobStatus(status="done", result=ar.result)
    if status == "failed":
        return JobStatus(status="failed", error=str(ar.result))
    return JobStatus(status=status)
```

**Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_api_simulate.py -v
```

Expected: 5 passed.

**Step 5: Run the full backend test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (schemas + presets + simulate + tasks).

**Step 6: Commit**

```bash
git add backend/app/api/simulate.py backend/tests/test_api_simulate.py
git commit -m "feat: add POST /simulate and GET /job/{id} endpoints"
```

---

## Task 6: Backend Dockerfile

**Files:**
- Create: `backend/Dockerfile`

**Step 1: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Build context is the repo root (docker-compose sets context: .)
WORKDIR /workspace

# Copy entire repo (needed for perovskite_sim package + configs)
COPY . .

# Install perovskite_sim + backend deps
RUN pip install --no-cache-dir -e . -r backend/requirements.txt

WORKDIR /workspace/backend

ENV CONFIGS_DIR=/workspace/configs
```

**Step 2: Verify it builds**

```bash
# From repo root:
docker build -f backend/Dockerfile -t perovskite-backend .
```

Expected: build succeeds, no errors.

**Step 3: Smoke-test the API image**

```bash
docker run --rm -p 8000:8000 \
  -e REDIS_URL=redis://localhost:6379/0 \
  perovskite-backend \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
# In another terminal:
curl http://localhost:8000/presets
# Expected: ["nip_MAPbI3","pin_MAPbI3"]
```

**Step 4: Commit**

```bash
git add backend/Dockerfile
git commit -m "feat: add backend Dockerfile"
```

---

## Task 7: Frontend Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/test/setup.ts`

**Step 1: Scaffold with Vite**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
```

**Step 2: Install additional deps**

```bash
npm install axios react-plotly.js plotly.js
npm install -D tailwindcss postcss autoprefixer @types/react-plotly.js \
              @testing-library/react @testing-library/user-event \
              @testing-library/jest-dom vitest jsdom @vitejs/plugin-react
npx tailwindcss init -p
```

**Step 3: Write `frontend/tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

**Step 4: Write `frontend/src/index.css`** (replace default)

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 5: Write `frontend/vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
})
```

**Step 6: Write `frontend/src/test/setup.ts`**

```typescript
import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Plotly doesn't work in jsdom — replace with a stub
vi.mock('react-plotly.js', () => ({
  default: (props: Record<string, unknown>) => {
    const { createElement } = require('react')
    return createElement('div', {
      'data-testid': 'plotly-chart',
      'data-traces': JSON.stringify(props.data ?? []),
    })
  },
}))
```

**Step 7: Write a smoke test**

```typescript
// frontend/src/App.test.tsx
import { render, screen } from '@testing-library/react'
import App from './App'

test('renders app header', () => {
  render(<App />)
  expect(screen.getByText(/Perovskite Simulator/i)).toBeInTheDocument()
})
```

**Step 8: Write minimal `frontend/src/App.tsx`**

```tsx
export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-indigo-700 text-white px-6 py-4">
        <h1 className="text-xl font-bold">Perovskite Simulator</h1>
      </header>
      <main className="p-6">
        <p className="text-gray-500">Loading...</p>
      </main>
    </div>
  )
}
```

**Step 9: Run the test**

```bash
npx vitest run
```

Expected: 1 passed.

**Step 10: Commit**

```bash
cd ..   # back to repo root
git add frontend/
git commit -m "feat: scaffold React + Vite + Tailwind + Vitest frontend"
```

---

## Task 8: API Client

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/client.test.ts`

**Step 1: Write the failing test**

```typescript
// frontend/src/api/client.test.ts
import { vi, describe, it, expect, beforeEach } from 'vitest'
import axios from 'axios'
import { postSimulate, pollJob, getPresets, getPreset } from './client'

vi.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

describe('postSimulate', () => {
  it('posts to /simulate and returns job_id', async () => {
    mockedAxios.post = vi.fn().mockResolvedValue({ data: { job_id: 'abc-123' } })
    const req = { device: { V_bi: 1.1, Phi: 2.5e21 }, layers: [], sim_type: 'jv' as const, sim_params: {} }
    const result = await postSimulate(req)
    expect(result.job_id).toBe('abc-123')
    expect(mockedAxios.post).toHaveBeenCalledWith('/api/simulate', req)
  })
})

describe('pollJob', () => {
  it('resolves when status is done', async () => {
    mockedAxios.get = vi.fn().mockResolvedValue({ data: { status: 'done', result: { V_fwd: [0] } } })
    const result = await pollJob('abc-123')
    expect(result.status).toBe('done')
  })
})

describe('getPresets', () => {
  it('returns preset names', async () => {
    mockedAxios.get = vi.fn().mockResolvedValue({ data: ['nip_MAPbI3', 'pin_MAPbI3'] })
    const presets = await getPresets()
    expect(presets).toContain('nip_MAPbI3')
  })
})
```

**Step 2: Run to verify it fails**

```bash
npx vitest run src/api/client.test.ts
```

Expected: `Cannot find module './client'`

**Step 3: Write `frontend/src/api/types.ts`**

```typescript
export interface LayerIn {
  name: string
  role: 'HTL' | 'absorber' | 'ETL'
  thickness: number
  eps_r: number; mu_n: number; mu_p: number; ni: number
  N_A: number; N_D: number; D_ion: number; P_lim: number; P0: number
  tau_n: number; tau_p: number; n1: number; p1: number
  B_rad: number; C_n: number; C_p: number; alpha: number
}

export interface DeviceIn {
  V_bi: number
  Phi: number
}

export interface SimulateRequest {
  device: DeviceIn
  layers: LayerIn[]
  sim_type: 'jv' | 'impedance' | 'degradation'
  sim_params: Record<string, number>
}

export interface JVMetrics {
  V_oc: number; J_sc: number; FF: number; PCE: number
}

export interface JVResult {
  V_fwd: number[]; J_fwd: number[]
  V_rev: number[]; J_rev: number[]
  metrics_fwd: JVMetrics; metrics_rev: JVMetrics
  hysteresis_index: number
}

export interface ImpedanceResult {
  frequencies: number[]; Z_real: number[]; Z_imag: number[]
}

export interface DegradationResult {
  times: number[]; PCE: number[]; V_oc: number[]; J_sc: number[]
}

export type SimResult = JVResult | ImpedanceResult | DegradationResult

export interface JobStatus {
  status: 'pending' | 'running' | 'done' | 'failed'
  result?: SimResult
  error?: string
}
```

**Step 4: Write `frontend/src/api/client.ts`**

```typescript
import axios from 'axios'
import type { SimulateRequest, JobStatus } from './types'

const BASE = import.meta.env.VITE_API_URL ?? ''

export async function postSimulate(req: SimulateRequest): Promise<{ job_id: string }> {
  const { data } = await axios.post(`${BASE}/api/simulate`, req)
  return data
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const { data } = await axios.get(`${BASE}/api/job/${jobId}`)
  return data
}

export async function getPresets(): Promise<string[]> {
  const { data } = await axios.get(`${BASE}/api/presets`)
  return data
}

export async function getPreset(name: string): Promise<{ device: { V_bi: number; Phi: number }; layers: unknown[] }> {
  const { data } = await axios.get(`${BASE}/api/presets/${name}`)
  return data
}
```

**Step 5: Run to verify tests pass**

```bash
npx vitest run src/api/client.test.ts
```

Expected: 3 passed.

**Step 6: Commit**

```bash
git add frontend/src/api/
git commit -m "feat: add TypeScript API client"
```

---

## Task 9: DeviceForm Component

**Files:**
- Create: `frontend/src/components/DeviceForm.tsx`
- Create: `frontend/src/components/DeviceForm.test.tsx`

**Step 1: Write the failing test**

```typescript
// frontend/src/components/DeviceForm.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import DeviceForm from './DeviceForm'
import { DEFAULT_NIP } from '../api/defaults'

test('renders V_bi field', () => {
  render(<DeviceForm value={DEFAULT_NIP} onChange={vi.fn()} />)
  expect(screen.getByLabelText(/V_bi/i)).toBeInTheDocument()
})

test('renders all three layer accordions', () => {
  render(<DeviceForm value={DEFAULT_NIP} onChange={vi.fn()} />)
  expect(screen.getByText(/HTL/i)).toBeInTheDocument()
  expect(screen.getByText(/absorber/i, { exact: false })).toBeInTheDocument()
  expect(screen.getByText(/ETL/i)).toBeInTheDocument()
})

test('calls onChange when V_bi is edited', () => {
  const onChange = vi.fn()
  render(<DeviceForm value={DEFAULT_NIP} onChange={onChange} />)
  const input = screen.getByLabelText(/V_bi/i)
  fireEvent.change(input, { target: { value: '1.2' } })
  expect(onChange).toHaveBeenCalled()
})
```

**Step 2: Run to verify it fails**

```bash
npx vitest run src/components/DeviceForm.test.tsx
```

Expected: `Cannot find module './DeviceForm'`

**Step 3: Write `frontend/src/api/defaults.ts`**

```typescript
import type { DeviceFormState } from '../components/DeviceForm'

export const DEFAULT_NIP: DeviceFormState = {
  V_bi: '1.1', Phi: '2.5e21',
  layers: [
    {
      name: 'spiro_HTL', role: 'HTL', thickness: '200e-9',
      eps_r: '3.0', mu_n: '1e-10', mu_p: '1e-8', ni: '1.0',
      N_A: '2e23', N_D: '0.0', D_ion: '0.0', P_lim: '1e30', P0: '0.0',
      tau_n: '1e-9', tau_p: '1e-9', n1: '1.0', p1: '1.0',
      B_rad: '1e-30', C_n: '1e-42', C_p: '1e-42', alpha: '0.0',
    },
    {
      name: 'MAPbI3', role: 'absorber', thickness: '400e-9',
      eps_r: '24.1', mu_n: '2e-4', mu_p: '2e-4', ni: '3.2e13',
      N_A: '0.0', N_D: '0.0', D_ion: '1e-16', P_lim: '1.6e27', P0: '1.6e24',
      tau_n: '1e-6', tau_p: '1e-6', n1: '3.2e13', p1: '3.2e13',
      B_rad: '5e-22', C_n: '1e-42', C_p: '1e-42', alpha: '1.3e7',
    },
    {
      name: 'TiO2_ETL', role: 'ETL', thickness: '100e-9',
      eps_r: '10.0', mu_n: '1e-7', mu_p: '1e-10', ni: '1.0',
      N_A: '0.0', N_D: '1e24', D_ion: '0.0', P_lim: '1e30', P0: '0.0',
      tau_n: '1e-9', tau_p: '1e-9', n1: '1.0', p1: '1.0',
      B_rad: '1e-30', C_n: '1e-42', C_p: '1e-42', alpha: '0.0',
    },
  ],
}
```

**Step 4: Write `frontend/src/components/DeviceForm.tsx`**

```tsx
import { useState } from 'react'

export interface LayerState {
  name: string; role: 'HTL' | 'absorber' | 'ETL'; thickness: string
  eps_r: string; mu_n: string; mu_p: string; ni: string
  N_A: string; N_D: string; D_ion: string; P_lim: string; P0: string
  tau_n: string; tau_p: string; n1: string; p1: string
  B_rad: string; C_n: string; C_p: string; alpha: string
}

export interface DeviceFormState {
  V_bi: string; Phi: string
  layers: LayerState[]
}

interface Props {
  value: DeviceFormState
  onChange: (next: DeviceFormState) => void
}

const LAYER_FIELDS: Array<{ key: keyof LayerState; label: string; unit: string }> = [
  { key: 'thickness', label: 'Thickness', unit: 'm' },
  { key: 'eps_r',     label: 'ε_r',       unit: '' },
  { key: 'mu_n',      label: 'μ_n',       unit: 'm²/Vs' },
  { key: 'mu_p',      label: 'μ_p',       unit: 'm²/Vs' },
  { key: 'ni',        label: 'nᵢ',        unit: 'm⁻³' },
  { key: 'N_A',       label: 'N_A',       unit: 'm⁻³' },
  { key: 'N_D',       label: 'N_D',       unit: 'm⁻³' },
  { key: 'D_ion',     label: 'D_ion',     unit: 'm²/s' },
  { key: 'P_lim',     label: 'P_lim',     unit: 'm⁻³' },
  { key: 'P0',        label: 'P₀',        unit: 'm⁻³' },
  { key: 'tau_n',     label: 'τ_n',       unit: 's' },
  { key: 'tau_p',     label: 'τ_p',       unit: 's' },
  { key: 'n1',        label: 'n₁',        unit: 'm⁻³' },
  { key: 'p1',        label: 'p₁',        unit: 'm⁻³' },
  { key: 'B_rad',     label: 'B_rad',     unit: 'm³/s' },
  { key: 'C_n',       label: 'C_n',       unit: 'm⁶/s' },
  { key: 'C_p',       label: 'C_p',       unit: 'm⁶/s' },
  { key: 'alpha',     label: 'α',         unit: 'm⁻¹' },
]

function LayerPanel({
  layer, index, onChange,
}: { layer: LayerState; index: number; onChange: (l: LayerState) => void }) {
  const [open, setOpen] = useState(index === 1)  // absorber open by default

  const set = (key: keyof LayerState, val: string) =>
    onChange({ ...layer, [key]: val })

  return (
    <div className="border rounded mb-2">
      <button
        type="button"
        className="w-full text-left px-4 py-2 font-semibold bg-gray-100 hover:bg-gray-200 flex justify-between"
        onClick={() => setOpen(o => !o)}
      >
        <span>{layer.name} <span className="text-gray-500 text-sm">({layer.role})</span></span>
        <span>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-4 py-3 grid grid-cols-2 gap-2">
          {LAYER_FIELDS.map(({ key, label, unit }) => (
            <label key={key} className="flex flex-col text-xs text-gray-600">
              <span>{label} {unit && <span className="text-gray-400">[{unit}]</span>}</span>
              <input
                aria-label={`${layer.name} ${label}`}
                className="border rounded px-2 py-1 font-mono text-sm"
                value={layer[key]}
                onChange={e => set(key, e.target.value)}
              />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

export default function DeviceForm({ value, onChange }: Props) {
  const set = (key: 'V_bi' | 'Phi', val: string) =>
    onChange({ ...value, [key]: val })

  const setLayer = (i: number, layer: LayerState) =>
    onChange({ ...value, layers: value.layers.map((l, j) => j === i ? layer : l) })

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {([['V_bi', 'V_bi', 'V'], ['Phi', 'Φ (photon flux)', 'm⁻²s⁻¹']] as const).map(
          ([key, label, unit]) => (
            <label key={key} className="flex flex-col text-xs text-gray-600">
              <span>{label} <span className="text-gray-400">[{unit}]</span></span>
              <input
                aria-label={key}
                className="border rounded px-2 py-1 font-mono text-sm"
                value={value[key]}
                onChange={e => set(key, e.target.value)}
              />
            </label>
          )
        )}
      </div>
      {value.layers.map((layer, i) => (
        <LayerPanel key={i} layer={layer} index={i} onChange={l => setLayer(i, l)} />
      ))}
    </div>
  )
}
```

**Step 5: Run to verify tests pass**

```bash
npx vitest run src/components/DeviceForm.test.tsx
```

Expected: 3 passed.

**Step 6: Commit**

```bash
git add frontend/src/components/DeviceForm.tsx frontend/src/components/DeviceForm.test.tsx \
        frontend/src/api/defaults.ts
git commit -m "feat: add DeviceForm component"
```

---

## Task 10: SimControls Component

**Files:**
- Create: `frontend/src/components/SimControls.tsx`
- Create: `frontend/src/components/SimControls.test.tsx`

**Step 1: Write the failing test**

```typescript
// frontend/src/components/SimControls.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import SimControls from './SimControls'

const defaultProps = {
  simType: 'jv' as const,
  onSimTypeChange: vi.fn(),
  simParams: { N_grid: 60, n_points: 20, v_rate: 5.0 },
  onSimParamsChange: vi.fn(),
  onRun: vi.fn(),
  status: 'idle' as const,
  error: null,
  onDismissError: vi.fn(),
}

test('renders run button', () => {
  render(<SimControls {...defaultProps} />)
  expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
})

test('run button is disabled while running', () => {
  render(<SimControls {...defaultProps} status="running" />)
  expect(screen.getByRole('button', { name: /running/i })).toBeDisabled()
})

test('calls onRun when button clicked', () => {
  const onRun = vi.fn()
  render(<SimControls {...defaultProps} onRun={onRun} />)
  fireEvent.click(screen.getByRole('button', { name: /run/i }))
  expect(onRun).toHaveBeenCalledOnce()
})

test('shows error banner when error is set', () => {
  render(<SimControls {...defaultProps} error="solver failed" />)
  expect(screen.getByText(/solver failed/i)).toBeInTheDocument()
})

test('sim type radio buttons present', () => {
  render(<SimControls {...defaultProps} />)
  expect(screen.getByLabelText(/j-v/i)).toBeInTheDocument()
  expect(screen.getByLabelText(/impedance/i)).toBeInTheDocument()
  expect(screen.getByLabelText(/degradation/i)).toBeInTheDocument()
})
```

**Step 2: Run to verify it fails**

```bash
npx vitest run src/components/SimControls.test.tsx
```

**Step 3: Write `frontend/src/components/SimControls.tsx`**

```tsx
type SimType = 'jv' | 'impedance' | 'degradation'
type Status = 'idle' | 'pending' | 'running' | 'done' | 'failed'

interface Props {
  simType: SimType
  onSimTypeChange: (t: SimType) => void
  simParams: Record<string, number>
  onSimParamsChange: (p: Record<string, number>) => void
  onRun: () => void
  status: Status
  error: string | null
  onDismissError: () => void
}

const SIM_TYPE_LABELS: Record<SimType, string> = {
  jv: 'J-V sweep',
  impedance: 'Impedance',
  degradation: 'Degradation',
}

const PARAM_DEFS: Record<SimType, Array<{ key: string; label: string; default: number }>> = {
  jv: [
    { key: 'N_grid',   label: 'Grid points', default: 60 },
    { key: 'n_points', label: 'V points',    default: 20 },
    { key: 'v_rate',   label: 'Sweep rate (V/s)', default: 5.0 },
  ],
  impedance: [
    { key: 'N_grid',   label: 'Grid points',   default: 40 },
    { key: 'V_dc',     label: 'V_dc (V)',       default: 0.9 },
    { key: 'freq_min', label: 'f_min (Hz)',     default: 1 },
    { key: 'freq_max', label: 'f_max (Hz)',     default: 1e6 },
    { key: 'n_freq',   label: 'Frequencies',   default: 10 },
  ],
  degradation: [
    { key: 'N_grid',      label: 'Grid points', default: 40 },
    { key: 't_end',       label: 't_end (s)',   default: 3600 },
    { key: 'n_snapshots', label: 'Snapshots',   default: 10 },
  ],
}

export default function SimControls({
  simType, onSimTypeChange, simParams, onSimParamsChange,
  onRun, status, error, onDismissError,
}: Props) {
  const isRunning = status === 'pending' || status === 'running'

  return (
    <div className="space-y-4">
      {/* Sim type selector */}
      <fieldset>
        <legend className="text-sm font-semibold text-gray-700 mb-1">Simulation type</legend>
        <div className="space-y-1">
          {(Object.keys(SIM_TYPE_LABELS) as SimType[]).map(t => (
            <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="radio"
                aria-label={SIM_TYPE_LABELS[t]}
                checked={simType === t}
                onChange={() => onSimTypeChange(t)}
              />
              {SIM_TYPE_LABELS[t]}
            </label>
          ))}
        </div>
      </fieldset>

      {/* Sim params */}
      <div className="grid grid-cols-2 gap-2">
        {PARAM_DEFS[simType].map(({ key, label, default: def }) => (
          <label key={key} className="flex flex-col text-xs text-gray-600">
            {label}
            <input
              type="number"
              className="border rounded px-2 py-1 font-mono text-sm"
              value={simParams[key] ?? def}
              onChange={e => onSimParamsChange({ ...simParams, [key]: parseFloat(e.target.value) })}
            />
          </label>
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border border-red-300 rounded px-3 py-2 text-sm text-red-700 flex justify-between">
          <span>{error}</span>
          <button onClick={onDismissError} className="ml-2 font-bold">×</button>
        </div>
      )}

      {/* Run button */}
      <button
        onClick={onRun}
        disabled={isRunning}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300
                   text-white font-semibold py-2 px-4 rounded flex items-center justify-center gap-2"
      >
        {isRunning ? (
          <>
            <span className="animate-spin">⟳</span>
            Running…
          </>
        ) : (
          '▶ Run'
        )}
      </button>
    </div>
  )
}
```

**Step 4: Run to verify tests pass**

```bash
npx vitest run src/components/SimControls.test.tsx
```

Expected: 5 passed.

**Step 5: Commit**

```bash
git add frontend/src/components/SimControls.tsx frontend/src/components/SimControls.test.tsx
git commit -m "feat: add SimControls component"
```

---

## Task 11: ResultsPanel Component

**Files:**
- Create: `frontend/src/components/ResultsPanel.tsx`
- Create: `frontend/src/components/ResultsPanel.test.tsx`

**Step 1: Write the failing test**

```typescript
// frontend/src/components/ResultsPanel.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import ResultsPanel from './ResultsPanel'
import type { JVResult, ImpedanceResult } from '../api/types'

const JV_RESULT: JVResult = {
  V_fwd: [0, 0.5, 1.0, 1.1],
  J_fwd: [200, 180, 50, 0],
  V_rev: [1.1, 1.0, 0.5, 0],
  J_rev: [0, 60, 185, 200],
  metrics_fwd: { V_oc: 1.05, J_sc: 200, FF: 0.78, PCE: 0.163 },
  metrics_rev: { V_oc: 1.07, J_sc: 200, FF: 0.80, PCE: 0.171 },
  hysteresis_index: 0.047,
}

test('shows J-V chart and metrics when result is provided', () => {
  render(<ResultsPanel simType="jv" result={JV_RESULT} />)
  expect(screen.getByTestId('plotly-chart')).toBeInTheDocument()
  expect(screen.getByText(/J_sc/i)).toBeInTheDocument()
  expect(screen.getByText('200')).toBeInTheDocument()
})

test('shows empty state when no result', () => {
  render(<ResultsPanel simType="jv" result={null} />)
  expect(screen.getByText(/run a simulation/i)).toBeInTheDocument()
})

const IS_RESULT: ImpedanceResult = {
  frequencies: [1, 10, 100],
  Z_real: [10, 8, 5],
  Z_imag: [-5, -3, -1],
}

test('shows Nyquist chart for impedance result', () => {
  render(<ResultsPanel simType="impedance" result={IS_RESULT} />)
  expect(screen.getByTestId('plotly-chart')).toBeInTheDocument()
})
```

**Step 2: Run to verify it fails**

```bash
npx vitest run src/components/ResultsPanel.test.tsx
```

**Step 3: Write `frontend/src/components/ResultsPanel.tsx`**

```tsx
import Plot from 'react-plotly.js'
import type { SimResult, JVResult, ImpedanceResult, DegradationResult } from '../api/types'

type SimType = 'jv' | 'impedance' | 'degradation'

interface Props {
  simType: SimType
  result: SimResult | null
}

function JVPanel({ result }: { result: JVResult }) {
  const traces = [
    { x: result.V_fwd, y: result.J_fwd.map(j => j / 10), name: 'Forward', line: { color: '#4f46e5' } },
    { x: result.V_rev, y: result.J_rev.map(j => j / 10), name: 'Reverse',  line: { color: '#ef4444', dash: 'dash' as const } },
  ]
  const m = result.metrics_fwd

  return (
    <div className="space-y-4">
      <Plot
        data={traces}
        layout={{
          xaxis: { title: 'Voltage (V)' },
          yaxis: { title: 'Current density (mA/cm²)' },
          margin: { t: 20, b: 50, l: 60, r: 20 },
          height: 300,
        }}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="border px-3 py-1 text-left">Scan</th>
            <th className="border px-3 py-1">J_sc (mA/cm²)</th>
            <th className="border px-3 py-1">V_oc (V)</th>
            <th className="border px-3 py-1">FF</th>
            <th className="border px-3 py-1">PCE (%)</th>
          </tr>
        </thead>
        <tbody>
          {([['Forward', result.metrics_fwd], ['Reverse', result.metrics_rev]] as const).map(
            ([label, met]) => (
              <tr key={label}>
                <td className="border px-3 py-1">{label}</td>
                <td className="border px-3 py-1 text-center">{(met.J_sc / 10).toFixed(1)}</td>
                <td className="border px-3 py-1 text-center">{met.V_oc.toFixed(3)}</td>
                <td className="border px-3 py-1 text-center">{(met.FF * 100).toFixed(1)}%</td>
                <td className="border px-3 py-1 text-center">{(met.PCE * 100).toFixed(1)}%</td>
              </tr>
            )
          )}
        </tbody>
      </table>
      <p className="text-sm text-gray-600">
        Hysteresis index: <strong>{(result.hysteresis_index * 100).toFixed(1)}%</strong>
      </p>
    </div>
  )
}

function ImpedancePanel({ result }: { result: ImpedanceResult }) {
  return (
    <Plot
      data={[{
        x: result.Z_real,
        y: result.Z_imag.map(z => -z),
        mode: 'lines+markers',
        name: 'Nyquist',
        line: { color: '#4f46e5' },
      }]}
      layout={{
        xaxis: { title: "Z' (Ω·m²)" },
        yaxis: { title: "-Z'' (Ω·m²)" },
        margin: { t: 20, b: 50, l: 70, r: 20 },
        height: 300,
      }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}

function DegradationPanel({ result }: { result: DegradationResult }) {
  return (
    <Plot
      data={[{
        x: result.times,
        y: result.PCE.map(p => p * 100),
        mode: 'lines+markers',
        name: 'PCE',
        line: { color: '#4f46e5' },
      }]}
      layout={{
        xaxis: { title: 'Time (hours)', type: 'log' },
        yaxis: { title: 'PCE (%)' },
        margin: { t: 20, b: 50, l: 60, r: 20 },
        height: 300,
      }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}

export default function ResultsPanel({ simType, result }: Props) {
  if (!result) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Run a simulation to see results here.
      </div>
    )
  }

  if (simType === 'jv') return <JVPanel result={result as JVResult} />
  if (simType === 'impedance') return <ImpedancePanel result={result as ImpedanceResult} />
  return <DegradationPanel result={result as DegradationResult} />
}
```

**Step 4: Run to verify tests pass**

```bash
npx vitest run src/components/ResultsPanel.test.tsx
```

Expected: 3 passed.

**Step 5: Commit**

```bash
git add frontend/src/components/ResultsPanel.tsx frontend/src/components/ResultsPanel.test.tsx
git commit -m "feat: add ResultsPanel with J-V, Impedance, Degradation charts"
```

---

## Task 12: App Composition

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

**Step 1: Write the integration test**

```typescript
// frontend/src/App.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'
import * as client from './api/client'

vi.mock('./api/client')

test('renders app header', () => {
  render(<App />)
  expect(screen.getByText(/Perovskite Simulator/i)).toBeInTheDocument()
})

test('loads presets on mount', async () => {
  vi.mocked(client.getPresets).mockResolvedValue(['nip_MAPbI3', 'pin_MAPbI3'])
  render(<App />)
  await waitFor(() => expect(client.getPresets).toHaveBeenCalled())
})

test('clicking Run posts a simulation', async () => {
  vi.mocked(client.getPresets).mockResolvedValue(['nip_MAPbI3'])
  vi.mocked(client.postSimulate).mockResolvedValue({ job_id: 'test-123' })
  vi.mocked(client.pollJob).mockResolvedValue({ status: 'done', result: undefined })

  render(<App />)
  fireEvent.click(screen.getByRole('button', { name: /run/i }))

  await waitFor(() => expect(client.postSimulate).toHaveBeenCalled())
})
```

**Step 2: Run to verify test fails**

```bash
npx vitest run src/App.test.tsx
```

**Step 3: Write full `frontend/src/App.tsx`**

```tsx
import { useState, useEffect, useCallback } from 'react'
import DeviceForm, { DeviceFormState } from './components/DeviceForm'
import SimControls from './components/SimControls'
import ResultsPanel from './components/ResultsPanel'
import { postSimulate, pollJob, getPresets, getPreset } from './api/client'
import { DEFAULT_NIP } from './api/defaults'
import type { SimResult } from './api/types'

type SimType = 'jv' | 'impedance' | 'degradation'
type JobStatus = 'idle' | 'pending' | 'running' | 'done' | 'failed'

const DEFAULT_PARAMS: Record<SimType, Record<string, number>> = {
  jv:          { N_grid: 60, n_points: 20, v_rate: 5.0 },
  impedance:   { N_grid: 40, V_dc: 0.9, freq_min: 1, freq_max: 1e6, n_freq: 10 },
  degradation: { N_grid: 40, t_end: 3600, n_snapshots: 10 },
}

function formToRequest(form: DeviceFormState, simType: SimType, simParams: Record<string, number>) {
  return {
    device: { V_bi: parseFloat(form.V_bi), Phi: parseFloat(form.Phi) },
    layers: form.layers.map(l => ({
      name: l.name, role: l.role as 'HTL' | 'absorber' | 'ETL',
      thickness: parseFloat(l.thickness),
      eps_r: parseFloat(l.eps_r), mu_n: parseFloat(l.mu_n), mu_p: parseFloat(l.mu_p),
      ni: parseFloat(l.ni), N_A: parseFloat(l.N_A), N_D: parseFloat(l.N_D),
      D_ion: parseFloat(l.D_ion), P_lim: parseFloat(l.P_lim), P0: parseFloat(l.P0),
      tau_n: parseFloat(l.tau_n), tau_p: parseFloat(l.tau_p),
      n1: parseFloat(l.n1), p1: parseFloat(l.p1),
      B_rad: parseFloat(l.B_rad), C_n: parseFloat(l.C_n), C_p: parseFloat(l.C_p),
      alpha: parseFloat(l.alpha),
    })),
    sim_type: simType,
    sim_params: simParams,
  }
}

export default function App() {
  const [form, setForm] = useState<DeviceFormState>(DEFAULT_NIP)
  const [simType, setSimType] = useState<SimType>('jv')
  const [simParams, setSimParams] = useState(DEFAULT_PARAMS.jv)
  const [status, setStatus] = useState<JobStatus>('idle')
  const [result, setResult] = useState<SimResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [presets, setPresets] = useState<string[]>([])
  const [selectedPreset, setSelectedPreset] = useState('')

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {})
  }, [])

  const handleSimTypeChange = (t: SimType) => {
    setSimType(t)
    setSimParams(DEFAULT_PARAMS[t])
  }

  const handleLoadPreset = async () => {
    if (!selectedPreset) return
    try {
      const data = await getPreset(selectedPreset)
      const layers = (data.layers as Record<string, unknown>[]).map(l => ({
        name: String(l.name), role: l.role as 'HTL' | 'absorber' | 'ETL',
        thickness: String(l.thickness), eps_r: String(l.eps_r),
        mu_n: String(l.mu_n), mu_p: String(l.mu_p), ni: String(l.ni),
        N_A: String(l.N_A), N_D: String(l.N_D), D_ion: String(l.D_ion),
        P_lim: String(l.P_lim), P0: String(l.P0),
        tau_n: String(l.tau_n), tau_p: String(l.tau_p),
        n1: String(l.n1), p1: String(l.p1),
        B_rad: String(l.B_rad), C_n: String(l.C_n), C_p: String(l.C_p),
        alpha: String(l.alpha),
      }))
      setForm({ V_bi: String(data.device.V_bi), Phi: String(data.device.Phi), layers })
    } catch {
      setError('Failed to load preset')
    }
  }

  const handleRun = useCallback(async () => {
    setError(null)
    setResult(null)
    setStatus('pending')
    try {
      const req = formToRequest(form, simType, simParams)
      const { job_id } = await postSimulate(req)

      const poll = async (): Promise<void> => {
        const js = await pollJob(job_id)
        if (js.status === 'done') {
          setStatus('done')
          setResult(js.result ?? null)
        } else if (js.status === 'failed') {
          setStatus('failed')
          setError(js.error ?? 'Simulation failed')
        } else {
          setStatus(js.status === 'running' ? 'running' : 'pending')
          setTimeout(poll, 2000)
        }
      }
      await poll()
    } catch (e) {
      setStatus('failed')
      setError(e instanceof Error ? e.message : 'Unknown error')
    }
  }, [form, simType, simParams])

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-indigo-700 text-white px-6 py-4 shadow">
        <h1 className="text-xl font-bold tracking-tight">Perovskite Solar Cell Simulator</h1>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left panel: device setup */}
        <aside className="w-96 bg-white border-r overflow-y-auto p-4 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <select
              className="flex-1 border rounded px-2 py-1 text-sm"
              value={selectedPreset}
              onChange={e => setSelectedPreset(e.target.value)}
            >
              <option value="">— select preset —</option>
              {presets.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
            <button
              onClick={handleLoadPreset}
              className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm"
            >
              Load
            </button>
          </div>

          <DeviceForm value={form} onChange={setForm} />

          <SimControls
            simType={simType}
            onSimTypeChange={handleSimTypeChange}
            simParams={simParams}
            onSimParamsChange={setSimParams}
            onRun={handleRun}
            status={status}
            error={error}
            onDismissError={() => setError(null)}
          />
        </aside>

        {/* Right panel: results */}
        <main className="flex-1 p-6 overflow-y-auto">
          <ResultsPanel simType={simType} result={result} />
        </main>
      </div>
    </div>
  )
}
```

**Step 4: Run to verify tests pass**

```bash
npx vitest run src/App.test.tsx
```

Expected: 3 passed.

**Step 5: Run the full frontend test suite**

```bash
npx vitest run
```

Expected: all tests pass.

**Step 6: Verify the dev server starts**

```bash
npm run dev
# Open http://localhost:5173 — should see the simulator UI
```

**Step 7: Commit**

```bash
cd ..
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: wire up full App with device form, sim controls, results panel"
```

---

## Task 13: Frontend Dockerfile + Nginx + Docker Compose

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Create: `docker-compose.yml`

**Step 1: Write `frontend/nginx.conf`**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # Proxy API calls to the backend
    location /api/ {
        proxy_pass http://api:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # SPA fallback: all non-asset routes serve index.html
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**Step 2: Write `frontend/Dockerfile`**

```dockerfile
# ── Stage 1: build ──────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── Stage 2: serve ──────────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

**Step 3: Write `docker-compose.yml`** (repo root)

```yaml
version: "3.9"

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: backend/Dockerfile
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    environment:
      REDIS_URL: redis://redis:6379/0
      CONFIGS_DIR: /workspace/configs
    depends_on: [redis]
    restart: unless-stopped

  worker:
    build:
      context: .
      dockerfile: backend/Dockerfile
    command: celery -A celery_worker worker --loglevel=info --concurrency=2
    environment:
      REDIS_URL: redis://redis:6379/0
      CONFIGS_DIR: /workspace/configs
    depends_on: [redis]
    restart: unless-stopped

  frontend:
    build:
      context: frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on: [api]
    restart: unless-stopped
```

**Step 4: Verify docker-compose builds and starts**

```bash
# From repo root:
docker-compose build
docker-compose up
```

Expected: all 4 services start. Open http://localhost:3000 — full simulator UI loads.

**Step 5: Smoke-test end-to-end**

1. Open http://localhost:3000
2. Select preset "nip_MAPbI3" and click **Load**
3. Sim type: J-V, Grid N: 30, n_points: 5, v_rate: 20
4. Click **▶ Run**
5. Spinner appears → after 20–60s, J-V chart and metrics table appear

**Step 6: Commit**

```bash
git add frontend/Dockerfile frontend/nginx.conf docker-compose.yml
git commit -m "feat: add Docker Compose orchestration and frontend Dockerfile"
```

---

## Final Verification

Run all backend tests:

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all backend tests pass.

Run all frontend tests:

```bash
cd frontend && npx vitest run
```

Expected: all frontend tests pass.

Check test coverage:

```bash
cd frontend && npx vitest run --coverage
```

Expected: ≥80% coverage on `src/components/` and `src/api/`.

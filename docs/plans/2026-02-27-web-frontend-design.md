# Web Frontend Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A React + FastAPI web application that lets users edit device parameters, submit simulations, and view interactive results.

**Architecture:** Monorepo with `backend/` (FastAPI + Celery + Redis) and `frontend/` (React + Vite + Plotly.js), orchestrated by Docker Compose. The existing `perovskite_sim/` package is unchanged and imported directly by the Celery worker.

**Tech Stack:** Python 3.11, FastAPI, Celery, Redis, Pydantic v2, React 18, TypeScript, Vite, Plotly.js, Tailwind CSS, Docker Compose.

---

## Repository Layout

```
perovskite-sim/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, mounts routes
│   │   ├── api/
│   │   │   └── simulate.py    # POST /simulate, GET /job/{id}, GET /presets
│   │   ├── tasks/
│   │   │   └── run_sim.py     # Celery tasks: run_jv, run_impedance, run_degradation
│   │   └── schemas.py         # Pydantic request/response models
│   ├── celery_worker.py       # Celery app entry-point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── DeviceForm.tsx       # accordion layer editor + device params
│   │   │   ├── SimControls.tsx      # sim type selector + run button + status
│   │   │   └── ResultsPanel.tsx     # tabbed charts + metrics table
│   │   └── api/
│   │       └── client.ts            # axios: postSimulate, pollJob, getPresets
│   ├── Dockerfile
│   ├── vite.config.ts
│   └── package.json
├── docker-compose.yml
├── docs/plans/
└── perovskite_sim/            # existing simulator (unchanged)
```

---

## Data Flow

```
User edits form
      │
      ▼
[DeviceForm + SimControls]
      │  POST /simulate  { layers, device, sim_type, sim_params }
      ▼
[FastAPI]  ──enqueue──▶  [Celery Worker]  ──imports──▶  perovskite_sim
      │                        │
      │  { job_id }            │  stores result in Redis
      ▼                        │
[Frontend polls]  ◀── GET /job/{id} ──────────────────────
      │
      ▼  status = "done"
[ResultsPanel]  renders Plotly charts from result arrays
```

---

## REST API

### `POST /simulate`

Request body (all fields validated by Pydantic):

```json
{
  "device": { "V_bi": 1.1, "Phi": 2.5e21 },
  "layers": [
    {
      "name": "spiro_HTL",  "role": "HTL",  "thickness": 2e-7,
      "eps_r": 3.0,  "mu_n": 1e-10,  "mu_p": 1e-8,  "ni": 1.0,
      "N_A": 2e23,  "N_D": 0.0,  "D_ion": 0.0,  "P_lim": 1e30,
      "P0": 0.0,  "tau_n": 1e-9,  "tau_p": 1e-9,
      "n1": 1.0,  "p1": 1.0,
      "B_rad": 1e-30,  "C_n": 1e-42,  "C_p": 1e-42,  "alpha": 0.0
    },
    { "name": "MAPbI3", "role": "absorber", ... },
    { "name": "TiO2_ETL", "role": "ETL", ... }
  ],
  "sim_type": "jv",
  "sim_params": { "N_grid": 60, "n_points": 20, "v_rate": 5.0 }
}
```

Response: `202 { "job_id": "uuid4-string" }`

Validation errors return `422` with per-field messages shown inline in the form.

### `GET /job/{job_id}`

```json
// pending / running:
{ "status": "pending" }
{ "status": "running" }

// J-V done:
{
  "status": "done",
  "result": {
    "V_fwd": [...], "J_fwd": [...], "V_rev": [...], "J_rev": [...],
    "metrics_fwd": { "V_oc": 1.02, "J_sc": 185.3, "FF": 0.78, "PCE": 0.147 },
    "metrics_rev": { "V_oc": 1.04, "J_sc": 185.3, "FF": 0.80, "PCE": 0.152 },
    "hysteresis_index": 0.033
  }
}

// impedance done:
{ "status": "done", "result": { "frequencies": [...], "Z_real": [...], "Z_imag": [...] } }

// failed:
{ "status": "failed", "error": "Newton solver did not converge" }
```

### `GET /presets`
Returns `["nip_MAPbI3", "pin_MAPbI3"]`.

### `GET /presets/{name}`
Returns the parsed device config as JSON to pre-fill the form.

---

## Frontend UI

### Layout (two-panel, single page)

```
┌─────────────────────────────────────────────────────────────┐
│  Perovskite Simulator                              [Run ▶]  │
├──────────────────┬──────────────────────────────────────────┤
│  DEVICE SETUP    │  RESULTS (tabs: J-V | Impedance | Degr.) │
│                  │                                           │
│  Preset: [▼ nip] │  ┌─ J-V Curve ──────────────────────┐   │
│                  │  │  Plotly: fwd (blue) + rev (red)   │   │
│  V_bi  [1.1  V]  │  └──────────────────────────────────┘   │
│  Phi   [2.5e21]  │                                           │
│                  │  ┌─ Metrics ────────────────────────┐   │
│  ▶ HTL (200 nm)  │  │        Jsc   Voc   FF    PCE  HI │   │
│  ▼ Absorber      │  │  Fwd   185   1.02  0.78  14.7    │   │
│  │ eps_r [24.1]  │  │  Rev   185   1.04  0.80  15.2    │   │
│  │ mu_n  [2e-4]  │  │  HI = 3.3%                       │   │
│  │ mu_p  [2e-4]  │  └──────────────────────────────────┘   │
│  │ ni    [3.2e13]│                                           │
│  │ tau_n [1e-6]  │                                           │
│  │ tau_p [1e-6]  │                                           │
│  │ D_ion [1e-16] │                                           │
│  │ P0    [1.6e24]│                                           │
│  │ P_lim [1.6e27]│                                           │
│  │ alpha [1.3e7] │                                           │
│  ▶ ETL (100 nm)  │                                           │
│                  │                                           │
│  Sim type:       │                                           │
│  ● J-V  ○ IS     │                                           │
│  ○ Degradation   │                                           │
│                  │                                           │
│  Grid N: [60]    │                                           │
│  n_points: [20]  │                                           │
└──────────────────┴──────────────────────────────────────────┘
```

### Component breakdown

| Component | Responsibility |
|---|---|
| `DeviceForm` | Accordion layers, device-level fields, Load Preset dropdown |
| `SimControls` | Sim type radio, solver params, Run button, status indicator |
| `ResultsPanel` | Tabs for J-V / IS / Degradation; Plotly charts; metrics table |
| `api/client.ts` | `postSimulate()`, `pollJob(id, interval, onDone)`, `getPresets()` |

### UX details

- Parameter inputs accept scientific notation (e.g. `1.3e7`)
- Each field has a unit label and tooltip with physical meaning
- Load Preset button fetches `/presets/{name}` and fills the whole form
- Run button disables while job is pending/running, shows spinner + message
- Polling stops on `done` or `failed`; errors shown as dismissible red banner
- Charts are interactive (zoom, pan, hover tooltips) via Plotly

---

## Docker Compose

```yaml
services:
  redis:
    image: redis:7-alpine

  worker:
    build: ./backend
    command: celery -A celery_worker worker --loglevel=info
    depends_on: [redis]
    volumes: [../perovskite_sim:/app/perovskite_sim]  # reuse existing package

  api:
    build: ./backend
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    depends_on: [redis]

  frontend:
    build: ./frontend
    ports: ["3000:80"]   # nginx serves the Vite build
    depends_on: [api]
```

---

## Error Handling

- **422 Pydantic validation** → field-level inline errors in `DeviceForm`
- **Job `failed`** → dismissible error banner with `error` message string
- **Network error during polling** → banner "Connection lost, retrying…"
- **Simulation timeout** (worker killed after 5 min) → status `failed` with timeout message

---

## Out of Scope (YAGNI)

- User authentication / multi-tenancy
- Persistent job history / database
- Export to CSV / PNG (Plotly's built-in toolbar handles PNG download)
- 2D simulation

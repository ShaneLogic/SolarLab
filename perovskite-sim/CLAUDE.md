# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

1D drift-diffusion + Poisson + mobile-ion simulator for thin-film solar cells (perovskite, CIGS, c-Si), with a FastAPI backend and a Vite/TypeScript/Plotly frontend that wraps the Python experiments as a browser UI.

## Repository Layout

```
perovskite-sim/
├── perovskite_sim/      Python simulation library (installable package)
├── backend/             FastAPI HTTP wrapper around perovskite_sim
├── frontend/            Vite + TS + Plotly single-page UI
├── configs/             YAML device presets consumed by both CLI and UI
├── tests/               pytest suite (unit + integration + regression)
└── notebooks/           exploratory notebooks / benchmarks
```

Every layer talks to the next one through `DeviceStack` + `MaterialParams` (immutable frozen dataclasses); there is no shared mutable state.

## Common Commands

```bash
# Install the Python package in dev mode
pip install -e ".[dev]"

# Full pytest suite (cov + -m 'not slow' by default — see pyproject.toml)
pytest
pytest -m slow                            # slow regression tests
pytest tests/unit/experiments/test_jv_sweep.py::test_name   # single test

# Backend (FastAPI, uvicorn)
# Run from the SolarLab root so --app-dir finds perovskite-sim/backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload

# Frontend (Vite dev server with HMR)
cd perovskite-sim/frontend
npm install          # first time only
npm run dev          # http://127.0.0.1:5173
npm run build        # tsc typecheck + production bundle into dist/
```

The frontend expects the backend at `http://127.0.0.1:8000` (see `frontend/src/api.ts`); CORS is open for that origin.

## Simulator Architecture (`perovskite_sim/`)

**Core method.** Method of Lines: Scharfetter–Gummel finite elements on a tanh-clustered multilayer grid for space; `scipy.integrate.solve_ivp(Radau)` for time. Poisson is a sparse tridiagonal solve with harmonic-mean face permittivity, evaluated once per RHS call (no inner Newton loop). State vector is `y = (n, p, P)` per grid node.

**Layer-wise arrays.** `solver/mol.py:_build_layerwise_arrays` expands per-layer `MaterialParams` into per-node numpy arrays and is the single source of truth for material properties at a grid point. Anything reading layer-local quantities (`recombination.py`, `ion_migration.py`, `poisson.py`) goes through this.

**Heterojunctions.** `chi` (electron affinity, eV) and `Eg` (band gap, eV) on each layer set the band offsets. Interface recombination uses per-interface `(v_n, v_p)` surface-recombination velocities carried by `DeviceStack.interfaces`.

**Experiments** (`perovskite_sim/experiments/`):
- `jv_sweep.run_jv_sweep` — forward then reverse scan; reuses the previous steady state as initial condition so ionic memory is preserved and the hysteresis loop comes out of the physics, not post-processing.
- `impedance.run_impedance` — at each frequency integrates a few AC cycles and extracts amplitude/phase with a lock-in (sin/cos multiply + low-pass). Adds the displacement current `ε₀·ε_r·∂E/∂t`.
- `degradation.run_degradation` — long-time transient; at each probe time it takes a **frozen-ion snapshot**: a `replace`-d copy of the stack with `D_ion = 0` in every layer is used for a short settle integration at each probe voltage (`_freeze_ions` + `_measure_snapshot_metrics`). This measures the instantaneous electronic response under the current ionic configuration and is the only correct way to compute snapshot J–V without averaging over ion drift.

### Solver gotcha — Radau max_step cap

Near flat-band (V ≈ V_bi) the Jacobian is nearly singular and Radau's adaptive LTE estimator under-reports the error, letting it accept one giant step on the wrong branch of the implicit system. This manifests as an unphysical J–V spike (e.g. `J=258` sandwiched between `188` and `101`). **All three experiments therefore cap `max_step = Δt / k`** on every `run_transient` sub-interval (`jv_sweep._integrate_step`, `impedance`, `degradation._measure_snapshot_metrics`). Do not remove these caps; if you need tighter control, lower the divisor instead. Any new experiment that calls `run_transient` across a voltage where the Jacobian can become near-singular must apply the same cap.

## Backend (`backend/main.py`)

Thin FastAPI wrapper. Three POST endpoints (`/api/jv`, `/api/impedance`, `/api/degradation`) each accept either an inline `device` dict or a `config_path` name from `configs/`; they build a `DeviceStack` via `build_stack` and call the corresponding `perovskite_sim.experiments` function. `GET /api/configs` auto-scans the `configs/` directory, so dropping a new YAML in that folder makes it visible to the frontend with no code change.

### YAML scientific-notation gotcha

PyYAML's default loader is YAML 1.1, which only treats `1.0e-9` as a float — a bare `1e-9` (no decimal point) is returned as a **string**. The frontend's numeric editor then throws `v.toExponential is not a function` when rendering such a field. The fix lives in `backend/main.py:_coerce_numbers`, a recursive pass applied in `get_config` that turns any numeric-looking string into a float. When editing the backend do not bypass this coercion, and when adding new YAML configs you can use either form, but `1e-9` is fine because the backend normalises it.

## Frontend (`frontend/src/`)

Plain TypeScript (no framework). `main.ts` wires five tabs — `J–V Sweep`, `Impedance`, `Degradation`, `Tutorial`, `Algorithm` — each mounted lazily into its panel `<section>`. Key modules:

- `api.ts` — typed fetch wrappers for the backend
- `config-editor.ts` — collapsible `<details>` layer editor with Geometry / Transport / Recombination / Ions & Optics groups. `fmt()`/`numAttr()` accept `unknown` and coerce via `Number(v)` so a non-numeric field can never crash the panel.
- `device-panel.ts` — preset dropdown + Reset button; fetches `/api/configs` and `/api/configs/{name}`
- `panels/{jv,impedance,degradation}.ts` — experiment panels: form + run button + Plotly plots + metric cards
- `panels/{tutorial,algorithm}.ts` — static documentation panels (user-guide + PDE / discretisation / solver description)
- `plot-theme.ts`, `plotly.d.ts`, `types.ts`, `ui-helpers.ts` — shared styling, type shims, small helpers

Plotly is pulled from `plotly.js-dist-min` which is why `vite build` warns about the 4 MB bundle; that warning is expected and not a bug.

## Configs (`configs/`)

Presets shipped with the repo: `nip_MAPbI3.yaml`, `pin_MAPbI3.yaml`, `ionmonger_benchmark.yaml` (Courtier 2019 reference), `driftfusion_benchmark.yaml`, `cigs_baseline.yaml` (ZnO/CdS/CIGS heterostructure, `D_ion = 0`), `cSi_homojunction.yaml` (n⁺/p wafer, `D_ion = 0`). The YAML schema mirrors `MaterialParams` + `DeviceStack.interfaces`; see any existing file for the field list. Non-perovskite stacks must set `D_ion = 0` in every layer — the ion equations still integrate but contribute zero flux.

Practical solver envelope: the Radau transient handles sub-micron perovskite stacks comfortably (ionmonger reference ≈ 25 s for a full J–V). Thick inorganic absorbers (2 µm CIGS, 180 µm Si) are structurally valid and `solve_equilibrium` converges, but a full transient J–V sweep is impractical at default tolerances. Use thinner absorbers or a coarser grid when iterating, and prefer equilibrium-level tests for those configs.

## Data Model Invariants

- `MaterialParams`, `SolverConfig`, `LayerSpec`, `DeviceStack` are **frozen dataclasses**. Never mutate in place — use `dataclasses.replace(...)` to produce a new instance (see `degradation._freeze_ions` for the canonical pattern).
- All physical quantities are SI (m, s, V, A/m², m⁻³). The only exceptions are `chi` and `Eg` which are in eV (numerically equal to volts through `q`).
- `DeviceStack.interfaces` is a tuple of `(v_n, v_p)` pairs with length `len(layers) - 1`, ordered from contact side to contact side.

## Test Structure

```
tests/
├── unit/          per-module physics + solver + experiments tests
├── integration/   end-to-end experiment runs on preset configs
└── regression/    physical sanity checks (V_oc range, J_sc bounds, ...)
```

`pytest` defaults (from `pyproject.toml`) include coverage and exclude `-m slow`. The regression suite is where to add new "result should look physically reasonable" checks.

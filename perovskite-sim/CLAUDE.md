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

**Core method.** Method of Lines: Scharfetter–Gummel finite elements on a tanh-clustered multilayer grid for space; `scipy.integrate.solve_ivp(Radau)` for time. Poisson is a tridiagonal solve with harmonic-mean face permittivity, evaluated once per RHS call (no inner Newton loop). State vector is `y = (n, p, P)` per grid node.

**MaterialArrays cache (hot path).** `solver/mol.py:build_material_arrays(x, stack)` returns an immutable `MaterialArrays` bundle holding every per-node / per-face array the RHS needs (`D_n_face`, `D_p_face`, `eps_r`, `N_A`, `N_D`, `P_ion0`, interface masks, boundary concentrations) plus a pre-factored Poisson operator. Every experiment builds this once per run and threads it through `assemble_rhs`, `run_transient`, `split_step`, `_compute_current`, and `_integrate_step` via the `mat=` kwarg. Rebuilding it per RHS call was the dominant cost before the cache landed — do not reintroduce a code path that calls `build_material_arrays` inside the inner loop. Degradation is the one exception: it rebuilds `mat` only when the absorber-damage state actually changes (`mat_active` in `run_degradation`).

**Poisson LU cache.** `physics/poisson.py:factor_poisson(x, eps_r)` does one LAPACK `dgttrf` call and returns a `PoissonFactor` holding `(dl, d, du, du2, ipiv)`. `solve_poisson_prefactored(fac, rho, phi_left, phi_right)` then runs a single `dgttrs` per call — ~40× faster than the legacy `solve_poisson` path. The factorization only depends on `(x, eps_r)`, which are constant across a transient, so it is cached inside `MaterialArrays.poisson_factor`. Legacy `solve_poisson` is kept as a fallback but is not on any hot path.

**Heterojunctions.** `chi` (electron affinity, eV) and `Eg` (band gap, eV) on each layer set the band offsets. Interface recombination uses per-interface `(v_n, v_p)` surface-recombination velocities carried by `DeviceStack.interfaces`.

**Band-offset contact BCs.** `DeviceStack.compute_V_bi()` derives the built-in potential from the Fermi-level difference across the heterostack (accounting for chi, Eg, doping, and ni). When chi=Eg=0 in all layers (legacy configs), it falls back to the manual `V_bi` field. The computed value is stored as `MaterialArrays.V_bi_eff` and used for voltage sweep range calculation (`V_max` defaults). The Poisson BC still uses `stack.V_bi` to match IonMonger's convention — do not substitute `V_bi_eff` into the Poisson boundary without careful validation, as IonMonger treats V_bi as a free parameter representing the degenerate-doping limit.

**Thermionic emission (TE) at heterointerfaces.** At interfaces where the conduction-band offset |delta_Ec| or valence-band offset |delta_Ev| exceeds 0.05 eV, the SG flux is capped to the Richardson-Dushman thermionic emission limit (`fe_operators.thermionic_emission_flux`). This prevents the SG scheme from overestimating current across sharp band discontinuities (a known artifact when the band offset is resolved in a single grid spacing). The capping is applied in `continuity.py:carrier_continuity_rhs`. Richardson constants `A_star_n` and `A_star_p` default to the free-electron value (1.2017e6 A/(m²·K²)) and can be overridden per layer in YAML configs. Interface faces where TE activates are pre-computed in `MaterialArrays.interface_faces`. Note: IonMonger does not use TE, so our V_oc is ~0.1 V higher than IonMonger's on the same parameter set; Phase 5 (tiered modes) will add a legacy mode that disables TE for exact IonMonger reproduction.

**Transfer-matrix optics (TMM).** `physics/optics.py` implements the coherent thin-film transfer-matrix method (Pettersson et al. 1999 / Burkhard et al. 2010) for position-resolved optical generation G(x). Each layer carries `optical_material` (string key for n,k CSV data in `perovskite_sim/data/nk/`) or `n_optical` (constant refractive index fallback). When any layer has `optical_material` set, `mol.py:_compute_tmm_generation` builds the TMM stack, loads the AM1.5G spectrum from `data/am15g.csv`, and computes G(x) once during `build_material_arrays`. The result is stored as `MaterialArrays.G_optical` and used in `assemble_rhs` instead of Beer-Lambert. Key physics: the absorption formula includes the `n_real / n_ambient` Poynting-vector correction so that R+T+A=1 (energy conservation). Without `optical_material`, the original Beer-Lambert `G = alpha * Phi * exp(-alpha*x)` path is unchanged.

**Activated presets:** `nip_MAPbI3_tmm.yaml`, `pin_MAPbI3_tmm.yaml` (Phase 2 — Apr 2026). Both prepend a 1 mm `role: substrate` glass layer (optical-only) and set `optical_material` on every electrical layer. The vanilla `nip_MAPbI3.yaml` and `pin_MAPbI3.yaml` remain Beer-Lambert for back-compat with existing benchmarks. Substrate layers are filtered out of the drift-diffusion grid by `electrical_layers()` in `models/device.py`; the TMM spatial grid is offset by the substrate cumulative thickness so G(x) lands on the correct electrical nodes.

**Experiments** (`perovskite_sim/experiments/`):
- `jv_sweep.run_jv_sweep` — forward then reverse scan; reuses the previous steady state as initial condition so ionic memory is preserved and the hysteresis loop comes out of the physics, not post-processing.
- `impedance.run_impedance` — at each frequency integrates a few AC cycles and extracts amplitude/phase with a lock-in (sin/cos multiply + low-pass). Adds the displacement current `ε₀·ε_r·∂E/∂t`.
- `degradation.run_degradation` — long-time transient; at each probe time it takes a **frozen-ion snapshot**: a `replace`-d copy of the stack with `D_ion = 0` in every layer is used for a short settle integration at each probe voltage (`_freeze_ions` + `_measure_snapshot_metrics`). This measures the instantaneous electronic response under the current ionic configuration and is the only correct way to compute snapshot J–V without averaging over ion drift.

### Solver gotcha — Radau max_step cap

Near flat-band (V ≈ V_bi) the Jacobian is nearly singular and Radau's adaptive LTE estimator under-reports the error, letting it accept one giant step on the wrong branch of the implicit system. This manifests as an unphysical J–V spike (e.g. `J=258` sandwiched between `188` and `101`). **All three experiments therefore cap `max_step = Δt / k`** on every `run_transient` sub-interval (`jv_sweep._integrate_step`, `impedance`, `degradation._measure_snapshot_metrics`). Do not remove these caps; if you need tighter control, lower the divisor instead. Any new experiment that calls `run_transient` across a voltage where the Jacobian can become near-singular must apply the same cap.

## Backend (`backend/main.py`)

Thin FastAPI wrapper. Two endpoint families:

**Legacy blocking endpoints** — `POST /api/jv`, `/api/impedance`, `/api/degradation`. Run synchronously and return the full result in one response. Kept for backwards compatibility; the frontend no longer uses them.

**Streaming job endpoints** (frontend uses these):
- `POST /api/jobs` — body `{kind, config_path|device, params}`. Dispatches the experiment onto a worker thread via `JobRegistry.submit` and returns `{job_id}`. The closure captured in `_run(reporter)` passes `reporter.report` as the `progress=` kwarg to the experiment.
- `GET /api/jobs/{id}/events` — Server-Sent-Events stream. Named events are `progress` (JSON `{stage, current, total, eta_s, message}`), `result` (final serialized result), `error` (on exception), and `done` (always last). The handler uses `run_in_executor` + a 0.5 s drain timeout to avoid blocking the event loop, and emits `: keepalive` SSE comments in the gap between progress frames.

`backend/progress.py` and `backend/jobs.py` are the pub/sub primitive and the thread-per-job registry behind this. Any new experiment that wants progress should take a `progress: Callable[[str, int, int, str], None] | None = None` kwarg (see `experiments/jv_sweep.py` for the pattern) and the backend `_run` closure wraps it with `reporter.report`.

`GET /api/configs` auto-scans `configs/`, so dropping a new YAML in that folder makes it visible to the frontend with no code change.

### Development loop — restart uvicorn after backend edits

`uvicorn backend.main:app --reload` picks up most edits, but the watcher is set on the SolarLab root and occasionally misses changes under `perovskite-sim/backend/`. If `/api/jobs` or a new endpoint returns 404 after an edit, kill and restart the process rather than trusting `--reload`.

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

### Backend URL: absolute, not proxied

`api.ts` and `job-stream.ts` both hardcode `http://127.0.0.1:8000` as the base URL. The Vite dev-server proxy is configured for `/api` in `vite.config.ts`, but its SPA history fallback intercepts `/api/configs` before the proxy has a chance to match, returning `index.html` — the JSON parse then fails with `Unexpected token '<'`. Hitting the backend directly sidesteps this. CORS is already wide open in `backend/main.py`, so this works from both `npm run dev` and `npm run preview`/`dist`.

SSE notes worth knowing:
- `EventSource` fires a native `error` event when the server closes the connection. `job-stream.ts:streamJobEvents` ignores events whose `e.data` is falsy — only SSE `event: error` frames sent from the backend are surfaced to the user.
- Progress frames and the terminal `done` frame go through named SSE events (`event: progress`, `event: result`, `event: error`, `event: done`), not the default message channel, so each has its own `addEventListener`.

### Streaming-job + progress-bar pattern

Each experiment panel (`panels/{jv,impedance,degradation}.ts`) follows the same shape:
1. Instantiate a `ProgressBarHandle` once in `mount*Panel` via `createProgressBar(progressContainer)`.
2. On button click, call `startJob(kind, device, params)` → get `job_id`.
3. Open the stream with `streamJobEvents<TResult>(jobId, { onProgress, onResult, onError, onDone })`.
4. `onResult` calls the panel-local `render*Results`; `onDone` re-enables the Run button.

The button is disabled between `startJob` and `onDone`, and the progress bar is `reset()` on every run so a second click doesn't briefly show the previous run's "done" state.

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

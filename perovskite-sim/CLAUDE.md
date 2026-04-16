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

**Transfer-matrix optics (TMM).** `physics/optics.py` implements the coherent thin-film transfer-matrix method (Pettersson et al. 1999 / Burkhard et al. 2010) for position-resolved optical generation G(x). Each layer carries `optical_material` (string key for n,k CSV data in `perovskite_sim/data/nk/`) or `n_optical` (constant refractive index fallback). When any layer has `optical_material` set, `mol.py:_compute_tmm_generation` builds the TMM stack, loads the AM1.5G spectrum from `data/am15g.csv`, and computes G(x) once during `build_material_arrays`. The result is stored as `MaterialArrays.G_optical` and used in `assemble_rhs` instead of Beer-Lambert. Key physics: the absorption formula includes the `n_real / n_ambient` Poynting-vector correction so that R+T+A=1 (energy conservation). Without `optical_material`, the original Beer-Lambert `G = alpha * Phi * exp(-alpha*x)` path is unchanged. The `_inv2x2` batched matrix inverse in `optics.py` has a determinant guard (clamps `|det| < 1e-30` to avoid division by zero at wavelengths where the transfer matrix is singular); do not remove this guard.

**Activated presets:** `nip_MAPbI3_tmm.yaml`, `pin_MAPbI3_tmm.yaml` (Phase 2 — Apr 2026). Both prepend a 1 mm `role: substrate` glass layer (optical-only) and set `optical_material` on every electrical layer. The vanilla `nip_MAPbI3.yaml` and `pin_MAPbI3.yaml` remain Beer-Lambert for back-compat with existing benchmarks. Substrate layers are filtered out of the drift-diffusion grid by `electrical_layers()` in `models/device.py`; the TMM spatial grid is offset by the substrate cumulative thickness so G(x) lands on the correct electrical nodes.

**Experiments** (`perovskite_sim/experiments/`):
- `jv_sweep.run_jv_sweep` — forward then reverse scan; reuses the previous steady state as initial condition so ionic memory is preserved and the hysteresis loop comes out of the physics, not post-processing. A per-step `_JV_RADAU_MAX_NFEV = 100_000` guard aborts any single voltage step that consumes too many RHS evaluations (prevents the solver from hanging on pathological substrate-stack configs). Supports `illuminated=False` for dark J-V (G=0, dark-equilibrium start) and `save_snapshots=True` for collecting `SpatialSnapshot` at every voltage point.
- `jv_sweep.compute_current_components` — decomposes the terminal current into J_n (electron), J_p (hole), J_ion (ionic, both species), and J_disp (displacement) at every mesh face. Uses the same SG flux formulas as `assemble_rhs` but multiplied by Q for A/m² units. Ion current includes the steric Blakemore correction and dual-ion support.
- `jv_sweep.extract_spatial_snapshot` — extracts a `SpatialSnapshot` (phi, E, n, p, P, rho) from a packed state vector at a given voltage. Used internally by `save_snapshots` and available as a standalone API.
- `impedance.run_impedance` — at each frequency integrates a few AC cycles and extracts amplitude/phase with a lock-in (sin/cos multiply + low-pass). Adds the displacement current `ε₀·ε_r·∂E/∂t`.
- `degradation.run_degradation` — long-time transient; at each probe time it takes a **frozen-ion snapshot**: a `replace`-d copy of the stack with `D_ion = 0` in every layer is used for a short settle integration at each probe voltage (`_freeze_ions` + `_measure_snapshot_metrics`). This measures the instantaneous electronic response under the current ionic configuration and is the only correct way to compute snapshot J–V without averaging over ion drift.
- `tpv.run_tpv` — transient photovoltage experiment. Equilibrates at V_oc under illumination, applies a fractional generation pulse (delta_G_frac), tracks J(t) at fixed V_app=V_oc, converts to V(t) via the small-signal relation delta_V = -J * R_oc, and fits a mono-exponential decay time tau. Returns a `TPVResult` with t, V, J arrays plus V_oc, tau, delta_V0.
- `tandem_jv.run_tandem_jv` — 2T monolithic tandem driver. Runs a combined TMM over the full stack (`physics/tandem_optics.py`) to partition G(x) into top/bottom sub-cell profiles, then performs independent drift-diffusion J–V sweeps with `fixed_generation`, and series-matches at a common current grid (`series_match_jv`). Returns a `TandemJVResult` holding per-sub-cell results plus the tandem J–V curve.

**Tandem optics** (`physics/tandem_optics.py`): Runs one TMM over the entire tandem stack (top + junction + bottom), then splits absorption by layer. Junction layers count as parasitic absorption. The `TandemGeneration` dataclass holds `G_top(x)` and `G_bot(x)` arrays. The tandem config model lives in `models/tandem_config.py` (`TandemConfig`, `JunctionLayer`).

### Solver gotcha — RHS finite-check

`assemble_rhs` includes a `np.all(np.isfinite(dydt))` guard that raises `ValueError` if any element is NaN or Inf. This catches blow-ups from singular TMM matrices, zero-thickness layers, or extreme doping imbalances early — before Radau can silently accept them and produce garbage. Do not remove this check.

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

**Custom stacks (Phase 2b — Apr 2026):** In full tier the Device pane renders a vertical layer visualizer with add/remove/reorder, a template library, structural validation, and a Save-As path that lands user presets in `configs/user/`. The accordion editor is preserved for fast/legacy tiers. New backend endpoints: `GET /api/layer-templates`, `POST /api/configs/user`. `GET /api/configs` now returns `{name, namespace}` entries.

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

Presets shipped with the repo:

- **Beer-Lambert:** `nip_MAPbI3`, `pin_MAPbI3`, `ionmonger_benchmark` (Courtier 2019), `driftfusion_benchmark`, `cigs_baseline` (ZnO/CdS/CIGS, `D_ion=0`), `cSi_homojunction` (n+/p wafer, `D_ion=0`)
- **TMM-enabled:** `nip_MAPbI3_tmm`, `pin_MAPbI3_tmm`, `ionmonger_benchmark_tmm`, `driftfusion_benchmark_tmm`
- **Tandem sub-cells:** `nip_wideGap_FACs_1p77` (1.77 eV top), `nip_SnPb_1p22` (1.22 eV bottom)
- **Tandem config:** `tandem_lin2019` (2T monolithic, uses `TandemConfig` not `DeviceStack`)

The YAML schema mirrors `MaterialParams` + `DeviceStack.interfaces`; see any existing file for the field list. Non-perovskite stacks must set `D_ion = 0` in every layer — the ion equations still integrate but contribute zero flux. Tandem configs use a separate `TandemConfig` schema (`models/tandem_config.py`) that references two sub-cell configs plus junction layers.

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

`pytest` defaults (from `pyproject.toml`) exclude `-m slow`. Coverage is **opt-in** — pass `--cov=perovskite_sim --cov-report=term-missing` explicitly when you want a report. The regression suite is where to add new "result should look physically reasonable" checks.

Regression tests:
- `test_jv_regression.py` — V_oc / J_sc / FF / HI envelopes for all presets
- `test_tmm_baseline.py` — TMM-specific baselines (slow, BLAS-pinned)
- `test_conservation.py` — energy conservation (R+T+A=1) and J-V monotonicity checks; catches TMM numerical issues and solver blow-ups

### Test gotcha — slow suite BLAS thread pinning

The TMM regression suite (`tests/regression/test_tmm_baseline.py`) drives ~4700 calls to `scipy.linalg.lu_factor` on the ~300×300 Radau Jacobian. These matrices are too small for multi-threaded BLAS to pay off — on a 10-core box OpenBLAS spins up every LU call across all cores and the thread-creation + contention overhead turns a 14 s test into a 5-10 minute test. A Phase 2a investigation wasted four run-kills diagnosing this as a "stall" (runs were being terminated at ~4 min wall, but they would have taken another 2-3 min each to finish under oversubscription). It is NOT a hang.

`tests/conftest.py` pins BLAS threads to 1 via `threadpoolctl` whenever the `slow` marker is selected (excluding the default `-m 'not slow'` unit run). numpy/scipy must be imported inside the hook before calling `threadpool_limits`, because threadpoolctl only sees already-loaded backends. Do not remove this hook; also do not re-bake `--cov=perovskite_sim` into `pyproject.toml` `addopts` — pytest-cov's line tracer adds a further ~17× overhead on the same hot loop and is equally sticky (the tracer is installed during pytest-cov's own `pytest_configure`, before any user conftest gets a chance to stop it).

Canonical invocations:
- Unit + integration: `pytest` (default, ~15 s, no coverage)
- Unit with coverage: `pytest --cov=perovskite_sim --cov-report=term-missing`
- Slow regression: `pytest -m slow` (BLAS pinned automatically, ~27 s for the TMM baselines)

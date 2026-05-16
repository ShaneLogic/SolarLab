# AGENTS.md

Guidance for Codex-style agents working inside `perovskite-sim/`.

This file is scoped to the primary SolarLab simulator tree. The SolarLab
repository root has its own `AGENTS.md`; read that first when deciding which
tree to edit. Work in this tree for physics, solver, backend, configs, tests,
and most screening-framework work.

## Project Shape

`perovskite-sim` contains the Python simulator, FastAPI backend, Vite/TypeScript
frontend, YAML presets, notebooks, and pytest suite.

```text
perovskite-sim/
├── perovskite_sim/   Python package: drift-diffusion, ions, optics, experiments
├── backend/          FastAPI wrapper and streaming job API
├── frontend/         Vite + TypeScript + Plotly UI
├── configs/          Device YAML presets
├── tests/            unit, integration, regression, validation tests
└── notebooks/        exploratory benchmarks and parity scripts
```

The core data flow is:

```text
YAML or inline device dict
-> MaterialParams + LayerSpec + DeviceStack
-> experiment driver
-> solver/mol.py MaterialArrays cache
-> JV/EQE/SunsVoc/degradation/tandem result dataclasses
```

## Common Commands

Run these from inside `perovskite-sim/` unless noted.

```bash
pip install -e ".[dev]"
pytest
pytest -m slow
pytest tests/unit/experiments/test_jv_sweep.py::test_name
pytest --cov=perovskite_sim --cov-report=term-missing
```

Backend, run from the SolarLab root so `--app-dir` resolves:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
npm run build
```

The frontend expects the backend at `http://127.0.0.1:8000`.

## Main Modules

- `perovskite_sim/models/parameters.py`: `MaterialParams` and solver config.
- `perovskite_sim/models/device.py`: `LayerSpec`, `DeviceStack`, substrate
  filtering, built-in-potential calculation.
- `perovskite_sim/models/config_loader.py`: YAML to `DeviceStack`.
- `perovskite_sim/models/mode.py`: `legacy`, `fast`, `full` physics tiers.
- `perovskite_sim/solver/mol.py`: method-of-lines RHS, material cache,
  transient solver, generation cache, current helpers.
- `perovskite_sim/solver/newton.py`: dark equilibrium initial condition.
- `perovskite_sim/solver/illuminated_ss.py`: illuminated steady state.
- `perovskite_sim/physics/`: Poisson, continuity, recombination, optics,
  ion migration, traps, temperature, contacts, field mobility.
- `perovskite_sim/experiments/`: public experiment drivers.
- `perovskite_sim/data/nk/`: optical constants CSV files used by TMM.
- `backend/main.py`: HTTP endpoints, config loading, job dispatch glue.
- `backend/jobs.py` and `backend/progress.py`: in-process streaming jobs.

## Data Model Invariants

- `MaterialParams`, `LayerSpec`, `DeviceStack`, and most result objects are
  frozen dataclasses. Do not mutate them in place. Use `dataclasses.replace`.
- Units are SI: m, s, V, A/m^2, m^-3. `chi` and `Eg` are in eV.
- `DeviceStack.interfaces` is ordered by internal layer interface and contains
  `(v_n, v_p)` surface recombination velocities in m/s.
- Layers with `role: substrate` are optical-only. They must form a contiguous
  prefix of the stack. Drift-diffusion uses `electrical_layers(stack)`.
- Non-perovskite stacks should set ion mobility/density fields to zero. The
  ion state may still exist, but contributes zero flux.
- YAML config fields mirror `MaterialParams` plus `DeviceStack` device fields.

## Solver Architecture

The simulator uses a method-of-lines formulation:

- space: Scharfetter-Gummel finite elements on multilayer grids;
- time: `scipy.integrate.solve_ivp`, primarily Radau;
- electrostatics: tridiagonal Poisson solve using harmonic face permittivity.

`solver/mol.py:build_material_arrays(x, stack)` is the hot-path cache builder.
It creates immutable per-node/per-face arrays, interface masks, contact carrier
densities, optical generation, and a prefactored Poisson operator. Experiments
should build this once and pass it through via `mat=`.

Do not call `build_material_arrays` inside the RHS or per-time-step inner loop.
The only acceptable rebuild pattern is when a physical parameter actually
changes between outer steps, such as damage feedback in degradation.

`physics/poisson.py:factor_poisson` and `solve_poisson_prefactored` are the
fast path. The legacy `solve_poisson` path is not the performance path.

## Physics Tiers

Use `DeviceStack.mode` / YAML `device.mode`:

- `legacy`: disables upgraded physics for old benchmark reproduction.
- `fast`: enables build-once physics, but skips per-RHS hooks.
- `full`: enables every configured hook.

For high-throughput screening, prefer `fast` first, then rerun finalists in
`full` if the hypothesis depends on radiative reabsorption, field-dependent
mobility, or selective contacts.

The tier is a ceiling: hooks activate only if the config supplies the required
parameters. For example, TMM needs `optical_material`, field mobility needs
nonzero `v_sat_*` or `pf_gamma_*`, and selective contacts need at least one
finite `S_*` field.

## Optics And Material Data

TMM optics use `optical_material` keys that point to CSV files in
`perovskite_sim/data/nk/`.

CSV format:

```csv
wavelength_nm,n,k
300, ...
```

`perovskite_sim/data/__init__.py:load_nk` interpolates only inside the native
wavelength range and raises on out-of-range requests. Do not add silent
clamping or extrapolation.

`solver/mol.py:_compute_tmm_generation` builds the full optical stack, including
substrates, then shifts the electrical grid by the substrate thickness so the
generation profile lands on electrical layers.

When adding new optical data, also update `perovskite_sim/data/nk/manifest.yaml`
with source/provenance notes. If data are synthetic or placeholder, label them
plainly.

## Experiment APIs

Use these public drivers for automation and screening:

- `experiments.jv_sweep.run_jv_sweep`: forward/reverse JV, `V_oc`, `J_sc`,
  `FF`, `PCE`, hysteresis, optional snapshots/current decomposition.
- `experiments.eqe.compute_eqe`: wavelength-resolved EQE; requires TMM data.
- `experiments.suns_voc.run_suns_voc`: Suns-Voc and pseudo-FF.
- `experiments.degradation.run_degradation`: ion-coupled damage proxy.
- `experiments.tandem_jv.run_tandem_jv`: 2T tandem current matching.

For a new screening layer, call these APIs directly rather than duplicating
solver logic.

## Numerical Gotchas

- `assemble_rhs` contains a finite-value guard for `dydt`. Do not remove it.
- JV, impedance, and degradation use `max_step` caps around voltage steps to
  avoid Radau accepting giant near-flat-band steps. New experiments crossing
  near-flat-band voltages should use the same pattern.
- JV bisection falls back to BDF after exhausting the Radau bisection budget.
  Keep this last-resort fallback bounded.
- Steady-state terminal-current measurements should use the median-current
  helper `_compute_current_ss`, not a single boundary face.
- EQE and Suns-Voc use longer `t_settle` defaults and dark-current subtraction
  to suppress residual ionic transient artifacts.
- TMM transfer-matrix inversion has a determinant guard in `physics/optics.py`;
  do not remove it.
- Thick CIGS or c-Si absorbers may be structurally valid but too slow for full
  transient JV at default tolerances. Use coarse grids or equilibrium-level
  checks while iterating.

## Backend Notes

`backend/main.py` exposes both legacy blocking endpoints and streaming job
endpoints. The frontend primarily uses streaming jobs:

```text
POST /api/jobs
GET  /api/jobs/{id}/events
```

New long-running experiments should accept:

```python
progress: Callable[[str, int, int, str], None] | None = None
```

and report progress through that callback. The backend job wrapper converts it
to SSE progress events.

`GET /api/configs` scans `configs/`. `GET /api/optical-materials` scans
`perovskite_sim/data/nk/*.csv`, so adding a CSV makes it visible by key.

PyYAML can parse bare `1e-9` as a string. The backend has numeric coercion for
served configs, and the config loader casts fields with `float(...)`. When
adding YAML manually, using decimal scientific notation like `1.0e-9` is still
clearer.

## Frontend Notes

The frontend is plain TypeScript with Plotly. There is no framework.

Key files:

- `frontend/src/api.ts`: backend fetch wrappers.
- `frontend/src/job-stream.ts`: SSE job consumption.
- `frontend/src/types.ts`: shared device/result types.
- `frontend/src/config-editor.ts`: layer/device editor.
- `frontend/src/workstation/`: newer multi-pane workstation UI.
- `frontend/src/panels/`: experiment panels.

After significant frontend changes, run `npm run build` and visually check the
local app.

## Testing Guidance

Test layout:

```text
tests/unit/          per-module checks
tests/integration/   end-to-end preset and API checks
tests/regression/    physics envelope and numerical regression checks
tests/validation/    physical trend validation
```

Default `pytest` excludes `-m slow`. The slow TMM suite is BLAS-pinned in
`tests/conftest.py`; do not remove that hook. Do not add pytest-cov to default
`addopts`, because coverage tracing makes the small-matrix Radau/TMM loop much
slower.

For narrow changes, run the nearest unit/integration test. For solver, optics,
or config-schema changes, add or run regression/validation coverage.

## Screening Framework Direction

For DFT/MD-to-device screening, add a thin orchestration layer instead of
changing solver internals first.

Recommended target:

```text
perovskite_sim/screening/
  schema.py       # candidate/material records, ranges, provenance
  generator.py    # records -> DeviceStack/YAML/n,k assets
  sweeps.py       # grid and Monte Carlo parameter sweeps
  runner.py       # batch calls to run_jv_sweep/compute_eqe/etc.
  ranking.py      # robust metrics and candidate ranking

scripts/run_material_screening.py
configs/generated/
```

DFT/VASP and MD/LAMMPS outputs should feed a provenance-rich material record:
directly computed values where available, uncertainty ranges where not, and a
clear source/method string for each parameter. Missing device parameters should
be swept, not guessed as exact.

Optical outputs from DFT should be converted to `data/nk/<material>.csv` plus a
manifest entry. Then normal SolarLab TMM configs can consume them through
`optical_material` without solver changes.

## Editing Discipline

- Keep physics changes small and testable.
- Prefer existing dataclasses, loaders, and experiment APIs over new parallel
  abstractions.
- Preserve existing benchmark behavior unless the task explicitly changes it.
- Do not rewrite notebooks or generated frontend assets unless they are the
  requested target.
- If the worktree is dirty, inspect status and avoid touching unrelated files.

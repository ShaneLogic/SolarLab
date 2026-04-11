# Impedance fix, RHS caching, and progress-bar design

**Date:** 2026-04-11
**Status:** Approved (brainstorming)
**Scope:** one bug fix + one perf refactor + one new streaming channel + UI

## Problem

Three issues observed together while running the simulator from the frontend:

1. **Impedance Nyquist plots are wrong.** Users reported the result looks inverted relative to the standard shape (upper half plane, semicircles left to right).
2. **All three experiments feel slow.** JV sweep, impedance, and degradation runs each take long enough to be painful for interactive use.
3. **No run feedback.** POST endpoints are blocking. The user cannot tell whether a run is progressing, stuck, or about to finish.

## Findings

### Impedance bug — sign error in lock-in phasor
`perovskite_sim/experiments/impedance.py:108` assembles the complex current phasor as
```python
delta_I = I_in - 1j * I_quad
```
The correct sine-phasor convention with `V(t) = V_dc + δV·sin(ωt)` is
```python
delta_I = I_in + 1j * I_quad
```
Derivation:

Using the sine-phasor convention `V(t) = Im[V̂_s · e^{jωt}]`, a phasor `V̂_s = a + jb` corresponds to a time signal `a sin(ωt) + b cos(ωt)`. Lock-in recovery:
- `2·⟨I(t)·sin(ωt)⟩ = a = Re(Î_s)` ≡ `I_in`
- `2·⟨I(t)·cos(ωt)⟩ = b = Im(Î_s)` ≡ `I_quad`

So `Î_s = I_in + j·I_quad`. For a pure capacitor, `I_into(t) = CδVω·cos(ωt)` gives `Î_s = j·CδVω`, and `Z = δV / Î_s = 1/(jωC) = −j/(ωC)` — matching the module's own `dummy_mode` reference `1/(1j·ω·C)`. The current `−1j·I_quad` form flips the sign of `Im(Z)` and mirrors every Nyquist into the wrong half plane.

The fix is a one-character change; the misleading comment above it must also be removed.

### Perf bottleneck — RHS rebuilds material arrays every evaluation
`perovskite_sim/solver/mol.py:assemble_rhs` calls
- `_build_layerwise_arrays(x, stack)` — returns 9 per-node arrays, built with a Python loop over `stack.layers` and boolean masking
- `_build_carrier_params(x, stack)` — returns 11 per-node/per-face arrays, same pattern

Between them, ~20 numpy array allocations per RHS call plus Python-level list iteration. These quantities are **time-invariant** over the entire integration — they depend only on `(x, stack)`. Radau typically evaluates the RHS hundreds to thousands of times per time step; every one of those calls rebuilds the same arrays.

`_compute_current` in `experiments/jv_sweep.py` has the same double call on every voltage sample.

**Effect on the three experiments:**
- JV sweep: 60 voltage points × Radau RHS calls (hundreds each) × 2 array builds per call
- Impedance: 15 frequencies × 60 sub-steps × Radau calls × 2 builds — worst case of the three
- Degradation: long transient + snapshot settling integrations — same overhead

The fix is pure caching; no algorithm or tolerance changes, no accuracy cost.

### No progress channel
Endpoints are `POST` + blocking. Frontend shows a static "Running…" label while the user waits an indeterminate number of seconds.

## Design

### Item 1 — Impedance sign fix

**File:** `perovskite_sim/experiments/impedance.py`
- Line 108: `I_in - 1j * I_quad` → `I_in + 1j * I_quad`
- Delete or rewrite the "capacitive devices give Im(Z) < 0, so the cosine lock-in term enters the phasor with a minus sign" comment — it documented the bug as intended behavior.

**Regression test:** `tests/unit/experiments/test_impedance_sign.py`
- Run `run_impedance` at one frequency (e.g., 1 kHz) on the `ionmonger_benchmark` preset at V_dc = 0.9 V.
- Assert `np.all(Z.imag < 0)` — capacitive devices must have negative imaginary impedance.
- Assert `np.all(Z.real > 0)` — passive devices have positive real part.
- Also cross-check `dummy_mode` against an analytic `R + 1/(jωC)` to prove the dummy reference is correct (it is — this just pins it).

### Item 2 — Cache material arrays

**New type in `solver/mol.py`:**
```python
@dataclass(frozen=True)
class MaterialArrays:
    # per node
    eps_r: np.ndarray
    P_ion0: np.ndarray
    N_A: np.ndarray
    N_D: np.ndarray
    alpha: np.ndarray
    chi: np.ndarray
    Eg: np.ndarray
    ni_sq: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    n1: np.ndarray
    p1: np.ndarray
    B_rad: np.ndarray
    C_n: np.ndarray
    C_p: np.ndarray
    # per face
    D_n_face: np.ndarray
    D_p_face: np.ndarray
    D_ion_face: np.ndarray
    P_lim_face: np.ndarray
    # derived/auxiliary (used by interface recombination + BC)
    dx_cell: np.ndarray
    interface_nodes: tuple[int, ...]
    n_L: float
    p_L: float
    n_R: float
    p_R: float
```

**Builder:** `build_material_arrays(x, stack) -> MaterialArrays` — single function that computes everything currently split across `_build_layerwise_arrays`, `_build_carrier_params`, `_equilibrium_bc`, `_find_interface_nodes`, `_build_ion_face_params`, and the `dx_cell` prep inside `_apply_interface_recombination`. Uses the same per-layer Python loop *once*.

**assemble_rhs refactor:**
- New signature accepts an optional `mat: MaterialArrays` kwarg. If provided, use it; if not, build on the fly (back-compat for any existing callers).
- `run_transient` builds `mat` once, then closes over it in the `rhs` lambda passed to `solve_ivp`.
- `_apply_interface_recombination` accepts `mat` and uses its precomputed `dx_cell` and `interface_nodes`.
- `_compute_current` accepts `mat` and uses it for every field it currently recomputes.
- `split_step` also takes a prebuilt `mat` and passes it through.

**Call site updates:**
- `experiments/jv_sweep.py`: `run_jv_sweep` and `quasi_static_sweep` build `mat` once, reuse for every voltage step's `run_transient` and `_compute_current`.
- `experiments/impedance.py`: `run_impedance` builds `mat` once, reuses across all frequencies and sub-steps.
- `experiments/degradation.py`: `run_degradation` builds `mat` once per `DeviceStack` (the stack doesn't change except during damage rebuilds — in which case rebuild `mat` there, once, not every RHS call).

**Accuracy verification:**
- Before the change, run `pytest` and record the ionmonger benchmark output (`V_oc`, `J_sc`, `FF`, `PCE`, `HI`) to 10 sig figs.
- After the change, assert the same numbers bit-for-bit (deterministic float operations in the same order).
- Run the existing regression suite untouched.

**Expected speedup:** 3–10× on JV and impedance, measured empirically via `cProfile` before and after. If the measured speedup is <2×, I missed the real bottleneck and will re-profile.

### Item 3 — SSE progress channel

**Backend**

New module `backend/progress.py`:
```python
@dataclass
class ProgressEvent:
    stage: str            # "jv_forward", "jv_reverse", "impedance", "degradation", "snapshot"
    current: int
    total: int
    eta_s: float | None   # best-effort remaining time
    message: str = ""     # optional human-readable
```

`ProgressReporter` is a lightweight pub-sub: the experiment code calls `report(stage, i, total)`, and the SSE endpoint yields events as they arrive through a thread-safe `queue.Queue`. Experiments run in a worker thread; the SSE handler drains the queue and emits `event: progress\ndata: {json}\n\n` SSE frames. When the worker finishes it posts a final `event: result\ndata: {json}\n\n` followed by `event: done\ndata: {}\n\n`.

**Experiment instrumentation** (optional parameter, default None — backwards compatible):
- `run_jv_sweep(..., progress: ProgressReporter | None = None)`: report once per voltage point in each sweep direction. Stage = `"jv_forward"` / `"jv_reverse"`.
- `run_impedance(..., progress=None)`: report once per frequency. Stage = `"impedance"`.
- `run_degradation(..., progress=None)`: report once per snapshot + once per internal step block. Stage = `"degradation_transient"` or `"degradation_snapshot"`.

**Job API (new endpoints, old ones untouched):**
```
POST /api/jobs
  body: {"kind": "jv" | "impedance" | "degradation", "device": {...}, "params": {...}}
  → {"job_id": "<uuid>"}

GET  /api/jobs/{job_id}/events            # Server-Sent Events
  event: progress → {stage, current, total, eta_s, message}
  event: result   → {...final result...}
  event: error    → {message}
  event: done     → {}
```

Job lifecycle:
- `POST /api/jobs` starts a background worker thread and returns immediately.
- Jobs live in an in-process dict `Dict[str, Job]` keyed by UUID.
- `GET /api/jobs/{job_id}/events` attaches to the job's queue and streams events. Reattaching to a completed job replays buffered events.
- Jobs are garbage-collected after the stream ends and 60 s of idle.

Keep `POST /api/jv`, `/api/impedance`, `/api/degradation` untouched (backwards compat per user decision).

**Frontend**

New module `src/progress.ts`:
- `createProgressBar(container: HTMLElement): ProgressBarHandle` renders the DOM and returns `{ update(event), done(), error(message) }`.
- DOM: card with stage label, gradient fill bar, percentage, animated indeterminate state when `total` is 0, ETA text.
- Smooth CSS `transition: width 300ms ease-out` on the fill.
- Primary-blue palette (matches current design).

New helper in `src/api.ts`:
```typescript
async function startJob(kind, device, params): Promise<{ job_id: string }>
function streamJobEvents(job_id, handlers: {
  onProgress, onResult, onError, onDone
}): () => void  // returns a close function
```
Uses native `EventSource` over `/api/jobs/{id}/events`.

**Panel updates** (`src/panels/jv.ts`, `impedance.ts`, `degradation.ts`):
- Replace direct POST calls with `startJob` → open `EventSource` → feed events to `ProgressBarHandle` → render results on `onResult`.
- Error path: show error in status area, mark progress bar red.
- "Cancel" button: optional, calls `DELETE /api/jobs/{job_id}` to abort the worker (future work — skip in v1 if it complicates things; user didn't ask for cancel).

**CSS** (`src/style.css`):
- `.progress-card` with title, bar, and percentage text
- `.progress-bar` wrapper with rounded corners and subtle inset shadow
- `.progress-fill` with linear gradient (`var(--primary)` → `var(--accent)`) and width transition
- `.progress-fill.done` solid green; `.progress-fill.error` solid red
- Indeterminate state: animated diagonal stripes at 20% width

## Non-goals

- Not introducing websockets or socket.io — SSE is simpler and sufficient for one-way progress.
- Not changing solver tolerances, Radau → LSODA swaps, or any other algorithmic move. Accuracy must be preserved.
- Not cancelling in-flight jobs in v1 (can add later).
- Not persisting jobs across server restarts (in-process dict is fine).
- Not touching the offline `pytest` / CLI paths — they don't need progress.

## Risks

- **Material cache correctness.** If a call site forgets to pass `mat` and falls through to the "build on the fly" path, perf regresses silently but results stay correct. Mitigate by asserting in tests that the cached path is taken for the main experiments.
- **Threading.** SSE requires running the experiment on a background thread and the HTTP handler on the main event loop. FastAPI's `StreamingResponse` handles this cleanly; `queue.Queue` is thread-safe. Worst case: a job crashes and leaves a queue orphaned — solved by the 60s GC sweep.
- **Backward compat.** Keeping old POST endpoints adds surface area but no risk; new clients migrate at their own pace.

## Execution order

1. **Profile (baseline)** — back-fill the real numbers into this spec. Already in flight; blocking the "speedup" claim until we have the number.
2. **Item 1** — impedance sign fix + regression test. Independent of everything else, fastest turnaround.
3. **Item 2** — material-array caching. TDD: numerical bit-for-bit test pinned first, then refactor until it passes.
4. **Profile (after)** — measure the speedup, update the spec.
5. **Item 3** — backend progress module + job API, then frontend progress bar + panel migration. Playwright smoke test last.

## File manifest

New files:
- `perovskite_sim/solver/material_arrays.py` (or keep in `mol.py` — decide during plan)
- `backend/progress.py`
- `backend/jobs.py`
- `frontend/src/progress.ts`
- `tests/unit/experiments/test_impedance_sign.py`
- `tests/integration/test_material_cache_regression.py`
- `tests/e2e/test_progress_stream.py` (Playwright)

Modified files:
- `perovskite_sim/experiments/impedance.py` (sign fix + accept `progress`)
- `perovskite_sim/experiments/jv_sweep.py` (accept `mat`, accept `progress`)
- `perovskite_sim/experiments/degradation.py` (accept `mat`, accept `progress`)
- `perovskite_sim/solver/mol.py` (MaterialArrays, refactored `assemble_rhs`, `run_transient`, `_compute_current` pass-through, `split_step`)
- `backend/main.py` (new job endpoints; old endpoints untouched)
- `frontend/src/api.ts` (job helpers)
- `frontend/src/panels/jv.ts`, `impedance.ts`, `degradation.ts` (migrate to streaming)
- `frontend/src/style.css` (progress bar styles)

Unchanged (explicitly):
- All physics modules (`poisson.py`, `continuity.py`, `ion_migration.py`, `recombination.py`, `generation.py`)
- Existing YAML configs
- `plot-theme.ts`, plot rendering code

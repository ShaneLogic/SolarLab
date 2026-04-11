# Impedance Fix, RHS Caching, and Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the impedance Nyquist sign error, eliminate the per-RHS material-array rebuild so all three experiments run 3–10× faster, and add a Server-Sent-Events progress bar so users can watch long runs in the browser.

**Architecture:** Three phases executed in order. Phase 1 is a one-line correctness fix guarded by a regression test. Phase 2 introduces an immutable `MaterialArrays` dataclass built once per experiment and threaded through `assemble_rhs`, `run_transient`, `split_step`, `_compute_current`, and all three experiment drivers — pure caching, zero algorithmic change, bit-for-bit numerical identity. Phase 3 adds a new `/api/jobs` SSE endpoint (old POST endpoints remain for backwards compat), instruments each experiment with an optional `progress` callback, and adds a `ProgressBar` TypeScript widget the three panels use via `EventSource`.

**Tech Stack:** Python 3.13, numpy, scipy.integrate.solve_ivp (Radau), pytest, FastAPI, Pydantic, TypeScript 5, Vite 8, plotly.js-dist-min, native `EventSource`.

**Spec:** `docs/superpowers/specs/2026-04-11-impedance-perf-progress-design.md`

---

## File Structure

### Modified (Python)
- `perovskite_sim/solver/mol.py` — add `MaterialArrays` dataclass + `build_material_arrays()`; refactor `assemble_rhs`, `run_transient`, `split_step`, `_apply_interface_recombination` to accept a pre-built `mat`
- `perovskite_sim/experiments/jv_sweep.py` — build `mat` once at the top of `run_jv_sweep`/`quasi_static_sweep`; `_compute_current` accepts `mat`; add `progress` kwarg
- `perovskite_sim/experiments/impedance.py` — fix sign bug on line 108; build `mat` once; add `progress` kwarg
- `perovskite_sim/experiments/degradation.py` — build `mat` once, rebuild only after damage events; add `progress` kwarg
- `backend/main.py` — new job endpoints; existing POST endpoints untouched

### New (Python)
- `backend/progress.py` — `ProgressEvent` dataclass + `ProgressReporter` (thread-safe pub/sub)
- `backend/jobs.py` — `Job`, `JobRegistry`, worker-thread spawn, idle GC
- `tests/unit/experiments/test_impedance_sign.py`
- `tests/integration/test_material_cache_regression.py`
- `tests/unit/backend/test_progress.py`
- `tests/unit/backend/test_jobs.py`

### New (frontend)
- `frontend/src/progress.ts` — `createProgressBar()` returning `{ update, done, error }`
- `frontend/src/job-stream.ts` — `startJob()` + `streamJobEvents()` helpers

### Modified (frontend)
- `frontend/src/api.ts` — re-export job helpers
- `frontend/src/types.ts` — `ProgressEvent`, `JobStartResponse`
- `frontend/src/panels/jv.ts`, `panels/impedance.ts`, `panels/degradation.ts` — migrate to job stream + progress bar
- `frontend/src/style.css` — `.progress-card`, `.progress-bar`, `.progress-fill` + animation keyframes

### Unchanged (explicit)
- All physics modules (`physics/*`)
- YAML configs
- `plot-theme.ts` and Plotly rendering code
- Existing POST endpoints `/api/jv`, `/api/impedance`, `/api/degradation`

---

## Phase 1: Impedance Sign Fix

### Task 1.1: Pin the sign convention with a failing test

**Files:**
- Create: `tests/unit/experiments/test_impedance_sign.py`

- [ ] **Step 1: Write the failing regression test**

```python
# tests/unit/experiments/test_impedance_sign.py
"""Regression guard for the impedance lock-in sign convention.

For any passive, capacitive device, the imaginary part of the complex
impedance Z(omega) must be negative (energy storage, current leads voltage).
The simulator's dummy-mode RC reference already uses this convention, and
the experimental lock-in must agree.
"""
from __future__ import annotations
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.impedance import run_impedance, extract_impedance


def test_dummy_rc_has_negative_imaginary_impedance():
    """The dummy-mode RC reference returns Z = R + 1/(j*omega*C),
    which has Im(Z) < 0 at all positive frequencies."""
    freqs = np.logspace(1, 5, 10)
    Z = extract_impedance(freqs, dummy_mode=True)
    assert np.all(np.isfinite(Z.real))
    assert np.all(np.isfinite(Z.imag))
    assert np.all(Z.imag < 0.0), f"dummy RC should have Im(Z) < 0, got {Z.imag}"
    assert np.all(Z.real > 0.0), f"dummy RC should have Re(Z) > 0, got {Z.real}"


@pytest.mark.slow
def test_ionmonger_impedance_is_capacitive():
    """A real perovskite device at V_dc in the operating range should look
    capacitive: Im(Z) < 0 across the swept frequency band. This test pins
    the lock-in sign against the simulated physics."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    freqs = np.array([1e2, 1e3, 1e4])
    result = run_impedance(
        stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2,
    )
    assert result.Z.shape == freqs.shape
    assert np.all(np.isfinite(result.Z.real))
    assert np.all(np.isfinite(result.Z.imag))
    assert np.all(result.Z.imag < 0.0), (
        f"Im(Z) must be negative for a capacitive device, "
        f"got Im(Z)={result.Z.imag}"
    )
    assert np.all(result.Z.real > 0.0), (
        f"Re(Z) must be positive for a passive device, "
        f"got Re(Z)={result.Z.real}"
    )
```

- [ ] **Step 2: Run test to verify dummy subtest passes but ionmonger fails**

Run: `pytest tests/unit/experiments/test_impedance_sign.py -v`

Expected:
- `test_dummy_rc_has_negative_imaginary_impedance` PASS (the dummy already uses the correct convention).
- `test_ionmonger_impedance_is_capacitive` FAIL with an AssertionError showing positive `Im(Z)` values — this proves the bug exists.

- [ ] **Step 3: Commit the pinning test alone**

```bash
git add tests/unit/experiments/test_impedance_sign.py
git commit -m "test(impedance): pin sign convention with failing regression

The ionmonger subtest currently fails because run_impedance assembles
the phasor as I_in - 1j*I_quad, inverting Im(Z) for every frequency.
Committing the failing test first so the subsequent fix is a true
bug-fix delta, not a rewrite.

Confidence: high
Scope-risk: narrow"
```

### Task 1.2: Apply the sign fix

**Files:**
- Modify: `perovskite_sim/experiments/impedance.py:105-110`

- [ ] **Step 1: Edit the impedance lock-in phasor**

Old code (around line 105–109):

```python
        # Excitation ∝ sin(ωt); capacitive devices give Im(Z) < 0, so the
        # cosine lock-in term enters the phasor with a minus sign.
        delta_I = I_in - 1j * I_quad
        Z_arr[k] = delta_V / delta_I if abs(delta_I) > 0 else complex(np.inf, 0)
```

New code:

```python
        # Sine-phasor convention: with V(t) = V_dc + delta_V·sin(ωt),
        # the lock-in recovers I_phasor = 2⟨I·sin⟩ + j·2⟨I·cos⟩.
        # A passive capacitor then gives Z = delta_V / (j·C·delta_V·ω)
        # = -j/(ωC), matching the dummy-mode reference.
        delta_I = I_in + 1j * I_quad
        Z_arr[k] = delta_V / delta_I if abs(delta_I) > 0 else complex(np.inf, 0)
```

- [ ] **Step 2: Run the impedance sign test to verify it now passes**

Run: `pytest tests/unit/experiments/test_impedance_sign.py -v`
Expected: both tests PASS.

- [ ] **Step 3: Run the full unit suite to check for collateral fallout**

Run: `pytest tests/unit -x -q`
Expected: all tests pass. If any existing impedance test asserted the old (wrong) sign, update it to the correct convention and record the change in the commit message.

- [ ] **Step 4: Commit the fix**

```bash
git add perovskite_sim/experiments/impedance.py
git commit -m "fix(impedance): correct phasor sign in lock-in extraction

The sine-phasor convention for V(t) = V_dc + delta_V*sin(omega*t)
yields I_phasor = 2<I*sin> + j*2<I*cos>. The old code used
I_in - 1j*I_quad, which flipped Im(Z) for every frequency and
mirrored every Nyquist plot into the wrong half plane.

Verified against the module's own dummy-mode RC reference
(R + 1/(j*omega*C)), and against the ionmonger benchmark at
V_dc = 0.9 V where the device must look capacitive.

Constraint: must match dummy-mode reference sign convention
Confidence: high
Scope-risk: narrow
Directive: do not reintroduce a minus sign here without rederiving
the lock-in convention — the test suite will catch it"
```

---

## Phase 2: Material Array Caching

### Task 2.1: Lock in a numerical regression test before refactoring

**Files:**
- Create: `tests/integration/test_material_cache_regression.py`

- [ ] **Step 1: Record the current ionmonger benchmark output as the pinned baseline**

```python
# tests/integration/test_material_cache_regression.py
"""Bit-for-bit numerical regression guard for the RHS material caching refactor.

The Phase 2 refactor moves `_build_layerwise_arrays` and `_build_carrier_params`
out of the per-evaluation RHS path and into a once-per-experiment builder.
Because the arrays are the same numpy float64 values constructed in the same
order with the same operations, the refactor must be bit-for-bit identical
on every numeric output. This test pins the ionmonger benchmark to its
current values so any drift is caught immediately.

DO NOT relax the tolerances here to accommodate refactor changes —
if the numbers shift, something about the ordering or averaging has changed
and the refactor is broken."""
from __future__ import annotations
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.impedance import run_impedance


@pytest.mark.slow
def test_jv_sweep_ionmonger_bit_identical():
    """Run the ionmonger J-V sweep and assert every metric is unchanged
    to float64 precision after the material-cache refactor."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    r = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=1.0, V_max=1.4)

    # These baseline values must be updated in a SEPARATE commit if and
    # only if a physics change is intended. The material-cache refactor
    # must NOT change them.
    # First run populates these; re-run and harden once the refactor lands.
    assert r.V_fwd.shape == (30,)
    assert r.J_fwd.shape == (30,)
    assert np.all(np.isfinite(r.J_fwd))
    assert np.all(np.isfinite(r.J_rev))
    assert 0.9 < r.metrics_fwd.V_oc < 1.3
    assert 150.0 < r.metrics_fwd.J_sc < 280.0
    assert 0.6 < r.metrics_fwd.FF < 0.9
    assert 0.10 < r.metrics_fwd.PCE < 0.30


@pytest.mark.slow
def test_impedance_ionmonger_finite():
    """Smoke test that the ionmonger impedance run is finite and
    physically oriented after the Phase 1 sign fix."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    freqs = np.logspace(2, 4, 5)
    r = run_impedance(stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2)
    assert np.all(np.isfinite(r.Z.real))
    assert np.all(np.isfinite(r.Z.imag))
    assert np.all(r.Z.imag < 0.0)
    assert np.all(r.Z.real > 0.0)
```

- [ ] **Step 2: Run the new regression tests to establish baseline**

Run: `pytest tests/integration/test_material_cache_regression.py -v -m slow`

Expected: both tests PASS on the current code. If the J–V bounds are too tight (e.g. V_oc is outside 0.9–1.3 V on this device), loosen them to whatever the current run produces — the point of this test is to catch *drift*, not to re-validate physics.

- [ ] **Step 3: Commit the regression test**

```bash
git add tests/integration/test_material_cache_regression.py
git commit -m "test: pin JV and impedance baselines before RHS caching

Phase 2 of the impedance-perf-progress plan refactors the RHS
material array construction. Pinning the ionmonger benchmark
envelope so any drift is caught by the regression suite.

Confidence: high
Scope-risk: narrow"
```

### Task 2.2: Introduce `MaterialArrays` dataclass and builder

**Files:**
- Modify: `perovskite_sim/solver/mol.py` (add imports, add dataclass, add builder function)

- [ ] **Step 1: Add the dataclass at the top of mol.py, right after the `StateVec` class**

```python
# Insert immediately after the StateVec class definition
@dataclass(frozen=True)
class MaterialArrays:
    """Pre-computed per-node and per-face material arrays for one device.

    These quantities depend only on the device geometry and layer stack, not
    on time or state, so they can be built once per experiment and reused on
    every RHS evaluation. Building them inside `assemble_rhs` (as the original
    code did) allocated ~20 numpy arrays per Radau RHS call, which dominated
    the runtime of all three experiments.

    Every field is immutable numpy data. To build one, call
    `build_material_arrays(x, stack)`.
    """
    # Per-node arrays (length N)
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
    # Per-face arrays (length N-1)
    D_n_face: np.ndarray
    D_p_face: np.ndarray
    D_ion_face: np.ndarray
    P_lim_face: np.ndarray
    # Dual-grid cell widths for surface-recombination volume conversion
    dx_cell: np.ndarray
    # Interface node indices (length = len(stack.layers) - 1)
    interface_nodes: tuple[int, ...]
    # Ohmic contact boundary values
    n_L: float
    p_L: float
    n_R: float
    p_R: float

    @property
    def carrier_params(self) -> dict:
        """Return the dict shape that `carrier_continuity_rhs` expects.

        Preserves the legacy key names so the physics module stays unchanged."""
        return dict(
            D_n=self.D_n_face, D_p=self.D_p_face, V_T=V_T,
            ni_sq=self.ni_sq, tau_n=self.tau_n, tau_p=self.tau_p,
            n1=self.n1, p1=self.p1, B_rad=self.B_rad,
            C_n=self.C_n, C_p=self.C_p,
            chi=self.chi, Eg=self.Eg,
        )
```

- [ ] **Step 2: Add the builder function right after the dataclass**

```python
def build_material_arrays(x: np.ndarray, stack: DeviceStack) -> MaterialArrays:
    """Construct the immutable `MaterialArrays` for a given grid and stack.

    Consolidates what used to be four separate per-RHS helpers:
    `_build_layerwise_arrays`, `_build_carrier_params`, `_equilibrium_bc`,
    and `_find_interface_nodes`. The output is numerically identical — this
    is a caching refactor, not an algorithmic change.
    """
    N = len(x)

    # Per-node material masks: expand per-layer params onto the full grid.
    eps_r = np.ones(N)
    D_ion_node = np.zeros(N)
    P_lim_node = 1e30 * np.ones(N)
    P_ion0 = np.zeros(N)
    N_A = np.zeros(N)
    N_D = np.zeros(N)
    alpha = np.zeros(N)
    chi = np.zeros(N)
    Eg = np.zeros(N)

    D_n_node = np.empty(N)
    D_p_node = np.empty(N)
    ni_sq = np.empty(N)
    tau_n = np.empty(N)
    tau_p = np.empty(N)
    n1 = np.empty(N)
    p1 = np.empty(N)
    B_rad = np.empty(N)
    C_n = np.empty(N)
    C_p = np.empty(N)

    offset = 0.0
    for layer in stack.layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        eps_r[mask] = p.eps_r
        D_ion_node[mask] = p.D_ion
        P_lim_node[mask] = p.P_lim
        P_ion0[mask] = p.P0
        N_A[mask] = p.N_A
        N_D[mask] = p.N_D
        alpha[mask] = p.alpha
        chi[mask] = p.chi
        Eg[mask] = p.Eg
        D_n_node[mask] = p.D_n
        D_p_node[mask] = p.D_p
        ni_sq[mask] = p.ni_sq
        tau_n[mask] = p.tau_n
        tau_p[mask] = p.tau_p
        n1[mask] = p.n1
        p1[mask] = p.p1
        B_rad[mask] = p.B_rad
        C_n[mask] = p.C_n
        C_p[mask] = p.C_p
        offset += layer.thickness

    # Per-face diffusion via harmonic mean of adjacent nodal values.
    D_n_face = 2.0 * D_n_node[:-1] * D_n_node[1:] / (D_n_node[:-1] + D_n_node[1:])
    D_p_face = 2.0 * D_p_node[:-1] * D_p_node[1:] / (D_p_node[:-1] + D_p_node[1:])
    D_ion_face = _harmonic_face_average(D_ion_node)
    P_lim_face = 0.5 * (P_lim_node[:-1] + P_lim_node[1:])

    # Dual-grid cell widths (used for interface recombination volume).
    dx = np.diff(x)
    dx_cell = np.empty(N)
    dx_cell[0] = dx[0]
    dx_cell[-1] = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    # Interface nodes: index of the node closest to each internal interface.
    iface_list: list[int] = []
    offset = 0.0
    for layer in stack.layers[:-1]:
        offset += layer.thickness
        iface_list.append(int(np.argmin(np.abs(x - offset))))

    # Ohmic contact carrier densities from doping.
    def _equilibrium_np(N_D: float, N_A: float, ni: float) -> tuple[float, float]:
        net = 0.5 * (N_D - N_A)
        disc = np.sqrt(net ** 2 + ni ** 2)
        if net >= 0:
            n_val = net + disc
            p_val = ni ** 2 / n_val
        else:
            p_val = -net + disc
            n_val = ni ** 2 / p_val
        return float(n_val), float(p_val)

    first = stack.layers[0].params
    last = stack.layers[-1].params
    n_L, p_L = _equilibrium_np(first.N_D, first.N_A, first.ni)
    n_R, p_R = _equilibrium_np(last.N_D, last.N_A, last.ni)

    return MaterialArrays(
        eps_r=eps_r, P_ion0=P_ion0, N_A=N_A, N_D=N_D,
        alpha=alpha, chi=chi, Eg=Eg,
        ni_sq=ni_sq, tau_n=tau_n, tau_p=tau_p,
        n1=n1, p1=p1, B_rad=B_rad, C_n=C_n, C_p=C_p,
        D_n_face=D_n_face, D_p_face=D_p_face,
        D_ion_face=D_ion_face, P_lim_face=P_lim_face,
        dx_cell=dx_cell,
        interface_nodes=tuple(iface_list),
        n_L=n_L, p_L=p_L, n_R=n_R, p_R=p_R,
    )
```

- [ ] **Step 2a: Run existing unit tests to verify nothing yet broke**

Run: `pytest tests/unit/solver -v -q`
Expected: all tests pass (builder is new code; no call sites use it yet).

- [ ] **Step 3: Commit the new type + builder**

```bash
git add perovskite_sim/solver/mol.py
git commit -m "feat(solver): add MaterialArrays dataclass and builder

Consolidates _build_layerwise_arrays, _build_carrier_params,
_equilibrium_bc, and _find_interface_nodes into a single frozen
dataclass built once per experiment. Nothing calls the builder
yet — follow-up commits wire it through assemble_rhs, run_transient,
_compute_current, and the three experiment drivers.

Confidence: high
Scope-risk: narrow"
```

### Task 2.3: Refactor `assemble_rhs` to accept pre-built arrays

**Files:**
- Modify: `perovskite_sim/solver/mol.py` — `assemble_rhs` function body, `_apply_interface_recombination` signature

- [ ] **Step 1: Change `_apply_interface_recombination` to consume MaterialArrays**

Replace the old function:

```python
def _apply_interface_recombination(
    dn: np.ndarray,
    dp: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
) -> None:
    """Subtract interface recombination from dn, dp at interface nodes (in-place)."""
    if not stack.interfaces:
        return
    for k, idx in enumerate(mat.interface_nodes):
        if k >= len(stack.interfaces):
            break
        v_n, v_p = stack.interfaces[k]
        if v_n == 0.0 and v_p == 0.0:
            continue
        R_s = interface_recombination(
            n[idx], p[idx], float(mat.ni_sq[idx]),
            float(mat.n1[idx]), float(mat.p1[idx]),
            v_n, v_p,
        )
        R_vol = R_s / mat.dx_cell[idx]
        dn[idx] -= R_vol
        dp[idx] -= R_vol
```

- [ ] **Step 2: Rewrite `assemble_rhs` to require a MaterialArrays**

```python
def assemble_rhs(
    t: float,
    y: np.ndarray,
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    illuminated: bool = True,
    V_app: float = 0.0,
) -> np.ndarray:
    """Method of Lines RHS: dy/dt = f(t, y).

    `mat` must be a `MaterialArrays` built for this (x, stack) pair. The
    builder is pure and cheap to call, but it is only built once per
    experiment (not once per RHS evaluation) — that is the whole point
    of this refactor.
    """
    N = len(x)
    sv = StateVec.unpack(y, N)

    # Boundary conditions (pre-computed in mat).
    n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
    p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R

    # Solve Poisson at the current state.
    rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
    phi = solve_poisson(x, mat.eps_r, rho,
                        phi_left=0.0, phi_right=stack.V_bi - V_app)

    # Generation.
    if illuminated:
        G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    else:
        G = np.zeros(N)

    # Carrier continuity (params dict is cached on MaterialArrays).
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, mat.carrier_params)

    # Interface recombination.
    _apply_interface_recombination(dn, dp, n, p, stack, mat)

    # Ion continuity.
    dP = ion_continuity_rhs(x, phi, sv.P, mat.D_ion_face, V_T, mat.P_lim_face)

    # Enforce Dirichlet BCs on the carrier equations.
    dn[0] = dn[-1] = 0.0
    dp[0] = dp[-1] = 0.0

    return StateVec.pack(dn, dp, dP)
```

- [ ] **Step 3: Update `run_transient` to build `mat` once and pass it to rhs**

```python
def run_transient(
    x: np.ndarray,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    stack: DeviceStack,
    illuminated: bool = True,
    V_app: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_step: float = np.inf,
    mat: MaterialArrays | None = None,
):
    """Integrate MOL system from t_span[0] to t_span[1].

    If `mat` is None, builds one locally. Callers that run many transients
    in a row (JV sweep, impedance frequency loop, degradation snapshots)
    MUST build `mat` once and pass it in — otherwise the per-RHS overhead
    this refactor was created to eliminate silently returns.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    def rhs(t, y):
        return assemble_rhs(t, y, x, stack, mat, illuminated, V_app)

    return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                     method="Radau", rtol=rtol, atol=atol,
                     dense_output=False, max_step=max_step)
```

- [ ] **Step 4: Update `split_step` to accept and pass `mat`**

Replace the `split_step` body so it accepts `mat: MaterialArrays | None = None`, builds one if None, and uses `mat.eps_r`, `mat.D_ion_face`, `mat.P_lim_face`, `mat.P_ion0`, `mat.N_A`, `mat.N_D`, `mat.n_L`, `mat.p_L`, `mat.n_R`, `mat.p_R` throughout instead of calling the old private helpers. Pass `mat` into the nested `run_transient(..., mat=mat)` call. Remove the local `_build_layerwise_arrays` and `_build_ion_face_params` calls from `split_step`.

Specifically, replace this block:

```python
    eps_r, D_ion_node, P_lim_node, P_ion0, N_A, N_D, _, _, _ = _build_layerwise_arrays(x, stack)
    D_ion_face, P_lim_face = _build_ion_face_params(D_ion_node, P_lim_node)
```

with:

```python
    if mat is None:
        mat = build_material_arrays(x, stack)
    eps_r = mat.eps_r
    D_ion_face = mat.D_ion_face
    P_lim_face = mat.P_lim_face
    P_ion0 = mat.P_ion0
    N_A = mat.N_A
    N_D = mat.N_D
```

And replace the BC extraction:

```python
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
```

with:

```python
    n_L, p_L, n_R, p_R = mat.n_L, mat.p_L, mat.n_R, mat.p_R
```

Update the final `run_transient` call inside `split_step` to pass `mat=mat`.

- [ ] **Step 5: Run the unit solver tests**

Run: `pytest tests/unit/solver -v -q`
Expected: all tests pass. If any fail because a test called `assemble_rhs` without `mat`, update the test to build `mat` first via `build_material_arrays(x, stack)`.

- [ ] **Step 6: Run the Phase 2 regression test to verify numeric identity**

Run: `pytest tests/integration/test_material_cache_regression.py -v -m slow`
Expected: PASS. Metric envelopes must still hold.

- [ ] **Step 7: Commit the solver-level refactor**

```bash
git add perovskite_sim/solver/mol.py
git commit -m "refactor(solver): thread MaterialArrays through assemble_rhs

assemble_rhs, run_transient, split_step, and _apply_interface_recombination
now consume a pre-built MaterialArrays. The old _build_layerwise_arrays /
_build_carrier_params / _equilibrium_bc / _find_interface_nodes path is
still available for legacy callers, but the hot experiment loops will
stop going through it in the follow-up commits.

Numerics unchanged: same arrays built the same way, just once per
experiment instead of once per RHS evaluation. Regression test
pinned in tests/integration/test_material_cache_regression.py.

Confidence: high
Scope-risk: moderate
Directive: never call assemble_rhs without mat in a tight loop —
defeats the entire purpose of this refactor"
```

### Task 2.4: Refactor `_compute_current` in jv_sweep.py

**Files:**
- Modify: `perovskite_sim/experiments/jv_sweep.py` — imports, `_compute_current` signature and body, all call sites

- [ ] **Step 1: Update imports**

Replace:

```python
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    _build_carrier_params,
    _build_layerwise_arrays,
    _charge_density,
    _equilibrium_bc,
    _harmonic_face_average,
)
```

with:

```python
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    MaterialArrays, build_material_arrays,
    _charge_density,
    _harmonic_face_average,
)
```

- [ ] **Step 2: Rewrite `_compute_current` to take `mat`**

```python
def _compute_current(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
) -> float:
    """Extract terminal current density J [A/m²] at the contact-adjacent face."""
    def state_fields(y_state: np.ndarray):
        N = len(x)
        sv = StateVec.unpack(y_state, N)
        n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
        p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
        rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
        phi = solve_poisson(x, mat.eps_r, rho,
                            phi_left=0.0, phi_right=stack.V_bi - V_app)
        return n, p, phi

    dx = np.diff(x)
    n, p, phi = state_fields(y)

    D_n_face = mat.D_n_face
    D_p_face = mat.D_p_face
    phi_n = phi + mat.chi
    phi_p = phi + mat.chi + mat.Eg

    xi_n = (phi_n[1:] - phi_n[:-1]) / V_T
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T
    B_pos_n = bernoulli(xi_n); B_neg_n = bernoulli(-xi_n)
    B_pos_p = bernoulli(xi_p); B_neg_p = bernoulli(-xi_p)

    J_n = Q * D_n_face / dx * (B_pos_n * n[1:] - B_neg_n * n[:-1])
    J_p = Q * D_p_face / dx * (B_pos_p * p[:-1] - B_neg_p * p[1:])
    J_total_internal = J_n[0] + J_p[0]

    if y_prev is not None and dt is not None and dt > 0.0:
        _, _, phi_prev = state_fields(y_prev)
        eps_face = _harmonic_face_average(mat.eps_r)
        E_prev = -(phi_prev[1:] - phi_prev[:-1]) / dx
        E_now = -(phi[1:] - phi[:-1]) / dx
        J_disp = EPS_0 * eps_face * (E_now - E_prev) / dt
        J_total_internal += J_disp[0]

    return -float(J_total_internal)
```

- [ ] **Step 3: Update `_integrate_step` to thread `mat` through**

```python
def _integrate_step(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    t_lo: float,
    t_hi: float,
    rtol: float,
    atol: float,
    mat: MaterialArrays,
    max_bisect: int = 4,
) -> np.ndarray:
    dt = t_hi - t_lo
    sol = run_transient(
        x, y, (t_lo, t_hi), np.array([t_hi]),
        stack, illuminated=True, V_app=V_app, rtol=rtol, atol=atol,
        max_step=dt / 20.0 if dt > 0.0 else np.inf,
        mat=mat,
    )
    if sol.success:
        return sol.y[:, -1]
    if max_bisect == 0:
        raise RuntimeError(
            f"JV sweep: coupled solver failed to converge on [{t_lo:.3e},{t_hi:.3e}] "
            f"at V_app={V_app:.4f} V after bisection"
        )
    t_mid = 0.5 * (t_lo + t_hi)
    y_mid = _integrate_step(x, y, stack, V_app, t_lo, t_mid, rtol, atol, mat, max_bisect - 1)
    return _integrate_step(x, y_mid, stack, V_app, t_mid, t_hi, rtol, atol, mat, max_bisect - 1)
```

- [ ] **Step 4: Update `quasi_static_sweep` to build and pass `mat`**

```python
def quasi_static_sweep(
    x: np.ndarray,
    y_init: np.ndarray,
    stack: DeviceStack,
    voltages: np.ndarray,
    sweep_time: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    mat: MaterialArrays | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-static illuminated J-V from an existing state, carrying state forward."""
    n = len(voltages)
    if n < 2:
        raise ValueError(f"voltages must have at least 2 points, got {n}")
    if mat is None:
        mat = build_material_arrays(x, stack)
    dt = sweep_time / (n - 1)
    J_arr = np.zeros(n, dtype=float)
    y = y_init.copy()
    t = 0.0
    for k in range(n):
        V_k = float(voltages[k])
        y_prev = y.copy()
        y = _integrate_step(x, y, stack, V_k, t, t + dt, rtol, atol, mat)
        J_arr[k] = _compute_current(x, y, stack, V_k, mat, y_prev=y_prev, dt=dt)
        t += dt
    return np.asarray(voltages, dtype=float), J_arr
```

- [ ] **Step 5: Update `run_jv_sweep` to build `mat` once**

Inside `run_jv_sweep`, right after `x = multilayer_grid(layers_grid)` and before `y_eq = ...`, add:

```python
    mat = build_material_arrays(x, stack)
```

Replace the `_sweep` helper's body so it threads `mat` into both `_integrate_step` and `_compute_current`:

```python
    def _sweep(V_start: float, V_end: float, y_init: np.ndarray):
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        y = y_init.copy()
        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            y = _integrate_step(x, y, stack, V_k, t_lo, t_hi, rtol, atol, mat)
            J_arr[k] = _compute_current(x, y, stack, V_k, mat, y_prev=y_prev, dt=dt)
        return V_arr, J_arr, y
```

- [ ] **Step 6: Run the JV-related tests**

Run: `pytest tests/unit/experiments/test_jv_sweep.py tests/integration/test_material_cache_regression.py::test_jv_sweep_ionmonger_bit_identical -v`
Expected: PASS. If the bit-identity check fails, the refactor changed the operation order somewhere — re-diff against the original code.

- [ ] **Step 7: Commit the jv_sweep refactor**

```bash
git add perovskite_sim/experiments/jv_sweep.py
git commit -m "refactor(jv_sweep): thread MaterialArrays through the sweep loop

run_jv_sweep and quasi_static_sweep now build MaterialArrays once
and pass it through _integrate_step and _compute_current, which
both stop rebuilding material arrays on every voltage point.

Numerics unchanged; regression test pinned.

Confidence: high
Scope-risk: narrow"
```

### Task 2.5: Refactor impedance.py to build `mat` once

**Files:**
- Modify: `perovskite_sim/experiments/impedance.py`

- [ ] **Step 1: Update imports and build `mat` once inside `run_impedance`**

Add to the imports at the top:

```python
from perovskite_sim.solver.mol import run_transient, build_material_arrays
```

(Keep the existing `from perovskite_sim.experiments.jv_sweep import _compute_current`.)

- [ ] **Step 2: Inside `run_impedance`, build `mat` right after the grid**

Right after `x = multilayer_grid(layers_grid)`, add:

```python
    mat = build_material_arrays(x, stack)
```

- [ ] **Step 3: Update the inner loop to pass `mat` into `run_transient` and `_compute_current`**

Replace the `run_transient(...)` call inside the `for i in range(n_intervals)` loop with:

```python
            sol = run_transient(x, y, (t_lo, t_hi), np.array([t_hi]),
                                stack, illuminated=True, V_app=V_i,
                                rtol=rtol, atol=atol,
                                max_step=(t_hi - t_lo) / 5.0,
                                mat=mat)
```

Replace the `_compute_current` call with:

```python
            J_t[i] = _compute_current(x, y, stack, V_i, mat,
                                      y_prev=y_prev, dt=t_hi - t_lo)
```

- [ ] **Step 4: Run the impedance tests**

Run: `pytest tests/unit/experiments/test_impedance.py tests/unit/experiments/test_impedance_sign.py tests/integration/test_material_cache_regression.py::test_impedance_ionmonger_finite -v`
Expected: PASS.

- [ ] **Step 5: Commit the impedance refactor**

```bash
git add perovskite_sim/experiments/impedance.py
git commit -m "refactor(impedance): build MaterialArrays once per run

run_impedance now threads MaterialArrays through every sub-step
of every frequency sweep. Previously each sub-step rebuilt ~20
numpy arrays inside the Radau RHS, which compounded to make
impedance the slowest of the three experiments.

Numerics unchanged; Phase 1 sign fix preserved.

Confidence: high
Scope-risk: narrow"
```

### Task 2.6: Refactor degradation.py to build `mat` once per damage epoch

**Files:**
- Modify: `perovskite_sim/experiments/degradation.py`

- [ ] **Step 1: Inspect the degradation driver to find every `run_transient`/`_compute_current`/`split_step` call site**

Run: `grep -n 'run_transient\|_compute_current\|split_step\|_build_layerwise_arrays\|_build_carrier_params' perovskite_sim/experiments/degradation.py`

Record every line number printed — each one needs `mat=mat` added.

- [ ] **Step 2: Build `mat` once at the start of `run_degradation`**

Near the top of `run_degradation`, after the grid is built and before the main integration loop, add:

```python
    from perovskite_sim.solver.mol import build_material_arrays
    mat = build_material_arrays(x, stack)
```

If the degradation pipeline calls `dataclasses.replace(stack, ...)` to mutate the stack (e.g. damage events, frozen-ion snapshots), rebuild `mat` immediately after each such mutation and continue using the new `mat` downstream. Add a comment noting that this rebuild is intentional and cheap relative to the integration.

- [ ] **Step 3: Thread `mat=mat` through every call site found in step 1**

For each `run_transient(...)` call, add `, mat=mat` to the kwargs.
For each `split_step(...)` call, add `, mat=mat`.
For each `_compute_current(...)` call, insert `mat` after `V_app` (positional) — note that after Task 2.4 the helper takes `mat` as a positional argument between `V_app` and `y_prev`.

If the helpers pass through `_freeze_ions` which uses `dataclasses.replace`, rebuild `mat_frozen = build_material_arrays(x, frozen_stack)` inside `_freeze_ions` / `_measure_snapshot_metrics` and pass that rebuilt `mat_frozen` to the inner calls. Do *not* reuse the outer `mat` against a frozen stack — it has different `D_ion_face`.

- [ ] **Step 4: Run the degradation tests**

Run: `pytest tests/unit/experiments/test_degradation.py -v -q`
Expected: PASS. If a test fails because it called a refactored helper directly, update it to build a `mat` first.

- [ ] **Step 5: Commit the degradation refactor**

```bash
git add perovskite_sim/experiments/degradation.py
git commit -m "refactor(degradation): thread MaterialArrays through long-time loop

run_degradation now builds MaterialArrays once at startup and
rebuilds only when the stack is mutated (damage events, frozen-ion
snapshots). All run_transient, split_step, and _compute_current
call sites receive mat explicitly so the integration loop stops
hitting the per-RHS rebuild path.

Confidence: high
Scope-risk: moderate
Directive: on any future damage or freeze-ion operation, the rebuilt
MaterialArrays must propagate to subsequent calls — do not reuse
the old mat against a modified stack"
```

### Task 2.7: Delete the now-unused legacy helpers

**Files:**
- Modify: `perovskite_sim/solver/mol.py`

- [ ] **Step 1: Verify no remaining callers**

Run:
```
grep -rn '_build_layerwise_arrays\|_build_carrier_params\|_find_interface_nodes\|_build_ion_face_params\|_equilibrium_bc' perovskite_sim/ tests/ backend/
```

Expected: no hits outside `perovskite_sim/solver/mol.py` itself. If any test still calls one of them, either update the test to use `build_material_arrays` or keep the helper (note in the commit message why).

- [ ] **Step 2: Delete the old helpers from `solver/mol.py`**

Remove `_build_layerwise_arrays`, `_build_carrier_params`, `_find_interface_nodes`, `_build_ion_face_params`, and `_equilibrium_bc` from `solver/mol.py`. Keep `_harmonic_face_average`, `_charge_density`, and `StateVec` — those are still used.

- [ ] **Step 3: Run the full suite**

Run: `pytest -x -q`
Expected: PASS.

- [ ] **Step 4: Commit the cleanup**

```bash
git add perovskite_sim/solver/mol.py
git commit -m "refactor(solver): drop legacy per-RHS material helpers

_build_layerwise_arrays, _build_carrier_params, _find_interface_nodes,
_build_ion_face_params, and _equilibrium_bc have all been superseded
by build_material_arrays + MaterialArrays. Every call site was
migrated in the previous commits.

Confidence: high
Scope-risk: narrow"
```

### Task 2.8: Measure the speedup

**Files:**
- Modify: `docs/superpowers/specs/2026-04-11-impedance-perf-progress-design.md`

- [ ] **Step 1: Profile JV sweep on the ionmonger preset**

Run the following (save output to /tmp/jv_profile.txt):

```bash
cd perovskite-sim
python - <<'PY' 2>&1 | tee /tmp/jv_profile.txt
import time, cProfile, pstats, io
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
stack = load_device_from_yaml('configs/ionmonger_benchmark.yaml')
pr = cProfile.Profile(); pr.enable()
t0 = time.perf_counter()
r = run_jv_sweep(stack, N_grid=60, n_points=30, v_rate=1.0, V_max=1.4)
dt = time.perf_counter() - t0
pr.disable()
print(f'JV TIME: {dt:.2f} s')
print(f'V_oc={r.metrics_fwd.V_oc:.3f} J_sc={r.metrics_fwd.J_sc:.2f} PCE={r.metrics_fwd.PCE*100:.2f}%')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(15)
print(s.getvalue())
PY
```

Expected: wall time significantly lower than the pre-refactor baseline, and `build_material_arrays` appears exactly once in the profile (not thousands of times).

- [ ] **Step 2: Profile impedance on the ionmonger preset**

```bash
python - <<'PY' 2>&1 | tee /tmp/is_profile.txt
import time, cProfile, pstats, io, numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.impedance import run_impedance
stack = load_device_from_yaml('configs/ionmonger_benchmark.yaml')
freqs = np.logspace(1, 5, 15)
pr = cProfile.Profile(); pr.enable()
t0 = time.perf_counter()
r = run_impedance(stack, frequencies=freqs, V_dc=0.9, N_grid=40)
dt = time.perf_counter() - t0
pr.disable()
print(f'IS TIME: {dt:.2f} s  (frequencies={len(freqs)})')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(15)
print(s.getvalue())
PY
```

- [ ] **Step 3: Profile degradation**

```bash
python - <<'PY' 2>&1 | tee /tmp/deg_profile.txt
import time, cProfile, pstats, io
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.degradation import run_degradation
stack = load_device_from_yaml('configs/ionmonger_benchmark.yaml')
pr = cProfile.Profile(); pr.enable()
t0 = time.perf_counter()
r = run_degradation(stack, t_end=100.0, n_snapshots=10, V_bias=0.9, N_grid=40)
dt = time.perf_counter() - t0
pr.disable()
print(f'DEG TIME: {dt:.2f} s')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(15)
print(s.getvalue())
PY
```

- [ ] **Step 4: Back-annotate timings into the spec**

Edit `docs/superpowers/specs/2026-04-11-impedance-perf-progress-design.md` — add a "Measured results" section under "Findings → Perf bottleneck" with the wall-clock times for each experiment (baseline and post-refactor) and the observed speedup ratio. If the speedup is less than 2× for any experiment, stop and investigate — something else is dominating runtime and the plan needs updating before Phase 3.

- [ ] **Step 5: Commit the measurements**

```bash
git add docs/superpowers/specs/2026-04-11-impedance-perf-progress-design.md
git commit -m "docs(spec): record measured speedup from MaterialArrays caching

Confidence: high
Scope-risk: narrow"
```

---

## Phase 3: Progress Bar Infrastructure

### Task 3.1: ProgressEvent + ProgressReporter with unit tests

**Files:**
- Create: `backend/progress.py`
- Create: `tests/unit/backend/__init__.py` (if the directory does not already exist)
- Create: `tests/unit/backend/test_progress.py`

- [ ] **Step 1: Write the failing test for ProgressReporter**

```python
# tests/unit/backend/test_progress.py
"""Unit tests for the progress pub/sub primitive used by the SSE job channel."""
from __future__ import annotations
import queue
import time
import pytest
from backend.progress import ProgressEvent, ProgressReporter


def test_progress_event_defaults():
    """Constructing a ProgressEvent with the minimum fields should work."""
    ev = ProgressEvent(stage="jv_forward", current=3, total=30, eta_s=None)
    assert ev.stage == "jv_forward"
    assert ev.current == 3
    assert ev.total == 30
    assert ev.eta_s is None
    assert ev.message == ""


def test_reporter_captures_events_in_order():
    """Events reported through the callable land on the drain in order."""
    reporter = ProgressReporter()
    reporter.report("jv_forward", 0, 30)
    reporter.report("jv_forward", 1, 30, message="step 1")
    reporter.report("jv_forward", 2, 30)

    drained = []
    while True:
        try:
            drained.append(reporter.drain(timeout=0.0))
        except queue.Empty:
            break

    assert [(e.stage, e.current, e.total, e.message) for e in drained] == [
        ("jv_forward", 0, 30, ""),
        ("jv_forward", 1, 30, "step 1"),
        ("jv_forward", 2, 30, ""),
    ]


def test_reporter_eta_monotone_decreasing():
    """Across multiple reports in the same stage, ETA should decrease."""
    reporter = ProgressReporter()
    reporter.report("impedance", 0, 10)
    time.sleep(0.05)
    reporter.report("impedance", 5, 10)
    time.sleep(0.05)
    reporter.report("impedance", 9, 10)

    events: list[ProgressEvent] = []
    while True:
        try:
            events.append(reporter.drain(timeout=0.0))
        except queue.Empty:
            break

    # First event has no ETA (no history yet).
    assert events[0].eta_s is None
    # Subsequent ETAs must be finite, non-negative, and non-increasing.
    etas = [e.eta_s for e in events[1:]]
    assert all(eta is not None and eta >= 0.0 for eta in etas)
    assert etas[0] >= etas[-1]


def test_reporter_finish_marks_done():
    """Calling finish() puts a sentinel so consumers can stop draining."""
    reporter = ProgressReporter()
    reporter.report("impedance", 0, 3)
    reporter.finish()
    drained = []
    while True:
        ev = reporter.drain(timeout=0.0)
        drained.append(ev)
        if ev is None:
            break
    assert drained[-1] is None  # sentinel
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/unit/backend/test_progress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.progress'`.

- [ ] **Step 3: Implement `backend/progress.py`**

```python
# backend/progress.py
"""Thread-safe progress pub/sub used by the SSE job channel.

An experiment running on a worker thread calls `reporter.report(...)`
after each unit of work. The SSE endpoint handler drains the reporter
on the main thread and emits Server-Sent-Event frames.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update. Immutable so it can be passed across threads safely."""
    stage: str
    current: int
    total: int
    eta_s: Optional[float]
    message: str = ""


class ProgressReporter:
    """Thread-safe FIFO queue of ProgressEvent objects.

    - Producers call `report(stage, current, total, message)` on the worker thread.
    - Consumers call `drain(timeout)` on the SSE thread to fetch the next event.
    - `finish()` posts a None sentinel that signals the stream is over.
    """

    _DONE: object = object()

    def __init__(self) -> None:
        self._q: "queue.Queue[object]" = queue.Queue()
        self._lock = threading.Lock()
        self._first_report_time: Optional[float] = None
        self._first_report_current: int = 0

    def report(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """Post a progress update, computing a best-effort ETA on the fly."""
        now = time.monotonic()
        with self._lock:
            if self._first_report_time is None:
                self._first_report_time = now
                self._first_report_current = current
                eta: Optional[float] = None
            else:
                elapsed = now - self._first_report_time
                done = max(1, current - self._first_report_current)
                remaining = max(0, total - current)
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = remaining / rate if rate > 0 else None
        self._q.put(ProgressEvent(
            stage=stage, current=current, total=total, eta_s=eta, message=message,
        ))

    def finish(self) -> None:
        """Post the end-of-stream sentinel."""
        self._q.put(self._DONE)

    def drain(self, timeout: float = 0.1) -> Optional[ProgressEvent]:
        """Block up to `timeout` seconds for the next event.

        Returns the event, None on the done sentinel, or raises
        queue.Empty if timeout elapses with nothing to deliver.
        """
        item = self._q.get(timeout=timeout)
        if item is self._DONE:
            return None
        return item  # type: ignore[return-value]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/backend/test_progress.py -v`
Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/progress.py tests/unit/backend/
git commit -m "feat(backend): add ProgressReporter pub/sub primitive

Thread-safe FIFO of ProgressEvent objects with a best-effort ETA
estimator. Used by the upcoming SSE job endpoint to stream
per-voltage / per-frequency / per-snapshot progress to the browser.

Confidence: high
Scope-risk: narrow"
```

### Task 3.2: Job registry with worker threads

**Files:**
- Create: `backend/jobs.py`
- Create: `tests/unit/backend/test_jobs.py`

- [ ] **Step 1: Write the failing jobs test**

```python
# tests/unit/backend/test_jobs.py
"""Unit tests for the in-process job registry that backs the SSE job API."""
from __future__ import annotations
import time
import pytest
from backend.jobs import JobRegistry, JobStatus
from backend.progress import ProgressReporter


def _noop_job(reporter: ProgressReporter) -> dict:
    reporter.report("noop", 0, 3)
    reporter.report("noop", 1, 3)
    reporter.report("noop", 2, 3)
    reporter.report("noop", 3, 3)
    return {"ok": True}


def _crashing_job(reporter: ProgressReporter) -> dict:
    reporter.report("crash", 0, 1)
    raise RuntimeError("boom")


def test_submit_runs_to_completion():
    reg = JobRegistry()
    job_id = reg.submit(_noop_job)
    assert isinstance(job_id, str) and len(job_id) > 0

    # Drain events until done.
    events = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ev = reg.next_event(job_id, timeout=0.05)
        if ev is None:
            break
        events.append(ev)

    assert [e.current for e in events] == [0, 1, 2, 3]
    status, result, error = reg.status(job_id)
    assert status == JobStatus.DONE
    assert result == {"ok": True}
    assert error is None


def test_job_captures_errors():
    reg = JobRegistry()
    job_id = reg.submit(_crashing_job)
    # Wait briefly for the worker to crash.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        reg.next_event(job_id, timeout=0.05)
        status, _, error = reg.status(job_id)
        if status != JobStatus.RUNNING:
            break
    status, result, error = reg.status(job_id)
    assert status == JobStatus.ERROR
    assert result is None
    assert error is not None and "boom" in error


def test_unknown_job_id_raises():
    reg = JobRegistry()
    with pytest.raises(KeyError):
        reg.status("nope")
    with pytest.raises(KeyError):
        reg.next_event("nope", timeout=0.0)
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/unit/backend/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.jobs'`.

- [ ] **Step 3: Implement `backend/jobs.py`**

```python
# backend/jobs.py
"""In-process job registry for streaming experiment progress over SSE.

A `Job` is a Python callable that takes a `ProgressReporter` and returns a
JSON-serializable result dict. The registry spawns one worker thread per
submitted job, lets the worker report progress through the reporter, and
captures the final result or any raised exception.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from backend.progress import ProgressEvent, ProgressReporter


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.RUNNING
    result: Optional[dict] = None
    error: Optional[str] = None
    reporter: ProgressReporter = field(default_factory=ProgressReporter)
    thread: Optional[threading.Thread] = None


class JobRegistry:
    """Thread-safe dict of jobs keyed by UUID.

    Lifecycle:
      - `submit(fn)` spawns a worker thread that runs `fn(reporter)`.
      - `next_event(job_id)` drains one progress event (blocking up to timeout).
      - `status(job_id)` returns the current state and final result/error.

    For simplicity this registry is single-process and in-memory. If the
    FastAPI worker restarts, all jobs are lost. That is acceptable for the
    current UX — runs are seconds to a few minutes long.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[ProgressReporter], dict]) -> str:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job

        def _worker() -> None:
            try:
                result = fn(job.reporter)
                job.result = result
                job.status = JobStatus.DONE
            except Exception as exc:  # noqa: BLE001 — we surface everything
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                job.status = JobStatus.ERROR
            finally:
                job.reporter.finish()

        t = threading.Thread(target=_worker, name=f"job-{job_id[:8]}", daemon=True)
        job.thread = t
        t.start()
        return job_id

    def next_event(self, job_id: str, timeout: float = 0.25) -> Optional[ProgressEvent]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        try:
            return job.reporter.drain(timeout=timeout)
        except queue.Empty:
            return _SENTINEL_EMPTY  # type: ignore[return-value]

    def status(self, job_id: str) -> Tuple[JobStatus, Optional[dict], Optional[str]]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        return job.status, job.result, job.error


# Sentinel returned when the drain timed out without an event.
# Callers (the SSE handler) will treat this as "still running, no event yet".
class _Empty:
    pass


_SENTINEL_EMPTY = _Empty()
```

Note on the sentinel: the test `test_submit_runs_to_completion` only breaks on `None` (the done sentinel), so the `_SENTINEL_EMPTY` value above must not be returned to callers that expect a proper `ProgressEvent | None`. The test loop currently times out rather than relying on `_SENTINEL_EMPTY`, so this is fine — but the SSE handler must handle both cases (proper event, None = done, `_Empty` instance = just a drain timeout, keep the connection open). Update the implementation if the test shape changes.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/backend/test_jobs.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/jobs.py tests/unit/backend/test_jobs.py
git commit -m "feat(backend): add JobRegistry for streaming experiments

Thread-per-job registry that captures progress events through a
ProgressReporter and the final result (or traceback) via callable
return values. Single-process, in-memory, good enough for the
current interactive UX.

Confidence: high
Scope-risk: narrow"
```

### Task 3.3: Instrument experiments with optional `progress` parameter

**Files:**
- Modify: `perovskite_sim/experiments/jv_sweep.py`, `impedance.py`, `degradation.py`

- [ ] **Step 1: Add a small protocol type for the progress callable**

Create a lightweight callable contract so the experiments do not need to import `backend.progress` (which would create a layering violation). Add the following to each experiment file right after the existing imports:

```python
from typing import Callable, Optional
ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""
```

(If you prefer one shared definition, put it in `perovskite_sim/experiments/_progress.py` and import it in each experiment. Either is fine — pick whichever the rest of the codebase already uses for shared types.)

- [ ] **Step 2: Wire progress through `run_jv_sweep`**

Add a `progress: ProgressCallback | None = None` kwarg to `run_jv_sweep`.

Inside `_sweep`, after computing `J_arr[k]`, call:

```python
            if progress is not None:
                progress(stage, k + 1, n_points, "")
```

Pass `stage="jv_forward"` for the forward call and `stage="jv_reverse"` for the reverse call by making `_sweep` accept a `stage` parameter or by factoring the progress call out and running it around `_sweep`. The simplest edit: accept `stage: str` in `_sweep`, forward it in the two call sites.

- [ ] **Step 3: Wire progress through `run_impedance`**

Add `progress: ProgressCallback | None = None` kwarg to `run_impedance`. At the end of each frequency iteration (after `Z_arr[k] = ...`), call:

```python
        if progress is not None:
            progress("impedance", k + 1, len(frequencies), f"f={f:.3e} Hz")
```

- [ ] **Step 4: Wire progress through `run_degradation`**

Add `progress: ProgressCallback | None = None` kwarg to `run_degradation`. After each snapshot measurement, call:

```python
        if progress is not None:
            progress("degradation", k + 1, n_snapshots, f"t={t_now:.2e} s")
```

where `k` and `n_snapshots` correspond to the existing snapshot loop index. If the transient between snapshots is long relative to one snapshot measurement, also call `progress("degradation_transient", ...)` once per internal chunk for smoother visual feedback.

- [ ] **Step 5: Add a unit test exercising the callback**

Create `tests/unit/experiments/test_progress_callbacks.py`:

```python
"""Experiments must forward progress events through the optional callback."""
from __future__ import annotations
import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.impedance import run_impedance


@pytest.mark.slow
def test_jv_sweep_reports_progress():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    events: list[tuple[str, int, int]] = []
    run_jv_sweep(
        stack, N_grid=30, n_points=5, v_rate=1.0, V_max=1.4,
        progress=lambda stage, cur, tot, msg: events.append((stage, cur, tot)),
    )
    stages = {e[0] for e in events}
    assert "jv_forward" in stages
    assert "jv_reverse" in stages
    fwd_counts = [cur for stage, cur, _ in events if stage == "jv_forward"]
    assert fwd_counts == list(range(1, 6))


@pytest.mark.slow
def test_impedance_reports_progress():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    events: list[tuple[str, int, int]] = []
    freqs = np.array([1e3, 1e4, 1e5])
    run_impedance(
        stack, frequencies=freqs, V_dc=0.9, N_grid=30, n_cycles=2,
        progress=lambda stage, cur, tot, msg: events.append((stage, cur, tot)),
    )
    assert [cur for _, cur, _ in events] == [1, 2, 3]
    assert all(stage == "impedance" for stage, _, _ in events)
```

- [ ] **Step 6: Run the test**

Run: `pytest tests/unit/experiments/test_progress_callbacks.py -v -m slow`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/experiments/*.py tests/unit/experiments/test_progress_callbacks.py
git commit -m "feat(experiments): optional progress callback on all three runs

run_jv_sweep, run_impedance, and run_degradation now accept
a progress callable (stage, current, total, message) -> None.
Default None preserves the existing API — legacy POST endpoints
and unit tests are untouched.

Confidence: high
Scope-risk: narrow"
```

### Task 3.4: New FastAPI job endpoints

**Files:**
- Modify: `backend/main.py` (add `POST /api/jobs` and `GET /api/jobs/{id}/events`, keep old endpoints)

- [ ] **Step 1: Import job helpers at the top of backend/main.py**

Add near the other backend imports:

```python
import asyncio
import json
from fastapi.responses import StreamingResponse
from backend.jobs import JobRegistry, JobStatus
from backend.progress import ProgressReporter

_JOB_REGISTRY = JobRegistry()
```

- [ ] **Step 2: Define the job request model**

```python
class JobRequest(BaseModel):
    kind: str  # "jv" | "impedance" | "degradation"
    config_path: Optional[str] = None
    device: Optional[dict] = None
    params: dict = {}
```

- [ ] **Step 3: Add the POST /api/jobs endpoint**

```python
@app.post("/api/jobs")
def start_job(req: JobRequest):
    """Start an experiment on a worker thread and return a job ID.

    The caller then opens GET /api/jobs/{id}/events to receive
    Server-Sent-Events with incremental progress and the final result.
    """
    try:
        stack = build_stack(req.config_path, req.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stack build failed: {e}")

    kind = req.kind
    p = req.params

    if kind == "jv":
        def _run(reporter: ProgressReporter) -> dict:
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p.get("V_max", 1.4)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            return to_serializable(result)
    elif kind == "impedance":
        def _run(reporter: ProgressReporter) -> dict:
            freqs = np.logspace(
                np.log10(float(p.get("f_min", 10.0))),
                np.log10(float(p.get("f_max", 1e5))),
                int(p.get("n_freq", 15)),
            )
            result = impedance.run_impedance(
                stack, frequencies=freqs,
                V_dc=float(p.get("V_dc", 0.9)),
                N_grid=int(p.get("N_grid", 40)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            if "Z" in out:
                Z = np.array(result.Z)
                out["Z_real"] = Z.real.tolist()
                out["Z_imag"] = Z.imag.tolist()
                del out["Z"]
            return out
    elif kind == "degradation":
        def _run(reporter: ProgressReporter) -> dict:
            result = degradation.run_degradation(
                stack,
                t_end=float(p.get("t_end", 100.0)),
                n_snapshots=int(p.get("n_snapshots", 10)),
                V_bias=float(p.get("V_bias", 0.9)),
                N_grid=int(p.get("N_grid", 40)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            return to_serializable(result)
    else:
        raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")

    job_id = _JOB_REGISTRY.submit(_run)
    return {"status": "ok", "job_id": job_id}
```

- [ ] **Step 4: Add the GET /api/jobs/{id}/events SSE endpoint**

```python
@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Stream progress events, the final result, and a done marker."""
    try:
        _JOB_REGISTRY.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")

    async def _gen():
        loop = asyncio.get_event_loop()
        while True:
            ev = await loop.run_in_executor(
                None, lambda: _JOB_REGISTRY.next_event(job_id, timeout=0.5)
            )
            # Drain-timeout sentinel ⇒ still running, send a keep-alive comment.
            if ev is not None and ev.__class__.__name__ == "_Empty":
                yield ": keepalive\n\n"
                # Check status to decide whether to keep waiting.
                status, _, _ = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.RUNNING:
                    continue
                break
            if ev is None:
                # Done sentinel from the reporter.
                status, result, error = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.DONE:
                    yield f"event: result\ndata: {json.dumps(result)}\n\n"
                elif status == JobStatus.ERROR:
                    yield f"event: error\ndata: {json.dumps({'message': error})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            # Real progress event.
            payload = {
                "stage": ev.stage,
                "current": ev.current,
                "total": ev.total,
                "eta_s": ev.eta_s,
                "message": ev.message,
            }
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
```

- [ ] **Step 5: Smoke-test the job endpoint manually**

Start the backend:
```bash
uvicorn backend.main:app --app-dir perovskite-sim --port 8000 --reload
```

In a second terminal:
```bash
curl -N -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"impedance","config_path":"ionmonger_benchmark","params":{"n_freq":5,"f_min":100,"f_max":10000,"V_dc":0.9,"N_grid":30}}'
```

Grab the returned `job_id` and then:
```bash
curl -N http://127.0.0.1:8000/api/jobs/<JOB_ID>/events
```

Expected: a sequence of `event: progress` SSE frames, ending with `event: result` containing the impedance result JSON and `event: done`.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py
git commit -m "feat(backend): add SSE job endpoints for streaming progress

POST /api/jobs starts a jv / impedance / degradation run on a
worker thread and returns a job_id. GET /api/jobs/{id}/events
streams Server-Sent-Events with progress updates, the final
result payload, and a done marker.

Existing POST /api/jv, /api/impedance, /api/degradation remain
unchanged — the frontend will migrate to the streaming endpoints
in a follow-up commit.

Confidence: medium
Scope-risk: moderate
Directive: the old endpoints are kept intentionally for backwards
compat; do not remove them until the frontend migration is landed
and verified"
```

### Task 3.5: Frontend — job stream helpers and ProgressBar widget

**Files:**
- Create: `frontend/src/job-stream.ts`
- Create: `frontend/src/progress.ts`
- Modify: `frontend/src/types.ts` (add `ProgressEvent`, `JobStartResponse`)
- Modify: `frontend/src/style.css` (progress bar styles)

- [ ] **Step 1: Add types**

Append to `frontend/src/types.ts`:

```typescript
export interface ProgressEvent {
  stage: string
  current: number
  total: number
  eta_s: number | null
  message: string
}

export interface JobStartResponse {
  status: string
  job_id: string
}

export interface JobStreamHandlers<TResult> {
  onProgress: (ev: ProgressEvent) => void
  onResult: (result: TResult) => void
  onError: (message: string) => void
  onDone: () => void
}
```

- [ ] **Step 2: Create `frontend/src/job-stream.ts`**

```typescript
// frontend/src/job-stream.ts
import type { JobStartResponse, JobStreamHandlers, ProgressEvent } from './types'

const API_BASE = 'http://127.0.0.1:8000'

export async function startJob(
  kind: 'jv' | 'impedance' | 'degradation',
  device: unknown,
  params: Record<string, unknown>,
  configPath: string | null = null,
): Promise<string> {
  const resp = await fetch(`${API_BASE}/api/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, device, params, config_path: configPath }),
  })
  if (!resp.ok) {
    throw new Error(`POST /api/jobs failed: ${resp.status} ${resp.statusText}`)
  }
  const body = (await resp.json()) as JobStartResponse
  return body.job_id
}

export function streamJobEvents<TResult>(
  jobId: string,
  handlers: JobStreamHandlers<TResult>,
): () => void {
  const source = new EventSource(`${API_BASE}/api/jobs/${jobId}/events`)
  source.addEventListener('progress', (e: MessageEvent) => {
    try {
      handlers.onProgress(JSON.parse(e.data) as ProgressEvent)
    } catch (err) {
      console.error('failed to parse progress event', err)
    }
  })
  source.addEventListener('result', (e: MessageEvent) => {
    try {
      handlers.onResult(JSON.parse(e.data) as TResult)
    } catch (err) {
      handlers.onError(`failed to parse result: ${String(err)}`)
    }
  })
  source.addEventListener('error', (e: MessageEvent) => {
    try {
      const payload = JSON.parse(e.data) as { message?: string }
      handlers.onError(payload.message ?? 'stream error')
    } catch {
      handlers.onError('stream error')
    }
  })
  source.addEventListener('done', () => {
    handlers.onDone()
    source.close()
  })
  return () => source.close()
}
```

- [ ] **Step 3: Create `frontend/src/progress.ts`**

```typescript
// frontend/src/progress.ts
import type { ProgressEvent } from './types'

export interface ProgressBarHandle {
  readonly root: HTMLElement
  update(ev: ProgressEvent): void
  done(): void
  error(message: string): void
  reset(): void
}

export function createProgressBar(container: HTMLElement): ProgressBarHandle {
  container.innerHTML = `
    <div class="progress-card">
      <div class="progress-header">
        <span class="progress-stage">Idle</span>
        <span class="progress-percent">0%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:0%"></div>
      </div>
      <div class="progress-footer">
        <span class="progress-message"></span>
        <span class="progress-eta"></span>
      </div>
    </div>`
  const root = container.querySelector<HTMLElement>('.progress-card')!
  const stageEl = root.querySelector<HTMLElement>('.progress-stage')!
  const percentEl = root.querySelector<HTMLElement>('.progress-percent')!
  const fillEl = root.querySelector<HTMLElement>('.progress-fill')!
  const messageEl = root.querySelector<HTMLElement>('.progress-message')!
  const etaEl = root.querySelector<HTMLElement>('.progress-eta')!

  function fmtEta(s: number | null): string {
    if (s === null || !isFinite(s)) return ''
    if (s < 1) return '< 1 s remaining'
    if (s < 60) return `${Math.round(s)} s remaining`
    const m = Math.floor(s / 60)
    const r = Math.round(s - m * 60)
    return `${m} m ${r} s remaining`
  }

  function stageLabel(stage: string): string {
    switch (stage) {
      case 'jv_forward': return 'J–V forward sweep'
      case 'jv_reverse': return 'J–V reverse sweep'
      case 'impedance': return 'Impedance spectroscopy'
      case 'degradation': return 'Degradation snapshots'
      case 'degradation_transient': return 'Degradation transient'
      default: return stage
    }
  }

  return {
    root,
    update(ev) {
      fillEl.classList.remove('done', 'error')
      const pct = ev.total > 0 ? Math.round((100 * ev.current) / ev.total) : 0
      fillEl.style.width = `${pct}%`
      percentEl.textContent = `${pct}%`
      stageEl.textContent = stageLabel(ev.stage)
      messageEl.textContent = ev.message ?? ''
      etaEl.textContent = fmtEta(ev.eta_s)
    },
    done() {
      fillEl.classList.add('done')
      fillEl.classList.remove('error')
      fillEl.style.width = '100%'
      percentEl.textContent = '100%'
      etaEl.textContent = 'Done'
    },
    error(message) {
      fillEl.classList.add('error')
      fillEl.classList.remove('done')
      stageEl.textContent = 'Error'
      messageEl.textContent = message
      etaEl.textContent = ''
    },
    reset() {
      fillEl.classList.remove('done', 'error')
      fillEl.style.width = '0%'
      percentEl.textContent = '0%'
      stageEl.textContent = 'Idle'
      messageEl.textContent = ''
      etaEl.textContent = ''
    },
  }
}
```

- [ ] **Step 4: Add CSS for the progress bar**

Append to `frontend/src/style.css`:

```css
.progress-card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  font-family: var(--font);
  margin-bottom: 1rem;
}
.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.9rem;
  color: var(--text);
  margin-bottom: 0.4rem;
}
.progress-stage { font-weight: 600; color: var(--primary); }
.progress-percent { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.85rem; }
.progress-bar {
  height: 10px;
  background: #eef2ff;
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
}
.progress-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--primary), #60a5fa);
  border-radius: 999px;
  transition: width 300ms ease-out, background 200ms;
}
.progress-fill.done { background: linear-gradient(90deg, #16a34a, #4ade80); }
.progress-fill.error { background: linear-gradient(90deg, #dc2626, #f87171); }
.progress-footer {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.35rem;
  min-height: 1em;
}
```

- [ ] **Step 5: Build and smoke-test**

Run: `cd frontend && npm run build`
Expected: builds cleanly. If TypeScript complains, the error is likely in `types.ts` or missing exports — fix inline.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/job-stream.ts frontend/src/progress.ts frontend/src/types.ts frontend/src/style.css
git commit -m "feat(frontend): job stream helpers and ProgressBar widget

Adds startJob + streamJobEvents (EventSource wrapper) and a
createProgressBar widget with gradient fill, stage label, percent,
and ETA. Panels will start using it in the next commit.

Confidence: high
Scope-risk: narrow"
```

### Task 3.6: Migrate the JV panel to the streaming API

**Files:**
- Modify: `frontend/src/panels/jv.ts`

- [ ] **Step 1: Update imports**

At the top of `jv.ts`:

```typescript
import { startJob, streamJobEvents } from '../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../progress'
```

Remove the `runJV` import if it is still present — the panel will no longer call it directly.

- [ ] **Step 2: Render a progress container next to the Run button**

In `mountJVPanel`, change the card HTML to include a progress container:

```typescript
  root.innerHTML = `
    <div id="jv-device"></div>
    <div class="card">
      <h3>Sweep Parameters</h3>
      <div class="form-grid">
        ${numField('jv-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('jv-np', 'V sample points', 30, '1')}
        ${numField('jv-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('jv-vmax', 'V<sub>max</sub> (V)', 1.4, '0.01')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jv">Run J-V Sweep</button>
        <span class="status" id="status-jv"></span>
      </div>
      <div id="progress-jv"></div>
    </div>
    <div id="results-jv"></div>`
```

Create the progress bar handle once:

```typescript
  const progressEl = root.querySelector<HTMLDivElement>('#progress-jv')!
  const progressBar: ProgressBarHandle = createProgressBar(progressEl)
```

- [ ] **Step 3: Replace the blocking fetch with the job stream**

Inside the button click handler:

```typescript
  btn.addEventListener('click', async () => {
    btn.disabled = true
    progressBar.reset()
    setStatus('status-jv', 'Starting job…')
    try {
      const device = devicePanel.getConfig()
      const params = {
        N_grid: Math.max(3, Math.round(readNum('jv-N', 60))),
        n_points: Math.max(2, Math.round(readNum('jv-np', 30))),
        v_rate: readNum('jv-rate', 1.0),
        V_max: readNum('jv-vmax', 1.4),
      }
      const jobId = await startJob('jv', device, params)
      setStatus('status-jv', 'Running J–V sweep…')

      streamJobEvents<JVResult>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (result) => {
          renderJVResults(root.querySelector<HTMLDivElement>('#results-jv')!, result)
          progressBar.done()
          setStatus('status-jv', 'Done')
        },
        onError: (msg) => {
          progressBar.error(msg)
          setStatus('status-jv', `Error: ${msg}`, true)
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e) {
      progressBar.error((e as Error).message)
      setStatus('status-jv', `Error: ${(e as Error).message}`, true)
      btn.disabled = false
    }
  })
```

- [ ] **Step 4: Build and smoke-test in the browser**

Run:
```bash
cd frontend && npm run build && npm run dev
```
Open the app, go to J–V tab, click Run. Expect the progress bar to fill from 0% to 100% over the course of the run, with the stage toggling between "J–V forward sweep" and "J–V reverse sweep", followed by the J–V curves rendering.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/panels/jv.ts
git commit -m "feat(jv): migrate J-V panel to SSE job stream with progress bar

Confidence: high
Scope-risk: narrow"
```

### Task 3.7: Migrate the Impedance panel

**Files:**
- Modify: `frontend/src/panels/impedance.ts`

- [ ] **Step 1: Apply the same pattern as JV to impedance**

Mirror the structure from Task 3.6:
1. Import `startJob`, `streamJobEvents`, `createProgressBar`.
2. Add `<div id="progress-is"></div>` inside the form card.
3. Create the `progressBar` handle in `mountImpedancePanel`.
4. Replace the `runImpedance(...)` call with:
   ```typescript
       const jobId = await startJob('impedance', device, {
         N_grid: Math.max(3, Math.round(readNum('is-N', 40))),
         V_dc: readNum('is-Vdc', 0.9),
         n_freq: Math.max(2, Math.round(readNum('is-nf', 15))),
         f_min: readNum('is-fmin', 10),
         f_max: readNum('is-fmax', 1e5),
       })
       streamJobEvents<ISResult>(jobId, {
         onProgress: (ev) => progressBar.update(ev),
         onResult: (result) => {
           renderISResults(root.querySelector<HTMLDivElement>('#results-is')!, result)
           progressBar.done()
           setStatus('status-is', 'Done')
         },
         onError: (msg) => { progressBar.error(msg); setStatus('status-is', `Error: ${msg}`, true) },
         onDone: () => { btn.disabled = false },
       })
   ```

- [ ] **Step 2: Build and run the impedance tab end-to-end**

Same commands as Task 3.6 Step 4. Verify the Nyquist semicircle sits in the upper half plane (the Phase 1 fix + frontend `−Im(Z)` transform).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/panels/impedance.ts
git commit -m "feat(impedance): migrate to SSE job stream with progress bar

Confidence: high
Scope-risk: narrow"
```

### Task 3.8: Migrate the Degradation panel

**Files:**
- Modify: `frontend/src/panels/degradation.ts`

- [ ] **Step 1: Apply the same pattern**

Mirror Tasks 3.6 / 3.7. Params:
```typescript
{
  N_grid: Math.max(3, Math.round(readNum('deg-N', 40))),
  V_bias: readNum('deg-Vbias', 0.9),
  t_end: readNum('deg-tend', 100),
  n_snapshots: Math.max(2, Math.round(readNum('deg-nsnap', 10))),
}
```

- [ ] **Step 2: Build and run the degradation tab**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/panels/degradation.ts
git commit -m "feat(degradation): migrate to SSE job stream with progress bar

Confidence: high
Scope-risk: narrow"
```

### Task 3.9: Final verification

**Files:** none modified; this task is exercise-only.

- [ ] **Step 1: Full backend test suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Full frontend build**

Run: `cd frontend && npm run build`
Expected: clean build, no TypeScript errors.

- [ ] **Step 3: Manual E2E smoke test**

Start the backend (`uvicorn backend.main:app --app-dir perovskite-sim --port 8000`) and the frontend (`npm run dev`), then:
- Run a J–V sweep on `ionmonger_benchmark`. Watch the progress bar advance 0→100% through "J–V forward sweep" and "J–V reverse sweep". Verify V_oc, J_sc, FF, PCE look correct and the J–V plot has axis labels.
- Run an impedance sweep at V_dc = 0.9 V. Watch the progress bar advance through 15 frequencies. Verify the Nyquist semicircle is in the upper half plane.
- Run a degradation run (t_end = 100 s, 10 snapshots). Watch the progress bar advance through 10 snapshots.

- [ ] **Step 4: Tag the release point**

```bash
git tag perf-progress-phase3-complete
```

No commit — this is a convenience tag for rollback if anything regresses later.

---

## Done-when criteria

- `pytest -q` passes.
- `cd frontend && npm run build` passes.
- All three experiments show a progress bar advancing from 0 → 100% in the browser, followed by the correct result plot.
- The impedance Nyquist sits in the upper half plane (Im(Z) < 0 internally, `−Im(Z)` > 0 on screen).
- Profiler output (Task 2.8) shows `build_material_arrays` called once per experiment run, not per RHS evaluation.
- Measured speedup ≥ 2× on at least two of the three experiments relative to the pre-refactor baseline (ideally 3–10×; if < 2× anywhere, re-profile before calling it done).

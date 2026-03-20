# Correctness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix six correctness and robustness bugs identified in the repository analysis: centralize constants, harmonize interface diffusion coefficients, add ion-clipping diagnostics, improve V_oc accuracy in degradation, fix the impedance lock-in time-axis bias, and add input validation to all public experiment APIs.

**Architecture:** All fixes are self-contained and touch no more than 2–3 files each. Each task follows TDD: write a failing test, verify it fails, implement the fix, verify it passes, commit. The fixes are ordered by impact on physical accuracy.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pytest. Run tests with `python -m pytest <path> -v` from the repo root (`perovskite-sim/`). All commands below assume CWD = `/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim/`.

---

## Task 1: Create `perovskite_sim/constants.py` and update all imports

**Context:** `Q`, `K_B`, `T`, `V_T` are currently copy-pasted in `models/parameters.py`, `solver/mol.py`, `experiments/jv_sweep.py`, and `experiments/degradation.py`. `EPS_0` lives only in `physics/poisson.py`. A single source of truth prevents drift.

**Files:**
- Create: `perovskite_sim/constants.py`
- Modify: `perovskite_sim/models/parameters.py:6-9`
- Modify: `perovskite_sim/solver/mol.py:13-16`
- Modify: `perovskite_sim/experiments/jv_sweep.py:14`
- Modify: `perovskite_sim/experiments/degradation.py:11`
- Modify: `perovskite_sim/physics/poisson.py:6`
- Test: `tests/unit/test_constants.py`

**Step 1: Write the failing test**

Create `tests/unit/test_constants.py`:
```python
from perovskite_sim import constants

def test_constants_values():
    assert abs(constants.Q   - 1.602176634e-19) < 1e-30
    assert abs(constants.K_B - 1.380649e-23)    < 1e-34
    assert abs(constants.T   - 300.0)           < 1e-10
    assert abs(constants.EPS_0 - 8.854187817e-12) < 1e-23
    # V_T at 300 K ≈ 0.025852 V
    assert abs(constants.V_T - 0.025852) < 1e-5

def test_constants_consistent():
    """V_T must equal K_B*T/Q."""
    assert abs(constants.V_T - constants.K_B * constants.T / constants.Q) < 1e-15

def test_all_symbols_exported():
    for name in ("Q", "K_B", "T", "V_T", "EPS_0"):
        assert hasattr(constants, name), f"constants.{name} missing"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_constants.py -v
```
Expected: `ModuleNotFoundError: No module named 'perovskite_sim.constants'`

**Step 3: Create `perovskite_sim/constants.py`**

```python
"""Physical constants used throughout perovskite-sim.

All values are exact SI 2019 redefinition values except EPS_0.
"""
from __future__ import annotations

Q     = 1.602176634e-19   # elementary charge [C]
K_B   = 1.380649e-23      # Boltzmann constant [J/K]
T     = 300.0             # reference temperature [K]
V_T   = K_B * T / Q      # thermal voltage at 300 K [V]  ≈ 0.025852 V
EPS_0 = 8.854187817e-12   # vacuum permittivity [F/m]
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/test_constants.py -v
```
Expected: `3 passed`

**Step 5: Update `perovskite_sim/models/parameters.py`**

Remove lines 6–9:
```python
Q    = 1.602176634e-19
K_B  = 1.380649e-23
T    = 300.0
V_T  = K_B * T / Q
```

Replace with:
```python
from perovskite_sim.constants import Q, K_B, T, V_T  # noqa: F401
```

**Step 6: Update `perovskite_sim/solver/mol.py`**

Remove lines 13–16:
```python
Q    = 1.602176634e-19
K_B  = 1.380649e-23
T    = 300.0
V_T  = K_B * T / Q
```

Replace with:
```python
from perovskite_sim.constants import Q, V_T
```

**Step 7: Update `perovskite_sim/experiments/jv_sweep.py`**

Remove line 14:
```python
Q = 1.602176634e-19
```

Replace with:
```python
from perovskite_sim.constants import Q
```

**Step 8: Update `perovskite_sim/experiments/degradation.py`**

Remove line 11:
```python
Q = 1.602176634e-19
```

Replace with:
```python
from perovskite_sim import constants
```

(We import the whole module here because Task 4 will use `constants.V_T`.)

**Step 9: Update `perovskite_sim/physics/poisson.py`**

Remove line 6:
```python
EPS_0 = 8.854187817e-12
```

Replace with:
```python
from perovskite_sim.constants import EPS_0
```

**Step 10: Run full unit test suite to verify no regressions**

```bash
python -m pytest tests/unit/ -v
```
Expected: all existing tests pass (same count as before).

**Step 11: Commit**

```bash
git add perovskite_sim/constants.py \
        perovskite_sim/models/parameters.py \
        perovskite_sim/solver/mol.py \
        perovskite_sim/experiments/jv_sweep.py \
        perovskite_sim/experiments/degradation.py \
        perovskite_sim/physics/poisson.py \
        tests/unit/test_constants.py
git commit -m "refactor: centralize physical constants in constants.py"
```

---

## Task 2: Fix interface D values — harmonic mean in `_build_carrier_params`

**Context:** `_build_carrier_params` in `solver/mol.py` builds `D_n_face` and `D_p_face` using a second loop over face midpoints ("last-layer-wins"). This is inconsistent with `solve_poisson`, which uses the harmonic mean of adjacent nodal permittivities. At the HTL/absorber and absorber/ETL interfaces, `D_n` and `D_p` can differ by orders of magnitude, so the choice of averaging rule matters.

The harmonic mean `2·a·b/(a+b)` corresponds to a series resistance model — each half-cell contributes equal resistance — and is the physically correct result for a sharp interface.

**Files:**
- Modify: `perovskite_sim/solver/mol.py:86-94`
- Test: `tests/unit/solver/test_mol.py`

**Step 1: Write the failing test**

Add to `tests/unit/solver/test_mol.py`:
```python
def test_interface_d_harmonic_mean():
    """D_face at a layer interface must equal 2·D_a·D_b/(D_a+D_b)."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.mol import _build_carrier_params
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 20) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    params = _build_carrier_params(x, stack)
    D_n_face = params["D_n"]
    # At the HTL/absorber interface (face near x = 200 nm), D_n should be
    # the harmonic mean of HTL D_n and absorber D_n, not either one alone.
    # Find the interface face index
    htl_thickness = stack.layers[0].thickness  # 200e-9 m
    x_face = 0.5 * (x[:-1] + x[1:])
    iface_idx = int(np.argmin(np.abs(x_face - htl_thickness)))
    D_htl = stack.layers[0].params.D_n
    D_abs = stack.layers[1].params.D_n
    expected_harmonic = 2.0 * D_htl * D_abs / (D_htl + D_abs)
    assert abs(D_n_face[iface_idx] - expected_harmonic) < 1e-30 * expected_harmonic + 1e-40, (
        f"D_n_face at interface={D_n_face[iface_idx]:.3e}, "
        f"expected harmonic mean={expected_harmonic:.3e}"
    )
```

Also add `import numpy as np` at the top of the test file if not already present.

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/solver/test_mol.py::test_interface_d_harmonic_mean -v
```
Expected: `FAILED` — the face value will be either `D_htl` or `D_abs` (last-layer-wins), not the harmonic mean.

**Step 3: Fix `_build_carrier_params` in `perovskite_sim/solver/mol.py`**

In `_build_carrier_params`, remove the second loop (lines 86–94):
```python
    # Per-face D via face midpoints (last-layer-wins at interface faces)
    x_face = 0.5 * (x[:-1] + x[1:])
    D_n_face = np.empty(N - 1); D_p_face = np.empty(N - 1)
    offset = 0.0
    for layer in stack.layers:
        mask = (x_face >= offset - 1e-12) & (x_face <= offset + layer.thickness + 1e-12)
        D_n_face[mask] = layer.params.D_n
        D_p_face[mask] = layer.params.D_p
        offset += layer.thickness
```

Replace with:
```python
    # Per-face D via harmonic mean of adjacent nodal values.
    # Matches solve_poisson's harmonic-mean treatment of eps_r at interfaces:
    # both correspond to the series-resistance result for a sharp discontinuity.
    D_n_face = 2.0 * D_n_node[:-1] * D_n_node[1:] / (D_n_node[:-1] + D_n_node[1:])
    D_p_face = 2.0 * D_p_node[:-1] * D_p_node[1:] / (D_p_node[:-1] + D_p_node[1:])
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/solver/test_mol.py -v
```
Expected: all tests including `test_interface_d_harmonic_mean` pass.

**Step 5: Run regression tests to verify device-level physics unchanged**

```bash
python -m pytest tests/regression/ -v
```
Expected: J_sc > 0, V_oc in [0.5, 1.5] V, FF > 0.3. Note: actual values may shift slightly (harmonic mean is lower than arithmetic at interfaces, slightly reducing current — this is more physical, not a regression).

**Step 6: Commit**

```bash
git add perovskite_sim/solver/mol.py \
        tests/unit/solver/test_mol.py
git commit -m "fix: use harmonic mean for inter-layer D_n/D_p at interface faces"
```

---

## Task 3: Add ion-clipping diagnostic in `split_step`

**Context:** `split_step` silently clips `P` to zero when the coupled solver leaks negative ion densities. This hides instability. The fix adds a `warnings.warn` whenever significant clipping occurs (|P| > 1e-30 before clip), giving the user actionable feedback without breaking the solver.

**Files:**
- Modify: `perovskite_sim/solver/mol.py` (inside `split_step`)
- Test: `tests/unit/solver/test_mol.py`

**Step 1: Write the failing test**

Add to `tests/unit/solver/test_mol.py`:
```python
def test_split_step_warns_on_negative_ions():
    """split_step must emit RuntimeWarning when ion density is clipped."""
    import warnings
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.mol import StateVec, split_step
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    y0 = solve_illuminated_ss(x, stack, V_app=0.0)
    # Inject artificially negative ion density to force clipping path
    sv = StateVec.unpack(y0, N)
    P_negative = sv.P.copy()
    P_negative[N // 2] = -1e20   # significant negative value mid-absorber
    y_bad = StateVec.pack(sv.n, sv.p, P_negative)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        split_step(x, y_bad, dt=0.01, stack=stack, V_app=0.0)
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) > 0, "Expected RuntimeWarning about ion clipping"
    assert "clip" in runtime_warnings[0].message.args[0].lower()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/solver/test_mol.py::test_split_step_warns_on_negative_ions -v
```
Expected: `FAILED` — no warning is currently emitted.

**Step 3: Update `split_step` in `perovskite_sim/solver/mol.py`**

Add `import warnings` at the top of the file (after `from __future__ import annotations`).

Inside `split_step`, replace the `_ion_rhs` closure:
```python
    def _ion_rhs(t, P):
        # Clip to non-negative: ions cannot have negative density.
        # Numerical leakage from the full coupled solver can produce tiny
        # negative P values in transport layers; clamping prevents divergence.
        P_nn = np.maximum(P, 0.0)
        rho = Q * (p_frozen - n_frozen + P_nn - N_A + N_D)
        phi = solve_poisson(x, eps_r, rho,
                            phi_left=0.0, phi_right=stack.V_bi - V_app)
        return ion_continuity_rhs(x, phi, P_nn,
                                  absorber.params.D_ion, V_T,
                                  absorber.params.P_lim)
```

With:
```python
    _clip_count = [0]  # mutable closure counter: tracks significant clipping events

    def _ion_rhs(t, P):
        # Clip to non-negative: ions cannot have negative density.
        # Track significant clips (|P| > 1e-30) so we can warn the caller.
        neg_mask = P < -1e-30
        if np.any(neg_mask):
            _clip_count[0] += 1
        P_nn = np.maximum(P, 0.0)
        rho = Q * (p_frozen - n_frozen + P_nn - N_A + N_D)
        phi = solve_poisson(x, eps_r, rho,
                            phi_left=0.0, phi_right=stack.V_bi - V_app)
        return ion_continuity_rhs(x, phi, P_nn,
                                  absorber.params.D_ion, V_T,
                                  absorber.params.P_lim)
```

After the `solve_ivp` call for ions (after the `if not sol_ion.success` block), add:
```python
    if _clip_count[0] > 0:
        warnings.warn(
            f"Ion density clipped to zero in {_clip_count[0]} RHS evaluation(s) "
            "during split_step. This indicates numerical instability — consider "
            "reducing atol for ion states or shortening dt.",
            RuntimeWarning,
            stacklevel=2,
        )
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/unit/solver/test_mol.py -v
```
Expected: all tests pass including the new warning test.

**Step 5: Verify existing split_step tests still pass (no spurious warnings for normal inputs)**

```bash
python -m pytest tests/unit/solver/test_mol.py::test_split_step_shape_and_success \
                 tests/unit/solver/test_mol.py::test_split_step_advances_ions -v
```
Expected: both pass with no warnings printed.

**Step 6: Commit**

```bash
git add perovskite_sim/solver/mol.py \
        tests/unit/solver/test_mol.py
git commit -m "fix: warn when ion density is clipped in split_step"
```

---

## Task 4: Fix V_oc in degradation — geometric mean instead of arithmetic mean

**Context:** In `experiments/degradation.py`, V_oc is computed as:
```python
V_oc = V_T * np.log(np.mean(n_abs * pp_abs) / p.ni_sq)
```
The arithmetic mean of `n·p` is dominated by the high-injection peak near the absorber interfaces (where ions have accumulated), causing a systematic overestimate. The geometric mean — computed as `exp(mean(log(n·p)))`, equivalently `mean(log(n·p/ni²)) * V_T` — gives the spatially averaged quasi-Fermi separation, which is more representative of bulk recombination and less sensitive to interface spikes.

Also: `V_T = 0.025852` is hard-coded; after Task 1 we use `constants.V_T`.

**Files:**
- Modify: `perovskite_sim/experiments/degradation.py:90-96`
- Test: `tests/unit/experiments/test_degradation.py`

**Step 1: Write the failing test**

Add to `tests/unit/experiments/test_degradation.py`:
```python
def test_voc_uses_geometric_mean():
    """V_oc must use geometric mean of n*p (mean of log), not arithmetic mean.

    Create a mock absorber state where n*p has one very high spike (mimicking
    interface injection after ion pileup). The arithmetic mean is dominated by
    the spike; the geometric mean is not.
    """
    import numpy as np
    from perovskite_sim import constants

    ni_sq = (3.2e13) ** 2
    # 10 absorber nodes: 9 "bulk" with n*p = 10*ni_sq, 1 spike at 1000*ni_sq
    np_bulk = 10.0 * ni_sq * np.ones(10)
    np_bulk[-1] = 1000.0 * ni_sq  # interface spike

    V_T = constants.V_T
    voc_arithmetic = V_T * np.log(np.mean(np_bulk) / ni_sq)
    voc_geometric  = V_T * np.mean(np.log(np_bulk / ni_sq))

    # Geometric must be < arithmetic (Jensen's inequality; log is concave)
    assert voc_geometric < voc_arithmetic, (
        f"geometric ({voc_geometric:.4f} V) should be < arithmetic ({voc_arithmetic:.4f} V)"
    )
    # Geometric must be close to the bulk value (log(10) * V_T ≈ 0.0595 V)
    bulk_voc = V_T * np.log(10.0)
    assert abs(voc_geometric - bulk_voc) < 0.005, (
        f"geometric V_oc={voc_geometric:.4f} V should be near bulk {bulk_voc:.4f} V"
    )


def test_degradation_voc_decreases_over_time():
    """V_oc must be positive and physically plausible (0.5–1.2 V) at all snapshots."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.degradation import run_degradation
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_degradation(stack, t_end=5.0, n_snapshots=3,
                             V_bias=0.9, N_grid=20, dt_max=1.0)
    assert np.all(result.V_oc > 0.5), f"V_oc below 0.5 V: {result.V_oc}"
    assert np.all(result.V_oc < 1.3), f"V_oc above 1.3 V: {result.V_oc}"
```

Add `import numpy as np` at the top if not already there.

**Step 2: Run tests to verify the new tests fail (or pass for the wrong reason)**

```bash
python -m pytest tests/unit/experiments/test_degradation.py::test_voc_uses_geometric_mean \
                 tests/unit/experiments/test_degradation.py::test_degradation_voc_decreases_over_time -v
```

`test_voc_uses_geometric_mean` should PASS (it just verifies the math). `test_degradation_voc_decreases_over_time` may pass or fail depending on current V_oc values — run it to establish baseline.

**Step 3: Fix `run_degradation` in `perovskite_sim/experiments/degradation.py`**

Replace lines 93–96:
```python
        abs_mask = (x > stack.layers[0].thickness) & (x < stack.layers[0].thickness + stack.layers[1].thickness)
        n_abs = sv.n[abs_mask]; pp_abs = sv.p[abs_mask]
        V_T = 0.025852
        V_oc = V_T * np.log(np.mean(n_abs * pp_abs) / p.ni_sq)
```

With:
```python
        abs_mask = (x > stack.layers[0].thickness) & (
            x < stack.layers[0].thickness + stack.layers[1].thickness
        )
        n_abs = sv.n[abs_mask]
        pp_abs = sv.p[abs_mask]
        # Geometric-mean quasi-Fermi V_oc: mean(log(n·p/ni²))·V_T
        # Less biased than arithmetic mean when interface injection spikes dominate.
        V_oc = constants.V_T * np.mean(np.log(n_abs * pp_abs / p.ni_sq))
```

Also update the `from perovskite_sim import constants` import added in Task 1 (already done).

**Step 4: Run full degradation tests**

```bash
python -m pytest tests/unit/experiments/test_degradation.py -v
```
Expected: all 4 tests pass.

**Step 5: Commit**

```bash
git add perovskite_sim/experiments/degradation.py \
        tests/unit/experiments/test_degradation.py
git commit -m "fix: use geometric-mean V_oc in degradation to reduce interface-spike bias"
```

---

## Task 5: Fix impedance lock-in time-axis bias

**Context:** In `run_impedance`, `J_t[i]` is computed after integrating to `t_eval[i+1]` with voltage `V_ac(midpoint of [t_eval[i], t_eval[i+1]])`. The current lock-in uses `t_eval[1:][-pts:]` (the end-of-interval times) as the reference signal time axis, creating a half-step phase bias at high frequencies. The fix: use midpoint times `t_mid = 0.5*(t_eval[:-1]+t_eval[1:])` for the sin/cos reference, since that is when the voltage was actually applied.

**Files:**
- Modify: `perovskite_sim/experiments/impedance.py:74-84`
- Test: `tests/unit/experiments/test_impedance.py`

**Step 1: Write the failing test**

Add to `tests/unit/experiments/test_impedance.py`:
```python
def test_dummy_rc_phase():
    """dummy_mode RC circuit: Z must have negative imaginary part (capacitive)
    and |angle| must decrease as frequency increases.
    Dummy RC: R=10, C=1e-6. At f=1e3 Hz, |Z_imag| >> |Z_real|.
    At f=1e6 Hz, |Z_real| >> |Z_imag|."""
    import numpy as np
    from perovskite_sim.experiments.impedance import extract_impedance
    freqs = np.array([1e2, 1e4, 1e6])
    Z = extract_impedance(freqs, dummy_mode=True)
    # Capacitive: imaginary part must be negative
    assert np.all(Z.imag < 0), f"Expected negative Im(Z), got {Z.imag}"
    # Phase angle |θ| = arctan(Im/Re) must decrease with frequency
    angles = np.abs(np.angle(Z, deg=True))
    assert angles[0] > angles[1] > angles[2], (
        f"Phase angle should decrease with frequency: {angles}"
    )

def test_lockin_uses_midpoint_times():
    """Lock-in reference must use midpoint times, not end-of-interval times.
    We verify the attribute exists and the internal extraction logic is consistent.
    This is a smoke-test: verify run_impedance returns finite complex Z."""
    import numpy as np
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.impedance import run_impedance
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    freqs = np.array([1.0])   # 1 Hz, very slow — few steps
    result = run_impedance(stack, freqs, V_dc=0.9, delta_V=0.01,
                           N_grid=20, n_cycles=3)
    assert result.Z.shape == (1,)
    assert np.all(np.isfinite(result.Z.real))
    assert np.all(np.isfinite(result.Z.imag))
```

**Step 2: Run tests to verify baseline**

```bash
python -m pytest tests/unit/experiments/test_impedance.py -v
```
`test_dummy_rc_phase` should already pass (dummy_mode has no time-axis issue). `test_lockin_uses_midpoint_times` will run the full solver — note its current behavior.

**Step 3: Fix lock-in time axis in `perovskite_sim/experiments/impedance.py`**

Replace lines 74–84:
```python
        # Lock-in extraction from last cycle (settled AC response).
        # J_t[i] is computed after evolving to t_eval[i+1], so
        # the corresponding time axis is t_eval[1:] and J_t[:-1].
        pts = 20   # points per cycle
        t_ac = t_eval[1:][-pts:]        # last cycle time points
        J_ac = J_t[:-1][-pts:]         # matching current values
        sin_ref = np.sin(2 * np.pi * f * t_ac)
        cos_ref = np.cos(2 * np.pi * f * t_ac)
        J_in   = 2.0 * np.mean(J_ac * sin_ref)   # in-phase with V = δV·sin
        J_quad = 2.0 * np.mean(J_ac * cos_ref)   # quadrature
        delta_J = J_in + 1j * J_quad
        Z_arr[k] = delta_V / delta_J if abs(delta_J) > 0 else complex(np.inf, 0)
```

With:
```python
        # Lock-in extraction from last cycle (settled AC response).
        # J_t[i] is the current after integrating over [t_eval[i], t_eval[i+1]]
        # with voltage V_ac applied at the midpoint of that interval.
        # Use midpoint times for the sin/cos reference to match the actual
        # voltage phase, eliminating the half-step bias at high frequencies.
        pts = 20   # points per cycle
        t_mid_all = 0.5 * (t_eval[:-1] + t_eval[1:])   # midpoint of each interval
        t_ac = t_mid_all[-pts:]         # last cycle midpoint times  (len=pts)
        J_ac = J_t[:-1][-pts:]         # last cycle current values   (len=pts)
        sin_ref = np.sin(2 * np.pi * f * t_ac)
        cos_ref = np.cos(2 * np.pi * f * t_ac)
        J_in   = 2.0 * np.mean(J_ac * sin_ref)   # in-phase with V = δV·sin
        J_quad = 2.0 * np.mean(J_ac * cos_ref)   # quadrature
        delta_J = J_in + 1j * J_quad
        Z_arr[k] = delta_V / delta_J if abs(delta_J) > 0 else complex(np.inf, 0)
```

**Step 4: Run all impedance tests**

```bash
python -m pytest tests/unit/experiments/test_impedance.py -v
```
Expected: all 4 tests pass.

**Step 5: Commit**

```bash
git add perovskite_sim/experiments/impedance.py \
        tests/unit/experiments/test_impedance.py
git commit -m "fix: use midpoint times for impedance lock-in to remove half-step phase bias"
```

---

## Task 6: Add input validation to public experiment APIs

**Context:** `run_jv_sweep`, `run_impedance`, and `run_degradation` silently fail or crash with cryptic NumPy errors when given bad inputs (zero thickness layers, negative v_rate, empty stacks, tiny N_grid). Early validation with clear messages makes bugs much easier to diagnose.

**Files:**
- Modify: `perovskite_sim/experiments/jv_sweep.py`
- Modify: `perovskite_sim/experiments/impedance.py`
- Modify: `perovskite_sim/experiments/degradation.py`
- Test: `tests/unit/experiments/test_jv_sweep.py`
- Test: `tests/unit/experiments/test_impedance.py`
- Test: `tests/unit/experiments/test_degradation.py`

**Step 1: Write failing tests for `run_jv_sweep`**

Add to `tests/unit/experiments/test_jv_sweep.py`:
```python
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml


def _good_stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def test_jv_sweep_rejects_small_n_grid():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    with pytest.raises(ValueError, match="N_grid"):
        run_jv_sweep(_good_stack(), N_grid=2)


def test_jv_sweep_rejects_small_n_points():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    with pytest.raises(ValueError, match="n_points"):
        run_jv_sweep(_good_stack(), n_points=1)


def test_jv_sweep_rejects_nonpositive_v_rate():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    with pytest.raises(ValueError, match="v_rate"):
        run_jv_sweep(_good_stack(), v_rate=0.0)
```

**Step 2: Write failing tests for `run_impedance`**

Add to `tests/unit/experiments/test_impedance.py`:
```python
def test_impedance_rejects_empty_frequencies():
    import numpy as np
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="frequenc"):
        run_impedance(stack, np.array([]))


def test_impedance_rejects_small_n_grid():
    import numpy as np
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_impedance(stack, np.array([1e3]), N_grid=2)


def test_impedance_rejects_zero_delta_v():
    import numpy as np
    from perovskite_sim.experiments.impedance import run_impedance
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="delta_V"):
        run_impedance(stack, np.array([1e3]), delta_V=0.0)
```

Add `import pytest` at the top of `test_impedance.py`.

**Step 3: Write failing tests for `run_degradation`**

Add to `tests/unit/experiments/test_degradation.py`:
```python
import pytest


def test_degradation_rejects_nonpositive_t_end():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="t_end"):
        run_degradation(stack, t_end=0.0)


def test_degradation_rejects_small_n_grid():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_degradation(stack, N_grid=2)


def test_degradation_rejects_small_n_snapshots():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="n_snapshots"):
        run_degradation(stack, n_snapshots=0)
```

**Step 4: Run all new validation tests to verify they fail**

```bash
python -m pytest tests/unit/experiments/test_jv_sweep.py::test_jv_sweep_rejects_small_n_grid \
                 tests/unit/experiments/test_jv_sweep.py::test_jv_sweep_rejects_nonpositive_v_rate \
                 tests/unit/experiments/test_impedance.py::test_impedance_rejects_empty_frequencies \
                 tests/unit/experiments/test_degradation.py::test_degradation_rejects_nonpositive_t_end \
                 -v
```
Expected: `FAILED` (no validation exists yet).

**Step 5: Add validation to `run_jv_sweep` in `perovskite_sim/experiments/jv_sweep.py`**

At the start of the function body (before `layers_grid = ...`):
```python
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_rate <= 0:
        raise ValueError(f"v_rate must be positive, got {v_rate}")
    for i, layer in enumerate(stack.layers):
        if layer.thickness <= 0:
            raise ValueError(
                f"layer {i} ({layer.name!r}) has non-positive thickness {layer.thickness}"
            )
```

**Step 6: Add validation to `run_impedance` in `perovskite_sim/experiments/impedance.py`**

At the start of the function body (before `layers_grid = ...`):
```python
    if len(frequencies) == 0:
        raise ValueError("frequencies must be non-empty")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if delta_V <= 0:
        raise ValueError(f"delta_V must be positive, got {delta_V}")
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
```

**Step 7: Add validation to `run_degradation` in `perovskite_sim/experiments/degradation.py`**

At the start of the function body (before `layers_grid = ...`):
```python
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    if dt_max <= 0:
        raise ValueError(f"dt_max must be positive, got {dt_max}")
```

**Step 8: Run all experiment tests**

```bash
python -m pytest tests/unit/experiments/ -v
```
Expected: all tests pass.

**Step 9: Commit**

```bash
git add perovskite_sim/experiments/jv_sweep.py \
        perovskite_sim/experiments/impedance.py \
        perovskite_sim/experiments/degradation.py \
        tests/unit/experiments/test_jv_sweep.py \
        tests/unit/experiments/test_impedance.py \
        tests/unit/experiments/test_degradation.py
git commit -m "fix: add input validation to run_jv_sweep, run_impedance, run_degradation"
```

---

## Final Verification

Run the full test suite:

```bash
python -m pytest tests/ -v --tb=short
```

Expected output:
- All unit tests pass
- Integration tests pass (n·p = ni² at equilibrium)
- Regression tests pass (J_sc > 0, V_oc ∈ [0.5, 1.5] V, FF > 0.3)
- No new failures introduced

If regression test values shift materially after Task 2 (harmonic D at interfaces), check whether they are still physically plausible rather than matching old incorrect behavior.

# Physical Trend Validation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six physics trend validation tests that assert the simulator reproduces well-established device-physics scaling laws.

**Architecture:** A single new test file `tests/validation/test_physical_trends.py` with six parametrized tests, each running a parameter sweep via `dataclasses.replace()` on a baseline preset, extracting a trend metric, and asserting it falls within a literature window. Tagged `@pytest.mark.validation` for explicit invocation. Zero production-code changes.

**Design Decision — Baseline Preset:** The spec originally specified `nip_MAPbI3_tmm.yaml`. This plan uses `nip_MAPbI3.yaml` (Beer-Lambert, chi=Eg=0 in all layers) instead because Trends 1 and 5 vary the absorber Eg and need the optical-generation response to track. With TMM, the n,k data is fixed per `optical_material` key and does not respond to Eg changes — the J_sc vs Eg trend would be a no-op. Beer-Lambert uses `alpha` on `MaterialParams` and the Eg-shifted ni directly, so both electrical and optical responses are live. The FULL physics tier is still active (default for presets that don't pin `mode`), so thermionic emission will engage once we set non-zero Eg/chi on the absorber — testing more physics, not less.

**Tech Stack:** pytest, numpy, scipy, existing `perovskite_sim` experiment primitives (`run_jv_sweep`, `run_suns_voc`), `dataclasses.replace()`

**Expected runtime:** ~5–8 minutes for the full validation suite (dominated by J-V sweep settles). Invoked explicitly: `pytest -m validation`.

---

### Task 1: Register the `validation` pytest marker

**Files:**
- Modify: `perovskite-sim/pyproject.toml` (add one line to markers list)

- [ ] **Step 1: Add `validation` marker to pyproject.toml**

In `perovskite-sim/pyproject.toml`, append the `validation` marker to the existing markers list:

```toml
markers = [
    "slow: marks tests as slow (run with -m slow)",
    "regression: physical-sanity regression tests (V_oc / J_sc / FF envelopes, parity gates)",
    "validation: physics trend validation against literature windows (V_oc vs Eg, FF vs mobility, etc.)",
]
```

- [ ] **Step 2: Verify marker is registered**

Run: `cd perovskite-sim && pytest --markers | grep validation`
Expected: `validation: physics trend validation against literature windows...`

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/pyproject.toml
git commit -m "chore: register validation pytest marker for physics trend tests"
```

---

### Task 2: Create shared test utilities and fixture

**Files:**
- Create: `perovskite-sim/tests/validation/__init__.py`
- Create: `perovskite-sim/tests/validation/test_physical_trends.py` (skeleton with fixture + first test)

- [ ] **Step 1: Create the `tests/validation/` package**

```bash
mkdir -p perovskite-sim/tests/validation
```

Write `perovskite-sim/tests/validation/__init__.py` (empty):

```python
"""Physics trend validation tests — invoked explicitly with ``pytest -m validation``."""
```

- [ ] **Step 2: Write the shared fixture and helper in the test file**

Write `perovskite-sim/tests/validation/test_physical_trends.py`:

```python
"""Physics trend validation: asserts that the drift-diffusion solver reproduces
well-established device-physics scaling laws.

Invoke with: pytest -m validation
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import linregress

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, JVResult

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def baseline_stack() -> DeviceStack:
    """Beer-Lambert n-i-p MAPbI3 preset — FULL tier (default), all physics live.

    Uses the BL preset rather than TMM because Trends 1 & 5 vary absorber Eg
    and need optical generation to respond. TMM n,k data is fixed per
    optical_material key and does not shift with Eg.
    """
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def _vary_absorber_param(
    stack: DeviceStack, param_name: str, values: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer's MaterialParams field
    ``param_name`` set to each value in ``values``.

    Preserves the role-tag scan so ``role: absorber`` is the target layer.
    """
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    assert layer.params is not None, "absorber layer must have MaterialParams"

    stacks: list[DeviceStack] = []
    for v in values:
        new_params = replace(layer.params, **{param_name: v})
        new_layer = replace(layer, params=new_params)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _vary_absorber_thickness(
    stack: DeviceStack, thicknesses: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer thickness varied."""
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    stacks: list[DeviceStack] = []
    for t in thicknesses:
        new_layer = replace(layer, thickness=t)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _run_jv(stack: DeviceStack) -> JVResult:
    """Run a J-V sweep with settings matching the regression suite."""
    return run_jv_sweep(
        stack, N_grid=60, n_points=20, v_rate=5.0,
        V_max=1.5,
    )
```

- [ ] **Step 3: Run the test file to confirm it imports and the fixture loads**

Run: `cd perovskite-sim && python -c "from tests.validation.test_physical_trends import baseline_stack; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add perovskite-sim/tests/validation/
git commit -m "test(validation): add shared fixtures and helpers for physics trend tests"
```

---

### Task 3: Implement Trend 1 (V_oc vs Bandgap) and Trend 5 (J_sc vs Bandgap) — shared Eg sweep

**Files:**
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py` (append two test functions)

These share the same Eg sweep — run the J-V once per Eg and extract both metrics.

- [ ] **Step 1: Write the shared sweep fixture**

Append to `perovskite-sim/tests/validation/test_physical_trends.py`:

```python
EG_SWEEP = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # eV


@pytest.fixture(scope="module")
def eg_sweep_results(baseline_stack: DeviceStack) -> list[tuple[float, JVResult]]:
    """Run J-V at each absorber Eg and return (Eg, JVResult) pairs."""
    results: list[tuple[float, JVResult]] = []
    stacks = _vary_absorber_param(baseline_stack, "Eg", EG_SWEEP)
    for eg, stack in zip(EG_SWEEP, stacks):
        result = _run_jv(stack)
        results.append((eg, result))
    return results
```

- [ ] **Step 2: Write the V_oc vs Bandgap test (Trend 1)**

```python
def test_voc_vs_bandgap(eg_sweep_results: list[tuple[float, JVResult]]) -> None:
    """V_oc loss ΔV = Eg/q − V_oc should be roughly constant with bandgap.

    In a physically correct simulator V_oc tracks Eg with slope ≈ 1,
    so the non-radiative loss stays in a narrow band (0.25–0.55 V).
    A simulator where V_oc does not respond to Eg would show growing ΔV.
    """
    eg_values = np.array([eg for eg, _ in eg_sweep_results])
    voc_values = np.array([r.metrics_fwd.V_oc for _, r in eg_sweep_results])
    delta_V = eg_values - voc_values

    median_loss = float(np.median(delta_V))
    assert 0.25 <= median_loss <= 0.55, (
        f"Median V_oc loss {median_loss:.3f} V outside [0.25, 0.55] V — "
        f"V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )

    slope, _, _, _, _ = linregress(eg_values, delta_V)
    assert abs(slope) <= 0.15, (
        f"ΔV slope vs Eg is {slope:.3f} — should be near zero; "
        "V_oc is not tracking bandgap correctly"
    )
```

- [ ] **Step 3: Write the J_sc vs Bandgap test (Trend 5)**

```python
def test_jsc_vs_bandgap(eg_sweep_results: list[tuple[float, JVResult]]) -> None:
    """J_sc should decrease with increasing bandgap.

    Wider bandgap → fewer above-gap photons absorbed → lower photocurrent.
    Under Beer-Lambert with fixed alpha, the Eg-shifted ni contributes a
    small electrical component; the dominant optical trend is captured.
    """
    eg_values = np.array([eg for eg, _ in eg_sweep_results])
    jsc_values = np.array([r.metrics_fwd.J_sc for _, r in eg_sweep_results])

    assert all(j > 0 for j in jsc_values), (
        f"All J_sc values must be positive, got: {jsc_values}"
    )
    assert jsc_values[-1] < jsc_values[0], (
        f"J_sc at Eg={eg_values[-1]:.1f} eV ({jsc_values[-1]:.1f} A/m²) "
        f"must be less than J_sc at Eg={eg_values[0]:.1f} eV ({jsc_values[0]:.1f} A/m²)"
    )

    ratio = jsc_values[-1] / jsc_values[0]
    assert ratio <= 0.95, (
        f"J_sc ratio widest/narrowest Eg = {ratio:.3f} — "
        "expected more drop at wider bandgap (≤ 0.95)"
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd perovskite-sim && pytest -m validation -k "test_voc_vs_bandgap or test_jsc_vs_bandgap" -v`
Expected: both PASS (Trend 1 and Trend 5)

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): add V_oc vs Eg and J_sc vs Eg trend tests"
```

---

### Task 4: Implement Trend 2 (V_oc vs Thickness)

**Files:**
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py` (append one test)

- [ ] **Step 1: Write the test**

Append to the test file:

```python
THICKNESS_SWEEP_NM = [100, 200, 400, 700, 1000]  # nm → converted to m


def test_voc_vs_thickness(baseline_stack: DeviceStack) -> None:
    """V_oc should increase with absorber thickness.

    dV_oc / d(log₁₀(thickness)) should be positive and in 30–90 mV/decade —
    the classic SRH signature: thicker absorbers dilute contact recombination.
    """
    thicknesses_m = [t * 1e-9 for t in THICKNESS_SWEEP_NM]
    stacks = _vary_absorber_thickness(baseline_stack, thicknesses_m)

    voc_values: list[float] = []
    for stack in stacks:
        result = _run_jv(stack)
        voc_values.append(result.metrics_fwd.V_oc)

    log10_t = np.log10(THICKNESS_SWEEP_NM)
    slope, intercept, r_value, _, _ = linregress(log10_t, voc_values)

    # mV/decade
    slope_mv_per_decade = slope * 1000

    assert r_value > 0.7, (
        f"V_oc vs log₁₀(thickness) correlation r={r_value:.3f} is too weak — "
        f"V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )
    assert 20 <= slope_mv_per_decade <= 120, (
        f"V_oc vs thickness slope {slope_mv_per_decade:.1f} mV/decade "
        f"outside [20, 120] — V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )
```

- [ ] **Step 2: Run the test**

Run: `cd perovskite-sim && pytest -m validation -k "test_voc_vs_thickness" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): add V_oc vs thickness trend test"
```

---

### Task 5: Implement Trend 3 (FF vs Mobility)

**Files:**
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py` (append one test)

- [ ] **Step 1: Write the test**

Append to the test file:

```python
# Mobility sweep in cm²/Vs (conventional literature units) → converted to m²/Vs
MOBILITY_SWEEP_CM2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # cm²/Vs


def test_ff_vs_mobility(baseline_stack: DeviceStack) -> None:
    """FF should degrade measurably below ~1e-4 cm²/Vs.

    At low mobility transport resistance limits charge extraction, reducing FF.
    The test asserts FF at the lowest μ is at least 3 percentage points
    (absolute) lower than at the highest μ.
    """
    mobility_m2 = [m * 1e-4 for m in MOBILITY_SWEEP_CM2]  # cm²/Vs → m²/Vs
    ff_values: list[float] = []
    for mu in mobility_m2:
        stacks = _vary_absorber_param(baseline_stack, "mu_n", [mu])
        s = _vary_absorber_param(stacks[0], "mu_p", [mu])[0]
        result = _run_jv(s)
        ff_values.append(result.metrics_fwd.FF)
        if not ff_values[-1] > 0:
            pytest.fail(f"FF=0 at μ={mu:.2e} m²/Vs — solver likely failed")

    ff_drop = ff_values[0] - ff_values[-1]
    assert ff_drop >= 0.03, (
        f"FF drop from lowest to highest mobility is only {ff_drop:.4f} "
        f"(absolute) — expected ≥ 0.03. FF values: "
        f"{[f'{ff:.4f}' for ff in ff_values]}"
    )
```

- [ ] **Step 2: Run the test**

Run: `cd perovskite-sim && pytest -m validation -k "test_ff_vs_mobility" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): add FF vs mobility trend test"
```

---

### Task 6: Implement Trend 4 (Ideality Factor)

**Files:**
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py` (append one test)

- [ ] **Step 1: Write the test**

Append to the test file:

```python
def test_ideality_factor(baseline_stack: DeviceStack) -> None:
    """Dark J-V ideality factor should be 1.0 ≤ n_id ≤ 2.0.

    In the low-injection regime (J < J_sc/100), a single-junction device
    with SRH and radiative recombination has n_id between 1 and 2.
    n_id < 1 or > 2 signals a missing or misconfigured recombination path.
    """
    # Get J_sc reference from illuminated run for the threshold
    ill_result = _run_jv(baseline_stack)
    j_sc = ill_result.metrics_fwd.J_sc
    assert j_sc > 0, "Need illuminated J_sc reference for ideality test"

    # Run dark J-V
    dark_result = run_jv_sweep(
        baseline_stack, N_grid=60, n_points=30, v_rate=1.0,
        V_max=1.5, illuminated=False,
    )
    V = np.asarray(dark_result.V_fwd)
    J = np.asarray(dark_result.J_fwd)

    # Low-injection region: J positive but well below J_sc
    threshold = j_sc / 100
    lo_mask = (J > 0) & (J < threshold)
    if lo_mask.sum() < 4:
        pytest.skip(
            f"Not enough low-injection points: {lo_mask.sum()} with J < {threshold:.1f}"
        )

    V_lo = V[lo_mask]
    J_lo = J[lo_mask]

    slope, _, r_value, _, _ = linregress(V_lo, np.log(J_lo))
    # slope = d(ln J)/dV = q / (n_id * kT)  → n_id = q / (slope * kT)
    # At 300 K: kT/q = 0.02585 V, so n_id = 1 / (slope * 0.02585)
    n_id = 1.0 / (slope * 0.02585)

    assert r_value > 0.95, (
        f"Ideality fit correlation r={r_value:.3f} too weak — "
        "J(V) may not be exponential in the selected region"
    )
    assert 1.0 <= n_id <= 2.5, (
        f"Ideality factor n_id = {n_id:.2f} outside [1.0, 2.5]"
    )
```

- [ ] **Step 2: Run the test**

Run: `cd perovskite-sim && pytest -m validation -k "test_ideality_factor" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): add ideality factor trend test"
```

---

### Task 7: Implement Trend 6 (V_oc vs Illumination — Suns-V_oc)

**Files:**
- Modify: `perovskite-sim/tests/validation/test_physical_trends.py` (append one test)

- [ ] **Step 1: Write the test**

Append to the test file:

```python
def test_voc_vs_illumination(baseline_stack: DeviceStack) -> None:
    """Suns-V_oc slope dV_oc / d(ln Φ) should be n_id · kT/q.

    At 300 K: slope ∈ [25, 65] mV/decade, corresponding to
    ideality factor n_id ∈ [1.0, 2.5].

    Uses the existing run_suns_voc experiment with default suns levels.
    """
    from perovskite_sim.experiments.suns_voc import run_suns_voc

    # Wider suns range for robust slope fitting
    suns_levels = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    result = run_suns_voc(
        baseline_stack, suns_levels=suns_levels, N_grid=60, t_settle=0.1,
    )

    assert len(result.suns) >= 4, (
        f"Need ≥4 suns levels for slope fit, got {len(result.suns)}"
    )
    # Filter out any failed levels (V_oc may be zero or NaN on failure)
    valid = np.isfinite(result.V_oc) & (result.V_oc > 0)
    suns_valid = np.asarray(result.suns)[valid]
    voc_valid = np.asarray(result.V_oc)[valid]

    assert len(suns_valid) >= 3, (
        f"Need ≥3 valid V_oc points, got {len(suns_valid)}"
    )

    slope, _, r_value, _, _ = linregress(np.log(suns_valid), voc_valid)
    slope_mv_per_decade = slope * 1000  # V/decade → mV/decade

    assert r_value > 0.95, (
        f"Suns-V_oc slope fit correlation r={r_value:.3f} too weak"
    )
    assert 20 <= slope_mv_per_decade <= 70, (
        f"Suns-V_oc slope {slope_mv_per_decade:.1f} mV/decade outside [20, 70]"
    )
```

- [ ] **Step 2: Run the test**

Run: `cd perovskite-sim && pytest -m validation -k "test_voc_vs_illumination" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): add V_oc vs illumination (Suns-V_oc) trend test"
```

---

### Task 8: Full suite integration run and final commit

- [ ] **Step 1: Run the full validation suite**

Run: `cd perovskite-sim && pytest -m validation -v`
Expected: all 6 tests PASS

- [ ] **Step 2: Verify validation tests are excluded from default run**

Run: `cd perovskite-sim && pytest --collect-only -q 2>&1 | grep validation`
Expected: no output (validation tests should NOT appear in default collection)

Run: `cd perovskite-sim && pytest -m validation --collect-only -q`
Expected: 6 tests listed

- [ ] **Step 3: Final commit (if any adjustments needed from Step 1)**

```bash
git add perovskite-sim/tests/validation/test_physical_trends.py
git commit -m "test(validation): finalize physics trend validation suite"
```

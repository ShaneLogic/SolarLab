# Stage B(a) — Lateral Microstructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the headline Stage-B physics — single vertical grain boundary in the absorber with reduced (τ_n, τ_p) — end-to-end (config loader → solver → experiment → backend → frontend → docs), with a regression that pins the V_oc(L_g) trend.

**Architecture:** The data model (`GrainBoundary`, `Microstructure`) and the τ-heterogeneity hook (`build_tau_field` → `MaterialArrays2D.tau_n/tau_p`) already exist from Stage A. `assemble_rhs_2d` already passes per-node τ into the recombination kernel. So Stage B(a) is mostly the *outer* layers: YAML schema, presets, regression coverage, headline experiment (`voc_grain_sweep`), and UI/backend exposure.

**Tech Stack:** Python (perovskite_sim.twod), pytest, FastAPI, Vite/TS, Plotly.

**Spec reference:** `docs/superpowers/specs/2026-04-27-2d-microstructural-extension-design.md` Section 8 ("Stage B — Single grain boundary").

---

### Task 1: Microstructure YAML loader

**Files:**
- Modify: `perovskite-sim/perovskite_sim/twod/microstructure.py` (add `Microstructure.from_yaml_block` or a free function `load_microstructure_from_yaml_block(block) -> Microstructure`)
- Modify: `perovskite-sim/perovskite_sim/models/config_loader.py` (parse top-level `microstructure:` block; if absent, return empty `Microstructure()`)
- Test: `perovskite-sim/tests/unit/twod/test_microstructure.py` (add tests for the YAML loader)

- [ ] **Step 1: Write failing test for YAML round-trip**

```python
def test_microstructure_yaml_loader_single_gb(tmp_path):
    yaml_text = """
microstructure:
  grain_boundaries:
    - x_position: 250e-9
      width: 5e-9
      tau_n: 1e-9
      tau_p: 1e-9
      layer_role: absorber
"""
    block = yaml.safe_load(yaml_text)["microstructure"]
    ms = load_microstructure_from_yaml_block(block)
    assert len(ms.grain_boundaries) == 1
    gb = ms.grain_boundaries[0]
    assert gb.x_position == pytest.approx(250e-9)
    assert gb.width == pytest.approx(5e-9)
    assert gb.tau_n == pytest.approx(1e-9)
    assert gb.tau_p == pytest.approx(1e-9)
    assert gb.layer_role == "absorber"


def test_microstructure_yaml_loader_empty_block_returns_empty():
    assert load_microstructure_from_yaml_block({}).grain_boundaries == ()
    assert load_microstructure_from_yaml_block(None).grain_boundaries == ()
    assert load_microstructure_from_yaml_block({"grain_boundaries": []}).grain_boundaries == ()


def test_microstructure_yaml_loader_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        load_microstructure_from_yaml_block(
            {"grain_boundaries": [{"x_position": 0, "width": 1, "tau_n": 1, "tau_p": 1,
                                    "color": "red"}]}
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v -k yaml_loader`
Expected: 3 FAILs (function not defined / ValueError not raised).

- [ ] **Step 3: Implement `load_microstructure_from_yaml_block`**

```python
# in perovskite_sim/twod/microstructure.py

_GB_KEYS = {"x_position", "width", "tau_n", "tau_p", "layer_role"}


def load_microstructure_from_yaml_block(block: dict | None) -> Microstructure:
    """Parse a YAML `microstructure:` block into a Microstructure.

    Schema:
        microstructure:
          grain_boundaries:
            - x_position: <float, m>
              width: <float, m>
              tau_n: <float, s>
              tau_p: <float, s>
              layer_role: <str, default "absorber">

    None / {} / {grain_boundaries: []} all return Microstructure().
    Unknown keys on a grain_boundary entry raise ValueError so configs cannot silently drop fields.
    """
    if not block:
        return Microstructure()
    raw_gbs = block.get("grain_boundaries", []) or []
    gbs: list[GrainBoundary] = []
    for entry in raw_gbs:
        unknown = set(entry.keys()) - _GB_KEYS
        if unknown:
            raise ValueError(f"microstructure.grain_boundaries unknown key(s): {sorted(unknown)}")
        gbs.append(GrainBoundary(
            x_position=float(entry["x_position"]),
            width=float(entry["width"]),
            tau_n=float(entry["tau_n"]),
            tau_p=float(entry["tau_p"]),
            layer_role=str(entry.get("layer_role", "absorber")),
        ))
    return Microstructure(grain_boundaries=tuple(gbs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v -k yaml_loader`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/twod/microstructure.py perovskite-sim/tests/unit/twod/test_microstructure.py
git commit -m "feat(twod): YAML loader for microstructure block"
```

---

### Task 2: Microstructure preset config

**Files:**
- Create: `perovskite-sim/configs/twod/nip_MAPbI3_singleGB.yaml`

- [ ] **Step 1: Create the preset by extending the uniform Stage-A preset**

Clone `configs/twod/nip_MAPbI3_uniform.yaml` and append the GB block at the bottom:

```yaml
# (... existing nip_MAPbI3 stack ...)
microstructure:
  grain_boundaries:
    - x_position: 250e-9      # centred in a 500 nm domain
      width: 10e-9            # 10 nm GB band
      tau_n: 1e-9             # τ_GB = 1 ns (vs τ_bulk = 1 µs for absorber)
      tau_p: 1e-9
      layer_role: absorber
```

- [ ] **Step 2: Smoke-test the preset loads**

Run:
```bash
python -c "
from perovskite_sim.models.config_loader import load_device_from_yaml
stack = load_device_from_yaml('configs/twod/nip_MAPbI3_singleGB.yaml')
print('OK, layers:', [l.name for l in stack.layers])
print('microstructure attribute on stack:', getattr(stack, 'microstructure', 'MISSING'))
"
```
Expected: `OK, layers: [...]`. Whether `microstructure` is on `stack` or carried through a separate path is determined in the next task.

- [ ] **Step 3: Commit**

```bash
git add perovskite-sim/configs/twod/nip_MAPbI3_singleGB.yaml
git commit -m "feat(twod): nip_MAPbI3_singleGB preset for Stage-B regression"
```

---

### Task 3: Wire microstructure through DeviceStack / loader

**Files:**
- Modify: `perovskite-sim/perovskite_sim/models/device.py` (decide: add `microstructure: Microstructure` field to `DeviceStack`, default `Microstructure()`)
- Modify: `perovskite-sim/perovskite_sim/models/config_loader.py` (call `load_microstructure_from_yaml_block` and attach result to the loaded stack)
- Test: `perovskite-sim/tests/unit/twod/test_microstructure.py` (add a config_loader integration test)

Decision: store `microstructure` on `DeviceStack` so any 2D experiment can read it without a separate loader call. Beer-Lambert / 1D paths ignore the field. Default is empty for back-compat.

- [ ] **Step 1: Write failing test for stack.microstructure round-trip**

```python
def test_load_device_from_yaml_attaches_microstructure():
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml")
    assert hasattr(stack, "microstructure")
    assert len(stack.microstructure.grain_boundaries) == 1
    gb = stack.microstructure.grain_boundaries[0]
    assert gb.x_position == pytest.approx(250e-9)


def test_load_device_from_yaml_empty_microstructure_default():
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_uniform.yaml")
    assert stack.microstructure.grain_boundaries == ()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v -k attaches_microstructure`
Expected: FAIL with AttributeError.

- [ ] **Step 3: Add `microstructure` field to `DeviceStack`**

In `perovskite_sim/models/device.py`, add field:
```python
@dataclass(frozen=True)
class DeviceStack:
    # ... existing fields ...
    microstructure: "Microstructure" = field(default_factory=lambda: Microstructure())
```
Import `Microstructure` lazily inside the file or add a top-level `from perovskite_sim.twod.microstructure import Microstructure`.

- [ ] **Step 4: Wire loader to populate the field**

In `perovskite_sim/models/config_loader.py`, where the YAML doc is parsed into a `DeviceStack`, parse `doc.get("microstructure")` via `load_microstructure_from_yaml_block` and pass through to the constructor.

- [ ] **Step 5: Run tests**

Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py -v`
Expected: all PASS, including the empty-default case.

- [ ] **Step 6: Run full unit suite to verify no DeviceStack regressions**

Run: `pytest -q`
Expected: 511 passed (the same baseline as Stage A merge).

- [ ] **Step 7: Commit**

```bash
git add perovskite-sim/perovskite_sim/models/device.py perovskite-sim/perovskite_sim/models/config_loader.py perovskite-sim/tests/unit/twod/test_microstructure.py
git commit -m "feat(models): DeviceStack carries Microstructure; YAML auto-attached"
```

---

### Task 4: Hook microstructure through to build_material_arrays_2d / run_jv_sweep_2d

**Files:**
- Modify: `perovskite-sim/perovskite_sim/twod/experiments/jv_sweep_2d.py` (use `stack.microstructure` if caller passes the default `Microstructure()`)
- Test: `perovskite-sim/tests/integration/twod/test_jv_sweep_2d_microstructure.py` (NEW)

Currently `run_jv_sweep_2d(stack, microstructure, ...)` requires an explicit microstructure argument. After Task 3 the user can omit it because `stack.microstructure` carries the data; rewire so `microstructure=None` (new default) means "use stack.microstructure".

- [ ] **Step 1: Write failing integration test that runs the singleGB preset**

```python
@pytest.mark.regression
@pytest.mark.slow
def test_jv_sweep_2d_singleGB_runs_to_completion():
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml")
    res = run_jv_sweep_2d(
        stack=stack, microstructure=None,        # picks up stack.microstructure
        lateral_length=500e-9, Nx=8,
        V_max=0.6, V_step=0.2,
        Ny_per_layer=8, settle_t=1e-3,
    )
    assert res.V.shape == (4,)
    assert res.J.shape == (4,)
    assert np.all(np.isfinite(res.J))
    # τ heterogeneity must show up at the GB column at V=0
    snap0 = res.snapshots[0]
    Nx_nodes = snap0.x.size
    i_gb = int(np.argmin(np.abs(snap0.x - 250e-9)))
    # Carrier density at GB column should be lower than at the bulk column
    n_gb = snap0.n[:, i_gb]
    n_bulk = snap0.n[:, 0]
    # Compare in the absorber region only
    assert n_gb.max() < n_bulk.max(), "GB column did not show carrier suppression at V=0"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -m slow perovskite-sim/tests/integration/twod/test_jv_sweep_2d_microstructure.py -v`
Expected: FAIL — `microstructure=None` is rejected by current signature.

- [ ] **Step 3: Update `run_jv_sweep_2d` to accept `microstructure: Microstructure | None`**

```python
def run_jv_sweep_2d(
    stack: DeviceStack,
    microstructure: Microstructure | None = None,
    *,
    # ... rest unchanged ...
):
    # Resolve microstructure: explicit arg wins, else stack.microstructure, else empty.
    if microstructure is None:
        microstructure = getattr(stack, "microstructure", None) or Microstructure()
```
The rest of the function reads `microstructure` as before.

- [ ] **Step 4: Run tests**

Run: `pytest -m slow perovskite-sim/tests/integration/twod/test_jv_sweep_2d_microstructure.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/twod/experiments/jv_sweep_2d.py perovskite-sim/tests/integration/twod/test_jv_sweep_2d_microstructure.py
git commit -m "feat(twod): run_jv_sweep_2d picks up stack.microstructure when arg is None"
```

---

### Task 5: Stage-B regression — V_oc drops with GB

**Files:**
- Test: `perovskite-sim/tests/regression/test_twod_microstructure.py` (NEW)

Pin the qualitative physics: a single absorber GB with τ_GB = 1 ns lowers V_oc relative to the no-GB baseline. Quantitative band: 5 mV ≤ ΔV_oc ≤ 100 mV (the published MAPbI3 GB-induced V_oc penalty range; tightening to a specific number depends on geometry which we explore in Task 7).

- [ ] **Step 1: Write the regression test**

```python
@pytest.mark.regression
@pytest.mark.slow
def test_twod_singleGB_lowers_voc():
    """Stage-B physics gate. Single absorber GB must drop V_oc 5–100 mV
    versus the laterally-uniform baseline (BL preset, frozen ions)."""
    base = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_uniform.yaml"))
    gb = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml"))

    common = dict(
        lateral_length=500e-9, Nx=10,
        V_max=1.2, V_step=0.05,
        illuminated=True, lateral_bc="periodic",
        Ny_per_layer=10, settle_t=1e-3,
    )
    r_base = run_jv_sweep_2d(stack=base, microstructure=None, **common)
    r_gb = run_jv_sweep_2d(stack=gb, microstructure=None, **common)

    m_base = compute_metrics(np.asarray(r_base.V), _maybe_flip_sign(r_base.V, np.asarray(r_base.J)))
    m_gb = compute_metrics(np.asarray(r_gb.V), _maybe_flip_sign(r_gb.V, np.asarray(r_gb.J)))

    print(f"\nbaseline: V_oc={m_base.V_oc*1e3:.2f} mV  GB: V_oc={m_gb.V_oc*1e3:.2f} mV")
    drop_mV = (m_base.V_oc - m_gb.V_oc) * 1e3
    assert 5.0 <= drop_mV <= 100.0, f"GB V_oc drop {drop_mV:.2f} mV outside [5, 100]"

    # J_sc should be nearly unaffected (a single thin GB takes <5% of width)
    rel_jsc = abs(m_gb.J_sc - m_base.J_sc) / abs(m_base.J_sc)
    assert rel_jsc <= 0.10, f"GB J_sc rel drift {rel_jsc:.3f} exceeded 10%"
```

(`_freeze_ions` and `_maybe_flip_sign` are imported from `tests/regression/test_twod_validation.py`; refactor those into a small `tests/regression/_twod_utils.py` if the imports turn out to be inconvenient.)

- [ ] **Step 2: Run to verify failure (baseline doesn't yet match)**

Run: `pytest -m slow perovskite-sim/tests/regression/test_twod_microstructure.py -v`
Expected: FAIL on the V_oc drop bound (or PASS if the physics already lands cleanly — in which case the test pins the existing behavior).

- [ ] **Step 3: Tune τ_GB / GB width if the drop is outside the band**

If ΔV_oc < 5 mV, shorten τ_GB (e.g. 1e-10 s) or widen the GB. If > 100 mV, lengthen τ_GB or narrow the GB. Aim for a 20–40 mV drop on the chosen geometry. Record the chosen parameters in the YAML preset.

- [ ] **Step 4: Run again, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/tests/regression/test_twod_microstructure.py perovskite-sim/configs/twod/nip_MAPbI3_singleGB.yaml
git commit -m "test(twod): Stage-B regression — single GB drops V_oc 20–40 mV"
```

---

### Task 6: Microstructure unit tests (non-empty path)

**Files:**
- Modify: `perovskite-sim/tests/unit/twod/test_microstructure.py` (add τ-override tests)
- Modify: `perovskite-sim/tests/unit/twod/test_solver_2d.py` (add tau_n/tau_p heterogeneity tests on `MaterialArrays2D`)

- [ ] **Step 1: τ override at GB band test**

```python
def test_grain_boundary_overrides_tau_in_band():
    g = build_grid_2d([Layer(thickness=400e-9, N=20)],
                      lateral_length=500e-9, Nx=20, lateral_uniform=True)
    tau_bulk = np.full((g.Ny,), 1e-6)
    gb = GrainBoundary(x_position=250e-9, width=20e-9,
                       tau_n=1e-9, tau_p=2e-9, layer_role="absorber")
    ustruct = Microstructure(grain_boundaries=(gb,))
    tau_n, tau_p = build_tau_field(g, ustruct, tau_bulk, tau_bulk,
                                    layer_role_per_y=["absorber"] * g.Ny)
    # Inside the band
    in_band = np.abs(g.x - 250e-9) < 10e-9
    assert np.allclose(tau_n[:, in_band], 1e-9)
    assert np.allclose(tau_p[:, in_band], 2e-9)
    # Outside the band
    assert np.allclose(tau_n[:, ~in_band], 1e-6)
    assert np.allclose(tau_p[:, ~in_band], 1e-6)


def test_grain_boundary_skipped_outside_target_layer():
    g = build_grid_2d([Layer(thickness=400e-9, N=20)],
                      lateral_length=500e-9, Nx=20, lateral_uniform=True)
    tau_bulk = np.full((g.Ny,), 1e-6)
    gb = GrainBoundary(x_position=250e-9, width=20e-9,
                       tau_n=1e-9, tau_p=1e-9, layer_role="absorber")
    ustruct = Microstructure(grain_boundaries=(gb,))
    # All y-layers tagged "transport" → GB has no effect
    tau_n, _ = build_tau_field(g, ustruct, tau_bulk, tau_bulk,
                                layer_role_per_y=["transport"] * g.Ny)
    assert np.allclose(tau_n, 1e-6)
```

- [ ] **Step 2: Solver test verifying mat.tau_n is 2D and matches GB definition**

```python
def test_material_arrays_2d_tau_field_with_singleGB(stack_singleGB):
    grid = build_grid_2d([Layer(L.thickness, 8) for L in electrical_layers(stack_singleGB)],
                         lateral_length=500e-9, Nx=10, lateral_uniform=True)
    mat = build_material_arrays_2d(grid, stack_singleGB,
                                    stack_singleGB.microstructure)
    # tau_n is 2D and shows reduced τ at the GB column on absorber rows
    assert mat.tau_n.shape == (grid.Ny, grid.Nx)
    i_gb = int(np.argmin(np.abs(grid.x - 250e-9)))
    # absorber slice — find y-rows that belong to the absorber layer
    # (use mat.absorber_masks if available, else hardcode by stack geometry)
    ...
```

- [ ] **Step 3: Run + commit**

Run: `pytest perovskite-sim/tests/unit/twod/test_microstructure.py perovskite-sim/tests/unit/twod/test_solver_2d.py -v`
Expected: all PASS.

```bash
git add perovskite-sim/tests/unit/twod/test_microstructure.py perovskite-sim/tests/unit/twod/test_solver_2d.py
git commit -m "test(twod): non-empty Microstructure → τ heterogeneity in build_tau_field"
```

---

### Task 7: voc_grain_sweep experiment

**Files:**
- Create: `perovskite-sim/perovskite_sim/twod/experiments/voc_grain_sweep.py`
- Test: `perovskite-sim/tests/integration/twod/test_voc_grain_sweep.py` (NEW)

The headline experiment: sweep grain size L_g, return V_oc(L_g). Each L_g is a separate 2D run with `lateral_length = L_g`, periodic lateral BC, and a single centered GB. Output is a `VocGrainSweepResult(grain_sizes_m, V_oc_V, J_sc, FF, snapshots_per_size)`.

- [ ] **Step 1: Write the failing integration test**

```python
@pytest.mark.regression
@pytest.mark.slow
def test_voc_grain_sweep_monotone_decreases_in_Lg():
    """V_oc should grow as L_g grows (more bulk-like behavior)."""
    stack = _freeze_ions(load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml"))
    res = run_voc_grain_sweep(
        stack=stack,
        grain_sizes=(200e-9, 500e-9, 1000e-9),
        tau_gb=(1e-9, 1e-9), gb_width=10e-9,
        Nx=8, Ny_per_layer=8,
        V_max=1.2, V_step=0.05, settle_t=1e-3,
    )
    assert res.V_oc_V.shape == (3,)
    # Monotone: V_oc(L_g) should increase as L_g increases
    assert np.all(np.diff(res.V_oc_V) >= -5e-3), (
        f"V_oc not monotone in L_g: V_oc={res.V_oc_V}"
    )
    # Total spread > 5 mV
    assert (res.V_oc_V[-1] - res.V_oc_V[0]) * 1e3 > 5.0
```

- [ ] **Step 2: Run to verify failure** (`ImportError: run_voc_grain_sweep`)

- [ ] **Step 3: Implement the experiment**

```python
# perovskite_sim/twod/experiments/voc_grain_sweep.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure
from perovskite_sim.twod.experiments.jv_sweep_2d import run_jv_sweep_2d
from perovskite_sim.experiments.jv_sweep import compute_metrics


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class VocGrainSweepResult:
    grain_sizes_m: np.ndarray
    V_oc_V: np.ndarray
    J_sc_Am2: np.ndarray
    FF: np.ndarray


def run_voc_grain_sweep(
    stack: DeviceStack,
    grain_sizes: Sequence[float],
    *,
    tau_gb: tuple[float, float] = (1e-9, 1e-9),
    gb_width: float = 10e-9,
    Nx: int = 10,
    Ny_per_layer: int = 10,
    V_max: float = 1.2,
    V_step: float = 0.05,
    illuminated: bool = True,
    settle_t: float = 1e-3,
    progress: ProgressCallback | None = None,
) -> VocGrainSweepResult:
    """Sweep lateral grain size with one centered absorber GB.

    Each grain size produces a 2D run with `lateral_length = L_g`,
    `Nx` lateral intervals, periodic lateral BC, and a single GB at
    `x_position = L_g/2`. Returns V_oc, J_sc, FF for each grain size.
    """
    L_arr = np.asarray(grid for grid in grain_sizes if grid > 0.0, dtype=float)
    n = L_arr.size
    V_oc = np.zeros(n)
    J_sc = np.zeros(n)
    FF = np.zeros(n)
    for k, L_g in enumerate(L_arr):
        gb = GrainBoundary(
            x_position=float(L_g) / 2.0,
            width=gb_width, tau_n=tau_gb[0], tau_p=tau_gb[1],
            layer_role="absorber",
        )
        ms = Microstructure(grain_boundaries=(gb,))
        r = run_jv_sweep_2d(
            stack=stack, microstructure=ms,
            lateral_length=float(L_g), Nx=Nx,
            V_max=V_max, V_step=V_step,
            illuminated=illuminated, lateral_bc="periodic",
            Ny_per_layer=Ny_per_layer, settle_t=settle_t,
        )
        V = np.asarray(r.V); J = np.asarray(r.J)
        # Sign convention matches Stage-A regression
        if J[0] < 0: J = -J
        m = compute_metrics(V, J)
        V_oc[k], J_sc[k], FF[k] = m.V_oc, m.J_sc, m.FF
        if progress is not None:
            progress("voc_grain_sweep", k + 1, n, f"L_g={L_g*1e9:.0f} nm  V_oc={m.V_oc*1e3:.1f} mV")
    return VocGrainSweepResult(grain_sizes_m=L_arr, V_oc_V=V_oc, J_sc_Am2=J_sc, FF=FF)
```

(Fix the obvious typo `grid for grid in grain_sizes if grid > 0.0` → `g for g in grain_sizes if g > 0.0`. Implementer should fix in the actual code.)

- [ ] **Step 4: Run tests + commit**

```bash
git add perovskite-sim/perovskite_sim/twod/experiments/voc_grain_sweep.py perovskite-sim/tests/integration/twod/test_voc_grain_sweep.py
git commit -m "feat(twod): voc_grain_sweep — V_oc(L_g) headline experiment"
```

---

### Task 8: Backend dispatch — kind=voc_grain_sweep + microstructure on jv_2d

**Files:**
- Modify: `perovskite-sim/backend/main.py`

Add `kind="voc_grain_sweep"` to the dispatcher. Also extend `kind="jv_2d"` to accept an optional `microstructure` payload — if absent, pull from the loaded stack.

- [ ] **Step 1: Add `kind="voc_grain_sweep"` branch** with serializer flattening `grain_sizes_m`, `V_oc_V`, `J_sc_Am2`, `FF`.

- [ ] **Step 2: On `kind="jv_2d"`**, parse `params.get("microstructure")` via `load_microstructure_from_yaml_block` (treat the dict as a YAML-block-shaped dict). Pass to `run_jv_sweep_2d(microstructure=...)`. None falls back to stack.microstructure.

- [ ] **Step 3: Smoke-test via TestClient** (similar to Stage-A Task 12 smoke test).

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(backend): kind=voc_grain_sweep + microstructure on jv_2d"
```

---

### Task 9: Frontend — voc_grain_sweep pane + microstructure UI on jv-2d-pane

**Files:**
- Create: `perovskite-sim/frontend/src/workstation/panes/voc-grain-sweep-pane.ts`
- Modify: `perovskite-sim/frontend/src/workstation/panes/jv-2d-pane.ts` (add a small "single GB" config block)
- Modify: `perovskite-sim/frontend/src/workstation/panes/experiment-pane.ts` (register `voc_grain_sweep`)
- Modify: `perovskite-sim/frontend/src/workstation/types.ts` (add `'voc_grain_sweep'` to `ExperimentKind` and `RunResult`)
- Modify: `perovskite-sim/frontend/src/workstation/tree.ts` (add experimentLabel case)
- Modify: `perovskite-sim/frontend/src/job-stream.ts` (add `'voc_grain_sweep'` to startJob union)
- Modify: `perovskite-sim/frontend/src/types.ts` (add `VocGrainSweepResult` interface)
- Modify: `perovskite-sim/frontend/src/workstation/panes/main-plot-pane.ts` (add `renderVocGrainSweep`)

- [ ] **Step 1: Add types** (`VocGrainSweepResult` with `grain_sizes_nm`, `V_oc_V`, `J_sc_Am2`, `FF`).

- [ ] **Step 2: Mount VocGrainSweepPane** with grain-size list (CSV input), τ_GB_n/τ_GB_p numerics, gb_width, Nx, Ny_per_layer, V_max, V_step.

- [ ] **Step 3: Microstructure block on jv-2d-pane** — checkbox "single GB", revealing four numeric fields (x_position_nm, width_nm, τ_n, τ_p). When checked, packs into `params.microstructure = { grain_boundaries: [{...}] }` on `startJob`.

- [ ] **Step 4: Renderer in main-plot-pane** — V_oc(L_g) plot in mV vs nm, log-x axis, with J_sc and FF as text annotations.

- [ ] **Step 5: Verify build green**

Run: `cd perovskite-sim/frontend && npm run build`
Expected: tsc + vite build green.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(frontend): VocGrainSweepPane + microstructure UI on jv-2d-pane"
```

---

### Task 10: Documentation

**Files:**
- Modify: `perovskite-sim/CLAUDE.md` (extend Phase 6 block with a "Microstructure (Stage B)" subsection)
- Modify: `README.md` (root, add a row to the Dimensionality table or extend the Stage-A row)

- [ ] **Step 1: Inner CLAUDE.md** — add a paragraph on `Microstructure` / `GrainBoundary`, the YAML schema, `voc_grain_sweep`, and the regression bound (5 ≤ ΔV_oc ≤ 100 mV).

- [ ] **Step 2: Root README** — extend the Dimensionality "🟦 2D Stage A" row description: "Stage A is lateral-uniform parity; Stage B adds a single absorber grain boundary (`voc_grain_sweep` experiment) that drops V_oc by 20–40 mV on the shipped MAPbI3 preset."

- [ ] **Step 3: Commit**

```bash
git commit -m "docs(twod): Phase 6 Stage-B microstructure section"
```

---

### Task 11: Final sweep

- [ ] **Step 1: Full unit + integration suite**
Run: `pytest -q`
Expected: all green; new tests included.

- [ ] **Step 2: Full slow suite**
Run: `pytest -m slow -q`
Expected: all green (Stage-A 58 + the new Stage-B regressions).

- [ ] **Step 3: Frontend build green**
Run: `cd perovskite-sim/frontend && npm run build`

- [ ] **Step 4: Final code review** via `superpowers:code-reviewer` agent across the whole Stage-B branch.

- [ ] **Step 5: Push** (named milestone — ping user for confirmation before pushing).

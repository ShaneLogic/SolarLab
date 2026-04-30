# Stage B(c.3) — 2D Self-Consistent Radiative Reabsorption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 1D Phase 3.1b self-consistent radiative reabsorption hook to the 2D solver. On every RHS call, integrate `R_tot_2D = ∬ B(y,x)·n(y,x)·p(y,x) dy dx` over each absorber's 2D cells and feed back the non-escaping fraction `G_rad = R_tot_2D · (1 − P_esc) / area` uniformly across the absorber.

**Architecture:** (1) New pure module `perovskite_sim/twod/radiative_reabsorption_2d.py` hosts the `recompute_g_with_rad_2d` helper (per-RHS, returns augmented `(Ny, Nx)` G). (2) `MaterialArrays2D` carries 5 new fields (1 flag + 4 parallel tuples per absorber). (3) `build_material_arrays_2d` translates the existing 1D `mat1d.absorber_*` tuples to 2D y-ranges. (4) `assemble_rhs_2d` guards on `mat.has_radiative_reabsorption_2d`: True path calls the helper and forwards augmented `G_to_use` to `continuity_rhs_2d`. False path is unchanged. (5) `jv_sweep_2d.run_jv_sweep_2d` adds a 1D-style lagged fallback: on `run_transient_2d` failure, bake `R_tot` once at entry state and retry with the disabled flag.

**Tech Stack:** Python/NumPy, `scipy.integrate.solve_ivp(Radau)`, existing `perovskite_sim` library; ONE new file (`radiative_reabsorption_2d.py`), no backend or frontend changes, no YAML schema changes, no changes to `perovskite_sim/physics/photon_recycling.py`, `perovskite_sim/solver/mol.py`, `perovskite_sim/models/parameters.py`, or `perovskite_sim/models/config_loader.py`.

---

## File Structure

| File | Role |
|------|------|
| `perovskite_sim/twod/radiative_reabsorption_2d.py` | **NEW** — pure helper `recompute_g_with_rad_2d` + shape validators |
| `perovskite_sim/twod/solver_2d.py` | Add 5 fields to `MaterialArrays2D`; build path translates 1D `mat1d.absorber_*` tuples; `assemble_rhs_2d` adds an `if mat.has_radiative_reabsorption_2d:` branch |
| `perovskite_sim/twod/experiments/jv_sweep_2d.py` | Add `_bake_radiative_reabsorption_step_2d` helper + try/except retry around `run_transient_2d` |
| `tests/unit/twod/test_radiative_reabsorption_2d.py` | **NEW** — unit tests for the helper (~7) |
| `tests/unit/twod/test_solver_2d.py` | Add T5/T6/T7 wiring + tier-gate + finite-RHS smoke tests |
| `tests/regression/test_twod_validation.py` | Add T1 disabled-bit-identical, T2 1D↔2D parity, T3 V_oc-boost, T4 coexistence smoke |
| `perovskite_sim/CLAUDE.md` | Stage B(c.3) section |

**Current sizes / projected adds:**
- `radiative_reabsorption_2d.py` = 0 → ~150 lines (NEW)
- `solver_2d.py` = ~720 → ~780 lines
- `experiments/jv_sweep_2d.py` = ~195 → ~245 lines

All within the 800-line cap.

---

## Background for the implementer

The 1D solver implements self-consistent radiative reabsorption in `solver/mol.py:874–895` via a per-RHS hook that integrates `R_tot = ∫ B·n·p dx` over each absorber and adds `R_tot · (1 − P_esc) / thickness` back as a uniform G_rad on absorber rows. The harness (`experiments/jv_sweep.py:328–378`) provides a lagged fallback that bakes R_tot once per voltage step on Newton-failure (TMM-knee stiffness at V≈0.21V).

Stage B(c.3) ports both: the per-RHS hook to `assemble_rhs_2d` and the lagged fallback to `jv_sweep_2d.run_jv_sweep_2d`. **Approved formulation:**
```
R_tot_2D = ∬ B(y,x) · n(y,x) · p(y,x) dy dx     over absorber rows × all x
area     = thickness × lateral_length            (precomputed at build)
G_rad    = R_tot_2D · (1 − P_esc) / area         (uniform over absorber 2D area)
G[absorber_y_range, :] += G_rad
```

Bit-equivalent to 1D in the lateral-uniform limit. Optical-profile-weighted redistribution is **explicitly deferred**.

**Sign and Einstein-relation rules (do NOT change):**
- `R_tot` is non-negative (it's an integral of a positive quantity).
- `(1 − P_esc)` is in [0, 1] (P_esc is the escape probability).
- `G_rad` is non-negative (added as a generation source).
- The cached `mat.G_optical` is **never mutated** — always work on a copy.

**Activation gate (mirror 1D):**
```
has_radiative_reabsorption_2d = sim_mode.use_radiative_reabsorption
                              AND sim_mode.use_photon_recycling
                              AND mat1d.has_radiative_reabsorption
```
The third clause means the 2D side reuses the 1D side's per-absorber tuples — no duplicate "find the absorber, compute P_esc" logic.

**Constraints (user-imposed):**
1. Disabled-path bit-identical to current Stage B(c.2).
2. No YAML schema changes.
3. No changes to `physics/photon_recycling.py`, `solver/mol.py`, `models/parameters.py`, `models/config_loader.py`.
4. Use the existing 1D `compute_p_esc_for_absorber` primitive (via `mat1d.absorber_p_esc`).
5. Optical-profile redistribution explicitly deferred.
6. Strict shape checks on all face/range arguments — fail early with clear errors.
7. Lagged fallback must mirror 1D pattern: default per-RHS self-consistent; retry only on Newton failure.
8. No broad cleanup mixed into B(c.3) milestone commits — carry-forward items tracked separately.

**Run all tests from the `perovskite-sim/` directory:**
```bash
cd perovskite-sim
pytest tests/unit/twod/                                  # fast unit tests
pytest -m slow                                           # slow regression incl. parity gates
pytest tests/regression/test_twod_validation.py -v -s -m slow
```

---

## Task 1: Pure helper module `radiative_reabsorption_2d.py`

**Goal:** Create the per-RHS helper as a pure function with shape validators. No solver dependencies. Bit-equivalent to 1D in the lateral-uniform limit.

**Files:**
- Create: `perovskite_sim/twod/radiative_reabsorption_2d.py`
- Test: `tests/unit/twod/test_radiative_reabsorption_2d.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/twod/test_radiative_reabsorption_2d.py`:

```python
from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d


def _setup(Ny=8, Nx=5, lateral=1e-6, n_const=1e16, p_const=1e16, B_rad=4e-17):
    x = np.linspace(0.0, lateral, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n = np.full((Ny, Nx), n_const)
    p = np.full((Ny, Nx), p_const)
    B = np.full((Ny, Nx), B_rad)
    G_optical = np.zeros((Ny, Nx))
    return x, y, n, p, B, G_optical


def test_recompute_returns_new_array_with_same_shape():
    """Result has shape (Ny, Nx) and is a NEW array (caller's G_optical not mutated)."""
    x, y, n, p, B, G = _setup()
    G_in = G.copy()
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2.0e-7 * 1e-6,),   # thickness 200 nm × lateral 1 µm
    )
    assert G_out.shape == G_in.shape
    assert G_out is not G_in
    np.testing.assert_array_equal(G_in, np.zeros_like(G_in))   # original untouched


def test_recompute_no_absorbers_returns_g_optical_copy():
    """Empty tuples → returned G is a copy of G_optical (no augmentation)."""
    x, y, n, p, B, G = _setup()
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=(),
        absorber_p_esc=(),
        absorber_areas=(),
    )
    np.testing.assert_array_equal(G_out, G_in)
    assert G_out is not G_in


def test_recompute_p_esc_one_no_augmentation():
    """P_esc = 1.0 → no reabsorption (everything escapes) → G_out == G_optical."""
    x, y, n, p, B, G = _setup()
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(1.0,),
        absorber_areas=(2.0e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out, G_in)


def test_recompute_uniform_state_lateral_extension_matches_1d():
    """Lateral-uniform n,p,B → G_rad must reduce to the 1D formula
    R_tot_1D · (1 − P_esc) / thickness, exactly. This catches a missing
    lateral_length factor in the area calculation."""
    Ny, Nx = 8, 5
    lateral = 1e-6                           # 1 µm
    thickness = 2.0e-7                       # 200 nm absorber span
    x = np.linspace(0.0, lateral, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n_const = 1e22
    p_const = 1e22
    B_rad = 4e-17
    n = np.full((Ny, Nx), n_const)
    p = np.full((Ny, Nx), p_const)
    B = np.full((Ny, Nx), B_rad)
    G_in = np.zeros((Ny, Nx))
    p_esc = 0.05
    y_lo, y_hi = 2, 6                        # absorber rows 2..5
    # Force y[2..5] to span exactly thickness (200 nm) so the 1D analog is well-defined.
    y_abs = np.linspace(0.0, thickness, y_hi - y_lo)
    y[y_lo:y_hi] = y_abs
    area = thickness * lateral
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((y_lo, y_hi),),
        absorber_p_esc=(p_esc,),
        absorber_areas=(area,),
    )
    # Expected 1D analog: R_tot_1D = B·n·p · thickness; G_rad = R_tot_1D · (1−P_esc) / thickness
    #                            = B·n·p · (1 − P_esc)
    expected_g_rad = B_rad * n_const * p_const * (1.0 - p_esc)
    # All absorber cells get this value uniformly (additively, base was 0)
    np.testing.assert_allclose(G_out[y_lo:y_hi, :], expected_g_rad, rtol=1e-12)
    # Non-absorber rows untouched
    np.testing.assert_array_equal(G_out[:y_lo, :], 0.0)
    np.testing.assert_array_equal(G_out[y_hi:, :], 0.0)


def test_recompute_only_absorber_rows_augmented():
    """Non-absorber rows must remain bit-identical to G_optical."""
    Ny, Nx = 10, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n = np.full((Ny, Nx), 1e22)
    p = np.full((Ny, Nx), 1e22)
    B = np.full((Ny, Nx), 4e-17)
    G_in = np.full((Ny, Nx), 7.0e25)         # non-zero baseline for visibility
    y_lo, y_hi = 4, 8
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((y_lo, y_hi),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out[:y_lo, :], G_in[:y_lo, :])
    np.testing.assert_array_equal(G_out[y_hi:, :], G_in[y_hi:, :])
    assert np.all(G_out[y_lo:y_hi, :] >= G_in[y_lo:y_hi, :])


def test_recompute_zero_n_p_returns_g_optical_copy():
    """When n·p = 0 inside the absorber, R_tot = 0 and no augmentation occurs."""
    x, y, _, _, B, G = _setup()
    Ny, Nx = G.shape
    n0 = np.zeros((Ny, Nx))                  # n = 0 everywhere
    p0 = np.full((Ny, Nx), 1e22)
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n0, p=p0, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out, G_in)


def test_recompute_shape_mismatch_raises():
    """Wrong shape on G_optical raises ValueError with the field name."""
    x, y, n, p, B, _ = _setup()
    bad_G = np.zeros((n.shape[0] + 1, n.shape[1]))
    with pytest.raises(ValueError, match="G_optical"):
        recompute_g_with_rad_2d(
            G_optical=bad_G, n=n, p=p, B_rad=B, x=x, y=y,
            absorber_y_ranges=((2, 6),),
            absorber_p_esc=(0.5,),
            absorber_areas=(2e-7 * 1e-6,),
        )
```

- [ ] **Step 2: Run tests to verify they fail with module not found**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_radiative_reabsorption_2d.py -v
```

Expected: `ModuleNotFoundError: No module named 'perovskite_sim.twod.radiative_reabsorption_2d'`

- [ ] **Step 3: Create the new module**

Create `perovskite_sim/twod/radiative_reabsorption_2d.py`:

```python
"""Self-consistent radiative reabsorption recompute for the 2D solver
(Stage B(c.3)).

On every RHS call, for each absorber layer:
  R_tot_2D = ∬ B(y,x) · n(y,x) · p(y,x) dy dx     over absorber rows × all x
  area     = thickness × lateral_length            (precomputed at build time)
  G_rad    = R_tot_2D · (1 − P_esc) / area         (uniform over absorber 2D area)
  G[absorber_y_range, :] += G_rad

Bit-equivalent to 1D Phase 3.1b in the lateral-uniform limit. Optical-profile-
weighted redistribution is explicitly deferred — see
docs/superpowers/specs/2026-04-30-2d-stage-b-c3-radiative-reabsorption-design.md.

The cached G_optical is never mutated. The helper returns a NEW (Ny, Nx)
array equal to G_optical augmented per absorber.
"""
from __future__ import annotations
import numpy as np


def _check_2d_shape(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.3) shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny}, {Nx})."
        )


def recompute_g_with_rad_2d(
    *,
    G_optical: np.ndarray,                              # (Ny, Nx)
    n: np.ndarray,                                      # (Ny, Nx)
    p: np.ndarray,                                      # (Ny, Nx)
    B_rad: np.ndarray,                                  # (Ny, Nx)
    x: np.ndarray,                                      # (Nx,)
    y: np.ndarray,                                      # (Ny,)
    absorber_y_ranges: tuple[tuple[int, int], ...],
    absorber_p_esc: tuple[float, ...],
    absorber_areas: tuple[float, ...],
) -> np.ndarray:
    """Return a NEW (Ny, Nx) array equal to G_optical augmented with the
    self-consistent radiative reabsorption source per absorber.

    Per-absorber operations:
      1. Slice (n, p, B_rad) along the absorber's y-range.
      2. Integrate B·n·p over y first (axis=0), then over x → scalar R_tot.
      3. Skip if R_tot ≤ 0, area ≤ 0, P_esc ≥ 1, or fewer than 2 nodes
         on either axis (matches 1D mol.py:874-895 safety guards).
      4. G_rad = R_tot · (1 − P_esc) / area   (uniform over absorber area).
      5. G_with_rad[y_lo:y_hi, :] += G_rad.

    The lengths of ``absorber_y_ranges``, ``absorber_p_esc``, and
    ``absorber_areas`` must match (one entry per absorber).
    """
    Ny, Nx = G_optical.shape
    _check_2d_shape("G_optical", G_optical, Ny, Nx)
    _check_2d_shape("n", n, Ny, Nx)
    _check_2d_shape("p", p, Ny, Nx)
    _check_2d_shape("B_rad", B_rad, Ny, Nx)
    if x.shape != (Nx,):
        raise ValueError(
            f"Stage B(c.3): x shape {x.shape} != ({Nx},)"
        )
    if y.shape != (Ny,):
        raise ValueError(
            f"Stage B(c.3): y shape {y.shape} != ({Ny},)"
        )
    if not (len(absorber_y_ranges) == len(absorber_p_esc) == len(absorber_areas)):
        raise ValueError(
            f"Stage B(c.3): per-absorber tuple length mismatch — "
            f"y_ranges={len(absorber_y_ranges)}, "
            f"p_esc={len(absorber_p_esc)}, "
            f"areas={len(absorber_areas)}"
        )

    G_with_rad = G_optical.copy()
    for (y_lo, y_hi), p_esc, area in zip(
        absorber_y_ranges, absorber_p_esc, absorber_areas
    ):
        if area <= 0.0 or p_esc >= 1.0:
            continue
        if y_hi - y_lo < 2 or Nx < 2:
            continue
        emission = B_rad[y_lo:y_hi, :] * n[y_lo:y_hi, :] * p[y_lo:y_hi, :]   # (n_y_abs, Nx)
        # Integrate over y first (axis=0), giving (Nx,), then over x → scalar.
        emission_x = np.trapezoid(emission, y[y_lo:y_hi], axis=0)            # (Nx,)
        R_tot = float(np.trapezoid(emission_x, x))                            # scalar
        if R_tot <= 0.0:
            continue
        G_rad = R_tot * (1.0 - p_esc) / area
        G_with_rad[y_lo:y_hi, :] += G_rad
    return G_with_rad


__all__ = ["recompute_g_with_rad_2d"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_radiative_reabsorption_2d.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/radiative_reabsorption_2d.py tests/unit/twod/test_radiative_reabsorption_2d.py
git commit -m "feat(twod): add radiative_reabsorption_2d module — pure helper for Stage B(c.3)"
```

**Pass condition:** 7 unit tests passing.

**Rollback / fallback:** if any test fails, the issue is local to the new module. No external code is affected. Rollback: `git reset --hard HEAD~1`.

---

## Task 2: `MaterialArrays2D` field wiring + tier gate

**Goal:** Add 5 new fields (1 flag + 4 parallel tuples) to `MaterialArrays2D`. Build path translates 1D `mat1d.absorber_*` tuples to 2D y-ranges with strict contiguity validation. Tier gate: `use_radiative_reabsorption AND use_photon_recycling AND mat1d.has_radiative_reabsorption`.

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Test: `tests/unit/twod/test_solver_2d.py`

- [ ] **Step 1: Write the failing wiring tests**

Add to `tests/unit/twod/test_solver_2d.py` (the existing helpers `_stack`, `_layers_for_stack`, `_make_grid_and_mat`, `_stack_with_layer_params`, and the `dc_replace` import are already present):

```python
def test_material_arrays_2d_default_no_radiative_reabsorption():
    """Default BL preset → has_radiative_reabsorption_2d=False, all 4 tuples empty."""
    stack = _stack()                                 # BL → no TMM → no rr
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is False
    assert mat.absorber_y_ranges_2d == ()
    assert mat.absorber_p_esc_2d == ()
    assert mat.absorber_thicknesses_2d == ()
    assert mat.absorber_areas_2d == ()


def test_material_arrays_2d_tmm_full_mode_activates_radiative_reabsorption():
    """TMM preset with mode='full' → has_radiative_reabsorption_2d=True with
    one entry per absorber. Validates the build-path translation from
    mat1d.absorber_masks to 2D y-ranges."""
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is True
    assert len(mat.absorber_y_ranges_2d) >= 1
    assert len(mat.absorber_p_esc_2d) == len(mat.absorber_y_ranges_2d)
    assert len(mat.absorber_thicknesses_2d) == len(mat.absorber_y_ranges_2d)
    assert len(mat.absorber_areas_2d) == len(mat.absorber_y_ranges_2d)
    for (y_lo, y_hi) in mat.absorber_y_ranges_2d:
        assert 0 <= y_lo < y_hi <= g.Ny


def test_material_arrays_2d_absorber_y_ranges_match_layer_role_per_y():
    """absorber_y_ranges_2d indices must match layer_role_per_y == 'absorber'
    indices. This catches a wrong absorber mask."""
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is True
    absorber_indices = [j for j, r in enumerate(mat.layer_role_per_y) if r == "absorber"]
    # The single absorber's y-range should span exactly these indices.
    y_lo, y_hi = mat.absorber_y_ranges_2d[0]
    # y-range is half-open [y_lo, y_hi) so indices in range are y_lo..y_hi-1.
    range_indices = list(range(y_lo, y_hi))
    assert range_indices == absorber_indices, (
        f"absorber_y_ranges_2d[0] = ({y_lo}, {y_hi}) → {range_indices} does not "
        f"match layer_role_per_y 'absorber' indices {absorber_indices}"
    )


def test_material_arrays_2d_absorber_area_equals_thickness_times_lateral():
    """absorber_areas_2d entries must equal thickness × lateral_length."""
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    lateral = 425e-9
    g = build_grid_2d(layers, lateral_length=lateral, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is True
    for thickness, area in zip(mat.absorber_thicknesses_2d, mat.absorber_areas_2d):
        assert area == pytest.approx(thickness * lateral, rel=1e-12)


def test_legacy_mode_disables_radiative_reabsorption_in_2d():
    """Tier-as-ceiling: device.mode='legacy' must keep
    has_radiative_reabsorption_2d=False even on a TMM preset. Mirrors B(c.1)
    Issue I1 reprise pattern."""
    stack = dc_replace(
        load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml"),
        mode="legacy",
    )
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is False
    assert mat.absorber_y_ranges_2d == ()


def test_fast_mode_disables_radiative_reabsorption_in_2d():
    """FAST tier excludes per-RHS hooks per CLAUDE.md tier matrix:
    has_radiative_reabsorption_2d=False even on a TMM preset."""
    stack = dc_replace(
        load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml"),
        mode="fast",
    )
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_radiative_reabsorption_2d is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_default_no_radiative_reabsorption \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_tmm_full_mode_activates_radiative_reabsorption \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_absorber_y_ranges_match_layer_role_per_y \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_absorber_area_equals_thickness_times_lateral \
       tests/unit/twod/test_solver_2d.py::test_legacy_mode_disables_radiative_reabsorption_in_2d \
       tests/unit/twod/test_solver_2d.py::test_fast_mode_disables_radiative_reabsorption_in_2d -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'has_radiative_reabsorption_2d'`.

- [ ] **Step 3: Add 5 fields to `MaterialArrays2D` in `solver_2d.py`**

Locate the existing Stage B(c.2) field-mobility block (the `has_field_mobility`, `v_sat_n_x_face`, etc. fields). Append after them:

```python
    # --- Stage B(c.3): Self-consistent radiative reabsorption ------------------
    # When mat.has_radiative_reabsorption_2d is True, assemble_rhs_2d augments
    # G_optical per RHS call by summing R_tot_2D = ∬ B·n·p dy dx per absorber
    # and adding the non-escaping fraction back as a uniform G_rad over the
    # absorber 2D area. See docs/superpowers/specs/2026-04-30-2d-stage-b-c3-
    # radiative-reabsorption-design.md. The disabled path (flag=False) is
    # bit-identical to current Stage B(c.2).
    has_radiative_reabsorption_2d:  bool                               = False
    absorber_y_ranges_2d:           tuple[tuple[int, int], ...]        = ()
    absorber_p_esc_2d:              tuple[float, ...]                  = ()
    absorber_thicknesses_2d:        tuple[float, ...]                  = ()
    absorber_areas_2d:              tuple[float, ...]                  = ()
```

- [ ] **Step 4: Add population logic in `build_material_arrays_2d`**

Add the following block at the end of `build_material_arrays_2d` (after the Stage B(c.2) field-mobility block, before the `return MaterialArrays2D(...)` call). The 1D `mat1d` already has `absorber_masks`, `absorber_p_esc`, `absorber_thicknesses` populated when its own gate (`has_radiative_reabsorption`) is True.

```python
    # --- Stage B(c.3): Radiative reabsorption ----------------------------------
    # Translate 1D mat1d.absorber_* tuples to 2D y-range form. Activation gate
    # mirrors 1D mol.py exactly: requires both tier flags AND the 1D-side
    # build path to have produced the per-absorber tuples.
    from perovskite_sim.models.mode import resolve_mode as _resolve_mode_rr
    _sim_mode_rr = _resolve_mode_rr(getattr(stack, "mode", "full"))
    absorber_y_ranges_list:    list[tuple[int, int]] = []
    absorber_p_esc_list:       list[float]            = []
    absorber_thicknesses_list: list[float]            = []
    absorber_areas_list:       list[float]            = []
    _has_rr_2d = False

    if (_sim_mode_rr.use_radiative_reabsorption
            and _sim_mode_rr.use_photon_recycling
            and getattr(mat1d, "has_radiative_reabsorption", False)):
        _lateral_length = float(grid.x[-1] - grid.x[0])
        for _mask_1d, _p_esc, _thickness in zip(
            mat1d.absorber_masks, mat1d.absorber_p_esc, mat1d.absorber_thicknesses
        ):
            _y_indices = np.where(_mask_1d)[0]
            if _y_indices.size < 2:
                continue
            _y_lo = int(_y_indices[0])
            _y_hi = int(_y_indices[-1] + 1)
            # Sanity: 1D absorber masks are always contiguous; assert so 2D
            # half-open slicing is well-defined.
            if not bool(np.all(_mask_1d[_y_lo:_y_hi])):
                raise ValueError(
                    f"Stage B(c.3): absorber mask non-contiguous "
                    f"between y={_y_lo} and y={_y_hi}"
                )
            if _p_esc >= 1.0 or _thickness <= 0.0:
                continue
            absorber_y_ranges_list.append((_y_lo, _y_hi))
            absorber_p_esc_list.append(float(_p_esc))
            absorber_thicknesses_list.append(float(_thickness))
            absorber_areas_list.append(float(_thickness) * _lateral_length)
        _has_rr_2d = len(absorber_y_ranges_list) > 0

    absorber_y_ranges_2d   = tuple(absorber_y_ranges_list)
    absorber_p_esc_2d      = tuple(absorber_p_esc_list)
    absorber_thicknesses_2d = tuple(absorber_thicknesses_list)
    absorber_areas_2d      = tuple(absorber_areas_list)
```

- [ ] **Step 5: Pass the new fields into `MaterialArrays2D(...)`**

In the `return MaterialArrays2D(...)` call, append after the existing Stage B(c.2) fields:

```python
        has_radiative_reabsorption_2d=_has_rr_2d,
        absorber_y_ranges_2d=absorber_y_ranges_2d,
        absorber_p_esc_2d=absorber_p_esc_2d,
        absorber_thicknesses_2d=absorber_thicknesses_2d,
        absorber_areas_2d=absorber_areas_2d,
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py -v
```

Expected: all 6 new tests pass + existing 2D unit tests still pass (no regressions).

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): MaterialArrays2D radiative-reabsorption fields and build path"
```

**Pass condition:** 6 new wiring/gate tests pass + no regressions in `tests/unit/twod/`.

**Rollback / fallback:** if a test fails on TMM preset (the tier-gated activation), check that `mat1d.absorber_masks` is non-empty for that preset and that `_lateral_length = grid.x[-1] - grid.x[0]` matches the build args. If the contiguity assertion fires, the 1D side has changed shape — escalate. Rollback: `git reset --hard HEAD~1`.

---

## Task 3: `assemble_rhs_2d` recompute branch + finite-RHS smoke

**Goal:** Wire `assemble_rhs_2d` to call `recompute_g_with_rad_2d` when the flag is True; pass augmented `G_to_use` to `continuity_rhs_2d`. Disabled path remains bit-identical: `G_to_use is mat.G_optical` (same Python object).

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Test: `tests/unit/twod/test_solver_2d.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def test_assemble_rhs_2d_radiative_reabsorption_disabled_does_not_call_helper():
    """When has_radiative_reabsorption_2d=False, recompute_g_with_rad_2d is NOT
    called. Verified via mock — if it's called, the side_effect raises and the
    test fails."""
    from unittest.mock import patch
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack()                                 # BL → has_rr_2d=False
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is False
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    with patch(
        "perovskite_sim.twod.solver_2d.recompute_g_with_rad_2d",
        side_effect=RuntimeError("recompute called when has_rr_2d=False"),
    ):
        dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    assert np.all(np.isfinite(dydt))


def test_assemble_rhs_2d_radiative_reabsorption_enabled_calls_helper_and_finite():
    """When has_radiative_reabsorption_2d=True (TMM preset, mode='full'),
    recompute_g_with_rad_2d IS called and the resulting RHS is finite even at
    a steep n·p gradient (catches per-RHS integral overflow / mis-shaped trapezoid)."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is True
    # Build a state with steep y-gradient so n·p varies significantly inside the absorber.
    n_grad = np.linspace(float(mat.n_eq_left[0]), float(mat.n_eq_right[0]), g.Ny)
    p_grad = np.linspace(float(mat.p_eq_left[0]), float(mat.p_eq_right[0]), g.Ny)
    n0 = np.broadcast_to(n_grad[:, None], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(p_grad[:, None], (g.Ny, g.Nx)).copy()
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.5)
    assert np.all(np.isfinite(dydt)), "Stage B(c.3) RHS went non-finite at steep gradient"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_radiative_reabsorption_disabled_does_not_call_helper \
       tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_radiative_reabsorption_enabled_calls_helper_and_finite -v
```

Expected:
- The "disabled does not call helper" test PASSES (because `recompute_g_with_rad_2d` is not yet imported/called in `assemble_rhs_2d`, so the mock patch is a no-op).
- The "enabled calls helper and finite" test FAILS or has incorrect physics — the RHS may still be finite because the recompute is not yet wired, so `mat.G_optical` is used directly without the rr augmentation. The test should pass at this point too — actually it will, because the test just asserts finiteness, not that the helper was called. Adjust: make the test more specific by also asserting the helper IS called.

For a stricter "enabled calls helper" test, add this assertion:

```python
def test_assemble_rhs_2d_radiative_reabsorption_enabled_helper_is_invoked():
    """When has_radiative_reabsorption_2d=True, recompute_g_with_rad_2d IS called."""
    from unittest.mock import patch
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is True
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    with patch(
        "perovskite_sim.twod.solver_2d.recompute_g_with_rad_2d",
        wraps=__import__(
            "perovskite_sim.twod.radiative_reabsorption_2d", fromlist=["recompute_g_with_rad_2d"]
        ).recompute_g_with_rad_2d,
    ) as mock_helper:
        dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    assert mock_helper.called, "recompute_g_with_rad_2d not called when has_rr_2d=True"
    assert np.all(np.isfinite(dydt))
```

This test fails with `AssertionError` (helper not called) before the wiring; passes after.

- [ ] **Step 3: Wire `assemble_rhs_2d` to call the helper**

Add the import at the top of `solver_2d.py` (next to the other twod.* imports):

```python
from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d
```

In `assemble_rhs_2d`, find the existing call to `continuity_rhs_2d`. Both branches (`if mat.has_field_mobility:` and the `else`) currently pass `mat.G_optical` as the G argument. Replace with:

```python
    # --- Stage B(c.3): Radiative reabsorption recompute -----------------------
    # When the flag is False, G_to_use is mat.G_optical (same Python object,
    # bit-identical to current Stage B(c.2)).
    if mat.has_radiative_reabsorption_2d and mat.absorber_y_ranges_2d:
        G_to_use = recompute_g_with_rad_2d(
            G_optical=mat.G_optical, n=n, p=p, B_rad=mat.B_rad,
            x=g.x, y=g.y,
            absorber_y_ranges=mat.absorber_y_ranges_2d,
            absorber_p_esc=mat.absorber_p_esc_2d,
            absorber_areas=mat.absorber_areas_2d,
        )
    else:
        G_to_use = mat.G_optical
```

Place this block **before** the `if mat.has_field_mobility:` branch (or at any point after `R` is computed and before the `continuity_rhs_2d` call). Then in BOTH branches of the `continuity_rhs_2d` call, replace `mat.G_optical` with `G_to_use`:

```python
    # In the field-mobility True branch:
    dn, dp = continuity_rhs_2d(
        g.x, g.y, phi, n, p,
        G_to_use, R,                # was: mat.G_optical, R
        mat.D_n, mat.D_p,
        mat.V_T,
        ...                          # rest unchanged
    )
    # In the field-mobility False branch:
    dn, dp = continuity_rhs_2d(
        g.x, g.y, phi, n, p,
        G_to_use, R,                # was: mat.G_optical, R
        mat.D_n, mat.D_p,
        mat.V_T,
        ...                          # rest unchanged
    )
```

**CRITICAL:** when `mat.has_radiative_reabsorption_2d=False`, the `else` branch sets `G_to_use = mat.G_optical` (same object, no copy). The `continuity_rhs_2d` call list is identical to the prior Stage B(c.2) code. This is the disabled-path bit-identicality guarantee.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/ -v
```

Expected: all new tests pass + existing 2D unit tests still pass (no regressions).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): assemble_rhs_2d augments G_optical with radiative reabsorption when enabled"
```

**Pass condition:** all 3 new T3-related tests pass + no regressions in `tests/unit/twod/`.

**Rollback / fallback:** if `test_assemble_rhs_2d_radiative_reabsorption_disabled_does_not_call_helper` fails (helper called when it shouldn't be), the gate guard is wrong. Re-check the `if mat.has_radiative_reabsorption_2d and mat.absorber_y_ranges_2d:` condition. If the "enabled and finite" test produces NaN/Inf, the helper may be using the wrong axis order in trapezoid. Rollback: `git reset --hard HEAD~1`.

---

## Task 4: Lagged fallback in `jv_sweep_2d`

**Goal:** Mirror the 1D `_bake_radiative_reabsorption_step` pattern in `jv_sweep_2d`. Default path is fully self-consistent per-RHS; on `run_transient_2d` failure, retry once with `R_tot` baked from the entry state of the failed voltage step.

**Files:**
- Modify: `perovskite_sim/twod/experiments/jv_sweep_2d.py`
- Test: `tests/unit/twod/test_solver_2d.py`

- [ ] **Step 1: Write the failing test for the bake helper**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def test_bake_radiative_reabsorption_step_2d_no_op_when_disabled():
    """When mat.has_radiative_reabsorption_2d=False, the bake helper returns
    mat unchanged (same object)."""
    from perovskite_sim.twod.experiments.jv_sweep_2d import _bake_radiative_reabsorption_step_2d
    stack = _stack()                                 # BL → has_rr_2d=False
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is False
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y_state = np.concatenate([n0.flatten(), p0.flatten()])
    mat_baked = _bake_radiative_reabsorption_step_2d(y_state, mat, illuminated=True)
    assert mat_baked is mat                          # no-op


def test_bake_radiative_reabsorption_step_2d_clears_flag_and_augments_G():
    """When has_radiative_reabsorption_2d=True, the bake helper returns a NEW
    mat with the flag cleared, the absorber tuples emptied, and G_optical
    augmented per absorber. The retry then takes the disabled path with G
    already pre-baked."""
    from perovskite_sim.twod.experiments.jv_sweep_2d import _bake_radiative_reabsorption_step_2d
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is True
    # Build a non-trivial state: steep y-gradient so n·p > 0 inside absorber.
    n_grad = np.linspace(float(mat.n_eq_left[0]), float(mat.n_eq_right[0]), g.Ny)
    p_grad = np.linspace(float(mat.p_eq_left[0]), float(mat.p_eq_right[0]), g.Ny)
    n0 = np.broadcast_to(n_grad[:, None], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(p_grad[:, None], (g.Ny, g.Nx)).copy()
    y_state = np.concatenate([n0.flatten(), p0.flatten()])
    mat_baked = _bake_radiative_reabsorption_step_2d(y_state, mat, illuminated=True)
    assert mat_baked is not mat
    assert mat_baked.has_radiative_reabsorption_2d is False
    assert mat_baked.absorber_y_ranges_2d == ()
    assert mat_baked.absorber_p_esc_2d == ()
    assert mat_baked.absorber_thicknesses_2d == ()
    assert mat_baked.absorber_areas_2d == ()
    # G_optical was augmented (some absorber rows changed; depends on n·p sign).
    # In a normal device with non-zero n·p, the augmentation is positive.
    y_lo, y_hi = mat.absorber_y_ranges_2d[0]
    assert np.any(mat_baked.G_optical[y_lo:y_hi, :] > mat.G_optical[y_lo:y_hi, :]), (
        "Bake helper did not augment G_optical inside the absorber"
    )


def test_bake_radiative_reabsorption_step_2d_no_op_when_dark():
    """When illuminated=False, the bake helper is a no-op (matches 1D)."""
    from perovskite_sim.twod.experiments.jv_sweep_2d import _bake_radiative_reabsorption_step_2d
    stack = load_device_from_yaml("configs/nip_MAPbI3_tmm.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_radiative_reabsorption_2d is True
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y_state = np.concatenate([n0.flatten(), p0.flatten()])
    mat_baked = _bake_radiative_reabsorption_step_2d(y_state, mat, illuminated=False)
    assert mat_baked is mat
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py::test_bake_radiative_reabsorption_step_2d_no_op_when_disabled \
       tests/unit/twod/test_solver_2d.py::test_bake_radiative_reabsorption_step_2d_clears_flag_and_augments_G \
       tests/unit/twod/test_solver_2d.py::test_bake_radiative_reabsorption_step_2d_no_op_when_dark -v
```

Expected: FAIL with `ImportError: cannot import name '_bake_radiative_reabsorption_step_2d'`.

- [ ] **Step 3: Add the bake helper to `jv_sweep_2d.py`**

In `perovskite_sim/twod/experiments/jv_sweep_2d.py`, after the existing imports and before `run_jv_sweep_2d`, add:

```python
import dataclasses
from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d


def _bake_radiative_reabsorption_step_2d(
    y_state: np.ndarray, mat, illuminated: bool,
):
    """Freeze the Stage B(c.3) G_rad source for one ``run_transient_2d`` call.

    Stage B(c.3)'s per-RHS hook recomputes ``R_tot_2D = ∬ B·n·p dy dx`` inside
    every Radau Newton iteration, which couples every absorber cell to every
    other through a non-local integral. At low forward bias on TMM presets
    (V≈0.21V — see project memory `tmm_jv_regression_021.md`), the diode-
    injection knee can prevent Newton convergence on the dense absorber block.

    Fix (mirrors 1D ``_bake_radiative_reabsorption_step`` in jv_sweep.py): on
    ``run_transient_2d`` failure, evaluate ``R_tot_2D`` once at the entry state
    ``y_state``, fold ``G_rad`` into a step-local ``G_optical`` copy, and clear
    ``has_radiative_reabsorption_2d`` on the returned ``mat``. Across voltage
    steps the warm-start chain refreshes ``R_tot`` from the freshly-settled
    state, so the lag is bounded by ``n·p`` drift inside one settle interval —
    sub-percent on the typical ``v_rate=1 V/s`` sweep, well below the 5 mV V_oc
    parity window.

    No-op when ``has_radiative_reabsorption_2d=False``, ``illuminated=False``,
    or there are no absorbers — returns the original ``mat``.
    """
    if not (
        mat.has_radiative_reabsorption_2d
        and illuminated
        and mat.absorber_y_ranges_2d
    ):
        return mat
    g = mat.grid
    N = g.Ny * g.Nx
    n0 = y_state[:N].reshape((g.Ny, g.Nx))
    p0 = y_state[N:].reshape((g.Ny, g.Nx))
    G_with_rad = recompute_g_with_rad_2d(
        G_optical=mat.G_optical, n=n0, p=p0, B_rad=mat.B_rad,
        x=g.x, y=g.y,
        absorber_y_ranges=mat.absorber_y_ranges_2d,
        absorber_p_esc=mat.absorber_p_esc_2d,
        absorber_areas=mat.absorber_areas_2d,
    )
    return dataclasses.replace(
        mat,
        G_optical=G_with_rad,
        has_radiative_reabsorption_2d=False,
        absorber_y_ranges_2d=(),
        absorber_p_esc_2d=(),
        absorber_thicknesses_2d=(),
        absorber_areas_2d=(),
    )
```

- [ ] **Step 4: Wire the retry into the voltage loop**

In `run_jv_sweep_2d`, find the voltage loop. Replace the bare `run_transient_2d` call with a try/except retry pattern:

```python
    for k, V in enumerate(voltages):
        try:
            y_state = run_transient_2d(
                y_state, mat,
                V_app=float(V),
                t_end=settle_t,
                max_step=settle_t / 50.0,
            )
        except RuntimeError:
            # Stage B(c.3) lagged fallback: bake R_tot once at the entry state,
            # retry with the disabled flag. Mirrors 1D jv_sweep.py:328+.
            if not (mat.has_radiative_reabsorption_2d and illuminated):
                raise                                      # nothing to fall back on
            mat_step = _bake_radiative_reabsorption_step_2d(
                y_state, mat, illuminated=illuminated,
            )
            y_state = run_transient_2d(
                y_state, mat_step,
                V_app=float(V),
                t_end=settle_t,
                max_step=settle_t / 50.0,
            )
        snap = extract_snapshot_2d(y_state, mat, V_app=float(V))
        J_list.append(compute_terminal_current_2d(snap))
        if save_snapshots:
            snap_list.append(snap)
        if progress is not None:
            progress("jv_2d", k + 1, len(voltages), f"V = {V:.3f} V")
```

The original `mat` (with `has_radiative_reabsorption_2d=True`) is preserved across voltage steps; only the failed step uses the baked `mat_step`. The next voltage step warm-starts from `y_state` and re-enters the self-consistent path with the original `mat`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/ -v
```

Expected: 3 new bake-helper tests pass + no regressions.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/twod/experiments/jv_sweep_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): jv_sweep_2d lagged-fallback for Stage B(c.3) radiative reabsorption"
```

**Pass condition:** 3 new bake-helper tests pass + no regressions.

**Rollback / fallback:** if the no-op-when-dark or no-op-when-disabled tests fail, the gate condition is wrong — re-check `if not (mat.has_radiative_reabsorption_2d and illuminated and mat.absorber_y_ranges_2d):`. If the augmentation test fails, check that `recompute_g_with_rad_2d` is called with the right kwargs. Rollback: `git reset --hard HEAD~1`.

---

## Task 5: Disabled-path bit-identical regression (slow)

**Goal:** Pin a slow regression that confirms the disabled path is truly bit-identical when `has_radiative_reabsorption_2d=False`. Compare a BL preset under `mode='full'` vs `mode='legacy'` — both yield `has_rr_2d=False`, and on a chi=Eg=0 BL preset the other tier flags don't change physics either, so the J-V must match exactly.

**Files:**
- Modify: `tests/regression/test_twod_validation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_disabled_path_bit_identical():
    """Stage B(c.3) bit-identical disabled-path regression.

    Compare two 2D J-V sweeps on the BL preset:
      A. mode='full'  → has_radiative_reabsorption_2d=False (no TMM, no P_esc)
      B. mode='legacy' → has_radiative_reabsorption_2d=False (tier disables)

    Both must produce IDENTICAL J-V because Stage B(c.3) keeps the constant-G
    code path (G_to_use is mat.G_optical when the flag is False).
    The chi=Eg=0 BL preset means the other tier-gated flags (TE, band offsets,
    etc.) don't change physics either, so this test isolates the B(c.3) gate.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))    # PRESET is BL
    stack_full   = base                                   # mode=full default
    stack_legacy = replace(base, mode="legacy")
    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_full   = run_jv_sweep_2d(stack=stack_full,   **common_kw)
    r_legacy = run_jv_sweep_2d(stack=stack_legacy, **common_kw)
    np.testing.assert_array_equal(r_full.V, r_legacy.V)
    np.testing.assert_allclose(
        r_full.J, r_legacy.J, rtol=1e-12, atol=0.0,
        err_msg="Stage B(c.3) disabled path is not bit-identical between mode=full and mode=legacy on BL",
    )
```

- [ ] **Step 2: Run test to verify it passes (no implementation needed)**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_radiative_reabsorption_disabled_path_bit_identical -v -s -m slow
```

Expected: PASS — Tasks 1–4 preserved the disabled path bit-identically.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.3) disabled-path bit-identical regression"
```

**Pass condition:** test PASSES on first run.

**Rollback / fallback:** if the test FAILS (J not bit-identical between modes on BL), Stage B(c.3) introduced an inadvertent regression in the constant-G code path. Investigate: check that the `else` branch in `assemble_rhs_2d` sets `G_to_use = mat.G_optical` (same object). If the difference is non-trivial, rollback Tasks 3 and 4. If the difference is small, it may indicate the BL preset has chi/Eg ≠ 0 and a tier flag toggles physics — in that case the test design is wrong and needs to use a structurally-cleaner comparison (e.g., snapshot-pin against the prior commit). Rollback: `git reset --hard HEAD~1`.

---

## Task 6: 1D↔2D parity gate + V_oc-boost-in-[40,100] mV (slow)

**Goal:** Primary correctness gate. T2 confirms 2D Stage B(c.3) matches 1D Phase 3.1b on a TMM preset within tight tolerances. T3 confirms the V_oc boost vs PR-off lies in the literature window [40, 100] mV.

**Files:**
- Modify: `tests/regression/test_twod_validation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/regression/test_twod_validation.py`:

```python
RR_TMM_PRESET = "configs/nip_MAPbI3_tmm.yaml"


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_parity_vs_1d():
    """Stage B(c.3) primary correctness gate: laterally-uniform 2D with face-
    normal radiative reabsorption matches 1D Phase 3.1b within (5 mV / 5e-4 / 1e-3).

    Uses TMM preset because radiative reabsorption requires
    use_photon_recycling=True (P_esc only computed under TMM). The known TMM
    1D regression at V≈0.21V (project_tmm_jv_regression_021.md) is at the
    diode-injection knee; V_oc≈0.91V is far from there, so the V_oc-only
    parity assertion is unaffected.
    """
    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    # 1D reference
    r1 = run_jv_sweep(base, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)
    # 2D Stage B(c.3)
    r2 = run_jv_sweep_2d(
        stack=base,
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    V2 = np.asarray(r2.V)
    J2 = _maybe_flip_sign(V2, np.asarray(r2.J))
    m2 = compute_metrics(V2, J2)
    print(
        f"\nrr 1D: V_oc={m1.V_oc*1e3:.4f} mV  J_sc={m1.J_sc:.4f} A/m²  FF={m1.FF:.6f}"
        f"\nrr 2D: V_oc={m2.V_oc*1e3:.4f} mV  J_sc={m2.J_sc:.4f} A/m²  FF={m2.FF:.6f}"
        f"\nΔV_oc = {(m2.V_oc - m1.V_oc)*1e3:+.4f} mV"
        f"  ΔJ_sc/J_sc = {(m2.J_sc - m1.J_sc)/m1.J_sc:+.2e}"
        f"  ΔFF = {m2.FF - m1.FF:+.4e}"
    )
    assert abs(m2.V_oc - m1.V_oc) <= 5e-3, (
        f"rr V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 5 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"rr J_sc rel diff {rel_jsc:.2e} > 5e-4 (2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"rr FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} (diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_radiative_reabsorption_voc_boost_in_literature_window():
    """Stage B(c.3) V_oc boost vs PR-off lies in [40, 100] mV.

    Compares 2D mode='full' (reabsorption ON) vs mode='legacy' (reabsorption
    OFF, photon recycling OFF) on the same TMM preset. Mirrors 1D
    test_radiative_reabsorption_preserves_voc_boost.

    A boost outside [40, 100] mV indicates either: reabsorption is not active
    (boost would be near 0 — would also fail the tier gate test T6),
    double-counting (boost would be > 100 mV), or wrong area weighting
    (boost would be far from window).
    """
    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    stack_on  = base
    stack_off = replace(base, mode="legacy")
    common_kw = dict(
        microstructure=Microstructure(),
        lateral_length=500e-9,
        Nx=4,
        V_max=1.2,
        V_step=0.1,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=10,
        settle_t=1e-3,
    )
    r_on  = run_jv_sweep_2d(stack=stack_on,  **common_kw)
    r_off = run_jv_sweep_2d(stack=stack_off, **common_kw)
    V_on  = np.asarray(r_on.V);  J_on  = _maybe_flip_sign(V_on,  np.asarray(r_on.J))
    V_off = np.asarray(r_off.V); J_off = _maybe_flip_sign(V_off, np.asarray(r_off.J))
    m_on  = compute_metrics(V_on,  J_on)
    m_off = compute_metrics(V_off, J_off)
    boost_mV = (m_on.V_oc - m_off.V_oc) * 1e3
    print(
        f"\nrr OFF: V_oc={m_off.V_oc*1e3:.2f} mV"
        f"\nrr ON:  V_oc={m_on.V_oc*1e3:.2f} mV"
        f"\nΔV_oc(boost) = {boost_mV:.2f} mV"
    )
    assert 40.0 <= boost_mV <= 100.0, (
        f"V_oc boost = {boost_mV:.2f} mV outside literature window [40, 100] mV "
        f"(rr ON: {m_on.V_oc*1e3:.2f} mV, rr OFF: {m_off.V_oc*1e3:.2f} mV)"
    )
```

- [ ] **Step 2: Run tests**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_radiative_reabsorption_parity_vs_1d \
       tests/regression/test_twod_validation.py::test_twod_radiative_reabsorption_voc_boost_in_literature_window -v -s -m slow
```

Expected: both pass. The parity gate should print sub-mV deltas (or mV-level on TMM where the 1D side has finer-than-2D adaptive-solver behaviour). If parity FAILS:
1. Read the printed `ΔV_oc / ΔJ_sc / ΔFF` values.
2. Do **NOT** loosen tolerances. First diagnose: re-check the 2D recompute formula (axis order in trapezoid, missing `lateral_length` factor, sign error). Re-run T5 absorber-mask test to confirm the absorber rows are correct. Re-run T7 finite-RHS smoke.
3. If diagnosis points to genuine adaptive-solver noise (not a logic bug), report measured deltas to the user and pin near 3× the noise floor (capped at 10 mV V_oc / 1e-3 J_sc / 2e-3 FF per the user's directive).

If V_oc boost is outside [40, 100] mV:
1. If boost ~ 0: reabsorption is silently disabled. Re-check T6 tier-gate test.
2. If boost > 100 mV: possible double-counting (B·n·p in both R AND G). Re-read mol.py:864-873 commentary on why this is NOT double-counting in the uniform-n·p limit; confirm 2D code follows the same pattern.
3. If boost < 40 mV: redistribution may be wrong (e.g., dividing by `lateral_length²` instead of `area`). Re-check `area = thickness × lateral_length`.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.3) 1D↔2D parity gate and V_oc-boost-in-[40,100]mV"
```

**Pass condition:** both tests pass with measured V_oc parity within 5 mV and boost in [40, 100] mV.

**Rollback / fallback:** if T2 fails by > 10 mV V_oc, the recompute math is wrong — escalate. If T3 falls outside [40, 100] mV, the activation logic is wrong (tier gate misconfigured) or the redistribution is wrong (area weighting). Do NOT widen the [40, 100] window blindly — that's the literature value for MAPbI3 and a different value indicates a real bug. Rollback: `git reset --hard HEAD~1`.

---

## Task 7: Coexistence smoke + CLAUDE.md + push

**Goal:** Confirm Stage B(c.3) composes with B(c.1) Robin and B(c.2) μ(E) without solver hang. Document the new feature in `CLAUDE.md`. Run the full fast and slow suites green; push.

**Files:**
- Modify: `tests/regression/test_twod_validation.py`
- Modify: `perovskite_sim/CLAUDE.md`

- [ ] **Step 1: Write the coexistence smoke test**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
def test_twod_radiative_reabsorption_robin_field_mobility_coexistence_smoke():
    """All three per-RHS hooks (radiative reabsorption + Robin contacts +
    μ(E)) plus a grain boundary on a coarse mesh produce a finite, well-
    ordered J-V (no NaN/Inf, J_sc>0). Cheap test — proves the four
    physics paths compose without solver hang."""
    from perovskite_sim.twod.microstructure import GrainBoundary
    base = _freeze_ions(load_device_from_yaml(RR_TMM_PRESET))
    stack = replace(_stack_with_layer_params(base, v_sat_n=1e3, v_sat_p=1e3),
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )
    ms = Microstructure(grain_boundaries=(
        GrainBoundary(
            x_position=150e-9, width=5e-9,
            tau_n=5e-8, tau_p=5e-8,
            layer_role="absorber",
        ),
    ))
    r = run_jv_sweep_2d(
        stack=stack,
        microstructure=ms,
        lateral_length=300e-9,
        Nx=6,
        V_max=1.0,
        V_step=0.25,
        illuminated=True,
        lateral_bc="periodic",
        Ny_per_layer=5,
        settle_t=1e-4,
    )
    V = np.asarray(r.V)
    J = np.asarray(r.J)
    assert np.all(np.isfinite(V)), "Non-finite V in rr+Robin+μ(E)+GB sweep"
    assert np.all(np.isfinite(J)), "Non-finite J in rr+Robin+μ(E)+GB sweep"
    J_sc_sign = _maybe_flip_sign(V, J)[0]
    assert J_sc_sign > 0, "J_sc should be positive under illumination"
```

- [ ] **Step 2: Run the coexistence test**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_radiative_reabsorption_robin_field_mobility_coexistence_smoke -v -s
```

Expected: PASS. Runtime under 60 s.

- [ ] **Step 3: Run the full fast suite**

```bash
cd perovskite-sim
pytest 2>&1 | tail -3
```

Expected: 0 failures. Specifically, all Stage A / B(a) / B(c.1) / B(c.2) regressions still pass.

- [ ] **Step 4: Run the slow suite**

```bash
cd perovskite-sim
pytest -m slow 2>&1 | tail -3
```

Expected: 0 failures. Specifically:
- `test_twod_uniform_matches_1d_within_tolerance` (Stage A parity)
- `test_twod_robin_parity_vs_1d` and `test_twod_robin_parity_vs_1d_aggressive_blocking` (Stage B(c.1))
- `test_twod_field_mobility_parity_vs_1d` (Stage B(c.2))
- New B(c.3): T1 disabled, T2 parity, T3 boost
all green.

- [ ] **Step 5: Update `perovskite_sim/CLAUDE.md`**

Locate the Stage B(c.2) field-mobility section (search for "Stage B(c.2)"). Append after that paragraph:

```markdown
**2D self-consistent radiative reabsorption — Stage B(c.3) (Phase 6 — Apr 2026).** Ports the 1D Phase 3.1b reabsorption hook to `assemble_rhs_2d`. When `SimulationMode.use_radiative_reabsorption` is True AND `SimulationMode.use_photon_recycling` is True AND `MaterialArrays.has_radiative_reabsorption` is True (the 1D side has populated per-absorber tuples under TMM optics), `build_material_arrays_2d` translates `mat1d.absorber_masks` to 2D `absorber_y_ranges_2d` (each is a contiguous `(y_lo, y_hi)` half-open range) and pre-computes `absorber_areas_2d = thickness × lateral_length` per absorber.

On every RHS call, `assemble_rhs_2d` calls `recompute_g_with_rad_2d` (in `perovskite_sim/twod/radiative_reabsorption_2d.py`) which:

1. For each absorber: compute `emission = B_rad[y_lo:y_hi, :] · n[y_lo:y_hi, :] · p[y_lo:y_hi, :]` of shape `(n_y_abs, Nx)`.
2. Trapezoid in y first (axis=0) → `(Nx,)`; then trapezoid in x → scalar `R_tot_2D` [units: 1/(m·s)].
3. Skip if `R_tot_2D ≤ 0`, `area ≤ 0`, `P_esc ≥ 1`, or fewer than 2 nodes on either axis (matches 1D safety guards).
4. Compute `G_rad = R_tot_2D · (1 − P_esc) / area` [units: 1/(m³·s)].
5. Augment `G[y_lo:y_hi, :] += G_rad` (uniform redistribution over absorber 2D area).

The cached `mat.G_optical` is **never mutated** — the helper returns a NEW `(Ny, Nx)` array. The augmented `G_to_use` is then forwarded to `continuity_rhs_2d`. When the flag is False, `G_to_use is mat.G_optical` (same Python object) — bit-identical to Stage A / B(a) / B(c.1) / B(c.2).

**Bit-equivalent to 1D in the lateral-uniform limit.** With constant `dx`, the lateral trapezoid evaluates to `lateral_length × emission_y(y)`, and dividing by `area = thickness × lateral_length` recovers `emission_y / thickness` — exactly 1D's `G_rad = R_tot · (1 − P_esc) / thickness`.

**Lagged fallback in `jv_sweep_2d.run_jv_sweep_2d`.** Default per-RHS is fully self-consistent (Newton iters re-evaluate `R_tot_2D` from the live state). On `RuntimeError` from `run_transient_2d` AND `mat.has_radiative_reabsorption_2d` AND `illuminated`, `_bake_radiative_reabsorption_step_2d` evaluates `R_tot_2D` once at the entry state of the failed voltage step, folds `G_rad` into a step-local `G_optical` copy, clears the flag and zeros the absorber tuples, and retries `run_transient_2d` once with the baked `mat`. Across voltage steps the warm-start chain refreshes `R_tot` from the freshly-settled state, so the lag is bounded by `n·p` drift inside one settle interval. Mirrors 1D `experiments/jv_sweep.py:_bake_radiative_reabsorption_step` exactly.

**Activation gate.** `_has_rr_2d = sim_mode.use_radiative_reabsorption AND sim_mode.use_photon_recycling AND mat1d.has_radiative_reabsorption` — same tier-as-ceiling pattern established by Stage B(c.1) and B(c.2). LEGACY tier disables the hook even on TMM presets; FAST tier currently lists radiative reabsorption among the per-RHS hooks it skips, so FAST also stays on the disabled path. FULL enables when the preset supports it.

**Validation.** Lateral-uniform 2D on `nip_MAPbI3_tmm.yaml` matches 1D Phase 3.1b within sub-mV V_oc / 5e-4 J_sc / 1e-3 FF. Pinned by `tests/regression/test_twod_validation.py::test_twod_radiative_reabsorption_parity_vs_1d`. The V_oc boost vs PR-off lies in the literature window [40, 100] mV — pinned by `test_twod_radiative_reabsorption_voc_boost_in_literature_window`. A coexistence smoke test (`test_twod_radiative_reabsorption_robin_field_mobility_coexistence_smoke`) confirms reabsorption + Robin + μ(E) + GB compose without NaN/Inf or solver hang. The disabled-path bit-identity is regression-pinned by `test_twod_radiative_reabsorption_disabled_path_bit_identical`. Tier-gate is unit-pinned by `test_legacy_mode_disables_radiative_reabsorption_in_2d` and `test_fast_mode_disables_radiative_reabsorption_in_2d`. The absorber-mask correctness is unit-pinned by `test_material_arrays_2d_absorber_y_ranges_match_layer_role_per_y` (validates `absorber_y_ranges_2d` matches `layer_role_per_y == "absorber"` indices) and `test_material_arrays_2d_absorber_area_equals_thickness_times_lateral` (validates `area = thickness × lateral_length`).

**Out of scope.** Optical-profile-weighted redistribution (where `G_rad` would be weighted by `α(x)·I(x)` rather than uniform) is explicitly deferred. See `docs/superpowers/specs/2026-04-30-2d-stage-b-c3-radiative-reabsorption-design.md`. Per-grain absorber heterogeneity in the reabsorption integral, μ(T)-coupled `B_rad(T)` beyond the existing temperature-scaling hook, and any backend / frontend changes are out of scope for Stage B(c.3).
```

- [ ] **Step 6: Commit docs**

```bash
git add perovskite_sim/CLAUDE.md tests/regression/test_twod_validation.py
git commit -m "docs(twod): Stage B(c.3) radiative reabsorption section + coexistence smoke test"
```

- [ ] **Step 7: Push**

```bash
git push origin 2d-extension
```

**Pass condition:** coexistence smoke passes; full fast suite (~5 min) green; full slow suite (~17 min including new T2 + T3) green; push successful.

**Rollback / fallback:** if any pre-existing test fails (Stage A / B(a) / B(c.1) / B(c.2)), Stage B(c.3) introduced a regression — escalate. If only B(c.3)'s own tests fail, follow the per-task rollback. Do NOT push if the slow suite has any failure. Rollback: `git reset --hard HEAD~1` repeated for any T7 commits, then re-run the slow suite to confirm a clean baseline.

---

## Self-review checklist

**Spec coverage** (each spec section maps to at least one task):
- §1 approved formulation (uniform-over-absorber-area) — Tasks 1, 3 implement; Option B deferred per spec
- §2 1D Phase 3.1b reference (where, formula, lagged fallback, tests) — Background section + T1
- §3 numerical: E-field-from-phi, harmonic mean, apply_field_mobility, Einstein recovery — N/A (this is for B(c.2))
- §3 numerical for B(c.3): trapezoid integration, area weighting, augmentation — Tasks 1, 3
- §3.4 lagged fallback in `jv_sweep_2d._integrate_step` — Task 4
- §3.5 disabled-path bit-identical — Task 5 (regression) + Task 3 step 1 (mock-patch unit test)
- §4 5 new fields on `MaterialArrays2D` — Task 2
- §4.2 build path — Task 2
- §4.3 no other module changes — verified by file list (only twod/* + tests + CLAUDE.md)
- §4.4 no YAML schema changes — verified
- §4.5 tier gating — Task 2 includes `test_legacy_mode_disables_*` and `test_fast_mode_disables_*`
- §5.T1 disabled bit-identical — Task 5
- §5.T2 1D↔2D parity — Task 6
- §5.T3 V_oc boost — Task 6
- §5.T4 coexistence smoke — Task 7
- §5.T5 absorber-mask correctness — Task 2
- §5.T6 tier gate — Task 2
- §5.T7 finite-RHS smoke — Task 3
- §6 risk register: R1 stiffness — Task 4 lagged fallback; R2 absorber mask — Task 2 T5 test; R3 area weighting — Task 1 lateral-uniform-matches-1D test; R4 double-counting — documented in Background; R5 redistribution — Task 1 only-absorber-rows-augmented test; R6 disabled-path — Task 5; R7 coexistence — Task 7; R8 TMM regression — V_oc-only assertion in Task 6 test docstring

**Placeholder scan:** no "TBD", no "TODO", no "implement later" — every step has full code or exact command.

**Type consistency:**
- `recompute_g_with_rad_2d` signature uses keyword-only args throughout — all Tasks 3, 4 invoke with the same arg names.
- `MaterialArrays2D` field names (`has_radiative_reabsorption_2d`, `absorber_y_ranges_2d`, etc.) defined in Task 2 step 3 — referenced by exact name in Task 2 step 5 (return), Task 3 step 3 (recompute call), Task 4 step 3 (bake helper).
- `_bake_radiative_reabsorption_step_2d(y_state, mat, illuminated)` signature defined in Task 4 step 3 — referenced with the same arg names in Task 4 step 4 (retry loop).
- Test names cross-referenced: `test_legacy_mode_disables_radiative_reabsorption_in_2d` (Task 2) and `test_fast_mode_disables_radiative_reabsorption_in_2d` (Task 2) referenced by name in CLAUDE.md (Task 7 step 5).

**Constraints checklist (user-imposed):**
- Disabled-path bit-identical to current Stage B(c.2) → Task 5 explicit regression. ✓
- No YAML schema changes → confirmed: only `mode` field used (already present). ✓
- No `physics/photon_recycling.py` / `solver/mol.py` / `models/parameters.py` / `models/config_loader.py` changes → confirmed by file list. ✓
- Use existing 1D `compute_p_esc_for_absorber` primitive → Task 2 reads `mat1d.absorber_p_esc` (computed by 1D mol.py; 2D side does not call the primitive directly). ✓
- Optical-profile redistribution deferred → spec §1 + CLAUDE.md update (Task 7 step 5). ✓
- Strict shape checks → Task 1 `_check_2d_shape` + tuple-length check. ✓
- Lagged fallback only on Newton failure → Task 4 step 4 (retry inside `try/except RuntimeError`). ✓
- Carry-forward cleanup NOT mixed into milestone commits → no cleanup steps in any task; carry-forward items remain visible in spec §7. ✓

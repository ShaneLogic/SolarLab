# Stage B(c.2) — 2D Field-dependent mobility μ(E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 1D Phase 3.2 field-dependent mobility hook to the 2D solver using a face-normal formulation (x-faces use only `|E_x_face|`, y-faces use only `|E_y_face|`), keeping the disabled path bit-identical to the current Stage B(c.1) constant-mobility code path.

**Architecture:** (1) A new `perovskite_sim/twod/field_mobility_2d.py` module hosts pure arithmetic-mean face-array builders and the per-RHS recompute helper. (2) `sg_fluxes_2d_n/p` and `continuity_rhs_2d` gain optional keyword-only `D_*_x_face` / `D_*_y_face` / `D_*_wrap` overrides — when `None` the harmonic-mean code path runs unchanged. (3) `MaterialArrays2D` carries 18 face arrays + a `has_field_mobility` flag, populated by `build_material_arrays_2d` from the existing `MaterialParams` fields. (4) `assemble_rhs_2d` guards on `mat.has_field_mobility`: True path computes `E_x_face` / `E_y_face` from the Poisson-solved `phi`, applies the existing 1D `apply_field_mobility` primitive per axis per carrier, recovers `D_eff` via Einstein relation, and forwards to `continuity_rhs_2d` via the new override kwargs. False path is unchanged.

**Tech Stack:** Python/NumPy, `scipy.integrate.solve_ivp(Radau)`, existing `perovskite_sim` library; ONE new file (`field_mobility_2d.py`), no backend or frontend changes, no YAML schema changes, no changes to `perovskite_sim/physics/field_mobility.py`.

---

## File Structure

| File | Role |
|------|------|
| `perovskite_sim/twod/field_mobility_2d.py` | **NEW** — pure arithmetic-mean face-array builders + the per-RHS recompute helper |
| `perovskite_sim/twod/flux_2d.py` | Add optional keyword-only `D_n_x_face` / `D_n_y_face` (and `D_p_*`) overrides to `sg_fluxes_2d_n/p` |
| `perovskite_sim/twod/continuity_2d.py` | Forward optional `D_n_x_face` / `D_n_y_face` / `D_p_x_face` / `D_p_y_face` / `D_n_wrap` / `D_p_wrap` to the flux helpers and the wrap-face block |
| `perovskite_sim/twod/solver_2d.py` | Add 19 fields to `MaterialArrays2D` (1 flag + 18 face arrays); populate in `build_material_arrays_2d`; gate the recompute in `assemble_rhs_2d` |
| `tests/unit/twod/test_field_mobility_2d.py` | **NEW** — unit tests for the pure builders and the recompute helper (Tasks 1, 4) |
| `tests/unit/twod/test_solver_2d.py` | Add `MaterialArrays2D` field-mobility wiring tests (Task 3) and tier-gate / periodic-wrap tests (Task 7) |
| `tests/unit/twod/test_flux_2d.py` | **NEW** — unit tests for the new override kwargs (Task 2) |
| `tests/regression/test_twod_validation.py` | Add disabled-path bit-identical (Task 5), 1D↔2D parity + bounded-shift (Task 6), coexistence smoke (Task 7) |
| `perovskite_sim/CLAUDE.md` | Stage B(c.2) section (Task 7) |

**Current sizes / projected adds:**
- `field_mobility_2d.py` = 0 → ~140 lines (NEW)
- `flux_2d.py` = 70 → ~95 lines
- `continuity_2d.py` = 237 → ~265 lines
- `solver_2d.py` = 602 → ~700 lines

All within the 800-line cap. No splits required.

---

## Background for the implementer

The 1D solver implements field-dependent mobility in `perovskite_sim/solver/mol.py:903–927` via `apply_field_mobility(mu0, |E|, v_sat, beta, gamma_pf)` (defined in `perovskite_sim/physics/field_mobility.py`). The primitive composes Poole-Frenkel field-enhancement with Caughey-Thomas velocity-saturation; `v_sat=0` and `gamma_pf=0` are degenerate sentinels that return `mu0` unchanged.

In 2D the field has two components, `E_x` and `E_y`, at each interior point. **Stage B(c.2) uses face-normal μ(E):** x-faces use only `|E_x_face|`, y-faces use only `|E_y_face|`. This is bit-equivalent to 1D under lateral-uniform devices (where `E_x ≈ 0`), matches the SG flux decomposition, and avoids cross-axis interpolation. **Option B (total-|E|) is explicitly deferred** per the design spec at `docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-mobility-design.md`.

**Sign and Einstein-relation rules (do NOT change):**
```
E_x_face = -(phi[:,  1:] - phi[:, :-1]) / dx[None, :]   shape (Ny,   Nx-1)
E_y_face = -(phi[1:, :] - phi[:-1, :]) / dy[:,  None]   shape (Ny-1, Nx)
mu_n_x_face_base = harmonic_mean(D_n along x) / V_T     shape (Ny,   Nx-1)
mu_n_y_face_base = harmonic_mean(D_n along y) / V_T     shape (Ny-1, Nx)
mu_n_x_face_eff  = apply_field_mobility(mu_n_x_face_base, |E_x_face|, ...)
D_n_x_face_eff   = mu_n_x_face_eff * V_T                ← Einstein roundtrip
```

**Activation gate (mirror 1D `mol.py:502–509`):**
```
has_field_mobility = sim_mode.use_field_dependent_mobility AND
                     any(v_sat_n > 0 or v_sat_p > 0 or pf_gamma_n > 0 or pf_gamma_p > 0)
```

**Mean choice for face averaging:**
- `D_n` / `D_p`: **harmonic** mean (matches existing `sg_fluxes_2d_*` and 1D convention).
- `v_sat`, `ct_beta`, `pf_gamma`: **arithmetic** mean (avoids harmonic suppression at one-side-zero, which would silently disable CT/PF at heterointerfaces). All three remain non-negative because the arithmetic mean of two non-negatives is non-negative.

**Run all tests from the `perovskite-sim/` directory:**
```bash
cd perovskite-sim
pytest tests/unit/twod/                                  # fast unit tests
pytest -m slow                                           # slow regression incl. parity gates
pytest tests/regression/test_twod_validation.py -v -s -m slow
```

The fast suite excludes `-m slow` by default (per `pyproject.toml`). Slow tests must be invoked explicitly.

**Constraints (user-imposed):**
1. Disabled-path bit-identical to current Stage B(c.1) results.
2. No YAML schema changes.
3. No changes to `perovskite_sim/physics/field_mobility.py`.
4. Use the existing 1D `apply_field_mobility` primitive (no new physics formulas).
5. Option B (total-|E|) explicitly deferred.
6. Strict shape checks on all face arrays — fail early with clear errors, no silent broadcasting.
7. Periodic-wrap test required.
8. At least one test that fails if `D_eff = μ_eff * V_T` is omitted or applied twice.

---

## Task 1: Pure field-mobility face-array builders + Einstein-roundtrip helper

**Files:**
- Create: `perovskite_sim/twod/field_mobility_2d.py`
- Create: `tests/unit/twod/test_field_mobility_2d.py`

This task introduces a new module containing pure helper functions:
1. Three arithmetic-mean face-array builders (used at build time for `v_sat` / `ct_beta` / `pf_gamma`).
2. The per-RHS `recompute_d_eff_2d` helper (used at runtime to compute `D_eff` per face from `phi`, `D_n`, `D_p`, `V_T`, and the face-mobility params).
3. Shape validators that fail early with clear messages.

Tests verify shapes, identity at uniform input, and the **Einstein roundtrip** — when all `v_sat` / `pf_gamma` face arrays are zero, `D_eff` equals harmonic-mean(D) (catches the case where `* V_T` is omitted or applied twice).

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/twod/test_field_mobility_2d.py`:

```python
from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.field_mobility_2d import (
    arith_mean_face_x, arith_mean_face_y, arith_mean_face_wrap,
    recompute_d_eff_2d, FieldMobilityDEff,
)


def test_arith_mean_face_x_shape_and_value():
    A = np.array([[1.0, 3.0, 5.0],
                  [2.0, 4.0, 6.0]])  # (2, 3)
    out = arith_mean_face_x(A)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[2.0, 4.0], [3.0, 5.0]]))


def test_arith_mean_face_y_shape_and_value():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # (3, 2)
    out = arith_mean_face_y(A)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[2.0, 3.0], [4.0, 5.0]]))


def test_arith_mean_face_wrap_shape_and_value():
    A = np.array([[1.0, 2.0, 3.0],
                  [10.0, 20.0, 30.0]])  # (2, 3); wrap face avg col 0 and col -1
    out = arith_mean_face_wrap(A)
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([2.0, 20.0]))


def test_arith_mean_face_x_uniform_identity():
    A = np.full((5, 4), 7.0)
    out = arith_mean_face_x(A)
    assert out.shape == (5, 3)
    np.testing.assert_array_equal(out, np.full((5, 3), 7.0))


def test_recompute_d_eff_einstein_roundtrip_zero_field_mobility_neumann():
    """When all field-mobility face params are zero (CT off, PF off), the recompute
    must return D_eff equal to harmonic-mean(D_node) on every face. This catches
    the bug where * V_T is omitted (D_eff would be off by 1/V_T) or applied twice
    (D_eff would be off by V_T)."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_const = 1.5e-3
    D_p_const = 7.0e-4
    D_n_node = np.full((Ny, Nx), D_n_const)
    D_p_node = np.full((Ny, Nx), D_p_const)
    V_T = 0.025852
    # Non-trivial phi to give non-zero E
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    # Zero field-mobility face arrays
    zero_x = np.zeros((Ny, Nx - 1))
    zero_y = np.zeros((Ny - 1, Nx))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=zero_x, v_sat_n_y_face=zero_y,
        ct_beta_n_x_face=zero_x, ct_beta_n_y_face=zero_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=zero_x, v_sat_p_y_face=zero_y,
        ct_beta_p_x_face=zero_x, ct_beta_p_y_face=zero_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    assert res.D_n_x.shape == (Ny, Nx - 1)
    assert res.D_n_y.shape == (Ny - 1, Nx)
    assert res.D_p_x.shape == (Ny, Nx - 1)
    assert res.D_p_y.shape == (Ny - 1, Nx)
    assert res.D_n_wrap is None and res.D_p_wrap is None
    # Harmonic mean of two equal values equals the value itself.
    np.testing.assert_allclose(res.D_n_x, D_n_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_n_y, D_n_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_x, D_p_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_y, D_p_const, rtol=1e-15)


def test_recompute_d_eff_einstein_roundtrip_zero_field_mobility_periodic():
    """Same Einstein roundtrip but with lateral_bc='periodic' — the wrap face must
    also be returned and equal to harmonic-mean of D[:, -1] and D[:, 0]."""
    Ny, Nx = 4, 5
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    zero_x = np.zeros((Ny, Nx - 1))
    zero_y = np.zeros((Ny - 1, Nx))
    zero_wrap = np.zeros((Ny,))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=zero_x, v_sat_n_y_face=zero_y,
        ct_beta_n_x_face=zero_x, ct_beta_n_y_face=zero_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=zero_x, v_sat_p_y_face=zero_y,
        ct_beta_p_x_face=zero_x, ct_beta_p_y_face=zero_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="periodic",
        v_sat_n_wrap=zero_wrap, v_sat_p_wrap=zero_wrap,
        ct_beta_n_wrap=zero_wrap, ct_beta_p_wrap=zero_wrap,
        pf_gamma_n_wrap=zero_wrap, pf_gamma_p_wrap=zero_wrap,
    )
    assert res.D_n_wrap is not None
    assert res.D_p_wrap is not None
    assert res.D_n_wrap.shape == (Ny,)
    assert res.D_p_wrap.shape == (Ny,)
    np.testing.assert_allclose(res.D_n_wrap, 1.5e-3, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_wrap, 7.0e-4, rtol=1e-15)


def test_recompute_d_eff_aggressive_ct_reduces_mobility():
    """At aggressive v_sat=1e2 m/s and a non-trivial E, D_eff must be strictly
    less than harmonic-mean(D_node) on every face — i.e., CT actually fires."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))  # non-zero E_y
    # Aggressive blocking
    v_sat_x = np.full((Ny, Nx - 1), 1e2)
    v_sat_y = np.full((Ny - 1, Nx), 1e2)
    beta_x  = np.full((Ny, Nx - 1), 2.0)
    beta_y  = np.full((Ny - 1, Nx), 2.0)
    zero_x  = np.zeros((Ny, Nx - 1))
    zero_y  = np.zeros((Ny - 1, Nx))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=v_sat_x, v_sat_n_y_face=v_sat_y,
        ct_beta_n_x_face=beta_x, ct_beta_n_y_face=beta_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=v_sat_x, v_sat_p_y_face=v_sat_y,
        ct_beta_p_x_face=beta_x, ct_beta_p_y_face=beta_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    # On y-faces (where E_y > 0), D_eff_n must be strictly less than D_n_node.
    assert np.all(res.D_n_y < 1.5e-3)
    assert np.all(res.D_p_y < 7.0e-4)
    # On x-faces, E_x = 0 (phi varies only in y), so CT should be inactive and D_eff = D_node.
    np.testing.assert_allclose(res.D_n_x, 1.5e-3, rtol=1e-12)
    np.testing.assert_allclose(res.D_p_x, 7.0e-4, rtol=1e-12)


def test_recompute_d_eff_sign_invariance():
    """Flipping the sign of phi must not change D_eff (apply_field_mobility uses |E|)."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi_pos = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    phi_neg = -phi_pos
    v_sat_x = np.full((Ny, Nx - 1), 1e2)
    v_sat_y = np.full((Ny - 1, Nx), 1e2)
    beta_x  = np.full((Ny, Nx - 1), 2.0)
    beta_y  = np.full((Ny - 1, Nx), 2.0)
    zero_x  = np.zeros((Ny, Nx - 1))
    zero_y  = np.zeros((Ny - 1, Nx))
    common = dict(
        x=x, y=y, D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=v_sat_x, v_sat_n_y_face=v_sat_y,
        ct_beta_n_x_face=beta_x, ct_beta_n_y_face=beta_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=v_sat_x, v_sat_p_y_face=v_sat_y,
        ct_beta_p_x_face=beta_x, ct_beta_p_y_face=beta_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    res_pos = recompute_d_eff_2d(phi=phi_pos, **common)
    res_neg = recompute_d_eff_2d(phi=phi_neg, **common)
    np.testing.assert_array_equal(res_pos.D_n_y, res_neg.D_n_y)
    np.testing.assert_array_equal(res_pos.D_p_y, res_neg.D_p_y)


def test_recompute_d_eff_shape_mismatch_raises():
    """Wrong shape on any face param must raise a clear ValueError, not silently broadcast."""
    Ny, Nx = 4, 3
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.zeros((Ny, Nx))
    bad_x = np.zeros((Ny, Nx))      # wrong: should be (Ny, Nx-1)
    zero_y = np.zeros((Ny - 1, Nx))
    with pytest.raises(ValueError, match="v_sat_n_x_face"):
        recompute_d_eff_2d(
            phi=phi, x=x, y=y,
            D_n=D_n_node, D_p=D_p_node, V_T=V_T,
            v_sat_n_x_face=bad_x, v_sat_n_y_face=zero_y,
            ct_beta_n_x_face=np.zeros((Ny, Nx - 1)), ct_beta_n_y_face=zero_y,
            pf_gamma_n_x_face=np.zeros((Ny, Nx - 1)), pf_gamma_n_y_face=zero_y,
            v_sat_p_x_face=np.zeros((Ny, Nx - 1)), v_sat_p_y_face=zero_y,
            ct_beta_p_x_face=np.zeros((Ny, Nx - 1)), ct_beta_p_y_face=zero_y,
            pf_gamma_p_x_face=np.zeros((Ny, Nx - 1)), pf_gamma_p_y_face=zero_y,
            lateral_bc="neumann",
        )
```

- [ ] **Step 2: Run tests to verify they fail with module not found**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_field_mobility_2d.py -v
```

Expected: `ModuleNotFoundError: No module named 'perovskite_sim.twod.field_mobility_2d'`

- [ ] **Step 3: Create the new module**

Create `perovskite_sim/twod/field_mobility_2d.py`:

```python
"""Field-dependent mobility recompute for the 2D solver (Stage B(c.2)).

Stage B(c.2) uses **face-normal** μ(E):
  - x-faces use only |E_x_face|
  - y-faces use only |E_y_face|
  - μ_n / μ_p are recomputed per face from the existing 1D
    ``apply_field_mobility`` primitive
  - D_eff is recovered via the Einstein relation D = μ V_T

Option B (total-|E| with cross-axis interpolation) is explicitly deferred —
see docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-mobility-design.md.

Mean choices:
  - D_n / D_p face averaging: HARMONIC mean (matches sg_fluxes_2d_* and the
    1D MaterialArrays D_n_face convention).
  - v_sat / ct_beta / pf_gamma face averaging: ARITHMETIC mean (avoids
    harmonic suppression when one side is zero, which would silently disable
    CT/PF at heterointerfaces; the empirical primitives already short-circuit
    on zero individually).
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np

from perovskite_sim.physics.field_mobility import apply_field_mobility


# Tiny floor used in harmonic mean to avoid 0/0 when both sides are zero.
_EPS_HARMONIC = 1e-300


class FieldMobilityDEff(NamedTuple):
    """Effective per-face diffusion coefficients computed from μ(E)."""
    D_n_x: np.ndarray                  # (Ny, Nx-1)
    D_n_y: np.ndarray                  # (Ny-1, Nx)
    D_p_x: np.ndarray                  # (Ny, Nx-1)
    D_p_y: np.ndarray                  # (Ny-1, Nx)
    D_n_wrap: np.ndarray | None        # (Ny,) when periodic, else None
    D_p_wrap: np.ndarray | None        # (Ny,) when periodic, else None


# ---------------------------------------------------------------------------
# Pure arithmetic-mean face builders (used at build time for v_sat / beta / gamma_pf)
# ---------------------------------------------------------------------------


def arith_mean_face_x(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A along the x-axis to produce x-face values.

    A is (Ny, Nx) per-node; output is (Ny, Nx-1) on interior x-faces.
    """
    return 0.5 * (A[:, :-1] + A[:, 1:])


def arith_mean_face_y(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A along the y-axis to produce y-face values.

    A is (Ny, Nx) per-node; output is (Ny-1, Nx) on interior y-faces.
    """
    return 0.5 * (A[:-1, :] + A[1:, :])


def arith_mean_face_wrap(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A across the periodic-x wrap face (col 0 ↔ col -1).

    A is (Ny, Nx) per-node; output is (Ny,) on the wrap face.
    """
    return 0.5 * (A[:, -1] + A[:, 0])


# ---------------------------------------------------------------------------
# Internal harmonic-mean helpers for D (matches sg_fluxes_2d_* convention)
# ---------------------------------------------------------------------------


def _harmonic_face_x(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:, :-1] * D[:, 1:] / (D[:, :-1] + D[:, 1:] + _EPS_HARMONIC)


def _harmonic_face_y(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:-1, :] * D[1:, :] / (D[:-1, :] + D[1:, :] + _EPS_HARMONIC)


def _harmonic_face_wrap(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:, -1] * D[:, 0] / (D[:, -1] + D[:, 0] + _EPS_HARMONIC)


# ---------------------------------------------------------------------------
# Shape validators
# ---------------------------------------------------------------------------


def _check_x_face(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny, Nx - 1):
        raise ValueError(
            f"Stage B(c.2) face-array shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny}, {Nx - 1})."
        )


def _check_y_face(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny - 1, Nx):
        raise ValueError(
            f"Stage B(c.2) face-array shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny - 1}, {Nx})."
        )


def _check_wrap(name: str, A: np.ndarray, Ny: int) -> None:
    if A.shape != (Ny,):
        raise ValueError(
            f"Stage B(c.2) wrap-face shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny},)."
        )


# ---------------------------------------------------------------------------
# The per-RHS recompute helper
# ---------------------------------------------------------------------------


def recompute_d_eff_2d(
    *,
    phi: np.ndarray,                                # (Ny, Nx)
    x: np.ndarray,                                  # (Nx,)
    y: np.ndarray,                                  # (Ny,)
    D_n: np.ndarray,                                # (Ny, Nx) per-node
    D_p: np.ndarray,                                # (Ny, Nx) per-node
    V_T: float,
    v_sat_n_x_face: np.ndarray,                     # (Ny, Nx-1)
    v_sat_n_y_face: np.ndarray,                     # (Ny-1, Nx)
    ct_beta_n_x_face: np.ndarray,
    ct_beta_n_y_face: np.ndarray,
    pf_gamma_n_x_face: np.ndarray,
    pf_gamma_n_y_face: np.ndarray,
    v_sat_p_x_face: np.ndarray,
    v_sat_p_y_face: np.ndarray,
    ct_beta_p_x_face: np.ndarray,
    ct_beta_p_y_face: np.ndarray,
    pf_gamma_p_x_face: np.ndarray,
    pf_gamma_p_y_face: np.ndarray,
    lateral_bc: str = "neumann",
    v_sat_n_wrap: np.ndarray | None = None,
    v_sat_p_wrap: np.ndarray | None = None,
    ct_beta_n_wrap: np.ndarray | None = None,
    ct_beta_p_wrap: np.ndarray | None = None,
    pf_gamma_n_wrap: np.ndarray | None = None,
    pf_gamma_p_wrap: np.ndarray | None = None,
) -> FieldMobilityDEff:
    """Recompute per-face D_eff from φ using face-normal μ(E).

    Stage B(c.2) face-normal convention:
        x-faces use |E_x_face|; y-faces use |E_y_face|.
    No cross-axis interpolation. apply_field_mobility takes np.abs(E)
    internally, so the input sign is irrelevant.

    Einstein roundtrip: when all v_sat / pf_gamma face arrays are zero,
    D_eff equals harmonic-mean(D_node) on every face — i.e.
    (D / V_T) * V_T = D exactly.
    """
    Ny, Nx = D_n.shape
    if D_p.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.2): D_n shape {D_n.shape} != D_p shape {D_p.shape}"
        )
    if phi.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.2): phi shape {phi.shape} != ({Ny}, {Nx})"
        )

    # Shape checks for interior face params (fail early, no silent broadcast).
    _check_x_face("v_sat_n_x_face",    v_sat_n_x_face,    Ny, Nx)
    _check_y_face("v_sat_n_y_face",    v_sat_n_y_face,    Ny, Nx)
    _check_x_face("ct_beta_n_x_face",  ct_beta_n_x_face,  Ny, Nx)
    _check_y_face("ct_beta_n_y_face",  ct_beta_n_y_face,  Ny, Nx)
    _check_x_face("pf_gamma_n_x_face", pf_gamma_n_x_face, Ny, Nx)
    _check_y_face("pf_gamma_n_y_face", pf_gamma_n_y_face, Ny, Nx)
    _check_x_face("v_sat_p_x_face",    v_sat_p_x_face,    Ny, Nx)
    _check_y_face("v_sat_p_y_face",    v_sat_p_y_face,    Ny, Nx)
    _check_x_face("ct_beta_p_x_face",  ct_beta_p_x_face,  Ny, Nx)
    _check_y_face("ct_beta_p_y_face",  ct_beta_p_y_face,  Ny, Nx)
    _check_x_face("pf_gamma_p_x_face", pf_gamma_p_x_face, Ny, Nx)
    _check_y_face("pf_gamma_p_y_face", pf_gamma_p_y_face, Ny, Nx)

    dx = np.diff(x)
    dy = np.diff(y)

    # Face fields. Sign cancels inside apply_field_mobility (np.abs).
    E_x_face = -(phi[:,  1:] - phi[:, :-1]) / dx[None, :]   # (Ny,   Nx-1)
    E_y_face = -(phi[1:, :] - phi[:-1, :]) / dy[:,  None]   # (Ny-1, Nx)

    # Base mobility at faces via Einstein on harmonic-mean D.
    mu_n_x_base = _harmonic_face_x(D_n) / V_T
    mu_n_y_base = _harmonic_face_y(D_n) / V_T
    mu_p_x_base = _harmonic_face_x(D_p) / V_T
    mu_p_y_base = _harmonic_face_y(D_p) / V_T

    # Apply field mobility (face-normal: x-face uses |E_x|, y-face uses |E_y|).
    mu_n_x_eff = apply_field_mobility(
        mu_n_x_base, np.abs(E_x_face),
        v_sat_n_x_face, ct_beta_n_x_face, pf_gamma_n_x_face,
    )
    mu_n_y_eff = apply_field_mobility(
        mu_n_y_base, np.abs(E_y_face),
        v_sat_n_y_face, ct_beta_n_y_face, pf_gamma_n_y_face,
    )
    mu_p_x_eff = apply_field_mobility(
        mu_p_x_base, np.abs(E_x_face),
        v_sat_p_x_face, ct_beta_p_x_face, pf_gamma_p_x_face,
    )
    mu_p_y_eff = apply_field_mobility(
        mu_p_y_base, np.abs(E_y_face),
        v_sat_p_y_face, ct_beta_p_y_face, pf_gamma_p_y_face,
    )

    # Recover D via Einstein relation. Use the SAME V_T as the divide above
    # (single source of truth) so the v_sat=pf_gamma=0 path is exact.
    D_n_x_eff = mu_n_x_eff * V_T
    D_n_y_eff = mu_n_y_eff * V_T
    D_p_x_eff = mu_p_x_eff * V_T
    D_p_y_eff = mu_p_y_eff * V_T

    # Periodic wrap face (only when lateral_bc == "periodic").
    D_n_wrap_eff: np.ndarray | None = None
    D_p_wrap_eff: np.ndarray | None = None
    if lateral_bc == "periodic":
        if (v_sat_n_wrap is None or v_sat_p_wrap is None
                or ct_beta_n_wrap is None or ct_beta_p_wrap is None
                or pf_gamma_n_wrap is None or pf_gamma_p_wrap is None):
            raise ValueError(
                "Stage B(c.2): lateral_bc='periodic' requires all six "
                "*_wrap face arrays to be provided."
            )
        _check_wrap("v_sat_n_wrap",    v_sat_n_wrap,    Ny)
        _check_wrap("v_sat_p_wrap",    v_sat_p_wrap,    Ny)
        _check_wrap("ct_beta_n_wrap",  ct_beta_n_wrap,  Ny)
        _check_wrap("ct_beta_p_wrap",  ct_beta_p_wrap,  Ny)
        _check_wrap("pf_gamma_n_wrap", pf_gamma_n_wrap, Ny)
        _check_wrap("pf_gamma_p_wrap", pf_gamma_p_wrap, Ny)
        dx_wrap = 0.5 * (dx[0] + dx[-1])
        E_x_wrap = -(phi[:, 0] - phi[:, -1]) / dx_wrap     # (Ny,)
        mu_n_wrap_base = _harmonic_face_wrap(D_n) / V_T
        mu_p_wrap_base = _harmonic_face_wrap(D_p) / V_T
        mu_n_wrap_eff = apply_field_mobility(
            mu_n_wrap_base, np.abs(E_x_wrap),
            v_sat_n_wrap, ct_beta_n_wrap, pf_gamma_n_wrap,
        )
        mu_p_wrap_eff = apply_field_mobility(
            mu_p_wrap_base, np.abs(E_x_wrap),
            v_sat_p_wrap, ct_beta_p_wrap, pf_gamma_p_wrap,
        )
        D_n_wrap_eff = mu_n_wrap_eff * V_T
        D_p_wrap_eff = mu_p_wrap_eff * V_T

    return FieldMobilityDEff(
        D_n_x=D_n_x_eff,
        D_n_y=D_n_y_eff,
        D_p_x=D_p_x_eff,
        D_p_y=D_p_y_eff,
        D_n_wrap=D_n_wrap_eff,
        D_p_wrap=D_p_wrap_eff,
    )


__all__ = [
    "arith_mean_face_x", "arith_mean_face_y", "arith_mean_face_wrap",
    "recompute_d_eff_2d", "FieldMobilityDEff",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_field_mobility_2d.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/twod/field_mobility_2d.py tests/unit/twod/test_field_mobility_2d.py
git commit -m "feat(twod): add field_mobility_2d module — pure builders + recompute helper"
```

---

## Task 2: Extend `sg_fluxes_2d_n/p` with optional per-face D-override kwargs

**Files:**
- Modify: `perovskite_sim/twod/flux_2d.py`
- Create: `tests/unit/twod/test_flux_2d.py`

This task adds keyword-only override arguments to both flux helpers. When the override is `None` the existing harmonic-mean code path runs unchanged → bit-identical to current Stage B(c.1). When provided, the helper uses the override directly, bypassing the internal harmonic mean.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/twod/test_flux_2d.py`:

```python
from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p


def _setup():
    Ny, Nx = 6, 5
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    phi = np.linspace(0.0, 0.5, Ny)[:, None] * np.ones((Ny, Nx))
    n = np.full((Ny, Nx), 1e16)
    p = np.full((Ny, Nx), 1e16)
    D_n = np.full((Ny, Nx), 1.5e-3)
    D_p = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    return x, y, phi, n, p, D_n, D_p, V_T, Ny, Nx


def test_sg_fluxes_2d_n_no_override_unchanged():
    """Calling without overrides reproduces the existing constant-D output."""
    x, y, phi, n, p, D_n, _, V_T, _, _ = _setup()
    Jx, Jy = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_alt, Jy_alt = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=None, D_n_y_face=None,
    )
    np.testing.assert_array_equal(Jx, Jx_alt)
    np.testing.assert_array_equal(Jy, Jy_alt)


def test_sg_fluxes_2d_p_no_override_unchanged():
    x, y, phi, n, p, _, D_p, V_T, _, _ = _setup()
    Jx, Jy = sg_fluxes_2d_p(phi, p, x, y, D_p, V_T)
    Jx_alt, Jy_alt = sg_fluxes_2d_p(
        phi, p, x, y, D_p, V_T,
        D_p_x_face=None, D_p_y_face=None,
    )
    np.testing.assert_array_equal(Jx, Jx_alt)
    np.testing.assert_array_equal(Jy, Jy_alt)


def test_sg_fluxes_2d_n_override_with_harmonic_mean_matches_no_override():
    """Passing the explicit harmonic-mean of D_n as an override must produce
    the same output as letting the function compute it internally."""
    x, y, phi, n, p, D_n, _, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_n_x_harm = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)
    D_n_y_harm = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)
    Jx_internal, Jy_internal = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_override, Jy_override = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=D_n_x_harm, D_n_y_face=D_n_y_harm,
    )
    np.testing.assert_allclose(Jx_internal, Jx_override, rtol=1e-15)
    np.testing.assert_allclose(Jy_internal, Jy_override, rtol=1e-15)


def test_sg_fluxes_2d_n_override_actually_overrides():
    """Passing 2x harmonic-mean D as override must produce 2x the J flux."""
    x, y, phi, n, p, D_n, _, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_n_x_harm = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)
    D_n_y_harm = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)
    Jx_base, Jy_base = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_2x, Jy_2x = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=2.0 * D_n_x_harm, D_n_y_face=2.0 * D_n_y_harm,
    )
    np.testing.assert_allclose(Jx_2x, 2.0 * Jx_base, rtol=1e-13)
    np.testing.assert_allclose(Jy_2x, 2.0 * Jy_base, rtol=1e-13)


def test_sg_fluxes_2d_p_override_actually_overrides():
    x, y, phi, n, p, _, D_p, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_p_x_harm = 2.0 * D_p[:, :-1] * D_p[:, 1:] / (D_p[:, :-1] + D_p[:, 1:] + _eps)
    D_p_y_harm = 2.0 * D_p[:-1, :] * D_p[1:, :] / (D_p[:-1, :] + D_p[1:, :] + _eps)
    Jx_base, Jy_base = sg_fluxes_2d_p(phi, p, x, y, D_p, V_T)
    Jx_2x, Jy_2x = sg_fluxes_2d_p(
        phi, p, x, y, D_p, V_T,
        D_p_x_face=2.0 * D_p_x_harm, D_p_y_face=2.0 * D_p_y_harm,
    )
    np.testing.assert_allclose(Jx_2x, 2.0 * Jx_base, rtol=1e-13)
    np.testing.assert_allclose(Jy_2x, 2.0 * Jy_base, rtol=1e-13)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_flux_2d.py -v
```

Expected: tests fail with `TypeError: sg_fluxes_2d_n() got an unexpected keyword argument 'D_n_x_face'`.

- [ ] **Step 3: Modify `sg_fluxes_2d_n`**

Replace the body of `sg_fluxes_2d_n` in `perovskite_sim/twod/flux_2d.py`:

```python
def sg_fluxes_2d_n(
    phi_n: np.ndarray,        # (Ny, Nx)
    n: np.ndarray,            # (Ny, Nx)
    x: np.ndarray,            # (Nx,)
    y: np.ndarray,            # (Ny,)
    D_n: np.ndarray,          # (Ny, Nx) — node-resolved; face avg taken inside
    V_T: float,
    *,
    D_n_x_face: np.ndarray | None = None,   # (Ny, Nx-1)  Stage B(c.2) override
    D_n_y_face: np.ndarray | None = None,   # (Ny-1, Nx)  Stage B(c.2) override
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG electron current density on horizontal edges (J_x, shape
    (Ny, Nx-1)) and vertical edges (J_y, shape (Ny-1, Nx)).

    Conventions (matching 1D fe_operators.sg_fluxes_n):
      J_x[j, i]  : flux from (i,j) to (i+1,j); positive = electron current
                   flowing in +x direction.
      J_y[j, i]  : flux from (i,j) to (i,j+1); positive = +y direction.

    Per-face D override (Stage B(c.2)): when ``D_n_x_face`` / ``D_n_y_face``
    are provided, they replace the internal harmonic-mean computation. This
    is how field-dependent μ(E) injects per-face effective diffusion. When
    both are ``None`` (default) the harmonic-mean path runs unchanged.
    """
    dx = np.diff(x)                              # (Nx-1,)
    dy = np.diff(y)                              # (Ny-1,)

    # Harmonic-mean face averaging (matches 1D physics/poisson harmonic eps_r
    # face average and 1D MaterialArrays harmonic D_n_face). Required at
    # heterointerfaces where D varies by orders of magnitude across a face.
    _eps = 1e-300
    if D_n_x_face is None:
        D_face_x = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)  # (Ny, Nx-1)
    else:
        D_face_x = D_n_x_face
    xi_x = (phi_n[:, 1:] - phi_n[:, :-1]) / V_T  # (Ny, Nx-1)
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * n[:, 1:] - bernoulli(-xi_x) * n[:, :-1]
    )

    if D_n_y_face is None:
        D_face_y = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)  # (Ny-1, Nx)
    else:
        D_face_y = D_n_y_face
    xi_y = (phi_n[1:, :] - phi_n[:-1, :]) / V_T  # (Ny-1, Nx)
    Jy = (Q * D_face_y / dy[:, None]) * (
        bernoulli(xi_y) * n[1:, :] - bernoulli(-xi_y) * n[:-1, :]
    )
    return Jx, Jy
```

- [ ] **Step 4: Modify `sg_fluxes_2d_p`**

Replace the body of `sg_fluxes_2d_p`:

```python
def sg_fluxes_2d_p(
    phi_p: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    D_p: np.ndarray,
    V_T: float,
    *,
    D_p_x_face: np.ndarray | None = None,   # (Ny, Nx-1)  Stage B(c.2) override
    D_p_y_face: np.ndarray | None = None,   # (Ny-1, Nx)  Stage B(c.2) override
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised SG hole current density on horizontal and vertical edges.
    Sign convention matches the 1D `sg_fluxes_p`.

    Per-face D override (Stage B(c.2)): see ``sg_fluxes_2d_n`` for semantics.
    """
    dx = np.diff(x)
    dy = np.diff(y)

    _eps = 1e-300
    if D_p_x_face is None:
        D_face_x = 2.0 * D_p[:, :-1] * D_p[:, 1:] / (D_p[:, :-1] + D_p[:, 1:] + _eps)
    else:
        D_face_x = D_p_x_face
    xi_x = (phi_p[:, 1:] - phi_p[:, :-1]) / V_T
    Jx = (Q * D_face_x / dx[None, :]) * (
        bernoulli(xi_x) * p[:, :-1] - bernoulli(-xi_x) * p[:, 1:]
    )

    if D_p_y_face is None:
        D_face_y = 2.0 * D_p[:-1, :] * D_p[1:, :] / (D_p[:-1, :] + D_p[1:, :] + _eps)
    else:
        D_face_y = D_p_y_face
    xi_y = (phi_p[1:, :] - phi_p[:-1, :]) / V_T
    Jy = (Q * D_face_y / dy[:, None]) * (
        bernoulli(xi_y) * p[:-1, :] - bernoulli(-xi_y) * p[1:, :]
    )
    return Jx, Jy
```

- [ ] **Step 5: Run unit tests + verify existing 2D tests still pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/ -v
```

Expected: all 5 new flux tests pass + all existing 2D unit tests still pass (no regressions in `sg_fluxes_2d_*` callers).

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/twod/flux_2d.py tests/unit/twod/test_flux_2d.py
git commit -m "feat(twod): sg_fluxes_2d_n/p accept optional per-face D overrides"
```

---

## Task 3: Wire `MaterialArrays2D` field-mobility fields and `build_material_arrays_2d`

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Modify: `tests/unit/twod/test_solver_2d.py`

Adds 19 fields to the `MaterialArrays2D` dataclass (1 flag + 18 face arrays), populates them in `build_material_arrays_2d` from the existing `MaterialParams.v_sat_*` / `ct_beta_*` / `pf_gamma_*` fields via arithmetic-mean face builders. Tier-gated on `sim_mode.use_field_dependent_mobility`. Periodic-x wrap arrays only set when `lateral_bc == "periodic"`.

- [ ] **Step 1: Write the failing wiring tests**

Add to `tests/unit/twod/test_solver_2d.py` (the existing helpers `_stack`, `_layers_for_stack`, `_make_grid_and_mat` and the `dc_replace` import are already present):

```python
def test_material_arrays_2d_default_no_field_mobility():
    """Default preset → has_field_mobility=False and all 18 face fields None."""
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_field_mobility is False
    for name in (
        "v_sat_n_x_face", "v_sat_n_y_face", "v_sat_p_x_face", "v_sat_p_y_face",
        "ct_beta_n_x_face", "ct_beta_n_y_face", "ct_beta_p_x_face", "ct_beta_p_y_face",
        "pf_gamma_n_x_face", "pf_gamma_n_y_face", "pf_gamma_p_x_face", "pf_gamma_p_y_face",
        "v_sat_n_wrap", "v_sat_p_wrap",
        "ct_beta_n_wrap", "ct_beta_p_wrap",
        "pf_gamma_n_wrap", "pf_gamma_p_wrap",
    ):
        assert getattr(mat, name) is None, f"{name} should be None when field-mobility is off"


def test_material_arrays_2d_v_sat_activates_flag_and_shapes_neumann():
    """v_sat>0 with mode='full' → has_field_mobility=True and all interior face
    arrays have correct shapes; wrap arrays remain None for non-periodic BC."""
    stack = dc_replace(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="neumann")
    assert mat.has_field_mobility is True
    assert mat.v_sat_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.v_sat_n_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.v_sat_p_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.v_sat_p_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.ct_beta_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.ct_beta_p_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.pf_gamma_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.pf_gamma_p_y_face.shape == (g.Ny - 1, g.Nx)
    # Wrap arrays not populated for Neumann BC
    assert mat.v_sat_n_wrap is None
    assert mat.v_sat_p_wrap is None


def test_material_arrays_2d_periodic_populates_wrap_arrays():
    """Periodic BC with v_sat>0 → all six wrap arrays populated with shape (Ny,)."""
    stack = dc_replace(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_field_mobility is True
    for name in (
        "v_sat_n_wrap", "v_sat_p_wrap",
        "ct_beta_n_wrap", "ct_beta_p_wrap",
        "pf_gamma_n_wrap", "pf_gamma_p_wrap",
    ):
        arr = getattr(mat, name)
        assert arr is not None, f"{name} must be populated under periodic BC"
        assert arr.shape == (g.Ny,)


def test_material_arrays_2d_field_mobility_values_match_layer_params():
    """Layer v_sat_n=1e2 → mat.v_sat_n_y_face equals 1e2 inside that layer (arithmetic mean
    of two equal nodes is the node value)."""
    stack = dc_replace(_stack(), v_sat_n=1e2, v_sat_p=2e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="neumann")
    # Every layer has the same v_sat → arithmetic mean across any face equals v_sat.
    np.testing.assert_allclose(mat.v_sat_n_y_face, 1e2)
    np.testing.assert_allclose(mat.v_sat_p_y_face, 2e2)
    np.testing.assert_allclose(mat.v_sat_n_x_face, 1e2)
    np.testing.assert_allclose(mat.v_sat_p_x_face, 2e2)


def test_legacy_mode_disables_field_mobility_in_2d():
    """Tier-as-ceiling: device.mode='legacy' must keep has_field_mobility=False
    even when v_sat is set on the stack. Mirrors the B(c.1) Robin tier-gate test
    that was added during Issue I1 fix."""
    base = _stack()
    stack_legacy = dc_replace(base, mode="legacy", v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack_legacy)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_legacy, Microstructure())
    assert mat.has_field_mobility is False
    # Sanity: same params with mode='full' enables.
    stack_full = dc_replace(stack_legacy, mode="full")
    mat_full = build_material_arrays_2d(g, stack_full, Microstructure())
    assert mat_full.has_field_mobility is True


def test_pf_gamma_alone_activates_flag():
    """Setting only pf_gamma (with v_sat=0) is enough to trip the activation gate."""
    stack = dc_replace(_stack(), pf_gamma_n=3e-4, pf_gamma_p=3e-4)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_field_mobility is True
    assert mat.pf_gamma_n_x_face is not None
    np.testing.assert_allclose(mat.pf_gamma_n_x_face, 3e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_default_no_field_mobility \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_v_sat_activates_flag_and_shapes_neumann \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_periodic_populates_wrap_arrays \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_field_mobility_values_match_layer_params \
       tests/unit/twod/test_solver_2d.py::test_legacy_mode_disables_field_mobility_in_2d \
       tests/unit/twod/test_solver_2d.py::test_pf_gamma_alone_activates_flag -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'has_field_mobility'`.

- [ ] **Step 3: Add 19 fields to `MaterialArrays2D` in `solver_2d.py`**

Locate the existing Stage B(c.1) Robin block (the `S_n_top` / `S_p_top` / `S_n_bot` / `S_p_bot` fields). Append after them:

```python
    # --- Stage B(c.2): Field-dependent mobility μ(E) ----------------------------
    # Face-normal formulation: x-faces use only |E_x_face|, y-faces use only
    # |E_y_face|.  See docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-
    # mobility-design.md.  All eighteen face arrays are None by default; they
    # are populated by build_material_arrays_2d when sim_mode.
    # use_field_dependent_mobility AND any layer sets v_sat>0 or pf_gamma>0.
    # The disabled path (has_field_mobility=False) is bit-identical to the
    # current Stage B(c.1) constant-mobility code path.
    has_field_mobility: bool = False
    # x-face arrays — (Ny, Nx-1)
    v_sat_n_x_face:    np.ndarray | None = None
    v_sat_p_x_face:    np.ndarray | None = None
    ct_beta_n_x_face:  np.ndarray | None = None
    ct_beta_p_x_face:  np.ndarray | None = None
    pf_gamma_n_x_face: np.ndarray | None = None
    pf_gamma_p_x_face: np.ndarray | None = None
    # y-face arrays — (Ny-1, Nx)
    v_sat_n_y_face:    np.ndarray | None = None
    v_sat_p_y_face:    np.ndarray | None = None
    ct_beta_n_y_face:  np.ndarray | None = None
    ct_beta_p_y_face:  np.ndarray | None = None
    pf_gamma_n_y_face: np.ndarray | None = None
    pf_gamma_p_y_face: np.ndarray | None = None
    # Periodic-x wrap face arrays — (Ny,); None unless lateral_bc=="periodic"
    v_sat_n_wrap:    np.ndarray | None = None
    v_sat_p_wrap:    np.ndarray | None = None
    ct_beta_n_wrap:  np.ndarray | None = None
    ct_beta_p_wrap:  np.ndarray | None = None
    pf_gamma_n_wrap: np.ndarray | None = None
    pf_gamma_p_wrap: np.ndarray | None = None
```

- [ ] **Step 4: Add population logic in `build_material_arrays_2d`**

Add the following block in `build_material_arrays_2d` just before the `return MaterialArrays2D(...)` call (after the Stage B(c.1) Robin S-population block). First add the import at the top of `solver_2d.py` alongside the existing top-of-file imports:

```python
from perovskite_sim.twod.field_mobility_2d import (
    arith_mean_face_x, arith_mean_face_y, arith_mean_face_wrap,
)
```

Then in `build_material_arrays_2d`:

```python
    # --- Stage B(c.2): Field-dependent mobility μ(E) ----------------------------
    # Build per-node v_sat / ct_beta / pf_gamma arrays from layer params via the
    # same y-mask construction used for D_n / D_p above.
    Ny_b, Nx_b = grid.Ny, grid.Nx
    v_sat_n_node    = np.zeros((Ny_b, Nx_b))
    v_sat_p_node    = np.zeros((Ny_b, Nx_b))
    ct_beta_n_node  = np.zeros((Ny_b, Nx_b))
    ct_beta_p_node  = np.zeros((Ny_b, Nx_b))
    pf_gamma_n_node = np.zeros((Ny_b, Nx_b))
    pf_gamma_p_node = np.zeros((Ny_b, Nx_b))
    _offset_fm = 0.0
    for _layer_fm in electrical_layers(stack):
        _mask_fm = (grid.y >= _offset_fm - 1e-12) & (
            grid.y <= _offset_fm + _layer_fm.thickness + 1e-12
        )
        _p_fm = _layer_fm.params
        v_sat_n_node[_mask_fm, :]    = _p_fm.v_sat_n
        v_sat_p_node[_mask_fm, :]    = _p_fm.v_sat_p
        ct_beta_n_node[_mask_fm, :]  = _p_fm.ct_beta_n
        ct_beta_p_node[_mask_fm, :]  = _p_fm.ct_beta_p
        pf_gamma_n_node[_mask_fm, :] = _p_fm.pf_gamma_n
        pf_gamma_p_node[_mask_fm, :] = _p_fm.pf_gamma_p
        _offset_fm += _layer_fm.thickness

    # Tier-as-ceiling activation gate (mirror 1D mol.py:502–509 and B(c.1) Robin gate).
    from perovskite_sim.models.mode import resolve_mode as _resolve_mode_fm
    _sim_mode_fm = _resolve_mode_fm(getattr(stack, "mode", "full"))
    _has_field_mobility = bool(
        _sim_mode_fm.use_field_dependent_mobility
        and (
            np.any(v_sat_n_node    > 0.0) or np.any(v_sat_p_node    > 0.0)
            or np.any(pf_gamma_n_node > 0.0) or np.any(pf_gamma_p_node > 0.0)
        )
    )

    if _has_field_mobility:
        # Arithmetic mean to faces (NOT harmonic — see field_mobility_2d.py docstring).
        v_sat_n_x_face    = arith_mean_face_x(v_sat_n_node)
        v_sat_n_y_face    = arith_mean_face_y(v_sat_n_node)
        v_sat_p_x_face    = arith_mean_face_x(v_sat_p_node)
        v_sat_p_y_face    = arith_mean_face_y(v_sat_p_node)
        ct_beta_n_x_face  = arith_mean_face_x(ct_beta_n_node)
        ct_beta_n_y_face  = arith_mean_face_y(ct_beta_n_node)
        ct_beta_p_x_face  = arith_mean_face_x(ct_beta_p_node)
        ct_beta_p_y_face  = arith_mean_face_y(ct_beta_p_node)
        pf_gamma_n_x_face = arith_mean_face_x(pf_gamma_n_node)
        pf_gamma_n_y_face = arith_mean_face_y(pf_gamma_n_node)
        pf_gamma_p_x_face = arith_mean_face_x(pf_gamma_p_node)
        pf_gamma_p_y_face = arith_mean_face_y(pf_gamma_p_node)
        if lateral_bc == "periodic":
            v_sat_n_wrap    = arith_mean_face_wrap(v_sat_n_node)
            v_sat_p_wrap    = arith_mean_face_wrap(v_sat_p_node)
            ct_beta_n_wrap  = arith_mean_face_wrap(ct_beta_n_node)
            ct_beta_p_wrap  = arith_mean_face_wrap(ct_beta_p_node)
            pf_gamma_n_wrap = arith_mean_face_wrap(pf_gamma_n_node)
            pf_gamma_p_wrap = arith_mean_face_wrap(pf_gamma_p_node)
        else:
            v_sat_n_wrap = v_sat_p_wrap = None
            ct_beta_n_wrap = ct_beta_p_wrap = None
            pf_gamma_n_wrap = pf_gamma_p_wrap = None
    else:
        v_sat_n_x_face = v_sat_n_y_face = None
        v_sat_p_x_face = v_sat_p_y_face = None
        ct_beta_n_x_face = ct_beta_n_y_face = None
        ct_beta_p_x_face = ct_beta_p_y_face = None
        pf_gamma_n_x_face = pf_gamma_n_y_face = None
        pf_gamma_p_x_face = pf_gamma_p_y_face = None
        v_sat_n_wrap = v_sat_p_wrap = None
        ct_beta_n_wrap = ct_beta_p_wrap = None
        pf_gamma_n_wrap = pf_gamma_p_wrap = None
```

- [ ] **Step 5: Pass the new fields into `MaterialArrays2D(...)`**

In the `return MaterialArrays2D(...)` call, append after the existing Stage B(c.1) Robin fields:

```python
        has_field_mobility=_has_field_mobility,
        v_sat_n_x_face=v_sat_n_x_face, v_sat_p_x_face=v_sat_p_x_face,
        ct_beta_n_x_face=ct_beta_n_x_face, ct_beta_p_x_face=ct_beta_p_x_face,
        pf_gamma_n_x_face=pf_gamma_n_x_face, pf_gamma_p_x_face=pf_gamma_p_x_face,
        v_sat_n_y_face=v_sat_n_y_face, v_sat_p_y_face=v_sat_p_y_face,
        ct_beta_n_y_face=ct_beta_n_y_face, ct_beta_p_y_face=ct_beta_p_y_face,
        pf_gamma_n_y_face=pf_gamma_n_y_face, pf_gamma_p_y_face=pf_gamma_p_y_face,
        v_sat_n_wrap=v_sat_n_wrap, v_sat_p_wrap=v_sat_p_wrap,
        ct_beta_n_wrap=ct_beta_n_wrap, ct_beta_p_wrap=ct_beta_p_wrap,
        pf_gamma_n_wrap=pf_gamma_n_wrap, pf_gamma_p_wrap=pf_gamma_p_wrap,
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py -v
```

Expected: all 6 new tests pass + existing 2D unit tests still pass.

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): MaterialArrays2D field-mobility fields and build path"
```

---

## Task 4: Wire `assemble_rhs_2d` to recompute `D_eff` faces from `phi`

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Modify: `perovskite_sim/twod/continuity_2d.py`
- Modify: `tests/unit/twod/test_solver_2d.py`

This task adds the per-RHS μ(E) recompute. `assemble_rhs_2d` branches on `mat.has_field_mobility`: True path calls `recompute_d_eff_2d` and forwards the six override arrays through to `continuity_rhs_2d` → `sg_fluxes_2d_*`. False path is unchanged. `continuity_rhs_2d` gets new optional kwargs that forward through to `sg_fluxes_2d_*` and the periodic-wrap block.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def test_assemble_rhs_2d_field_mobility_disabled_path_unchanged():
    """When v_sat=pf_gamma=0 (default preset), mat.has_field_mobility is False
    and assemble_rhs_2d output is bit-identical to legacy-mode-with-vsat (which
    is also disabled via the tier gate)."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    base = _stack()
    stack_off    = base                                                       # mode=full, no v_sat
    stack_legacy = dc_replace(base, mode="legacy", v_sat_n=1e2, v_sat_p=1e2)  # tier-disabled
    layers = _layers_for_stack(base)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat_off    = build_material_arrays_2d(g, stack_off,    Microstructure(), lateral_bc="periodic")
    mat_legacy = build_material_arrays_2d(g, stack_legacy, Microstructure(), lateral_bc="periodic")
    assert mat_off.has_field_mobility is False
    assert mat_legacy.has_field_mobility is False
    n0 = float(mat_off.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_off.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt_off    = assemble_rhs_2d(0.0, y0, mat_off,    V_app=0.0)
    dydt_legacy = assemble_rhs_2d(0.0, y0, mat_legacy, V_app=0.0)
    np.testing.assert_array_equal(dydt_off, dydt_legacy)


def test_assemble_rhs_2d_field_mobility_enabled_changes_dydt():
    """When v_sat=1e2 with mode='full', assemble_rhs_2d output differs from
    the constant-mobility baseline at a state with non-zero E."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    base = _stack()
    stack_off = base
    stack_on  = dc_replace(base, v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(base)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat_off = build_material_arrays_2d(g, stack_off, Microstructure(), lateral_bc="periodic")
    mat_on  = build_material_arrays_2d(g, stack_on,  Microstructure(), lateral_bc="periodic")
    assert mat_off.has_field_mobility is False
    assert mat_on.has_field_mobility is True
    # Build a state with non-trivial gradients in y so E_y ≠ 0.
    n_grad = np.linspace(float(mat_off.n_eq_left[0]),
                        float(mat_off.n_eq_right[0]), g.Ny)
    p_grad = np.linspace(float(mat_off.p_eq_left[0]),
                        float(mat_off.p_eq_right[0]), g.Ny)
    n0 = np.broadcast_to(n_grad[:, None], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(p_grad[:, None], (g.Ny, g.Nx)).copy()
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt_off = assemble_rhs_2d(0.0, y0, mat_off, V_app=0.5)
    dydt_on  = assemble_rhs_2d(0.0, y0, mat_on,  V_app=0.5)
    # μ(E) actively perturbs the RHS at non-trivial fields.
    assert not np.array_equal(dydt_off, dydt_on)
    rel = np.max(np.abs(dydt_on - dydt_off)) / max(1.0, np.max(np.abs(dydt_off)))
    assert rel > 1e-6, f"μ(E) effect on RHS too small (rel diff {rel:.2e})"


def test_assemble_rhs_2d_field_mobility_finite_periodic():
    """μ(E) on with lateral_bc='periodic' produces a finite RHS at a non-trivial
    state. Catches a missing wrap-face override or a periodic-wrap shape bug."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = dc_replace(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=5, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_field_mobility is True
    assert mat.v_sat_n_wrap is not None
    # Build a state with broken lateral symmetry → non-zero E_x at wrap face
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    n0[:, 0] *= 1.5   # asymmetric — drives non-trivial wrap-face E_x
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    assert np.all(np.isfinite(dydt)), "μ(E) periodic wrap produced non-finite RHS"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd perovskite-sim
pytest tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_field_mobility_disabled_path_unchanged \
       tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_field_mobility_enabled_changes_dydt \
       tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_field_mobility_finite_periodic -v
```

Expected: the disabled-path test passes (because `has_field_mobility=False` already takes the existing path), but the "enabled changes RHS" test fails (because `assemble_rhs_2d` does not yet route to `recompute_d_eff_2d`).

- [ ] **Step 3: Add per-face D-override forwarding to `continuity_rhs_2d`**

Modify `perovskite_sim/twod/continuity_2d.py` — extend the function signature with six optional keyword-only kwargs and forward them to `sg_fluxes_2d_n/p` and the periodic-wrap block.

Update the signature:

```python
def continuity_rhs_2d(
    x: np.ndarray, y: np.ndarray,
    phi: np.ndarray, n: np.ndarray, p: np.ndarray,
    G: np.ndarray, R: np.ndarray,
    D_n: np.ndarray, D_p: np.ndarray,
    V_T: float,
    *,
    chi: np.ndarray | None = None,
    Eg: np.ndarray | None = None,
    lateral_bc: str = "periodic",
    interface_y_faces: tuple[int, ...] = (),
    A_star_n: np.ndarray | None = None,
    A_star_p: np.ndarray | None = None,
    T: float | None = None,
    # Stage B(c.2) field-mobility per-face D overrides:
    D_n_x_face: np.ndarray | None = None,
    D_n_y_face: np.ndarray | None = None,
    D_p_x_face: np.ndarray | None = None,
    D_p_y_face: np.ndarray | None = None,
    D_n_wrap: np.ndarray | None = None,
    D_p_wrap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

Forward to `sg_fluxes_2d_n/p`. Find the existing call sites (search for `sg_fluxes_2d_n(` and `sg_fluxes_2d_p(`) and replace with:

```python
    Jx_n, Jy_n = sg_fluxes_2d_n(
        phi_n, n, x, y, D_n, V_T,
        D_n_x_face=D_n_x_face, D_n_y_face=D_n_y_face,
    )   # Jx (Ny, Nx-1), Jy (Ny-1, Nx)
    Jx_p, Jy_p = sg_fluxes_2d_p(
        phi_p, p, x, y, D_p, V_T,
        D_p_x_face=D_p_x_face, D_p_y_face=D_p_y_face,
    )
```

Update the periodic-wrap face block (around line 161/175). Replace the lines:

```python
        D_face_wrap_n = 2.0 * D_n[:, -1] * D_n[:, 0] / (D_n[:, -1] + D_n[:, 0] + _eps_face)
```
and
```python
        D_face_wrap_p = 2.0 * D_p[:, -1] * D_p[:, 0] / (D_p[:, -1] + D_p[:, 0] + _eps_face)
```

with:

```python
        # Stage B(c.2): if D_n_wrap was provided (μ(E) on), use it directly;
        # else fall back to harmonic mean.
        if D_n_wrap is None:
            D_face_wrap_n = 2.0 * D_n[:, -1] * D_n[:, 0] / (D_n[:, -1] + D_n[:, 0] + _eps_face)
        else:
            D_face_wrap_n = D_n_wrap
```

and analogously for `D_face_wrap_p`.

- [ ] **Step 4: Add μ(E) recompute branch in `assemble_rhs_2d`**

Add the import at the top of `solver_2d.py` next to the other field-mobility imports:

```python
from perovskite_sim.twod.field_mobility_2d import (
    arith_mean_face_x, arith_mean_face_y, arith_mean_face_wrap,
    recompute_d_eff_2d,
)
```

(Consolidate this with the import added in Task 3 — same line group.)

In `assemble_rhs_2d`, find the existing call to `continuity_rhs_2d` and replace it with the branched call:

```python
    # --- Continuity --------------------------------------------------------
    if mat.has_field_mobility:
        # Stage B(c.2): face-normal μ(E) recompute. x-faces use |E_x_face|,
        # y-faces use |E_y_face|. apply_field_mobility takes np.abs(E).
        d_eff = recompute_d_eff_2d(
            phi=phi, x=g.x, y=g.y,
            D_n=mat.D_n, D_p=mat.D_p, V_T=mat.V_T,
            v_sat_n_x_face=mat.v_sat_n_x_face,
            v_sat_n_y_face=mat.v_sat_n_y_face,
            ct_beta_n_x_face=mat.ct_beta_n_x_face,
            ct_beta_n_y_face=mat.ct_beta_n_y_face,
            pf_gamma_n_x_face=mat.pf_gamma_n_x_face,
            pf_gamma_n_y_face=mat.pf_gamma_n_y_face,
            v_sat_p_x_face=mat.v_sat_p_x_face,
            v_sat_p_y_face=mat.v_sat_p_y_face,
            ct_beta_p_x_face=mat.ct_beta_p_x_face,
            ct_beta_p_y_face=mat.ct_beta_p_y_face,
            pf_gamma_p_x_face=mat.pf_gamma_p_x_face,
            pf_gamma_p_y_face=mat.pf_gamma_p_y_face,
            lateral_bc=mat.lateral_bc,
            v_sat_n_wrap=mat.v_sat_n_wrap,
            v_sat_p_wrap=mat.v_sat_p_wrap,
            ct_beta_n_wrap=mat.ct_beta_n_wrap,
            ct_beta_p_wrap=mat.ct_beta_p_wrap,
            pf_gamma_n_wrap=mat.pf_gamma_n_wrap,
            pf_gamma_p_wrap=mat.pf_gamma_p_wrap,
        )
        dn, dp = continuity_rhs_2d(
            g.x, g.y, phi, n, p,
            mat.G_optical, R,
            mat.D_n, mat.D_p,
            mat.V_T,
            chi=chi_2d, Eg=Eg_2d,
            lateral_bc=mat.lateral_bc,
            interface_y_faces=mat.interface_y_faces,
            A_star_n=mat.A_star_n, A_star_p=mat.A_star_p, T=mat.T_device,
            D_n_x_face=d_eff.D_n_x, D_n_y_face=d_eff.D_n_y,
            D_p_x_face=d_eff.D_p_x, D_p_y_face=d_eff.D_p_y,
            D_n_wrap=d_eff.D_n_wrap, D_p_wrap=d_eff.D_p_wrap,
        )
    else:
        dn, dp = continuity_rhs_2d(
            g.x, g.y, phi, n, p,
            mat.G_optical, R,
            mat.D_n, mat.D_p,
            mat.V_T,
            chi=chi_2d, Eg=Eg_2d,
            lateral_bc=mat.lateral_bc,
            interface_y_faces=mat.interface_y_faces,
            A_star_n=mat.A_star_n, A_star_p=mat.A_star_p, T=mat.T_device,
        )
```

**Note:** The existing call's exact `interface_y_faces` / `A_star_n` / `A_star_p` / `T` kwargs may need to be matched against what's currently passed; check the existing call site and preserve the original kwargs in BOTH branches. The two branches must differ ONLY in the six `D_*_face` / `D_*_wrap` overrides.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd perovskite-sim
pytest tests/unit/twod/ -v
```

Expected: all 3 new tests pass + existing 2D unit tests still pass.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py perovskite_sim/twod/continuity_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): assemble_rhs_2d recomputes D_eff via face-normal μ(E)"
```

---

## Task 5: Disabled-path bit-identical regression

**Files:**
- Modify: `tests/regression/test_twod_validation.py`

The disabled path is bit-identical to the constant-D path by construction (the `else` branch in `assemble_rhs_2d` is unchanged). This task adds a regression test that explicitly verifies J-V output is invariant to whether the field-mobility machinery exists on the dataclass.

- [ ] **Step 1: Write the failing test**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_disabled_path_bit_identical():
    """Stage B(c.2) bit-identical disabled-path regression.

    Compare two 2D J-V sweeps:
      A. Default preset, mode='full', no v_sat / pf_gamma
         → has_field_mobility=False (no params set).
      B. Same preset, mode='legacy', WITH aggressive v_sat=1e2
         → has_field_mobility=False (tier gate disables despite params).

    Both must produce IDENTICAL J-V to floating-point precision because
    the disabled path bypasses the recompute and the constant-D code path
    is unchanged.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack_off    = base                                                       # default mode=full, no v_sat
    stack_legacy = replace(base, mode="legacy", v_sat_n=1e2, v_sat_p=1e2)     # tier-disabled
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
    r_off    = run_jv_sweep_2d(stack=stack_off,    **common_kw)
    r_legacy = run_jv_sweep_2d(stack=stack_legacy, **common_kw)
    V_off    = np.asarray(r_off.V);    J_off    = np.asarray(r_off.J)
    V_legacy = np.asarray(r_legacy.V); J_legacy = np.asarray(r_legacy.J)
    np.testing.assert_array_equal(V_off, V_legacy)
    np.testing.assert_allclose(J_off, J_legacy, rtol=1e-12, atol=0.0,
        err_msg="Disabled field-mobility path is not bit-identical")
```

- [ ] **Step 2: Run test to verify it passes (no implementation needed)**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_field_mobility_disabled_path_bit_identical -v -s -m slow
```

Expected: PASS — the disabled path is already bit-identical because Tasks 1–4 preserved it.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.2) disabled-path bit-identical regression"
```

---

## Task 6: 1D ↔ 2D parity gate + bounded-shift sanity test

**Files:**
- Modify: `tests/regression/test_twod_validation.py`

Two slow regression tests. T6.1 is the **primary correctness gate** — runs 1D Phase 3.2 and 2D Stage B(c.2) on the same lateral-uniform stack with `v_sat=1e2` and asserts the same tight envelope as the B(c.1) Robin parity gates. T6.2 is a sanity check that μ(E) materially perturbs the J-V curve in 2D.

**Tolerance discipline:** keep `(1e-3 V, 5e-4 rel, 1e-3 abs)` tight. If T6.1 measures larger deltas, do NOT loosen blindly — first report the measured deltas and pin near 3× the noise floor (matching the discipline established for the B(c.1) aggressive Robin parity gate).

- [ ] **Step 1: Write the failing tests**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_parity_vs_1d():
    """Stage B(c.2) primary correctness gate: laterally-uniform 2D with face-normal
    μ(E) at v_sat=1e2 matches 1D Phase 3.2 within (1 mV / 5e-4 / 1e-3).

    In a lateral-uniform device E_x ≈ 0, so face-normal μ(E) reduces to
    "y-face μ(E) only" — exactly what 1D does. Expected: bit-identical or
    sub-microvolt deltas, mirroring the B(c.1) Robin parity gates.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack = replace(base, v_sat_n=1e2, v_sat_p=1e2)
    # 1D reference
    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)
    # 2D Stage B(c.2)
    r2 = run_jv_sweep_2d(
        stack=stack,
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
        f"\nμ(E) 1D: V_oc={m1.V_oc*1e3:.4f} mV  J_sc={m1.J_sc:.4f} A/m²  FF={m1.FF:.6f}"
        f"\nμ(E) 2D: V_oc={m2.V_oc*1e3:.4f} mV  J_sc={m2.J_sc:.4f} A/m²  FF={m2.FF:.6f}"
        f"\nΔV_oc = {(m2.V_oc - m1.V_oc)*1e3:+.4f} mV"
        f"  ΔJ_sc/J_sc = {(m2.J_sc - m1.J_sc)/m1.J_sc:+.2e}"
        f"  ΔFF = {m2.FF - m1.FF:+.4e}"
    )
    assert abs(m2.V_oc - m1.V_oc) <= 1e-3, (
        f"μ(E) V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 1 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"μ(E) J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"μ(E) FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_twod_field_mobility_bounded_shift():
    """v_sat=1e2 in 2D shifts J(V) measurably vs v_sat=0 baseline.

    Confirms the μ(E) hook is materially active rather than silently bypassed.
    Asserts max(|J_on - J_off| / |J_off|) > 1e-3 across the sweep.
    """
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack_off = base
    stack_on  = replace(base, v_sat_n=1e2, v_sat_p=1e2)
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
    r_off = run_jv_sweep_2d(stack=stack_off, **common_kw)
    r_on  = run_jv_sweep_2d(stack=stack_on,  **common_kw)
    J_off = np.asarray(r_off.J)
    J_on  = np.asarray(r_on.J)
    rel = np.max(np.abs(J_on - J_off) / (np.abs(J_off) + 1e-12))
    print(f"\nμ(E) bounded-shift: max(|ΔJ|/|J|) = {rel:.3e}")
    assert rel > 1e-3, (
        f"μ(E) shift max(|ΔJ|/|J|) = {rel:.3e} below 1e-3 — hook may be inactive"
    )
```

- [ ] **Step 2: Run tests**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_field_mobility_parity_vs_1d \
       tests/regression/test_twod_validation.py::test_twod_field_mobility_bounded_shift -v -s -m slow
```

Expected: both pass. The parity gate should print sub-mV deltas. If parity FAILS:
1. Read the printed `ΔV_oc / ΔJ_sc / ΔFF` values.
2. Do NOT loosen tolerances. First diagnose: re-check `E_y_face` sign in the recompute, the harmonic vs arithmetic mean choice, the `apply_field_mobility` parameter order, and the Einstein roundtrip (`* V_T` exactly once).
3. If diagnosis points to genuine adaptive-solver noise (not a logic bug), report measured deltas to the user and pin near 3× the noise floor (capped at 2 mV / 1e-3 / 2e-3 per the user's directive).

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.2) 1D↔2D parity gate and bounded-shift sanity"
```

---

## Task 7: Coexistence smoke + periodic-wrap finite-RHS + docs + push

**Files:**
- Modify: `tests/regression/test_twod_validation.py`
- Modify: `perovskite_sim/CLAUDE.md`

Three tests + docs:
- T7.1: μ(E) + Robin contacts + grain boundary, coarse mesh, finite J-V (regression, fast).
- T7.2: μ(E) + lateral_bc="periodic" finite RHS at non-trivial state (already covered by Task 4 step 1's `test_assemble_rhs_2d_field_mobility_finite_periodic` — no extra test needed).
- T7.3: CLAUDE.md Stage B(c.2) section.

- [ ] **Step 1: Write the coexistence smoke test**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
def test_twod_field_mobility_robin_microstructure_coexistence_smoke():
    """μ(E) + Robin contacts + grain boundary on a coarse mesh produce a finite,
    well-ordered J-V (no NaN/Inf, J_sc>0). Cheap test — proves the three per-RHS
    hooks compose without solver hang."""
    from perovskite_sim.twod.microstructure import GrainBoundary
    base = _freeze_ions(load_device_from_yaml(PRESET))
    stack = replace(base,
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
        v_sat_n=1e3, v_sat_p=1e3,
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
    assert np.all(np.isfinite(V)), "Non-finite V in μ(E)+Robin+GB sweep"
    assert np.all(np.isfinite(J)), "Non-finite J in μ(E)+Robin+GB sweep"
    J_sc_sign = _maybe_flip_sign(V, J)[0]
    assert J_sc_sign > 0, "J_sc should be positive under illumination"
```

- [ ] **Step 2: Run test**

```bash
cd perovskite-sim
pytest tests/regression/test_twod_validation.py::test_twod_field_mobility_robin_microstructure_coexistence_smoke -v -s
```

Expected: PASS, runtime under 30 s.

- [ ] **Step 3: Run the FULL test suite to verify zero regressions**

```bash
cd perovskite-sim
pytest                       # fast suite
pytest -m slow               # slow regression suite
```

Expected: both green. Specifically:
- All Stage A parity tests still pass (constant-D unchanged).
- All Stage B(a) microstructure tests still pass.
- All Stage B(c.1) Robin parity tests (ohmic + aggressive) still pass.
- New Stage B(c.2) tests pass.

- [ ] **Step 4: Update `perovskite_sim/CLAUDE.md`**

Locate the Stage B(c.1) Robin section (search for "Stage B(c.1)"). Append after that paragraph:

```markdown
**2D field-dependent mobility — Stage B(c.2) (Phase 6 — Apr 2026).** Ports the 1D Phase 3.2 μ(E) hook to `assemble_rhs_2d` using a **face-normal** formulation: x-faces use only `|E_x_face|`, y-faces use only `|E_y_face|`. When `SimulationMode.use_field_dependent_mobility` is True AND any layer sets a non-zero `v_sat_{n,p}` or `pf_gamma_{n,p}`, `build_material_arrays_2d` sets `MaterialArrays2D.has_field_mobility = True` and pre-computes 18 face arrays of `v_sat / ct_beta / pf_gamma` via **arithmetic mean** (interior x-faces, interior y-faces, periodic-x wrap face). On every RHS call, `assemble_rhs_2d` then calls `recompute_d_eff_2d` (in `perovskite_sim/twod/field_mobility_2d.py`) which:

1. Computes `E_x_face = -(phi[:, 1:] - phi[:, :-1]) / dx[None, :]` and analogously `E_y_face` and the periodic wrap `E_x_wrap`.
2. Forms base mobility `μ_base = harmonic_mean(D_node) / V_T` per face (Einstein relation).
3. Applies the existing 1D `apply_field_mobility(μ_base, |E|, v_sat, beta, gamma_pf)` primitive face-normally — x-faces with `|E_x_face|`, y-faces with `|E_y_face|`. The primitive uses `np.abs(E)` internally so the input sign is irrelevant.
4. Recovers `D_eff = μ_eff · V_T` via Einstein.

The effective per-face D is then forwarded through `continuity_rhs_2d` to `sg_fluxes_2d_n/p` via new keyword-only override kwargs (`D_n_x_face`, `D_n_y_face`, `D_p_x_face`, `D_p_y_face`, plus `D_n_wrap`/`D_p_wrap` for periodic). When the override is `None` (default) the harmonic-mean path runs unchanged → bit-identical to Stage A / B(a) / B(c.1). The `D_n` / `D_p` cache stays at the constant-mobility limit; only the per-face D injected into the flux helpers carries the field correction.

**Mean choice.** `D_n` / `D_p` continue to use harmonic mean (matches `sg_fluxes_2d_*` and 1D); `v_sat` / `ct_beta` / `pf_gamma` use arithmetic mean (avoids harmonic suppression at one-side-zero, which would silently disable CT/PF at heterointerfaces).

**Activation gate.** `_has_field_mobility = sim_mode.use_field_dependent_mobility AND any(v_sat>0 or pf_gamma>0)` — same tier-as-ceiling pattern established by Stage B(c.1). LEGACY tier disables the hook even with params set; FAST tier currently lists μ(E) among the per-RHS hooks it skips, so FAST also stays on the constant-D path.

**Validation.** Lateral-uniform 2D with `v_sat=1e2` matches 1D Phase 3.2 to within sub-mV V_oc / 5e-4 J_sc / 1e-3 FF (`tests/regression/test_twod_validation.py::test_twod_field_mobility_parity_vs_1d`). A bounded-shift sanity test asserts `max(|ΔJ|/|J|) > 1e-3` between μ(E) on and off (`test_twod_field_mobility_bounded_shift`). A coexistence smoke test (`test_twod_field_mobility_robin_microstructure_coexistence_smoke`) confirms μ(E) + Robin + GB compose without NaN/Inf or solver hang. The disabled-path bit-identity is regression-pinned by `test_twod_field_mobility_disabled_path_bit_identical`. Tier-gate is unit-pinned by `test_legacy_mode_disables_field_mobility_in_2d`.

**Out of scope.** Total-|E| (Option B) with cross-axis `E_x` / `E_y` interpolation is explicitly deferred — see `docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-mobility-design.md`. Per-grain mobility (mobility variation inside the absorber driven by microstructure), μ(T) coupling beyond the existing temperature-scaling hook, and any backend / frontend changes are out of scope for Stage B(c.2).
```

- [ ] **Step 5: Commit docs**

```bash
git add perovskite_sim/CLAUDE.md tests/regression/test_twod_validation.py
git commit -m "docs(twod): Stage B(c.2) field-mobility section + coexistence smoke test"
```

- [ ] **Step 6: Push**

```bash
git push origin 2d-extension
```

---

## Self-review checklist

**Spec coverage:**
- §1 Approved formulation (face-normal μ(E)) — Tasks 1, 4 implement it; Option B deferred per spec.
- §2.1 `E_x_face` / `E_y_face` from `phi` — Task 4, Step 4 (in `recompute_d_eff_2d`, called from `assemble_rhs_2d`).
- §2.2 μ₀ and `D_eff` per face — Task 1 (helper) + Task 4 (wired).
- §2.3 Override kwargs on `sg_fluxes_2d_*` and `continuity_rhs_2d` — Tasks 2 and 4.
- §2.4 Periodic-x wrap — Tasks 1 (helper), 3 (build path), 4 (forwarding).
- §2.5 Constant-D path bit-identical — Task 5 explicit regression.
- §3.1 19 new fields on `MaterialArrays2D` — Task 3.
- §3.2 Build path with arithmetic-mean — Task 3.
- §3.3 No `MaterialParams` changes — verified by Task 3 test that uses existing fields via `dc_replace`.
- §3.4 No YAML schema changes — verified by Task 5 (default presets unchanged).
- §3.5 Tier gating — Task 3 includes `test_legacy_mode_disables_field_mobility_in_2d`.
- §4.T1 Disabled bit-identical — Task 5.
- §4.T2 1D↔2D parity — Task 6.
- §4.T3 Bounded-shift — Task 6.
- §4.T4 Coexistence — Task 7.
- §4.T5 Tier gate — Task 3.
- §4.T6 Periodic-wrap finite RHS — Task 4 step 1 (`test_assemble_rhs_2d_field_mobility_finite_periodic`).
- §5 Risk register: R1 sign/magnitude (Task 6 parity), R2 shape (Task 1 shape-mismatch test + per-face validators), R3 wrap (Task 4 finite-periodic test), R4 Einstein (Task 1 Einstein-roundtrip test), R5 stiffness (Task 6 v_sat=1e2 mirrors 1D), R6 face-normal limitation (documented in CLAUDE.md), R7 tier bypass (Task 3 tier-gate test), R8 API back-compat (Task 2 no-override test).

**Placeholder scan:** No "TBD", no "TODO", no "implement later". Every step has full code or exact command. ✓

**Type consistency:**
- `recompute_d_eff_2d` signature uses keyword-only args throughout — all Tasks 4–7 invoke with the same arg names. ✓
- `FieldMobilityDEff.D_n_x` / `D_n_y` / `D_p_x` / `D_p_y` / `D_n_wrap` / `D_p_wrap` — referenced by name in Task 4 Step 4 (`d_eff.D_n_x`, etc.); names match the NamedTuple definition in Task 1 Step 3. ✓
- `MaterialArrays2D` field names (`v_sat_n_x_face`, etc.) defined in Task 3 Step 3 — referenced by exact name in Task 3 Step 5 (return) and Task 4 Step 4 (recompute call). ✓
- `sg_fluxes_2d_n` keyword-only override arg names (`D_n_x_face`, `D_n_y_face`) defined in Task 2 — referenced by `continuity_rhs_2d` forwarding in Task 4. ✓
- `continuity_rhs_2d` signature extension (Task 4 Step 3) lists kwargs in the same order they're forwarded from `assemble_rhs_2d` (Task 4 Step 4). ✓

**Constraints checklist (user-imposed):**
- Disabled path bit-identical → Task 5 explicit regression. ✓
- No YAML schema changes → confirmed: only `MaterialParams` fields used, no loader changes. ✓
- No `physics/field_mobility.py` changes → confirmed: only consumed via import. ✓
- `apply_field_mobility` primitive used unchanged → Task 1 Step 3. ✓
- Option B deferred → spec §1 + CLAUDE.md update (Task 7 Step 4). ✓
- Strict shape checks → Task 1 `_check_x_face` / `_check_y_face` / `_check_wrap` validators. ✓
- Periodic-wrap test → Task 4 Step 1 `test_assemble_rhs_2d_field_mobility_finite_periodic`. ✓
- Einstein-roundtrip test (catches `* V_T` omitted or applied twice) → Task 1 Step 1 `test_recompute_d_eff_einstein_roundtrip_zero_field_mobility_*`. ✓

---

## Rollback / fallback points

| If this fails... | Roll back to... | Diagnosis |
|---|---|---|
| Task 1 unit tests fail (Einstein roundtrip wrong, shape mismatch unexpected, etc.) | Pre-Task-1 HEAD on `2d-extension` | Bug in `recompute_d_eff_2d` math or shape validators. Re-read spec §2.2 and step through with a debugger. |
| Task 2 `sg_fluxes_2d_*` no-override test fails | Pre-Task-2 commit (Task 1 already merged) | API extension changed behaviour when override is None. Re-check the `if override is None: harmonic_mean(...)` branch matches the original inline code. |
| Task 3 `has_field_mobility=False` default test fails | Pre-Task-3 commit | Activation gate or default-init logic wrong. Check tier-gate `AND` and that `None`-init paths are exercised. |
| Task 4 disabled-path RHS test fails | Pre-Task-4 commit | The `else` branch in `assemble_rhs_2d` no longer matches the original call. Compare arg-by-arg. |
| Task 4 enabled-path RHS test passes but Task 6 parity FAILS at >1 mV | Pre-Task-4 commit (do not merge) | Likely a sign or axis bug in `recompute_d_eff_2d`. Re-check `E_x_face` vs `E_y_face` use, harmonic-D mean orientation, axis arguments to `apply_field_mobility`. |
| Task 6 parity passes but bounded-shift FAILS | Investigate, do NOT merge | μ(E) is being computed but somehow not affecting the J-V. Likely the override kwargs aren't being forwarded all the way through `continuity_rhs_2d` to `sg_fluxes_2d_*`. |
| Task 7 coexistence smoke fails (NaN/Inf) | Investigate first; rollback only if unfixable | Probable solver-stiffness interaction. Try lower `v_sat` (1e3 → 1e4) or larger `settle_t`. If physics-level conflict, consult and consider isolating the failure mode. |
| Task 5 disabled-path regression passes locally but Task 6 parity fails on the slow suite | Investigate `BLAS_NUM_THREADS` pinning | The slow-suite BLAS pinning hook may interact with the new code path differently. Confirm `tests/conftest.py` still pins BLAS for slow tests after Stage B(c.2). |

**Branch hygiene:** all 7 tasks land on `2d-extension` directly (per the user's stated integration-branch policy). If any task introduces a regression that can't be fixed within the same task's commit window, isolate to a new feature branch (`feat/2d-stage-b-c2-field-mobility-recovery`) for safekeeping and revert on `2d-extension` so the integration branch stays green.

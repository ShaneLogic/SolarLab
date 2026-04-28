# Stage B(c.1) — 2D Robin/Selective Contacts Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 1D Robin/selective-contact boundary condition (Phase 3.3) to the 2D solver, enabling per-carrier surface-recombination velocities at y=0 (HTL) and y=Ny-1 (ETL).

**Architecture:** Private helper `_apply_robin_contacts_2d` in `solver_2d.py` calls the existing `physics/contacts.selective_contact_flux` for each boundary x-column, corrects the Neumann base computation in `continuity_rhs_2d`, and replaces the four Dirichlet pins when selective contacts are active.

**Tech stack:** Python/NumPy, existing `perovskite_sim` library; no new files, no backend/frontend changes.

---

## 1. Scope

**In scope — Stage B(c.1):**
- 2D Robin boundary condition for `n` and `p` at the top (y=0) and bottom (y=Ny-1) contacts.
- Contacts are **spatially uniform in x** — `S_n_top`, `S_p_top`, `S_n_bot`, `S_p_bot` are scalar floats.
- `S = 0` is the perfectly-blocking (Neumann) limit. `S → ∞` recovers the ohmic Dirichlet pin.
- Parity gate: laterally-uniform 2D with Robin must match 1D Phase 3.3 within sub-mV V_oc / 5×10⁻⁴ J_sc / 10⁻³ FF.

**Out of scope — deferred:**
- Schottky barrier contacts (requires `schottky_equilibrium_n/p` override of `n_eq`; machinery is available but not wired in this stage).
- Spatially patterned contacts across x (per-column S arrays).
- Stage B(c.2) μ(E) and Stage B(c.3) radiative reabsorption.
- 2D ions, 2D trap profiles, frontend/backend changes.

---

## 2. Architecture

### Files touched

| File | Change |
|------|--------|
| `perovskite_sim/twod/solver_2d.py` | Add 5 fields to `MaterialArrays2D`; populate in `build_material_arrays_2d`; add `_apply_robin_contacts_2d`; update `assemble_rhs_2d` guard |
| `perovskite_sim/twod/continuity_2d.py` | Add one-sided Neumann y-divergence at boundary rows j=0 and j=Ny-1 (2 lines per carrier); update docstring |
| `tests/unit/twod/test_solver_2d.py` | Unit tests for `_apply_robin_contacts_2d` through `assemble_rhs_2d` |
| `tests/regression/test_twod_validation.py` | Robin parity gate (1D↔2D) + bounded-shift test |

**No new files. No changes to `physics/contacts.py`, `microstructure.py`, backend, or frontend.**

### New `MaterialArrays2D` fields

```python
has_selective_contacts: bool = False
S_n_top: float = 0.0   # electron SRV at y=0 (HTL); mapped from DeviceStack.S_n_left
S_p_top: float = 0.0   # hole SRV at y=0
S_n_bot: float = 0.0   # electron SRV at y=Ny-1 (ETL); mapped from DeviceStack.S_n_right
S_p_bot: float = 0.0   # hole SRV at y=Ny-1
```

All five have defaults that disable the feature, preserving backward compatibility.

### `_build_material_arrays_2d` mapping (with required comment)

```python
# "left/right" are 1D transport-axis names inherited from DeviceStack.
# In 2D the transport axis is y, so left→top (y=0, HTL) and right→bottom (y=Ny-1, ETL).
has_selective_contacts = stack.has_selective_contacts
S_n_top = float(stack.S_n_left)  if stack.S_n_left  is not None else 0.0
S_p_top = float(stack.S_p_left)  if stack.S_p_left  is not None else 0.0
S_n_bot = float(stack.S_n_right) if stack.S_n_right is not None else 0.0
S_p_bot = float(stack.S_p_right) if stack.S_p_right is not None else 0.0
```

**DeviceStack fields are not renamed** — `S_n_left` / `S_p_left` / `S_n_right` / `S_p_right` stay as-is for backward compatibility with 1D configs and tests.

---

## 3. Boundary / Contact Convention

```
y = 0         → top contact    = HTL = p-selective = 1D "left"
y = Ny-1      → bottom contact = ETL = n-selective = 1D "right"
```

| 1D field | 2D alias | Contact side | y row |
|----------|----------|--------------|-------|
| `DeviceStack.S_n_left`   | `mat.S_n_top` | HTL / p-contact | y = 0 |
| `DeviceStack.S_p_left`   | `mat.S_p_top` | HTL / p-contact | y = 0 |
| `DeviceStack.S_n_right`  | `mat.S_n_bot` | ETL / n-contact | y = Ny-1 |
| `DeviceStack.S_p_right`  | `mat.S_p_bot` | ETL / n-contact | y = Ny-1 |

### Equilibrium carrier densities

Use the **existing** `MaterialArrays2D` fields (already `(Nx,)` arrays):

- Top contact (y=0):     `mat.n_eq_left`, `mat.p_eq_left`   ← values = `n_L`, `p_L`
- Bottom contact (y=Ny-1): `mat.n_eq_right`, `mat.p_eq_right` ← values = `n_R`, `p_R`

⚠️ **Code comment bug:** The current comments on these fields at lines 67–70 of `solver_2d.py` say "bottom-contact" for `n_eq_left` and "top-contact" for `n_eq_right` — this is **backwards**. The values are correct; only the comments are wrong. Fix the comments in the same commit that adds Robin support.

---

## 4. Continuity RHS Change

### 4a. `continuity_rhs_2d` — add one-sided Neumann at boundary rows

**Current code** (lines 213–217) — boundary rows are skipped entirely (zero y-divergence):
```python
div_y_n = np.zeros_like(phi)
div_y_p = np.zeros_like(phi)
if Ny > 2:
    div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / hy_cell[1:-1, None]
    div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / hy_cell[1:-1, None]
```

**New code** — add Neumann one-sided divergence at j=0 and j=Ny-1:
```python
div_y_n = np.zeros_like(phi)
div_y_p = np.zeros_like(phi)
if Ny > 2:
    div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / hy_cell[1:-1, None]
    div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / hy_cell[1:-1, None]
    # One-sided Neumann at contacts (zero wall flux). Dirichlet path in
    # assemble_rhs_2d overwrites these rows to 0 when contacts are ohmic;
    # Robin path subtracts the contact flux correction.
    div_y_n[0,  :] =  Jy_n[0,  :] / hy_cell[0]
    div_y_n[-1, :] = -Jy_n[-1, :] / hy_cell[-1]
    div_y_p[0,  :] =  Jy_p[0,  :] / hy_cell[0]
    div_y_p[-1, :] = -Jy_p[-1, :] / hy_cell[-1]
```

**Dirichlet path is unchanged.** `assemble_rhs_2d` still overwrites the boundary rows to 0 when `has_selective_contacts` is False, discarding whatever `continuity_rhs_2d` produced. The new lines are no-ops for all existing tests.

**Sign derivation for Neumann boundary rows:**

Both `Jy_n` and `Jy_p` use the **same +y sign convention** (positive = conventional current in +y direction). For the boundary control volumes:

- j=0 (top): one face only at j=0→1; wall flux = 0.
  `div_y[0] = (Jy[0] − 0) / hy_cell[0] = +Jy[0] / hy_cell[0]`
- j=Ny-1 (bot): one face only at j=Ny-2→Ny-1; wall flux = 0.
  `div_y[-1] = (0 − Jy[-1]) / hy_cell[-1] = −Jy[-1] / hy_cell[-1]`

### 4b. `_apply_robin_contacts_2d` — Robin wall-flux correction in `solver_2d.py`

```python
from perovskite_sim.physics.contacts import selective_contact_flux
from perovskite_sim.constants import Q


def _apply_robin_contacts_2d(
    dn: np.ndarray,   # (Ny, Nx) — modified in-place and returned
    dp: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    mat: "MaterialArrays2D",
) -> tuple[np.ndarray, np.ndarray]:
    """Replace Neumann wall flux with Robin contact flux at y=0 and y=Ny-1.

    continuity_rhs_2d already included a zero-flux Neumann contribution
    at the boundary rows.  This helper subtracts that implicit zero and
    adds the actual Robin flux J = ±q·S·(density − density_eq) instead.
    The four Dirichlet pins (dn/dp = 0) must be skipped by the caller.

    Outward-normal sign:
      y=0   (top / HTL / 1D-left)  outward normal = -y → side="left"
      y=Ny-1 (bot / ETL / 1D-right) outward normal = +y → side="right"
    """
    # Half-cell thickness at each contact (control-volume scaling)
    hy_top = (mat.grid.y[1]  - mat.grid.y[0])  / 2.0
    hy_bot = (mat.grid.y[-1] - mat.grid.y[-2]) / 2.0

    # --- top contact (y=0, HTL, side="left") --------------------------------
    J_n_top = selective_contact_flux(
        n[0, :], mat.n_eq_left, mat.S_n_top, carrier="n", side="left"
    )
    J_p_top = selective_contact_flux(
        p[0, :], mat.p_eq_left, mat.S_p_top, carrier="p", side="left"
    )
    dn[0, :] -= J_n_top / (Q * hy_top)   # electrons: dn = +div_n/Q → subtract
    dp[0, :] += J_p_top / (Q * hy_top)   # holes:     dp = −div_p/Q → add (opposite sign)

    # --- bottom contact (y=Ny-1, ETL, side="right") -------------------------
    J_n_bot = selective_contact_flux(
        n[-1, :], mat.n_eq_right, mat.S_n_bot, carrier="n", side="right"
    )
    J_p_bot = selective_contact_flux(
        p[-1, :], mat.p_eq_right, mat.S_p_bot, carrier="p", side="right"
    )
    dn[-1, :] += J_n_bot / (Q * hy_bot)   # electrons: add (J_n_bot < 0 when n > n_eq)
    dp[-1, :] -= J_p_bot / (Q * hy_bot)   # holes:     subtract (J_p_bot > 0 when p > p_eq)

    return dn, dp
```

### 4c. `assemble_rhs_2d` guard

Replace the current unconditional Dirichlet block:
```python
# Before:
dn[0, :] = 0.0
dn[-1, :] = 0.0
dp[0, :] = 0.0
dp[-1, :] = 0.0

# After:
if mat.has_selective_contacts:
    dn, dp = _apply_robin_contacts_2d(dn, dp, n, p, mat)
else:
    # Dirichlet ohmic contacts — unchanged from Stage A
    dn[0, :] = 0.0
    dn[-1, :] = 0.0
    dp[0, :] = 0.0
    dp[-1, :] = 0.0
```

---

## 5. Sign Conventions

### Flux sign from `selective_contact_flux`

Both `Jy_n` and `Jy_p` in `flux_2d.py` use **+y = conventional current positive**. `selective_contact_flux` returns values in the same +y sign convention when adapted to the y-transport axis.

### Robin residual correction — approved implementation target

```python
dn[0,  :] -= J_n_top / (Q * hy_top)
dp[0,  :] += J_p_top / (Q * hy_top)   # ← +=, NOT -=
dn[-1, :] += J_n_bot / (Q * hy_bot)
dp[-1, :] -= J_p_bot / (Q * hy_bot)   # ← -=, NOT +=
```

### Sign table

| Boundary | Carrier | `selective_contact_flux` result | Physical meaning | Residual correction |
|----------|---------|--------------------------------|-----------------|---------------------|
| y=0, side="left" | n | `+Q·S·(n−n_eq)` | e⁻ leaving upward (−y) | `dn[0,:] −= J / (Q·hy_top)` |
| y=0, side="left" | p | `−Q·S·(p−p_eq)` | holes leaving upward (−y) | `dp[0,:] += J / (Q·hy_top)` |
| y=Ny-1, side="right" | n | `−Q·S·(n−n_eq)` | e⁻ leaving downward (+y) | `dn[-1,:] += J / (Q·hy_bot)` |
| y=Ny-1, side="right" | p | `+Q·S·(p−p_eq)` | holes leaving downward (+y) | `dp[-1,:] −= J / (Q·hy_bot)` |

**Key rule:** `dn` and `dp` corrections are always **opposite sign** at each boundary. This follows from `dn = +div_n/Q` (positive sign) vs `dp = −div_p/Q` (negative sign) in the continuity equations.

**Previously proposed incorrect signs (do not use):**
- `dp[0, :] -= J_p_top` — wrong, must be `+=`
- `dp[-1, :] += J_p_bot` — wrong, must be `-=`

### Physical limit checks

| S value | Result |
|---------|--------|
| S = 0 | Zero Robin flux; boundary node evolves freely (Neumann limit — blocking contact) |
| S → ∞ | Relaxation time `dx/S → 0`; node pins to `n_eq`; recovers Dirichlet ohmic limit |
| All S = 0 + disable guard | Exactly equivalent to Stage A Dirichlet (via the `else` branch) |

---

## 6. Tests

### T1 — Backward-compatibility unit test (bit-exact Dirichlet)

Configure `build_material_arrays_2d` without selective contacts (`has_selective_contacts = False`). Assert `assemble_rhs_2d` output is bit-identical to the pre-Robin run. Run against the existing Stage-A regression values to confirm the `else` branch is untouched.

### T2 — Left/right → top/bottom mapping swap test

Configure only `S_n_right` (ETL) with a finite value; leave all other S = 0. Call `assemble_rhs_2d` and inspect the returned `dn`. Assert the non-trivial Robin correction appears on `dn[-1, :]` (bottom row), not `dn[0, :]` (top row). Mirror: configure only `S_p_left` (HTL) and assert non-trivial `dp[0, :]` only.

### T3 — dp sign correctness test

Configure a state with p[0, :] > mat.p_eq_left (excess holes at top contact). Assert `dp[0, :]` DECREASES relative to the zero-S baseline (i.e., holes are removed, not added). Repeat for `dp[-1, :]` with p[-1, :] > mat.p_eq_right.

### T4 — 1D ↔ 2D Robin parity gate (regression, primary correctness check)

Config: Beer-Lambert nip MAPbI3 with `S_n_right = 1e-2 m/s` (and matching S values on other carriers as needed to produce a 1D-2D comparable state). Run both 1D Phase 3.3 `run_jv_sweep` and 2D `run_jv_sweep_2d` on a laterally-uniform grid. Assert:

```
|V_oc_2D − V_oc_1D| < 1e-3 V        (sub-mV)
|J_sc_2D − J_sc_1D| / |J_sc_1D| < 5e-4
|FF_2D − FF_1D| < 1e-3
```

This is the **primary correctness gate**. If this fails, the Robin port is wrong. Passes should be pinned as a regression baseline.

### T5 — Bounded-shift sanity test (Robin-enabled vs Robin-disabled 2D)

Same Beer-Lambert MAPbI3, compare Robin-enabled 2D to Stage-A Dirichlet 2D baseline. Assert:

```
1 mV ≤ ΔV_oc ≤ 150 mV   (Robin must move V_oc by a physically nonzero amount)
```

This test confirms the feature is active and non-trivial, but it is **secondary** to the parity gate. It must not be used as the primary correctness criterion.

### T6 — Microstructure + Robin coexistence smoke test

Configure `nip_MAPbI3_singleGB.yaml` (Stage B(a) preset) AND a finite `S_n_right`. Run `run_jv_sweep_2d`. Assert no NaN/Inf in the J-V output and the curve is physically ordered (monotone, no spikes). This verifies that the Robin hook and GB τ-field coexist without numerical issues.

---

## 7. Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **dp sign error** (`+=` vs `-=`) | Critical | Traced explicitly in §5; T3 catches it |
| **left/right ↔ top/bottom swap** | Critical | Documented comment fix; T2 catches it |
| **Dirichlet back-compat regression** | High | T1 bit-exact check; `else` branch is structurally identical to current code |
| **hy_top/hy_bot scaling error** | Medium | Derive from `mat.grid.y[1]−mat.grid.y[0]` / `y[-1]−y[-2]`; matches `continuity_rhs_2d`'s `hy_cell` formula |
| **Slow parity regression (T4) runtime** | Low | Uses Beer-Lambert (not TMM); mark `@pytest.mark.slow` only if wall time exceeds ~5s; can use coarse grid `Ny_per_layer=5, Nx=5` |
| **Neumann base computation wrong** | Medium | Two new lines in `continuity_rhs_2d` are straightforward; T1 (Dirichlet path unchanged) provides guard |
| **`mat.n_eq_left` comment mismatch** | Low | Fix in same commit; no functional risk |

---

## 8. Unresolved Decisions

**None.** All architectural, sign, and field-naming questions have been resolved in the design session. The spec is implementation-ready.

---

## 9. Implementation Task List (for writing-plans / subagent-driven-development)

**Task 1 — Fix misleading comments on `n_eq_left/right` in `solver_2d.py`**
Files: `solver_2d.py` lines 67–70. Change "bottom-contact" → "top contact (y=0, HTL)" and "top-contact" → "bottom contact (y=Ny-1, ETL)". Commit.

**Task 2 — Add Robin fields to `MaterialArrays2D` and `build_material_arrays_2d`**
Files: `solver_2d.py`. Add `has_selective_contacts`, `S_n_top`, `S_p_top`, `S_n_bot`, `S_p_bot` with zero defaults. Populate in `build_material_arrays_2d` using the `left/right → top/bot` bridge with the required comment. Write unit test that verifies correct mapping (T2 mapping direction). Commit.

**Task 3 — Update `continuity_rhs_2d` to compute one-sided Neumann at boundary rows**
Files: `continuity_2d.py`. Add 4 lines (see §4a). Update docstring. Write test that the Dirichlet path in `assemble_rhs_2d` is still bit-exact after this change (T1). Commit.

**Task 4 — Implement `_apply_robin_contacts_2d` + `assemble_rhs_2d` guard**
Files: `solver_2d.py`. Add private helper (§4b) and update the guard (§4c). Write T3 (dp sign test) and T2 (mapping swap test). Commit.

**Task 5 — Robin parity gate: 1D ↔ 2D regression test (T4)**
Files: `tests/regression/test_twod_validation.py`. Add `test_twod_robin_parity_vs_1d`. Run the test to confirm it passes. Pin result. Commit.

**Task 6 — Bounded-shift sanity test (T5) + microstructure coexistence smoke test (T6)**
Files: `tests/regression/test_twod_validation.py`. Add `test_twod_robin_bounded_shift` and `test_twod_robin_microstructure_coexistence`. Commit.

**Task 7 — Full test suite green + CLAUDE.md update**
Run `pytest` (unit + integration). Run `pytest -m slow` to confirm the parity gate is clean. Update the Stage B section in `perovskite-sim/CLAUDE.md` to document the Robin wiring, sign conventions, and the `has_selective_contacts` guard. Commit.

# Stage B(c.1) — 2D Robin/Selective Contacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 1D Phase 3.3 Robin/selective-contact boundary condition to the 2D solver by adding a private `_apply_robin_contacts_2d` helper in `solver_2d.py`, reusing `physics/contacts.selective_contact_flux` unchanged.

**Architecture:** (1) `continuity_rhs_2d` gains one-sided Neumann y-divergence at boundary rows; (2) `assemble_rhs_2d` guards on `mat.has_selective_contacts` — when True it calls `_apply_robin_contacts_2d` instead of zeroing the four boundary rows; (3) five new scalar fields on `MaterialArrays2D` carry the Robin configuration, populated from `DeviceStack.S_*_left/right` via a documented `left/right → top/bot` bridge.

**Tech Stack:** Python/NumPy, `scipy.integrate.solve_ivp (Radau)`, existing `perovskite_sim` library; no new files, no backend or frontend changes.

---

## File Structure

| File | Role |
|------|------|
| `perovskite_sim/twod/solver_2d.py` | Fix comments (Task 1); add 5 `MaterialArrays2D` fields + build mapping (Task 2); add `_apply_robin_contacts_2d` + `assemble_rhs_2d` guard (Task 4) |
| `perovskite_sim/twod/continuity_2d.py` | Add 4 lines: one-sided Neumann at j=0 and j=Ny-1 (Task 3) |
| `tests/unit/twod/test_solver_2d.py` | Unit tests: Dirichlet backward-compat (Task 3), Robin field mapping (Task 2), dp sign + boundary routing (Task 4) |
| `tests/regression/test_twod_validation.py` | Robin 1D↔2D parity gate (Task 5), bounded-shift (Task 6), coexistence smoke (Task 6) |
| `perovskite_sim/CLAUDE.md` | Stage B(c.1) section update (Task 7) |

**Current sizes:** `solver_2d.py` = 494 lines, `continuity_2d.py` = 226 lines. After all changes solver_2d.py adds ~60 lines → ~554, well within the 800-line cap.

---

## Background for the implementer

The perovskite simulator historically used **Dirichlet (ohmic) contacts** at the outer boundaries: the carrier densities at y=0 and y=Ny-1 were pinned to equilibrium values and their time derivatives were forced to zero. Phase 3.3 added **Robin (selective) contacts** to the 1D solver; this plan ports that to 2D.

**Key physics:** A selective contact with surface recombination velocity S applies the boundary condition
`J_contact = ±q·S·(density − density_eq)` at the contact face. `S=0` is a blocking contact (Neumann), `S→∞` recovers ohmic Dirichlet.

**Sign rules (verified from first principles — do not change):**
```
dn[0,  :] -= J_n_top / (Q * hy_top)    # top, electrons
dp[0,  :] += J_p_top / (Q * hy_top)    # top, holes ← += NOT -=
dn[-1, :] += J_n_bot / (Q * hy_bot)    # bottom, electrons
dp[-1, :] -= J_p_bot / (Q * hy_bot)    # bottom, holes ← -= NOT +=
```
The `dp` signs are OPPOSITE to `dn` because `dp = −div_p/Q` has a leading minus in the continuity equation.

**Contact / axis convention:**
```
y = 0       → top contact    = HTL = p-selective = 1D "left"  = DeviceStack.S_*_left
y = Ny-1    → bottom contact = ETL = n-selective = 1D "right" = DeviceStack.S_*_right
```

**Equilibrium field naming:** `MaterialArrays2D` already has `n_eq_left (= n_L)` and `n_eq_right (= n_R)` as `(Nx,)` arrays. Despite misleading comments ("bottom-contact" / "top-contact"), the values are correct: `n_eq_left` is the top/HTL equilibrium, `n_eq_right` is the bottom/ETL equilibrium. Task 1 fixes those comments.

**Run all tests from the `perovskite-sim/` directory:**
```bash
cd /path/to/SolarLab/perovskite-sim
pytest tests/unit/twod/test_solver_2d.py -v       # unit tests
pytest tests/regression/test_twod_validation.py -v # regression
pytest -m slow                                     # full slow suite
```

---

## Task 1: Fix misleading comments on `n_eq_left/right`

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py:67-70`

This is a non-functional comment fix. No test needed. It prevents future confusion when wiring Robin contacts.

- [ ] **Step 1: Open `solver_2d.py` and locate lines 67–70**

The current comments are wrong:
```python
n_eq_left: np.ndarray         # (Nx,)  bottom-contact electron density
p_eq_left: np.ndarray         # (Nx,)
n_eq_right: np.ndarray        # (Nx,)  top-contact
p_eq_right: np.ndarray        # (Nx,)
```

- [ ] **Step 2: Replace with corrected comments**

```python
n_eq_left: np.ndarray         # (Nx,)  top contact (y=0, HTL); value = mat1d.n_L
p_eq_left: np.ndarray         # (Nx,)  top contact (y=0, HTL); value = mat1d.p_L
n_eq_right: np.ndarray        # (Nx,)  bottom contact (y=Ny-1, ETL); value = mat1d.n_R
p_eq_right: np.ndarray        # (Nx,)  bottom contact (y=Ny-1, ETL); value = mat1d.p_R
```

- [ ] **Step 3: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py
git commit -m "fix(twod): correct misleading n_eq_left/right comments in MaterialArrays2D"
```

---

## Task 2: Add Robin fields to `MaterialArrays2D` and populate in `build_material_arrays_2d`

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Test: `tests/unit/twod/test_solver_2d.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def test_material_arrays_2d_default_no_selective_contacts():
    """Without S values on the stack, has_selective_contacts is False and S fields are 0."""
    stack = _stack()  # configs/nip_MAPbI3.yaml — no S values
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_selective_contacts is False
    assert mat.S_n_top == 0.0
    assert mat.S_p_top == 0.0
    assert mat.S_n_bot == 0.0
    assert mat.S_p_bot == 0.0


def test_material_arrays_2d_right_maps_to_bot():
    """DeviceStack.S_n_right must appear in mat.S_n_bot (bottom contact, ETL)."""
    from dataclasses import replace as dc_replace
    import pytest
    stack_with_s = dc_replace(_stack(), S_n_right=1e-2)
    layers = _layers_for_stack(stack_with_s)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_with_s, Microstructure())
    assert mat.has_selective_contacts is True
    assert mat.S_n_bot == pytest.approx(1e-2)
    assert mat.S_n_top == 0.0
    assert mat.S_p_top == 0.0
    assert mat.S_p_bot == 0.0


def test_material_arrays_2d_left_maps_to_top():
    """DeviceStack.S_p_left must appear in mat.S_p_top (top contact, HTL)."""
    from dataclasses import replace as dc_replace
    import pytest
    stack_with_s = dc_replace(_stack(), S_p_left=5e3)
    layers = _layers_for_stack(stack_with_s)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_with_s, Microstructure())
    assert mat.has_selective_contacts is True
    assert mat.S_p_top == pytest.approx(5e3)
    assert mat.S_n_top == 0.0
    assert mat.S_n_bot == 0.0
    assert mat.S_p_bot == 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_default_no_selective_contacts \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_right_maps_to_bot \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_left_maps_to_top -v
```

Expected: FAIL — `MaterialArrays2D` has no `has_selective_contacts` attribute.

- [ ] **Step 3: Add 5 new fields to `MaterialArrays2D` in `solver_2d.py`**

After line 74 (`layer_role_per_y: tuple[str, ...]`), add:

```python
    # --- Stage B(c.1): Robin / selective contacts --------------------------------
    # has_selective_contacts is True iff any of the four S values is not None on
    # the originating DeviceStack.  S values default to 0.0 (zero = Neumann /
    # blocking; Dirichlet ohmic behaviour is restored via the assemble_rhs_2d
    # guard when has_selective_contacts is False).
    #
    # "left/right" are 1D transport-axis names inherited from DeviceStack.
    # In 2D the transport axis is y, so left→top (y=0, HTL) and right→bottom
    # (y=Ny-1, ETL).  The DeviceStack field names are intentionally unchanged.
    has_selective_contacts: bool = False
    S_n_top: float = 0.0   # electron SRV at y=0  (HTL); from DeviceStack.S_n_left
    S_p_top: float = 0.0   # hole    SRV at y=0  (HTL); from DeviceStack.S_p_left
    S_n_bot: float = 0.0   # electron SRV at y=Ny-1 (ETL); from DeviceStack.S_n_right
    S_p_bot: float = 0.0   # hole    SRV at y=Ny-1 (ETL); from DeviceStack.S_p_right
```

- [ ] **Step 4: Add computation and pass-through in `build_material_arrays_2d`**

Just before the `return MaterialArrays2D(...)` call (currently at line 203), insert:

```python
    # Selective contacts: mirror the 1D mol.py logic.  has_selective_contacts
    # is True iff any S value is explicitly set (not None) on the DeviceStack.
    _has_sc = bool(
        stack.S_n_left  is not None
        or stack.S_p_left  is not None
        or stack.S_n_right is not None
        or stack.S_p_right is not None
    )
    S_n_top = float(stack.S_n_left)  if stack.S_n_left  is not None else 0.0
    S_p_top = float(stack.S_p_left)  if stack.S_p_left  is not None else 0.0
    S_n_bot = float(stack.S_n_right) if stack.S_n_right is not None else 0.0
    S_p_bot = float(stack.S_p_right) if stack.S_p_right is not None else 0.0
```

Then add the new fields to the `return MaterialArrays2D(...)` call (after `layer_role_per_y=layer_role_per_y,`):

```python
        has_selective_contacts=_has_sc,
        S_n_top=S_n_top, S_p_top=S_p_top,
        S_n_bot=S_n_bot, S_p_bot=S_p_bot,
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_default_no_selective_contacts \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_right_maps_to_bot \
       tests/unit/twod/test_solver_2d.py::test_material_arrays_2d_left_maps_to_top -v
```

Expected: PASS.

- [ ] **Step 6: Run full unit suite to check no regressions**

```bash
pytest tests/unit/twod/test_solver_2d.py -v
```

Expected: all existing tests plus the 3 new ones pass.

- [ ] **Step 7: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): add Robin fields to MaterialArrays2D with left/right→top/bot mapping"
```

---

## Task 3: One-sided Neumann at boundary rows in `continuity_rhs_2d`

**Files:**
- Modify: `perovskite_sim/twod/continuity_2d.py:209-217`
- Test: `tests/unit/twod/test_solver_2d.py`

**Why this change:** Currently `continuity_rhs_2d` skips the y-divergence at boundary rows entirely, leaving them zero. Then `assemble_rhs_2d` overwrites the boundary rows to 0 for Dirichlet. The Robin path needs the one-sided interior face flux to already be in `dn/dp` before applying the Robin wall correction. This change adds it — the Dirichlet path is unchanged because `assemble_rhs_2d` still overwrites those rows to 0 in its `else` branch.

- [ ] **Step 1: Write the backward-compatibility test**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def test_assemble_rhs_2d_dirichlet_boundary_rows_exactly_zero():
    """Backward-compat: without selective contacts, all four boundary rows of dydt are 0."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack()  # no S values → Dirichlet
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    Nn = g.n_nodes
    dn = dydt[:Nn].reshape((g.Ny, g.Nx))
    dp = dydt[Nn:].reshape((g.Ny, g.Nx))
    np.testing.assert_array_equal(dn[0, :],  0.0, err_msg="dn top row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dn[-1, :], 0.0, err_msg="dn bot row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dp[0, :],  0.0, err_msg="dp top row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dp[-1, :], 0.0, err_msg="dp bot row should be 0 (Dirichlet)")
```

- [ ] **Step 2: Run test to confirm it passes (it tests current behaviour)**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_dirichlet_boundary_rows_exactly_zero -v
```

Expected: PASS — this test documents the current Dirichlet behaviour before we change anything.

- [ ] **Step 3: Update `continuity_rhs_2d` in `continuity_2d.py`**

Find the block at lines 209–217:
```python
    # -----------------------------------------------------------------------
    # y-divergence on interior rows (j=1..Ny-2). j=0 and j=Ny-1 are Dirichlet
    # contacts — caller pins them to zero so we skip them here.
    # -----------------------------------------------------------------------
    div_y_n = np.zeros_like(phi)
    div_y_p = np.zeros_like(phi)
    if Ny > 2:
        div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / hy_cell[1:-1, None]
        div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / hy_cell[1:-1, None]
```

Replace with:
```python
    # -----------------------------------------------------------------------
    # y-divergence. Interior rows (j=1..Ny-2) use the full two-face formula.
    # Boundary rows (j=0, j=Ny-1) use a one-sided Neumann formula (zero wall
    # flux assumed). The Dirichlet path in assemble_rhs_2d overwrites these
    # rows to 0 for ohmic contacts; the Robin path applies a flux correction.
    #
    # Sign derivation for boundary rows (both carriers, +y convention):
    #   j=0   (top):    div[0]  = (Jy[0]   - 0) / hy_cell[0]   = +Jy[0]/hy
    #   j=Ny-1 (bot):   div[-1] = (0 - Jy[-1]) / hy_cell[-1]   = -Jy[-1]/hy
    # -----------------------------------------------------------------------
    div_y_n = np.zeros_like(phi)
    div_y_p = np.zeros_like(phi)
    if Ny > 2:
        div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / hy_cell[1:-1, None]
        div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / hy_cell[1:-1, None]
        # One-sided Neumann at y=0 and y=Ny-1 (zero wall flux).
        div_y_n[0,  :] =  Jy_n[0,  :] / hy_cell[0]
        div_y_n[-1, :] = -Jy_n[-1, :] / hy_cell[-1]
        div_y_p[0,  :] =  Jy_p[0,  :] / hy_cell[0]
        div_y_p[-1, :] = -Jy_p[-1, :] / hy_cell[-1]
```

- [ ] **Step 4: Run the backward-compat test again — it must still pass**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_assemble_rhs_2d_dirichlet_boundary_rows_exactly_zero -v
```

Expected: PASS — `assemble_rhs_2d` still overwrites boundary rows to 0 in the Dirichlet path.

- [ ] **Step 5: Run full unit suite**

```bash
pytest tests/unit/twod/test_solver_2d.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/twod/continuity_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): add one-sided Neumann y-divergence at boundary rows for Robin port"
```

---

## Task 4: `_apply_robin_contacts_2d` helper + `assemble_rhs_2d` guard

**Files:**
- Modify: `perovskite_sim/twod/solver_2d.py`
- Test: `tests/unit/twod/test_solver_2d.py`

- [ ] **Step 1: Write the dp-sign tests**

Add to `tests/unit/twod/test_solver_2d.py`:

```python
def _make_grid_and_mat(stack, Nx=4):
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=Nx, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    return g, mat


def test_robin_dp_top_decreases_with_excess_holes():
    """dp[0,:] must be smaller under Robin (S_p_top>0, p>p_eq) than under pure Neumann."""
    from dataclasses import replace as dc_replace
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    # Neumann baseline: S_p_left=0.0 triggers Robin mode but contributes zero correction.
    stack_neumann = dc_replace(stack_base, S_p_left=0.0)
    stack_robin   = dc_replace(stack_base, S_p_left=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    # State: p[0,:] = 2 × p_eq (excess holes at top boundary)
    n0 = float(mat_neumann.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0[0, :] = 2.0 * float(mat_neumann.p_eq_left[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dp_neumann = dydt_n[Nn:].reshape(g.Ny, g.Nx)
    dp_robin   = dydt_r[Nn:].reshape(g.Ny, g.Nx)
    # Robin removes excess holes → dp[0,:] must decrease
    assert np.all(dp_robin[0, :] < dp_neumann[0, :]), (
        "dp[0,:] should decrease under Robin when p > p_eq (wrong sign or no correction)"
    )


def test_robin_dp_bot_decreases_with_excess_holes():
    """dp[-1,:] must be smaller under Robin (S_p_bot>0, p>p_eq) than pure Neumann."""
    from dataclasses import replace as dc_replace
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    stack_neumann = dc_replace(stack_base, S_p_right=0.0)
    stack_robin   = dc_replace(stack_base, S_p_right=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    n0 = float(mat_neumann.n_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0[-1, :] = 2.0 * float(mat_neumann.p_eq_right[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dp_neumann = dydt_n[Nn:].reshape(g.Ny, g.Nx)
    dp_robin   = dydt_r[Nn:].reshape(g.Ny, g.Nx)
    assert np.all(dp_robin[-1, :] < dp_neumann[-1, :]), (
        "dp[-1,:] should decrease under Robin when p > p_eq at bottom"
    )


def test_robin_correction_routes_to_correct_boundary():
    """S_n_right correction appears on dn[-1,:] not dn[0,:]; top row is unaffected."""
    from dataclasses import replace as dc_replace
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    # Only S_n_right set (bottom, ETL). Top correction should be zero.
    stack_neumann = dc_replace(stack_base, S_n_right=0.0)
    stack_robin   = dc_replace(stack_base, S_n_right=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    # State: n[-1,:] = 2 × n_eq_right (excess electrons at bottom boundary)
    n0 = float(mat_neumann.n_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_right[0]) * np.ones((g.Ny, g.Nx))
    n0[-1, :] = 2.0 * float(mat_neumann.n_eq_right[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dn_neumann = dydt_n[:Nn].reshape(g.Ny, g.Nx)
    dn_robin   = dydt_r[:Nn].reshape(g.Ny, g.Nx)
    # Bottom row: Robin removes excess electrons → dn[-1,:] decreases
    assert np.all(dn_robin[-1, :] < dn_neumann[-1, :]), (
        "dn[-1,:] should decrease under Robin when n > n_eq at bottom (mapping swap?)"
    )
    # Top row: no S_n_top → correction = 0 → top rows identical
    np.testing.assert_array_equal(
        dn_robin[0, :], dn_neumann[0, :],
        err_msg="dn[0,:] should be unchanged when only S_n_right is set (mapping swap?)"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_robin_dp_top_decreases_with_excess_holes \
       tests/unit/twod/test_solver_2d.py::test_robin_dp_bot_decreases_with_excess_holes \
       tests/unit/twod/test_solver_2d.py::test_robin_correction_routes_to_correct_boundary -v
```

Expected: FAIL — `assemble_rhs_2d` currently always uses the Dirichlet path (no `has_selective_contacts` check yet, and with S_p_left=0 triggering Robin mode but no helper implemented).

Note: some tests may fail with `AttributeError` because `has_selective_contacts` field is already added in Task 2, but `_apply_robin_contacts_2d` does not exist yet. That is expected.

- [ ] **Step 3: Add imports to `solver_2d.py`**

At the top of `perovskite_sim/twod/solver_2d.py`, after the existing imports (after line 11), add:

```python
from perovskite_sim.constants import Q
from perovskite_sim.physics.contacts import selective_contact_flux
```

- [ ] **Step 4: Add `_apply_robin_contacts_2d` to `solver_2d.py`**

Insert the following function just before the `assemble_rhs_2d` function definition (currently at line ~281 after the new imports shift lines):

```python
def _apply_robin_contacts_2d(
    dn: np.ndarray,
    dp: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    mat: "MaterialArrays2D",
) -> tuple[np.ndarray, np.ndarray]:
    """Replace Neumann wall flux with Robin contact flux at y=0 and y=Ny-1.

    ``continuity_rhs_2d`` already included a one-sided Neumann y-divergence
    at the boundary rows (zero wall flux assumed). This helper subtracts the
    implicit zero-flux assumption and adds the actual Robin flux
    J = ±q·S·(density − density_eq) instead. The four Dirichlet pins in
    ``assemble_rhs_2d`` are skipped when this helper is called.

    Outward-normal and side conventions:
      y=0    (top / HTL / 1D-left)   outward normal = −y  →  side="left"
      y=Ny−1 (bot / ETL / 1D-right)  outward normal = +y  →  side="right"

    Sign table (verified from first principles):
      dn[0,  :] −= J_n_top / (Q·hy_top)   electrons top:    dn = +div_n/Q
      dp[0,  :] += J_p_top / (Q·hy_top)   holes top:        dp = −div_p/Q → opposite sign
      dn[−1, :] += J_n_bot / (Q·hy_bot)   electrons bottom: J_n_bot < 0 when n > n_eq
      dp[−1, :] −= J_p_bot / (Q·hy_bot)   holes bottom:     J_p_bot > 0 when p > p_eq
    """
    # Half-cell control-volume thickness at each contact boundary
    hy_top = (mat.grid.y[1]  - mat.grid.y[0])  / 2.0
    hy_bot = (mat.grid.y[-1] - mat.grid.y[-2]) / 2.0

    # --- top contact (y=0, HTL, side="left") --------------------------------
    # selective_contact_flux(carrier="n", side="left") = +Q·S·(n − n_eq)
    # selective_contact_flux(carrier="p", side="left") = −Q·S·(p − p_eq)
    J_n_top = selective_contact_flux(
        n[0, :], mat.n_eq_left, mat.S_n_top, carrier="n", side="left",
    )
    J_p_top = selective_contact_flux(
        p[0, :], mat.p_eq_left, mat.S_p_top, carrier="p", side="left",
    )
    dn[0, :] -= J_n_top / (Q * hy_top)   # subtract: dn = +div_n/Q
    dp[0, :] += J_p_top / (Q * hy_top)   # add:      dp = −div_p/Q (opposite sign)

    # --- bottom contact (y=Ny−1, ETL, side="right") -------------------------
    # selective_contact_flux(carrier="n", side="right") = −Q·S·(n − n_eq)
    # selective_contact_flux(carrier="p", side="right") = +Q·S·(p − p_eq)
    J_n_bot = selective_contact_flux(
        n[-1, :], mat.n_eq_right, mat.S_n_bot, carrier="n", side="right",
    )
    J_p_bot = selective_contact_flux(
        p[-1, :], mat.p_eq_right, mat.S_p_bot, carrier="p", side="right",
    )
    dn[-1, :] += J_n_bot / (Q * hy_bot)   # add:      J_n_bot < 0 when n > n_eq
    dp[-1, :] -= J_p_bot / (Q * hy_bot)   # subtract: J_p_bot > 0 when p > p_eq

    return dn, dp
```

- [ ] **Step 5: Replace the Dirichlet pin block in `assemble_rhs_2d`**

Find the current block at the end of `assemble_rhs_2d` (lines 370–374):
```python
    # --- Dirichlet pin at y=0 and y=Ly (ohmic contacts) -------------------
    dn[0, :] = 0.0
    dn[-1, :] = 0.0
    dp[0, :] = 0.0
    dp[-1, :] = 0.0
```

Replace with:
```python
    # --- Contact boundary conditions ---------------------------------------
    # Dirichlet (ohmic) path: pin all four boundary rows to zero (unchanged
    # from Stage A).  Robin path: apply surface-recombination flux correction
    # at each boundary row; the four pins are skipped entirely.
    if mat.has_selective_contacts:
        dn, dp = _apply_robin_contacts_2d(dn, dp, n, p, mat)
    else:
        dn[0, :] = 0.0
        dn[-1, :] = 0.0
        dp[0, :] = 0.0
        dp[-1, :] = 0.0
```

- [ ] **Step 6: Run the new tests**

```bash
pytest tests/unit/twod/test_solver_2d.py::test_robin_dp_top_decreases_with_excess_holes \
       tests/unit/twod/test_solver_2d.py::test_robin_dp_bot_decreases_with_excess_holes \
       tests/unit/twod/test_solver_2d.py::test_robin_correction_routes_to_correct_boundary -v
```

Expected: PASS.

- [ ] **Step 7: Run full unit suite**

```bash
pytest tests/unit/twod/test_solver_2d.py -v
```

Expected: all tests pass including the Dirichlet backward-compat test from Task 3.

- [ ] **Step 8: Commit**

```bash
git add perovskite_sim/twod/solver_2d.py tests/unit/twod/test_solver_2d.py
git commit -m "feat(twod): add _apply_robin_contacts_2d helper and assemble_rhs_2d Robin guard"
```

---

## Task 5: 1D ↔ 2D Robin parity gate (primary correctness regression)

**Files:**
- Test: `tests/regression/test_twod_validation.py`

This is the primary correctness gate. It runs the 1D Phase 3.3 selective-contact solver and the new 2D Robin solver on the same stack and asserts they agree within the Stage-A parity tolerances. If this fails, the Robin port is wrong.

The preset to use is `configs/selective_contacts_demo.yaml` — a Beer-Lambert nip MAPbI3 stack with all four S values set. It already ships with the repo.

- [ ] **Step 1: Write the failing parity gate test**

Add to `tests/regression/test_twod_validation.py` (the existing helpers `_freeze_ions`, `_maybe_flip_sign`, and the imports `run_jv_sweep`, `compute_metrics`, `run_jv_sweep_2d`, `Microstructure`, `load_device_from_yaml` are already present):

```python
ROBIN_PRESET = "configs/selective_contacts_demo.yaml"


@pytest.mark.regression
@pytest.mark.slow
def test_twod_robin_parity_vs_1d():
    """Stage B(c.1) primary gate: laterally-uniform 2D with Robin contacts matches
    1D Phase 3.3 within sub-mV V_oc / 5×10⁻⁴ J_sc / 10⁻³ FF.

    Uses configs/selective_contacts_demo.yaml (Beer-Lambert nip MAPbI3 with all
    four S values set). Ions frozen on both sides for a clean comparison.
    """
    stack = _freeze_ions(load_device_from_yaml(ROBIN_PRESET))

    # 1D reference with Phase 3.3 Robin contacts active (has_selective_contacts=True)
    r1 = run_jv_sweep(stack, N_grid=31, V_max=1.2, n_points=13, illuminated=True)
    V1 = np.asarray(r1.V_fwd)
    J1 = _maybe_flip_sign(V1, np.asarray(r1.J_fwd))
    m1 = compute_metrics(V1, J1)

    # 2D with Stage B(c.1) Robin contacts, same grid resolution as Stage-A gate
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
        f"\nRobin 1D: V_oc={m1.V_oc*1e3:.3f} mV  J_sc={m1.J_sc:.3f} A/m²  FF={m1.FF:.4f}"
        f"\nRobin 2D: V_oc={m2.V_oc*1e3:.3f} mV  J_sc={m2.J_sc:.3f} A/m²  FF={m2.FF:.4f}"
    )

    assert abs(m2.V_oc - m1.V_oc) <= 1e-3, (
        f"Robin V_oc(2D)={m2.V_oc:.6f} V vs V_oc(1D)={m1.V_oc:.6f} V "
        f"(diff {(m2.V_oc - m1.V_oc)*1e3:.3f} mV, limit 1 mV)"
    )
    rel_jsc = abs(m2.J_sc - m1.J_sc) / abs(m1.J_sc)
    assert rel_jsc <= 5e-4, (
        f"Robin J_sc rel diff {rel_jsc:.2e} > 5e-4 "
        f"(2D={m2.J_sc:.4f}, 1D={m1.J_sc:.4f} A/m²)"
    )
    assert abs(m2.FF - m1.FF) <= 1e-3, (
        f"Robin FF(2D)={m2.FF:.6f} vs FF(1D)={m1.FF:.6f} "
        f"(diff {abs(m2.FF - m1.FF):.4f}, limit 1e-3)"
    )
```

- [ ] **Step 2: Run the test to confirm it fails (Robin not wired in yet)**

```bash
pytest tests/regression/test_twod_validation.py::test_twod_robin_parity_vs_1d -v -s
```

Wait — at this point Tasks 1–4 are already done, so Robin IS wired in. This test should pass. Run it now:

```bash
pytest tests/regression/test_twod_validation.py::test_twod_robin_parity_vs_1d -v -s
```

Expected: PASS with printed V_oc, J_sc, FF values matching within tolerances. If it fails, the Robin boundary math is wrong — re-read the sign table in the spec before debugging.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.1) Robin 1D↔2D parity gate"
```

---

## Task 6: Bounded-shift sanity test + microstructure coexistence smoke test

**Files:**
- Test: `tests/regression/test_twod_validation.py`

- [ ] **Step 1: Write both tests**

Add to `tests/regression/test_twod_validation.py`:

```python
@pytest.mark.regression
@pytest.mark.slow
def test_twod_robin_bounded_shift_vs_dirichlet():
    """Robin contacts (S > 0) must shift V_oc by 1–150 mV vs Dirichlet baseline.

    Compares: (a) nip_MAPbI3_uniform.yaml (Dirichlet) vs
              (b) selective_contacts_demo.yaml (Robin, same geometry).
    This confirms the Robin hook is active and physically non-trivial.
    It is a secondary sanity test — the parity gate (test_twod_robin_parity_vs_1d)
    is the primary correctness criterion.
    """
    stack_dirichlet = _freeze_ions(load_device_from_yaml(PRESET))        # no S values
    stack_robin     = _freeze_ions(load_device_from_yaml(ROBIN_PRESET))  # S values set

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
    r_d = run_jv_sweep_2d(stack=stack_dirichlet, **common_kw)
    r_r = run_jv_sweep_2d(stack=stack_robin,     **common_kw)

    V_d = np.asarray(r_d.V); J_d = _maybe_flip_sign(V_d, np.asarray(r_d.J))
    V_r = np.asarray(r_r.V); J_r = _maybe_flip_sign(V_r, np.asarray(r_r.J))
    m_d = compute_metrics(V_d, J_d)
    m_r = compute_metrics(V_r, J_r)

    shift_mV = abs(m_d.V_oc - m_r.V_oc) * 1e3
    print(
        f"\nDirichlet: V_oc={m_d.V_oc*1e3:.1f} mV"
        f"\nRobin:     V_oc={m_r.V_oc*1e3:.1f} mV"
        f"\n|ΔV_oc| = {shift_mV:.1f} mV"
    )
    assert 1.0 <= shift_mV <= 150.0, (
        f"|ΔV_oc| = {shift_mV:.1f} mV is outside [1, 150] mV "
        f"(Robin hook inactive or unphysical)"
    )


@pytest.mark.regression
def test_twod_robin_microstructure_coexistence_smoke():
    """Robin contacts + grain boundary produce finite, ordered J-V (no NaN/Inf).

    Uses a coarse fast mesh — correctness is covered by the parity gate.
    """
    from perovskite_sim.twod.microstructure import GrainBoundary
    stack = _freeze_ions(load_device_from_yaml(ROBIN_PRESET))
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
    assert np.all(np.isfinite(V)), "Non-finite V in Robin+GB sweep"
    assert np.all(np.isfinite(J)), "Non-finite J in Robin+GB sweep"
    J_sc_sign = _maybe_flip_sign(V, J)[0]
    assert J_sc_sign > 0, "J_sc should be positive under illumination (sign/convergence issue)"
```

- [ ] **Step 2: Run both tests**

```bash
pytest tests/regression/test_twod_validation.py::test_twod_robin_bounded_shift_vs_dirichlet \
       tests/regression/test_twod_validation.py::test_twod_robin_microstructure_coexistence_smoke -v -s
```

Expected:
- Bounded-shift: PASS with `|ΔV_oc|` printed in [1, 150] mV range.
- Coexistence smoke: PASS with finite J-V output.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_twod_validation.py
git commit -m "test(twod): Stage B(c.1) Robin bounded-shift and microstructure coexistence tests"
```

---

## Task 7: Full test suite green + CLAUDE.md update

**Files:**
- Test run: all tests
- Modify: `perovskite_sim/CLAUDE.md`

- [ ] **Step 1: Run the full fast suite**

```bash
cd perovskite-sim
pytest
```

Expected: all passing (including existing unit, integration, and regression tests). Zero failures, zero errors.

- [ ] **Step 2: Run the slow suite**

```bash
pytest -m slow
```

Expected: all slow tests pass, including the existing Stage-A parity gate, TMM baselines, and the new Robin parity gate.

- [ ] **Step 3: Add Stage B(c.1) section to `CLAUDE.md`**

In `perovskite_sim/CLAUDE.md`, find the **"2D Microstructure — Stage B"** section (which ends with "Stage-B scope boundaries"). After that paragraph, add:

```markdown
**2D Robin/selective contacts — Stage B(c.1) (Phase 6 — Apr 2026).** Ports the 1D Phase 3.3 Robin/selective-contact boundary condition to `assemble_rhs_2d`. When `SimulationMode.use_selective_contacts` is True and `DeviceStack` supplies any of `S_n_left`, `S_p_left`, `S_n_right`, `S_p_right`, `build_material_arrays_2d` sets `MaterialArrays2D.has_selective_contacts = True` and maps the four scalar S values via the 1D-transport-axis bridge:

```
DeviceStack.S_n_left  → MaterialArrays2D.S_n_top  (y=0,    HTL / p-contact)
DeviceStack.S_p_left  → MaterialArrays2D.S_p_top
DeviceStack.S_n_right → MaterialArrays2D.S_n_bot  (y=Ny-1, ETL / n-contact)
DeviceStack.S_p_right → MaterialArrays2D.S_p_bot
```

`continuity_rhs_2d` computes a one-sided Neumann y-divergence at the boundary rows (previously skipped). `assemble_rhs_2d` guards on `has_selective_contacts`: the `else` branch still zeros the four Dirichlet rows (Stage A/B(a) behaviour unchanged); the `if` branch calls `_apply_robin_contacts_2d` which subtracts the Robin wall flux from the Neumann base. The sign table (verified from first principles; `dp` signs are opposite to `dn` because `dp = −div_p/Q`):

```
dn[0,  :] -= J_n_top / (Q * hy_top)   # top electrons: J_n_top = +q·S·(n−n_eq)
dp[0,  :] += J_p_top / (Q * hy_top)   # top holes:     J_p_top = −q·S·(p−p_eq)  ← +=
dn[-1, :] += J_n_bot / (Q * hy_bot)   # bot electrons: J_n_bot = −q·S·(n−n_eq)
dp[-1, :] -= J_p_bot / (Q * hy_bot)   # bot holes:     J_p_bot = +q·S·(p−p_eq)  ← -=
```

Equilibrium densities use `mat.n_eq_left` / `mat.p_eq_left` for y=0 and `mat.n_eq_right` / `mat.p_eq_right` for y=Ny-1 (existing `(Nx,)` arrays; note the code comments on these fields had "top/bottom" reversed — fixed in this stage). The parity gate `tests/regression/test_twod_validation.py::test_twod_robin_parity_vs_1d` runs 1D Phase 3.3 vs 2D Stage B(c.1) on `configs/selective_contacts_demo.yaml` (frozen ions) and pins |ΔV_oc| ≤ 1 mV. Stage B(c.1) does not add Schottky barriers, per-column S patterning, or changes to the backend/frontend.
```

- [ ] **Step 4: Commit**

```bash
git add perovskite_sim/CLAUDE.md
git commit -m "docs(twod): Stage B(c.1) Robin contacts section in CLAUDE.md"
```

- [ ] **Step 5: Push to origin**

```bash
git push origin 2d-extension
```

---

## Self-review checklist

**Spec coverage:**
- §1 Scope — no Schottky, no patterning, no μ(E)/reabsorption: enforced by not touching those code paths ✓
- §2 Architecture — helper in solver_2d.py, no new files: Task 4 ✓
- §3 Contact convention — comment fix Task 1, mapping Task 2, `n_eq_left/right` used: Tasks 1–4 ✓
- §4 Continuity RHS change — Task 3 ✓; Dirichlet backward-compat preserved ✓
- §5 Sign conventions — sign table reproduced in Task 4 code and CLAUDE.md Task 7 ✓; wrong signs called out in comments ✓
- §6 Tests — T1 Task 3, T2 Task 2+4, T3 Task 4, T4 Task 5, T5 Task 6, T6 Task 6 ✓
- §7 Risk register — dp sign: T3 catches; mapping swap: T2 catches; Dirichlet backcompat: T1 catches ✓

**Placeholder scan:** No TBD, no "similar to", no "add appropriate" — all steps have full code. ✓

**Type consistency:** `selective_contact_flux(density, density_eq, S, *, carrier, side)` signature used consistently across Tasks 4, 5, 6. `MaterialArrays2D` fields `S_n_top`, `S_p_top`, `S_n_bot`, `S_p_bot` defined in Task 2 and used in Task 4. `has_selective_contacts` defined in Task 2, read in Task 4. ✓

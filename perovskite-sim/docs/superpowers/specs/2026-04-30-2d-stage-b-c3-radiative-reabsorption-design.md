# Stage B(c.3) — Self-consistent radiative reabsorption in 2D — design

**Status:** approved formulation, awaiting plan.

**Goal:** Port the 1D Phase 3.1b self-consistent radiative reabsorption hook to the 2D solver. On every RHS call, integrate `R_tot_2D = ∬ B(y,x)·n(y,x)·p(y,x) dy dx` over each absorber's 2D cells and feed back the non-escaping fraction `G_rad = R_tot_2D · (1 − P_esc) / area` uniformly across the absorber. Preserve bit-identical Stage A / B(a) / B(c.1) / B(c.2) behaviour when the hook is disabled.

**Out of scope:** optical-profile-weighted redistribution (deferred follow-up), per-grain absorber heterogeneity in the reabsorption integral, μ(T)-coupled `B_rad(T)` beyond the existing temperature-scaling hook, any backend or frontend changes.

---

## 1. Approved formulation

Stage B(c.3) implements **uniform-over-absorber-area redistribution** (the natural 2D extension of 1D's "uniform over absorber thickness"):

```
For each absorber layer:
  R_tot_2D = ∬ B(y,x) · n(y,x) · p(y,x) dy dx     over absorber rows × all x
  area     = thickness × lateral_length            (precomputed at build)
  G_rad    = R_tot_2D · (1 − P_esc) / area         (uniform over absorber 2D area)
  G[absorber_y_range, :] += G_rad
```

This is **bit-equivalent to 1D in the lateral-uniform limit**: with constant `dx`, the lateral trapezoid evaluates to `lateral_length × emission_y(y)`, and dividing by `area = thickness × lateral_length` recovers `emission_y / thickness` — exactly 1D's `G_rad = R_tot · (1 − P_esc) / thickness`.

**Optical-profile-weighted redistribution is explicitly deferred.** Under TMM optics with strong absorption gradients, redistributing `G_rad` weighted by `α(x) · I(x)` (rather than uniform) is more physically accurate. It is documented as a possible future extension but is out of scope for Stage B(c.3) v1. If a future regression on a strongly non-uniform G(x) profile flags physically suspect reabsorption behaviour near absorber edges, optical-profile redistribution can be added as a separate follow-up with its own tests.

**Approval rationale (from brainstorm):**
1. Smallest correct 2D port of 1D Phase 3.1b.
2. Bit-equivalent to 1D under lateral-uniform parity.
3. Matches the established Stage B(c.2) "smallest correct port → defer fancy variant" pattern.
4. Avoids cross-axis coupling between the optical profile and the radiative emission integral, which would introduce extra failure modes for the parity gate.

---

## 2. 1D Phase 3.1b reference (load-bearing — verified by code inspection)

**Where it lives:**
- `perovskite_sim/solver/mol.py:597–741` — build path stores `MaterialArrays.absorber_masks: tuple[np.ndarray, ...]`, `absorber_p_esc: tuple[float, ...]`, `absorber_thicknesses: tuple[float, ...]`, `has_radiative_reabsorption: bool`. One tuple entry per absorber-tagged layer with `Eg > 0` under TMM.
- `mol.py:874–895` — per-RHS hook in `assemble_rhs`. Uses `np.trapezoid(B_rad·n·p, x)` over the absorber mask, then `G[mask] += R_tot·(1 − P_esc) / thickness`.
- `experiments/jv_sweep.py:328–378` — **lagged fallback** `_bake_radiative_reabsorption_step` evaluates `R_tot` once at the entry state of a voltage step, folds `G_rad` into a step-local `G_optical` copy, clears `has_radiative_reabsorption` on the returned `mat`. Used only when the standard call fails (TMM diode-injection knee at V≈0.21V — see `project_tmm_jv_regression_021.md`).

**Activation gate (1D, in `mol.py:597+`):** `has_radiative_reabsorption = sim_mode.use_radiative_reabsorption AND sim_mode.use_photon_recycling AND any absorber has Eg > 0 under TMM`. The `use_photon_recycling` AND is critical — without TMM optics there is no `P_esc`.

**Tests (in `tests/regression/test_radiative_reabsorption.py`):**
- `test_radiative_reabsorption_matches_phase_3_1_at_voc` — at V_oc, Phase 3.1b reproduces Phase 3.1 V_oc within 5 mV.
- `test_radiative_reabsorption_preserves_voc_boost` — V_oc boost vs PR-off lies in [40, 100] mV.
- Plus activation-flag and disabled-fallback tests.

**P_esc primitive:** `physics/photon_recycling.compute_p_esc_for_absorber(...)`. Stage B(c.3) does **NOT** modify this module — it is reused unchanged.

---

## 3. Numerical implementation

### 3.1 Compute `R_tot_2D` per absorber from `B_rad`, `n`, `p`

Inside `assemble_rhs_2d`, after the recombination `R` is computed and **before** the `continuity_rhs_2d` call (which currently consumes `mat.G_optical`), when `mat.has_radiative_reabsorption_2d` is True:

```python
emission = mat.B_rad[y_lo:y_hi, :] * n[y_lo:y_hi, :] * p[y_lo:y_hi, :]   # (n_y_abs, Nx)
emission_x = np.trapezoid(emission, g.y[y_lo:y_hi], axis=0)              # (Nx,)
R_tot      = float(np.trapezoid(emission_x, g.x))                        # scalar  [1/(m·s)]
```

Two-axis trapezoid: integrate over y first (axis=0), then over x. Skipped per absorber when `area ≤ 0`, `P_esc ≥ 1`, fewer than 2 absorber rows, fewer than 2 x-nodes, or `R_tot ≤ 0`. Mirrors the 1D safety guards.

### 3.2 Recover `G_rad` and augment `G`

```python
G_rad = R_tot * (1.0 - P_esc) / area                                     # scalar  [1/(m³·s)]
G_with_rad[y_lo:y_hi, :] += G_rad                                        # uniform over absorber rows × all x
```

`G_with_rad = mat.G_optical.copy()` is allocated once at the top of the recompute branch — the cached `mat.G_optical` is never mutated (matches 1D's `G = G.copy()` semantics).

### 3.3 Plumbing — new module + thin call site in `assemble_rhs_2d`

**New module `perovskite_sim/twod/radiative_reabsorption_2d.py`** (mirrors B(c.2) pattern):

```python
def recompute_g_with_rad_2d(
    *,
    G_optical: np.ndarray,                                  # (Ny, Nx)
    n: np.ndarray, p: np.ndarray,                           # (Ny, Nx)
    B_rad: np.ndarray,                                      # (Ny, Nx)
    x: np.ndarray, y: np.ndarray,                           # grid axes
    absorber_y_ranges: tuple[tuple[int, int], ...],
    absorber_p_esc:    tuple[float, ...],
    absorber_areas:    tuple[float, ...],
) -> np.ndarray:                                            # returns (Ny, Nx)
    ...
```

Pure function. No solver dependencies. Returns a NEW `(Ny, Nx)` array equal to `G_optical` augmented per absorber. Includes shape validators that fail early with `ValueError` (mirroring `field_mobility_2d`'s `_check_*_face` style).

**`assemble_rhs_2d` branching:**

```python
if mat.has_radiative_reabsorption_2d:
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

Then pass `G_to_use` to `continuity_rhs_2d` (current code passes `mat.G_optical` directly). The `else` branch is **bit-identical** to the current Stage B(c.2) code: `G_to_use is mat.G_optical` (no copy, no augmentation).

### 3.4 Lagged fallback in `jv_sweep_2d._integrate_step`

Mirror the 1D pattern from `jv_sweep.py:328–378` exactly. Add a 2D analog `_bake_radiative_reabsorption_step_2d(y_state, mat, illuminated, lateral_length)`:

1. Default path: `_integrate_step` calls Radau with the fully self-consistent `mat` (per-RHS `R_tot` recompute on every Newton iteration). Most steps converge.
2. On `RuntimeError` from `run_transient_2d` (Newton-fail / bisection-exhausted) AND `mat.has_radiative_reabsorption_2d` AND `illuminated`:
   - Unpack `y_state` to `(n0, p0)` at the entry state of the failed voltage step.
   - Compute `R_tot_2D` once from `(n0, p0)` per absorber (same formula as the per-RHS hook).
   - Build `G_with_rad = mat.G_optical.copy()`, augment per absorber.
   - Return `dataclasses.replace(mat, G_optical=G_with_rad, has_radiative_reabsorption_2d=False, absorber_y_ranges_2d=(), absorber_p_esc_2d=(), absorber_areas_2d=(), absorber_thicknesses_2d=())` — clears the flag and zeros the tuples so the retry takes the disabled path.
   - Retry `run_transient_2d` once with the baked `mat`. If it still fails, raise.

3. Across voltage steps: the warm-start chain refreshes `R_tot` from the freshly-settled state, so the lag is bounded by how much `n·p` drifts inside one settle interval — sub-percent on the typical `v_rate=1 V/s` sweep, well below the 5 mV V_oc parity window.

The lagged fallback only fires on stiff steps (TMM diode-injection knee). On Beer-Lambert presets the fallback never triggers because BL doesn't have the TMM-knee pathology. The fallback is in place for safety so future TMM 2D runs do not need a separate cleanup.

### 3.5 Disabled-path bit-identicality

When `mat.has_radiative_reabsorption_2d` is False, `assemble_rhs_2d` skips the recompute branch entirely. `G_to_use is mat.G_optical` (same Python object, no copy). The `continuity_rhs_2d` call list is identical to current Stage B(c.2). This is verified by T1 in §5 with `np.testing.assert_array_equal` on full J-V output.

### 3.6 Interaction with B(c.1) Robin and B(c.2) μ(E)

Orthogonal:
- Robin contacts modify `dn`/`dp` at boundary rows (y=0 and y=Ny-1) — different rows than absorber-interior rows.
- μ(E) modifies per-face D inside `continuity_rhs_2d` (via override kwargs) — affects flux, not generation.
- Reabsorption augments `G` on absorber-interior rows — affects generation, not flux or BC.

All three compose in `assemble_rhs_2d` with no coupling. T4 in §5 verifies finite J-V when all three are simultaneously active.

---

## 4. Data, API, and config

### 4.1 New `MaterialArrays2D` fields

Mirror the 1D tuple structure (matching the 1D `MaterialArrays.absorber_*` pattern):

```python
# Stage B(c.3): self-consistent radiative reabsorption.
has_radiative_reabsorption_2d: bool                        = False
absorber_y_ranges_2d:   tuple[tuple[int, int], ...]        = ()   # one (y_lo, y_hi) per absorber
absorber_p_esc_2d:      tuple[float, ...]                  = ()   # Yablonovitch escape probability per absorber
absorber_thicknesses_2d: tuple[float, ...]                 = ()   # physical absorber thickness [m]
absorber_areas_2d:      tuple[float, ...]                  = ()   # thickness × lateral_length [m²]
```

**5 new fields total** (1 flag + 4 parallel tuples). All defaults preserve disabled-path bit-identity. The tuples are empty `()` when the hook is off — no `None`-vs-empty divergence.

**Why per-absorber tuples** (not a single mask): tandem stacks have multiple absorbers with different `P_esc` values. Even though Stage B(c.3) v1 only ships single-absorber presets, the tuple structure makes future tandem 2D extension trivial. Mirrors 1D exactly.

**Why `absorber_y_ranges_2d` (tuple of int pairs) vs `absorber_masks_2d` (tuple of bool arrays):** y-ranges are simpler and equivalent. Each absorber spans a contiguous row range derived from `_layer_role_at_each_y`. Slicing `arr[y_lo:y_hi, :]` is cleaner than `arr[mask_2d]` (which flattens the result and loses 2D structure for the trapezoid). The build path validates that the absorber rows actually form a contiguous range (no gaps) — see T5.

### 4.2 Build path in `build_material_arrays_2d`

After the existing Stage B(c.2) field-mobility population block (and after `layer_role_per_y` is computed), add:

```python
# Stage B(c.3): radiative reabsorption (mirror 1D mol.py:597+ logic).
sim_mode = resolve_mode(getattr(stack, "mode", "full"))   # may be hoisted from B(c.1)/B(c.2)
absorber_y_ranges_list:   list[tuple[int, int]] = []
absorber_p_esc_list:      list[float]            = []
absorber_thicknesses_list: list[float]           = []
absorber_areas_list:      list[float]            = []
_has_rr_2d = False

if sim_mode.use_radiative_reabsorption and sim_mode.use_photon_recycling and mat1d.has_radiative_reabsorption:
    # mat1d already has the per-absorber tuples populated from the 1D build.
    # Translate them to 2D y-range form using layer_role_per_y.
    for mask_1d, p_esc, thickness in zip(
        mat1d.absorber_masks, mat1d.absorber_p_esc, mat1d.absorber_thicknesses
    ):
        # mat1d.absorber_masks are (N,) boolean over the 1D x-grid which is the same as the 2D y-grid.
        y_indices = np.where(mask_1d)[0]
        if y_indices.size < 2: continue
        y_lo, y_hi = int(y_indices[0]), int(y_indices[-1] + 1)   # half-open [lo, hi)
        # Sanity: range must be contiguous (mat1d.absorber_masks are always contiguous in 1D)
        assert np.all(mask_1d[y_lo:y_hi]), "absorber mask non-contiguous"
        if p_esc >= 1.0 or thickness <= 0.0: continue
        absorber_y_ranges_list.append((y_lo, y_hi))
        absorber_p_esc_list.append(float(p_esc))
        absorber_thicknesses_list.append(float(thickness))
        absorber_areas_list.append(float(thickness) * float(grid.lateral_length))
    _has_rr_2d = len(absorber_y_ranges_list) > 0
```

The 2D build leverages the 1D `mat1d.absorber_*` tuples that are already computed — no duplicate "find the absorber" logic. The y-range translation is direct because the 2D y-axis is the 1D x-axis (the transport axis). `lateral_length` is read from the `Grid2D` object.

### 4.3 No changes to other modules

- **`perovskite_sim/physics/photon_recycling.py`** — UNCHANGED. The `compute_p_esc_for_absorber(...)` primitive is reused via the 1D build path that fills `mat1d.absorber_p_esc`.
- **`perovskite_sim/solver/mol.py`** — UNCHANGED. The 1D code is the source of truth for P_esc and absorber-mask construction.
- **`perovskite_sim/models/parameters.py`** — UNCHANGED. No new `MaterialParams` fields.
- **`perovskite_sim/models/config_loader.py`** — UNCHANGED. No new YAML keys.

### 4.4 YAML / preset compatibility

**No YAML schema change.** Activation requires:
1. `device.mode: full` (FULL tier sets `use_radiative_reabsorption=True` and `use_photon_recycling=True`).
2. A TMM-enabled preset (`optical_material` set on the absorber) — needed for `compute_p_esc_for_absorber` to run.
3. An absorber layer with `Eg > 0` (every shipped TMM preset satisfies this).

For Beer-Lambert presets, `mat1d.has_radiative_reabsorption=False` (no TMM → no P_esc), so `_has_rr_2d=False` automatically. Bit-identical to current Stage B(c.2) behaviour.

### 4.5 Tier gating

Mirrors B(c.1) and B(c.2). The activation AND-chain is:
1. `sim_mode.use_radiative_reabsorption` (off in LEGACY and FAST, on in FULL — per `models/mode.py`).
2. `sim_mode.use_photon_recycling` (on in FAST and FULL, off in LEGACY).
3. `mat1d.has_radiative_reabsorption` (only True when 1D side computed P_esc).

LEGACY tier disables the hook (matches 1D). FAST tier currently has `use_radiative_reabsorption=False` (it's a per-RHS hook), so FAST also stays on the disabled path. FULL enables when the preset supports it.

---

## 5. Validation strategy

Seven tests, mirroring the B(c.2) shape:

### T1 — Disabled-path bit-identical (slow regression)
Beer-Lambert preset (`nip_MAPbI3_uniform.yaml`) with `_freeze_ions`, `mode='full'` → `mat.has_radiative_reabsorption_2d=False` (BL has no TMM, no P_esc). Compare the 2D J-V sweep against the same sweep with reabsorption explicitly disabled (mode='legacy'). Assert J-V identical via `np.testing.assert_array_equal(V)` and `np.testing.assert_allclose(J, rtol=1e-12, atol=0.0)`.

### T2 — 1D ↔ 2D parity gate at V_oc (slow regression, primary correctness gate)
**TMM-enabled preset** (`nip_MAPbI3_tmm.yaml`) with `_freeze_ions`, `mode='full'`. Run 1D `run_jv_sweep` and 2D `run_jv_sweep_2d` with matched grids (1D `N_grid=31`, 2D `Ny_per_layer=10` × layers + 1, `Nx=4`, `settle_t=1e-3`, `lateral_bc='periodic'`). Assert:
- `|ΔV_oc| ≤ 5e-3 V` (5 mV — matches 1D `test_radiative_reabsorption_matches_phase_3_1_at_voc`)
- `|ΔJ_sc/J_sc| ≤ 5e-4`
- `|ΔFF| ≤ 1e-3`

**Tolerance discipline:** if measured deltas exceed these bounds, **do NOT loosen blindly**. First report the measured values, diagnose (sign error in trapezoid, wrong axis order, missing `lateral_length` factor, or genuine adaptive-solver noise), and only loosen by pinning at ~3× the measured noise floor. Cap fallback: 10 mV V_oc, 1e-3 J_sc, 2e-3 FF (matches the B(c.2) discipline).

The TMM preset is required because radiative reabsorption only activates with `use_photon_recycling=True`, which itself requires TMM. The known TMM 1D regression at V≈0.21V (`project_tmm_jv_regression_021.md`) is at the diode-injection knee; V_oc≈0.91V is far from there, so the parity assertion at V_oc is unaffected.

### T3 — V_oc boost in [40, 100] mV (slow regression)
Same TMM preset. Compare 2D `mode='full'` (reabsorption on) vs `mode='legacy'` (reabsorption off, photon recycling off). Assert `40 mV ≤ V_oc(on) − V_oc(off) ≤ 100 mV`. Mirrors 1D `test_radiative_reabsorption_preserves_voc_boost`.

### T4 — Coexistence smoke (regression, fast)
Reabsorption + Robin contacts + μ(E) + grain boundary on a coarse mesh. Assert finite V, finite J, J_sc > 0 under illumination. **Smoke test only — no tight physics window.** Cheap test that proves the four per-RHS hooks (reabsorption, Robin, μ(E), GB-tau) compose without solver hang or NaN/Inf.

### T5 — Absorber-mask correctness (unit, fast)
Build `MaterialArrays2D` from a single-absorber TMM preset. Assert:
- `len(mat.absorber_y_ranges_2d) == 1`
- `len(mat.absorber_p_esc_2d) == 1`
- `len(mat.absorber_thicknesses_2d) == 1`
- `len(mat.absorber_areas_2d) == 1`
- `mat.absorber_y_ranges_2d[0]` matches `[j for j, r in enumerate(mat.layer_role_per_y) if r == "absorber"]` (contiguous range; lo = first absorber index, hi = last absorber index + 1)
- `mat.absorber_areas_2d[0] == pytest.approx(thickness * lateral_length)` to fp precision

### T6 — Tier-flag gating (unit, fast)
- `mode='legacy'` with TMM preset → `has_radiative_reabsorption_2d=False`, all four tuples empty.
- `mode='full'` with TMM preset → `has_radiative_reabsorption_2d=True`, tuples populated.
- `mode='fast'` with TMM preset → `has_radiative_reabsorption_2d=False` (FAST excludes per-RHS hooks per CLAUDE.md tier matrix).

Mirrors the B(c.1) Issue I1 reprise pattern.

### T7 — Aggressive-stiffness finite-RHS smoke (unit, fast)
Build a `MaterialArrays2D` with reabsorption on. Construct a non-trivial `(n, p)` state with a steep y-gradient (drives `B·n·p` integrand variation). Call `assemble_rhs_2d` once. Assert `np.all(np.isfinite(dydt))`. Catches a per-RHS integral overflow or mis-shaped trapezoid output.

---

## 6. Risk register

| # | Risk | Mitigation |
|---|---|---|
| **R1** | Non-local RHS coupling → Jacobian stiffness, Newton-fail at TMM diode knee | Port 1D lagged-fallback to `jv_sweep_2d._integrate_step` per §3.4. Activates only on Newton-fail; default path stays self-consistent. |
| **R2** | Wrong absorber mask (e.g. picks up HTL/ETL rows) | T5 explicitly asserts `absorber_y_ranges_2d` matches `layer_role_per_y` indices. Build path uses 1D `mat1d.absorber_masks` (already validated by 1D tests). |
| **R3** | Wrong area/volume weighting (forgot `× lateral_length`, used wrong axis order in trapezoid, divided by thickness instead of area) | T2 1D↔2D parity gate at TMM. In the lateral-uniform limit, missing `lateral_length` would scale `G_rad` by `1/lateral_length` — unmissable on V_oc. T5 asserts `area = thickness × lateral_length` numerically. |
| **R4** | Double-counting `B·n·p` recombination | **No double count.** `B·n·p` stays in recombination `R` (passed to `continuity_rhs_2d`). The reabsorption hook adds the **non-escaping fraction** `(1 − P_esc)·R` BACK as a **G source**. Net effect: an effective `B_rad · P_esc` recombination rate. This is the `B_rad *= P_esc` Phase 3.1 limit in the uniform-n·p case and the rationale Phase 3.1b uses to argue equivalence. T3 V_oc-boost-in-[40, 100] mV would fail if double-counted (V_oc would NOT boost, or would boost by 2× the literature window). |
| **R5** | Incorrect redistribution (non-uniform G_rad applied to absorber rows) | Stage B(c.3) v1 is uniform-over-absorber by design (Section 1). Optical-profile-weighted is deferred. T2 lateral-uniform parity verifies the uniform redistribution matches 1D. |
| **R6** | Breaking disabled-path bit-identicality | T1 explicit slow regression. Structural pattern: when flag is False, `G_to_use is mat.G_optical` (same object), and the `continuity_rhs_2d` call is identical to the current Stage B(c.2) call — verified by visual diff in code review. |
| **R7** | Interaction with B(c.1) Robin or B(c.2) μ(E) | T4 coexistence smoke. Physics is orthogonal: Robin → BC rows, μ(E) → flux faces, reabsorption → absorber-interior G. All three modify disjoint state regions of the RHS. |
| **R8** | 1D↔2D parity drift on TMM preset due to existing 1D TMM regression at V=0.21V (`project_tmm_jv_regression_021.md`) | TMM parity gate (T2) uses `_freeze_ions` and asserts at V_oc only. V_oc ≈ 0.91 V is far from the 0.21 V diode-injection knee where the 1D regression bites. The 0.21 V issue does NOT affect V_oc parity. If T2 fails near V_oc, the cause is in the reabsorption math, not the pre-existing TMM regression. |

---

## 7. Carry-forward cleanup (tracked, NOT in B(c.3) milestone commits)

These items are visible in the B(c.2) and B(c.1) final reviews and remain candidates for a dedicated cleanup commit AFTER B(c.3) lands. The B(c.3) milestone commits should NOT mix these in (per the established pattern of cleanup PR `cleanup/stage-bc1-robin-followups`):

**From B(c.2) final review:**
- **B(c.2) S1**: hoist the duplicated `resolve_mode` import in `solver_2d.py` to a single top-of-file import. B(c.3) will add a third site that benefits from the same hoisted reference; the hoist is a natural consolidation point.
- **B(c.2) S2**: promote the duplicated `1e-300` harmonic-mean epsilon (currently in `flux_2d.py`, `continuity_2d.py`, `field_mobility_2d.py`) into a single shared constant.

**From B(c.1) final review:**
- **B(c.1) S3**: `_apply_robin_contacts_2d` per-RHS `dn/dp.copy()` — the immutability copy is defensive but adds one allocation per RHS. Trade-off documented in B(c.1); defer.
- **B(c.1) S2**: hoist mid-file imports in `solver_2d.py` to top — partially addressed in B(c.2); can complete in the cleanup pass.

**Recommendation:** ship a `cleanup/stage-bc-followups` branch AFTER Stage B(c.3) merges to `2d-extension`, modeled after the Stage B(c.1) cleanup PR. That cleanup PR consolidates all four items into separate scoped commits (S1, S2, S3, S2-completion) so each can be reviewed independently.

---

## 8. Implementation task ordering (high-level — the planning step will expand)

Tentative arc, ~7 commits, similar shape to Stage B(c.1) and B(c.2):

1. **T1** — Create `perovskite_sim/twod/radiative_reabsorption_2d.py` with the pure `recompute_g_with_rad_2d` helper + shape validators + ~7 unit tests (shape correctness, identity at zero P_esc, redistribution sums to expected total, area-weighting check, lateral-uniform reduces to 1D analytic, sign correctness, shape-mismatch ValueError).
2. **T2** — Add 5 new fields to `MaterialArrays2D` (`has_radiative_reabsorption_2d`, 4 tuples). Populate via build path translating 1D `mat1d.absorber_masks` to 2D y-ranges. Add T5 absorber-mask correctness test + T6 tier-gate test.
3. **T3** — Wire `assemble_rhs_2d` to call `recompute_g_with_rad_2d` when flag is True; pass augmented `G_to_use` to `continuity_rhs_2d`. Add T7 finite-RHS smoke test.
4. **T4** — Lagged fallback in `jv_sweep_2d._integrate_step`: add `_bake_radiative_reabsorption_step_2d` helper + Newton-fail retry path. Unit test: simulate Newton-fail (mock or pathological state) and verify the retry takes the disabled-flag path.
5. **T5** — T1 disabled-path bit-identical regression (slow) on BL preset.
6. **T6** — T2 1D↔2D TMM parity gate at V_oc + T3 V_oc-boost-in-[40, 100] mV regression (both slow).
7. **T7** — T4 coexistence smoke + `CLAUDE.md` Stage B(c.3) section + full fast/slow suite green + push.

---

## 9. Open decisions / unresolved

None remaining at the design level. The brainstorm resolved:
- **Formulation**: uniform-over-absorber-area (Option A); optical-profile redistribution deferred.
- **Parity preset**: TMM (`nip_MAPbI3_tmm.yaml`) — required because reabsorption is gated on `use_photon_recycling`.
- **Tolerances**: T2 = (5 mV V_oc, 5e-4 J_sc, 1e-3 FF) initial; do not loosen blindly. T3 = [40, 100] mV V_oc boost.
- **Lagged fallback**: port 1D pattern verbatim to `jv_sweep_2d._integrate_step`; activates on Newton-fail only.
- **Module placement**: new pure module `perovskite_sim/twod/radiative_reabsorption_2d.py`, mirroring the B(c.2) pattern.
- **API extension**: 5 new `MaterialArrays2D` fields (1 flag + 4 parallel tuples), no `MaterialParams`/YAML changes.
- **Tier gating**: `use_radiative_reabsorption AND use_photon_recycling AND mat1d.has_radiative_reabsorption` (mirror 1D mol.py).
- **Coexistence**: T4 smoke verifies finiteness with all four hooks active (reabsorption + Robin + μ(E) + GB-tau).

The implementation plan (next step, via `superpowers:writing-plans`) should produce a 7-task TDD-style plan with full code per step, mirroring the structure of the B(c.1) and B(c.2) plans.

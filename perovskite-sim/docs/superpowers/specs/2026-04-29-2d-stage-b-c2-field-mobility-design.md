# Stage B(c.2) — Field-dependent mobility μ(E) in 2D — design

**Status:** approved formulation, awaiting plan.

**Goal:** Port the 1D Phase 3.2 field-dependent mobility hook (`apply_field_mobility` over Caughey–Thomas + Poole–Frenkel) to the 2D solver, using a **face-normal** formulation: each face flux uses only the field component normal to that face. Preserve bit-identical Stage A / B(a) / B(c.1) behaviour when μ(E) is disabled.

**Out of scope:** total-|E| mobility (Option B), per-grain mobility patterning, μ(T) coupling beyond the existing temperature-scaling hook, any backend or frontend changes.

---

## 1. Approved formulation — face-normal μ(E)

Stage B(c.2) implements **Option A — face-normal μ(E)**:

- x-faces use only `|E_x_face|` to evaluate `μ(E)`.
- y-faces use only `|E_y_face|` to evaluate `μ(E)`.
- `μ_n` and `μ_p` are recomputed per face on every RHS call using the existing 1D `apply_field_mobility(mu0, |E|, v_sat, beta, gamma_pf)` primitive, unchanged.
- Effective per-face diffusion is recovered via the Einstein relation `D_eff = μ_eff · V_T`.
- The disabled path (when `has_field_mobility=False`) is **bit-identical** to the current constant-mobility code path.

Approval rationale (from brainstorm):
1. Smallest correct 2D port of 1D Phase 3.2.
2. Bit-equivalent to 1D in the laterally-uniform parity case (because `E_x ≈ 0` everywhere).
3. Matches the Scharfetter–Gummel flux decomposition, in which each face flux is assembled from the field component normal to that face.
4. Avoids interpolation of orthogonal field components at boundary faces, which would introduce extra numerical and indexing risk.
5. Mirrors the successful B(c.1) Robin-port pattern (small surface area, narrow scope, full parity gate).

### Option B — total |E| — deferred extension

Option B (total-|E| with cross-axis interpolation of `E_x` to y-faces and `E_y` to x-faces) is **explicitly out of scope** for Stage B(c.2).

It may be reconsidered later for strongly non-uniform 2D fields — in particular, μ(E) interacting with microstructure / grain-boundary configurations where the lateral component `E_x` is comparable to `E_y`. If future validation shows that face-normal μ(E) produces physically suspect lateral-field behaviour, total-|E| mobility can be added as a separate follow-up with its own implementation, parity tests, and rollout plan. The deferral is recorded so a future reviewer does not mistake the simpler choice for an oversight.

---

## 2. Numerical implementation

### 2.1 Compute `E_x` on x-faces and `E_y` on y-faces from `phi`

Inside `assemble_rhs_2d`, after the Poisson solve, when `mat.has_field_mobility` is True:

```python
dx = np.diff(g.x)                                          # (Nx-1,)
dy = np.diff(g.y)                                          # (Ny-1,)
E_x_face = -(phi[:,  1:] - phi[:, :-1]) / dx[None, :]      # (Ny,   Nx-1)   — interior x-faces
E_y_face = -(phi[1:, :] - phi[:-1, :]) / dy[:,  None]      # (Ny-1, Nx)     — interior y-faces
```

Sign convention: `E = −∂φ/∂x` (matching 1D). The `apply_field_mobility` primitive calls `np.abs(E)` internally, so the sign of `E_x_face` / `E_y_face` is irrelevant for the μ recompute.

For the periodic-x wrap face (only when `lateral_bc == "periodic"`):

```python
dx_wrap  = 0.5 * (dx[0] + dx[-1])
E_x_wrap = -(phi[:, 0] - phi[:, -1]) / dx_wrap             # (Ny,)
```

### 2.2 Construct μ₀ and `D_eff` per face

The cached per-node diffusion `mat.D_n` / `mat.D_p` (shape `(Ny, Nx)`) is harmonic-mean-averaged to faces, matching the existing logic inside `sg_fluxes_2d_*`:

```python
def _harmonic_face_x(D_node):                              # (Ny, Nx) → (Ny, Nx-1)
    _eps = 1e-300
    return 2.0 * D_node[:, :-1] * D_node[:, 1:] / (D_node[:, :-1] + D_node[:, 1:] + _eps)

def _harmonic_face_y(D_node):                              # (Ny, Nx) → (Ny-1, Nx)
    _eps = 1e-300
    return 2.0 * D_node[:-1, :] * D_node[1:, :] / (D_node[:-1, :] + D_node[1:, :] + _eps)

def _harmonic_face_wrap(D_node):                           # (Ny, Nx) → (Ny,)
    _eps = 1e-300
    return 2.0 * D_node[:, -1] * D_node[:, 0] / (D_node[:, -1] + D_node[:, 0] + _eps)
```

Base mobility at faces (Einstein relation):

```python
mu_n_x_face_base = _harmonic_face_x(mat.D_n) / mat.V_T
mu_n_y_face_base = _harmonic_face_y(mat.D_n) / mat.V_T
mu_p_x_face_base = _harmonic_face_x(mat.D_p) / mat.V_T
mu_p_y_face_base = _harmonic_face_y(mat.D_p) / mat.V_T
```

Apply the 1D primitive per axis, per carrier:

```python
mu_n_x_face_eff = apply_field_mobility(
    mu_n_x_face_base, np.abs(E_x_face),
    mat.v_sat_n_x_face, mat.ct_beta_n_x_face, mat.pf_gamma_n_x_face,
)
mu_n_y_face_eff = apply_field_mobility(
    mu_n_y_face_base, np.abs(E_y_face),
    mat.v_sat_n_y_face, mat.ct_beta_n_y_face, mat.pf_gamma_n_y_face,
)
# (mu_p analogous)
```

Reconstruct effective diffusion via Einstein:

```python
D_n_x_face_eff = mu_n_x_face_eff * mat.V_T
D_n_y_face_eff = mu_n_y_face_eff * mat.V_T
D_p_x_face_eff = mu_p_x_face_eff * mat.V_T
D_p_y_face_eff = mu_p_y_face_eff * mat.V_T
```

For the periodic-x wrap face:

```python
mu_n_wrap_base = _harmonic_face_wrap(mat.D_n) / mat.V_T
mu_n_wrap_eff  = apply_field_mobility(
    mu_n_wrap_base, np.abs(E_x_wrap),
    mat.v_sat_n_wrap, mat.ct_beta_n_wrap, mat.pf_gamma_n_wrap,
)
D_n_wrap_eff = mu_n_wrap_eff * mat.V_T
# (mu_p_wrap analogous)
```

In a laterally-uniform device `E_x_wrap ≈ 0`, so `μ_eff = μ_base` and the wrap path collapses cleanly to the constant-D limit.

### 2.3 Plumbing — extend `sg_fluxes_2d_*` and `continuity_rhs_2d` with optional per-face D overrides

This is the smallest change that preserves the constant-D path bit-identically. Add keyword-only override arguments to `sg_fluxes_2d_n` and `sg_fluxes_2d_p`:

```python
def sg_fluxes_2d_n(
    phi_n, n, x, y, D_n, V_T,
    *,
    D_n_x_face: np.ndarray | None = None,    # (Ny, Nx-1)  — Stage B(c.2) override
    D_n_y_face: np.ndarray | None = None,    # (Ny-1, Nx)  — Stage B(c.2) override
):
    ...
    # If D_n_x_face is provided, use it directly; else fall back to harmonic mean.
    D_face_x = D_n_x_face if D_n_x_face is not None else (
        2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)
    )
    # (D_face_y: same pattern)
```

Same kwargs added to `sg_fluxes_2d_p`. `continuity_rhs_2d` gets `D_n_x_face`, `D_n_y_face`, `D_p_x_face`, `D_p_y_face`, `D_n_wrap`, `D_p_wrap` — all default `None`, all forwarded through.

When all overrides are `None`, the existing harmonic-mean path executes unchanged → bit-identical to Stage A / B(a) / B(c.1).

### 2.4 Periodic x-boundary face D handling

The wrap face logic lives in `continuity_rhs_2d`, not `sg_fluxes_2d_*`. Stage B(c.2) extends the wrap block analogously: when `D_n_wrap` is provided, use it directly; else fall back to the existing harmonic-mean wrap.

```python
if lateral_bc == "periodic":
    if D_n_wrap is not None:
        D_face_wrap_n = D_n_wrap                  # (Ny,)
    else:
        D_face_wrap_n = 2.0 * D_n[:, -1] * D_n[:, 0] / (D_n[:, -1] + D_n[:, 0] + _eps_face)
    # ... (same for D_face_wrap_p)
```

### 2.5 Constant-D path remains bit-identical when μ(E) is disabled

`assemble_rhs_2d` branches on `mat.has_field_mobility`:

```python
if mat.has_field_mobility:
    # Compute E_x_face, E_y_face, (E_x_wrap if periodic), build D_*_face_eff, D_*_wrap_eff
    dn, dp = continuity_rhs_2d(
        g.x, g.y, phi, n, p,
        mat.G_optical, R,
        mat.D_n, mat.D_p,
        mat.V_T,
        chi=chi_2d, Eg=Eg_2d,
        lateral_bc=mat.lateral_bc,
        D_n_x_face=D_n_x_face_eff, D_n_y_face=D_n_y_face_eff,
        D_p_x_face=D_p_x_face_eff, D_p_y_face=D_p_y_face_eff,
        D_n_wrap=D_n_wrap_eff, D_p_wrap=D_p_wrap_eff,
    )
else:
    # Bit-identical to current call site:
    dn, dp = continuity_rhs_2d(
        g.x, g.y, phi, n, p,
        mat.G_optical, R,
        mat.D_n, mat.D_p,
        mat.V_T,
        chi=chi_2d, Eg=Eg_2d,
        lateral_bc=mat.lateral_bc,
    )
```

The `else` branch is identical (same args, same order) to the current call site, so when `has_field_mobility=False` the entire μ(E) machinery is bypassed.

---

## 3. Data, API, and config

### 3.1 New `MaterialArrays2D` fields

```python
# Field-dependent mobility (Stage B(c.2)). All defaults are None when
# has_field_mobility=False, so the constant-mobility cache is unchanged.
has_field_mobility: bool = False

# x-face arrays — (Ny, Nx-1). Used at interior x-faces in sg_fluxes_2d_*.
v_sat_n_x_face:    np.ndarray | None = None
v_sat_p_x_face:    np.ndarray | None = None
ct_beta_n_x_face:  np.ndarray | None = None
ct_beta_p_x_face:  np.ndarray | None = None
pf_gamma_n_x_face: np.ndarray | None = None
pf_gamma_p_x_face: np.ndarray | None = None

# y-face arrays — (Ny-1, Nx). Used at interior y-faces.
v_sat_n_y_face:    np.ndarray | None = None
v_sat_p_y_face:    np.ndarray | None = None
ct_beta_n_y_face:  np.ndarray | None = None
ct_beta_p_y_face:  np.ndarray | None = None
pf_gamma_n_y_face: np.ndarray | None = None
pf_gamma_p_y_face: np.ndarray | None = None

# Periodic wrap-face — (Ny,). Set only when lateral_bc=="periodic".
v_sat_n_wrap:    np.ndarray | None = None
v_sat_p_wrap:    np.ndarray | None = None
ct_beta_n_wrap:  np.ndarray | None = None
ct_beta_p_wrap:  np.ndarray | None = None
pf_gamma_n_wrap: np.ndarray | None = None
pf_gamma_p_wrap: np.ndarray | None = None
```

**Why per-face storage rather than per-node:** μ(E) lives on faces because E lives on faces. Pre-computing the face arrays at build time is cheaper than recomputing in the hot path, and it sidesteps the harmonic-vs-arithmetic-mean question for `v_sat` / `β` / `γ_PF` (which would otherwise need a separate convention from `D_n`).

**Why full `(Ny, Nx-1)` / `(Ny-1, Nx)` shapes rather than `(Ny,)` broadcast:** the params are layer-uniform today (constant in x), but Stage B(a) microstructure already established the precedent of per-node `(Ny, Nx)` arrays for `τ`, and a future per-grain mobility extension would paint x-variation. Storing the face arrays at full shape now means no API churn later.

**Mean choice:** Use **arithmetic mean** for `v_sat`, `β`, `γ_PF` (in contrast to **harmonic mean** for `D_n`/`D_p`). Reasoning: harmonic mean penalises a one-side-zero, which would zero `v_sat` at a heterointerface and silently disable CT there. Arithmetic mean preserves the "active on either side" semantics the helper expects (and `caughey_thomas` itself already short-circuits when `v_sat=0`, so an arithmetic-meaned face inherits the disable correctly).

### 3.2 Build path in `build_material_arrays_2d`

Mirror the existing per-node construction (used for `D_n`, `D_p`, etc.), then average to faces:

```python
# Per-node arrays for field-mobility params, populated by layer mask.
v_sat_n_node    = np.zeros((Ny, Nx))
v_sat_p_node    = np.zeros((Ny, Nx))
ct_beta_n_node  = np.zeros((Ny, Nx))
ct_beta_p_node  = np.zeros((Ny, Nx))
pf_gamma_n_node = np.zeros((Ny, Nx))
pf_gamma_p_node = np.zeros((Ny, Nx))

for layer in elec_layers:
    mask_y = ...                                   # already used for D_n
    p = layer.params
    v_sat_n_node[mask_y, :]    = p.v_sat_n
    v_sat_p_node[mask_y, :]    = p.v_sat_p
    ct_beta_n_node[mask_y, :]  = p.ct_beta_n
    ct_beta_p_node[mask_y, :]  = p.ct_beta_p
    pf_gamma_n_node[mask_y, :] = p.pf_gamma_n
    pf_gamma_p_node[mask_y, :] = p.pf_gamma_p

# Tier-as-ceiling activation gate (mirrors B(c.1) Robin gate).
sim_mode = resolve_mode(getattr(stack, "mode", "full"))
_has_field_mobility = bool(
    sim_mode.use_field_dependent_mobility
    and (
        np.any(v_sat_n_node    > 0.0) or np.any(v_sat_p_node    > 0.0)
        or np.any(pf_gamma_n_node > 0.0) or np.any(pf_gamma_p_node > 0.0)
    )
)

if _has_field_mobility:
    # Average per-node → per-face: arithmetic for v_sat / beta / gamma_pf.
    def _arith_face_x(A): return 0.5 * (A[:, :-1] + A[:, 1:])
    def _arith_face_y(A): return 0.5 * (A[:-1, :] + A[1:, :])
    def _arith_face_wrap(A): return 0.5 * (A[:, -1] + A[:, 0])

    v_sat_n_x_face    = _arith_face_x(v_sat_n_node)
    v_sat_n_y_face    = _arith_face_y(v_sat_n_node)
    ct_beta_n_x_face  = _arith_face_x(ct_beta_n_node)
    ct_beta_n_y_face  = _arith_face_y(ct_beta_n_node)
    pf_gamma_n_x_face = _arith_face_x(pf_gamma_n_node)
    pf_gamma_n_y_face = _arith_face_y(pf_gamma_n_node)
    # (same for p)

    if lateral_bc == "periodic":
        v_sat_n_wrap    = _arith_face_wrap(v_sat_n_node)
        ct_beta_n_wrap  = _arith_face_wrap(ct_beta_n_node)
        pf_gamma_n_wrap = _arith_face_wrap(pf_gamma_n_node)
        # (same for p)
else:
    # All eighteen face arrays stay None (the dataclass default).
    pass
```

`D_n` / `D_p` continue to use **harmonic** mean inside `sg_fluxes_2d_*` — the constant-D physics is unchanged.

### 3.3 Mapping from existing `MaterialParams`

`MaterialParams` already carries the six 1D fields with sensible defaults:

```python
v_sat_n: float = 0.0       # 0 → CT disabled at this layer
v_sat_p: float = 0.0
ct_beta_n: float = 2.0     # silicon Canali for electrons
ct_beta_p: float = 1.0     # silicon Thornber for holes
pf_gamma_n: float = 0.0    # 0 → PF disabled at this layer
pf_gamma_p: float = 0.0
```

Stage B(c.2) reuses these unchanged — **no new fields on `MaterialParams` or `DeviceStack`**.

### 3.4 YAML schema and preset compatibility

**No YAML schema change.** All shipped presets leave `v_sat_{n,p}=pf_gamma_{n,p}=0`, so `_has_field_mobility=False` for every existing preset → bit-identical Stage A / B(a) / B(c.1) behaviour.

The 1D test pattern of opting-in via `dataclasses.replace(stack, ..., v_sat_n=1e2, v_sat_p=1e2)` works in 2D the moment Stage B(c.2) lands.

### 3.5 Tier gating

Mirror B(c.1):
- `LEGACY` tier disables the hook even when params are set (tier-as-ceiling).
- `FAST` tier currently lists μ(E) as one of the three per-RHS hooks it skips (per CLAUDE.md), so the gate keeps FAST on the constant-mobility path.
- `FULL` enables the hook when params are present.

The gating logic is `_has_field_mobility = sim_mode.use_field_dependent_mobility AND any(v_sat>0 or pf_gamma>0)`.

---

## 4. Validation strategy

Five tests, mirroring the B(c.1) shape:

### T1 — Disabled μ(E) reproduces current 2D behaviour (unit, fast)
- Every shipped preset → `mat.has_field_mobility is False` and all eighteen face arrays are `None`.
- A backward-compat regression on `nip_MAPbI3_uniform.yaml`: 2D J-V matches the current frozen-ion snapshot bit-identically (V_oc=910.634 mV, J_sc=424.036 A/m², FF=0.8247).

### T2 — 1D ↔ 2D parity at non-zero `v_sat` (regression, slow) — **primary correctness gate**
- Lateral-uniform device, `_freeze_ions`, then `replace(stack, v_sat_n=1e2, v_sat_p=1e2)` (same aggressive value as the 1D `test_field_mobility_changes_jv_curve`).
- Run `run_jv_sweep` (1D Phase 3.2) and `run_jv_sweep_2d` (Stage B(c.2)) on the same stack with matched grids (1D `N_grid=31`, 2D `Ny_per_layer=10`, `Nx=4`, `settle_t=1e-3`).
- Assert `|ΔV_oc| ≤ 1e-3 V`, `|ΔJ_sc/J_sc| ≤ 5e-4`, `|ΔFF| ≤ 1e-3` (same envelope as the B(c.1) Robin parity gates).
- Expectation: bit-identical or sub-microvolt deltas, because in lateral-uniform devices `E_x ≈ 0` and only `|E_y_face|` contributes — i.e., 2D face-normal μ(E) reduces to the 1D path.

### T3 — Aggressive bounded-shift sanity (regression, slow)
- `replace(stack, v_sat_n=1e2, v_sat_p=1e2)` versus baseline (`v_sat=0`) on the same lateral-uniform 2D device.
- Assert `np.max(|J_on - J_off| / |J_off|) > 1e-3`.
- Confirms μ(E) materially perturbs the J-V curve when the params are set — i.e., the hook is actually being applied, not silently bypassed.

### T4 — μ(E) + Robin + microstructure coexistence smoke (regression, fast)
- Compose all three per-RHS hooks: aggressive Robin (`S_n_left=1e-4, S_p_left=1e-3, S_n_right=1e-3, S_p_right=1e-4`) + grain boundary in absorber + `v_sat=1e3` (modest).
- Coarse mesh (`Nx=6`, `Ny_per_layer=5`, `V_step=0.25`).
- Assert finite V, finite J, J_sc > 0 under illumination.
- Cheap and proves the three hooks compose without NaN/Inf or solver hang.

### T5 — Tier-flag gating (unit, fast)
- `replace(stack, mode="legacy", v_sat_n=1e2, v_sat_p=1e2)` → `has_field_mobility is False`.
- `replace(stack, mode="full",   v_sat_n=1e2, v_sat_p=1e2)` → `has_field_mobility is True`.
- Same shape as the B(c.1) `test_legacy_mode_disables_selective_contacts_in_2d` test that was added during the I1 fix — pre-empts the same class of bug.

### T6 — Periodic-wrap finite-RHS (unit, fast)
- Tiny grid (Nx=4, Ny=11), `lateral_bc="periodic"`, `v_sat=1e2`.
- Call `assemble_rhs_2d` once at a non-trivial state.
- Assert `np.all(np.isfinite(dydt))`.
- Catches a wrong wrap-face shape or a missing wrap kwarg in the override path.

---

## 5. Risk register

| # | Risk | Detection | Mitigation |
|---|---|---|---|
| **R1** | Wrong E-face sign or magnitude. | `apply_field_mobility` calls `np.abs(E)` so a sign flip is invisible to the helper, but a wrong **magnitude** (e.g. forgetting `dy[:, None]` broadcast) surfaces immediately as a 1D parity divergence. | T2 parity gate at non-zero `v_sat`. Sign-trace docstring on the recompute block, mirroring the B(c.1) sign-table comment style. |
| **R2** | x-face / y-face shape mismatch (swapped axes, `(Nx-1, Ny)` vs `(Ny, Nx-1)`). | A swap would either NaN inside `apply_field_mobility` (broadcast failure) or pass through and yield wrong fluxes silently. | T1 backward-compat (currently-passing tests would break) and T2 parity (sub-mV). Hard `assert E_x_face.shape == (Ny, Nx-1)` etc. inside the recompute block as the first defence. |
| **R3** | Periodic wrap-face handling missed. | The wrap-face logic lives in `continuity_rhs_2d`, not `sg_fluxes_2d_*`, and its harmonic-mean must be replaced when μ(E) is on. Easy to miss because `sg_fluxes_2d_*` only handles interior faces. | T6 explicit periodic-wrap unit test. T2 1D parity is on a `lateral_bc="periodic"` device, so a bug in the wrap path also surfaces there. |
| **R4** | Einstein-relation `D = μ V_T` mistakes (e.g. `V_T_300` on one side and `V_T_device` on the other). | The recompute block does `μ_base = D / V_T` then `D_eff = μ_eff · V_T`. If the two `V_T` values differ, the `μ_eff = μ_base` limit (CT disabled) would not reproduce the cached `D` — i.e. T1 bit-identical backward-compat would fail. | Use `mat.V_T` for both the divide and the multiply — single source of truth on the cache. T1 catches a mismatch. |
| **R5** | Radau stiffness from aggressive `v_sat`. | `v_sat=1e2` makes μ steeply state-dependent, which can stretch Newton iterations inside Radau. 1D Phase 3.2 already runs this regime and passes; the 2D extra dimension does not introduce a new stiffness mode. | T2 / T3 use `v_sat=1e2` (matches 1D). The existing Radau `max_step` cap in `jv_sweep_2d._integrate_step` already guards against the flat-band stiffness mode. If T2 hangs, fall back to `v_sat=1e3` and document the tolerance. |
| **R6** | Face-normal limitation under strong lateral `E_x`. | Face-normal disagrees with total-|E| only when `|E_x|` is comparable to `|E_y|`, e.g. near grain boundaries. The disagreement is bounded above by `μ(|E_y|) − μ(√(E_x² + E_y²))`, which is small when `|E_x| ≪ |E_y|` (a typical bias-driven device). For Stage A and Stage B(a) regimes this is ≤ 1 %. **This is the documented limitation of Option A, not a bug.** | If a future regression on a strongly non-uniform 2D field (e.g. μ(E) + finer GB) flags physically suspect μ behaviour near grain boundaries, ship Option B as a separate follow-up. T4 coexistence smoke at least verifies finiteness of μ(E) + GB. |
| **R7** | Tier-flag bypass (Issue I1 reprise). | The B(c.1) end-of-stage review caught a tier-flag bypass on the 2D Robin path. The same shape of bug is possible here if `_has_field_mobility` forgets to AND with `sim_mode.use_field_dependent_mobility`. | T5 explicitly pins this — Issue I1's lesson baked into the test surface from the start. |
| **R8** | `sg_fluxes_2d_*` API extension breaks 1D-shaped callers. | Adding keyword-only override args is a non-breaking change in Python, but a positional-call site or a custom subclass could still break. | Run the full unit suite after T2/T3 (`pytest tests/unit/twod/`) and the integration suite. Existing call sites in `continuity_rhs_2d` use keyword args, so the risk is small. |

---

## 6. Implementation task ordering (high-level — the planning step will expand)

Tentative arc, ~7 commits, similar shape to Stage B(c.1):

1. **T1** — Add 18 + 1 field-mobility fields to `MaterialArrays2D`; populate via build path (per-node masks → arithmetic-meaned face arrays); add T5 tier-gate test. Mirror B(c.1) Task 2.
2. **T2** — Extend `sg_fluxes_2d_n/p` and `continuity_rhs_2d` with optional per-face D-override kwargs (`D_n_x_face`, `D_n_y_face`, `D_p_x_face`, `D_p_y_face`, `D_n_wrap`, `D_p_wrap`). Backward-compat unit test: no override → bit-identical to current.
3. **T3** — Add μ(E) recompute block in `assemble_rhs_2d`. Backward-compat snapshot test (Stage-A V_oc / J_sc / FF unchanged when default presets used).
4. **T4** — T6 periodic-wrap finite-RHS unit test.
5. **T5** — T2 1D ↔ 2D parity gate (slow regression).
6. **T6** — T3 bounded-shift + T4 coexistence smoke.
7. **T7** — `CLAUDE.md` Stage B(c.2) section + full fast/slow suite green + push.

---

## 7. Open decisions / unresolved

None remaining at the design level. The brainstorm resolved:

- Formulation: face-normal (Option A); Option B deferred.
- Storage: per-face `(Ny, Nx-1)` / `(Ny-1, Nx)` / `(Ny,)` arrays on `MaterialArrays2D`, defaulting `None`.
- Mean for `v_sat` / `β` / `γ_PF`: arithmetic.
- `D_n` / `D_p` mean: still harmonic (unchanged).
- API extension: keyword-only override on `sg_fluxes_2d_*` and `continuity_rhs_2d`.
- Tier gating: `sim_mode.use_field_dependent_mobility` AND any param > 0 (mirror B(c.1)).
- Validation envelope: T2 at `v_sat=1e2` with `(1 mV / 5e-4 / 1e-3)` tolerance (same as B(c.1) Robin gates).

The implementation plan (next step, via `superpowers:writing-plans`) should produce a 7-task TDD-style plan with full code per step.

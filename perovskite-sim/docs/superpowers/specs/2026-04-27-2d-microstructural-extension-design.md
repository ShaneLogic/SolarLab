# 2D Microstructural Drift-Diffusion Extension — Design

> **For Claude:** This is the design (spec) phase. The implementation plan should be produced via `superpowers:writing-plans` from this document. Do not start implementation from this file directly.

**Goal.** Extend the existing 1D drift-diffusion + Poisson + ion migration simulator to a 2D variant that resolves lateral microstructure inside the absorber (grain boundaries, defect clusters, halide-segregated patches) while leaving the 1D solver fully intact and improving its tooling, not replacing it. The first scientific deliverable is the V_oc penalty as a function of grain size for a single vertical grain boundary on a MAPbI3 stack, validated against published data.

**Non-goals.** Cell-scale lateral physics (finger / busbar geometry), 2D light management (textures, photonic structures), module-scale string modelling, full 3D, and axisymmetric geometries are explicitly out of scope for this design. Each is a separate project with a different solver core.

---

## 1. Scope

This design covers two stages.

| Stage | Name | Purpose | Ships when |
|------|------|---------|-----------|
| 2D-α (Stage A) | Validation gate | 2D solver on a laterally uniform device must reproduce the 1D answer within tight tolerances | All six checks in §7 pass |
| 2D-β (Stage B) | First 2D physics | Single vertical grain boundary in the absorber; J–V sweep and field maps as a function of grain size | V_oc(L_g) trend matches published MAPbI3 data within ~30 mV |

The following are deliberately deferred to future stages and are NOT part of this spec:

- 2D-γ — periodic / Voronoi-tessellated grain networks
- 2D-δ — ionic accumulation at grain boundaries
- 2D-ε — halide segregation in mixed-halide absorbers
- 2D-ζ — coupling 2D-microstructural with 2D-light-management (textured front)

The architecture in this spec must be forward-compatible with all four follow-ons; none should require a solver-core rework.

---

## 2. Repository layout

The 2D code lives as a subpackage inside the existing `perovskite_sim/` Python package, on a long-lived feature branch `2d-extension`. The 1D code is untouched.

```
SolarLab/                                          [git branch: 2d-extension]
└── perovskite-sim/
    ├── perovskite_sim/
    │   ├── constants.py                           shared with 2D
    │   ├── data/   (n,k spectra, AM1.5G)          shared with 2D
    │   ├── models/ (MaterialParams, DeviceStack…) shared with 2D
    │   ├── physics/ (recombination, mobility…)    shared with 2D (dimension-agnostic kernels)
    │   ├── discretization/                        1D only — untouched
    │   ├── solver/ (mol.py, newton.py)            1D only — untouched
    │   ├── experiments/                           1D only — untouched
    │   └── twod/                                  NEW
    │       ├── __init__.py
    │       ├── grid_2d.py        # 2D mesh on (x, y); tanh clustering in both axes
    │       ├── poisson_2d.py     # 5-point sparse Poisson on rectilinear grid
    │       ├── flux_2d.py        # Scharfetter–Gummel flux on horizontal AND vertical edges
    │       ├── continuity_2d.py  # 2D carrier + ion continuity
    │       ├── microstructure.py # grain-map representation; τ(x,y), Eg(x,y), χ(x,y) builders
    │       ├── solver_2d.py      # method-of-lines RHS assembly; calls physics/* kernels
    │       ├── snapshot.py       # SpatialSnapshot2D dataclass
    │       └── experiments/
    │           ├── __init__.py
    │           ├── jv_sweep_2d.py
    │           ├── field_maps_2d.py     # standalone steady-state field extractor
    │           └── voc_grain_sweep.py   # the Stage-B headline experiment
    ├── backend/
    │   └── main.py                                # adds kind: "jv_2d", "field_maps_2d", "voc_grain_sweep"
    ├── frontend/
    │   └── src/panels/
    │       ├── jv-2d.ts                           NEW (Phase 1 — parallel-tab MVP)
    │       ├── field-maps-2d.ts                   NEW
    │       └── voc-grain-sweep.ts                 NEW
    ├── configs/
    │   └── twod/                                  NEW directory for 2D presets (stack + grain map)
    │       ├── nip_MAPbI3_uniform.yaml            Stage-A validation preset
    │       └── nip_MAPbI3_single_gb.yaml          Stage-B headline preset
    └── tests/
        ├── unit/twod/                             NEW
        ├── integration/twod/                      NEW
        └── regression/test_twod_validation.py     NEW (Stage-A gate)
```

`perovskite_sim/twod/` and `experiments/` deliberately do not share filenames — `solver_2d.py` next to `solver/mol.py`, not `solver/mol.py` patched to do 2D. This prevents accidental import-shadowing and keeps every git diff scoped.

The `physics/` modules (recombination, mobility, traps, optics back-end interfaces) are dimension-agnostic — `R(n, p, T)` does not care whether `n` is a 1D or 2D array. `twod/solver_2d.py` calls them on flattened arrays exactly as `solver/mol.py` does. Any module that *is* dimension-specific (Poisson, SG flux, mesh) gets a parallel `*_2d` file under `twod/` and stays out of `physics/`.

---

## 3. Numerical method (2D core)

**Mesh.** Rectilinear tensor-product grid on (x, y). Vertical (y) axis is the existing 1D stack with tanh clustering at layer interfaces and contacts. Lateral (x) axis is a separate tanh-clustered 1D grid clustered around grain-boundary positions. Default size: ~50 × 300 = 15 000 nodes for Stage B. Domain in y is the full multilayer stack; domain in x is one or more grains.

**State vector.** `y = (n, p, P_pos, P_neg)` per node, flattened in C order (y-major). Ion species count is unchanged from 1D (single or dual ion).

**Poisson.** 5-point finite-difference stencil with harmonic-mean face permittivity. Sparse LU via `scipy.sparse.linalg.splu`, factored once at build time on the layer-only `eps_r(x,y)` field and re-used. Banded structure has ~5N non-zeros for an N-node grid. Direct LU is feasible up to ~10⁵ nodes; multigrid is the upgrade path beyond that and is not required for Stage A or B.

**Carrier flux.** Scharfetter–Gummel on every internal edge — both horizontal (between (i,j) and (i+1,j)) and vertical (between (i,j) and (i,j+1)). Each edge uses its own face-averaged `D_n`, `D_p`, `μ_n`, `μ_p`, `eps_r`. The 1D thermionic-emission cap (Phase 1) generalises to per-edge: any edge crossing a layer-interface line in y, or a grain-boundary line in x, applies the Richardson cap if |ΔE_c| > 0.05 eV.

**Time integration.** Same `scipy.integrate.solve_ivp(Radau)` as 1D, on the larger flattened state vector. The Jacobian is provided as a sparse matrix to avoid Radau's default dense fallback. `max_step` cap from 1D experiments is preserved per voltage interval. Per-voltage `_JV_RADAU_MAX_NFEV` guard preserved.

**MaterialArrays2D cache.** Direct analogue of the 1D `MaterialArrays`. `twod.solver_2d.build_material_arrays_2d(x, y, stack, microstructure)` returns an immutable `MaterialArrays2D` holding all per-node and per-edge arrays plus a pre-factored 2D Poisson operator. Microstructure inputs (τ_GB, GB locations, defect cluster maps) feed into this build step and produce the spatially-varying τ_n(x,y), τ_p(x,y), and (later) Eg(x,y), χ(x,y) fields.

**Boundary conditions.** Lateral (x): periodic by default for grain-network problems, Neumann for finite-grain problems. Vertical (y): unchanged from 1D — Dirichlet contacts (with optional Phase 3.3 Robin/Schottky overrides).

**Optical generation.** Stage A and B use extruded 1D TMM — `G(x,y) = G_1D(y)` for every x. This is exact for laterally uniform illumination and laterally uniform optical properties (which holds even in the presence of GBs, since GB widths are sub-wavelength and don't affect the optical response). Replacing this with a real 2D Maxwell solver is the separate light-management project (Stage C in the brainstorm), not part of this spec.

---

## 4. Frontend architecture

Two-phase plan on the existing single Vite frontend at `perovskite-sim/frontend/`. No new Vite project; no fork.

**Phase 1 — parallel tabs (MVP).** Add three new tabs:

- `2D J–V Sweep` (kind: `jv_2d`) — same form chrome as the 1D J–V panel, line plot of J–V plus heatmap thumbnails of n(x,y), p(x,y), φ(x,y) at user-selected V.
- `2D Field Maps` (kind: `field_maps_2d`) — single steady-state run at a given V, full-resolution heatmaps + vector field of (J_x, J_y).
- `V_oc vs Grain Size` (kind: `voc_grain_sweep`) — Stage-B headline experiment. Sweeps L_g, returns V_oc(L_g) plus per-L_g field maps.

These reuse `device-panel.ts`, `api.ts`, `job-stream.ts`, the SSE progress-bar pattern, and `plot-theme.ts` unchanged. Each new panel adds its own form module and its own Plotly render function.

**Phase 2 — pluggable refactor (deferred).** Factor out `plot-widgets/{line-plot,heatmap,contour,vector-field}.ts` and consolidate the 1D and 2D J–V panels into a single panel with a `dim: "1d" | "2d"` toggle. No timeline. The refactor is triggered only when concrete duplication or plot needs force the issue — e.g. when a third dimension-conditional panel would copy the same form chrome a third time, or when a 2D plot widget needs a feature (synchronised colour scale, shared axes across heatmaps) that the duplicated code can't support cleanly. Until then, parallel `*-2d.ts` panels stay.

**Phase 3 — comparison tab.** A `Compare 1D vs 2D` tab that fires both `kind: "jv"` and `kind: "jv_2d"` on the same device, overlays the J–V curves, and shows 2D field maps next to vertical-line cuts of the 1D solution at matched x. This is the scientifically valuable artefact the shared-frontend approach unlocks.

---

## 5. Backend changes

`backend/main.py`:

- Three new entries in the `_DISPATCH` table: `jv_2d`, `field_maps_2d`, `voc_grain_sweep`.
- Each dispatches to a closure that calls the corresponding `twod.experiments.*` function with the existing `progress: Callable[...]` kwarg.
- The result serialiser (`_serialize_result`) gains branches for `JV2DResult`, `FieldMaps2DResult`, `VocGrainSweepResult` — each emits arrays as flat lists with shape metadata so the frontend can `np.reshape`-equivalent on the JS side.
- `_describe_active_physics` gains a `dim` field in the SSE result payload (`"1d"` or `"2d"`) so the frontend label can disambiguate.
- New endpoint `GET /api/configs/twod` returns 2D-specific presets from `configs/twod/`. Existing `GET /api/configs` is unchanged.

The streaming-job pattern (SSE, `progress`, `result`, `error`, `done` events) is preserved exactly. No new endpoint family.

---

## 6. Microstructure representation

A small data model in `twod/microstructure.py`:

```python
@dataclass(frozen=True)
class GrainBoundary:
    x_position: float    # m, location in lateral coordinate
    width: float         # m, GB width (typical ~5 nm)
    tau_n: float         # s, electron SRH lifetime inside the GB
    tau_p: float         # s, hole SRH lifetime inside the GB
    layer_role: str      # "absorber" by default; could be "ETL" / "HTL" later

@dataclass(frozen=True)
class Microstructure:
    grain_boundaries: tuple[GrainBoundary, ...] = ()
    # future: defect_clusters, halide_patches, ion_traps_sites
```

Stage A uses `Microstructure()` (empty). Stage B uses `Microstructure(grain_boundaries=(GrainBoundary(x=L_g/2, width=5e-9, tau_n=1e-9, tau_p=1e-9, layer_role="absorber"),))`. The `build_material_arrays_2d` step turns this into a per-node τ_n(x,y), τ_p(x,y) by setting GB-band nodes to the GB lifetimes and bulk nodes to the absorber's bulk τ. Forward-compatible with future GB types (the dataclass can grow fields without breaking older configs).

YAML schema for `configs/twod/*.yaml` adds an optional top-level `microstructure:` block that the loader parses into a `Microstructure`. Files without that block produce `Microstructure()` and run the uniform Stage-A path.

---

## 7. Stage A — Validation gate

A single regression test, `tests/regression/test_twod_validation.py`, runs the 2D solver on a laterally uniform device (no grain boundaries, periodic lateral BC) and asserts:

| Check | Tolerance |
|-------|-----------|
| V_oc(2D) − V_oc(1D) | ≤ 0.1 mV |
| (J_sc(2D) − J_sc(1D)) / J_sc(1D) | ≤ 5×10⁻⁴ |
| FF(2D) − FF(1D) | ≤ 1×10⁻³ |
| max\|n(x,y) − n(0,y)\| / max\|n(0,y)\| | ≤ 1×10⁻⁹ (lateral invariance of n) |
| max\|∇·J\| at interior nodes in steady state | ≤ 1×10⁻³ A/m³ |
| Poisson residual after solve | ≤ 1×10⁻¹² |

Failure of any check is a release-blocker for any 2D physics result. The same preset (`nip_MAPbI3_uniform.yaml`) drives both the 1D run and the 2D run; the 1D run uses the existing `experiments.run_jv_sweep`. Both are in the same test so they cannot drift.

This is the *only* test that gates Stage A. There is no scientific deliverable for Stage A — its purpose is solely to prove the numerics.

---

## 8. Stage B — Single grain boundary

The headline experiment is `twod.experiments.voc_grain_sweep.run_voc_grain_sweep`:

```python
def run_voc_grain_sweep(
    stack: DeviceStack,
    grain_sizes: Sequence[float],   # e.g. (200e-9, 500e-9, 1e-6, 2e-6)
    tau_gb: tuple[float, float] = (1e-9, 1e-9),  # (tau_n, tau_p) at GB
    gb_width: float = 5e-9,
    *,
    progress: ProgressCallback | None = None,
) -> VocGrainSweepResult: ...
```

For each L_g it constructs a `Microstructure` with one GB at x = L_g/2, runs a forward-only J–V sweep (illuminated, periodic lateral BC, lateral domain length = L_g), extracts V_oc, J_sc, FF, and stores spatial snapshots of n(x,y), p(x,y), φ(x,y), J_x(x,y), J_y(x,y) at V_oc and at MPP.

Deliverable artefacts:

1. J–V curve overlay — 1D bulk τ vs 2D for each L_g (line plot)
2. n(x,y) heatmap at V_oc (one per L_g)
3. |J|(x,y) heatmap at MPP (one per L_g)
4. (J_x, J_y) vector field at MPP (one per L_g)
5. V_oc(L_g) plot, with the 1D-bulk-τ V_oc as a horizontal reference line and the published deQuilettes / Stranks data as overlay points
6. τ_eff(L_g) plot — the effective bulk τ that, when fed into the 1D solver, reproduces the 2D V_oc at that L_g. Useful as a 1D-modelling shortcut for future workflows.

The Stage-B success criterion is artefact 5: V_oc(L_g) trend must agree with published MAPbI3 data in shape (monotone increasing toward bulk) and absolute value (within ~30 mV at L_g = 1 µm), and the τ_eff(L_g = 1 µm) extracted from artefact 6 must be within a factor of 2 of the absorber bulk τ used in the 1D simulation.

---

## 9. Numerical envelope

| Quantity | 1D today | 2D Stage-B target | Cost ratio |
|----------|----------|-------------------|-----------|
| Mesh nodes | ~300 | 50 × 300 = 15 000 | 50× |
| State vector | ~900 | ~45 000 | 50× |
| Poisson solve | tridiagonal LU, cached | sparse LU on 5-point stencil, cached | ~100× |
| RHS / Jacobian | tridiagonal | 5-band sparse | ~30× |
| Full forward J–V (ionmonger preset) | ~25 s | target ≤ 30 min | ~70× |

Performance levers if the 30-min target is missed (in order of effort):

1. Numba-JIT the SG flux assembly inner loop (hot path).
2. Replace sparse LU with algebraic multigrid for Poisson (`pyamg`).
3. Move from method-of-lines + Radau to a fully implicit Newton-coupled solve at each voltage step.
4. GPU acceleration via CuPy on the Poisson + flux assembly.

None of these are required for Stage A or B and none are part of this spec. They are listed only to confirm headroom exists.

---

## 10. Test strategy

Three new test trees mirror the existing 1D structure:

- `tests/unit/twod/` — per-module: mesh construction, microstructure parsing, sparse Poisson solve, SG flux on a single edge with a known analytical answer, dimension-agnostic kernel sharing (recombination R(n,p,T) gives the same value when called from 1D and 2D code paths).
- `tests/integration/twod/` — end-to-end: build a 2D problem on a real preset, integrate one second of transient, check finiteness and conservation.
- `tests/regression/test_twod_validation.py` — Stage-A gate (above).

Coverage target: 80% on `perovskite_sim/twod/**`, matching the user's global testing rule.

The existing 1D test suite must remain green throughout the 2D feature branch — `pytest -m 'not slow'` and `pytest -m slow` both pass before any merge to main.

---

## 11. Documentation update plan (per project policy)

Per the standing rule that README and technical docs must stay in sync with code changes, the following updates ship alongside the code in each phase:

**Stage A merge:**
- `perovskite-sim/CLAUDE.md` — new "## 2D microstructural extension (Phase 6 — 2026)" top-level section after the existing Phase 5 block, describing the `twod/` subpackage, the mesh / Poisson / SG approach, the validation-gate semantics, and the `MaterialArrays2D` cache.
- `perovskite-sim/README.md` — new "Dimensionality" subsection under Key Features, briefly stating that 1D is the default and 2D is opt-in via `twod.experiments.*` for microstructural studies.
- `SolarLab/CLAUDE.md` (root) — add a sentence to "Which Tree To Work In" mentioning that 2D work lives in the same `perovskite-sim/` tree under `perovskite_sim/twod/`.

**Stage B merge:**
- `perovskite-sim/CLAUDE.md` — extend the Phase 6 block with a "Single grain boundary" subsection describing the `voc_grain_sweep` experiment, the `GrainBoundary` / `Microstructure` data model, and the artefact list.
- `perovskite-sim/README.md` — a single user-facing paragraph in Key Features describing the V_oc vs grain size experiment.
- A new `perovskite-sim/docs/twod_overview.md` written for end users, describing how to construct a `Microstructure`, run the headline experiment, and interpret the artefacts. This is the "tutorial" companion to the existing `perovskite-sim/docs/benchmark_*` files.

**Future stages (γ/δ/ε):** each adds its own subsection to the Phase 6 block in `CLAUDE.md`, following the existing "Phase 4a, 4b, 5" structure. The README gets at most one paragraph per major capability, not per stage.

Roadmap entries describing future stages stay in this design doc and in chat — not in the public README, per the prior `feedback_readme_scope.md` rule.

---

## 12. Resolved decisions

The five points raised during brainstorming are settled as follows. They are recorded here so future contributors can see the reasoning rather than re-litigating them.

1. **Mesh.** Tensor-product rectilinear, tanh-clustered in both axes. Sufficient for Stages A → γ. Unstructured FE is revisited only if a future stage (Voronoi grain networks, non-axis-aligned GBs at oblique angles) actually demands it.
2. **Grain-boundary representation.** Volumetric — a band of nodes with width δ = 5 nm and reduced τ_n, τ_p. Spec must document the δ ↔ v_GB conversion (`v_GB ≈ δ / τ_GB` in the thin-GB limit) so users targeting a literature surface-recombination velocity can pick the right τ_GB.
3. **Lateral boundary conditions.** Periodic by default. Neumann is supported as a per-experiment override (e.g. `voc_grain_sweep(..., lateral_bc="neumann")`). Stage-A validation runs on both BC choices to confirm neither path has a bug.
4. **Frontend Phase 2 (pluggable refactor).** Deferred — no timeline. Trigger conditions: a third dimension-conditional panel would copy the same form chrome a third time, or a 2D plot widget needs a cross-panel feature (synchronised colour scale, shared axes across heatmaps) that duplicated code cannot support cleanly. Until either trigger fires, parallel `*-2d.ts` panels stay.
5. **Tandem 2D.** Out of scope. Stages A and B are single-junction only. Tandem-2D is a future stage and gets its own spec when it lands.

---

## 13. Sign-off

The user accepts this design before any code is written. The next step is to invoke `superpowers:writing-plans` from this design document to produce an implementation plan broken into tasks. Implementation only begins from the plan, not from this spec.

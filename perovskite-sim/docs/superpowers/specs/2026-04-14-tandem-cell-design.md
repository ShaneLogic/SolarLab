# Tandem Cell Feature — Design Spec

**Date:** 2026-04-14
**Status:** Approved for planning
**Scope:** v1 — 2T monolithic all-perovskite tandem, J-V only, validated against Lin 2019 benchmark

---

## 1. Goal

Add 2-terminal (2T) monolithic tandem solar cell simulation to perovskite-sim. v1 targets all-perovskite tandems (wide-gap FA-Cs top + Sn-Pb narrow-gap bottom), restricted to J-V sweep, and must reproduce the Lin 2019 Nature Energy benchmark within tolerance before the feature ships.

**Thesis context:** This feature supports PhD-thesis work on lab stack screening. Physics validation takes priority over UX polish.

## 2. Benchmark Anchor

**Paper:** Lin et al., "Monolithic all-perovskite tandem solar cells with 24.8% efficiency exploiting comproportionation to suppress Sn(ii) oxidation in precursor ink", *Nature Energy* **4**, 864–873 (2019).
**URL:** https://www.nature.com/articles/s41560-019-0466-3

**Reported figures for v1 target:**
- PCE: 24.8% (small-area 0.049 cm²)
- Top sub-cell gap: ~1.77 eV (FA₀.₈Cs₀.₂Pb(I₀.₆Br₀.₄)₃)
- Bottom sub-cell gap: ~1.22 eV (FA₀.₇₅Cs₀.₂₅Sn₀.₅Pb₀.₅I₃)
- Stack order: glass / ITO / PEDOT:PSS / wide-gap perovskite / C₆₀ / SnO₂ / Au-PEDOT:PSS recomb / PEDOT:PSS / narrow-gap perovskite / C₆₀ / BCP / Cu
- Recombination junction: near-ohmic at 1-sun operation

**v1 pass criteria:** simulated tandem J_sc, V_oc, FF, PCE each within ±10% of Lin 2019 Table 1 values using the paper's nominal layer thicknesses.

## 3. Architecture

**Chosen approach:** Independent sub-cell drift-diffusion solves + combined-stack TMM for optical coupling + J-V interpolation + series addition for electrical coupling.

**Rationale:** Reuses ~90% of existing single-junction solver code. Avoids coupled drift-diffusion across the full stack (which would require ~2× the grid, new boundary conditions at the junction, and fragile convergence). Captures the two physical couplings that matter most for J-V prediction:
1. **Optical coupling** via per-sub-cell absorption partitioning of one combined TMM run
2. **Electrical coupling** via series current-matching across the two sub-cell J-V curves

**Limitations accepted for v1:**
- No lateral transport effects
- No explicit recombination-layer physics (ideal ohmic assumption)
- No coupling between top-cell back contact and bottom-cell front contact beyond the shared current boundary condition

## 4. Physics Model

### 4.1 Optical model — combined TMM with partitioning

Run a single transfer-matrix calculation over the full stack (glass / top sub-cell / recombination junction / bottom sub-cell / back contact). For each layer, compute absorption per wavelength A_layer(λ) as in the current single-junction TMM path.

Tag each layer in the stack as belonging to either the **top sub-cell**, the **junction**, or the **bottom sub-cell** via a new `sub_cell` field in the tandem YAML schema (see §5). Absorption in junction layers is treated as parasitic (no photocurrent generation).

Sub-cell generation profiles:
- G_top(x, λ) = photogeneration rate density in the top sub-cell's absorber only, from A_top(x, λ)
- G_bot(x, λ) = same for the bottom sub-cell

Both profiles are passed to the existing drift-diffusion solver as the generation source term. Each sub-cell then solves on its own grid with its own contacts and doping.

**Why combined TMM (not sequential two-pass):** Captures back-reflection from the bottom back contact into the top cell, interference effects across the full stack, and avoids the systematic J_sc under-prediction that sequential two-pass TMM produces (~1-3 mA/cm² in typical all-perovskite stacks). Extra code is modest — just per-layer sub-cell tagging.

### 4.2 Electrical model — independent sub-cells + series match

Each sub-cell is solved as an independent single-junction device:
1. Top sub-cell receives G_top, runs full J-V sweep → (J, V_top(J)) curve
2. Bottom sub-cell receives G_bot, runs full J-V sweep → (J, V_bot(J)) curve

Both sweeps use the existing `mol.py` / `jv_sweep.py` solver path unchanged. The two sweeps are independent and can run in parallel.

**Series matching:** For the 2T tandem, both sub-cells carry the same current at every operating point. Build the tandem J-V curve by:

For each current J in a common J grid:
- V_top = interp(J, top_sweep.J, top_sweep.V)
- V_bot = interp(J, bot_sweep.J, bot_sweep.V)
- V_tandem = V_top + V_bot + V_junction (= 0 for ideal ohmic)

Then tandem metrics (J_sc, V_oc, FF, PCE) are extracted from (J, V_tandem) using the existing metrics module.

**Junction model (v1):** ideal ohmic. V_junction = 0, R_junction = 0. Future PRs may add lumped R or tunnel-diode junction models behind a `junction.model` discriminator field.

## 5. YAML Schema

Tandem configs are a new file type referencing two existing single-junction YAMLs:

```yaml
# configs/tandem_lin2019.yaml
schema_version: 1
device_type: tandem_2T_monolithic

tandem:
  top_cell: nip_wideGap_FACs_1p77.yaml       # existing single-junction YAML
  bottom_cell: nip_SnPb_1p22.yaml             # existing single-junction YAML
  junction:
    model: ideal_ohmic                        # v1: only ideal_ohmic supported
  light_direction: top_first                  # light enters top sub-cell first
  benchmark:
    reference: lin2019_nature_energy
    target_pce: 24.8
    tolerance_pct: 10

# Each referenced single-junction YAML gets an additional sub_cell tag per layer
# applied automatically by the tandem loader (top_cell → sub_cell: top, etc.)
# Junction layers listed below are treated as parasitic absorbers.
junction_stack:
  - material: Au_PEDOT_PSS
    thickness_nm: 1.5
    sub_cell: junction
  - material: PEDOT_PSS
    thickness_nm: 20
    sub_cell: junction
```

**Loader rules:**
- `device_type: tandem_2T_monolithic` routes through the tandem code path
- Top and bottom cell YAMLs are loaded via the existing single-junction loader unchanged
- Junction stack is inserted between top and bottom layers for the combined TMM run
- Validation rejects any junction model other than `ideal_ohmic` in v1

## 6. File Structure

**New files:**
- `perovskite_sim/physics/tandem_optics.py` — combined TMM + per-sub-cell partitioning
- `perovskite_sim/experiments/tandem_jv.py` — tandem J-V driver (runs two sub-cell sweeps + series match)
- `perovskite_sim/models/tandem_config.py` — tandem YAML loader + validation
- `configs/nip_wideGap_FACs_1p77.yaml` — wide-gap top sub-cell preset matching Lin 2019
- `configs/nip_SnPb_1p22.yaml` — narrow-gap bottom sub-cell preset matching Lin 2019
- `configs/tandem_lin2019.yaml` — tandem benchmark config
- `perovskite_sim/data/nk/` — add n,k CSVs for any materials not already present (FA-Cs wide-gap, Sn-Pb narrow-gap, PEDOT:PSS, C₆₀, SnO₂, BCP, Cu, Au) — reuse existing sources where possible
- `tests/unit/physics/test_tandem_optics.py` — unit tests for partitioning logic
- `tests/unit/experiments/test_tandem_jv.py` — unit tests for series-match interpolation
- `tests/integration/test_tandem_lin2019.py` — benchmark regression test (marked `slow`)
- `backend/routes/tandem.py` — POST /simulate/tandem endpoint
- `frontend/src/panels/tandem.ts` — Tandem tab UI with two StackVisualizer instances
- `frontend/src/stack/tandem-stack-visualizer.ts` — side-by-side sub-stack layout wrapper
- `frontend/src/panels/benchmark-compare.ts` — compare panel for reference vs simulated metrics

**Modified files:**
- `perovskite_sim/models/config_loader.py` — dispatch on `device_type` to tandem loader
- `perovskite_sim/experiments/__init__.py` — export `run_tandem_jv`
- `backend/main.py` — register /simulate/tandem route
- `frontend/src/App.ts` (or equivalent router) — add Tandem tab
- `frontend/src/types.ts` — add tandem config types

**Unchanged but reused:**
- `perovskite_sim/solver/mol.py` — drift-diffusion solve per sub-cell
- `perovskite_sim/experiments/jv_sweep.py` — per-sub-cell J-V sweep
- `perovskite_sim/physics/optics.py` — TMM primitives (tandem_optics.py calls these)
- `frontend/src/stack/stack-visualizer.ts` — reused per sub-stack

## 7. Data Flow

```
tandem.yaml
   │
   ▼
tandem_config.py loader
   │
   ├─► top_cell params (single-junction device)
   ├─► bottom_cell params (single-junction device)
   └─► junction_stack params
   │
   ▼
tandem_optics.py
   │
   ├─► build combined stack (top + junction + bottom)
   ├─► run combined TMM → A(x, λ) per layer
   ├─► partition: G_top(x), G_bot(x), parasitic(x)
   │
   ▼
tandem_jv.py
   │
   ├─► run jv_sweep(top_cell, G_top) → top_JV
   ├─► run jv_sweep(bottom_cell, G_bot) → bot_JV
   ├─► common J grid
   ├─► V_tandem(J) = interp(top_JV) + interp(bot_JV) + 0
   └─► extract tandem metrics (J_sc, V_oc, FF, PCE)
   │
   ▼
result payload → backend → frontend
```

## 8. Testing Strategy

### 8.1 Unit tests (fast)
- **tandem_optics partitioning**: synthetic 3-layer stack (top absorber + junction + bottom absorber) with known A(λ); verify G_top integrates to top-only absorption, G_bot to bottom-only, parasitic accounted for.
- **tandem_optics tagging**: verify junction layers are correctly classified as parasitic.
- **series-match interpolation**: two synthetic J-V curves with known series sum; verify V_tandem(J) reconstruction and monotonicity.
- **tandem YAML loader**: reject unsupported junction models, reject missing `sub_cell` tags, accept valid Lin 2019 config.
- **current-matching edge cases**: sub-cell J_sc mismatch (smaller sub-cell limits tandem J_sc); verify V_oc is near top+bottom single-cell V_oc sum.

### 8.2 Integration test (slow, marked `@pytest.mark.slow`)
- **test_tandem_lin2019**: load `configs/tandem_lin2019.yaml`, run full tandem J-V sweep, assert PCE within ±10% of 24.8%, J_sc/V_oc/FF each within ±10% of Lin 2019 Table 1.

### 8.3 Backend test
- POST /simulate/tandem with Lin 2019 config returns a valid tandem J-V payload with all metrics fields populated.

### 8.4 Frontend manual test
- Load Tandem tab, import Lin 2019 config, run simulation, verify side-by-side visualizer shows both sub-stacks, benchmark compare panel shows both reference and simulated values side-by-side.

## 9. Out of Scope for v1 (explicit)

- Perovskite/Si and perovskite/CIGS sub-cells
- Impedance (C-f, EIS) on tandems
- Degradation sweeps on tandems
- Lumped R_junction and tunnel-diode junction models
- 3T / 4T mechanically-stacked tandems
- Tandem integration into the single-junction Layer builder tab (tandem gets its own tab)
- Spectral splitting beyond standard AM1.5G
- Angular / concentrated illumination
- Self-consistent coupled drift-diffusion across both sub-cells

## 10. Open Questions — none

All 11 design questions resolved during brainstorming. Ready for implementation planning.

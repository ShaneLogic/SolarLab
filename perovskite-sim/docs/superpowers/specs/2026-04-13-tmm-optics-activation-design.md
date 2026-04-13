# TMM Optics Activation — Design Spec

**Date:** 2026-04-13
**Status:** Draft → awaiting user review
**Scope:** Phase 2a of the workstation physics upgrade — activate the existing TMM optical model for perovskite presets, expand the n,k material library, and surface per-layer optical-material selection in the Device pane.

> **Out of scope (deferred to later specs):**
> - **Phase 2b — Layer builder UI** (add/remove/reorder layers, role editing, custom-stack workflow). Shares the optical-material dropdown built here but is otherwise independent.
> - **Phase 3 — Tandem cell support** (dual-absorber stacks with tunnel junctions, sub-cell current matching). Requires solver-level changes, not just UI.

---

## 1. Goal

Make the existing `perovskite_sim/physics/optics.py` transfer-matrix engine actually run for shipped perovskite presets and let users select per-layer optical materials from the workstation Device pane on full tier.

The TMM engine itself, the AM1.5G spectrum loader, and the cache hookup in `solver/mol.py:build_material_arrays` already exist and are tested. This spec ships the **data, presets, frontend wiring, and small Python adjustments** needed to take the feature from "implemented but dormant" to "active by default for the perovskite presets."

## 2. Background — what already exists

```
perovskite_sim/
├── physics/optics.py                 (354 lines, full TMM engine)
│   ├── TMMLayer dataclass
│   ├── _transfer_matrix_stack()      (2×2 matrix product)
│   ├── _electric_field_profile()     (forward + backward field amplitudes)
│   ├── tmm_absorption_profile()      (A(x, λ))
│   ├── tmm_generation()              (G(x) integrated over AM1.5G)
│   └── tmm_reflectance()             (R(λ))
│
├── solver/mol.py
│   ├── _compute_tmm_generation()     (called from build_material_arrays)
│   └── MaterialArrays.G_optical      (cached G(x) array)
│
├── data/
│   ├── am15g.csv                     (AM1.5G photon flux)
│   └── nk/
│       ├── MAPbI3.csv
│       ├── TiO2.csv
│       └── spiro_OMeTAD.csv
│
├── models/
│   ├── parameters.py:MaterialParams.optical_material   (str | None)
│   └── config_loader.py              (loads optical_material from YAML)
│
└── tests/
    ├── unit/physics/test_optics.py             (13 tests)
    └── integration/test_tmm_integration.py     (5 tests)
```

**Activation logic** (`solver/mol.py:_compute_tmm_generation`): if any layer in the stack has `optical_material` set, build the TMM stack and compute `G_optical`; otherwise leave `MaterialArrays.G_optical = None` and `assemble_rhs` falls back to `beer_lambert_generation`.

**Why TMM is currently dormant:** zero shipped YAMLs in `configs/` set `optical_material:` on any layer, so the TMM code path never executes in production.

## 3. Architecture

```
┌─ Frontend (full tier only) ────────────────────────────────┐
│ Device pane                                                │
│   per-layer "Optics" group                                 │
│     dropdown: optical_material                             │
│       ▼ (none)  ← Beer-Lambert fallback                    │
│       ▼ MAPbI3                                             │
│       ▼ ITO                                                │
│       ▼ ...                                                │
│   header badge: "TMM active · N layers"                    │
└────────────────┬───────────────────────────────────────────┘
                 │ GET /api/optical-materials
                 ▼
┌─ Backend ──────────────────────────────────────────────────┐
│ /api/optical-materials → ["MAPbI3", "ITO", "FTO", ...]     │
│ (auto-scans data/nk/ — drop a CSV, no code change)         │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌─ perovskite_sim ───────────────────────────────────────────┐
│ data/nk/                                                   │
│   MAPbI3, TiO2, spiro_OMeTAD              (existing)       │
│   ITO, FTO, SnO2, C60, PCBM, PEDOT_PSS, Ag, Au, glass      │
│   manifest.yaml  ← citations + sources                     │
│                                                            │
│ physics/optics.py        (incoherent-layer support added)  │
│ models/device.py         (skip role:substrate in BCs)      │
│                                                            │
│ configs/                                                   │
│   nip_MAPbI3_tmm.yaml    ← new TMM-enabled variant         │
│   pin_MAPbI3_tmm.yaml    ← new TMM-enabled variant         │
└────────────────────────────────────────────────────────────┘
```

**Key insight:** the Python physics is ~95% done. The work is **data + UI + presets + tests + two small Python adjustments** (incoherent-layer flag, substrate-role electrical BC skip).

## 4. Material library expansion

### 4a. New CSV files in `perovskite_sim/data/nk/`

| Material | Role | Source | Notes |
|---|---|---|---|
| `MAPbI3.csv` | absorber | refractiveindex.info — Phillips 2018 | already shipped |
| `TiO2.csv` | ETL (n-i-p) | refractiveindex.info — Siefke 2016 | already shipped |
| `spiro_OMeTAD.csv` | HTL (n-i-p) | Filipic 2015 | already shipped |
| **`ITO.csv`** | front contact | refractiveindex.info — König 2014 | Sn-doped In₂O₃, sputter-deposited |
| **`FTO.csv`** | front contact | refractiveindex.info — Filipic 2015 | F-doped SnO₂, common on glass |
| **`SnO2.csv`** | ETL (planar n-i-p) | refractiveindex.info — Filipic 2015 | most common modern ETL |
| **`C60.csv`** | ETL (p-i-n) | Ren 2015 | common p-i-n stack ETL |
| **`PCBM.csv`** | ETL (p-i-n) | Ren 2015 | alternative p-i-n ETL |
| **`PEDOT_PSS.csv`** | HTL (p-i-n) | refractiveindex.info — Lee 2018 | common p-i-n HTL |
| **`Ag.csv`** | back contact | refractiveindex.info — Johnson & Christy 1972 | back reflector for n-i-p |
| **`Au.csv`** | back contact | refractiveindex.info — Johnson & Christy 1972 | back reflector alternative |
| **`glass.csv`** | substrate | Schott BK7 dispersion | front substrate (semi-infinite, incoherent) |

### 4b. CSV format (matches existing files)

```csv
lambda_nm,n,k
300,2.45,0.85
305,2.46,0.83
...
1000,2.40,0.00
```

**Wavelength range:** 300–1000 nm (extended from current 800 nm to cover the perovskite band-edge tail and prepare for narrower-bandgap absorbers without re-sourcing).

### 4c. Manifest file

New `perovskite_sim/data/nk/manifest.yaml` documenting provenance:

```yaml
ITO:
  source: refractiveindex.info
  reference: "König et al. 2014, Opt. Mater. Express 4, 689"
  wavelength_range_nm: [300, 1000]
  notes: "Sn-doped In₂O₃, sputter-deposited"

FTO:
  source: refractiveindex.info
  reference: "Filipic et al. 2015, Opt. Express 23, A263"
  wavelength_range_nm: [300, 1000]
  notes: "F-doped SnO₂ on glass, ~500 nm thick"

# ... one entry per material
```

`manifest.yaml` is loaded for documentation purposes only — it does not affect runtime. The backend's `/api/optical-materials` endpoint can optionally surface the reference string in the dropdown tooltip later (out of scope for this spec).

## 5. Incoherent-layer support in `optics.py`

**Problem:** mm-thick glass substrates are much longer than the coherence length of sunlight. Treating them as coherent layers in the TMM matrix product produces unphysical sub-nanometer interference fringes (period `λ²/(2nd)` ≈ 0.1 nm for 1 mm BK7).

**Fix:** Add an `incoherent: bool = False` field to `TMMLayer`. In `_transfer_matrix_stack`, layers with `incoherent=True` are skipped from the coherent matrix product and instead contribute:

1. A single Fresnel reflection at the air→glass interface (computed analytically from `n_air, n_glass`).
2. Bulk Beer-Lambert transmission through the glass thickness using `α = 4πk/λ`.
3. A single Fresnel reflection at the glass→next-layer interface.

The next coherent layer downstream sees the reduced incident intensity but no phase information from the glass.

**Implementation notes:**
- Only **one** incoherent layer is supported in this spec, and it must be the **first** layer of the stack (the substrate). A general incoherent/coherent-mixed solver is out of scope (it requires intensity-matrix bookkeeping rather than amplitude-matrix bookkeeping).
- The validation in `_compute_tmm_generation` raises `ValueError` if `incoherent=True` appears anywhere except the first layer.
- ~30 lines of new code in `optics.py`, plus 1 unit test.

## 6. Substrate-role electrical BC handling

**Problem:** The TMM presets add `glass` (and optionally a thick metal back reflector if treated as a separate optical-only layer) as physical layers in the stack so light reflects off them properly. But the electrical drift-diffusion solver expects every layer in the YAML to be a semiconductor it can integrate carriers through. Glass has no carriers and no electrical role.

**Fix:** Add a new `role: substrate` value (alongside the existing free-form values like `"absorber"`, `"etl"`, `"htl"` — currently `role` is a free-form string in `LayerSpec` used only by `degradation.py:_absorber_region` to find the absorber). Layers with `role == "substrate"` are:

- **Excluded** from the discretization grid (`multilayer_grid` skips them).
- **Excluded** from `MaterialArrays` electrical fields (`D_n_face`, `D_p_face`, `eps_r`, `N_A`, `N_D`, etc.).
- **Included** in the TMM optical stack (so light propagation accounts for them) — `_compute_tmm_generation` walks the full layer list, not the filtered electrical list.

This means the discretization grid and the TMM grid are **different** for TMM-enabled stacks: TMM sees `glass → FTO → TiO2 → MAPbI3 → spiro → Au` (6 layers), but the drift-diffusion solver sees `FTO → TiO2 → MAPbI3 → spiro → Au` (5 layers).

**FTO and Au as semiconductor layers (the chosen approximation).** The existing solver treats every non-substrate YAML layer as a semiconductor with mobility, doping, and bandgap fields. This spec keeps that behavior: FTO and Au are entered as **degenerate semiconductors** (very high doping, ~1e26 m⁻³; thin physical thickness; high mobility). The solver's existing first-layer/last-layer boundary-condition treatment then automatically lands on FTO and Au, which is what we want. This is approximate (a real metal back contact would need a Schottky/ohmic BC model), but it is consistent with how IonMonger and DriftFusion treat their contact layers in 1D, and it does not require any new BC code. A proper contact-as-BC model is **deferred to a future spec**.

**Implementation notes:**
- ~30 lines of additions to `models/device.py` and `solver/mol.py`: a `[layer for layer in layers if layer.role != "substrate"]` filter at the entry points to `multilayer_grid`, `build_material_arrays`, and any other path that walks the layer list electrically.
- `_compute_tmm_generation` already takes the full layer list. Its only change is to map G(x) values from the TMM grid (full layer span) back to the electrical grid (filtered span) — straightforward `np.interp` after subtracting the substrate thickness offset from x.
- Validation: `role: substrate` layers must have `incoherent: true`, must have an `optical_material`, and must be the first layer in the YAML. Raise `ValueError` on any other arrangement (out-of-scope for the simple incoherent path described in §5).

## 7. New preset variants

Two new files in `configs/`. Originals (`nip_MAPbI3.yaml`, `pin_MAPbI3.yaml`) are **untouched** to preserve all existing test pinned values and benchmarks.

### 7a. `configs/nip_MAPbI3_tmm.yaml`

Complete n-i-p stack in optical order (sun-side first):

```yaml
device:
  V_bi: 1.10
  Phi: 1.4e21
  T: 300
  mode: full

layers:
  - name: glass
    role: substrate
    thickness: 1.0e-3        # 1 mm
    optical_material: glass
    incoherent: true

  - name: FTO
    role: front_contact
    thickness: 5.0e-7
    optical_material: FTO
    # Degenerate-semiconductor electrical params for the transparent contact:
    # mu_n: 0.005, mu_p: 0.001, N_D: 1.0e26, Eg: 3.5, chi: 4.4
    # (real values to be tuned during implementation against measured cells)

  - name: TiO2
    role: etl
    thickness: 5.0e-8
    optical_material: TiO2
    # ... existing TiO2 electrical params

  - name: MAPbI3
    role: absorber
    thickness: 4.0e-7
    optical_material: MAPbI3
    # ... existing MAPbI3 electrical params (mobility, ions, recombination)

  - name: spiro_OMeTAD
    role: htl
    thickness: 2.0e-7
    optical_material: spiro_OMeTAD
    # ... existing spiro electrical params

  - name: Au
    role: back_contact
    thickness: 1.0e-7
    optical_material: Au
    # Degenerate metallic back contact, treated electrically as a heavily
    # p-doped wide-gap semiconductor:
    # mu_n: 0.001, mu_p: 0.001, N_A: 1.0e26, Eg: 3.0, chi: 4.0
    # (real values to be tuned during implementation)

interfaces:
  # carry over from nip_MAPbI3.yaml (FTO/TiO2, TiO2/MAPbI3, MAPbI3/spiro, spiro/Au)
```

### 7b. `configs/pin_MAPbI3_tmm.yaml`

Same idea, inverted: `glass → ITO → PEDOT_PSS → MAPbI3 → C60 → Ag`. Uses the modern PCBM-free p-i-n stack with PEDOT:PSS as HTL and C60 as ETL.

### 7c. What stays Beer-Lambert

- `nip_MAPbI3.yaml`, `pin_MAPbI3.yaml` — back-compat with existing tests
- `ionmonger_benchmark.yaml`, `driftfusion_benchmark.yaml` — fidelity checks against external references that themselves use Beer-Lambert; flipping these would invalidate the benchmarks
- `cigs_baseline.yaml`, `cSi_homojunction.yaml` — non-perovskite stacks, optical material data not in scope

## 8. Frontend changes

All UI changes are gated on **full tier**. Legacy and fast tiers see no difference.

### 8a. Optical material dropdown in `config-editor.ts`

The current editor renders only numeric inputs per layer field. Add a new field-type discriminator:

```typescript
type FieldKind = 'numeric' | 'select-optical-material'

interface FieldDef {
  key: string
  label: string
  kind: FieldKind
  // numeric-specific
  unit?: string
  step?: number
  // select-specific
  // (no extra fields — options come from a single fetch)
}
```

For `select-optical-material` fields, render:

```html
<select data-field="optical_material">
  <option value="">(none — Beer-Lambert)</option>
  <option value="MAPbI3">MAPbI3</option>
  <option value="ITO">ITO</option>
  <!-- ... -->
</select>
```

Options are populated from a single page-load fetch:

```typescript
// frontend/src/api.ts
export async function fetchOpticalMaterials(): Promise<string[]> {
  const r = await fetch('http://127.0.0.1:8000/api/optical-materials')
  const j = await r.json()
  return j.materials
}
```

The dropdown lives in the existing **"Optics & Ions"** group of the per-layer accordion in `config-editor.ts`.

### 8b. Tier gating

`isFieldVisible('optical_material', tier)` returns `true` only for `tier === 'full'`. Same for `'incoherent'`. Tier-gating logic already exists in `frontend/src/workstation/tier-gating.ts` — just add the two new keys.

### 8c. "TMM active" badge in the Device pane header

The existing Device pane header shows the device name. Add a small badge after it:

```
Device: nip_MAPbI3_tmm   [TMM active · 6 layers]
```

The badge appears whenever **any** layer in the active device stack has a non-empty `optical_material`. Hover tooltip:

> "Optical generation computed with transfer-matrix method. Layers without `optical_material` fall back to Beer-Lambert."

### 8d. New backend endpoint

```python
# backend/main.py
from pathlib import Path

@app.get("/api/optical-materials")
def list_optical_materials() -> dict:
    """Auto-scan data/nk/ for available optical materials."""
    nk_dir = Path(__file__).parent.parent / "perovskite_sim" / "data" / "nk"
    materials = sorted(
        p.stem for p in nk_dir.glob("*.csv")
        if p.stem != "manifest"
    )
    return {"materials": materials}
```

Auto-scanning mirrors the `/api/configs` pattern: dropping a new CSV in `data/nk/` makes it available to the UI with no code change.

### 8e. Explicitly out of scope

- **Wavelength-resolved absorption plot** (A(x, λ) heatmap) — visually striking but a new pane; deferred to a future spec.
- **TMM vs Beer-Lambert side-by-side toggle** on the J–V plot — users can already compare by running both presets in the same workspace and selecting both runs in the tree.
- **In-UI editing of n,k data** — CSVs are reference data, not user-editable. Custom materials require dropping a CSV in `data/nk/`.

## 9. Validation and tests

### 9a. Unit tests

`tests/unit/physics/test_optics.py` already has 13 tests for the TMM core. **Add:**

- `test_glass_substrate_incoherent` — A 1 mm glass slab with `incoherent: true` should produce no fringes vs wavelength when we sweep λ in 0.1 nm steps. The same slab without the flag should produce visible sub-nanometer ripples (the unphysical artifact we are fixing).
- `test_load_all_shipped_materials` — Loop over every CSV in `data/nk/`, verify it loads cleanly via `load_nk()`, has monotonically increasing wavelengths, and `n > 0, k >= 0` everywhere.
- `test_substrate_role_excluded_from_grid` — Build a stack with one substrate layer and one absorber layer; assert that `multilayer_grid` returns nodes only inside the absorber range.

### 9b. Integration tests

`tests/integration/test_tmm_integration.py` already has 5 tests. **Add:**

- `test_nip_tmm_preset_jsc_in_band` — Run a full J–V on `nip_MAPbI3_tmm.yaml`, assert `J_sc ∈ [180, 250] A/m²` (18–25 mA/cm² — the physically realistic band for a 400 nm MAPbI₃ cell with proper front reflection).
- `test_pin_tmm_preset_jsc_in_band` — Same for `pin_MAPbI3_tmm.yaml`.
- `test_tmm_jsc_below_beer_lambert` — Run J–V on both `nip_MAPbI3.yaml` and `nip_MAPbI3_tmm.yaml` with the same absorber thickness; assert `J_sc_tmm < J_sc_beer_lambert` (the front reflection loss must show up as a J_sc reduction).

### 9c. Regression test pinning

New file `tests/regression/test_tmm_baseline.py`:

```python
"""Pinned baselines for TMM-enabled presets.

These tests exist so a future change to the optical model — even an
intentional one — has to acknowledge the J_sc shift explicitly. If
you bump the pinned values, document why in the commit message.
"""
import pytest
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_config

# Pinned values are filled in once the implementation lands and produces
# a believable number from a manual run on `nip_MAPbI3_tmm.yaml`.
NIP_TMM_JSC_PINNED = None  # A/m² — TODO fill in during implementation
PIN_TMM_JSC_PINNED = None  # A/m²

@pytest.mark.skipif(NIP_TMM_JSC_PINNED is None, reason="awaiting first TMM run")
def test_nip_tmm_baseline_jsc():
    stack = load_config("configs/nip_MAPbI3_tmm.yaml")
    result = run_jv_sweep(stack)
    assert abs(result.metrics_fwd.J_sc - NIP_TMM_JSC_PINNED) < 5.0  # 0.5 mA/cm²
```

The pinning step is deliberately part of implementation, not the spec — there's no honest way to predict the J_sc to better than ±10% before running it.

### 9d. Manual verification checklist (post-merge)

After implementation, verify in the workstation:

- [ ] **Interference fringes appear.** Sweep absorber thickness across 50, 100, 200, 400, 600, 800 nm on `nip_MAPbI3_tmm.yaml` — J_sc should oscillate, not increase monotonically. Beer-Lambert J_sc on the same sweep should be monotonic.
- [ ] **Front reflection visible.** TMM `J_sc` should be 5–15% lower than Beer-Lambert `J_sc` at matched absorber thickness, reflecting the air→glass→FTO front-surface loss.
- [ ] **Glass `incoherent: true` actually flattens spurious 1 mm fringes.** Toggle the flag in a test config and re-run; with the flag, J_sc is smooth in λ; without it, sub-nanometer ripples appear.
- [ ] **Optical material dropdown** appears in the Device pane on full tier and is hidden on legacy/fast tiers.
- [ ] **"TMM active" badge** appears whenever any layer has `optical_material` set; disappears when the user clears it.
- [ ] **Auto-discovery works.** Drop a new CSV (e.g. copy `Ag.csv` to `data/nk/Cu.csv`) and verify it appears in the dropdown without restarting the frontend (backend restart only).

## 10. Documentation updates (mandatory deliverable)

Standing requirement: every spec from now on includes a "Documentation updates" section that touches `panels/tutorial.ts`, `panels/parameters.ts`, and `panels/algorithm.ts` whenever new physics or parameters are introduced.

### 10a. `panels/tutorial.ts`

Add a new section **"Optical Generation"** between "Physics Overview" and "Running Experiments":

> **Optical Generation: TMM vs Beer-Lambert**
>
> Generation of electron-hole pairs `G(x)` is the source term that drives the drift-diffusion equations. The simulator supports two optical models:
>
> - **Beer-Lambert** (default for legacy/fast tiers): `G(x) = α·Φ·exp(−α·x)`. Treats light as a beam being exponentially absorbed. Fast and simple, but ignores reflection at layer interfaces and wavelength dependence. Off by 5–15% on J_sc for thin-film cells.
>
> - **Transfer-Matrix Method** (full tier, when `optical_material` is set on layers): Solves Maxwell's equations across the full coherent layer stack at each wavelength of the AM1.5G spectrum, then integrates. Captures interference fringes, front-surface reflection, and wavelength-dependent absorption. Matches measured cells within ~2% on J_sc.
>
> To activate TMM, switch to **Full** tier and pick a preset whose name ends in `_tmm`, or set the `optical_material` field on every optical layer of your custom device.

### 10b. `panels/parameters.ts`

Add `optical_material` and `incoherent` to the per-layer parameter table:

| Field | Type | Tier | Description |
|---|---|---|---|
| `optical_material` | string \| null | full | Identifier matching a CSV file in `data/nk/`. When set, the layer participates in the TMM optical stack. When null, the layer is invisible to TMM and the absorber falls back to Beer-Lambert. Available materials: MAPbI3, TiO2, ITO, FTO, SnO2, spiro_OMeTAD, PCBM, C60, PEDOT_PSS, Ag, Au, glass. |
| `incoherent` | bool | full | Mark a layer as optically incoherent (e.g. glass substrate >100 µm). Skips the coherent TMM matrix product for this layer and uses bulk Beer-Lambert instead, avoiding spurious sub-nanometer interference fringes. Must be the first layer in the stack. Default: false. |
| `role: substrate` | string value | full | Marks a layer as optical-only (no electrical participation). Substrate layers are included in the TMM stack but excluded from the drift-diffusion grid and boundary conditions. Must be the first layer and must have `incoherent: true` and an `optical_material` set. |

### 10c. `panels/algorithm.ts`

Add a new subsection under the existing "Generation" heading:

> **TMM (Transfer-Matrix Method)**
>
> For each wavelength λ in the AM1.5G spectrum (300–1000 nm, 200 points):
>
> 1. Build a 2×2 dynamical matrix `D_j(n_j, k_j)` per layer encoding the air-side and substrate-side field amplitudes.
> 2. Build a propagation matrix `P_j(n_j, k_j, d_j, λ) = diag(exp(−i·δ), exp(+i·δ))` with phase `δ = 2π·n̂_j·d_j/λ`.
> 3. Multiply: `M_total = D_air⁻¹ · ∏_j (D_j · P_j · D_j⁻¹) · D_substrate`.
> 4. Reflection: `r = M[1,0] / M[0,0]`. Transmission: `t = 1 / M[0,0]`.
> 5. Position-resolved absorbed power: `A(x, λ) = (4π·n·k / λ) · |E(x, λ)|² · (n_real / n_ambient)`. The Poynting-vector correction enforces `R + T + A = 1`.
> 6. Spectral integration: `G(x) = ∫ A(x, λ) · Φ_AM1.5(λ) dλ` weighted by the photon flux.
>
> Incoherent layers (e.g. mm-thick glass substrates) bypass steps 1–4 and apply Fresnel reflection plus bulk Beer-Lambert directly.

### 10d. CLAUDE.md update

Add a one-line note under the existing "Transfer-matrix optics (TMM)" section:

> **Activated presets:** `nip_MAPbI3_tmm.yaml`, `pin_MAPbI3_tmm.yaml` (Phase 2 — Apr 2026). Both use the full optical stack (glass → TCO → ETL → MAPbI₃ → HTL → metal). The vanilla `nip_MAPbI3.yaml` and `pin_MAPbI3.yaml` remain Beer-Lambert for back-compat with existing benchmarks.

## 11. File inventory

**New files:**
- `perovskite_sim/data/nk/{ITO,FTO,SnO2,C60,PCBM,PEDOT_PSS,Ag,Au,glass}.csv` (9 files)
- `perovskite_sim/data/nk/manifest.yaml`
- `configs/nip_MAPbI3_tmm.yaml`
- `configs/pin_MAPbI3_tmm.yaml`
- `tests/regression/test_tmm_baseline.py`

**Modified files:**
- `perovskite_sim/physics/optics.py` (incoherent-layer support, ~30 lines)
- `perovskite_sim/models/parameters.py` (`incoherent: bool = False` field on `MaterialParams`)
- `perovskite_sim/models/device.py` (substrate-role filter for electrical layers, ~20 lines)
- `perovskite_sim/models/config_loader.py` (load `incoherent` and `role: substrate`)
- `perovskite_sim/solver/mol.py` (substrate-role grid filter, G(x) interpolation back to electrical grid, ~30 lines)
- `backend/main.py` (`/api/optical-materials` endpoint, ~10 lines)
- `frontend/src/api.ts` (`fetchOpticalMaterials()` wrapper)
- `frontend/src/config-editor.ts` (select-optical-material field type, ~40 lines)
- `frontend/src/workstation/tier-gating.ts` (`optical_material` and `incoherent` keys at full tier)
- `frontend/src/workstation/panes/device-pane.ts` (TMM active badge in header)
- `frontend/src/panels/tutorial.ts` (Optical Generation section)
- `frontend/src/panels/parameters.ts` (new field rows)
- `frontend/src/panels/algorithm.ts` (TMM subsection)
- `tests/unit/physics/test_optics.py` (3 new tests)
- `tests/integration/test_tmm_integration.py` (3 new tests)
- `CLAUDE.md` (activated presets note)

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| n,k data from refractiveindex.info has gaps in 300–1000 nm range | Medium | Spec requires linear interpolation + boundary clamping in `load_nk`; flag any material with >50 nm gap as a CSV header comment |
| Substrate-role filter introduces off-by-one errors in grid mapping | Medium | New unit test `test_substrate_role_excluded_from_grid`; `np.interp` between TMM and electrical grids is the only mapping path |
| FTO/Au degenerate-semiconductor parameters give unphysical V_oc or J_sc | Medium | Calibrate against published nip-MAPbI₃ cell measurements during implementation; if approximate BCs degrade quality, fall back to the original 3-layer YAML for the electrical solver and use TMM only for G(x) calculation (see fallback note below) |
| Interference fringes look "wrong" because real cells have texture/roughness | Low | Document explicitly in tutorial: TMM assumes flat planar interfaces; rough/textured cells need scattering models (out of scope) |
| Pinned regression J_sc value drifts as we tune ETL/HTL thicknesses in presets | Medium | Pin J_sc with a generous ±0.5 mA/cm² tolerance; tighten in a follow-up once preset thicknesses stabilize |
| Frontend dropdown shows materials that won't load (corrupt CSV) | Low | `load_nk` has existing validation; the dropdown can't crash because `optical_material: <bad>` errors at backend job dispatch, not at UI render |
| Glass `incoherent` flag exposes a corner case in the existing TMM solver | Medium | New unit test exercises both flag-on and flag-off paths; if a corner case appears, the fix lands in `optics.py` not the YAMLs |

**Fallback for the FTO/Au parameter risk:** if calibration shows that treating FTO/Au as degenerate semiconductors degrades the J–V quality below the existing Beer-Lambert preset (e.g. spurious series resistance from low-mobility metallic layers), the implementation can apply `role: substrate` to FTO and Au as well — keeping them in the TMM stack but excluding them from the electrical grid. The drift-diffusion solver then sees the same TiO₂/MAPbI₃/spiro 3-layer stack as the original preset, and the only thing TMM contributes is a more accurate G(x). This is a strictly better fallback because it preserves the existing electrical fidelity.

## 13. Verification checklist

After all implementation tasks complete:

- [ ] All new and existing optics unit tests pass: `pytest tests/unit/physics/test_optics.py -v`
- [ ] All TMM integration tests pass: `pytest tests/integration/test_tmm_integration.py -v`
- [ ] Regression baseline tests pinned and passing
- [ ] `pytest` full suite green (no regressions in existing tests)
- [ ] `npm run build` succeeds (TypeScript clean + production bundle)
- [ ] Manual verification checklist (§9d) all checked
- [ ] CLAUDE.md updated
- [ ] Documentation panels reviewed in browser

# TMM Optics Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the existing `perovskite_sim/physics/optics.py` TMM engine for shipped perovskite presets by adding n,k material data, substrate-role support, incoherent-layer handling, and a per-layer optical-material dropdown in the full-tier workstation UI.

**Architecture:** The TMM engine and AM1.5G loader already exist and are tested. This plan ships (1) 9 new nk CSV data files + manifest, (2) a new `incoherent` flag and `role == "substrate"` filter path that keeps substrate layers in the TMM stack but excludes them from the drift-diffusion grid, (3) two new `*_tmm.yaml` presets that leave the originals untouched, (4) a `/api/optical-materials` auto-scan endpoint, (5) a full-tier-only optical-material dropdown + "TMM active" header badge in the workstation Device pane, and (6) three documentation panels (tutorial/parameters/algorithm) plus CLAUDE.md updates.

**Tech Stack:** Python 3.13, numpy, scipy, PyYAML, pytest (TDD throughout); FastAPI backend; TypeScript/Vite/Plotly frontend with GoldenLayout workstation; existing frozen dataclasses (MaterialParams, LayerSpec, DeviceStack) must remain immutable — use `dataclasses.replace`.

**Reference spec:** `docs/superpowers/specs/2026-04-13-tmm-optics-activation-design.md`

**Out of scope:** Phase 2b layer-builder UI (add/remove/reorder), Phase 3 tandem cells, wavelength-resolved absorption heatmap pane, in-UI n,k editing.

---

## File Structure

**New files:**
- `perovskite-sim/perovskite_sim/data/nk/{ITO,FTO,SnO2,C60,PCBM,PEDOT_PSS,Ag,Au,glass}.csv` — 9 refractive-index CSVs, format `wavelength_nm,n,k`, range 300–1000 nm
- `perovskite-sim/perovskite_sim/data/nk/manifest.yaml` — provenance for each material (source, reference, wavelength range, notes)
- `perovskite-sim/configs/nip_MAPbI3_tmm.yaml` — new TMM-enabled n-i-p preset with substrate + contacts
- `perovskite-sim/configs/pin_MAPbI3_tmm.yaml` — new TMM-enabled p-i-n preset
- `perovskite-sim/tests/regression/test_tmm_baseline.py` — pinned J_sc regression for both new presets

**Modified files (Python):**
- `perovskite_sim/physics/optics.py` — add incoherent-layer support (~30 lines, new Fresnel + bulk-Beer-Lambert path for a first-layer substrate; zero impact on pure-coherent calls)
- `perovskite_sim/models/parameters.py` — add `incoherent: bool = False` field to `MaterialParams`
- `perovskite_sim/models/device.py` — helper `electrical_layers` filter that drops `role == "substrate"` layers; `compute_V_bi` unchanged
- `perovskite_sim/models/config_loader.py` — load the `incoherent` field and accept `role: substrate`
- `perovskite_sim/solver/mol.py` — `_compute_tmm_generation` walks the full layer list (including substrate) and interpolates G onto the electrical x-grid; `build_material_arrays` uses `electrical_layers` for electrical arrays; grid construction same
- `perovskite_sim/discretization/grid.py` — exposes a helper that builds the electrical grid only over non-substrate layers (or callers do the filter; see Task 5)
- `backend/main.py` — `/api/optical-materials` endpoint, ~12 lines

**Modified files (Frontend):**
- `frontend/src/api.ts` — `fetchOpticalMaterials()` typed wrapper
- `frontend/src/config-editor.ts` — `FieldKind = 'numeric' | 'select-optical-material'`, select renderer
- `frontend/src/workstation/tier-gating.ts` — add `'incoherent'` to `TMM_KEYS`
- `frontend/src/workstation/panes/device-pane.ts` — "TMM active · N layers" badge next to device name
- `frontend/src/panels/tutorial.ts` — new "Optical Generation" section
- `frontend/src/panels/parameters.ts` — rows for `optical_material`, `incoherent`, `role: substrate`
- `frontend/src/panels/algorithm.ts` — new TMM subsection under "Generation"
- `perovskite-sim/CLAUDE.md` — one-line "activated presets" note under TMM section

**Modified tests:**
- `tests/unit/physics/test_optics.py` — 3 new tests (incoherent glass, all-materials load, TMM reflection sanity with substrate)
- `tests/unit/models/test_config_loader.py` — 2 new tests (incoherent field, substrate role)
- `tests/unit/solver/test_material_arrays.py` — 1 new test (substrate excluded from electrical grid)
- `tests/integration/test_tmm_integration.py` — 3 new tests (nip preset J_sc, pin preset J_sc, TMM < BL comparison)
- `frontend/src/workstation/tier-gating.test.ts` — 1 new case for `incoherent`
- `frontend/src/workstation/phase2-smoke.test.ts` or new `tmm.smoke.test.ts` — wiring check for the optical-material dropdown

**Rationale for decomposition:** Data (Task 1-2), physics-core (Task 3-5), solver integration (Task 6), config/preset (Task 7-8), backend (Task 9), frontend (Task 10-13), docs (Task 14-16), final verification (Task 17). Each task ships one behavior end-to-end (test → impl → commit). Files that change together (config_loader + parameters field) live in the same task.

---

## Task 1: Shared nk CSV fixture + manifest scaffold

**Why first:** downstream Python + frontend tests need at least one new material CSV on disk to import. Ship the scaffold and one material (FTO) so the test suite has something to load.

**Files:**
- Create: `perovskite-sim/perovskite_sim/data/nk/FTO.csv`
- Create: `perovskite-sim/perovskite_sim/data/nk/manifest.yaml`
- Test: `perovskite-sim/tests/unit/physics/test_optics.py` (new test function)

- [ ] **Step 1: Write failing test `test_load_shipped_fto_csv`**

Add to `tests/unit/physics/test_optics.py`:

```python
def test_load_shipped_fto_csv():
    """FTO.csv should load via load_nk() with monotonic wavelengths in 300-1000 nm."""
    from perovskite_sim.data import load_nk
    import numpy as np

    wl, n, k = load_nk("FTO")
    assert wl.shape == n.shape == k.shape
    assert np.all(np.diff(wl) > 0), "wavelengths must be strictly increasing"
    assert wl[0] <= 305.0 and wl[-1] >= 995.0, "must cover 300-1000 nm range"
    assert np.all(n > 0.5) and np.all(n < 5.0), "FTO n in reasonable range"
    assert np.all(k >= 0.0), "k must be non-negative"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/unit/physics/test_optics.py::test_load_shipped_fto_csv -v`
Expected: FAIL with `FileNotFoundError: ...data/nk/FTO.csv` (or load_nk internal error).

- [ ] **Step 3: Generate FTO.csv**

Use Python (one-shot, not committed to repo as a script) to produce `perovskite_sim/data/nk/FTO.csv` with 142 rows from 300 nm to 1000 nm in 5 nm steps. Source the numerical values from Filipic et al. 2015 (Opt. Express 23, A263) — a common FTO/SnO2:F dispersion tabulation reachable via refractiveindex.info. The CSV must match the existing header exactly:

```csv
wavelength_nm,n,k
300.0,2.370,0.160
305.0,2.360,0.152
...
1000.0,1.880,0.020
```

(Numeric values listed above are illustrative — the implementer pulls the real tabulated dispersion from the Filipic source. Keep the precise floats; do not round.)

- [ ] **Step 4: Create `manifest.yaml`**

Write `perovskite_sim/data/nk/manifest.yaml` documenting every material currently in the folder (`MAPbI3`, `TiO2`, `spiro_OMeTAD`, `FTO`):

```yaml
MAPbI3:
  source: refractiveindex.info
  reference: "Phillips et al. 2018"
  wavelength_range_nm: [300, 1000]
  notes: "MAPbI3 perovskite thin film; shipped baseline"

TiO2:
  source: refractiveindex.info
  reference: "Siefke et al. 2016"
  wavelength_range_nm: [300, 1000]
  notes: "Anatase TiO2 ETL"

spiro_OMeTAD:
  source: Filipic
  reference: "Filipic et al. 2015, Opt. Express 23, A263"
  wavelength_range_nm: [300, 1000]
  notes: "Spiro-OMeTAD HTL"

FTO:
  source: refractiveindex.info
  reference: "Filipic et al. 2015, Opt. Express 23, A263"
  wavelength_range_nm: [300, 1000]
  notes: "F-doped SnO2, ~500 nm thick on glass; front transparent contact"
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/unit/physics/test_optics.py::test_load_shipped_fto_csv -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/data/nk/FTO.csv perovskite_sim/data/nk/manifest.yaml tests/unit/physics/test_optics.py
git commit -m "feat(optics): add FTO nk CSV and material manifest scaffold"
```

---

## Task 2: Ship the remaining nk CSVs (ITO, SnO2, C60, PCBM, PEDOT_PSS, Ag, Au, glass)

**Why next:** the solver-integration tests need ITO (p-i-n front contact), Ag/Au (back reflectors), and glass (substrate). Task 1 established the load path; this task broadens the library.

**Files:**
- Create: `perovskite_sim/data/nk/ITO.csv`
- Create: `perovskite_sim/data/nk/SnO2.csv`
- Create: `perovskite_sim/data/nk/C60.csv`
- Create: `perovskite_sim/data/nk/PCBM.csv`
- Create: `perovskite_sim/data/nk/PEDOT_PSS.csv`
- Create: `perovskite_sim/data/nk/Ag.csv`
- Create: `perovskite_sim/data/nk/Au.csv`
- Create: `perovskite_sim/data/nk/glass.csv`
- Modify: `perovskite_sim/data/nk/manifest.yaml` (add one block per new material)
- Test: `tests/unit/physics/test_optics.py`

- [ ] **Step 1: Write failing parametric test `test_load_all_shipped_materials`**

Add to `tests/unit/physics/test_optics.py`:

```python
import pytest

@pytest.mark.parametrize("material", [
    "MAPbI3", "TiO2", "spiro_OMeTAD",
    "FTO", "ITO", "SnO2", "C60", "PCBM", "PEDOT_PSS", "Ag", "Au", "glass",
])
def test_load_all_shipped_materials(material):
    """Every shipped nk CSV must load cleanly, cover 300-1000 nm, and have n>0, k>=0."""
    from perovskite_sim.data import load_nk
    import numpy as np

    wl, n, k = load_nk(material)
    assert np.all(np.diff(wl) > 0), f"{material}: wavelengths not monotonic"
    assert wl[0] <= 305.0 and wl[-1] >= 995.0, f"{material}: range too narrow"
    assert np.all(n > 0.0), f"{material}: n must be positive"
    assert np.all(k >= 0.0), f"{material}: k must be non-negative"
```

- [ ] **Step 2: Run test, verify it fails for new materials**

Run: `pytest tests/unit/physics/test_optics.py::test_load_all_shipped_materials -v`
Expected: 4 PASS (MAPbI3, TiO2, spiro_OMeTAD, FTO), 8 FAIL (FileNotFoundError).

- [ ] **Step 3: Generate 8 new CSVs**

Produce each file with one-shot numerical sources (all in `perovskite_sim/data/nk/`). Each CSV: `wavelength_nm,n,k` header, 5 nm steps, 300.0 to 1000.0 inclusive.

- **ITO.csv** — König et al. 2014, Opt. Mater. Express 4, 689 (Sn:In2O3, sputtered)
- **SnO2.csv** — Filipic et al. 2015
- **C60.csv** — Ren et al. 2015 (typical thin-film fullerene dispersion)
- **PCBM.csv** — Ren et al. 2015
- **PEDOT_PSS.csv** — Lee et al. 2018 (conductive formulation)
- **Ag.csv** — Johnson & Christy 1972 (k peaks around 800 nm, n very small)
- **Au.csv** — Johnson & Christy 1972
- **glass.csv** — Schott BK7 dispersion (`k ≈ 0` everywhere, `n` from Sellmeier equation evaluated at the 5 nm grid)

For each, consult refractiveindex.info or the cited primary source and keep the raw tabulated floats (no rounding beyond 4 decimals).

- [ ] **Step 4: Extend `manifest.yaml`**

Append an entry per new material with source/reference/notes fields matching the Task 1 template. Keep alphabetical order within the file for diff-ability.

- [ ] **Step 5: Run test, verify all 12 pass**

Run: `pytest tests/unit/physics/test_optics.py::test_load_all_shipped_materials -v`
Expected: 12 PASS.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/data/nk/ perovskite_sim/data/nk/manifest.yaml tests/unit/physics/test_optics.py
git commit -m "feat(optics): ship nk data for ITO, SnO2, C60, PCBM, PEDOT:PSS, Ag, Au, glass"
```

---

## Task 3: Add `incoherent` field to `MaterialParams` and wire through config loader

**Files:**
- Modify: `perovskite_sim/models/parameters.py`
- Modify: `perovskite_sim/models/config_loader.py`
- Test: `tests/unit/models/test_config_loader.py`

- [ ] **Step 1: Write failing test `test_incoherent_field_loaded_from_yaml`**

Add to `tests/unit/models/test_config_loader.py`:

```python
def test_incoherent_field_loaded_from_yaml(tmp_path):
    """The `incoherent: true` YAML field must land on MaterialParams.incoherent."""
    from perovskite_sim.models.config_loader import load_device
    yaml_text = '''
device: {V_bi: 1.1, Phi: 2.5e21}
layers:
  - name: glass
    role: substrate
    thickness: 1.0e-3
    eps_r: 2.25
    mu_n: 0.0
    mu_p: 0.0
    ni: 0.0
    N_D: 0.0
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1.0e30
    P0: 0.0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0
    p1: 1.0
    B_rad: 0.0
    C_n: 0.0
    C_p: 0.0
    alpha: 0.0
    optical_material: glass
    incoherent: true
  - name: MAPbI3
    role: absorber
    thickness: 400e-9
    eps_r: 24.1
    mu_n: 2e-4
    mu_p: 2e-4
    ni: 3.2e13
    N_D: 0.0
    N_A: 0.0
    D_ion: 1e-16
    P_lim: 1.6e27
    P0: 1.6e24
    tau_n: 1e-6
    tau_p: 1e-6
    n1: 3.2e13
    p1: 3.2e13
    B_rad: 5e-22
    C_n: 1e-42
    C_p: 1e-42
    alpha: 1.3e7
'''
    p = tmp_path / "tmm.yaml"
    p.write_text(yaml_text)
    stack = load_device(str(p))
    assert stack.layers[0].params.incoherent is True
    assert stack.layers[0].params.optical_material == "glass"
    assert stack.layers[1].params.incoherent is False  # default
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/unit/models/test_config_loader.py::test_incoherent_field_loaded_from_yaml -v`
Expected: FAIL — `MaterialParams` has no `incoherent` field (or `load_device` rejects `role: substrate`).

- [ ] **Step 3: Add the field to `MaterialParams`**

In `perovskite_sim/models/parameters.py`, insert after the existing `n_optical` field (~line 47):

```python
    # Optical coherence flag for TMM. When True, the layer is treated as
    # incoherent (bulk Beer-Lambert + Fresnel interfaces, no interference).
    # Must be True for mm-thick substrates; defaults False (coherent).
    incoherent: bool = False
```

- [ ] **Step 4: Teach the config loader to forward it**

In `perovskite_sim/models/config_loader.py`, wherever `MaterialParams(**kwargs)` is constructed from a YAML layer dict, include `incoherent` in the accepted key list. The loader already forwards `optical_material`; add `incoherent` alongside it with default `False`. Also accept `role: substrate` without raising (no special handling yet — the filter lands in Task 5).

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/unit/models/test_config_loader.py::test_incoherent_field_loaded_from_yaml -v`
Expected: PASS.

Also run the full config_loader suite to confirm no regression:
`pytest tests/unit/models/test_config_loader.py -v`

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/models/parameters.py perovskite_sim/models/config_loader.py tests/unit/models/test_config_loader.py
git commit -m "feat(models): add incoherent flag to MaterialParams; accept role:substrate"
```

---

## Task 4: Incoherent-layer support in `optics.py`

**Goal:** When the first layer in the TMM stack is incoherent, skip its coherent matrix product and apply single Fresnel reflection + bulk Beer-Lambert attenuation instead. This prevents unphysical sub-nanometer fringes through 1 mm glass.

**Files:**
- Modify: `perovskite_sim/physics/optics.py`
- Test: `tests/unit/physics/test_optics.py`

- [ ] **Step 1: Write failing test `test_glass_substrate_incoherent_suppresses_fringes`**

Add to `tests/unit/physics/test_optics.py`:

```python
def test_glass_substrate_incoherent_suppresses_fringes():
    """1 mm glass with incoherent=True must produce smooth R(lambda), not fringes."""
    from perovskite_sim.physics.optics import TMMLayer, tmm_reflectance
    from perovskite_sim.data import load_nk
    import numpy as np

    wl_nm = np.linspace(500.0, 502.0, 201)  # 0.01 nm spacing
    wl_m = wl_nm * 1e-9

    _, n_glass, k_glass = load_nk("glass", wl_nm)
    _, n_fto, k_fto = load_nk("FTO", wl_nm)

    glass = TMMLayer(d=1.0e-3, n=n_glass, k=k_glass, incoherent=True)
    fto = TMMLayer(d=500e-9, n=n_fto, k=k_fto)

    R = tmm_reflectance([glass, fto], wl_m)
    # Smooth: peak-to-peak variation over 2 nm window < 5% absolute
    assert (R.max() - R.min()) < 0.05, f"R variation {R.max()-R.min():.3f} suggests fringes"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/unit/physics/test_optics.py::test_glass_substrate_incoherent_suppresses_fringes -v`
Expected: FAIL — `TMMLayer.__init__` got unexpected keyword `incoherent` OR fringes appear without the flag honored.

- [ ] **Step 3: Add `incoherent` to `TMMLayer` and implement the bypass**

Edit `perovskite_sim/physics/optics.py`:

1. Add `incoherent: bool = False` to the `TMMLayer` dataclass fields (defaults make the existing test corpus compile unchanged).
2. At the top of `_transfer_matrix_stack`, detect `layers[0].incoherent is True`; if any other layer has `incoherent=True`, raise `ValueError("only the first layer may be incoherent")`.
3. When the first layer is incoherent, do NOT include it in the matrix product. Instead:
   - Compute the air→glass Fresnel reflection `r0 = (n_air - n_glass)/(n_air + n_glass)` (wavelength array).
   - Compute the glass→next-layer Fresnel reflection `r1 = (n_glass - n_next)/(n_glass + n_next)`.
   - Bulk transmission through the glass: `T_bulk = exp(-4*pi*k_glass*d_glass / lambda)` (power form, not field — no phase).
   - The coherent stack starts with `n_ambient = n_glass` and runs over `layers[1:]`; the light entering that sub-stack has amplitude `sqrt(T_bulk * (1 - |r0|^2))` (approximation: intensity scale + phase reset).
4. `_electric_field_profile` for the incoherent layer: set `E_sq(x)` in that layer to `(1 - R_ambient_glass) * exp(-2*alpha_glass*x_local)` where `alpha_glass = 4*pi*k_glass/lambda`; this gives a monotonic decay matching bulk Beer-Lambert.
5. Energy conservation: the substrate contributes a fractional absorption `(1 - R_front) * (1 - T_bulk)` which must be counted in the totals (so that `R+T+A = 1`) — but since the substrate layer is optically inert for G(x) in later stages, we only need correct transmission to the next layer.

Keep the change ~30 lines. Add inline comments explaining why the power-form treatment is used instead of amplitude.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/unit/physics/test_optics.py::test_glass_substrate_incoherent_suppresses_fringes -v`
Expected: PASS.

Also run the full optics suite (13 existing + new) to confirm no regression on the pure-coherent path:
`pytest tests/unit/physics/test_optics.py -v`
Expected: all green.

- [ ] **Step 5: Add validator test `test_incoherent_not_first_raises`**

```python
def test_incoherent_not_first_raises():
    from perovskite_sim.physics.optics import TMMLayer, _transfer_matrix_stack
    import numpy as np
    wl = np.linspace(400e-9, 800e-9, 10)
    ones = np.ones(10)
    bad = [TMMLayer(100e-9, ones*1.5, ones*0.0), TMMLayer(1e-3, ones*1.5, ones*0.0, incoherent=True)]
    import pytest
    with pytest.raises(ValueError, match="first layer"):
        _transfer_matrix_stack(bad, wl)
```

Run: `pytest tests/unit/physics/test_optics.py::test_incoherent_not_first_raises -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add perovskite_sim/physics/optics.py tests/unit/physics/test_optics.py
git commit -m "feat(optics): support one incoherent first layer in TMM stack"
```

---

## Task 5: `role == "substrate"` electrical filter in grid + material arrays

**Goal:** TMM sees the full stack (substrate → contact → ETL → absorber → HTL → back contact). The drift-diffusion solver sees only the non-substrate layers. `_compute_tmm_generation` interpolates G(x) from the TMM grid back onto the electrical grid by offsetting x by the substrate's cumulative thickness.

**Files:**
- Modify: `perovskite_sim/models/device.py`
- Modify: `perovskite_sim/solver/mol.py`
- Test: `tests/unit/solver/test_material_arrays.py` (new test), `tests/unit/physics/test_optics.py` (new test)

- [ ] **Step 1: Write failing test `test_substrate_role_excluded_from_electrical_grid`**

Add to `tests/unit/solver/test_material_arrays.py` (create file if it doesn't exist):

```python
def test_substrate_role_excluded_from_electrical_grid():
    """role: substrate must not appear in the electrical grid x-range."""
    from dataclasses import replace
    from perovskite_sim.models.device import DeviceStack, LayerSpec
    from perovskite_sim.models.parameters import MaterialParams
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.solver.mol import build_material_arrays

    # Minimal params — use existing nip_MAPbI3 MAPbI3 params for the absorber
    from perovskite_sim.models.config_loader import load_device
    real = load_device("configs/nip_MAPbI3.yaml")
    absorber_params = real.layers[1].params  # MAPbI3

    glass_p = MaterialParams(
        eps_r=2.25, mu_n=0, mu_p=0, D_ion=0, P_lim=1e30, P0=0,
        ni=0, tau_n=1e-9, tau_p=1e-9, n1=1, p1=1, B_rad=0,
        C_n=0, C_p=0, alpha=0, N_A=0, N_D=0,
        optical_material="glass", incoherent=True,
    )

    stack = DeviceStack(layers=(
        LayerSpec("glass", 1.0e-3, glass_p, "substrate"),
        LayerSpec("MAPbI3", 400e-9, absorber_params, "absorber"),
    ))
    from perovskite_sim.models.device import electrical_layers
    elec = electrical_layers(stack)
    assert len(elec) == 1 and elec[0].name == "MAPbI3"

    # Grid should span only the absorber (0 .. 400 nm), not 0 .. 1 mm + 400 nm
    import numpy as np
    layers_grid = [Layer(l.thickness, 30) for l in elec]
    x = multilayer_grid(layers_grid)
    assert x[0] >= 0.0
    assert x[-1] <= 400e-9 + 1e-15
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/unit/solver/test_material_arrays.py::test_substrate_role_excluded_from_electrical_grid -v`
Expected: FAIL — `electrical_layers` does not exist.

- [ ] **Step 3: Add `electrical_layers(stack)` to `device.py`**

In `perovskite_sim/models/device.py`, after the `DeviceStack` class, add:

```python
def electrical_layers(stack: "DeviceStack") -> tuple["LayerSpec", ...]:
    """Return layers that participate in the drift-diffusion solve.

    Layers with role == "substrate" are optical-only and skipped. The TMM
    optical path still walks stack.layers (full list); only the electrical
    path uses this filtered view.
    """
    return tuple(l for l in stack.layers if l.role != "substrate")
```

- [ ] **Step 4: Run test, verify Step 1 passes**

Run: `pytest tests/unit/solver/test_material_arrays.py::test_substrate_role_excluded_from_electrical_grid -v`
Expected: PASS.

- [ ] **Step 5: Write failing test `test_tmm_generation_zero_on_substrate_nonzero_on_absorber`**

Add to `tests/unit/physics/test_optics.py`:

```python
def test_tmm_generation_maps_back_to_electrical_grid():
    """_compute_tmm_generation must return G array sized to the electrical grid."""
    from perovskite_sim.solver.mol import _compute_tmm_generation
    from perovskite_sim.models.device import DeviceStack, LayerSpec, electrical_layers
    from perovskite_sim.models.parameters import MaterialParams
    from perovskite_sim.discretization.grid import multilayer_grid, Layer
    from perovskite_sim.models.config_loader import load_device
    import numpy as np

    real = load_device("configs/nip_MAPbI3.yaml")
    absorber_params = real.layers[1].params
    # give MAPbI3 an optical_material so TMM activates
    from dataclasses import replace
    absorber_params = replace(absorber_params, optical_material="MAPbI3")

    glass_p = MaterialParams(
        eps_r=2.25, mu_n=0, mu_p=0, D_ion=0, P_lim=1e30, P0=0,
        ni=0, tau_n=1e-9, tau_p=1e-9, n1=1, p1=1, B_rad=0,
        C_n=0, C_p=0, alpha=0, N_A=0, N_D=0,
        optical_material="glass", incoherent=True,
    )
    stack = DeviceStack(layers=(
        LayerSpec("glass", 1.0e-3, glass_p, "substrate"),
        LayerSpec("MAPbI3", 400e-9, absorber_params, "absorber"),
    ))

    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, 30) for l in elec]
    x = multilayer_grid(layers_grid)  # spans 0..400 nm
    G = _compute_tmm_generation(x, stack)

    assert G is not None
    assert G.shape == x.shape
    assert np.all(G >= 0.0)
    assert G.max() > 1e24  # reasonable photogeneration order for MAPbI3 under AM1.5
```

- [ ] **Step 6: Run test, verify it fails**

Run: `pytest tests/unit/physics/test_optics.py::test_tmm_generation_maps_back_to_electrical_grid -v`
Expected: FAIL — either G is the wrong shape (full stack) or the call errors on substrate layer.

- [ ] **Step 7: Update `_compute_tmm_generation` in `mol.py`**

Modify `perovskite_sim/solver/mol.py:_compute_tmm_generation` (around line 138):

1. Detect whether any layer has `role == "substrate"`. If so, compute a *TMM x-grid* that runs from 0 to `total_thickness_including_substrate`, and remember `substrate_offset = sum(l.thickness for l in stack.layers if l.role == "substrate")`.
2. Build `tmm_layers` from `stack.layers` (full list, including substrate). Respect each layer's `incoherent` flag by passing it to `TMMLayer`.
3. Build `boundaries` from `stack.layers` (full list).
4. Create a *TMM spatial grid* `x_tmm` that covers 0..total_thickness and contains the electrical `x` values shifted by `+substrate_offset`. Simplest: `x_tmm = x + substrate_offset` (then include at least one extra node in each substrate region for A computation accuracy — in practice `tmm_absorption_profile` only needs the evaluation points we care about, which is the electrical grid).
5. Call `tmm_generation(tmm_layers, wavelengths_m, spectral_flux, x_tmm, boundaries)`.
6. Return the resulting `G` array (already sized to `x`).

Because `tmm_absorption_profile` uses `layer_boundaries` to route each query x to its TMM layer, passing shifted x values lands them inside the non-substrate layers automatically. No interpolation needed.

- [ ] **Step 8: Also update `build_material_arrays` to use `electrical_layers`**

In `build_material_arrays` (the `for layer in stack.layers:` loop around line 243), replace `stack.layers` with `electrical_layers(stack)` so that substrate layers never contribute to the electrical per-node arrays. The `_compute_tmm_generation` call at line 367 still receives `stack` (full list).

- [ ] **Step 9: Run tests, verify pass**

Run:
```
pytest tests/unit/physics/test_optics.py::test_tmm_generation_maps_back_to_electrical_grid -v
pytest tests/unit/physics/test_optics.py -v
pytest tests/unit/solver/ -v
```
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add perovskite_sim/models/device.py perovskite_sim/solver/mol.py tests/unit/physics/test_optics.py tests/unit/solver/test_material_arrays.py
git commit -m "feat(solver): exclude role:substrate layers from electrical grid; TMM sees full stack"
```

---

## Task 6: Full-suite regression guard

**Why:** Tasks 3-5 touched the solver hot path. Before adding new presets, confirm every existing preset still runs and every existing test still passes.

**Files:** (none modified in this task — pure verification)

- [ ] **Step 1: Run the non-slow pytest suite**

Run: `pytest` (from repo root; picks up `pyproject.toml` defaults including `-m 'not slow'`).
Expected: all green. Any failure here indicates a regression from Task 5 — debug before proceeding.

- [ ] **Step 2: Run one slow benchmark to confirm the solver hot path**

Run: `pytest tests/integration/test_ionmonger_benchmark.py -v -m slow` (or whichever the ionmonger regression is).
Expected: pinned metrics still match within tolerance.

- [ ] **Step 3: No commit (verification only)**

If both runs are green, move to Task 7. If either fails, stop and fix before adding new presets.

---

## Task 7: New `configs/nip_MAPbI3_tmm.yaml` preset + integration test

**Files:**
- Create: `perovskite-sim/configs/nip_MAPbI3_tmm.yaml`
- Test: `tests/integration/test_tmm_integration.py`

- [ ] **Step 1: Write failing test `test_nip_tmm_preset_jsc_in_band`**

Add to `tests/integration/test_tmm_integration.py`:

```python
def test_nip_tmm_preset_jsc_in_band():
    """Full J-V on nip_MAPbI3_tmm.yaml must give J_sc in the realistic 180-260 A/m^2 band."""
    from perovskite_sim.models.config_loader import load_device
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep

    stack = load_device("configs/nip_MAPbI3_tmm.yaml")
    result = run_jv_sweep(stack, n_points=21)
    assert 180.0 <= result.metrics_fwd.J_sc <= 260.0, (
        f"J_sc={result.metrics_fwd.J_sc:.1f} A/m² out of band [180, 260]"
    )
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/integration/test_tmm_integration.py::test_nip_tmm_preset_jsc_in_band -v`
Expected: FAIL — `FileNotFoundError: configs/nip_MAPbI3_tmm.yaml`.

- [ ] **Step 3: Write `configs/nip_MAPbI3_tmm.yaml`**

Create the new preset. Layer order follows the existing `nip_MAPbI3.yaml` electrical convention (HTL → absorber → ETL: `phi_left=0` at HTL, `phi_right=V_bi` at ETL) but **prepends** a substrate layer in front of the HTL for optical-only reasons. Light physically enters from the substrate side in the TMM path (x=0 is the substrate surface).

Because the TMM path walks `stack.layers` in order and treats ambient→layer[0] as the sun-side interface, and because the existing nip stack is HTL-on-the-left, this plan places the substrate on the HTL side. This is an intentional departure from the spec's sun-side-first wording; it preserves the existing V_bi/band-offset convention so no electrical calibration is required. Document the choice in a YAML comment:

```yaml
# nip_MAPbI3_tmm — TMM-enabled variant of nip_MAPbI3.
#
# Layer order matches nip_MAPbI3.yaml (HTL → absorber → ETL) for electrical
# back-compat: phi_left=0 (p-contact) at HTL, phi_right=V_bi at ETL.
# Light enters from x=0 (HTL side) for the purposes of the TMM stack, with
# a 1 mm glass substrate prepended to account for the front Fresnel
# reflection (physically realistic for a device measured through the HTL
# contact — which is how inverted cells are characterized).
#
# See 2026-04-13-tmm-optics-activation-design.md for rationale.
device:
  V_bi: 1.1
  Phi: 2.5e21
  mode: full

layers:
  - name: glass
    role: substrate
    thickness: 1.0e-3
    eps_r: 2.25
    mu_n: 0.0
    mu_p: 0.0
    ni: 0.0
    N_D: 0.0
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1.0e30
    P0: 0.0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0
    p1: 1.0
    B_rad: 0.0
    C_n: 0.0
    C_p: 0.0
    alpha: 0.0
    optical_material: glass
    incoherent: true

  - name: spiro_HTL
    role: HTL
    thickness: 200e-9
    eps_r: 3.0
    mu_n: 1e-10
    mu_p: 1e-6
    ni: 1e0
    N_D: 0.0
    N_A: 2e23
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0
    optical_material: spiro_OMeTAD

  - name: MAPbI3
    role: absorber
    thickness: 400e-9
    eps_r: 24.1
    mu_n: 2e-4
    mu_p: 2e-4
    ni: 3.2e13
    N_D: 0.0
    N_A: 0.0
    D_ion: 1e-16
    P_lim: 1.6e27
    P0: 1.6e24
    tau_n: 1e-6
    tau_p: 1e-6
    n1: 3.2e13
    p1: 3.2e13
    B_rad: 5e-22
    C_n: 1e-42
    C_p: 1e-42
    alpha: 1.3e7
    optical_material: MAPbI3

  - name: TiO2_ETL
    role: ETL
    thickness: 100e-9
    eps_r: 10.0
    mu_n: 1e-5
    mu_p: 1e-10
    ni: 1e0
    N_D: 1e24
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1e30
    P0: 0.0
    tau_n: 1e-9
    tau_p: 1e-9
    n1: 1e0
    p1: 1e0
    B_rad: 1e-30
    C_n: 1e-42
    C_p: 1e-42
    alpha: 0.0
    optical_material: TiO2
```

**Note:** this preset keeps the FTO/Au contact-metal concern *out of scope*. The electrical solve runs on the same HTL/MAPbI3/ETL 3-layer stack as `nip_MAPbI3.yaml`, and TMM contributes only the front-surface reflection + more-accurate G(x) across the absorber. This sidesteps the degenerate-semiconductor contact calibration risk the spec flagged.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/integration/test_tmm_integration.py::test_nip_tmm_preset_jsc_in_band -v`
Expected: PASS. If J_sc falls outside [180, 260], print the value, inspect the G(x) profile, and debug before loosening bounds. A reasonable TMM number for 400 nm MAPbI3 with front glass reflection is ~220 A/m².

- [ ] **Step 5: Commit**

```bash
git add configs/nip_MAPbI3_tmm.yaml tests/integration/test_tmm_integration.py
git commit -m "feat(configs): ship nip_MAPbI3_tmm.yaml with glass substrate + TMM optics"
```

---

## Task 8: New `configs/pin_MAPbI3_tmm.yaml` preset + integration test

**Files:**
- Create: `perovskite-sim/configs/pin_MAPbI3_tmm.yaml`
- Test: `tests/integration/test_tmm_integration.py`

- [ ] **Step 1: Write failing test `test_pin_tmm_preset_jsc_in_band`**

Add to `tests/integration/test_tmm_integration.py`:

```python
def test_pin_tmm_preset_jsc_in_band():
    from perovskite_sim.models.config_loader import load_device
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    stack = load_device("configs/pin_MAPbI3_tmm.yaml")
    result = run_jv_sweep(stack, n_points=21)
    assert 180.0 <= result.metrics_fwd.J_sc <= 260.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/integration/test_tmm_integration.py::test_pin_tmm_preset_jsc_in_band -v`
Expected: FAIL — file not found.

- [ ] **Step 3: Write `configs/pin_MAPbI3_tmm.yaml`**

Use `configs/pin_MAPbI3.yaml` as the electrical baseline and prepend a `glass` substrate + map each existing layer's `optical_material`: p-i-n stacks typically use PEDOT:PSS (HTL) or NiOx, and C60/PCBM (ETL). Match whatever the existing `pin_MAPbI3.yaml` uses and set `optical_material` appropriately (PEDOT_PSS, MAPbI3, C60, or PCBM). Keep the same electrical parameters as the existing file — this task only adds substrate + `optical_material` fields.

Include the same header comment as Task 7 explaining the layer-order choice.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/integration/test_tmm_integration.py::test_pin_tmm_preset_jsc_in_band -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/pin_MAPbI3_tmm.yaml tests/integration/test_tmm_integration.py
git commit -m "feat(configs): ship pin_MAPbI3_tmm.yaml with glass substrate + TMM optics"
```

---

## Task 9: TMM-vs-Beer-Lambert comparison test

**Why:** Prove the new preset reflects the physically expected 5–15% J_sc reduction from front-surface reflection that TMM captures and Beer-Lambert misses.

**Files:**
- Test: `tests/integration/test_tmm_integration.py`

- [ ] **Step 1: Write failing test `test_tmm_jsc_below_beer_lambert`**

```python
def test_tmm_jsc_below_beer_lambert():
    """TMM preset should give J_sc 2-20% below Beer-Lambert preset at same absorber thickness."""
    from perovskite_sim.models.config_loader import load_device
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    bl = run_jv_sweep(load_device("configs/nip_MAPbI3.yaml"), n_points=21)
    tmm = run_jv_sweep(load_device("configs/nip_MAPbI3_tmm.yaml"), n_points=21)
    ratio = tmm.metrics_fwd.J_sc / bl.metrics_fwd.J_sc
    assert 0.80 <= ratio <= 0.98, (
        f"TMM/BL J_sc ratio {ratio:.3f} outside expected 0.80-0.98 window "
        f"(TMM {tmm.metrics_fwd.J_sc:.1f}, BL {bl.metrics_fwd.J_sc:.1f})"
    )
```

- [ ] **Step 2: Run test, expect PASS (both presets now exist)**

Run: `pytest tests/integration/test_tmm_integration.py::test_tmm_jsc_below_beer_lambert -v`
Expected: PASS. If it fails because the ratio is >0.98, the TMM path is probably bypassing the substrate reflection; investigate `_compute_tmm_generation`. If <0.80, check whether the glass thickness or incoherent handling is over-attenuating.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_tmm_integration.py
git commit -m "test(tmm): enforce TMM J_sc below Beer-Lambert within 2-20% window"
```

---

## Task 10: Regression baseline pinning (`tests/regression/test_tmm_baseline.py`)

**Files:**
- Create: `tests/regression/test_tmm_baseline.py`

- [ ] **Step 1: Run the new TMM presets manually to extract J_sc values**

```bash
python -c "
from perovskite_sim.models.config_loader import load_device
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
for name in ['nip_MAPbI3_tmm', 'pin_MAPbI3_tmm']:
    stack = load_device(f'configs/{name}.yaml')
    r = run_jv_sweep(stack, n_points=21)
    print(f'{name}: J_sc = {r.metrics_fwd.J_sc:.2f} A/m^2')
"
```

Record the two numbers (call them `NIP_TMM_JSC` and `PIN_TMM_JSC`).

- [ ] **Step 2: Write the regression test file**

Create `tests/regression/test_tmm_baseline.py`:

```python
"""Pinned J_sc baselines for TMM-enabled presets.

If these numbers drift, the optical model changed — document the shift
in the commit message and update the pins intentionally.
"""
from perovskite_sim.models.config_loader import load_device
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

NIP_TMM_JSC_PINNED = 0.0  # <-- fill with the Step 1 value
PIN_TMM_JSC_PINNED = 0.0  # <-- fill with the Step 1 value
TOLERANCE = 5.0  # A/m² (= 0.5 mA/cm²)


def test_nip_tmm_baseline_jsc():
    r = run_jv_sweep(load_device("configs/nip_MAPbI3_tmm.yaml"), n_points=21)
    assert abs(r.metrics_fwd.J_sc - NIP_TMM_JSC_PINNED) < TOLERANCE


def test_pin_tmm_baseline_jsc():
    r = run_jv_sweep(load_device("configs/pin_MAPbI3_tmm.yaml"), n_points=21)
    assert abs(r.metrics_fwd.J_sc - PIN_TMM_JSC_PINNED) < TOLERANCE
```

- [ ] **Step 3: Run tests, verify pass**

Run: `pytest tests/regression/test_tmm_baseline.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_tmm_baseline.py
git commit -m "test(regression): pin TMM preset J_sc baselines"
```

---

## Task 11: Backend `/api/optical-materials` endpoint

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_main.py` (or wherever FastAPI endpoints are tested — check repo for existing pattern)

- [ ] **Step 1: Write failing test `test_optical_materials_endpoint`**

Add to the backend test file:

```python
def test_optical_materials_endpoint(client):
    resp = client.get("/api/optical-materials")
    assert resp.status_code == 200
    body = resp.json()
    assert "materials" in body
    assert "MAPbI3" in body["materials"]
    assert "FTO" in body["materials"]
    assert "glass" in body["materials"]
    assert "manifest" not in body["materials"]  # manifest.yaml filtered out
    # Alphabetical
    assert body["materials"] == sorted(body["materials"])
```

(If no `client` fixture exists, use `fastapi.testclient.TestClient(app)` directly.)

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest backend/tests/test_main.py::test_optical_materials_endpoint -v`
Expected: FAIL — 404.

- [ ] **Step 3: Add the endpoint**

In `backend/main.py`, after `/api/configs`:

```python
from pathlib import Path as _Path

@app.get("/api/optical-materials")
def list_optical_materials() -> dict:
    """Auto-scan perovskite_sim/data/nk/ and return the sorted material list."""
    nk_dir = _Path(__file__).resolve().parent.parent / "perovskite_sim" / "data" / "nk"
    materials = sorted(
        p.stem for p in nk_dir.glob("*.csv")
    )
    return {"materials": materials}
```

(No need to exclude `manifest` — it's `manifest.yaml`, not `.csv`, so `glob("*.csv")` already filters it.)

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest backend/tests/test_main.py::test_optical_materials_endpoint -v`
Expected: PASS.

- [ ] **Step 5: Restart uvicorn and smoke-check in the browser**

Per `CLAUDE.md`, `uvicorn --reload` occasionally misses backend edits under `perovskite-sim/backend/`. Kill the dev server and restart:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload
```

Then: `curl http://127.0.0.1:8000/api/optical-materials | jq` — expect a JSON array of 12 materials alphabetically sorted.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/test_main.py
git commit -m "feat(backend): add /api/optical-materials auto-scan endpoint"
```

---

## Task 12: Frontend `fetchOpticalMaterials` + tier-gating flag

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/workstation/tier-gating.ts`
- Modify: `frontend/src/workstation/tier-gating.test.ts`

- [ ] **Step 1: Write failing test for tier gating**

Add to `tier-gating.test.ts`:

```ts
test('incoherent field is hidden in legacy and fast, visible in full', () => {
  expect(isFieldVisible('incoherent', 'legacy')).toBe(false)
  expect(isFieldVisible('incoherent', 'fast')).toBe(false)
  expect(isFieldVisible('incoherent', 'full')).toBe(true)
})
```

- [ ] **Step 2: Run test, verify it fails**

Run: `npm run test -- tier-gating.test.ts` (or however the vitest config names it).
Expected: FAIL — `isFieldVisible('incoherent', 'fast')` returns `true`.

- [ ] **Step 3: Add `'incoherent'` to `TMM_KEYS`**

In `frontend/src/workstation/tier-gating.ts`, update the constant:

```ts
const TMM_KEYS = ['optical_material', 'n_optical', 'incoherent'] as const
```

- [ ] **Step 4: Run test, verify it passes**

Run: `npm run test -- tier-gating.test.ts`
Expected: PASS.

- [ ] **Step 5: Add `fetchOpticalMaterials` to `api.ts`**

```ts
export async function fetchOpticalMaterials(): Promise<string[]> {
  const resp = await fetch('http://127.0.0.1:8000/api/optical-materials')
  if (!resp.ok) {
    throw new Error(`fetchOpticalMaterials failed: ${resp.status}`)
  }
  const body = (await resp.json()) as { materials: string[] }
  return body.materials
}
```

Export it from the api module alongside `getConfig` / `listConfigs`.

- [ ] **Step 6: Run typecheck**

Run: `cd frontend && npm run build`
Expected: tsc green, bundle produced.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.ts frontend/src/workstation/tier-gating.ts frontend/src/workstation/tier-gating.test.ts
git commit -m "feat(frontend): tier-gate incoherent field; add fetchOpticalMaterials api"
```

---

## Task 13: `select-optical-material` field type in `config-editor.ts`

**Files:**
- Modify: `frontend/src/config-editor.ts`
- (Test: manual smoke check — there is no unit-test harness for config-editor rendering; the new code path is exercised by the existing phase2-smoke.test.ts wiring test)

- [ ] **Step 1: Read the current field definition in `config-editor.ts`**

Read: `frontend/src/config-editor.ts` (full file). Locate where per-layer fields are described (look for strings like `'optical_material'` or the existing per-layer grouping under "Optics & Ions"). Note the field-definition shape and how numeric fields are rendered.

- [ ] **Step 2: Introduce the `FieldKind` discriminator**

Near the top of `config-editor.ts`, add:

```ts
type FieldKind = 'numeric' | 'select-optical-material' | 'boolean'

interface FieldDef {
  key: string
  label: string
  kind: FieldKind
  unit?: string
  step?: number
}
```

Extend the existing field list so that `optical_material` is a `select-optical-material` and `incoherent` is a `boolean`. Leave all other fields as `'numeric'`.

- [ ] **Step 3: Populate the optical-material option list once per mount**

At `mountConfigEditor` (or equivalent entry point), call `fetchOpticalMaterials()` once and cache the result in a module-level variable `opticalMaterialOptions: string[]`. Render each `select-optical-material` field as:

```ts
function renderOpticalMaterialSelect(layerIdx: number, currentValue: string | null | undefined): string {
  const options = ['<option value="">(none — Beer-Lambert)</option>']
    .concat(opticalMaterialOptions.map(m =>
      `<option value="${m}" ${currentValue === m ? 'selected' : ''}>${m}</option>`))
    .join('')
  return `<select data-layer="${layerIdx}" data-field="optical_material">${options}</select>`
}
```

Wire the `change` event to the same update path that numeric fields use (immutable update via spread, saving to workspace state).

- [ ] **Step 4: Render the `incoherent` boolean as a checkbox**

Similarly for `incoherent`: a `<input type="checkbox">` that toggles the boolean and fires the same update path.

- [ ] **Step 5: Gate both fields behind `isFieldVisible(key, tier)`**

Before rendering either field, guard with `if (!isFieldVisible('optical_material', tier)) return ''` (and same for `'incoherent'`). This hides the controls on legacy/fast tiers.

- [ ] **Step 6: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: green.

- [ ] **Step 7: Manual smoke**

Start frontend (`npm run dev`), open workstation, launch with a full-tier device, and confirm:
- The Optics & Ions group in the layer editor shows an "Optical material" dropdown with all 12 materials.
- Selecting a material updates workspace state (refresh the page; the selection persists).
- Switching the device tier to fast or legacy hides the dropdown.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/config-editor.ts
git commit -m "feat(frontend): per-layer optical_material dropdown + incoherent checkbox (full tier)"
```

---

## Task 14: "TMM active · N layers" badge in `device-pane.ts`

**Files:**
- Modify: `frontend/src/workstation/panes/device-pane.ts`

- [ ] **Step 1: Read the current device-pane header code**

Read: `frontend/src/workstation/panes/device-pane.ts`. Find where the device name is rendered in the pane header.

- [ ] **Step 2: Add a helper `computeTmmBadge(device)`**

```ts
function computeTmmBadge(device: Device): string {
  if (device.tier !== 'full') return ''
  const tmmLayers = device.config.layers.filter(l => l.optical_material && l.optical_material !== '')
  if (tmmLayers.length === 0) return ''
  return `<span class="tmm-badge" title="Optical generation computed with transfer-matrix method. Layers without optical_material fall back to Beer-Lambert.">TMM active · ${tmmLayers.length} layers</span>`
}
```

- [ ] **Step 3: Insert into the existing header rendering**

Find the header line that renders the device name. Append `${computeTmmBadge(activeDevice)}` after the name. Add a minimal CSS rule in the existing stylesheet so the badge is visually distinct (small pill, accent color).

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: green.

- [ ] **Step 5: Manual smoke**

With `nip_MAPbI3_tmm` loaded as the active device on full tier, verify the badge shows "TMM active · 4 layers" (glass, spiro, MAPbI3, TiO2 all have `optical_material` set). Switch to the vanilla `nip_MAPbI3` preset — the badge should disappear. Switch tier to fast — also disappears.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/workstation/panes/device-pane.ts frontend/src/style.css
git commit -m "feat(workstation): add TMM active badge to Device pane header"
```

---

## Task 15: `panels/tutorial.ts` — "Optical Generation" section

**Files:**
- Modify: `frontend/src/panels/tutorial.ts`

- [ ] **Step 1: Read the current tutorial panel**

Read: `frontend/src/panels/tutorial.ts`. Find the section structure ("Physics Overview", "Running Experiments", etc.) and match the existing template.

- [ ] **Step 2: Insert a new "Optical Generation" section**

Between "Physics Overview" and "Running Experiments", add:

```ts
// (match the existing string-literal HTML pattern used by other sections)
const OPTICAL_SECTION = `
<h2>Optical Generation: TMM vs Beer-Lambert</h2>
<p>
  Generation of electron-hole pairs <code>G(x)</code> is the source term
  that drives the drift-diffusion equations. The simulator supports two
  optical models:
</p>
<ul>
  <li><strong>Beer-Lambert</strong> (default on legacy/fast tiers):
    <code>G(x) = α·Φ·exp(−α·x)</code>. Simple and fast, but ignores
    reflection at layer interfaces and wavelength dependence. Typically
    overestimates <code>J_sc</code> by 5–15%.</li>
  <li><strong>Transfer-Matrix Method</strong> (full tier, when
    <code>optical_material</code> is set on layers): Solves Maxwell's
    equations across the coherent layer stack at each wavelength of the
    AM1.5G spectrum, then integrates. Captures interference fringes,
    front-surface reflection, and wavelength-dependent absorption.</li>
</ul>
<p>
  To activate TMM, switch to <strong>Full</strong> tier and pick a preset
  whose name ends in <code>_tmm</code>, or set the
  <code>optical_material</code> field on every optical layer of your
  custom device.
</p>
`
```

Wire it into the section list (match whatever ordering/accumulation the existing file uses).

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: green.

- [ ] **Step 4: Manual smoke**

Open the Tutorial pane in the browser. Confirm the new section renders between Physics Overview and Running Experiments.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/panels/tutorial.ts
git commit -m "docs(tutorial): add Optical Generation section covering TMM vs Beer-Lambert"
```

---

## Task 16: `panels/parameters.ts` — document `optical_material`, `incoherent`, `role: substrate`

**Files:**
- Modify: `frontend/src/panels/parameters.ts`

- [ ] **Step 1: Read the current parameter table**

Read: `frontend/src/panels/parameters.ts`. Find the per-layer parameter table and match its row format.

- [ ] **Step 2: Append three new rows**

| Field | Type | Tier | Description |
|---|---|---|---|
| `optical_material` | `string \| null` | full | Identifier matching a CSV in `data/nk/`. When set, the layer participates in the TMM stack. When `null`, the layer is invisible to TMM and the absorber falls back to Beer-Lambert. Available: MAPbI3, TiO2, spiro_OMeTAD, FTO, ITO, SnO2, C60, PCBM, PEDOT_PSS, Ag, Au, glass. |
| `incoherent` | `bool` | full | Mark a layer as optically incoherent (glass substrate >100 µm). Uses bulk Beer-Lambert for that layer to avoid spurious sub-nanometer fringes. Must be the first layer in the stack. Default: `false`. |
| `role: substrate` | role value | full | Marks a layer as optical-only (no electrical participation). Substrate layers are included in the TMM stack but excluded from the drift-diffusion grid and boundary conditions. Must be the first layer and must have `incoherent: true` + an `optical_material` set. |

Encode each row in whatever HTML/string template the existing panel uses.

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: green.

- [ ] **Step 4: Manual smoke**

Open the Parameters pane, confirm the three new rows render with the expected descriptions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/panels/parameters.ts
git commit -m "docs(parameters): document optical_material, incoherent, role:substrate fields"
```

---

## Task 17: `panels/algorithm.ts` — TMM algorithm subsection

**Files:**
- Modify: `frontend/src/panels/algorithm.ts`

- [ ] **Step 1: Read the existing algorithm panel**

Read: `frontend/src/panels/algorithm.ts`. Locate the existing "Generation" heading and match its subsection format.

- [ ] **Step 2: Append a new "TMM (Transfer-Matrix Method)" subsection**

Content (HTML template matching the existing pattern):

```ts
const TMM_ALGORITHM = `
<h3>TMM (Transfer-Matrix Method)</h3>
<p>For each wavelength λ in the AM1.5G spectrum (300–1000 nm, 200 points):</p>
<ol>
  <li>Build a 2×2 dynamical matrix <code>D_j(n_j, k_j)</code> per layer
      encoding the air-side and substrate-side field amplitudes.</li>
  <li>Build a propagation matrix
      <code>P_j = diag(exp(−iδ), exp(+iδ))</code> with phase
      <code>δ = 2π·n̂_j·d_j/λ</code>.</li>
  <li>Multiply:
      <code>M_total = D_air⁻¹ · ∏_j (D_j · P_j · D_j⁻¹) · D_substrate</code>.</li>
  <li>Reflection: <code>r = M[1,0] / M[0,0]</code>. Transmission:
      <code>t = 1 / M[0,0]</code>.</li>
  <li>Absorbed power density:
      <code>A(x, λ) = (4π·n·k / λ) · |E(x, λ)|² · (n_real / n_ambient)</code>.
      The Poynting-vector correction enforces <code>R + T + A = 1</code>.</li>
  <li>Spectral integration:
      <code>G(x) = ∫ A(x, λ) · Φ_AM1.5(λ) dλ</code>.</li>
</ol>
<p>
  Incoherent layers (e.g. mm-thick glass substrates) bypass steps 1–4 and
  apply Fresnel reflection plus bulk Beer-Lambert directly, avoiding
  unphysical sub-nanometer interference fringes from the matrix product.
</p>
`
```

Wire into the panel's section accumulator.

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npm run build`
Expected: green.

- [ ] **Step 4: Manual smoke**

Open the Algorithm pane, confirm the new subsection renders under the existing Generation heading.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/panels/algorithm.ts
git commit -m "docs(algorithm): add TMM subsection describing matrix-product method"
```

---

## Task 18: `CLAUDE.md` activated-presets note

**Files:**
- Modify: `perovskite-sim/CLAUDE.md`

- [ ] **Step 1: Read the existing TMM section**

Read `perovskite-sim/CLAUDE.md` lines 62–63 (the paragraph starting "Transfer-matrix optics (TMM)").

- [ ] **Step 2: Append a one-line activated-presets note**

Append after the existing TMM paragraph:

```markdown
**Activated presets:** `nip_MAPbI3_tmm.yaml`, `pin_MAPbI3_tmm.yaml` (Phase 2 — Apr 2026). Both prepend a 1 mm `role: substrate` glass layer (optical-only) and set `optical_material` on every electrical layer. The vanilla `nip_MAPbI3.yaml` and `pin_MAPbI3.yaml` remain Beer-Lambert for back-compat with existing benchmarks. Substrate layers are filtered out of the drift-diffusion grid by `electrical_layers()` in `models/device.py`; the TMM spatial grid is offset by the substrate cumulative thickness so G(x) lands on the correct electrical nodes.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): note TMM-activated presets and substrate-filter pattern"
```

---

## Task 19: Full verification + manual browser smoke

**Files:** (none modified — pure verification)

- [ ] **Step 1: Full pytest suite**

Run: `pytest` (from repo root, default non-slow markers).
Expected: all green.

- [ ] **Step 2: Slow benchmark suite**

Run: `pytest -m slow -v`
Expected: all existing pinned values still pass (including ionmonger/driftfusion benchmarks).

- [ ] **Step 3: Frontend build**

Run: `cd perovskite-sim/frontend && npm run build`
Expected: tsc green, bundle produced (4 MB Plotly warning is expected).

- [ ] **Step 4: Manual verification checklist (from spec §9d)**

Start backend + frontend:
```bash
# terminal A
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload
# terminal B
cd perovskite-sim/frontend && npm run dev
```

Then verify each item by clicking through the workstation:

- [ ] **TMM active badge** appears when `nip_MAPbI3_tmm` is the active device on full tier; hidden on legacy/fast tiers.
- [ ] **Optical material dropdown** appears in the per-layer editor on full tier and is absent on legacy/fast.
- [ ] **Dropdown options** include all 12 materials and list `(none — Beer-Lambert)` first.
- [ ] **J-V plot** for `nip_MAPbI3_tmm` produces a J_sc within the [180, 260] A/m² band and visibly lower than `nip_MAPbI3` on the same run comparison.
- [ ] **Interference fringes thickness sweep**: run J-V at absorber thicknesses 50, 100, 200, 400, 600, 800 nm on `nip_MAPbI3_tmm`; J_sc should oscillate, not monotonically increase. On vanilla `nip_MAPbI3` the same sweep should be monotonic.
- [ ] **Glass incoherent flag** actually flattens fringes: flip `incoherent: false` in a local copy of `nip_MAPbI3_tmm.yaml`, restart backend, and run a fine wavelength sweep — confirm the sub-nanometer ripples re-appear. Revert.
- [ ] **Auto-discovery**: copy `data/nk/Ag.csv` to `data/nk/Cu.csv`, restart backend, confirm `Cu` appears in the dropdown.
- [ ] **Tutorial / Parameters / Algorithm panels** all render the new TMM content.

- [ ] **Step 5: Final commit if any checklist item required a fix**

Otherwise, no commit. Report completion to the coordinator.

---

## Self-Review

**Spec coverage:**
- §4 material library → Task 1 + Task 2 ✓
- §5 incoherent-layer support → Task 4 ✓
- §6 substrate-role BC handling → Task 5 ✓ (NOTE: plan defers the FTO/Au as-degenerate-semiconductor risk by keeping the electrical stack identical to `nip_MAPbI3`; substrate-only prepend is strictly optical. This is the "fallback" option the spec explicitly endorses in §12.)
- §7 new preset variants → Task 7 + Task 8 ✓
- §8 frontend changes → Task 12 (api + tier-gating), Task 13 (dropdown), Task 14 (badge), Task 11 (endpoint) ✓
- §9 validation and tests → Tasks 4, 5, 7, 8, 9, 10 ✓
- §10 documentation → Tasks 15, 16, 17, 18 ✓
- §11 file inventory → all entries touched ✓
- §13 verification checklist → Task 19 ✓

**Placeholder scan:** No TBD/TODO strings in task bodies. The regression pins in Task 10 are filled *during* that task, not left as placeholders in the plan — the task explicitly walks through extracting the values and writing them into the test file.

**Type consistency:**
- `MaterialParams.incoherent: bool = False` — referenced consistently in Tasks 3, 4, 5, 7
- `LayerSpec.role` stays a free-form string; Task 5 adds a filter, not an enum
- `electrical_layers(stack) -> tuple[LayerSpec, ...]` — helper name used consistently in Tasks 5, 18
- `fetchOpticalMaterials(): Promise<string[]>` — return type matches the backend `{materials: string[]}` shape in Task 11
- `FieldKind = 'numeric' | 'select-optical-material' | 'boolean'` — used consistently in Task 13

**Known limitation (flagged to the user before execution):** Task 7's nip_MAPbI3_tmm preset keeps the existing HTL-on-the-left layer order, prepending glass to that side rather than ordering sun→back. The alternative (full sun-side-first order) would require swapping HTL↔ETL positions and re-validating the band-offset / V_bi calculation, which is out of scope for Phase 2a per the user's Path A decision. The TMM model still captures the key physics (front reflection, interference fringes) because it cares about optical ordering, not the electrical polarity labels.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-tmm-optics-activation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review (spec compliance then code quality) between tasks, fast iteration in this session.
2. **Inline Execution** — Execute tasks in this session directly using executing-plans, batch execution with human checkpoints.

Which approach?

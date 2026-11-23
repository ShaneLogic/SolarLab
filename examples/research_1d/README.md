# 1D Research Workflows (`examples/research_1d/`)

Runnable one-dimensional research examples exercising `solarlab`'s configuration compiler, unified sparse equation assembly, nonlinear/time/frequency-domain solvers, and lossless `RunStore` persistence.

For repository installation, V2 server startup, and governing PDEs, see the root [README.md](../../README.md) and [DevelopmentGuide.md](../../docs/DevelopmentGuide.md). Run all commands below from the repository root with single-thread BLAS:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
```

---

## Shared Provenance, Storage & Qualification Contract

All 1D workflows below enforce the same runtime invariants:

1. **Fresh Local Destinations:** Output JSON files and SQLite/NPY `RunStore` directories must be fresh, local, and outside cloud-sync folders. Existing paths are refused rather than overwritten.
2. **Cryptographic State Chaining:** Multi-stage workflows bind upstream accepted results via explicit `--initial-result` and `--initial-sha256` digests (`H_input`, `H_physics`, `H_execution`), verifying physical gates before running downstream solves.
3. **Solver-Free Readback & Failure Preservation:** Every `read` subcommand or reader script decodes committed artifacts without importing or invoking any numerical solver. Incomplete or failed calculations retain their accepted prefix and first failed attempt, returning exit status `2`.
4. **Scientific Qualification Boundary:** Passing solver residual and conservation checks (`succeeded`) certifies discrete numerical execution on the declared grid (`37` electrical nodes by default), not continuum mesh/time convergence (`scientific_qualification = False`). Missing $V_{\mathrm{oc}}$ brackets or undeclared incident irradiance (`--incident-power`) remain explicitly `null` (`unavailable`).

---

## Script Index (6 Workflow Tracks)

| Track | Module | Purpose |
|:------|:-------|:--------|
| **1. Ion-Free Steady & Sweeps** | [steady.py](steady.py) | Baseline dark equilibrium & illuminated DC solve (`--additional-recombination` demos custom local law) |
| | [jv_scan.py](jv_scan.py), [read_results.py](read_results.py) | Cold-compiled parameter sweep (`tau_n`, `tau_p`, `mu_n`, `mu_p`) persisted to `RunStore` + solver-free reader |
| | [voc_resume.py](voc_resume.py) | Midpoint root-bracketing recovery on external recorded fine-mesh state bundles (`1 µV` target) |
| **2. Small-Signal AC** | [impedance.py](impedance.py) | Ion-free sparse complex small-signal solve $(G + i\omega C)\delta u = b$ across Fixed / Robin contacts |
| | [ionic_impedance.py](ionic_impedance.py) | Coupled mobile-ion vs frozen-ion AC admittance around a saved steady operating state |
| | [multiple_ionic_impedance.py](multiple_ionic_impedance.py) | Small-signal AC response across two separated blocking ion domains |
| | [field_transport_ac.py](field_transport_ac.py) | Small-signal AC linearization including Caughey–Thomas field-dependent mobility tangents |
| **3. Ion-Free Transients** | [transient.py](transient.py), [read_transient.py](read_transient.py) | Fixed-bias illumination-ramp Backward Euler transient (`expm1` carrier inventory differences) + reader |
| | [transient_voltage.py](transient_voltage.py) | Driven time-dependent terminal-voltage ramp with interval displacement current $\Delta D / \Delta t$ |
| **4. Coupled Mobile-Ion** | [ionic_steady.py](ionic_steady.py), [ionic_transient.py](ionic_transient.py) | Single-species finite-site mobile-ion steady state (`--multilayer` confines ions to absorber) & BE ramp |
| | [dual_ionic.py](dual_ionic.py) | Coupled positive + negative mobile ions with independent finite site capacities |
| | [shared_site_ionic.py](shared_site_ionic.py) | Coupled dual-ion steady & transient sharing a common vacancy fraction |
| | [immobile_ions.py](immobile_ions.py) | Zero-diffusivity ($D_{\mathrm{ion}} = 0$) exact profile preservation in single and mixed shared-site stacks |
| | [ionic_initialization.py](ionic_initialization.py) | `direct_density_seed_v1` dark-to-illuminated coupled ionic state preparation |
| | [ionic_jv.py](ionic_jv.py), [ionic_pv.py](ionic_pv.py) | Equilibrium coupled-ion $J\text{–}V$ slices and full 51-point illuminated PV curve ($0 \to 1.25\text{ V}$) |
| | [multiple_ionic_domains.py](multiple_ionic_domains.py), [multiple_ionic_steady.py](multiple_ionic_steady.py), [multiple_ionic_transient.py](multiple_ionic_transient.py) | Two separated blocking ionic layer domains coupled to one shared electronic mesh (steady & BE transient) |
| **5. Physics Extensions** | [neutral_defects.py](neutral_defects.py) | Explicit neutral single-level bulk SRH defects (including `--unequal-dos` $N_C \neq N_V$ binding) |
| | [field_transport.py](field_transport.py) | Smooth Caughey–Thomas electric-field velocity saturation ($\beta > 1$) in DC assembly |
| | [optical_generation.py](optical_generation.py) | Explicit wavelength photon flux + $(n, k)$ coherent TMM control-volume generation & photon budget |
| **6. Process & HTTP Workers** | [process_sweep.py](process_sweep.py), [process_transient.py](process_transient.py), [process_ionic_jv.py](process_ionic_jv.py), [process_ionic_transient.py](process_ionic_transient.py) | Out-of-process `ProcessSupervisor` execution binding committed Git source digests (`--build-source-commit`) |
| | [service.py](service.py), [transient_service.py](transient_service.py), [ionic_jv_service.py](ionic_jv_service.py), [ionic_transient_service.py](ionic_transient_service.py) | Minimal FastAPI `/experiments` + `/jobs` + reconnectable SSE services for dedicated workstation views |

*(Supporting fixture/helper modules: [device.json](device.json), [ionic_multilayer.py](ionic_multilayer.py), [ionic_residual.py](ionic_residual.py), [shared_site_ionic_fixture.py](shared_site_ionic_fixture.py).)*

---

## Representative Command Recipes

### 1. Ion-Free Steady DC, Sweep & Direct $V_{\mathrm{oc}}$ Refinement

```sh
# Dark equilibrium + illuminated steady solve (add --additional-recombination for custom law demo)
.venv/bin/python -B examples/research_1d/steady.py

# Two-point absorber electron-lifetime sweep persisted to a fresh RunStore
.venv/bin/python -B examples/research_1d/jv_scan.py \
  --store outputs/research_1d/lifetime-demo \
  --study-id lifetime --parameter tau_n --values "1 us" "2 us"

# Solver-free readback (--include-arrays decodes persisted NumPy vectors)
.venv/bin/python -I -B examples/research_1d/read_results.py \
  --store outputs/research_1d/lifetime-demo --study-id lifetime
```

To refine a bracketed open-circuit voltage in Python without grid interpolation:

```python
from solarlab.experiments.jv.operating_points import solve_direct_voc

voc = solve_direct_voc(compiled, curve, voltage_tolerance_V=1e-6)
```

### 2. Ion-Free Transients & Small-Signal Impedance

```sh
# Conservative Backward Euler illumination ramp with 4/8/16-step refinement ladder
.venv/bin/python -B -m examples.research_1d.transient \
  --output outputs/research_1d/transient.json --refine

# Driven voltage transient
.venv/bin/python -B -m examples.research_1d.transient_voltage \
  --output outputs/research_1d/transient_voltage.json --refine

# Complex sparse AC admittance at 0, 1, and 1000 Hz (phasor convention exp(+i*omega*t))
.venv/bin/python -B -m examples.research_1d.impedance
```

### 3. Coupled Shared-Site Mobile Ions (Dark $\to$ Illuminated $J\text{–}V$ $\to$ AC)

```sh
# 1) Coupled dark steady state + 4-step ramp for shared-site positive & negative ions
.venv/bin/python -B -m examples.research_1d.shared_site_ionic \
  --output-directory outputs/research_1d/shared_site

# 2) Prepare illuminated coupled steady state via direct_density_seed_v1
DARK_JSON=outputs/research_1d/shared_site/Steady.json
DARK_SHA=$(shasum -a 256 "$DARK_JSON" | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_initialization solve \
  --policy direct_density_seed_v1 \
  --initial-result "$DARK_JSON" --initial-sha256 "$DARK_SHA" \
  --voltage 0.002 --generation-scale 1 \
  --output outputs/research_1d/ionic_init.json \
  --accepted-steady-output outputs/research_1d/illuminated_steady.json

# 3) Full 51-point illuminated coupled-ion J-V curve (0 to 1.25 V)
LIGHT_JSON=outputs/research_1d/illuminated_steady.json
LIGHT_SHA=$(shasum -a 256 "$LIGHT_JSON" | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.ionic_pv solve \
  --initial-result "$LIGHT_JSON" --initial-sha256 "$LIGHT_SHA" \
  --output outputs/research_1d/ionic_pv.json

# 4) Coupled mobile-ion vs frozen-ion small-signal AC response (--ion-mode mobile|frozen)
.venv/bin/python -B -m examples.research_1d.ionic_impedance solve \
  --kind shared --initial-result "$DARK_JSON" --initial-sha256 "$DARK_SHA" \
  --frequencies 0 0.01 1 1000 --ion-mode mobile \
  --output outputs/research_1d/ionic_ac.json
```

### 4. Multiple Blocking Ion Domains & Constitutive Extensions

```sh
# Two separated blocking ion domains on one electronic mesh (steady + BE transient)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_steady \
  --output outputs/research_1d/multi_steady.json
MULTI_SHA=$(shasum -a 256 outputs/research_1d/multi_steady.json | cut -d ' ' -f 1)
.venv/bin/python -B -m examples.research_1d.multiple_ionic_transient solve \
  --initial-result outputs/research_1d/multi_steady.json --initial-sha256 "$MULTI_SHA" \
  --output outputs/research_1d/multi_transient.json

# Explicit neutral bulk defects, Caughey-Thomas field mobility, and coherent spectral TMM
.venv/bin/python -B -m examples.research_1d.neutral_defects --unequal-dos \
  --output outputs/research_1d/neutral_defects.json
.venv/bin/python -B -m examples.research_1d.field_transport \
  --output outputs/research_1d/field_transport.json
.venv/bin/python -B -m examples.research_1d.optical_generation \
  --output outputs/research_1d/optical_generation.json
```

### 5. Supervised Process & Local HTTP Execution

```sh
# Supervised out-of-process illuminated ionic J-V job + solver-free readback
.venv/bin/python -B -m examples.research_1d.process_ionic_jv run \
  --build-source-commit "$(git rev-parse HEAD)" \
  --store outputs/research_1d/proc_store --work outputs/research_1d/proc_work \
  --run-id illuminated-jv
.venv/bin/python -B -m examples.research_1d.process_ionic_jv read \
  --store outputs/research_1d/proc_store --run-id illuminated-jv
```

Each `*_service.py` script (`service.py`, `transient_service.py`, `ionic_jv_service.py`, `ionic_transient_service.py`) exposes the same `/experiments` header handshake (`X-Solarlab-Registry`, `X-Solarlab-Job-Schema`), `POST /jobs`, `/runs/{run_id}/attempts/{attempt_id}/events` SSE stream, and `/results/...` endpoints consumed by `web/?view={steady-jv,transient,ionic-jv,ionic-transient}`.

---

## Persisted Physical Units

| Quantity | Unit / Convention |
|:---------|:------------------|
| `nodes_m`, `voltage_V` | $\mathrm{m}$, $\mathrm{V}$ |
| `electron_density_m3`, `hole_density_m3`, `positive_ion_density_m3` | $\mathrm{m}^{-3}$ |
| `current_density_A_m2`, `output_current_A_m2`, $J_{\mathrm{sc}}$ | $\mathrm{A\ m^{-2}}$ (`-voltage_polarity * left_face_current` for PV output) |
| Maximum power $P_{\max}$ | $\mathrm{W\ m^{-2}}$ |
| Fill factor $\mathrm{FF}$, efficiency $\mathrm{PCE}$ | Dimensionless fractions (not percentages) |
| Small-signal admittance $Y(\omega)$ / impedance-area $Z(\omega)$ | $\mathrm{S\ m^{-2}}$ / $\Omega\ \mathrm{m}^2$ under $\exp(+i\omega t)$ phasor convention |

Focused end-to-end verification of this directory runs via:

```sh
.venv/bin/python -B -m pytest -q tests/integration/test_research_1d_workflow.py
```

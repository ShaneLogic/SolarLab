# Physics Accuracy Upgrade — Design Spec

**Date**: 2026-04-13
**Goal**: Comprehensive physics upgrade to the perovskite drift-diffusion simulator for more physically reasonable and accurate results.
**Validation targets**: IonMonger (Courtier 2019) and Driftfusion (Calado 2016) published benchmarks.
**Approach**: Bottom-up layered (5 phases), each independently testable.

---

## Current State

- 26/28 physics checks pass
- 2 failing checks: V_oc too low (0.749 V vs expected ~1.07 V), V_oc insensitive to n_i
- Root cause: contact BCs ignore band offsets (chi, Eg)
- Single-species ion model, scalar Beer-Lambert optics, uniform SRH lifetimes, fixed T=300 K

## Constraints

- Backward compatibility: existing configs must work unchanged in legacy mode
- Tiered performance: fast mode (~20 s J-V) and full mode (~2-5 min J-V)
- Spectral optics: 300-800 nm, 200 wavelength points (2.5 nm resolution)
- Immutable data model: all dataclasses remain frozen; use `replace()` for mutations

---

## Phase 1: Contact & Interface Physics

### 1a. Band-Offset-Corrected Contact Boundary Conditions

**Problem**: `build_material_arrays` computes contact equilibrium densities using only doping (`N_D - N_A` and `ni`), ignoring the chi/Eg mismatch between transport layers. This makes contacts "leaky" — too many minority carriers inject from contacts into the absorber, killing V_oc.

**Solution**: Compute V_bi from the Fermi level difference across the full heterostack rather than relying on the manually-set `DeviceStack.V_bi` scalar.

```
V_bi = (1/q) * (E_F,left - E_F,right)
     = (chi_R + Eg_R - kT*ln(p_R/Nv_R)) - (chi_L + kT*ln(n_L/Nc_L))
```

When chi/Eg are not configured (all zero), `compute_V_bi()` falls back to the existing manual `V_bi` field, preserving legacy behavior.

**Changes**:
- `DeviceStack`: add `compute_V_bi()` classmethod deriving V_bi from layer chi, Eg, doping
- `build_material_arrays`: contact densities use each contact layer's own ni (already correct)
- `assemble_rhs`: use computed V_bi as Poisson right BC

### 1b. Thermionic Emission at Heterointerfaces

**Problem**: At abrupt band discontinuities, drift-diffusion SG fluxes overestimate current. The thermionic emission rate limits current across the barrier.

**Solution**: At each interface node, compute band offsets:
```
delta_Ec = chi[i+1] - chi[i]
delta_Ev = (chi[i] + Eg[i]) - (chi[i+1] + Eg[i+1])
```

When |delta_Ec| or |delta_Ev| > 0.05 eV, apply Richardson-Dushman correction:
```
J_TE = A* * T^2 * (n_left * exp(-delta_Ec/kT) - n_right)
```

**Changes**:
- `fe_operators.py`: add `thermionic_correction(n_L, n_R, delta_E, T)` function
- `continuity.py`: at interface faces, cap SG flux to thermionic emission limit (take the minimum magnitude — TE acts as an upper bound on current, not an additive correction)
- `MaterialParams`: add optional `A_star_n`, `A_star_p` (Richardson constants)

**Validation**: V_oc = 1.07 V on ionmonger_benchmark config (currently 0.912 V with band offsets, 0.749 V without).

---

## Phase 2: Transfer-Matrix Optics

### 2a. TMM Engine

**Problem**: Beer-Lambert with scalar alpha misses thin-film interference, wavelength-dependent absorption, reflection losses, and parasitic absorption. Generation accuracy ~5-10% off.

**Solution**: New module `perovskite_sim/physics/optics.py` implementing the 2x2 transfer matrix method.

For each wavelength lambda:
```
M_total = D_0^-1 * [product_j (D_j * P_j * D_j^-1)] * D_s

where:
  D_j = dynamical matrix (n_j, theta_j)
  P_j = propagation matrix (n_j, k_j, d_j, lambda)
  r = M_total[1,0] / M_total[0,0]
  t = 1 / M_total[0,0]
```

Position-resolved absorbed power density A(x, lambda) integrated over AM1.5G:
```
G(x) = integral A(x, lambda) * Phi_AM1.5(lambda) d_lambda   [m^-3 s^-1]
```

**Spectral config**: 300-800 nm, 200 points (2.5 nm resolution).

### 2b. Optical Material Data

New config section per layer:
```yaml
layers:
  - name: MAPbI3
    optical:
      n_data: "data/nk/MAPbI3.csv"   # columns: lambda_nm, n, k
      # OR for simple cases:
      n_const: 2.5
      k_model: "urbach"
      Eu: 0.015                       # Urbach energy [eV]
```

Ship default n(lambda), k(lambda) for: MAPbI3, TiO2, spiro-OMeTAD, ITO/FTO.
Store in `perovskite_sim/data/nk/` and `perovskite_sim/data/am15g.csv`.

### 2c. Tiered Integration

- **Full mode**: TMM at 200 wavelengths -> G(x) array, cached in MaterialArrays as `G_optical`
- **Fast mode**: TMM once -> effective alpha_eff(x) -> Beer-Lambert path (captures reflection + parasitic)
- **Legacy mode**: Current Beer-Lambert with scalar alpha

### 2d. Performance

TMM at 200 wavelengths on 3-layer stack: ~5 ms (one-time, before transient). G(x) cached — zero impact on RHS evaluation speed.

**Changes**:
- New: `perovskite_sim/physics/optics.py`
- New: `perovskite_sim/data/nk/*.csv`, `perovskite_sim/data/am15g.csv`
- Modified: `MaterialParams` — optional `optical` field
- Modified: `build_material_arrays` — compute G_optical when optical data present
- Modified: `assemble_rhs` — use `mat.G_optical` instead of `beer_lambert_generation` when available

**Validation**: J_sc within 2% of TMM reference. Interference fringe positions match analytical expectations.

---

## Phase 3: Dual-Species Ion Migration

### 3a. State Vector Extension

Current: `y = (n, p, P+)` — 3N unknowns
New: `y = (n, p, P+, P-)` — 4N unknowns

Negative species (e.g., methylammonium vacancy V_MA-) has independent transport params:
- `D_ion_neg`: diffusion coefficient [m^2/s]
- `P0_neg`: equilibrium density [m^-3]
- `P_lim_neg`: steric limit [m^-3]

### 3b. Negative Ion Flux

Same SG discretization, reversed drift direction:
```
F_neg = D_neg/h * (B(-xi)*P-[i] - B(xi)*P-[i+1])
```

Same steric correction: `D_eff = D_neg / (1 - P-_avg/P_lim_neg)`.
Zero-flux BCs at both contacts.

### 3c. Charge Density Update

```python
rho = Q * (p - n + (P_pos - P0_pos) - (P_neg - P0_neg) - N_A + N_D)
```

### 3d. Config

```yaml
layers:
  - name: MAPbI3
    D_ion: 1.01e-17       # positive species (existing)
    P0: 1.6e25
    P_lim: 1.6e27
    D_ion_neg: 3.2e-18    # negative species (new, optional)
    P0_neg: 1.6e25
    P_lim_neg: 1.6e27
```

When `D_ion_neg` absent or zero: single-species mode (backward compatible).

### 3e. Operator Splitting Update

`split_step` advances both species simultaneously while freezing carriers. Ion sub-state: `y_ions = (P_pos, P_neg)` — 2N unknowns.

### 3f. Tiered Mode

- **Fast**: Single species (current)
- **Full**: Dual species when configured

**Changes**:
- Modified: `StateVec` — add `P_neg` field
- Modified: `ion_migration.py` — `ion_continuity_rhs_dual()` with reversed drift for negative species
- Modified: `_charge_density` — include `(P_neg - P0_neg)` term
- Modified: `MaterialArrays` — add `D_ion_neg_*`, `P0_neg`, `P_lim_neg_*` arrays
- Modified: `MaterialParams` — add `D_ion_neg`, `P0_neg`, `P_lim_neg` fields
- Modified: `split_step` — advance both species
- Modified: `solve_equilibrium` — initialize P_neg from P0_neg

**Validation**:
- Ion conservation: both species < 0.1% error independently
- Hysteresis index increases with dual species vs single species
- P+ accumulates cathode-side, P- accumulates anode-side (mirror image)
- Quantitative match to IonMonger dual-species J-V at 0.04 V/s

---

## Phase 4: Position-Dependent Traps & Temperature

### 4a. Spatially Varying Trap Profiles

**Problem**: Uniform tau per layer misses interface-enhanced defect densities.

**Solution**: Exponential trap profile from interfaces into bulk:
```
N_t(x) = N_t_bulk + (N_t_interface - N_t_bulk) * [exp(-d_left/L_d) + exp(-d_right/L_d)]
tau(x) = 1 / (sigma * v_th * N_t(x))
```

Config:
```yaml
layers:
  - name: MAPbI3
    tau_n: 1e-6
    trap_profile:
      type: "exponential"
      N_t_interface: 1e17   # [m^-3]
      N_t_bulk: 1e14        # [m^-3]
      decay_length: 50e-9   # [m]
```

When `trap_profile` absent: uniform tau (current behavior). Existing interface recombination (v_n, v_p) remains as a separate mechanism.

`build_material_arrays` computes tau_n(x), tau_p(x) arrays. Recombination code already accepts array tau.

### 4b. Temperature-Dependent Parameters

New module: `perovskite_sim/physics/temperature.py`

Scaling functions:
```python
ni(T) = sqrt(Nc * Nv) * exp(-Eg / 2kT)       # Nc, Nv proportional to T^(3/2)
mu(T) = mu_300 * (T/300)^gamma                 # gamma ~ -1.5 (phonon scattering)
D_ion(T) = D_ion_300 * exp(-E_a/k * (1/T - 1/300))   # Arrhenius
V_T = kT/q
```

### 4c. Temperature Modes

1. **Isothermal** (default): T=300 K, all params fixed
2. **Uniform T != 300 K**: User sets T, `build_material_arrays` applies scaling

```yaml
device:
  T: 350   # [K], optional
```

Self-consistent thermal (heat equation + iteration) deferred to future work.

### 4d. Config Extensions

```yaml
layers:
  - name: MAPbI3
    Nc300: 2.2e24       # effective DOS at 300 K [m^-3]
    Nv300: 1.8e24
    mu_T_gamma: -1.5    # mobility T exponent
    E_a_ion: 0.58       # ion activation energy [eV]
```

When T=300 or Nc300/Nv300 absent: no scaling (current behavior).

**Changes**:
- New: `perovskite_sim/physics/temperature.py`
- Modified: `MaterialParams` — add `Nc300`, `Nv300`, `mu_T_gamma`, `E_a_ion`, trap profile fields
- Modified: `build_material_arrays` — accept T param, apply scaling, compute tau(x) from trap profile
- Modified: config schema

**Validation**:
- Recombination peaks near interfaces with trap profiles
- V_oc temperature coefficient ~ -2 mV/K
- Higher T -> faster ions -> less hysteresis at same scan rate
- Arrhenius activation energy extractable from log(D) vs 1/T

---

## Phase 5: Tiered Mode Integration

### 5a. Mode Config

```yaml
device:
  mode: "full"   # "fast" | "full" | "legacy"
```

| Feature | Legacy | Fast | Full |
|---------|--------|------|------|
| Contact BCs | Doping-only | Band-offset corrected | Band-offset corrected |
| Thermionic emission | No | No | Yes |
| Optics | Beer-Lambert scalar | TMM -> effective alpha | TMM 200 wavelengths |
| Ion species | Single | Single | Dual (if configured) |
| Trap profile | Uniform tau | Uniform tau | Spatially varying |
| Temperature | 300 K fixed | 300 K fixed | T-dependent (if configured) |
| Default grid | N=100 | N=80 | N=150 |
| Est. J-V time | ~25 s | ~20 s | ~2-5 min |

### 5b. Mode Resolution

New module: `perovskite_sim/models/mode.py`

```python
@dataclass(frozen=True)
class SimulationMode:
    use_band_offset_contacts: bool
    use_thermionic_emission: bool
    optics: Literal["beer_lambert", "tmm_effective", "tmm_full"]
    dual_ions: bool
    use_trap_profile: bool
    use_temperature_scaling: bool
    N_grid: int

def resolve_mode(config, mode_override=None) -> SimulationMode:
    """Mode is a ceiling: enables physics up to specified level,
    but only if config provides necessary parameters."""
```

### 5c. Experiment Integration

Each experiment gains optional `mode` parameter:
```python
def run_jv_sweep(stack, mode=None, N_grid=None, ...):
    resolved = resolve_mode(stack.config, mode)
    ...
```

### 5d. Frontend

Mode toggle dropdown in device panel: Legacy / Fast / Full.
Sent as parameter in job request. Progress bar handles variable runtimes.

### 5e. Invariants

- `assemble_rhs` signature and hot-path structure unchanged — mode resolution before transient
- `MaterialArrays` cache pattern preserved — built once, fields populated per mode
- Existing configs work unchanged in legacy mode
- Existing tests run in legacy mode by default

**Validation**:
- Legacy: zero regression on 26 passing checks
- Fast: V_oc within 0.05 V of full mode
- Full: match IonMonger within 5% on V_oc, J_sc, FF; match Driftfusion hysteresis shape
- Graceful fallback: full mode without optical data uses Beer-Lambert

---

## File Change Summary

### New Files
- `perovskite_sim/physics/optics.py` — TMM engine + spectral integration
- `perovskite_sim/physics/temperature.py` — T-dependent parameter scaling
- `perovskite_sim/models/mode.py` — simulation mode resolution
- `perovskite_sim/data/nk/*.csv` — optical constants (MAPbI3, TiO2, spiro, ITO)
- `perovskite_sim/data/am15g.csv` — AM1.5G spectrum

### Modified Files
- `perovskite_sim/models/parameters.py` — new optional fields (A_star, optical, D_ion_neg, Nc300, trap_profile, etc.)
- `perovskite_sim/models/device.py` — `compute_V_bi()`, mode field
- `perovskite_sim/models/config_loader.py` — parse new config sections
- `perovskite_sim/solver/mol.py` — StateVec 4N, MaterialArrays extensions, charge density with P_neg
- `perovskite_sim/solver/newton.py` — initialize P_neg
- `perovskite_sim/physics/continuity.py` — thermionic emission blending at interfaces
- `perovskite_sim/physics/ion_migration.py` — `ion_continuity_rhs_dual()`
- `perovskite_sim/discretization/fe_operators.py` — `thermionic_correction()`
- `perovskite_sim/experiments/jv_sweep.py` — mode parameter
- `perovskite_sim/experiments/impedance.py` — mode parameter
- `perovskite_sim/experiments/degradation.py` — mode parameter, dual-ion damage
- `backend/main.py` — pass mode through to experiments
- `frontend/src/device-panel.ts` — mode toggle UI
- Config YAML files — add optical, D_ion_neg, trap_profile, T sections to benchmark configs

### New Tests
- `tests/unit/physics/test_optics.py` — TMM against analytical solutions
- `tests/unit/physics/test_temperature.py` — parameter scaling functions
- `tests/unit/physics/test_thermionic.py` — thermionic emission current
- `tests/unit/test_mode.py` — mode resolution logic
- `tests/integration/test_dual_ions.py` — conservation, mirror accumulation
- `tests/integration/test_tiered_mode.py` — legacy/fast/full regression
- `tests/regression/test_ionmonger_full.py` — quantitative IonMonger match
- `tests/regression/test_driftfusion_full.py` — quantitative Driftfusion match

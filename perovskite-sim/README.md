# perovskite-sim

The Python simulation package, FastAPI backend, and Vite/TypeScript
frontend that make up the SolarLab simulator.

> 📖 **Start here:** the [root README](../README.md) covers installation,
> physics, equations, UI walkthrough, and shipped presets. This file is
> just a short orientation to the `perovskite-sim/` subtree.

## Layout

```
perovskite-sim/
├── perovskite_sim/   Python simulation library (drift-diffusion + ions + TMM)
├── backend/          FastAPI HTTP wrapper — see backend/README.md
├── frontend/         Vite + TypeScript + Plotly single-page UI
├── configs/          Shipped YAML device presets
├── tests/            pytest suite (unit / integration / regression)
└── notebooks/        Exploratory benchmarks
```

## Quick install

```bash
pip install -e ".[dev]"        # Python package in editable mode
cd frontend && npm install     # frontend dependencies
```

## Tests

```bash
pytest                                                  # default unit + integration (~15 s)
pytest -m slow                                          # physics regression (~30 s, BLAS pinned)
pytest --cov=perovskite_sim --cov-report=term-missing   # with coverage
```

## Notebooks

Interactive notebooks under `notebooks/`:

- `01_jv_hysteresis.ipynb` — J–V sweep with hysteresis
- `02_impedance.ipynb` — impedance spectroscopy (Nyquist plot)
- `03_degradation.ipynb` — long-term degradation simulation

Benchmark scripts (`.py`, run with `python`):

- `04_ionmonger_benchmark.py`
- `05_comprehensive_benchmark.py`
- `06_e2e_notebook_vs_api.py`

## Physical Model

The simulator models a 1D thin-film solar cell as a stack of semiconductor
layers between two metallic contacts. Light enters from one side, generates
electron-hole pairs, and the built-in electric field separates them to
produce current.

### Device Structure

```
                      Light (AM1.5G)
                          |
                          v
    ┌─────────────────────────────────────────────┐
    │              Left Contact (x = 0)           │  Ohmic: n = n_L, p = p_L
    │               phi = 0  (grounded)           │
    ├─────────────────────────────────────────────┤
    │                                             │
    │   HTL  (hole transport layer)               │  High N_A doping
    │   e.g. spiro-OMeTAD, 200 nm                 │  Blocks electrons (low mu_n)
    │                                             │
    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  Interface: (v_n, v_p) SRH
    │                                             │
    │   Absorber (perovskite)                     │  Intrinsic (low doping)
    │   e.g. MAPbI3, 400 nm                       │  Optical absorption: G(x)
    │                                             │  Mobile ions: P+(x,t), P-(x,t)
    │   n(x,t) ←──drift──→ p(x,t)                │  SRH + radiative + Auger R
    │       ↕ diffusion ↕                         │
    │                                             │
    ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  Interface: (v_n, v_p) SRH
    │                                             │
    │   ETL  (electron transport layer)           │  High N_D doping
    │   e.g. TiO2, 100 nm                        │  Blocks holes (low mu_p)
    │                                             │
    ├─────────────────────────────────────────────┤
    │             Right Contact (x = L)           │  Ohmic: n = n_R, p = p_R
    │          phi = V_bi - V_app                 │
    └─────────────────────────────────────────────┘

    x = 0 ──────────────────────────────────> x = L
                    1D spatial domain
```

### Band Diagram

```
  Energy
    ^
    │    ┌──────┐                              ┌──────┐
    │    │      │  ΔEc (conduction band offset)│      │
    │    │  Ec  └──────────────────────────────┘  Ec  │
    │    │  ┄┄┄┄┄ EF ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ │
    │    │  Ev  ┌──────────────────────────────┐  Ev  │
    │    │      │  ΔEv (valence band offset)   │      │
    │    └──────┘                              └──────┘
    │     HTL          Absorber                  ETL
    │   (p-type)     (intrinsic)              (n-type)
    │
    │        <── Built-in field E = -dφ/dx ──>
    │            (separates photogenerated
    │             electrons and holes)
    └───────────────────────────────────────────────> x
```

### Transport Processes

```
  Carrier currents (Scharfetter-Gummel discretised):

      Jn = q μn n E  +  q Dn dn/dx        (electron: drift + diffusion)
      Jp = q μp p E  -  q Dp dp/dx        (hole:     drift + diffusion)

  Ion flux (steric Scharfetter-Gummel):

      F_ion = -D_ion [ dP/dx + (q/kT) P (1 - P/P_lim) dφ/dx ]

  Recombination:

      R = R_SRH + R_rad + R_Auger + R_interface

      R_SRH   = (np - ni²) / [ τp(n + n1) + τn(p + p1) ]    (bulk traps)
      R_rad   = B (np - ni²)                                   (radiative)
      R_Auger = (Cn·n + Cp·p)(np - ni²)                       (Auger)
      R_iface = (np - ni²) / [ (p+p1)/vn + (n+n1)/vp ]       (surface)

  Optical generation:

      Beer-Lambert:  G(x) = α Φ exp(-αx)           (Legacy/Fast tier)
      TMM:           G(x) = ∫ a(x,λ) Φ_AM1.5(λ) dλ  (Full tier)
```

### How It All Fits Together

```
  ┌─────────────────────────────────────────────────────┐
  │                 State vector y(t)                    │
  │              y = [ n, p, P+, (P-) ]                 │
  │                  per grid node                      │
  └───────────────┬─────────────────────────────────────┘
                  │
                  v
  ┌─────────────────────────────────────────────────────┐
  │           assemble_rhs(t, y)  →  dy/dt              │
  │                                                     │
  │  1. Apply ohmic contact BCs to n, p                 │
  │  2. Compute charge density ρ = q(p - n + P - P0     │
  │                                  - NA + ND)         │
  │  3. Solve Poisson: ε₀εr d²φ/dx² = -ρ  (Dirichlet) │
  │  4. Compute G(x) from TMM or Beer-Lambert           │
  │  5. Carrier continuity: dn/dt, dp/dt (SG fluxes)   │
  │  6. Interface recombination at heterojunctions       │
  │  7. Thermionic emission cap at band offsets > 50 meV│
  │  8. Ion continuity: dP/dt (zero-flux BCs)           │
  │  9. Enforce dn=dp=0 at contacts (hold Dirichlet)    │
  └───────────────┬─────────────────────────────────────┘
                  │
                  v
  ┌─────────────────────────────────────────────────────┐
  │       Radau (implicit Runge-Kutta) time stepper     │
  │                                                     │
  │  - Stiff ODE: dy/dt = f(t,y)                       │
  │  - max_step cap near V_bi to prevent branch jumps   │
  │  - Radau Jacobian via finite differences            │
  └─────────────────────────────────────────────────────┘
```

### Supported Device Architectures

| Config | Structure | Ions | Optics |
|--------|-----------|------|--------|
| `nip_MAPbI3` | spiro / MAPbI3 / TiO2 | Single species | Beer-Lambert |
| `nip_MAPbI3_tmm` | Glass / spiro / MAPbI3 / TiO2 | Single species | TMM |
| `pin_MAPbI3` | TiO2 / MAPbI3 / spiro | Single species | Beer-Lambert |
| `ionmonger_benchmark` | Courtier 2019 reference | Single species | Beer-Lambert |
| `cigs_baseline` | ZnO / CdS / CIGS | None (D_ion=0) | Beer-Lambert |
| `cSi_homojunction` | n+ / p Si wafer | None (D_ion=0) | Beer-Lambert |
| `tandem_lin2019` | Wide-gap / narrow-gap tandem | Single species | TMM |

## Initial Conditions and Boundary Conditions

The simulator solves the coupled 1D drift-diffusion + Poisson + mobile-ion
system. Below are the mathematical conditions applied at the device contacts
(boundary conditions) and the strategies used to seed the state vector
(initial conditions).

### Governing Equations

| Equation | PDE |
|----------|-----|
| Poisson | `d/dx(ε₀ εᵣ dφ/dx) = −ρ` |
| Electron continuity | `∂n/∂t = (1/q) dJₙ/dx + G − R` |
| Hole continuity | `∂p/∂t = −(1/q) dJₚ/dx + G − R` |
| Ion (vacancy) continuity | `∂P/∂t = −dFₚ/dx` |

State vector per grid node: **y = (n, p, P)** — electron density, hole
density, and positive-ion (vacancy) density. Dual-species mode adds a
negative-ion field P⁻.

### Boundary Conditions

#### Electrostatic potential (Poisson equation) — Dirichlet

| Contact | Value |
|---------|-------|
| Left (x = 0) | `φ = 0` (grounded) |
| Right (x = L) | `φ = V_bi − V_app` |

Forward bias (`V_app > 0`) reduces the built-in field; `V_app ≈ V_oc` yields
near-open-circuit conditions.

The Poisson operator uses **harmonic-mean face permittivities**:

```
ε̃_{i+½} = 2 εᵣ[i] εᵣ[i+1] / (εᵣ[i] + εᵣ[i+1])
```

This is the exact series-capacitor result for a sharp dielectric interface and
avoids the field concentration artefact of nodal averaging.

*Source:* `perovskite_sim/physics/poisson.py`

#### Electron and hole densities — Dirichlet (ohmic contacts)

Both contacts are treated as ideal ohmic contacts. Carrier densities at
the boundaries are clamped to the **thermal-equilibrium values** derived
from the doping of the outermost layers:

```
n·p = nᵢ²          (mass-action law)
n − p = N_D − N_A  (charge neutrality)
```

Solved via the numerically stable two-branch formula (avoids cancellation
and overflow):

```
net  = N_D − N_A
disc = √(net² + 4·nᵢ²)

n-type (net ≥ 0):  n = ½(net + disc),   p = nᵢ²/n
p-type (net < 0):  p = ½(−net + disc),  n = nᵢ²/p
```

These values are computed once per experiment in `build_material_arrays()`
and stored as `n_L, p_L, n_R, p_R`. The time derivatives at the contact
nodes are set to zero (`dn[0] = dn[-1] = dp[0] = dp[-1] = 0`) so the
Dirichlet values remain constant throughout the transient.

*Source:* `perovskite_sim/solver/mol.py` (lines 369–384, 523–524, 564–566)

#### Ion (vacancy) densities — Neumann (zero-flux)

Ions cannot leave the device. At both contacts the vacancy flux is set to
zero:

```
F_P(x = 0) = F_P(x = L) = 0
```

Implemented by padding the internal flux array with zeros at each end
before computing the finite-difference divergence. The same zero-flux
condition applies to both positive and negative ion species.

A **steric saturation limit** prevents unphysical ion pile-up:

```
steric = 1 / max(1 − P_avg/P_lim, 10⁻⁶)
```

*Source:* `perovskite_sim/physics/ion_migration.py`

#### Thermionic emission at heterointerfaces

At internal interfaces where the conduction-band offset `|ΔEᶜ|` or
valence-band offset `|ΔEᵛ|` exceeds 0.05 eV, the Scharfetter–Gummel
carrier flux is capped to the **Richardson–Dushman thermionic emission
limit**. This prevents the SG scheme from overestimating current across
sharp band discontinuities (a known single-grid-spacing artefact).
Interface faces where TE activates are pre-computed in `MaterialArrays`.

*Source:* `perovskite_sim/physics/continuity.py`,
`perovskite_sim/discretization/fe_operators.py`

#### Interface recombination

At each heterointerface, surface recombination is parameterised by
velocities `(v_n, v_p)` [m/s] carried in `DeviceStack.interfaces`. The
surface SRH rate is converted to a volumetric rate by dividing by the
local dual-grid cell width.

### Initial Conditions

#### Dark equilibrium (default)

The default initial state is a **quasi-neutral dark equilibrium** with a
neutral ionic background. At every grid node:

```
n·p = nᵢ²(layer)      mass-action law
n − p = N_D − N_A      local charge neutrality (ions treated as neutral background)
```

The configured vacancy density P₀ is treated as a **neutral ionic
background** — it does not appear as net space charge in the initial
carrier balance. This avoids the enormous artificial carrier imbalance
that arises when P₀ is treated as net positive charge.

The ion profile is initialised to the uniform per-layer value `P_ion0`.
Contact nodes are overwritten with the ohmic-contact equilibrium densities.

*Source:* `perovskite_sim/solver/newton.py`

#### Illuminated steady-state (light-soaked)

For experiments that begin under illumination (J–V sweep, impedance,
degradation), the initial state is obtained by **integrating the full MOL
system for t_settle = 1 ms** starting from dark equilibrium, under
illumination at the starting voltage:

```
y_dark  = solve_equilibrium(x, stack)
y_light = run_transient(x, y_dark, [0, t_settle], illuminated=True, V_app)
```

Carrier dynamics equilibrate on a sub-microsecond timescale, so 1 ms is
more than sufficient. Ion displacement over this interval is negligible
(D_ion · t_settle ≈ 0.3 nm). If the transient solve fails, the solver
falls back to the dark equilibrium.

*Source:* `perovskite_sim/solver/illuminated_ss.py`

### Built-in Potential

`DeviceStack.compute_V_bi()` derives the built-in potential from the
Fermi-level difference across the electrical layers (accounting for χ,
E_g, doping, and nᵢ). When all layers have `chi = Eg = 0`
(homojunction/legacy configs), it falls back to the manual `V_bi` field
(default: 1.1 V).

### Summary Table

| Variable | Contact BCs | Type | Source file |
|----------|-------------|------|-------------|
| φ | φ(0) = 0, φ(L) = V_bi − V_app | Dirichlet | `physics/poisson.py` |
| n | n(0) = n_L, n(L) = n_R (equilibrium) | Dirichlet | `solver/mol.py` |
| p | p(0) = p_L, p(L) = p_R (equilibrium) | Dirichlet | `solver/mol.py` |
| P (ions) | F(0) = F(L) = 0 | Neumann (zero-flux) | `physics/ion_migration.py` |
| P⁻ (neg ions) | F(0) = F(L) = 0 | Neumann (zero-flux) | `physics/ion_migration.py` |

## Python-only quick start

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE: {result.metrics_fwd.PCE*100:.2f} %")
```

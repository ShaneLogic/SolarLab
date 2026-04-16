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

![Device Structure](docs/images/device_structure.png?v=2)

### Band Diagram

![Energy Band Diagram](docs/images/band_diagram.png?v=2)

### Transport Processes and Boundary Conditions

![Transport Processes](docs/images/transport_equations.png?v=2)

### Solver Pipeline

![Solver Pipeline](docs/images/solver_pipeline.png?v=2)

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
| Poisson | $\partial/\partial x(\varepsilon_0 \varepsilon_r\, \partial\varphi/\partial x) = -\rho$ |
| Electron continuity | $\partial n/\partial t = (1/q)\, \partial J_n/\partial x + G - R$ |
| Hole continuity | $\partial p/\partial t = -(1/q)\, \partial J_p/\partial x + G - R$ |
| Ion (vacancy) continuity | $\partial P/\partial t = -\partial F_P/\partial x$ |

State vector per grid node: $\mathbf{y} = (n, p, P)$ — electron density, hole
density, and positive-ion (vacancy) density. Dual-species mode adds a
negative-ion field $P^-$.

### Boundary Conditions

#### Electrostatic potential (Poisson equation) — Dirichlet

| Contact | Value |
|---------|-------|
| Left ($x = 0$) | $\varphi = 0$ (grounded) |
| Right ($x = L$) | $\varphi = V_{\text{bi}} - V_{\text{app}}$ |

Forward bias ($V_{\text{app}} > 0$) reduces the built-in field; $V_{\text{app}} \approx V_{\text{oc}}$ yields
near-open-circuit conditions.

The Poisson operator uses **harmonic-mean face permittivities**:

$$\tilde{\varepsilon}_{i+\frac{1}{2}} = \frac{2\,\varepsilon_r[i]\,\varepsilon_r[i+1]}{\varepsilon_r[i] + \varepsilon_r[i+1]}$$

This is the exact series-capacitor result for a sharp dielectric interface and
avoids the field concentration artefact of nodal averaging.

*Source:* `perovskite_sim/physics/poisson.py`

#### Electron and hole densities — Dirichlet (ohmic contacts)

Both contacts are treated as ideal ohmic contacts. Carrier densities at
the boundaries are clamped to the **thermal-equilibrium values** derived
from the doping of the outermost layers:

$$n \cdot p = n_i^2 \quad\text{(mass-action law)}$$

$$n - p = N_D - N_A \quad\text{(charge neutrality)}$$

Solved via the numerically stable two-branch formula (avoids cancellation
and overflow):

$$\text{net} = N_D - N_A, \qquad \text{disc} = \sqrt{\text{net}^2 + 4\,n_i^2}$$

$$\text{n-type (net} \ge 0\text{):}\quad n = \tfrac{1}{2}(\text{net} + \text{disc}),\quad p = n_i^2 / n$$

$$\text{p-type (net} < 0\text{):}\quad p = \tfrac{1}{2}(-\text{net} + \text{disc}),\quad n = n_i^2 / p$$

These values are computed once per experiment in `build_material_arrays()`
and stored as `n_L, p_L, n_R, p_R`. The time derivatives at the contact
nodes are set to zero (`dn[0] = dn[-1] = dp[0] = dp[-1] = 0`) so the
Dirichlet values remain constant throughout the transient.

*Source:* `perovskite_sim/solver/mol.py` (lines 369-384, 523-524, 564-566)

#### Ion (vacancy) densities — Neumann (zero-flux)

Ions cannot leave the device. At both contacts the vacancy flux is set to
zero:

$$F_P(x = 0) = F_P(x = L) = 0$$

Implemented by padding the internal flux array with zeros at each end
before computing the finite-difference divergence. The same zero-flux
condition applies to both positive and negative ion species.

A **steric saturation limit** prevents unphysical ion pile-up:

$$\text{steric} = \frac{1}{\max(1 - P_{\text{avg}}/P_{\text{lim}},\; 10^{-6})}$$

*Source:* `perovskite_sim/physics/ion_migration.py`

#### Thermionic emission at heterointerfaces

At internal interfaces where the conduction-band offset $|\Delta E_c|$ or
valence-band offset $|\Delta E_v|$ exceeds 0.05 eV, the Scharfetter-Gummel
carrier flux is capped to the **Richardson-Dushman thermionic emission
limit**. This prevents the SG scheme from overestimating current across
sharp band discontinuities (a known single-grid-spacing artefact).
Interface faces where TE activates are pre-computed in `MaterialArrays`.

*Source:* `perovskite_sim/physics/continuity.py`,
`perovskite_sim/discretization/fe_operators.py`

#### Interface recombination

At each heterointerface, surface recombination is parameterised by
velocities $(v_n, v_p)$ [m/s] carried in `DeviceStack.interfaces`. The
surface SRH rate is converted to a volumetric rate by dividing by the
local dual-grid cell width.

### Initial Conditions

#### Dark equilibrium (default)

The default initial state is a **quasi-neutral dark equilibrium** with a
neutral ionic background. At every grid node:

$$n \cdot p = n_i^2(\text{layer}) \quad\text{(mass-action law)}$$

$$n - p = N_D - N_A \quad\text{(charge neutrality, ions treated as neutral background)}$$

The configured vacancy density $P_0$ is treated as a **neutral ionic
background** — it does not appear as net space charge in the initial
carrier balance. This avoids the enormous artificial carrier imbalance
that arises when $P_0$ is treated as net positive charge.

The ion profile is initialised to the uniform per-layer value `P_ion0`.
Contact nodes are overwritten with the ohmic-contact equilibrium densities.

*Source:* `perovskite_sim/solver/newton.py`

#### Illuminated steady-state (light-soaked)

For experiments that begin under illumination (J-V sweep, impedance,
degradation), the initial state is obtained by **integrating the full MOL
system for $t_{\text{settle}} = 1$ ms** starting from dark equilibrium, under
illumination at the starting voltage:

```python
y_dark  = solve_equilibrium(x, stack)
y_light = run_transient(x, y_dark, [0, t_settle], illuminated=True, V_app)
```

Carrier dynamics equilibrate on a sub-microsecond timescale, so 1 ms is
more than sufficient. Ion displacement over this interval is negligible
($D_{\text{ion}} \cdot t_{\text{settle}} \approx 0.3$ nm). If the transient solve fails, the solver
falls back to the dark equilibrium.

*Source:* `perovskite_sim/solver/illuminated_ss.py`

### Built-in Potential

`DeviceStack.compute_V_bi()` derives the built-in potential from the
Fermi-level difference across the electrical layers (accounting for $\chi$,
$E_g$, doping, and $n_i$). When all layers have `chi = Eg = 0`
(homojunction/legacy configs), it falls back to the manual `V_bi` field
(default: 1.1 V).

### Summary Table

| Variable | Contact BCs | Type | Source file |
|----------|-------------|------|-------------|
| $\varphi$ | $\varphi(0) = 0$, $\varphi(L) = V_{\text{bi}} - V_{\text{app}}$ | Dirichlet | `physics/poisson.py` |
| $n$ | $n(0) = n_L$, $n(L) = n_R$ (equilibrium) | Dirichlet | `solver/mol.py` |
| $p$ | $p(0) = p_L$, $p(L) = p_R$ (equilibrium) | Dirichlet | `solver/mol.py` |
| $P$ (ions) | $F(0) = F(L) = 0$ | Neumann (zero-flux) | `physics/ion_migration.py` |
| $P^-$ (neg ions) | $F(0) = F(L) = 0$ | Neumann (zero-flux) | `physics/ion_migration.py` |

## Python-only quick start

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
print(f"PCE: {result.metrics_fwd.PCE*100:.2f} %")
```

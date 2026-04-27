<div align="center">

# ☀️ SolarLab

**Thin-Film Solar Cell Simulator**

*1D Drift-Diffusion · Poisson · Mobile Ions · Transfer-Matrix Optics*

Perovskite · CIGS · c-Si

<br>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/frontend-Vite%20%2B%20TypeScript-646CFF.svg)](https://vitejs.dev)
[![License](https://img.shields.io/badge/license-research-lightgrey.svg)](#)

</div>

<br>

## Table of Contents

| | Section | Description |
|--:|---------|-------------|
| 1 | [Overview](#overview) | What SolarLab does |
| 2 | [Key Features](#key-features) | Capabilities at a glance |
| 3 | [Repository Layout](#repository-layout) | Directory structure |
| 4 | [Installation](#installation) | Setup instructions |
| 5 | [Running the Application](#running-the-application) | Backend + frontend startup |
| 6 | [Physical Principles & Equations](#physical-principles--equations) | Governing PDEs and physics |
| 7 | [Numerical Method](#numerical-method) | Solver architecture |
| 8 | [Using the Web UI](#using-the-web-ui) | UI walkthrough |
| 9 | [Shipped Device Presets](#shipped-device-presets) | Available YAML configs |
| 10 | [Testing](#testing) | Test suite overview |
| 11 | [References](#references) | Key literature |

<br>

---

## Overview

**SolarLab** is a research-grade simulator for thin-film solar cells. The core is a one-dimensional **drift-diffusion + Poisson + mobile-ion** solver backed by a **FastAPI** HTTP service and a **Vite / TypeScript / Plotly** single-page web application.

It reproduces eight kinds of experiments from a single device definition, grouped into four families:

| Family | Experiment | What it does |
|:-------|:-----------|:-------------|
| **Illuminated J-V** | J-V sweep | Forward + reverse scans with ionic memory preserved (hysteresis); optional current decomposition ($J_n$ / $J_p$ / $J_\text{ion}$ / $J_\text{disp}$) and spatial profiles ($\varphi$, $E$, $n$, $p$, $P$) per voltage point |
|  | Suns-V<sub>oc</sub> | Intensity sweep producing the Sinton pseudo-JV (immune to series resistance) and pseudo-FF upper bound |
| **Dark characterisation** | Dark J-V | Injection-current diode curve with auto-selected linear window → ideality factor $n$ and saturation current $J_0$ |
|  | Mott-Schottky (C-V) | Dark C-V + auto-windowed $1/C^2$ fit → built-in voltage $V_\text{bi}$ and effective dopant density $N_\text{eff}$ |
| **Spectral** | EQE / IPCE | Wavelength-resolved external quantum efficiency via monochromatic TMM; integrates against AM1.5G for $J_\text{sc}$ |
| **Transient** | Impedance spectroscopy | Lock-in extraction of $Z(f)$ with displacement current; Nyquist + Bode plots |
|  | Transient photovoltage (TPV) | Small-signal light-pulse perturbation at open circuit; mono-exponential fit for recombination lifetime $\tau$ |
|  | Degradation | Long-time transient with a **frozen-ion snapshot J-V** at each probe, decoupling slow ionic drift from the instantaneous electronic response |

The simulator works for perovskite cells (with mobile ions), inorganic thin films (CIGS, CdTe-style stacks), and crystalline silicon homojunctions — all through the same YAML-based device schema. A separate **2T monolithic tandem** driver performs a combined TMM over top + junction + bottom, runs independent sub-cell J-V sweeps, and series-matches at a common current grid.

<br>

---

## Key Features

### Core solver

| | Capability | Details |
|:-:|:-----------|:--------|
| 🧮 | **Drift-diffusion core** | Scharfetter-Gummel finite-element fluxes on a tanh-clustered multilayer grid; Method-of-Lines with Radau implicit time integration |
| ⚡ | **Poisson solve** | Pre-factored LAPACK tridiagonal `dgttrf`/`dgttrs` (cached once per run) — ~40× faster than naive assembly |
| 🔬 | **Mobile ions** | Steric Blakemore flux with excluded-volume correction; dual-ion support (mobile cation + anion vacancies) |
| 🏗️ | **Heterostacks** | Band offsets from $\chi$ and $E_g$; per-interface $(v_n, v_p)$ surface recombination; auto-computed $V_\text{bi}$ from Fermi-level difference |

### Physics upgrades (Phase 2–4)

| | Capability | Details |
|:-:|:-----------|:--------|
| 🌈 | **Transfer-matrix optics** | Coherent TMM for position-resolved $G(x)$ with the Poynting-vector correction so $R+T+A=1$; AM1.5G-weighted per-wavelength absorption |
| 🔥 | **Thermionic emission** | Richardson-Dushman flux cap at heterointerfaces with band offsets $> 0.05$ eV — prevents SG over-current across sharp discontinuities |
| 💡 | **Photon recycling** | Yablonovitch single-pass escape probability $P_\text{esc} = \min(1, 1/(4 n^2 \alpha d))$ scales $B_\text{rad}$ per absorber (Phase 3.1) |
| 🔁 | **Self-consistent reabsorption** | Per-RHS radiative reabsorption feeds the trapped emission fraction back into $G(x)$ on absorber nodes (Phase 3.1b, FULL only) |
| 🌀 | **Field-dependent mobility** | Caughey-Thomas velocity saturation + Poole-Frenkel hopping applied per RHS from the Poisson face field (Phase 3.2, FULL only) |
| 🚪 | **Selective / Schottky contacts** | Robin-type flux $J = \pm q S (n - n_\text{eq})$ at outer contacts; $S \to \infty$ is ohmic, $S = 0$ is blocking (Phase 3.3, FULL only) |
| 🎯 | **Position-dependent traps** | Exponential or Gaussian defect profiles concentrated at transport-layer interfaces; $\tau(x) = \tau_\text{bulk} \cdot N_{t,\text{bulk}} / N_t(x)$ (Phase 4a) |
| 🌡️ | **Temperature scaling** | Varshni bandgap shift referenced to 300 K and power-law $B_\text{rad}(T) \propto (T/300)^\gamma$; self-consistent with $n_i(T)$ (Phase 4b) |

### Tiered fidelity modes

| Tier | Physics set | Use case |
|:-----|:------------|:---------|
| **LEGACY** | No TE, no TMM, uniform traps, $T = 300$ K — classic IonMonger reproduction | Regression parity against published benchmarks |
| **FAST** | All build-once upgrades on (TE, TMM, dual ions, trap profile, $T$-scaling, photon recycling); per-RHS hooks off | Default for J-V / impedance / degradation sweeps |
| **FULL** | Everything on, including per-RHS radiative reabsorption, $\mu(E)$, and Robin contacts | Highest-fidelity single runs; tandem sub-cells |

### Experiments (single device → 8 observables)

| | Capability | Details |
|:-:|:-----------|:--------|
| 📈 | **J-V sweep** | Forward + reverse scans with ionic memory preserved (hysteresis from physics, not post-processing); optional current decomposition and spatial-profile export |
| ☀️ | **Suns-V<sub>oc</sub>** | Intensity sweep → Sinton pseudo-JV (series-resistance-free) and pseudo-FF upper bound |
| 🌓 | **Dark J-V** | Auto-windowed linear fit on $\ln J$ vs $V$ → ideality factor $n$ and saturation current $J_0$ |
| 📊 | **Mott-Schottky** | Dark C-V with $1/C^2$ linearisation → built-in voltage $V_\text{bi}$ and effective dopant density $N_\text{eff}$ |
| 🌈 | **EQE / IPCE** | Wavelength-resolved external quantum efficiency via monochromatic TMM; AM1.5G integration yields $J_\text{sc}$ |
| 🎵 | **Impedance** | Lock-in $Z(f)$ with displacement current; Nyquist + Bode output |
| ⚡ | **Transient photovoltage** | Small-signal light pulse at open circuit with mono-exponential fit for recombination lifetime $\tau$ |
| 🧊 | **Frozen-ion degradation** | Long-time transient with snapshot J-V at each probe ($D_\text{ion} \to 0$) — decouples ionic drift from electronic response |
| 🔗 | **2T tandem driver** | Combined TMM over top + junction + bottom, independent sub-cell sweeps, series-matched on a common current grid |

### Dimensionality

| | Capability | Details |
|:-:|:-----------|:--------|
| 📏 | **1D drift-diffusion (default)** | All experiments above run on a tanh-clustered multilayer 1D grid with the cached `MaterialArrays` hot path |
| 🟦 | **2D Stage A (lateral-uniform)** | Tensor-product (Ny × Nx) grid with sparse 5-point Poisson, vectorised 2D Scharfetter-Gummel fluxes, periodic / Neumann lateral BCs; bootstraps from the 1D illuminated steady state and freezes ions as a static Poisson background. On a laterally-uniform stack the 2D solver reproduces the 1D J-V to within sub-mV $V_\text{oc}$, $5 \times 10^{-4}$ relative $J_\text{sc}$, and $10^{-3}$ FF — pinned by `tests/regression/test_twod_validation.py`. Available as `kind='jv_2d'` from the backend and as the **J-V Sweep (2D)** entry in the workstation experiment selector. |

### Tooling

| | Capability | Details |
|:-:|:-----------|:--------|
| 🖥️ | **Interactive web UI** | Live SSE streaming, grouped experiment dropdown, GoldenLayout dockable panes, Plotly plots, layer-builder editor |
| 🧪 | **Full test suite** | Unit, integration, and physics regression tests ($V_\text{oc}$ / $J_\text{sc}$ / FF / HI bounds; TMM $R+T+A$ conservation; 1D ↔ 2D parity gate) |

<br>

---

## Repository Layout

```
SolarLab/
├── perovskite-sim/                 Main simulator + backend + frontend
│   ├── perovskite_sim/             Python simulation library
│   │   ├── discretization/         Grid + Scharfetter–Gummel operators
│   │   ├── physics/                Poisson, continuity, recombination, ions, optics
│   │   ├── models/                 Frozen dataclasses: DeviceStack, MaterialParams
│   │   ├── solver/                 Method-of-Lines assembler, Newton equilibrium
│   │   ├── experiments/            J–V sweep, impedance, degradation
│   │   └── data/nk/                Complex refractive-index n,k data for TMM
│   ├── backend/                    FastAPI HTTP wrapper (SSE job streaming)
│   ├── frontend/                   Vite + TypeScript + Plotly single-page UI
│   ├── configs/                    Shipped YAML device presets
│   ├── tests/                      pytest suite (unit / integration / regression)
│   └── notebooks/                  Exploratory benchmarks
├── perovskite-sim-phase2b/         Git worktree for feat/tandem-cell
├── docs/                           Research notes, plans, specs
└── README.md                       (this file)
```

<br>

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **Node.js 18+** with `npm`
- A C compiler + BLAS/LAPACK (bundled with `numpy`/`scipy` wheels on most platforms)

### 1. Clone the repository

```bash
git clone https://github.com/ShaneLogic/SolarLab.git
cd SolarLab
```

### 2. Install the Python package

```bash
cd perovskite-sim
pip install -e ".[dev]"
```

This installs `perovskite_sim` as an editable package along with `numpy`, `scipy`, `fastapi`, `uvicorn`, `pytest`, and the dev tooling.

> 💡 **Recommended:** use a virtual environment.
> ```bash
> python -m venv .venv && source .venv/bin/activate
> ```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

### 4. Verify the install

```bash
# From perovskite-sim/
pytest                                    # full unit + integration suite (~15 s)
pytest -m slow                            # slow physics regression (~30 s, BLAS pinned)
```

<br>

---

## Running the Application

SolarLab runs as **two processes**: a FastAPI backend that executes the simulations, and a Vite dev server that serves the UI.

### Start the backend

From the **SolarLab root** (not `perovskite-sim/`):

```bash
uvicorn backend.main:app \
    --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload
```

Check it is alive:

```bash
curl http://127.0.0.1:8000/api/configs
```

You should get JSON listing the shipped presets.

### Start the frontend

In a second terminal:

```bash
cd perovskite-sim/frontend
npm run dev
```

Open **<http://127.0.0.1:5173>** in your browser. The UI will connect directly to `http://127.0.0.1:8000` (the backend URL is hardcoded — CORS is already open, so this works under both `npm run dev` and `npm run preview`).

### Run a simulation from Python (no UI)

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)

print(f"PCE:  {result.metrics_fwd.PCE*100:.2f} %")
print(f"V_oc: {result.metrics_fwd.V_oc:.3f} V")
print(f"J_sc: {result.metrics_fwd.J_sc:.1f} A/m²")
print(f"FF:   {result.metrics_fwd.FF:.3f}")
print(f"Hysteresis index: {result.hysteresis_index:.3f}")
```

<br>

---

## Physical Principles & Equations

### Physical Model Overview

<p align="center">
  <img src="perovskite-sim/docs/images/device_structure.png?v=2" alt="Device Structure" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/band_diagram.png?v=3" alt="Energy Band Diagram" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/transport_equations.png?v=2" alt="Transport Processes and Boundary Conditions" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/solver_pipeline.png?v=2" alt="Solver Pipeline" width="700">
</p>

### Governing Equations

SolarLab solves the coupled **Poisson + drift-diffusion + mobile-ion** system in one spatial dimension. State variables at every grid node are the electron density $n$, hole density $p$, and the mobile-ion density $P$.

<br>

### 1. Poisson's Equation

The electrostatic potential $\varphi(x,t)$ satisfies

$$
-\frac{\partial}{\partial x}\!\left(\varepsilon_0 \varepsilon_r(x)\,\frac{\partial \varphi}{\partial x}\right) = q\bigl(p - n + N_D(x) - N_A(x) + P - P_0(x)\bigr)
$$

with Dirichlet boundaries

$$
\varphi(0,t) = 0, \qquad \varphi(L,t) = V_{\text{bi}} - V_{\text{app}}(t).
$$

$V_{\text{bi}}$ is computed from the Fermi-level offset across the heterostack (`DeviceStack.compute_V_bi()`), accounting for $\chi$, $E_g$, doping, and $n_i$. The operator is discretized with a harmonic-mean face permittivity and pre-factored once per run (LAPACK `dgttrf`), then solved at every RHS call with a single `dgttrs` sweep.

<br>

### 2. Carrier Continuity (Drift-Diffusion)

Electrons and holes obey

$$
\frac{\partial n}{\partial t} = \frac{1}{q}\frac{\partial J_n}{\partial x} + G(x) - R(n,p,x), \qquad
\frac{\partial p}{\partial t} = -\frac{1}{q}\frac{\partial J_p}{\partial x} + G(x) - R(n,p,x)
$$

with the conventional drift-diffusion fluxes

$$
J_n = q\mu_n n E + qD_n\frac{\partial n}{\partial x}, \qquad
J_p = q\mu_p p E - qD_p\frac{\partial p}{\partial x}, \qquad E = -\frac{\partial\varphi}{\partial x}.
$$

**Scharfetter–Gummel discretization.** The flux between nodes $i$ and $i+1$ is written so it is exact for a constant-flux, exponential-profile solution of the local drift-diffusion equation:

$$
J_{n,\,i+\tfrac12} = \frac{qD_n}{\Delta x_i}\Bigl[n_{i+1}\,B(\Delta\varphi_i/V_t) - n_i\,B(-\Delta\varphi_i/V_t)\Bigr]
$$

where $B(x) = x/(e^{x}-1)$ is the Bernoulli function and $V_t = k_B T / q$. This removes the classical upwind/central-difference stability problem when $|E|\Delta x \gg V_t$.

<br>

### 3. Recombination

The net recombination rate is the sum of Shockley–Read–Hall, radiative (bimolecular), and Auger channels:

$$
R = R_{\text{SRH}} + R_{\text{rad}} + R_{\text{Auger}}
$$

$$
R_{\text{SRH}} = \frac{np - n_i^2}{\tau_p(n + n_1) + \tau_n(p + p_1)}, \qquad
R_{\text{rad}} = k_{\text{rad}}\,(np - n_i^2), \qquad
R_{\text{Auger}} = (C_n n + C_p p)(np - n_i^2).
$$

Interface recombination is applied at heterointerfaces via the per-interface surface-recombination velocities $(v_n, v_p)$ carried in `DeviceStack.interfaces`.

<br>

### 4. Mobile-Ion Migration (Blakemore-Limited)

For perovskite cells, a single mobile ionic species (typically iodide vacancy $V_{\mathrm{I}}^{+}$) is tracked with a **steric Blakemore flux** that enforces an excluded-volume site-density limit $N_{\max}$:

$$
J_P = -qD_{\text{ion}}\Bigl[\frac{\partial P}{\partial x} + \frac{P}{V_t}\frac{\partial \varphi}{\partial x}\Bigr]\cdot\left(1 - \frac{P}{N_{\max}}\right)
$$

The $(1 - P/N_{\max})$ factor prevents the ion density from diverging when the quasi-Fermi level of the ions approaches the site energy. Non-perovskite stacks (CIGS, c-Si) set $D_{\text{ion}}=0$ so this term drops out cleanly.

<br>

### 5. Optical Generation

Two optional models share the same interface.

**Beer–Lambert** (default / fallback):

$$
G_{\text{BL}}(x) = \int_\lambda \alpha(\lambda,x)\,\Phi_0(\lambda)\,\exp\!\left(-\int_0^x \alpha(\lambda,x')\,dx'\right)\,d\lambda.
$$

**Transfer-matrix (coherent thin-film) optics.** When any layer carries an `optical_material` key, `physics/optics.py` loads complex $n(\lambda)$, $k(\lambda)$ CSVs from `perovskite_sim/data/nk/`, builds the layer transfer matrices against AM1.5G, and computes the position-resolved generation rate as

$$
G_{\text{TMM}}(x,\lambda) = \frac{1}{\hbar\omega}\cdot\frac{n(x,\lambda)}{n_{\text{amb}}}\cdot\alpha(x,\lambda)\,|E(x,\lambda)|^2,
$$

integrated over the AM1.5G spectrum. The $n/n_{\text{amb}}$ prefactor is the Poynting-vector correction that guarantees $R+T+A=1$. `G_TMM(x)` is computed once during `build_material_arrays` and cached on `MaterialArrays.G_optical` — the hot path never recomputes optics.

<br>

### 6. Thermionic Emission at Heterointerfaces

When the conduction-band offset $|\Delta E_c|$ or valence-band offset $|\Delta E_v|$ across an interface exceeds $0.05$ eV, the Scharfetter–Gummel flux is capped to the Richardson–Dushman thermionic-emission current

$$
J_{\text{TE},n} = A^*_n T^2 \exp\!\left(-\frac{\Delta E_c}{k_B T}\right)\bigl[\exp(qV/k_B T) - 1\bigr],
$$

where $A^*$ is the effective Richardson constant (defaults to the free-electron value $1.2017\times 10^{6}\,\text{A/m}^2\text{K}^2$, tunable per layer). Without this cap, SG overestimates current across sharp band discontinuities resolved in a single grid spacing.

<br>

### 7. Boundary Conditions at the Contacts

At each metal contact, ohmic boundary conditions fix the majority-carrier density to its bulk equilibrium value, and the minority-carrier boundary flux is the extraction current controlled by an effective surface-recombination velocity $S_{n,p}$:

$$
J_{n}\big|_{0} = -qS_n(n - n_0), \qquad J_{p}\big|_{L} = qS_p(p - p_0).
$$

Ions are blocked at both contacts ($J_P = 0$), enforcing ionic-species conservation.

<br>

---

## Numerical Method

| Ingredient | Choice |
|:-----------|:-------|
| **Method** | Method of Lines: spatial FE discretization -> stiff ODE in time |
| **Grid** | Tanh-clustered multilayer grid (refined near interfaces and contacts) |
| **Spatial** | Scharfetter-Gummel finite elements for drift-diffusion; harmonic-mean faces for Poisson |
| **Time** | `scipy.integrate.solve_ivp` with the **Radau** IIA 5th-order implicit method |
| **Poisson** | LAPACK `dgttrf`/`dgttrs` tridiagonal LU, pre-factored once per run |
| **Current** | Computed directly from the converged SG fluxes (consistent with continuity) |
| **Safety cap** | `max_step` capped on every `run_transient` sub-interval to prevent Radau from accepting a giant step near flat-band |

All physical data is held in **immutable frozen dataclasses** (`MaterialParams`, `LayerSpec`, `DeviceStack`, `SolverConfig`). In-place mutation is forbidden — updates use `dataclasses.replace(...)`.

<br>

---

## Using the Web UI

After launching the backend and frontend (see [Running the Application](#running-the-application)), open **<http://127.0.0.1:5173>**. The UI is split into three regions.

### Layout

<p align="center">
  <img src="perovskite-sim/docs/images/ui_layout.png?v=3" alt="Web UI Layout" width="700">
</p>

### Left Rail — Devices / Results

| Element | Description |
|:--------|:------------|
| **DEVICES** | Active device tab with simulation tier (FAST / FULL / LEGACY). Click to focus configuration. |
| **RESULTS / COMPARE** | Completed runs are archived here. Select two or more to overlay plots. |

### Center Pane — Device Configuration

| Element | Description |
|:--------|:------------|
| **Preset dropdown** | Choose a shipped or user preset. Switching reloads the stack from YAML. |
| **Reset** | Discards unsaved edits and re-loads the last saved version. |
| **Stack Visualizer** | *(Full tier)* Vertical layer column — click to edit, **+** to insert, drag to reorder, **x** to delete. |
| **Detail Editor** | Collapsible groups: Geometry, Transport, Recombination, Ions & Optics. |
| **TMM badge** | Appears when any layer has a non-empty `optical_material`. |
| **Save As** | Save edited stack to `configs/user/` or download YAML directly. |

<details>
<summary><strong>Detail Editor parameter groups</strong></summary>

- **Geometry** — thickness, grid density, role (contact / transport / absorber / substrate)
- **Transport** — $\mu_n$, $\mu_p$, $N_c$, $N_v$, $N_A$, $N_D$, $\chi$, $E_g$, $\varepsilon_r$
- **Recombination** — $\tau_n$, $\tau_p$, $k_{\text{rad}}$, $C_n$, $C_p$, $E_t$
- **Ions & Optics** — $D_{\text{ion}}$, $N_{\max}$, $P_0$, `optical_material`, `n_optical`, `incoherent` flag

</details>

### Right Pane — Experiments

Experiment tabs sharing a common pattern: parameters form -> **Run** button -> live progress bar -> Plotly plot.

#### J-V Sweep

| Parameter | Description |
|:----------|:------------|
| $N_{\text{grid}}$ | Number of spatial nodes |
| V sample points | Number of voltage samples per scan direction |
| Scan rate (V/s) | Ionic memory effects — fast scans produce larger hysteresis |
| $V_{\max}$ | Upper voltage bound (defaults to $V_{\text{bi}}$) |
| Decompose current | Per-face breakdown into $J_n$ / $J_p$ / $J_\text{ion}$ / $J_\text{disp}$ at every voltage |
| Save spatial profiles | Snapshot $\varphi(x)$, $E(x)$, $n(x)$, $p(x)$, $P(x)$ at each voltage |

The experiment runs a **forward** scan (short-circuit to $V_{\max}$) immediately followed by a **reverse** scan, reusing the final state so the ionic population is preserved across the turn. Output: overlaid forward/reverse curves plus metric cards for $V_{\text{oc}}$, $J_{\text{sc}}$, FF, PCE, and hysteresis index. For a dark diode curve, use the dedicated **Dark J-V** experiment, which runs the same $G = 0$ sweep and adds an ideality-factor / $J_0$ fit.

The two optional output views (decomposition, spatial profiles) are mutually exclusive on a single run — pick one per sweep or re-run for the other.

#### Impedance

| Parameter | Description |
|:----------|:------------|
| Frequency sweep | $\omega_{\min}$, $\omega_{\max}$, $N_\omega$ |
| DC bias | Steady-state bias voltage |
| AC amplitude | Small-signal perturbation |

At each frequency, the solver integrates several AC cycles, then a lock-in amplifier extracts amplitude and phase. Displacement current $\varepsilon_0 \varepsilon_r \, \partial E / \partial t$ is included. Output: Nyquist plot and Bode magnitude/phase curves.

#### Degradation

| Parameter | Description |
|:----------|:------------|
| Total time | Simulation duration |
| Number of probes | Snapshot count over the simulation |
| Probe bias | Voltage for snapshot J-V |

At each probe time, the solver takes a **frozen-ion snapshot**: a copy of the stack with $D_{\text{ion}}=0$ measures the instantaneous $J(V)$ response. This decouples slow ionic drift from the electronic response. Output: PCE / $V_{\text{oc}}$ / $J_{\text{sc}}$ vs aging time.

#### Transient Photovoltage (TPV)

| Parameter | Description |
|:----------|:------------|
| $N_{\text{grid}}$ | Number of spatial nodes |
| $\delta G$ fraction | Fractional generation perturbation (e.g. 0.05 = 5% pulse) |
| Pulse duration | Duration of the light pulse [s] |
| Observation window | Total time including decay [s] |

The device is equilibrated at open circuit under steady illumination, then a small light pulse is applied. The voltage transient $V(t)$ decays back to $V_\text{oc}$ as excess carriers recombine. A mono-exponential fit extracts the effective recombination lifetime $\tau$. Output: $V(t)$ decay curve, $J(t)$ transient, fitted $\tau$.

### Docs Tabs — Tutorial & Algorithm

The **Tutorial** pane is a guided walkthrough (Device Setup -> First Simulation -> Interpreting Results -> Advanced Topics). The **Algorithm** pane is a formal write-up of the PDEs, discretization, solver tiers, and the transfer-matrix optical model. Both are always available — no backend required.

<br>

---

## Shipped Device Presets

All presets live in `perovskite-sim/configs/`. Drop a new `.yaml` file there and it is auto-discovered by `GET /api/configs`.

| Preset | Material System | Ions | Optics | Notes |
|:-------|:----------------|:----:|:------:|:------|
| `nip_MAPbI3` | MAPbI3 n-i-p | Yes | Beer-Lambert | Canonical perovskite reference |
| `pin_MAPbI3` | MAPbI3 p-i-n | Yes | Beer-Lambert | Inverted-architecture reference |
| `nip_MAPbI3_tmm` | MAPbI3 n-i-p + glass | Yes | TMM | TMM-enabled with 1 mm substrate |
| `pin_MAPbI3_tmm` | MAPbI3 p-i-n + glass | Yes | TMM | TMM-enabled |
| `ionmonger_benchmark` | Courtier 2019 | Yes | Beer-Lambert | IonMonger cross-check |
| `driftfusion_benchmark` | Driftfusion params | Yes | Beer-Lambert | Driftfusion cross-check |
| `cigs_baseline` | ZnO / CdS / CIGS | No | Beer-Lambert | $D_{\text{ion}}=0$ everywhere |
| `cSi_homojunction` | n+ / p c-Si | No | Beer-Lambert | Homojunction wafer cell |

<br>

---

## Testing

```bash
# From perovskite-sim/
pytest                                                  # unit + integration, ~15 s
pytest -m slow                                          # physics regression, ~30 s
pytest --cov=perovskite_sim --cov-report=term-missing   # with coverage report
pytest tests/unit/experiments/test_jv_sweep.py          # a single file
```

| Suite | Scope |
|:------|:------|
| **Unit** | Per-module physics + solver coverage |
| **Integration** | End-to-end experiment runs on shipped presets |
| **Regression** | Physical sanity envelopes ($V_{\text{oc}}$, $J_{\text{sc}}$, HI bounds); BLAS pinned via `conftest.py` |

<br>

---

## References

1. **Scharfetter, D. L. & Gummel, H. K.** (1969) — *Large-signal analysis of a silicon Read diode oscillator*. Foundational SG flux scheme.
2. **Courtier, N. E. et al.** (2019) — *IonMonger: a free and fast planar perovskite solar cell simulator with coupled ion vacancy and charge carrier dynamics*. J. Comput. Electron.
3. **Calado, P. et al.** — *Driftfusion: an open source code for simulating ordered semiconductor devices*. Reference MATLAB implementation used for cross-checking.
4. **Pettersson, L. A. A. et al.** (1999) — *Modeling photocurrent action spectra of photovoltaic devices based on organic thin films*. J. Appl. Phys.
5. **Burkhard, G. F. et al.** (2010) — *Accounting for interference, scattering, and electrode absorption to make accurate internal quantum efficiency measurements in organic and other thin solar cells*. Adv. Mater.
6. **Richardson, G. et al.** — Theoretical basis for ion-migration modelling in perovskite solar cells.

---

<div align="center">

Made with 🧪 and ☕ at **HKUST Guangzhou · SolarLab**

</div>

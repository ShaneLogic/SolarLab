<div align="center">

# ☀️ SolarLab

### Thin-Film Solar Cell Simulator

*1D Drift-Diffusion · Poisson · Mobile Ions · Transfer-Matrix Optics*

**Perovskite · CIGS · c-Si**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/frontend-Vite%20%2B%20TypeScript-646CFF.svg)](https://vitejs.dev)
[![License](https://img.shields.io/badge/license-research-lightgrey.svg)](#)

</div>

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Repository Layout](#-repository-layout)
4. [Installation](#-installation)
5. [Running the Application](#-running-the-application)
6. [Physical Principles & Equations](#-physical-principles--equations)
7. [Numerical Method](#-numerical-method)
8. [Using the Web UI](#-using-the-web-ui)
9. [Shipped Device Presets](#-shipped-device-presets)
10. [Testing](#-testing)
11. [References](#-references)

---

## 🌐 Overview

**SolarLab** is a research-grade simulator for thin-film solar cells. The core is a one-dimensional **drift-diffusion + Poisson + mobile-ion** solver backed by a **FastAPI** HTTP service and a **Vite / TypeScript / Plotly** single-page web application.

It reproduces three kinds of experiments from a single device definition:

- **J–V sweep** with hysteresis (forward + reverse scans, ionic memory preserved)
- **Impedance spectroscopy** (Nyquist diagrams via lock-in detection of AC response)
- **Degradation** (long-time transient with frozen-ion snapshot J–V at each probe)

The simulator works for perovskite cells (with mobile ions), inorganic thin films (CIGS, CdTe-style stacks), and crystalline silicon homojunctions — all through the same YAML-based device schema.

---

## ✨ Key Features

| Capability | Details |
|---|---|
| 🧮 **Drift-diffusion core** | Scharfetter–Gummel finite-element fluxes on a tanh-clustered multilayer grid |
| ⚡ **Poisson solve** | Pre-factored LAPACK tridiagonal `dgttrf`/`dgttrs` — ~40× faster than naive assembly |
| 🔬 **Mobile ions** | Steric Blakemore flux with excluded-volume correction (perovskite vacancy migration) |
| 🌈 **Transfer-matrix optics** | Coherent TMM for position-resolved $G(x)$ with the Poynting-vector correction so $R+T+A=1$ |
| 🔥 **Thermionic emission** | Richardson–Dushman flux cap at heterointerfaces with band offsets $>0.05$ eV |
| 🏗️ **Heterostacks** | Band offsets from electron affinity $\chi$ and bandgap $E_g$; per-interface $(v_n, v_p)$ surface recombination |
| 🧊 **Frozen-ion degradation** | Snapshot J–V at each probe time with $D_{\text{ion}}\to 0$, the only physically correct way to decouple ionic drift from electronic response |
| 🖥️ **Interactive web UI** | Live experiment streaming over Server-Sent Events, progress bars, Plotly plots, layer-builder editor |
| 🧪 **Full test suite** | Unit, integration, and physics regression tests (V_oc / J_sc / hysteresis bounds) |

---

## 🗂️ Repository Layout

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

---

## 🛠️ Installation

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

---

## 🚀 Running the Application

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

---

## 🔬 Physical Principles & Equations

### Physical Model Overview

<p align="center">
  <img src="perovskite-sim/docs/images/device_structure.png?v=2" alt="Device Structure" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/band_diagram.png?v=2" alt="Energy Band Diagram" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/transport_equations.png?v=2" alt="Transport Processes and Boundary Conditions" width="700">
</p>

<p align="center">
  <img src="perovskite-sim/docs/images/solver_pipeline.png?v=2" alt="Solver Pipeline" width="700">
</p>

### Equations

SolarLab solves the coupled **Poisson + drift-diffusion + mobile-ion** system in one spatial dimension. State variables at every grid node are the electron density $n$, hole density $p$, and the mobile-ion density $P$.

### 1. Poisson's equation

The electrostatic potential $\varphi(x,t)$ satisfies

$$
-\frac{\partial}{\partial x}\!\left(\varepsilon_0 \varepsilon_r(x)\,\frac{\partial \varphi}{\partial x}\right) = q\bigl(p - n + N_D(x) - N_A(x) + P - P_0(x)\bigr)
$$

with Dirichlet boundaries

$$
\varphi(0,t) = 0, \qquad \varphi(L,t) = V_{\text{bi}} - V_{\text{app}}(t).
$$

$V_{\text{bi}}$ is computed from the Fermi-level offset across the heterostack (`DeviceStack.compute_V_bi()`), accounting for $\chi$, $E_g$, doping, and $n_i$. The operator is discretized with a harmonic-mean face permittivity and pre-factored once per run (LAPACK `dgttrf`), then solved at every RHS call with a single `dgttrs` sweep.

### 2. Carrier continuity (drift-diffusion)

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

### 4. Mobile-ion migration (Blakemore-limited)

For perovskite cells, a single mobile ionic species (typically iodide vacancy $V_{\mathrm{I}}^{+}$) is tracked with a **steric Blakemore flux** that enforces an excluded-volume site-density limit $N_{\max}$:

$$
J_P = -qD_{\text{ion}}\Bigl[\frac{\partial P}{\partial x} + \frac{P}{V_t}\frac{\partial \varphi}{\partial x}\Bigr]\cdot\left(1 - \frac{P}{N_{\max}}\right)
$$

The $(1 - P/N_{\max})$ factor prevents the ion density from diverging when the quasi-Fermi level of the ions approaches the site energy. Non-perovskite stacks (CIGS, c-Si) set $D_{\text{ion}}=0$ so this term drops out cleanly.

### 5. Optical generation

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

### 6. Thermionic emission at heterointerfaces

When the conduction-band offset $|\Delta E_c|$ or valence-band offset $|\Delta E_v|$ across an interface exceeds $0.05$ eV, the Scharfetter–Gummel flux is capped to the Richardson–Dushman thermionic-emission current

$$
J_{\text{TE},n} = A^*_n T^2 \exp\!\left(-\frac{\Delta E_c}{k_B T}\right)\bigl[\exp(qV/k_B T) - 1\bigr],
$$

where $A^*$ is the effective Richardson constant (defaults to the free-electron value $1.2017\times 10^{6}\,\text{A/m}^2\text{K}^2$, tunable per layer). Without this cap, SG overestimates current across sharp band discontinuities resolved in a single grid spacing.

### 7. Boundary conditions at the contacts

At each metal contact, ohmic boundary conditions fix the majority-carrier density to its bulk equilibrium value, and the minority-carrier boundary flux is the extraction current controlled by an effective surface-recombination velocity $S_{n,p}$:

$$
J_{n}\big|_{0} = -qS_n(n - n_0), \qquad J_{p}\big|_{L} = qS_p(p - p_0).
$$

Ions are blocked at both contacts ($J_P = 0$), enforcing ionic-species conservation.

---

## 🧮 Numerical Method

| Ingredient | Choice |
|---|---|
| **Method** | Method of Lines: spatial FE discretization → stiff ODE in time |
| **Grid** | Tanh-clustered multilayer grid (refined near interfaces and contacts) |
| **Spatial** | Scharfetter–Gummel finite elements for drift-diffusion; harmonic-mean faces for Poisson |
| **Time** | `scipy.integrate.solve_ivp` with the **Radau** IIA 5th-order implicit method |
| **Poisson** | LAPACK `dgttrf`/`dgttrs` tridiagonal LU, pre-factored once per run |
| **Current** | Computed directly from the converged SG fluxes (consistent with continuity) |
| **Safety cap** | `max_step = Δt / k` on every `run_transient` sub-interval, to stop Radau from accepting a giant step on the wrong branch near flat-band |

All physical data is held in **immutable frozen dataclasses** (`MaterialParams`, `LayerSpec`, `DeviceStack`, `SolverConfig`). In-place mutation is forbidden anywhere in the library — updates use `dataclasses.replace(...)`.

---

## 🖥️ Using the Web UI

After launching the backend and frontend (see [Running the Application](#-running-the-application)), open **<http://127.0.0.1:5173>**. The UI is split into three regions.

### Layout

<p align="center">
  <img src="perovskite-sim/docs/images/ui_layout.png?v=2" alt="Web UI Layout" width="700">
</p>

### Left rail — Devices / Results

- **DEVICES** — shows the active device tab and its simulation tier (FAST / FULL / LEGACY). Click a device to focus its configuration.
- **RESULTS / COMPARE** — every completed run is archived here. Select two or more runs to overlay their plots.

### Center pane — Device Configuration

1. **Preset dropdown** — choose a shipped or user preset. Switching presets reloads the stack from YAML.
2. **Reset button** — discards unsaved edits and re-loads the last saved version.
3. **Stack Visualizer** *(full tier only)* — a vertical column showing every layer in physical order. Click a layer to open it in the Detail Editor; use **➕** between layers to insert from the template library; drag handles to reorder; **✕** to delete.
4. **Detail Editor** — collapsible groups:
   - **Geometry** — thickness, grid density, role (contact / transport / absorber / substrate)
   - **Transport** — $\mu_n$, $\mu_p$, $N_c$, $N_v$, $N_A$, $N_D$, $\chi$, $E_g$, $\varepsilon_r$
   - **Recombination** — $\tau_n$, $\tau_p$, $k_{\text{rad}}$, $C_n$, $C_p$, $E_t$
   - **Ions & Optics** — $D_{\text{ion}}$, $N_{\max}$, $P_0$, `optical_material`, `n_optical`, `incoherent` flag
5. **TMM badge** — in full tier, a `TMM active · N layers` pill appears in the header whenever at least one layer has a non-empty `optical_material`.
6. **Save As / Download YAML** — in the visualizer's action row, save your edited stack to `configs/user/` (visible to all future runs) or download the YAML directly.

### Right pane — Experiments

Three tabs sharing a common pattern: parameters form → **Run** button → live progress bar → Plotly plot.

#### 📈 J–V Sweep

Parameters:

- $N_{\text{grid}}$ — number of spatial nodes
- **V sample points** — number of voltage samples on each scan direction
- **Scan rate (V/s)** — determines ionic memory effects; fast scans see larger hysteresis
- $V_{\max}$ — upper voltage bound (defaults to $V_{\text{bi}}$)

The experiment runs a **forward** scan (short-circuit → $V_{\max}$) immediately followed by a **reverse** scan, reusing the final state so the ionic population is preserved across the turn. Output: overlaid forward/reverse curves plus metric cards for $V_{\text{oc}}$, $J_{\text{sc}}$, FF, PCE, and hysteresis index.

#### 🌀 Impedance

Parameters: frequency sweep $(\omega_{\min}, \omega_{\max}, N_\omega)$, DC bias, AC amplitude.

At each frequency the solver integrates several AC cycles, then a lock-in amplifier (in-phase/quadrature multiply + low-pass) extracts amplitude and phase. Displacement current $\varepsilon_0\varepsilon_r\,\partial E/\partial t$ is included. Output: Nyquist plot (real vs imaginary $Z$) and Bode magnitude/phase curves.

#### ⏳ Degradation

Parameters: total simulation time, number of probes, probe bias.

At each probe time, the solver takes a **frozen-ion snapshot**: it creates a `replace`-d copy of the stack with $D_{\text{ion}}=0$ and runs a short settle integration to measure the instantaneous $J(V)$ response. This decouples the slow ionic drift from the electronic response and is the only physically correct way to extract snapshot PCE decay over device lifetime. Output: PCE / $V_{\text{oc}}$ / $J_{\text{sc}}$ vs aging time.

### Docs tabs — Tutorial & Algorithm

The **Tutorial** pane is a guided walkthrough (Device Setup → First Simulation → Interpreting Results → Advanced Topics). The **Algorithm** pane is a formal write-up of the PDEs, discretization, solver tiers, and the transfer-matrix optical model. Both are always available — no backend required.

---

## 📦 Shipped Device Presets

All presets live in `perovskite-sim/configs/`. Drop a new `.yaml` file there and it is auto-discovered by `GET /api/configs`.

| Preset | Material system | Mobile ions | Optics | Notes |
|---|---|---|---|---|
| `nip_MAPbI3.yaml` | MAPbI₃ n-i-p | ✓ | Beer–Lambert | Canonical perovskite reference |
| `pin_MAPbI3.yaml` | MAPbI₃ p-i-n | ✓ | Beer–Lambert | Inverted-architecture reference |
| `nip_MAPbI3_tmm.yaml` | MAPbI₃ n-i-p + glass | ✓ | TMM | TMM-enabled with 1 mm substrate |
| `pin_MAPbI3_tmm.yaml` | MAPbI₃ p-i-n + glass | ✓ | TMM | TMM-enabled |
| `ionmonger_benchmark.yaml` | Courtier 2019 parameters | ✓ | Beer–Lambert | IonMonger cross-check |
| `driftfusion_benchmark.yaml` | Driftfusion parameters | ✓ | Beer–Lambert | Driftfusion cross-check |
| `cigs_baseline.yaml` | ZnO / CdS / CIGS | ✗ | Beer–Lambert | $D_{\text{ion}}=0$ everywhere |
| `cSi_homojunction.yaml` | n⁺ / p c-Si | ✗ | Beer–Lambert | Homojunction wafer cell |

---

## 🧪 Testing

```bash
# From perovskite-sim/
pytest                                                  # unit + integration, ~15 s
pytest -m slow                                          # physics regression, ~30 s
pytest --cov=perovskite_sim --cov-report=term-missing   # with coverage report
pytest tests/unit/experiments/test_jv_sweep.py          # a single file
```

- **Unit tests** — per-module physics + solver coverage
- **Integration tests** — end-to-end experiment runs on shipped presets
- **Regression tests** — physical sanity envelopes ($V_{\text{oc}}$ range, $J_{\text{sc}}$ bounds, hysteresis index), BLAS threads pinned automatically via `tests/conftest.py`

---

## 📚 References

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

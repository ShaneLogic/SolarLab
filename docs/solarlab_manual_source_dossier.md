# SolarLab Manual Source Dossier

Status: collection pass before PDF generation.

This document gathers the information needed to write a SCAPS-style SolarLab
manual. It is not the final manual text and it is not a PDF draft. Its purpose
is to make the source material, framework, technical priorities, and remaining
gaps explicit before we generate the final document.

## 1. Manual Goal

The final PDF should be a beginner-readable but technically serious simulator
manual. It should explain both:

- how a user defines a device, runs a simulation, and reads results;
- what physical and numerical models the simulator solves underneath.

The SCAPS manual is useful as a structural reference because it combines user
workflow, device-definition panels, calculation settings, result analysis,
batch runs, scripting, and references. SolarLab should follow the same spirit
but not copy the SCAPS structure mechanically, because SolarLab is code/API/UI
driven rather than a legacy Windows-panel simulator.

## 2. PDF Production Requirements

User-stated requirements for the final PDF:

- Body font: Arial.
- Beginner-friendly explanation, not only formulas.
- Strong logic: concepts before equations, equations before implementation,
  implementation before user workflow details.
- Correct subscripts and superscripts, for example `V_{oc}`, `J_{sc}`,
  `E_g`, `m^{-3}`, `cm^2 V^{-1} s^{-1}`.
- Correct math formatting, preferably through LaTeX-style source math:
  `\frac{\partial n}{\partial t}`, `\nabla`, `\phi`, `\varepsilon_0`.
- Equation symbols must be defined close to first use.
- Units must be shown consistently in SI.
- Figures should use existing simulator images where possible:
  `perovskite-sim/docs/images/device_structure.png`,
  `band_diagram.png`, `transport_equations.png`,
  `solver_pipeline.png`, and `ui_layout.png`.

Recommended production stack for the final PDF:

- Source: Markdown or Typst with LaTeX math blocks.
- PDF engine: Typst, Pandoc + XeLaTeX, or LaTeX directly.
- Font handling: set body/sans font to Arial. If using XeLaTeX, use
  `fontspec` with `\setmainfont{Arial}` and `\setsansfont{Arial}`.
- Math handling: keep math in LaTeX notation. Use a compatible math font
  rather than forcing Arial inside equations, because Arial does not provide
  complete mathematical glyph coverage.
- Cross references: number equations, figures, tables, and sections.

## 3. SCAPS Manual Structure To Learn From

The SCAPS Manual February 2016 has 111 pages and follows this broad structure:

1. About SCAPS.
2. Getting started: run SCAPS, define problem, set working point, select
   measurements, calculate, display curves, edit problem, batch, recorder.
3. Solar-cell definition: layers, contacts, thickness, graded layers,
   defects/recombination, interfaces, tunneling, numerical tunnel settings,
   saving and loading definitions.
4. Working point: temperature, voltage, illumination, generation profile,
   initial state, series and shunt resistance.
5. Single-shot calculations: meshing, solution path, small-signal settings,
   numerical parameters, limitations.
6. Result analysis: bands, generation/recombination, I-V, ac, C-V, C-f, QE,
   measured files, saving curves.
7. Batch calculations.
8. Recorder calculations.
9. Curve fitting.
10. Scripting.
11. References.

What SolarLab should borrow:

- Start with workflow, then deepen into model details.
- Treat every input panel or YAML field as a scientific assumption.
- Pair result curves with interpretation notes.
- Include numerical warnings near the relevant workflow, not only in an
  appendix.
- Include file formats and automation after the core manual chapters.

What SolarLab should change:

- Explain Python API, YAML presets, FastAPI/SSE jobs, and frontend workstation
  instead of SCAPS GUI panels.
- Put the physical model first enough that beginners understand why each input
  field exists.
- Separate 1D core physics from the experimental 2D extension and screening
  bridge layers.

## 4. Technical Details That Need Most Attention

These are the details that should receive the most careful treatment in the
final manual.

1. State variables and sign conventions.
   The 1D core state is `y = (n, p, P)` at every electrical grid node, with an
   optional negative ion species. Current sign conventions differ internally and
   in solar-cell reporting, so the manual must define the reported sign for
   `J_sc`, dark injection current, and current decomposition.

2. Boundary and initial conditions.
   Beginners commonly misunderstand contacts. The manual must distinguish
   Poisson Dirichlet boundary conditions, ohmic carrier pins, optional Robin
   selective contacts, and zero-flux ion boundaries.

3. Built-in potential.
   `DeviceStack.V_bi` is the Poisson boundary-condition value from YAML, while
   `stack.compute_V_bi()` derives an effective built-in potential from
   Fermi-level differences across the electrical stack. The final manual should
   explain why both exist.

4. Multilayer and heterojunction physics.
   SolarLab uses electron affinity `\chi`, bandgap `E_g`, band offsets, and
   thermionic-emission caps at sharp interfaces. This is central for explaining
   the IonMonger benchmark behavior.

5. Optical generation.
   The simulator supports scalar Beer-Lambert generation and wavelength-
   resolved transfer-matrix optics. EQE and EL require TMM optical materials.

6. Recombination channels.
   SRH, radiative, Auger, interface SRH, trap profiles, photon recycling, and
   radiative reabsorption must be explained separately.

7. Mobile ions.
   Ion migration uses Scharfetter-Gummel style drift-diffusion with steric
   saturation and zero-flux boundaries. The final manual should make clear why
   ions cause hysteresis and why non-perovskite stacks usually set ion terms to
   zero.

8. Numerical method and stability.
   Scharfetter-Gummel fluxes, method of lines, Radau/BDF integration,
   prefactored tridiagonal Poisson solves, `max_step` caps, and finite-value
   guards are not optional implementation trivia; they are part of the model's
   reliability.

9. Experiment wrappers.
   Each experiment has a different physical measurement meaning and different
   assumptions. The manual should not present all jobs as generic "runs".

10. Validation envelope and limitations.
    The manual should state what is validated, what is benchmarked, and where
    the code is intentionally approximate.

## 5. Repository Sources Collected

Primary tree:

- `perovskite-sim/AGENTS.md`: authoritative architecture and gotchas.
- `README.md`: top-level public overview and physics summary.
- `perovskite-sim/README.md`: subtree orientation, equations, examples.
- `perovskite-sim/backend/README.md`: API endpoints and job streaming.
- `perovskite-sim/docs/benchmark_analysis.md`: validation notes and old
  benchmark diagnosis.
- `perovskite-sim/docs/screening/solarscale_import.md`: SolarScale import and
  screening workflow.
- `perovskite-sim/configs/*.yaml`: shipped 1D device presets.
- `perovskite-sim/configs/twod/*.yaml`: shipped 2D presets.
- `perovskite-sim/perovskite_sim/models/*.py`: dataclasses and result schemas.
- `perovskite-sim/perovskite_sim/physics/*.py`: physical models.
- `perovskite-sim/perovskite_sim/solver/*.py`: numerical solver.
- `perovskite-sim/perovskite_sim/experiments/*.py`: experiment wrappers.
- `perovskite-sim/perovskite_sim/twod/*.py`: 2D extension.
- `perovskite-sim/backend/main.py`: HTTP API dispatch and job parameters.
- `perovskite-sim/frontend/src/types.ts`: frontend result payloads.

Important diagrams:

- `perovskite-sim/docs/images/device_structure.png`
- `perovskite-sim/docs/images/band_diagram.png`
- `perovskite-sim/docs/images/transport_equations.png`
- `perovskite-sim/docs/images/solver_pipeline.png`
- `perovskite-sim/docs/images/ui_layout.png`

## 6. Core Device Model

### 6.1 Layer And Device Objects

`MaterialParams` stores per-layer material parameters. Main fields:

| Field | Meaning | Unit |
|---|---|---|
| `eps_r` | relative permittivity | dimensionless |
| `mu_n`, `mu_p` | electron and hole mobility | `m^2 V^{-1} s^{-1}` |
| `D_ion` | positive ion diffusion coefficient | `m^2 s^{-1}` |
| `P_lim` | positive ion steric density limit | `m^{-3}` |
| `P0` | initial positive ion density | `m^{-3}` |
| `ni` | intrinsic carrier density | `m^{-3}` |
| `tau_n`, `tau_p` | SRH lifetimes | `s` |
| `n1`, `p1` | SRH trap reference densities | `m^{-3}` |
| `B_rad` | radiative recombination coefficient | `m^3 s^{-1}` |
| `C_n`, `C_p` | Auger coefficients | `m^6 s^{-1}` |
| `alpha` | Beer-Lambert absorption coefficient | `m^{-1}` |
| `N_A`, `N_D` | acceptor and donor doping | `m^{-3}` |
| `chi` | electron affinity | `eV` |
| `Eg` | band gap | `eV` |
| `A_star_n`, `A_star_p` | Richardson constants | `A m^{-2} K^{-2}` |
| `D_ion_neg`, `P0_neg`, `P_lim_neg` | negative ion species parameters | SI |
| `Nc300`, `Nv300` | effective density of states at 300 K | `m^{-3}` |
| `mu_T_gamma` | mobility temperature exponent | dimensionless |
| `E_a_ion` | ion activation energy | `eV` |
| `B_rad_T_gamma` | radiative coefficient temperature exponent | dimensionless |
| `varshni_alpha`, `varshni_beta` | Varshni bandgap parameters | `eV K^{-1}`, `K` |
| `trap_N_t_interface`, `trap_N_t_bulk` | trap densities | `m^{-3}` |
| `trap_decay_length` | trap profile length scale | `m` |
| `trap_profile_shape` | `exponential` or `gaussian` | string |
| `optical_material` | key into `data/nk/*.csv` | string |
| `n_optical` | constant optical refractive index fallback | dimensionless |
| `incoherent` | TMM incoherent layer flag | boolean |
| `v_sat_n`, `v_sat_p` | saturation velocities | `m s^{-1}` |
| `ct_beta_n`, `ct_beta_p` | Caughey-Thomas exponents | dimensionless |
| `pf_gamma_n`, `pf_gamma_p` | Poole-Frenkel prefactors | `(V m^{-1})^{-1/2}` |

`LayerSpec` stores `name`, `thickness`, `params`, and `role`.

`DeviceStack` stores the layer tuple and global settings:

- `phi_left`
- `V_bi`
- `Phi`
- `interfaces`
- `T`
- `mode`
- `S_n_left`, `S_p_left`, `S_n_right`, `S_p_right`
- `microstructure`

Layer roles in the frontend schema:

- `substrate`
- `front_contact`
- `ETL`
- `absorber`
- `HTL`
- `back_contact`

Substrate layers are optical-only and must form a contiguous prefix. Electrical
drift-diffusion uses `electrical_layers(stack)`, which excludes substrate
layers.

### 6.2 YAML Device Schema

YAML files define:

- a top-level `device` block;
- a `layers` list;
- optional `interfaces`;
- optional contact surface velocities;
- optional `microstructure` for 2D.

The loader casts numeric strings to floats because YAML 1.1 can parse bare
scientific notation such as `1e-9` as a string.

Contact velocities can be flat fields or nested:

```yaml
device:
  mode: full
  contacts:
    left:
      S_n: 1.0e-3
      S_p: 1.0e3
    right:
      S_n: 1.0e3
      S_p: 1.0e-3
```

Missing or `null` contact values keep the default ohmic boundary. `S = 0`
means blocking. Large `S` approaches ohmic behavior.

## 7. Governing Equations

### 7.1 Poisson Equation

SolarLab solves the electrostatic potential from:

```math
\frac{\partial}{\partial x}
\left(
\varepsilon_0 \varepsilon_r \frac{\partial \phi}{\partial x}
\right)
= -\rho .
```

The finite-volume discretization uses harmonic face permittivity:

```math
\varepsilon_{i+1/2}
=
\frac{2 \varepsilon_i \varepsilon_{i+1}}
{\varepsilon_i + \varepsilon_{i+1}} .
```

Boundary conditions:

```math
\phi(0) = \phi_\mathrm{left}, \qquad
\phi(L) = \phi_\mathrm{left} + V_\mathrm{bi} - V_\mathrm{app}.
```

Implementation source: `perovskite_sim/physics/poisson.py`.

### 7.2 Electron And Hole Continuity

Electron density:

```math
\frac{\partial n}{\partial t}
=
\frac{1}{q}\frac{\partial J_n}{\partial x}
+ G - R .
```

Hole density:

```math
\frac{\partial p}{\partial t}
=
-\frac{1}{q}\frac{\partial J_p}{\partial x}
+ G - R .
```

`G` is optical generation and `R` is total recombination. In heterostacks, the
flux calculation uses band-corrected potentials:

```math
\phi_n = \phi + \chi ,
\qquad
\phi_p = \phi + \chi + E_g .
```

Implementation source: `perovskite_sim/physics/continuity.py`.

### 7.3 Scharfetter-Gummel Fluxes

The Bernoulli function is:

```math
B(\xi) = \frac{\xi}{\exp(\xi)-1}.
```

For a face between nodes `i` and `i+1`, the code uses stable numerical
branches for small and large `\xi`.

Electron face flux:

```math
J_{n,i+1/2}
=
\frac{q D_n}{\Delta x}
\left[
B(\xi)n_{i+1} - B(-\xi)n_i
\right].
```

Hole face flux:

```math
J_{p,i+1/2}
=
\frac{q D_p}{\Delta x}
\left[
B(\xi)p_i - B(-\xi)p_{i+1}
\right].
```

Implementation source: `perovskite_sim/discretization/fe_operators.py`.

### 7.4 Mobile Ion Continuity

Positive vacancies use:

```math
\frac{\partial P}{\partial t}
=
-\frac{\partial F_P}{\partial x}.
```

The flux has Scharfetter-Gummel drift-diffusion form with steric saturation:

```math
s(P)
=
\frac{1}
{\max(1 - P_\mathrm{avg}/P_\mathrm{lim}, 10^{-6})}.
```

Ion boundaries are zero-flux:

```math
F_P(0) = F_P(L) = 0 .
```

Negative ions are optional and reverse the drift sign.

Implementation source: `perovskite_sim/physics/ion_migration.py`.

### 7.5 Recombination

SRH recombination:

```math
R_\mathrm{SRH}
=
\frac{np - n_i^2}
{\tau_p(n+n_1)+\tau_n(p+p_1)} .
```

Radiative recombination:

```math
R_\mathrm{rad} = B_\mathrm{rad}(np-n_i^2).
```

Auger recombination:

```math
R_\mathrm{Auger}
=
(C_n n + C_p p)(np-n_i^2).
```

Total recombination:

```math
R = R_\mathrm{SRH} + R_\mathrm{rad} + R_\mathrm{Auger}.
```

Interface SRH is represented through surface recombination velocities and then
converted to a local volumetric contribution.

Implementation source: `perovskite_sim/physics/recombination.py`.

## 8. Boundary And Initial Conditions

Poisson:

- left potential fixed at `phi_left`;
- right potential fixed at `phi_left + V_bi - V_app`.

Carriers:

- default contacts are ohmic Dirichlet pins;
- boundary carrier densities are computed from mass action and charge
  neutrality;
- FULL mode can replace selected sides/carriers with Robin selective-contact
  fluxes:

```math
J = \pm q S(u-u_\mathrm{eq}).
```

Ions:

- positive and negative ion species use zero-flux boundaries.

Dark equilibrium:

```math
np = n_i^2, \qquad n-p = N_D - N_A .
```

Illuminated steady state:

- start from dark equilibrium;
- integrate under illumination at the starting voltage;
- use this state for experiments that need light-soaked initial conditions.

Implementation sources:

- `perovskite_sim/solver/newton.py`
- `perovskite_sim/solver/illuminated_ss.py`
- `perovskite_sim/physics/contacts.py`

## 9. Physics Tiers

`DeviceStack.mode` controls the maximum physics fidelity:

| Mode | Meaning | Typical use |
|---|---|---|
| `legacy` | Disable upgraded physics for old benchmark reproduction | historical parity |
| `fast` | Enable build-once upgrades, skip per-RHS expensive hooks | screening and routine sweeps |
| `full` | Enable every configured hook | highest-fidelity single runs |

The mode is a ceiling. A feature activates only when both the mode and the
configuration support it.

Examples:

- TMM needs `optical_material`.
- Field-dependent mobility needs `v_sat_*` or `pf_gamma_*`.
- Selective contacts need finite `S_*` values.
- Radiative reabsorption is FULL-only.

Implementation source: `perovskite_sim/models/mode.py`.

## 10. Optical Generation

### 10.1 Beer-Lambert

Beer-Lambert generation is:

```math
G(x) = \Phi \alpha(x)
\exp\left[-\int_0^x \alpha(x')\,dx'\right].
```

Implementation source: `perovskite_sim/physics/generation.py`.

### 10.2 Transfer-Matrix Method

TMM uses normal-incidence coherent thin-film optics. `optical_material` keys
load `n,k` data from `perovskite_sim/data/nk/*.csv`.

CSV format:

```csv
wavelength_nm,n,k
300,...
```

Important rules:

- `load_nk` interpolates only inside the native wavelength range.
- It should not silently clamp or extrapolate.
- Thick substrates can be marked `incoherent: true`.
- Substrate layers participate optically but not electrically.
- TMM returns position-resolved absorption and generation on the electrical
  grid after accounting for substrate offset.

Implementation sources:

- `perovskite_sim/physics/optics.py`
- `perovskite_sim/data/__init__.py`
- `perovskite_sim/solver/mol.py`

## 11. Other Physics Hooks

Thermionic emission:

- enabled at internal interfaces when `|\Delta E_c|` or `|\Delta E_v|` exceeds
  `0.05 eV`;
- caps SG flux by a Richardson-Dushman type limit;
- prevents over-current across sharp single-grid-spacing band discontinuities.

Photon recycling:

```math
P_\mathrm{esc}
=
\min\left(1,\frac{1}{4 n^2 \alpha d}\right).
```

The effective radiative coefficient may be scaled by escape probability, and
FULL mode can add radiative reabsorption back into `G(x)`.

Field-dependent mobility:

```math
\mu_\mathrm{CT}
=
\frac{\mu_0}
{\left[1+\left(\frac{\mu_0 |E|}{v_\mathrm{sat}}\right)^\beta\right]^{1/\beta}}.
```

```math
\mu_\mathrm{PF} = \mu_0 \exp(\gamma \sqrt{|E|}).
```

Temperature:

- `V_T = kT/q`;
- density of states scales as `(T/300)^{3/2}`;
- ion diffusion follows Arrhenius scaling;
- bandgap can shift by Varshni parameters;
- radiative coefficient can scale as `(T/300)^\gamma`.

Trap profiles:

- exponential edge profile;
- Gaussian edge profile;
- local lifetime is reduced according to local trap density.

Implementation sources:

- `perovskite_sim/physics/continuity.py`
- `perovskite_sim/physics/photon_recycling.py`
- `perovskite_sim/physics/field_mobility.py`
- `perovskite_sim/physics/temperature.py`
- `perovskite_sim/physics/traps.py`

## 12. Numerical Architecture

Core flow:

```text
YAML or inline device dict
-> MaterialParams + LayerSpec + DeviceStack
-> experiment driver
-> solver/mol.py MaterialArrays cache
-> result dataclass
```

Spatial discretization:

- multilayer grid over electrical layers;
- Scharfetter-Gummel finite-element/finite-volume fluxes;
- harmonic face permittivity for Poisson.

Time integration:

- method of lines;
- `scipy.integrate.solve_ivp`;
- Radau as primary implicit solver;
- bounded BDF fallback in difficult J-V steps.

Performance-critical cache:

- `build_material_arrays(x, stack)` builds per-node and per-face arrays once;
- it also builds interface masks, contact densities, optical generation, and
  a prefactored Poisson operator;
- it must not be called inside the RHS or per-time-step inner loop.

Poisson fast path:

- `factor_poisson`;
- `solve_poisson_prefactored`;
- LAPACK tridiagonal factor/solve.

Numerical safety notes:

- RHS has finite-value guards.
- J-V, impedance, and degradation cap `max_step` around voltage steps.
- Steady-state terminal current should use the median-current helper, not a
  single boundary face.
- EQE and Suns-Voc use longer settles and dark-current subtraction to suppress
  residual ionic transients.
- TMM has a determinant guard for transfer-matrix inversion.

## 13. Experiment Workflows

### 13.1 J-V Sweep

Public driver: `run_jv_sweep`.

Outputs:

- `V_fwd`, `J_fwd`
- `V_rev`, `J_rev`
- `metrics_fwd`, `metrics_rev`
- `hysteresis_index`
- optional current decomposition
- optional spatial snapshots

Metrics:

- `J_sc` interpolated at `V=0`.
- `V_oc` interpolated from the zero-current crossing.
- `FF = P_mpp/(V_oc J_sc)`.
- `PCE = P_mpp / 1000` for standard incident power density.
- `voc_bracketed` is false if the sweep did not cross zero current.

Default upper voltage:

```math
V_\mathrm{upper}
=
\max(1.3 V_{\mathrm{bi,eff}}, 1.4\,\mathrm{V}).
```

### 13.2 Dark J-V

Public driver: `run_dark_jv`.

Purpose:

- dark diode curve;
- ideality factor;
- saturation current density `J_0`.

Fit model:

```math
J = J_0\left[\exp\left(\frac{V}{nV_T}\right)-1\right].
```

The code fits a linear region in `log|J|` vs `V`.

### 13.3 Suns-Voc

Public driver: `run_suns_voc`.

Purpose:

- intensity-dependent `V_oc`;
- pseudo-JV curve;
- pseudo-fill factor.

Pseudo-JV convention:

```math
J_\mathrm{pseudo}(X)
=
J_{\mathrm{sc,ref}} - J_\mathrm{sc}(X),
\qquad
V_\mathrm{pseudo}(X) = V_\mathrm{oc}(X).
```

### 13.4 V<sub>oc</sub>(T)

Public driver: `run_voc_t`.

Purpose:

- temperature-dependent open-circuit voltage;
- activation-energy estimate from the intercept at `T=0`.

Approximate relationship:

```math
V_\mathrm{oc}(T)
\approx
\frac{E_A}{q}
-
\frac{kT}{q}\ln\left(\frac{J_{00}}{J_\mathrm{sc}}\right).
```

The driver rebuilds an immutable `DeviceStack` at each temperature.

### 13.5 EQE / IPCE

Public driver: `compute_eqe`.

Purpose:

- wavelength-resolved external quantum efficiency;
- AM1.5G-integrated `J_sc`.

Equation:

```math
\mathrm{EQE}(\lambda)
=
\frac{|J_\mathrm{sc}(\lambda)|}
{q\Phi_\mathrm{inc}(\lambda)}.
```

Requires TMM optical data. Beer-Lambert-only stacks cannot produce
wavelength-resolved EQE.

### 13.6 Electroluminescence

Public driver: `run_el_spectrum`.

Purpose:

- reciprocity EL spectrum;
- `EQE_EL`;
- non-radiative voltage loss `\Delta V_{nr}`.

Rau reciprocity form:

```math
\Phi_\mathrm{EL}(\lambda)
=
A_\mathrm{abs}(\lambda)\,
\phi_\mathrm{bb}(\lambda,T)\,
\exp\left(\frac{qV_\mathrm{inj}}{kT}\right).
```

Blackbody photon flux:

```math
\phi_\mathrm{bb}(\lambda,T)
=
\frac{2\pi c/\lambda^4}
{\exp(hc/\lambda kT)-1}.
```

Radiative emission current:

```math
J_\mathrm{em,rad}
=
q\int \Phi_\mathrm{EL}(\lambda)\,d\lambda.
```

External EL efficiency:

```math
\mathrm{EQE}_\mathrm{EL}
=
\frac{J_\mathrm{em,rad}}{|J_\mathrm{inj}|}.
```

Non-radiative voltage loss:

```math
\Delta V_\mathrm{nr}
=
-V_T\ln(\mathrm{EQE}_\mathrm{EL}).
```

Requires TMM optical data.

### 13.7 Impedance

Public driver: `run_impedance`.

Purpose:

- small-signal impedance spectrum;
- Nyquist and Bode plots.

The code integrates a sinusoidal voltage perturbation around `V_dc` and uses
lock-in extraction to compute `Z(f)`. It includes displacement current and
reports area-normalized impedance in `\Omega m^2`.

### 13.8 Mott-Schottky

Public driver: `run_mott_schottky`.

Purpose:

- dark capacitance-voltage curve;
- `1/C^2` fit;
- built-in voltage and effective doping estimate.

One-sided depletion relationship:

```math
C(V)
=
\sqrt{
\frac{q\varepsilon_s\varepsilon_0 N_\mathrm{eff}}
{2(V_\mathrm{bi}-V)}
}.
```

Linearized:

```math
\frac{1}{C^2} = aV + b,
\qquad
V_\mathrm{bi} = -\frac{b}{a},
\qquad
N_\mathrm{eff} =
-\frac{2}{q\varepsilon_s\varepsilon_0 a}.
```

The manual must warn that fully depleted thin absorbers can produce flat
`1/C^2` curves and nonsensical fits.

### 13.9 Transient Photovoltage

Public driver: `run_tpv`.

Purpose:

- light-pulse perturbation at open circuit;
- voltage transient;
- recombination lifetime from decay.

Workflow:

1. Solve illuminated steady state.
2. Find `V_oc`.
3. Apply a small generation pulse.
4. At every reported time, adjust `V_app` by Newton iteration to enforce
   terminal `J = 0`.
5. Fit:

```math
V(t) \approx V_\mathrm{oc}
+ \Delta V_0 \exp(-t/\tau).
```

### 13.10 Degradation

Public driver: `run_degradation`.

Purpose:

- slow ion-coupled damage proxy;
- time-dependent `PCE`, `V_oc`, `J_sc`;
- ion profiles.

Important manual point:

- snapshot J-V freezes ions by setting ion diffusion to zero while carriers
  relax, so slow ionic drift is separated from instantaneous electronic
  response.

### 13.11 Tandem J-V

Public driver: `run_tandem_jv`.

Purpose:

- 2T monolithic tandem simulation.

Workflow:

1. Combined TMM over top cell, junction, and bottom cell.
2. Independent sub-cell J-V sweeps with fixed generation.
3. Series matching at common current.
4. Sum sub-cell voltages plus junction voltage.

### 13.12 2D J-V

Public driver: `run_jv_sweep_2d`.

Purpose:

- tensor-product 2D extension;
- lateral-uniform parity with 1D;
- microstructure and grain-boundary studies.

Important limitations:

- ions are frozen as static Poisson background during 2D J-V;
- lateral boundary can be periodic or Neumann;
- 2D is an extension, not the default production path for all experiments.

### 13.13 V<sub>oc</sub>(L_g) Grain Sweep

Public driver: `run_voc_grain_sweep`.

Purpose:

- repeat 2D J-V over grain sizes;
- report `V_oc`, `J_sc`, and `FF` vs grain size.

Inputs:

- `grain_sizes_nm`;
- `tau_gb_n`, `tau_gb_p`;
- `gb_width`;
- 2D mesh controls.

## 14. Backend And Frontend User Surfaces

### 14.1 Configuration Endpoints

Backend:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/configs` | list shipped and user presets |
| `GET` | `/api/configs/{name}` | load one preset |
| `POST` | `/api/configs/user` | save edited user stack |
| `GET` | `/api/layer-templates` | layer template library |
| `GET` | `/api/optical-materials` | available TMM optical keys |

### 14.2 Streaming Jobs

Preferred experiment API:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | submit `{kind, config_path|device, params}` |
| `GET` | `/api/jobs/{id}/events` | Server-Sent Events stream |

SSE event names:

- `progress`
- `result`
- `error`
- `done`

Progress payload:

```json
{
  "stage": "jv",
  "current": 1,
  "total": 30,
  "eta_s": 12.3,
  "message": "V=0.400 V"
}
```

Supported job kinds:

- `jv`
- `current_decomp`
- `spatial`
- `dark_jv`
- `suns_voc`
- `voc_t`
- `eqe`
- `el`
- `impedance`
- `tpv`
- `degradation`
- `tandem`
- `mott_schottky`
- `jv_2d`
- `voc_grain_sweep`

Legacy blocking endpoints:

- `POST /api/jv`
- `POST /api/impedance`
- `POST /api/degradation`

### 14.3 Frontend Workstation Experiments

The frontend exposes grouped experiments:

- J-V Sweep
- Suns-Voc
- V<sub>oc</sub>(T) activation energy
- Dark J-V
- Mott-Schottky
- EQE / IPCE
- Electroluminescence
- Impedance
- Transient Photovoltage
- Degradation
- J-V Sweep (2D)
- V<sub>oc</sub>(L_g) Grain Sweep

Result payloads are defined in `frontend/src/types.ts`.

## 15. Shipped Presets

Top-level presets:

- `cSi_homojunction.yaml`
- `cigs_baseline.yaml`
- `driftfusion_benchmark.yaml`
- `driftfusion_benchmark_tmm.yaml`
- `field_mobility_demo.yaml`
- `ionmonger_benchmark.yaml`
- `ionmonger_benchmark_tmm.yaml`
- `nip_MAPbI3.yaml`
- `nip_MAPbI3_tmm.yaml`
- `nip_SnPb_1p22.yaml`
- `nip_wideGap_FACs_1p77.yaml`
- `pin_MAPbI3.yaml`
- `pin_MAPbI3_tmm.yaml`
- `radiative_limit.yaml`
- `selective_contacts_demo.yaml`
- `solarscale_nip_band_aligned.yaml`
- `tandem_lin2019.yaml`

2D presets:

- `twod/bcx_combined_demo.yaml`
- `twod/nip_MAPbI3_singleGB.yaml`
- `twod/nip_MAPbI3_uniform.yaml`

The manual should include a preset table with at least:

- structure;
- material family;
- whether ions are active;
- optics mode;
- intended experiment;
- fidelity tier compatibility;
- limitations.

## 16. Validation Evidence Collected

Existing validation and regression evidence includes:

- Beer-Lambert generation error around `0.19%`.
- Scharfetter-Gummel `J_sc` vs theory error around `0.92%`.
- SRH lifetime trend and ideality behavior.
- `J_sc` linearity with photon flux.
- Ion conservation error around `0.03%` in older benchmark notes.
- Dark equilibrium charge neutrality tests.
- Finite RHS guard tests.
- IonMonger benchmark target envelope:
  - `V_oc` around `1.19 V` in current band-offset-aware model;
  - `J_sc` around `200-260 A/m^2`;
  - FF in physically expected envelope.
- Integration pinned reference:
  - IonMonger reference `V_oc = 1.1932 V`;
  - `J_sc = 231.70 A/m^2`;
  - `FF = 0.7774`;
  - `PCE = 0.2149`.
- TMM regression:
  - `nip_MAPbI3_tmm` baseline `J_sc = 211.02 A/m^2`;
  - `pin_MAPbI3_tmm` baseline `J_sc = 216.62 A/m^2`;
  - tolerance `+-5 A/m^2`.
- Physical trend validation:
  - `V_oc` below `E_g`;
  - `J_sc` decreases as bandgap widens;
  - `V_oc` vs absorber thickness slope in a plausible mV/decade range;
  - high mobility improves FF;
  - dark ideality factor between about `1.0` and `2.5`;
  - Suns-Voc slope around `20-70 mV/decade`.
- Photon recycling regression:
  - radiative-limit `V_oc` boost expected `40-100 mV`.
- 2D validation:
  - lateral-uniform 2D reproduces 1D `V_oc` within about `0.1 mV`;
  - `J_sc` relative error below about `5e-4`;
  - FF difference below about `1e-3`;
  - Robin contacts, field mobility, and radiative reabsorption have targeted
    tests.

The final manual should separate:

- validated numerical behavior;
- benchmark agreement;
- physically plausible trend tests;
- approximate research extensions.

## 17. Known Limitations To State

Important limitations and warning text for the manual:

- TMM-based EQE and EL require wavelength-resolved `optical_material`; scalar
  Beer-Lambert stacks cannot produce wavelength-resolved EQE/EL.
- `load_nk` does not extrapolate outside the optical-data wavelength range.
- `V_oc`, FF, and PCE are sentinel zero values when `voc_bracketed=false`.
  Users must increase `V_max`.
- C-V/Mott-Schottky fits can fail or become meaningless for fully depleted thin
  absorbers.
- Thick CIGS or c-Si absorbers can be structurally valid but slow at full
  transient settings.
- 2D simulations freeze mobile ions and are not a full dynamic 2D ion model.
- Tandem preset currently has warning around stub/placeholder n,k data for
  some materials unless replaced with real source data.
- Screening/smoke results are process validation unless run with production
  settings and validated material provenance.
- `legacy`, `fast`, and `full` modes change active physics; users must report
  the mode in any study.

## 18. SolarScale Screening Bridge

SolarLab can consume SolarScale material records through a readiness-gated
importer.

Important manual separation:

- SolarLab core simulator: solves device physics.
- SolarScale bridge: converts upstream material evidence into simulator-ready
  inputs and sweep plans.

Mapped fields when provenance supports them:

- `dielectric_static_avg -> eps_r`
- electron mobility in `cm^2 V^{-1} s^{-1} -> mu_n` in `m^2 V^{-1} s^{-1}`
- hole mobility in `cm^2 V^{-1} s^{-1} -> mu_p`
- ion diffusion -> `D_ion`
- ion activation energy -> `E_a_ion`

By default, `band_gap_hse_ev` remains metadata unless an explicitly band-aligned
template is used.

Device-only unknowns should become sweep dimensions:

- absorber thickness;
- SRH lifetime;
- trap density;
- surface/interface recombination velocity;
- contact work function/band alignment.

## 19. Proposed Final PDF Framework

Recommended final manual chapters:

1. Title, scope, version, and citation note.
2. What SolarLab simulates.
3. Beginner physics primer:
   - solar-cell stack;
   - electrons, holes, ions;
   - bands and quasi-Fermi levels;
   - illumination and recombination.
4. Installation and quick start.
5. Device definition:
   - YAML structure;
   - layers and roles;
   - material parameter table;
   - contacts and interfaces;
   - optical materials;
   - modes.
6. Governing equations:
   - Poisson;
   - carrier continuity;
   - ion continuity;
   - recombination;
   - optical generation.
7. Numerical method:
   - grid;
   - Scharfetter-Gummel;
   - method of lines;
   - Poisson factorization;
   - solver tolerances;
   - failure modes.
8. Running simulations:
   - Python;
   - backend;
   - web UI;
   - job streaming;
   - user presets.
9. Experiment manual:
   - J-V;
   - current decomposition;
   - spatial profiles;
   - dark J-V;
   - Suns-Voc;
   - V<sub>oc</sub>(T);
   - EQE;
   - EL;
   - impedance;
   - Mott-Schottky;
   - TPV;
   - degradation;
   - tandem;
   - 2D J-V;
   - grain sweep.
10. Interpreting outputs.
11. Preset library.
12. Validation and benchmark envelope.
13. Limitations and best practices.
14. Automation, scripting, and batch workflows.
15. SolarScale screening bridge.
16. Glossary and symbol table.
17. References.
18. Appendices:
   - full parameter tables;
   - API schema;
   - YAML examples;
   - equation derivations;
   - troubleshooting.

## 20. Remaining Gaps Before PDF Drafting

The code and docs contain enough information to draft a strong first version of
the manual. Before generating the final PDF, the following decisions would make
the PDF better:

1. Manual title and branding.
   Options: `SolarLab User Manual`, `SolarLab Technical Manual`, or
   `SolarLab: Thin-Film Solar Cell Simulator Manual`.

2. Target PDF length.
   A SCAPS-like detailed manual will likely be 80-140 pages. A compact manual
   could be 35-60 pages but would not satisfy the "extremely detailed for
   beginners" requirement as well.

3. Citation style.
   Need choose IEEE, ACS, APA, or simple numbered references.

4. Figure policy.
   Existing figures can be used now, but final PDF quality may improve if we
   regenerate clean vector-style diagrams for equations, stack geometry, and
   solver flow.

5. Parameter provenance.
   Preset tables can be generated from YAML now, but source/provenance notes
   for every material parameter may need manual research if the PDF is meant
   for publication rather than internal use.

6. Validation freshness.
   The dossier collected validation targets from docs/tests. Before publishing
   the PDF, rerun the current test suite and record exact current outputs.

7. References.
   Code mentions Courtier/IonMonger, Calado/Driftfusion, Rau reciprocity, and
   transfer-matrix optics, but a publication-quality bibliography should be
   verified and formatted.

8. Screenshots.
   A user manual should include current UI screenshots. These should be
   captured after the manual framework is approved and the app is running.

## 21. Readiness Assessment

Ready now:

- Core simulator architecture.
- Main equations and physical models.
- Device and material parameter schema.
- YAML/user workflow.
- Backend and frontend experiment surfaces.
- Experiment definitions and outputs.
- Major limitations.
- Existing validation envelope.
- Proposed PDF chapter framework.

Not ready without one more pass:

- final bibliography;
- current UI screenshots;
- exact fresh validation output from a test run;
- publication-grade provenance for every preset parameter.

Practical conclusion:

We can start writing the manual draft from this dossier. If the goal is an
internal SCAPS-like technical/user manual, this is enough to proceed. If the
goal is a publication-grade manual, collect bibliography/provenance and fresh
validation outputs before final PDF export.

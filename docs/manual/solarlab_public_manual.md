---
title: "SolarLab Technical and User Manual"
subtitle: "Technical guide to thin-film solar-cell simulation, workflows, and validation"
author: "SolarLab Project"
date: "2026-05-19"
lang: en-US
documentclass: report
fontsize: 11pt
geometry: margin=0.85in
mainfont: Arial
sansfont: Arial
monofont: Menlo
colorlinks: true
linkcolor: blue
urlcolor: blue
linktoc: all
toc: true
lof: true
lot: true
numbersections: true
header-includes:
  - \hypersetup{linktoc=all}
  - |
    \makeatletter
    \renewcommand*\l@section{\@dottedtocline{1}{1.5em}{3.2em}}
    \renewcommand*\l@subsection{\@dottedtocline{2}{4.7em}{4.1em}}
    \makeatother
---

\newpage

# Executive Summary

SolarLab is a thin-film photovoltaic device simulator that couples
drift-diffusion transport, Poisson electrostatics, mobile-ion redistribution,
recombination, and optical generation within a reproducible software workflow.
The numerical core is exposed through a Python API, a FastAPI service layer,
and a TypeScript/Plotly workstation. An experimental two-dimensional extension
supports lateral microstructure and grain-boundary studies.

This manual serves two purposes:

1. It introduces the device variables, parameter conventions, and simulation
   workflow for readers who are new to SolarLab or to semiconductor device
   simulation.
2. It documents the implemented physics, numerical method, validation gates,
   assumptions, and limitations needed for technical review and reproducible
   use.

The manual is based on the current SolarLab repository state:

- Commit: `43c81d7fefd009ffca598f58f85a40ad4e661e1e`
- Validation date: 2026-05-19
- Primary simulator tree: `perovskite-sim/`

The validation evidence collected before writing this PDF is summarized in
Chapter \ref{validation-and-evidence}. The short version is:

| Validation gate | Result |
|---|---:|
| `pytest` | 647 passed, 1 skipped |
| `pytest -m slow` | 72 passed |
| `pytest -m validation` | 18 passed |
| `npm run build` | passed |
| `npm run test:run` | 320 passed |

The validation evidence should be interpreted as implementation evidence, not
as a guarantee of predictive accuracy for arbitrary material stacks. It shows
that the equations, numerical coupling, backend/frontend interfaces,
regression envelopes, and documented benchmark cases are internally consistent
for the repository state listed above.

# How To Read This Manual

If you are new to device simulation, read Chapters 2 to 6 in order. These
chapters explain the device, the variables, the equations, and the numerical
method before introducing the software workflow.

If you are already familiar with semiconductor drift-diffusion, start with
Chapters 5, 7, 8, and 11:

- Chapter 5 defines the model equations.
- Chapter 7 explains the solver.
- Chapter 8 explains YAML/Python/API workflows.
- Chapter 11 gives validation evidence and limitations.

The notation in this manual follows standard solar-cell convention:

- $V_\mathrm{oc}$: open-circuit voltage.
- $J_\mathrm{sc}$: short-circuit current density.
- $FF$: fill factor.
- $PCE$: power-conversion efficiency.
- $n$: electron density.
- $p$: hole density.
- $P$: mobile positive ion-vacancy density.
- $\phi$: electrostatic potential.
- $E$: electric field.
- $G$: optical generation rate.
- $R$: recombination rate.

All simulator inputs are SI unless explicitly stated otherwise. Length is in
meters, time in seconds, current density in $A\,m^{-2}$, density in $m^{-3}$,
mobility in $m^2 V^{-1} s^{-1}$, and diffusion coefficient in $m^2 s^{-1}$.
The electron affinity $\chi$ and band gap $E_g$ are stored in eV.

# Simulator Architecture

SolarLab should be read as a coupled model implementation rather than as a
stand-alone user interface. YAML files, the Python API, the FastAPI backend,
and the TypeScript frontend all operate on the same device schema. A parameter
edited in the browser is therefore serialized into the same scientific object
used by the solver.

![SolarLab architecture flow](figures/architecture_flow.png)

The architecture preserves four technical boundaries:

- device definition is separated from experiment settings;
- material and grid arrays are cached before the time integrator runs;
- physics hooks are activated by the resolved simulation mode and by the
  presence of required parameters;
- backend and frontend layers transport result dataclasses instead of
  reimplementing solver logic.

For reproducible interpretation, each reported curve should identify the device
stack, physics tier, experiment driver, and solver configuration. Missing
metadata weakens reproducibility even when the numerical run completes
successfully.

# What SolarLab Simulates

SolarLab represents a solar cell as a sequence of material layers between two
contacts. The default coordinate follows the stack direction. Incident light
generates electron-hole pairs, while the electrostatic field and contact
selectivity drive carrier separation and extraction. In perovskite devices,
mobile ionic defects can redistribute under the same field, producing
history-dependent current-voltage response.

The default solver is one-dimensional and is appropriate for:

- J-V sweeps and hysteresis;
- dark diode curves;
- EQE and electroluminescence;
- impedance;
- transient photovoltage;
- ion-coupled degradation;
- tandem current matching;
- screening many material candidates.

SolarLab also includes an experimental two-dimensional solver. The 2D solver
extrudes the same vertical device stack laterally and is intended for:

- 1D/2D parity checks;
- vertical grain boundaries;
- lateral microstructure;
- $V_\mathrm{oc}(L_g)$ grain-size sweeps.

The 2D solver is not a replacement for the 1D workflow. Mobile ions are held as
a static Poisson background during 2D J-V runs; this is an explicit model
assumption and should be reported when interpreting 2D results.

![SolarLab device structure](perovskite-sim/docs/images/device_structure.png)

![SolarLab band-diagram concept](perovskite-sim/docs/images/band_diagram.png)

# Beginner Physics Primer

## Electrons, Holes, And Bands

In a semiconductor, electrons can move in the conduction band and holes can
move in the valence band. A solar cell works because light creates electron-hole
pairs and the device structure makes it more likely that electrons leave
through one contact while holes leave through the other.

SolarLab stores two energy parameters that are essential for heterojunctions:

- $\chi$, the electron affinity;
- $E_g$, the band gap.

Together these determine approximate conduction-band and valence-band
positions. At an interface, a change in $\chi$ or $E_g$ creates band
offsets. Band offsets are important because they can block one carrier type
while allowing the other carrier type to pass.

## Generation And Recombination

Optical generation $G(x)$ is the rate at which absorbed photons create
electron-hole pairs per unit volume. Recombination $R(n,p)$ is the rate at
which electrons and holes annihilate. A good solar cell has high generation and
carrier collection, but low recombination.

SolarLab includes several recombination channels:

- Shockley-Read-Hall (SRH) trap-assisted recombination;
- radiative recombination;
- Auger recombination;
- interface recombination through surface recombination velocities;
- optional trap-density profiles near interfaces.

## Mobile Ions And Hysteresis

Perovskite cells can contain mobile ionic defects. These defects move much more
slowly than electronic carriers. During a voltage scan, the ion profile may not
reach equilibrium at each voltage. This creates J-V hysteresis. SolarLab treats
this as a physical transient, not as a post-processing correction.

In the state vector, the default mobile ion variable is $P$, the positive
vacancy density. A second negative species can be configured through
`D_ion_neg`, `P0_neg`, and `P_lim_neg`.

# Device Definition

## Core Data Flow

The simulator data flow is:

```text
YAML or inline device dictionary
-> MaterialParams + LayerSpec + DeviceStack
-> experiment driver
-> MaterialArrays cache
-> result dataclass
```

The main dataclasses are:

- `MaterialParams`: physical parameters for one material.
- `LayerSpec`: layer name, role, thickness, and material parameters.
- `DeviceStack`: full multilayer stack and global device settings.

## Layer Roles

Layer roles used by the frontend and YAML schema are:

| Role | Meaning |
|---|---|
| `substrate` | optical-only layer, excluded from electrical drift-diffusion |
| `front_contact` | front electrode or transparent conductor |
| `ETL` | electron transport layer |
| `absorber` | active photovoltaic absorber |
| `HTL` | hole transport layer |
| `back_contact` | rear electrode |

Substrate layers must form a contiguous prefix of the stack. They participate
in transfer-matrix optics but do not appear in the drift-diffusion grid.

## Main Material Parameters

| Field | Meaning | Unit |
|---|---|---|
| `eps_r` | relative permittivity | dimensionless |
| `mu_n`, `mu_p` | electron and hole mobilities | $m^2 V^{-1} s^{-1}$ |
| `D_ion` | positive ion diffusion coefficient | $m^2 s^{-1}$ |
| `P_lim` | positive ion steric limit | $m^{-3}$ |
| `P0` | initial positive ion density | $m^{-3}$ |
| `ni` | intrinsic carrier density | $m^{-3}$ |
| `tau_n`, `tau_p` | SRH lifetimes | $s$ |
| `n1`, `p1` | SRH trap reference densities | $m^{-3}$ |
| `B_rad` | radiative recombination coefficient | $m^3 s^{-1}$ |
| `C_n`, `C_p` | Auger coefficients | $m^6 s^{-1}$ |
| `alpha` | Beer-Lambert absorption coefficient | $m^{-1}$ |
| `N_A`, `N_D` | acceptor and donor densities | $m^{-3}$ |
| `chi` | electron affinity | eV |
| `Eg` | band gap | eV |
| `optical_material` | key into `data/nk/*.csv` | string |
| `incoherent` | incoherent TMM layer flag | boolean |

Additional optional parameters cover negative ions, temperature scaling,
field-dependent mobility, trap profiles, and optical fallbacks.

## Parameter Dictionary

This section is intentionally detailed because input specification is often the
dominant source of error in device simulation. SolarLab cannot infer whether a
parameter comes from measurement, literature, fitting, or exploratory
screening. Each field should therefore be treated as a documented scientific
assumption.

### Electrical And Transport Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{eps_r} & 1 & relative dielectric constant in Poisson's equation & Larger values screen charge and reduce field gradients. Use layer-specific literature values rather than a single generic perovskite value. \\
\path{mu_n} & \(m^2 V^{-1} s^{-1}\) & low-field electron mobility & Controls electron extraction and transport resistance. Remember that \(1\,cm^2 V^{-1}s^{-1}=10^{-4}\,m^2 V^{-1}s^{-1}\). \\
\path{mu_p} & \(m^2 V^{-1} s^{-1}\) & low-field hole mobility & Same unit conversion convention as \path{mu_n}. Low mobility typically affects FF before producing a large change in \(V_\mathrm{oc}\). \\
\path{N_A} & \(m^{-3}\) & ionized acceptor density & Represents p-type doping. Values reported in \(cm^{-3}\) must be multiplied by \(10^6\). \\
\path{N_D} & \(m^{-3}\) & ionized donor density & Represents n-type doping. Use compensated doping only when it is part of the intended physical model. \\
\path{ni} & \(m^{-3}\) & intrinsic carrier density & Appears in mass action and SRH terms. It should be consistent with the effective band gap and temperature when doing parameter studies. \\
\path{chi} & eV & electron affinity & Sets approximate conduction-band alignment. Inconsistent \(\chi\) values can introduce artificial transport barriers. \\
\path{Eg} & eV & band gap & Sets band offsets and thermodynamic interpretation. Beer-Lambert optics do not automatically shift spectral absorption when \path{Eg} changes. \\
\bottomrule
\end{longtable}
\endgroup

### Ion And Hysteresis Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{D_ion} & \(m^2 s^{-1}\) & positive mobile-ion diffusion coefficient & Set to zero when mobile ions are excluded from the model. Nonzero values can produce scan-rate dependence. \\
\path{P0} & \(m^{-3}\) & initial positive ion density & Represents the equilibrium mobile-vacancy population before redistribution. \\
\path{P_lim} & \(m^{-3}\) & steric upper density for positive ions & Limits ion accumulation through a finite-site-density approximation. The value should be consistent with the assumed mobile-site density. \\
\path{D_ion_neg} & \(m^2 s^{-1}\) & negative mobile-ion diffusion coefficient & Enables a second mobile species. Leave zero for the default single-ion model. \\
\path{P0_neg} & \(m^{-3}\) & initial negative ion density & Use only when the negative species is part of the physical hypothesis. \\
\path{P_lim_neg} & \(m^{-3}\) & steric upper density for negative ions & Same interpretation as \path{P_lim}. \\
\path{E_a_ion} & eV & Arrhenius activation energy for ion diffusion & Used for temperature-dependent ion mobility. Fitted or literature-derived values should be reported with provenance. \\
\bottomrule
\end{longtable}
\endgroup

### Recombination And Trap Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{tau_n} & s & electron SRH lifetime & Shorter lifetime increases trap-assisted recombination. It is often an effective fitted parameter. \\
\path{tau_p} & s & hole SRH lifetime & Interpret with \path{tau_n}; asymmetry can model carrier-selective trap response. \\
\path{n1} & \(m^{-3}\) & SRH electron reference density & Encodes the effective trap-energy position and should be changed consistently with the trap model. \\
\path{p1} & \(m^{-3}\) & SRH hole reference density & Companion to \path{n1}. \\
\path{B_rad} & \(m^3 s^{-1}\) & radiative recombination coefficient & Central to radiative-limit and photon-recycling studies. \\
\path{C_n} & \(m^6 s^{-1}\) & electron Auger coefficient & Most relevant when high carrier densities are expected. \\
\path{C_p} & \(m^6 s^{-1}\) & hole Auger coefficient & Same caution as \path{C_n}. \\
\path{trap_N_t_interface} & \(m^{-3}\) & interface-near trap density & Activates spatial trap profiles when supplied with decay information. \\
\path{trap_N_t_bulk} & \(m^{-3}\) & bulk trap density & The asymptotic trap density away from the interface. \\
\path{trap_decay_length} & m & decay length or Gaussian width & Must be physically resolvable by the grid. \\
\path{trap_profile_shape} & string & \path{exponential} or \path{gaussian} trap profile & Select the profile shape that corresponds to the assumed defect distribution. \\
\bottomrule
\end{longtable}
\endgroup

### Optical Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{alpha} & \(m^{-1}\) & scalar Beer-Lambert absorption coefficient & Suitable for simplified optical studies. It is not wavelength-resolved. \\
\path{optical_material} & string & key into tabulated \(n,k\) data & Required for TMM, EQE, and EL. The key must exist in \path{perovskite_sim/data/nk}. \\
\path{n_optical} & 1 & constant refractive-index fallback & Useful for approximate optical calculations but not a substitute for measured \(n,k\). \\
\path{incoherent} & boolean & thick-layer incoherent TMM treatment & Intended for the first thick substrate. The current TMM path does not support arbitrary incoherent layers mid-stack. \\
\bottomrule
\end{longtable}
\endgroup

### Temperature And Field-Dependent Mobility Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{Nc300} & \(m^{-3}\) & conduction-band density of states at 300 K & Optional temperature-scaling input. Use only when the temperature model is calibrated. \\
\path{Nv300} & \(m^{-3}\) & valence-band density of states at 300 K & Companion to \path{Nc300}. \\
\path{mu_T_gamma} & 1 & mobility power-law temperature exponent & Default keeps a common scattering-style trend. Document any changed value. \\
\path{B_rad_T_gamma} & 1 & radiative coefficient temperature exponent & Default zero preserves legacy behavior. \\
\path{varshni_alpha} & eV K\(^{-1}\) & Varshni band-gap coefficient & Zero disables band-gap temperature shift. \\
\path{varshni_beta} & K & Varshni temperature parameter & Used with \path{varshni_alpha}. \\
\path{v_sat_n} & \(m\,s^{-1}\) & electron velocity-saturation limit & Zero disables Caughey-Thomas saturation for electrons. \\
\path{v_sat_p} & \(m\,s^{-1}\) & hole velocity-saturation limit & Zero disables Caughey-Thomas saturation for holes. \\
\path{ct_beta_n} & 1 & electron Caughey-Thomas exponent & Controls sharpness of velocity saturation. \\
\path{ct_beta_p} & 1 & hole Caughey-Thomas exponent & Same role for holes. \\
\path{pf_gamma_n} & \((V/m)^{-1/2}\) & electron Poole-Frenkel mobility coefficient & Zero disables field-enhanced hopping for electrons. \\
\path{pf_gamma_p} & \((V/m)^{-1/2}\) & hole Poole-Frenkel mobility coefficient & Zero disables field-enhanced hopping for holes. \\
\bottomrule
\end{longtable}
\endgroup

### Device-Level Fields

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.20\linewidth}>{\raggedleft\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}@{}}
\toprule
Field & Unit & Physical role & Beginner guidance \\
\midrule
\endhead
\path{V_bi} & V & built-in voltage in Poisson boundary condition & This is not always identical to the derived heterostack Fermi-level separation. \\
\path{Phi} & \(m^{-2}s^{-1}\) & incident photon flux for Beer-Lambert generation & For spectral experiments use TMM data rather than only changing \path{Phi}. \\
\path{T} & K & device temperature & Affects thermal voltage and enabled temperature-dependent hooks. \\
\path{mode} & string & \path{legacy}, \path{fast}, or \path{full} physics tier & The mode is a ceiling; hooks still need required parameters. \\
\path{interfaces} & \(m\,s^{-1}\) pairs & interface recombination velocities & One pair per internal electrical interface, ordered from the front contact toward the rear contact. \\
\path{S_n_left} & \(m\,s^{-1}\) & left electron contact velocity & \path{None} gives ohmic default. \path{0} is blocking. Large values approach ohmic extraction. \\
\path{S_p_left} & \(m\,s^{-1}\) & left hole contact velocity & Same convention as \path{S_n_left}. \\
\path{S_n_right} & \(m\,s^{-1}\) & right electron contact velocity & Same convention as \path{S_n_left}. \\
\path{S_p_right} & \(m\,s^{-1}\) & right hole contact velocity & Same convention as \path{S_n_left}. \\
\path{microstructure} & object & 2D grain-boundary specification & Used by 2D drivers; ignored by standard 1D paths. \\
\bottomrule
\end{longtable}
\endgroup

## Global Device Parameters

`DeviceStack` stores:

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.34\linewidth}>{\raggedright\arraybackslash}p{0.58\linewidth}@{}}
\toprule
Field & Meaning \\
\midrule
\endhead
\path{V_bi} & built-in voltage used in the Poisson boundary condition \\
\path{Phi} & incident photon flux for Beer-Lambert generation \\
\path{interfaces} & interface recombination velocities \((v_n,v_p)\) \\
\path{T} & device temperature in K \\
\path{mode} & \path{legacy}, \path{fast}, or \path{full} \\
\path{S_n_left}, \path{S_p_left}, \path{S_n_right}, \path{S_p_right} & selective-contact velocities \\
\path{microstructure} & optional 2D grain-boundary block \\
\bottomrule
\end{longtable}
\endgroup

## Built-In Potential

SolarLab has two related built-in potentials:

1. `stack.V_bi`: the value read from YAML and used in the Poisson boundary
   condition.
2. `stack.compute_V_bi()`: a derived effective built-in potential computed from
   Fermi-level differences across the electrical stack.

This distinction matters. Some legacy and benchmark configurations use a
manual $V_\mathrm{bi}$ convention. Heterostacks with $\chi$ and $E_g$
also have band-alignment information that can imply a different effective
Fermi-level separation.

The Poisson boundary is:

$$
\phi(0)=\phi_\mathrm{left}, \qquad
\phi(L)=\phi_\mathrm{left}+V_\mathrm{bi}-V_\mathrm{app}.
$$

The effective built-in potential is used for defaults such as the automatic
upper voltage in J-V sweeps.

## YAML Example

```yaml
device:
  V_bi: 1.1
  Phi: 2.5e21
  T: 300.0
  mode: full
  contacts:
    left:
      S_p: 1.0e3
      S_n: 1.0e-3
    right:
      S_n: 1.0e3
      S_p: 1.0e-3

layers:
  - name: HTL
    role: HTL
    thickness: 2.0e-7
    eps_r: 3.0
    mu_n: 1.0e-8
    mu_p: 1.0e-6
    ni: 1.0e9
    N_A: 1.0e24
    N_D: 0.0
    D_ion: 0.0
    P_lim: 1.0e27
    P0: 0.0
    tau_n: 1.0e-6
    tau_p: 1.0e-6
    n1: 1.0e9
    p1: 1.0e9
    B_rad: 0.0
    C_n: 0.0
    C_p: 0.0
    alpha: 0.0
    chi: 2.2
    Eg: 3.0
```

YAML 1.1 can parse bare scientific notation such as `1e-9` as a string.
SolarLab coerces numeric-looking strings to floats, but writing values as
`1.0e-9` is clearer and safer.

# Governing Equations

This chapter states the governing equations in dimensional SI form. The
implementation evaluates discretized node and face arrays, but the continuous
form clarifies the model assumptions and the physical meaning of each state
variable.

## Poisson Equation

The electrostatic potential is obtained from:

\begin{equation}
\label{eq:poisson}
\frac{\partial}{\partial x}
\left(
\varepsilon_0 \varepsilon_r
\frac{\partial \phi}{\partial x}
\right)
= -\rho .
\end{equation}

Here $\rho$ includes electrons, holes, dopants, and mobile ions. The
discretization uses a harmonic face permittivity:

\begin{equation}
\label{eq:harmonic-eps}
\varepsilon_{i+1/2}
=
\frac{2\varepsilon_i\varepsilon_{i+1}}
{\varepsilon_i+\varepsilon_{i+1}} .
\end{equation}

The harmonic mean is appropriate for a sharp dielectric interface because it
matches the series-capacitor limit.

Assumptions and limits:

- electrostatics is quasistatic;
- magnetic effects and wave propagation in the electrical solve are ignored;
- layer interfaces are abrupt on the scale of the grid;
- $\rho$ is built from the configured mobile species, carriers, and dopants.

## Carrier Continuity

The electron continuity equation is:

\begin{equation}
\label{eq:electron-continuity}
\frac{\partial n}{\partial t}
=
\frac{1}{q}\frac{\partial J_n}{\partial x}
+G-R .
\end{equation}

The hole continuity equation is:

\begin{equation}
\label{eq:hole-continuity}
\frac{\partial p}{\partial t}
=
-\frac{1}{q}\frac{\partial J_p}{\partial x}
+G-R .
\end{equation}

In heterojunctions, SolarLab uses band-corrected potentials:

\begin{equation}
\label{eq:band-corrected-potentials}
\phi_n=\phi+\chi, \qquad
\phi_p=\phi+\chi+E_g .
\end{equation}

These effective potentials allow the same numerical flux formulation to
represent conduction-band and valence-band offsets in heterojunction stacks.

Assumptions and limits:

- carriers are described by drift-diffusion transport, not ballistic
  transport;
- the model is most natural when local quasi-equilibrium is a reasonable
  approximation;
- strongly quantum-confined structures, tunneling-dominated contacts, and hot
  carrier effects require additional modeling beyond this implementation.

## Scharfetter-Gummel Flux

The Bernoulli function is:

\begin{equation}
\label{eq:bernoulli}
B(\xi)=\frac{\xi}{\exp(\xi)-1}.
\end{equation}

The electron face current is:

\begin{equation}
\label{eq:sg-electron}
J_{n,i+1/2}
=
\frac{qD_n}{\Delta x}
\left[
B(\xi)n_{i+1}-B(-\xi)n_i
\right].
\end{equation}

The hole face current is:

\begin{equation}
\label{eq:sg-hole}
J_{p,i+1/2}
=
\frac{qD_p}{\Delta x}
\left[
B(\xi)p_i-B(-\xi)p_{i+1}
\right].
\end{equation}

The Scharfetter-Gummel form is used because it preserves stable exponential
fitting in drift-dominated regimes.

Assumptions and limits:

- the potential and material properties are represented between adjacent grid
  nodes by a face-based discretization;
- the method is robust for high drift fields, but accuracy still depends on
  resolving layer thicknesses, interfaces, and trap profiles.

## Ion Migration

Positive ion vacancies satisfy:

\begin{equation}
\label{eq:ion-continuity}
\frac{\partial P}{\partial t}
=
-\frac{\partial F_P}{\partial x}.
\end{equation}

Ions cannot leave the device, so:

\begin{equation}
\label{eq:ion-zero-flux}
F_P(0)=F_P(L)=0.
\end{equation}

The ion flux includes a steric saturation correction:

\begin{equation}
\label{eq:steric}
s(P)
=
\frac{1}
{\max(1-P_\mathrm{avg}/P_\mathrm{lim},10^{-6})}.
\end{equation}

This term limits ion accumulation as the local density approaches the
configured site-density limit.

Assumptions and limits:

- the default mobile species is a positive vacancy-like species;
- negative mobile species are optional and must be explicitly configured;
- contacts are ion-blocking in the implemented boundary condition;
- the 2D J-V solver treats ions as a frozen Poisson background.

## Recombination

SRH recombination is:

\begin{equation}
\label{eq:srh}
R_\mathrm{SRH}
=
\frac{np-n_i^2}
{\tau_p(n+n_1)+\tau_n(p+p_1)} .
\end{equation}

Radiative recombination is:

\begin{equation}
\label{eq:radiative}
R_\mathrm{rad}=B_\mathrm{rad}(np-n_i^2).
\end{equation}

Auger recombination is:

\begin{equation}
\label{eq:auger}
R_\mathrm{Auger}
=(C_n n+C_p p)(np-n_i^2).
\end{equation}

The total bulk rate is:

\begin{equation}
\label{eq:total-recombination}
R=R_\mathrm{SRH}+R_\mathrm{rad}+R_\mathrm{Auger}.
\end{equation}

Interface recombination uses surface velocities and is converted to a local
volumetric source term near the interface.

Assumptions and limits:

- SRH parameters are effective trap parameters unless trap-energy provenance is
  supplied;
- radiative and Auger coefficients should be layer-appropriate;
- interface recombination is localized numerically near the interface, so grid
  resolution matters when very large surface velocities are used.

## Optical Generation

For Beer-Lambert absorption:

\begin{equation}
\label{eq:beer-lambert}
G(x)
=
\Phi \alpha(x)
\exp\left[
-\int_0^x \alpha(x')\,dx'
\right].
\end{equation}

For transfer-matrix optics, SolarLab loads tabulated $n(\lambda)$ and
$k(\lambda)$ data and computes a wavelength-resolved absorption profile. The
generation profile is integrated over the incident spectrum and cached before
the transient solve.

EQE and electroluminescence require this wavelength-resolved optical
machinery; Beer-Lambert-only stacks do not contain sufficient spectral
information for those experiments.

Assumptions and limits:

- Beer-Lambert optics are fast and useful for trends, but cannot represent
  interference, reflection, or wavelength-dependent collection;
- TMM quality is limited by the provenance and wavelength coverage of the
  supplied optical constants;
- synthetic or placeholder optical constants should be treated as workflow
  demonstrations, not material-specific evidence.

# Boundary And Initial Conditions

## Potential

The potential is Dirichlet at both contacts:

$$
\phi(0)=\phi_\mathrm{left},
\qquad
\phi(L)=\phi_\mathrm{left}+V_\mathrm{bi}-V_\mathrm{app}.
$$

Forward bias reduces the built-in field.

## Carrier Contacts

By default, carrier densities at contacts are ohmic pins derived from local
charge neutrality and mass action:

$$
np=n_i^2, \qquad n-p=N_D-N_A.
$$

The stable two-branch solution avoids numerical cancellation for heavily doped
layers.

In `full` mode, selected contacts can use a Robin flux:

$$
J=\pm qS(u-u_\mathrm{eq}),
$$

where $u$ is $n$ or $p$, and $S$ is a surface recombination/extraction
velocity. $S=0$ is blocking; large $S$ approaches the ohmic limit.

## Ion Boundaries

Both positive and negative mobile ions use zero-flux boundaries. This reflects
the physical assumption that ions redistribute inside the device but do not
leave through the contacts.

## Initial States

The dark equilibrium state starts from charge neutrality and mass action. The
illuminated state is obtained by integrating the transient equations under
illumination at the starting voltage.

For J-V, impedance, and degradation, this light-soaked initial state matters
because ion and carrier memory affect the measurement.

# Physics Tiers

SolarLab uses three fidelity modes:

| Mode | Active physics | Typical use |
|---|---|---|
| `legacy` | disables upgraded hooks | historical benchmark reproduction |
| `fast` | build-once upgrades, no expensive per-RHS hooks | screening |
| `full` | every configured hook | high-fidelity single runs |

The tier is a ceiling, not a command to fabricate missing data. For example:

- TMM requires `optical_material`.
- Field-dependent mobility requires nonzero `v_sat_*` or `pf_gamma_*`.
- Selective contacts require finite `S_*` values.
- Radiative reabsorption requires TMM/photon-recycling support.

# Numerical Method

![SolarLab solver pipeline](perovskite-sim/docs/images/solver_pipeline.png)

The solver discretizes Eqs. \ref{eq:poisson}--\ref{eq:beer-lambert} on the
device grid and then advances the coupled state in time. The important point
for users is that a J-V point is not a closed-form diode equation; it is the
terminal current read from a relaxed drift-diffusion state at a specified
applied voltage.

## Grid

The 1D solver builds a multilayer grid over electrical layers. Substrate layers
are excluded from the electrical grid but retained for TMM optics.

## Method Of Lines

The PDEs are discretized in space first. The result is a coupled system of
ordinary differential equations in time. This is the method-of-lines approach.

The state vector contains the node values for:

$$
\mathbf{y}=(n,p,P)
$$

and optionally a negative ion species.

## Time Integration

SolarLab uses SciPy `solve_ivp`, primarily with the implicit Radau method.
Selected difficult J-V steps use bounded fallback logic, including step
bisection and BDF fallback. The code caps `max_step` around voltage transitions
to avoid large near-flat-band integration jumps.

## Poisson Fast Path

The Poisson equation becomes a tridiagonal linear system. SolarLab prefactors
this operator and reuses the factorization. This is why `build_material_arrays`
must be called outside the RHS and not inside the time-step loop.

## Numerical Safety

Important safety mechanisms include:

- finite-value guards in the RHS;
- harmonic face permittivity;
- stable Bernoulli-function branches;
- TMM transfer-matrix determinant guard;
- median-current steady-state readout;
- explicit `voc_bracketed` flags when a J-V sweep misses the zero crossing.

For publication work, numerical settings are part of the method. Report the
grid size, voltage spacing, settling time or scan rate, tolerances when
modified, and whether the run used `legacy`, `fast`, or `full` mode.

# Running SolarLab

## Installation

From `perovskite-sim/`:

```bash
pip install -e ".[dev]"
```

For the frontend:

```bash
cd frontend
npm install
```

## Backend

From the SolarLab root:

```bash
uvicorn backend.main:app \
  --host 127.0.0.1 --port 8000 \
  --app-dir perovskite-sim --reload
```

Do not run `uvicorn main:app` from inside `backend/`; the backend imports are
written around `--app-dir perovskite-sim`.

## Frontend

```bash
cd perovskite-sim/frontend
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Python API

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)

print(result.metrics_fwd.V_oc)
print(result.metrics_fwd.J_sc)
print(result.metrics_fwd.FF)
print(result.metrics_fwd.PCE)
```

## Worked Example 1: Baseline J-V Sweep

This example runs a standard n-i-p MAPbI3 J-V sweep and reports the principal
photovoltaic metrics.

```bash
cd perovskite-sim
python - <<'PY'
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
result = run_jv_sweep(stack, N_grid=80, n_points=40, v_rate=1.0)
m = result.metrics_fwd
print(f"Voc = {m.V_oc:.4f} V")
print(f"Jsc = {m.J_sc:.2f} A/m^2")
print(f"FF  = {m.FF:.3f}")
print(f"PCE = {100*m.PCE:.2f} %")
print(f"Voc bracketed: {m.voc_bracketed}")
PY
```

Metric interpretation:

- $J_\mathrm{sc}$ is the current density near zero applied voltage.
- $V_\mathrm{oc}$ is interpolated where the terminal current crosses zero.
- FF and PCE are meaningful only when `voc_bracketed` is true.
- Forward and reverse metrics can differ when mobile ions retain scan history.

## Worked Example 2: Absorber Thickness Study

This example varies absorber thickness while leaving the rest of the stack
unchanged, allowing optical collection and recombination trends to be compared
under controlled assumptions.

```python
from dataclasses import replace
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
absorber_index = next(i for i, layer in enumerate(base.layers)
                      if layer.role == "absorber")

for thickness_nm in (200, 400, 700, 1000):
    layers = list(base.layers)
    layers[absorber_index] = replace(
        layers[absorber_index],
        thickness=thickness_nm * 1e-9,
    )
    stack = replace(base, layers=tuple(layers))
    result = run_jv_sweep(stack, N_grid=80, n_points=30, v_rate=2.0)
    m = result.metrics_fwd
    print(thickness_nm, m.V_oc, m.J_sc, m.FF, m.PCE)
```

Expected interpretation:

- Larger absorbers can absorb more light, but can also increase
  recombination or transport loss.
- If $J_\mathrm{sc}$ does not change under Beer-Lambert optics, check
  whether `alpha` and `Phi` are physically configured.
- If FF collapses, inspect spatial profiles and contact selectivity before
  concluding that the absorber is intrinsically poor.

## Worked Example 3: Enabling TMM Optics

This example compares scalar Beer-Lambert generation with wavelength-resolved
transfer-matrix generation.

```python
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

for config in ("configs/nip_MAPbI3.yaml", "configs/nip_MAPbI3_tmm.yaml"):
    stack = load_device_from_yaml(config)
    result = run_jv_sweep(stack, N_grid=60, n_points=21)
    print(config, result.metrics_fwd.J_sc)
```

Expected interpretation:

- TMM includes interference, reflection, parasitic absorption, and spectral
  weighting.
- TMM results are only as credible as the `optical_material` data and AM1.5G
  spectrum used.
- EQE and EL should use TMM-enabled configurations, not Beer-Lambert-only
  presets.

# Backend API

Configuration endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/configs` | list shipped and user presets |
| `GET` | `/api/configs/{name}` | load one preset |
| `POST` | `/api/configs/user` | save a user-edited stack |
| `GET` | `/api/layer-templates` | layer template library |
| `GET` | `/api/optical-materials` | available TMM optical keys |

Preferred experiment interface:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | submit `{kind, config_path|device, params}` |
| `GET` | `/api/jobs/{id}/events` | stream progress/result/error/done events |

Supported job kinds:

```text
jv, current_decomp, spatial, dark_jv, suns_voc, voc_t,
eqe, el, impedance, tpv, degradation, tandem,
mott_schottky, jv_2d, voc_grain_sweep
```

The frontend primarily uses streaming jobs. Long-running experiments should
report progress through callbacks that the backend converts to SSE frames.

# Experiment Manual

## J-V Sweep

The J-V sweep runs forward and reverse voltage scans while preserving state
history. This state-preserving scan is the implemented representation of
hysteresis in an ion-coupled perovskite cell.

Outputs:

- `V_fwd`, `J_fwd`;
- `V_rev`, `J_rev`;
- `metrics_fwd`, `metrics_rev`;
- `hysteresis_index`.

Metrics:

$$
FF=\frac{P_\mathrm{mpp}}{V_\mathrm{oc}J_\mathrm{sc}}.
$$

$$
PCE=\frac{P_\mathrm{mpp}}{1000\,W\,m^{-2}}.
$$

If the sweep never crosses $J=0$, `voc_bracketed=false`. In that case
$V_\mathrm{oc}$, FF, and PCE are sentinel values and the user should increase
`V_max`.

## Current Decomposition

Current decomposition separates:

- electron current $J_n$;
- hole current $J_p$;
- ion current $J_\mathrm{ion}$;
- displacement current $J_\mathrm{disp}$;
- total current.

This separation helps identify whether a transient is dominated by electronic,
ionic, or capacitive response.

## Spatial Profiles

Spatial-profile runs save snapshots of:

- $x$;
- $\phi$;
- $E$;
- $n$;
- $p$;
- $P$;
- $\rho$;
- applied voltage.

These profiles are essential for interpreting unusual J-V behavior. For
example, a high $V_\mathrm{oc}$ with poor FF may indicate selective-contact
or recombination issues rather than poor optical generation.

## Dark J-V

Dark J-V fits the diode equation:

$$
J=J_0\left[\exp\left(\frac{V}{n_\mathrm{id}V_T}\right)-1\right].
$$

The output includes ideality factor $n_\mathrm{id}$, saturation current
$J_0$, and the fitted voltage window.

## Suns-Voc

Suns-Voc varies illumination intensity and extracts $V_\mathrm{oc}$ at each
level. It constructs a pseudo-JV curve:

$$
J_\mathrm{pseudo}(X)
=J_{\mathrm{sc,ref}}-J_\mathrm{sc}(X),
\qquad
V_\mathrm{pseudo}(X)=V_\mathrm{oc}(X).
$$

The pseudo-JV representation is less affected by series resistance than a
standard terminal J-V sweep.

## $V_\mathrm{oc}(T)$

The temperature sweep estimates recombination activation energy by fitting
$V_\mathrm{oc}$ versus temperature:

$$
V_\mathrm{oc}(T)
\approx
\frac{E_A}{q}
-
\frac{kT}{q}
\ln\left(\frac{J_{00}}{J_\mathrm{sc}}\right).
$$

The intercept at $T=0$ is reported as $E_A$ in eV.

## EQE / IPCE

EQE is:

$$
\mathrm{EQE}(\lambda)
=
\frac{|J_\mathrm{sc}(\lambda)|}
{q\Phi_\mathrm{inc}(\lambda)}.
$$

SolarLab computes monochromatic TMM generation, solves the drift-diffusion
problem at short circuit, and integrates against AM1.5G for $J_\mathrm{sc}$.
This experiment requires `optical_material`.

## Electroluminescence

EL uses reciprocity:

$$
\Phi_\mathrm{EL}(\lambda)
=
A_\mathrm{abs}(\lambda)
\phi_\mathrm{bb}(\lambda,T)
\exp\left(\frac{qV_\mathrm{inj}}{kT}\right).
$$

The blackbody photon flux is:

$$
\phi_\mathrm{bb}(\lambda,T)
=
\frac{2\pi c/\lambda^4}
{\exp(hc/\lambda kT)-1}.
$$

The non-radiative voltage loss is:

$$
\Delta V_\mathrm{nr}
=
-V_T\ln(\mathrm{EQE}_\mathrm{EL}).
$$

This experiment also requires TMM optical data.

## Impedance

Impedance applies a small sinusoidal voltage perturbation around a DC bias and
extracts the current response by lock-in fitting. The result is $Z(f)$ in
$\Omega m^2$, suitable for Nyquist and Bode plots.

## Mott-Schottky

Mott-Schottky wraps dark impedance at a fixed frequency and fits:

$$
\frac{1}{C^2}=aV+b.
$$

Then:

$$
V_\mathrm{bi}=-\frac{b}{a},
\qquad
N_\mathrm{eff}
=
-\frac{2}{q\varepsilon_s\varepsilon_0 a}.
$$

For thin fully depleted absorbers, this analysis can over-interpret geometric
capacitance or nonideal response. Fitted slopes should therefore not be
reported automatically as physical dopant densities.

## Transient Photovoltage

TPV applies a small generation pulse at open circuit. SolarLab enforces the
open-circuit condition by adjusting $V_\mathrm{app}$ so that terminal
current remains near zero at each reported time. The voltage decay is fitted as:

$$
V(t)
\approx
V_\mathrm{oc}
+\Delta V_0\exp(-t/\tau).
$$

## Degradation

The degradation experiment is a long-time ion-coupled damage proxy. It
advances the ion/carrier system under bias and periodically performs snapshot
J-V measurements. During each snapshot, ions are frozen while carriers relax,
separating slow ionic redistribution from the instantaneous electronic
response.

## Tandem J-V

The tandem driver:

1. runs one combined TMM optical calculation over the full tandem stack;
2. runs independent top and bottom sub-cell J-V sweeps with fixed generation;
3. series-matches at a common current;
4. sums the sub-cell voltages.

Some tandem optical constants are documented as placeholder/stub data. These
must be replaced before claiming publication-grade material-specific tandem
predictions.

## 2D J-V

The 2D J-V driver uses a tensor-product grid. It is validated against 1D in the
lateral-uniform limit and can add absorber grain boundaries with reduced SRH
lifetimes. Ions are frozen as static Poisson background in 2D.

## $V_\mathrm{oc}(L_g)$ Grain Sweep

The grain sweep repeats 2D J-V simulations over grain sizes $L_g$ and reports
$V_\mathrm{oc}$, $J_\mathrm{sc}$, and FF as functions of grain size. It is
intended to quantify microstructure-driven voltage loss within the frozen-ion
2D approximation.

# Shipped Presets

Representative shipped presets:

| Preset | Purpose |
|---|---|
| `nip_MAPbI3.yaml` | n-i-p MAPbI3, Beer-Lambert optics |
| `nip_MAPbI3_tmm.yaml` | n-i-p MAPbI3 with TMM optics |
| `pin_MAPbI3.yaml` | p-i-n MAPbI3, Beer-Lambert optics |
| `pin_MAPbI3_tmm.yaml` | p-i-n MAPbI3 with TMM optics |
| `ionmonger_benchmark.yaml` | Courtier/IonMonger-style benchmark |
| `driftfusion_benchmark.yaml` | Driftfusion-inspired benchmark |
| `cigs_baseline.yaml` | CIGS-like inorganic thin-film stack |
| `cSi_homojunction.yaml` | crystalline silicon homojunction |
| `radiative_limit.yaml` | photon-recycling and radiative-limit checks |
| `selective_contacts_demo.yaml` | Robin/selective-contact demonstration |
| `field_mobility_demo.yaml` | field-dependent mobility demonstration |
| `tandem_lin2019.yaml` | 2T tandem workflow demonstration |
| `twod/nip_MAPbI3_uniform.yaml` | 2D lateral-uniform parity preset |
| `twod/nip_MAPbI3_singleGB.yaml` | 2D single grain-boundary preset |
| `twod/bcx_combined_demo.yaml` | combined 2D advanced-physics demo |

Preset values are simulation inputs, not universal material constants. For
publication studies, users should document the source of every parameter they
change or use.

# Troubleshooting And Diagnostics

## `voc_bracketed=false`

Meaning: the J-V sweep did not find a current zero crossing inside the sampled
voltage range.

Likely causes:

- `V_max` is too low;
- the voltage grid is too coarse near open circuit;
- the device is numerically unstable near flat band;
- contact or recombination settings produce an unusual current sign.

Recommended response:

1. Increase `V_max`.
2. Increase `n_points` or add a finer voltage region near the expected
   $V_\mathrm{oc}$.
3. Inspect spatial profiles near the highest voltage.
4. Do not report FF or PCE from an unbracketed result.

## Sentinel Or Zero FF/PCE

Zero FF or zero PCE does not always mean a physically dead solar cell. In
SolarLab it can also mean the metric extraction refused to infer a maximum
power point because $V_\mathrm{oc}$ was not bracketed. Always check
`voc_bracketed` first.

## TMM Or EQE Fails Because Optical Data Are Missing

EQE and EL require at least one layer with `optical_material`. If a stack only
uses scalar `alpha`, it can run Beer-Lambert J-V but not wavelength-resolved
experiments. Use a `*_tmm.yaml` preset or add verified $n,k$ data.

## YAML Values Look Numeric But Behave Strangely

YAML 1.1 can parse bare scientific notation such as `1e-9` as a string.
SolarLab coerces numeric-looking fields in the config loader, but users should
write `1.0e-9` for clarity and to avoid surprises in external tools.

## Slow Or Stiff Runs

Thick absorbers, high fields, sharp interfaces, and ion-coupled transients can
make Radau solves slow. Use a coarse grid while developing the configuration,
then rerun final cases at publication settings. For CIGS or crystalline
silicon, transient J-V may be much slower than equilibrium-style checks.

## Unphysical Band Barriers

Large or inconsistent jumps in `chi` and `Eg` can create artificial
heterojunction barriers. If a device has unexpectedly poor FF or current,
plot the band diagram, check the transport-layer band offsets, and verify the
contact velocities.

## 2D Results Differ From 1D

For a lateral-uniform 2D stack, differences from 1D should be small when ions
are frozen consistently and the grids are comparable. If the difference is
large, check:

- whether 1D ions were allowed to move while 2D ions were frozen;
- lateral boundary condition;
- `Nx`, `Ny_per_layer`, and settlement time;
- whether a microstructure block was unintentionally active.

## Frontend Job Appears To Hang

Long-running experiments use server-sent events. If the UI does not update,
check that the backend was started from the SolarLab root with
`--app-dir perovskite-sim`, then verify `/api/jobs/{id}/events` is reachable.

# Validation And Evidence {#validation-and-evidence}

This manual was generated after a full evidence pass on 2026-05-19.

![Validation gate summary](figures/validation_gate_summary.png)

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}>{\raggedright\arraybackslash}p{0.19\linewidth}@{}}
\toprule
Gate & Command & Result & Time \\
\midrule
\endhead
Python default suite & \path{pytest} & 647 passed, 1 skipped, 72 deselected & 704.90 s \\
Python slow suite & \path{pytest -m slow} & 72 passed, 648 deselected & 2156.96 s \\
Python validation suite & \path{pytest -m validation} & 18 passed, 702 deselected & 217.65 s \\
Frontend build & \path{npm run build} & passed & build step 0.25 s \\
Frontend unit tests & \path{npm run test:run} & 22 files, 320 tests passed & 1.29 s \\
\bottomrule
\end{longtable}
\endgroup

The passing suites cover:

- core Poisson, recombination, ion, optics, contact, temperature, and trap
  modules;
- J-V, dark J-V, Suns-Voc, EQE, EL, impedance, Mott-Schottky, TPV,
  degradation, tandem, and 2D workflows;
- backend API and SSE job dispatch;
- frontend result rendering and validation UI;
- IonMonger and Driftfusion-inspired benchmark envelopes;
- TMM Jsc baselines;
- photon-recycling voltage boost;
- 1D/2D lateral-uniform parity;
- 2D microstructure regression;
- physical trends for bandgap, thickness, mobility, dark ideality, and
  Suns-Voc slope.

## Validation Figures

The figures below summarize the validation evidence and slow-suite reference
envelopes. They are derived from the tests and evidence files listed in the
traceability matrix, rather than from independent simulation runs. The script
used to generate them is `docs/manual/generate_manual_figures.py`.

![IonMonger reference metric envelope](figures/ionmonger_reference_metrics.png)

The IonMonger benchmark gate uses `configs/ionmonger_benchmark.yaml` with
`N_grid=40`, `n_points=20`, and `v_rate=5.0`. The plotted interval is the
allowed tolerance around the pinned reverse-scan reference metrics:
$V_\mathrm{oc}=1.1932\,V$, $J_\mathrm{sc}=231.70\,A\,m^{-2}$, $FF=0.7774$,
and $PCE=0.2149$.

![TMM optical baseline envelope](figures/tmm_jsc_baselines.png)

The TMM baseline gate constrains the n-i-p and p-i-n short-circuit currents to
remain within the pinned optical-regression tolerance. This protects the
transfer-matrix implementation from silent drift while allowing small numerical
variation.

![Photon-recycling validation window](figures/photon_recycling_window.png)

The photon-recycling regression requires the radiative-limit $V_\mathrm{oc}$
boost to remain inside the 40--100 mV literature window encoded by the test.
The plotted marker gives the approximate MAPbI3 context value used for the
regression.

![Physics trend validation matrix](figures/physical_trend_matrix.png)

The validation suite checks qualitative semiconductor-physics trends: current
decreases with increasing band gap, mobility affects FF, dark J-V returns an
ideality factor in the expected range, and Suns-Voc produces a physically
plausible slope.

![2D validation and microstructure summary](figures/twod_validation_summary.png)

The 2D slow-suite gates compare the lateral-uniform 2D solver against the 1D
reference and verify that a single grain boundary produces a bounded voltage
penalty rather than an uncontrolled numerical artifact.

## Validation Traceability Matrix

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.32\linewidth}>{\raggedright\arraybackslash}p{0.36\linewidth}@{}}
\toprule
Evidence item & Source test or file & What it defends \\
\midrule
\endhead
Default Python suite & \path{pytest} & broad unit/integration coverage for solver, models, backend, and experiments \\
Slow Python suite & \path{pytest -m slow} & expensive regression baselines, TMM, 2D parity, microstructure, benchmark envelopes \\
Validation suite & \path{pytest -m validation} & expected semiconductor-physics trends \\
IonMonger metric envelope & \path{tests/integration/test_voc_benchmark.py} & benchmark-scale J-V metrics and hysteresis bounds \\
TMM \(J_\mathrm{sc}\) baselines & \path{tests/regression/test_tmm_baseline.py} & optical-stack regression stability \\
Photon-recycling boost & \path{tests/regression/test_photon_recycling_voc.py} & radiative-limit voltage boost sanity \\
2D parity & \path{tests/regression/test_twod_validation.py} & lateral-uniform 2D consistency with 1D \\
Grain-boundary voltage loss & \path{tests/regression/test_twod_microstructure.py} & bounded microstructure-induced voltage penalty \\
Frontend production build & \path{npm run build} & TypeScript/Vite production surface \\
Frontend unit tests & \path{npm run test:run} & UI result rendering, validation, progress, and workstation state \\
\bottomrule
\end{longtable}
\endgroup

Observed warnings:

- `pytest_asyncio` future-default warning for fixture loop scope;
- `np.trapz` deprecation warnings in compatibility and conservation tests;
- Vite bundle chunk-size warning.

None of these warnings are simulator correctness failures, but they should be
tracked before a formal software release.

# Limitations And Best Practices

## Model Assumptions

SolarLab is a numerical research simulator. Predictive accuracy for a material
stack requires physically justified inputs and a validation envelope that
covers the intended regime.

## Data And Optical Limits

TMM, EQE, and EL depend on tabulated $n,k$ data. `load_nk` does not silently
extrapolate outside the source wavelength range. Placeholder optical constants
support workflow testing, but they do not support material-specific
publication claims.

## Numerical Limits

If a J-V sweep does not bracket $V_\mathrm{oc}$, increase `V_max` or refine
the voltage grid near the expected zero crossing. Sentinel zero FF or PCE
should not be interpreted as physical failure until `voc_bracketed` has been
checked.

Mott-Schottky fits can be unreliable for fully depleted thin absorbers.

Thick CIGS and crystalline silicon absorbers can be slow in transient mode.

## 2D Limits

The 2D solver freezes ions. It is suitable for lateral parity and
microstructure studies under that approximation, but not for full dynamic ion
migration in two dimensions.

## Reporting Checklist

A reproducible SolarLab study should report:

- commit hash;
- device YAML or full parameter table;
- physics tier;
- grid settings;
- solver tolerances;
- optical data source;
- validation command results;
- whether $V_\mathrm{oc}$ was bracketed;
- known limitations relevant to the chosen experiment.

## Publication Boundary Statement

A SolarLab figure is publication-ready only when all of the following are
true:

- the device stack and all changed parameters are reported;
- optical constants and material parameters have provenance;
- the chosen physics tier is justified;
- the grid and solver settings are sufficient for the conclusion;
- validation gates relevant to the claimed regime have passed;
- limitations such as frozen 2D ions, placeholder optical data, or screening
  smoke settings are stated explicitly.

If any item is missing, the result may still be useful for debugging or
hypothesis screening, but it should not be described as a calibrated
material-specific prediction.

# SolarScale Screening Bridge

SolarLab can consume readiness-gated material records from SolarScale. This is
a workflow layer, not the core solver.

Safe mappings include:

- dielectric constant to `eps_r`;
- electron and hole mobility to `mu_n` and `mu_p` after unit conversion;
- ion diffusion to `D_ion`;
- ion activation energy to `E_a_ion`.

Device-only unknowns should be swept rather than guessed:

- absorber thickness;
- SRH lifetime;
- trap density;
- interface recombination velocity;
- contact band alignment.

Smoke J-V results in screening workflows validate the process. They are not
publication-grade device predictions unless production settings, provenance,
and validation are also supplied.

# References

1. D. L. Scharfetter and H. K. Gummel, "Large-signal analysis of a silicon
   Read diode oscillator," *IEEE Transactions on Electron Devices* **16**,
   64--77 (1969). doi:10.1109/T-ED.1969.16566.
2. W. Shockley and W. T. Read, "Statistics of the recombinations of holes and
   electrons," *Physical Review* **87**, 835--842 (1952). doi:10.1103/PhysRev.87.835.
3. R. N. Hall, "Electron-hole recombination in germanium," *Physical Review*
   **87**, 387 (1952). doi:10.1103/PhysRev.87.387.
4. N. E. Courtier, J. M. Cave, J. M. Foster, A. B. Walker, and G. Richardson,
   "IonMonger: a free and fast planar perovskite solar cell simulator with
   coupled ion vacancy and charge carrier dynamics," *Journal of
   Computational Electronics* **18**, 1435--1449 (2019).
5. P. Calado, B. Telford, D. Bryant, X. Li, J. Nelson, B. C. O'Regan, and
   P. R. F. Barnes, "Driftfusion: an open source code for simulating ordered
   semiconductor devices with mixed ionic-electronic conducting materials in
   one dimension," *Journal of Computational Electronics* **15**, 1--20
   (2016).
6. M. Burgelman, P. Nollet, and S. Degrave, "Modelling polycrystalline
   semiconductor solar cells," *Thin Solid Films* **361--362**, 527--532
   (2000). doi:10.1016/S0040-6090(99)00825-1.
7. U. Rau, "Reciprocity relation between photovoltaic quantum efficiency and
   electroluminescent emission of solar cells," *Physical Review B* **76**,
   085303 (2007). doi:10.1103/PhysRevB.76.085303.
8. L. A. A. Pettersson, L. S. Roman, and O. Inganas, "Modeling photocurrent
   action spectra of photovoltaic devices based on organic thin films,"
   *Journal of Applied Physics* **86**, 487--496 (1999). doi:10.1063/1.370757.
9. E. Yablonovitch, "Statistical ray optics," *Journal of the Optical Society
   of America* **72**, 899--907 (1982). doi:10.1364/JOSA.72.000899.
10. D. M. Caughey and R. E. Thomas, "Carrier mobilities in silicon empirically
    related to doping and field," *Proceedings of the IEEE* **55**, 2192--2193
    (1967). doi:10.1109/PROC.1967.6123.
11. J. Frenkel, "On pre-breakdown phenomena in insulators and electronic
    semiconductors," *Physical Review* **54**, 647--648 (1938).
    doi:10.1103/PhysRev.54.647.
12. M. A. Green, "Solar cell fill factors: General graph and empirical
    expressions," *Solid-State Electronics* **24**, 788--789 (1981).
    doi:10.1016/0038-1101(81)90062-9.
13. J. Nelson, *The Physics of Solar Cells* (Imperial College Press, London,
    2003).
14. P. Würfel, *Physics of Solar Cells: From Principles to New Concepts*
    (Wiley-VCH, Weinheim, 2005).
15. H. J. Snaith, "Perovskites: The emergence of a new era for low-cost,
    high-efficiency solar cells," *Journal of Physical Chemistry Letters*
    **4**, 3623--3630 (2013). doi:10.1021/jz4020162.
16. L. M. Pazos-Outón et al., "Photon recycling in lead iodide perovskite
    solar cells," *Science* **351**, 1430--1433 (2016).

# Appendix A: Manual Source Trail

The manual was drafted from:

- `docs/solarlab_manual_source_dossier.md`
- `docs/manual/validation_evidence_2026-05-19.md`
- `perovskite-sim/AGENTS.md`
- `README.md`
- `perovskite-sim/README.md`
- `perovskite-sim/backend/README.md`
- `perovskite-sim/perovskite_sim/models/`
- `perovskite-sim/perovskite_sim/physics/`
- `perovskite-sim/perovskite_sim/solver/`
- `perovskite-sim/perovskite_sim/experiments/`
- `perovskite-sim/perovskite_sim/twod/`
- `perovskite-sim/backend/main.py`
- `perovskite-sim/frontend/src/types.ts`

This PDF is intended to be defensible as a project manual. For a journal
supplement, add parameter provenance for every material value and verify the
full bibliography format against the target journal style.

# Appendix B: Figure Source Trail

\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.18}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.32\linewidth}>{\raggedright\arraybackslash}p{0.36\linewidth}@{}}
\toprule
Figure & Source & Reproducibility note \\
\midrule
\endhead
Architecture flow & \path{docs/manual/generate_manual_figures.py} & Diagrammatic summary of repo architecture described in \path{perovskite-sim/AGENTS.md}. \\
Validation gate summary & \path{docs/manual/validation_evidence_2026-05-19.md} & Uses pass counts and runtimes from the completed validation pass. \\
IonMonger reference metrics & \path{tests/integration/test_voc_benchmark.py} & Shows pinned benchmark metrics and tolerances, not a new simulation run. \\
TMM \(J_\mathrm{sc}\) baselines & \path{tests/regression/test_tmm_baseline.py} & Shows pinned n-i-p and p-i-n TMM baselines with ±5 \(A\,m^{-2}\) tolerance. \\
Photon-recycling window & \path{tests/regression/test_photon_recycling_voc.py} & Shows the acceptance window used by the slow regression gate. \\
Physics trend matrix & \path{tests/validation/test_physical_trends.py} & Summarizes physical trend assertions represented by the validation suite. \\
2D validation summary & \path{tests/regression/test_twod_validation.py}, \path{tests/regression/test_twod_microstructure.py} & Summarizes parity and grain-boundary gates represented by the slow suite. \\
\bottomrule
\end{longtable}
\endgroup

# Appendix C: Quick Glossary

| Term | Meaning |
|---|---|
| AM1.5G | Standard terrestrial solar spectrum used for photovoltaic testing. |
| Beer-Lambert | Scalar absorption law using an absorption coefficient $\alpha$. |
| Drift-diffusion | Continuum transport model combining diffusion from concentration gradients and drift from electric fields. |
| FF | Fill factor, the ratio between maximum power and $V_\mathrm{oc}J_\mathrm{sc}$. |
| Hysteresis | Difference between forward and reverse J-V scans caused by state memory, often ionic in perovskites. |
| Method of lines | Numerical method that discretizes space first and integrates the resulting ODE system in time. |
| PCE | Power-conversion efficiency. |
| SG flux | Scharfetter-Gummel flux discretization, used for stable drift-dominated carrier transport. |
| SRH | Shockley-Read-Hall trap-assisted recombination. |
| TMM | Transfer-matrix method for wavelength-resolved multilayer optics. |
| $J_\mathrm{sc}$ | Short-circuit current density. |
| $V_\mathrm{oc}$ | Open-circuit voltage. |

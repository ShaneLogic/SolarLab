---
name: psc-jv-hysteresis-simulation
description: Simulate J-V hysteresis curves in perovskite solar cells using drift-diffusion models with mobile ion transport. Use when generating simulated hysteresis curves to match experimental measurements, implementing numerical drift-diffusion models, or calculating ion distributions and electric potentials in perovskite devices.
---

# PSC J-V Hysteresis Simulation

Use this skill to simulate current-voltage hysteresis in perovskite solar cells using drift-diffusion models that include mobile ion transport.

## When to Use
- Generating J-V hysteresis curves to simulate experimental measurements
- Implementing finite element drift-diffusion models for PSCs
- Calculating ion distribution profiles under applied bias
- Solving coupled Poisson-ion transport equations

## Ion Transport Equations

Apply the following governing equations for mobile anion vacancies:

**Anion Vacancy Conservation:**
```
∂P/∂t + (∂F_P/∂x) = 0
```

**Flux Expression:**
```
F_P = -D_+ * (∂P/∂x) + V_T * P * (∂φ/∂x)
```
Where:
- `P`: Mobile anion vacancy density
- `D_+`: Anion vacancy diffusion coefficient
- `V_T`: Thermal voltage (~0.026 V)
- `φ`: Electric potential

**Poisson's Equation:**
```
ε * (∂²φ/∂x²) = -q * (p - n + P - N₀)
```
Where:
- `ε`: Permittivity of the perovskite
- `p`, `n`: Hole and electron densities
- `N₀`: Uniform density of cation vacancies (ensures neutrality)
- `q`: Elementary charge

## Boundary and Initial Conditions

**Spatial Domain:**
- Perovskite layer bounded by ETL at x = 0
- Perovskite layer bounded by HTL at x = 1

**Potential Definition:**
- Potentials measured in units of thermal voltage (V_T ≈ 0.026 V)
- Total potential drop: Δφ = φ|x=0 - φ|x=1 = φ_bi - φ(t)
- `φ_bi`: Built-in potential (~1.1 V, dimensionless ≈ 42)
- `φ(t)`: Time-dependent applied potential
- Typical applied range: -0.5 V to 2 V

**Apply boundary conditions at x=0 and x=1 for:**
- Anion vacancy fluxes (F_P)
- Electric potential (φ)
- Electron and hole densities (n, p)

**Initial conditions at t=0 required for:**
- All spatial points for variables P, n, p, φ

## J-V Hysteresis Simulation Protocol

### 1. Preconditioning Phase
- Increase applied voltage from built-in voltage (V_bi) to 1.2 V
- Duration of increase: 5 seconds
- Hold voltage at 1.2 V for additional 5 seconds
- This establishes initial ion distribution

### 2. Scanning Phase
- **Forward scan:** Scan from forward bias (φ_ap > φ_bi) to short circuit (φ_ap = 0)
- **Reverse scan:** Scan from short circuit back to forward bias
- Scan rate: 100 mV/s (≈7.1 in dimensionless units)
- Record current density at each voltage point

### 3. Numerical Parameters
- **Method:** Finite element scheme
- **Spatial resolution:** N = 400 mesh points
- **Solver tolerances:** RelTol (default), AbsTol = 10⁻⁸
- **Expected simulation time:** ~6 seconds on standard desktop

### 4. Expected Output
Current density as a function of applied voltage (J-V curve) showing hysteresis between forward and reverse scans.
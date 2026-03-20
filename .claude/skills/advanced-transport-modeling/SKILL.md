---
name: advanced-transport-modeling
description: Configure and apply advanced transport model features including quasi-Fermi level input handling and steric effects in ion transport. Use when modeling high ion vacancy densities, enabling non-Boltzmann statistics, or setting flexible doping parameters for transport layers.
---

# Advanced Transport Modeling

## When to Use
- Setting transport layer doping parameters with quasi-Fermi level specifications
- Modeling high ion vacancy densities approaching saturation
- Enabling steric hindrance effects in ion transport
- Configuring non-Boltzmann statistics in transport layers
- Bypassing manual Boltzmann distribution conversions

## Quasi-Fermi Level Input Handling

### Flexible Input Options
Transport layer parameters can be set using EITHER:
1. Direct carrier density (doping_density)
2. Quasi-Fermi Level (QFL)

### Configuration Procedure

1. **Set Parameters Independently for Each Layer:**
   - ETL may use doping density input
   - HTL may use QFL input simultaneously
   - Each layer treated independently

2. **When User Sets QFL:**
   a. System calculates relevant doping density automatically
   b. Uses the specific statistical model assigned to that layer
   c. Applies inverse statistical integral S^-1

3. **When User Sets Doping Density:**
   a. Direct specification without conversion
   b. Compatible with standard workflows

**Advantage:** Bypasses manual conversion required by Boltzmann distributions.

## Steric Effects - Modified Drift Model

### When to Enable
- Ion vacancy density P approaches P_lim (site density)
- High ion concentrations cause lattice site blocking
- Standard Poisson-Nernst-Planck (PNP) assumes P << P_lim (invalid at high densities)

### Activation Condition
`IF (Steric effects enabled AND NonlinearFP = 'Drift') THEN apply modified drift flux`

### Modified Ion Flux Equation

**Electrochemical Potential:**
```
μ = k_B T ln(γ P / P_lim) + φ
```

**Activity Coefficient (Lattice Diffusion):**
```
γ = (1 - P / P_lim)^-1
```

**Mobility with Steric Effects:**
```
M = γ D_I / (k_B T)
```

**Modified Ion Flux F_P:**
```
F_P = -D_I (∂P/∂x) + (qP / k_B T)(∂φ/∂x) [1 / (1 - P/P_lim)]
```

### Physical Justification
- Based on hopping model where adjacent sites may be occupied
- Enforces maximum of one ion per lattice site
- Prevents unphysical ion concentrations exceeding site availability
- Divisor [1/(1-P/P_lim)] increases drift term as density approaches limit

## Key Parameters

### Steric Effects
- P: Ion vacancy density
- P_lim: Density of anion sites (max vacancy density)
- D_I: Constant diffusion coefficient
- φ: Electric potential
- q: Elementary charge
- k_B: Boltzmann constant
- T: Temperature

### QFL Input
- QFL: Quasi-Fermi level input by user
- doping_density: Calculated or input carrier density
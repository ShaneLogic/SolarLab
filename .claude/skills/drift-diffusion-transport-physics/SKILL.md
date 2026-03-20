---
name: drift-diffusion-transport-physics
description: Model charge transport in perovskite solar cells using drift-diffusion equations. Apply conservation equations, flux equations, and Poisson's equation for perovskite, ETL, and HTL regions. Use when setting up the core physics model for PSC simulations.
---

# Drift-Diffusion Transport Physics

## When to Use
- Setting up the core physics model for PSC drift-diffusion simulations
- Defining transport equations for perovskite absorber layer (0 < x < b)
- Modeling electron and hole transport layers (ETL and HTL)
- Establishing boundary conditions at metal contacts and interfaces

## Perovskite Layer (0 < x < b)

### Conservation Equations
Apply continuity equations balancing generation, recombination, and flux divergence:
- **Conduction Electrons:** Continuity equation
- **Valence Holes:** Continuity equation  
- **Halide Ion Vacancies:** Continuity equation

### Flux Equations (Drift-Diffusion)
For each carrier type, flux includes:
- Drift term (electric field dependence)
- Diffusion term (concentration gradient)
- Thermodiffusion term (temperature gradient)

Current densities: J_n (electrons), J_p (holes), J_P (vacancies)

### Poisson's Equation
Couple electric potential (φ) to charge densities:
```
∇²φ = -(q/ε)(n - p + P - N̂_0)
```
where N̂_0 is constant, uniform background density of immobile cation vacancies.

### Source Terms
- Photo-generation: G(x, t)
- Recombination: R(n, p)

## Transport Layers

### Electron Transport Layer (ETL): -bE < x < 0
- **Model:** Majority carriers (free electrons) only
- **Equations:** Conservation + Flux + Poisson's
- **Statistics:** Uses ETL-specific statistical integral S_E

### Hole Transport Layer (HTL): b < x < b+bH
- **Model:** Majority carriers (holes) only
- **Equations:** Conservation + Flux + Poisson's
- **Statistics:** Uses HTL-specific statistical integral S_H

## Boundary Conditions

### ETL/Metal Contact (x = -bE)
Ohmic condition:
```
n = g_E * S_E^-1(E_ct - V(t) + T_E)
```
where E_ct is cathode workfunction and V(t) is applied voltage.

### HTL/Metal Contact (x = b+bH)
Ohmic condition with parasitic resistance:
```
p = g_H * S_H^-1(E_an - V(t) + T_H)
```
where E_an is anode workfunction.

**Built-in Voltage:** V_bi = E_ct - E_an

## Interface Continuity (x=0, x=b)

### Flux Continuity
```
J_n(0-) = J_n(0+) - R̂_l
```
where R̂_l represents interface recombination.

### Potential Continuity
```
φ(0-) = φ(0+)
```
---
name: carrier-transport-mechanics
description: Analyze and calculate carrier transport properties in semiconductors including thermal velocity, drift velocity, conductivity, and mobility. Use this when analyzing carrier motion, current flow, or transport parameters under thermal equilibrium or with applied electric fields.
---

# Carrier Transport Mechanics

## When to Use
- Calculating thermal motion of carriers in semiconductors
- Analyzing drift velocity and current under electric fields
- Determining electrical conductivity and mobility
- Calculating Joule heating from carrier transport
- Analyzing relationships between scattering time, effective mass, and transport properties

## Thermal Velocity (Thermal Equilibrium)

Use for carriers in non-degenerate semiconductors (carrier density < 0.1 Nc or 0.1 Nv) at temperature > 0K with no external field.

1. **Calculate root-mean-square thermal velocity** using equipartition principle:
   - Kinetic energy: 1/2 m* v² = 3/2 kT
   - Formula: v_rms = sqrt(3kT/m*)

2. **Understand motion characteristics**:
   - Ideal lattice: No scattering, waves not localized
   - Real lattice: Random walk with mean free path ~ several hundred Å

## Drift Velocity and Conductivity (Applied Electric Field)

Use when an external electric field is applied to a homogeneous semiconductor.

### Sign Conventions
- Electron charge: -e
- Hole charge: +e
- Electric field F: F = -d(Psi)/dx (negative gradient of vacuum level)
- Forces: f_n = -eF (electrons), f_p = +eF (holes)

### Calculate Drift Velocity
1. **Incremental velocity gain**: delta_v = (F * tau) / m*
2. **Drift velocities**:
   - Electrons: vD_n = -(e * F * tau) / m_n
   - Holes: vD_p = (e * F * tau) / m_p

### Calculate Current Density
- j_n = n * (-e) * vD_n = (e² * n * tau / m_n) * F

### Calculate Conductivity
- Electron: sigma_n = e² * n * tau / m_n
- For homogeneous: F = V/d, V = I * R
- Resistance: R = d / (A * sigma)
- Specific resistivity: rho_n = 1/sigma_n

### Calculate Joule Heating
- Energy per collision: (m*/2) * (delta_v)²
- Events per second: n/tau
- Thermal energy gain: Q = sigma_n * F²

## Mobility Principles

### Mobility Factors
- Higher when: time between collisions is larger, effective mass is smaller
- Charge signs: -e for electrons, +e for holes
- Mobility signs: negative for electrons, positive for holes
- **Conductivity is always positive** despite sign of mobility
- Convention: Use magnitude μ = μ_n

### Mobility Definition
- μ = ratio of drift velocity to field magnitude
- μ_n = e * tau / m_n

## Key Variables
| Variable | Description |
|----------|-------------|
| v_rms | Root mean square thermal velocity |
| k | Boltzmann constant |
| T | Absolute temperature |
| m* | Effective mass of carrier |
| F | Electric field strength |
| tau | Mean scattering time (relaxation time) |
| sigma | Electrical conductivity |
| mu | Carrier mobility |
| j | Current density |
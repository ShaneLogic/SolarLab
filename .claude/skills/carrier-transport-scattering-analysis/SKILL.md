---
name: carrier-transport-scattering-analysis
description: Analyze carrier transport behavior in semiconductors including gas-kinetic scattering models, momentum relaxation, and energy relaxation. Use when calculating mobility, mean free path, scattering times, or understanding how carriers lose momentum and energy to the lattice.
---

# Carrier Transport and Scattering Analysis

## When to Use
- Calculating carrier mobility, mean free path, or scattering times
- Analyzing how carriers scatter and lose directional momentum
- Understanding energy exchange between carriers and lattice (Joule heating)
- Estimating transport properties from scattering cross-sections

## Core Workflow

### Step 1: Gas-Kinetic Scattering Model
Calculate basic scattering statistics when analyzing carrier transport.

#### Calculate Mean Free Path (λ):
```
λ = 1 / (Nsc × Sn)
```
- `Nsc`: density of scattering centers (cm⁻³)
- `Sn`: scattering cross-section

#### Calculate Scattering Time (τsc):
```
τsc = λ / vrms
```
- `vrms`: root mean square velocity

#### Estimate Carrier Mobility (Drude):
```
μ = e × τsc / m
```
- `e`: elementary charge
- `m`: effective mass

**Note**: This model tends to overestimate tolerable defect densities due to simplified assumptions.

### Step 2: Momentum Relaxation Time
Determine how quickly carriers lose directional memory (randomization of path).

#### Calculate from Scattering Angle:
```
τm = τsc / (1 - <cosΘ>)
```
- `Θ`: scattering angle
- `<cosΘ>`: average cosine of scattering angle

#### Scattering Types:
- **Isotropic scattering**: `<cosΘ>` = 0 → τm = τsc
- **Large angle scattering** (Θ > 90°): Memory-erasing events
- **Small angle scattering**: Multiple events needed → τm > τsc

#### Alternative from Mobility:
```
μ = e × τm / m
```

### Step 3: Energy Relaxation Time
Determine how quickly carriers reach thermal equilibrium with the lattice.

#### Calculate:
```
τE = ΔE / (dE/dt)
```
- `ΔE`: average surplus energy
- `dE/dt`: rate of energy loss

#### Analyze by Phonon Type:
- **Acoustic phonons**: Energy loss ~0.1% per event (negligible)
  - Equivalent phonon mass: `M* = 2 × k × T_lattice / vs²`
- **Optical phonons**: `τE / τsc ≈ 1` at room temperature
- **Defect scattering**: Energy loss negligible when M >> m

#### Key Insight:
**Momentum relaxes after one or few collisions, but energy takes many more events** to dissipate.

## Output
- Mean free path (λ) in cm
- Scattering time (τsc) and momentum relaxation time (τm) in seconds
- Energy relaxation time (τE) in seconds
- Carrier mobility (μ) in cm²/V·s
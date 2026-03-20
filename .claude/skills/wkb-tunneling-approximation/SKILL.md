---
name: wkb-tunneling-approximation
description: Calculate quantum tunneling transmission probability through triangular barriers, parabolic barriers, and band-to-band transitions using the WKB (Wentzel-Kramers-Brillouin) approximation. Use when analyzing tunneling phenomena in semiconductor devices, quantum wells, or when barrier shapes can be approximated as triangular or parabolic with defined electric fields.
---

# WKB Tunneling Approximation

## When to Use This Skill

Use the WKB tunneling approximation when:
- Analyzing quantum tunneling through potential barriers
- Barrier shape is triangular, parabolic, or involves band-to-band transitions
- Electric field is well-defined and constant
- Pre-exponential factors can be neglected (order of 1)
- Working with semiconductor devices requiring tunneling current calculations

## Prerequisites

- Understanding of WKB approximation theory
- Knowledge of calculus integration
- Barrier shape must be mathematically defined
- Effective mass and barrier height must be known

## Core Workflow

### 1. Identify Barrier Type

Determine which barrier approximation applies:
- **Triangular barrier**: Linear potential variation with position
- **Parabolic barrier**: Quadratic potential, often from image force lowering
- **Band-to-band tunneling**: Tunneling across semiconductor bandgap
- **Overlapping fields**: Combined Coulomb and external electric fields

### 2. Apply General WKB Formula

The transmission probability follows:
```
Te = exp(-2 * integral(k(x) dx))
```
where:
- `k(x) = sqrt(2m * DeltaE(x)) / hbar`
- Pre-exponential factor is neglected (assumed ≈ 1)

### 3. Select Specific Formula

Choose the appropriate formula based on barrier type:
- Triangular barrier: Use Eq (19.13) with linear field dependence
- Parabolic barrier: Use modified formula with pi factor
- Band-to-band: Replace DeltaE with bandgap Eg
- Overlapping fields: Use combined field expression

### 4. Calculate Transmission Probability

Compute the exponential transmission coefficient using the selected formula.

## Key Constraints

- Pre-exponential factor is neglected (order of 1)
- Barrier shape must be clearly defined
- Electric field must be constant or well-characterized
- Effective mass approximation applies
- For band-to-band tunneling, fields > 10^6 V/cm typically required

## Critical Insights

- Parabolic barriers have reduced exponent by factor 3pi/16 ≈ 0.59 compared to triangular barriers
- Barrier height/field relation is superlinear: doubling barrier height requires 2^(3/2) ≈ 2.83 times the field for same probability
- Narrow gap semiconductors enable significant tunneling at lower fields

## Common Applications

- Zener diode breakdown analysis
- Tunnel diode current calculations
- Gate leakage in MOSFETs
- Quantum well tunneling
- Trap-assisted tunneling in dielectrics
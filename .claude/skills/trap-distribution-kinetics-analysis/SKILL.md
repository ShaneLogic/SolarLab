---
name: trap-distribution-kinetics-analysis
description: Analyze carrier kinetics in semiconductors with continuous trap distributions using quasi-stationary approximation. Use when modeling decay transients, calculating effective carrier lifetimes, or extracting trap density profiles from kinetic measurements in materials with distributed defect levels rather than discrete traps.
---

# Trap Distribution Kinetics Analysis

## When to Use

Apply this skill when:
- Analyzing materials with continuous trap distributions (not discrete levels)
- Modeling carrier decay kinetics in semiconductors with multiple defect levels
- Extracting trap distribution profiles from transient measurements
- Conditions permit quasi-stationary approximation (high temperature, high light intensity, slow n(t) changes)

## Prerequisites

Verify these conditions before applying:
- Sufficiently high temperature for rapid trap kinetics
- High light intensity enabling quasi-Fermi level description
- Slow changes in carrier density n(t) allowing trap filling to follow

## Procedure

### Step 1: Validate Quasi-Stationary Approximation

Confirm that temperature and light intensity are high enough to justify:
- Quasi-Fermi level (EF) can describe trap filling/depletion
- Traps can equilibrate with the conduction band during measurement

### Step 2: Calculate Total Trapped Electron Density

Integrate the trap distribution from the conduction band edge to the quasi-Fermi level:

```
nt = ∫[0 to EF] Nt(E) dE
```

Where:
- Nt(E) = trap density distribution function [cm⁻³(eV)⁻¹]
- EF = quasi-Fermi level position
- nt = total trapped electron density [cm⁻³]

### Step 3: Determine Trapped Electron Change Rate

Calculate dn_t/dn as a function of electron density using the integral of the trap distribution.

### Step 4: Apply Modified Decay Time Relation

Substitute the distribution-based dn_t/dn into the decay relation:

```
τ / τn = 1 + (1/n) × ∫[0 to EF] Nt(E) dE
```

Where:
- τ = measured decay time
- τn = intrinsic carrier lifetime
- n = free electron density

### Step 5: Extract Trap Distribution from Measurements

1. Obtain τ₀ = n/g₀ from steady-state conditions
2. Measure τ from decay immediately after switching off optical excitation
3. Deconvolve the trap distribution profile from the modified decay time

### Step 6: Analyze Relaxation Components

For n distinct groups of trap levels, expect (n-1) relaxation times corresponding to:
- Band-to-band recombination
- Transitions between bands and different trap levels

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| Nt(E) | Distribution | Trap density distribution function [cm⁻³(eV)⁻¹] |
| EF | Energy | Quasi-Fermi level position [eV] |
| nt | Density | Total trapped electron density [cm⁻³] |
| τ₀ | Time | Inverse generation rate constant (n/g₀) [s] |
| τ | Time | Measured decay time [s] |
| τn | Time | Intrinsic carrier lifetime [s] |

## Output

- Modified decay time (τ) accounting for continuous trap distribution
- Trap distribution profile Nt(E) via deconvolution from kinetic data
- Relaxation time components for multi-level systems

## Constraints

- n(t) must change slowly enough for trap filling to follow (quasi-steady-state condition)
- Quasi-Fermi level must be able to follow changes in n during measurement

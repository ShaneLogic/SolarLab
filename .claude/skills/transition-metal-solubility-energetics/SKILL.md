---
name: transition-metal-solubility-energetics
description: Calculate solubility limits and crystal field splitting energies for transition metal impurities in semiconductors. Use when estimating impurity concentrations in Si, III-V, or II-VI semiconductors, or when calculating energy levels and crystal field parameters.
---

# Transition Metal Solubility and Energetics

## When to Use

Use this skill when you need to:
- Estimate solubility limits of transition metals in semiconductor hosts (Si, III-V, II-VI)
- Calculate crystal field splitting (Δ) for transition metal impurities
- Analyze gettering mechanisms for transition metal contamination
- Understand energy level structures of transition metal defects

## Prerequisites

- Host lattice type (Si, III-V, or II-VI)
- Transition metal impurity type
- Ligand distance (R) for crystal field calculations

## Solubility Estimation

### Determine Solubility by Host Material

**Silicon hosts:**
- 3d transition metals: ~10^14 cm^-3 (extremely low)
- Cu and Ni: ~5 × 10^17 cm^-3 (much higher)

**III-V and II-VI semiconductors:**
- General transition metals: ~10^17 cm^-3
- Exception: Mn forms continuous solid solutions (dilute magnetic semiconductors)

### Key Considerations
- Below 10^14 cm^-3, positive identification of 3d metals may be difficult
- Analytical chemistry thresholds may limit detection accuracy

## Crystal Field Splitting Calculation

### Apply Crystal Field Formula

Calculate Δ using:
```
Δ = 5 × Λ × <r^4> / R^5
```

Where:
- Δ = Crystal field splitting parameter (Energy)
- Λ = Proportionality constant related to covalency
- <r^4> = Expectation value of r^4 for 3d wavefunction
- R = Distance to nearest neighbor (ligand)

### Factors Affecting Δ

The splitting energy increases with:
- Higher covalency
- Higher charge of transition metal
- Higher transition series

## Energy Level Structure

### Td Symmetry Levels

Multi-electron levels in tetrahedral symmetry: a1, a2, e, t1, t2

### Substitutional vs Interstitial Defects

**Substitutional defects:**
- Less structured energy spectrum
- Some elements (Ti, Fe) yield no deep levels

**Interstitial vs Substitutional:**
- Energy sequences are inverted between configurations

## Gettering Mechanism

Oxygen in silicon attracts transition metals via stress-enhanced diffusion. This principle enables intentional gettering of metallic impurities.

Example: Cu decoration of dislocations
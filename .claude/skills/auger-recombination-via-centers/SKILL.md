---
name: auger-recombination-via-centers
description: Calculate Auger recombination lifetime and capture cross-section when carriers recombine through defect centers under high excitation conditions. Use this when analyzing recombination in materials with significant defect densities or high carrier injection levels where Auger processes via recombination centers dominate.
---

# Auger Recombination via Recombination Centers

## When to Use
- High excitation conditions with sufficient carrier density
- Materials with significant recombination center density (Nr)
- Analyzing carrier lifetime dependence on majority carrier density
- Distinguishing intrinsic Auger recombination from other mechanisms

## Prerequisites
- Recombination center density (Nr)
- Carrier density (n)
- Energy of recombination center (Er) and conduction band edge (Ec)
- Effective mass (m*) of carriers

## Procedure

### 1. Calculate Auger Lifetime
Use the center-specific formula:
```
τ_A = [Q × exp(ΔE/kT)] / (R × Nr × n²)
```
Where:
- Q = 0.5 × (m₀/m*)³/² × (1 + m*/m₀)² (mass-dependent factor)
- R = 2.6 (enhancement factor)
- ΔE = Ec - Er (energy difference from conduction band to center)
- k = Boltzmann constant
- T = temperature

### 2. Estimate Auger Coefficient for Specific Materials
For GaAs and similar materials:
```
B ∝ (Ec - Er)⁻³.⁵
```
This power law describes the energy dependence.

### 3. Calculate Capture Cross-Section
```
σ_Auger = B / (v_th × n)
```
Where v_th is the thermal velocity of carriers.

### 4. Verify Intrinsic Auger Behavior
Check if the process is intrinsic by examining the dependence:
- τ_minority ∝ 1/(majority_density)²
- This quadratic dependence confirms intrinsic Auger recombination

## Typical Values
- For Ec - Er = 0.5 eV and m* = 0.1m₀: B ≈ 10⁻²⁶ cm⁶s⁻¹
- For Nr = 10¹⁷ cm⁻³ and n = 10¹⁴ cm⁻³: τ_A ≈ 10⁻⁵ s
- Shallow trap capture cross-section: ~10⁻¹¹ cm²
- Deep trap capture cross-section: ~10⁻¹⁸ cm²

## Key Insight
The exponential term exp(ΔE/kT) accounts for the energy barrier between the band and the recombination center, making deep centers less effective for Auger recombination.
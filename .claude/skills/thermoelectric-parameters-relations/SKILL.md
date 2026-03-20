---
name: thermoelectric-parameters-relations
description: Calculate and analyze thermoelectric parameters (electrical resistivity, Seebeck coefficient, Peltier coefficient, thermal conductivity) and their interrelations for thermoelectric materials, semiconductors under temperature gradients, and metal thermocouples. Use when analyzing thermoelectric effects, applying Kelvin relations, or using the Wiedemann-Franz law for metals.
---

# Thermoelectric Parameters and Relations

## When to Use
- Analyzing thermoelectric effects in materials
- Calculating transport properties under temperature gradients
- Working with metal thermocouples or thermoelectric devices
- Applying the Kelvin relation between Peltier and Seebeck coefficients
- Using the Wiedemann-Franz law for metallic systems
- Estimating typical thermoelectric values for common metals

## Core Parameters

Identify and calculate the four conventional thermoelectric parameters:

1. **Electrical resistivity** (ρ = 1/σ)
2. **Thermoelectric power/Seebeck coefficient** (α)
3. **Peltier coefficient** (πi)
4. **Thermal conductivity** (κ)

These parameters are obtained by inverting the transport equations (Eq. 16.13) after solving the Boltzmann equation for small perturbations.

## Key Relations

### Kelvin Relation
Connects Peltier coefficient to thermoelectric power:
```
πi = αT
```
Use for practical calculations when one coefficient is known.

### Wiedemann-Franz Law (Metals Only)
```
κ/σT = L
```
Where L is the Lorentz number.

**Critical constraints:**
- Applies ONLY to metals
- Requires thermal conductivity determined by electron gas ALONE
- Lattice conductivity must be negligible

## Physical Mechanism

When a temperature gradient is applied:
1. Electrons at the hotter end gain higher kinetic energy
2. In simple metals (alkali metals), electrons move preferentially to the cooler end
3. The cooler end becomes negatively charged
4. This establishes the thermoelectric voltage

## Calculation Workflow

1. **Identify the material type** (metal vs. semiconductor)
2. **Determine applicable relations**:
   - Metals: Use Wiedemann-Franz law if lattice conductivity is negligible
   - All materials: Use Kelvin relation for Peltier/Seebeck conversion
3. **Apply classical Drude result** for simple metals when appropriate
4. **Verify values** against typical material data

## Constraints and Limitations

- Wiedemann-Franz law does NOT apply when lattice conductivity is significant
- Classical Drude results may be off by factors (e.g., factor of 2) due to insufficient scattering considerations
- Boltzmann equation solution requires small perturbation conditions

## Validation

Compare calculated values with typical material data:
- Seebeck coefficient α typically on order of 1 μV/K
- k/e = 86 μV/K is a useful reference scale
- Consult material-specific reference values (Na, K, Pt, Au, Cu, Li, W) for verification
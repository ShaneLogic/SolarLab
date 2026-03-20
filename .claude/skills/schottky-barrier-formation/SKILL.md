---
name: schottky-barrier-formation
description: Calculate Schottky barrier formation when high work function metals contact n-type semiconductors. Use to determine interface electron density, depletion region formation, and assess rectifying contact behavior in metal-semiconductor junctions.
---

# Schottky Barrier Formation

Calculate Schottky barrier characteristics when a metal contacts an n-type semiconductor and determine whether rectifying behavior occurs.

## When to Use
- Metal with high work function contacting n-type semiconductor
- Designing metal-semiconductor contacts
- Analyzing rectifying or ohmic contact behavior
- Calculating interface electron density and depletion region width

## Formation Mechanism

### 1. Carrier Redistribution Process

- Electrons from semiconductor **leak out** into adjacent metal
- Electron density at interface **reduced** below equilibrium bulk value (n₁₀)
- **Positive space-charge region** created within semiconductor near metal contact
- **Negative charge** located at metal/semiconductor interface
- Total device remains neutral

### 2. Interface Electron Density Calculation

**Linear model equation:**
```
n_c = N_c * exp[-φ_MS / (kT)]
```

Where:
- φ_MS = φ_M - χ_Sc (metal-semiconductor work function difference)
- N_c = effective level density at metal-semiconductor interface
- φ_M = metal work function
- χ_Sc = semiconductor electron affinity

**Initial assumption:** n_c is independent of current and applied voltage

### 3. Bulk Electron Density

```
n₁₀ = N_d
```

Where N_d = density of shallow, uncompensated donors

## Depletion Region Formation

### Condition for Rectifying Contact

**IF** n_c < n₁₀ (interface density lower than bulk):

**THEN:**
- **Depletion region** forms
- Properties similar to depletion region in highly doped half of nn+ junction
- Produces **rectifying (blocking) contact**

### Rectification Strength

Schottky barrier can be **substantially more rectifying** than nn+ junction because:
```
n_c / n₁₀ << n₁₀ / n₂₀
```

(Larger ratio of interface to bulk density difference)

## Calculation Steps

### 1. Determine Material Parameters
- Metal work function (φ_M)
- Semiconductor electron affinity (χ_Sc)
- Donor density (N_d)
- Temperature (T)
- Effective level density at interface (N_c)

### 2. Calculate Interface Electron Density
```
n_c = N_c * exp[-(φ_M - χ_Sc) / (kT)]
```

### 3. Compare with Bulk Density
- If n_c << n₁₀: Strong rectifying contact
- If n_c ≈ n₁₀: Ohmic or weakly rectifying contact

### 4. Assess Barrier Height
Barrier height = φ_MS = φ_M - χ_Sc

## Example Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Electron mobility | μ_n | 100 cm²/Vs |
| Dielectric constant | ε | 10 |
| Temperature | T | 300 K |
| Donor density | N_d₁ | 10¹⁷ cm⁻³ |
| Interface density | n_c | 10¹⁰ cm⁻³ |

**Result:** Pronounced Schottky barrier behavior

## Output
- Interface electron density (n_c)
- Assessment of rectifying vs ohmic contact behavior
- Barrier height calculation
- Depletion region characteristics
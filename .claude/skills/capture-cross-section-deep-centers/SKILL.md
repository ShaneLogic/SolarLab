---
name: capture-cross-section-deep-centers
description: Calculate capture cross-section pre-exponential factors for deep centers in semiconductors, including thermal activation corrections. Use when analyzing carrier capture by deep traps where thermal energy effects are significant.
---

# Capture Cross-Section for Deep Centers

## When to Use
- Calculating capture cross-sections for deep trap centers in semiconductors
- Analyzing carrier recombination at deep levels with thermal activation
- Estimating capture rates for defects requiring substantial thermal energy

## Core Procedure

### 1. Calculate Pre-exponential Factor s∞
Use Eq. (22.14):
```
s∞ = (e²/4πε₀εr)² × (m*/ħ²) × (1/Ec - Et)
```

Where:
- `e`: elementary charge
- `ε₀`: vacuum permittivity
- `εr`: relative permittivity
- `m*`: effective mass
- `ħ`: reduced Planck constant
- `Ec - Et`: energy difference between conduction band and trap level

### 2. Apply Thermal Correction Factor
Calculate correction factor η using Eq. (22.15):
```
η = (1 + 2m*Ea/ħ²k²)⁻¹
```

Where:
- `Ea`: activation energy
- `k`: wave vector

Apply correction: `modified s∞ = η × s∞`

### 3. Calculate Material Parameter
Use Eq. (22.16):
```
Parameter = √(2m*/ħ²) × (Ec - Et)
```

### 4. Interpret Results
- Typical s∞ for deep centers: ~10⁻¹⁵ cm
- Thermal activation energies range from few meV to 0.6 eV
- Larger energies indicate significant thermal activation requirements

## Output
Modified capture cross-section pre-exponential factor (ηs∞) in cm
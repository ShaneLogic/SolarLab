---
name: germanium-semiconductor-parameters
description: Access standardized material constants and device parameters for Germanium Schottky barriers and pn-junctions at 300K. Use when setting up numerical simulations, validating models, or performing calculations for Ge-based semiconductor devices.
---

# Germanium Semiconductor Parameters

## Overview
This skill provides the complete set of physical constants, material parameters, and device specifications for Germanium semiconductor devices at T = 300 K. Use these parameters when setting up numerical simulations or performing analytical calculations.

## When to Use
- Setting up numerical simulation for Ge Schottky barrier
- Setting up numerical simulation for Ge pn-junction
- Validating device models with reference data
- Performing analytical calculations requiring Ge constants
- Comparing experimental data with theoretical models

## Germanium Schottky Barrier Parameters (Table 29.1)

### Doping and Densities
```
Nd  (Donor density)             = 10^16 cm⁻³
Nr  (Recombination center)      = 10^16 cm⁻³
Nc  (Effective DOS conduction)  = 10^19 cm⁻³
Nv  (Effective DOS valence)     = 6 × 10^18 cm⁻³
n10 (Electron density at x=0)    = 10^16 cm⁻³
```

### Energy and Potentials
```
Eg                (Band gap)                    = 0.66 eV
Ei - Er           (Intrinsic - Recombination)   = 0.10 eV
ψ_MS,n  (Metal-semiconductor, n-type)          = 0.319 V
ψ_MS,p  (Metal-semiconductor, p-type)          = 0.341 V
p10     (Hole density at x=d1)                 = 5.13 × 10^10 cm⁻³
```

### Transport and Capture
```
ccv   (Capture coefficient)        = 10^-9 cm³/s
μn    (Electron mobility)          = 3900 cm²/Vs
μp    (Hole mobility)              = 1900 cm²/Vs
nc    (Electron density at interface) = 4.48 × 10^13 cm⁻³
pc    (Hole density at interface)     = 1.15 × 10^13 cm⁻³
vn*   (Thermal velocity)          = 5.7 × 10^6 cm/s
```

### Environment
```
ε     (Dielectric constant)       = 16
T     (Temperature)               = 300 K
ni    (Intrinsic carrier density) = 2.265 × 10^13 cm⁻³
```

## Abrupt Germanium PN-Junction Parameters (Table 30.1)

### Doping Concentrations
```
Na (Acceptor doping) = 10^17 cm⁻³
Nd (Donor doping)   = 10^16 cm⁻³
```

### Recombination Centers
```
Nr1 (Recombination center 1) = 10^17 cm⁻³
Nr2 (Recombination center 2) = 10^16 cm⁻³
```

### Effective Densities of States
```
Nc (Conduction band) = 1.04 × 10^19 cm⁻³
Nv (Valence band)    = 5.76 × 10^18 cm⁻³
```

### Equilibrium Carrier Densities
```
n10 (p-side electron)  = 5.138 × 10^9 cm⁻³
p10 (p-side hole)      = 10^17 cm⁻³
n20 (n-side electron)  = 10^16 cm⁻³
p20 (n-side hole)      = 5.138 × 10^10 cm⁻³
```

### Mobilities
```
un0 (Electron mobility) = 3900 cm²/Vs
up0 (Hole mobility)     = 1900 cm²/Vs
```

### Energy and Capture Constants
```
Eg (Band gap)           = 0.66 eV
El - Er (Energy level)  = 0.1 eV
Ccr (Capture coefficient) = 10^-9 cm³/s
Ccv (Capture coefficient) = 10^-9 cm³/s
vn* (Thermal velocity)    = 5.7 × 10^6 cm/s
vp* (Thermal velocity)    = 5.7 × 10^6 cm/s
```

### Material Constants
```
ε (Dielectric constant) = 16
T (Temperature)         = 300 K
```

## Common Derived Parameters

### Intrinsic Carrier Density
```
ni = 2.265 × 10^13 cm⁻³ (for both Schottky and pn-junction)
```

### Diffusion Potential (PN-Junction)
```
ψD = (kT/e) * ln(Na * Nd / ni²)
```

### Thermal Voltage
```
kT/e at 300 K ≈ 0.0259 V
```

## Usage Guidelines

### For Schottky Barrier Analysis
- Use Table 29.1 parameters for Schottky-specific calculations
- Key for interface phenomena: nc, pc, vn*, ψ_MS

### For PN-Junction Analysis
- Use Table 30.1 parameters for junction electrostatics
- Key for space charge: Na, Nd, Nr1, Nr2
- Key for transport: un0, up0, Nc, Nv

### For Mixed Analysis
- Both parameter sets share: Eg, ε, T, ni, mobilities
- Device-specific: interface densities (Schottky) vs equilibrium densities (pn-junction)

## Output
- Complete parameter set for numerical simulation
- Validated constants matching reference tables
- Ready-to-use values for Eqs. (30.32)-(30.42) for pn-junction
- Ready-to-use values for Schottky barrier modeling

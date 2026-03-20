---
name: acoustic-phonon-scattering-mobility
description: Calculate electron mobility limited by acoustic phonon scattering in direct bandgap semiconductors and determine directional elastic constants. Use when analyzing temperature-dependent mobility, phonon scattering effects, or elastic properties in specific crystallographic directions.
---

# Acoustic Phonon Scattering Mobility

## When to Use
- Calculating temperature-dependent electron mobility in direct bandgap semiconductors
- Analyzing mobility when acoustic phonon scattering is the dominant mechanism
- Determining longitudinal elastic constants for specific crystal directions
- Characterizing semiconductor transport at temperatures above phonon freeze-out

## Prerequisites
- Semiconductor must have a direct bandgap (T^(-3/2) dependence NOT observed for indirect bandgap)
- Temperature high enough for acoustic phonon scattering to be predominant
- Deformation potential (Ξ) must be known
- Elastic tensor components (c_ik) known for directional calculations

## Calculate Electron Mobility from Acoustic Phonon Scattering

Use when acoustic phonon scattering dominates and semiconductor has direct bandgap.

### Formula (Eq. 17.14)
```
μ_n,ac = √(38π) × m_n^(5/2) / (e × k × h × T^(4/3) × c_l^(3/2) × Ξ²)
```

### Key Temperature Dependence
- μ ∝ T^(-3/2) at higher temperatures
- This dependence is characteristic of direct bandgap semiconductors
- For indirect bandgap: intervalley scattering dominates, T^(-3/2) NOT observed

### Important Considerations
1. **Deformation potential Ξ**: Used here has only slowly varying components in space
2. **Effective mass**: Requires proper mix of density-of-state and mobility effective masses (Eq. 17.15)
3. **Indirect bandgap limitation**: Intervalley scattering has significant influence

## Calculate Directional Elastic Constants

Use when calculating elastic response for longitudinal acoustic phonons in specific crystallographic directions.

### Longitudinal Elastic Constant (c_l) by Direction

**For (100) direction:**
```
c_l = c_11
```

**For (110) direction:**
```
c_l = 1/3(c_11 + c_12 + c_44)
```

**For (111) direction:**
```
c_l = 1/3(c_11 + 2c_12 + c_44)
```

### Where
- c_ik = components of the elastic tensor
- Formulas represent weighted averages based on geometric projection along crystallographic directions
- Directional dependencies arise from anisotropic nature of crystal lattices

## Key Variables
| Variable | Description |
|----------|-------------|
| μ_n,ac | Electron mobility due to acoustic phonon scattering (cm²/V·s) |
| m_n | Electron effective mass |
| T | Temperature (K) |
| c_l | Longitudinal elastic constant |
| Ξ | Deformation potential (eV) |
| c_11, c_12, c_44 | Components of the elastic tensor |
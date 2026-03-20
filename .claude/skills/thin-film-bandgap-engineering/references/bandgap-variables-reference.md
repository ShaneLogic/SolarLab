# Bandgap Engineering Variables Reference

## Eg - Bandgap Energy
- Units: eV (electron volts)
- a-Si:H: ~1.7 eV
- α-SiGe: 1.1-1.7 eV (Ge-dependent)
- α-SiC: >1.7 eV (C-dependent)

## Ge_conc - Germanium Concentration
- Units: atomic % or gas flow ratio
- Higher Ge concentration → Lower bandgap
- Trade-off: Quality decreases at high Ge

## Grading Parameters

### Bandgap Profile Types
- **Flat**: Constant bandgap across i-layer
- **V-shaped**: Wide-narrow-wide profile
- **U-shaped**: Narrow-wide-narrow profile (not recommended)

### Grading Slope
- Controls electric field strength in i-layer
- Steeper grading → Stronger field
- Trade-off: May create interface defects

## Multijunction Parameters

### Current Matching
- Total current limited by lowest current junction
- Optimize thicknesses for current balance
- Spectral dependence affects matching

### Tunnel Junction
- Must be low-resistance, optically transparent
- Critical for multijunction performance

## Quality Metrics

### Defect Density
- Increases as bandgap narrows in SiGe
- Degrades below Eg = 1.4 eV
- Measured by subgap absorption

### Transport Properties
- Mobility-lifetime product μτ
- Diffusion length
- Both degrade at narrow bandgaps
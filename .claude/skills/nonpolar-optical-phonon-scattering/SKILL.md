---
name: nonpolar-optical-phonon-scattering
description: Calculate carrier mobility due to optical phonon scattering in nonpolar semiconductors (Si, Ge). Use when analyzing transport properties in elemental nonpolar semiconductors or nonpolar compound semiconductors, particularly for p-bands or when temperature is near or above the Debye temperature.
---

# Nonpolar Optical Phonon Scattering

## When to Use
Apply this skill when:
- Analyzing carrier mobility in elemental nonpolar semiconductors (Si, Ge)
- Working with nonpolar compound semiconductors
- Studying p-band transport (important for holes)
- Temperature is near or above the Debye temperature
- High-purity semiconductor materials where optical phonon scattering becomes dominant

**Do NOT use for:**
- Polar semiconductors (use polar optical phonon scattering instead)
- Γ_6-band electrons in materials like n-type InSb (scattering vanishes)

## Prerequisites
Before applying this scattering mechanism, ensure you have:
- Optical deformation potential D_o (eV)
- Semiconductor density ρ (kg/m³)
- Debye temperature θ (K)
- Operating temperature T (K)

## Procedure

### 1. Identify Applicability
Confirm the semiconductor is nonpolar and the band structure is appropriate:
- Elemental semiconductors: Si, Ge
- Nonpolar compound semiconductors
- For p-bands: scattering is significant
- For Γ_6-band: scattering vanishes (unimportant for n-type InSb)

### 2. Calculate Temperature Function Φ(T)
Determine the temperature-dependent auxiliary function:
- At low temperatures (T < 0.3θ): Φ(T) is large (10⁴-10⁵ at T = θ/10)
- Near Debye temperature (T ≈ θ): Φ(T) decreases to order of 1
- Above Debye temperature: Φ(T) continues decreasing
- Use approximation forms Φ_f(T) or Φ_g(T) (see references)

### 3. Compute Carrier Mobility
Use the deformation potential formalism for longitudinal optical phonons:

μ = f(ρ, D_o, Φ(T))

Where the mobility depends on:
- Semiconductor density ρ
- Optical deformation potential D_o
- Temperature function Φ(T)

### 4. Interpret Results
- Mobility becomes strongly temperature-dependent through Φ(T)
- Near and above Debye temperature, optical phonon scattering becomes the determining factor in high-purity semiconductors
- Compare with other scattering mechanisms to determine overall mobility

## Key Physical Insights
- **Mechanism**: Annihilation of optical phonon creates high-energy electron, which immediately creates optical phonon in a highly probable transition
- **Energy**: Electron energy is conserved in turnaround, but momentum is NOT conserved
- **Modes**: Couples both longitudinal and transverse optical modes with scattering electron
- **Inelasticity**: When electrons have sufficient energy to create optical phonons, scattering becomes very effective and inelastic

## For p-Bands
For p-band calculations, relate parameter b to optical deformation potential:
- D_o = -(3/2)b

## References
See `optical-phonon-formulas.md` for detailed formulas, numerical expressions, and deformation potential values.
---
name: frohlich-optical-phonon-mobility
description: Calculate carrier mobility limited by ionic optical phonon scattering in polar semiconductors using Fröhlich interaction formalism. Use for polar or partially ionic semiconductors with longitudinal optical (LO) phonon modes (e.g., GaAs, InSb, III-V compounds, II-VI compounds). Particularly important when the material has a partially ionic lattice and optical phonon effects dominate over deformation potential scattering.
---

# Fröhlich Optical Phonon Mobility Calculation

## When to Use This Skill

Apply this formalism when:
- Working with polar semiconductors or partially ionic lattices
- The material has longitudinal optical (LO) phonon modes
- Optical phonon scattering is significant (often dominates at room temperature)
- You need to account for the larger dipole moment effects in ionic crystals
- Analyzing materials like GaAs, InSb, InP, ZnSe, or similar III-V/II-VI compounds

## Prerequisites

Before proceeding, ensure you have:
- Coupling constant (α_c) from Eq. 17.6
- Debye temperature (θ) in Kelvin
- Callen effective charge (e_c)
- Unit lattice cell volume (V_o)
- Temperature (T) of the system
- Carrier density (n), especially if n > 10^19 cm⁻³

## Core Workflow

### 1. Verify Material Properties

Confirm the semiconductor has:
- Partially ionic lattice structure
- LO phonon modes present
- Known effective charge and coupling parameters

### 2. Calculate Polarization Field

The physical origin is the induced field from lattice vibration dipoles:

```
P = f(d_r, V_o, e_c)
```

where d_r is the change in interatomic distance during vibration.

**Note**: See references/frohlich-equations.md for the complete Eq. 17.25.

### 3. Determine Temperature Regime

Compare the operating temperature (T) to the Debye temperature (θ):

- **Low temperature regime**: T < θ
- **High temperature regime**: T ≥ θ

The calculation approach differs significantly between regimes.

### 4. Apply Low-Temperature Formula (T < θ)

Use the variational method (Howarth and Sondheimer, 1953):

```
μ_n,ion,opt = 2m_n × e × α_c × k × θ × exp(θ/T)
```

Key characteristics:
- Mobility decreases **linearly** with increasing coupling constant
- Mobility increases **exponentially** with decreasing temperature
- This represents "optical phonon freeze-out" at low temperatures

### 5. Apply High-Temperature Formula (T ≥ θ)

Use the Seeger (1973) approximation for higher temperatures.

**Note**: See references/frohlich-equations.md for Eq. 17.28.

### 6. Account for Screening Effects

At high carrier densities (n > 10^19 cm⁻³), screening becomes important:

- Use Yukawa-type screened potential (Zawadzki and Szymanska, 1971)
- This reduces the effectiveness of optical phonon scattering
- At higher temperatures, screening parameter F_op ≈ 1
- Mobility increases by factor of ~3.5 for InSb at 300K with n = 10^19 cm⁻³

### 7. Handle Degenerate Semiconductor Case

When the Fermi level shifts into the conduction band:

- Mobility becomes explicitly electron-density-dependent
- Use Eq. 17.30 for the modified calculation

**Note**: See references/frohlich-equations.md for the complete formula.

## Key Physical Insights

1. **Dominance over deformation potential**: The Fröhlich interaction typically has larger influence than deformation potential scattering due to the larger dipole moment in ionic lattices

2. **Temperature dependence**:
   - Below θ: Strong exponential increase as T decreases
   - Above θ: Different power-law dependence applies

3. **Screening importance**:
   - Negligible at low carrier densities
   - Significant at n > 10^19 cm⁻³
   - Can increase mobility by factor of 3-4 in heavily doped materials

## Output Interpretation

The result μ_n,ion,opt represents the electron mobility limited by ionic optical phonon scattering. To obtain total mobility, combine with other scattering mechanisms using Matthiessen's rule.

## References

- Detailed equations and derivations: references/frohlich-equations.md
- Variational method: Howarth and Sondheimer (1953)
- High-T approximation: Seeger (1973)
- Screening effects: Zawadzki and Szymanska (1971)
- Original formulation: Ehrenreich (1961)
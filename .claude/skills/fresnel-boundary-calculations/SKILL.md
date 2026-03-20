---
name: fresnel-boundary-calculations
description: Calculate reflected and transmitted wave amplitudes and energy fluxes at dielectric interfaces using Fresnel equations. Use this skill when analyzing light reflection/transmission at material boundaries, determining reflectance and transmittance ratios, or working with air-dielectric interfaces at normal or oblique incidence.
---

# Fresnel Boundary Calculations

Calculate the behavior of electromagnetic waves at dielectric interfaces, including amplitude coefficients for reflected and transmitted waves and energy-based reflectance/transmittance.

## When to Use
- Calculating amplitude of reflected and transmitted waves at an interface
- Analyzing reflection/transmission at air-dielectric interfaces
- Determining energy flux ratios (reflectance R and transmittance T)
- Working with normal or oblique incidence angles
- Modeling interface behavior in optical systems

## Fresnel Equations for Wave Amplitudes

Use for calculating amplitude coefficients at general dielectric interfaces.

**Procedure:**
1. **Apply boundary conditions at z=0:**
   - Tangential components of electric and magnetic fields must be continuous
   - This applies to both incident, reflected, and transmitted waves
2. **Set up the system:**
   - Four equations (two from electric field, two from magnetic field)
   - Four unknowns: components of reflected and transmitted electric vectors
3. **Solve for amplitude coefficients:**
   - The resulting Fresnel equations relate reflected (r) and transmitted (t) amplitudes to incident (i) amplitude
   - Components are split into:
     - Parallel (∥) component: in the plane of incidence
     - Perpendicular (⊥) component: perpendicular to plane of incidence
4. **Use the general Fresnel equations (Eq 20.45)** for any angle of incidence

## Simplified Reflectance and Transmittance

Use for air-dielectric interfaces with specific simplifications.

**Procedure:**
1. **Define energy flux quantities:**
   - Reflectance (R) and Transmittance (T) are ratios of energy flux normal to interface
   - Based on Poynting vector magnitude
2. **Apply simplifications for air-dielectric interface:**
   - Assumption 1: First medium is air → n_r1 = 1, σ₁ = 0
   - Assumption 2: Second medium is non-absorbing → σ₂ = 0
3. **Calculate for general incidence angles:**
   - Reflected beam components (Eq 20.49)
   - Transmitted beam components (Eq 20.50)
4. **Calculate for normal incidence (simplified):**
   - When incident angle = 0°, parallel and perpendicular components are identical
   - Reflected beam (Eq 20.51): R = ((n_r2 - 1) / (n_r2 + 1))²
   - Transmitted beam (Eq 20.51): T = 4 × n_r2 / (n_r2 + 1)²

## Key Variables
- **r_⊥, r_∥**: Reflection coefficients (perpendicular and parallel components)
- **t_⊥, t_∥**: Transmission coefficients (perpendicular and parallel components)
- **R**: Reflectance (energy ratio)
- **T**: Transmittance (energy ratio)
- **n_r1, n_r2**: Refractive indices of first and second media
- **θ_i, θ_t**: Angles of incidence and transmission

## Constraints
- Tangential field components are continuous at the boundary
- Energy conservation: R + T = 1 (for non-absorbing media)
- Simplified normal incidence formulas assume first medium is air
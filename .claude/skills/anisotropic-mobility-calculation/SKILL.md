---
name: anisotropic-mobility-calculation
description: Calculate direction-dependent mobility in anisotropic semiconductors (Ge, Si) accounting for band structure anisotropy and ionized impurity scattering effects. Use when working with multi-valley semiconductors with ellipsoidal constant energy surfaces or when anisotropy corrections to standard ionized impurity scattering models are required.
---

# Anisotropic Mobility Calculation

## When to Use
- Calculating mobility in anisotropic semiconductors (Ge, Si)
- Working with multi-valley semiconductors having ellipsoidal constant energy surfaces
- Applying ionized impurity scattering models where band structure anisotropy significantly affects results
- Determining directional mobility differences in crystals

## Prerequisites
Verify the following before proceeding:
- Effective mass anisotropy ratio (m_n∥/m_n⊥) is known for the material
- Carrier density (n) is available
- Screening length is substantially smaller than mean free path
- Low-angle scattering events dominate (typical for ionized impurity scattering)

## Calculation Procedure

### 1. Identify Material Parameters
- Determine effective mass ratio m_n∥/m_n⊥ for the semiconductor
- Common values: Ge = 19, Si = 5.2
- Identify carrier density n

### 2. Calculate Screening Length
- Compute Debye screening length: λ_D ∝ 1/√n
- Verify: λ_D << mean free path (ensures no successive collisions in same defect region)

### 3. Apply Anisotropy Factor
- Calculate anisotropy factor K_a = μ∥/μ⊥
- Use screened Yukawa potential model
- Note: K_a decreases with increasing carrier density

### 4. Compute Direction-Dependent Mobility
- Apply anisotropy-corrected mobility formula (Samoilovich et al. 1961)
- Account for density of states variation along valley ellipsoid axes
- Factor in increased randomization with higher carrier densities

### 5. Validate Results
- Compare measured mobility anisotropy with calculated values
- Reference experimental data at relevant temperatures (e.g., 77K measurements for i-Ge)

## Key Relationships

- Parallel mobility (μ∥) and perpendicular mobility (μ⊥) differ due to effective mass anisotropy
- Higher carrier density → smaller scattering cross-section → more randomizing ion scattering
- Density of states is largest along long axes of valley ellipsoids

## Material-Specific Values

See `references/material-parameters.md` for detailed values and formulas.

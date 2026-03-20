---
name: complex-refractive-index-relations
description: Convert between complex dielectric function components (ε₁, ε₂) and optical constants (refractive index nᵣ, extinction coefficient κ) for absorbing media. Use when analyzing optical properties of semiconductors or materials with absorption/damping.
---

# Complex Refractive Index Relations

## When to Use
- Relating optical constants to dielectric function in absorbing media
- Calculating ε₁ and ε₂ from measured nᵣ and κ (or vice versa)
- Analyzing optical properties of semiconductors with absorption
- Converting between optical and electrical response parameters

## Core Relations

The complex dielectric function ε = ε₁ + iε₂ is related to optical constants:

```
ε₁ = nᵣ² - κ²
ε₂ = 2 × nᵣ × κ
```

**Where:**
- `ε₁` = Real part of complex dielectric function
- `ε₂` = Imaginary part of complex dielectric function  
- `nᵣ` = Refractive index
- `κ` (kappa) = Extinction coefficient

## Physical Meaning

### ε₁ (Real Part):
- Represents the dispersive (energy storage) component
- Related to polarization and refractive index
- Describes how the medium responds to the electric field

### ε₂ (Imaginary Part):
- Represents the absorptive (energy dissipation) component
- Directly related to optical absorption coefficient α
- Describes energy loss in the medium

### nᵣ (Refractive Index):
- Determines phase velocity of light in the medium
- Related to ε₁ in non-absorbing materials

### κ (Extinction Coefficient):
- Determines attenuation of light amplitude
- Related to absorption coefficient: α = 4πκ/λ

## Constraints

- **Assumes damping/absorption is present** (κ > 0)
- For non-absorbing materials (κ = 0): ε₁ = nᵣ² and ε₂ = 0
- Valid for homogeneous, isotropic media

## Inverse Relations

To solve for nᵣ and κ given ε₁ and ε₂:
```
nᵣ = √[(|ε| + ε₁) / 2]
κ = √[(|ε| - ε₁) / 2]
```
where |ε| = √(ε₁² + ε₂²)

## Variables
| Symbol | Description | Type |
|--------|-------------|------|
| ε₁ | Real part of complex dielectric function | real |
| ε₂ | Imaginary part of complex dielectric function | real |
| nᵣ | Refractive index | real |
| κ | Extinction coefficient | real |
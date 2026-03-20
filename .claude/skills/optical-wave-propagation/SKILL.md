---
name: optical-wave-propagation
description: Model electromagnetic wave propagation in dielectric media and semiconductors, calculating phase velocity, refractive index, energy flow, and absorption parameters. Use this skill when analyzing light transmission through materials, determining wave parameters in absorbing or non-absorbing media, or working with complex refractive indices.
---

# Optical Wave Propagation

Analyze electromagnetic wave behavior in dielectric materials and semiconductors, including both ideal non-absorbing media and materials with finite conductivity.

## When to Use
- Modeling light propagation in dielectric media
- Calculating wave parameters (phase velocity, refractive index)
- Analyzing energy flow and energy density of optical waves
- Working with absorbing semiconductors or materials with finite conductivity
- Determining complex optical parameters (n, k, ε₁, ε₂)

## Wave Propagation in Non-Absorbing Dielectrics

Use for ideal optical media with μ=1 and σ=0 (non-conductive).

**Procedure:**
1. Start with the undamped wave equation from Maxwell's equations:
   - ∇²E - (με/c²) × ∂²E/∂t² = 0
2. Solve for a plane wave traveling in x-direction with linear polarization in y-direction:
   - E_y = f(x) × exp(iωt)
   - Solution: E_y = A × exp[iω(t - x/v)]
3. Calculate phase velocity v:
   - v = c / n_r
   - Where n_r = √ε (refractive index)
4. Calculate energy flow using the Poynting vector S:
   - S = E × H
   - Magnitude |S| = (1/2) × c × ε × |E|²
5. Calculate total energy density w:
   - w = (1/2) × (ε|E|² + μ|H|²) = ε|E|²
   - (Assumes equal energy in electrical and magnetic components)

## Wave Propagation in Absorbing Semiconductors

Use for materials with finite conductivity σ(ω) that exhibit damping/absorption.

**Procedure:**
1. Introduce finite conductivity to obtain the damped wave equation:
   - ∇²E - (με/c²) × ∂²E/∂t² - (μσ/c²) × ∂E/∂t = 0
2. Solve for the plane wave traveling in x-direction:
   - E_y = A × exp[iω(t - nx/c)]
   - Where n is the complex index of refraction
3. Define the complex index of refraction:
   - n = n_r + ik
   - n_r: Real part (refractive index)
   - k: Imaginary part (extinction coefficient)
   - Relationship: n² = ε
4. Define the complex dielectric constant:
   - ε = ε₁ + iε₂
   - Relationship to conductivity: ε = ε_∞ + iσ/(ωε₀)
   - ε₁: Related to dispersion
   - ε₂: Related to absorption

## Key Variables
- **E**: Electric field vector
- **H**: Magnetic field vector
- **n_r**: Real refractive index
- **n**: Complex refractive index (n_r + ik)
- **k**: Extinction coefficient
- **v**: Phase velocity
- **σ**: Frequency-dependent electrical conductivity
- **ε**: Complex dielectric constant (ε₁ + iε₂)
- **S**: Poynting vector (energy flux)
- **w**: Energy density
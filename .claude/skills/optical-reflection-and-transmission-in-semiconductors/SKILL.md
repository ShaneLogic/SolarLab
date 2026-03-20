---
name: optical-reflection-and-transmission-in-semiconductors
description: Calculate reflection, refraction, and transmission properties of semiconductor interfaces and slabs including Snell's law, Brewster angle, critical angle, and interference effects in thin films. Use this for designing photosensing devices, analyzing light trapping, or characterizing semiconductor optical properties.
---

# Optical Reflection and Transmission in Semiconductors

## When to Use
- Light incident on semiconductor interfaces
- Designing light-trapping structures for photosensors
- Analyzing thin-film interference patterns
- Calculating absorption coefficients from optical measurements

## Snell's Law and Critical Angle

### Snell's Law
```
n_r1 × sin(Φ_i) = n_r2 × sin(Φ_t)
```
Where:
- n_r1, n_r2: refractive indices of incident and transmitted media
- Φ_i: angle of incidence
- Φ_t: angle of transmission (refraction)

### Critical Angle for Total Reflection
**Condition:** n_r1 > n_r2 (light traveling from higher to lower index)

**Formula:**
```
sin(Φ_c) = n_r2 / n_r1
```
When Φ_i ≥ Φ_c, total internal reflection occurs.

**Application:** Used in photosensing devices for light trapping to increase optical path and photosensitivity.

## Reflection Properties

### Reflected Wave Approximation (n_r >> κ)
Use Eq 20.52 for parallel and perpendicular components at low incidence angles.

### Brewster Angle (Φ_B)
**Definition:** Angle at which reflected light becomes nearly linearly polarized (parallel component negligible)

**Formula:**
```
tan(Φ_B) = n_r2 / n_r1
```

### Normal Incidence Reflectance
```
R₀ = [(n_r - 1)² + κ²] / [(n_r + 1)² + κ²]
```
Where:
- n_r: refractive index
- κ: extinction coefficient

**Geometric interpretation:** Equation of a circle centered at n = (1+R₀)/(1-R₀) with radius 2√R₀/(1-R₀).

## Multi-Reflection and Interference

### Phase Shift Between Reflections
```
δ = (2π × n_r × d) / λ
```
Where:
- d: slab thickness
- λ: wavelength in vacuum

### Total Reflectivity (R_Σ)
```
R_Σ = [sinh²(α₀d/2) + sin²(δ)] / [sinh²(α₀d/2 + γ) + sin²(δ + ψ)]
```

### Total Transmissivity (T_Σ)
```
T_Σ = [sinh²(γ) + sin²(ψ)] / [sinh²(α₀d/2 + γ) + sin²(δ + ψ)]
```

### Auxiliary Functions
```
γ = ln(1/R₀)
ψ = tan⁻¹[2κ / (n_r² + κ² - 1)]
```

**Note:** If absorption vanishes (κ=0, ψ=0), then R_Σ + T_Σ = 1

## Practical Measurement Techniques

### Averaging for Absorption Coefficient
Interference patterns make evaluation of κ (or α₀) difficult. Solutions:
- Make surfaces slightly nonplanar (rough)
- Use slightly polychromatic light

### Average Reflectivity and Transmissivity
Use Eq 20.61 and 20.62 for averaged values.

### Abac Chart Method
Use Fig 20.6 with measured T and R to find:
- α₀: optical absorption coefficient
- R₀: normal incidence reflectance

## Key Applications
1. **Light Trapping:** Critical angle for total reflection increases effective path
2. **Thin-Film Characterization:** Interference patterns reveal thickness and optical constants
3. **Surface Quality Analysis:** Deviations from ideal interference indicate roughness or defects
4. **Material Characterization:** Abac chart method extracts α₀ from T and R measurements
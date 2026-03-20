---
name: optical-absorption-wave-propagation
description: Model electromagnetic wave propagation in absorbing media and calculate optical absorption coefficients including Urbach tail behavior. Use when analyzing light absorption in disordered semiconductors, calculating attenuation of energy flux, or studying wave propagation with complex refractive index.
---

# Optical Absorption and Wave Propagation

## When to Use
- Modeling electromagnetic wave propagation in absorbing media
- Calculating optical absorption coefficients
- Analyzing absorption in disordered/heavily doped semiconductors (Urbach tail)
- Determining attenuation of energy flux vs field amplitude
- Working with complex refractive index and extinction coefficient

## Damped Wave Equation in Absorbing Media

Use when modeling wave propagation in a medium with damping.

### Wave Solution
```
E = E₀ × exp[i(ωt - (ω/c)n_r x)] × exp[-(ω/c)κ x]
```

### Components
- **First exponential**: Oscillating wave propagating with phase velocity c/n_r
- **Second exponential**: Damping factor (attenuation) as wave travels distance x
- Damping explicitly depends on extinction coefficient κ

## Optical Absorption Coefficient

Use when calculating attenuation of energy intensity (not field).

### Energy Flux Damping
- Energy flux damps as: exp(-α₀ x)

### Relation to Field Damping
- Energy flow (Poynting vector) = product of electric and magnetic vectors
- This introduces factor of 2 in exponent compared to field damping

### Absorption Coefficient Formula
```
α₀ = (2ωκ) / c
```

### Relation to Electrical Conductivity (Eq. 20.24)
```
α₀ = σ / (n_r × c × ε₀)
```

### Additional Notes
- Magnetic field H_z is phase-shifted by δ
- tan δ = κ/n_r
- Conductivity σ used is the optical frequency conductivity

## Urbach Tail in Disordered Semiconductors

Use when analyzing optical absorption in heavily doped, amorphous, or disordered semiconductors with large concentration of point defects.

### Prerequisites
- Large concentration of point defects
- Disorder due to dopants with correlation length ~ interatomic spacing

### Identify Phenomenon
- "Tail of band states" into the band gap (Lifshitz tail)
- Well-pronounced in heavily doped and amorphous semiconductors
- Referred to as the "Urbach tail"

### Absorption Coefficient Formula (Eq. 12.1)
```
α(E) = α₀ × exp((E - E₀) / E₀)
```

### Where
- α₀ and E₀ are empirical parameters
- These parameters depend on:
  - Semiconductor type
  - Defect structure
  - Preparation method
  - Doping level
  - Treatment history

### Interpretation
- Shows exponential decline of absorption into band gap
- Characteristic of disordered semiconductor systems

## Key Variables
| Variable | Description |
|----------|-------------|
| ω | Angular frequency |
| c | Speed of light |
| x | Propagation distance |
| n_r | Refractive index (real part) |
| κ | Extinction coefficient |
| α₀ | Optical absorption coefficient |
| σ | Electrical conductivity at optical frequency |
| ε₀ | Permittivity of free space |
| δ | Phase shift angle |
| α(E) | Absorption coefficient as function of energy |
| α₀ (Urbach) | Empirical absorption constant (Urbach formula) |
| E₀ (Urbach) | Empirical energy parameter |
| E | Photon energy |
# Detailed Optical Relations

## Complex Refractive Index Definition

The complex refractive index is defined as:
```
ñ = nᵣ + iκ
```

This describes a plane wave propagating through an absorbing medium:
```
E(z,t) = E₀ × exp[i(ωt - (nᵣ + iκ)k₀z)]
       = E₀ × exp(-κk₀z) × exp[i(ωt - nᵣk₀z)]
```

The amplitude decays as exp(-κk₀z), while the phase propagates with velocity c/nᵣ.

## Relation to Absorption Coefficient

The intensity absorption coefficient α is related to κ:
```
α = 2κk₀ = 4πκ/λ
```

Intensity decays as I(z) = I₀ × exp(-αz)

## Dielectric Function Connection

Starting from Maxwell's equations:
```
ε = ε₁ + iε₂ = (nᵣ + iκ)²
```

Expanding:
```
(nᵣ + iκ)² = nᵣ² + 2inᵣκ - κ²
```

Equating real and imaginary parts:
```
ε₁ = nᵣ² - κ²
ε₂ = 2nᵣκ
```

## Example Calculation

**Given:** nᵣ = 3.5, κ = 0.1

**Calculate ε₁ and ε₂:**
```
ε₁ = (3.5)² - (0.1)² = 12.25 - 0.01 = 12.24
ε₂ = 2 × 3.5 × 0.1 = 0.7
```

**Verify inverse:**
```
|ε| = √(12.24² + 0.7²) = √(149.82 + 0.49) = √150.31 = 12.26
nᵣ = √[(12.26 + 12.24)/2] = √[24.5/2] = √12.25 = 3.5 ✓
κ = √[(12.26 - 12.24)/2] = √[0.02/2] = √0.01 = 0.1 ✓
```

## Kramers-Kronig Relations

The real and imaginary parts of the dielectric function are related by Kramers-Kronig relations:
```
ε₁(ω) = 1 + (2/π)P ∫₀^∞ [ω'ε₂(ω')/(ω'² - ω²)] dω'
ε₂(ω) = -(2ω/π)P ∫₀^∞ [ε₁(ω')/(ω'² - ω²)] dω'
```

This means that dispersion (ε₁) and absorption (ε₂) are fundamentally linked—you cannot change one without affecting the other.
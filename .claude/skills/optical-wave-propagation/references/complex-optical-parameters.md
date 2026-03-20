# Complex Optical Parameters

## Refractive Index Relationships

**Complex refractive index:**
```
n = n_r + ik
```

**Squared relationship to dielectric constant:**
```
n² = ε = ε₁ + iε₂
```

Expanding:
```
n_r² - k² = ε₁
2n_r k = ε₂
```

**Solving for n_r and k:**
```
n_r = √[(|ε| + ε₁)/2]
k = √[(|ε| - ε₁)/2]
```

Where |ε| = √(ε₁² + ε₂²)

## Conductivity Relationship

**Complex dielectric constant from conductivity:**
```
ε = ε_∞ + iσ/(ωε₀)
```

Where:
- ε_∞ = high-frequency dielectric constant
- σ = frequency-dependent conductivity
- ω = angular frequency
- ε₀ = permittivity of free space

Therefore:
```
ε₁ = ε_∞
ε₂ = σ/(ωε₀)
```

## Physical Interpretation

- **ε₁ (or n_r)**: Describes how much the material slows down light (dispersion)
- **ε₂ (or k)**: Describes how much the material absorbs light
- **σ**: Finite conductivity leads to absorption (k > 0)

## Absorption Coefficient

The intensity attenuation follows Beer-Lambert law:
```
I(x) = I₀ exp(-αx)
```

Where the absorption coefficient α is:
```
α = 2ωk/c = 4πk/λ
```

λ is the wavelength in vacuum.
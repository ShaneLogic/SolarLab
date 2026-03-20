# Material Parameters and Formulas

## Effective Mass Anisotropy Ratios

| Material | m_n∥/m_n⊥ |
|----------|----------|
| Ge (Germanium) | 19 |
| Si (Silicon) | 5.2 |

## Mobility Anisotropy Formula

### Samoilovich et al. (1961) Expression

The mobility in anisotropic semiconductors accounts for:

```
μ = f(m_n∥/m_n⊥, n, T)
```

Where:
- m_n∥ = effective mass parallel to valley axis
- m_n⊥ = effective mass perpendicular to valley axis
- n = carrier density
- T = temperature

### Anisotropy Factor K_a

```
K_a = μ∥/μ⊥
```

For screened Yukawa potential:
- K_a decreases with increasing carrier density n
- Relationship derived from Dakhovskii and Mikhai (1964)

## Screening Length

```
λ_D ∝ 1/√n
```

Constraint: λ_D must be substantially smaller than mean free path to prevent successive collisions in the same defect-potential region.

## Physical Mechanisms

### Low-Angle Scattering Dominance
- Ionized impurity scattering primarily produces low-angle scattering events
- These events interfere with randomizing electron velocities (Herring and Vogt, 1956)
- Anisotropy of effective mass and density of states strongly influences mobility (Boiko, 1959)

### Carrier Density Effects
- Higher carrier density n → smaller screening length λ_D
- Smaller cross-section of scattering centers
- Ion scattering becomes more randomizing with increasing n

## References

1. Herring, C. and Vogt, E. (1956) - Transport and deformation-potential theory for many-valley semiconductors with anisotropic scattering
2. Boiko, I.I. (1959) - Influence of anisotropy on mobility
3. Samoilovich, A.G., et al. (1961) - Mobility expression with anisotropy factor
4. Dakhovskii, I.V. and Mikhai, I.M. (1964) - Calculated anisotropy factor curves

## Experimental Data Reference

Fig. 17.8: Mobility anisotropy factor K_a measured at 77K in intrinsic Ge, showing calculated curve agreement with Dakhovskii and Mikhai (1964) model.

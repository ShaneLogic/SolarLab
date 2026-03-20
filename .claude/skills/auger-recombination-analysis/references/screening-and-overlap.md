# Screening and Overlap Integrals

## Electron Screening Factor

The screening factor λ determines the range of Coulomb interaction:

```
λ = √(n × e² / (ε₀ × εr × kT))
```

For degenerate semiconductors, use Thomas-Fermi screening:

```
λ_TF = √(3 × n × e² / (2 × ε₀ × εr × EF))
```

Where EF is the Fermi energy.

## Overlap Integral Calculation

### I₁ - Bloch Function Overlap

Represents overlap between conduction band states:

```
I₁ = ∫ Φc*(k) Φc(k') dk
```

For parabolic bands: I₁ ≈ 1

### I₂ - Momentum Conservation Factor

Accounts for momentum transfer requirements:

```
I₂ = ∫ Φv*(k) Φv(k'') dk
```

Typically I₂ ≈ 0.1 due to:
- Indirect transitions in some materials
- Phonon assistance requirements
- Band structure complexity

## Approximation Guidelines

| Material Type | I₁ | I₂ | I₁²I₂² |
|---------------|----|----|--------|
| Direct gap | 1.0 | 0.1-0.2 | 0.01-0.04 |
| Indirect gap | 0.8-1.0 | 0.05-0.1 | 0.0016-0.01 |
| Narrow gap | 1.0 | 0.1 | 0.01 |

## Impact on Lifetime

Uncertainty in overlap integrals leads to:
- Factor of 2-3 uncertainty in calculated lifetime
- Need for experimental calibration
- Temperature-dependent corrections
# Complex Conductivity Reference

## Mathematical Derivation

Starting from Maxwell's equations:

```
J_tot = σE + ∂D/∂t
```

In frequency domain (time harmonic assumption):

```
∂/∂t → iω
```

Therefore:
```
∂D/∂t = ∂(εε₀E)/∂t = iωεε₀E
```

Total current becomes:
```
J_tot = σE + iωεε₀E = (σ + iωεε₀)E = σ*E
```

Where σ* is the complex conductivity.

## Physical Interpretation

- **Real part (σ)**: Conduction current from free charges
- **Imaginary part (σ_d = ωεε₀)**: Displacement current from bound electrons oscillating out of phase

The displacement current conductivity represents the reactive component of the material's response.

## Relationship to Dielectric Constant

```
σ = ωε₀ε₂
σ_d = ωε₀ε₁
```

Where ε = ε₁ + iε₂ is the complex dielectric constant.

## Equation References
- Eq 20.26, 20.27
# Anisotropic Dielectric Tensors

## General Tensor Form

For anisotropic media, the dielectric relationship becomes:

```
D_i = Σⱼ ε_{ij} E_j
```

In matrix notation:
```
[D_x]   [ε₁₁ ε₁₂ ε₁₃] [E_x]
[D_y] = [ε₂₁ ε₂₂ ε₂₃] [E_y]
[D_z]   [ε₃₁ ε₃₂ ε₃₃] [E_z]
```

## Symmetry Reductions

### Cubic Crystals
```
[ε 0 0]
[0 ε 0]
[0 0 ε]
```
Only 1 independent coefficient.

### Tetragonal Crystals
```
[ε₁₁ 0   0  ]
[0   ε₁₁ 0  ]
[0   0   ε₃₃]
```
2 independent coefficients.

### Orthorhombic Crystals
```
[ε₁₁ 0   0  ]
[0   ε₂₂ 0  ]
[0   0   ε₃₃]
```
3 independent coefficients.

### Triclinic Crystals
Full 3×3 tensor with 6 independent coefficients (since ε_{ij} = ε_{ji}).

## Applying Maxwell's Equations

Poisson's equation in anisotropic media:
```
∇·E = ρ/(εε₀)
```

Becomes component-specific:
```
∂E_i/∂x_i = ρ/(ε_{ij}ε₀)
```

## Equation References
- Eq 20.32, 20.33
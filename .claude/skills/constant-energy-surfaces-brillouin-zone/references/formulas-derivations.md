# Effective Mass Derivation

## Effective Mass Tensor

The effective mass is derived from the band curvature:

```
m*_ij = ħ² [∂²E/∂k_i∂k_j]⁻¹
```

Where:
- ħ = reduced Planck constant
- i, j = directional indices (x, y, z)

## Isotropic Case (Spherical Surfaces)

For spherical surfaces:
```
E(k) = ħ²k² / 2m*
```

Effective mass is constant in all directions (scalar).

## Anisotropic Case (Non-Spherical Surfaces)

For ellipsoidal or non-spherical surfaces:
```
E(k) = ħ²(k_x²/m*_x + k_y²/m*_y + k_z²/m*_z) / 2
```

Effective mass depends on direction (tensor).

## Dispersion Relation Near Band Edges

Near conduction band minimum:
```
E_c(k) = E_c + ħ²k² / 2m*_e
```

Near valence band maximum:
```
E_v(k) = E_v - ħ²k² / 2m*_h
```

Where m*_e and m*_h are electron and hole effective masses respectively.
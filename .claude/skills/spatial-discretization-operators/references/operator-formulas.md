# Detailed Operator Formulas

## Difference Operator Derivation

The difference operator D_{i+1/2} approximates the first derivative ∂w/∂x at the midpoint x_{i+1/2} using a central difference:

```
∂w/∂x|_{x_{i+1/2}} ≈ (w_{i+1} - w_i) / (x_{i+1} - x_i)
              = (w_{i+1} - w_i) / Δx_{i+1/2}
```

For non-uniform grids, Δx_{i+1/2} varies with i. For uniform grids, Δx_{i+1/2} = Δx is constant.

## Midpoint Operator

The midpoint operator I_{i+1/2} provides a second-order accurate approximation:

```
w(x_{i+1/2}) ≈ (w_i + w_{i+1}) / 2
```

This is equivalent to the arithmetic mean and maintains conservation properties.

## Linear Operator L_i

For cell-centered finite volume schemes, the L_i operator interpolates from cell centers to grid points:

```
L_i(w) = (Δx_{i+1/2} * w_{i+1} + Δx_{i-1/2} * w_{i-1}) / (Δx_{i+1/2} + Δx_{i-1/2})
```

For uniform grids (Δx_{i+1/2} = Δx_{i-1/2} = Δx):

```
L_i(w) = (w_{i+1} + w_{i-1}) / 2
```

## Complete Equation References

### Electric Field (Equation 28)
```
E_{i+1/2} = -D_{i+1/2}(φ)
```

### Anion Flux (Equation 29)
```
FP_{i+1/2} = -D_{i+1/2}(P) - I_{i+1/2}(P) * E_{i+1/2}
```

### Electron Current (Equation 30)
```
jn_{i+1/2} = -D_{i+1/2}(n) + I_{i+1/2}(n) * E_{i+1/2}
```

### Hole Current (Equation 31)
```
jp_{i+1/2} = -D_{i+1/2}(p) - I_{i+1/2}(p) * E_{i+1/2}
```

### Anion Vacancy ODEs (Equations 32-34)

Evolution equations for P density with appropriate flux boundary conditions at i=0 and i=N-1.

### Potential Algebraic Equations (Equations 35-37)

Discrete Poisson equation:
```
-ε * D_{i+1/2}(E_{i+1/2}) - ε * D_{i-1/2}(E_{i-1/2}) = q(p - n - P + C)
```

With Dirichlet or Neumann boundary conditions for φ.

### Electron/Hole ODEs (Equations 38-43)

Continuity equations:
```
dn_i/dt = -(jn_{i+1/2} - jn_{i-1/2})/Δx_i + (G_{i+1/2} - R_{i+1/2})
dp_i/dt = -(jp_{i+1/2} - jp_{i-1/2})/Δx_i + (G_{i+1/2} - R_{i+1/2})
```

## Edge Cases

1. **Boundary Points (i=0 or i=N-1)**: Requires one-sided differences or ghost points
2. **Non-uniform Grids**: Use actual Δx_{i+1/2} values for each interface
3. **Singular Points**: May require special treatment at material interfaces
4. **Conservation Check**: Verify flux continuity across cell boundaries
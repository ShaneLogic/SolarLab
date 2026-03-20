# Drift-Diffusion Equation Derivations

## Dimensional Current Density Components

### Hole Current Density
The hole current density j_p consists of two components:

1. **Diffusion Component**: Proportional to concentration gradient
   - j_p,diff = -q * D_p * (∂p/∂x)
   - Flows from high to low concentration

2. **Drift Component**: Proportional to electric field
   - j_p,drift = V_T * p * (∂φ/∂x)
   - Flows in direction of electric field for positive carriers

### Electron Current Density
The electron current density j_n consists of:

1. **Diffusion Component**:
   - j_n,diff = q * D_n * (∂n/∂x)
   - Flows from high to low concentration

2. **Drift Component**:
   - j_n,drift = V_T * n * (∂φ/∂x)
   - Flows opposite to electric field for negative carriers

## Non-Dimensionalization Process

The non-dimensional formulation is obtained by scaling:
- Length by perovskite layer width L
- Time by characteristic transport time τ
- Concentrations by typical values

### Key Scaling Parameters

**λ (lambda)**: Ratio of Debye length to perovskite layer width
- λ = L_D / L
- Controls the strength of electrostatic coupling

**δ (delta)**: Ratio of typical carrier concentration to typical vacancy concentration
- δ = n_0 / P_0
- Determines relative importance of electronic vs ionic charges

## Physical Interpretation of Terms

### Conservation Equations
Each conservation equation follows the form:
```
∂(concentration)/∂t + ∂(flux)/∂x = generation - recombination
```

### Poisson's Equation in Non-Dimensional Form
```
∂E/∂x = (λ^2 / 2) * (P^(-1) + δ(p - n))
```

The right-hand side represents total charge density:
- **P^(-1)**: Mobile anion vacancies (positive charge)
- **-1**: Stationary cation vacancies (negative charge, fixed background)
- **δ*p**: Mobile holes (positive charge)
- **-δ*n**: Mobile electrons (negative charge)

## Boundary Conditions

Typical boundary conditions for perovskite solar cells:

### At interfaces (x = 0, x = L):
- Carrier concentrations determined by equilibrium with transport layers
- Electric field related to applied voltage
- Ion flux may be zero (blocking contacts) or specified

### Initial conditions:
- Carrier densities at thermal equilibrium
- Ion vacancy distribution (often uniform)

## Numerical Solution Approaches

1. **Finite Difference Method**: Discretize spatial derivatives
2. **Finite Element Method**: Handle complex geometries
3. **Spectral Methods**: High accuracy for smooth solutions
4. **Time-stepping schemes**: Implicit for stiff systems, explicit for simple cases

## Common Challenges

- Stiffness due to multiple time scales (electronic vs ionic)
- Coupling between Poisson's equation and carrier equations
- Boundary condition specification at heterojunctions
- Handling large parameter ranges in realistic devices
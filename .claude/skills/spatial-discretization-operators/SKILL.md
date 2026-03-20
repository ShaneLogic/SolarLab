---
name: spatial-discretization-operators
description: Implement spatial discretization operators (difference, midpoint, and linear interpolation) for finite difference schemes. Use this skill when converting continuous PDEs to discrete form, approximating derivatives at grid points or midpoints, or setting up discrete flux calculations in numerical simulations of electrochemical or semiconductor systems.
---

# Spatial Discretization Operators

This skill implements discrete operators for spatial discretization in finite difference schemes, enabling the conversion of continuous differential equations into discrete algebraic and ordinary differential equations.

## When to Use

- Converting continuous PDEs to discrete form on a defined grid
- Approximating derivatives at grid interfaces (i+1/2 indices)
- Calculating discrete fluxes for transport equations
- Setting up finite difference schemes for Poisson and continuity equations
- Writing discrete systems in concise operator notation

## Prerequisites

- Grid is defined with spacing Δx between nodes
- Finite element/difference scheme is specified
- Column vectors of unknowns are identified

## Implementation Steps

### 1. Define Discrete Operators

Apply the following operators to a column vector w with entries w_i:

- **Difference Operator D_{i+1/2}**: Approximates ∂w/∂x at interface x_{i+1/2}
  
  `D_{i+1/2}(w) = (w_{i+1} - w_i) / Δx_{i+1/2}`

- **Midpoint Operator I_{i+1/2}**: Evaluates variable at interface x_{i+1/2}
  
  `I_{i+1/2}(w) = (w_{i+1} + w_i) / 2`

- **Linear Operator L_i**: Interpolates at grid point i using adjacent cells
  
  `L_i(w) = (Δx_{i+1/2} * w_{i+1} + Δx_{i-1/2} * w_{i-1}) / (Δx_{i+1/2} + Δx_{i-1/2})`

These operators are valid for i = 0 to N-1.

### 2. Compute Discrete Fields and Fluxes

Use the operators to calculate discrete fields:

- Electric Field: `E_{i+1/2} ≈ -D_{i+1/2}(φ)`
- Anion Flux: `FP_{i+1/2} ≈ -D_{i+1/2}(P) - I_{i+1/2}(P) * E_{i+1/2}`
- Electron Current: `jn_{i+1/2} ≈ -D_{i+1/2}(n) + I_{i+1/2}(n) * E_{i+1/2}`
- Hole Current: `jp_{i+1/2} ≈ -D_{i+1/2}(p) - I_{i+1/2}(p) * E_{i+1/2}`

### 3. Construct System Equations

Combine operators and fluxes to form the discrete system:

- **Anion Vacancy ODEs**: Evolution of P density with flux boundary conditions
- **Potential Algebraic Equations**: Discretized Poisson equation with potential boundary conditions  
- **Electron/Hole ODEs**: Continuity equations with source terms G and R evaluated at midpoints (e.g., G_{i+1/2})

## Verification

- Confirm all operators use correct indexing (i or i+1/2 as appropriate)
- Verify boundary conditions are properly incorporated
- Ensure source terms (G, R) are evaluated at midpoints when multiplied by fluxes
- Check consistency with the underlying continuous equations

## Output

The skill produces:
- Specific algebraic equations for electrostatic potential φ
- Ordinary differential equations for carrier densities (P, n, p)
- Discrete flux expressions at grid interfaces

For detailed formula derivations, specific equation references, and edge case handling, refer to the references section.
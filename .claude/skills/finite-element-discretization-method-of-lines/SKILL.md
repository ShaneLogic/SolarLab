---
name: finite-element-discretization-method-of-lines
description: Convert continuous PDEs to discrete Differential-Algebraic Equations (DAEs) using finite element discretization with the Method of Lines. Use when solving semiconductor device equations that require resolving narrow Debye layers with non-uniform grids, or when you have non-dimensional model equations and need to create a system suitable for DAE solvers.
---

# Finite Element Discretization with Method of Lines

This skill converts continuous partial differential equations (PDEs) into a discrete system of DAEs suitable for numerical time integration.

## When to Use
- Converting semiconductor PDEs to a form solvable by DAE integrators
- Working with systems requiring non-uniform grid spacing to resolve boundary layers
- Your model equations are already non-dimensionalized
- You need second-order spatial accuracy without the Scharfetter-Gummel scheme

## Prerequisites
- Non-dimensional model equations available
- Computational grid defined with appropriate resolution for Debye layers

## Discretization Procedure

### 1. Method Selection
- Apply the **Method of Lines** as the central technique
- Use **Finite Element Scheme** with second-order local accuracy for spatial discretization
- Temporal integration will be handled by a specialized DAE solver (e.g., ode15s)
- Use a non-uniform grid with N+1 points to resolve narrow Debye layers

### 2. Basis Functions (Piecewise Linear)

Approximate the solution using hat/tent functions:

```
w(x,t) = Σ w_i(t) * φ_i(x)
```

Where each basis function φ_i(x) is defined as:
- (x - x_{i-1}) / (x_i - x_{i-1}) for x ∈ (x_{i-1}, x_i)
- (x_{i+1} - x) / (x_{i+1} - x_i) for x ∈ (x_i, x_{i+1})
- 0 otherwise

### 3. Galerkin Formulation

1. Multiply each PDE by test functions φ_j(x)
2. Integrate over the domain x ∈ (0, 1)
3. Apply integration by parts where appropriate to handle derivative terms

### 4. Treatment of Nonlinear Source Terms

Nonlinear source terms (generation G, recombination R) require special handling:

- Replace dependent variables in source term integrands with **piecewise constant functions** over subintervals
- Use the value equal to the full series at the **midpoint of each interval**
- This approximation preserves second-order accuracy

**Note:** This is a special case of the Skeel & Berzins method, applied only to source terms, not the entire equation.

## Output

The result is a system of Differential-Algebraic Equations (DAEs) ready for time integration with a DAE solver.
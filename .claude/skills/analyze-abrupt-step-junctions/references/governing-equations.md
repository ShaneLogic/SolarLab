# Governing Equations Reference

## Fundamental Equations

The electrical behavior of one-carrier abrupt step-junctions is governed by the following simultaneous non-linear differential equations:

### 1. Transport/Potential Relation (Eq. 25.3)
Relates electron potential to the electric field. 
*(Specific form available in source text)*

### 2. Poisson's Equation (Eq. 25.4)
$$ \frac{dF}{dx} = \frac{\rho}{\varepsilon} $$

Where:
- $F$: Electric field
- $\rho$: Space charge density
- $\varepsilon$: Dielectric constant

This equation determines the electric field distribution from the space charge.

### 3. Current Continuity (Eq. 25.5)
$$ \frac{dj}{dx} = 0 $$

Where:
- $j$: Current density

This ensures the current density is divergence-free (constant throughout the junction).

## Expanded Equation Set

Rewriting the fundamental equations yields a set of four simultaneous non-linear differential equations (Eqs. 25.6 - 25.9).

## Solution Constraints

- **Closed-form integration:** Impossible
- **Solution Method:** Numerical integration only
- **Boundary Conditions Required:** 6 total